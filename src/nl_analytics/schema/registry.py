from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path
import yaml

from nl_analytics.exceptions.errors import SchemaValidationError
from nl_analytics.logging.logger import get_logger

log = get_logger("schema.registry")

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class TableSpec:
    name: str
    file_patterns: List[str]
    description: str
    primary_key: List[str]
    columns: Dict[str, ColumnSpec]

@dataclass(frozen=True)
class JoinRule:
    left_table: str
    right_table: str
    left_keys: List[str]
    right_keys: List[str]
    join_type: str = "inner"

class SchemaRegistry:
    def __init__(self, tables: Dict[str, TableSpec], joins: List[JoinRule], version: int = 1):
        self.version = version
        self.tables = tables
        self.joins = joins
        self._adj: Dict[str, List[JoinRule]] = {}
        for j in joins:
            self._adj.setdefault(j.left_table, []).append(j)
            self._adj.setdefault(j.right_table, []).append(
                JoinRule(
                    left_table=j.right_table,
                    right_table=j.left_table,
                    left_keys=j.right_keys,
                    right_keys=j.left_keys,
                    join_type=j.join_type,
                )
            )

    @staticmethod
    def load(path: str = "schemas/schema_registry.yaml") -> "SchemaRegistry":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Schema registry not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        version = int(raw.get("version", 1))

        tables: Dict[str, TableSpec] = {}
        for tname, tval in (raw.get("tables") or {}).items():
            cols = {}
            for cname, cval in (tval.get("columns") or {}).items():
                aliases = cval.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                aliases = [str(a) for a in (aliases or []) if str(a).strip()]
                cols[cname] = ColumnSpec(
                    name=cname,
                    type=str(cval.get("type", "string")),
                    description=str(cval.get("description", "")),
                    aliases=aliases,
                )
            tables[tname] = TableSpec(
                name=tname,
                file_patterns=list(tval.get("file_patterns", [])),
                description=str(tval.get("description", "")),
                primary_key=list(tval.get("primary_key", [])),
                columns=cols,
            )

        joins: List[JoinRule] = []
        for j in raw.get("joins", []) or []:
            joins.append(
                JoinRule(
                    left_table=j["left_table"],
                    right_table=j["right_table"],
                    left_keys=list(j["left_keys"]),
                    right_keys=list(j["right_keys"]),
                    join_type=j.get("join_type", "inner"),
                )
            )

        reg = SchemaRegistry(tables=tables, joins=joins, version=version)
        reg.validate()
        return reg

    def validate(self) -> None:
        if not self.tables:
            raise SchemaValidationError("Schema registry has no tables.")
        for j in self.joins:
            if j.left_table not in self.tables or j.right_table not in self.tables:
                raise SchemaValidationError(f"Join references unknown table: {j}")
            lt = self.tables[j.left_table]
            rt = self.tables[j.right_table]
            for k in j.left_keys:
                if k not in lt.columns:
                    raise SchemaValidationError(f"Join key missing in {lt.name}: {k}")
            for k in j.right_keys:
                if k not in rt.columns:
                    raise SchemaValidationError(f"Join key missing in {rt.name}: {k}")
        log.info("Schema registry validated")

    def list_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def columns_for_table(self, table: str) -> List[str]:
        self._ensure_table(table)
        return sorted(self.tables[table].columns.keys())

    def column_alias_map_for_tables(self, tables: List[str]) -> Dict[str, str]:
        """Return a case-insensitive map of alias -> canonical column name across selected tables."""
        alias_map: Dict[str, str] = {}

        def _norm(s: str) -> str:
            return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

        for t in tables:
            self._ensure_table(t)
            for cname, cspec in self.tables[t].columns.items():
                # Always include the canonical column itself as a lookup key.
                for k in {cname.lower(), _norm(cname)}:
                    if k:
                        alias_map.setdefault(k, cname)

                for a in (cspec.aliases or []):
                    a = str(a).strip()
                    if not a:
                        continue
                    for k in {a.lower(), _norm(a)}:
                        if k:
                            alias_map.setdefault(k, cname)

        return alias_map

    def get_table(self, table: str) -> TableSpec:
        self._ensure_table(table)
        return self.tables[table]

    def find_join_path(self, tables: List[str]) -> List[JoinRule]:
        if not tables:
            return []
        for t in tables:
            self._ensure_table(t)
        if len(tables) == 1:
            return []

        connected = {tables[0]}
        path: List[JoinRule] = []
        remaining = set(tables[1:])

        while remaining:
            found = False
            for target in list(remaining):
                join_seq = self._bfs_path_any_source(connected, target)
                if join_seq:
                    for e in join_seq:
                        path.append(e)
                        connected.add(e.right_table)
                    remaining.remove(target)
                    found = True
                    break
            if not found:
                raise SchemaValidationError(
                    f"No registry join path can connect requested tables: {sorted(connected)} + {sorted(remaining)}"
                )
        return path

    def _bfs_path_any_source(self, sources: set, target: str) -> List[JoinRule]:
        from collections import deque
        q = deque()
        prev: Dict[str, Tuple[str, JoinRule]] = {}
        visited = set(sources)
        for s in sources:
            q.append(s)
        while q:
            u = q.popleft()
            if u == target:
                break
            for e in self._adj.get(u, []):
                v = e.right_table
                if v in visited:
                    continue
                visited.add(v)
                prev[v] = (u, e)
                q.append(v)

        if target not in visited:
            return []
        seq: List[JoinRule] = []
        cur = target
        while cur not in sources:
            pu, edge = prev[cur]
            seq.append(edge)
            cur = pu
        seq.reverse()
        return seq

    def _ensure_table(self, table: str) -> None:
        if table not in self.tables:
            raise SchemaValidationError(f"Unknown table: {table}")
