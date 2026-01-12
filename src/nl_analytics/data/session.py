from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.logging.logger import get_logger

log = get_logger("data.session")


@dataclass
class DataSession:
    """In-memory table session for the running app.

    Key points:
      - Uses registry-canonical table names (case-insensitive matching) to avoid mismatches like
        'PVR00400' vs 'pvr00400' from the planner.
      - Stores multiple ingested files for the same logical table by appending rows.
    """

    registry: SchemaRegistry
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    source_files: Dict[str, List[str]] = field(default_factory=dict)

    def canonical_table_name(self, name: str) -> str:
        n = (name or "").strip()
        if not n:
            return n
        try:
            for t in self.registry.list_tables():
                if t.lower() == n.lower():
                    return t
        except Exception:
            pass
        return n

    def register_table(self, table_name: str, df: pd.DataFrame, source_filename: str) -> None:
        t = self.canonical_table_name(table_name)

        if t in self.tables:
            self.tables[t] = pd.concat([self.tables[t], df], ignore_index=True)
            self.source_files.setdefault(t, []).append(source_filename)
            log.info("Appended table", extra={"table": t, "file": source_filename, "rows_total": len(self.tables[t])})
        else:
            self.tables[t] = df
            self.source_files[t] = [source_filename]
            log.info("Registered table", extra={"table": t, "file": source_filename, "rows": len(df)})

    def has_table(self, table_name: str) -> bool:
        t = self.canonical_table_name(table_name)
        return t in self.tables

    def get_table(self, table_name: str) -> pd.DataFrame:
        t = self.canonical_table_name(table_name)
        return self.tables[t]

    def available_tables(self) -> List[str]:
        return sorted(self.tables.keys())

    def clear(self) -> None:
        self.tables.clear()
        self.source_files.clear()
