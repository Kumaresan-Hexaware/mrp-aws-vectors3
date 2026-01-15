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
    # Aggregated tables (logical table -> combined dataframe)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Per-source storage to avoid accidental duplication when the same file is ingested
    # multiple times (e.g., auto-load on startup + manual ingest, or repeated uploads).
    # This preserves the original behavior of "multiple files per logical table" while
    # making ingestion idempotent per filename.
    file_tables: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

    # Convenience list of sources per logical table (kept for UI display).
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

        # Store/overwrite by source filename (idempotent per filename).
        self.file_tables.setdefault(t, {})[source_filename] = df
        self.source_files[t] = list(self.file_tables[t].keys())

        # Rebuild the aggregated table.
        parts = list(self.file_tables[t].values())
        self.tables[t] = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

        log.info(
            "Registered table (per-file)",
            extra={
                "table": t,
                "file": source_filename,
                "files_for_table": len(self.source_files.get(t, [])),
                "rows_total": len(self.tables[t]),
            },
        )

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
        self.file_tables.clear()
        self.source_files.clear()
