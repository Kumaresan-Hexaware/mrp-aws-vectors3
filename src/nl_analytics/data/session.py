from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.logging.logger import get_logger

log = get_logger("data.session")

@dataclass
class DataSession:
    registry: SchemaRegistry
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    source_files: Dict[str, List[str]] = field(default_factory=dict)

    def register_table(self, table_name: str, df: pd.DataFrame, source_filename: str) -> None:
        if table_name in self.tables:
            self.tables[table_name] = pd.concat([self.tables[table_name], df], ignore_index=True)
            self.source_files.setdefault(table_name, []).append(source_filename)
            log.info("Appended table", extra={"table": table_name, "file": source_filename, "rows_total": len(self.tables[table_name])})
        else:
            self.tables[table_name] = df
            self.source_files[table_name] = [source_filename]
            log.info("Registered table", extra={"table": table_name, "file": source_filename, "rows": len(df)})

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def available_tables(self) -> List[str]:
        return sorted(self.tables.keys())
