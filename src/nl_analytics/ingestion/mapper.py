from __future__ import annotations
from pathlib import Path
from typing import Optional
import fnmatch

from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.logging.logger import get_logger

log = get_logger("ingestion.mapper")

def map_file_to_table(registry: SchemaRegistry, filename: str) -> Optional[str]:
    for tname in registry.list_tables():
        spec = registry.get_table(tname)
        for pat in spec.file_patterns:
            if fnmatch.fnmatch(Path(filename).name, pat):
                log.info("Mapped file to table", extra={"file": filename, "table": tname, "pattern": pat})
                return tname
    return None
