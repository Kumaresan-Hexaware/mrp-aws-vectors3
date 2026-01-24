from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass(frozen=True)
class SqlDialect:
    """Very small dialect shim for identifier quoting.

    DuckDB and Redshift happily accept ANSI double-quotes for identifiers.
    Athena (Presto/Trino) commonly uses backticks in Glue/Hive-style schemas.

    We keep this intentionally tiny so the rest of the executor logic stays
    unchanged.
    """

    ident_quote: str  # either '"' or '`'

    def ident(self, name: str) -> str:
        q = self.ident_quote
        if q == '"':
            return '"' + name.replace('"', '""') + '"'
        # backtick
        return '`' + name.replace('`', '``') + '`'


def dialect_for(db_type: str) -> SqlDialect:
    t = (db_type or "duckdb").strip().lower()
    if t == "athena":
        return SqlDialect(ident_quote="`")
    return SqlDialect(ident_quote='"')


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key -> (bucket, key)."""
    m = re.match(r"^s3://([^/]+)/(.+)$", (uri or "").strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return m.group(1), m.group(2)
