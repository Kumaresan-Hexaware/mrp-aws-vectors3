from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass(frozen=True)
class SqlDialect:
    """Tiny dialect shim.

    This project primarily executes on DuckDB / Athena / Redshift, but we
    occasionally need to *render* SQL for other engines (ex: Postgres) for
    testing purposes.

    Keep this minimal so executor logic remains stable.
    """

    ident_quote: str  # either '"' or '`'
    try_cast_double_template: str = "TRY_CAST({expr} AS DOUBLE)"

    def ident(self, name: str) -> str:
        q = self.ident_quote
        if q == '"':
            return '"' + name.replace('"', '""') + '"'
        # backtick
        return '`' + name.replace('`', '``') + '`'

    def try_cast_double(self, expr: str) -> str:
        """Return a "best-effort" cast-to-double expression for this dialect."""
        return self.try_cast_double_template.format(expr=expr)


def dialect_for(db_type: str) -> SqlDialect:
    t = (db_type or "duckdb").strip().lower()
    if t == "athena":
        return SqlDialect(ident_quote="`", try_cast_double_template="TRY_CAST({expr} AS DOUBLE)")
    if t in {"postgres", "postgresql", "pg"}:
        # Postgres has no TRY_CAST. For testing, emit a defensive CASE expression that
        # avoids hard failures on non-numeric strings.
        # NOTE: This is only for logging/testing output; execution is still driven by db_type.
        pg_try_cast = (
            "CASE "
            "WHEN {expr} IS NULL THEN NULL "
            "WHEN ({expr})::text ~ '^[+-]?[0-9]+(\\.[0-9]+)?$' THEN ({expr})::double precision "
            "ELSE NULL END"
        )
        return SqlDialect(ident_quote='"', try_cast_double_template=pg_try_cast)
    # duckdb / redshift / default
    return SqlDialect(ident_quote='"', try_cast_double_template="TRY_CAST({expr} AS DOUBLE)")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key -> (bucket, key)."""
    m = re.match(r"^s3://([^/]+)/(.+)$", (uri or "").strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return m.group(1), m.group(2)
