from __future__ import annotations
from typing import Dict
import pandas as pd

from nl_analytics.logging.logger import get_logger

log = get_logger("preprocessing.cleaning")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def coerce_types(df: pd.DataFrame, column_types: Dict[str, str], date_format: str | None = None) -> pd.DataFrame:
    out = df.copy()
    for col, typ in column_types.items():
        if col not in out.columns:
            log.warning("Missing column", extra={"column": col, "expected_type": typ})
            continue
        try:
            if typ in ("date", "datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce", format=date_format)
            elif typ == "int":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            elif typ == "float":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif typ == "bool":
                out[col] = out[col].astype("boolean")
            else:
                out[col] = out[col].astype("string")
            log.info("Type coerced", extra={"column": col, "type": typ})
        except Exception as e:
            log.warning("Type coercion failed", extra={"column": col, "type": typ, "error": str(e)})
    return out
