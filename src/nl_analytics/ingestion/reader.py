from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import pandas as pd

from nl_analytics.logging.logger import get_logger
from nl_analytics.exceptions.errors import DataIngestionError

log = get_logger("ingestion.reader")

@dataclass(frozen=True)
class IngestionResult:
    df: pd.DataFrame
    encoding_used: str
    rows_read: int
    bad_lines_skipped: bool

def read_nzf(
    file_path: str,
    delimiter: str,
    fallback_encodings: List[str],
    skip_bad_lines: bool = True,
) -> IngestionResult:
    p = Path(file_path)
    if not p.exists():
        raise DataIngestionError(f"File not found: {file_path}")

    last_err: Optional[Exception] = None
    for enc in fallback_encodings:
        try:
            log.info("Reading file", extra={"source_file": p.name, "encoding": enc})
            df = pd.read_csv(
                p,
                sep=delimiter,
                encoding=enc,
                engine="python",
                on_bad_lines="skip" if skip_bad_lines else "error",
            )
            return IngestionResult(df=df, encoding_used=enc, rows_read=len(df), bad_lines_skipped=skip_bad_lines)
        except UnicodeDecodeError as e:
            last_err = e
            log.error("Encoding error", extra={"source_file": p.name, "encoding": enc, "error": str(e)})
        except Exception as e:
            last_err = e
            log.exception("Unexpected ingestion error", extra={"source_file": p.name, "encoding": enc})

    raise DataIngestionError(f"Failed to decode {p.name} with encodings: {fallback_encodings}") from last_err
