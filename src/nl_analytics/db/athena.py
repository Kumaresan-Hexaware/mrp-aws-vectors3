from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import time
import re

import boto3
import pandas as pd

from nl_analytics.config.settings import Settings
from nl_analytics.exceptions.errors import AgentExecutionError
from nl_analytics.logging.logger import get_logger
from nl_analytics.db.utils import parse_s3_uri


log = get_logger("db.athena")


# --- Athena type-safety helpers -------------------------------------------------
# Many upstream flat files land in Athena with columns typed as VARCHAR.
# When the planner emits numeric predicates like:  col > 100
# Athena errors with TYPE_MISMATCH (varchar vs integer/double).
#
# We defensively rewrite numeric predicates to use TRY_CAST(<col> AS DOUBLE)
# so queries succeed even if the underlying storage has VARCHAR numerics.


_NUMERIC_COMPARE_RE = re.compile(
    r"(?P<lhs>(?:[A-Za-z_][\w]*\.)?(?:\"[^\"]+\"|[A-Za-z_][\w]*))\s*"
    r"(?P<op>>=|<=|<>|!=|=|<|>)\s*"
    r"(?P<num>-?\d+(?:\.\d+)?)"
)


_BETWEEN_RE = re.compile(
    r"(?P<lhs>(?:[A-Za-z_][\w]*\.)?(?:\"[^\"]+\"|[A-Za-z_][\w]*))\s+BETWEEN\s+"
    r"(?P<lo>-?\d+(?:\.\d+)?)\s+AND\s+(?P<hi>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _needs_cast(expr: str) -> bool:
    up = expr.upper()
    return "CAST(" not in up and "TRY_CAST(" not in up


def rewrite_sql_for_athena(sql: str) -> str:
    """Rewrite SQL for Athena to avoid VARCHAR numeric predicate failures.

    Safe, targeted rewrites only:
      - <col> <op> <number>  -> TRY_CAST(<col> AS DOUBLE) <op> <number>
      - <col> BETWEEN n AND m -> TRY_CAST(<col> AS DOUBLE) BETWEEN n AND m

    Does NOT touch aggregation expressions (AVG/SUM) to avoid corrupting SQL.
    """

    def repl_cmp(m: re.Match) -> str:
        lhs = m.group("lhs")
        op = m.group("op")
        num = m.group("num")
        if not _needs_cast(lhs):
            return m.group(0)
        return f"TRY_CAST({lhs} AS DOUBLE) {op} {num}"

    def repl_between(m: re.Match) -> str:
        lhs = m.group("lhs")
        lo = m.group("lo")
        hi = m.group("hi")
        if not _needs_cast(lhs):
            return m.group(0)
        return f"TRY_CAST({lhs} AS DOUBLE) BETWEEN {lo} AND {hi}"

    out = _BETWEEN_RE.sub(repl_between, sql)
    out = _NUMERIC_COMPARE_RE.sub(repl_cmp, out)
    return out


# Backward-compatible alias (older callers may import the private name).
_rewrite_sql_for_athena = rewrite_sql_for_athena


@dataclass
class AthenaExecutor:
    settings: Settings

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute SQL in Athena and return a pandas DataFrame.

        We rely on Athena's configured output location (S3) and then read the
        resulting CSV back into pandas.

        Required settings when DB_TYPE=athena:
          - athena_database
          - athena_output_location (s3://...)

        Optional:
          - athena_workgroup
          - aws_region
        """

        s = self.settings
        if not s.athena_database:
            raise AgentExecutionError("ATHENA_DATABASE is required when DB_TYPE=athena")
        if not s.athena_output_location:
            raise AgentExecutionError("ATHENA_OUTPUT_LOCATION is required when DB_TYPE=athena")

        # Athena is strict about types; numeric comparisons against VARCHAR columns fail.
        sql_to_run = rewrite_sql_for_athena(sql)
        if sql_to_run != sql:
            log.debug(
                "Rewrote SQL for Athena numeric safety",
                extra={"orig_head": sql[:300], "rewritten_head": sql_to_run[:300]},
            )

        ath = boto3.client("athena", region_name=s.aws_region or None)

        start_args = {
            "QueryString": sql_to_run,
            "QueryExecutionContext": {
                "Database": s.athena_database,
                "Catalog": s.athena_catalog or "AwsDataCatalog",
            },
            "ResultConfiguration": {"OutputLocation": s.athena_output_location},
        }
        if s.athena_workgroup:
            start_args["WorkGroup"] = s.athena_workgroup

        # Log the full SQL (users requested printing Athena SQL for audit/debug).
        # Keep a short head in structured fields, and the full SQL in the message.
        log.info(
            "ATHENA QUERY :::\n%s",
            sql_to_run,
            extra={
                "database": s.athena_database,
                "workgroup": s.athena_workgroup,
                "output": s.athena_output_location,
                "sql_head": sql_to_run[:300],
            },
        )

        qid = ath.start_query_execution(**start_args)["QueryExecutionId"]

        # Poll
        state = "QUEUED"
        reason = ""
        for _ in range(240):
            resp = ath.get_query_execution(QueryExecutionId=qid)
            status = resp.get("QueryExecution", {}).get("Status", {})
            state = status.get("State", "")
            reason = status.get("StateChangeReason", "") or ""
            if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.5)

        if state != "SUCCEEDED":
            raise AgentExecutionError(f"Athena query {state}: {reason}")

        out_loc = (
            resp.get("QueryExecution", {})
            .get("ResultConfiguration", {})
            .get("OutputLocation", "")
        )
        if not out_loc:
            # Fallback to configured output; Athena typically writes to the configured location anyway.
            out_loc = s.athena_output_location.rstrip("/") + f"/{qid}.csv"

        bucket, key = parse_s3_uri(out_loc)
        s3 = boto3.client("s3", region_name=s.aws_region or None)
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()

        # Athena writes CSV with header row.
        return pd.read_csv(BytesIO(body))
