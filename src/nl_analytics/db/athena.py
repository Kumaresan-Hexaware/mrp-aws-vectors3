from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import time

import boto3
import pandas as pd

from nl_analytics.config.settings import Settings
from nl_analytics.exceptions.errors import AgentExecutionError
from nl_analytics.logging.logger import get_logger
from nl_analytics.db.utils import parse_s3_uri


log = get_logger("db.athena")


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

        ath = boto3.client("athena", region_name=s.aws_region or None)

        start_args = {
            "QueryString": sql,
            "QueryExecutionContext": {
                "Database": s.athena_database,
                "Catalog": s.athena_catalog or "AwsDataCatalog",
            },
            "ResultConfiguration": {"OutputLocation": s.athena_output_location},
        }
        if s.athena_workgroup:
            start_args["WorkGroup"] = s.athena_workgroup

        log.info(
            "Athena start_query_execution",
            extra={
                "database": s.athena_database,
                "workgroup": s.athena_workgroup,
                "output": s.athena_output_location,
                "sql_head": sql[:300],
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
