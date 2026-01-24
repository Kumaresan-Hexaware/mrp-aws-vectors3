from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import boto3
import pandas as pd

from nl_analytics.config.settings import Settings
from nl_analytics.exceptions.errors import AgentExecutionError
from nl_analytics.logging.logger import get_logger


log = get_logger("db.redshift")


def _field_to_py(v: Dict[str, Any]) -> Any:
    if not v:
        return None
    if v.get("isNull") is True:
        return None
    for k in ("stringValue", "longValue", "doubleValue", "booleanValue", "blobValue"):
        if k in v:
            return v[k]
    return None


@dataclass
class RedshiftExecutor:
    settings: Settings

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute SQL using the Redshift Data API and return a DataFrame.

        Required when DB_TYPE=redshift:
          - redshift_database
          - (redshift_cluster_id OR redshift_workgroup_name)
          - (redshift_secret_arn OR redshift_db_user)
        """

        s = self.settings
        if not s.redshift_database:
            raise AgentExecutionError("REDSHIFT_DATABASE is required when DB_TYPE=redshift")
        if not (s.redshift_cluster_id or s.redshift_workgroup_name):
            raise AgentExecutionError(
                "REDSHIFT_CLUSTER_ID (provisioned) or REDSHIFT_WORKGROUP_NAME (serverless) is required"
            )
        if not (s.redshift_secret_arn or s.redshift_db_user):
            raise AgentExecutionError("REDSHIFT_SECRET_ARN (preferred) or REDSHIFT_DB_USER is required")

        client = boto3.client("redshift-data", region_name=s.aws_region or None)

        exec_args: Dict[str, Any] = {
            "Sql": sql,
            "Database": s.redshift_database,
        }
        if s.redshift_cluster_id:
            exec_args["ClusterIdentifier"] = s.redshift_cluster_id
        else:
            exec_args["WorkgroupName"] = s.redshift_workgroup_name

        if s.redshift_secret_arn:
            exec_args["SecretArn"] = s.redshift_secret_arn
        else:
            exec_args["DbUser"] = s.redshift_db_user

        log.info(
            "Redshift execute_statement",
            extra={
                "database": s.redshift_database,
                "cluster_id": s.redshift_cluster_id,
                "workgroup": s.redshift_workgroup_name,
                "sql_head": sql[:300],
            },
        )

        statement_id = client.execute_statement(**exec_args)["Id"]

        # Poll
        status = "STARTED"
        err = ""
        for _ in range(240):
            d = client.describe_statement(Id=statement_id)
            status = d.get("Status", "")
            if status in {"FINISHED", "FAILED", "ABORTED"}:
                err = d.get("Error", "") or ""
                break
            time.sleep(0.5)

        if status != "FINISHED":
            raise AgentExecutionError(f"Redshift query {status}: {err}")

        # Fetch all pages
        cols: List[str] = []
        rows: List[List[Any]] = []

        next_token: Optional[str] = None
        first = True
        while True:
            page_args = {"Id": statement_id}
            if next_token:
                page_args["NextToken"] = next_token
            r = client.get_statement_result(**page_args)

            if first:
                cols = [c.get("name", "") for c in r.get("ColumnMetadata", [])]
                first = False

            for rec in r.get("Records", []):
                rows.append([_field_to_py(x) for x in rec])

            next_token = r.get("NextToken")
            if not next_token:
                break

        return pd.DataFrame(rows, columns=cols)
