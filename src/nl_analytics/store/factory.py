from __future__ import annotations

from nl_analytics.config.settings import Settings
from nl_analytics.logging.logger import get_logger

from .types import QueryStore, NoopQueryStore
from .file_store import FileQueryStore
from .dynamodb_store import DynamoDBQueryStore


log = get_logger("store.factory")


def build_query_store(settings: Settings) -> QueryStore:
    """Create a QueryStore from Settings.

    Backends:
      - none: NoopQueryStore
      - file: FileQueryStore
      - dynamodb: DynamoDBQueryStore
    """
    backend = (getattr(settings, "query_store_backend", "none") or "none").strip().lower()
    if backend in ("", "none", "noop", "off"):
        return NoopQueryStore()

    if backend == "file":
        out_dir = getattr(settings, "query_store_dir", "data/query_logs")
        return FileQueryStore(out_dir)

    if backend == "dynamodb":
        table = getattr(settings, "dynamodb_table_name", "")
        if not table:
            log.warning("DynamoDB store enabled but DYNAMODB_TABLE_NAME is empty; falling back to noop")
            return NoopQueryStore()

        region = getattr(settings, "dynamodb_region", None) or getattr(settings, "aws_region", "us-east-1")
        endpoint_url = getattr(settings, "dynamodb_endpoint_url", None) or None
        ttl_attr = getattr(settings, "dynamodb_ttl_attribute", None) or None
        ttl_seconds = getattr(settings, "dynamodb_ttl_seconds", None)
        pk_name = getattr(settings, "dynamodb_pk_name", "session_id")
        sk_name = getattr(settings, "dynamodb_sk_name", "sk")

        mode = (getattr(settings, "query_store_mode", None) or "detailed").strip().lower()

        return DynamoDBQueryStore(
            table_name=table,
            region_name=region,
            endpoint_url=endpoint_url,
            ttl_attribute=ttl_attr,
            ttl_seconds=ttl_seconds,
            pk_name=pk_name,
            sk_name=sk_name,
            mode=mode,
        )

    log.warning("Unknown QUERY_STORE_BACKEND; falling back to noop", extra={"backend": backend})
    return NoopQueryStore()
