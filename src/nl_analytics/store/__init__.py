"""Query persistence / audit logging backends.

This package provides a small, config-driven abstraction to persist
user questions, LLM plans, confidence scores, SQL and execution metadata.

Backends:
  - file: JSONL events + latest snapshot JSON
  - dynamodb: writes events/snapshots into a DynamoDB table
  - none: no-op
"""

from .factory import build_query_store
from .types import QueryStore
