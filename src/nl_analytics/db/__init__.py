"""Database / query execution backends.

This project uses RAG+ReAct to *plan* a query, but the query still needs a
deterministic SQL engine to actually compute aggregates and return results.

Backends supported:
  - DuckDB   : local, in-memory (default)
  - Athena   : serverless SQL on S3 (Glue catalog)
  - Redshift : cluster/serverless via Redshift Data API
"""
