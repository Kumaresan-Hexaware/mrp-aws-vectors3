from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass

from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.exceptions.errors import SchemaValidationError

PLAN_SCHEMA = {
  "type": "object",
  "required": ["mode", "tables", "metrics", "dimensions", "filters", "limit"],
  "properties": {
    "mode": {"type": "string", "enum": ["report", "dashboard"]},
    "tables": {"type": "array", "items": {"type": "string"}},
    "metrics": {
      "type": "array",
      "items": {"type": "object", "required": ["name", "expr"], "properties": {"name": {"type": "string"}, "expr": {"type": "string"}}}
    },
    "dimensions": {"type": "array", "items": {"type": "string"}},
    "filters": {"type": "array", "items": {"type": "string"}},
    "limit": {"type": "integer"},
    "sort": {"type": "array", "items": {"type": "object", "properties": {"by": {"type": "string"}, "desc": {"type": "boolean"}}}},
    "chart": {"type": "object"}
  }
}

ALLOWED_AGGS = {"sum", "avg", "min", "max", "count"}

@dataclass(frozen=True)
class QueryPlan:
    mode: str
    tables: List[str]
    metrics: List[Dict[str, str]]
    dimensions: List[str]
    filters: List[str]
    limit: int
    sort: List[Dict[str, Any]] | None = None
    chart: Dict[str, Any] | None = None

def validate_plan(registry: SchemaRegistry, plan: Dict[str, Any]) -> QueryPlan:
    for k in ["mode", "tables", "metrics", "dimensions", "filters", "limit"]:
        if k not in plan:
            raise SchemaValidationError(f"Plan missing required key: {k}")

    mode = plan["mode"]
    if mode not in ("report", "dashboard"):
        raise SchemaValidationError("Invalid plan mode")

    tables = list(plan["tables"] or [])
    if not tables:
        raise SchemaValidationError("Plan has no tables")
    for t in tables:
        registry.get_table(t)

    registry.find_join_path(tables)

    all_cols = set()
    for t in tables:
        all_cols |= set(registry.columns_for_table(t))

    dimensions = list(plan.get("dimensions") or [])
    for d in dimensions:
        if d not in all_cols:
            raise SchemaValidationError(f"Unknown dimension column: {d}")

    metrics = list(plan.get("metrics") or [])
    if not metrics:
        raise SchemaValidationError("No metrics defined")

    for m in metrics:
        expr = str(m.get("expr", "")).strip().lower()
        if "(" not in expr or not expr.endswith(")"):
            raise SchemaValidationError(f"Metric expr must look like agg(col): {m}")
        agg = expr.split("(", 1)[0].strip()
        col = expr.split("(", 1)[1].rstrip(")").strip()
        if agg not in ALLOWED_AGGS:
            raise SchemaValidationError(f"Unsupported aggregation: {agg}")
        if col != "*" and col not in all_cols:
            raise SchemaValidationError(f"Unknown metric column: {col}")

    filters = list(plan.get("filters") or [])
    for f in filters:
        parts = f.replace("==", "=").split()
        if parts:
            col = parts[0].strip()
            if col not in all_cols:
                raise SchemaValidationError(f"Unknown filter column: {col}")

    limit = int(plan.get("limit", 5000))
    limit = max(1, min(limit, 200000))

    return QueryPlan(
        mode=mode,
        tables=tables,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        limit=limit,
        sort=plan.get("sort"),
        chart=plan.get("chart"),
    )
