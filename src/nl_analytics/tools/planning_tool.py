from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.exceptions.errors import SchemaValidationError

# JSON schema (guidance for the planner model). Validation logic below is the real authority.
# NOTE: `limit` is intentionally NOT required because some models omit it. We default it safely.
PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["mode", "tables", "metrics", "dimensions", "filters"],
    "properties": {
        "mode": {"type": "string", "enum": ["report", "dashboard"]},
        "tables": {"type": "array", "items": {"type": "string"}},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "expr"],
                "properties": {"name": {"type": "string"}, "expr": {"type": "string"}},
            },
        },
        "dimensions": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer"},
        "sort": {
            "type": "array",
            "items": {"type": "object", "properties": {"by": {"type": "string"}, "desc": {"type": "boolean"}}},
        },
        "chart": {"type": "object"},
        "charts": {"type": "array", "items": {"type": "object"}},
    },
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
    charts: List[Dict[str, Any]] | None = None


def _extract_metric_column(expr: str) -> str | None:
    expr = (expr or "").strip()
    if "(" not in expr or not expr.endswith(")"):
        return None
    inside = expr.split("(", 1)[1].rstrip(")").strip()
    if not inside or inside == "*":
        return None
    # Support COUNT(DISTINCT col) (used for pruning tables, not for SQL construction)
    if inside.lower().startswith("distinct "):
        parts = inside.split(None, 1)
        if len(parts) == 2:
            inside = parts[1].strip()
    return inside


def _canonicalize_col(col: str, all_cols: set[str], col_map: Dict[str, str]) -> str:
    def _norm(s: str) -> str:
        return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

    if col in all_cols:
        return col

    c = (col or "").strip()
    if not c:
        raise SchemaValidationError(f"Unknown column: {col}")

    # Handle fully-qualified names the LLM may emit, e.g. pvr01000.PortfolioID
    if "." in c:
        last = c.split(".")[-1].strip()
        if last and last in all_cols:
            return last
        c = last or c

    key = c.lower()
    if key in col_map:
        return col_map[key]
    nkey = _norm(c)
    if nkey in col_map:
        return col_map[nkey]

    raise SchemaValidationError(f"Unknown column: {col}")


def validate_plan(registry: SchemaRegistry, plan: Dict[str, Any]) -> QueryPlan:
    if not isinstance(plan, dict):
        raise SchemaValidationError("Plan must be a JSON object")

    # Defaults (apply before required checks so missing keys don't crash)
    plan.setdefault("filters", [])
    plan.setdefault("sort", [])
    plan.setdefault("limit", 5000)
    # Some planners emit a single 'chart' object; others can emit 'charts' (a list).
    # We support both and normalize to plan.charts.
    if "charts" in plan and plan["charts"] is not None and not isinstance(plan["charts"], list):
        plan["charts"] = [plan["charts"]]

    # Required keys (limit is optional)
    for k in ["mode", "tables", "metrics", "dimensions", "filters"]:
        if k not in plan:
            raise SchemaValidationError(f"Plan missing required key: {k}")

    mode = str(plan["mode"])
    if mode not in ("report", "dashboard"):
        raise SchemaValidationError("Invalid plan mode")

    tables = list(plan.get("tables") or [])
    if not tables:
        raise SchemaValidationError("Plan has no tables")

    # Validate tables exist in registry
    for t in tables:
        registry.get_table(t)

    # ---- prune extra tables if they are not needed by referenced columns ----
    referenced_cols: List[str] = []
    referenced_cols.extend([str(d) for d in (plan.get("dimensions") or [])])

    for m in list(plan.get("metrics") or []):
        col = _extract_metric_column(str(m.get("expr", "")))
        if col:
            referenced_cols.append(col)

    for f in list(plan.get("filters") or []):
        parts = str(f).replace("==", "=").split()
        if parts:
            referenced_cols.append(parts[0].strip())

    needed_tables: List[str] = []
    for col in referenced_cols:
        # Strip any table qualification to make pruning work when the model emits t.col
        col_tok = str(col).split(".")[-1].strip()
        col_l = col_tok.lower()
        for t in tables:
            tcols_l = {c.lower() for c in registry.columns_for_table(t)}
            if col_l in tcols_l:
                if t not in needed_tables:
                    needed_tables.append(t)
                break

    if needed_tables:
        tables = needed_tables

    # Ensure joinability
    registry.find_join_path(tables)

    # Collect columns across selected tables
    all_cols: set[str] = set()
    for t in tables:
        all_cols |= set(registry.columns_for_table(t))

    # Column map supports:
    #   1) exact column names
    #   2) normalized column keys (remove punctuation/spaces)
    #   3) business aliases declared in the schema registry
    col_map: Dict[str, str] = {c.lower(): c for c in all_cols}
    for c in list(all_cols):
        n = "".join(ch.lower() for ch in c if ch.isalnum())
        if n:
            col_map.setdefault(n, c)

    alias_map = registry.column_alias_map_for_tables(tables)
    for k, v in alias_map.items():
        col_map.setdefault(str(k).lower(), v)

    # Dimensions canonicalization
    dimensions_in = list(plan.get("dimensions") or [])
    dimensions: List[str] = []
    for d in dimensions_in:
        dimensions.append(_canonicalize_col(str(d), all_cols, col_map))

    # Metrics validation / canonicalization
    metrics_in = list(plan.get("metrics") or [])
    if not metrics_in:
        raise SchemaValidationError("No metrics defined")

    metrics: List[Dict[str, str]] = []
    for m in metrics_in:
        expr_raw = str(m.get("expr", "")).strip()
        if "(" not in expr_raw or not expr_raw.endswith(")"):
            raise SchemaValidationError(f"Metric expr must look like agg(col): {m}")

        agg_raw = expr_raw.split("(", 1)[0].strip()
        col_raw = expr_raw.split("(", 1)[1].rstrip(")").strip()

        agg = agg_raw.lower()
        if agg not in ALLOWED_AGGS:
            raise SchemaValidationError(f"Unsupported aggregation: {agg_raw}")

        distinct = False
        col_inner = col_raw
        # Support COUNT(DISTINCT col)
        if isinstance(col_inner, str) and col_inner.lower().startswith("distinct "):
            distinct = True
            col_inner = col_inner.split(None, 1)[1].strip() if len(col_inner.split(None, 1)) == 2 else ""

        if col_inner != "*":
            col = _canonicalize_col(col_inner, all_cols, col_map)
            if distinct:
                # Only COUNT(DISTINCT ..) is guaranteed to be supported by our executor.
                # We still let other aggs through, but DISTINCT will only be respected for COUNT.
                expr = f"{agg}(DISTINCT {col})"
            else:
                expr = f"{agg}({col})"
        else:
            expr = f"{agg}(*)"

        metrics.append({"name": str(m.get("name", "metric")), "expr": expr})

    # Filters canonicalize first token as column
    filters_in = list(plan.get("filters") or [])
    filters: List[str] = []
    for f in filters_in:
        parts = str(f).replace("==", "=").split()
        if not parts:
            continue
        col = _canonicalize_col(parts[0].strip(), all_cols, col_map)
        filters.append(" ".join([col] + parts[1:]))

    # Limit optional default + safety clamp
    try:
        limit = int(plan.get("limit", 5000))
    except Exception:
        limit = 5000
    limit = max(1, min(limit, 200000))

    charts = list(plan.get("charts") or [])
    if not charts and isinstance(plan.get("chart"), dict) and plan.get("chart"):
        charts = [plan.get("chart")]

    return QueryPlan(
        mode=mode,
        tables=tables,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        limit=limit,
        sort=plan.get("sort"),
        chart=plan.get("chart"),
        charts=charts or None,
    )
