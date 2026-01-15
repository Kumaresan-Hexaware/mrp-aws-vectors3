from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
import duckdb
import pandas as pd

from nl_analytics.data.session import DataSession
from nl_analytics.schema.registry import SchemaRegistry, JoinRule
from nl_analytics.tools.planning_tool import QueryPlan
from nl_analytics.exceptions.errors import AgentExecutionError, SchemaValidationError
from nl_analytics.logging.logger import get_logger

log = get_logger("tools.execution")

def _sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def build_join_sql(registry: SchemaRegistry, plan_tables: List[str]) -> Tuple[str, List[JoinRule]]:
    join_path = registry.find_join_path(plan_tables)
    root = plan_tables[0]
    sql = f"FROM {_sql_ident(root)} AS {root}"
    joined = {root}
    for e in join_path:
        if e.left_table not in joined:
            raise SchemaValidationError("Join path order invalid; cannot execute joins safely.")
        lt_alias = e.left_table
        rt_alias = e.right_table
        conds = []
        for lk, rk in zip(e.left_keys, e.right_keys):
            conds.append(f"{lt_alias}.{_sql_ident(lk)} = {rt_alias}.{_sql_ident(rk)}")
        cond = " AND ".join(conds) if conds else "1=1"
        sql += f" {e.join_type.upper()} JOIN {_sql_ident(rt_alias)} AS {rt_alias} ON {cond}"
        joined.add(rt_alias)
    return sql, join_path

def _strip_expr_alias(expr: str) -> str:
    # Remove any accidental "AS alias" the planner might include inside expr.
    return re.sub(r"(?i)\s+as\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", (expr or "").strip())


def _quote_cols_in_expr(expr: str, allowed_cols: List[str]) -> str:
    """Quote column identifiers within a SQL expression, avoiding quoted string literals.

    This lets us support planner-generated expressions like:
      SUM(CASE WHEN PaymentStructureTypeCode='PAY' THEN 1 ELSE 0 END)
      ROUND(100.0 * SUM(...) / NULLIF(COUNT(*),0), 2)

    while still preventing the executor from incorrectly treating the whole expression as a single identifier.
    """
    if not expr:
        return ""

    # Strip any table qualification (t.col). Safe: does not affect numeric literals like 100.0
    expr = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", expr)

    # Safety: prohibit multi-statement / comments
    low = expr.lower()
    if ";" in low or "--" in low or "/*" in low or "*/" in low:
        raise AgentExecutionError("Unsafe tokens in metric expression")

    cols_sorted = sorted(set(allowed_cols), key=len, reverse=True)

    out = []
    in_single = False
    in_double = False
    buf = ""
    for i, ch in enumerate(expr):
        if ch == "'" and not in_double:
            # handle escaped ''
            if in_single and i + 1 < len(expr) and expr[i + 1] == "'":
                buf += "''"
                continue
            # flush current buffer segment with replacements
            if buf:
                seg = buf
                for c in cols_sorted:
                    seg = re.sub(rf"\b{re.escape(c)}\b", _sql_ident(c), seg)
                out.append(seg)
                buf = ""
            out.append("'")
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            if buf:
                seg = buf
                for c in cols_sorted:
                    seg = re.sub(rf"\b{re.escape(c)}\b", _sql_ident(c), seg)
                out.append(seg)
                buf = ""
            out.append('"')
            in_double = not in_double
            continue
        buf += ch
    if buf:
        seg = buf
        for c in cols_sorted:
            seg = re.sub(rf"\b{re.escape(c)}\b", _sql_ident(c), seg)
        out.append(seg)
    return "".join(out)


def _metric_sql(metrics: List[Dict[str, str]], allowed_cols: List[str]) -> List[str]:
    out = []
    for m in metrics:
        name = m["name"]
        expr = _strip_expr_alias(m["expr"])
        # Normalize whitespace from LLM formatting.
        expr = re.sub(r"\s+", " ", expr).strip()

        # Repair the common case where an identifier is split by whitespace under DISTINCT,
        # e.g., COUNT(DISTINCT Instrument ID) -> COUNT(DISTINCT InstrumentID)
        if re.search(r"(?i)\bdistinct\b", expr):
            m_dist = re.match(
                r"(?is)^(sum|avg|min|max|count)\s*\(\s*distinct\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
                expr,
            )
            if m_dist:
                combined = f"{m_dist.group(2)}{m_dist.group(3)}"
                if combined in set(allowed_cols):
                    expr = f"{m_dist.group(1)}(DISTINCT {combined})"

        # Fast path: simple AGG(col) and AGG(*) patterns
        m_simple = re.match(
            r"(?is)^\s*(sum|avg|min|max|count)\s*\(\s*(distinct\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
            expr,
        )
        if re.match(r"(?is)^\s*count\s*\(\s*\*\s*\)\s*$", expr):
            out.append(f"COUNT(*) AS {_sql_ident(name)}")
            continue

        if not m_simple:
            # Complex expression (CASE WHEN, ROUND wrapper, arithmetic, etc.).
            expr_sql = _quote_cols_in_expr(expr, allowed_cols)
            out.append(f"{expr_sql} AS {_sql_ident(name)}")
            continue

        agg = m_simple.group(1)
        distinct_kw = m_simple.group(2) or ""
        col_inner = m_simple.group(3)
        agg_l = agg.lower()
        distinct = bool(distinct_kw)

        # SUM/AVG should be numeric-safe because many CSV/NZF fields load as strings.
        col_sql = _sql_ident(col_inner)
        if distinct:
            if agg_l in {"sum", "avg"}:
                col_sql = f"DISTINCT TRY_CAST({col_sql} AS DOUBLE)"
            else:
                col_sql = f"DISTINCT {col_sql}"
        else:
            if agg_l in {"sum", "avg"}:
                col_sql = f"TRY_CAST({col_sql} AS DOUBLE)"

        out.append(f"{agg.upper()}({col_sql}) AS {_sql_ident(name)}")
    return out

def _filters_sql(filters: List[str]) -> str:
    if not filters:
        return ""
    safe_parts = []
    for f in filters:
        f = f.strip()
        m = re.match(r'^([A-Za-z0-9_]+)\s*(=|!=|>=|<=|>|<)\s*(.+)$', f)
        if not m:
            continue
        col, op, val = m.group(1), m.group(2), m.group(3).strip()
        if val.startswith(("'", '"')) and val.endswith(("'", '"')):
            safe_val = val
        else:
            if re.match(r'^-?\d+(\.\d+)?$', val):
                safe_val = val
            else:
                safe_val = "'" + val.replace("'", "''") + "'"
        safe_parts.append(f"{_sql_ident(col)} {op} {safe_val}")
    if not safe_parts:
        return ""
    return "WHERE " + " AND ".join(safe_parts)

def execute_plan(session: DataSession, plan: QueryPlan) -> pd.DataFrame:
    registry = session.registry

    # Resolve plan tables to registry-canonical names (case-insensitive)
    tables = [session.canonical_table_name(t) for t in plan.tables]

    for t in tables:
        if not session.has_table(t):
            raise AgentExecutionError(f"Missing table data for '{t}'")

    # NOTE: `duckdb.connect()` returns a connection object; it is **not** callable.
    # Calling it like a function raises:
    #   TypeError: '_duckdb.DuckDBPyConnection' object is not callable
    con = duckdb.connect(database=":memory:")
    try:
        for t in tables:
            con.register(t, session.get_table(t))

        select_cols = []
        for d in plan.dimensions:
            select_cols.append(_sql_ident(d))
        # Build a set of allowed columns across the selected tables for safe quoting inside expressions.
        allowed_cols: List[str] = []
        for t in tables:
            try:
                allowed_cols.extend(list(registry.columns_for_table(t)))
            except Exception:
                # Fallback to the registered dataframe columns
                allowed_cols.extend(list(session.get_table(t).columns))

        select_cols.extend(_metric_sql(plan.metrics, allowed_cols))

        join_sql, _ = build_join_sql(registry, tables)

        group_by = ""
        if plan.dimensions:
            gb = ", ".join(_sql_ident(d) for d in plan.dimensions)
            group_by = f"GROUP BY {gb}"

        where_sql = _filters_sql(plan.filters)

        order_sql = ""
        if plan.sort:
            parts = []
            for s in plan.sort:
                by = s.get("by")
                desc = bool(s.get("desc", False))
                if by:
                    parts.append(f"{_sql_ident(by)} {'DESC' if desc else 'ASC'}")
            if parts:
                order_sql = "ORDER BY " + ", ".join(parts)

        # LIMIT is validated in planning_tool (min 1, max 200000). Still keep this defensive.
        limit_sql = f"LIMIT {int(plan.limit)}" if int(plan.limit) > 0 else ""

        sql = f"""
        SELECT {", ".join(select_cols)}
        {join_sql}
        {where_sql}
        {group_by}
        {order_sql}
        {limit_sql}
        """.strip()

        log.info("Executing SQL", extra={"sql": (sql[:500] + ("..." if len(sql) > 500 else ""))})
        return con.execute(sql).df()
    finally:
        con.close()
