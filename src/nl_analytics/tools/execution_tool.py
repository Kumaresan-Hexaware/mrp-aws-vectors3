from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import re
import duckdb
import pandas as pd

from nl_analytics.data.session import DataSession
from nl_analytics.schema.registry import SchemaRegistry, JoinRule
from nl_analytics.tools.planning_tool import QueryPlan
from nl_analytics.exceptions.errors import AgentExecutionError, SchemaValidationError
from nl_analytics.logging.logger import get_logger
from nl_analytics.db.utils import SqlDialect, dialect_for
from nl_analytics.db.athena import AthenaExecutor
from nl_analytics.db.redshift import RedshiftExecutor

log = get_logger("tools.execution")


def _sql_ident(name: str, dialect: SqlDialect) -> str:
    return dialect.ident(name)


def build_join_sql(registry: SchemaRegistry, plan_tables: List[str], dialect: SqlDialect) -> Tuple[str, List[JoinRule]]:
    """Build FROM + JOIN clause using registry join rules.

    Note: registry.find_join_path() may return intermediate tables. We rely on plan_tables to already
    include those intermediates (planning_tool expands them), but this function is robust either way.
    """
    join_path = registry.find_join_path(plan_tables)
    root = plan_tables[0]
    sql = f"FROM {_sql_ident(root, dialect)} AS {root}"
    joined = {root}
    for e in join_path:
        if e.left_table not in joined:
            raise SchemaValidationError("Join path order invalid; cannot execute joins safely.")
        lt_alias = e.left_table
        rt_alias = e.right_table
        conds = []
        for lk, rk in zip(e.left_keys, e.right_keys):
            conds.append(f"{lt_alias}.{_sql_ident(lk, dialect)} = {rt_alias}.{_sql_ident(rk, dialect)}")
        cond = " AND ".join(conds) if conds else "1=1"
        sql += f" {e.join_type.upper()} JOIN {_sql_ident(rt_alias, dialect)} AS {rt_alias} ON {cond}"
        joined.add(rt_alias)
    return sql, join_path


def _strip_expr_alias(expr: str) -> str:
    # Remove any accidental "AS alias" the planner might include inside expr.
    return re.sub(r"(?i)\s+as\s+[A-Za-z_][A-Za-z0-9_]*\s*$", "", (expr or "").strip())


def _build_column_ref_map(registry: SchemaRegistry, tables: List[str]) -> Dict[str, str]:
    """Build column -> qualified reference mapping for the selected tables.

    If a column exists in multiple joined tables, default to the root table to avoid ambiguity.
    """
    root = tables[0] if tables else ""
    col_to_tables: Dict[str, List[str]] = {}
    for t in tables:
        try:
            cols = registry.columns_for_table(t)
        except Exception:
            cols = []
        for c in cols:
            col_to_tables.setdefault(c, []).append(t)

    ref: Dict[str, str] = {}
    for c, ts in col_to_tables.items():
        if len(ts) == 1:
            chosen = ts[0]
        else:
            chosen = root  # deterministic
        ref[c] = f"{chosen}.{c}"
    return ref


def _quote_cols_in_expr(expr: str, col_ref: Dict[str, str], dialect: SqlDialect) -> str:
    """Quote + qualify column identifiers within a SQL expression, avoiding string literals."""
    if not expr:
        return ""

    # Strip any table qualification the model might already include (t.col) to keep rewriting consistent
    expr = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", expr)

    # Safety: prohibit multi-statement / comments
    low = expr.lower()
    if ";" in low or "--" in low or "/*" in low or "*/" in low:
        raise AgentExecutionError("Unsafe tokens in metric expression")

    cols_sorted = sorted(col_ref.keys(), key=len, reverse=True)

    out: List[str] = []
    in_single = False
    in_double = False
    buf = ""
    for i, ch in enumerate(expr):
        if ch == "'" and not in_double:
            # handle escaped ''
            if in_single and i + 1 < len(expr) and expr[i + 1] == "'":
                buf += "''"
                continue
            if buf:
                seg = buf
                for c in cols_sorted:
                    qual = col_ref[c]
                    alias, col = qual.split(".", 1)
                    seg = re.sub(rf"\b{re.escape(c)}\b", f"{alias}.{_sql_ident(col, dialect)}", seg)
                out.append(seg)
                buf = ""
            out.append("'")
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            if buf:
                seg = buf
                for c in cols_sorted:
                    qual = col_ref[c]
                    alias, col = qual.split(".", 1)
                    seg = re.sub(rf"\b{re.escape(c)}\b", f"{alias}.{_sql_ident(col, dialect)}", seg)
                out.append(seg)
                buf = ""
            out.append('"')
            in_double = not in_double
            continue
        buf += ch

    if buf:
        seg = buf
        for c in cols_sorted:
            qual = col_ref[c]
            alias, col = qual.split(".", 1)
            seg = re.sub(rf"\b{re.escape(c)}\b", f"{alias}.{_sql_ident(col, dialect)}", seg)
        out.append(seg)
    return "".join(out)


def _metric_sql(metrics: List[Dict[str, str]], col_ref: Dict[str, str], dialect: SqlDialect) -> List[str]:
    out: List[str] = []
    for m in metrics:
        name = m["name"]
        expr = _strip_expr_alias(m["expr"])
        expr = re.sub(r"\s+", " ", expr).strip()

        # Repair COUNT(DISTINCT Instrument ID) -> COUNT(DISTINCT InstrumentID) if needed.
        if re.search(r"(?i)\bdistinct\b", expr):
            m_dist = re.match(
                r"(?is)^(sum|avg|min|max|count)\s*\(\s*distinct\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
                expr,
            )
            if m_dist:
                combined = f"{m_dist.group(2)}{m_dist.group(3)}"
                if combined in set(col_ref.keys()):
                    expr = f"{m_dist.group(1)}(DISTINCT {combined})"

        # Simple AGG(col) patterns
        if re.match(r"(?is)^\s*count\s*\(\s*\*\s*\)\s*$", expr):
            out.append(f"COUNT(*) AS {_sql_ident(name, dialect)}")
            continue

        m_simple = re.match(
            r"(?is)^\s*(sum|avg|min|max|count)\s*\(\s*(distinct\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
            expr,
        )
        if not m_simple:
            # Complex expression (CASE WHEN, ROUND wrapper, arithmetic, etc.)
            expr_sql = _quote_cols_in_expr(expr, col_ref, dialect)
            out.append(f"{expr_sql} AS {_sql_ident(name, dialect)}")
            continue

        agg = m_simple.group(1)
        distinct_kw = m_simple.group(2) or ""
        col_inner = m_simple.group(3)
        distinct = bool(distinct_kw)

        # Determine qualified column reference; if not known, fall back to quoted identifier.
        if col_inner in col_ref:
            qual = col_ref[col_inner]
            alias, col = qual.split(".", 1)
            col_sql = f"{alias}.{_sql_ident(col, dialect)}"
        else:
            col_sql = _sql_ident(col_inner, dialect)

        agg_l = agg.lower()
        if distinct:
            if agg_l in {"sum", "avg"}:
                col_sql = f"DISTINCT TRY_CAST({col_sql} AS DOUBLE)"
            else:
                col_sql = f"DISTINCT {col_sql}"
        else:
            if agg_l in {"sum", "avg"}:
                col_sql = f"TRY_CAST({col_sql} AS DOUBLE)"

        out.append(f"{agg.upper()}({col_sql}) AS {_sql_ident(name, dialect)}")
    return out


def _filters_sql(filters: List[str], col_ref: Dict[str, str], dialect: SqlDialect) -> str:
    if not filters:
        return ""
    safe_parts: List[str] = []
    for f in filters:
        f = f.strip()
        m = re.match(r'^([A-Za-z0-9_]+)\s*(=|!=|>=|<=|>|<)\s*(.+)$', f)
        if not m:
            continue
        col, op, val = m.group(1), m.group(2), m.group(3).strip()

        # Qualify ambiguous columns if possible
        if col in col_ref:
            qual = col_ref[col]
            alias, col2 = qual.split(".", 1)
            col_sql = f"{alias}.{_sql_ident(col2, dialect)}"
        else:
            col_sql = _sql_ident(col, dialect)

        if val.startswith(("'", '"')) and val.endswith(("'", '"')):
            safe_val = val
        else:
            if re.match(r'^-?\d+(\.\d+)?$', val):
                safe_val = val
            else:
                safe_val = "'" + val.replace("'", "''") + "'"
        safe_parts.append(f"{col_sql} {op} {safe_val}")
    if not safe_parts:
        return ""
    return "WHERE " + " AND ".join(safe_parts)


def execute_plan(session: DataSession, plan: QueryPlan) -> pd.DataFrame:
    registry = session.registry
    db_type = (session.settings.db_type or "duckdb").strip().lower()
    dialect = dialect_for(db_type)

    # Resolve plan tables to registry-canonical names (case-insensitive)
    tables = [session.canonical_table_name(t) for t in plan.tables]

    # Build column->qualified reference map to prevent ambiguous column errors
    col_ref = _build_column_ref_map(registry, tables)
    root = tables[0]

    # SELECT list
    select_cols: List[str] = []
    for d in plan.dimensions:
        # Always qualify; keep output column name stable
        if d in col_ref:
            alias, col = col_ref[d].split(".", 1)
            select_cols.append(f"{alias}.{_sql_ident(col, dialect)} AS {_sql_ident(d, dialect)}")
        else:
            select_cols.append(_sql_ident(d, dialect))

    select_cols.extend(_metric_sql(plan.metrics, col_ref, dialect))

    join_sql, _ = build_join_sql(registry, tables, dialect)

    group_by = ""
    if plan.dimensions:
        gb_parts = []
        for d in plan.dimensions:
            if d in col_ref:
                alias, col = col_ref[d].split(".", 1)
                gb_parts.append(f"{alias}.{_sql_ident(col, dialect)}")
            else:
                gb_parts.append(_sql_ident(d, dialect))
        group_by = f"GROUP BY {', '.join(gb_parts)}"

    where_sql = _filters_sql(plan.filters, col_ref, dialect)

    order_sql = ""
    if plan.sort:
        parts = []
        for s in plan.sort:
            by = s.get("by")
            desc = bool(s.get("desc", False))
            if not by:
                continue
            by_s = str(by)
            # If ordering by a dimension column, qualify it. If ordering by metric name, keep as identifier.
            if by_s in plan.dimensions and by_s in col_ref:
                alias, col = col_ref[by_s].split(".", 1)
                parts.append(f"{alias}.{_sql_ident(col, dialect)} {'DESC' if desc else 'ASC'}")
            else:
                parts.append(f"{_sql_ident(by_s, dialect)} {'DESC' if desc else 'ASC'}")
        if parts:
            order_sql = "ORDER BY " + ", ".join(parts)

    limit_sql = f"LIMIT {int(plan.limit)}" if int(plan.limit) > 0 else ""

    sql = f"""
    SELECT {", ".join(select_cols)}
    {join_sql}
    {where_sql}
    {group_by}
    {order_sql}
    {limit_sql}
    """.strip()

    log.info(f"SQL  ::: {str(sql)}")
    log.info(
        "Executing SQL",
        extra={
            "db_type": db_type,
            "sql": (str(sql[:500] + ("..." if len(sql) > 500 else ""))),
        },
    )

    if db_type == "duckdb":
        for t in tables:
            if not session.has_table(t):
                raise AgentExecutionError(f"Missing table data for '{t}'")

        con = duckdb.connect(database=":memory:")
        try:
            for t in tables:
                con.register(t, session.get_table(t))
            return con.execute(sql).df()
        finally:
            con.close()

    if db_type == "athena":
        return AthenaExecutor(session.settings).execute(sql)
    if db_type == "redshift":
        return RedshiftExecutor(session.settings).execute(sql)

    raise AgentExecutionError(f"Unknown DB_TYPE: {db_type}")
