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

def _metric_sql(metrics: List[Dict[str, str]]) -> List[str]:
    out = []
    for m in metrics:
        name = m["name"]
        expr = m["expr"].strip()
        agg = expr.split("(", 1)[0].strip()
        col = expr.split("(", 1)[1].rstrip(")").strip()
        agg_l = agg.lower()

        if col == "*":
            out.append(f"{agg.upper()}(*) AS {_sql_ident(name)}")
            continue

        # SUM/AVG should be numeric-safe because many CSV/NZF fields load as strings.
        if agg_l in {"sum", "avg"}:
            out.append(
                f"{agg.upper()}(TRY_CAST({_sql_ident(col)} AS DOUBLE)) AS {_sql_ident(name)}"
            )
        else:
            out.append(f"{agg.upper()}({_sql_ident(col)}) AS {_sql_ident(name)}")
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
        select_cols.extend(_metric_sql(plan.metrics))

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
