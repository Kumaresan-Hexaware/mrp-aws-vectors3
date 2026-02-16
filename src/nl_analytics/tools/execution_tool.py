from __future__ import annotations

from dataclasses import replace
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
from nl_analytics.db.athena import AthenaExecutor, rewrite_sql_for_athena, filter_existing_tables
from nl_analytics.db.redshift import RedshiftExecutor

log = get_logger("tools.execution")


def _sql_ident(name: str, dialect: SqlDialect) -> str:
    return dialect.ident(name)


def build_join_sql(registry: SchemaRegistry, plan_tables: List[str], dialect: str = "athena") -> Tuple[str, str]:
    """Build a FROM/JOIN clause.

    The registry join graph is directional (left_table -> right_table). Some
    table sets are only joinable if you pick the correct root. We therefore try
    each candidate table as root (in the provided order) and pick the first that
    yields a valid join path.
    """
    if not plan_tables:
        return "", ""

    if len(plan_tables) == 1:
        t = plan_tables[0]
        return f'FROM { _sql_ident(t, dialect) } AS {t}', t

    chosen_tables = plan_tables
    join_path = None
    last_err: Exception | None = None

    for cand in plan_tables:
        ordered = [cand] + [t for t in plan_tables if t != cand]
        try:
            join_path = registry.find_join_path(ordered)
            chosen_tables = ordered
            break
        except Exception as e:
            last_err = e
            continue

    if join_path is None:
        # Re-raise the most informative error we saw.
        if last_err:
            raise last_err
        raise SchemaValidationError(f"No registry join path can connect requested tables: {plan_tables}")

    root = chosen_tables[0]

    # Start FROM root
    sql = f'FROM { _sql_ident(root, dialect) } AS {root}'

    # Then apply join rules along the join_path
    for jr in join_path:
        join_type = jr.join_type.upper()
        right = jr.right_table
        on_pairs = []
        for lk, rk in zip(jr.left_keys, jr.right_keys):
            on_pairs.append(
                f"{jr.left_table}.{_sql_ident(lk, dialect)} = {right}.{_sql_ident(rk, dialect)}"
            )
        sql += f"\n{join_type} JOIN { _sql_ident(right, dialect) } AS {right} ON " + " AND ".join(on_pairs)

    return sql, root

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

    def _has_arith(s: str) -> bool:
        in_single = False
        in_double = False
        for ch in s:
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if ch in "+-*/":
                return True
        return False

    arith = _has_arith(expr)

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
                    seg = re.sub(rf"\b{re.escape(c)}\b", (dialect.try_cast_double(f"{alias}.{_sql_ident(col, dialect)}") if arith else f"{alias}.{_sql_ident(col, dialect)}"), seg)
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
                    seg = re.sub(rf"\b{re.escape(c)}\b", (dialect.try_cast_double(f"{alias}.{_sql_ident(col, dialect)}") if arith else f"{alias}.{_sql_ident(col, dialect)}"), seg)
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
            seg = re.sub(rf"\b{re.escape(c)}\b", (dialect.try_cast_double(f"{alias}.{_sql_ident(col, dialect)}") if arith else f"{alias}.{_sql_ident(col, dialect)}"), seg)
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
                col_sql = f"DISTINCT {dialect.try_cast_double(col_sql)}"
            else:
                col_sql = f"DISTINCT {col_sql}"
        else:
            if agg_l in {"sum", "avg"}:
                col_sql = dialect.try_cast_double(col_sql)

        out.append(f"{agg.upper()}({col_sql}) AS {_sql_ident(name, dialect)}")
    return out


def _compile_sql(
    registry: SchemaRegistry,
    plan: QueryPlan,
    tables: List[str],
    col_ref: Dict[str, str],
    dialect: SqlDialect,
) -> str:
    """Compile a QueryPlan into SQL for the given dialect.

    This is used for execution (db_type dialect) and also for rendering alternate
    dialect SQL (ex: Postgres) for testing/logging.
    """

    # Special case: "average X across all unique instruments".
    # Meaning: compute per-InstrumentID value first, then average across instruments
    # so each instrument contributes once (avoids row-weighting bias).
    #
    # If there *are* dimensions (e.g., "... by instrument types"), we compute:
    #   AVG( per_instrument_metric ) GROUP BY <dimensions excluding InstrumentID>
    #
    try:
        hints = getattr(plan, "hints", None) or {}
    except Exception:
        hints = {}

    if hints.get("unique_entity_avg") == "InstrumentID" and tables and (plan.metrics or []) and len(plan.metrics) == 1:
        metric = (plan.metrics or [])[0]
        metric_name = str(metric.get("name", "metric"))

        # Build the inner per-instrument aggregate using the same metric compiler
        # so we get:
        #  - proper column qualification (alias."col")
        #  - TRY_CAST(... AS DOUBLE) for Athena for SUM/AVG over CSV varchar columns
        # Note: we override the output name to _inst_value.
        inst_metric_sql = _metric_sql(
            [{"name": "_inst_value", "expr": str(metric.get("expr", "")).strip() or "AVG(1)"}],
            col_ref,
            dialect,
        )[0]

        # Resolve InstrumentID reference.
        inst = "InstrumentID"
        if inst in col_ref:
            inst_alias, inst_col = col_ref[inst].split(".", 1)
            inst_sql = f"{inst_alias}.{_sql_ident(inst_col, dialect)}"
        else:
            inst_sql = _sql_ident(inst, dialect)

        # Dimensions (drop InstrumentID if it accidentally appears alongside other dims).
        dims = list(plan.dimensions or [])
        dims_clean = [
            d for d in dims
            if str(d).strip().lower() not in ("instrumentid",) and not str(d).strip().lower().endswith(".instrumentid")
        ]

        dim_sqls = []
        dim_out = []
        for d in dims_clean:
            d0 = str(d).strip()
            key = d0.split(".")[-1]
            if key in col_ref:
                a, c = col_ref[key].split(".", 1)
                dim_sqls.append(f"{a}.{_sql_ident(c, dialect)}")
                dim_out.append(_sql_ident(key, dialect))
            else:
                # Fallback: try to use the raw dimension token as-is
                dim_sqls.append(_sql_ident(d0, dialect))
                dim_out.append(_sql_ident(key, dialect))

        join_sql, _ = build_join_sql(registry, tables, dialect)
        where_sql = _filters_sql(plan.filters, col_ref, dialect, registry=registry)

        inner_select = [f"{inst_sql} AS {_sql_ident(inst, dialect)}"]
        inner_select += [f"{s} AS {o}" for s, o in zip(dim_sqls, dim_out)]
        inner_select.append(inst_metric_sql)

        inner_group_by = [inst_sql] + dim_sqls

        if dim_sqls:
            outer_select = ", ".join(dim_out + [f"AVG({_sql_ident('_inst_value', dialect)}) AS {_sql_ident(metric_name, dialect)}"])
            outer_group_by = ", ".join(dim_out)
        else:
            outer_select = f"AVG({_sql_ident('_inst_value', dialect)}) AS {_sql_ident(metric_name, dialect)}"
            outer_group_by = ""

        outer_group_sql = f"\nGROUP BY {outer_group_by}" if outer_group_by else ""

        return f"""WITH _per_inst AS (
    SELECT {', '.join(inner_select)}
    {join_sql}
    {where_sql}
    GROUP BY {', '.join(inner_group_by)}
)
SELECT {outer_select}
FROM _per_inst{outer_group_sql}""".strip()

# Special case: overall distinct count across multiple tables with no dimensions.
    # Example: COUNT(DISTINCT InstrumentID) across all tables should NOT pick a single base table.
    if tables and len(tables) > 1 and (not (plan.dimensions or [])) and (plan.metrics or []) and len(plan.metrics) == 1:
        expr0 = str((plan.metrics or [])[0].get("expr", "")).strip()
        m = re.match(r"(?is)^count\s*\(\s*distinct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$", expr0)
        if m:
            col = m.group(1)
            selects: List[str] = []
            for t in tables:
                try:
                    cols = set(registry.columns_for_table(t))
                except Exception:
                    cols = set()
                if col in cols:
                    selects.append(f'SELECT {_sql_ident(col, dialect)} AS {_sql_ident(col, dialect)} FROM {_sql_ident(t, dialect)}')
            if selects:
                # UNION ALL is safe here because we apply COUNT(DISTINCT ...) on top.
                # Using UNION (distinct) can be slower and does not improve correctness.
                union_sql = "\nUNION ALL\n".join(selects)
                metric_name = str((plan.metrics or [])[0].get("name", "metric"))
                return f"""WITH _u AS (
{union_sql}
)
SELECT COUNT(DISTINCT {_sql_ident(col, dialect)}) AS {_sql_ident(metric_name, dialect)}
FROM _u""".strip()


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
    # Only GROUP BY when we are producing aggregated metrics.
    # Row-level "exception" / "listing" queries may have dimensions only and must not be grouped.
    if plan.metrics and plan.dimensions:
        gb_parts = []
        for d in plan.dimensions:
            if d in col_ref:
                alias, col = col_ref[d].split(".", 1)
                gb_parts.append(f"{alias}.{_sql_ident(col, dialect)}")
            else:
                gb_parts.append(_sql_ident(d, dialect))
        group_by = f"GROUP BY {', '.join(gb_parts)}"

    where_sql = _filters_sql(plan.filters, col_ref, dialect, registry=registry)

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
                col_sql = f"{alias}.{_sql_ident(col, dialect)}"

                # IMPORTANT: Many MRP nzf/CSV-backed Athena external tables end up with numeric-looking
                # fields inferred/declared as VARCHAR. Sorting those lexicographically produces wrong
                # results (e.g., '99' > '100'). If the schema registry says the column is numeric, use a
                # defensive TRY_CAST in ORDER BY.
                try:
                    col_type = (
                        registry.tables.get(alias).columns.get(col).type  # type: ignore[union-attr]
                        or ""
                    ).strip().lower()
                except Exception:
                    col_type = ""
                numeric_types = {
                    "int",
                    "integer",
                    "bigint",
                    "smallint",
                    "tinyint",
                    "float",
                    "double",
                    "real",
                    "decimal",
                    "numeric",
                    "number",
                }
                if col_type in numeric_types:
                    col_sql = dialect.try_cast_double(col_sql)

                parts.append(f"{col_sql} {'DESC' if desc else 'ASC'}")
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

    return sql


def _filters_sql(
    filters: List[str],
    col_ref: Dict[str, str],
    dialect: SqlDialect,
    registry: Optional[SchemaRegistry] = None,
) -> str:
    """Render simple AND-combined filters safely.

    Supported patterns (case-insensitive):
      - Col = 10
      - Col != 'abc'
      - Col >= 1.2
      - Col IS NULL
      - Col IS NOT NULL

    Note: planning_tool canonicalizes the first token to a column name, so we primarily expect bare columns,
    but we defensively handle optional table qualification (t.Col) here as well.
    """
    if not filters:
        return ""

    safe_parts: List[str] = []
    for f in filters:
        f = (f or "").strip()
        if not f:
            continue

        # IS NULL / IS NOT NULL
        m_null = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s+is\s+(not\s+)?null\s*$", f, flags=re.I)
        if m_null:
            col_tok = m_null.group(1)
            is_not = bool(m_null.group(2))
            if "." in col_tok:
                alias, col2 = col_tok.split(".", 1)
                col_sql = f"{alias}.{_sql_ident(col2, dialect)}"
            elif col_tok in col_ref:
                alias, col2 = col_ref[col_tok].split(".", 1)
                col_sql = f"{alias}.{_sql_ident(col2, dialect)}"
            else:
                col_sql = _sql_ident(col_tok, dialect)
            safe_parts.append(f"{col_sql} IS {'NOT ' if is_not else ''}NULL")
            continue

        # Simple binary comparisons
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*(=|!=|>=|<=|>|<)\s*(.+)$', f)
        if not m:
            continue
        col_tok, op, val = m.group(1), m.group(2), m.group(3).strip()

        # Qualify ambiguous columns if possible (or preserve explicit table qualifier)
        if "." in col_tok:
            alias, col2 = col_tok.split(".", 1)
            col_sql = f"{alias}.{_sql_ident(col2, dialect)}"
        elif col_tok in col_ref:
            alias, col2 = col_ref[col_tok].split(".", 1)
            col_sql = f"{alias}.{_sql_ident(col2, dialect)}"
        else:
            col_sql = _sql_ident(col_tok, dialect)

        if val.startswith(("'", '"')) and val.endswith(("'", '"')):
            safe_val = val
        else:
            if re.match(r'^-?\d+(\.\d+)?$', val):
                safe_val = val
            else:
                safe_val = "'" + val.replace("'", "''") + "'"

        # Defensive numeric comparisons:
        # Athena external tables frequently declare numeric-looking columns as VARCHAR.
        # If the schema registry says the column is numeric AND we compare against a numeric literal,
        # wrap the column in TRY_CAST(.. AS DOUBLE) for correctness and to avoid TYPE_MISMATCH.
        if registry is not None and re.match(r'^-?\d+(\.\d+)?$', safe_val):
            try:
                alias_name = None
                col_name = None
                if "." in col_tok:
                    alias_name, col_name = col_tok.split(".", 1)
                elif col_tok in col_ref:
                    alias_name, col_name = col_ref[col_tok].split(".", 1)

                col_type = ""
                if alias_name and col_name:
                    col_type = (
                        registry.tables.get(alias_name).columns.get(col_name).type  # type: ignore[union-attr]
                        or ""
                    ).strip().lower()

                is_numeric = any(k in col_type for k in ["int", "float", "double", "decimal", "numeric", "real"])
                if is_numeric and op in {">", "<", ">=", "<=", "=", "!="}:
                    col_sql = dialect.try_cast_double(col_sql)
            except Exception:
                pass

        safe_parts.append(f"{col_sql} {op} {safe_val}")

    if not safe_parts:
        return ""
    return "WHERE " + " AND ".join(safe_parts)


def _choose_base_table(plan: QueryPlan, tables: List[str]) -> str:
    """Choose a driving (root) table for the FROM/JOIN clause.

    The execution layer builds a join path starting from the first table in `tables`.
    In earlier revisions we referenced this helper but never defined it, which causes:
      NameError: name '_choose_base_table' is not defined

    A *perfect* choice would consider where filters/dimensions originate so that JOIN
    direction (especially for LEFT JOIN graphs) preserves expected semantics.

    For now, we use a deterministic heuristic that works well for the most common
    PVR join patterns in this project:
      - schedule/reset questions: drive from PVR01200 (schedule)
      - payment structure questions: drive from PVR00500
      - valuation/risk questions: drive from PVR00600
      - otherwise: keep the existing order (first table)

    This fixes the runtime error and yields stable, predictable SQL.
    """

    if not tables:
        return ""

    # Normalize to lowercase for matching.
    tset = {str(t).strip().lower(): t for t in tables}

    priority = [
        "pvr01200",  # payment schedule / resets (best anchor for 012/014/015/016/017 joins)
        "pvr00500",  # payment structure / margins / caps
        "pvr00600",  # valuation analytics / risk metrics
        "pvr01100",  # loan/instrument analytics
        "pvr00400",  # instrument master
        "pvr00300",  # modeling outputs
        "pvr00100",  # run header
        "pvr01000",  # run-portfolio bridge
        "pvr01300",  # option exercise
        "pvr01400",
        "pvr01500",
        "pvr01600",
        "pvr01900",
        "pvr00900",
    ]

    for p in priority:
        if p in tset:
            return tset[p]

    return tables[0]


# --- Plan sanitation helpers ----------------------------------------------------

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _extract_identifiers(expr: str) -> List[str]:
    """Extract identifier-like tokens from an expression (best-effort)."""
    if not expr:
        return []
    tmp = re.sub(r"'([^']|'')*'", " ", expr)
    tmp = re.sub(r'"([^"]|"")*"', " ", tmp)
    toks = _IDENT_RE.findall(tmp)
    drop = {k.lower() for k in (
        "select","from","where","group","by","order","limit","asc","desc","and","or","not","in",
        "like","is","null","case","when","then","else","end","distinct","as","on","join","left","right",
        "inner","outer","between","try_cast","cast","sum","avg","min","max","count","coalesce","round","date_trunc","strftime"
    )}
    return [t for t in toks if t.lower() not in drop]


def _is_aggregate_expr(expr: str) -> bool:
    e = (expr or "").strip().lower()
    return bool(re.match(r"^(sum|avg|min|max|count)\s*\(", e))


def _default_agg_for_metric(name: str, col: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ["count", "num", "number", "rows"]):
        return f"COUNT({col})"
    if any(k in n for k in ["avg", "average", "mean"]):
        return f"AVG({col})"
    if any(k in n for k in ["min", "minimum", "lowest"]):
        return f"MIN({col})"
    if any(k in n for k in ["max", "maximum", "highest"]):
        return f"MAX({col})"
    return f"SUM({col})"


def _sanitize_metrics_for_groupby(metrics: List[Dict[str, str]], dimensions: List[str]) -> List[Dict[str, str]]:
    """If dimensions exist, ensure metrics are aggregated to avoid GROUP BY errors."""
    if not dimensions:
        return metrics

    fixed: List[Dict[str, str]] = []
    for m in metrics or []:
        name = m.get("name", "")
        expr = _strip_expr_alias(m.get("expr", ""))
        expr = re.sub(r"\s+", " ", expr).strip()

        if _is_aggregate_expr(expr) or re.search(r"(?i)\b(sum|avg|min|max|count)\s*\(", expr):
            fixed.append({"name": name, "expr": expr})
            continue

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr):
            fixed.append({"name": name or expr, "expr": _default_agg_for_metric(name or expr, expr)})
            continue

        fixed.append({"name": name or "metric", "expr": f"AVG({expr})"})
    return fixed


def _prune_plan_to_tables(plan: QueryPlan, registry: SchemaRegistry, tables: List[str]) -> QueryPlan:
    """Return a NEW plan pruned to columns available in the provided tables.

    QueryPlan is a frozen dataclass, so we must not mutate it.
    """
    allowed_cols: set[str] = set()
    for t in tables or []:
        try:
            allowed_cols.update(registry.columns_for_table(t))
        except Exception:
            continue

    new_dimensions = [d for d in (plan.dimensions or []) if d in allowed_cols]

    kept_filters: List[str] = []
    for f in plan.filters or []:
        ids = _extract_identifiers(f)
        if ids and all(i in allowed_cols for i in ids):
            kept_filters.append(f)
        elif not ids:
            kept_filters.append(f)
    new_filters = kept_filters

    kept_metrics: List[Dict[str, str]] = []
    for m in plan.metrics or []:
        expr = _strip_expr_alias(m.get("expr", ""))
        if re.match(r"(?is)^\s*count\s*\(\s*\*\s*\)\s*$", expr):
            kept_metrics.append(m)
            continue
        ids = _extract_identifiers(expr)
        if ids and all(i in allowed_cols for i in ids):
            kept_metrics.append(m)
        elif not ids:
            kept_metrics.append(m)
    new_metrics = kept_metrics

    return replace(plan, dimensions=new_dimensions, filters=new_filters, metrics=new_metrics)


def _infer_tables_from_plan(registry: SchemaRegistry, plan: QueryPlan) -> List[str]:
    """Infer candidate tables from referenced columns in metrics/dimensions/filters."""
    wanted: set[str] = set(plan.dimensions or [])
    for m in plan.metrics or []:
        wanted.update(_extract_identifiers(m.get("expr", "")))
    for f in plan.filters or []:
        wanted.update(_extract_identifiers(f))

    candidates: List[str] = []
    for col in wanted:
        for t in registry.tables():
            try:
                if col in registry.columns_for_table(t):
                    candidates.append(t)
            except Exception:
                continue

    seen = set()
    ordered: List[str] = []
    for t in (plan.tables or []) + candidates:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def execute_plan(session: DataSession, plan: QueryPlan) -> pd.DataFrame:
    registry = session.registry
    db_type = (session.settings.db_type or "duckdb").strip().lower()
    dialect = dialect_for(db_type)

    # QueryPlan is a frozen dataclass; never mutate it in-place.
    # Resolve plan tables to registry-canonical names (case-insensitive)
    tables = [session.canonical_table_name(t) for t in plan.tables]

    # Athena runs against Glue catalog; the registry may include tables that are not loaded to Athena yet.
    if db_type == "athena":
        existing, missing = filter_existing_tables(session.settings, tables)
        if missing:
            log.warning("Some plan tables are missing in Athena/Glue; pruning", extra={"missing_tables": missing})
        tables = existing
        plan = _prune_plan_to_tables(plan, registry, tables)

        if not tables:
            inferred = _infer_tables_from_plan(registry, plan)
            existing2, missing2 = filter_existing_tables(session.settings, inferred)
            if existing2:
                log.warning(
                    "Recovered plan tables by inference",
                    extra={"inferred_tables": existing2, "missing_tables": missing2},
                )
                tables = existing2
                plan = _prune_plan_to_tables(plan, registry, tables)

        if not tables:
            raise AgentExecutionError(
                "None of the planned/inferred tables exist in Athena/Glue catalog. Load/create the tables or adjust the query."
            )


    # Choose driving table for correct LEFT JOIN semantics (important for filters on joined tables)
    base = _choose_base_table(plan, tables)
    if base and tables and base != tables[0]:
        tables = [base] + [t for t in tables if t != base]

    # Ensure metrics are aggregated when dimensions exist (avoids GROUP BY errors)
    plan = replace(plan, metrics=_sanitize_metrics_for_groupby(plan.metrics, plan.dimensions))

    # Build column->qualified reference map to prevent ambiguous column errors
    col_ref = _build_column_ref_map(registry, tables)
    sql = _compile_sql(registry, plan, tables, col_ref, dialect)

    # For Athena, persist/log the *actual* SQL we send to Athena.
    # (AthenaExecutor also applies a safety rewrite for numeric predicates, but
    # we do it here so session.last_sql + DynamoDB payload contain the real query.)
    if db_type == "athena":
        try:
            sql = rewrite_sql_for_athena(sql)
        except Exception:
            # Best-effort; if rewrite fails, keep original SQL.
            pass

    # Expose SQL to the session (for downstream persistence / audit logging).
    try:
        session.last_sql = sql
        session.last_db_type = db_type
    except Exception:
        pass

    # Always print the engine SQL used for execution.
    log.info(f"{db_type.upper()} SQL :::\n{sql}")

    # For testing only: also render Postgres SQL (do NOT execute it).
    try:
        pg_dialect = dialect_for("postgres")
        pg_sql = _compile_sql(registry, plan, tables, col_ref, pg_dialect)
        #log.info(f"POSTGRES SQL :::\n{pg_sql}")
    except Exception as e:
        log.warning(f"Failed to render POSTGRES SQL (render-only): {e}")
    # Keep the old "Executing SQL" message for continuity (extra fields are not printed by default logger).
    log.info("Executing SQL")

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
