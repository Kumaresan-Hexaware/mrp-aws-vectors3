from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

import re

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

# Scalar/wrapper functions that are safe in DuckDB and commonly emitted by the planner.
# These are NOT aggregations, but they can wrap aggregations.
ALLOWED_FUNCS = {
    "round",
    "coalesce",
    "nullif",
    "cast",
    "try_cast",
    "date_trunc",
    "strftime",
}

SQL_KEYWORDS = {
    "case",
    "when",
    "then",
    "else",
    "end",
    "distinct",
    "as",
    "and",
    "or",
    "not",
    "in",
    "like",
    "is",
    "null",
    "true",
    "false",
}

DISALLOWED_SQL_FRAGMENTS = {
    ";",
    "--",
    "/*",
    "*/",
    " drop ",
    " alter ",
    " create ",
    " attach ",
    " pragma ",
    " copy ",
    " insert ",
    " update ",
    " delete ",
    " merge ",
    " select ",
    " from ",
    " join ",
    " union ",
}

# Scalar/wrapper functions that are safe in DuckDB and commonly emitted by the planner.
# These are *not* required to be the outer-most function, but we allow them in metric expressions.
ALLOWED_FUNCS = {
    "round",
    "coalesce",
    "nullif",
    "cast",
    "try_cast",
    "date_trunc",
    "strftime",
}

SQL_KEYWORDS = {
    "case",
    "when",
    "then",
    "else",
    "end",
    "distinct",
    "as",
    "and",
    "or",
    "not",
    "in",
    "like",
    "is",
    "null",
    "true",
    "false",
}

DISALLOWED_TOKENS = {
    ";",
    "--",
    "/*",
    "*/",
}


def _strip_qualifiers(expr: str) -> str:
    """Remove table qualifiers like t.Col -> Col (letters-only prefixes).

    This keeps numeric literals like 100.0 intact.
    """
    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", expr or "")


def _split_outside_quotes(expr: str) -> List[str]:
    """Split by single-quotes preserving content so we can avoid rewriting literals."""
    # We keep it simple: odd indices are inside quotes.
    return (expr or "").split("'")


def _canonicalize_expr_columns(expr: str, col_map: Dict[str, str]) -> str:
    """Replace any recognized column aliases in an expression with canonical column names.

    This only rewrites outside of single-quoted string literals.
    """
    parts = _split_outside_quotes(expr)
    tokens = sorted(set(col_map.keys()), key=len, reverse=True)
    # Only apply replacements outside quoted literals (even indices).
    for i in range(0, len(parts), 2):
        seg = parts[i]
        for tok in tokens:
            canon = col_map[tok]
            # Match whole-word column/alias tokens
            seg = re.sub(rf"(?i)\b{re.escape(tok)}\b", canon, seg)
        parts[i] = seg
    return "'".join(parts)


def _referenced_columns(expr: str, col_map: Dict[str, str]) -> Set[str]:
    """Extract canonical columns referenced by expr (best-effort)."""
    expr = _strip_qualifiers(expr)
    parts = _split_outside_quotes(expr)
    out: Set[str] = set()
    for i in range(0, len(parts), 2):
        seg = parts[i]
        for w in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", seg):
            k = w.lower()
            if k in col_map:
                out.add(col_map[k])
    return out


def _validate_metric_expr(expr: str) -> None:
    expr = expr or ""
    for tok in DISALLOWED_TOKENS:
        if tok in expr:
            raise SchemaValidationError("Unsafe token in metric expression")



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


def _strip_string_literals(sql: str) -> str:
    """Remove content inside single-quoted string literals to avoid false token matches."""
    if not sql:
        return ""
    out = []
    in_quote = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            out.append("'")
            if in_quote:
                # handle escaped ''
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    # stay in quote, keep placeholder
                    out.append("'")
                    i += 2
                    continue
                in_quote = False
            else:
                in_quote = True
            i += 1
            continue
        if in_quote:
            # replace literal content with a space
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _expr_tokens(expr: str) -> Set[str]:
    expr = _strip_string_literals(expr)
    return set(m.group(0) for m in _IDENT_RE.finditer(expr or ""))


def _expr_contains_agg(expr: str) -> bool:
    if not expr:
        return False
    return bool(re.search(r"(?i)\b(?:sum|avg|min|max|count)\s*\(", expr))


def _canonicalize_expr_columns(expr: str, all_cols: set[str], col_map: Dict[str, str]) -> str:
    """Replace known aliases/columns in an expression with canonical column names.

    This keeps the executor simple and ensures consistent quoting.
    """
    if not expr:
        return ""

    # Normalize whitespace a bit so DISTINCT / identifiers split across lines become readable.
    expr = re.sub(r"\s+", " ", expr).strip()

    # Strip any table qualification (t.col) from identifiers (safe: doesn't touch numeric literals like 100.0).
    expr = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", expr)

    # Replace tokens outside string literals only.
    parts: List[str] = []
    buf: List[str] = []
    in_quote = False
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "'":
            # flush buffer
            if buf:
                parts.append("".join(buf))
                buf = []
            # keep the quote and flip state
            parts.append("'")
            if in_quote:
                if i + 1 < len(expr) and expr[i + 1] == "'":
                    # escaped quote inside literal
                    parts.append("'")
                    i += 2
                    continue
                in_quote = False
            else:
                in_quote = True
            i += 1
            continue
        if in_quote:
            parts.append(ch)
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))

    # Canonicalize only in non-literal segments (even indexes if we consider quotes as separators).
    # We used a flat list with quotes preserved, so we'll just process segments that are NOT inside quotes.
    out_parts: List[str] = []
    inside = False
    for seg in parts:
        if seg == "'":
            inside = not inside
            out_parts.append(seg)
            continue
        if inside:
            out_parts.append(seg)
            continue

        seg_out = seg
        # Replace exact-word matches for aliases/column names.
        # Sort keys by length to avoid partial overlaps.
        keys = sorted(col_map.keys(), key=len, reverse=True)
        for k in keys:
            canon = col_map[k]
            seg_out = re.sub(rf"(?i)\b{re.escape(k)}\b", canon, seg_out)
        out_parts.append(seg_out)
    return "".join(out_parts)


def _referenced_cols_in_expr(expr: str, col_map: Dict[str, str]) -> Set[str]:
    """Best-effort extraction of referenced canonical columns from an expression."""
    cols: Set[str] = set()
    for tok in _expr_tokens(expr):
        t = tok.lower()
        if t in SQL_KEYWORDS or t in ALLOWED_AGGS or t in ALLOWED_FUNCS:
            continue
        if t in col_map:
            cols.add(col_map[t])
        else:
            # normalized fallback
            n = "".join(ch for ch in tok.lower() if ch.isalnum())
            if n in col_map:
                cols.add(col_map[n])
    return cols


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

    # Precompute a column/alias map across the initially requested tables.
    # Used only for table pruning (we rebuild it later after pruning).
    candidate_cols: set[str] = set()
    for t in tables:
        candidate_cols |= set(registry.columns_for_table(t))
    candidate_col_map: Dict[str, str] = {c.lower(): c for c in candidate_cols}
    for c in list(candidate_cols):
        n = "".join(ch.lower() for ch in c if ch.isalnum())
        if n:
            candidate_col_map.setdefault(n, c)
    for k, v in registry.column_alias_map_for_tables(tables).items():
        candidate_col_map.setdefault(str(k).lower(), v)

    # ---- prune extra tables if they are not needed by referenced columns ----
    referenced_cols: List[str] = []
    referenced_cols.extend([str(d) for d in (plan.get("dimensions") or [])])

    # Metric expressions can be complex (CASE WHEN, ROUND wrapper, etc.).
    # Extract any referenced columns we can find for table pruning.
    for m in list(plan.get("metrics") or []):
        expr = str(m.get("expr", ""))
        referenced_cols.extend(list(_referenced_cols_in_expr(expr, candidate_col_map)))

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

    # Re-run pruning metric extraction with the now-populated col_map.
    # This ensures pruning works even when metrics use CASE/ROUND and rely on aliases.
    referenced_cols = []
    referenced_cols.extend([str(d) for d in (plan.get("dimensions") or [])])
    for m in list(plan.get("metrics") or []):
        expr = str(m.get("expr", ""))
        referenced_cols.extend(list(_referenced_cols_in_expr(expr, col_map)))
    for f in list(plan.get("filters") or []):
        parts = str(f).replace("==", "=").split()
        if parts:
            referenced_cols.append(parts[0].strip())

    needed_tables = []
    for col in referenced_cols:
        col_tok = str(col).split(".")[-1].strip()
        # if it's an alias, canonicalize to real column
        if col_tok.lower() in col_map:
            col_tok = col_map[col_tok.lower()]
        col_l = col_tok.lower()
        for t in tables:
            tcols_l = {c.lower() for c in registry.columns_for_table(t)}
            if col_l in tcols_l:
                if t not in needed_tables:
                    needed_tables.append(t)
                break
    if needed_tables:
        tables = needed_tables

    # Ensure joinability after pruning
    registry.find_join_path(tables)

    # Dimensions canonicalization
    dimensions_in = list(plan.get("dimensions") or [])
    dimensions: List[str] = []
    for d in dimensions_in:
        dimensions.append(_canonicalize_col(str(d), all_cols, col_map))

    # Metrics validation / canonicalization
    metrics_in = list(plan.get("metrics") or [])

    # Reports may legitimately be "row-level" selects (dimensions only) with filters.
    # Example: "List InstrumentID and PaymentStructureID where ...".
    # In that case, metrics can be empty.
    if not metrics_in:
        if plan.get("mode") == "report" and dimensions_in:
            metrics_in = []
        else:
            raise SchemaValidationError("No metrics defined")

    metrics: List[Dict[str, str]] = []
    allowed_idents = set(SQL_KEYWORDS) | set(ALLOWED_AGGS) | set(ALLOWED_FUNCS) | {
        "double",
        "float",
        "real",
        "integer",
        "int",
        "bigint",
        "varchar",
        "date",
        "timestamp",
        "decimal",
        "numeric",
        "try",
    }

    for m in metrics_in:
        name = str(m.get("name", "metric"))
        expr_raw = str(m.get("expr", "")).strip()
        if not expr_raw:
            raise SchemaValidationError("Metric expr cannot be empty")

        # Basic safety: disallow multi-statement or comment injection.
        low = expr_raw.lower()
        if ";" in low or "--" in low or "/*" in low or "*/" in low:
            raise SchemaValidationError("Unsafe tokens in metric expression")

        # Normalize whitespace and canonicalize any aliases/column references.
        expr_norm = re.sub(r"\s+", " ", expr_raw).strip()
        expr_canon = _canonicalize_expr_columns(expr_norm, all_cols, col_map)

        # Require at least one aggregate function somewhere in the expression.
        if not _expr_contains_agg(expr_canon):
            raise SchemaValidationError(
                "Metric expr must contain an aggregation (SUM/AVG/MIN/MAX/COUNT)"
            )

        # Validate identifiers used in the expression are either known columns, allowed funcs, or SQL keywords.
        for tok in _expr_tokens(expr_canon):
            t = tok.lower()
            if t in allowed_idents:
                continue
            if t in col_map:
                continue
            # normalized fallback
            n = "".join(ch for ch in t if ch.isalnum())
            if n in col_map:
                continue
            # If token is inside a string literal, _expr_tokens already strips them.
            raise SchemaValidationError(f"Unknown identifier in metric expression: {tok}")

        metrics.append({"name": name, "expr": expr_canon})

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
