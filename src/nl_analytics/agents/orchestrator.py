from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import os
import time
import json
import hashlib
from pathlib import Path
import re

from nl_analytics.config.settings import Settings
from nl_analytics.schema.registry import SchemaRegistry
from nl_analytics.data.session import DataSession
from nl_analytics.bedrock.client import BedrockClient, BedrockConfig
from nl_analytics.rag.vector_store import (
    ChromaVectorStore,
    S3VectorStore,
    S3VectorsVectorStore,
    VectorStore,
)
from nl_analytics.tools.schema_tool import build_schema_context
from nl_analytics.tools.retrieval_tool import retrieve_schema_chunks
from nl_analytics.tools.planning_tool import PLAN_SCHEMA, validate_plan
from nl_analytics.tools.execution_tool import execute_plan
from nl_analytics.viz.plotly_factory import build_figure
from nl_analytics.exceptions.errors import RetrievalError, SchemaValidationError, AgentExecutionError, PlotlyRenderError
from nl_analytics.logging.logger import get_logger


log = get_logger("agents.orchestrator")

INSUFFICIENT = "Insufficient data."

@dataclass(frozen=True)
class AgentResult:
    ok: bool
    message: str
    plan: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    df: Optional[pd.DataFrame] = None
    figure: Any = None
    figures: Any = None

class AgentOrchestrator:
    def __init__(self, settings: Settings, registry: SchemaRegistry):
        self.settings = settings
        self.registry = registry

        self.bedrock = BedrockClient(
            BedrockConfig(
                region=settings.aws_region,
                chat_model_id=settings.bedrock_chat_model_id,
                embed_model_id=settings.bedrock_embed_model_id,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                use_mock=settings.use_mock_bedrock,
            )
        )

        class _EmbedFn:
            def __init__(self, br: BedrockClient):
                self.br = br
                # Optional: S3 Vectors index can require a specific embedding dimension.
                # Stores the desired dimension (if known) and passes it to BedrockClient.embed().
                self.desired_dimensions = None

            # ✅ IMPORTANT: parameter name must be `input` (not texts)
            def __call__(self, input):
                # Chroma passes a list[str] typically; handle str too
                if isinstance(input, str):
                    input = [input]
                return self.br.embed(list(input), dimensions=self.desired_dimensions)

            def name(self) -> str:
                return "default"

            def get_config(self) -> dict:
                return {"name": self.name()}

        backend = (settings.vector_backend or "chroma").lower().strip()
        if backend == "chroma":
            self.store: VectorStore = ChromaVectorStore(settings.chroma_dir, embedding_fn=_EmbedFn(self.bedrock))
        elif backend == "s3":
            self.store = S3VectorStore(
                bucket=settings.s3_vector_bucket,
                prefix=settings.s3_vector_prefix,
                cache_dir=settings.s3_vector_cache_dir,
                refresh_seconds=settings.s3_vector_refresh_seconds,
                embedding_fn=_EmbedFn(self.bedrock),
            )
        elif backend == "s3vectors":
            self.store = S3VectorsVectorStore(
                bucket=settings.s3vectors_bucket,
                index=settings.s3vectors_index,
                namespace=settings.s3vectors_namespace,
                refresh_seconds=settings.s3vectors_refresh_seconds,
                embedding_fn=_EmbedFn(self.bedrock),
                region_name=getattr(settings, "aws_region", None),
            )
        else:
            raise NotImplementedError(f"VECTOR_BACKEND not supported: {backend}")

        self._ensure_schema_index_if_missing()

    def _schema_state_path(self) -> Path:
        p = Path("data") / ".cache"
        p.mkdir(parents=True, exist_ok=True)
        return p / "schema_index_state.json"

    def _schema_fingerprint(self, schema_ctx) -> str:
        payload = {"tables": schema_ctx.tables, "joins": schema_ctx.joins}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _ensure_schema_index_if_missing(self) -> None:
        """Ensure schema chunks exist in the vector store without re-embedding on every startup.

        Uses a local fingerprint file and, when supported, a cheap sentinel lookup in the vector store.
        Set FORCE_SCHEMA_REINDEX=true to force schema re-embedding/upsert.
        """
        schema_ctx = build_schema_context(self.registry)
        fp = self._schema_fingerprint(schema_ctx)

        force = str(os.environ.get("FORCE_SCHEMA_REINDEX", "")).strip().lower() in ("1", "true", "yes", "y")
        if force:
            self._upsert_schema_index(schema_ctx)
            self._write_schema_state(fp)
            return

        state_ok = self._read_schema_state(fp)
        store_ok = self._schema_sentinel_exists(schema_ctx)

        if state_ok and (store_ok is None or store_ok):
            log.info("Schema index present; skipping schema upsert", extra={"fingerprint": fp})
            return

        self._upsert_schema_index(schema_ctx)
        self._write_schema_state(fp)

    def _read_schema_state(self, fingerprint: str) -> bool:
        p = self._schema_state_path()
        if not p.exists():
            return False
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
            return bool(state and state.get("fingerprint") == fingerprint)
        except Exception:
            return False

    def _write_schema_state(self, fingerprint: str) -> None:
        try:
            self._schema_state_path().write_text(
                json.dumps({"fingerprint": fingerprint, "updated_at": time.time()}),
                encoding="utf-8",
            )
        except Exception:
            # Best-effort only
            pass

    def _schema_sentinel_exists(self, schema_ctx) -> Optional[bool]:
        """Return True/False if we can cheaply detect a schema record in the store; else None."""
        if not schema_ctx.tables:
            return None
        sentinel_id = f"table::{next(iter(schema_ctx.tables.keys()))}"
        if hasattr(self.store, "has_ids"):
            try:
                return bool(getattr(self.store, "has_ids")([sentinel_id])[0])
            except Exception:
                return None
        return None

    def _upsert_schema_index(self, schema_ctx) -> None:
        """Upsert schema into the vector store.

        IMPORTANT (S3 Vectors): metadata is filterable by default and has a strict 2048-byte limit.
        Our S3VectorsVectorStore stores the chunk text in metadata['source_text'] so we MUST keep
        each schema chunk reasonably small. Therefore we chunk large tables into multiple records.
        """
        docs, metas, ids = [], [], []

        # Chunk table schemas so each chunk stays small enough to fit inside S3 Vectors metadata limits
        MAX_CHUNK_CHARS = 1200  # conservative; 1200 chars ~= < 2048 bytes in most cases
        COLS_PER_CHUNK = 18     # fallback bound on very wide tables

        for tname, t in schema_ctx.tables.items():
            desc = (t.get("description") or "").strip()
            t_aliases = t.get("aliases") or []
            t_tags = t.get("business_tags") or []
            # NOTE: inside an f-string, the expression must be fully contained in the {...} braces.
            # Use single quotes around the delimiter to avoid breaking the f-string parser.
            alias_line = f"TABLE_ALIASES: {', '.join(t_aliases)}\n" if t_aliases else ""
            tag_line = f"TAGS: {', '.join(t_tags)}\n" if t_tags else ""
            header = f"TABLE {tname}: {desc}\n" + alias_line + tag_line


            # Convert columns into lines
            col_lines = []
            for c, cinfo in (t.get("columns") or {}).items():
                ctype = (cinfo.get("type") or "").strip()
                cdesc = (cinfo.get("description") or "").strip()
                aliases = (cinfo.get("aliases") or [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                aliases = [str(a).strip() for a in aliases if str(a).strip()]
                if aliases:
                    aliases = aliases[:6]
                    col_lines.append(f"- {c} ({ctype}): {cdesc} | aliases: {', '.join(aliases)}")
                else:
                    col_lines.append(f"- {c} ({ctype}): {cdesc}")

            # Build chunks (first chunk keeps the legacy id 'table::{tname}' for sentinel checks)
            chunk_idx = 0
            cur = header
            cur_count = 0

            def _flush(text: str, idx: int) -> None:
                # Keep first chunk id stable for schema sentinel detection.
                rid = f"table::{tname}" if idx == 0 else f"table::{tname}::chunk{idx}"
                docs.append(text)
                metas.append({"kind": "table", "table": tname, "chunk": idx})
                ids.append(rid)

            for line in col_lines:
                # Hard split by count to avoid pathological long descriptions
                if cur_count >= COLS_PER_CHUNK:
                    _flush(cur + "\n", chunk_idx)
                    chunk_idx += 1
                    cur = header
                    cur_count = 0

                # Soft split by size
                tentative = cur + line + "\n"
                if len(tentative) > MAX_CHUNK_CHARS and cur != header:
                    _flush(cur + "\n", chunk_idx)
                    chunk_idx += 1
                    cur = header + line + "\n"
                    cur_count = 1
                else:
                    cur = tentative
                    cur_count += 1

            # Flush remaining
            if cur.strip():
                _flush(cur if cur.endswith("\n") else (cur + "\n"), chunk_idx)

        # Joins are small and safe
        for i, j in enumerate(schema_ctx.joins):
            text = (
                f"JOIN RULE: {j['left_table']} -> {j['right_table']} "
                f"ON {j['left_keys']} = {j['right_keys']} TYPE {j['join_type']}"
            )
            docs.append(text)
            metas.append({"kind": "join", "left": j["left_table"], "right": j["right_table"]})
            ids.append(f"join::{i}")

        self.store.upsert(ids=ids, texts=docs, metadatas=metas)
    def _clamp01(self, x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _tables_mentioned_in_question(self, question: str) -> set[str]:
        """Best-effort: infer which registry tables are referenced by the NL question via alias resolution.

        Used only for confidence scoring/logging. This does NOT change planning/execution behavior.
        """
        q = (question or "").strip().lower()
        if not q:
            return set()

        words = re.findall(r"[a-z0-9_]+", q)

        # Build 1-3 gram candidates to catch aliases like "net margin".
        candidates: list[str] = []
        for n in (3, 2, 1):
            for i in range(0, max(0, len(words) - n + 1)):
                cand = " ".join(words[i : i + n]).strip()
                if cand:
                    candidates.append(cand)

        tables: set[str] = set()
        seen: set[str] = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            try:
                matches = self.registry.resolve_alias_global(cand)  # [(table, column), ...]
            except Exception:
                matches = []
            for t, _c in (matches or []):
                if t:
                    tables.add(str(t))
        return tables

    def _compute_plan_confidence(self, question: str, plan) -> float:
        """Heuristic plan confidence (0..1) based on validation success and intent alignment."""
        p = 1.0

        mentioned_tables = self._tables_mentioned_in_question(question)
        if len(mentioned_tables) >= 2 and len(getattr(plan, "tables", []) or []) == 1:
            p -= 0.30

        ql = (question or "").lower()
        relationship_intent = any(
            k in ql
            for k in (
                " vs ",
                "versus",
                "relationship",
                "correlat",
                "compare",
                "impact",
                "higher",
                "lower",
            )
        )
        has_metrics = bool(getattr(plan, "metrics", []) or [])
        dims = list(getattr(plan, "dimensions", []) or [])
        if relationship_intent and (not has_metrics) and len(dims) >= 3:
            p -= 0.15

        wants_group = any(k in ql for k in (" for each ", " by ", " per ", "group by"))
        if wants_group and has_metrics and len(dims) == 0:
            p -= 0.20

        return self._clamp01(p)

    def _compute_result_confidence(self, df: pd.DataFrame, plan) -> float:
        """Heuristic result confidence (0..1) based on simple output sanity checks."""
        if df is None or df.empty:
            return 0.0

        q = 1.0

        metric_cols: list[str] = []
        for m in (getattr(plan, "metrics", []) or []):
            if isinstance(m, dict):
                name = m.get("name")
                if name and name in df.columns:
                    metric_cols.append(name)

        if not metric_cols:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    metric_cols.append(c)

        if metric_cols:
            null_ratios = []
            for c in metric_cols[:5]:
                try:
                    null_ratios.append(float(df[c].isna().mean()))
                except Exception:
                    pass
            if null_ratios:
                avg_null = sum(null_ratios) / max(1, len(null_ratios))
                if avg_null > 0.70:
                    q -= 0.30
                elif avg_null > 0.40:
                    q -= 0.15

        dims = list(getattr(plan, "dimensions", []) or [])
        if dims:
            dim0 = dims[0]
            if dim0 in df.columns:
                try:
                    nunique = int(df[dim0].nunique(dropna=True))
                    if nunique <= 1 and len(df) >= 5:
                        q -= 0.15
                except Exception:
                    pass

        return self._clamp01(q)

    def _compute_final_confidence(self, r: float, p: float, q: float, e: float) -> float:
        """Composite confidence = 0.35R + 0.30P + 0.20Q + 0.15E (clamped)."""
        final = (0.35 * self._clamp01(r)) + (0.30 * self._clamp01(p)) + (0.20 * self._clamp01(q)) + (0.15 * self._clamp01(e))
        return self._clamp01(final)



    def run(self, session: DataSession, question: str, mode_hint: str) -> AgentResult:
        """Run the agent in a lightweight ReAct loop (Reason → Act → Observe → retry).

        This preserves existing business logic/tools:
        - RAG retrieval via vector store
        - LLM planning (JSON) + registry validation
        - deterministic execution via execute_plan()
        - Plotly rendering via build_figure()
        """
        question = (question or "").strip()
        if not question:
            return AgentResult(ok=False, message=INSUFFICIENT, confidence=0.0)

        # Log the user query once up-front so failures/retries can be tied back
        # to the originating natural-language request.
        log.info(
            "User query received",
            extra={"question": question, "mode_hint": mode_hint},
        )

        # ReAct retry loop: self-correct common failures (schema mismatch, empty results, execution errors)
        MAX_ITERS = 3
        last_raw_plan: Optional[Dict[str, Any]] = None
        last_confidence: float = 0.0
        feedback: str = ""

        for attempt in range(1, MAX_ITERS + 1):
            try:
                # --- ACT: Retrieve relevant schema chunks (RAG) ---
                retrieval = retrieve_schema_chunks(self.store, question, top_k=self.settings.rag_top_k)

                # --- OBSERVE: confidence (retrieval + plan + execution + result quality) ---
                data_avail = 1.0 if session.tables else 0.0

                # Retrieval confidence (existing behavior)
                r_conf = float(retrieval.confidence * data_avail)

                # Plan/Execution/Result confidences start at 0 until we actually reach those stages
                p_conf = 0.0
                e_conf = 0.0
                q_conf = 0.0

                confidence = self._compute_final_confidence(r_conf, p_conf, q_conf, e_conf)
                last_confidence = confidence

                log.info(
                    "Confidence computed",
                    extra={
                        "final_confidence": round(confidence, 4),
                        "retrieval_confidence": round(r_conf, 4),
                        "plan_confidence": round(p_conf, 4),
                        "execution_confidence": round(e_conf, 4),
                        "result_confidence": round(q_conf, 4),
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                    },
                )

                if self.settings.refuse_below_confidence and r_conf < self.settings.agent_min_confidence:
                    log.info(
                        "Refused due to low confidence",
                        extra={"final_confidence": confidence, "retrieval_confidence": r_conf, "attempt": attempt, "max_attempts": MAX_ITERS},
                    )
                    return AgentResult(ok=False, message=INSUFFICIENT, confidence=confidence)

                # --- Build grounded prompt for planner ---
                sys = (
                    "You are a query planning agent. You must ONLY propose plans using tables/columns that exist in the schema registry. "
                    "Never invent join keys; joins must follow the registry. "
                    "Users may refer to columns by business names/aliases; map those to real columns from the schema. "
                    "IMPORTANT: Choose the correct plan shape: "
                    "(A) If the user asks for counts/sums/averages/min/max or grouping (e.g., 'by', 'per', 'average'), put those in metrics using an aggregation (SUM/AVG/MIN/MAX/COUNT) and list grouping fields in dimensions. "
                    "(B) If the user asks to list rows / exceptions / missing or present values (e.g., 'missing', 'null', 'blank', 'present', 'show instruments where...'), return a row-level plan: put the columns to display in dimensions, leave metrics empty, and express conditions in filters using 'IS NULL' / 'IS NOT NULL' (or simple comparisons). "
                    "For dashboards, you may include either a single 'chart' object or a list of 'charts' (e.g., KPI + bar chart). "
                    "Output JSON only."
                )
                grounding = "\n\n".join(
                    [f"[score={c.score:.3f}] {c.text}" for c in retrieval.chunks[: self.settings.rag_top_k]]
                )

                retry_note = ""
                if feedback:
                    retry_note = (
                        "\n\nPrevious attempt feedback (fix this in the new plan):\n"
                        f"{feedback}\n"
                    )

                user = (
                    f"Mode hint: {mode_hint}\n\n"
                    f"Relevant schema:\n{grounding}"
                    f"{retry_note}\n\n"
                    f"Question: {question}"
                )

                # --- ACT: Plan (LLM) + validate ---
                raw_plan = self.bedrock.chat_json(sys, user, PLAN_SCHEMA)
                if mode_hint in ("report", "dashboard"):
                    raw_plan["mode"] = mode_hint
                last_raw_plan = raw_plan

                plan = validate_plan(self.registry, raw_plan)

                # Update confidence after plan validation
                p_conf = self._compute_plan_confidence(question, plan)
                confidence = self._compute_final_confidence(r_conf, p_conf, q_conf, e_conf)
                last_confidence = confidence
                log.info(
                    "Confidence updated after plan",
                    extra={
                        "final_confidence": round(confidence, 4),
                        "retrieval_confidence": round(r_conf, 4),
                        "plan_confidence": round(p_conf, 4),
                        "execution_confidence": round(e_conf, 4),
                        "result_confidence": round(q_conf, 4),
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                    },
                )

                # Update confidence after plan validation
                p_conf = self._compute_plan_confidence(question, plan)
                confidence = self._compute_final_confidence(r_conf, p_conf, q_conf, e_conf)
                last_confidence = confidence
                log.info(
                    "Confidence updated after plan",
                    extra={
                        "final_confidence": round(confidence, 4),
                        "retrieval_confidence": round(r_conf, 4),
                        "plan_confidence": round(p_conf, 4),
                        "execution_confidence": round(e_conf, 4),
                        "result_confidence": round(q_conf, 4),
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                    },
                )

                # --- ACT: Execute deterministic plan ---
                log.info(f"USER QUESTION ::: {question}")
                df = execute_plan(session, plan)

                # Update confidence after execution + result sanity checks
                e_conf = 1.0
                q_conf = self._compute_result_confidence(df, plan)
                confidence = self._compute_final_confidence(r_conf, p_conf, q_conf, e_conf)
                last_confidence = confidence
                log.info(
                    "Final confidence computed",
                    extra={
                        "final_confidence": round(confidence, 4),
                        "retrieval_confidence": round(r_conf, 4),
                        "plan_confidence": round(p_conf, 4),
                        "execution_confidence": round(e_conf, 4),
                        "result_confidence": round(q_conf, 4),
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                    },
                )

                # --- OBSERVE: empty results -> retry with simpler guidance ---
                if df is None or df.empty:
                    feedback = (
                        "Execution returned no rows. Try a simpler aggregation, remove restrictive filters, "
                        "reduce dimensions, or broaden date constraints. If this is a row-level listing query, metrics may be empty; otherwise ensure at least one aggregated metric is produced."
                    )
                    log.info(
                        "Execution produced empty results; retrying",
                        extra={"attempt": attempt, "max_attempts": MAX_ITERS},
                    )
                    continue

                # --- If dashboard: render charts (keep existing behavior) ---
                if plan.mode == "dashboard":
                    def _repair_spec(spec_in: Dict[str, Any]) -> Dict[str, Any]:
                        """Fill in missing x/y/names/values so plotly_factory can render reliably."""
                        spec = dict(spec_in or {})
                        spec_type = (spec.get("type") or "bar").lower()
                        spec["type"] = spec_type

                        def _as_col(v: Any) -> str | None:
                            """Coerce planner-emitted values to a column name string.

                            Some LLMs emit objects for x/y (e.g., {"expr": "COUNT(*)"}), which breaks
                            Plotly and column lookups. We treat non-string values as missing and let
                            downstream inference pick a safe default.
                            """
                            if isinstance(v, str):
                                v = v.strip()
                                return v or None
                            if isinstance(v, dict):
                                for k in ("column", "name", "value", "field", "expr"):
                                    vv = v.get(k)
                                    if isinstance(vv, str) and vv.strip():
                                        return vv.strip()
                            return None

                        # Normalize common keys up front.
                        for k in ("x", "y", "color", "names", "values", "value"):
                            if k in spec:
                                spec[k] = _as_col(spec.get(k))

                        if spec_type in {"bar", "line", "scatter", "area"}:
                            # If provided x/y aren't real dataframe columns, drop them so we can infer safely.
                            if spec.get("x") and spec.get("x") not in df.columns:
                                spec["x"] = None
                            if spec.get("y") and spec.get("y") not in df.columns:
                                spec["y"] = None

                            if not spec.get("x"):
                                spec["x"] = (
                                    plan.dimensions[0]
                                    if plan.dimensions
                                    else (df.columns[0] if len(df.columns) else None)
                                )
                            if not spec.get("y"):
                                y0 = plan.metrics[0]["name"] if (plan.metrics and isinstance(plan.metrics[0], dict)) else None
                                if y0 and y0 in df.columns:
                                    spec["y"] = y0
                                else:
                                    xcol = spec.get("x")
                                    numeric_cols = [
                                        c for c in df.columns
                                        if c != xcol and pd.api.types.is_numeric_dtype(df[c])
                                    ]
                                    if numeric_cols:
                                        spec["y"] = numeric_cols[0]
                                    elif len(df.columns) > 1:
                                        spec["y"] = df.columns[1]

                        if spec_type == "pie":
                            if spec.get("names") and spec.get("names") not in df.columns:
                                spec["names"] = None
                            if spec.get("values") and spec.get("values") not in df.columns:
                                spec["values"] = None
                            if not spec.get("names") and plan.dimensions:
                                spec["names"] = plan.dimensions[0]
                            if not spec.get("values") and plan.metrics:
                                y0 = plan.metrics[0]["name"]
                                if y0 in df.columns:
                                    spec["values"] = y0

                        if spec_type == "kpi":
                            if spec.get("value") and spec.get("value") not in df.columns:
                                spec["value"] = None
                            if not spec.get("value") and plan.metrics:
                                cand = plan.metrics[0].get("name")
                                if cand in df.columns:
                                    spec["value"] = cand
                            spec.setdefault("reduce", "sum")

                        return spec

                    charts = list(plan.charts or [])
                    if not charts and plan.chart:
                        charts = [plan.chart]
                    if not charts:
                        charts = [{"type": "bar"}]

                    ql = question.lower()
                    if "kpi" in ql:
                        has_kpi = any(
                            str(c.get("type", "")).lower() == "kpi"
                            for c in charts
                            if isinstance(c, dict)
                        )
                        if not has_kpi:
                            charts = ([{"type": "kpi"}] + charts)

                    figures = []
                    try:
                        for c in charts:
                            spec = _repair_spec(dict(c or {}))
                            df_use = df

                            # Special handling: pie charts without a categorical dimension.
                            # If the planner returns multiple metric columns and a single-row df,
                            # we can pivot metrics into (Category, Value) for a valid pie.
                            if (spec.get("type") or "").lower() == "pie":
                                if (not spec.get("names") or not spec.get("values")) and not plan.dimensions:
                                    metric_cols = [
                                        m.get("name") for m in (plan.metrics or [])
                                        if isinstance(m, dict) and m.get("name") in df.columns
                                    ]
                                    if len(metric_cols) >= 2 and len(df) == 1:
                                        df_use = pd.DataFrame(
                                            {
                                                "Category": metric_cols,
                                                "Value": [df.iloc[0][mc] for mc in metric_cols],
                                            }
                                        )
                                        spec["names"] = "Category"
                                        spec["values"] = "Value"

                            fig = build_figure(df_use, spec)
                            figures.append(fig)
                        primary = figures[0] if figures else None
                        return AgentResult(
                            ok=True,
                            message="OK",
                            plan=last_raw_plan,
                            confidence=confidence,
                            df=df,
                            figure=primary,
                            figures=figures,
                        )
                    except PlotlyRenderError:
                        # Fallback to table without retrying further (chart issue isn't data/planning)
                        return AgentResult(
                            ok=True,
                            message="OK (fallback table)",
                            plan=last_raw_plan,
                            confidence=confidence,
                            df=df,
                            figure=None,
                            figures=None,
                        )

                # --- Report success ---
                return AgentResult(ok=True, message="OK", plan=last_raw_plan, confidence=confidence, df=df)

            except SchemaValidationError as e:
                feedback = (
                    f"Schema validation failed: {str(e)}. "
                    "Do NOT use unknown tables/columns. Use only the columns shown in the provided schema context. "
                    "If you need a derived field, compute it using supported functions or omit it."
                )
                log.warning(
                    "Planning/validation failed; retrying | question=%s | error=%s",
                    question,
                    str(e),
                    extra={
                        "question": question,
                        "mode_hint": mode_hint,
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                continue

            except RetrievalError as e:
                feedback = (
                    f"Schema retrieval failed: {str(e)}. "
                    "Use only the provided schema context and avoid uncommon/ambiguous business aliases."
                )
                log.warning(
                    "Retrieval failed; retrying | question=%s | error=%s",
                    question,
                    str(e),
                    extra={
                        "question": question,
                        "mode_hint": mode_hint,
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                continue

            except Exception as e:
                # Generic failure: encourage a minimal one-table plan on the next attempt.
                feedback = (
                    f"Execution failed: {str(e)}. "
                    "Generate a simpler plan: use one table, one metric, one dimension, no joins, no complex expressions."
                )
                log.warning(
                    "Attempt failed; retrying | question=%s | error=%s",
                    question,
                    str(e),
                    extra={
                        "question": question,
                        "mode_hint": mode_hint,
                        "attempt": attempt,
                        "max_attempts": MAX_ITERS,
                        "error": str(e),
                    },
                    exc_info=True,
                )
                continue

        # All attempts exhausted
        return AgentResult(ok=False, message=INSUFFICIENT, plan=last_raw_plan, confidence=last_confidence)