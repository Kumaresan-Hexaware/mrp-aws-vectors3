from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

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

            # ✅ IMPORTANT: parameter name must be `input` (not texts)
            def __call__(self, input):
                # Chroma passes a list[str] typically; handle str too
                if isinstance(input, str):
                    input = [input]
                return self.br.embed(list(input))

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

        self._ensure_schema_index()

    def _ensure_schema_index(self) -> None:
        schema_ctx = build_schema_context(self.registry)
        docs, metas, ids = [], [], []

        for tname, t in schema_ctx.tables.items():
            text = f"TABLE {tname}: {t['description']}\n"
            for c, cinfo in t["columns"].items():
                text += f"- {c} ({cinfo['type']}): {cinfo['description']}\n"
            docs.append(text)
            metas.append({"kind": "table", "table": tname})
            ids.append(f"table::{tname}")

        for i, j in enumerate(schema_ctx.joins):
            text = f"JOIN RULE: {j['left_table']} -> {j['right_table']} ON {j['left_keys']} = {j['right_keys']} TYPE {j['join_type']}"
            docs.append(text)
            metas.append({"kind": "join", "left": j["left_table"], "right": j["right_table"]})
            ids.append(f"join::{i}")

        self.store.upsert(ids=ids, texts=docs, metadatas=metas)

    def run(self, session: DataSession, question: str, mode_hint: str) -> AgentResult:
        question = (question or "").strip()
        if not question:
            return AgentResult(ok=False, message=INSUFFICIENT, confidence=0.0)

        try:
            retrieval = retrieve_schema_chunks(self.store, question, top_k=self.settings.rag_top_k)
        except RetrievalError:
            return AgentResult(ok=False, message=INSUFFICIENT, confidence=0.0)

        data_avail = 1.0 if session.tables else 0.0
        confidence = float(retrieval.confidence * data_avail)

        log.info("Confidence computed", extra={"confidence": round(confidence, 4)})

        if self.settings.refuse_below_confidence and confidence < self.settings.agent_min_confidence:
            log.info("Refused due to low confidence", extra={"confidence": confidence})
            return AgentResult(ok=False, message=INSUFFICIENT, confidence=confidence)

        sys = (
            "You are a query planning agent. You must ONLY propose plans using tables/columns that exist in the schema registry. "
            "Never invent join keys; joins must follow the registry. Output JSON only."
        )
        grounding = "\n\n".join([f"[score={c.score:.3f}] {c.text}" for c in retrieval.chunks[: self.settings.rag_top_k]])
        user = f"Mode hint: {mode_hint}\n\nRelevant schema:\n{grounding}\n\nQuestion: {question}"

        try:
            raw_plan = self.bedrock.chat_json(sys, user, PLAN_SCHEMA)
            if mode_hint in ("report", "dashboard"):
                raw_plan["mode"] = mode_hint
            plan = validate_plan(self.registry, raw_plan)
        except (SchemaValidationError, AgentExecutionError, Exception) as e:
            log.warning("Planning/validation failed", extra={"error": str(e)})
            return AgentResult(ok=False, message=INSUFFICIENT, confidence=confidence)

        try:
            df = execute_plan(session, plan)
            if df is None or df.empty:
                return AgentResult(ok=False, message=INSUFFICIENT, plan=raw_plan, confidence=confidence, df=df)
        except Exception as e:
            log.warning("Execution failed", extra={"error": str(e)})
            return AgentResult(ok=False, message=INSUFFICIENT, plan=raw_plan, confidence=confidence)

        if plan.mode == "dashboard":
            spec = plan.chart or {"type": "bar", "x": (plan.dimensions[0] if plan.dimensions else df.columns[0]), "y": plan.metrics[0]["name"]}
            try:
                fig = build_figure(df, spec)
                return AgentResult(ok=True, message="OK", plan=raw_plan, confidence=confidence, df=df, figure=fig)
            except PlotlyRenderError:
                return AgentResult(ok=True, message="OK (fallback table)", plan=raw_plan, confidence=confidence, df=df, figure=None)

        return AgentResult(ok=True, message="OK", plan=raw_plan, confidence=confidence, df=df)
