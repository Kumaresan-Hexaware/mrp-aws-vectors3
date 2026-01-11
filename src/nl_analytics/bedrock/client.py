from __future__ import annotations
import json
import time
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from nl_analytics.bedrock.circuit import CircuitBreaker
from nl_analytics.logging.logger import get_logger
from nl_analytics.exceptions.errors import AgentExecutionError

import re
from typing import Any, Dict

log = get_logger("bedrock.client")

def _sleep_backoff(attempt: int) -> None:
    base = 0.5 * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.2)
    time.sleep(min(6.0, base + jitter))

@dataclass
class BedrockConfig:
    region: str
    chat_model_id: str
    embed_model_id: str
    max_tokens: int = 1200
    temperature: float = 0.1
    use_mock: bool = False

class BedrockClient:
    def __init__(self, cfg: BedrockConfig):
        self.cfg = cfg
        self.cb = CircuitBreaker()
        if not cfg.use_mock:
            import boto3
            self.br = boto3.client("bedrock-runtime", region_name=cfg.region)


    def embed(self, texts: List[str], *, dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts via Amazon Bedrock.

        Parameters
        ----------
        texts:
            List of strings to embed.
        dimensions:
            Optional embedding dimension hint. Some Bedrock embedding models (e.g., Titan Text Embeddings V2)
            support requesting a specific output dimension. If the underlying model does not support this
            parameter, it will be ignored.
        """
        if self.cfg.use_mock:
            dim = int(dimensions) if dimensions else 512
            return [self._mock_embed(t, dimensions=dim) for t in texts]

        if not self.cb.allow():
            raise AgentExecutionError("Bedrock circuit breaker is open (embedding).")

        model_id = (self.cfg.embed_model_id or "").lower()
        supports_dimensions = ("titan-embed-text-v2" in model_id)

        last_err = None
        for attempt in range(1, 4):
            try:
                body: Dict[str, Any] = {"inputText": None}
                if dimensions is not None and supports_dimensions:
                    body["dimensions"] = int(dimensions)

                vectors: List[List[float]] = []
                for t in texts:
                    body["inputText"] = t
                    resp = self.br.invoke_model(
                        modelId=self.cfg.embed_model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )
                    payload = json.loads(resp["body"].read().decode("utf-8"))
                    emb = payload.get("embedding") or payload.get("vector") or payload.get("embeddings")
                    if emb is None:
                        raise AgentExecutionError(f"Embedding response missing 'embedding' field: keys={list(payload.keys())}")
                    vectors.append(emb)
                self.cb.record_success()
                return vectors
            except Exception as e:
                last_err = e
                log.warning(
                    "Bedrock embed failed",
                    extra={
                        "attempt": attempt,
                        "error": str(e),
                        "model_id": self.cfg.embed_model_id,
                        "dimensions": dimensions,
                    },
                )
                self.cb.record_failure()
                _sleep_backoff(attempt)
        raise AgentExecutionError("Bedrock embedding call failed after retries.") from last_err


    def _extract_claude_text_from_messages_api(self,payload: Dict[str, Any]) -> str:
        """
        Claude 3 Bedrock Messages API returns:
          {"content":[{"type":"text","text":"..."}], ...}
        """
        parts = []
        for c in payload.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "".join(parts).strip()



    def _extract_first_json_object(self,text: str) -> Dict[str, Any]:
        """
        Robustly extract JSON object from a model response that may include prose.
        Supports:
          - prose + JSON
          - ```json ... ```
        """
        _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        if not text or not text.strip():
            raise ValueError("LLM returned empty text; cannot parse JSON.")

        t = text.strip()

        # If fenced code block exists, prefer it.
        m = _JSON_FENCE_RE.search(t)
        if m:
            t = m.group(1).strip()

        # If it is already pure JSON, parse directly.
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)

        # Otherwise, take substring between first '{' and last '}'.
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not locate JSON object in LLM text. Head: {t[:200]!r}")

        candidate = t[start: end + 1].strip()
        return json.loads(candidate)

    def chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.use_mock:
            return self._mock_plan(user)

        if not self.cb.allow():
            raise AgentExecutionError("Bedrock circuit breaker is open (chat).")

        last_err = None
        for attempt in range(1, 4):
            try:
                prompt = (
                    f"{system}\n\n"
                    f"Return ONLY JSON that matches this schema:\n{json.dumps(schema)}\n\n"
                    f"{user}"
                )
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.cfg.max_tokens,
                    "temperature": self.cfg.temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                }
                resp = self.br.invoke_model(
                    modelId=self.cfg.chat_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body).encode("utf-8")
                )

                raw = json.loads(resp["body"].read())
                model_text = self._extract_claude_text_from_messages_api(raw)
                log.info("Claude text head: %r", model_text[:300])

                payload = json.loads(resp["body"].read().decode("utf-8"))
                text = payload.get("completion") or payload.get("generation") or payload.get("outputText") or ""
                out = json.loads(text.strip())
                self.cb.record_success()
                return out
            except Exception as e:
                last_err = e
                log.warning("Bedrock chat failed", extra={"attempt": attempt, "error": str(e)})
                self.cb.record_failure()
                _sleep_backoff(attempt)
                log.warning("Bedrock chat failed", exc_info=True)
                raise

        raise AgentExecutionError("Bedrock chat call failed after retries.") from last_err

    # ---- mocks ----
    def _mock_embed(self, text: str, *, dimensions: int = 512) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        dim = max(1, int(dimensions))
        for i in range(dim):
            b = h[i % len(h)]
            vec.append((b / 255.0) * 2 - 1)
        return vec

    def _mock_plan(self, user: str) -> Dict[str, Any]:
        """Heuristic mock planner.

        When USE_MOCK_BEDROCK=1, we still want the app to work with *your* schema
        (pvr/krd/pnl...), not the sample "sales" table.

        The Streamlit orchestrator passes a prompt that already contains:
        - Mode hint
        - Top retrieved schema snippets (TABLE ... and JOIN RULE ...)
        - The user's question

        This function extracts table + column candidates from that prompt and
        produces a plan that passes validate_plan().
        """

        import re

        text = user or ""
        u = text.lower()
        mode = "dashboard" if ("mode hint: dashboard" in u or "dashboard" in u or "chart" in u or "kpi" in u) else "report"

        # --- extract tables from grounding ---
        tables = re.findall(r"\bTABLE\s+([A-Za-z0-9_]+)\s*:", text)
        tables = [t.strip() for t in tables if t.strip()]
        if not tables:
            # fallback: try to spot pvr/krd/pnl tokens
            for token in ("pvr", "krd", "pnl", "pnr"):
                if token in u:
                    tables = [token]
                    break
        table = tables[0] if tables else "pvr00400"

        # --- extract columns (name + type) from grounding ---
        cols = re.findall(r"-\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)\s*:", text)
        col_types = {c: (t or "").strip().lower() for c, t in cols}
        col_names = list(col_types.keys())

        def pick(cols_list, pred, prefer_keywords=None):
            prefer_keywords = prefer_keywords or []
            scored = []
            for c in cols_list:
                if not pred(c):
                    continue
                score = 0
                lc = c.lower()
                for kw in prefer_keywords:
                    if kw in lc:
                        score += 10
                if lc in u:
                    score += 5
                scored.append((score, c))
            scored.sort(reverse=True)
            return scored[0][1] if scored else None

        date_cols = [c for c, t in col_types.items() if "date" in t]
        num_cols = [c for c, t in col_types.items() if any(x in t for x in ("float", "double", "decimal", "int"))]
        str_cols = [c for c, t in col_types.items() if "string" in t]

        # --- choose metric ---
        if "count" in u or "how many" in u:
            metric = {"name": "row_count", "expr": "count(*)"}
        else:
            num = pick(
                num_cols,
                lambda _c: True,
                prefer_keywords=["upb", "amount", "balance", "price", "oas", "duration", "analytic", "return"],
            )
            if num is None and num_cols:
                num = num_cols[0]
            if num:
                metric = {"name": f"total_{num}", "expr": f"sum({num})"}
            else:
                metric = {"name": "row_count", "expr": "count(*)"}

        # --- choose dimension(s) ---
        dims = []
        # try to honor "by <col>" pattern
        for c in col_names:
            if f"by {c.lower()}" in u:
                dims = [c]
                break

        if not dims:
            if mode == "dashboard" and date_cols:
                dims = [date_cols[0]]
            elif str_cols:
                dims = [str_cols[0]]
            elif col_names:
                dims = [col_names[0]]

        plan: Dict[str, Any] = {
            "mode": mode,
            "tables": [table],
            "metrics": [metric],
            "dimensions": dims,
            "filters": [],
            "limit": 5000 if mode == "dashboard" else 200,
            "sort": [{"by": metric["name"], "desc": True}],
        }

        if mode == "dashboard":
            x = dims[0] if dims else (date_cols[0] if date_cols else (str_cols[0] if str_cols else None))
            plan["chart"] = {
                "type": "line" if (x in date_cols) else "bar",
                "x": x,
                "y": metric["name"],
                "color": None,
            }

        return plan
