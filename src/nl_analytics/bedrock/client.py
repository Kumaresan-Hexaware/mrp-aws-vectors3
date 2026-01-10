from __future__ import annotations
import json
import time
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

from nl_analytics.bedrock.circuit import CircuitBreaker
from nl_analytics.logging.logger import get_logger
from nl_analytics.exceptions.errors import AgentExecutionError

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

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.cfg.use_mock:
            return [self._mock_embed(t) for t in texts]

        if not self.cb.allow():
            raise AgentExecutionError("Bedrock circuit breaker is open (embedding).")

        last_err = None
        for attempt in range(1, 4):
            try:
                body = {"inputText": None}
                vectors = []
                for t in texts:
                    body["inputText"] = t
                    resp = self.br.invoke_model(
                        modelId=self.cfg.embed_model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )
                    payload = json.loads(resp["body"].read().decode("utf-8"))
                    vectors.append(payload["embedding"])
                self.cb.record_success()
                return vectors
            except Exception as e:
                last_err = e
                log.warning("Bedrock embed failed", extra={"attempt": attempt, "error": str(e)})
                self.cb.record_failure()
                _sleep_backoff(attempt)
        raise AgentExecutionError("Bedrock embedding call failed after retries.") from last_err

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
                    "prompt": prompt,
                    "max_tokens_to_sample": self.cfg.max_tokens,
                    "temperature": self.cfg.temperature,
                }
                resp = self.br.invoke_model(
                    modelId=self.cfg.chat_model_id,
                    body=json.dumps(body).encode("utf-8"),
                    accept="application/json",
                    contentType="application/json",
                )
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
        raise AgentExecutionError("Bedrock chat call failed after retries.") from last_err

    # ---- mocks ----
    def _mock_embed(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(512):
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
