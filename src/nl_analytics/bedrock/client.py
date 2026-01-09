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
        u = user.lower()
        if "dashboard" in u or "chart" in u or "kpi" in u:
            return {
                "mode": "dashboard",
                "tables": ["sales"],
                "metrics": [{"name": "total_sales", "expr": "sum(sales)"}],
                "dimensions": ["order_date"],
                "filters": [],
                "limit": 5000,
                "chart": {"type": "line", "x": "order_date", "y": "total_sales", "color": None},
            }
        return {
            "mode": "report",
            "tables": ["sales"],
            "metrics": [{"name": "total_sales", "expr": "sum(sales)"}],
            "dimensions": ["region"],
            "filters": [],
            "limit": 100,
            "sort": [{"by": "total_sales", "desc": True}],
        }
