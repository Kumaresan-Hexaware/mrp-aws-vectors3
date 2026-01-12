from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nl_analytics.bedrock.circuit import CircuitBreaker
from nl_analytics.exceptions.errors import AgentExecutionError
from nl_analytics.logging.logger import get_logger

log = get_logger("bedrock.client")


def _sleep_backoff(attempt: int) -> None:
    base = 0.5 * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, 0.2)
    time.sleep(min(6.0, base + jitter))


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


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
        self.br = None
        if not cfg.use_mock:
            import boto3

            self.br = boto3.client("bedrock-runtime", region_name=cfg.region)

    # -----------------------------
    # Embeddings
    # -----------------------------
    def embed(self, texts: List[str], *, dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts via Amazon Bedrock.

        - Reads StreamingBody exactly once per request.
        - Supports optional dimensions for models that support it (e.g., Titan v2).
        """
        if self.cfg.use_mock:
            dim = int(dimensions) if dimensions else 512
            return [self._mock_embed(t, dimensions=dim) for t in texts]

        if not self.cb.allow():
            raise AgentExecutionError("Bedrock circuit breaker is open (embedding).")

        if not self.br:
            raise AgentExecutionError("Bedrock runtime client is not initialized.")

        model_id_l = (self.cfg.embed_model_id or "").lower()
        supports_dimensions = ("titan-embed-text-v2" in model_id_l)

        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                vectors: List[List[float]] = []

                for t in texts:
                    body: Dict[str, Any] = {"inputText": t}
                    if dimensions is not None and supports_dimensions:
                        body["dimensions"] = int(dimensions)

                    resp = self.br.invoke_model(
                        modelId=self.cfg.embed_model_id,
                        body=json.dumps(body).encode("utf-8"),
                        accept="application/json",
                        contentType="application/json",
                    )

                    raw_bytes = resp["body"].read()
                    if not raw_bytes:
                        raise AgentExecutionError("Empty embedding response body from Bedrock.")

                    payload = json.loads(raw_bytes.decode("utf-8"))

                    emb = payload.get("embedding") or payload.get("vector") or payload.get("embeddings")
                    if emb is None:
                        raise AgentExecutionError(
                            f"Embedding response missing vector field. keys={list(payload.keys())}"
                        )

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
                    exc_info=True,
                )
                self.cb.record_failure()
                _sleep_backoff(attempt)

        raise AgentExecutionError("Bedrock embedding call failed after retries.") from last_err

    # -----------------------------
    # Chat -> JSON
    # -----------------------------
    def chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Bedrock chat model (Claude Messages API) and return a JSON object.

        Robustness:
          - Reads the streaming body ONCE (StreamingBody is consumable).
          - Extracts Claude text from the Messages API response.
          - Tolerates prose around JSON by extracting the first JSON object.
          - Retries with exponential backoff and circuit breaker support.
        """
        if self.cfg.use_mock:
            return self._mock_plan(user)

        if not self.cb.allow():
            raise AgentExecutionError("Bedrock circuit breaker is open (chat).")

        if not self.br:
            raise AgentExecutionError("Bedrock runtime client is not initialized.")

        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                prompt = (
                    f"{system}\n\n"
                    f"Return ONLY a JSON object that matches this schema (include all required keys):\n"
                    f"{json.dumps(schema)}\n\n"
                    f"{user}"
                )

                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.cfg.max_tokens,
                    "temperature": self.cfg.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }

                resp = self.br.invoke_model(
                    modelId=self.cfg.chat_model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body).encode("utf-8"),
                )

                raw_bytes = resp["body"].read()
                if not raw_bytes:
                    raise AgentExecutionError("Empty Bedrock response body.")

                payload = json.loads(raw_bytes.decode("utf-8"))

                # Claude 3 Messages API shape:
                # {"content":[{"type":"text","text":"..."}], ...}
                if isinstance(payload, dict) and "content" in payload:
                    model_text = self._extract_claude_text_from_messages_api(payload)
                else:
                    # fallback for other shapes
                    model_text = (
                        (payload.get("completion") if isinstance(payload, dict) else None)
                        or (payload.get("generation") if isinstance(payload, dict) else None)
                        or (payload.get("outputText") if isinstance(payload, dict) else None)
                        or ""
                    )

                log.info("Claude text head: %r", (model_text or "")[:300])

                out = self._extract_first_json_object(model_text)
                self.cb.record_success()
                return out

            except Exception as e:
                last_err = e
                log.warning(
                    "Bedrock chat failed",
                    extra={"attempt": attempt, "error": str(e), "model_id": self.cfg.chat_model_id},
                    exc_info=True,
                )
                self.cb.record_failure()
                _sleep_backoff(attempt)

        raise AgentExecutionError("Bedrock chat call failed after retries.") from last_err

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_claude_text_from_messages_api(self, payload: Dict[str, Any]) -> str:
        parts: List[str] = []
        for c in payload.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
        return "".join(parts).strip()

    def _extract_first_json_object(self, text: str) -> Dict[str, Any]:
        """
        Extract the first JSON object from a model response that may include prose.
        - Supports fenced ```json ... ```
        - Repairs trailing commas before } or ]
        - Parses the first balanced {...} region
        """
        if not text or not text.strip():
            raise AgentExecutionError("LLM returned empty text; cannot parse JSON.")

        t = text.strip()

        # Prefer fenced JSON blocks if present
        m = _JSON_FENCE_RE.search(t)
        if m:
            t = m.group(1).strip()

        # Find first '{' then scan for balanced braces (string-aware)
        start = t.find("{")
        if start < 0:
            raise AgentExecutionError(f"No JSON object found in model text. Head: {t[:200]!r}")

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = t[start : i + 1].strip()
                        # remove trailing commas before } or ]
                        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
                        return json.loads(candidate)

        raise AgentExecutionError("Unbalanced JSON braces in model response.")

    # -----------------------------
    # Mocks
    # -----------------------------
    def _mock_embed(self, text: str, *, dimensions: int = 512) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        dim = max(1, int(dimensions))
        vec: List[float] = []
        for i in range(dim):
            b = h[i % len(h)]
            vec.append((b / 255.0) * 2 - 1)
        return vec

    def _mock_plan(self, user: str) -> Dict[str, Any]:
        # Minimal but valid plan for local/mock testing
        return {
            "mode": "report",
            "tables": ["pvr00400"],
            "metrics": [{"name": "total_CurrentUPBAmt", "expr": "SUM(CurrentUPBAmt)"}],
            "dimensions": ["PortfolioID"],
            "filters": [],
            "sort": [{"by": "total_CurrentUPBAmt", "desc": True}],
            "limit": 5000,
        }
