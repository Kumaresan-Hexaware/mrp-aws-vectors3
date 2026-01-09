from __future__ import annotations
from typing import List
from dataclasses import dataclass

from nl_analytics.rag.vector_store import VectorStore, RetrievedChunk
from nl_analytics.logging.logger import get_logger

log = get_logger("tools.retrieval")

@dataclass(frozen=True)
class RetrievalResult:
    chunks: List[RetrievedChunk]
    confidence: float

def retrieve_schema_chunks(store: VectorStore, question: str, top_k: int) -> RetrievalResult:
    chunks = store.query(question, top_k=top_k)
    top = chunks[:3]
    confidence = sum(c.score for c in top) / max(1, len(top))
    log.info("Retrieved chunks", extra={"top_k": top_k, "confidence": round(confidence, 4)})
    return RetrievalResult(chunks=chunks, confidence=confidence)
