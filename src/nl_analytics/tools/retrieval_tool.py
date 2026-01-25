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

    # Keep confidence grounded on schema objects only (tables/joins), even if the vector index also contains data-row chunks.
    schema_chunks = [c for c in chunks if (c.metadata or {}).get("kind") in ("table", "join")]
    top = (schema_chunks or chunks)[:3]
    confidence = sum(c.score for c in top) / max(1, len(top))

    # Log full top-k retrieval details for debugging (which chunks were retrieved, scores, and a short preview).
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        md = c.metadata or {}
        kind = md.get("kind", "?")

        if kind == "table":
            ref = f'table={md.get("table")} chunk={md.get("chunk")}'
        elif kind == "join":
            ref = f'join={md.get("left")}->{md.get("right")}'
        else:
            # fallback to a minimal metadata reference
            ref = f'meta_keys={",".join(sorted(md.keys()))}' if md else "meta=None"

        raw_text = c.text or ""
        preview = " ".join(raw_text.split())[:180]
        try:
            score_str = f"{float(c.score):.4f}"
        except Exception:
            score_str = str(c.score)

        lines.append(f"{i:02d}. score={score_str} {kind} {ref} | {preview}")

    log.info(
        "RETRIEVED TOPK CHUNKS | top_k=%d | confidence=%.4f\n%s",
        top_k,
        round(confidence, 4),
        "\n".join(lines),
    )

    return RetrievalResult(chunks=chunks, confidence=confidence)
