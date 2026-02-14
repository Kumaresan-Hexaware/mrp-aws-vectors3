from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import re

from nl_analytics.rag.vector_store import VectorStore, RetrievedChunk
from nl_analytics.logging.logger import get_logger

log = get_logger("tools.retrieval")


@dataclass(frozen=True)
class RetrievalResult:
    chunks: List[RetrievedChunk]
    confidence: float


def _tables_mentioned(question: str) -> set[str]:
    q = (question or "").lower()
    # crude: table-like tokens such as pvr00600, krd00200 etc
    return set(re.findall(r"\b[a-z]{3}\d{5}\b", q))


def retrieve_planning_chunks(
    store: VectorStore,
    question: str,
    *,
    top_k: int,
    oversample_factor: int = 5,
    schema_kinds: Sequence[str] = ("table", "join", "column"),
    schema_only: bool = True,
    enable_rerank: bool = True,
    candidate_columns: Optional[Sequence[Tuple[str, str]]] = None,
) -> RetrievalResult:
    """Retrieve chunks for *planning*.

    Key ideas:
    - Oversample then rerank. Vector similarity alone is noisy with very wide schemas.
    - Prefer schema chunks (table/join/column) over data rows.
    - Boost column chunks that match ColumnResolver candidates.
    """

    k = max(1, int(top_k))
    over = max(k, k * max(1, int(oversample_factor)))

    # Base semantic retrieval
    raw = store.query(question, top_k=over)

    # IMPORTANT:
    # Reranking can only boost candidates that are already present in the pool.
    # In practice, "Delivery Price" and similar business terms can sometimes
    # fail to appear in the initial top-N semantic results (especially with
    # wide schemas and many price-like columns).
    #
    # To make candidate boosting reliable, we *seed* the pool with a few extra
    # targeted queries for the candidate columns (cheap, small top_k).
    extra: List[RetrievedChunk] = []
    cand_list = list(candidate_columns or [])
    if cand_list:
        # Limit to avoid excessive embedding calls
        for (t, c) in cand_list[:10]:
            try:
                # Canonical patterns that match how column chunks are written in the index.
                q2 = f"COLUMN {t}.{c}"
                extra.extend(store.query(q2, top_k=5))
            except Exception:
                # Best-effort only
                continue

    # Merge + de-dup (by stable meta signature + first 120 chars)
    pool_all = list(raw) + list(extra)
    seen_sig = set()
    merged: List[RetrievedChunk] = []
    for ch in pool_all:
        md = ch.metadata or {}
        sig = (
            md.get("kind"),
            md.get("table") or md.get("left"),
            md.get("column") or md.get("right"),
            (ch.text or "")[:120],
        )
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        merged.append(ch)

    # Use the merged pool (base retrieval + candidate-seeded retrieval)
    allowed_kinds = set(str(x).strip() for x in (schema_kinds or []))
    if schema_only and allowed_kinds:
        pool = [c for c in merged if (c.metadata or {}).get("kind") in allowed_kinds]
        if not pool:
            pool = merged
    else:
        pool = merged

    cand_set = set(candidate_columns or [])
    mentioned_tables = _tables_mentioned(question)

    def _rank(c: RetrievedChunk) -> float:
        base = float(getattr(c, "score", 0.0) or 0.0)
        if not enable_rerank:
            return base

        md = c.metadata or {}
        kind = md.get("kind")
        bonus = 0.0

        # Kind-based prior: joins and tables help table selection; columns help exact matching.
        if kind == "join":
            bonus += 0.12
        elif kind == "table":
            bonus += 0.08
        elif kind == "column":
            bonus += 0.10

        # Boost if question explicitly mentions a table.
        t = str(md.get("table") or md.get("left") or "").lower()
        if mentioned_tables and t and t in mentioned_tables:
            bonus += 0.10

        # Strong boost if this is a candidate column.
        if kind == "column":
            table = str(md.get("table") or "")
            col = str(md.get("column") or "")
            if (table, col) in cand_set:
                bonus += 0.25

        return base + bonus

    ranked = sorted(pool, key=_rank, reverse=True)
    chosen = ranked[:k]

    # Confidence: combine similarity + schema ratio.
    # similarity: average of top 3 chosen
    top3 = chosen[:3] if chosen else []
    sim = (sum(float(c.score or 0.0) for c in top3) / max(1, len(top3))) if top3 else 0.0
    schema = sum(1 for c in chosen if (c.metadata or {}).get("kind") in allowed_kinds) / max(1, len(chosen))
    confidence = 0.70 * sim + 0.30 * schema

    # Log full details for debugging.
    lines: List[str] = []
    for i, c in enumerate(chosen, start=1):
        md = c.metadata or {}
        kind = md.get("kind", "?")
        if kind == "table":
            ref = f"table={md.get('table')}"
        elif kind == "join":
            ref = f"join={md.get('left')}->{md.get('right')}"
        elif kind == "column":
            ref = f"col={md.get('table')}.{md.get('column')}"
        else:
            ref = f"meta_keys={','.join(sorted(md.keys()))}" if md else "meta=None"

        preview = " ".join((c.text or "").split())[:180]
        lines.append(f"{i:02d}. score={float(c.score):.4f} kind={kind} {ref} | {preview}")

    log.info(
        "RETRIEVED TOPK PLANNING CHUNKS | top_k=%d | oversample=%d | confidence=%.4f\n%s",
        k,
        over,
        round(float(confidence), 4),
        "\n".join(lines),
    )

    return RetrievalResult(chunks=chosen, confidence=float(confidence))


def retrieve_schema_chunks(
    store: VectorStore,
    question: str,
    top_k: int,
    *,
    candidate_columns: Optional[Sequence[Tuple[str, str]]] = None,
) -> RetrievalResult:
    """Backward-compatible wrapper (legacy name used by orchestrator).

    We keep this wrapper so older code paths can call `retrieve_schema_chunks`,
    while still allowing the orchestrator to pass registry-driven candidate
    columns to improve accuracy (e.g., "Delivery Price" -> pvr00400.DeliveryPrice).
    """
    return retrieve_planning_chunks(store, question, top_k=top_k, candidate_columns=candidate_columns)
