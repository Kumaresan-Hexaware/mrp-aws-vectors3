from __future__ import annotations

"""Column resolution helpers.

Why this exists:
- Vector similarity alone often struggles to pick the *exact* right column when tables have 500+ columns.
- The registry already contains strong signals: canonical names + business aliases.

This module builds a deterministic shortlist of candidate columns using:
1) exact alias resolution (registry.resolve_alias_global)
2) lightweight fuzzy matching over registry alias keys

The shortlist is then used to:
- boost schema retrieval (reranking)
- constrain the LLM prompt ("pick from these columns")
"""

from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
import re
from typing import Dict, Iterable, List, Optional, Tuple

from nl_analytics.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class CandidateColumn:
    table: str
    column: str
    score: float
    reason: str


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())


def _ngrams(words: List[str], n: int) -> Iterable[str]:
    for i in range(0, max(0, len(words) - n + 1)):
        yield " ".join(words[i : i + n]).strip()


class ColumnResolver:
    """Registry-driven column resolver.

    This is intentionally lightweight (no extra dependencies) and fast enough for
    interactive usage.
    """

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        # Build a stable list of alias keys for fuzzy matching.
        self._alias_keys = registry.all_alias_keys()

    def suggest(
        self,
        question: str,
        max_candidates: int = 40,
        within_tables: Optional[Iterable[str]] = None,
        fuzzy_cutoff: float = 0.86,
    ) -> List[CandidateColumn]:
        """Return a shortlist of candidate columns for the given question."""
        q = (question or "").strip().lower()
        if not q:
            return []

        # Tokenize to words; keep underscores (common in column names).
        words = re.findall(r"[a-z0-9_]+", q)
        if not words:
            return []

        candidates: Dict[Tuple[str, str], CandidateColumn] = {}

        # 1) Exact alias matches using 1-4 grams.
        seen_phrases = set()
        for n in (4, 3, 2, 1):
            for phrase in _ngrams(words, n):
                if not phrase or phrase in seen_phrases:
                    continue
                seen_phrases.add(phrase)
                matches = self.registry.resolve_alias_global(phrase)
                for t, c in matches:
                    if within_tables is not None and t not in set(within_tables):
                        continue
                    key = (t, c)
                    # Strong signal.
                    cand = CandidateColumn(table=t, column=c, score=1.0, reason=f"alias match: '{phrase}'")
                    if key not in candidates or candidates[key].score < cand.score:
                        candidates[key] = cand

        # 2) Fuzzy match: try to map key phrases to the closest alias keys.
        # We only run fuzzy on a few high-signal phrases to keep it fast.
        fuzzy_phrases: List[str] = []
        for n in (3, 2, 1):
            for phrase in _ngrams(words, n):
                if len(phrase) < 4:
                    continue
                fuzzy_phrases.append(phrase)
            if len(fuzzy_phrases) >= 20:
                break

        for phrase in fuzzy_phrases[:20]:
            p_norm = _norm(phrase)
            if len(p_norm) < 4:
                continue
            # get_close_matches uses SequenceMatcher internally.
            close = get_close_matches(p_norm, self._alias_keys, n=8, cutoff=fuzzy_cutoff)
            for k in close:
                sim = SequenceMatcher(a=p_norm, b=k).ratio()
                # Map fuzzy similarity [0..1] to a useful score range.
                base = 0.60 + 0.40 * sim
                for t, c in self.registry._global_alias.get(k, []):  # internal index, read-only
                    if within_tables is not None and t not in set(within_tables):
                        continue
                    key = (t, c)
                    cand = CandidateColumn(table=t, column=c, score=float(base), reason=f"fuzzy match: '{phrase}'")
                    if key not in candidates or candidates[key].score < cand.score:
                        candidates[key] = cand

        # Sort: strongest first; keep small list.
        out = sorted(candidates.values(), key=lambda x: (x.score, x.table, x.column), reverse=True)
        return out[: max(1, int(max_candidates))]
