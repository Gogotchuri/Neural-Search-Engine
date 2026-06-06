"""Ranking metrics for retrieval evaluation.

Each function scores one query from ``ranked_ids`` (retrieved chunk ids, best
first) and ``relevant_ids`` (the labelled relevant ids). They're pure, so the
harness averages them over the eval set and tests pin them to known values.
Relevance is binary: every relevant chunk counts as gain 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant chunks found in the top k: ``|relevant ∩ top-k| / |relevant|``.

    With a single relevant chunk this is just hit@k (1.0 if found, else 0.0).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must be non-empty")

    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """``1 / rank`` of the first relevant chunk in the top k, or 0.0 if none.

    Averaged over queries this gives MRR@k, which rewards ranking a good answer high.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must be non-empty")

    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of a gain list already in ranked order."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """DCG of the ranking divided by the ideal DCG (binary relevance), in [0, 1]."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must be non-empty")

    gains = [1.0 if chunk_id in relevant else 0.0 for chunk_id in ranked_ids[:k]]
    dcg = _dcg(gains)

    ideal_hits = min(len(relevant), k)  # every relevant chunk that fits, at the top
    idcg = _dcg([1.0] * ideal_hits)

    return dcg / idcg if idcg > 0 else 0.0
