"""Run a Retriever over the eval set and report the metrics.

Works through the ``Retriever`` interface alone, so BM25 and the encoders all
go through the same path and stay directly comparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from neural_search.evaluation.eval_set import EvalQuery
from neural_search.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_RECALL_KS: tuple[int, ...] = (1, 5, 10)
DEFAULT_RANK_K: int = 10


class _Retriever(Protocol):
    def retrieve(self, query: str, k: int = ...) -> Sequence[dict]: ...


def metric_names(
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    rank_k: int = DEFAULT_RANK_K,
) -> list[str]:
    """Ordered metric column names, e.g. Recall@1, Recall@5, ..., MRR@10, nDCG@10."""
    return [f"Recall@{k}" for k in recall_ks] + [f"MRR@{rank_k}", f"nDCG@{rank_k}"]


def evaluate(
    retriever: _Retriever,
    eval_set: Sequence[EvalQuery],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    rank_k: int = DEFAULT_RANK_K,
) -> dict[str, float]:
    """Score the retriever on every query and average each metric.

    Returns a dict keyed by the names from :func:`metric_names`.
    """
    if not eval_set:
        raise ValueError("eval_set is empty - nothing to evaluate")

    # Retrieve once per query, deep enough for the largest cutoff; metrics slice it.
    retrieve_k = max([*recall_ks, rank_k])
    names = metric_names(recall_ks, rank_k)
    totals = {name: 0.0 for name in names}

    for item in eval_set:
        results = retriever.retrieve(item.query, k=retrieve_k)
        ranked_ids = [r["id"] for r in results]

        for k in recall_ks:
            totals[f"Recall@{k}"] += recall_at_k(ranked_ids, item.relevant_ids, k)
        totals[f"MRR@{rank_k}"] += reciprocal_rank(ranked_ids, item.relevant_ids, rank_k)
        totals[f"nDCG@{rank_k}"] += ndcg_at_k(ranked_ids, item.relevant_ids, rank_k)

    n = len(eval_set)
    return {name: total / n for name, total in totals.items()}


def compare(
    retrievers: dict[str, _Retriever],
    eval_set: Sequence[EvalQuery],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    rank_k: int = DEFAULT_RANK_K,
) -> dict[str, dict[str, float]]:
    """Evaluate several named retrievers on the same eval set."""
    return {
        name: evaluate(retriever, eval_set, recall_ks=recall_ks, rank_k=rank_k)
        for name, retriever in retrievers.items()
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Render a ``compare`` result as a fixed-width table, best per column marked ``*``."""
    if not results:
        return "(no retrievers evaluated)"

    metrics = next(iter(results.values())).keys()
    name_width = max(len("Retriever"), *(len(name) for name in results))
    col_width = max(8, *(len(m) for m in metrics)) + 1

    best = {m: max(scores[m] for scores in results.values()) for m in metrics}

    header = "Retriever".ljust(name_width) + "".join(m.rjust(col_width) for m in metrics)
    lines = [header, "-" * len(header)]

    for name, scores in results.items():
        row = name.ljust(name_width)
        for m in metrics:
            mark = "*" if math.isclose(scores[m], best[m]) else " "
            row += f"{scores[m]:.4f}{mark}".rjust(col_width)
        lines.append(row)

    return "\n".join(lines)
