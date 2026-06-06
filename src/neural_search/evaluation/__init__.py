"""Retrieval evaluation: ranking metrics, eval-set loading, and a harness."""

from neural_search.evaluation.eval_set import (
    EvalQuery,
    load_corpus_ids,
    load_eval_set,
)
from neural_search.evaluation.harness import (
    DEFAULT_RANK_K,
    DEFAULT_RECALL_KS,
    compare,
    evaluate,
    format_table,
    metric_names,
)
from neural_search.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvalQuery",
    "load_corpus_ids",
    "load_eval_set",
    "evaluate",
    "compare",
    "format_table",
    "metric_names",
    "DEFAULT_RECALL_KS",
    "DEFAULT_RANK_K",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
]
