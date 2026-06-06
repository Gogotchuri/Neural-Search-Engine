"""Tests for the evaluation metrics and harness.

Run with:  PYTHONPATH=src pytest tests/test_evaluation.py
"""

import json
import math

import pytest

from neural_search.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from neural_search.evaluation.eval_set import EvalQuery, load_eval_set
from neural_search.evaluation.harness import compare, evaluate, format_table


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
def test_recall_single_relevant_hit_and_miss():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], k=3) == 1.0  # found within top 3
    assert recall_at_k(ranked, ["c"], k=2) == 0.0  # outside top 2


def test_recall_multiple_relevant_is_a_fraction():
    ranked = ["a", "b", "c", "d"]
    # 2 of 3 relevant chunks land in the top 3.
    assert recall_at_k(ranked, ["a", "c", "z"], k=3) == pytest.approx(2 / 3)


def test_recall_ignores_duplicate_retrievals():
    # A retriever that somehow returns "a" twice must not score recall > 1.
    assert recall_at_k(["a", "a", "b"], ["a"], k=3) == 1.0


def test_recall_rejects_bad_inputs():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


# --------------------------------------------------------------------------- #
# reciprocal_rank
# --------------------------------------------------------------------------- #
def test_reciprocal_rank_uses_first_relevant():
    ranked = ["x", "a", "b"]  # first relevant at rank 2
    assert reciprocal_rank(ranked, ["a", "b"], k=10) == pytest.approx(0.5)


def test_reciprocal_rank_zero_when_outside_k():
    assert reciprocal_rank(["x", "y", "a"], ["a"], k=2) == 0.0


def test_reciprocal_rank_perfect():
    assert reciprocal_rank(["a", "x"], ["a"], k=10) == 1.0


# --------------------------------------------------------------------------- #
# ndcg_at_k
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_ranking_is_one():
    ranked = ["a", "b", "x", "y"]
    assert ndcg_at_k(ranked, ["a", "b"], k=4) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_two():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    ranked = ["x", "a", "y"]
    assert ndcg_at_k(ranked, ["a"], k=3) == pytest.approx(1 / math.log2(3))


def test_ndcg_zero_when_no_relevant_retrieved():
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0


# --------------------------------------------------------------------------- #
# eval_set loading + validation
# --------------------------------------------------------------------------- #
def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_eval_set_parses_entries(tmp_path):
    path = _write(
        tmp_path / "eval.json",
        [{"query": "what is beam search", "relevant_ids": ["ch9-0042"]}],
    )
    queries = load_eval_set(path)
    assert queries == [EvalQuery(query="what is beam search", relevant_ids=("ch9-0042",))]


def test_load_eval_set_validates_against_corpus(tmp_path):
    path = _write(
        tmp_path / "eval.json",
        [{"query": "q", "relevant_ids": ["ch1-0000", "ghost-9999"]}],
    )
    with pytest.raises(ValueError, match="ghost-9999"):
        load_eval_set(path, corpus_ids={"ch1-0000"})


def test_load_eval_set_rejects_empty_relevant_ids(tmp_path):
    path = _write(tmp_path / "eval.json", [{"query": "q", "relevant_ids": []}])
    with pytest.raises(ValueError, match="relevant_ids"):
        load_eval_set(path)


def test_load_eval_set_rejects_string_relevant_ids(tmp_path):
    # A bare string would be exploded into characters by tuple() - must be rejected.
    path = _write(tmp_path / "eval.json", [{"query": "q", "relevant_ids": "ch2-0000"}])
    with pytest.raises(ValueError, match="list of strings"):
        load_eval_set(path)


def test_load_eval_set_rejects_non_string_query(tmp_path):
    path = _write(tmp_path / "eval.json", [{"query": 42, "relevant_ids": ["ch2-0000"]}])
    with pytest.raises(ValueError, match="query"):
        load_eval_set(path)


# --------------------------------------------------------------------------- #
# harness aggregation
# --------------------------------------------------------------------------- #
class _StubRetriever:
    """Returns a fixed ranking per query, ignoring k beyond truncation."""

    def __init__(self, rankings):
        self._rankings = rankings

    def retrieve(self, query, k=10):
        return [{"id": cid} for cid in self._rankings[query][:k]]


def test_evaluate_averages_over_queries():
    eval_set = [
        EvalQuery("q1", ("a",)),  # a at rank 1 -> RR 1.0, recall@1 1.0
        EvalQuery("q2", ("b",)),  # b at rank 2 -> RR 0.5, recall@1 0.0
    ]
    retriever = _StubRetriever({"q1": ["a", "z"], "q2": ["z", "b"]})

    scores = evaluate(retriever, eval_set, recall_ks=(1,), rank_k=10)

    assert scores["Recall@1"] == pytest.approx(0.5)
    assert scores["MRR@10"] == pytest.approx(0.75)


def test_compare_and_format_table_cover_all_systems():
    eval_set = [EvalQuery("q1", ("a",))]
    retrievers = {
        "perfect": _StubRetriever({"q1": ["a"]}),
        "miss": _StubRetriever({"q1": ["z"]}),
    }
    results = compare(retrievers, eval_set, recall_ks=(1,), rank_k=10)

    assert results["perfect"]["Recall@1"] == 1.0
    assert results["miss"]["Recall@1"] == 0.0

    table = format_table(results)
    assert "perfect" in table and "miss" in table
    assert "Recall@1" in table
