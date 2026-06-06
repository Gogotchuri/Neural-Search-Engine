"""Loading and validation for the hand-built evaluation set.

The eval set is a JSON list following the project's interface contract:

    [{"query": "what is beam search", "relevant_ids": ["ch9-0042", "ch13-0007"]}, ...]

``relevant_ids`` reference chunks by their stable string ``id``, not integer
position. Validating them against the live corpus catches typo'd or stale ids
that would otherwise silently never match.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuery:
    """One labelled query: the text and the set of relevant chunk ids."""

    query: str
    relevant_ids: tuple[str, ...]


def load_corpus_ids(chunks_path: str | Path) -> set[str]:
    """Collect the set of chunk ids from a chunks.jsonl file (for validation)."""
    ids: set[str] = set()
    with open(chunks_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_eval_set(
    path: str | Path,
    corpus_ids: Iterable[str] | None = None,
) -> list[EvalQuery]:
    """Load and validate the eval set.

    Args:
        path:       Path to eval_set.json.
        corpus_ids: If provided, every ``relevant_id`` must be one of these;
                    unknown ids raise a ValueError naming the offenders. Pass
                    the output of ``load_corpus_ids(chunks_path)``.

    Raises:
        ValueError: on malformed entries or relevant_ids missing from the corpus.
    """
    with open(path, "r", encoding="utf-8") as in_file:
        raw = json.load(in_file)

    if not isinstance(raw, list):
        raise ValueError(f"Eval set must be a JSON list, got {type(raw).__name__}")

    queries: list[EvalQuery] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {index} must be an object, got {type(entry).__name__}")

        query = entry.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Entry {index}: 'query' must be a non-empty string")
        query = query.strip()

        # A bare string would be exploded into characters by tuple(); require a list.
        relevant_ids = entry.get("relevant_ids", [])
        if not isinstance(relevant_ids, list) or not all(
            isinstance(cid, str) for cid in relevant_ids
        ):
            raise ValueError(
                f"Entry {index} ('{query}'): 'relevant_ids' must be a list of strings"
            )
        if not relevant_ids:
            raise ValueError(f"Entry {index} ('{query}'): 'relevant_ids' must be non-empty")

        queries.append(EvalQuery(query=query, relevant_ids=tuple(relevant_ids)))

    if corpus_ids is not None:
        known = set(corpus_ids)
        unknown = {
            cid
            for q in queries
            for cid in q.relevant_ids
            if cid not in known
        }
        if unknown:
            raise ValueError(
                f"Eval set references {len(unknown)} chunk id(s) not in the corpus: "
                f"{sorted(unknown)}. Fix the ids or rebuild the corpus."
            )

    return queries
