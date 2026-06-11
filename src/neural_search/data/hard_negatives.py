from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from neural_search.data.msmarco import (
    clean_text,
    extract_msmarco_passages,
    iter_msmarco_rows,
    normalize_text,
)
from neural_search.retrieval.bm25 import BM25Retriever


def write_msmarco_bm25_corpus(
    output_path: str | Path,
    split: str = "train",
    config_name: str = "v1.1",
    max_rows: int | None = None,
    streaming: bool = True,
    cache_dir: str | None = None,
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
) -> int:
    """
    Write all candidate MS MARCO passages to a BM25-compatible JSONL corpus.

    The output schema matches BM25Retriever:
        chunk_id, text, chapter, section
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_passages: set[str] = set()
    written = 0
    rows_seen = 0

    with open(output_path, "w", encoding="utf-8") as out_file:
        for row in iter_msmarco_rows(
            split=split,
            config_name=config_name,
            streaming=streaming,
            cache_dir=cache_dir,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        ):
            all_passages, _ = extract_msmarco_passages(row)

            for passage in all_passages:
                normalized = normalize_text(passage)

                if normalized in seen_passages:
                    continue

                seen_passages.add(normalized)

                record = {
                    "id": f"msmarco-{written}",
                    "chunk_id": written,
                    "text": passage,
                    "chapter": "MS MARCO",
                    "section": "candidate-passages",
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            rows_seen += 1

            if max_rows is not None and rows_seen >= max_rows:
                break

    return written


class BM25HardNegativeMiner:
    """
    Mine BM25 hard negatives from a BM25-compatible JSONL corpus.
    """

    def __init__(self, corpus_path: str | Path, retrieve_k: int = 50) -> None:
        if retrieve_k <= 0:
            raise ValueError(f"retrieve_k must be positive, got {retrieve_k}")

        self.retriever = BM25Retriever(str(corpus_path))
        self.retrieve_k = retrieve_k

    def mine(
        self,
        query: str,
        known_positive_passages: list[str],
        num_negatives: int = 1,
        min_score: float | None = None,
        rank_start: int = 0,
        rank_end: int | None = None,
    ) -> list[dict[str, Any]]:
        if num_negatives <= 0:
            raise ValueError(f"num_negatives must be positive, got {num_negatives}")
        
        if rank_start < 0:
            raise ValueError(f"rank_start must be non-negative, got {rank_start}")

        if rank_end is not None and rank_end <= rank_start:
            raise ValueError(
                f"rank_end must be greater than rank_start, got rank_start={rank_start}, "
                f"rank_end={rank_end}"
            )

        known_positive = {normalize_text(text) for text in known_positive_passages}
        seen = set(known_positive)

        results = self.retriever.retrieve(query, k=self.retrieve_k)
        negatives: list[dict[str, Any]] = []

        for result in results:
            candidate_text = clean_text(result.get("text", ""))
            if not candidate_text:
                continue

            candidate_normalized = normalize_text(candidate_text)

            if candidate_normalized in seen:
                continue

            score = float(result.get("score", 0.0))
            if min_score is not None and score < min_score:
                continue

            seen.add(candidate_normalized)
            negatives.append(result)

        negatives_window = negatives[rank_start:rank_end]


        return negatives_window[:num_negatives]


class MinedHardNegativeDataset(Dataset):
    """
    Dataset for mined hard-negative JSONL files.

    Each line should contain:
        query
        positive_passage
        hard_negatives
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        examples: list[dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as in_file:
            for line in in_file:
                if not line.strip():
                    continue

                example = json.loads(line)

                query = clean_text(example.get("query", ""))
                positive_passage = clean_text(example.get("positive_passage", ""))
                hard_negatives = example.get("hard_negatives", [])

                if not query or not positive_passage or not hard_negatives:
                    continue

                mined_example = {
                    "query": query,
                    "positive_passage": positive_passage,
                    "hard_negatives": [clean_text(text) for text in hard_negatives],
                }

                if "hard_negative_scores" in example:
                    mined_example["hard_negative_scores"] = example["hard_negative_scores"]

                examples.append(mined_example)

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]