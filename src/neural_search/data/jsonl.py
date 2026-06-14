from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from neural_search.data.msmarco import clean_text


class ContrastiveJSONLDataset(Dataset):
    """
    Dataset for local cached contrastive-training JSONL files.

    Supported line formats:

        {"query": str, "positive_passage": str}

    or:

        {
            "query": str,
            "positive_passage": str,
            "hard_negatives": list[str],
            "hard_negative_scores": list[float]  # optional
        }

    Dataset-level shuffle/max_examples are useful for selecting a reproducible
    subset from a cached JSONL file. Use DataLoader(shuffle=True) separately
    to reshuffle batch order during training.
    """

    def __init__(
        self,
        path: str | Path,
        require_hard_negatives: bool = False,
        keep_scores: bool = True,
        max_examples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        if max_examples is not None and max_examples <= 0:
            raise ValueError(f"max_examples must be positive or None, got {max_examples}")

        self.path = Path(path)
        examples: list[dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as in_file:
            for line in in_file:
                if not line.strip():
                    continue

                raw = json.loads(line)

                query = clean_text(raw.get("query", ""))
                positive_passage = clean_text(raw.get("positive_passage", ""))

                if not query or not positive_passage:
                    continue

                hard_negatives = [
                    clean_text(text)
                    for text in raw.get("hard_negatives", [])
                    if clean_text(text)
                ]

                if require_hard_negatives and not hard_negatives:
                    continue

                example: dict[str, Any] = {
                    "query": query,
                    "positive_passage": positive_passage,
                }

                if hard_negatives:
                    example["hard_negatives"] = hard_negatives

                if keep_scores and "hard_negative_scores" in raw:
                    example["hard_negative_scores"] = raw["hard_negative_scores"]

                examples.append(example)

        if shuffle:
            random.Random(seed).shuffle(examples)

        if max_examples is not None:
            examples = examples[:max_examples]

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def load_combined_hard_negatives(
    paths: str | list[str],
    upsample: str | list[int] | None = None,
    max_examples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> ConcatDataset:
    """Load one or more hard-negative JSONL files into a single ConcatDataset.

    ``paths`` may be a comma-separated string or a list. ``upsample`` gives a
    per-file integer replication factor (default 1 each) so a smaller source
    (e.g. synthetic book queries) can be weighted up against a larger one
    (e.g. MS MARCO). ``max_examples`` is applied per file.

    Every file is loaded with ``require_hard_negatives=True`` so each example
    carries explicit negatives; keep the mined negative count uniform across
    files (the contrastive collator requires a single count per batch).
    """
    path_list = _as_list(paths)
    if not path_list:
        raise ValueError("No hard-negative paths provided")

    if upsample is None:
        factors = [1] * len(path_list)
    else:
        factors = [int(x) for x in _as_list(upsample)] if isinstance(
            upsample, str
        ) else [int(x) for x in upsample]

    if len(factors) != len(path_list):
        raise ValueError(
            f"upsample count ({len(factors)}) must match path count "
            f"({len(path_list)})"
        )

    datasets: list[Dataset] = []
    for path, factor in zip(path_list, factors):
        if factor <= 0:
            raise ValueError(f"upsample factor must be positive, got {factor}")
        dataset = ContrastiveJSONLDataset(
            path=path,
            require_hard_negatives=True,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
        )
        print(f"  {path}: {len(dataset)} examples x{factor}")
        # Repeat the same underlying dataset object `factor` times (no data copy).
        datasets.extend([dataset] * factor)

    return ConcatDataset(datasets)