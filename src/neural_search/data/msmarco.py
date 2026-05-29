from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Iterator

from torch.utils.data import Dataset


def clean_text(text: Any) -> str:
    """Convert a value to a stripped string."""
    if text is None:
        return ""

    return str(text).strip()


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace for duplicate checks."""
    return " ".join(text.lower().strip().split())


def extract_msmarco_passages(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Extract all candidate passages and selected positive passages from one MS MARCO row.

    Returns:
        all_passages: every non-empty passage_text
        positive_passages: passages where is_selected == 1
    """
    passages = row.get("passages")

    if not isinstance(passages, dict):
        return [], []

    passage_texts = passages.get("passage_text", [])
    selected_flags = passages.get("is_selected", [])

    all_passages: list[str] = []
    positive_passages: list[str] = []

    for passage_text, is_selected in zip(passage_texts, selected_flags):
        passage = clean_text(passage_text)
        if not passage:
            continue

        all_passages.append(passage)

        if int(is_selected) == 1:
            positive_passages.append(passage)

    return all_passages, positive_passages

def iter_msmarco_rows(
    split: str = "train",
    config_name: str = "v1.1",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows from MS MARCO."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load MS MARCO. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(
        "microsoft/ms_marco",
        config_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )

    yield from dataset


def collect_msmarco_positive_examples(
    split: str = "train",
    config_name: str = "v1.1",
    max_examples: int | None = 50_000,
    shuffle: bool = False,
    seed: int = 42,
    streaming: bool = True,
    cache_dir: str | None = None,
    include_known_positive_passages: bool = False,
) -> list[dict[str, Any]]:
    """
    Collect MS MARCO query-positive examples.

    If include_known_positive_passages=True, each example also includes all known
    positive passages for the same query row. This is useful for hard-negative
    mining so known positives are not mined as negatives.
    """
    examples: list[dict[str, Any]] = []

    for row in iter_msmarco_rows(
        split=split,
        config_name=config_name,
        streaming=streaming,
        cache_dir=cache_dir,
    ):
        query = clean_text(row.get("query", ""))
        if not query:
            continue

        _, positive_passages = extract_msmarco_passages(row)

        if not positive_passages:
            continue

        for positive_passage in positive_passages:
            example: dict[str, Any] = {
                "query": query,
                "positive_passage": positive_passage,
            }

            if include_known_positive_passages:
                example["known_positive_passages"] = list(positive_passages)

            examples.append(example)

            if max_examples is not None and len(examples) >= max_examples:
                if shuffle:
                    random.Random(seed).shuffle(examples)

                return examples

    if shuffle:
        random.Random(seed).shuffle(examples)

    return examples

def write_msmarco_positive_pairs(
    output_path: str | Path,
    split: str = "train",
    config_name: str = "v1.1",
    max_examples: int | None = 50_000,
    shuffle: bool = False,
    seed: int = 42,
    streaming: bool = True,
    cache_dir: str | None = None,
) -> int:
    """
    Write MS MARCO query-positive pairs to a local JSONL file.

    Each line has:
        {"query": str, "positive_passage": str}

    This avoids repeatedly loading/parsing MS MARCO during training.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = collect_msmarco_positive_examples(
        split=split,
        config_name=config_name,
        max_examples=max_examples,
        shuffle=shuffle,
        seed=seed,
        streaming=streaming,
        cache_dir=cache_dir,
        include_known_positive_passages=False,
    )

    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in examples:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(examples)


class MSMARCOPairsDataset(Dataset):
    """
    PyTorch Dataset for MS MARCO query-positive passage pairs.

    Each item has the form:
        {
            "query": str,
            "positive_passage": str,
        }

    If include_known_positive_passages=True, items also contain:
        {
            "known_positive_passages": list[str]
        }
    """

    def __init__(
        self,
        split: str = "train",
        config_name: str = "v1.1",
        max_examples: int | None = 50_000,
        shuffle: bool = False,
        seed: int = 42,
        streaming: bool = True,
        cache_dir: str | None = None,
        include_known_positive_passages: bool = False,
    ) -> None:
        self.examples = collect_msmarco_positive_examples(
            split=split,
            config_name=config_name,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
            streaming=streaming,
            cache_dir=cache_dir,
            include_known_positive_passages=include_known_positive_passages,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]