from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class MSMARCOPairsDataset(Dataset):
    """
    PyTorch Dataset for MS MARCO query-positive passage pairs.

    Each item has the form:
        {
            "query": str,
            "positive_passage": str,
        }

    We use passages where MS MARCO marks is_selected == 1.
    Rows without a selected positive passage are skipped.
    """

    def __init__(
        self,
        split: str = "train",
        config_name: str = "v1.1",
        max_examples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> None:
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
            cache_dir=cache_dir,
        )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        examples: list[dict[str, str]] = []

        for row in dataset:
            query = self._clean_text(row.get("query", ""))
            if not query:
                continue

            positives = self._extract_positive_passages(row)

            for positive_passage in positives:
                examples.append(
                    {
                        "query": query,
                        "positive_passage": positive_passage,
                    }
                )

                if max_examples is not None and len(examples) >= max_examples:
                    self.examples = examples
                    return

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.examples[index]

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""

        return str(text).strip()

    @classmethod
    def _extract_positive_passages(cls, row: dict[str, Any]) -> list[str]:
        passages = row.get("passages")

        if not isinstance(passages, dict):
            return []

        selected_flags = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])

        positives: list[str] = []

        for is_selected, passage_text in zip(selected_flags, passage_texts):
            if int(is_selected) != 1:
                continue

            cleaned_passage = cls._clean_text(passage_text)
            if cleaned_passage:
                positives.append(cleaned_passage)

        return positives