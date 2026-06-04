from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ContrastiveBatchCollator:
    """
    Collator for contrastive query-positive passage training.

    Input examples:
        [
            {"query": "...", "positive_passage": "..."},
            ...
        ]

    or, with explicit hard negatives:
        [
            {
                "query": "...",
                "positive_passage": "...",
                "hard_negatives": ["...", "..."],
            },
            ...
        ]

    Output batch:
        {
            "query_input_ids": Tensor(B, query_max_length),
            "query_attention_mask": Tensor(B, query_max_length),
            "pos_input_ids": Tensor(B, passage_max_length),
            "pos_attention_mask": Tensor(B, passage_max_length),

            # only present when hard_negatives are provided
            "neg_input_ids": Tensor(B * num_negatives, passage_max_length),
            "neg_attention_mask": Tensor(B * num_negatives, passage_max_length),
        }
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        query_max_length: int = 64,
        passage_max_length: int = 256,
        pad_token: str = "[PAD]",
    ) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'tokenizers' package is required. "
                "Install it with: pip install tokenizers"
            ) from exc

        if query_max_length <= 0:
            raise ValueError(
                f"query_max_length must be positive, got {query_max_length}"
            )

        if passage_max_length <= 0:
            raise ValueError(
                f"passage_max_length must be positive, got {passage_max_length}"
            )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.pad_token = pad_token

        pad_token_id = self.tokenizer.token_to_id(pad_token)
        if pad_token_id is None:
            raise ValueError(f"Tokenizer does not contain pad token {pad_token!r}")

        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [str(example["query"]) for example in examples]
        positive_passages = [
            str(example["positive_passage"]) for example in examples
        ]

        query_batch = self._encode_texts(
            queries,
            max_length=self.query_max_length,
        )
        positive_batch = self._encode_texts(
            positive_passages,
            max_length=self.passage_max_length,
        )

        batch = {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "pos_input_ids": positive_batch["input_ids"],
            "pos_attention_mask": positive_batch["attention_mask"],
        }

        has_negatives = [bool(example.get("hard_negatives")) for example in examples]

        if any(has_negatives):
            if not all(has_negatives):
                raise ValueError(
                    "Either all examples in a batch must contain hard_negatives, "
                    "or none of them should."
                )

            negative_counts = [len(example["hard_negatives"]) for example in examples]

            if len(set(negative_counts)) != 1:
                raise ValueError(
                    "All examples in a batch must contain the same number of hard negatives, "
                    f"got counts {negative_counts}"
                )

            negative_passages = [
                str(negative)
                for example in examples
                for negative in example["hard_negatives"]
            ]

            negative_batch = self._encode_texts(
                negative_passages,
                max_length=self.passage_max_length,
            )

            batch["neg_input_ids"] = negative_batch["input_ids"]
            batch["neg_attention_mask"] = negative_batch["attention_mask"]

        return batch
    
    
    def _encode_texts(
        self,
        texts: list[str],
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(
            length=max_length,
            pad_id=self.pad_token_id,
            pad_token=self.pad_token,
        )

        encodings = self.tokenizer.encode_batch(texts)

        input_ids = torch.tensor(
            [encoding.ids for encoding in encodings],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [encoding.attention_mask for encoding in encodings],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }