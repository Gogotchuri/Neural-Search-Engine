"""Collator that tokenizes raw text and applies BERT-style MLM masking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# Special token IDs (from tokenizer/train.py SPECIAL_TOKENS order)
_PAD_ID = 0
_UNK_ID = 1
_CLS_ID = 2
_SEP_ID = 3
_MASK_ID = 4
_SPECIAL_IDS = {_PAD_ID, _UNK_ID, _CLS_ID, _SEP_ID, _MASK_ID}


class MLMBatchCollator:
    """Tokenise raw text and apply 80/10/10 MLM masking.

    Input examples:  [{"text": "..."}, ...]
    Output batch:    {"input_ids": (B,L), "attention_mask": (B,L), "labels": (B,L)}

    Labels are the original token IDs at masked positions, -100 elsewhere.
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        max_length: int = 256,
        mask_prob: float = 0.15,
        vocab_size: int = 30_000,
    ) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [example["text"] for example in examples]

        # Tokenize with truncation and padding
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            length=self.max_length,
            pad_id=_PAD_ID,
            pad_token="[PAD]",
        )
        encodings = self.tokenizer.encode_batch(texts)

        input_ids = torch.tensor(
            [enc.ids for enc in encodings], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [enc.attention_mask for enc in encodings], dtype=torch.long
        )

        # Clone original IDs for labels before masking
        labels = input_ids.clone()

        # Build mask: only mask non-special, non-padding tokens
        maskable = torch.ones_like(input_ids, dtype=torch.bool)
        for sid in _SPECIAL_IDS:
            maskable &= input_ids != sid

        # Random selection of ~mask_prob fraction of maskable tokens
        rand = torch.rand_like(input_ids, dtype=torch.float)
        selected = maskable & (rand < self.mask_prob)

        # Out of the selected tokens to mask we
        # change: 80% -> [MASK], 10% -> random token, 10% -> unchanged
        replace_rand = torch.rand_like(input_ids, dtype=torch.float)
        mask_token = selected & (replace_rand < 0.8)
        random_token = selected & (replace_rand >= 0.8) & (replace_rand < 0.9)
        # remaining 10% of selected: keep original (no modification needed)

        input_ids[mask_token] = _MASK_ID
        input_ids[random_token] = torch.randint(
            len(_SPECIAL_IDS), self.vocab_size, (random_token.sum(),)
        )

        # Only compute loss on selected (masked) positions
        labels[~selected] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
