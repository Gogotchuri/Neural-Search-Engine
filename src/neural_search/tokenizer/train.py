"""Train a 30k-vocab BPE tokenizer on J&M book text + MS MARCO passages."""

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


VOCAB_SIZE = 30_000
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _jm_texts(chunks_path: str) -> Iterator[str]:
    with open(chunks_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            yield json.loads(line)["text"]


def _msmarco_texts(max_passages: int = 100_000) -> Iterator[str]:
    """Stream passage text from MS MARCO v1.1 (HuggingFace datasets)."""
    from datasets import load_dataset
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    passages_yielded = 0
    for example in dataset:
        yield example["query"]
        for passage in example["passages"]["passage_text"]:
            yield passage
            passages_yielded += 1
            if passages_yielded >= max_passages:
                return


def _all_texts(chunks_path: str, n_msmarco: int) -> Iterator[str]:
    yield from _jm_texts(chunks_path)
    yield from _msmarco_texts(n_msmarco)


def train_tokenizer(chunks_path: str, output_path: str, n_msmarco: int = 100_000) -> Tokenizer:
    """
    Train BPE tokenizer and save to output_path.

    Args:
        chunks_path: path to data/chunks.jsonl
        output_path: where to save tokenizer.json
        n_msmarco:   number of MS MARCO passages to include
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE}) on J&M + {n_msmarco:,} MS MARCO passages...")
    tokenizer.train_from_iterator(_all_texts(chunks_path, n_msmarco), trainer=trainer)

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved -> {output_path}")
    return tokenizer
