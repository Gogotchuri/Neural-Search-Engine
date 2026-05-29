from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from neural_search.data import ContrastiveBatchCollator, MSMARCOPairsDataset, ContrastiveJSONLDataset, MinedHardNegativeDataset


BASE_EXPECTED_KEYS = {
    "query_input_ids",
    "query_attention_mask",
    "pos_input_ids",
    "pos_attention_mask",
}

HARD_NEGATIVE_EXPECTED_KEYS = BASE_EXPECTED_KEYS | {
    "neg_input_ids",
    "neg_attention_mask",
}


def print_example(example: dict[str, Any]) -> None:
    print("\nFirst example")
    print(f"  query: {example['query']}")
    print(f"  positive passage: {example['positive_passage'][:300]}...")

    if "hard_negatives" in example:
        print(f"  number of hard negatives: {len(example['hard_negatives'])}")
        print(f"  first hard negative: {example['hard_negatives'][0][:300]}...")


def check_batch_shapes(
    batch: dict[str, Any],
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    num_negatives: int | None = None,
) -> None:
    print("\nBatch shape check")
    for key, value in batch.items():
        print(f"  {key}: {tuple(value.shape)}")

    if num_negatives is None:
        assert set(batch.keys()) == BASE_EXPECTED_KEYS
    else:
        assert set(batch.keys()) == HARD_NEGATIVE_EXPECTED_KEYS

    assert batch["query_input_ids"].shape == (batch_size, query_max_length)
    assert batch["query_attention_mask"].shape == (batch_size, query_max_length)
    assert batch["pos_input_ids"].shape == (batch_size, passage_max_length)
    assert batch["pos_attention_mask"].shape == (batch_size, passage_max_length)

    if num_negatives is not None:
        expected_negative_rows = batch_size * num_negatives

        assert batch["neg_input_ids"].shape == (
            expected_negative_rows,
            passage_max_length,
        )
        assert batch["neg_attention_mask"].shape == (
            expected_negative_rows,
            passage_max_length,
        )


def run_dataset_check(
    name: str,
    dataset: Any,
    collator: ContrastiveBatchCollator,
    batch_size: int,
    query_max_length: int,
    passage_max_length: int,
    expect_hard_negatives: bool = False,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"{name} check")
    print(f"{'=' * 80}")

    print(f"  number of examples: {len(dataset)}")

    if len(dataset) < batch_size:
        raise ValueError(
            f"{name} has only {len(dataset)} examples, but batch_size={batch_size}"
        )

    first_example = dataset[0]
    print_example(first_example)

    num_negatives: int | None = None

    if expect_hard_negatives:
        if "hard_negatives" not in first_example:
            raise AssertionError(f"{name} expected hard_negatives but none were found")

        num_negatives = len(first_example["hard_negatives"])

        if num_negatives <= 0:
            raise AssertionError(f"{name} has empty hard_negatives list")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(dataloader))

    check_batch_shapes(
        batch=batch,
        batch_size=batch_size,
        query_max_length=query_max_length,
        passage_max_length=passage_max_length,
        num_negatives=num_negatives,
    )

    print(f"\n{name} check passed.")


def build_msmarco_dataset(max_examples: int, seed: int) -> MSMARCOPairsDataset:
    return MSMARCOPairsDataset(
        split="train",
        max_examples=max_examples,
        shuffle=False,
        seed=seed,
        streaming=True,
    )


def build_jsonl_dataset(
    path: str | Path,
    max_examples: int | None,
    shuffle: bool,
    seed: int,
) -> ContrastiveJSONLDataset:
    return ContrastiveJSONLDataset(
        path=path,
        require_hard_negatives=False,
        keep_scores=False,
        max_examples=max_examples,
        shuffle=shuffle,
        seed=seed,
    )


def build_hard_negative_dataset(
    path: str | Path,
    max_examples: int | None,
    shuffle: bool,
    seed: int,
) -> MinedHardNegativeDataset:
    return MinedHardNegativeDataset(
        path=path,
        require_hard_negatives=True,
        keep_scores=False,
        max_examples=max_examples,
        shuffle=shuffle,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check MS MARCO/JSONL datasets and contrastive collator."
    )
    parser.add_argument(
        "--source",
        choices=["msmarco", "jsonl", "hard-negatives", "all"],
        default="all",
        help="Which dataset source to check.",
    )
    parser.add_argument("--tokenizer-path", default="data/tokenizer.json")
    parser.add_argument("--jsonl-path", default="data/cache/msmarco_pairs_train.jsonl")
    parser.add_argument(
        "--hard-negatives-path",
        default="data/cache/msmarco_hard_negatives.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--passage-max-length", type=int, default=256)
    parser.add_argument("--shuffle-jsonl", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    collator = ContrastiveBatchCollator(
        tokenizer_path=args.tokenizer_path,
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
    )

    sources = (
        ["msmarco", "jsonl", "hard-negatives"]
        if args.source == "all"
        else [args.source]
    )

    if "msmarco" in sources:
        dataset = build_msmarco_dataset(
            max_examples=max(args.max_examples, args.batch_size),
            seed=args.seed,
        )

        run_dataset_check(
            name="MSMARCOPairsDataset",
            dataset=dataset,
            collator=collator,
            batch_size=args.batch_size,
            query_max_length=args.query_max_length,
            passage_max_length=args.passage_max_length,
            expect_hard_negatives=False,
        )

    if "jsonl" in sources:
        jsonl_path = Path(args.jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"{jsonl_path} does not exist. Run scripts/cache_msmarco_pairs.py first."
            )

        dataset = build_jsonl_dataset(
            path=jsonl_path,
            max_examples=max(args.max_examples, args.batch_size),
            shuffle=args.shuffle_jsonl,
            seed=args.seed,
        )

        run_dataset_check(
            name="ContrastiveJSONLDataset baseline",
            dataset=dataset,
            collator=collator,
            batch_size=args.batch_size,
            query_max_length=args.query_max_length,
            passage_max_length=args.passage_max_length,
            expect_hard_negatives=False,
        )

    if "hard-negatives" in sources:
        hard_negatives_path = Path(args.hard_negatives_path)
        if not hard_negatives_path.exists():
            raise FileNotFoundError(
                f"{hard_negatives_path} does not exist. "
                "Run scripts/mine_hard_negatives.py first."
            )

        dataset = build_hard_negative_dataset(
            path=hard_negatives_path,
            max_examples=max(args.max_examples, args.batch_size),
            shuffle=args.shuffle_jsonl,
            seed=args.seed,
        )

        run_dataset_check(
            name="ContrastiveJSONLDataset hard negatives",
            dataset=dataset,
            collator=collator,
            batch_size=args.batch_size,
            query_max_length=args.query_max_length,
            passage_max_length=args.passage_max_length,
            expect_hard_negatives=True,
        )

    print("\nData pipeline checks passed.")


if __name__ == "__main__":
    main()