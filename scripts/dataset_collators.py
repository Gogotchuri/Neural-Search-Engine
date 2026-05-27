from __future__ import annotations

from torch.utils.data import DataLoader

from neural_search.data import ContrastiveBatchCollator, MSMARCOPairsDataset


def main() -> None:
    dataset = MSMARCOPairsDataset(
        split="train",
        max_examples=8,
        shuffle=False,
    )

    print("MS MARCO dataset check")
    print(f"  number of examples: {len(dataset)}")

    first_example = dataset[0]
    print("\nFirst example")
    print(f"  query: {first_example['query']}")
    print(f"  positive passage: {first_example['positive_passage'][:300]}...")

    collator = ContrastiveBatchCollator(
        tokenizer_path="data/tokenizer.json",
        query_max_length=64,
        passage_max_length=256,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(dataloader))

    print("\nBatch shape check")
    for key, value in batch.items():
        print(f"  {key}: {tuple(value.shape)}")

    expected_keys = {
        "query_input_ids",
        "query_attention_mask",
        "pos_input_ids",
        "pos_attention_mask",
    }
    assert set(batch.keys()) == expected_keys

    assert batch["query_input_ids"].shape == (4, 64)
    assert batch["query_attention_mask"].shape == (4, 64)
    assert batch["pos_input_ids"].shape == (4, 256)
    assert batch["pos_attention_mask"].shape == (4, 256)

    print("\ndata pipeline check passed.")


if __name__ == "__main__":
    main()