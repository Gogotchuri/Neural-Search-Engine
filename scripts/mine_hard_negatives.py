from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.data import (
    BM25HardNegativeMiner,
    MSMARCOPairsDataset,
    write_msmarco_bm25_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine BM25 hard negatives for MS MARCO query-positive pairs."
    )
    parser.add_argument("--max-examples", type=int, default=5_000)
    parser.add_argument("--max-corpus-rows", type=int, default=5_000)
    parser.add_argument("--num-negatives", type=int, default=2)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--corpus-output",
        default="data/cache/msmarco_bm25_corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/cache/msmarco_hard_negatives.jsonl",
    )
    args = parser.parse_args()

    print("Loading query-positive examples...")
    dataset = MSMARCOPairsDataset(
        split="train",
        max_examples=args.max_examples,
        shuffle=True,
        seed=args.seed,
        streaming=True,
        include_known_positive_passages=True,
    )

    print(f"Loaded {len(dataset)} query-positive examples.")

    print("Writing BM25 corpus from MS MARCO candidate passages...")
    corpus_size = write_msmarco_bm25_corpus(
        output_path=args.corpus_output,
        split="train",
        max_rows=args.max_corpus_rows,
        streaming=True,
    )

    print(f"Wrote {corpus_size} unique candidate passages to {args.corpus_output}.")

    print("Building BM25 index...")
    miner = BM25HardNegativeMiner(
        corpus_path=args.corpus_output,
        retrieve_k=args.retrieve_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    print("Mining hard negatives...")
    with open(output_path, "w", encoding="utf-8") as out_file:
        for example in dataset:
            negatives = miner.mine(
                query=example["query"],
                known_positive_passages=example["known_positive_passages"],
                num_negatives=args.num_negatives,
                min_score=args.min_score,
            )

            if len(negatives) < args.num_negatives:
                skipped += 1
                continue

            record = {
                "query": example["query"],
                "positive_passage": example["positive_passage"],
                "hard_negatives": [negative["text"] for negative in negatives],
                "hard_negative_scores": [negative["score"] for negative in negatives],
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} mined examples to {output_path}.")
    print(f"Skipped {skipped} examples with too few hard negatives.")


if __name__ == "__main__":
    main()