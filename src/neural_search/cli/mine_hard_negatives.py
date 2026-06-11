from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.data import (
    BM25HardNegativeMiner,
    MSMARCOPairsDataset,
    write_msmarco_bm25_corpus,
)

def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

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
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=None)
    parser.add_argument("--shuffle-corpus", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=100)
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
        shuffle=args.shuffle_corpus,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
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

    total = len(dataset)
    start_time = time.time()

    print("Mining hard negatives...")
    with open(output_path, "w", encoding="utf-8") as out_file:
        for index, example in enumerate(dataset, start=1):
            negatives = miner.mine(
                query=example["query"],
                known_positive_passages=example["known_positive_passages"],
                num_negatives=args.num_negatives,
                min_score=args.min_score,
                rank_start=args.rank_start,
                rank_end=args.rank_end,
            )

            if len(negatives) < args.num_negatives:
                skipped += 1
            else:
                record = {
                    "query": example["query"],
                    "positive_passage": example["positive_passage"],
                    "hard_negatives": [negative["text"] for negative in negatives],
                    "hard_negative_scores": [negative["score"] for negative in negatives],
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if index % args.progress_every == 0 or index == total:
                elapsed = time.time() - start_time
                examples_per_second = index / max(elapsed, 1e-8)
                remaining = total - index
                eta_seconds = remaining / max(examples_per_second, 1e-8)

                print(
                    f"  processed {index:,}/{total:,} "
                    f"({index / total:.1%}) | "
                    f"kept {kept:,} | skipped {skipped:,} | "
                    f"{examples_per_second:.2f} examples/s | "
                    f"elapsed {format_seconds(elapsed)} | "
                    f"eta {format_seconds(eta_seconds)}",
                    flush=True,
                )

    print(f"Wrote {kept} mined examples to {output_path}.")
    print(f"Skipped {skipped} examples with too few hard negatives.")


if __name__ == "__main__":
    main()