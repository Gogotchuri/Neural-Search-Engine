from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.data import write_msmarco_positive_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache MS MARCO query-positive pairs to local JSONL."
    )
    parser.add_argument("--max-examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/cache/msmarco_pairs_train.jsonl")
    args = parser.parse_args()

    count = write_msmarco_positive_pairs(
        output_path=args.output,
        split="train",
        max_examples=args.max_examples,
        shuffle=True,
        seed=args.seed,
        streaming=True,
    )

    print(f"Wrote {count} MS MARCO query-positive pairs to {args.output}")


if __name__ == "__main__":
    main()