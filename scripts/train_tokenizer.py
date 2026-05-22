"""Train the BPE tokenizer on J&M + MS MARCO text → data/tokenizer.json.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --chunks data/chunks.jsonl --n-msmarco 100000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.tokenizer.train import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--output", default="data/tokenizer.json")
    parser.add_argument("--n-msmarco", type=int, default=100_000,
                        help="Number of MS MARCO passages to include (default: 100000)")
    args = parser.parse_args()

    train_tokenizer(args.chunks, args.output, n_msmarco=args.n_msmarco)


if __name__ == "__main__":
    main()
