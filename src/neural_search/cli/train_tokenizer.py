"""Train the BPE tokenizer on J&M + MS MARCO text → data/tokenizer.json.

Usage:
    train-tokenizer
    train-tokenizer --chunks data/chunks.jsonl --n-msmarco 100000
"""

import argparse

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
