"""Quick sanity-check: run BM25 search from the command line.

Usage:
    python scripts/bm25_search.py "what is beam search"
    python scripts/bm25_search.py "hidden Markov model" --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.retrieval.bm25 import BM25Retriever


def main():
    parser = argparse.ArgumentParser(description="BM25 search over J&M chunks")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever = BM25Retriever(args.chunks)
    results = retriever.retrieve(args.query, k=args.k)

    print(f"\nTop {args.k} results for: '{args.query}'\n{'='*60}")
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] chunk_id={result['chunk_id']}  score={result['score']:.3f}")
        print(f"    {result['chapter']} | {result['section']}")
        print(f"    {result['text'][:200]}...")


if __name__ == "__main__":
    main()
