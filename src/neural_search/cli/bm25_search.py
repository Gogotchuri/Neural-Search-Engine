"""Quick sanity-check: run BM25 search from the command line.

Usage:
    bm25-search "what is beam search"
    bm25-search "hidden Markov model" --k 5
"""

import argparse

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
        print(f"\n[{rank}] id={result['id']}  score={result['score']:.3f}")
        print(f"    {result['chapter']} | {result['section']}")
        print(f"    {result['text']}...")


if __name__ == "__main__":
    main()
