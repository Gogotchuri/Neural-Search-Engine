"""Quick sanity-check: run BM25 search from the command line.

Usage:
    python scripts/bm25_search.py "what is beam search"
    python scripts/bm25_search.py "hidden Markov model" --k 5
"""

from neural_search.cli.bm25_search import main

if __name__ == "__main__":
    main()
