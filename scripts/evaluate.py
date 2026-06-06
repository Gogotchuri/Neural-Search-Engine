"""Evaluate retrievers on the eval set and print a comparison table.

Usage:
    python scripts/evaluate.py                                  # BM25 only
    python scripts/evaluate.py --checkpoint checkpoints/encoder.pt
"""

from neural_search.cli.evaluate import main

if __name__ == "__main__":
    main()
