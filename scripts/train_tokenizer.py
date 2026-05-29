"""Train the BPE tokenizer on J&M + MS MARCO text → data/tokenizer.json.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --chunks data/chunks.jsonl --n-msmarco 100000
"""

from neural_search.cli.train_tokenizer import main

if __name__ == "__main__":
    main()
