"""Encode all corpus chunks and build a FAISS index for neural search.

Usage:
    uv run python scripts/build_index.py \
        --chunks data/chunks.jsonl \
        --tokenizer data/tokenizer.json \
        --checkpoint checkpoints/encoder.pt \
        --output data/index.faiss
"""

from neural_search.cli.build_index import main

if __name__ == "__main__":
    main()
