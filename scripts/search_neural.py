"""Neural search over J&M chunks using a trained encoder + FAISS index.

Usage:
    uv run python scripts/search_neural.py "hidden markov models" \
        --index data/index.faiss \
        --chunks data/chunks.jsonl \
        --tokenizer data/tokenizer.json \
        --checkpoint checkpoints/encoder.pt \
        --k 5
"""

from neural_search.cli.search_neural import main

if __name__ == "__main__":
    main()
