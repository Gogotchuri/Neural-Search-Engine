"""Train the contrastive encoder on MS MARCO query-passage pairs.

Usage:
    uv run python scripts/train_encoder.py \
        --tokenizer data/tokenizer.json \
        --epochs 3 \
        --batch-size 64 \
        --lr 2e-4 \
        --device cuda \
        --checkpoint-dir checkpoints/
"""

from neural_search.cli.train_encoder import main

if __name__ == "__main__":
    main()
