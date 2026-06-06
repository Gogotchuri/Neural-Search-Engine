# Neural Search Engine

Semantic search over Jurafsky & Martin's *Speech and Language Processing* textbook, using a from-scratch transformer encoder trained with contrastive learning (InfoNCE) on MS MARCO.

## Setup

```bash
uv sync
```

## Pipeline

```bash
# 1. Extract PDF and chunk into passages
uv run python scripts/build_corpus.py "Speech and Language Processing.pdf"

# 2. Train BPE tokenizer (30k vocab)
uv run python scripts/train_tokenizer.py

# 3. Pretrain the encoder with masked language modeling (Wikipedia + book chunks)
uv run pretrain-encoder --device cuda --epochs 5  # -> checkpoints/pretrain.pt

# 4. Mine BM25 hard negatives for MS MARCO query-positive pairs
uv run python scripts/mine_hard_negatives.py  # -> data/cache/msmarco_hard_negatives.jsonl

# 5. Train the encoder contrastively, resuming from the pretrained weights
uv run python scripts/train_encoder.py --device cuda --epochs 3 \
    --resume checkpoints/pretrain.pt --dataset hard-negatives
# Omit --dataset (defaults to msmarco) to train on streamed pairs with in-batch
# negatives only, skipping the mining step above.

# 6. Build FAISS index over book chunks
uv run python scripts/build_index.py --checkpoint checkpoints/encoder.pt

# 7. Search
uv run python scripts/search_neural.py "hidden markov models" --checkpoint checkpoints/encoder.pt
uv run python scripts/bm25_search.py "hidden markov models"  # BM25 baseline
```

**Encoder:** 6-layer pre-norm transformer, 384-dim, 6 heads, ~20M params. Produces L2-normalized embeddings; cosine similarity = dot product via FAISS `IndexFlatIP`.

## Using on Kaggle

Build a wheel and upload it along with your data files as a Kaggle Dataset:

```bash
uv build --wheel
# upload dist/neural_search-0.1.0-py3-none-any.whl, data/tokenizer.json,
# data/chunks.jsonl, and the PDF as a single Kaggle Dataset
```

Then in a Kaggle notebook:

```python
DATA = "/kaggle/input/<your-dataset-name>"

# 1. Install the package
!pip install {DATA}/neural_search-0.1.0-py3-none-any.whl

# 2. Extract PDF and chunk into passages
!build-corpus /path/to/book/dataset --output data/chunks.jsonl

# 3. Train BPE tokenizer (30k vocab)
!train-tokenizer --chunks data/chunks.jsonl --output data/tokenizer.json

# 4. Pretrain the encoder with masked language modeling
!pretrain-encoder --tokenizer data/tokenizer.json --device cuda --epochs 5

# 5. Mine BM25 hard negatives for MS MARCO query-positive pairs
!mine-hard-negatives --output data/cache/msmarco_hard_negatives.jsonl

# 6. Train the encoder contrastively, resuming from the pretrained weights
!train-encoder --tokenizer data/tokenizer.json --device cuda --epochs 3 \
    --resume checkpoints/pretrain.pt --dataset hard-negatives

# 7. Build FAISS index over book chunks
!build-index --tokenizer data/tokenizer.json --chunks data/chunks.jsonl \
    --checkpoint checkpoints/encoder.pt --device cuda

# 8. Search
!search-neural "hidden markov models" --tokenizer data/tokenizer.json \
    --chunks data/chunks.jsonl --checkpoint checkpoints/encoder.pt --device cuda
!bm25-search "hidden markov models" --chunks data/chunks.jsonl
```

If you already have a trained tokenizer and chunks, upload them in the dataset and skip steps 2-3.
