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

# 3. Train the encoder on MS MARCO
uv run python scripts/train_encoder.py --device cuda --epochs 3

# 4. Build FAISS index over book chunks
uv run python scripts/build_index.py --checkpoint checkpoints/encoder.pt

# 5. Search
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

# 4. Train the encoder on MS MARCO
!train-encoder --tokenizer data/tokenizer.json --device cuda --epochs 3

# 5. Build FAISS index over book chunks
!build-index --tokenizer data/tokenizer.json --chunks data/chunks.jsonl \
    --checkpoint checkpoints/encoder.pt --device cuda

# 6. Search
!search-neural "hidden markov models" --tokenizer data/tokenizer.json \
    --chunks data/chunks.jsonl --checkpoint checkpoints/encoder.pt --device cuda
!bm25-search "hidden markov models" --chunks data/chunks.jsonl
```

If you already have a trained tokenizer and chunks, upload them in the dataset and skip steps 2-3.
