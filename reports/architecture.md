# Architecture (Ilia Gogotchuri)

This document explains how the search engine is put together and, more importantly, why
we made the choices we did. The focus is on the encoder, since that is the part I have worked
the most and built from scratch.

## The short version of the story

The task was a neural search engine: given a query, find the matching section in a book.
Our baseline was BM25. BM25 is a strong lexical baseline, but it has no language model
behind it -- it matches words, not meaning. The moment a query is paraphrased or avoids
the exact keywords from the text, BM25 falls apart. On our "hard" query set (queries
deliberately written without any of the original keywords) BM25 scores close to zero across
metrics. That is exactly where a transformer should win.

We decided early on to build almost everything from scratch -- only torch's basic building
blocks and a tokenizer were off-the-shelf. We considered using a small pre-trained BERT,
but building it ourselves was the more honest way to actually understand the architecture.
The design is BERT-style at its core, with the retrieval-specific pieces borrowed from
SBERT (Sentence-BERT). It ended up at about 22M parameters, which is small by modern
standards but turned out to be enough to beat BM25 in the end.

## The encoder backbone

All the model code lives in `src/neural_search/encoder/`. The standard
multi-head self-attention transformer encoder with pre-normalization.

### Pre-norm instead of post-norm

`TransformerBlock` (`block.py`) puts the LayerNorm *before* each sub-layer rather than
after it:

```
x = x + dropout(attn(norm(x)))
x = x + dropout(ffn(norm(x)))
```

As far as I have researched, the original "Attention Is All You Need" used post-norm, but post-norm transformers are
harder to train -- they need more learning-rate warmu. 
Pre-norm keeps a clean residual path from input to output, which makes the gradients behave more perdictably.
For a model we have been training on a single GPU from scratch, stability mattered a lot, I have only so much time to use.

### Sinusoidal positional encoding

We use fixed sinusoidal position encodings (`positional.py`) rather than learned position
embeddings. The sinusoidal version costs zero parameters and zero training,
which is a nice thing to have when the parameter budget is tight. It is stored as a non-learnable
buffer and simply added to the token embeddings.


### Initialization

This is one of the places we deliberately did not just take the default. In `_init_weights`:

- Linear layers use Kaiming (He) uniform init rather than Xavier. For deep ReLU/GELU-style
  stacks, Kaiming converges faster (in theory). We didn't have enough compute to also experiment here.
- The token embedding uses a normal init with std `1/sqrt(hidden_dim)`, following the GPT-2
  recipe. This keeps the embedding scale sane for a residual network of this depth.
- The padding row of the embedding is zeroed - contributes nothing.

### GELU feed-forward, 4x expansion

The FFN (`feedforward.py`) expands to 4x the hidden dimension and uses GELU. The 4x ratio is
the standard from the original transformer. We chose GELU over ReLU because it is the de
facto activation for transformers now -- it is smooth and does not hard-zero negative inputs.

## From token states to a search vector

A search engine needs one vector per passage. Using SBERT as an example we implemented:

## Masked mean pooling

`Encoder.forward` averages the per-token hidden states, but only over the real (non-padding)
tokens -- the attention mask zeroes out padding before the sum, and we divide by the count of
real tokens. We use mean pooling rather than the BERT-style `[CLS]` token because SBERT
showed that mean pooling produces better sentence embeddings for similarity tasks.


### L2 normalization

After pooling we L2-normalize every vector onto the unit sphere. Once everything is unit
length, the dot product *is* cosine similarity, so the retrieval index can use plain inner
product.

### Why a bi-encoder

The query and the passage are encoded *independently* by the same shared encoder
(`train.py` runs both through the one `encoder` object). This is a bi-encoder, as opposed to
a cross-encoder that would feed query and passage together. The bi-encoder is the only design
that scales for search: passage vectors can be computed once, offline, and indexed; at query
time we encode just the query and do a nearest-neighbor lookup.

## Training: three phases, learned the hard way

The training pipeline has three stages, and the shape of it is a consequence of
mistakes we made. The first version of the model skipped straight to fine-tuning, and it
barely beat random. The lesson was that a from-scratch encoder has no idea what language is,
and you cannot teach it retrieval before you teach it English.

### Phase 1 -- MLM pre-training

`pretrain.py` trains the masked-language-model on WikiText-103 plus the book text. 
The `MLMHead` (`mlm_head.py`) sits on top of the per-token hidden states
and predicts the masked tokens.

### Phase 2 -- domain-adaptive pre-training on arXiv (DAP)

After the first real evaluations, the model could handle paraphrase but kept failing on
technical queries -- it had no idea that "HMM" and "Hidden Markov Models" were the same thing,
because academic text barely appeared in its training data. So we added a second pre-training
pass: continued MLM, starting from the Wikipedia checkpoint, on arXiv abstracts from the
relevant CS and stats categories (cs.CL, cs.LG, cs.IR, cs.AI, stat.ML, cs.NE) plus the book
chunks. 

### Phase 3 -- contrastive fine-tuning with InfoNCE

`train.py` fine-tunes the pre-trained encoder for retrieval using InfoNCE loss
(`losses/contrastive.py`). The idea is simple: for each query, its matching passage should
score higher than every other passage in the batch.

- **In-batch negatives.** With a batch of `B` (query, positive) pairs, every query treats the
  other `B-1` positives as negatives. This gives a lot of negative signal at almost
  no cost and is why bigger batches help contrastive training.
- **Hard negatives.** In-batch negatives are mostly *easy* -- random passages are obviously
  wrong. The failure mode we kept hitting was the "almost right" passage: plausible, similar,
  but not the actual answer. To fix that we mine hard negatives (from BM25 results on MSMARCO,
  and LLM-generated queries with mined negatives for the book) and feed them explicitly. This is
  what finally pushed us past BM25 on the combined set.

### Weight decay and accumulation

Both training loops use AdamW with decoupled weight decay, linear warmup into cosine decay,
and gradient clipping. The warmup avoids a destructive first step on randomly initialized (or
freshly adapted) weights; the cosine decay makes big corrections early and small ones late.
Pre-training also supports gradient accumulation, which lets us simulate the large batch sizes
that MLM likes on a GPU that cannot hold them in one go.