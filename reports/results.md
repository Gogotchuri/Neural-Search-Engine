# Results 
## The First Iteration
We have tried direct fine-tuning of the current Bi-Encoder model on the MSMarco dataset without pre-training it first.
We haven't evaluated with any real evaluation metrics, because it simply didn't work, but the model learned something.
We got InfoNCE loss down to ~0.52 as average, but with manual checking, we haven't had success with any query.

## The Second Iteration
### Training
We have added MLM pre-training on Wiki-103 dataset and book chunks before fine-tunning it on the same marko model.
This work rather well, In hard queries, not containing exact keywords, we beat BM25 and had comparable results in exact queries too.
Here is the data:
```pretrain-encoder --device cuda --log-every 20 --accumulation-steps 4 --epochs 5 --batch-size 32```
```
  WikiText-103: 774806 chunks
  Book text: 1388 chunks × 10 = 13880
  788686 total chunks
  24646 batches per epoch (batch_size=32)
  Weight tying: ON (embedding ↔ MLM projection)
  Encoder: 22,167,552 params | MLM head: 11,668,608 params

Pretraining for 5 epochs on cuda...
  step     20 | loss 10.4331 | avg 10.6119 | ppl 40614.1 | acc 0.0% | lr 3.25e-06 | 1.1 steps/s
...
Epoch 1/5 — avg loss: 5.7067 | ppl: 300.9 | acc: 23.0%
...
Epoch 2/5 — avg loss: 3.6125 | ppl: 37.1 | acc: 41.3%
...
Epoch 3/5 — avg loss: 3.1008 | ppl: 22.2 | acc: 46.9%
...
Epoch 4/5 — avg loss: 2.8796 | ppl: 17.8 | acc: 49.5%
...
Epoch 5/5 — avg loss: 2.7938 | ppl: 16.3 | acc: 50.5%
```

After which we did fine-tuning on MSMarco dataset with selected labels and in-batch negatives:
```train-encoder --device cuda --resume checkpoints/pretrain.pt --epochs 3 --batch-size 64 --log-every 50 --max-grad-norm 5.0 --lr 5e-5```
```
Loading MS MARCO dataset...
  88523 training pairs loaded
  1383 batches per epoch (batch_size=64)
  checkpoint loaded ← checkpoints/pretrain.pt (step 30807)
  Encoder: 22,167,552 parameters

Training for 3 epochs on cuda...
  step     50 | loss 2.1641 | avg 2.2534 | lr 6.04e-06 | grad_norm 3.06 | 6.1 steps/s
...
Epoch 1/3 — avg loss: 0.8307
...
Epoch 2/3 — avg loss: 0.3768
...
Epoch 3/3 — avg loss: 0.2975
```
### Evaluation

(* marks the best score in each column)

#### Basic MLM (no fine-tuning) -- 111 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.2748*  | 0.5931*  | 0.6547*   | 0.5379* | 0.5470* |
| Encoder (untrained) | 0.0000   | 0.0180   | 0.0420    | 0.0127  | 0.0184  |
| Encoder (trained)   | 0.0571   | 0.2027   | 0.3138    | 0.1759  | 0.1950  |

#### MLM + in-batch negative fine-tuning
##### [all] -- 111 queries
(* All of the queries combined)
 
| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.2748*  | 0.5931*  | 0.6547*   | 0.5379* | 0.5470* |
| Encoder (untrained) | 0.0000   | 0.0180   | 0.0420    | 0.0127  | 0.0184  |
| Encoder (trained)   | 0.2147   | 0.4850   | 0.6006    | 0.4725  | 0.4513  |

##### [normal] -- 79 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.3861*  | 0.8333*  | 0.9198*   | 0.7558* | 0.7686* |
| Encoder (untrained) | 0.0000   | 0.0253   | 0.0591    | 0.0178  | 0.0259  |
| Encoder (trained)   | 0.2764   | 0.5928   | 0.7363    | 0.6188  | 0.5743  |

##### [bm25_hard] -- 32 queries (no original keywords mentioned in queries)
(* Queries that have been created with paraphrased question, not containing keywords)
 
| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (untrained) | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (trained)   | 0.0625*  | 0.2188*  | 0.2656*   | 0.1116* | 0.1476* |

## The Third Iteration
We really wanted to beat BM25 with our 22M parameter model. And after analysis of our pitfalls, we identified the following -
the model had hard time with technical term correlation. It couldn't connect keywords like HMM and Viterbi to Hidden Markov Models.
And the second issue, was the answers that were almost right, but not actually correct. We needed to teach it:
a) More technical language for the CS and NLP and 
b) A better way to distinguish between the similar passages better.

so we implemented following changes here:
- Added hard-negatives in MSMarco, mined using heuristically selected results from BM25 (fine-tuning)
- From the target book chunks we used LLMs to generate 3 types of queries: a) Containing exact keywords; b) Keywords in a question; c) Reworded question, containing no keywords.
- We also generated hard-negatives from those using the same method as for MSMarco.
- To teach the model better technical language, we have added a second pre-training step containing paper abstracts from Arxiv (From relevant CS and stat categories)

This resulted in a much better performance, allowing our model to beat BM25.
Here is the data:

### Training

*Pretraining #1 - wikipedia and upscaled book*

```
pretrain-encoder 
--device cuda 
--chunk-words 180 --max-length 256 --log-every 200 
--book-upsample 5 --book-path data/chunks_pretrain.jsonl
--accumulation-steps 4 --epochs 5 --batch-size 32 --lr 5e-4
```

```
Loading pretraining data...
Loading WikiText-103...
  WikiText-103: 717533 chunks
  Book text: 1272 source chunks → 2527 chunks × 5 = 12635
  730168 total chunks
  22817 batches per epoch (batch_size=32)
  Weight tying: ON (embedding ↔ MLM projection)
  Encoder: 22,167,552 params | MLM head: 11,668,608 params

Pretraining for 5 epochs on cuda...

  step    200 | loss 8.0772 | avg 9.3548 | ppl 11553.9 | acc 3.8% | lr 3.51e-05 | 2.1 steps/s
...
Epoch 1/5 - avg loss: 5.7485 | ppl: 313.7 | acc: 22.5%
...
Epoch 2/5 - avg loss: 3.4937 | ppl: 32.9 | acc: 43.1%
...
Epoch 3/5 - avg loss: 3.0300 | ppl: 20.7 | acc: 48.1%
...
Epoch 4/5 - avg loss: 2.8240 | ppl: 16.8 | acc: 50.5%
...
Epoch 5/5 - avg loss: 2.7415 | ppl: 15.5 | acc: 51.4%
```

*Pretraining #2 - arxiv and upscaled book*

```
pretrain-encoder
--accumulation-steps 4 --epochs 3 --batch-size 32 --lr 1e-4
--arxiv-categories cs.CL cs.LG cs.IR cs.AI stat.ML cs.NE
--book-upsample 5 --book-path data/chunks_pretrain.jsonl
--no-wiki --arxiv --arxiv-max-papers 200000 --arxiv-upsample 1
--device cuda 
--chunk-words 180 --max-length 256 --log-every 200
--checkpoint-dir checkpoints-arxiv2/ --resume checkpoints/pretrain.pt
```

```
Loading pretraining data...
  Book text: 1272 source chunks → 2527 chunks × 5 = 12635
Loading arXiv abstracts from gfissore/arxiv-abstracts-2021 (categories: cs.CL, cs.LG, cs.IR, cs.AI, stat.ML, cs.NE)...
  arXiv abstracts: 160955 papers → 225023 chunks × 1 = 225023
  237658 total chunks
  7426 batches per epoch (batch_size=32)
  Weight tying: ON (embedding ↔ MLM projection)
  pretrain checkpoint loaded ← checkpoints/pretrain.pt (step 28521)
  Encoder: 22,167,552 params | MLM head: 11,668,608 params

Pretraining for 3 epochs on cuda...

  step    200 | loss 3.9218 | avg 3.7712 | ppl 43.4 | acc 40.5% | lr 3.60e-05 | 2.2 steps/s
...
Epoch 1/3 - avg loss: 3.1626 | ppl: 23.6 | acc: 46.7%
...
Epoch 2/3 - avg loss: 2.7310 | ppl: 15.3 | acc: 51.4%
...
Epoch 3/3 - avg loss: 2.6475 | ppl: 14.1 | acc: 52.4%
```

*Fine-tuning*
```
train-encoder 
--device cuda
--dataset hard-negatives
--hard-negatives-path data/cache/msmarco_semihard_4.jsonl,data/book_hard_negatives.jsonl
--hard-negatives-upsample 1,6
--resume checkpoints-arxiv2/pretrain.pt 
--epochs 3 --batch-size 32 --log-every 10 
--max-grad-norm 5.0 --lr 1e-4
--query-max-length 32 --passage-max-length 256
```
```
Loading hard-negative dataset(s) from data/cache/msmarco_semihard_4.jsonl,data/book_hard_negatives.jsonl...
  data/cache/msmarco_semihard_4.jsonl: 73661 examples x1
  data/book_hard_negatives.jsonl: 5315 examples x6
  105551 training pairs loaded
  Sequence lengths: query=32, passage=256
  3298 batches per epoch (batch_size=32)
  checkpoint loaded ← checkpoints-arxiv2/pretrain.pt (step 5569)
  Encoder: 22,167,552 parameters

Training for 3 epochs on cuda...

  step     10 | loss 3.6817 | avg 3.7201 | lr 1.01e-06 | grad_norm 3.78 | 1.1 steps/s
  ...
Epoch 1/3 - avg loss: 1.2697
...
Epoch 2/3 - avg loss: 0.4918
...
Epoch 3/3 - avg loss: 0.2970
```

### Evaluation

(* marks the best score in each column)

*Note that the evaluation set has been changed since the last results, so those are not directly comparable, we have reduced the chunk length - increasing the number of chunks so this hard metric is more difficult.*

#### No fine-tuning (pretraining only)

##### [all] -- 111 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.1525*  | 0.3934*  | 0.4587*   | 0.5559* | 0.4306* |
| Encoder (untrained) | 0.0000   | 0.0075   | 0.0111    | 0.0106  | 0.0072  |
| Encoder (trained)   | 0.0480   | 0.1174   | 0.1494    | 0.2201  | 0.1348  |

##### [normal] -- 79 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.2143*  | 0.5527*  | 0.6445*   | 0.7811* | 0.6050* |
| Encoder (untrained) | 0.0000   | 0.0105   | 0.0156    | 0.0149  | 0.0101  |
| Encoder (trained)   | 0.0611   | 0.1385   | 0.1766    | 0.2786  | 0.1659  |

##### [bm25_hard] -- 32 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (untrained) | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (trained)   | 0.0156*  | 0.0654*  | 0.0821*   | 0.0755* | 0.0580* |

#### With fine-tuning

##### [all] -- 111 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.1525   | 0.3934   | 0.4587    | 0.5559  | 0.4306  |
| Encoder (untrained) | 0.0000   | 0.0075   | 0.0111    | 0.0106  | 0.0072  |
| Encoder (trained)   | 0.1806*  | 0.4266*  | 0.4911*   | 0.6374* | 0.4739* |

##### [normal] -- 79 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.2143   | 0.5527*  | 0.6445*   | 0.7811  | 0.6050  |
| Encoder (untrained) | 0.0000   | 0.0105   | 0.0156    | 0.0149  | 0.0101  |
| Encoder (trained)   | 0.2400*  | 0.5494   | 0.6273    | 0.8396* | 0.6192* |

##### [bm25_hard] -- 32 queries

| Retriever           | Recall@1 | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
| ------------------- | -------- | -------- | --------- | ------- | ------- |
| BM25                | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (untrained) | 0.0000   | 0.0000   | 0.0000    | 0.0000  | 0.0000  |
| Encoder (trained)   | 0.0339*  | 0.1234*  | 0.1547*   | 0.1383* | 0.1152* |

