# Corpus Processing, Search Baseline, and Evaluation (Alex Inauri)

My part of the project was getting the textbook ready to search, building the baseline and neural search on top of it, and measuring how well everything actually worked. The other half of the team taught the model how to match queries to passages. I handled what gets searched and how we know if it's any good.

What I worked on:

- pulling clean text out of the J&M PDF
- cutting that text into small passages with stable ids
- training the shared BPE tokenizer
- the BM25 baseline and the neural FAISS search behind one shared interface
- generating queries over the book with an LLM
- building the evaluation set and the metrics to score it

---

## Getting the book ready

The model trains on MS MARCO but searches the textbook, and those are two separate things. So before anything could be retrieved, the book had to become a clean set of passages, each small enough to be a sensible result and tagged with enough info to cite.

The book is only a PDF, which is annoying to work with. `pypdf` gives you the text but keeps the page layout instead of the actual structure of the writing. So I split this into two steps: clean the text first, then cut it into chunks. They break for different reasons, so keeping them apart made each one much easier to debug.

```text
J&M PDF → clean each page → cut into chunks → chunks.jsonl
```

## Cleaning the PDF text

`pypdf` text looks fine to a human but is full of little problems that quietly hurt search. I handled each one directly:

- normalize unicode so ligatures like `ﬁ` become normal letters
- rejoin words split across lines (`intro-\nduce` → `introduce`) while keeping real hyphenated words intact, checked against the system dictionary
- the book prints key terms in the margin and `pypdf` glues them onto nearby words (`systemELIZA`), so I split those back apart
- strip figure/table captions, page numbers, and the line breaks the PDF leaves everywhere

Two bigger things needed care. The chapter and section numbers only show up in the page headers, so I parse those and attach them to each page (with a fix so the section isn't wiped every time the chapter header repeats). And extraction stops once it hits the Bibliography and Index at the end of the book, since those pages are just citation noise with no chapter to attach them to.

## Cutting it into chunks

I cut the text at sentence boundaries. Chunking on paragraphs doesn't work because `pypdf` doesn't give reliable paragraph breaks, and an earlier version that tried it produced tiny broken fragments.

Instead I add sentences one at a time until a chunk hits its target size, then start the next one with a bit of overlap so an important sentence never gets stranded on a boundary. Because the overlap is whole sentences, chunks never start or end mid-sentence. The passage size is a setting, kept short on purpose so the book passages look similar to the short MS MARCO passages the model trained on.

Two filters keep junk out: anything that's really a figure grid or equation block (mostly single letters and digits) gets dropped, and exact duplicates get removed. Each surviving chunk gets a stable id like `ch9-0042`. This id is important — the eval set points at chunks by id, not by position, because positions shift every time you re-chunk and that would break every label.

The final corpus is **4,133 chunks across 24 chapters**.

---

## The tokenizer

The whole team shares one tokenizer, so this was part of my job. It's a 30k-vocab BPE tokenizer trained with HuggingFace `tokenizers` (we're not allowed to use pretrained models, but training our own vocab is fine).

The main decision was what to train it on. It sees two kinds of text: short search queries and dense textbook prose. So I trained it on both the book chunks and MS MARCO together. That way it handles everyday words and book jargon like "Viterbi" or "perplexity" without chopping them into tiny pieces. Training on only one side would have hurt the other.

---

## Search: one interface, two backends

Both the baseline and the neural model are reached through one small interface:

```python
class Retriever(ABC):
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        ...
```

That's what makes them directly comparable — the eval harness and the demo just call `retrieve()` and don't care which one is behind it.

**BM25** is the baseline, using `rank_bm25`. It needs no training, so we had a working search from day one. It's also genuinely strong on technical text, where queries often contain the exact term ("BLEU", "Viterbi") that's in the answer, so beating it isn't easy.

**Neural** wraps the trained encoder with a FAISS index. Since the corpus is small (~4k passages), it uses exact search, and because the embeddings are L2-normalized this is just cosine similarity. There's also a check that fails loudly if the saved index doesn't match the current chunks, so a stale index can't silently return wrong passages.

---

## Making queries with an LLM

The book has passages but no queries, and we needed queries to evaluate against. Writing enough by hand is slow, so I used an LLM to generate them — about **5,315 queries** total.

I generated them in a mix of styles on purpose: easy keyword queries that share words with the passage, plain direct questions, and harder paraphrased ones with no keyword overlap. The hard ones matter most — they test whether the model actually understands meaning instead of just matching words.

---

## The evaluation set

Metrics are only as good as their labels, so the final eval set is hand-checked: **111 queries across the 24 chapters**, each with the chunks that actually answer it.

I split them into two groups:

- **normal (79)** — keyword queries where BM25 does well ("what is beam search")
- **bm25_hard (32)** — paraphrased queries with no shared keywords, which BM25 basically can't answer

This split is the whole point. A single overall number would hide the one thing we want to show: the neural model wins exactly on the hard semantic queries where keyword matching fails. Loading the eval set also checks every chunk id against the real corpus, so a typo'd id can't silently never match.

---

## Metrics

Three metrics, kept simple and unit-tested:

- **Recall@k** — is the right chunk in the top k? Reported at 1, 5, and 10.
- **MRR@10** — rewards putting a good answer near the top, not just anywhere.
- **nDCG@10** — the standard IR ranking metric.

Together they answer: did we find it, how high was it, and how good was the order.

The harness runs any retriever over the eval set, scores all three, and prints a comparison table — overall and per category — so BM25 and the neural model go through the exact same path.

---

## Results

The comparison is BM25 vs the untrained encoder vs the trained one, by category. Roughly, BM25 looks like:

```text
overall    R@10 ≈ 0.65
normal     R@10 ≈ 0.98   (great at keywords)
bm25_hard  R@10 ≈ 0.00   (no keywords to match)
```

That `bm25_hard` row is the money shot: BM25 scores ~0 there, so any decent neural score on it is a clear win for understanding meaning. BM25 stays strong on the normal queries, and that's fine to report honestly — beating it everywhere on technical text was never the realistic goal.

---

## Summary

I turned the textbook PDF into 4,133 clean, stably-ided passages across 24 chapters, trained the shared tokenizer on book + MS MARCO text, and put both BM25 and the neural FAISS search behind one interface so they're directly comparable. Then I generated queries with an LLM, hand-built a 111-query eval set split into easy and hard categories, and wrote the metrics and harness that produce the BM25-vs-neural tables. The story those tables tell: the neural model earns its place on exactly the semantic queries where keyword search falls apart.
