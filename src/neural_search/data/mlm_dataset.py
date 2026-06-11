"""MLM pretraining dataset: WikiText-103 + book chunks + arXiv abstracts.

Each source can be independently included or excluded so the pretraining
mix can be tuned from the ``pretrain-encoder`` CLI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from torch.utils.data import Dataset

# Sentence boundary: end punctuation followed by whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Default HuggingFace source for arXiv abstracts. Has ``title``, ``abstract``
# and a space-separated ``categories`` string (e.g. "cs.LG stat.ML").
DEFAULT_ARXIV_DATASET = "gfissore/arxiv-abstracts-2021"
# Keep papers whose category list contains any token with one of these
DEFAULT_ARXIV_CATEGORIES = ("cs.", "stat.", "math.")


class MLMTextDataset(Dataset):
    """Plain-text dataset for masked language model pretraining.

    Combines up to three plain-text sources, each optional:

    * **WikiText-103** - streamed from HuggingFace ``datasets``.
    * **Book chunks** - a local JSONL file (upsampled so it isn't drowned
      by Wikipedia text).
    * **arXiv abstracts** - streamed from HuggingFace and filtered to
      cs/stat categories (also upsampled).

    Every source is chunked uniformly to ``chunk_words`` words (sentence-aware
    packing), so each item is ``{"text": str}`` of roughly that length.
    Actual tokenization and masking happens in the collator.
    """

    def __init__(
        self,
        book_path: str | Path | None = None,
        book_upsample: int = 5,
        include_wiki: bool = True,
        wiki_split: str = "train",
        include_arxiv: bool = False,
        arxiv_dataset: str = DEFAULT_ARXIV_DATASET,
        arxiv_categories: tuple[str, ...] = DEFAULT_ARXIV_CATEGORIES,
        arxiv_upsample: int = 1,
        arxiv_max_papers: int = 50_000,
        chunk_words: int = 64,
    ) -> None:
        self.chunk_words = chunk_words
        self.chunks: list[str] = []
        # Contiguous (name, start, end) index ranges per source, for reporting.
        self.source_spans: list[tuple[str, int, int]] = []

        # --- WikiText-103 ---
        if include_wiki:
            start = len(self.chunks)
            self._load_wiki(wiki_split)
            self.source_spans.append(("wiki", start, len(self.chunks)))

        # --- Book chunks ---
        if book_path is not None:
            start = len(self.chunks)
            self._load_book(Path(book_path), upsample=book_upsample)
            self.source_spans.append(("book", start, len(self.chunks)))

        # --- arXiv abstracts (cs/stat) ---
        if include_arxiv:
            start = len(self.chunks)
            self._load_arxiv(
                arxiv_dataset,
                categories=tuple(arxiv_categories),
                upsample=arxiv_upsample,
                max_papers=arxiv_max_papers,
            )
            self.source_spans.append(("arxiv", start, len(self.chunks)))

        if not self.chunks:
            raise ValueError(
                "MLMTextDataset has no data: enable at least one source "
                "(wiki, book_path, or arxiv)."
            )

    def _load_wiki(self, split: str) -> None:
        from datasets import load_dataset

        print("Loading WikiText-103...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

        # WikiText-103 has many blank/header-only lines; concatenate
        # paragraphs into articles then chunk them.
        before = len(self.chunks)
        buffer: list[str] = []
        for row in ds:
            text = row["text"].strip()
            if not text:
                # Blank line = article boundary - flush buffer
                if buffer:
                    self._chunk_and_add("\n".join(buffer))
                    buffer.clear()
                continue
            # Skip section headers (lines starting with " = ")
            if text.startswith("= ") and text.endswith(" ="):
                continue
            buffer.append(text)

        # Flush remaining
        if buffer:
            self._chunk_and_add("\n".join(buffer))

        print(f"  WikiText-103: {len(self.chunks) - before} chunks")

    def _load_book(self, path: Path, upsample: int) -> None:
        # Re-chunk the retrieval corpus to the pretraining target size; the
        # source JSONL keeps its own (larger) chunking for indexing/eval.
        book_chunks: list[str] = []
        n_source = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                if text:
                    n_source += 1
                    book_chunks.extend(self._chunk_text(text))

        # Upsample so the book isn't drowned by Wikipedia
        upsampled = book_chunks * upsample
        self.chunks.extend(upsampled)
        print(
            f"  Book text: {n_source} source chunks → {len(book_chunks)} chunks "
            f"× {upsample} = {len(upsampled)}"
        )

    def _load_arxiv(
        self,
        dataset_name: str,
        categories: tuple[str, ...],
        upsample: int,
        max_papers: int,
    ) -> None:
        """Stream arXiv abstracts and keep those in the given categories.

        ``max_papers <= 0`` means no limit. Abstracts are short (usually a
        single chunk); title and abstract are concatenated as one passage.
        """
        from datasets import load_dataset

        print(
            f"Loading arXiv abstracts from {dataset_name} "
            f"(categories: {', '.join(categories)})..."
        )
        ds = load_dataset(dataset_name, split="train", streaming=True)

        arxiv_chunks: list[str] = []
        n_papers = 0
        for row in ds:
            raw_cats = row.get("categories") or ""
            if isinstance(raw_cats, list):
                raw_cats = " ".join(raw_cats)
            if not any(tok.startswith(categories) for tok in raw_cats.split()):
                continue

            abstract = " ".join((row.get("abstract") or "").split())
            if not abstract:
                continue
            title = " ".join((row.get("title") or "").split())
            text = f"{title}\n\n{abstract}" if title else abstract

            arxiv_chunks.extend(self._chunk_text(text))
            n_papers += 1
            if max_papers > 0 and n_papers >= max_papers:
                break

        upsampled = arxiv_chunks * upsample
        self.chunks.extend(upsampled)
        print(
            f"  arXiv abstracts: {n_papers} papers → {len(arxiv_chunks)} chunks "
            f"× {upsample} = {len(upsampled)}"
        )

    def _chunk_and_add(self, text: str) -> None:
        """Chunk ``text`` and append the pieces to ``self.chunks``."""
        self.chunks.extend(self._chunk_text(text))

    def _chunk_text(self, text: str) -> list[str]:
        """Pack sentences into chunks of at most ``chunk_words`` words.

        Sentence-aware: chunk boundaries fall on sentence ends where possible.
        A single sentence longer than the budget is hard-split on word
        boundaries so no chunk exceeds ``chunk_words``.
        """
        text = text.strip()
        if not text:
            return []
        if len(text.split()) <= self.chunk_words:
            return [text]

        chunks: list[str] = []
        current: list[str] = []  # words in the chunk being built
        for sentence in _SENT_SPLIT.split(text):
            words = sentence.split()
            if not words:
                continue
            # A single over-long sentence: flush, then hard-split it.
            if len(words) > self.chunk_words:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                for i in range(0, len(words), self.chunk_words):
                    chunks.append(" ".join(words[i : i + self.chunk_words]))
                continue
            # Would adding this sentence overflow the budget? Flush first.
            if len(current) + len(words) > self.chunk_words:
                chunks.append(" ".join(current))
                current = []
            current.extend(words)

        if current:
            chunks.append(" ".join(current))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"text": self.chunks[index]}
