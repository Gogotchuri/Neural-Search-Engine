"""MLM pretraining dataset: WikiText-103 + book text."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class MLMTextDataset(Dataset):
    """Plain-text dataset for masked language model pretraining.

    Loads WikiText-103 from HuggingFace ``datasets`` and optionally
    includes book chunks from a JSONL file (upsampled to avoid being
    drowned by Wikipedia text).

    Each item is ``{"text": str}`` - a chunk of ~256 tokens worth of text.
    Actual tokenization and masking happens in the collator.
    """

    CHARS_PER_CHUNK = 900  # targets ~256 BPE tokens per chunk

    def __init__(
        self,
        book_path: str | Path | None = None,
        book_upsample: int = 5,
        wiki_split: str = "train",
    ) -> None:
        self.chunks: list[str] = []

        # --- WikiText-103 ---
        self._load_wiki(wiki_split)

        # --- Book text ---
        if book_path is not None:
            self._load_book(Path(book_path), upsample=book_upsample)

    def _load_wiki(self, split: str) -> None:
        from datasets import load_dataset

        print("Loading WikiText-103...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

        # WikiText-103 has many blank/header-only lines; concatenate
        # paragraphs into articles then chunk them.
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

        print(f"  WikiText-103: {len(self.chunks)} chunks")

    def _load_book(self, path: Path, upsample: int) -> None:
        book_chunks: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"].strip()
                if text:
                    book_chunks.append(text)

        # Upsample so the book isn't drowned by Wikipedia
        upsampled = book_chunks * upsample
        self.chunks.extend(upsampled)
        print(f"  Book text: {len(book_chunks)} chunks × {upsample} = {len(upsampled)}")

    def _chunk_and_add(self, text: str) -> None:
        """Split long text into chunks of ~CHARS_PER_CHUNK on sentence boundaries."""
        if len(text) <= self.CHARS_PER_CHUNK:
            self.chunks.append(text)
            return

        start = 0
        while start < len(text):
            end = start + self.CHARS_PER_CHUNK
            if end >= len(text):
                self.chunks.append(text[start:])
                break

            # Find the last sentence boundary before the limit
            boundary = text.rfind(". ", start, end)
            if boundary == -1:
                boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start:
                # No good boundary - hard cut
                boundary = end
            else:
                boundary += 1  # include the '.' or '\n'

            chunk = text[start:boundary].strip()
            if chunk:
                self.chunks.append(chunk)
            start = boundary

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"text": self.chunks[index]}
