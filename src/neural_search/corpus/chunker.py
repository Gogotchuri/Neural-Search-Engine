"""Chunk extracted pages into 200–300 word passages with ~50-word overlap."""

import json
import re
from pathlib import Path
from typing import List, Dict


_MIN_WORDS = 200
_MAX_WORDS = 300
_OVERLAP_WORDS = 50


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Rough sentence splitter used only to break oversized paragraphs."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _split_paragraphs(text: str) -> List[str]:
    raw_paragraphs = re.split(r'\n\n+', text)
    paragraphs = []
    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or _word_count(paragraph) < 5:
            continue
        # Break oversized paragraphs at sentence boundaries so they fit chunks
        if _word_count(paragraph) > _MAX_WORDS:
            sentences = _split_sentences(paragraph)
            buffer, buffer_words = [], 0
            for sentence in sentences:
                sentence_words = _word_count(sentence)
                if buffer_words + sentence_words > _MAX_WORDS and buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer, buffer_words = [], 0
                buffer.append(sentence)
                buffer_words += sentence_words
                # Flush immediately when a single sentence already fills MAX_WORDS
                if buffer_words >= _MAX_WORDS:
                    paragraphs.append(" ".join(buffer))
                    buffer, buffer_words = [], 0
            if buffer:
                paragraphs.append(" ".join(buffer))
        else:
            paragraphs.append(paragraph)
    return paragraphs


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Chunk pages into passages of 200–300 words with ~50-word overlap.
    Paragraph boundaries are respected.

    Args:
        pages: list of {page_num, text, chapter, section} from extractor

    Returns:
        list of {chunk_id, chapter, section, text, word_count}
    """
    # Flatten all paragraphs, carrying chapter/section metadata
    paragraphs: List[Dict] = []
    for page in pages:
        for paragraph_text in _split_paragraphs(page["text"]):
            paragraphs.append({
                "text": paragraph_text,
                "chapter": page["chapter"],
                "section": page["section"],
            })

    chunks: List[Dict] = []
    chunk_id = 0
    start = 0

    while start < len(paragraphs):
        current_texts: List[str] = []
        current_words = 0
        chapter = paragraphs[start]["chapter"]
        section = paragraphs[start]["section"]
        end = start

        # Phase 1: grow the chunk until it reaches MIN_WORDS without exceeding MAX_WORDS
        while end < len(paragraphs) and current_words < _MIN_WORDS:
            paragraph = paragraphs[end]
            paragraph_words = _word_count(paragraph["text"])
            if current_texts and current_words + paragraph_words > _MAX_WORDS:
                break
            current_texts.append(paragraph["text"])
            current_words += paragraph_words
            if paragraph["section"]:
                section = paragraph["section"]
            end += 1

        # Phase 2: keep packing more paragraphs in while still under MAX_WORDS
        while end < len(paragraphs):
            paragraph_words = _word_count(paragraphs[end]["text"])
            if current_words + paragraph_words > _MAX_WORDS:
                break
            current_texts.append(paragraphs[end]["text"])
            current_words += paragraph_words
            if paragraphs[end]["section"]:
                section = paragraphs[end]["section"]
            end += 1

        text = " ".join(current_texts)
        if _word_count(text) >= 20:
            chunks.append({
                "chunk_id": chunk_id,
                "chapter": chapter,
                "section": section,
                "text": text,
                "word_count": _word_count(text),
            })
            chunk_id += 1

        # Back up by ~50 words so the next chunk overlaps with this one
        overlap_words = 0
        overlap_count = 0
        for paragraph_text in reversed(current_texts):
            paragraph_words = _word_count(paragraph_text)
            if overlap_words + paragraph_words > _OVERLAP_WORDS:
                break
            overlap_words += paragraph_words
            overlap_count += 1

        start = max(start + 1, end - overlap_count)

    return chunks


def save_chunks(chunks: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for chunk in chunks:
            out_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks → {output_path}")


def load_chunks(path: str) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            chunks.append(json.loads(line))
    return chunks
