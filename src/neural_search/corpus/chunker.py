"""Chunk extracted pages into 200–300 word passages of clean, flowing text.

The text reaching this module has already been flattened to single-spaced prose by
``extractor._clean_body``. Because pypdf does not give reliable blank-line paragraph
boundaries, we chunk at *sentence* granularity instead: split each page into
sentences, then greedily pack them into 200–300 word passages with ~50 words of
sentence-aligned overlap. This guarantees chunks never start or end mid-sentence and
never emit the tiny duplicate fragments the old paragraph-based logic produced.
"""

import json
import re
from pathlib import Path
from typing import List, Dict


# Defaults target ~64-word passages to match the encoder's short-sequence regime.
# build-corpus derives all four from --chunk-words; these are the library fallback.
_MIN_WORDS = 64           # grow a chunk until it reaches at least this many words
_MAX_WORDS = 80           # never exceed this many words
_OVERLAP_WORDS = 12       # carry ~this many words of context into the next chunk
_MIN_CHUNK_WORDS = 24     # drop anything shorter (epigraphs, stray headings, fragments)

# Split on sentence-final punctuation only when the next sentence clearly begins
# (capital letter, quote, or paren). Keeps "U.S." and "Fig. 2" from splitting.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z(“"\'])')


def _word_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _split_long_sentence(sentence: str, max_words: int = _MAX_WORDS) -> List[str]:
    """Hard-split a sentence that exceeds ``max_words`` into word-aligned windows.

    The greedy packer adds the first sentence of a chunk unconditionally, so a single
    "sentence" longer than ``max_words`` (an equation dump, an ARPAbet table, a stray
    index run that never sentence-splits) would otherwise blow past the cap. Breaking
    it into ``max_words``-sized pieces guarantees every chunk stays within the cap.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _chapter_slug(chapter: str) -> str:
    """Turn 'Chapter 2' / 'Unknown' into a stable id prefix like 'ch2' / 'ch0'."""
    match = re.search(r'(\d+)', chapter or "")
    return f"ch{match.group(1)}" if match else "ch0"


def _dedup_key(text: str) -> str:
    """A whitespace-normalized signature of a chunk's *full* text.

    Hashing only the opening (the old behavior) dropped genuinely distinct chunks that
    happened to share a first line, and missed duplicates that started differently.
    Using the whole normalized text makes dedup exact: identical chunks collapse, and
    nothing distinct is lost. (Adjacent chunks share ~50 overlap words by design but
    are never byte-identical, so the intentional overlap is preserved.)
    """
    return re.sub(r'\s+', ' ', text).strip().lower()


# A chunk is a "layout dump" - a figure grid, equation block, count matrix, or chart
# axis pypdf flattened into the body - when single-character tokens or digits dominate.
# Thresholds are deliberately strict: this is an NLP textbook, so ordinary prose that
# discusses equations (with inline variables like f, w, x) must survive untouched.
_MAX_ONE_CHAR_FRAC = 0.45   # e.g. "e x e c u t i o n", "x1 x2 0 1"
_MAX_DIGIT_FRAC = 0.22      # e.g. count matrices, cosine tables, chart axis labels


def _is_layout_dump(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    one_char_frac = sum(1 for tok in tokens if len(tok) == 1) / len(tokens)
    digit_frac = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return one_char_frac > _MAX_ONE_CHAR_FRAC or digit_frac > _MAX_DIGIT_FRAC


def chunk_pages(
    pages: List[Dict],
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
    overlap_words: int = _OVERLAP_WORDS,
    min_chunk_words: int = _MIN_CHUNK_WORDS,
) -> List[Dict]:
    """
    Chunk pages into passages of ~``min_words``–``max_words`` words with
    ~``overlap_words`` of sentence-aligned overlap.

    Args:
        pages: list of {page_num, text, chapter, section} from the extractor
        min_words: grow each chunk until it reaches at least this many words
        max_words: never exceed this many words (also the hard-split window)
        overlap_words: carry ~this many words of context into the next chunk
        min_chunk_words: drop chunks shorter than this

    Returns:
        list of chunk dicts (see ``_make_chunk`` for the schema)
    """
    # Flatten every page into a flat list of sentences carrying their metadata.
    sentences: List[Dict] = []
    for page in pages:
        for sentence_text in _split_sentences(page["text"]):
            for piece in _split_long_sentence(sentence_text, max_words):
                sentences.append({
                    "text": piece,
                    "chapter": page["chapter"],
                    "section": page["section"],
                    "page": page["page_num"],
                })

    raw_chunks: List[Dict] = []
    total = len(sentences)
    start = 0

    while start < total:
        current: List[Dict] = []
        words = 0
        end = start

        # Grow the chunk one sentence at a time until it reaches min_words,
        # but never push it past max_words.
        while end < total:
            sentence_words = _word_count(sentences[end]["text"])
            if current and words + sentence_words > max_words:
                break
            current.append(sentences[end])
            words += sentence_words
            end += 1
            if words >= min_words:
                break

        chunk = _make_chunk(current, min_chunk_words)
        if chunk is not None:
            raw_chunks.append(chunk)

        # Step back ~overlap_words worth of whole sentences so the next chunk
        # overlaps this one without ever splitting a sentence.
        seen_overlap = 0
        overlap_sentences = 0
        for sentence in reversed(current):
            sentence_words = _word_count(sentence["text"])
            if seen_overlap + sentence_words > overlap_words:
                break
            seen_overlap += sentence_words
            overlap_sentences += 1

        start = max(start + 1, end - overlap_sentences)

    return _finalize(raw_chunks)


def _make_chunk(sentences: List[Dict], min_chunk_words: int = _MIN_CHUNK_WORDS) -> Dict | None:
    """Assemble a chunk dict from packed sentences, or None if it's too short."""
    if not sentences:
        return None
    text = " ".join(s["text"] for s in sentences)
    word_count = _word_count(text)
    if word_count < min_chunk_words:
        return None
    if _is_layout_dump(text):  # figure/equation/table grid, not real prose
        return None

    chapter = sentences[0]["chapter"]
    # Use the most specific (last non-empty) section seen within the chunk.
    section = sentences[0]["section"]
    for sentence in sentences:
        if sentence["section"]:
            section = sentence["section"]

    return {
        "chapter": chapter,
        "section": section,
        "page_start": sentences[0]["page"],
        "page_end": sentences[-1]["page"],
        "n_words": word_count,
        "text": text,
    }


def _finalize(chunks: List[Dict]) -> List[Dict]:
    """Drop duplicates and chapterless chunks, then assign a stable string id."""
    seen = set()
    finalized: List[Dict] = []
    next_index = 0
    for chunk in chunks:
        # Part-divider / frontmatter pages never pick up a chapter; they carry no
        # citable context and are low-value overview text, so drop them.
        if chunk["chapter"] == "Unknown":
            continue
        key = _dedup_key(chunk["text"])
        if key in seen:
            continue
        seen.add(key)
        chunk["id"] = f"{_chapter_slug(chunk['chapter'])}-{next_index:04d}"
        finalized.append(chunk)
        next_index += 1
    return finalized


def save_chunks(chunks: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for chunk in chunks:
            out_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks -> {output_path}")


def load_chunks(path: str) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            chunks.append(json.loads(line))
    return chunks
