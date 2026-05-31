"""Tests for the corpus extraction and chunking fixes.

Run with:  PYTHONPATH=src pytest tests/test_corpus.py
"""

from neural_search.corpus.extractor import (
    _END_MATTER_RE,
    _extract_context_from_header,
    _strip_chapter_heading,
)
from neural_search.corpus.chunker import (
    _MAX_WORDS,
    _finalize,
    _is_layout_dump,
    _split_long_sentence,
    chunk_pages,
)


def test_end_matter_re_matches_running_headers():
    # The exact running-header forms seen in the J&M PDF tail.
    assert _END_MATTER_RE.match("Bibliography")
    assert _END_MATTER_RE.match("582 Bibliography")
    assert _END_MATTER_RE.match("Bibliography 583")
    assert _END_MATTER_RE.match("612 Subject Index")
    assert _END_MATTER_RE.match("Subject Index 616")


def test_end_matter_re_ignores_real_body_and_chapter_notes():
    # Must NOT fire on body prose or the per-chapter notes section.
    assert not _END_MATTER_RE.match("Bibliographical and Historical Notes")
    assert not _END_MATTER_RE.match("574 CHAPTER 25 • CONVERSATION AND ITS STRUCTURE")
    assert not _END_MATTER_RE.match("The bibliography lists several references.")
    assert not _END_MATTER_RE.match("25.1 • PROPERTIES OF HUMAN CONVERSATION 575")


def test_split_long_sentence_caps_each_piece():
    sentence = " ".join(f"w{i}" for i in range(_MAX_WORDS * 2 + 5))
    pieces = _split_long_sentence(sentence)
    assert len(pieces) >= 3
    assert all(len(p.split()) <= _MAX_WORDS for p in pieces)
    # No words lost or reordered.
    assert " ".join(pieces) == sentence


def test_split_long_sentence_passes_through_normal_sentences():
    sentence = "This is a short and ordinary sentence."
    assert _split_long_sentence(sentence) == [sentence]


def test_no_chunk_exceeds_max_words_even_with_unsplittable_block():
    # A single long run with no sentence boundaries (e.g. an unwrapped list/table).
    blob = " ".join(["alpha", "beta", "gamma", "delta", "epsilon"] * 160)
    pages = [{"text": blob, "chapter": "Chapter 1", "section": "", "page_num": 1}]
    chunks = chunk_pages(pages)
    assert chunks, "expected at least one chunk"
    assert all(c["n_words"] <= _MAX_WORDS for c in chunks)


def test_section_survives_repeated_chapter_header():
    ctx = {"chapter": "Chapter 25", "section": "25.1"}
    # Same chapter's running header repeats on the next page; section must persist.
    ctx = _extract_context_from_header("576 CHAPTER 25 • CONVERSATION", ctx)
    assert ctx == {"chapter": "Chapter 25", "section": "25.1"}


def test_section_resets_on_new_chapter():
    ctx = {"chapter": "Chapter 25", "section": "25.1"}
    ctx = _extract_context_from_header("600 CHAPTER 26 • SOMETHING NEW", ctx)
    assert ctx["chapter"] == "Chapter 26"
    assert ctx["section"] == ""


def test_strip_chapter_heading_removes_heading_block():
    lines = ["CHAPTER", "2", "Words and Tokens",
             "User: I need some help, that much seems certain."]
    assert _strip_chapter_heading(lines) == [
        "User: I need some help, that much seems certain."
    ]


def test_strip_chapter_heading_leaves_normal_pages_untouched():
    lines = ["The standard way to tokenize text is to use the input characters."]
    assert _strip_chapter_heading(lines) == lines


def test_layout_dump_flags_letter_grid_and_number_grid():
    assert _is_layout_dump("e x e c u t i o n i n t e n t i o n")          # spaced grid
    assert _is_layout_dump("0.50 0.39 0.28 0.17 0.06 12 34 56 78 90 11")   # number grid


def test_layout_dump_spares_math_prose():
    prose = ("Formally, then, the gradient of a multi-variable function f is a vector "
             "in which each component expresses how much the function changes.")
    assert not _is_layout_dump(prose)



def test_finalize_drops_unknown_chapter_and_dedups_and_clean_schema():
    base = {"chapter": "Chapter 2", "section": "2.4", "page_start": 12,
            "page_end": 12, "n_words": 120, "text": "a unique passage of prose"}
    raw = [
        {**base, "chapter": "Unknown"},   # chapterless → dropped
        dict(base),                        # kept → id ch2-0000
        dict(base),                        # exact duplicate → dropped
    ]
    out = _finalize(raw)
    assert len(out) == 1
    assert out[0]["chapter"] == "Chapter 2"
    assert out[0]["id"] == "ch2-0000"
    # Deprecated aliases are gone; canonical fields remain.
    assert "word_count" not in out[0] and "chunk_id" not in out[0]
    assert "n_words" in out[0] and "id" in out[0]
