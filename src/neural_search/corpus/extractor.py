"""Extract and clean text from the J&M PDF using pypdf."""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Set

from pypdf import PdfReader


# Running page headers look like:
#   "14 CHAPTER 2 • W ORDS AND TOKENS"
#   "2.4 • S UBWORD TOKENIZATION : B YTE-PAIR ENCODING 13"
_CHAPTER_HEADER_RE = re.compile(r'\bCHAPTER\s+(\d+)\b', re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*[•·]', re.MULTILINE)

# Many J&M PDF versions use the book title as the running header on left-hand pages.
# These lines carry no metadata but the body content beneath them is real.
_BOOK_TITLE_RE = re.compile(r'speech\s+and\s+language\s+processing', re.IGNORECASE)

# End matter: the consolidated Bibliography and Subject Index that follow the last
# chapter. Their running headers ("Bibliography", "582 Bibliography", "612 Subject
# Index") carry no chapter, so every page below them otherwise inherits the previous
# chapter's label and chunks into pure citation/index noise. We stop extraction the
# moment one of these headers appears. The pattern requires the line to be *only* the
# header (plus an optional page number) so it never fires on body prose or on the
# per-chapter "Bibliographical and Historical Notes" sections.
_END_MATTER_RE = re.compile(
    r'^\s*(?:\d{1,4}\s+)?(?:Bibliography|Subject\s+Index|Index)(?:\s+\d{1,4})?\s*$',
    re.IGNORECASE,
)

# Noise to strip from body text
_FIGURE_RE = re.compile(r'Figure\s+\d+\.\d+[:\.].*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)
_TABLE_RE = re.compile(r'Table\s+\d+\.\d+[:\.].*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)

# Pages before this number can genuinely be frontmatter (title, TOC, preface).
_FRONTMATTER_CUTOFF = 20


def _load_dictionary() -> Set[str]:
    """Load the system word list for de-hyphenation; empty set if unavailable."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        if Path(path).exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return {line.strip().lower() for line in f if line.strip()}
    return set()


_DICTIONARY = _load_dictionary()


def _keep_hyphen(left: str, right: str) -> bool:
    """Decide whether a hyphen at a line break is a genuine compound (keep it).

    'retrieval' + 'based' → both real words, 'retrievalbased' is not → keep the hyphen.
    'intro' + 'duce'      → 'introduce' is a real word                → drop the hyphen.
    Falls back to a capitalization heuristic when no dictionary is available.
    """
    left_l, right_l = left.lower(), right.lower()
    if _DICTIONARY:
        combined = left_l + right_l
        if left_l in _DICTIONARY and right_l in _DICTIONARY and combined not in _DICTIONARY:
            return True
        return False
    return right[:1].isupper()


def _dehyphenate(text: str) -> str:
    def repl(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        joiner = "-" if _keep_hyphen(left, right) else ""
        return f"{left}{joiner}{right}"

    return re.sub(r"(\w+)-\n(\w+)", repl, text)


# J&M sets a fixed glossary of key terms in a smaller font in the page margin. pypdf
# appends each margin term to an adjacent body token with no space, producing glue like
# "systemELIZA", "ortokenization", "Anutterance". This is a *known, finite* list, so we
# split those terms back out — guarded by the dictionary so we never break a genuine
# word (e.g. "lemma" inside "dilemma", or compounds that are real words on their own).
_MARGIN_TERMS = [
    "tokenization", "tokenizing", "ELIZA", "BPE", "disfluency", "fragment",
    "utterance", "lemma", "wordform", "corpus", "corpora", "morpheme", "morphemes",
    "stemming", "lemmatization", "segmentation", "normalization", "clitic", "clitics",
    "wordtype", "wordtoken", "wordinstance", "filledpause",
]
# Longest-first so "morphemes" is tried before "morpheme".
_MARGIN_TERMS.sort(key=len, reverse=True)


def _is_real_word(token: str) -> bool:
    """True if the glued run is itself a legitimate word we must not split."""
    return bool(_DICTIONARY) and token.lower() in _DICTIONARY


def _split_margin_terms(text: str) -> str:
    """Insert a space where a known margin glossary term is glued to a body word."""
    for term in _MARGIN_TERMS:
        # term glued onto the end of the preceding word: "system" + "ELIZA"
        def split_before(m: re.Match, term=term) -> str:
            prefix = m.group(1)
            return m.group(0) if _is_real_word(prefix + term) else f"{prefix} {term}"

        text = re.sub(rf"([A-Za-z]+)({re.escape(term)})\b", split_before, text)

        # term glued onto the start of the following word: "tokenization" + "the"
        def split_after(m: re.Match, term=term) -> str:
            suffix = m.group(2)
            return m.group(0) if _is_real_word(term + suffix) else f"{term} {suffix}"

        text = re.sub(rf"\b({re.escape(term)})([a-z]+)", split_after, text)

    return text


def _is_frontmatter(first_line: str, page_num: int) -> bool:
    """
    Return True only for the actual title/TOC pages at the start of the PDF.

    The book title also appears as a running header on left-hand content pages,
    so we restrict this check to the first _FRONTMATTER_CUTOFF pages.
    """
    if page_num > _FRONTMATTER_CUTOFF:
        return False
    upper = first_line.upper()
    return (
        _BOOK_TITLE_RE.search(upper) is not None
        or "CONTENTS" in upper
        or "SUMMARY OF CONTENTS" in upper
    )


def _extract_context_from_header(line: str, current: Dict) -> Dict:
    """Pull chapter and section numbers out of a running page header."""
    context = dict(current)
    chapter_match = _CHAPTER_HEADER_RE.search(line)
    if chapter_match:
        new_chapter = f"Chapter {chapter_match.group(1)}"
        # The chapter running header repeats on (almost) every page, so only reset the
        # section when the chapter number actually *changes*. Resetting on every header
        # — the old behavior — wiped the section on every right-hand page, leaving ~45%
        # of chunks with no section metadata.
        if new_chapter != context["chapter"]:
            context["chapter"] = new_chapter
            context["section"] = ""
    section_match = _SECTION_HEADER_RE.search(line)
    if section_match:
        context["section"] = section_match.group(1)
        # Infer chapter from section number when chapter is still unknown
        if context["chapter"] == "Unknown":
            context["chapter"] = f"Chapter {section_match.group(1).split('.')[0]}"
    return context


def _strip_chapter_heading(lines: List[str]) -> List[str]:
    """Drop the 'CHAPTER / <n> / <Title>' block pypdf dumps into a chapter opener's body.

    On a chapter's first page the large display heading is emitted as separate body
    lines ("CHAPTER", "2", "Words and Tokens") above the real text, which otherwise
    glue onto the following prose as "CHAPTER Words and Tokens User: ...". We strip a
    leading bare "CHAPTER" line, an optional standalone chapter number, and the short
    title line(s) that follow (no sentence-ending punctuation), stopping at the first
    line of real prose or the epigraph.
    """
    if not lines or lines[0].strip().upper() != "CHAPTER":
        return lines
    i = 1
    if i < len(lines) and lines[i].strip().isdigit():
        i += 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        # A title is short and unpunctuated; anything longer or sentence-final is body.
        if len(stripped.split()) <= 7 and not stripped.endswith((".", "!", "?", ":")):
            i += 1
        else:
            break
    return lines[i:]


def _clean_body(text: str) -> str:
    """Normalize unicode, de-hyphenate, drop page numbers, and join wrapped lines.

    pypdf preserves the PDF's physical line wraps as ``\\n`` and renders ligatures
    (ﬁ, ﬂ) and hyphenated line breaks (``intro-\\nduce``) literally. Left untouched
    these make every chunk a wall of broken text, so we flatten a page into clean,
    flowing prose here before it ever reaches the chunker.
    """
    text = unicodedata.normalize("NFKC", text)       # ﬁ→fi, ﬂ→fl, non-breaking spaces
    text = _dehyphenate(text)                         # join split words, keep compounds
    text = _split_margin_terms(text)                 # un-glue margin glossary terms
    text = _FIGURE_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)   # drop standalone page numbers
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)       # wrapped lines → single spaces
    text = re.sub(r"\s{2,}", " ", text)               # collapse remaining whitespace
    return text.strip()


def extract_book(pdf_path: str) -> List[Dict]:
    """
    Extract text from all content pages of the J&M PDF.

    Skips title/TOC pages at the start of the PDF only.
    Handles both chapter-title running headers and book-title running headers
    (which appear on alternating pages in many PDF layouts).

    Returns a list of dicts: {page_num, text, chapter, section}
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Extracting {total_pages} pages...")

    pages = []
    context = {"chapter": "Unknown", "section": ""}

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue

        lines = raw.strip().split("\n")
        first_line = lines[0]

        # Stop once we reach the Bibliography / Subject Index. Everything after is
        # end matter (citations, page-number index entries) with no chapter context.
        if _END_MATTER_RE.match(first_line):
            print(f"Reached end matter ('{first_line.strip()}') at page {page_num}; stopping.")
            break

        # Skip genuine frontmatter (title page, TOC) near the start of the PDF only
        if _is_frontmatter(first_line, page_num):
            continue

        # Book-title running headers carry no chapter/section info; just drop them.
        # Chapter/section running headers update context and are also dropped.
        if _BOOK_TITLE_RE.search(first_line):
            body_lines = lines[1:]
        else:
            context = _extract_context_from_header(first_line, context)
            body_lines = lines[1:]

        body_lines = _strip_chapter_heading(body_lines)
        body = _clean_body("\n".join(body_lines))
        if not body:
            continue

        pages.append({
            "page_num": page_num,
            "text": body,
            "chapter": context["chapter"],
            "section": context["section"],
        })

        if page_num % 100 == 0:
            print(f"  {page_num}/{total_pages} pages done")

    print(f"Extracted {len(pages)} content pages.")
    return pages
