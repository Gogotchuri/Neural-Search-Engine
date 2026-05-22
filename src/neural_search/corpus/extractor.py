"""Extract and clean text from the J&M PDF using pypdf."""

import re
from typing import List, Dict

from pypdf import PdfReader


# Running page headers look like:
#   "14 CHAPTER 2 • W ORDS AND TOKENS"
#   "2.4 • S UBWORD TOKENIZATION : B YTE-PAIR ENCODING 13"
_CHAPTER_HEADER_RE = re.compile(r'\bCHAPTER\s+(\d+)\b', re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*[•·]', re.MULTILINE)

# Many J&M PDF versions use the book title as the running header on left-hand pages.
# These lines carry no metadata but the body content beneath them is real.
_BOOK_TITLE_RE = re.compile(r'speech\s+and\s+language\s+processing', re.IGNORECASE)

# Noise to strip from body text
_FIGURE_RE = re.compile(r'Figure\s+\d+\.\d+[:\.].*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)
_TABLE_RE = re.compile(r'Table\s+\d+\.\d+[:\.].*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE)

# Pages before this number can genuinely be frontmatter (title, TOC, preface).
_FRONTMATTER_CUTOFF = 20


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
        context["chapter"] = f"Chapter {chapter_match.group(1)}"
        context["section"] = ""  # reset section when chapter changes
    section_match = _SECTION_HEADER_RE.search(line)
    if section_match:
        context["section"] = section_match.group(1)
        # Infer chapter from section number when chapter is still unknown
        if context["chapter"] == "Unknown":
            context["chapter"] = f"Chapter {section_match.group(1).split('.')[0]}"
    return context


def _clean_body(text: str) -> str:
    """Remove figure/table captions and collapse excess whitespace."""
    text = _FIGURE_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
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
