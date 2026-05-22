"""Extract and chunk the J&M PDF → data/chunks.jsonl.

Usage:
    python scripts/build_corpus.py "Speech and Language Processing.pdf"
    python scripts/build_corpus.py /path/to/book.pdf --output data/chunks.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_search.corpus.extractor import extract_book
from neural_search.corpus.chunker import chunk_pages, save_chunks


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk the J&M PDF")
    parser.add_argument("pdf_path", help="Path to the J&M PDF")
    parser.add_argument("--output", default="data/chunks.jsonl")
    args = parser.parse_args()

    pages = extract_book(args.pdf_path)
    chunks = chunk_pages(pages)

    word_counts = [chunk["word_count"] for chunk in chunks]
    print(f"Chunks: {len(chunks)}  |  words: min={min(word_counts)} avg={sum(word_counts)//len(word_counts)} max={max(word_counts)}")

    save_chunks(chunks, args.output)


if __name__ == "__main__":
    main()
