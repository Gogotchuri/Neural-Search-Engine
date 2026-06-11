"""Extract and chunk the J&M PDF -> data/chunks.jsonl.

Usage:
    build-corpus "Speech and Language Processing.pdf"
    build-corpus /path/to/book.pdf --output data/chunks.jsonl
"""

import argparse
import textwrap
from collections import Counter

from neural_search.corpus.extractor import extract_book
from neural_search.corpus.chunker import chunk_pages, save_chunks


def _print_report(chunks, sample=3):
    """Print a length histogram and a few sample chunks for a quick eyeball check."""
    word_counts = [chunk["n_words"] for chunk in chunks]
    print(
        f"\nChunks: {len(chunks)}  |  words: "
        f"min={min(word_counts)} avg={sum(word_counts) // len(word_counts)} max={max(word_counts)}"
    )

    buckets = Counter((wc // 50) * 50 for wc in word_counts)
    print("Length histogram (words):")
    for lo in sorted(buckets):
        bar = "#" * (buckets[lo] * 40 // max(buckets.values()))
        print(f"  {lo:>4}-{lo + 49:<4} | {buckets[lo]:>4} {bar}")

    print(f"\nFirst {sample} chunks:")
    for chunk in chunks[:sample]:
        preview = textwrap.shorten(chunk["text"], width=280, placeholder=" …")
        print(
            f"\n  [{chunk['id']}] {chunk['chapter']} §{chunk['section'] or '-'} "
            f"p{chunk['page_start']}–{chunk['page_end']} ({chunk['n_words']}w)"
        )
        print(textwrap.indent(textwrap.fill(preview, width=96), "    "))


def main():
    parser = argparse.ArgumentParser(description="Extract and chunk the J&M PDF")
    parser.add_argument("pdf_path", help="Path to the J&M PDF")
    parser.add_argument("--output", default="data/chunks.jsonl")
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=64,
        help="Target words per passage. min/max/overlap are derived from it.",
    )
    args = parser.parse_args()

    # Derive the four chunker knobs from a single target size.
    cw = args.chunk_words
    pages = extract_book(args.pdf_path)
    chunks = chunk_pages(
        pages,
        min_words=cw,
        max_words=round(cw * 1.25),
        overlap_words=round(cw * 0.2),
        min_chunk_words=round(cw * 0.4),
    )

    _print_report(chunks)
    save_chunks(chunks, args.output)


if __name__ == "__main__":
    main()
