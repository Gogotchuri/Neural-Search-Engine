"""Extract and chunk the J&M PDF → data/chunks.jsonl.

Usage:
    python scripts/build_corpus.py "Speech and Language Processing.pdf"
    python scripts/build_corpus.py /path/to/book.pdf --output data/chunks.jsonl
"""

from neural_search.cli.build_corpus import main

if __name__ == "__main__":
    main()
