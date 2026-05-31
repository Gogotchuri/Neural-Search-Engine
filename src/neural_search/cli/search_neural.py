"""Neural search over J&M chunks using a trained encoder + FAISS index.

Usage:
    search-neural "hidden markov models" --checkpoint checkpoints/encoder.pt
"""

import argparse

from tokenizers import Tokenizer

from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.encoder.train import load_checkpoint
from neural_search.retrieval.neural import NeuralRetriever


def main():
    parser = argparse.ArgumentParser(description="Neural search over J&M chunks")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index", default="data/index.faiss")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load model
    config = EncoderConfig()
    encoder = Encoder(config)
    load_checkpoint(args.checkpoint, encoder)
    encoder = encoder.to(args.device)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Load retriever (with pre-built FAISS index)
    retriever = NeuralRetriever(
        index_path=args.index,
        chunks_path=args.chunks,
        encoder=encoder,
        tokenizer=tokenizer,
    )

    # Search
    results = retriever.retrieve(args.query, k=args.k)

    print(f"\nTop {args.k} results for: '{args.query}'\n{'='*60}")
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] id={result['id']}  score={result['score']:.4f}")
        print(f"    {result['chapter']} | {result['section']}")
        print(f"    {result['text'][:200]}...")


if __name__ == "__main__":
    main()
