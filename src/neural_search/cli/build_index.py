"""Encode all corpus chunks and build a FAISS index for neural search.

Usage:
    build-index --checkpoint checkpoints/encoder.pt
"""

import argparse

import torch
from tokenizers import Tokenizer

from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.encoder.train import load_checkpoint
from neural_search.retrieval.neural import NeuralRetriever


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from trained encoder")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--checkpoint", required=True, help="Path to encoder checkpoint")
    parser.add_argument("--output", default="data/index.faiss")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Load model
    config = EncoderConfig()
    encoder = Encoder(config)
    load_checkpoint(args.checkpoint, encoder)
    encoder = encoder.to(args.device)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Build index
    retriever = NeuralRetriever(
        index_path=args.output,
        chunks_path=args.chunks,
        encoder=encoder,
        tokenizer=tokenizer,
    )
    retriever.build_index(device=args.device, batch_size=args.batch_size)

    print("Done.")


if __name__ == "__main__":
    main()
