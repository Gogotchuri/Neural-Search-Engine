"""Score retrievers on the eval set and print a comparison table.

Compares BM25, the untrained encoder, and the trained encoder by default; the
untrained-vs-trained gap is what training earned over the architecture alone.

Usage:
    evaluate-retrievers --checkpoint checkpoints/encoder.pt
    evaluate-retrievers --checkpoint checkpoints/encoder.pt --no-untrained
    evaluate-retrievers --checkpoint checkpoints/encoder.pt --json results.json
"""

import argparse
import json
import tempfile
from pathlib import Path

from neural_search.evaluation import (
    compare,
    format_table,
    load_corpus_ids,
    load_eval_set,
)
from neural_search.retrieval.bm25 import BM25Retriever


def _neural_retriever(encoder, tokenizer, chunks_path, index_path, device):
    """Wrap an encoder in a NeuralRetriever, building the index if it's missing."""
    from neural_search.retrieval.neural import NeuralRetriever

    retriever = NeuralRetriever(
        index_path=index_path,
        chunks_path=chunks_path,
        encoder=encoder,
        tokenizer=tokenizer,
    )
    if not Path(index_path).exists():
        retriever.build_index(device=device)
    return retriever


def _build_neural_retrievers(args):
    """Build the trained encoder retriever, plus the untrained baseline by default.

    torch/faiss are imported here, not at module top, so BM25-only runs stay light.
    """
    import torch
    from tokenizers import Tokenizer

    from neural_search.encoder.config import EncoderConfig
    from neural_search.encoder.encoder import Encoder
    from neural_search.encoder.train import load_checkpoint

    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = EncoderConfig()
    retrievers = {}

    if not args.no_untrained:
        print("Scoring the untrained encoder (random init)...")
        torch.manual_seed(args.seed)  # stable baseline across runs
        untrained = Encoder(config).to(args.device)
        # Throwaway index; it lives in memory on the retriever, outliving the dir.
        with tempfile.TemporaryDirectory() as tmp:
            retrievers["Encoder (untrained)"] = _neural_retriever(
                untrained, tokenizer, args.chunks,
                str(Path(tmp) / "untrained.faiss"), args.device,
            )

    print("Scoring the trained encoder...")
    trained = Encoder(config)
    load_checkpoint(args.checkpoint, trained)
    trained = trained.to(args.device)
    retrievers["Encoder (trained)"] = _neural_retriever(
        trained, tokenizer, args.chunks, args.index, args.device,
    )
    return retrievers


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrievers on the eval set")
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--index", default="data/index.faiss", help="Trained encoder's FAISS index")
    parser.add_argument("--checkpoint", help="Trained encoder checkpoint (omit to skip neural)")
    parser.add_argument("--no-untrained", action="store_true", help="Skip the random-init baseline")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the untrained baseline's random init")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", help="Optional path to also write metrics as JSON")
    args = parser.parse_args()

    # Validate relevant_ids against the corpus up front (fail loud, not silent zero).
    eval_set = load_eval_set(args.eval_set, corpus_ids=load_corpus_ids(args.chunks))
    print(f"Loaded {len(eval_set)} labelled queries from {args.eval_set}")

    retrievers = {"BM25": BM25Retriever(args.chunks)}
    if args.checkpoint:
        retrievers.update(_build_neural_retrievers(args))
    else:
        print("No --checkpoint given, so we'll just score the BM25 baseline.")

    results = compare(retrievers, eval_set)

    print(f"\nResults over {len(eval_set)} queries\n{'=' * 60}")
    print(format_table(results))
    print(f"{'=' * 60}\n(* marks the best score in each column)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved the full numbers to {args.json}")


if __name__ == "__main__":
    main()
