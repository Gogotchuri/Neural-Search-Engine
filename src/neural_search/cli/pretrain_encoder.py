"""Pretrain the encoder with MLM on WikiText-103 + book chunks + arXiv abstracts.

Each source can be toggled independently, e.g.:
    pretrain-encoder --device cuda --epochs 5                 # wiki + book
    pretrain-encoder --arxiv --device cuda                    # wiki + book + arXiv
    pretrain-encoder --no-wiki --no-book --arxiv --device cuda  # arXiv only

Staged / domain-adaptive pretraining (general -> domain):
    # stage 1: general English, 3 epochs
    pretrain-encoder --epochs 3
    # stage 2: continue on cs/stat arXiv from the stage-1 checkpoint
    pretrain-encoder --no-wiki --no-book --arxiv --resume checkpoints/pretrain.pt

Use --dry-run on any combination to print exact token counts before committing
to a run.
"""

import argparse

from torch.utils.data import DataLoader

from neural_search.data.mlm_collator import MLMBatchCollator
from neural_search.data.mlm_dataset import (
    DEFAULT_ARXIV_CATEGORIES,
    DEFAULT_ARXIV_DATASET,
    MLMTextDataset,
)
from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.encoder.mlm_head import MLMHead
from neural_search.encoder.pretrain import load_pretrain_checkpoint, pretrain


def _token_report(dataset, tokenizer_path, max_length, epochs, mask_prob=0.15):
    """Tokenize every chunk through the real tokenizer and print exact counts.

    Mirrors the collator: same tokenizer, truncation at ``max_length`` (which
    includes the [CLS]/[SEP] specials added by the tokenizer's post-processor),
    no padding. "Effective" tokens are what the model actually sees per epoch.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))

    spans = dataset.source_spans or [("all", 0, len(dataset))]
    header = (
        f"{'source':>8} | {'chunks':>9} | {'raw tok':>13} | {'effective':>13} | "
        f"{'trunc%':>6} | {'tok/chunk':>9}"
    )
    print("\n=== token report (max_length=%d) ===" % max_length)
    print(header)
    print("-" * len(header))

    tot_chunks = tot_raw = tot_eff = 0
    for name, start, end in spans:
        texts = dataset.chunks[start:end]
        if not texts:
            continue
        tok.no_truncation()
        raw = sum(len(e.ids) for e in tok.encode_batch(texts))
        tok.enable_truncation(max_length=max_length)
        eff = sum(len(e.ids) for e in tok.encode_batch(texts))
        n = len(texts)
        trunc = 100.0 * (raw - eff) / raw if raw else 0.0
        print(
            f"{name:>8} | {n:>9,} | {raw:>13,} | {eff:>13,} | "
            f"{trunc:>5.1f}% | {eff / n:>9.1f}"
        )
        tot_chunks += n
        tot_raw += raw
        tot_eff += eff

    print("-" * len(header))
    trunc = 100.0 * (tot_raw - tot_eff) / tot_raw if tot_raw else 0.0
    print(
        f"{'TOTAL':>8} | {tot_chunks:>9,} | {tot_raw:>13,} | {tot_eff:>13,} | "
        f"{trunc:>5.1f}% | {tot_eff / max(tot_chunks, 1):>9.1f}"
    )

    # Supervision signal: MLM only predicts ~mask_prob of non-special tokens.
    # Approximate specials as 2 per chunk ([CLS]/[SEP]).
    content = max(tot_eff - 2 * tot_chunks, 0)
    targets_per_epoch = content * mask_prob
    print(
        f"\nEffective tokens/epoch: {tot_eff:,}"
        f"  (×{epochs} epochs = {tot_eff * epochs:,})"
    )
    print(
        f"~MLM targets/epoch (≈{mask_prob:.0%} of content): {targets_per_epoch:,.0f}"
        f"  (×{epochs} = {targets_per_epoch * epochs:,.0f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain encoder with MLM on WikiText-103 + book chunks + arXiv"
    )
    parser.add_argument(
        "--tokenizer", default="data/tokenizer.json", help="Path to BPE tokenizer"
    )
    # --- WikiText-103 ---
    parser.add_argument(
        "--no-wiki", action="store_true", help="Exclude WikiText-103 from the mix"
    )
    # --- Book chunks ---
    parser.add_argument(
        "--no-book", action="store_true", help="Exclude book chunks from the mix"
    )
    parser.add_argument(
        "--book-path", default="data/chunks.jsonl", help="Path to book chunks JSONL"
    )
    parser.add_argument(
        "--book-upsample", type=int, default=10, help="Upsample factor for book text"
    )
    # --- arXiv abstracts (cs/stat) ---
    parser.add_argument(
        "--arxiv",
        action="store_true",
        help="Include cs/stat arXiv abstracts (streamed from HuggingFace)",
    )
    parser.add_argument(
        "--arxiv-dataset",
        default=DEFAULT_ARXIV_DATASET,
        help="HuggingFace dataset id for arXiv abstracts",
    )
    parser.add_argument(
        "--arxiv-categories",
        nargs="+",
        default=list(DEFAULT_ARXIV_CATEGORIES),
        help="Keep papers whose categories match these prefixes (e.g. cs. stat.)",
    )
    parser.add_argument(
        "--arxiv-upsample", type=int, default=3, help="Upsample factor for arXiv text"
    )
    parser.add_argument(
        "--arxiv-max-papers",
        type=int,
        default=50_000,
        help="Cap on arXiv papers to stream (<=0 for no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load sources, print exact tokenized counts, and exit (no training)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=64,
        help="Target words per training chunk (applied to wiki, book, arXiv). "
        "~1.4 tokens/word on this corpus, so 64 words ≈ 90 tokens.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Max sequence length for tokenization (truncation + padding). "
        "Should cover chunk-words × ~1.4 + 2 specials. When resuming, must "
        "be <= the checkpoint's position-table size (256).",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to pretrain checkpoint to resume from",
    )
    parser.add_argument(
        "--no-tie-weights",
        action="store_true",
        help="Disable weight tying between embedding and MLM projection",
    )
    args = parser.parse_args()

    # Dataset
    print("Loading pretraining data...")
    dataset = MLMTextDataset(
        include_wiki=not args.no_wiki,
        book_path=None if args.no_book else args.book_path,
        book_upsample=args.book_upsample,
        include_arxiv=args.arxiv,
        arxiv_dataset=args.arxiv_dataset,
        arxiv_categories=tuple(args.arxiv_categories),
        arxiv_upsample=args.arxiv_upsample,
        arxiv_max_papers=args.arxiv_max_papers,
        chunk_words=args.chunk_words,
    )
    print(f"  {len(dataset)} total chunks")

    if args.dry_run:
        _token_report(
            dataset, args.tokenizer, max_length=args.max_length, epochs=args.epochs
        )
        return

    # Collator
    collator = MLMBatchCollator(
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"  {len(dataloader)} batches per epoch (batch_size={args.batch_size})")

    # Model
    config = EncoderConfig()
    if args.max_length > config.max_seq_len:
        # Grow the sinusoidal position table to cover longer sequences.
        # NOTE: this changes the `pe` buffer shape, so --resume from a
        # checkpoint built at the old length will fail the strict load.
        if args.resume:
            parser.error(
                f"--max-length {args.max_length} exceeds the checkpoint's "
                f"position table ({config.max_seq_len}); cannot resume. "
                "Use --max-length <= 256 or start a fresh run."
            )
        config.max_seq_len = args.max_length
    encoder = Encoder(config)
    mlm_head = MLMHead(config.hidden_dim, config.vocab_size)

    if not args.no_tie_weights:
        mlm_head.tie_weights(encoder.token_emb.weight)
        print("  Weight tying: ON (embedding ↔ MLM projection)")

    if args.resume:
        load_pretrain_checkpoint(args.resume, encoder, mlm_head)

    n_encoder = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in mlm_head.parameters() if p.requires_grad)
    print(f"  Encoder: {n_encoder:,} params | MLM head: {n_head:,} params")

    # Train
    print(f"\nPretraining for {args.epochs} epochs on {args.device}...\n")
    pretrain(
        encoder,
        mlm_head,
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        accumulation_steps=args.accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        validate_every=args.validate_every,
        log_every=args.log_every,
        device=args.device,
    )

    print("\nPretraining complete.")
