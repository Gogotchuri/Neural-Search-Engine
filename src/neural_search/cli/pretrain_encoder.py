"""Pretrain the encoder with MLM on WikiText-103 + book text.

Usage:
    pretrain-encoder --device cuda --epochs 5
"""

import argparse

from torch.utils.data import DataLoader

from neural_search.data.mlm_collator import MLMBatchCollator
from neural_search.data.mlm_dataset import MLMTextDataset
from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.encoder.mlm_head import MLMHead
from neural_search.encoder.pretrain import load_pretrain_checkpoint, pretrain


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain encoder with MLM on WikiText-103 + book text"
    )
    parser.add_argument(
        "--tokenizer", default="data/tokenizer.json", help="Path to BPE tokenizer"
    )
    parser.add_argument(
        "--book-path", default="data/chunks.jsonl", help="Path to book chunks JSONL"
    )
    parser.add_argument(
        "--book-upsample", type=int, default=10, help="Upsample factor for book text"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
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
        book_path=args.book_path,
        book_upsample=args.book_upsample,
    )
    print(f"  {len(dataset)} total chunks")

    # Collator
    collator = MLMBatchCollator(
        tokenizer_path=args.tokenizer,
        max_length=256,
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
