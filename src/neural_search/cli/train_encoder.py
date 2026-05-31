"""Train the contrastive encoder on MS MARCO query-passage pairs.

Usage:
    train-encoder --device cuda --epochs 3
"""

import argparse

from torch.utils.data import DataLoader

from neural_search.data import ContrastiveBatchCollator, MSMARCOPairsDataset
from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.encoder.train import load_checkpoint, train


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive encoder on MS MARCO"
    )
    parser.add_argument(
        "--tokenizer", default="data/tokenizer.json", help="Path to BPE tokenizer"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to load model weights from (fresh optimizer/scheduler)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit dataset size for fast iteration",
    )
    args = parser.parse_args()

    # Dataset: MS MARCO query-positive passage pairs
    print("Loading MS MARCO dataset...")
    dataset = MSMARCOPairsDataset(
        split="train",
        max_examples=args.max_examples,
        shuffle=True,
    )
    print(f"  {len(dataset)} training pairs loaded")

    # Collator: tokenizes and pads queries (max 64) and passages (max 256)
    collator = ContrastiveBatchCollator(
        tokenizer_path=args.tokenizer,
        query_max_length=64,
        passage_max_length=256,
    )

    # DataLoader: batches with shuffling for diverse in-batch negatives
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,  # avoid tiny final batches that produce weak negatives
    )
    print(f"  {len(dataloader)} batches per epoch (batch_size={args.batch_size})")

    # Model
    config = EncoderConfig()
    encoder = Encoder(config)
    if args.resume:
        load_checkpoint(args.resume, encoder)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder: {n_params:,} parameters")

    # Train
    print(f"\nTraining for {args.epochs} epochs on {args.device}...\n")
    train(
        encoder,
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        device=args.device,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
