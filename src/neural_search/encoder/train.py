"""Training loop for the contrastive encoder.

Consumes batches from ContrastiveBatchCollator and trains the encoder
using InfoNCE loss with in-batch negatives.
"""

import dataclasses
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.encoder import Encoder
from neural_search.losses import infonce_loss


def train(
    encoder: Encoder,
    dataloader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.1,
    max_grad_norm: float = 5.0,
    temperature: float = 0.05,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 1000,
    log_every: int = 100,
    device: str = "cpu",
) -> Encoder:
    """Train the encoder with InfoNCE contrastive loss.

    Args:
        encoder:           The Encoder model to train.
        dataloader:        DataLoader yielding ContrastiveBatchCollator output dicts.
        epochs:            Number of full passes over the dataset.
        lr:                Peak learning rate for AdamW.
        weight_decay:      AdamW weight decay (decoupled from adaptive rate).
        warmup_fraction:   Fraction of total steps for linear warmup.
        max_grad_norm:     Max L2 norm for gradient clipping.
        temperature:       InfoNCE temperature (lower = sharper distribution).
        checkpoint_dir:    Directory to save checkpoints (None = no saving).
        checkpoint_every:  Save a checkpoint every N steps.
        log_every:         Print metrics every N steps.
        device:            "cpu" or "cuda".

    Returns:
        The trained encoder (same object, modified in-place).
    """
    encoder = encoder.to(device)
    encoder.train()

    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=lr, weight_decay=weight_decay
    )

    # LR schedule: linear warmup then cosine decay
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * warmup_fraction)

    # Scheduler to first ramp up the learning rate during the warmup_steps
    # And then use the cosine curve to gradually reduce the learning rate over time
    # The first part makes sure we are avoiding large gradient at the start and not loss information
    # The seond part gradually reduces the learning rate to zero, to make large corrections at the start
    # and gradually tansition to fine-tuning near the end
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )

    # Checkpoint directory
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    t_start = time.time()

    # Run epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Move batch to device
            query_ids = batch["query_input_ids"].to(device)
            query_mask = batch["query_attention_mask"].to(device).float()
            positive_ids = batch["pos_input_ids"].to(device)
            positive_mask = batch["pos_attention_mask"].to(device).float()

            # Forward: encode queries and passages through shared encoder
            query_emb = encoder(query_ids, query_mask)  # (B, H)
            positive_emb = encoder(positive_ids, positive_mask)         # (B, H)

            # Explicit hard negatives are only present when the dataset provides
            # them; otherwise InfoNCE falls back to in-batch negatives alone.
            hard_negative_emb = None
            if "neg_input_ids" in batch:
                hard_negative_ids = batch["neg_input_ids"].to(device)
                hard_negative_mask = batch["neg_attention_mask"].to(device).float()
                hard_negative_emb = encoder(hard_negative_ids, hard_negative_mask)

            # Loss: InfoNCE with in-batch (and optional hard) negatives
            loss = infonce_loss(
                query_emb,
                positive_emb,
                negative_embeddings=hard_negative_emb,
                temperature=temperature,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                encoder.parameters(), max_norm=max_grad_norm
            )

            # Step
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if global_step % log_every == 0:
                avg = epoch_loss / n_batches
                current_lr = scheduler.get_last_lr()[0]
                steps_per_sec = global_step / (time.time() - t_start)
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {loss.item():.4f} | "
                    f"avg {avg:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

            # --- Checkpoint ---
            if (
                checkpoint_dir is not None
                and global_step % checkpoint_every == 0
            ):
                save_checkpoint(
                    encoder, optimizer, scheduler, global_step, ckpt_path
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

    # Final checkpoint
    if checkpoint_dir is not None:
        save_checkpoint(encoder, optimizer, scheduler, global_step, ckpt_path)

    return encoder


def save_checkpoint(
    encoder: Encoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    checkpoint_dir: Path,
) -> None:
    """Save training state to disk."""
    path = checkpoint_dir / "encoder.pt"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": dataclasses.asdict(encoder.config),
        },
        path,
    )
    print(f"  checkpoint saved -> {path} (step {step})")


def load_checkpoint(
    path: str | Path,
    encoder: Encoder,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) from checkpoint.

    Args:
        path:      Path to the checkpoint file.
        encoder:   Encoder model to load weights into.
        optimizer: If provided, restore optimizer state.
        scheduler: If provided, restore scheduler state.

    Returns:
        The global step number at which the checkpoint was saved.
    """
    # Map to CPU so a CUDA-saved checkpoint loads on a CPU-only machine; callers
    # move the encoder to the target device afterwards.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Support both contrastive checkpoints ("model_state_dict")
    # and pretrain checkpoints ("encoder_state_dict")
    if "model_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["model_state_dict"])
    elif "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    else:
        raise KeyError(f"Checkpoint has neither 'model_state_dict' nor 'encoder_state_dict'")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    print(f"  checkpoint loaded ← {path} (step {step})")
    return step
