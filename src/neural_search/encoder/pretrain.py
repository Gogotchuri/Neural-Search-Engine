"""MLM pretraining loop for the encoder.

Trains the encoder backbone with a masked language modelling objective
on WikiText-103 + book text before contrastive fine-tuning.
"""

import dataclasses
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_search.encoder.encoder import Encoder
from neural_search.encoder.mlm_head import MLMHead


def pretrain(
    encoder: Encoder,
    mlm_head: MLMHead,
    dataloader: DataLoader,
    *,
    epochs: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.1,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 2000,
    validate_every: int = 2000,
    log_every: int = 100,
    device: str = "cpu",
) -> Encoder:
    """Pretrain the encoder with MLM.

    Returns:
        The pretrained encoder (same object, modified in-place).
    """
    encoder = encoder.to(device)
    mlm_head = mlm_head.to(device)
    encoder.train()
    mlm_head.train()

    # Combine parameters, deduplicating tied weights
    all_params = list(encoder.parameters()) + list(mlm_head.parameters())
    params = list({id(p): p for p in all_params}.values())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = len(dataloader) * epochs // accumulation_steps
    warmup_steps = int(total_steps * warmup_fraction)

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

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    micro_step = 0
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_masked = 0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).float()
            labels = batch["labels"].to(device)

            # Forward
            hidden_states = encoder.encode_tokens(input_ids, attention_mask)
            logits = mlm_head(hidden_states)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / accumulation_steps

            # Backward
            loss.backward()

            micro_step += 1

            # Accumulate accuracy stats (on the unscaled loss)
            with torch.no_grad():
                masked = labels != -100
                if masked.any():
                    preds = logits[masked].argmax(dim=-1)
                    epoch_correct += (preds == labels[masked]).sum().item()
                    epoch_masked += masked.sum().item()

            epoch_loss += loss.item() * accumulation_steps
            n_batches += 1

            # We are simulating bigger batches with accumulation steps
            if micro_step % accumulation_steps != 0:
                continue

            # Gradient clipping + step
            nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_every == 0:
                avg_loss = epoch_loss / n_batches
                perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                accuracy = epoch_correct / max(epoch_masked, 1) * 100
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_start
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {loss.item() * accumulation_steps:.4f} | "
                    f"avg {avg_loss:.4f} | "
                    f"ppl {perplexity:.1f} | "
                    f"acc {accuracy:.1f}% | "
                    f"lr {current_lr:.2e} | "
                    f"{global_step / elapsed:.1f} steps/s"
                )

            if checkpoint_dir and global_step % checkpoint_every == 0:
                save_pretrain_checkpoint(
                    encoder, mlm_head, optimizer, scheduler, global_step, Path(checkpoint_dir)
                )

            if global_step % validate_every == 0:
                _run_validation(encoder, mlm_head, dataloader, criterion, device)

        avg_loss = epoch_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))
        accuracy = epoch_correct / max(epoch_masked, 1) * 100
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"avg loss: {avg_loss:.4f} | ppl: {perplexity:.1f} | acc: {accuracy:.1f}%"
        )

    # Final checkpoint
    if checkpoint_dir:
        save_pretrain_checkpoint(
            encoder, mlm_head, optimizer, scheduler, global_step, Path(checkpoint_dir)
        )

    return encoder


@torch.no_grad()
def _run_validation(
    encoder: Encoder,
    mlm_head: MLMHead,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_batches: int = 50,
) -> None:
    """Quick validation on a few batches (reuses train dataloader)."""
    encoder.eval()
    mlm_head.eval()

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).float()
        labels = batch["labels"].to(device)

        hidden = encoder.encode_tokens(input_ids, attention_mask)
        logits = mlm_head(hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        masked = labels != -100
        if masked.any():
            preds = logits[masked].argmax(dim=-1)
            total_correct += (preds == labels[masked]).sum().item()
            total_masked += masked.sum().item()

    avg_loss = total_loss / max(max_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    accuracy = total_correct / max(total_masked, 1) * 100
    print(
        f"  [val] loss {avg_loss:.4f} | ppl {perplexity:.1f} | acc {accuracy:.1f}%"
    )

    encoder.train()
    mlm_head.train()


def save_pretrain_checkpoint(
    encoder: Encoder,
    mlm_head: MLMHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    checkpoint_dir: Path,
) -> None:
    """Save pretraining state. Encoder is saved separately for easy fine-tuning loading."""
    path = checkpoint_dir / "pretrain.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "mlm_head_state_dict": mlm_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": dataclasses.asdict(encoder.config),
        },
        path,
    )
    print(f"  checkpoint saved -> {path} (step {step})")


def load_pretrain_checkpoint(
    path: str | Path,
    encoder: Encoder,
    mlm_head: MLMHead | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Load pretraining checkpoint.

    If only encoder is provided, loads just the encoder weights
    (for transitioning to contrastive fine-tuning).
    """
    checkpoint = torch.load(path, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    if mlm_head is not None and "mlm_head_state_dict" in checkpoint:
        mlm_head.load_state_dict(checkpoint["mlm_head_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    print(f"  pretrain checkpoint loaded ← {path} (step {step})")
    return step
