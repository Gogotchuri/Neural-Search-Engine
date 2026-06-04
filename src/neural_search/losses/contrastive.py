from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float = 0.05,
    negative_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute InfoNCE loss for dense retrieval using in-batch negatives
    and optional explicit hard negatives.

    query_embeddings[i] should match positive_embeddings[i].
    All other positive_embeddings[j] where j != i are treated as negatives.
    The embeddings are expected to be L2-normalized before calling this function.

    Without explicit negatives:
        logits shape is (B, B), where other positives in the batch are negatives.

    With explicit negatives:
        logits shape is (B, B + M), where M is the number of provided negatives.

    Args:
        query_embeddings: Tensor of shape (batch_size, embedding_dim).
        positive_embeddings: Tensor of shape (batch_size, embedding_dim).
        temperature: Softmax temperature. Lower values make the task sharper.
        negative_embeddings: Optional tensor of shape (M, embedding_dim) or (B, N, embedding_dim)

    Returns:
        Scalar loss tensor.
    """
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"query_embeddings must be 2D, got shape {query_embeddings.shape}"
        )

    if positive_embeddings.ndim != 2:
        raise ValueError(
            f"positive_embeddings must be 2D, got shape {positive_embeddings.shape}"
        )

    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError(
            "query_embeddings and positive_embeddings must have the same shape, "
            f"got {query_embeddings.shape} and {positive_embeddings.shape}"
        )

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    
    batch_size, embedding_dim = query_embeddings.shape

    candidate_embeddings = positive_embeddings

    if negative_embeddings is not None:
        if negative_embeddings.ndim == 3:
            if negative_embeddings.shape[0] != batch_size:
                raise ValueError(
                    "When negative_embeddings is 3D, its first dimension must match "
                    f"batch size {batch_size}, got {negative_embeddings.shape}"
                )

            if negative_embeddings.shape[2] != embedding_dim:
                raise ValueError(
                    "negative_embeddings embedding dimension must match "
                    f"{embedding_dim}, got {negative_embeddings.shape[2]}"
                )

            negative_embeddings = negative_embeddings.reshape(-1, embedding_dim)

        elif negative_embeddings.ndim == 2:
            if negative_embeddings.shape[1] != embedding_dim:
                raise ValueError(
                    "negative_embeddings embedding dimension must match "
                    f"{embedding_dim}, got {negative_embeddings.shape[1]}"
                )
        else:
            raise ValueError(
                "negative_embeddings must have shape (M, D), (B, N, D), or None; "
                f"got {negative_embeddings.shape}"
            )

        candidate_embeddings = torch.cat(
            [positive_embeddings, negative_embeddings],
            dim=0,
        )

    logits = query_embeddings @ candidate_embeddings.T
    logits = logits / temperature

    # [0, 1, 2, 3...,n-1] is the correct label for each (the diagonal)
    labels = torch.arange(batch_size, device=query_embeddings.device)

    return F.cross_entropy(logits, labels)