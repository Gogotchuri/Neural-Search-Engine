from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neural_search.losses import infonce_loss
from neural_search.models import SinusoidalPositionalEncoding


def check_infonce_random_embeddings() -> None:
    batch_size = 64
    hidden_dim = 384

    query_embeddings = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)
    positive_embeddings = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)

    loss = infonce_loss(
        query_embeddings,
        positive_embeddings,
        temperature=1.0,
    )

    expected = math.log(batch_size)

    print("Random embeddings InfoNCE check")
    print(f"  loss:     {loss.item():.4f}")
    print(f"  log(B):   {expected:.4f}")
    print("  expected: loss should be roughly close to log(B)")


def check_infonce_aligned_embeddings() -> None:
    batch_size = 64
    hidden_dim = 384

    query_embeddings = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)
    positive_embeddings = query_embeddings.clone()

    loss = infonce_loss(
        query_embeddings,
        positive_embeddings,
        temperature=0.05,
    )

    print("\nAligned embeddings InfoNCE check")
    print(f"  loss:     {loss.item():.4f}")
    print("  expected: loss should be much smaller than random case")


def check_positional_encoding() -> None:
    batch_size = 2
    seq_len = 10
    hidden_dim = 384

    x = torch.zeros(batch_size, seq_len, hidden_dim)
    positional_encoding = SinusoidalPositionalEncoding(
        hidden_dim=hidden_dim,
        max_length=512,
    )

    y = positional_encoding(x)

    print("\nPositional encoding shape check")
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(y.shape)}")

    assert y.shape == x.shape


def main() -> None:
    check_infonce_random_embeddings()
    check_infonce_aligned_embeddings()
    check_positional_encoding()


if __name__ == "__main__":
    main()