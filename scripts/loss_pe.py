from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neural_search.losses import infonce_loss
from neural_search.models import SinusoidalPositionalEncoding


def check_infonce_with_hard_negatives() -> None:
    torch.manual_seed(42)

    batch_size = 64
    embedding_dim = 384
    num_negatives = 2

    print("\nInfoNCE with explicit hard negatives check")

    query_embeddings = F.normalize(
        torch.randn(batch_size, embedding_dim),
        dim=-1,
    )
    positive_embeddings = F.normalize(
        torch.randn(batch_size, embedding_dim),
        dim=-1,
    )
    negative_embeddings = F.normalize(
        torch.randn(batch_size * num_negatives, embedding_dim),
        dim=-1,
    )

    random_loss = infonce_loss(
        query_embeddings,
        positive_embeddings,
        temperature=1.0,
        negative_embeddings=negative_embeddings,
    )

    expected_random_loss = math.log(batch_size + batch_size * num_negatives)

    print("Random embeddings with hard negatives")
    print(f"  loss:     {random_loss.item():.4f}")
    print(f"  log(C):   {expected_random_loss:.4f}")
    print("  expected: loss should be roughly close to log(number of candidates)")

    aligned_query_embeddings = F.normalize(
        torch.randn(batch_size, embedding_dim),
        dim=-1,
    )
    aligned_positive_embeddings = aligned_query_embeddings.clone()

    aligned_negative_embeddings = F.normalize(
        torch.randn(batch_size * num_negatives, embedding_dim),
        dim=-1,
    )

    aligned_loss = infonce_loss(
        aligned_query_embeddings,
        aligned_positive_embeddings,
        temperature=0.05,
        negative_embeddings=aligned_negative_embeddings,
    )

    print("\nAligned positives with hard negatives")
    print(f"  loss:     {aligned_loss.item():.4f}")
    print("  expected: loss should be much smaller than random hard-negative case")

    negative_embeddings_3d = negative_embeddings.reshape(
        batch_size,
        num_negatives,
        embedding_dim,
    )

    flat_negative_loss = infonce_loss(
        query_embeddings,
        positive_embeddings,
        temperature=1.0,
        negative_embeddings=negative_embeddings,
    )

    grouped_negative_loss = infonce_loss(
        query_embeddings,
        positive_embeddings,
        temperature=1.0,
        negative_embeddings=negative_embeddings_3d,
    )

    print("\nFlat vs grouped hard-negative shape check")
    print(f"  flat negatives loss:    {flat_negative_loss.item():.4f}")
    print(f"  grouped negatives loss: {grouped_negative_loss.item():.4f}")
    print("  expected: values should match")

    assert random_loss.ndim == 0
    assert aligned_loss.ndim == 0
    assert aligned_loss < random_loss
    assert torch.allclose(flat_negative_loss, grouped_negative_loss, atol=1e-6)

    print("\nHard-negative InfoNCE checks passed.")

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
    check_infonce_with_hard_negatives()


if __name__ == "__main__":
    main()