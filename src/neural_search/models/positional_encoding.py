from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.

    Adds a fixed, non-trainable position vector to each token embedding.

    Expected input shape:
        x: (batch_size, seq_len, hidden_dim)

    Output shape:
        same as input
    """

    def __init__(self, hidden_dim: int, max_length: int = 512) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )

        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)

        if hidden_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_length, hidden_dim)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch_size, seq_len, hidden_dim), got {x.shape}"
            )

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum supported "
                f"length {self.pe.size(1)}"
            )

        return x + self.pe[:, :seq_len, :]