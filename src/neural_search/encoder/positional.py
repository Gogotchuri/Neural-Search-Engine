import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'.

    Precomputes a (1, max_len, hidden_dim) table of sine/cosine values
    and adds it to the input embeddings. Stored as a non-learnable buffer.
    """

    def __init__(self, hidden_dim: int, max_len: int = 256):
        super().__init__()

        # pe shape: (1, max_len, hidden_dim) - the leading 1 broadcasts over batch
        pe = torch.zeros(1, max_len, hidden_dim)

        # position indices: (max_len, 1) - column vector for broadcasting
        position = torch.arange(max_len).unsqueeze(1).float()

        # div_term: 10000^(2i/d) computed in log-space
        # log(10000^(2i/d)) = 2i/d * log(10000), since we need 1/div_term for multiplication, we have minus in front of exponent
        # so the final form is exp(-2i/d * log(10000))
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        # Even indices get sin, odd indices get cos
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, hidden_dim) - only take the first L positions
        return x + self.pe[:, : x.size(1)]
