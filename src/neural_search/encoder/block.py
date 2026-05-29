import torch.nn as nn
from torch import Tensor

from neural_search.encoder.attention import MultiHeadAttention
from neural_search.encoder.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Single pre-norm transformer encoder block. With residual and dropout one each sublayer

    Note: residual values in the flow below do not change anything, we simply take the values at that point and add them at the next (+)
    Data flow:
        x -> (residual value 1) -> LayerNorm -> MultiHeadAttention -> Dropout -> (+) ->

        (residual value 2) -> LayerNorm -> FeedForward -> Dropout -> (+) -> output
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        # Pre-norm attention sub-layer with residual
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        # Pre-norm FFN sub-layer with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
