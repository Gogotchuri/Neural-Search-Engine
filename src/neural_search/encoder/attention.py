import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention as described in 'Attention Is All You Need'.

    The input is projected into queries, keys, and values, split across
    ``num_heads`` parallel attention heads, and recombined via a final
    output projection.

    The symbols and meaning:
    B - batch size
    L - sequence length
    h - number of heads
    d_h - per head dimension
    q - represents Query
    k - represents Key
    v - represents Value
    respectively W_q, W_k, W_v represent their weight matrices
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads

        # Four linear projections: Q, K, V, and output
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Stored for inspection during testing
        self._attn_weights: Tensor | None = None

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        x has B batches and each of length L. We must have each sequence the same length, so we use paddings.
        To ignore paddings, we have binary valued attention mask, having 0s on the positions of paddings
        Args:
            x:              (B, L, hidden_dim) — input token representations
            attention_mask: (B, L) — 1 for real tokens, 0 for padding

        Returns:
            (B, L, hidden_dim) — contextualized representations
        """
        B, L, _ = x.shape

        # Project to Q, K, V and reshape into (B, num_heads, L, d_h)
        # Each linear layer outputs (B, L, hidden_dim).
        # We view as (B, L, num_heads, d_h) then transpose to (B, num_heads, L, d_h).
        # This groups the dimensions by head so each head operates independently.
        q = self.W_q(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.num_heads, self.d_h).transpose(1, 2)

        # Input mask is (B, L). We need (B, 1, 1, L) so it broadcasts over
        # the num_heads dimension and the query-length dimension.
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        else:
            mask = None

        # Compute attention
        context, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )
        # Stored for debugging
        self._attn_weights = attn_weights.detach()

        # Concatenate heads and apply output projection, returns in the primary form
        # (B, num_heads, L, d_h) -> (B, L, num_heads, d_h) -> (B, L, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)

        return self.W_o(context)

def scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.
    This function considers the possibility that we are going to use it for cross-attention down the line
    Hence, the L_k (seq length of Key) and L_q (seq length of Query) might be different variables.
    For the case of regular self-attention, we can assume that L_q = L_k = L

    Args:
        query:   (B, h, L_q, d_h)
        key:     (B, h, L_k, d_h)
        value:   (B, h, L_k, d_h)
        mask:    (B, 1, 1, L_k) — 1 for real tokens, 0 for padding
        dropout: optional dropout applied to attention weights

    Returns:
        context:      (B, h, L_q, d_h)
        attn_weights: (B, h, L_q, L_k)
    """
    d_h = query.size(-1)

    # We do 4D matrix multiplication to get the scores' matrix.
    # Generally we can treat the first two dimensions as batch dimensions
    # and applying the dimension matchin rules to the last two dimensions.
    # Hence, to multiply [Q](L_q, d_h) @ [K](L_k, d_h), we need to transponse K on the last 2 dimensions
    # The final multiplication dimension will be:
    # (B, h, L_q, d_h) @ (B, h, d_h, L_k) -> (B, h, L_q, L_k)
    # And we normalize the whole thing by the square root of dimensions per head
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_h)

    if mask is not None:
        # mask shape: (B, 1, 1, L_k) broadcasts over heads and query positions
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # (B, h, L_q, L_k) @ (B, h, L_k, d_h) -> (B, h, L_q, d_h)
    context = torch.matmul(attn_weights, value)

    return context, attn_weights

