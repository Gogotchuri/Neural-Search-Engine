import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Applies two linear transformations with a GELU activation in between:
        FFN(x) = fc2(GELU(fc1(x)))

    The intermediate dimension (ffn_dim) - Per original paper 4x the hidden dims work well
    So we use double linear layer first fc1 to essentially expand hidden_dim -> ffn_dim and the fc2 to compress back
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        # Main reason for GELU instead of RELU is smooth approximation that doesn't zero out the negative values
        self.act = nn.GELU() # Gaussian Error Linear Unit is the new standard for transformers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # (B, L, hidden_dim) -> (B, L, ffn_dim) -> (B, L, hidden_dim)
        return self.dropout(self.fc2(self.act(self.fc1(x))))
