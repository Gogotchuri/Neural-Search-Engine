"""MLM prediction head - training scaffold that's discarded after pretraining.

Takes the encoder's per-token hidden states and produces a score for every
word in the vocabulary at each position. The highest-scoring word is the
model's guess for what was masked. This forces the encoder to build
representations that actually understand language.

Discarded after pretraining - only the encoder is kept for fine-tuning.
"""

import torch.nn as nn
from torch import Tensor


class MLMHead(nn.Module):
    """Hidden state (384) -> vocabulary logits (30k).

    Architecture: Linear(H->H) -> GELU -> LayerNorm(H) -> Linear(H->V)

    With weight tying, the final projection shares the encoder's embedding
    matrix - so "word -> vector" (embedding) and "vector -> word scores"
    (projection) use the same parameters. Gradients from MLM loss improve
    the embeddings directly.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        # Adapter: transforms encoder's general-purpose hidden state into
        # a representation suited for word prediction
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        # Scores each word: dot product between transformed state and word embeddings
        # With tie_weights(), the embedding matrix (transposed)
        self.projection = nn.Linear(hidden_dim, vocab_size, bias=False)

    def tie_weights(self, embedding_weight: Tensor) -> None:
        """Make projection share memory with the encoder's token embedding.

        After this, projection.weight and encoder.token_emb.weight are the
        same tensor - updates to one update both.
        """
        self.projection.weight = embedding_weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (B, L, H) from encoder.encode_tokens()

        Returns:
            (B, L, vocab_size) - per-position score for each word in vocabulary
        """
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.norm(x)
        x = self.projection(x)
        return x
