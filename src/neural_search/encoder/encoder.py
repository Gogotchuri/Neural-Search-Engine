import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from neural_search.encoder.block import TransformerBlock
from neural_search.encoder.config import EncoderConfig
from neural_search.encoder.positional import PositionalEncoding


class Encoder(nn.Module):
    """Transformer encoder that produces L2-normalized sentence embeddings.

    Takes token IDs and an attention mask, passes them through an embedding
    layer, sinusoidal positional encoding, a stack of pre-norm transformer
    blocks, and finally applies masked mean pooling + L2 normalization to
    produce a fixed-size vector for each input sequence.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Embedding layer from nn - Simple lookup table from token ID -> vector identity
        self.token_emb = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id
        )
        # Positional sinusoid encoding and it's dropout layer
        self.pos_enc = PositionalEncoding(config.hidden_dim, config.max_seq_len)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Tranformer {num_layers} layers
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_dim, config.num_heads, config.ffn_dim, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )
        # Final normalization to account for accumulated residual effect
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight)
                # He et al., "Delving Deep into Rectifiers" (ICCV 2015).
                # For deep neural nets should outperform xavier in convergence speed
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Based on the GPT-2 paper the std of 1/sqrt(H) (H for residual layers) works well for large nets
                # Since we are applying residual twice, 1/sqrt(2*H)
                # Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
                std = 1 / self.config.hidden_dim ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            input_ids:      (B, L) - token indices
            attention_mask: (B, L) - 1 for real tokens, 0 for padding

        Returns:
            (B, hidden_dim) - L2-normalized sentence embeddings
        """
        # Token embeddings (scaled to match positional encoding magnitude)
        x = self.token_emb(input_ids) * math.sqrt(self.config.hidden_dim)  # (B, L, H)
        x = self.pos_enc(x)                   # (B, L, H)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)       # (B, L, H)

        x = self.final_norm(x)                # (B, L, H)
        # https://ar5iv.labs.arxiv.org/html/1908.10084 (SBert)
        # The paper demonstrates that the mean pooling layer is better than the other pooling alternatives
        # Masked mean pooling: average only over real (non-padding) tokens
        mask = attention_mask.unsqueeze(-1)    # (B, L, 1)
        x = (x * mask).sum(dim=1)             # (B, H)
        x = x / mask.sum(dim=1).clamp(min=1)  # (B, H)

        # L2 normalize onto the unit sphere surface, so the cosine similarity is equivalent to the dot product
        x = nn.functional.normalize(x, dim=-1)

        return x

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        tokenizer,
        batch_size: int = 32,
        device: str = "cpu",
        max_len: int = 256,
    ) -> np.ndarray:
        """Encode a list of strings into L2-normalized embeddings.

        Convenience method for the search pipeline - handles tokenization,
        batching, and padding internally.

        Args:
            texts:      list of N strings to encode
            tokenizer:  a HuggingFace ``tokenizers.Tokenizer`` instance
            batch_size: inference batch size
            device:     "cpu" or "cuda"
            max_len:    maximum sequence length (truncates longer inputs)

        Returns:
            numpy array of shape (N, hidden_dim), L2-normalized
        """
        self.eval()
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encodings = tokenizer.encode_batch(batch_texts)

            # Truncate and collect IDs
            ids_list = [enc.ids[:max_len] for enc in encodings]

            # Pad to the longest sequence in this batch
            max_batch_len = max(len(ids) for ids in ids_list)
            input_ids = torch.zeros(len(ids_list), max_batch_len, dtype=torch.long)
            attention_mask = torch.zeros(len(ids_list), max_batch_len, dtype=torch.float)

            for i, ids in enumerate(ids_list):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1.0

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Run the model
            emb = self(input_ids, attention_mask)  # (batch, hidden_dim)
            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
