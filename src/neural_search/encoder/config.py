from dataclasses import dataclass


@dataclass
class EncoderConfig:
    vocab_size: int = 30_000
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6 # number of transformer layers
    ffn_dim: int = 1536  # 4x hidden_dim
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0
