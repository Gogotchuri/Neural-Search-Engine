"""Component-level verification for the transformer encoder.

Run with:  uv run python scripts/test_encoder.py
"""

import sys

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f"  - {detail}"
        print(msg)


def test_encoder():
    print("\n=== Phase 3: Full Encoder ===\n")
    from neural_search.encoder.config import EncoderConfig
    from neural_search.encoder.encoder import Encoder

    torch.manual_seed(42)
    config = EncoderConfig()  # default: 384 hidden, 6 heads, 6 layers
    encoder = Encoder(config)
    encoder.eval()

    B, L = 4, 32
    input_ids = torch.randint(1, config.vocab_size, (B, L))
    mask = torch.ones(B, L)
    mask[0, 20:] = 0
    mask[1, 25:] = 0

    out = encoder(input_ids, mask)

    # Test 1: output shape is (B, hidden_dim)
    check(
        "Encoder output shape == (B, hidden_dim)",
        out.shape == (B, config.hidden_dim),
        f"got {tuple(out.shape)}",
    )

    # Test 2: L2 norm ~= 1.0 for all embeddings
    norms = out.norm(dim=-1)
    check(
        "All embeddings have unit L2 norm",
        torch.allclose(norms, torch.ones(B), atol=1e-5),
        f"norms = {norms.tolist()}",
    )

    # Test 3: identical input with different padding -> same embedding
    # Create two copies of the same tokens, one padded to 32, one to 48
    tokens = torch.randint(1, config.vocab_size, (1, 20))

    ids_short = torch.zeros(1, 32, dtype=torch.long)
    ids_short[0, :20] = tokens
    mask_short = torch.zeros(1, 32)
    mask_short[0, :20] = 1.0

    ids_long = torch.zeros(1, 48, dtype=torch.long)
    ids_long[0, :20] = tokens
    mask_long = torch.zeros(1, 48)
    mask_long[0, :20] = 1.0

    emb_short = encoder(ids_short, mask_short)
    emb_long = encoder(ids_long, mask_long)

    pad_diff = (emb_short - emb_long).abs().max().item()
    check(
        "Same input + different padding -> identical embedding",
        torch.allclose(emb_short, emb_long, atol=1e-5),
        f"max diff = {pad_diff:.2e}",
    )

    # Test 4: parameter count in expected range (~20M)
    n_params = sum(p.numel() for p in encoder.parameters())
    check(
        "Parameter count ~20M (15M-25M range)",
        15_000_000 < n_params < 25_000_000,
        f"actual = {n_params:,}",
    )


# Training (overfit-one-batch)
def test_overfit():
    print("\n=== Phase 4: Overfit One Batch ===\n")
    from neural_search.encoder.config import EncoderConfig
    from neural_search.encoder.encoder import Encoder
    from neural_search.losses import infonce_loss

    # Use a smaller encoder for speed (2 layers instead of 6)
    torch.manual_seed(6)
    config = EncoderConfig(num_layers=2, hidden_dim=128, num_heads=4, ffn_dim=512)
    encoder = Encoder(config)
    encoder.train()

    # Synthetic batch: 8 query-passage pairs with random token IDs
    # Using random here for simplicity, we should still be able to overfit the random data
    B, Q_LEN, P_LEN = 8, 16, 32
    query_ids = torch.randint(1, config.vocab_size, (B, Q_LEN))
    query_mask = torch.ones(B, Q_LEN)
    pos_ids = torch.randint(1, config.vocab_size, (B, P_LEN))
    pos_mask = torch.ones(B, P_LEN)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=5e-4)

    # Train 200 steps on the same batch
    losses = []
    for step in range(200):
        query_emb = encoder(query_ids, query_mask)
        pos_emb = encoder(pos_ids, pos_mask)
        loss = infonce_loss(query_emb, pos_emb, temperature=0.05)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    first_loss = losses[0]
    final_loss = losses[-1]

    check(
        "Loss decreases during training",
        final_loss < first_loss,
        f"first={first_loss:.4f}, final={final_loss:.4f}",
    )
    check(
        "Overfit-one-batch: final loss < 0.1",
        final_loss < 0.1,
        f"final loss = {final_loss:.4f}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_encoder()
    test_overfit()
    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL > 0 else 0)
