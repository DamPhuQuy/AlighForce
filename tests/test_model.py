import pytest
import torch
import torch.nn.functional as F
from alignforge.infrastructure.models import AlignForgeModel


def test_model_parameter_count():
    model = AlignForgeModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
    )
    total_params = model.count_parameters()
    # Should be around 3.67M parameters with tied embeddings
    assert 3_000_000 < total_params < 4_500_000
    print(f"AlignForge model parameters: {total_params:,}")


def test_forward_output_shape():
    model = AlignForgeModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
    )
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 2048, (batch_size, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (batch_size, seq_len, 2048)


def test_strict_causality_masking():
    """Verify that future tokens do not affect past token logits (no attention leakage)."""
    model = AlignForgeModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.0,
    )
    model.eval()

    prefix = torch.tensor([[10, 25, 33, 47, 58]])
    # Variant A: appends token 100
    seq_a = torch.cat([prefix, torch.tensor([[100]])], dim=1)
    # Variant B: appends token 200
    seq_b = torch.cat([prefix, torch.tensor([[200]])], dim=1)

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # Prefix logits (positions 0 to 4) MUST be identical
    diff = (logits_a[:, :5, :] - logits_b[:, :5, :]).abs().max().item()
    assert diff < 1e-5, f"Causality leak detected! Max diff on past tokens: {diff}"

    # Logits at the final position (position 5) should differ because input token differed
    diff_final = (logits_a[:, 5, :] - logits_b[:, 5, :]).abs().max().item()
    assert diff_final > 1e-4


def test_autograd_and_gradient_flow():
    """Verify loss computation and backward pass produces finite gradients across all weights."""
    model = AlignForgeModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.0,
    )
    model.train()

    input_ids = torch.randint(0, 2048, (2, 8))
    targets = torch.randint(0, 2048, (2, 8))

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 2048), targets.view(-1))
    loss.backward()

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"


def test_tiny_batch_overfitting():
    """Verify that the model can rapidly overfit a small batch, proving gradient validity."""
    torch.manual_seed(42)
    model = AlignForgeModel(
        vocab_size=128,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # 4 distinct sequences of length 8
    input_ids = torch.randint(4, 120, (4, 8))
    # Targets are next-token shifted
    targets = torch.roll(input_ids, shifts=-1, dims=1)

    initial_loss = None
    final_loss = None

    for step in range(80):
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 128), targets.view(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss > 4.0, f"Expected high initial loss, got {initial_loss}"
    assert final_loss < 0.1, f"Expected model to overfit below 0.1, got {final_loss}"
    print(f"Overfitting test: initial loss={initial_loss:.4f} -> final loss={final_loss:.4f}")

