import tempfile
from pathlib import Path
import pytest
import torch
from alignforge.infrastructure.models import (
    AlignForgeModel,
    apply_lora_to_model,
    get_lora_state_dict,
    save_lora_checkpoint,
    load_lora_checkpoint,
)


def test_lora_initialization_identity():
    """Verify that attaching LoRA produces zero change to model output at step 0."""
    torch.manual_seed(42)
    model = AlignForgeModel(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        base_logits, _ = model(input_ids)

    # Attach LoRA to Wq and Wv
    stats = apply_lora_to_model(model, r=8, alpha=16.0, target_modules=("q_proj", "v_proj"))

    with torch.no_grad():
        lora_logits, _ = model(input_ids)

    # Must be identical because B is initialized to 0
    diff = (lora_logits - base_logits).abs().max().item()
    assert diff < 1e-5, f"LoRA initialization changed outputs! Max diff: {diff}"


def test_lora_parameter_freezing_and_reduction():
    """Verify base weights are frozen and trainable parameters represent < 10% of total."""
    model = AlignForgeModel(
        vocab_size=2048,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
    )
    stats = apply_lora_to_model(model, r=8, alpha=16.0)

    trainable = stats["trainable_params"]
    total = stats["total_params"]

    # Parameter reduction: trainable must be less than 5% of total
    param_pct = (trainable / total) * 100.0
    print(f"LoRA trainable parameters: {trainable:,} ({param_pct:.2f}%)")
    assert param_pct < 5.0, f"Expected <5% trainable parameters, got {param_pct:.2f}%"

    # Verify base weights are frozen
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_lora_gradient_flow():
    """Verify backprop updates LoRA parameters while base weights receive no gradients."""
    model = AlignForgeModel(
        vocab_size=128,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
    )
    apply_lora_to_model(model, r=4, alpha=8.0)
    model.train()

    input_ids = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))

    logits, _ = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 128), targets.view(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
        else:
            assert param.grad is None, f"Frozen parameter {name} received gradient!"


def test_lora_save_and_load(tmp_path):
    """Verify saving adapter weights to disk and loading them into another instance with same base weights."""
    torch.manual_seed(42)
    model1 = AlignForgeModel(vocab_size=128, d_model=64, n_heads=2, n_layers=2, d_ff=128)
    apply_lora_to_model(model1, r=4, alpha=8.0)

    # Mutate lora weights
    with torch.no_grad():
        for param in model1.parameters():
            if param.requires_grad:
                param.add_(0.5)

    ckpt_path = tmp_path / "lora_test.pt"
    save_lora_checkpoint(model1, ckpt_path)

    # Create model2 with identical base weights
    torch.manual_seed(42)
    model2 = AlignForgeModel(vocab_size=128, d_model=64, n_heads=2, n_layers=2, d_ff=128)
    apply_lora_to_model(model2, r=4, alpha=8.0)
    load_lora_checkpoint(model2, ckpt_path)

    model1.eval()
    model2.eval()

    input_ids = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        out1, _ = model1(input_ids)
        out2, _ = model2(input_ids)

    assert torch.allclose(out1, out2, atol=1e-5)
