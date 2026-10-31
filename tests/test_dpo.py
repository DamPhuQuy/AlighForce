import math
import pytest
import torch
from alignforge.infrastructure.training import get_batch_logps, dpo_loss


def test_get_batch_logps_masking():
    """Verify that prompt tokens (-100) are completely excluded from logps calculation."""
    B, T, V = 2, 8, 64
    logits = torch.randn(B, T, V)

    # Sequence with first 4 tokens masked (-100) and remaining 4 labeled
    labels = torch.tensor([
        [-100, -100, -100, -100, 10, 20, 30, 40],
        [-100, -100, 5, 15, 25, -100, -100, -100],
    ])

    logps = get_batch_logps(logits, labels)
    assert logps.shape == (B,)
    assert not torch.isnan(logps).any()
    assert not torch.isinf(logps).any()

    # Changing logits at the masked prompt positions (position 0, 1, 2) must NOT change logps
    mutated_logits = logits.clone()
    mutated_logits[:, 0:2, :] += 50.0  # Mutate first 2 positions
    mutated_logps = get_batch_logps(mutated_logits, labels)

    diff = (logps - mutated_logps).abs().max().item()
    assert diff < 1e-5, f"Masked prompt tokens leaked into logps! Diff: {diff}"


def test_dpo_loss_properties():
    """Verify standard mathematical properties of the DPO objective."""
    # Case 1: Policy identical to Reference -> margin is 0, loss is -log(sigmoid(0)) = log(2) ~= 0.6931
    pi_c = torch.tensor([-2.0, -1.5])
    pi_r = torch.tensor([-4.0, -3.5])
    ref_c = torch.tensor([-2.0, -1.5])
    ref_r = torch.tensor([-4.0, -3.5])

    loss_neutral, margin_neutral, win_rate = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert abs(loss_neutral.item() - math.log(2)) < 1e-4
    assert abs(margin_neutral.item()) < 1e-4

    # Case 2: Policy strictly improves preference over reference (chosen logps increase, rejected decrease)
    pi_c_better = torch.tensor([-1.0, -0.8])
    pi_r_worse = torch.tensor([-6.0, -5.5])

    loss_improved, margin_improved, win_rate_better = dpo_loss(
        pi_c_better, pi_r_worse, ref_c, ref_r, beta=0.1
    )
    assert loss_improved < loss_neutral
    assert margin_improved > 0.0
    assert win_rate_better == 1.0


def test_dpo_gradient_direction():
    """Verify that minimizing DPO loss pushes chosen log-probs up and rejected log-probs down."""
    pi_c = torch.tensor([-3.0], requires_grad=True)
    pi_r = torch.tensor([-3.0], requires_grad=True)
    ref_c = torch.tensor([-3.0])
    ref_r = torch.tensor([-3.0])

    loss, _, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=1.0)
    loss.backward()

    # Gradient of loss w.r.t pi_c should be negative (moving in negative gradient direction increases pi_c)
    assert pi_c.grad.item() < 0.0, "Expected negative gradient for chosen logps"
    # Gradient of loss w.r.t pi_r should be positive (moving in negative gradient direction decreases pi_r)
    assert pi_r.grad.item() > 0.0, "Expected positive gradient for rejected logps"
