from types import SimpleNamespace

import torch
import torch.nn as nn

from common.source_arbiter import (
    SourceArbiter,
    UtilityTarget,
    apply_loss_space_arbitration,
    compute_source_arbiter_loss,
    compute_source_gates,
)


def test_segmentation_and_router_gradients_are_isolated():
    torch.manual_seed(3407)
    student = nn.Conv2d(3, 1, 1).double()
    router = SourceArbiter().double()
    x = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    evidence = torch.randn(2, 18, 5, 5, dtype=torch.float64)
    logits = student(x)
    residual = router(evidence.detach())
    gate_dabe, gate_teacher = compute_source_gates(
        residual,
        0.5,
        torch.ones_like(residual),
        1.0,
    )
    target = torch.rand_like(logits)
    seg = apply_loss_space_arbitration(
        logits,
        target,
        torch.ones_like(logits),
        (target >= 0.5).double(),
        torch.ones_like(logits),
        gate_dabe,
        gate_teacher,
        1.0,
    )["loss"]
    seg.backward()
    assert all(
        parameter.grad is None or parameter.grad.abs().max().item() == 0.0
        for parameter in router.parameters()
    )
    student_grad = [parameter.grad.clone() for parameter in student.parameters()]

    router.zero_grad(set_to_none=True)
    utility = UtilityTarget(
        target_teacher=torch.full_like(residual, 0.8),
        valid=torch.ones_like(residual, dtype=torch.bool),
        weight=torch.ones_like(residual),
        consensus=torch.full_like(residual, 0.8),
        consensus_confidence=torch.ones_like(residual),
        cross_view_difference=torch.zeros_like(residual),
        source_disagreement=torch.ones_like(residual),
        utility_gap=torch.ones_like(residual),
        delta_target=torch.ones_like(residual),
        age_valid=torch.ones_like(residual, dtype=torch.bool),
    )
    cfg = SimpleNamespace()
    router_loss, _ = compute_source_arbiter_loss(
        residual,
        gate_teacher,
        0.5,
        utility,
        torch.zeros_like(residual),
        torch.ones(2, dtype=torch.bool),
        cfg,
    )
    router_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in router.parameters()
    )
    for before, parameter in zip(student_grad, student.parameters()):
        assert torch.equal(before, parameter.grad)


def test_router_loss_is_finite_with_no_valid_utility_pixels():
    router = SourceArbiter().double()
    residual = router(torch.randn(2, 18, 5, 5, dtype=torch.float64))
    utility = UtilityTarget(
        target_teacher=torch.full_like(residual, 0.5),
        valid=torch.zeros_like(residual, dtype=torch.bool),
        weight=torch.zeros_like(residual),
        consensus=torch.full_like(residual, 0.5),
        consensus_confidence=torch.zeros_like(residual),
        cross_view_difference=torch.zeros_like(residual),
        source_disagreement=torch.zeros_like(residual),
        utility_gap=torch.zeros_like(residual),
        delta_target=torch.zeros_like(residual),
        age_valid=torch.zeros_like(residual, dtype=torch.bool),
    )
    gate = torch.full_like(residual, 0.5)
    loss, stats = compute_source_arbiter_loss(
        residual,
        gate,
        0.5,
        utility,
        torch.zeros_like(residual),
        torch.ones(2, dtype=torch.bool),
        SimpleNamespace(),
    )
    assert torch.isfinite(loss)
    assert stats["utility_valid_ratio"] == 0.0
