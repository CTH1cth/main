import torch
import torch.nn as nn

from common.source_arbiter import (
    SourceArbiter,
    apply_loss_space_arbitration,
    compute_r1_shadow_stats,
    compute_source_gates,
)


def test_r2_segmentation_and_router_gradients_remain_isolated():
    torch.manual_seed(3407)
    student = nn.Conv2d(3, 1, 1).double()
    router = SourceArbiter().double()
    evaluator = nn.Conv2d(3, 1, 1).double()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    logits = student(torch.randn(2, 3, 5, 5, dtype=torch.float64))
    residual = router(torch.randn(2, 18, 5, 5, dtype=torch.float64))
    with torch.no_grad():
        evaluator_target = evaluator(
            torch.randn(2, 3, 5, 5, dtype=torch.float64)
        ).sigmoid()
    gate_dabe, gate_teacher = compute_source_gates(
        residual,
        0.5,
        torch.ones_like(residual),
        1.0,
    )
    target = evaluator_target.detach()
    loss = apply_loss_space_arbitration(
        logits,
        target,
        torch.ones_like(logits),
        (target >= 0.5).double(),
        torch.ones_like(logits),
        gate_dabe,
        gate_teacher,
        1.0,
    )["loss"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in router.parameters())
    assert all(parameter.grad is None for parameter in evaluator.parameters())


def test_router_backward_does_not_update_student_or_frozen_evaluator():
    torch.manual_seed(3407)
    student = nn.Conv2d(3, 1, 1).double()
    router = SourceArbiter().double()
    evaluator = nn.Conv2d(3, 1, 1).double()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)

    student_prob = student(
        torch.randn(2, 3, 5, 5, dtype=torch.float64)
    ).sigmoid().detach()
    with torch.no_grad():
        evaluator_prob = evaluator(
            torch.randn(2, 3, 5, 5, dtype=torch.float64)
        ).sigmoid()
    residual = router(torch.randn(2, 18, 5, 5, dtype=torch.float64))
    router_loss = (
        torch.sigmoid(residual) - 0.5 * (student_prob + evaluator_prob)
    ).square().mean()
    router_loss.backward()

    assert all(parameter.grad is None for parameter in student.parameters())
    assert any(parameter.grad is not None for parameter in router.parameters())
    assert all(parameter.grad is None for parameter in evaluator.parameters())


def test_r1_shadow_differs_only_through_teacher_source_weight():
    torch.manual_seed(3407)
    logits = torch.randn(2, 1, 5, 5, dtype=torch.float64)
    dabe_target = torch.rand_like(logits)
    teacher_target = (torch.rand_like(logits) >= 0.5).double()
    gates = (torch.full_like(logits, 0.4), torch.full_like(logits, 0.6))
    common = dict(
        logits=logits,
        dabe_target=dabe_target,
        dabe_weight=torch.rand_like(logits),
        teacher_target=teacher_target,
        gate_dabe=gates[0],
        gate_teacher=gates[1],
        source_sum=1.0,
    )
    raw = apply_loss_space_arbitration(
        **common, teacher_weight=torch.ones_like(logits)
    )
    shadow = apply_loss_space_arbitration(
        **common, teacher_weight=torch.full_like(logits, 0.35)
    )
    masks = {
        "fg_core": torch.ones_like(logits, dtype=torch.bool),
        "bg_core": torch.zeros_like(logits, dtype=torch.bool),
    }
    stats = compute_r1_shadow_stats([raw], [shadow], [1.0], masks)
    assert torch.equal(raw["dabe_loss_map"], shadow["dabe_loss_map"])
    assert stats["shadow_teacher_map_abs_delta"] > 0.0
    expected = 0.6 * (
        shadow["loss_teacher"].item() - raw["loss_teacher"].item()
    )
    assert abs(stats["shadow_total_group_delta"] - expected) <= 1e-12
