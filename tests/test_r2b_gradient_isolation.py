from types import SimpleNamespace

import torch
import torch.nn as nn

from common.source_arbiter import (
    DirectionalUtilityTarget,
    SignAwareSourceArbiter,
    apply_sign_aware_loss_space_arbitration,
    compute_sign_aware_arbiter_loss,
    compute_sign_aware_source_gates,
)


def _utility_target(shape):
    valid = torch.ones(shape, dtype=torch.bool)
    positive = torch.zeros(shape, dtype=torch.bool)
    positive[..., : shape[-1] // 2] = True
    negative = valid & (~positive)
    target = torch.where(
        torch.arange(shape[-1]).reshape(1, 1, 1, -1) % 2 == 0,
        torch.full(shape, 0.8),
        torch.full(shape, 0.2),
    ).double()
    ones = torch.ones(shape, dtype=torch.float64)
    return DirectionalUtilityTarget(
        target_teacher=target,
        valid=valid,
        weight=ones,
        positive_valid=positive,
        negative_valid=negative,
        teacher_hard=positive.double(),
        dabe_hard=negative.double(),
        dabe_valid=valid,
        consensus=ones * 0.5,
        cross_view_difference=ones * 0.0,
        future_move=ones * 0.2,
        semantic_probability=ones * 0.5,
        semantic_move=ones * 0.2,
        direction_agreement=valid,
        source_disagreement=valid,
        teacher_advantage=target - 0.5,
        age_valid=valid,
        history_valid=valid,
        prototype_valid=valid,
    )


def _cfg():
    return SimpleNamespace(
        SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MIN=0.5,
        SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MAX=4.0,
        SOURCE_ARBITER_LAMBDA_UTILITY=1.0,
        SOURCE_ARBITER_LAMBDA_PRIOR=0.10,
        SOURCE_ARBITER_LAMBDA_MASS=0.01,
        SOURCE_ARBITER_LAMBDA_SMOOTH=0.005,
        SOURCE_ARBITER_MASS_TOLERANCE=0.30,
        SOURCE_ARBITER_SMOOTH_KAPPA=5.0,
    )


def test_segmentation_backward_does_not_reach_router():
    torch.manual_seed(3407)
    student = nn.Conv2d(3, 1, 1).double()
    router = SignAwareSourceArbiter().double()
    logits = student(torch.randn(2, 3, 5, 5, dtype=torch.float64))
    teacher_prob = torch.rand_like(logits)
    teacher_target = (teacher_prob >= 0.5).double()
    dabe_target = torch.rand_like(logits)
    output = router(torch.randn(2, 18, 5, 5, dtype=torch.float64))
    gates = compute_sign_aware_source_gates(
        output,
        teacher_target,
        0.5,
        teacher_prob,
        dabe_target,
        1.0,
    )
    loss = apply_sign_aware_loss_space_arbitration(
        logits,
        dabe_target,
        torch.ones_like(logits),
        teacher_target,
        gates["gate_dabe"],
        gates["gate_teacher"],
        1.0,
    )["loss"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in router.parameters())


def test_router_backward_does_not_reach_student_or_evaluator():
    torch.manual_seed(3407)
    student = nn.Conv2d(3, 1, 1).double()
    evaluator = nn.Conv2d(3, 1, 1).double()
    router = SignAwareSourceArbiter().double()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)
    shape = (2, 1, 5, 6)
    student_prob = student(
        torch.randn(2, 3, 5, 6, dtype=torch.float64)
    ).sigmoid().detach()
    with torch.no_grad():
        evaluator_prob = evaluator(
            torch.randn(2, 3, 5, 6, dtype=torch.float64)
        ).sigmoid()
    output = router(torch.randn(2, 18, 5, 6, dtype=torch.float64))
    gates = compute_sign_aware_source_gates(
        output,
        (evaluator_prob >= 0.5).double(),
        0.5,
        evaluator_prob,
        student_prob,
        1.0,
    )
    loss, _ = compute_sign_aware_arbiter_loss(
        old_router_output=output,
        current_gate_positive=gates["gate_teacher_positive"],
        current_gate_negative=gates["gate_teacher_negative"],
        teacher_prior=0.5,
        utility_target=_utility_target(shape),
        dino_margin=torch.zeros(shape, dtype=torch.float64),
        train_image_mask=torch.tensor([True, True]),
        cfg=_cfg(),
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in evaluator.parameters())
    assert router.head.weight.grad is not None
    assert router.head.weight.grad.abs().sum().item() > 0.0
