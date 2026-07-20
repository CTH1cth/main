from types import SimpleNamespace

import torch

from common.source_arbiter import (
    DirectionalUtilityTarget,
    compute_sign_aware_arbiter_loss,
)


def _target(positive_valid, negative_valid):
    shape = tuple(positive_valid.shape)
    values = torch.tensor([0.8, 0.2, 0.8, 0.2]).reshape(shape).double()
    ones = torch.ones(shape, dtype=torch.float64)
    bool_ones = torch.ones(shape, dtype=torch.bool)
    valid = positive_valid | negative_valid
    return DirectionalUtilityTarget(
        target_teacher=values,
        valid=valid,
        weight=ones,
        positive_valid=positive_valid,
        negative_valid=negative_valid,
        teacher_hard=positive_valid.double(),
        dabe_hard=(~positive_valid).double(),
        dabe_valid=bool_ones,
        consensus=ones * 0.5,
        cross_view_difference=ones * 0.0,
        future_move=ones * 0.2,
        semantic_probability=ones * 0.5,
        semantic_move=ones * 0.2,
        direction_agreement=bool_ones,
        source_disagreement=bool_ones,
        teacher_advantage=values - 0.5,
        age_valid=bool_ones,
        history_valid=bool_ones,
        prototype_valid=bool_ones,
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


def test_branch_balancing_is_independent_for_positive_and_negative_signs():
    positive = torch.tensor([[[[True, True, False, False]]]])
    negative = ~positive
    raw_positive = torch.zeros((1, 1, 1, 4), dtype=torch.float64, requires_grad=True)
    raw_negative = torch.zeros_like(raw_positive, requires_grad=True)
    loss, stats = compute_sign_aware_arbiter_loss(
        old_router_output={
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
        },
        current_gate_positive=torch.full_like(raw_positive, 0.5),
        current_gate_negative=torch.full_like(raw_negative, 0.5),
        teacher_prior=0.5,
        utility_target=_target(positive, negative),
        dino_margin=torch.zeros_like(raw_positive),
        train_image_mask=torch.tensor([True]),
        cfg=_cfg(),
    )
    assert torch.isfinite(loss)
    assert stats["active_sign_branches"] == 2.0
    assert stats["positive_branch_teacher_preferred_count"] == 1.0
    assert stats["positive_branch_dabe_preferred_count"] == 1.0
    assert stats["negative_branch_teacher_preferred_count"] == 1.0
    assert stats["negative_branch_dabe_preferred_count"] == 1.0
    loss.backward()
    assert raw_positive.grad is not None
    assert raw_negative.grad is not None


def test_empty_sign_branch_keeps_other_branch_finite():
    positive = torch.zeros((1, 1, 1, 4), dtype=torch.bool)
    negative = torch.ones_like(positive)
    raw_positive = torch.zeros((1, 1, 1, 4), requires_grad=True)
    raw_negative = torch.zeros_like(raw_positive, requires_grad=True)
    loss, stats = compute_sign_aware_arbiter_loss(
        old_router_output={
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
        },
        current_gate_positive=torch.full_like(raw_positive, 0.5),
        current_gate_negative=torch.full_like(raw_negative, 0.5),
        teacher_prior=0.5,
        utility_target=_target(positive, negative),
        dino_margin=torch.zeros_like(raw_positive),
        train_image_mask=torch.tensor([True]),
        cfg=_cfg(),
    )
    assert torch.isfinite(loss)
    assert stats["active_sign_branches"] == 1.0
    assert stats["positive_branch_valid_pixels"] == 0.0
    assert stats["negative_branch_valid_pixels"] == 4.0


def test_branch_class_weights_are_fixed_external_values():
    positive = torch.tensor([[[[True, True, False, False]]]])
    negative = ~positive
    raw_positive = torch.zeros((1, 1, 1, 4), dtype=torch.float64)
    raw_negative = torch.zeros_like(raw_positive)
    _, stats = compute_sign_aware_arbiter_loss(
        old_router_output={
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
        },
        current_gate_positive=torch.full_like(raw_positive, 0.5),
        current_gate_negative=torch.full_like(raw_negative, 0.5),
        teacher_prior=0.5,
        utility_target=_target(positive, negative),
        dino_margin=torch.zeros_like(raw_positive),
        train_image_mask=torch.tensor([True]),
        cfg=_cfg(),
        class_weights={
            "positive": {"teacher": 2.5, "dabe": 0.75},
            "negative": {"teacher": 1.75, "dabe": 0.60},
        },
    )
    assert stats["positive_branch_teacher_class_weight"] == 2.5
    assert stats["positive_branch_dabe_class_weight"] == 0.75
    assert stats["negative_branch_teacher_class_weight"] == 1.75
    assert stats["negative_branch_dabe_class_weight"] == 0.60
