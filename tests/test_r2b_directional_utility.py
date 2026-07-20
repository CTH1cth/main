from types import SimpleNamespace

import torch

from common.source_arbiter import build_directional_utility_target


def _cfg():
    return SimpleNamespace(
        SOURCE_ARBITER_UTILITY_DABE_FG_THRESH=0.70,
        SOURCE_ARBITER_UTILITY_DABE_BG_THRESH=0.30,
        SOURCE_ARBITER_UTILITY_DABE_WEIGHT_THRESH=0.70,
        SOURCE_ARBITER_UTILITY_USE_FG_CORE=True,
        SOURCE_ARBITER_UTILITY_USE_BG_CORE=True,
        SOURCE_ARBITER_MIN_MEMORY_AGE_EPOCH=1,
        SOURCE_ARBITER_MAX_MEMORY_AGE_EPOCH=2,
        SOURCE_ARBITER_UTILITY_MIN_HISTORY=3,
        SOURCE_ARBITER_UTILITY_MAX_VIEW_DIFF=0.15,
        SOURCE_ARBITER_UTILITY_MIN_FUTURE_MOVE=0.03,
        SOURCE_ARBITER_UTILITY_MIN_SEMANTIC_MOVE=0.05,
        SOURCE_ARBITER_UTILITY_REQUIRE_DIRECTION_AGREEMENT=True,
        SOURCE_ARBITER_DINO_MARGIN_TAU=0.05,
        SOURCE_ARBITER_UTILITY_TAU=0.05,
        SOURCE_ARBITER_UTILITY_VIEW_TAU=0.05,
        SOURCE_ARBITER_UTILITY_FUTURE_WEIGHT_SCALE=0.15,
        SOURCE_ARBITER_UTILITY_SEMANTIC_WEIGHT_SCALE=0.25,
    )


def test_directional_target_covers_both_teacher_signs_and_preferences():
    old_teacher = torch.tensor(
        [[[[0.8, 0.8, 0.2, 0.2]]]], dtype=torch.float64
    )
    old_student = torch.full_like(old_teacher, 0.5)
    consensus = torch.tensor(
        [[[[0.7, 0.3, 0.3, 0.7]]]], dtype=torch.float64
    )
    dabe_target = torch.tensor(
        [[[[0.0, 0.0, 1.0, 1.0]]]], dtype=torch.float64
    )
    positive_margin = 0.05 * torch.log(torch.tensor(4.0)).item()
    margin = torch.tensor(
        [[[[positive_margin, -positive_margin, -positive_margin, positive_margin]]]],
        dtype=torch.float64,
    )
    empty = torch.zeros_like(old_teacher, dtype=torch.bool)
    result = build_directional_utility_target(
        old_teacher_prob=old_teacher,
        old_student_prob=old_student,
        evaluator_prob_weak=consensus,
        evaluator_prob_flip=consensus,
        old_history_count=torch.tensor([6]),
        old_epoch=torch.tensor([6]),
        current_epoch=7,
        old_valid=torch.tensor([True]),
        dabe_target=dabe_target,
        dabe_weight=torch.ones_like(old_teacher),
        masks={"fg_core": empty, "bg_core": empty},
        dino_margin=margin,
        prototype_valid=torch.tensor([True]),
        cfg=_cfg(),
    )
    assert result.valid.all()
    assert torch.equal(
        result.positive_valid,
        torch.tensor([[[[True, True, False, False]]]]),
    )
    assert torch.equal(
        result.negative_valid,
        torch.tensor([[[[False, False, True, True]]]]),
    )
    assert result.target_teacher[0, 0, 0, 0] > 0.5
    assert result.target_teacher[0, 0, 0, 1] < 0.5
    assert result.target_teacher[0, 0, 0, 2] > 0.5
    assert result.target_teacher[0, 0, 0, 3] < 0.5
    assert not result.target_teacher.requires_grad
    assert not result.weight.requires_grad


def test_direction_disagreement_rejects_candidate():
    shape = (1, 1, 1, 1)
    old_teacher = torch.full(shape, 0.8)
    old_student = torch.full(shape, 0.5)
    future = torch.full(shape, 0.7)
    empty = torch.zeros(shape, dtype=torch.bool)
    result = build_directional_utility_target(
        old_teacher_prob=old_teacher,
        old_student_prob=old_student,
        evaluator_prob_weak=future,
        evaluator_prob_flip=future,
        old_history_count=torch.tensor([6]),
        old_epoch=torch.tensor([6]),
        current_epoch=7,
        old_valid=torch.tensor([True]),
        dabe_target=torch.zeros(shape),
        dabe_weight=torch.ones(shape),
        masks={"fg_core": empty, "bg_core": empty},
        dino_margin=torch.full(shape, -0.1),
        prototype_valid=torch.tensor([True]),
        cfg=_cfg(),
    )
    assert not result.valid.any()
    assert result.weight.sum().item() == 0.0
