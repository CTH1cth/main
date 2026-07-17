from types import SimpleNamespace

import torch

from common.source_arbiter import (
    SourceArbiter,
    build_delayed_utility_target,
    compute_source_gates,
    get_arbiter_influence_scale,
    source_arbiter_parameter_count,
)


def test_zero_init_prior_and_complement_float64():
    router = SourceArbiter().double()
    evidence = torch.randn(2, 18, 9, 11, dtype=torch.float64)
    residual = router(evidence)
    assert residual.abs().max().item() == 0.0
    dabe, teacher = compute_source_gates(
        residual,
        teacher_prior=0.35,
        source_disagreement=torch.ones_like(residual),
        influence_scale=1.0,
    )
    assert torch.max(torch.abs(teacher - 0.35)).item() <= 1e-12
    assert torch.max(torch.abs(dabe + teacher - 1.0)).item() <= 1e-12
    assert source_arbiter_parameter_count(router) < 25000


def test_prior_extremes_and_disagreement_bypass():
    residual = torch.full((1, 1, 4, 4), 1.25, dtype=torch.float64)
    zero = torch.zeros_like(residual)
    _, teacher_zero = compute_source_gates(residual, 0.0, residual, 1.0)
    _, teacher_one = compute_source_gates(residual, 1.0, residual, 1.0)
    _, teacher_agree = compute_source_gates(residual, 0.6, zero, 1.0)
    assert teacher_zero.max().item() == 0.0
    assert teacher_one.min().item() == 1.0
    assert torch.max(torch.abs(teacher_agree - 0.6)).item() <= 1e-12


def test_arbiter_schedule():
    cfg = SimpleNamespace(
        USE_SOURCE_ARBITER=True,
        SOURCE_ARBITER_START_EPOCH=7,
        SOURCE_ARBITER_RAMP_END_EPOCH=15,
        SOURCE_ARBITER_STOP_EPOCH=21,
    )
    assert get_arbiter_influence_scale(6, cfg) == 0.0
    assert get_arbiter_influence_scale(7, cfg) == 1.0 / 9.0
    assert get_arbiter_influence_scale(15, cfg) == 1.0
    assert get_arbiter_influence_scale(20, cfg) == 1.0
    assert get_arbiter_influence_scale(21, cfg) == 0.0


def test_delayed_utility_direction():
    shape = (1, 1, 2, 2)
    cfg = SimpleNamespace(
        SOURCE_ARBITER_RESIDUAL_BOUND=1.5,
        SOURCE_ARBITER_UTILITY_TAU=0.10,
        SOURCE_ARBITER_MIN_MEMORY_AGE_EPOCH=1,
        SOURCE_ARBITER_MAX_MEMORY_AGE_EPOCH=2,
        SOURCE_ARBITER_UTILITY_MIN_HISTORY=3,
        SOURCE_ARBITER_UTILITY_MIN_CONSENSUS_CONF=0.30,
        SOURCE_ARBITER_UTILITY_MAX_VIEW_DIFF=0.15,
        SOURCE_ARBITER_UTILITY_MIN_SOURCE_DISAGREEMENT=0.15,
        SOURCE_ARBITER_UTILITY_VIEW_TAU=0.05,
        SOURCE_ARBITER_UTILITY_GAP_SCALE=0.10,
    )
    consensus = torch.full(shape, 0.9)
    target = build_delayed_utility_target(
        old_teacher_prob=torch.full(shape, 0.8),
        dabe_target=torch.full(shape, 0.2),
        evaluator_prob_weak=consensus,
        evaluator_prob_flip=consensus,
        old_history_count=torch.tensor([3]),
        old_epoch=torch.tensor([6]),
        current_epoch=7,
        old_valid=torch.tensor([True]),
        cfg=cfg,
    )
    assert target.valid.all()
    assert (target.utility_gap > 0).all()
    assert (target.target_teacher > 0.5).all()

    reverse = build_delayed_utility_target(
        old_teacher_prob=torch.full(shape, 0.2),
        dabe_target=torch.full(shape, 0.8),
        evaluator_prob_weak=consensus,
        evaluator_prob_flip=consensus,
        old_history_count=torch.tensor([3]),
        old_epoch=torch.tensor([6]),
        current_epoch=7,
        old_valid=torch.tensor([True]),
        cfg=cfg,
    )
    assert reverse.valid.all()
    assert (reverse.utility_gap < 0).all()
    assert (reverse.target_teacher < 0.5).all()
