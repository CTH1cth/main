from copy import deepcopy
from types import SimpleNamespace

import torch

from common.ecst import (
    build_ecst_evidence_states,
    build_ecst_teacher_weight_map,
    build_ecst_teacher_weight_map_from_states,
)
from common.source_arbiter import (
    SourceArbiter,
    apply_loss_space_arbitration,
    build_delayed_utility_target,
    build_source_arbiter_inputs,
    build_teacher_source_weight,
    compute_source_gates,
)


def _fixture():
    torch.manual_seed(3407)
    batch = {
        "feature": torch.randn(2, 384, 37, 37),
        "pu_fg_core": torch.zeros(2, 1, 8, 8),
        "pu_bg_core": torch.zeros(2, 1, 8, 8),
        "pu_extent": torch.zeros(2, 1, 8, 8),
        "pu_unknown": torch.zeros(2, 1, 8, 8),
        "pu_target_soft": torch.rand(2, 1, 8, 8),
        "pu_weight_map": torch.rand(2, 1, 8, 8),
    }
    batch["pu_fg_core"][:, :, :2, :2] = 1
    batch["pu_bg_core"][:, :, -2:, -2:] = 1
    batch["pu_extent"][:, :, 2:6, 2:6] = 1
    batch["pu_unknown"][:, :, :2, -2:] = 1
    teacher = torch.rand(2, 1, 8, 8)
    mean = torch.full_like(teacher, 0.3)
    second = torch.full_like(teacher, 0.1)
    count = torch.full((2,), 6, dtype=torch.long)
    cfg = SimpleNamespace(
        USE_ECST=True,
        ECST_START_EPOCH=7,
        ECST_RAMP_END_EPOCH=15,
        ECST_STOP_EPOCH=21,
        ECST_VARIANCE_TAU=0.02,
        ECST_CONF_GAMMA=1.0,
        ECST_MIN_HISTORY=3,
        ECST_INSUFFICIENT_HISTORY_MODE="dino_only",
        ECST_FEATURE_SIZE=37,
        ECST_DETACH_PROTO=True,
        ECST_DETACH_MASK=True,
        ECST_CORE_CONFLICT_WEIGHT=0.2,
        ECST_EXTENT_FG_WEIGHT=1.0,
        ECST_EXTENT_BG_WEIGHT_FLOOR=0.25,
        ECST_EXTENT_DINO_LAMBDA=1.386294,
        ECST_MARGIN_TAU=0.05,
        ECST_UNKNOWN_WEIGHT=0.5,
        ECST_WEIGHT_MIN=0.2,
        ECST_WEIGHT_MAX=1.0,
        ECST_MARGIN_FG_LIKE=0.05,
        ECST_MARGIN_BG_LIKE=-0.05,
    )
    return cfg, batch, teacher, mean, second, count


def test_fixed_map_constants_do_not_change_r2_evidence_or_router_input():
    cfg, batch, teacher, mean, second, count = _fixture()
    changed = deepcopy(cfg)
    changed.ECST_CORE_CONFLICT_WEIGHT = 0.7
    changed.ECST_EXTENT_BG_WEIGHT_FLOOR = 0.8
    changed.ECST_UNKNOWN_WEIGHT = 0.9
    states_a = build_ecst_evidence_states(
        cfg, batch, teacher, mean, second, count, torch.device("cpu")
    )
    states_b = build_ecst_evidence_states(
        changed, batch, teacher, mean, second, count, torch.device("cpu")
    )
    for key in (
        "mean",
        "variance",
        "bg_reliability",
        "margin_68",
        "core_conflict",
        "extent_teacher_bg",
    ):
        assert torch.equal(states_a[key], states_b[key])

    student_prob = torch.rand_like(teacher)
    router_input_a = build_source_arbiter_inputs(
        batch,
        teacher,
        student_prob,
        mean,
        states_a["variance"],
        count,
        states_a["margin_68"],
        states_a["masks"],
        0.5,
        3,
        0.05,
        "raw_clamped",
        1.0,
    )
    router_input_b = build_source_arbiter_inputs(
        batch,
        teacher,
        student_prob,
        mean,
        states_b["variance"],
        count,
        states_b["margin_68"],
        states_b["masks"],
        0.5,
        3,
        0.05,
        "raw_clamped",
        1.0,
    )
    assert torch.equal(router_input_a, router_input_b)
    router = SourceArbiter()
    residual_a = router(router_input_a)
    residual_b = router(router_input_b)
    assert torch.equal(residual_a, residual_b)
    gates_a = compute_source_gates(
        residual_a,
        teacher_prior=0.5,
        source_disagreement=teacher - batch["pu_target_soft"],
        influence_scale=1.0,
    )
    gates_b = compute_source_gates(
        residual_b,
        teacher_prior=0.5,
        source_disagreement=teacher - batch["pu_target_soft"],
        influence_scale=1.0,
    )
    assert torch.equal(gates_a[0], gates_b[0])
    assert torch.equal(gates_a[1], gates_b[1])

    logits_a = torch.randn_like(teacher, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    common_loss = {
        "dabe_target": batch["pu_target_soft"],
        "dabe_weight": batch["pu_weight_map"],
        "teacher_target": (teacher >= 0.5).float(),
        "teacher_weight": build_teacher_source_weight(
            "pure_loss_space", teacher
        ),
        "source_sum": 1.0,
    }
    loss_a = apply_loss_space_arbitration(
        logits=logits_a,
        gate_dabe=gates_a[0],
        gate_teacher=gates_a[1],
        **common_loss,
    )["loss"]
    loss_b = apply_loss_space_arbitration(
        logits=logits_b,
        gate_dabe=gates_b[0],
        gate_teacher=gates_b[1],
        **common_loss,
    )["loss"]
    grad_a = torch.autograd.grad(loss_a, logits_a)[0]
    grad_b = torch.autograd.grad(loss_b, logits_b)[0]
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(grad_a, grad_b)

    evaluator_weak = torch.rand_like(teacher)
    evaluator_flip = torch.rand_like(teacher)
    old_epoch = torch.full((2,), 7, dtype=torch.long)
    old_valid = torch.ones(2, dtype=torch.bool)
    utility_a = build_delayed_utility_target(
        teacher,
        batch["pu_target_soft"],
        evaluator_weak,
        evaluator_flip,
        count,
        old_epoch,
        8,
        old_valid,
        cfg,
    )
    utility_b = build_delayed_utility_target(
        teacher,
        batch["pu_target_soft"],
        evaluator_weak,
        evaluator_flip,
        count,
        old_epoch,
        8,
        old_valid,
        changed,
    )
    for name in ("target_teacher", "valid", "weight", "utility_gap"):
        assert torch.equal(getattr(utility_a, name), getattr(utility_b, name))

    audit_a, _ = build_ecst_teacher_weight_map_from_states(
        cfg, teacher, count, 15, torch.device("cpu"), states_a
    )
    audit_b, _ = build_ecst_teacher_weight_map_from_states(
        changed, teacher, count, 15, torch.device("cpu"), states_b
    )
    assert not torch.equal(audit_a, audit_b)
    assert torch.equal(
        build_teacher_source_weight("pure_loss_space", teacher),
        torch.ones_like(teacher),
    )


def test_r1_compatibility_wrapper_matches_split_builder():
    cfg, batch, teacher, mean, second, count = _fixture()
    wrapped_map, wrapped_stats = build_ecst_teacher_weight_map(
        cfg,
        batch,
        teacher,
        mean,
        second,
        count,
        15,
        torch.device("cpu"),
    )
    states = build_ecst_evidence_states(
        cfg, batch, teacher, mean, second, count, torch.device("cpu")
    )
    split_map, split_stats = build_ecst_teacher_weight_map_from_states(
        cfg, teacher, count, 15, torch.device("cpu"), states
    )
    assert torch.equal(wrapped_map, split_map)
    for key in (
        "teacher_map_raw_min",
        "teacher_map_raw_mean",
        "teacher_map_raw_max",
        "teacher_map_min",
        "teacher_map_mean",
        "teacher_map_max",
    ):
        assert wrapped_stats[key] == split_stats[key]
