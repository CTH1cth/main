from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.ecst import build_ecst_teacher_weight_map
from common.static_weight import (
    STATIC_WEIGHT_REGIONS,
    accumulate_static_weight_audit,
    build_effective_static_weight,
    build_static_weight_region_masks,
    finalize_static_weight_audit,
    new_static_weight_audit_accumulator,
    static_weight_protocol_fingerprint,
)
from common.utils import config_to_dict, load_config
from train import (
    build_dabe_pu_despl_static_target,
    validate_stop_after_epoch,
    weighted_bce_with_logits,
)


BASE_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
ONES_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5_sw_ones.py"
)
KNOWN_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5_sw_known.py"
)


def make_batch(batch_size=2, size=6):
    shape = (batch_size, 1, size, size)
    fg = torch.zeros(shape)
    fallback = torch.zeros(shape)
    bg = torch.zeros(shape)
    extent = torch.zeros(shape)
    unknown = torch.zeros(shape)
    fg[:, :, 0, :] = 1.0
    fallback[:, :, 0:2, 0:2] = 1.0
    bg[:, :, 2, :] = 1.0
    extent[:, :, 3, :] = 1.0
    unknown[:, :, 4, :] = 1.0
    return {
        "pu_fg_core": fg,
        "pu_fg_fallback": fallback,
        "pu_bg_core": bg,
        "pu_extent": extent,
        "pu_unknown": unknown,
    }


def test_effective_static_weight_modes_and_boundary():
    batch = make_batch()
    raw = torch.linspace(0.0, 1.0, 72).reshape(2, 1, 6, 6)
    expected = {
        "cache": raw,
        "ones": torch.ones_like(raw),
        "known_uniform": (batch["pu_unknown"] <= 0.5).float(),
    }
    for mode, expected_map in expected.items():
        effective, actual_mode = build_effective_static_weight(
            SimpleNamespace(STATIC_WEIGHT_MODE=mode),
            batch,
            raw,
            torch.device("cpu"),
        )
        assert actual_mode == mode
        assert torch.equal(effective, expected_map)
        assert effective.requires_grad is False
    boundary = torch.tensor([0.49, 0.50, 0.51]).view(1, 1, 1, 3)
    boundary_batch = {"pu_unknown": boundary}
    effective, _ = build_effective_static_weight(
        SimpleNamespace(STATIC_WEIGHT_MODE="known_uniform"),
        boundary_batch,
        torch.ones_like(boundary),
        torch.device("cpu"),
    )
    assert torch.equal(
        effective,
        torch.tensor([1.0, 1.0, 0.0]).view_as(boundary),
    )


def test_region_priority_is_mutually_exclusive():
    batch = make_batch()
    masks = build_static_weight_region_masks(
        batch,
        torch.device("cpu"),
        expected_shape=batch["pu_fg_core"].shape,
    )
    coverage = sum(mask.to(torch.int64) for mask in masks.values())
    assert torch.equal(coverage, torch.ones_like(coverage))
    assert not bool(masks["fg_fallback"][:, :, 0, :].any().item())
    assert set(masks) == set(STATIC_WEIGHT_REGIONS)


def test_cache_mode_preserves_three_branch_loss_and_gradients():
    torch.manual_seed(3407)
    batch = make_batch()
    raw = torch.rand(2, 1, 6, 6, dtype=torch.float32)
    target = torch.rand_like(raw)
    effective, _ = build_effective_static_weight(
        SimpleNamespace(STATIC_WEIGHT_MODE="cache"),
        batch,
        raw,
        torch.device("cpu"),
    )
    for _ in ("final", "coarse", "base"):
        logits_reference = torch.randn_like(raw, requires_grad=True)
        logits_effective = logits_reference.detach().clone().requires_grad_(True)
        expected = (
            F.binary_cross_entropy_with_logits(
                logits_reference,
                target,
                reduction="none",
            )
            * raw
        ).sum() / (raw.sum() + 1e-6)
        actual = weighted_bce_with_logits(
            logits_effective,
            target,
            effective,
            eps=1e-6,
        )
        grad_expected = torch.autograd.grad(expected, logits_reference)[0]
        grad_actual = torch.autograd.grad(actual, logits_effective)[0]
        assert abs(float(expected - actual)) <= 1e-7
        assert float((grad_expected - grad_actual).abs().max()) <= 1e-7


def test_static_target_and_resolved_configs_only_change_allowed_fields():
    configs = [
        load_config(BASE_CONFIG),
        load_config(ONES_CONFIG),
        load_config(KNOWN_CONFIG),
    ]
    target = torch.tensor([0.2, 0.5, 0.8]).view(1, 1, 1, 3)
    raw = torch.tensor([0.1, 0.5, 1.0]).view_as(target)
    targets = [
        build_dabe_pu_despl_static_target(cfg, target, raw)[0]
        for cfg in configs
    ]
    assert all(torch.equal(item, target) for item in targets)
    fingerprints = [static_weight_protocol_fingerprint(cfg) for cfg in configs]
    assert len(set(fingerprints)) == 1
    baseline = config_to_dict(configs[0])
    for cfg in configs[1:]:
        candidate = config_to_dict(cfg)
        differing = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key) != candidate.get(key)
        }
        assert differing == {"EXP_NAME", "STATIC_WEIGHT_MODE"}


def _ecst_batch():
    batch = make_batch(batch_size=1, size=68)
    feature = torch.zeros(1, 384, 37, 37)
    fg37 = F.interpolate(
        batch["pu_fg_core"],
        size=(37, 37),
        mode="nearest",
    ).bool()
    bg37 = F.interpolate(
        batch["pu_bg_core"],
        size=(37, 37),
        mode="nearest",
    ).bool()
    feature[:, 0][fg37[:, 0]] = 1.0
    feature[:, 0][bg37[:, 0]] = -1.0
    feature[:, 1] = 1e-4
    batch["feature"] = feature
    return batch


def test_ecst_teacher_route_is_static_mode_invariant():
    batch = _ecst_batch()
    teacher_prob = torch.full((1, 1, 68, 68), 0.4)
    temporal_mean = torch.full_like(teacher_prob, 0.3)
    temporal_second = temporal_mean.square()
    history = torch.tensor([6])
    outputs = []
    teacher_losses = []
    logits = torch.zeros_like(teacher_prob)
    teacher_target = (teacher_prob >= 0.5).float()
    for path in (BASE_CONFIG, ONES_CONFIG, KNOWN_CONFIG):
        cfg = load_config(path)
        route, _ = build_ecst_teacher_weight_map(
            cfg,
            batch,
            teacher_prob,
            temporal_mean,
            temporal_second,
            history,
            epoch=15,
            device=torch.device("cpu"),
        )
        outputs.append(route)
        teacher_losses.append(
            weighted_bce_with_logits(
                logits,
                teacher_target,
                route,
            )
        )
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
    assert torch.equal(teacher_losses[0], teacher_losses[1])
    assert torch.equal(teacher_losses[0], teacher_losses[2])


def test_epoch_audit_mass_and_inactive_gradient_are_safe():
    batch = make_batch()
    raw = torch.full((2, 1, 6, 6), 0.5)
    effective = torch.ones_like(raw)
    target = torch.full_like(raw, 0.4)
    logits = torch.zeros_like(raw)
    teacher = torch.full_like(raw, 0.6)
    masks = build_static_weight_region_masks(
        batch,
        torch.device("cpu"),
        expected_shape=raw.shape,
    )
    accumulator = new_static_weight_audit_accumulator()
    accumulate_static_weight_audit(
        accumulator,
        raw,
        effective,
        target,
        logits,
        teacher,
        masks,
        static_weight=0.0,
    )
    row = finalize_static_weight_audit(
        accumulator,
        epoch=21,
        mode="ones",
        global_static_weight=0.0,
        global_teacher_weight=1.0,
    )
    assert row["static_gradient_active"] is False
    assert abs(
        sum(row[f"{name}_weight_mass"] for name in STATIC_WEIGHT_REGIONS) - 1.0
    ) <= 1e-12
    assert sum(row[f"{name}_gradmass"] for name in STATIC_WEIGHT_REGIONS) == 0.0


def test_invalid_mode_shape_and_stop_epoch_are_rejected():
    batch = make_batch()
    raw = torch.ones(2, 1, 6, 6)
    try:
        build_effective_static_weight(
            SimpleNamespace(STATIC_WEIGHT_MODE="bad"),
            batch,
            raw,
            torch.device("cpu"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid STATIC_WEIGHT_MODE was accepted")
    try:
        build_effective_static_weight(
            SimpleNamespace(STATIC_WEIGHT_MODE="cache"),
            batch,
            torch.ones(2, 6, 6),
            torch.device("cpu"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid static-weight shape was accepted")
    assert validate_stop_after_epoch(0, 45, 1) == 0
    assert validate_stop_after_epoch(25, 45, 1) == 25
    for stop, maximum, start in ((-1, 45, 1), (46, 45, 1), (20, 45, 21)):
        try:
            validate_stop_after_epoch(stop, maximum, start)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid stop tuple {(stop, maximum, start)} was accepted"
            )


def test_stop_after_epoch_runs_after_validation_checkpoint_and_reset():
    source = Path("train.py").read_text(encoding="utf-8")
    validation_pos = source.index("result = validate_one_dataset(")
    checkpoint_pos = source.index(
        "if should_save_epoch_checkpoint(",
        validation_pos,
    )
    reset_pos = source.index(
        "optimizer, scheduler, global_step, lr_floor_activated_logged = "
        "apply_finetune_reset(",
        checkpoint_pos,
    )
    stop_pos = source.index(
        "if stop_after_epoch > 0 and int(epoch) == int(stop_after_epoch):",
        reset_pos,
    )
    assert validation_pos < checkpoint_pos < reset_pos < stop_pos
