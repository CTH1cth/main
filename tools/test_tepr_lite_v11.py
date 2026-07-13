import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.tepr import build_tepr_lite_v11_teacher_weight_map, get_tepr_scale
from common.utils import load_config


def make_cfg():
    return SimpleNamespace(
        USE_TEPR_LITE=True,
        TEPR_START_EPOCH=7,
        TEPR_RAMP_END_EPOCH=15,
        TEPR_STOP_EPOCH=21,
        TEPR_MIN_HISTORY=3,
        TEPR_VARIANCE_TAU=0.02,
        TEPR_CONF_GAMMA=1.0,
        TEPR_INSUFFICIENT_HISTORY_MODE="dino_only",
        TEPR_CORE_CONFLICT_WEIGHT=0.20,
        TEPR_EXTENT_BG_WEIGHT_FLOOR=0.25,
        TEPR_EXTENT_DINO_LAMBDA=math.log(4.0),
        TEPR_MARGIN_TAU=0.05,
        TEPR_UNKNOWN_WEIGHT=0.50,
        TEPR_OTHER_WEIGHT=1.00,
        TEPR_WEIGHT_MIN=0.20,
        TEPR_WEIGHT_MAX=1.00,
        ESA_FEATURE_SIZE=37,
        ESA_DETACH_MASK=True,
        ESA_DETACH_PROTO=True,
        ESA_MARGIN_FG_LIKE=0.05,
        ESA_MARGIN_BG_LIKE=-0.05,
        # Legacy v1 fields must be ignored by the v1.1 builder.
        TEPR_UNKNOWN_WEIGHT_MIN=0.50,
        TEPR_UNKNOWN_WEIGHT_MAX=0.80,
        TEPR_EXTENT_TEMPORAL_MIN=0.60,
        TEPR_CORE_LAMBDA=math.log(5.0),
        TEPR_EXTENT_LAMBDA=math.log(4.0),
    )


def make_batch():
    batch_size = 6
    fg = torch.zeros(batch_size, 1, 68, 68)
    bg = torch.zeros_like(fg)
    extent = torch.zeros_like(fg)
    unknown = torch.zeros_like(fg)
    fg[:, :, :12, :12] = 1.0
    bg[:, :, 56:, 56:] = 1.0
    extent[:, :, 20:48, 20:48] = 1.0
    unknown[:, :, :12, 56:] = 1.0

    feature = torch.zeros(batch_size, 384, 37, 37)
    fg37 = torch.nn.functional.interpolate(fg, size=(37, 37), mode="nearest").bool()
    bg37 = torch.nn.functional.interpolate(bg, size=(37, 37), mode="nearest").bool()
    extent37 = torch.nn.functional.interpolate(extent, size=(37, 37), mode="nearest").bool()
    feature[:, 0][fg37[:, 0]] = 1.0
    feature[:, 0][bg37[:, 0]] = -1.0
    for index in (0, 1):
        feature[index, 0][extent37[index, 0]] = 1.0
    for index in (2, 3):
        feature[index, 1][extent37[index, 0]] = 1.0
    for index in (4, 5):
        feature[index, 0][extent37[index, 0]] = -1.0
    return {
        "feature": feature,
        "pu_fg_core": fg,
        "pu_bg_core": bg,
        "pu_extent": extent,
        "pu_unknown": unknown,
    }


def make_inputs(batch):
    teacher_prob = torch.full((6, 1, 68, 68), 0.10)
    teacher_prob[:, :, :12, :6] = 0.90
    teacher_prob[:, :, 56:, 62:] = 0.90
    teacher_prob[:, :, 20:24, 20:48] = 0.90
    mean_values = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    temporal_mean = mean_values.view(-1, 1, 1, 1).expand_as(teacher_prob).clone()
    temporal_second = temporal_mean.square()
    history_count = torch.full((6,), 6, dtype=torch.long)
    return teacher_prob, temporal_mean, temporal_second, history_count


def masked_mean(value, mask):
    return float(value[mask].mean().item())


def test_schedule():
    cfg = make_cfg()
    expected = {1: 0.0, 6: 0.0, 7: 1.0 / 9.0, 15: 1.0, 20: 1.0, 21: 0.0}
    for epoch, value in expected.items():
        assert abs(get_tepr_scale(cfg, epoch) - value) <= 1e-12


def test_v11_endpoints():
    cfg = make_cfg()
    batch = make_batch()
    teacher_prob, mean, second, count = make_inputs(batch)
    effective, stats = build_tepr_lite_v11_teacher_weight_map(
        cfg, batch, teacher_prob, mean, second, count, 15, torch.device("cpu")
    )
    fg = batch["pu_fg_core"].bool()
    bg = batch["pu_bg_core"].bool()
    extent = batch["pu_extent"].bool()
    unknown = batch["pu_unknown"].bool()
    teacher_fg = teacher_prob >= 0.5
    fg_conflict = fg & (~teacher_fg)
    fg_no_conflict = fg & teacher_fg
    bg_conflict = bg & teacher_fg
    bg_no_conflict = bg & (~teacher_fg)
    extent_fg = extent & teacher_fg
    extent_bg = extent & (~teacher_fg)

    assert torch.allclose(effective[fg_conflict], torch.full_like(effective[fg_conflict], 0.2))
    assert torch.allclose(effective[fg_no_conflict], torch.ones_like(effective[fg_no_conflict]))
    assert torch.allclose(effective[bg_conflict], torch.full_like(effective[bg_conflict], 0.2))
    assert torch.allclose(effective[bg_no_conflict], torch.ones_like(effective[bg_no_conflict]))
    assert torch.allclose(effective[extent_fg], torch.ones_like(effective[extent_fg]))
    assert torch.allclose(effective[unknown], torch.full_like(effective[unknown], 0.5))

    expected_extent_bg = (0.25, 0.25, 0.50, 0.25, 1.00, 0.25)
    for index, expected in enumerate(expected_extent_bg):
        actual = masked_mean(effective[index], extent_bg[index])
        assert abs(actual - expected) < 0.02, (index, actual, expected)
    assert stats["teacher_map_min"] >= 0.20 - 1e-5
    assert stats["teacher_map_max"] <= 1.00 + 1e-5


def test_insufficient_history_falls_back_to_dino_only():
    cfg = make_cfg()
    batch = make_batch()
    teacher_prob, mean, second, count = make_inputs(batch)
    effective, _ = build_tepr_lite_v11_teacher_weight_map(
        cfg,
        batch,
        teacher_prob,
        mean,
        second,
        torch.zeros_like(count),
        15,
        torch.device("cpu"),
    )
    extent_bg = batch["pu_extent"].bool() & (teacher_prob < 0.5)
    expected = (0.25, 0.25, 0.50, 0.50, 1.00, 1.00)
    for index, expected_value in enumerate(expected):
        actual = masked_mean(effective[index], extent_bg[index])
        assert abs(actual - expected_value) < 0.02, (index, actual, expected_value)


def test_ramp_and_legacy_fields_are_ignored():
    cfg = make_cfg()
    batch = make_batch()
    teacher_prob, mean, second, count = make_inputs(batch)
    full, _ = build_tepr_lite_v11_teacher_weight_map(
        cfg, batch, teacher_prob, mean, second, count, 15, torch.device("cpu")
    )
    ramped, _ = build_tepr_lite_v11_teacher_weight_map(
        cfg, batch, teacher_prob, mean, second, count, 7, torch.device("cpu")
    )
    assert torch.allclose(ramped, (8.0 / 9.0) + full / 9.0, atol=1e-6)

    changed = copy.deepcopy(cfg)
    changed.TEPR_UNKNOWN_WEIGHT_MIN = 0.01
    changed.TEPR_UNKNOWN_WEIGHT_MAX = 0.99
    changed.TEPR_EXTENT_TEMPORAL_MIN = 0.01
    changed.TEPR_CORE_LAMBDA = 99.0
    changed.TEPR_EXTENT_LAMBDA = 99.0
    changed_out, _ = build_tepr_lite_v11_teacher_weight_map(
        changed, batch, teacher_prob, mean, second, count, 15, torch.device("cpu")
    )
    assert torch.equal(full, changed_out)


def test_config_parity():
    base = load_config(
        "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_tepr_lite_long35_lrfloor_2e5.py"
    )
    v11 = load_config(
        "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_tepr_lite_v11_asymneg_long35_lrfloor_2e5.py"
    )
    locked = (
        "HEAD_TYPE",
        "USE_NDR_BRANCH",
        "BATCH_SIZE",
        "SEED",
        "EMA_WEIGHT",
        "MAX_EPOCH",
        "LOSS_SIZE",
        "DABE_PU_VERSION",
        "DABE_PU_ROOT",
        "P_INIT_MODE",
        "TEACHER_FUSION_MODE",
        "DABE_PU_STATIC_TARGET_MODE",
        "TEACHER_TARGET_MODE",
        "FINETUNE_RESET_EPOCH",
        "FINETUNE_RESET_TIMING",
        "FINETUNE_RESET_TEACHER",
        "USE_LR_FLOOR",
        "LR_FLOOR",
    )
    for field in locked:
        assert getattr(v11, field) == getattr(base, field), field
    assert v11.DINO == base.DINO


def main():
    test_schedule()
    test_v11_endpoints()
    test_insufficient_history_falls_back_to_dino_only()
    test_ramp_and_legacy_fields_are_ignored()
    test_config_parity()
    print("TEPR-Lite-v1.1 checks passed.")


if __name__ == "__main__":
    main()
