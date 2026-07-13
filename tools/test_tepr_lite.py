import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.tepr import (
    TemporalTeacherMemory,
    build_tepr_lite_teacher_weight_map,
    get_tepr_scale,
)
from common.utils import load_config
from train import teacher_weighted_bce_with_logits


def make_cfg():
    return SimpleNamespace(
        USE_TEPR_LITE=True,
        TEPR_START_EPOCH=7,
        TEPR_RAMP_END_EPOCH=15,
        TEPR_STOP_EPOCH=21,
        TEPR_MIN_HISTORY=3,
        TEPR_VARIANCE_TAU=0.02,
        TEPR_CONF_GAMMA=1.0,
        TEPR_CORE_LAMBDA=math.log(5.0),
        TEPR_EXTENT_LAMBDA=math.log(4.0),
        TEPR_MARGIN_TAU=0.05,
        TEPR_UNKNOWN_WEIGHT_MIN=0.50,
        TEPR_UNKNOWN_WEIGHT_MAX=0.80,
        TEPR_EXTENT_TEMPORAL_MIN=0.60,
        TEPR_WEIGHT_MIN=0.20,
        TEPR_WEIGHT_MAX=1.00,
        ESA_FEATURE_SIZE=37,
        ESA_DETACH_MASK=True,
        ESA_DETACH_PROTO=True,
    )


def test_schedule():
    cfg = make_cfg()
    expected = {1: 0.0, 6: 0.0, 7: 1.0 / 9.0, 15: 1.0, 16: 1.0, 20: 1.0, 21: 0.0}
    for epoch, value in expected.items():
        assert abs(get_tepr_scale(cfg, epoch) - value) <= 1e-12


def test_memory():
    memory = TemporalTeacherMemory(3, 2, 2, dtype="float16")
    indices = torch.tensor([0, 2])
    first = torch.tensor(
        [[[[0.2, 0.4], [0.6, 0.8]]], [[[0.1, 0.3], [0.5, 0.7]]]],
        dtype=torch.float32,
    )
    before_mean, before_second, before_count = memory.fetch(indices, "cpu")
    assert torch.count_nonzero(before_mean) == 0
    assert torch.count_nonzero(before_second) == 0
    assert torch.equal(before_count, torch.zeros_like(before_count))
    memory.update(indices, first, rho=0.9)
    mean, second, count = memory.fetch(indices, "cpu")
    assert torch.allclose(mean, first, atol=5e-4)
    assert torch.allclose(second, first.square(), atol=5e-4)
    assert torch.equal(count, torch.ones_like(count))
    second_observation = torch.full_like(first, 0.9)
    memory.update(indices, second_observation, rho=0.9)
    mean2, second2, count2 = memory.fetch(indices, "cpu")
    assert torch.allclose(mean2, 0.9 * first + 0.1 * second_observation, atol=8e-4)
    assert torch.allclose(second2, 0.9 * first.square() + 0.1 * second_observation.square(), atol=8e-4)
    assert torch.equal(count2, torch.full_like(count2, 2))
    clone = TemporalTeacherMemory(3, 2, 2, dtype="float16")
    clone.load_state_dict(memory.state_dict())
    for left, right in zip(memory.fetch(indices, "cpu"), clone.fetch(indices, "cpu")):
        assert torch.equal(left, right)
    clone.clear()
    assert int(clone.count.sum().item()) == 0
    assert float(clone.mean.abs().sum().item()) == 0.0


def make_formula_batch():
    batch_size = 3
    fg = torch.zeros(batch_size, 1, 68, 68)
    bg = torch.zeros_like(fg)
    extent = torch.zeros_like(fg)
    unknown = torch.zeros_like(fg)
    fg[:, :, :17, :17] = 1.0
    bg[:, :, 51:, 51:] = 1.0
    extent[:, :, 20:48, 20:48] = 1.0
    unknown[:, :, :17, 51:] = 1.0
    feature = torch.zeros(batch_size, 384, 37, 37)
    fg37 = torch.nn.functional.interpolate(fg, size=(37, 37), mode="nearest").bool()
    bg37 = torch.nn.functional.interpolate(bg, size=(37, 37), mode="nearest").bool()
    extent37 = torch.nn.functional.interpolate(extent, size=(37, 37), mode="nearest").bool()
    feature[:, 0][fg37[:, 0]] = 1.0
    feature[:, 0][bg37[:, 0]] = -1.0
    feature[0, 0][extent37[0, 0]] = 1.0
    feature[1, 1][extent37[1, 0]] = 1.0
    feature[2, 0][extent37[2, 0]] = -1.0
    feature[:, 1] += 1e-4
    return {
        "feature": feature,
        "pu_fg_core": fg,
        "pu_bg_core": bg,
        "pu_extent": extent,
        "pu_unknown": unknown,
    }


def test_formula_endpoints():
    cfg = make_cfg()
    batch = make_formula_batch()
    teacher_prob = torch.full((3, 1, 68, 68), 0.5)
    mean = torch.zeros_like(teacher_prob)
    second = mean.square()
    count = torch.full((3,), 6, dtype=torch.long)
    effective, stats = build_tepr_lite_teacher_weight_map(
        cfg,
        batch,
        teacher_prob,
        mean,
        second,
        count,
        epoch=15,
        device=torch.device("cpu"),
    )
    fg_mask = batch["pu_fg_core"].bool()
    bg_mask = batch["pu_bg_core"].bool()
    assert torch.allclose(effective[fg_mask], torch.full_like(effective[fg_mask], 0.2), atol=1e-5)
    assert torch.allclose(effective[bg_mask], torch.ones_like(effective[bg_mask]), atol=1e-5)
    extent_mask = batch["pu_extent"].bool()
    extent_means = [float(effective[i][extent_mask[i]].mean().item()) for i in range(3)]
    assert abs(extent_means[0] - 0.25) < 0.02
    assert abs(extent_means[1] - 0.50) < 0.02
    assert abs(extent_means[2] - 1.00) < 0.02
    unknown_mask = batch["pu_unknown"].bool()
    assert torch.allclose(effective[unknown_mask], torch.full_like(effective[unknown_mask], 0.8), atol=1e-5)
    assert stats["teacher_map_min"] >= 0.2 - 1e-5
    off, _ = build_tepr_lite_teacher_weight_map(
        cfg,
        batch,
        teacher_prob,
        mean,
        second,
        torch.zeros_like(count),
        epoch=1,
        device=torch.device("cpu"),
    )
    assert torch.equal(off, torch.ones_like(off))


def test_plain_bce_equivalence():
    logits = torch.tensor([[[[-1.0, 0.2], [1.5, -0.4]]]])
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    expected = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    actual = teacher_weighted_bce_with_logits(
        logits,
        target,
        torch.ones_like(logits),
        enabled=False,
    )
    assert torch.equal(actual, expected)


def test_config_parity():
    baseline = load_config(
        "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long35_lrfloor_2e5.py"
    )
    tepr = load_config(
        "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_tepr_lite_long35_lrfloor_2e5.py"
    )
    locked_fields = (
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
    for field in locked_fields:
        assert getattr(tepr, field) == getattr(baseline, field), field
    assert tepr.DINO == baseline.DINO


def main():
    test_schedule()
    test_memory()
    test_formula_endpoints()
    test_plain_bce_equivalence()
    test_config_parity()
    print("TEPR-Lite formula and memory checks passed.")


if __name__ == "__main__":
    main()
