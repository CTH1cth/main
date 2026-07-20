from pathlib import Path
from types import SimpleNamespace

import pytest

from common.utils import config_to_dict, load_config
from train import (
    get_dabe_pu_despl_schedule,
    validate_supervision_handover_config,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5_sw_ones_noecst.py"
)
SLOW15_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5_sw_ones_noecst_slow15.py"
)
LINEAR30_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_"
    "long50_lrfloor_2e5_sw_ones_noecst_linear30.py"
)


def load_pair():
    return load_config(BASE_CONFIG), load_config(SLOW15_CONFIG)


def test_slow15_resolved_config_only_changes_handover_fields():
    baseline, slow15 = load_pair()
    baseline_values = config_to_dict(baseline)
    slow15_values = config_to_dict(slow15)
    differing = {
        key
        for key in set(baseline_values) | set(slow15_values)
        if baseline_values.get(key) != slow15_values.get(key)
    }
    assert differing == {
        "EXP_NAME",
        "SUPERVISION_HANDOVER_GAMMA",
        "SUPERVISION_HANDOVER_MODE",
    }


@pytest.mark.parametrize(
    ("epoch", "expected_static", "expected_teacher"),
    (
        (1, 1.000000000, 0.000000000),
        (7, 0.831414554, 0.168585446),
        (10, 0.690288759, 0.309711241),
        (15, 0.399123447, 0.600876553),
        (20, 0.050000000, 0.950000000),
        (21, 0.000000000, 1.000000000),
    ),
)
def test_slow15_schedule(epoch, expected_static, expected_teacher):
    _, slow15 = load_pair()
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, slow15)
    assert static_weight == pytest.approx(expected_static, abs=5e-10)
    assert teacher_weight == pytest.approx(expected_teacher, abs=5e-10)
    assert abs(static_weight + teacher_weight - 1.0) < 1e-8


def test_legacy_noecst_keeps_linear_schedule():
    baseline, _ = load_pair()
    assert not hasattr(baseline, "SUPERVISION_HANDOVER_MODE")
    for epoch in range(1, 21):
        progress = float(epoch - 1) / 19.0
        expected_teacher = 0.95 * progress
        expected_static = 1.0 - expected_teacher
        static_weight, teacher_weight = get_dabe_pu_despl_schedule(
            epoch, baseline
        )
        assert static_weight == pytest.approx(expected_static, abs=1e-15)
        assert teacher_weight == pytest.approx(expected_teacher, abs=1e-15)
    assert get_dabe_pu_despl_schedule(21, baseline) == (0.0, 1.0)


def test_power_handover_validation_rejects_invalid_inputs():
    base = {
        "SUPERVISION_HANDOVER_MODE": "power",
        "TEACHER_FUSION_MODE": "dabe_pu_despl_sched",
        "DABE_PU_DESPL_STAGE_START": 1,
        "DABE_PU_DESPL_STAGE_END": 20,
        "DABE_PU_DESPL_STATIC_START": 1.0,
        "DABE_PU_DESPL_STATIC_END": 0.05,
        "DABE_PU_DESPL_TEACHER_START": 0.0,
        "DABE_PU_DESPL_TEACHER_END": 0.95,
    }
    with pytest.raises(RuntimeError, match="finite and positive"):
        validate_supervision_handover_config(
            SimpleNamespace(**base, SUPERVISION_HANDOVER_GAMMA=0.0)
        )
    with pytest.raises(RuntimeError, match="complementary"):
        validate_supervision_handover_config(
            SimpleNamespace(
                **{
                    **base,
                    "SUPERVISION_HANDOVER_GAMMA": 1.5,
                    "DABE_PU_DESPL_STATIC_END": 0.10,
                }
            )
        )


@pytest.mark.parametrize(
    ("epoch", "expected_static", "expected_teacher"),
    (
        (1, 1.000000000, 0.000000000),
        (7, 0.796428571, 0.203571429),
        (10, 0.694642857, 0.305357143),
        (15, 0.525000000, 0.475000000),
        (20, 0.355357143, 0.644642857),
        (25, 0.185714286, 0.814285714),
        (29, 0.050000000, 0.950000000),
        (30, 0.000000000, 1.000000000),
        (50, 0.000000000, 1.000000000),
    ),
)
def test_linear30_schedule(epoch, expected_static, expected_teacher):
    cfg = load_config(LINEAR30_CONFIG)
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
    assert static_weight == pytest.approx(expected_static, abs=5e-10)
    assert teacher_weight == pytest.approx(expected_teacher, abs=5e-10)
    assert abs(static_weight + teacher_weight - 1.0) < 1e-8


def test_linear30_reset_and_teacher_only_boundaries():
    cfg = load_config(LINEAR30_CONFIG)
    assert validate_supervision_handover_config(cfg) == ("linear", 1.0)
    assert cfg.DABE_PU_DESPL_STAGE_END == 29
    assert cfg.FINETUNE_RESET_EPOCH == 29
    assert cfg.FINETUNE_RESET_TIMING == "after_epoch"
    assert cfg.DABE_PU_DESPL_TEACHER_ONLY_START == 30
    assert get_dabe_pu_despl_schedule(29, cfg) == (0.050000000000000044, 0.95)
    assert get_dabe_pu_despl_schedule(30, cfg) == (0.0, 1.0)


def test_linear30_only_changes_schedule_reset_and_length():
    baseline = config_to_dict(load_config(BASE_CONFIG))
    linear30 = config_to_dict(load_config(LINEAR30_CONFIG))
    differing = {
        key
        for key in set(baseline) | set(linear30)
        if baseline.get(key) != linear30.get(key)
    }
    assert differing == {
        "DABE_PU_DESPL_STAGE_END",
        "DABE_PU_DESPL_TEACHER_ONLY_START",
        "EXP_NAME",
        "FINETUNE_RESET_EPOCH",
        "FUSION_ORIG_DECAY_EPOCHS",
        "LR_LINEAR_STAGE1_EPOCHS",
        "MAX_EPOCH",
        "SUPERVISION_HANDOVER_MODE",
        "TEACHER_FUSION_PRE_RESET_EPOCHS",
    }
