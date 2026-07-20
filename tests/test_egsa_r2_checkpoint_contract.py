from types import SimpleNamespace

import pytest

from train import validate_source_arbiter_resume_mode


R1_EXP = (
    "dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)


def _complete_r1_epoch6():
    return {
        "epoch": 6,
        "config": {
            "EXP_NAME": R1_EXP,
            "SOURCE_ARBITER_MODE": "residual_over_ecst",
        },
        "checkpoint_phase": "active_pre_reset",
        "source_arbiter": {},
        "source_arbiter_optimizer": {},
        "utility_evaluator": {},
        "route_trajectory_memory": {},
        "ecst_temporal_memory": {},
        "global_step": 1,
        "rng_state": {},
        "train_loader_generator_state": object(),
        "source_arbiter_lifecycle": {
            "route_memory_active": True,
            "utility_evaluator_active": True,
            "ecst_memory_active": True,
        },
    }


def test_only_complete_r1_epoch6_can_fork_to_r2():
    cfg = SimpleNamespace(SOURCE_ARBITER_MODE="pure_loss_space")
    contract = validate_source_arbiter_resume_mode(cfg, _complete_r1_epoch6())
    assert contract["is_r1_epoch6_fork"]
    assert contract["temporal_memory_key"] == "ecst_temporal_memory"

    bad = _complete_r1_epoch6()
    bad["epoch"] = 7
    with pytest.raises(RuntimeError, match="only cross-mode path"):
        validate_source_arbiter_resume_mode(cfg, bad)


def test_same_mode_r2_requires_explicit_training_weight_metadata():
    cfg = SimpleNamespace(SOURCE_ARBITER_MODE="pure_loss_space")
    checkpoint = {
        "epoch": 10,
        "source_arbiter_mode": "pure_loss_space",
        "teacher_source_weight_mode": "ones",
        "ecst_weighting_used_for_training": False,
        "source_arbiter_lifecycle": {
            "source_temporal_memory_active": True,
        },
    }
    contract = validate_source_arbiter_resume_mode(cfg, checkpoint)
    assert not contract["is_r1_epoch6_fork"]
    assert contract["temporal_memory_key"] == "source_temporal_memory"

    checkpoint["ecst_weighting_used_for_training"] = True
    with pytest.raises(RuntimeError, match="ecst_weighting_used_for_training"):
        validate_source_arbiter_resume_mode(cfg, checkpoint)
