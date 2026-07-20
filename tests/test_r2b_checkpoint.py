from types import SimpleNamespace

import pytest
import torch

from common.source_arbiter import SignAwareSourceArbiter, SourceArbiter
from train import (
    migrate_r1_router_to_sign_aware,
    validate_source_arbiter_resume_mode,
)


R1_EXP = (
    "dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)


def _r1_epoch6_checkpoint():
    router = SourceArbiter()
    with torch.no_grad():
        for name, value in router.state_dict().items():
            if name.startswith("body.") and value.is_floating_point():
                value.copy_(torch.randn_like(value))
    return {
        "epoch": 6,
        "config": {
            "EXP_NAME": R1_EXP,
            "SOURCE_ARBITER_MODE": "residual_over_ecst",
        },
        "checkpoint_phase": "active_pre_reset",
        "source_arbiter": router.state_dict(),
        "source_arbiter_optimizer": {"state": {}, "param_groups": []},
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


def test_r1_epoch6_migrates_body_but_not_single_head():
    checkpoint = _r1_epoch6_checkpoint()
    router = SignAwareSourceArbiter()
    migrate_r1_router_to_sign_aware(router, checkpoint)
    migrated = router.state_dict()
    for name, value in checkpoint["source_arbiter"].items():
        if name.startswith("body."):
            assert torch.equal(migrated[name], value)
    assert migrated["head.weight"].abs().max().item() == 0.0
    assert migrated["head.bias"].abs().max().item() == 0.0


def test_nonzero_r1_head_is_rejected():
    checkpoint = _r1_epoch6_checkpoint()
    checkpoint["source_arbiter"]["head.bias"].fill_(0.1)
    with pytest.raises(RuntimeError, match="zero R1 router head"):
        migrate_r1_router_to_sign_aware(
            SignAwareSourceArbiter(),
            checkpoint,
        )


def test_exploratory_long45_accepts_only_canonical_r1_epoch6_fork():
    cfg = SimpleNamespace(
        SOURCE_ARBITER_MODE="sign_aware_pure_loss_space",
        SOURCE_ARBITER_REQUIRED_RESUME_EPOCH=6,
        ECST_MEMORY_UPDATE_END_EPOCH=45,
    )
    contract = validate_source_arbiter_resume_mode(
        cfg,
        _r1_epoch6_checkpoint(),
    )
    assert contract["is_r1_epoch6_fork"] is True
    assert contract["is_r2b_stage20_resume"] is False
    assert contract["temporal_memory_key"] == "ecst_temporal_memory"


def test_stage20_pending_reset_is_the_only_long45_resume_contract():
    stage_exp = (
        "dinov1_s8_dabepu_v11_egsa_r2b_signaware_dagp_uncgate_ndr_"
        "stage20_lrfloor_2e5"
    )
    cfg = SimpleNamespace(
        SOURCE_ARBITER_MODE="sign_aware_pure_loss_space",
        SOURCE_ARBITER_REQUIRED_RESUME_EPOCH=20,
        ECST_MEMORY_UPDATE_END_EPOCH=45,
        SOURCE_ARBITER_POS_RESIDUAL_BOUND=1.5,
        SOURCE_ARBITER_NEG_RESIDUAL_BOUND=4.0,
    )
    checkpoint = {
        "epoch": 20,
        "config": {"EXP_NAME": stage_exp},
        "source_arbiter_mode": "sign_aware_pure_loss_space",
        "teacher_source_weight_mode": "ones",
        "ecst_weighting_used_for_training": False,
        "source_arbiter_version": "egsa_r2b_signaware_directional_v1",
        "source_arbiter_utility_mode": "directional_gradient_alignment",
        "source_arbiter_output_channels": 2,
        "positive_residual_bound": 1.5,
        "negative_residual_bound": 4.0,
        "source_arbiter_stage": "stage20",
        "checkpoint_phase": "pending_after_epoch_reset",
        "route_trajectory_memory": {},
        "utility_evaluator": {},
        "source_temporal_memory": {},
        "source_arbiter_lifecycle": {
            "source_temporal_memory_active": True,
            "route_memory_active": True,
            "utility_evaluator_active": True,
        },
    }
    contract = validate_source_arbiter_resume_mode(cfg, checkpoint)
    assert contract["is_r2b_stage20_resume"] is True
    assert contract["temporal_memory_key"] == "source_temporal_memory"

    checkpoint["checkpoint_phase"] = "post_reset_frozen"
    with pytest.raises(RuntimeError, match="pending-reset"):
        validate_source_arbiter_resume_mode(cfg, checkpoint)
