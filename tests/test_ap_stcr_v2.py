from types import SimpleNamespace

import pytest
import torch

from common.ap_stcr import AnchorPropagatedSemanticTemporalCorrection
from common.utils import config_to_dict, load_config
from train import (
    ap_stcr_checkpoint_phase,
    ap_stcr_should_clear_history_on_reset,
    get_dabe_pu_despl_schedule,
    validate_ap_stcr_config,
)


V2_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_apstcr_v2_conflictpass_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
BASE_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5_sw_ones_noecst.py"
)


def _v2_module():
    config = {
        "enabled": True,
        "version": "ap_stcr_v2_conflict_only_pass_through",
        "evidence_resolution": 37,
        "loss_resolution": 68,
        "fg_anchor_ratio": 0.20,
        "bg_anchor_ratio": 0.20,
        "min_fg_anchors": 4,
        "min_bg_anchors": 4,
        "max_fg_anchors": 64,
        "max_bg_anchors": 64,
        "prefer_dabe_background_seed": True,
        "tau_delta": 0.25,
        "tau_margin": 0.50,
        "temporal_window": 3,
        "tau_temporal": 0.20,
        "temporal_empty_support": 0.50,
        "history_dtype": "float16",
        "conflict_only": True,
        "rejection_max": 0.35,
        "support_neutral_point": 0.50,
        "eps": 1e-6,
    }
    return AnchorPropagatedSemanticTemporalCorrection(
        config, [("TR-CAMO", "sample_0")]
    )


def test_v2_config_only_adds_ap_stcr_identity_and_module():
    base = load_config(BASE_CONFIG)
    v2 = load_config(V2_CONFIG)
    assert validate_ap_stcr_config(v2)
    base_values = config_to_dict(base)
    v2_values = config_to_dict(v2)
    changed = {
        name
        for name in set(base_values) | set(v2_values)
        if base_values.get(name) != v2_values.get(name)
    }
    assert changed == {
        "AP_STCR",
        "EXP_NAME",
        "SUPERVISION_MODE",
        "USE_AP_STCR",
    }

    expected = {
        1: (1.0, 0.0),
        7: (0.7, 0.3),
        10: (0.55, 0.45),
        15: (0.3, 0.7),
        20: (0.05, 0.95),
        21: (0.0, 1.0),
        45: (0.0, 1.0),
    }
    for epoch, pair in expected.items():
        assert get_dabe_pu_despl_schedule(epoch, v2) == pytest.approx(
            pair, abs=1e-12
        )
    assert ap_stcr_checkpoint_phase(v2, 19) == "pre_reset_active"
    assert ap_stcr_checkpoint_phase(v2, 20) == "pending_after_epoch_reset"
    assert ap_stcr_checkpoint_phase(v2, 21) == "post_reset_active"
    assert ap_stcr_should_clear_history_on_reset(v2)


def test_v2_conflict_only_acceptance_endpoints():
    module = _v2_module()
    semantic = torch.tensor([0.0, 0.5, 1.0, 0.0]).view(1, 1, 1, 4)
    temporal = torch.tensor([0.0, 0.5, 1.0, 0.0]).view(1, 1, 1, 4)
    conflict = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 1, 4)
    result = module.compute_conflict_only_acceptance(
        semantic, temporal, conflict
    )
    torch.testing.assert_close(
        result["semantic_negative_37"],
        torch.tensor([1.0, 0.0, 0.0, 1.0]).view(1, 1, 1, 4),
    )
    torch.testing.assert_close(
        result["temporal_negative_37"],
        torch.tensor([1.0, 0.0, 0.0, 1.0]).view(1, 1, 1, 4),
    )
    torch.testing.assert_close(
        result["local_acceptance_37"],
        torch.tensor([0.65, 1.0, 1.0, 1.0]).view(1, 1, 1, 4),
    )
    assert float(result["local_acceptance_37"].min()) >= 0.65
    assert float(result["local_acceptance_37"].max()) <= 1.0


def test_v2_source_conflict_uses_strict_fixed_threshold():
    module = _v2_module()
    fixed = torch.tensor([0.49, 0.50, 0.51]).view(1, 1, 1, 3)
    teacher = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
    result = module.compute_source_conflict(fixed, teacher)
    torch.testing.assert_close(
        result["fixed_hard_37"],
        torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3),
    )
    torch.testing.assert_close(
        result["source_conflict_37"],
        torch.tensor([0.0, 1.0, 1.0]).view(1, 1, 1, 3),
    )


def test_v2_empty_history_is_neutral():
    module = _v2_module()
    teacher = torch.full((1, 1, 37, 37), 0.8)
    fixed = torch.full_like(teacher, 0.2)
    result = module.compute_temporal_support(
        teacher,
        fixed,
        torch.tensor([0]),
        ["TR-CAMO"],
        ["sample_0"],
    )
    assert not bool(result["history_valid"].item())
    torch.testing.assert_close(
        result["temporal_support"],
        torch.full_like(result["temporal_support"], 0.5),
    )


def test_v2_target_global_endpoints():
    module = _v2_module()
    fixed = torch.rand(1, 1, 68, 68)
    teacher = (torch.rand_like(fixed) > 0.5).float()
    acceptance = torch.ones(1, 1, 37, 37)
    alpha_zero = module.build_target(fixed, teacher, 0.0, acceptance)
    assert torch.equal(alpha_zero["mixed_target_68"], fixed)
    alpha_one = module.build_target(fixed, teacher, 1.0, acceptance)
    assert torch.equal(alpha_one["mixed_target_68"], teacher)


def test_history_clear_tracks_actual_teacher_reset():
    ap = {
        "clear_history_on_teacher_reset": True,
    }
    keep_teacher = SimpleNamespace(
        AP_STCR=ap,
        FINETUNE_RESET_TEACHER=False,
    )
    reset_teacher = SimpleNamespace(
        AP_STCR=ap,
        FINETUNE_RESET_TEACHER=True,
    )
    assert not ap_stcr_should_clear_history_on_reset(keep_teacher)
    assert ap_stcr_should_clear_history_on_reset(reset_teacher)


def test_v1_reciprocal_path_is_unchanged():
    config = {
        "enabled": True,
        "evidence_resolution": 37,
        "loss_resolution": 68,
        "fg_anchor_ratio": 0.20,
        "bg_anchor_ratio": 0.20,
        "min_fg_anchors": 4,
        "min_bg_anchors": 4,
        "max_fg_anchors": 64,
        "max_bg_anchors": 64,
        "prefer_dabe_background_seed": True,
        "tau_delta": 0.25,
        "tau_margin": 0.50,
        "temporal_window": 3,
        "tau_temporal": 0.20,
        "lambda_semantic": 1.0,
        "lambda_temporal": 1.0,
        "eps": 1e-6,
    }
    module = AnchorPropagatedSemanticTemporalCorrection(
        config, [("TR-CAMO", "sample_0")]
    )
    zeros = torch.zeros(1, 1, 2, 2)
    acceptance = module.compute_local_acceptance(zeros, zeros)
    torch.testing.assert_close(
        acceptance, torch.full_like(acceptance, 1.0 / 3.0)
    )
    temporal = module.compute_temporal_support(
        torch.full((1, 1, 37, 37), 0.8),
        torch.full((1, 1, 37, 37), 0.2),
        torch.tensor([0]),
        ["TR-CAMO"],
        ["sample_0"],
    )
    torch.testing.assert_close(
        temporal["temporal_support"],
        torch.ones_like(temporal["temporal_support"]),
    )
