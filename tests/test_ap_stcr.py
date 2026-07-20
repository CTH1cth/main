from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from common.ap_stcr import (
    AnchorPropagatedSemanticTemporalCorrection,
    strict_teacher_binary,
)
from common.utils import config_to_dict, load_config
from train import (
    ap_stcr_checkpoint_phase,
    build_ap_stcr_segmentation_group,
    get_dabe_pu_despl_schedule,
    validate_ap_stcr_config,
)


def _config():
    return {
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


def _module(num_samples=2):
    keys = [("TR-CAMO", f"sample_{index}") for index in range(num_samples)]
    return AnchorPropagatedSemanticTemporalCorrection(_config(), keys)


def test_ap_stcr_config_preserves_linear30_protocol():
    base = load_config(
        "configs/"
        "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_"
        "long50_lrfloor_2e5_sw_ones_noecst_linear30.py"
    )
    ap = load_config(
        "configs/"
        "dinov1_s8_dabepu_v11_apstcr_dagp_uncgate_ndr_"
        "long50_lrfloor_2e5.py"
    )
    assert validate_ap_stcr_config(ap)
    base_dict = config_to_dict(base)
    ap_dict = config_to_dict(ap)
    changed = {
        name
        for name in set(base_dict) | set(ap_dict)
        if base_dict.get(name) != ap_dict.get(name)
    }
    assert changed == {
        "AP_STCR",
        "EXP_NAME",
        "SUPERVISION_MODE",
        "USE_AP_STCR",
    }
    expected = {
        1: (1.0, 0.0),
        7: (0.7964285714285715, 0.20357142857142854),
        15: (0.525, 0.475),
        29: (0.05, 0.95),
        30: (0.0, 1.0),
        50: (0.0, 1.0),
    }
    for epoch, pair in expected.items():
        actual = get_dabe_pu_despl_schedule(epoch, ap)
        assert actual == pytest.approx(pair, abs=1e-12)
    assert ap_stcr_checkpoint_phase(ap, 28) == "pre_reset_active"
    assert ap_stcr_checkpoint_phase(ap, 29) == "pending_after_epoch_reset"
    assert ap_stcr_checkpoint_phase(ap, 30) == "post_reset_active"


def test_anchor_selection_is_bounded_stable_and_disjoint():
    module = _module()
    target = torch.zeros(2, 1, 37, 37)
    target[0, 0].view(-1)[:100] = torch.linspace(0.51, 1.0, 100)
    target[1, 0, 3] = 0.9
    background = torch.zeros_like(target)
    background[:, :, 20:, :] = 1.0

    first = module.build_anchor_masks(target, background)
    second = module.build_anchor_masks(target, background)
    for lhs, rhs in zip(first[:2], second[:2]):
        assert torch.equal(lhs, rhs)
    fg, bg, sources = first
    assert not bool((fg & bg).any())
    assert torch.all(fg.flatten(1).sum(1) >= 4)
    assert torch.all(fg.flatten(1).sum(1) <= 64)
    assert torch.all(bg.flatten(1).sum(1) >= 4)
    assert torch.all(bg.flatten(1).sum(1) <= 64)
    assert sources == ["dabe_seed:bg_anchor_37"] * 2


def test_strict_teacher_binary_uses_greater_than():
    probability = torch.tensor([0.49, 0.50, 0.51]).view(1, 1, 1, 3)
    result = strict_teacher_binary(probability)
    assert torch.equal(
        result,
        torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3),
    )
    assert not result.requires_grad


def test_target_endpoints_and_full_support_global_equivalence():
    module = _module(num_samples=1)
    fixed = torch.rand(1, 1, 68, 68)
    teacher = (torch.rand_like(fixed) > 0.5).float()
    ones = torch.ones(1, 1, 37, 37)

    alpha_zero = module.build_target(fixed, teacher, 0.0, ones)
    assert torch.equal(alpha_zero["mixed_target_68"], fixed)

    alpha_one = module.build_target(fixed, teacher, 1.0, ones)
    assert torch.equal(alpha_one["mixed_target_68"], teacher)

    alpha = 0.37
    mixed = module.build_target(fixed, teacher, alpha, ones)
    expected = (1.0 - alpha) * fixed + alpha * teacher
    torch.testing.assert_close(mixed["mixed_target_68"], expected)
    assert not mixed["mixed_target_68"].requires_grad


def test_temporal_history_is_past_only_and_resettable():
    module = _module(num_samples=1)
    sample_indices = torch.tensor([0])
    datasets = ["TR-CAMO"]
    stems = ["sample_0"]
    current = torch.full((1, 1, 37, 37), 0.8)
    fixed = torch.full_like(current, 0.2)

    before = module.compute_temporal_support(
        current, fixed, sample_indices, datasets, stems
    )
    assert not bool(before["history_valid"].item())
    assert torch.equal(
        before["temporal_support"],
        torch.ones_like(before["temporal_support"]),
    )

    module.update_history(
        sample_indices, datasets, stems, current, epoch=1
    )
    after = module.history_bank.fetch(
        sample_indices, datasets, stems, device=current.device
    )
    assert bool(after["history_valid"].item())
    assert int(after["history_count"].item()) == 1
    torch.testing.assert_close(after["history_mean"], current)

    with pytest.raises(RuntimeError, match="same sample twice"):
        module.update_history(
            sample_indices, datasets, stems, current, epoch=1
        )

    module.clear_temporal_history()
    cleared = module.history_bank.fetch(
        sample_indices, datasets, stems, device=current.device
    )
    assert not bool(cleared["history_valid"].item())
    assert int(cleared["history_count"].item()) == 0


def test_batch_outputs_are_detached_and_finite():
    torch.manual_seed(3407)
    module = _module(num_samples=1)
    feature = torch.randn(1, 384, 37, 37, requires_grad=True)
    fixed37 = torch.rand(1, 1, 37, 37)
    fixed68 = F.interpolate(
        fixed37, size=(68, 68), mode="bilinear", align_corners=False
    )
    background = (fixed37 < 0.25).float()
    teacher_soft = torch.rand(1, 1, 68, 68, requires_grad=True)
    teacher_binary = strict_teacher_binary(teacher_soft)
    result = module.build_batch(
        dino_features=feature,
        fixed_pseudo_37=fixed37,
        fixed_pseudo_68=fixed68,
        dabe_background_seed_37=background,
        teacher_soft_68=teacher_soft,
        teacher_binary_68=teacher_binary,
        global_teacher_ratio=0.5,
        sample_indices=torch.tensor([0]),
        datasets=["TR-CAMO"],
        stems=["sample_0"],
    )
    for value in result.values():
        if torch.is_tensor(value):
            assert not value.requires_grad
            if value.is_floating_point():
                assert bool(torch.isfinite(value).all())


def test_final_coarse_base_use_same_mixed_target_and_normalization():
    torch.manual_seed(3407)
    cfg = SimpleNamespace(
        USE_AP_STCR=True,
        SUPERVISION_MODE="ap_stcr",
        USE_NDR_COARSE_AUX=True,
        USE_BASE_AUX_LOSS=True,
        LAMBDA_NDR_COARSE_AUX=0.5,
        LAMBDA_BASE_AUX=0.5,
        LAMBDA_BASE_AUX_AFTER_RESET=0.3,
        FINETUNE_RESET_EPOCH=29,
        FINETUNE_RESET_TIMING="after_epoch",
        LOSS_SIZE=68,
    )
    final = torch.randn(2, 1, 68, 68, requires_grad=True)
    coarse = torch.randn(2, 1, 68, 68, requires_grad=True)
    base = torch.randn(2, 1, 37, 37, requires_grad=True)
    target = torch.rand(2, 1, 68, 68)
    output = {
        "coarse_logits_68": coarse,
        "base_logits": base,
    }

    result = build_ap_stcr_segmentation_group(
        cfg, epoch=20, student_out=output, student_logits=final,
        mixed_target=target,
    )
    expected_final = F.binary_cross_entropy_with_logits(final, target)
    expected_coarse = F.binary_cross_entropy_with_logits(coarse, target)
    expected_base = F.binary_cross_entropy_with_logits(
        F.interpolate(
            base,
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        ),
        target,
    )
    expected = (
        expected_final + 0.5 * expected_coarse + 0.5 * expected_base
    ) / 2.0
    torch.testing.assert_close(result["loss"], expected)

    post_reset = build_ap_stcr_segmentation_group(
        cfg, epoch=30, student_out=output, student_logits=final,
        mixed_target=target,
    )
    expected_post = (
        expected_final + 0.5 * expected_coarse + 0.3 * expected_base
    ) / 1.8
    torch.testing.assert_close(post_reset["loss"], expected_post)
