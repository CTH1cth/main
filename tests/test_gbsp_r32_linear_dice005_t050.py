from types import SimpleNamespace

import torch

from common.dabev2hard_static_only import (
    validate_dabev2hard_static_only_config,
)
from common.r1hard_linear_pure_student import (
    GBSP_R16_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_DICE005_T054_SEED2027_CONFIG_PATH,
    GBSP_R32_LINEAR_NODICE_T050_SEED2027_CONFIG_PATH,
    is_r1hard_linear_pure_student_config,
)
from common.utils import load_config
from model import SimpleConvSegHead, build_seg_head
from train import (
    add_lcic_soft_dice_loss,
    build_gbsp_confidence_bce_weight_map,
    dabe_static_bce_with_logits,
    gbsp_bias_orthogonal_correlation,
    gbsp_ssboc_static_loss,
    lcic_soft_dice_loss,
    make_single_feature_model_input,
)


def test_r32_linear_ssboc_control_is_parameter_free_and_audited():
    cfg = load_config(
        GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
    )
    assert cfg.EXP_NAME == "34-gbsp-r32-1x1-ssboc-dice005-t050-seed2027"
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.GBSP_FIXED_PCA_RANK == 32
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert cfg.GBSP_SSBOC_VARIANT
    assert not getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False)
    assert not hasattr(cfg, "GBSP_SSBOC_WEIGHT")
    assert not hasattr(cfg, "GBSP_SSBOC_MARGIN")
    assert not hasattr(cfg, "GBSP_SSBOC_TOPK")
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["new_tunable_hyperparameters"] == 0


def test_ssboc_correlation_is_bias_and_positive_scale_invariant():
    score = torch.tensor(
        [[[[0.05, 0.20], [0.70, 0.95]]]], dtype=torch.float32
    )
    logits = torch.tensor(
        [[[[-1.5, -0.2], [0.4, 1.1]]]], dtype=torch.float32
    )
    reference = gbsp_bias_orthogonal_correlation(logits, score)
    shifted_scaled = gbsp_bias_orthogonal_correlation(
        3.7 * logits + 11.0,
        score,
    )
    assert torch.allclose(reference, shifted_scaled, atol=1e-6)

    bias = torch.tensor(2.5, requires_grad=True)
    correlation = gbsp_bias_orthogonal_correlation(logits + bias, score)
    (1.0 - correlation).sum().backward()
    assert bias.grad is not None
    assert abs(float(bias.grad.item())) < 1e-6


def test_ssboc_matches_per_image_self_scaled_formula_and_backpropagates():
    cfg = load_config(
        GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
    )
    score = torch.tensor(
        [
            [[[0.10, 0.30], [0.70, 0.90]]],
            [[[0.05, 0.80], [0.20, 0.65]]],
        ],
        dtype=torch.float32,
    )
    target = (score > 0.50).float()
    logits = torch.tensor(
        [
            [[[-0.7, 0.1], [0.2, 1.0]]],
            [[[-0.4, 0.6], [-0.1, 0.3]]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight_map = torch.ones_like(target)
    actual, stats = gbsp_ssboc_static_loss(
        logits,
        target,
        score,
        weight_map,
        cfg,
    )
    bce_per_image = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).flatten(1).mean(dim=1)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = (probability + target).flatten(1).sum(dim=1)
    dice_per_image = 1.0 - (
        (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    )
    base_per_image = bce_per_image + 0.05 * dice_per_image
    correlation = gbsp_bias_orthogonal_correlation(logits, score)
    expected = (
        base_per_image + base_per_image.detach() * (1.0 - correlation)
    ).mean()
    assert torch.allclose(actual, expected)
    assert abs(stats["correlation_mean"] - float(correlation.mean())) < 1e-6
    assert stats["valid_ratio"] == 1.0
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ssboc_skips_degenerate_correlation_without_unstable_gradient():
    cfg = load_config(
        GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
    )
    score = torch.tensor(
        [[[[0.05, 0.20], [0.70, 0.95]]]], dtype=torch.float32
    )
    target = (score > 0.50).float()
    logits = torch.zeros_like(score, requires_grad=True)
    loss, stats = gbsp_ssboc_static_loss(
        logits,
        target,
        score,
        torch.ones_like(target),
        cfg,
    )
    assert stats["valid_ratio"] == 0.0
    assert stats["auxiliary_loss"] == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_r32_linear_confidence_bce_control_is_matched_and_audited():
    cfg = load_config(
        GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH
    )
    assert cfg.EXP_NAME == (
        "34-gbsp-r32-1x1-confbce-f050-g1-dice005-t050-seed2027"
    )
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.GBSP_FIXED_PCA_RANK == 32
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert cfg.GBSP_CONFIDENCE_BCE_VARIANT
    assert cfg.GBSP_CONFIDENCE_BCE_FLOOR == 0.50
    assert cfg.GBSP_CONFIDENCE_BCE_GAMMA == 1.0
    assert cfg.GBSP_CONFIDENCE_BCE_CLASSWISE_NORMALIZE
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["loss"] == (
        "classwise_normalized_GBSP_confidence_BCE_floor0.50_"
        "gamma1.0_plus_0.05_SoftDice"
    )


def test_gbsp_confidence_bce_weights_distance_but_preserves_class_mass():
    cfg = load_config(
        GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH
    )
    score = torch.tensor(
        [[[[0.05, 0.30, 0.45, 0.49], [0.51, 0.55, 0.70, 0.95]]]],
        dtype=torch.float32,
    )
    target = (score > 0.50).float()
    weight = build_gbsp_confidence_bce_weight_map(
        score,
        target,
        torch.tensor([0.50]),
        cfg,
    )
    assert weight.shape == target.shape
    assert not weight.requires_grad
    assert torch.isfinite(weight).all()
    assert (weight > 0.0).all()
    assert torch.allclose(weight[target == 0].mean(), torch.tensor(1.0))
    assert torch.allclose(weight[target == 1].mean(), torch.tensor(1.0))
    assert weight[0, 0, 0, 0] > weight[0, 0, 0, 3]
    assert weight[0, 0, 1, 3] > weight[0, 0, 1, 0]


def test_gbsp_confidence_bce_enters_loss_and_keeps_soft_dice():
    cfg = load_config(
        GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH
    )
    score = torch.tensor(
        [[[[0.10, 0.49], [0.51, 0.90]]]], dtype=torch.float32
    )
    target = (score > 0.50).float()
    weight = build_gbsp_confidence_bce_weight_map(
        score, target, 0.50, cfg
    )
    logits = torch.tensor(
        [[[[1.0, -0.5], [0.25, -1.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    actual = dabe_static_bce_with_logits(
        logits, target, weight, cfg
    )
    bce_map = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    expected = (bce_map * weight).mean()
    expected = expected + 0.05 * lcic_soft_dice_loss(logits, target)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_r32_linear_control_is_matched_and_audited():
    cfg = load_config(GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH)
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.GBSP_FIXED_PCA_RANK == 32
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert cfg.PURE_STUDENT_RESET_PERMANENTLY_DISABLED
    assert cfg.FINETUNE_RESET_EPOCH == 0
    assert is_r1hard_linear_pure_student_config(cfg)
    assert validate_dabev2hard_static_only_config(cfg)["status"] == "PASS"


def test_r32_linear_control_builds_one_conv_and_keeps_dice():
    cfg = load_config(GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH)
    head = build_seg_head(384, cfg)
    assert isinstance(head, SimpleConvSegHead)
    assert sum(parameter.numel() for parameter in head.parameters()) == 385

    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    target = torch.ones_like(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    combined = add_lcic_soft_dice_loss(bce, logits, target, cfg)
    expected = bce + 0.05 * lcic_soft_dice_loss(logits, target)
    assert torch.allclose(combined, expected)


def test_r32_linear_scaled_l2_normalizes_before_resize():
    cfg = load_config(
        GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_CONFIG_PATH
    )
    assert cfg.EXP_NAME == (
        "33-gbsp-r32-1x1-l2s1059-dice005-t050-seed2027"
    )
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.SIMPLE_FEATURE_L2_NORMALIZE is True
    assert cfg.SIMPLE_FEATURE_L2_SCALE == 10.59
    assert cfg.SIMPLE_FEATURE_L2_EPS == 1e-12
    assert cfg.GBSP_FIXED_PCA_RANK == 32
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert "L2_fixed_norm10.59" in report["contract"]["feature"]

    feature = torch.empty(1, 2, 37, 37)
    feature[:, 0].fill_(3.0)
    feature[:, 1].fill_(4.0)
    normalized = make_single_feature_model_input(cfg, feature)
    assert list(normalized.shape) == [1, 2, 68, 68]
    expected = torch.tensor([0.6, 0.8]) * 10.59
    assert torch.allclose(normalized[0, :, 0, 0], expected, atol=1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(normalized, dim=1),
        torch.full((1, 68, 68), 10.59),
        atol=1e-5,
    )

    head = build_seg_head(384, cfg)
    assert isinstance(head, SimpleConvSegHead)
    assert sum(parameter.numel() for parameter in head.parameters()) == 385


def test_r32_linear_t054_changes_only_pseudo_threshold():
    cfg = load_config(GBSP_R32_LINEAR_DICE005_T054_SEED2027_CONFIG_PATH)
    assert cfg.EXP_NAME == "33-gbsp-r32-1x1-dice005-t054-seed2027"
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.54,
        "TR-COD10K": 0.54,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert cfg.LCIC_SOFT_DICE_VARIANT
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["pseudo"] == (
        "fixed_R32_minmax_strict_gt_0.54_both_datasets"
    )
    assert report["contract"]["prediction_threshold"] == 0.50


def test_r16_linear_t050_changes_only_fixed_rank_cache():
    cfg = load_config(GBSP_R16_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH)
    assert cfg.EXP_NAME == "33-gbsp-r16-1x1-dice005-t050-seed2027"
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.GBSP_FIXED_PCA_RANK == 16
    assert cfg.DABE_CLEAN_GBSP_FIXED_PCA_RANK == 16
    assert cfg.GBSP_VERSION == "gbsp_pca_absmm_r16_v1"
    assert cfg.DABE_CLEAN_GBSP_VERSION == "gbsp_pca_absmm_r16_v1"
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.THRESHOLD == 0.50
    assert cfg.SEED == 2027
    assert cfg.LCIC_SOFT_DICE_VARIANT
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.05
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["pseudo"] == (
        "fixed_R16_minmax_strict_gt_0.50_both_datasets"
    )


def test_r32_linear_nodice_control_is_matched_and_audited():
    cfg = load_config(GBSP_R32_LINEAR_NODICE_T050_SEED2027_CONFIG_PATH)
    assert cfg.EXP_NAME == "33-gbsp-r32-1x1-nodice-t050-seed2027"
    assert cfg.HEAD_TYPE == "simple"
    assert cfg.GBSP_FIXED_PCA_RANK == 32
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.SEED == 2027
    assert not cfg.LCIC_SOFT_DICE_VARIANT
    assert cfg.LCIC_SOFT_DICE_WEIGHT == 0.0
    assert cfg.PURE_STUDENT_RESET_PERMANENTLY_DISABLED
    assert cfg.FINETUNE_RESET_EPOCH == 0
    assert is_r1hard_linear_pure_student_config(cfg)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["loss"] == "BCEWithLogits_mean"

    head = build_seg_head(384, cfg)
    assert isinstance(head, SimpleConvSegHead)
    assert sum(parameter.numel() for parameter in head.parameters()) == 385

    logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
    target = torch.ones_like(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    combined = add_lcic_soft_dice_loss(bce, logits, target, cfg)
    assert combined is bce


def test_linear_soft_dice_control_is_not_a_general_bypass():
    cfg = SimpleNamespace(
        LCIC_SOFT_DICE_VARIANT=True,
        LCIC_SOFT_DICE_WEIGHT=0.05,
        GBSP_LCIC_V1=False,
        GBSP_LINEAR_SOFT_DICE_CONTROL=False,
    )
    logits = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    try:
        add_lcic_soft_dice_loss(bce, logits, target, cfg)
    except RuntimeError as error:
        assert "GBSP_LINEAR_SOFT_DICE_CONTROL" in str(error)
    else:
        raise AssertionError("Unaudited linear Soft-Dice control was accepted")
