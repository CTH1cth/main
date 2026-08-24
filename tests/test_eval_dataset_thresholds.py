from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.utils import load_config
from common.metrics import CODMetrics
from eval import (
    binary_hysteresis_mask,
    finalize_eval_metric_results,
    float_otsu_threshold,
    logit_gmm_threshold,
    logit_triangle_threshold,
    otsu_anchor_hysteresis_mask,
    parse_dataset_threshold_overrides,
    probability_valley_threshold,
    resolve_eval_binary_threshold,
)


def test_lcic_config_keeps_uniform_default_output_threshold():
    cfg = load_config(
        "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice010_t050_seed2027.py"
    )
    assert resolve_eval_binary_threshold(cfg, "CHAMELEON") == 0.50
    assert resolve_eval_binary_threshold(cfg, "TE-CAMO") == 0.50
    assert resolve_eval_binary_threshold(cfg, "TE-COD10K") == 0.50
    assert resolve_eval_binary_threshold(cfg, "NC4K") == 0.50


def test_one_run_threshold_override_changes_only_named_dataset():
    cfg = SimpleNamespace(THRESHOLD=0.47)
    overrides = parse_dataset_threshold_overrides(
        ["TE-CAMO=0.50", "TE-COD10K=0.60"],
        ["CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"],
    )
    assert resolve_eval_binary_threshold(cfg, "CHAMELEON", overrides) == 0.47
    assert resolve_eval_binary_threshold(cfg, "TE-CAMO", overrides) == 0.50
    assert resolve_eval_binary_threshold(cfg, "TE-COD10K", overrides) == 0.60
    assert resolve_eval_binary_threshold(cfg, "NC4K", overrides) == 0.47


def test_one_run_global_threshold_override_changes_all_datasets():
    cfg = SimpleNamespace(THRESHOLD=0.50)
    for dataset_name in ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"):
        assert resolve_eval_binary_threshold(
            cfg, dataset_name, default_threshold=0.60
        ) == 0.60


def test_probability_valley_threshold_is_bounded_and_nonfallback_for_two_modes():
    generator = torch.Generator().manual_seed(7)
    background = 0.12 + 0.05 * torch.randn(9000, generator=generator)
    foreground = 0.82 + 0.05 * torch.randn(1000, generator=generator)
    probability = torch.cat((background, foreground)).clamp(0.0, 1.0)
    threshold, diagnostics = probability_valley_threshold(probability)
    assert not diagnostics["fallback"], diagnostics
    assert 0.45 <= threshold <= 0.60
    assert diagnostics["left_peak"] < diagnostics["raw_threshold"]
    assert diagnostics["raw_threshold"] < diagnostics["right_peak"]


def test_probability_valley_threshold_falls_back_for_single_class_map():
    probability = torch.full((1, 1, 32, 32), 0.10)
    threshold, diagnostics = probability_valley_threshold(probability)
    assert threshold == 0.50
    assert diagnostics["fallback"]


def test_float_otsu_threshold_separates_float_probability_modes_without_uint8():
    probability = torch.tensor(
        [0.101, 0.103, 0.107, 0.111, 0.823, 0.827, 0.829, 0.833],
        dtype=torch.float32,
    )
    threshold, diagnostics = float_otsu_threshold(probability)
    assert not diagnostics["fallback"], diagnostics
    assert 0.111 < threshold < 0.823
    assert 0.0 <= threshold <= 1.0
    assert not np.isclose(threshold * 255.0, round(threshold * 255.0))


def test_float_otsu_threshold_falls_back_for_constant_probability_map():
    probability = torch.full((1, 1, 16, 16), 0.37)
    threshold, diagnostics = float_otsu_threshold(
        probability, fallback_threshold=0.50
    )
    assert threshold == 0.50
    assert diagnostics["fallback"]
    assert diagnostics["reason"] == "constant_probability_map"


def test_logit_triangle_threshold_is_float_gt_free_and_deterministic():
    generator = torch.Generator().manual_seed(23)
    background_logits = -3.0 + 0.55 * torch.randn(9000, generator=generator)
    foreground_logits = 1.5 + 0.75 * torch.randn(1000, generator=generator)
    probability = torch.sigmoid(torch.cat((background_logits, foreground_logits)))

    first, first_diagnostics = logit_triangle_threshold(probability)
    second, second_diagnostics = logit_triangle_threshold(probability)

    assert not first_diagnostics["fallback"], first_diagnostics
    assert 0.0 < first < 1.0
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["tail_direction"] in {"left", "right"}


def test_logit_triangle_threshold_falls_back_for_constant_map():
    threshold, diagnostics = logit_triangle_threshold(
        torch.full((16, 16), 0.37), fallback_threshold=0.50
    )
    assert threshold == 0.50
    assert diagnostics["fallback"]
    assert diagnostics["reason"] == "constant_probability_map"


def test_logit_gmm_threshold_separates_imbalanced_probability_modes():
    generator = torch.Generator().manual_seed(19)
    background_logits = -3.0 + 0.70 * torch.randn(9000, generator=generator)
    foreground_logits = 2.0 + 0.55 * torch.randn(1000, generator=generator)
    probability = torch.sigmoid(torch.cat((background_logits, foreground_logits)))

    threshold, diagnostics = logit_gmm_threshold(probability)

    assert not diagnostics["fallback"], diagnostics
    assert 0.05 < threshold < 0.90
    assert diagnostics["background_logit_mean"] < diagnostics["foreground_logit_mean"]
    assert diagnostics["background_weight"] > diagnostics["foreground_weight"]
    assert diagnostics["standardized_separation"] > 1.0


def test_logit_gmm_threshold_is_deterministic_and_falls_back_for_constant_map():
    probability = torch.linspace(0.01, 0.99, 4096)
    first, first_diagnostics = logit_gmm_threshold(probability)
    second, second_diagnostics = logit_gmm_threshold(probability)
    assert first == second
    assert first_diagnostics["reason"] == second_diagnostics["reason"]

    fallback, diagnostics = logit_gmm_threshold(torch.full((32, 32), 0.37))
    assert fallback == 0.50
    assert diagnostics["fallback"]
    assert diagnostics["reason"] == "constant_probability_map"


def test_binary_hysteresis_keeps_only_low_region_connected_to_high_seed():
    probability = torch.tensor(
        [
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.80, 0.45, 0.45, 0.05, 0.45],
            [0.05, 0.05, 0.05, 0.45, 0.05, 0.45],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        ]
    )
    mask, diagnostics = binary_hysteresis_mask(
        probability, low_threshold=0.40, high_threshold=0.60
    )
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(mask, expected)
    assert diagnostics["component_count"] == 2
    assert diagnostics["kept_component_count"] == 1


def test_otsu_anchor_hysteresis_is_gt_free_bounded_and_shape_preserving():
    probability = torch.zeros(1, 1, 20, 20)
    probability[:, :, 5:15, 5:15] = 0.45
    probability[:, :, 8:12, 8:12] = 0.90
    mask, diagnostics = otsu_anchor_hysteresis_mask(
        probability, anchor_threshold=0.50
    )
    assert mask.shape == probability.shape
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert diagnostics["low_threshold"] <= diagnostics["high_threshold"]
    assert diagnostics["output_area"] >= diagnostics["seed_area"]
    assert diagnostics["output_area"] <= diagnostics["candidate_area"]


def test_probability_metrics_keep_binary_acc_and_iou_reference():
    gt = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    probability = torch.tensor([[[[0.0, 0.4], [0.6, 1.0]]]])
    binary = (probability > 0.5).float()
    probability_metrics = CODMetrics()
    binary_metrics = CODMetrics()
    probability_metrics.step(gt, probability)
    binary_metrics.step(gt, binary)

    result, binary_reference = finalize_eval_metric_results(
        probability_metrics, binary_metrics
    )

    assert result["MAE"] == pytest.approx(0.2)
    assert binary_reference["MAE"] == pytest.approx(0.0)
    assert result["ACC"] == pytest.approx(1.0)
    assert result["mIOU"] == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_eval_threshold_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match="Invalid evaluation threshold"):
        parse_dataset_threshold_overrides(
            [f"TE-COD10K={threshold}"], ["TE-COD10K"]
        )
