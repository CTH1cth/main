import inspect
import math
from types import SimpleNamespace

import torch

import common.ecst_causal_audit as audit_module
from common.ecst_causal_audit import (
    build_block_shuffle_map,
    build_constant_mean_map,
    build_pixel_permutation_map,
    build_spatial_roll_map,
    compute_bootstrap_ci,
    compute_correction_masks,
    compute_gradient_magnitude_match_scalar,
    compute_logit_gradient_field,
    compute_precision_at_coverage,
    compute_score_ranking_metrics,
    deterministic_roll_shift,
    normalized_weighted_bce,
    plain_mean_bce,
)


def _random_case():
    generator = torch.Generator().manual_seed(20260731)
    logits = torch.randn(3, 1, 68, 68, generator=generator)
    target = (torch.rand(3, 1, 68, 68, generator=generator) > 0.5).float()
    weight = 0.2 + 0.8 * torch.rand(3, 1, 68, 68, generator=generator)
    return logits, target, weight


def test_global_constant_normalized_weight_equals_plain_mean():
    logits, target, weight = _random_case()
    constant = build_constant_mean_map(weight, per_image=False)
    error = abs(
        float(normalized_weighted_bce(logits, target, constant))
        - float(plain_mean_bce(logits, target))
    )
    assert error <= 1e-7


def test_per_image_constant_is_registered_as_non_equivalent_batch_control():
    logits = torch.tensor([[[[-2.0]]], [[[2.0]]]])
    target = torch.tensor([[[[1.0]]], [[[1.0]]]])
    weight = torch.tensor([[[[0.2]]], [[[1.0]]]])
    per_image = build_constant_mean_map(weight, per_image=True)
    assert abs(
        float(normalized_weighted_bce(logits, target, per_image))
        - float(plain_mean_bce(logits, target))
    ) > 1e-3


def test_gmg_matches_original_gradient_l1():
    logits, target, weight = _random_case()
    original = compute_logit_gradient_field(
        logits, target, weight, mode="normalized_weighted"
    )
    scalar = compute_gradient_magnitude_match_scalar(
        logits, target, weight, norm="l1"
    )
    gmg = compute_logit_gradient_field(
        logits, target, mode="gmg", global_scalar=scalar
    )
    assert abs(float(original.abs().sum() - gmg.abs().sum())) <= 1e-6


def test_gmg_is_only_a_global_multiple_of_plain_gradient():
    logits, target, weight = _random_case()
    scalar = compute_gradient_magnitude_match_scalar(logits, target, weight)
    plain = compute_logit_gradient_field(logits, target, mode="plain")
    gmg = compute_logit_gradient_field(
        logits, target, mode="gmg", global_scalar=scalar
    )
    assert torch.allclose(gmg, scalar * plain, atol=0.0, rtol=0.0)


def test_spatial_roll_preserves_histogram_and_is_repeatable():
    weight = torch.arange(2 * 68 * 68, dtype=torch.float32).reshape(2, 1, 68, 68)
    datasets = ["TR-CAMO", "TR-COD10K"]
    stems = ["a", "b"]
    first = build_spatial_roll_map(weight, datasets, stems)
    second = build_spatial_roll_map(weight, datasets, stems)
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first.flatten(1)).values, torch.sort(weight.flatten(1)).values)
    assert torch.equal(first.mean(dim=(1, 2, 3)), weight.mean(dim=(1, 2, 3)))
    assert torch.equal(first.amin(dim=(1, 2, 3)), weight.amin(dim=(1, 2, 3)))
    assert torch.equal(first.amax(dim=(1, 2, 3)), weight.amax(dim=(1, 2, 3)))


def test_most_samples_receive_different_roll_shifts():
    shifts = {
        deterministic_roll_shift("TR-CAMO", f"sample_{index}")
        for index in range(20)
    }
    assert len(shifts) >= 15
    assert all(17 <= abs(y) <= 51 and 17 <= abs(x) <= 51 for y, x in shifts)


def _sorted_blocks(value, block_size=4):
    channels, height, width = value.shape
    blocks = (
        value.reshape(channels, height // block_size, block_size, width // block_size, block_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, channels * block_size * block_size)
    )
    return sorted(tuple(row.tolist()) for row in blocks)


def test_block_shuffle_preserves_complete_blocks():
    weight = torch.arange(68 * 68, dtype=torch.float32).reshape(1, 1, 68, 68)
    shuffled = build_block_shuffle_map(weight, ["TR-CAMO"], ["sample"])
    assert _sorted_blocks(weight[0]) == _sorted_blocks(shuffled[0])


def test_pixel_permutation_preserves_histogram():
    weight = torch.arange(68 * 68, dtype=torch.float32).reshape(1, 1, 68, 68)
    shuffled = build_pixel_permutation_map(weight, ["TR-CAMO"], ["sample"])
    assert torch.equal(torch.sort(weight.flatten()).values, torch.sort(shuffled.flatten()).values)


def test_correction_directions_and_correctness():
    static = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.float32)
    teacher = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.float32)
    gt = torch.tensor([[[0, 1, 1, 1]]], dtype=torch.float32)
    masks = compute_correction_masks(static, teacher, gt)
    assert torch.equal(masks["add_fg"], torch.tensor([[[False, True, False, False]]]))
    assert torch.equal(masks["erase_fg"], torch.tensor([[[False, False, True, False]]]))
    assert torch.equal(
        masks["correct"][masks["conflict"]],
        (teacher.bool() == gt.bool())[masks["conflict"]],
    )


def test_precision_at_coverage_known_order():
    score = torch.tensor([0.9, 0.8, 0.7, 0.1])
    label = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    rows = compute_precision_at_coverage(score, label, coverages=(0.5, 1.0))
    assert rows[0]["precision"] == 0.5
    assert rows[1]["precision"] == 0.5
    assert rows[0]["correct_correction_retention"] == 0.5


def test_ranking_degenerate_cases_are_na():
    assert compute_score_ranking_metrics(torch.empty(0), torch.empty(0, dtype=torch.bool))["status"] == "N/A"
    assert compute_score_ranking_metrics(torch.tensor([0.1, 0.2]), torch.ones(2, dtype=torch.bool))["reason"] == "ALL_POSITIVE"
    assert compute_score_ranking_metrics(torch.tensor([0.1, 0.2]), torch.zeros(2, dtype=torch.bool))["reason"] == "ALL_NEGATIVE"
    assert compute_score_ranking_metrics(torch.ones(4), torch.tensor([0, 1, 0, 1], dtype=torch.bool))["reason"] == "ALL_SCORES_EQUAL"


def test_bootstrap_is_image_level_and_repeatable():
    records = [{"value": float(index)} for index in range(8)]
    metric = lambda selected: sum(row["value"] for row in selected) / len(selected)
    first = compute_bootstrap_ci(records, metric, repetitions=100)
    second = compute_bootstrap_ci(records, metric, repetitions=100)
    assert first == second
    assert first["bootstrap_unit"] == "image"


def test_training_imported_helper_has_no_gt_reader():
    source = inspect.getsource(audit_module)
    assert "build_image_items" not in source
    assert "Image.open" not in source
    assert "gt_path" not in source.lower()


def test_default_none_mode_preserves_full_ecst_weighted_bce():
    from train import teacher_route_bce_with_logits

    logits, target, weight = _random_case()
    cfg = SimpleNamespace(TEACHER_ROUTING_MODE="ecst", USE_ECST=True)
    actual = teacher_route_bce_with_logits(
        logits, target, weight, cfg, routing_scale=1.0
    )
    expected = normalized_weighted_bce(logits, target, weight)
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)
