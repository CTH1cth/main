from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.dataset import _load_dabe_clean_dabe_v2_static
from common.r1hard_linear_pure_student import GBSP_CF_BRC_HC_CONFIG_PATH
from common.utils import load_config
from models.gbsp_thresholding import (
    BackgroundQuantileThreshold,
    BackgroundTailShrinkageThreshold,
    BackgroundTailCalibrator,
    CrossFittedBackgroundResidual,
    HigherCriticismThreshold,
    RobustMADThreshold,
    deterministic_spatial_folds,
)


def test_bts_lambda_zero_exactly_reproduces_fixed_058_and_lambda_half_is_same_domain():
    raw = torch.linspace(0.0, 2.0, 37 * 37).reshape(1, 37, 37)
    minmax = raw / 2.0
    background_z = torch.linspace(-2.0, 2.0, 101)
    locations = torch.log(torch.tensor([0.30, 0.32, 0.34, 0.36, 0.38]))
    scales = torch.full((5,), 0.20)
    fixed = BackgroundTailShrinkageThreshold(shrinkage=0.0).apply(
        background_z, locations, scales, raw, minmax
    )
    adaptive = BackgroundTailShrinkageThreshold(shrinkage=0.5).apply(
        background_z, locations, scales, raw, minmax
    )
    assert fixed.threshold == 0.58
    assert torch.equal(fixed.binary_mask.reshape_as(minmax), minmax > 0.58)
    expected = 0.5 * 0.58 + 0.5 * adaptive.background_minmax_threshold
    assert abs(adaptive.threshold - expected) < 1e-12
    assert 0.29 <= adaptive.threshold <= 0.79


def test_bts_is_deterministic_finite_and_records_minmax_clipping():
    raw = torch.linspace(0.0, 0.1, 37 * 37).reshape(1, 37, 37)
    minmax = raw / 0.1
    background_z = torch.linspace(-1.0, 8.0, 83)
    locations = torch.zeros(5)
    scales = torch.ones(5)
    method = BackgroundTailShrinkageThreshold(shrinkage=0.5)
    first = method.apply(background_z, locations, scales, raw, minmax)
    second = method.apply(background_z, locations, scales, raw, minmax)
    assert first.threshold == second.threshold
    assert torch.equal(first.binary_mask, second.binary_mask)
    assert first.minmax_mapping_clipped
    assert first.background_minmax_threshold == 1.0
    assert math.isfinite(first.background_raw_threshold)


def test_spatial_fold_assignment_is_exactly_deterministic_and_balanced():
    indices = torch.arange(37 * 37)
    first = deterministic_spatial_folds(indices, grid=37, num_folds=5)
    second = deterministic_spatial_folds(indices, grid=37, num_folds=5)
    expected = (
        torch.div(indices, 37, rounding_mode="floor") + 2 * indices.remainder(37)
    ).remainder(5)
    assert torch.equal(first, second)
    assert torch.equal(first, expected)
    counts = torch.bincount(first, minlength=5)
    assert int(counts.max() - counts.min()) <= 1


def test_irregular_spatial_subset_is_rebalanced_without_randomness():
    indices = torch.tensor([0, 1, 2, 3, 4, 37, 38, 74, 111, 148, 185, 222, 259])
    fold_ids = deterministic_spatial_folds(indices)
    counts = torch.bincount(fold_ids, minlength=5)
    assert int(counts.max() - counts.min()) <= 1
    assert bool((counts > 0).all())
    assert torch.equal(fold_ids, deterministic_spatial_folds(indices))


def test_crossfit_has_exactly_one_heldout_assignment_and_no_leakage():
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(37 * 37, 12, generator=generator)
    indices = torch.arange(0, 37 * 37, 13)[:80]
    background = query.index_select(0, indices)
    result = CrossFittedBackgroundResidual(
        num_folds=5,
        grid=37,
        pca_max_rank=8,
        pca_min_rank=1,
        min_cluster_size=2,
    ).fit_score(background, indices, query)

    heldout_all = torch.cat(result.fold_heldout_background_positions)
    assert torch.equal(torch.sort(heldout_all).values, torch.arange(indices.numel()))
    for fit, heldout in zip(
        result.fold_fit_background_positions,
        result.fold_heldout_background_positions,
    ):
        assert not bool(torch.isin(heldout, fit).any())
    assert tuple(result.query_raw_residual_each_fold.shape) == (5, 37 * 37)
    assert tuple(result.query_z_each_fold.shape) == (5, 37 * 37)
    assert tuple(result.background_oof_z.shape) == (indices.numel(),)
    assert float(result.fold_basis_orthonormal_max_error.max()) < 1e-4


def test_background_tail_probability_range_and_monotonicity_for_1000_pairs():
    background = torch.linspace(-3.0, 4.0, 211)
    query = torch.linspace(-5.0, 7.0, 1369)
    result = BackgroundTailCalibrator().calibrate(query, background)
    assert float(result.p_value.min()) > 0.0
    assert float(result.p_value.max()) <= 1.0
    assert bool(torch.isfinite(result.p_value).all())
    assert result.monotonicity_violation_count == 0
    generator = torch.Generator().manual_seed(43)
    left = torch.randint(query.numel(), (1000,), generator=generator)
    right = torch.randint(query.numel(), (1000,), generator=generator)
    higher = query[left] > query[right]
    assert bool((result.p_value[left][higher] <= result.p_value[right][higher]).all())


def test_higher_criticism_is_exactly_reproducible():
    p = torch.linspace(1.0 / 401.0, 1.0, 1369)
    # Add a deterministic excess of low p-values.
    p[:120] = torch.linspace(1.0 / 401.0, 0.025, 120)
    threshold = HigherCriticismThreshold()
    first = threshold.apply(p, background_count=400)
    second = threshold.apply(p, background_count=400)
    assert first.k_star == second.k_star
    assert first.threshold == second.threshold
    assert torch.equal(first.binary_mask, second.binary_mask)
    assert torch.equal(first.score_curve, second.score_curve)
    assert not first.fallback


def test_hc_fallback_and_simple_baselines_follow_declared_inequalities():
    p = torch.full((1369,), 0.8)
    hc = HigherCriticismThreshold().apply(p, background_count=100)
    assert hc.fallback
    assert hc.threshold == 0.05
    assert not bool(hc.binary_mask.any())

    p2 = torch.tensor([0.049, 0.050, 0.051])
    assert torch.equal(
        BackgroundQuantileThreshold(0.05).apply(p2).binary_mask,
        torch.tensor([True, True, False]),
    )
    z = torch.tensor([2.999, 3.0, 3.001])
    assert torch.equal(
        RobustMADThreshold(3.0).apply(z).binary_mask,
        torch.tensor([False, False, True]),
    )


def test_cf_brc_hc_pure_student_config_is_strictly_audited():
    cfg = load_config(GBSP_CF_BRC_HC_CONFIG_PATH)
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["schema"] == "gbsp_cf_brc_hc_linear_pure_student_config_audit_v1"
    assert report["contract"]["source"] == "hc_mask"
    assert report["contract"]["student"] == "single_1x1_conv"
    assert not report["contract"]["teacher_instantiated"]


def test_training_loader_uses_cached_hc_binary_mask_only():
    mask = torch.zeros(1, 37, 37)
    mask[:, 8:29, 11:27] = 1.0
    payload = {
        "dataset": "TR-CAMO",
        "stem": "synthetic",
        "backbone_key": "dinov1-s8",
        "source_dabe_version": "v2",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "gbsp_threshold_version": "gbsp_cf_brc_hc_v1",
        "gt_used_for_generation": False,
        "r1_used_for_generation": False,
        "target_area_prior_used": False,
        "hc_mask": mask,
    }
    cfg = SimpleNamespace(
        BACKBONE_KEY="dinov1-s8",
        LOSS_SIZE=68,
        DABE_CLEAN_STATIC_TARGET_SOURCE="gbsp_cf_brc_hc_hard_68",
        DABE_CLEAN_DABE_V2_VERSION="v2",
        DABE_CLEAN_DABE_V2_SOURCE_KEY="hc_mask",
        DABE_CLEAN_GBSP_THRESHOLD_VERSION="gbsp_cf_brc_hc_v1",
        DABE_CLEAN_GBSP_AUGS=["identity"],
        DABE_CLEAN_DABE_V2_HARD_THRESHOLD=0.5,
        USE_EAOGP=False,
    )
    workdir = Path(__file__).resolve().parents[2] / "workdir"
    with tempfile.TemporaryDirectory(dir=workdir) as temporary:
        cache_path = Path(temporary) / "synthetic.pt"
        torch.save(payload, cache_path)
        result = _load_dabe_clean_dabe_v2_static(
            {"cache_path": str(cache_path)}, "TR-CAMO", "synthetic", cfg
        )
    expected_soft = F.interpolate(
        mask.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    ).squeeze(0)
    assert torch.equal(result["dabe_v2_soft_68"], expected_soft)
    assert torch.equal(result["dabe_v2_hard_68"], (expected_soft > 0.5).float())
