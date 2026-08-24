from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.gbsp_core_variants import NUM_PATCHES, PCARankSelector, fit_pca_from_svd, score_all_patches
from models.gbsp_cvbr_weighted_pca import (
    build_cvbr_soft_weights,
    fit_weighted_affine_subspace,
    projector_distance,
)


def test_cvbr_top_fraction_is_exact_stable_and_unscored_stays_one():
    score = torch.tensor([.1, .9, .8, .8, .4, .7, .3, .2, .6, .5, .0, .0])
    valid = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    result = build_cvbr_soft_weights(score, valid, top_frac=.20, low_weight=.25)
    assert result.num_valid == 10
    assert result.num_suppressed == 2
    assert torch.equal(torch.where(result.suppressed_mask)[0], torch.tensor([1, 2]))
    assert torch.equal(result.weights[-2:], torch.ones(2))
    assert math.isnan(float(result.risk_percentile[-1]))
    assert float(result.risk_percentile[1]) == pytest.approx(1.0)


def test_registered_soft_and_trim_weights_have_expected_counts():
    score = torch.arange(136, dtype=torch.float32)
    valid = torch.ones(136, dtype=torch.bool)
    soft = build_cvbr_soft_weights(score, valid, .10, .25)
    trim = build_cvbr_soft_weights(score, valid, .10, 0.0)
    assert soft.num_suppressed == trim.num_suppressed == 14
    assert int((soft.weights == .25).sum()) == 14
    assert int((trim.weights == 0).sum()) == 14
    assert trim.effective_sample_size == pytest.approx(122.0)


def test_uniform_weighted_pca_reproduces_vanilla_residual():
    generator = torch.Generator().manual_seed(20260814)
    background = F.normalize(torch.randn(411, 48, generator=generator), dim=1)
    query = F.normalize(torch.randn(NUM_PATCHES, 48, generator=generator), dim=1)
    vanilla = fit_pca_from_svd(background, "fixed", PCARankSelector(), 8)
    weighted = fit_weighted_affine_subspace(background, torch.ones(411), rank=8).fit
    assert torch.equal(vanilla.mean, weighted.mean)
    assert projector_distance(vanilla.basis, weighted.basis) < 1e-6
    left, right = score_all_patches(query, vanilla), score_all_patches(query, weighted)
    assert float((left - right).abs().max()) < 1e-5


def test_weighted_center_svd_and_squared_query_residual_are_exact():
    generator = torch.Generator().manual_seed(9)
    background = F.normalize(torch.randn(30, 20, generator=generator), dim=1)
    weights = torch.linspace(0.0, 1.0, 30)
    result = fit_weighted_affine_subspace(background, weights, rank=5)
    expected_mean = (weights[:, None] * background).sum(0) / weights.sum()
    assert torch.allclose(result.fit.mean, expected_mean, atol=1e-7)
    query = F.normalize(torch.randn(NUM_PATCHES, 20, generator=generator), dim=1)
    centered = query - result.fit.mean
    expected = (centered - (centered @ result.fit.basis) @ result.fit.basis.t()).square().sum(1)
    assert torch.allclose(score_all_patches(query, result.fit), expected, atol=1e-6)
    assert result.num_positive_weight == 29


def test_invalid_weighting_inputs_are_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        build_cvbr_soft_weights(torch.ones(3), torch.ones(2, dtype=torch.bool), .1, .25)
    with pytest.raises(ValueError, match="positive top_frac"):
        build_cvbr_soft_weights(torch.ones(3), torch.zeros(3, dtype=torch.bool), .1, .25)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_weighted_affine_subspace(torch.randn(20, 10), -torch.ones(20), rank=3)

