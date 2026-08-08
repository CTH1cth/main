from __future__ import annotations

import numpy as np
import pytest
import torch

from models.gbsp_spe_calibration import (
    BoundedResidualCalibrator,
    GammaSPELimit,
    JacksonMudholkarSPELimit,
    PCASpectrumExtractor,
)


def _spectrum(background: torch.Tensor, rank: int):
    centered = background - background.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    return PCASpectrumExtractor().extract(singular, background.shape[0], background.shape[1], rank)


def test_svd_and_explicit_covariance_spectra_match():
    generator = torch.Generator().manual_seed(1)
    background = torch.randn(200, 32, generator=generator, dtype=torch.float64)
    spectrum = _spectrum(background, 8)
    centered = background - background.mean(dim=0)
    explicit = torch.linalg.eigvalsh(centered.t() @ centered / 199).flip(0)
    assert float((spectrum.covariance_eigenvalues - explicit).abs().max()) < 1e-8


def test_direct_residual_matches_discarded_basis_plus_null_space():
    generator = torch.Generator().manual_seed(2)
    background = torch.randn(10, 16, generator=generator, dtype=torch.float64)
    query = torch.randn(37, 16, generator=generator, dtype=torch.float64)
    centered_background = background - background.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered_background, full_matrices=True)
    rank = 4
    centered_query = query - background.mean(dim=0)
    retained = vh[:rank].t()
    direct = (centered_query - (centered_query @ retained) @ retained.t()).square().sum(1)
    sample_discarded = (centered_query @ vh[rank:9].t()).square().sum(1)
    null_energy = (centered_query @ vh[9:].t()).square().sum(1)
    assert float((direct - sample_discarded - null_energy).abs().max()) < 1e-6


def test_calibration_mask_is_raw_limit_decision():
    raw = torch.rand(10000, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 4
    background = torch.randn(200, 16, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    limit = JacksonMudholkarSPELimit().apply(_spectrum(background, 4), 0.95)
    result = BoundedResidualCalibrator().apply(raw, limit)
    assert result.numerical_validity
    assert torch.equal(result.mask, raw > float(limit.tau))
    assert torch.equal(result.score > 0.5, raw > float(limit.tau))


def test_calibration_is_strictly_monotone_for_10000_pairs():
    q = torch.linspace(0, 10, 20002, dtype=torch.float64)
    background = torch.randn(200, 12, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    limit = GammaSPELimit().apply(_spectrum(background, 3), 0.95)
    score = BoundedResidualCalibrator().apply(q, limit).score
    assert bool((score[1:] > score[:-1]).all())


def test_control_points():
    background = torch.randn(400, 10, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    limit = JacksonMudholkarSPELimit().apply(_spectrum(background, 3), 0.95)
    tau = float(limit.tau)
    values = torch.tensor([0.0, tau, tau * 1e12], dtype=torch.float64)
    score = BoundedResidualCalibrator().apply(values, limit).score
    assert float(score[0]) == 0.0
    assert float(score[1]) == pytest.approx(0.5, abs=1e-10)
    assert float(score[2]) == pytest.approx(1.0, abs=1e-10)


def test_synthetic_gaussian_background_exceedance_is_near_five_percent():
    generator = torch.Generator().manual_seed(7)
    dimension, rank = 12, 4
    scales = torch.tensor([3.0, 2.5, 2.0, 1.5, .8, .7, .6, .5, .4, .3, .2, .1], dtype=torch.float64)
    background = torch.randn(5000, dimension, generator=generator, dtype=torch.float64) * scales
    centered = background - background.mean(0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    spectrum = _spectrum(background, rank)
    limit = JacksonMudholkarSPELimit().apply(spectrum, 0.95)
    query = torch.randn(50000, dimension, generator=generator, dtype=torch.float64) * scales
    query = query - background.mean(0)
    basis = vh[:rank].t()
    residual = (query - (query @ basis) @ basis.t()).square().sum(1)
    rate = float((residual > float(limit.tau)).double().mean())
    assert 0.02 <= rate <= 0.08


def test_uniform_feature_scale_leaves_calibrated_score_unchanged():
    generator = torch.Generator().manual_seed(8)
    background = torch.randn(300, 20, generator=generator, dtype=torch.float64)
    raw = torch.rand(1000, generator=generator, dtype=torch.float64)
    scale = 3.7
    first = _spectrum(background, 5)
    second = _spectrum(background * scale, 5)
    limit1 = JacksonMudholkarSPELimit().apply(first, 0.95)
    limit2 = JacksonMudholkarSPELimit().apply(second, 0.95)
    score1 = BoundedResidualCalibrator().apply(raw, limit1).score
    score2 = BoundedResidualCalibrator().apply(raw * scale * scale, limit2).score
    assert float((score1 - score2).abs().max()) < 1e-8


@pytest.mark.parametrize("rank", [0, 1, 6, 7])
def test_rank_boundaries_are_explicit(rank: int):
    background = torch.randn(9, 12, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    spectrum = _spectrum(background, rank)
    assert spectrum.effective_spectrum_dimension == 8
    assert spectrum.pca_rank == rank
    assert spectrum.numerical_validity
    assert spectrum.discarded_eigenvalues.numel() == 8 - rank


def test_empty_discarded_spectrum_is_invalid_without_rank_change():
    background = torch.randn(9, 12, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    spectrum = _spectrum(background, 8)
    assert spectrum.pca_rank == 8
    assert not spectrum.numerical_validity
    assert spectrum.failure_reason == "discarded_spectrum_empty"
    result = BoundedResidualCalibrator().apply(torch.ones(3), JacksonMudholkarSPELimit().apply(spectrum))
    assert not result.numerical_validity
    assert result.score is None and result.mask is None

