from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.gbsp_core_variants import (
    BOUNDARY_COUNT,
    NUM_PATCHES,
    BackgroundCandidateSelector,
    ConnectivityWeightedPCA,
    PCARankSelector,
    boundary_two_ring_mask,
    decompose_pca,
    fit_pca_from_svd,
    pca_fit_from_decomposition,
    principal_angles_degrees,
    score_all_patches,
)


def test_rank_selector_fixed_current_and_energy_rules():
    singular = torch.tensor([5., 4., 3., 2., 1., .5, .25, .1])
    selector = PCARankSelector(current_energy=.90, current_max_rank=4, current_min_rank=1)
    assert selector.select("fixed", singular, 9, 12, 0) == 0
    assert selector.select("fixed", singular, 9, 12, 4) == 4
    assert selector.select("current", singular, 9, 12) <= 4
    assert selector.select("ev95", singular, 9, 12) >= selector.select("ev90", singular, 9, 12)
    with pytest.raises(ValueError, match="invalid"):
        selector.select("fixed", singular, 9, 12, 8)


def test_boundary_and_matched_candidate_selection_are_exact():
    boundary = boundary_two_ring_mask()
    assert int(boundary.sum()) == BOUNDARY_COUNT
    interior = torch.where(~boundary)[0][:131]
    full = torch.cat((torch.where(boundary)[0], interior)).sort().values
    connectivity = torch.full((NUM_PATCHES,), .2)
    connectivity[boundary] = 1.0
    selector = BackgroundCandidateSelector()
    matched = selector.select("fullbc_matched280", full, connectivity)
    assert torch.equal(matched.sort().values, torch.where(boundary)[0])
    assert torch.equal(selector.select("boundary280", full, connectivity), torch.where(boundary)[0])
    assert torch.equal(selector.select("fullbc", full, connectivity), full)


def test_path_weights_follow_formula_and_mean_one():
    connectivity = torch.tensor([1.0, .8, .4, .1])
    weights, path, median, effective, warning = ConnectivityWeightedPCA().weights(connectivity)
    expected = 1 / (1 + path.double() / (median + 1e-8)); expected *= 4 / expected.sum()
    assert torch.allclose(weights.double(), expected, atol=1e-7)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-7)
    assert 1 <= effective <= 4
    assert isinstance(warning, bool)


def test_equal_connectivity_weighted_pca_matches_unweighted_pca():
    generator = torch.Generator().manual_seed(3)
    background = F.normalize(torch.randn(40, 12, generator=generator), dim=1)
    query = F.normalize(torch.randn(NUM_PATCHES, 12, generator=generator), dim=1)
    selector = PCARankSelector()
    equal = fit_pca_from_svd(background, "fixed", selector, 4)
    weighted = ConnectivityWeightedPCA().fit(background, torch.ones(40), "fixed", selector, 4)
    assert torch.allclose(equal.mean, weighted.fit.mean, atol=1e-7)
    assert torch.allclose(score_all_patches(query, equal), score_all_patches(query, weighted.fit), atol=1e-6)


def test_rank_sweep_reuses_one_decomposition_exactly():
    background = torch.randn(40, 12, generator=torch.Generator().manual_seed(31))
    selector = PCARankSelector()
    decomposition = decompose_pca(background)
    for rank in (0, 4, 8):
        direct = fit_pca_from_svd(background, "fixed", selector, rank)
        reused = pca_fit_from_decomposition(decomposition, "fixed", selector, rank)
        assert torch.equal(direct.mean, reused.mean)
        assert torch.equal(direct.singular_values, reused.singular_values)
        assert torch.equal(direct.basis, reused.basis)


def test_weighted_mean_svd_and_query_formula_are_correct():
    generator = torch.Generator().manual_seed(4)
    background = F.normalize(torch.randn(50, 16, generator=generator), dim=1)
    query = F.normalize(torch.randn(NUM_PATCHES, 16, generator=generator), dim=1)
    connectivity = torch.linspace(.1, 1.0, 50)
    result = ConnectivityWeightedPCA().fit(background, connectivity, "fixed", PCARankSelector(), 5)
    weights = result.normalized_weights
    expected_mean = (weights[:, None] * background).sum(0) / weights.sum()
    assert torch.allclose(result.fit.mean, expected_mean, atol=1e-7)
    assert result.fit.orthonormal_error < 1e-4
    residual = score_all_patches(query, result.fit)
    centered = query - result.fit.mean
    expected = (centered - (centered @ result.fit.basis) @ result.fit.basis.t()).square().sum(1)
    assert torch.allclose(residual, expected, atol=1e-6)


def test_every_patch_including_background_is_scored_without_overwrite():
    generator = torch.Generator().manual_seed(5)
    query = F.normalize(torch.randn(NUM_PATCHES, 10, generator=generator), dim=1)
    background_indices = torch.arange(411)
    fit = fit_pca_from_svd(query[background_indices], "fixed", PCARankSelector(), 4)
    residual = score_all_patches(query, fit)
    assert residual.shape == (NUM_PATCHES,)
    assert bool(torch.isfinite(residual).all())
    assert bool((residual[background_indices] > 0).any())


def test_principal_angles_are_zero_for_identical_subspaces():
    basis = torch.linalg.qr(torch.randn(20, 5, generator=torch.Generator().manual_seed(6))).Q[:, :5]
    angles = principal_angles_degrees(basis, basis)
    assert float(angles.abs().max()) < .05
