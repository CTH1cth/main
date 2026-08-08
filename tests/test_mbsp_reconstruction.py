from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from models.mbsp_reconstruction import MultiBackgroundSubspaceProjector


def _projector(**overrides):
    settings = {
        "num_subspaces": 1,
        "min_cluster_size": 2,
        "pca_energy": 0.90,
        "pca_max_rank": 1,
        "pca_min_rank": 1,
        "seed": 0,
        "kmeans_n_init": 3,
        "kmeans_max_iter": 30,
    }
    settings.update(overrides)
    return MultiBackgroundSubspaceProjector(**settings)


def test_two_dimensional_line_projection():
    background = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    model = _projector().fit(background)
    result = model.score(torch.tensor([[4.0, 0.0], [2.0, 1.0]]))
    assert float(result.absolute_residual[0]) < 1e-7
    assert float(result.absolute_residual[1]) > 0.05


def test_every_nonzero_basis_is_orthonormal():
    generator = torch.Generator().manual_seed(7)
    background = torch.randn(40, 12, generator=generator)
    model = _projector(pca_max_rank=6).fit(background)
    for basis in model.subspace_bases:
        if basis.shape[1]:
            identity = torch.eye(basis.shape[1])
            assert float((basis.t() @ basis - identity).abs().max()) < 1e-4


def test_relative_residual_is_clipped_to_unit_interval():
    generator = torch.Generator().manual_seed(9)
    background = torch.randn(64, 16, generator=generator)
    query = torch.randn(101, 16, generator=generator)
    result = _projector(num_subspaces=4, min_cluster_size=8, pca_max_rank=4).fit(
        background
    ).score(query)
    assert float(result.relative_residual.min()) >= 0.0
    assert float(result.relative_residual.max()) <= 1.0
    assert tuple(result.per_subspace_relative.shape) == (101, 4)


def test_single_subspace_matches_manual_pca_projection():
    generator = torch.Generator().manual_seed(11)
    background = torch.randn(25, 8, generator=generator)
    query = torch.randn(17, 8, generator=generator)
    model = _projector(pca_energy=1.0, pca_max_rank=3).fit(background)
    result = model.score(query)

    normalized_background = F.normalize(background, p=2, dim=1)
    normalized_query = F.normalize(query, p=2, dim=1)
    mean = normalized_background.mean(dim=0)
    _, _, right = torch.linalg.svd(normalized_background - mean, full_matrices=False)
    basis = right[:3].t()
    centered = normalized_query - mean
    manual = (centered - (centered @ basis) @ basis.t()).square().sum(dim=1)
    assert torch.allclose(result.absolute_residual, manual, atol=1e-6, rtol=1e-5)


def test_rank_zero_is_exact_background_mean_distance_without_svd(monkeypatch):
    generator = torch.Generator().manual_seed(23)
    background = torch.randn(25, 8, generator=generator)
    query = torch.randn(17, 8, generator=generator)

    def forbidden_svd(*args, **kwargs):
        raise AssertionError("rank-0 must not call SVD")

    monkeypatch.setattr(torch.linalg, "svd", forbidden_svd)
    model = _projector(pca_max_rank=0, pca_min_rank=0).fit(background)
    result = model.score(query)

    normalized_background = F.normalize(background, p=2, dim=1)
    normalized_query = F.normalize(query, p=2, dim=1)
    mean = normalized_background.mean(dim=0)
    expected = (normalized_query - mean).square().sum(dim=1)
    assert model.selected_ranks.tolist() == [0]
    assert model.subspace_bases[0].shape == (8, 0)
    assert torch.allclose(model.subspace_means[0], mean, atol=1e-7, rtol=0.0)
    assert torch.allclose(result.absolute_residual, expected, atol=1e-7, rtol=0.0)


def test_pca_residual_is_pointwise_no_larger_than_rank_zero_residual():
    generator = torch.Generator().manual_seed(29)
    background = torch.randn(40, 12, generator=generator)
    query = torch.randn(31, 12, generator=generator)
    rank_zero = _projector(pca_max_rank=0, pca_min_rank=0).fit(background)
    rank_r = _projector(pca_energy=0.90, pca_max_rank=8, pca_min_rank=1).fit(background)

    q0 = rank_zero.score(query).absolute_residual
    qr = rank_r.score(query).absolute_residual

    assert torch.equal(rank_zero.subspace_means, rank_r.subspace_means)
    assert bool((qr <= q0 + 1e-6).all())


def test_zero_variance_cluster_is_finite_center_model():
    background = torch.tensor([[1.0, 2.0, 3.0]]).repeat(20, 1)
    query = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    model = _projector(pca_max_rank=4).fit(background)
    result = model.score(query)
    assert model.selected_ranks.tolist() == [0]
    assert model.zero_variance_clusters.tolist() == [True]
    assert bool(torch.isfinite(result.relative_residual).all())
    assert bool(torch.isfinite(result.absolute_residual).all())


def test_same_seed_is_exactly_deterministic():
    generator = torch.Generator().manual_seed(13)
    background = torch.randn(80, 10, generator=generator)
    query = torch.randn(23, 10, generator=generator)
    settings = dict(num_subspaces=4, min_cluster_size=8, pca_max_rank=4, seed=19)
    first = _projector(**settings).fit(background)
    second = _projector(**settings).fit(background)
    first_score = first.score(query)
    second_score = second.score(query)
    assert torch.equal(first.cluster_assignments, second.cluster_assignments)
    assert torch.equal(first.selected_ranks, second.selected_ranks)
    assert torch.equal(first_score.relative_residual, second_score.relative_residual)


def test_small_dictionary_uses_recorded_center_fallback():
    background = torch.randn(7, 5, generator=torch.Generator().manual_seed(17))
    model = _projector(num_subspaces=4, min_cluster_size=16).fit(background)
    assert model.num_effective_subspaces == 1
    assert model.selected_ranks.tolist() == [0]
    assert model.fallback_flags["fallback_single_center"]


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        _projector().fit(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_core_has_no_forbidden_operator_calls():
    source = inspect.getsource(MultiBackgroundSubspaceProjector).lower()
    forbidden = (
        "soft" + "max",
        "atten" + "tion",
        "q" + "kv",
        "scaled_dot_product_" + "atten" + "tion",
    )
    assert not any(token in source for token in forbidden)
