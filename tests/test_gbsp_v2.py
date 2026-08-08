"""Unit contracts for GBSP-V2. These tests use no dataset GT or formal caches."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.gbsp_v2_background import (
    LocalGraph,
    boundary_two_ring_mask,
    build_local_graph,
    confidence_diversity_coreset,
    random_walk_with_restart,
    soft_boundary_seed,
)
from models.gbsp_v2_pca import (
    FixedPCAModel,
    confidence_rank_weights,
    euclidean_residual_score,
    fit_fixed_pca,
)
from models.gbsp_v2_residual import shrinkage_whitened_score


def _features(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(1369, 384, generator=generator), dim=1)


def _manual_graph() -> LocalGraph:
    # 0--1--2 is a high-affinity path; 2--3 crosses a strong edge.
    source = torch.tensor([0, 1, 1, 2, 2, 3])
    target = torch.tensor([1, 0, 2, 1, 3, 2])
    affinity = torch.tensor([1.0, 1.0, 1.0, 1.0, .01, .01])
    degree = torch.zeros(4); degree.index_add_(0, source, affinity)
    transition = affinity / degree.index_select(0, source)
    return LocalGraph(source, target, -torch.log(affinity), affinity, transition, degree, 1.0, 0.0, 4)


def test_graph_similarity_is_monotone_symmetric_and_stochastic() -> None:
    features = _features()
    rgb = torch.linspace(0, 1, 37).reshape(1, 1, 37).expand(3, 37, 37).contiguous()
    graph = build_local_graph(features, rgb)
    order = torch.argsort(graph.cost)
    assert graph.affinity[order[0]] >= graph.affinity[order[-1]]
    reverse = {(int(s), int(t)): float(a) for s, t, a in zip(graph.source, graph.target, graph.affinity)}
    assert all(np.isclose(value, reverse[(target, source)]) for (source, target), value in reverse.items())
    transition = graph.dense_transition()
    assert torch.allclose(transition.sum(1), torch.ones(1369, dtype=torch.float64), atol=1e-10)
    assert bool(torch.isfinite(transition).all()) and bool((transition >= 0).all())


def test_random_walk_crosses_smooth_path_but_not_strong_edge() -> None:
    graph = _manual_graph()
    result = random_walk_with_restart(graph, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert result.confidence[2] > 5 * result.confidence[3]
    # Smooth-path depth does not impose an additive linear cost penalty.
    assert result.confidence[2] > 0
    assert result.convergence_error < 1e-7


def test_soft_seed_downweights_anomalous_boundary_patch() -> None:
    base = torch.zeros(384); base[0] = 1
    features = base.repeat(1369, 1)
    anomaly = 37 + 10
    features[anomaly] = 0; features[anomaly, 1] = 1
    rgb = torch.zeros(3, 37, 37); rgb[:, 1, 10] = 1
    graph = build_local_graph(features, rgb)
    result = soft_boundary_seed(graph, features)
    normal = 37 + 20
    assert result.seed[anomaly] < result.seed[normal]
    assert result.inward_consistency[anomaly] < result.inward_consistency[normal]
    assert result.global_dino_density[anomaly] < result.global_dino_density[normal]


def test_confidence_diversity_retains_distinct_modes_deterministically() -> None:
    features = torch.zeros(1369, 384)
    features[:, 0] = 1
    confidence = torch.linspace(1.0, 0.0, 1369)
    # Twenty near-duplicates followed by five slightly lower-confidence modes.
    for index in range(20):
        features[index, 0] = 1; features[index, 1] = index * 1e-4
    for offset in range(5):
        features[20 + offset] = 0; features[20 + offset, 2 + offset] = 1
    features = F.normalize(features, dim=1)
    first = confidence_diversity_coreset(confidence, features, 13)
    second = confidence_diversity_coreset(confidence, features, 13)
    assert torch.equal(first.indices, second.indices)
    top_confidence = torch.argsort(confidence, descending=True, stable=True)[:13]
    selected_modes = int((first.indices >= 20).logical_and(first.indices < 25).sum())
    top_confidence_modes = int((top_confidence >= 20).logical_and(top_confidence < 25).sum())
    assert selected_modes > top_confidence_modes


def test_weighted_pca_resists_low_confidence_contamination() -> None:
    generator = torch.Generator().manual_seed(4)
    clean = torch.randn(100, 384, generator=generator) * .01
    clean[:, 0] += torch.linspace(-1, 1, 100)
    contamination = torch.zeros(5, 384); contamination[:, 10] = 5
    background = torch.cat((clean, contamination))
    confidence = torch.cat((torch.ones(100), torch.zeros(5)))
    weights = confidence_rank_weights(confidence)
    equal = fit_fixed_pca(background, rank=8)
    weighted = fit_fixed_pca(background, rank=8, weights=weights)
    assert torch.linalg.vector_norm(weighted.mean) < torch.linalg.vector_norm(equal.mean)
    manual_mean = (weights[:, None] * background).sum(0) / weights.sum()
    assert torch.allclose(weighted.mean, manual_mean, atol=1e-6)


def test_swor_prioritizes_low_variance_residual_direction_and_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(7)
    count = 256
    background = torch.zeros(count, 384)
    background[:, 8] = 2.0 * torch.randn(count, generator=generator)
    background[:, 9] = .1 * torch.randn(count, generator=generator)
    basis = torch.eye(384)[:, :8]
    model = FixedPCAModel(torch.zeros(384), basis, torch.ones(8), torch.ones(count), 8, 0.0, 0.0, float(count), False)
    query = torch.zeros(1369, 384); query[0, 8] = 1; query[1, 9] = 1
    euclidean = euclidean_residual_score(query, model)
    assert torch.isclose(euclidean[0], euclidean[1])
    first = shrinkage_whitened_score(query, background, model)
    second = shrinkage_whitened_score(query, background, model)
    assert first.score[1] > first.score[0]
    assert torch.equal(first.score, second.score)
    assert 0 <= first.shrinkage <= 1 and first.covariance_condition_number > 0


def test_all_1369_patches_are_scored_without_candidate_overwrite() -> None:
    features = _features(9)
    candidates = torch.where(boundary_two_ring_mask())[0]
    model = fit_fixed_pca(features.index_select(0, candidates), rank=8)
    score = euclidean_residual_score(features, model)
    assert score.shape == (1369,)
    assert bool(torch.isfinite(score).all())
    assert bool((score.index_select(0, candidates) > 0).any())
