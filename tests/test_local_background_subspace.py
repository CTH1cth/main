import torch
import numpy as np

from models.local_background_subspace import (
    NUM_PATCHES,
    local_reconstruction_from_neighbors,
    retrieve_background_neighbors,
)
from tools.gbsp_knn_lsr_common import right_ecdf
from tools.analyze_knn8_gbsp_complementarity import _ranking_counts


def _features(dim=16):
    generator = torch.Generator().manual_seed(20260807)
    return torch.randn(NUM_PATCHES, dim, generator=generator)


def test_leave_one_out_and_knn_order():
    background = torch.arange(48)
    result = retrieve_background_neighbors(_features(), background, max_k=32)
    assert result.self_match_violation_count == 0
    assert result.neighbor_indices.shape == (1369, 32)
    assert not torch.any(result.neighbor_indices[background] == background[:, None])
    assert torch.all(result.cosine_similarities[:, :-1] >= result.cosine_similarities[:, 1:])


def test_local_pca_shapes_orthogonality_and_residual():
    feature = _features()
    background = torch.arange(48)
    retrieval = retrieve_background_neighbors(feature, background, max_k=16)
    normalized = retrieval.features
    mean = normalized[background].mean(0)
    global_residual = (normalized - mean).square().sum(1)
    result = local_reconstruction_from_neighbors(
        retrieval,
        k=16,
        rank=4,
        global_mean=mean,
        global_residual=global_residual,
        query_batch_size=97,
    )
    assert result.local_residual.shape == (1369,)
    assert result.singular_values.shape == (1369, 16)
    assert result.neighbor_indices.shape == (1369, 16)
    assert result.max_orthonormal_error < 1e-4
    assert torch.isfinite(result.local_residual).all()
    assert torch.all(result.local_residual >= 0)
    assert torch.all(result.local_residual <= result.distance_to_local_mean.square() + 1e-5)


def test_query_count_and_manual_projection_agree():
    feature = _features(dim=12)
    background = torch.arange(64)
    retrieval = retrieve_background_neighbors(feature, background, max_k=8)
    mean = retrieval.features[background].mean(0)
    global_residual = (retrieval.features - mean).square().sum(1)
    result = local_reconstruction_from_neighbors(
        retrieval,
        k=8,
        rank=2,
        global_mean=mean,
        global_residual=global_residual,
        query_batch_size=1369,
    )
    query = retrieval.features[0]
    local = retrieval.features[result.neighbor_indices[0]]
    local_mean = local.mean(0)
    _, _, vh = torch.linalg.svd(local - local_mean, full_matrices=False)
    centered = query - local_mean
    manual = (centered - (centered @ vh[:2].T) @ vh[:2]).square().sum()
    assert torch.allclose(result.local_residual[0], manual, atol=1e-5, rtol=1e-5)
    assert result.knn8_anomaly.numel() == NUM_PATCHES


def test_right_continuous_ecdf_preserves_ties():
    value = np.array([2.0, 1.0, 2.0, 4.0])
    rank = right_ecdf(value)
    assert np.allclose(rank, [0.75, 0.25, 0.75, 1.0])


def test_ranking_disagreement_counts_strict_pairs_and_ties():
    knn = np.array([0.8, 0.2, 0.8, 0.9])
    gbsp = np.array([0.1, 0.2, 0.7, 0.6])
    labels = np.array([1, 0, 0, 1], dtype=np.uint8)
    counts = _ranking_counts(knn, gbsp, labels, np.ones(4, dtype=bool))
    assert counts["pairs"] == 4
    assert sum(counts[key] for key in ("both", "knn_only", "gbsp_only", "both_wrong")) == 4
    assert counts["knn_ties"] == 1
