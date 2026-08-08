"""Query-conditioned local PCA reconstruction with separate retrieval geometry."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .retrieval import NUM_PATCHES, RetrievalResult, flatten_feature


@dataclass(frozen=True)
class LocalPCAResult:
    k: int
    rank: int
    feature_geometry: str
    residual: torch.Tensor
    local_mean_norm: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: torch.Tensor
    distance_to_local_mean: torch.Tensor
    svd_orthonormal_error_before_qr_max: float
    orthonormal_error_max: float


def local_pca_reconstruct(
    raw_features: torch.Tensor,
    retrieval: RetrievalResult,
    *,
    k: int = 16,
    rank: int = 4,
    feature_geometry: str = "l2",
    query_batch_size: int = 128,
) -> LocalPCAResult:
    feature_geometry = str(feature_geometry).lower()
    if feature_geometry not in {"l2", "raw"}:
        raise ValueError("feature_geometry must be 'l2' or 'raw'")
    feature = (
        retrieval.normalized_features
        if feature_geometry == "l2"
        else flatten_feature(raw_features, normalize=False)
    )
    k, rank, query_batch_size = int(k), int(rank), int(query_batch_size)
    if k < 3 or k > retrieval.neighbor_indices.shape[1]:
        raise ValueError(f"invalid LSR K={k}")
    if rank < 1 or rank >= k - 1:
        raise ValueError(f"rank must be in [1,K-2], got K={k}, rank={rank}")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be positive")

    outputs = {name: [] for name in (
        "residual", "local_mean_norm", "singular_values", "effective_rank",
        "distance_to_local_mean",
    )}
    raw_max_error = 0.0
    max_error = 0.0
    neighbors = retrieval.neighbor_indices[:, :k]
    for start in range(0, NUM_PATCHES, query_batch_size):
        stop = min(start + query_batch_size, NUM_PATCHES)
        query = feature[start:stop]
        index = neighbors[start:stop]
        local = feature.index_select(0, index.reshape(-1)).reshape(stop - start, k, feature.shape[1])
        mean = local.mean(dim=1)
        centered = local - mean.unsqueeze(1)
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        raw_basis_rows = vh[:, :rank, :]
        raw_gram = raw_basis_rows @ raw_basis_rows.transpose(1, 2)
        identity = torch.eye(rank, dtype=feature.dtype, device=feature.device).expand_as(raw_gram)
        raw_max_error = max(raw_max_error, float((raw_gram - identity).abs().max()))
        basis_columns, _ = torch.linalg.qr(raw_basis_rows.transpose(1, 2), mode="reduced")
        basis_rows = basis_columns.transpose(1, 2).contiguous()
        gram = basis_rows @ basis_rows.transpose(1, 2)
        max_error = max(max_error, float((gram - identity).abs().max()))
        query_centered = query - mean
        coefficients = torch.einsum("brd,bd->br", basis_rows, query_centered)
        projection = torch.einsum("br,brd->bd", coefficients, basis_rows)
        residual = (query_centered - projection).square().sum(dim=1)
        energy = singular.double().square()
        probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-24)
        effective_rank = (-(probability * probability.clamp_min(1e-24).log()).sum(dim=1)).exp()
        outputs["residual"].append(residual)
        outputs["local_mean_norm"].append(mean.norm(dim=1))
        outputs["singular_values"].append(singular)
        outputs["effective_rank"].append(effective_rank.to(feature.dtype))
        outputs["distance_to_local_mean"].append(query_centered.norm(dim=1))
    result = {name: torch.cat(value, dim=0).contiguous() for name, value in outputs.items()}
    if max_error >= 1e-4:
        raise RuntimeError(f"local PCA basis is not orthonormal: {max_error}")
    if any(value.shape[0] != NUM_PATCHES or not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("local PCA returned incomplete/non-finite diagnostics")
    return LocalPCAResult(
        k=k,
        rank=rank,
        feature_geometry=feature_geometry,
        residual=result["residual"].clamp_min(0).float(),
        local_mean_norm=result["local_mean_norm"].float(),
        singular_values=result["singular_values"].float(),
        effective_rank=result["effective_rank"].float(),
        distance_to_local_mean=result["distance_to_local_mean"].float(),
        svd_orthonormal_error_before_qr_max=raw_max_error,
        orthonormal_error_max=max_error,
    )
