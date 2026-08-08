"""Fixed-rank equal and confidence-weighted PCA for GBSP-V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.gbsp_v2_background import FEATURE_DIM, NUM_PATCHES, average_rank


@dataclass(frozen=True)
class FixedPCAModel:
    mean: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    weights: torch.Tensor
    rank: int
    retained_variance_ratio: float
    orthonormal_error: float
    effective_sample_size: float
    weight_concentration_warning: bool


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a non-empty Tensor[N,D]")
    result = value.detach().cpu().float().contiguous()
    if result.shape[1] != FEATURE_DIM or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite Tensor[N,{FEATURE_DIM}]")
    return result


def confidence_rank_weights(confidence: torch.Tensor) -> torch.Tensor:
    value = confidence.detach().cpu().float().reshape(-1)
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError("candidate confidence must be non-empty and finite")
    rank = (average_rank(value) - .5) / value.numel()
    weights = 2.0 * rank
    if abs(float(weights.mean()) - 1.0) > 1e-6 or bool((weights <= 0).any()):
        raise RuntimeError("confidence-rank weights failed unit-mean normalization")
    return weights.float().contiguous()


def fit_fixed_pca(
    background_features: torch.Tensor,
    *,
    rank: int = 8,
    weights: torch.Tensor | None = None,
) -> FixedPCAModel:
    background = _matrix(background_features, "background_features")
    rank = int(rank)
    if background.shape[0] <= rank or not 0 < rank < min(background.shape):
        raise ValueError(f"rank={rank} invalid for background shape {tuple(background.shape)}")
    if weights is None:
        candidate_weights = torch.ones(background.shape[0], dtype=torch.float32)
    else:
        candidate_weights = weights.detach().cpu().float().reshape(-1)
        if candidate_weights.numel() != background.shape[0]:
            raise ValueError("PCA weights/background size mismatch")
        if not bool(torch.isfinite(candidate_weights).all()) or bool((candidate_weights <= 0).any()):
            raise ValueError("PCA weights must be finite and positive")
        candidate_weights = candidate_weights * (candidate_weights.numel() / candidate_weights.sum())
    mean = (candidate_weights[:, None] * background).sum(0) / candidate_weights.sum()
    weighted_centered = torch.sqrt(candidate_weights)[:, None] * (background - mean)
    _, singular, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
    basis = vh[:rank].t().contiguous()
    orthonormal_error = float((basis.t() @ basis - torch.eye(rank)).abs().max())
    if orthonormal_error >= 1e-4 or not bool(torch.isfinite(basis).all()):
        raise RuntimeError(f"invalid PCA basis: orthonormal_error={orthonormal_error}")
    effective_dimension = min(background.shape[0] - 1, background.shape[1])
    energy = singular[:effective_dimension].double().square()
    retained = float(energy[:rank].sum() / energy.sum()) if float(energy.sum()) > 0 else 0.0
    effective_sample = float(candidate_weights.sum().square() / candidate_weights.square().sum())
    return FixedPCAModel(
        mean.float().contiguous(),
        basis.float().contiguous(),
        singular.float().contiguous(),
        candidate_weights.float().contiguous(),
        rank,
        retained,
        orthonormal_error,
        effective_sample,
        bool(effective_sample < .5 * background.shape[0]),
    )


def residual_vectors(query_features: torch.Tensor, model: FixedPCAModel) -> torch.Tensor:
    query = _matrix(query_features, "query_features")
    if query.shape[0] != NUM_PATCHES:
        raise ValueError(f"all {NUM_PATCHES} patches must be queried")
    centered = query - model.mean
    residual = centered - (centered @ model.basis) @ model.basis.t()
    if tuple(residual.shape) != (NUM_PATCHES, FEATURE_DIM) or not bool(torch.isfinite(residual).all()):
        raise RuntimeError("PCA residual vectors are invalid")
    return residual.float().contiguous()


def euclidean_residual_score(query_features: torch.Tensor, model: FixedPCAModel) -> torch.Tensor:
    score = residual_vectors(query_features, model).square().sum(1)
    if score.numel() != NUM_PATCHES or not bool(torch.isfinite(score).all()):
        raise RuntimeError("Euclidean residual score is incomplete")
    return score.float().contiguous()


def principal_angles_degrees(left: FixedPCAModel, right: FixedPCAModel) -> torch.Tensor:
    cosine = torch.linalg.svdvals(left.basis.t() @ right.basis).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).float().contiguous()
