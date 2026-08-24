"""CVBR-guided percentile weighting for the frozen rank-8 GBSP estimator.

This module deliberately contains no GT, graph, candidate-selection, resize or
threshold logic.  It only turns an already validated CVBR risk into candidate
weights and fits a weighted affine PCA subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.gbsp_core_variants import PCAFit, PCARankSelector


@dataclass(frozen=True)
class CVBRWeightResult:
    weights: torch.Tensor
    risk_percentile: torch.Tensor
    suppressed_mask: torch.Tensor
    num_valid: int
    num_suppressed: int
    effective_sample_size: float


@dataclass(frozen=True)
class CVBRWeightedPCAFit:
    fit: PCAFit
    weights: torch.Tensor
    effective_sample_size: float
    num_positive_weight: int


def _vector(value: torch.Tensor, name: str, *, boolean: bool = False) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or not value.numel():
        raise ValueError(f"{name} must be a non-empty Tensor[N]")
    value = value.detach().cpu()
    if boolean:
        return value.bool().contiguous()
    value = value.float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def effective_sample_size(weights: torch.Tensor, eps: float = 1e-12) -> float:
    value = _vector(weights, "weights").double()
    if bool((value < 0).any()) or float(value.sum()) <= eps:
        raise ValueError("weights must be nonnegative with positive total mass")
    return float(value.sum().square() / value.square().sum().clamp_min(eps))


def build_cvbr_soft_weights(
    cvbr_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    top_frac: float,
    low_weight: float,
) -> CVBRWeightResult:
    """Downweight an exact, stable top-risk fraction among scored candidates.

    Unscored candidates always retain unit weight.  ``risk_percentile`` is NaN
    for those candidates and lies in ``(0, 1]`` for scored candidates, with a
    larger value denoting a more suspicious candidate.  Stable index order is
    the deterministic tie break used for the exact top-count intervention.
    """
    scores = _vector(cvbr_scores, "cvbr_scores")
    valid = _vector(valid_mask, "valid_mask", boolean=True)
    if scores.shape != valid.shape:
        raise ValueError("cvbr_scores/valid_mask shape mismatch")
    top_frac, low_weight = float(top_frac), float(low_weight)
    if not 0.0 <= top_frac <= 1.0:
        raise ValueError("top_frac must be in [0,1]")
    if not 0.0 <= low_weight <= 1.0:
        raise ValueError("low_weight must be in [0,1]")
    num_valid = int(valid.sum())
    if top_frac > 0.0 and num_valid == 0:
        raise ValueError("positive top_frac requires at least one valid CVBR score")

    weights = torch.ones_like(scores)
    percentiles = torch.full_like(scores, float("nan"))
    suppressed = torch.zeros_like(valid)
    if num_valid:
        valid_index = torch.where(valid)[0]
        order = torch.argsort(scores.index_select(0, valid_index), descending=True, stable=True)
        ordered_index = valid_index.index_select(0, order)
        # Highest risk receives percentile 1.0; lowest receives 1/N.
        percentiles[ordered_index] = torch.arange(
            num_valid, 0, -1, dtype=scores.dtype
        ) / float(num_valid)
        num_suppressed = int(torch.ceil(torch.tensor(top_frac * num_valid)).item()) if top_frac else 0
        if num_suppressed:
            selected = ordered_index[:num_suppressed]
            suppressed[selected] = True
            weights[selected] = low_weight
    else:
        num_suppressed = 0

    return CVBRWeightResult(
        weights=weights.contiguous(),
        risk_percentile=percentiles.contiguous(),
        suppressed_mask=suppressed.contiguous(),
        num_valid=num_valid,
        num_suppressed=num_suppressed,
        effective_sample_size=effective_sample_size(weights),
    )


def fit_weighted_affine_subspace(
    features: torch.Tensor,
    weights: torch.Tensor,
    rank: int = 8,
    eps: float = 1e-8,
) -> CVBRWeightedPCAFit:
    """Fit a weighted affine PCA using right singular vectors.

    Zero weights are kept as exact zeros so the registered R5 variant is a
    genuine hard-trim diagnostic.  They do not alter the candidate set itself.
    """
    if not torch.is_tensor(features) or features.ndim != 2 or not features.numel():
        raise ValueError("features must be a non-empty Tensor[M,D]")
    x = features.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(x).all()):
        raise ValueError("features contain NaN/Inf")
    w = _vector(weights, "weights")
    if w.numel() != x.shape[0]:
        raise ValueError("weights/features size mismatch")
    if bool((w < 0).any()) or float(w.sum()) <= float(eps):
        raise ValueError("weights must be nonnegative with positive total mass")
    positive = int((w > 0).sum())
    rank = int(rank)
    effective = min(positive - 1, int(x.shape[1]))
    if rank < 1 or rank >= effective:
        raise ValueError(f"rank {rank} is invalid for effective dimension {effective}")

    # Preserve the ordinary arithmetic-mean reduction for the all-ones audit.
    mean = x.mean(0) if bool(torch.equal(w, torch.ones_like(w))) else (
        (w[:, None] * x).sum(0) / w.sum().clamp_min(float(eps))
    )
    centered = x - mean
    weighted_centered = torch.sqrt(w.clamp_min(0.0))[:, None] * centered
    _, singular, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
    basis = vh[:rank].t().contiguous()
    orthonormal_error = float(
        (basis.t() @ basis - torch.eye(rank, dtype=basis.dtype)).abs().max()
    )
    if orthonormal_error >= 1e-4 or not bool(torch.isfinite(basis).all()):
        raise RuntimeError(f"weighted PCA basis invalid: {orthonormal_error}")
    energy = singular[:effective].square()
    retained = float(energy[:rank].sum() / energy.sum()) if float(energy.sum()) > 0 else 0.0
    fit = PCAFit(
        mean=mean.contiguous(),
        basis=basis,
        singular_values=singular.contiguous(),
        selected_rank=rank,
        effective_spectrum_dimension=effective,
        retained_variance_ratio=retained,
        discarded_variance_ratio=1.0 - retained,
        orthonormal_error=orthonormal_error,
    )
    return CVBRWeightedPCAFit(
        fit=fit,
        weights=w,
        effective_sample_size=effective_sample_size(w),
        num_positive_weight=positive,
    )


def projector_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Normalized Frobenius distance between equal-rank projectors."""
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("basis matrices must have the same [D,r] shape")
    rank = int(left.shape[1])
    if rank < 1:
        raise ValueError("projector distance requires positive rank")
    delta = left.float() @ left.float().t() - right.float() @ right.float().t()
    return float(torch.linalg.vector_norm(delta) / (2.0 * rank) ** 0.5)

