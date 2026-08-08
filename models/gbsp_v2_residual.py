"""Shrinkage-whitened orthogonal residual (SWOR) for GBSP-V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from models.gbsp_v2_background import FEATURE_DIM, NUM_PATCHES
from models.gbsp_v2_pca import FixedPCAModel, residual_vectors


@dataclass(frozen=True)
class SWORResult:
    score: torch.Tensor
    shrinkage: float
    covariance_condition_number: float
    precision_condition_number: float
    minimum: float
    maximum: float
    mean: float
    valid: bool


def shrinkage_whitened_score(
    query_features: torch.Tensor,
    background_features: torch.Tensor,
    model: FixedPCAModel,
    *,
    negative_tolerance: float = 1e-8,
) -> SWORResult:
    background = background_features.detach().cpu().float().contiguous()
    if background.ndim != 2 or background.shape[1] != FEATURE_DIM:
        raise ValueError(f"background_features must be Tensor[N,{FEATURE_DIM}]")
    if background.shape[0] != model.weights.numel():
        raise ValueError("SWOR background/weight size mismatch")
    centered = background - model.mean
    background_residual = centered - (centered @ model.basis) @ model.basis.t()
    weighted_residual = torch.sqrt(model.weights)[:, None] * background_residual
    matrix = weighted_residual.double().numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("weighted background residual contains NaN/Inf")

    estimator = LedoitWolf(assume_centered=True).fit(matrix)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    precision = np.asarray(estimator.precision_, dtype=np.float64)
    shrinkage = float(estimator.shrinkage_)
    covariance_condition = float(np.linalg.cond(covariance))
    precision_condition = float(np.linalg.cond(precision))
    if not (np.isfinite(covariance).all() and np.isfinite(precision).all()):
        raise RuntimeError("SWOR covariance/precision contains NaN/Inf")
    if not 0.0 <= shrinkage <= 1.0:
        raise RuntimeError(f"invalid Ledoit-Wolf shrinkage: {shrinkage}")

    query_residual = residual_vectors(query_features, model).double().numpy()
    raw_score = np.einsum("ni,ij,nj->n", query_residual, precision, query_residual, optimize=True)
    score = torch.from_numpy(raw_score / (FEATURE_DIM - model.rank)).double()
    if score.numel() != NUM_PATCHES or not bool(torch.isfinite(score).all()):
        raise RuntimeError("SWOR score is incomplete or non-finite")
    minimum = float(score.min())
    if minimum < -float(negative_tolerance):
        raise RuntimeError(f"SWOR score has a genuinely negative value: {minimum}")
    score = score.clamp_min(0.0).float().contiguous()
    return SWORResult(
        score,
        shrinkage,
        covariance_condition,
        precision_condition,
        float(score.min()),
        float(score.max()),
        float(score.mean()),
        True,
    )
