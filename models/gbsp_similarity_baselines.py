"""Similarity baselines used by the GBSP narrative audit.

All returned scores follow the same convention: larger means more foreground-like.
Background-candidate queries are evaluated leave-one-out, including prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class SimilarityBaselineResult:
    scores: Dict[str, torch.Tensor]
    nn_background_similarity: torch.Tensor
    nn_background_index: torch.Tensor
    self_match_violation_count: int
    num_background: int


def _as_flat_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1).T
    if features.ndim != 2:
        raise ValueError(f"features must be [C,H,W] or [N,C], got {tuple(features.shape)}")
    if features.shape[0] != 37 * 37:
        raise ValueError(f"expected 1369 patches, got {features.shape[0]}")
    features = F.normalize(features.float(), dim=1, eps=1e-12)
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def minmax_per_image(score: torch.Tensor) -> torch.Tensor:
    score = score.float()
    lo, hi = score.min(), score.max()
    return (score - lo) / (hi - lo).clamp_min(1e-12)


def compute_similarity_baselines(
    features: torch.Tensor,
    background_indices: torch.Tensor,
    *,
    k: int = 8,
) -> SimilarityBaselineResult:
    """Compute Mean-L2, Proto-Cos, NN-Cos and KNN-Cos on one image."""
    x = _as_flat_features(features)
    bg_idx = torch.as_tensor(background_indices, dtype=torch.long, device=x.device).flatten()
    if bg_idx.numel() != torch.unique(bg_idx).numel():
        raise ValueError("background_indices contain duplicates")
    if bg_idx.numel() < k + 1:
        raise ValueError(f"leave-one-out KNN-{k} needs at least {k + 1} background patches")
    if bg_idx.min() < 0 or bg_idx.max() >= x.shape[0]:
        raise ValueError("background index out of range")

    bg = x[bg_idx]
    # Per-query prototype. A BC query cannot contribute to its own prototype.
    proto = bg.mean(dim=0, keepdim=True).expand(x.shape[0], -1).clone()
    proto[bg_idx] = (bg.sum(dim=0, keepdim=True) - bg) / float(bg.shape[0] - 1)
    mean_l2 = (x - proto).square().sum(dim=1)
    proto_cos = 1.0 - (x * F.normalize(proto, dim=1, eps=1e-12)).sum(dim=1)

    similarity = x @ bg.T
    # Map each candidate's global query row to its column, then mask diagonal.
    candidate_columns = torch.arange(bg_idx.numel(), device=x.device)
    similarity[bg_idx, candidate_columns] = -torch.inf
    top = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)
    nn_similarity = top.values[:, 0]
    nn_index = bg_idx[top.indices[:, 0]]
    violations = int((nn_index[bg_idx] == bg_idx).sum().item())
    if violations:
        raise RuntimeError(f"self-match exclusion failed for {violations} candidates")

    scores = {
        "mean_l2": mean_l2,
        "proto_cos": proto_cos,
        "nn_cos": 1.0 - nn_similarity,
        "knn8_cos": 1.0 - top.values.mean(dim=1),
    }
    for name, score in scores.items():
        if score.shape != (37 * 37,) or not torch.isfinite(score).all():
            raise RuntimeError(f"invalid {name} score")
    return SimilarityBaselineResult(
        scores=scores,
        nn_background_similarity=nn_similarity,
        nn_background_index=nn_index,
        self_match_violation_count=violations,
        num_background=int(bg_idx.numel()),
    )
