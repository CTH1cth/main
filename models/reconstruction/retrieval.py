"""Frozen Full-BC cosine retrieval with strict leave-one-out."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


GRID = 37
NUM_PATCHES = GRID * GRID


@dataclass(frozen=True)
class RetrievalResult:
    normalized_features: torch.Tensor
    background_indices: torch.Tensor
    neighbor_indices: torch.Tensor
    cosine_similarities: torch.Tensor
    self_match_violation_count: int


def flatten_feature(value: torch.Tensor, *, normalize: bool = False) -> torch.Tensor:
    """Return a finite ``[1369,D]`` feature matrix without changing order."""
    if not torch.is_tensor(value):
        raise TypeError("feature must be a torch.Tensor")
    if value.ndim == 4 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim == 3:
        value = value.reshape(value.shape[0], -1).T
    if value.ndim != 2 or value.shape[0] != NUM_PATCHES:
        raise ValueError(f"feature must be [D,37,37] or [1369,D], got {tuple(value.shape)}")
    value = value.float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("feature contains NaN/Inf")
    norms = torch.linalg.vector_norm(value, dim=1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("feature contains a zero-norm patch")
    return F.normalize(value, dim=1, eps=1e-12).contiguous() if normalize else value


def retrieve_fullbc_neighbors(
    raw_features: torch.Tensor,
    background_indices: torch.Tensor,
    *,
    max_k: int = 32,
) -> RetrievalResult:
    """Retrieve Top-K atoms using L2-normalized cosine geometry.

    Retrieval geometry is deliberately separated from reconstruction geometry.
    The returned global patch indices can be reused with either raw or L2
    reconstruction features.
    """
    features = flatten_feature(raw_features, normalize=True)
    background = torch.as_tensor(
        background_indices, dtype=torch.long, device=features.device
    ).reshape(-1)
    if background.numel() == 0 or background.numel() != torch.unique(background).numel():
        raise ValueError("background_indices must be non-empty and unique")
    if int(background.min()) < 0 or int(background.max()) >= NUM_PATCHES:
        raise ValueError("background index out of range")
    max_k = int(max_k)
    if max_k < 1 or background.numel() < max_k + 1:
        raise ValueError(f"leave-one-out KNN-{max_k} needs at least {max_k + 1} atoms")

    memory = features.index_select(0, background)
    similarity = features @ memory.T
    # Each candidate appears exactly once, so its row/column mapping is exact.
    columns = torch.arange(background.numel(), device=features.device)
    similarity[background, columns] = -torch.inf
    top = torch.topk(similarity, k=max_k, dim=1, largest=True, sorted=True)
    indices = background.index_select(0, top.indices.reshape(-1)).reshape(NUM_PATCHES, max_k)
    query = torch.arange(NUM_PATCHES, device=features.device).unsqueeze(1)
    violations = int((indices == query).sum().item())
    if violations:
        raise RuntimeError(f"self-match exclusion failed: {violations}")
    if not bool(torch.isfinite(top.values).all()):
        raise RuntimeError("retrieval returned a non-finite similarity")
    if bool((top.values[:, 1:] > top.values[:, :-1] + 1e-7).any()):
        raise RuntimeError("retrieval order is not descending")
    return RetrievalResult(
        normalized_features=features,
        background_indices=background,
        neighbor_indices=indices.contiguous(),
        cosine_similarities=top.values.contiguous(),
        self_match_violation_count=violations,
    )
