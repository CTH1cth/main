"""Query-conditioned local background subspace reconstruction.

This module deliberately contains no GT handling.  Cosine KNN only chooses a
query-specific subset of the frozen Full-BC candidates; the final anomaly
score is the squared orthogonal residual to a PCA model fitted on that subset.
It is therefore different from the old fixed global M-subspace mixture.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


GRID = 37
NUM_PATCHES = GRID * GRID


@dataclass(frozen=True)
class NeighborRetrieval:
    features: torch.Tensor
    background_indices: torch.Tensor
    neighbor_indices: torch.Tensor
    cosine_similarities: torch.Tensor
    self_match_violation_count: int


@dataclass(frozen=True)
class LocalReconstructionResult:
    k: int
    rank: int
    neighbor_indices: torch.Tensor
    cosine_similarities: torch.Tensor
    local_mean_norm: torch.Tensor
    singular_values: torch.Tensor
    effective_rank: torch.Tensor
    distance_to_local_mean: torch.Tensor
    distance_to_global_mean: torch.Tensor
    local_residual: torch.Tensor
    global_residual: torch.Tensor
    knn8_anomaly: torch.Tensor
    max_orthonormal_error: float
    self_match_violation_count: int


def _features(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError("features must be a torch.Tensor")
    if value.ndim == 3:
        value = value.reshape(value.shape[0], -1).T
    if value.ndim != 2 or value.shape[0] != NUM_PATCHES:
        raise ValueError(f"features must be [C,37,37] or [1369,C], got {tuple(value.shape)}")
    value = F.normalize(value.float(), dim=1, eps=1e-12).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("features contain NaN/Inf")
    return value


def retrieve_background_neighbors(
    features: torch.Tensor,
    background_indices: torch.Tensor,
    *,
    max_k: int = 32,
) -> NeighborRetrieval:
    """Retrieve sorted Full-BC cosine neighbors with strict leave-one-out."""
    x = _features(features)
    bg_idx = torch.as_tensor(background_indices, dtype=torch.long, device=x.device).reshape(-1)
    if not bg_idx.numel() or bg_idx.numel() != torch.unique(bg_idx).numel():
        raise ValueError("background_indices must be non-empty and unique")
    if int(bg_idx.min()) < 0 or int(bg_idx.max()) >= NUM_PATCHES:
        raise ValueError("background index out of range")
    if bg_idx.numel() < int(max_k) + 1:
        raise ValueError(f"leave-one-out KNN-{max_k} needs at least {int(max_k) + 1} candidates")

    similarity = x @ x.index_select(0, bg_idx).T
    columns = torch.arange(bg_idx.numel(), device=x.device)
    similarity[bg_idx, columns] = -torch.inf
    top = torch.topk(similarity, k=int(max_k), dim=1, largest=True, sorted=True)
    neighbor_indices = bg_idx.index_select(0, top.indices.reshape(-1)).reshape(NUM_PATCHES, int(max_k))
    query_index = torch.arange(NUM_PATCHES, device=x.device).unsqueeze(1)
    violations = int((neighbor_indices == query_index).sum().item())
    if violations:
        raise RuntimeError(f"self-match exclusion failed: {violations} violations")
    if not bool(torch.isfinite(top.values).all()):
        raise RuntimeError("neighbor similarities contain NaN/Inf")
    if bool((top.values[:, 1:] > top.values[:, :-1] + 1e-7).any()):
        raise RuntimeError("KNN similarities are not sorted descending")
    return NeighborRetrieval(
        features=x,
        background_indices=bg_idx,
        neighbor_indices=neighbor_indices.contiguous(),
        cosine_similarities=top.values.contiguous(),
        self_match_violation_count=violations,
    )


def local_reconstruction_from_neighbors(
    retrieval: NeighborRetrieval,
    *,
    k: int,
    rank: int,
    global_mean: torch.Tensor,
    global_residual: torch.Tensor,
    query_batch_size: int = 128,
) -> LocalReconstructionResult:
    """Fit one local PCA per query and return all required diagnostics."""
    k, rank = int(k), int(rank)
    if k < 2 or k > retrieval.neighbor_indices.shape[1]:
        raise ValueError(f"invalid local K={k}")
    if rank < 1 or rank >= k - 1:
        raise ValueError(f"rank must be in [1,K-2], got K={k}, rank={rank}")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be positive")
    x = retrieval.features
    mean_global = torch.as_tensor(global_mean, dtype=x.dtype, device=x.device).reshape(-1)
    if mean_global.numel() != x.shape[1] or not bool(torch.isfinite(mean_global).all()):
        raise ValueError("global_mean has the wrong dimension or contains NaN/Inf")
    global_q = torch.as_tensor(global_residual, dtype=x.dtype, device=x.device).reshape(-1)
    if global_q.numel() != NUM_PATCHES or not bool(torch.isfinite(global_q).all()):
        raise ValueError("global_residual must contain 1369 finite scores")

    neighbors = retrieval.neighbor_indices[:, :k]
    similarities = retrieval.cosine_similarities[:, :k]
    outputs: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "local_mean_norm", "singular_values", "effective_rank",
            "distance_to_local_mean", "distance_to_global_mean", "local_residual",
        )
    }
    max_orthonormal_error = 0.0
    for start in range(0, NUM_PATCHES, int(query_batch_size)):
        stop = min(start + int(query_batch_size), NUM_PATCHES)
        query = x[start:stop]
        index = neighbors[start:stop]
        local = x.index_select(0, index.reshape(-1)).reshape(stop - start, k, x.shape[1])
        local_mean = local.mean(dim=1)
        centered = local - local_mean.unsqueeze(1)
        # K is at most 32 while D=384.  Eigh on the KxK sample Gram matrix is
        # algebraically equivalent to a thin SVD of [K,D], but is much faster
        # for all 8.86M query-specific fits in the formal run.
        gram_samples = centered @ centered.transpose(1, 2)
        eigenvalues, left_vectors = torch.linalg.eigh(gram_samples)
        eigenvalues = eigenvalues.flip(1).clamp_min(0.0)
        left_vectors = left_vectors.flip(2)
        singular = eigenvalues.sqrt()
        singular_top = singular[:, :rank].clamp_min(1e-12)
        left_top = left_vectors[:, :, :rank]
        basis_rows = torch.einsum("bkr,bkd->brd", left_top, centered) / singular_top.unsqueeze(2)
        gram = basis_rows @ basis_rows.transpose(1, 2)
        identity = torch.eye(rank, dtype=x.dtype, device=x.device).expand_as(gram)
        orth_error = float((gram - identity).abs().max().item())
        max_orthonormal_error = max(max_orthonormal_error, orth_error)

        query_centered = query - local_mean
        coefficient = torch.einsum("brd,bd->br", basis_rows, query_centered)
        residual = (query_centered.square().sum(dim=1) - coefficient.square().sum(dim=1)).clamp_min(0.0)
        energy = singular.double().square()
        probability = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-24)
        entropy = -(probability * probability.clamp_min(1e-24).log()).sum(dim=1)

        outputs["local_mean_norm"].append(local_mean.norm(dim=1))
        outputs["singular_values"].append(singular)
        outputs["effective_rank"].append(entropy.exp().to(x.dtype))
        outputs["distance_to_local_mean"].append(query_centered.norm(dim=1))
        outputs["distance_to_global_mean"].append((query - mean_global).norm(dim=1))
        outputs["local_residual"].append(residual)

    result = {name: torch.cat(parts, dim=0).contiguous() for name, parts in outputs.items()}
    if max_orthonormal_error >= 1e-4:
        raise RuntimeError(f"local PCA basis is not orthonormal: {max_orthonormal_error}")
    for name, value in result.items():
        if value.shape[0] != NUM_PATCHES or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"invalid local diagnostic {name}: {tuple(value.shape)}")
    if bool((result["local_residual"] < -1e-7).any()):
        raise RuntimeError("local residual became negative")

    return LocalReconstructionResult(
        k=k,
        rank=rank,
        neighbor_indices=neighbors,
        cosine_similarities=similarities,
        local_mean_norm=result["local_mean_norm"],
        singular_values=result["singular_values"],
        effective_rank=result["effective_rank"],
        distance_to_local_mean=result["distance_to_local_mean"],
        distance_to_global_mean=result["distance_to_global_mean"],
        local_residual=result["local_residual"],
        global_residual=global_q,
        knn8_anomaly=1.0 - retrieval.cosine_similarities[:, :8].mean(dim=1),
        max_orthonormal_error=max_orthonormal_error,
        self_match_violation_count=retrieval.self_match_violation_count,
    )


def compute_local_background_subspace(
    features: torch.Tensor,
    background_indices: torch.Tensor,
    *,
    k: int,
    rank: int,
    global_mean: torch.Tensor,
    global_residual: torch.Tensor,
    query_batch_size: int = 128,
) -> LocalReconstructionResult:
    """Convenience wrapper for a single predeclared LSR configuration."""
    retrieval = retrieve_background_neighbors(features, background_indices, max_k=int(k))
    return local_reconstruction_from_neighbors(
        retrieval,
        k=int(k),
        rank=int(rank),
        global_mean=global_mean,
        global_residual=global_residual,
        query_batch_size=int(query_batch_size),
    )
