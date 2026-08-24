"""Mathematical core for candidate-level GBSP influence diagnosis.

The functions in this module are deliberately estimator-agnostic diagnostics:
they neither change candidate weights nor produce a pseudo-label.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InfluenceProxy:
    in_subspace_energy: torch.Tensor
    orthogonal_energy: torch.Tensor
    raw: torch.Tensor
    gap_normalized: torch.Tensor
    spectral_gap: float


@dataclass(frozen=True)
class ExactLOO:
    mean: torch.Tensor
    basis: torch.Tensor
    influence: torch.Tensor
    oracle_distance_after: torch.Tensor
    harmful_oracle_improvement: torch.Tensor


def fit_scatter_basis(background: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit an affine PCA basis through a symmetric scatter eigendecomposition."""
    background = _matrix(background, "background")
    mean = background.mean(0)
    centered = background - mean
    _, eigenvectors = torch.linalg.eigh(centered.transpose(0, 1) @ centered)
    return mean, eigenvectors[:, -int(rank):].contiguous()


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a non-empty Tensor[N,D]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def projector_distance_from_bases(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Normalized Frobenius distance between equal-rank projectors.

    Supports ``left=[...,D,r]`` and ``right=[D,r]`` or matching batched input.
    The identity ``||UU'-VV'||_F^2 = 2r-2||U'V||_F^2`` avoids materializing
    384x384 projectors in the formal batched LOO pass.
    """
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError("basis shapes must agree in [D,r]")
    rank = int(left.shape[-1])
    if rank < 1:
        raise ValueError("rank must be positive")
    # The relevant distances are often 1e-4--1e-2.  Computing the overlap in
    # float32 can round ``||U'V||_F^2`` to exactly r and erase a real small
    # rotation, so the tiny r x r product is deliberately evaluated in fp64.
    left64, right64 = left.double(), right.double()
    cross = left64.transpose(-2, -1) @ right64
    overlap = cross.square().sum(dim=(-2, -1))
    return torch.sqrt((1.0 - overlap / float(rank)).clamp_min(0.0))


def projector_frobenius_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Unnormalized projector Frobenius error used by the correctness gate."""
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError("basis shapes must agree in [D,r]")
    left64, right64 = left.double(), right.double()
    delta = left64 @ left64.transpose(-2, -1) - right64 @ right64.transpose(-2, -1)
    return torch.linalg.vector_norm(delta, dim=(-2, -1))


def candidate_influence_proxy(
    background: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> InfluenceProxy:
    background = _matrix(background, "background")
    if mean.shape != (background.shape[1],):
        raise ValueError("mean shape mismatch")
    if basis.ndim != 2 or basis.shape[0] != background.shape[1]:
        raise ValueError("basis shape mismatch")
    rank = int(basis.shape[1])
    singular = singular_values.reshape(-1).to(background)
    if singular.numel() <= rank:
        raise ValueError("spectrum must contain lambda_(r+1)")
    centered = background - mean
    coordinates = centered @ basis
    in_energy = torch.linalg.vector_norm(coordinates, dim=1)
    projected = coordinates @ basis.transpose(0, 1)
    orthogonal = torch.linalg.vector_norm(centered - projected, dim=1)
    raw = in_energy * orthogonal
    eigenvalues = singular.square()
    gap = float((eigenvalues[rank - 1] - eigenvalues[rank]).clamp_min(0.0))
    return InfluenceProxy(
        in_subspace_energy=in_energy,
        orthogonal_energy=orthogonal,
        raw=raw,
        gap_normalized=raw / (gap + float(eps)),
        spectral_gap=gap,
    )


def loo_means(background: torch.Tensor, candidate_indices: torch.Tensor) -> torch.Tensor:
    background = _matrix(background, "background")
    index = candidate_indices.reshape(-1).long().to(background.device)
    count = int(background.shape[0])
    if count < 3 or bool((index < 0).any()) or bool((index >= count).any()):
        raise ValueError("invalid leave-one-out candidate index")
    mean = background.mean(0)
    return (count * mean.unsqueeze(0) - background.index_select(0, index)) / float(count - 1)


def exact_loo_batch(
    background: torch.Tensor,
    candidate_indices: torch.Tensor,
    original_basis: torch.Tensor,
    oracle_bases: torch.Tensor,
) -> ExactLOO:
    """Exact top-r LOO eigenspaces using the rank-one scatter downdate."""
    background = _matrix(background, "background")
    device, dtype = background.device, background.dtype
    index = candidate_indices.reshape(-1).long().to(device)
    count, dimension = map(int, background.shape)
    rank = int(original_basis.shape[1])
    if original_basis.shape != (dimension, rank):
        raise ValueError("original_basis shape mismatch")
    if oracle_bases.ndim != 3 or oracle_bases.shape[1:] != (dimension, rank):
        raise ValueError("oracle_bases must be Tensor[S,D,r]")
    mean = background.mean(0)
    centered = background - mean
    scatter = centered.transpose(0, 1) @ centered
    selected = centered.index_select(0, index)
    alpha = float(count) / float(count - 1)
    downdated = scatter.unsqueeze(0) - alpha * selected.unsqueeze(2) * selected.unsqueeze(1)
    # eigh is ascending; the final r columns are the exact top-r eigenspace.
    _, eigenvectors = torch.linalg.eigh(downdated)
    basis = eigenvectors[:, :, -rank:].contiguous()
    influence = projector_distance_from_bases(
        basis, original_basis.to(device=device, dtype=dtype).unsqueeze(0).expand_as(basis)
    )
    oracle = oracle_bases.to(device=device, dtype=dtype)
    original = original_basis.to(device=device, dtype=dtype)
    distance_before = torch.stack([
        projector_distance_from_bases(original, item) for item in oracle
    ]).mean()
    distance_after = torch.stack([
        projector_distance_from_bases(basis, item.unsqueeze(0).expand_as(basis))
        for item in oracle
    ], dim=0).mean(0)
    return ExactLOO(
        mean=loo_means(background, index),
        basis=basis,
        influence=influence,
        oracle_distance_after=distance_after,
        harmful_oracle_improvement=distance_before - distance_after,
    )


def brute_force_loo(
    background: torch.Tensor,
    candidate_index: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal delete/recenter/SVD reference used only by correctness tests."""
    background = _matrix(background, "background")
    keep = torch.ones(background.shape[0], dtype=torch.bool, device=background.device)
    keep[int(candidate_index)] = False
    reduced = background[keep]
    mean = reduced.mean(0)
    _, _, vh = torch.linalg.svd(reduced - mean, full_matrices=False)
    return mean, vh[: int(rank)].transpose(0, 1).contiguous()


def score_queries(query: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    query = _matrix(query, "query")
    centered = query - mean
    residual = centered - (centered @ basis) @ basis.transpose(0, 1)
    return residual.square().sum(1)
