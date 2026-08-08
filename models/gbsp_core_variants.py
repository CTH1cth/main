"""Controlled single-variable variants of the frozen GBSP PCA core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


GRID = 37
NUM_PATCHES = GRID * GRID
BOUNDARY_WIDTH = 2
BOUNDARY_COUNT = 280


@dataclass(frozen=True)
class PCAFit:
    mean: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    selected_rank: int
    effective_spectrum_dimension: int
    retained_variance_ratio: float
    discarded_variance_ratio: float
    orthonormal_error: float


@dataclass(frozen=True)
class PCADecomposition:
    mean: torch.Tensor
    singular_values: torch.Tensor
    right_singular_vectors: torch.Tensor
    num_samples: int
    feature_dimension: int
    effective_spectrum_dimension: int


@dataclass(frozen=True)
class WeightedPCAFit:
    fit: PCAFit
    normalized_weights: torch.Tensor
    path_cost: torch.Tensor
    path_cost_median: float
    effective_sample_size: float
    weight_concentration_warning: bool


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a non-empty Tensor[N,D]")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def boundary_two_ring_mask(grid: int = GRID) -> torch.Tensor:
    if int(grid) != GRID:
        raise ValueError(f"formal boundary mask requires grid={GRID}")
    mask = torch.zeros(GRID, GRID, dtype=torch.bool)
    mask[:BOUNDARY_WIDTH, :] = True
    mask[-BOUNDARY_WIDTH:, :] = True
    mask[:, :BOUNDARY_WIDTH] = True
    mask[:, -BOUNDARY_WIDTH:] = True
    flat = mask.reshape(-1)
    if int(flat.sum()) != BOUNDARY_COUNT:
        raise RuntimeError(f"two-ring boundary must contain {BOUNDARY_COUNT} patches")
    return flat.contiguous()


class PCARankSelector:
    """Frozen fixed/current/explained-variance rank rules."""

    def __init__(self, current_energy: float = .90, current_max_rank: int = 8, current_min_rank: int = 1) -> None:
        if not 0.0 < float(current_energy) <= 1.0:
            raise ValueError("current_energy must be in (0,1]")
        self.current_energy = float(current_energy)
        self.current_max_rank = int(current_max_rank)
        self.current_min_rank = int(current_min_rank)

    @staticmethod
    def effective_dimension(num_samples: int, feature_dimension: int) -> int:
        return min(int(num_samples) - 1, int(feature_dimension))

    @staticmethod
    def _energy_rank(singular_values: torch.Tensor, effective: int, target: float) -> int:
        energy = singular_values[:effective].double().square()
        if not energy.numel() or float(energy.sum()) <= 0.0:
            raise ValueError("PCA spectrum has no positive energy")
        cumulative = torch.cumsum(energy, 0) / energy.sum()
        return int(torch.searchsorted(cumulative, torch.tensor(float(target), dtype=cumulative.dtype)).item()) + 1

    def select(self, mode: str, singular_values: torch.Tensor, num_samples: int, feature_dimension: int, fixed_rank: int | None = None) -> int:
        effective = self.effective_dimension(num_samples, feature_dimension)
        if effective < 1:
            raise ValueError("PCA requires at least two background candidates")
        if mode == "fixed":
            if fixed_rank is None:
                raise ValueError("fixed rank mode requires fixed_rank")
            rank = int(fixed_rank)
        elif mode == "current":
            energy_rank = self._energy_rank(singular_values, effective, self.current_energy)
            rank = min(self.current_max_rank, max(self.current_min_rank, energy_rank))
        elif mode == "ev90":
            rank = self._energy_rank(singular_values, effective, .90)
        elif mode == "ev95":
            rank = self._energy_rank(singular_values, effective, .95)
        else:
            raise ValueError(f"unknown rank mode: {mode}")
        if rank < 0 or rank >= effective:
            raise ValueError(f"rank {rank} is invalid for effective dimension {effective}")
        return rank


def decompose_pca(background_features: torch.Tensor) -> PCADecomposition:
    """Compute the shared decomposition once for a controlled rank sweep."""
    background = _matrix(background_features, "background_features")
    mean = background.mean(0)
    _, singular, vh = torch.linalg.svd(background - mean, full_matrices=False)
    effective = PCARankSelector.effective_dimension(background.shape[0], background.shape[1])
    return PCADecomposition(
        mean=mean.contiguous(),
        singular_values=singular.contiguous(),
        right_singular_vectors=vh.contiguous(),
        num_samples=int(background.shape[0]),
        feature_dimension=int(background.shape[1]),
        effective_spectrum_dimension=effective,
    )


def pca_fit_from_decomposition(
    decomposition: PCADecomposition,
    rank_mode: str,
    selector: PCARankSelector,
    fixed_rank: int | None = None,
) -> PCAFit:
    rank = selector.select(
        rank_mode,
        decomposition.singular_values,
        decomposition.num_samples,
        decomposition.feature_dimension,
        fixed_rank,
    )
    basis = (
        decomposition.right_singular_vectors[:rank].t().contiguous()
        if rank
        else decomposition.mean.new_zeros((decomposition.feature_dimension, 0))
    )
    error = float((basis.t() @ basis - torch.eye(rank)).abs().max()) if rank else 0.0
    if error >= 1e-4 or not bool(torch.isfinite(basis).all()):
        raise RuntimeError(f"PCA basis invalid: orthonormal_error={error}")
    energy = decomposition.singular_values[: decomposition.effective_spectrum_dimension].square()
    ratio = float(energy[:rank].sum() / energy.sum()) if float(energy.sum()) > 0 else 0.0
    return PCAFit(
        decomposition.mean,
        basis,
        decomposition.singular_values,
        rank,
        decomposition.effective_spectrum_dimension,
        ratio,
        1.0 - ratio,
        error,
    )


def fit_pca_from_svd(
    background_features: torch.Tensor,
    rank_mode: str,
    selector: PCARankSelector,
    fixed_rank: int | None = None,
) -> PCAFit:
    return pca_fit_from_decomposition(
        decompose_pca(background_features), rank_mode, selector, fixed_rank
    )


def score_all_patches(query_features: torch.Tensor, fit: PCAFit) -> torch.Tensor:
    query = _matrix(query_features, "query_features")
    if int(query.shape[0]) != NUM_PATCHES:
        raise ValueError(f"all {NUM_PATCHES} patches must be queried")
    centered = query - fit.mean
    projected = (centered @ fit.basis) @ fit.basis.t() if fit.selected_rank else torch.zeros_like(centered)
    residual = (centered - projected).square().sum(1)
    if int(residual.shape[0]) != NUM_PATCHES or not bool(torch.isfinite(residual).all()):
        raise RuntimeError("PCA residual is incomplete or non-finite")
    return residual.float().contiguous()


def minmax_score(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = value.detach().cpu().float()
    low, high = value.min(), value.max()
    return torch.zeros_like(value) if float(high - low) <= eps else ((value - low) / (high - low + eps)).clamp(0, 1)


class BackgroundCandidateSelector:
    """Boundary-280, reliability-matched Full-BC-280 and current Full-BC."""

    def __init__(self, matched_count: int = BOUNDARY_COUNT) -> None:
        self.matched_count = int(matched_count)

    def select(self, mode: str, full_bc_indices: torch.Tensor, connectivity: torch.Tensor) -> torch.Tensor:
        full = full_bc_indices.detach().cpu().long().reshape(-1)
        if full.numel() == 0 or full.numel() != torch.unique(full).numel():
            raise ValueError("full_bc_indices must contain unique candidates")
        confidence = connectivity.detach().cpu().float().reshape(-1)
        if confidence.numel() != NUM_PATCHES or not bool(torch.isfinite(confidence).all()):
            raise ValueError("connectivity must contain 1369 finite values")
        if mode == "boundary280":
            selected = torch.where(boundary_two_ring_mask())[0]
        elif mode == "fullbc_matched280":
            if full.numel() < self.matched_count:
                raise ValueError("Full BC cannot support matched capacity")
            reliability = confidence.index_select(0, full)
            order = torch.argsort(reliability, descending=True, stable=True)
            selected = full.index_select(0, order[: self.matched_count])
        elif mode == "fullbc":
            selected = full
        else:
            raise ValueError(f"unknown background mode: {mode}")
        if selected.numel() == 0 or selected.numel() != torch.unique(selected).numel():
            raise RuntimeError(f"invalid candidate selection for {mode}")
        return selected.long().contiguous()


class ConnectivityWeightedPCA:
    """Estimate a weighted mean/subspace using frozen path confidence."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def weights(self, candidate_connectivity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float, bool]:
        connectivity = candidate_connectivity.detach().cpu().double().reshape(-1)
        if not connectivity.numel() or not bool(torch.isfinite(connectivity).all()) or bool((connectivity <= 0).any()):
            raise ValueError("candidate connectivity must be finite and positive")
        # bc = exp(-normalized_path/tau); -log(bc) differs from the stored
        # normalized path only by a positive image-wise constant.  Dividing by
        # the candidate median removes that constant exactly.
        path = -torch.log(connectivity.clamp_min(self.eps))
        median = float(path.median())
        normalized_path = path / (median + self.eps)
        weights = 1.0 / (1.0 + normalized_path)
        weights = weights * (weights.numel() / weights.sum())
        effective = float(weights.sum().square() / weights.square().sum())
        warning = effective < .5 * weights.numel()
        return weights.float().contiguous(), path.float().contiguous(), median, effective, warning

    def fit(
        self,
        background_features: torch.Tensor,
        candidate_connectivity: torch.Tensor,
        rank_mode: str,
        selector: PCARankSelector,
        fixed_rank: int | None = None,
    ) -> WeightedPCAFit:
        background = _matrix(background_features, "background_features")
        weights, path, median, effective_sample, warning = self.weights(candidate_connectivity)
        if weights.numel() != background.shape[0]:
            raise ValueError("weights/background size mismatch")
        w = weights.to(background.dtype)
        mean = (w[:, None] * background).sum(0) / w.sum()
        centered = background - mean
        weighted_centered = torch.sqrt(w)[:, None] * centered
        _, singular, vh = torch.linalg.svd(weighted_centered, full_matrices=False)
        rank = selector.select(rank_mode, singular, background.shape[0], background.shape[1], fixed_rank)
        basis = vh[:rank].t().contiguous() if rank else background.new_zeros((background.shape[1], 0))
        error = float((basis.t() @ basis - torch.eye(rank)).abs().max()) if rank else 0.0
        if error >= 1e-4 or not bool(torch.isfinite(basis).all()):
            raise RuntimeError(f"weighted PCA basis invalid: {error}")
        effective_dimension = selector.effective_dimension(background.shape[0], background.shape[1])
        energy = singular[:effective_dimension].square()
        ratio = float(energy[:rank].sum() / energy.sum()) if float(energy.sum()) > 0 else 0.0
        fit = PCAFit(mean, basis, singular.contiguous(), rank, effective_dimension, ratio, 1.0 - ratio, error)
        return WeightedPCAFit(fit, weights, path, median, effective_sample, warning)


def principal_angles_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("basis matrices must share their ambient dimension")
    count = min(left.shape[1], right.shape[1])
    if count == 0:
        return torch.empty(0)
    cosine = torch.linalg.svdvals(left.t() @ right)[:count].clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine)).float().contiguous()
