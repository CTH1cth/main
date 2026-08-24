"""Resolution-agnostic Full-BC + global PCA reconstruction for GBSP audits.

This module deliberately contains no 37/64/68 constants.  The native token
grid is inferred from the cached feature tensor, while all method parameters
are supplied by the experiment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from common.dabe_pseudo import (
    _background_anchor,
    _background_connectivity,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
)


EPS = 1e-8


@dataclass(frozen=True)
class PreparedResolutionGraph:
    feature: torch.Tensor
    normalized_feature: torch.Tensor
    rgb: torch.Tensor
    sobel: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_affinity: torch.Tensor
    graph_terms: dict[str, torch.Tensor]
    grid_h: int
    grid_w: int
    feature_dim: int
    rgb_seconds: float
    graph_seconds: float


@dataclass(frozen=True)
class GBSPResolutionResult:
    bc: torch.Tensor
    border: torch.Tensor
    background_anchor: torch.Tensor
    background_indices: torch.Tensor
    raw_residual: torch.Tensor
    minmax_residual: torch.Tensor
    selected_rank: int
    required_rank_uncapped: int
    energy_at_rank_max: float
    retained_energy: float
    hit_rank_cap: bool
    singular_values: torch.Tensor
    energy_at_rank_8: float
    energy_at_rank_12: float
    bc_seconds: float
    pca_seconds: float


def _energy_at(cumulative: torch.Tensor, rank: int) -> float:
    if not cumulative.numel():
        return 0.0
    index = min(max(int(rank), 1), int(cumulative.numel())) - 1
    return float(cumulative[index])


def _fit_background_subspace(
    background: torch.Tensor,
    query: torch.Tensor,
    *,
    pca_energy: float,
    pca_min_rank: int,
    pca_max_rank: int,
) -> tuple[torch.Tensor, int, int, torch.Tensor, float, float, bool]:
    """Fit one affine background PCA and score arbitrary compatible queries."""
    mean = background.mean(0)
    centered_background = background - mean
    _, singular_values, vh = torch.linalg.svd(centered_background, full_matrices=False)
    effective = min(int(background.shape[0]) - 1, int(background.shape[1]))
    if effective < 1:
        raise RuntimeError("PCA effective dimension is zero")
    singular_values = singular_values[:effective]
    required_rank, cumulative = _energy_rank(singular_values, pca_energy)
    selected_rank = min(int(pca_max_rank), max(int(pca_min_rank), int(required_rank)))
    selected_rank = min(selected_rank, effective)
    basis = vh[:selected_rank].t().contiguous()
    identity = torch.eye(selected_rank, dtype=basis.dtype)
    orthonormal_error = float((basis.t() @ basis - identity).abs().max())
    if not bool(torch.isfinite(basis).all()) or orthonormal_error >= 1e-3:
        raise RuntimeError(f"PCA basis invalid: orthonormal_error={orthonormal_error}")
    centered_query = query - mean
    projected = (centered_query @ basis) @ basis.t()
    residual = (centered_query - projected).square().sum(1)
    return (
        residual.float().contiguous(),
        selected_rank,
        required_rank,
        singular_values.float().contiguous(),
        _energy_at(cumulative, 8),
        _energy_at(cumulative, 12),
        required_rank >= int(pca_max_rank),
    )


def _validate_feature(feature: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if not torch.is_tensor(feature) or feature.ndim != 3:
        raise ValueError(f"feature must be Tensor[C,H,W], got {type(feature)!r}")
    value = feature.detach().cpu().float().contiguous()
    channels, grid_h, grid_w = map(int, value.shape)
    if grid_h != grid_w:
        raise ValueError(f"GBSP currently requires a square native grid, got {grid_h}x{grid_w}")
    if channels < 1 or grid_h < 2 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"invalid feature tensor: shape={list(value.shape)}")
    return value, grid_h, grid_w, channels


def prepare_resolution_graph(
    feature: torch.Tensor,
    image_path: str,
    graph_params: dict,
) -> PreparedResolutionGraph:
    """Prepare RGB/Sobel grids and the frozen eight-neighbour graph."""
    value, grid_h, grid_w, channels = _validate_feature(feature)
    rgb_started = time.perf_counter()
    rgb = _load_rgb_grid(image_path, grid_h).float().contiguous()
    sobel = _sobel_magnitude(rgb).float().contiguous()
    rgb_seconds = time.perf_counter() - rgb_started
    if tuple(rgb.shape) != (3, grid_h, grid_w):
        raise RuntimeError(f"RGB grid mismatch: {tuple(rgb.shape)}")
    if tuple(sobel.shape) != (grid_h, grid_w):
        raise RuntimeError(f"Sobel grid mismatch: {tuple(sobel.shape)}")

    normalized = F.normalize(
        value.permute(1, 2, 0).reshape(grid_h * grid_w, channels), p=2, dim=1
    )
    rgb_flat = rgb.permute(1, 2, 0).reshape(grid_h * grid_w, 3)
    sobel_flat = sobel.reshape(-1)
    graph_started = time.perf_counter()
    neighbor_indices, neighbor_affinity = _build_local_graph(
        normalized, rgb_flat, sobel_flat, grid_h, graph_params
    )
    valid = neighbor_affinity > 0
    src = torch.arange(grid_h * grid_w).view(-1, 1).expand_as(neighbor_indices)[valid]
    dst = neighbor_indices[valid]
    semantic = (1.0 - (normalized[src] * normalized[dst]).sum(1)).float()
    color = (rgb_flat[src] - rgb_flat[dst]).square().sum(1).float()
    edge = torch.maximum(sobel_flat[src], sobel_flat[dst]).float()
    affinity = neighbor_affinity[valid].float()
    graph_seconds = time.perf_counter() - graph_started
    terms = {
        "semantic_distance": semantic.contiguous(),
        "rgb_distance": color.contiguous(),
        "sobel_term": edge.contiguous(),
        "graph_affinity": affinity.contiguous(),
    }
    for name, tensor in terms.items():
        if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"invalid graph term {name}")
    return PreparedResolutionGraph(
        feature=value,
        normalized_feature=normalized.contiguous(),
        rgb=rgb,
        sobel=sobel,
        neighbor_indices=neighbor_indices,
        neighbor_affinity=neighbor_affinity,
        graph_terms=terms,
        grid_h=grid_h,
        grid_w=grid_w,
        feature_dim=channels,
        rgb_seconds=rgb_seconds,
        graph_seconds=graph_seconds,
    )


def _energy_rank(singular_values: torch.Tensor, target: float) -> tuple[int, torch.Tensor]:
    energy = singular_values.double().square()
    total = energy.sum()
    if not energy.numel() or float(total) <= 0.0:
        raise ValueError("PCA spectrum has no positive energy")
    cumulative = torch.cumsum(energy, 0) / total
    rank = int(
        torch.searchsorted(
            cumulative, torch.tensor(float(target), dtype=cumulative.dtype)
        ).item()
    ) + 1
    return rank, cumulative


def fit_gbsp_from_prepared(
    prepared: PreparedResolutionGraph,
    graph_params: dict,
    *,
    pca_energy: float,
    pca_min_rank: int,
    pca_max_rank: int,
) -> GBSPResolutionResult:
    """Run BC selection and one image-specific affine PCA reconstruction."""
    grid = prepared.grid_h
    if grid != prepared.grid_w:
        raise RuntimeError("non-square prepared graph")
    bc_started = time.perf_counter()
    bc, border = _background_connectivity(
        prepared.neighbor_indices,
        prepared.neighbor_affinity,
        grid,
        graph_params,
    )
    anchor = _background_anchor(bc, border, graph_params)
    indices = torch.where(anchor)[0]
    bc_seconds = time.perf_counter() - bc_started
    if indices.numel() < 2:
        raise RuntimeError("Full-BC returned fewer than two candidates")

    pca_started = time.perf_counter()
    background = prepared.normalized_feature.index_select(0, indices)
    (
        residual,
        selected_rank,
        required_rank,
        singular_values,
        energy_at_8,
        energy_at_12,
        hit_rank_cap,
    ) = _fit_background_subspace(
        background,
        prepared.normalized_feature,
        pca_energy=pca_energy,
        pca_min_rank=pca_min_rank,
        pca_max_rank=pca_max_rank,
    )
    residual = residual.reshape(1, grid, grid).float().contiguous()
    low, high = residual.min(), residual.max()
    calibrated = (
        torch.zeros_like(residual)
        if float(high - low) <= EPS
        else ((residual - low) / (high - low + EPS)).clamp(0.0, 1.0)
    )
    pca_seconds = time.perf_counter() - pca_started
    energy = singular_values.double().square()
    cumulative = torch.cumsum(energy, 0) / energy.sum()
    energy_at_cap = _energy_at(cumulative, int(pca_max_rank))
    retained = _energy_at(cumulative, selected_rank)
    return GBSPResolutionResult(
        bc=bc.reshape(1, grid, grid).float().contiguous(),
        border=border.reshape(1, grid, grid).bool().contiguous(),
        background_anchor=anchor.reshape(1, grid, grid).bool().contiguous(),
        background_indices=indices.long().contiguous(),
        raw_residual=residual,
        minmax_residual=calibrated.float().contiguous(),
        selected_rank=selected_rank,
        required_rank_uncapped=required_rank,
        energy_at_rank_max=energy_at_cap,
        retained_energy=retained,
        hit_rank_cap=hit_rank_cap,
        singular_values=singular_values,
        energy_at_rank_8=energy_at_8,
        energy_at_rank_12=energy_at_12,
        bc_seconds=bc_seconds,
        pca_seconds=pca_seconds,
    )


def fit_coarse_background_fine_query(
    feature: torch.Tensor,
    image_path: str,
    graph_params: dict,
    *,
    pca_energy: float,
    pca_min_rank: int,
    pca_max_rank: int,
    pooling: int = 2,
) -> GBSPResolutionResult:
    """Fit BC/PCA on a pooled grid while retaining the native fine query grid."""
    value, fine_h, fine_w, channels = _validate_feature(feature)
    pooling = int(pooling)
    if pooling != 2 or fine_h % pooling or fine_w % pooling:
        raise ValueError(f"only exact 2x2 pooling is supported, got {fine_h}x{fine_w}")
    pooled = F.avg_pool2d(value.unsqueeze(0), kernel_size=pooling, stride=pooling).squeeze(0)
    coarse = prepare_resolution_graph(pooled, image_path, graph_params)

    bc_started = time.perf_counter()
    bc, border = _background_connectivity(
        coarse.neighbor_indices, coarse.neighbor_affinity, coarse.grid_h, graph_params
    )
    anchor = _background_anchor(bc, border, graph_params)
    indices = torch.where(anchor)[0]
    bc_seconds = time.perf_counter() - bc_started
    if indices.numel() < 2:
        raise RuntimeError("coarse Full-BC returned fewer than two candidates")

    fine_query = F.normalize(
        value.permute(1, 2, 0).reshape(fine_h * fine_w, channels), p=2, dim=1
    )
    background = coarse.normalized_feature.index_select(0, indices)
    pca_started = time.perf_counter()
    (
        residual,
        selected_rank,
        required_rank,
        singular_values,
        energy_at_8,
        energy_at_12,
        hit_rank_cap,
    ) = _fit_background_subspace(
        background,
        fine_query,
        pca_energy=pca_energy,
        pca_min_rank=pca_min_rank,
        pca_max_rank=pca_max_rank,
    )
    residual = residual.reshape(1, fine_h, fine_w).float().contiguous()
    low, high = residual.min(), residual.max()
    calibrated = (
        torch.zeros_like(residual)
        if float(high - low) <= EPS
        else ((residual - low) / (high - low + EPS)).clamp(0.0, 1.0)
    )
    energy = singular_values.double().square()
    cumulative = torch.cumsum(energy, 0) / energy.sum()
    pca_seconds = time.perf_counter() - pca_started
    return GBSPResolutionResult(
        bc=bc.reshape(1, coarse.grid_h, coarse.grid_w).float().contiguous(),
        border=border.reshape(1, coarse.grid_h, coarse.grid_w).bool().contiguous(),
        background_anchor=anchor.reshape(1, coarse.grid_h, coarse.grid_w).bool().contiguous(),
        background_indices=indices.long().contiguous(),
        raw_residual=residual,
        minmax_residual=calibrated.float().contiguous(),
        selected_rank=selected_rank,
        required_rank_uncapped=required_rank,
        energy_at_rank_max=_energy_at(cumulative, int(pca_max_rank)),
        retained_energy=_energy_at(cumulative, selected_rank),
        hit_rank_cap=hit_rank_cap,
        singular_values=singular_values,
        energy_at_rank_8=energy_at_8,
        energy_at_rank_12=energy_at_12,
        bc_seconds=bc_seconds,
        pca_seconds=pca_seconds,
    )


def hard_mask_at_native(
    score: torch.Tensor,
    threshold: float,
    original_hw: tuple[int, int],
    *,
    legacy_intermediate_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold on the declared supervision grid, then resize only for evaluation."""
    if score.ndim != 3 or int(score.shape[0]) != 1:
        raise ValueError(f"score must be [1,H,W], got {list(score.shape)}")
    native = score.float()
    if legacy_intermediate_size is not None:
        native = F.interpolate(
            native.unsqueeze(0),
            size=(int(legacy_intermediate_size), int(legacy_intermediate_size)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    hard_native = native > float(threshold)
    hard_original = F.interpolate(
        hard_native.float().unsqueeze(0),
        size=tuple(map(int, original_hw)),
        mode="nearest",
    ).squeeze(0)
    return hard_native.float().contiguous(), hard_original.float().contiguous()
