"""Frozen formulas for DABE-TF cross-validated boundary reliability (CVBR-v1)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as torch_f

from common.dabe_background_null import (
    BackgroundReconstructionResult,
    reconstruct_from_background_atoms,
)
from common.dabe_pseudo import (
    EPS,
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _border_mask,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
    _validate_feature,
)


CVBR_VERSION = "dabe_cvbr_v1"
CVBR_EXCLUSION_RADIUS = 1
CVBR_MAD_SCALE = 1.4826
CVBR_SCALE_EPS = 1e-6
CVBR_Q_MIN = 1e-6

PROBABILITY_FIELDS = (
    "b0_r1_bw2_37", "b1_r1_bw1_37",
    "v1_cvbr_second_ring_37", "v2_cvbr_all_border_37",
    "bc_b0_37", "bc_b1_37", "bc_v1_37", "bc_v2_37",
    "anchor_b0_37", "anchor_b1_37", "anchor_v1_37", "anchor_v2_37",
    "border_ring1_37", "border_ring2_only_37", "border_ring2_full_37",
    "source_q_v1_37", "source_q_v2_37",
)
NONNEGATIVE_FIELDS = (
    "boundary_cv_error_37", "boundary_support_norm_37",
    "boundary_weight_entropy_37", "boundary_color_dispersion_37",
    "raw_b0_37", "raw_b1_37", "raw_v1_37", "raw_v2_37",
)


@dataclass(frozen=True)
class BoundaryCrossDetails:
    reconstruction: BackgroundReconstructionResult
    boundary_cv_error: torch.Tensor
    boundary_support_norm: torch.Tensor
    boundary_weight_entropy: torch.Tensor
    boundary_color_dispersion: torch.Tensor
    fallback_mask: torch.Tensor


def border_ring_masks(grid: int = 37) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened B1, R2-only and B2 masks."""
    grid = int(grid)
    if grid < 5:
        raise ValueError("grid must contain at least two border rings")
    ring1 = _border_mask(grid, 1).bool().contiguous()
    ring2_full = _border_mask(grid, 2).bool().contiguous()
    ring2_only = ring2_full & ~ring1
    if bool((ring1 & ring2_only).any()) or not torch.equal(ring1 | ring2_only, ring2_full):
        raise RuntimeError("invalid border-ring partition")
    return ring1, ring2_only, ring2_full


def background_connectivity_with_source_reliability(
    neigh_idx: torch.Tensor,
    neigh_weight: torch.Tensor,
    source_mask: torch.Tensor,
    source_reliability: torch.Tensor,
    grid: int,
    tau_bc: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Current multi-source Dijkstra with source initialization ``-log(q)``."""
    grid, tau_bc, eps = int(grid), float(tau_bc), float(eps)
    count = grid * grid
    if tuple(neigh_idx.shape) != (count, 8) or tuple(neigh_weight.shape) != (count, 8):
        raise ValueError("neighbor tensors must be [grid^2,8]")
    if source_mask.numel() != count or source_reliability.numel() != count:
        raise ValueError("source tensors must contain grid^2 values")
    if neigh_idx.device.type != "cpu" or neigh_weight.device.type != "cpu":
        raise ValueError("weighted Dijkstra requires CPU tensors")
    if tau_bc <= 0 or eps <= 0:
        raise ValueError("tau_bc and eps must be positive")
    source = source_mask.detach().cpu().bool().reshape(-1)
    reliability = source_reliability.detach().cpu().float().reshape(-1)
    if not bool(source.any()):
        raise ValueError("at least one source is required")
    source_q = reliability[source]
    # Compare against the representable float32 form of q_min.  A tensor
    # clamped with ``1e-6`` stores 9.999999974752427e-7; comparing that value
    # with the Python float literal would incorrectly reject the frozen lower
    # bound itself.
    q_min_float32 = float(torch.tensor(CVBR_Q_MIN, dtype=source_q.dtype))
    if not torch.isfinite(source_q).all() or float(source_q.min()) < q_min_float32 or float(source_q.max()) > 1.0:
        raise ValueError("source reliability must be finite in [CVBR_Q_MIN,1]")
    if not torch.isfinite(neigh_weight).all() or float(neigh_weight.min()) < 0:
        raise ValueError("neighbor weights must be finite and nonnegative")

    distance = np.full(count, np.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for node in torch.where(source)[0].tolist():
        initial = -np.log(float(reliability[node]))
        distance[node] = initial
        heapq.heappush(heap, (initial, int(node)))
    idx_np = neigh_idx.detach().cpu().numpy()
    weight_np = neigh_weight.detach().cpu().numpy()
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distance[node]:
            continue
        for slot in range(weight_np.shape[1]):
            weight = float(weight_np[node, slot])
            if weight <= 0.0:
                continue
            nxt = int(idx_np[node, slot])
            candidate = dist - np.log(weight + eps)
            if candidate < distance[nxt]:
                distance[nxt] = candidate
                heapq.heappush(heap, (candidate, nxt))
    finite = np.isfinite(distance)
    if not finite.all():
        fill = float(np.max(distance[finite])) if finite.any() else 0.0
        distance[~finite] = fill
    max_distance = float(np.max(distance))
    normalized = distance / (max_distance + eps) if max_distance > eps else distance
    bc = np.exp(-normalized / tau_bc)
    return torch.from_numpy(bc.astype(np.float32)).clamp(0.0, 1.0).contiguous()


def cvbr_reliability(
    boundary_error: torch.Tensor,
    reference_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute frozen median/MAD reliability from B1 raw cross residuals."""
    error = boundary_error.detach().cpu().float().reshape(-1)
    reference = reference_mask.detach().cpu().bool().reshape(-1)
    if error.shape != reference.shape or not bool(reference.any()):
        raise ValueError("boundary error/reference mask mismatch")
    if not torch.isfinite(error).all() or float(error.min()) < 0:
        raise ValueError("boundary error must be finite and nonnegative")
    values = error[reference]
    median = torch.median(values)
    mad = torch.median(torch.abs(values - median))
    scale = max(CVBR_MAD_SCALE * float(mad), CVBR_SCALE_EPS)
    positive_z = ((error - median) / scale).clamp_min(0.0)
    q = torch.exp(-positive_z).clamp(CVBR_Q_MIN, 1.0).float().contiguous()
    return q, {
        "reference_median": float(median),
        "reference_mad": float(mad),
        "reference_scale": float(scale),
    }


def cross_reconstruct_boundary(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    reference_anchor: torch.Tensor,
    boundary_mask: torch.Tensor,
    params: dict,
) -> BoundaryCrossDetails:
    """Cross-reconstruct B2 from B1 anchors with fixed Chebyshev-radius exclusion."""
    boundary = boundary_mask.detach().cpu().bool().reshape(-1)
    reconstruction = reconstruct_from_background_atoms(
        feat_n, rgb_n, reference_anchor, params,
        exclude_chebyshev_radius=CVBR_EXCLUSION_RADIUS,
    )
    weights = reconstruction.topk_weight
    global_index = reconstruction.topk_anchor_global_index
    feature_top = feat_n.detach().cpu().float()[global_index]
    rgb_top = rgb_n.detach().cpu().float()[global_index]
    feature_raw = (weights.unsqueeze(-1) * feature_top).sum(dim=1)
    support_norm = torch.linalg.norm(feature_raw, dim=1)
    effective_k = (weights > 0).sum(dim=1)
    entropy_numerator = -(weights * torch.log(weights + EPS)).sum(dim=1)
    entropy = torch.where(
        effective_k > 1,
        entropy_numerator / torch.log(effective_k.clamp_min(2).float()),
        torch.zeros_like(entropy_numerator),
    )
    rgb_hat = reconstruction.reconstructed_rgb
    color_dispersion = (
        weights * torch.linalg.norm(rgb_top - rgb_hat[:, None, :], dim=-1)
    ).sum(dim=1)
    def boundary_only(value: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(value, dtype=torch.float32)
        output[boundary] = value.float()[boundary]
        return output.contiguous()
    return BoundaryCrossDetails(
        reconstruction=reconstruction,
        boundary_cv_error=boundary_only(reconstruction.raw_residual),
        boundary_support_norm=boundary_only(support_norm),
        boundary_weight_entropy=boundary_only(entropy),
        boundary_color_dispersion=boundary_only(color_dispersion),
        fallback_mask=(reconstruction.fallback_mask & boundary).contiguous(),
    )


def _map(value: torch.Tensor, grid: int) -> torch.Tensor:
    return value.detach().cpu().float().reshape(1, grid, grid).contiguous()


def _stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    value = value.detach().cpu().float().reshape(-1)
    return {
        f"{prefix}_min": float(value.min()),
        f"{prefix}_mean": float(value.mean()),
        f"{prefix}_median": float(torch.median(value)),
        f"{prefix}_max": float(value.max()),
        f"{prefix}_std": float(value.std(unbiased=False)),
    }


def _overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    left, right = left.bool(), right.bool()
    return float((left & right).sum() / ((left | right).sum().float() + EPS))


def build_cvbr_candidates(
    *,
    feature_37: torch.Tensor,
    image_path: str,
    cached_r1_37=None,
    effective_params: dict,
    include_v2: bool = True,
) -> dict:
    """Build frozen CVBR candidates without reading GT.

    ``include_v2=False`` is the lean training-target path: it stops after B0,
    B1 and V1 and never constructs the all-border V2 candidate.
    """
    grid = int(effective_params["GRID"])
    if grid != 37:
        raise ValueError("CVBR-v1 requires GRID=37")
    feature = _validate_feature(feature_37, grid)
    cached = None
    if cached_r1_37 is not None:
        cached = cached_r1_37.detach().cpu().float().contiguous()
        if tuple(cached.shape) != (1, grid, grid) or not torch.isfinite(cached).all():
            raise ValueError("cached_r1_37 must be finite Tensor[1,37,37]")
    rgb_chw = _load_rgb_grid(image_path, grid)
    # Keep the same channel-major view/stride used by current
    # ``_run_single_view_v2``.  Materializing these matrices as contiguous can
    # change GEMM rounding enough to alter a top-k reconstruction tie.
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(grid * grid, 3).float()
    feat_n = torch_f.normalize(
        feature.permute(1, 2, 0).reshape(grid * grid, 384), dim=1, p=2
    )
    edge = _sobel_magnitude(rgb_chw).reshape(-1)
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge, grid, effective_params)
    ring1, ring2_only, ring2_full = border_ring_masks(grid)

    params_b0, params_b1 = dict(effective_params), dict(effective_params)
    params_b0["BORDER_WIDTH"], params_b1["BORDER_WIDTH"] = 2, 1
    bc_b0, source_b0 = _background_connectivity(neigh_idx, neigh_weight, grid, params_b0)
    if not torch.equal(source_b0, ring2_full):
        raise RuntimeError("current B0 border mask changed")
    anchor_b0 = _background_anchor(bc_b0, source_b0, params_b0)
    # Recompute and verify the frozen B0 baseline before allocating tensors for
    # any experimental branch.  This mirrors the original DABE-v2 operation
    # order and keeps the strict 1e-6 identity check meaningful around top-k
    # ties on CPU.
    current_b0 = _background_residual(feat_n, rgb_n, anchor_b0, params_b0)
    current_error = (
        0.0
        if cached is None
        else float((current_b0.reshape_as(cached) - cached).abs().max())
    )
    if current_error > 1e-6:
        raise RuntimeError(f"B0 baseline mismatch: current={current_error}")
    b0 = reconstruct_from_background_atoms(
        feat_n, rgb_n, anchor_b0, params_b0, exclude_chebyshev_radius=None
    )
    comparison_r1 = (
        current_b0.reshape(1, grid, grid) if cached is None else cached
    )
    detail_error = float(
        (b0.normalized_residual.reshape_as(comparison_r1) - comparison_r1)
        .abs()
        .max()
    )

    bc_b1, source_b1 = _background_connectivity(neigh_idx, neigh_weight, grid, params_b1)
    if not torch.equal(source_b1, ring1):
        raise RuntimeError("current B1 border mask changed")
    anchor_b1 = _background_anchor(bc_b1, source_b1, params_b1)
    b1 = reconstruct_from_background_atoms(
        feat_n, rgb_n, anchor_b1, params_b1, exclude_chebyshev_radius=None
    )
    cross = cross_reconstruct_boundary(feat_n, rgb_n, anchor_b1, ring2_full, params_b1)
    q_all, reference = cvbr_reliability(cross.boundary_cv_error, ring1)
    q_v1 = torch.zeros(grid * grid, dtype=torch.float32)
    q_v1[ring1], q_v1[ring2_only] = 1.0, q_all[ring2_only]
    q_v2 = None
    if include_v2:
        q_v2 = torch.zeros(grid * grid, dtype=torch.float32)
        q_v2[ring2_full] = q_all[ring2_full]

    unit = torch.ones(grid * grid, dtype=torch.float32)
    bc_unit = background_connectivity_with_source_reliability(
        neigh_idx, neigh_weight, ring2_full, unit, grid,
        float(effective_params["TAU_BC"]), EPS,
    )
    weighted_error = float((bc_unit - bc_b0).abs().max())
    if weighted_error > 1e-6:
        raise RuntimeError(f"weighted Dijkstra q=1 mismatch: {weighted_error}")
    bc_v1 = background_connectivity_with_source_reliability(
        neigh_idx, neigh_weight, ring2_full, q_v1, grid,
        float(effective_params["TAU_BC"]), EPS,
    )
    anchor_v1 = _background_anchor(bc_v1, ring2_full, params_b0)
    v1 = reconstruct_from_background_atoms(
        feat_n, rgb_n, anchor_v1, params_b0, exclude_chebyshev_radius=None
    )
    bc_v2 = anchor_v2 = v2 = None
    if include_v2:
        bc_v2 = background_connectivity_with_source_reliability(
            neigh_idx, neigh_weight, ring2_full, q_v2, grid,
            float(effective_params["TAU_BC"]), EPS,
        )
        anchor_v2 = _background_anchor(bc_v2, ring2_full, params_b0)
        v2 = reconstruct_from_background_atoms(
            feat_n, rgb_n, anchor_v2, params_b0, exclude_chebyshev_radius=None
        )

    boundary_error = cross.boundary_cv_error[ring2_full]
    ring1_error, ring2_error = cross.boundary_cv_error[ring1], cross.boundary_cv_error[ring2_only]
    ring2_q, ring1_q = q_all[ring2_only], q_all[ring1]
    diagnostics = {
        "b0_external_cache_checked": cached is not None,
        "b0_cached_r1_max_abs": current_error,
        "b0_detail_vs_cached_max_abs": detail_error,
        "weighted_dijkstra_unit_reliability_max_abs": weighted_error,
        "ring1_count": int(ring1.sum()), "ring2_only_count": int(ring2_only.sum()),
        "ring2_full_count": int(ring2_full.sum()),
        "bw1_anchor_count": int(anchor_b1.sum()), "bw2_anchor_count": int(anchor_b0.sum()),
        "v1_anchor_count": int(anchor_v1.sum()),
        **_stats(boundary_error, "boundary_cv_error"),
        "ring1_cv_error_mean": float(ring1_error.mean()),
        "ring1_cv_error_median": float(torch.median(ring1_error)),
        "ring2_cv_error_mean": float(ring2_error.mean()),
        "ring2_cv_error_median": float(torch.median(ring2_error)),
        **reference,
        **_stats(q_v1[ring2_full], "q_v1"),
        "ring2_q_mean": float(ring2_q.mean()),
        "ring2_q_below_09_ratio": float((ring2_q < .9).float().mean()),
        "ring2_q_below_05_ratio": float((ring2_q < .5).float().mean()),
        "ring2_q_below_01_ratio": float((ring2_q < .1).float().mean()),
        "ring1_q_v2_mean": float(ring1_q.mean()),
        "ring1_q_v2_below_09_ratio": float((ring1_q < .9).float().mean()),
        "ring1_q_v2_below_05_ratio": float((ring1_q < .5).float().mean()),
        "ring1_q_v2_below_01_ratio": float((ring1_q < .1).float().mean()),
        "effective_source_mass_b0": float(ring2_full.sum()),
        "effective_source_mass_b1": float(ring1.sum()),
        "effective_source_mass_v1": float(q_v1[ring2_full].sum()),
        "cross_fallback_count": int(cross.fallback_mask.sum()),
    }
    if include_v2:
        diagnostics.update({
            "v2_anchor_count": int(anchor_v2.sum()),
            **_stats(q_v2[ring2_full], "q_v2"),
            "effective_source_mass_v2": float(q_v2[ring2_full].sum()),
        })
    diagnostics["b0_area_gt_05"] = float((current_b0 > .5).float().mean())
    detail_items = [("b1", b1), ("v1", v1)]
    bc_items = [("b0", bc_b0), ("b1", bc_b1), ("v1", bc_v1)]
    if include_v2:
        detail_items.append(("v2", v2))
        bc_items.append(("v2", bc_v2))
    for name, details in detail_items:
        diagnostics[f"{name}_area_gt_05"] = float((details.normalized_residual > .5).float().mean())
    for name, value in bc_items:
        diagnostics[f"bc_{name}_mean"] = float(value.mean())
    diagnostics.update({
        "anchor_overlap_v1_b0": _overlap(anchor_v1, anchor_b0),
        "anchor_overlap_v1_b1": _overlap(anchor_v1, anchor_b1),
    })
    if include_v2:
        diagnostics.update({
            "anchor_overlap_v2_b0": _overlap(anchor_v2, anchor_b0),
            "anchor_overlap_v2_b1": _overlap(anchor_v2, anchor_b1),
        })
    result = {
        "b0_r1_bw2_37": _map(current_b0, grid),
        "b1_r1_bw1_37": _map(b1.normalized_residual, grid),
        "v1_cvbr_second_ring_37": _map(v1.normalized_residual, grid),
        "bc_b0_37": _map(bc_b0, grid), "bc_b1_37": _map(bc_b1, grid),
        "bc_v1_37": _map(bc_v1, grid),
        "anchor_b0_37": _map(anchor_b0.float(), grid),
        "anchor_b1_37": _map(anchor_b1.float(), grid),
        "anchor_v1_37": _map(anchor_v1.float(), grid),
        "border_ring1_37": _map(ring1.float(), grid),
        "border_ring2_only_37": _map(ring2_only.float(), grid),
        "border_ring2_full_37": _map(ring2_full.float(), grid),
        "boundary_cv_error_37": _map(cross.boundary_cv_error, grid),
        "boundary_support_norm_37": _map(cross.boundary_support_norm, grid),
        "boundary_weight_entropy_37": _map(cross.boundary_weight_entropy, grid),
        "boundary_color_dispersion_37": _map(cross.boundary_color_dispersion, grid),
        "source_q_v1_37": _map(q_v1, grid),
        "raw_b0_37": _map(b0.raw_residual, grid), "raw_b1_37": _map(b1.raw_residual, grid),
        "raw_v1_37": _map(v1.raw_residual, grid),
        "diagnostics": diagnostics,
    }
    if include_v2:
        result.update({
            "v2_cvbr_all_border_37": _map(v2.normalized_residual, grid),
            "bc_v2_37": _map(bc_v2, grid),
            "anchor_v2_37": _map(anchor_v2.float(), grid),
            "source_q_v2_37": _map(q_v2, grid),
            "raw_v2_37": _map(v2.raw_residual, grid),
        })
    return result
