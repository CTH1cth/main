"""Pure formulas for DABE-TF background cross-reconstruction and local nulls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common.dabe_pseudo import (
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _build_local_graph,
    _minmax,
    _sobel_magnitude,
    _validate_feature,
)
from common.dabe_rank_calibration import average_percentile_rank, rank_transport


BGNULL_VERSION = "dabe_bgnull_v1"
CROSS_EXCLUSION_RADIUS = 1
EPS = 1e-8
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class BackgroundReconstructionResult:
    raw_residual: torch.Tensor
    normalized_residual: torch.Tensor
    topk_anchor_local_index: torch.Tensor
    topk_anchor_global_index: torch.Tensor
    topk_score: torch.Tensor
    topk_weight: torch.Tensor
    reconstructed_feature: torch.Tensor
    reconstructed_rgb: torch.Tensor
    valid_anchor_count: torch.Tensor
    fallback_mask: torch.Tensor
    local_excluded_weight_mass: torch.Tensor | None


def _validate_flat_inputs(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor_mask: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not all(torch.is_tensor(value) for value in (feat_n, rgb_n, anchor_mask)):
        raise TypeError("feat_n, rgb_n and anchor_mask must be tensors")
    if feat_n.device.type != "cpu" or rgb_n.device.type != "cpu" or anchor_mask.device.type != "cpu":
        raise ValueError("background reconstruction must run on CPU tensors")
    if feat_n.ndim != 2 or rgb_n.ndim != 2 or rgb_n.shape[1] != 3:
        raise ValueError("feat_n must be [N,D] and rgb_n must be [N,3]")
    if feat_n.shape[0] != rgb_n.shape[0] or anchor_mask.numel() != feat_n.shape[0]:
        raise ValueError("feature, RGB and anchor sizes do not match")
    if not torch.isfinite(feat_n).all() or not torch.isfinite(rgb_n).all():
        raise ValueError("feature/RGB contains NaN or Inf")
    grid = int(params["GRID"])
    if feat_n.shape[0] != grid * grid:
        raise ValueError(f"N must equal GRID^2, got {feat_n.shape[0]} and GRID={grid}")
    anchor_mask = anchor_mask.detach().cpu().bool().reshape(-1)
    if int(anchor_mask.sum()) == 0:
        raise ValueError("background anchor set must not be empty")
    return (
        feat_n.detach().cpu().float().contiguous(),
        rgb_n.detach().cpu().float().contiguous(),
        anchor_mask,
        grid,
    )


def reconstruct_from_background_atoms(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor_mask: torch.Tensor,
    params: dict,
    *,
    exclude_chebyshev_radius: int | None,
    chunk_size: int = 512,
) -> BackgroundReconstructionResult:
    """Reconstruct every query from background atoms, optionally excluding a local square."""
    feat_n, rgb_n, anchor_mask, grid = _validate_flat_inputs(
        feat_n, rgb_n, anchor_mask, params
    )
    if exclude_chebyshev_radius is not None and int(exclude_chebyshev_radius) < 0:
        raise ValueError("exclude_chebyshev_radius must be non-negative or None")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    anchor_global = torch.where(anchor_mask)[0]
    feat_anchor = feat_n.index_select(0, anchor_global)
    rgb_anchor = rgb_n.index_select(0, anchor_global)
    requested_k = min(int(params["K_RECON"]), int(anchor_global.numel()))
    tau = float(params["TAU_RECON"])
    sigma_color = float(params["SIGMA_COLOR_RECON"])
    lambda_color = float(params["LAMBDA_COLOR_RECON"])
    if requested_k <= 0 or tau <= 0 or sigma_color <= 0:
        raise ValueError("invalid reconstruction parameters")

    anchor_y = torch.div(anchor_global, grid, rounding_mode="floor")
    anchor_x = anchor_global.remainder(grid)
    result_parts: dict[str, list[torch.Tensor]] = {
        "local": [],
        "global": [],
        "score": [],
        "weight": [],
        "feature": [],
        "rgb": [],
        "valid": [],
        "fallback": [],
        "excluded_mass": [],
    }

    for start in range(0, feat_n.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), feat_n.shape[0])
        feat_chunk = feat_n[start:end]
        rgb_chunk = rgb_n[start:end]
        sim_feat = feat_chunk @ feat_anchor.t()
        color_dist2 = torch.cdist(rgb_chunk, rgb_anchor, p=2.0).square()
        similarity = sim_feat + lambda_color * torch.exp(-color_dist2 / sigma_color)

        regular_score, regular_local = torch.topk(similarity, k=requested_k, dim=1)
        regular_weight = torch.softmax(regular_score / tau, dim=1)

        if exclude_chebyshev_radius is None:
            chosen_score = regular_score
            chosen_local = regular_local
            chosen_weight = regular_weight
            valid_count = torch.full(
                (end - start,), int(anchor_global.numel()), dtype=torch.long
            )
            fallback = torch.zeros((end - start,), dtype=torch.bool)
            excluded_mass = None
        else:
            radius = int(exclude_chebyshev_radius)
            query_global = torch.arange(start, end, dtype=torch.long)
            query_y = torch.div(query_global, grid, rounding_mode="floor")
            query_x = query_global.remainder(grid)
            excluded = (
                (query_y[:, None] - anchor_y[None, :]).abs().maximum(
                    (query_x[:, None] - anchor_x[None, :]).abs()
                )
                <= radius
            )
            valid_count = (~excluded).sum(dim=1).long()
            regular_excluded = torch.gather(excluded, 1, regular_local)
            excluded_mass = (regular_weight * regular_excluded.float()).sum(dim=1)

            masked_similarity = similarity.masked_fill(excluded, float("-inf"))
            masked_score, masked_local = torch.topk(masked_similarity, k=requested_k, dim=1)
            finite = torch.isfinite(masked_score)
            safe_logits = torch.where(finite, masked_score / tau, torch.full_like(masked_score, -1e30))
            exp_logits = torch.where(finite, torch.exp(safe_logits - safe_logits.max(dim=1, keepdim=True).values), torch.zeros_like(safe_logits))
            denominator = exp_logits.sum(dim=1, keepdim=True)
            fallback = valid_count == 0
            masked_weight = exp_logits / denominator.clamp_min(EPS)

            chosen_local = masked_local.clone()
            chosen_score = torch.where(finite, masked_score, torch.zeros_like(masked_score))
            chosen_weight = torch.where(finite, masked_weight, torch.zeros_like(masked_weight))
            if fallback.any():
                chosen_local[fallback] = regular_local[fallback]
                chosen_score[fallback] = regular_score[fallback]
                chosen_weight[fallback] = regular_weight[fallback]

        chosen_global = anchor_global.index_select(0, chosen_local.reshape(-1)).reshape_as(chosen_local)
        feat_top = feat_anchor[chosen_local]
        rgb_top = rgb_anchor[chosen_local]
        feat_hat = F.normalize((chosen_weight.unsqueeze(-1) * feat_top).sum(dim=1), dim=1, p=2)
        rgb_hat = (chosen_weight.unsqueeze(-1) * rgb_top).sum(dim=1)

        result_parts["local"].append(chosen_local)
        result_parts["global"].append(chosen_global)
        result_parts["score"].append(chosen_score)
        result_parts["weight"].append(chosen_weight)
        result_parts["feature"].append(feat_hat)
        result_parts["rgb"].append(rgb_hat)
        result_parts["valid"].append(valid_count)
        result_parts["fallback"].append(fallback)
        if excluded_mass is not None:
            result_parts["excluded_mass"].append(excluded_mass)

    reconstructed_feature = torch.cat(result_parts["feature"], dim=0)
    reconstructed_rgb = torch.cat(result_parts["rgb"], dim=0)
    feature_residual = (1.0 - (feat_n * reconstructed_feature).sum(dim=1)).clamp_min(0.0)
    color_residual = torch.linalg.norm(rgb_n - reconstructed_rgb, dim=1)
    raw_residual = (feature_residual + 0.2 * color_residual).float()
    normalized_residual = _minmax(raw_residual)
    topk_weight = torch.cat(result_parts["weight"], dim=0).float()

    tensors = [
        raw_residual,
        normalized_residual,
        torch.cat(result_parts["score"], dim=0),
        topk_weight,
        reconstructed_feature,
        reconstructed_rgb,
    ]
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("background reconstruction produced NaN or Inf")
    weight_error = float((topk_weight.sum(dim=1) - 1.0).abs().max())
    if weight_error > 1e-6:
        raise RuntimeError(f"top-k weights do not sum to one: max error={weight_error}")

    return BackgroundReconstructionResult(
        raw_residual=raw_residual.detach().contiguous(),
        normalized_residual=normalized_residual.detach().float().contiguous(),
        topk_anchor_local_index=torch.cat(result_parts["local"], dim=0).detach().contiguous(),
        topk_anchor_global_index=torch.cat(result_parts["global"], dim=0).detach().contiguous(),
        topk_score=torch.cat(result_parts["score"], dim=0).detach().float().contiguous(),
        topk_weight=topk_weight.detach().contiguous(),
        reconstructed_feature=reconstructed_feature.detach().float().contiguous(),
        reconstructed_rgb=reconstructed_rgb.detach().float().contiguous(),
        valid_anchor_count=torch.cat(result_parts["valid"], dim=0).detach().contiguous(),
        fallback_mask=torch.cat(result_parts["fallback"], dim=0).detach().contiguous(),
        local_excluded_weight_mass=(
            torch.cat(result_parts["excluded_mass"], dim=0).detach().float().contiguous()
            if result_parts["excluded_mass"]
            else None
        ),
    )


def make_anchor_cross_error_map(
    cross_raw_residual: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross_raw_residual = cross_raw_residual.detach().cpu().float().reshape(-1)
    anchor_mask = anchor_mask.detach().cpu().bool().reshape(-1)
    if cross_raw_residual.shape != anchor_mask.shape:
        raise ValueError("cross residual and anchor mask shapes differ")
    anchor_indices = torch.where(anchor_mask)[0]
    anchor_errors = cross_raw_residual.index_select(0, anchor_indices)
    error_map = torch.zeros_like(cross_raw_residual)
    error_map[anchor_indices] = anchor_errors
    return anchor_errors.contiguous(), error_map.contiguous()


def weighted_local_null_score(
    query_raw_residual: torch.Tensor,
    selected_anchor_cross_error: torch.Tensor,
    selected_anchor_weight: torch.Tensor,
    *,
    tie_atol: float = TIE_ATOL,
) -> torch.Tensor:
    query = query_raw_residual.detach().cpu().float().reshape(-1)
    errors = selected_anchor_cross_error.detach().cpu().float()
    weights = selected_anchor_weight.detach().cpu().float()
    if errors.ndim != 2 or weights.shape != errors.shape or errors.shape[0] != query.numel():
        raise ValueError("local-null inputs must be query [N] and errors/weights [N,K]")
    if not torch.isfinite(query).all() or not torch.isfinite(errors).all() or not torch.isfinite(weights).all():
        raise ValueError("local-null input contains NaN or Inf")
    if float(weights.min()) < 0.0:
        raise ValueError("local-null weights must be non-negative")
    weight_error = float((weights.sum(dim=1) - 1.0).abs().max())
    if weight_error > 1e-6:
        raise ValueError(f"local-null weights must sum to one, max error={weight_error}")
    difference = errors - query[:, None]
    strict_less = errors < query[:, None] - float(tie_atol)
    ties = difference.abs() <= float(tie_atol)
    comparison = strict_less.float() + 0.5 * ties.float()
    return (weights * comparison).sum(dim=1).clamp(0.0, 1.0).float().contiguous()


def retrieval_control_from_regular_top1(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    regular_details: BackgroundReconstructionResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_n = feat_n.detach().cpu().float()
    rgb_n = rgb_n.detach().cpu().float()
    top1_global = regular_details.topk_anchor_global_index[:, 0]
    nearest_feature = feat_n.index_select(0, top1_global)
    nearest_rgb = rgb_n.index_select(0, top1_global)
    raw = (
        (1.0 - (feat_n * nearest_feature).sum(dim=1)).clamp_min(0.0)
        + 0.2 * torch.linalg.norm(rgb_n - nearest_rgb, dim=1)
    ).float()
    return raw.contiguous(), _minmax(raw).float().contiguous()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().cpu().double().reshape(-1)
    right = right.detach().cpu().double().reshape(-1)
    if left.numel() != right.numel() or left.numel() == 0:
        raise ValueError("correlation inputs must have equal nonzero size")
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.norm(left) * torch.linalg.norm(right)
    if float(denominator) <= EPS:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    return _pearson(average_percentile_rank(left), average_percentile_rank(right))


def _stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    value = value.detach().cpu().float().reshape(-1)
    return {
        f"{prefix}_min": float(value.min()),
        f"{prefix}_mean": float(value.mean()),
        f"{prefix}_max": float(value.max()),
        f"{prefix}_std": float(value.std(unbiased=False)),
    }


def build_background_null_candidates(
    *,
    feature_37: torch.Tensor,
    rgb_37: torch.Tensor,
    cached_r1_37: torch.Tensor,
    effective_params: dict,
) -> dict[str, torch.Tensor | dict]:
    """Build all frozen BGNull-v1 candidates on the native 37x37 grid."""
    grid = int(effective_params["GRID"])
    if grid != 37:
        raise ValueError(f"BGNull-v1 requires GRID=37, got {grid}")
    feature = _validate_feature(feature_37, grid)
    if not torch.is_tensor(rgb_37) or tuple(rgb_37.shape) != (3, grid, grid):
        raise ValueError("rgb_37 must be Tensor[3,37,37]")
    rgb = rgb_37.detach().cpu().float().contiguous()
    cached_r1 = cached_r1_37.detach().cpu().float().contiguous()
    if tuple(cached_r1.shape) != (1, grid, grid):
        raise ValueError("cached_r1_37 must be Tensor[1,37,37]")
    if not torch.isfinite(rgb).all() or not torch.isfinite(cached_r1).all():
        raise ValueError("RGB or cached R1 contains NaN/Inf")

    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = _sobel_magnitude(rgb).reshape(-1).float()
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, grid, effective_params)
    bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, effective_params)
    anchor_mask = _background_anchor(bc, border, effective_params)

    original_r1 = _background_residual(feat_n, rgb_n, anchor_mask, effective_params)
    regular = reconstruct_from_background_atoms(
        feat_n,
        rgb_n,
        anchor_mask,
        effective_params,
        exclude_chebyshev_radius=None,
    )
    cross = reconstruct_from_background_atoms(
        feat_n,
        rgb_n,
        anchor_mask,
        effective_params,
        exclude_chebyshev_radius=CROSS_EXCLUSION_RADIUS,
    )
    cached_flat = cached_r1.reshape(-1)
    r1_recompute_error = float((original_r1 - cached_flat).abs().max())
    regular_cache_error = float((regular.normalized_residual - cached_flat).abs().max())
    regular_original_error = float(
        (regular.normalized_residual - original_r1).abs().max()
    )
    if max(r1_recompute_error, regular_cache_error, regular_original_error) > 1e-6:
        raise RuntimeError(
            "R1 reconstruction mismatch: "
            f"original_vs_cache={r1_recompute_error}, "
            f"details_vs_cache={regular_cache_error}, "
            f"details_vs_original={regular_original_error}"
        )

    anchor_errors, anchor_error_map = make_anchor_cross_error_map(
        cross.raw_residual, anchor_mask
    )
    selected_anchor_errors = anchor_errors[regular.topk_anchor_local_index]
    local_null = weighted_local_null_score(
        regular.raw_residual,
        selected_anchor_errors,
        regular.topk_weight,
    )
    local_null_37 = local_null.reshape(1, grid, grid)
    n2 = rank_transport(local_null_37, cached_r1)
    rc_raw, rc_minmax = retrieval_control_from_regular_top1(feat_n, rgb_n, regular)
    excluded_mass = cross.local_excluded_weight_mass
    if excluded_mass is None:
        raise RuntimeError("cross reconstruction did not return excluded weight mass")

    anchor_indices = torch.where(anchor_mask)[0]
    top1_self = regular.topk_anchor_global_index[:, 0] == torch.arange(grid * grid)
    topk_self = (
        regular.topk_anchor_global_index
        == torch.arange(grid * grid)[:, None]
    ).any(dim=1)
    anchor_count = int(anchor_indices.numel())
    local_unique_ratio = float(torch.unique(local_null).numel()) / float(local_null.numel())
    n0_area = float((cached_flat > 0.5).float().mean())
    n1_area = float((cross.normalized_residual > 0.5).float().mean())
    n2_area = float((n2.reshape(-1) > 0.5).float().mean())
    rc_area = float((rc_minmax > 0.5).float().mean())
    diagnostics = {
        "anchor_count": anchor_count,
        "anchor_ratio": anchor_count / float(grid * grid),
        "r1_recompute_max_abs": r1_recompute_error,
        "regular_detail_vs_cache_r1_max_abs": regular_cache_error,
        "regular_detail_vs_original_r1_max_abs": regular_original_error,
        **_stats(regular.raw_residual, "regular_raw"),
        **_stats(cross.raw_residual, "cross_raw"),
        "anchor_regular_raw_mean": float(regular.raw_residual[anchor_indices].mean()),
        "anchor_cross_raw_mean": float(anchor_errors.mean()),
        "anchor_cross_raw_median": float(anchor_errors.median()),
        "anchor_cross_raw_max": float(anchor_errors.max()),
        "cross_minus_regular_mean": float((cross.raw_residual - regular.raw_residual).mean()),
        "cross_minus_regular_anchor_mean": float(
            (cross.raw_residual[anchor_indices] - regular.raw_residual[anchor_indices]).mean()
        ),
        "cross_fallback_count": int(cross.fallback_mask.sum()),
        "cross_fallback_ratio": float(cross.fallback_mask.float().mean()),
        "regular_top1_self_count": int(top1_self.sum()),
        "regular_top1_self_ratio_on_anchor": float(top1_self[anchor_indices].float().mean()),
        "anchor_top1_self_ratio": float(top1_self[anchor_indices].float().mean()),
        "regular_topk_contains_self_count": int(topk_self.sum()),
        "regular_topk_contains_self_ratio_on_anchor": float(topk_self[anchor_indices].float().mean()),
        "local_excluded_weight_mass_mean": float(excluded_mass.mean()),
        "local_excluded_weight_mass_anchor_mean": float(excluded_mass[anchor_indices].mean()),
        "local_excluded_weight_mass_max": float(excluded_mass.max()),
        **_stats(local_null, "local_null"),
        "local_null_unique_ratio": local_unique_ratio,
        "local_null_tie_ratio": 1.0 - local_unique_ratio,
        "n0_area_gt_05": n0_area,
        "n1_area_gt_05": n1_area,
        "n2_area_gt_05": n2_area,
        "rc_area_gt_05": rc_area,
        "n1_area_delta_vs_n0": n1_area - n0_area,
        "n2_area_delta_vs_n0": n2_area - n0_area,
        "rc_area_delta_vs_n0": rc_area - n0_area,
        "n2_sorted_l1_vs_n0": float(
            (
                torch.sort(n2.reshape(-1)).values
                - torch.sort(cached_flat).values
            ).abs().mean()
        ),
        "n2_constant_source_fallback": bool(local_null.max() == local_null.min()),
        "r1_localnull_spearman": _spearman(cached_flat, local_null),
        "r1_crossr1_spearman": _spearman(cached_flat, cross.normalized_residual),
        "r1_nn_spearman": _spearman(cached_flat, rc_minmax),
        "bg_anchor_vs_cached_max_abs": None,
    }

    def map37(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().float().reshape(1, grid, grid).contiguous()

    return {
        "n0_r1_37": map37(cached_flat).clamp(0.0, 1.0),
        "regular_raw_residual_37": map37(regular.raw_residual),
        "n1_cross_raw_residual_37": map37(cross.raw_residual),
        "n1_cross_r1_37": map37(cross.normalized_residual).clamp(0.0, 1.0),
        "anchor_cross_error_37": map37(anchor_error_map),
        "bg_anchor_37": map37(anchor_mask.float()),
        "n2_local_null_raw_37": map37(local_null).clamp(0.0, 1.0),
        "n2_local_null_r1dist_37": map37(n2).clamp(0.0, 1.0),
        "rc_nn_raw_37": map37(rc_raw),
        "rc_nn_minmax_37": map37(rc_minmax).clamp(0.0, 1.0),
        "local_excluded_weight_mass_37": map37(excluded_mass).clamp(0.0, 1.0),
        "diagnostics": diagnostics,
    }
