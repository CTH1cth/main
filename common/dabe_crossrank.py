"""Pure frozen formula and diagnostics for DABE-TF CRMC-v1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_f

from common.dabe_rank_calibration import average_percentile_rank, rank_transport


DABE_CROSSRANK_VERSION = "dabe_crossrank_v1"
EXPECTED_SHAPE = (1, 37, 37)
HARD_THRESHOLD = 0.5


def _probability_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{name} must have shape [1,37,37], got {list(value.shape)}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must have floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{name} must be in [0,1], got min={minimum}, max={maximum}")
    return value.detach().cpu().to(torch.float32).contiguous()


def build_crossrank_r1dist(
    r1_37: torch.Tensor,
    cross_r1_37: torch.Tensor,
) -> torch.Tensor:
    """Give CrossR1 ordering the per-image empirical distribution of R1."""
    r1 = _probability_tensor(r1_37, "r1_37")
    cross = _probability_tensor(cross_r1_37, "cross_r1_37")
    if r1.shape != cross.shape:
        raise ValueError("r1_37 and cross_r1_37 must have identical shapes")
    return (
        rank_transport(source=cross, reference=r1)
        .detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .clamp(0.0, 1.0)
    )


def _stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float(value.min()),
        f"{prefix}_mean": float(value.mean()),
        f"{prefix}_max": float(value.max()),
        f"{prefix}_std": float(value.std(unbiased=False)),
    }


def _unique_and_tie_ratio(value: torch.Tensor) -> tuple[float, float]:
    unique_ratio = float(torch.unique(value.reshape(-1)).numel()) / float(value.numel())
    return unique_ratio, 1.0 - unique_ratio


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = average_percentile_rank(left).reshape(-1).double()
    right_rank = average_percentile_rank(right).reshape(-1).double()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(torch.linalg.norm(left_rank) * torch.linalg.norm(right_rank))
    if denominator == 0.0:
        return float("nan")
    return float(torch.dot(left_rank, right_rank) / denominator)


def _monotonic_violation_count(source: torch.Tensor, output: torch.Tensor) -> int:
    source_flat = source.reshape(-1)
    output_flat = output.reshape(-1)
    order = torch.argsort(source_flat, stable=True)
    source_sorted = source_flat[order]
    output_sorted = output_flat[order]
    strictly_increasing = source_sorted[1:] > source_sorted[:-1]
    decreasing_output = output_sorted[1:] < output_sorted[:-1] - 1e-7
    return int((strictly_increasing & decreasing_output).sum())


def _area(value: torch.Tensor) -> float:
    return float((value > HARD_THRESHOLD).float().mean())


def _to_68(value: torch.Tensor) -> torch.Tensor:
    return torch_f.interpolate(
        value.unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def crossrank_diagnostics(
    r1_37: torch.Tensor,
    cross_r1_37: torch.Tensor,
    crossrank_37: torch.Tensor,
) -> dict:
    """Return fixed GT-free CRMC generation diagnostics."""
    r1 = _probability_tensor(r1_37, "r1_37")
    cross = _probability_tensor(cross_r1_37, "cross_r1_37")
    crossrank = _probability_tensor(crossrank_37, "crossrank_37")
    if r1.shape != cross.shape or r1.shape != crossrank.shape:
        raise ValueError("r1_37, cross_r1_37 and crossrank_37 must have identical shapes")

    cross_unique, cross_tie = _unique_and_tie_ratio(cross)
    crossrank_unique, crossrank_tie = _unique_and_tie_ratio(crossrank)
    r1_sorted = torch.sort(r1.reshape(-1), stable=True).values
    crossrank_sorted = torch.sort(crossrank.reshape(-1), stable=True).values
    sorted_difference = torch.abs(crossrank_sorted - r1_sorted)
    constant_fallback = bool(cross.max() == cross.min()) or bool(r1.max() == r1.min())

    r1_68 = _to_68(r1)
    cross_68 = _to_68(cross)
    crossrank_68 = _to_68(crossrank)
    r1_area_37 = _area(r1)
    cross_area_37 = _area(cross)
    crossrank_area_37 = _area(crossrank)
    r1_area_68 = _area(r1_68)
    cross_area_68 = _area(cross_68)
    crossrank_area_68 = _area(crossrank_68)

    diagnostics = {
        **_stats(r1, "r1"),
        **_stats(cross, "cross"),
        **_stats(crossrank, "crossrank"),
        "cross_unique_ratio": cross_unique,
        "cross_tie_ratio": cross_tie,
        "crossrank_unique_ratio": crossrank_unique,
        "crossrank_tie_ratio": crossrank_tie,
        "crossrank_sorted_l1_vs_r1": float(sorted_difference.mean()),
        "crossrank_sorted_max_abs_vs_r1": float(sorted_difference.max()),
        "crossrank_sorted_invariant_violation": bool(
            not constant_fallback
            and cross_tie == 0.0
            and float(sorted_difference.max()) > 1e-6
        ),
        "crossrank_spearman_vs_cross": _spearman(crossrank, cross),
        "crossrank_spearman_vs_r1": _spearman(crossrank, r1),
        "crossrank_monotonic_violation_count": _monotonic_violation_count(
            cross, crossrank
        ),
        "crossrank_constant_source_fallback": constant_fallback,
        "r1_hard_area_37": r1_area_37,
        "cross_hard_area_37": cross_area_37,
        "crossrank_hard_area_37": crossrank_area_37,
        "cross_hard_area_delta_vs_r1_37": cross_area_37 - r1_area_37,
        "crossrank_hard_area_delta_vs_r1_37": crossrank_area_37 - r1_area_37,
        "r1_hard_area_68": r1_area_68,
        "cross_hard_area_68": cross_area_68,
        "crossrank_hard_area_68": crossrank_area_68,
        "crossrank_hard_area_delta_vs_r1_68": crossrank_area_68 - r1_area_68,
    }
    for key, value in diagnostics.items():
        if isinstance(value, float) and not math.isfinite(value):
            if key not in {"crossrank_spearman_vs_cross", "crossrank_spearman_vs_r1"}:
                raise RuntimeError(f"Unexpected non-finite diagnostic: {key}")
    return diagnostics
