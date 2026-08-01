"""Pure, fixed formulas for the first DABE rank/calibration experiment."""

from __future__ import annotations

import torch


RANK_CALIBRATION_VERSION = "dabe_rankcal_v1"
RANK_CALIBRATION_METHODS = (
    "C0-F-MinMax",
    "C1-FRank-R1Dist",
    "C2-MedianRank-R1Dist",
)
RANK_CALIBRATION_EPS = 1e-8


def _as_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value.detach().to(dtype=torch.float32)


def minmax_per_image(value: torch.Tensor, eps: float = RANK_CALIBRATION_EPS) -> torch.Tensor:
    """Per-tensor min-max score; a constant tensor maps to all zeros."""
    value = _as_float_tensor(value, "value")
    minimum = value.min()
    value_range = value.max() - minimum
    if float(value_range) < float(eps):
        return torch.zeros_like(value)
    return ((value - minimum) / (value_range + float(eps))).clamp(0.0, 1.0)


def average_percentile_rank(value: torch.Tensor) -> torch.Tensor:
    """Average percentile rank with stable sorting and exact tie groups.

    For a non-constant tensor, the smallest and largest ranks are 0 and 1.
    A tie occupying sorted zero-based positions [l, r] receives
    ``(l + r) / (2 * (N - 1))``. A constant tensor receives rank 0.5.
    """
    value = _as_float_tensor(value, "value")
    flat = value.reshape(-1)
    count = flat.numel()
    if count == 1 or bool(flat.max() == flat.min()):
        return torch.full_like(value, 0.5)

    order = torch.argsort(flat, stable=True)
    sorted_value = flat[order]
    _, group_counts = torch.unique_consecutive(sorted_value, return_counts=True)
    group_ends = torch.cumsum(group_counts, dim=0) - 1
    group_starts = group_ends - group_counts + 1
    group_ranks = (group_starts + group_ends).to(torch.float32) / (2.0 * (count - 1))
    sorted_ranks = torch.repeat_interleave(group_ranks, group_counts)
    ranks = torch.empty_like(flat)
    ranks[order] = sorted_ranks
    return ranks.reshape_as(value).clamp(0.0, 1.0)


def empirical_quantile(reference: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Linear empirical quantiles at ``u * (N - 1)``, with ``u`` clamped."""
    reference = _as_float_tensor(reference, "reference")
    quantiles = _as_float_tensor(quantiles, "quantiles")
    sorted_reference = torch.sort(reference.reshape(-1), stable=True).values
    if sorted_reference.numel() == 1:
        return torch.full_like(quantiles, sorted_reference[0])

    position = quantiles.clamp(0.0, 1.0) * (sorted_reference.numel() - 1)
    lower = torch.floor(position).long()
    upper = torch.ceil(position).long()
    weight = position - lower.to(position.dtype)
    return sorted_reference[lower] * (1.0 - weight) + sorted_reference[upper] * weight


def rank_transport(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Give ``source`` ordering the empirical value distribution of ``reference``."""
    source = _as_float_tensor(source, "source")
    reference = _as_float_tensor(reference, "reference")
    if source.shape != reference.shape:
        raise ValueError(
            f"source/reference shape mismatch: {tuple(source.shape)} vs {tuple(reference.shape)}"
        )
    if bool(reference.max() == reference.min()) or bool(source.max() == source.min()):
        return reference.clone()
    return empirical_quantile(reference, average_percentile_rank(source)).reshape_as(source)


def build_f_minmax(fg_score_37: torch.Tensor) -> torch.Tensor:
    return minmax_per_image(fg_score_37).detach().float().contiguous().clamp(0.0, 1.0)


def build_f_rank_to_r1(
    fg_score_37: torch.Tensor,
    residual_pass1_37: torch.Tensor,
) -> torch.Tensor:
    return (
        rank_transport(fg_score_37, residual_pass1_37)
        .detach()
        .float()
        .contiguous()
        .clamp(0.0, 1.0)
    )


def build_median_rank_to_r1(
    residual_pass1_37: torch.Tensor,
    residual_37: torch.Tensor,
    fg_score_37: torch.Tensor,
) -> torch.Tensor:
    r1 = _as_float_tensor(residual_pass1_37, "residual_pass1_37")
    residual = _as_float_tensor(residual_37, "residual_37")
    foreground = _as_float_tensor(fg_score_37, "fg_score_37")
    if r1.shape != residual.shape or r1.shape != foreground.shape:
        raise ValueError(
            "residual_pass1_37, residual_37 and fg_score_37 must have identical shapes"
        )

    rank_stack = torch.stack(
        (
            average_percentile_rank(r1),
            average_percentile_rank(residual),
            average_percentile_rank(foreground),
        ),
        dim=0,
    )
    median_rank = torch.median(rank_stack, dim=0).values
    return (
        r1.clone()
        if bool(median_rank.max() == median_rank.min())
        else empirical_quantile(r1, median_rank).reshape_as(r1)
    ).detach().float().contiguous().clamp(0.0, 1.0)


def build_rank_calibration_candidates(
    residual_pass1_37: torch.Tensor,
    residual_37: torch.Tensor,
    fg_score_37: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the three frozen rank/calibration candidates on the native grid."""
    c0 = build_f_minmax(fg_score_37)
    c1 = build_f_rank_to_r1(fg_score_37, residual_pass1_37)
    c2 = build_median_rank_to_r1(
        residual_pass1_37,
        residual_37,
        fg_score_37,
    )

    return {
        "c0_f_minmax_37": c0.detach().float().contiguous().clamp(0.0, 1.0),
        "c1_f_rank_r1_37": c1.detach().float().contiguous().clamp(0.0, 1.0),
        "c2_median_rank_r1_37": c2.detach().float().contiguous().clamp(0.0, 1.0),
    }
