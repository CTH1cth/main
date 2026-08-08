"""GT-free calibration primitives for single-global-PCA GBSP residuals.

Only the stage-A area-transfer diagnostic is exposed initially.  Statistical
calibrators are added after the stage-A continuation gate is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AreaTransferResult:
    binary_mask: torch.Tensor
    source_foreground_count: int
    selected_foreground_count: int
    source_area_ratio: float
    selected_area_ratio: float
    cutoff_score: float
    boundary_tie_count: int


def exact_area_transfer_mask(
    foreground_score: torch.Tensor,
    source_binary_mask: torch.Tensor,
) -> AreaTransferResult:
    """Transfer the source foreground count onto ``foreground_score`` ranks.

    Selection uses a stable descending order and therefore returns exactly the
    source number of foreground patches, including when scores are tied.  No
    ground truth is consumed.
    """

    if not torch.is_tensor(foreground_score) or not torch.is_tensor(
        source_binary_mask
    ):
        raise TypeError("foreground_score/source_binary_mask must be tensors")
    if foreground_score.shape != source_binary_mask.shape:
        raise ValueError(
            "foreground_score/source_binary_mask shape mismatch: "
            f"{tuple(foreground_score.shape)} != {tuple(source_binary_mask.shape)}"
        )
    score = foreground_score.detach().cpu().float().contiguous()
    source = source_binary_mask.detach().cpu().bool().contiguous()
    if score.numel() == 0 or not bool(torch.isfinite(score).all()):
        raise ValueError("foreground_score must be non-empty and finite")

    flat = score.reshape(-1)
    foreground_count = int(source.sum())
    selected = torch.zeros_like(flat, dtype=torch.bool)
    if foreground_count > 0:
        order = torch.argsort(flat, descending=True, stable=True)
        selected[order[:foreground_count]] = True
        cutoff = float(flat[order[foreground_count - 1]])
        tie_count = int((flat == cutoff).sum())
    else:
        # Conceptual threshold for an empty foreground is just above the
        # maximum score.  Keep the diagnostic finite so downstream summary
        # statistics never acquire an artificial Inf.
        cutoff = float(np.nextafter(float(flat.max()), np.inf))
        tie_count = 0

    selected_count = int(selected.sum())
    if selected_count != foreground_count:
        raise RuntimeError(
            f"area transfer count mismatch: {selected_count} != {foreground_count}"
        )
    size = int(flat.numel())
    return AreaTransferResult(
        binary_mask=selected.reshape_as(source),
        source_foreground_count=foreground_count,
        selected_foreground_count=selected_count,
        source_area_ratio=foreground_count / size,
        selected_area_ratio=selected_count / size,
        cutoff_score=cutoff,
        boundary_tie_count=tie_count,
    )
