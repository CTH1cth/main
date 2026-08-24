from __future__ import annotations

import numpy as np

from tools.gbsp_teaser_analysis.common import distribution_stats, probability_hist


def summarize_residual(
    residual: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    bins: np.ndarray,
) -> dict:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    bg = residual[valid & (label == 0)]
    fg = residual[valid & (label == 1)]
    if bg.size == 0 or fg.size == 0:
        raise ValueError(f"insufficient patches: bg={bg.size}, fg={fg.size}")
    return {
        "bg_hist": probability_hist(bg, bins),
        "fg_hist": probability_hist(fg, bins),
        "num_bg_patches": int(bg.size), "num_fg_patches": int(fg.size),
        **distribution_stats(bg, "bg"),
        **distribution_stats(fg, "fg"),
    }

