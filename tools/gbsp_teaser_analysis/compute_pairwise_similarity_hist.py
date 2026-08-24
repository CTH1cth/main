from __future__ import annotations

import numpy as np
import torch

from tools.gbsp_teaser_analysis.common import distribution_stats, probability_hist


def pairwise_values(
    similarity: torch.Tensor,
    label: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique BG--BG and all FG--BG pairs; no aggregation or dictionary."""
    device = similarity.device
    bg = torch.as_tensor(np.flatnonzero(valid & (label == 0)), device=device, dtype=torch.long)
    fg = torch.as_tensor(np.flatnonzero(valid & (label == 1)), device=device, dtype=torch.long)
    if bg.numel() < 2 or fg.numel() < 1:
        raise ValueError(f"insufficient patches: bg={bg.numel()}, fg={fg.numel()}")
    bb_matrix = similarity.index_select(0, bg).index_select(1, bg)
    upper = torch.triu_indices(bg.numel(), bg.numel(), offset=1, device=device)
    bb = bb_matrix[upper[0], upper[1]].detach().cpu().numpy()
    fb = similarity.index_select(0, fg).index_select(1, bg).reshape(-1).detach().cpu().numpy()
    return bb, fb


def summarize_pairwise(
    similarity: torch.Tensor,
    label: np.ndarray,
    valid: np.ndarray,
    bins: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict:
    bb, fb = pairwise_values(similarity, label, valid)
    return {
        "bb_hist": probability_hist(bb, bins),
        "fb_hist": probability_hist(fb, bins),
        "num_bb_pairs": int(bb.size), "num_fb_pairs": int(fb.size),
        **distribution_stats(bb, "bb", thresholds),
        **distribution_stats(fb, "fb", thresholds),
    }

