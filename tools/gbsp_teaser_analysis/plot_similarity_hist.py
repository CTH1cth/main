from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.gbsp_teaser_analysis.common import BG_COLOR, FG_COLOR, Settings, save_vector_figure


def draw_similarity(ax, bb: np.ndarray, fb: np.ndarray, bins: np.ndarray, *, compact: bool = False) -> None:
    centers = (bins[:-1] + bins[1:]) / 2; width = np.diff(bins)
    ax.step(centers, bb / width, where="mid", color=BG_COLOR, lw=1.8, label="BG--BG pairs")
    ax.fill_between(centers, bb / width, step="mid", color=BG_COLOR, alpha=.25)
    ax.step(centers, fb / width, where="mid", color=FG_COLOR, lw=1.8, label="FG--BG pairs")
    ax.fill_between(centers, fb / width, step="mid", color=FG_COLOR, alpha=.25)
    ax.set_xlim(-1, 1); ax.set_xlabel("Pairwise Cosine Similarity")
    ax.set_ylabel("Density"); ax.grid(axis="y", alpha=.16, linewidth=.6)
    ax.legend(frameon=False, fontsize=8 if compact else 9)


def plot(settings: Settings, out: str | Path) -> None:
    out = Path(out)
    for protocol, directory, name, subtitle in (
        ("main", out / "CAMO/similarity", "similarity_hist", "Pairwise DINO cosine similarity on CAMO-Test"),
        ("core", out / "CAMO/robustness_core", "similarity_hist_core", "Core-patch robustness (0.2/0.8)"),
    ):
        prefix = "" if protocol == "main" else ""
        bb_path = directory / ("bb_hist.npy" if protocol == "main" else "bb_hist.npy")
        fb_path = directory / ("fb_hist.npy" if protocol == "main" else "fb_hist.npy")
        if not bb_path.is_file() or not fb_path.is_file():
            continue
        bb, fb = np.load(bb_path), np.load(fb_path)
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        draw_similarity(ax, bb, fb, settings.similarity_bins)
        ax.set_title("Similarity Ambiguity in Pretrained Feature Space\n" + subtitle, fontsize=11)
        fig.tight_layout(); save_vector_figure(fig, directory / name); plt.close(fig)

