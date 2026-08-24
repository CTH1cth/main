from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.gbsp_teaser_analysis.common import BG_COLOR, FG_COLOR, Settings, save_vector_figure


def draw_residual(ax, bg: np.ndarray, fg: np.ndarray, bins: np.ndarray, *, compact: bool = False) -> None:
    centers = (bins[:-1] + bins[1:]) / 2; width = np.diff(bins)
    ax.step(centers, bg / width, where="mid", color=BG_COLOR, lw=1.8, label="Background patches")
    ax.fill_between(centers, bg / width, step="mid", color=BG_COLOR, alpha=.25)
    ax.step(centers, fg / width, where="mid", color=FG_COLOR, lw=1.8, label="Foreground patches")
    ax.fill_between(centers, fg / width, step="mid", color=FG_COLOR, alpha=.25)
    ax.set_xlim(0, 1); ax.set_xlabel("Normalized Residual Score")
    ax.set_ylabel("Density"); ax.grid(axis="y", alpha=.16, linewidth=.6)
    ax.legend(frameon=False, fontsize=8 if compact else 9)


def plot(settings: Settings, out: str | Path) -> None:
    out = Path(out)
    for protocol, directory, name, subtitle in (
        ("main", out / "CAMO/residual", "residual_hist", "Patch-level residual scores on CAMO-Test"),
        ("core", out / "CAMO/robustness_core", "residual_hist_core", "Core-patch robustness (0.2/0.8)"),
    ):
        bg_path = directory / "bg_hist.npy"; fg_path = directory / "fg_hist.npy"
        if not bg_path.is_file() or not fg_path.is_file():
            continue
        bg, fg = np.load(bg_path), np.load(fg_path)
        fig, ax = plt.subplots(figsize=(4.8, 3.55))
        draw_residual(ax, bg, fg, settings.residual_bins)
        ax.set_title("GBSP Residual Evidence\n" + subtitle, fontsize=11)
        fig.tight_layout(); save_vector_figure(fig, directory / name); plt.close(fig)

