from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.gbsp_teaser_analysis.common import Settings, save_vector_figure
from tools.gbsp_teaser_analysis.plot_projection_schematic import draw_projection
from tools.gbsp_teaser_analysis.plot_residual_hist import draw_residual
from tools.gbsp_teaser_analysis.plot_similarity_hist import draw_similarity


def plot(settings: Settings, out: str | Path) -> None:
    out = Path(out)
    bb = np.load(out / "CAMO/similarity/bb_hist.npy")
    fb = np.load(out / "CAMO/similarity/fb_hist.npy")
    bg = np.load(out / "CAMO/residual/bg_hist.npy")
    fg = np.load(out / "CAMO/residual/fg_hist.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.65), gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]})
    draw_similarity(axes[0], bb, fb, settings.similarity_bins, compact=True)
    axes[0].set_title("Similarity Ambiguity\nPairwise DINO affinity", fontsize=11)
    draw_projection(axes[1], compact=True)
    draw_residual(axes[2], bg, fg, settings.residual_bins, compact=True)
    axes[2].set_title("GBSP Residual Evidence\nBackground vs foreground patches", fontsize=11)
    for x in (.335, .665):
        fig.text(x, .50, r"$\rightarrow$", ha="center", va="center", fontsize=21, color="#666666")
    fig.suptitle("Similarity Ambiguity  →  Background Explainability  →  Unexplained Residual", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=2.4)
    directory = out / "CAMO/teaser_preview"; directory.mkdir(parents=True, exist_ok=True)
    save_vector_figure(fig, directory / "teaser_preview"); plt.close(fig)
