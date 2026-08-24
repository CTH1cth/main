from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.gbsp_teaser_analysis.common import BG_COLOR, FG_COLOR, save_vector_figure


def draw_projection(ax, *, compact: bool = False) -> None:
    mu = np.array([.75, .75]); direction = np.array([1.0, .35]); direction /= np.linalg.norm(direction)
    query = np.array([4.1, 3.65]); projected = mu + direction * np.dot(query - mu, direction)
    line = np.stack([mu - .55 * direction, mu + 4.45 * direction])
    ax.plot(line[:, 0], line[:, 1], color=BG_COLOR, lw=3.0)
    ax.scatter(*mu, color=BG_COLOR, s=42, zorder=3); ax.text(mu[0]-.18, mu[1]-.38, r"$\boldsymbol{\mu}$", fontsize=10)
    ax.scatter(*projected, color="#54A24B", s=55, zorder=4)
    ax.text(projected[0]-.12, projected[1]-.52,
            r"$\hat{\mathbf{f}}_i$" + "\nbackground-explainable", fontsize=8, ha="center")
    ax.scatter(*query, color=FG_COLOR, s=62, zorder=4); ax.text(query[0]+.10, query[1], r"query $\mathbf{f}_i$", fontsize=10)
    ax.annotate("", xy=projected, xytext=query, arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=2.2))
    middle = (query + projected) / 2
    ax.text(middle[0]+.13, middle[1], r"$\mathbf{r}_i$" + "\nunexplained deviation", color="#E45756", fontsize=8)
    ax.text(2.3, .18,
            r"$\hat{\mathbf{f}}_i=\boldsymbol{\mu}+\mathbf{U}\mathbf{U}^{\top}(\mathbf{f}_i-\boldsymbol{\mu})$" "\n"
            r"$q_i=\|\mathbf{f}_i-\hat{\mathbf{f}}_i\|_2^2$",
            ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#BBBBBB"))
    ax.set_xlim(.05, 5.2); ax.set_ylim(-.05, 4.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Background Explainability", fontsize=11 if compact else 12)


def plot(out: str | Path) -> None:
    directory = Path(out) / "CAMO/projection"; directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.55)); draw_projection(ax)
    fig.tight_layout(); save_vector_figure(fig, directory / "projection_schematic"); plt.close(fig)

