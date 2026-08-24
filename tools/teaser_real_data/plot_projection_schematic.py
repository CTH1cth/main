#!/usr/bin/env python3
"""Draw a mathematically exact affine projection/residual schematic."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import save_vector_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def plot(out_dir: str | Path) -> None:
    out = Path(out_dir) / "projection"
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    mu = np.array([1.3, 1.15])
    direction = np.array([1.0, 0.42]); direction /= np.linalg.norm(direction)
    x = np.array([5.9, 4.25])
    projection = mu + direction * np.dot(x - mu, direction)
    line = np.stack([mu - 1.0 * direction, mu + 5.7 * direction])
    ax.plot(line[:, 0], line[:, 1], color="#4C78A8", linewidth=4.0, alpha=.9,
            label=r"image-specific affine background subspace $\mu+\mathrm{span}(U)$")
    ax.scatter(*mu, color="#4C78A8", s=70, zorder=4)
    ax.text(*(mu + np.array([-.25, -.38])), r"$\mu$", fontsize=13)
    ax.scatter(*projection, color="#54A24B", s=80, zorder=4)
    ax.text(*(projection + np.array([-.30, -.42])), r"$\hat{\mathbf{f}}_i$", fontsize=13)
    ax.scatter(*x, color="#F58518", s=90, zorder=5)
    ax.text(*(x + np.array([.10, .03])), r"query $\mathbf{f}_i$", fontsize=13)
    ax.annotate("", xy=projection, xytext=x,
                arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=2.5))
    midpoint = (x + projection) / 2
    ax.text(*(midpoint + np.array([.18, .04])), r"unexplained residual $q_i$", color="#E45756", fontsize=11)
    ax.annotate("", xy=projection, xytext=mu,
                arrowprops=dict(arrowstyle="-|>", color="#54A24B", lw=1.8, alpha=.9))
    ax.text(3.0, .46,
            r"$\hat{\mathbf{f}}_i=\boldsymbol{\mu}+\mathbf{U}\mathbf{U}^\top"
            r"(\mathbf{f}_i-\boldsymbol{\mu})$" "\n"
            r"$q_i=\left\|(\mathbf{I}-\mathbf{U}\mathbf{U}^\top)"
            r"(\mathbf{f}_i-\boldsymbol{\mu})\right\|_2^2$",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#BBBBBB", alpha=.96))
    ax.set_xlim(.1, 7.0); ax.set_ylim(.05, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("Background explainability and its projection residual", fontsize=14, pad=10)
    fig.tight_layout()
    save_vector_figure(fig, out / "projection_schematic")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.out_dir)


if __name__ == "__main__":
    main()
