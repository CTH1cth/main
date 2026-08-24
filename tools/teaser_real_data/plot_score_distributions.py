#!/usr/bin/env python3
"""Draw fixed-protocol real FG/BG score distributions as vector figures."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import (  # noqa: E402
    load_npz, load_score_rows, load_settings, save_vector_figure,
)

BG_COLOR = "#4C78A8"
FG_COLOR = "#F58518"
METHOD_FIELDS = {
    "KNN8 reference matching": "knn8_foreground_score_normalized",
    "GBSP projection residual": "gbsp_normalized_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _collect(rows: list[dict]):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        with load_npz(row) as payload:
            for protocol, label, valid in (
                ("core", payload["core_label"], payload["core_label"] >= 0),
                ("allpatch", payload["binary_label"], np.ones(payload["binary_label"].shape, dtype=bool)),
            ):
                for method, field in METHOD_FIELDS.items():
                    score = payload[field]
                    data[row["dataset"]][protocol][method].append((score[valid & (label == 0)], score[valid & (label == 1)]))
    return data


def _draw_axis(ax, bg: np.ndarray, fg: np.ndarray, bins: int, title: str, *, legend: bool) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ax.hist(bg, bins=edges, density=True, histtype="stepfilled", alpha=0.34,
            linewidth=1.7, edgecolor=BG_COLOR, color=BG_COLOR, label="Background")
    ax.hist(fg, bins=edges, density=True, histtype="stepfilled", alpha=0.34,
            linewidth=1.7, edgecolor=FG_COLOR, color=FG_COLOR, label="Foreground")
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Foreground-oriented normalized score")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    if legend:
        ax.legend(frameon=False, fontsize=8.5)


def plot(config: str | Path, score_root: str | Path, out_dir: str | Path) -> None:
    settings = load_settings(config)
    rows = load_score_rows(score_root)
    data = _collect(rows)
    out = Path(out_dir) / "distributions"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    for dataset in settings.diagnostic_datasets:
        if dataset not in data:
            continue
        for protocol in ("core", "allpatch"):
            fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharex=True, sharey=True)
            for index, (method, parts) in enumerate(data[dataset][protocol].items()):
                bg = np.concatenate([part[0] for part in parts])
                fg = np.concatenate([part[1] for part in parts])
                _draw_axis(axes[index], bg, fg, settings.hist_bins, method, legend=index == 0)
                axes[index].text(.98, .95, f"BG {len(bg):,}\nFG {len(fg):,}", transform=axes[index].transAxes,
                                 ha="right", va="top", fontsize=7.5)
            axes[0].set_ylabel("Density")
            label = "Core patches" if protocol == "core" else "All patches"
            fig.suptitle(f"{dataset}: real FG/BG score distributions ({label})", fontsize=12)
            fig.tight_layout()
            save_vector_figure(fig, out / f"{dataset}_{protocol}")
            plt.close(fig)

    present = [dataset for dataset in settings.diagnostic_datasets if dataset in data]
    if present:
        fig, axes = plt.subplots(len(present), 2, figsize=(8.4, 2.55 * len(present)), sharex=True)
        axes = np.asarray(axes).reshape(len(present), 2)
        for row_index, dataset in enumerate(present):
            for col_index, (method, parts) in enumerate(data[dataset]["core"].items()):
                bg = np.concatenate([part[0] for part in parts])
                fg = np.concatenate([part[1] for part in parts])
                _draw_axis(axes[row_index, col_index], bg, fg, settings.hist_bins,
                           f"{dataset} — {method}", legend=row_index == 0 and col_index == 0)
                if col_index == 0:
                    axes[row_index, col_index].set_ylabel("Density")
        fig.suptitle("Cross-dataset consistency under the strict core-patch protocol", fontsize=12)
        fig.tight_layout()
        save_vector_figure(fig, out / "aggregate_diagnostic_core")
        plt.close(fig)

    if settings.main_dataset in data:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), sharex=True)
        for row_index, protocol in enumerate(("core", "allpatch")):
            for col_index, (method, parts) in enumerate(data[settings.main_dataset][protocol].items()):
                bg = np.concatenate([part[0] for part in parts])
                fg = np.concatenate([part[1] for part in parts])
                _draw_axis(axes[row_index, col_index], bg, fg, settings.hist_bins,
                           f"{method} — {protocol}", legend=row_index == 0 and col_index == 0)
                if col_index == 0:
                    axes[row_index, col_index].set_ylabel("Density")
        fig.suptitle("CAMO robustness: core patches versus all patches", fontsize=12)
        fig.tight_layout()
        save_vector_figure(fig, out / "CAMO_core_vs_allpatch")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    plot(args.config, args.score_root, args.out_dir)


if __name__ == "__main__":
    main()
