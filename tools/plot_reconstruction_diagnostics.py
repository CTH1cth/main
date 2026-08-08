#!/usr/bin/env python3
"""Create compact paper-facing plots from completed rescue CSV outputs."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reconstruction_rescue_common import require_output_outside_main


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous_dir", required=True)
    parser.add_argument("--residual_dir")
    parser.add_argument("--threshold_dir")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); out = require_output_outside_main(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    continuous = rows(Path(args.continuous_dir) / "per_dataset_continuous.csv")
    macro = [row for row in continuous if row["scope"] == "dataset_macro"]
    methods = [row["method"] for row in macro]
    figure, axes = plt.subplots(1, 2, figsize=(max(8, len(methods) * .85), 3.6))
    for axis, metric in zip(axes, ("AP", "AUROC")):
        axis.bar(methods, [float(row[metric]) for row in macro])
        axis.set_title(f"Dataset-macro {metric}"); axis.tick_params(axis="x", rotation=55)
        axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(out / "continuous_dataset_macro.png", dpi=220); plt.close(figure)

    if args.residual_dir and (Path(args.residual_dir) / "residual_quantile_calibration.csv").is_file():
        quantiles = rows(Path(args.residual_dir) / "residual_quantile_calibration.csv")
        figure, axis = plt.subplots(figsize=(6, 4))
        for method in sorted({row["method"] for row in quantiles}):
            selected = sorted((row for row in quantiles if row["method"] == method), key=lambda row: int(row["score_decile"]))
            axis.plot([int(row["score_decile"]) for row in selected],
                      [float(row["mean_foreground_fraction"]) for row in selected], marker="o", label=method)
        axis.set_xlabel("Residual score decile"); axis.set_ylabel("Foreground fraction")
        axis.grid(alpha=.25); axis.legend(fontsize=7); figure.tight_layout()
        figure.savefig(out / "residual_decile_calibration.png", dpi=220); plt.close(figure)

    if args.threshold_dir and (Path(args.threshold_dir) / "threshold_curves.csv").is_file():
        curves = rows(Path(args.threshold_dir) / "threshold_curves.csv")
        selected = [row for row in curves if row["scope"] == "dataset_macro"]
        figure, axis = plt.subplots(figsize=(6, 4))
        for method in sorted({row["method"] for row in selected}):
            subset = sorted((row for row in selected if row["method"] == method), key=lambda row: float(row["threshold"]))
            axis.plot([float(row["threshold"]) for row in subset],
                      [float(row["F_beta_w"]) for row in subset], label=method)
        axis.set_xlabel("Image-wise MinMax threshold"); axis.set_ylabel("Dataset-macro Fβw")
        axis.grid(alpha=.25); axis.legend(fontsize=7); figure.tight_layout()
        figure.savefig(out / "threshold_fbw_curves.png", dpi=220); plt.close(figure)


if __name__ == "__main__":
    main()
