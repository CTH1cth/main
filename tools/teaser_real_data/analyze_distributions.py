#!/usr/bin/env python3
"""Compute fixed-bin overlap and separation diagnostics for real teaser scores."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import (  # noqa: E402
    load_npz, load_score_rows, load_settings, write_csv, write_json,
)

METHOD_FIELDS = {
    "KNN8": "knn8_foreground_score_normalized",
    "GBSP-r8": "gbsp_normalized_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _metrics(bg: np.ndarray, fg: np.ndarray, bins: int) -> dict[str, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bg_hist, _ = np.histogram(bg, bins=edges, density=True)
    fg_hist, _ = np.histogram(fg, bins=edges, density=True)
    width = edges[1] - edges[0]
    labels = np.r_[np.zeros(bg.size, dtype=np.uint8), np.ones(fg.size, dtype=np.uint8)]
    scores = np.r_[bg, fg]
    return {
        "OVL": float(np.minimum(bg_hist, fg_hist).sum() * width),
        "Wasserstein": float(wasserstein_distance(bg, fg)),
        "AUROC": float(roc_auc_score(labels, scores)),
        "AP": float(average_precision_score(labels, scores)),
        "BG_mean": float(bg.mean()), "FG_mean": float(fg.mean()),
        "mean_gap": float(fg.mean() - bg.mean()),
        "BG_median": float(np.median(bg)), "FG_median": float(np.median(fg)),
        "median_gap": float(np.median(fg) - np.median(bg)),
    }


def analyze(config: str | Path, score_root: str | Path, out_dir: str | Path) -> dict:
    settings = load_settings(config)
    rows = load_score_rows(score_root)
    out = Path(out_dir)
    distributions: dict[tuple[str, str, str, int], list[np.ndarray]] = defaultdict(list)
    patch_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_image = []
    for row in rows:
        dataset = row["dataset"]
        with load_npz(row) as payload:
            core_label = payload["core_label"]
            binary_label = payload["binary_label"]
            for protocol, label, valid in (
                ("core", core_label, core_label >= 0),
                ("allpatch", binary_label, np.ones(binary_label.shape, dtype=bool)),
            ):
                for method, field in METHOD_FIELDS.items():
                    score = payload[field].astype(np.float64)
                    bg = score[valid & (label == 0)]
                    fg = score[valid & (label == 1)]
                    distributions[(dataset, protocol, method, 0)].append(bg)
                    distributions[(dataset, protocol, method, 1)].append(fg)
                    if bg.size and fg.size:
                        item = _metrics(bg, fg, settings.hist_bins)
                        per_image.append({
                            "dataset": dataset, "stem": row["stem"],
                            "protocol": protocol, "method": method,
                            "num_bg": int(bg.size), "num_fg": int(fg.size), **item,
                        })
                patch_counts[(dataset, protocol)]["images"] += 1
                patch_counts[(dataset, protocol)]["background"] += int((valid & (label == 0)).sum())
                patch_counts[(dataset, protocol)]["foreground"] += int((valid & (label == 1)).sum())
                patch_counts[(dataset, protocol)]["ignored"] += int((~valid).sum())

    metric_rows = []
    scopes = list(settings.diagnostic_datasets) + ["ALL_DIAGNOSTIC"]
    for scope in scopes:
        datasets = settings.diagnostic_datasets if scope == "ALL_DIAGNOSTIC" else (scope,)
        for protocol in ("core", "allpatch"):
            for method in METHOD_FIELDS:
                bg_parts = [part for dataset in datasets for part in distributions.get((dataset, protocol, method, 0), [])]
                fg_parts = [part for dataset in datasets for part in distributions.get((dataset, protocol, method, 1), [])]
                if not bg_parts or not fg_parts:
                    continue
                bg, fg = np.concatenate(bg_parts), np.concatenate(fg_parts)
                metric_rows.append({
                    "dataset": scope, "protocol": protocol, "method": method,
                    "hist_bins": settings.hist_bins, "num_bg": int(bg.size), "num_fg": int(fg.size),
                    **_metrics(bg, fg, settings.hist_bins),
                })
    count_rows = [
        {"dataset": dataset, "protocol": protocol, **dict(values)}
        for (dataset, protocol), values in sorted(patch_counts.items())
    ]
    summary_rows = []
    for dataset in scopes:
        for protocol in ("core", "allpatch"):
            lookup = {
                row["method"]: row for row in metric_rows
                if row["dataset"] == dataset and row["protocol"] == protocol
            }
            if set(lookup) == set(METHOD_FIELDS):
                knn, gbsp = lookup["KNN8"], lookup["GBSP-r8"]
                summary_rows.append({
                    "dataset": dataset, "protocol": protocol,
                    "OVL_KNN8": knn["OVL"], "OVL_GBSP": gbsp["OVL"],
                    "delta_OVL_GBSP_minus_KNN8": gbsp["OVL"] - knn["OVL"],
                    "Wasserstein_KNN8": knn["Wasserstein"], "Wasserstein_GBSP": gbsp["Wasserstein"],
                    "delta_Wasserstein_GBSP_minus_KNN8": gbsp["Wasserstein"] - knn["Wasserstein"],
                    "AUROC_KNN8": knn["AUROC"], "AUROC_GBSP": gbsp["AUROC"],
                    "delta_AUROC_GBSP_minus_KNN8": gbsp["AUROC"] - knn["AUROC"],
                })
    diagnostics = out / "diagnostics"
    write_csv(diagnostics / "distribution_metrics.csv", metric_rows)
    write_csv(diagnostics / "patch_counts.csv", count_rows)
    write_csv(diagnostics / "per_dataset_summary.csv", summary_rows)
    write_csv(diagnostics / "per_image_distribution_metrics.csv", per_image)
    validity = {
        "images": len(rows),
        "counts": {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in settings.diagnostic_datasets},
        "metric_rows": len(metric_rows), "summary_rows": len(summary_rows),
        "histogram_bins": settings.hist_bins, "score_range": [0.0, 1.0],
    }
    write_json(diagnostics / "analysis_validity.json", validity)
    return {"validity": validity, "metrics": metric_rows, "summary": summary_rows}


def main() -> None:
    args = parse_args()
    result = analyze(args.config, args.score_root, args.out_dir)
    print(json.dumps(result["validity"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
