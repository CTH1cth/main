#!/usr/bin/env python3
"""Post-hoc failure-mode and threshold-behavior audit for GBSP V2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    evaluation = Path(args.eval_dir).resolve()
    output = Path(args.out_dir).resolve() if args.out_dir else evaluation
    output.mkdir(parents=True, exist_ok=True)
    per_image = _rows(evaluation / "per_image_metrics.csv")
    adaptive = [row for row in per_image if row["method"] != "fixed_058"]
    by_method = defaultdict(list)
    classified = []
    for row in adaptive:
        area = _float(row, "Area")
        precision_delta = _float(row, "delta_Precision_vs_fixed_058")
        recall_delta = _float(row, "delta_Recall_vs_fixed_058")
        mean_gap = _float(row, "mean_gap")
        delta_bic = _float(row, "delta_bic")
        if area > 0.16 and precision_delta < 0 and recall_delta > 0:
            failure_mode = "foreground_inflation"
        elif area < 0.06 and precision_delta > 0 and recall_delta < 0:
            failure_mode = "over_conservative"
        elif (math.isfinite(mean_gap) and mean_gap < 0.05) or (
            math.isfinite(delta_bic) and delta_bic <= 10.0
        ):
            failure_mode = "component_unidentifiable_or_weak_change"
        elif int(float(row.get("numerical_failure", 0))) != 0:
            failure_mode = "numerical_failure"
        else:
            failure_mode = "other_or_success"
        classified.append(
            {
                "dataset": row["dataset"], "stem": row["stem"], "method": row["method"],
                "failure_mode": failure_mode, "F_beta_w_delta": _float(row, "delta_F_beta_w_vs_fixed_058"),
                "Precision_delta": precision_delta, "Recall_delta": recall_delta, "Area": area,
                "equivalent_minmax_threshold": row.get("equivalent_minmax_threshold", ""),
            }
        )
        by_method[row["method"]].append(row)
    _write_csv(output / "failure_mode_per_image.csv", classified)
    summary = []
    for method, rows in sorted(by_method.items()):
        modes = Counter(
            item["failure_mode"] for item in classified if item["method"] == method
        )
        thresholds = np.asarray(
            [value for value in (_float(row, "equivalent_minmax_threshold") for row in rows) if math.isfinite(value)]
        )
        summary.append(
            {
                "method": method, "num_images": len(rows),
                **{f"count_{key}": modes.get(key, 0) for key in (
                    "foreground_inflation", "over_conservative",
                    "component_unidentifiable_or_weak_change", "numerical_failure", "other_or_success",
                )},
                "threshold_mean": float(thresholds.mean()) if thresholds.size else "",
                "threshold_median": float(np.median(thresholds)) if thresholds.size else "",
                "threshold_q10": float(np.quantile(thresholds, 0.10)) if thresholds.size else "",
                "threshold_q90": float(np.quantile(thresholds, 0.90)) if thresholds.size else "",
            }
        )
    _write_csv(output / "failure_mode_summary.csv", summary)
    report = [
        "# GBSP Adaptive Threshold V2 Diagnostics", "",
        "This audit classifies failures without changing any method parameter or threshold.", "",
    ]
    for row in summary:
        report.append(
            f"- {row['method']}: inflation={row['count_foreground_inflation']}, "
            f"conservative={row['count_over_conservative']}, "
            f"unidentifiable={row['count_component_unidentifiable_or_weak_change']}, "
            f"numerical={row['count_numerical_failure']}."
        )
    (output / "METHOD_DIAGNOSTICS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"num_rows": len(adaptive), "methods": len(by_method), "out_dir": str(output)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", "--eval-dir", dest="eval_dir", required=True)
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir")
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
