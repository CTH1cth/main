#!/usr/bin/env python3
"""Evaluate thresholdability only for explicitly selected finalists."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    COD_METRICS, adaptive_threshold, aggregate_per_dataset, cod_metrics, load_manifest,
    load_native_gt, load_torch, minmax, plateau_rows, resize_score_to_native,
    score_from_payload, score_path, write_csv, write_json,
)
from tools.reconstruction_rescue_common import require_output_outside_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--threshold_start", type=float, default=.30)
    parser.add_argument("--threshold_end", type=float, default=.70)
    parser.add_argument("--threshold_step", type=float, default=.01)
    parser.add_argument("--fixed_thresholds", nargs="+", type=float, default=(.50, .58))
    parser.add_argument("--adaptive_thresholds", nargs="+", default=("otsu", "multi_otsu_3"))
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def aggregate_curve(accumulator: dict) -> list[dict]:
    rows = []
    for (dataset, method, threshold), state in sorted(accumulator.items()):
        rows.append({"scope": "dataset", "dataset": dataset, "method": method,
                     "threshold": threshold, "num_samples": state["count"],
                     **{metric: state[metric] / state["count"] for metric in COD_METRICS}})
    for method in sorted({row["method"] for row in rows}):
        for threshold in sorted({row["threshold"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["threshold"] == threshold]
            rows.append({"scope": "dataset_macro", "dataset": "ALL", "method": method,
                         "threshold": threshold, "num_samples": sum(row["num_samples"] for row in selected),
                         **{metric: float(np.mean([row[metric] for row in selected])) for metric in COD_METRICS}})
    return rows


def main() -> None:
    args = parse_args(); output = require_output_outside_main(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.score_root, score=True, split=args.split, max_samples=args.max_samples)
    thresholds = tuple(float(round(x, 10)) for x in np.arange(
        args.threshold_start, args.threshold_end + args.threshold_step / 2, args.threshold_step
    ))
    fixed_rows, adaptive_rows = [], []
    curves = defaultdict(lambda: {"count": 0, **{metric: 0.0 for metric in COD_METRICS}})
    failures = []
    for index, row in enumerate(rows, 1):
        try:
            payload = load_torch(score_path(row)); gt = load_native_gt(row["gt_path"])
            context = FastCODContext(gt); shape = tuple(gt.shape[-2:])
            for method in args.methods:
                probability37 = minmax(score_from_payload(payload, method)).reshape(1, 37, 37)
                probability = resize_score_to_native(probability37, shape)
                for threshold in args.fixed_thresholds:
                    fixed_rows.append({"dataset": row["dataset"], "stem": row["stem"], "method": method,
                                       "protocol": f"fixed_{threshold:.2f}", "threshold": threshold,
                                       **cod_metrics(context, probability, threshold)})
                for adaptive in args.adaptive_thresholds:
                    threshold, fallback = adaptive_threshold(probability37, adaptive)
                    adaptive_rows.append({"dataset": row["dataset"], "stem": row["stem"], "method": method,
                                          "protocol": adaptive, "threshold": threshold, "fallback": int(fallback),
                                          **cod_metrics(context, probability, threshold)})
                for threshold in thresholds:
                    metric = cod_metrics(context, probability, threshold)
                    state = curves[(row["dataset"], method, threshold)]; state["count"] += 1
                    for key in COD_METRICS:
                        state[key] += metric[key]
        except Exception as error:
            failures.append({"dataset": row.get("dataset"), "stem": row.get("stem"), "error": repr(error)})
        if index % 50 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] failed={len(failures)}", flush=True)
    curve = aggregate_curve(curves)
    write_csv(output / "fixed_thresholds.csv", aggregate_per_dataset(
        fixed_rows, group_fields=("method", "protocol"), metric_fields=COD_METRICS))
    write_csv(output / "adaptive_thresholds.csv", aggregate_per_dataset(
        adaptive_rows, group_fields=("method", "protocol"), metric_fields=(*COD_METRICS, "threshold", "fallback")))
    write_csv(output / "threshold_curves.csv", curve)
    write_csv(output / "threshold_plateau_width.csv", plateau_rows(curve))
    validity = {"images": len(rows), "methods": args.methods, "failures": failures,
                "image_wise_minmax": True, "gt_used_only_for_evaluation": True}
    write_json(output / "thresholdability_validity.json", validity)
    if failures:
        raise RuntimeError(f"thresholdability failed for {len(failures)} samples")
    print(json.dumps(validity, ensure_ascii=False))


if __name__ == "__main__":
    main()
