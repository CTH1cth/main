#!/usr/bin/env python3
"""Stage 2: compare KNN8 and Global GBSP thresholdability identically."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext
from tools.gbsp_knn_lsr_common import (
    COD_METRICS,
    adaptive_threshold,
    aggregate_per_dataset,
    cod_metrics,
    index_manifest,
    load_manifest,
    load_native_gt,
    load_torch,
    minmax,
    normalize_dataset,
    plateau_rows,
    resize_score_to_native,
    score_from_payload,
    score_path,
    write_csv,
    write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
METHODS = {"knn8": "knn8_cos", "gbsp": "gbsp_r8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=tuple(METHODS))
    parser.add_argument("--image_minmax", action="store_true")
    parser.add_argument("--threshold_start", type=float, default=0.30)
    parser.add_argument("--threshold_end", type=float, default=0.70)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--fixed_thresholds", nargs="+", type=float, default=(0.50, 0.58))
    parser.add_argument("--adaptive_thresholds", nargs="+", choices=("otsu", "multi_otsu_3"), default=("otsu", "multi_otsu_3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _aggregate_curve(accumulator: dict) -> list[dict]:
    rows = []
    for (dataset, method, threshold), state in sorted(accumulator.items()):
        rows.append({
            "scope": "dataset", "dataset": dataset, "method": method,
            "threshold": threshold, "num_samples": state["count"],
            **{metric: state[metric] / state["count"] for metric in COD_METRICS},
        })
    for method in sorted({row["method"] for row in rows}):
        for threshold in sorted({row["threshold"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["threshold"] == threshold]
            rows.append({
                "scope": "dataset_macro", "dataset": "ALL", "method": method,
                "threshold": threshold, "num_samples": sum(row["num_samples"] for row in selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in COD_METRICS},
            })
    return rows


def _init_worker() -> None:
    torch.set_num_threads(1)


def _process_one(task: dict) -> dict:
    row, gbsp_row = task["row"], task["gbsp_row"]
    dataset, stem = normalize_dataset(row["dataset"]), row["stem"]
    try:
        sim_payload = load_torch(score_path(row))
        gbsp_payload = sim_payload if score_path(row).resolve() == score_path(gbsp_row).resolve() else load_torch(score_path(gbsp_row))
        gt = load_native_gt(row["gt_path"])
        context = FastCODContext(gt)
        shape = tuple(gt.shape[-2:])
        fixed, adaptive, curve, histograms = [], [], [], []
        for method in task["methods"]:
            payload = sim_payload if method == "knn8" else gbsp_payload
            probability37 = minmax(score_from_payload(payload, METHODS[method])).reshape(1, 37, 37)
            probability = resize_score_to_native(probability37, shape)
            for label, mask in (("foreground", gt > 0.5), ("background", gt <= 0.5)):
                values = probability[mask]
                bins = np.clip((values.numpy() * 255).astype(np.int64), 0, 255)
                histograms.append((dataset, method, label, np.bincount(bins, minlength=256)))
            for threshold in task["fixed_thresholds"]:
                fixed.append({
                    "dataset": dataset, "stem": stem, "method": method,
                    "protocol": f"fixed_{threshold:.2f}", "threshold": threshold,
                    **cod_metrics(context, probability, threshold),
                })
            for name in task["adaptive_thresholds"]:
                threshold, fallback = adaptive_threshold(probability37, name)
                adaptive.append({
                    "dataset": dataset, "stem": stem, "method": method,
                    "protocol": name, "threshold": threshold, "fallback": int(fallback),
                    **cod_metrics(context, probability, threshold),
                })
            for threshold in task["thresholds"]:
                curve.append({
                    "dataset": dataset, "method": method, "threshold": threshold,
                    **cod_metrics(context, probability, threshold),
                })
        return {"fixed": fixed, "adaptive": adaptive, "curve": curve, "histograms": histograms}
    except Exception as error:
        return {"error": repr(error), "dataset": dataset, "stem": stem}


def main() -> None:
    args = parse_args()
    if not args.image_minmax:
        raise ValueError("formal thresholdability protocol requires --image_minmax")
    if tuple(round(v, 2) for v in args.fixed_thresholds) != (0.50, 0.58):
        raise ValueError("formal fixed thresholds are frozen to 0.50 and 0.58")
    output = Path(args.out_dir).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("output must stay outside the main code tree")
    output.mkdir(parents=True, exist_ok=True)
    sim_rows = load_manifest(args.score_root, score=True, split=args.split, max_samples=args.max_samples)
    gbsp_index = index_manifest(load_manifest(args.gbsp_root, score=True, split=args.split))
    thresholds = tuple(float(round(value, 10)) for value in np.arange(
        args.threshold_start, args.threshold_end + 0.5 * args.threshold_step, args.threshold_step
    ))
    fixed_rows, adaptive_rows = [], []
    curve_accumulator = defaultdict(lambda: {"count": 0, **{metric: 0.0 for metric in COD_METRICS}})
    hist_accumulator = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    failures = []
    tasks = [{
        "row": row, "gbsp_row": gbsp_index[(normalize_dataset(row["dataset"]), row["stem"])],
        "methods": tuple(args.methods), "fixed_thresholds": tuple(args.fixed_thresholds),
        "adaptive_thresholds": tuple(args.adaptive_thresholds), "thresholds": thresholds,
    } for row in sim_rows]
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        results = executor.map(_process_one, tasks, chunksize=1)
    else:
        executor = None
        results = map(_process_one, tasks)
    for index, result in enumerate(results, 1):
        if "error" in result:
            failures.append(result)
        else:
            fixed_rows.extend(result["fixed"])
            adaptive_rows.extend(result["adaptive"])
            for dataset, method, label, histogram in result["histograms"]:
                hist_accumulator[(dataset, method, label)] += histogram
            for row in result["curve"]:
                state = curve_accumulator[(row["dataset"], row["method"], row["threshold"])]
                state["count"] += 1
                for name in COD_METRICS:
                    state[name] += row[name]
        if index % 50 == 0 or index == len(sim_rows):
            print(f"[{index}/{len(sim_rows)}] thresholdability failed={len(failures)}", flush=True)
    if executor is not None:
        executor.shutdown()
    if failures:
        write_json(output / "numerical_validity.json", {"status": "failed", "failures": failures})
        raise RuntimeError(f"thresholdability failed for {len(failures)} images")

    fixed_summary = aggregate_per_dataset(fixed_rows, group_fields=("method", "protocol"), metric_fields=COD_METRICS)
    adaptive_summary = aggregate_per_dataset(adaptive_rows, group_fields=("method", "protocol"), metric_fields=(*COD_METRICS, "threshold", "fallback"))
    curve = _aggregate_curve(curve_accumulator)
    write_csv(output / "thresholdability_comparison.csv", fixed_summary)
    write_csv(output / "adaptive_threshold_comparison.csv", adaptive_summary)
    write_csv(output / "knn8_threshold_curve.csv", [row for row in curve if row["method"] == "knn8"])
    write_csv(output / "gbsp_threshold_curve.csv", [row for row in curve if row["method"] == "gbsp"])
    plateaus = plateau_rows(curve)
    write_csv(output / "threshold_plateau_width.csv", plateaus)
    histogram_rows = []
    for (dataset, method, label), histogram in sorted(hist_accumulator.items()):
        total = max(int(histogram.sum()), 1)
        for bin_index, count in enumerate(histogram):
            histogram_rows.append({
                "dataset": dataset, "method": method, "class": label,
                "bin_left": bin_index / 255.0, "count": int(count), "density": float(count / total),
            })
    write_csv(output / "score_histograms.csv", histogram_rows)
    validity = {
        "stage": 2, "images": len(sim_rows), "methods": args.methods,
        "image_wise_minmax": True, "resize": "bilinear_37_to_68_to_native",
        "threshold_count": len(thresholds), "fixed_thresholds": args.fixed_thresholds,
        "adaptive_thresholds": args.adaptive_thresholds, "failures": 0,
        "gt_used_for_score_generation": False, "gt_used_only_for_evaluation": True,
    }
    write_json(output / "numerical_validity.json", validity)
    print(json.dumps(validity, ensure_ascii=False))


if __name__ == "__main__":
    main()
