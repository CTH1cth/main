#!/usr/bin/env python3
"""Formal native-size continuous AP/AUROC evaluation for rescue scores."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import load_config  # noqa: E402
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    aggregate_per_dataset, load_manifest, load_native_gt, load_torch, rank_metrics,
    resize_score_to_native, score_from_payload, score_path, stratified_paired_bootstrap,
    write_csv, write_json,
)
from tools.reconstruction_rescue_common import require_output_outside_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dinov1_s8_reconstruction_rescue.py")
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--knn8_root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap_repetitions", type=int)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _available_methods(row: dict) -> list[str]:
    payload = load_torch(score_path(row))
    if not isinstance(payload.get("scores"), dict):
        raise KeyError("score payload has no scores mapping")
    return sorted(payload["scores"])


def _evaluate_one(task: tuple[dict, tuple[str, ...]]) -> list[dict]:
    row, methods = task
    payload = load_torch(score_path(row))
    gt = load_native_gt(row["gt_path"])
    shape = tuple(gt.shape[-2:])
    target = gt.numpy().reshape(-1)
    output = []
    for method in methods:
        score = resize_score_to_native(score_from_payload(payload, method), shape).numpy().reshape(-1)
        output.append({
            "dataset": row["dataset"], "stem": row["stem"], "method": method,
            "protocol": "native_gt_after_37_to_68_to_native_bilinear",
            "fg_pixels": int((target > .5).sum()), "bg_pixels": int((target <= .5).sum()),
            **rank_metrics(score, target),
        })
    return output


def _hist_metrics(histogram: np.ndarray) -> dict[str, float]:
    negative = histogram[0].astype(np.float64)
    positive = histogram[1].astype(np.float64)
    p_count, n_count = positive.sum(), negative.sum()
    if p_count <= 0 or n_count <= 0:
        return {"AP": float("nan"), "AUROC": float("nan")}
    descending_p = positive[::-1]
    descending_n = negative[::-1]
    true_positive = np.cumsum(descending_p)
    false_positive = np.cumsum(descending_n)
    recall = true_positive / p_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    ap = np.sum((recall - np.r_[0.0, recall[:-1]]) * precision)
    lower_negative = np.cumsum(negative) - negative
    wins = np.sum(positive * (lower_negative + .5 * negative))
    return {"AP": float(ap), "AUROC": float(wins / (p_count * n_count))}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = require_output_outside_main(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.score_root, score=True, split=args.split, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("empty score manifest")
    available = _available_methods(rows[0])
    methods = tuple(args.methods or available)
    missing = sorted(set(methods) - set(available))
    if missing:
        raise KeyError(f"methods absent from first payload: {missing}; available={available}")
    # The score cache embeds a freshly reproduced strict-LOO KNN8 baseline.
    # An external root is accepted only as provenance and never silently mixed.
    if args.knn8_root:
        external = load_manifest(args.knn8_root, score=True, split=args.split, max_samples=args.max_samples)
        external_ids = {(row["dataset"], row["stem"]) for row in external}
        current_ids = {(row["dataset"], row["stem"]) for row in rows}
        if external_ids != current_ids:
            raise RuntimeError("external KNN8 manifest identity set differs from rescue scores")
    per_image: list[dict] = []
    tasks = ((row, methods) for row in rows)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, image_rows in enumerate(pool.map(_evaluate_one, tasks, chunksize=1), 1):
                per_image.extend(image_rows)
                if index % 100 == 0 or index == len(rows):
                    print(f"[{index}/{len(rows)}] native continuous", flush=True)
    else:
        for index, task in enumerate(tasks, 1):
            per_image.extend(_evaluate_one(task))
            if index % 100 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] native continuous", flush=True)
    summary = aggregate_per_dataset(per_image, group_fields=("method",), metric_fields=("AP", "AUROC"))
    write_csv(output / "per_image_continuous.csv", per_image)
    write_csv(output / "per_dataset_continuous.csv", summary)

    # Streaming pooled native metrics with method-specific global ranges.
    ranges = {method: [float("inf"), float("-inf")] for method in methods}
    for row in rows:
        payload = load_torch(score_path(row))
        for method in methods:
            score = score_from_payload(payload, method)
            ranges[method][0] = min(ranges[method][0], float(score.min()))
            ranges[method][1] = max(ranges[method][1], float(score.max()))
    histograms = defaultdict(lambda: np.zeros((2, 65536), dtype=np.int64))
    for index, row in enumerate(rows, 1):
        payload = load_torch(score_path(row)); gt = load_native_gt(row["gt_path"])
        shape = tuple(gt.shape[-2:]); target = gt.numpy().reshape(-1) > .5
        for method in methods:
            value = resize_score_to_native(score_from_payload(payload, method), shape).numpy().reshape(-1)
            low, high = ranges[method]
            bins = np.clip(((value - low) / max(high - low, 1e-12) * 65535).astype(np.int64), 0, 65535)
            for dataset in (row["dataset"], "ALL"):
                histograms[(dataset, method)][0] += np.bincount(bins[~target], minlength=65536)
                histograms[(dataset, method)][1] += np.bincount(bins[target], minlength=65536)
        if index % 500 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] pooled histograms", flush=True)
    pooled = [
        {"dataset": dataset, "method": method,
         "protocol": "pooled_native_hist65536_method_global_range", **_hist_metrics(histogram)}
        for (dataset, method), histogram in sorted(histograms.items())
    ]
    write_csv(output / "pooled_continuous.csv", pooled)

    repetitions = args.bootstrap_repetitions or cfg.RECONSTRUCTION_BOOTSTRAP_REPETITIONS
    bootstrap = []
    for method in methods:
        if method != "knn8":
            bootstrap.extend(stratified_paired_bootstrap(
                per_image, candidate=method, baseline="knn8", method_field="method",
                metrics=("AP", "AUROC"), repetitions=repetitions,
                seed=cfg.RECONSTRUCTION_BOOTSTRAP_SEED,
            ))
    write_csv(output / "paired_bootstrap_vs_knn8.csv", bootstrap)
    dataset_macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    reproduction = []
    for method, reference in cfg.RECONSTRUCTION_BASELINE_REFERENCE.items():
        if method not in dataset_macro:
            continue
        row = dataset_macro[method]
        reproduction.append({
            "method": method, "images": len(rows),
            "reference_AP": reference["dataset_macro_AP"], "current_AP": row["AP"],
            "AP_abs_error": abs(row["AP"] - reference["dataset_macro_AP"]),
            "reference_AUROC": reference["dataset_macro_AUROC"], "current_AUROC": row["AUROC"],
            "AUROC_abs_error": abs(row["AUROC"] - reference["dataset_macro_AUROC"]),
            "formal_gate_applicable": len(rows) == 6473,
        })
    write_csv(output / "baseline_reproduction.csv", reproduction)
    reproduction_failures = [
        row for row in reproduction
        if row["formal_gate_applicable"]
        and max(row["AP_abs_error"], row["AUROC_abs_error"])
        > cfg.RECONSTRUCTION_BASELINE_TOLERANCE
    ]
    ranking = sorted(
        ({"method": method, "dataset_macro_AP": row["AP"], "dataset_macro_AUROC": row["AUROC"]}
         for method, row in dataset_macro.items()), key=lambda row: row["dataset_macro_AP"], reverse=True,
    )
    validity = {
        "images": len(rows), "methods": list(methods), "failures": 0,
        "gt_used_only_for_evaluation": True,
        "primary_aggregation": "unweighted macro of four dataset per-image means",
        "ranking": ranking, "baseline_reproduction_failures": reproduction_failures,
    }
    write_json(output / "continuous_validity.json", validity)
    (output / "CONTINUOUS_REPORT.md").write_text(
        "# Reconstruction Rescue Continuous Evaluation\n\n"
        + "主口径：四数据集分别做逐图 native AP/AUROC，再对四库等权平均。\n\n"
        + "| Method | AP | AUROC |\n|---|---:|---:|\n"
        + "".join(f"| {row['method']} | {row['dataset_macro_AP']:.6f} | {row['dataset_macro_AUROC']:.6f} |\n" for row in ranking),
        encoding="utf-8",
    )
    if reproduction_failures:
        raise RuntimeError(f"formal baseline reproduction failed: {reproduction_failures}")
    print(json.dumps(validity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
