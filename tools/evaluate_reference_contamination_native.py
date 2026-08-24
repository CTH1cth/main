#!/usr/bin/env python3
"""CPU-parallel native-resolution evaluation for deferred mechanism caches."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gbsp_knn_lsr_common import (  # noqa: E402
    DATASETS,
    EXPECTED_COUNTS,
    index_manifest,
    load_manifest,
    load_native_gt,
    load_torch,
    normalize_dataset,
    rank_metrics,
    resize_score_to_native,
    write_csv,
    write_json,
)


VERSION = "reference_contamination_native_eval_v1"
METHODS = ("knn8", "gbsp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--reference_root")
    parser.add_argument("--out_dir")
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _native_metrics(score: torch.Tensor, native_gt: torch.Tensor) -> dict[str, float]:
    value = torch.as_tensor(score).detach().cpu().float().reshape(-1)
    if value.numel() != 37 * 37 or not bool(torch.isfinite(value).all()):
        raise ValueError("native evaluation needs 1369 finite raw score values")
    native = resize_score_to_native(value, tuple(native_gt.shape[-2:]))
    return rank_metrics(native, native_gt)


def _score_pair_metrics(container: dict, native_gt: torch.Tensor) -> dict:
    return {
        method: _native_metrics(container[f"{method}_score"], native_gt)
        for method in METHODS
    }


def _metric_error(current: dict, reference: dict) -> float:
    errors = []
    for method in METHODS:
        for metric in ("AP", "AUROC"):
            errors.append(abs(float(current[method][metric]) - float(reference[method][metric])))
    return max(errors, default=0.0)


def _score_error(current: torch.Tensor, reference: torch.Tensor) -> float:
    left = torch.as_tensor(current).detach().cpu().float().reshape(-1)
    right = torch.as_tensor(reference).detach().cpu().float().reshape(-1)
    if left.shape != right.shape:
        return float("inf")
    return float((left - right).abs().max())


def _raw_score_reproduction(payload: dict, reference: dict) -> float:
    errors = []
    for section in ("natural", "oracle_clean"):
        for method in METHODS:
            errors.append(_score_error(
                payload[section][f"{method}_score"], reference[section][f"{method}_score"]
            ))
    reference_conditions = {
        (int(item["seed"]), float(item["p"])): item
        for item in reference["contamination"] if item["valid"]
    }
    for item in payload["contamination"]:
        if not item["valid"]:
            continue
        other = reference_conditions.get((int(item["seed"]), float(item["p"])))
        if other is None:
            return float("inf")
        for method in METHODS:
            errors.append(_score_error(item[f"{method}_score"], other[f"{method}_score"]))
    return max(errors, default=0.0)


def _evaluate_one(task: dict) -> dict:
    started = time.perf_counter()
    path = Path(task["cache_path"])
    try:
        payload = load_torch(path)
        if payload.get("native_metrics_complete") and not task["overwrite"]:
            return {
                "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(path),
                "skipped": True, "runtime_seconds": 0.0,
                "raw_score_max_abs_error": float("nan"),
                "native_metric_max_abs_error": float("nan"),
            }
        native_gt = load_native_gt(payload["gt_path"])
        natural_metrics = _score_pair_metrics(payload["natural"], native_gt)
        clean_metrics = _score_pair_metrics(payload["oracle_clean"], native_gt)
        matched_metrics = []
        for item in payload.get("matched_size_clean", []):
            if not item["valid"]:
                matched_metrics.append(None)
                continue
            if not all(torch.is_tensor(item.get(f"{method}_score")) for method in METHODS):
                raise KeyError("matched-size raw scores are missing; regenerate with --defer_native_metrics")
            matched_metrics.append(_score_pair_metrics(item, native_gt))

        raw_error = metric_error = float("nan")
        if task.get("reference_path"):
            reference = load_torch(task["reference_path"])
            raw_error = _raw_score_reproduction(payload, reference)
            metric_error = max(
                _metric_error(natural_metrics, reference["oracle_clean"]["natural_native_metrics"]),
                _metric_error(clean_metrics, reference["oracle_clean"]["clean_native_metrics"]),
                max((
                    _metric_error(metrics, reference["matched_size_clean"][index]["native_metrics"])
                    for index, metrics in enumerate(matched_metrics) if metrics is not None
                ), default=0.0),
            )

        payload["oracle_clean"]["natural_native_metrics"] = natural_metrics
        payload["oracle_clean"]["clean_native_metrics"] = clean_metrics
        for item, metrics in zip(payload.get("matched_size_clean", []), matched_metrics):
            if metrics is not None:
                item["native_metrics"] = metrics
        payload["native_metrics_complete"] = True
        payload["native_metrics_evaluation"] = {
            "version": VERSION,
            "completed_at": _now(),
            "protocol": "raw_37_to_68_to_native_bilinear; exact per-image AP/AUROC",
            "cpu_process_threads": 1,
            "raw_scores_modified": False,
        }
        _atomic_save(payload, path)
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(path),
            "skipped": False, "runtime_seconds": time.perf_counter() - started,
            "raw_score_max_abs_error": raw_error,
            "native_metric_max_abs_error": metric_error,
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "cache_path": str(path), "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker() -> None:
    torch.set_num_threads(1)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    cache_root = Path(args.cache_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else cache_root / "native_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(cache_root, split=args.split, max_samples=args.max_samples)
    reference_index = {}
    if args.reference_root:
        reference_index = index_manifest(load_manifest(
            args.reference_root, split=args.split, max_samples=-1
        ))
    tasks = []
    for row in rows:
        key = (normalize_dataset(row["dataset"]), str(row["stem"]))
        reference = reference_index.get(key)
        if args.reference_root and reference is None:
            raise KeyError(f"reference cache missing identity {key}")
        tasks.append({
            "dataset": key[0], "stem": key[1], "cache_path": row["cache_path"],
            "reference_path": reference["cache_path"] if reference else None,
            "overwrite": bool(args.overwrite),
        })

    results, failures = [], []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=context, initializer=_init_worker
    ) as pool:
        for index, result in enumerate(pool.map(_evaluate_one, tasks, chunksize=1), 1):
            (failures if "error" in result else results).append(result)
            if index % max(1, args.progress_every) == 0 or index == len(tasks):
                print(
                    f"[{index}/{len(tasks)}] evaluated={len(results)} failed={len(failures)} "
                    f"workers={args.workers}", flush=True,
                )

    write_csv(out_dir / "native_metric_reproduction.csv", results)
    write_json(out_dir / "failures.json", failures)
    counts = Counter(row["dataset"] for row in results)
    raw_errors = [float(row["raw_score_max_abs_error"]) for row in results if row["raw_score_max_abs_error"] == row["raw_score_max_abs_error"]]
    metric_errors = [float(row["native_metric_max_abs_error"]) for row in results if row["native_metric_max_abs_error"] == row["native_metric_max_abs_error"]]
    complete = len(rows) == 6473 and dict(counts) == EXPECTED_COUNTS and not failures
    validity = {
        "version": VERSION,
        "requested": len(rows), "evaluated": len(results), "failed": len(failures),
        "workers": int(args.workers), "dataset_counts": dict(counts),
        "formal_full6473": complete, "native_metrics_complete": not failures,
        "reference_root": str(Path(args.reference_root).resolve()) if args.reference_root else None,
        "raw_score_max_abs_error": max(raw_errors, default=None),
        "native_metric_max_abs_error": max(metric_errors, default=None),
        "raw_scores_modified": False,
    }
    write_json(out_dir / "validity_summary.json", validity)
    print(json.dumps(validity, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} native evaluations failed; see {out_dir / 'failures.json'}")
    if args.reference_root and (
        validity["raw_score_max_abs_error"] != 0.0
        or validity["native_metric_max_abs_error"] != 0.0
    ):
        raise RuntimeError("deferred pipeline did not exactly reproduce the reference")


if __name__ == "__main__":
    main()
