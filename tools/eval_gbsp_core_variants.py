#!/usr/bin/env python3
"""Formal-size COD evaluation for cached GBSP core variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_core_ablation.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
METRICS = (
    "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall",
    "Area", "IoU", "Dice", "pixel_AP", "pixel_AUROC", "fg_raw_median",
    "fg_raw_q10", "fg_raw_q90", "bg_raw_q50", "bg_raw_q90", "bg_raw_q95",
    "bg_raw_q99", "fg_median_over_bg_q95",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in seen:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}")
        seen.add(key)
        if not Path(row["cache_path"]).is_file() or not Path(row["gt_path"]).is_file():
            raise FileNotFoundError(row)
    return rows


def _load_gt(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((value > .5).astype(np.float32)).unsqueeze(0)


def _resize(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = F.interpolate(value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False)
    return F.interpolate(value, size=shape, mode="bilinear", align_corners=False).squeeze(0)


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    values = score.numpy().astype(np.float64).reshape(-1)
    labels = gt.numpy().reshape(-1) > .5
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}
    order = np.argsort(-values, kind="stable")
    ordered_score, ordered_label = values[order], labels[order]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positives
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    group_pos = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_neg = ends - starts - group_pos
    negative_below = negatives - np.cumsum(group_neg)
    wins = np.sum(group_pos * (negative_below + .5 * group_neg), dtype=np.float64)
    return {"pixel_AP": ap, "pixel_AUROC": float(wins / (positives * negatives))}


def _cod(context: FastCODContext, probability: torch.Tensor, threshold: float) -> dict:
    value = context.evaluate_many([("candidate", "hard", probability)], threshold)[("candidate", "hard")]
    precision, recall = float(value["Precision"]), float(value["Recall"])
    return {
        "S_m": float(value["S_m"]), "F_beta_w": float(value["F_beta_w"]),
        "F_beta_mean": float(value["F_beta_mean"]), "E_mean": float(value["E_mean"]),
        "MAE": float(value["MAE"]), "Precision": precision, "Recall": recall,
        "Area": float(value["Area"]), "IoU": float(value["IoU"]),
        "Dice": 2 * precision * recall / (precision + recall + 1e-12),
    }


def _q(value: torch.Tensor, quantile: float) -> float:
    if not value.numel():
        return float("nan")
    # torch.quantile rejects very large CPU tensors. NumPy's partition-based
    # implementation preserves the exact diagnostic on rare native-size images.
    return float(np.quantile(value.detach().cpu().float().numpy(), quantile, method="linear"))


def _process(task: dict) -> dict:
    started = time.perf_counter()
    try:
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("gbsp_core_version") != "gbsp_core_optimization_v1":
            raise RuntimeError(f"invalid GBSP core cache: {path}")
        if (payload.get("dataset"), payload.get("stem")) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"identity mismatch: {path}")
        gt = _load_gt(Path(task["gt_path"]))
        shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt)
        gt_bool = gt > .5
        gt37 = F.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze().reshape(-1) > .5
        rows, diagnostics = [], []
        for name, item in payload["results"].items():
            raw = item["absolute_raw"].detach().cpu().float()
            calibrated = item["absolute_minmax"].detach().cpu().float()
            if tuple(raw.shape) != (1, 37, 37) or tuple(calibrated.shape) != (1, 37, 37):
                raise ValueError(f"invalid response shape for {name}: {path}")
            raw_native, probability = _resize(raw, shape), _resize(calibrated, shape)
            rank_metrics = _rank_metrics(raw_native, gt)
            fg, bg = raw_native[gt_bool], raw_native[~gt_bool]
            fg_q10, fg_median, fg_q90 = _q(fg, .1), _q(fg, .5), _q(fg, .9)
            bg_q50, bg_q90, bg_q95, bg_q99 = _q(bg, .5), _q(bg, .9), _q(bg, .95), _q(bg, .99)
            indices = item["background_indices"].long().reshape(-1)
            contamination = gt37.index_select(0, indices).float()
            boundary_count = int(item["num_boundary_candidates"])
            interior_mask = torch.ones(indices.numel(), dtype=torch.bool)
            # Payload records counts; identify interior geometrically for contamination.
            row_index, col_index = torch.div(indices, 37, rounding_mode="floor"), indices % 37
            interior_mask = (row_index >= 2) & (row_index < 35) & (col_index >= 2) & (col_index < 35)
            weights = item.get("normalized_weights", torch.ones(indices.numel())).detach().cpu().float().reshape(-1)
            if weights.numel() != indices.numel():
                raise ValueError(f"candidate weight count mismatch for {name}: {path}")
            weight_order = torch.argsort(weights, stable=True)
            weight_quantile_contamination = []
            for chunk in torch.tensor_split(weight_order, 4):
                weight_quantile_contamination.append(float(contamination.index_select(0, chunk).mean()) if chunk.numel() else float("nan"))
            diagnostics.append({
                "dataset": task["dataset"], "stem": task["stem"],
                "experiment": payload["experiment"], "variant": name,
                "method": f"{payload['experiment']}/{name}",
                "selected_rank": int(item["selected_rank"]),
                "retained_variance_ratio": float(item["retained_variance_ratio"]),
                "discarded_variance_ratio": float(item["discarded_variance_ratio"]),
                "num_background_candidates": int(indices.numel()),
                "num_boundary_candidates": boundary_count,
                "num_interior_candidates": int(interior_mask.sum()),
                "candidate_gt_contamination": float(contamination.mean()),
                "candidate_gt_background_coverage": float((~gt37.index_select(0, indices)).sum()) / max(1, int((~gt37).sum())),
                "interior_candidate_gt_contamination": (
                    float(contamination[interior_mask].mean()) if bool(interior_mask.any()) else float("nan")
                ),
                "candidate_connectivity_mean": float(item["candidate_connectivity_mean"]),
                "candidate_path_cost_mean": float(item["candidate_path_cost_mean"]),
                "candidate_percentage": float(item["candidate_percentage"]),
                "background_feature_covariance_trace": float(item["background_feature_covariance_trace"]),
                "spectral_effective_rank": float(item["spectral_effective_rank"]),
                "effective_sample_size": float(item.get("effective_sample_size", indices.numel())),
                "weight_concentration_warning": int(bool(item.get("weight_concentration_warning", False))),
                "path_cost_median": float(item.get("path_cost_median", float("nan"))),
                "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                "weight_mean": float(weights.mean()), "weight_median": float(weights.median()),
                "weight_q1_gt_contamination": weight_quantile_contamination[0],
                "weight_q2_gt_contamination": weight_quantile_contamination[1],
                "weight_q3_gt_contamination": weight_quantile_contamination[2],
                "weight_q4_gt_contamination": weight_quantile_contamination[3],
                "mean_shift_from_equal": float(item.get("mean_shift_from_equal", 0.0)),
                "principal_angle_mean_degrees": (
                    float(item.get("principal_angles_from_equal_degrees", torch.empty(0)).float().mean())
                    if item.get("principal_angles_from_equal_degrees", torch.empty(0)).numel() else float("nan")
                ),
                "source_baseline_max_abs_error": float(payload["source_baseline_max_abs_error"]),
                "boundary_matched_exact": int(bool(payload["boundary_matched_exact"])),
            })
            shared = {
                "dataset": task["dataset"], "stem": task["stem"],
                "experiment": payload["experiment"], "variant": name,
                "method": f"{payload['experiment']}/{name}",
                **rank_metrics, "fg_raw_q10": fg_q10, "fg_raw_median": fg_median,
                "fg_raw_q90": fg_q90, "bg_raw_q50": bg_q50, "bg_raw_q90": bg_q90,
                "bg_raw_q95": bg_q95, "bg_raw_q99": bg_q99,
                "fg_median_over_bg_q95": fg_median / (bg_q95 + 1e-12),
            }
            for threshold in task["thresholds"]:
                rows.append({**shared, "threshold": float(threshold), **_cod(context, probability, threshold)})
        return {
            "rows": rows, "diagnostics": diagnostics,
            "runtime": {
                "dataset": task["dataset"], "stem": task["stem"], "experiment": payload["experiment"],
                "runtime_seconds": time.perf_counter() - started,
                "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            },
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _mean(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    methods = sorted({row["method"] for row in rows})
    thresholds = sorted({float(row["threshold"]) for row in rows})
    for method in methods:
        for threshold in thresholds:
            for dataset in DATASETS:
                subset = [r for r in rows if r["method"] == method and r["dataset"] == dataset and float(r["threshold"]) == threshold]
                if subset:
                    output.append({
                        "scope": "dataset", "dataset": dataset, "method": method,
                        "threshold": threshold, "num_samples": len(subset),
                        **{metric: _mean(r[metric] for r in subset) for metric in METRICS},
                    })
            dataset_rows = [r for r in output if r["scope"] == "dataset" and r["method"] == method and float(r["threshold"]) == threshold]
            if dataset_rows:
                output.append({
                    "scope": "dataset_macro", "dataset": "ALL", "method": method,
                    "threshold": threshold, "num_samples": sum(r["num_samples"] for r in dataset_rows),
                    **{metric: _mean(r[metric] for r in dataset_rows) for metric in METRICS},
                })
    return output


def _bootstrap(rows: list[dict], baselines: dict[str, str], repetitions: int = 2000, seed: int = 20260806) -> list[dict]:
    output, rng = [], np.random.default_rng(seed)
    for experiment, baseline in baselines.items():
        exp_rows = [r for r in rows if r["experiment"] == experiment]
        for threshold in sorted({float(r["threshold"]) for r in exp_rows}):
            keys = sorted({(r["dataset"], r["stem"]) for r in exp_rows if float(r["threshold"]) == threshold})
            lookup = {(r["dataset"], r["stem"], r["variant"]): r for r in exp_rows if float(r["threshold"]) == threshold}
            for variant in sorted({r["variant"] for r in exp_rows} - {baseline}):
                valid_keys = [k for k in keys if (*k, baseline) in lookup and (*k, variant) in lookup]
                if not valid_keys:
                    continue
                for metric in ("pixel_AP", "pixel_AUROC", "S_m", "F_beta_w", "E_mean", "MAE"):
                    if metric in ("pixel_AP", "pixel_AUROC") and threshold != min(float(r["threshold"]) for r in exp_rows):
                        continue
                    deltas = np.array([float(lookup[(*k, variant)][metric]) - float(lookup[(*k, baseline)][metric]) for k in valid_keys], dtype=np.float64)
                    if metric == "MAE":
                        deltas *= -1
                    dataset_arrays = [
                        deltas[[index for index, key in enumerate(valid_keys) if key[0] == dataset]]
                        for dataset in DATASETS
                    ]
                    dataset_arrays = [array for array in dataset_arrays if array.size]
                    draws = np.empty(repetitions, dtype=np.float64)
                    chunk = 64
                    for start in range(0, repetitions, chunk):
                        stop = min(repetitions, start + chunk)
                        means = []
                        for array in dataset_arrays:
                            indices = rng.integers(0, array.size, size=(stop - start, array.size))
                            means.append(array[indices].mean(axis=1))
                        draws[start:stop] = np.stack(means, axis=1).mean(axis=1)
                    point = float(np.mean([array.mean() for array in dataset_arrays]))
                    output.append({
                        "experiment": experiment, "variant": variant, "baseline": baseline,
                        "threshold": threshold, "metric": metric,
                        "oriented_delta": point, "ci_low": float(np.quantile(draws, .025)),
                        "ci_high": float(np.quantile(draws, .975)), "num_samples": len(deltas),
                        "repetitions": repetitions,
                    })
    return output


def evaluate(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("GBSP core evaluator only supports --split test")
    cfg = load_config(_resolve(args.config))
    root_values = list(args.cache_root or []) + list(args.roots or [])
    if not root_values:
        raise ValueError("provide --cache-root or --roots")
    roots = [_resolve(root) for root in root_values]
    output = _resolve(args.out_dir)
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("output directory must stay outside main code tree")
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for root in roots:
        for row in _manifest(root / "manifest_test.jsonl"):
            tasks.append({
                "dataset": row["dataset"], "stem": row["stem"],
                "cache_path": row["cache_path"], "gt_path": row["gt_path"],
                "experiment": row.get("experiment"),
                "thresholds": tuple(float(v) for v in (args.thresholds or cfg.GBSP_CORE_THRESHOLDS)),
            })
    if args.max_samples >= 0:
        tasks = tasks[: args.max_samples]
    if not tasks:
        raise RuntimeError("no cache samples selected")
    all_tasks = list(tasks)
    previous_rows = _read_csv(output / "per_image_metrics.csv") if args.resume else []
    previous_diagnostics = _read_csv(output / "candidate_weight_diagnostics.csv") if args.resume else []
    previous_runtime = _read_csv(output / "runtime_memory.csv") if args.resume else []
    completed = {
        (row.get("dataset"), row.get("stem"), row.get("experiment"))
        for row in previous_runtime
    }
    if args.resume:
        tasks = [task for task in tasks if (task["dataset"], task["stem"], task.get("experiment")) not in completed]
        print(f"GBSP core eval resume: completed={len(completed)}, pending={len(tasks)}", flush=True)
    started, results = time.perf_counter(), []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)) as pool:
            for index, result in enumerate(pool.map(_process, tasks, chunksize=1), 1):
                results.append(result)
                if index % 20 == 0 or index == len(tasks):
                    print(f"GBSP core eval: {index}/{len(tasks)}", flush=True)
    failures = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]
    if not valid and not previous_runtime:
        raise RuntimeError("no valid evaluation samples")
    rows = previous_rows + [row for result in valid for row in result["rows"]]
    diagnostics = previous_diagnostics + [row for result in valid for row in result["diagnostics"]]
    runtime = previous_runtime + [result["runtime"] for result in valid]
    summary = _aggregate(rows)
    experiments = {d["experiment"] for d in diagnostics}
    baseline_by_experiment = {
        key: value for key, value in {
            "rank": "current", "background_source": "fullbc",
            "pca_weight": "equal", "combined": "current",
        }.items() if key in experiments
    }
    identities = {(r["dataset"], r["stem"]) for r in rows}
    full = (
        len(identities) == sum(dict(cfg.GBSP_CORE_EXPECTED_COUNTS).values())
        and len(runtime) == len(all_tasks)
        and not failures
    )
    bootstrap = _bootstrap(rows, baseline_by_experiment, args.bootstrap_repetitions, args.bootstrap_seed) if full else []
    metadata = {
        "schema": "gbsp_core_eval_v1", "config": str(_resolve(args.config)),
        "cache_roots": [str(r) for r in roots], "output_dir": str(output),
        "requested_cache_items": len(all_tasks), "valid_cache_items": len(runtime), "failed": len(failures),
        "resume_previous_valid_cache_items": len(previous_runtime),
        "resume_pending_cache_items": len(tasks),
        "unique_images": len(identities), "full_formal_evaluation": full,
        "raw_metric": "native-size Absolute Raw Pixel AP/AUROC",
        "hard_protocol": "image MinMax at 37, bilinear 37->68->native, strict fixed threshold",
        "thresholds": sorted({float(r["threshold"]) for r in rows}),
        "threshold_search_used": False, "gt_used_during_cache_generation": False,
        "gt_used_during_evaluation": True, "wall_seconds": time.perf_counter() - started,
        "selected_configuration": None,
        "test20_selection_forbidden": not full,
    }
    _write_csv(output / "per_image_metrics.csv", rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "candidate_weight_diagnostics.csv", diagnostics)
    _write_csv(output / "runtime_memory.csv", runtime)
    _write_csv(output / "bootstrap_ci.csv", bootstrap)
    _write_csv(output / "threshold_curve.csv", summary)
    _write_json(output / "per_dataset_metrics.json", {"summary": summary})
    _write_json(output / "evaluation_failures.json", failures)
    _write_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and failures:
        raise RuntimeError(f"{len(failures)} samples failed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--cache-root", "--cache_root", dest="cache_root", action="append")
    value.add_argument("--roots", nargs="+")
    value.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--thresholds", nargs="+", type=float)
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--strict-failures", action="store_true")
    value.add_argument("--resume", action="store_true", help="reuse valid per-cache-item rows already present in out-dir")
    value.add_argument("--save-threshold-curve", "--save_threshold_curve", dest="save_threshold_curve", action="store_true", help="compatibility flag; threshold output is always saved")
    value.add_argument("--bootstrap-repetitions", "--bootstrap_repetitions", dest="bootstrap_repetitions", type=int, default=2000)
    value.add_argument("--bootstrap-seed", "--bootstrap_seed", dest="bootstrap_seed", type=int, default=20260806)
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
