#!/usr/bin/env python3
"""Formal original-size evaluation for GBSP-V2 B0--B5 caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_multiotsu, threshold_otsu

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_v2.py"
VERSION = "gbsp_v2_softbc_swor_v1"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
VARIANTS = ("b0", "b1", "b2", "b3", "b4", "b5")
COD_METRICS = ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area", "IoU", "Dice")
CONTINUOUS_METRICS = ("pixel_AP", "pixel_AUROC")


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in seen:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}: {key}")
        seen.add(key)
        for field in ("cache_path", "gt_path"):
            if not row.get(field) or not Path(row[field]).is_file():
                raise FileNotFoundError(row.get(field, f"missing {field}"))
    return rows


def _load_gt(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy((value > .5).astype(np.float32)).unsqueeze(0)


def _resize(value: torch.Tensor, shape: tuple[int, int], *, nearest: bool = False) -> torch.Tensor:
    mode = "nearest" if nearest else "bilinear"
    kwargs = {} if nearest else {"align_corners": False}
    intermediate = F.interpolate(value.unsqueeze(0), size=(68, 68), mode=mode, **kwargs)
    return F.interpolate(intermediate, size=shape, mode=mode, **kwargs).squeeze(0)


def _rank_metrics(score: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray) -> dict[str, float]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(gt).reshape(-1) > .5
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


def _cod(context: FastCODContext, probability: torch.Tensor, threshold: float) -> dict[str, float]:
    value = context.evaluate_many([("candidate", "hard", probability)], float(threshold))[("candidate", "hard")]
    precision, recall = float(value["Precision"]), float(value["Recall"])
    return {
        "S_m": float(value["S_m"]), "F_beta_w": float(value["F_beta_w"]),
        "F_beta_mean": float(value["F_beta_mean"]), "E_mean": float(value["E_mean"]),
        "MAE": float(value["MAE"]), "Precision": precision, "Recall": recall,
        "Area": float(value["Area"]), "IoU": float(value["IoU"]),
        "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
    }


def _quantile(value: torch.Tensor, level: float) -> float:
    if not value.numel():
        return float("nan")
    return float(np.quantile(value.detach().cpu().float().numpy(), level, method="linear"))


def _adaptive_threshold(score37: torch.Tensor, method: str) -> tuple[float, bool]:
    values = score37.detach().cpu().float().numpy().reshape(-1)
    unique = np.unique(values)
    if unique.size < 2:
        return .5, True
    if method == "otsu":
        return float(threshold_otsu(values)), False
    if method == "multi_otsu_3":
        if unique.size < 3:
            return float(threshold_otsu(values)), True
        return float(threshold_multiotsu(values, classes=3)[-1]), False
    raise ValueError(f"unknown adaptive threshold: {method}")


def _candidate_row(task: dict, name: str, item: dict, gt37: torch.Tensor) -> dict:
    indices = item["background_indices"].long().reshape(-1)
    labels = gt37.index_select(0, indices).float()
    rows = torch.div(indices, 37, rounding_mode="floor")
    cols = indices % 37
    interior = (rows >= 2) & (rows < 35) & (cols >= 2) & (cols < 35)
    weights = item["normalized_weights"].float().reshape(-1)
    confidence = item["background_confidence"].float().reshape(-1)
    return {
        "dataset": task["dataset"], "stem": task["stem"], "variant": name,
        "candidate_count": int(indices.numel()),
        "boundary_candidate_count": int(item["boundary_candidate_count"]),
        "interior_candidate_count": int(item["interior_candidate_count"]),
        "boundary_candidate_ratio": float(item["boundary_candidate_ratio"]),
        "candidate_gt_foreground_contamination": float(labels.mean()),
        "internal_candidate_contamination": float(labels[interior].mean()) if bool(interior.any()) else float("nan"),
        "candidate_spatial_depth_mean": float(item["candidate_spatial_depth_mean"]),
        "candidate_spatial_depth_median": float(item["candidate_spatial_depth_median"]),
        "dino_feature_coverage": float(item["feature_coverage"]),
        "candidate_pairwise_cosine_distance": float(item["candidate_pairwise_cosine_distance"]),
        "nearest_neighbor_distance": float(item["nearest_neighbor_distance"]),
        "effective_rank": float(item["effective_rank"]),
        "confidence_mean": float(confidence.mean()),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()), "weight_median": float(weights.median()),
    }


def _process(task: dict) -> dict:
    started = time.perf_counter()
    try:
        continuous_only = bool(task.get("continuous_only", False))
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("gbsp_v2_version") != VERSION:
            raise RuntimeError(f"invalid GBSP-V2 cache: {path}")
        if (payload.get("dataset"), payload.get("stem")) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"identity mismatch: {path}")
        gt = _load_gt(Path(task["gt_path"]))
        native_shape = tuple(gt.shape[-2:])
        gt_bool = gt > .5
        gt37 = (
            F.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").reshape(-1) > .5
            if not continuous_only
            else None
        )
        context = FastCODContext(gt) if not continuous_only else None
        continuous_rows, fixed_rows, adaptive_rows, curve_rows = [], [], [], []
        candidate_rows, random_walk_rows, soft_seed_rows = [], [], []
        coreset_rows, weighted_rows, swor_rows = [], [], []
        pooled_scores: dict[str, np.ndarray] = {}
        shared = payload["shared_diagnostics"] if not continuous_only else None
        boundary = torch.zeros(37, 37, dtype=torch.bool)
        boundary[:2] = boundary[-2:] = True
        boundary[:, :2] = boundary[:, -2:] = True
        boundary_flat = boundary.reshape(-1)
        depth = torch.minimum(
            torch.minimum(torch.div(torch.arange(1369), 37, rounding_mode="floor"), 36 - torch.div(torch.arange(1369), 37, rounding_mode="floor")),
            torch.minimum(torch.arange(1369) % 37, 36 - torch.arange(1369) % 37),
        ).float()

        for name, item in payload.get("results", {}).items():
            raw37 = item["raw_score"].detach().cpu().float()
            probability37 = item["minmax_score"].detach().cpu().float()
            if tuple(raw37.shape) != (1, 37, 37) or tuple(probability37.shape) != (1, 37, 37):
                raise ValueError(f"invalid response shape for {name}: {path}")
            raw_native = _resize(raw37, native_shape)
            rank = _rank_metrics(raw_native.numpy(), gt.numpy())
            if continuous_only:
                continuous_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    **rank,
                })
                continue

            probability_native = _resize(probability37, native_shape)
            foreground, background = raw_native[gt_bool], raw_native[~gt_bool]
            fg_values = {f"fg_q{int(q * 100):02d}": _quantile(foreground, q) for q in (.10, .25, .50, .75, .90)}
            bg_values = {f"bg_q{int(q * 100):02d}": _quantile(background, q) for q in (.50, .75, .90, .95, .99)}
            fg_median, bg_q95 = fg_values["fg_q50"], bg_values["bg_q95"]
            continuous_rows.append({
                "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                **rank, **fg_values, **bg_values,
                "separation_ratio": fg_median / (bg_q95 + 1e-12),
                "fg_median_minus_bg_q95": fg_median - bg_q95,
                "cliffs_delta": 2.0 * rank["pixel_AUROC"] - 1.0,
                "mann_whitney_effect_size": rank["pixel_AUROC"],
                "background_high_residual_ratio": float((probability_native[~gt_bool] >= .5).float().mean()),
                "foreground_low_residual_ratio": float((probability_native[gt_bool] < .5).float().mean()),
            })
            pooled_scores[name] = probability37.reshape(-1).numpy().astype(np.float32)
            candidate_rows.append(_candidate_row(task, name, item, gt37))

            for threshold in task["fixed_thresholds"]:
                fixed_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "protocol": f"fixed_{threshold:.2f}", "threshold": threshold,
                    **_cod(context, probability_native, threshold),
                })
            for method in task["adaptive_methods"]:
                threshold, fallback = _adaptive_threshold(probability37, method)
                adaptive_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "protocol": method, "threshold": threshold, "fallback": int(fallback),
                    **_cod(context, probability_native, threshold),
                })
            for threshold in task["curve_thresholds"]:
                curve_rows.append({
                    "dataset": task["dataset"], "variant": name, "threshold": threshold,
                    **_cod(context, probability_native, threshold),
                })

            if name != "b0":
                rw_key = "hard_random_walk" if name == "b1" else "soft_random_walk"
                confidence_key = "hard_random_walk_confidence" if name == "b1" else "soft_random_walk_confidence"
                rw = shared[rw_key]
                confidence = shared[confidence_key].reshape(-1).float()
                random_walk_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    **rw,
                    "confidence_depth_correlation": float(np.corrcoef(confidence.numpy(), depth.numpy())[0, 1]),
                })
            if name in ("b2", "b3", "b4", "b5"):
                soft_boundary = shared["soft_seed_map"].reshape(-1)[boundary_flat]
                soft_seed_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "inward_consistency_mean": float(shared["inward_consistency"].reshape(-1)[boundary_flat].mean()),
                    "global_dino_density_mean": float(shared["global_dino_density"].reshape(-1)[boundary_flat].mean()),
                    "soft_seed_weight_mean": float(soft_boundary.mean()),
                    "soft_seed_weight_min": float(soft_boundary.min()),
                    "soft_seed_weight_max": float(soft_boundary.max()),
                    "significantly_downweighted_fraction": float((soft_boundary < .5).float().mean()),
                    "boundary_gt_foreground_downweighted_mean": float(soft_boundary[gt37[boundary_flat]].mean()) if bool(gt37[boundary_flat].any()) else float("nan"),
                    "boundary_gt_background_weight_mean": float(soft_boundary[~gt37[boundary_flat]].mean()) if bool((~gt37[boundary_flat]).any()) else float("nan"),
                })
            if name in ("b3", "b4", "b5"):
                coreset_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "candidate_pool_size": int(item["candidate_pool_size"]),
                    "selected_count": int(item["candidate_count"]),
                    "selected_boundary_count": int(item["boundary_candidate_count"]),
                    "selected_interior_count": int(item["interior_candidate_count"]),
                    "nearest_neighbor_distance": float(item["nearest_neighbor_distance"]),
                    "effective_rank": float(item["effective_rank"]),
                    "feature_coverage": float(item["feature_coverage"]),
                })
            if name in ("b4", "b5"):
                angles = item.get("principal_angles_vs_equal_pca", torch.empty(0)).float()
                weighted_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "weight_min": float(item["normalized_weights"].min()),
                    "weight_max": float(item["normalized_weights"].max()),
                    "weight_mean": float(item["normalized_weights"].mean()),
                    "weight_median": float(item["normalized_weights"].median()),
                    "effective_sample_size": float(item["effective_sample_size"]),
                    "weight_concentration_warning": int(bool(item["weight_concentration_warning"])),
                    "mean_shift": float(item.get("mean_shift", float("nan"))),
                    "principal_angle_mean_degrees": float(angles.mean()) if angles.numel() else float("nan"),
                })
            if name == "b5":
                swor_rows.append({
                    "dataset": task["dataset"], "stem": task["stem"], "variant": name,
                    "ledoit_wolf_shrinkage": float(item["ledoit_wolf_shrinkage"]),
                    "covariance_condition_number": float(item["covariance_condition_number"]),
                    "precision_condition_number": float(item["precision_condition_number"]),
                    "swor_min": float(item["swor_min"]), "swor_max": float(item["swor_max"]),
                    "swor_mean": float(item["swor_mean"]), "numerical_validity": int(bool(item["swor_valid"])),
                })

        return {
            "continuous": continuous_rows, "fixed": fixed_rows, "adaptive": adaptive_rows,
            "curve": curve_rows, "candidate": candidate_rows, "random_walk": random_walk_rows,
            "soft_seed": soft_seed_rows, "coreset": coreset_rows, "weighted": weighted_rows,
            "swor": swor_rows, "pooled_scores": pooled_scores,
            "pooled_gt": (
                gt37.numpy().astype(np.uint8)
                if gt37 is not None
                else np.empty(0, dtype=np.uint8)
            ),
            "variant_failures": payload.get("variant_failures", {}),
            "runtime": {"dataset": task["dataset"], "stem": task["stem"], "runtime_seconds": time.perf_counter() - started, "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
        }
    except Exception as error:
        return {"dataset": task.get("dataset", ""), "stem": task.get("stem", ""), "error": repr(error), "traceback": traceback.format_exc()}


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict], group_fields: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output.append({
            **dict(zip(group_fields, key)), "num_samples": len(subset),
            **{metric: _finite_mean(row[metric] for row in subset) for metric in metrics},
        })
    return output


def _with_scopes(rows: list[dict], group_fields: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    dataset_rows = _aggregate(rows, ("dataset",) + group_fields, metrics)
    output = [{"scope": "dataset", **row} for row in dataset_rows]
    variants = sorted({tuple(row[field] for field in group_fields) for row in rows})
    for key in variants:
        selected = [row for row in dataset_rows if tuple(row[field] for field in group_fields) == key]
        if selected:
            output.append({
                "scope": "dataset_macro", "dataset": "ALL", **dict(zip(group_fields, key)),
                "num_samples": sum(int(row["num_samples"]) for row in selected),
                **{metric: _finite_mean(row[metric] for row in selected) for metric in metrics},
            })
    return output


def _threshold_aggregate(accumulator: dict) -> list[dict]:
    rows = []
    for (dataset, variant, threshold), state in sorted(accumulator.items()):
        rows.append({
            "scope": "dataset", "dataset": dataset, "variant": variant, "threshold": threshold,
            "num_samples": state["count"],
            **{metric: state[metric] / state["count"] for metric in COD_METRICS},
        })
    for variant in sorted({row["variant"] for row in rows}):
        for threshold in sorted({row["threshold"] for row in rows if row["variant"] == variant}):
            selected = [row for row in rows if row["variant"] == variant and row["threshold"] == threshold]
            rows.append({
                "scope": "dataset_macro", "dataset": "ALL", "variant": variant, "threshold": threshold,
                "num_samples": sum(row["num_samples"] for row in selected),
                **{metric: _finite_mean(row[metric] for row in selected) for metric in COD_METRICS},
            })
    return rows


def _plateaus(curves: list[dict]) -> list[dict]:
    output = []
    for variant in sorted({row["variant"] for row in curves}):
        rows = sorted((row for row in curves if row["scope"] == "dataset_macro" and row["variant"] == variant), key=lambda row: row["threshold"])
        if not rows:
            continue
        best = {"S_m": max(row["S_m"] for row in rows), "F_beta_w": max(row["F_beta_w"] for row in rows), "E_mean": max(row["E_mean"] for row in rows), "MAE": min(row["MAE"] for row in rows)}
        stable = [row for row in rows if row["S_m"] >= best["S_m"] - .005 and row["F_beta_w"] >= best["F_beta_w"] - .005 and row["E_mean"] >= best["E_mean"] - .005 and row["MAE"] <= best["MAE"] + .003]
        best_thresholds = {f"best_{metric}_threshold": float((max if metric != "MAE" else min)(rows, key=lambda row: row[metric])["threshold"]) for metric in ("S_m", "F_beta_w", "E_mean", "MAE")}
        if stable:
            low, high = stable[0]["threshold"], stable[-1]["threshold"]
            compromise = .5 * (low + high)
        else:
            low = high = float("nan")
            compromise = float(np.median(list(best_thresholds.values())))
        output.append({"variant": variant, **best_thresholds, "compromise_threshold": compromise, "plateau_min": low, "plateau_max": high, "plateau_width": high - low if stable else 0.0, "stable_threshold_count": len(stable)})
    return output


def _bootstrap(continuous: list[dict], fixed: list[dict], repetitions: int, seed: int) -> list[dict]:
    if int(repetitions) <= 0:
        return []
    rng, output = np.random.default_rng(seed), []
    tables = [(continuous, CONTINUOUS_METRICS, "continuous")]
    for protocol in sorted({row["protocol"] for row in fixed}):
        tables.append(([row for row in fixed if row["protocol"] == protocol], ("S_m", "F_beta_w", "E_mean", "MAE"), protocol))
    for rows, metrics, protocol in tables:
        lookup = {(row["dataset"], row["stem"], row["variant"]): row for row in rows}
        for variant in VARIANTS[1:]:
            keys = sorted({(row["dataset"], row["stem"]) for row in rows if (row["dataset"], row["stem"], "b0") in lookup and (row["dataset"], row["stem"], variant) in lookup})
            for metric in metrics:
                arrays = []
                for dataset in DATASETS:
                    values = [float(lookup[(*key, variant)][metric]) - float(lookup[(*key, "b0")][metric]) for key in keys if key[0] == dataset]
                    if metric == "MAE":
                        values = [-value for value in values]
                    if values:
                        arrays.append(np.asarray(values, dtype=np.float64))
                if not arrays:
                    continue
                draws = np.empty(repetitions, dtype=np.float64)
                for index in range(repetitions):
                    draws[index] = np.mean([array[rng.integers(0, array.size, array.size)].mean() for array in arrays])
                output.append({
                    "variant": variant, "baseline": "b0", "protocol": protocol, "metric": metric,
                    "oriented_delta": float(np.mean([array.mean() for array in arrays])),
                    "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                    "repetitions": repetitions, "num_samples": sum(array.size for array in arrays),
                })
    return output


def _pareto(continuous_summary: list[dict], binary050: list[dict]) -> dict:
    continuous = {row["variant"]: row for row in continuous_summary if row["scope"] == "dataset_macro"}
    binary = {row["variant"]: row for row in binary050 if row["scope"] == "dataset_macro"}
    fields = ("pixel_AP", "pixel_AUROC", "S_m", "F_beta_w", "E_mean", "MAE")
    points = []
    for variant in VARIANTS:
        if variant in continuous and variant in binary:
            point = {"variant": variant, **{field: (continuous[variant] if field.startswith("pixel_") else binary[variant])[field] for field in fields}}
            points.append(point)
    front = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            better_or_equal = all(other[field] >= point[field] for field in fields[:-1]) and other["MAE"] <= point["MAE"]
            strictly = any(other[field] > point[field] for field in fields[:-1]) or other["MAE"] < point["MAE"]
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            front.append(point["variant"])
    return {"objectives": list(fields), "points": points, "pareto_front": front, "selection_used_gt": False}


def evaluate(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("GBSP-V2 evaluator supports the test split only")
    cfg = load_config(_resolve(args.config))
    root = _resolve(args.variant_root)
    output = _resolve(args.out_dir)
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("evaluation output must stay outside main code tree")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(root / "manifest_test.jsonl")
    if args.sample_list:
        selected = {tuple(line.strip().split("\t")) for line in _resolve(args.sample_list).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
        manifest = [row for row in manifest if (row["dataset"], row["stem"]) in selected]
    if args.max_samples >= 0:
        manifest = manifest[: args.max_samples]
    if not manifest:
        raise RuntimeError("no samples selected")
    start, end, step = args.threshold_curve_start, args.threshold_curve_end, args.threshold_curve_step
    curve_thresholds = (
        ()
        if args.continuous_only
        else tuple(float(round(value, 10)) for value in np.arange(start, end + .5 * step, step))
    )
    fixed_thresholds = (
        ()
        if args.continuous_only
        else tuple(float(value) for value in (args.thresholds or cfg.GBSP_V2_THRESHOLDS))
    )
    adaptive_methods = (
        ()
        if args.continuous_only
        else tuple(args.simple_adaptive_thresholds or cfg.GBSP_V2_ADAPTIVE_THRESHOLDS)
    )
    tasks = [{
        "dataset": row["dataset"], "stem": row["stem"], "cache_path": row["cache_path"], "gt_path": row["gt_path"],
        "fixed_thresholds": fixed_thresholds, "adaptive_methods": adaptive_methods, "curve_thresholds": curve_thresholds,
        "continuous_only": bool(args.continuous_only),
    } for row in manifest]

    continuous: list[dict] = []
    fixed: list[dict] = []
    adaptive: list[dict] = []
    candidate: list[dict] = []
    random_walk: list[dict] = []
    soft_seed: list[dict] = []
    coreset: list[dict] = []
    weighted: list[dict] = []
    swor: list[dict] = []
    runtime: list[dict] = []
    failures: list[dict] = []
    variant_failures: list[dict] = []
    curve_accumulator: dict[tuple, dict] = {}
    pooled_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    pooled_labels: list[np.ndarray] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)) as pool:
        for index, result in enumerate(pool.map(_process, tasks, chunksize=1), 1):
            if "error" in result:
                failures.append(result)
            else:
                continuous.extend(result["continuous"]); fixed.extend(result["fixed"]); adaptive.extend(result["adaptive"])
                candidate.extend(result["candidate"]); random_walk.extend(result["random_walk"]); soft_seed.extend(result["soft_seed"])
                coreset.extend(result["coreset"]); weighted.extend(result["weighted"]); swor.extend(result["swor"]); runtime.append(result["runtime"])
                pooled_labels.append(result["pooled_gt"])
                for variant, values in result["pooled_scores"].items():
                    pooled_scores[variant].append(values)
                for row in result["curve"]:
                    key = (row["dataset"], row["variant"], row["threshold"])
                    state = curve_accumulator.setdefault(key, {"count": 0, **{metric: 0.0 for metric in COD_METRICS}})
                    state["count"] += 1
                    for metric in COD_METRICS:
                        state[metric] += float(row[metric])
                for variant, detail in result["variant_failures"].items():
                    variant_failures.append({"dataset": result["runtime"]["dataset"], "stem": result["runtime"]["stem"], "variant": variant, **detail})
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP-V2 evaluation: {index}/{len(tasks)}", flush=True)
    if not continuous:
        raise RuntimeError("no valid GBSP-V2 responses were evaluated")

    continuous_summary_metrics = (
        CONTINUOUS_METRICS
        if args.continuous_only
        else CONTINUOUS_METRICS + ("fg_q50", "bg_q95", "separation_ratio", "fg_median_minus_bg_q95", "cliffs_delta", "mann_whitney_effect_size", "background_high_residual_ratio", "foreground_low_residual_ratio")
    )
    continuous_summary = _with_scopes(continuous, ("variant",), continuous_summary_metrics)
    continuous_summary.extend(
        {
            "scope": "image_macro",
            "dataset": "ALL",
            **row,
        }
        for row in _aggregate(continuous, ("variant",), continuous_summary_metrics)
    )
    label_vector = np.concatenate(pooled_labels) if pooled_labels else np.empty(0, dtype=np.uint8)
    for variant, arrays in pooled_scores.items():
        if arrays and label_vector.size == sum(array.size for array in arrays):
            continuous_summary.append({"scope": "pooled_minmax_patch37", "dataset": "ALL", "variant": variant, "num_samples": len(arrays), **_rank_metrics(np.concatenate(arrays), label_vector)})
    fixed_summary = _with_scopes(fixed, ("variant", "protocol", "threshold"), COD_METRICS)
    adaptive_summary = _with_scopes(adaptive, ("variant", "protocol"), COD_METRICS + ("threshold", "fallback"))
    curves = _threshold_aggregate(curve_accumulator)
    plateaus = _plateaus(curves)
    binary050 = [row for row in fixed_summary if abs(float(row["threshold"]) - .50) < 1e-8]
    binary058 = [row for row in fixed_summary if abs(float(row["threshold"]) - .58) < 1e-8]
    residual_distribution = (
        []
        if args.continuous_only
        else _with_scopes(continuous, ("variant",), ("fg_q10", "fg_q25", "fg_q50", "fg_q75", "fg_q90", "bg_q50", "bg_q75", "bg_q90", "bg_q95", "bg_q99", "separation_ratio", "fg_median_minus_bg_q95", "cliffs_delta", "mann_whitney_effect_size", "background_high_residual_ratio", "foreground_low_residual_ratio"))
    )
    bootstrap = _bootstrap(continuous, fixed, args.bootstrap_repetitions, args.bootstrap_seed)
    pareto = _pareto(continuous_summary, binary050)

    expected = dict(cfg.GBSP_V2_EXPECTED_COUNTS)
    observed_counts = Counter(row["dataset"] for row in runtime)
    full = len(runtime) == sum(expected.values()) and dict(observed_counts) == expected and not failures and not variant_failures and all(sum(1 for row in continuous if row["variant"] == variant) == sum(expected.values()) for variant in VARIANTS)
    macro_cont = {row["variant"]: row for row in continuous_summary if row["scope"] == "dataset_macro"}
    macro_050 = {row["variant"]: row for row in binary050 if row["scope"] == "dataset_macro"}
    baseline_rows = []
    baseline_reference = {
        metric: reference
        for metric, reference in cfg.GBSP_V2_BASELINE_REFERENCE.items()
        if not args.continuous_only or metric in CONTINUOUS_METRICS
    }
    for metric, reference in baseline_reference.items():
        observed = macro_cont.get("b0", {}).get(metric) if metric.startswith("pixel_") else macro_050.get("b0", {}).get(metric)
        error = abs(float(observed) - float(reference)) if observed is not None else float("nan")
        baseline_rows.append({
            "metric": metric, "reference": reference, "observed": observed,
            "absolute_error": error, "tolerance": cfg.GBSP_V2_BASELINE_TOLERANCE,
            "applicable": int(full),
            "passed": int(error < cfg.GBSP_V2_BASELINE_TOLERANCE) if full and math.isfinite(error) else "",
            "note": "formal 6473 gate" if full else "not applicable to a Test20 subset",
        })
    numerical_summary = {
        "requested_images": len(tasks), "valid_images": len(runtime), "evaluation_failed": len(failures),
        "variant_failed": len(variant_failures), "full_formal_evaluation": full,
        "evaluation_failures": failures, "variant_failures": variant_failures,
        "rare_failures_were_recorded_without_restarting": True,
    }
    per_dataset = [{"table": "continuous", **row} for row in continuous_summary] + [{"table": "binary", **row} for row in fixed_summary] + [{"table": "adaptive", **row} for row in adaptive_summary]
    _write_csv(output / "baseline_reproduction.csv", baseline_rows)
    _write_csv(output / "full6473_continuous_metrics.csv", continuous_summary)
    _write_csv(output / "full6473_binary_050.csv", binary050)
    _write_csv(output / "full6473_binary_058.csv", binary058)
    _write_csv(output / "simple_adaptive_threshold_metrics.csv", adaptive_summary)
    _write_csv(output / "threshold_curves.csv", curves)
    _write_csv(output / "threshold_plateau_width.csv", plateaus)
    _write_csv(output / "per_dataset_metrics.csv", per_dataset)
    _write_csv(output / "per_image_metrics.csv", continuous)
    _write_csv(output / "per_image_binary_metrics.csv", fixed)
    _write_csv(output / "per_image_adaptive_metrics.csv", adaptive)
    _write_csv(output / "residual_distribution.csv", residual_distribution)
    _write_csv(output / "candidate_statistics.csv", candidate)
    _write_csv(output / "random_walk_statistics.csv", random_walk)
    _write_csv(output / "soft_seed_statistics.csv", soft_seed)
    _write_csv(output / "coreset_statistics.csv", coreset)
    _write_csv(output / "weighted_pca_statistics.csv", weighted)
    _write_csv(output / "swor_statistics.csv", swor)
    _write_csv(output / "bootstrap_ci95.csv", bootstrap)
    _write_csv(output / "runtime_memory.csv", runtime)
    _write_json(output / "pareto_front.json", pareto)
    _write_json(output / "numerical_failure_summary.json", numerical_summary)
    _write_json(output / "evaluation_metadata.json", {
        "schema": "gbsp_v2_eval_v1", "config": str(_resolve(args.config)), "variant_root": str(root),
        "output_dir": str(output), "requested_images": len(tasks), "valid_images": len(runtime),
        "counts": dict(observed_counts), "full_formal_evaluation": full,
        "continuous_only": bool(args.continuous_only),
        "continuous_protocol": "per-image native-size raw AP/AUROC, four-dataset macro; pooled diagnostic uses image-MinMax 37x37 patches" if not args.continuous_only else "per-image native-size raw AP/AUROC, four-dataset macro; no binary metrics or threshold scan",
        "binary_protocol": "disabled" if args.continuous_only else "image MinMax at 37, bilinear 37->68->native, strict threshold",
        "fixed_thresholds": fixed_thresholds, "adaptive_methods": adaptive_methods,
        "threshold_curve": [] if args.continuous_only else [start, end, step], "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed, "gt_used_during_generation": False,
        "gt_used_during_evaluation": True, "wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps(numerical_summary, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and (failures or variant_failures):
        raise RuntimeError(f"evaluation failures={len(failures)}, variant failures={len(variant_failures)}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--variant-root", "--variant_root", dest="variant_root", required=True)
    value.add_argument("--sample-list", "--sample_list", dest="sample_list")
    value.add_argument("--split", default="test")
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--thresholds", nargs="+", type=float)
    value.add_argument("--threshold-curve-start", "--threshold_curve_start", dest="threshold_curve_start", type=float, default=.30)
    value.add_argument("--threshold-curve-end", "--threshold_curve_end", dest="threshold_curve_end", type=float, default=.70)
    value.add_argument("--threshold-curve-step", "--threshold_curve_step", dest="threshold_curve_step", type=float, default=.01)
    value.add_argument("--simple-adaptive-thresholds", "--simple_adaptive_thresholds", dest="simple_adaptive_thresholds", nargs="+", choices=("otsu", "multi_otsu_3"))
    value.add_argument("--bootstrap-repetitions", "--bootstrap_repetitions", dest="bootstrap_repetitions", type=int, default=2000)
    value.add_argument("--bootstrap-seed", "--bootstrap_seed", dest="bootstrap_seed", type=int, default=20260806)
    value.add_argument("--continuous-only", "--continuous_only", dest="continuous_only", action="store_true", help="evaluate native-size AP/AUROC only; skip all binary metrics and threshold scans")
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--strict-failures", action="store_true")
    value.add_argument("--save-threshold-curve", "--save_threshold_curve", action="store_true", help="compatibility flag; curve is always saved")
    value.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
