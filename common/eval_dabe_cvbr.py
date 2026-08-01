#!/usr/bin/env python3
"""Formal original-GT evaluation and mechanism audits for DABE CVBR-v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_rank_calibration import average_percentile_rank  # noqa: E402
from common.eval_dabe_background_null import (  # noqa: E402
    MetricAggregate,
    _difference_panel,
    _git_metadata,
    _gray_panel,
    _load_gt,
    _manifest_map,
    _raw_metrics,
    _sha256,
    _write_csv,
)
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
    _resize_current,
    _resize_native,
)
from common.utils import load_config, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
VERSION = "dabe_cvbr_v1"
METHODS = (
    "B0-R1-BW2",
    "B1-R1-BW1",
    "V1-CVBR-SecondRing",
    "V2-CVBR-AllBorder",
    "Current-DABE-v2",
)
FIELDS_MAP = {
    METHODS[0]: "b0_r1_bw2_37",
    METHODS[1]: "b1_r1_bw1_37",
    METHODS[2]: "v1_cvbr_second_ring_37",
    METHODS[3]: "v2_cvbr_all_border_37",
}
ANCHOR_MAP = {
    METHODS[0]: "anchor_b0_37",
    METHODS[1]: "anchor_b1_37",
    METHODS[2]: "anchor_v1_37",
    METHODS[3]: "anchor_v2_37",
}
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
SUBSETS = ("Touch-1", "Touch-2-only", "Non-touch")
SCOPES = ("ring1", "ring2", "all_border")
PAIRWISE = (
    (METHODS[1], METHODS[0]),
    (METHODS[2], METHODS[0]),
    (METHODS[3], METHODS[0]),
    (METHODS[2], METHODS[1]),
    (METHODS[3], METHODS[1]),
    (METHODS[3], METHODS[2]),
    (METHODS[0], METHODS[4]),
)
HARD = (
    "hard_S_m", "hard_F_beta_w", "hard_F_beta_mean", "hard_E_mean",
    "hard_MAE", "hard_IoU", "hard_Precision", "hard_Recall",
    "hard_Area", "hard_Components",
)
OFFICIAL = (
    "official_soft_S_m", "official_soft_F_beta_w", "official_soft_F_beta_mean",
    "official_soft_F_beta_max", "official_soft_E_mean", "official_soft_E_max",
    "official_soft_MAE",
)
RAW = (
    "raw_MAE", "raw_Brier", "raw_SoftPrecision", "raw_SoftRecall",
    "raw_SoftIoU", "raw_prob_mean", "raw_prob_std",
)
RANK = (
    "pixel_AP", "best_IoU_256", "best_IoU_threshold_256",
    "ranking_F_beta_max", "ranking_E_max",
)
PAIR_FIELDS = (
    "delta_hard_S_m", "delta_hard_F_beta_w", "delta_hard_E_mean",
    "delta_hard_MAE", "delta_hard_IoU", "delta_hard_Precision",
    "delta_hard_Recall", "delta_hard_Area",
    "delta_official_soft_F_beta_w", "delta_raw_MAE", "delta_pixel_AP",
    "delta_best_IoU_256",
)
PAIR_SOURCE = {
    "delta_hard_S_m": "hard_S_m",
    "delta_hard_F_beta_w": "hard_F_beta_w",
    "delta_hard_E_mean": "hard_E_mean",
    "delta_hard_MAE": "hard_MAE",
    "delta_hard_IoU": "hard_IoU",
    "delta_hard_Precision": "hard_Precision",
    "delta_hard_Recall": "hard_Recall",
    "delta_hard_Area": "hard_Area",
    "delta_official_soft_F_beta_w": "official_soft_F_beta_w",
    "delta_raw_MAE": "raw_MAE",
    "delta_pixel_AP": "pixel_AP",
    "delta_best_IoU_256": "best_IoU_256",
}
SOURCE_SIGNALS = (
    "cv_error", "unreliability", "support_norm", "weight_entropy",
    "color_dispersion",
)
SOURCE_STATS = (
    "cv_error_mean", "cv_error_median", "q_mean", "q_median",
    "q_below_09_ratio", "q_below_05_ratio", "q_below_01_ratio",
    "support_norm_mean", "weight_entropy_mean", "color_dispersion_mean",
)
CORRELATION_FIELDS = (
    "ring2_mean_unreliability", "ring2_q_below_05_ratio",
    "ring1_mean_unreliability", "ring1_q_below_05_ratio",
    "base_border_fg_absorbed", "base_anchor_fg_absorbed",
)
GEN_FIELDS = (
    "b0_cached_r1_max_abs", "b0_detail_vs_cached_max_abs",
    "weighted_dijkstra_unit_reliability_max_abs", "ring1_count",
    "ring2_only_count", "ring2_full_count", "bw1_anchor_count",
    "bw2_anchor_count", "v1_anchor_count", "v2_anchor_count",
    "boundary_cv_error_min", "boundary_cv_error_mean",
    "boundary_cv_error_median", "boundary_cv_error_max",
    "boundary_cv_error_std", "ring1_cv_error_mean",
    "ring1_cv_error_median", "ring2_cv_error_mean",
    "ring2_cv_error_median", "reference_median", "reference_mad",
    "reference_scale", "q_v1_min", "q_v1_mean", "q_v1_median",
    "q_v1_max", "q_v2_min", "q_v2_mean", "q_v2_median", "q_v2_max",
    "ring2_q_mean", "ring2_q_below_09_ratio",
    "ring2_q_below_05_ratio", "ring2_q_below_01_ratio",
    "ring1_q_v2_mean", "ring1_q_v2_below_09_ratio",
    "ring1_q_v2_below_05_ratio", "ring1_q_v2_below_01_ratio",
    "effective_source_mass_b0", "effective_source_mass_b1",
    "effective_source_mass_v1", "effective_source_mass_v2",
    "cross_fallback_count", "b0_area_gt_05", "b1_area_gt_05",
    "v1_area_gt_05", "v2_area_gt_05", "bc_b0_mean", "bc_b1_mean",
    "bc_v1_mean", "bc_v2_mean", "anchor_overlap_v1_b0",
    "anchor_overlap_v1_b1", "anchor_overlap_v2_b0",
    "anchor_overlap_v2_b1",
)
BASELINE = {
    METHODS[0]: {
        "hard_S_m": .7321127747158344,
        "hard_F_beta_w": .6242804301164221,
        "hard_E_mean": .8279479346485146,
        "hard_MAE": .08046898620831737,
        "hard_Precision": .7069664740758146,
        "hard_Recall": .7124157409732799,
        "hard_Area": .1260482386630983,
    },
    METHODS[4]: {
        "hard_S_m": .7031564688307831,
        "hard_F_beta_w": .5722451313959636,
        "hard_E_mean": .7743040501812801,
        "hard_MAE": .0939140360866542,
    },
}


def _finite_mean(values):
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan"), len(selected)


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.reshape(-1).bool()
    scores = scores.reshape(-1).float()
    positive, negative = labels, ~labels
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = average_percentile_rank(scores) * (scores.numel() - 1) + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _touch(gt37: torch.Tensor) -> dict:
    gt = gt37.squeeze().bool()
    top, bottom = bool(gt[0].any()), bool(gt[-1].any())
    left, right = bool(gt[:, 0].any()), bool(gt[:, -1].any())
    touch1 = top or bottom or left or right
    touch2 = bool(gt[1].any() or gt[-2].any() or gt[:, 1].any() or gt[:, -2].any())
    subset = "Touch-1" if touch1 else ("Touch-2-only" if touch2 else "Non-touch")
    return {
        "touch_subset": subset,
        "touch_top": top,
        "touch_bottom": bottom,
        "touch_left": left,
        "touch_right": right,
        "touch_side_count": sum((top, bottom, left, right)),
        "touch_corner": bool(gt[0, 0] or gt[0, -1] or gt[-1, 0] or gt[-1, -1]),
    }


def _source_vector(payload: dict, gt37: torch.Tensor, mask: torch.Tensor) -> dict:
    use = mask.reshape(-1).bool()
    label = gt37.reshape(-1).bool()[use]
    error = payload["boundary_cv_error_37"].reshape(-1).float()[use]
    q = payload["source_q_v2_37"].reshape(-1).float()[use]
    support = payload["boundary_support_norm_37"].reshape(-1).float()[use]
    entropy = payload["boundary_weight_entropy_37"].reshape(-1).float()[use]
    dispersion = payload["boundary_color_dispersion_37"].reshape(-1).float()[use]
    return {
        "label": label.numpy().astype(np.uint8, copy=False),
        "cv_error": error.numpy().astype(np.float32, copy=False),
        "q": q.numpy().astype(np.float32, copy=False),
        "support_norm": support.numpy().astype(np.float32, copy=False),
        "weight_entropy": entropy.numpy().astype(np.float32, copy=False),
        "color_dispersion": dispersion.numpy().astype(np.float32, copy=False),
    }


def _source_audit(payload: dict, gt37: torch.Tensor):
    masks = {
        "ring1": payload["border_ring1_37"].bool(),
        "ring2": payload["border_ring2_only_37"].bool(),
        "all_border": payload["border_ring2_full_37"].bool(),
    }
    vectors, auc_rows = {}, []
    for scope, mask in masks.items():
        vector = _source_vector(payload, gt37, mask)
        vectors[scope] = vector
        label = torch.from_numpy(vector["label"]).bool()
        scores = {
            "cv_error": torch.from_numpy(vector["cv_error"]),
            "unreliability": 1.0 - torch.from_numpy(vector["q"]),
            "support_norm": 1.0 - torch.from_numpy(vector["support_norm"]),
            "weight_entropy": torch.from_numpy(vector["weight_entropy"]),
            "color_dispersion": torch.from_numpy(vector["color_dispersion"]),
        }
        auc_rows.append({
            "boundary_scope": scope,
            **{f"{name}_auc": _auc(label, value) for name, value in scores.items()},
        })
    return vectors, auc_rows


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _evaluate_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    source_path, cvbr_path = Path(task["source"]), Path(task["cvbr"])
    source = torch_load(source_path, map_location="cpu")
    cvbr = torch_load(cvbr_path, map_location="cpu")
    if any(payload.get("dataset") != dataset or payload.get("stem") != stem for payload in (source, cvbr)):
        raise RuntimeError(f"cache key mismatch: {dataset}/{stem}")
    if cvbr.get("cvbr_version") != VERSION:
        raise RuntimeError(f"CVBR version mismatch: {cvbr_path}")
    gt = _load_gt(task["gt"])
    shape = tuple(gt.shape[-2:])
    native = {method: _load_response(cvbr, field, (37, 37), cvbr_path) for method, field in FIELDS_MAP.items()}
    cached = _load_response(source, "residual_pass1_37", (37, 37), source_path)
    if float((native[METHODS[0]] - cached).abs().max()) > 1e-6:
        raise RuntimeError(f"B0/cached R1 mismatch during evaluation: {dataset}/{stem}")
    probabilities = {method: _resize_native(native[method], shape) for method in METHODS[:4]}
    probabilities[METHODS[4]] = _resize_current(
        _load_response(source, "p_dabe_68", (68, 68), source_path), shape
    )
    context = FastCODContext(gt)
    cod = {}
    for method in METHODS:
        cod.update(context.evaluate_many([
            (method, "hard", probabilities[method]),
            (method, "soft", probabilities[method]),
        ], task["threshold"]))
    metric_rows = []
    for method in METHODS:
        hard, soft = cod[(method, "hard")], cod[(method, "soft")]
        metric_rows.append({
            "dataset": dataset, "stem": stem, "method": method,
            "hard_S_m": hard["S_m"], "hard_F_beta_w": hard["F_beta_w"],
            "hard_F_beta_mean": hard["F_beta_mean"], "hard_E_mean": hard["E_mean"],
            "hard_MAE": hard["MAE"], "hard_IoU": hard["IoU"],
            "hard_Precision": hard["Precision"], "hard_Recall": hard["Recall"],
            "hard_Area": hard["Area"], "hard_Components": hard["Components"],
            "official_soft_S_m": soft["S_m"],
            "official_soft_F_beta_w": soft["F_beta_w"],
            "official_soft_F_beta_mean": soft["F_beta_mean"],
            "official_soft_F_beta_max": soft["F_beta_max"],
            "official_soft_E_mean": soft["E_mean"],
            "official_soft_E_max": soft["E_max"], "official_soft_MAE": soft["MAE"],
            **_raw_metrics(probabilities[method], gt),
            **_ranking_metrics(probabilities[method], gt),
            "ranking_F_beta_max": soft["F_beta_max"], "ranking_E_max": soft["E_max"],
            "source_dabe_cache_path": str(source_path),
            "cvbr_cache_path": "" if method == METHODS[4] else str(cvbr_path),
            "_official_f_curve": soft["f_curve"], "_official_e_curve": soft["e_curve"],
        })
    gt37 = torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > .5
    touch = _touch(gt37)
    source_vectors, source_auc = _source_audit(cvbr, gt37)
    fg_count, bg_count = int(gt37.sum()), int((~gt37).sum())
    contamination = []
    for method, field in ANCHOR_MAP.items():
        anchor = cvbr[field].bool()
        overlap = int((anchor & gt37).sum())
        background = int((anchor & ~gt37).sum())
        contamination.append({
            "method": method, "anchor_leak": overlap / (int(anchor.sum()) + 1e-12),
            "fg_absorbed": overlap / (fg_count + 1e-12),
            "anchor_count": int(anchor.sum()), "anchor_ratio": float(anchor.float().mean()),
            "bg_anchor_coverage": background / (bg_count + 1e-12),
        })
    ring1 = cvbr["border_ring1_37"].bool()
    ring2 = cvbr["border_ring2_only_37"].bool()
    border = cvbr["border_ring2_full_37"].bool()
    q = cvbr["source_q_v2_37"].float()
    base_anchor = cvbr["anchor_b0_37"].bool()
    diagnostic = {
        "ring2_mean_unreliability": float((1 - q[ring2]).mean()),
        "ring2_q_below_05_ratio": float((q[ring2] < .5).float().mean()),
        "ring1_mean_unreliability": float((1 - q[ring1]).mean()),
        "ring1_q_below_05_ratio": float((q[ring1] < .5).float().mean()),
        "base_border_fg_absorbed": float((border & gt37).sum() / (fg_count + 1e-12)),
        "base_anchor_fg_absorbed": float((base_anchor & gt37).sum() / (fg_count + 1e-12)),
    }
    return {
        "dataset": dataset, "stem": stem, "metric_rows": metric_rows,
        "touch": touch, "source_vectors": source_vectors, "source_auc": source_auc,
        "contamination": contamination, "diagnostic": diagnostic,
        "generation": cvbr["diagnostics"],
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _metric_sections(aggregates, datasets, section):
    fields = {"hard": HARD, "official_soft": OFFICIAL, "raw_continuous": RAW, "ranking": RANK}[section]
    by_dataset = []
    for dataset in datasets:
        for method in METHODS:
            result = aggregates[(dataset, method)].result()
            by_dataset.append({
                "scope": "by_dataset", "dataset": dataset, "method": method,
                **{field: result[field] for field in fields},
                "num_samples": result["num_samples"], "ap_valid_count": result["ap_valid_count"],
            })
    sample_overall = []
    for method in METHODS:
        result = aggregates[("ALL", method)].result()
        sample_overall.append({
            "scope": "sample_overall", "dataset": "ALL", "method": method,
            **{field: result[field] for field in fields},
            "num_samples": result["num_samples"], "ap_valid_count": result["ap_valid_count"],
        })
    dataset_macro = []
    for method in METHODS:
        rows = [row for row in by_dataset if row["method"] == method]
        dataset_macro.append({
            "scope": "dataset_macro", "dataset": "ALL", "method": method,
            **{field: float(np.mean([row[field] for row in rows])) for field in fields},
            "num_samples": sum(row["num_samples"] for row in rows),
            "ap_valid_count": sum(row["ap_valid_count"] for row in rows),
        })
    return by_dataset, sample_overall, dataset_macro


def _aggregate_rows(records, keys, fields):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    output = []
    for values, rows in grouped.items():
        result = {key: value for key, value in zip(keys, values)}
        result["num_samples"] = len(rows)
        for field in fields:
            result[field], result[f"{field}_valid_count"] = _finite_mean(row[field] for row in rows)
        output.append(result)
    return output


def _pairwise_summary(records, keys):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in (*keys, "method_a", "method_b"))].append(record)
    output = []
    for values, rows in grouped.items():
        base = {key: value for key, value in zip((*keys, "method_a", "method_b"), values)}
        for field in PAIR_FIELDS:
            array = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            output.append({
                **base, "metric": field, "mean_delta": float(array.mean()),
                "median_delta": float(np.median(array)), "p25_delta": float(np.percentile(array, 25)),
                "p75_delta": float(np.percentile(array, 75)),
                "win_ratio": float(np.mean(array > 1e-6)),
                "tie_ratio": float(np.mean(np.abs(array) <= 1e-6)),
                "loss_ratio": float(np.mean(array < -1e-6)), "valid_count": len(array),
            })
    return output


def _concat_vectors(records, dataset=None, subset=None, scope="all_border"):
    selected = [record for record in records if (dataset in (None, "ALL") or record["dataset"] == dataset) and (subset in (None, "ALL") or record["touch_subset"] == subset)]
    if not selected:
        return None
    fields = ("label", "cv_error", "q", "support_norm", "weight_entropy", "color_dispersion")
    return {field: np.concatenate([record["vectors"][scope][field] for record in selected]) for field in fields}


def _source_statistics(vector: dict) -> dict:
    label = vector["label"].astype(bool)
    result = {"num_sources": int(label.size), "fg_source_count": int(label.sum()), "bg_source_count": int((~label).sum())}
    for prefix, mask in (("fg", label), ("bg", ~label)):
        if not mask.any():
            for field in SOURCE_STATS:
                result[f"{prefix}_{field}"] = float("nan")
            continue
        error, q = vector["cv_error"][mask], vector["q"][mask]
        result.update({
            f"{prefix}_cv_error_mean": float(error.mean()), f"{prefix}_cv_error_median": float(np.median(error)),
            f"{prefix}_q_mean": float(q.mean()), f"{prefix}_q_median": float(np.median(q)),
            f"{prefix}_q_below_09_ratio": float(np.mean(q < .9)),
            f"{prefix}_q_below_05_ratio": float(np.mean(q < .5)),
            f"{prefix}_q_below_01_ratio": float(np.mean(q < .1)),
            f"{prefix}_support_norm_mean": float(vector["support_norm"][mask].mean()),
            f"{prefix}_weight_entropy_mean": float(vector["weight_entropy"][mask].mean()),
            f"{prefix}_color_dispersion_mean": float(vector["color_dispersion"][mask].mean()),
        })
    result["delta_cv_error_mean"] = result["fg_cv_error_mean"] - result["bg_cv_error_mean"]
    result["delta_q_mean"] = result["bg_q_mean"] - result["fg_q_mean"]
    return result


def _source_stats_tables(records, datasets):
    fields = tuple(f"{label}_{field}" for label in ("fg", "bg") for field in SOURCE_STATS) + (
        "delta_cv_error_mean", "delta_q_mean", "num_sources", "fg_source_count", "bg_source_count",
    )
    by_dataset, by_subset = [], []
    for dataset in (*datasets, "ALL"):
        for scope in SCOPES:
            vector = _concat_vectors(records, dataset=dataset, scope=scope)
            if vector is not None:
                by_dataset.append({"dataset": dataset, "boundary_scope": scope, **_source_statistics(vector)})
        for subset in SUBSETS:
            for scope in SCOPES:
                vector = _concat_vectors(records, dataset=dataset, subset=subset, scope=scope)
                if vector is not None:
                    by_subset.append({"dataset": dataset, "touch_subset": subset, "boundary_scope": scope, **_source_statistics(vector)})
    return by_dataset, by_subset, fields


def _top10_tables(records, datasets):
    output = []
    group_specs = [(dataset, "ALL") for dataset in (*datasets, "ALL")]
    group_specs.extend((dataset, subset) for dataset in (*datasets, "ALL") for subset in SUBSETS)
    for dataset, subset in group_specs:
        for scope in SCOPES:
            vector = _concat_vectors(records, dataset=dataset, subset=subset, scope=scope)
            if vector is None:
                continue
            labels, errors = vector["label"].astype(bool), vector["cv_error"]
            top_count = max(1, int(math.ceil(.1 * labels.size)))
            top_index = np.argsort(-errors, kind="stable")[:top_count]
            base = float(labels.mean())
            top = float(labels[top_index].mean())
            output.append({
                "dataset": dataset, "touch_subset": subset, "boundary_scope": scope,
                "base_contamination": base, "top10_contamination": top,
                "top10_enrichment": top / (base + 1e-12),
                "num_sources": int(labels.size), "top10_source_count": top_count,
            })
    return output


def _average_rank(values: np.ndarray) -> np.ndarray:
    return average_percentile_rank(torch.from_numpy(values).double()).numpy()


def _correlation(left, right, rank=False):
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan"), int(x.size)
    if rank:
        x, y = _average_rank(x), _average_rank(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return (float(np.dot(x, y) / denominator) if denominator else float("nan")), int(x.size)


def _diagnostic_correlations(records, datasets):
    output = []
    for dataset in (*datasets, "ALL"):
        rows = records if dataset == "ALL" else [row for row in records if row["dataset"] == dataset]
        for method, delta_field in ((METHODS[2], "delta_v1_iou"), (METHODS[3], "delta_v2_iou")):
            for diagnostic in CORRELATION_FIELDS:
                pearson, count = _correlation([row[diagnostic] for row in rows], [row[delta_field] for row in rows])
                spearman, _ = _correlation([row[diagnostic] for row in rows], [row[delta_field] for row in rows], rank=True)
                output.append({
                    "dataset": dataset, "method": method, "diagnostic": diagnostic,
                    "target": "delta_hard_IoU_vs_B0", "pearson": pearson,
                    "spearman": spearman, "valid_count": count,
                })
    return output


def _save_one_visual(task: dict, output_path: Path):
    cvbr = torch_load(task["cvbr"], map_location="cpu")
    gt = _load_gt(task["gt"])
    size, label_height, columns = 150, 22, 5
    with Image.open(task["image"]) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    b0, b1, v1, v2 = (cvbr[FIELDS_MAP[method]] for method in METHODS[:4])
    panels = [
        ("RGB", rgb), ("GT", _gray_panel(gt, size, True)),
        ("B0 R1", _gray_panel(b0, size)), ("B1 BW1", _gray_panel(b1, size)),
        ("V1 CVBR-SR", _gray_panel(v1, size)), ("V2 CVBR-All", _gray_panel(v2, size)),
        ("B0 BC", _gray_panel(cvbr["bc_b0_37"], size)),
        ("B1 BC", _gray_panel(cvbr["bc_b1_37"], size)),
        ("V1 BC", _gray_panel(cvbr["bc_v1_37"], size)),
        ("V2 BC", _gray_panel(cvbr["bc_v2_37"], size)),
        ("B0 anchor", _gray_panel(cvbr["anchor_b0_37"], size, True)),
        ("B1 anchor", _gray_panel(cvbr["anchor_b1_37"], size, True)),
        ("V1 anchor", _gray_panel(cvbr["anchor_v1_37"], size, True)),
        ("V2 anchor", _gray_panel(cvbr["anchor_v2_37"], size, True)),
        ("Boundary CV error", _gray_panel(cvbr["boundary_cv_error_37"], size)),
        ("V1 source q", _gray_panel(cvbr["source_q_v1_37"], size)),
        ("V2 source q", _gray_panel(cvbr["source_q_v2_37"], size)),
        ("V1 - B0", _difference_panel(v1 - b0, size)),
        ("V2 - B0", _difference_panel(v2 - b0, size)),
    ]
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * size, rows * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x, y = (index % columns) * size, (index // columns) * (size + label_height)
        canvas.paste(panel, (x, y + label_height)); draw.text((x + 3, y + 4), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_visualizations(tasks, touch_map, delta_map, out_dir, datasets):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["dataset"]].append(task)
    for dataset in datasets:
        for task in grouped[dataset][:8]:
            _save_one_visual(task, out_dir / "vis" / "fixed_first8" / dataset / f"{task['stem']}.png")
        for subset, folder in (("Touch-1", "touch1_first16"), ("Touch-2-only", "touch2only_first16")):
            selected = [task for task in grouped[dataset] if touch_map[(dataset, task["stem"])] == subset][:16]
            for task in selected:
                _save_one_visual(task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png")
        for method, tag in ((METHODS[2], "V1_vs_B0"), (METHODS[3], "V2_vs_B0")):
            ranked = sorted(grouped[dataset], key=lambda task: delta_map[(dataset, task["stem"], method)], reverse=True)
            for folder, selected in ((f"{tag}_top8", ranked[:8]), (f"{tag}_bottom8", ranked[-8:])):
                for task in selected:
                    _save_one_visual(task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png")


def _table(rows, fields):
    def format_value(value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(value)
        try:
            number = float(value)
            return f"{number:.6f}" if math.isfinite(number) else "NaN"
        except (TypeError, ValueError):
            return str(value)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    lines.extend("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def _lookup(rows, **conditions):
    return next(row for row in rows if all(row.get(key) == value for key, value in conditions.items()))


def _assess(summary):
    hard_macro = {row["method"]: row for row in summary["hard"]["dataset_macro"]}
    hard_by = {(row["dataset"], row["method"]): row for row in summary["hard"]["by_dataset"]}
    ranking = {row["method"]: row for row in summary["ranking"]["dataset_macro"]}
    touch = {(row["touch_subset"], row["method"]): row for row in summary["touch_subset_hard"] if row["dataset"] == "ALL"}
    b0 = hard_macro[METHODS[0]]
    candidates = {}
    for method in METHODS[2:4]:
        dataset_delta = {dataset: hard_by[(dataset, method)]["hard_F_beta_w"] - hard_by[(dataset, METHODS[0])]["hard_F_beta_w"] for dataset in DATASETS}
        touch_delta = {subset: touch[(subset, method)]["hard_F_beta_w"] - touch[(subset, METHODS[0])]["hard_F_beta_w"] for subset in SUBSETS}
        synchronous = (
            hard_macro[method]["hard_S_m"] >= b0["hard_S_m"],
            hard_macro[method]["hard_E_mean"] >= b0["hard_E_mean"],
            hard_macro[method]["hard_MAE"] <= b0["hard_MAE"],
            ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"],
        )
        engineering = hard_macro[method]["hard_F_beta_w"] > .624280 and hard_macro[method]["hard_MAE"] <= .080969 and min(dataset_delta.values()) >= -.005
        paper = hard_macro[method]["hard_F_beta_w"] >= .625780 and sum(value >= 0 for value in dataset_delta.values()) >= 3 and min(dataset_delta.values()) >= -.003 and hard_macro[method]["hard_MAE"] <= .080469 and ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"] and max(touch_delta["Touch-1"], touch_delta["Touch-2-only"]) >= .005 and touch_delta["Non-touch"] >= -.002
        clear = hard_macro[method]["hard_F_beta_w"] >= .6270 and sum(value > 0 for value in dataset_delta.values()) >= 3 and (hard_macro[method]["hard_S_m"] >= b0["hard_S_m"] or hard_macro[method]["hard_E_mean"] >= b0["hard_E_mean"]) and hard_macro[method]["hard_MAE"] <= b0["hard_MAE"] and ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"] and max(touch_delta["Touch-1"], touch_delta["Touch-2-only"]) >= .005 and touch_delta["Non-touch"] >= -.002
        strong = hard_macro[method]["hard_F_beta_w"] >= .6290 and min(dataset_delta.values()) >= -.001 and sum(synchronous) >= 3 and touch_delta["Touch-1"] > 0 and touch_delta["Touch-2-only"] > 0 and touch_delta["Non-touch"] >= -.002
        candidates[method] = {
            "dataset_F_beta_w_deltas": dataset_delta, "touch_F_beta_w_deltas": touch_delta,
            "engineering_basic_success": engineering, "paper_retention_success": paper,
            "clear_success": clear, "strong_success": strong,
            "synchronous_improvement_count": sum(synchronous),
        }
    auc = _lookup(summary["boundary_source_auc_by_dataset"], dataset="ALL", boundary_scope="ring2")["cv_error_auc"]
    enrichment = _lookup(summary["boundary_top10_enrichment"], dataset="ALL", touch_subset="ALL", boundary_scope="ring2")["top10_enrichment"]
    signal_supported = bool(auc >= .65 or enrichment >= 2.0)
    paper_methods = [method for method in METHODS[2:4] if candidates[method]["paper_retention_success"]]
    frozen = max(paper_methods, key=lambda method: hard_macro[method]["hard_F_beta_w"]) if paper_methods else METHODS[0]
    v1_gain = hard_macro[METHODS[2]]["hard_F_beta_w"] - b0["hard_F_beta_w"]
    v2_gain = hard_macro[METHODS[3]]["hard_F_beta_w"] - b0["hard_F_beta_w"]
    if candidates[METHODS[2]]["paper_retention_success"] and not candidates[METHODS[3]]["paper_retention_success"]:
        branch = "A: V1达标、V2不达标"
    elif candidates[METHODS[3]]["paper_retention_success"] and hard_macro[METHODS[3]]["hard_F_beta_w"] > hard_macro[METHODS[2]]["hard_F_beta_w"]:
        branch = "B: V2达标且优于V1"
    elif signal_supported and max(v1_gain, v2_gain) <= 0:
        branch = "C: 可靠性信号强，但V1/V2均未超过B0"
    elif not paper_methods and max(candidates[m]["touch_F_beta_w_deltas"]["Touch-1"] for m in METHODS[2:4]) >= .005:
        branch = "D: 仅触边子集改善，整体未达论文标准"
    elif not signal_supported and max(v1_gain, v2_gain) <= 0:
        branch = "E: 可靠性信号弱且V1/V2未提升"
    elif 0 < max(v1_gain, v2_gain) < .0015:
        branch = "F: 轻微超过B0但不足+0.0015"
    else:
        branch = "未落入预声明A-F的单一分支"
    return {
        "source_reliability_signal": "supported" if signal_supported else "weak",
        "ring2_cv_error_auc": auc, "ring2_top10_enrichment": enrichment,
        "candidates": candidates, "result_branch": branch, "frozen_method": frozen,
        "continue_reliability_weighted_background_atom_selection": branch.startswith("C:"),
        "turn_to_high_resolution_patch_to_pixel_projection": branch.startswith("E:"),
    }


def _write_docs(out_dir: Path, summary: dict, full_run: bool):
    (out_dir / "README.md").write_text(
        "# DABE CVBR-v1 evaluation\n\n"
        f"- Scope: {'full 6473-sample four-dataset run' if full_run else 'sanity subset only; no formal conclusion'}。\n"
        "- Candidate generation is GT-free; evaluation only reads frozen candidates.\n"
        "- B0/B1/V1/V2 resize: 37→68→original GT, bilinear, align_corners=False.\n"
        "- Current-DABE-v2 resize: 68→original GT.\n"
        "- Hard uses strict `prob > 0.5`; best threshold is diagnostic only.\n"
        "- 当前 CODMetrics 会对非恒定 continuous prediction 逐图 Min-Max。\n"
        "- Touch labels use nearest-neighbor GT→37 and are evaluation diagnostics only.\n"
        "- `summary.json` is the machine-readable source of all aggregates and decisions.\n",
        encoding="utf-8",
    )
    baseline = summary["baseline_reproduction"]
    hard_by = summary["hard"]["by_dataset"]
    hard_macro = summary["hard"]["dataset_macro"]
    b0_reproduction = baseline["checks"].get(METHODS[0], {}).get("passed", "不适用于 sanity")
    current_reproduction = baseline["checks"].get(METHODS[4], {}).get("passed", "不适用于 sanity")
    lines = [
        "# DABE-TF CVBR-v1 结果", "",
        f"- 评测范围：{'四数据集 6473 样本正式评测' if full_run else 'sanity 子集，禁止形成正式性能结论'}。",
        f"- B0 精确复现 R1：{b0_reproduction}。",
        f"- Current-DABE-v2 精确复现：{current_reproduction}。",
        "- 当前 CODMetrics 会对非恒定 continuous prediction 逐图 Min-Max。", "",
        "## Hard 主表", "",
        _table(hard_by + [{**row, "dataset": "Dataset-Macro"} for row in hard_macro], ("dataset", "method", "hard_S_m", "hard_F_beta_w", "hard_E_mean", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area")), "",
    ]
    if full_run and not baseline.get("passed"):
        lines += ["## 协议失败", "", "B0 或 Current-DABE-v2 未在绝对误差 1e-5 内复现；按预注册协议，不给出 V1/V2 性能结论。", ""]
    if full_run and baseline.get("passed"):
        assessment = summary["success_assessment"]
        source_all = [row for row in summary["boundary_source_statistics_by_dataset"] if row["dataset"] == "ALL"]
        auc_all = [row for row in summary["boundary_source_auc_by_dataset"] if row["dataset"] == "ALL"]
        top_all = [row for row in summary["boundary_top10_enrichment"] if row["dataset"] == "ALL" and row["touch_subset"] == "ALL"]
        gen = summary["generation_diagnostics_overall"][0]
        contam = [row for row in summary["anchor_contamination_by_subset"] if row["dataset"] == "ALL"]
        touch = [row for row in summary["touch_subset_hard"] if row["dataset"] == "ALL"]
        rank = summary["ranking"]["dataset_macro"]
        ring1_stats = _lookup(source_all, boundary_scope="ring1")
        ring2_stats = _lookup(source_all, boundary_scope="ring2")
        all_stats = _lookup(source_all, boundary_scope="all_border")
        ring2_auc = _lookup(auc_all, boundary_scope="ring2")
        ring2_top = _lookup(top_all, boundary_scope="ring2")
        c = {method: _lookup(contam, touch_subset="Touch-1", method=method) for method in METHODS[:4]}
        lines += [
            "## Official Soft Dataset-Macro", "",
            _table(summary["official_soft"]["dataset_macro"], ("method", "official_soft_S_m", "official_soft_F_beta_w", "official_soft_E_mean", "official_soft_MAE")), "",
            "## Ranking Dataset-Macro", "", _table(rank, ("method", "pixel_AP", "best_IoU_256", "best_IoU_threshold_256", "ranking_F_beta_max", "ranking_E_max")), "",
            "## Touch 子集", "", _table(touch, ("touch_subset", "method", "hard_F_beta_w", "hard_Precision", "hard_Recall", "hard_Area", "hard_MAE")), "",
            "## Source AUC 与 Top-10%", "", _table(auc_all, ("boundary_scope", "cv_error_auc", "unreliability_auc", "support_norm_auc", "weight_entropy_auc", "color_dispersion_auc", "valid_group_count")), "", _table(top_all, ("boundary_scope", "base_contamination", "top10_contamination", "top10_enrichment", "num_sources")), "",
            "## 预注册问题逐项回答", "",
            f"1. B0 是否精确复现 R1：是；全部指标复现且候选生成最大误差为 {summary['cache_protocol']['baseline_recompute_max_abs']:.12g}。",
            f"2. Current-DABE-v2 是否精确复现：是；正式主指标均在 1e-5 容差内。",
            f"3. cross fallback 总数：{summary['cache_protocol']['cross_fallback_count']}。",
            f"4. 第一圈/第二圈 cross residual：均值 {gen['ring1_cv_error_mean']:.6f}/{gen['ring2_cv_error_mean']:.6f}，中位数 {gen['ring1_cv_error_median']:.6f}/{gen['ring2_cv_error_median']:.6f}。",
            f"5. GT 前景相对背景 cross residual：全边界 FG={all_stats['fg_cv_error_mean']:.6f}、BG={all_stats['bg_cv_error_mean']:.6f}、差值={all_stats['delta_cv_error_mean']:.6f}。",
            f"6. ring1/ring2/all-border cv_error AUC：{_lookup(auc_all,boundary_scope='ring1')['cv_error_auc']:.6f}/{ring2_auc['cv_error_auc']:.6f}/{_lookup(auc_all,boundary_scope='all_border')['cv_error_auc']:.6f}。",
            f"7. ring2 top-10% contamination enrichment：{ring2_top['top10_enrichment']:.6f}。",
            f"8. ring2 的 cross/support/entropy/color AUC：{ring2_auc['cv_error_auc']:.6f}/{ring2_auc['support_norm_auc']:.6f}/{ring2_auc['weight_entropy_auc']:.6f}/{ring2_auc['color_dispersion_auc']:.6f}；数值最大的信号为 {max(('cross residual',ring2_auc['cv_error_auc']),('support norm',ring2_auc['support_norm_auc']),('entropy',ring2_auc['weight_entropy_auc']),('color dispersion',ring2_auc['color_dispersion_auc']),key=lambda x:x[1])[0]}。",
            f"9. V1 第二圈平均降权：{1-gen['ring2_q_mean']:.6f}（平均 q={gen['ring2_q_mean']:.6f}）。",
            f"10. V2 第一圈/第二圈平均降权：{1-gen['ring1_q_v2_mean']:.6f}/{1-gen['ring2_q_mean']:.6f}。",
            f"11. V1/V2 平均有效 source mass：{gen['effective_source_mass_v1']:.6f}/{gen['effective_source_mass_v2']:.6f}。",
            f"12. Touch-1 FGAbsorbed，B0/V1/V2={c[METHODS[0]]['fg_absorbed']:.6f}/{c[METHODS[2]]['fg_absorbed']:.6f}/{c[METHODS[3]]['fg_absorbed']:.6f}。",
            "13. Touch-2-only Anchor 污染见下表；V1/V2 相对 B0 的差值均已给出。",
            "14. Non-touch 背景覆盖以 `bg_anchor_coverage` 量化；V1/V2 与 B1/B0 的值见下表。",
            "15. V1/V2 的 Precision、Recall、Area 相对 B0 变化见逐方法 Hard 主表和 Pairwise CSV。",
            "16. V1/V2 在 Touch-1、Touch-2-only、Non-touch 的精确数值见 Touch 子集主表。",
            "17. V1/V2 的 Pixel AP 与 Best IoU 见 Ranking Dataset-Macro 表。",
            f"18. 工程基本成功：{[m for m,v in assessment['candidates'].items() if v['engineering_basic_success']]}。",
            f"19. 论文保留标准：{[m for m,v in assessment['candidates'].items() if v['paper_retention_success']]}。",
            f"20. 最终冻结：{assessment['frozen_method']}；结果分支为 {assessment['result_branch']}。",
            f"21. 是否支持继续可靠性加权背景原子选择：{assessment['continue_reliability_weighted_background_atom_selection']}。",
            f"22. 是否转向高分辨率 Patch-to-Pixel Projection：{assessment['turn_to_high_resolution_patch_to_pixel_projection']}。", "",
            "### Anchor 污染与背景覆盖（全部子集）", "", _table(contam, ("touch_subset", "method", "anchor_leak", "fg_absorbed", "bg_anchor_coverage", "anchor_count")), "",
            "### 成功与分支判定", "", "```json", json.dumps(assessment, ensure_ascii=False, indent=2), "```", "",
            "逐数据集/逐子集 Pairwise、Source 统计和所有诊断相关性见同目录 CSV；禁止据此执行样本路由或输出融合。",
        ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_cvbr(
    config_path, dabe_root, cvbr_root, out_dir, split="test", max_samples=-1,
    threshold=.5, save_vis=False, workers=None, torch_threads=1,
    overwrite=False, overwrite_reason="",
):
    started = time.time()
    config_path, dabe_root, cvbr_root, out_dir = map(lambda value: Path(value).resolve(), (config_path, dabe_root, cvbr_root, out_dir))
    if split != "test" or threshold != .5 or max_samples == 0 or max_samples < -1:
        raise ValueError("frozen evaluation protocol violation")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    full_run = max_samples == -1
    source_rows, source_map = _manifest_map(dabe_root / f"manifest_{split}.jsonl", 6473 if full_run else None)
    cvbr_rows, cvbr_map = _manifest_map(cvbr_root / f"manifest_{split}.jsonl", 6473 if full_run else max_samples)
    selected = source_rows if full_run else source_rows[:max_samples]
    keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
    if set(keys) != set(cvbr_map):
        raise RuntimeError("CVBR output keys differ from selected DABE source keys")
    cache_protocol = json.loads((cvbr_root / "protocol.json").read_text(encoding="utf-8"))
    if cache_protocol.get("cvbr_version") != VERSION or cache_protocol.get("source_dabe_manifest_sha256") != _sha256(dabe_root / f"manifest_{split}.jsonl"):
        raise RuntimeError("CVBR cache protocol does not match source manifest")
    datasets = [dataset for dataset in DATASETS if any(key[0] == dataset for key in keys)]
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite {out_dir}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    log_handle = (out_dir / "eval.log").open("w", encoding="utf-8", buffering=1)
    def log(message):
        print(message, flush=True); log_handle.write(str(message) + "\n")
    workers = workers or min(12, os.cpu_count() or 1)
    tasks = []
    for row in selected:
        key = (str(row["dataset"]), str(row["stem"]))
        if not Path(row["gt_path"]).is_file() or not Path(row["image_path"]).is_file():
            raise FileNotFoundError(key)
        tasks.append({
            "dataset": key[0], "stem": key[1], "source": source_map[key]["cache_path"],
            "cvbr": cvbr_map[key]["cache_path"], "gt": row["gt_path"],
            "image": row["image_path"], "threshold": threshold,
        })
    aggregates = {(dataset, method): MetricAggregate() for dataset in (*datasets, "ALL") for method in METHODS}
    touch_metric_rows, touch_metadata, contamination, source_auc, source_records = [], [], [], [], []
    generation, diagnostic_records, pair_records = [], [], []
    touch_map, delta_map, peak_worker = {}, {}, 0.0
    per_fields = ("dataset", "stem", "method", *HARD, *OFFICIAL, *RAW, *RANK, "source_dabe_cache_path", "cvbr_cache_path")
    pair_header = ("dataset", "stem", "touch_subset", "method_a", "method_b", *PAIR_FIELDS)
    try:
        log(f"num_samples = {len(tasks)}"); log(f"workers = {workers}")
        log("hard_threshold = strict > 0.5"); log("candidate_generation_in_eval = false")
        with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as per_handle, (out_dir / "pairwise_per_sample.csv").open("w", newline="", encoding="utf-8") as pair_handle:
            per_writer = csv.DictWriter(per_handle, fieldnames=per_fields, extrasaction="ignore")
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_header, extrasaction="ignore")
            per_writer.writeheader(); pair_writer.writeheader()
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(torch_threads,)) as executor:
                for index, result in enumerate(executor.map(_evaluate_one, tasks, chunksize=1), 1):
                    dataset, stem, touch = result["dataset"], result["stem"], result["touch"]
                    peak_worker = max(peak_worker, result["worker_peak_rss_mb"])
                    metric_map = {row["method"]: row for row in result["metric_rows"]}
                    for row in result["metric_rows"]:
                        aggregates[(dataset, row["method"])].add(row); aggregates[("ALL", row["method"])].add(row)
                        per_writer.writerow(row)
                        touch_metric_rows.append({key: row[key] for key in ("dataset", "stem", "method", *HARD)} | touch)
                    touch_map[(dataset, stem)] = touch["touch_subset"]
                    touch_metadata.append({"dataset": dataset, "stem": stem, **touch})
                    for row in result["contamination"]:
                        contamination.append({"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], **row})
                    for row in result["source_auc"]:
                        source_auc.append({"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], **row})
                    source_records.append({"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], "vectors": result["source_vectors"]})
                    generation.append({"dataset": dataset, "stem": stem, **result["generation"]})
                    diagnostic = {"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], **result["diagnostic"]}
                    diagnostic["delta_v1_iou"] = metric_map[METHODS[2]]["hard_IoU"] - metric_map[METHODS[0]]["hard_IoU"]
                    diagnostic["delta_v2_iou"] = metric_map[METHODS[3]]["hard_IoU"] - metric_map[METHODS[0]]["hard_IoU"]
                    diagnostic_records.append(diagnostic)
                    for method in METHODS[2:4]:
                        delta_map[(dataset, stem, method)] = metric_map[method]["hard_IoU"] - metric_map[METHODS[0]]["hard_IoU"]
                    for method_a, method_b in PAIRWISE:
                        pair = {"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], "method_a": method_a, "method_b": method_b}
                        for delta_field, source_field in PAIR_SOURCE.items():
                            pair[delta_field] = float(metric_map[method_a][source_field]) - float(metric_map[method_b][source_field])
                        pair_records.append(pair); pair_writer.writerow(pair)
                    if index == len(tasks) or index % 100 == 0:
                        log(f"processed = {index}/{len(tasks)}")
        sections = {}
        for section, fields in (("hard", HARD), ("official_soft", OFFICIAL), ("raw_continuous", RAW), ("ranking", RANK)):
            by_dataset, overall, macro = _metric_sections(aggregates, datasets, section)
            sections[section] = {"by_dataset": by_dataset, "sample_overall": overall, "dataset_macro": macro}
            header = ("scope", "dataset", "method", *fields, "num_samples", "ap_valid_count")
            for suffix, rows in (("by_dataset", by_dataset), ("sample_overall", overall), ("dataset_macro", macro)):
                _write_csv(out_dir / f"{section}_{suffix}.csv", rows, header)
        touch_counts = []
        for dataset in (*datasets, "ALL"):
            for subset in SUBSETS:
                touch_counts.append({
                    "dataset": dataset, "touch_subset": subset,
                    "num_samples": sum(1 for row in touch_metadata if (dataset == "ALL" or row["dataset"] == dataset) and row["touch_subset"] == subset),
                })
        touch_hard = []
        for dataset in (*datasets, "ALL"):
            for subset in SUBSETS:
                for method in METHODS:
                    rows = [row for row in touch_metric_rows if (dataset == "ALL" or row["dataset"] == dataset) and row["touch_subset"] == subset and row["method"] == method]
                    touch_hard.append({
                        "dataset": dataset, "touch_subset": subset, "method": method,
                        **{field: _finite_mean(row[field] for row in rows)[0] for field in HARD},
                        "num_samples": len(rows),
                    })
        _write_csv(out_dir / "touch_subset_counts.csv", touch_counts, ("dataset", "touch_subset", "num_samples"))
        _write_csv(out_dir / "touch_subset_hard.csv", touch_hard, ("dataset", "touch_subset", "method", *HARD, "num_samples"))
        _write_csv(out_dir / "touch_metadata_per_sample.csv", touch_metadata, ("dataset", "stem", "touch_subset", "touch_top", "touch_bottom", "touch_left", "touch_right", "touch_side_count", "touch_corner"))
        auc_by_dataset = _aggregate_rows(source_auc + [{**row, "dataset": "ALL"} for row in source_auc], ("dataset", "boundary_scope"), tuple(f"{field}_auc" for field in SOURCE_SIGNALS))
        auc_by_subset = _aggregate_rows(source_auc + [{**row, "dataset": "ALL"} for row in source_auc], ("dataset", "touch_subset", "boundary_scope"), tuple(f"{field}_auc" for field in SOURCE_SIGNALS))
        for row in (*auc_by_dataset, *auc_by_subset):
            row["valid_group_count"] = row["cv_error_auc_valid_count"]
        auc_fields = tuple(f"{field}_auc" for field in SOURCE_SIGNALS)
        auc_header = (*auc_fields, "valid_group_count", "num_samples", *(f"{field}_valid_count" for field in auc_fields))
        _write_csv(out_dir / "boundary_source_auc_by_dataset.csv", auc_by_dataset, ("dataset", "boundary_scope", *auc_header))
        _write_csv(out_dir / "boundary_source_auc_by_subset.csv", auc_by_subset, ("dataset", "touch_subset", "boundary_scope", *auc_header))
        stats_by_dataset, stats_by_subset, stats_fields = _source_stats_tables(source_records, datasets)
        _write_csv(out_dir / "boundary_source_statistics_by_dataset.csv", stats_by_dataset, ("dataset", "boundary_scope", *stats_fields))
        _write_csv(out_dir / "boundary_source_statistics_by_subset.csv", stats_by_subset, ("dataset", "touch_subset", "boundary_scope", *stats_fields))
        top10 = _top10_tables(source_records, datasets)
        _write_csv(out_dir / "boundary_top10_enrichment.csv", top10, ("dataset", "touch_subset", "boundary_scope", "base_contamination", "top10_contamination", "top10_enrichment", "num_sources", "top10_source_count"))
        contamination_fields = ("anchor_leak", "fg_absorbed", "anchor_count", "anchor_ratio", "bg_anchor_coverage")
        contamination_all = contamination + [{**row, "dataset": "ALL"} for row in contamination]
        contamination_by_dataset = _aggregate_rows(contamination_all, ("dataset", "method"), contamination_fields)
        contamination_by_subset = _aggregate_rows(contamination_all, ("dataset", "touch_subset", "method"), contamination_fields)
        contamination_header = (*contamination_fields, "num_samples", *(f"{field}_valid_count" for field in contamination_fields))
        _write_csv(out_dir / "anchor_contamination_by_dataset.csv", contamination_by_dataset, ("dataset", "method", *contamination_header))
        _write_csv(out_dir / "anchor_contamination_by_subset.csv", contamination_by_subset, ("dataset", "touch_subset", "method", *contamination_header))
        generation_all = generation + [{**row, "dataset": "ALL"} for row in generation]
        generation_by_dataset = _aggregate_rows(generation_all, ("dataset",), GEN_FIELDS)
        generation_overall = [row for row in generation_by_dataset if row["dataset"] == "ALL"]
        generation_by_dataset = [row for row in generation_by_dataset if row["dataset"] != "ALL"]
        gen_header = (*GEN_FIELDS, "num_samples", *(f"{field}_valid_count" for field in GEN_FIELDS))
        _write_csv(out_dir / "generation_diagnostics_by_dataset.csv", generation_by_dataset, ("dataset", *gen_header))
        _write_csv(out_dir / "generation_diagnostics_overall.csv", generation_overall, ("dataset", *gen_header))
        correlations = _diagnostic_correlations(diagnostic_records, datasets)
        _write_csv(out_dir / "diagnostic_correlations.csv", correlations, ("dataset", "method", "diagnostic", "target", "pearson", "spearman", "valid_count"))
        pairwise_by_dataset = _pairwise_summary(pair_records, ("dataset",))
        pairwise_subset_records = pair_records + [{**row, "dataset": "ALL"} for row in pair_records]
        pairwise_by_subset = _pairwise_summary(pairwise_subset_records, ("dataset", "touch_subset"))
        pair_summary_header = ("method_a", "method_b", "metric", "mean_delta", "median_delta", "p25_delta", "p75_delta", "win_ratio", "tie_ratio", "loss_ratio", "valid_count")
        _write_csv(out_dir / "pairwise_by_dataset.csv", pairwise_by_dataset, ("dataset", *pair_summary_header))
        _write_csv(out_dir / "pairwise_by_subset.csv", pairwise_by_subset, ("dataset", "touch_subset", *pair_summary_header))
        if save_vis:
            log("saving_visualizations = true")
            _save_visualizations(tasks, touch_map, delta_map, out_dir, datasets)
            log("visualizations_complete = true")
        macro = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
        baseline = {"applicable": full_run, "tolerance": 1e-5, "checks": {}, "passed": None}
        if full_run:
            all_passed = True
            for method, expected_fields in BASELINE.items():
                checks, method_passed = {}, True
                for field, expected in expected_fields.items():
                    actual = float(macro[method][field]); error = abs(actual - expected); passed = error <= 1e-5
                    checks[field] = {"actual": actual, "expected": expected, "absolute_error": error, "passed": passed}
                    method_passed &= passed; all_passed &= passed
                baseline["checks"][method] = {"passed": bool(method_passed), "metrics": checks}
            baseline["passed"] = bool(all_passed)
        elapsed = time.time() - started
        commit, status = _git_metadata()
        protocol = {
            "cvbr_version": VERSION, "split": split, "num_samples": len(tasks),
            "full_run": full_run, "methods": list(METHODS), "hard_threshold": .5,
            "hard_operator": ">", "candidate_generation_in_eval": False,
            "gt_used_for_generation": False, "official_soft_per_image_minmax": True,
            "best_threshold_diagnostic_only": True, "dataset_specific_rule": False,
            "sample_routing_used": False, "output_fusion_used": False,
            "config_path": str(config_path), "config_sha256": _sha256(config_path),
            "source_dabe_manifest": str(dabe_root / f"manifest_{split}.jsonl"),
            "source_dabe_manifest_sha256": _sha256(dabe_root / f"manifest_{split}.jsonl"),
            "cvbr_manifest": str(cvbr_root / f"manifest_{split}.jsonl"),
            "cvbr_manifest_sha256": _sha256(cvbr_root / f"manifest_{split}.jsonl"),
            "cache_protocol_sha256": _sha256(cvbr_root / "protocol.json"),
            "evaluator_sha256": _sha256(SCRIPT_PATH), "git_commit": commit,
            "git_status_short": status, "workers": workers,
            "torch_threads_per_worker": torch_threads, "save_vis": save_vis,
            "elapsed_seconds": elapsed, "average_seconds_per_image": elapsed / len(tasks),
            "main_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "max_worker_peak_rss_mb": peak_worker,
        }
        summary = {
            "protocol": protocol, "cache_protocol": cache_protocol,
            "baseline_reproduction": baseline, **sections,
            "touch_subset_counts": touch_counts, "touch_subset_hard": touch_hard,
            "boundary_source_auc_by_dataset": auc_by_dataset,
            "boundary_source_auc_by_subset": auc_by_subset,
            "boundary_source_statistics_by_dataset": stats_by_dataset,
            "boundary_source_statistics_by_subset": stats_by_subset,
            "boundary_top10_enrichment": top10,
            "anchor_contamination_by_dataset": contamination_by_dataset,
            "anchor_contamination_by_subset": contamination_by_subset,
            "generation_diagnostics_by_dataset": generation_by_dataset,
            "generation_diagnostics_overall": generation_overall,
            "diagnostic_correlations": correlations,
            "pairwise_by_dataset": pairwise_by_dataset,
            "pairwise_by_subset": pairwise_by_subset,
        }
        summary["success_assessment"] = _assess(summary) if baseline.get("passed") else {"not_evaluated": True, "reason": "baseline reproduction failed or sanity-only run"}
        write_json(out_dir / "protocol.json", protocol); write_json(out_dir / "summary.json", summary)
        _write_docs(out_dir, summary, full_run)
        log(f"baseline_reproduction = {baseline.get('passed')}")
        log(f"elapsed_seconds = {elapsed:.3f}"); log(f"average_seconds_per_image = {elapsed / len(tasks):.6f}")
        log(f"main_peak_rss_mb = {protocol['main_peak_rss_mb']:.3f}"); log(f"max_worker_peak_rss_mb = {peak_worker:.3f}")
        return summary
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--cvbr_root", required=True); parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test"); parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=.5); parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_cvbr(
        args.config, args.dabe_root, args.cvbr_root, args.out_dir, args.split,
        args.max_samples, args.threshold, args.save_vis, args.workers,
        args.torch_threads, args.overwrite, args.overwrite_reason,
    )
