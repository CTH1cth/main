#!/usr/bin/env python3
"""Formal original-GT evaluation and mechanism audit for BGNull-v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
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
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
    _resize_current,
    _resize_native,
)
from common.utils import load_config, read_jsonl, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
METHODS = (
    "N0-R1",
    "N1-CrossR1",
    "N2-LocalNull-R1Dist",
    "RC-NN-MinMax",
    "Current-DABE-v2",
    "N2-LocalNull-Raw",
)
MAIN_METHODS = METHODS[:-1]
DATASET_ORDER = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
PAIRWISE = (
    ("N1-CrossR1", "N0-R1"),
    ("N2-LocalNull-R1Dist", "N0-R1"),
    ("RC-NN-MinMax", "N0-R1"),
    ("N2-LocalNull-R1Dist", "N1-CrossR1"),
    ("N2-LocalNull-R1Dist", "RC-NN-MinMax"),
    ("N2-LocalNull-R1Dist", "Current-DABE-v2"),
)
HARD_FIELDS = (
    "hard_S_m",
    "hard_F_beta_w",
    "hard_F_beta_mean",
    "hard_E_mean",
    "hard_MAE",
    "hard_IoU",
    "hard_Precision",
    "hard_Recall",
    "hard_Area",
    "hard_Components",
)
OFFICIAL_FIELDS = (
    "official_soft_S_m",
    "official_soft_F_beta_w",
    "official_soft_F_beta_mean",
    "official_soft_F_beta_max",
    "official_soft_E_mean",
    "official_soft_E_max",
    "official_soft_MAE",
)
RAW_FIELDS = (
    "raw_MAE",
    "raw_Brier",
    "raw_SoftPrecision",
    "raw_SoftRecall",
    "raw_SoftIoU",
    "raw_prob_mean",
    "raw_prob_std",
)
RANKING_FIELDS = (
    "pixel_AP",
    "best_IoU_256",
    "best_IoU_threshold_256",
    "ranking_F_beta_max",
    "ranking_E_max",
)
PER_SAMPLE_FIELDS = (
    "dataset",
    "stem",
    "method",
    *HARD_FIELDS,
    *OFFICIAL_FIELDS,
    *RAW_FIELDS,
    *RANKING_FIELDS,
    "source_dabe_cache_path",
    "bgnull_cache_path",
)
PAIRWISE_FIELDS = (
    "delta_hard_S_m",
    "delta_hard_F_beta_w",
    "delta_hard_E_mean",
    "delta_hard_MAE",
    "delta_hard_IoU",
    "delta_hard_Precision",
    "delta_hard_Recall",
    "delta_hard_Area",
    "delta_official_soft_F_beta_w",
    "delta_raw_MAE",
    "delta_pixel_AP",
    "delta_best_IoU_256",
)
SWAP_FIELDS = (
    "added_area",
    "removed_area",
    "area_balance_error",
    "added_fg_precision",
    "added_bg_error_ratio",
    "removed_bg_precision",
    "removed_fg_error_ratio",
    "correct_swap_count",
    "incorrect_swap_count",
    "swap_accuracy",
    "hard_disagreement_ratio",
)
INTERNAL_FIELDS = (
    "anchor_count",
    "anchor_ratio",
    "r1_recompute_max_abs",
    "regular_detail_vs_cache_r1_max_abs",
    "regular_detail_vs_original_r1_max_abs",
    "bg_anchor_vs_cached_max_abs",
    "regular_raw_min",
    "regular_raw_mean",
    "regular_raw_max",
    "regular_raw_std",
    "cross_raw_min",
    "cross_raw_mean",
    "cross_raw_max",
    "cross_raw_std",
    "anchor_regular_raw_mean",
    "anchor_cross_raw_mean",
    "anchor_cross_raw_median",
    "anchor_cross_raw_max",
    "cross_minus_regular_mean",
    "cross_minus_regular_anchor_mean",
    "cross_fallback_count",
    "cross_fallback_ratio",
    "regular_top1_self_count",
    "regular_top1_self_ratio_on_anchor",
    "anchor_top1_self_ratio",
    "regular_topk_contains_self_count",
    "regular_topk_contains_self_ratio_on_anchor",
    "local_excluded_weight_mass_mean",
    "local_excluded_weight_mass_anchor_mean",
    "local_excluded_weight_mass_max",
    "local_null_min",
    "local_null_mean",
    "local_null_max",
    "local_null_std",
    "local_null_unique_ratio",
    "local_null_tie_ratio",
    "n0_area_gt_05",
    "n1_area_gt_05",
    "n2_area_gt_05",
    "rc_area_gt_05",
    "n1_area_delta_vs_n0",
    "n2_area_delta_vs_n0",
    "rc_area_delta_vs_n0",
    "n2_sorted_l1_vs_n0",
    "n2_constant_source_fallback",
    "r1_localnull_spearman",
    "r1_crossr1_spearman",
    "r1_nn_spearman",
)
BASELINE_EXACT = {
    "N0-R1": {
        "hard_S_m": 0.7321127747158344,
        "hard_F_beta_w": 0.6242804301164221,
        "hard_E_mean": 0.8279479346485146,
        "hard_MAE": 0.08046898620831737,
        "hard_Precision": 0.7069664740758146,
        "hard_Recall": 0.7124157409732799,
    },
    "Current-DABE-v2": {
        "hard_S_m": 0.7031564688307831,
        "hard_F_beta_w": 0.5722451313959636,
        "hard_E_mean": 0.7743040501812801,
        "hard_MAE": 0.0939140360866542,
    },
}
R1_DATASET_F_W = {
    "CHAMELEON": 0.6088483776633922,
    "TE-CAMO": 0.6332896888559385,
    "TE-COD10K": 0.5717894577162118,
    "NC4K": 0.6831941962301459,
}


def _raw_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict:
    probability = probability.float().clamp(0.0, 1.0)
    target = (target > 0.5).float()
    intersection = float((probability * target).sum())
    pred_mass = float(probability.sum())
    target_mass = float(target.sum())
    union = pred_mass + target_mass - intersection
    eps = 1e-12
    return {
        "raw_MAE": float(torch.abs(probability - target).mean()),
        "raw_Brier": float(torch.square(probability - target).mean()),
        "raw_SoftPrecision": intersection / (pred_mass + eps),
        "raw_SoftRecall": intersection / (target_mass + eps),
        "raw_SoftIoU": intersection / (union + eps),
        "raw_prob_mean": float(probability.mean()),
        "raw_prob_std": float(probability.std(unbiased=False)),
    }


def _load_gt(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _swap_metrics(n0: torch.Tensor, n2: torch.Tensor, gt: torch.Tensor, threshold: float):
    n0_mask = n0 > threshold
    n2_mask = n2 > threshold
    gt_fg = gt > 0.5
    added = n2_mask & ~n0_mask
    removed = n0_mask & ~n2_mask
    added_count = int(added.sum())
    removed_count = int(removed.sum())
    total = int(gt.numel())
    added_fg = int((added & gt_fg).sum())
    added_bg = added_count - added_fg
    removed_fg = int((removed & gt_fg).sum())
    removed_bg = removed_count - removed_fg
    changed = added_count + removed_count
    correct = added_fg + removed_bg
    incorrect = added_bg + removed_fg
    return {
        "added_area": added_count / total,
        "removed_area": removed_count / total,
        "area_balance_error": abs(added_count - removed_count) / total,
        "added_fg_precision": added_fg / added_count if added_count else float("nan"),
        "added_bg_error_ratio": added_bg / added_count if added_count else float("nan"),
        "removed_bg_precision": removed_bg / removed_count if removed_count else float("nan"),
        "removed_fg_error_ratio": removed_fg / removed_count if removed_count else float("nan"),
        "correct_swap_count": correct,
        "incorrect_swap_count": incorrect,
        "swap_accuracy": correct / changed if changed else float("nan"),
        "hard_disagreement_ratio": changed / total,
    }


def _init_worker(torch_threads: int):
    torch.set_num_threads(torch_threads)


def _evaluate_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    source_path = Path(task["source_dabe_cache_path"])
    bgnull_path = Path(task["bgnull_cache_path"])
    source = torch_load(source_path, map_location="cpu")
    bgnull = torch_load(bgnull_path, map_location="cpu")
    if source.get("dataset") != dataset or source.get("stem") != stem:
        raise RuntimeError(f"Source key mismatch: {source_path}")
    if bgnull.get("dataset") != dataset or bgnull.get("stem") != stem:
        raise RuntimeError(f"BGNull key mismatch: {bgnull_path}")
    if bgnull.get("bgnull_version") != "dabe_bgnull_v1":
        raise RuntimeError(f"Wrong BGNull version: {bgnull_path}")
    diagnostics = bgnull.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError(f"Missing diagnostics: {bgnull_path}")
    for field in INTERNAL_FIELDS:
        if field not in diagnostics or diagnostics[field] is None:
            raise KeyError(f"Diagnostic {field} missing from {bgnull_path}")

    gt = _load_gt(task["gt_path"])
    gt_shape = tuple(gt.shape[-2:])
    probabilities = {
        "N0-R1": _resize_native(_load_response(bgnull, "n0_r1_37", (37, 37), bgnull_path), gt_shape),
        "N1-CrossR1": _resize_native(_load_response(bgnull, "n1_cross_r1_37", (37, 37), bgnull_path), gt_shape),
        "N2-LocalNull-R1Dist": _resize_native(_load_response(bgnull, "n2_local_null_r1dist_37", (37, 37), bgnull_path), gt_shape),
        "RC-NN-MinMax": _resize_native(_load_response(bgnull, "rc_nn_minmax_37", (37, 37), bgnull_path), gt_shape),
        "Current-DABE-v2": _resize_current(_load_response(source, "p_dabe_68", (68, 68), source_path), gt_shape),
        "N2-LocalNull-Raw": _resize_native(_load_response(bgnull, "n2_local_null_raw_37", (37, 37), bgnull_path), gt_shape),
    }
    context = FastCODContext(gt)
    cod_inputs = []
    for method in METHODS:
        cod_inputs.extend(
            ((method, "hard", probabilities[method]), (method, "soft", probabilities[method]))
        )
    cod = context.evaluate_many(cod_inputs, task["threshold"])

    rows = []
    for method in METHODS:
        hard = cod[(method, "hard")]
        soft = cod[(method, "soft")]
        ranking = _ranking_metrics(probabilities[method], gt)
        row = {
            "dataset": dataset,
            "stem": stem,
            "method": method,
            "hard_S_m": hard["S_m"],
            "hard_F_beta_w": hard["F_beta_w"],
            "hard_F_beta_mean": hard["F_beta_mean"],
            "hard_E_mean": hard["E_mean"],
            "hard_MAE": hard["MAE"],
            "hard_IoU": hard["IoU"],
            "hard_Precision": hard["Precision"],
            "hard_Recall": hard["Recall"],
            "hard_Area": hard["Area"],
            "hard_Components": hard["Components"],
            "official_soft_S_m": soft["S_m"],
            "official_soft_F_beta_w": soft["F_beta_w"],
            "official_soft_F_beta_mean": soft["F_beta_mean"],
            "official_soft_F_beta_max": soft["F_beta_max"],
            "official_soft_E_mean": soft["E_mean"],
            "official_soft_E_max": soft["E_max"],
            "official_soft_MAE": soft["MAE"],
            **_raw_metrics(probabilities[method], gt),
            **ranking,
            "ranking_F_beta_max": soft["F_beta_max"],
            "ranking_E_max": soft["E_max"],
            "source_dabe_cache_path": str(source_path),
            "bgnull_cache_path": "" if method == "Current-DABE-v2" else str(bgnull_path),
            "_official_f_curve": soft["f_curve"],
            "_official_e_curve": soft["e_curve"],
        }
        for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, *RANKING_FIELDS):
            value = float(row[field])
            if field == "pixel_AP" and math.isnan(value):
                continue
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric: {dataset}/{stem}/{method}/{field}")
        rows.append(row)
    swap = _swap_metrics(
        probabilities["N0-R1"],
        probabilities["N2-LocalNull-R1Dist"],
        gt,
        task["threshold"],
    )
    return {
        "dataset": dataset,
        "stem": stem,
        "rows": rows,
        "diagnostics": {field: diagnostics[field] for field in INTERNAL_FIELDS},
        "swap": swap,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


class MetricAggregate:
    def __init__(self):
        self.count = 0
        self.sums = defaultdict(float)
        self.valid = defaultdict(int)
        self.f_curve_sum = np.zeros(256, dtype=np.float64)
        self.e_curve_sum = np.zeros(256, dtype=np.float64)

    def add(self, row: dict):
        self.count += 1
        for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, *RANKING_FIELDS):
            value = float(row[field])
            if math.isfinite(value):
                self.sums[field] += value
                self.valid[field] += 1
        self.f_curve_sum += row["_official_f_curve"]
        self.e_curve_sum += row["_official_e_curve"]

    def result(self):
        result = {
            field: self.sums[field] / self.valid[field] if self.valid[field] else float("nan")
            for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, *RANKING_FIELDS)
        }
        result["official_soft_F_beta_max"] = float(np.max(self.f_curve_sum / self.count))
        result["official_soft_E_max"] = float(np.max(self.e_curve_sum / self.count))
        result["ranking_F_beta_max"] = result["official_soft_F_beta_max"]
        result["ranking_E_max"] = result["official_soft_E_max"]
        result["num_samples"] = self.count
        result["ap_valid_count"] = self.valid["pixel_AP"]
        return result


class NumericAggregate:
    def __init__(self, fields):
        self.fields = tuple(fields)
        self.count = 0
        self.values = {field: [] for field in self.fields}

    def add(self, row):
        self.count += 1
        for field in self.fields:
            value = float(row[field])
            if math.isfinite(value):
                self.values[field].append(value)

    def result(self, sum_fields=()):
        output = {}
        for field, values in self.values.items():
            output[field] = (
                float(np.sum(values)) if field in sum_fields else float(np.mean(values))
            ) if values else float("nan")
            output[f"{field}_valid_count"] = len(values)
        output["num_samples"] = self.count
        return output


def _manifest_map(path: Path, expected_count: int | None = None):
    rows = read_jsonl(path)
    mapping = {}
    for index, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{index}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"Duplicate key in {path}: {key}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        mapping[key] = row
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(f"{path} must have {expected_count} rows, got {len(rows)}")
    return rows, mapping


def _write_csv(path: Path, rows: list[dict], fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _section_rows(aggregates, datasets, section):
    section_fields = {
        "hard": HARD_FIELDS,
        "official_soft": OFFICIAL_FIELDS,
        "raw_continuous": RAW_FIELDS,
        "ranking": RANKING_FIELDS,
    }
    fields = section_fields[section]
    by_dataset = []
    for dataset in datasets:
        for method in METHODS:
            values = aggregates[(dataset, method)].result()
            by_dataset.append(
                {
                    "scope": "by_dataset",
                    "dataset": dataset,
                    "method": method,
                    **{field: values[field] for field in fields},
                    "num_samples": values["num_samples"],
                    "ap_valid_count": values["ap_valid_count"],
                }
            )
    overall = []
    for method in METHODS:
        values = aggregates[("ALL", method)].result()
        overall.append(
            {
                "scope": "sample_overall",
                "dataset": "ALL",
                "method": method,
                **{field: values[field] for field in fields},
                "num_samples": values["num_samples"],
                "ap_valid_count": values["ap_valid_count"],
            }
        )
    macro = []
    for method in METHODS:
        rows = [row for row in by_dataset if row["method"] == method]
        result = {"scope": "dataset_macro", "dataset": "ALL", "method": method}
        for field in fields:
            values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
            result[field] = float(np.mean(values)) if values else float("nan")
        result["num_samples"] = sum(int(row["num_samples"]) for row in rows)
        result["ap_valid_count"] = sum(int(row["ap_valid_count"]) for row in rows)
        macro.append(result)
    return by_dataset, overall, macro


def _average_rank(values: np.ndarray) -> np.ndarray:
    return average_percentile_rank(torch.from_numpy(values.astype(np.float64))).numpy()


def _correlation(left: list[float], right: list[float], rank: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    if rank:
        x, y = _average_rank(x), _average_rank(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _diagnostic_correlations(records: dict[str, list[dict]], datasets: list[str]):
    specs = (
        ("delta_n1_iou_vs_n0", "local_excluded_weight_mass_mean"),
        ("delta_n1_iou_vs_n0", "anchor_top1_self_ratio"),
        ("delta_n2_iou_vs_n0", "anchor_cross_raw_mean"),
        ("delta_n2_iou_vs_n0", "r1_localnull_spearman"),
        ("delta_n2_iou_vs_n0", "local_null_std"),
    )
    rows = []
    for dataset in datasets:
        for outcome, diagnostic in specs:
            outcome_values = [float(row[outcome]) for row in records[dataset]]
            diagnostic_values = [float(row[diagnostic]) for row in records[dataset]]
            valid_count = int(
                np.sum(np.isfinite(outcome_values) & np.isfinite(diagnostic_values))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "outcome": outcome,
                    "diagnostic": diagnostic,
                    "pearson": _correlation(outcome_values, diagnostic_values),
                    "spearman": _correlation(outcome_values, diagnostic_values, rank=True),
                    "valid_count": valid_count,
                }
            )
    return rows


def _gray_panel(value: torch.Tensor, size=224, nearest=False) -> Image.Image:
    value = value.detach().cpu().float().squeeze().clamp(0.0, 1.0)
    mode = "nearest" if nearest else "bilinear"
    resized = torch_f.interpolate(
        value[None, None], size=(size, size), mode=mode,
        **({"align_corners": False} if mode == "bilinear" else {}),
    )[0, 0]
    array = (resized.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def _difference_panel(value: torch.Tensor, size=224) -> Image.Image:
    value = value.detach().cpu().float().squeeze()
    resized = torch_f.interpolate(value[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    scale = max(float(resized.abs().max()), 1e-8)
    normalized = (resized / scale).numpy()
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
    array[..., 2] = (np.clip(-normalized, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _save_one_visual(task: dict, output_path: Path):
    source = torch_load(task["source_dabe_cache_path"], map_location="cpu")
    bgnull = torch_load(task["bgnull_cache_path"], map_location="cpu")
    gt = _load_gt(task["gt_path"])
    size = 224
    with Image.open(task["image_path"]) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    panels = [
        ("RGB", rgb),
        ("GT", _gray_panel(gt, size, nearest=True)),
        ("BG anchor", _gray_panel(bgnull["bg_anchor_37"], size, nearest=True)),
        ("R1", _gray_panel(bgnull["n0_r1_37"], size)),
        ("Cross-R1", _gray_panel(bgnull["n1_cross_r1_37"], size)),
        ("LocalNull Raw", _gray_panel(bgnull["n2_local_null_raw_37"], size)),
        ("LocalNull-R1Dist", _gray_panel(bgnull["n2_local_null_r1dist_37"], size)),
        ("NN Control", _gray_panel(bgnull["rc_nn_minmax_37"], size)),
        ("Current DABE-v2", _gray_panel(source["p_dabe_68"], size)),
        ("N2 - R1", _difference_panel(bgnull["n2_local_null_r1dist_37"] - bgnull["n0_r1_37"], size)),
    ]
    label_height = 24
    canvas = Image.new("RGB", (5 * size, 2 * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = (index % 5) * size
        y = (index // 5) * (size + label_height)
        canvas.paste(panel, (x, y + label_height))
        draw.text((x + 4, y + 5), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_visualizations(tasks, delta_map, out_dir: Path, datasets):
    by_dataset = defaultdict(list)
    for task in tasks:
        by_dataset[task["dataset"]].append(task)
    for dataset in datasets:
        available = by_dataset[dataset]
        for task in available[:8]:
            _save_one_visual(
                task,
                out_dir / "vis" / "fixed_first8" / dataset / f"{task['stem']}.png",
            )
        ranked = sorted(
            available,
            key=lambda task: delta_map[(task["dataset"], task["stem"])],
            reverse=True,
        )
        for folder, selected in (("n2_vs_r1_top8", ranked[:8]), ("n2_vs_r1_bottom8", ranked[-8:])):
            for task in selected:
                _save_one_visual(
                    task,
                    out_dir / "vis" / folder / dataset / f"{task['stem']}.png",
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata():
    try:
        commit = subprocess.check_output(["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(MAIN_ROOT), "status", "--short"], text=True, stderr=subprocess.DEVNULL).rstrip()
        return commit, status
    except (OSError, subprocess.CalledProcessError):
        return "", ""


def _markdown_table(rows, fields):
    headings = ("Dataset", "Method", *fields)
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["dataset"], row["method"]]
                + [f"{float(row[field]):.4f}" for field in fields]
            )
            + " |"
        )
    return "\n".join(lines)


def _success_and_route(hard_by, hard_macro, ranking_macro):
    by = {(row["dataset"], row["method"]): row for row in hard_by}
    macro = {row["method"]: row for row in hard_macro}
    ranking = {row["method"]: row for row in ranking_macro}
    r1 = macro["N0-R1"]
    drops = {
        method: [
            float(by[(dataset, method)]["hard_F_beta_w"])
            - float(by[(dataset, "N0-R1")]["hard_F_beta_w"])
            for dataset in DATASET_ORDER
        ]
        for method in ("N1-CrossR1", "N2-LocalNull-R1Dist", "RC-NN-MinMax")
    }
    n1_basic = (
        macro["N1-CrossR1"]["hard_F_beta_w"] > 0.6243
        and macro["N1-CrossR1"]["hard_MAE"] <= 0.0815
        and min(drops["N1-CrossR1"]) >= -0.010
    )
    n2_ranking = (
        ranking["N2-LocalNull-R1Dist"]["pixel_AP"] > ranking["N0-R1"]["pixel_AP"]
        and ranking["N2-LocalNull-R1Dist"]["best_IoU_256"] > ranking["N0-R1"]["best_IoU_256"]
    )
    n2_basic = (
        macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"] > 0.6243
        and ranking["N2-LocalNull-R1Dist"]["pixel_AP"] > ranking["N0-R1"]["pixel_AP"]
        and macro["N2-LocalNull-R1Dist"]["hard_MAE"] <= 0.0815
        and min(drops["N2-LocalNull-R1Dist"]) >= -0.010
    )
    n2_wins = sum(value >= 0 for value in drops["N2-LocalNull-R1Dist"])
    n2_clear = (
        macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"] >= 0.6280
        and n2_wins >= 3
        and ranking["N2-LocalNull-R1Dist"]["pixel_AP"] > ranking["N0-R1"]["pixel_AP"]
        and macro["N2-LocalNull-R1Dist"]["hard_MAE"] <= 0.0805
    )
    n2_strong = (
        macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"] >= 0.6320
        and min(drops["N2-LocalNull-R1Dist"]) >= -0.010
        and ranking["N2-LocalNull-R1Dist"]["pixel_AP"] > ranking["N0-R1"]["pixel_AP"]
        and ranking["N2-LocalNull-R1Dist"]["best_IoU_256"] > ranking["N0-R1"]["best_IoU_256"]
    )
    n1_improved = macro["N1-CrossR1"]["hard_F_beta_w"] > r1["hard_F_beta_w"]
    n2_improved = macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"] > r1["hard_F_beta_w"]
    rc_improved = macro["RC-NN-MinMax"]["hard_F_beta_w"] > r1["hard_F_beta_w"]
    if rc_improved and macro["RC-NN-MinMax"]["hard_F_beta_w"] > max(macro["N1-CrossR1"]["hard_F_beta_w"], macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"]):
        route = "D"
    elif n1_improved and n2_improved and macro["N2-LocalNull-R1Dist"]["hard_F_beta_w"] > macro["N1-CrossR1"]["hard_F_beta_w"]:
        route = "C"
    elif n1_improved and not n2_improved:
        route = "A"
    elif n2_improved and not n1_improved:
        route = "B"
    elif not n1_improved and not n2_improved and not rc_improved:
        route = "E"
    else:
        route = "mixed"
    return {
        "n1_basic_success": bool(n1_basic),
        "n2_ranking_success": bool(n2_ranking),
        "n2_basic_success": bool(n2_basic),
        "n2_clear_success": bool(n2_clear),
        "n2_strong_success": bool(n2_strong),
        "dataset_f_beta_w_deltas_vs_n0": dict(zip(("N1", "N2", "RC"), [drops["N1-CrossR1"], drops["N2-LocalNull-R1Dist"], drops["RC-NN-MinMax"]])),
        "route_case": route,
    }


def _write_readme(out_dir: Path):
    text = """# BGNull-v1 formal evaluation

- DINO is not rerun and GT is not used in candidate generation.
- Native maps use 37→68→original-GT bilinear resize with `align_corners=False`.
- Current-DABE-v2 uses cached `p_dabe_68` followed by bilinear resize.
- GT and Hard prediction both use strict `>0.5` where applicable.
- Repository Official Soft COD metrics perform per-image Min-Max for non-constant predictions.
- Raw Continuous applies no additional normalization.
- Best-IoU uses 256 fixed thresholds for diagnosis only and never changes the Hard result.
- Dataset Macro is the unweighted mean of the four datasets.
- Pairwise deltas are method A minus method B; positive MAE delta means larger error.
- Visualizations include manifest-first samples and symmetric N2-vs-R1 top/bottom samples.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_results(out_dir, sections, internal_by, swap_by, correlations, baseline, success, full_run):
    hard_by = sections["hard"]["by_dataset"]
    hard_macro = sections["hard"]["dataset_macro"]
    ranking_macro = sections["ranking"]["dataset_macro"]
    macro = {row["method"]: row for row in hard_macro}
    ranking = {row["method"]: row for row in ranking_macro}
    diagnostics = {row["dataset"]: row for row in internal_by}
    swap = {row["dataset"]: row for row in swap_by}
    hard_table = hard_by + [{**row, "dataset": "Dataset-Macro"} for row in hard_macro]
    lines = [
        "# DABE-TF BGNull-v1 结果",
        "",
        "## 协议确认",
        "",
        "- DINO rerun: false",
        "- GT used for generation: false",
        "- Source DABE: v2 identity",
        "- Cross radius: 1",
        "- K_RECON: 32",
        "- Hard threshold: strict > 0.5",
        "- Dataset-specific parameters: none",
        "- Candidate formulas frozen before full evaluation: true",
        "- Official Soft 对非恒定预测逐图 Min-Max。",
        "",
        "## 基线复现",
        "",
        ("通过：所有 N0-R1 与 Current-DABE-v2 核查误差均 ≤ 1e-5。" if baseline.get("passed") else ("失败：评测协议或输入未对齐，停止给出性能结论。" if full_run else "Sanity run 不适用完整基线复现。")),
        "",
        "## Hard 主表",
        "",
        _markdown_table(hard_table, ("hard_S_m", "hard_F_beta_w", "hard_E_mean", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area")),
        "",
    ]
    if not full_run or not baseline.get("passed"):
        lines += ["## 方法判断", "", "非完整评测或基线未通过，不给出方法性能结论。", ""]
    else:
        n0 = macro["N0-R1"]
        n1 = macro["N1-CrossR1"]
        n2 = macro["N2-LocalNull-R1Dist"]
        rc = macro["RC-NN-MinMax"]
        n2raw_rank = ranking["N2-LocalNull-Raw"]
        corr_lookup = {(row["dataset"], row["outcome"], row["diagnostic"]): row for row in correlations}
        mean_self = float(np.mean([diagnostics[d]["anchor_top1_self_ratio"] for d in DATASET_ORDER]))
        mean_anchor_cross = float(np.mean([diagnostics[d]["cross_minus_regular_anchor_mean"] for d in DATASET_ORDER]))
        mean_excluded = float(np.mean([diagnostics[d]["local_excluded_weight_mass_mean"] for d in DATASET_ORDER]))
        mean_swap_accuracy = float(np.nanmean([swap[d]["swap_accuracy"] for d in DATASET_ORDER]))
        mean_added_precision = float(np.nanmean([swap[d]["added_fg_precision"] for d in DATASET_ORDER]))
        mean_removed_precision = float(np.nanmean([swap[d]["removed_bg_precision"] for d in DATASET_ORDER]))
        n1_wins = sum(
            row["hard_F_beta_w"] >= next(item["hard_F_beta_w"] for item in hard_by if item["dataset"] == row["dataset"] and item["method"] == "N0-R1")
            for row in hard_by if row["method"] == "N1-CrossR1"
        )
        n2_wins = sum(
            row["hard_F_beta_w"] >= next(item["hard_F_beta_w"] for item in hard_by if item["dataset"] == row["dataset"] and item["method"] == "N0-R1")
            for row in hard_by if row["method"] == "N2-LocalNull-R1Dist"
        )
        lines += [
            "## N1 判断",
            "",
            f"1. N1 相对 R1 的 Dataset-Macro F_beta^w 为 {n1['hard_F_beta_w']-n0['hard_F_beta_w']:+.6f}，四数据集非劣数量 {n1_wins}/4。",
            f"2. Precision/Recall 改变量分别为 {n1['hard_Precision']-n0['hard_Precision']:+.6f} / {n1['hard_Recall']-n0['hard_Recall']:+.6f}。",
            f"3. Hard Area 改变量为 {n1['hard_Area']-n0['hard_Area']:+.6f}。",
            f"4. 排除局部原子的原 R1 平均权重质量为 {mean_excluded:.6f}；与 N1 ΔIoU 的逐数据集相关性见 diagnostic_correlations.csv。",
            f"5. 背景 anchor 自身成为 top-1 的宏平均比例为 {mean_self:.6f}。",
            f"6. 排除自身及 8 邻域后，anchor raw residual 平均提高 {mean_anchor_cross:+.6f}。",
            "7. 是否存在明显局部复制，以自身 top-1、局部权重质量及 N1 是否提升联合判断，不能仅由误差上升断言。",
            "",
            "## N2 判断",
            "",
            f"1. N2 相对 R1 的 Dataset-Macro F_beta^w 为 {n2['hard_F_beta_w']-n0['hard_F_beta_w']:+.6f}，四数据集非劣数量 {n2_wins}/4。",
            f"2. N2 相对 R1 的 Hard Area 改变量为 {n2['hard_Area']-n0['hard_Area']:+.8f}。",
            f"3. Pixel AP 改变量为 {ranking['N2-LocalNull-R1Dist']['pixel_AP']-ranking['N0-R1']['pixel_AP']:+.6f}。",
            f"4. Best IoU-256 改变量为 {ranking['N2-LocalNull-R1Dist']['best_IoU_256']-ranking['N0-R1']['best_IoU_256']:+.6f}。",
            f"5. 新增像素 GT 前景比例宏平均为 {mean_added_precision:.6f}。",
            f"6. 删除像素 GT 背景比例宏平均为 {mean_removed_precision:.6f}。",
            f"7. 像素交换准确率宏平均为 {mean_swap_accuracy:.6f}；结合 AP/F_beta^w 判断 LocalNull 改善或破坏排序。",
            f"8. N2-Raw Pixel AP={n2raw_rank['pixel_AP']:.6f}，R1 Pixel AP={ranking['N0-R1']['pixel_AP']:.6f}，可判断 Raw 排序是否有效但 Hard 未体现。",
            "9. 是否统一有效由四数据集 Hard/Ranking 表逐项判断，不进行数据集特定选择。",
            "",
            "## Retrieval-Control 判断",
            "",
            f"1. RC 相对 R1 的 Dataset-Macro F_beta^w 为 {rc['hard_F_beta_w']-n0['hard_F_beta_w']:+.6f}。",
            f"2. RC 相对 R1 的 Pixel AP 为 {ranking['RC-NN-MinMax']['pixel_AP']-ranking['N0-R1']['pixel_AP']:+.6f}。",
            f"3. RC 与 R1/N1/N2 的相对结果用于判断单原子检索是否优于加权背景重构。",
            "4. RC 若最强，应修正组合重构优势主张；RC 若更弱，则支持加权重构的必要性。",
            "",
            "## 成功标准与路线",
            "",
            "```json",
            json.dumps(success, ensure_ascii=False, indent=2),
            "```",
            "",
            f"最终预定义路线分支：{success['route_case']}。",
            "",
        ]
        route_text = {
            "A": "主要问题是自身/局部复制；保留 Cross-R1，暂不采用 LocalNull。",
            "B": "主要问题是不同背景模式难度不可全局比较；Local Background-Null Calibration 有效。",
            "C": "局部复制和背景模式校准均存在问题；冻结 N2 为下一版主候选。",
            "D": "单原子检索更可靠；下一阶段重新审视 reconstructability 设计。",
            "E": "N1/N2/RC 均未超过 R1；停止背景零分布路线，转向高分辨率边界投影或多层 DINO 表征。",
            "mixed": "结果不属于预定义 A–E 的单一强分支，保持 N0 并审慎复核各固定候选。",
        }
        lines.append(route_text[success["route_case"]])
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_background_null(
    config_path,
    dabe_root,
    bgnull_root,
    out_dir,
    split="test",
    max_samples=-1,
    threshold=0.5,
    save_vis=False,
    workers=None,
    torch_threads=1,
    overwrite=False,
    overwrite_reason="",
):
    started = time.time()
    config_path = Path(config_path).resolve()
    dabe_root = Path(dabe_root).resolve()
    bgnull_root = Path(bgnull_root).resolve()
    out_dir = Path(out_dir).resolve()
    if split != "test" or threshold != 0.5:
        raise ValueError("Frozen protocol requires split=test and threshold=0.5")
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    full_run = max_samples == -1
    source_rows_all, source_map = _manifest_map(dabe_root / f"manifest_{split}.jsonl", 6473 if full_run else None)
    bgnull_rows_all, bgnull_map = _manifest_map(bgnull_root / f"manifest_{split}.jsonl", 6473 if full_run else None)
    selected = source_rows_all if full_run else source_rows_all[:max_samples]
    selected_keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
    if full_run and set(source_map) != set(bgnull_map):
        raise RuntimeError("Full DABE/BGNull manifest keys differ")
    if not full_run and set(selected_keys) != set(bgnull_map):
        raise RuntimeError("Sanity BGNull manifest must exactly match selected DABE keys")
    missing = [key for key in selected_keys if key not in bgnull_map]
    if missing:
        raise RuntimeError(f"Missing BGNull keys: {missing[:10]}")
    datasets = [name for name in DATASET_ORDER if any(key[0] == name for key in selected_keys)]
    for row in selected:
        if "gt_path" not in row or not Path(row["gt_path"]).is_file():
            raise FileNotFoundError(f"GT missing for {row['dataset']}/{row['stem']}")
        if "image_path" not in row or not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Image missing for {row['dataset']}/{row['stem']}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_dir: {out_dir}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (out_dir / "eval.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    workers = workers or min(12, os.cpu_count() or 1)
    tasks = []
    for row in selected:
        key = (str(row["dataset"]), str(row["stem"]))
        tasks.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "source_dabe_cache_path": row["cache_path"],
                "bgnull_cache_path": bgnull_map[key]["cache_path"],
                "gt_path": row["gt_path"],
                "image_path": row["image_path"],
                "threshold": threshold,
            }
        )
    metrics = {(scope, method): MetricAggregate() for scope in [*datasets, "ALL"] for method in METHODS}
    internal = {scope: NumericAggregate(INTERNAL_FIELDS) for scope in [*datasets, "ALL"]}
    swaps = {scope: NumericAggregate(SWAP_FIELDS) for scope in [*datasets, "ALL"]}
    pair_values = defaultdict(list)
    correlation_records = defaultdict(list)
    vis_delta_map = {}
    peak_worker_rss = 0.0
    pair_sample_fields = ("dataset", "stem", "method_a", "method_b", *PAIRWISE_FIELDS)
    swap_sample_fields = ("dataset", "stem", *SWAP_FIELDS)
    internal_sample_fields = ("dataset", "stem", *INTERNAL_FIELDS)
    internal_per_sample = []
    try:
        log(f"num_samples = {len(tasks)}")
        log(f"workers = {workers}")
        log("hard_threshold = strict > 0.5")
        log("candidate_generation_in_eval = false")
        with (
            (out_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as sample_handle,
            (out_dir / "pairwise_per_sample.csv").open("w", encoding="utf-8", newline="") as pair_handle,
            (out_dir / "swap_analysis_per_sample.csv").open("w", encoding="utf-8", newline="") as swap_handle,
        ):
            sample_writer = csv.DictWriter(sample_handle, fieldnames=PER_SAMPLE_FIELDS, extrasaction="ignore")
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_sample_fields, extrasaction="ignore")
            swap_writer = csv.DictWriter(swap_handle, fieldnames=swap_sample_fields, extrasaction="ignore")
            sample_writer.writeheader()
            pair_writer.writeheader()
            swap_writer.writeheader()
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(torch_threads,)) as executor:
                for index, result in enumerate(executor.map(_evaluate_one, tasks, chunksize=1), 1):
                    dataset, stem = result["dataset"], result["stem"]
                    peak_worker_rss = max(peak_worker_rss, float(result["worker_peak_rss_mb"]))
                    rows_by_method = {row["method"]: row for row in result["rows"]}
                    for row in result["rows"]:
                        metrics[(dataset, row["method"])].add(row)
                        metrics[("ALL", row["method"])].add(row)
                        sample_writer.writerow(row)
                    diag_row = {"dataset": dataset, "stem": stem, **result["diagnostics"]}
                    internal_per_sample.append(diag_row)
                    internal[dataset].add(diag_row)
                    internal["ALL"].add(diag_row)
                    swap_row = {"dataset": dataset, "stem": stem, **result["swap"]}
                    swaps[dataset].add(swap_row)
                    swaps["ALL"].add(swap_row)
                    swap_writer.writerow(swap_row)

                    n1_delta_iou = rows_by_method["N1-CrossR1"]["hard_IoU"] - rows_by_method["N0-R1"]["hard_IoU"]
                    n2_delta_iou = rows_by_method["N2-LocalNull-R1Dist"]["hard_IoU"] - rows_by_method["N0-R1"]["hard_IoU"]
                    correlation_records[dataset].append(
                        {
                            **result["diagnostics"],
                            "delta_n1_iou_vs_n0": n1_delta_iou,
                            "delta_n2_iou_vs_n0": n2_delta_iou,
                        }
                    )
                    vis_delta_map[(dataset, stem)] = n2_delta_iou
                    mapping = {
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
                    for method_a, method_b in PAIRWISE:
                        row_a, row_b = rows_by_method[method_a], rows_by_method[method_b]
                        pair_row = {"dataset": dataset, "stem": stem, "method_a": method_a, "method_b": method_b}
                        for delta_field, source_field in mapping.items():
                            left, right = float(row_a[source_field]), float(row_b[source_field])
                            delta = left - right if math.isfinite(left) and math.isfinite(right) else float("nan")
                            pair_row[delta_field] = delta
                            if math.isfinite(delta):
                                pair_values[(dataset, method_a, method_b, delta_field)].append(delta)
                        pair_writer.writerow(pair_row)
                    if index == len(tasks) or index % 100 == 0:
                        log(f"processed = {index}/{len(tasks)}")

        _write_csv(out_dir / "internal_diagnostics_per_sample.csv", internal_per_sample, internal_sample_fields)
        sections = {}
        for section in ("hard", "official_soft", "raw_continuous", "ranking"):
            by, overall, macro = _section_rows(metrics, datasets, section)
            sections[section] = {"by_dataset": by, "sample_overall": overall, "dataset_macro": macro}
            fields = {"hard": HARD_FIELDS, "official_soft": OFFICIAL_FIELDS, "raw_continuous": RAW_FIELDS, "ranking": RANKING_FIELDS}[section]
            csv_fields = ("scope", "dataset", "method", *fields, "num_samples", "ap_valid_count")
            _write_csv(out_dir / f"{section}_by_dataset.csv", by, csv_fields)
            _write_csv(out_dir / f"{section}_sample_overall.csv", overall, csv_fields)
            _write_csv(out_dir / f"{section}_dataset_macro.csv", macro, csv_fields)

        internal_by = []
        for dataset in datasets:
            values = internal[dataset].result(sum_fields=("cross_fallback_count", "regular_top1_self_count", "regular_topk_contains_self_count"))
            internal_by.append({"scope": "by_dataset", "dataset": dataset, **values})
        internal_overall = [{"scope": "sample_overall", "dataset": "ALL", **internal["ALL"].result(sum_fields=("cross_fallback_count", "regular_top1_self_count", "regular_topk_contains_self_count"))}]
        internal_fields_out = ("scope", "dataset", *INTERNAL_FIELDS, *[f"{field}_valid_count" for field in INTERNAL_FIELDS], "num_samples")
        _write_csv(out_dir / "internal_diagnostics_by_dataset.csv", internal_by, internal_fields_out)
        _write_csv(out_dir / "internal_diagnostics_sample_overall.csv", internal_overall, internal_fields_out)

        swap_by = []
        for dataset in datasets:
            values = swaps[dataset].result(sum_fields=("correct_swap_count", "incorrect_swap_count"))
            swap_by.append({"scope": "by_dataset", "dataset": dataset, **values})
        swap_fields_out = ("scope", "dataset", *SWAP_FIELDS, *[f"{field}_valid_count" for field in SWAP_FIELDS], "num_samples")
        _write_csv(out_dir / "swap_analysis_by_dataset.csv", swap_by, swap_fields_out)

        pair_summary = []
        for dataset in datasets:
            for method_a, method_b in PAIRWISE:
                for metric in PAIRWISE_FIELDS:
                    values = np.asarray(pair_values[(dataset, method_a, method_b, metric)], dtype=np.float64)
                    if values.size == 0:
                        continue
                    eps = 1e-6
                    pair_summary.append(
                        {
                            "dataset": dataset,
                            "method_a": method_a,
                            "method_b": method_b,
                            "metric": metric,
                            "mean_delta": float(np.mean(values)),
                            "median_delta": float(np.median(values)),
                            "p25_delta": float(np.percentile(values, 25)),
                            "p75_delta": float(np.percentile(values, 75)),
                            "win_ratio": float(np.mean(values > eps)),
                            "tie_ratio": float(np.mean(np.abs(values) <= eps)),
                            "loss_ratio": float(np.mean(values < -eps)),
                            "valid_count": int(values.size),
                        }
                    )
        pair_summary_fields = ("dataset", "method_a", "method_b", "metric", "mean_delta", "median_delta", "p25_delta", "p75_delta", "win_ratio", "tie_ratio", "loss_ratio", "valid_count")
        _write_csv(out_dir / "pairwise_by_dataset.csv", pair_summary, pair_summary_fields)
        correlations = _diagnostic_correlations(correlation_records, datasets)
        _write_csv(out_dir / "diagnostic_correlations.csv", correlations, ("dataset", "outcome", "diagnostic", "pearson", "spearman", "valid_count"))

        if save_vis:
            log("saving_visualizations = true")
            _save_visualizations(tasks, vis_delta_map, out_dir, datasets)
            log("visualizations_complete = true")

        hard_macro_map = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
        hard_by_map = {(row["dataset"], row["method"]): row for row in sections["hard"]["by_dataset"]}
        baseline = {"applicable": full_run, "tolerance": 1e-5, "checks": {}}
        if full_run:
            passed = True
            for method, expected_fields in BASELINE_EXACT.items():
                baseline["checks"][method] = {}
                for field, expected in expected_fields.items():
                    actual = float(hard_macro_map[method][field])
                    error = abs(actual - expected)
                    ok = error <= 1e-5
                    passed &= ok
                    baseline["checks"][method][field] = {"actual": actual, "expected": expected, "absolute_error": error, "passed": bool(ok)}
            baseline["checks"]["N0-R1_by_dataset_F_beta_w"] = {}
            for dataset, expected in R1_DATASET_F_W.items():
                actual = float(hard_by_map[(dataset, "N0-R1")]["hard_F_beta_w"])
                error = abs(actual - expected)
                ok = error <= 1e-5
                passed &= ok
                baseline["checks"]["N0-R1_by_dataset_F_beta_w"][dataset] = {"actual": actual, "expected": expected, "absolute_error": error, "passed": bool(ok)}
            baseline["passed"] = bool(passed)
        else:
            baseline["passed"] = None
        success = _success_and_route(sections["hard"]["by_dataset"], sections["hard"]["dataset_macro"], sections["ranking"]["dataset_macro"]) if full_run and baseline["passed"] else {"not_evaluated": True}

        elapsed = time.time() - started
        main_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        git_commit, git_status = _git_metadata()
        protocol = {
            "bgnull_version": "dabe_bgnull_v1",
            "split": split,
            "num_samples": len(tasks),
            "full_run": full_run,
            "methods": list(METHODS),
            "main_methods": list(MAIN_METHODS),
            "diagnostic_method": "N2-LocalNull-Raw",
            "pairwise_comparisons": [list(value) for value in PAIRWISE],
            "dino_rerun": False,
            "gt_used_for_generation": False,
            "source_dabe": "v2 identity",
            "cross_radius": 1,
            "k_recon": 32,
            "hard_threshold": threshold,
            "hard_operator": ">",
            "dataset_specific_parameters": False,
            "candidate_formulas_frozen_before_full_evaluation": True,
            "native_resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
            "current_resize": "p_dabe_68 -> original GT bilinear align_corners=False",
            "official_soft_per_image_minmax": True,
            "raw_continuous_extra_minmax": False,
            "best_iou_thresholds": "torch.linspace(0,1,256), diagnostic only",
            "save_vis": bool(save_vis),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_dabe_manifest": str((dabe_root / f"manifest_{split}.jsonl").resolve()),
            "source_dabe_manifest_sha256": _sha256(dabe_root / f"manifest_{split}.jsonl"),
            "bgnull_manifest": str((bgnull_root / f"manifest_{split}.jsonl").resolve()),
            "bgnull_manifest_sha256": _sha256(bgnull_root / f"manifest_{split}.jsonl"),
            "evaluator_sha256": _sha256(SCRIPT_PATH),
            "git_commit": git_commit,
            "git_status_short": git_status,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(tasks),
            "main_peak_rss_mb": main_peak_rss,
            "max_worker_peak_rss_mb": peak_worker_rss,
            "overwrite": bool(overwrite),
            "overwrite_reason": overwrite_reason.strip(),
        }
        summary = {
            "protocol": protocol,
            "baseline_reproduction": baseline,
            "success_assessment": success,
            **sections,
            "internal_diagnostics_by_dataset": internal_by,
            "internal_diagnostics_sample_overall": internal_overall,
            "swap_analysis_by_dataset": swap_by,
            "diagnostic_correlations": correlations,
            "pairwise_by_dataset": pair_summary,
        }
        write_json(out_dir / "protocol.json", protocol)
        write_json(out_dir / "summary.json", summary)
        _write_readme(out_dir)
        _write_results(out_dir, sections, internal_by, swap_by, correlations, baseline, success, full_run)
        log(f"baseline_reproduction = {baseline.get('passed')}")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(tasks):.6f}")
        log(f"main_peak_rss_mb = {main_peak_rss:.3f}")
        log(f"max_worker_peak_rss_mb = {peak_worker_rss:.3f}")
        return summary
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--bgnull_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_background_null(
        config_path=args.config,
        dabe_root=args.dabe_root,
        bgnull_root=args.bgnull_root,
        out_dir=args.out_dir,
        split=args.split,
        max_samples=args.max_samples,
        threshold=args.threshold,
        save_vis=args.save_vis,
        workers=args.workers,
        torch_threads=args.torch_threads,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
    )
