#!/usr/bin/env python3
"""Formal original-GT evaluation for frozen DABE rank-calibration caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt, label

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import Emeasure, Fmeasure, Smeasure, _EPS, _prepare_data  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
METHODS = (
    "R1",
    "R",
    "F",
    "Current-DABE-v2",
    "C0-F-MinMax",
    "C1-FRank-R1Dist",
    "C2-MedianRank-R1Dist",
)
PAIRWISE = (
    ("C0-F-MinMax", "F"),
    ("C1-FRank-R1Dist", "R1"),
    ("C2-MedianRank-R1Dist", "R1"),
    ("C1-FRank-R1Dist", "Current-DABE-v2"),
    ("C2-MedianRank-R1Dist", "Current-DABE-v2"),
    ("C2-MedianRank-R1Dist", "C1-FRank-R1Dist"),
)
SOURCE_FIELDS = {
    "R1": "residual_pass1_37",
    "R": "residual_37",
    "F": "fg_score_37",
    "Current-DABE-v2": "p_dabe_68",
}
RANKCAL_FIELDS = {
    "C0-F-MinMax": "c0_f_minmax_37",
    "C1-FRank-R1Dist": "c1_f_rank_r1_37",
    "C2-MedianRank-R1Dist": "c2_median_rank_r1_37",
}
DATASET_ORDER = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
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
    "raw_fg_mass_ratio",
)
RANKING_FIELDS = (
    "pixel_AP",
    "best_IoU_256",
    "best_IoU_threshold_256",
)
PER_SAMPLE_FIELDS = (
    "dataset",
    "stem",
    "method",
    *HARD_FIELDS,
    *OFFICIAL_FIELDS,
    *RAW_FIELDS,
    *RANKING_FIELDS,
    "source_cache_path",
    "rankcal_cache_path",
)
PAIRWISE_FIELDS = (
    "delta_hard_F_beta_w",
    "delta_hard_IoU",
    "delta_hard_Precision",
    "delta_hard_Recall",
    "delta_hard_MAE",
    "delta_hard_Area",
    "delta_official_soft_F_beta_w",
    "delta_raw_MAE",
    "delta_pixel_AP",
    "delta_best_IoU_256",
)
BASELINE_EXACT = {
    "R1": {
        "hard_S_m": 0.7321127747158344,
        "hard_F_beta_w": 0.6242804301164221,
        "hard_E_mean": 0.8279479346485146,
        "hard_MAE": 0.08046898620831737,
    },
    "Current-DABE-v2": {
        "hard_S_m": 0.7031564688307831,
        "hard_F_beta_w": 0.5722451313959636,
        "hard_E_mean": 0.7743040501812801,
        "hard_MAE": 0.0939140360866542,
    },
}


def _gaussian_kernel(shape=(7, 7), sigma=5.0):
    m, n = [(size - 1) / 2 for size in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    total = kernel.sum()
    if total:
        kernel /= total
    return kernel


WFM_KERNEL = _gaussian_kernel()


class FastCODContext:
    """Repository COD formulas with the GT-only weighted-F geometry cached."""

    def __init__(self, target: torch.Tensor):
        target_np = target.detach().cpu().numpy().astype(float).squeeze()
        _, self.gt = _prepare_data(gt=target_np, pred=target_np)
        self.gt_float = self.gt.astype(float)
        self.gt_fg_numel = int(np.count_nonzero(self.gt))
        self.gt_size = int(self.gt.size)
        if self.gt_fg_numel:
            self.distance, self.nearest = distance_transform_edt(self.gt == 0, return_indices=True)
        else:
            self.distance = self.nearest = None
        self.smeasure = Smeasure()
        self.emeasure = Emeasure()
        self.fmeasure = Fmeasure()

    def weighted_f_many(self, predictions: np.ndarray) -> np.ndarray:
        if self.gt_fg_numel == 0:
            return np.zeros((predictions.shape[0],), dtype=np.float64)
        error = np.abs(predictions - self.gt[None, ...])
        propagated = np.copy(error)
        background = self.gt == 0
        propagated[:, background] = error[:, self.nearest[0][background], self.nearest[1][background]]
        averaged = convolve(propagated, weights=WFM_KERNEL[None, ...], mode="constant", cval=0)
        minimum_error = np.where(self.gt[None, ...] & (averaged < error), averaged, error)
        weight = np.where(
            background,
            2 - np.exp(np.log(0.5) / 5 * self.distance),
            np.ones_like(self.gt),
        )
        weighted_error = minimum_error * weight[None, ...]
        tp_weighted = np.sum(self.gt) - np.sum(weighted_error[:, self.gt == 1], axis=1)
        fp_weighted = np.sum(weighted_error[:, background], axis=1)
        recall = 1 - np.mean(weighted_error[:, self.gt == 1], axis=1)
        precision = tp_weighted / (tp_weighted + fp_weighted + _EPS)
        return 2 * recall * precision / (recall + precision + _EPS)

    def evaluate_many(self, predictions: list[tuple[str, str, torch.Tensor]], threshold: float):
        prepared = []
        evaluated = []
        for method, mode, probability in predictions:
            value = (probability > threshold).float() if mode == "hard" else probability.float()
            pred_np, _ = _prepare_data(gt=self.gt_float, pred=value.numpy().astype(float).squeeze())
            prepared.append(pred_np)
            evaluated.append((method, mode, value))
        weighted = self.weighted_f_many(np.stack(prepared, axis=0))

        outputs = {}
        gt_tensor = torch.from_numpy(self.gt.astype(np.bool_)).unsqueeze(0)
        for (method, mode, value), pred_np, wfm in zip(evaluated, prepared, weighted):
            self.emeasure.gt_fg_numel = self.gt_fg_numel
            self.emeasure.gt_size = self.gt_size
            e_curve = self.emeasure.cal_changeable_em(pred_np, self.gt)
            _, _, f_curve = self.fmeasure.cal_pr(pred_np, self.gt)
            base = {
                "S_m": float(self.smeasure.cal_sm(pred_np, self.gt)),
                "F_beta_w": float(wfm),
                "F_beta_mean": float(np.mean(f_curve)),
                "F_beta_max": float(np.max(f_curve)),
                "E_mean": float(np.mean(e_curve)),
                "E_max": float(np.max(e_curve)),
                "MAE": float(np.mean(np.abs(pred_np - self.gt))),
                "f_curve": np.asarray(f_curve, dtype=np.float32),
                "e_curve": np.asarray(e_curve, dtype=np.float32),
            }
            if mode == "hard":
                predicted = value > 0.5
                intersection = float((predicted & gt_tensor).sum())
                union = float((predicted | gt_tensor).sum())
                pred_count = float(predicted.sum())
                gt_count = float(gt_tensor.sum())
                precision = intersection / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
                recall = intersection / gt_count if gt_count else 1.0
                base.update(
                    {
                        "IoU": intersection / union if union else 1.0,
                        "Precision": precision,
                        "Recall": recall,
                        "Area": float(predicted.float().mean()),
                        "Components": int(
                            label(
                                predicted.squeeze().numpy().astype(np.uint8),
                                structure=np.ones((3, 3), dtype=np.uint8),
                            )[1]
                        ),
                    }
                )
            outputs[(method, mode)] = base
        return outputs


def _load_response(payload: dict, field: str, expected_shape: tuple[int, int], path: Path):
    if field not in payload:
        raise KeyError(f"{field} missing from {path}")
    value = payload[field]
    if not torch.is_tensor(value) or tuple(value.shape) != (1, *expected_shape):
        raise ValueError(f"{field} must be Tensor[1,{expected_shape[0]},{expected_shape[1]}]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all() or float(value.min()) < -1e-6 or float(value.max()) > 1 + 1e-6:
        raise ValueError(f"Invalid {field}: {path}")
    return value.clamp(0.0, 1.0)


def _resize_native(value: torch.Tensor, gt_shape: tuple[int, int]) -> torch.Tensor:
    value = torch_f.interpolate(value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False)
    value = torch_f.interpolate(value, size=gt_shape, mode="bilinear", align_corners=False)
    return value.squeeze(0).clamp(0.0, 1.0)


def _resize_current(value: torch.Tensor, gt_shape: tuple[int, int]) -> torch.Tensor:
    value = torch_f.interpolate(value.unsqueeze(0), size=gt_shape, mode="bilinear", align_corners=False)
    return value.squeeze(0).clamp(0.0, 1.0)


def _raw_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict:
    probability = probability.float().clamp(0.0, 1.0)
    target = (target > 0.5).float()
    intersection = float((probability * target).sum())
    pred_mass = float(probability.sum())
    gt_mass = float(target.sum())
    union = pred_mass + gt_mass - intersection
    eps = 1e-12
    mean_probability = float(probability.mean())
    return {
        "raw_MAE": float(torch.abs(probability - target).mean()),
        "raw_Brier": float(torch.square(probability - target).mean()),
        "raw_SoftPrecision": intersection / (pred_mass + eps),
        "raw_SoftRecall": intersection / (gt_mass + eps),
        "raw_SoftIoU": intersection / (union + eps),
        "raw_prob_mean": mean_probability,
        "raw_fg_mass_ratio": mean_probability,
    }


def _ranking_metrics(probability: torch.Tensor, target: torch.Tensor) -> dict:
    score = probability.detach().cpu().numpy().astype(np.float64).reshape(-1)
    positive = (target.detach().cpu().numpy().reshape(-1) > 0.5)
    positive_count = int(positive.sum())
    if positive_count:
        order = np.argsort(-score, kind="stable")
        ranked_positive = positive[order]
        cumulative = np.cumsum(ranked_positive, dtype=np.float64)
        locations = np.flatnonzero(ranked_positive)
        pixel_ap = float(np.mean(cumulative[locations] / (locations + 1)))
    else:
        pixel_ap = float("nan")

    thresholds = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    positive_scores = np.sort(score[positive])
    negative_scores = np.sort(score[~positive])
    tp = positive_count - np.searchsorted(positive_scores, thresholds, side="right")
    fp = negative_scores.size - np.searchsorted(negative_scores, thresholds, side="right")
    union = positive_count + fp
    ious = np.divide(tp, union, out=np.ones_like(tp, dtype=np.float64), where=union != 0)
    best_index = int(np.argmax(ious))
    return {
        "pixel_AP": pixel_ap,
        "best_IoU_256": float(ious[best_index]),
        "best_IoU_threshold_256": float(thresholds[best_index]),
    }


def _init_worker(torch_threads: int):
    torch.set_num_threads(torch_threads)


def _evaluate_one(task: dict) -> dict:
    source_path = Path(task["source_cache_path"])
    rankcal_path = Path(task["rankcal_cache_path"])
    source = torch_load(source_path, map_location="cpu")
    rankcal = torch_load(rankcal_path, map_location="cpu")
    dataset, stem = task["dataset"], task["stem"]
    if source.get("dataset") != dataset or source.get("stem") != stem:
        raise RuntimeError(f"Source key mismatch: {source_path}")
    if rankcal.get("dataset") != dataset or rankcal.get("stem") != stem:
        raise RuntimeError(f"Rankcal key mismatch: {rankcal_path}")
    if rankcal.get("rankcal_version") != "dabe_rankcal_v1":
        raise RuntimeError(f"Wrong rankcal_version: {rankcal_path}")

    with Image.open(task["gt_path"]) as image:
        gt_array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    gt = torch.from_numpy((gt_array > 0.5).astype(np.float32)).unsqueeze(0)
    gt_shape = tuple(gt.shape[-2:])

    probabilities = {
        "R1": _resize_native(_load_response(source, "residual_pass1_37", (37, 37), source_path), gt_shape),
        "R": _resize_native(_load_response(source, "residual_37", (37, 37), source_path), gt_shape),
        "F": _resize_native(_load_response(source, "fg_score_37", (37, 37), source_path), gt_shape),
        "Current-DABE-v2": _resize_current(_load_response(source, "p_dabe_68", (68, 68), source_path), gt_shape),
        "C0-F-MinMax": _resize_native(_load_response(rankcal, "c0_f_minmax_37", (37, 37), rankcal_path), gt_shape),
        "C1-FRank-R1Dist": _resize_native(_load_response(rankcal, "c1_f_rank_r1_37", (37, 37), rankcal_path), gt_shape),
        "C2-MedianRank-R1Dist": _resize_native(_load_response(rankcal, "c2_median_rank_r1_37", (37, 37), rankcal_path), gt_shape),
    }
    context = FastCODContext(gt)
    metric_inputs = []
    for method in METHODS:
        metric_inputs.append((method, "hard", probabilities[method]))
        metric_inputs.append((method, "soft", probabilities[method]))
    cod = context.evaluate_many(metric_inputs, task["threshold"])

    rows = []
    for method in METHODS:
        hard = cod[(method, "hard")]
        soft = cod[(method, "soft")]
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
            **_ranking_metrics(probabilities[method], gt),
            "source_cache_path": str(source_path),
            "rankcal_cache_path": "" if method == "Current-DABE-v2" else str(rankcal_path),
            "_official_f_curve": soft["f_curve"],
            "_official_e_curve": soft["e_curve"],
        }
        if not all(math.isfinite(float(row[field])) for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, "best_IoU_256", "best_IoU_threshold_256")):
            raise RuntimeError(f"Non-finite metric for {dataset}/{stem}/{method}")
        rows.append(row)
    return {"dataset": dataset, "stem": stem, "rows": rows}


class Aggregate:
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

    def means(self) -> dict:
        result = {
            field: self.sums[field] / self.valid[field] if self.valid[field] else float("nan")
            for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, *RANKING_FIELDS)
        }
        result["official_soft_F_beta_max"] = float(np.max(self.f_curve_sum / self.count))
        result["official_soft_E_max"] = float(np.max(self.e_curve_sum / self.count))
        result["num_samples"] = self.count
        result["ap_valid_count"] = self.valid["pixel_AP"]
        return result


def _manifest_map(path: Path, required_count: int | None = None) -> tuple[list[dict], dict]:
    rows = read_jsonl(path)
    mapping = {}
    for index, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{index}")
        key = (row["dataset"], row["stem"])
        if key in mapping:
            raise RuntimeError(f"Duplicate key in {path}: {key}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        mapping[key] = row
    if required_count is not None and len(rows) != required_count:
        raise RuntimeError(f"{path} must have {required_count} rows, got {len(rows)}")
    return rows, mapping


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _section_rows(aggregates: dict, datasets: list[str], section: str):
    fields = {"hard": HARD_FIELDS, "official_soft": OFFICIAL_FIELDS, "raw_continuous": RAW_FIELDS, "ranking": RANKING_FIELDS}[section]
    by_dataset = []
    for dataset in datasets:
        for method in METHODS:
            values = aggregates[(dataset, method)].means()
            by_dataset.append({"scope": "by_dataset", "dataset": dataset, "method": method, **{f: values[f] for f in fields}, "num_samples": values["num_samples"], "ap_valid_count": values["ap_valid_count"]})
    overall = []
    for method in METHODS:
        values = aggregates[("ALL", method)].means()
        overall.append({"scope": "sample_overall", "dataset": "ALL", "method": method, **{f: values[f] for f in fields}, "num_samples": values["num_samples"], "ap_valid_count": values["ap_valid_count"]})
    macro = []
    for method in METHODS:
        method_rows = [row for row in by_dataset if row["method"] == method]
        row = {"scope": "dataset_macro", "dataset": "ALL", "method": method}
        for field in fields:
            finite = [float(item[field]) for item in method_rows if math.isfinite(float(item[field]))]
            row[field] = float(np.mean(finite)) if finite else float("nan")
        row["num_samples"] = sum(int(item["num_samples"]) for item in method_rows)
        row["ap_valid_count"] = sum(int(item["ap_valid_count"]) for item in method_rows)
        macro.append(row)
    return by_dataset, overall, macro


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


def _markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    headings = ("Dataset", "Method", *fields)
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in rows:
        values = [row["dataset"], row["method"]] + [f"{float(row[field]):.4f}" for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _success_assessment(hard_by_dataset: list[dict], hard_macro: list[dict]) -> dict:
    macro = {row["method"]: row for row in hard_macro}
    by = {(row["dataset"], row["method"]): row for row in hard_by_dataset}
    r1 = macro["R1"]
    result = {}
    for method in ("C1-FRank-R1Dist", "C2-MedianRank-R1Dist"):
        drops = [by[(dataset, method)]["hard_F_beta_w"] - by[(dataset, "R1")]["hard_F_beta_w"] for dataset in DATASET_ORDER]
        no_large_drop = min(drops) >= -0.010
        basic = macro[method]["hard_F_beta_w"] > 0.6243 and macro[method]["hard_MAE"] <= 0.0815 and no_large_drop
        wins = sum(by[(dataset, method)]["hard_F_beta_w"] >= by[(dataset, "R1")]["hard_F_beta_w"] for dataset in DATASET_ORDER)
        clear = macro[method]["hard_F_beta_w"] >= 0.6300 and wins >= 3 and by[("TE-COD10K", method)]["hard_F_beta_w"] >= 0.5806 and macro[method]["hard_MAE"] <= 0.0805
        strong = macro[method]["hard_F_beta_w"] >= 0.6350 and no_large_drop
        result[method] = {"basic": bool(basic), "clear": bool(clear), "strong": bool(strong), "dataset_f_beta_w_deltas_vs_r1": dict(zip(DATASET_ORDER, drops))}
    result["all_candidates_fail_r1"] = all(macro[name]["hard_F_beta_w"] <= r1["hard_F_beta_w"] for name in ("C0-F-MinMax", "C1-FRank-R1Dist", "C2-MedianRank-R1Dist"))
    return result


def _write_readme(out_dir: Path):
    content = """# DABE rank-calibration v1 evaluation

This directory is a formal original-GT-size audit of the seven frozen methods.

- Native responses and C0/C1/C2 use 37→68→original-size bilinear resizing with `align_corners=False`.
- Current-DABE-v2 uses cached `p_dabe_68` and 68→original-size bilinear resizing.
- GT is grayscale, normalized to [0,1], and binarized by strict `>0.5`.
- Hard prediction uses strict `probability > 0.5`; no threshold is selected.
- Official Soft uses the repository COD formulas. The repository `CODMetrics` protocol performs per-image min-max normalization for every non-constant continuous prediction.
- Raw Continuous never performs an extra min-max normalization.
- `best_IoU_256` uses the fixed diagnostic grid `torch.linspace(0,1,256)` and is never reused as a Hard threshold.
- Dataset Macro is the unweighted mean of CHAMELEON, TE-CAMO, TE-COD10K and NC4K.
- Pairwise deltas are `method_a - method_b`; positive MAE deltas therefore mean larger error.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def _write_results(out_dir: Path, hard_by: list[dict], hard_macro: list[dict], official_macro: list[dict], raw_macro: list[dict], ranking_macro: list[dict], baseline: dict, success: dict, full_run: bool):
    hard_rows = hard_by + [{**row, "dataset": "Dataset-Macro"} for row in hard_macro]
    hard_fields = ("hard_S_m", "hard_F_beta_w", "hard_E_mean", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area")
    macro = {row["method"]: row for row in hard_macro}
    official = {row["method"]: row for row in official_macro}
    raw = {row["method"]: row for row in raw_macro}
    ranking = {row["method"]: row for row in ranking_macro}
    baseline_text = "通过（所有核查字段绝对误差 ≤ 1e-5）" if baseline.get("passed") else ("未通过；停止给出方法结论" if full_run else "sanity run 不适用")
    lines = [
        "# DABE-TF 第一轮残差排序—分数校准解耦实验结果",
        "",
        "## 协议确认",
        "",
        "- DINO rerun: false",
        "- DABE rebuild: false",
        "- GT used in candidate generation: false",
        "- Source view: identity only",
        "- Source backbone: DINOv1-S/8",
        "- Hard threshold: strict > 0.5",
        "- Dataset-specific parameters: none",
        "- Candidate methods fixed before evaluation: C0/C1/C2",
        "- Official Soft 注意：仓库 CODMetrics 对每张非恒定 continuous prediction 做逐图 min-max normalization。",
        "",
        "## 基线复现",
        "",
        f"结论：{baseline_text}。",
        "",
        "## Hard 主结果",
        "",
        _markdown_table(hard_rows, hard_fields),
        "",
    ]
    if not full_run or not baseline.get("passed"):
        lines += ["## 方法结论", "", "因当前不是完整评测或基线未通过，不给出方法结论。", ""]
    else:
        c0_gain = macro["C0-F-MinMax"]["hard_F_beta_w"] - macro["F"]["hard_F_beta_w"]
        c1 = macro["C1-FRank-R1Dist"]
        c2 = macro["C2-MedianRank-R1Dist"]
        r1 = macro["R1"]
        c1_ap = ranking["C1-FRank-R1Dist"]["pixel_AP"] - ranking["R1"]["pixel_AP"]
        c2_ap = ranking["C2-MedianRank-R1Dist"]["pixel_AP"] - ranking["R1"]["pixel_AP"]
        c1_area = c1["hard_Area"] - r1["hard_Area"]
        c2_area = c2["hard_Area"] - r1["hard_Area"]
        lines += [
            "## Official Soft / Raw / Ranking（Dataset Macro）",
            "",
            _markdown_table(official_macro, ("official_soft_F_beta_w", "official_soft_F_beta_max", "official_soft_MAE")),
            "",
            _markdown_table(raw_macro, ("raw_MAE", "raw_Brier", "raw_SoftIoU", "raw_prob_mean")),
            "",
            _markdown_table(ranking_macro, ("pixel_AP", "best_IoU_256")),
            "",
            "## 机制判断",
            "",
            f"1. C0 相对 F 的 Hard F_beta^w 改变量为 {c0_gain:+.6f}；" + ("该结果支持 F 存在固定阈值尺度压缩。" if c0_gain > 1e-6 else "该结果不支持尺度压缩是 F 掉点的主要原因。"),
            f"2. C1 相对 R1：Precision {c1['hard_Precision']-r1['hard_Precision']:+.6f}，Recall {c1['hard_Recall']-r1['hard_Recall']:+.6f}，F_beta^w {c1['hard_F_beta_w']-r1['hard_F_beta_w']:+.6f}。",
            f"3. C2 相对 C1 的 Dataset-Macro F_beta^w 为 {c2['hard_F_beta_w']-c1['hard_F_beta_w']:+.6f}；逐数据集稳定性见 pairwise_by_dataset.csv。",
            f"4. C1/C2 相对 R1 的 Pixel AP 分别为 {c1_ap:+.6f} / {c2_ap:+.6f}。",
            "5. C1/C2 是否形成统一收益由四数据集逐项差值判定，详见 success_assessment 与 pairwise 表。",
            f"6. C1/C2 相对 R1 的 Hard Area 分别为 {c1_area:+.6f} / {c2_area:+.6f}。",
            "7. 面积近似不变时的性能差异可归因于像素重排序；面积明显变化时不能只作此归因。",
            "8. 若 C0/C1/C2 均未超过 R1，则当前证据支持 R1 的空间排序本身更可靠，并应停止 rank-transport 路线。",
            "",
            "## 成功判定",
            "",
            "```json",
            json.dumps(success, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        if success["all_candidates_fail_r1"]:
            lines.append("结论：三个候选均未超过 R1；停止 rank-transport，本任务不实现 LOO，下一步转向 Leave-One-Out Local Background-Null Calibration。")
        else:
            lines.append("结论：候选是否可冻结以预定义的 basic / clear / strong 判据为准；本评测未进行阈值搜索或数据集特定选择。")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_rank_calibration(config_path, dabe_root, rankcal_root, out_dir, split="test", max_samples=-1, threshold=0.5, workers=None, torch_threads=1, overwrite=False, overwrite_reason=""):
    started = time.time()
    config_path = Path(config_path).resolve()
    dabe_root = Path(dabe_root).resolve()
    rankcal_root = Path(rankcal_root).resolve()
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
    rank_rows_all, rank_map_all = _manifest_map(rankcal_root / f"manifest_{split}.jsonl", 6473 if full_run else None)
    source_rows = source_rows_all if full_run else source_rows_all[:max_samples]
    selected_keys = [(row["dataset"], row["stem"]) for row in source_rows]
    rank_keys = set(rank_map_all)
    if full_run and rank_keys != set(source_map):
        raise RuntimeError("Full source and rankcal manifest keys differ")
    missing = [key for key in selected_keys if key not in rank_map_all]
    if missing:
        raise RuntimeError(f"Rankcal manifest missing selected keys: {missing[:10]}")
    if not full_run and set(selected_keys) != rank_keys:
        raise RuntimeError("Sanity rankcal manifest must exactly match the selected source subset")
    datasets = [name for name in DATASET_ORDER if any(key[0] == name for key in selected_keys)]
    if full_run and datasets != list(DATASET_ORDER):
        raise RuntimeError(f"Unexpected datasets: {datasets}")
    for row in source_rows:
        if "gt_path" not in row or not Path(row["gt_path"]).is_file():
            raise FileNotFoundError(f"GT missing for {row['dataset']}/{row['stem']}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_dir: {out_dir}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (out_dir / "eval.log").open("w", encoding="utf-8", buffering=1)

    def log_message(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    workers = workers or min(12, os.cpu_count() or 1)
    tasks = [
        {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "source_cache_path": row["cache_path"],
            "rankcal_cache_path": rank_map_all[(row["dataset"], row["stem"])]["cache_path"],
            "gt_path": row["gt_path"],
            "threshold": threshold,
        }
        for row in source_rows
    ]
    aggregates = {(scope, method): Aggregate() for scope in [*datasets, "ALL"] for method in METHODS}
    pair_values = defaultdict(list)
    per_sample_path = out_dir / "per_sample.csv"
    pair_sample_path = out_dir / "pairwise_per_sample.csv"
    pair_sample_fields = ("dataset", "stem", "method_a", "method_b", *PAIRWISE_FIELDS)
    try:
        log_message(f"num_samples = {len(tasks)}")
        log_message(f"workers = {workers}")
        log_message("hard_threshold = strict > 0.5")
        log_message("candidate_generation_in_eval = false")
        with per_sample_path.open("w", encoding="utf-8", newline="") as sample_handle, pair_sample_path.open("w", encoding="utf-8", newline="") as pair_handle:
            sample_writer = csv.DictWriter(sample_handle, fieldnames=PER_SAMPLE_FIELDS, extrasaction="ignore")
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_sample_fields, extrasaction="ignore")
            sample_writer.writeheader()
            pair_writer.writeheader()
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(torch_threads,)) as executor:
                for index, result in enumerate(executor.map(_evaluate_one, tasks, chunksize=1), 1):
                    method_rows = {row["method"]: row for row in result["rows"]}
                    for row in result["rows"]:
                        aggregates[(row["dataset"], row["method"])].add(row)
                        aggregates[("ALL", row["method"])].add(row)
                        sample_writer.writerow(row)
                    for method_a, method_b in PAIRWISE:
                        a, b = method_rows[method_a], method_rows[method_b]
                        pair_row = {"dataset": result["dataset"], "stem": result["stem"], "method_a": method_a, "method_b": method_b}
                        mapping = {
                            "delta_hard_F_beta_w": "hard_F_beta_w",
                            "delta_hard_IoU": "hard_IoU",
                            "delta_hard_Precision": "hard_Precision",
                            "delta_hard_Recall": "hard_Recall",
                            "delta_hard_MAE": "hard_MAE",
                            "delta_hard_Area": "hard_Area",
                            "delta_official_soft_F_beta_w": "official_soft_F_beta_w",
                            "delta_raw_MAE": "raw_MAE",
                            "delta_pixel_AP": "pixel_AP",
                            "delta_best_IoU_256": "best_IoU_256",
                        }
                        for delta_field, source_field in mapping.items():
                            av, bv = float(a[source_field]), float(b[source_field])
                            delta = av - bv if math.isfinite(av) and math.isfinite(bv) else float("nan")
                            pair_row[delta_field] = delta
                            if math.isfinite(delta):
                                pair_values[(result["dataset"], method_a, method_b, delta_field)].append(delta)
                        pair_writer.writerow(pair_row)
                    if index == len(tasks) or index % 100 == 0:
                        log_message(f"processed = {index}/{len(tasks)}")

        sections = {}
        for section in ("hard", "official_soft", "raw_continuous", "ranking"):
            by, overall, macro = _section_rows(aggregates, datasets, section)
            sections[section] = {"by_dataset": by, "sample_overall": overall, "dataset_macro": macro}
            fields = {"hard": HARD_FIELDS, "official_soft": OFFICIAL_FIELDS, "raw_continuous": RAW_FIELDS, "ranking": RANKING_FIELDS}[section]
            csv_fields = ("scope", "dataset", "method", *fields, "num_samples", "ap_valid_count")
            _write_csv(out_dir / f"{section}_by_dataset.csv", by, csv_fields)
            _write_csv(out_dir / f"{section}_sample_overall.csv", overall, csv_fields)
            _write_csv(out_dir / f"{section}_dataset_macro.csv", macro, csv_fields)

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
        _write_csv(out_dir / "pairwise_by_dataset.csv", pair_summary, ("dataset", "method_a", "method_b", "metric", "mean_delta", "median_delta", "p25_delta", "p75_delta", "win_ratio", "tie_ratio", "loss_ratio", "valid_count"))

        hard_macro_map = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
        baseline = {"applicable": full_run, "tolerance": 1e-5, "checks": {}}
        if full_run:
            passed = True
            for method, expected_fields in BASELINE_EXACT.items():
                baseline["checks"][method] = {}
                for field, expected in expected_fields.items():
                    actual = float(hard_macro_map[method][field])
                    error = abs(actual - expected)
                    ok = error <= 1e-5
                    passed = passed and ok
                    baseline["checks"][method][field] = {"actual": actual, "expected": expected, "absolute_error": error, "passed": ok}
            baseline["passed"] = passed
        else:
            baseline["passed"] = None
        success = _success_assessment(sections["hard"]["by_dataset"], sections["hard"]["dataset_macro"]) if full_run and baseline["passed"] else {"not_evaluated": True}

        git_commit, git_status = _git_metadata()
        protocol = {
            "split": split,
            "num_samples": len(tasks),
            "full_run": full_run,
            "methods": list(METHODS),
            "pairwise_comparisons": [list(item) for item in PAIRWISE],
            "hard_threshold": threshold,
            "hard_operator": ">",
            "native_resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
            "current_resize": "p_dabe_68 -> original GT bilinear align_corners=False",
            "gt_binarization": "grayscale / 255, strict > 0.5",
            "official_soft_per_image_minmax": True,
            "raw_continuous_extra_minmax": False,
            "best_iou_thresholds": "torch.linspace(0.0, 1.0, 256), diagnostic only",
            "candidate_generation_in_eval": False,
            "dataset_specific_rule": False,
            "threshold_search_used_for_hard": False,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_manifest": str((dabe_root / f"manifest_{split}.jsonl").resolve()),
            "source_manifest_sha256": _sha256(dabe_root / f"manifest_{split}.jsonl"),
            "rankcal_manifest": str((rankcal_root / f"manifest_{split}.jsonl").resolve()),
            "rankcal_manifest_sha256": _sha256(rankcal_root / f"manifest_{split}.jsonl"),
            "evaluator_sha256": _sha256(SCRIPT_PATH),
            "git_commit": git_commit,
            "git_status_short": git_status,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "overwrite": overwrite,
            "overwrite_reason": overwrite_reason.strip(),
            "elapsed_seconds": time.time() - started,
        }
        summary = {"protocol": protocol, "baseline_reproduction": baseline, "success_assessment": success, **sections, "pairwise_by_dataset": pair_summary}
        write_json(out_dir / "protocol.json", protocol)
        write_json(out_dir / "summary.json", summary)
        _write_readme(out_dir)
        _write_results(out_dir, sections["hard"]["by_dataset"], sections["hard"]["dataset_macro"], sections["official_soft"]["dataset_macro"], sections["raw_continuous"]["dataset_macro"], sections["ranking"]["dataset_macro"], baseline, success, full_run)
        log_message(f"baseline_reproduction = {baseline.get('passed')}")
        log_message(f"elapsed_seconds = {time.time() - started:.3f}")
        return summary
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--rankcal_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rank_calibration(
        config_path=args.config,
        dabe_root=args.dabe_root,
        rankcal_root=args.rankcal_root,
        out_dir=args.out_dir,
        split=args.split,
        max_samples=args.max_samples,
        threshold=args.threshold,
        workers=args.workers,
        torch_threads=args.torch_threads,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
    )
