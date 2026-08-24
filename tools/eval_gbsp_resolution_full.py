#!/usr/bin/env python3
"""Full-set continuous and diagnostic evaluation for frozen 512-A/B/C GBSP scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.metrics import _prepare_data  # noqa: E402
from common.utils import read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from tools.eval_gbsp_resolution_probe import (  # noqa: E402
    _boundary_f1,
    _load_gt,
    _rank_metrics,
    _sample_rows,
    _write_csv,
)


METHODS = ("512-A", "512-B", "512-C")
METHOD_DIRS = {
    "512-A": "A_bw2_r8",
    "512-B": "B_bw3_r8",
    "512-C": "C_bw2_r12",
}
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
FIXED_FIELDS = (
    "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision",
    "Recall", "Area", "IoU", "Dice", "boundary_F1",
)
CONTINUOUS_FIELDS = ("pixel_AP", "pixel_AUROC")
CURVE_SIZE = 256
HIST_BINS = 65536


def _safe_nanmean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if bool(finite.any()) else float("nan")


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _score_path(root: Path, method: str, dataset: str, stem: str) -> Path:
    return root / METHOD_DIRS[method] / "scores" / dataset / f"{stem}.pt"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: float("nan") for key in ("mean", "median", "p10", "p25", "p50", "p75", "p90", "std")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, .10)),
        "p25": float(np.quantile(values, .25)),
        "p50": float(np.quantile(values, .50)),
        "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)),
        "std": float(values.std()),
    }


def _score_histogram(prediction: np.ndarray, gt: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    value = prediction.astype(np.float64, copy=False)
    low, high = float(value.min()), float(value.max())
    if high > low:
        value = (value - low) / (high - low)
    else:
        value = np.zeros_like(value)
    index = np.minimum((value * (bins - 1)).astype(np.int64), bins - 1)
    fg = np.bincount(index[gt], minlength=bins).astype(np.int64)
    bg = np.bincount(index[~gt], minlength=bins).astype(np.int64)
    return fg, bg


def _hist_ap_auroc(fg_hist: np.ndarray, bg_hist: np.ndarray) -> tuple[float, float]:
    positives, negatives = int(fg_hist.sum()), int(bg_hist.sum())
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    tp = np.cumsum(fg_hist[::-1], dtype=np.float64)
    fp = np.cumsum(bg_hist[::-1], dtype=np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    negative_below = np.cumsum(bg_hist, dtype=np.float64) - bg_hist
    wins = float(np.sum(fg_hist * (negative_below + .5 * bg_hist)))
    return ap, wins / (positives * negatives)


def _curves(prediction: np.ndarray, gt: np.ndarray, context: FastCODContext) -> dict[str, np.ndarray]:
    pred, target = _prepare_data(gt=gt.astype(float), pred=prediction.astype(float))
    precision, recall, f_curve = context.fmeasure.cal_pr(pred, target)
    context.emeasure.gt_fg_numel = int(np.count_nonzero(target))
    context.emeasure.gt_size = int(target.size)
    e_curve = context.emeasure.cal_changeable_em(pred, target)
    denom = precision + recall
    dice = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    iou_denom = precision + recall - precision * recall
    iou = np.divide(precision * recall, iou_denom, out=np.zeros_like(iou_denom), where=iou_denom > 0)
    return {
        "precision": np.asarray(precision, dtype=np.float32),
        "recall": np.asarray(recall, dtype=np.float32),
        "f": np.asarray(f_curve, dtype=np.float32),
        "e": np.asarray(e_curve, dtype=np.float32),
        "dice": np.asarray(dice, dtype=np.float32),
        "iou": np.asarray(iou, dtype=np.float32),
    }


def _load_score(path: Path, method: str, dataset: str, stem: str) -> tuple[dict, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch_load(path, map_location="cpu")
    if (payload.get("method"), payload.get("dataset"), payload.get("stem")) != (method, dataset, stem):
        raise RuntimeError(f"score identity mismatch: {path}")
    score = payload.get("absolute_minmax")
    if not torch.is_tensor(score) or tuple(score.shape) != (1, 64, 64):
        raise RuntimeError(f"score shape mismatch: {path}")
    if not bool(torch.isfinite(score).all()) or float(score.min()) < 0 or float(score.max()) > 1:
        raise RuntimeError(f"score numerical failure: {path}")
    return payload, score.float().contiguous()


def _aggregate_metrics(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    output: list[dict] = []
    for method in METHODS:
        by_dataset = []
        for dataset in DATASETS:
            selected = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
            summary = {"scope": "dataset", "dataset": dataset, "method": method, "num_images": len(selected)}
            for field in fields:
                summary[field] = _safe_nanmean(float(row[field]) for row in selected)
            output.append(summary)
            by_dataset.append(summary)
        macro = {"scope": "dataset_macro", "dataset": "ALL", "method": method, "num_images": sum(row["num_images"] for row in by_dataset)}
        for field in fields:
            macro[field] = _safe_nanmean(row[field] for row in by_dataset)
        output.append(macro)
        selected = [row for row in rows if row["method"] == method]
        image_macro = {"scope": "image_macro", "dataset": "ALL", "method": method, "num_images": len(selected)}
        for field in fields:
            image_macro[field] = _safe_nanmean(float(row[field]) for row in selected)
        output.append(image_macro)
    return output


def _aggregate_curve_records(records: list[dict], scope: str = "all") -> tuple[list[dict], dict[tuple[str, str], dict]]:
    threshold = np.arange(255, -1, -1, dtype=np.float64) / 255.0
    curve_rows: list[dict] = []
    summaries: dict[tuple[str, str], dict] = {}
    for method in METHODS:
        dataset_curves: dict[str, dict[str, np.ndarray]] = {}
        for dataset in DATASETS:
            selected = [row for row in records if row["method"] == method and row["dataset"] == dataset]
            if not selected:
                continue
            means = {
                field: np.mean(np.stack([row[field] for row in selected]), axis=0)
                for field in ("precision", "recall", "f", "e", "dice", "iou")
            }
            dataset_curves[dataset] = means
            summary = {"scope": scope, "dataset": dataset, "method": method, "num_images": len(selected)}
            for field, label in (("f", "F"), ("e", "E"), ("dice", "Dice"), ("iou", "IoU")):
                index = int(np.nanargmax(means[field]))
                summary[f"max{label}"] = float(means[field][index])
                summary[f"best_{label}_threshold"] = float(threshold[index])
            summaries[(method, dataset)] = summary
        macro = {
            field: np.mean(np.stack([dataset_curves[dataset][field] for dataset in DATASETS if dataset in dataset_curves]), axis=0)
            for field in ("precision", "recall", "f", "e", "dice", "iou")
        }
        summary = {"scope": scope, "dataset": "ALL", "method": method, "num_images": sum(int(summaries[(method, d)]["num_images"]) for d in dataset_curves)}
        for field, label in (("f", "F"), ("e", "E"), ("dice", "Dice"), ("iou", "IoU")):
            index = int(np.nanargmax(macro[field]))
            summary[f"max{label}"] = float(macro[field][index])
            summary[f"best_{label}_threshold"] = float(threshold[index])
        summaries[(method, "ALL")] = summary
        for dataset, curves in [*(dataset_curves.items()), ("Dataset-Macro", macro)]:
            for index, tau in enumerate(threshold):
                curve_rows.append({
                    "scope": scope,
                    "dataset": dataset,
                    "method": method,
                    "threshold": float(tau),
                    **{field: float(curves[field][index]) for field in ("precision", "recall", "f", "e", "dice", "iou")},
                })
    return curve_rows, summaries


def _read_296_continuous(summary_path: Path, per_dataset_path: Path) -> list[dict]:
    rows = [
        *csv.DictReader(summary_path.open(encoding="utf-8")),
        *csv.DictReader(per_dataset_path.open(encoding="utf-8")),
    ]
    output = []
    seen: set[tuple[str, str]] = set()
    dataset_aliases = {"CAMO": "TE-CAMO", "COD10K": "TE-COD10K"}
    for row in rows:
        if row.get("method") != "gbsp_r8":
            continue
        protocol = row.get("protocol")
        if protocol not in {
            "dataset_macro_of_per_image_native",
            "per_image_macro_native_gt",
            "pooled_native_hist65536_per_image_minmax",
        }:
            continue
        dataset = dataset_aliases.get(row["dataset"], row["dataset"])
        if protocol == "pooled_native_hist65536_per_image_minmax":
            if dataset != "ALL":
                continue
            scope = "global_pooled_hist65536_per_image_minmax"
        else:
            scope = "dataset_macro" if dataset == "DATASET_MACRO" else ("image_macro" if dataset == "ALL" else "dataset")
        identity = (scope, "ALL" if dataset in {"ALL", "DATASET_MACRO"} else dataset)
        if identity in seen:
            continue
        seen.add(identity)
        output.append({
            "scope": scope,
            "dataset": "ALL" if dataset in {"ALL", "DATASET_MACRO"} else dataset,
            "method": "296-baseline",
            "num_images": int(row["valid_images"]),
            "pixel_AP": float(row["ap"]),
            "pixel_AUROC": float(row["auroc"]),
            "protocol": protocol,
        })
    return output


def _read_296_fixed(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    output = []
    for row in rows:
        if row.get("method") != "background_source/fullbc" or abs(float(row.get("threshold", -1)) - .58) > 1e-9:
            continue
        output.append({
            "scope": row["scope"],
            "dataset": row["dataset"],
            "method": "296-baseline",
            "num_images": int(row["num_samples"]),
            **{field: float(row[field]) for field in ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area", "IoU", "Dice")},
            "boundary_F1": float("nan"),
            "threshold": .58,
        })
    return output


def _plot_pr(curve_rows: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for method in METHODS:
        rows = [row for row in curve_rows if row["method"] == method and row["dataset"] == "Dataset-Macro" and row["scope"] == "all"]
        axis.plot([row["recall"] for row in rows], [row["precision"] for row in rows], label=method)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _old_296_per_image(path: Path) -> dict[tuple[str, str], dict]:
    mapping = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row.get("method") == "rank/current" and abs(float(row.get("threshold", -1)) - .58) < 1e-9:
            mapping[(row["dataset"], row["stem"])] = row
    return mapping


def _visualize(
    samples: list[dict], metric_rows: list[dict], dist_rows: list[dict], score_root: Path,
    old_core_root: Path, old_rows: dict[tuple[str, str], dict], output: Path,
) -> list[dict]:
    sample_map = {(str(row["dataset"]), str(row["stem"])): row for row in samples}
    metrics = {(row["dataset"], row["stem"], row["method"]): row for row in metric_rows}
    bg_std = {(row["dataset"], row["image"], row["method"]): row["BG_std"] for row in dist_rows}
    candidates = []
    for key, old in old_rows.items():
        if key not in sample_map or (*key, "512-C") not in metrics:
            continue
        current = metrics[(*key, "512-C")]
        delta_ap = float(current["pixel_AP"]) - float(old["pixel_AP"])
        delta_dice = float(current["Dice"]) - float(old["Dice"])
        candidates.append({
            "dataset": key[0], "stem": key[1], "delta_AP": delta_ap,
            "delta_Dice": delta_dice, "gain_score": delta_ap + .25 * delta_dice,
            "gt_area": float(current["gt_area"]), "BG_std": float(bg_std[(*key, "512-C")]),
        })
    groups = [
        ("largest_gain", sorted(candidates, key=lambda row: -row["gain_score"])[:10]),
        ("largest_drop", sorted(candidates, key=lambda row: row["gain_score"])[:10]),
        ("small_object", sorted(candidates, key=lambda row: row["gt_area"])[:10]),
        ("complex_background_proxy", sorted(candidates, key=lambda row: -row["BG_std"])[:10]),
    ]
    output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for category, selected in groups:
        category_root = output / category
        category_root.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(selected, 1):
            key = (row["dataset"], row["stem"])
            sample = sample_map[key]
            with Image.open(sample["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            gt = _load_gt(sample["gt_path"])
            original_hw = tuple(gt.shape[-2:])
            scores, masks = {}, {}
            for method in ("512-A", "512-C"):
                _, score = _load_score(_score_path(score_root, method, *key), method, *key)
                scores[method] = F.interpolate(score.unsqueeze(0), size=original_hw, mode="bilinear", align_corners=False).squeeze().numpy()
                masks[method] = F.interpolate((score > .58).float().unsqueeze(0), size=original_hw, mode="nearest").squeeze().numpy()
            old_path = old_core_root / "test" / key[0] / f"{key[1]}.pt"
            old_payload = torch_load(old_path, map_location="cpu")
            old_score = old_payload["results"]["current"]["absolute_minmax"].float()
            old_score = F.interpolate(old_score.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False)
            old_score = F.interpolate(old_score, size=original_hw, mode="bilinear", align_corners=False).squeeze().numpy()
            panels = [
                (rgb, "RGB", None), (gt.squeeze().numpy(), "GT", "gray"),
                (old_score, "296 continuous", "gray"),
                (scores["512-A"], "512-A continuous", "gray"),
                (scores["512-C"], "512-C continuous", "gray"),
                (masks["512-A"], "512-A hard@.58", "gray"),
                (masks["512-C"], "512-C hard@.58", "gray"),
            ]
            fig, axes = plt.subplots(1, len(panels), figsize=(22, 4))
            for axis, (value, title, cmap) in zip(axes, panels):
                axis.imshow(value, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            fig.suptitle(f"{key[0]}/{key[1]} | ΔAP={row['delta_AP']:+.4f} ΔDice={row['delta_Dice']:+.4f}")
            fig.tight_layout()
            path = category_root / f"{index:02d}_{key[0]}_{key[1]}.png"
            fig.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            index_rows.append({"category": category, "index": index, **row, "path": str(path)})
    _write_csv(output / "visualization_index.csv", index_rows)
    return index_rows


def _markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" if field in {"method", "dataset"} else "---:" for field in fields) + "|"
    lines = [header, divider]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "N/A")
            if isinstance(value, float):
                values.append("N/A" if not math.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path, continuous: list[dict], fixed: list[dict], distributions: list[dict],
    pca_summary: list[dict], small: list[dict], curve_rows: list[dict], sources: dict,
    num_valid: int, visual_count: int,
) -> None:
    cont_macro = {row["method"]: row for row in continuous if row["scope"] == "dataset_macro"}
    fixed_macro = {row["method"]: row for row in fixed if row["scope"] == "dataset_macro"}
    dist_macro = {row["method"]: row for row in distributions if row["scope"] == "dataset_macro"}
    pca = {row["method"]: row for row in pca_summary}
    small_macro = {row["method"]: row for row in small if row["scope"] == "dataset_macro"}
    curve_macro = {
        method: [row for row in curve_rows if row["scope"] == "all" and row["dataset"] == "Dataset-Macro" and row["method"] == method]
        for method in METHODS
    }
    c_beats_a = float(np.mean([
        c["f"] > a["f"] for a, c in zip(curve_macro["512-A"], curve_macro["512-C"])
    ]))
    ap_consistency = {
        method: sum(
            next(row for row in continuous if row["method"] == method and row["dataset"] == dataset and row["scope"] == "dataset")["pixel_AP"]
            > next(row for row in continuous if row["method"] == "296-baseline" and row["dataset"] == dataset and row["scope"] == "dataset")["pixel_AP"]
            for dataset in DATASETS
        )
        for method in METHODS
    }
    best512 = max(METHODS, key=lambda method: cont_macro[method]["pixel_AP"])
    ap_gain = cont_macro[best512]["pixel_AP"] - cont_macro["296-baseline"]["pixel_AP"]
    auc_gain = cont_macro[best512]["pixel_AUROC"] - cont_macro["296-baseline"]["pixel_AUROC"]
    fixed_drop = fixed_macro[best512]["F_beta_w"] - fixed_macro["296-baseline"]["F_beta_w"]
    if ap_gain < 0 and auc_gain < 0:
        case = "Case 4：512没有提高全量连续残差质量，停止512主线。"
    elif ap_gain > 0 and auc_gain > 0:
        case = "Case 1候选：连续排序提高；若固定阈值仍下降，则主要矛盾转向分辨率稳定校准。"
    elif ap_gain > 0:
        case = "Case 2/3：AP提高但AUROC未同步，属于局部排序/边界潜力而非全面判别提升。"
    else:
        case = "Case 3/4：高分辨率没有形成稳定的全局排序增益。"

    cont_rows = []
    for method in ("296-baseline", *METHODS):
        row = cont_macro[method]
        cont_rows.append({
            "method": method, "AP": row["pixel_AP"], "AUROC": row["pixel_AUROC"],
            "maxF": row.get("maxF", float("nan")), "best-F-tau": row.get("best_F_threshold", float("nan")),
            "maxE": row.get("maxE", float("nan")), "best-E-tau": row.get("best_E_threshold", float("nan")),
            "Boundary-F1@.58": fixed_macro[method].get("boundary_F1", float("nan")),
            "maxBoundaryF1": float("nan"),
        })
    fixed_rows = [{"method": method, **{field: fixed_macro[method][field] for field in ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")}} for method in ("296-baseline", *METHODS)]
    aggregation_rows = []
    for method in ("296-baseline", *METHODS):
        for scope, label in (
            ("dataset_macro", "dataset-macro"),
            ("image_macro", "6473-image-macro"),
            ("global_pooled_hist65536_per_image_minmax", "global-pooled-hist65536"),
        ):
            matches = [row for row in continuous if row["method"] == method and row["scope"] == scope]
            if matches:
                aggregation_rows.append({
                    "method": method, "aggregation": label,
                    "AP": matches[0]["pixel_AP"], "AUROC": matches[0]["pixel_AUROC"],
                })
    dist_rows = [{"method": method, **{field: dist_macro[method][field] for field in ("FG_mean", "FG_median", "BG_mean", "BG_median", "mean_gap", "median_gap")}} for method in METHODS]
    pca_rows = [{"method": method, **pca[method]} for method in METHODS]
    small_rows = [{"method": method, **{field: small_macro[method].get(field, float("nan")) for field in ("num_images", "pixel_AP", "pixel_AUROC", "maxF", "boundary_F1")}} for method in ("296-baseline", *METHODS) if method in small_macro]

    per_dataset = [row for row in continuous if row["scope"] == "dataset"]
    lines = [
        "# 512分辨率 GBSP 全量连续分数评估（A/B/C）", "",
        f"- 完整性：valid = {num_valid} / 6473，failed = {6473 - num_valid}。",
        "- 生成阶段未读取GT；GT仅用于离线评估。未训练，未重跑296。",
        f"- 296 continuous source: `{sources['continuous_296']}`",
        f"- 296 per-dataset continuous source: `{sources['continuous_296_per_dataset']}`",
        f"- 296 fixed source: `{sources['fixed_296']}`", "",
        "## 核心判断", "", f"**{case}**", "",
        f"全量AP最好的512版本是 **{best512}**：相对296 AP {ap_gain:+.6f}，AUROC {auc_gain:+.6f}，固定阈值Fw {fixed_drop:+.6f}。",
        f"A→C的Dataset-Macro F曲线在 {c_beats_a * 100:.1f}% 的阈值点上更高。", "",
        "## Continuous Quality（主表）", "",
        "主表采用 dataset-macro：先在各数据集内逐图平均，再对四个数据集等权平均。", "",
        _markdown_table(cont_rows, ("method", "AP", "AUROC", "maxF", "best-F-tau", "maxE", "best-E-tau", "Boundary-F1@.58", "maxBoundaryF1")), "",
        "`maxF/maxE`仅为oracle operating-point诊断；296未重跑，因此对应项为N/A。", "",
        "## AP/AUROC aggregation audit", "",
        _markdown_table(aggregation_rows, ("method", "aggregation", "AP", "AUROC")), "",
        "global-pooled 使用每图 Min-Max 后的 65536-bin pooled histogram；它与 dataset-macro/image-macro 是不同聚合口径，不混作主指标。", "",
        "## Per-dataset AP/AUROC", "",
        _markdown_table(per_dataset, ("dataset", "method", "pixel_AP", "pixel_AUROC")), "",
        "512相对296的AP提高数据集数：" + "；".join(f"{method}={count}/4" for method, count in ap_consistency.items()) + "。", "",
        "## Fixed-threshold diagnostic only", "",
        _markdown_table(fixed_rows, ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")), "",
        "## Score Distribution", "",
        _markdown_table(dist_rows, ("method", "FG_mean", "FG_median", "BG_mean", "BG_median", "mean_gap", "median_gap")), "",
        "## PCA Rank", "",
        _markdown_table(pca_rows, ("method", "rank_cap", "mean_required_rank", "hit_cap_ratio", "mean_energy_at_cap")), "",
        "## Small-object（全量GT面积最低25%）", "",
        _markdown_table(small_rows, ("method", "num_images", "pixel_AP", "pixel_AUROC", "maxF", "boundary_F1")), "",
        "## 任务书15项回答", "",
        f"1. 512-A full AP是否高于296：{cont_macro['512-A']['pixel_AP'] > cont_macro['296-baseline']['pixel_AP']}。",
        f"2. 512-C full AP是否高于296：{cont_macro['512-C']['pixel_AP'] > cont_macro['296-baseline']['pixel_AP']}。",
        f"3. AUROC趋势是否与AP一致：{np.sign(cont_macro[best512]['pixel_AP'] - cont_macro['296-baseline']['pixel_AP']) == np.sign(cont_macro[best512]['pixel_AUROC'] - cont_macro['296-baseline']['pixel_AUROC'])}。",
        f"4. A→C是否改善ranking：AP差={cont_macro['512-C']['pixel_AP'] - cont_macro['512-A']['pixel_AP']:+.6f}，AUROC差={cont_macro['512-C']['pixel_AUROC'] - cont_macro['512-A']['pixel_AUROC']:+.6f}。",
        f"5. A→C Recall/Area：Recall {fixed_macro['512-A']['Recall']:.6f}→{fixed_macro['512-C']['Recall']:.6f}；Area {fixed_macro['512-A']['Area']:.6f}→{fixed_macro['512-C']['Area']:.6f}。",
        "6. fixed-threshold掉点是否主要来自calibration shift：结合AP/AUROC、maxF与面积变化判断；若AP和maxF保留而固定Fw下降，才支持。",
        f"7. PR/F曲线：C在A之上的阈值比例为{c_beats_a * 100:.1f}%；296完整曲线不可恢复且按任务书未重跑。",
        "8. 512 Boundary gain是否在6473张仍成立：296全量Boundary F1不存在，不能作严格差值；报告A/B/C全量值，不包装结论。",
        "9. small-object gain：见上表；296 AP/Fw来自既有逐图结果，296 Boundary不存在。",
        f"10. BW3是否可放弃：A/B AP差={cont_macro['512-B']['pixel_AP'] - cont_macro['512-A']['pixel_AP']:+.6f}，结合四库一致性判断。",
        f"11. PCA rank是否主要瓶颈：C-A AP={cont_macro['512-C']['pixel_AP'] - cont_macro['512-A']['pixel_AP']:+.6f}，且C hit-cap={pca['512-C']['hit_cap_ratio']:.3f}；不得自动推导r16。",
        f"12. 512是否值得下一阶段：按核心判断“{case}”执行。",
        "13. 若值得继续：固定阈值失配且连续质量改善时研究calibration；连续质量未改善时停止，而非盲调graph/rank。",
        "14. 是否建议开始训练：否；等待人工确认。",
        "15. 是否建议继续r16：否。", "",
        f"- 可视化数量：{visual_count}。", "- 本任务到此停止，没有执行r16、阈值调参、graph sigma搜索或训练。", "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--continuous-296", required=True)
    parser.add_argument("--per-dataset-continuous-296", required=True)
    parser.add_argument("--fixed-296", required=True)
    parser.add_argument("--per-image-296", required=True)
    parser.add_argument("--core-296-root", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    started = time.perf_counter()
    main_root = Path(__file__).resolve().parents[1]
    score_root = Path(args.score_root).expanduser().resolve()
    sample_list = Path(args.sample_list).expanduser().resolve()
    samples = _sample_rows(sample_list, int(args.max_samples))
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    eval_cache = output / "eval_cache"
    eval_cache.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    (output / "evaluation_command.txt").write_text(command + "\n", encoding="utf-8")
    shutil.copy2(sample_list, output / "sample_list.jsonl")
    sample_sha = hashlib.sha256(sample_list.read_bytes()).hexdigest()

    metric_rows: list[dict] = []
    dist_rows: list[dict] = []
    pca_rows: list[dict] = []
    curve_records: list[dict] = []
    failures: list[dict] = []
    pooled_fg = {method: np.zeros(HIST_BINS, dtype=np.int64) for method in METHODS}
    pooled_bg = {method: np.zeros(HIST_BINS, dtype=np.int64) for method in METHODS}

    for index, sample in enumerate(samples, 1):
        dataset, stem = str(sample["dataset"]), str(sample["stem"])
        cache_path = eval_cache / dataset / f"{stem}.pt"
        try:
            gt = _load_gt(sample["gt_path"])
            gt_np = gt.squeeze().numpy() > .5
            original_hw = tuple(map(int, gt.shape[-2:]))
            payloads, scores64, continuous, hard = {}, {}, {}, {}
            for method in METHODS:
                payload, score = _load_score(_score_path(score_root, method, dataset, stem), method, dataset, stem)
                payloads[method], scores64[method] = payload, score
                continuous[method] = F.interpolate(score.unsqueeze(0), size=original_hw, mode="bilinear", align_corners=False).squeeze(0)
                hard[method] = F.interpolate((score > .58).float().unsqueeze(0), size=original_hw, mode="nearest").squeeze(0)
                fg_hist, bg_hist = _score_histogram(continuous[method].squeeze().numpy(), gt_np, HIST_BINS)
                pooled_fg[method] += fg_hist
                pooled_bg[method] += bg_hist

            cached = None
            if cache_path.is_file():
                candidate = torch_load(cache_path, map_location="cpu")
                if candidate.get("dataset") == dataset and candidate.get("stem") == stem and set(candidate.get("methods", [])) == set(METHODS):
                    cached = candidate
            if cached is None:
                context = FastCODContext(gt)
                hard_outputs = context.evaluate_many([(method, "hard", hard[method]) for method in METHODS], .58)
                image_metrics, image_dist, image_pca, image_curves = [], [], [], []
                gt_area = float(gt.float().mean())
                for method in METHODS:
                    hard_metric = hard_outputs[(method, "hard")]
                    rank = _rank_metrics(continuous[method], gt)
                    dice = 2 * hard_metric["IoU"] / (1 + hard_metric["IoU"]) if hard_metric["IoU"] >= 0 else float("nan")
                    image_metrics.append({
                        "dataset": dataset, "stem": stem, "method": method, "gt_area": gt_area,
                        **{field: float(hard_metric[field]) for field in ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area", "IoU")},
                        "Dice": float(dice), "boundary_F1": _boundary_f1(hard[method], gt), **rank,
                    })
                    pred_np = continuous[method].squeeze().numpy()
                    fg_stats, bg_stats = _quantiles(pred_np[gt_np]), _quantiles(pred_np[~gt_np])
                    image_dist.append({
                        "version": method, "method": method, "dataset": dataset, "image": stem,
                        **{f"FG_{key}": value for key, value in fg_stats.items()},
                        **{f"BG_{key}": value for key, value in bg_stats.items()},
                        "mean_gap": fg_stats["mean"] - bg_stats["mean"],
                        "median_gap": fg_stats["median"] - bg_stats["median"],
                    })
                    payload = payloads[method]
                    image_pca.append({
                        "version": method, "method": method, "dataset": dataset, "image": stem,
                        "selected_rank": int(payload["selected_rank"]),
                        "required_rank_uncapped": int(payload["required_rank_uncapped"]),
                        "rank_cap": int(payload["pca_rank_cap"]),
                        "energy_at_cap": float(payload["energy_at_cap"]),
                        "hit_rank_cap": int(payload["hit_rank_cap"]),
                        "candidate_count": int(payload["candidate_count"]),
                    })
                    image_curves.append({"dataset": dataset, "stem": stem, "method": method, "gt_area": gt_area, **_curves(pred_np, gt_np, context)})
                cached = {
                    "dataset": dataset, "stem": stem, "methods": list(METHODS),
                    "metric_rows": image_metrics, "dist_rows": image_dist,
                    "pca_rows": image_pca, "curve_records": image_curves,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cached, cache_path)
            metric_rows.extend(cached["metric_rows"])
            dist_rows.extend(cached["dist_rows"])
            pca_rows.extend(cached["pca_rows"])
            curve_records.extend(cached["curve_records"])
        except Exception as exc:
            failures.append({"index": index, "dataset": dataset, "stem": stem, "exception_type": type(exc).__name__, "exception": repr(exc)})
            print(f"FAILED {dataset}/{stem}: {exc!r}", flush=True)
        if index % max(1, int(args.checkpoint_every)) == 0 or index == len(samples):
            valid = len({(row["dataset"], row["stem"]) for row in metric_rows})
            print(f"[{index}/{len(samples)}] evaluation valid={valid} failed={len(failures)}", flush=True)
            write_jsonl(output / "evaluation_failures.jsonl", failures)

    valid_keys = {(row["dataset"], row["stem"]) for row in metric_rows}
    if failures or len(valid_keys) != len(samples):
        write_jsonl(output / "evaluation_failures.jsonl", failures)
        raise RuntimeError(f"evaluation incomplete: valid={len(valid_keys)} requested={len(samples)} failed={len(failures)}")

    fixed_summary = _aggregate_metrics(metric_rows, FIXED_FIELDS)
    continuous_summary = _aggregate_metrics(metric_rows, CONTINUOUS_FIELDS)
    curve_rows, curve_summary = _aggregate_curve_records(curve_records, scope="all")
    for row in continuous_summary:
        if row["scope"] in {"dataset", "dataset_macro"}:
            key = (row["method"], row["dataset"])
            if key in curve_summary:
                row.update({key2: value for key2, value in curve_summary[key].items() if key2.startswith("max") or key2.startswith("best_")})
    for method in METHODS:
        ap, auroc = _hist_ap_auroc(pooled_fg[method], pooled_bg[method])
        continuous_summary.append({
            "scope": "global_pooled_hist65536_per_image_minmax", "dataset": "ALL", "method": method,
            "num_images": len(samples), "pixel_AP": ap, "pixel_AUROC": auroc,
        })

    dist_summary = _aggregate_metrics(dist_rows, ("FG_mean", "FG_median", "BG_mean", "BG_median", "mean_gap", "median_gap"))
    pca_summary = []
    for method in METHODS:
        rows = [row for row in pca_rows if row["method"] == method]
        pca_summary.append({
            "method": method, "num_images": len(rows), "rank_cap": int(rows[0]["rank_cap"]),
            "mean_selected_rank": float(np.mean([row["selected_rank"] for row in rows])),
            "mean_required_rank": float(np.mean([row["required_rank_uncapped"] for row in rows])),
            "hit_cap_ratio": float(np.mean([row["hit_rank_cap"] for row in rows])),
            "mean_energy_at_cap": float(np.mean([row["energy_at_cap"] for row in rows])),
            "mean_candidate_count": float(np.mean([row["candidate_count"] for row in rows])),
        })

    gt_areas = {}
    for row in metric_rows:
        gt_areas[(row["dataset"], row["stem"])] = float(row["gt_area"])
    small_cutoff = float(np.quantile(list(gt_areas.values()), .25))
    small_keys = {key for key, area in gt_areas.items() if area <= small_cutoff}
    small_metric_rows = [row for row in metric_rows if (row["dataset"], row["stem"]) in small_keys]
    small_curve_records = [row for row in curve_records if (row["dataset"], row["stem"]) in small_keys]
    small_summary = _aggregate_metrics(small_metric_rows, (*CONTINUOUS_FIELDS, "boundary_F1", "F_beta_w"))
    _, small_curves = _aggregate_curve_records(small_curve_records, scope="small_q25")
    for row in small_summary:
        if row["scope"] in {"dataset", "dataset_macro"}:
            key = (row["method"], row["dataset"])
            if key in small_curves:
                row["maxF"] = small_curves[key]["maxF"]
                row["best_F_threshold"] = small_curves[key]["best_F_threshold"]

    old_296 = _old_296_per_image(Path(args.per_image_296).expanduser().resolve())
    old_small_rows = []
    for key in small_keys:
        if key not in old_296:
            continue
        row = old_296[key]
        old_small_rows.append({
            "dataset": key[0], "stem": key[1], "method": "296-baseline",
            "pixel_AP": float(row["pixel_AP"]), "pixel_AUROC": float(row["pixel_AUROC"]),
            "boundary_F1": float("nan"), "F_beta_w": float(row["F_beta_w"]),
        })
    if old_small_rows:
        # _aggregate_metrics is fixed to METHODS; aggregate the 296 subset explicitly.
        by_dataset = []
        for dataset in DATASETS:
            selected = [row for row in old_small_rows if row["dataset"] == dataset]
            ds = {"scope": "dataset", "dataset": dataset, "method": "296-baseline", "num_images": len(selected)}
            for field in (*CONTINUOUS_FIELDS, "boundary_F1", "F_beta_w"):
                ds[field] = _safe_nanmean(row[field] for row in selected)
            by_dataset.append(ds)
        macro = {"scope": "dataset_macro", "dataset": "ALL", "method": "296-baseline", "num_images": sum(row["num_images"] for row in by_dataset)}
        for field in (*CONTINUOUS_FIELDS, "boundary_F1", "F_beta_w"):
            macro[field] = _safe_nanmean(row[field] for row in by_dataset)
        macro["maxF"] = float("nan")
        small_summary.extend([*by_dataset, macro])

    continuous_296_path = Path(args.continuous_296).expanduser().resolve()
    per_dataset_continuous_296_path = Path(args.per_dataset_continuous_296).expanduser().resolve()
    fixed_296_path = Path(args.fixed_296).expanduser().resolve()
    continuous_summary = [
        *_read_296_continuous(continuous_296_path, per_dataset_continuous_296_path),
        *continuous_summary,
    ]
    fixed_summary = [*_read_296_fixed(fixed_296_path), *fixed_summary]

    _write_csv(output / "continuous_metrics.csv", continuous_summary)
    _write_csv(output / "fixed_threshold_metrics.csv", fixed_summary)
    _write_csv(output / "per_image_metrics.csv", metric_rows)
    _write_csv(output / "per_dataset_metrics.csv", [row for row in [*continuous_summary, *fixed_summary] if row.get("scope") in {"dataset", "dataset_macro"}])
    _write_csv(output / "score_distribution_stats.csv", dist_rows)
    _write_csv(output / "score_distribution_summary.csv", dist_summary)
    _write_csv(output / "pca_rank_full6473.csv", pca_rows)
    _write_csv(output / "pca_rank_summary.csv", pca_summary)
    _write_csv(output / "threshold_sweep.csv", curve_rows)
    _write_csv(output / "small_object_metrics.csv", small_summary)
    for method, filename in (("512-A", "pr_curve_512A.csv"), ("512-B", "pr_curve_512B.csv"), ("512-C", "pr_curve_512C.csv")):
        _write_csv(output / filename, [row for row in curve_rows if row["method"] == method])
    _plot_pr(curve_rows, output / "PR_512ABC.png")

    visual_rows = _visualize(
        samples, metric_rows, dist_rows, score_root,
        Path(args.core_296_root).expanduser().resolve(), old_296, output / "visualizations",
    )

    feature_rows = read_jsonl(Path(args.feature_manifest).expanduser().resolve())
    generation_metadata = json.loads((score_root / "generation_metadata.json").read_text(encoding="utf-8"))
    generation_rows = list(csv.DictReader((score_root / "generation_runtime.csv").open(encoding="utf-8")))
    runtime_rows = [{
        "stage": "shared_dino512_cache", "num_images": len(feature_rows),
        "seconds": float(np.nansum([float(row.get("total_seconds", "nan")) for row in feature_rows])),
    }, {
        "stage": "A_C_B_generation_wall", "num_images": len(samples), "seconds": float(generation_metadata["wall_seconds"]),
    }]
    for method in METHODS:
        selected = [row for row in generation_rows if row["method"] == method]
        runtime_rows.append({
            "stage": f"{method}_bc_pca", "num_images": len(selected),
            "seconds": float(np.nansum([float(row.get("bc_seconds", 0)) + float(row.get("pca_seconds", 0)) for row in selected])),
        })
    eval_seconds = time.perf_counter() - started
    runtime_rows.append({"stage": "evaluation_wall", "num_images": len(samples), "seconds": eval_seconds})
    for row in runtime_rows:
        row["sec_per_image"] = row["seconds"] / max(1, row["num_images"])
        row["hours"] = row["seconds"] / 3600
    _write_csv(output / "runtime_full6473.csv", runtime_rows)

    counts = {dataset: sum(key[0] == dataset for key in valid_keys) for dataset in DATASETS}
    is_full = len(samples) == 6473 and counts == EXPECTED_COUNTS
    metadata = {
        "version": "gbsp_resolution_full_eval_v1", "num_requested": len(samples),
        "num_valid": len(valid_keys), "num_failed": len(failures), "counts": counts,
        "is_full_complete": is_full and not failures, "sample_list_sha256": sample_sha,
        "small_object_rule": "global GT area <= full-set q25, identical to probe200 rule",
        "small_object_cutoff": small_cutoff, "small_object_count": len(small_keys),
        "gt_used_for_generation": False, "gt_used_for_evaluation": True,
        "training_triggered": False, "reran_296": False,
        "threshold_curve_bins": CURVE_SIZE, "fixed_threshold": .58,
        "git_commit": _git_commit(main_root), "wall_seconds": eval_seconds,
        "visualization_count": len(visual_rows),
    }
    write_json(output / "evaluation_metadata.json", metadata)
    sources = {
        "continuous_296": str(continuous_296_path),
        "continuous_296_per_dataset": str(per_dataset_continuous_296_path),
        "fixed_296": str(fixed_296_path),
    }
    _write_report(
        output, continuous_summary, fixed_summary, dist_summary, pca_summary,
        small_summary, curve_rows, sources, len(valid_keys), len(visual_rows),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if len(samples) == 6473 and not metadata["is_full_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
