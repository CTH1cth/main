import argparse
import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import ensure_dir, read_jsonl  # noqa: E402

try:
    from scipy.ndimage import binary_dilation as scipy_binary_dilation
    from scipy.ndimage import binary_erosion as scipy_binary_erosion
    from scipy.ndimage import label as scipy_label

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


FAILURE_TYPES = [
    "hard_semantic_miss",
    "mislocalization",
    "under_segmentation",
    "over_segmentation",
    "fragmentation",
    "boundary_error",
    "calibration_error",
    "mixed_or_mild",
]

PER_IMAGE_FIELDS = [
    "dataset",
    "stem",
    "run_tag",
    "IoU",
    "Dice",
    "Precision",
    "Recall",
    "Specificity",
    "Accuracy",
    "MAE_binary",
    "MAE_soft",
    "pred_area",
    "gt_area",
    "area_ratio",
    "area_diff",
    "FP",
    "FN",
    "TP",
    "TN",
    "FP_rate",
    "FN_rate",
    "centroid_distance_norm",
    "bbox_iou",
    "num_pred_components",
    "largest_component_area_ratio",
    "largest_component_gt_overlap",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "soft_fg_mean",
    "soft_bg_mean",
    "soft_gap",
    "best_threshold",
    "best_threshold_iou",
    "threshold_gain_iou",
    "best_threshold_dice",
    "threshold_gain_dice",
    "failure_type",
    "image_path",
    "gt_path",
    "prob_path",
    "bin_path",
]


def parse_csv_arg(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def finite_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def mean(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def pearson_corr(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None
        and y is not None
        and not math.isnan(float(x))
        and not math.isnan(float(y))
        and not math.isinf(float(x))
        and not math.isinf(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def binary_dilation(mask, radius=1):
    radius = int(radius)
    if radius <= 0:
        return mask.astype(bool)
    if HAS_SCIPY:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return scipy_binary_dilation(mask.astype(bool), structure=structure)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def binary_erosion(mask, radius=1):
    radius = int(radius)
    if radius <= 0:
        return mask.astype(bool)
    if HAS_SCIPY:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return scipy_binary_erosion(mask.astype(bool), structure=structure, border_value=0)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def mask_boundary(mask):
    mask = mask.astype(bool)
    return np.logical_xor(binary_dilation(mask, 1), binary_erosion(mask, 1))


def connected_components(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0
    if HAS_SCIPY:
        labels, num = scipy_label(mask, structure=np.ones((3, 3), dtype=np.int8))
        return labels.astype(np.int32), int(num)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def bbox(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_iou(pred, gt):
    box_a = bbox(pred)
    box_b = bbox(gt)
    if box_a is None and box_b is None:
        return 1.0
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def centroid_distance(pred, gt):
    pred_pts = np.argwhere(pred.astype(bool))
    gt_pts = np.argwhere(gt.astype(bool))
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 1.0
    pc = pred_pts.mean(axis=0)
    gc = gt_pts.mean(axis=0)
    h, w = gt.shape
    diag = math.sqrt(float(h * h + w * w))
    return float(np.linalg.norm(pc - gc) / max(diag, 1.0))


def boundary_metrics(pred, gt):
    pred_b = mask_boundary(pred)
    gt_b = mask_boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0, 1.0, 1.0
    if not pred_b.any():
        return 0.0, 0.0, 0.0
    if not gt_b.any():
        return 0.0, 0.0, 0.0
    gt_tol = binary_dilation(gt_b, 2)
    pred_tol = binary_dilation(pred_b, 2)
    precision = float((pred_b & gt_tol).sum() / max(pred_b.sum(), 1))
    recall = float((gt_b & pred_tol).sum() / max(gt_b.sum(), 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, float(f1)


def threshold_sweep(prob, gt, current_iou, current_dice):
    if prob is None:
        return {
            "best_threshold": float("nan"),
            "best_threshold_iou": float("nan"),
            "threshold_gain_iou": float("nan"),
            "best_threshold_dice": float("nan"),
            "threshold_gain_dice": float("nan"),
        }
    best_iou = -1.0
    best_iou_thresh = float("nan")
    best_dice = -1.0
    best_dice_thresh = float("nan")
    for thresh in np.arange(0.05, 1.0, 0.05):
        pred = prob > float(thresh)
        tp = float((pred & gt).sum())
        fp = float((pred & ~gt).sum())
        fn = float((~pred & gt).sum())
        iou = tp / max(tp + fp + fn, 1.0)
        dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        if iou > best_iou:
            best_iou = iou
            best_iou_thresh = float(thresh)
        if dice > best_dice:
            best_dice = dice
            best_dice_thresh = float(thresh)
    return {
        "best_threshold": best_iou_thresh,
        "best_threshold_iou": float(best_iou),
        "threshold_gain_iou": float(best_iou - current_iou),
        "best_threshold_dice": float(best_dice),
        "threshold_gain_dice": float(best_dice - current_dice),
    }


def classify_failure(metrics):
    recall = metrics["Recall"]
    precision = metrics["Precision"]
    area_ratio = metrics["area_ratio"]
    iou = metrics["IoU"]
    soft_fg_mean = metrics["soft_fg_mean"]
    threshold_gain_iou = metrics["threshold_gain_iou"]
    if recall < 0.40 and (math.isnan(soft_fg_mean) or soft_fg_mean < 0.35):
        return "hard_semantic_miss"
    if metrics["bbox_iou"] < 0.30 and 0.60 <= area_ratio <= 1.60 and iou < 0.40:
        return "mislocalization"
    if recall < 0.75 and precision >= 0.75 and area_ratio < 0.85:
        return "under_segmentation"
    if precision < 0.70 and area_ratio > 1.20:
        return "over_segmentation"
    if recall >= 0.60 and metrics["num_pred_components"] >= 4 and metrics["largest_component_gt_overlap"] < 0.70:
        return "fragmentation"
    if iou >= 0.50 and metrics["boundary_f1"] < 0.50:
        return "boundary_error"
    if not math.isnan(threshold_gain_iou) and threshold_gain_iou > 0.08:
        return "calibration_error"
    return "mixed_or_mild"


def compute_metrics(gt, pred, prob):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    total = float(gt.size)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    tn = float((~pred & ~gt).sum())
    pred_pixels = float(pred.sum())
    gt_pixels = float(gt.sum())
    bg_pixels = total - gt_pixels
    iou = tp / max(tp + fp + fn, 1.0)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    accuracy = (tp + tn) / max(total, 1.0)
    gt_float = gt.astype(np.float32)
    pred_float = pred.astype(np.float32)
    mae_binary = float(np.abs(pred_float - gt_float).mean())
    mae_soft = float(np.abs(prob.astype(np.float32) - gt_float).mean()) if prob is not None else float("nan")
    labels, num_components = connected_components(pred)
    largest_area = 0.0
    largest_overlap = 0.0
    if int(num_components) > 0:
        for idx in range(1, int(num_components) + 1):
            comp = labels == idx
            area = float(comp.sum())
            if area > largest_area:
                largest_area = area
                largest_overlap = float((comp & gt).sum())
    boundary_precision, boundary_recall, boundary_f1 = boundary_metrics(pred, gt)
    if prob is not None and gt.any():
        soft_fg_mean = float(prob[gt].mean())
    else:
        soft_fg_mean = float("nan")
    if prob is not None and (~gt).any():
        soft_bg_mean = float(prob[~gt].mean())
    else:
        soft_bg_mean = float("nan")
    soft_gap = (
        float(soft_fg_mean - soft_bg_mean)
        if not math.isnan(soft_fg_mean) and not math.isnan(soft_bg_mean)
        else float("nan")
    )
    thresh_stats = threshold_sweep(prob, gt, iou, dice)
    metrics = {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Accuracy": float(accuracy),
        "MAE_binary": mae_binary,
        "MAE_soft": mae_soft,
        "pred_area": float(pred_pixels / max(total, 1.0)),
        "gt_area": float(gt_pixels / max(total, 1.0)),
        "area_ratio": float(pred_pixels / max(gt_pixels, 1.0)),
        "area_diff": float((pred_pixels - gt_pixels) / max(total, 1.0)),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN": int(tn),
        "FP_rate": float(fp / max(bg_pixels, 1.0)),
        "FN_rate": float(fn / max(gt_pixels, 1.0)),
        "centroid_distance_norm": centroid_distance(pred, gt),
        "bbox_iou": bbox_iou(pred, gt),
        "num_pred_components": int(num_components),
        "largest_component_area_ratio": float(largest_area / max(pred_pixels, 1.0)),
        "largest_component_gt_overlap": float(largest_overlap / max(gt_pixels, 1.0)),
        "boundary_precision": float(boundary_precision),
        "boundary_recall": float(boundary_recall),
        "boundary_f1": float(boundary_f1),
        "soft_fg_mean": soft_fg_mean,
        "soft_bg_mean": soft_bg_mean,
        "soft_gap": soft_gap,
        **thresh_stats,
    }
    metrics["failure_type"] = classify_failure(metrics)
    return metrics


def load_manifest(root):
    root = Path(root).expanduser()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.jsonl not found: {manifest_path}")
    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    mapping = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"Duplicate manifest key {key} in {manifest_path}")
        mapping[key] = row
    return root, rows, mapping


def load_gt(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"GT missing: {path}")
    return np.asarray(Image.open(path).convert("L")) > 128


def load_pred(row):
    bin_path = str(row.get("bin_path", ""))
    prob_path = str(row.get("prob_path", ""))
    prob = None
    if prob_path:
        if not Path(prob_path).exists():
            raise FileNotFoundError(f"prob.npy missing: {prob_path}")
        prob = np.clip(np.load(prob_path).astype(np.float32), 0.0, 1.0)
    if bin_path:
        if not Path(bin_path).exists():
            raise FileNotFoundError(f"bin.png missing: {bin_path}")
        pred = np.asarray(Image.open(bin_path).convert("L")) > 128
    elif prob is not None:
        pred = prob > 0.5
    else:
        raise RuntimeError(f"No bin_path or prob_path for {row.get('dataset')}/{row.get('stem')}")
    gt = load_gt(row["gt_path"])
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch for {row['dataset']}/{row['stem']}: {pred.shape} != {gt.shape}")
    if prob is not None and prob.shape != gt.shape:
        raise RuntimeError(f"Prob/GT shape mismatch for {row['dataset']}/{row['stem']}: {prob.shape} != {gt.shape}")
    return gt, pred, prob


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataset_summary_rows(per_rows, tags, datasets):
    rows = []
    metrics = [
        "IoU",
        "Dice",
        "Precision",
        "Recall",
        "MAE_binary",
        "MAE_soft",
        "pred_area",
        "gt_area",
        "area_ratio",
        "boundary_f1",
        "threshold_gain_iou",
    ]
    for dataset in datasets:
        for tag in tags:
            group = [r for r in per_rows if r["dataset"] == dataset and r["run_tag"] == tag]
            if not group:
                continue
            row = {"dataset": dataset, "run_tag": tag, "num_images": len(group)}
            for metric in metrics:
                row[f"mean_{metric}"] = mean([r[metric] for r in group])
            rows.append(row)
    return rows


def failure_count_rows(per_rows, tags, datasets):
    rows = []
    for dataset in datasets:
        for tag in tags:
            group = [r for r in per_rows if r["dataset"] == dataset and r["run_tag"] == tag]
            if not group:
                continue
            total = len(group)
            for failure_type in FAILURE_TYPES:
                subset = [r for r in group if r["failure_type"] == failure_type]
                rows.append(
                    {
                        "dataset": dataset,
                        "run_tag": tag,
                        "failure_type": failure_type,
                        "count": len(subset),
                        "ratio": float(len(subset) / max(total, 1)),
                        "mean_iou": mean([r["IoU"] for r in subset]),
                        "mean_recall": mean([r["Recall"] for r in subset]),
                        "mean_precision": mean([r["Precision"] for r in subset]),
                        "mean_area_ratio": mean([r["area_ratio"] for r in subset]),
                        "mean_boundary_f1": mean([r["boundary_f1"] for r in subset]),
                    }
                )
    return rows


def build_metric_map(per_rows):
    return {(r["dataset"], r["stem"], r["run_tag"]): r for r in per_rows}


def pairwise_rows(per_rows, tags, datasets):
    base_tag = "orig35" if "orig35" in tags else tags[0]
    metric_map = build_metric_map(per_rows)
    rows = []
    for dataset in datasets:
        stems = sorted({r["stem"] for r in per_rows if r["dataset"] == dataset})
        for stem in stems:
            base = metric_map.get((dataset, stem, base_tag))
            if base is None:
                raise RuntimeError(f"Base tag {base_tag} missing for {dataset}/{stem}")
            for tag in tags:
                if tag == base_tag:
                    continue
                comp = metric_map.get((dataset, stem, tag))
                if comp is None:
                    raise RuntimeError(f"Compare tag {tag} missing for {dataset}/{stem}")
                rows.append(
                    {
                        "dataset": dataset,
                        "image_id": stem,
                        "base_tag": base_tag,
                        "compare_tag": tag,
                        "delta_iou": comp["IoU"] - base["IoU"],
                        "delta_dice": comp["Dice"] - base["Dice"],
                        "delta_recall": comp["Recall"] - base["Recall"],
                        "delta_precision": comp["Precision"] - base["Precision"],
                        "delta_area_ratio": comp["area_ratio"] - base["area_ratio"],
                        "delta_fn_rate": comp["FN_rate"] - base["FN_rate"],
                        "delta_fp_rate": comp["FP_rate"] - base["FP_rate"],
                        "delta_boundary_f1": comp["boundary_f1"] - base["boundary_f1"],
                        "delta_centroid_distance": comp["centroid_distance_norm"] - base["centroid_distance_norm"],
                        "base_failure_type": base["failure_type"],
                        "compare_failure_type": comp["failure_type"],
                        "failure_changed": base["failure_type"] != comp["failure_type"],
                    }
                )
    return rows


def parse_csd_epoch(tag):
    match = re.match(r"csd(\d+)", str(tag))
    return int(match.group(1)) if match else None


def epoch_trend_rows(per_rows, tags, datasets):
    csd_tags = [(tag, parse_csd_epoch(tag)) for tag in tags if parse_csd_epoch(tag) is not None]
    csd_tags = sorted(csd_tags, key=lambda x: x[1])
    if not csd_tags:
        return []
    metric_map = build_metric_map(per_rows)
    rows = []
    for dataset in datasets:
        stems = sorted({r["stem"] for r in per_rows if r["dataset"] == dataset})
        for stem in stems:
            samples = []
            for tag, epoch in csd_tags:
                row = metric_map.get((dataset, stem, tag))
                if row is not None:
                    samples.append((tag, epoch, row))
            if not samples:
                continue
            best_iou_item = max(samples, key=lambda x: x[2]["IoU"])
            best_dice_item = max(samples, key=lambda x: x[2]["Dice"])
            max_iou = best_iou_item[2]["IoU"]
            epoch35_row = next((r for _, e, r in samples if e == 35), samples[-1][2])
            epoch20_row = next((r for _, e, r in samples if e == 20), samples[0][2])
            is_early_good_late_bad = best_iou_item[1] <= 27 and epoch35_row["IoU"] <= max_iou - 0.05
            is_always_bad = max_iou < 0.30
            is_late_good = epoch35_row["IoU"] >= max_iou - 0.02 and epoch35_row["IoU"] >= epoch20_row["IoU"] + 0.05
            for tag, epoch, row in samples:
                rows.append(
                    {
                        "image_id": stem,
                        "dataset": dataset,
                        "tag": tag,
                        "epoch": epoch,
                        "iou": row["IoU"],
                        "dice": row["Dice"],
                        "precision": row["Precision"],
                        "recall": row["Recall"],
                        "area_ratio": row["area_ratio"],
                        "pred_area": row["pred_area"],
                        "centroid_distance": row["centroid_distance_norm"],
                        "failure_type": row["failure_type"],
                        "best_epoch_by_iou": best_iou_item[1],
                        "best_epoch_by_dice": best_dice_item[1],
                        "is_early_good_late_bad": bool(is_early_good_late_bad),
                        "is_late_good": bool(is_late_good),
                        "is_always_bad": bool(is_always_bad),
                    }
                )
    return rows


def oracle_rows(per_rows, tags, datasets):
    metric_map = build_metric_map(per_rows)
    baseline_csd = "csd35" if "csd35" in tags else tags[-1]
    baseline_orig = "orig35" if "orig35" in tags else tags[0]
    rows = []
    summary = {}
    for dataset in datasets:
        stems = sorted({r["stem"] for r in per_rows if r["dataset"] == dataset})
        best_counts = Counter()
        oracle_ious = []
        single_means = {}
        for tag in tags:
            single_means[tag] = mean(
                [metric_map[(dataset, stem, tag)]["IoU"] for stem in stems if (dataset, stem, tag) in metric_map]
            )
        for stem in stems:
            candidates = [metric_map[(dataset, stem, tag)] for tag in tags if (dataset, stem, tag) in metric_map]
            if len(candidates) != len(tags):
                raise RuntimeError(f"Oracle candidate missing for {dataset}/{stem}")
            best = max(candidates, key=lambda r: r["IoU"])
            best_counts[best["run_tag"]] += 1
            oracle_ious.append(best["IoU"])
            csd = metric_map[(dataset, stem, baseline_csd)]
            orig = metric_map[(dataset, stem, baseline_orig)]
            rows.append(
                {
                    "dataset": dataset,
                    "image_id": stem,
                    "oracle_best_tag": best["run_tag"],
                    "oracle_best_iou": best["IoU"],
                    "oracle_gain_over_csd35": best["IoU"] - csd["IoU"],
                    "oracle_gain_over_orig35": best["IoU"] - orig["IoU"],
                }
            )
        best_single_tag = max(single_means, key=lambda t: -1e9 if math.isnan(single_means[t]) else single_means[t])
        oracle_mean = mean(oracle_ious)
        summary[dataset] = {
            "single_best_run": best_single_tag,
            "single_best_run_mean_iou": single_means[best_single_tag],
            "oracle_mean_iou": oracle_mean,
            "oracle_gain": oracle_mean - single_means[best_single_tag],
            "oracle_best_tag_distribution": dict(best_counts),
        }
    return rows, summary


def threshold_summary_rows(per_rows, tags, datasets):
    rows = []
    for dataset in datasets:
        for tag in tags:
            group = [r for r in per_rows if r["dataset"] == dataset and r["run_tag"] == tag]
            if not group:
                continue
            gains = [r["threshold_gain_iou"] for r in group]
            rows.append(
                {
                    "dataset": dataset,
                    "run_tag": tag,
                    "mean_best_threshold": mean([r["best_threshold"] for r in group]),
                    "mean_best_threshold_iou": mean([r["best_threshold_iou"] for r in group]),
                    "mean_threshold_gain_iou": mean(gains),
                    "mean_best_threshold_dice": mean([r["best_threshold_dice"] for r in group]),
                    "mean_threshold_gain_dice": mean([r["threshold_gain_dice"] for r in group]),
                    "calibration_error_ratio": float(
                        sum(1 for r in group if not math.isnan(r["threshold_gain_iou"]) and r["threshold_gain_iou"] > 0.08)
                        / max(len(group), 1)
                    ),
                }
            )
    return rows


def select_primary_tag(tags):
    if "csd35" in tags:
        return "csd35"
    for tag in reversed(tags):
        if str(tag).startswith("csd"):
            return tag
    return tags[-1]


def area_correlation(per_rows, dataset, tag):
    group = [r for r in per_rows if r["dataset"] == dataset and r["run_tag"] == tag]
    corr = {
        "tag": tag,
        "corr_area_ratio_iou": pearson_corr([r["area_ratio"] for r in group], [r["IoU"] for r in group]),
        "corr_area_ratio_recall": pearson_corr([r["area_ratio"] for r in group], [r["Recall"] for r in group]),
        "corr_area_ratio_precision": pearson_corr([r["area_ratio"] for r in group], [r["Precision"] for r in group]),
        "corr_pred_area_iou": pearson_corr([r["pred_area"] for r in group], [r["IoU"] for r in group]),
    }
    failure_counter = Counter(r["failure_type"] for r in group)
    under_ratio = failure_counter.get("under_segmentation", 0) / max(len(group), 1)
    hard_mis_ratio = (
        failure_counter.get("mislocalization", 0) + failure_counter.get("hard_semantic_miss", 0)
    ) / max(len(group), 1)
    if not math.isnan(corr["corr_area_ratio_iou"]) and corr["corr_area_ratio_iou"] > 0.55 and under_ratio > 0.25:
        verdict = "likely"
    elif hard_mis_ratio > 0.35:
        verdict = "unlikely"
    else:
        verdict = "mixed"
    corr["area_explains_failure"] = verdict
    return corr


def overlay_contours(rgb, gt, pred):
    rgb = rgb.convert("RGB").resize((220, 160))
    gt_r = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((220, 160), Image.NEAREST)) > 128
    pred_r = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize((220, 160), Image.NEAREST)) > 128
    arr = np.asarray(rgb).copy()
    gt_b = mask_boundary(gt_r)
    pred_b = mask_boundary(pred_r)
    arr[gt_b] = np.array([255, 230, 0], dtype=np.uint8)
    arr[pred_b] = np.array([0, 220, 255], dtype=np.uint8)
    return Image.fromarray(arr)


def make_panel(image, title, text="", size=(220, 200)):
    image = image.convert("RGB")
    canvas = Image.new("RGB", size, "white")
    img_h = size[1] - 40
    image.thumbnail((size[0], img_h), Image.BILINEAR)
    x = (size[0] - image.width) // 2
    y = 18
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), str(title)[:32], fill=(0, 0, 0))
    if text:
        draw.text((4, size[1] - 20), str(text)[:42], fill=(0, 0, 0))
    return canvas


def mask_to_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")


def heatmap(prob):
    if prob is None:
        return Image.new("RGB", (220, 160), "gray")
    p = np.clip(prob, 0.0, 1.0)
    arr = np.zeros((*p.shape, 3), dtype=np.uint8)
    arr[..., 0] = (p * 255).astype(np.uint8)
    arr[..., 1] = (np.sqrt(p) * 180).astype(np.uint8)
    arr[..., 2] = ((1.0 - p) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def fp_fn_overlay(rgb, gt, pred):
    rgb = rgb.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    gt = Image.fromarray(gt.astype(np.uint8) * 255).resize(rgb.size, Image.NEAREST)
    pred = Image.fromarray(pred.astype(np.uint8) * 255).resize(rgb.size, Image.NEAREST)
    gt = np.asarray(gt) > 128
    pred = np.asarray(pred) > 128
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    arr[tp] = 0.45 * arr[tp] + 0.55 * np.array([0, 255, 0])
    arr[fp] = 0.35 * arr[fp] + 0.65 * np.array([255, 0, 0])
    arr[fn] = 0.35 * arr[fn] + 0.65 * np.array([0, 80, 255])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_lookup(manifests):
    return {
        (tag, str(row["dataset"]), str(row["stem"])): row
        for tag, rows in manifests.items()
        for row in rows
    }


def load_case_arrays(row):
    gt = load_gt(row["gt_path"])
    pred = np.asarray(Image.open(row["bin_path"]).convert("L")) > 128
    prob_path = str(row.get("prob_path", ""))
    prob = np.clip(np.load(prob_path).astype(np.float32), 0.0, 1.0) if prob_path and Path(prob_path).exists() else None
    rgb = Image.open(row["image_path"]).convert("RGB")
    return rgb, gt, pred, prob


def make_case_grid(out_path, case, tags, lookup, metric_map):
    dataset = case["dataset"]
    stem = case["stem"]
    preferred = [tag for tag in ["orig35", "csd20", "csd27", "csd35", "lceg44"] if tag in tags]
    if not preferred:
        preferred = tags[: min(len(tags), 4)]
    target_tag = "csd35" if "csd35" in tags else preferred[-1]
    first_row = lookup[(preferred[0], dataset, stem)]
    rgb, gt, _, _ = load_case_arrays(first_row)
    panels = [make_panel(rgb.copy(), "Image"), make_panel(mask_to_image(gt), "GT")]
    for tag in preferred:
        row = lookup[(tag, dataset, stem)]
        _, _, pred, _ = load_case_arrays(row)
        m = metric_map[(dataset, stem, tag)]
        panels.append(make_panel(mask_to_image(pred), tag, f"IoU {m['IoU']:.2f} R {m['Recall']:.2f}"))
    target_row = lookup[(target_tag, dataset, stem)]
    _, _, target_pred, target_prob = load_case_arrays(target_row)
    target_metric = metric_map[(dataset, stem, target_tag)]
    panels.append(make_panel(fp_fn_overlay(rgb.copy(), gt, target_pred), f"{target_tag} FP/FN", target_metric["failure_type"]))
    panels.append(make_panel(heatmap(target_prob), f"{target_tag} prob", f"gain {target_metric['threshold_gain_iou']:.2f}"))
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    ensure_dir(Path(out_path).parent)
    canvas.save(out_path, quality=85)


def make_contact_sheet(out_path, rows, lookup, metric_map, tag, max_cases):
    rows = rows[: int(max_cases)]
    if not rows:
        return
    tile_w, tile_h = 260, 210
    cols = 4
    rows_n = int(math.ceil(len(rows) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows_n * tile_h), "white")
    for idx, metric in enumerate(rows):
        dataset, stem = metric["dataset"], metric["stem"]
        pred_row = lookup[(tag, dataset, stem)]
        rgb, gt, pred, _ = load_case_arrays(pred_row)
        image = overlay_contours(rgb, gt, pred)
        text = (
            f"{stem}\n{metric['failure_type']}\n"
            f"IoU {metric['IoU']:.2f} R {metric['Recall']:.2f} "
            f"P {metric['Precision']:.2f} A {metric['area_ratio']:.2f}"
        )
        tile = make_panel(image, tag, text, size=(tile_w, tile_h))
        canvas.paste(tile, ((idx % cols) * tile_w, (idx // cols) * tile_h))
    ensure_dir(Path(out_path).parent)
    canvas.save(out_path, quality=85)


def simple_plot(path, title, points=None, bars=None, line=None):
    width, height = 800, 520
    margin = 60
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), title, fill=(0, 0, 0))
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(0, 0, 0))
    if points:
        xs = [p[0] for p in points if not math.isnan(p[0]) and not math.isnan(p[1])]
        ys = [p[1] for p in points if not math.isnan(p[0]) and not math.isnan(p[1])]
        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax == xmin:
                xmax += 1.0
            if ymax == ymin:
                ymax += 1.0
            for x, y in zip(xs, ys):
                px = margin + int((x - xmin) / (xmax - xmin) * (width - 2 * margin))
                py = height - margin - int((y - ymin) / (ymax - ymin) * (height - 2 * margin))
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(30, 90, 200))
    if bars:
        max_v = max([v for _, v in bars] or [1.0])
        bar_w = max(8, int((width - 2 * margin) / max(len(bars), 1)))
        for idx, (name, value) in enumerate(bars):
            x1 = margin + idx * bar_w
            x2 = x1 + int(bar_w * 0.8)
            y2 = height - margin
            y1 = y2 - int(float(value) / max(max_v, 1e-12) * (height - 2 * margin))
            draw.rectangle((x1, y1, x2, y2), fill=(80, 150, 80))
            draw.text((x1, y2 + 4), str(name)[:8], fill=(0, 0, 0))
    if line:
        vals = [(x, y) for x, y in line if not math.isnan(float(y))]
        if len(vals) >= 2:
            xs, ys = [v[0] for v in vals], [v[1] for v in vals]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax == xmin:
                xmax += 1
            if ymax == ymin:
                ymax += 1.0
            pts = []
            for x, y in vals:
                px = margin + int((x - xmin) / (xmax - xmin) * (width - 2 * margin))
                py = height - margin - int((y - ymin) / (ymax - ymin) * (height - 2 * margin))
                pts.append((px, py))
            draw.line(pts, fill=(200, 60, 60), width=3)
            for px, py in pts:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(200, 60, 60))
    ensure_dir(Path(path).parent)
    canvas.save(path)


def select_representative_cases(per_rows, pair_rows, trend_rows, primary_dataset, tags, max_cases):
    target_tag = select_primary_tag(tags)
    primary = [r for r in per_rows if r["dataset"] == primary_dataset and r["run_tag"] == target_tag]
    cases = defaultdict(list)
    for failure_type in FAILURE_TYPES:
        if failure_type == "mixed_or_mild":
            continue
        subset = sorted(
            [r for r in primary if r["failure_type"] == failure_type],
            key=lambda r: (r["IoU"], -abs(r["area_ratio"] - 1.0)),
        )
        cases[failure_type] = [{"dataset": r["dataset"], "stem": r["stem"]} for r in subset[:10]]
    if "csd35" in tags and pair_rows:
        comp = [r for r in pair_rows if r["dataset"] == primary_dataset and r["compare_tag"] == "csd35"]
        improved = sorted(comp, key=lambda r: r["delta_iou"], reverse=True)[:10]
        hurt = sorted(comp, key=lambda r: r["delta_iou"])[:10]
        cases["csd_improved_most"] = [{"dataset": r["dataset"], "stem": r["image_id"]} for r in improved]
        cases["csd_hurt_most"] = [{"dataset": r["dataset"], "stem": r["image_id"]} for r in hurt]
    trend_unique = {}
    for r in trend_rows:
        if r["dataset"] != primary_dataset:
            continue
        trend_unique[(r["dataset"], r["image_id"])] = r
    cases["early_good_late_bad"] = [
        {"dataset": d, "stem": s}
        for (d, s), r in trend_unique.items()
        if r.get("is_early_good_late_bad")
    ][:10]
    cases["always_bad"] = [
        {"dataset": d, "stem": s}
        for (d, s), r in trend_unique.items()
        if r.get("is_always_bad")
    ][:10]

    total = 0
    limited = {}
    for key, values in cases.items():
        keep = []
        for value in values:
            if total >= int(max_cases):
                break
            if value not in keep:
                keep.append(value)
                total += 1
        limited[key] = keep
    return limited


def build_outputs(args, manifests, per_rows, tags, datasets):
    out = Path(args.out).expanduser()
    ensure_dir(out)
    metric_map = build_metric_map(per_rows)
    lookup = build_lookup(manifests)
    method_rows = dataset_summary_rows(per_rows, tags, datasets)
    failure_rows = failure_count_rows(per_rows, tags, datasets)
    pair_rows = pairwise_rows(per_rows, tags, datasets)
    trend_rows = epoch_trend_rows(per_rows, tags, datasets)
    oracle_per_rows, oracle_summary = oracle_rows(per_rows, tags, datasets)
    thresh_rows = threshold_summary_rows(per_rows, tags, datasets)
    primary_tag = select_primary_tag(tags)
    area_corr = area_correlation(per_rows, args.primary_dataset, primary_tag)
    reps = select_representative_cases(per_rows, pair_rows, trend_rows, args.primary_dataset, tags, args.max_case_grids)

    write_csv(out / "per_image_metrics.csv", per_rows, PER_IMAGE_FIELDS)
    top_rows = build_top_per_image(per_rows, pair_rows, args.primary_dataset)
    write_csv(out / "per_image_metrics_top.csv", top_rows, PER_IMAGE_FIELDS)
    write_csv(out / "method_dataset_summary.csv", method_rows, sorted({k for r in method_rows for k in r.keys()}))
    write_csv(out / "failure_type_counts.csv", failure_rows, sorted({k for r in failure_rows for k in r.keys()}))
    write_csv(out / "pairwise_deltas.csv", pair_rows, sorted({k for r in pair_rows for k in r.keys()}))
    write_csv(out / "epoch_trends.csv", trend_rows, sorted({k for r in trend_rows for k in r.keys()}))
    write_csv(out / "oracle_analysis.csv", oracle_per_rows, sorted({k for r in oracle_per_rows for k in r.keys()}))
    write_csv(out / "threshold_analysis.csv", thresh_rows, sorted({k for r in thresh_rows for k in r.keys()}))
    with (out / "representative_cases.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(reps), f, indent=2, ensure_ascii=False)

    make_visuals(out, per_rows, pair_rows, trend_rows, reps, tags, datasets, args.primary_dataset, lookup, metric_map, args)
    summary = build_summary_json(per_rows, method_rows, failure_rows, oracle_summary, thresh_rows, area_corr, reps, tags, datasets, args.primary_dataset)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    write_summary_md(out / "summary.md", summary, method_rows, failure_rows, pair_rows, trend_rows, reps)
    provisional_zip = out.parent / "cfaa_v1_report_pack.zip"
    write_upload_manifest(out, provisional_zip)
    zip_path = write_upload_pack(out)
    write_upload_manifest(out, zip_path)
    return zip_path


def build_top_per_image(per_rows, pair_rows, primary_dataset):
    selected = [r for r in per_rows if r["dataset"] == primary_dataset]
    for dataset in sorted({r["dataset"] for r in per_rows} - {primary_dataset}):
        group = [r for r in per_rows if r["dataset"] == dataset]
        if dataset == "TE-COD10K":
            deltas = {
                (r["dataset"], r["image_id"], r["compare_tag"]): abs(r["delta_recall"] - r["delta_precision"])
                for r in pair_rows
                if r["dataset"] == dataset
            }
            group = sorted(
                group,
                key=lambda r: deltas.get((r["dataset"], r["stem"], r["run_tag"]), 0.0),
                reverse=True,
            )
        else:
            group = sorted(group, key=lambda r: r["IoU"])
        selected.extend(group[:50])
    return selected


def make_visuals(out, per_rows, pair_rows, trend_rows, reps, tags, datasets, primary_dataset, lookup, metric_map, args):
    case_dir = out / "case_grids"
    idx = 0
    case_records = []
    for category, cases in reps.items():
        for case in cases:
            idx += 1
            out_path = case_dir / f"{category}_{idx:03d}.jpg"
            make_case_grid(out_path, case, tags, lookup, metric_map)
            case_records.append((category, case, str(out_path)))

    contact_dir = out / "contact_sheets"
    target_tag = select_primary_tag(tags)
    primary_rows = sorted(
        [r for r in per_rows if r["dataset"] == primary_dataset and r["run_tag"] == target_tag],
        key=lambda r: r["IoU"],
    )
    make_contact_sheet(contact_dir / f"{primary_dataset}_overview.jpg", primary_rows, lookup, metric_map, target_tag, args.contact_sheet_max_cases)
    make_contact_sheet(contact_dir / f"{primary_dataset}_failure_types.jpg", primary_rows, lookup, metric_map, target_tag, args.contact_sheet_max_cases)
    trend_bad = sorted(
        [
            metric_map[(r["dataset"], r["image_id"], target_tag)]
            for r in trend_rows
            if r["dataset"] == primary_dataset and r.get("is_early_good_late_bad") and (r["dataset"], r["image_id"], target_tag) in metric_map
        ],
        key=lambda r: r["IoU"],
    )
    make_contact_sheet(contact_dir / f"{primary_dataset}_epoch_trends.jpg", trend_bad or primary_rows, lookup, metric_map, target_tag, args.contact_sheet_max_cases)
    if "TE-CAMO" in datasets:
        rows = sorted([r for r in per_rows if r["dataset"] == "TE-CAMO" and r["run_tag"] == target_tag], key=lambda r: r["IoU"])
        make_contact_sheet(contact_dir / "CAMO_overview.jpg", rows, lookup, metric_map, target_tag, args.contact_sheet_max_cases)
    if "TE-COD10K" in datasets:
        rows = sorted([r for r in per_rows if r["dataset"] == "TE-COD10K" and r["run_tag"] == target_tag], key=lambda r: r["IoU"])
        make_contact_sheet(contact_dir / "COD10K_contrast.jpg", rows, lookup, metric_map, target_tag, args.contact_sheet_max_cases)

    plot_dir = out / "plots"
    primary = [r for r in per_rows if r["dataset"] == primary_dataset and r["run_tag"] == target_tag]
    simple_plot(plot_dir / f"area_vs_iou_{primary_dataset}.png", f"Area ratio vs IoU ({primary_dataset}/{target_tag})", points=[(r["area_ratio"], r["IoU"]) for r in primary])
    simple_plot(plot_dir / f"recall_precision_tradeoff_{primary_dataset}.png", f"Recall vs Precision ({primary_dataset}/{target_tag})", points=[(r["Recall"], r["Precision"]) for r in primary])
    epoch_means = []
    for tag in tags:
        epoch = parse_csd_epoch(tag)
        if epoch is not None:
            epoch_means.append((epoch, mean([r["IoU"] for r in per_rows if r["dataset"] == primary_dataset and r["run_tag"] == tag])))
    simple_plot(plot_dir / f"epoch_trend_{primary_dataset}.png", f"CSD Epoch IoU Trend ({primary_dataset})", line=sorted(epoch_means))
    counts = Counter(r["failure_type"] for r in primary)
    simple_plot(plot_dir / f"failure_type_bar_{primary_dataset}.png", f"Failure Types ({primary_dataset}/{target_tag})", bars=list(counts.items()))
    oracle_gain = []
    for r in oracle_rows(per_rows, tags, [primary_dataset])[0]:
        oracle_gain.append((len(oracle_gain), r["oracle_gain_over_csd35"]))
    simple_plot(plot_dir / f"oracle_gain_{primary_dataset}.png", f"Oracle Gain ({primary_dataset})", points=oracle_gain)


def build_summary_json(per_rows, method_rows, failure_rows, oracle_summary, thresh_rows, area_corr, reps, tags, datasets, primary_dataset):
    target_tag = select_primary_tag(tags)
    primary = [r for r in per_rows if r["dataset"] == primary_dataset and r["run_tag"] == target_tag]
    failure_counts = Counter(r["failure_type"] for r in primary)
    total = max(len(primary), 1)
    failure_ratio = {k: failure_counts.get(k, 0) / total for k in FAILURE_TYPES}
    mean_precision = mean([r["Precision"] for r in primary])
    mean_recall = mean([r["Recall"] for r in primary])
    mean_iou = mean([r["IoU"] for r in primary])
    thresh_primary = [
        r for r in thresh_rows if r["dataset"] == primary_dataset and r["run_tag"] == target_tag
    ]
    mean_thresh_gain = thresh_primary[0]["mean_threshold_gain_iou"] if thresh_primary else float("nan")
    oracle_gain = oracle_summary.get(primary_dataset, {}).get("oracle_gain", float("nan"))
    under_dom = failure_ratio.get("under_segmentation", 0.0) > 0.35 and mean_precision > mean_recall + 0.15
    misloc_dom = failure_ratio.get("mislocalization", 0.0) + failure_ratio.get("hard_semantic_miss", 0.0) > 0.35
    frag_dom = failure_ratio.get("fragmentation", 0.0) > 0.20
    boundary_dom = failure_ratio.get("boundary_error", 0.0) > 0.25 and mean_iou > 0.50
    calib_likely = (
        (not math.isnan(mean_thresh_gain) and mean_thresh_gain > 0.05)
        or failure_ratio.get("calibration_error", 0.0) > 0.20
    )
    oracle_large = not math.isnan(oracle_gain) and oracle_gain > 0.05
    findings = []
    directions = []
    if under_dom:
        findings.append("CHAMELEON errors are mainly under-segmentation.")
        directions.append("Consider object-completion or structure-level recovery instead of global expansion.")
    if misloc_dom:
        findings.append("Mislocalization or hard semantic miss is substantial on CHAMELEON.")
        directions.append("Prioritize localization / feature source / semantic consistency diagnostics.")
    if frag_dom:
        findings.append("Fragmentation is a visible failure mode.")
        directions.append("Consider connectivity or component-level grouping constraints.")
    if boundary_dom:
        findings.append("Boundary error is dominant after coarse localization succeeds.")
        directions.append("Consider HR decoder or boundary-aware refinement.")
    if calib_likely:
        findings.append("Probability maps may be useful but threshold 0.5 is suboptimal.")
        directions.append("Consider calibration, temperature, or adaptive thresholding.")
    if oracle_large:
        findings.append("Different epochs/methods are complementary.")
        directions.append("Consider image-level router or uncertainty-based checkpoint selection.")
    if not findings:
        findings.append("No single failure family dominates automatically; inspect representative cases.")
        directions.append("Use contact sheets and case grids to choose the next intervention.")
    return {
        "runs": tags,
        "datasets": datasets,
        "primary_dataset": primary_dataset,
        "dataset_summary": method_rows,
        "failure_type_summary": failure_rows,
        "area_correlation": area_corr,
        "oracle_analysis": oracle_summary,
        "threshold_analysis": thresh_rows,
        "main_findings_auto": findings,
        "diagnosis_hints": {
            "undersegmentation_dominant": under_dom,
            "mislocalization_dominant": misloc_dom,
            "fragmentation_dominant": frag_dom,
            "boundary_error_dominant": boundary_dom,
            "calibration_issue_likely": calib_likely,
            "oracle_gain_large": oracle_large,
            "area_explains_failure": area_corr.get("area_explains_failure", "mixed"),
        },
        "recommended_next_directions_auto": directions[:5],
        "representative_cases": reps,
    }


def write_summary_md(path, summary, method_rows, failure_rows, pair_rows, trend_rows, reps):
    lines = []
    lines.append("# CFAA-v1 Failure Anatomy Summary")
    lines.append("")
    lines.append(f"Runs: {', '.join(summary['runs'])}")
    lines.append(f"Datasets: {', '.join(summary['datasets'])}")
    lines.append(f"Primary dataset: {summary['primary_dataset']}")
    lines.append("")
    lines.append("## Dataset / Run Summary")
    lines.append("| dataset | run | mean IoU | mean Dice | mean Recall | mean Precision | mean AreaRatio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in method_rows:
        lines.append(
            f"| {row['dataset']} | {row['run_tag']} | {row.get('mean_IoU', float('nan')):.4f} | "
            f"{row.get('mean_Dice', float('nan')):.4f} | {row.get('mean_Recall', float('nan')):.4f} | "
            f"{row.get('mean_Precision', float('nan')):.4f} | {row.get('mean_area_ratio', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("## Primary Failure Type Distribution")
    primary = summary["primary_dataset"]
    for row in failure_rows:
        if row["dataset"] == primary:
            lines.append(
                f"- {row['run_tag']} / {row['failure_type']}: "
                f"{row['count']} ({row['ratio']:.2%}), mean_iou={row['mean_iou']:.4f}"
            )
    lines.append("")
    lines.append("## Auto Findings")
    for finding in summary["main_findings_auto"]:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Diagnosis Hints")
    for key, value in summary["diagnosis_hints"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Oracle / Epoch / Threshold Notes")
    for dataset, info in summary["oracle_analysis"].items():
        lines.append(
            f"- {dataset}: best_single={info['single_best_run']} "
            f"({info['single_best_run_mean_iou']:.4f}), oracle={info['oracle_mean_iou']:.4f}, "
            f"gain={info['oracle_gain']:.4f}"
        )
    lines.append(f"- Area correlation verdict: {summary['area_correlation'].get('area_explains_failure', 'mixed')}")
    lines.append("")
    lines.append("## Representative Cases")
    for category, cases in reps.items():
        if cases:
            lines.append(f"- {category}: " + ", ".join(f"{c['dataset']}/{c['stem']}" for c in cases[:10]))
    lines.append("")
    lines.append("Upload these files to ChatGPT:")
    lines.append("    cfaa_v1_report_pack.zip")
    lines.append("")
    lines.append("If the zip is too large, upload summary.md, summary.json, failure_type_counts.csv, pairwise_deltas.csv, oracle_analysis.csv, threshold_analysis.csv and CHAMELEON contact sheets.")
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_upload_manifest(out, zip_path):
    lines = [
        f"Primary upload: {zip_path}",
        "Do not upload full prediction masks/probability arrays unless specifically requested.",
        "Core files in zip: summary.md, summary.json, CSV summaries, contact sheets, plots, representative case grids.",
    ]
    (out / "upload_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_upload_pack(out):
    out = Path(out)
    zip_path = out.parent / "cfaa_v1_report_pack.zip"
    core_names = [
        "summary.md",
        "summary.json",
        "method_dataset_summary.csv",
        "failure_type_counts.csv",
        "per_image_metrics_top.csv",
        "pairwise_deltas.csv",
        "epoch_trends.csv",
        "oracle_analysis.csv",
        "threshold_analysis.csv",
        "representative_cases.json",
        "upload_manifest.txt",
    ]
    image_files = []
    for subdir in ("contact_sheets", "plots", "case_grids"):
        image_files.extend(sorted((out / subdir).glob("*.*")))
    limit = 80 * 1024 * 1024
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        total = 0
        for name in core_names:
            path = out / name
            if path.exists():
                zf.write(path, arcname=name)
                total += path.stat().st_size
        for path in image_files:
            size = path.stat().st_size
            if total + size > limit:
                continue
            zf.write(path, arcname=str(path.relative_to(out)))
            total += size
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze prediction failure anatomy and build CFAA-v1 report pack.")
    parser.add_argument("--pred-roots", required=True)
    parser.add_argument("--tags", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--primary-dataset", default="CHAMELEON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-case-grids", type=int, default=80)
    parser.add_argument("--contact-sheet-max-cases", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_roots = parse_csv_arg(args.pred_roots)
    tags = parse_csv_arg(args.tags)
    datasets = parse_csv_arg(args.datasets)
    if len(pred_roots) != len(tags):
        raise ValueError("--pred-roots and --tags must have the same length.")
    if not tags:
        raise ValueError("At least one tag is required.")
    if args.primary_dataset not in datasets:
        raise ValueError("--primary-dataset must be included in --datasets.")

    manifests = {}
    maps = {}
    for tag, root in zip(tags, pred_roots):
        _, rows, mapping = load_manifest(root)
        manifests[tag] = rows
        maps[tag] = mapping

    expected_by_dataset = {}
    first_tag = tags[0]
    for dataset in datasets:
        keys = sorted(key for key in maps[first_tag] if key[0] == dataset)
        if not keys:
            raise RuntimeError(f"No samples for dataset {dataset} in tag {first_tag}")
        expected_by_dataset[dataset] = keys
        for tag in tags[1:]:
            missing = [key for key in keys if key not in maps[tag]]
            if missing:
                raise RuntimeError(f"Missing samples for tag {tag}, first 10: {missing[:10]}")

    per_rows = []
    for tag in tags:
        for dataset in datasets:
            for key in expected_by_dataset[dataset]:
                row = maps[tag][key]
                gt, pred, prob = load_pred(row)
                metrics = compute_metrics(gt, pred, prob)
                per_rows.append(
                    {
                        "dataset": dataset,
                        "stem": key[1],
                        "run_tag": tag,
                        "image_path": row["image_path"],
                        "gt_path": row["gt_path"],
                        "prob_path": row.get("prob_path", ""),
                        "bin_path": row.get("bin_path", ""),
                        **metrics,
                    }
                )
                print(f"[Analyze] {tag} | {dataset}/{key[1]} | IoU={metrics['IoU']:.4f} | {metrics['failure_type']}", flush=True)

    zip_path = build_outputs(args, manifests, per_rows, tags, datasets)
    print(f"[CFAA-v1] report = {Path(args.out).expanduser()}", flush=True)
    print(f"[CFAA-v1] zip = {zip_path}", flush=True)


if __name__ == "__main__":
    main()
