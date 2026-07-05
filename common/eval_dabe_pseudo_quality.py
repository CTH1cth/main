import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import CODMetrics  # noqa: E402
from common.utils import (  # noqa: E402
    build_image_items,
    despl_light_cache_manifest_path,
    ensure_dir,
    load_config,
    manifest_to_map,
    pseudo_manifest_path,
    read_jsonl,
    split_dataset_names,
    torch_load,
    write_json,
)


METRIC_FIELDS = [
    "dabe_version",
    "S_m",
    "F_beta^w",
    "F_beta^m",
    "E_phi^m",
    "MAE",
    "IoU",
    "Precision",
    "Recall",
    "area_ratio",
    "area",
    "num_components",
    "components",
    "fallback_count",
    "small_area_count",
    "large_area_count",
    "num_samples",
]
SUMMARY_FIELDS = ["scope", "dataset", "method", *METRIC_FIELDS]
PER_SAMPLE_FIELDS = [
    "dataset",
    "stem",
    "method",
    "dabe_version",
    "area_ratio",
    "area",
    "precision",
    "recall",
    "iou",
    "mae",
    "fallback_flag",
    "fallback_reason",
    "small_area_flag",
    "large_area_flag",
    "num_components",
    "components",
    "cache_path",
]
PU_DIAG_FIELDS = [
    "dataset",
    "stem",
    "dabe_version",
    "fg_core_precision",
    "fg_core_recall",
    "fg_core_area",
    "fg_core_fallback_precision",
    "fg_core_fallback_recall",
    "fg_core_fallback_area",
    "fg_core_all_precision",
    "fg_core_all_recall",
    "fg_core_all_area",
    "bg_core_precision",
    "bg_core_error_on_gt",
    "bg_core_area",
    "extent_precision",
    "extent_recall",
    "extent_area",
    "fg_core_plus_extent_precision",
    "fg_core_plus_extent_recall",
    "unknown_gt_ratio",
    "unknown_area",
    "strong_bg_false_negative_ratio",
    "strong_fg_false_positive_ratio",
    "weighted_gt_conflict",
    "target_soft_gt_mae",
    "weight_mean",
    "fallback_flag",
    "fallback_reason",
    "cache_path",
]


def _load_gt(path):
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _resize_to_gt(tensor, gt):
    if list(tensor.shape[-2:]) == list(gt.shape[-2:]):
        return tensor.float().clamp(0.0, 1.0)
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)


def _resize_nearest_to_gt(tensor, gt):
    if list(tensor.shape[-2:]) == list(gt.shape[-2:]):
        return tensor.float().clamp(0.0, 1.0)
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=gt.shape[-2:],
        mode="nearest",
    ).squeeze(0).clamp(0.0, 1.0)


def _single_channel(payload, name, cache_path, required=True):
    if name not in payload:
        if required:
            raise KeyError(f"{name} missing from {cache_path}")
        return None
    tensor = payload[name]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor: {cache_path}")
    tensor = tensor.float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"{name} must be [1,H,W], got {list(tensor.shape)}: {cache_path}")
    return tensor.clamp(0.0, 1.0)


def _ratio(num, den, empty_value=0.0):
    den = float(den)
    if den <= 0.0:
        return float(empty_value)
    return float(num) / den


def _pu_diagnostics(gt, payload, cache_path):
    gt = gt.float().clamp(0.0, 1.0)
    gt_fg = gt > 0.5
    gt_bg = ~gt_fg
    fg_core = _resize_nearest_to_gt(_single_channel(payload, "fg_core_pu_68", cache_path), gt) > 0.5
    fallback_tensor = _single_channel(payload, "fg_core_fallback_68", cache_path, required=False)
    if fallback_tensor is None:
        fg_fallback = torch.zeros_like(fg_core)
    else:
        fg_fallback = _resize_nearest_to_gt(fallback_tensor, gt) > 0.5
    bg_core = _resize_nearest_to_gt(_single_channel(payload, "bg_core_pu_68", cache_path), gt) > 0.5
    extent = _resize_to_gt(_single_channel(payload, "extent_candidate_68", cache_path), gt) > 0.0
    unknown = _resize_nearest_to_gt(_single_channel(payload, "unknown_68", cache_path), gt) > 0.5
    target_soft = _resize_to_gt(_single_channel(payload, "target_soft_68", cache_path), gt)
    weight_map = _resize_to_gt(_single_channel(payload, "weight_map_68", cache_path), gt)

    fg_count = float(gt_fg.sum().item())
    bg_count = float(gt_bg.sum().item())
    fg_core_count = float(fg_core.sum().item())
    fg_fallback_count = float(fg_fallback.sum().item())
    fg_all = fg_core | fg_fallback
    fg_all_count = float(fg_all.sum().item())
    bg_core_count = float(bg_core.sum().item())
    extent_count = float(extent.sum().item())
    fg_plus_extent = fg_core | fg_fallback | extent
    fg_plus_extent_count = float(fg_plus_extent.sum().item())
    unknown_count = float(unknown.sum().item())
    strong_bg_conflict = (weight_map >= 0.8) & (target_soft <= 0.1) & gt_fg
    strong_fg_conflict = (weight_map >= 0.8) & (target_soft >= 0.9) & gt_bg
    return {
        "fg_core_precision": _ratio(torch.logical_and(fg_core, gt_fg).sum().item(), fg_core_count, empty_value=1.0),
        "fg_core_recall": _ratio(torch.logical_and(fg_core, gt_fg).sum().item(), fg_count),
        "fg_core_area": float(fg_core.float().mean().item()),
        "fg_core_fallback_precision": _ratio(torch.logical_and(fg_fallback, gt_fg).sum().item(), fg_fallback_count, empty_value=1.0),
        "fg_core_fallback_recall": _ratio(torch.logical_and(fg_fallback, gt_fg).sum().item(), fg_count),
        "fg_core_fallback_area": float(fg_fallback.float().mean().item()),
        "fg_core_all_precision": _ratio(torch.logical_and(fg_all, gt_fg).sum().item(), fg_all_count, empty_value=1.0),
        "fg_core_all_recall": _ratio(torch.logical_and(fg_all, gt_fg).sum().item(), fg_count),
        "fg_core_all_area": float(fg_all.float().mean().item()),
        "bg_core_precision": _ratio(torch.logical_and(bg_core, gt_bg).sum().item(), bg_core_count, empty_value=1.0),
        "bg_core_error_on_gt": _ratio(torch.logical_and(bg_core, gt_fg).sum().item(), fg_count),
        "bg_core_area": float(bg_core.float().mean().item()),
        "extent_precision": _ratio(torch.logical_and(extent, gt_fg).sum().item(), extent_count, empty_value=1.0),
        "extent_recall": _ratio(torch.logical_and(extent, gt_fg).sum().item(), fg_count),
        "extent_area": float(extent.float().mean().item()),
        "fg_core_plus_extent_precision": _ratio(
            torch.logical_and(fg_plus_extent, gt_fg).sum().item(),
            fg_plus_extent_count,
            empty_value=1.0,
        ),
        "fg_core_plus_extent_recall": _ratio(torch.logical_and(fg_plus_extent, gt_fg).sum().item(), fg_count),
        "unknown_gt_ratio": _ratio(torch.logical_and(unknown, gt_fg).sum().item(), unknown_count),
        "unknown_area": float(unknown.float().mean().item()),
        "strong_bg_false_negative_ratio": _ratio(strong_bg_conflict.sum().item(), fg_count),
        "strong_fg_false_positive_ratio": _ratio(strong_fg_conflict.sum().item(), bg_count),
        "weighted_gt_conflict": float((weight_map * torch.abs(target_soft - gt)).mean().item()),
        "target_soft_gt_mae": float(torch.abs(target_soft - gt).mean().item()),
        "weight_mean": float(weight_map.mean().item()),
    }


def _load_dabe(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"DABE cache key mismatch: {row['cache_path']}")
    pseudo = _single_channel(payload, "p_dabe_68", row["cache_path"], required=True)
    return pseudo, payload


def _detect_dabe_version(dabe_map):
    if not dabe_map:
        return "v2"
    first_row = next(iter(dabe_map.values()))
    payload = torch_load(first_row["cache_path"], map_location="cpu")
    if isinstance(payload, dict):
        return str(payload.get("dabe_version", "v2")).lower()
    return "v2"


def _load_light_payload(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DESPL light payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"DESPL light key mismatch: {row['cache_path']}")
    return payload


def _load_fixed_payload(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Fixed pseudo payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Fixed pseudo key mismatch: {row['cache_path']}")
    return _single_channel(payload, "tensor", row["cache_path"], required=True)


def _safe_manifest_map(path, name, logger):
    path = Path(path)
    if not path.exists():
        logger(f"[Warn] {name} manifest missing; comparison disabled: {path}")
        return None
    return manifest_to_map(read_jsonl(path), path)


def _binary_stats(gt, prob, threshold):
    pred = prob > float(threshold)
    gt_b = gt > 0.5
    inter = torch.logical_and(pred, gt_b).sum().item()
    union = torch.logical_or(pred, gt_b).sum().item()
    pred_count = pred.sum().item()
    gt_count = gt_b.sum().item()
    if union == 0:
        iou = 1.0
    else:
        iou = float(inter / union)
    if pred_count == 0:
        precision = 1.0 if gt_count == 0 else 0.0
    else:
        precision = float(inter / pred_count)
    recall = 1.0 if gt_count == 0 else float(inter / gt_count)
    labels, num_labels = ndimage.label(
        pred.squeeze().detach().cpu().numpy().astype(np.uint8),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    del labels
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "area_ratio": float(pred.float().mean().item()),
        "num_components": int(num_labels),
    }


def _cod_result(gt, prob, threshold):
    binary = (prob > float(threshold)).float()
    metrics = CODMetrics()
    metrics.step(gt.unsqueeze(0), binary.unsqueeze(0))
    result = metrics.get_result()
    stats = _binary_stats(gt, prob, threshold)
    return {
        "S_m": float(result["SMeasure"]),
        "F_beta^w": float(result["WFM"]),
        "F_beta^m": float(result["F_MEAN"]),
        "E_phi^m": float(result["E_MEAN"]),
        "MAE": float(result["MAE"]),
        **stats,
        "area": stats["area_ratio"],
        "components": stats["num_components"],
        "fallback_count": 0,
        "small_area_count": 0,
        "large_area_count": 0,
        "num_samples": 1,
    }


class Accumulator:
    def __init__(self):
        self.metrics = CODMetrics()
        self.ious = []
        self.precisions = []
        self.recalls = []
        self.areas = []
        self.components = []
        self.fallback_count = 0
        self.small_area_count = 0
        self.large_area_count = 0
        self.count = 0

    def step(self, gt, prob, threshold, fallback=False, small_area=False, large_area=False):
        binary = (prob > float(threshold)).float()
        self.metrics.step(gt.unsqueeze(0), binary.unsqueeze(0))
        stats = _binary_stats(gt, prob, threshold)
        self.ious.append(stats["IoU"])
        self.precisions.append(stats["Precision"])
        self.recalls.append(stats["Recall"])
        self.areas.append(stats["area_ratio"])
        self.components.append(stats["num_components"])
        self.fallback_count += int(bool(fallback))
        self.small_area_count += int(bool(small_area))
        self.large_area_count += int(bool(large_area))
        self.count += 1

    def result(self):
        result = self.metrics.get_result()
        return {
            "S_m": float(result["SMeasure"]),
            "F_beta^w": float(result["WFM"]),
            "F_beta^m": float(result["F_MEAN"]),
            "E_phi^m": float(result["E_MEAN"]),
            "MAE": float(result["MAE"]),
            "IoU": float(np.mean(self.ious)) if self.ious else 0.0,
            "Precision": float(np.mean(self.precisions)) if self.precisions else 0.0,
            "Recall": float(np.mean(self.recalls)) if self.recalls else 0.0,
            "area_ratio": float(np.mean(self.areas)) if self.areas else 0.0,
            "area": float(np.mean(self.areas)) if self.areas else 0.0,
            "num_components": float(np.mean(self.components)) if self.components else 0.0,
            "components": float(np.mean(self.components)) if self.components else 0.0,
            "fallback_count": int(self.fallback_count),
            "small_area_count": int(self.small_area_count),
            "large_area_count": int(self.large_area_count),
            "num_samples": int(self.count),
        }


def _make_summary_row(scope, dataset, method, accumulator, dabe_version=""):
    return {
        "scope": scope,
        "dataset": dataset,
        "method": method,
        "dabe_version": dabe_version,
        **accumulator.result(),
    }


def _selected_items(cfg, split, max_samples):
    items = build_image_items(cfg.DATA_ROOT, split_dataset_names(cfg, split), require_gt=True)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError(f"No {split} images selected for DABE quality eval.")
    return items


def _write_csv(path, fieldnames, rows):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_meta_version(method, primary_version):
    if method in {"pu_target_hard", "pu_base_hard"}:
        return "pu"
    if method in {"pu_v11_target_hard", "pu_v11_base_hard"}:
        return "pu_v11"
    if method == _dabe_method_name(primary_version):
        return str(primary_version)
    if method == "DABE-PU-v1":
        return "pu"
    if method == "DABE-RAC-Safe":
        return "rac_safe"
    if method == "DABE-RAC":
        return "rac"
    if method == "DABE-GC":
        return "gc"
    if method == "DABE-v2":
        return "v2"
    if method == "DABE-v3.1":
        return "v3_1"
    if method.startswith("DABE-"):
        return method.split("DABE-", 1)[1]
    return ""


def _dabe_method_name(version):
    version = str(version).lower()
    if version == "pu":
        return "pu_target_hard"
    if version == "pu_v11":
        return "pu_v11_target_hard"
    if version == "rac_safe":
        return "DABE-RAC-Safe"
    if version == "rac":
        return "DABE-RAC"
    if version == "gc":
        return "DABE-GC"
    if version == "v3_1":
        return "DABE-v3.1"
    return f"DABE-{version}"


def _quality_prefix(version):
    version = str(version).lower()
    if version == "pu":
        return "dabe_pu_quality"
    if version == "pu_v11":
        return "dabe_pu_v11_quality"
    if version == "rac_safe":
        return "dabe_rac_safe_quality"
    if version == "rac":
        return "dabe_rac_quality"
    if version == "gc":
        return "dabe_gc_quality"
    if version == "v3_1":
        return "dabe_v31_quality"
    return f"dabe_{version}_quality"


def eval_dabe_pseudo_quality(
    cfg,
    dabe_root,
    split="train",
    max_samples=-1,
    compare_fixed=False,
    compare_despl=False,
    compare_dabe_v2_root="",
    compare_dabe_v31_root="",
    compare_dabe_gc_root="",
    compare_dabe_rac_root="",
    compare_dabe_rac_safe_root="",
    compare_dabe_pu_root="",
    out_dir="../workdir/dabe_v2_offline_eval",
    threshold=0.5,
    logger=print,
):
    dabe_root = Path(dabe_root).expanduser()
    manifest_path = dabe_root / f"manifest_{split}.jsonl"
    dabe_map = manifest_to_map(read_jsonl(manifest_path), manifest_path)
    primary_version = _detect_dabe_version(dabe_map)
    primary_method = _dabe_method_name(primary_version)
    items = _selected_items(cfg, split, max_samples)
    out_dir = Path(out_dir).expanduser()
    ensure_dir(out_dir)

    light_map = None
    fixed_map = None
    dabe_v2_map = None
    dabe_v31_map = None
    dabe_gc_map = None
    dabe_rac_map = None
    dabe_rac_safe_map = None
    dabe_pu_map = None
    is_pu = primary_version in {"pu", "pu_v11"}
    if primary_version == "pu_v11":
        methods = ["pu_v11_target_hard", "pu_v11_base_hard"]
    elif primary_version == "pu":
        methods = ["pu_target_hard", "pu_base_hard"]
    else:
        methods = [primary_method]
    if compare_fixed or compare_despl:
        light_map = _safe_manifest_map(despl_light_cache_manifest_path(cfg), "DESPL light", logger)
    if compare_fixed:
        fixed_map = _safe_manifest_map(pseudo_manifest_path(cfg), "fixed pseudo", logger)
        if light_map is not None or fixed_map is not None:
            methods.append("fixed")
    if compare_despl and light_map is not None:
        methods.append("DESPL")
    if compare_dabe_v2_root:
        dabe_v2_manifest = Path(compare_dabe_v2_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_v2_map = _safe_manifest_map(dabe_v2_manifest, "DABE-v2", logger)
        if dabe_v2_map is not None and "DABE-v2" not in methods:
            methods.append("DABE-v2")
    if compare_dabe_v31_root:
        dabe_v31_manifest = Path(compare_dabe_v31_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_v31_map = _safe_manifest_map(dabe_v31_manifest, "DABE-v3.1", logger)
        if dabe_v31_map is not None and "DABE-v3.1" not in methods:
            methods.append("DABE-v3.1")
    if compare_dabe_gc_root:
        dabe_gc_manifest = Path(compare_dabe_gc_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_gc_map = _safe_manifest_map(dabe_gc_manifest, "DABE-GC", logger)
        if dabe_gc_map is not None and "DABE-GC" not in methods:
            methods.append("DABE-GC")
    if compare_dabe_rac_root:
        dabe_rac_manifest = Path(compare_dabe_rac_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_rac_map = _safe_manifest_map(dabe_rac_manifest, "DABE-RAC", logger)
        if dabe_rac_map is not None and "DABE-RAC" not in methods:
            methods.append("DABE-RAC")
    if compare_dabe_rac_safe_root:
        dabe_rac_safe_manifest = Path(compare_dabe_rac_safe_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_rac_safe_map = _safe_manifest_map(dabe_rac_safe_manifest, "DABE-RAC-Safe", logger)
        if dabe_rac_safe_map is not None and "DABE-RAC-Safe" not in methods:
            methods.append("DABE-RAC-Safe")
    if compare_dabe_pu_root:
        dabe_pu_manifest = Path(compare_dabe_pu_root).expanduser() / f"manifest_{split}.jsonl"
        dabe_pu_map = _safe_manifest_map(dabe_pu_manifest, "DABE-PU-v1", logger)
        if dabe_pu_map is not None and "DABE-PU-v1" not in methods:
            methods.append("DABE-PU-v1")

    logger("train_gt_used = true | diagnostic_only = true")
    logger(f"dabe_manifest = {manifest_path}")
    logger(f"dabe_version = {primary_version}")
    logger(f"compare_dabe_v2_root = {compare_dabe_v2_root or ''}")
    logger(f"compare_dabe_v31_root = {compare_dabe_v31_root or ''}")
    logger(f"compare_dabe_gc_root = {compare_dabe_gc_root or ''}")
    logger(f"compare_dabe_rac_root = {compare_dabe_rac_root or ''}")
    logger(f"compare_dabe_rac_safe_root = {compare_dabe_rac_safe_root or ''}")
    logger(f"compare_dabe_pu_root = {compare_dabe_pu_root or ''}")
    logger(f"out_dir = {out_dir}")
    logger(f"split = {split}")
    logger(f"threshold = {float(threshold):.4f}")
    logger(f"methods = {', '.join(methods)}")
    logger(f"num_items = {len(items)}")

    by_dataset = defaultdict(lambda: {method: Accumulator() for method in methods})
    overall = {method: Accumulator() for method in methods}
    per_sample_rows = []
    pu_diag_rows = []

    for item in items:
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        if key not in dabe_map:
            raise RuntimeError(f"DABE cache missing for {dataset}/{stem}")
        gt = _load_gt(item["gt_path"]).float()
        dabe_prob, dabe_payload = _load_dabe(dabe_map[key], dataset, stem)
        dabe_prob = _resize_to_gt(dabe_prob, gt)
        if is_pu:
            target_prob = _resize_to_gt(_single_channel(dabe_payload, "target_soft_68", dabe_map[key]["cache_path"]), gt)
            base_prob = _resize_to_gt(_single_channel(dabe_payload, "p_base_68", dabe_map[key]["cache_path"]), gt)
            target_method = "pu_v11_target_hard" if primary_version == "pu_v11" else "pu_target_hard"
            base_method = "pu_v11_base_hard" if primary_version == "pu_v11" else "pu_base_hard"
            tensors = {target_method: target_prob, base_method: base_prob}
            method_meta = {
                target_method: {
                    "dabe_version": primary_version,
                    "fallback_flag": bool(dabe_payload.get("fallback_flag", False)),
                    "fallback_reason": str(dabe_payload.get("fallback_reason", "none")),
                    "small_area_flag": bool(dabe_payload.get("small_area_flag", False)),
                    "large_area_flag": bool(dabe_payload.get("large_area_flag", False)),
                    "cache_path": dabe_map[key]["cache_path"],
                },
                base_method: {
                    "dabe_version": primary_version,
                    "fallback_flag": bool(dabe_payload.get("fallback_flag", False)),
                    "fallback_reason": str(dabe_payload.get("fallback_reason", "none")),
                    "small_area_flag": bool(dabe_payload.get("small_area_flag", False)),
                    "large_area_flag": bool(dabe_payload.get("large_area_flag", False)),
                    "cache_path": dabe_map[key]["cache_path"],
                },
            }
            pu_diag = _pu_diagnostics(gt, dabe_payload, dabe_map[key]["cache_path"])
            pu_diag_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "dabe_version": primary_version,
                    **pu_diag,
                    "fallback_flag": bool(dabe_payload.get("fallback_flag", False)),
                    "fallback_reason": str(dabe_payload.get("fallback_reason", "none")),
                    "cache_path": dabe_map[key]["cache_path"],
                }
            )
        else:
            tensors = {primary_method: dabe_prob}
            method_meta = {
                primary_method: {
                    "dabe_version": str(dabe_payload.get("dabe_version", primary_version)),
                    "fallback_flag": bool(dabe_payload.get("fallback_flag", False)),
                    "fallback_reason": str(dabe_payload.get("fallback_reason", "none")),
                    "small_area_flag": bool(dabe_payload.get("small_area_flag", False)),
                    "large_area_flag": bool(dabe_payload.get("large_area_flag", False)),
                    "cache_path": dabe_map[key]["cache_path"],
                }
            }

        if "DABE-PU-v1" in methods and method_meta.get("DABE-PU-v1") is None:
            if dabe_pu_map is None or key not in dabe_pu_map:
                raise RuntimeError(f"DABE-PU-v1 comparison cache missing for {dataset}/{stem}")
            dabe_pu_prob, dabe_pu_payload = _load_dabe(dabe_pu_map[key], dataset, stem)
            tensors["DABE-PU-v1"] = _resize_to_gt(dabe_pu_prob, gt)
            method_meta["DABE-PU-v1"] = {
                "dabe_version": str(dabe_pu_payload.get("dabe_version", "pu")),
                "fallback_flag": bool(dabe_pu_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_pu_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_pu_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_pu_payload.get("large_area_flag", False)),
                "cache_path": dabe_pu_map[key]["cache_path"],
            }

        if "fixed" in methods:
            fixed_prob = None
            fixed_cache_path = ""
            if light_map is not None and key in light_map:
                light_payload = _load_light_payload(light_map[key], dataset, stem)
                fixed_prob = _single_channel(
                    light_payload,
                    "p_fixed_68",
                    light_map[key]["cache_path"],
                    required=False,
                )
                fixed_cache_path = light_map[key]["cache_path"]
            if fixed_prob is None and fixed_map is not None and key in fixed_map:
                fixed_prob = _load_fixed_payload(fixed_map[key], dataset, stem)
                fixed_cache_path = fixed_map[key]["cache_path"]
            if fixed_prob is None:
                raise RuntimeError(f"Fixed pseudo missing for {dataset}/{stem}")
            tensors["fixed"] = _resize_to_gt(fixed_prob, gt)
            method_meta["fixed"] = {
                "dabe_version": "",
                "fallback_flag": False,
                "fallback_reason": "none",
                "small_area_flag": False,
                "large_area_flag": False,
                "cache_path": fixed_cache_path,
            }

        if "DESPL" in methods:
            if key not in light_map:
                raise RuntimeError(f"DESPL light cache missing for {dataset}/{stem}")
            light_payload = _load_light_payload(light_map[key], dataset, stem)
            despl_prob = _single_channel(
                light_payload,
                "p_despl_68",
                light_map[key]["cache_path"],
                required=True,
            )
            tensors["DESPL"] = _resize_to_gt(despl_prob, gt)
            method_meta["DESPL"] = {
                "dabe_version": "",
                "fallback_flag": False,
                "fallback_reason": "none",
                "small_area_flag": False,
                "large_area_flag": False,
                "cache_path": light_map[key]["cache_path"],
            }

        if "DABE-v2" in methods and method_meta.get("DABE-v2") is None:
            if dabe_v2_map is None or key not in dabe_v2_map:
                raise RuntimeError(f"DABE-v2 comparison cache missing for {dataset}/{stem}")
            dabe_v2_prob, dabe_v2_payload = _load_dabe(dabe_v2_map[key], dataset, stem)
            tensors["DABE-v2"] = _resize_to_gt(dabe_v2_prob, gt)
            method_meta["DABE-v2"] = {
                "dabe_version": str(dabe_v2_payload.get("dabe_version", "v2")),
                "fallback_flag": bool(dabe_v2_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_v2_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_v2_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_v2_payload.get("large_area_flag", False)),
                "cache_path": dabe_v2_map[key]["cache_path"],
            }

        if "DABE-v3.1" in methods and method_meta.get("DABE-v3.1") is None:
            if dabe_v31_map is None or key not in dabe_v31_map:
                raise RuntimeError(f"DABE-v3.1 comparison cache missing for {dataset}/{stem}")
            dabe_v31_prob, dabe_v31_payload = _load_dabe(dabe_v31_map[key], dataset, stem)
            tensors["DABE-v3.1"] = _resize_to_gt(dabe_v31_prob, gt)
            method_meta["DABE-v3.1"] = {
                "dabe_version": str(dabe_v31_payload.get("dabe_version", "v3_1")),
                "fallback_flag": bool(dabe_v31_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_v31_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_v31_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_v31_payload.get("large_area_flag", False)),
                "cache_path": dabe_v31_map[key]["cache_path"],
            }

        if "DABE-GC" in methods and method_meta.get("DABE-GC") is None:
            if dabe_gc_map is None or key not in dabe_gc_map:
                raise RuntimeError(f"DABE-GC comparison cache missing for {dataset}/{stem}")
            dabe_gc_prob, dabe_gc_payload = _load_dabe(dabe_gc_map[key], dataset, stem)
            tensors["DABE-GC"] = _resize_to_gt(dabe_gc_prob, gt)
            method_meta["DABE-GC"] = {
                "dabe_version": str(dabe_gc_payload.get("dabe_version", "gc")),
                "fallback_flag": bool(dabe_gc_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_gc_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_gc_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_gc_payload.get("large_area_flag", False)),
                "cache_path": dabe_gc_map[key]["cache_path"],
            }

        if "DABE-RAC" in methods and method_meta.get("DABE-RAC") is None:
            if dabe_rac_map is None or key not in dabe_rac_map:
                raise RuntimeError(f"DABE-RAC comparison cache missing for {dataset}/{stem}")
            dabe_rac_prob, dabe_rac_payload = _load_dabe(dabe_rac_map[key], dataset, stem)
            tensors["DABE-RAC"] = _resize_to_gt(dabe_rac_prob, gt)
            method_meta["DABE-RAC"] = {
                "dabe_version": str(dabe_rac_payload.get("dabe_version", "rac")),
                "fallback_flag": bool(dabe_rac_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_rac_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_rac_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_rac_payload.get("large_area_flag", False)),
                "cache_path": dabe_rac_map[key]["cache_path"],
            }

        if "DABE-RAC-Safe" in methods and method_meta.get("DABE-RAC-Safe") is None:
            if dabe_rac_safe_map is None or key not in dabe_rac_safe_map:
                raise RuntimeError(f"DABE-RAC-Safe comparison cache missing for {dataset}/{stem}")
            dabe_rac_safe_prob, dabe_rac_safe_payload = _load_dabe(dabe_rac_safe_map[key], dataset, stem)
            tensors["DABE-RAC-Safe"] = _resize_to_gt(dabe_rac_safe_prob, gt)
            method_meta["DABE-RAC-Safe"] = {
                "dabe_version": str(dabe_rac_safe_payload.get("dabe_version", "rac_safe")),
                "fallback_flag": bool(dabe_rac_safe_payload.get("fallback_flag", False)),
                "fallback_reason": str(dabe_rac_safe_payload.get("fallback_reason", "none")),
                "small_area_flag": bool(dabe_rac_safe_payload.get("small_area_flag", False)),
                "large_area_flag": bool(dabe_rac_safe_payload.get("large_area_flag", False)),
                "cache_path": dabe_rac_safe_map[key]["cache_path"],
            }

        for method in methods:
            meta = method_meta[method]
            fallback = bool(meta["fallback_flag"]) if method.startswith("DABE") else False
            small_area = bool(meta.get("small_area_flag", False)) if method.startswith("DABE") else False
            large_area = bool(meta["large_area_flag"]) if method.startswith("DABE") else False
            by_dataset[dataset][method].step(
                gt,
                tensors[method],
                threshold,
                fallback=fallback,
                small_area=small_area,
                large_area=large_area,
            )
            overall[method].step(
                gt,
                tensors[method],
                threshold,
                fallback=fallback,
                small_area=small_area,
                large_area=large_area,
            )
            sample_values = _cod_result(gt, tensors[method], threshold)
            sample_values["fallback_count"] = int(fallback)
            sample_values["large_area_count"] = int(large_area)
            per_sample_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "method": method,
                    "dabe_version": meta.get("dabe_version", ""),
                    "area_ratio": sample_values["area_ratio"],
                    "area": sample_values["area"],
                    "precision": sample_values["Precision"],
                    "recall": sample_values["Recall"],
                    "iou": sample_values["IoU"],
                    "mae": sample_values["MAE"],
                    "fallback_flag": bool(meta["fallback_flag"]),
                    "fallback_reason": meta["fallback_reason"],
                    "small_area_flag": bool(meta.get("small_area_flag", False)),
                    "large_area_flag": bool(meta["large_area_flag"]),
                    "num_components": sample_values["num_components"],
                    "components": sample_values["components"],
                    "cache_path": meta["cache_path"],
                }
            )

    by_dataset_rows = []
    for dataset in sorted(by_dataset):
        for method in methods:
            version = method_meta_version(method, primary_version)
            by_dataset_rows.append(
                _make_summary_row("dataset", dataset, method, by_dataset[dataset][method], dabe_version=version)
            )

    summary_rows = [
        _make_summary_row(
            "overall",
            "ALL",
            method,
            overall[method],
            dabe_version=method_meta_version(method, primary_version),
        )
        for method in methods
    ]
    all_rows = [*by_dataset_rows, *summary_rows]

    prefix = _quality_prefix(primary_version)
    summary_json_path = out_dir / f"{prefix}_summary.json"
    summary_csv_path = out_dir / f"{prefix}_summary.csv"
    by_dataset_csv_path = out_dir / f"{prefix}_by_dataset.csv"
    per_sample_csv_path = out_dir / f"{prefix}_per_sample.csv"
    pu_diag_csv_path = out_dir / f"{prefix}_pu_diagnostics.csv"

    write_json(
        summary_json_path,
        {
            "split": split,
            "threshold": float(threshold),
            "dabe_root": str(dabe_root.resolve()),
            "dabe_version": primary_version,
            "manifest_path": str(manifest_path.resolve()),
            "compare_dabe_v2_root": str(Path(compare_dabe_v2_root).expanduser().resolve()) if compare_dabe_v2_root else "",
            "compare_dabe_v31_root": str(Path(compare_dabe_v31_root).expanduser().resolve()) if compare_dabe_v31_root else "",
            "compare_dabe_gc_root": str(Path(compare_dabe_gc_root).expanduser().resolve()) if compare_dabe_gc_root else "",
            "compare_dabe_rac_root": str(Path(compare_dabe_rac_root).expanduser().resolve()) if compare_dabe_rac_root else "",
            "compare_dabe_rac_safe_root": str(Path(compare_dabe_rac_safe_root).expanduser().resolve()) if compare_dabe_rac_safe_root else "",
            "compare_dabe_pu_root": str(Path(compare_dabe_pu_root).expanduser().resolve()) if compare_dabe_pu_root else "",
            "methods": methods,
            "overall": {row["method"]: row for row in summary_rows},
            "pu_diagnostics": pu_diag_rows if is_pu else [],
            "by_dataset": {
                dataset: {
                    row["method"]: row
                    for row in by_dataset_rows
                    if row["dataset"] == dataset
                }
                for dataset in sorted(by_dataset)
            },
        },
    )
    _write_csv(summary_csv_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(by_dataset_csv_path, SUMMARY_FIELDS, by_dataset_rows)
    _write_csv(per_sample_csv_path, PER_SAMPLE_FIELDS, per_sample_rows)
    if is_pu:
        _write_csv(pu_diag_csv_path, PU_DIAG_FIELDS, pu_diag_rows)

    for row in summary_rows:
        logger(
            f"[Quality] {row['method']} | S_m={row['S_m']:.4f} | "
            f"Fw={row['F_beta^w']:.4f} | E={row['E_phi^m']:.4f} | "
            f"MAE={row['MAE']:.4f} | IoU={row['IoU']:.4f} | "
            f"Precision={row['Precision']:.4f} | Recall={row['Recall']:.4f} | "
            f"Area={row['area_ratio']:.4f} | components={row['num_components']:.2f} | "
            f"fallback_count={row['fallback_count']} | "
            f"small_area_count={row['small_area_count']} | "
            f"large_area_count={row['large_area_count']} | n={row['num_samples']}"
        )
    logger(f"wrote_summary_json = {summary_json_path}")
    logger(f"wrote_summary_csv = {summary_csv_path}")
    logger(f"wrote_by_dataset_csv = {by_dataset_csv_path}")
    logger(f"wrote_per_sample_csv = {per_sample_csv_path}")
    if is_pu:
        logger(f"wrote_pu_diagnostics_csv = {pu_diag_csv_path}")
    return {
        "summary_json": summary_json_path,
        "summary_csv": summary_csv_path,
        "by_dataset_csv": by_dataset_csv_path,
        "per_sample_csv": per_sample_csv_path,
        "pu_diagnostics_csv": pu_diag_csv_path if is_pu else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DABE pseudo labels against GT for diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", default="../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--compare_fixed", action="store_true")
    parser.add_argument("--compare_despl", action="store_true")
    parser.add_argument("--compare_dabe_v2_root", default="")
    parser.add_argument("--compare_dabe_v31_root", default="")
    parser.add_argument("--compare_dabe_gc_root", default="")
    parser.add_argument("--compare_dabe_rac_root", default="")
    parser.add_argument("--compare_dabe_rac_safe_root", default="")
    parser.add_argument("--compare_dabe_pu_root", default="")
    parser.add_argument("--out_dir", default="../workdir/dabe_v2_offline_eval")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold must be in [0,1].")

    cfg = load_config(args.config)
    eval_dabe_pseudo_quality(
        cfg,
        dabe_root=args.dabe_root,
        split=args.split,
        max_samples=args.max_samples,
        compare_fixed=args.compare_fixed,
        compare_despl=args.compare_despl,
        compare_dabe_v2_root=args.compare_dabe_v2_root,
        compare_dabe_v31_root=args.compare_dabe_v31_root,
        compare_dabe_gc_root=args.compare_dabe_gc_root,
        compare_dabe_rac_root=args.compare_dabe_rac_root,
        compare_dabe_rac_safe_root=args.compare_dabe_rac_safe_root,
        compare_dabe_pu_root=args.compare_dabe_pu_root,
        out_dir=args.out_dir,
        threshold=args.threshold,
        logger=print,
    )


if __name__ == "__main__":
    main()
