#!/usr/bin/env python3
"""Offline GT-only audit for DABE Bridge/Clean targets.

GT is used only inside this research audit.  No GT value is written to either
training cache and this tool does not instantiate the training Dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_cache import parse_bool  # noqa: E402
from common.utils import find_gt_path, manifest_to_map, read_jsonl, torch_load  # noqa: E402


DEFAULT_SOURCE = "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
DEFAULT_BRIDGE = "../datasets/cache/dabe_bridge_v1_pseudo_cache/dinov1-s8"
DEFAULT_CLEAN = "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8"
DEFAULT_DATA = "/home/dell01/CTH/MY-baseline/datasets/COD"
DEFAULT_OUTPUT = "../workdir/dabe_clean_v1_audit"
TARGET_KEYS = {
    "O0_p_base": ("source", "p_base_68"),
    "O1_pu_target_soft": ("source", "target_soft_68"),
    "O3_bridge": ("bridge", "bridge_target_68"),
    "O4_target_dp": ("clean", "target_dp_68"),
    "O5_target_diff": ("clean", "target_diff_68"),
}
AUDIT_THRESHOLDS = {
    "precision_drop_max": 0.02,
    "fp_ratio_increase_max": 0.01,
    "iou_drop_max": 0.03,
    "binary_area_ratio_min": 0.75,
    "binary_area_ratio_max": 1.25,
    "neutral_ratio_min": 0.01,
    "neutral_ratio_max": 0.95,
    "strong_bg_ratio_min": 0.01,
}


def _prepare_output(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tensor(payload, key, shape=(1, 68, 68)):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Audit payload is missing tensor {key!r}.")
    value = value.detach().cpu().float()
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(f"{key} shape mismatch: {list(value.shape)} != {list(shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} contains NaN/Inf.")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"{key} is outside [0,1].")
    return value.clamp(0.0, 1.0)


def _load_gt(data_root, dataset, stem):
    path = find_gt_path(data_root, dataset, stem)
    array = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return torch.from_numpy(array > 127), str(path.resolve())


def _resize_binary(target, height, width):
    resized = F.interpolate(
        target.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized > 0.5


def _sample_counts(pred, gt):
    pred = pred.bool()
    gt = gt.bool()
    tp = int((pred & gt).sum().item())
    fp = int((pred & (~gt)).sum().item())
    fn = int(((~pred) & gt).sum().item())
    tn = int(((~pred) & (~gt)).sum().item())
    components = int(ndimage.label(pred.numpy(), structure=np.ones((3, 3)))[1])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "components": components}


def _metrics_from_counts(counts):
    eps = 1e-12
    tp, fp, fn, tn = (float(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    beta2 = 0.3
    f_beta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_beta_beta2_0.3": f_beta,
        "iou": tp / (tp + fp + fn + eps),
        "false_positive_ratio": fp / (fp + tn + eps),
        "false_negative_ratio": fn / (fn + tp + eps),
        "binary_area": (tp + fp) / (tp + fp + fn + tn + eps),
        "gt_area": (tp + fn) / (tp + fp + fn + tn + eps),
        "prediction_gt_area_ratio": (tp + fp) / (tp + fn + eps),
    }


def _distribution(target):
    flat = target.double().reshape(-1)
    count = int(flat.numel())
    return {
        "target_sum": float(flat.sum().item()),
        "target_sq_sum": float(flat.square().sum().item()),
        "target_count": count,
        "bin_0_0.1": int(((flat >= 0.0) & (flat < 0.1)).sum().item()),
        "bin_0.1_0.4": int(((flat >= 0.1) & (flat < 0.4)).sum().item()),
        "bin_0.4_0.6": int(((flat >= 0.4) & (flat <= 0.6)).sum().item()),
        "bin_0.6_0.9": int(((flat > 0.6) & (flat <= 0.9)).sum().item()),
        "bin_0.9_1.0": int(((flat > 0.9) & (flat <= 1.0)).sum().item()),
        "neutral_count": int(((flat - 0.5).abs() <= 0.1).sum().item()),
        "strong_fg_count": int((flat >= 0.9).sum().item()),
        "strong_bg_count": int((flat <= 0.1).sum().item()),
    }


def _merge_accumulator(acc, counts, distribution):
    for key in ("tp", "fp", "fn", "tn", "components"):
        acc[key] += counts[key]
    acc["samples"] += 1
    for key, value in distribution.items():
        acc[key] += value


def _finalize_accumulator(acc):
    row = _metrics_from_counts(acc)
    count = max(1, int(acc["target_count"]))
    mean = float(acc["target_sum"]) / count
    variance = max(0.0, float(acc["target_sq_sum"]) / count - mean * mean)
    row.update(
        {
            "samples": int(acc["samples"]),
            "connected_component_count_mean": float(acc["components"])
            / max(1, int(acc["samples"])),
            "target_mean": mean,
            "target_std": math.sqrt(variance),
            "ratio_[0.0,0.1)": float(acc["bin_0_0.1"]) / count,
            "ratio_[0.1,0.4)": float(acc["bin_0.1_0.4"]) / count,
            "ratio_[0.4,0.6]": float(acc["bin_0.4_0.6"]) / count,
            "ratio_(0.6,0.9]": float(acc["bin_0.6_0.9"]) / count,
            "ratio_(0.9,1.0]": float(acc["bin_0.9_1.0"]) / count,
            "neutral_ratio": float(acc["neutral_count"]) / count,
            "strong_foreground_ratio": float(acc["strong_fg_count"]) / count,
            "strong_background_ratio": float(acc["strong_bg_count"]) / count,
        }
    )
    return row


def _gradient_metrics(reference, candidate):
    reference = reference.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    error = (reference - candidate).abs()
    ref_centered = reference - reference.mean()
    cand_centered = candidate - candidate.mean()
    eps = 1e-18
    cosine = float(torch.dot(reference, candidate) / (reference.norm() * candidate.norm() + eps))
    pearson = float(
        torch.dot(ref_centered, cand_centered)
        / (ref_centered.norm() * cand_centered.norm() + eps)
    )
    # A float32 target stores 0.5 + delta as exactly 0.5 when delta is below
    # one ULP around 0.5. Treat both sides inside this explicitly reported
    # numerical-neutral band as zero before the strict sign comparison.
    sign_zero_tolerance = 1e-7
    # Apply the neutral decision jointly.  Thresholding both gradients
    # independently can turn two same-sign float32 values into ``0`` and ``+1``
    # merely because one of them lies on either side of the tolerance boundary.
    jointly_neutral = (
        (reference.abs() <= sign_zero_tolerance)
        & (candidate.abs() <= sign_zero_tolerance)
    )
    reference_sign = torch.sign(reference)
    candidate_sign = torch.sign(candidate)
    reference_sign = torch.where(
        jointly_neutral, torch.zeros_like(reference_sign), reference_sign
    )
    candidate_sign = torch.where(
        jointly_neutral, torch.zeros_like(candidate_sign), candidate_sign
    )
    sign_agreement = float((reference_sign == candidate_sign).double().mean())
    k = max(1, int(math.ceil(reference.numel() * 0.10)))
    ref_top = set(torch.topk(reference.abs(), k=k).indices.tolist())
    cand_top = set(torch.topk(candidate.abs(), k=k).indices.tolist())
    top_overlap = len(ref_top.intersection(cand_top)) / float(k)
    positive_mass = float(candidate.clamp_min(0.0).sum().item())
    negative_mass = float((-candidate.clamp_max(0.0)).sum().item())
    return {
        "max_abs_error": float(error.max().item()),
        "pearson": pearson,
        "cosine": cosine,
        "sign_agreement": sign_agreement,
        "sign_zero_tolerance": sign_zero_tolerance,
        "top10_abs_overlap": top_overlap,
        "mean_abs_gradient": float(candidate.abs().mean().item()),
        "positive_gradient_mass": positive_mass,
        "negative_gradient_mass": negative_mass,
        "positive_negative_mass_ratio": positive_mass / (negative_mass + eps),
    }


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_pass(candidate, reference):
    checks = {
        "precision": candidate["precision"] >= reference["precision"] - 0.02,
        "false_positive_ratio": candidate["false_positive_ratio"]
        <= reference["false_positive_ratio"] + 0.01,
        "iou": candidate["iou"] >= reference["iou"] - 0.03,
        "binary_area_ratio": 0.75
        <= candidate["binary_area"] / max(reference["binary_area"], 1e-12)
        <= 1.25,
        "neutral_ratio": 0.01 <= candidate["neutral_ratio"] <= 0.95,
        "strong_background_ratio": candidate["strong_background_ratio"] >= 0.01,
    }
    return bool(all(checks.values())), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--bridge-root", default=DEFAULT_BRIDGE)
    parser.add_argument("--clean-root", default=DEFAULT_CLEAN)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer.")

    output_root = _prepare_output(args.output_root, args.overwrite)
    source_manifest = Path(args.source_root) / "manifest_train.jsonl"
    source_rows = read_jsonl(source_manifest)
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    source_rows = [row for row in source_rows if str(row.get("dataset")) in datasets]
    if args.max_samples > 0:
        source_rows = source_rows[: args.max_samples]
    bridge_map = manifest_to_map(
        read_jsonl(Path(args.bridge_root) / "manifest_train.jsonl"),
        Path(args.bridge_root) / "manifest_train.jsonl",
    )
    clean_map = manifest_to_map(
        read_jsonl(Path(args.clean_root) / "manifest_train.jsonl"),
        Path(args.clean_root) / "manifest_train.jsonl",
    )
    if not source_rows:
        raise RuntimeError("No audit samples selected.")

    accumulators = defaultdict(lambda: defaultdict(float))
    dataset_accumulators = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    per_sample_rows = []
    gradient_rows = []
    reduction_scales = []
    for row in source_rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key not in bridge_map or key not in clean_map:
            raise RuntimeError(f"Bridge/Clean cache is missing audit key {key}.")
        source = torch_load(row["cache_path"], map_location="cpu")
        bridge = torch_load(bridge_map[key]["cache_path"], map_location="cpu")
        clean = torch_load(clean_map[key]["cache_path"], map_location="cpu")
        payloads = {"source": source, "bridge": bridge, "clean": clean}
        gt, gt_path = _load_gt(args.data_root, *key)
        targets = {
            name: _tensor(payloads[payload_name], tensor_key)
            for name, (payload_name, tensor_key) in TARGET_KEYS.items()
        }
        for name, target in targets.items():
            pred = _resize_binary(target, int(gt.shape[0]), int(gt.shape[1]))
            counts = _sample_counts(pred, gt)
            distribution = _distribution(target)
            _merge_accumulator(accumulators[name], counts, distribution)
            _merge_accumulator(dataset_accumulators[key[0]][name], counts, distribution)
            sample_metrics = _metrics_from_counts(counts)
            sample_metrics.update(
                {
                    "dataset": key[0],
                    "stem": key[1],
                    "target": name,
                    "gt_path": gt_path,
                    "connected_component_count": counts["components"],
                    "target_mean": float(target.mean().item()),
                    "neutral_ratio": float(((target - 0.5).abs() <= 0.1).float().mean().item()),
                }
            )
            per_sample_rows.append(sample_metrics)

        old_gradient = _tensor(source, "weight_map_68") * (
            _tensor(source, "target_soft_68") - 0.5
        )
        gradient_candidates = {
            "O3_bridge": targets["O3_bridge"] - 0.5,
            "O4_target_dp": targets["O4_target_dp"] - 0.5,
            "O5_target_diff": targets["O5_target_diff"] - 0.5,
        }
        for name, gradient in gradient_candidates.items():
            stats = _gradient_metrics(old_gradient, gradient)
            stats.update({"dataset": key[0], "stem": key[1], "candidate": name})
            gradient_rows.append(stats)
        bridge_stats = gradient_rows[-3]
        if bridge_stats["max_abs_error"] >= 1e-6:
            raise RuntimeError(f"Bridge gradient max error failed for {key}: {bridge_stats}")
        if bridge_stats["cosine"] <= 0.999999 or bridge_stats["sign_agreement"] != 1.0:
            raise RuntimeError(f"Bridge gradient direction failed for {key}: {bridge_stats}")
        weight = _tensor(source, "weight_map_68")
        reduction_scales.append(float(weight.numel()) / max(float(weight.sum()), 1e-12))

    overall_rows = []
    finalized = {}
    for name in TARGET_KEYS:
        finalized[name] = _finalize_accumulator(accumulators[name])
        overall_rows.append({"target": name, "kind": "pseudo_target", **finalized[name]})
    overall_rows.insert(
        2,
        {
            "target": "O2_old_effective_gradient",
            "kind": "gradient_reference_only",
            "note": "No pseudo-label quality metrics: O2 is not an independent target.",
        },
    )
    by_dataset_rows = []
    for dataset in sorted(dataset_accumulators):
        for name in TARGET_KEYS:
            by_dataset_rows.append(
                {
                    "dataset": dataset,
                    "target": name,
                    **_finalize_accumulator(dataset_accumulators[dataset][name]),
                }
            )

    reference = finalized["O0_p_base"]
    dp_pass, dp_checks = _candidate_pass(finalized["O4_target_dp"], reference)
    diff_pass, diff_checks = _candidate_pass(finalized["O5_target_diff"], reference)
    recommendation = "dp" if dp_pass else ("diff" if diff_pass else "STOP")
    diagnostic_only = int(args.max_samples) != -1
    gradient_summary = {}
    for candidate in ("O3_bridge", "O4_target_dp", "O5_target_diff"):
        rows = [row for row in gradient_rows if row["candidate"] == candidate]
        gradient_summary[candidate] = {
            key: (max(row[key] for row in rows) if key == "max_abs_error" else float(np.mean([row[key] for row in rows])))
            for key in (
                "max_abs_error",
                "pearson",
                "cosine",
                "sign_agreement",
                "top10_abs_overlap",
                "mean_abs_gradient",
                "positive_gradient_mass",
                "negative_gradient_mass",
                "positive_negative_mass_ratio",
            )
        }

    recommendation_payload = {
        "recommendation": recommendation,
        "diagnostic_only": diagnostic_only,
        "full_training_authorized": False if diagnostic_only else recommendation != "STOP",
        "thresholds": AUDIT_THRESHOLDS,
        "dp_pass": dp_pass,
        "dp_checks": dp_checks,
        "diff_pass": diff_pass,
        "diff_checks": diff_checks,
        "bridge_gradient": gradient_summary["O3_bridge"],
        "actual_reduced_gradient_equivalence": {
            "exact_absolute_equivalence": False,
            "direction_equivalence": True,
            "reason": "legacy weighted BCE divides by sum(W); unweighted BCE divides by N",
            "N_over_sumW_min": min(reduction_scales),
            "N_over_sumW_mean": float(np.mean(reduction_scales)),
            "N_over_sumW_max": max(reduction_scales),
        },
        "samples": len(source_rows),
        "datasets": sorted(datasets),
        "TRAIN_GT_USED_FOR_AUDIT_ONLY": True,
        "TRAIN_GT_USED_FOR_TRAINING": False,
    }
    _write_csv(output_root / "overall.csv", overall_rows)
    _write_csv(output_root / "by_dataset.csv", by_dataset_rows)
    _write_csv(output_root / "per_sample.csv", per_sample_rows)
    _write_csv(output_root / "gradient_audit.csv", gradient_rows)
    with (output_root / "recommendation.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation_payload, handle, ensure_ascii=False, indent=2)

    print("TRAIN_GT_USED_FOR_AUDIT_ONLY=True")
    print("TRAIN_GT_USED_FOR_TRAINING=False")
    print(f"samples={len(source_rows)} diagnostic_only={diagnostic_only}")
    print(f"recommendation={recommendation}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
