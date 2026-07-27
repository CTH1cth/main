#!/usr/bin/env python3
"""Compare authoritative and A1 residual-only DABE/Clean caches directly.

This audit intentionally has no Bridge/PU-target dependency.  It compares the
upstream DABE maps, the final Clean-DP target and GT-only offline quality.
Soft-target quality is the primary GT audit because the training target is
continuous; thresholded quality is retained only as a support-set diagnostic.
A PASS means the cache identities and formulas are valid; it is not a training
performance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_cache import parse_bool  # noqa: E402
from common.utils import (  # noqa: E402
    find_gt_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


DEFAULT_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_BASELINE_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_BASELINE_SOURCE = (
    "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
)
DEFAULT_BASELINE_CLEAN = (
    "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8"
)
DEFAULT_CANDIDATE_SOURCE = (
    "../datasets/cache/dabe_pu_v11_a1_residual_only_pseudo_cache/dinov1-s8"
)
DEFAULT_CANDIDATE_CLEAN = (
    "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/dinov1-s8"
)
DEFAULT_DATA_ROOT = "/home/dell01/CTH/MY-baseline/datasets/COD"
DEFAULT_OUTPUT_ROOT = (
    "/home/dell01/CTH/MY-baseline/workdir/"
    "dabe_clean_a1_residual_only_full_audit"
)
EXPECTED_CONFIG_DIFFS = {
    "EXP_NAME",
    "DABE_BC_LAMBDA",
    "DABE_CLEAN_ABLATION",
    "DABE_CLEAN_SOURCE_ROOT",
    "DABE_CLEAN_ROOT",
}
SOURCE_MAP_KEYS = (
    "residual_pass1_37",
    "residual_37",
    "fg_score_37",
    "fg_core_37",
    "bg_core_37",
    "p_rw_37",
    "evidence_37",
    "p_base_37",
    "p_base_68",
)
CLEAN_MAP_KEYS = (
    "background_evidence_37",
    "target_dp_37",
    "target_dp_68",
)
GT_TARGETS = {
    "p_base": ("source", "p_base_68"),
    "target_dp": ("clean", "target_dp_68"),
}


def _prepare_output(path, overwrite):
    path = Path(path).resolve()
    if MAIN_ROOT == path or MAIN_ROOT in path.parents:
        raise RuntimeError("Audit output must remain outside the source repository.")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _public_config(cfg):
    return {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if name.isupper() and not name.startswith("__")
    }


def _config_diff(baseline_cfg, candidate_cfg):
    baseline = _public_config(baseline_cfg)
    candidate = _public_config(candidate_cfg)
    diff = {}
    for name in sorted(set(baseline) | set(candidate)):
        left = baseline.get(name, "<MISSING>")
        right = candidate.get(name, "<MISSING>")
        if left != right:
            diff[name] = {"baseline": repr(left), "candidate": repr(right)}
    unexpected = set(diff) - EXPECTED_CONFIG_DIFFS
    if unexpected:
        raise RuntimeError(f"Unexpected A1 config differences: {sorted(unexpected)}")
    if float(candidate.get("DABE_BC_LAMBDA", -1.0)) != 0.0:
        raise RuntimeError("A1 config must set DABE_BC_LAMBDA=0.0.")
    return diff


def _load_manifest(root):
    manifest = Path(root) / "manifest_train.jsonl"
    rows = read_jsonl(manifest)
    return rows, manifest_to_map(rows, manifest)


def _tensor(payload, key, shape):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Cache is missing tensor {key!r}.")
    value = value.detach().cpu().float()
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(f"{key} shape mismatch: {list(value.shape)} != {list(shape)}")
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} must be detached and finite.")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"{key} is outside [0,1].")
    return value.clamp(0.0, 1.0)


def _pair_metrics(baseline, candidate):
    baseline = baseline.double().reshape(-1)
    candidate = candidate.double().reshape(-1)
    baseline_centered = baseline - baseline.mean()
    candidate_centered = candidate - candidate.mean()
    denominator = baseline_centered.norm() * candidate_centered.norm()
    if float(denominator) <= 1e-15:
        pearson = float(torch.equal(baseline, candidate))
    else:
        pearson = float(
            torch.dot(baseline_centered, candidate_centered) / denominator
        )
    baseline_hard = baseline > 0.5
    candidate_hard = candidate > 0.5
    union = int((baseline_hard | candidate_hard).sum())
    hard_iou = (
        1.0
        if union == 0
        else int((baseline_hard & candidate_hard).sum()) / float(union)
    )
    return {
        "pearson": pearson,
        "mae": float((baseline - candidate).abs().mean()),
        "max_abs": float((baseline - candidate).abs().max()),
        "hard_iou": hard_iou,
        "hard_equal": float(torch.equal(baseline_hard, candidate_hard)),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": float(candidate.mean() - baseline.mean()),
    }


def _formula_audit(source, clean, expected_lambda):
    bc = _tensor(source, "bc_map_37", (1, 37, 37))
    residual = _tensor(source, "residual_37", (1, 37, 37))
    fg_score = _tensor(source, "fg_score_37", (1, 37, 37))
    gamma = float(source["params"].get("BC_SUPPRESS_GAMMA", 1.0))
    expected_score = residual * (1.0 - float(expected_lambda) * bc).clamp(0.0, 1.0).pow(gamma)
    score_error = float((fg_score - expected_score).abs().max())
    p_rw = _tensor(source, "p_rw_37", (1, 37, 37))
    evidence = _tensor(source, "evidence_37", (1, 37, 37))
    p_base_37 = _tensor(source, "p_base_37", (1, 37, 37))
    pbase_error = float((p_base_37 - p_rw * evidence).abs().max())
    clean_foreground_37 = _tensor(clean, "foreground_evidence_37", (1, 37, 37))
    clean_foreground_68 = _tensor(clean, "foreground_evidence_68", (1, 68, 68))
    source_p_base_68 = _tensor(source, "p_base_68", (1, 68, 68))
    foreground_error = max(
        float((clean_foreground_37 - p_base_37).abs().max()),
        float((clean_foreground_68 - source_p_base_68).abs().max()),
    )
    background = _tensor(clean, "background_evidence_37", (1, 37, 37))
    background_error = float((background - bc * (1.0 - residual)).abs().max())
    target_37 = _tensor(clean, "target_dp_37", (1, 37, 37))
    target_68 = _tensor(clean, "target_dp_68", (1, 68, 68))
    expected_37 = 0.5 + torch.maximum(p_base_37, background) * (p_base_37 - 0.5)
    background_68 = _tensor(clean, "background_evidence_68", (1, 68, 68))
    expected_68 = 0.5 + torch.maximum(source_p_base_68, background_68) * (
        source_p_base_68 - 0.5
    )
    target_error = max(
        float((target_37 - expected_37).abs().max()),
        float((target_68 - expected_68).abs().max()),
    )
    direction_equal = bool(
        torch.equal(target_37 > 0.5, p_base_37 > 0.5)
        and torch.equal(target_68 > 0.5, source_p_base_68 > 0.5)
    )
    result = {
        "fg_score_formula_max_abs": score_error,
        "p_base_formula_max_abs": pbase_error,
        "clean_foreground_source_max_abs": foreground_error,
        "background_formula_max_abs": background_error,
        "target_dp_formula_max_abs": target_error,
        "target_dp_direction_equal_p_base": direction_equal,
    }
    if max(score_error, pbase_error, foreground_error, background_error, target_error) > 1e-6:
        raise RuntimeError(f"Formula audit failed: {result}")
    if not direction_equal:
        raise RuntimeError("Clean-DP direction differs from p_base.")
    return result


def _prediction_counts(target, gt):
    prediction = F.interpolate(
        target.unsqueeze(0),
        size=tuple(gt.shape),
        mode="bilinear",
        align_corners=False,
    )[0, 0] > 0.5
    return {
        "tp": int((prediction & gt).sum()),
        "fp": int((prediction & ~gt).sum()),
        "fn": int((~prediction & gt).sum()),
        "tn": int((~prediction & ~gt).sum()),
    }


def _aligned_soft_maps(target, gt):
    """Return target/GT pairs at output and native supervision resolutions."""
    gt_float = gt.float()
    target_at_gt = F.interpolate(
        target.unsqueeze(0),
        size=tuple(gt.shape),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    gt_at_68 = F.interpolate(
        gt_float.unsqueeze(0).unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return {
        "gt_resolution_target_bilinear": (target_at_gt, gt_float),
        "native_68_gt_bilinear": (target[0], gt_at_68),
    }


def _soft_sufficient_stats(prediction, gt):
    prediction = prediction.detach().cpu().double().clamp(0.0, 1.0)
    gt = gt.detach().cpu().double().clamp(0.0, 1.0)
    if prediction.shape != gt.shape:
        raise RuntimeError(
            f"Soft GT shape mismatch: {list(prediction.shape)} != {list(gt.shape)}"
        )
    if not bool(torch.isfinite(prediction).all().item()) or not bool(
        torch.isfinite(gt).all().item()
    ):
        raise RuntimeError("Soft GT audit inputs must be finite.")
    error = prediction - gt
    abs_error = error.abs()
    eps = 1e-6
    probability = prediction.clamp(eps, 1.0 - eps)
    log_loss = -(
        gt * probability.log() + (1.0 - gt) * torch.log1p(-probability)
    )
    entropy = -(
        probability * probability.log()
        + (1.0 - probability) * torch.log1p(-probability)
    )
    return {
        "pixels": float(prediction.numel()),
        "abs_error_sum": float(abs_error.sum()),
        "sq_error_sum": float(error.square().sum()),
        "bce_sum": float(log_loss.sum()),
        "prediction_sum": float(prediction.sum()),
        "gt_sum": float(gt.sum()),
        "intersection_sum": float((prediction * gt).sum()),
        "foreground_abs_error_sum": float((gt * abs_error).sum()),
        "background_abs_error_sum": float(((1.0 - gt) * abs_error).sum()),
        "foreground_mass": float(gt.sum()),
        "background_mass": float((1.0 - gt).sum()),
        "entropy_sum": float(entropy.sum()),
        "confidence_sum": float((2.0 * (prediction - 0.5).abs()).sum()),
        "neutral_0.4_0.6_count": float(
            ((prediction >= 0.4) & (prediction <= 0.6)).sum()
        ),
    }


def _add_soft_stats(accumulator, stats):
    for key, value in stats.items():
        accumulator[key] += float(value)


def _soft_quality_metrics(stats):
    eps = 1e-12
    pixels = float(stats["pixels"])
    prediction_sum = float(stats["prediction_sum"])
    gt_sum = float(stats["gt_sum"])
    intersection = float(stats["intersection_sum"])
    precision = intersection / (prediction_sum + eps)
    recall = intersection / (gt_sum + eps)
    beta2 = 0.3
    brier = float(stats["sq_error_sum"]) / (pixels + eps)
    return {
        "soft_mae": float(stats["abs_error_sum"]) / (pixels + eps),
        "soft_brier_mse": brier,
        "soft_rmse": brier ** 0.5,
        "soft_bce": float(stats["bce_sum"]) / (pixels + eps),
        "soft_precision": precision,
        "soft_recall": recall,
        "soft_f1_dice": 2.0 * intersection
        / (prediction_sum + gt_sum + eps),
        "soft_f_beta_beta2_0.3": (1.0 + beta2) * precision * recall
        / (beta2 * precision + recall + eps),
        "soft_iou": intersection
        / (prediction_sum + gt_sum - intersection + eps),
        "foreground_weighted_mae": float(stats["foreground_abs_error_sum"])
        / (float(stats["foreground_mass"]) + eps),
        "background_weighted_mae": float(stats["background_abs_error_sum"])
        / (float(stats["background_mass"]) + eps),
        "prediction_mean": prediction_sum / (pixels + eps),
        "gt_mean": gt_sum / (pixels + eps),
        "prediction_to_gt_area_ratio": prediction_sum / (gt_sum + eps),
        "target_entropy": float(stats["entropy_sum"]) / (pixels + eps),
        "target_confidence": float(stats["confidence_sum"]) / (pixels + eps),
        "neutral_ratio_0.4_0.6": float(stats["neutral_0.4_0.6_count"])
        / (pixels + eps),
    }


def _add_counts(accumulator, counts):
    for key, value in counts.items():
        accumulator[key] += int(value)


def _quality_metrics(counts):
    tp, fp, fn, tn = (float(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta2 = 0.3
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + eps),
        "f_beta_beta2_0.3": (1.0 + beta2) * precision * recall
        / (beta2 * precision + recall + eps),
        "iou": tp / (tp + fp + fn + eps),
        "false_positive_ratio": fp / (fp + tn + eps),
        "false_negative_ratio": fn / (fn + tp + eps),
        "binary_area": (tp + fp) / (tp + fp + fn + tn + eps),
    }


def _summarize_pair_rows(rows, scope_name):
    grouped = defaultdict(list)
    for row in rows:
        if scope_name != "ALL" and row["dataset"] != scope_name:
            continue
        grouped[row["map"]].append(row)
    output = []
    for map_name, values in sorted(grouped.items()):
        result = {"scope": scope_name, "map": map_name, "samples": len(values)}
        for metric in (
            "pearson",
            "mae",
            "max_abs",
            "hard_iou",
            "hard_equal",
            "baseline_mean",
            "candidate_mean",
            "mean_delta",
        ):
            array = np.asarray([row[metric] for row in values], dtype=np.float64)
            result[f"{metric}_mean"] = float(array.mean())
            result[f"{metric}_p05"] = float(np.quantile(array, 0.05))
            result[f"{metric}_p50"] = float(np.quantile(array, 0.50))
            result[f"{metric}_p95"] = float(np.quantile(array, 0.95))
        output.append(result)
    return output


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--baseline-source-root", default=DEFAULT_BASELINE_SOURCE)
    parser.add_argument("--baseline-clean-root", default=DEFAULT_BASELINE_CLEAN)
    parser.add_argument("--candidate-source-root", default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--candidate-clean-root", default=DEFAULT_CANDIDATE_CLEAN)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer.")

    output_root = _prepare_output(args.output_root, args.overwrite)
    config_diff = _config_diff(
        load_config(args.baseline_config), load_config(args.config)
    )
    baseline_rows, baseline_source_map = _load_manifest(args.baseline_source_root)
    _, baseline_clean_map = _load_manifest(args.baseline_clean_root)
    candidate_rows, candidate_source_map = _load_manifest(args.candidate_source_root)
    _, candidate_clean_map = _load_manifest(args.candidate_clean_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    dataset_set = set(datasets)
    selected = [
        row for row in candidate_rows if str(row.get("dataset")) in dataset_set
    ]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    if not selected:
        raise RuntimeError("No A1 cache rows selected.")

    pair_rows = []
    formula_maxima = defaultdict(float)
    gt_counts = defaultdict(lambda: defaultdict(int))
    soft_gt_stats = defaultdict(lambda: defaultdict(float))
    soft_gt_per_sample_rows = []
    source_param_diff_reference = None
    selected_keys = {(str(row["dataset"]), str(row["stem"])) for row in selected}
    baseline_available = {
        (str(row["dataset"]), str(row["stem"])) for row in baseline_rows
    }
    missing = selected_keys - baseline_available
    if missing:
        raise RuntimeError(f"Baseline source misses A1 keys: {sorted(missing)[:5]}")

    for row in selected:
        dataset, stem = str(row["dataset"]), str(row["stem"])
        key = (dataset, stem)
        for name, mapping in (
            ("baseline source", baseline_source_map),
            ("baseline clean", baseline_clean_map),
            ("candidate source", candidate_source_map),
            ("candidate clean", candidate_clean_map),
        ):
            if key not in mapping:
                raise RuntimeError(f"{name} misses {key}.")
        baseline_source = torch_load(
            baseline_source_map[key]["cache_path"], map_location="cpu"
        )
        baseline_clean = torch_load(
            baseline_clean_map[key]["cache_path"], map_location="cpu"
        )
        candidate_source = torch_load(
            candidate_source_map[key]["cache_path"], map_location="cpu"
        )
        candidate_clean = torch_load(
            candidate_clean_map[key]["cache_path"], map_location="cpu"
        )
        baseline_params = dict(baseline_source.get("params", {}))
        candidate_params = dict(candidate_source.get("params", {}))
        param_diff = {
            name: {
                "baseline": baseline_params.get(name),
                "candidate": candidate_params.get(name),
            }
            for name in sorted(set(baseline_params) | set(candidate_params))
            if baseline_params.get(name) != candidate_params.get(name)
        }
        if set(param_diff) != {"BC_LAMBDA"}:
            raise RuntimeError(f"Source params differ beyond A1 for {key}: {param_diff}")
        if float(param_diff["BC_LAMBDA"]["baseline"]) != 0.5 or float(
            param_diff["BC_LAMBDA"]["candidate"]
        ) != 0.0:
            raise RuntimeError(f"Unexpected BC_LAMBDA transition for {key}: {param_diff}")
        if source_param_diff_reference is None:
            source_param_diff_reference = param_diff
        elif param_diff != source_param_diff_reference:
            raise RuntimeError("A1 source parameter diff varies across samples.")

        for variant, source, clean, expected_lambda in (
            ("baseline", baseline_source, baseline_clean, 0.5),
            ("candidate", candidate_source, candidate_clean, 0.0),
        ):
            formula = _formula_audit(source, clean, expected_lambda)
            for name, value in formula.items():
                if isinstance(value, bool):
                    if not value:
                        raise RuntimeError(f"{variant} formula flag failed: {name}")
                else:
                    formula_maxima[f"{variant}.{name}"] = max(
                        formula_maxima[f"{variant}.{name}"], float(value)
                    )

        payload_pairs = {
            **{
                name: (
                    _tensor(baseline_source, name, (1, 68, 68) if name.endswith("_68") else (1, 37, 37)),
                    _tensor(candidate_source, name, (1, 68, 68) if name.endswith("_68") else (1, 37, 37)),
                )
                for name in SOURCE_MAP_KEYS
            },
            **{
                name: (
                    _tensor(baseline_clean, name, (1, 68, 68) if name.endswith("_68") else (1, 37, 37)),
                    _tensor(candidate_clean, name, (1, 68, 68) if name.endswith("_68") else (1, 37, 37)),
                )
                for name in CLEAN_MAP_KEYS
            },
        }
        for map_name, (baseline_tensor, candidate_tensor) in payload_pairs.items():
            pair_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "map": map_name,
                    **_pair_metrics(baseline_tensor, candidate_tensor),
                }
            )

        gt_path = find_gt_path(args.data_root, dataset, stem)
        gt = torch.from_numpy(np.asarray(Image.open(gt_path).convert("L")) > 127)
        for target_name, (payload_name, tensor_key) in GT_TARGETS.items():
            target_tensors = {}
            for variant, source, clean in (
                ("baseline", baseline_source, baseline_clean),
                ("candidate", candidate_source, candidate_clean),
            ):
                payload = source if payload_name == "source" else clean
                target_tensor = _tensor(payload, tensor_key, (1, 68, 68))
                target_tensors[variant] = target_tensor
                counts = _prediction_counts(target_tensor, gt)
                _add_counts(gt_counts[("ALL", target_name, variant)], counts)
                _add_counts(gt_counts[(dataset, target_name, variant)], counts)
            aligned = {
                variant: _aligned_soft_maps(target_tensor, gt)
                for variant, target_tensor in target_tensors.items()
            }
            for resolution in aligned["baseline"]:
                per_variant_metrics = {}
                for variant in ("baseline", "candidate"):
                    prediction, aligned_gt = aligned[variant][resolution]
                    stats = _soft_sufficient_stats(prediction, aligned_gt)
                    _add_soft_stats(
                        soft_gt_stats[("ALL", target_name, resolution, variant)],
                        stats,
                    )
                    _add_soft_stats(
                        soft_gt_stats[(dataset, target_name, resolution, variant)],
                        stats,
                    )
                    per_variant_metrics[variant] = _soft_quality_metrics(stats)
                    soft_gt_per_sample_rows.append(
                        {
                            "dataset": dataset,
                            "stem": stem,
                            "target": target_name,
                            "resolution": resolution,
                            "variant": variant,
                            **per_variant_metrics[variant],
                        }
                    )
                delta = {
                    name: per_variant_metrics["candidate"][name]
                    - per_variant_metrics["baseline"][name]
                    for name in per_variant_metrics["baseline"]
                }
                soft_gt_per_sample_rows.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "target": target_name,
                        "resolution": resolution,
                        "variant": "delta_candidate_minus_baseline",
                        **delta,
                    }
                )

    summary_rows = _summarize_pair_rows(pair_rows, "ALL")
    for dataset in datasets:
        summary_rows.extend(_summarize_pair_rows(pair_rows, dataset))
    quality_rows = []
    quality_payload = {}
    for scope in ["ALL", *datasets]:
        quality_payload[scope] = {}
        for target_name in GT_TARGETS:
            baseline_metrics = _quality_metrics(
                gt_counts[(scope, target_name, "baseline")]
            )
            candidate_metrics = _quality_metrics(
                gt_counts[(scope, target_name, "candidate")]
            )
            delta = {
                name: candidate_metrics[name] - baseline_metrics[name]
                for name in baseline_metrics
            }
            quality_payload[scope][target_name] = {
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "delta_candidate_minus_baseline": delta,
            }
            quality_rows.extend(
                [
                    {"scope": scope, "target": target_name, "variant": "baseline", **baseline_metrics},
                    {"scope": scope, "target": target_name, "variant": "candidate", **candidate_metrics},
                    {"scope": scope, "target": target_name, "variant": "delta", **delta},
                ]
            )

    soft_quality_rows = []
    soft_quality_payload = {}
    soft_resolutions = (
        "gt_resolution_target_bilinear",
        "native_68_gt_bilinear",
    )
    for scope in ["ALL", *datasets]:
        soft_quality_payload[scope] = {}
        for target_name in GT_TARGETS:
            soft_quality_payload[scope][target_name] = {}
            for resolution in soft_resolutions:
                baseline_metrics = _soft_quality_metrics(
                    soft_gt_stats[(scope, target_name, resolution, "baseline")]
                )
                candidate_metrics = _soft_quality_metrics(
                    soft_gt_stats[(scope, target_name, resolution, "candidate")]
                )
                delta = {
                    name: candidate_metrics[name] - baseline_metrics[name]
                    for name in baseline_metrics
                }
                soft_quality_payload[scope][target_name][resolution] = {
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "delta_candidate_minus_baseline": delta,
                }
                soft_quality_rows.extend(
                    [
                        {
                            "scope": scope,
                            "target": target_name,
                            "resolution": resolution,
                            "variant": "baseline",
                            **baseline_metrics,
                        },
                        {
                            "scope": scope,
                            "target": target_name,
                            "resolution": resolution,
                            "variant": "candidate",
                            **candidate_metrics,
                        },
                        {
                            "scope": scope,
                            "target": target_name,
                            "resolution": resolution,
                            "variant": "delta_candidate_minus_baseline",
                            **delta,
                        },
                    ]
                )

    _write_csv(output_root / "overall_and_by_dataset_difference.csv", summary_rows)
    _write_csv(output_root / "per_sample_difference.csv", pair_rows)
    _write_csv(output_root / "gt_quality_binary_diagnostic.csv", quality_rows)
    _write_csv(output_root / "gt_quality_soft_comparison.csv", soft_quality_rows)
    _write_csv(output_root / "gt_quality_soft_per_sample.csv", soft_gt_per_sample_rows)
    result = {
        "status": "PASS",
        "status_meaning": "cache/config/formula integrity passed; not a training-performance claim",
        "diagnostic_only": args.max_samples != -1,
        "samples": len(selected),
        "datasets": datasets,
        "config_diff": config_diff,
        "source_param_diff": source_param_diff_reference,
        "formula_max_abs": dict(sorted(formula_maxima.items())),
        "map_difference": {
            row["map"]: row
            for row in summary_rows
            if row["scope"] == "ALL"
        },
        "soft_gt_quality_primary": soft_quality_payload,
        "binary_gt_quality_diagnostic": quality_payload,
        "gt_quality": quality_payload,
        "safety": {
            "training_run": False,
            "model_created": False,
            "backward_run": False,
            "optimizer_used": False,
            "teacher_used": False,
            "validation_or_eval_entrypoint_used": False,
            "gt_used_for_offline_audit_only": True,
        },
    }
    with (output_root / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "status": result["status"],
        "diagnostic_only": result["diagnostic_only"],
        "samples": result["samples"],
        "p_base_68": result["map_difference"]["p_base_68"],
        "target_dp_68": result["map_difference"]["target_dp_68"],
        "soft_gt_quality_primary": {
            scope: values["target_dp"]
            for scope, values in result["soft_gt_quality_primary"].items()
        },
        "binary_gt_quality_diagnostic": result["binary_gt_quality_diagnostic"],
        "output_root": str(output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
