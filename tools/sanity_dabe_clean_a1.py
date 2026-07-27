#!/usr/bin/env python3
"""Bounded real-sample audit for the A1 residual-only DABE-Clean ablation.

The candidate is regenerated in memory from the authoritative DINO feature and
RGB sources.  GT is read only after pseudo-label construction for diagnostic
metrics; no cache payload, model, optimizer, teacher, validation or evaluation
entry point is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.dabe_clean import build_background_evidence, build_clean_targets  # noqa: E402
from common.dabe_pseudo import generate_dabe_pseudo  # noqa: E402
from common.utils import (  # noqa: E402
    feature_manifest_path,
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
EXPECTED_CONFIG_DIFFS = {
    "EXP_NAME",
    "DABE_BC_LAMBDA",
    "DABE_CLEAN_ABLATION",
    "DABE_CLEAN_SOURCE_ROOT",
    "DABE_CLEAN_ROOT",
}
MAP_KEYS = (
    "residual_pass1_37",
    "residual_37",
    "fg_score_37",
    "fg_core_37",
    "bg_core_37",
    "p_rw_37",
    "evidence_37",
    "p_base_37",
)


def _prepare_output(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _public_config(cfg) -> dict:
    return {
        name: getattr(cfg, name)
        for name in dir(cfg)
        if name.isupper() and not name.startswith("__")
    }


def _config_diff(baseline_cfg, candidate_cfg) -> dict:
    baseline = _public_config(baseline_cfg)
    candidate = _public_config(candidate_cfg)
    names = sorted(set(baseline) | set(candidate))
    diff = {}
    for name in names:
        left = baseline.get(name, "<MISSING>")
        right = candidate.get(name, "<MISSING>")
        if left != right:
            diff[name] = {"baseline": repr(left), "candidate": repr(right)}
    unexpected = set(diff) - EXPECTED_CONFIG_DIFFS
    if unexpected:
        raise RuntimeError(f"A1 has unexpected effective config diffs: {sorted(unexpected)}")
    if float(candidate.get("DABE_BC_LAMBDA", -1.0)) != 0.0:
        raise RuntimeError("A1 requires DABE_BC_LAMBDA=0.0.")
    return diff


def _tensor(payload, key, expected_shape=None):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Missing tensor {key!r}.")
    value = value.detach().cpu().float()
    if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{key} shape mismatch: {list(value.shape)} != {list(expected_shape)}"
        )
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} must be detached and finite.")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"{key} is outside [0,1].")
    return value.clamp(0.0, 1.0)


def _clean_payload(source):
    foreground_37 = _tensor(source, "p_base_37", (1, 37, 37))
    foreground_68 = _tensor(source, "p_base_68", (1, 68, 68))
    background_37 = build_background_evidence(
        _tensor(source, "bc_map_37", (1, 37, 37)),
        _tensor(source, "residual_37", (1, 37, 37)),
    )
    background_68 = F.interpolate(
        background_37.unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).detach()
    clean_37 = build_clean_targets(foreground_37, background_37)
    clean_68 = build_clean_targets(foreground_68, background_68)
    return {
        "foreground_evidence_37": foreground_37,
        "foreground_evidence_68": foreground_68,
        "background_evidence_37": background_37,
        "background_evidence_68": background_68,
        "target_dp_37": clean_37["target_dp"],
        "target_dp_68": clean_68["target_dp"],
    }


def _pair_metrics(left, right):
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    pearson = (
        float(torch.dot(left_centered, right_centered) / denominator)
        if float(denominator) > 1e-15
        else float(torch.equal(left, right))
    )
    left_hard = left > 0.5
    right_hard = right > 0.5
    union = int((left_hard | right_hard).sum())
    return {
        "pearson": pearson,
        "mae": float((left - right).abs().mean()),
        "max_abs": float((left - right).abs().max()),
        "hard_iou": int((left_hard & right_hard).sum()) / max(union, 1),
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
    }


def _counts(target_68, gt_path):
    gt = torch.from_numpy(np.asarray(Image.open(gt_path).convert("L")) > 127)
    prediction = F.interpolate(
        target_68.unsqueeze(0),
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


def _metric_row(counts):
    tp, fp, fn, tn = (float(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + eps),
        "iou": tp / (tp + fp + fn + eps),
        "fp_bg": fp / (fp + tn + eps),
        "fn_fg": fn / (fn + tp + eps),
        "area": (tp + fp) / (tp + fp + fn + tn + eps),
    }


def _add_counts(accumulator, counts):
    for key, value in counts.items():
        accumulator[key] += int(value)


def _save_visual(path, image_path, gt_path, baseline, candidate, baseline_clean, candidate_clean):
    rgb = Image.open(image_path).convert("RGB")
    gt = Image.open(gt_path).convert("L")
    panels = [
        (rgb, "RGB", None),
        (gt, "GT (audit only)", "gray"),
        (_tensor(baseline, "residual_37")[0], "Baseline final residual", "gray"),
        (_tensor(candidate, "residual_37")[0], "A1 final residual", "gray"),
        (_tensor(baseline, "fg_score_37")[0], "Baseline R×(1-0.5BC)", "gray"),
        (_tensor(candidate, "fg_score_37")[0], "A1 fg score = R", "gray"),
        (_tensor(baseline, "p_base_37")[0], "Baseline p_base", "viridis"),
        (_tensor(candidate, "p_base_37")[0], "A1 p_base", "viridis"),
        (_tensor(baseline_clean, "target_dp_68")[0], "Baseline Clean-DP", "viridis"),
        (_tensor(candidate_clean, "target_dp_68")[0], "A1 Clean-DP", "viridis"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    for axis, (value, title, cmap) in zip(axes.flat, panels):
        axis.imshow(value, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--baseline-source-root", default=DEFAULT_BASELINE_SOURCE)
    parser.add_argument("--baseline-clean-root", default=DEFAULT_BASELINE_CLEAN)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--samples-per-dataset", type=int, default=8)
    parser.add_argument("--visuals-per-dataset", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.samples_per_dataset <= 25:
        raise ValueError("--samples-per-dataset must be in [1,25].")
    if not 0 <= args.visuals_per_dataset <= args.samples_per_dataset:
        raise ValueError("--visuals-per-dataset must be in [0,samples-per-dataset].")
    output_root = _prepare_output(Path(args.output_root).resolve(), args.overwrite)
    candidate_cfg = load_config(args.config)
    baseline_cfg = load_config(args.baseline_config)
    config_diff = _config_diff(baseline_cfg, candidate_cfg)
    cfg_params = _params_from_cfg(candidate_cfg)
    if set(cfg_params) != {"BC_LAMBDA"} or float(cfg_params["BC_LAMBDA"]) != 0.0:
        raise RuntimeError(f"Unexpected config-side DABE params: {cfg_params}")

    source_root = Path(args.baseline_source_root)
    clean_root = Path(args.baseline_clean_root)
    source_rows = read_jsonl(source_root / "manifest_train.jsonl")
    source_map = manifest_to_map(source_rows, source_root / "manifest_train.jsonl")
    clean_map = manifest_to_map(
        read_jsonl(clean_root / "manifest_train.jsonl"),
        clean_root / "manifest_train.jsonl",
    )
    feature_manifest = feature_manifest_path(candidate_cfg, "train")
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    selected = []
    for dataset in datasets:
        rows = [row for row in source_rows if str(row.get("dataset")) == dataset]
        if len(rows) < args.samples_per_dataset:
            raise RuntimeError(f"Not enough rows for {dataset}: {len(rows)}")
        selected.extend(rows[: args.samples_per_dataset])

    pair_acc = defaultdict(list)
    count_acc = defaultdict(lambda: defaultdict(int))
    sample_rows = []
    visual_count = defaultdict(int)
    baseline_params_reference = None

    with torch.no_grad():
        for row in selected:
            dataset, stem = str(row["dataset"]), str(row["stem"])
            key = (dataset, stem)
            baseline = torch_load(source_map[key]["cache_path"], map_location="cpu")
            baseline_clean = torch_load(clean_map[key]["cache_path"], map_location="cpu")
            feature_payload = torch_load(feature_map[key]["cache_path"], map_location="cpu")
            feature = feature_payload.get("tensor")
            if not torch.is_tensor(feature) or tuple(feature.shape) != (384, 37, 37):
                raise RuntimeError(f"Invalid feature cache for {dataset}/{stem}.")
            baseline_params = dict(baseline.get("params", {}))
            if baseline_params_reference is None:
                baseline_params_reference = baseline_params
            elif baseline_params != baseline_params_reference:
                raise RuntimeError("Baseline DABE params vary across selected samples.")
            candidate_params = dict(baseline_params)
            candidate_params.update(cfg_params)
            candidate_params["VERSION"] = "pu_v11"
            changed_params = {
                name for name in set(baseline_params) | set(candidate_params)
                if baseline_params.get(name) != candidate_params.get(name)
            }
            if changed_params != {"BC_LAMBDA"}:
                raise RuntimeError(f"A1 source params differ by {sorted(changed_params)}")

            candidate = generate_dabe_pseudo(
                feature.detach().cpu().float(),
                baseline["image_path"],
                params=candidate_params,
                augs=["identity"],
            )
            candidate_clean = _clean_payload(candidate)

            baseline_formula_error = float(
                (
                    _tensor(baseline, "fg_score_37")
                    - _tensor(baseline, "residual_37")
                    * (1.0 - 0.5 * _tensor(baseline, "bc_map_37"))
                ).abs().max()
            )
            candidate_formula_error = float(
                (
                    _tensor(candidate, "fg_score_37")
                    - _tensor(candidate, "residual_37")
                ).abs().max()
            )
            candidate_pbase_error = float(
                (
                    _tensor(candidate, "p_base_37")
                    - _tensor(candidate, "p_rw_37")
                    * _tensor(candidate, "evidence_37")
                ).abs().max()
            )
            if max(baseline_formula_error, candidate_formula_error, candidate_pbase_error) > 1e-6:
                raise RuntimeError(
                    f"Formula regression for {dataset}/{stem}: "
                    f"{baseline_formula_error}/{candidate_formula_error}/{candidate_pbase_error}"
                )
            if not torch.equal(
                candidate_clean["target_dp_68"] > 0.5,
                candidate_clean["foreground_evidence_68"] > 0.5,
            ):
                raise RuntimeError(f"Clean-DP direction changed for {dataset}/{stem}.")

            comparisons = {}
            for map_key in MAP_KEYS:
                metrics = _pair_metrics(_tensor(baseline, map_key), _tensor(candidate, map_key))
                comparisons[map_key] = metrics
                for metric, value in metrics.items():
                    pair_acc[f"{map_key}.{metric}"].append(value)
            for clean_key in ("background_evidence_37", "target_dp_37", "target_dp_68"):
                metrics = _pair_metrics(
                    _tensor(baseline_clean, clean_key),
                    _tensor(candidate_clean, clean_key),
                )
                comparisons[clean_key] = metrics
                for metric, value in metrics.items():
                    pair_acc[f"{clean_key}.{metric}"].append(value)

            gt_path = baseline["gt_path"]
            for variant, target in (
                ("baseline", _tensor(baseline_clean, "target_dp_68")),
                ("candidate", candidate_clean["target_dp_68"]),
            ):
                counts = _counts(target, gt_path)
                _add_counts(count_acc[("ALL", variant)], counts)
                _add_counts(count_acc[(dataset, variant)], counts)

            sample_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "baseline_formula_max_abs": baseline_formula_error,
                    "candidate_formula_max_abs": candidate_formula_error,
                    "candidate_pbase_formula_max_abs": candidate_pbase_error,
                    "residual_37_pearson": comparisons["residual_37"]["pearson"],
                    "p_rw_37_pearson": comparisons["p_rw_37"]["pearson"],
                    "p_base_37_pearson": comparisons["p_base_37"]["pearson"],
                    "p_base_37_hard_iou": comparisons["p_base_37"]["hard_iou"],
                    "target_dp_68_pearson": comparisons["target_dp_68"]["pearson"],
                    "target_dp_68_hard_iou": comparisons["target_dp_68"]["hard_iou"],
                }
            )
            if visual_count[dataset] < args.visuals_per_dataset:
                _save_visual(
                    output_root / f"{dataset}_{stem}_a1_compare.png",
                    baseline["image_path"],
                    gt_path,
                    baseline,
                    candidate,
                    baseline_clean,
                    candidate_clean,
                )
                visual_count[dataset] += 1

    comparison_summary = {}
    for name, values in sorted(pair_acc.items()):
        array = np.asarray(values, dtype=np.float64)
        comparison_summary[name] = {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "p50": float(np.quantile(array, 0.50)),
            "max": float(array.max()),
        }
    gt_metrics = {}
    for (dataset, variant), counts in sorted(count_acc.items()):
        gt_metrics.setdefault(dataset, {})[variant] = _metric_row(counts)
    for dataset in gt_metrics:
        baseline_metrics = gt_metrics[dataset]["baseline"]
        candidate_metrics = gt_metrics[dataset]["candidate"]
        gt_metrics[dataset]["delta_candidate_minus_baseline"] = {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
        }

    result = {
        "status": "PASS",
        "diagnostic_only": True,
        "samples": len(selected),
        "samples_per_dataset": args.samples_per_dataset,
        "datasets": datasets,
        "candidate_config": str(Path(args.config)),
        "baseline_config": str(Path(args.baseline_config)),
        "config_diff": config_diff,
        "source_param_diff": {
            "BC_LAMBDA": {
                "baseline": float(baseline_params_reference["BC_LAMBDA"]),
                "candidate": 0.0,
            }
        },
        "comparison": comparison_summary,
        "gt_metrics": gt_metrics,
        "safety": {
            "model_created": False,
            "training_run": False,
            "backward_run": False,
            "optimizer_used": False,
            "teacher_used": False,
            "validation_or_eval_entrypoint_used": False,
            "candidate_cache_written": False,
        },
    }
    with (output_root / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (output_root / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    print(json.dumps({
        "status": result["status"],
        "diagnostic_only": True,
        "samples": result["samples"],
        "source_param_diff": result["source_param_diff"],
        "p_base_37": {
            "pearson": comparison_summary["p_base_37.pearson"]["mean"],
            "hard_iou": comparison_summary["p_base_37.hard_iou"]["mean"],
            "mae": comparison_summary["p_base_37.mae"]["mean"],
        },
        "target_dp_68": {
            "pearson": comparison_summary["target_dp_68.pearson"]["mean"],
            "hard_iou": comparison_summary["target_dp_68.hard_iou"]["mean"],
            "mae": comparison_summary["target_dp_68.mae"]["mean"],
        },
        "gt_metrics": gt_metrics,
        "output_root": str(output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
