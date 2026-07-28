#!/usr/bin/env python3
"""Audit DABE-Clean v3 formulas, distributions, GT quality, and visuals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_offline import (  # noqa: E402
    DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT,
    DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
    offline_mode_target,
)
from common.dabe_clean_cache import parse_bool  # noqa: E402
from common.utils import read_jsonl, torch_load  # noqa: E402


MODE_ORDER = (
    "baseline_dp",
    "complementary_fg",
    "complementary_fg_conflict",
    "direct_consolidated_fg",
)
DEFAULT_ROOTS = {
    "baseline_dp": "../datasets/cache/dabe_clean_v3_v0_baseline_dp_pseudo_cache/dinov1-s8",
    "complementary_fg": "../datasets/cache/dabe_clean_v3_v1_compfg_pseudo_cache/dinov1-s8",
    "complementary_fg_conflict": "../datasets/cache/dabe_clean_v3_v2_compfg_conflict_pseudo_cache/dinov1-s8",
    "direct_consolidated_fg": "../datasets/cache/dabe_clean_v3_v3_direct_fg_pseudo_cache/dinov1-s8",
}
FORMULA_TOLERANCE = 1e-6


def _tensor(payload, key, shape, path):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Missing tensor {key}: {path}")
    value = value.detach().cpu().float()
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(
            f"{key} shape mismatch: {list(value.shape)} != {list(shape)} | {path}"
        )
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} must be detached and finite: {path}")
    if float(value.min()) < -FORMULA_TOLERANCE or float(value.max()) > 1 + FORMULA_TOLERANCE:
        raise RuntimeError(f"{key} is outside [0,1]: {path}")
    return value.clamp(0.0, 1.0)


def _resize(value, size):
    return F.interpolate(
        value.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)


def _max_error(actual, expected):
    return float((actual.float() - expected.float()).abs().max().item())


def _load_and_audit(row, expected_mode):
    path = Path(row["cache_path"])
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Payload must be dict: {path}")
    identity = f"{row['dataset']}/{row['stem']}"
    if payload.get("dataset") != row["dataset"] or payload.get("stem") != row["stem"]:
        raise RuntimeError(f"Payload identity mismatch: {identity} | {path}")
    for field, expected in (
        ("version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
        ("payload_version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
        ("offline_mode", expected_mode),
        ("formula_fingerprint", DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT),
    ):
        if str(payload.get(field)) != expected:
            raise RuntimeError(
                f"{identity}: {field}={payload.get(field)!r} != {expected!r}"
            )
    for flag in (
        "training_gt_read",
        "teacher_checkpoint_read",
        "student_checkpoint_read",
        "teacher_prediction_read",
        "epoch_dependent",
        "history_read",
        "static_weight_map_written",
        "routing_map_written",
    ):
        if bool(payload.get(flag, True)):
            raise RuntimeError(f"{identity}: forbidden {flag}=True")
    forbidden_fields = {
        "gt",
        "gt_path",
        "static_weight_map",
        "weight_map",
        "recovery_map",
        "routing_map",
        "teacher_region_map",
        "history",
        "epoch",
    }.intersection(payload)
    if forbidden_fields:
        raise RuntimeError(
            f"{identity}: payload leaked forbidden fields {sorted(forbidden_fields)}"
        )

    shape37 = (1, 37, 37)
    residual = _tensor(payload, "residual_37", shape37, path)
    bc = _tensor(payload, "bc_map_37", shape37, path)
    p_rw = _tensor(payload, "p_rw_37", shape37, path)
    evidence = _tensor(payload, "evidence_37", shape37, path)
    primary = _tensor(payload, "primary_fg_37", shape37, path)
    background = _tensor(payload, "background_evidence_37", shape37, path)
    semantic = _tensor(payload, "semantic_fg_tendency_37", shape37, path)
    latent = _tensor(payload, "latent_support_37", shape37, path)
    complementary = _tensor(payload, "complementary_fg_37", shape37, path)
    increment = _tensor(payload, "complementary_increment_37", shape37, path)
    consolidated = _tensor(payload, "consolidated_fg_37", shape37, path)
    conflict = _tensor(payload, "conflict_37", shape37, path)
    commitment = _tensor(payload, "commitment_37", shape37, path)
    target37 = _tensor(payload, "target_offline_37", shape37, path)
    target68 = _tensor(payload, "target_offline_68", (1, 68, 68), path)

    primary_ref = (p_rw * evidence).clamp(0.0, 1.0)
    background_ref = (bc * (1.0 - residual)).clamp(0.0, 1.0)
    latent_ref = torch.relu(p_rw - primary_ref).clamp(0.0, 1.0)
    complementary_ref = (
        latent_ref * (1.0 - background_ref) * semantic
    ).clamp(0.0, 1.0).pow(1.0 / 3.0)
    mode37 = offline_mode_target(
        primary_ref, background_ref, complementary_ref, expected_mode
    )
    mode68 = offline_mode_target(
        _resize(primary_ref, (68, 68)),
        _resize(background_ref, (68, 68)),
        _resize(complementary_ref, (68, 68)),
        expected_mode,
    )
    errors = {
        "primary_fg_37": _max_error(primary, primary_ref),
        "background_evidence_37": _max_error(background, background_ref),
        "latent_support_37": _max_error(latent, latent_ref),
        "complementary_fg_37": _max_error(complementary, complementary_ref),
        "consolidated_fg_37": _max_error(
            consolidated, mode37["consolidated_fg"]
        ),
        "complementary_increment_37": _max_error(
            increment, mode37["consolidated_fg"] - primary_ref
        ),
        "conflict_37": _max_error(conflict, mode37["conflict"]),
        "commitment_37": _max_error(commitment, mode37["commitment"]),
        "target_offline_37": _max_error(target37, mode37["target"]),
        "target_offline_68": _max_error(target68, mode68["target"]),
    }
    failed = {name: value for name, value in errors.items() if value > FORMULA_TOLERANCE}
    if expected_mode == "baseline_dp":
        baseline_error = float(payload.get("baseline_reference_max_error", float("inf")))
        errors["baseline_clean_dp_68"] = baseline_error
        if baseline_error > FORMULA_TOLERANCE:
            failed["baseline_clean_dp_68"] = baseline_error
    if failed:
        raise RuntimeError(f"Formula audit failed for {identity}: {failed}")
    if not bool((consolidated + FORMULA_TOLERANCE >= primary).all().item()):
        raise RuntimeError(f"Noisy-OR monotonicity failed for {identity}")
    return payload, errors


def _manifest_map(root, expected_mode, datasets, max_samples):
    manifest = Path(root) / "manifest_train.jsonl"
    rows = read_jsonl(manifest)
    allowed = {item.strip() for item in datasets.split(",") if item.strip()}
    rows = [row for row in rows if str(row.get("dataset")) in allowed]
    if max_samples > 0:
        rows = rows[:max_samples]
    mapping = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"Duplicate manifest key {key}: {manifest}")
        if str(row.get("offline_mode")) != expected_mode:
            raise RuntimeError(f"Manifest mode mismatch for {key}: {manifest}")
        mapping[key] = row
    if not mapping:
        raise RuntimeError(f"No rows selected from {manifest}")
    return mapping


def _distribution_stats(payload):
    target = payload["target_offline_68"].float()
    primary = payload["primary_fg_37"].float()
    complementary = payload["complementary_fg_37"].float()
    consolidated = payload["consolidated_fg_37"].float()
    background = payload["background_evidence_37"].float()
    conflict = payload["conflict_37"].float()
    commitment = payload["commitment_37"].float()
    cross_up = (primary <= 0.5) & (consolidated > 0.5)
    cross_down = (primary > 0.5) & (consolidated <= 0.5)
    return {
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
        "hard_area@0.5": float((target > 0.5).float().mean()),
        "neutral_ratio_0.4_0.6": float(((target >= 0.4) & (target <= 0.6)).float().mean()),
        "strong_fg_ratio_gt_0.8": float((target > 0.8).float().mean()),
        "strong_bg_ratio_lt_0.2": float((target < 0.2).float().mean()),
        "primary_fg_mean": float(primary.mean()),
        "primary_fg_hard_area": float((primary > 0.5).float().mean()),
        "complementary_fg_mean": float(complementary.mean()),
        "complementary_fg_gt_0.1": float((complementary > 0.1).float().mean()),
        "complementary_fg_gt_0.3": float((complementary > 0.3).float().mean()),
        "consolidated_fg_mean": float(consolidated.mean()),
        "consolidated_fg_hard_area": float((consolidated > 0.5).float().mean()),
        "foreground_area_delta": float(
            (consolidated > 0.5).float().mean() - (primary > 0.5).float().mean()
        ),
        "cross_bg_to_fg_ratio": float(cross_up.float().mean()),
        "cross_fg_to_bg_ratio": float(cross_down.float().mean()),
        "cross_bg_to_fg_background_mean": float(
            background[cross_up].mean() if bool(cross_up.any()) else 0.0
        ),
        "background_mean": float(background.mean()),
        "conflict_mean": float(conflict.mean()),
        "conflict_gt_0.3": float((conflict > 0.3).float().mean()),
        "commitment_mean": float(commitment.mean()),
        "commitment_lt_0.2": float((commitment < 0.2).float().mean()),
        "commitment_lt_0.5": float((commitment < 0.5).float().mean()),
    }


def _find_file(root, dataset, folder, stem):
    directory = Path(root) / dataset / folder
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {folder} for {dataset}/{stem}: {directory}")


def _load_gt(data_root, dataset, stem):
    path = _find_file(data_root, dataset, "gt", stem)
    image = Image.open(path).convert("L")
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None], path


def _quality(prediction, gt):
    pred = _resize(prediction.float(), gt.shape[-2:]).clamp(0.0, 1.0)
    hard = pred > 0.5
    truth = gt > 0.5
    tp = float((hard & truth).sum())
    fp = float((hard & ~truth).sum())
    fn = float((~hard & truth).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "MAE": float((pred - gt).abs().mean()),
        "soft_BCE": float(F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), gt)),
        "foreground_area_ratio": float(hard.float().mean()),
        "FP_ratio": float((hard & ~truth).float().mean()),
        "FN_ratio": float((~hard & truth).float().mean()),
    }


def _panel_from_tensor(value, title, panel_size=240):
    value = value.detach().cpu().float().squeeze().clamp(0.0, 1.0).numpy()
    image = Image.fromarray(np.uint8(np.round(value * 255)), mode="L").resize(
        (panel_size, panel_size), Image.Resampling.BILINEAR
    ).convert("RGB")
    canvas = Image.new("RGB", (panel_size, panel_size + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), title, fill="black")
    draw.text((5, 18), f"mean={float(value.mean()):.4f}", fill="black")
    return canvas


def _panel_from_image(path, title, panel_size=240, binary=False):
    image = Image.open(path).convert("L" if binary else "RGB")
    if binary:
        image = image.point(lambda x: 255 if x > 127 else 0).convert("RGB")
    image = image.resize(
        (panel_size, panel_size),
        Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (panel_size, panel_size + 34), "white")
    canvas.paste(image, (0, 34))
    ImageDraw.Draw(canvas).text((5, 8), title, fill="black")
    return canvas


def _save_visual(key, payloads, data_root, output_path, reason):
    dataset, stem = key
    reference = payloads["complementary_fg_conflict"]
    panels = []
    if data_root:
        panels.append(
            _panel_from_image(_find_file(data_root, dataset, "im", stem), "RGB")
        )
        panels.append(
            _panel_from_image(
                _find_file(data_root, dataset, "gt", stem), "GT (audit only)", binary=True
            )
        )
    else:
        zero = torch.zeros((1, 37, 37))
        panels.extend((_panel_from_tensor(zero, "RGB unavailable"), _panel_from_tensor(zero, "GT unavailable")))
    for title, field in (
        ("Residual R", "residual_37"),
        ("Random-walk P_rw", "p_rw_37"),
        ("Primary foreground F0", "primary_fg_37"),
        ("Background evidence B", "background_evidence_37"),
        ("Semantic tendency H", "semantic_fg_tendency_37"),
        ("Latent support L", "latent_support_37"),
        ("Complementary foreground G", "complementary_fg_37"),
        ("Consolidated foreground F", "consolidated_fg_37"),
        ("Conflict F*B", "conflict_37"),
        ("Conflict-calibrated C", "commitment_37"),
    ):
        panels.append(_panel_from_tensor(reference[field], title))
    panels.extend(
        [
            _panel_from_tensor(payloads["baseline_dp"]["target_offline_68"], "V0 target"),
            _panel_from_tensor(payloads["complementary_fg"]["target_offline_68"], "V1 target"),
            _panel_from_tensor(payloads["complementary_fg_conflict"]["target_offline_68"], "V2 target"),
        ]
    )
    cols = 5
    rows = (len(panels) + cols - 1) // cols
    width, height = panels[0].size
    header = 34
    grid = Image.new("RGB", (cols * width, rows * height + header), "white")
    ImageDraw.Draw(grid).text(
        (6, 8), f"{dataset}/{stem} | selection={reason}", fill="black"
    )
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % cols) * width, header + (index // cols) * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def _mean_dict(rows):
    keys = rows[0].keys()
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode, option in (
        ("baseline_dp", "v0-root"),
        ("complementary_fg", "v1-root"),
        ("complementary_fg_conflict", "v2-root"),
        ("direct_consolidated_fg", "v3-root"),
    ):
        parser.add_argument(f"--{option}", default=DEFAULT_ROOTS[mode])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer.")
    output_root = Path(args.output_root)
    existing = [path for path in output_root.rglob("*") if path.is_file()] if output_root.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"Output root is non-empty and overwrite=false: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    root_values = {
        "baseline_dp": args.v0_root,
        "complementary_fg": args.v1_root,
        "complementary_fg_conflict": args.v2_root,
        "direct_consolidated_fg": args.v3_root,
    }
    maps = {
        mode: _manifest_map(root_values[mode], mode, args.datasets, args.max_samples)
        for mode in MODE_ORDER
    }
    reference_keys = list(maps["baseline_dp"])
    reference_set = set(reference_keys)
    if args.max_samples < 0 and len(reference_keys) != 4040:
        raise RuntimeError(
            "Full DABE-Clean v3 audit requires exactly 4040 samples, got "
            f"{len(reference_keys)}."
        )
    for mode in MODE_ORDER[1:]:
        if set(maps[mode]) != reference_set:
            raise RuntimeError(f"Cache key mismatch: baseline_dp vs {mode}")

    formula_max = defaultdict(float)
    sample_rows = []
    gt_rows = []
    gt_diagnostics = {
        "v1_new_fg_true_fg_count": 0.0,
        "v1_new_fg_count": 0.0,
        "complementary_bins": defaultdict(lambda: {"gt_fg": 0.0, "pixels": 0.0}),
        "conflict_bins": defaultdict(lambda: {"errors": 0.0, "pixels": 0.0}),
        "commitment_bins": defaultdict(lambda: {"abs_error": 0.0, "pixels": 0.0}),
    }
    selection_scores = defaultdict(dict)
    for key in reference_keys:
        per_mode_payloads = {}
        gt = _load_gt(args.data_root, *key)[0] if args.data_root else None
        for mode in MODE_ORDER:
            payload, errors = _load_and_audit(maps[mode][key], mode)
            per_mode_payloads[mode] = payload
            for name, value in errors.items():
                formula_max[f"{mode}:{name}"] = max(
                    formula_max[f"{mode}:{name}"], value
                )
            stats = _distribution_stats(payload)
            sample_rows.append(
                {"dataset": key[0], "stem": key[1], "offline_mode": mode, **stats}
            )
            if args.data_root:
                for name, prediction in (
                    ("primary_fg", payload["primary_fg_37"]),
                    ("consolidated_fg", payload["consolidated_fg_37"]),
                    (f"{mode}_target", payload["target_offline_68"]),
                ):
                    gt_rows.append(
                        {
                            "dataset": key[0],
                            "stem": key[1],
                            "offline_mode": mode,
                            "prediction": name,
                            **_quality(prediction, gt),
                        }
                    )
        ref = per_mode_payloads["complementary_fg_conflict"]
        selection_scores["complementary_high"][key] = float(ref["complementary_fg_37"].mean())
        selection_scores["v1_expansion"][key] = float(
            ((ref["consolidated_fg_37"] > 0.5).float() - (ref["primary_fg_37"] > 0.5).float()).mean()
        )
        selection_scores["v2_conflict"][key] = float(ref["conflict_37"].mean())
        selection_scores["v2_over_neutral"][key] = float(
            ((per_mode_payloads["complementary_fg_conflict"]["target_offline_68"] >= 0.4)
             & (per_mode_payloads["complementary_fg_conflict"]["target_offline_68"] <= 0.6)).float().mean()
        )
        if args.data_root:
            gt68 = _resize(gt, (68, 68))
            v1 = per_mode_payloads["complementary_fg"]["target_offline_68"]
            v2 = per_mode_payloads["complementary_fg_conflict"]["target_offline_68"]
            bg = gt68 <= 0.5
            selection_scores["v1_fp_v2_soft_suppression"][key] = float(
                ((v1 - v2).clamp_min(0.0) * bg.float()).mean()
            )
            primary_gt = _resize(ref["primary_fg_37"], gt.shape[-2:])
            consolidated_gt = _resize(ref["consolidated_fg_37"], gt.shape[-2:])
            complementary_gt = _resize(ref["complementary_fg_37"], gt.shape[-2:])
            conflict_gt = _resize(ref["conflict_37"], gt.shape[-2:])
            commitment_gt = _resize(ref["commitment_37"], gt.shape[-2:])
            target_v2_gt = _resize(
                ref["target_offline_68"], gt.shape[-2:]
            ).clamp(0.0, 1.0)
            truth = gt > 0.5
            new_fg = (primary_gt <= 0.5) & (consolidated_gt > 0.5)
            gt_diagnostics["v1_new_fg_true_fg_count"] += float(
                (new_fg & truth).sum()
            )
            gt_diagnostics["v1_new_fg_count"] += float(new_fg.sum())
            for lower, upper in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.000001)):
                label = f"[{lower:.1f},{min(upper, 1.0):.1f}{']' if upper > 1.0 else ')'}"
                mask = (complementary_gt >= lower) & (complementary_gt < upper)
                gt_diagnostics["complementary_bins"][label]["gt_fg"] += float(
                    (truth & mask).sum()
                )
                gt_diagnostics["complementary_bins"][label]["pixels"] += float(mask.sum())
                conflict_mask = (conflict_gt >= lower) & (conflict_gt < upper)
                error = (target_v2_gt > 0.5) != truth
                gt_diagnostics["conflict_bins"][label]["errors"] += float(
                    (error & conflict_mask).sum()
                )
                gt_diagnostics["conflict_bins"][label]["pixels"] += float(
                    conflict_mask.sum()
                )
                commitment_mask = (commitment_gt >= lower) & (commitment_gt < upper)
                gt_diagnostics["commitment_bins"][label]["abs_error"] += float(
                    ((target_v2_gt - gt).abs() * commitment_mask.float()).sum()
                )
                gt_diagnostics["commitment_bins"][label]["pixels"] += float(
                    commitment_mask.sum()
                )

    with (output_root / "offline_stats_by_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    if gt_rows:
        with (output_root / "gt_quality_by_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gt_rows[0]))
            writer.writeheader()
            writer.writerows(gt_rows)

    summary = {
        "payload_version": DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
        "formula_fingerprint": DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT,
        "samples": len(reference_keys),
        "diagnostic_only": args.max_samples > 0,
        "formula_tolerance": FORMULA_TOLERANCE,
        "formula_max_abs_errors": dict(formula_max),
        "formula_pass": max(formula_max.values(), default=0.0) <= FORMULA_TOLERANCE,
        "distribution_by_mode": {
            mode: _mean_dict(
                [
                    {key: value for key, value in row.items() if key not in {"dataset", "stem", "offline_mode"}}
                    for row in sample_rows
                    if row["offline_mode"] == mode
                ]
            )
            for mode in MODE_ORDER
        },
        "gt_enabled": bool(args.data_root),
    }
    v0_summary = summary["distribution_by_mode"]["baseline_dp"]
    v1_summary = summary["distribution_by_mode"]["complementary_fg"]
    v2_summary = summary["distribution_by_mode"]["complementary_fg_conflict"]
    summary["cross_variant_deltas"] = {
        "v1_minus_v0_hard_area": v1_summary["hard_area@0.5"]
        - v0_summary["hard_area@0.5"],
        "v1_minus_v0_target_mean": v1_summary["target_mean"]
        - v0_summary["target_mean"],
        "v2_minus_v1_hard_area": v2_summary["hard_area@0.5"]
        - v1_summary["hard_area@0.5"],
        "v2_minus_v1_target_mean": v2_summary["target_mean"]
        - v1_summary["target_mean"],
        "v2_minus_v1_neutral_ratio": v2_summary["neutral_ratio_0.4_0.6"]
        - v1_summary["neutral_ratio_0.4_0.6"],
    }
    if gt_rows:
        grouped = defaultdict(list)
        for row in gt_rows:
            grouped[row["prediction"]].append(
                {key: value for key, value in row.items() if key not in {"dataset", "stem", "offline_mode", "prediction"}}
            )
        summary["gt_quality"] = {
            name: _mean_dict(rows) for name, rows in grouped.items()
        }
        def ratio(numerator, denominator):
            return float(numerator) / max(float(denominator), 1.0)

        summary["gt_diagnostics"] = {
            "v1_new_foreground_true_fg_ratio": ratio(
                gt_diagnostics["v1_new_fg_true_fg_count"],
                gt_diagnostics["v1_new_fg_count"],
            ),
            "v1_new_foreground_pixels": gt_diagnostics["v1_new_fg_count"],
            "complementary_fg_bins_gt_fg_ratio": {
                label: ratio(values["gt_fg"], values["pixels"])
                for label, values in gt_diagnostics["complementary_bins"].items()
            },
            "conflict_bins_error_ratio": {
                label: ratio(values["errors"], values["pixels"])
                for label, values in gt_diagnostics["conflict_bins"].items()
            },
            "commitment_bins_soft_mae": {
                label: ratio(values["abs_error"], values["pixels"])
                for label, values in gt_diagnostics["commitment_bins"].items()
            },
            "v2_minus_v1": {
                metric: summary["gt_quality"]["complementary_fg_conflict_target"][metric]
                - summary["gt_quality"]["complementary_fg_target"][metric]
                for metric in ("Precision", "Recall", "F1", "IoU", "MAE", "FP_ratio", "FN_ratio")
            },
        }
    (output_root / "offline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rng = random.Random(0)
    random_keys = rng.sample(reference_keys, k=min(3, len(reference_keys)))
    selections = {
        f"random_{index + 1}": key for index, key in enumerate(random_keys)
    }
    for reason, scores in selection_scores.items():
        if scores:
            selections[reason] = max(scores, key=scores.get)
    visual_root = output_root / "visualizations"
    used = set()
    for reason, key in selections.items():
        suffix = reason if key not in used else f"{reason}_duplicate"
        used.add(key)
        selected_payloads = {
            mode: _load_and_audit(maps[mode][key], mode)[0]
            for mode in MODE_ORDER
        }
        _save_visual(
            key,
            selected_payloads,
            args.data_root,
            visual_root / f"{key[0]}_{key[1]}_{suffix}.png",
            reason,
        )

    print(f"FORMULA_AUDIT_PASS={summary['formula_pass']}")
    print(f"samples={len(reference_keys)}")
    print(f"gt_enabled={bool(args.data_root)}")
    print(f"summary={output_root / 'offline_summary.json'}")
    print(f"sample_csv={output_root / 'offline_stats_by_sample.csv'}")
    print(f"visualizations={visual_root}")


if __name__ == "__main__":
    main()
