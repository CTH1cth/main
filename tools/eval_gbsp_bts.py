#!/usr/bin/env python3
"""Evaluate the predeclared BTS-0.5 branch from existing GBSP cross-fit caches.

No DINO forward, PCA refit, GT calibration, threshold search, R1 mask, or target
area prior is used.  GT is opened only after all image-specific thresholds and
patch masks have been computed.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, torch_load  # noqa: E402
from models.gbsp_thresholding import BackgroundTailShrinkageThreshold  # noqa: E402
from tools.eval_gbsp_threshold import (  # noqa: E402
    DATASETS,
    EXPECTED,
    METRICS,
    _aggregate,
    _balanced_subset,
    _binary_metrics,
    _load_gt,
    _manifest,
    _resize,
    _sample_keys,
    _table,
    _write_csv,
    _write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold.py"
METHODS = ("fixed_058", "bts_000", "bts_050")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _tensor(payload: dict, field: str, path: Path, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value):
        raise ValueError(f"{field} must be a tensor: {path}")
    value = value.detach().cpu().float().contiguous()
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{field} shape {tuple(value.shape)} != {shape}: {path}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or (
            str(payload.get("dataset")), str(payload.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"cross-fit cache identity mismatch: {path}")
        if bool(payload.get("gt_used_for_generation", True)) or bool(
            payload.get("r1_used_for_generation", True)
        ) or bool(payload.get("target_area_prior_used", True)):
            raise RuntimeError(f"cache violates GT/R1/area independence: {path}")

        # Compute and freeze all thresholds before opening GT.
        raw = _tensor(payload, "gbsp_absolute_raw", path, (1, 37, 37))
        minmax = _tensor(payload, "gbsp_absolute_minmax", path, (1, 37, 37))
        if float(minmax.min()) < -1e-6 or float(minmax.max()) > 1.0 + 1e-6:
            raise ValueError(f"GBSP Min-Max score escaped [0,1]: {path}")
        background_z = _tensor(payload, "background_oof_z", path)
        fold_location = _tensor(payload, "fold_scale_median", path, (5,))
        fold_scale = _tensor(payload, "fold_scale_mad", path, (5,))
        common = dict(
            background_quantile=float(task["background_quantile"]),
            global_minmax_threshold=float(task["global_minmax_threshold"]),
            eps=float(task["eps"]),
        )
        bts_000 = BackgroundTailShrinkageThreshold(
            shrinkage=float(task["baseline_shrinkage"]), **common
        ).apply(background_z, fold_location, fold_scale, raw, minmax)
        bts_050 = BackgroundTailShrinkageThreshold(
            shrinkage=float(task["shrinkage"]), **common
        ).apply(background_z, fold_location, fold_scale, raw, minmax)
        repeated = BackgroundTailShrinkageThreshold(
            shrinkage=float(task["shrinkage"]), **common
        ).apply(background_z, fold_location, fold_scale, raw, minmax)
        fixed_patch = minmax > float(task["global_minmax_threshold"])
        if (
            bts_000.threshold != float(task["global_minmax_threshold"])
            or not torch.equal(bts_000.binary_mask.reshape_as(fixed_patch), fixed_patch)
        ):
            raise RuntimeError(f"lambda=0 does not exactly reproduce fixed-0.58: {path}")
        if (
            bts_050.threshold != repeated.threshold
            or not torch.equal(bts_050.binary_mask, repeated.binary_mask)
        ):
            raise RuntimeError(f"BTS-0.5 is not exactly reproducible: {path}")
        thresholds = {
            "fixed_058": float(task["global_minmax_threshold"]),
            "bts_000": bts_000.threshold,
            "bts_050": bts_050.threshold,
        }
        patch_masks = {
            method: (minmax > threshold).float()
            for method, threshold in thresholds.items()
        }

        gt_path = Path(task["gt_path"] or payload.get("gt_path", ""))
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt = _load_gt(gt_path)
        native_score = _resize(minmax, tuple(gt.shape[-2:]))
        native_masks = {
            method: (native_score > threshold).float()
            for method, threshold in thresholds.items()
        }
        if not torch.equal(native_masks["fixed_058"], native_masks["bts_000"]):
            raise RuntimeError(f"native lambda=0 reproduction failed: {path}")

        context = FastCODContext(gt)
        rows = []
        for method in METHODS:
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "cache_path": str(path),
                    "image_path": task["image_path"] or payload.get("image_path", ""),
                    "gt_path": str(gt_path),
                    "method": method,
                    **_binary_metrics(context, native_masks[method]),
                    "empty_mask": int(float(patch_masks[method].mean()) == 0.0),
                    "area_over_50": int(float(patch_masks[method].mean()) > 0.5),
                }
            )
        by_method = {row["method"]: row for row in rows}
        fixed = by_method["fixed_058"]
        for row in rows:
            row["delta_F_beta_w_vs_fixed_058"] = row["F_beta_w"] - fixed["F_beta_w"]
            row["delta_Precision_vs_fixed_058"] = row["Precision"] - fixed["Precision"]
            row["delta_Recall_vs_fixed_058"] = row["Recall"] - fixed["Recall"]

        threshold_rows = []
        for method in METHODS:
            result = bts_000 if method != "bts_050" else bts_050
            threshold_rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    "lambda": 0.0 if method != "bts_050" else float(task["shrinkage"]),
                    "threshold_minmax": thresholds[method],
                    "background_z_q95": result.background_z_quantile,
                    "background_raw_threshold": result.background_raw_threshold,
                    "background_minmax_threshold_unclipped": result.background_minmax_threshold_unclipped,
                    "background_minmax_threshold": result.background_minmax_threshold,
                    "minmax_mapping_clipped": int(result.minmax_mapping_clipped),
                    "foreground_area_patch": float(patch_masks[method].mean()),
                    "background_candidate_count": int(payload["background_candidate_count"]),
                    "background_log_residual_mad": float(fold_scale.mean()),
                }
            )
        diagnostic = {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(path),
            "image_path": task["image_path"] or payload.get("image_path", ""),
            "gt_path": str(gt_path),
            "bts_threshold": bts_050.threshold,
            "background_minmax_threshold": bts_050.background_minmax_threshold,
            "mapping_clipped": int(bts_050.minmax_mapping_clipped),
            "bts_delta_F_beta_w": by_method["bts_050"]["F_beta_w"] - fixed["F_beta_w"],
            "lambda0_exact": True,
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        return {"rows": rows, "threshold_rows": threshold_rows, "diagnostic": diagnostic}
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""),
            "stem": task.get("stem", ""),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def _plots(output_dir: Path, summary: list[dict], rows: list[dict], thresholds: list[dict], diagnostics: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return [{"plot": "all", "error": repr(error)}]

    def run(name, function):
        try:
            function(plt, output_dir / name)
        except Exception as error:
            failures.append({"plot": name, "error": repr(error), "traceback": traceback.format_exc()})

    bts_thresholds = [row for row in thresholds if row["method"] == "bts_050"]

    def threshold_plot(plt, path):
        values = [row["threshold_minmax"] for row in bts_thresholds]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(30, max(5, len(values))), color="#2563eb", alpha=0.85)
        ax.axvline(0.58, color="#dc2626", linestyle="--", label="fixed 0.58")
        ax.set(xlabel="BTS-0.5 Min-Max threshold", ylabel="Images")
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def delta_plot(plt, path):
        values = [row["delta_F_beta_w_vs_fixed_058"] for row in rows if row["method"] == "bts_050"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=min(30, max(5, len(values))), color="#10b981", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set(xlabel="Per-image Fbeta_w delta vs fixed 0.58", ylabel="Images")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def comparison_plot(plt, path):
        macro = [row for row in summary if row["scope"] == "dataset_macro"]
        x = np.arange(len(macro)); width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        for offset, field in ((-width, "Precision"), (0.0, "Recall"), (width, "Area")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, [row["method"] for row in macro])
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def visualizations(plt, path):
        ordered = sorted(diagnostics, key=lambda row: row["bts_delta_F_beta_w"])
        selected = (ordered[:3] + ordered[-3:]) if len(ordered) > 6 else ordered
        fig, axes = plt.subplots(len(selected), 5, figsize=(14, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            with Image.open(item["image_path"]) as image:
                rgb = np.array(image.convert("RGB"), copy=True)
            with Image.open(item["gt_path"]) as image:
                gt = np.array(image.convert("L"), copy=True)
            mm = payload["gbsp_absolute_minmax"][0].detach().cpu().numpy()
            fixed = mm > 0.58
            bts = mm > float(item["bts_threshold"])
            panels = (
                (rgb, "RGB"), (gt, "GT"), (mm, "GBSP Min-Max"),
                (fixed, "fixed 0.58"),
                (bts, f"BTS-0.5 t={item['bts_threshold']:.3f}"),
            )
            for ax, (array, title) in zip(row_axes, panels):
                ax.imshow(array, cmap=None if array.ndim == 3 else "gray")
                ax.set_title(title); ax.axis("off")
            row_axes[0].set_ylabel(
                f"{item['dataset']}/{item['stem']}\ndF={item['bts_delta_F_beta_w']:+.3f}"
            )
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

    run("bts_threshold_distribution.png", threshold_plot)
    run("bts_per_image_fbw_delta.png", delta_plot)
    run("bts_precision_recall_area.png", comparison_plot)
    run("bts_visualizations.png", visualizations)
    return failures


def _report(output_dir: Path, summary: list[dict], audit: dict) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    lines = [
        "# GBSP BTS-0.5 Report",
        "",
        "- Fixed branch after CF-BRC-HC failure; lambda is not searched.",
        "- OOF background z Q95 is inverted through fold log-MAD scales, median-fused in raw space, then mapped to image Min-Max before shrinkage.",
        "- GT, R1 masks/area, target-area priors and dataset-specific thresholds are not used for calibration.",
        "- lambda=0 must exactly reproduce fixed-0.58.",
        "",
        "## Macro metrics",
        "",
        _table(macro, ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")),
        "",
        "## Audit",
        "",
        f"- Stage: {audit['stage']}",
        f"- Valid/failed: {audit['num_valid']}/{audit['num_failed']}",
        f"- lambda=0 exact mismatches: {audit['lambda0_mismatch_count']}",
        f"- BTS threshold mean/median: {audit['bts_threshold_mean']:.6f}/{audit['bts_threshold_median']:.6f}",
        f"- Min-Max mapping clipping ratio: {audit['mapping_clipped_ratio']:.6f}",
        f"- BTS empty/large-mask ratio: {audit['bts_empty_ratio']:.6f}/{audit['bts_large_ratio']:.6f}",
        f"- Gate: **{audit['gate']['status']}** — {audit['gate']['reason']}",
    ]
    (output_dir / "BTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    root = _resolve(args.crossfit_root)
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _manifest(root / "manifest_test.jsonl")
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list))
        mapping = {(row["dataset"], row["stem"]): row for row in rows}
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample list identities missing: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if args.max_samples >= 0:
        rows = _balanced_subset(rows, args.max_samples)
    if not rows:
        raise RuntimeError("no samples selected")
    settings = {
        "background_quantile": float(cfg.GBSP_THRESHOLD_BTS_BACKGROUND_QUANTILE),
        "global_minmax_threshold": float(cfg.GBSP_THRESHOLD_BTS_GLOBAL_MINMAX),
        "shrinkage": float(cfg.GBSP_THRESHOLD_BTS_LAMBDA),
        "baseline_shrinkage": float(cfg.GBSP_THRESHOLD_BTS_BASELINE_LAMBDA),
        "domain_mapping": str(cfg.GBSP_THRESHOLD_BTS_DOMAIN_MAPPING),
        "eps": float(cfg.GBSP_THRESHOLD_EPS),
        "gt_used_for_calibration": False,
        "r1_used_for_calibration": False,
        "target_area_prior_used": False,
        "threshold_search_used": False,
        "dino_forward_used": False,
        "pca_refit_used": False,
    }
    tasks = [
        {
            **settings,
            "dataset": str(row["dataset"]),
            "stem": str(row["stem"]),
            "cache_path": row["cache_path"],
            "image_path": row.get("image_path", ""),
            "gt_path": row.get("gt_path", ""),
        }
        for row in rows
    ]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP BTS eval {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("BTS evaluation produced no valid samples")
    per_image = [row for result in valid for row in result["rows"]]
    threshold_rows = [row for result in valid for row in result["threshold_rows"]]
    diagnostics = [result["diagnostic"] for result in valid]
    summary = _aggregate(per_image)
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    fixed, bts = macro["fixed_058"], macro["bts_050"]
    bts_rows = [row for row in per_image if row["method"] == "bts_050"]
    threshold_values = np.asarray([row["bts_threshold"] for row in diagnostics])
    counts = Counter(row["dataset"] for row in tasks)
    full = len(valid) == len(tasks) == sum(EXPECTED.values()) and dict(counts) == EXPECTED
    stage = "full6473" if full else ("pilot200" if len(tasks) == 200 and bool(args.sample_list) else "smoke")
    lambda0_mismatch = sum(not row["lambda0_exact"] for row in diagnostics)
    empty_ratio = _mean(row["empty_mask"] for row in bts_rows)
    large_ratio = _mean(row["area_over_50"] for row in bts_rows)
    clipped_ratio = _mean(row["mapping_clipped"] for row in diagnostics)
    if failures or lambda0_mismatch:
        gate = {"status": "STOP", "reason": "numerical/identity failure or lambda=0 reproduction mismatch"}
    elif stage == "smoke":
        gate = {"status": "PASS_TO_PILOT200", "reason": "20-image implementation and identity audit passed; quality is provisional"}
    elif empty_ratio > 0.01 or large_ratio > 0.01:
        gate = {"status": "STOP", "reason": "empty or >50% mask ratio exceeds 1%"}
    elif bts["F_beta_w"] - fixed["F_beta_w"] < -0.008 or bts["MAE"] - fixed["MAE"] > 0.005:
        gate = {"status": "STOP", "reason": "quality gate versus fixed-0.58 failed"}
    else:
        gate = {"status": "PASS_TO_FULL_OR_DOWNSTREAM", "reason": "predeclared stability and quality gates passed"}
    audit = {
        "schema": "gbsp_bts_eval_v1",
        "stage": stage,
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "dataset_counts": dict(counts),
        **settings,
        "lambda0_mismatch_count": lambda0_mismatch,
        "bts_threshold_mean": float(threshold_values.mean()),
        "bts_threshold_median": float(np.median(threshold_values)),
        "bts_threshold_std": float(threshold_values.std()),
        "bts_threshold_q10": float(np.quantile(threshold_values, 0.10)),
        "bts_threshold_q90": float(np.quantile(threshold_values, 0.90)),
        "mapping_clipped_ratio": clipped_ratio,
        "bts_empty_ratio": empty_ratio,
        "bts_large_ratio": large_ratio,
        "delta_F_beta_w_vs_fixed_058": bts["F_beta_w"] - fixed["F_beta_w"],
        "delta_MAE_vs_fixed_058": bts["MAE"] - fixed["MAE"],
        "gate": gate,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_csv(output_dir / "per_image_metrics.csv", per_image)
    _write_csv(output_dir / "per_dataset_metrics.csv", summary)
    _write_csv(output_dir / "bts_threshold_distribution.csv", threshold_rows)
    _write_json(output_dir / "numerical_audit.json", audit)
    _write_csv(
        output_dir / "bts_summary.csv",
        [row for row in summary if row["scope"] == "dataset_macro"],
    )
    plot_failures = _plots(output_dir, summary, per_image, threshold_rows, diagnostics)
    _write_json(output_dir / "visualization_failures.json", plot_failures)
    _report(output_dir, summary, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"BTS evaluation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--crossfit_root", "--crossfit-root", dest="crossfit_root", required=True)
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
