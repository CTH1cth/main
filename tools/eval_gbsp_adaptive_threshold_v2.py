#!/usr/bin/env python3
"""Original-size evaluation, gating and audit for GBSP threshold V2."""

from __future__ import annotations

import argparse
import csv
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
from scipy.stats import beta as beta_distribution
from scipy.stats import norm as normal_distribution

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from tools.eval_gbsp_threshold import (  # noqa: E402
    DATASETS,
    EXPECTED,
    _aggregate,
    _binary_metrics,
    _load_gt,
    _resize,
    _sample_keys,
    _table,
    _write_csv,
    _write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v2.py"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _manifest_path(root: Path) -> Path:
    candidates = (
        root / "manifest_test.jsonl",
        root / "ablations/M1_pilot200/manifest_test.jsonl",
        root / "dinov1-s8/manifest_test.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no manifest_test.jsonl below {root}")


def _manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        cache = Path(str(row.get("cache_path", "")))
        if not all(key) or key in seen or not cache.is_file():
            raise RuntimeError(f"invalid manifest row {path}:{line}: {key}")
        seen.add(key)
    return rows


def _select(rows: list[dict], sample_list: str | None, max_samples: int) -> list[dict]:
    if sample_list:
        keys = _sample_keys(_resolve(sample_list))
        mapping = {(str(row["dataset"]), str(row["stem"])): row for row in rows}
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample identities missing: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if max_samples >= 0:
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError("no samples selected")
    return rows


def _tensor(payload: dict, fields: tuple[str, ...], path: Path) -> torch.Tensor:
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value) and tuple(value.shape) == (1, 37, 37):
            value = value.detach().cpu().float().contiguous()
            if bool(torch.isfinite(value).all()):
                return value
    raise ValueError(f"missing/nonfinite {fields}: {path}")


def _scalar_diagnostics(value: dict) -> dict:
    output = {}
    for key, item in value.items():
        if item is None or isinstance(item, (str, bool, int, float, np.integer, np.floating)):
            output[key] = item.item() if isinstance(item, (np.integer, np.floating)) else item
    return output


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or (
            str(payload.get("dataset")), str(payload.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"cache identity mismatch: {path}")
        adaptive = bool(task["adaptive"])
        if adaptive:
            if bool(payload.get("gt_used_for_generation", True)) or bool(
                payload.get("r1_used_for_generation", True)
            ) or bool(payload.get("fixed_058_used_for_generation", True)) or bool(
                payload.get("target_area_prior_used", True)
            ):
                raise RuntimeError(f"V2 cache violates independence contract: {path}")
            score = _tensor(payload, ("minmax_residual",), path)
            raw = _tensor(payload, ("raw_residual",), path)
            bc = payload.get("background_indices")
            available = payload.get("results", {})
            names = task["methods"] or list(available)
            missing = [name for name in names if name not in available]
            if missing:
                raise KeyError(f"variants missing from {path}: {missing}")
            adaptive_results = {name: available[name] for name in names}
        else:
            score = _tensor(payload, ("absolute_minmax", "gbsp_abs_minmax_37"), path)
            raw = _tensor(payload, ("absolute_raw", "gbsp_abs_raw_37"), path)
            bc = payload.get("background_indices")
            adaptive_results = {}
        if not torch.is_tensor(bc) or bc.ndim != 1:
            raise ValueError(f"background_indices missing: {path}")
        bc = bc.detach().cpu().long().contiguous()

        # Adaptive masks and thresholds are already frozen.  Fixed-0.58 is
        # constructed here solely as an evaluation reference before GT opens.
        patch_masks = {"fixed_058": (score > float(task["reference_threshold"])).float()}
        metadata = {
            "fixed_058": {
                "method_family": "fixed_reference",
                "variant": "fixed_058",
                "equivalent_minmax_threshold": float(task["reference_threshold"]),
                "foreground_area": float(patch_masks["fixed_058"].mean()),
                "bc_selected_as_foreground_ratio": float(
                    patch_masks["fixed_058"].reshape(-1).index_select(0, bc).mean()
                ),
                "empty_mask": bool(float(patch_masks["fixed_058"].mean()) == 0.0),
                "area_over_50pct": bool(float(patch_masks["fixed_058"].mean()) > 0.5),
                "numerical_failure": False,
                "diagnostics": {},
            }
        }
        for name, result in adaptive_results.items():
            mask = result.get("mask_37")
            if not torch.is_tensor(mask) or tuple(mask.shape) != (1, 37, 37):
                raise ValueError(f"invalid mask for {name}: {path}")
            mask = mask.detach().cpu().float().contiguous()
            if not bool(torch.isfinite(mask).all()) or float(mask.min()) < 0 or float(mask.max()) > 1:
                raise ValueError(f"non-binary-domain mask for {name}: {path}")
            patch_masks[name] = mask
            metadata[name] = {
                "method_family": str(result.get("method", name)),
                "variant": name,
                "equivalent_minmax_threshold": result.get("equivalent_minmax_threshold"),
                "foreground_area": float(result.get("foreground_area", mask.mean())),
                "bc_selected_as_foreground_ratio": float(result.get("bc_selected_as_foreground_ratio", 0.0)),
                "empty_mask": bool(result.get("empty_mask", float(mask.mean()) == 0.0)),
                "area_over_50pct": bool(result.get("area_over_50pct", float(mask.mean()) > 0.5)),
                "numerical_failure": bool(result.get("numerical_failure", True)),
                "diagnostics": _scalar_diagnostics(result.get("diagnostics", {})),
            }

        gt_path = Path(task["gt_path"] or payload.get("gt_path", ""))
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt = _load_gt(gt_path)
        shape = tuple(gt.shape[-2:])
        native_masks = {
            "fixed_058": (_resize(score, shape) > float(task["reference_threshold"])).float(),
            **{
                name: (_resize(mask, shape) > 0.5).float()
                for name, mask in patch_masks.items() if name != "fixed_058"
            },
        }
        context = FastCODContext(gt)
        rows = []
        for name in ("fixed_058", *adaptive_results):
            info = metadata[name]
            rows.append(
                {
                    "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(path),
                    "image_path": task["image_path"] or payload.get("image_path", ""),
                    "gt_path": str(gt_path), "method": name, "method_family": info["method_family"],
                    **_binary_metrics(context, native_masks[name]),
                    "foreground_area": info["foreground_area"],
                    "empty_mask": int(info["empty_mask"]),
                    "area_over_50": int(info["area_over_50pct"]),
                    "numerical_failure": int(info["numerical_failure"]),
                    "equivalent_minmax_threshold": info["equivalent_minmax_threshold"],
                    "bc_selected_as_foreground_ratio": info["bc_selected_as_foreground_ratio"],
                    **{
                        ("diagnostic_method" if key == "method" else key): value
                        for key, value in info["diagnostics"].items()
                    },
                }
            )
        by_method = {row["method"]: row for row in rows}
        fixed = by_method["fixed_058"]
        for row in rows:
            row["delta_F_beta_w_vs_fixed_058"] = row["F_beta_w"] - fixed["F_beta_w"]
            row["delta_Precision_vs_fixed_058"] = row["Precision"] - fixed["Precision"]
            row["delta_Recall_vs_fixed_058"] = row["Recall"] - fixed["Recall"]
        diagnostics = {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(path),
            "image_path": task["image_path"] or payload.get("image_path", ""), "gt_path": str(gt_path),
            "raw_max": float(raw.max()), "raw_min": float(raw.min()),
            "background_candidate_count": int(bc.numel()),
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        return {"rows": rows, "diagnostic": diagnostics}
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _mean(values) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _union_fields(rows: list[dict]) -> list[str]:
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    return fields


def _enrich_summary(summary: list[dict], per_image: list[dict]) -> None:
    """Attach V2 stability/threshold fields without changing COD metric means."""
    for aggregate in summary:
        subset = [
            row
            for row in per_image
            if row["method"] == aggregate["method"]
            and (aggregate["dataset"] == "ALL" or row["dataset"] == aggregate["dataset"])
        ]
        aggregate["method_family"] = _variant_family(per_image, aggregate["method"])
        aggregate["numerical_failure"] = _mean(row["numerical_failure"] for row in subset)
        aggregate["foreground_area_patch"] = _mean(row["foreground_area"] for row in subset)
        aggregate["bc_selected_as_foreground_ratio"] = _mean(
            row["bc_selected_as_foreground_ratio"] for row in subset
        )
        aggregate["equivalent_minmax_threshold"] = _mean(
            row.get("equivalent_minmax_threshold") for row in subset
        )


def _stage(adaptive: bool, count: int, counts: dict) -> str:
    if not adaptive:
        return "baseline_test20" if count == 20 else "baseline"
    if count == 20:
        return "test20"
    if count == 200:
        return "pilot200"
    if count == sum(EXPECTED.values()) and counts == EXPECTED:
        return "full6473"
    return "custom"


def _variant_family(rows: list[dict], method: str) -> str:
    return next(row["method_family"] for row in rows if row["method"] == method)


def _selection(
    stage: str,
    summary: list[dict],
    per_image: list[dict],
    cfg,
) -> dict:
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    fixed = macro["fixed_058"]
    methods = [method for method in macro if method != "fixed_058"]
    dataset = {
        (row["dataset"], row["method"]): row
        for row in summary if row["scope"] == "dataset"
    }
    evaluations = []
    for method in methods:
        row = macro[method]
        image_rows = [item for item in per_image if item["method"] == method]
        family = _variant_family(per_image, method)
        numerical_ratio = _mean(item["numerical_failure"] for item in image_rows)
        empty_ratio = _mean(item["empty_mask"] for item in image_rows)
        large_ratio = _mean(item["area_over_50"] for item in image_rows)
        dataset_deltas = {
            ds: dataset[(ds, method)]["F_beta_w"] - dataset[(ds, "fixed_058")]["F_beta_w"]
            for ds in DATASETS
        }
        noninferior = sum(item["delta_F_beta_w_vs_fixed_058"] >= -1e-12 for item in image_rows)
        reasons = []
        if stage == "test20":
            gate = cfg.GBSP_V2_TEST20_GATE
            checks = (
                (row["F_beta_w"] >= gate["F_beta_w_min"], "F_beta_w"),
                (row["Precision"] >= gate["Precision_min"], "Precision"),
                (gate["Area_min"] <= row["Area"] <= gate["Area_max"], "Area"),
                (numerical_ratio <= gate["numerical_failure_ratio_max"], "numerical_failure"),
                (empty_ratio <= gate["empty_ratio_max"], "empty_mask"),
                (large_ratio <= gate["large_ratio_max"], "large_mask"),
                (not all(value < 0 for value in dataset_deltas.values()), "all_datasets_decline"),
            )
        elif stage == "pilot200":
            gate = cfg.GBSP_V2_PILOT200_GATE
            checks = (
                (row["F_beta_w"] >= gate["F_beta_w_min"], "F_beta_w"),
                (row["MAE"] <= gate["MAE_max"], "MAE"),
                (row["Precision"] >= gate["Precision_min"], "Precision"),
                (gate["Area_min"] <= row["Area"] <= gate["Area_max"], "Area"),
                (numerical_ratio == 0.0, "numerical_failure"),
                (empty_ratio <= gate["empty_ratio_max"], "empty_mask"),
                (large_ratio <= gate["large_ratio_max"], "large_mask"),
                (dataset_deltas["TE-COD10K"] >= -gate["COD10K_F_beta_w_drop_max"], "COD10K"),
                (dataset_deltas["NC4K"] >= -gate["NC4K_F_beta_w_drop_max"], "NC4K"),
                (not all(value < 0 for value in dataset_deltas.values()), "all_datasets_decline"),
                (noninferior >= gate["per_image_noninferior_min"] or row["F_beta_w"] >= fixed["F_beta_w"], "per_image_noninferior"),
                (abs(row.get("equivalent_minmax_threshold", float("nan")) - 0.40) > 0.02 if math.isfinite(float(row.get("equivalent_minmax_threshold", float("nan")))) else True, "threshold_near_040"),
            )
        else:
            checks = ((numerical_ratio == 0.0, "numerical_failure"),)
        reasons.extend(name for passed, name in checks if not passed)
        evaluations.append(
            {
                "variant": method, "method_family": family, "passed": not reasons,
                "failure_reasons": reasons, "F_beta_w": row["F_beta_w"], "Precision": row["Precision"],
                "MAE": row["MAE"], "Area": row["Area"], "per_image_noninferior": noninferior,
                "numerical_failure_ratio": numerical_ratio, "empty_ratio": empty_ratio,
                "large_ratio": large_ratio, "dataset_F_beta_w_delta": dataset_deltas,
            }
        )
    eligible = [row for row in evaluations if row["passed"]]
    eligible.sort(key=lambda row: (-row["F_beta_w"], -row["Precision"], row["MAE"], -row["per_image_noninferior"], row["numerical_failure_ratio"], row["variant"]))
    selected, families = [], set()
    limit = 2 if stage == "test20" else 1
    for row in eligible:
        if row["method_family"] in families:
            continue
        selected.append(row)
        families.add(row["method_family"])
        if len(selected) == limit:
            break
    return {
        "stage": stage,
        "reference": {key: fixed[key] for key in ("S_m", "F_beta_w", "E_mean", "MAE", "Precision", "Recall", "Area")},
        "evaluations": evaluations,
        "selected": selected,
        "status": "PASS" if len(selected) == limit else "STOP",
        "required_selection_count": limit,
    }


def _frozen_config(selection: dict, run_config: dict | None) -> dict:
    source_variants = {
        row["variant"]: row
        for row in ((run_config or {}).get("settings", {}).get("variants", []))
    }
    selected = []
    for row in selection.get("selected", []):
        variant = row["variant"]
        selected.append(source_variants.get(variant, {"variant": variant, "method": row["method_family"]}))
    return {
        "schema": "gbsp_adaptive_threshold_v2_frozen_config",
        "source_stage": selection["stage"],
        "selected_variants": selected,
        "selection_order": [row["variant"] for row in selection.get("selected", [])],
        "gt_used_in_formula": False,
        "fixed_058_used_in_formula": False,
        "parameters_may_not_be_modified_after_freeze": True,
    }


def _plots(output_dir: Path, stage: str, summary: list[dict], rows: list[dict], threshold_rows: list[dict], diagnostics: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return [{"plot": "all", "error": repr(error)}]

    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    methods = [row["method"] for row in macro]
    adaptive_macro = [row for row in macro if row["method"] != "fixed_058"]

    def run(name, function):
        try:
            function(plt, output_dir / name)
        except Exception as error:
            failures.append({"plot": name, "error": repr(error), "traceback": traceback.format_exc()})

    def method_comparison(plt, path):
        fig, ax = plt.subplots(figsize=(max(9, len(macro) * 1.1), 4))
        x = np.arange(len(macro)); width = 0.2
        for offset, field in ((-1.5*width, "S_m"), (-0.5*width, "F_beta_w"), (0.5*width, "E_mean"), (1.5*width, "MAE")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, methods, rotation=30, ha="right"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def pra(plt, path):
        fig, ax = plt.subplots(figsize=(max(9, len(macro) * 1.1), 4))
        x = np.arange(len(macro)); width = 0.25
        for offset, field in ((-width, "Precision"), (0, "Recall"), (width, "Area")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, methods, rotation=30, ha="right"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def thresholds(plt, path):
        adaptive = [row for row in threshold_rows if row["method"] != "fixed_058" and row["equivalent_minmax_threshold"] is not None]
        grouped = {method: [float(row["equivalent_minmax_threshold"]) for row in adaptive if row["method"] == method] for method in methods if method != "fixed_058"}
        fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 1.1), 4))
        if grouped:
            ax.boxplot(list(grouped.values()), labels=list(grouped), showfliers=False)
            ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Equivalent Min-Max threshold")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def deltas(plt, path):
        adaptive_methods = [method for method in methods if method != "fixed_058"]
        data = [[row["delta_F_beta_w_vs_fixed_058"] for row in rows if row["method"] == method] for method in adaptive_methods]
        fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.1), 4))
        if data:
            ax.boxplot(data, labels=adaptive_methods, showfliers=False); ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Per-image Fbeta_w delta")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def dataset_plot(plt, path):
        dataset_rows = [row for row in summary if row["scope"] == "dataset"]
        x = np.arange(len(DATASETS)); width = 0.8 / max(len(methods), 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        for index, method in enumerate(methods):
            values = [next(row["F_beta_w"] for row in dataset_rows if row["dataset"] == ds and row["method"] == method) for ds in DATASETS]
            ax.bar(x + (index - (len(methods)-1)/2) * width, values, width, label=method)
        ax.set_xticks(x, DATASETS); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def empty_figure(plt, path, message):
        fig, ax = plt.subplots(figsize=(7, 3)); ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

    def visual_grid(plt, path, success: bool):
        if not adaptive_macro:
            empty_figure(plt, path, "No adaptive methods in this stage"); return
        best = max(adaptive_macro, key=lambda row: row["F_beta_w"])["method"]
        candidates = [row for row in rows if row["method"] == best]
        candidates.sort(key=lambda row: row["delta_F_beta_w_vs_fixed_058"], reverse=success)
        selected = candidates[:5]
        fig, axes = plt.subplots(len(selected), 7, figsize=(19, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"][best]
            score = payload["minmax_residual"][0].numpy()
            bc = np.zeros(37 * 37, dtype=np.float32)
            bc[payload["background_indices"].numpy()] = 1.0
            continuous = result.get("continuous_map_37")
            continuous = continuous[0].numpy() if torch.is_tensor(continuous) else np.zeros((37, 37))
            with Image.open(item["image_path"]) as image:
                rgb = np.array(image.convert("RGB"), copy=True)
            with Image.open(item["gt_path"]) as image:
                gt = np.array(image.convert("L"), copy=True)
            panels = (
                (rgb, "RGB"), (gt, "GT"), (score, "GBSP Min-Max"),
                (bc.reshape(37, 37), "Full BC"), (continuous, "posterior / p-value"),
                (score > 0.58, "fixed 0.58 reference"),
                (result["mask_37"][0].numpy(), best),
            )
            for ax, (array, title) in zip(row_axes, panels):
                ax.imshow(array, cmap=None if array.ndim == 3 else "gray")
                ax.set_title(title); ax.axis("off")
            row_axes[0].set_ylabel(
                f"{item['dataset']}/{item['stem']}\ndF={item['delta_F_beta_w_vs_fixed_058']:+.3f}"
            )
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    def family_examples(plt, path, families: set[str], kind: str, failure: bool = False):
        family_macro = [row for row in adaptive_macro if row.get("method_family") in families]
        if not family_macro:
            empty_figure(plt, path, f"No {kind} variants in this stage"); return
        best = max(family_macro, key=lambda row: row["F_beta_w"])["method"]
        candidates = [row for row in rows if row["method"] == best]
        candidates.sort(
            key=(
                (lambda row: (-row["numerical_failure"], row["delta_F_beta_w_vs_fixed_058"]))
                if failure
                else (lambda row: (row["numerical_failure"], -row["delta_F_beta_w_vs_fixed_058"]))
            )
        )
        selected = candidates[:4]
        fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"][best]
            score = payload["minmax_residual"][0].numpy()
            continuous = result.get("continuous_map_37")
            row_axes[0].hist(score.reshape(-1), bins=50, density=True, color="#64748b")
            diagnostic = result.get("diagnostics", {})
            x_grid = np.linspace(1e-4, 1.0 - 1e-4, 500)
            if result.get("method") == "ba_bmc" and all(
                key in diagnostic for key in ("alpha_bg", "beta_bg", "alpha_fg", "beta_fg", "pi_fg")
            ):
                pi = float(diagnostic["pi_fg"])
                bg_density = (1.0 - pi) * beta_distribution.pdf(
                    x_grid, float(diagnostic["alpha_bg"]), float(diagnostic["beta_bg"])
                )
                fg_density = pi * beta_distribution.pdf(
                    x_grid, float(diagnostic["alpha_fg"]), float(diagnostic["beta_fg"])
                )
                row_axes[0].plot(x_grid, bg_density, label="background")
                row_axes[0].plot(x_grid, fg_density, label="foreground")
                row_axes[0].plot(x_grid, bg_density + fg_density, label="mixture", color="black")
                row_axes[0].legend(fontsize=7)
            if result.get("method") == "ba_lgmc" and all(
                key in diagnostic for key in ("mu_bg", "sigma_bg", "mu_fg", "sigma_fg", "pi_fg")
            ):
                pi = float(diagnostic["pi_fg"])
                y_grid = np.log(x_grid / (1.0 - x_grid))
                jacobian = 1.0 / (x_grid * (1.0 - x_grid))
                bg_density = (1.0 - pi) * normal_distribution.pdf(
                    y_grid, float(diagnostic["mu_bg"]), float(diagnostic["sigma_bg"])
                ) * jacobian
                fg_density = pi * normal_distribution.pdf(
                    y_grid, float(diagnostic["mu_fg"]), float(diagnostic["sigma_fg"])
                ) * jacobian
                row_axes[0].plot(x_grid, bg_density, label="background")
                row_axes[0].plot(x_grid, fg_density, label="foreground")
                row_axes[0].plot(x_grid, bg_density + fg_density, label="mixture", color="black")
                row_axes[0].legend(fontsize=7)
            threshold = result.get("equivalent_minmax_threshold")
            if threshold is not None:
                row_axes[0].axvline(float(threshold), color="red", linestyle="--")
            row_axes[0].set_title(f"{kind} score distribution")
            row_axes[1].imshow(
                continuous[0].numpy() if torch.is_tensor(continuous) else np.zeros((37, 37)),
                cmap="magma",
            ); row_axes[1].set_title("posterior / p-value"); row_axes[1].axis("off")
            row_axes[2].imshow(result["mask_37"][0].numpy(), cmap="gray", vmin=0, vmax=1)
            row_axes[2].set_title(f"{best} mask"); row_axes[2].axis("off")
            row_axes[0].set_ylabel(f"{item['dataset']}/{item['stem']}")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

    def qdcp_examples(plt, path):
        qdcp_macro = [row for row in adaptive_macro if row.get("method_family") in {"qdcp_pl", "qdcp_k"}]
        if not qdcp_macro:
            empty_figure(plt, path, "No QDCP variants in this stage"); return
        best = max(qdcp_macro, key=lambda row: row["F_beta_w"])["method"]
        selected = [row for row in rows if row["method"] == best][:4]
        fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"][best]
            score = payload["minmax_residual"][0].numpy()
            ordered = np.sort(score.reshape(-1))[::-1]
            row_axes[0].plot(np.linspace(0, 1, ordered.size), ordered)
            threshold = result.get("equivalent_minmax_threshold")
            if threshold is not None:
                row_axes[0].axhline(float(threshold), color="red", linestyle="--")
            row_axes[0].set_title(f"{item['dataset']}/{item['stem']} sorted query")
            row_axes[1].imshow(result["mask_37"][0].numpy(), cmap="gray", vmin=0, vmax=1)
            row_axes[1].set_title(best); row_axes[1].axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

    run("method_comparison.png", method_comparison)
    if stage == "test20":
        run("method_comparison_test20.png", method_comparison)
    if stage == "pilot200":
        run("method_comparison_pilot200.png", method_comparison)
    run("precision_recall_area_comparison.png", pra)
    run("threshold_distribution.png", thresholds)
    run("per_image_fbw_delta.png", deltas)
    run("dataset_wise_comparison.png", dataset_plot)
    run("success_visualizations.png", lambda plt, path: visual_grid(plt, path, True))
    run("failure_visualizations.png", lambda plt, path: visual_grid(plt, path, False))
    run(
        "mixture_fit_success_examples.png",
        lambda plt, path: family_examples(plt, path, {"ba_bmc", "ba_lgmc"}, "mixture"),
    )
    run(
        "mixture_fit_failure_examples.png",
        lambda plt, path: family_examples(plt, path, {"ba_bmc", "ba_lgmc"}, "mixture", True),
    )
    run("evt_tail_examples.png", lambda plt, path: family_examples(plt, path, {"eb_evt"}, "EVT"))
    run("qdcp_examples.png", qdcp_examples)
    return failures


def _report(output_dir: Path, stage: str, summary: list[dict], selection: dict | None, audit: dict) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    lines = [
        "# GBSP Adaptive Threshold V2 Report", "",
        f"- Stage: {stage}",
        "- V2 calibration uses no GT, R1, target-area prior, dataset threshold, or fixed-0.58 formula component.",
        "- Fixed-0.58 is evaluated only as a frozen external reference.",
        "- DINO forward and GBSP PCA refit are not used.", "", "## Macro metrics", "",
        _table(macro, ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")),
        "", "## Audit", "",
        f"- Valid/failed images: {audit['num_valid']}/{audit['num_failed']}",
        f"- Baseline reproduction: {audit.get('baseline_reproduction_status', 'not_applicable')}",
    ]
    if selection:
        lines.extend(
            [
                f"- Selection status: **{selection['status']}**",
                f"- Selected variants: {[row['variant'] for row in selection['selected']]}",
            ]
        )
    (output_dir / "GBSP_ADAPTIVE_THRESHOLD_V2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    if bool(args.gbsp_root) == bool(args.threshold_root):
        raise ValueError("provide exactly one of --gbsp_root or --threshold_root")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    adaptive = bool(args.threshold_root)
    root = _resolve(args.threshold_root or args.gbsp_root)
    manifest_path = _manifest_path(root)
    rows = _select(_manifest(manifest_path), args.sample_list, args.max_samples)
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods
    if not adaptive and methods and set(methods) != {"fixed_058"}:
        raise ValueError("baseline mode only accepts --methods fixed_058")
    tasks = [
        {
            "dataset": str(row["dataset"]), "stem": str(row["stem"]),
            "cache_path": row["cache_path"], "image_path": row.get("image_path", ""),
            "gt_path": row.get("gt_path", ""), "adaptive": adaptive, "methods": methods,
            "reference_threshold": float(cfg.GBSP_V2_REFERENCE_FIXED_THRESHOLD),
        }
        for row in rows
    ]
    started = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP threshold V2 eval {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("V2 evaluation produced no valid samples")
    per_image = [row for result in valid for row in result["rows"]]
    diagnostics = [result["diagnostic"] for result in valid]
    summary = _aggregate(per_image)
    _enrich_summary(summary, per_image)
    counts = dict(Counter(task["dataset"] for task in tasks))
    stage = _stage(adaptive, len(tasks), counts)
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    target = cfg.GBSP_V2_BASELINE_TEST20_TARGET
    reproduction_error = {
        key: abs(macro["fixed_058"][key] - float(value)) for key, value in target.items()
    }
    reproduction_pass = bool(
        stage == "baseline_test20"
        and max(reproduction_error.values()) < float(cfg.GBSP_V2_BASELINE_TOLERANCE)
    )
    threshold_rows = [
        {
            "dataset": row["dataset"], "stem": row["stem"], "method": row["method"],
            "method_family": row["method_family"],
            "equivalent_minmax_threshold": row.get("equivalent_minmax_threshold"),
            "foreground_area": row["foreground_area"],
            "bc_selected_as_foreground_ratio": row.get("bc_selected_as_foreground_ratio"),
            "empty_mask": row["empty_mask"], "area_over_50pct": row["area_over_50"],
            "numerical_failure": row["numerical_failure"],
        }
        for row in per_image
    ]
    parameter_exclusions = {
        "cache_path", "image_path", "gt_path", "S_m", "F_beta_w", "F_beta_mean",
        "E_mean", "MAE", "Precision", "Recall", "Area", "IoU", "Dice",
        "foreground_area", "empty_mask", "area_over_50", "numerical_failure",
        "equivalent_minmax_threshold", "bc_selected_as_foreground_ratio",
        "delta_F_beta_w_vs_fixed_058", "delta_Precision_vs_fixed_058",
        "delta_Recall_vs_fixed_058",
    }
    parameter_rows = [
        {key: value for key, value in row.items() if key not in parameter_exclusions}
        for row in per_image if row["method"] != "fixed_058"
    ]
    selection = _selection(stage, summary, per_image, cfg) if adaptive else None
    run_config_path = root / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else None
    frozen = _frozen_config(selection, run_config) if selection else None
    audit = {
        "schema": "gbsp_adaptive_threshold_v2_eval",
        "stage": stage, "num_requested": len(tasks), "num_valid": len(valid), "num_failed": len(failures),
        "dataset_counts": counts, "gt_used_for_calibration": False, "r1_used_for_calibration": False,
        "fixed_058_used_in_adaptive_formula": False, "target_area_prior_used": False,
        "dino_forward_used": False, "pca_refit_used": False,
        "baseline_reproduction_abs_error": reproduction_error,
        "baseline_reproduction_status": "PASS" if reproduction_pass else ("NOT_APPLICABLE" if adaptive else "FAIL"),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_csv(output_dir / "per_image_metrics.csv", per_image, _union_fields(per_image))
    _write_csv(output_dir / "per_dataset_metrics.csv", summary, _union_fields(summary))
    _write_csv(output_dir / "threshold_distribution.csv", threshold_rows, _union_fields(threshold_rows))
    _write_csv(output_dir / "mixture_parameter_distribution.csv", parameter_rows, _union_fields(parameter_rows))
    _write_json(output_dir / "numerical_failure_summary.json", audit)
    if stage == "baseline_test20":
        _write_csv(output_dir / "baseline_reproduction.csv", [macro["fixed_058"]])
    if stage == "test20":
        _write_csv(output_dir / "test20_summary.csv", [row for row in summary if row["scope"] == "dataset_macro"])
    elif stage == "pilot200":
        _write_csv(output_dir / "pilot200_summary.csv", [row for row in summary if row["scope"] == "dataset_macro"])
    elif stage == "full6473":
        _write_csv(output_dir / "full6473_summary.csv", [row for row in summary if row["scope"] == "dataset_macro"])
    if selection:
        _write_json(output_dir / "method_selection.json", selection)
        _write_json(output_dir / ("final_config.json" if stage == "pilot200" else "frozen_method_config.json"), frozen)
        if stage == "test20":
            _write_json(output_dir / "selected_config.json", frozen)
    _write_csv(
        output_dir / "downstream_1x1_results.csv",
        [{"method": row["variant"], "status": "not_run_gate_required", "checkpoint": ""} for row in (selection or {}).get("selected", [])],
    )
    plot_failures = _plots(output_dir, stage, summary, per_image, threshold_rows, diagnostics)
    _write_json(output_dir / "visualization_failures.json", plot_failures)
    _report(output_dir, stage, summary, selection, audit)
    print(json.dumps({**audit, "selection": selection}, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"V2 evaluation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gbsp_root", "--gbsp-root", dest="gbsp_root")
    parser.add_argument("--threshold_root", "--threshold-root", dest="threshold_root")
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--compare", nargs="+", default=("fixed_058",))
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
