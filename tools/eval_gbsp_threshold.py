#!/usr/bin/env python3
"""Original-size evaluation and audit for GBSP fixed/adaptive thresholds."""

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
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as connected_components
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_thresholding import (  # noqa: E402
    BackgroundQuantileThreshold,
    HigherCriticismThreshold,
    RobustMADThreshold,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
METHODS = (
    "r1_fixed_050",
    "fixed_050",
    "fixed_058",
    "cf_bq95",
    "cf_rmad",
    "cf_brc_hc",
)
REQUESTABLE = METHODS[1:]
METRICS = (
    "S_m",
    "F_beta_w",
    "F_beta_mean",
    "E_mean",
    "MAE",
    "Precision",
    "Recall",
    "Area",
    "IoU",
    "Dice",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ("status",)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in seen:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}: {key}")
        if not Path(str(row.get("cache_path", ""))).is_file():
            raise FileNotFoundError(row.get("cache_path"))
        seen.add(key)
    return rows


def _sample_keys(path: Path) -> list[tuple[str, str]]:
    keys = []
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.replace("/", "\t", 1).split()
        if len(parts) != 2:
            raise ValueError(f"invalid identity at {path}:{line}: {raw!r}")
        keys.append((parts[0], parts[1]))
    if len(keys) != len(set(keys)):
        raise RuntimeError("sample list contains duplicate identities")
    return keys


def _balanced_subset(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0 or limit >= len(rows):
        return rows
    grouped = {dataset: [] for dataset in DATASETS}
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    selected, cursor = [], 0
    while len(selected) < limit:
        progressed = False
        for dataset in DATASETS:
            if cursor < len(grouped[dataset]):
                selected.append(grouped[dataset][cursor])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _load_gt(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _map(payload: dict, field: str, path: Path, unit: bool = False) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    if unit and (float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6):
        raise ValueError(f"{field} escaped [0,1]: {path}")
    return value.clamp(0.0, 1.0) if unit else value


def _resize(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = F.interpolate(
        value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )
    value = F.interpolate(value, size=shape, mode="bilinear", align_corners=False)
    return value.squeeze(0)


def _binary_metrics(context: FastCODContext, mask: torch.Tensor) -> dict:
    values = context.evaluate_many([("candidate", "hard", mask.float())], 0.5)[
        ("candidate", "hard")
    ]
    precision, recall = float(values["Precision"]), float(values["Recall"])
    return {
        "S_m": float(values["S_m"]),
        "F_beta_w": float(values["F_beta_w"]),
        "F_beta_mean": float(values["F_beta_mean"]),
        "E_mean": float(values["E_mean"]),
        "MAE": float(values["MAE"]),
        "Precision": precision,
        "Recall": recall,
        "Area": float(values["Area"]),
        "IoU": float(values["IoU"]),
        "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
    }


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or (
            str(payload.get("dataset")), str(payload.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"crossfit identity mismatch: {path}")
        if bool(payload.get("gt_used_for_generation", True)) or bool(
            payload.get("r1_used_for_generation", True)
        ) or bool(payload.get("target_area_prior_used", True)):
            raise RuntimeError(f"crossfit cache violates GT/R1/area independence: {path}")
        gt_path = Path(task["gt_path"] or payload.get("gt_path", ""))
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)

        # All calibration outputs are loaded before GT is opened.
        minmax = _map(payload, "gbsp_absolute_minmax", path, unit=True)
        p_value = _map(payload, "p_value_map", path, unit=True)
        query_z = _map(payload, "query_z_median", path)
        if float(p_value.min()) <= 0.0:
            raise ValueError(f"p_value_map must be strictly positive: {path}")
        adaptive_patch = {
            "cf_bq95": _map(payload, "bq95_mask", path, unit=True),
            "cf_rmad": _map(payload, "rmad_mask", path, unit=True),
            "cf_brc_hc": _map(payload, "hc_mask", path, unit=True),
        }
        numerical_audit = payload.get("numerical_audit", {})
        if (
            int(numerical_audit.get("fold_count_difference", -1)) > 1
            or int(numerical_audit.get("leakage_count", -1)) != 0
            or float(numerical_audit.get("max_basis_orthonormal_error", float("inf"))) >= 1e-4
            or int(numerical_audit.get("p_monotonicity_violation_count", -1)) != 0
        ):
            raise RuntimeError(f"crossfit numerical audit failed: {path}: {numerical_audit}")
        heldout_sets = payload.get("fold_heldout_background_positions")
        fit_sets = payload.get("fold_fit_background_positions")
        background_count = int(payload.get("background_candidate_count", -1))
        if not isinstance(heldout_sets, list) or not isinstance(fit_sets, list) or len(heldout_sets) != 5 or len(fit_sets) != 5:
            raise ValueError(f"five fit/held-out index sets are required: {path}")
        for fit_index, heldout_index in zip(fit_sets, heldout_sets):
            if not torch.is_tensor(fit_index) or not torch.is_tensor(heldout_index) or bool(
                torch.isin(heldout_index.long(), fit_index.long()).any()
            ):
                raise RuntimeError(f"held-out background leakage detected: {path}")
        if not torch.equal(
            torch.sort(torch.cat(heldout_sets).long()).values,
            torch.arange(background_count),
        ):
            raise RuntimeError(f"background candidates are not held out exactly once: {path}")
        repeated_hc = HigherCriticismThreshold(
            float(task["hc_p_max"]), float(task["hc_fallback_p"]), float(task["eps"])
        ).apply(p_value.reshape(-1), background_count)
        repeated_bq = BackgroundQuantileThreshold(float(task["bq_alpha"])).apply(
            p_value.reshape(-1)
        )
        repeated_rmad = RobustMADThreshold(float(task["rmad_kappa"])).apply(
            query_z.reshape(-1)
        )
        if (
            repeated_hc.k_star != int(payload["hc_k_star"])
            or repeated_hc.threshold != float(payload["hc_p_threshold"])
            or not torch.equal(
                repeated_hc.binary_mask.reshape(1, 37, 37),
                adaptive_patch["cf_brc_hc"].bool(),
            )
            or not torch.equal(
                repeated_bq.binary_mask.reshape(1, 37, 37),
                adaptive_patch["cf_bq95"].bool(),
            )
            or not torch.equal(
                repeated_rmad.binary_mask.reshape(1, 37, 37),
                adaptive_patch["cf_rmad"].bool(),
            )
        ):
            raise RuntimeError(f"cached threshold outputs are not reproducible: {path}")
        source_dabe = Path(str(payload.get("source_dabe_path", "")))
        if not source_dabe.is_file():
            raise FileNotFoundError(source_dabe)
        dabe = torch_load(source_dabe, map_location="cpu")
        if not isinstance(dabe, dict) or (
            str(dabe.get("dataset")), str(dabe.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"R1 source identity mismatch: {source_dabe}")
        r1 = dabe.get("residual_pass1_37")
        if not torch.is_tensor(r1) or tuple(r1.shape) != (1, 37, 37):
            raise ValueError(f"residual_pass1_37 missing: {source_dabe}")
        r1 = r1.detach().cpu().float().contiguous()
        if not bool(torch.isfinite(r1).all()):
            raise ValueError(f"R1 contains NaN/Inf: {source_dabe}")

        gt = _load_gt(gt_path)
        shape = tuple(gt.shape[-2:])
        masks = {
            "r1_fixed_050": (_resize(r1, shape) > 0.50).float(),
            "fixed_050": (_resize(minmax, shape) > 0.50).float(),
            "fixed_058": (_resize(minmax, shape) > 0.58).float(),
            **{
                method: (_resize(mask, shape) > 0.50).float()
                for method, mask in adaptive_patch.items()
            },
        }
        requested = ("r1_fixed_050", *task["methods"])
        context = FastCODContext(gt)
        rows = []
        for method in requested:
            metric = _binary_metrics(context, masks[method])
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "cache_path": str(path),
                    "image_path": task["image_path"] or payload.get("image_path", ""),
                    "gt_path": str(gt_path),
                    "method": method,
                    **metric,
                    "empty_mask": int(float(adaptive_patch.get(method, masks[method]).mean()) == 0.0),
                    "area_over_50": int(float(adaptive_patch.get(method, masks[method]).mean()) > 0.5),
                }
            )
        by_method = {row["method"]: row for row in rows}
        fixed = by_method["fixed_058"]
        for row in rows:
            row["delta_F_beta_w_vs_fixed_058"] = row["F_beta_w"] - fixed["F_beta_w"]
            row["delta_Precision_vs_fixed_058"] = row["Precision"] - fixed["Precision"]
            row["delta_Recall_vs_fixed_058"] = row["Recall"] - fixed["Recall"]

        fold_scale = payload.get("fold_scale_mad")
        if not torch.is_tensor(fold_scale) or fold_scale.numel() != 5:
            raise ValueError(f"fold_scale_mad must have five values: {path}")
        equivalent_mm = payload.get("equivalent_minmax_threshold", {})
        equivalent_raw = payload.get("equivalent_raw_threshold", {})
        patch_areas = payload.get("foreground_area", {})
        thresholds = {
            "r1_fixed_050": (0.50, "r1_score"),
            "fixed_050": (0.50, "minmax"),
            "fixed_058": (0.58, "minmax"),
            "cf_bq95": (0.05, "p_value"),
            "cf_rmad": (float(task["rmad_kappa"]), "robust_z"),
            "cf_brc_hc": (float(payload["hc_p_threshold"]), "p_value"),
        }
        threshold_rows = []
        for method in requested:
            value, domain = thresholds[method]
            threshold_rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    "threshold": value,
                    "threshold_domain": domain,
                    "equivalent_minmax_threshold": (
                        value if domain == "minmax" else equivalent_mm.get(method)
                    ),
                    "equivalent_raw_threshold": equivalent_raw.get(method),
                    "foreground_area_patch": float(
                        patch_areas.get(method, adaptive_patch.get(method, masks[method]).mean())
                    ),
                    "background_candidate_count": background_count,
                    "background_log_residual_mad": float(fold_scale.float().mean()),
                    "hc_fallback": int(bool(payload["hc_fallback"])),
                    "numerical_fallback": int(bool(payload["numerical_fallback"])),
                }
            )
        hc_native = masks["cf_brc_hc"]
        gt_bool = gt > 0.5
        boundary_touch = bool(
            gt_bool[:, 0, :].any()
            or gt_bool[:, -1, :].any()
            or gt_bool[:, :, 0].any()
            or gt_bool[:, :, -1].any()
        )
        component_count = int(connected_components(gt_bool.squeeze(0).numpy())[1])
        diagnostic = {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(path),
            "image_path": task["image_path"] or payload.get("image_path", ""),
            "gt_path": str(gt_path),
            "gt_area": float(gt.mean()),
            "gt_component_count": component_count,
            "gt_boundary_touch": int(boundary_touch),
            "background_log_residual_mad": float(fold_scale.float().mean()),
            "fold_count_difference": int(numerical_audit["fold_count_difference"]),
            "max_basis_orthonormal_error": float(
                numerical_audit["max_basis_orthonormal_error"]
            ),
            "p_min": float(p_value.min()),
            "p_max": float(p_value.max()),
            "hc_equivalent_minmax_threshold": equivalent_mm.get("cf_brc_hc"),
            "hc_foreground_area_patch": float(patch_areas["cf_brc_hc"]),
            "fixed058_foreground_area_patch": float(patch_areas["fixed_058"]),
            "hc_fixed058_mask_iou": float(
                ((hc_native > 0.5) & (masks["fixed_058"] > 0.5)).sum()
            )
            / max(
                int(((hc_native > 0.5) | (masks["fixed_058"] > 0.5)).sum()), 1
            ),
            "hc_delta_F_beta_w": by_method["cf_brc_hc"]["F_beta_w"] - fixed["F_beta_w"],
            "hc_delta_Precision": by_method["cf_brc_hc"]["Precision"] - fixed["Precision"],
            "hc_delta_Recall": by_method["cf_brc_hc"]["Recall"] - fixed["Recall"],
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
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    methods = tuple(dict.fromkeys(row["method"] for row in rows))
    extra = (
        "empty_mask",
        "area_over_50",
        "delta_F_beta_w_vs_fixed_058",
        "delta_Precision_vs_fixed_058",
        "delta_Recall_vs_fixed_058",
    )
    for dataset in DATASETS:
        for method in methods:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if subset:
                output.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "method": method,
                        "num_samples": len(subset),
                        **{field: _mean(row[field] for row in subset) for field in (*METRICS, *extra)},
                    }
                )
    for method in methods:
        subset = [row for row in output if row["method"] == method]
        output.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                "num_samples": sum(row["num_samples"] for row in subset),
                **{field: _mean(row[field] for row in subset) for field in (*METRICS, *extra)},
            }
        )
    return output


def _correlation(threshold_rows: list[dict]) -> dict:
    rows = [
        row
        for row in threshold_rows
        if row["method"] == "cf_brc_hc"
        and row["equivalent_minmax_threshold"] is not None
    ]
    threshold = np.asarray([float(row["equivalent_minmax_threshold"]) for row in rows])
    output = {}
    for field in (
        "background_candidate_count",
        "background_log_residual_mad",
        "foreground_area_patch",
    ):
        values = np.asarray([float(row[field]) for row in rows])
        result = spearmanr(threshold, values) if len(rows) >= 3 else None
        output[f"hc_mm_threshold_vs_{field}_spearman"] = (
            float(result.statistic) if result is not None and math.isfinite(float(result.statistic)) else None
        )
    return output


def _threshold_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in (*DATASETS, "ALL"):
        for method in METHODS:
            subset = [
                row
                for row in rows
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            values = [
                float(row["equivalent_minmax_threshold"])
                for row in subset
                if row["equivalent_minmax_threshold"] is not None
            ]
            if not subset:
                continue
            array = np.asarray(values, dtype=np.float64)
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "num_samples": len(subset),
                    "num_nonempty": len(values),
                    "mean": float(array.mean()) if len(array) else None,
                    "std": float(array.std()) if len(array) else None,
                    "q10": float(np.quantile(array, 0.10)) if len(array) else None,
                    "q25": float(np.quantile(array, 0.25)) if len(array) else None,
                    "median": float(np.quantile(array, 0.50)) if len(array) else None,
                    "q75": float(np.quantile(array, 0.75)) if len(array) else None,
                    "q90": float(np.quantile(array, 0.90)) if len(array) else None,
                    "mean_delta_vs_058": float(array.mean() - 0.58) if len(array) else None,
                }
            )
    return output


def _plot_outputs(
    output_dir: Path,
    summary: list[dict],
    rows: list[dict],
    threshold_rows: list[dict],
    diagnostics: list[dict],
) -> list[dict]:
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
        except Exception as error:  # plots are diagnostics; preserve metric outputs
            failures.append({"plot": name, "error": repr(error), "traceback": traceback.format_exc()})

    hc = [row for row in threshold_rows if row["method"] == "cf_brc_hc"]
    hc_nonempty = [row for row in hc if row["equivalent_minmax_threshold"] is not None]
    cache_paths = {
        (row["dataset"], row["stem"]): row["cache_path"]
        for row in diagnostics
    }

    def threshold_hist(plt, path):
        values = [float(row["equivalent_minmax_threshold"]) for row in hc_nonempty]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=40, color="#3b82f6", alpha=0.85)
        ax.axvline(0.58, color="#dc2626", linestyle="--", label="fixed 0.58")
        ax.set(xlabel="HC equivalent Min-Max threshold", ylabel="Images")
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def threshold_dataset(plt, path):
        data = [[float(row["equivalent_minmax_threshold"]) for row in hc_nonempty if row["dataset"] == ds] for ds in DATASETS]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(data, labels=DATASETS, showfliers=False)
        ax.axhline(0.58, color="#dc2626", linestyle="--")
        ax.set_ylabel("Equivalent Min-Max threshold")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def scatter(field, xlabel):
        def draw(plt, path):
            x = [float(row[field]) for row in hc_nonempty]
            y = [float(row["equivalent_minmax_threshold"]) for row in hc_nonempty]
            fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(x, y, s=5, alpha=0.25)
            ax.axhline(0.58, color="#dc2626", linestyle="--")
            ax.set(xlabel=xlabel, ylabel="HC equivalent Min-Max threshold")
            fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        return draw

    def pra(plt, path):
        macro = [row for row in summary if row["scope"] == "dataset_macro"]
        x = np.arange(len(macro)); width = 0.25
        fig, ax = plt.subplots(figsize=(10, 4))
        for offset, field in ((-width, "Precision"), (0, "Recall"), (width, "Area")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, [row["method"] for row in macro], rotation=20, ha="right")
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def delta_hist(plt, path):
        values = [row["delta_F_beta_w_vs_fixed_058"] for row in rows if row["method"] == "cf_brc_hc"]
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(values, bins=50, color="#10b981")
        ax.axvline(0, color="black", linewidth=1)
        ax.set(xlabel="Per-image Fbeta_w delta vs fixed 0.58", ylabel="Images")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def hc_examples(plt, path):
        ordered = sorted(hc, key=lambda row: float(row["threshold"]))
        selected = [ordered[index] for index in sorted(set([0, len(ordered)//3, 2*len(ordered)//3, len(ordered)-1]))]
        fig, axes = plt.subplots(len(selected), 1, figsize=(7, 2.5 * len(selected)), squeeze=False)
        for ax, row in zip(axes[:, 0], selected):
            payload = torch_load(cache_paths[(row["dataset"], row["stem"])], map_location="cpu")
            p = payload["hc_sorted_p"].numpy(); curve = payload["hc_score_curve"].numpy()
            candidate = payload["hc_candidate_mask"].numpy().astype(bool)
            ax.plot(p[candidate], curve[candidate]); ax.axvline(payload["hc_p_threshold"], color="red", linestyle="--")
            ax.set_title(f"{row['dataset']}/{row['stem']} k*={payload['hc_k_star']}")
            ax.set(xlabel="sorted p", ylabel="HC")
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def visual_grid(selected: list[dict], plt, path):
        if not selected:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No samples", ha="center"); ax.axis("off")
            fig.savefig(path, dpi=120); plt.close(fig); return
        fig, axes = plt.subplots(len(selected), 11, figsize=(30, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            with Image.open(item["image_path"]) as image:
                rgb = np.array(image.convert("RGB"), copy=True)
            with Image.open(item["gt_path"]) as image:
                gt = np.array(image.convert("L"), copy=True)
            bc = np.zeros(37 * 37, dtype=np.float32)
            bc[payload["background_indices"].numpy()] = 1
            panels = (
                (rgb, "RGB"), (gt, "GT"), (bc.reshape(37, 37), "Full BC"),
                (payload["gbsp_absolute_raw"][0].numpy(), "GBSP raw"),
                (payload["gbsp_absolute_minmax"][0].numpy(), "GBSP MinMax"),
            )
            for ax, (array, title) in zip(row_axes[:5], panels):
                ax.imshow(array, cmap=None if array.ndim == 3 else "gray"); ax.set_title(title); ax.axis("off")
            row_axes[5].hist(payload["background_oof_raw_residual"].numpy(), bins=30)
            row_axes[5].set_title("OOF bg residual")
            row_axes[6].imshow(payload["anomaly_score_map"][0].numpy(), cmap="magma"); row_axes[6].set_title("-log10(p)"); row_axes[6].axis("off")
            candidate = payload["hc_candidate_mask"].numpy().astype(bool)
            row_axes[7].plot(payload["hc_sorted_p"].numpy()[candidate], payload["hc_score_curve"].numpy()[candidate]); row_axes[7].set_title("HC curve")
            for ax, field, title in zip(row_axes[8:], ("fixed_058_mask", "bq95_mask", "hc_mask"), ("fixed 0.58", "CF-BQ95", "CF-BRC-HC")):
                ax.imshow(payload[field][0].numpy(), cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
            row_axes[0].set_ylabel(
                f"{item['selection_reason']}\n{item['dataset']}/{item['stem']}\n"
                f"dF={item['hc_delta_F_beta_w']:+.3f}"
            )
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    def unique_reasons(candidates):
        chosen, seen = [], set()
        for reason, row in candidates:
            key = (row["dataset"], row["stem"])
            if key in seen:
                continue
            chosen.append({**row, "selection_reason": reason})
            seen.add(key)
        return chosen[:5]

    conservative = sorted(diagnostics, key=lambda row: row["hc_delta_Recall"], reverse=True)
    suppression = sorted(
        diagnostics,
        key=lambda row: (
            row["hc_delta_Precision"],
            row["fixed058_foreground_area_patch"] - row["hc_foreground_area_patch"],
        ),
        reverse=True,
    )
    agreement = sorted(diagnostics, key=lambda row: row["hc_fixed058_mask_iou"], reverse=True)
    success = unique_reasons(
        [
            ("fixed058_conservative_hc_recovers", conservative[0]),
            ("fixed058_false_positive_hc_suppresses", suppression[0]),
            ("fixed058_hc_agree", agreement[0]),
            ("small_target", min(diagnostics, key=lambda row: row["gt_area"])),
            ("large_target", max(diagnostics, key=lambda row: row["gt_area"])),
            *[("high_fbw_gain", row) for row in sorted(diagnostics, key=lambda row: row["hc_delta_F_beta_w"], reverse=True)],
        ]
    )
    aquatic = [
        row
        for row in diagnostics
        if "aquatic" in row["stem"].lower()
        or any(token in row["stem"].lower() for token in ("fish", "crab", "turtle", "shrimp"))
    ]
    boundary = [row for row in diagnostics if row["gt_boundary_touch"]]
    multi = [row for row in diagnostics if row["gt_component_count"] > 1]
    failure = unique_reasons(
        [
            ("hc_failure", min(diagnostics, key=lambda row: row["hc_delta_F_beta_w"])),
            ("multi_target", max(multi or diagnostics, key=lambda row: row["gt_component_count"])),
            ("complex_background_high_mad", max(diagnostics, key=lambda row: row["background_log_residual_mad"])),
            ("aquatic_background", (aquatic or diagnostics)[0]),
            ("boundary_touch", (boundary or diagnostics)[0]),
            *[("low_fbw_delta", row) for row in sorted(diagnostics, key=lambda row: row["hc_delta_F_beta_w"])],
        ]
    )
    _write_csv(
        output_dir / "visualization_selection.csv",
        [
            {
                "group": group,
                "selection_reason": row["selection_reason"],
                "dataset": row["dataset"],
                "stem": row["stem"],
                "hc_delta_F_beta_w": row["hc_delta_F_beta_w"],
            }
            for group, selected in (("success", success), ("failure_and_categories", failure))
            for row in selected
        ],
    )
    run("threshold_distribution.png", threshold_hist)
    run("threshold_by_dataset.png", threshold_dataset)
    run("threshold_vs_background_mad.png", scatter("background_log_residual_mad", "Mean fold log-residual MAD"))
    run("threshold_vs_foreground_area.png", scatter("foreground_area_patch", "HC patch foreground area"))
    run("precision_recall_area_comparison.png", pra)
    run("per_image_fbw_delta_vs_fixed058.png", delta_hist)
    run("hc_curve_examples.png", hc_examples)
    run("success_visualizations.png", lambda plt, path: visual_grid(success, plt, path))
    run("failure_visualizations.png", lambda plt, path: visual_grid(failure, plt, path))
    return failures


def _table(rows: list[dict], fields: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" if field in {"dataset", "method"} else "---:" for field in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(f"{float(row[field]):.6f}" if isinstance(row.get(field), (float, np.floating)) else str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def _report(output_dir: Path, summary: list[dict], metadata: dict, threshold_summary: list[dict]) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    hc_threshold = next((row for row in threshold_summary if row["dataset"] == "ALL" and row["method"] == "cf_brc_hc"), None)
    lines = [
        "# GBSP Threshold Report",
        "",
        "- Calibration uses raw residuals and out-of-fold background statistics only.",
        "- No GT, R1 mask/area, target-area prior, dataset threshold, or threshold search is used.",
        "- Fixed 0.50/0.58 remain image-MinMax baselines and are not part of CF calibration.",
        "",
        "## Macro metrics",
        "",
        _table(macro, ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")),
        "",
        "## Audit",
        "",
        f"- Valid/failed: {metadata['num_valid']}/{metadata['num_failed']}",
        f"- HC fallback ratio: {metadata['hc_fallback_ratio']:.6f}",
        f"- Numerical fallback ratio: {metadata['numerical_fallback_ratio']:.6f}",
        f"- HC empty/large-mask ratio: {metadata['hc_empty_ratio']:.6f}/{metadata['hc_large_ratio']:.6f}",
        f"- Pilot/full gate status: **{metadata['gate']['status']}**",
        f"- Gate reason: {metadata['gate']['reason']}",
    ]
    if hc_threshold:
        lines.extend(
            [
                f"- HC equivalent Min-Max mean/median: {hc_threshold['mean']}/{hc_threshold['median']}",
                f"- HC equivalent Min-Max q10/q25/q75/q90: {hc_threshold['q10']}/{hc_threshold['q25']}/{hc_threshold['q75']}/{hc_threshold['q90']}",
            ]
        )
    (output_dir / "GBSP_THRESHOLD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(_resolve(args.config))
    root = _resolve(args.crossfit_root)
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.methods) - set(REQUESTABLE))
    if unknown or "fixed_058" not in args.methods or "cf_brc_hc" not in args.methods:
        raise ValueError(f"methods must include fixed_058/cf_brc_hc; unknown={unknown}")
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
    tasks = []
    for row in rows:
        tasks.append(
            {
                "dataset": row["dataset"], "stem": row["stem"],
                "cache_path": row["cache_path"], "image_path": row.get("image_path", ""),
                "gt_path": row.get("gt_path", ""), "methods": tuple(args.methods),
                "rmad_kappa": float(cfg.GBSP_THRESHOLD_RMAD_KAPPA),
                "hc_p_max": float(cfg.GBSP_THRESHOLD_HC_P_MAX),
                "hc_fallback_p": float(cfg.GBSP_THRESHOLD_HC_FALLBACK_P),
                "bq_alpha": float(cfg.GBSP_THRESHOLD_BQ_ALPHA),
                "eps": float(cfg.GBSP_THRESHOLD_EPS),
            }
        )
    started = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP threshold eval {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("threshold evaluation produced no valid samples")
    per_image = [row for result in valid for row in result["rows"]]
    threshold_rows = [row for result in valid for row in result["threshold_rows"]]
    diagnostics = [result["diagnostic"] for result in valid]
    summary = _aggregate(per_image)
    threshold_summary = _threshold_summary(threshold_rows)
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    hc, fixed = macro["cf_brc_hc"], macro["fixed_058"]
    hc_rows = [row for row in threshold_rows if row["method"] == "cf_brc_hc"]
    fallback_ratio = _mean(row["hc_fallback"] for row in hc_rows)
    numerical_ratio = _mean(row["numerical_fallback"] for row in hc_rows)
    empty_ratio = _mean(row["empty_mask"] for row in per_image if row["method"] == "cf_brc_hc")
    large_ratio = _mean(row["area_over_50"] for row in per_image if row["method"] == "cf_brc_hc")
    if fallback_ratio > 0.01:
        gate = {"status": "STOP_BEFORE_DOWNSTREAM", "reason": "HC fallback ratio exceeds 1%"}
    elif empty_ratio > 0.01 or large_ratio > 0.01:
        gate = {"status": "STOP_BEFORE_DOWNSTREAM", "reason": "empty or >50% mask ratio exceeds 1%"}
    elif hc["F_beta_w"] - fixed["F_beta_w"] < -0.008 or hc["MAE"] - fixed["MAE"] > 0.005:
        gate = {"status": "STOP_BEFORE_DOWNSTREAM", "reason": "pilot quality gate versus fixed 0.58 failed"}
    else:
        gate = {"status": "PASS_TO_FULL_OR_DOWNSTREAM", "reason": "all predeclared stability/quality gates passed"}
    counts = Counter(task["dataset"] for task in tasks)
    full = len(valid) == len(tasks) == sum(EXPECTED.values()) and not failures and dict(counts) == EXPECTED
    fixed050_expected = {"S_m": 0.7236768609432043, "F_beta_w": 0.6093427897842387, "E_mean": 0.8088151710888547, "MAE": 0.08971997462709005}
    fixed050 = macro.get("fixed_050", {})
    reproduction_error = {field: abs(float(fixed050.get(field, float("nan"))) - value) for field, value in fixed050_expected.items()} if full else {}
    correlations = _correlation(threshold_rows)
    metadata = {
        "schema": "gbsp_threshold_eval_v1", "num_requested": len(tasks),
        "num_valid": len(valid), "num_failed": len(failures), "dataset_counts": dict(counts),
        "full_formal_evaluation": full, "gt_used_for_calibration": False,
        "r1_used_for_calibration": False, "target_area_prior_used": False,
        "threshold_search_used": False, "hc_fallback_ratio": fallback_ratio,
        "numerical_fallback_ratio": numerical_ratio, "hc_empty_ratio": empty_ratio,
        "hc_large_ratio": large_ratio, "fixed050_reproduction_abs_error": reproduction_error,
        "fixed050_reproduction_pass": bool(full and max(reproduction_error.values()) < 1e-4),
        "crossfit_invariants": {
            "max_fold_count_difference": max(
                int(row["fold_count_difference"]) for row in diagnostics
            ),
            "max_basis_orthonormal_error": max(
                float(row["max_basis_orthonormal_error"]) for row in diagnostics
            ),
            "global_p_min": min(float(row["p_min"]) for row in diagnostics),
            "global_p_max": max(float(row["p_max"]) for row in diagnostics),
            "heldout_once_and_no_leakage": True,
            "threshold_reproduction_exact": True,
        },
        "correlations": correlations, "gate": gate, "wall_seconds": time.perf_counter() - started,
    }
    _write_csv(output_dir / "per_image_metrics.csv", per_image)
    _write_csv(output_dir / "per_dataset_metrics.csv", summary)
    _write_csv(output_dir / "full_threshold_metrics.csv", [row for row in summary if row["scope"] == "dataset_macro"])
    _write_csv(output_dir / "baseline_reproduction.csv", [row for row in summary if row["method"] in {"r1_fixed_050", "fixed_050", "fixed_058"}])
    _write_csv(output_dir / "threshold_distribution.csv", threshold_rows)
    _write_csv(output_dir / "foreground_area_distribution.csv", [{key: row[key] for key in ("dataset", "stem", "method", "foreground_area_patch")} for row in threshold_rows])
    _write_json(output_dir / "fallback_summary.json", {key: metadata[key] for key in ("hc_fallback_ratio", "numerical_fallback_ratio", "hc_empty_ratio", "hc_large_ratio", "gate")})
    _write_json(output_dir / "numerical_audit.json", metadata)
    _write_json(output_dir / "threshold_summary.json", threshold_summary)
    _write_csv(
        output_dir / "downstream_1x1_results.csv",
        [
            {"pseudo_label": "GBSP fixed-0.58", "status": "existing_run_not_imported", "checkpoint": "", "notes": "Import the frozen reference evaluation explicitly."},
            {"pseudo_label": "GBSP CF-BRC-HC", "status": "not_run", "checkpoint": "", "notes": "Run only after the offline gate passes."},
        ],
    )
    plot_failures = _plot_outputs(output_dir, summary, per_image, threshold_rows, diagnostics)
    _write_json(output_dir / "visualization_failures.json", plot_failures)
    _report(output_dir, summary, metadata, threshold_summary)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if failures and args.strict_failures:
        raise RuntimeError(f"threshold evaluation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--crossfit_root", "--crossfit-root", dest="crossfit_root", required=True)
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--methods", nargs="+", default=list(REQUESTABLE))
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--strict_failures", "--strict-failures", dest="strict_failures", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
