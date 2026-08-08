#!/usr/bin/env python3
"""Original-size COD evaluation, gating and audit for GBSP threshold V3."""

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
from scipy.signal import savgol_filter

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
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v3.py"


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


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    values = score.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels = gt.detach().cpu().numpy().reshape(-1) > 0.5
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}
    descending = np.argsort(-values, kind="stable")
    ordered_score, ordered_label = values[descending], labels[descending]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positive_count
    average_precision = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    group_positive = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_size = ends - starts
    group_negative = group_size - group_positive
    negative_below = negative_count - np.cumsum(group_negative)
    pair_wins = np.sum(
        group_positive * (negative_below + 0.5 * group_negative), dtype=np.float64
    )
    return {
        "pixel_AP": average_precision,
        "pixel_AUROC": float(pair_wins / (positive_count * negative_count)),
    }


def _state_areas(result: dict) -> tuple[float | None, float | None, float | None]:
    diagnostics = result.get("diagnostics", {})
    if all(key in diagnostics for key in ("class0_area", "class1_area", "class2_area")):
        return tuple(float(diagnostics[key]) for key in ("class0_area", "class1_area", "class2_area"))
    maps = result.get("continuous_maps_37", {})
    if all(torch.is_tensor(maps.get(key)) for key in ("posterior_c0", "posterior_c1", "posterior_c2")):
        state = torch.stack((maps["posterior_c0"], maps["posterior_c1"], maps["posterior_c2"])).argmax(0)
        return tuple(float((state == index).float().mean()) for index in range(3))
    state = maps.get("state")
    if torch.is_tensor(state):
        return tuple(float((state == index).float().mean()) for index in range(3))
    return None, None, None


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
            forbidden = (
                "gt_used_for_generation",
                "r1_used_for_generation",
                "fixed_058_used_for_generation",
                "target_area_prior_used",
                "fixed_topk_used",
                "dino_forward_used",
                "pca_refit_used",
                "morphology_used",
            )
            if any(bool(payload.get(field, True)) for field in forbidden):
                raise RuntimeError(f"V3 cache violates independence contract: {path}")
            score = _tensor(payload, ("minmax_residual",), path)
            raw = _tensor(payload, ("raw_residual",), path)
            available = payload.get("results", {})
            names = task["methods"] or list(available)
            missing = [name for name in names if name not in available]
            if missing:
                raise KeyError(f"methods missing from {path}: {missing}")
            adaptive_results = {name: available[name] for name in names}
        else:
            score = _tensor(payload, ("absolute_minmax", "gbsp_abs_minmax_37"), path)
            raw = _tensor(payload, ("absolute_raw", "gbsp_abs_raw_37"), path)
            adaptive_results = {}
        bc = payload.get("background_indices")
        if not torch.is_tensor(bc) or bc.ndim != 1:
            raise ValueError(f"background_indices missing: {path}")
        bc = bc.detach().cpu().long().contiguous()

        # All masks are finalized before GT is opened.  Fixed-0.58 exists only
        # in this evaluator as an external audit baseline.
        reference_names = list(dict.fromkeys(("fixed_058", *task["compare"])))
        allowed_references = {"fixed_058", "gbsp_fixed_050", "r1_fixed_050"}
        unknown_references = set(reference_names) - allowed_references
        if unknown_references:
            raise ValueError(f"unknown comparison baselines: {sorted(unknown_references)}")
        reference_scores = {"fixed_058": score}
        patch_masks = {"fixed_058": (score > float(task["reference_threshold"])).float()}
        if "gbsp_fixed_050" in reference_names:
            reference_scores["gbsp_fixed_050"] = score
            patch_masks["gbsp_fixed_050"] = (score > 0.50).float()
        if "r1_fixed_050" in reference_names:
            source_gbsp = payload if not adaptive else torch_load(
                Path(str(payload.get("source_gbsp_path", ""))), map_location="cpu"
            )
            source_dabe = Path(str(source_gbsp.get("source_dabe_path", "")))
            if not source_dabe.is_file():
                raise FileNotFoundError(f"R1 source is missing: {source_dabe}")
            dabe = torch_load(source_dabe, map_location="cpu")
            if (str(dabe.get("dataset")), str(dabe.get("stem"))) != (
                task["dataset"], task["stem"]
            ):
                raise RuntimeError(f"R1 source identity mismatch: {source_dabe}")
            r1 = _tensor(dabe, ("residual_pass1_37",), source_dabe)
            reference_scores["r1_fixed_050"] = r1
            patch_masks["r1_fixed_050"] = (r1 > 0.50).float()
        metadata = {
            "fixed_058": {
                "method_family": "fixed_reference",
                "threshold_high": float(task["reference_threshold"]),
                "threshold_low": None,
                "foreground_area": float(patch_masks["fixed_058"].mean()),
                "bc_selected_as_foreground_ratio": float(
                    patch_masks["fixed_058"].reshape(-1).index_select(0, bc).mean()
                ),
                "empty_mask": bool(float(patch_masks["fixed_058"].mean()) == 0.0),
                "area_over_50pct": bool(float(patch_masks["fixed_058"].mean()) > 0.5),
                "numerical_failure": False,
                "state_areas": (None, None, None),
                "diagnostics": {},
            }
        }
        for name in reference_names:
            if name == "fixed_058":
                continue
            reference_mask = patch_masks[name]
            metadata[name] = {
                "method_family": "fixed_reference",
                "threshold_high": 0.50,
                "threshold_low": None,
                "foreground_area": float(reference_mask.mean()),
                "bc_selected_as_foreground_ratio": float(
                    reference_mask.reshape(-1).index_select(0, bc).mean()
                ) if name == "gbsp_fixed_050" else None,
                "empty_mask": bool(float(reference_mask.mean()) == 0.0),
                "area_over_50pct": bool(float(reference_mask.mean()) > 0.5),
                "numerical_failure": False,
                "state_areas": (None, None, None),
                "diagnostics": {},
            }
        for name, result in adaptive_results.items():
            mask = result.get("mask_37")
            if not torch.is_tensor(mask) or tuple(mask.shape) != (1, 37, 37):
                raise ValueError(f"invalid mask for {name}: {path}")
            mask = mask.detach().cpu().float().contiguous()
            if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"non-binary mask for {name}: {path}")
            patch_masks[name] = mask
            metadata[name] = {
                "method_family": str(result.get("method_family", result.get("method", name))),
                "threshold_high": result.get("threshold_high"),
                "threshold_low": result.get("threshold_low"),
                "foreground_area": float(result.get("foreground_area", mask.mean())),
                "bc_selected_as_foreground_ratio": float(result.get("bc_selected_as_foreground_ratio", 0.0)),
                "empty_mask": bool(result.get("empty_mask", float(mask.mean()) == 0.0)),
                "area_over_50pct": bool(result.get("area_over_50pct", float(mask.mean()) > 0.5)),
                "numerical_failure": bool(result.get("numerical_failure", True)),
                "state_areas": _state_areas(result),
                "diagnostics": _scalar_diagnostics(result.get("diagnostics", {})),
            }

        gt_path = Path(task["gt_path"] or payload.get("gt_path", ""))
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt = _load_gt(gt_path)
        native_rank_scores = {
            name: _resize(reference_scores[name], tuple(gt.shape[-2:])) for name in reference_names
        }
        native_rank_scores.update(
            {name: _resize(score, tuple(gt.shape[-2:])) for name in adaptive_results}
        )
        rank_metric_by_name, shared_rank_cache = {}, {}
        for name, rank_score in native_rank_scores.items():
            cache_key = "r1" if name == "r1_fixed_050" else "gbsp"
            if cache_key not in shared_rank_cache:
                shared_rank_cache[cache_key] = _rank_metrics(rank_score, gt)
            rank_metric_by_name[name] = shared_rank_cache[cache_key]
        posterior_rank_metrics = {}
        for name, result in adaptive_results.items():
            p2 = result.get("continuous_maps_37", {}).get("posterior_c2")
            if torch.is_tensor(p2):
                posterior_rank_metrics[name] = _rank_metrics(
                    _resize(p2.detach().cpu().float(), tuple(gt.shape[-2:])), gt
                )
        native = {
            name: (
                _resize(reference_scores[name], tuple(gt.shape[-2:]))
                > (float(task["reference_threshold"]) if name == "fixed_058" else 0.50)
            ).float()
            for name in reference_names
        }
        native.update(
            {
                name: (_resize(mask, tuple(gt.shape[-2:])) > 0.5).float()
                for name, mask in patch_masks.items()
                if name not in reference_names
            }
        )
        context = FastCODContext(gt)
        rows = []
        for name in (*reference_names, *adaptive_results):
            info = metadata[name]
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "cache_path": str(path),
                    "image_path": task["image_path"] or payload.get("image_path", ""),
                    "gt_path": str(gt_path),
                    "method": name,
                    "method_family": info["method_family"],
                    **_binary_metrics(context, native[name]),
                    **rank_metric_by_name[name],
                    "posterior_pixel_AP": posterior_rank_metrics.get(name, {}).get("pixel_AP"),
                    "posterior_pixel_AUROC": posterior_rank_metrics.get(name, {}).get("pixel_AUROC"),
                    "foreground_area": info["foreground_area"],
                    "class0_area": info["state_areas"][0],
                    "class1_area": info["state_areas"][1],
                    "class2_area": info["state_areas"][2],
                    "empty_mask": int(info["empty_mask"]),
                    "area_over_50": int(info["area_over_50pct"]),
                    "numerical_failure": int(info["numerical_failure"]),
                    "threshold_high": info["threshold_high"],
                    "threshold_low": info["threshold_low"],
                    "equivalent_minmax_threshold": info["threshold_high"],
                    "bc_selected_as_foreground_ratio": info["bc_selected_as_foreground_ratio"],
                    **{
                        ("diagnostic_method" if key == "method" else key): value
                        for key, value in info["diagnostics"].items()
                    },
                }
            )
        fixed = next(row for row in rows if row["method"] == "fixed_058")
        for row in rows:
            row["delta_F_beta_w_vs_fixed_058"] = row["F_beta_w"] - fixed["F_beta_w"]
            row["delta_Precision_vs_fixed_058"] = row["Precision"] - fixed["Precision"]
            row["delta_Recall_vs_fixed_058"] = row["Recall"] - fixed["Recall"]
        return {
            "rows": rows,
            "diagnostic": {
                "dataset": task["dataset"],
                "stem": task["stem"],
                "cache_path": str(path),
                "image_path": task["image_path"] or payload.get("image_path", ""),
                "gt_path": str(gt_path),
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "background_candidate_count": int(bc.numel()),
                "runtime_seconds": time.perf_counter() - started,
                "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            },
        }
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


def _union_fields(rows: list[dict]) -> list[str]:
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _enrich_summary(summary: list[dict], per_image: list[dict]) -> None:
    for aggregate in summary:
        subset = [
            row
            for row in per_image
            if row["method"] == aggregate["method"]
            and (aggregate["dataset"] == "ALL" or row["dataset"] == aggregate["dataset"])
        ]
        aggregate["method_family"] = subset[0]["method_family"]
        aggregate["numerical_failure"] = _mean(row["numerical_failure"] for row in subset)
        aggregate["foreground_area_patch"] = _mean(row["foreground_area"] for row in subset)
        aggregate["bc_selected_as_foreground_ratio"] = _mean(
            row["bc_selected_as_foreground_ratio"] for row in subset
        )
        aggregate["threshold_high"] = _mean(row.get("threshold_high") for row in subset)
        aggregate["threshold_low"] = _mean(row.get("threshold_low") for row in subset)
        aggregate["pixel_AP"] = _mean(row.get("pixel_AP") for row in subset)
        aggregate["pixel_AUROC"] = _mean(row.get("pixel_AUROC") for row in subset)
        aggregate["posterior_pixel_AP"] = _mean(row.get("posterior_pixel_AP") for row in subset)
        aggregate["posterior_pixel_AUROC"] = _mean(row.get("posterior_pixel_AUROC") for row in subset)


def _bootstrap_ci(per_image: list[dict], repetitions: int = 2000) -> list[dict]:
    """Deterministic dataset-stratified paired bootstrap against fixed-0.58."""
    rng = np.random.default_rng(20260806)
    methods = sorted({row["method"] for row in per_image if row["method"] != "fixed_058"})
    lookup = {(row["dataset"], row["stem"], row["method"]): row for row in per_image}
    output = []
    for method in methods:
        for metric in ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area"):
            grouped = {}
            for dataset in DATASETS:
                keys = [
                    (row["dataset"], row["stem"])
                    for row in per_image
                    if row["dataset"] == dataset and row["method"] == method
                ]
                grouped[dataset] = np.asarray(
                    [
                        float(lookup[(*key, method)][metric])
                        - float(lookup[(*key, "fixed_058")][metric])
                        for key in keys
                    ],
                    dtype=np.float64,
                )
            estimates = np.empty(repetitions, dtype=np.float64)
            for index in range(repetitions):
                dataset_means = []
                for dataset in DATASETS:
                    values = grouped[dataset]
                    if values.size:
                        dataset_means.append(float(values[rng.integers(0, values.size, values.size)].mean()))
                estimates[index] = float(np.mean(dataset_means))
            observed = float(np.mean([values.mean() for values in grouped.values() if values.size]))
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "paired_macro_delta": observed,
                    "bootstrap_ci95_low": float(np.quantile(estimates, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(estimates, 0.975)),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": 20260806,
                }
            )
    return output


def _stage(adaptive: bool, count: int, counts: dict, methods: list[str]) -> str:
    if not adaptive:
        return "baseline_test20" if count == 20 else "baseline"
    if count == 20:
        return "test20_hysteresis" if any(method.endswith("_h") for method in methods) else "test20_core"
    if count == 200:
        return "pilot200"
    if count == sum(EXPECTED.values()) and counts == EXPECTED:
        return "full6473"
    return "custom"


def _dataset_deltas(summary: list[dict], method: str) -> dict[str, float]:
    lookup = {(row["dataset"], row["method"]): row for row in summary if row["scope"] == "dataset"}
    return {
        dataset: lookup[(dataset, method)]["F_beta_w"] - lookup[(dataset, "fixed_058")]["F_beta_w"]
        for dataset in DATASETS
    }


def _rank_candidates(evaluations: list[dict], limit: int) -> list[dict]:
    eligible = [row for row in evaluations if row.get("passed")]
    eligible.sort(
        key=lambda row: (
            -row["F_beta_w"],
            -row["Precision"],
            row["MAE"],
            abs(row["Area"] - 0.125833),
            -row.get("per_image_noninferior", 0),
            row.get("numerical_failure_ratio", 1.0),
            {"otgc": 0, "ut_3cp": 1, "multi_otsu_3": 2}.get(row["method_family"], 3),
            row["method"],
        )
    )
    selected, families = [], set()
    for row in eligible:
        if row["method_family"] in families:
            continue
        selected.append(row)
        families.add(row["method_family"])
        if len(selected) == limit:
            break
    return selected


def _core_selection(summary: list[dict], per_image: list[dict], cfg) -> dict:
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    evaluations, hysteresis = [], []
    for method, row in macro.items():
        if method == "fixed_058":
            continue
        images = [item for item in per_image if item["method"] == method]
        numerical = _mean(item["numerical_failure"] for item in images)
        empty = _mean(item["empty_mask"] for item in images)
        large = _mean(item["area_over_50"] for item in images)
        deltas = _dataset_deltas(summary, method)
        direct_gate = cfg.GBSP_V3_TEST20_DIRECT_GATE
        direct_checks = {
            "F_beta_w": row["F_beta_w"] >= direct_gate["F_beta_w_min"],
            "Precision": row["Precision"] >= direct_gate["Precision_min"],
            "Area": direct_gate["Area_min"] <= row["Area"] <= direct_gate["Area_max"],
            "numerical_failure": numerical <= direct_gate["numerical_failure_ratio_max"],
            "empty_mask": empty <= direct_gate["empty_ratio_max"],
            "large_mask": large <= direct_gate["large_ratio_max"],
            "all_datasets_decline": not all(value < 0 for value in deltas.values()),
        }
        direct_pass = all(direct_checks.values())
        trigger = cfg.GBSP_V3_HYSTERESIS_TRIGGER
        hysteresis_allowed = bool(
            not direct_pass
            and row["Precision"] >= trigger["Precision_min"]
            and row["F_beta_w"] >= trigger["F_beta_w_min"]
            and (row["Recall"] < trigger["Recall_max"] or row["Area"] < trigger["Area_max"])
            and row["Area"] <= trigger["forbid_area_over"]
            and row["Precision"] >= trigger["forbid_precision_below"]
            and numerical == 0.0
            and large == 0.0
        )
        if hysteresis_allowed:
            hysteresis.append(f"{method}_h")
        failure_class = None
        if row["Area"] > 0.16 or row["Precision"] < 0.70:
            failure_class = "foreground_inflation"
        elif row["Area"] < 0.05 or empty > 0.20:
            failure_class = "extreme_contraction"
        elif not direct_pass and not hysteresis_allowed:
            failure_class = "gate_failure"
        evaluations.append(
            {
                "method": method,
                "method_family": row["method_family"],
                "passed": direct_pass,
                "hysteresis_allowed": hysteresis_allowed,
                "failure_class": failure_class,
                "failure_reasons": [name for name, passed in direct_checks.items() if not passed],
                "F_beta_w": row["F_beta_w"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "MAE": row["MAE"],
                "Area": row["Area"],
                "numerical_failure_ratio": numerical,
                "empty_ratio": empty,
                "large_ratio": large,
                "per_image_noninferior": sum(item["delta_F_beta_w_vs_fixed_058"] >= 0 for item in images),
                "dataset_F_beta_w_delta": deltas,
            }
        )
    selected = _rank_candidates(evaluations, 2)
    status = "HYSTERESIS_REQUIRED" if hysteresis else ("PASS" if selected else "STOP")
    return {
        "stage": "test20_core",
        "status": status,
        "evaluations": evaluations,
        "hysteresis_candidates": hysteresis,
        "selected": selected,
        "selected_methods": [row["method"] for row in selected],
    }


def _hysteresis_selection(summary: list[dict], per_image: list[dict], cfg) -> dict:
    prior_path = _resolve(Path(cfg.GBSP_V3_OUTPUT_ROOT) / "eval_test20/method_selection.json")
    if not prior_path.is_file():
        raise RuntimeError(f"Core A1 selection is missing: {prior_path}")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_by_method = {row["method"]: row for row in prior.get("evaluations", [])}
    pool = [row for row in prior.get("evaluations", []) if row.get("passed")]
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    current = []
    for method, row in macro.items():
        if method == "fixed_058":
            continue
        family = row["method_family"]
        core = prior_by_method.get(family)
        if core is None or method not in prior.get("hysteresis_candidates", []):
            raise RuntimeError(f"Hysteresis result lacks an eligible Core: {method}")
        images = [item for item in per_image if item["method"] == method]
        gate = cfg.GBSP_V3_HYSTERESIS_GATE
        checks = {
            "F_beta_w_improved": row["F_beta_w"] > core["F_beta_w"],
            "Precision_preserved": row["Precision"] >= core["Precision"] - gate["Precision_drop_max"],
            "Area": row["Area"] <= gate["Area_max"],
            "hard_area_stop": row["Area"] <= gate["hard_stop_area"],
            "hard_precision_stop": row["Precision"] >= core["Precision"] - gate["hard_stop_precision_drop"],
            "numerical_failure": _mean(item["numerical_failure"] for item in images) == 0.0,
        }
        current.append(
            {
                "method": method,
                "method_family": family,
                "passed": all(checks.values()),
                "failure_reasons": [name for name, passed in checks.items() if not passed],
                "F_beta_w": row["F_beta_w"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "MAE": row["MAE"],
                "Area": row["Area"],
                "numerical_failure_ratio": _mean(item["numerical_failure"] for item in images),
                "per_image_noninferior": sum(item["delta_F_beta_w_vs_fixed_058"] >= 0 for item in images),
                "dataset_F_beta_w_delta": _dataset_deltas(summary, method),
            }
        )
    selected = _rank_candidates(pool + current, 2)
    return {
        "stage": "test20_hysteresis",
        "status": "PASS" if selected else "STOP",
        "core_evaluations": prior.get("evaluations", []),
        "hysteresis_evaluations": current,
        "selected": selected,
        "selected_methods": [row["method"] for row in selected],
        "hysteresis_candidates": [],
    }


def _pilot_selection(summary: list[dict], per_image: list[dict], cfg) -> dict:
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    evaluations = []
    for method, row in macro.items():
        if method == "fixed_058":
            continue
        images = [item for item in per_image if item["method"] == method]
        deltas = _dataset_deltas(summary, method)
        gate = cfg.GBSP_V3_PILOT200_GATE
        threshold_values = [
            float(item["threshold_high"])
            for item in images
            if item.get("threshold_high") is not None and math.isfinite(float(item["threshold_high"]))
        ]
        checks = {
            "F_beta_w": row["F_beta_w"] >= gate["F_beta_w_min"],
            "MAE": row["MAE"] <= gate["MAE_max"],
            "Precision": row["Precision"] >= gate["Precision_min"],
            "Area": gate["Area_min"] <= row["Area"] <= gate["Area_max"],
            "all_valid": len(images) == 200 and _mean(item["numerical_failure"] for item in images) == 0.0,
            "empty_mask": _mean(item["empty_mask"] for item in images) <= gate["empty_ratio_max"],
            "large_mask": _mean(item["area_over_50"] for item in images) <= gate["large_ratio_max"],
            "COD10K": deltas["TE-COD10K"] >= -gate["COD10K_F_beta_w_drop_max"],
            "NC4K": deltas["NC4K"] >= -gate["NC4K_F_beta_w_drop_max"],
            "all_datasets_decline": not all(value < 0 for value in deltas.values()),
            "threshold_not_below_040": not threshold_values or float(np.median(threshold_values)) >= 0.40,
        }
        evaluations.append(
            {
                "method": method,
                "method_family": row["method_family"],
                "passed": all(checks.values()),
                "failure_reasons": [name for name, passed in checks.items() if not passed],
                "F_beta_w": row["F_beta_w"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "MAE": row["MAE"],
                "Area": row["Area"],
                "numerical_failure_ratio": _mean(item["numerical_failure"] for item in images),
                "per_image_noninferior": sum(item["delta_F_beta_w_vs_fixed_058"] >= 0 for item in images),
                "dataset_F_beta_w_delta": deltas,
            }
        )
    selected = _rank_candidates(evaluations, 1)
    return {
        "stage": "pilot200",
        "status": "PASS" if len(selected) == 1 else "STOP",
        "evaluations": evaluations,
        "selected": selected,
        "selected_methods": [row["method"] for row in selected],
    }


def _frozen_config(selection: dict, stage: str) -> dict:
    return {
        "schema": "gbsp_threshold_v3_frozen_config",
        "source_stage": stage,
        "selected_methods": selection.get("selected_methods", []),
        "hysteresis_candidates": selection.get("hysteresis_candidates", []),
        "frozen_parameters": {
            "multi_otsu_classes": 3,
            "multi_otsu_nbins": 256,
            "otgc_anchor_strength": 1.0,
            "otgc_shared_variance": True,
            "otgc_posterior_threshold": 0.5,
            "ut3cp_window": 21,
            "ut3cp_polyorder": 2,
            "ut3cp_min_lengths": [8, 24, 64],
        },
        "gt_used_in_formula": False,
        "r1_used_in_formula": False,
        "fixed_058_used_in_formula": False,
        "target_area_prior_used": False,
        "parameters_may_not_be_modified_after_freeze": True,
    }


def _distribution_rows(per_image: list[dict], fields: tuple[str, ...]) -> list[dict]:
    return [
        {key: row.get(key) for key in ("dataset", "stem", "method", *fields)}
        for row in per_image
        if row["method"] != "fixed_058"
    ]


def _plots(output_dir: Path, stage: str, summary: list[dict], rows: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return [{"plot": "all", "error": repr(error)}]
    macro = [row for row in summary if row["scope"] == "dataset_macro"]

    def run(name, function):
        try:
            function(plt, output_dir / name)
        except Exception as error:
            failures.append({"plot": name, "error": repr(error), "traceback": traceback.format_exc()})

    def bars(plt, path):
        x, width = np.arange(len(macro)), 0.2
        fig, ax = plt.subplots(figsize=(max(8, len(macro) * 1.6), 4))
        for offset, field in ((-1.5 * width, "S_m"), (-0.5 * width, "F_beta_w"), (0.5 * width, "E_mean"), (1.5 * width, "MAE")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, [row["method"] for row in macro], rotation=25, ha="right")
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def pra(plt, path):
        x, width = np.arange(len(macro)), 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(macro) * 1.6), 4))
        for offset, field in ((-width, "Precision"), (0, "Recall"), (width, "Area")):
            ax.bar(x + offset, [row[field] for row in macro], width, label=field)
        ax.set_xticks(x, [row["method"] for row in macro], rotation=25, ha="right")
        ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def thresholds(plt, path):
        methods = [row["method"] for row in macro if row["method"] != "fixed_058"]
        data = [[float(item["threshold_high"]) for item in rows if item["method"] == method and item.get("threshold_high") is not None] for method in methods]
        fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.6), 4))
        if data:
            ax.boxplot(data, labels=methods, showfliers=False)
        ax.set_ylabel("High threshold"); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def states(plt, path):
        methods = [row["method"] for row in macro if row["method"] != "fixed_058"]
        fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.6), 4))
        bottom = np.zeros(len(methods))
        for field, label, color in (("class0_area", "C0", "#60a5fa"), ("class1_area", "C1", "#fbbf24"), ("class2_area", "C2", "#ef4444")):
            values = [_mean(item.get(field) for item in rows if item["method"] == method) for method in methods]
            values = np.nan_to_num(values)
            ax.bar(methods, values, bottom=bottom, label=label, color=color); bottom += values
        ax.legend(); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def deltas(plt, path):
        methods = [row["method"] for row in macro if row["method"] != "fixed_058"]
        data = [[item["delta_F_beta_w_vs_fixed_058"] for item in rows if item["method"] == method] for method in methods]
        fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.6), 4))
        if data:
            ax.boxplot(data, labels=methods, showfliers=False)
        ax.axhline(0, color="black", linewidth=1); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def datasets(plt, path):
        methods = [row["method"] for row in macro]
        fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.6), 4))
        x, width = np.arange(len(methods)), 0.18
        for index, dataset in enumerate(DATASETS):
            values = [next(row["F_beta_w"] for row in summary if row["scope"] == "dataset" and row["dataset"] == dataset and row["method"] == method) for method in methods]
            ax.bar(x + (index - 1.5) * width, values, width, label=dataset)
        ax.set_xticks(x, methods, rotation=25, ha="right"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

    def visual(plt, path, success: bool):
        adaptive = [row for row in rows if row["method"] != "fixed_058"]
        ordered = sorted(adaptive, key=lambda row: row["delta_F_beta_w_vs_fixed_058"], reverse=success)
        selected, seen = [], set()
        for item in ordered:
            key = (item["dataset"], item["stem"])
            if key not in seen:
                seen.add(key); selected.append(item)
            if len(selected) == 6:
                break
        if not selected:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No samples", ha="center"); ax.axis("off")
            fig.savefig(path); plt.close(fig); return
        fig, axes = plt.subplots(len(selected), 7, figsize=(20, 3 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"][item["method"]]
            with Image.open(item["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(item["gt_path"]) as image:
                gt = np.asarray(image.convert("L"))
            bc = np.zeros(37 * 37); bc[payload["background_indices"].numpy()] = 1
            maps = result.get("continuous_maps_37", {})
            if all(key in maps for key in ("posterior_c0", "posterior_c1", "posterior_c2")):
                state = torch.stack((maps["posterior_c0"], maps["posterior_c1"], maps["posterior_c2"])).argmax(0)[0].numpy()
            elif "state" in maps:
                state = maps["state"][0].numpy()
            else:
                state = result["mask_37"][0].numpy() * 2
            panels = (
                (rgb, "RGB"), (gt, "GT"), (bc.reshape(37, 37), "Full BC"),
                (payload["minmax_residual"][0].numpy(), "GBSP residual"),
                ((payload["minmax_residual"] > 0.58)[0].numpy(), "fixed-0.58"),
                (result["mask_37"][0].numpy(), item["method"]), (state, "tri-state"),
            )
            for ax, (array, title) in zip(row_axes, panels):
                ax.imshow(array, cmap=None if array.ndim == 3 else "viridis"); ax.set_title(title); ax.axis("off")
            row_axes[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndFw={item['delta_F_beta_w_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    def otgc_examples(plt, path, success: bool):
        candidates = [row for row in rows if row["method"] == "otgc"]
        candidates.sort(key=lambda row: row["delta_F_beta_w_vs_fixed_058"], reverse=success)
        selected = candidates[:4]
        if not selected:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No OTGC samples", ha="center"); ax.axis("off")
            fig.savefig(path); plt.close(fig); return
        fig, axes = plt.subplots(len(selected), 5, figsize=(18, 3.2 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"]["otgc"]
            diagnostics = result["diagnostics"]
            score = payload["minmax_residual"].reshape(-1).numpy().clip(1e-4, 1 - 1e-4)
            y = np.log(score) - np.log1p(-score)
            grid = np.linspace(np.quantile(y, 0.005), np.quantile(y, 0.995), 500)
            row_axes[0].hist(y, bins=50, density=True, alpha=0.35, color="gray")
            mixture = np.zeros_like(grid)
            for index, color in enumerate(("#2563eb", "#f59e0b", "#dc2626")):
                mean = float(diagnostics[f"mu{index}"])
                sigma = float(diagnostics["sigma_shared"])
                weight = float(diagnostics[f"pi{index}"])
                density = weight * np.exp(-0.5 * ((grid - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                mixture += density; row_axes[0].plot(grid, density, color=color)
            row_axes[0].plot(grid, mixture, color="black", linewidth=1.5)
            row_axes[0].set_title(f"Logit mixture, dBIC={float(diagnostics['delta_bic_3_vs_2']):.1f}")
            maps = result["continuous_maps_37"]
            for index, key in enumerate(("posterior_c0", "posterior_c1", "posterior_c2"), 1):
                row_axes[index].imshow(maps[key][0].numpy(), cmap="magma", vmin=0, vmax=1)
                row_axes[index].set_title(key); row_axes[index].axis("off")
            ordered = np.argsort(score, kind="stable")
            row_axes[4].plot(score[ordered], maps["posterior_c2"].reshape(-1).numpy()[ordered])
            row_axes[4].axhline(0.5, color="red", linestyle="--")
            row_axes[4].set(xlabel="Min-Max residual", ylabel="P(C2)")
            row_axes[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndFw={item['delta_F_beta_w_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    def ut3cp_examples(plt, path, success: bool):
        candidates = [row for row in rows if row["method"] == "ut_3cp"]
        candidates.sort(key=lambda row: row["delta_F_beta_w_vs_fixed_058"], reverse=success)
        selected = candidates[:4]
        if not selected:
            fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No UT-3CP samples", ha="center"); ax.axis("off")
            fig.savefig(path); plt.close(fig); return
        fig, axes = plt.subplots(len(selected), 3, figsize=(14, 3.2 * len(selected)), squeeze=False)
        for row_axes, item in zip(axes, selected):
            payload = torch_load(item["cache_path"], map_location="cpu")
            result = payload["results"]["ut_3cp"]
            diagnostics = result["diagnostics"]
            ordered = np.sort(payload["minmax_residual"].reshape(-1).numpy())[::-1]
            x = np.log((np.arange(1, ordered.size + 1) - 0.5) / ordered.size)
            smooth = savgol_filter(ordered, 21, 2, mode="interp")
            row_axes[0].plot(x, ordered, alpha=0.45, label="raw")
            row_axes[0].plot(x, smooth, label="smooth")
            for key, color in (("k1", "red"), ("k2", "orange")):
                row_axes[0].axvline(x[int(diagnostics[key]) - 1], color=color, linestyle="--")
            row_axes[0].legend(); row_axes[0].set_title(f"dBIC={float(diagnostics['delta_bic_3_vs_2']):.1f}")
            row_axes[1].imshow(result["continuous_maps_37"]["state"][0].numpy(), cmap="viridis", vmin=0, vmax=2)
            row_axes[1].set_title(f"k1={diagnostics['k1']}, k2={diagnostics['k2']}"); row_axes[1].axis("off")
            row_axes[2].imshow(result["mask_37"][0].numpy(), cmap="gray", vmin=0, vmax=1)
            row_axes[2].set_title(f"Core area={float(result['foreground_area']):.3f}"); row_axes[2].axis("off")
            row_axes[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndFw={item['delta_F_beta_w_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

    comparison_name = {
        "test20_core": "core_method_comparison_test20.png",
        "test20_hysteresis": "hysteresis_comparison_test20.png",
        "pilot200": "pilot200_method_comparison.png",
    }.get(stage, "method_comparison.png")
    run(comparison_name, bars)
    run("precision_recall_area_comparison.png", pra)
    run("threshold_distribution.png", thresholds)
    run("three_state_area_distribution.png", states)
    run("per_image_fbw_delta.png", deltas)
    run("dataset_wise_comparison.png", datasets)
    run("success_visualizations.png", lambda plt, path: visual(plt, path, True))
    run("failure_visualizations.png", lambda plt, path: visual(plt, path, False))
    run("otgc_component_examples.png", lambda plt, path: otgc_examples(plt, path, True))
    run("otgc_failure_examples.png", lambda plt, path: otgc_examples(plt, path, False))
    run("ut3cp_examples.png", lambda plt, path: ut3cp_examples(plt, path, True))
    run("ut3cp_failure_examples.png", lambda plt, path: ut3cp_examples(plt, path, False))
    return failures


def _report(output_dir: Path, stage: str, summary: list[dict], selection: dict | None, audit: dict) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    lines = [
        "# GBSP Adaptive Threshold V3 Report",
        "",
        f"- Stage: {stage}",
        "- Adaptive formulas use no GT, R1, fixed-0.58, target-area prior or fixed Top-K.",
        "- Fixed-0.58 is constructed only inside the evaluator as an external audit reference.",
        "- No DINO forward or PCA refit is performed.",
        "",
        "## Macro metrics",
        "",
        _table(macro, ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")),
        "",
        "## Audit",
        "",
        f"- Valid/failed images: {audit['num_valid']}/{audit['num_failed']}",
        f"- Baseline reproduction: {audit.get('baseline_reproduction_status', 'NOT_APPLICABLE')}",
    ]
    if selection:
        lines.extend(
            (
                f"- Selection status: **{selection['status']}**",
                f"- Selected methods: {selection.get('selected_methods', [])}",
                f"- Hysteresis candidates: {selection.get('hysteresis_candidates', [])}",
            )
        )
    (output_dir / "GBSP_THRESHOLD_V3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    if bool(args.gbsp_root) == bool(args.threshold_root):
        raise ValueError("provide exactly one of --gbsp_root or --threshold_root")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    adaptive = bool(args.threshold_root)
    root = _resolve(args.threshold_root or args.gbsp_root)
    manifest_path = _manifest_path(root)
    selected_rows = _select(_manifest(manifest_path), args.sample_list, args.max_samples)
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(args.methods or [])
    if not adaptive and methods and set(methods) != {"fixed_058"}:
        raise ValueError("baseline mode only accepts --methods fixed_058")
    if adaptive and not methods:
        first = torch_load(Path(selected_rows[0]["cache_path"]), map_location="cpu")
        methods = list(first.get("results", {}))
    tasks = [
        {
            "dataset": str(row["dataset"]),
            "stem": str(row["stem"]),
            "cache_path": row["cache_path"],
            "image_path": row.get("image_path", ""),
            "gt_path": row.get("gt_path", ""),
            "adaptive": adaptive,
            "methods": methods,
            "reference_threshold": float(cfg.GBSP_V3_REFERENCE_FIXED_THRESHOLD),
            "compare": list(args.compare),
        }
        for row in selected_rows
    ]
    started, results = time.perf_counter(), []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP threshold V3 eval {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("V3 evaluation produced no valid samples")
    per_image = [row for result in valid for row in result["rows"]]
    summary = _aggregate(per_image)
    _enrich_summary(summary, per_image)
    counts = dict(Counter(task["dataset"] for task in tasks))
    stage = _stage(adaptive, len(tasks), counts, methods)
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    target = cfg.GBSP_V3_BASELINE_TEST20_TARGET
    reproduction_error = {key: abs(macro["fixed_058"][key] - float(value)) for key, value in target.items()}
    reproduction_pass = bool(
        stage == "baseline_test20"
        and max(reproduction_error.values()) < float(cfg.GBSP_V3_BASELINE_TOLERANCE)
    )
    if stage == "test20_core":
        selection = _core_selection(summary, per_image, cfg)
    elif stage == "test20_hysteresis":
        selection = _hysteresis_selection(summary, per_image, cfg)
    elif stage == "pilot200":
        selection = _pilot_selection(summary, per_image, cfg)
    else:
        selection = None
    audit = {
        "schema": "gbsp_threshold_v3_eval",
        "stage": stage,
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "dataset_counts": counts,
        "gt_used_for_calibration": False,
        "r1_used_for_calibration": False,
        "fixed_058_used_in_adaptive_formula": False,
        "target_area_prior_used": False,
        "fixed_topk_used": False,
        "dino_forward_used": False,
        "pca_refit_used": False,
        "baseline_reproduction_abs_error": reproduction_error,
        "baseline_reproduction_status": "PASS" if reproduction_pass else ("NOT_APPLICABLE" if adaptive else "FAIL"),
        "wall_seconds": time.perf_counter() - started,
    }
    threshold_rows = _distribution_rows(
        per_image,
        ("threshold_low", "threshold_high", "foreground_area", "bc_selected_as_foreground_ratio", "empty_mask", "area_over_50", "numerical_failure"),
    )
    state_rows = _distribution_rows(per_image, ("class0_area", "class1_area", "class2_area", "foreground_area"))
    otgc_rows = [row for row in _distribution_rows(per_image, ("mu0", "mu1", "mu2", "sigma_shared", "pi0", "pi1", "pi2", "gap01_std", "gap12_std", "anchor_loss", "mixture_log_likelihood", "objective", "selected_initialization", "optimizer_converged", "bic1", "bic2", "bic3", "delta_bic_3_vs_2", "weak_high_component", "monotonicity_passed", "bc_selected_as_foreground_ratio")) if row["method"].removesuffix("_h") == "otgc"]
    ut_rows = [row for row in _distribution_rows(per_image, ("k1", "k2", "high_segment_area", "middle_segment_area", "background_segment_area", "threshold_high", "threshold_low", "slope_high", "slope_middle", "slope_background", "sse1", "sse2", "sse3", "bic1", "bic2", "bic3", "delta_bic_3_vs_2", "weak_three_segment_support")) if row["method"].removesuffix("_h") == "ut_3cp"]
    _write_csv(output_dir / "per_image_metrics.csv", per_image, _union_fields(per_image))
    _write_csv(output_dir / "per_dataset_metrics.csv", summary, _union_fields(summary))
    _write_csv(output_dir / "threshold_distribution.csv", threshold_rows, _union_fields(threshold_rows))
    _write_csv(output_dir / "three_state_distribution.csv", state_rows, _union_fields(state_rows))
    _write_csv(output_dir / "otgc_parameter_distribution.csv", otgc_rows, _union_fields(otgc_rows))
    _write_csv(output_dir / "ut3cp_change_point_distribution.csv", ut_rows, _union_fields(ut_rows))
    bootstrap_rows = _bootstrap_ci(per_image)
    _write_csv(output_dir / "bootstrap_ci95.csv", bootstrap_rows, _union_fields(bootstrap_rows))
    _write_json(output_dir / "numerical_failure_summary.json", audit)
    if stage == "baseline_test20":
        _write_csv(output_dir / "baseline_test20.csv", [macro["fixed_058"]])
    elif stage == "test20_core":
        _write_csv(output_dir / "test20_core_summary.csv", list(macro.values()))
    elif stage == "test20_hysteresis":
        _write_csv(output_dir / "test20_hysteresis_summary.csv", list(macro.values()))
    elif stage == "pilot200":
        _write_csv(output_dir / "pilot200_summary.csv", list(macro.values()))
    elif stage == "full6473":
        _write_csv(output_dir / "full6473_summary.csv", list(macro.values()))
    if selection:
        _write_json(output_dir / "method_selection.json", selection)
        frozen = _frozen_config(selection, stage)
        _write_json(output_dir / ("final_config.json" if stage == "pilot200" else "selected_config.json"), frozen)
    _write_csv(
        output_dir / "downstream_1x1_results.csv",
        [{"method": method, "status": "not_run_full_gate_required", "checkpoint": ""} for method in (selection or {}).get("selected_methods", [])],
    )
    plot_failures = _plots(output_dir, stage, summary, per_image)
    _write_json(output_dir / "visualization_failures.json", plot_failures)
    _report(output_dir, stage, summary, selection, audit)
    print(json.dumps({**audit, "selection": selection}, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"V3 evaluation recorded {len(failures)} failures")


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
