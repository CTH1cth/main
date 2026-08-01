#!/usr/bin/env python3
"""Formal original-GT evaluation for DABE-TF CRMC-v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_crossrank import DABE_CROSSRANK_VERSION  # noqa: E402
from common.eval_dabe_background_null import (  # noqa: E402
    MetricAggregate,
    NumericAggregate,
    _difference_panel,
    _git_metadata,
    _gray_panel,
    _load_gt,
    _manifest_map,
    _markdown_table,
    _raw_metrics,
    _sha256,
    _write_csv,
)
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
    _resize_current,
    _resize_native,
)
from common.utils import load_config, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
METHODS = (
    "X0-R1",
    "X0-CrossR1",
    "X1-CrossRank-R1Dist",
    "Current-DABE-v2",
)
DATASET_ORDER = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
PAIRWISE = (
    ("X1-CrossRank-R1Dist", "X0-R1"),
    ("X1-CrossRank-R1Dist", "X0-CrossR1"),
    ("X1-CrossRank-R1Dist", "Current-DABE-v2"),
    ("X0-CrossR1", "X0-R1"),
)
HARD_FIELDS = (
    "hard_S_m",
    "hard_F_beta_w",
    "hard_F_beta_mean",
    "hard_E_mean",
    "hard_MAE",
    "hard_IoU",
    "hard_Precision",
    "hard_Recall",
    "hard_Area",
    "hard_Components",
)
OFFICIAL_FIELDS = (
    "official_soft_S_m",
    "official_soft_F_beta_w",
    "official_soft_F_beta_mean",
    "official_soft_F_beta_max",
    "official_soft_E_mean",
    "official_soft_E_max",
    "official_soft_MAE",
)
RAW_FIELDS = (
    "raw_MAE",
    "raw_Brier",
    "raw_SoftPrecision",
    "raw_SoftRecall",
    "raw_SoftIoU",
    "raw_prob_mean",
    "raw_prob_std",
)
RANKING_FIELDS = (
    "pixel_AP",
    "best_IoU_256",
    "best_IoU_threshold_256",
    "ranking_F_beta_max",
    "ranking_E_max",
)
PER_SAMPLE_FIELDS = (
    "dataset",
    "stem",
    "method",
    *HARD_FIELDS,
    *OFFICIAL_FIELDS,
    *RAW_FIELDS,
    *RANKING_FIELDS,
    "source_dabe_cache_path",
    "source_bgnull_cache_path",
    "crossrank_cache_path",
)
PAIRWISE_FIELDS = (
    "delta_hard_S_m",
    "delta_hard_F_beta_w",
    "delta_hard_E_mean",
    "delta_hard_MAE",
    "delta_hard_IoU",
    "delta_hard_Precision",
    "delta_hard_Recall",
    "delta_hard_Area",
    "delta_official_soft_F_beta_w",
    "delta_raw_MAE",
    "delta_pixel_AP",
    "delta_best_IoU_256",
)
SWAP_FIELDS = (
    "added_area",
    "removed_area",
    "area_balance_error",
    "added_fg_precision",
    "added_bg_error_ratio",
    "removed_bg_precision",
    "removed_fg_error_ratio",
    "correct_swap_count",
    "incorrect_swap_count",
    "swap_accuracy",
    "hard_disagreement_ratio",
)
NATIVE37_FIELDS = (
    "native37_r1_area",
    "native37_x1_area",
    "native37_area_delta",
    "native37_added_count",
    "native37_removed_count",
    "native37_balance_error",
    "native37_added_fg_precision",
    "native37_removed_bg_precision",
    "native37_swap_accuracy",
)
GENERATION_FIELDS = (
    "r1_min", "r1_mean", "r1_max", "r1_std",
    "cross_min", "cross_mean", "cross_max", "cross_std",
    "crossrank_min", "crossrank_mean", "crossrank_max", "crossrank_std",
    "cross_unique_ratio", "cross_tie_ratio",
    "crossrank_unique_ratio", "crossrank_tie_ratio",
    "crossrank_sorted_l1_vs_r1", "crossrank_sorted_max_abs_vs_r1",
    "crossrank_sorted_invariant_violation",
    "crossrank_spearman_vs_cross", "crossrank_spearman_vs_r1",
    "crossrank_monotonic_violation_count",
    "crossrank_constant_source_fallback",
    "r1_hard_area_37", "cross_hard_area_37", "crossrank_hard_area_37",
    "cross_hard_area_delta_vs_r1_37",
    "crossrank_hard_area_delta_vs_r1_37",
    "r1_hard_area_68", "cross_hard_area_68", "crossrank_hard_area_68",
    "crossrank_hard_area_delta_vs_r1_68",
)
BASELINE_EXACT = {
    "X0-R1": {
        "hard_S_m": 0.7321127747158344,
        "hard_F_beta_w": 0.6242804301164221,
        "hard_E_mean": 0.8279479346485146,
        "hard_MAE": 0.08046898620831737,
        "hard_Precision": 0.7069664740758146,
        "hard_Recall": 0.7124157409732799,
        "hard_Area": 0.1260482386630983,
        "pixel_AP": 0.7673479264931065,
        "best_IoU_256": 0.6172645760728195,
    },
    "X0-CrossR1": {
        "hard_S_m": 0.7300490470801322,
        "hard_F_beta_w": 0.6225884413113011,
        "hard_E_mean": 0.8268769915470675,
        "hard_MAE": 0.0802121861926629,
        "hard_Precision": 0.7146538916738563,
        "hard_Recall": 0.6948888351781436,
        "hard_Area": 0.1210548485545675,
        "pixel_AP": 0.7707675872061015,
        "best_IoU_256": 0.6189410971107876,
    },
    "Current-DABE-v2": {
        "hard_S_m": 0.7031564688307831,
        "hard_F_beta_w": 0.5722451313959636,
        "hard_E_mean": 0.7743040501812801,
        "hard_MAE": 0.0939140360866542,
    },
}


def _swap_metrics(r1: torch.Tensor, x1: torch.Tensor, gt: torch.Tensor, threshold: float):
    r1_mask = r1 > threshold
    x1_mask = x1 > threshold
    gt_fg = gt > 0.5
    added = x1_mask & ~r1_mask
    removed = r1_mask & ~x1_mask
    added_count, removed_count = int(added.sum()), int(removed.sum())
    total = int(gt.numel())
    added_fg = int((added & gt_fg).sum())
    removed_bg = int((removed & ~gt_fg).sum())
    added_bg = added_count - added_fg
    removed_fg = removed_count - removed_bg
    changed = added_count + removed_count
    correct = added_fg + removed_bg
    incorrect = added_bg + removed_fg
    return {
        "added_area": added_count / total,
        "removed_area": removed_count / total,
        "area_balance_error": abs(added_count - removed_count) / total,
        "added_fg_precision": added_fg / added_count if added_count else float("nan"),
        "added_bg_error_ratio": added_bg / added_count if added_count else float("nan"),
        "removed_bg_precision": removed_bg / removed_count if removed_count else float("nan"),
        "removed_fg_error_ratio": removed_fg / removed_count if removed_count else float("nan"),
        "correct_swap_count": correct,
        "incorrect_swap_count": incorrect,
        "swap_accuracy": correct / changed if changed else float("nan"),
        "hard_disagreement_ratio": changed / total,
    }


def _native37_swap(r1: torch.Tensor, x1: torch.Tensor, gt: torch.Tensor, threshold: float):
    gt37 = torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > 0.5
    r1_mask, x1_mask = r1 > threshold, x1 > threshold
    added, removed = x1_mask & ~r1_mask, r1_mask & ~x1_mask
    added_count, removed_count = int(added.sum()), int(removed.sum())
    changed = added_count + removed_count
    added_fg = int((added & gt37).sum())
    removed_bg = int((removed & ~gt37).sum())
    return {
        "native37_r1_area": float(r1_mask.float().mean()),
        "native37_x1_area": float(x1_mask.float().mean()),
        "native37_area_delta": float(x1_mask.float().mean() - r1_mask.float().mean()),
        "native37_added_count": added_count,
        "native37_removed_count": removed_count,
        "native37_balance_error": abs(added_count - removed_count) / r1.numel(),
        "native37_added_fg_precision": added_fg / added_count if added_count else float("nan"),
        "native37_removed_bg_precision": removed_bg / removed_count if removed_count else float("nan"),
        "native37_swap_accuracy": (added_fg + removed_bg) / changed if changed else float("nan"),
    }


def _init_worker(torch_threads: int):
    torch.set_num_threads(torch_threads)


def _evaluate_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    dabe_path = Path(task["source_dabe_cache_path"])
    bgnull_path = Path(task["source_bgnull_cache_path"])
    crossrank_path = Path(task["crossrank_cache_path"])
    dabe = torch_load(dabe_path, map_location="cpu")
    bgnull = torch_load(bgnull_path, map_location="cpu")
    crossrank = torch_load(crossrank_path, map_location="cpu")
    for payload, path in ((dabe, dabe_path), (bgnull, bgnull_path), (crossrank, crossrank_path)):
        if payload.get("dataset") != dataset or payload.get("stem") != stem:
            raise RuntimeError(f"Payload key mismatch: {path}")
    if bgnull.get("bgnull_version") != "dabe_bgnull_v1":
        raise RuntimeError(f"Wrong BGNull version: {bgnull_path}")
    if crossrank.get("crossrank_version") != DABE_CROSSRANK_VERSION:
        raise RuntimeError(f"Wrong CrossRank version: {crossrank_path}")
    diagnostics = crossrank.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError(f"Missing CrossRank diagnostics: {crossrank_path}")
    for field in GENERATION_FIELDS:
        if field not in diagnostics or diagnostics[field] is None:
            raise KeyError(f"Diagnostic {field} missing: {crossrank_path}")

    r1_37 = _load_response(bgnull, "n0_r1_37", (37, 37), bgnull_path)
    cross_37 = _load_response(bgnull, "n1_cross_r1_37", (37, 37), bgnull_path)
    x1_37 = _load_response(
        crossrank, "x1_crossrank_r1dist_37", (37, 37), crossrank_path
    )
    gt = _load_gt(task["gt_path"])
    gt_shape = tuple(gt.shape[-2:])
    probabilities = {
        "X0-R1": _resize_native(r1_37, gt_shape),
        "X0-CrossR1": _resize_native(cross_37, gt_shape),
        "X1-CrossRank-R1Dist": _resize_native(x1_37, gt_shape),
        "Current-DABE-v2": _resize_current(
            _load_response(dabe, "p_dabe_68", (68, 68), dabe_path), gt_shape
        ),
    }
    context = FastCODContext(gt)
    cod_inputs = []
    for method in METHODS:
        cod_inputs.extend(
            ((method, "hard", probabilities[method]), (method, "soft", probabilities[method]))
        )
    cod = context.evaluate_many(cod_inputs, task["threshold"])

    rows = []
    for method in METHODS:
        hard, soft = cod[(method, "hard")], cod[(method, "soft")]
        row = {
            "dataset": dataset,
            "stem": stem,
            "method": method,
            "hard_S_m": hard["S_m"],
            "hard_F_beta_w": hard["F_beta_w"],
            "hard_F_beta_mean": hard["F_beta_mean"],
            "hard_E_mean": hard["E_mean"],
            "hard_MAE": hard["MAE"],
            "hard_IoU": hard["IoU"],
            "hard_Precision": hard["Precision"],
            "hard_Recall": hard["Recall"],
            "hard_Area": hard["Area"],
            "hard_Components": hard["Components"],
            "official_soft_S_m": soft["S_m"],
            "official_soft_F_beta_w": soft["F_beta_w"],
            "official_soft_F_beta_mean": soft["F_beta_mean"],
            "official_soft_F_beta_max": soft["F_beta_max"],
            "official_soft_E_mean": soft["E_mean"],
            "official_soft_E_max": soft["E_max"],
            "official_soft_MAE": soft["MAE"],
            **_raw_metrics(probabilities[method], gt),
            **_ranking_metrics(probabilities[method], gt),
            "ranking_F_beta_max": soft["F_beta_max"],
            "ranking_E_max": soft["E_max"],
            "source_dabe_cache_path": str(dabe_path),
            "source_bgnull_cache_path": "" if method == "Current-DABE-v2" else str(bgnull_path),
            "crossrank_cache_path": str(crossrank_path) if method == "X1-CrossRank-R1Dist" else "",
            "_official_f_curve": soft["f_curve"],
            "_official_e_curve": soft["e_curve"],
        }
        for field in (*HARD_FIELDS, *OFFICIAL_FIELDS, *RAW_FIELDS, *RANKING_FIELDS):
            value = float(row[field])
            if field == "pixel_AP" and math.isnan(value):
                continue
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite metric: {dataset}/{stem}/{method}/{field}")
        rows.append(row)
    return {
        "dataset": dataset,
        "stem": stem,
        "rows": rows,
        "diagnostics": {field: diagnostics[field] for field in GENERATION_FIELDS},
        "swap": _swap_metrics(
            probabilities["X0-R1"], probabilities["X1-CrossRank-R1Dist"], gt, task["threshold"]
        ),
        "native37": _native37_swap(r1_37, x1_37, gt, task["threshold"]),
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _section_rows(aggregates, datasets, section):
    fields = {
        "hard": HARD_FIELDS,
        "official_soft": OFFICIAL_FIELDS,
        "raw_continuous": RAW_FIELDS,
        "ranking": RANKING_FIELDS,
    }[section]
    by_dataset = []
    for dataset in datasets:
        for method in METHODS:
            values = aggregates[(dataset, method)].result()
            by_dataset.append({
                "scope": "by_dataset", "dataset": dataset, "method": method,
                **{field: values[field] for field in fields},
                "num_samples": values["num_samples"], "ap_valid_count": values["ap_valid_count"],
            })
    overall = []
    for method in METHODS:
        values = aggregates[("ALL", method)].result()
        overall.append({
            "scope": "sample_overall", "dataset": "ALL", "method": method,
            **{field: values[field] for field in fields},
            "num_samples": values["num_samples"], "ap_valid_count": values["ap_valid_count"],
        })
    macro = []
    for method in METHODS:
        method_rows = [row for row in by_dataset if row["method"] == method]
        result = {"scope": "dataset_macro", "dataset": "ALL", "method": method}
        for field in fields:
            values = [float(row[field]) for row in method_rows if math.isfinite(float(row[field]))]
            result[field] = float(np.mean(values)) if values else float("nan")
        result["num_samples"] = sum(int(row["num_samples"]) for row in method_rows)
        result["ap_valid_count"] = sum(int(row["ap_valid_count"]) for row in method_rows)
        macro.append(result)
    return by_dataset, overall, macro


def _aggregate_numeric(records, datasets, fields, sum_fields=()):
    aggregates = {scope: NumericAggregate(fields) for scope in [*datasets, "ALL"]}
    for row in records:
        aggregates[row["dataset"]].add(row)
        aggregates["ALL"].add(row)
    by = []
    for dataset in datasets:
        by.append({"scope": "by_dataset", "dataset": dataset, **aggregates[dataset].result(sum_fields=sum_fields)})
    overall = [{"scope": "sample_overall", "dataset": "ALL", **aggregates["ALL"].result(sum_fields=sum_fields)}]
    return by, overall


def _swap_overlay(r1: torch.Tensor, x1: torch.Tensor, size=224) -> Image.Image:
    r1 = torch_f.interpolate(r1.unsqueeze(0), size=(size, size), mode="nearest").squeeze() > 0.5
    x1 = torch_f.interpolate(x1.unsqueeze(0), size=(size, size), mode="nearest").squeeze() > 0.5
    common, added, removed = r1 & x1, x1 & ~r1, r1 & ~x1
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[common.numpy()] = (145, 145, 145)
    array[added.numpy()] = (0, 220, 0)
    array[removed.numpy()] = (230, 30, 30)
    return Image.fromarray(array, mode="RGB")


def _save_one_visual(task: dict, output_path: Path):
    dabe = torch_load(task["source_dabe_cache_path"], map_location="cpu")
    bgnull = torch_load(task["source_bgnull_cache_path"], map_location="cpu")
    crossrank = torch_load(task["crossrank_cache_path"], map_location="cpu")
    r1 = bgnull["n0_r1_37"].detach().cpu().float()
    cross = bgnull["n1_cross_r1_37"].detach().cpu().float()
    x1 = crossrank["x1_crossrank_r1dist_37"].detach().cpu().float()
    gt = _load_gt(task["gt_path"])
    size = 224
    with Image.open(task["image_path"]) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    panels = [
        ("RGB", rgb),
        ("GT", _gray_panel(gt, size, nearest=True)),
        ("R1", _gray_panel(r1, size)),
        ("CrossR1", _gray_panel(cross, size)),
        ("CrossRank-R1Dist", _gray_panel(x1, size)),
        ("Current DABE-v2", _gray_panel(dabe["p_dabe_68"], size)),
        ("CrossRank - R1", _difference_panel(x1 - r1, size)),
        ("Hard swap", _swap_overlay(r1, x1, size)),
    ]
    label_height = 24
    canvas = Image.new("RGB", (4 * size, 2 * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x, y = (index % 4) * size, (index // 4) * (size + label_height)
        canvas.paste(panel, (x, y + label_height))
        draw.text((x + 4, y + 5), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_visualizations(tasks, delta_map, out_dir: Path, datasets):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["dataset"]].append(task)
    for dataset in datasets:
        available = grouped[dataset]
        for task in available[:8]:
            _save_one_visual(task, out_dir / "vis" / "fixed_first8" / dataset / f"{task['stem']}.png")
        ranked = sorted(available, key=lambda task: delta_map[(task["dataset"], task["stem"])], reverse=True)
        for folder, selected in (("x1_vs_r1_top8", ranked[:8]), ("x1_vs_r1_bottom8", ranked[-8:])):
            for task in selected:
                _save_one_visual(task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png")


def _baseline_check(sections, full_run):
    baseline = {"applicable": full_run, "tolerance": 1e-5, "checks": {}, "passed": None}
    if not full_run:
        return baseline
    hard = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
    ranking = {row["method"]: row for row in sections["ranking"]["dataset_macro"]}
    passed = True
    for method, expected_fields in BASELINE_EXACT.items():
        baseline["checks"][method] = {}
        for field, expected in expected_fields.items():
            source = ranking[method] if field in {"pixel_AP", "best_IoU_256"} else hard[method]
            actual = float(source[field])
            error = abs(actual - expected)
            ok = error <= 1e-5
            passed &= ok
            baseline["checks"][method][field] = {
                "actual": actual, "expected": expected,
                "absolute_error": error, "passed": bool(ok),
            }
    baseline["passed"] = bool(passed)
    return baseline


def _success(sections, baseline):
    if not baseline.get("passed"):
        return {"not_evaluated": True}
    hard_by = {(row["dataset"], row["method"]): row for row in sections["hard"]["by_dataset"]}
    macro = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
    r1, cross, x1 = macro["X0-R1"], macro["X0-CrossR1"], macro["X1-CrossRank-R1Dist"]
    deltas_r1 = {
        dataset: float(hard_by[(dataset, "X1-CrossRank-R1Dist")]["hard_F_beta_w"])
        - float(hard_by[(dataset, "X0-R1")]["hard_F_beta_w"])
        for dataset in DATASET_ORDER
    }
    deltas_cross = {
        dataset: float(hard_by[(dataset, "X1-CrossRank-R1Dist")]["hard_F_beta_w"])
        - float(hard_by[(dataset, "X0-CrossR1")]["hard_F_beta_w"])
        for dataset in DATASET_ORDER
    }
    calibration = (
        x1["hard_F_beta_w"] > cross["hard_F_beta_w"]
        and x1["hard_Recall"] > cross["hard_Recall"]
        and abs(x1["hard_Area"] - r1["hard_Area"]) < abs(cross["hard_Area"] - r1["hard_Area"])
    )
    basic = (
        x1["hard_F_beta_w"] > 0.624280
        and x1["hard_MAE"] <= 0.080969
        and min(deltas_r1.values()) >= -0.005
    )
    paper = (
        x1["hard_F_beta_w"] >= 0.625780
        and sum(value >= 0.0 for value in deltas_r1.values()) >= 3
        and min(deltas_r1.values()) >= -0.003
        and x1["hard_MAE"] <= 0.080469
    )
    clear = (
        x1["hard_F_beta_w"] >= 0.6270
        and sum(value > 0.0 for value in deltas_r1.values()) >= 3
        and (x1["hard_S_m"] >= r1["hard_S_m"] or x1["hard_E_mean"] >= r1["hard_E_mean"])
        and x1["hard_MAE"] <= r1["hard_MAE"]
    )
    strong = (
        x1["hard_F_beta_w"] >= 0.6290
        and min(deltas_r1.values()) >= -0.003
        and sum(
            (x1[field] >= r1[field] if field != "hard_MAE" else x1[field] <= r1[field])
            for field in ("hard_S_m", "hard_E_mean", "hard_MAE")
        ) >= 2
    )
    delta_macro = x1["hard_F_beta_w"] - r1["hard_F_beta_w"]
    if paper:
        route = "A"
    elif x1["hard_F_beta_w"] > cross["hard_F_beta_w"] and x1["hard_F_beta_w"] <= r1["hard_F_beta_w"]:
        route = "B"
    elif 0.0 < delta_macro < 0.0015:
        route = "C"
    else:
        route = "D"
    return {
        "calibration_hypothesis_verified": bool(calibration),
        "engineering_basic_success": bool(basic),
        "paper_retention_success": bool(paper),
        "clear_success": bool(clear),
        "strong_success": bool(strong),
        "x1_dataset_f_beta_w_delta_vs_r1": deltas_r1,
        "x1_dataset_f_beta_w_delta_vs_crossr1": deltas_cross,
        "x1_macro_f_beta_w_delta_vs_r1": float(delta_macro),
        "route_case": route,
        "frozen_method": "X1-CrossRank-R1Dist" if paper else "R1 Direct",
        "move_to_high_resolution_boundary_projection": not paper,
    }


def _write_readme(out_dir: Path, num_samples: int, full_run: bool):
    text = f"""# CRMC-v1 formal evaluation

- Scope: {'full four-dataset evaluation' if full_run else '5-sample sanity only'}.
- Number of samples: {num_samples}.
- Candidate generation reads only frozen DABE-v2 and BGNull-v1 caches; no GT or DINO forward.
- Native candidates use 37→68→original-GT bilinear resize with `align_corners=False`.
- Current-DABE-v2 uses cached `p_dabe_68`→original-GT bilinear resize.
- GT and Hard prediction use strict `>0.5`.
- Official Soft CODMetrics performs per-image Min-Max for non-constant predictions.
- Raw Continuous performs no additional normalization.
- Best-IoU uses 256 fixed thresholds for diagnosis only and never changes Hard results.
- Dataset Macro is the unweighted mean of the four datasets.
- Pairwise deltas are method A minus method B; positive MAE delta means larger error.
- Native-37 GT is nearest-resized and is diagnostic only.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _write_results(out_dir, sections, generation_by, swap_by, native_by, baseline, success, full_run):
    hard_by = sections["hard"]["by_dataset"]
    hard_macro = sections["hard"]["dataset_macro"]
    ranking_macro = sections["ranking"]["dataset_macro"]
    official_macro = sections["official_soft"]["dataset_macro"]
    hard_map = {row["method"]: row for row in hard_macro}
    rank_map = {row["method"]: row for row in ranking_macro}
    gen_map = {row["dataset"]: row for row in generation_by}
    swap_map = {row["dataset"]: row for row in swap_by}
    native_map = {row["dataset"]: row for row in native_by}
    hard_table = hard_by + [{**row, "dataset": "Dataset-Macro"} for row in hard_macro]
    lines = [
        "# DABE-TF CRMC-v1 结果", "", "## 评测范围", "",
        f"- {'完整四数据集 6473 样本正式评测。' if full_run else '仅 5 样本 sanity；不得作为正式性能结论。'}",
        "- X1 固定公式：`rank_transport(source=X0-CrossR1, reference=X0-R1)`。",
        "- Hard 使用 strict `>0.5`；无阈值扫描、融合、路由或数据集特定规则。", "",
        "## 基线复现", "",
        ("X0-R1、X0-CrossR1、Current-DABE-v2 均在 1e-5 容差内精确复现。" if baseline.get("passed") else ("协议失败：停止给出 X1 性能结论。" if full_run else "Sanity run 不适用完整基线复现。")),
        "", "## Hard 主表", "",
        _markdown_table(hard_table, ("hard_S_m", "hard_F_beta_w", "hard_E_mean", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area")),
        "", "## Ranking Dataset-Macro", "",
        _markdown_table([{**row, "dataset": "Dataset-Macro"} for row in ranking_macro], ("pixel_AP", "best_IoU_256", "ranking_F_beta_max", "ranking_E_max")),
        "", "## Official Soft Dataset-Macro", "",
        _markdown_table([{**row, "dataset": "Dataset-Macro"} for row in official_macro], ("official_soft_S_m", "official_soft_F_beta_w", "official_soft_E_mean", "official_soft_MAE")),
        "",
    ]
    if not full_run or not baseline.get("passed"):
        lines += ["## 判定", "", "非完整评测或基线未通过，不给出 CRMC 性能结论。", ""]
    else:
        r1, cross, x1 = hard_map["X0-R1"], hard_map["X0-CrossR1"], hard_map["X1-CrossRank-R1Dist"]
        gen_macro = {field: float(np.mean([float(gen_map[d][field]) for d in DATASET_ORDER])) for field in GENERATION_FIELDS if field not in {"crossrank_monotonic_violation_count", "crossrank_constant_source_fallback", "crossrank_sorted_invariant_violation"}}
        gen_violation = sum(float(gen_map[d]["crossrank_monotonic_violation_count"]) for d in DATASET_ORDER)
        gen_fallback = sum(float(gen_map[d]["crossrank_constant_source_fallback"]) for d in DATASET_ORDER)
        gen_sorted_violation = sum(float(gen_map[d]["crossrank_sorted_invariant_violation"]) for d in DATASET_ORDER)
        swap_macro = {field: float(np.nanmean([float(swap_map[d][field]) for d in DATASET_ORDER])) for field in SWAP_FIELDS}
        native_macro = {field: float(np.nanmean([float(native_map[d][field]) for d in DATASET_ORDER])) for field in NATIVE37_FIELDS}
        per_r1 = success["x1_dataset_f_beta_w_delta_vs_r1"]
        per_cross = success["x1_dataset_f_beta_w_delta_vs_crossr1"]
        lines += [
            "## 19 项固定问题回答", "",
            "1. X0-R1 是否精确复现：是。",
            "2. X0-CrossR1 是否精确复现：是。",
            "3. Current-DABE-v2 是否精确复现：是。",
            f"4. X1 是否保持 R1 native-37 分布：sorted mean L1={gen_macro['crossrank_sorted_l1_vs_r1']:.8g}，最大绝对误差宏平均={gen_macro['crossrank_sorted_max_abs_vs_r1']:.8g}，超过 1e-6 的无-tie样本={gen_sorted_violation:.0f}，native area delta={native_macro['native37_area_delta']:+.8f}。",
            f"5. CrossR1 ties：unique ratio={gen_macro['cross_unique_ratio']:.6f}，tie ratio={gen_macro['cross_tie_ratio']:.6f}。",
            f"6. X1 是否保持 CrossR1 排序：Spearman={gen_macro['crossrank_spearman_vs_cross']:.8f}，monotonic violations={gen_violation:.0f}。",
            f"7. X1 是否恢复前景面积：CrossR1→R1 area gap={cross['hard_Area']-r1['hard_Area']:+.6f}；X1→R1 gap={x1['hard_Area']-r1['hard_Area']:+.6f}。",
            f"8. X1 是否恢复 Recall：CrossR1={cross['hard_Recall']:.6f}，X1={x1['hard_Recall']:.6f}，R1={r1['hard_Recall']:.6f}。",
            f"9. X1 Precision 是否高于 R1：X1={x1['hard_Precision']:.6f}，R1={r1['hard_Precision']:.6f}，delta={x1['hard_Precision']-r1['hard_Precision']:+.6f}。",
            f"10. X1 Pixel AP 是否接近 CrossR1：X1={rank_map['X1-CrossRank-R1Dist']['pixel_AP']:.6f}，CrossR1={rank_map['X0-CrossR1']['pixel_AP']:.6f}。",
            f"11. X1 Best IoU 是否接近 CrossR1：X1={rank_map['X1-CrossRank-R1Dist']['best_IoU_256']:.6f}，CrossR1={rank_map['X0-CrossR1']['best_IoU_256']:.6f}。",
            f"12. X1 新增像素真前景比例：{swap_macro['added_fg_precision']:.6f}。",
            f"13. X1 删除像素真背景比例：{swap_macro['removed_bg_precision']:.6f}。",
            f"14. swap accuracy 是否高于 0.5：{swap_macro['swap_accuracy']:.6f}，{'是' if swap_macro['swap_accuracy'] > 0.5 else '否'}。",
            f"15. 四数据集统一有效：逐数据集 F_beta^w delta vs R1 = {json.dumps(per_r1, ensure_ascii=False)}。",
            f"16. 工程基本成功：{str(success['engineering_basic_success']).lower()}。",
            f"17. 论文保留标准：{str(success['paper_retention_success']).lower()}。",
            f"18. 最终冻结：{success['frozen_method']}。",
            f"19. 是否转向高分辨率边界投影：{str(success['move_to_high_resolution_boundary_projection']).lower()}。",
            "", "## 关键变化", "",
            f"- X1 vs R1 Dataset-Macro：F_beta^w {x1['hard_F_beta_w']-r1['hard_F_beta_w']:+.6f}，Precision {x1['hard_Precision']-r1['hard_Precision']:+.6f}，Recall {x1['hard_Recall']-r1['hard_Recall']:+.6f}，Area {x1['hard_Area']-r1['hard_Area']:+.6f}，MAE {x1['hard_MAE']-r1['hard_MAE']:+.6f}。",
            f"- X1 vs CrossR1 Dataset-Macro：F_beta^w {x1['hard_F_beta_w']-cross['hard_F_beta_w']:+.6f}，Recall {x1['hard_Recall']-cross['hard_Recall']:+.6f}，Area {x1['hard_Area']-cross['hard_Area']:+.6f}。",
            f"- X1 vs CrossR1 逐数据集 F_beta^w：{json.dumps(per_cross, ensure_ascii=False)}。",
            f"- constant fallback 宏汇总={gen_fallback:.0f}；native37 swap accuracy={native_macro['native37_swap_accuracy']:.6f}。",
            "", "## 预定义路线", "",
            f"情况 {success['route_case']}。最终冻结 `{success['frozen_method']}`。",
        ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_crossrank(
    config_path,
    dabe_root,
    bgnull_root,
    crossrank_root,
    out_dir,
    split="test",
    max_samples=-1,
    threshold=0.5,
    save_vis=False,
    workers=None,
    torch_threads=1,
    overwrite=False,
    overwrite_reason="",
):
    started = time.time()
    config_path = Path(config_path).resolve()
    dabe_root = Path(dabe_root).resolve()
    bgnull_root = Path(bgnull_root).resolve()
    crossrank_root = Path(crossrank_root).resolve()
    out_dir = Path(out_dir).resolve()
    if split != "test" or threshold != 0.5:
        raise ValueError("Frozen protocol requires split=test and threshold=0.5")
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    full_run = max_samples == -1

    dabe_manifest = dabe_root / f"manifest_{split}.jsonl"
    bgnull_manifest = bgnull_root / f"manifest_{split}.jsonl"
    crossrank_manifest = crossrank_root / f"manifest_{split}.jsonl"
    dabe_rows, dabe_map = _manifest_map(dabe_manifest, 6473 if full_run else None)
    _, bgnull_map = _manifest_map(bgnull_manifest, 6473 if full_run else None)
    crossrank_rows, crossrank_map = _manifest_map(crossrank_manifest, 6473 if full_run else max_samples)
    selected = dabe_rows if full_run else dabe_rows[:max_samples]
    selected_keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
    if set(dabe_map) != set(bgnull_map):
        raise RuntimeError("DABE/BGNull manifest keys differ")
    if set(selected_keys) != set(crossrank_map):
        raise RuntimeError("CrossRank manifest keys do not exactly match selected DABE items")
    protocol_path = crossrank_root / "protocol.json"
    crossrank_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if crossrank_protocol.get("crossrank_version") != DABE_CROSSRANK_VERSION:
        raise ValueError("Wrong CrossRank protocol version")
    if crossrank_protocol.get("source_dabe_manifest_sha256") != _sha256(dabe_manifest):
        raise RuntimeError("CrossRank protocol DABE manifest hash mismatch")
    if crossrank_protocol.get("source_bgnull_manifest_sha256") != _sha256(bgnull_manifest):
        raise RuntimeError("CrossRank protocol BGNull manifest hash mismatch")
    datasets = [d for d in DATASET_ORDER if any(key[0] == d for key in selected_keys)]
    for row in selected:
        for field in ("gt_path", "image_path"):
            if field not in row or not Path(row[field]).is_file():
                raise FileNotFoundError(f"{field} missing for {row['dataset']}/{row['stem']}")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_dir: {out_dir}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (out_dir / "eval.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    workers = workers or min(12, os.cpu_count() or 1)
    tasks = []
    for row in selected:
        key = (str(row["dataset"]), str(row["stem"]))
        tasks.append({
            "dataset": key[0], "stem": key[1],
            "source_dabe_cache_path": dabe_map[key]["cache_path"],
            "source_bgnull_cache_path": bgnull_map[key]["cache_path"],
            "crossrank_cache_path": crossrank_map[key]["cache_path"],
            "gt_path": row["gt_path"], "image_path": row["image_path"],
            "threshold": threshold,
        })
    metrics = {(scope, method): MetricAggregate() for scope in [*datasets, "ALL"] for method in METHODS}
    generation_records, swap_records, native_records = [], [], []
    pair_values = defaultdict(list)
    vis_delta_map = {}
    peak_worker_rss = 0.0
    pair_sample_fields = ("dataset", "stem", "method_a", "method_b", *PAIRWISE_FIELDS)
    try:
        log(f"num_samples = {len(tasks)}")
        log(f"workers = {workers}")
        log("hard_threshold = strict > 0.5")
        log("candidate_generation_in_eval = false")
        with (
            (out_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as sample_handle,
            (out_dir / "pairwise_per_sample.csv").open("w", encoding="utf-8", newline="") as pair_handle,
            (out_dir / "swap_analysis_per_sample.csv").open("w", encoding="utf-8", newline="") as swap_handle,
            (out_dir / "native37_swap_per_sample.csv").open("w", encoding="utf-8", newline="") as native_handle,
        ):
            sample_writer = csv.DictWriter(sample_handle, fieldnames=PER_SAMPLE_FIELDS, extrasaction="ignore")
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_sample_fields, extrasaction="ignore")
            swap_writer = csv.DictWriter(swap_handle, fieldnames=("dataset", "stem", *SWAP_FIELDS), extrasaction="ignore")
            native_writer = csv.DictWriter(native_handle, fieldnames=("dataset", "stem", *NATIVE37_FIELDS), extrasaction="ignore")
            for writer in (sample_writer, pair_writer, swap_writer, native_writer):
                writer.writeheader()
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(torch_threads,)) as executor:
                for index, result in enumerate(executor.map(_evaluate_one, tasks, chunksize=1), 1):
                    dataset, stem = result["dataset"], result["stem"]
                    peak_worker_rss = max(peak_worker_rss, float(result["worker_peak_rss_mb"]))
                    rows_by_method = {row["method"]: row for row in result["rows"]}
                    for row in result["rows"]:
                        metrics[(dataset, row["method"])].add(row)
                        metrics[("ALL", row["method"])].add(row)
                        sample_writer.writerow(row)
                    generation_records.append({"dataset": dataset, "stem": stem, **result["diagnostics"]})
                    swap_row = {"dataset": dataset, "stem": stem, **result["swap"]}
                    native_row = {"dataset": dataset, "stem": stem, **result["native37"]}
                    swap_records.append(swap_row)
                    native_records.append(native_row)
                    swap_writer.writerow(swap_row)
                    native_writer.writerow(native_row)
                    vis_delta_map[(dataset, stem)] = (
                        rows_by_method["X1-CrossRank-R1Dist"]["hard_IoU"]
                        - rows_by_method["X0-R1"]["hard_IoU"]
                    )
                    mapping = {
                        "delta_hard_S_m": "hard_S_m", "delta_hard_F_beta_w": "hard_F_beta_w",
                        "delta_hard_E_mean": "hard_E_mean", "delta_hard_MAE": "hard_MAE",
                        "delta_hard_IoU": "hard_IoU", "delta_hard_Precision": "hard_Precision",
                        "delta_hard_Recall": "hard_Recall", "delta_hard_Area": "hard_Area",
                        "delta_official_soft_F_beta_w": "official_soft_F_beta_w",
                        "delta_raw_MAE": "raw_MAE", "delta_pixel_AP": "pixel_AP",
                        "delta_best_IoU_256": "best_IoU_256",
                    }
                    for method_a, method_b in PAIRWISE:
                        pair_row = {"dataset": dataset, "stem": stem, "method_a": method_a, "method_b": method_b}
                        for delta_field, source_field in mapping.items():
                            left, right = float(rows_by_method[method_a][source_field]), float(rows_by_method[method_b][source_field])
                            delta = left - right if math.isfinite(left) and math.isfinite(right) else float("nan")
                            pair_row[delta_field] = delta
                            if math.isfinite(delta):
                                pair_values[(dataset, method_a, method_b, delta_field)].append(delta)
                        pair_writer.writerow(pair_row)
                    if index == len(tasks) or index % 100 == 0:
                        log(f"processed = {index}/{len(tasks)}")

        sections = {}
        field_map = {"hard": HARD_FIELDS, "official_soft": OFFICIAL_FIELDS, "raw_continuous": RAW_FIELDS, "ranking": RANKING_FIELDS}
        for section, fields in field_map.items():
            by, overall, macro = _section_rows(metrics, datasets, section)
            sections[section] = {"by_dataset": by, "sample_overall": overall, "dataset_macro": macro}
            csv_fields = ("scope", "dataset", "method", *fields, "num_samples", "ap_valid_count")
            _write_csv(out_dir / f"{section}_by_dataset.csv", by, csv_fields)
            _write_csv(out_dir / f"{section}_sample_overall.csv", overall, csv_fields)
            _write_csv(out_dir / f"{section}_dataset_macro.csv", macro, csv_fields)

        generation_by, generation_overall = _aggregate_numeric(
            generation_records, datasets, GENERATION_FIELDS,
            sum_fields=("crossrank_monotonic_violation_count", "crossrank_constant_source_fallback", "crossrank_sorted_invariant_violation"),
        )
        swap_by, _ = _aggregate_numeric(
            swap_records, datasets, SWAP_FIELDS,
            sum_fields=("correct_swap_count", "incorrect_swap_count"),
        )
        native_by, _ = _aggregate_numeric(
            native_records, datasets, NATIVE37_FIELDS,
            sum_fields=("native37_added_count", "native37_removed_count"),
        )
        generation_out_fields = ("scope", "dataset", *GENERATION_FIELDS, *[f"{field}_valid_count" for field in GENERATION_FIELDS], "num_samples")
        swap_out_fields = ("scope", "dataset", *SWAP_FIELDS, *[f"{field}_valid_count" for field in SWAP_FIELDS], "num_samples")
        native_out_fields = ("scope", "dataset", *NATIVE37_FIELDS, *[f"{field}_valid_count" for field in NATIVE37_FIELDS], "num_samples")
        _write_csv(out_dir / "generation_diagnostics_by_dataset.csv", generation_by, generation_out_fields)
        _write_csv(out_dir / "generation_diagnostics_sample_overall.csv", generation_overall, generation_out_fields)
        _write_csv(out_dir / "swap_analysis_by_dataset.csv", swap_by, swap_out_fields)
        _write_csv(out_dir / "native37_swap_by_dataset.csv", native_by, native_out_fields)

        pair_summary = []
        for dataset in datasets:
            for method_a, method_b in PAIRWISE:
                for metric in PAIRWISE_FIELDS:
                    values = np.asarray(pair_values[(dataset, method_a, method_b, metric)], dtype=np.float64)
                    if not values.size:
                        continue
                    pair_summary.append({
                        "dataset": dataset, "method_a": method_a, "method_b": method_b, "metric": metric,
                        "mean_delta": float(np.mean(values)), "median_delta": float(np.median(values)),
                        "p25_delta": float(np.percentile(values, 25)), "p75_delta": float(np.percentile(values, 75)),
                        "win_ratio": float(np.mean(values > 1e-6)), "tie_ratio": float(np.mean(np.abs(values) <= 1e-6)),
                        "loss_ratio": float(np.mean(values < -1e-6)), "valid_count": int(values.size),
                    })
        pair_fields = ("dataset", "method_a", "method_b", "metric", "mean_delta", "median_delta", "p25_delta", "p75_delta", "win_ratio", "tie_ratio", "loss_ratio", "valid_count")
        _write_csv(out_dir / "pairwise_by_dataset.csv", pair_summary, pair_fields)

        if save_vis:
            log("saving_visualizations = true")
            _save_visualizations(tasks, vis_delta_map, out_dir, datasets)
            log("visualizations_complete = true")

        baseline = _baseline_check(sections, full_run)
        success = _success(sections, baseline)
        elapsed = time.time() - started
        main_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        git_commit, git_status = _git_metadata()
        protocol = {
            "crossrank_version": DABE_CROSSRANK_VERSION,
            "split": split, "num_samples": len(tasks), "full_run": full_run,
            "methods": list(METHODS), "pairwise_comparisons": [list(value) for value in PAIRWISE],
            "dino_rerun": False, "gt_used_for_generation": False,
            "background_reconstruction_rerun": False, "bgnull_reconstruction_rerun": False,
            "candidate_generation_in_eval": False, "dataset_specific_parameters": False,
            "threshold_search_used": False, "trainable_parameter_count": 0,
            "hard_threshold": threshold, "hard_operator": ">",
            "native_resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
            "current_resize": "p_dabe_68 -> original GT bilinear align_corners=False",
            "official_soft_per_image_minmax": True, "raw_continuous_extra_minmax": False,
            "best_iou_thresholds": "torch.linspace(0,1,256), diagnostic only",
            "native37_gt_resize": "nearest, diagnostic only", "save_vis": bool(save_vis),
            "config_path": str(config_path), "config_sha256": _sha256(config_path),
            "source_dabe_manifest": str(dabe_manifest), "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_bgnull_manifest": str(bgnull_manifest), "source_bgnull_manifest_sha256": _sha256(bgnull_manifest),
            "crossrank_manifest": str(crossrank_manifest), "crossrank_manifest_sha256": _sha256(crossrank_manifest),
            "crossrank_protocol": str(protocol_path), "crossrank_protocol_sha256": _sha256(protocol_path),
            "evaluator_sha256": _sha256(SCRIPT_PATH), "git_commit": git_commit, "git_status_short": git_status,
            "workers": workers, "torch_threads_per_worker": torch_threads,
            "elapsed_seconds": elapsed, "average_seconds_per_image": elapsed / len(tasks),
            "main_peak_rss_mb": main_peak_rss, "max_worker_peak_rss_mb": peak_worker_rss,
            "overwrite": bool(overwrite), "overwrite_reason": overwrite_reason.strip(),
        }
        summary = {
            "protocol": protocol, "baseline_reproduction": baseline,
            "success_assessment": success, **sections,
            "generation_diagnostics_by_dataset": generation_by,
            "generation_diagnostics_sample_overall": generation_overall,
            "swap_analysis_by_dataset": swap_by,
            "native37_swap_by_dataset": native_by,
            "pairwise_by_dataset": pair_summary,
        }
        write_json(out_dir / "protocol.json", protocol)
        write_json(out_dir / "summary.json", summary)
        _write_readme(out_dir, len(tasks), full_run)
        _write_results(out_dir, sections, generation_by, swap_by, native_by, baseline, success, full_run)
        log(f"baseline_reproduction = {baseline.get('passed')}")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(tasks):.6f}")
        log(f"main_peak_rss_mb = {main_peak_rss:.3f}")
        log(f"max_worker_peak_rss_mb = {peak_worker_rss:.3f}")
        return summary
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--bgnull_root", required=True)
    parser.add_argument("--crossrank_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_crossrank(
        config_path=args.config, dabe_root=args.dabe_root,
        bgnull_root=args.bgnull_root, crossrank_root=args.crossrank_root,
        out_dir=args.out_dir, split=args.split, max_samples=args.max_samples,
        threshold=args.threshold, save_vis=args.save_vis, workers=args.workers,
        torch_threads=args.torch_threads, overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
    )
