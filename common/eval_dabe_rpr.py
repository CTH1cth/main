#!/usr/bin/env python3
"""Formal original-GT evaluation and mechanism audit for DABE RPR-v1."""

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

from common.dabe_rank_calibration import average_percentile_rank  # noqa: E402
from common.eval_dabe_background_null import (  # noqa: E402
    MetricAggregate,
    _difference_panel,
    _git_metadata,
    _gray_panel,
    _load_gt,
    _manifest_map,
    _raw_metrics,
    _sha256,
    _write_csv,
)
from common.eval_dabe_cvbr import _touch  # noqa: E402
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
    _resize_current,
    _resize_native,
)
from common.utils import load_config, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
VERSION = "dabe_rpr_v1"
CVBR_VERSION = "dabe_cvbr_v1"
METHODS = (
    "B0-R1-BW2",
    "B1-R1-BW1",
    "V1-CVBR-SecondRing",
    "V2-CVBR-AllBorder",
    "P1-RPR-SecondRing",
    "P2-RPR-AllBorder",
    "Current-DABE-v2",
)
CVBR_FIELDS = {
    METHODS[0]: "b0_r1_bw2_37",
    METHODS[1]: "b1_r1_bw1_37",
    METHODS[2]: "v1_cvbr_second_ring_37",
    METHODS[3]: "v2_cvbr_all_border_37",
}
RPR_FIELDS = {
    METHODS[4]: "p1_rpr_secondring_37",
    METHODS[5]: "p2_rpr_allborder_37",
}
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
SUBSETS = ("Touch-1", "Touch-2-only", "Non-touch")
ERROR_TYPES = ("FN", "TN", "TP", "FP")
PAIRWISE = (
    (METHODS[4], METHODS[0]),
    (METHODS[5], METHODS[0]),
    (METHODS[4], METHODS[2]),
    (METHODS[5], METHODS[3]),
    (METHODS[4], METHODS[1]),
    (METHODS[5], METHODS[1]),
    (METHODS[5], METHODS[4]),
    (METHODS[0], METHODS[6]),
)
HARD = (
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
OFFICIAL = (
    "official_soft_S_m",
    "official_soft_F_beta_w",
    "official_soft_F_beta_mean",
    "official_soft_F_beta_max",
    "official_soft_E_mean",
    "official_soft_E_max",
    "official_soft_MAE",
)
RAW = (
    "raw_MAE",
    "raw_Brier",
    "raw_SoftPrecision",
    "raw_SoftRecall",
    "raw_SoftIoU",
    "raw_prob_mean",
    "raw_prob_std",
)
RANK = (
    "pixel_AP",
    "best_IoU_256",
    "best_IoU_threshold_256",
    "ranking_F_beta_max",
    "ranking_E_max",
)
PAIR_FIELDS = (
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
PAIR_SOURCE = {
    "delta_hard_S_m": "hard_S_m",
    "delta_hard_F_beta_w": "hard_F_beta_w",
    "delta_hard_E_mean": "hard_E_mean",
    "delta_hard_MAE": "hard_MAE",
    "delta_hard_IoU": "hard_IoU",
    "delta_hard_Precision": "hard_Precision",
    "delta_hard_Recall": "hard_Recall",
    "delta_hard_Area": "hard_Area",
    "delta_official_soft_F_beta_w": "official_soft_F_beta_w",
    "delta_raw_MAE": "raw_MAE",
    "delta_pixel_AP": "pixel_AP",
    "delta_best_IoU_256": "best_IoU_256",
}
ERROR_DIAGNOSTIC_FIELDS = (
    "u_base",
    "topk_prior_min",
    "topk_prior_mean",
    "topk_low_q_ratio",
    "weight_l1_shift",
    "u_reduction",
    "delta_raw",
)
QUERY_FIELDS = (
    "mean_U_base",
    "ratio_U_gt_0",
    "ratio_U_gt_001",
    "ratio_U_gt_005",
    "ratio_U_gt_010",
    "mean_topk_low_q_ratio",
    "mean_topk_prior_min",
    "mean_topk_prior_mean",
)
NATIVE_SWAP_FIELDS = (
    "recovered_fg_count",
    "added_bg_error_count",
    "added_patch_precision",
    "lost_fg_count",
    "removed_bg_correct_count",
    "removed_patch_background_precision",
    "correct_swap_count",
    "incorrect_swap_count",
    "native37_swap_accuracy",
    "native37_area_delta",
)
ORIGINAL_SWAP_FIELDS = (
    "added_area",
    "removed_area",
    "hard_disagreement_ratio",
    "area_balance_error",
    "added_fg_precision",
    "added_bg_error_ratio",
    "removed_bg_precision",
    "removed_fg_error_ratio",
    "correct_swap_count",
    "incorrect_swap_count",
    "swap_accuracy",
)
BASELINE = {
    METHODS[0]: {
        "hard_S_m": 0.7321127747158344,
        "hard_F_beta_w": 0.6242804301164221,
        "hard_E_mean": 0.8279479346485145,
        "hard_MAE": 0.08046898620831737,
        "hard_Precision": 0.7069664740758147,
        "hard_Recall": 0.7124157409732799,
        "hard_Area": 0.1260482386630983,
    },
    METHODS[2]: {
        "hard_S_m": 0.7322427176486375,
        "hard_F_beta_w": 0.6237979809619487,
        "hard_E_mean": 0.8267134010106899,
        "hard_MAE": 0.08086713215389411,
    },
    METHODS[3]: {
        "hard_S_m": 0.7282715583581764,
        "hard_F_beta_w": 0.6164219592032302,
        "hard_E_mean": 0.8210760059901125,
        "hard_MAE": 0.08381993663411161,
    },
    METHODS[6]: {
        "hard_S_m": 0.7031564688307831,
        "hard_F_beta_w": 0.5722451313959636,
        "hard_E_mean": 0.7743040501812801,
        "hard_MAE": 0.0939140360866542,
    },
}


def _finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return (float(np.mean(values)) if values else float("nan")), len(values)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = average_percentile_rank(torch.from_numpy(scores)).double().numpy()
    ranks = ranks * (scores.size - 1) + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _mean_on(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value.reshape(-1).float()[mask.reshape(-1).bool()]
    return float(selected.mean()) if selected.numel() else float("nan")


def _query_exposure(payload: dict, method: str) -> dict:
    tag = "p1" if method == METHODS[4] else "p2"
    u = payload[f"u_base_{tag}_37"].reshape(-1).float()
    return {
        "method": method,
        "mean_U_base": float(u.mean()),
        "ratio_U_gt_0": float((u > 0).float().mean()),
        "ratio_U_gt_001": float((u > 0.01).float().mean()),
        "ratio_U_gt_005": float((u > 0.05).float().mean()),
        "ratio_U_gt_010": float((u > 0.10).float().mean()),
        "mean_topk_low_q_ratio": float(
            payload[f"topk_low_q_ratio_{tag}_37"].float().mean()
        ),
        "mean_topk_prior_min": float(payload[f"topk_prior_min_{tag}_37"].float().mean()),
        "mean_topk_prior_mean": float(payload[f"topk_prior_mean_{tag}_37"].float().mean()),
    }


def _native_swap(b0: torch.Tensor, candidate: torch.Tensor, gt37: torch.Tensor) -> dict:
    b0_hard = b0 > 0.5
    candidate_hard = candidate > 0.5
    gt = gt37.bool()
    added = ~b0_hard & candidate_hard
    removed = b0_hard & ~candidate_hard
    recovered = int((added & gt).sum())
    added_bg = int((added & ~gt).sum())
    lost_fg = int((removed & gt).sum())
    removed_bg = int((removed & ~gt).sum())
    correct, incorrect = recovered + removed_bg, added_bg + lost_fg
    return {
        "recovered_fg_count": recovered,
        "added_bg_error_count": added_bg,
        "added_patch_precision": recovered / (recovered + added_bg + 1e-12),
        "lost_fg_count": lost_fg,
        "removed_bg_correct_count": removed_bg,
        "removed_patch_background_precision": removed_bg / (removed_bg + lost_fg + 1e-12),
        "correct_swap_count": correct,
        "incorrect_swap_count": incorrect,
        "native37_swap_accuracy": correct / (correct + incorrect + 1e-12),
        "native37_area_delta": float(candidate_hard.float().mean() - b0_hard.float().mean()),
    }


def _original_swap(b0: torch.Tensor, candidate: torch.Tensor, gt: torch.Tensor) -> dict:
    b0_hard, candidate_hard, target = b0 > 0.5, candidate > 0.5, gt > 0.5
    added = candidate_hard & ~b0_hard
    removed = b0_hard & ~candidate_hard
    added_count, removed_count = int(added.sum()), int(removed.sum())
    added_fg, removed_fg = int((added & target).sum()), int((removed & target).sum())
    added_bg, removed_bg = added_count - added_fg, removed_count - removed_fg
    correct, incorrect = added_fg + removed_bg, added_bg + removed_fg
    total = int(gt.numel())
    return {
        "added_area": added_count / total,
        "removed_area": removed_count / total,
        "hard_disagreement_ratio": (added_count + removed_count) / total,
        "area_balance_error": abs(added_count - removed_count) / total,
        "added_fg_precision": added_fg / added_count if added_count else float("nan"),
        "added_bg_error_ratio": added_bg / added_count if added_count else float("nan"),
        "removed_bg_precision": removed_bg / removed_count if removed_count else float("nan"),
        "removed_fg_error_ratio": removed_fg / removed_count if removed_count else float("nan"),
        "correct_swap_count": correct,
        "incorrect_swap_count": incorrect,
        "swap_accuracy": correct / (correct + incorrect) if correct + incorrect else float("nan"),
    }


def _validate_rpr_payload(payload: dict, dataset: str, stem: str, path: Path):
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"RPR cache key mismatch: {path}")
    if payload.get("rpr_version") != VERSION:
        raise RuntimeError(f"RPR version mismatch: {path}")
    if payload.get("source_augs") != ["identity"] or int(payload.get("source_num_views", 0)) != 1:
        raise RuntimeError(f"RPR source view mismatch: {path}")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _evaluate_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    dabe_path, cvbr_path, rpr_path = map(
        Path, (task["dabe"], task["cvbr"], task["rpr"])
    )
    dabe = torch_load(dabe_path, map_location="cpu")
    cvbr = torch_load(cvbr_path, map_location="cpu")
    rpr = torch_load(rpr_path, map_location="cpu")
    for payload, name, path in (
        (dabe, "DABE", dabe_path),
        (cvbr, "CVBR", cvbr_path),
    ):
        if payload.get("dataset") != dataset or payload.get("stem") != stem:
            raise RuntimeError(f"{name} cache key mismatch: {path}")
    if cvbr.get("cvbr_version") != CVBR_VERSION:
        raise RuntimeError(f"CVBR version mismatch: {cvbr_path}")
    _validate_rpr_payload(rpr, dataset, stem, rpr_path)

    gt = _load_gt(task["gt"])
    gt_shape = tuple(gt.shape[-2:])
    native = {
        method: _load_response(cvbr, field, (37, 37), cvbr_path)
        for method, field in CVBR_FIELDS.items()
    }
    native.update(
        {
            method: _load_response(rpr, field, (37, 37), rpr_path)
            for method, field in RPR_FIELDS.items()
        }
    )
    cached = _load_response(dabe, "residual_pass1_37", (37, 37), dabe_path)
    if float((native[METHODS[0]] - cached).abs().max()) > 1e-6:
        raise RuntimeError(f"B0/cached R1 mismatch: {dataset}/{stem}")
    probabilities = {method: _resize_native(native[method], gt_shape) for method in METHODS[:6]}
    probabilities[METHODS[6]] = _resize_current(
        _load_response(dabe, "p_dabe_68", (68, 68), dabe_path), gt_shape
    )

    context = FastCODContext(gt)
    inputs = []
    for method in METHODS:
        inputs.extend(
            ((method, "hard", probabilities[method]), (method, "soft", probabilities[method]))
        )
    cod = context.evaluate_many(inputs, task["threshold"])
    metric_rows = []
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
            "source_cvbr_cache_path": str(cvbr_path) if method != METHODS[6] else "",
            "rpr_cache_path": str(rpr_path) if method in METHODS[4:6] else "",
            "_official_f_curve": soft["f_curve"],
            "_official_e_curve": soft["e_curve"],
        }
        for field in (*HARD, *OFFICIAL, *RAW, *RANK):
            value = float(row[field])
            if field == "pixel_AP" and math.isnan(value):
                continue
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite metric: {dataset}/{stem}/{method}/{field}")
        metric_rows.append(row)

    gt37 = torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > 0.5
    touch = _touch(gt37)
    b0_hard = native[METHODS[0]] > 0.5
    masks = {
        "FN": gt37 & ~b0_hard,
        "TN": ~gt37 & ~b0_hard,
        "TP": gt37 & b0_hard,
        "FP": ~gt37 & b0_hard,
    }
    error_rows, fn_tn_vectors = [], {}
    for method, tag in ((METHODS[4], "p1"), (METHODS[5], "p2")):
        values = {
            "u_base": rpr[f"u_base_{tag}_37"],
            "topk_prior_min": rpr[f"topk_prior_min_{tag}_37"],
            "topk_prior_mean": rpr[f"topk_prior_mean_{tag}_37"],
            "topk_low_q_ratio": rpr[f"topk_low_q_ratio_{tag}_37"],
            "weight_l1_shift": rpr[f"weight_l1_shift_{tag}_37"],
            "u_reduction": rpr[f"u_reduction_{tag}_37"],
            "delta_raw": rpr[f"delta_raw_{tag}_vs_b0_37"],
        }
        for error_type, mask in masks.items():
            error_rows.append(
                {
                    "method": method,
                    "error_type": error_type,
                    "patch_count": int(mask.sum()),
                    **{field: _mean_on(value, mask) for field, value in values.items()},
                }
            )
        negative = ~b0_hard.reshape(-1)
        labels = (gt37.reshape(-1)[negative]).cpu().numpy().astype(np.uint8, copy=False)
        scores = (
            values["u_base"].reshape(-1)[negative].cpu().numpy().astype(np.float32, copy=False)
        )
        fn_tn_vectors[method] = {
            "labels": labels,
            "scores": scores,
            "image_auc": _auc(labels, scores),
        }

    native_swap, original_swap = [], []
    for method in METHODS[4:6]:
        native_swap.append({"method": method, **_native_swap(native[METHODS[0]], native[method], gt37)})
        original_swap.append(
            {"method": method, **_original_swap(probabilities[METHODS[0]], probabilities[method], gt)}
        )
    query_rows = [_query_exposure(rpr, method) for method in METHODS[4:6]]
    generation = rpr.get("diagnostics")
    if not isinstance(generation, dict):
        raise RuntimeError(f"RPR diagnostics missing: {rpr_path}")
    return {
        "dataset": dataset,
        "stem": stem,
        "metric_rows": metric_rows,
        "touch": touch,
        "query_rows": query_rows,
        "error_rows": error_rows,
        "fn_tn_vectors": fn_tn_vectors,
        "native_swap": native_swap,
        "original_swap": original_swap,
        "generation": generation,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _metric_sections(aggregates, datasets, section):
    fields = {
        "hard": HARD,
        "official_soft": OFFICIAL,
        "raw_continuous": RAW,
        "ranking": RANK,
    }[section]
    by_dataset = []
    for dataset in datasets:
        for method in METHODS:
            result = aggregates[(dataset, method)].result()
            by_dataset.append(
                {
                    "scope": "by_dataset",
                    "dataset": dataset,
                    "method": method,
                    **{field: result[field] for field in fields},
                    "num_samples": result["num_samples"],
                    "ap_valid_count": result["ap_valid_count"],
                }
            )
    sample_overall = []
    for method in METHODS:
        result = aggregates[("ALL", method)].result()
        sample_overall.append(
            {
                "scope": "sample_overall",
                "dataset": "ALL",
                "method": method,
                **{field: result[field] for field in fields},
                "num_samples": result["num_samples"],
                "ap_valid_count": result["ap_valid_count"],
            }
        )
    dataset_macro = []
    for method in METHODS:
        rows = [row for row in by_dataset if row["method"] == method]
        dataset_macro.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                **{field: float(np.mean([row[field] for row in rows])) for field in fields},
                "num_samples": sum(row["num_samples"] for row in rows),
                "ap_valid_count": sum(row["ap_valid_count"] for row in rows),
            }
        )
    return by_dataset, sample_overall, dataset_macro


def _aggregate_mean(records: list[dict], keys, fields):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    output = []
    for values, rows in grouped.items():
        row = {key: value for key, value in zip(keys, values)}
        row["num_samples"] = len(rows)
        for field in fields:
            row[field], row[f"{field}_valid_count"] = _finite_mean(
                item[field] for item in rows
            )
        output.append(row)
    return output


def _aggregate_error_records(records: list[dict], datasets):
    output = []
    for dataset in (*datasets, "ALL"):
        selected_dataset = [
            row for row in records if dataset == "ALL" or row["dataset"] == dataset
        ]
        for method in METHODS[4:6]:
            for error_type in ERROR_TYPES:
                rows = [
                    row
                    for row in selected_dataset
                    if row["method"] == method and row["error_type"] == error_type
                ]
                total = sum(int(row["patch_count"]) for row in rows)
                result = {
                    "dataset": dataset,
                    "method": method,
                    "error_type": error_type,
                    "patch_count": total,
                    "num_samples": len(rows),
                }
                for field in ERROR_DIAGNOSTIC_FIELDS:
                    numerator = sum(
                        float(row[field]) * int(row["patch_count"])
                        for row in rows
                        if math.isfinite(float(row[field]))
                    )
                    valid_count = sum(
                        int(row["patch_count"])
                        for row in rows
                        if math.isfinite(float(row[field]))
                    )
                    result[field] = numerator / valid_count if valid_count else float("nan")
                    result[f"{field}_valid_patch_count"] = valid_count
                output.append(result)
    return output


def _residual_selectivity(error_summary: list[dict], datasets):
    lookup = {
        (row["dataset"], row["method"], row["error_type"]): row
        for row in error_summary
    }
    output = []
    for dataset in (*datasets, "ALL"):
        for method in METHODS[4:6]:
            fn = lookup[(dataset, method, "FN")]
            tn = lookup[(dataset, method, "TN")]
            tp = lookup[(dataset, method, "TP")]
            fp = lookup[(dataset, method, "FP")]
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_delta_raw_FN": fn["delta_raw"],
                    "mean_delta_raw_TN": tn["delta_raw"],
                    "mean_delta_raw_TP": tp["delta_raw"],
                    "mean_delta_raw_FP": fp["delta_raw"],
                    "selectivity": fn["delta_raw"] - tn["delta_raw"],
                    "FN_u_reduction": fn["u_reduction"],
                    "TN_u_reduction": tn["u_reduction"],
                    "FN_weight_shift": fn["weight_l1_shift"],
                    "TN_weight_shift": tn["weight_l1_shift"],
                    "FN_patch_count": fn["patch_count"],
                    "TN_patch_count": tn["patch_count"],
                }
            )
    return output


def _fn_tn_tables(vector_records: list[dict], datasets):
    auc_rows, enrichment_rows = [], []
    for dataset in (*datasets, "ALL"):
        records = [
            row for row in vector_records if dataset == "ALL" or row["dataset"] == dataset
        ]
        for method in METHODS[4:6]:
            labels = np.concatenate([row["vectors"][method]["labels"] for row in records])
            scores = np.concatenate([row["vectors"][method]["scores"] for row in records])
            image_auc = [row["vectors"][method]["image_auc"] for row in records]
            valid_auc = [value for value in image_auc if math.isfinite(float(value))]
            pooled_auc = _auc(labels, scores)
            auc_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "pooled_auc": pooled_auc,
                    "mean_image_auc": float(np.mean(valid_auc)) if valid_auc else float("nan"),
                    "valid_image_count": len(valid_auc),
                    "num_samples": len(records),
                    "fn_count": int(labels.sum()),
                    "tn_count": int((~labels.astype(bool)).sum()),
                }
            )
            top_count = max(1, int(math.ceil(0.1 * labels.size)))
            top_index = np.argsort(-scores, kind="stable")[:top_count]
            base_rate = float(labels.mean())
            top_rate = float(labels[top_index].mean())
            enrichment_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "base_FN_rate": base_rate,
                    "top10_FN_rate": top_rate,
                    "top10_FN_enrichment": top_rate / (base_rate + 1e-12),
                    "hard_negative_count": int(labels.size),
                    "top10_count": top_count,
                }
            )
    return auc_rows, enrichment_rows


def _aggregate_native_swap(records: list[dict], datasets):
    output = []
    for dataset in (*datasets, "ALL"):
        for method in METHODS[4:6]:
            rows = [
                row
                for row in records
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            recovered = sum(int(row["recovered_fg_count"]) for row in rows)
            added_bg = sum(int(row["added_bg_error_count"]) for row in rows)
            lost_fg = sum(int(row["lost_fg_count"]) for row in rows)
            removed_bg = sum(int(row["removed_bg_correct_count"]) for row in rows)
            correct, incorrect = recovered + removed_bg, added_bg + lost_fg
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "recovered_fg_count": recovered,
                    "added_bg_error_count": added_bg,
                    "added_patch_precision": recovered / (recovered + added_bg + 1e-12),
                    "lost_fg_count": lost_fg,
                    "removed_bg_correct_count": removed_bg,
                    "removed_patch_background_precision": removed_bg / (removed_bg + lost_fg + 1e-12),
                    "correct_swap_count": correct,
                    "incorrect_swap_count": incorrect,
                    "native37_swap_accuracy": correct / (correct + incorrect + 1e-12),
                    "native37_area_delta": float(np.mean([row["native37_area_delta"] for row in rows])),
                    "num_samples": len(rows),
                }
            )
    return output


def _aggregate_original_swap(records: list[dict], datasets):
    output = []
    for dataset in (*datasets, "ALL"):
        for method in METHODS[4:6]:
            rows = [
                row
                for row in records
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            result = {"dataset": dataset, "method": method, "num_samples": len(rows)}
            for field in ORIGINAL_SWAP_FIELDS:
                result[field], result[f"{field}_valid_count"] = _finite_mean(
                    row[field] for row in rows
                )
            output.append(result)
    return output


def _pairwise_summary(records, keys):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in (*keys, "method_a", "method_b"))].append(record)
    output = []
    for values, rows in grouped.items():
        base = {
            key: value
            for key, value in zip((*keys, "method_a", "method_b"), values)
        }
        for field in PAIR_FIELDS:
            array = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            output.append(
                {
                    **base,
                    "metric": field,
                    "mean_delta": float(array.mean()),
                    "median_delta": float(np.median(array)),
                    "p25_delta": float(np.percentile(array, 25)),
                    "p75_delta": float(np.percentile(array, 75)),
                    "win_ratio": float(np.mean(array > 1e-6)),
                    "tie_ratio": float(np.mean(np.abs(array) <= 1e-6)),
                    "loss_ratio": float(np.mean(array < -1e-6)),
                    "valid_count": len(array),
                }
            )
    return output


def _swap_overlay(b0: torch.Tensor, candidate: torch.Tensor, gt: torch.Tensor, size: int):
    b0_hard, candidate_hard, target = b0 > 0.5, candidate > 0.5, gt > 0.5
    added, removed = candidate_hard & ~b0_hard, b0_hard & ~candidate_hard
    correct = (added & target) | (removed & ~target)
    incorrect = (added & ~target) | (removed & target)
    canvas = np.full((*b0_hard.shape[-2:], 3), 32, dtype=np.uint8)
    canvas[correct.squeeze().numpy()] = (40, 200, 70)
    canvas[incorrect.squeeze().numpy()] = (230, 50, 50)
    return Image.fromarray(canvas).resize((size, size), Image.Resampling.NEAREST)


def _save_one_visual(task: dict, output_path: Path):
    cvbr = torch_load(task["cvbr"], map_location="cpu")
    rpr = torch_load(task["rpr"], map_location="cpu")
    gt = _load_gt(task["gt"])
    size, label_height, columns = 150, 22, 5
    with Image.open(task["image"]) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    b0 = cvbr[CVBR_FIELDS[METHODS[0]]]
    p1, p2 = rpr[RPR_FIELDS[METHODS[4]]], rpr[RPR_FIELDS[METHODS[5]]]
    panels = [
        ("RGB", rgb),
        ("GT", _gray_panel(gt, size, True)),
        ("B0 R1", _gray_panel(b0, size)),
        ("B1 BW1", _gray_panel(cvbr[CVBR_FIELDS[METHODS[1]]], size)),
        ("V1 CVBR-SR", _gray_panel(cvbr[CVBR_FIELDS[METHODS[2]]], size)),
        ("V2 CVBR-All", _gray_panel(cvbr[CVBR_FIELDS[METHODS[3]]], size)),
        ("P1 RPR-SR", _gray_panel(p1, size)),
        ("P2 RPR-All", _gray_panel(p2, size)),
        ("B0 anchor", _gray_panel(cvbr["anchor_b0_37"], size, True)),
        ("P1 atom prior", _gray_panel(rpr["atom_prior_p1_37"], size)),
        ("P2 atom prior", _gray_panel(rpr["atom_prior_p2_37"], size)),
        ("U base P1", _gray_panel(rpr["u_base_p1_37"], size)),
        ("U base P2", _gray_panel(rpr["u_base_p2_37"], size)),
        ("U reduction P1", _gray_panel(rpr["u_reduction_p1_37"], size)),
        ("U reduction P2", _gray_panel(rpr["u_reduction_p2_37"], size)),
        ("WeightShift P1", _gray_panel(rpr["weight_l1_shift_p1_37"], size)),
        ("WeightShift P2", _gray_panel(rpr["weight_l1_shift_p2_37"], size)),
        ("RawDelta P1-B0", _difference_panel(rpr["delta_raw_p1_vs_b0_37"], size)),
        ("RawDelta P2-B0", _difference_panel(rpr["delta_raw_p2_vs_b0_37"], size)),
        ("P1-B0 swap", _swap_overlay(b0, p1, torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0), size)),
        ("P2-B0 swap", _swap_overlay(b0, p2, torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0), size)),
    ]
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * size, rows * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x, y = (index % columns) * size, (index // columns) * (size + label_height)
        canvas.paste(panel, (x, y + label_height))
        draw.text((x + 3, y + 4), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_visualizations(tasks, touch_map, exposure_map, delta_map, out_dir, datasets):
    grouped = defaultdict(list)
    for task in tasks:
        grouped[task["dataset"]].append(task)
    for dataset in datasets:
        rows = grouped[dataset]
        for task in rows[:8]:
            _save_one_visual(
                task, out_dir / "vis" / "fixed_first8" / dataset / f"{task['stem']}.png"
            )
        for subset, folder in (
            ("Touch-1", "touch1_first16"),
            ("Touch-2-only", "touch2only_first16"),
        ):
            selected = [
                task
                for task in rows
                if touch_map[(dataset, task["stem"])] == subset
            ][:16]
            for task in selected:
                _save_one_visual(
                    task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png"
                )
        exposure_ranked = sorted(
            rows,
            key=lambda task: exposure_map[(dataset, task["stem"], METHODS[4])],
            reverse=True,
        )
        for folder, selected in (
            ("p1_exposure_top8", exposure_ranked[:8]),
            ("p1_exposure_bottom8", exposure_ranked[-8:]),
        ):
            for task in selected:
                _save_one_visual(
                    task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png"
                )
        for method, tag in ((METHODS[4], "P1_vs_B0"), (METHODS[5], "P2_vs_B0")):
            ranked = sorted(
                rows,
                key=lambda task: delta_map[(dataset, task["stem"], method)],
                reverse=True,
            )
            for folder, selected in (
                (f"{tag}_top8", ranked[:8]),
                (f"{tag}_bottom8", ranked[-8:]),
            ):
                for task in selected:
                    _save_one_visual(
                        task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png"
                    )


def _table(rows, fields):
    def format_value(value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, np.integer)):
            return str(value)
        try:
            number = float(value)
            return f"{number:.6f}" if math.isfinite(number) else "NaN"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    lines.extend(
        "| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _lookup(rows, **conditions):
    return next(
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    )


def _assess(summary):
    macro = {row["method"]: row for row in summary["hard"]["dataset_macro"]}
    hard_by = {
        (row["dataset"], row["method"]): row
        for row in summary["hard"]["by_dataset"]
    }
    ranking = {row["method"]: row for row in summary["ranking"]["dataset_macro"]}
    touch = {
        (row["touch_subset"], row["method"]): row
        for row in summary["touch_subset_hard"]
        if row["dataset"] == "ALL"
    }
    auc = {
        row["method"]: row
        for row in summary["fn_tn_auc_overall"]
        if row["dataset"] == "ALL"
    }
    enrichment = {
        row["method"]: row
        for row in summary["fn_top10_enrichment"]
        if row["dataset"] == "ALL"
    }
    selectivity = {
        row["method"]: row
        for row in summary["residual_selectivity_overall"]
        if row["dataset"] == "ALL"
    }
    native = {
        row["method"]: row
        for row in summary["native37_swap_overall"]
        if row["dataset"] == "ALL"
    }
    query = {
        row["method"]: row
        for row in summary["query_exposure_by_dataset"]
        if row["dataset"] == "ALL"
    }
    b0 = macro[METHODS[0]]
    candidates = {}
    for method in METHODS[4:6]:
        dataset_delta = {
            dataset: hard_by[(dataset, method)]["hard_F_beta_w"]
            - hard_by[(dataset, METHODS[0])]["hard_F_beta_w"]
            for dataset in DATASETS
        }
        touch_delta = {
            subset: touch[(subset, method)]["hard_F_beta_w"]
            - touch[(subset, METHODS[0])]["hard_F_beta_w"]
            for subset in SUBSETS
        }
        mechanism_supported = (
            auc[method]["pooled_auc"] >= 0.60
            or enrichment[method]["top10_FN_enrichment"] >= 1.5
        )
        engineering = (
            macro[method]["hard_F_beta_w"] > 0.624280
            and macro[method]["hard_MAE"] <= 0.080969
            and min(dataset_delta.values()) >= -0.005
        )
        paper = (
            macro[method]["hard_F_beta_w"] >= 0.625780
            and sum(value >= 0 for value in dataset_delta.values()) >= 3
            and min(dataset_delta.values()) >= -0.003
            and macro[method]["hard_MAE"] <= 0.080469
            and ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"]
            and max(touch_delta["Touch-1"], touch_delta["Touch-2-only"]) >= 0.005
            and touch_delta["Non-touch"] >= -0.002
            and selectivity[method]["selectivity"] > 0
            and native[method]["added_patch_precision"] > 0.5
        )
        clear = (
            macro[method]["hard_F_beta_w"] >= 0.6270
            and sum(value > 0 for value in dataset_delta.values()) >= 3
            and (
                macro[method]["hard_S_m"] >= b0["hard_S_m"]
                or macro[method]["hard_E_mean"] >= b0["hard_E_mean"]
            )
            and macro[method]["hard_MAE"] <= b0["hard_MAE"]
            and ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"]
            and max(touch_delta["Touch-1"], touch_delta["Touch-2-only"]) > 0
            and touch_delta["Non-touch"] >= -0.002
            and auc[method]["pooled_auc"] >= 0.60
            and selectivity[method]["selectivity"] > 0
        )
        strong = (
            macro[method]["hard_F_beta_w"] >= 0.6290
            and min(dataset_delta.values()) >= -0.001
            and sum(
                (
                    macro[method]["hard_S_m"] >= b0["hard_S_m"],
                    macro[method]["hard_E_mean"] >= b0["hard_E_mean"],
                    macro[method]["hard_MAE"] <= b0["hard_MAE"],
                    ranking[method]["pixel_AP"] >= ranking[METHODS[0]]["pixel_AP"],
                )
            )
            >= 3
            and max(touch_delta["Touch-1"], touch_delta["Touch-2-only"]) >= 0.005
            and touch_delta["Non-touch"] >= -0.001
        )
        candidates[method] = {
            "dataset_macro_F_beta_w_delta": macro[method]["hard_F_beta_w"]
            - b0["hard_F_beta_w"],
            "dataset_F_beta_w_delta": dataset_delta,
            "touch_F_beta_w_delta": touch_delta,
            "mechanism_supported": mechanism_supported,
            "pooled_FN_vs_TN_auc": auc[method]["pooled_auc"],
            "top10_FN_enrichment": enrichment[method]["top10_FN_enrichment"],
            "selectivity": selectivity[method]["selectivity"],
            "U_reduction_FN_gt_TN": selectivity[method]["FN_u_reduction"]
            > selectivity[method]["TN_u_reduction"],
            "added_patch_precision": native[method]["added_patch_precision"],
            "mean_U_base": query[method]["mean_U_base"],
            "engineering_basic_success": engineering,
            "paper_retention_success": paper,
            "clear_success": clear,
            "strong_success": strong,
        }

    paper_methods = [method for method, values in candidates.items() if values["paper_retention_success"]]
    if paper_methods:
        frozen = max(paper_methods, key=lambda method: macro[method]["hard_F_beta_w"])
        branch = "A" if frozen == METHODS[4] and not candidates[METHODS[5]]["paper_retention_success"] else "B"
        support_audit, stop_reliability, turn_hr = False, False, False
    else:
        frozen = METHODS[0]
        touch_improved = any(
            max(values["touch_F_beta_w_delta"]["Touch-1"], values["touch_F_beta_w_delta"]["Touch-2-only"])
            > 0
            for values in candidates.values()
        )
        slight = any(
            0 < values["dataset_macro_F_beta_w_delta"] < 0.0015
            for values in candidates.values()
        )
        rank_only = any(
            ranking[method]["pixel_AP"] > ranking[METHODS[0]]["pixel_AP"]
            and macro[method]["hard_F_beta_w"] <= b0["hard_F_beta_w"]
            for method in METHODS[4:6]
        )
        mechanism = any(values["mechanism_supported"] for values in candidates.values())
        low_exposure = all(values["mean_U_base"] < 0.01 for values in candidates.values())
        nonselective = all(values["selectivity"] <= 0 for values in candidates.values())
        clearly_down = all(
            values["dataset_macro_F_beta_w_delta"] <= -0.0015
            for values in candidates.values()
        )
        if touch_improved:
            branch, support_audit, stop_reliability, turn_hr = "C", False, False, False
        elif mechanism and low_exposure:
            branch, support_audit, stop_reliability, turn_hr = "D1", True, False, False
        elif mechanism and nonselective:
            branch, support_audit, stop_reliability, turn_hr = "D2", False, True, True
        elif rank_only:
            branch, support_audit, stop_reliability, turn_hr = "E", False, False, True
        elif slight:
            branch, support_audit, stop_reliability, turn_hr = "F", False, False, True
        elif clearly_down:
            branch, support_audit, stop_reliability, turn_hr = "G", False, True, True
        else:
            branch, support_audit, stop_reliability, turn_hr = "No-retention", False, False, True
    return {
        "candidates": candidates,
        "frozen_method": frozen,
        "result_branch": branch,
        "support_boundary_source_attribution_audit": support_audit,
        "stop_background_reliability_direction": stop_reliability,
        "turn_to_high_resolution_patch_to_pixel_projection": turn_hr,
    }


def _write_docs(out_dir: Path, summary: dict, full_run: bool):
    protocol = summary["protocol"]
    readme = [
        "# DABE RPR-v1 Evaluation",
        "",
        "本目录固定比较 B0、B1、V1、V2、P1、P2 与 Current-DABE-v2。",
        "P1/P2 候选生成不读取 GT；GT 仅在本评测阶段用于指标、触边分组与机制审计。",
        "",
        "- B0/B1/V1/V2/P1/P2：37×37 → bilinear 68×68 → bilinear 原始 GT 尺寸，`align_corners=False`。",
        "- Current-DABE-v2：`p_dabe_68` → bilinear 原始 GT 尺寸。",
        "- Hard：严格 `prob > 0.5`。",
        "- Official Soft：连续响应进入当前 `CODMetrics` 等价公式；当前实现会对非恒定预测逐图 Min-Max。",
        "- Best IoU threshold 仅为 256 个固定阈值上的排序诊断，不参与主结果。",
        "- 最终版本判断以四数据集不加权 Dataset Macro 为主。",
        "",
        f"样本数：{protocol['num_samples']}；full run：{full_run}；可视化：{protocol['save_vis']}。",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    macro = summary["hard"]["dataset_macro"]
    by_dataset = summary["hard"]["by_dataset"]
    official = summary["official_soft"]["dataset_macro"]
    ranking = summary["ranking"]["dataset_macro"]
    touch = [row for row in summary["touch_subset_hard"] if row["dataset"] == "ALL"]
    auc = summary["fn_tn_auc_overall"]
    enrichment = [row for row in summary["fn_top10_enrichment"] if row["dataset"] == "ALL"]
    selectivity = summary["residual_selectivity_overall"]
    native = summary["native37_swap_overall"]
    original = summary["original_size_swap_overall"]
    query = [row for row in summary["query_exposure_by_dataset"] if row["dataset"] == "ALL"]
    generation = summary["generation_diagnostics_overall"][0]
    baseline = summary["baseline_reproduction"]
    assessment = summary["success_assessment"]
    status = "PASS" if baseline.get("passed") else ("SANITY" if not full_run else "FAIL")

    lines = [
        "# DABE-TF RPR-v1 Results",
        "",
        f"协议状态：**{status}**。正式结论仅在 full run 且基线复现通过时有效。",
        "",
        "## Dataset Macro Hard",
        "",
        _table(
            macro,
            (
                "method",
                "hard_S_m",
                "hard_F_beta_w",
                "hard_F_beta_mean",
                "hard_E_mean",
                "hard_MAE",
                "hard_IoU",
                "hard_Precision",
                "hard_Recall",
                "hard_Area",
            ),
        ),
        "",
        "## 四数据集 Hard Fβw",
        "",
        _table(by_dataset, ("dataset", "method", "hard_F_beta_w", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area")),
        "",
        "## Dataset Macro Official Soft",
        "",
        _table(official, ("method", "official_soft_S_m", "official_soft_F_beta_w", "official_soft_F_beta_mean", "official_soft_E_mean", "official_soft_MAE")),
        "",
        "当前 CODMetrics 对非恒定 continuous prediction 会逐图 Min-Max。",
        "",
        "## Dataset Macro Ranking",
        "",
        _table(ranking, ("method", "pixel_AP", "best_IoU_256", "best_IoU_threshold_256", "ranking_F_beta_max", "ranking_E_max")),
        "",
        "## Touch 子集 Hard",
        "",
        _table(touch, ("touch_subset", "method", "hard_F_beta_w", "hard_MAE", "hard_Precision", "hard_Recall", "hard_Area", "num_samples")),
        "",
        "## RPR 机制主表",
        "",
        "### Query exposure",
        "",
        _table(query, ("method", *QUERY_FIELDS, "num_samples")),
        "",
        "### FN-vs-TN AUC 与 Top-10% 富集",
        "",
        _table(auc, ("method", "pooled_auc", "mean_image_auc", "valid_image_count", "fn_count", "tn_count")),
        "",
        _table(enrichment, ("method", "base_FN_rate", "top10_FN_rate", "top10_FN_enrichment", "hard_negative_count")),
        "",
        "### Raw residual selectivity",
        "",
        _table(selectivity, ("method", "mean_delta_raw_FN", "mean_delta_raw_TN", "selectivity", "FN_u_reduction", "TN_u_reduction", "FN_weight_shift", "TN_weight_shift")),
        "",
        "### Native-37 与原图交换精度",
        "",
        _table(native, ("method", "recovered_fg_count", "added_bg_error_count", "added_patch_precision", "native37_swap_accuracy", "native37_area_delta")),
        "",
        _table(original, ("method", "added_area", "removed_area", "added_fg_precision", "removed_bg_precision", "swap_accuracy")),
        "",
        "## 回归控制",
        "",
        f"- B0 cached R1 最大误差：`{generation.get('b0_cached_r1_max_abs', float('nan')):.12g}`",
        f"- B0 重算最大误差：`{generation.get('b0_recomputed_cached_r1_max_abs', float('nan')):.12g}`",
        f"- Unit-prior top-K mismatch：`{generation.get('unit_prior_topk_mismatch_count', float('nan')):.6f}`（逐图平均；全局总数见 cache protocol）",
        f"- Unit-prior weight 最大误差：`{generation.get('unit_prior_weight_max_abs', float('nan')):.12g}`",
        f"- Unit-prior raw 最大误差：`{generation.get('unit_prior_raw_max_abs', float('nan')):.12g}`",
        f"- Unit-prior normalized 最大误差：`{generation.get('unit_prior_normalized_max_abs', float('nan')):.12g}`",
        "",
        "## 任务书问题的定量回答",
        "",
    ]
    if full_run and baseline.get("passed"):
        c = assessment["candidates"]
        p1, p2 = c[METHODS[4]], c[METHODS[5]]
        lines.extend(
            [
                f"1. B0 精确复现：是；V1/V2 精确复现：是；Current-DABE-v2 精确复现：是。",
                f"2. Unit-prior 与 B0 的回归控制通过；全局 top-K mismatch 为 {summary['cache_protocol']['unit_prior_topk_mismatch_count']}。",
                f"3. P1/P2 修改原子数量、先验分布和生成诊断见 generation CSV；P1/P2 平均 U 分别为 {p1['mean_U_base']:.6f}/{p2['mean_U_base']:.6f}。",
                f"4. FN-vs-TN pooled AUC：P1={p1['pooled_FN_vs_TN_auc']:.6f}，P2={p2['pooled_FN_vs_TN_auc']:.6f}。",
                f"5. Top-10% FN 富集：P1={p1['top10_FN_enrichment']:.6f}，P2={p2['top10_FN_enrichment']:.6f}。",
                f"6. Selectivity：P1={p1['selectivity']:.6f}，P2={p2['selectivity']:.6f}。",
                f"7. Native-37 新增 Patch 前景精度：P1={p1['added_patch_precision']:.6f}，P2={p2['added_patch_precision']:.6f}。",
                "8. Precision、Recall、Area、四数据集和三个触边子集的完整数值见上表与 CSV。",
                "9. P1 vs B0/V1/B1、P2 vs B0/V2/B1、P2 vs P1 的逐数据集和逐子集差值见 pairwise CSV。",
                f"10. 工程基本成功：{[method for method, value in c.items() if value['engineering_basic_success']]}。",
                f"11. 论文保留标准：{[method for method, value in c.items() if value['paper_retention_success']]}。",
                f"12. 最终冻结：{assessment['frozen_method']}；结果分支：{assessment['result_branch']}。",
                f"13. 支持 Boundary-Source Attribution Audit：{assessment['support_boundary_source_attribution_audit']}。",
                f"14. 停止背景可靠性方向：{assessment['stop_background_reliability_direction']}。",
                f"15. 转向高分辨率 Patch-to-Pixel Projection：{assessment['turn_to_high_resolution_patch_to_pixel_projection']}。",
            ]
        )
    else:
        lines.append("基线复现未通过或当前仅为 sanity，不给出 P1/P2 性能与路线结论。")
    lines.extend(
        [
            "",
            "## 成功与分支判定",
            "",
            "```json",
            json.dumps(assessment, ensure_ascii=False, indent=2),
            "```",
            "",
            "所有逐样本、逐数据集、逐触边子集及机制诊断均保存在同目录 CSV。",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_rpr(
    config_path,
    dabe_root,
    cvbr_root,
    rpr_root,
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
    config_path, dabe_root, cvbr_root, rpr_root, out_dir = map(
        lambda value: Path(value).resolve(),
        (config_path, dabe_root, cvbr_root, rpr_root, out_dir),
    )
    if split != "test" or threshold != 0.5 or max_samples == 0 or max_samples < -1:
        raise ValueError("frozen RPR evaluation protocol violation")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    full_run = max_samples == -1
    dabe_manifest = dabe_root / "manifest_test.jsonl"
    cvbr_manifest = cvbr_root / "manifest_test.jsonl"
    rpr_manifest = rpr_root / "manifest_test.jsonl"
    dabe_rows, dabe_map = _manifest_map(dabe_manifest, 6473)
    cvbr_rows, cvbr_map = _manifest_map(cvbr_manifest, 6473)
    selected = dabe_rows if full_run else dabe_rows[:max_samples]
    rpr_rows, rpr_map = _manifest_map(rpr_manifest, 6473 if full_run else max_samples)
    source_keys = {(str(row["dataset"]), str(row["stem"])) for row in dabe_rows}
    if source_keys != set(cvbr_map):
        raise RuntimeError("CVBR keys differ from DABE source keys")
    selected_keys = {(str(row["dataset"]), str(row["stem"])) for row in selected}
    if selected_keys != set(rpr_map):
        raise RuntimeError("RPR keys differ from selected DABE source keys")
    cache_protocol_path = rpr_root / "protocol.json"
    cache_protocol = json.loads(cache_protocol_path.read_text(encoding="utf-8"))
    if cache_protocol.get("rpr_version") != VERSION:
        raise RuntimeError("RPR cache protocol version mismatch")
    if cache_protocol.get("source_dabe_manifest_sha256") != _sha256(dabe_manifest):
        raise RuntimeError("RPR/DABE manifest hash mismatch")
    if cache_protocol.get("source_cvbr_manifest_sha256") != _sha256(cvbr_manifest):
        raise RuntimeError("RPR/CVBR manifest hash mismatch")
    if int(cache_protocol.get("num_samples", -1)) != len(rpr_rows):
        raise RuntimeError("RPR cache protocol sample count mismatch")
    if cache_protocol.get("unit_prior_topk_mismatch_count") != 0:
        raise RuntimeError("RPR unit-prior TopK regression failed")
    if float(cache_protocol.get("unit_prior_weight_max_abs", float("inf"))) > 1e-7:
        raise RuntimeError("RPR unit-prior weight regression failed")
    if float(cache_protocol.get("unit_prior_raw_max_abs", float("inf"))) > 1e-6:
        raise RuntimeError("RPR unit-prior raw regression failed")
    if float(cache_protocol.get("unit_prior_normalized_max_abs", float("inf"))) > 1e-6:
        raise RuntimeError("RPR unit-prior normalized regression failed")

    datasets = [dataset for dataset in DATASETS if any(key[0] == dataset for key in selected_keys)]
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite {out_dir}")
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
        image_path, gt_path = Path(row["image_path"]), Path(row["gt_path"])
        if not image_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(key)
        tasks.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "dabe": dabe_map[key]["cache_path"],
                "cvbr": cvbr_map[key]["cache_path"],
                "rpr": rpr_map[key]["cache_path"],
                "image": str(image_path),
                "gt": str(gt_path),
                "threshold": threshold,
            }
        )

    aggregates = {
        (dataset, method): MetricAggregate()
        for dataset in (*datasets, "ALL")
        for method in METHODS
    }
    touch_metric_rows: list[dict] = []
    touch_metadata: list[dict] = []
    query_records: list[dict] = []
    error_records: list[dict] = []
    vector_records: list[dict] = []
    native_records: list[dict] = []
    original_records: list[dict] = []
    generation_records: list[dict] = []
    pair_records: list[dict] = []
    touch_map, exposure_map, delta_map = {}, {}, {}
    peak_worker = 0.0
    per_fields = (
        "dataset",
        "stem",
        "method",
        *HARD,
        *OFFICIAL,
        *RAW,
        *RANK,
        "source_dabe_cache_path",
        "source_cvbr_cache_path",
        "rpr_cache_path",
    )
    pair_header = ("dataset", "stem", "touch_subset", "method_a", "method_b", *PAIR_FIELDS)
    try:
        log(f"num_samples = {len(tasks)}")
        log(f"workers = {workers}")
        log("hard_threshold = strict > 0.5")
        log("candidate_generation_in_eval = false")
        with (
            (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as per_handle,
            (out_dir / "pairwise_per_sample.csv").open("w", newline="", encoding="utf-8") as pair_handle,
        ):
            per_writer = csv.DictWriter(per_handle, fieldnames=per_fields, extrasaction="ignore")
            pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_header, extrasaction="ignore")
            per_writer.writeheader()
            pair_writer.writeheader()
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(torch_threads,),
            ) as executor:
                for index, result in enumerate(executor.map(_evaluate_one, tasks, chunksize=1), 1):
                    dataset, stem, touch = result["dataset"], result["stem"], result["touch"]
                    peak_worker = max(peak_worker, result["worker_peak_rss_mb"])
                    metric_map = {row["method"]: row for row in result["metric_rows"]}
                    for row in result["metric_rows"]:
                        aggregates[(dataset, row["method"])].add(row)
                        aggregates[("ALL", row["method"])].add(row)
                        per_writer.writerow(row)
                        touch_metric_rows.append(
                            {key: row[key] for key in ("dataset", "stem", "method", *HARD)}
                            | touch
                        )
                    touch_map[(dataset, stem)] = touch["touch_subset"]
                    touch_metadata.append({"dataset": dataset, "stem": stem, **touch})
                    for row in result["query_rows"]:
                        record = {
                            "dataset": dataset,
                            "stem": stem,
                            "touch_subset": touch["touch_subset"],
                            **row,
                        }
                        query_records.append(record)
                        exposure_map[(dataset, stem, row["method"])] = row["mean_U_base"]
                    for row in result["error_rows"]:
                        error_records.append(
                            {
                                "dataset": dataset,
                                "stem": stem,
                                "touch_subset": touch["touch_subset"],
                                **row,
                            }
                        )
                    vector_records.append(
                        {
                            "dataset": dataset,
                            "stem": stem,
                            "vectors": result["fn_tn_vectors"],
                        }
                    )
                    for row in result["native_swap"]:
                        native_records.append(
                            {"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], **row}
                        )
                    for row in result["original_swap"]:
                        original_records.append(
                            {"dataset": dataset, "stem": stem, "touch_subset": touch["touch_subset"], **row}
                        )
                    generation_records.append({"dataset": dataset, "stem": stem, **result["generation"]})
                    for method in METHODS[4:6]:
                        delta_map[(dataset, stem, method)] = (
                            metric_map[method]["hard_IoU"] - metric_map[METHODS[0]]["hard_IoU"]
                        )
                    for method_a, method_b in PAIRWISE:
                        pair = {
                            "dataset": dataset,
                            "stem": stem,
                            "touch_subset": touch["touch_subset"],
                            "method_a": method_a,
                            "method_b": method_b,
                        }
                        for delta_field, source_field in PAIR_SOURCE.items():
                            pair[delta_field] = float(metric_map[method_a][source_field]) - float(
                                metric_map[method_b][source_field]
                            )
                        pair_records.append(pair)
                        pair_writer.writerow(pair)
                    if index == len(tasks) or index % 100 == 0:
                        log(f"processed = {index}/{len(tasks)}")

        sections = {}
        for section, fields in (
            ("hard", HARD),
            ("official_soft", OFFICIAL),
            ("raw_continuous", RAW),
            ("ranking", RANK),
        ):
            by_dataset, overall, macro = _metric_sections(aggregates, datasets, section)
            sections[section] = {
                "by_dataset": by_dataset,
                "sample_overall": overall,
                "dataset_macro": macro,
            }
            header = ("scope", "dataset", "method", *fields, "num_samples", "ap_valid_count")
            for suffix, rows in (
                ("by_dataset", by_dataset),
                ("sample_overall", overall),
                ("dataset_macro", macro),
            ):
                _write_csv(out_dir / f"{section}_{suffix}.csv", rows, header)

        touch_counts, touch_hard = [], []
        for dataset in (*datasets, "ALL"):
            for subset in SUBSETS:
                count = sum(
                    1
                    for row in touch_metadata
                    if (dataset == "ALL" or row["dataset"] == dataset)
                    and row["touch_subset"] == subset
                )
                touch_counts.append(
                    {"dataset": dataset, "touch_subset": subset, "num_samples": count}
                )
                for method in METHODS:
                    rows = [
                        row
                        for row in touch_metric_rows
                        if (dataset == "ALL" or row["dataset"] == dataset)
                        and row["touch_subset"] == subset
                        and row["method"] == method
                    ]
                    touch_hard.append(
                        {
                            "dataset": dataset,
                            "touch_subset": subset,
                            "method": method,
                            **{field: _finite_mean(row[field] for row in rows)[0] for field in HARD},
                            "num_samples": len(rows),
                        }
                    )
        _write_csv(out_dir / "touch_subset_counts.csv", touch_counts, ("dataset", "touch_subset", "num_samples"))
        _write_csv(out_dir / "touch_subset_hard.csv", touch_hard, ("dataset", "touch_subset", "method", *HARD, "num_samples"))

        query_header = ("dataset", "stem", "touch_subset", "method", *QUERY_FIELDS)
        _write_csv(out_dir / "query_exposure_per_sample.csv", query_records, query_header)
        query_dataset_records = query_records + [{**row, "dataset": "ALL"} for row in query_records]
        query_by_dataset = _aggregate_mean(query_dataset_records, ("dataset", "method"), QUERY_FIELDS)
        query_subset_records = query_records + [{**row, "dataset": "ALL"} for row in query_records]
        query_by_subset = _aggregate_mean(
            query_subset_records, ("dataset", "touch_subset", "method"), QUERY_FIELDS
        )
        query_summary_header = (
            *QUERY_FIELDS,
            "num_samples",
            *(f"{field}_valid_count" for field in QUERY_FIELDS),
        )
        _write_csv(out_dir / "query_exposure_by_dataset.csv", query_by_dataset, ("dataset", "method", *query_summary_header))
        _write_csv(out_dir / "query_exposure_by_subset.csv", query_by_subset, ("dataset", "touch_subset", "method", *query_summary_header))

        error_summary = _aggregate_error_records(error_records, datasets)
        error_header = (
            "dataset",
            "method",
            "error_type",
            "patch_count",
            "num_samples",
            *ERROR_DIAGNOSTIC_FIELDS,
            *(f"{field}_valid_patch_count" for field in ERROR_DIAGNOSTIC_FIELDS),
        )
        _write_csv(
            out_dir / "r1_error_type_diagnostics_by_dataset.csv",
            [row for row in error_summary if row["dataset"] != "ALL"],
            error_header,
        )
        _write_csv(
            out_dir / "r1_error_type_diagnostics_overall.csv",
            [row for row in error_summary if row["dataset"] == "ALL"],
            error_header,
        )
        auc_rows, enrichment_rows = _fn_tn_tables(vector_records, datasets)
        auc_header = (
            "dataset",
            "method",
            "pooled_auc",
            "mean_image_auc",
            "valid_image_count",
            "num_samples",
            "fn_count",
            "tn_count",
        )
        _write_csv(out_dir / "fn_tn_auc_by_dataset.csv", [row for row in auc_rows if row["dataset"] != "ALL"], auc_header)
        _write_csv(out_dir / "fn_tn_auc_overall.csv", [row for row in auc_rows if row["dataset"] == "ALL"], auc_header)
        _write_csv(
            out_dir / "fn_top10_enrichment.csv",
            enrichment_rows,
            ("dataset", "method", "base_FN_rate", "top10_FN_rate", "top10_FN_enrichment", "hard_negative_count", "top10_count"),
        )
        residual = _residual_selectivity(error_summary, datasets)
        residual_header = tuple(residual[0].keys())
        _write_csv(out_dir / "residual_selectivity_by_dataset.csv", [row for row in residual if row["dataset"] != "ALL"], residual_header)
        _write_csv(out_dir / "residual_selectivity_overall.csv", [row for row in residual if row["dataset"] == "ALL"], residual_header)

        native_by = _aggregate_native_swap(native_records, datasets)
        native_per_header = ("dataset", "stem", "touch_subset", "method", *NATIVE_SWAP_FIELDS)
        _write_csv(out_dir / "native37_swap_per_sample.csv", native_records, native_per_header)
        native_header = ("dataset", "method", *NATIVE_SWAP_FIELDS, "num_samples")
        _write_csv(out_dir / "native37_swap_by_dataset.csv", [row for row in native_by if row["dataset"] != "ALL"], native_header)
        _write_csv(out_dir / "native37_swap_overall.csv", [row for row in native_by if row["dataset"] == "ALL"], native_header)
        original_by = _aggregate_original_swap(original_records, datasets)
        original_per_header = ("dataset", "stem", "touch_subset", "method", *ORIGINAL_SWAP_FIELDS)
        _write_csv(out_dir / "original_size_swap_per_sample.csv", original_records, original_per_header)
        original_header = (
            "dataset",
            "method",
            *ORIGINAL_SWAP_FIELDS,
            "num_samples",
            *(f"{field}_valid_count" for field in ORIGINAL_SWAP_FIELDS),
        )
        _write_csv(out_dir / "original_size_swap_by_dataset.csv", [row for row in original_by if row["dataset"] != "ALL"], original_header)
        _write_csv(out_dir / "original_size_swap_overall.csv", [row for row in original_by if row["dataset"] == "ALL"], original_header)

        generation_fields = sorted(
            {
                key
                for row in generation_records
                for key, value in row.items()
                if key not in {"dataset", "stem"} and isinstance(value, (int, float))
            }
        )
        generation_dataset_records = generation_records + [
            {**row, "dataset": "ALL"} for row in generation_records
        ]
        generation_summary = _aggregate_mean(
            generation_dataset_records, ("dataset",), generation_fields
        )
        generation_header = (
            "dataset",
            *generation_fields,
            "num_samples",
            *(f"{field}_valid_count" for field in generation_fields),
        )
        _write_csv(out_dir / "generation_diagnostics_by_dataset.csv", [row for row in generation_summary if row["dataset"] != "ALL"], generation_header)
        _write_csv(out_dir / "generation_diagnostics_overall.csv", [row for row in generation_summary if row["dataset"] == "ALL"], generation_header)

        pair_by_dataset = _pairwise_summary(pair_records, ("dataset",))
        pair_subset_records = pair_records + [{**row, "dataset": "ALL"} for row in pair_records]
        pair_by_subset = _pairwise_summary(pair_subset_records, ("dataset", "touch_subset"))
        pair_summary_header = (
            "method_a",
            "method_b",
            "metric",
            "mean_delta",
            "median_delta",
            "p25_delta",
            "p75_delta",
            "win_ratio",
            "tie_ratio",
            "loss_ratio",
            "valid_count",
        )
        _write_csv(out_dir / "pairwise_by_dataset.csv", pair_by_dataset, ("dataset", *pair_summary_header))
        _write_csv(out_dir / "pairwise_by_subset.csv", pair_by_subset, ("dataset", "touch_subset", *pair_summary_header))

        if save_vis:
            log("saving_visualizations = true")
            _save_visualizations(tasks, touch_map, exposure_map, delta_map, out_dir, datasets)
            log("visualizations_complete = true")

        macro_map = {row["method"]: row for row in sections["hard"]["dataset_macro"]}
        baseline = {
            "applicable": full_run,
            "tolerance": 1e-5,
            "checks": {},
            "passed": None,
        }
        if full_run:
            all_passed = True
            for method, expected_fields in BASELINE.items():
                checks, method_passed = {}, True
                for field, expected in expected_fields.items():
                    actual = float(macro_map[method][field])
                    error = abs(actual - expected)
                    passed = error <= 1e-5
                    checks[field] = {
                        "actual": actual,
                        "expected": expected,
                        "absolute_error": error,
                        "passed": passed,
                    }
                    method_passed &= passed
                    all_passed &= passed
                baseline["checks"][method] = {
                    "passed": bool(method_passed),
                    "metrics": checks,
                }
            baseline["passed"] = bool(all_passed)

        elapsed = time.time() - started
        commit, status = _git_metadata()
        protocol = {
            "rpr_version": VERSION,
            "split": split,
            "num_samples": len(tasks),
            "full_run": full_run,
            "methods": list(METHODS),
            "hard_threshold": 0.5,
            "hard_operator": ">",
            "native_resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
            "current_dabe_resize": "68->original GT bilinear align_corners=False",
            "candidate_generation_in_eval": False,
            "gt_used_for_generation": False,
            "official_soft_per_image_minmax": True,
            "best_threshold_diagnostic_only": True,
            "dataset_specific_rule": False,
            "sample_routing_used": False,
            "output_fusion_used": False,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_dabe_manifest": str(dabe_manifest),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_cvbr_manifest": str(cvbr_manifest),
            "source_cvbr_manifest_sha256": _sha256(cvbr_manifest),
            "rpr_manifest": str(rpr_manifest),
            "rpr_manifest_sha256": _sha256(rpr_manifest),
            "rpr_cache_protocol": str(cache_protocol_path),
            "rpr_cache_protocol_sha256": _sha256(cache_protocol_path),
            "evaluator_sha256": _sha256(SCRIPT_PATH),
            "git_commit": commit,
            "git_status_short": status,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "save_vis": save_vis,
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(tasks),
            "main_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "max_worker_peak_rss_mb": peak_worker,
        }
        summary = {
            "protocol": protocol,
            "cache_protocol": cache_protocol,
            "baseline_reproduction": baseline,
            **sections,
            "touch_subset_counts": touch_counts,
            "touch_subset_hard": touch_hard,
            "query_exposure_by_dataset": query_by_dataset,
            "query_exposure_by_subset": query_by_subset,
            "r1_error_type_diagnostics_by_dataset": [row for row in error_summary if row["dataset"] != "ALL"],
            "r1_error_type_diagnostics_overall": [row for row in error_summary if row["dataset"] == "ALL"],
            "fn_tn_auc_by_dataset": [row for row in auc_rows if row["dataset"] != "ALL"],
            "fn_tn_auc_overall": [row for row in auc_rows if row["dataset"] == "ALL"],
            "fn_top10_enrichment": enrichment_rows,
            "residual_selectivity_by_dataset": [row for row in residual if row["dataset"] != "ALL"],
            "residual_selectivity_overall": [row for row in residual if row["dataset"] == "ALL"],
            "native37_swap_by_dataset": [row for row in native_by if row["dataset"] != "ALL"],
            "native37_swap_overall": [row for row in native_by if row["dataset"] == "ALL"],
            "original_size_swap_by_dataset": [row for row in original_by if row["dataset"] != "ALL"],
            "original_size_swap_overall": [row for row in original_by if row["dataset"] == "ALL"],
            "generation_diagnostics_by_dataset": [row for row in generation_summary if row["dataset"] != "ALL"],
            "generation_diagnostics_overall": [row for row in generation_summary if row["dataset"] == "ALL"],
            "pairwise_by_dataset": pair_by_dataset,
            "pairwise_by_subset": pair_by_subset,
        }
        summary["success_assessment"] = (
            _assess(summary)
            if baseline.get("passed")
            else {
                "not_evaluated": True,
                "reason": "baseline reproduction failed or sanity-only run",
            }
        )
        write_json(out_dir / "protocol.json", protocol)
        write_json(out_dir / "summary.json", summary)
        _write_docs(out_dir, summary, full_run)
        log(f"baseline_reproduction = {baseline.get('passed')}")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(tasks):.6f}")
        log(f"main_peak_rss_mb = {protocol['main_peak_rss_mb']:.3f}")
        log(f"max_worker_peak_rss_mb = {peak_worker:.3f}")
        return summary
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--cvbr_root", required=True)
    parser.add_argument("--rpr_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rpr(
        args.config,
        args.dabe_root,
        args.cvbr_root,
        args.rpr_root,
        args.out_dir,
        args.split,
        args.max_samples,
        args.threshold,
        args.save_vis,
        args.workers,
        args.torch_threads,
        args.overwrite,
        args.overwrite_reason,
    )
