#!/usr/bin/env python3
"""Formal original-size evaluation for DINOv2-B/14 R1 Direct."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import (  # noqa: E402
    _expected_feature_shape,
    _load_feature,
    _params_from_cfg,
)
from common.dabe_pseudo import (  # noqa: E402
    DABE_V2_DEFAULT_PARAMS,
    _background_anchor,
    _background_connectivity,
    _build_local_graph,
    _load_rgb_grid,
    _minmax,
    _sobel_magnitude,
)
from common.eval_dabe_background_null import (  # noqa: E402
    _git_metadata,
    _gray_panel,
    _load_gt,
    _sha256,
)
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
)
from common.utils import (  # noqa: E402
    build_image_items,
    check_exact_keys,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    split_dataset_names,
    torch_load,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
METHOD = "DINOv2-B/14-R1-Direct"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
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
RANKING_FIELDS = (
    "pixel_AP",
    "pixel_AUROC",
    "best_IoU_256",
    "best_IoU_threshold_256",
    "ranking_F_beta_max",
    "ranking_E_max",
)
ALL_METRIC_FIELDS = (*HARD_FIELDS, *OFFICIAL_FIELDS, *RANKING_FIELDS)
EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "TE-CAMO": 250,
    "TE-COD10K": 2026,
    "NC4K": 4121,
}


def _resize_r1_to_gt(value: torch.Tensor, gt_shape: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"R1 must be Tensor[1,37,37], got {getattr(value, 'shape', None)}")
    value = F.interpolate(
        value.unsqueeze(0).float(),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )
    value = F.interpolate(
        value,
        size=(int(gt_shape[0]), int(gt_shape[1])),
        mode="bilinear",
        align_corners=False,
    )
    return value.squeeze(0).clamp(0.0, 1.0).contiguous()


def _hard_mask(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return probability.float() > float(threshold)


def _pixel_auroc(probability: torch.Tensor, target: torch.Tensor) -> float:
    score = probability.detach().cpu().numpy().astype(np.float64).reshape(-1)
    positive = target.detach().cpu().numpy().reshape(-1) > 0.5
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _reconstruction_details(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor: torch.Tensor,
    params: dict,
) -> dict:
    anchor_idx = torch.where(anchor.reshape(-1).bool())[0]
    if not int(anchor_idx.numel()):
        raise RuntimeError("R1 background anchor is empty.")
    feat_anchor = feat_n.index_select(0, anchor_idx)
    rgb_anchor = rgb_n.index_select(0, anchor_idx)
    k = min(int(params["K_RECON"]), int(anchor_idx.numel()))
    raw_chunks = []
    entropy_chunks = []
    effective_chunks = []
    max_weight_chunks = []
    for start in range(0, int(feat_n.shape[0]), 512):
        end = min(start + 512, int(feat_n.shape[0]))
        feat_chunk = feat_n[start:end]
        rgb_chunk = rgb_n[start:end]
        sim_feat = feat_chunk @ feat_anchor.t()
        color_dist2 = torch.cdist(rgb_chunk, rgb_anchor, p=2.0).square()
        sim_color = torch.exp(-color_dist2 / float(params["SIGMA_COLOR_RECON"]))
        score = sim_feat + float(params["LAMBDA_COLOR_RECON"]) * sim_color
        top_value, top_index = torch.topk(score, k=k, dim=1)
        weight = torch.softmax(top_value / float(params["TAU_RECON"]), dim=1)
        feat_hat = F.normalize(
            (weight.unsqueeze(-1) * feat_anchor[top_index]).sum(dim=1), dim=1, p=2
        )
        rgb_hat = (weight.unsqueeze(-1) * rgb_anchor[top_index]).sum(dim=1)
        feature_residual = (1.0 - (feat_chunk * feat_hat).sum(dim=1)).clamp_min(0.0)
        color_residual = torch.linalg.norm(rgb_chunk - rgb_hat, dim=1)
        raw_chunks.append(feature_residual + 0.2 * color_residual)
        entropy_chunks.append(-(weight * weight.clamp_min(1e-12).log()).sum(dim=1))
        effective_chunks.append(1.0 / weight.square().sum(dim=1).clamp_min(1e-12))
        max_weight_chunks.append(weight.max(dim=1).values)
    raw = torch.cat(raw_chunks).float()
    return {
        "raw": raw,
        "normalized": _minmax(raw),
        "entropy": torch.cat(entropy_chunks).float(),
        "effective_atoms": torch.cat(effective_chunks).float(),
        "max_weight": torch.cat(max_weight_chunks).float(),
        "k": k,
    }


def _r1_diagnostics(
    feature: torch.Tensor,
    image_path: str,
    cached_r1: torch.Tensor,
    gt37: torch.Tensor,
    params: dict,
    cached_bc: torch.Tensor | None = None,
    cached_anchor: torch.Tensor | None = None,
) -> dict:
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb = _load_rgb_grid(image_path, grid)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = _sobel_magnitude(rgb).reshape(-1).float()
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, grid, params)
    if cached_bc is not None and cached_anchor is not None:
        bc = cached_bc.reshape(-1).float().clamp(0.0, 1.0)
        anchor = cached_anchor.reshape(-1).float() > 0.5
    else:
        bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, params)
        anchor = _background_anchor(bc, border, params)
    recon = _reconstruction_details(feat_n, rgb_n, anchor, params)
    normalized = recon["normalized"].reshape(1, grid, grid)
    max_abs = float((normalized - cached_r1).abs().max())

    src = torch.arange(grid * grid).view(-1, 1).expand_as(neigh_idx)
    valid = (neigh_weight > 0) & (neigh_idx > src)
    src_index = src[valid]
    dst_index = neigh_idx[valid]
    feature_distance = 1.0 - (feat_n[src_index] * feat_n[dst_index]).sum(dim=1)
    graph_weight = neigh_weight[valid]
    raw = recon["raw"]
    gt_flat = gt37.reshape(-1).bool()
    fg = raw[gt_flat]
    bg = raw[~gt_flat]
    return {
        "feature_distance": feature_distance.detach().cpu().numpy().astype(np.float32, copy=False),
        "graph_weight": graph_weight.detach().cpu().numpy().astype(np.float32, copy=False),
        "entropy": recon["entropy"].detach().cpu().numpy().astype(np.float32, copy=False),
        "effective_atoms": recon["effective_atoms"].detach().cpu().numpy().astype(np.float32, copy=False),
        "max_weight": recon["max_weight"].detach().cpu().numpy().astype(np.float32, copy=False),
        "raw_count": int(raw.numel()),
        "raw_sum": float(raw.double().sum()),
        "raw_sumsq": float(raw.double().square().sum()),
        "fg_count": int(fg.numel()),
        "fg_sum": float(fg.double().sum()),
        "bg_count": int(bg.numel()),
        "bg_sum": float(bg.double().sum()),
        "r1_recompute_max_abs": max_abs,
        "r1_recompute_tolerance_exceeded": bool(max_abs > 1e-6),
        "r1_recompute_large_exception": bool(max_abs > 1e-4),
        "anchor_count": int(anchor.sum()),
        "k": int(recon["k"]),
        "raw_map": raw.reshape(1, grid, grid),
        "bc_map": bc.reshape(1, grid, grid),
        "anchor_map": anchor.float().reshape(1, grid, grid),
    }


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _evaluate_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    cache_path = Path(task["cache_path"])
    payload = torch_load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"R1 cache payload must be dict: {cache_path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"R1 cache key mismatch: {cache_path}")
    if payload.get("backbone_key") != "dinov2-b14":
        raise RuntimeError(f"R1 cache backbone mismatch: {cache_path}")
    cached_r1 = _load_response(payload, "residual_pass1_37", (37, 37), cache_path)
    cached_bc = _load_response(payload, "bc_map_37", (37, 37), cache_path)
    cached_anchor = _load_response(payload, "bg_anchor_37", (37, 37), cache_path)
    feature = _load_feature(task["feature_row"], dataset, stem, task["cfg"])
    gt = _load_gt(task["gt_path"])
    gt37 = F.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > 0.5
    diagnostics = _r1_diagnostics(
        feature,
        task["image_path"],
        cached_r1,
        gt37,
        task["params"],
        cached_bc=cached_bc,
        cached_anchor=cached_anchor,
    )
    probability = _resize_r1_to_gt(cached_r1, tuple(gt.shape[-2:]))
    context = FastCODContext(gt)
    cod = context.evaluate_many(
        [(METHOD, "hard", probability), (METHOD, "soft", probability)],
        task["threshold"],
    )
    hard = cod[(METHOD, "hard")]
    soft = cod[(METHOD, "soft")]
    ranking = _ranking_metrics(probability, gt)
    row = {
        "dataset": dataset,
        "stem": stem,
        "method": METHOD,
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
        **ranking,
        "pixel_AUROC": _pixel_auroc(probability, gt),
        "ranking_F_beta_max": soft["F_beta_max"],
        "ranking_E_max": soft["E_max"],
        "feature_distance_mean": float(np.mean(diagnostics["feature_distance"])),
        "graph_weight_mean": float(np.mean(diagnostics["graph_weight"])),
        "reconstruction_entropy_mean": float(np.mean(diagnostics["entropy"])),
        "effective_atoms_mean": float(np.mean(diagnostics["effective_atoms"])),
        "maximum_atom_weight_mean": float(np.mean(diagnostics["max_weight"])),
        "raw_residual_mean": diagnostics["raw_sum"] / diagnostics["raw_count"],
        "raw_foreground_mean": diagnostics["fg_sum"] / max(1, diagnostics["fg_count"]),
        "raw_background_mean": diagnostics["bg_sum"] / max(1, diagnostics["bg_count"]),
        "r1_recompute_max_abs": diagnostics["r1_recompute_max_abs"],
        "r1_recompute_tolerance_exceeded": diagnostics["r1_recompute_tolerance_exceeded"],
        "r1_recompute_large_exception": diagnostics["r1_recompute_large_exception"],
        "anchor_count": diagnostics["anchor_count"],
        "source_cache_path": str(cache_path),
        "feature_cache_path": str(task["feature_row"]["cache_path"]),
        "_official_f_curve": soft["f_curve"],
        "_official_e_curve": soft["e_curve"],
    }
    for field in ALL_METRIC_FIELDS:
        value = float(row[field])
        if field in {"pixel_AP", "pixel_AUROC"} and math.isnan(value):
            continue
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite metric: {dataset}/{stem}/{field}")
    for key in ("raw_map", "bc_map", "anchor_map"):
        diagnostics.pop(key)
    return {
        "dataset": dataset,
        "stem": stem,
        "row": row,
        "diagnostics": diagnostics,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


class _MetricAggregate:
    def __init__(self):
        self.count = 0
        self.sums = defaultdict(float)
        self.valid = defaultdict(int)
        self.f_curve_sum = np.zeros(256, dtype=np.float64)
        self.e_curve_sum = np.zeros(256, dtype=np.float64)

    def add(self, row: dict):
        self.count += 1
        for field in ALL_METRIC_FIELDS:
            value = float(row[field])
            if math.isfinite(value):
                self.sums[field] += value
                self.valid[field] += 1
        self.f_curve_sum += row["_official_f_curve"]
        self.e_curve_sum += row["_official_e_curve"]

    def result(self) -> dict:
        result = {
            field: self.sums[field] / self.valid[field]
            if self.valid[field]
            else float("nan")
            for field in ALL_METRIC_FIELDS
        }
        result["official_soft_F_beta_max"] = float(np.max(self.f_curve_sum / self.count))
        result["official_soft_E_max"] = float(np.max(self.e_curve_sum / self.count))
        result["ranking_F_beta_max"] = result["official_soft_F_beta_max"]
        result["ranking_E_max"] = result["official_soft_E_max"]
        result["num_samples"] = self.count
        result["ap_valid_count"] = self.valid["pixel_AP"]
        result["auroc_valid_count"] = self.valid["pixel_AUROC"]
        return result


def _metric_tables(rows: list[dict], datasets: tuple[str, ...], fields: tuple[str, ...]):
    by_dataset = []
    for dataset in datasets:
        aggregate = _MetricAggregate()
        for row in rows:
            if row["dataset"] == dataset:
                aggregate.add(row)
        values = aggregate.result()
        by_dataset.append(
            {
                "scope": "by_dataset",
                "dataset": dataset,
                "method": METHOD,
                **{field: values[field] for field in fields},
                "num_samples": values["num_samples"],
                "ap_valid_count": values["ap_valid_count"],
                "auroc_valid_count": values["auroc_valid_count"],
            }
        )
    macro = {
        "scope": "dataset_macro",
        "dataset": "ALL",
        "method": METHOD,
        **{field: float(np.mean([row[field] for row in by_dataset])) for field in fields},
        "num_samples": sum(int(row["num_samples"]) for row in by_dataset),
        "ap_valid_count": sum(int(row["ap_valid_count"]) for row in by_dataset),
        "auroc_valid_count": sum(int(row["auroc_valid_count"]) for row in by_dataset),
    }
    return by_dataset, [macro]


def _array_stats(array: np.ndarray) -> dict:
    value = np.asarray(array, dtype=np.float32).reshape(-1)
    return {
        "count": int(value.size),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "p10": float(np.percentile(value, 10)),
        "p50": float(np.percentile(value, 50)),
        "p90": float(np.percentile(value, 90)),
    }


def _diagnostic_tables(diagnostic_records: list[dict], datasets: tuple[str, ...]):
    feature_rows, graph_rows, reconstruction_rows, raw_rows = [], [], [], []
    for dataset in (*datasets, "ALL"):
        selected = [
            record
            for record in diagnostic_records
            if dataset == "ALL" or record["dataset"] == dataset
        ]
        distances = np.concatenate([record["feature_distance"] for record in selected])
        weights = np.concatenate([record["graph_weight"] for record in selected])
        entropy = np.concatenate([record["entropy"] for record in selected])
        effective = np.concatenate([record["effective_atoms"] for record in selected])
        maximum = np.concatenate([record["max_weight"] for record in selected])
        feature_rows.append({"dataset": dataset, **{f"distance_{k}": v for k, v in _array_stats(distances).items()}})
        graph_rows.append(
            {
                "dataset": dataset,
                **{f"weight_{k}": v for k, v in _array_stats(weights).items()},
                "ratio_weight_lt_001": float(np.mean(weights < 0.01)),
                "ratio_weight_gt_09": float(np.mean(weights > 0.9)),
            }
        )
        reconstruction_rows.append(
            {
                "dataset": dataset,
                **{f"entropy_{k}": v for k, v in _array_stats(entropy).items()},
                **{f"effective_atoms_{k}": v for k, v in _array_stats(effective).items()},
                **{f"maximum_weight_{k}": v for k, v in _array_stats(maximum).items()},
                "ratio_effective_atoms_lt_2": float(np.mean(effective < 2.0)),
                "ratio_maximum_weight_gt_08": float(np.mean(maximum > 0.8)),
                "k_recon": int(selected[0]["k"]),
            }
        )
        raw_count = sum(int(record["raw_count"]) for record in selected)
        raw_sum = sum(float(record["raw_sum"]) for record in selected)
        raw_sumsq = sum(float(record["raw_sumsq"]) for record in selected)
        fg_count = sum(int(record["fg_count"]) for record in selected)
        fg_sum = sum(float(record["fg_sum"]) for record in selected)
        bg_count = sum(int(record["bg_count"]) for record in selected)
        bg_sum = sum(float(record["bg_sum"]) for record in selected)
        raw_mean = raw_sum / raw_count
        raw_std = math.sqrt(max(0.0, raw_sumsq / raw_count - raw_mean * raw_mean))
        fg_mean = fg_sum / fg_count
        bg_mean = bg_sum / bg_count
        raw_rows.append(
            {
                "dataset": dataset,
                "raw_residual_count": raw_count,
                "raw_residual_mean": raw_mean,
                "raw_residual_std": raw_std,
                "foreground_raw_count": fg_count,
                "foreground_raw_mean": fg_mean,
                "background_raw_count": bg_count,
                "background_raw_mean": bg_mean,
                "foreground_background_raw_margin": fg_mean - bg_mean,
                "r1_recompute_max_abs": max(float(record["r1_recompute_max_abs"]) for record in selected),
                "r1_recompute_tolerance_exception_count": sum(
                    bool(record["r1_recompute_tolerance_exceeded"]) for record in selected
                ),
                "r1_recompute_large_exception_count": sum(
                    bool(record["r1_recompute_large_exception"]) for record in selected
                ),
                "anchor_count_mean": float(np.mean([record["anchor_count"] for record in selected])),
            }
        )
    return feature_rows, graph_rows, reconstruction_rows, raw_rows


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    columns = ("dataset", *fields)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = [str(row["dataset"])] + [f"{float(row[field]):.6f}" for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _parameter_judgement(graph_all: dict, recon_all: dict) -> dict:
    graph_collapsed = (
        graph_all["ratio_weight_lt_001"] >= 0.95
        or graph_all["ratio_weight_gt_09"] >= 0.95
    )
    recon_nearest = (
        recon_all["ratio_effective_atoms_lt_2"] >= 0.50
        or recon_all["ratio_maximum_weight_gt_08"] >= 0.50
    )
    recon_uniform = recon_all["effective_atoms_mean"] >= 0.90 * recon_all["k_recon"]
    if graph_collapsed:
        case = "B"
        conclusion = "图权重明显接近全 0 或全 1，可能需要单独调整 SIGMA_F。"
    elif recon_nearest or recon_uniform:
        case = "C"
        conclusion = "重构权重接近单最近邻或均匀平均，可能需要单独调整 TAU_RECON。"
    else:
        case = "A"
        conclusion = "图权重和重构权重均正常，当前 R1 参数可直接用于 DINOv2-B/14。"
    return {
        "case": case,
        "conclusion": conclusion,
        "adjust_sigma_f": bool(graph_collapsed),
        "adjust_tau_recon": bool((not graph_collapsed) and (recon_nearest or recon_uniform)),
        "graph_collapsed": bool(graph_collapsed),
        "reconstruction_nearest_neighbor_collapse": bool(recon_nearest),
        "reconstruction_uniform_collapse": bool(recon_uniform),
    }


def _save_visualization(task: dict, out_path: Path):
    payload = torch_load(task["cache_path"], map_location="cpu")
    cache_path = Path(task["cache_path"])
    cached_r1 = _load_response(payload, "residual_pass1_37", (37, 37), cache_path)
    cached_bc = _load_response(payload, "bc_map_37", (37, 37), cache_path)
    cached_anchor = _load_response(payload, "bg_anchor_37", (37, 37), cache_path)
    feature = _load_feature(task["feature_row"], task["dataset"], task["stem"], task["cfg"])
    gt = _load_gt(task["gt_path"])
    gt37 = F.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > 0.5
    diagnostics = _r1_diagnostics(
        feature,
        task["image_path"],
        cached_r1,
        gt37,
        task["params"],
        cached_bc=cached_bc,
        cached_anchor=cached_anchor,
    )
    hard = _hard_mask(cached_r1, task["threshold"]).float()
    size, label_height = 180, 24
    with Image.open(task["image_path"]) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    panels = [
        ("RGB", rgb),
        ("GT", _gray_panel(gt, size, True)),
        ("DINOv2 R1", _gray_panel(cached_r1, size)),
        ("DINOv2 BC", _gray_panel(diagnostics["bc_map"], size)),
        ("Background anchor", _gray_panel(diagnostics["anchor_map"], size, True)),
        ("Raw residual", _gray_panel(_minmax(diagnostics["raw_map"]), size)),
        ("R1 hard >0.5", _gray_panel(hard, size, True)),
    ]
    canvas = Image.new("RGB", (len(panels) * size, size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = index * size
        canvas.paste(panel, (x, label_height))
        draw.text((x + 4, 5), title, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_documents(
    out_dir: Path,
    hard_by: list[dict],
    hard_macro: list[dict],
    official_by: list[dict],
    official_macro: list[dict],
    ranking_by: list[dict],
    ranking_macro: list[dict],
    feature_rows: list[dict],
    graph_rows: list[dict],
    reconstruction_rows: list[dict],
    raw_rows: list[dict],
    judgement: dict,
):
    feature_all = next(row for row in feature_rows if row["dataset"] == "ALL")
    graph_all = next(row for row in graph_rows if row["dataset"] == "ALL")
    recon_all = next(row for row in reconstruction_rows if row["dataset"] == "ALL")
    raw_all = next(row for row in raw_rows if row["dataset"] == "ALL")
    results = f"""# DINOv2-B/14 R1 Direct 正式结果

本实验直接复用已有 DINOv2-B/14 Last-Layer Attention Key 缓存；未运行 DINO，未调整 R1 参数，未重新评测 DINOv1。

## 四数据集 Hard

{_markdown_table(hard_by, ("hard_S_m", "hard_F_beta_w", "hard_F_beta_mean", "hard_E_mean", "hard_MAE", "hard_IoU", "hard_Precision", "hard_Recall", "hard_Area"))}

## Dataset Macro Hard

{_markdown_table(hard_macro, ("hard_S_m", "hard_F_beta_w", "hard_F_beta_mean", "hard_E_mean", "hard_MAE", "hard_IoU", "hard_Precision", "hard_Recall", "hard_Area"))}

## Official Soft

{_markdown_table(official_by + official_macro, ("official_soft_S_m", "official_soft_F_beta_w", "official_soft_E_mean", "official_soft_MAE"))}

当前 CODMetrics 对非恒定 continuous prediction 会逐图 Min-Max。

## Ranking

{_markdown_table(ranking_by + ranking_macro, ("pixel_AP", "pixel_AUROC", "best_IoU_256", "best_IoU_threshold_256"))}

## 固定参数诊断（所有数据 pooled）

- 局部特征距离 mean/std/p10/p50/p90：`{feature_all['distance_mean']:.6f}/{feature_all['distance_std']:.6f}/{feature_all['distance_p10']:.6f}/{feature_all['distance_p50']:.6f}/{feature_all['distance_p90']:.6f}`
- 图权重 mean/p10/p50/p90：`{graph_all['weight_mean']:.6f}/{graph_all['weight_p10']:.6f}/{graph_all['weight_p50']:.6f}/{graph_all['weight_p90']:.6f}`
- 图权重 `<0.01`/`>0.9` 比例：`{graph_all['ratio_weight_lt_001']:.6f}/{graph_all['ratio_weight_gt_09']:.6f}`
- 重构有效原子数 mean/p10/p50/p90：`{recon_all['effective_atoms_mean']:.6f}/{recon_all['effective_atoms_p10']:.6f}/{recon_all['effective_atoms_p50']:.6f}/{recon_all['effective_atoms_p90']:.6f}`
- 最大重构权重 mean/p10/p50/p90：`{recon_all['maximum_weight_mean']:.6f}/{recon_all['maximum_weight_p10']:.6f}/{recon_all['maximum_weight_p50']:.6f}/{recon_all['maximum_weight_p90']:.6f}`
- `N_eff<2`/`max_weight>0.8` 比例：`{recon_all['ratio_effective_atoms_lt_2']:.6f}/{recon_all['ratio_maximum_weight_gt_08']:.6f}`
- Raw residual mean/std：`{raw_all['raw_residual_mean']:.6f}/{raw_all['raw_residual_std']:.6f}`
- Raw foreground/background/margin：`{raw_all['foreground_raw_mean']:.6f}/{raw_all['background_raw_mean']:.6f}/{raw_all['foreground_background_raw_margin']:.6f}`
- R1 重算最大误差：`{raw_all['r1_recompute_max_abs']:.12g}`
- R1 重算 `1e-6` 诊断容差例外样本数：`{raw_all['r1_recompute_tolerance_exception_count']}`；仅记录放行。
- R1 重算大于 `1e-4` 的非致命例外样本数：`{raw_all['r1_recompute_large_exception_count']}`；正式指标始终读取缓存 R1。

## 参数判断

- 情况：`{judgement['case']}`
- 是否需要调整 SIGMA_F：`{judgement['adjust_sigma_f']}`
- 是否需要调整 TAU_RECON：`{judgement['adjust_tau_recon']}`
- 结论：{judgement['conclusion']}
"""
    readme = """# DINOv2-B/14 R1 Direct Evaluation

- 特征：已有 DINOv2-B/14 最后一层 Attention Key `[768,37,37]`。
- 生成：identity 单视图；无 DINO forward、无训练、无参数搜索。
- Hard：`37→68 bilinear→原始 GT 尺寸 bilinear→strict >0.5`，两次 `align_corners=False`。
- Official Soft：连续响应进入当前 CODMetrics；非恒定预测会逐图 Min-Max。
- Best IoU threshold 只用于诊断，不用于主结果。
- 诊断统计使用所有有效无向 8 邻域边；图实现中的双向重复边具有相同值，因此不改变分布统计。
"""
    (out_dir / "RESULTS.md").write_text(results, encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def evaluate(args):
    started = time.time()
    cfg = load_config(args.config)
    if cfg.BACKBONE_KEY != "dinov2-b14":
        raise RuntimeError(f"Expected BACKBONE_KEY=dinov2-b14, got {cfg.BACKBONE_KEY}")
    expected_shape = _expected_feature_shape(cfg)
    if expected_shape != [768, 37, 37]:
        raise RuntimeError(f"Expected DINOv2 feature [768,37,37], got {expected_shape}")
    params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg), "VERSION": "v2"}
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Evaluation output already exists and is non-empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_manifest = feature_manifest_path(cfg, args.split).resolve()
    feature_rows = read_jsonl(feature_manifest)
    feature_map = manifest_to_map(feature_rows, feature_manifest)
    dabe_root = Path(args.dabe_root).expanduser().resolve()
    dabe_manifest = dabe_root / f"manifest_{args.split}.jsonl"
    dabe_rows = read_jsonl(dabe_manifest)
    dabe_map = manifest_to_map(dabe_rows, dabe_manifest)
    all_items = build_image_items(
        cfg.DATA_ROOT, split_dataset_names(cfg, args.split), require_gt=True
    )
    all_item_map = {(item["dataset"], item["stem"]): item for item in all_items}
    check_exact_keys("DINOv2 feature manifest", feature_map, all_item_map)
    items = all_items
    if args.max_samples >= 0:
        items = items[: args.max_samples]
    selected_item_map = {(item["dataset"], item["stem"]): item for item in items}
    check_exact_keys("DINOv2 R1 manifest", dabe_map, selected_item_map)
    if not items:
        raise RuntimeError("No evaluation samples selected.")
    active_datasets = tuple(
        dataset for dataset in DATASETS if any(item["dataset"] == dataset for item in items)
    )
    if args.max_samples < 0:
        if len(items) != 6473:
            raise RuntimeError(f"Full evaluation requires 6473 samples, got {len(items)}")
        counts = {dataset: sum(item["dataset"] == dataset for item in items) for dataset in DATASETS}
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(f"Dataset counts mismatch: {counts}")

    feature_cfg = SimpleNamespace(DINO=dict(cfg.DINO))
    tasks = []
    for item in items:
        key = (item["dataset"], item["stem"])
        tasks.append(
            {
                **item,
                "cache_path": dabe_map[key]["cache_path"],
                "feature_row": feature_map[key],
                "cfg": feature_cfg,
                "params": params,
                "threshold": args.threshold,
            }
        )
    print(f"feature_manifest = {feature_manifest}", flush=True)
    print(f"dabe_manifest = {dabe_manifest}", flush=True)
    print(f"expected_feature_shape = {expected_shape}", flush=True)
    print(f"num_samples = {len(tasks)}", flush=True)
    print("dino_forward_used = false", flush=True)
    print("training_used = false", flush=True)

    metric_rows = []
    diagnostic_records = []
    max_worker_rss = 0.0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_evaluate_one, tasks), 1):
            metric_rows.append(result["row"])
            diagnostic_records.append({"dataset": result["dataset"], **result["diagnostics"]})
            max_worker_rss = max(max_worker_rss, float(result["worker_peak_rss_mb"]))
            if index % 100 == 0 or index == len(tasks):
                print(f"processed = {index}/{len(tasks)}", flush=True)

    hard_by, hard_macro = _metric_tables(metric_rows, active_datasets, HARD_FIELDS)
    official_by, official_macro = _metric_tables(metric_rows, active_datasets, OFFICIAL_FIELDS)
    ranking_by, ranking_macro = _metric_tables(metric_rows, active_datasets, RANKING_FIELDS)
    feature_diag, graph_diag, recon_diag, raw_diag = _diagnostic_tables(
        diagnostic_records, active_datasets
    )
    judgement = _parameter_judgement(
        next(row for row in graph_diag if row["dataset"] == "ALL"),
        next(row for row in recon_diag if row["dataset"] == "ALL"),
    )

    common_metric_fields = ["scope", "dataset", "method"]
    tail_fields = ["num_samples", "ap_valid_count", "auroc_valid_count"]
    _write_csv(out_dir / "hard_by_dataset.csv", hard_by, [*common_metric_fields, *HARD_FIELDS, *tail_fields])
    _write_csv(out_dir / "hard_dataset_macro.csv", hard_macro, [*common_metric_fields, *HARD_FIELDS, *tail_fields])
    _write_csv(out_dir / "official_soft_by_dataset.csv", official_by, [*common_metric_fields, *OFFICIAL_FIELDS, *tail_fields])
    _write_csv(out_dir / "official_soft_dataset_macro.csv", official_macro, [*common_metric_fields, *OFFICIAL_FIELDS, *tail_fields])
    _write_csv(out_dir / "ranking_by_dataset.csv", ranking_by, [*common_metric_fields, *RANKING_FIELDS, *tail_fields])
    _write_csv(out_dir / "ranking_dataset_macro.csv", ranking_macro, [*common_metric_fields, *RANKING_FIELDS, *tail_fields])
    _write_csv(out_dir / "feature_diagnostics.csv", feature_diag, list(feature_diag[0]))
    _write_csv(out_dir / "graph_diagnostics.csv", graph_diag, list(graph_diag[0]))
    _write_csv(out_dir / "reconstruction_diagnostics.csv", recon_diag, list(recon_diag[0]))
    _write_csv(out_dir / "raw_residual_diagnostics.csv", raw_diag, list(raw_diag[0]))
    per_sample_fields = [
        "dataset", "stem", "method", *ALL_METRIC_FIELDS,
        "feature_distance_mean", "graph_weight_mean", "reconstruction_entropy_mean",
        "effective_atoms_mean", "maximum_atom_weight_mean", "raw_residual_mean",
        "raw_foreground_mean", "raw_background_mean", "r1_recompute_max_abs",
        "r1_recompute_tolerance_exceeded", "r1_recompute_large_exception",
        "anchor_count", "source_cache_path", "feature_cache_path",
    ]
    _write_csv(out_dir / "per_sample.csv", metric_rows, per_sample_fields)

    if args.save_vis:
        by_key = {(row["dataset"], row["stem"]): row for row in metric_rows}
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task["dataset"]].append(task)
        for dataset in active_datasets:
            dataset_tasks = grouped[dataset]
            ranked = sorted(
                dataset_tasks,
                key=lambda task: by_key[(dataset, task["stem"])]["hard_F_beta_w"],
                reverse=True,
            )
            for folder, selected in (
                ("fixed_first8", dataset_tasks[:8]),
                ("best8", ranked[:8]),
                ("worst8", ranked[-8:]),
            ):
                for task in selected:
                    _save_visualization(
                        task, out_dir / "vis" / folder / dataset / f"{task['stem']}.png"
                    )
        print("visualizations_complete = true", flush=True)

    elapsed = time.time() - started
    commit, status = _git_metadata()
    protocol = {
        "experiment": "dinov2_b14_dabe_r1_direct",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": _sha256(Path(args.config).resolve()),
        "evaluator_sha256": _sha256(SCRIPT_PATH),
        "feature_manifest": str(feature_manifest),
        "feature_manifest_sha256": _sha256(feature_manifest),
        "dabe_manifest": str(dabe_manifest),
        "dabe_manifest_sha256": _sha256(dabe_manifest),
        "backbone": "DINOv2-B/14",
        "model_name": cfg.DINO["model_name"],
        "feature_type": "last-layer attention key",
        "feature_input_size": int(cfg.DINO["feature_input_size"]),
        "patch_size": int(cfg.DINO["patch_size"]),
        "feature_shape": expected_shape,
        "feature_finite": True,
        "source_augs": ["identity"],
        "split": args.split,
        "num_samples": len(tasks),
        "hard_threshold": float(args.threshold),
        "hard_operator": "strict >",
        "resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
        "dino_forward_used": False,
        "feature_cache_regenerated": False,
        "training_used": False,
        "parameter_search_used": False,
        "gt_used_for_generation": False,
        "r1_params": {key: params[key] for key in (
            "GRID", "LOSS_SIZE", "SIGMA_F", "SIGMA_C", "SIGMA_E", "BORDER_WIDTH",
            "TAU_BC", "BG_ANCHOR_TOP_PERCENT", "BG_ANCHOR_MIN_RATIO",
            "BG_ANCHOR_FALLBACK_TOP_PERCENT", "K_RECON", "LAMBDA_COLOR_RECON",
            "SIGMA_COLOR_RECON", "TAU_RECON",
        )},
        "parameter_judgement": judgement,
        "parameter_judgement_rules": {
            "graph_collapsed": "ratio(weight<0.01)>=0.95 or ratio(weight>0.9)>=0.95",
            "nearest_neighbor_collapse": "ratio(effective_atoms<2)>=0.50 or ratio(max_weight>0.8)>=0.50",
            "uniform_collapse": "mean(effective_atoms)>=0.90*K_RECON",
        },
        "r1_recompute_max_abs": next(row for row in raw_diag if row["dataset"] == "ALL")["r1_recompute_max_abs"],
        "r1_recompute_reference_tolerance": 1e-6,
        "r1_recompute_large_exception_threshold": 1e-4,
        "rare_numerical_exceptions_allowed": True,
        "r1_recompute_tolerance_exception_count": sum(
            float(row["r1_recompute_max_abs"]) > 1e-6 for row in metric_rows
        ),
        "r1_recompute_large_exception_count": sum(
            float(row["r1_recompute_max_abs"]) > 1e-4 for row in metric_rows
        ),
        "r1_recompute_tolerance_exception_samples": [
            {"dataset": row["dataset"], "stem": row["stem"], "max_abs": row["r1_recompute_max_abs"]}
            for row in metric_rows
            if float(row["r1_recompute_max_abs"]) > 1e-6
        ],
        "workers": int(args.workers),
        "torch_threads_per_worker": int(args.torch_threads),
        "elapsed_seconds": elapsed,
        "average_seconds_per_image": elapsed / len(tasks),
        "main_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "max_worker_peak_rss_mb": max_worker_rss,
        "git_commit": commit,
        "git_status_short": status,
    }
    summary = {
        "hard": {"by_dataset": hard_by, "dataset_macro": hard_macro},
        "official_soft": {"by_dataset": official_by, "dataset_macro": official_macro},
        "ranking": {"by_dataset": ranking_by, "dataset_macro": ranking_macro},
        "feature_diagnostics": feature_diag,
        "graph_diagnostics": graph_diag,
        "reconstruction_diagnostics": recon_diag,
        "raw_residual_diagnostics": raw_diag,
        "parameter_judgement": judgement,
        "protocol": protocol,
    }
    write_json(out_dir / "protocol.json", protocol)
    write_json(out_dir / "summary.json", summary)
    _write_documents(
        out_dir, hard_by, hard_macro, official_by, official_macro,
        ranking_by, ranking_macro, feature_diag, graph_diag, recon_diag, raw_diag,
        judgement,
    )
    print(f"elapsed_seconds = {elapsed:.3f}", flush=True)
    print(f"average_seconds_per_image = {elapsed / len(tasks):.6f}", flush=True)
    print(f"parameter_case = {judgement['case']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=1)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be in (0,1)")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("--workers and --torch_threads must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
