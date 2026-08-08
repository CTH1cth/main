#!/usr/bin/env python3
"""Controlled DINOv2 Last-Key versus Final-Patch-Token R1 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _load_feature, _params_from_cfg  # noqa: E402
from common.cache_features import call_dino, load_dino, preprocess_image  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS, generate_dabe_pseudo  # noqa: E402
from common.eval_dabe_background_null import _load_gt, _sha256  # noqa: E402
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _load_response,
    _ranking_metrics,
)
from common.eval_dinov2_r1 import _pixel_auroc, _resize_r1_to_gt  # noqa: E402
from common.utils import (  # noqa: E402
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
METHODS = ("Last-Layer Key", "Final Patch Token")
METRIC_FIELDS = (
    "hard_S_m",
    "hard_F_beta_w",
    "hard_F_beta_mean",
    "hard_E_mean",
    "hard_MAE",
    "hard_IoU",
    "hard_Precision",
    "hard_Recall",
    "hard_Area",
    "official_soft_S_m",
    "official_soft_F_beta_w",
    "official_soft_E_mean",
    "official_soft_MAE",
    "pixel_AP",
    "pixel_AUROC",
    "best_IoU_256",
    "best_IoU_threshold_256",
)


def final_hidden_to_feature(last_hidden_state: torch.Tensor, cfg) -> torch.Tensor:
    if not torch.is_tensor(last_hidden_state) or last_hidden_state.ndim != 3:
        raise TypeError("last_hidden_state must be a rank-3 torch.Tensor.")
    channels = int(cfg.DINO["embed_dim"])
    input_size = int(cfg.DINO["feature_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    if input_size % patch_size:
        raise RuntimeError("feature_input_size must be divisible by patch_size.")
    grid = input_size // patch_size
    expected = (1, 1 + grid * grid, channels)
    if tuple(last_hidden_state.shape) != expected:
        raise RuntimeError(
            f"Expected final hidden state {list(expected)}, got "
            f"{list(last_hidden_state.shape)}."
        )
    patch = last_hidden_state[:, 1:, :].reshape(1, grid, grid, channels)
    feature = patch.permute(0, 3, 1, 2).squeeze(0).detach().cpu().float().contiguous()
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("Final patch token contains NaN/Inf.")
    return feature


def select_uniform_per_dataset(rows: list[dict], per_dataset: int, max_samples: int) -> list[dict]:
    grouped = {dataset: [] for dataset in DATASETS}
    for row in rows:
        dataset = row.get("dataset")
        if dataset in grouped:
            grouped[dataset].append(row)
    selected_by_dataset = {}
    for dataset in DATASETS:
        values = grouped[dataset]
        if len(values) < int(per_dataset):
            raise RuntimeError(
                f"Dataset {dataset} has {len(values)} rows, fewer than {per_dataset}."
            )
        indices = np.linspace(0, len(values) - 1, int(per_dataset), dtype=int)
        if len(set(indices.tolist())) != int(per_dataset):
            raise RuntimeError(f"Uniform selection produced duplicate indices for {dataset}.")
        selected_by_dataset[dataset] = [values[int(index)] for index in indices]

    selected = []
    for position in range(int(per_dataset)):
        for dataset in DATASETS:
            selected.append(selected_by_dataset[dataset][position])
    if int(max_samples) >= 0:
        selected = selected[: int(max_samples)]
    return selected


def select_all_samples(rows: list[dict], datasets: tuple[str, ...] = DATASETS) -> list[dict]:
    selected = [row for row in rows if row.get("dataset") in datasets]
    keys = [(row.get("dataset"), row.get("stem")) for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Full feature manifest contains duplicate dataset/stem keys.")
    counts = {dataset: sum(row["dataset"] == dataset for row in selected) for dataset in datasets}
    official = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
    expected = {dataset: official[dataset] for dataset in datasets}
    if counts != expected:
        raise RuntimeError(f"Unexpected full-test dataset counts: {counts}; expected {expected}.")
    return selected


def _local_distances(feature: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    value = F.normalize(feature.float(), dim=0, p=2)
    height, width = value.shape[-2:]
    all_edges = []
    top_left_edges = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        src = value[:, y0:y1, x0:x1]
        dst = value[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        distance = 1.0 - (src * dst).sum(dim=0)
        all_edges.append(distance.reshape(-1))
        ys = torch.arange(y0, y1).view(-1, 1).expand(y1 - y0, x1 - x0)
        xs = torch.arange(x0, x1).view(1, -1).expand(y1 - y0, x1 - x0)
        affected = ((ys < 2) & (xs < 2)) | (((ys + dy) < 2) & ((xs + dx) < 2))
        top_left_edges.append(distance[affected])
    return (
        torch.cat(all_edges).detach().cpu().numpy().astype(np.float32, copy=False),
        torch.cat(top_left_edges).detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _feature_record(dataset: str, stem: str, method: str, feature: torch.Tensor) -> dict:
    norm = torch.linalg.vector_norm(feature.float(), dim=0)
    median = float(norm.median())
    argmax = int(torch.argmax(norm))
    y, x = divmod(argmax, int(norm.shape[1]))
    local, top_left = _local_distances(feature)
    return {
        "dataset": dataset,
        "stem": stem,
        "method": method,
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std(unbiased=False)),
        "norm_cv": float(norm.std(unbiased=False) / norm.mean().clamp_min(1e-12)),
        "patch00_norm_ratio_to_median": float(norm[0, 0]) / (median + 1e-12),
        "max_norm_ratio_to_median": float(norm.max()) / (median + 1e-12),
        "norm_argmax_y": y,
        "norm_argmax_x": x,
        "norm_argmax_is_00": bool(y == 0 and x == 0),
        "norm_argmax_in_top_left_2x2": bool(y < 2 and x < 2),
        "local_distance": local,
        "top_left_incident_distance": top_left,
    }


def _evaluate_pair(key_probability: torch.Tensor, final_probability: torch.Tensor, gt: torch.Tensor):
    context = FastCODContext(gt)
    predictions = []
    for method, probability in zip(METHODS, (key_probability, final_probability)):
        predictions.extend(((method, "hard", probability), (method, "soft", probability)))
    evaluated = context.evaluate_many(predictions, 0.5)
    output = {}
    for method, probability in zip(METHODS, (key_probability, final_probability)):
        hard = evaluated[(method, "hard")]
        soft = evaluated[(method, "soft")]
        ranking = _ranking_metrics(probability, gt)
        output[method] = {
            "hard_S_m": hard["S_m"],
            "hard_F_beta_w": hard["F_beta_w"],
            "hard_F_beta_mean": hard["F_beta_mean"],
            "hard_E_mean": hard["E_mean"],
            "hard_MAE": hard["MAE"],
            "hard_IoU": hard["IoU"],
            "hard_Precision": hard["Precision"],
            "hard_Recall": hard["Recall"],
            "hard_Area": hard["Area"],
            "official_soft_S_m": soft["S_m"],
            "official_soft_F_beta_w": soft["F_beta_w"],
            "official_soft_E_mean": soft["E_mean"],
            "official_soft_MAE": soft["MAE"],
            **ranking,
            "pixel_AUROC": _pixel_auroc(probability, gt),
        }
    return output


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _aggregate_metrics(
    rows: list[dict], datasets: tuple[str, ...] = DATASETS
) -> tuple[list[dict], list[dict]]:
    by_dataset = []
    for method in METHODS:
        for dataset in datasets:
            selected = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
            if not selected:
                continue
            by_dataset.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "num_samples": len(selected),
                    **{field: _mean(row[field] for row in selected) for field in METRIC_FIELDS},
                }
            )
    macro = []
    for method in METHODS:
        selected = [row for row in by_dataset if row["method"] == method]
        macro.append(
            {
                "dataset": "ALL",
                "method": method,
                "num_samples": sum(int(row["num_samples"]) for row in selected),
                **{field: _mean(row[field] for row in selected) for field in METRIC_FIELDS},
            }
        )
    return by_dataset, macro


def _aggregate_features(records: list[dict], datasets: tuple[str, ...] = DATASETS) -> list[dict]:
    rows = []
    for method in METHODS:
        for dataset in (*datasets, "ALL"):
            selected = [
                row for row in records
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            if not selected:
                continue
            local = np.concatenate([row["local_distance"] for row in selected])
            top_left = np.concatenate([row["top_left_incident_distance"] for row in selected])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "num_samples": len(selected),
                    "norm_mean": _mean(row["norm_mean"] for row in selected),
                    "norm_cv_mean": _mean(row["norm_cv"] for row in selected),
                    "patch00_norm_ratio_mean": _mean(
                        row["patch00_norm_ratio_to_median"] for row in selected
                    ),
                    "max_norm_ratio_mean": _mean(
                        row["max_norm_ratio_to_median"] for row in selected
                    ),
                    "norm_argmax_00_rate": _mean(row["norm_argmax_is_00"] for row in selected),
                    "norm_argmax_top_left_2x2_rate": _mean(
                        row["norm_argmax_in_top_left_2x2"] for row in selected
                    ),
                    "local_distance_mean": float(local.mean()),
                    "local_distance_std": float(local.std()),
                    "local_distance_p10": float(np.percentile(local, 10)),
                    "local_distance_p50": float(np.percentile(local, 50)),
                    "local_distance_p90": float(np.percentile(local, 90)),
                    "top_left_incident_distance_mean": float(top_left.mean()),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"No rows for {path}.")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and not key.startswith("_"):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _panel_rgb(image: Image.Image, size=(220, 220)) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def _panel_map(value: torch.Tensor, size=(220, 220)) -> Image.Image:
    array = value.detach().cpu().float().squeeze().clamp(0, 1).numpy()
    image = Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="L").convert("RGB")
    return image.resize(size, Image.Resampling.BILINEAR)


def _save_visualization(record: dict, path: Path):
    image = Image.open(record["image_path"]).convert("RGB")
    gt = Image.open(record["gt_path"]).convert("L")
    key = record["key_r1"]
    final = record["final_r1"]
    panels = (
        ("RGB", _panel_rgb(image)),
        ("GT", _panel_rgb(gt.convert("RGB"))),
        ("Key R1", _panel_map(key)),
        ("Final Token R1", _panel_map(final)),
        ("Key Hard", _panel_map((key > 0.5).float())),
        ("Final Hard", _panel_map((final > 0.5).float())),
        ("Abs Diff", _panel_map((final - key).abs())),
    )
    width, height = panels[0][1].size
    title_height = 28
    canvas = Image.new("RGB", (width * len(panels), height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        canvas.paste(panel, (index * width, title_height))
        draw.text((index * width + 6, 7), title, fill="black")
    ensure_dir(path.parent)
    canvas.save(path)


def _markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    columns = ("dataset", "method", *fields)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_results(out_dir: Path, by_dataset: list[dict], macro: list[dict], feature_rows: list[dict]):
    hard_fields = (
        "hard_S_m", "hard_F_beta_w", "hard_E_mean", "hard_MAE", "hard_IoU",
        "hard_Precision", "hard_Recall", "hard_Area",
    )
    ranking_fields = ("pixel_AP", "pixel_AUROC", "best_IoU_256")
    diagnostics = [row for row in feature_rows if row["dataset"] == "ALL"]
    text = f"""# DINOv2 Final Patch Token R1 Pilot

同一批样本、同一输入、同一 R1 参数；Last-Layer Key 使用已有缓存，Final Patch Token 来自 `last_hidden_state[:,1:]`。本 pilot 不保存新特征缓存。

## Hard by dataset

{_markdown_table(by_dataset, hard_fields)}

## Dataset macro

{_markdown_table(macro, hard_fields)}

## Ranking diagnostic（逐图平均）

{_markdown_table(macro, ranking_fields)}

## Feature geometry

{_markdown_table(diagnostics, ("patch00_norm_ratio_mean", "max_norm_ratio_mean", "norm_argmax_top_left_2x2_rate", "local_distance_mean", "top_left_incident_distance_mean"))}
"""
    (out_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def _final_r1_cache_path(cache_root: Path, dataset: str, stem: str) -> Path:
    return cache_root / dataset / f"{stem}.pt"


def _load_final_r1_cache(path: Path, dataset: str, stem: str) -> tuple[torch.Tensor, dict]:
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Final R1 cache payload must be dict: {path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Final R1 cache key mismatch: {path}")
    if payload.get("feature_type") != "final_patch_token_after_model_layernorm":
        raise RuntimeError(f"Final R1 cache feature type mismatch: {path}")
    r1 = _load_response(payload, "residual_pass1_37", (37, 37), path)
    record = payload.get("feature_record")
    if not isinstance(record, dict):
        raise RuntimeError(f"Final R1 cache missing feature_record: {path}")
    for field in ("local_distance", "top_left_incident_distance"):
        value = np.asarray(record.get(field), dtype=np.float32)
        if not value.size or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid cached {field}: {path}")
        record[field] = value
    return r1, record


def _save_final_r1_cache(
    path: Path,
    dataset: str,
    stem: str,
    image_path: str,
    gt_path: str,
    result: dict,
    feature_record: dict,
):
    ensure_dir(path.parent)
    payload = {
        "dataset": dataset,
        "stem": stem,
        "image_path": image_path,
        "gt_path": gt_path,
        "backbone_key": "dinov2-b14",
        "feature_type": "final_patch_token_after_model_layernorm",
        "residual_pass1_37": result["residual_pass1_37"].detach().cpu().float().contiguous(),
        "bc_map_37": result["bc_map_37"].detach().cpu().float().contiguous(),
        "bg_anchor_37": result["bg_anchor_37"].detach().cpu().float().contiguous(),
        "feature_record": feature_record,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args):
    start = time.time()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    ensure_dir(out_dir)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    feature_manifest = feature_manifest_path(cfg, "test")
    feature_rows = read_jsonl(feature_manifest)
    run_datasets = DATASETS[:3] if args.stop_after_cod10k else DATASETS
    selected = (
        select_all_samples(feature_rows, run_datasets)
        if args.all_samples
        else select_uniform_per_dataset(feature_rows, args.samples_per_dataset, args.max_samples)
    )
    feature_map = manifest_to_map(feature_rows, feature_manifest)
    dabe_manifest = Path(args.dabe_manifest).expanduser().resolve()
    dabe_rows = read_jsonl(dabe_manifest)
    dabe_map = manifest_to_map(dabe_rows, dabe_manifest)
    for row in selected:
        key = (row["dataset"], row["stem"])
        if key not in dabe_map:
            raise RuntimeError(f"DABE cache missing selected key: {key}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg), "VERSION": "v2"}
    final_r1_cache_root = (
        Path(args.final_r1_cache_root).expanduser().resolve()
        if args.final_r1_cache_root
        else None
    )
    if final_r1_cache_root is not None:
        ensure_dir(final_r1_cache_root)
    metric_rows = []
    per_sample = []
    feature_records = []
    vis_records = []
    cache_rows = []
    dino_forward_count = 0
    cache_reuse_count = 0

    print(f"device = {device}", flush=True)
    print(f"num_samples = {len(selected)}", flush=True)
    print(f"feature_manifest = {feature_manifest}", flush=True)
    print("dino_forward_used = true", flush=True)
    print("training_used = false", flush=True)
    print("final_feature_cache_saved = false", flush=True)
    print(f"final_r1_cache_root = {final_r1_cache_root}", flush=True)

    with torch.no_grad():
        for index, feature_row in enumerate(selected, 1):
            dataset, stem = feature_row["dataset"], feature_row["stem"]
            key = (dataset, stem)
            dabe_row = dabe_map[key]
            dabe_payload = torch_load(dabe_row["cache_path"], map_location="cpu")
            key_r1 = _load_response(
                dabe_payload, "residual_pass1_37", (37, 37), Path(dabe_row["cache_path"])
            )
            key_feature = _load_feature(feature_map[key], dataset, stem, cfg)

            final_cache_path = (
                _final_r1_cache_path(final_r1_cache_root, dataset, stem)
                if final_r1_cache_root is not None
                else None
            )
            if final_cache_path is not None and final_cache_path.exists() and args.resume:
                final_r1, final_feature_record = _load_final_r1_cache(
                    final_cache_path, dataset, stem
                )
                final_result = None
                cache_reuse_count += 1
                source = "cache"
            else:
                if final_cache_path is not None and final_cache_path.exists():
                    raise FileExistsError(
                        f"Final R1 cache exists; pass --resume to reuse: {final_cache_path}"
                    )
                inputs, _ = preprocess_image(
                    dabe_row["image_path"], int(cfg.DINO["feature_input_size"]), "bicubic"
                )
                outputs = call_dino(model, inputs.to(device))
                final_feature = final_hidden_to_feature(outputs.last_hidden_state, cfg)
                final_result = generate_dabe_pseudo(
                    final_feature,
                    dabe_row["image_path"],
                    params=params,
                    augs=["identity"],
                )
                final_r1 = final_result["residual_pass1_37"].float().contiguous()
                final_feature_record = _feature_record(
                    dataset, stem, METHODS[1], final_feature
                )
                if final_cache_path is not None:
                    _save_final_r1_cache(
                        final_cache_path,
                        dataset,
                        stem,
                        dabe_row["image_path"],
                        dabe_row["gt_path"],
                        final_result,
                        final_feature_record,
                    )
                dino_forward_count += 1
                source = "dino"
            if tuple(final_r1.shape) != (1, 37, 37) or not bool(torch.isfinite(final_r1).all()):
                raise RuntimeError(f"Invalid Final Patch Token R1: {dataset}/{stem}")

            gt = _load_gt(dabe_row["gt_path"])
            key_probability = _resize_r1_to_gt(key_r1, tuple(gt.shape[-2:]))
            final_probability = _resize_r1_to_gt(final_r1, tuple(gt.shape[-2:]))
            evaluated = _evaluate_pair(key_probability, final_probability, gt)
            for method in METHODS:
                metric_rows.append({"dataset": dataset, "stem": stem, "method": method, **evaluated[method]})

            key_feature_record = _feature_record(dataset, stem, METHODS[0], key_feature)
            feature_records.extend((key_feature_record, final_feature_record))
            sample = {
                "dataset": dataset,
                "stem": stem,
                "image_path": dabe_row["image_path"],
                "gt_path": dabe_row["gt_path"],
                "key_r1_map_mae_vs_final": float((key_r1 - final_r1).abs().mean()),
                "key_r1_map_max_abs_vs_final": float((key_r1 - final_r1).abs().max()),
            }
            for field in METRIC_FIELDS:
                sample[f"key_{field}"] = evaluated[METHODS[0]][field]
                sample[f"final_{field}"] = evaluated[METHODS[1]][field]
                sample[f"delta_{field}"] = evaluated[METHODS[1]][field] - evaluated[METHODS[0]][field]
            per_sample.append(sample)
            vis_records.append(
                {
                    **sample,
                    "key_r1": key_r1,
                    "final_r1": final_r1,
                }
            )
            if final_cache_path is not None:
                cache_rows.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "cache_path": str(final_cache_path),
                        "feature_type": "final_patch_token_after_model_layernorm",
                        "shape_37": [1, 37, 37],
                    }
                )
            print(
                f"processed = {index}/{len(selected)} | {dataset}/{stem} | source={source}",
                flush=True,
            )
            if source == "dino":
                del outputs, inputs, final_feature

    by_dataset, macro = _aggregate_metrics(metric_rows, run_datasets)
    feature_summary = _aggregate_features(feature_records, run_datasets)
    _write_csv(out_dir / "selected_samples.csv", [
        {"selection_index": i, "dataset": row["dataset"], "stem": row["stem"]}
        for i, row in enumerate(selected, 1)
    ])
    _write_csv(out_dir / "per_sample.csv", per_sample)
    _write_csv(out_dir / "metrics_by_dataset.csv", by_dataset)
    _write_csv(out_dir / "metrics_dataset_macro.csv", macro)
    _write_csv(out_dir / "feature_geometry.csv", feature_summary)
    if final_r1_cache_root is not None:
        write_jsonl(final_r1_cache_root / "manifest_test.jsonl", cache_rows)

    if args.save_vis:
        for dataset in run_datasets:
            records = [row for row in vis_records if row["dataset"] == dataset]
            groups = {
                "fixed2": records[:2],
                "best_delta2": sorted(records, key=lambda row: row["delta_hard_IoU"], reverse=True)[:2],
                "worst_delta2": sorted(records, key=lambda row: row["delta_hard_IoU"])[:2],
            }
            for group, values in groups.items():
                for record in values:
                    _save_visualization(record, out_dir / "vis" / group / dataset / f"{record['stem']}.png")

    elapsed = time.time() - start
    protocol = {
        "experiment": str(cfg.EXP_NAME),
        "config_path": str(Path(args.config).resolve()),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": _sha256(SCRIPT_PATH),
        "feature_manifest": str(feature_manifest),
        "dabe_manifest": str(dabe_manifest),
        "datasets": list(run_datasets),
        "samples_per_dataset_requested": int(args.samples_per_dataset),
        "num_samples": len(selected),
        "selection": (
            "all test manifest samples"
            if args.all_samples
            else "deterministic uniform linspace per dataset, round-robin truncation"
        ),
        "backbone": "facebook/dinov2-base",
        "input_size": 518,
        "patch_size": 14,
        "key_feature": "last-layer attention key existing cache",
        "candidate_feature": "outputs.last_hidden_state[:,1:] after model final layernorm",
        "feature_shape": [768, 37, 37],
        "r1_params_unchanged": True,
        "hard_threshold": "strict > 0.5",
        "resize": "37->68 bilinear align_corners=False -> original GT bilinear align_corners=False",
        "dino_forward_used": True,
        "dino_forward_count": dino_forward_count,
        "final_r1_cache_reuse_count": cache_reuse_count,
        "training_used": False,
        "final_feature_cache_saved": False,
        "final_r1_cache_saved": final_r1_cache_root is not None,
        "final_r1_cache_root": str(final_r1_cache_root) if final_r1_cache_root else None,
        "elapsed_seconds": elapsed,
    }
    summary = {
        "protocol": protocol,
        "metrics_by_dataset": by_dataset,
        "metrics_dataset_macro": macro,
        "feature_geometry": feature_summary,
    }
    write_json(out_dir / "protocol.json", protocol)
    write_json(out_dir / "summary.json", summary)
    _write_results(out_dir, by_dataset, macro, feature_summary)
    print(f"elapsed_seconds = {elapsed:.3f}", flush=True)
    print(f"wrote = {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dinov2_b14_final_patch_r1_pilot.py")
    parser.add_argument(
        "--dabe_manifest",
        default="../workdir/dabe_r1_dinov2_b14_identity/dinov2-b14/manifest_test.jsonl",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--samples_per_dataset", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--all_samples", action="store_true")
    parser.add_argument("--stop_after_cod10k", action="store_true")
    parser.add_argument("--final_r1_cache_root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=6)
    parser.add_argument("--save_vis", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
