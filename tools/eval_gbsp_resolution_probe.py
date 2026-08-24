#!/usr/bin/env python3
"""Evaluate 296 vs 512-A/B/C GBSP on one deterministic native-grid subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_resolution import (  # noqa: E402
    fit_gbsp_from_prepared,
    hard_mask_at_native,
    prepare_resolution_graph,
)


METHODS = ("296-baseline", "512-A", "512-B", "512-C")
METHOD_DIRS = {
    "296-baseline": "dinov1-s8_296",
    "512-A": "dinov1-s8_512_native64_bw2",
    "512-B": "dinov1-s8_512_native64_bw3",
    "512-C": "dinov1-s8_512_native64_bw2_r12",
}
QUALITY_FIELDS = (
    "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision",
    "Recall", "Area", "IoU", "boundary_F1", "pixel_AP", "pixel_AUROC",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary(value: torch.Tensor) -> dict[str, float]:
    array = value.detach().cpu().float().numpy().reshape(-1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, .10)),
        "p50": float(np.quantile(array, .50)),
        "p90": float(np.quantile(array, .90)),
        "p95": float(np.quantile(array, .95)),
    }


def _load_gt(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy((array > .5).astype(np.float32)).unsqueeze(0)


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    values = score.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels = gt.detach().cpu().numpy().reshape(-1) > .5
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}
    order = np.argsort(-values, kind="stable")
    ordered_score, ordered_label = values[order], labels[order]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positives
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    group_pos = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_neg = ends - starts - group_pos
    negative_below = negatives - np.cumsum(group_neg)
    wins = np.sum(group_pos * (negative_below + .5 * group_neg), dtype=np.float64)
    return {"pixel_AP": ap, "pixel_AUROC": float(wins / (positives * negatives))}


def _boundary_f1(prediction: torch.Tensor, gt: torch.Tensor) -> float:
    pred = prediction.squeeze().numpy() > .5
    target = gt.squeeze().numpy() > .5
    pred_boundary = pred ^ binary_erosion(pred, structure=np.ones((3, 3)), border_value=0)
    gt_boundary = target ^ binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
    tolerance = max(1, int(round(.003 * math.hypot(*target.shape))))
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0
    precision = float((pred_boundary & binary_dilation(gt_boundary, structure)).sum()) / float(pred_boundary.sum())
    recall = float((gt_boundary & binary_dilation(pred_boundary, structure)).sum()) / float(gt_boundary.sum())
    return 2.0 * precision * recall / (precision + recall + 1e-12)


def _quality(hard_original: torch.Tensor, continuous: torch.Tensor, gt: torch.Tensor) -> dict:
    context = FastCODContext(gt)
    value = context.evaluate_many([("candidate", "hard", hard_original)], .5)[("candidate", "hard")]
    return {
        "S_m": float(value["S_m"]),
        "F_beta_w": float(value["F_beta_w"]),
        "F_beta_mean": float(value["F_beta_mean"]),
        "E_mean": float(value["E_mean"]),
        "MAE": float(value["MAE"]),
        "Precision": float(value["Precision"]),
        "Recall": float(value["Recall"]),
        "Area": float(value["Area"]),
        "IoU": float(value["IoU"]),
        "boundary_F1": _boundary_f1(hard_original, gt),
        **_rank_metrics(continuous, gt),
    }


def _manifest_map(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in result:
            raise RuntimeError(f"duplicate manifest key {key}: {path}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        result[key] = row
    return result


def _sample_rows(path: Path, max_samples: int) -> list[dict]:
    rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        key = (str(row["dataset"]), str(row["stem"]))
        if key in seen:
            raise RuntimeError(f"duplicate sample key: {key}")
        seen.add(key)
        rows.append(row)
    if max_samples >= 0:
        rows = rows[:max_samples]
    if not rows:
        raise RuntimeError("sample list is empty")
    return rows


def _load_feature(row: dict, cfg, key: tuple[str, str]) -> torch.Tensor:
    payload = torch_load(row["cache_path"], map_location="cpu")
    if (str(payload.get("dataset")), str(payload.get("stem"))) != key:
        raise RuntimeError(f"feature identity mismatch: {row['cache_path']}")
    input_size = int(cfg.DINO["feature_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    grid = input_size // patch_size
    expected = (int(cfg.DINO["embed_dim"]), grid, grid)
    feature = payload.get("tensor")
    if not torch.is_tensor(feature) or tuple(feature.shape) != expected:
        raise RuntimeError(f"feature shape mismatch: {row['cache_path']} -> {getattr(feature, 'shape', None)} != {expected}")
    for field, expected_value in (
        ("input_size", input_size), ("grid_size", grid),
        ("token_count", grid * grid), ("feature_dim", expected[0]),
    ):
        actual = payload.get(field, row.get(field))
        if actual is not None and int(actual) != int(expected_value):
            raise RuntimeError(f"feature metadata {field} mismatch: {actual} != {expected_value}")
    return feature.detach().cpu().float().contiguous()


def _cfg_params(cfg) -> dict:
    return {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg), "VERSION": "v2"}


def _validate_protocol(cfg296, cfg_a, cfg_b, cfg_c) -> None:
    checks = [
        (int(cfg296.DINO["feature_input_size"]), 296, "296 input"),
        (int(cfg296.GRID_SIZE), 37, "296 grid"),
        (int(cfg296.GBSP_BORDER_WIDTH), 2, "296 BW"),
        (bool(cfg296.USE_LEGACY_68_INTERPOLATION), True, "296 legacy68"),
        (int(cfg_a.DINO["feature_input_size"]), 512, "512-A input"),
        (int(cfg_a.GRID_SIZE), 64, "512-A grid"),
        (int(cfg_a.GBSP_BORDER_WIDTH), 2, "512-A BW"),
        (bool(cfg_a.USE_LEGACY_68_INTERPOLATION), False, "512-A native"),
        (int(cfg_b.DINO["feature_input_size"]), 512, "512-B input"),
        (int(cfg_b.GRID_SIZE), 64, "512-B grid"),
        (int(cfg_b.GBSP_BORDER_WIDTH), 3, "512-B BW"),
        (bool(cfg_b.USE_LEGACY_68_INTERPOLATION), False, "512-B native"),
        (int(cfg_c.DINO["feature_input_size"]), 512, "512-C input"),
        (int(cfg_c.GRID_SIZE), 64, "512-C grid"),
        (int(cfg_c.GBSP_BORDER_WIDTH), 2, "512-C BW"),
        (bool(cfg_c.USE_LEGACY_68_INTERPOLATION), False, "512-C native"),
        (int(cfg296.GBSP_PCA_MAX_RANK), 8, "296 rank cap"),
        (int(cfg_a.GBSP_PCA_MAX_RANK), 8, "512-A rank cap"),
        (int(cfg_b.GBSP_PCA_MAX_RANK), 8, "512-B rank cap"),
        (int(cfg_c.GBSP_PCA_MAX_RANK), 12, "512-C rank cap"),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            raise RuntimeError(f"protocol mismatch {name}: {actual!r} != {expected!r}")
    frozen = ("GBSP_BG_RATIO", "GBSP_PCA_ENERGY", "GBSP_PCA_MIN_RANK", "GBSP_THRESHOLD")
    for field in frozen:
        values = [getattr(cfg, field) for cfg in (cfg296, cfg_a, cfg_b, cfg_c)]
        if values[1:] != values[:1] * 3:
            raise RuntimeError(f"frozen field differs across resolution: {field}={values}")
    for field in ("DABE_SIGMA_F", "DABE_SIGMA_C", "DABE_SIGMA_E", "DABE_TAU_BC", "DABE_BG_ANCHOR_TOP_PERCENT"):
        values = [getattr(cfg, field) for cfg in (cfg296, cfg_a, cfg_b, cfg_c)]
        if values[1:] != values[:1] * 3:
            raise RuntimeError(f"graph field differs across resolution: {field}={values}")


def _existing_reference_error(root: Path, dataset: str, stem: str, score: torch.Tensor) -> float:
    path = root / "test" / dataset / f"{stem}.pt"
    if not path.is_file():
        return float("nan")
    payload = torch_load(path, map_location="cpu")
    reference = payload["results"]["current"]["absolute_minmax"].float()
    if tuple(reference.shape) != tuple(score.shape):
        raise RuntimeError(f"formal 296 reference shape mismatch: {path}")
    return float((reference - score).abs().max())


def _aggregate(rows: list[dict]) -> list[dict]:
    outputs = []
    datasets = sorted({row["dataset"] for row in rows})
    for method in METHODS:
        dataset_rows = []
        for dataset in datasets:
            selected = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
            if not selected:
                continue
            summary = {"dataset": dataset, "method": method, "num_images": len(selected)}
            for field in QUALITY_FIELDS:
                summary[field] = float(np.nanmean([float(row[field]) for row in selected]))
            outputs.append(summary)
            dataset_rows.append(summary)
        macro = {"dataset": "Dataset-Macro", "method": method, "num_images": sum(int(row["num_images"]) for row in dataset_rows)}
        for field in QUALITY_FIELDS:
            macro[field] = float(np.nanmean([float(row[field]) for row in dataset_rows]))
        outputs.append(macro)
    return outputs


def _save_visualizations(records: list[dict], out: Path, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    gt_areas = np.asarray([record["gt_area"] for record in records], dtype=float)
    categories = [
        ("small_object", lambda r: r["gt_area"]),
        ("thin_complex_boundary", lambda r: -r["boundary_complexity"]),
        ("large_object", lambda r: -r["gt_area"]),
        ("border_touch", lambda r: (not r["border_touch"], -r["boundary_complexity"])),
        ("textured_background", lambda r: -r["background_texture"]),
        ("largest_512_gain", lambda r: -r["delta_fbw_b_vs_296"]),
        ("largest_512_drop", lambda r: r["delta_fbw_b_vs_296"]),
    ]
    chosen, seen = [], set()
    quota = max(1, int(math.ceil(limit / len(categories))))
    for category, key_fn in categories:
        for record in sorted(records, key=key_fn):
            key = (record["dataset"], record["stem"])
            if key in seen or (category == "border_touch" and not record["border_touch"]):
                continue
            chosen.append((category, record))
            seen.add(key)
            if sum(name == category for name, _ in chosen) >= quota or len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break
    for record in records:
        if len(chosen) >= limit:
            break
        key = (record["dataset"], record["stem"])
        if key not in seen:
            chosen.append(("representative", record)); seen.add(key)

    out.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for index, (category, record) in enumerate(chosen[:limit], 1):
        with Image.open(record["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        gt = record["gt"].squeeze().numpy()
        panels = [
            (rgb, "RGB", None), (gt, "GT", "gray"),
            (record["scores"]["296-baseline"], "296 residual", "gray"),
            (record["masks"]["296-baseline"], "296 hard@68", "gray"),
            (record["scores"]["512-A"], "512-A residual", "gray"),
            (record["masks"]["512-A"], "512-A hard@64", "gray"),
            (record["scores"]["512-B"], "512-B residual", "gray"),
            (record["masks"]["512-B"], "512-B hard@64", "gray"),
            (record["scores"]["512-C"], "512-C residual", "gray"),
            (record["masks"]["512-C"], "512-C hard@64", "gray"),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(30, 4.2))
        for axis, (value, title, cmap) in zip(axes, panels):
            axis.imshow(value, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(
            f"{record['dataset']}/{record['stem']} | {category} | "
            f"ΔFw(B-296)={record['delta_fbw_b_vs_296']:+.4f}",
            fontsize=11,
        )
        fig.tight_layout()
        path = out / f"{index:02d}_{category}_{record['dataset']}_{record['stem']}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        index_rows.append({
            "index": index, "category": category, "dataset": record["dataset"],
            "stem": record["stem"], "path": str(path),
            "gt_area": record["gt_area"], "boundary_complexity": record["boundary_complexity"],
            "border_touch": int(record["border_touch"]),
            "background_texture": record["background_texture"],
            "delta_F_beta_w_512B_vs_296": record["delta_fbw_b_vs_296"],
        })
    _write_csv(out / "visualization_index.csv", index_rows)
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-296", required=True)
    parser.add_argument("--config-512a", required=True)
    parser.add_argument("--config-512b", required=True)
    parser.add_argument("--config-512c", required=True)
    parser.add_argument("--manifest-296", required=True)
    parser.add_argument("--manifest-512", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--formal-296-root", default="../workdir/gbsp_core_optimization/full_rank")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--visualizations", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    cfg296, cfg_a, cfg_b, cfg_c = (
        load_config(path)
        for path in (args.config_296, args.config_512a, args.config_512b, args.config_512c)
    )
    _validate_protocol(cfg296, cfg_a, cfg_b, cfg_c)
    configs = {
        "296-baseline": cfg296,
        "512-A": cfg_a,
        "512-B": cfg_b,
        "512-C": cfg_c,
    }
    params = {method: _cfg_params(cfg) for method, cfg in configs.items()}
    manifests = {
        "296": _manifest_map(Path(args.manifest_296).expanduser().resolve()),
        "512": _manifest_map(Path(args.manifest_512).expanduser().resolve()),
    }
    samples = _sample_rows(Path(args.sample_list).expanduser().resolve(), args.max_samples)
    for row in samples:
        key = (str(row["dataset"]), str(row["stem"]))
        if key not in manifests["296"] or key not in manifests["512"]:
            raise KeyError(f"feature missing for sample {key}")

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    command = " ".join([sys.executable, *sys.argv])
    (out / "run_command.txt").write_text(command + "\n", encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    shutil.copy2(Path(args.sample_list).expanduser().resolve(), out / "sample_list.jsonl")

    graph_rows, candidate_rows, pca_rows = [], [], []
    residual_rows, quality_rows, runtime_rows = [], [], []
    reproduction_rows, visualization_records = [], []
    pooled_graph: dict[tuple[str, str], list[torch.Tensor]] = {}
    formal_root = Path(args.formal_296_root).expanduser().resolve()

    for sample_index, sample in enumerate(samples, 1):
        dataset, stem = str(sample["dataset"]), str(sample["stem"])
        key = (dataset, stem)
        gt = _load_gt(sample["gt_path"])
        original_hw = tuple(map(int, gt.shape[-2:]))
        features = {
            "296": _load_feature(manifests["296"][key], cfg296, key),
            "512": _load_feature(manifests["512"][key], cfg_a, key),
        }
        prepared296 = prepare_resolution_graph(features["296"], sample["image_path"], params["296-baseline"])
        prepared512 = prepare_resolution_graph(features["512"], sample["image_path"], params["512-A"])
        prepared_by = {
            "296-baseline": prepared296,
            "512-A": prepared512,
            "512-B": prepared512,
            "512-C": prepared512,
        }
        results = {
            method: fit_gbsp_from_prepared(
                prepared_by[method], params[method],
                pca_energy=float(configs[method].GBSP_PCA_ENERGY),
                pca_min_rank=int(configs[method].GBSP_PCA_MIN_RANK),
                pca_max_rank=int(configs[method].GBSP_PCA_MAX_RANK),
            )
            for method in METHODS
        }

        for resolution, prepared in (("296", prepared296), ("512", prepared512)):
            row = {
                "image": stem, "dataset": dataset, "resolution": resolution,
                "grid_h": prepared.grid_h, "grid_w": prepared.grid_w,
                "rgb_mean": float(prepared.rgb.mean()), "rgb_std": float(prepared.rgb.std()),
                "sobel_mean": float(prepared.sobel.mean()), "sobel_std": float(prepared.sobel.std()),
            }
            for metric, values in prepared.graph_terms.items():
                for stat, value in _summary(values).items():
                    row[f"{metric}_{stat}"] = value
                pooled_graph.setdefault((resolution, metric), []).append(values)
            graph_rows.append(row)

        method_quality = {}
        native_masks, scores = {}, {}
        gt_area = float(gt.mean())
        gt_np = gt.squeeze().numpy() > .5
        boundary = gt_np ^ binary_erosion(gt_np, structure=np.ones((3, 3)), border_value=0)
        boundary_complexity = float(boundary.sum()) / max(1.0, float(gt_np.sum()))
        border_touch = bool(gt_np[0].any() or gt_np[-1].any() or gt_np[:, 0].any() or gt_np[:, -1].any())
        gt512 = F.interpolate(gt.unsqueeze(0), size=(64, 64), mode="nearest").squeeze(0).squeeze(0) > .5
        background_texture = float(prepared512.sobel.reshape(-1)[~gt512.reshape(-1)].mean()) if bool((~gt512).any()) else 0.0

        for method in METHODS:
            cfg, prepared, result = configs[method], prepared_by[method], results[method]
            legacy = 68 if bool(cfg.USE_LEGACY_68_INTERPOLATION) else None
            hard_native, hard_original = hard_mask_at_native(
                result.minmax_residual, float(cfg.GBSP_THRESHOLD), original_hw,
                legacy_intermediate_size=legacy,
            )
            continuous = F.interpolate(
                result.minmax_residual.unsqueeze(0), size=original_hw,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            quality = _quality(hard_original, continuous, gt)
            method_quality[method] = quality
            native_masks[method] = hard_native.squeeze().numpy()
            scores[method] = result.minmax_residual.squeeze().numpy()
            quality_rows.append({
                "image": stem, "dataset": dataset, "method": method,
                "input_size": int(cfg.DINO["feature_input_size"]),
                "grid_size": int(cfg.GRID_SIZE), "border_width": int(cfg.GBSP_BORDER_WIDTH),
                "threshold": float(cfg.GBSP_THRESHOLD),
                "native_target_size": int(cfg.GBSP_NATIVE_TARGET_SIZE),
                "gt_area": gt_area, **quality,
            })

            gt_grid = F.interpolate(
                gt.unsqueeze(0), size=(prepared.grid_h, prepared.grid_w), mode="nearest"
            ).squeeze().reshape(-1) > .5
            contamination = gt_grid.index_select(0, result.background_indices).float()
            bc_summary = _summary(result.bc)
            candidate_rows.append({
                "image": stem, "dataset": dataset, "method": method,
                "resolution": int(cfg.DINO["feature_input_size"]),
                "grid_size": int(cfg.GRID_SIZE), "border_width": int(cfg.GBSP_BORDER_WIDTH),
                "candidate_count": int(result.background_indices.numel()),
                "candidate_ratio": float(result.background_indices.numel() / (prepared.grid_h * prepared.grid_w)),
                "candidate_precision": float(1.0 - contamination.mean()),
                "foreground_contamination_ratio": float(contamination.mean()),
                **{f"bc_{name}": value for name, value in bc_summary.items()},
            })
            pca_rows.append({
                "image": stem, "dataset": dataset, "method": method,
                "resolution": int(cfg.DINO["feature_input_size"]),
                "candidate_count": int(result.background_indices.numel()),
                "selected_rank": result.selected_rank,
                "required_rank_uncapped": result.required_rank_uncapped,
                "rank_cap": int(cfg.GBSP_PCA_MAX_RANK),
                "energy_at_rank_cap": result.energy_at_rank_max,
                "energy_at_rank8": (
                    result.energy_at_rank_max
                    if int(cfg.GBSP_PCA_MAX_RANK) == 8 else float("nan")
                ),
                "retained_energy": result.retained_energy,
                "hit_rank_cap": int(result.hit_rank_cap),
            })
            raw_flat = result.raw_residual.reshape(-1)
            mm_flat = result.minmax_residual.reshape(-1)
            fg = raw_flat[gt_grid]
            bg = raw_flat[~gt_grid]
            residual_rows.append({
                "image": stem, "dataset": dataset, "method": method,
                "resolution": int(cfg.DINO["feature_input_size"]),
                "residual_mean": float(raw_flat.mean()), "residual_std": float(raw_flat.std()),
                "residual_p50": float(torch.quantile(raw_flat, .5)),
                "residual_p90": float(torch.quantile(raw_flat, .9)),
                "minmax_mean": float(mm_flat.mean()), "minmax_std": float(mm_flat.std()),
                "foreground_area": float(hard_native.mean()),
                "GT_BG_residual_mean": float(bg.mean()) if bg.numel() else float("nan"),
                "GT_FG_residual_mean": float(fg.mean()) if fg.numel() else float("nan"),
            })
            feature_row = manifests["296" if method == "296-baseline" else "512"][key]
            feature_sec = float(feature_row.get("total_seconds", feature_row.get("dino_seconds", float("nan"))))
            gbsp_sec = prepared.rgb_seconds + prepared.graph_seconds + result.bc_seconds + result.pca_seconds
            runtime_rows.append({
                "image": stem, "dataset": dataset, "method": method,
                "feature_sec_per_image": feature_sec,
                "dino_sec_per_image": float(feature_row.get("dino_seconds", float("nan"))),
                "rgb_sec_per_image": prepared.rgb_seconds,
                "graph_sec_per_image": prepared.graph_seconds,
                "bc_sec_per_image": result.bc_seconds,
                "pca_sec_per_image": result.pca_seconds,
                "gbsp_sec_per_image": gbsp_sec,
                "total_sec_per_image": feature_sec + gbsp_sec,
                "peak_gpu_memory_mb": float(feature_row.get("gpu_peak_mb", float("nan"))),
                "cache_mb_per_image": float(feature_row.get("cache_bytes", Path(feature_row["cache_path"]).stat().st_size) / (1024 ** 2)),
            })
            method_root = out / METHOD_DIRS[method]
            score_path = method_root / "scores" / dataset / f"{stem}.pt"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "version": str(cfg.GBSP_RESOLUTION_VERSION), "method": method,
                "dataset": dataset, "stem": stem, "input_size": int(cfg.DINO["feature_input_size"]),
                "grid_size": int(cfg.GRID_SIZE), "border_width": int(cfg.GBSP_BORDER_WIDTH),
                "bg_ratio": float(cfg.GBSP_BG_RATIO), "pca_energy": float(cfg.GBSP_PCA_ENERGY),
                "pca_rank_max": int(cfg.GBSP_PCA_MAX_RANK), "pca_rank_min": int(cfg.GBSP_PCA_MIN_RANK),
                "threshold": float(cfg.GBSP_THRESHOLD), "native_target_size": int(cfg.GBSP_NATIVE_TARGET_SIZE),
                "dino_backbone": str(cfg.DINO["model_name"]), "commit_hash": commit,
                "gt_used_for_generation": False, "absolute_raw": result.raw_residual,
                "absolute_minmax": result.minmax_residual, "pseudo_native": hard_native,
                "background_indices": result.background_indices, "bc": result.bc,
                "selected_rank": result.selected_rank, "required_rank_uncapped": result.required_rank_uncapped,
                "energy_at_rank_cap": result.energy_at_rank_max,
            }, score_path)

        reference_error = _existing_reference_error(
            formal_root, dataset, stem, results["296-baseline"].minmax_residual
        )
        reproduction_rows.append({"dataset": dataset, "stem": stem, "max_abs_error": reference_error})
        visualization_records.append({
            "dataset": dataset, "stem": stem, "image_path": sample["image_path"], "gt": gt,
            "gt_area": gt_area, "boundary_complexity": boundary_complexity,
            "border_touch": border_touch, "background_texture": background_texture,
            "delta_fbw_b_vs_296": method_quality["512-B"]["F_beta_w"] - method_quality["296-baseline"]["F_beta_w"],
            "delta_fbw_c_vs_296": method_quality["512-C"]["F_beta_w"] - method_quality["296-baseline"]["F_beta_w"],
            "scores": scores, "masks": native_masks,
        })
        if sample_index <= 5 or sample_index % 20 == 0 or sample_index == len(samples):
            print(
                f"[{sample_index}/{len(samples)}] {dataset}/{stem} | "
                f"feature37={list(features['296'].shape)} feature64={list(features['512'].shape)} | "
                f"rank={results['296-baseline'].selected_rank}/{results['512-A'].selected_rank}/"
                f"{results['512-B'].selected_rank}/{results['512-C'].selected_rank}",
                flush=True,
            )

    graph_summary_rows = []
    for (resolution, metric), tensors in sorted(pooled_graph.items()):
        graph_summary_rows.append({
            "resolution": resolution, "metric": metric,
            **_summary(torch.cat(tensors)),
        })
    quality_summary = _aggregate(quality_rows)
    gt_cutoff = float(np.quantile([row["gt_area"] for row in quality_rows if row["method"] == "296-baseline"], .25))
    small_rows = [row for row in quality_rows if float(row["gt_area"]) <= gt_cutoff]
    small_summary = _aggregate(small_rows)
    pca_summary = []
    for method in METHODS:
        rows = [row for row in pca_rows if row["method"] == method]
        ranks = np.asarray([row["selected_rank"] for row in rows], dtype=float)
        rank_cap = int(rows[0]["rank_cap"])
        pca_summary.append({
            "method": method, "num_images": len(rows), "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "rank_cap": rank_cap,
            "p_rank_eq_cap": float(np.mean(ranks == rank_cap)),
            "p_required_gt_cap": float(np.mean([
                row["required_rank_uncapped"] > rank_cap for row in rows
            ])),
            "mean_energy_at_rank_cap": float(np.mean([
                row["energy_at_rank_cap"] for row in rows
            ])),
            "rank_histogram": json.dumps({str(rank): int(np.sum(ranks == rank)) for rank in sorted(set(ranks.astype(int)))}, sort_keys=True),
        })

    combined = {
        "graph_stats.csv": graph_rows, "candidate_stats.csv": candidate_rows,
        "pca_rank_stats.csv": pca_rows, "residual_stats.csv": residual_rows,
        "pseudo_quality.csv": quality_rows, "runtime_stats.csv": runtime_rows,
        "resolution_graph_stats.csv": graph_summary_rows,
        "pseudo_quality_summary.csv": quality_summary,
        "small_object_summary.csv": small_summary,
        "pca_rank_summary.csv": pca_summary,
        "baseline_reproduction.csv": reproduction_rows,
    }
    for name, rows in combined.items():
        _write_csv(out / name, rows)
    for method in METHODS:
        method_root = out / METHOD_DIRS[method]
        method_root.mkdir(parents=True, exist_ok=True)
        cfg_path = Path({
            "296-baseline": args.config_296,
            "512-A": args.config_512a,
            "512-B": args.config_512b,
            "512-C": args.config_512c,
        }[method]).expanduser().resolve()
        shutil.copy2(cfg_path, method_root / "config_snapshot.py")
        (method_root / "run_command.txt").write_text(command + "\n", encoding="utf-8")
        (method_root / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        for filename, rows in combined.items():
            if filename in {"graph_stats.csv", "resolution_graph_stats.csv"}:
                expected_resolution = "296" if method == "296-baseline" else "512"
                filtered = [row for row in rows if str(row.get("resolution")) == expected_resolution]
            elif any("method" in row for row in rows):
                filtered = [row for row in rows if row.get("method") == method]
            else:
                filtered = list(rows)
            if filtered:
                _write_csv(method_root / filename, filtered)
        if method != "296-baseline":
            _write_csv(
                method_root / "pca_rank_stats_512.csv",
                [row for row in pca_rows if row["method"] == method],
            )

    visualization_index = _save_visualizations(
        visualization_records, out / "visualizations", int(args.visualizations)
    )
    for method in METHODS:
        link = out / METHOD_DIRS[method] / "visualizations"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("..") / "visualizations", target_is_directory=True)
    valid_reproduction = [row["max_abs_error"] for row in reproduction_rows if math.isfinite(float(row["max_abs_error"]))]
    runtime_summary = []
    for method in METHODS:
        rows = [row for row in runtime_rows if row["method"] == method]
        runtime_summary.append({
            "method": method,
            **{
                field: float(np.nanmean([row[field] for row in rows]))
                for field in (
                    "feature_sec_per_image", "dino_sec_per_image", "gbsp_sec_per_image",
                    "total_sec_per_image", "peak_gpu_memory_mb", "cache_mb_per_image",
                )
            },
            "estimated_full6473_hours": float(np.nanmean([row["total_sec_per_image"] for row in rows]) * 6473 / 3600),
        })
    _write_csv(out / "runtime_summary.csv", runtime_summary)
    metadata = {
        "version": "gbsp_resolution_probe_v1", "num_samples": len(samples),
        "sample_list_sha256": hashlib.sha256((out / "sample_list.jsonl").read_bytes()).hexdigest(),
        "methods": list(METHODS), "training_triggered": False,
        "full_evaluation_triggered": False, "gt_used_for_generation": False,
        "gt_used_for_offline_analysis": True, "small_object_gt_area_q25": gt_cutoff,
        "visualizations": len(visualization_index), "git_commit": commit,
        "baseline_reproduction_max_abs_error": max(valid_reproduction) if valid_reproduction else None,
        "wall_seconds": time.perf_counter() - started,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "quality_summary": quality_summary, "runtime_summary": runtime_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
