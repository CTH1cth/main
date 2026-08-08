#!/usr/bin/env python3
"""Formal original-size COD evaluation for cached MBSP responses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load, write_json  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_CONFIG = MAIN_ROOT / "configs" / "dinov1_s8_mbsp.py"
DEFAULT_R1_CORE_ROOT = REPO_ROOT / "workdir" / "22"
DEFAULT_BGNULL_MANIFEST = (
    REPO_ROOT / "workdir/dabe_bgnull_v1_identity/dinov1-s8/manifest_test.jsonl"
)
DEFAULT_MBSP_ROOT = REPO_ROOT / "workdir/mbsp_pca_v1/dinov1-s8"
DEFAULT_EVAL_ROOT = REPO_ROOT / "workdir/mbsp_pca_v1_eval"
VERSION = "mbsp_pca_v1"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
SCORE_FIELDS = {
    "relative_raw": "mbsp_rel_raw",
    "relative_minmax": "mbsp_rel_minmax",
    "absolute_raw": "mbsp_abs_raw",
    "absolute_minmax": "mbsp_abs_minmax",
}
FIXED_FIELDS = (
    "S_m",
    "F_beta_w",
    "F_beta_mean",
    "E_mean",
    "MAE",
    "Precision",
    "Recall",
    "Area",
)
SOFT_FIELDS = ("S_m", "F_beta_w", "F_beta_mean", "F_beta_max", "E_mean", "E_max", "MAE")


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_map(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    mapping = {}
    for line, row in enumerate(rows, 1):
        for field in ("dataset", "stem"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{line}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"duplicate key in {path}: {key}")
        mapping[key] = row
    return rows, mapping


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped = {dataset: [] for dataset in DATASETS}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    selected, cursor = [], 0
    while len(selected) < max_samples:
        progressed = False
        for dataset in DATASETS:
            if cursor < len(grouped.get(dataset, ())):
                selected.append(grouped[dataset][cursor])
                progressed = True
                if len(selected) == max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _load_gt(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _load_score(descriptor: dict) -> torch.Tensor:
    path = Path(descriptor["path"])
    payload = torch_load(path, map_location="cpu")
    if descriptor["kind"] == "tensor":
        value = payload
    else:
        if not isinstance(payload, dict):
            raise TypeError(f"expected dict payload: {path}")
        value = payload.get(descriptor["field"])
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{descriptor['field']} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"score contains NaN or Inf: {path}")
    if float(value.min()) < -1e-6:
        raise ValueError(f"score contains negative values: {path}")
    return value.clamp_min(0.0)


def _resize_native(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = torch_f.interpolate(
        value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )
    value = torch_f.interpolate(value, size=shape, mode="bilinear", align_corners=False)
    return value.squeeze(0)


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict:
    values = score.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels = gt.detach().cpu().numpy().reshape(-1) > 0.5
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}

    descending = np.argsort(-values, kind="stable")
    ordered_score = values[descending]
    ordered_label = labels[descending]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positive_count
    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))

    starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    group_positive = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_size = ends - starts
    group_negative = group_size - group_positive
    negative_below = negative_count - np.cumsum(group_negative)
    pair_wins = np.sum(
        group_positive * (negative_below + 0.5 * group_negative), dtype=np.float64
    )
    auc = pair_wins / (positive_count * negative_count)
    return {"pixel_AP": average_precision, "pixel_AUROC": float(auc)}


def _threshold_curve(score: torch.Tensor, gt: torch.Tensor, thresholds: np.ndarray) -> dict:
    values = score.detach().cpu().numpy().astype(np.float64).reshape(-1)
    labels = gt.detach().cpu().numpy().reshape(-1) > 0.5
    positive = np.sort(values[labels])
    negative = np.sort(values[~labels])
    tp = positive.size - np.searchsorted(positive, thresholds, side="right")
    fp = negative.size - np.searchsorted(negative, thresholds, side="right")
    predicted = tp + fp
    precision = np.divide(
        tp,
        predicted,
        out=np.ones_like(tp, dtype=np.float64),
        where=predicted != 0,
    )
    recall = tp / positive.size if positive.size else np.ones_like(tp, dtype=np.float64)
    beta2 = 0.3
    f_beta = np.divide(
        (1.0 + beta2) * precision * recall,
        beta2 * precision + recall,
        out=np.zeros_like(precision),
        where=(beta2 * precision + recall) != 0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "F_beta": f_beta,
        "area": predicted / labels.size,
    }


def _raw_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict:
    clipped = score.clamp(0.0, 1.0)
    target = (gt > 0.5).float()
    return {
        "raw_MAE": float(torch.abs(clipped - target).mean()),
        "raw_mean": float(score.mean()),
        "raw_std": float(score.std(unbiased=False)),
        "raw_max": float(score.max()),
        "raw_clipped_fraction": float((score > 1.0).float().mean()),
    }


def _first_pass_one(task: dict) -> dict:
    try:
        gt = _load_gt(task["gt_path"])
        shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt)
        thresholds = np.asarray(task["thresholds"], dtype=np.float64)
        scores = {
            method: _resize_native(_load_score(descriptor), shape)
            for method, descriptor in task["descriptors"].items()
        }
        cod_inputs = []
        for method, score in scores.items():
            cod_inputs.append((method, "hard", score))
            cod_inputs.append((method, "soft", score))
        cod = context.evaluate_many(cod_inputs, 0.5)
        rows, curves = [], {}
        for method, score in scores.items():
            fixed = cod[(method, "hard")]
            soft = cod[(method, "soft")]
            rank = _rank_metrics(score, gt)
            row = {
                "dataset": task["dataset"],
                "stem": task["stem"],
                "method": method,
                **{f"fixed_{field}": float(fixed[field]) for field in FIXED_FIELDS},
                **{f"soft_{field}": float(soft[field]) for field in SOFT_FIELDS},
                **rank,
                **_raw_metrics(score, gt),
            }
            rows.append(row)
            curves[method] = _threshold_curve(score, gt, thresholds)
        return {"dataset": task["dataset"], "stem": task["stem"], "rows": rows, "curves": curves}
    except Exception as error:
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _continuous_pass_one(task: dict) -> dict:
    """Evaluate native-size AP/AUROC without any thresholded COD metrics."""
    try:
        gt = _load_gt(task["gt_path"])
        shape = tuple(gt.shape[-2:])
        rows = []
        for method, descriptor in task["descriptors"].items():
            score = _resize_native(_load_score(descriptor), shape)
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    **_rank_metrics(score, gt),
                }
            )
        return {"dataset": task["dataset"], "stem": task["stem"], "rows": rows}
    except Exception as error:
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _selected_pass_one(task: dict) -> dict:
    try:
        gt = _load_gt(task["gt_path"])
        shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt)
        binary = []
        for method, descriptor in task["descriptors"].items():
            score = _resize_native(_load_score(descriptor), shape)
            binary.append((method, "hard", (score > task["best_thresholds"][method]).float()))
        cod = context.evaluate_many(binary, 0.5)
        rows = []
        for method, _, _ in binary:
            values = cod[(method, "hard")]
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    "selected_threshold": task["best_thresholds"][method],
                    **{f"selected_{field}": float(values[field]) for field in FIXED_FIELDS},
                }
            )
        return {"dataset": task["dataset"], "stem": task["stem"], "rows": rows}
    except Exception as error:
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate_rows(rows: list[dict], prefix: str, datasets: tuple[str, ...], methods: list[str]) -> list[dict]:
    fields = [key for key in rows[0] if key.startswith(prefix)]
    output = []
    for dataset in datasets:
        for method in methods:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not subset:
                continue
            output.append(
                {
                    "scope": "dataset",
                    "dataset": dataset,
                    "method": method,
                    "num_samples": len(subset),
                    **{field: _mean(row[field] for row in subset) for field in fields},
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
                **{field: _mean(row[field] for row in subset) for field in fields},
            }
        )
    return output


def _aggregate_rank(rows: list[dict], datasets: tuple[str, ...], methods: list[str]) -> list[dict]:
    fields = (
        "pixel_AP",
        "pixel_AUROC",
        "raw_MAE",
        "raw_mean",
        "raw_std",
        "raw_max",
        "raw_clipped_fraction",
    )
    output = []
    for dataset in datasets:
        for method in methods:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if subset:
                output.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "method": method,
                        "num_samples": len(subset),
                        **{field: _mean(row[field] for row in subset) for field in fields},
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
                **{field: _mean(row[field] for row in subset) for field in fields},
            }
        )
    return output


def _aggregate_continuous(
    rows: list[dict], datasets: tuple[str, ...], methods: list[str]
) -> list[dict]:
    metrics = ("pixel_AP", "pixel_AUROC")
    output = []
    for dataset in datasets:
        for method in methods:
            subset = [
                row for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            if subset:
                output.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "method": method,
                        "num_samples": len(subset),
                        **{
                            metric: _mean(row[metric] for row in subset)
                            for metric in metrics
                        },
                    }
                )
    for method in methods:
        dataset_rows = [
            row for row in output
            if row["scope"] == "dataset" and row["method"] == method
        ]
        image_rows = [row for row in rows if row["method"] == method]
        output.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                "num_samples": sum(row["num_samples"] for row in dataset_rows),
                **{
                    metric: _mean(row[metric] for row in dataset_rows)
                    for metric in metrics
                },
            }
        )
        output.append(
            {
                "scope": "image_macro",
                "dataset": "ALL",
                "method": method,
                "num_samples": len(image_rows),
                **{
                    metric: _mean(row[metric] for row in image_rows)
                    for metric in metrics
                },
            }
        )
    return output


def _aggregate_curves(
    results: list[dict], thresholds: np.ndarray, datasets: tuple[str, ...], methods: list[str]
) -> tuple[list[dict], dict[str, float], dict[tuple[str, str], float]]:
    curve_rows = []
    dataset_arrays = {}
    for dataset in datasets:
        subset = [result for result in results if result["dataset"] == dataset]
        for method in methods:
            arrays = {
                field: np.mean([result["curves"][method][field] for result in subset], axis=0)
                for field in ("precision", "recall", "F_beta", "area")
            }
            dataset_arrays[(dataset, method)] = arrays
            for index, threshold in enumerate(thresholds):
                curve_rows.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "method": method,
                        "threshold": float(threshold),
                        **{field: float(value[index]) for field, value in arrays.items()},
                    }
                )

    best_global = {}
    best_dataset = {}
    for method in methods:
        available = [dataset_arrays[(dataset, method)] for dataset in datasets]
        macro = {
            field: np.mean([values[field] for values in available], axis=0)
            for field in ("precision", "recall", "F_beta", "area")
        }
        for index, threshold in enumerate(thresholds):
            curve_rows.append(
                {
                    "scope": "dataset_macro",
                    "dataset": "ALL",
                    "method": method,
                    "threshold": float(threshold),
                    **{field: float(value[index]) for field, value in macro.items()},
                }
            )
        best_index = int(np.argmax(macro["F_beta"]))
        best_global[method] = float(thresholds[best_index])
        for dataset in datasets:
            values = dataset_arrays[(dataset, method)]["F_beta"]
            best_dataset[(dataset, method)] = float(thresholds[int(np.argmax(values))])
    return curve_rows, best_global, best_dataset


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--variant-root must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip():
        raise ValueError("variant name is empty")
    return name.strip(), _resolve(path)


def _core_descriptor(row: dict, field: str) -> dict:
    return {"kind": "dict", "path": row["score_path"], "field": field}


def _build_tasks(args, cfg) -> tuple[list[dict], list[str], dict[str, dict], tuple[str, ...]]:
    mbsp_root = _resolve(args.mbsp_root or cfg.MBSP_OUT_ROOT)
    rows, main_map = _manifest_map(mbsp_root / "manifest_test.jsonl")
    rows = [row for row in rows if row["dataset"] in (args.dataset or DATASETS)]
    rows = _balanced_subset(rows, args.max_samples)
    if not rows:
        raise RuntimeError("no MBSP samples selected")
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    if args.max_samples < 0 and not args.dataset:
        if len(rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
            raise RuntimeError(
                f"formal evaluation requires 6473 complete samples, got {len(rows)} {dict(counts)}"
            )
    datasets = tuple(dataset for dataset in DATASETS if counts[dataset])

    variant_maps = [("MBSP", mbsp_root, main_map)]
    for raw in args.variant_root:
        name, root = _parse_variant(raw)
        _, mapping = _manifest_map(root / "manifest_test.jsonl")
        variant_maps.append((name, root, mapping))

    method_metadata = {}
    methods = []
    score_types = args.score_types
    for variant_name, root, _ in variant_maps:
        run_config_path = root / "run_config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
        for score_type in score_types:
            method = (
                SCORE_FIELDS[score_type]
                if variant_name == "MBSP"
                else f"{variant_name}/{SCORE_FIELDS[score_type]}"
            )
            methods.append(method)
            method_metadata[method] = {
                "family": "MBSP",
                "variant": variant_name,
                "score_type": score_type,
                "root": str(root),
                "settings": run_config.get("settings", {}),
            }

    core_maps = {}
    bgnull_map = {}
    if not args.no_baselines:
        r1_core_root = _resolve(args.r1_core_root)
        for name in ("feature_only_reconstruction", "soft_similarity"):
            _, core_maps[name] = _manifest_map(r1_core_root / name / "manifest.jsonl")
        _, bgnull_map = _manifest_map(_resolve(args.bgnull_manifest))
        baseline_methods = (
            "Current Full R1",
            "Feature-only R1",
            "Soft Background Similarity",
            "RC-NN",
        )
        methods.extend(baseline_methods)
        for method in baseline_methods:
            method_metadata[method] = {"family": "baseline"}

    tasks = []
    for row in rows:
        key = (row["dataset"], row["stem"])
        descriptors = {}
        for variant_name, _, mapping in variant_maps:
            if key not in mapping:
                raise KeyError(f"variant {variant_name} missing {key}")
            cache_path = mapping[key]["cache_path"]
            if not Path(cache_path).is_file():
                raise FileNotFoundError(cache_path)
            for score_type in score_types:
                method = (
                    SCORE_FIELDS[score_type]
                    if variant_name == "MBSP"
                    else f"{variant_name}/{SCORE_FIELDS[score_type]}"
                )
                descriptors[method] = {"kind": "dict", "path": cache_path, "field": score_type}
        if not args.no_baselines:
            source_path = row.get("source_dabe_path")
            if source_path is None:
                source_path = torch_load(row["cache_path"], map_location="cpu")["source_dabe_path"]
            descriptors["Current Full R1"] = {
                "kind": "dict",
                "path": source_path,
                "field": "residual_pass1_37",
            }
            descriptors["Feature-only R1"] = _core_descriptor(
                core_maps["feature_only_reconstruction"][key], "normalized_score_37"
            )
            descriptors["Soft Background Similarity"] = _core_descriptor(
                core_maps["soft_similarity"][key], "normalized_score_37"
            )
            descriptors["RC-NN"] = {
                "kind": "dict",
                "path": bgnull_map[key]["cache_path"],
                "field": "rc_nn_minmax_37",
            }
        tasks.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "gt_path": row["gt_path"],
                "image_path": row["image_path"],
                "mbsp_cache_path": row["cache_path"],
                "descriptors": descriptors,
            }
        )
    return tasks, methods, method_metadata, datasets


def _table(rows: list[dict], fields: tuple[str, ...]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" if field in {"method", "dataset"} else "---:" for field in fields) + "|"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, divider, *body))


def _panel(image: Image.Image, title: str, size: int = 160, nearest: bool = False) -> Image.Image:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    content = image.resize((size, size), mode).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(content, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def _gray(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().float().squeeze().numpy()
    minimum, maximum = float(array.min()), float(array.max())
    if maximum > minimum:
        array = (array - minimum) / (maximum - minimum)
    else:
        array = np.zeros_like(array)
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")


def _color_labels(value: np.ndarray, count: int) -> Image.Image:
    palette = np.asarray(
        [[45, 45, 45], [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75], [227, 119, 194], [188, 189, 34]],
        dtype=np.uint8,
    )
    output = np.zeros((*value.shape, 3), dtype=np.uint8)
    output[:] = palette[0]
    for index in range(count):
        output[value == index] = palette[(index % (len(palette) - 1)) + 1]
    return Image.fromarray(output, mode="RGB")


def _save_visual(task: dict, threshold: float, r1_threshold: float, output: Path) -> None:
    payload = torch_load(task["mbsp_cache_path"], map_location="cpu")
    with Image.open(task["image_path"]) as source:
        rgb = source.convert("RGB").copy()
    gt = _load_gt(task["gt_path"])
    mbsp = payload["relative_raw"].float()
    r1 = _load_score(task["descriptors"]["Current Full R1"])
    feature = _load_score(task["descriptors"]["Feature-only R1"])
    soft = _load_score(task["descriptors"]["Soft Background Similarity"])
    background = torch.zeros(37 * 37, dtype=torch.long) - 1
    background[payload["background_indices"].long()] = payload["cluster_assignment_background"].long()
    background = background.reshape(37, 37).numpy()
    best = payload["best_subspace_index"].squeeze().numpy()
    subspace_count = int(payload["num_effective_subspaces"])

    r1_mask = r1 > r1_threshold
    mbsp_mask = mbsp > threshold
    target37 = torch_f.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze(0) > 0.5
    transition = np.zeros((37, 37, 3), dtype=np.uint8) + 80
    common = (r1_mask & mbsp_mask & target37).squeeze().numpy()
    corrected = (r1_mask & ~mbsp_mask & ~target37).squeeze().numpy()
    lost = (r1_mask & ~mbsp_mask & target37).squeeze().numpy()
    transition[common] = [255, 255, 255]
    transition[corrected] = [0, 200, 0]
    transition[lost] = [220, 0, 0]

    panels = [
        _panel(rgb, "RGB"),
        _panel(_gray(gt), "GT", nearest=True),
        _panel(_gray((torch.from_numpy(background) >= 0).float()), "Full BC", nearest=True),
        _panel(_color_labels(background, subspace_count), "BG clusters", nearest=True),
        _panel(_color_labels(best, subspace_count), "Best subspace", nearest=True),
        _panel(_gray(mbsp), "MBSP rel raw"),
        _panel(_gray(payload["relative_minmax"]), "MBSP rel minmax"),
        _panel(_gray((mbsp > 0.5).float()), "MBSP @0.5", nearest=True),
        _panel(_gray(mbsp_mask.float()), f"MBSP @{threshold:.2f}", nearest=True),
        _panel(_gray(feature), "Feature-only R1"),
        _panel(_gray(soft), "Soft BG similarity"),
        _panel(Image.fromarray(transition, mode="RGB"), "Error transition", nearest=True),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    detail = {
        "dataset": task["dataset"],
        "stem": task["stem"],
        "cluster_sizes": payload["cluster_sizes"].tolist(),
        "selected_ranks": payload["selected_ranks"].tolist(),
        "representative_background_indices": payload["representative_background_indices"].tolist(),
        "energy_curves": [],
        "fallback_flags": payload["fallback_flags"],
    }
    for singular in payload["singular_values"]:
        energy = singular.float().square()
        curve = torch.cumsum(energy, dim=0) / (energy.sum() + 1e-8)
        detail["energy_curves"].append(curve.tolist())
    output.with_suffix(".json").write_text(json.dumps(detail, indent=2), encoding="utf-8")


def _report(
    output_dir: Path,
    summary_rows: list[dict],
    per_dataset: list[dict],
    best_dataset: dict,
    metadata: dict,
    full: bool,
) -> None:
    mbsp_method = "mbsp_rel_raw"
    by_method = {row["method"]: row for row in summary_rows}
    conclusion = "INCONCLUSIVE"
    reasons = []
    if mbsp_method in by_method and "Current Full R1" in by_method:
        candidate, baseline = by_method[mbsp_method], by_method["Current Full R1"]
        ap_delta = candidate["pixel_AP"] - baseline["pixel_AP"]
        fw_delta = candidate["selected_F_beta_w"] - baseline["selected_F_beta_w"]
        if ap_delta < -0.01 or fw_delta < -0.01:
            conclusion = "C：停止该路线"
        elif abs(ap_delta) <= 0.005 and abs(fw_delta) <= 0.005:
            conclusion = "B：性能接近，方法学口径更严格"
        elif ap_delta >= 0 and fw_delta >= 0:
            conclusion = "A 候选：达到或超过当前 R1"
        reasons = [f"Pixel AP Δ={ap_delta:+.6f}", f"最佳统一阈值 Fβw Δ={fw_delta:+.6f}"]

    fields = (
        "method",
        "fixed_F_beta_w",
        "selected_F_beta_w",
        "selected_S_m",
        "selected_E_mean",
        "selected_MAE",
        "pixel_AP",
        "pixel_AUROC",
        "best_global_threshold",
        "max_F_beta",
        "selected_Precision",
        "selected_Recall",
        "selected_Area",
    )
    lines = [
        "# MBSP PCA 最终报告",
        "",
        f"- 评估范围：{'完整 6473 张四数据集正式评测' if full else '非完整样本，仅用于流程核验，不形成正式结论'}。",
        "- 全部响应严格按 37→68→原始 GT 尺寸双线性上采样，`align_corners=False`。",
        f"- 最佳统一阈值由{'四数据集' if full else '当前可用数据集'}的图像级 Fβ 宏平均选择；它仅用于离线机制分析。",
        "- Pixel AP / AUROC 为先逐图计算、再按数据集宏平均的阈值无关指标。",
        "- 生成阶段未读取 GT；未执行训练或特征提取。",
        *(
            [
                "- 阈值边界命中："
                + ", ".join(
                    f"{method}={boundary}" for method, boundary in metadata["threshold_boundary_hits"].items()
                )
                + "；对应 Raw 分数应追加更宽阈值区间诊断。"
            ]
            if metadata.get("threshold_boundary_hits")
            else []
        ),
        "",
        "## 总表",
        "",
        _table(summary_rows, fields),
        "",
        "## 判断",
        "",
        f"- 当前分级：{conclusion if full else '需完成 6473 张后判定'}。",
        *([f"- {reason}" for reason in reasons] if full else []),
        "",
        "## 每数据集最佳阈值（仅诊断）",
        "",
        _table(
            [
                {"dataset": dataset, "method": method, "threshold": threshold}
                for (dataset, method), threshold in best_dataset.items()
            ],
            ("dataset", "method", "threshold"),
        ),
        "",
        "## 复现信息",
        "",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "```",
    ]
    (output_dir / "MBSP_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args) -> None:
    cfg = load_config(_resolve(args.config))
    if args.mbsp_root is None:
        args.mbsp_root = getattr(cfg, "MBSP_OUT_ROOT", str(DEFAULT_MBSP_ROOT))
    if args.out_dir is None:
        args.out_dir = getattr(cfg, "MBSP_EVAL_ROOT", str(DEFAULT_EVAL_ROOT))
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError(f"evaluation output must stay outside the code repository: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, methods, method_metadata, datasets = _build_tasks(args, cfg)
    if args.continuous_only:
        started = time.perf_counter()
        results = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.torch_threads,),
        ) as pool:
            for index, result in enumerate(
                pool.map(_continuous_pass_one, tasks, chunksize=1), 1
            ):
                results.append(result)
                if index % 20 == 0 or index == len(tasks):
                    print(f"MBSP continuous: {index}/{len(tasks)}", flush=True)
        failures = [result for result in results if "error" in result]
        write_json(output_dir / "evaluation_failures.json", failures)
        if failures:
            raise RuntimeError(f"continuous evaluation failed for {len(failures)} samples")
        per_image = [row for result in results for row in result["rows"]]
        summary = _aggregate_continuous(per_image, datasets, methods)
        _write_csv(
            output_dir / "per_image_continuous_metrics.csv",
            per_image,
            ("dataset", "stem", "method", "pixel_AP", "pixel_AUROC"),
        )
        _write_csv(
            output_dir / "continuous_metrics.csv",
            summary,
            ("scope", "dataset", "method", "num_samples", "pixel_AP", "pixel_AUROC"),
        )
        full = len(tasks) == EXPECTED_TOTAL and all(
            sum(task["dataset"] == dataset for task in tasks) == count
            for dataset, count in EXPECTED_COUNTS.items()
        )
        metadata = {
            "version": VERSION,
            "config": str(_resolve(args.config)),
            "mbsp_root": str(_resolve(args.mbsp_root)),
            "out_dir": str(output_dir),
            "datasets": datasets,
            "num_samples": len(tasks),
            "methods": methods,
            "continuous_only": True,
            "continuous_protocol": (
                "per-image native-size raw AP/AUROC; dataset and image macro; "
                "37->68->native bilinear"
            ),
            "threshold_scan_used": False,
            "hard_metrics_used": False,
            "resize": "37->68->original, bilinear, align_corners=False",
            "wall_seconds": time.perf_counter() - started,
            "full_formal_evaluation": full,
            "training_used": False,
        }
        write_json(output_dir / "evaluation_metadata.json", metadata)
        print(json.dumps({"summary": summary, "metadata": metadata}, indent=2), flush=True)
        return
    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + args.threshold_step * 0.5,
        args.threshold_step,
        dtype=np.float64,
    )
    if thresholds.size < 2 or bool(np.diff(thresholds).min() <= 0):
        raise ValueError("invalid threshold range")
    for task in tasks:
        task["thresholds"] = thresholds.tolist()

    started = time.perf_counter()
    first_results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_first_pass_one, tasks, chunksize=1), 1):
            first_results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"MBSP evaluate pass 1: {index}/{len(tasks)}", flush=True)
    failures = [result for result in first_results if "error" in result]
    write_json(output_dir / "evaluation_failures.json", failures)
    if failures:
        raise RuntimeError(f"evaluation failed for {len(failures)} samples")

    per_sample_first = [row for result in first_results for row in result["rows"]]
    curve_rows, best_global, best_dataset = _aggregate_curves(
        first_results, thresholds, datasets, methods
    )
    for task in tasks:
        task["best_thresholds"] = best_global

    selected_results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_selected_pass_one, tasks, chunksize=1), 1):
            selected_results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"MBSP evaluate pass 2: {index}/{len(tasks)}", flush=True)
    failures = [result for result in selected_results if "error" in result]
    if failures:
        write_json(output_dir / "selected_threshold_failures.json", failures)
        raise RuntimeError(f"selected-threshold evaluation failed for {len(failures)} samples")
    per_sample_selected = [row for result in selected_results for row in result["rows"]]

    selected_map = {
        (row["dataset"], row["stem"], row["method"]): row for row in per_sample_selected
    }
    per_sample = []
    for row in per_sample_first:
        selected = selected_map[(row["dataset"], row["stem"], row["method"])]
        per_sample.append({**row, **selected})

    fixed_aggregate = _aggregate_rows(per_sample, "fixed_", datasets, methods)
    soft_aggregate = _aggregate_rows(per_sample, "soft_", datasets, methods)
    selected_aggregate = _aggregate_rows(per_sample, "selected_", datasets, methods)
    rank_aggregate = _aggregate_rank(per_sample, datasets, methods)
    macro_fixed = {row["method"]: row for row in fixed_aggregate if row["scope"] == "dataset_macro"}
    macro_soft = {row["method"]: row for row in soft_aggregate if row["scope"] == "dataset_macro"}
    macro_selected = {row["method"]: row for row in selected_aggregate if row["scope"] == "dataset_macro"}
    macro_rank = {row["method"]: row for row in rank_aggregate if row["scope"] == "dataset_macro"}
    macro_curve = {
        method: [
            row
            for row in curve_rows
            if row["scope"] == "dataset_macro" and row["method"] == method
        ]
        for method in methods
    }
    summary_rows = []
    for method in methods:
        best_curve_row = next(
            row for row in macro_curve[method] if abs(row["threshold"] - best_global[method]) < 1e-9
        )
        summary_rows.append(
            {
                "method": method,
                **{field: value for field, value in macro_fixed[method].items() if field.startswith("fixed_")},
                **{field: value for field, value in macro_soft[method].items() if field.startswith("soft_")},
                **{field: value for field, value in macro_selected[method].items() if field.startswith("selected_")},
                **{field: macro_rank[method][field] for field in ("pixel_AP", "pixel_AUROC", "raw_MAE", "raw_clipped_fraction")},
                "best_global_threshold": best_global[method],
                "max_F_beta": best_curve_row["F_beta"],
                "num_samples": len(tasks),
            }
        )

    per_dataset_metrics = {
        "fixed_0_5": fixed_aggregate,
        "continuous_official": soft_aggregate,
        "selected_global_threshold": selected_aggregate,
        "ranking": rank_aggregate,
        "best_global_thresholds": best_global,
        "best_per_dataset_thresholds": {
            f"{dataset}|{method}": threshold
            for (dataset, method), threshold in best_dataset.items()
        },
    }
    write_json(output_dir / "per_dataset_metrics.json", per_dataset_metrics)
    _write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    _write_csv(
        output_dir / "threshold_sweep.csv",
        curve_rows,
        ("scope", "dataset", "method", "threshold", "precision", "recall", "F_beta", "area"),
    )
    _write_csv(output_dir / "per_sample_metrics.csv", per_sample, list(per_sample[0]))

    ablation_rows = []
    for row in summary_rows:
        metadata = method_metadata[row["method"]]
        if metadata["family"] != "MBSP":
            continue
        settings = metadata.get("settings", {})
        ablation_rows.append(
            {
                **row,
                "variant": metadata["variant"],
                "score_type": metadata["score_type"],
                "num_subspaces": settings.get("num_subspaces"),
                "pca_energy": settings.get("pca_energy"),
                "pca_max_rank": settings.get("pca_max_rank"),
                "pca_min_rank": settings.get("pca_min_rank"),
                "min_cluster_size": settings.get("min_cluster_size"),
            }
        )
    _write_csv(output_dir / "ablation_summary.csv", ablation_rows, list(ablation_rows[0]))

    full = len(tasks) == EXPECTED_TOTAL and all(
        sum(task["dataset"] == dataset for task in tasks) == count
        for dataset, count in EXPECTED_COUNTS.items()
    )
    metadata = {
        "version": VERSION,
        "config": str(_resolve(args.config)),
        "mbsp_root": str(_resolve(args.mbsp_root)),
        "out_dir": str(output_dir),
        "datasets": datasets,
        "num_samples": len(tasks),
        "methods": methods,
        "threshold_protocol": (
            "four-dataset image-level F_beta dataset-macro"
            if full
            else "available-dataset image-level F_beta dataset-macro (smoke only)"
        ),
        "thresholds": thresholds.tolist(),
        "threshold_boundary_hits": {
            method: (
                "lower"
                if abs(threshold - float(thresholds[0])) < 1e-12
                else "upper"
            )
            for method, threshold in best_global.items()
            if abs(threshold - float(thresholds[0])) < 1e-12
            or abs(threshold - float(thresholds[-1])) < 1e-12
        },
        "fixed_threshold": 0.5,
        "resize": "37->68->original, bilinear, align_corners=False",
        "wall_seconds": time.perf_counter() - started,
        "full_formal_evaluation": full,
        "training_used": False,
    }
    write_json(output_dir / "evaluation_metadata.json", metadata)
    _report(output_dir, summary_rows, per_dataset_metrics["selected_global_threshold"], best_dataset, metadata, full)

    if not args.skip_visuals and not args.no_baselines and "mbsp_rel_raw" in methods:
        first_map = {
            (row["dataset"], row["stem"], row["method"]): row for row in per_sample_first
        }
        ranked = sorted(
            tasks,
            key=lambda task: first_map[(task["dataset"], task["stem"], "mbsp_rel_raw")]["fixed_F_beta_w"]
            - first_map[(task["dataset"], task["stem"], "Current Full R1")]["fixed_F_beta_w"],
        )
        selected = [
            ("failure", task) for task in ranked[: args.visual_count]
        ] + [("success", task) for task in ranked[-args.visual_count :]]
        for category, task in selected:
            _save_visual(
                task,
                best_global["mbsp_rel_raw"],
                best_global["Current Full R1"],
                output_dir / "visuals" / category / f"{task['dataset']}__{task['stem']}.png",
            )
    print(json.dumps({"summary": summary_rows, "metadata": metadata}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mbsp-root", "--mbsp_root", dest="mbsp_root")
    parser.add_argument("--variant-root", "--variant_root", dest="variant_root", action="append", default=[])
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir")
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    parser.add_argument(
        "--score-types",
        "--score_types",
        dest="score_types",
        nargs="+",
        choices=tuple(SCORE_FIELDS),
        default=list(SCORE_FIELDS),
    )
    parser.add_argument("--threshold-start", "--threshold_start", dest="threshold_start", type=float, default=0.0)
    parser.add_argument("--threshold-end", "--threshold_end", dest="threshold_end", type=float, default=1.0)
    parser.add_argument("--threshold-step", "--threshold_step", dest="threshold_step", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--r1-core-root", "--r1_core_root", dest="r1_core_root", default=str(DEFAULT_R1_CORE_ROOT))
    parser.add_argument("--bgnull-manifest", "--bgnull_manifest", dest="bgnull_manifest", default=str(DEFAULT_BGNULL_MANIFEST))
    parser.add_argument("--no-baselines", "--no_baselines", dest="no_baselines", action="store_true")
    parser.add_argument("--skip-visuals", "--skip_visuals", dest="skip_visuals", action="store_true")
    parser.add_argument("--visual-count", "--visual_count", dest="visual_count", type=int, default=5)
    parser.add_argument(
        "--continuous-only",
        "--continuous_only",
        dest="continuous_only",
        action="store_true",
        help="evaluate native-size AP/AUROC only; skip all thresholds and hard metrics",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
