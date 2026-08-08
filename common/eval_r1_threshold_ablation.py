#!/usr/bin/env python3
"""Formal fixed/adaptive threshold ablation for cached Full-R1 responses.

The script never extracts DINO features, rebuilds R1, or trains a model.  It
only reads production ``residual_pass1_37``, estimates a scalar threshold on
that 37x37 map, reuses the formal 37->68->GT bilinear resize, and applies a
strict ``score > threshold`` comparison at original GT resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu, threshold_triangle

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_rank_calibration import average_percentile_rank  # noqa: E402
from common.eval_dabe_background_null import _load_gt  # noqa: E402
from common.eval_dabe_rank_calibration import FastCODContext, _resize_native  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load, write_json  # noqa: E402


VERSION = "dabe_r1_threshold_ablation_v1"
SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5.py"
DEFAULT_R1_ROOT = (
    REPO_ROOT / "datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8"
)
DEFAULT_OUT_DIR = REPO_ROOT / "workdir/dabe_r1_threshold_ablation_v1"
DEFAULT_FIXED_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
DEFAULT_ADAPTIVE_METHODS = ("mean", "otsu", "triangle")
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
DATASET_ALIASES = {
    "CHAMELEON": "CHAMELEON",
    "CAMO": "TE-CAMO",
    "TE-CAMO": "TE-CAMO",
    "COD10K": "TE-COD10K",
    "TE-COD10K": "TE-COD10K",
    "NC4K": "NC4K",
}
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
BASELINE_EXACT = {
    "hard_S_m": 0.7321127747158344,
    "hard_F_beta_w": 0.6242804301164221,
    "hard_E_mean": 0.8279479346485145,
    "hard_MAE": 0.08046898620831737,
    "hard_Precision": 0.7069664740758147,
    "hard_Recall": 0.7124157409732799,
    "hard_Area": 0.1260482386630983,
}
BASELINE_TOLERANCE = 5e-5
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
    "hard_Components",
)
PER_IMAGE_FIELDS = (
    "dataset",
    "stem",
    "method",
    "threshold",
    "threshold_source",
    "fallback_reason",
    "score_min_37",
    "score_max_37",
    *METRIC_FIELDS,
    "image_path",
    "gt_path",
    "source_cache_path",
    "mask_path",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fixed_method_name(value: float) -> str:
    return f"fixed_{int(round(float(value) * 100)):03d}"


def method_label(method: str) -> str:
    if method.startswith("fixed_"):
        return f"Fixed {int(method.split('_')[1]) / 100:.2f}"
    return {"mean": "Image Mean", "otsu": "Otsu", "triangle": "Triangle"}[method]


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_manifest(r1_root: str | Path) -> Path:
    path = Path(r1_root).resolve()
    if path.is_file():
        return path
    candidate = path / "manifest_test.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"R1 manifest not found under {path}")


def _selected_datasets(values: list[str]) -> tuple[str, ...]:
    selected = []
    for value in values:
        key = value.strip().upper()
        if key not in DATASET_ALIASES:
            raise ValueError(f"unsupported dataset: {value}")
        canonical = DATASET_ALIASES[key]
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("datasets must not be empty")
    return tuple(selected)


def _manifest_rows(args, require_full: bool = True) -> list[dict]:
    manifest_path = _resolve_manifest(args.r1_root)
    rows = read_jsonl(manifest_path)
    selected = set(_selected_datasets(args.datasets))
    rows = [row for row in rows if row.get("dataset") in selected]
    seen = set()
    counts = defaultdict(int)
    for line, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path", "image_path", "gt_path"):
            if field not in row:
                raise KeyError(f"{field} missing at filtered manifest row {line}")
        key = (row["dataset"], row["stem"])
        if key in seen:
            raise RuntimeError(f"duplicate R1 key: {key}")
        seen.add(key)
        for field in ("cache_path", "image_path", "gt_path"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        counts[row["dataset"]] += 1
    if require_full:
        expected = {name: EXPECTED_COUNTS[name] for name in _selected_datasets(args.datasets)}
        if dict(counts) != expected:
            raise RuntimeError(f"dataset count mismatch: {dict(counts)} != {expected}")
        if tuple(_selected_datasets(args.datasets)) == DATASETS and len(rows) != EXPECTED_TOTAL:
            raise RuntimeError(f"expected {EXPECTED_TOTAL} images, got {len(rows)}")
    return rows


def load_normalized_r1_37(path: str | Path) -> torch.Tensor:
    path = Path(path)
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"R1 payload must be dict: {path}")
    score = payload.get("residual_pass1_37")
    if not torch.is_tensor(score):
        raise KeyError(f"residual_pass1_37 missing: {path}")
    score = score.detach().cpu().float()
    if tuple(score.shape) == (37, 37):
        score = score.unsqueeze(0)
    if tuple(score.shape) != (1, 37, 37):
        raise ValueError(f"residual_pass1_37 must be [1,37,37], got {tuple(score.shape)}: {path}")
    if not bool(torch.isfinite(score).all()):
        raise ValueError(f"R1 score contains NaN/Inf: {path}")
    if float(score.min()) < -1e-6 or float(score.max()) > 1.0 + 1e-6:
        raise ValueError(f"normalized R1 score is outside [0,1]: {path}")
    return score.clamp(0.0, 1.0).contiguous()


def upsample_score_to_gt(score_37: torch.Tensor, gt_height: int, gt_width: int) -> torch.Tensor:
    """Reuse the formal two-stage bilinear interpolation verbatim."""
    return _resize_native(score_37, (int(gt_height), int(gt_width)))


def fixed_threshold(score_37: torch.Tensor, value: float) -> tuple[float, str]:
    del score_37
    return float(value), ""


def image_mean_threshold(score_37: torch.Tensor) -> tuple[float, str]:
    reason = "constant_score_range_below_1e-8_mean_uses_constant" if _constant_score(score_37) else ""
    return float(score_37.mean()), reason


def _constant_score(score_37: torch.Tensor) -> bool:
    return float(score_37.max() - score_37.min()) < 1e-8


def otsu_threshold_value(score_37: torch.Tensor, nbins: int = 256) -> tuple[float, str]:
    if _constant_score(score_37):
        return 0.5, "constant_score_range_below_1e-8"
    value = threshold_otsu(score_37.squeeze(0).numpy(), nbins=int(nbins))
    return float(value), ""


def triangle_threshold_value(score_37: torch.Tensor, nbins: int = 256) -> tuple[float, str]:
    if _constant_score(score_37):
        return 0.5, "constant_score_range_below_1e-8"
    value = threshold_triangle(score_37.squeeze(0).numpy(), nbins=int(nbins))
    return float(value), ""


def estimate_threshold(method: str, score_37: torch.Tensor, nbins: int) -> tuple[float, str]:
    if method.startswith("fixed_"):
        return fixed_threshold(score_37, int(method.split("_")[1]) / 100.0)
    if method == "mean":
        return image_mean_threshold(score_37)
    if method == "otsu":
        return otsu_threshold_value(score_37, nbins)
    if method == "triangle":
        return triangle_threshold_value(score_37, nbins)
    raise ValueError(f"unsupported threshold method: {method}")


def binarize_strict(score_full: torch.Tensor, threshold: float) -> torch.Tensor:
    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"invalid threshold: {value}")
    return score_full > value


def _mask_path(out_dir: Path, method: str, dataset: str, stem: str) -> Path:
    return out_dir / "masks" / method / dataset / f"{stem}.png"


def _atomic_save_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    array = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
    Image.fromarray(array, mode="L").save(temporary)
    os.replace(temporary, path)


def _load_existing_mask(path: Path, shape: tuple[int, int]) -> torch.Tensor | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if tuple(array.shape) != tuple(shape) or not np.all((array == 0) | (array == 255)):
            return None
        return torch.from_numpy((array > 127).copy()).unsqueeze(0)
    except OSError:
        return None


def _hard_metric_row(
    metric: dict,
    method: str,
    threshold: float,
    fallback_reason: str,
    score_37: torch.Tensor,
    source_path: str,
    mask_path: Path,
    dataset: str,
    stem: str,
) -> dict:
    return {
        "dataset": dataset,
        "stem": stem,
        "method": method,
        "threshold": float(threshold),
        "threshold_source": "fixed" if method.startswith("fixed_") else "image_37x37",
        "fallback_reason": fallback_reason,
        "score_min_37": float(score_37.min()),
        "score_max_37": float(score_37.max()),
        "hard_S_m": metric["S_m"],
        "hard_F_beta_w": metric["F_beta_w"],
        "hard_F_beta_mean": metric["F_beta_mean"],
        "hard_E_mean": metric["E_mean"],
        "hard_MAE": metric["MAE"],
        "hard_IoU": metric["IoU"],
        "hard_Precision": metric["Precision"],
        "hard_Recall": metric["Recall"],
        "hard_Area": metric["Area"],
        "hard_Components": metric["Components"],
        "source_cache_path": source_path,
        "mask_path": str(mask_path),
    }


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _evaluate_one(task: dict) -> dict:
    try:
        row = task["row"]
        score_37 = load_normalized_r1_37(row["cache_path"])
        gt = _load_gt(row["gt_path"])
        gt_shape = tuple(gt.shape[-2:])
        score_full = upsample_score_to_gt(score_37, *gt_shape)
        context = FastCODContext(gt)
        prepared = []
        for method in task["methods"]:
            threshold, fallback_reason = estimate_threshold(method, score_37, task["nbins"])
            candidate = binarize_strict(score_full, threshold)
            mask_path = _mask_path(
                Path(task["out_dir"]), method, row["dataset"], row["stem"]
            )
            existing = _load_existing_mask(mask_path, gt_shape) if task["resume"] else None
            if existing is None:
                _atomic_save_mask(candidate, mask_path)
                mask = candidate
            else:
                if not torch.equal(existing, candidate):
                    raise RuntimeError(f"resumed mask differs from current rule: {mask_path}")
                mask = existing
            prepared.append((method, threshold, fallback_reason, mask, mask_path))
        metrics = context.evaluate_many(
            [(method, "hard", mask.float()) for method, _, _, mask, _ in prepared],
            0.5,
        )
        records = []
        for method, threshold, fallback_reason, mask, mask_path in prepared:
            del mask
            records.append(
                _hard_metric_row(
                    metrics[(method, "hard")],
                    method,
                    threshold,
                    fallback_reason,
                    score_37,
                    row["cache_path"],
                    mask_path,
                    row["dataset"],
                    row["stem"],
                )
            )
            records[-1]["image_path"] = row["image_path"]
            records[-1]["gt_path"] = row["gt_path"]
        return {"dataset": row["dataset"], "stem": row["stem"], "records": records}
    except Exception as exc:
        return {
            "dataset": task["row"].get("dataset", ""),
            "stem": task["row"].get("stem", ""),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _run_evaluation(args, methods: tuple[str, ...], log_name: str) -> list[dict]:
    rows = _manifest_rows(args)
    tasks = [
        {
            "row": row,
            "methods": methods,
            "nbins": args.nbins,
            "out_dir": str(Path(args.out_dir).resolve()),
            "resume": args.resume,
        }
        for row in rows
    ]
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_evaluate_one, tasks, chunksize=1), 1):
            if "error" in result:
                write_json(
                    Path(args.out_dir) / "logs" / f"{log_name}_failures.json",
                    [result],
                )
                raise RuntimeError(
                    f"{log_name} failed at {result['dataset']}/{result['stem']}: "
                    f"{result['error']}"
                )
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"[{_now()}] {log_name} {index}/{len(tasks)}", flush=True)
    write_json(Path(args.out_dir) / "logs" / f"{log_name}_failures.json", [])
    return [record for result in results for record in result["records"]]


def _aggregate(rows: list[dict], datasets: tuple[str, ...]) -> tuple[list[dict], list[dict], list[dict]]:
    by_dataset = []
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for dataset in datasets:
        for method in methods:
            selected = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not selected:
                continue
            by_dataset.append(
                {
                    "scope": "by_dataset",
                    "dataset": dataset,
                    "method": method,
                    **{field: float(np.mean([float(row[field]) for row in selected])) for field in METRIC_FIELDS},
                    "num_samples": len(selected),
                }
            )
    pooled = []
    macro = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        pooled.append(
            {
                "scope": "pooled_overall",
                "dataset": "ALL",
                "method": method,
                **{field: float(np.mean([float(row[field]) for row in selected])) for field in METRIC_FIELDS},
                "num_samples": len(selected),
            }
        )
        dataset_selected = [row for row in by_dataset if row["method"] == method]
        macro.append(
            {
                "scope": "macro_average",
                "dataset": "ALL",
                "method": method,
                **{field: float(np.mean([float(row[field]) for row in dataset_selected])) for field in METRIC_FIELDS},
                "num_samples": len(selected),
            }
        )
    return by_dataset, macro, pooled


def command_preflight(args) -> None:
    start = time.time()
    cfg = load_config(args.config)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise RuntimeError(f"expected BACKBONE_KEY=dinov1-s8, got {getattr(cfg, 'BACKBONE_KEY', None)}")
    rows = _manifest_rows(args)
    stats = defaultdict(list)
    failures = []
    for row in rows:
        try:
            payload = torch_load(row["cache_path"], map_location="cpu")
            if payload.get("dataset") != row["dataset"] or payload.get("stem") != row["stem"]:
                raise RuntimeError("cache key mismatch")
            score = load_normalized_r1_37(row["cache_path"])
            stats[row["dataset"]].append((float(score.min()), float(score.max())))
        except Exception as exc:
            failures.append(
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "cache_path": row["cache_path"],
                    "error": repr(exc),
                }
            )
    out_dir = Path(args.out_dir)
    write_json(out_dir / "logs/preflight_failures.json", failures)
    if failures:
        raise RuntimeError(f"preflight failed for {len(failures)} samples")
    summary = {
        "version": VERSION,
        "timestamp": _now(),
        "config": str(Path(args.config).resolve()),
        "r1_manifest": str(_resolve_manifest(args.r1_root)),
        "r1_field": "residual_pass1_37",
        "out_dir": str(out_dir.resolve()),
        "datasets": list(_selected_datasets(args.datasets)),
        "num_images": len(rows),
        "fixed_thresholds": list(args.fixed_thresholds),
        "adaptive_methods": list(args.adaptive_methods),
        "nbins": args.nbins,
        "strict_greater": True,
        "skimage_version": __import__("skimage").__version__,
        "score_min_global": min(value[0] for values in stats.values() for value in values),
        "score_max_global": max(value[1] for values in stats.values() for value in values),
        "elapsed_seconds": time.time() - start,
        "passed": True,
    }
    write_json(out_dir / "configs/threshold_ablation_config.json", summary)
    write_json(out_dir / "logs/preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_regression20(args) -> None:
    rows = _manifest_rows(args)
    selected = []
    for dataset in _selected_datasets(args.datasets):
        selected.extend([row for row in rows if row["dataset"] == dataset][:5])
    out_dir = Path(args.out_dir)
    sample_path = out_dir / "regression/regression_samples.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        "".join(f"{row['dataset']}/{row['stem']}\n" for row in selected), encoding="utf-8"
    )
    records = []
    for row in selected:
        score = load_normalized_r1_37(row["cache_path"])
        gt = _load_gt(row["gt_path"])
        gt_shape = tuple(gt.shape[-2:])
        # Reference path is the unchanged formal evaluator used by existing R1 audits.
        official_mask = _resize_native(score, gt_shape) > 0.5
        new_mask = binarize_strict(upsample_score_to_gt(score, *gt_shape), 0.5)
        mismatch = int((official_mask != new_mask).sum())
        records.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "exact_equal": mismatch == 0,
                "mismatched_pixels": mismatch,
                "num_pixels": int(new_mask.numel()),
                "source_cache_path": row["cache_path"],
                "reference": "common.eval_dabe_rank_calibration._resize_native + strict > 0.5",
            }
        )
    _write_csv(
        out_dir / "regression/mask_equality.csv",
        records,
        tuple(records[0]),
    )
    summary = {
        "timestamp": _now(),
        "num_samples": len(records),
        "all_exact": all(row["exact_equal"] for row in records),
        "total_mismatched_pixels": sum(row["mismatched_pixels"] for row in records),
        "passed": all(row["exact_equal"] for row in records),
    }
    write_json(out_dir / "regression/regression20_summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError(f"20-image mask regression failed: {summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _write_method_rows(out_dir: Path, rows: list[dict]) -> None:
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != EXPECTED_TOTAL:
            raise RuntimeError(f"{method} must have {EXPECTED_TOTAL} rows, got {len(selected)}")
        _write_csv(out_dir / "per_image" / f"{method}.csv", selected, PER_IMAGE_FIELDS)


def command_baseline(args) -> None:
    out_dir = Path(args.out_dir)
    rows = _run_evaluation(args, ("fixed_050",), "baseline_fixed050")
    _write_method_rows(out_dir, rows)
    by_dataset, macro, pooled = _aggregate(rows, _selected_datasets(args.datasets))
    macro_row = macro[0]
    errors = {
        field: abs(float(macro_row[field]) - expected) for field, expected in BASELINE_EXACT.items()
    }
    summary = {
        "version": VERSION,
        "timestamp": _now(),
        "num_images": len(rows),
        "macro": macro_row,
        "pooled": pooled[0],
        "expected_macro": BASELINE_EXACT,
        "absolute_errors": errors,
        "tolerance": BASELINE_TOLERANCE,
        "passed": max(errors.values()) <= BASELINE_TOLERANCE,
    }
    _write_csv(out_dir / "regression/baseline_by_dataset.csv", by_dataset, tuple(by_dataset[0]))
    write_json(out_dir / "regression/baseline_metrics.json", summary)
    if not summary["passed"]:
        raise RuntimeError(f"6473-image Fixed-0.5 regression failed: {errors}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _require_baseline_pass(out_dir: Path) -> None:
    path = out_dir / "regression/baseline_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"run baseline before other thresholds: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("passed") or int(data.get("num_images", 0)) != EXPECTED_TOTAL:
        raise RuntimeError(f"Fixed-0.5 baseline regression did not pass: {path}")


def command_evaluate(args) -> None:
    out_dir = Path(args.out_dir)
    _require_baseline_pass(out_dir)
    fixed_methods = tuple(
        fixed_method_name(value)
        for value in args.fixed_thresholds
        if abs(float(value) - 0.5) > 1e-12
    )
    adaptive = tuple(args.adaptive_methods)
    invalid = [method for method in adaptive if method not in DEFAULT_ADAPTIVE_METHODS]
    if invalid:
        raise ValueError(f"unsupported adaptive methods: {invalid}")
    methods = (*fixed_methods, *adaptive)
    if not methods:
        raise ValueError("no non-baseline threshold methods selected")
    rows = _run_evaluation(args, methods, "threshold_methods")
    _write_method_rows(out_dir, rows)
    print(f"Wrote {len(rows)} per-image method records under {out_dir / 'per_image'}")


def _load_all_per_image(args) -> tuple[list[dict], tuple[str, ...]]:
    out_dir = Path(args.out_dir)
    methods = tuple(fixed_method_name(value) for value in args.fixed_thresholds) + tuple(
        args.adaptive_methods
    )
    all_rows = []
    for method in methods:
        path = out_dir / "per_image" / f"{method}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _read_csv(path)
        if len(rows) != EXPECTED_TOTAL:
            raise RuntimeError(f"{path} must have {EXPECTED_TOTAL} rows, got {len(rows)}")
        all_rows.extend(rows)
    return all_rows, methods


def _pearson_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if x_np.size != y_np.size or x_np.size == 0:
        raise ValueError("correlation inputs must have equal non-zero size")
    pearson = 0.0 if x_np.std() < 1e-12 or y_np.std() < 1e-12 else float(np.corrcoef(x_np, y_np)[0, 1])
    x_rank = average_percentile_rank(torch.from_numpy(x_np)).numpy()
    y_rank = average_percentile_rank(torch.from_numpy(y_np)).numpy()
    spearman = 0.0 if x_rank.std() < 1e-12 or y_rank.std() < 1e-12 else float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def _threshold_statistics(rows: list[dict], adaptive: tuple[str, ...], datasets: tuple[str, ...]):
    threshold_rows = [row for row in rows if row["method"] in adaptive]
    summary_rows = []
    correlation_rows = []
    for method in adaptive:
        for dataset in (*datasets, "ALL"):
            selected = [
                row for row in threshold_rows
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            values = np.asarray([float(row["threshold"]) for row in selected], dtype=np.float64)
            areas = [float(row["hard_Area"]) for row in selected]
            pearson, spearman = _pearson_spearman(values.tolist(), areas)
            summary_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "count": values.size,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "q10": float(np.quantile(values, 0.10)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q90": float(np.quantile(values, 0.90)),
                    "max": float(values.max()),
                    "num_below_035": int(np.sum(values < 0.35)),
                    "num_035_to_065": int(np.sum((values >= 0.35) & (values <= 0.65))),
                    "num_above_065": int(np.sum(values > 0.65)),
                }
            )
            correlation_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "count": len(selected),
                    "threshold_area_pearson": pearson,
                    "threshold_area_spearman": spearman,
                }
            )
    fallback = [
        {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "method": row["method"],
            "score_min": row["score_min_37"],
            "score_max": row["score_max_37"],
            "fallback_reason": row["fallback_reason"],
            "image_path": row["image_path"],
            "source_cache_path": row["source_cache_path"],
        }
        for row in threshold_rows
        if row["fallback_reason"]
    ]
    return threshold_rows, summary_rows, correlation_rows, fallback


def _pairwise_vs_fixed(rows: list[dict], adaptive: tuple[str, ...], out_dir: Path):
    baseline = {
        (row["dataset"], row["stem"]): row for row in rows if row["method"] == "fixed_050"
    }
    delta_fields = (
        "hard_S_m",
        "hard_F_beta_w",
        "hard_E_mean",
        "hard_MAE",
        "hard_Precision",
        "hard_Recall",
        "hard_Area",
    )
    pairwise = []
    for row in rows:
        if row["method"] not in adaptive:
            continue
        key = (row["dataset"], row["stem"])
        reference = baseline[key]
        pairwise.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "method": row["method"],
                "threshold": float(row["threshold"]),
                **{
                    f"delta_{field}": float(row[field]) - float(reference[field])
                    for field in delta_fields
                },
                "candidate_mask_path": row["mask_path"],
                "fixed050_mask_path": reference["mask_path"],
                "source_cache_path": row["source_cache_path"],
            }
        )
    counts = []
    for method in adaptive:
        selected = [row for row in pairwise if row["method"] == method]
        counts.append(
            {
                "method": method,
                "num_images": len(selected),
                "fw_improved": sum(row["delta_hard_F_beta_w"] > 0 for row in selected),
                "fw_degraded": sum(row["delta_hard_F_beta_w"] < 0 for row in selected),
                "fw_equal": sum(row["delta_hard_F_beta_w"] == 0 for row in selected),
                "mae_improved": sum(row["delta_hard_MAE"] < 0 for row in selected),
                "mae_degraded": sum(row["delta_hard_MAE"] > 0 for row in selected),
                "mae_equal": sum(row["delta_hard_MAE"] == 0 for row in selected),
                "fw_up_and_mae_improved": sum(
                    row["delta_hard_F_beta_w"] > 0 and row["delta_hard_MAE"] < 0
                    for row in selected
                ),
                "fw_down_and_mae_degraded": sum(
                    row["delta_hard_F_beta_w"] < 0 and row["delta_hard_MAE"] > 0
                    for row in selected
                ),
            }
        )
        categories = {
            "fw_improved_top30": sorted(
                selected, key=lambda row: row["delta_hard_F_beta_w"], reverse=True
            )[:30],
            "fw_degraded_top30": sorted(
                selected, key=lambda row: row["delta_hard_F_beta_w"]
            )[:30],
            "area_expanded_top30": sorted(
                selected, key=lambda row: row["delta_hard_Area"], reverse=True
            )[:30],
            "area_shrunk_top30": sorted(
                selected, key=lambda row: row["delta_hard_Area"]
            )[:30],
        }
        for name, top_rows in categories.items():
            _write_csv(
                out_dir / "comparison" / f"{method}_{name}.csv",
                top_rows,
                tuple(top_rows[0]),
            )
    return pairwise, counts


def _comparison_fields() -> tuple[str, ...]:
    return ("scope", "dataset", "method", *METRIC_FIELDS, "num_samples")


def _table_rows(rows: list[dict]) -> str:
    header = (
        "| Threshold Rule | S_m | F_beta^w | F_beta^m | E_phi^m | MAE | "
        "Precision | Recall | Area |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {method_label(row['method'])} | {float(row['hard_S_m']):.4f} | "
        f"{float(row['hard_F_beta_w']):.4f} | {float(row['hard_F_beta_mean']):.4f} | "
        f"{float(row['hard_E_mean']):.4f} | {float(row['hard_MAE']):.4f} | "
        f"{float(row['hard_Precision']):.4f} | {float(row['hard_Recall']):.4f} | "
        f"{float(row['hard_Area']):.4f} |\n"
        for row in rows
    )
    return header + body


def _latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Threshold Rule & $S_m$ & $F_\beta^w$ & $F_\beta^m$ & $E_\phi^m$ & MAE & Precision & Recall & Area \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{method_label(row['method'])} & {float(row['hard_S_m']):.4f} & "
            f"{float(row['hard_F_beta_w']):.4f} & {float(row['hard_F_beta_mean']):.4f} & "
            f"{float(row['hard_E_mean']):.4f} & {float(row['hard_MAE']):.4f} & "
            f"{float(row['hard_Precision']):.4f} & {float(row['hard_Recall']):.4f} & "
            f"{float(row['hard_Area']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _generate_figures(
    out_dir: Path,
    by_dataset: list[dict],
    macro: list[dict],
    threshold_rows: list[dict],
    fixed_methods: tuple[str, ...],
    adaptive: tuple[str, ...],
    datasets: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed_values = [int(method.split("_")[1]) / 100.0 for method in fixed_methods]
    fixed_order = [method for _, method in sorted(zip(fixed_values, fixed_methods))]
    fixed_values = sorted(fixed_values)
    dataset_labels = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}

    def series(scope_rows, dataset, field):
        mapping = {
            row["method"]: float(row[field])
            for row in scope_rows
            if row["dataset"] == dataset and row["method"] in fixed_order
        }
        return [mapping[method] for method in fixed_order]

    curve_specs = (
        ("hard_F_beta_w", "Weighted F-measure", "fixed_threshold_Fw_curve.pdf"),
        ("hard_MAE", "MAE", "fixed_threshold_MAE_curve.pdf"),
        ("hard_Area", "Predicted foreground area", "fixed_threshold_area_curve.pdf"),
    )
    for field, ylabel, filename in curve_specs:
        fig, axis = plt.subplots(figsize=(6.4, 4.5))
        for dataset in datasets:
            axis.plot(
                fixed_values,
                series(by_dataset, dataset, field),
                marker="o",
                label=dataset_labels.get(dataset, dataset),
            )
        macro_mapping = {row["method"]: float(row[field]) for row in macro}
        axis.plot(
            fixed_values,
            [macro_mapping[method] for method in fixed_order],
            marker="o",
            linewidth=2.6,
            color="black",
            label="Macro Average",
        )
        axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set(xlabel="Threshold", ylabel=ylabel, xlim=(min(fixed_values), max(fixed_values)))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, bbox_inches="tight")
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    for dataset in datasets:
        label = dataset_labels.get(dataset, dataset)
        axis.plot(
            fixed_values,
            series(by_dataset, dataset, "hard_Precision"),
            marker="o",
            linewidth=1.1,
            label=f"{label} Precision",
        )
        axis.plot(
            fixed_values,
            series(by_dataset, dataset, "hard_Recall"),
            marker="s",
            linewidth=1.1,
            linestyle="--",
            label=f"{label} Recall",
        )
    macro_mapping = {row["method"]: row for row in macro}
    axis.plot(
        fixed_values,
        [float(macro_mapping[method]["hard_Precision"]) for method in fixed_order],
        marker="o",
        linewidth=2.5,
        color="black",
        label="Macro Precision",
    )
    axis.plot(
        fixed_values,
        [float(macro_mapping[method]["hard_Recall"]) for method in fixed_order],
        marker="s",
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Macro Recall",
    )
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(xlabel="Threshold", ylabel="Score", xlim=(min(fixed_values), max(fixed_values)))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "fixed_threshold_precision_recall_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    for method in adaptive:
        fig, axis = plt.subplots(figsize=(6.4, 4.5))
        for dataset in datasets:
            values = [
                float(row["threshold"])
                for row in threshold_rows
                if row["method"] == method and row["dataset"] == dataset
            ]
            axis.hist(
                values,
                bins=50,
                range=(0.0, 1.0),
                histtype="step",
                linewidth=1.4,
                density=True,
                label=dataset_labels.get(dataset, dataset),
            )
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Fixed 0.5")
        axis.set(xlabel="Image-level threshold", ylabel="Density", xlim=(0.0, 1.0))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{method}_threshold_histogram.pdf", bbox_inches="tight")
        plt.close(fig)


def _heatmap(score: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    import matplotlib

    cmap = matplotlib.colormaps["inferno"]
    rgba = cmap(score.squeeze(0).numpy())
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _mask_panel(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").resize(size, Image.Resampling.NEAREST).convert("RGB")


def _visual_panel(source_row: dict, per_method: dict, size=(220, 220)) -> Image.Image:
    with Image.open(source_row["image_path"]) as image:
        image_panel = image.convert("RGB").resize(size)
    with Image.open(source_row["gt_path"]) as image:
        gt_panel = image.convert("L").resize(size, Image.Resampling.NEAREST).convert("RGB")
    score = load_normalized_r1_37(source_row["cache_path"])
    panels = [
        ("Image", image_panel),
        ("GT", gt_panel),
        ("R1 heatmap", _heatmap(score, size)),
    ]
    for method in ("fixed_050", "mean", "otsu", "triangle"):
        row = per_method[method]
        panels.append(
            (f"{method_label(method)}\nt={float(row['threshold']):.4f}", _mask_panel(row["mask_path"], size))
        )
    canvas = Image.new("RGB", (size[0] * len(panels), size[1] + 42), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * size[0]
        canvas.paste(panel, (x, 42))
        draw.multiline_text((x + 4, 4), label, fill="black", spacing=2)
    return canvas


def _generate_visualizations(
    args,
    all_rows: list[dict],
    pairwise: list[dict],
    adaptive: tuple[str, ...],
) -> None:
    source_rows = _manifest_rows(args)
    source_map = {(row["dataset"], row["stem"]): row for row in source_rows}
    record_map = {
        (row["dataset"], row["stem"], row["method"]): row for row in all_rows
    }
    out_dir = Path(args.out_dir)
    for method in adaptive:
        for dataset in _selected_datasets(args.datasets):
            selected = [row for row in pairwise if row["method"] == method and row["dataset"] == dataset]
            categories = {
                "ordinary": sorted(selected, key=lambda row: abs(row["delta_hard_F_beta_w"]))[:5],
                "improved": sorted(selected, key=lambda row: row["delta_hard_F_beta_w"], reverse=True)[:5],
                "degraded": sorted(selected, key=lambda row: row["delta_hard_F_beta_w"])[:5],
            }
            for category, chosen in categories.items():
                for rank, delta_row in enumerate(chosen, 1):
                    key = (dataset, delta_row["stem"])
                    per_method = {
                        name: record_map[(dataset, delta_row["stem"], name)]
                        for name in ("fixed_050", "mean", "otsu", "triangle")
                    }
                    panel = _visual_panel(source_map[key], per_method)
                    path = (
                        out_dir / "visualizations" / method / dataset / category
                        / f"{rank:02d}_{delta_row['stem']}.jpg"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    panel.save(path, quality=90)


def _adaptive_decisions(
    by_dataset: list[dict], macro: list[dict], pooled: list[dict], adaptive: tuple[str, ...]
) -> list[dict]:
    macro_map = {row["method"]: row for row in macro}
    pooled_map = {row["method"]: row for row in pooled}
    dataset_map = {(row["dataset"], row["method"]): row for row in by_dataset}
    base_macro = macro_map["fixed_050"]
    base_pooled = pooled_map["fixed_050"]
    decisions = []
    for method in adaptive:
        macro_row = macro_map[method]
        pooled_row = pooled_map[method]
        dataset_deltas = [
            float(dataset_map[(dataset, method)]["hard_F_beta_w"])
            - float(dataset_map[(dataset, "fixed_050")]["hard_F_beta_w"])
            for dataset in DATASETS
        ]
        area_ratio = float(macro_row["hard_Area"]) / float(base_macro["hard_Area"])
        checks = {
            "macro_fw_gain_ge_0003": float(macro_row["hard_F_beta_w"])
            - float(base_macro["hard_F_beta_w"])
            >= 0.003,
            "pooled_fw_gain_ge_0003": float(pooled_row["hard_F_beta_w"])
            - float(base_pooled["hard_F_beta_w"])
            >= 0.003,
            "macro_mae_not_worse": float(macro_row["hard_MAE"])
            <= float(base_macro["hard_MAE"]),
            "at_least_three_datasets_fw_not_lower": sum(delta >= 0.0 for delta in dataset_deltas)
            >= 3,
            "worst_dataset_fw_drop_within_0005": min(dataset_deltas) >= -0.005,
            "macro_area_expansion_within_25pct": area_ratio <= 1.25,
        }
        decisions.append(
            {
                "method": method,
                "macro_delta_fw": float(macro_row["hard_F_beta_w"])
                - float(base_macro["hard_F_beta_w"]),
                "pooled_delta_fw": float(pooled_row["hard_F_beta_w"])
                - float(base_pooled["hard_F_beta_w"]),
                "macro_delta_mae": float(macro_row["hard_MAE"])
                - float(base_macro["hard_MAE"]),
                "macro_area_ratio": area_ratio,
                "num_datasets_fw_not_lower": sum(delta >= 0.0 for delta in dataset_deltas),
                "worst_dataset_delta_fw": min(dataset_deltas),
                **checks,
                "qualifies_for_decoder_validation": all(checks.values()),
            }
        )
    return decisions


def _results_markdown(
    out_dir: Path,
    macro: list[dict],
    pooled: list[dict],
    by_dataset: list[dict],
    threshold_summary: list[dict],
    outcome_counts: list[dict],
    decisions: list[dict],
) -> str:
    macro_map = {row["method"]: row for row in macro}
    pooled_map = {row["method"]: row for row in pooled}
    baseline = macro_map["fixed_050"]
    fixed_focus = ("fixed_045", "fixed_050", "fixed_055")
    fixed_range = ("fixed_040", "fixed_045", "fixed_050", "fixed_055", "fixed_060")
    focus_lines = [
        f"- {method_label(method)}: Fβw={float(macro_map[method]['hard_F_beta_w']):.6f}, "
        f"MAE={float(macro_map[method]['hard_MAE']):.6f}, "
        f"P={float(macro_map[method]['hard_Precision']):.6f}, "
        f"R={float(macro_map[method]['hard_Recall']):.6f}, "
        f"Area={float(macro_map[method]['hard_Area']):.6f}"
        for method in fixed_focus
    ]
    fw_range_values = [float(macro_map[method]["hard_F_beta_w"]) for method in fixed_range]
    mae_range_values = [float(macro_map[method]["hard_MAE"]) for method in fixed_range]
    near_fw_values = [float(macro_map[method]["hard_F_beta_w"]) for method in fixed_focus]
    near_mae_values = [float(macro_map[method]["hard_MAE"]) for method in fixed_focus]
    near_stable = (max(near_fw_values) - min(near_fw_values) < 0.01) and (
        max(near_mae_values) - min(near_mae_values) < 0.005
    )
    count_map = {row["method"]: row for row in outcome_counts}
    stat_map = {
        row["method"]: row
        for row in threshold_summary
        if row["dataset"] == "ALL"
    }
    decision_map = {row["method"]: row for row in decisions}
    adaptive_lines = []
    for method in DEFAULT_ADAPTIVE_METHODS:
        row = macro_map[method]
        pooled_row = pooled_map[method]
        stats = stat_map[method]
        counts = count_map[method]
        decision = decision_map[method]
        adaptive_lines.append(
            f"### {method_label(method)}\n\n"
            f"- Macro: ΔFβw={float(row['hard_F_beta_w']) - float(baseline['hard_F_beta_w']):+.6f}, "
            f"ΔMAE={float(row['hard_MAE']) - float(baseline['hard_MAE']):+.6f}, "
            f"ΔP={float(row['hard_Precision']) - float(baseline['hard_Precision']):+.6f}, "
            f"ΔR={float(row['hard_Recall']) - float(baseline['hard_Recall']):+.6f}, "
            f"Area={float(row['hard_Area']):.6f}.\n"
            f"- Pooled: ΔFβw={float(pooled_row['hard_F_beta_w']) - float(pooled_map['fixed_050']['hard_F_beta_w']):+.6f}, "
            f"ΔMAE={float(pooled_row['hard_MAE']) - float(pooled_map['fixed_050']['hard_MAE']):+.6f}.\n"
            f"- Threshold: mean={float(stats['mean']):.6f}, q10={float(stats['q10']):.6f}, "
            f"median={float(stats['median']):.6f}, q90={float(stats['q90']):.6f}.\n"
            f"- Per-image: Fβw improve/degrade={counts['fw_improved']}/{counts['fw_degraded']}, "
            f"MAE improve/degrade={counts['mae_improved']}/{counts['mae_degraded']}.\n"
            f"- Decoder-candidate rule: {'PASS' if decision['qualifies_for_decoder_validation'] else 'FAIL'}."
        )
    qualified = [row for row in decisions if row["qualifies_for_decoder_validation"]]
    if qualified:
        chosen = max(qualified, key=lambda row: row["macro_delta_fw"])["method"]
        recommendation = f"Adaptive candidate: {method_label(chosen)}。停止在此处，等待后续 1×1 解码器确认。"
    else:
        recommendation = "Keep Fixed 0.5。没有自适应方法同时满足预设的后续 1×1 验证门槛。"
    git_hash = subprocess.run(
        ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "N/A"
    dataset_sections = []
    for dataset in DATASETS:
        dataset_sections.append(
            f"### {dataset}\n\n"
            + _table_rows([row for row in by_dataset if row["dataset"] == dataset])
        )
    return f"""# R1 固定阈值敏感性与图像自适应阈值实验

生成时间：{_now()}  
仓库 HEAD：`{git_hash}`  
正式缓存：`{DEFAULT_R1_ROOT / 'manifest_test.jsonl'}`  
输出目录：`{out_dir.resolve()}`

## Macro Average

{_table_rows(macro)}
## Pooled Overall

{_table_rows(pooled)}
## 四数据集逐项结果

{chr(10).join(dataset_sections)}
## 固定阈值稳定性

{chr(10).join(focus_lines)}

- 0.40–0.60 Macro Fβw 范围：[{min(fw_range_values):.6f}, {max(fw_range_values):.6f}]，跨度={max(fw_range_values) - min(fw_range_values):.6f}。
- 0.40–0.60 Macro MAE 范围：[{min(mae_range_values):.6f}, {max(mae_range_values):.6f}]，跨度={max(mae_range_values) - min(mae_range_values):.6f}。
- 0.45/0.50/0.55 邻域判定：{'0.5 位于平稳区间。' if near_stable else '该邻域波动不应表述为平稳区间。'}
- 阈值升高时应结合曲线核查 Precision 上升、Recall 与 Area 下降的单调趋势；固定阈值扫描仅作敏感性分析，不据测试集最优值修改正式阈值。

## 自适应阈值对比

{chr(10).join(adaptive_lines)}

## 最终建议

{recommendation}

该结论只覆盖 Full BC + Full R1 的二值化规则，不外推到 Feature-only 或训练版解码器。未启动任何训练或跨框架迁移。
"""


def command_report(args) -> None:
    out_dir = Path(args.out_dir)
    _require_baseline_pass(out_dir)
    all_rows, methods = _load_all_per_image(args)
    datasets = _selected_datasets(args.datasets)
    by_dataset, macro, pooled = _aggregate(all_rows, datasets)
    fields = _comparison_fields()
    _write_csv(out_dir / "comparison/metrics_by_dataset.csv", by_dataset, fields)
    _write_csv(out_dir / "comparison/metrics_macro_average.csv", macro, fields)
    _write_csv(out_dir / "comparison/metrics_pooled_overall.csv", pooled, fields)

    adaptive = tuple(args.adaptive_methods)
    threshold_rows, threshold_summary, correlations, fallback = _threshold_statistics(
        all_rows, adaptive, datasets
    )
    _write_csv(
        out_dir / "threshold_stats/thresholds_per_image.csv",
        threshold_rows,
        PER_IMAGE_FIELDS,
    )
    _write_csv(
        out_dir / "threshold_stats/threshold_summary_by_dataset.csv",
        threshold_summary,
        tuple(threshold_summary[0]),
    )
    _write_csv(
        out_dir / "threshold_stats/threshold_area_correlation.csv",
        correlations,
        tuple(correlations[0]),
    )
    fallback_fields = (
        "dataset",
        "stem",
        "method",
        "score_min",
        "score_max",
        "fallback_reason",
        "image_path",
        "source_cache_path",
    )
    _write_csv(out_dir / "threshold_stats/fallback_cases.csv", fallback, fallback_fields)

    pairwise, outcome_counts = _pairwise_vs_fixed(all_rows, adaptive, out_dir)
    _write_csv(
        out_dir / "comparison/pairwise_delta_vs_fixed050.csv",
        pairwise,
        tuple(pairwise[0]),
    )
    _write_csv(
        out_dir / "comparison/adaptive_outcome_counts.csv",
        outcome_counts,
        tuple(outcome_counts[0]),
    )
    decisions = _adaptive_decisions(by_dataset, macro, pooled, adaptive)
    _write_csv(
        out_dir / "comparison/adaptive_candidate_decisions.csv",
        decisions,
        tuple(decisions[0]),
    )

    table_md = _table_rows(macro)
    (out_dir / "comparison/threshold_table.md").write_text(table_md, encoding="utf-8")
    (out_dir / "comparison/threshold_table.tex").write_text(
        _latex_table(macro), encoding="utf-8"
    )
    fixed_methods = tuple(fixed_method_name(value) for value in args.fixed_thresholds)
    _generate_figures(
        out_dir,
        by_dataset,
        macro,
        threshold_rows,
        fixed_methods,
        adaptive,
        datasets,
    )
    if not args.skip_visualizations:
        _generate_visualizations(args, all_rows, pairwise, adaptive)
    results = _results_markdown(
        out_dir,
        macro,
        pooled,
        by_dataset,
        threshold_summary,
        outcome_counts,
        decisions,
    )
    (out_dir / "comparison/RESULTS.md").write_text(results, encoding="utf-8")
    summary = {
        "version": VERSION,
        "timestamp": _now(),
        "num_images": EXPECTED_TOTAL,
        "num_methods": len(methods),
        "num_per_image_records": len(all_rows),
        "num_fallback_cases": len(fallback),
        "results": str((out_dir / "comparison/RESULTS.md").resolve()),
        "passed": len(all_rows) == EXPECTED_TOTAL * len(methods),
    }
    write_json(out_dir / "logs/report_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--r1_root", "--r1-root", default=str(DEFAULT_R1_ROOT))
    parser.add_argument("--out_dir", "--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CHAMELEON", "CAMO", "COD10K", "NC4K"],
    )
    parser.add_argument(
        "--fixed_thresholds",
        "--fixed-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_FIXED_THRESHOLDS),
    )
    parser.add_argument(
        "--adaptive_methods",
        "--adaptive-methods",
        nargs="+",
        default=list(DEFAULT_ADAPTIVE_METHODS),
    )
    parser.add_argument("--nbins", type=int, default=256)
    parser.add_argument("--strict_greater", "--strict-greater", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch_threads", "--torch-threads", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, function in (
        ("preflight", command_preflight),
        ("regression20", command_regression20),
        ("baseline", command_baseline),
        ("evaluate", command_evaluate),
        ("report", command_report),
    ):
        child = subparsers.add_parser(command)
        _add_common_args(child)
        if command == "report":
            child.add_argument("--skip_visualizations", "--skip-visualizations", action="store_true")
        child.set_defaults(func=function)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.strict_greater:
        raise RuntimeError("this formal ablation only permits strict score > threshold")
    if args.nbins != 256:
        raise RuntimeError(f"formal Otsu/Triangle ablation requires nbins=256, got {args.nbins}")
    if len(set(args.fixed_thresholds)) != len(args.fixed_thresholds):
        raise ValueError("fixed_thresholds contains duplicates")
    if not all(0.0 <= value <= 1.0 for value in args.fixed_thresholds):
        raise ValueError("fixed thresholds must lie in [0,1]")
    args.func(args)


if __name__ == "__main__":
    main()
