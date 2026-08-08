#!/usr/bin/env python3
"""Generate, regress, and formally evaluate the frozen R1 core ablations.

No feature extraction or training is performed.  Every command consumes the
existing identity DINOv1-S/8 feature and DABE-v2 manifests.
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
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _load_feature, _params_from_cfg  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS, _load_rgb_grid  # noqa: E402
from common.dabe_rank_calibration import average_percentile_rank  # noqa: E402
from common.dabe_r1_core_ablation import (  # noqa: E402
    ABLATION_VERSION,
    GRID,
    MATCHED_CAPACITY,
    MODES,
    build_r1_core_ablation_scores,
)
from common.eval_dabe_background_null import _load_gt  # noqa: E402
from common.eval_dabe_rank_calibration import (  # noqa: E402
    FastCODContext,
    _ranking_metrics,
    _raw_metrics,
    _resize_native,
)
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "workdir" / "22"
DEFAULT_CONFIG = MAIN_ROOT / "configs" / "dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5.py"
DEFAULT_FEATURE_MANIFEST = (
    REPO_ROOT / "datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl"
)
DEFAULT_DABE_MANIFEST = (
    REPO_ROOT
    / "datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8/manifest_test.jsonl"
)
DEFAULT_RAW_REFERENCE_MANIFEST = (
    REPO_ROOT / "workdir/dabe_bgnull_v1_identity/dinov1-s8/manifest_test.jsonl"
)
DEFAULT_EXISTING_EVAL = REPO_ROOT / "workdir/dabe_bgnull_v1_eval_identity"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
METHOD_LABELS = {
    "full_bc_full_r1": "Full BC + Full R1",
    "soft_similarity": "Full BC + Soft background similarity",
    "feature_only_reconstruction": "Full BC + Feature-only reconstruction",
    "boundary280_full_r1": "Boundary-280 + Full R1",
    "bc280_full_r1": "BC-280 + Full R1",
}
METHOD_ORDER = ("full_bc_full_r1", *MODES)
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
SOFT_FIELDS = (
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
    "raw_fg_mass_ratio",
)
RANK_FIELDS = ("pixel_AP", "best_IoU_256", "best_IoU_threshold_256")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _atomic_png(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    array = (mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255)
    Image.fromarray(array, mode="L").save(temp)
    os.replace(temp, path)


def _manifest(path: Path, required_count: int = EXPECTED_TOTAL) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    if len(rows) != required_count:
        raise RuntimeError(f"{path} must contain {required_count} rows, got {len(rows)}")
    mapping = {}
    counts = defaultdict(int)
    for line, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{line}")
        key = (row["dataset"], row["stem"])
        if key in mapping:
            raise RuntimeError(f"duplicate manifest key: {key}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        mapping[key] = row
        counts[row["dataset"]] += 1
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"dataset counts differ: {dict(counts)} != {EXPECTED_COUNTS}")
    return rows, mapping


def _aligned_inputs(args) -> tuple[list[dict], dict, dict, dict]:
    dabe_rows, dabe_map = _manifest(Path(args.dabe_manifest))
    _, feature_map = _manifest(Path(args.feature_manifest))
    _, raw_map = _manifest(Path(args.raw_reference_manifest))
    keys = set(dabe_map)
    if set(feature_map) != keys or set(raw_map) != keys:
        raise RuntimeError("DABE, feature, and raw-reference manifests have different keys")
    for row in dabe_rows:
        for field in ("image_path", "gt_path"):
            if field not in row or not Path(row[field]).is_file():
                raise FileNotFoundError(row.get(field, f"missing {field}"))
    return dabe_rows, dabe_map, feature_map, raw_map


def _effective_params(config_path: Path) -> tuple[object, dict]:
    cfg = load_config(config_path)
    params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg)}
    params["VERSION"] = "v2"
    required = {
        "GRID": 37,
        "BORDER_WIDTH": 2,
        "K_RECON": 32,
        "TAU_RECON": 0.07,
        "LAMBDA_COLOR_RECON": 0.20,
        "SIGMA_COLOR_RECON": 0.05,
    }
    differences = {key: (params[key], value) for key, value in required.items() if params[key] != value}
    if differences:
        raise RuntimeError(f"frozen R1 parameters changed: {differences}")
    return cfg, params


def _load_single_map(payload: dict, field: str, path: Path, bounded: bool) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, GRID, GRID):
        raise RuntimeError(f"{field} must be Tensor[1,{GRID},{GRID}]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{field} contains NaN/Inf: {path}")
    if bounded and (float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6):
        raise RuntimeError(f"{field} is outside [0,1]: {path}")
    return value.clamp(0.0, 1.0) if bounded else value


def _dictionary_stats(values: list[int], dataset: str) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "dataset": dataset,
        "num_images": int(array.size),
        "min_size": int(array.min()),
        "max_size": int(array.max()),
        "mean_size": float(array.mean()),
        "median_size": float(np.median(array)),
        "num_below_280": int(np.sum(array < MATCHED_CAPACITY)),
    }


def command_preflight(args) -> None:
    start = time.time()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows, _, _, _ = _aligned_inputs(args)
    values = defaultdict(list)
    failures = []
    for row in rows:
        path = Path(row["cache_path"])
        payload = torch_load(path, map_location="cpu")
        anchor = _load_single_map(payload, "bg_anchor_37", path, bounded=True)
        size = int((anchor > 0.5).sum())
        values[row["dataset"]].append(size)
        if size < MATCHED_CAPACITY:
            failures.append({"dataset": row["dataset"], "stem": row["stem"], "size": size, "path": str(path)})
    stats = [_dictionary_stats(values[name], name) for name in DATASETS]
    stats.append(_dictionary_stats([item for name in DATASETS for item in values[name]], "ALL"))
    _write_csv(
        out_root / "diagnostics/dictionary_size_stats.csv",
        stats,
        ("dataset", "num_images", "min_size", "max_size", "mean_size", "median_size", "num_below_280"),
    )
    write_json(out_root / "diagnostics/dictionary_size_failures.json", failures)
    summary = {
        "version": ABLATION_VERSION,
        "timestamp": _now(),
        "source": "cached production bg_anchor_37",
        "num_images": len(rows),
        "num_below_280": len(failures),
        "passed": not failures,
        "elapsed_seconds": time.time() - start,
    }
    write_json(out_root / "diagnostics/preflight_summary.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} Full-BC dictionaries have fewer than 280 atoms")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _compute_bundle(row: dict, feature_row: dict, cfg, params):
    feature = _load_feature(feature_row, row["dataset"], row["stem"], cfg)
    rgb = _load_rgb_grid(row["image_path"], GRID)
    return build_r1_core_ablation_scores(feature, rgb, params)


def command_regression(args) -> None:
    out_root = Path(args.out_root)
    rows, _, feature_map, raw_map = _aligned_inputs(args)
    cfg, params = _effective_params(Path(args.config))
    selected = []
    for dataset in DATASETS:
        selected.extend([row for row in rows if row["dataset"] == dataset][:5])
    sample_path = out_root / "regression_samples.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        "".join(f"{row['dataset']}/{row['stem']}\n" for row in selected), encoding="utf-8"
    )
    results = []
    for row in selected:
        key = (row["dataset"], row["stem"])
        bundle = _compute_bundle(row, feature_map[key], cfg, params)
        dabe_path = Path(row["cache_path"])
        raw_path = Path(raw_map[key]["cache_path"])
        dabe = torch_load(dabe_path, map_location="cpu")
        raw_ref = torch_load(raw_path, map_location="cpu")
        reference_raw = _load_single_map(raw_ref, "regular_raw_residual_37", raw_path, bounded=False)
        reference_norm = _load_single_map(dabe, "residual_pass1_37", dabe_path, bounded=True)
        current_raw = bundle.raw["full_bc_full_r1"]
        current_norm = bundle.normalized["full_bc_full_r1"]
        gt_shape = tuple(_load_gt(row["gt_path"]).shape[-2:])
        current_hard = _resize_native(current_norm, gt_shape) > args.threshold
        reference_hard = _resize_native(reference_norm, gt_shape) > args.threshold
        results.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "raw_max_abs_error": float((current_raw - reference_raw).abs().max()),
                "normalized_max_abs_error": float((current_norm - reference_norm).abs().max()),
                "hard_exact": bool(torch.equal(current_hard, reference_hard)),
                "full_bc_size": bundle.dictionaries.full_bc_size,
                "boundary_bc280_equal": bundle.dictionaries.boundary_bc280_equal,
            }
        )
        _atomic_torch_save(
            {
                "version": ABLATION_VERSION,
                "dataset": row["dataset"],
                "stem": row["stem"],
                "raw_score_37": current_raw,
                "normalized_score_37": current_norm,
                "hard_mask_original": current_hard,
                "reference_raw_score_37": reference_raw,
                "reference_normalized_score_37": reference_norm,
                "reference_hard_mask_original": reference_hard,
            },
            out_root / "regression" / row["dataset"] / f"{row['stem']}.pt",
        )
    _write_csv(
        out_root / "regression/regression_results.csv",
        results,
        tuple(results[0]),
    )
    max_raw = max(row["raw_max_abs_error"] for row in results)
    max_norm = max(row["normalized_max_abs_error"] for row in results)
    passed = max_raw <= args.regression_tolerance and max_norm <= args.regression_tolerance and all(
        row["hard_exact"] for row in results
    )
    summary = {
        "version": ABLATION_VERSION,
        "timestamp": _now(),
        "num_samples": len(results),
        "max_raw_abs_error": max_raw,
        "max_normalized_abs_error": max_norm,
        "all_hard_exact": all(row["hard_exact"] for row in results),
        "tolerance": args.regression_tolerance,
        "passed": passed,
    }
    write_json(out_root / "regression/regression_summary.json", summary)
    if not passed:
        raise RuntimeError(f"R1 regression failed: {summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


_WORKER_CFG = None
_WORKER_PARAMS = None


def _init_generation_worker(config_path: str, torch_threads: int) -> None:
    global _WORKER_CFG, _WORKER_PARAMS
    torch.set_num_threads(torch_threads)
    _WORKER_CFG, _WORKER_PARAMS = _effective_params(Path(config_path))


def _valid_generated(paths: dict) -> bool:
    try:
        payload = torch_load(paths["score_path"], map_location="cpu")
        _load_single_map(payload, "raw_score_37", Path(paths["score_path"]), bounded=False)
        _load_single_map(payload, "normalized_score_37", Path(paths["score_path"]), bounded=True)
        soft = torch_load(paths["soft_path"], map_location="cpu")
        if not torch.is_tensor(soft) or soft.ndim != 3 or soft.shape[0] != 1:
            return False
        with Image.open(paths["hard_path"]) as image:
            image.verify()
        return True
    except (FileNotFoundError, RuntimeError, ValueError, TypeError, OSError, KeyError):
        return False


def _mode_paths(out_root: Path, mode: str, dataset: str, stem: str) -> dict:
    return {
        "score_path": str(out_root / mode / "scores_37" / dataset / f"{stem}.pt"),
        "soft_path": str(out_root / mode / "soft_masks" / dataset / f"{stem}.pt"),
        "hard_path": str(out_root / mode / "hard_masks" / dataset / f"{stem}.png"),
    }


def _generate_one(task: dict) -> dict:
    row = task["row"]
    feature_row = task["feature_row"]
    out_root = Path(task["out_root"])
    mode_paths = {
        mode: _mode_paths(out_root, mode, row["dataset"], row["stem"]) for mode in MODES
    }
    complete = {mode: _valid_generated(paths) for mode, paths in mode_paths.items()}
    if all(complete.values()):
        return {"dataset": row["dataset"], "stem": row["stem"], "skipped": True, "rows": []}
    try:
        bundle = _compute_bundle(row, feature_row, _WORKER_CFG, _WORKER_PARAMS)
        gt_shape = tuple(_load_gt(row["gt_path"]).shape[-2:])
        output_rows = []
        for mode in MODES:
            paths = mode_paths[mode]
            normalized = bundle.normalized[mode]
            native = _resize_native(normalized, gt_shape)
            hard = native > task["threshold"]
            if not complete[mode]:
                raw = bundle.raw[mode]
                payload = {
                    "version": ABLATION_VERSION,
                    "mode": mode,
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "raw_score_37": raw,
                    "normalized_score_37": normalized,
                    "raw_score_min": float(raw.min()),
                    "raw_score_max": float(raw.max()),
                    "raw_score_mean": float(raw.mean()),
                    "raw_score_std": float(raw.std(unbiased=False)),
                    "normalized_score_mean": float(normalized.mean()),
                    "predicted_foreground_area": float(hard.float().mean()),
                    "full_bc_size": bundle.dictionaries.full_bc_size,
                    "dictionary_size": (
                        bundle.dictionaries.full_bc_size
                        if mode in {"soft_similarity", "feature_only_reconstruction"}
                        else MATCHED_CAPACITY
                    ),
                    "boundary_bc280_equal": bundle.dictionaries.boundary_bc280_equal,
                    "source_feature_path": feature_row["cache_path"],
                    "source_dabe_path": row["cache_path"],
                    "image_path": row["image_path"],
                    "gt_path": row["gt_path"],
                }
                _atomic_torch_save(payload, Path(paths["score_path"]))
                _atomic_torch_save(native.float().contiguous(), Path(paths["soft_path"]))
                _atomic_png(hard, Path(paths["hard_path"]))
            output_rows.append(
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "mode": mode,
                    **paths,
                    "image_path": row["image_path"],
                    "gt_path": row["gt_path"],
                    "source_dabe_path": row["cache_path"],
                }
            )
        return {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "skipped": False,
            "rows": output_rows,
            "full_bc_size": bundle.dictionaries.full_bc_size,
            "boundary_bc280_equal": bundle.dictionaries.boundary_bc280_equal,
        }
    except Exception as exc:
        return {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def command_generate(args) -> None:
    out_root = Path(args.out_root)
    rows, _, feature_map, _ = _aligned_inputs(args)
    _effective_params(Path(args.config))
    tasks = [
        {
            "row": row,
            "feature_row": feature_map[(row["dataset"], row["stem"])],
            "out_root": str(out_root),
            "threshold": args.threshold,
        }
        for row in rows
    ]
    start = time.time()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_generation_worker,
        initargs=(str(args.config), args.torch_threads),
    ) as pool:
        for index, result in enumerate(pool.map(_generate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"[{_now()}] generate {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    write_json(out_root / "logs/generation_failures.json", failures)
    if failures:
        raise RuntimeError(f"generation failed for {len(failures)} samples")

    # Rebuild every manifest from validated final files, including resumed samples.
    for mode in MODES:
        manifest_rows = []
        for row in rows:
            paths = _mode_paths(out_root, mode, row["dataset"], row["stem"])
            if not _valid_generated(paths):
                raise RuntimeError(f"incomplete generated sample: {mode}/{row['dataset']}/{row['stem']}")
            manifest_rows.append(
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "mode": mode,
                    **paths,
                    "image_path": row["image_path"],
                    "gt_path": row["gt_path"],
                    "source_dabe_path": row["cache_path"],
                }
            )
        write_jsonl(out_root / mode / "manifest.jsonl", manifest_rows)

    summary_rows = []
    for dataset in DATASETS:
        subset = [row for row in results if row["dataset"] == dataset]
        generated = [row for row in subset if not row.get("skipped", False)]
        for mode in MODES:
            payloads = [
                torch_load(
                    _mode_paths(out_root, mode, dataset, row["stem"])["score_path"],
                    map_location="cpu",
                )
                for row in subset
            ]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "num_images": len(subset),
                    "successfully_available": len(payloads),
                    "processed_this_run": len(generated),
                    "skipped_this_run": sum(bool(row.get("skipped")) for row in subset),
                    "failed": 0,
                    "mean_dictionary_size": float(np.mean([item["dictionary_size"] for item in payloads])),
                    "mean_foreground_area": float(np.mean([item["predicted_foreground_area"] for item in payloads])),
                    "mean_raw_score_range": float(
                        np.mean([item["raw_score_max"] - item["raw_score_min"] for item in payloads])
                    ),
                    "elapsed_seconds_total_run": time.time() - start,
                }
            )
    _write_csv(out_root / "logs/generation_summary.csv", summary_rows, tuple(summary_rows[0]))


def _score_probability(score_path: str, gt_shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    path = Path(score_path)
    payload = torch_load(path, map_location="cpu")
    score = _load_single_map(payload, "normalized_score_37", path, bounded=True)
    return _resize_native(score, gt_shape), payload


def _correlation(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    x = a.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y = b.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        pearson = 1.0 if np.allclose(x, y) else 0.0
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    x_rank = average_percentile_rank(torch.from_numpy(x)).numpy().astype(np.float64)
    y_rank = average_percentile_rank(torch.from_numpy(y)).numpy().astype(np.float64)
    if float(x_rank.std()) < 1e-12 or float(y_rank.std()) < 1e-12:
        spearman = 1.0 if np.allclose(x_rank, y_rank) else 0.0
    else:
        spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def _evaluate_one(task: dict) -> dict:
    try:
        gt = _load_gt(task["gt_path"])
        gt_shape = tuple(gt.shape[-2:])
        probabilities = {}
        native37 = {}
        payloads = {}
        source_path = Path(task["source_dabe_path"])
        source = torch_load(source_path, map_location="cpu")
        baseline37 = _load_single_map(source, "residual_pass1_37", source_path, bounded=True)
        probabilities["full_bc_full_r1"] = _resize_native(baseline37, gt_shape)
        native37["full_bc_full_r1"] = baseline37
        for mode in MODES:
            probability, payload = _score_probability(task["mode_paths"][mode], gt_shape)
            probabilities[mode] = probability
            payloads[mode] = payload
            native37[mode] = payload["normalized_score_37"].float()

        context = FastCODContext(gt)
        cod = context.evaluate_many(
            [(method, "hard", probabilities[method]) for method in METHOD_ORDER]
            + [(method, "soft", probabilities[method]) for method in METHOD_ORDER],
            task["threshold"],
        )
        metric_rows = []
        for method in METHOD_ORDER:
            hard = cod[(method, "hard")]
            soft = cod[(method, "soft")]
            metric_rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
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
                    "source_path": str(source_path if method == "full_bc_full_r1" else task["mode_paths"][method]),
                    "_f_curve": soft["f_curve"],
                    "_e_curve": soft["e_curve"],
                }
            )

        correlation_rows = []
        reference_pairs = (
            ("soft_similarity", "full_bc_full_r1"),
            ("feature_only_reconstruction", "full_bc_full_r1"),
            ("feature_only_reconstruction", "soft_similarity"),
            ("boundary280_full_r1", "bc280_full_r1"),
            ("bc280_full_r1", "full_bc_full_r1"),
        )
        for candidate, reference in reference_pairs:
            pearson, spearman = _correlation(native37[candidate], native37[reference])
            candidate_mask = probabilities[candidate] > task["threshold"]
            reference_mask = probabilities[reference] > task["threshold"]
            union = int((candidate_mask | reference_mask).sum())
            intersection = int((candidate_mask & reference_mask).sum())
            correlation_rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "candidate": candidate,
                    "reference": reference,
                    "pearson_37": pearson,
                    "spearman_37": spearman,
                    "binary_mask_iou": intersection / union if union else 1.0,
                    "candidate_area": float(candidate_mask.float().mean()),
                    "reference_area": float(reference_mask.float().mean()),
                    "foreground_area_difference": float(candidate_mask.float().mean() - reference_mask.float().mean()),
                }
            )
        return {"metric_rows": metric_rows, "correlation_rows": correlation_rows}
    except Exception as exc:
        return {"error": repr(exc), "traceback": traceback.format_exc(), "dataset": task["dataset"], "stem": task["stem"]}


def _init_evaluation_worker(torch_threads: int) -> None:
    torch.set_num_threads(torch_threads)


class _Aggregate:
    def __init__(self):
        self.count = 0
        self.sums = defaultdict(float)
        self.valid = defaultdict(int)
        self.f_curve = np.zeros(256, dtype=np.float64)
        self.e_curve = np.zeros(256, dtype=np.float64)

    def add(self, row: dict) -> None:
        self.count += 1
        for field in (*HARD_FIELDS, *SOFT_FIELDS, *RAW_FIELDS, *RANK_FIELDS):
            value = float(row[field])
            if math.isfinite(value):
                self.sums[field] += value
                self.valid[field] += 1
        self.f_curve += row["_f_curve"]
        self.e_curve += row["_e_curve"]

    def means(self) -> dict:
        result = {
            field: self.sums[field] / self.valid[field] if self.valid[field] else float("nan")
            for field in (*HARD_FIELDS, *SOFT_FIELDS, *RAW_FIELDS, *RANK_FIELDS)
        }
        result["official_soft_F_beta_max"] = float(np.max(self.f_curve / self.count))
        result["official_soft_E_max"] = float(np.max(self.e_curve / self.count))
        result["num_samples"] = self.count
        result["ap_valid_count"] = self.valid["pixel_AP"]
        return result


def _aggregate_metric_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_dataset = defaultdict(_Aggregate)
    pooled = defaultdict(_Aggregate)
    for row in rows:
        by_dataset[(row["dataset"], row["method"])].add(row)
        pooled[row["method"]].add(row)
    dataset_rows = []
    for dataset in DATASETS:
        for method in METHOD_ORDER:
            dataset_rows.append({"scope": "by_dataset", "dataset": dataset, "method": method, **by_dataset[(dataset, method)].means()})
    pooled_rows = [
        {"scope": "pooled_6473", "dataset": "ALL", "method": method, **pooled[method].means()}
        for method in METHOD_ORDER
    ]
    macro_rows = []
    numeric = (*HARD_FIELDS, *SOFT_FIELDS, *RAW_FIELDS, *RANK_FIELDS)
    for method in METHOD_ORDER:
        selected = [row for row in dataset_rows if row["method"] == method]
        macro_rows.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                **{field: float(np.nanmean([float(row[field]) for row in selected])) for field in numeric},
                "num_samples": EXPECTED_TOTAL,
                "ap_valid_count": sum(int(row["ap_valid_count"]) for row in selected),
            }
        )
    return dataset_rows, pooled_rows, macro_rows


def _clean_metric_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _panel(image_path: str, gt_path: str, candidate: torch.Tensor, reference: torch.Tensor, titles: tuple[str, str], threshold: float) -> Image.Image:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    with Image.open(gt_path) as opened:
        gt = opened.convert("L")
    size = (256, 256)
    heat = lambda value: Image.fromarray((value.squeeze().numpy() * 255).astype(np.uint8), mode="L").resize(size).convert("RGB")
    mask = lambda value: Image.fromarray(((value.squeeze().numpy() > threshold).astype(np.uint8) * 255), mode="L").resize(size).convert("RGB")
    panels = [image.resize(size), gt.resize(size).convert("RGB"), heat(candidate), mask(candidate), heat(reference), mask(reference)]
    labels = ("Image", "GT", f"{titles[0]} score", f"{titles[0]} mask", f"{titles[1]} score", f"{titles[1]} mask")
    canvas = Image.new("RGB", (size[0] * len(panels), size[1] + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (part, label_text) in enumerate(zip(panels, labels)):
        canvas.paste(part, (index * size[0], 28))
        draw.text((index * size[0] + 4, 6), label_text, fill="black")
    return canvas


def _save_visuals(out_root: Path, source_rows: list[dict], metric_rows: list[dict], threshold: float) -> None:
    row_map = {(row["dataset"], row["stem"]): row for row in source_rows}
    metric_map = {(row["dataset"], row["stem"], row["method"]): row for row in metric_rows}
    for mode in MODES:
        for dataset in DATASETS:
            fixed = [row for row in source_rows if row["dataset"] == dataset][:20]
            for row in fixed:
                key = (dataset, row["stem"])
                candidate = torch_load(_mode_paths(out_root, mode, *key)["soft_path"], map_location="cpu")
                baseline37 = _load_single_map(torch_load(row["cache_path"], map_location="cpu"), "residual_pass1_37", Path(row["cache_path"]), True)
                baseline = _resize_native(baseline37, tuple(candidate.shape[-2:]))
                panel = _panel(row["image_path"], row["gt_path"], candidate, baseline, (mode, "Full R1"), threshold)
                path = out_root / mode / "vis/fixed" / dataset / f"{row['stem']}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                panel.save(path, quality=90)

    comparisons = (
        ("soft_similarity", "full_bc_full_r1", "soft_vs_full"),
        ("boundary280_full_r1", "bc280_full_r1", "boundary_vs_bc280"),
    )
    for candidate_mode, reference_mode, dirname in comparisons:
        ranked = sorted(
            source_rows,
            key=lambda source: abs(
                metric_map[(source["dataset"], source["stem"], candidate_mode)]["hard_F_beta_w"]
                - metric_map[(source["dataset"], source["stem"], reference_mode)]["hard_F_beta_w"]
            ),
            reverse=True,
        )[:30]
        for rank, row in enumerate(ranked, 1):
            key = (row["dataset"], row["stem"])
            candidate = torch_load(_mode_paths(out_root, candidate_mode, *key)["soft_path"], map_location="cpu")
            if reference_mode == "full_bc_full_r1":
                source = torch_load(row["cache_path"], map_location="cpu")
                reference = _resize_native(_load_single_map(source, "residual_pass1_37", Path(row["cache_path"]), True), tuple(candidate.shape[-2:]))
            else:
                reference = torch_load(_mode_paths(out_root, reference_mode, *key)["soft_path"], map_location="cpu")
            panel = _panel(row["image_path"], row["gt_path"], candidate, reference, (candidate_mode, reference_mode), threshold)
            path = out_root / candidate_mode / "vis" / dirname / f"{rank:02d}_{row['dataset']}_{row['stem']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(path, quality=90)

    # The feature/RGB audit explicitly requires both directional Top-30 lists.
    feature_rows = []
    for row in source_rows:
        candidate = metric_map[(row["dataset"], row["stem"], "feature_only_reconstruction")]
        baseline = metric_map[(row["dataset"], row["stem"], "full_bc_full_r1")]
        feature_rows.append((float(baseline["hard_F_beta_w"] - candidate["hard_F_beta_w"]), row))
    directional = (
        ("full_better_top30", sorted(feature_rows, key=lambda item: item[0], reverse=True)[:30]),
        ("feature_better_top30", sorted(feature_rows, key=lambda item: item[0])[:30]),
    )
    for dirname, ranked in directional:
        for rank, (_, row) in enumerate(ranked, 1):
            key = (row["dataset"], row["stem"])
            feature = torch_load(
                _mode_paths(out_root, "feature_only_reconstruction", *key)["soft_path"],
                map_location="cpu",
            )
            source = torch_load(row["cache_path"], map_location="cpu")
            baseline = _resize_native(
                _load_single_map(source, "residual_pass1_37", Path(row["cache_path"]), True),
                tuple(feature.shape[-2:]),
            )
            panel = _panel(
                row["image_path"], row["gt_path"], feature, baseline,
                ("Feature-only", "Full R1"), threshold,
            )
            path = (
                out_root / "feature_only_reconstruction" / "vis" / dirname
                / f"{rank:02d}_{row['dataset']}_{row['stem']}.jpg"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(path, quality=90)


def _existing_reference_rows(existing_eval: Path) -> tuple[dict, dict]:
    hard = _read_csv(existing_eval / "hard_dataset_macro.csv")
    rank = _read_csv(existing_eval / "ranking_dataset_macro.csv")
    hard_map = {row["method"]: row for row in hard}
    rank_map = {row["method"]: row for row in rank}
    required = ("RC-NN-MinMax", "N0-R1")
    if any(method not in hard_map or method not in rank_map for method in required):
        raise RuntimeError(f"existing evaluation lacks {required}: {existing_eval}")
    return hard_map, rank_map


def _comparison_outputs(out_root: Path, dataset_rows: list[dict], pooled_rows: list[dict], macro_rows: list[dict], existing_eval: Path, correlations: list[dict]) -> None:
    hard_existing, rank_existing = _existing_reference_rows(existing_eval)
    macro_map = {row["method"]: row for row in macro_rows}
    baseline_existing = hard_existing["N0-R1"]
    regression_fields = {
        "hard_S_m": "hard_S_m",
        "hard_F_beta_w": "hard_F_beta_w",
        "hard_E_mean": "hard_E_mean",
        "hard_MAE": "hard_MAE",
    }
    baseline_errors = {
        field: abs(float(macro_map["full_bc_full_r1"][field]) - float(baseline_existing[source]))
        for field, source in regression_fields.items()
    }
    if max(baseline_errors.values()) > 1e-6:
        raise RuntimeError(f"formal Full-R1 metric regression failed: {baseline_errors}")
    comparison = []
    single = hard_existing["RC-NN-MinMax"]
    comparison.append(
        {
            "dictionary": "Full BC",
            "scoring": "Single-atom reconstruction",
            "method": "RC-NN-MinMax",
            "S_m": float(single["hard_S_m"]),
            "F_beta_w": float(single["hard_F_beta_w"]),
            "E_phi_m": float(single["hard_E_mean"]),
            "MAE": float(single["hard_MAE"]),
            "Precision": float(single["hard_Precision"]),
            "Recall": float(single["hard_Recall"]),
            "Area": float(single["hard_Area"]),
            "Pixel_AP": float(rank_existing["RC-NN-MinMax"]["pixel_AP"]),
        }
    )
    descriptors = {
        "soft_similarity": ("Full BC", "Soft background similarity"),
        "feature_only_reconstruction": ("Full BC", "Feature-only reconstruction"),
        "boundary280_full_r1": ("Boundary-280", "Full R1"),
        "bc280_full_r1": ("BC-280", "Full R1"),
    }
    for method in MODES:
        row = macro_map[method]
        dictionary, scoring = descriptors[method]
        comparison.append(
            {
                "dictionary": dictionary,
                "scoring": scoring,
                "method": method,
                "S_m": row["hard_S_m"],
                "F_beta_w": row["hard_F_beta_w"],
                "E_phi_m": row["hard_E_mean"],
                "MAE": row["hard_MAE"],
                "Precision": row["hard_Precision"],
                "Recall": row["hard_Recall"],
                "Area": row["hard_Area"],
                "Pixel_AP": row["pixel_AP"],
            }
        )
    baseline = baseline_existing
    comparison.append(
        {
            "dictionary": "Full BC",
            "scoring": "Full R1",
            "method": "N0-R1",
            "S_m": float(baseline["hard_S_m"]),
            "F_beta_w": float(baseline["hard_F_beta_w"]),
            "E_phi_m": float(baseline["hard_E_mean"]),
            "MAE": float(baseline["hard_MAE"]),
            "Precision": float(baseline["hard_Precision"]),
            "Recall": float(baseline["hard_Recall"]),
            "Area": float(baseline["hard_Area"]),
            "Pixel_AP": float(rank_existing["N0-R1"]["pixel_AP"]),
        }
    )
    _write_csv(out_root / "comparison/metrics_by_dataset.csv", dataset_rows, tuple(dataset_rows[0]))
    _write_csv(out_root / "comparison/metrics_overall_6473.csv", pooled_rows + macro_rows, tuple((pooled_rows + macro_rows)[0]))
    _write_csv(out_root / "comparison/comparison_table.csv", comparison, tuple(comparison[0]))
    header = "| Dictionary | Scoring | S_m | F_beta^w | E_phi^m | MAE | Precision | Recall | Area | Pixel AP |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(
        f"| {row['dictionary']} | {row['scoring']} | {row['S_m']:.4f} | {row['F_beta_w']:.4f} | {row['E_phi_m']:.4f} | {row['MAE']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['Area']:.4f} | {row['Pixel_AP']:.4f} |\n"
        for row in comparison
    )
    (out_root / "comparison/comparison_table.md").write_text(header + body, encoding="utf-8")

    def delta(a: str, b: str, field: str) -> float:
        return float(macro_map[a][field]) - float(macro_map[b][field])

    correlation_means = defaultdict(list)
    for row in correlations:
        correlation_means[(row["candidate"], row["reference"])].append(row)
    corr_lines = []
    for pair, values in correlation_means.items():
        corr_lines.append(
            f"- `{pair[0]}` vs `{pair[1]}`: Pearson={np.mean([r['pearson_37'] for r in values]):.6f}, Spearman={np.mean([r['spearman_37'] for r in values]):.6f}, mask IoU={np.mean([r['binary_mask_iou'] for r in values]):.6f}, ΔArea={np.mean([r['foreground_area_difference'] for r in values]):+.6f}"
        )
    feature_small = abs(delta("full_bc_full_r1", "feature_only_reconstruction", "hard_F_beta_w")) < 0.001 and abs(
        delta("full_bc_full_r1", "feature_only_reconstruction", "hard_MAE")
    ) < 0.0005
    boundary_same = abs(delta("bc280_full_r1", "boundary280_full_r1", "hard_F_beta_w")) < 1e-12 and abs(
        delta("bc280_full_r1", "boundary280_full_r1", "hard_MAE")
    ) < 1e-12
    reconstruction_small = abs(delta("feature_only_reconstruction", "soft_similarity", "hard_F_beta_w")) < 0.001 and abs(
        delta("feature_only_reconstruction", "soft_similarity", "hard_MAE")
    ) < 0.0005
    git_hash = subprocess.run(
        ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or "N/A"
    results = f"""# R1 核心机制四项公平消融结果

生成时间：{_now()}  
代码版本：`{git_hash}`  
协议：DINOv1-S/8，identity，37→68→原始 GT，strict `> 0.5`。
输出根目录：`{out_root.resolve()}`。

## 汇总表

{header + body}
## Reconstruction vs Soft Similarity

- Full R1 − Soft Similarity: ΔFβw={delta('full_bc_full_r1', 'soft_similarity', 'hard_F_beta_w'):+.6f}，ΔMAE={delta('full_bc_full_r1', 'soft_similarity', 'hard_MAE'):+.6f}。
- Feature-only − Soft Similarity: ΔFβw={delta('feature_only_reconstruction', 'soft_similarity', 'hard_F_beta_w'):+.6f}，ΔMAE={delta('feature_only_reconstruction', 'soft_similarity', 'hard_MAE'):+.6f}。
- 结论：{'两者差异很小，重构分数与相似度加权背景匹配紧密相关，不能强行宣称重构明显优越。' if reconstruction_small else ('Feature-only 的 Fβw 更高，且差异未落入极小范围，可支持 patch-specific feature reconstruction 提供了额外判别信息。' if delta('feature_only_reconstruction', 'soft_similarity', 'hard_F_beta_w') > 0 else 'Soft Similarity 的 Fβw 不低于 Feature-only，不能支持“重构优于直接背景相似度”。')}

## Feature-only vs Full R1

- Full R1 − Feature-only: ΔFβw={delta('full_bc_full_r1', 'feature_only_reconstruction', 'hard_F_beta_w'):+.6f}，ΔMAE={delta('full_bc_full_r1', 'feature_only_reconstruction', 'hard_MAE'):+.6f}。
- RGB 项决策：{'差异落入简化阈值，建议最终论文移除 RGB reconstruction residual。' if feature_small else '差异未落入简化阈值；是否保留 RGB 项应结合定量方向和 Top-30 样本。'}

## Boundary-280 vs BC-280

- BC-280 − Boundary-280: ΔFβw={delta('bc280_full_r1', 'boundary280_full_r1', 'hard_F_beta_w'):+.6f}，ΔMAE={delta('bc280_full_r1', 'boundary280_full_r1', 'hard_MAE'):+.6f}。
- 判读：{'二者结果完全一致；在当前 BC 定义下，BC-280 与两圈边界集合退化为同一字典，不能据此声称容量匹配时连通筛选有效。' if boundary_same else '二者不相同，该差异隔离反映容量匹配时的 connectivity selection effect。'}

## BC-280 vs Full BC

- Full BC − BC-280: ΔFβw={delta('full_bc_full_r1', 'bc280_full_r1', 'hard_F_beta_w'):+.6f}，ΔMAE={delta('full_bc_full_r1', 'bc280_full_r1', 'hard_MAE'):+.6f}。
- 该比较只解释为 dictionary capacity / coverage effect，不归因于是否使用 BC 规则。

## 分数与面积诊断

{chr(10).join(corr_lines)}

## 实现口径说明

Soft Similarity 固定复用生产 Full R1 已选出的 Top-K 原子及其 `a_ij`，评分中不加入 RGB residual。当前生产实现的检索与权重本身包含冻结的颜色相似度项；这样才能满足“邻居与权重完全不变”的单变量消融。若将 `a_ij` 改成纯特征权重，会同时改变检索和权重，不属于本次 scoring-only 对照。
"""
    (out_root / "comparison/RESULTS.md").write_text(results, encoding="utf-8")


def command_evaluate(args) -> None:
    out_root = Path(args.out_root)
    source_rows, _, _, _ = _aligned_inputs(args)
    mode_maps = {}
    for mode in MODES:
        rows = read_jsonl(out_root / mode / "manifest.jsonl")
        if len(rows) != EXPECTED_TOTAL:
            raise RuntimeError(f"{mode} manifest is incomplete: {len(rows)}")
        mode_maps[mode] = {(row["dataset"], row["stem"]): row for row in rows}
    tasks = []
    for row in source_rows:
        key = (row["dataset"], row["stem"])
        tasks.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "gt_path": row["gt_path"],
                "source_dabe_path": row["cache_path"],
                "mode_paths": {mode: mode_maps[mode][key]["score_path"] for mode in MODES},
                "threshold": args.threshold,
            }
        )
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_evaluation_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_evaluate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"[{_now()}] evaluate {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    write_json(out_root / "logs/evaluation_failures.json", failures)
    if failures:
        raise RuntimeError(f"evaluation failed for {len(failures)} samples")
    metric_rows = [row for result in results for row in result["metric_rows"]]
    correlations = [row for result in results for row in result["correlation_rows"]]
    dataset_rows, pooled_rows, macro_rows = _aggregate_metric_rows(metric_rows)
    clean_rows = [_clean_metric_row(row) for row in metric_rows]
    fields = tuple(clean_rows[0])
    _write_csv(out_root / "diagnostics/per_image_metrics.csv", clean_rows, fields)
    _write_csv(out_root / "diagnostics/score_correlations.csv", correlations, tuple(correlations[0]))
    area_rows = []
    for dataset in (*DATASETS, "ALL"):
        for method in METHOD_ORDER:
            values = [
                float(row["hard_Area"])
                for row in clean_rows
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            area_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "num_images": len(values),
                    "mean_area": float(np.mean(values)),
                    "std_area": float(np.std(values)),
                    "min_area": float(np.min(values)),
                    "max_area": float(np.max(values)),
                }
            )
    _write_csv(out_root / "diagnostics/foreground_area_stats.csv", area_rows, tuple(area_rows[0]))
    for mode in MODES:
        per_mode = [row for row in clean_rows if row["method"] == mode]
        by_dataset = [row for row in dataset_rows if row["method"] == mode]
        pooled = [row for row in pooled_rows if row["method"] == mode]
        macro = [row for row in macro_rows if row["method"] == mode]
        _write_csv(out_root / mode / "metrics/per_sample.csv", per_mode, tuple(per_mode[0]))
        _write_csv(out_root / mode / "metrics/by_dataset.csv", by_dataset, tuple(by_dataset[0]))
        _write_csv(out_root / mode / "metrics/pooled_6473.csv", pooled, tuple(pooled[0]))
        _write_csv(out_root / mode / "metrics/dataset_macro.csv", macro, tuple(macro[0]))
    _comparison_outputs(out_root, dataset_rows, pooled_rows, macro_rows, Path(args.existing_eval), correlations)
    if not args.skip_visuals:
        _save_visuals(out_root, source_rows, metric_rows, args.threshold)
    print(f"Results: {out_root / 'comparison/RESULTS.md'}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--feature-manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--dabe-manifest", default=str(DEFAULT_DABE_MANIFEST))
    parser.add_argument("--raw-reference-manifest", default=str(DEFAULT_RAW_REFERENCE_MANIFEST))
    parser.add_argument("--threshold", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    _add_common(preflight)
    preflight.set_defaults(func=command_preflight)
    regression = sub.add_parser("regression")
    _add_common(regression)
    regression.add_argument("--regression-tolerance", type=float, default=1e-6)
    regression.set_defaults(func=command_regression)
    generate = sub.add_parser("generate")
    _add_common(generate)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--torch-threads", type=int, default=1)
    generate.set_defaults(func=command_generate)
    evaluate = sub.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--existing-eval", default=str(DEFAULT_EXISTING_EVAL))
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--torch-threads", type=int, default=1)
    evaluate.add_argument("--skip-visuals", action="store_true")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if Path(args.out_root).resolve() != DEFAULT_OUT_ROOT.resolve():
        print(f"[Info] non-default output root explicitly selected: {Path(args.out_root).resolve()}")
    args.func(args)


if __name__ == "__main__":
    main()
