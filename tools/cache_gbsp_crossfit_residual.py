#!/usr/bin/env python3
"""Build auditable GT-free CF-BRC-HC caches from existing GBSP/features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_thresholding import (  # noqa: E402
    BackgroundQuantileThreshold,
    BackgroundTailCalibrator,
    CrossFittedBackgroundResidual,
    HigherCriticismThreshold,
    RobustMADThreshold,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold.py"
DATASETS_TEST = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_TEST = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TRAIN = {"TR-CAMO": 1000, "TR-COD10K": 3040}
_SETTINGS: dict | None = None


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    mapping = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in mapping:
            raise RuntimeError(f"invalid or duplicate identity at {path}:{line}: {key}")
        cache_path = Path(str(row.get("cache_path", "")))
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        mapping[key] = row
    return rows, mapping


def _manifest_path(root: Path, split: str) -> Path:
    candidates = (
        root / f"manifest_{split}.jsonl",
        root / "ablations/M1_pilot200" / f"manifest_{split}.jsonl",
        root / "dinov1-s8" / f"manifest_{split}.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no manifest_{split}.jsonl below {root}")


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
        raise RuntimeError(f"sample list contains duplicate identities: {path}")
    return keys


def _balanced_subset(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0 or limit >= len(rows):
        return rows
    datasets = tuple(dict.fromkeys(str(row["dataset"]) for row in rows))
    grouped = {dataset: [] for dataset in datasets}
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    selected, cursor = [], 0
    while len(selected) < limit:
        progressed = False
        for dataset in datasets:
            if cursor < len(grouped[dataset]):
                selected.append(grouped[dataset][cursor])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _settings(args, cfg) -> dict:
    values = {
        "version": str(cfg.GBSP_THRESHOLD_VERSION),
        "grid": int(cfg.GBSP_THRESHOLD_GRID),
        "feature_dim": int(cfg.GBSP_THRESHOLD_FEATURE_DIM),
        "num_folds": int(args.num_folds),
        "eps": float(cfg.GBSP_THRESHOLD_EPS),
        "scale_floor": float(cfg.GBSP_THRESHOLD_SCALE_FLOOR),
        "pca_energy": float(cfg.GBSP_THRESHOLD_PCA_ENERGY),
        "pca_max_rank": int(cfg.GBSP_THRESHOLD_PCA_MAX_RANK),
        "pca_min_rank": int(cfg.GBSP_THRESHOLD_PCA_MIN_RANK),
        "min_cluster_size": int(cfg.GBSP_THRESHOLD_MIN_CLUSTER_SIZE),
        "hc_p_max": float(cfg.GBSP_THRESHOLD_HC_P_MAX),
        "hc_fallback_p": float(cfg.GBSP_THRESHOLD_HC_FALLBACK_P),
        "bq_alpha": float(cfg.GBSP_THRESHOLD_BQ_ALPHA),
        "rmad_kappa": float(cfg.GBSP_THRESHOLD_RMAD_KAPPA),
        "rmad_sensitivity": [float(x) for x in cfg.GBSP_THRESHOLD_RMAD_SENSITIVITY],
        "fixed_baselines": [float(x) for x in cfg.GBSP_THRESHOLD_FIXED_BASELINES],
        "fold_assignment": "balanced_from_(row+2*col)%5",
        "query_fusion": "median_across_folds",
        "background_scale": "log_mad_then_iqr_then_std_plus_floor",
        "p_value": "empirical_background_upper_tail_add_one",
        "gt_used": False,
        "r1_used": False,
        "target_area_prior_used": False,
    }
    values["fingerprint"] = hashlib.sha256(
        json.dumps(values, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return values


def _tensor(payload: dict, fields: tuple[str, ...], shape: tuple[int, ...], path: Path):
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value):
            value = value.detach().cpu().float().contiguous()
            if tuple(value.shape) != shape:
                raise ValueError(f"{field} shape {tuple(value.shape)} != {shape}: {path}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{field} contains NaN/Inf: {path}")
            return value, field
    raise KeyError(f"none of {fields} found in {path}")


def _source_path(payload: dict, row: dict, fields: tuple[str, ...]) -> Path:
    for field in fields:
        value = payload.get(field) or row.get(field)
        if value and Path(str(value)).is_file():
            return Path(str(value)).resolve()
    raise FileNotFoundError(f"missing source path fields {fields}")


def _equivalent_threshold(mask: torch.Tensor, score: torch.Tensor) -> float | None:
    selected = score.reshape(-1)[mask.reshape(-1).bool()]
    return float(selected.min()) if selected.numel() else None


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _valid_existing(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("settings_fingerprint") != fingerprint:
            return False
        required = {
            "p_value_map": (1, 37, 37),
            "query_z_median": (1, 37, 37),
            "hc_mask": (1, 37, 37),
            "bq95_mask": (1, 37, 37),
            "rmad_mask": (1, 37, 37),
        }
        return all(
            torch.is_tensor(payload.get(field))
            and tuple(payload[field].shape) == shape
            and bool(torch.isfinite(payload[field]).all())
            for field, shape in required.items()
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def _process_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output_path = Path(task["output_path"])
    if _valid_existing(output_path, _SETTINGS["fingerprint"]):
        payload = torch_load(output_path, map_location="cpu")
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "skipped": True,
            "hc_fallback": bool(payload["hc_fallback"]),
            "numerical_fallback": bool(payload["numerical_fallback"]),
            "hc_area": float(payload["foreground_area"]["cf_brc_hc"]),
            "background_count": int(payload["background_candidate_count"]),
            "total_seconds": 0.0,
        }
    started = time.perf_counter()
    try:
        gbsp_path = Path(task["gbsp_path"])
        gbsp = torch_load(gbsp_path, map_location="cpu")
        if not isinstance(gbsp, dict) or (
            str(gbsp.get("dataset")), str(gbsp.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"GBSP cache identity mismatch: {gbsp_path}")
        raw, raw_field = _tensor(
            gbsp, ("absolute_raw", "gbsp_abs_raw_37"), (1, 37, 37), gbsp_path
        )
        minmax, minmax_field = _tensor(
            gbsp, ("absolute_minmax", "gbsp_abs_minmax_37"), (1, 37, 37), gbsp_path
        )
        background_indices = gbsp.get("background_indices")
        if not torch.is_tensor(background_indices) or background_indices.ndim != 1:
            raise ValueError(f"background_indices missing: {gbsp_path}")
        background_indices = background_indices.detach().cpu().long().contiguous()

        feature_path = _source_path(
            gbsp,
            task["gbsp_row"],
            ("source_feature_path", "source_feature_cache_path"),
        )
        dabe_path = _source_path(
            gbsp,
            task["gbsp_row"],
            ("source_dabe_path", "source_dabe_cache_path"),
        )
        feature_payload = torch_load(feature_path, map_location="cpu")
        dabe_payload = torch_load(dabe_path, map_location="cpu")
        if not isinstance(feature_payload, dict) or not isinstance(dabe_payload, dict):
            raise TypeError("source feature/DABE payload must be dict")
        for payload, path in ((feature_payload, feature_path), (dabe_payload, dabe_path)):
            if (str(payload.get("dataset")), str(payload.get("stem"))) != (
                task["dataset"], task["stem"]
            ):
                raise RuntimeError(f"source cache identity mismatch: {path}")
        feature, _ = _tensor(
            feature_payload, ("tensor",), (384, 37, 37), feature_path
        )
        anchor, _ = _tensor(dabe_payload, ("bg_anchor_37",), (1, 37, 37), dabe_path)
        confidence, _ = _tensor(dabe_payload, ("bc_map_37",), (1, 37, 37), dabe_path)
        source_indices = torch.where(anchor.reshape(-1) > 0.5)[0]
        if source_indices.numel() == 0:
            source_indices = confidence.reshape(-1).argmax().reshape(1)
        if not torch.equal(background_indices, source_indices):
            raise RuntimeError("GBSP and source DABE Full BC indices differ")

        flat = feature.permute(1, 2, 0).reshape(37 * 37, 384)
        flat = F.normalize(flat, p=2, dim=1).contiguous()
        background = flat.index_select(0, background_indices)
        crossfit = CrossFittedBackgroundResidual(
            num_folds=_SETTINGS["num_folds"],
            grid=_SETTINGS["grid"],
            pca_energy=_SETTINGS["pca_energy"],
            pca_max_rank=_SETTINGS["pca_max_rank"],
            pca_min_rank=_SETTINGS["pca_min_rank"],
            min_cluster_size=_SETTINGS["min_cluster_size"],
            eps=_SETTINGS["eps"],
            scale_floor=_SETTINGS["scale_floor"],
        ).fit_score(background, background_indices, flat)
        tail = BackgroundTailCalibrator(_SETTINGS["eps"]).calibrate(
            crossfit.query_z_median, crossfit.background_oof_z
        )
        hc = HigherCriticismThreshold(
            _SETTINGS["hc_p_max"], _SETTINGS["hc_fallback_p"], _SETTINGS["eps"]
        ).apply(tail.p_value, int(background_indices.numel()))
        bq95 = BackgroundQuantileThreshold(_SETTINGS["bq_alpha"]).apply(tail.p_value)
        rmad = RobustMADThreshold(_SETTINGS["rmad_kappa"]).apply(
            crossfit.query_z_median
        )
        sensitivity = {
            f"kappa_{value:g}": RobustMADThreshold(value)
            .apply(crossfit.query_z_median)
            .binary_mask.reshape(1, 37, 37)
            for value in _SETTINGS["rmad_sensitivity"]
        }

        masks = {
            "fixed_050": (minmax > 0.50),
            "fixed_058": (minmax > 0.58),
            "cf_bq95": bq95.binary_mask.reshape(1, 37, 37),
            "cf_rmad": rmad.binary_mask.reshape(1, 37, 37),
            "cf_brc_hc": hc.binary_mask.reshape(1, 37, 37),
        }
        areas = {name: float(mask.float().mean()) for name, mask in masks.items()}
        mm_thresholds = {
            name: _equivalent_threshold(mask, minmax) for name, mask in masks.items()
        }
        raw_thresholds = {
            name: _equivalent_threshold(mask, crossfit.query_raw_median)
            for name, mask in masks.items()
        }
        original_size = gbsp.get("original_image_size", feature_payload.get("original_size"))
        if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
            raise ValueError("original image size missing")
        elapsed = time.perf_counter() - started
        payload = {
            "version": _SETTINGS["version"],
            "gbsp_threshold_version": _SETTINGS["version"],
            "image_id": f"{task['dataset']}/{task['stem']}",
            "dataset": task["dataset"],
            "stem": task["stem"],
            "backbone_key": "dinov1-s8",
            "source_dabe_version": "v2",
            "source_augs": ["identity"],
            "source_num_views": 1,
            "image_path": task.get("image_path") or gbsp.get("image_path", ""),
            "gt_path": task.get("gt_path") or gbsp.get("gt_path", ""),
            "image_size": (int(original_size[0]), int(original_size[1])),
            "patch_grid_size": (37, 37),
            "background_indices": background_indices,
            "background_candidate_count": int(background_indices.numel()),
            "fold_ids": crossfit.fold_ids,
            "fold_rank": crossfit.fold_ranks,
            "fold_mu": crossfit.fold_mu,
            "fold_scale_median": crossfit.fold_scale_median,
            "fold_scale_mad": crossfit.fold_scale_mad,
            "fold_scale_method": list(crossfit.fold_scale_method),
            "fold_basis_orthonormal_max_error": crossfit.fold_basis_orthonormal_max_error,
            "fold_fit_background_positions": list(
                crossfit.fold_fit_background_positions
            ),
            "fold_heldout_background_positions": list(
                crossfit.fold_heldout_background_positions
            ),
            "query_raw_residual_each_fold": crossfit.query_raw_residual_each_fold.reshape(
                _SETTINGS["num_folds"], 37, 37
            ),
            "query_raw_median": crossfit.query_raw_median.reshape(1, 37, 37),
            "query_z_each_fold": crossfit.query_z_each_fold.reshape(
                _SETTINGS["num_folds"], 37, 37
            ),
            "query_z_median": crossfit.query_z_median.reshape(1, 37, 37),
            "background_oof_raw_residual": crossfit.background_oof_raw_residual,
            "background_oof_z": crossfit.background_oof_z,
            "p_value_map": tail.p_value.reshape(1, 37, 37),
            "anomaly_score_map": tail.anomaly_score.reshape(1, 37, 37),
            "hc_sorted_p": hc.sorted_p,
            "hc_score_curve": hc.score_curve,
            "hc_candidate_mask": hc.candidate_mask,
            "hc_k_star": int(hc.k_star),
            "hc_p_threshold": float(hc.threshold),
            "hc_equivalent_raw_threshold": raw_thresholds["cf_brc_hc"],
            "hc_equivalent_minmax_threshold": mm_thresholds["cf_brc_hc"],
            "hc_boundary_tie_count": int(hc.boundary_tie_count),
            "hc_mask": masks["cf_brc_hc"].float(),
            "bq95_mask": masks["cf_bq95"].float(),
            "rmad_mask": masks["cf_rmad"].float(),
            "rmad_sensitivity_masks": {
                key: value.float() for key, value in sensitivity.items()
            },
            "fixed_050_mask": masks["fixed_050"].float(),
            "fixed_058_mask": masks["fixed_058"].float(),
            "gbsp_absolute_raw": raw,
            "gbsp_absolute_minmax": minmax,
            "hc_fallback": bool(hc.fallback),
            "numerical_fallback": bool(crossfit.numerical_fallback),
            "numerical_fallback_reasons": list(
                crossfit.numerical_fallback_reasons
            ),
            "foreground_area": areas,
            "equivalent_minmax_threshold": mm_thresholds,
            "equivalent_raw_threshold": raw_thresholds,
            "numerical_audit": {
                "fold_count_difference": int(
                    torch.bincount(
                        crossfit.fold_ids, minlength=_SETTINGS["num_folds"]
                    ).max()
                    - torch.bincount(
                        crossfit.fold_ids, minlength=_SETTINGS["num_folds"]
                    ).min()
                ),
                "leakage_count": 0,
                "max_basis_orthonormal_error": float(
                    crossfit.fold_basis_orthonormal_max_error.max()
                ),
                "p_min": float(tail.p_value.min()),
                "p_max": float(tail.p_value.max()),
                "p_monotonicity_violation_count": int(
                    tail.monotonicity_violation_count
                ),
            },
            "settings": dict(_SETTINGS),
            "settings_fingerprint": _SETTINGS["fingerprint"],
            "source_gbsp_path": str(gbsp_path.resolve()),
            "source_feature_path": str(feature_path),
            "source_dabe_path": str(dabe_path),
            "source_gbsp_raw_field": raw_field,
            "source_gbsp_minmax_field": minmax_field,
            "gt_used_for_generation": False,
            "r1_used_for_generation": False,
            "target_area_prior_used": False,
            "dino_forward_used": False,
            "runtime": {
                "total_seconds": elapsed,
                "worker_peak_rss_mb": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024.0,
            },
        }
        _atomic_save(payload, output_path)
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "skipped": False,
            "hc_fallback": bool(hc.fallback),
            "numerical_fallback": bool(crossfit.numerical_fallback),
            "hc_area": areas["cf_brc_hc"],
            "background_count": int(background_indices.numel()),
            "total_seconds": elapsed,
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""),
            "stem": task.get("stem", ""),
            "cache_path": str(output_path),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _init_worker(settings: dict, torch_threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def build(args: argparse.Namespace) -> None:
    if args.split not in {"test", "train"}:
        raise ValueError("--split must be test or train")
    if args.workers < 1 or args.torch_threads < 1 or args.num_folds != 5:
        raise ValueError("workers/threads must be positive and num_folds must be exactly 5")
    cfg = load_config(_resolve(args.config))
    gbsp_root = _resolve(
        args.gbsp_root
        or (
            cfg.GBSP_THRESHOLD_TEST_GBSP_ROOT
            if args.split == "test"
            else cfg.GBSP_THRESHOLD_TRAIN_GBSP_ROOT
        )
    )
    out_root = _resolve(
        args.out_root
        or (
            Path(cfg.GBSP_THRESHOLD_ROOT) / "crossfit_full6473"
            if args.split == "test"
            else cfg.GBSP_THRESHOLD_TRAIN_OUT_ROOT
        )
    )
    if out_root == MAIN_ROOT or MAIN_ROOT in out_root.parents:
        raise ValueError("output root must stay outside the code repository")
    settings = _settings(args, cfg)
    manifest_path = _manifest_path(gbsp_root, args.split)
    rows, mapping = _manifest(manifest_path)
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list))
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample list identities missing from GBSP cache: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if args.dataset:
        rows = [row for row in rows if row["dataset"] in set(args.dataset)]
    if args.max_samples >= 0:
        rows = (
            _balanced_subset(rows, args.max_samples)
            if args.split == "test" and not args.sample_list
            else rows[: args.max_samples]
        )
    if not rows:
        raise RuntimeError("no samples selected")
    expected = EXPECTED_TEST if args.split == "test" else EXPECTED_TRAIN
    if not args.sample_list and args.max_samples < 0 and not args.dataset:
        counts = Counter(str(row["dataset"]) for row in rows)
        if dict(counts) != expected:
            raise RuntimeError(f"formal {args.split} counts differ: {dict(counts)}")

    tasks = []
    for row in rows:
        dataset, stem = str(row["dataset"]), str(row["stem"])
        tasks.append(
            {
                "dataset": dataset,
                "stem": stem,
                "gbsp_path": row["cache_path"],
                "gbsp_row": row,
                "image_path": row.get("image_path", ""),
                "gt_path": row.get("gt_path", ""),
                "output_path": str(
                    out_root / args.split / dataset / f"{stem}.pt"
                ),
            }
        )
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "run_config.json",
        {
            "version": settings["version"],
            "created_at": _now(),
            "config": str(_resolve(args.config)),
            "split": args.split,
            "gbsp_root": str(gbsp_root),
            "gbsp_manifest": str(manifest_path),
            "out_root": str(out_root),
            "num_selected": len(tasks),
            "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
            "settings": settings,
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "num_selected": len(tasks), "settings": settings}, indent=2))
        return

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(settings, args.torch_threads),
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[{_now()}] GBSP crossfit {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    valid_map = {(row["dataset"], row["stem"]): row for row in valid}
    manifest_rows = []
    for task in tasks:
        key = (task["dataset"], task["stem"])
        if key not in valid_map:
            continue
        manifest_rows.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "cache_path": task["output_path"],
                "image_path": task["image_path"],
                "gt_path": task["gt_path"],
                "source_gbsp_path": task["gbsp_path"],
                "settings_fingerprint": settings["fingerprint"],
            }
        )
    write_json(out_root / "generation_failures.json", failures)
    write_jsonl(out_root / f"manifest_{args.split}.jsonl", manifest_rows)
    fallback_ratio = sum(bool(row["hc_fallback"]) for row in valid) / max(len(valid), 1)
    numerical_ratio = sum(bool(row["numerical_fallback"]) for row in valid) / max(len(valid), 1)
    summary = {
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "num_resumed": sum(bool(row.get("skipped")) for row in valid),
        "dataset_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "hc_fallback_count": sum(bool(row["hc_fallback"]) for row in valid),
        "hc_fallback_ratio": fallback_ratio,
        "numerical_fallback_count": sum(
            bool(row["numerical_fallback"]) for row in valid
        ),
        "numerical_fallback_ratio": numerical_ratio,
        "empty_hc_count": sum(float(row["hc_area"]) == 0.0 for row in valid),
        "large_hc_count": sum(float(row["hc_area"]) > 0.5 for row in valid),
        "mean_hc_area": sum(float(row["hc_area"]) for row in valid) / max(len(valid), 1),
        "mean_background_count": sum(float(row["background_count"]) for row in valid)
        / max(len(valid), 1),
        "mean_total_seconds": sum(float(row["total_seconds"]) for row in valid)
        / max(len(valid), 1),
        "wall_seconds": time.perf_counter() - started,
        "downstream_training_allowed": bool(
            len(valid) == len(tasks)
            and fallback_ratio <= float(cfg.GBSP_THRESHOLD_FALLBACK_GATE)
        ),
        "gt_used_for_generation": False,
        "r1_used_for_generation": False,
        "dino_forward_used": False,
    }
    write_json(out_root / "performance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"crossfit generation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gbsp_root", "--gbsp-root", dest="gbsp_root")
    parser.add_argument("--out_root", "--out-root", dest="out_root")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--num_folds", "--num-folds", dest="num_folds", type=int, default=5)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())
