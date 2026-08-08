#!/usr/bin/env python3
"""Generate GT-free GBSP Adaptive Thresholding V2 masks from existing caches."""

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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_adaptive_threshold_v2 import (  # noqa: E402
    BCBetaMixtureCalibrator,
    BCLogitGaussianMixtureCalibrator,
    ExpandedBackgroundEVTCalibrator,
    QueryDistributionChangePoint,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v2.py"
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
_SETTINGS: dict | None = None


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    mapping = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        source = Path(str(row.get("cache_path", "")))
        if not all(key) or key in mapping or not source.is_file():
            raise RuntimeError(f"invalid source manifest row {path}:{line}: {key}")
        mapping[key] = row
    return rows, mapping


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
        raise RuntimeError(f"duplicate identities in {path}")
    return keys


def _balanced_subset(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0 or limit >= len(rows):
        return rows
    datasets = tuple(EXPECTED)
    grouped = {dataset: [] for dataset in datasets}
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    selected, cursor = [], 0
    while len(selected) < limit:
        for dataset in datasets:
            if cursor < len(grouped[dataset]):
                selected.append(grouped[dataset][cursor])
                if len(selected) == limit:
                    break
        cursor += 1
    return selected


def _variant_name(method: str, value: float | None = None) -> str:
    if method in {"ba_bmc", "ba_lgmc"}:
        return f"{method}_eta{int(round(float(value) * 100)):03d}"
    if method == "eb_evt":
        return f"eb_evt_q{int(round(float(value) * 1000)):04d}"
    return method


def _variants(args: argparse.Namespace, cfg) -> list[dict]:
    if args.frozen_method_config:
        payload = json.loads(_resolve(args.frozen_method_config).read_text(encoding="utf-8"))
        variants = payload.get("selected_variants") or payload.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("frozen config must contain selected_variants")
        return variants
    methods = args.methods or list(cfg.GBSP_V2_METHODS)
    unknown = sorted(set(methods) - set(cfg.GBSP_V2_METHODS))
    if unknown:
        raise ValueError(f"unknown V2 methods: {unknown}")
    variants = []
    for method in methods:
        if method in {"ba_bmc", "ba_lgmc"}:
            for strength in args.anchor_strengths:
                if float(strength) not in {0.5, 1.0}:
                    raise ValueError("anchor strengths are frozen to 0.5 and 1.0")
                variants.append(
                    {"variant": _variant_name(method, strength), "method": method, "anchor_strength": float(strength)}
                )
        elif method == "eb_evt":
            for q in args.evt_q_values:
                if float(q) not in {0.01, 0.025, 0.05}:
                    raise ValueError("EVT q values are frozen to 0.01, 0.025 and 0.05")
                variants.append({"variant": _variant_name(method, q), "method": method, "evt_q": float(q)})
        else:
            variants.append({"variant": method, "method": method})
    names = [row["variant"] for row in variants]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate V2 variants")
    return variants


def _settings(args: argparse.Namespace, cfg, variants: list[dict]) -> dict:
    settings = {
        "version": str(cfg.GBSP_V2_VERSION),
        "grid": int(cfg.GBSP_V2_GRID),
        "eps": float(cfg.GBSP_V2_EPS),
        "score_clip": float(cfg.GBSP_V2_SCORE_CLIP),
        "variants": variants,
        "bmc_max_iter": int(cfg.GBSP_V2_BMC_MAX_ITER),
        "lgmc_max_iter": int(cfg.GBSP_V2_LGMC_MAX_ITER),
        "tolerance": float(cfg.GBSP_V2_OPT_TOLERANCE),
        "evt_tail_quantile": float(cfg.GBSP_V2_EVT_TAIL_START_QUANTILE),
        "evt_min_tail_count": int(cfg.GBSP_V2_EVT_MIN_TAIL_COUNT),
        "qdcp_min_segment": int(cfg.GBSP_V2_QDCP_MIN_SEGMENT),
        "qdcp_savgol_window": int(cfg.GBSP_V2_QDCP_SAVGOL_WINDOW),
        "qdcp_savgol_order": int(cfg.GBSP_V2_QDCP_SAVGOL_ORDER),
        "save_diagnostics": bool(args.save_diagnostics),
        "gt_used": False,
        "r1_used": False,
        "fixed_058_used": False,
        "target_area_prior_used": False,
        "dino_forward_used": False,
        "pca_refit_used": False,
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return settings


def _tensor(payload: dict, fields: tuple[str, ...], shape: tuple[int, ...], path: Path) -> torch.Tensor:
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value):
            value = value.detach().cpu().float().contiguous()
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid {field} in {path}")
            return value
    raise KeyError(f"none of {fields} found in {path}")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _valid_existing(path: Path, fingerprint: str, variants: list[dict]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        if payload.get("settings_fingerprint") != fingerprint:
            return False
        results = payload.get("results", {})
        return all(
            name in results
            and torch.is_tensor(results[name].get("mask_37"))
            and tuple(results[name]["mask_37"].shape) == (1, 37, 37)
            for name in (row["variant"] for row in variants)
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def _method_result(variant: dict, raw: torch.Tensor, score: torch.Tensor, bc: torch.Tensor):
    method = variant["method"]
    assert _SETTINGS is not None
    if method == "ba_bmc":
        return BCBetaMixtureCalibrator(
            anchor_strength=float(variant["anchor_strength"]),
            score_clip=_SETTINGS["score_clip"],
            max_iter=_SETTINGS["bmc_max_iter"],
            tolerance=_SETTINGS["tolerance"],
        ).apply(score, bc)
    if method == "ba_lgmc":
        return BCLogitGaussianMixtureCalibrator(
            anchor_strength=float(variant["anchor_strength"]),
            score_clip=_SETTINGS["score_clip"],
            max_iter=_SETTINGS["lgmc_max_iter"],
            tolerance=_SETTINGS["tolerance"],
        ).apply(score, bc)
    if method == "eb_evt":
        return ExpandedBackgroundEVTCalibrator(
            evt_q=float(variant["evt_q"]),
            reference_quantile=_SETTINGS["evt_tail_quantile"],
            min_tail_count=_SETTINGS["evt_min_tail_count"],
            eps=_SETTINGS["eps"],
        ).apply(raw, score, bc)
    if method == "qdcp_pl":
        return QueryDistributionChangePoint(
            method="pl", min_segment=_SETTINGS["qdcp_min_segment"], score_clip=_SETTINGS["score_clip"]
        ).apply(score)
    if method == "qdcp_k":
        return QueryDistributionChangePoint(
            method="kneedle",
            savgol_window=_SETTINGS["qdcp_savgol_window"],
            savgol_order=_SETTINGS["qdcp_savgol_order"],
            score_clip=_SETTINGS["score_clip"],
        ).apply(score)
    raise ValueError(method)


def _process_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output_path = Path(task["output_path"])
    variants = _SETTINGS["variants"]
    if _valid_existing(output_path, _SETTINGS["fingerprint"], variants):
        payload = torch_load(output_path, map_location="cpu")
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output_path),
            "skipped": True,
            "numerical_failures": sum(bool(row["numerical_failure"]) for row in payload["results"].values()),
            "total_seconds": 0.0,
        }
    started = time.perf_counter()
    try:
        source_path = Path(task["source_path"])
        source = torch_load(source_path, map_location="cpu")
        if not isinstance(source, dict) or (
            str(source.get("dataset")), str(source.get("stem"))
        ) != (task["dataset"], task["stem"]):
            raise RuntimeError(f"GBSP cache identity mismatch: {source_path}")
        raw = _tensor(source, ("absolute_raw", "gbsp_abs_raw_37"), (1, 37, 37), source_path)
        score = _tensor(source, ("absolute_minmax", "gbsp_abs_minmax_37"), (1, 37, 37), source_path)
        bc = source.get("background_indices")
        if not torch.is_tensor(bc) or bc.ndim != 1 or bc.numel() == 0:
            raise ValueError(f"background_indices missing: {source_path}")
        bc = bc.detach().cpu().long().contiguous()
        if torch.unique(bc).numel() != bc.numel():
            raise ValueError(f"background_indices duplicated: {source_path}")
        results = {}
        for variant in variants:
            name = variant["variant"]
            try:
                result = _method_result(variant, raw, score, bc)
                mask = result.binary_mask.float().contiguous()
                diagnostics = dict(result.diagnostics)
                results[name] = {
                    "method": variant["method"],
                    "variant": name,
                    "parameters": dict(variant),
                    "mask_37": mask,
                    "continuous_map_37": result.continuous_map,
                    "equivalent_minmax_threshold": result.equivalent_minmax_threshold,
                    "foreground_area": float(mask.mean()),
                    "bc_selected_as_foreground_ratio": float(mask.reshape(-1).index_select(0, bc).mean()),
                    "empty_mask": bool(float(mask.mean()) == 0.0),
                    "area_over_50pct": bool(float(mask.mean()) > 0.5),
                    "numerical_failure": bool(result.numerical_failure),
                    "diagnostics": diagnostics,
                }
            except Exception as error:
                results[name] = {
                    "method": variant["method"],
                    "variant": name,
                    "parameters": dict(variant),
                    "mask_37": torch.zeros(1, 37, 37),
                    "continuous_map_37": None,
                    "equivalent_minmax_threshold": None,
                    "foreground_area": 0.0,
                    "bc_selected_as_foreground_ratio": 0.0,
                    "empty_mask": True,
                    "area_over_50pct": False,
                    "numerical_failure": True,
                    "diagnostics": {
                        "failure_reason": "uncaught_method_exception",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                }
        size = source.get("original_image_size")
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(f"original_image_size missing: {source_path}")
        payload = {
            "version": _SETTINGS["version"],
            "image_id": f"{task['dataset']}/{task['stem']}",
            "dataset": task["dataset"],
            "stem": task["stem"],
            "image_path": task.get("image_path") or source.get("image_path", ""),
            "gt_path": task.get("gt_path") or source.get("gt_path", ""),
            "image_size": (int(size[0]), int(size[1])),
            "patch_grid_size": (37, 37),
            "source_feature_id": source.get("source_feature_path", ""),
            "pca_rank": int(source["selected_ranks"].reshape(-1)[0]) if torch.is_tensor(source.get("selected_ranks")) else None,
            "background_indices": bc,
            "background_candidate_count": int(bc.numel()),
            "raw_residual": raw,
            "minmax_residual": score,
            "results": results,
            "settings": dict(_SETTINGS),
            "settings_fingerprint": _SETTINGS["fingerprint"],
            "source_gbsp_path": str(source_path.resolve()),
            "gt_used_for_generation": False,
            "r1_used_for_generation": False,
            "fixed_058_used_for_generation": False,
            "target_area_prior_used": False,
            "dino_forward_used": False,
            "pca_refit_used": False,
            "runtime": {
                "total_seconds": time.perf_counter() - started,
                "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            },
        }
        _atomic_save(payload, output_path)
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output_path),
            "skipped": False,
            "numerical_failures": sum(bool(row["numerical_failure"]) for row in results.values()),
            "total_seconds": payload["runtime"]["total_seconds"],
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "cache_path": str(output_path), "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(settings: dict, torch_threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def build(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("V2 currently supports the formal test split only")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    variants = _variants(args, cfg)
    settings = _settings(args, cfg, variants)
    gbsp_root = _resolve(args.gbsp_root or cfg.GBSP_V2_TEST_ROOT)
    out_root = _resolve(args.out_root or Path(cfg.GBSP_V2_OUTPUT_ROOT) / "full6473")
    if out_root == MAIN_ROOT or MAIN_ROOT in out_root.parents:
        raise ValueError("output root must stay outside the code repository")
    manifest_path = _manifest_path(gbsp_root, args.split)
    rows, mapping = _manifest(manifest_path)
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list))
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample identities missing from GBSP cache: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if args.max_samples >= 0:
        rows = _balanced_subset(rows, args.max_samples) if not args.sample_list else rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no GBSP samples selected")
    if not args.sample_list and args.max_samples < 0:
        counts = Counter(str(row["dataset"]) for row in rows)
        if dict(counts) != EXPECTED:
            raise RuntimeError(f"formal test counts differ: {dict(counts)}")
    if len(rows) == 20 and not args.frozen_method_config:
        families = {row["method"] for row in variants}
        if families != set(cfg.GBSP_V2_METHODS) or len(variants) != 9:
            raise RuntimeError(
                "stage A1 must run all five declared method entries and exactly nine variants"
            )
    if len(rows) == 200:
        if not args.frozen_method_config or len(variants) != 2:
            raise RuntimeError("Pilot200 requires exactly two variants from frozen_method_config")
    if len(rows) == sum(EXPECTED.values()):
        if not args.frozen_method_config or len(variants) != 1:
            raise RuntimeError("full6473 requires exactly one frozen final variant")
    if not args.dry_run:
        baseline_audit_path = _resolve(cfg.GBSP_V2_BASELINE_AUDIT)
        if not baseline_audit_path.is_file():
            raise RuntimeError(
                f"stage A0 baseline audit is missing: {baseline_audit_path}"
            )
        baseline_audit = json.loads(baseline_audit_path.read_text(encoding="utf-8"))
        if baseline_audit.get("baseline_reproduction_status") != "PASS":
            raise RuntimeError("stage A0 fixed-0.58 reproduction has not passed")
    tasks = []
    for row in rows:
        dataset, stem = str(row["dataset"]), str(row["stem"])
        tasks.append(
            {
                "dataset": dataset, "stem": stem, "source_path": row["cache_path"],
                "image_path": row.get("image_path", ""), "gt_path": row.get("gt_path", ""),
                "output_path": str(out_root / "test" / dataset / f"{stem}.pt"),
            }
        )
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "run_config.json",
        {
            "version": settings["version"], "created_at": _now(), "config": str(_resolve(args.config)),
            "split": args.split, "gbsp_root": str(gbsp_root), "gbsp_manifest": str(manifest_path),
            "out_root": str(out_root), "num_selected": len(tasks),
            "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
            "settings": settings,
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "num_selected": len(tasks), "variants": variants}, indent=2))
        return
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(settings, args.torch_threads)
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[{_now()}] GBSP threshold V2 {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    valid_keys = {(row["dataset"], row["stem"]) for row in valid}
    manifest_rows = [
        {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": task["output_path"],
            "image_path": task["image_path"], "gt_path": task["gt_path"],
            "source_gbsp_path": task["source_path"], "settings_fingerprint": settings["fingerprint"],
        }
        for task in tasks if (task["dataset"], task["stem"]) in valid_keys
    ]
    write_json(out_root / "generation_failures.json", failures)
    write_jsonl(out_root / "manifest_test.jsonl", manifest_rows)
    summary = {
        "num_requested": len(tasks), "num_valid": len(valid), "num_failed": len(failures),
        "num_resumed": sum(bool(row.get("skipped")) for row in valid),
        "dataset_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "per_variant_numerical_failure_total": sum(int(row["numerical_failures"]) for row in valid),
        "mean_total_seconds": sum(float(row["total_seconds"]) for row in valid) / max(len(valid), 1),
        "wall_seconds": time.perf_counter() - started,
        "gt_used_for_generation": False, "r1_used_for_generation": False,
        "fixed_058_used_for_generation": False, "dino_forward_used": False, "pca_refit_used": False,
    }
    write_json(out_root / "performance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"V2 generation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gbsp_root", "--gbsp-root", dest="gbsp_root")
    parser.add_argument("--out_root", "--out-root", dest="out_root")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--method", action="append", dest="method_single")
    parser.add_argument("--anchor_strengths", "--anchor-strengths", nargs="+", type=float, default=(0.5, 1.0))
    parser.add_argument("--evt_q_values", "--evt-q-values", nargs="+", type=float, default=(0.01, 0.025, 0.05))
    parser.add_argument("--frozen_method_config", "--frozen-method-config", dest="frozen_method_config")
    parser.add_argument("--save_diagnostics", "--save-diagnostics", action="store_true")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.method_single:
        parsed.methods = (parsed.methods or []) + parsed.method_single
    build(parsed)
