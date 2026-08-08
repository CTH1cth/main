#!/usr/bin/env python3
"""Generate GT-free GBSP tri-state V3 masks from existing GBSP caches."""

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
from models.gbsp_threshold_v3 import (  # noqa: E402
    MultiOtsuThreeState,
    OrderedTriGaussianCore,
    TriStateHysteresis,
    UpperTailThreeSegmentCP,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v3.py"
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
    grouped = {dataset: [] for dataset in EXPECTED}
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    selected, cursor = [], 0
    while len(selected) < limit:
        for dataset in EXPECTED:
            if cursor < len(grouped[dataset]):
                selected.append(grouped[dataset][cursor])
                if len(selected) == limit:
                    break
        cursor += 1
    return selected


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _tensor(payload: dict, fields: tuple[str, ...], path: Path) -> torch.Tensor:
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value) and tuple(value.shape) == (1, 37, 37):
            value = value.detach().cpu().float().contiguous()
            if bool(torch.isfinite(value).all()):
                return value
    raise ValueError(f"missing/nonfinite {fields}: {path}")


def _settings(args: argparse.Namespace, cfg, methods: list[str]) -> dict:
    settings = {
        "version": str(cfg.GBSP_V3_VERSION),
        "grid": int(cfg.GBSP_V3_GRID),
        "eps": float(cfg.GBSP_V3_EPS),
        "score_clip": float(cfg.GBSP_V3_SCORE_CLIP),
        "methods": methods,
        "otgc": {
            "anchor_strength": float(cfg.GBSP_V3_OTGC_ANCHOR_STRENGTH),
            "min_mean_gap": float(cfg.GBSP_V3_OTGC_MIN_MEAN_GAP),
            "sigma_floor": float(cfg.GBSP_V3_OTGC_SIGMA_FLOOR),
            "sigma_cap": float(cfg.GBSP_V3_OTGC_SIGMA_CAP),
            "foreground_mixture_floor": float(cfg.GBSP_V3_OTGC_FOREGROUND_MIXTURE_FLOOR),
            "foreground_mixture_cap": float(cfg.GBSP_V3_OTGC_FOREGROUND_MIXTURE_CAP),
            "max_iter": int(cfg.GBSP_V3_OTGC_MAX_ITER),
            "history_size": int(cfg.GBSP_V3_OTGC_HISTORY_SIZE),
            "tolerance_grad": float(cfg.GBSP_V3_OTGC_TOLERANCE_GRAD),
            "tolerance_change": float(cfg.GBSP_V3_OTGC_TOLERANCE_CHANGE),
        },
        "ut3cp": {
            "smooth_window": int(cfg.GBSP_V3_UT3CP_SMOOTH_WINDOW),
            "smooth_polyorder": int(cfg.GBSP_V3_UT3CP_SMOOTH_POLYORDER),
            "min_high_length": int(cfg.GBSP_V3_UT3CP_MIN_HIGH_LENGTH),
            "min_middle_length": int(cfg.GBSP_V3_UT3CP_MIN_MIDDLE_LENGTH),
            "min_background_length": int(cfg.GBSP_V3_UT3CP_MIN_BACKGROUND_LENGTH),
        },
        "save_diagnostics": bool(args.save_diagnostics),
        "source_is_core_cache": bool(args.core_root),
        **dict(cfg.GBSP_V3_INDEPENDENCE_CONTRACT),
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return settings


def _core_result(method: str, score: torch.Tensor, bc: torch.Tensor):
    assert _SETTINGS is not None
    if method == "multi_otsu_3":
        return MultiOtsuThreeState().apply(score, bc)
    if method == "otgc":
        return OrderedTriGaussianCore(
            score_clip=_SETTINGS["score_clip"], **_SETTINGS["otgc"]
        ).apply(score, bc)
    if method == "ut_3cp":
        return UpperTailThreeSegmentCP(eps=_SETTINGS["eps"], **_SETTINGS["ut3cp"]).apply(
            score, bc
        )
    raise ValueError(method)


def _hysteresis_result(method: str, core: dict, score: torch.Tensor) -> dict:
    family = method.removesuffix("_h")
    if family not in core:
        raise KeyError(f"required Core result {family!r} is absent")
    source = core[family]
    high = source["mask_37"].float()
    maps = source.get("continuous_maps_37", {})
    if family == "otgc":
        p0, p2 = maps.get("posterior_c0"), maps.get("posterior_c2")
        if not torch.is_tensor(p0) or not torch.is_tensor(p2):
            raise ValueError("OTGC posterior maps missing from Core cache")
        low = p2 >= p0
    elif family in {"ut_3cp", "multi_otsu_3"}:
        threshold_low = source.get("threshold_low")
        if threshold_low is None:
            raise ValueError(f"{family} low threshold missing from Core cache")
        low = score > float(threshold_low)
    else:
        raise ValueError(method)
    mask, diagnostics = TriStateHysteresis.apply(high, low)
    bc = core["_background_indices"]
    return {
        "method": method,
        "method_family": family,
        "variant": method,
        "mask_37": mask.float().contiguous(),
        "continuous_maps_37": {},
        "threshold_high": source.get("threshold_high"),
        "threshold_low": source.get("threshold_low"),
        "equivalent_minmax_threshold": source.get("threshold_high"),
        "foreground_area": float(mask.mean()),
        "bc_selected_as_foreground_ratio": float(mask.reshape(-1).index_select(0, bc).mean()),
        "empty_mask": bool(float(mask.mean()) == 0.0),
        "area_over_50pct": bool(float(mask.mean()) > 0.5),
        "numerical_failure": False,
        "diagnostics": {**diagnostics, "source_core_method": family},
    }


def _serialize_result(method: str, result, bc: torch.Tensor) -> dict:
    mask = result.mask_core.float().contiguous()
    return {
        "method": method,
        "method_family": method,
        "variant": method,
        "mask_37": mask,
        "continuous_maps_37": {
            key: value.float().contiguous() for key, value in result.continuous_maps.items()
        },
        "threshold_high": result.threshold_high,
        "threshold_low": result.threshold_low,
        "equivalent_minmax_threshold": result.threshold_high,
        "foreground_area": float(mask.mean()),
        "bc_selected_as_foreground_ratio": float(mask.reshape(-1).index_select(0, bc).mean()),
        "empty_mask": bool(float(mask.mean()) == 0.0),
        "area_over_50pct": bool(float(mask.mean()) > 0.5),
        "numerical_failure": bool(result.numerical_failure),
        "diagnostics": dict(result.diagnostics),
    }


def _valid_existing(path: Path, fingerprint: str, methods: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        if payload.get("settings_fingerprint") != fingerprint:
            return False
        results = payload.get("results", {})
        return all(
            method in results
            and torch.is_tensor(results[method].get("mask_37"))
            and tuple(results[method]["mask_37"].shape) == (1, 37, 37)
            for method in methods
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def _process_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output_path = Path(task["output_path"])
    methods = list(_SETTINGS["methods"])
    if _valid_existing(output_path, _SETTINGS["fingerprint"], methods):
        payload = torch_load(output_path, map_location="cpu")
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
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
            raise RuntimeError(f"cache identity mismatch: {source_path}")
        source_is_core = bool(_SETTINGS["source_is_core_cache"])
        if source_is_core:
            raw = _tensor(source, ("raw_residual",), source_path)
            score = _tensor(source, ("minmax_residual",), source_path)
            bc = source.get("background_indices")
            core_results = dict(source.get("results", {}))
            source_gbsp_path = str(source.get("source_gbsp_path", ""))
        else:
            raw = _tensor(source, ("absolute_raw", "gbsp_abs_raw_37"), source_path)
            score = _tensor(source, ("absolute_minmax", "gbsp_abs_minmax_37"), source_path)
            bc = source.get("background_indices")
            core_results = {}
            source_gbsp_path = str(source_path.resolve())
        if not torch.is_tensor(bc) or bc.ndim != 1 or bc.numel() == 0:
            raise ValueError(f"background_indices missing: {source_path}")
        bc = bc.detach().cpu().long().contiguous()
        if torch.unique(bc).numel() != bc.numel():
            raise ValueError(f"background_indices duplicated: {source_path}")
        results = {}
        if not source_is_core:
            required_core = {
                method.removesuffix("_h") for method in methods if method.endswith("_h")
            }
            for family in sorted(required_core):
                core_results[family] = _serialize_result(
                    family, _core_result(family, score, bc), bc
                )
        for method in methods:
            try:
                if method.endswith("_h"):
                    results[method] = _hysteresis_result(
                        method, {**core_results, "_background_indices": bc}, score
                    )
                else:
                    results[method] = _serialize_result(method, _core_result(method, score, bc), bc)
            except Exception as error:
                results[method] = {
                    "method": method,
                    "method_family": method.removesuffix("_h"),
                    "variant": method,
                    "mask_37": torch.zeros(1, 37, 37),
                    "continuous_maps_37": {},
                    "threshold_high": None,
                    "threshold_low": None,
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
        size = source.get("image_size") if source_is_core else source.get("original_image_size")
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(f"original image size missing: {source_path}")
        payload = {
            "version": _SETTINGS["version"],
            "image_id": f"{task['dataset']}/{task['stem']}",
            "dataset": task["dataset"],
            "stem": task["stem"],
            "image_path": task.get("image_path") or source.get("image_path", ""),
            "gt_path": task.get("gt_path") or source.get("gt_path", ""),
            "image_size": (int(size[0]), int(size[1])),
            "patch_grid_size": (37, 37),
            "source_feature_id": source.get("source_feature_id") or source.get("source_feature_path", ""),
            "pca_rank": source.get("pca_rank") if source_is_core else (
                int(source["selected_ranks"].reshape(-1)[0])
                if torch.is_tensor(source.get("selected_ranks")) else None
            ),
            "background_indices": bc,
            "background_candidate_count": int(bc.numel()),
            "raw_residual": raw,
            "minmax_residual": score,
            "results": results,
            "settings": dict(_SETTINGS),
            "settings_fingerprint": _SETTINGS["fingerprint"],
            "source_gbsp_path": source_gbsp_path,
            "source_core_path": str(source_path.resolve()) if source_is_core else None,
            "gt_used_for_generation": False,
            "r1_used_for_generation": False,
            "fixed_058_used_for_generation": False,
            "target_area_prior_used": False,
            "fixed_topk_used": False,
            "dino_forward_used": False,
            "pca_refit_used": False,
            "morphology_used": False,
            "runtime": {
                "total_seconds": time.perf_counter() - started,
                "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            },
        }
        _atomic_save(payload, output_path)
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "skipped": False,
            "numerical_failures": sum(bool(row["numerical_failure"]) for row in results.values()),
            "total_seconds": payload["runtime"]["total_seconds"],
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


def _frozen_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gbsp_threshold_v3_frozen_config":
        raise ValueError(f"unexpected frozen config schema: {path}")
    return payload


def build(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("V3 currently supports the formal test split only")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    core_methods = set(cfg.GBSP_V3_CORE_METHODS)
    hysteresis_methods = set(cfg.GBSP_V3_HYSTERESIS_METHODS)
    methods = list(args.methods or cfg.GBSP_V3_CORE_METHODS)
    unknown = sorted(set(methods) - core_methods - hysteresis_methods)
    if unknown or not methods or len(methods) != len(set(methods)):
        raise ValueError(f"invalid/duplicate V3 methods: {unknown or methods}")
    is_hysteresis = all(method.endswith("_h") for method in methods)

    frozen = None
    frozen_path = _resolve(args.frozen_config) if args.frozen_config else None
    if frozen_path:
        frozen = _frozen_payload(frozen_path)
        allowed = list(frozen.get("selected_methods", []))
        if args.core_root:
            allowed += list(frozen.get("hysteresis_candidates", []))
        if set(methods) - set(allowed):
            raise ValueError(f"methods are not allowed by frozen config: {set(methods) - set(allowed)}")
    if is_hysteresis and args.core_root:
        if frozen is None:
            default_gate = _resolve(Path(cfg.GBSP_V3_OUTPUT_ROOT) / "eval_test20/selected_config.json")
            if not default_gate.is_file():
                raise RuntimeError("Hysteresis is gated; run A1 evaluation before A2")
            frozen_path, frozen = default_gate, _frozen_payload(default_gate)
        allowed_h = set(frozen.get("hysteresis_candidates", []))
        if set(methods) - allowed_h:
            raise RuntimeError(f"Hysteresis trigger was not met for: {set(methods) - allowed_h}")
    elif is_hysteresis and frozen is None:
        raise ValueError("Hysteresis generation requires a frozen gate configuration")

    settings = _settings(args, cfg, methods)
    source_root = _resolve(args.core_root or args.gbsp_root or cfg.GBSP_V3_TEST_ROOT)
    out_root = _resolve(args.out_root or Path(cfg.GBSP_V3_OUTPUT_ROOT) / "full6473")
    if out_root == MAIN_ROOT or MAIN_ROOT in out_root.parents:
        raise ValueError("output root must stay outside the code repository")
    manifest_path = _manifest_path(source_root, args.split)
    rows, mapping = _manifest(manifest_path)
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list))
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample identities missing from source cache: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if args.max_samples >= 0:
        rows = _balanced_subset(rows, args.max_samples) if not args.sample_list else rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no samples selected")
    counts = dict(Counter(str(row["dataset"]) for row in rows))
    if not args.sample_list and args.max_samples < 0 and counts != EXPECTED:
        raise RuntimeError(f"formal test counts differ: {counts}")
    if len(rows) == 20 and not is_hysteresis and not frozen:
        if set(methods) != core_methods or len(methods) != 3:
            raise RuntimeError("stage A1 must run all three frozen Core methods")
    if len(rows) == 20 and is_hysteresis and not args.core_root:
        raise RuntimeError("stage A2 Hysteresis must reuse --core_root from stage A1")
    if len(rows) == 200 and (frozen is None or not 1 <= len(methods) <= 2):
        raise RuntimeError("Pilot200 requires one or two methods from selected_config.json")
    if len(rows) == sum(EXPECTED.values()) and (frozen is None or len(methods) != 1):
        raise RuntimeError("full6473 requires exactly one method from final_config.json")
    if not args.dry_run:
        audit_path = _resolve(cfg.GBSP_V3_BASELINE_AUDIT)
        if not audit_path.is_file():
            raise RuntimeError(f"stage A0 baseline audit is missing: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("baseline_reproduction_status") != "PASS":
            raise RuntimeError("stage A0 fixed-0.58 reproduction has not passed")

    tasks = [
        {
            "dataset": str(row["dataset"]),
            "stem": str(row["stem"]),
            "source_path": row["cache_path"],
            "image_path": row.get("image_path", ""),
            "gt_path": row.get("gt_path", ""),
            "output_path": str(out_root / "test" / str(row["dataset"]) / f"{row['stem']}.pt"),
        }
        for row in rows
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        out_root / "run_config.json",
        {
            "version": settings["version"],
            "created_at": _now(),
            "config": str(_resolve(args.config)),
            "source_root": str(source_root),
            "source_manifest": str(manifest_path),
            "out_root": str(out_root),
            "num_selected": len(tasks),
            "dataset_counts": counts,
            "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
            "frozen_config": str(frozen_path) if frozen_path else None,
            "is_hysteresis": is_hysteresis,
            "settings": settings,
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "num_selected": len(tasks), "methods": methods}, indent=2))
        return
    started, results = time.perf_counter(), []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(settings, args.torch_threads)
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[{_now()}] GBSP threshold V3 {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    valid_keys = {(row["dataset"], row["stem"]) for row in valid}
    manifest_rows = [
        {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": task["output_path"],
            "image_path": task["image_path"],
            "gt_path": task["gt_path"],
            "source_cache_path": task["source_path"],
            "settings_fingerprint": settings["fingerprint"],
        }
        for task in tasks
        if (task["dataset"], task["stem"]) in valid_keys
    ]
    write_json(out_root / "generation_failures.json", failures)
    write_jsonl(out_root / "manifest_test.jsonl", manifest_rows)
    summary = {
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "num_resumed": sum(bool(row.get("skipped")) for row in valid),
        "dataset_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "per_variant_numerical_failure_total": sum(int(row["numerical_failures"]) for row in valid),
        "mean_total_seconds": sum(float(row["total_seconds"]) for row in valid) / max(len(valid), 1),
        "wall_seconds": time.perf_counter() - started,
        **dict(cfg.GBSP_V3_INDEPENDENCE_CONTRACT),
    }
    write_json(out_root / "performance_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"V3 generation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gbsp_root", "--gbsp-root", dest="gbsp_root")
    parser.add_argument("--core_root", "--core-root", dest="core_root")
    parser.add_argument("--out_root", "--out-root", dest="out_root")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--method", action="append", dest="method_single")
    parser.add_argument("--frozen_config", "--frozen-config", dest="frozen_config")
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
