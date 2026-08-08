#!/usr/bin/env python3
"""Build GT-free PCA-SPE calibration caches from frozen single-global GBSP."""

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
from models.gbsp_spe_calibration import (  # noqa: E402
    BoundedResidualCalibrator,
    GammaSPELimit,
    JacksonMudholkarSPELimit,
    PCASpectrumExtractor,
    equivalent_minmax_threshold,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_spe95.py"
ALLOWED_METHODS = {"jm_spe", "gamma_spe"}
ALLOWED_LEVELS = {0.90, 0.95, 0.99}
_SETTINGS: dict | None = None


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _manifest_path(root: Path) -> Path:
    path = root / "manifest_test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    mapping: dict[tuple[str, str], dict] = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        cache = Path(str(row.get("cache_path", "")))
        if not all(key) or key in mapping or not cache.is_file():
            raise RuntimeError(f"invalid GBSP manifest row {path}:{line}: {key}")
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
            raise ValueError(f"invalid identity {path}:{line}: {raw!r}")
        keys.append((parts[0], parts[1]))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate identities in {path}")
    return keys


def _balanced(rows: list[dict], limit: int, datasets: tuple[str, ...]) -> list[dict]:
    if limit < 0 or limit >= len(rows):
        return rows
    groups = {name: [] for name in datasets}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
    output, cursor = [], 0
    while len(output) < limit:
        progressed = False
        for name in datasets:
            if cursor < len(groups.get(name, ())):
                output.append(groups[name][cursor])
                progressed = True
                if len(output) == limit:
                    break
        if not progressed:
            break
        cursor += 1
    return output


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _raw_map(payload: dict, path: Path) -> torch.Tensor:
    value = payload.get("absolute_raw")
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"absolute_raw must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()) or float(value.min()) < 0.0:
        raise ValueError(f"absolute_raw must be finite and non-negative: {path}")
    return value


def _formal_source(payload: dict, path: Path) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    assert _SETTINGS is not None
    if int(payload.get("num_requested_subspaces", -1)) != 1 or int(payload.get("num_effective_subspaces", -1)) != 1:
        raise RuntimeError(f"SPE requires formal single-global GBSP: {path}")
    settings = payload.get("settings", {})
    formal = _SETTINGS["formal_gbsp"]
    for key, expected in formal.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"formal GBSP setting differs ({key}): {actual} != {expected}: {path}")
        elif actual != expected:
            raise RuntimeError(f"formal GBSP setting differs ({key}): {actual!r} != {expected!r}: {path}")
    ranks = payload.get("selected_ranks")
    singular = payload.get("singular_values")
    indices = payload.get("background_indices")
    if not torch.is_tensor(ranks) or ranks.numel() != 1:
        raise ValueError(f"selected_ranks missing: {path}")
    if not isinstance(singular, (tuple, list)) or len(singular) != 1 or not torch.is_tensor(singular[0]):
        raise ValueError(f"single-global singular_values missing: {path}")
    if not torch.is_tensor(indices) or indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(f"background_indices missing: {path}")
    cluster_sizes = payload.get("cluster_sizes")
    if not torch.is_tensor(cluster_sizes) or cluster_sizes.numel() != 1 or int(cluster_sizes[0]) != int(indices.numel()):
        raise RuntimeError(f"cluster/background count mismatch: {path}")
    feature_dimension = int(settings.get("feature_dim", _SETTINGS["feature_dimension"]))
    return _raw_map(payload, path), singular[0], int(indices.numel()), feature_dimension, int(ranks[0])


def _level_tag(level: float) -> str:
    return f"{int(round(float(level) * 100)):03d}"


def _serialize_tensor(value: torch.Tensor | None, *, binary: bool = False) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().to(torch.uint8 if binary else torch.float32).contiguous()


def _process(task: dict) -> dict:
    assert _SETTINGS is not None
    started = time.perf_counter()
    output = Path(task["output_path"])
    try:
        source_path = Path(task["source_path"])
        source = torch_load(source_path, map_location="cpu")
        identity = (task["dataset"], task["stem"])
        if not isinstance(source, dict) or (str(source.get("dataset")), str(source.get("stem"))) != identity:
            raise RuntimeError(f"GBSP cache identity mismatch: {source_path}")
        raw, singular, background_count, feature_dimension, rank = _formal_source(source, source_path)
        spectrum = PCASpectrumExtractor().extract(singular, background_count, feature_dimension, rank)
        calibrator = BoundedResidualCalibrator(_SETTINGS["eps"])

        jm_limits = {
            float(level): JacksonMudholkarSPELimit().apply(spectrum, float(level))
            for level in _SETTINGS["control_levels"]
        }
        gamma_limit = GammaSPELimit().apply(spectrum, _SETTINGS["primary_level"])
        jm_results = {level: calibrator.apply(raw, limit) for level, limit in jm_limits.items()}
        gamma_result = calibrator.apply(raw, gamma_limit)
        primary = jm_results[_SETTINGS["primary_level"]]
        background_indices = source["background_indices"].detach().cpu().long().contiguous()

        payload = {
            "version": _SETTINGS["version"],
            "dataset": identity[0],
            "stem": identity[1],
            "image_id": f"{identity[0]}/{identity[1]}",
            "image_path": task.get("image_path") or source.get("image_path", ""),
            "gt_path": task.get("gt_path") or source.get("gt_path", ""),
            "original_image_size": tuple(source.get("original_image_size", ())),
            "patch_grid_size": (37, 37),
            "background_candidate_count": background_count,
            "feature_dimension": feature_dimension,
            "effective_spectrum_dimension": spectrum.effective_spectrum_dimension,
            "pca_rank": rank,
            "singular_values": spectrum.singular_values.float().contiguous(),
            "covariance_eigenvalues": spectrum.covariance_eigenvalues.float().contiguous(),
            "retained_eigenvalues": spectrum.retained_eigenvalues.float().contiguous(),
            "discarded_eigenvalues": spectrum.discarded_eigenvalues.float().contiguous(),
            "theta1": spectrum.theta1,
            "theta2": spectrum.theta2,
            "theta3": spectrum.theta3,
            "h0": jm_limits[_SETTINGS["primary_level"]].h0,
            "jm_bracket": jm_limits[_SETTINGS["primary_level"]].bracket,
            "gamma_shape": gamma_limit.gamma_shape,
            "gamma_scale": gamma_limit.gamma_scale,
            "raw_residual_map": raw.float().contiguous(),
            "background_indices": background_indices,
            "source_gbsp_path": str(source_path.resolve()),
            "source_feature_path": str(source.get("source_feature_path", "")),
            "source_dabe_path": str(source.get("source_dabe_path", "")),
            "settings": dict(_SETTINGS),
            "settings_fingerprint": _SETTINGS["fingerprint"],
            "numerical_validity": {
                "spectrum": spectrum.numerical_validity,
                **{f"jm_{_level_tag(level)}": result.numerical_validity for level, result in jm_results.items()},
                "gamma_095": gamma_result.numerical_validity,
            },
            "failure_reason": {
                "spectrum": spectrum.failure_reason,
                **{f"jm_{_level_tag(level)}": result.failure_reason for level, result in jm_results.items()},
                "gamma_095": gamma_result.failure_reason,
            },
            **{f"{key}_for_generation": value for key, value in _SETTINGS["independence"].items()},
        }
        for level, limit in jm_limits.items():
            tag = _level_tag(level)
            result = jm_results[level]
            payload[f"jm_tau_{tag}"] = limit.tau
            payload[f"jm_h0_{tag}"] = limit.h0
            payload[f"jm_bracket_{tag}"] = limit.bracket
            payload[f"jm_calibrated_map_{tag}"] = _serialize_tensor(result.score)
            payload[f"jm_mask_{tag}"] = _serialize_tensor(result.mask, binary=True)
            payload[f"equivalent_minmax_threshold_jm{tag}"] = (
                equivalent_minmax_threshold(raw, float(limit.tau), _SETTINGS["eps"])
                if limit.numerical_validity and limit.tau is not None else None
            )
            payload[f"foreground_area_jm{tag}"] = float(result.mask.float().mean()) if result.mask is not None else None
        payload["gamma_tau_095"] = gamma_limit.tau
        payload["gamma_calibrated_map_095"] = _serialize_tensor(gamma_result.score)
        payload["gamma_mask_095"] = _serialize_tensor(gamma_result.mask, binary=True)
        payload["equivalent_minmax_threshold_gamma095"] = (
            equivalent_minmax_threshold(raw, float(gamma_limit.tau), _SETTINGS["eps"])
            if gamma_limit.numerical_validity and gamma_limit.tau is not None else None
        )
        payload["foreground_area_gamma095"] = (
            float(gamma_result.mask.float().mean()) if gamma_result.mask is not None else None
        )
        if primary.mask is not None:
            flat = primary.mask.reshape(-1)
            payload["background_candidate_exceedance_rate"] = float(
                flat.index_select(0, background_indices).float().mean()
            )
        else:
            payload["background_candidate_exceedance_rate"] = None
        if gamma_result.mask is not None:
            payload["background_candidate_exceedance_rate_gamma095"] = float(
                gamma_result.mask.reshape(-1).index_select(0, background_indices).float().mean()
            )
        else:
            payload["background_candidate_exceedance_rate_gamma095"] = None
        payload["runtime"] = {
            "total_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        _atomic_save(payload, output)
        return {
            "dataset": identity[0], "stem": identity[1], "cache_path": str(output),
            "jm_valid": int(primary.numerical_validity),
            "gamma_valid": int(gamma_result.numerical_validity),
            "failure_reason": primary.failure_reason,
            "total_seconds": payload["runtime"]["total_seconds"],
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init(settings: dict, torch_threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def build(args) -> None:
    cfg = load_config(_resolve(args.config))
    methods = tuple(args.methods or cfg.GBSP_SPE_METHODS)
    levels = tuple(float(value) for value in (args.control_levels or cfg.GBSP_SPE_CONTROL_LEVELS))
    if not methods or len(methods) != len(set(methods)) or set(methods) - ALLOWED_METHODS:
        raise ValueError(f"invalid methods: {methods}")
    if 0.95 not in levels or set(levels) - ALLOWED_LEVELS or len(levels) != len(set(levels)):
        raise ValueError("control_levels must be a unique subset of {0.90,0.95,0.99} containing 0.95")
    settings = {
        "version": cfg.GBSP_SPE_VERSION,
        "methods": methods,
        "control_levels": levels,
        "primary_level": float(cfg.GBSP_SPE_PRIMARY_CONTROL_LEVEL),
        "feature_dimension": int(cfg.GBSP_SPE_FEATURE_DIM),
        "eps": float(cfg.GBSP_SPE_EPS),
        "formal_gbsp": dict(cfg.GBSP_SPE_FORMAL_GBSP),
        "independence": dict(cfg.GBSP_SPE_INDEPENDENCE),
        "save_diagnostics": bool(args.save_diagnostics),
    }
    settings["fingerprint"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    root = _resolve(args.gbsp_root or cfg.GBSP_SPE_GBSP_ROOT)
    out = _resolve(args.out_root or Path(cfg.GBSP_SPE_OUTPUT_ROOT) / "test20")
    if out == MAIN_ROOT or MAIN_ROOT in out.parents:
        raise ValueError("output must be outside the source repository")
    rows, mapping = _manifest(_manifest_path(root))
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list))
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"sample identities missing from GBSP manifest: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    elif args.max_samples >= 0:
        rows = _balanced(rows, args.max_samples, tuple(cfg.GBSP_SPE_EXPECTED_COUNTS))
    if args.sample_list and args.max_samples >= 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no samples selected")

    tasks = []
    for row in rows:
        dataset, stem = str(row["dataset"]), str(row["stem"])
        tasks.append({
            "dataset": dataset,
            "stem": stem,
            "source_path": row["cache_path"],
            "image_path": row.get("image_path", ""),
            "gt_path": row.get("gt_path", ""),
            "output_path": str(out / "test" / dataset / f"{stem}.pt"),
        })
    counts = dict(Counter(task["dataset"] for task in tasks))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gbsp_root": str(root),
        "gbsp_manifest": str(_manifest_path(root)),
        "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
        "num_selected": len(tasks),
        "dataset_counts": counts,
        "out_root": str(out),
        "settings": settings,
    })
    if args.dry_run:
        print(json.dumps({"num_selected": len(tasks), "dataset_counts": counts, "out_root": str(out)}, indent=2))
        return

    results = []
    with ProcessPoolExecutor(
        max_workers=int(args.workers), initializer=_init, initargs=(settings, args.torch_threads)
    ) as pool:
        for index, result in enumerate(pool.map(_process, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP-SPE cache {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    write_json(out / "generation_failures.json", failures)
    write_json(out / "generation_summary.json", {
        "num_requested": len(tasks), "num_generated": len(valid), "generation_failed": len(failures),
        "jm_valid": sum(int(row["jm_valid"]) for row in valid),
        "jm_invalid": sum(1 - int(row["jm_valid"]) for row in valid),
        "gamma_valid": sum(int(row["gamma_valid"]) for row in valid),
        "dataset_counts": counts,
        "failure_reasons": dict(Counter(str(row.get("failure_reason")) for row in valid if row.get("failure_reason"))),
    })
    manifest_rows = []
    task_lookup = {(task["dataset"], task["stem"]): task for task in tasks}
    for row in valid:
        task = task_lookup[(row["dataset"], row["stem"])]
        manifest_rows.append({
            "dataset": row["dataset"], "stem": row["stem"], "cache_path": row["cache_path"],
            "image_path": task["image_path"], "gt_path": task["gt_path"],
            "source_gbsp_path": task["source_path"],
        })
    write_jsonl(out / "manifest_test.jsonl", manifest_rows)
    print(json.dumps({"num_requested": len(tasks), "num_generated": len(valid), "generation_failed": len(failures)}, indent=2))
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} GBSP-SPE cache failures")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp_root")
    value.add_argument("--sample_list")
    value.add_argument("--split", default="test", choices=("test",))
    value.add_argument("--max_samples", type=int, default=-1)
    value.add_argument("--methods", nargs="+")
    value.add_argument("--control_levels", nargs="+", type=float)
    value.add_argument("--save_diagnostics", action="store_true")
    value.add_argument("--out_root")
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--torch_threads", type=int, default=1)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    value.add_argument("--dry_run", action="store_true")
    return value


if __name__ == "__main__":
    build(parser().parse_args())

