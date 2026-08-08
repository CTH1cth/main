#!/usr/bin/env python3
"""Build image-specific MBSP responses from existing frozen caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.mbsp_reconstruction import MultiBackgroundSubspaceProjector  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_CONFIG = MAIN_ROOT / "configs" / "dinov1_s8_mbsp.py"
VERSION = "mbsp_pca_v1"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
DEFAULT_FEATURE_MANIFEST = REPO_ROOT / "datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl"
DEFAULT_DABE_MANIFEST = (
    REPO_ROOT / "datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8/manifest_test.jsonl"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "workdir/mbsp_pca_v1/dinov1-s8"
DEFAULT_SETTINGS = {
    "MBSP_GRID": 37,
    "MBSP_FEATURE_DIM": 384,
    "MBSP_NUM_SUBSPACES": 4,
    "MBSP_MIN_CLUSTER_SIZE": 16,
    "MBSP_PCA_ENERGY": 0.90,
    "MBSP_PCA_MAX_RANK": 8,
    "MBSP_PCA_MIN_RANK": 1,
    "MBSP_KMEANS_SEED": 0,
    "MBSP_KMEANS_N_INIT": 10,
    "MBSP_KMEANS_MAX_ITER": 100,
    "MBSP_EPS": 1e-8,
}

_WORKER_SETTINGS: dict | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    mapping = {}
    for line, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{line}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"duplicate manifest key: {key}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        mapping[key] = row
    return rows, mapping


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped = {dataset: [] for dataset in DATASETS}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    selected = []
    cursor = 0
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


def _settings_from_args(args, cfg) -> dict:
    def choose(argument, config_name):
        return getattr(cfg, config_name, DEFAULT_SETTINGS[config_name]) if argument is None else argument

    settings = {
        "version": VERSION,
        "grid": int(choose(args.grid, "MBSP_GRID")),
        "feature_dim": int(choose(args.feature_dim, "MBSP_FEATURE_DIM")),
        "num_subspaces": int(choose(args.num_subspaces, "MBSP_NUM_SUBSPACES")),
        "min_cluster_size": int(choose(args.min_cluster_size, "MBSP_MIN_CLUSTER_SIZE")),
        "pca_energy": float(choose(args.pca_energy, "MBSP_PCA_ENERGY")),
        "pca_max_rank": int(choose(args.pca_max_rank, "MBSP_PCA_MAX_RANK")),
        "pca_min_rank": int(choose(args.pca_min_rank, "MBSP_PCA_MIN_RANK")),
        "seed": int(choose(args.seed, "MBSP_KMEANS_SEED")),
        "kmeans_n_init": int(choose(args.kmeans_n_init, "MBSP_KMEANS_N_INIT")),
        "kmeans_max_iter": int(choose(args.kmeans_max_iter, "MBSP_KMEANS_MAX_ITER")),
        "eps": float(choose(args.eps, "MBSP_EPS")),
        "save_per_subspace": bool(args.save_per_subspace),
        "full_bc_source": "cached_bg_anchor_37",
        "bc_confidence_source": "cached_bc_map_37",
        "feature_source": "frozen_identity_dinov1_s8",
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return settings


def _load_tensor(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN or Inf: {path}")
    return value


def _load_inputs(task: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    dataset, stem = task["dataset"], task["stem"]
    feature_path = Path(task["feature_path"])
    dabe_path = Path(task["dabe_path"])
    feature_payload = torch_load(feature_path, map_location="cpu")
    dabe_payload = torch_load(dabe_path, map_location="cpu")
    for payload, path in ((feature_payload, feature_path), (dabe_payload, dabe_path)):
        if not isinstance(payload, dict):
            raise TypeError(f"cache payload must be a dict: {path}")
        if payload.get("dataset") != dataset or payload.get("stem") != stem:
            raise RuntimeError(f"cache identity mismatch: {path}")

    assert _WORKER_SETTINGS is not None
    grid = int(_WORKER_SETTINGS["grid"])
    dim = int(_WORKER_SETTINGS["feature_dim"])
    feature = _load_tensor(feature_payload, "tensor", (dim, grid, grid), feature_path)
    if int(dabe_payload.get("num_views", 1)) != 1:
        raise RuntimeError(f"MBSP requires an identity single-view DABE cache: {dabe_path}")
    augmentations = dabe_payload.get("augs")
    if augmentations not in (None, [], ["identity"], ("identity",)):
        raise RuntimeError(f"MBSP requires identity augmentation only: {dabe_path}")
    anchor = _load_tensor(dabe_payload, "bg_anchor_37", (1, grid, grid), dabe_path)
    confidence = _load_tensor(dabe_payload, "bc_map_37", (1, grid, grid), dabe_path)
    if float(anchor.min()) < -1e-6 or float(anchor.max()) > 1.0 + 1e-6:
        raise ValueError(f"invalid bg_anchor_37 range: {dabe_path}")
    if float(confidence.min()) < -1e-6 or float(confidence.max()) > 1.0 + 1e-6:
        raise ValueError(f"invalid bc_map_37 range: {dabe_path}")
    original_size = feature_payload.get("original_size")
    if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
        raise ValueError(f"invalid original_size: {feature_path}")
    return feature, anchor, confidence, (int(original_size[0]), int(original_size[1]))


def _minmax(value: torch.Tensor, eps: float) -> torch.Tensor:
    minimum = value.min()
    maximum = value.max()
    if float(maximum - minimum) <= eps:
        return torch.zeros_like(value)
    return ((value - minimum) / (maximum - minimum + eps)).clamp(0.0, 1.0)


def _valid_existing(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("mbsp_version") != VERSION:
            return False
        if payload.get("settings_fingerprint") != fingerprint:
            return False
        for field in ("relative_raw", "relative_minmax", "absolute_raw", "absolute_minmax"):
            value = payload.get(field)
            if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
                return False
            if not bool(torch.isfinite(value).all()):
                return False
        return True
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def _generate_one(task: dict) -> dict:
    output_path = Path(task["output_path"])
    assert _WORKER_SETTINGS is not None
    if _valid_existing(output_path, _WORKER_SETTINGS["fingerprint"]):
        payload = torch_load(output_path, map_location="cpu")
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "skipped": True,
            "fallback_flags": payload["fallback_flags"],
            **payload["runtime_summary"],
        }

    started = time.perf_counter()
    try:
        feature, anchor, confidence, original_size = _load_inputs(task)
        grid = int(_WORKER_SETTINGS["grid"])
        dim = int(_WORKER_SETTINGS["feature_dim"])
        flat_feature = feature.permute(1, 2, 0).reshape(grid * grid, dim).contiguous()
        feature_norm = torch.linalg.vector_norm(flat_feature, dim=1)
        if bool((feature_norm <= 0).any()):
            raise ValueError(f"feature cache contains a zero vector: {task['feature_path']}")
        normalized = torch.nn.functional.normalize(flat_feature, p=2, dim=1)

        background_indices = torch.where(anchor.reshape(-1) > 0.5)[0]
        empty_dictionary = int(background_indices.numel()) == 0
        if empty_dictionary:
            background_indices = confidence.reshape(-1).argmax().reshape(1)
        duplicate_count = int(background_indices.numel() - torch.unique(background_indices).numel())
        background = normalized.index_select(0, background_indices)

        projector = MultiBackgroundSubspaceProjector(
            num_subspaces=_WORKER_SETTINGS["num_subspaces"],
            min_cluster_size=_WORKER_SETTINGS["min_cluster_size"],
            pca_energy=_WORKER_SETTINGS["pca_energy"],
            pca_max_rank=_WORKER_SETTINGS["pca_max_rank"],
            pca_min_rank=_WORKER_SETTINGS["pca_min_rank"],
            seed=_WORKER_SETTINGS["seed"],
            eps=_WORKER_SETTINGS["eps"],
            kmeans_n_init=_WORKER_SETTINGS["kmeans_n_init"],
            kmeans_max_iter=_WORKER_SETTINGS["kmeans_max_iter"],
        ).fit(background)
        score = projector.score(normalized)
        diagnostics = projector.diagnostics()
        diagnostics["fallback_flags"]["empty_dictionary_replaced_by_bc_argmax"] = empty_dictionary
        diagnostics["background_duplicate_count"] = duplicate_count

        relative_raw = score.relative_residual.reshape(1, grid, grid)
        absolute_raw = score.absolute_residual.reshape(1, grid, grid)
        relative_minmax = _minmax(relative_raw, _WORKER_SETTINGS["eps"])
        absolute_minmax = _minmax(absolute_raw, _WORKER_SETTINGS["eps"])
        for name, value in (
            ("relative_raw", relative_raw),
            ("relative_minmax", relative_minmax),
            ("absolute_raw", absolute_raw),
            ("absolute_minmax", absolute_minmax),
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{name} contains NaN or Inf")
        if float(relative_raw.min()) < 0.0 or float(relative_raw.max()) > 1.0:
            raise RuntimeError("relative residual escaped [0,1]")

        elapsed = time.perf_counter() - started
        runtime_summary = {
            "num_background_atoms": int(background_indices.numel()),
            "num_effective_subspaces": int(projector.num_effective_subspaces),
            "mean_cluster_size": float(projector.cluster_sizes.float().mean()),
            "mean_selected_rank": float(projector.selected_ranks.float().mean()),
            "kmeans_seconds": float(projector.timing["kmeans_seconds"]),
            "svd_seconds": float(projector.timing["svd_seconds"]),
            "score_seconds": float(projector.timing["score_seconds"]),
            "total_seconds": float(elapsed),
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "gpu_peak_mb": 0.0,
            "fallback_used": bool(any(diagnostics["fallback_flags"].values())),
        }
        payload = {
            "mbsp_version": VERSION,
            "dataset": task["dataset"],
            "stem": task["stem"],
            "settings": {key: value for key, value in _WORKER_SETTINGS.items() if key != "save_per_subspace"},
            "settings_fingerprint": _WORKER_SETTINGS["fingerprint"],
            "relative_raw": relative_raw.float().contiguous(),
            "relative_minmax": relative_minmax.float().contiguous(),
            "absolute_raw": absolute_raw.float().contiguous(),
            "absolute_minmax": absolute_minmax.float().contiguous(),
            "best_subspace_index": score.best_subspace_index.reshape(1, grid, grid),
            "cluster_assignment_background": projector.cluster_assignments,
            "background_indices": background_indices,
            "background_confidence": confidence.reshape(-1).index_select(0, background_indices),
            "cluster_sizes": projector.cluster_sizes,
            "selected_ranks": projector.selected_ranks,
            "singular_values": list(projector.singular_values),
            "subspace_means": projector.subspace_means,
            "representative_background_indices": background_indices.index_select(
                0, projector.representative_background_indices
            ),
            "num_requested_subspaces": int(projector.num_subspaces),
            "num_effective_subspaces": int(projector.num_effective_subspaces),
            "fallback_flags": diagnostics["fallback_flags"],
            "diagnostics": diagnostics,
            "original_image_size": original_size,
            "patch_grid_size": (grid, grid),
            "image_path": task["image_path"],
            "gt_path": task["gt_path"],
            "source_feature_path": task["feature_path"],
            "source_dabe_path": task["dabe_path"],
            "runtime_summary": runtime_summary,
        }
        if _WORKER_SETTINGS["save_per_subspace"]:
            payload["per_subspace_relative"] = score.per_subspace_relative.t().reshape(
                projector.num_effective_subspaces, grid, grid
            )
            payload["per_subspace_absolute"] = score.per_subspace_absolute.t().reshape(
                projector.num_effective_subspaces, grid, grid
            )
        _atomic_save(payload, output_path)
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "skipped": False,
            "fallback_flags": diagnostics["fallback_flags"],
            **runtime_summary,
        }
    except Exception as error:
        return {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "cache_path": str(output_path),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _init_worker(settings: dict, torch_threads: int) -> None:
    global _WORKER_SETTINGS
    _WORKER_SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def _run_config(out_root: Path, settings: dict, args, sample_count: int) -> dict:
    return {
        "version": VERSION,
        "created_at": _now(),
        "config": str(_resolve(args.config)),
        "feature_manifest": str(_resolve(args.feature_manifest)),
        "dabe_manifest": str(_resolve(args.dabe_manifest)),
        "output_root": str(out_root),
        "sample_count_this_run": sample_count,
        "settings": settings,
        "training_used": False,
        "feature_extraction_used": False,
        "gt_used_during_generation": False,
    }


def _summarize(results: list[dict], wall_seconds: float, workers: int) -> dict:
    valid = [row for row in results if "error" not in row]
    fields = {
        "num_background_atoms": "mean_num_background_atoms",
        "num_effective_subspaces": "mean_num_effective_subspaces",
        "mean_cluster_size": "mean_cluster_size",
        "mean_selected_rank": "mean_selected_rank",
        "kmeans_seconds": "mean_kmeans_seconds",
        "svd_seconds": "mean_svd_seconds",
        "score_seconds": "mean_score_seconds",
        "total_seconds": "mean_total_seconds",
        "worker_peak_rss_mb": "mean_worker_peak_rss_mb",
        "gpu_peak_mb": "mean_gpu_peak_mb",
    }
    means = {}
    for source, output in fields.items():
        values = [float(row[source]) for row in valid if row.get(source) is not None]
        means[output] = sum(values) / len(values) if values else None
    estimate = None
    if valid:
        estimate = means["mean_total_seconds"] * EXPECTED_TOTAL
    fallback_counts = defaultdict(int)
    for row in valid:
        for flag, active in row.get("fallback_flags", {}).items():
            fallback_counts[flag] += int(bool(active))
    return {
        "num_requested": len(results),
        "num_valid": len(valid),
        "num_failed": len(results) - len(valid),
        "num_resumed": sum(bool(row.get("skipped")) for row in valid),
        "num_fallback": sum(bool(row.get("fallback_used")) for row in valid),
        "wall_seconds": wall_seconds,
        "estimated_serial_full_seconds": estimate,
        "estimated_full_wall_seconds_at_current_workers": (
            estimate / max(1, int(workers)) if estimate is not None else None
        ),
        "fallback_counts": dict(fallback_counts),
        **means,
    }


def build_cache(args) -> None:
    if args.split != "test":
        raise ValueError("this MBSP task only supports --split test")
    cfg = load_config(_resolve(args.config))
    if args.feature_manifest is None:
        args.feature_manifest = getattr(cfg, "MBSP_FEATURE_MANIFEST", str(DEFAULT_FEATURE_MANIFEST))
    if args.dabe_manifest is None:
        args.dabe_manifest = getattr(cfg, "MBSP_DABE_MANIFEST", str(DEFAULT_DABE_MANIFEST))
    if args.out_root is None:
        args.out_root = getattr(cfg, "MBSP_OUT_ROOT", str(DEFAULT_OUT_ROOT))
    out_root = _resolve(args.out_root)
    if out_root == MAIN_ROOT or MAIN_ROOT in out_root.parents:
        raise ValueError(f"output root must stay outside the code repository: {out_root}")
    settings = _settings_from_args(args, cfg)

    dabe_rows, _ = _manifest(_resolve(args.dabe_manifest))
    _, feature_map = _manifest(_resolve(args.feature_manifest))
    selected_datasets = tuple(args.dataset or DATASETS)
    unknown = sorted(set(selected_datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    rows = [row for row in dabe_rows if row["dataset"] in selected_datasets]
    if args.max_samples < 0 and selected_datasets == DATASETS:
        counts = defaultdict(int)
        for row in rows:
            counts[row["dataset"]] += 1
        if len(rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
            raise RuntimeError(f"formal manifest counts differ: {len(rows)}, {dict(counts)}")
    rows = _balanced_subset(rows, args.max_samples)
    if not rows:
        raise RuntimeError("no samples selected")

    tasks = []
    for row in rows:
        key = (row["dataset"], row["stem"])
        if key not in feature_map:
            raise KeyError(f"feature cache missing for {key}")
        feature_row = feature_map[key]
        for field in ("image_path", "gt_path"):
            if field not in row or not Path(row[field]).is_file():
                raise FileNotFoundError(row.get(field, f"missing {field} for {key}"))
        tasks.append(
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "image_path": row["image_path"],
                "gt_path": row["gt_path"],
                "feature_path": feature_row["cache_path"],
                "dabe_path": row["cache_path"],
                "output_path": str(out_root / "test" / row["dataset"] / f"{row['stem']}.pt"),
            }
        )

    previous_config_path = out_root / "run_config.json"
    if previous_config_path.is_file():
        previous = json.loads(previous_config_path.read_text(encoding="utf-8"))
        previous_fingerprint = previous.get("settings", {}).get("fingerprint")
        if previous_fingerprint not in (None, settings["fingerprint"]):
            raise RuntimeError(
                "output root already contains a different MBSP configuration; use a new directory"
            )
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(previous_config_path, _run_config(out_root, settings, args, len(tasks)))
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "num_samples": len(tasks), "settings": settings}, indent=2))
        return

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(settings, args.torch_threads),
    ) as pool:
        for index, result in enumerate(pool.map(_generate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[{_now()}] MBSP {index}/{len(tasks)}", flush=True)
    wall_seconds = time.perf_counter() - started
    failures = [row for row in results if "error" in row]
    write_json(out_root / "generation_failures.json", failures)

    valid_by_key = {(row["dataset"], row["stem"]): row for row in results if "error" not in row}
    manifest_rows = []
    for task in tasks:
        key = (task["dataset"], task["stem"])
        if key not in valid_by_key:
            continue
        manifest_rows.append(
            {
                "dataset": task["dataset"],
                "stem": task["stem"],
                "cache_path": task["output_path"],
                "image_path": task["image_path"],
                "gt_path": task["gt_path"],
                "source_feature_path": task["feature_path"],
                "source_dabe_path": task["dabe_path"],
                "settings_fingerprint": settings["fingerprint"],
            }
        )
    write_jsonl(out_root / "manifest_test.jsonl", manifest_rows)
    summary = _summarize(results, wall_seconds, args.workers)
    write_json(out_root / "performance_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        print(
            f"[Warn] {len(failures)} samples were recorded in generation_failures.json; "
            "successful samples were kept.",
            flush=True,
        )
        if args.failure_policy == "strict":
            raise RuntimeError(f"MBSP generation failed for {len(failures)} samples")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-root", "--out_root", dest="out_root")
    parser.add_argument("--feature-manifest", "--feature_manifest", dest="feature_manifest")
    parser.add_argument("--dabe-manifest", "--dabe_manifest", dest="dabe_manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--feature-dim", "--feature_dim", dest="feature_dim", type=int)
    parser.add_argument("--num-subspaces", "--num_subspaces", dest="num_subspaces", type=int)
    parser.add_argument("--min-cluster-size", "--min_cluster_size", dest="min_cluster_size", type=int)
    parser.add_argument("--pca-energy", "--pca_energy", dest="pca_energy", type=float)
    parser.add_argument("--pca-max-rank", "--pca_max_rank", dest="pca_max_rank", type=int)
    parser.add_argument("--pca-min-rank", "--pca_min_rank", dest="pca_min_rank", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--kmeans-n-init", "--kmeans_n_init", dest="kmeans_n_init", type=int)
    parser.add_argument("--kmeans-max-iter", "--kmeans_max_iter", dest="kmeans_max_iter", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--save-per-subspace", "--save_per_subspace", dest="save_per_subspace", action="store_true")
    parser.add_argument("--save-raw", "--save_raw", action="store_true", help="compatibility flag; raw responses are always saved")
    parser.add_argument("--save-minmax", "--save_minmax", action="store_true", help="compatibility flag; calibrated responses are always saved")
    parser.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    build_cache(build_parser().parse_args())
