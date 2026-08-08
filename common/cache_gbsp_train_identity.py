#!/usr/bin/env python3
"""Build identity-only GBSP training targets from frozen DINO features."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_r1_design import (  # noqa: E402
    _feature,
    _manifest_map,
    _validate_output_root,
)
from common.dabe_pseudo import _minmax  # noqa: E402
from common.utils import feature_manifest_path, load_config, torch_load, write_json, write_jsonl  # noqa: E402
from models.mbsp_reconstruction import MultiBackgroundSubspaceProjector  # noqa: E402


VERSION = "gbsp_pca_absmm_v1"
EXPECTED_TRAIN_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
EXPECTED_TRAIN_TOTAL = sum(EXPECTED_TRAIN_COUNTS.values())
GRID = 37
FEATURE_DIM = 384
NUM_SUBSPACES = 1
PCA_ENERGY = 0.90
PCA_MAX_RANK = 8
PCA_MIN_RANK = 1
THRESHOLD = 0.63

_WORKER_THREADS = 1


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _init_worker(torch_threads: int) -> None:
    global _WORKER_THREADS
    _WORKER_THREADS = int(torch_threads)
    torch.set_num_threads(_WORKER_THREADS)


def _valid_existing(path: Path, dataset: str, stem: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return False
        if payload.get("dataset") != dataset or payload.get("stem") != stem:
            return False
        if payload.get("gbsp_version") != VERSION:
            return False
        if payload.get("generation_stage") not in {
            "cached_full_bc_single_pca_absolute_minmax_only",
            "cached_full_bc_single_pca_absolute_minmax_r1_fallback",
        }:
            return False
        if not Path(str(payload.get("source_dabe_cache_path", ""))).is_file():
            return False
        if int(payload.get("num_subspaces", -1)) != NUM_SUBSPACES:
            return False
        if int(payload.get("pca_max_rank", -1)) != PCA_MAX_RANK:
            return False
        if abs(float(payload.get("pca_energy", -1.0)) - PCA_ENERGY) > 1e-12:
            return False
        for field in ("gbsp_abs_raw_37", "gbsp_abs_minmax_37"):
            value = payload.get(field)
            if not torch.is_tensor(value) or tuple(value.shape) != (1, GRID, GRID):
                return False
            if not bool(torch.isfinite(value).all()):
                return False
        calibrated = payload["gbsp_abs_minmax_37"]
        return float(calibrated.min()) >= 0.0 and float(calibrated.max()) <= 1.0
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return False


def _process_one(task: dict) -> dict:
    row = task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    output_path = Path(task["output_path"])
    if _valid_existing(output_path, dataset, stem):
        payload = torch_load(output_path, map_location="cpu")
        return {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path),
            "source_dabe_cache_path": str(payload["source_dabe_cache_path"]),
            "skipped": True,
            "num_background_atoms": int(payload["background_indices"].numel()),
            "selected_rank": int(payload["selected_ranks"][0]),
            "hard_area_063": float(payload["hard_area_063"]),
            "fallback_used": bool(payload.get("fallback_used", False)),
            "fallback_reason": str(payload.get("fallback_reason", "")),
            "total_seconds": float(payload.get("runtime", {}).get("total_seconds", 0.0)),
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }

    started = time.perf_counter()
    try:
        feature_path = Path(row["cache_path"]).resolve()
        feature_payload = torch_load(feature_path, map_location="cpu")
        feature = _feature(feature_payload, dataset, stem, feature_path)
        if tuple(feature.shape) != (FEATURE_DIM, GRID, GRID):
            raise ValueError(f"feature must be Tensor[{FEATURE_DIM},{GRID},{GRID}]")
        flat_feature = feature.permute(1, 2, 0).reshape(GRID * GRID, FEATURE_DIM)
        if bool((torch.linalg.vector_norm(flat_feature, dim=1) <= 0).any()):
            raise ValueError(f"feature cache contains a zero vector: {feature_path}")
        normalized_feature = F.normalize(flat_feature, p=2, dim=1).contiguous()

        dabe_row = task["dabe_row"]
        dabe_path = Path(dabe_row["cache_path"]).resolve()
        dabe_payload = torch_load(dabe_path, map_location="cpu")
        if (
            str(dabe_payload.get("dataset")) != dataset
            or str(dabe_payload.get("stem")) != stem
        ):
            raise RuntimeError(f"DABE cache identity mismatch: {dabe_path}")
        if str(dabe_payload.get("backbone_key")) != "dinov1-s8":
            raise RuntimeError(f"DABE cache backbone mismatch: {dabe_path}")
        if str(dabe_payload.get("dabe_version", "")).lower() != "v2":
            raise RuntimeError(f"DABE cache version mismatch: {dabe_path}")
        anchor = dabe_payload.get("bg_anchor_37")
        confidence = dabe_payload.get("bc_map_37")
        for name, value in (("bg_anchor_37", anchor), ("bc_map_37", confidence)):
            if not torch.is_tensor(value) or tuple(value.shape) != (1, GRID, GRID):
                raise ValueError(f"{name} must be Tensor[1,{GRID},{GRID}]: {dabe_path}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN/Inf: {dabe_path}")
        anchor = anchor.detach().cpu().float().contiguous()
        confidence = confidence.detach().cpu().float().contiguous()
        background_indices = torch.where(anchor.reshape(-1) > 0.5)[0]
        empty_dictionary = int(background_indices.numel()) == 0
        if empty_dictionary:
            background_indices = confidence.reshape(-1).argmax().reshape(1)
        background = normalized_feature.index_select(0, background_indices)
        projector = MultiBackgroundSubspaceProjector(
            num_subspaces=NUM_SUBSPACES,
            min_cluster_size=16,
            pca_energy=PCA_ENERGY,
            pca_max_rank=PCA_MAX_RANK,
            pca_min_rank=PCA_MIN_RANK,
            seed=0,
            kmeans_n_init=10,
            kmeans_max_iter=100,
        ).fit(background)
        score = projector.score(normalized_feature)
        absolute_raw = score.absolute_residual.reshape(1, GRID, GRID).float().contiguous()
        absolute_minmax = _minmax(absolute_raw).float().contiguous()
        if not bool(torch.isfinite(absolute_raw).all()) or not bool(
            torch.isfinite(absolute_minmax).all()
        ):
            raise RuntimeError("GBSP response contains NaN or Inf")
        if float(absolute_minmax.min()) < 0.0 or float(absolute_minmax.max()) > 1.0:
            raise RuntimeError("GBSP Min-Max response escaped [0,1]")
        response_68 = F.interpolate(
            absolute_minmax.unsqueeze(0),
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        hard_area = float((response_68 > THRESHOLD).float().mean())
        elapsed = time.perf_counter() - started
        payload = {
            "dataset": dataset,
            "stem": stem,
            "image_path": str(
                row.get("image_path") or feature_payload.get("image_path") or ""
            ),
            "backbone_key": "dinov1-s8",
            "dabe_version": "v2",
            "source_dabe_version": "v2",
            "gbsp_version": VERSION,
            "source_augs": ["identity"],
            "source_num_views": 1,
            "generation_stage": "cached_full_bc_single_pca_absolute_minmax_only",
            "dino_forward_used": False,
            "gt_used_for_generation": False,
            "foreground_seed_used": False,
            "random_walk_used": False,
            "evidence_gate_used": False,
            "p_dabe_generated": False,
            "fallback_used": False,
            "fallback_reason": "",
            "num_subspaces": NUM_SUBSPACES,
            "pca_energy": PCA_ENERGY,
            "pca_max_rank": PCA_MAX_RANK,
            "pca_min_rank": PCA_MIN_RANK,
            "static_threshold": THRESHOLD,
            "gbsp_abs_raw_37": absolute_raw,
            "gbsp_abs_minmax_37": absolute_minmax,
            "background_indices": background_indices.contiguous(),
            "bc_map_37": confidence,
            "bg_anchor_37": anchor,
            "empty_dictionary_replaced_by_bc_argmax": empty_dictionary,
            "cluster_sizes": projector.cluster_sizes,
            "selected_ranks": projector.selected_ranks,
            "singular_values": list(projector.singular_values),
            "subspace_mean": projector.subspace_means[0],
            "hard_area_063": hard_area,
            "runtime": {
                "svd_seconds": float(projector.timing["svd_seconds"]),
                "score_seconds": float(projector.timing["score_seconds"]),
                "total_seconds": elapsed,
                "torch_threads": _WORKER_THREADS,
            },
            "source_feature_cache_path": str(feature_path),
            "source_dabe_cache_path": str(dabe_path),
        }
        _atomic_save(payload, output_path)
        return {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path),
            "source_dabe_cache_path": str(dabe_path),
            "skipped": False,
            "num_background_atoms": int(background_indices.numel()),
            "selected_rank": int(projector.selected_ranks[0]),
            "hard_area_063": hard_area,
            "fallback_used": False,
            "fallback_reason": "",
            "total_seconds": elapsed,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
    except Exception as error:
        failure_traceback = traceback.format_exc()
        try:
            dabe_path = Path(task["dabe_row"]["cache_path"]).resolve()
            dabe_payload = torch_load(dabe_path, map_location="cpu")
            fallback = dabe_payload.get("residual_pass1_37")
            if (
                str(dabe_payload.get("dataset")) != dataset
                or str(dabe_payload.get("stem")) != stem
            ):
                raise RuntimeError("fallback DABE identity mismatch")
            if not torch.is_tensor(fallback) or tuple(fallback.shape) != (1, GRID, GRID):
                raise ValueError("fallback residual_pass1_37 must be Tensor[1,37,37]")
            fallback = fallback.detach().cpu().float().contiguous()
            if (
                not bool(torch.isfinite(fallback).all())
                or float(fallback.min()) < 0.0
                or float(fallback.max()) > 1.0
            ):
                raise ValueError("fallback residual_pass1_37 must be finite in [0,1]")
            anchor = dabe_payload.get("bg_anchor_37")
            confidence = dabe_payload.get("bc_map_37")
            if not torch.is_tensor(anchor) or tuple(anchor.shape) != (1, GRID, GRID):
                anchor = torch.zeros_like(fallback)
            if not torch.is_tensor(confidence) or tuple(confidence.shape) != (1, GRID, GRID):
                confidence = torch.zeros_like(fallback)
            anchor = anchor.detach().cpu().float().contiguous()
            confidence = confidence.detach().cpu().float().contiguous()
            background_indices = torch.where(anchor.reshape(-1) > 0.5)[0]
            response_68 = F.interpolate(
                fallback.unsqueeze(0),
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            hard_area = float((response_68 > THRESHOLD).float().mean())
            elapsed = time.perf_counter() - started
            fallback_reason = f"GBSP generation exception; substituted cached R1: {error!r}"
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": str(dabe_payload.get("image_path", "")),
                "backbone_key": "dinov1-s8",
                "dabe_version": "v2",
                "source_dabe_version": "v2",
                "gbsp_version": VERSION,
                "source_augs": ["identity"],
                "source_num_views": 1,
                "generation_stage": (
                    "cached_full_bc_single_pca_absolute_minmax_r1_fallback"
                ),
                "dino_forward_used": False,
                "gt_used_for_generation": False,
                "foreground_seed_used": False,
                "random_walk_used": False,
                "evidence_gate_used": False,
                "p_dabe_generated": False,
                "num_subspaces": NUM_SUBSPACES,
                "pca_energy": PCA_ENERGY,
                "pca_max_rank": PCA_MAX_RANK,
                "pca_min_rank": PCA_MIN_RANK,
                "static_threshold": THRESHOLD,
                "gbsp_abs_raw_37": fallback.clone(),
                "gbsp_abs_minmax_37": fallback.clone(),
                "background_indices": background_indices.contiguous(),
                "bc_map_37": confidence,
                "bg_anchor_37": anchor,
                "cluster_sizes": torch.tensor([], dtype=torch.long),
                "selected_ranks": torch.tensor([-1], dtype=torch.long),
                "singular_values": [],
                "subspace_mean": torch.zeros(FEATURE_DIM, dtype=torch.float32),
                "hard_area_063": hard_area,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "generation_exception_traceback": failure_traceback,
                "runtime": {
                    "svd_seconds": 0.0,
                    "score_seconds": 0.0,
                    "total_seconds": elapsed,
                    "torch_threads": _WORKER_THREADS,
                },
                "source_feature_cache_path": str(
                    Path(row["cache_path"]).resolve()
                ),
                "source_dabe_cache_path": str(dabe_path),
            }
            _atomic_save(payload, output_path)
            return {
                "dataset": dataset,
                "stem": stem,
                "cache_path": str(output_path),
                "source_dabe_cache_path": str(dabe_path),
                "skipped": False,
                "num_background_atoms": int(background_indices.numel()),
                "selected_rank": -1,
                "hard_area_063": hard_area,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "generation_exception_traceback": failure_traceback,
                "total_seconds": elapsed,
                "worker_peak_rss_mb": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                ),
            }
        except Exception as fallback_error:
            return {
                "dataset": dataset,
                "stem": stem,
                "cache_path": str(output_path),
                "error": repr(error),
                "traceback": failure_traceback,
                "fallback_error": repr(fallback_error),
                "fallback_traceback": traceback.format_exc(),
            }


def _mean(rows: list[dict], field: str):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def build_gbsp_train_cache(
    config_path: str | Path,
    out_root: str | Path,
    max_samples: int = -1,
    workers: int = 2,
    torch_threads: int = 4,
    strict_failures: bool = False,
) -> dict:
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    config_path = Path(config_path).resolve()
    output_root = Path(out_root).resolve()
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    feature_manifest = feature_manifest_path(cfg, "train").resolve()
    feature_rows, feature_map = _manifest_map(feature_manifest)
    counts = Counter(str(row["dataset"]) for row in feature_rows)
    if len(feature_rows) != EXPECTED_TRAIN_TOTAL or dict(counts) != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError(
            f"training feature counts differ: {len(feature_rows)} {dict(counts)}"
        )
    if len(feature_map) != len(feature_rows):
        raise RuntimeError("training feature manifest contains duplicate keys")
    source_dabe_root = Path(
        str(getattr(cfg, "GBSP_SOURCE_DABE_ROOT", "")).strip()
    ).resolve()
    source_dabe_manifest = source_dabe_root / "manifest_train.jsonl"
    dabe_rows, dabe_map = _manifest_map(source_dabe_manifest)
    if len(dabe_rows) != EXPECTED_TRAIN_TOTAL or len(dabe_map) != len(dabe_rows):
        raise RuntimeError("training DABE manifest must contain 4040 unique rows")
    if set(feature_map) != set(dabe_map):
        raise RuntimeError("training feature/DABE manifest identities differ")
    selected = feature_rows if max_samples == -1 else feature_rows[:max_samples]
    if max_samples > 0 and len(selected) != max_samples:
        raise RuntimeError("requested more training samples than available")
    _validate_output_root(
        output_root,
        [feature_manifest.parent.resolve(), source_dabe_root],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "feature_row": row,
            "dabe_row": dabe_map[(str(row["dataset"]), str(row["stem"]))],
            "output_path": str(
                output_root / "train" / str(row["dataset"]) / f"{row['stem']}.pt"
            ),
        }
        for row in selected
    ]

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"GBSP train cache {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if "error" in result]
    fallbacks = [result for result in results if result.get("fallback_used")]
    write_json(output_root / "generation_failures.json", failures)
    write_json(output_root / "generation_fallbacks.json", fallbacks)
    valid = [result for result in results if "error" not in result]
    valid_map = {(row["dataset"], row["stem"]): row for row in valid}
    manifest_rows = []
    for row in selected:
        key = (str(row["dataset"]), str(row["stem"]))
        if key not in valid_map:
            continue
        result = valid_map[key]
        manifest_rows.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "cache_path": result["cache_path"],
                "source_feature_cache_path": row["cache_path"],
                "source_dabe_cache_path": result["source_dabe_cache_path"],
                "backbone_key": "dinov1-s8",
                "source_dabe_version": "v2",
                "gbsp_version": VERSION,
                "source_augs": ["identity"],
                "source_num_views": 1,
                "shape": [1, GRID, GRID],
                "fallback_used": bool(result.get("fallback_used", False)),
                "fallback_reason": str(result.get("fallback_reason", "")),
            }
        )
    write_jsonl(output_root / "manifest_train.jsonl", manifest_rows)
    summary = {
        "gbsp_version": VERSION,
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "num_fallback": len(fallbacks),
        "num_resumed": sum(bool(row.get("skipped")) for row in valid),
        "dataset_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "mean_num_background_atoms": _mean(valid, "num_background_atoms"),
        "mean_selected_rank": _mean(valid, "selected_rank"),
        "mean_hard_area_063": _mean(valid, "hard_area_063"),
        "mean_total_seconds": _mean(valid, "total_seconds"),
        "max_worker_peak_rss_mb": max(
            (float(row["worker_peak_rss_mb"]) for row in valid), default=None
        ),
        "wall_seconds": time.perf_counter() - started,
        "config_path": str(config_path),
        "source_feature_manifest": str(feature_manifest),
        "source_dabe_manifest": str(source_dabe_manifest),
        "output_root": str(output_root),
        "num_subspaces": NUM_SUBSPACES,
        "pca_energy": PCA_ENERGY,
        "pca_max_rank": PCA_MAX_RANK,
        "static_threshold": THRESHOLD,
        "gt_used_for_generation": False,
        "dino_forward_used": False,
        "training_used": False,
    }
    write_json(output_root / "protocol.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        print(
            f"[Warn] recorded {len(failures)} failures; successful caches were retained",
            flush=True,
        )
        if strict_failures:
            raise RuntimeError(f"GBSP cache generation failed for {len(failures)} samples")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--strict_failures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_gbsp_train_cache(
        config_path=args.config,
        out_root=args.out_root,
        max_samples=args.max_samples,
        workers=args.workers,
        torch_threads=args.torch_threads,
        strict_failures=args.strict_failures,
    )
