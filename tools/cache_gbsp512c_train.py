#!/usr/bin/env python3
"""Build the full 512-C native-64 GBSP training pseudo-label cache.

This script reads only frozen DINO features plus RGB images.  Training GT and
legacy DABE caches are neither indexed nor loaded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (  # noqa: E402
    build_image_items,
    load_config,
    read_jsonl,
    torch_load,
    write_json,
    write_jsonl,
)
from models.gbsp_resolution import (  # noqa: E402
    fit_gbsp_from_prepared,
    prepare_resolution_graph,
)
from tools.eval_gbsp_resolution_probe import _cfg_params  # noqa: E402


VERSION = "gbsp_resolution_512c_native64_v1"
SOURCE_KEY = "gbsp_abs_minmax_64"
_WORKER_PARAMS: dict = {}
_WORKER_META: dict = {}


def _fingerprint(config_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.resolve().as_posix().encode())
    digest.update(config_path.read_bytes())
    digest.update(VERSION.encode())
    return digest.hexdigest()


def _initialize_worker(params: dict, meta: dict, torch_threads: int) -> None:
    global _WORKER_PARAMS, _WORKER_META
    torch.set_num_threads(max(1, int(torch_threads)))
    _WORKER_PARAMS = dict(params)
    _WORKER_META = dict(meta)


def _load_feature(row: dict, dataset: str, stem: str) -> torch.Tensor:
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"feature payload must be dict: {row['cache_path']}")
    if (str(payload.get("dataset")), str(payload.get("stem"))) != (
        dataset,
        stem,
    ):
        raise RuntimeError(f"feature identity mismatch: {row['cache_path']}")
    feature = payload.get("tensor")
    if not torch.is_tensor(feature) or tuple(feature.shape) != (384, 64, 64):
        raise RuntimeError(
            f"feature must be [384,64,64]: {row['cache_path']} -> "
            f"{getattr(feature, 'shape', None)}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError(f"feature contains NaN/Inf: {row['cache_path']}")
    return feature.detach().cpu().float().contiguous()


def _valid_existing(
    path: Path, dataset: str, stem: str, fingerprint: str
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        score = payload.get(SOURCE_KEY)
        hard = payload.get("gbsp_hard_t060_64")
        return bool(
            isinstance(payload, dict)
            and payload.get("gbsp_version") == VERSION
            and payload.get("dataset") == dataset
            and payload.get("stem") == stem
            and payload.get("settings_fingerprint") == fingerprint
            and torch.is_tensor(score)
            and tuple(score.shape) == (1, 64, 64)
            and bool(torch.isfinite(score).all().item())
            and float(score.min()) >= 0.0
            and float(score.max()) <= 1.0
            and torch.is_tensor(hard)
            and tuple(hard.shape) == (1, 64, 64)
        )
    except Exception:
        return False


def _manifest_row(path: Path, payload: dict, status: str) -> dict:
    return {
        "dataset": str(payload["dataset"]),
        "stem": str(payload["stem"]),
        "cache_path": str(path.resolve()),
        "backbone_key": str(payload["backbone_key"]),
        "gbsp_version": str(payload["gbsp_version"]),
        "source": str(payload["source"]),
        "source_key": SOURCE_KEY,
        "shape": [1, 64, 64],
        "threshold": float(payload["threshold"]),
        "hard_area": float(payload["hard_area"]),
        "selected_rank": int(payload["selected_rank"]),
        "required_rank_uncapped": int(payload["required_rank_uncapped"]),
        "hit_rank_cap": int(bool(payload["hit_rank_cap"])),
        "candidate_count": int(payload["candidate_count"]),
        "gt_used_for_generation": False,
        "status": status,
    }


def _generate_one(task: dict) -> dict:
    dataset, stem = str(task["dataset"]), str(task["stem"])
    out_path = Path(task["out_path"])
    feature = _load_feature(task["feature_row"], dataset, stem)
    started = time.perf_counter()
    prepared = prepare_resolution_graph(
        feature,
        str(task["image_path"]),
        _WORKER_PARAMS,
    )
    result = fit_gbsp_from_prepared(
        prepared,
        _WORKER_PARAMS,
        pca_energy=float(_WORKER_META["pca_energy"]),
        pca_min_rank=int(_WORKER_META["pca_min_rank"]),
        pca_max_rank=int(_WORKER_META["pca_max_rank"]),
    )
    score = result.minmax_residual.detach().cpu().float()
    hard = (score > float(_WORKER_META["threshold"])).float()
    payload = {
        "version": VERSION,
        "gbsp_version": VERSION,
        "dataset": dataset,
        "stem": stem,
        "image_path": str(task["image_path"]),
        "feature_cache_path": str(task["feature_row"]["cache_path"]),
        "backbone_key": str(_WORKER_META["backbone_key"]),
        "source": "dino_feature_cache_direct",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "training_gt_read": False,
        "gt_used_for_generation": False,
        "settings_fingerprint": str(task["fingerprint"]),
        "input_size": 512,
        "grid_size": 64,
        "border_width": 2,
        "bg_ratio": float(_WORKER_META["bg_ratio"]),
        "pca_energy": float(_WORKER_META["pca_energy"]),
        "pca_rank_cap": int(_WORKER_META["pca_max_rank"]),
        "threshold": float(_WORKER_META["threshold"]),
        "gbsp_abs_raw_64": result.raw_residual.detach().cpu().float(),
        SOURCE_KEY: score,
        "gbsp_hard_t060_64": hard,
        "hard_area": float(hard.mean().item()),
        "selected_rank": int(result.selected_rank),
        "required_rank_uncapped": int(result.required_rank_uncapped),
        "energy_at_cap": float(result.energy_at_rank_max),
        "retained_energy": float(result.retained_energy),
        "hit_rank_cap": bool(result.hit_rank_cap),
        "candidate_count": int(result.background_indices.numel()),
        "background_indices": result.background_indices.detach().cpu(),
        "bc": result.bc.detach().cpu().float(),
        "rgb_seconds": float(prepared.rgb_seconds),
        "graph_seconds": float(prepared.graph_seconds),
        "bc_seconds": float(result.bc_seconds),
        "pca_seconds": float(result.pca_seconds),
        "wall_seconds": float(time.perf_counter() - started),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return _manifest_row(out_path, payload, "generated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--failure-policy", choices=("record", "strict"), default="record"
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    required = {
        "input_size": int(cfg.DINO["feature_input_size"]),
        "grid": int(cfg.GRID_SIZE),
        "border_width": int(cfg.GBSP_BORDER_WIDTH),
        "rank_cap": int(cfg.GBSP_PCA_MAX_RANK),
    }
    if required != {
        "input_size": 512,
        "grid": 64,
        "border_width": 2,
        "rank_cap": 12,
    }:
        raise RuntimeError(f"not the frozen 512-C protocol: {required}")
    threshold = float(getattr(cfg, "GBSP_THRESHOLD", 0.60))
    if abs(threshold - 0.60) > 1e-12:
        raise RuntimeError(f"512-C training cache requires threshold=0.60: {threshold}")

    feature_manifest = Path(args.feature_manifest).expanduser().resolve()
    feature_rows = read_jsonl(feature_manifest)
    feature_map = {
        (str(row["dataset"]), str(row["stem"])): row for row in feature_rows
    }
    if len(feature_map) != len(feature_rows):
        raise RuntimeError(f"duplicate feature manifest identities: {feature_manifest}")

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if int(args.max_samples) >= 0:
        items = items[: int(args.max_samples)]
    if not items:
        raise RuntimeError("empty 512-C training selection")
    missing = [
        (str(item["dataset"]), str(item["stem"]))
        for item in items
        if (str(item["dataset"]), str(item["stem"])) not in feature_map
    ]
    if missing:
        raise RuntimeError(
            f"DINO512 feature cache misses {len(missing)} training samples; "
            f"first={missing[:5]}"
        )

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(config_path)
    manifest_path = out_root / "manifest_train.jsonl"
    failures_path = out_root / "generation_failures.jsonl"
    metadata_path = out_root / "generation_metadata.json"
    params = _cfg_params(cfg)
    meta = {
        "backbone_key": str(cfg.BACKBONE_KEY),
        "bg_ratio": float(cfg.GBSP_BG_RATIO),
        "pca_energy": float(cfg.GBSP_PCA_ENERGY),
        "pca_min_rank": int(cfg.GBSP_PCA_MIN_RANK),
        "pca_max_rank": int(cfg.GBSP_PCA_MAX_RANK),
        "threshold": threshold,
    }

    rows: list[dict] = []
    failures: list[dict] = []
    tasks: list[dict] = []
    for item in items:
        dataset, stem = str(item["dataset"]), str(item["stem"])
        path = out_root / dataset / f"{stem}.pt"
        if _valid_existing(path, dataset, stem, fingerprint):
            rows.append(
                _manifest_row(
                    path,
                    torch_load(path, map_location="cpu"),
                    "reused",
                )
            )
        else:
            tasks.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": str(item["image_path"]),
                    "feature_row": feature_map[(dataset, stem)],
                    "out_path": str(path),
                    "fingerprint": fingerprint,
                }
            )

    started = time.perf_counter()
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(params, meta, int(args.torch_threads)),
    ) as pool:
        futures = [(task, pool.submit(_generate_one, task)) for task in tasks]
        for index, (task, future) in enumerate(futures, 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                record = {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "exception_type": type(exc).__name__,
                    "exception": repr(exc),
                }
                failures.append(record)
                print(f"FAILED {task['dataset']}/{task['stem']}: {exc!r}", flush=True)
                if args.failure_policy == "strict":
                    write_jsonl(failures_path, failures)
                    raise
            if index % max(1, int(args.checkpoint_every)) == 0 or index == len(tasks):
                rows.sort(key=lambda row: (row["dataset"], row["stem"]))
                write_jsonl(manifest_path, rows)
                write_jsonl(failures_path, failures)
                print(
                    f"[{index}/{len(tasks)} new] valid={len(rows)}/{len(items)} "
                    f"failed={len(failures)} reused={len(items) - len(tasks)}",
                    flush=True,
                )

    order = {
        (str(item["dataset"]), str(item["stem"])): index
        for index, item in enumerate(items)
    }
    rows.sort(key=lambda row: order[(row["dataset"], row["stem"])])
    write_jsonl(manifest_path, rows)
    write_jsonl(failures_path, failures)
    valid_keys = {(row["dataset"], row["stem"]) for row in rows}
    expected_keys = set(order)
    metadata = {
        "version": VERSION,
        "num_requested": len(items),
        "num_valid": len(valid_keys),
        "num_failed": len(failures),
        "is_full_complete": (
            len(items) == 4040 and valid_keys == expected_keys and not failures
        ),
        "counts": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in ("TR-CAMO", "TR-COD10K")
        },
        "feature_manifest": str(feature_manifest),
        "out_root": str(out_root),
        "settings_fingerprint": fingerprint,
        "source_key": SOURCE_KEY,
        "threshold": threshold,
        "gt_used_for_generation": False,
        "legacy_dabe_cache_used": False,
        "training_triggered": False,
        "workers": workers,
        "torch_threads_per_worker": int(args.torch_threads),
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if failures or valid_keys != expected_keys:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
