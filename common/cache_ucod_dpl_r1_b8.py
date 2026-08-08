#!/usr/bin/env python3
"""Build identity-only DINOv1-B/8 R1 caches for UCOD-DPL.

The generator stops immediately after the first background reconstruction
residual.  It never reads ground truth and never runs foreground seeding,
random walk, evidence gating, multi-view fusion, CVBR, or the final DABE
pseudo-label construction.

Two synchronized outputs are written:

1. A keyed, auditable cache containing ``residual_pass1_37`` and the final
   ``r1_hard_68`` target for every training image.
2. The index-ordered ``MetaListPickleIO`` layout expected by UCOD-DPL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_r1_design import effective_dabe_v2_params  # noqa: E402
from common.dabe_pseudo import (  # noqa: E402
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
)
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_TRAIN_TOTAL = 4040
EXPECTED_CHANNELS = 768
EXPECTED_GRID = 37
EXPECTED_LOSS_SIZE = 68
EXPECTED_DATASETS = ("TR-CAMO", "TR-COD10K")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).rstrip()
        return commit, status
    except (OSError, subprocess.CalledProcessError):
        return "", ""


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_pickle_save(value, path: Path) -> None:
    """Write the format consumed by UCOD-DPL's MetaListPickleIO."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _load_feature(row: dict) -> torch.Tensor:
    path = Path(row["cache_path"]).resolve()
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature payload must be a dict: {path}")
    if payload.get("dataset") != row["dataset"] or payload.get("stem") != row["stem"]:
        raise RuntimeError(f"Feature cache key mismatch: {path}")
    if payload.get("backbone_key") != "dinov1-b8":
        raise ValueError(f"Feature payload is not DINOv1-B/8: {path}")
    if payload.get("resize_interpolation") != "bilinear":
        raise ValueError(f"Feature payload does not use UCOD-DPL bilinear resize: {path}")
    feature = payload.get("tensor")
    expected_shape = (EXPECTED_CHANNELS, EXPECTED_GRID, EXPECTED_GRID)
    if not torch.is_tensor(feature) or tuple(feature.shape) != expected_shape:
        actual = None if not torch.is_tensor(feature) else tuple(feature.shape)
        raise ValueError(f"Expected feature {expected_shape}, got {actual}: {path}")
    feature = feature.detach().cpu().float().contiguous()
    if not torch.isfinite(feature).all():
        raise ValueError(f"Feature contains NaN/Inf: {path}")
    return feature


def _first_background_residual(
    feature: torch.Tensor,
    image_path: str,
    params: dict,
) -> torch.Tensor:
    """Reproduce the first DABE-v2 background reconstruction only."""
    grid = int(params["GRID"])
    if grid != EXPECTED_GRID:
        raise ValueError(f"R1-B8 requires GRID={EXPECTED_GRID}, got {grid}")

    rgb = _load_rgb_grid(image_path, grid)
    feat_n = F.normalize(
        feature.permute(1, 2, 0).reshape(grid * grid, -1),
        dim=1,
        p=2,
    )
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = _sobel_magnitude(rgb).reshape(-1).float()
    neigh_idx, neigh_weight = _build_local_graph(
        feat_n,
        rgb_n,
        edge_n,
        grid,
        params,
    )
    bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, params)
    background_anchor = _background_anchor(bc, border, params)
    residual = _background_residual(feat_n, rgb_n, background_anchor, params)
    residual = residual.reshape(1, grid, grid).detach().cpu().float().contiguous()
    if not torch.isfinite(residual).all():
        raise ValueError(f"R1 contains NaN/Inf: {image_path}")
    if float(residual.min()) < 0.0 or float(residual.max()) > 1.0:
        raise ValueError(f"R1 is outside [0,1]: {image_path}")
    return residual


def _hard68(residual_pass1_37: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        residual_pass1_37.unsqueeze(0),
        size=(EXPECTED_LOSS_SIZE, EXPECTED_LOSS_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (resized > 0.5).float().contiguous()


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict) -> dict:
    row = task["row"]
    dataset = str(row["dataset"])
    stem = str(row["stem"])
    image_path = str(Path(row["image_path"]).resolve())
    feature = _load_feature(row)
    residual = _first_background_residual(feature, image_path, task["params"])
    target = _hard68(residual)

    raw_path = Path(task["raw_root"]) / "train" / dataset / f"{stem}.pt"
    raw_payload = {
        "dataset": dataset,
        "stem": stem,
        "image_path": image_path,
        "backbone_key": "dinov1-b8",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "generation_stage": "first_background_reconstruction_only",
        "gt_used_for_generation": False,
        "dino_forward_used": False,
        "foreground_seed_used": False,
        "random_walk_used": False,
        "evidence_gate_used": False,
        "cvbr_used": False,
        "residual_pass1_37": residual,
        "r1_hard_68": target,
    }
    _atomic_torch_save(raw_payload, raw_path)
    return {
        "dataset": dataset,
        "stem": stem,
        "image_path": image_path,
        "source_feature_cache_path": str(Path(row["cache_path"]).resolve()),
        "cache_path": str(raw_path.resolve()),
        "residual_shape": [1, EXPECTED_GRID, EXPECTED_GRID],
        "target_shape": [1, EXPECTED_LOSS_SIZE, EXPECTED_LOSS_SIZE],
        "foreground_ratio": float(target.mean()),
        "target": target,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _ordered_feature_rows(feature_manifest: Path, data_root: Path) -> list[dict]:
    rows = read_jsonl(feature_manifest)
    if len(rows) != EXPECTED_TRAIN_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_TOTAL} feature rows, got {len(rows)}"
        )

    by_image = {}
    keys = set()
    for line_number, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "image_path", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {feature_manifest}:{line_number}")
        key = (str(row["dataset"]), str(row["stem"]))
        image_path = str(Path(row["image_path"]).resolve())
        if key in keys or image_path in by_image:
            raise RuntimeError(f"Duplicate feature row: {key} / {image_path}")
        if row["dataset"] not in EXPECTED_DATASETS:
            raise ValueError(f"Unexpected training dataset: {row['dataset']}")
        if row.get("backbone_key") != "dinov1-b8":
            raise ValueError(f"Feature manifest is not DINOv1-B/8: {row}")
        if row.get("feature_cache_key") != "dinov1-b8-ucod-dpl":
            raise ValueError(f"Unexpected feature cache protocol: {row}")
        if row.get("resize_interpolation") != "bilinear":
            raise ValueError(f"Feature manifest does not use bilinear resize: {row}")
        if row.get("shape") != [EXPECTED_CHANNELS, EXPECTED_GRID, EXPECTED_GRID]:
            raise ValueError(f"Unexpected feature shape in manifest: {row}")
        if not Path(row["cache_path"]).resolve().is_file():
            raise FileNotFoundError(row["cache_path"])
        keys.add(key)
        by_image[image_path] = row

    image_paths = []
    for dataset in EXPECTED_DATASETS:
        image_dir = data_root / dataset / "im"
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        image_paths.extend(path.resolve() for path in image_dir.iterdir() if path.is_file())
    image_paths = sorted(image_paths)
    if len(image_paths) != EXPECTED_TRAIN_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_TOTAL} training images, got {len(image_paths)}"
        )
    missing = [str(path) for path in image_paths if str(path) not in by_image]
    extras = sorted(set(by_image) - {str(path) for path in image_paths})
    if missing or extras:
        raise RuntimeError(
            f"Feature/image mapping mismatch: missing={missing[:3]}, extras={extras[:3]}"
        )
    return [by_image[str(path)] for path in image_paths]


def build_cache(
    config_path: str,
    feature_manifest: str,
    out_root: str,
    workers: int,
    torch_threads: int,
    overwrite: bool,
    overwrite_reason: str,
) -> dict:
    started = time.time()
    config_path = Path(config_path).resolve()
    feature_manifest = Path(feature_manifest).resolve()
    out_root = Path(out_root).resolve()
    raw_root = out_root / "r1_identity_b8"
    ucod_root = out_root / "pseudo_label_cache_r1_b8" / "TR-CAMO+TR-COD10K"

    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    if not config_path.is_file() or not feature_manifest.is_file():
        raise FileNotFoundError(config_path if not config_path.is_file() else feature_manifest)
    if raw_root.exists() or ucod_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists; use --overwrite with --overwrite_reason: {raw_root} / {ucod_root}"
            )
        if not str(overwrite_reason).strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        for target in (raw_root, ucod_root):
            if target.exists():
                shutil.rmtree(target)

    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-b8":
        raise ValueError("Config BACKBONE_KEY must be dinov1-b8")
    if int(cfg.DINO["feature_input_size"]) != 296:
        raise ValueError("DINOv1-B/8 feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    if int(params["GRID"]) != EXPECTED_GRID or int(params["LOSS_SIZE"]) != EXPECTED_LOSS_SIZE:
        raise ValueError("R1-B8 requires GRID=37 and LOSS_SIZE=68")

    data_root = Path(cfg.DATA_ROOT).resolve()
    rows = _ordered_feature_rows(feature_manifest, data_root)
    source_snapshot = {
        Path(row["cache_path"]).resolve(): (
            Path(row["cache_path"]).resolve().stat().st_size,
            Path(row["cache_path"]).resolve().stat().st_mtime_ns,
        )
        for row in rows
    }

    raw_root.mkdir(parents=True, exist_ok=False)
    ucod_root.mkdir(parents=True, exist_ok=False)
    tasks = [
        {"row": row, "params": params, "raw_root": str(raw_root)}
        for row in rows
    ]
    manifest_rows = []
    index_map = {}
    foreground_sum = 0.0
    worker_peak = 0.0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(torch_threads,),
    ) as executor:
        results = executor.map(_process_one, tasks, chunksize=1)
        for index, result in enumerate(
            tqdm(results, total=len(tasks), desc="cache UCOD-DPL R1-B8")
        ):
            target = result.pop("target")
            worker_peak = max(worker_peak, float(result.pop("worker_peak_rss_mb")))
            foreground_sum += float(result["foreground_ratio"])
            manifest_rows.append(result)
            filename = f"data_{index}.pkl"
            _atomic_pickle_save(target, ucod_root / filename)
            index_map[str(index)] = filename

    current_snapshot = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_snapshot
    }
    if current_snapshot != source_snapshot:
        raise RuntimeError("Source DINOv1-B/8 feature cache changed during generation")

    _atomic_json(ucod_root / "index.json", index_map)
    write_jsonl(raw_root / "manifest_train.jsonl", manifest_rows)
    elapsed = time.time() - started
    commit, status = _git_metadata()
    protocol = {
        "schema": "ucod_dpl_static_r1_b8_cache_v1",
        "backbone_key": "dinov1-b8",
        "feature_channels": EXPECTED_CHANNELS,
        "feature_grid": EXPECTED_GRID,
        "source_feature_manifest": str(feature_manifest),
        "source_feature_manifest_sha256": _sha256(feature_manifest),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "generator_path": str(SCRIPT_PATH),
        "generator_sha256": _sha256(SCRIPT_PATH),
        "git_commit": commit,
        "git_status_short": status,
        "num_samples": len(manifest_rows),
        "datasets": list(EXPECTED_DATASETS),
        "source_augs": ["identity"],
        "source_num_views": 1,
        "gt_used_for_generation": False,
        "dino_forward_used": False,
        "foreground_seed_used": False,
        "random_walk_used": False,
        "evidence_gate_used": False,
        "cvbr_used": False,
        "source_field": "residual_pass1_37",
        "resize": "bilinear_37_to_68_align_corners_false",
        "threshold": "strict_greater_than_0.5",
        "ucod_cache_path": str(ucod_root),
        "ucod_serialization": "python_pickle",
        "mean_foreground_ratio": foreground_sum / len(manifest_rows),
        "workers": workers,
        "torch_threads_per_worker": torch_threads,
        "max_worker_peak_rss_mb": worker_peak,
        "elapsed_seconds": elapsed,
        "average_seconds_per_image": elapsed / len(manifest_rows),
        "source_feature_cache_modified": False,
        "overwrite": bool(overwrite),
        "overwrite_reason": str(overwrite_reason).strip(),
        "effective_r1_params": params,
    }
    write_json(raw_root / "protocol.json", protocol)
    write_json(ucod_root / "protocol.json", protocol)
    return protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = build_cache(
        config_path=args.config,
        feature_manifest=args.feature_manifest,
        out_root=args.out_root,
        workers=args.workers,
        torch_threads=args.torch_threads,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
