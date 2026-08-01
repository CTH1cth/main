#!/usr/bin/env python3
"""Build GT-free BGNull-v1 candidates from frozen DABE and DINO caches."""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_background_null import (  # noqa: E402
    BGNULL_VERSION,
    CROSS_EXCLUSION_RADIUS,
    TIE_ATOL,
    build_background_null_candidates,
)
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS, _load_rgb_grid  # noqa: E402
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    read_jsonl,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
PROBABILITY_FIELDS = (
    "n0_r1_37",
    "n1_cross_r1_37",
    "n2_local_null_raw_37",
    "n2_local_null_r1dist_37",
    "rc_nn_minmax_37",
    "bg_anchor_37",
    "local_excluded_weight_mass_37",
)
RAW_FIELDS = (
    "regular_raw_residual_37",
    "n1_cross_raw_residual_37",
    "anchor_cross_error_37",
    "rc_nn_raw_37",
)


def effective_dabe_v2_params(cfg) -> dict:
    params = {}
    for key in DABE_V2_DEFAULT_PARAMS:
        cfg_key = f"DABE_{key}"
        if hasattr(cfg, cfg_key):
            params[key] = getattr(cfg, cfg_key)
    params["VERSION"] = "v2"
    return {**DABE_V2_DEFAULT_PARAMS, **params}


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


def _manifest_map(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    mapping = {}
    for index, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{index}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"Duplicate key in {path}: {key}")
        cache_path = Path(row["cache_path"]).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        mapping[key] = row
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows, mapping


def _snapshot(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _same_or_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents


def _validate_output_root(output: Path, source_roots: list[Path]) -> None:
    for source in source_roots:
        if _same_or_nested(source, output) or _same_or_nested(output, source):
            raise ValueError(
                f"Output must not equal or nest with source root: output={output}, source={source}"
            )


def _validate_source_dabe(payload: dict, row: dict, path: Path) -> torch.Tensor:
    dataset, stem = str(row["dataset"]), str(row["stem"])
    if not isinstance(payload, dict):
        raise TypeError(f"DABE payload must be dict: {path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"DABE payload key mismatch: {path}")
    if str(payload.get("dabe_version", "")).lower() != "v2":
        raise ValueError(f"DABE payload must be v2: {path}")
    if payload.get("augs") != ["identity"] or payload.get("num_views") != 1:
        raise ValueError(f"DABE payload must be identity single-view: {path}")
    value = payload.get("residual_pass1_37")
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"residual_pass1_37 must be Tensor[1,37,37]: {path}")
    if value.requires_grad or not value.dtype.is_floating_point or not torch.isfinite(value).all():
        raise ValueError(f"Invalid residual_pass1_37: {path}")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise ValueError(f"residual_pass1_37 outside [0,1]: {path}")
    return value.detach().cpu().float().contiguous().clamp(0.0, 1.0)


def _load_feature(row: dict, dataset: str, stem: str) -> tuple[torch.Tensor, Path]:
    path = Path(row["cache_path"]).resolve()
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature payload must be dict: {path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature payload key mismatch: {path}")
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor) or tuple(tensor.shape) != (384, 37, 37):
        raise ValueError(f"Feature tensor must be [384,37,37]: {path}")
    if tensor.requires_grad or not tensor.dtype.is_floating_point or not torch.isfinite(tensor).all():
        raise ValueError(f"Invalid feature tensor: {path}")
    return tensor.detach().cpu().float().contiguous(), path


def _validate_outputs(result: dict, context: str) -> None:
    for field in (*PROBABILITY_FIELDS, *RAW_FIELDS):
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} must be Tensor[1,37,37]: {context}")
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"{field} must be CPU float32: {context}")
        if value.requires_grad or not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"{field} must be detached, contiguous and finite: {context}")
    for field in PROBABILITY_FIELDS:
        value = result[field]
        if float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise ValueError(f"{field} outside [0,1]: {context}")
    for field in RAW_FIELDS:
        if float(result[field].min()) < -1e-7:
            raise ValueError(f"{field} has negative residual: {context}")


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict) -> dict:
    row = task["dabe_row"]
    feature_row = task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    dabe_path = Path(row["cache_path"]).resolve()
    dabe_payload = torch_load(dabe_path, map_location="cpu")
    cached_r1 = _validate_source_dabe(dabe_payload, row, dabe_path)
    feature, feature_path = _load_feature(feature_row, dataset, stem)
    image_path_text = row.get("image_path") or feature_row.get("image_path")
    if not image_path_text:
        raise KeyError(f"image_path missing for {dataset}/{stem}")
    image_path = Path(image_path_text).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    feature_image = feature_row.get("image_path")
    if feature_image and Path(feature_image).resolve() != image_path:
        raise RuntimeError(f"DABE/feature image paths differ for {dataset}/{stem}")
    rgb = _load_rgb_grid(image_path, 37)
    try:
        result = build_background_null_candidates(
            feature_37=feature,
            rgb_37=rgb,
            cached_r1_37=cached_r1,
            effective_params=task["effective_params"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"BGNull candidate failure at {dataset}/{stem}; "
            f"dabe_cache={dabe_path}; feature_cache={feature_path}: {exc}"
        ) from exc
    cached_anchor = dabe_payload.get("bg_anchor_37")
    if not torch.is_tensor(cached_anchor) or tuple(cached_anchor.shape) != (1, 37, 37):
        raise ValueError(f"bg_anchor_37 missing or invalid: {dabe_path}")
    anchor_error = float(
        (
            result["bg_anchor_37"]
            - cached_anchor.detach().cpu().float().clamp(0.0, 1.0)
        ).abs().max()
    )
    if anchor_error > 0.0:
        raise RuntimeError(
            f"Recomputed background anchor differs from cached anchor at "
            f"{dataset}/{stem}: max_abs={anchor_error}"
        )
    result["diagnostics"]["bg_anchor_vs_cached_max_abs"] = anchor_error
    _validate_outputs(result, f"{dataset}/{stem}")
    payload = {
        "dataset": dataset,
        "stem": stem,
        "bgnull_version": BGNULL_VERSION,
        "source_dabe_version": "v2",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "source_dabe_cache_path": str(dabe_path),
        "source_feature_cache_path": str(feature_path),
        "image_path": str(image_path),
        **result,
    }
    output_path = Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return {
        "manifest_row": {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path.resolve()),
            "source_dabe_cache_path": str(dabe_path),
            "source_feature_cache_path": str(feature_path),
            "bgnull_version": BGNULL_VERSION,
            "shape": [1, 37, 37],
        },
        "r1_error": max(
            float(result["diagnostics"]["r1_recompute_max_abs"]),
            float(result["diagnostics"]["regular_detail_vs_cache_r1_max_abs"]),
        ),
        "cross_fallback_count": int(result["diagnostics"]["cross_fallback_count"]),
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_background_null_cache(
    config_path: str | Path,
    dabe_root: str | Path,
    out_root: str | Path,
    split: str = "test",
    max_samples: int = -1,
    overwrite: bool = False,
    overwrite_reason: str = "",
    workers: int = 1,
    torch_threads: int = 1,
) -> dict:
    started = time.time()
    config_path = Path(config_path).resolve()
    dabe_root = Path(dabe_root).resolve()
    output_root = Path(out_root).resolve()
    if split != "test":
        raise ValueError("BGNull-v1 frozen protocol supports split=test only")
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    effective_params = effective_dabe_v2_params(cfg)
    if int(effective_params["GRID"]) != 37 or int(effective_params["K_RECON"]) != 32:
        raise ValueError("Frozen protocol requires GRID=37 and K_RECON=32")

    dabe_manifest = dabe_root / f"manifest_{split}.jsonl"
    feature_manifest = feature_manifest_path(cfg, split).resolve()
    dabe_rows_all, dabe_map = _manifest_map(dabe_manifest)
    feature_rows_all, feature_map = _manifest_map(feature_manifest)
    if set(dabe_map) != set(feature_map):
        missing_feature = sorted(set(dabe_map) - set(feature_map))
        missing_dabe = sorted(set(feature_map) - set(dabe_map))
        raise RuntimeError(
            f"DABE/feature keys differ; missing feature={missing_feature[:10]}, "
            f"missing DABE={missing_dabe[:10]}"
        )
    if max_samples == -1 and (
        len(dabe_rows_all) != 6473 or len(feature_rows_all) != 6473
    ):
        raise RuntimeError(
            f"Expected 6473 DABE/feature rows, got {len(dabe_rows_all)}/{len(feature_rows_all)}"
        )
    if max_samples > 0 and len(dabe_rows_all) < max_samples:
        raise RuntimeError(
            f"Requested {max_samples} samples but manifests contain only {len(dabe_rows_all)}"
        )
    selected_rows = dabe_rows_all if max_samples == -1 else dabe_rows_all[:max_samples]

    feature_root = feature_manifest.parent.resolve()
    _validate_output_root(output_root, [dabe_root, feature_root])
    dabe_source_paths = [dabe_manifest] + [Path(row["cache_path"]).resolve() for row in dabe_rows_all]
    feature_source_paths = [feature_manifest] + [Path(row["cache_path"]).resolve() for row in feature_rows_all]
    dabe_before = _snapshot(dabe_source_paths)
    feature_before = _snapshot(feature_source_paths)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_root: {output_root}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires a non-empty --overwrite_reason")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    log_handle = (output_root / "cache_build.log").open("w", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")

    output_rows = []
    global_r1_error = 0.0
    total_fallback = 0
    max_worker_peak_rss = 0.0
    try:
        for line in (
            "bgnull_version = dabe_bgnull_v1",
            "gt_used_for_generation = false",
            "dino_forward_used = false",
            "source_dabe_cache_modified = false",
            "source_feature_cache_modified = false",
            "dataset_specific_rule = false",
            "threshold_search_used = false",
            "external_model_used = false",
        ):
            log(line)
        log(f"source_dabe_manifest = {dabe_manifest}")
        log(f"source_feature_manifest = {feature_manifest}")
        log(f"output_root = {output_root}")
        log(f"overwrite = {str(bool(overwrite)).lower()}")
        log(f"workers = {workers}")
        log(f"torch_threads_per_worker = {torch_threads}")
        if overwrite:
            log(f"overwrite_reason = {overwrite_reason.strip()}")

        tasks = [
            {
                "dabe_row": row,
                "feature_row": feature_map[(str(row["dataset"]), str(row["stem"]))],
                "effective_params": effective_params,
                "output_root": str(output_root),
                "split": split,
            }
            for row in selected_rows
        ]

        if workers == 1:
            iterator = map(_process_one, tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(torch_threads,),
            )
            iterator = executor.map(_process_one, tasks, chunksize=1)
        try:
            for index, processed in enumerate(iterator, 1):
                output_rows.append(processed["manifest_row"])
                global_r1_error = max(global_r1_error, float(processed["r1_error"]))
                total_fallback += int(processed["cross_fallback_count"])
                max_worker_peak_rss = max(
                    max_worker_peak_rss, float(processed["worker_peak_rss_mb"])
                )
                if index == len(selected_rows) or index % 100 == 0:
                    log(f"processed = {index}/{len(selected_rows)}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        output_keys = {(row["dataset"], row["stem"]) for row in output_rows}
        selected_keys = {(str(row["dataset"]), str(row["stem"])) for row in selected_rows}
        if len(output_rows) != len(output_keys) or output_keys != selected_keys:
            raise RuntimeError("Output keys do not exactly match selected DABE keys")
        if max_samples == -1 and len(output_rows) != 6473:
            raise RuntimeError(f"Full output must contain 6473 rows, got {len(output_rows)}")
        manifest_path = output_root / f"manifest_{split}.jsonl"
        write_jsonl(manifest_path, output_rows)

        if _snapshot(dabe_source_paths) != dabe_before:
            raise RuntimeError("A source DABE manifest/cache changed during generation")
        if _snapshot(feature_source_paths) != feature_before:
            raise RuntimeError("A source feature manifest/cache changed during generation")
        elapsed = time.time() - started
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        git_commit, git_status = _git_metadata()
        protocol = {
            "bgnull_version": BGNULL_VERSION,
            "source_dabe_root": str(dabe_root),
            "source_dabe_manifest": str(dabe_manifest),
            "source_feature_manifest": str(feature_manifest),
            "output_root": str(output_root),
            "split": split,
            "num_samples": len(output_rows),
            "backbone": "DINOv1-S/8",
            "feature_type": "last-block attention key",
            "source_augs": ["identity"],
            "gt_used_for_generation": False,
            "dino_forward_used": False,
            "source_dabe_cache_modified": False,
            "source_feature_cache_modified": False,
            "threshold_search_used": False,
            "dataset_specific_rule": False,
            "external_model_used": False,
            "cross_exclusion_metric": "chebyshev",
            "cross_exclusion_radius": CROSS_EXCLUSION_RADIUS,
            "k_recon": int(effective_params["K_RECON"]),
            "tie_atol": TIE_ATOL,
            "methods": {
                "N0-R1": "cached residual_pass1_37 unchanged",
                "N1-CrossR1": "R1 reconstruction excluding query and Chebyshev radius-1 anchors, then per-image MinMax",
                "N2-LocalNull-Raw": "weighted empirical exceedance of regular raw residual over selected anchors' cross errors",
                "N2-LocalNull-R1Dist": "rank_transport(LocalNull-Raw, cached R1)",
                "RC-NN-MinMax": "single highest combined-similarity background atom residual, then per-image MinMax",
            },
            "effective_dabe_params": effective_params,
            "r1_recompute_global_max_abs": global_r1_error,
            "cross_fallback_count": total_fallback,
            "all_outputs_finite": True,
            "all_probability_outputs_in_unit_range": True,
            "git_commit": git_commit,
            "git_status_short": git_status,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_background_null.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "overwrite": bool(overwrite),
            "overwrite_reason": overwrite_reason.strip(),
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(output_rows),
            "peak_rss_mb": peak_rss_mb,
            "max_worker_peak_rss_mb": max_worker_peak_rss,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_dabe_cache_modified = false (verified size and mtime)")
        log("source_feature_cache_modified = false (verified size and mtime)")
        log(f"num_samples = {len(output_rows)}")
        log(f"r1_recompute_global_max_abs = {global_r1_error:.12g}")
        log(f"cross_fallback_count = {total_fallback}")
        log("all_outputs_finite = true")
        log("all_probability_outputs_in_unit_range = true")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(output_rows):.6f}")
        log(f"peak_rss_mb = {peak_rss_mb:.3f}")
        log(f"max_worker_peak_rss_mb = {max_worker_peak_rss:.3f}")
        return protocol
    finally:
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_background_null_cache(
        config_path=args.config,
        dabe_root=args.dabe_root,
        out_root=args.out_root,
        split=args.split,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
        workers=args.workers,
        torch_threads=args.torch_threads,
    )
