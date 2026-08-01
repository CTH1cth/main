#!/usr/bin/env python3
"""Build GT-free R1-Design-v1 candidates from frozen DABE/DINO caches."""

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

from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from common.dabe_r1_design import (  # noqa: E402
    DABE_R1_DESIGN_VERSION,
    NONNEGATIVE_MAP_FIELDS,
    PROBABILITY_FIELDS,
    build_r1_design_candidates,
)
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
EXPECTED_TOTAL = 6473


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
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "status", "--short"],
            text=True, stderr=subprocess.DEVNULL,
        ).rstrip()
        return commit, status
    except (OSError, subprocess.CalledProcessError):
        return "", ""


def _manifest_map(path: Path):
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    mapping = {}
    for index, row in enumerate(rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing at {path}:{index}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"Duplicate key in {path}: {key}")
        cache = Path(row["cache_path"]).resolve()
        if not cache.is_file():
            raise FileNotFoundError(cache)
        mapping[key] = row
    return rows, mapping


def _snapshot(paths):
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _same_or_nested(left: Path, right: Path):
    return left == right or left in right.parents


def _validate_output_root(output: Path, sources):
    for source in sources:
        if _same_or_nested(source, output) or _same_or_nested(output, source):
            raise ValueError(f"Output/source trees overlap: {output} vs {source}")


def _source_r1(payload: dict, row: dict, path: Path):
    if not isinstance(payload, dict):
        raise TypeError(f"DABE payload must be dict: {path}")
    if payload.get("dataset") != row["dataset"] or payload.get("stem") != row["stem"]:
        raise RuntimeError(f"DABE payload key mismatch: {path}")
    if str(payload.get("dabe_version", "")).lower() != "v2":
        raise ValueError(f"DABE payload must be v2: {path}")
    if payload.get("augs") != ["identity"] or payload.get("num_views") != 1:
        raise ValueError(f"DABE payload must be identity single-view: {path}")
    value = payload.get("residual_pass1_37")
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"residual_pass1_37 must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all() or float(value.min()) < -1e-6 or float(value.max()) > 1 + 1e-6:
        raise ValueError(f"Invalid residual_pass1_37: {path}")
    return value.clamp(0.0, 1.0)


def _feature(payload: dict, dataset: str, stem: str, path: Path):
    if not isinstance(payload, dict) or payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature payload key mismatch: {path}")
    value = payload.get("tensor")
    if not torch.is_tensor(value) or tuple(value.shape) != (384, 37, 37):
        raise ValueError(f"Feature must be Tensor[384,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"Feature contains NaN/Inf: {path}")
    return value


def _validate_result(result: dict, context: str):
    for field in PROBABILITY_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError(f"{field} contract invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
            raise ValueError(f"{field} outside finite [0,1]: {context}")
    for field in NONNEGATIVE_MAP_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError(f"{field} contract invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < -1e-7:
            raise ValueError(f"{field} invalid nonnegative map: {context}")
    shapes = {
        "neigh_idx_37": (1369, 8), "neigh_valid_37": (1369, 8),
        "edge_cue_node": (1369, 8), "edge_cue_hr": (1369, 8),
        "graph_weight_node": (1369, 8), "graph_weight_hr": (1369, 8),
    }
    for field, shape in shapes.items():
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"{field} shape invalid: {context}")
        if value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError(f"{field} contract invalid: {context}")
    for field in ("edge_cue_node", "edge_cue_hr", "graph_weight_node", "graph_weight_hr"):
        value = result[field]
        if value.dtype != torch.float32 or not torch.isfinite(value).all() or float(value.min()) < 0:
            raise ValueError(f"{field} invalid finite nonnegative graph map: {context}")
    if result["neigh_idx_37"].dtype != torch.long:
        raise ValueError(f"neigh_idx_37 must be int64: {context}")
    if result["neigh_valid_37"].dtype != torch.bool:
        raise ValueError(f"neigh_valid_37 must be bool: {context}")
    if not isinstance(result.get("diagnostics"), dict):
        raise TypeError(f"diagnostics missing: {context}")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict):
    row, feature_row = task["dabe_row"], task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    dabe_path = Path(row["cache_path"]).resolve()
    feature_path = Path(feature_row["cache_path"]).resolve()
    dabe = torch_load(dabe_path, map_location="cpu")
    feature_payload = torch_load(feature_path, map_location="cpu")
    cached = _source_r1(dabe, row, dabe_path)
    feature = _feature(feature_payload, dataset, stem, feature_path)
    image_text = row.get("image_path") or feature_row.get("image_path") or dabe.get("image_path")
    if not image_text:
        raise KeyError(f"image_path missing for {dataset}/{stem}")
    image_path = Path(image_text).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    try:
        result = build_r1_design_candidates(
            feature_37=feature, image_path=str(image_path), cached_r1_37=cached,
            effective_params=task["effective_params"],
            feature_input_size=task["feature_input_size"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"R1-Design failure at {dataset}/{stem}; dabe={dabe_path}; feature={feature_path}: {exc}"
        ) from exc
    _validate_result(result, f"{dataset}/{stem}")
    output_path = Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset, "stem": stem,
        "design_version": DABE_R1_DESIGN_VERSION,
        "source_dabe_version": "v2", "source_augs": ["identity"], "source_num_views": 1,
        "source_dabe_cache_path": str(dabe_path),
        "source_feature_cache_path": str(feature_path), "image_path": str(image_path),
        **result,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    diagnostics = result["diagnostics"]
    return {
        "manifest_row": {
            "dataset": dataset, "stem": stem, "cache_path": str(output_path.resolve()),
            "source_dabe_cache_path": str(dabe_path),
            "source_feature_cache_path": str(feature_path),
            "design_version": DABE_R1_DESIGN_VERSION, "shape": [1, 37, 37],
        },
        "m0_error": max(float(diagnostics["m0_cached_r1_max_abs"]), float(diagnostics["m0_current_function_max_abs"])),
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_r1_design_cache(
    config_path, dabe_root, out_root, split="test", max_samples=-1,
    overwrite=False, overwrite_reason="", workers=1, torch_threads=1,
):
    started = time.time()
    config_path, dabe_root, output_root = map(lambda p: Path(p).resolve(), (config_path, dabe_root, out_root))
    if split != "test" or max_samples == 0 or max_samples < -1:
        raise ValueError("Frozen protocol requires split=test and max_samples=-1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers/torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    feature_input_size = int(getattr(cfg, "DINO", {}).get("feature_input_size", 0))
    if feature_input_size != 296:
        raise ValueError("DINO feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    frozen = {
        "GRID": 37, "LOSS_SIZE": 68, "SIGMA_F": .10, "SIGMA_C": .05, "SIGMA_E": .30,
        "TAU_BC": .30, "BG_ANCHOR_TOP_PERCENT": 30.0, "BG_ANCHOR_MIN_RATIO": .05,
        "BG_ANCHOR_FALLBACK_TOP_PERCENT": 40.0, "K_RECON": 32,
        "LAMBDA_COLOR_RECON": .20, "SIGMA_COLOR_RECON": .05, "TAU_RECON": .07,
    }
    for key, expected in frozen.items():
        if float(params[key]) != float(expected):
            raise ValueError(f"Frozen parameter {key} must be {expected}, got {params[key]}")
    dabe_manifest = dabe_root / f"manifest_{split}.jsonl"
    feature_manifest = feature_manifest_path(cfg, split).resolve()
    dabe_rows, dabe_map = _manifest_map(dabe_manifest)
    feature_rows, feature_map = _manifest_map(feature_manifest)
    if set(dabe_map) != set(feature_map):
        raise RuntimeError("DABE/feature manifest keys differ")
    if max_samples == -1 and (len(dabe_rows) != EXPECTED_TOTAL or len(feature_rows) != EXPECTED_TOTAL):
        raise RuntimeError(f"Full run requires {EXPECTED_TOTAL} source rows")
    if max_samples > 0 and len(dabe_rows) < max_samples:
        raise RuntimeError("Requested more samples than available")
    selected = dabe_rows if max_samples == -1 else dabe_rows[:max_samples]
    feature_root = feature_manifest.parent.resolve()
    _validate_output_root(output_root, [dabe_root, feature_root])
    dabe_paths = [dabe_manifest] + [Path(row["cache_path"]).resolve() for row in dabe_rows]
    feature_paths = [feature_manifest] + [Path(row["cache_path"]).resolve() for row in feature_rows]
    dabe_before, feature_before = _snapshot(dabe_paths), _snapshot(feature_paths)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_root}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    log_handle = (output_root / "cache_build.log").open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message, flush=True); log_handle.write(str(message) + "\n")

    output_rows, max_error, max_worker_rss = [], 0.0, 0.0
    try:
        for line in (
            "design_version = dabe_r1_design_v1", "gt_used_for_generation = false",
            "dino_forward_used = false", "source_dabe_cache_modified = false",
            "source_feature_cache_modified = false", "dataset_specific_rule = false",
            "threshold_search_used = false", "external_model_used = false",
            "trainable_parameter_count = 0",
        ): log(line)
        log(f"source_dabe_manifest = {dabe_manifest}")
        log(f"source_feature_manifest = {feature_manifest}")
        log(f"output_root = {output_root}")
        log(f"workers = {workers}"); log(f"torch_threads_per_worker = {torch_threads}")
        log(f"overwrite = {str(bool(overwrite)).lower()}")
        if overwrite: log(f"overwrite_reason = {overwrite_reason.strip()}")
        tasks = [{
            "dabe_row": row,
            "feature_row": feature_map[(str(row["dataset"]), str(row["stem"]))],
            "effective_params": params, "feature_input_size": feature_input_size,
            "output_root": str(output_root), "split": split,
        } for row in selected]
        if workers == 1:
            iterator, executor = map(_process_one, tasks), None
        else:
            executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(torch_threads,))
            iterator = executor.map(_process_one, tasks, chunksize=1)
        try:
            for index, item in enumerate(iterator, 1):
                output_rows.append(item["manifest_row"])
                max_error = max(max_error, float(item["m0_error"]))
                max_worker_rss = max(max_worker_rss, float(item["worker_peak_rss_mb"]))
                if index == len(tasks) or index % 100 == 0: log(f"processed = {index}/{len(tasks)}")
        finally:
            if executor is not None: executor.shutdown(wait=True, cancel_futures=True)
        output_keys = {(row["dataset"], row["stem"]) for row in output_rows}
        selected_keys = {(str(row["dataset"]), str(row["stem"])) for row in selected}
        if len(output_rows) != len(output_keys) or output_keys != selected_keys:
            raise RuntimeError("Output keys do not match selected sources")
        if max_samples == -1 and len(output_rows) != EXPECTED_TOTAL:
            raise RuntimeError("Full output count mismatch")
        write_jsonl(output_root / f"manifest_{split}.jsonl", output_rows)
        if _snapshot(dabe_paths) != dabe_before: raise RuntimeError("Source DABE cache changed")
        if _snapshot(feature_paths) != feature_before: raise RuntimeError("Source feature cache changed")
        elapsed = time.time() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        commit, status = _git_metadata()
        protocol = {
            "design_version": DABE_R1_DESIGN_VERSION,
            "source_dabe_root": str(dabe_root), "source_dabe_manifest": str(dabe_manifest),
            "source_feature_manifest": str(feature_manifest), "output_root": str(output_root),
            "split": split, "num_samples": len(output_rows), "backbone": "DINOv1-S/8",
            "feature_type": "last-block attention key", "feature_input_size": 296,
            "grid": 37, "patch_size": 8, "source_augs": ["identity"],
            "gt_used_for_generation": False, "dino_forward_used": False,
            "dataset_specific_rule": False, "threshold_search_used": False,
            "external_model_used": False, "trainable_parameter_count": 0,
            "source_dabe_cache_modified": False, "source_feature_cache_modified": False,
            "fixed_candidates": {
                "M0": "BW2 + Node-Sobel + Prototype Residual",
                "M1": "BW1 + Node-Sobel + Prototype Residual",
                "M2": "BW2 + No Edge + Prototype Residual",
                "M3": "BW2 + HR Interface Edge + Prototype Residual",
                "M4": "BW2 + Node-Sobel + Support-Consistent Residual",
                "M5": "BW1 + Node-Sobel + Support-Consistent Residual",
            },
            "effective_dabe_params": params,
            "support_consistent_formula": "sum_b a_ib*(1-cos(f_i,f_b)) + 0.2*sum_b a_ib*||c_i-c_b||_2",
            "hr_interface_formula": "feature/color graph cost plus mean 296x296 Sobel on 2-pixel patch interface",
            "git_commit": commit, "git_status_short": status,
            "config_path": str(config_path), "config_sha256": _sha256(config_path),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_r1_design.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "baseline_recompute_max_abs": max_error,
            "all_outputs_finite": True, "all_probability_outputs_in_unit_range": True,
            "overwrite": bool(overwrite), "overwrite_reason": overwrite_reason.strip(),
            "elapsed_seconds": elapsed, "average_seconds_per_image": elapsed / len(output_rows),
            "peak_rss_mb": peak, "max_worker_peak_rss_mb": max_worker_rss,
            "workers": workers, "torch_threads_per_worker": torch_threads,
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_dabe_cache_modified = false (verified size and mtime)")
        log("source_feature_cache_modified = false (verified size and mtime)")
        log(f"num_samples = {len(output_rows)}")
        log(f"m0_cached_r1_global_max_abs = {max_error:.12g}")
        log("all_outputs_finite = true"); log("all_probability_outputs_in_unit_range = true")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(output_rows):.6f}")
        log(f"peak_rss_mb = {peak:.3f}"); log(f"max_worker_peak_rss_mb = {max_worker_rss:.3f}")
        return protocol
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--out_root", required=True); parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_r1_design_cache(
        args.config, args.dabe_root, args.out_root, args.split, args.max_samples,
        args.overwrite, args.overwrite_reason, args.workers, args.torch_threads,
    )
