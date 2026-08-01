#!/usr/bin/env python3
"""Build GT-free DABE CVBR-v1 candidates from frozen DABE/DINO caches.

The original audit used the four-dataset test split.  The same frozen,
GT-free formula is also exposed for the 4040-image training split so a
trainable student can consume a CVBR candidate as a static pseudo target.
"""

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

from common.cache_dabe_r1_design import (  # noqa: E402
    _feature,
    _manifest_map,
    _snapshot,
    _source_r1,
    _validate_output_root,
    effective_dabe_v2_params,
)
from common.dabe_cvbr import (  # noqa: E402
    CVBR_VERSION,
    NONNEGATIVE_FIELDS,
    PROBABILITY_FIELDS,
    build_cvbr_candidates,
)
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_TOTALS = {"train": 4040, "test": 6473}
EXPECTED_AUGS = {
    "train": ["identity"],
    "test": ["identity"],
}


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


def _validate_result(result: dict, context: str):
    for field in PROBABILITY_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError(f"{field} tensor contract invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
            raise ValueError(f"{field} outside finite [0,1]: {context}")
    for field in NONNEGATIVE_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or not value.is_contiguous():
            raise ValueError(f"{field} tensor contract invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < -1e-7:
            raise ValueError(f"{field} invalid nonnegative map: {context}")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError(f"diagnostics missing: {context}")
    for key, value in diagnostics.items():
        if isinstance(value, (float, int)) and not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"diagnostic {key} is nonfinite: {context}")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _source_r1_for_split(payload: dict, row: dict, path: Path, split: str):
    expected_augs = EXPECTED_AUGS[split]
    value = _source_r1(payload, row, path)
    actual_augs = [str(item).strip().lower() for item in payload.get("augs", [])]
    if actual_augs != expected_augs:
        raise ValueError(
            f"DABE source augmentation mismatch: {actual_augs} != "
            f"{expected_augs} | {path}"
        )
    stored_views = payload.get("num_views")
    if stored_views is not None and int(stored_views) != len(expected_augs):
        raise ValueError(
            f"DABE source num_views mismatch: {stored_views} != "
            f"{len(expected_augs)} | {path}"
        )
    return value, actual_augs


def _process_one(task: dict):
    row, feature_row = task["dabe_row"], task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    dabe_path = Path(row["cache_path"]).resolve()
    feature_path = Path(feature_row["cache_path"]).resolve()
    dabe = torch_load(dabe_path, map_location="cpu")
    feature_payload = torch_load(feature_path, map_location="cpu")
    cached, source_augs = _source_r1_for_split(
        dabe, row, dabe_path, task["split"]
    )
    feature = _feature(feature_payload, dataset, stem, feature_path)
    image_text = row.get("image_path") or feature_row.get("image_path") or dabe.get("image_path")
    if not image_text:
        raise KeyError(f"image_path missing for {dataset}/{stem}")
    image_path = Path(image_text).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    try:
        result = build_cvbr_candidates(
            feature_37=feature,
            image_path=str(image_path),
            cached_r1_37=cached,
            effective_params=task["effective_params"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"CVBR failure at {dataset}/{stem}; dabe={dabe_path}; feature={feature_path}: {exc}"
        ) from exc
    _validate_result(result, f"{dataset}/{stem}")
    output_path = Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset, "stem": stem, "cvbr_version": CVBR_VERSION,
        "source_dabe_version": "v2", "source_augs": source_augs,
        "source_num_views": len(source_augs),
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
            "cvbr_version": CVBR_VERSION, "shape": [1, 37, 37],
            "source_augs": source_augs, "source_num_views": len(source_augs),
        },
        "b0_error": float(diagnostics["b0_cached_r1_max_abs"]),
        "weighted_error": float(diagnostics["weighted_dijkstra_unit_reliability_max_abs"]),
        "fallback_count": int(diagnostics["cross_fallback_count"]),
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_cvbr_cache(
    config_path, dabe_root, out_root, split="test", max_samples=-1,
    overwrite=False, overwrite_reason="", workers=1, torch_threads=1,
):
    started = time.time()
    config_path, dabe_root, output_root = map(
        lambda value: Path(value).resolve(), (config_path, dabe_root, out_root)
    )
    split = str(split).strip().lower()
    if split not in EXPECTED_TOTALS or max_samples == 0 or max_samples < -1:
        raise ValueError(
            "Frozen protocol requires split=train/test and "
            "max_samples=-1 or positive"
        )
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers/torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    if int(getattr(cfg, "DINO", {}).get("feature_input_size", 0)) != 296:
        raise ValueError("DINO feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    frozen = {
        "GRID": 37, "LOSS_SIZE": 68, "SIGMA_F": .10, "SIGMA_C": .05,
        "SIGMA_E": .30, "TAU_BC": .30, "BG_ANCHOR_TOP_PERCENT": 30.0,
        "BG_ANCHOR_MIN_RATIO": .05, "BG_ANCHOR_FALLBACK_TOP_PERCENT": 40.0,
        "K_RECON": 32, "LAMBDA_COLOR_RECON": .20,
        "SIGMA_COLOR_RECON": .05, "TAU_RECON": .07,
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
    expected_total = EXPECTED_TOTALS[split]
    if max_samples == -1 and (
        len(dabe_rows) != expected_total or len(feature_rows) != expected_total
    ):
        raise RuntimeError(
            f"Full {split} run requires {expected_total} source rows"
        )
    if max_samples > 0 and len(dabe_rows) < max_samples:
        raise RuntimeError("Requested more samples than available")
    selected = dabe_rows if max_samples == -1 else dabe_rows[:max_samples]
    _validate_output_root(output_root, [dabe_root, feature_manifest.parent.resolve()])
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
    rows, max_b0, max_weighted, fallback_total, max_worker_rss = [], 0.0, 0.0, 0, 0.0
    try:
        for line in (
            "cvbr_version = dabe_cvbr_v1", "gt_used_for_generation = false",
            "dino_forward_used = false", "training_used = false",
            "threshold_search_used = false", "dataset_specific_rule = false",
            "sample_routing_used = false", "external_model_used = false",
            "trainable_parameter_count = 0", "source_dabe_cache_modified = false",
            "source_feature_cache_modified = false",
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
            "effective_params": params, "output_root": str(output_root), "split": split,
        } for row in selected]
        if workers == 1:
            iterator, executor = map(_process_one, tasks), None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker, initargs=(torch_threads,)
            )
            iterator = executor.map(_process_one, tasks, chunksize=1)
        try:
            for index, item in enumerate(iterator, 1):
                rows.append(item["manifest_row"])
                max_b0 = max(max_b0, item["b0_error"])
                max_weighted = max(max_weighted, item["weighted_error"])
                fallback_total += item["fallback_count"]
                max_worker_rss = max(max_worker_rss, item["worker_peak_rss_mb"])
                if index == len(tasks) or index % 100 == 0: log(f"processed = {index}/{len(tasks)}")
        finally:
            if executor is not None: executor.shutdown(wait=True, cancel_futures=True)
        output_keys = {(row["dataset"], row["stem"]) for row in rows}
        selected_keys = {(str(row["dataset"]), str(row["stem"])) for row in selected}
        if len(rows) != len(output_keys) or output_keys != selected_keys:
            raise RuntimeError("Output keys do not match selected sources")
        if max_samples == -1 and len(rows) != expected_total:
            raise RuntimeError("Full output count mismatch")
        write_jsonl(output_root / f"manifest_{split}.jsonl", rows)
        if _snapshot(dabe_paths) != dabe_before: raise RuntimeError("Source DABE cache changed")
        if _snapshot(feature_paths) != feature_before: raise RuntimeError("Source feature cache changed")
        elapsed = time.time() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        commit, status = _git_metadata()
        protocol = {
            "cvbr_version": CVBR_VERSION,
            "source_dabe_root": str(dabe_root), "source_dabe_manifest": str(dabe_manifest),
            "source_feature_manifest": str(feature_manifest), "output_root": str(output_root),
            "split": split, "num_samples": len(rows), "backbone": "DINOv1-S/8",
            "feature_type": "last-block attention key", "feature_input_size": 296,
            "grid": 37, "source_augs": EXPECTED_AUGS[split],
            "source_num_views": len(EXPECTED_AUGS[split]),
            "gt_used_for_generation": False, "dino_forward_used": False,
            "training_used": False, "threshold_search_used": False,
            "dataset_specific_rule": False, "sample_routing_used": False,
            "external_model_used": False, "trainable_parameter_count": 0,
            "source_dabe_cache_modified": False, "source_feature_cache_modified": False,
            "fixed_candidates": {
                "B0": "BW2, q=1", "B1": "BW1, q=1",
                "V1": "BW2, ring1 q=1, ring2 q=CVBR",
                "V2": "BW2, all boundary q=CVBR",
            },
            "cvbr_formula": {
                "cross_error": "current raw Prototype residual after radius-1 B1-anchor exclusion",
                "reference_median": "median B1 cross error",
                "reference_mad": "max(1.4826 * median absolute deviation, 1e-6)",
                "reliability": "q=clip(exp(-relu((e-m)/s)),1e-6,1)",
                "source_cost": "-log(q)",
            },
            "implementation_bug_fixes": [{
                "stage": "pre-full-run",
                "issue": "float32 q_min was compared against a Python float64 literal and rejected at the exact frozen lower bound",
                "fix": "validate against the float32-representable form of 1e-6",
                "affected_full_candidates": 0,
                "action": "unit tests and sanity rerun; partial aborted full output rebuilt from scratch",
            }],
            "effective_dabe_params": params,
            "git_commit": commit, "git_status_short": status,
            "config_path": str(config_path), "config_sha256": _sha256(config_path),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_cvbr.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "baseline_recompute_max_abs": max_b0,
            "weighted_dijkstra_unit_reliability_max_abs": max_weighted,
            "cross_fallback_count": fallback_total,
            "all_outputs_finite": True, "all_probability_outputs_in_unit_range": True,
            "overwrite": bool(overwrite), "overwrite_reason": overwrite_reason.strip(),
            "elapsed_seconds": elapsed, "average_seconds_per_image": elapsed / len(rows),
            "peak_rss_mb": peak, "max_worker_peak_rss_mb": max_worker_rss,
            "workers": workers, "torch_threads_per_worker": torch_threads,
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_dabe_cache_modified = false (verified size and mtime)")
        log("source_feature_cache_modified = false (verified size and mtime)")
        log(f"num_samples = {len(rows)}")
        log(f"b0_cached_r1_global_max_abs = {max_b0:.12g}")
        log(f"weighted_dijkstra_unit_reliability_max_abs = {max_weighted:.12g}")
        log(f"cross_fallback_count = {fallback_total}")
        log("all_outputs_finite = true"); log("all_probability_outputs_in_unit_range = true")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(rows):.6f}")
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
    build_cvbr_cache(
        args.config, args.dabe_root, args.out_root, args.split, args.max_samples,
        args.overwrite, args.overwrite_reason, args.workers, args.torch_threads,
    )
