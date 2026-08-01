#!/usr/bin/env python3
"""Build identity-only training R1/V1-CVBR caches from frozen DINO features.

This path stops before DABE foreground seeds, random walk, evidence gating and
final p_dabe construction.  It computes the first background reconstruction
residual (R1) and the V1-CVBR-SecondRing candidate in one pass and stores only
those two response maps plus provenance/diagnostics.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_cvbr import _git_metadata, _sha256  # noqa: E402
from common.cache_dabe_r1_design import (  # noqa: E402
    _feature,
    _manifest_map,
    _snapshot,
    _validate_output_root,
    effective_dabe_v2_params,
)
from common.dabe_cvbr import CVBR_VERSION, build_cvbr_candidates  # noqa: E402
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_TRAIN_TOTAL = 4040
STORED_FIELDS = ("b0_r1_bw2_37", "v1_cvbr_second_ring_37")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _validate_response(value, name, context):
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{name} must be Tensor[1,37,37]: {context}")
    if (
        value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.requires_grad
        or not value.is_contiguous()
    ):
        raise ValueError(f"{name} tensor contract invalid: {context}")
    if (
        not torch.isfinite(value).all()
        or float(value.min()) < 0.0
        or float(value.max()) > 1.0
    ):
        raise ValueError(f"{name} must remain finite in [0,1]: {context}")


def _process_one(task):
    row = task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    feature_path = Path(row["cache_path"]).resolve()
    feature_payload = torch_load(feature_path, map_location="cpu")
    feature = _feature(feature_payload, dataset, stem, feature_path)
    image_text = row.get("image_path") or feature_payload.get("image_path")
    if not image_text:
        raise KeyError(f"image_path missing for {dataset}/{stem}")
    image_path = Path(image_text).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    try:
        result = build_cvbr_candidates(
            feature_37=feature,
            image_path=str(image_path),
            cached_r1_37=None,
            effective_params=task["effective_params"],
            include_v2=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Identity R1/CVBR failure at {dataset}/{stem}; "
            f"feature={feature_path}: {exc}"
        ) from exc
    for field in STORED_FIELDS:
        _validate_response(result[field], field, f"{dataset}/{stem}")
    diagnostics = dict(result["diagnostics"])
    if diagnostics.get("b0_external_cache_checked") is not False:
        raise RuntimeError("Direct identity generation must not read a DABE cache")

    output_path = (
        Path(task["output_root"]) / "train" / dataset / f"{stem}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "stem": stem,
        "image_path": str(image_path),
        "cvbr_version": CVBR_VERSION,
        "source_dabe_version": "v2",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "generation_stage": "identity_r1_plus_cvbr_v1_only",
        "dino_forward_used": False,
        "gt_used_for_generation": False,
        "foreground_seed_used": False,
        "random_walk_used": False,
        "evidence_gate_used": False,
        "p_dabe_generated": False,
        "b0_r1_bw2_37": result["b0_r1_bw2_37"],
        "v1_cvbr_second_ring_37": result["v1_cvbr_second_ring_37"],
        "diagnostics": diagnostics,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return {
        "manifest_row": {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path.resolve()),
            "source_feature_cache_path": str(feature_path),
            "cvbr_version": CVBR_VERSION,
            "source_augs": ["identity"],
            "source_num_views": 1,
            "shape": [1, 37, 37],
        },
        "fallback_count": int(diagnostics["cross_fallback_count"]),
        "weighted_error": float(
            diagnostics["weighted_dijkstra_unit_reliability_max_abs"]
        ),
        "worker_peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }


def build_identity_train_cache(
    config_path,
    out_root,
    max_samples=-1,
    overwrite=False,
    overwrite_reason="",
    workers=1,
    torch_threads=1,
):
    started = time.time()
    config_path = Path(config_path).resolve()
    output_root = Path(out_root).resolve()
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers/torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    if int(getattr(cfg, "DINO", {}).get("feature_input_size", 0)) != 296:
        raise ValueError("DINO feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    feature_manifest = feature_manifest_path(cfg, "train").resolve()
    feature_rows, feature_map = _manifest_map(feature_manifest)
    if len(feature_rows) != EXPECTED_TRAIN_TOTAL:
        raise RuntimeError(
            f"Training feature manifest must contain {EXPECTED_TRAIN_TOTAL} rows"
        )
    selected = feature_rows if max_samples == -1 else feature_rows[:max_samples]
    if max_samples > 0 and len(selected) != max_samples:
        raise RuntimeError("Requested more samples than available")
    if len(feature_map) != len(feature_rows):
        raise RuntimeError("Training feature manifest contains duplicate keys")
    _validate_output_root(output_root, [feature_manifest.parent.resolve()])
    source_paths = [feature_manifest] + [
        Path(row["cache_path"]).resolve() for row in feature_rows
    ]
    source_before = _snapshot(source_paths)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_root}")
        if not str(overwrite_reason).strip():
            raise ValueError("--overwrite requires --overwrite_reason")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    log_handle = (output_root / "cache_build.log").open(
        "w", encoding="utf-8", buffering=1
    )

    def log(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    rows = []
    fallback_total = 0
    max_weighted_error = 0.0
    max_worker_rss = 0.0
    try:
        for line in (
            "generation_stage = identity_r1_plus_cvbr_v1_only",
            "source_augs = identity",
            "gt_used_for_generation = false",
            "dino_forward_used = false",
            "full_dabe_v2_used = false",
            "foreground_seed_used = false",
            "random_walk_used = false",
            "evidence_gate_used = false",
            "p_dabe_generated = false",
        ):
            log(line)
        log(f"source_feature_manifest = {feature_manifest}")
        log(f"output_root = {output_root}")
        log(f"num_samples = {len(selected)}")
        log(f"workers = {workers}")
        log(f"torch_threads_per_worker = {torch_threads}")
        tasks = [
            {
                "feature_row": row,
                "effective_params": params,
                "output_root": str(output_root),
            }
            for row in selected
        ]
        if workers == 1:
            iterator, executor = map(_process_one, tasks), None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(torch_threads,),
            )
            iterator = executor.map(_process_one, tasks, chunksize=1)
        try:
            for index, item in enumerate(iterator, 1):
                rows.append(item["manifest_row"])
                fallback_total += item["fallback_count"]
                max_weighted_error = max(
                    max_weighted_error, item["weighted_error"]
                )
                max_worker_rss = max(
                    max_worker_rss, item["worker_peak_rss_mb"]
                )
                if index == len(tasks) or index % 100 == 0:
                    log(f"processed = {index}/{len(tasks)}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        keys = {(row["dataset"], row["stem"]) for row in rows}
        expected_keys = {
            (str(row["dataset"]), str(row["stem"])) for row in selected
        }
        if len(rows) != len(keys) or keys != expected_keys:
            raise RuntimeError("Output keys do not match selected features")
        write_jsonl(output_root / "manifest_train.jsonl", rows)
        if _snapshot(source_paths) != source_before:
            raise RuntimeError("Source feature cache changed")
        elapsed = time.time() - started
        commit, status = _git_metadata()
        protocol = {
            "cvbr_version": CVBR_VERSION,
            "generation_stage": "identity_r1_plus_cvbr_v1_only",
            "split": "train",
            "num_samples": len(rows),
            "source_feature_manifest": str(feature_manifest),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "source_augs": ["identity"],
            "source_num_views": 1,
            "grid": 37,
            "stored_fields": list(STORED_FIELDS),
            "gt_used_for_generation": False,
            "dino_forward_used": False,
            "full_dabe_v2_used": False,
            "foreground_seed_used": False,
            "random_walk_used": False,
            "evidence_gate_used": False,
            "p_dabe_generated": False,
            "source_feature_cache_modified": False,
            "cross_fallback_count": fallback_total,
            "weighted_dijkstra_unit_reliability_max_abs": max_weighted_error,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "formula_file_sha256": _sha256(
                SCRIPT_PATH.with_name("dabe_cvbr.py")
            ),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "git_commit": commit,
            "git_status_short": status,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(rows),
            "peak_rss_mb": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / 1024.0,
            "max_worker_peak_rss_mb": max_worker_rss,
            "overwrite": bool(overwrite),
            "overwrite_reason": str(overwrite_reason).strip(),
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_feature_cache_modified = false")
        log(f"cross_fallback_count = {fallback_total}")
        log(f"elapsed_seconds = {elapsed:.3f}")
        return protocol
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_identity_train_cache(
        args.config,
        args.out_root,
        args.max_samples,
        args.overwrite,
        args.overwrite_reason,
        args.workers,
        args.torch_threads,
    )
