#!/usr/bin/env python3
"""Build identity-only P1-RPR training targets from frozen DINO features.

The builder performs no DINO forward and never reads training GT.  For each
training image it reconstructs the frozen identity CVBR support required by
RPR-v1, applies the P1 second-ring post-TopK reliability reweighting, and stores
only B0/R1 plus the P1 response needed by the linear pure-Student experiment.
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
    _validate_output_root,
    effective_dabe_v2_params,
)
from common.dabe_cvbr import (  # noqa: E402
    CVBR_VERSION,
    build_cvbr_rpr_p1_support,
)
from common.dabe_rpr import (  # noqa: E402
    DABE_RPR_VERSION,
    build_rpr_p1_candidate,
)
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_TRAIN_TOTAL = 4040
STORED_FIELDS = ("b0_r1_bw2_37", "p1_rpr_secondring_37")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _validate_response(value, name: str, context: str):
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{name} must be Tensor[1,37,37]: {context}")
    value = value.detach().cpu()
    if value.dtype != torch.float32 or not value.is_contiguous():
        raise ValueError(f"{name} tensor contract invalid: {context}")
    if (
        not torch.isfinite(value).all()
        or float(value.min()) < 0.0
        or float(value.max()) > 1.0
    ):
        raise ValueError(f"{name} must remain finite in [0,1]: {context}")


def _manifest_row(dataset, stem, output_path, feature_path):
    return {
        "dataset": dataset,
        "stem": stem,
        "cache_path": str(output_path.resolve()),
        "source_feature_cache_path": str(feature_path.resolve()),
        "backbone_key": "dinov1-s8",
        "cvbr_version": CVBR_VERSION,
        "rpr_version": DABE_RPR_VERSION,
        "source_augs": ["identity"],
        "source_num_views": 1,
        "shape": [1, 37, 37],
    }


def _resume_one(task):
    row = task["feature_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    feature_path = Path(row["cache_path"]).resolve()
    output_path = Path(task["output_root"]) / "train" / dataset / f"{stem}.pt"
    payload = torch_load(output_path, map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"resumed RPR identity mismatch: {output_path}")
    if payload.get("rpr_version") != DABE_RPR_VERSION:
        raise RuntimeError(f"resumed RPR version mismatch: {output_path}")
    if Path(payload.get("source_feature_cache_path", "")).resolve() != feature_path:
        raise RuntimeError(f"resumed RPR feature provenance mismatch: {output_path}")
    for field in STORED_FIELDS:
        _validate_response(payload.get(field), field, f"resume:{dataset}/{stem}")
    return {
        "manifest_row": _manifest_row(
            dataset, stem, output_path, feature_path
        ),
        "p1_area": float((payload["p1_rpr_secondring_37"] > 0.5).float().mean()),
        "rare_u_count": int(
            payload.get("diagnostics", {}).get(
                "p1_u_monotonicity_violation_count", 0
            )
        ),
        "rare_u_max": float(
            payload.get("diagnostics", {}).get(
                "p1_u_monotonicity_max_violation", 0.0
            )
        ),
        "worker_peak_rss_mb": 0.0,
    }


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

    cvbr_result = build_cvbr_rpr_p1_support(
        feature_37=feature,
        image_path=str(image_path),
        cached_r1_37=None,
        effective_params=task["effective_params"],
    )
    cvbr_payload = {
        "dataset": dataset,
        "stem": stem,
        "cvbr_version": CVBR_VERSION,
        "source_augs": ["identity"],
        "source_num_views": 1,
        **cvbr_result,
    }
    rpr_result = build_rpr_p1_candidate(
        feature_37=feature,
        image_path=str(image_path),
        cached_r1_37=cvbr_result["b0_r1_bw2_37"],
        cvbr_payload=cvbr_payload,
        effective_params=task["effective_params"],
    )
    responses = {
        "b0_r1_bw2_37": cvbr_result["b0_r1_bw2_37"],
        "p1_rpr_secondring_37": rpr_result["p1_rpr_secondring_37"],
    }
    for field, value in responses.items():
        _validate_response(value, field, f"{dataset}/{stem}")

    output_path = Path(task["output_root"]) / "train" / dataset / f"{stem}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(rpr_result["diagnostics"])
    payload = {
        "dataset": dataset,
        "stem": stem,
        "image_path": str(image_path),
        "backbone_key": "dinov1-s8",
        "source_dabe_version": "v2",
        "cvbr_version": CVBR_VERSION,
        "rpr_version": DABE_RPR_VERSION,
        "source_augs": ["identity"],
        "source_num_views": 1,
        "source_feature_cache_path": str(feature_path),
        "generation_stage": "identity_cvbr_support_plus_rpr_p1_only",
        "dino_forward_used": False,
        "gt_used_for_generation": False,
        "training_used_for_generation": False,
        **responses,
        "diagnostics": diagnostics,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return {
        "manifest_row": _manifest_row(dataset, stem, output_path, feature_path),
        "p1_area": float((responses["p1_rpr_secondring_37"] > 0.5).float().mean()),
        "rare_u_count": int(diagnostics["p1_u_monotonicity_violation_count"]),
        "rare_u_max": float(diagnostics["p1_u_monotonicity_max_violation"]),
        "worker_peak_rss_mb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }


def build_identity_rpr_train_cache(
    config_path,
    out_root,
    max_samples=-1,
    overwrite=False,
    overwrite_reason="",
    resume=False,
    workers=2,
    torch_threads=12,
):
    started = time.time()
    config_path = Path(config_path).resolve()
    output_root = Path(out_root).resolve()
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers/torch_threads must be positive")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    if int(getattr(cfg, "DINO", {}).get("feature_input_size", 0)) != 296:
        raise ValueError("DINO feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    feature_manifest = feature_manifest_path(cfg, "train").resolve()
    feature_rows, feature_map = _manifest_map(feature_manifest)
    if len(feature_rows) != EXPECTED_TRAIN_TOTAL or len(feature_map) != len(feature_rows):
        raise RuntimeError("Training feature manifest must contain 4040 unique rows")
    selected = feature_rows if max_samples == -1 else feature_rows[:max_samples]
    if max_samples > 0 and len(selected) != max_samples:
        raise RuntimeError("Requested more samples than available")
    _validate_output_root(output_root, [feature_manifest.parent.resolve()])

    if output_root.exists():
        if overwrite:
            if not str(overwrite_reason).strip():
                raise ValueError("--overwrite requires --overwrite_reason")
            shutil.rmtree(output_root)
        elif not resume:
            raise FileExistsError(f"Refusing to overwrite {output_root}")
        elif (output_root / "manifest_train.jsonl").exists():
            raise RuntimeError("Completed RPR cache already has manifest_train.jsonl")
    output_root.mkdir(parents=True, exist_ok=True)
    log_handle = (output_root / "cache_build.log").open(
        "a" if resume else "w", encoding="utf-8", buffering=1
    )

    def log(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    try:
        log("generation_stage = identity_cvbr_support_plus_rpr_p1_only")
        log("source_augs = identity")
        log("gt_used_for_generation = false")
        log("dino_forward_used = false")
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
        items = {}
        if resume:
            for task in tasks:
                row = task["feature_row"]
                key = (str(row["dataset"]), str(row["stem"]))
                path = output_root / "train" / key[0] / f"{key[1]}.pt"
                if path.is_file():
                    items[key] = _resume_one(task)
            log(f"resumed_existing = {len(items)}")
        missing = [
            task
            for task in tasks
            if (
                str(task["feature_row"]["dataset"]),
                str(task["feature_row"]["stem"]),
            )
            not in items
        ]
        log(f"missing_to_generate = {len(missing)}")
        if workers == 1:
            iterator, executor = map(_process_one, missing), None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(torch_threads,),
            )
            iterator = executor.map(_process_one, missing, chunksize=1)
        try:
            for index, item in enumerate(iterator, 1):
                row = item["manifest_row"]
                items[(row["dataset"], row["stem"])] = item
                if index == len(missing) or index % 100 == 0:
                    log(f"generated_missing = {index}/{len(missing)}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
        if set(items) != set(keys):
            raise RuntimeError("Output keys do not match selected training features")
        rows = [items[key]["manifest_row"] for key in keys]
        write_jsonl(output_root / "manifest_train.jsonl", rows)
        elapsed = time.time() - started
        rare_count = sum(item["rare_u_count"] for item in items.values())
        rare_max = max((item["rare_u_max"] for item in items.values()), default=0.0)
        commit, status = _git_metadata()
        protocol = {
            "rpr_version": DABE_RPR_VERSION,
            "cvbr_version": CVBR_VERSION,
            "generation_stage": "identity_cvbr_support_plus_rpr_p1_only",
            "split": "train",
            "num_samples": len(rows),
            "backbone": "DINOv1-S/8",
            "feature_input_size": 296,
            "grid": 37,
            "stored_fields": list(STORED_FIELDS),
            "source_feature_manifest": str(feature_manifest),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "source_augs": ["identity"],
            "source_num_views": 1,
            "gt_used_for_generation": False,
            "dino_forward_used": False,
            "training_used_for_generation": False,
            "threshold_search_used": False,
            "p1_formula": "B0 TopK fixed, q_v1 post-TopK softmax reweight",
            "rare_numerical_exceptions_allowed": True,
            "p1_u_monotonicity_violation_count": rare_count,
            "p1_u_monotonicity_max_violation": rare_max,
            "p1_hard_area_mean": sum(item["p1_area"] for item in items.values()) / len(items),
            "effective_dabe_params": params,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_rpr.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "git_commit": commit,
            "git_status_short": status,
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "resume": bool(resume),
            "resumed_existing_count": len(tasks) - len(missing),
            "generated_missing_count": len(missing),
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(rows),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "max_worker_peak_rss_mb": max(
                (item["worker_peak_rss_mb"] for item in items.values()), default=0.0
            ),
        }
        write_json(output_root / "protocol.json", protocol)
        log(f"p1_u_monotonicity_violation_count = {rare_count}")
        log(f"p1_u_monotonicity_max_violation = {rare_max:.12g}")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_identity_rpr_train_cache(
        args.config,
        args.out_root,
        args.max_samples,
        args.overwrite,
        args.overwrite_reason,
        args.resume,
        args.workers,
        args.torch_threads,
    )
