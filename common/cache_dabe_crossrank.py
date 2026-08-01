#!/usr/bin/env python3
"""Build GT-free CRMC-v1 candidates from frozen DABE and BGNull caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from common.dabe_crossrank import (  # noqa: E402
    DABE_CROSSRANK_VERSION,
    build_crossrank_r1dist,
    crossrank_diagnostics,
)
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_TOTAL = 6473


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
        cache_path = Path(row["cache_path"]).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        mapping[key] = row
    return rows, mapping


def _snapshot(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _same_or_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents


def _validate_output_root(output: Path, source_roots: list[Path]) -> None:
    for source in source_roots:
        if _same_or_nested(source, output) or _same_or_nested(output, source):
            raise ValueError(
                f"Output must not equal or nest with a source root: {output} vs {source}"
            )


def _validate_bgnull_protocol(
    protocol_path: Path,
    dabe_manifest: Path,
    split: str,
) -> dict:
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read BGNull protocol: {protocol_path}") from exc
    required = {
        "bgnull_version": "dabe_bgnull_v1",
        "split": split,
        "backbone": "DINOv1-S/8",
        "source_augs": ["identity"],
        "gt_used_for_generation": False,
        "dino_forward_used": False,
        "cross_exclusion_metric": "chebyshev",
        "cross_exclusion_radius": 1,
        "k_recon": 32,
    }
    for field, expected in required.items():
        if protocol.get(field) != expected:
            raise ValueError(
                f"BGNull protocol {field} must be {expected!r}, got {protocol.get(field)!r}"
            )
    current_dabe_hash = _sha256(dabe_manifest)
    if protocol.get("source_dabe_manifest_sha256") != current_dabe_hash:
        raise RuntimeError(
            "BGNull source_dabe_manifest_sha256 does not match current DABE manifest"
        )
    return protocol


def _probability(payload: dict, field: str, path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a torch.Tensor: {path}")
    if tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]: {path}")
    if not value.dtype.is_floating_point or value.requires_grad:
        raise ValueError(f"{field} must be detached floating tensor: {path}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"{field} outside [0,1]: {path}")
    return value.detach().cpu().float().contiguous().clamp(0.0, 1.0)


def _validate_payload_key(payload: dict, dataset: str, stem: str, path: Path) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"Payload must be a dict: {path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Payload key mismatch: {path}")


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict) -> dict:
    dataset, stem = task["dataset"], task["stem"]
    dabe_path = Path(task["dabe_cache_path"]).resolve()
    bgnull_path = Path(task["bgnull_cache_path"]).resolve()
    dabe = torch_load(dabe_path, map_location="cpu")
    bgnull = torch_load(bgnull_path, map_location="cpu")
    _validate_payload_key(dabe, dataset, stem, dabe_path)
    _validate_payload_key(bgnull, dataset, stem, bgnull_path)
    if str(dabe.get("dabe_version", "")).lower() != "v2":
        raise ValueError(f"DABE payload must be v2: {dabe_path}")
    if dabe.get("augs") != ["identity"] or dabe.get("num_views") != 1:
        raise ValueError(f"DABE payload must be identity single-view: {dabe_path}")
    if bgnull.get("bgnull_version") != "dabe_bgnull_v1":
        raise ValueError(f"Wrong BGNull payload version: {bgnull_path}")
    if bgnull.get("source_augs") != ["identity"] or bgnull.get("source_num_views") != 1:
        raise ValueError(f"BGNull payload must be identity single-view: {bgnull_path}")

    dabe_r1 = _probability(dabe, "residual_pass1_37", dabe_path)
    r1 = _probability(bgnull, "n0_r1_37", bgnull_path)
    cross = _probability(bgnull, "n1_cross_r1_37", bgnull_path)
    r1_error = float(torch.max(torch.abs(r1 - dabe_r1)))
    if r1_error > 1e-6:
        raise RuntimeError(
            f"BGNull R1 differs from DABE R1 at {dataset}/{stem}: max_abs={r1_error}"
        )

    crossrank = build_crossrank_r1dist(r1, cross)
    diagnostics = crossrank_diagnostics(r1, cross, crossrank)
    if diagnostics["crossrank_monotonic_violation_count"] != 0:
        raise RuntimeError(f"CrossRank monotonic violation at {dataset}/{stem}")
    has_cross_ties = diagnostics["cross_tie_ratio"] > 0.0
    constant_fallback = diagnostics["crossrank_constant_source_fallback"]
    if not has_cross_ties and not constant_fallback:
        if diagnostics["crossrank_spearman_vs_cross"] < 1.0 - 1e-6:
            raise RuntimeError(f"CrossRank ordering invariant failed at {dataset}/{stem}")
    if (
        crossrank.dtype != torch.float32
        or crossrank.device.type != "cpu"
        or tuple(crossrank.shape) != (1, 37, 37)
        or crossrank.requires_grad
        or not crossrank.is_contiguous()
        or not torch.isfinite(crossrank).all()
        or float(crossrank.min()) < 0.0
        or float(crossrank.max()) > 1.0
    ):
        raise RuntimeError(f"Invalid CrossRank output contract at {dataset}/{stem}")

    output_path = (
        Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "stem": stem,
        "crossrank_version": DABE_CROSSRANK_VERSION,
        "source_dabe_version": "v2",
        "source_bgnull_version": "dabe_bgnull_v1",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "source_dabe_cache_path": str(dabe_path),
        "source_bgnull_cache_path": str(bgnull_path),
        "x1_crossrank_r1dist_37": crossrank,
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
            "source_dabe_cache_path": str(dabe_path),
            "source_bgnull_cache_path": str(bgnull_path),
            "crossrank_version": DABE_CROSSRANK_VERSION,
            "shape": [1, 37, 37],
        },
        "r1_error": r1_error,
        "diagnostics": diagnostics,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_crossrank_cache(
    config_path: str | Path,
    dabe_root: str | Path,
    bgnull_root: str | Path,
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
    bgnull_root = Path(bgnull_root).resolve()
    output_root = Path(out_root).resolve()
    if split != "test":
        raise ValueError("CRMC-v1 supports split=test only")
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")

    dabe_manifest = dabe_root / f"manifest_{split}.jsonl"
    bgnull_manifest = bgnull_root / f"manifest_{split}.jsonl"
    bgnull_protocol_path = bgnull_root / "protocol.json"
    bgnull_protocol = _validate_bgnull_protocol(
        bgnull_protocol_path, dabe_manifest, split
    )
    dabe_rows, dabe_map = _manifest_map(dabe_manifest)
    bgnull_rows, bgnull_map = _manifest_map(bgnull_manifest)
    if set(dabe_map) != set(bgnull_map):
        missing_bgnull = sorted(set(dabe_map) - set(bgnull_map))
        missing_dabe = sorted(set(bgnull_map) - set(dabe_map))
        raise RuntimeError(
            f"DABE/BGNull manifest keys differ; missing BGNull={missing_bgnull[:10]}, "
            f"missing DABE={missing_dabe[:10]}"
        )
    if max_samples == -1 and (len(dabe_rows) != EXPECTED_TOTAL or len(bgnull_rows) != EXPECTED_TOTAL):
        raise RuntimeError(
            f"Full run requires {EXPECTED_TOTAL} rows, got {len(dabe_rows)}/{len(bgnull_rows)}"
        )
    if max_samples > 0 and len(dabe_rows) < max_samples:
        raise RuntimeError(f"Requested {max_samples} samples from {len(dabe_rows)} rows")
    selected = dabe_rows if max_samples == -1 else dabe_rows[:max_samples]
    selected_keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
    if len(selected_keys) != len(set(selected_keys)):
        raise RuntimeError("Selected items contain duplicate keys")
    if any(key not in bgnull_map for key in selected_keys):
        raise RuntimeError("Selected items are not fully present in BGNull manifest")

    _validate_output_root(output_root, [dabe_root, bgnull_root])
    dabe_paths = [dabe_manifest] + [Path(row["cache_path"]).resolve() for row in dabe_rows]
    bgnull_paths = [bgnull_manifest, bgnull_protocol_path] + [
        Path(row["cache_path"]).resolve() for row in bgnull_rows
    ]
    dabe_before = _snapshot(dabe_paths)
    bgnull_before = _snapshot(bgnull_paths)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_root: {output_root}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires a non-empty --overwrite_reason")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    log_handle = (output_root / "cache_build.log").open(
        "w", encoding="utf-8", buffering=1
    )

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")

    output_rows = []
    r1_source_max_abs = 0.0
    fallback_count = 0
    monotonic_violation_count = 0
    sorted_invariant_violation_count = 0
    sorted_l1_values = []
    sorted_max_abs = 0.0
    max_worker_peak_rss = 0.0
    try:
        for line in (
            "crossrank_version = dabe_crossrank_v1",
            "gt_used_for_generation = false",
            "dino_forward_used = false",
            "background_reconstruction_rerun = false",
            "bgnull_reconstruction_rerun = false",
            "dataset_specific_rule = false",
            "threshold_search_used = false",
            "external_model_used = false",
            "trainable_parameter_count = 0",
        ):
            log(line)
        log(f"source_dabe_manifest = {dabe_manifest}")
        log(f"source_bgnull_manifest = {bgnull_manifest}")
        log(f"output_root = {output_root}")
        log(f"workers = {workers}")
        log(f"torch_threads_per_worker = {torch_threads}")
        log(f"overwrite = {str(bool(overwrite)).lower()}")
        if overwrite:
            log(f"overwrite_reason = {overwrite_reason.strip()}")

        tasks = [
            {
                "dataset": key[0],
                "stem": key[1],
                "dabe_cache_path": dabe_map[key]["cache_path"],
                "bgnull_cache_path": bgnull_map[key]["cache_path"],
                "output_root": str(output_root),
                "split": split,
            }
            for key in selected_keys
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
            for index, result in enumerate(iterator, 1):
                output_rows.append(result["manifest_row"])
                diagnostics = result["diagnostics"]
                r1_source_max_abs = max(r1_source_max_abs, float(result["r1_error"]))
                fallback_count += int(diagnostics["crossrank_constant_source_fallback"])
                monotonic_violation_count += int(
                    diagnostics["crossrank_monotonic_violation_count"]
                )
                sorted_invariant_violation_count += int(
                    diagnostics["crossrank_sorted_invariant_violation"]
                )
                sorted_l1_values.append(float(diagnostics["crossrank_sorted_l1_vs_r1"]))
                sorted_max_abs = max(
                    sorted_max_abs,
                    float(diagnostics["crossrank_sorted_max_abs_vs_r1"]),
                )
                max_worker_peak_rss = max(
                    max_worker_peak_rss, float(result["worker_peak_rss_mb"])
                )
                if index == len(tasks) or index % 100 == 0:
                    log(f"processed = {index}/{len(tasks)}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        output_keys = {(row["dataset"], row["stem"]) for row in output_rows}
        if len(output_rows) != len(output_keys) or output_keys != set(selected_keys):
            raise RuntimeError("Output keys do not exactly match selected items")
        if max_samples == -1 and len(output_rows) != EXPECTED_TOTAL:
            raise RuntimeError(f"Full output must have {EXPECTED_TOTAL} rows")
        write_jsonl(output_root / f"manifest_{split}.jsonl", output_rows)

        if _snapshot(dabe_paths) != dabe_before:
            raise RuntimeError("A source DABE manifest/cache changed during generation")
        if _snapshot(bgnull_paths) != bgnull_before:
            raise RuntimeError("A source BGNull manifest/cache changed during generation")
        if monotonic_violation_count != 0:
            raise RuntimeError("Full CrossRank output contains monotonic violations")

        elapsed = time.time() - started
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        git_commit, git_status = _git_metadata()
        protocol = {
            "crossrank_version": DABE_CROSSRANK_VERSION,
            "source_dabe_root": str(dabe_root),
            "source_bgnull_root": str(bgnull_root),
            "source_dabe_manifest": str(dabe_manifest),
            "source_bgnull_manifest": str(bgnull_manifest),
            "output_root": str(output_root),
            "split": split,
            "num_samples": len(output_rows),
            "backbone": "DINOv1-S/8",
            "source_augs": ["identity"],
            "gt_used_for_generation": False,
            "dino_forward_used": False,
            "background_reconstruction_rerun": False,
            "bgnull_reconstruction_rerun": False,
            "dataset_specific_rule": False,
            "threshold_search_used": False,
            "external_model_used": False,
            "trainable_parameter_count": 0,
            "source_dabe_cache_modified": False,
            "source_bgnull_cache_modified": False,
            "formula": {
                "X1-CrossRank-R1Dist":
                    "rank_transport(source=N1-CrossR1, reference=N0-R1)"
            },
            "git_commit": git_commit,
            "git_status_short": git_status,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_bgnull_manifest_sha256": _sha256(bgnull_manifest),
            "source_bgnull_protocol_sha256": _sha256(bgnull_protocol_path),
            "rank_calibration_file_sha256": _sha256(
                SCRIPT_PATH.with_name("dabe_rank_calibration.py")
            ),
            "crossrank_formula_file_sha256": _sha256(
                SCRIPT_PATH.with_name("dabe_crossrank.py")
            ),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "constant_source_fallback_count": fallback_count,
            "monotonic_violation_count": monotonic_violation_count,
            "sorted_invariant_violation_count": sorted_invariant_violation_count,
            "r1_source_max_abs": r1_source_max_abs,
            "sorted_l1_vs_r1_mean": float(sum(sorted_l1_values) / len(sorted_l1_values)),
            "sorted_max_abs_vs_r1_max": sorted_max_abs,
            "all_outputs_finite": True,
            "all_outputs_in_unit_range": True,
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
        log("source_bgnull_cache_modified = false (verified size and mtime)")
        log(f"num_samples = {len(output_rows)}")
        log(f"r1_source_max_abs = {r1_source_max_abs:.12g}")
        log(f"constant_source_fallback_count = {fallback_count}")
        log(f"monotonic_violation_count = {monotonic_violation_count}")
        log(f"sorted_invariant_violation_count = {sorted_invariant_violation_count}")
        log(f"sorted_l1_vs_r1_mean = {protocol['sorted_l1_vs_r1_mean']:.12g}")
        log(f"sorted_max_abs_vs_r1_max = {sorted_max_abs:.12g}")
        log("all_outputs_finite = true")
        log("all_outputs_in_unit_range = true")
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
    parser.add_argument("--bgnull_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_crossrank_cache(
        config_path=args.config,
        dabe_root=args.dabe_root,
        bgnull_root=args.bgnull_root,
        out_root=args.out_root,
        split=args.split,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
        workers=args.workers,
        torch_threads=args.torch_threads,
    )
