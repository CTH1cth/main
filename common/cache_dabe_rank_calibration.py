#!/usr/bin/env python3
"""Build GT-free DABE rank/calibration candidates from an existing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_rank_calibration import (  # noqa: E402
    RANK_CALIBRATION_EPS,
    RANK_CALIBRATION_VERSION,
    average_percentile_rank,
    build_rank_calibration_candidates,
)
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
REQUIRED_SOURCE_FIELDS = (
    "residual_pass1_37",
    "residual_37",
    "fg_score_37",
)
CANDIDATE_FIELDS = (
    "c0_f_minmax_37",
    "c1_f_rank_r1_37",
    "c2_median_rank_r1_37",
)


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


def _is_parent_or_same(left: Path, right: Path) -> bool:
    return left == right or left in right.parents


def _validate_roots(source_root: Path, output_root: Path) -> None:
    if _is_parent_or_same(source_root, output_root) or _is_parent_or_same(output_root, source_root):
        raise ValueError(
            "out_root and dabe_root must be distinct sibling trees, not equal or parent/child: "
            f"{output_root} vs {source_root}"
        )


def _validate_tensor(payload: dict, field: str, cache_path: Path) -> torch.Tensor:
    if field not in payload:
        raise KeyError(f"{field} missing from {cache_path}")
    value = payload[field]
    if not torch.is_tensor(value):
        raise TypeError(f"{field} must be a torch.Tensor: {cache_path}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{field} must have floating dtype, got {value.dtype}: {cache_path}")
    if tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be [1,37,37], got {list(value.shape)}: {cache_path}")
    if value.requires_grad:
        raise ValueError(f"{field}.requires_grad must be false: {cache_path}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains NaN/Inf: {cache_path}")
    minimum = float(value.min())
    maximum = float(value.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"{field} outside [0,1] tolerance: min={minimum}, max={maximum}: {cache_path}")
    return value.detach().cpu().float().contiguous().clamp(0.0, 1.0)


def _validate_candidate(value: torch.Tensor, field: str, cache_path: Path) -> None:
    if value.dtype != torch.float32:
        raise TypeError(f"{field} must be float32: {cache_path}")
    if tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be [1,37,37]: {cache_path}")
    if value.requires_grad or not value.is_contiguous():
        raise ValueError(f"{field} must be detached and contiguous: {cache_path}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{field} contains NaN/Inf: {cache_path}")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"{field} is outside [0,1]: {cache_path}")


def _unique_and_tie_ratio(value: torch.Tensor) -> tuple[float, float]:
    unique_ratio = float(torch.unique(value.reshape(-1)).numel()) / float(value.numel())
    return unique_ratio, 1.0 - unique_ratio


def _diagnostics(
    r1: torch.Tensor,
    residual: torch.Tensor,
    foreground: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> dict:
    r1_rank = average_percentile_rank(r1)
    r_rank = average_percentile_rank(residual)
    f_rank = average_percentile_rank(foreground)
    median_rank = torch.median(torch.stack((r1_rank, r_rank, f_rank), dim=0), dim=0).values
    f_unique, f_tie = _unique_and_tie_ratio(foreground)
    median_unique, median_tie = _unique_and_tie_ratio(median_rank)
    c0 = candidates["c0_f_minmax_37"]
    c1 = candidates["c1_f_rank_r1_37"]
    c2 = candidates["c2_median_rank_r1_37"]
    r1_area = float((r1 > 0.5).float().mean())

    def range_stats(value: torch.Tensor, prefix: str) -> dict:
        minimum = float(value.min())
        maximum = float(value.max())
        return {f"{prefix}_min": minimum, f"{prefix}_max": maximum, f"{prefix}_range": maximum - minimum}

    c1_area = float((c1 > 0.5).float().mean())
    c2_area = float((c2 > 0.5).float().mean())
    return {
        **range_stats(r1, "r1"),
        **range_stats(residual, "r"),
        **range_stats(foreground, "f"),
        "f_unique_ratio": f_unique,
        "f_tie_ratio": f_tie,
        "median_rank_unique_ratio": median_unique,
        "median_rank_tie_ratio": median_tie,
        "c0_area_gt_05": float((c0 > 0.5).float().mean()),
        "c1_area_gt_05": c1_area,
        "c2_area_gt_05": c2_area,
        "r1_area_gt_05": r1_area,
        "c1_area_delta_vs_r1": c1_area - r1_area,
        "c2_area_delta_vs_r1": c2_area - r1_area,
        "c1_sorted_l1_vs_r1": float(
            torch.mean(torch.abs(torch.sort(c1.reshape(-1)).values - torch.sort(r1.reshape(-1)).values))
        ),
        "c2_sorted_l1_vs_r1": float(
            torch.mean(torch.abs(torch.sort(c2.reshape(-1)).values - torch.sort(r1.reshape(-1)).values))
        ),
        "c1_constant_source_fallback": bool(foreground.max() == foreground.min()),
        "c2_constant_source_fallback": bool(median_rank.max() == median_rank.min()),
    }


def _source_snapshot(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def build_rank_calibration_cache(
    config_path: str | Path,
    dabe_root: str | Path,
    out_root: str | Path,
    split: str = "test",
    max_samples: int = -1,
    overwrite: bool = False,
    overwrite_reason: str = "",
) -> dict:
    config_path = Path(config_path).resolve()
    source_root = Path(dabe_root).resolve()
    output_root = Path(out_root).resolve()
    _validate_roots(source_root, output_root)
    if split != "test":
        raise ValueError(f"This frozen experiment supports split=test only, got {split!r}")
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or a positive integer")

    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError(f"BACKBONE_KEY must be dinov1-s8, got {getattr(cfg, 'BACKBONE_KEY', None)!r}")
    source_manifest = source_root / f"manifest_{split}.jsonl"
    source_rows_all = read_jsonl(source_manifest)
    if not source_rows_all:
        raise RuntimeError(f"Empty source manifest: {source_manifest}")
    source_rows = source_rows_all if max_samples == -1 else source_rows_all[:max_samples]
    if max_samples == -1 and len(source_rows) != 6473:
        raise RuntimeError(f"Full test run must contain exactly 6473 rows, got {len(source_rows)}")

    keys: set[tuple[str, str]] = set()
    source_paths: list[Path] = [source_manifest]
    for index, row in enumerate(source_rows, 1):
        for field in ("dataset", "stem", "cache_path"):
            if field not in row:
                raise KeyError(f"{field} missing from source manifest row {index}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in keys:
            raise RuntimeError(f"Duplicate source key: {key}")
        keys.add(key)
        cache_path = Path(row["cache_path"]).resolve()
        if cache_path != source_root and source_root not in cache_path.parents:
            raise ValueError(f"Source cache escapes dabe_root: {cache_path}")
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        source_paths.append(cache_path)
    before = _source_snapshot(source_paths)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing out_root: {output_root}")
        if not overwrite_reason.strip():
            raise ValueError("--overwrite requires a non-empty --overwrite_reason")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    log_path = output_root / "cache_build.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")

    started = time.time()
    output_rows = []
    try:
        log("rankcal_version = dabe_rankcal_v1")
        log("gt_used_for_generation = false")
        log("dino_forward_used = false")
        log("source_cache_modified = false")
        log("dataset_specific_rule = false")
        log("threshold_search_used = false")
        log(f"source_manifest = {source_manifest}")
        log(f"output_root = {output_root}")
        if overwrite:
            log(f"overwrite = true; reason = {overwrite_reason.strip()}")
        else:
            log("overwrite = false")

        for index, row in enumerate(source_rows, 1):
            dataset = str(row["dataset"])
            stem = str(row["stem"])
            source_path = Path(row["cache_path"]).resolve()
            payload = torch_load(source_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"Source payload must be a dict: {source_path}")
            if payload.get("dataset") != dataset or payload.get("stem") != stem:
                raise RuntimeError(f"Source payload key mismatch: {source_path}")
            if str(payload.get("dabe_version", "")).lower() != "v2":
                raise ValueError(f"dabe_version must be v2: {source_path}")
            if payload.get("augs") != ["identity"]:
                raise ValueError(f"augs must equal ['identity']: {source_path}")
            if payload.get("num_views") != 1:
                raise ValueError(f"num_views must equal 1: {source_path}")

            r1, residual, foreground = (
                _validate_tensor(payload, field, source_path) for field in REQUIRED_SOURCE_FIELDS
            )
            candidates = build_rank_calibration_candidates(r1, residual, foreground)
            for field in CANDIDATE_FIELDS:
                _validate_candidate(candidates[field], field, source_path)
            derived_payload = {
                "dataset": dataset,
                "stem": stem,
                "rankcal_version": RANK_CALIBRATION_VERSION,
                "source_dabe_version": "v2",
                "source_augs": ["identity"],
                "source_num_views": 1,
                "source_cache_path": str(source_path),
                **candidates,
                "diagnostics": _diagnostics(r1, residual, foreground, candidates),
            }
            output_path = output_root / split / dataset / f"{stem}.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
            torch.save(derived_payload, temporary)
            os.replace(temporary, output_path)
            output_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "cache_path": str(output_path.resolve()),
                    "source_cache_path": str(source_path),
                    "rankcal_version": RANK_CALIBRATION_VERSION,
                    "shape": [1, 37, 37],
                }
            )
            if index == len(source_rows) or index % 250 == 0:
                log(f"processed = {index}/{len(source_rows)}")

        output_keys = {(row["dataset"], row["stem"]) for row in output_rows}
        if output_keys != keys or len(output_rows) != len(keys):
            raise RuntimeError("Output manifest keys do not exactly match selected source keys")
        manifest_path = output_root / f"manifest_{split}.jsonl"
        write_jsonl(manifest_path, output_rows)

        after = _source_snapshot(source_paths)
        if after != before:
            changed = [str(path) for path in before if before[path] != after.get(path)]
            raise RuntimeError(f"Source cache changed during generation: {changed[:10]}")
        git_commit, git_status = _git_metadata()
        protocol = {
            "rankcal_version": RANK_CALIBRATION_VERSION,
            "source_dabe_root": str(source_root),
            "source_manifest": str(source_manifest),
            "output_root": str(output_root),
            "split": split,
            "num_samples": len(output_rows),
            "backbone": "DINOv1-S/8",
            "source_augs": ["identity"],
            "gt_used_for_generation": False,
            "dino_forward_used": False,
            "source_cache_modified": False,
            "threshold_search_used": False,
            "dataset_specific_rule": False,
            "external_model_used": False,
            "methods": {
                "C0-F-MinMax": "(F-min(F))/(max(F)-min(F)+1e-8); constant F -> zeros",
                "C1-FRank-R1Dist": "Q_R1(average_percentile_rank(F))",
                "C2-MedianRank-R1Dist": "Q_R1(median(rank(R1), rank(R), rank(F)))",
            },
            "eps": RANK_CALIBRATION_EPS,
            "git_commit": git_commit,
            "git_status_short": git_status,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_manifest_sha256": _sha256(source_manifest),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_rank_calibration.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "overwrite": bool(overwrite),
            "overwrite_reason": overwrite_reason.strip(),
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_cache_modified = false (verified size and mtime)")
        log(f"num_samples = {len(output_rows)}")
        log(f"elapsed_seconds = {time.time() - started:.3f}")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_rank_calibration_cache(
        config_path=args.config,
        dabe_root=args.dabe_root,
        out_root=args.out_root,
        split=args.split,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        overwrite_reason=args.overwrite_reason,
    )
