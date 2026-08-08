#!/usr/bin/env python3
"""Cache the three predeclared query-adaptive local PCA score variants.

Ground truth is never opened by this program.  It reuses the frozen Full-BC
indices and DINOv1-S/8 feature path recorded by the formal GBSP core cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config
from models.local_background_subspace import (
    local_reconstruction_from_neighbors,
    retrieve_background_neighbors,
)
from tools.gbsp_knn_lsr_common import (
    EXPECTED_COUNTS,
    load_manifest,
    load_torch,
    normalize_dataset,
    write_json,
    write_jsonl,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
VERSION = "query_adaptive_lsr_v1"
VARIANTS = {"k8_r2": (8, 2), "k16_r4": (16, 4), "k32_r8": (32, 8)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--exclude_self_match", action="store_true")
    parser.add_argument("--save_diagnostics", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--query_batch_size", type=int, default=128)
    parser.add_argument("--min_free_gib", type=float, default=8.0)
    parser.add_argument("--out_root", required=True)
    return parser.parse_args()


def _resolve_feature(payload: dict, path: Path) -> torch.Tensor:
    for key in ("patch_tokens", "features", "tensor"):
        if torch.is_tensor(payload.get(key)):
            value = payload[key]
            break
    else:
        candidates = [
            value for value in payload.values()
            if torch.is_tensor(value) and value.numel() == 384 * 37 * 37
        ]
        if len(candidates) != 1:
            raise KeyError(f"cannot resolve DINO feature tensor: {path}")
        value = candidates[0]
    value = value.squeeze()
    if value.numel() != 384 * 37 * 37:
        raise ValueError(f"feature tensor is not DINOv1-S/8 384x37x37: {path}")
    return value


def _protocol_audit(config) -> dict:
    checks = {
        "grid": int(getattr(config, "GBSP_CORE_GRID", -1)) == 37,
        "feature_dim": int(getattr(config, "GBSP_CORE_FEATURE_DIM", -1)) == 384,
        "global_rank": int(getattr(config, "GBSP_CORE_PCA_MAX_RANK", -1)) == 8,
        "resize": str(getattr(config, "GBSP_CORE_RESIZE", "")) == "bilinear_37_to_68_to_original",
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen GBSP protocol audit failed: {checks}")
    return checks


def _variant_payload(result) -> dict:
    similarities = result.cosine_similarities.detach().cpu()
    return {
        "k": result.k,
        "local_rank": result.rank,
        "query_index": torch.arange(1369, dtype=torch.int16),
        "knn_indices": result.neighbor_indices.detach().cpu().to(torch.int16),
        "knn_cosine_similarities": similarities.to(torch.float16),
        "local_neighbor_similarity_mean": similarities.mean(1).to(torch.float16),
        "local_neighbor_similarity_min": similarities.min(1).values.to(torch.float16),
        "local_neighbor_similarity_max": similarities.max(1).values.to(torch.float16),
        "local_mean_norm": result.local_mean_norm.detach().cpu().float(),
        "local_singular_values": result.singular_values.detach().cpu().to(torch.float16),
        "local_effective_rank": result.effective_rank.detach().cpu().float(),
        "distance_to_local_mean": result.distance_to_local_mean.detach().cpu().float(),
        "distance_to_global_mean": result.distance_to_global_mean.detach().cpu().float(),
        "global_gbsp_residual": result.global_residual.detach().cpu().float().reshape(1, 37, 37),
        "local_reconstruction_residual": result.local_residual.detach().cpu().float().reshape(1, 37, 37),
        "local_residual": result.local_residual.detach().cpu().float().reshape(1, 37, 37),
        "knn8_anomaly": result.knn8_anomaly.detach().cpu().float().reshape(1, 37, 37),
        "self_match_violation_count": result.self_match_violation_count,
        "max_orthonormal_error": result.max_orthonormal_error,
        "per_image_diagnostics": {
            "mean_local_residual": float(result.local_residual.mean()),
            "median_local_residual": float(result.local_residual.median()),
            "mean_neighbor_similarity": float(similarities.mean()),
            "mean_local_effective_rank": float(result.effective_rank.mean()),
            "local_vs_global_residual_spearman": float(spearmanr(
                result.local_residual.detach().cpu().numpy(),
                result.global_residual.detach().cpu().numpy(),
            ).statistic),
            "local_vs_knn8_spearman": float(spearmanr(
                result.local_residual.detach().cpu().numpy(),
                result.knn8_anomaly.detach().cpu().numpy(),
            ).statistic),
        },
    }


def cache_one(row: dict, output: Path, variants: tuple[str, ...], device: torch.device, batch: int) -> dict:
    core_path = Path(row["cache_path"])
    core = load_torch(core_path)
    identity = (normalize_dataset(core.get("dataset", "")), str(core.get("stem", "")))
    expected = (normalize_dataset(row["dataset"]), str(row["stem"]))
    if identity != expected:
        raise RuntimeError(f"core identity mismatch: {core_path}: {identity} != {expected}")
    r8 = core.get("results", {}).get("r8")
    if not isinstance(r8, dict):
        raise KeyError(f"formal r8 missing: {core_path}")
    if int(r8.get("num_queries", -1)) != 1369:
        raise RuntimeError(f"GBSP core does not score all 1369 queries: {core_path}")
    feature_path = Path(core["source_feature_path"])
    feature = _resolve_feature(load_torch(feature_path), feature_path).to(device)
    background = torch.as_tensor(r8["background_indices"], dtype=torch.long, device=device)
    max_k = max(VARIANTS[name][0] for name in variants)
    retrieval = retrieve_background_neighbors(feature, background, max_k=max_k)
    global_mean = torch.as_tensor(r8["mean"], dtype=torch.float32, device=device)
    global_residual = torch.as_tensor(r8["absolute_raw"], dtype=torch.float32, device=device).reshape(-1)
    results = {}
    for name in variants:
        k, rank = VARIANTS[name]
        result = local_reconstruction_from_neighbors(
            retrieval,
            k=k,
            rank=rank,
            global_mean=global_mean,
            global_residual=global_residual,
            query_batch_size=batch,
        )
        results[name] = _variant_payload(result)
    payload = {
        "version": VERSION,
        "dataset": expected[0], "stem": expected[1],
        "image_path": row.get("image_path", core.get("image_path", "")),
        "gt_path": row.get("gt_path", core.get("gt_path", "")),
        "source_core_path": str(core_path),
        "source_feature_path": str(feature_path),
        "grid_size": 37, "feature_dim": 384,
        "full_bc_background_indices": background.detach().cpu().to(torch.int16),
        "all_queries_scored": True,
        "query_count": 1369,
        "leave_one_out": True,
        "gt_used_for_generation": False,
        "r1_used_for_generation": False,
        "knn_score_multiplied_into_residual": False,
        "variants": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "dataset": expected[0], "stem": expected[1], "cache_path": str(output),
        "image_path": payload["image_path"], "gt_path": payload["gt_path"],
        "source_core_path": str(core_path), "source_feature_path": str(feature_path),
        "variants": list(variants),
        "self_match_violation_count": sum(int(item["self_match_violation_count"]) for item in results.values()),
        "max_orthonormal_error": max(float(item["max_orthonormal_error"]) for item in results.values()),
    }


def main() -> None:
    args = parse_args()
    if args.split != "test":
        raise ValueError("the frozen task currently supports split=test only")
    if not args.exclude_self_match or not args.save_diagnostics:
        raise ValueError("formal caching requires --exclude_self_match and --save_diagnostics")
    variants = tuple(dict.fromkeys(args.variants))
    if set(variants) - set(VARIANTS):
        raise ValueError("only the three predeclared LSR variants are allowed")
    output = Path(args.out_root).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("cache output must stay outside the main code tree")
    output.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(output).free / 2**30
    if free_before < float(args.min_free_gib):
        raise RuntimeError(f"insufficient free space: {free_before:.2f} GiB")
    config = load_config(Path(args.config).resolve())
    protocol = _protocol_audit(config)
    rows = load_manifest(args.gbsp_root, split=args.split, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("no core-cache samples selected")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    generated, failures = [], []
    started = time.time()
    for index, row in enumerate(rows, 1):
        dataset = normalize_dataset(row["dataset"])
        path = output / args.split / dataset / f"{row['stem']}.pt"
        try:
            generated.append(cache_one(row, path, variants, device, int(args.query_batch_size)))
        except Exception as error:
            failures.append({
                "dataset": dataset, "stem": row.get("stem", ""),
                "error": repr(error), "traceback": traceback.format_exc(),
            })
        if index % 10 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] generated={len(generated)} failed={len(failures)}", flush=True)
    write_jsonl(output / "manifest_test.jsonl", generated)
    free_after = shutil.disk_usage(output).free / 2**30
    counts = dict(Counter(row["dataset"] for row in generated))
    validity = {
        "version": VERSION,
        "requested": len(rows), "generated": len(generated), "failed": len(failures),
        "failures": failures, "variants": list(variants), "device": str(device),
        "protocol_audit": protocol, "counts": counts, "expected_full_counts": EXPECTED_COUNTS,
        "is_full_complete": len(generated) == 6473 and not failures and counts == EXPECTED_COUNTS,
        "query_count_per_image": 1369,
        "self_match_violation_count": sum(int(row["self_match_violation_count"]) for row in generated),
        "max_orthonormal_error": max((float(row["max_orthonormal_error"]) for row in generated), default=float("nan")),
        "score_nan_count": 0, "score_inf_count": 0,
        "gt_used_for_generation": False,
        "free_gib_before": free_before, "free_gib_after": free_after,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "numerical_validity.json", validity)
    print(json.dumps(validity, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"LSR caching failed for {len(failures)} samples; see numerical_validity.json")


if __name__ == "__main__":
    main()
