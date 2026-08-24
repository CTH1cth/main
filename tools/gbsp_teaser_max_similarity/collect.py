#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_analysis.common import (  # noqa: E402
    GRID, camo_rows, labels, load_gt_occupancy, load_settings, load_torch,
    normalized_feature, validate_output, write_json, write_jsonl,
)
from tools.gbsp_teaser_max_similarity.compute import (  # noqa: E402
    brute_force_query, max_valid_background_similarity,
)

PROTOCOLS = (("allpatch_0.5", "main"), ("core_0.2_0.8", "core"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_formal_inputs(core_path: str | Path, rank: int, device: torch.device) -> dict:
    """Load the frozen feature and GBSP scores without accepting any GT input."""
    core = load_torch(core_path)
    if core["settings"].get("background_source") != "fullbc":
        raise RuntimeError("formal core is not Full-BC")
    result = core["results"][f"r{rank}"]
    if int(result["selected_rank"]) != rank:
        raise RuntimeError(f"formal selected rank is not r{rank}")
    raw = torch.as_tensor(result["absolute_raw"]).float().reshape(-1)
    normalized = torch.as_tensor(result["absolute_minmax"]).float().reshape(-1)
    reproduced = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-12)
    error = float((reproduced - normalized).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"formal GBSP Min-Max mismatch: {error}")
    return {
        "feature": normalized_feature(core, device),
        "raw_residual": raw.numpy().astype(np.float32, copy=False),
        "normalized_residual": normalized.numpy().astype(np.float32, copy=False),
        "background_count": len(result["background_indices"]),
        "minmax_error": error,
        "source_feature_path": core["source_feature_path"],
    }


def _runtime_checks(
    feature: torch.Tensor,
    label: np.ndarray,
    valid: np.ndarray,
    score: np.ndarray,
    matched: np.ndarray,
    rng: np.random.Generator,
    *,
    self_samples: int,
    brute_samples: int,
) -> dict:
    background = np.flatnonzero(valid & (label == 0))
    bg_queries = background
    chosen_bg = rng.choice(bg_queries, size=min(self_samples, bg_queries.size), replace=False)
    self_violations = int(np.count_nonzero(matched[chosen_bg] == chosen_bg))
    if self_violations:
        raise AssertionError("sampled BG self-match exclusion failed")

    valid_query = np.flatnonzero(valid)
    chosen_query = rng.choice(valid_query, size=min(brute_samples, valid_query.size), replace=False)
    max_score_error = 0.0; max_index_mismatch = 0
    for query_index in chosen_query:
        reference_score, reference_index = brute_force_query(feature, background, int(query_index))
        max_score_error = max(max_score_error, abs(float(score[query_index]) - reference_score))
        # Exact ties may legitimately select another index; only flag an index
        # mismatch if the selected score itself is not tied to the maximum.
        if int(matched[query_index]) != reference_index and abs(float(score[query_index]) - reference_score) > 1e-6:
            max_index_mismatch += 1
    if max_score_error >= 1e-6 or max_index_mismatch:
        raise AssertionError(f"maximum audit failed: error={max_score_error}, mismatch={max_index_mismatch}")

    pair_count = min(8, valid_query.size)
    first = rng.choice(valid_query, size=pair_count, replace=False)
    second = rng.choice(valid_query, size=pair_count, replace=False)
    cosine_error = 0.0
    for i, j in zip(first, second):
        dot = float(torch.dot(feature[int(i)], feature[int(j)]).item())
        cosine = float(torch.nn.functional.cosine_similarity(
            feature[int(i)].unsqueeze(0), feature[int(j)].unsqueeze(0), dim=1, eps=1e-12
        ).item())
        cosine_error = max(cosine_error, abs(dot - cosine))
    if cosine_error >= 1e-6:
        raise AssertionError(f"cosine audit failed: {cosine_error}")
    return {
        "self_checks": int(chosen_bg.size), "self_violations": self_violations,
        "brute_force_checks": int(chosen_query.size), "max_score_abs_error": max_score_error,
        "max_index_mismatch": max_index_mismatch,
        "cosine_checks": pair_count, "cosine_max_abs_error": cosine_error,
    }


def collect(args: argparse.Namespace) -> dict:
    settings = load_settings(args.config)
    out = validate_output(args.out_dir)
    rows = camo_rows(settings)
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    rng = np.random.default_rng(settings.random_seed)
    generated, failures, manifest, audits = 0, [], [], []
    aggregate_audit = {
        "self_checks": 0, "self_violations": 0, "brute_force_checks": 0,
        "max_score_abs_error": 0.0, "max_index_mismatch": 0,
        "cosine_checks": 0, "cosine_max_abs_error": 0.0,
    }
    for number, row in enumerate(rows, 1):
        target = out / "CAMO" / "per_image" / f"{row['stem']}.npz"
        if target.is_file() and not args.overwrite:
            manifest.append({
                "dataset": "CAMO", "stem": row["stem"], "cache_path": str(target),
                "image_path": row["image_path"], "gt_path": row["gt_path"],
                "source_core_path": row["cache_path"],
            })
            generated += 1
            continue
        try:
            # GBSP and DINO scores are loaded before the diagnostic GT is accessed.
            formal = load_formal_inputs(row["cache_path"], settings.gbsp_rank, device)
            feature = formal["feature"]
            occupancy = load_gt_occupancy(row["gt_path"])
            payload: dict[str, np.ndarray] = {
                "image_id": np.asarray(f"CAMO/{row['stem']}"), "stem": np.asarray(row["stem"]),
                "gt_occupancy": occupancy.astype(np.float32),
                "gbsp_raw_residual": formal["raw_residual"],
                "gbsp_normalized_residual": formal["normalized_residual"],
                "gbsp_candidate_count": np.asarray(formal["background_count"], dtype=np.int16),
                "gbsp_minmax_max_abs_error": np.asarray(formal["minmax_error"], dtype=np.float32),
            }
            per_image_audit = {}
            for protocol, suffix in PROTOCOLS:
                label, valid = labels(occupancy, settings, protocol)
                result = max_valid_background_similarity(feature, label, valid)
                payload[f"gt_label_{suffix}"] = label.astype(np.uint8)
                payload[f"valid_{suffix}"] = valid.astype(np.uint8)
                payload[f"max_bg_similarity_{suffix}"] = result.score
                payload[f"similarity_fg_score_{suffix}"] = (1.0 - result.score).astype(np.float32)
                payload[f"matched_bg_patch_index_{suffix}"] = result.matched_index
                payload[f"matched_bg_patch_y_{suffix}"] = np.where(
                    result.matched_index >= 0, result.matched_index // GRID, -1
                ).astype(np.int16)
                payload[f"matched_bg_patch_x_{suffix}"] = np.where(
                    result.matched_index >= 0, result.matched_index % GRID, -1
                ).astype(np.int16)
                checks = _runtime_checks(
                    feature, label, valid, result.score, result.matched_index, rng,
                    self_samples=20 if protocol == "allpatch_0.5" and number <= 100 else 0,
                    brute_samples=4 if protocol == "allpatch_0.5" else 0,
                )
                per_image_audit[protocol] = {
                    "num_background_references": result.num_background,
                    "num_valid_queries": int(valid.sum()), **checks,
                }
                for key, value in checks.items():
                    if key.endswith("error"):
                        aggregate_audit[key] = max(float(aggregate_audit[key]), float(value))
                    else:
                        aggregate_audit[key] += int(value)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **payload)
            manifest.append({
                "dataset": "CAMO", "stem": row["stem"], "cache_path": str(target),
                "image_path": row["image_path"], "gt_path": row["gt_path"],
                "source_core_path": row["cache_path"],
                "source_feature_path": formal["source_feature_path"],
            })
            audits.append({"dataset": "CAMO", "stem": row["stem"], "protocols": per_image_audit})
            generated += 1
        except Exception as error:
            failures.append({"dataset": "CAMO", "stem": row.get("stem"), "error": repr(error)})
            if args.failure_policy == "strict":
                raise
        if number % 10 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] generated={generated} failed={len(failures)}", flush=True)
    write_jsonl(out / "CAMO" / "manifest.jsonl", manifest)
    write_jsonl(out / "audit" / "per_image_runtime_audit.jsonl", audits)
    summary = {
        "requested": len(rows), "generated": generated, "failed": len(failures), "failures": failures,
        "formal_camo_complete": len(rows) == 250 and generated == 250 and not failures,
        "device": str(device), "runtime_correctness": aggregate_audit,
        "feature": "reused formal cached DINO-v1 ViT-S/8 37x37 key features; per-patch L2 normalized",
        "gbsp": "reused frozen Full-BC fixed-r8 raw and official per-image Min-Max residual",
        "diagnostic_gt_only": True, "gt_used_for_gbsp_inference": False,
        "formal_loader_parameters": list(inspect.signature(load_formal_inputs).parameters),
    }
    write_json(out / "audit" / "collection_summary.json", summary)
    return summary


def main() -> None:
    result = collect(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
