#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_analysis.common import (  # noqa: E402
    camo_rows, labels, load_gt_occupancy, load_settings, load_torch,
    normalized_feature, validate_output, write_json, write_jsonl,
)
from tools.gbsp_teaser_analysis.compute_gbsp_residual_hist import summarize_residual  # noqa: E402
from tools.gbsp_teaser_analysis.compute_pairwise_similarity_hist import summarize_pairwise  # noqa: E402

PROTOCOLS = ("allpatch_0.5", "core_0.2_0.8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    generated, invalid, failures, audits = [], [], [], []
    for number, row in enumerate(rows, 1):
        target = out / "CAMO" / "per_image" / f"{row['stem']}.npz"
        try:
            core = load_torch(row["cache_path"])
            if core["settings"].get("background_source") != "fullbc":
                raise RuntimeError("formal core is not Full-BC")
            result = core["results"][f"r{settings.gbsp_rank}"]
            if int(result["selected_rank"]) != settings.gbsp_rank:
                raise RuntimeError("formal fixed-rank result is not rank 8")
            raw = torch.as_tensor(result["absolute_raw"]).float().reshape(-1)
            residual = torch.as_tensor(result["absolute_minmax"]).float().reshape(-1)
            reproduced = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-12)
            minmax_error = float((residual - reproduced).abs().max())
            if minmax_error > 1e-6:
                raise RuntimeError(f"formal GBSP Min-Max mismatch: {minmax_error}")
            # Both diagnostics are fully generated before GT is loaded.
            feature = normalized_feature(core, device)
            similarity = feature @ feature.T
            if not bool(torch.isfinite(similarity).all()):
                raise RuntimeError("pairwise similarity contains non-finite values")
            occupancy = load_gt_occupancy(row["gt_path"])
            payload: dict[str, np.ndarray] = {
                "image_id": np.asarray(f"CAMO/{row['stem']}"),
                "stem": np.asarray(row["stem"]),
                "gbsp_candidate_count": np.asarray(len(result["background_indices"]), dtype=np.int16),
                "gbsp_minmax_max_abs_error": np.asarray(minmax_error, dtype=np.float32),
            }
            row_invalid = []
            per_protocol_audit = {}
            for protocol in PROTOCOLS:
                label, valid = labels(occupancy, settings, protocol)
                try:
                    pair = summarize_pairwise(
                        similarity, label, valid, settings.similarity_bins,
                        settings.descriptive_thresholds,
                    )
                    res = summarize_residual(residual.numpy(), label, valid, settings.residual_bins)
                    suffix = "main" if protocol == "allpatch_0.5" else "core"
                    for key, value in {**pair, **{f"gbsp_{k}": v for k, v in res.items()}}.items():
                        payload[f"{key}_{suffix}"] = np.asarray(value)
                    per_protocol_audit[protocol] = {
                        "valid": True,
                        "num_fg_patches": res["num_fg_patches"],
                        "num_bg_patches": res["num_bg_patches"],
                        "num_bb_pairs": pair["num_bb_pairs"],
                        "num_fb_pairs": pair["num_fb_pairs"],
                    }
                except ValueError as error:
                    row_invalid.append({"dataset": "CAMO", "stem": row["stem"], "protocol": protocol, "reason": str(error)})
                    per_protocol_audit[protocol] = {"valid": False, "reason": str(error)}
            if not any(item.get("valid") for item in per_protocol_audit.values()):
                raise RuntimeError("image is invalid under both GT grouping protocols")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **payload)
            generated.append({
                "dataset": "CAMO", "stem": row["stem"], "cache_path": str(target),
                "image_path": row["image_path"], "gt_path": row["gt_path"],
                "source_core_path": row["cache_path"], "source_feature_path": core["source_feature_path"],
            })
            invalid.extend(row_invalid)
            audits.append({
                "dataset": "CAMO", "stem": row["stem"],
                "candidate_source": "formal_fullbc_cache", "selected_rank": int(result["selected_rank"]),
                "candidate_count": len(result["background_indices"]),
                "gbsp_minmax_max_abs_error": minmax_error,
                "similarity_uses_background_dictionary": False,
                "gt_used_for_score_generation": False,
                "protocols": per_protocol_audit,
            })
        except Exception as error:
            failures.append({"dataset": "CAMO", "stem": row.get("stem"), "error": repr(error)})
            if args.failure_policy == "strict":
                raise
        if number % 10 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] generated={len(generated)} failed={len(failures)}", flush=True)
    write_jsonl(out / "CAMO" / "manifest.jsonl", generated)
    write_jsonl(out / "diagnostics" / "per_image_audit.jsonl", audits)
    write_jsonl(out / "diagnostics" / "invalid_images.jsonl", invalid)
    summary = {
        "requested": len(rows), "generated": len(generated), "failed": len(failures),
        "invalid_protocol_rows": len(invalid), "failures": failures,
        "dataset_counts": dict(Counter(row["dataset"] for row in generated)),
        "formal_camo_complete": len(rows) == 250 and len(generated) == 250 and not failures,
        "feature": "DINO-v1 ViT-S/8 formal 37x37 key features, per-patch L2 normalized",
        "similarity": "all BG-BG upper-triangle and all FG-BG pairwise cosine; no dictionary/KNN/aggregation",
        "residual": "frozen Full-BC fixed-r8 affine PCA absolute squared residual, formal per-image Min-Max",
        "gt_used_for_score_generation": False,
    }
    write_json(out / "diagnostics" / "collection_summary.json", summary)
    return summary


def main() -> None:
    summary = collect(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
