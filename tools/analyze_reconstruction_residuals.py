#!/usr/bin/env python3
"""Residual/error diagnostics and optional GT-free local-scale score variants."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reconstruction import flatten_feature, normalize_local_residual  # noqa: E402
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    correlation_metrics, load_manifest, load_patch_area, load_torch, patch_labels,
    rank_metrics, score_from_payload, score_path, write_csv, write_json, write_jsonl,
)
from tools.reconstruction_rescue_common import (  # noqa: E402
    atomic_torch_save, feature_tensor, require_output_outside_main,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--background_root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--normalizations", nargs="+", choices=("raw", "centroid", "pairwise"),
                        default=("raw", "centroid", "pairwise"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_root", required=True)
    return parser.parse_args()


def method_geometry(method: str) -> tuple[str, int]:
    if not method.startswith(("lcbr_", "lsr_", "lar_")):
        raise ValueError(f"local scale is only defined for a local reconstruction method: {method}")
    geometry = "raw" if "_raw_" in method else "l2"
    try:
        k = int(next(piece[1:] for piece in method.split("_") if piece.startswith("k")))
    except (StopIteration, ValueError) as error:
        raise ValueError(f"cannot parse K from {method}") from error
    return geometry, k


def main() -> None:
    args = parse_args(); out = require_output_outside_main(args.out_root); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows = load_manifest(args.score_root, score=True, split=args.split, max_samples=args.max_samples)
    manifest, per_image, quantile_accumulator, failures = [], [], defaultdict(list), []
    for index, row in enumerate(rows, 1):
        try:
            payload = load_torch(score_path(row))
            feature_path = Path(payload["source_feature_path"])
            raw = feature_tensor(load_torch(feature_path), feature_path).to(device)
            raw_flat = flatten_feature(raw, normalize=False)
            l2_flat = flatten_feature(raw, normalize=True)
            neighbors = torch.as_tensor(payload["neighbor_indices"], dtype=torch.long, device=device)
            output_scores = dict(payload["scores"])
            area = load_patch_area(row["gt_path"])
            labels, valid = patch_labels(area, "main_0.5")
            labels_valid = labels[valid]
            output_diagnostics = {}
            for method in args.methods:
                geometry, k = method_geometry(method)
                feature = raw_flat if geometry == "raw" else l2_flat
                index_k = neighbors[:, :k]
                atoms = feature.index_select(0, index_k.reshape(-1)).reshape(1369, k, feature.shape[1])
                residual = score_from_payload(payload, method).to(device)
                scaled = normalize_local_residual(residual, atoms)
                variants = {
                    "raw": scaled.raw_residual,
                    "centroid": scaled.centroid_normalized,
                    "pairwise": scaled.pairwise_normalized,
                }
                for normalization in args.normalizations:
                    name = method if normalization == "raw" else f"{method}__{normalization}_scale"
                    output_scores[name] = variants[normalization].reshape(1, 37, 37).cpu()
                    metric = rank_metrics(variants[normalization].detach().cpu().numpy()[valid], labels_valid)
                    per_image.append({"dataset": row["dataset"], "stem": row["stem"], "method": method,
                                      "normalization": normalization, **metric})
                correlations_centroid = correlation_metrics(
                    residual.detach().cpu().numpy(), scaled.centroid_scale.detach().cpu().numpy())
                correlations_pairwise = correlation_metrics(
                    residual.detach().cpu().numpy(), scaled.pairwise_scale.detach().cpu().numpy())
                output_diagnostics[method] = {
                    "centroid_scale": scaled.centroid_scale.cpu().half(),
                    "pairwise_scale": scaled.pairwise_scale.cpu().half(),
                    "centroid_floor": scaled.centroid_floor.cpu().half(),
                    "pairwise_floor": scaled.pairwise_floor.cpu().half(),
                    "residual_centroid_correlation": correlations_centroid,
                    "residual_pairwise_correlation": correlations_pairwise,
                }
                score_np = residual.detach().cpu().numpy()
                rank = np.argsort(np.argsort(score_np, kind="mergesort"), kind="mergesort")
                bins = np.minimum(rank * 10 // len(rank), 9)
                for bin_index in range(10):
                    selected = valid & (bins == bin_index)
                    if selected.any():
                        quantile_accumulator[(method, bin_index)].append(float(labels[selected].mean()))
                existing = payload.get("diagnostics", {}).get(method, {})
                for field in ("effective_active_atoms", "largest_alpha", "top2_alpha_sum", "top4_alpha_sum", "alpha_entropy"):
                    value = existing.get(field)
                    if torch.is_tensor(value) and value.numel() == 1369:
                        array = value.float().numpy()
                        for class_name, class_mask in (("foreground", valid & (labels == 1)), ("background", valid & (labels == 0))):
                            if class_mask.any():
                                per_image.append({"dataset": row["dataset"], "stem": row["stem"],
                                                  "method": method, "normalization": "alpha_diagnostic",
                                                  "alpha_field": field, "class": class_name,
                                                  "value": float(array[class_mask].mean())})
            target = out / "scores" / args.split / row["dataset"] / f"{row['stem']}.pt"
            new_payload = dict(payload)
            new_payload["scores"] = output_scores
            new_payload["scale_diagnostics"] = output_diagnostics
            new_payload["scale_normalization_gt_used"] = False
            atomic_torch_save(new_payload, target)
            manifest.append({"dataset": row["dataset"], "stem": row["stem"],
                             "image_path": row["image_path"], "gt_path": row["gt_path"],
                             "score_path": str(target), "source_score_path": str(score_path(row))})
        except Exception as error:
            failures.append({"dataset": row.get("dataset"), "stem": row.get("stem"), "error": repr(error)})
        if index % 20 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] valid={len(manifest)} failed={len(failures)}", flush=True)
    write_jsonl(out / f"score_manifest_{args.split}.jsonl", manifest)
    write_csv(out / "per_image_residual_diagnostics.csv", per_image)
    quantile_rows = [
        {"method": method, "score_decile": bin_index, "mean_foreground_fraction": float(np.mean(values)),
         "num_images": len(values)}
        for (method, bin_index), values in sorted(quantile_accumulator.items())
    ]
    write_csv(out / "residual_quantile_calibration.csv", quantile_rows)
    summary = {"requested": len(rows), "generated": len(manifest), "failures": failures,
               "methods": args.methods, "normalizations": args.normalizations,
               "gt_used_for_scale_generation": False, "gt_used_for_diagnostics_only": True}
    write_json(out / "residual_analysis_summary.json", summary)
    if failures:
        raise RuntimeError(f"residual analysis failed for {len(failures)} samples")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
