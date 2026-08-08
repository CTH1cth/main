#!/usr/bin/env python3
"""Build GT-free 68x68 hard/soft pseudo labels from a selected rescue score."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gbsp_knn_lsr_common import (  # noqa: E402
    adaptive_threshold, load_manifest, load_torch, minmax, score_from_payload,
    score_path, write_json, write_jsonl,
)
from tools.reconstruction_rescue_common import atomic_torch_save, require_output_outside_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=.58)
    parser.add_argument("--adaptive", choices=("none", "otsu", "multi_otsu_3"), default="none")
    parser.add_argument("--out_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); output = require_output_outside_main(args.out_root); output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.score_root, score=True, split=args.split, max_samples=args.max_samples)
    manifest = []
    for index, row in enumerate(rows, 1):
        source = load_torch(score_path(row)); score37 = minmax(score_from_payload(source, args.method)).reshape(1, 1, 37, 37)
        soft68 = F.interpolate(score37, size=(68, 68), mode="bilinear", align_corners=False)[0]
        if args.adaptive == "none":
            threshold, fallback = float(args.threshold), False
        else:
            threshold, fallback = adaptive_threshold(score37, args.adaptive)
        hard68 = (soft68 >= threshold).float()
        target = output / args.split / row["dataset"] / f"{row['stem']}.pt"
        atomic_torch_save({
            "version": "reconstruction_rescue_pseudo_v1", "dataset": row["dataset"], "stem": row["stem"],
            "image_path": row["image_path"], "gt_path": row.get("gt_path"),
            "source_score_path": str(score_path(row)), "method": args.method,
            "score_37": score37[0], "soft_pseudo_68": soft68, "hard_pseudo_68": hard68,
            "threshold": threshold, "threshold_protocol": args.adaptive if args.adaptive != "none" else "fixed",
            "threshold_fallback": fallback, "gt_used_for_generation": False,
        }, target)
        manifest.append({"dataset": row["dataset"], "stem": row["stem"], "cache_path": str(target),
                         "image_path": row["image_path"], "gt_path": row.get("gt_path")})
        if index % 100 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] pseudo", flush=True)
    write_jsonl(output / f"manifest_{args.split}.jsonl", manifest)
    summary = {"images": len(manifest), "method": args.method, "threshold": args.threshold,
               "adaptive": args.adaptive, "gt_used_for_generation": False}
    write_json(output / "pseudo_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
