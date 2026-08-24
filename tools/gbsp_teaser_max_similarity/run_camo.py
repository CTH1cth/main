#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from tools.gbsp_teaser_max_similarity.analyze import analyze
from tools.gbsp_teaser_max_similarity.collect import collect
from tools.gbsp_teaser_max_similarity.report import build_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_collect", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if not args.skip_collect:
        collect(Namespace(
            config=args.config, out_dir=args.out_dir, max_samples=args.max_samples,
            device=args.device, failure_policy=args.failure_policy, overwrite=args.overwrite,
        ))
    result = analyze(args.config, args.out_dir)
    case = build_outputs(args.config, args.out_dir, result)
    return {"status": "complete", "case": case, "images": result["images"],
            "formal_camo_complete": result["formal_camo_complete"],
            "out_dir": str(Path(args.out_dir).resolve())}


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
