#!/usr/bin/env python3
"""Create the deterministic stratified sample list for the resolution probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import build_image_items, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("TE-CAMO", "TE-COD10K", "NC4K"),
    )
    args = parser.parse_args()
    if args.num_samples == 0 or args.num_samples < -1:
        raise ValueError("num-samples must be positive, or -1 for the complete split")
    cfg = load_config(args.config)
    requested = list(args.datasets)
    items = build_image_items(cfg.DATA_ROOT, requested, require_gt=True)
    by_dataset = {dataset: [] for dataset in requested}
    for item in items:
        by_dataset[item["dataset"]].append(item)
    missing = [dataset for dataset, rows in by_dataset.items() if not rows]
    if missing:
        raise RuntimeError(f"datasets have no samples: {missing}")

    rng = random.Random(int(args.seed))
    selected = []
    if args.num_samples == -1:
        for dataset in requested:
            selected.extend(sorted(by_dataset[dataset], key=lambda item: item["stem"]))
    else:
        base, remainder = divmod(int(args.num_samples), len(requested))
        for index, dataset in enumerate(requested):
            count = base + (1 if index < remainder else 0)
            rows = sorted(by_dataset[dataset], key=lambda item: item["stem"])
            if count > len(rows):
                raise RuntimeError(f"{dataset}: requested {count}, available {len(rows)}")
            chosen = rng.sample(rows, count)
            chosen.sort(key=lambda item: item["stem"])
            selected.extend(chosen)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "dataset": item["dataset"],
                "stem": item["stem"],
                "image_path": item["image_path"],
                "gt_path": item["gt_path"],
            },
            ensure_ascii=False,
        )
        for item in selected
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(output),
                "num_samples": len(selected),
                "seed": int(args.seed),
                "counts": {
                    dataset: sum(item["dataset"] == dataset for item in selected)
                    for dataset in requested
                },
                "sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
