#!/usr/bin/env python3
"""Merge independently evaluated MBSP ablation tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", "--eval_dir", dest="eval_dirs", action="append", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_directory in args.eval_dirs:
        directory = Path(raw_directory).expanduser().resolve()
        for row in _read(directory / "ablation_summary.csv"):
            rows.append({"source_eval_dir": str(directory), **row})
    if not rows:
        raise RuntimeError("no ablation rows found")

    identity_fields = (
        "variant",
        "score_type",
        "num_subspaces",
        "pca_energy",
        "pca_max_rank",
        "pca_min_rank",
        "min_cluster_size",
    )
    deduplicated = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in identity_fields)
        deduplicated[key] = row
    merged = sorted(
        deduplicated.values(),
        key=lambda row: tuple(row.get(field, "") for field in identity_fields),
    )
    fields = list(merged[0])
    path = output_dir / "ablation_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"wrote {len(merged)} rows to {path}")


if __name__ == "__main__":
    main()
