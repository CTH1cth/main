import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (
    ccr_manifest_path,
    ensure_dir,
    load_config,
    read_jsonl,
)


CSV_FIELDS = [
    "dataset",
    "stem",
    "quality",
    "fixed_area",
    "despl_area",
    "corr_area",
    "iou_fixed_despl",
    "raw_expand_area",
    "raw_shrink_area",
    "trusted_expand_area",
    "trusted_shrink_area",
    "anchor_ratio",
    "num_cc_corr",
]


def as_float(row, key):
    return float(row.get(key, 0.0))


def mean(rows, key):
    if not rows:
        return 0.0
    return float(np.mean([as_float(row, key) for row in rows]))


def percentage_lt(rows, key, threshold):
    if not rows:
        return 0.0
    values = np.asarray([as_float(row, key) for row in rows], dtype=np.float64)
    return float((values < threshold).mean())


def write_csv(path, rows):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def write_summary(path, rows):
    ensure_dir(Path(path).parent)
    qualities = [int(row.get("quality", 0)) for row in rows]
    q0 = sum(quality == 0 for quality in qualities)
    q1 = sum(quality == 1 for quality in qualities)
    q2 = sum(quality == 2 for quality in qualities)
    lines = [
        f"total samples: {len(rows)}",
        f"q0 count: {q0}",
        f"q1 count: {q1}",
        f"q2 count: {q2}",
        f"mean fixed_area: {mean(rows, 'fixed_area'):.6f}",
        f"mean despl_area: {mean(rows, 'despl_area'):.6f}",
        f"mean corr_area: {mean(rows, 'corr_area'):.6f}",
        f"mean iou_fixed_despl: {mean(rows, 'iou_fixed_despl'):.6f}",
        f"mean trusted_expand_area: {mean(rows, 'trusted_expand_area'):.6f}",
        f"mean trusted_shrink_area: {mean(rows, 'trusted_shrink_area'):.6f}",
        f"mean anchor_ratio: {mean(rows, 'anchor_ratio'):.6f}",
        (
            "percentage trusted_expand_area < 0.01: "
            f"{percentage_lt(rows, 'trusted_expand_area', 0.01):.6f}"
        ),
        (
            "percentage trusted_shrink_area < 0.01: "
            f"{percentage_lt(rows, 'trusted_shrink_area', 0.01):.6f}"
        ),
        "",
        "diagnostic hints:",
        "- mean iou_fixed_despl > 0.85: DESPL and fixed are very similar; correction room is small.",
        "- trusted_expand/shrink means < 0.01: CCR barely changes fixed pseudo.",
        "- trusted_expand_area > 0.30: possible over-expansion.",
        "- corr_area much larger than fixed_area: possible false-positive increase.",
        "- corr_area much smaller than fixed_area: possible missing-object increase.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnosis(cfg):
    manifest = ccr_manifest_path(cfg)
    rows = read_jsonl(manifest)
    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "diagnosis"
    csv_path = out_dir / "ccr_train_stats.csv"
    summary_path = out_dir / "ccr_summary.txt"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows)
    print(f"wrote_csv = {csv_path}")
    print(f"wrote_summary = {summary_path}")
    print(f"num_rows = {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose CCR-DPL pseudo cache.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_diagnosis(cfg)


if __name__ == "__main__":
    main()
