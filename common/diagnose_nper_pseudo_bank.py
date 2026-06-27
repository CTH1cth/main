import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (  # noqa: E402
    ensure_dir,
    load_config,
    nper_pseudo_bank_manifest_path,
    read_jsonl,
    torch_load,
)


CSV_FIELDS = [
    "dataset",
    "stem",
    "quality_score",
    "hard_score",
    "mnp_score_fixed",
    "mnp_score_despl",
    "mnp_score_gcm",
    "agreement_score",
    "stability_score",
    "fixed_area",
    "despl_area",
    "gcm_area",
    "init_area",
    "anchor_ratio",
    "weight_fixed",
    "weight_despl",
    "weight_gcm",
]


def diagnose_nper_pseudo_bank(cfg, logger=print):
    manifest_path = nper_pseudo_bank_manifest_path(cfg)
    rows = read_jsonl(manifest_path)
    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "diagnosis"
    ensure_dir(out_dir)
    csv_path = out_dir / "pseudo_bank_train_stats.csv"
    summary_path = out_dir / "pseudo_bank_summary.txt"

    records = []
    for row in rows:
        payload = torch_load(row["cache_path"], map_location="cpu")
        record = {field: payload.get(field, row.get(field, 0.0)) for field in CSV_FIELDS}
        record["dataset"] = payload.get("dataset", row.get("dataset"))
        record["stem"] = payload.get("stem", row.get("stem"))
        records.append(record)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    def values(name):
        return np.asarray([float(record.get(name, 0.0)) for record in records], dtype=np.float64)

    lines = [
        f"P_INIT_MODE = {getattr(cfg, 'P_INIT_MODE', 'quality_fusion')}",
        f"PSEUDO_USE_GCM = {bool(getattr(cfg, 'PSEUDO_USE_GCM', True))}",
        f"PSEUDO_USE_DESPL = {bool(getattr(cfg, 'PSEUDO_USE_DESPL', True))}",
        f"PSEUDO_USE_TEACHER = {bool(getattr(cfg, 'PSEUDO_USE_TEACHER', True))}",
        f"num_rows = {len(records)}",
    ]
    for name in (
        "quality_score",
        "hard_score",
        "agreement_score",
        "stability_score",
        "fixed_area",
        "despl_area",
        "gcm_area",
        "init_area",
        "anchor_ratio",
    ):
        arr = values(name)
        if arr.size == 0:
            continue
        lines.append(
            f"{name}: mean={arr.mean():.6f} p25={np.quantile(arr, 0.25):.6f} "
            f"p50={np.quantile(arr, 0.50):.6f} p75={np.quantile(arr, 0.75):.6f}"
        )
    if records:
        weights = np.stack([values("weight_fixed"), values("weight_despl"), values("weight_gcm")], axis=1)
        winners = weights.argmax(axis=1)
        names = ["fixed", "despl", "gcm"]
        for idx, name in enumerate(names):
            lines.append(f"winner_{name}_ratio = {float((winners == idx).mean()):.6f}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"wrote_csv = {csv_path}")
    logger(f"wrote_summary = {summary_path}")
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Diagnose NPER-UCOD-V1 pseudo bank.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    diagnose_nper_pseudo_bank(cfg, logger=print)


if __name__ == "__main__":
    main()
