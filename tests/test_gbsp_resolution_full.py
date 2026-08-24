from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tools.eval_gbsp_resolution_full import (
    _hist_ap_auroc,
    _read_296_continuous,
    _score_histogram,
)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_histogram_ap_auroc_perfect_ranking() -> None:
    prediction = np.asarray([[0.05, 0.20], [0.80, 0.95]], dtype=np.float32)
    gt = np.asarray([[False, False], [True, True]])
    fg, bg = _score_histogram(prediction, gt, 256)
    ap, auroc = _hist_ap_auroc(fg, bg)
    assert ap == 1.0
    assert auroc == 1.0


def test_read_296_continuous_combines_and_aliases_datasets(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    per_dataset = tmp_path / "per_dataset.csv"
    fields = {
        "method": "gbsp_r8",
        "protocol": "per_image_macro_native_gt",
        "valid_images": "250",
        "ap": "0.7",
        "auroc": "0.9",
    }
    _write(summary, [{**fields, "dataset": "ALL"}, {
        **fields,
        "dataset": "DATASET_MACRO",
        "protocol": "dataset_macro_of_per_image_native",
    }])
    _write(per_dataset, [{**fields, "dataset": "CAMO"}, {**fields, "dataset": "COD10K"}])
    rows = _read_296_continuous(summary, per_dataset)
    identities = {(row["scope"], row["dataset"]) for row in rows}
    assert ("image_macro", "ALL") in identities
    assert ("dataset_macro", "ALL") in identities
    assert ("dataset", "TE-CAMO") in identities
    assert ("dataset", "TE-COD10K") in identities
