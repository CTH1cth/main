import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import CODMetrics  # noqa: E402
from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    find_gt_path,
    load_config,
    manifest_to_map,
    nper_pseudo_bank_manifest_path,
    read_jsonl,
    torch_load,
)


METHODS = ("p_fixed", "p_despl", "p_gcm", "p_init")
CSV_FIELDS = [
    "scope",
    "dataset",
    "method",
    "S_m",
    "F_beta^w",
    "F_beta^m",
    "E_phi^m",
    "M",
    "IoU",
    "Recall",
    "Area",
    "num_samples",
]


def load_gt(path):
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def extra_stats(gt, pred):
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    inter = torch.logical_and(gt_b, pred_b).sum().item()
    union = torch.logical_or(gt_b, pred_b).sum().item()
    gt_count = gt_b.sum().item()
    return {
        "iou": 1.0 if union == 0 else float(inter / union),
        "recall": 1.0 if gt_count == 0 else float(inter / gt_count),
        "area": float(pred_b.float().mean().item()),
    }


class Accumulator:
    def __init__(self):
        self.metrics = CODMetrics()
        self.ious = []
        self.recalls = []
        self.areas = []
        self.count = 0

    def step(self, gt, pred):
        self.metrics.step(gt.unsqueeze(0), pred.unsqueeze(0))
        stats = extra_stats(gt, pred)
        self.ious.append(stats["iou"])
        self.recalls.append(stats["recall"])
        self.areas.append(stats["area"])
        self.count += 1

    def result(self):
        result = self.metrics.get_result()
        return {
            "S_m": float(result["SMeasure"]),
            "F_beta^w": float(result["WFM"]),
            "F_beta^m": float(result["F_MEAN"]),
            "E_phi^m": float(result["E_MEAN"]),
            "M": float(result["MAE"]),
            "IoU": float(np.mean(self.ious)) if self.ious else 0.0,
            "Recall": float(np.mean(self.recalls)) if self.recalls else 0.0,
            "Area": float(np.mean(self.areas)) if self.areas else 0.0,
            "num_samples": int(self.count),
        }


def make_row(scope, dataset, method, accumulator):
    values = accumulator.result()
    return {
        "scope": scope,
        "dataset": dataset,
        "method": method,
        **values,
    }


def eval_nper_pseudo_bank(cfg, max_samples=-1, logger=print):
    manifest_path = nper_pseudo_bank_manifest_path(cfg)
    rows = read_jsonl(manifest_path)
    row_map = manifest_to_map(rows, manifest_path)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]

    by_dataset = defaultdict(lambda: {method: Accumulator() for method in METHODS})
    overall = {method: Accumulator() for method in METHODS}
    logger("train_gt_used = true | diagnostic_only = true")
    for item in items:
        key = (item["dataset"], item["stem"])
        if key not in row_map:
            raise RuntimeError(f"NPER pseudo bank missing {key}")
        payload = torch_load(row_map[key]["cache_path"], map_location="cpu")
        gt_path = find_gt_path(cfg.DATA_ROOT, item["dataset"], item["stem"])
        gt = load_gt(gt_path).float()
        for method in METHODS:
            pseudo = payload[method].float()
            pseudo = F.interpolate(
                pseudo.unsqueeze(0),
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            pred = (pseudo > 0.5).float()
            by_dataset[item["dataset"]][method].step(gt, pred)
            overall[method].step(gt, pred)

    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "diagnosis"
    ensure_dir(out_dir)
    csv_path = out_dir / "pseudo_bank_eval.csv"
    summary_path = out_dir / "summary.txt"

    out_rows = []
    for dataset in sorted(by_dataset):
        for method in METHODS:
            out_rows.append(make_row("dataset", dataset, method, by_dataset[dataset][method]))
    for method in METHODS:
        out_rows.append(make_row("overall", "ALL", method, overall[method]))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = ["NPER pseudo bank GT diagnostic", "train_gt_used = true", ""]
    for row in out_rows:
        if row["scope"] != "overall":
            continue
        lines.append(
            f"{row['method']}: S_m={row['S_m']:.4f} Fw={row['F_beta^w']:.4f} "
            f"Fm={row['F_beta^m']:.4f} E={row['E_phi^m']:.4f} M={row['M']:.4f} "
            f"IoU={row['IoU']:.4f} Recall={row['Recall']:.4f} "
            f"Area={row['Area']:.4f} n={row['num_samples']}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"wrote_csv = {csv_path}")
    logger(f"wrote_summary = {summary_path}")
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate NPER pseudo bank against train GT for diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_nper_pseudo_bank(cfg, max_samples=args.max_samples, logger=print)


if __name__ == "__main__":
    main()
