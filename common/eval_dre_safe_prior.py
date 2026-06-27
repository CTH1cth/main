import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as cc_label

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.metrics import CODMetrics  # noqa: E402
from common.utils import ensure_dir, find_gt_path, load_config  # noqa: E402


METHODS = ("p_fixed", "p_despl", "p_base", "p_safe")
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
    "CC",
    "num_samples",
]


def load_gt(path):
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def connected_components(mask):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    _, num = cc_label(array)
    return int(num)


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
        "cc": connected_components(pred_b),
    }


class Accumulator:
    def __init__(self):
        self.metrics = CODMetrics()
        self.ious = []
        self.recalls = []
        self.areas = []
        self.ccs = []
        self.count = 0

    def step(self, gt, pred):
        self.metrics.step(gt.unsqueeze(0), pred.unsqueeze(0))
        stats = extra_stats(gt, pred)
        self.ious.append(stats["iou"])
        self.recalls.append(stats["recall"])
        self.areas.append(stats["area"])
        self.ccs.append(stats["cc"])
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
            "CC": float(np.mean(self.ccs)) if self.ccs else 0.0,
            "num_samples": int(self.count),
        }


def make_row(scope, dataset, method, accumulator):
    return {
        "scope": scope,
        "dataset": dataset,
        "method": method,
        **accumulator.result(),
    }


def sample_pseudos(sample):
    return {
        "p_fixed": sample["pseudo_fixed"].float(),
        "p_despl": sample["pseudo_despl"].float(),
        "p_base": sample["pseudo_base"].float(),
        "p_safe": sample["pseudo_safe"].float(),
    }


def eval_dre_safe_prior(cfg, max_samples=-1, logger=print):
    if not bool(getattr(cfg, "USE_DRE_SAFE_PRIOR", False)):
        raise RuntimeError("eval_dre_safe_prior requires USE_DRE_SAFE_PRIOR=True config.")
    dataset = CachedTrainDataset(cfg, max_samples=max_samples)
    by_dataset = defaultdict(lambda: {method: Accumulator() for method in METHODS})
    overall = {method: Accumulator() for method in METHODS}
    logger("train_gt_used = true | diagnostic_only = true")
    logger("use_gcm = false")
    for index in range(len(dataset)):
        sample = dataset[index]
        gt_path = find_gt_path(cfg.DATA_ROOT, sample["dataset"], sample["stem"])
        gt = load_gt(gt_path).float()
        for method, pseudo in sample_pseudos(sample).items():
            resized = F.interpolate(
                pseudo.unsqueeze(0),
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            pred = (resized > 0.5).float()
            by_dataset[sample["dataset"]][method].step(gt, pred)
            overall[method].step(gt, pred)

    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "diagnosis"
    ensure_dir(out_dir)
    csv_path = out_dir / "dre_safe_prior_eval.csv"
    summary_path = out_dir / "dre_safe_prior_summary.txt"

    rows = []
    for dataset_name in sorted(by_dataset):
        for method in METHODS:
            rows.append(make_row("dataset", dataset_name, method, by_dataset[dataset_name][method]))
    for method in METHODS:
        rows.append(make_row("overall", "ALL", method, overall[method]))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    overall_rows = {row["method"]: row for row in rows if row["scope"] == "overall"}
    safe = overall_rows["p_safe"]
    base = overall_rows["p_base"]
    recommend_train = safe["M"] <= base["M"] and safe["F_beta^w"] >= base["F_beta^w"]
    lines = [
        "DRE-SAFE prior GT diagnostic",
        "train_gt_used = true",
        "use_gcm = false",
        f"recommend_train = {recommend_train}",
        "",
    ]
    for method in METHODS:
        row = overall_rows[method]
        lines.append(
            f"{method}: S_m={row['S_m']:.4f} Fw={row['F_beta^w']:.4f} "
            f"Fm={row['F_beta^m']:.4f} E={row['E_phi^m']:.4f} M={row['M']:.4f} "
            f"IoU={row['IoU']:.4f} Recall={row['Recall']:.4f} "
            f"Area={row['Area']:.4f} CC={row['CC']:.4f} n={row['num_samples']}"
        )
    if not recommend_train:
        lines.append("")
        lines.append("WARNING: p_safe is worse than p_base by MAE or Fw; do not train dre_safe yet.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"wrote_csv = {csv_path}")
    logger(f"wrote_summary = {summary_path}")
    logger(f"recommend_train = {recommend_train}")
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRE-SAFE pseudo priors against train GT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_dre_safe_prior(cfg, max_samples=args.max_samples, logger=print)


if __name__ == "__main__":
    main()
