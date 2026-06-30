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
    despl_paper_manifest_path,
    despl_pseudo_bank_manifest_path,
    ensure_dir,
    find_gt_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


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
    "view_consistency",
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
        self.consistencies = []
        self.count = 0

    def step(self, gt, pred, view_consistency=0.0):
        self.metrics.step(gt.unsqueeze(0), pred.unsqueeze(0))
        stats = extra_stats(gt, pred)
        self.ious.append(stats["iou"])
        self.recalls.append(stats["recall"])
        self.areas.append(stats["area"])
        self.consistencies.append(float(view_consistency))
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
            "view_consistency": float(np.mean(self.consistencies)) if self.consistencies else 0.0,
            "num_samples": int(self.count),
        }


def load_single_channel(row, name):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if name not in payload:
        raise KeyError(f"{name} missing from {row['cache_path']}")
    tensor = payload[name].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"{name} must be [1,H,W], got {list(tensor.shape)}: {row['cache_path']}")
    return tensor, payload


def resize_to_gt(tensor, gt):
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def make_row(scope, dataset, method, accumulator):
    return {
        "scope": scope,
        "dataset": dataset,
        "method": method,
        **accumulator.result(),
    }


def optional_paper_maps(cfg):
    out = {}
    for sign_mode in ("paper", "fixed_iou", "area_small"):
        manifest_path = despl_paper_manifest_path(cfg, sign_mode=sign_mode)
        if manifest_path.exists():
            rows = read_jsonl(manifest_path)
            out[sign_mode] = manifest_to_map(rows, manifest_path)
    return out


def eval_despl_paper(cfg, max_samples=-1, logger=print):
    old_manifest = despl_pseudo_bank_manifest_path(cfg)
    old_map = manifest_to_map(read_jsonl(old_manifest), old_manifest)
    paper_maps = optional_paper_maps(cfg)
    if "paper" not in paper_maps:
        raise RuntimeError(
            "DESPL-paper cache missing. Run: "
            "python common/cache_despl_paper.py --config configs/despl_paper_dinov1_s8.py --overwrite"
        )

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]

    methods = ["p_fixed", "p_despl_old", "p_despl_paper_soft", "p_despl_paper_binary"]
    if "fixed_iou" in paper_maps:
        methods.append("p_despl_paper_fixed_iou")
    if "area_small" in paper_maps:
        methods.append("p_despl_paper_area_small")

    by_dataset = defaultdict(lambda: {method: Accumulator() for method in methods})
    overall = {method: Accumulator() for method in methods}
    logger("train_gt_used = true | diagnostic_only = true")
    logger(f"methods = {', '.join(methods)}")

    for item in items:
        key = (item["dataset"], item["stem"])
        if key not in old_map:
            raise RuntimeError(f"Old DESPL pseudo bank missing {key}")
        gt = load_gt(find_gt_path(cfg.DATA_ROOT, item["dataset"], item["stem"])).float()

        old_payload = torch_load(old_map[key]["cache_path"], map_location="cpu")
        tensors = {
            "p_fixed": old_payload["p_fixed"].float(),
            "p_despl_old": old_payload["p_despl"].float(),
        }
        consistencies = {name: 0.0 for name in methods}
        paper_row = paper_maps["paper"][key]
        paper_payload = torch_load(paper_row["cache_path"], map_location="cpu")
        tensors["p_despl_paper_soft"] = paper_payload["p_despl_paper_soft"].float()
        tensors["p_despl_paper_binary"] = paper_payload["p_despl_paper"].float()
        consistencies["p_despl_paper_soft"] = float(paper_payload.get("view_consistency", 0.0))
        consistencies["p_despl_paper_binary"] = float(paper_payload.get("view_consistency", 0.0))

        if "fixed_iou" in paper_maps:
            payload = torch_load(paper_maps["fixed_iou"][key]["cache_path"], map_location="cpu")
            tensors["p_despl_paper_fixed_iou"] = payload["p_despl_paper_soft"].float()
            consistencies["p_despl_paper_fixed_iou"] = float(payload.get("view_consistency", 0.0))
        if "area_small" in paper_maps:
            payload = torch_load(paper_maps["area_small"][key]["cache_path"], map_location="cpu")
            tensors["p_despl_paper_area_small"] = payload["p_despl_paper_soft"].float()
            consistencies["p_despl_paper_area_small"] = float(payload.get("view_consistency", 0.0))

        for method in methods:
            pred = resize_to_gt(tensors[method], gt)
            by_dataset[item["dataset"]][method].step(gt, pred, consistencies.get(method, 0.0))
            overall[method].step(gt, pred, consistencies.get(method, 0.0))

    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "diagnosis"
    ensure_dir(out_dir)
    csv_path = out_dir / "despl_paper_eval.csv"
    summary_path = out_dir / "despl_paper_summary.txt"

    rows = []
    for dataset_name in sorted(by_dataset):
        for method in methods:
            rows.append(make_row("dataset", dataset_name, method, by_dataset[dataset_name][method]))
    for method in methods:
        rows.append(make_row("overall", "ALL", method, overall[method]))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    overall_rows = {row["method"]: row for row in rows if row["scope"] == "overall"}
    paper = overall_rows["p_despl_paper_soft"]
    old = overall_rows["p_despl_old"]
    recommend_train = paper["M"] <= old["M"] and paper["F_beta^w"] >= old["F_beta^w"]
    lines = [
        "DESPL-paper train GT diagnostic",
        "train_gt_used = true",
        f"recommend_train = {recommend_train}",
        "",
    ]
    for method in methods:
        row = overall_rows[method]
        lines.append(
            f"{method}: S_m={row['S_m']:.4f} Fw={row['F_beta^w']:.4f} "
            f"Fm={row['F_beta^m']:.4f} E={row['E_phi^m']:.4f} M={row['M']:.4f} "
            f"IoU={row['IoU']:.4f} Recall={row['Recall']:.4f} Area={row['Area']:.4f} "
            f"view_consistency={row['view_consistency']:.4f} n={row['num_samples']}"
        )
    if not recommend_train:
        lines.append("")
        lines.append("WARNING: p_despl_paper_soft is not better than p_despl_old by MAE/Fw; inspect sign mode or augmentations before full training.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"wrote_csv = {csv_path}")
    logger(f"wrote_summary = {summary_path}")
    logger(f"recommend_train = {recommend_train}")
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate DESPL-paper pseudo cache against train GT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_despl_paper(cfg, max_samples=args.max_samples, logger=print)


if __name__ == "__main__":
    main()
