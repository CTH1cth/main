#!/usr/bin/env python3
"""Offline, protocol-preserving audit of HSD base/coarse/final predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedEvalDataset
from common.metrics import CODMetrics
from common.utils import ensure_dir, load_config, torch_load
from model import build_seg_head
from models.online_dino_last4 import FrozenDINOv1Last4Extractor


STAGES = ("base37", "coarse74", "final148")
TRANSITIONS = (("base37", "coarse74"), ("coarse74", "final148"))
TRANSITION_CATEGORIES = (
    (0, 1, 1, "0_to_1_gt1"),
    (0, 1, 0, "0_to_1_gt0"),
    (1, 0, 0, "1_to_0_gt0"),
    (1, 0, 1, "1_to_0_gt1"),
)


def transition_category_counts(source, target, gt):
    """Return the four task-defined, GT-conditioned transition counts."""

    source = source.bool()
    target = target.bool()
    gt = gt.bool()
    if source.shape != target.shape or source.shape != gt.shape:
        raise RuntimeError(
            "Transition tensors must have identical shapes, got "
            f"{list(source.shape)}/{list(target.shape)}/{list(gt.shape)}."
        )
    result = {}
    for source_value, target_value, gt_value, name in TRANSITION_CATEGORIES:
        mask = (
            (source == bool(source_value))
            & (target == bool(target_value))
            & (gt == bool(gt_value))
        )
        result[name] = int(mask.sum().item())
    return result


def _new_binary_stats():
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "pred_fg": 0, "pixels": 0}


def _accumulate_binary_stats(stats, pred, gt):
    pred, gt = pred.bool(), gt.bool()
    stats["tp"] += int((pred & gt).sum().item())
    stats["fp"] += int((pred & ~gt).sum().item())
    stats["tn"] += int((~pred & ~gt).sum().item())
    stats["fn"] += int((~pred & gt).sum().item())
    stats["pred_fg"] += int(pred.sum().item())
    stats["pixels"] += int(pred.numel())


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _stage_row(dataset, stage, metrics, stats):
    result = metrics.get_result()
    tp, fp, tn, fn = (stats[name] for name in ("tp", "fp", "tn", "fn"))
    return {
        "dataset": dataset,
        "stage": stage,
        "S_m": float(result["SMeasure"]),
        "F_beta_w": float(result["WFM"]),
        "F_beta_m": float(result["F_MEAN"]),
        "E_phi_m": float(result["E_MEAN"]),
        "MAE": float(result["MAE"]),
        "prediction_area": _safe_ratio(stats["pred_fg"], stats["pixels"]),
        "Precision": _safe_ratio(tp, tp + fp),
        "Recall": _safe_ratio(tp, tp + fn),
        "IoU": _safe_ratio(tp, tp + fp + fn),
        "FP_per_BG": _safe_ratio(fp, fp + tn),
        "FN_per_FG": _safe_ratio(fn, tp + fn),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "pixels": stats["pixels"],
    }


def _infer_in_channels(state):
    for key in ("adapters.f9.0.weight", "adapters.f12.0.weight"):
        if key in state:
            return int(state[key].shape[1])
    raise KeyError("Checkpoint does not look like an HSD checkpoint (adapter weights absent).")


def _model_input(cfg, batch, device, online_dino):
    if bool(getattr(cfg, "ONLINE_DINO_LAST4", False)):
        if online_dino is None or "dino_input_296" not in batch:
            raise RuntimeError("Online HSD audit requires dino_input_296 and extractor.")
        return online_dino(
            batch["dino_input_296"].to(device, non_blocking=True).float()
        )
    return {
        f"f{layer}": batch[f"feature_l{layer}"].to(
            device, non_blocking=True
        ).float()
        for layer in (9, 10, 11, 12)
    }


@torch.no_grad()
def audit_dataset(cfg, model, dataset_name, device, online_dino, aggregates):
    dataset = CachedEvalDataset(cfg, split="test", datasets=[dataset_name])
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(cfg, "VAL_BATCH_SIZE", 1)),
        shuffle=False,
        num_workers=int(getattr(cfg, "NUM_WORKERS", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    metrics = {stage: CODMetrics() for stage in STAGES}
    binary_stats = {stage: _new_binary_stats() for stage in STAGES}
    transition_counts = {
        transition: defaultdict(int) for transition in TRANSITIONS
    }
    threshold = float(getattr(cfg, "THRESHOLD", 0.5))

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True).float()
        features = _model_input(cfg, batch, device, online_dino)
        image_148 = (
            batch["image_148"].to(device, non_blocking=True).float()
            if bool(getattr(cfg, "USE_DETAIL", False))
            else None
        )
        output = model(features, image_148=image_148)
        logits = {
            "base37": output["base_logits_37"],
            "coarse74": output["coarse_logits_74"],
            "final148": output["final_logits"],
        }
        predictions = {}
        gt_binary = gt > 0.5
        for stage, stage_logits in logits.items():
            resized = F.interpolate(
                stage_logits,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.sigmoid(resized) > threshold
            predictions[stage] = pred
            pred_float = pred.float()
            metrics[stage].step(gt, pred_float)
            aggregates["metrics"][stage].step(gt, pred_float)
            _accumulate_binary_stats(binary_stats[stage], pred, gt_binary)
            _accumulate_binary_stats(
                aggregates["binary_stats"][stage], pred, gt_binary
            )

        for transition in TRANSITIONS:
            counts = transition_category_counts(
                predictions[transition[0]], predictions[transition[1]], gt_binary
            )
            for name, count in counts.items():
                transition_counts[transition][name] += count
                aggregates["transition_counts"][transition][name] += count
        aggregates["pixels"] += int(gt.numel())

    stage_rows = [
        _stage_row(dataset_name, stage, metrics[stage], binary_stats[stage])
        for stage in STAGES
    ]
    transition_rows = _transition_rows(
        dataset_name,
        transition_counts,
        sum(row["pixels"] for row in stage_rows[:1]),
    )
    return stage_rows, transition_rows


def _transition_rows(dataset, transition_counts, pixels):
    rows = []
    for source, target in TRANSITIONS:
        counts = transition_counts[(source, target)]
        for _, _, _, category in TRANSITION_CATEGORIES:
            rows.append(
                {
                    "dataset": dataset,
                    "source_stage": source,
                    "target_stage": target,
                    "category": category,
                    "count": int(counts[category]),
                    "ratio_all_pixels": _safe_ratio(counts[category], pixels),
                }
            )
    return rows


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write empty audit CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if str(getattr(cfg, "DECODER_TYPE", "")).lower() != "hsd_v1":
        raise RuntimeError("Decoder stage audit supports HSD checkpoints only.")
    checkpoint = torch_load(args.checkpoint, map_location="cpu")
    state = checkpoint["student"] if "student" in checkpoint else checkpoint
    if "backbone_key" in checkpoint and checkpoint["backbone_key"] != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint['backbone_key']} != {cfg.BACKBONE_KEY}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_seg_head(_infer_in_channels(state), cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    online_dino = None
    if bool(getattr(cfg, "ONLINE_DINO_LAST4", False)):
        online_dino = FrozenDINOv1Last4Extractor(cfg).to(device).eval()

    aggregates = {
        "metrics": {stage: CODMetrics() for stage in STAGES},
        "binary_stats": {stage: _new_binary_stats() for stage in STAGES},
        "transition_counts": {
            transition: defaultdict(int) for transition in TRANSITIONS
        },
        "pixels": 0,
    }
    stage_rows, transition_rows = [], []
    for dataset_name in args.datasets:
        dataset_stage, dataset_transition = audit_dataset(
            cfg, model, dataset_name, device, online_dino, aggregates
        )
        stage_rows.extend(dataset_stage)
        transition_rows.extend(dataset_transition)

    stage_rows.extend(
        _stage_row(
            "ALL", stage, aggregates["metrics"][stage],
            aggregates["binary_stats"][stage]
        )
        for stage in STAGES
    )
    transition_rows.extend(
        _transition_rows(
            "ALL", aggregates["transition_counts"], aggregates["pixels"]
        )
    )

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    _write_csv(out_dir / "stage_metrics.csv", stage_rows)
    _write_csv(out_dir / "stage_transitions.csv", transition_rows)
    summary = {
        "schema": "hsd_decoder_stage_audit_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "datasets": list(args.datasets),
        "threshold": float(getattr(cfg, "THRESHOLD", 0.5)),
        "prediction_protocol": "sigmoid_then_strict_threshold; bilinear_to_GT",
        "stage_metrics": stage_rows,
        "stage_transitions": transition_rows,
    }
    with (out_dir / "stage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "PASS", "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
