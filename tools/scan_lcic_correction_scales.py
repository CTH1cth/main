#!/usr/bin/env python3
"""Diagnostic grid scan for frozen LCIC consensus/innovation contributions.

The checkpoint is never modified. Each feature is forwarded once to obtain the
anchor and both learned correction maps, then all requested scale pairs are
evaluated without exporting prediction PNGs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedEvalDataset  # noqa: E402
from common.metrics import Smeasure, _prepare_data  # noqa: E402
from common.utils import ensure_dir, load_config, torch_load  # noqa: E402
from eval import (  # noqa: E402
    dataloader_worker_kwargs,
    infer_in_channels,
    make_model_input,
)
from model import build_seg_head  # noqa: E402
from models.lcic import (  # noqa: E402
    LCICHead,
    build_local_dino_affinity,
    local_affinity_propagate,
)


DEFAULT_SCALES = "0,0.5,1,1.5,2,3"


def parse_scales(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"LCIC scales must be finite and non-negative: {item!r}.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("At least one LCIC scale is required.")
    return tuple(values)


def build_scale_pairs(consensus_scales, innovation_scales):
    return tuple(
        (float(consensus_scale), float(innovation_scale))
        for consensus_scale in consensus_scales
        for innovation_scale in innovation_scales
    )


@torch.no_grad()
def lcic_scaled_logits(model, feature, scale_pairs):
    """Return [num_pairs,B,1,H,W] logits while sharing affinity computation."""

    if not isinstance(model, LCICHead) or not (
        model.use_consensus and model.use_innovation
    ):
        raise RuntimeError("Scale scan requires an LCIC-D full decoder.")
    if not scale_pairs:
        raise ValueError("Scale scan requires at least one scale pair.")

    affinity = build_local_dino_affinity(feature, eps=model.affinity_eps)
    anchor_logits = model.anchor(feature)
    consensus_delta = (
        local_affinity_propagate(affinity, anchor_logits) - anchor_logits
    )
    local_consensus_feature = local_affinity_propagate(
        affinity, feature.detach()
    )
    innovation_logits = model.innovation_head(
        feature.detach() - local_consensus_feature
    )
    consensus_scales = feature.new_tensor(
        [item[0] for item in scale_pairs]
    ).view(-1, 1, 1, 1, 1)
    innovation_scales = feature.new_tensor(
        [item[1] for item in scale_pairs]
    ).view(-1, 1, 1, 1, 1)
    return (
        anchor_logits.unsqueeze(0)
        + consensus_scales
        * model.consensus_gain
        * model.alpha
        * consensus_delta.unsqueeze(0)
        + innovation_scales
        * model.innovation_gain
        * model.beta
        * innovation_logits.unsqueeze(0)
    )


def adaptive_e_score(prediction, ground_truth):
    prediction, ground_truth = _prepare_data(
        pred=prediction, gt=ground_truth
    )
    ground_truth_foreground = int(np.count_nonzero(ground_truth))
    size = int(ground_truth.size)
    threshold = min(2.0 * float(prediction.mean()), 1.0)
    binary_prediction = prediction >= threshold
    foreground_foreground = int(
        np.count_nonzero(binary_prediction & ground_truth)
    )
    foreground_background = int(
        np.count_nonzero(binary_prediction & ~ground_truth)
    )
    predicted_foreground = foreground_foreground + foreground_background
    predicted_background = size - predicted_foreground
    if ground_truth_foreground == 0:
        enhanced_sum = float(predicted_background)
    elif ground_truth_foreground == size:
        enhanced_sum = float(predicted_foreground)
    else:
        background_foreground = ground_truth_foreground - foreground_foreground
        background_background = predicted_background - background_foreground
        part_counts = (
            foreground_foreground,
            foreground_background,
            background_foreground,
            background_background,
        )
        prediction_mean = predicted_foreground / size
        ground_truth_mean = ground_truth_foreground / size
        combinations = (
            (1 - prediction_mean, 1 - ground_truth_mean),
            (1 - prediction_mean, -ground_truth_mean),
            (-prediction_mean, 1 - ground_truth_mean),
            (-prediction_mean, -ground_truth_mean),
        )
        enhanced_sum = 0.0
        for count, (left, right) in zip(part_counts, combinations):
            alignment = 2.0 * left * right / (
                left * left + right * right + np.spacing(1)
            )
            enhanced_sum += ((alignment + 1.0) ** 2 / 4.0) * count
    return enhanced_sum / (size - 1 + np.spacing(1))


class CoreMetricAccumulator:
    """Accumulate only the three bottleneck metrics plus prediction area."""

    def __init__(self):
        self.smeasure = Smeasure()
        self.mae_sum = 0.0
        self.e_adp_sum = 0.0
        self.area_sum = 0.0
        self.count = 0

    def step(self, ground_truth_tensor, prediction_tensor):
        ground_truth = (
            ground_truth_tensor.detach().cpu().numpy().astype(float).squeeze()
        )
        prediction = (
            prediction_tensor.detach().cpu().numpy().astype(float).squeeze()
        )
        prepared_prediction, prepared_ground_truth = _prepare_data(
            pred=prediction, gt=ground_truth
        )
        self.smeasure.step(prediction, ground_truth)
        self.mae_sum += float(
            np.mean(np.abs(prepared_prediction - prepared_ground_truth))
        )
        self.e_adp_sum += float(adaptive_e_score(prediction, ground_truth))
        self.area_sum += float(np.mean(prediction))
        self.count += 1

    def result(self):
        if self.count == 0:
            raise RuntimeError("No samples were accumulated.")
        return {
            "SMeasure": float(np.mean(self.smeasure.sms)),
            "E_ADP": self.e_adp_sum / self.count,
            "MAE": self.mae_sum / self.count,
            "prediction_area": self.area_sum / self.count,
            "images": self.count,
        }


@torch.no_grad()
def scan_dataset(
    cfg,
    model,
    dataset_name,
    scale_pairs,
    threshold,
    device,
    max_samples=-1,
):
    dataset = CachedEvalDataset(
        cfg,
        split="test",
        datasets=[dataset_name],
        max_samples=int(max_samples),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(cfg),
    )
    accumulators = {
        pair: CoreMetricAccumulator() for pair in scale_pairs
    }
    model.eval()
    for batch in loader:
        ground_truth = batch["gt"].to(device, non_blocking=True).float()
        feature = make_model_input(cfg, batch, device)
        all_logits = lcic_scaled_logits(model, feature, scale_pairs)
        pair_count, batch_size = all_logits.shape[:2]
        resized_logits = F.interpolate(
            all_logits.reshape(pair_count * batch_size, *all_logits.shape[2:]),
            size=ground_truth.shape[-2:],
            mode="bilinear",
        ).reshape(pair_count, batch_size, 1, *ground_truth.shape[-2:])
        predictions = (resized_logits.sigmoid() > float(threshold)).float()
        for index, pair in enumerate(scale_pairs):
            accumulators[pair].step(ground_truth, predictions[index])
    return {pair: accumulator.result() for pair, accumulator in accumulators.items()}


def write_csv(path, rows):
    ensure_dir(path.parent)
    fieldnames = (
        "dataset",
        "consensus_scale",
        "innovation_scale",
        "SMeasure",
        "E_ADP",
        "MAE",
        "prediction_area",
        "images",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Scan frozen LCIC correction multipliers without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["TE-CAMO", "TE-COD10K"]
    )
    parser.add_argument("--consensus_scales", default=DEFAULT_SCALES)
    parser.add_argument("--innovation_scales", default=DEFAULT_SCALES)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--work_dir", default=None)
    args = parser.parse_args()

    if int(args.max_samples) == 0 or int(args.max_samples) < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    cfg = load_config(args.config)
    if str(getattr(cfg, "LCIC_VARIANT", "")).lower() != "d_full":
        raise RuntimeError("LCIC correction scan requires LCIC_VARIANT='d_full'.")
    unknown_datasets = sorted(set(args.datasets).difference(cfg.TEST_DATASETS))
    if unknown_datasets:
        raise ValueError(f"Unknown test datasets: {unknown_datasets}.")
    threshold = float(cfg.THRESHOLD if args.threshold is None else args.threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid output threshold: {threshold}.")

    consensus_scales = parse_scales(args.consensus_scales)
    innovation_scales = parse_scales(args.innovation_scales)
    scale_pairs = build_scale_pairs(consensus_scales, innovation_scales)
    output_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(cfg.WORK_ROOT)
        / cfg.EXP_NAME
        / f"eval_lcic_scale_scan_{Path(args.ckpt).stem}"
    )
    ensure_dir(output_dir)

    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError("Checkpoint/config backbone mismatch.")
    student_state = checkpoint["student"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_seg_head(infer_in_channels(student_state), cfg).to(device)
    model.load_state_dict(student_state, strict=True)
    if not isinstance(model, LCICHead):
        raise RuntimeError("Checkpoint/config did not build an LCIC decoder.")

    log_path = output_dir / "scan.log"
    rows = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        def log(message):
            print(message, flush=True)
            print(message, file=log_handle, flush=True)

        log(f"device={device}")
        log(f"checkpoint={Path(args.ckpt).resolve()}")
        log(f"checkpoint_epoch={int(checkpoint.get('epoch', -1))}")
        log(f"datasets={','.join(args.datasets)}")
        log(f"threshold={threshold:.6f}")
        log(f"scale_pairs={len(scale_pairs)}")
        log("prediction_png_saved=False")
        log("test_gt_used_for_diagnostic_metrics=True")
        log("result_is_not_a_leakage_free_final_protocol=True")
        for dataset_name in args.datasets:
            log(f"[Scan] dataset={dataset_name} | status=running")
            results = scan_dataset(
                cfg=cfg,
                model=model,
                dataset_name=dataset_name,
                scale_pairs=scale_pairs,
                threshold=threshold,
                device=device,
                max_samples=args.max_samples,
            )
            for pair in scale_pairs:
                result = results[pair]
                row = {
                    "dataset": dataset_name,
                    "consensus_scale": pair[0],
                    "innovation_scale": pair[1],
                    **result,
                }
                rows.append(row)
                log(
                    f"[Result] dataset={dataset_name} | "
                    f"consensus_scale={pair[0]:g} | innovation_scale={pair[1]:g} | "
                    f"S={result['SMeasure']:.6f} | "
                    f"adp_E={result['E_ADP']:.6f} | "
                    f"MAE={result['MAE']:.6f} | "
                    f"area={result['prediction_area']:.6f}"
                )
            log(f"[Scan] dataset={dataset_name} | status=complete")
    csv_path = output_dir / "metrics.csv"
    write_csv(csv_path, rows)
    print(f"metrics_csv={csv_path.resolve()}", flush=True)
    print(f"scan_log={log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
