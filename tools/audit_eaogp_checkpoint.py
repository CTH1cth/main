#!/usr/bin/env python3
"""GT-free checkpoint shadow audit for EAOGP-v1.

The tool loads an existing EMA-Teacher checkpoint and computes EAOGP targets
without training, optimizer steps, EMA updates, validation, or GT access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CTH_ROOT = MAIN_ROOT.parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.eaogp import build_eaogp_target, validate_eaogp_config  # noqa: E402
from common.utils import load_config, set_seed, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from tools.visualize_eaogp_batch import render_eaogp_visualization  # noqa: E402
from train import (  # noqa: E402
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DEFAULT_CONFIG = MAIN_ROOT / "configs" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_eaogp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workdir" / "eaogp_v1_shadow_audit"
FORBIDDEN_GT_KEYS = {
    "gt",
    "gt_path",
    "mask",
    "mask_path",
    "ground_truth",
}
MEAN_FIELDS = (
    "teacher_binary_area",
    "teacher_prob_mean",
    "dabe_static_area",
    "anchor_confidence_mean",
    "dino_graph_consensus_mean",
    "dual_graph_consensus_mean",
    "dual_vs_dino_shift_abs_mean",
    "teacher_uncertainty_mean",
    "raw_eaogp_target_mean",
    "effective_eaogp_target_mean",
    "effective_eaogp_hard_area",
    "new_fg_ratio",
    "new_bg_ratio",
    "anchor_delta_abs_mean",
    "graph_delta_abs_mean",
    "task_gate_mean",
    "dual_graph_entropy",
    "dino_graph_entropy",
    "fallback_row_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a no-GT EAOGP shadow audit from an EMA-Teacher checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Manifest-order sample count; use -1 for the full training manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--visualize-samples", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_inside_cth(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path != CTH_ROOT and CTH_ROOT not in path.parents:
        raise RuntimeError(f"Path must stay inside {CTH_ROOT}: {path}")
    return path


def _prepare_output(root: Path, *, overwrite: bool) -> None:
    tracked = (
        root / "summary.json",
        root / "by_sample.csv",
    )
    existing = [path for path in tracked if path.exists()]
    visualization_root = root / "visualizations"
    if visualization_root.is_dir() and any(visualization_root.iterdir()):
        existing.append(visualization_root)
    if existing and not overwrite:
        raise FileExistsError(
            "EAOGP shadow-audit output already exists; pass --overwrite to replace "
            "matching files: " + ", ".join(str(path) for path in existing)
        )
    root.mkdir(parents=True, exist_ok=True)
    visualization_root.mkdir(parents=True, exist_ok=True)
    (root / "payloads").mkdir(parents=True, exist_ok=True)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_allocated(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _batch_strings(value: Any, batch_size: int, name: str) -> list[str]:
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return [str(item) for item in value]
    if isinstance(value, str) and batch_size == 1:
        return [value]
    raise RuntimeError(f"Cannot decode batch string field {name!r}: {value!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("EAOGP shadow audit produced no sample rows.")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid values for EAOGP summary field {key!r}.")
    return sum(values) / len(values)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.epoch < 1:
        raise ValueError("--epoch must be positive.")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.visualize_samples < 0:
        raise ValueError("--visualize-samples must be non-negative.")

    config_path = _resolve_inside_cth(args.config)
    checkpoint_path = _resolve_inside_cth(args.checkpoint)
    output_root = _resolve_inside_cth(args.output_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    _prepare_output(output_root, overwrite=bool(args.overwrite))

    cfg = load_config(config_path)
    config_audit = validate_eaogp_config(cfg)
    set_seed(int(cfg.SEED))
    device = _device(args.device)
    dataset = CachedTrainDataset(cfg, max_samples=int(args.max_samples))
    if len(dataset) < 1:
        raise RuntimeError("EAOGP shadow audit dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=min(int(args.batch_size), len(dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "teacher" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no EMA-Teacher state: {checkpoint_path}")
    checkpoint_epoch = int(checkpoint.get("epoch", args.epoch))
    if checkpoint_epoch != int(args.epoch):
        raise RuntimeError(
            f"Checkpoint epoch mismatch: requested={args.epoch}, stored={checkpoint_epoch}."
        )
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(checkpoint["teacher"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    set_model_epoch(teacher, int(args.epoch))

    rows: list[dict[str, Any]] = []
    visualizations: list[dict[str, str]] = []
    prediction_max_error = 0.0
    dino_row_sum_max_error = 0.0
    dual_row_sum_max_error = 0.0
    baseline_time_seconds = 0.0
    eaogp_time_seconds = 0.0
    baseline_peak_values: list[int] = []
    eaogp_peak_values: list[int] = []
    batch_count = 0
    visualized = 0

    for batch in loader:
        leaked_gt = sorted(
            key for key in batch if str(key).strip().lower() in FORBIDDEN_GT_KEYS
        )
        if leaked_gt:
            raise RuntimeError(f"EAOGP shadow batch contains GT fields: {leaked_gt}")
        gt_read_value = batch.get("dabe_clean_dabe_v2_training_gt_read", False)
        gt_read_reported = (
            bool(gt_read_value.bool().any().item())
            if torch.is_tensor(gt_read_value)
            else bool(gt_read_value)
        )
        if gt_read_reported:
            raise RuntimeError("EAOGP same-source cache reports training GT access.")

        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        foreground = batch["dabe_clean_dabe_v2_soft_68"].to(device).float().detach()
        background = batch["dabe_clean_dabe_v2_bg_evidence_68"].to(device).float().detach()
        static_target = batch["dabe_clean_static_target_68"].to(device).float().detach()
        if not torch.equal(static_target, (foreground > 0.5).float()):
            raise RuntimeError("EAOGP shadow static target is not strict DABE-v2 F>0.5.")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        baseline_output = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
            return_eaogp_aux=False,
        )
        _sync(device)
        baseline_time_seconds += time.perf_counter() - start
        baseline_peak = _peak_allocated(device)
        if baseline_peak is not None:
            baseline_peak_values.append(baseline_peak)
        baseline_logits = resize_logits_for_loss(
            extract_logits(baseline_output), cfg
        ).detach()
        del baseline_output

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        teacher_output = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=True,
            return_eaogp_aux=True,
        )
        teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), cfg)
        teacher_prob = teacher_logits.sigmoid().detach()
        teacher_binary = (teacher_prob >= 0.5).float().detach()
        scale = teacher_output["eaogp_dagp_scale"].detach()
        effective_target, result = build_eaogp_target(
            foreground_response_68=foreground,
            background_evidence_68=background,
            static_target_68=static_target,
            teacher_prob_68=teacher_prob,
            teacher_binary_68=teacher_binary,
            teacher_embedding_37=teacher_output["eaogp_teacher_embedding_37"],
            dino_topk_idx=teacher_output["eaogp_dino_topk_idx"],
            dino_topk_weight=teacher_output["eaogp_dino_topk_weight"],
            scale=scale,
        )
        _sync(device)
        eaogp_time_seconds += time.perf_counter() - start
        eaogp_peak = _peak_allocated(device)
        if eaogp_peak is not None:
            eaogp_peak_values.append(eaogp_peak)
        batch_count += 1

        prediction_error = float((baseline_logits - teacher_logits).abs().max().item())
        prediction_max_error = max(prediction_max_error, prediction_error)
        if prediction_error > 1e-6:
            raise RuntimeError(
                "EAOGP auxiliary output changed Teacher predictions: "
                f"max_abs_error={prediction_error:.9g}."
            )
        if effective_target.requires_grad:
            raise RuntimeError("EAOGP effective target unexpectedly requires gradients.")
        identity_route = torch.ones_like(effective_target).detach()
        if identity_route.requires_grad or not torch.equal(
            identity_route, torch.ones_like(identity_route)
        ):
            raise RuntimeError("Internal identity-route construction failed.")

        batch_size = int(teacher_prob.shape[0])
        datasets = _batch_strings(batch["dataset"], batch_size, "dataset")
        stems = _batch_strings(batch["stem"], batch_size, "stem")
        dino_row_sum_max_error = max(
            dino_row_sum_max_error, float(result["dino_row_sum_error"])
        )
        dual_row_sum_max_error = max(
            dual_row_sum_max_error, float(result["dual_row_sum_error"])
        )
        for local_index in range(batch_size):
            sample_teacher = teacher_prob[local_index : local_index + 1]
            sample_binary = teacher_binary[local_index : local_index + 1]
            sample_static = static_target[local_index : local_index + 1]
            sample_row = {
                "dataset": datasets[local_index],
                "stem": stems[local_index],
                "teacher_binary_area": float(sample_binary.mean().item()),
                "teacher_prob_mean": float(sample_teacher.mean().item()),
                "dabe_static_area": float(sample_static.mean().item()),
                "anchor_confidence_mean": float(
                    result["anchor_confidence_68"][local_index].mean().item()
                ),
                "dino_graph_consensus_mean": float(
                    result["dino_graph_consensus_68"][local_index].mean().item()
                ),
                "dual_graph_consensus_mean": float(
                    result["dual_graph_consensus_68"][local_index].mean().item()
                ),
                "dual_vs_dino_shift_abs_mean": float(
                    (
                        result["dual_graph_consensus_68"][local_index]
                        - result["dino_graph_consensus_68"][local_index]
                    ).abs().mean().item()
                ),
                "teacher_uncertainty_mean": float(
                    result["teacher_uncertainty_68"][local_index].mean().item()
                ),
                "raw_eaogp_target_mean": float(
                    result["raw_eaogp_target_68"][local_index].mean().item()
                ),
                "effective_eaogp_target_mean": float(
                    effective_target[local_index].mean().item()
                ),
                "effective_eaogp_hard_area": float(
                    (effective_target[local_index] >= 0.5).float().mean().item()
                ),
                "new_fg_ratio": float(
                    result["new_foreground_68"][local_index].mean().item()
                ),
                "new_bg_ratio": float(
                    result["new_background_68"][local_index].mean().item()
                ),
                "anchor_delta_abs_mean": float(
                    result["anchor_delta_68"][local_index].abs().mean().item()
                ),
                "graph_delta_abs_mean": float(
                    result["graph_delta_68"][local_index].abs().mean().item()
                ),
                "task_gate_mean": float(
                    result["teacher_task_gate"][local_index].mean().item()
                ),
                "task_gate_min": float(
                    result["teacher_task_gate"][local_index].min().item()
                ),
                "task_gate_max": float(
                    result["teacher_task_gate"][local_index].max().item()
                ),
                "dual_graph_entropy": float(
                    (
                        -result["dual_graph_weight"][local_index]
                        * result["dual_graph_weight"][local_index].clamp_min(1e-6).log()
                    ).sum(-1).mean().item()
                ),
                "dino_graph_entropy": float(
                    (
                        -teacher_output["eaogp_dino_topk_weight"][local_index]
                        * teacher_output["eaogp_dino_topk_weight"][local_index]
                        .clamp_min(1e-6)
                        .log()
                    ).sum(-1).mean().item()
                ),
                "fallback_row_ratio": float(result["fallback_row_ratio"]),
                "eaogp_scale": float(scale.item()),
            }
            rows.append(sample_row)

            if visualized < int(args.visualize_samples):
                payload = {
                    "schema": "eaogp_shadow_visual_payload_v1",
                    "dataset": datasets[local_index],
                    "stem": stems[local_index],
                    "training_gt_used": False,
                    "image_68": batch["image_68"][local_index].detach().cpu(),
                    "dabe_static_68": sample_static.detach().cpu(),
                    "foreground_response_68": foreground[
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "background_evidence_68": background[
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "anchor_confidence_68": result["anchor_confidence_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "teacher_binary_68": sample_binary.detach().cpu(),
                    "teacher_uncertainty_68": result["teacher_uncertainty_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "dino_graph_consensus_68": result["dino_graph_consensus_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "dual_graph_consensus_68": result["dual_graph_consensus_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "effective_teacher_target_68": effective_target[
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "new_foreground_68": result["new_foreground_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                    "new_background_68": result["new_background_68"][
                        local_index : local_index + 1
                    ].detach().cpu(),
                }
                payload_name = (
                    f"{visualized:03d}_{datasets[local_index]}__"
                    f"{stems[local_index]}.pt"
                ).replace("/", "_")
                payload_path = output_root / "payloads" / payload_name
                torch.save(payload, payload_path)
                visualization = render_eaogp_visualization(
                    payload,
                    output_root / "visualizations",
                )
                visualization["payload"] = str(payload_path)
                visualizations.append(visualization)
                visualized += 1

        del (
            baseline_logits,
            teacher_output,
            teacher_logits,
            teacher_prob,
            teacher_binary,
            effective_target,
            result,
            identity_route,
        )

    _write_csv(output_root / "by_sample.csv", rows)
    baseline_batch_ms = 1000.0 * baseline_time_seconds / max(1, batch_count)
    eaogp_batch_ms = 1000.0 * eaogp_time_seconds / max(1, batch_count)
    extra_batch_ms = eaogp_batch_ms - baseline_batch_ms
    time_change_percent = (
        100.0 * extra_batch_ms / baseline_batch_ms if baseline_batch_ms > 0.0 else None
    )
    baseline_peak = max(baseline_peak_values) if baseline_peak_values else None
    eaogp_peak = max(eaogp_peak_values) if eaogp_peak_values else None
    peak_increment = (
        eaogp_peak - baseline_peak
        if baseline_peak is not None and eaogp_peak is not None
        else None
    )
    peak_change_percent = (
        100.0 * peak_increment / baseline_peak
        if baseline_peak is not None and baseline_peak > 0 and peak_increment is not None
        else None
    )
    summary_means = {key: _mean(rows, key) for key in MEAN_FIELDS}
    summary = {
        "schema": "eaogp_v1_checkpoint_shadow_audit",
        "status": "PASS",
        "config": str(config_path),
        "config_validation": config_audit["status"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "sample_count": len(rows),
        "batch_count": batch_count,
        "device": str(device),
        "training_gt_used": False,
        "history_used": False,
        "legacy_regions_used": False,
        "future_teacher_used": False,
        "learnable_router_used": False,
        "new_trainable_parameters": 0,
        "new_checkpoint_state_entries": 0,
        "full_teacher_dense_graph_constructed": False,
        "graph_complexity": "O(B*N*K*D), N=1369, K=12, D=64",
        "eaogp_scale": _mean(rows, "eaogp_scale"),
        "prediction_aux_max_abs_error": prediction_max_error,
        "dino_row_sum_max_error": dino_row_sum_max_error,
        "dual_row_sum_max_error": dual_row_sum_max_error,
        "task_gate_min": min(float(row["task_gate_min"]) for row in rows),
        "task_gate_max": max(float(row["task_gate_max"]) for row in rows),
        **summary_means,
        "baseline_batch_time_ms": baseline_batch_ms,
        "eaogp_batch_time_ms": eaogp_batch_ms,
        "batch_average_extra_time_ms": extra_batch_ms,
        "batch_time_change_percent_vs_no_eaogp_aux": time_change_percent,
        "baseline_peak_memory_bytes": baseline_peak,
        "eaogp_peak_memory_bytes": eaogp_peak,
        "peak_memory_increment_bytes": peak_increment,
        "peak_memory_change_percent_vs_no_eaogp_aux": peak_change_percent,
        "by_sample_csv": str(output_root / "by_sample.csv"),
        "visualizations": visualizations,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
