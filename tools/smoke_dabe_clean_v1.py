#!/usr/bin/env python3
"""One-real-batch DABE-Clean forward/backward smoke without optimizer/EMA."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.ecst import build_ecst_teacher_weight_map  # noqa: E402
from common.ecst_minimal import build_ecst_minimal_teacher_weight_map  # noqa: E402
from common.utils import load_config  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    teacher_weighted_bce_with_logits,
)


DEFAULT_FULL_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5.py"
)
DEFAULT_MINIMAL_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_minimal_ecst_r1_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5.py"
)


def _prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Smoke output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_detached(name, value):
    if not torch.is_tensor(value):
        return
    if value.requires_grad:
        raise RuntimeError(f"{name} unexpectedly requires gradients.")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", default=DEFAULT_FULL_CONFIG)
    parser.add_argument("--minimal-config", default=DEFAULT_MINIMAL_CONFIG)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 16:
        parser.error("--max-samples must be in [1,16].")
    if not 1 <= args.batch_size <= args.max_samples:
        parser.error("--batch-size must be in [1,max_samples].")
    output_root = _prepare_output(args.output_root)

    cfg = load_config(args.full_config)
    cfg.DABE_CLEAN_ROOT = str(Path(args.clean_root).resolve())
    cfg.DABE_CLEAN_LEGACY_REGION_ROOT = str(Path(args.legacy_root).resolve())
    cfg.NUM_WORKERS = 0
    minimal_cfg = load_config(args.minimal_config)
    minimal_cfg.DABE_CLEAN_ROOT = cfg.DABE_CLEAN_ROOT

    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    forbidden = sorted(
        key
        for key in batch
        if key in {"gt", "mask", "weight_map", "pu_weight_map"}
        or "static_weight_map" in key
    )
    if forbidden:
        raise RuntimeError(f"Clean smoke batch leaked forbidden fields: {forbidden}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    set_model_epoch(student, args.epoch)
    set_model_epoch(teacher, args.epoch)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    output = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
    )
    final_logits = resize_logits_for_loss(extract_logits(output), cfg)
    if not isinstance(output, dict):
        raise RuntimeError("DABE-Clean smoke expects DAGP-Safe auxiliary outputs.")
    required_logits = ("coarse_logits_68", "base_logits")
    missing_logits = [key for key in required_logits if key not in output]
    if missing_logits:
        raise RuntimeError(f"Smoke model output is missing: {missing_logits}")
    branch_logits = {
        "final": final_logits,
        "coarse": resize_logits_for_loss(output["coarse_logits_68"], cfg),
        "base": resize_logits_for_loss(output["base_logits"], cfg),
    }

    with torch.no_grad():
        teacher_output = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
        )
        teacher_prob = resize_logits_for_loss(
            extract_logits(teacher_output), cfg
        ).sigmoid().detach()
        teacher_target = (teacher_prob >= 0.5).float().detach()

    sample_indices = batch["sample_index"].long()
    temporal_mean = torch.zeros_like(teacher_prob)
    temporal_second = torch.zeros_like(teacher_prob)
    history_count = torch.zeros(
        int(teacher_prob.shape[0]), dtype=torch.long, device=device
    )
    full_map, full_stats, full_states = build_ecst_teacher_weight_map(
        cfg=cfg,
        batch=batch,
        teacher_prob=teacher_prob,
        temporal_mean=temporal_mean,
        temporal_second=temporal_second,
        history_count=history_count,
        epoch=args.epoch,
        device=device,
        return_states=True,
        region_key_prefix="legacy_ecst",
    )
    minimal_map, minimal_stats, minimal_states = (
        build_ecst_minimal_teacher_weight_map(
            cfg=minimal_cfg,
            batch=batch,
            teacher_prob=teacher_prob,
            temporal_mean=temporal_mean,
            temporal_second=temporal_second,
            history_count=history_count,
            epoch=args.epoch,
            device=device,
            return_states=True,
        )
    )

    target = batch["dabe_clean_target_68"].to(device).float().detach()
    _check_detached("target", target)
    _check_detached("teacher_prob", teacher_prob)
    _check_detached("teacher_target", teacher_target)
    _check_detached("full_map", full_map)
    _check_detached("minimal_map", minimal_map)
    _check_detached("temporal_mean", temporal_mean)
    _check_detached("temporal_second", temporal_second)
    _check_detached("model_input", model_input)
    for prefix, states in (("full", full_states), ("minimal", minimal_states)):
        for name, value in states.items():
            if torch.is_tensor(value):
                _check_detached(f"{prefix}.{name}", value)

    static_weight, teacher_weight = get_dabe_pu_despl_schedule(args.epoch, cfg)
    static_losses = {}
    teacher_losses = {}
    target_object_ids = []
    for name, logits in branch_logits.items():
        target_object_ids.append(id(target))
        static_losses[name] = F.binary_cross_entropy_with_logits(
            logits, target, reduction="mean"
        )
        teacher_losses[name] = teacher_weighted_bce_with_logits(
            logits,
            teacher_target,
            full_map,
            enabled=True,
            apply_to_loss=True,
            eps=float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6)),
        )
    if len(set(target_object_ids)) != 1:
        raise RuntimeError("The three static branches did not share one target object.")
    branch_weights = {
        "final": 1.0,
        "coarse": float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)),
        "base": float(getattr(cfg, "LAMBDA_BASE_AUX", 0.5)),
    }
    denominator = sum(branch_weights.values())
    static_group = sum(
        branch_weights[name] * static_losses[name] for name in branch_weights
    ) / denominator
    teacher_group = sum(
        branch_weights[name] * teacher_losses[name] for name in branch_weights
    ) / denominator
    loss = static_weight * static_group + teacher_weight * teacher_group
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Smoke loss is not finite.")
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in student.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(grad).all().item()) for grad in gradients):
        raise RuntimeError("Smoke gradients are missing or non-finite.")
    gradient_norm = float(
        torch.sqrt(sum(grad.float().square().sum() for grad in gradients)).item()
    )
    if gradient_norm <= 0.0:
        raise RuntimeError("Smoke gradient norm is zero.")

    diagnostics_path = output_root / "routing_diagnostics.pt"
    torch.save(
        {
            "sample_indices": sample_indices,
            "dataset_name": list(batch["dataset_name"]),
            "stem": list(batch["stem"]),
            "target": target.cpu(),
            "teacher_prob": teacher_prob.cpu(),
            "teacher_target": teacher_target.cpu(),
            "full_ecst_map": full_map.cpu(),
            "minimal_ecst_map": minimal_map.cpu(),
            "full_stats": full_stats,
            "minimal_stats": minimal_stats,
        },
        diagnostics_path,
    )
    result = {
        "schema": "dabe_clean_v1_real_batch_smoke",
        "device": str(device),
        "samples": int(target.shape[0]),
        "sample_indices": sample_indices.tolist(),
        "dataset_name": list(batch["dataset_name"]),
        "stem": list(batch["stem"]),
        "target_shape": list(target.shape),
        "branch_shapes": {key: list(value.shape) for key, value in branch_logits.items()},
        "same_static_target_for_three_branches": len(set(target_object_ids)) == 1,
        "clean_batch_has_static_weight_map": False,
        "legacy_routing_is_namespaced": all(
            key.startswith("legacy_ecst_")
            for key in batch
            if key.startswith("legacy_ecst_")
        ),
        "static_weight": float(static_weight),
        "teacher_weight": float(teacher_weight),
        "loss_static_group": float(static_group.detach()),
        "loss_teacher_group": float(teacher_group.detach()),
        "loss_total": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "full_ecst_map_min_mean_max": [
            float(full_map.min()),
            float(full_map.mean()),
            float(full_map.max()),
        ],
        "minimal_ecst_map_min_mean_max": [
            float(minimal_map.min()),
            float(minimal_map.mean()),
            float(minimal_map.max()),
        ],
        "optimizer_called": False,
        "ema_update_called": False,
        "validation_called": False,
        "evaluation_called": False,
        "diagnostics_path": str(diagnostics_path.resolve()),
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"result={result_path.resolve()}")


if __name__ == "__main__":
    main()

