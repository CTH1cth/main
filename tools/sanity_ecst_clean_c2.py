#!/usr/bin/env python3
"""Bounded real-cache C2 sanity: at most five samples and one optimizer step."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.ecst_clean import (  # noqa: E402
    build_clean_history_bg_reliability,
    build_ecst_clean_teacher_weight_map,
    get_ecst_clean_continuous_strengths,
    get_ecst_clean_scale,
    validate_ecst_clean_config,
)
from common.utils import load_config  # noqa: E402
from model import build_seg_head, update_ema  # noqa: E402
from train import (  # noqa: E402
    C2_PARENT_CONFIG,
    audit_c2_config_difference,
    build_clean_ecst_checkpoint_extra,
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    teacher_route_bce_with_logits,
    teacher_routing_apply_flag,
)


DEFAULT_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_c2_nohist_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_OUTPUT = (
    MAIN_ROOT.parent
    / "workdir"
    / "dinov1_s8_dabe_clean_v1_dp_clean_ecst_c2_nohist_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
    / "sanity_5x1"
)


def _inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _prepare_output(path):
    path = Path(path).resolve()
    if _inside(path, MAIN_ROOT):
        raise RuntimeError(f"Sanity output must be outside the repository: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Sanity output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _branch_logits(output, cfg):
    if not isinstance(output, dict):
        raise RuntimeError("C2 sanity requires DAGP-NDR auxiliary outputs.")
    return {
        "final": resize_logits_for_loss(extract_logits(output), cfg),
        "coarse": resize_logits_for_loss(output["coarse_logits_68"], cfg),
        "base": resize_logits_for_loss(output["base_logits"], cfg),
    }


def _legacy_c3_reference(cfg, batch, teacher, mean, second, count, epoch):
    target = batch["dabe_clean_target_68"].detach().to(teacher.device).float()
    latent = batch["dabe_clean_recoverability_68"].detach().to(teacher.device).float()
    temporal = build_clean_history_bg_reliability(
        mean,
        second,
        count,
        min_history=int(cfg.ECST_CLEAN_MIN_HISTORY),
        variance_tau=float(cfg.ECST_CLEAN_VARIANCE_TAU),
    )
    signed = (2.0 * target - 1.0).detach()
    positive = signed.clamp_min(0.0).detach()
    negative = (-signed).clamp_min(0.0).detach()
    teacher_fg = (teacher.detach().float() > 0.5).detach()
    scale = float(get_ecst_clean_scale(cfg, epoch))
    strengths = get_ecst_clean_continuous_strengths(cfg)
    weight_min = float(cfg.ECST_CLEAN_WEIGHT_MIN)
    suppression_max = 1.0 - weight_min
    erase_raw = ((1.0 - float(cfg.ECST_CLEAN_CONFLICT_FLOOR)) * positive).detach()
    erase = (scale * strengths["erase"] * erase_raw).clamp(
        0.0, suppression_max
    ).detach()
    recovery_raw = (
        (1.0 - float(cfg.ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR))
        * latent
        * (1.0 - temporal["history_bg_reliability"])
    ).detach()
    recovery = (scale * strengths["recovery"] * recovery_raw).clamp(
        0.0, suppression_max
    ).detach()
    bg_suppression = (
        1.0 - (1.0 - erase) * (1.0 - recovery)
    ).clamp(0.0, suppression_max).detach()
    bg_weight = (1.0 - bg_suppression).clamp(weight_min, 1.0).detach()
    add_raw = ((1.0 - float(cfg.ECST_CLEAN_CONFLICT_FLOOR)) * negative).detach()
    add = (scale * strengths["add"] * add_raw).clamp(
        0.0, suppression_max
    ).detach()
    fg_weight = (1.0 - add).clamp(weight_min, 1.0).detach()
    return torch.where(teacher_fg, fg_weight, bg_weight).clamp(
        weight_min, 1.0
    ).detach()


def _assert_detached_finite(name, value):
    if value.requires_grad:
        raise RuntimeError(f"{name} unexpectedly requires gradients.")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if not 1 <= int(args.max_samples) <= 5:
        parser.error("--max_samples must be in [1,5].")
    if int(args.max_epochs) != 1:
        parser.error("--max_epochs must equal 1 for the bounded C2 sanity.")
    if not torch.cuda.is_available():
        raise RuntimeError("C2 real-batch sanity requires CUDA.")

    output_root = _prepare_output(args.output_root)
    os.chdir(MAIN_ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = MAIN_ROOT / config_path
    cfg = load_config(config_path)
    validate_ecst_clean_config(cfg)
    config_differences = list(audit_c2_config_difference(cfg))
    parent_cfg = load_config(MAIN_ROOT / C2_PARENT_CONFIG)
    validate_ecst_clean_config(parent_cfg)
    cfg.NUM_WORKERS = 0

    dataset = CachedTrainDataset(cfg, max_samples=int(args.max_samples))
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    forbidden = sorted(
        key
        for key in batch
        if key.startswith("legacy_ecst_")
        or "static_weight_map" in key
        or key in {"weight_map", "pu_weight_map"}
    )
    if forbidden:
        raise RuntimeError(f"C2 batch leaked forbidden fields: {forbidden}")

    device = torch.device("cuda")
    student = build_seg_head(dataset.in_channels, cfg).to(device).train()
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.DINO["lr"]),
        weight_decay=float(cfg.DINO.get("weight_decay", 0.0)),
    )
    set_model_epoch(student, 1)
    set_model_epoch(teacher, 1)
    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    student_output = forward_seg_head(
        student, model_input, cfg, image_68=image_68, return_aux=True
    )
    branches = _branch_logits(student_output, cfg)
    with torch.no_grad():
        teacher_output = forward_seg_head(
            teacher, model_input, cfg, image_68=image_68, return_aux=False
        )
        teacher_prob = resize_logits_for_loss(
            extract_logits(teacher_output), cfg
        ).sigmoid().detach()
        teacher_target = (teacher_prob > 0.5).float().detach()

    route_epoch1, stats_epoch1, states_epoch1 = (
        build_ecst_clean_teacher_weight_map(
            cfg=cfg,
            batch=batch,
            teacher_prob=teacher_prob,
            epoch=1,
            device=device,
            return_states=True,
        )
    )
    route_active, stats_active, states_active = (
        build_ecst_clean_teacher_weight_map(
            cfg=cfg,
            batch=batch,
            teacher_prob=teacher_prob,
            epoch=10,
            device=device,
            return_states=True,
        )
    )
    for name, stats, states in (
        ("epoch1", stats_epoch1, states_epoch1),
        ("active", stats_active, states_active),
    ):
        if stats.get("history_fields_consumed") is not False:
            raise RuntimeError(f"C2 {name} route consumed history.")
        if any("history" in key or "temporal" in key for key in states):
            raise RuntimeError(f"C2 {name} states contain history fields.")
        if float(stats["latent_effective_max_abs_error"]) > 1e-6:
            raise RuntimeError(f"C2 {name} latent support was altered.")
        if not torch.allclose(
            states["latent_effective"],
            states["latent_support"],
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError(f"C2 {name} latent identity failed.")
    if not torch.equal(route_epoch1, torch.ones_like(route_epoch1)):
        raise RuntimeError("Inherited epoch-1 schedule should produce identity route.")
    if torch.equal(route_active, torch.ones_like(route_active)):
        raise RuntimeError("Active epoch-10 C2 route is unexpectedly all one.")
    for name, value in (
        ("target", batch["dabe_clean_target_68"].to(device)),
        ("teacher_prob", teacher_prob),
        ("teacher_target", teacher_target),
        ("route_epoch1", route_epoch1),
        ("route_active", route_active),
        ("latent_support", states_active["latent_support"]),
        ("latent_effective", states_active["latent_effective"]),
    ):
        _assert_detached_finite(name, value)

    # C3 regression uses separate synthetic past-only moments; C2 never sees them.
    history_mean = torch.full_like(teacher_prob, 0.25)
    history_second = history_mean.square()
    history_count = torch.full(
        (teacher_prob.shape[0],), 3, dtype=torch.long, device=device
    )
    c3_map, _, _ = build_ecst_clean_teacher_weight_map(
        cfg=parent_cfg,
        batch=batch,
        teacher_prob=teacher_prob,
        temporal_mean=history_mean,
        temporal_second=history_second,
        history_count=history_count,
        epoch=10,
        device=device,
        return_states=True,
    )
    c3_reference = _legacy_c3_reference(
        parent_cfg,
        batch,
        teacher_prob,
        history_mean,
        history_second,
        history_count,
        10,
    )
    c3_max_abs_error = float((c3_map - c3_reference).abs().max().item())
    if c3_max_abs_error > 1e-6:
        raise RuntimeError(
            f"C3 v5 numerical regression failed: {c3_max_abs_error}."
        )

    target = batch["dabe_clean_target_68"].to(device).float().detach()
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(1, cfg)
    branch_weights = {
        "final": 1.0,
        "coarse": float(cfg.LAMBDA_NDR_COARSE_AUX),
        "base": float(cfg.LAMBDA_BASE_AUX),
    }
    static_losses = {}
    teacher_losses = {}
    map_ids = []
    for name, logits in branches.items():
        static_losses[name] = F.binary_cross_entropy_with_logits(
            logits, target, reduction="mean"
        )
        map_ids.append(id(route_epoch1))
        teacher_losses[name] = teacher_route_bce_with_logits(
            logits,
            teacher_target,
            route_epoch1,
            cfg,
            routing_scale=float(stats_epoch1["ecst_scale"]),
            apply_to_loss=teacher_routing_apply_flag(cfg, name),
            eps=float(cfg.DABE_PU_WEIGHTED_BCE_EPS),
        )
    if len(set(map_ids)) != 1:
        raise RuntimeError("C2 three Teacher branches did not share one map.")
    denominator = sum(branch_weights.values())
    static_group = sum(
        branch_weights[name] * static_losses[name] for name in branch_weights
    ) / denominator
    teacher_group = sum(
        branch_weights[name] * teacher_losses[name] for name in branch_weights
    ) / denominator
    loss = static_weight * static_group + teacher_weight * teacher_group
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("C2 sanity loss is non-finite.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in student.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("C2 Student gradients are missing or non-finite.")
    gradient_norm = float(
        torch.sqrt(sum(value.float().square().sum() for value in gradients)).item()
    )
    optimizer.step()
    update_ema(student, teacher, global_step=1, ema_weight=float(cfg.EMA_WEIGHT))
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("C2 EMA Teacher received gradients.")

    checkpoint_extra = build_clean_ecst_checkpoint_extra(cfg, None)
    if "clean_ecst_temporal_memory" in checkpoint_extra:
        raise RuntimeError("C2 checkpoint payload contains temporal memory.")
    diagnostics_path = output_root / "diagnostics.pt"
    torch.save(
        {
            "sample_index": batch["sample_index"].cpu(),
            "dataset_name": list(batch["dataset_name"]),
            "stem": list(batch["stem"]),
            "target": target.cpu(),
            "teacher_target": teacher_target.cpu(),
            "route_epoch1": route_epoch1.cpu(),
            "route_active_epoch10": route_active.cpu(),
            "latent_support": states_active["latent_support"].cpu(),
            "latent_effective": states_active["latent_effective"].cpu(),
        },
        diagnostics_path,
    )
    result = {
        "schema": "clean_ecst_c2_nohistory_sanity_v1",
        "config": str(config_path.resolve()),
        "config_differences_vs_c3": config_differences,
        "samples": int(target.shape[0]),
        "max_epochs": int(args.max_epochs),
        "sample_index": batch["sample_index"].tolist(),
        "dataset_name": list(batch["dataset_name"]),
        "stem": list(batch["stem"]),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "optimizer_step_called": True,
        "ema_update_called": True,
        "temporal_memory_initialized": False,
        "temporal_memory_fetched": False,
        "temporal_memory_updated": False,
        "temporal_memory_saved": False,
        "history_fields_consumed": False,
        "latent_support_mean": float(states_active["latent_support"].mean()),
        "latent_effective_mean": float(states_active["latent_effective"].mean()),
        "latent_effective_max_abs_error": float(
            (states_active["latent_effective"] - states_active["latent_support"])
            .abs()
            .max()
        ),
        "epoch1_schedule_scale": float(stats_epoch1["schedule_scale"]),
        "epoch1_map_is_identity": bool(
            torch.equal(route_epoch1, torch.ones_like(route_epoch1))
        ),
        "active_probe_epoch": 10,
        "active_map_is_identity": bool(
            torch.equal(route_active, torch.ones_like(route_active))
        ),
        "active_map_min_mean_max": [
            float(route_active.min()),
            float(route_active.mean()),
            float(route_active.max()),
        ],
        "same_map_for_final_coarse_base": len(set(map_ids)) == 1,
        "teacher_map_requires_grad": bool(route_active.requires_grad),
        "teacher_has_gradient": False,
        "dino_has_gradient": bool(model_input.requires_grad),
        "static_target_requires_grad": bool(target.requires_grad),
        "c3_regression_max_abs_error": c3_max_abs_error,
        "validation_called": False,
        "evaluation_called": False,
        "checkpoint_saved": False,
        "diagnostics_path": str(diagnostics_path.resolve()),
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"result={result_path.resolve()}")


if __name__ == "__main__":
    main()
