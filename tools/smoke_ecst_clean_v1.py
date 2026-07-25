#!/usr/bin/env python3
"""One-real-CUDA-batch Clean-ECST-v1 forward/backward/update smoke.

This tool never runs validation or evaluation and refuses to place output
inside the source repository.  It is intentionally separate from train.py so
that executing it cannot start an epoch loop.
"""

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
from common.ecst import TemporalTeacherMemory  # noqa: E402
from common.ecst_clean import (  # noqa: E402
    build_ecst_clean_teacher_weight_map,
    validate_ecst_clean_config,
)
from common.utils import load_config  # noqa: E402
from model import build_seg_head, update_ema  # noqa: E402
from train import (  # noqa: E402
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
    "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v1_r1_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_OUTPUT = (
    MAIN_ROOT.parent
    / "workdir"
    / "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v1_r1_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
    / "smoke_one_batch"
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
        raise RuntimeError(f"Smoke output must be outside the repository: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Smoke output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_detached_finite(name, value):
    if value.requires_grad:
        raise RuntimeError(f"{name} unexpectedly requires gradients.")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")


def _three_branch_logits(output, cfg):
    if not isinstance(output, dict):
        raise RuntimeError("Clean-ECST smoke expects DAGP-Safe auxiliary outputs.")
    required = ("coarse_logits_68", "base_logits")
    missing = [key for key in required if key not in output]
    if missing:
        raise RuntimeError(f"Clean-ECST smoke output is missing: {missing}")
    return {
        "final": resize_logits_for_loss(extract_logits(output), cfg),
        "coarse": resize_logits_for_loss(output["coarse_logits_68"], cfg),
        "base": resize_logits_for_loss(output["base_logits"], cfg),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--clean-root",
        default=None,
        help="Optional DABE-Clean cache override; no PU/legacy root is accepted.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 8:
        parser.error("--max-samples must be in [1,8].")
    if not 1 <= args.batch_size <= args.max_samples:
        parser.error("--batch-size must be in [1,max-samples].")
    if not 1 <= args.epoch <= 21:
        parser.error("--epoch must be in [1,21].")
    if not torch.cuda.is_available():
        raise RuntimeError("Clean-ECST real-batch smoke requires CUDA.")

    output_root = _prepare_output(args.output_root)
    os.chdir(MAIN_ROOT)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = MAIN_ROOT / config_path
    cfg = load_config(config_path)
    if args.clean_root is not None:
        cfg.DABE_CLEAN_ROOT = str(Path(args.clean_root).resolve())
    cfg.NUM_WORKERS = 0
    validate_ecst_clean_config(cfg)

    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    legacy_or_weight_keys = sorted(
        key
        for key in batch
        if key.startswith("legacy_ecst_")
        or key in {"weight_map", "pu_weight_map", "dabe_clean_static_weight_map"}
        or "static_weight_map" in key
    )
    if legacy_or_weight_keys:
        raise RuntimeError(
            "Clean-ECST smoke batch leaked legacy/static-map fields: "
            f"{legacy_or_weight_keys}"
        )

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
    set_model_epoch(student, args.epoch)
    set_model_epoch(teacher, args.epoch)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    student_output = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
    )
    branch_logits = _three_branch_logits(student_output, cfg)
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
        teacher_target = (teacher_prob > 0.5).float().detach()

    indices = batch["sample_index"].long()
    memory = TemporalTeacherMemory(
        len(dataset), int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE), dtype="float16"
    )
    temporal_mean, temporal_second, history_count = memory.fetch(indices, device)
    if bool((history_count != 0).any().item()):
        raise RuntimeError("Fresh temporal memory is not empty.")
    route_map, route_stats, route_states = build_ecst_clean_teacher_weight_map(
        cfg=cfg,
        batch=batch,
        teacher_prob=teacher_prob,
        temporal_mean=temporal_mean,
        temporal_second=temporal_second,
        history_count=history_count,
        epoch=args.epoch,
        device=device,
        return_states=True,
    )
    is_contrec = str(getattr(cfg, "ECST_CLEAN_VERSION", "")) == (
        "v5_asym_continuous_recoverability"
    )
    epoch21_identity = None
    if is_contrec:
        recoverability = batch.get("dabe_clean_recoverability_68")
        if not torch.is_tensor(recoverability) or tuple(recoverability.shape) != tuple(
            teacher_prob.shape
        ):
            raise RuntimeError(
                "Clean-ECST v5 smoke requires recoverability [B,1,68,68]."
            )
        route21, _, states21 = build_ecst_clean_teacher_weight_map(
            cfg=cfg,
            batch=batch,
            teacher_prob=teacher_prob,
            temporal_mean=temporal_mean,
            temporal_second=temporal_second,
            history_count=history_count,
            epoch=21,
            device=device,
            return_states=True,
        )
        epoch21_identity = bool(torch.equal(route21, torch.ones_like(route21)))
        if not epoch21_identity:
            raise RuntimeError("Clean-ECST v5 epoch-21 route is not strict identity.")
        if any("ring" in key.lower() for key in states21):
            raise RuntimeError("Clean-ECST v5 unexpectedly generated Hard Ring state.")

    # The current prediction is written only after its route map is complete.
    route_before_update = route_map.clone()
    memory.update(
        indices,
        teacher_prob,
        rho=float(cfg.ECST_CLEAN_TEMPORAL_RHO),
    )
    if not torch.equal(route_before_update, route_map):
        raise RuntimeError("Temporal update mutated the already-built route map.")
    _, _, count_after = memory.fetch(indices, device)
    if not bool((count_after == 1).all().item()):
        raise RuntimeError("Temporal memory did not update exactly once.")

    target = batch["dabe_clean_target_68"].to(device).float().detach()
    for name, tensor in (
        ("DINO feature", model_input),
        ("Clean target", target),
        ("Teacher probability", teacher_prob),
        ("Teacher target", teacher_target),
        ("Teacher route map", route_map),
        ("temporal mean", temporal_mean),
        ("temporal second", temporal_second),
    ):
        _check_detached_finite(name, tensor)
    for name, tensor in route_states.items():
        if torch.is_tensor(tensor):
            _check_detached_finite(f"route_states.{name}", tensor)

    static_schedule, teacher_schedule = get_dabe_pu_despl_schedule(args.epoch, cfg)
    branch_weights = {
        "final": 1.0,
        "coarse": float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)),
        "base": float(getattr(cfg, "LAMBDA_BASE_AUX", 0.5)),
    }
    static_losses = {}
    teacher_losses = {}
    route_object_ids = []
    for name, logits in branch_logits.items():
        static_losses[name] = F.binary_cross_entropy_with_logits(
            logits, target, reduction="mean"
        )
        route_object_ids.append(id(route_map))
        teacher_losses[name] = teacher_route_bce_with_logits(
            logits=logits,
            target=teacher_target,
            teacher_map=route_map,
            cfg=cfg,
            routing_scale=float(route_stats["ecst_scale"]),
            apply_to_loss=teacher_routing_apply_flag(cfg, name),
            eps=float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6)),
        )
    if len(set(route_object_ids)) != 1:
        raise RuntimeError("Teacher branches did not share the same map object.")
    denominator = sum(branch_weights.values())
    static_group = sum(
        branch_weights[name] * static_losses[name] for name in branch_weights
    ) / denominator
    teacher_group = sum(
        branch_weights[name] * teacher_losses[name] for name in branch_weights
    ) / denominator
    loss = static_schedule * static_group + teacher_schedule * teacher_group
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Clean-ECST smoke loss is NaN/Inf.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad.detach()
        for parameter in student.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(grad).all().item()) for grad in gradients):
        raise RuntimeError("Student gradients are missing or non-finite.")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("EMA-Teacher unexpectedly received gradients.")
    gradient_norm = float(
        torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients)).item()
    )
    student_before_step = [parameter.detach().clone() for parameter in student.parameters()]
    optimizer.step()
    optimizer_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(student_before_step, student.parameters())
    )
    if not optimizer_changed:
        raise RuntimeError("AdamW optimizer.step() did not change Student parameters.")
    teacher_before_ema = [parameter.detach().clone() for parameter in teacher.parameters()]
    update_ema(
        student,
        teacher,
        global_step=1,
        ema_weight=float(cfg.EMA_WEIGHT),
    )
    ema_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(teacher_before_ema, teacher.parameters())
    )
    if not ema_changed:
        raise RuntimeError("EMA update did not change Teacher parameters.")

    diagnostics_path = output_root / "routing_diagnostics.pt"
    torch.save(
        {
            "sample_index": indices.cpu(),
            "dataset_name": list(batch["dataset_name"]),
            "stem": list(batch["stem"]),
            "target": target.cpu(),
            "teacher_prob": teacher_prob.cpu(),
            "teacher_target": teacher_target.cpu(),
            "route_map": route_map.cpu(),
            "route_stats": route_stats,
            "history_count_preupdate": history_count.cpu(),
            "history_count_postupdate": count_after.cpu(),
        },
        diagnostics_path,
    )
    result = {
        "schema": "clean_ecst_real_cuda_one_batch_smoke",
        "config": str(config_path.resolve()),
        "device": str(device),
        "samples": int(target.shape[0]),
        "sample_index": indices.tolist(),
        "dataset_name": list(batch["dataset_name"]),
        "stem": list(batch["stem"]),
        "loss": float(loss.detach()),
        "loss_static": float(static_group.detach()),
        "loss_teacher": float(teacher_group.detach()),
        "static_schedule": float(static_schedule),
        "teacher_schedule": float(teacher_schedule),
        "gradient_norm": gradient_norm,
        "optimizer_step_called": True,
        "optimizer_changed_student": optimizer_changed,
        "ema_update_called": True,
        "ema_changed_teacher": ema_changed,
        "teacher_has_gradient": False,
        "dino_has_gradient": bool(model_input.requires_grad),
        "clean_target_has_gradient": bool(target.requires_grad),
        "teacher_map_has_gradient": bool(route_map.requires_grad),
        "same_map_for_final_coarse_base": len(set(route_object_ids)) == 1,
        "continuous_recoverability_enabled": is_contrec,
        "hard_ring_generated": False if is_contrec else None,
        "epoch21_route_identity": epoch21_identity,
        "legacy_cache_accessed": False,
        "static_weight_map_used": False,
        "route_map_min_mean_max": [
            float(route_map.min()),
            float(route_map.mean()),
            float(route_map.max()),
        ],
        "preupdate_history_count": history_count.tolist(),
        "postupdate_history_count": count_after.tolist(),
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
