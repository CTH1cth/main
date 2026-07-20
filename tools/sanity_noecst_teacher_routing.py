#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.static_weight import build_effective_static_weight
from common.teacher_routing import (
    build_identity_teacher_route,
    get_teacher_routing_mode,
    teacher_routing_uses_ecst,
    validate_teacher_routing_config,
)
from common.utils import load_config, set_seed
from model import build_seg_head
from train import (
    build_loaders,
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    teacher_route_bce_with_logits,
    weighted_bce_with_logits,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst.py"
)


def main():
    parser = argparse.ArgumentParser(
        description="No-grad real-batch sanity for sw_ones No-ECST routing."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.max_samples < 1 or args.max_samples > 5:
        raise ValueError("--max-samples must be in [1,5] for this sanity")

    cfg = load_config(args.config)
    mode = validate_teacher_routing_config(cfg)
    if mode != "none" or teacher_routing_uses_ecst(cfg):
        raise RuntimeError(
            f"Sanity requires explicit No-ECST mode, got mode={mode!r}"
        )
    set_seed(int(cfg.SEED))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset, loader, _ = build_loaders(
        cfg,
        max_train_samples=int(args.max_samples),
    )
    batch = next(iter(loader))
    forbidden_gt_keys = {
        key
        for key in batch
        if str(key).lower() in {"gt", "ground_truth", "mask"}
    }
    if forbidden_gt_keys:
        raise RuntimeError(f"Training batch leaked GT fields: {forbidden_gt_keys}")

    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    student.eval()
    teacher.eval()
    set_model_epoch(student, 1)
    set_model_epoch(teacher, 1)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    with torch.no_grad():
        student_out = forward_seg_head(
            student,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=True,
        )
        teacher_out = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
        )
        final_logits = resize_logits_for_loss(
            extract_logits(student_out), cfg
        )
        teacher_logits = resize_logits_for_loss(
            extract_logits(teacher_out), cfg
        )
        teacher_prob = teacher_logits.sigmoid()
        teacher_target = (teacher_prob >= 0.5).float()
        route_map, _ = build_identity_teacher_route(teacher_prob)

        target_soft = batch["pu_target_soft"].to(device).float()
        raw_static_weight = batch["pu_weight_map"].to(device).float()
        static_weight, static_mode = build_effective_static_weight(
            cfg,
            batch,
            raw_static_weight,
            device,
        )
        if static_mode != "ones":
            raise RuntimeError(f"Expected static mode ones, got {static_mode}")
        if not torch.equal(static_weight, torch.ones_like(static_weight)):
            raise RuntimeError("Effective static map is not exactly one")

        logits_by_branch = {
            "final": final_logits,
            "coarse": resize_logits_for_loss(
                student_out["coarse_logits_68"], cfg
            ),
            "base": resize_logits_for_loss(student_out["base_logits"], cfg),
        }
        losses = {}
        differences = {}
        static_losses = {}
        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
        for branch, logits in logits_by_branch.items():
            routed = teacher_route_bce_with_logits(
                logits,
                teacher_target,
                route_map,
                cfg,
                routing_scale=0.0,
                apply_to_loss=True,
                eps=eps,
            )
            plain = F.binary_cross_entropy_with_logits(
                logits, teacher_target, reduction="mean"
            )
            difference = abs(float(routed.item()) - float(plain.item()))
            if difference >= 1e-7:
                raise RuntimeError(
                    f"{branch} teacher BCE mismatch: {difference:.12g}"
                )
            static_loss = weighted_bce_with_logits(
                logits,
                target_soft,
                static_weight,
                eps=eps,
            )
            if not bool(torch.isfinite(routed).item()) or not bool(
                torch.isfinite(static_loss).item()
            ):
                raise RuntimeError(f"{branch} loss contains NaN/Inf")
            losses[branch] = float(routed.item())
            differences[branch] = difference
            static_losses[branch] = float(static_loss.item())

    print(f"config={args.config}")
    print(f"device={device}")
    print(f"samples={len(dataset)}")
    print(f"batch_keys={sorted(batch)}")
    print(f"teacher_routing_mode={get_teacher_routing_mode(cfg)}")
    print("ecst_memory_initialized=False")
    print(f"teacher_target_shape={list(teacher_target.shape)}")
    print(
        "teacher_route_map_min/mean/max="
        f"{float(route_map.min()):.6f}/"
        f"{float(route_map.mean()):.6f}/"
        f"{float(route_map.max()):.6f}"
    )
    print(
        "static_weight_min/mean/max="
        f"{float(static_weight.min()):.6f}/"
        f"{float(static_weight.mean()):.6f}/"
        f"{float(static_weight.max()):.6f}"
    )
    print(f"teacher_losses={losses}")
    print(f"teacher_plain_bce_abs_diff={differences}")
    print(f"static_losses={static_losses}")
    print("sanity_passed=True")


if __name__ == "__main__":
    main()
