import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import check_cssd_hr_feature_cache, load_config  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_csd_aux_losses,
    build_csd_bg_reliable_mask,
    build_cssd_distillation_losses,
    build_dabe_pu_despl_static_target,
    build_rast_teacher_weight_map,
    compute_cssd_original_group_loss,
    extract_logits,
    forward_cssd_high_microbatches,
    forward_seg_head,
    get_cssd_scale,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


def finite_gradients(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def run_epoch_probe(cfg, student, teacher, batch, epoch, device):
    set_model_epoch(student, epoch)
    set_model_epoch(teacher, epoch)
    student.train()
    teacher.eval()
    student.zero_grad(set_to_none=True)

    if "gt" in batch:
        raise RuntimeError("CSSD train sanity unexpectedly received training GT.")
    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    static_target, static_weight_map, _ = build_dabe_pu_despl_static_target(
        cfg, target_soft, weight_map
    )
    bg_reliable = build_csd_bg_reliable_mask(
        batch,
        cfg,
        (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
        device,
    ).float()

    normal_output = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
        bg_reliable_68=bg_reliable,
    )
    teacher_normal_forward_count = 0
    with torch.no_grad():
        teacher_output = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
        )
        teacher_normal_forward_count += 1
        teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), cfg)
        teacher_binary = (teacher_logits.sigmoid() >= 0.5).float()

    teacher_map, rast_stats = build_rast_teacher_weight_map(
        cfg,
        batch,
        teacher_binary,
        epoch,
        device,
    )
    normal_parts = compute_cssd_original_group_loss(
        normal_output,
        static_target,
        static_weight_map,
        teacher_binary,
        teacher_map,
        rast_stats["rast_scale"],
        epoch,
        cfg,
    )
    loss_csd_bg, loss_csd_boundary, _ = build_csd_aux_losses(
        normal_output, batch, cfg, epoch, device
    )
    loss_normal = normal_parts["loss_group"] + loss_csd_bg + loss_csd_boundary

    cssd_scale = float(get_cssd_scale(epoch, cfg))
    high_forward_count = 0
    teacher_high_forward_count = 0
    lambda_hr = 0.0
    lambda_pred = 0.0
    lambda_boundary = 0.0
    loss_total = loss_normal
    high_output = None
    distill_stats = None
    if cssd_scale > 0.0:
        feature_hr = batch[str(getattr(cfg, "CSSD_HR_FEATURE_FIELD", "feature_cssd_hr"))]
        if feature_hr.dtype != torch.float32:
            raise RuntimeError(f"CSSD sanity expected float32 high cache, got {feature_hr.dtype}.")
        feature_hr = feature_hr.to(device, non_blocking=True)
        high_output, high_forward_count = forward_cssd_high_microbatches(
            student,
            feature_hr,
            image_68,
            bg_reliable,
            cfg,
        )
        high_parts = compute_cssd_original_group_loss(
            high_output,
            static_target,
            static_weight_map,
            teacher_binary,
            teacher_map,
            rast_stats["rast_scale"],
            epoch,
            cfg,
        )
        loss_core, loss_transfer, loss_pred, loss_boundary, distill_stats = (
            build_cssd_distillation_losses(
                normal_output,
                high_output,
                batch,
                image_68,
                cfg,
                cssd_scale,
            )
        )
        lambda_hr = float(cfg.CSSD_HR_SUP_WEIGHT_MAX) * cssd_scale
        lambda_pred = float(cfg.CSSD_PRED_WEIGHT_MAX) * cssd_scale
        lambda_boundary = float(cfg.CSSD_BOUNDARY_WEIGHT_MAX) * cssd_scale
        loss_dual = (normal_parts["loss_group"] + lambda_hr * high_parts["loss_group"]) / (
            1.0 + lambda_hr
        )
        loss_total = (
            loss_dual
            + loss_csd_bg
            + loss_csd_boundary
            + lambda_pred * loss_pred
            + lambda_boundary * loss_boundary
        )
        del loss_core, loss_transfer

    if teacher_normal_forward_count != 1 or teacher_high_forward_count != 0:
        raise RuntimeError("CSSD teacher must forward exactly once on the normal view only.")
    if not bool(torch.isfinite(loss_total).item()):
        raise RuntimeError(f"CSSD sanity loss is not finite at epoch {epoch}.")
    if epoch == 1:
        if cssd_scale != 0.0 or high_forward_count != 0:
            raise RuntimeError("epoch1 must skip the high-resolution forward entirely.")
        difference = float(torch.abs(loss_total.detach() - loss_normal.detach()).item())
        if difference != 0.0:
            raise RuntimeError(f"epoch1 CSSD loss differs from normal baseline: {difference}")
    if epoch == 7 and abs(cssd_scale - 1.0 / 9.0) > 1e-12:
        raise RuntimeError(f"epoch7 cssd_scale mismatch: {cssd_scale}")
    if epoch == 15:
        expected = (
            float(cfg.CSSD_HR_SUP_WEIGHT_MAX),
            float(cfg.CSSD_PRED_WEIGHT_MAX),
            float(cfg.CSSD_BOUNDARY_WEIGHT_MAX),
        )
        actual = (lambda_hr, lambda_pred, lambda_boundary)
        if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected)):
            raise RuntimeError(f"epoch15 CSSD lambda mismatch: actual={actual}, expected={expected}")
    if epoch == 21:
        static_schedule, teacher_schedule = get_dabe_pu_despl_schedule(epoch, cfg)
        if static_schedule != 0.0 or teacher_schedule != 1.0 or cssd_scale != 1.0:
            raise RuntimeError(
                "epoch21 must be teacher-only while CSSD stays active: "
                f"static={static_schedule}, teacher={teacher_schedule}, cssd={cssd_scale}"
            )

    loss_total.backward()
    if not finite_gradients(student):
        raise RuntimeError(f"CSSD sanity gradients contain NaN/Inf at epoch {epoch}.")

    print(
        f"epoch={epoch} | cssd_scale={cssd_scale:.8f} | "
        f"normal_shape={list(model_input.shape)} | "
        f"high_shape={list(batch[str(getattr(cfg, 'CSSD_HR_FEATURE_FIELD', 'feature_cssd_hr'))].shape)} | "
        f"normal_final={list(extract_logits(normal_output).shape)} | "
        f"high_final={list(extract_logits(high_output).shape) if high_output is not None else None} | "
        f"high_forward_count={high_forward_count} | teacher_normal/high=1/0 | "
        f"lambda_hr/pred/boundary={lambda_hr:.8f}/{lambda_pred:.8f}/{lambda_boundary:.8f} | "
        f"transfer_ratio={float(distill_stats['transfer_capped_ratio']) if distill_stats else 0.0:.8f} | "
        f"loss_normal={float(loss_normal.detach().item()):.8f} | "
        f"loss_total={float(loss_total.detach().item()):.8f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Single-batch CSSD-v1a schedule/loss/backward sanity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hr-cache-root", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    cfg = load_config(args.config)
    cfg.CSSD_HR_CACHE_ROOT = str(Path(args.hr_cache_root).resolve())
    check_cssd_hr_feature_cache(cfg, max_samples=args.max_samples)
    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    print(f"device={device} | num_samples={len(dataset)} | shared_model=True | separate_hr_head=False")
    for epoch in (1, 7, 15, 21):
        run_epoch_probe(cfg, student, teacher, batch, epoch, device)
    print("CSSD-v1a sanity passed.")


if __name__ == "__main__":
    main()
