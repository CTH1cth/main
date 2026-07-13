import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import check_dabe_pu_cache, load_config  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_csd_aux_losses,
    build_csd_bg_reliable_mask,
    build_dabe_pu_despl_static_target,
    build_pa_dagp_aux_loss,
    build_rast_teacher_weight_map,
    compute_cssd_original_group_loss,
    extract_logits,
    forward_seg_head,
    get_pa_dagp_aux_scale,
    get_pa_dagp_edge_scale,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


CONTROL_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_"
    "rast_v12_noesa_clean35_lrfloor_2e5.py"
)


def load_shared_pa_state_into_control(pa_model, control_model):
    pa_state = pa_model.state_dict()
    control_state = control_model.state_dict()
    for key in control_state:
        if key not in pa_state or pa_state[key].shape != control_state[key].shape:
            raise RuntimeError(f"Shared PA/control state mismatch at {key}")
        control_state[key] = pa_state[key].detach().clone()
    control_model.load_state_dict(control_state, strict=True)


def finite_gradients(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def run_epoch_probe(cfg, student, teacher, control, batch, epoch, device):
    set_model_epoch(student, epoch)
    set_model_epoch(teacher, epoch)
    set_model_epoch(control, epoch)
    student.train()
    teacher.eval()
    control.train()
    student.zero_grad(set_to_none=True)
    if "gt" in batch:
        raise RuntimeError("PA-DAGP train sanity unexpectedly received training GT.")

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    static_target, static_weight, _ = build_dabe_pu_despl_static_target(
        cfg, target_soft, weight_map
    )
    bg_reliable = build_csd_bg_reliable_mask(
        batch, cfg, (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)), device
    ).float()

    student_out = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
        bg_reliable_68=bg_reliable,
        pa_compare_original=True,
    )
    with torch.no_grad():
        teacher_out = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
        )
        if isinstance(teacher_out, dict) and "pa_raw_polarity" in teacher_out:
            raise RuntimeError("PA-DAGP teacher forward unexpectedly returned auxiliary polarity tensors.")
        teacher_logits = resize_logits_for_loss(extract_logits(teacher_out), cfg)
        teacher_binary = (teacher_logits.sigmoid() >= 0.5).float()
    teacher_map, rast_stats = build_rast_teacher_weight_map(
        cfg, batch, teacher_binary, epoch, device
    )
    group = compute_cssd_original_group_loss(
        student_out,
        static_target,
        static_weight,
        teacher_binary,
        teacher_map,
        rast_stats["rast_scale"],
        epoch,
        cfg,
    )
    loss_csd_bg, loss_csd_boundary, _ = build_csd_aux_losses(
        student_out, batch, cfg, epoch, device
    )
    loss_current = group["loss_group"] + loss_csd_bg + loss_csd_boundary
    loss_pa, pa_stats = build_pa_dagp_aux_loss(cfg, epoch, student_out, batch, device)
    loss_total = loss_current + loss_pa
    if not bool(torch.isfinite(loss_total).item()):
        raise RuntimeError(f"PA-DAGP sanity loss is not finite at epoch {epoch}.")

    edge_scale = float(get_pa_dagp_edge_scale(epoch, cfg))
    aux_scale = float(get_pa_dagp_aux_scale(epoch, cfg))
    expected = {
        1: (0.0, 0.0),
        7: (1.0 / 9.0, 1.0 / 9.0),
        15: (1.0, 1.0),
        21: (1.0, 0.0),
    }[epoch]
    if abs(edge_scale - expected[0]) > 1e-12 or abs(aux_scale - expected[1]) > 1e-12:
        raise RuntimeError(
            f"PA-DAGP schedule mismatch at epoch {epoch}: edge={edge_scale}, aux={aux_scale}"
        )
    if epoch in (1, 21) and float(loss_pa.detach().item()) != 0.0:
        raise RuntimeError(f"PA-DAGP weighted auxiliary must be zero at epoch {epoch}.")
    if epoch in (7, 15) and float(loss_pa.detach().item()) <= 0.0:
        raise RuntimeError(f"PA-DAGP weighted auxiliary must be positive at epoch {epoch}.")
    if pa_stats["gate_min"] < 0.5 - 1e-5 or pa_stats["gate_max"] > 1.0 + 1e-5:
        raise RuntimeError(f"PA-DAGP gate range failed at epoch {epoch}: {pa_stats}")
    if pa_stats.get("edge_normalization_max_error", 0.0) > 1e-5:
        raise RuntimeError(f"PA-DAGP edge normalization failed at epoch {epoch}: {pa_stats}")

    if epoch == 1:
        with torch.no_grad():
            control_out = control(
                model_input,
                image_68=image_68,
                return_aux=True,
                bg_reliable_68=bg_reliable,
            )
        for key in ("base_logits", "coarse_logits_native", "coarse_logits_68", "final_logits"):
            difference = float((student_out[key] - control_out[key]).abs().max().item())
            if difference > 1e-6:
                raise RuntimeError(f"epoch1 PA/control equivalence failed for {key}: {difference}")
        if float(torch.abs(loss_total.detach() - loss_current.detach()).item()) != 0.0:
            raise RuntimeError("epoch1 PA total loss differs from the current control loss.")

    loss_total.backward()
    if not finite_gradients(student):
        raise RuntimeError(f"PA-DAGP sanity gradients contain NaN/Inf at epoch {epoch}.")
    print(
        f"epoch={epoch} | edge_scale={edge_scale:.8f} | aux_scale={aux_scale:.8f} | "
        f"feature={list(model_input.shape)} | raw_pol={list(student_out['pa_raw_polarity'].shape)} | "
        f"final={list(student_out['final_logits'].shape)} | gate={pa_stats['gate_min']:.6f}..{pa_stats['gate_max']:.6f} | "
        f"anchor_valid={pa_stats['anchor_valid_ratio']:.6f} | core_gap={pa_stats['core_gap']:.6f} | "
        f"loss_current={float(loss_current.detach().item()):.8f} | "
        f"loss_pa={float(loss_pa.detach().item()):.8f} | loss_total={float(loss_total.detach().item()):.8f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Real-cache PA-DAGP-v1 epoch schedule/loss sanity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    cfg = load_config(args.config)
    check_dabe_pu_cache(cfg, max_samples=args.max_samples)
    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    control_cfg = load_config(CONTROL_CONFIG)
    control = build_seg_head(dataset.in_channels, control_cfg).to(device)
    load_shared_pa_state_into_control(student, control)
    print(
        f"device={device} | samples={len(dataset)} | student_teacher_state_equal=True | "
        "train_gt_used=False"
    )
    for epoch in (1, 7, 15, 21):
        run_epoch_probe(cfg, student, teacher, control, batch, epoch, device)
    print("PA-DAGP-v1 real-cache sanity: PASS")


if __name__ == "__main__":
    main()
