import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import check_dabe_pu_cache, load_config, set_seed  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_esa_ber_loss,
    build_rast_teacher_weight_map,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    rast_teacher_bce_with_logits,
    resize_logits_for_loss,
    set_model_epoch,
    weighted_bce_with_logits,
)


DEFAULT_CONFIG = "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_v2_ber_long35_lrfloor_2e5.py"


def gradients_finite(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)


def main():
    parser = argparse.ArgumentParser(description="Run ESA-v2-BER epoch20/21/23 sanity checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    cfg = load_config(args.config)
    check_dabe_pu_cache(cfg, max_samples=args.max_samples)
    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=min(int(cfg.BATCH_SIZE), len(dataset)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    if "gt" in batch:
        raise RuntimeError("ESA-v2-BER sanity unexpectedly received training GT.")

    set_seed(20260711)
    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    target_soft = batch["pu_target_soft"].to(device).float()
    weight_map = batch["pu_weight_map"].to(device).float()

    for epoch in (20, 21, 23):
        student.zero_grad(set_to_none=True)
        set_model_epoch(student, epoch)
        set_model_epoch(teacher, epoch)
        student.train()
        output = student(model_input, image_68=image_68, return_aux=True)
        final_logits = resize_logits_for_loss(output["logits"], cfg)
        with torch.no_grad():
            teacher_output = teacher(model_input, image_68=image_68, return_aux=False)
            teacher_logits = resize_logits_for_loss(
                teacher_output["logits"] if isinstance(teacher_output, dict) else teacher_output,
                cfg,
            )
            teacher_prob = teacher_logits.sigmoid().detach()
            teacher_binary = (teacher_prob >= 0.5).float().detach()
        teacher_map, rast_stats = build_rast_teacher_weight_map(
            cfg, batch, teacher_binary, epoch, device
        )
        static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
        loss_static = weighted_bce_with_logits(final_logits, target_soft, weight_map)
        loss_teacher = rast_teacher_bce_with_logits(
            final_logits,
            teacher_binary,
            teacher_map,
            cfg,
            rast_stats["teacher_routing_scale"],
        )
        loss_main = static_weight * loss_static + teacher_weight * loss_teacher
        loss_ber, stats, aux = build_esa_ber_loss(
            cfg,
            epoch,
            output,
            final_logits,
            batch,
            teacher_prob,
            teacher_binary,
            teacher_map,
            rast_stats,
            device,
        )
        loss_total = loss_main + loss_ber
        if not bool(torch.isfinite(loss_total).item()):
            raise RuntimeError(f"ESA-v2-BER sanity loss is not finite at epoch {epoch}.")

        if epoch == 20:
            if stats["ber_scale"] != 0.0 or float(loss_ber.detach().item()) != 0.0:
                raise RuntimeError("ESA-v2-BER must be disabled at epoch20.")
            if "dagp_topk_idx" in output:
                raise RuntimeError("ESA-v2-BER graph aux must not be returned at epoch20.")
        else:
            expected_scale = 1.0 / 3.0 if epoch == 21 else 1.0
            if abs(stats["ber_scale"] - expected_scale) > 1e-8:
                raise RuntimeError(f"ESA-v2-BER scale mismatch at epoch{epoch}.")
            if abs(static_weight) > 1e-8 or abs(teacher_weight - 1.0) > 1e-8:
                raise RuntimeError(f"ESA-v2-BER active epoch{epoch} is not teacher-only.")
            if float((teacher_map - 1.0).abs().max().item()) > 1e-5:
                raise RuntimeError(f"ESA-v2-BER epoch{epoch} teacher map is not all one.")
            if output["dagp_topk_idx"].shape != output["dagp_topk_sem_weight"].shape:
                raise RuntimeError("ESA-v2-BER graph aux shape mismatch.")
            if float((output["dagp_topk_sem_weight"].sum(-1) - 1.0).abs().max().item()) > 1e-5:
                raise RuntimeError("ESA-v2-BER graph weights are not normalized.")
            pos_count = aux["selected_pos"].flatten(1).sum(1)
            neg_count = aux["selected_neg"].flatten(1).sum(1)
            if not torch.equal(pos_count, neg_count):
                raise RuntimeError("ESA-v2-BER selected candidates are not balanced.")
            if aux["selected_pos"].requires_grad or aux["selected_neg"].requires_grad:
                raise RuntimeError("ESA-v2-BER candidate masks require gradients.")

        loss_total.backward()
        if not gradients_finite(student):
            raise RuntimeError(f"ESA-v2-BER student gradients are invalid at epoch {epoch}.")
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("ESA-v2-BER propagated gradients into the EMA teacher.")
        print(
            f"epoch={epoch} | static/teacher={static_weight:.6f}/{teacher_weight:.6f} | "
            f"ber_scale={stats['ber_scale']:.6f} | valid={stats['valid_image_ratio']:.6f} | "
            f"pairs={stats['selected_pairs_mean']:.4f} | loss_main={float(loss_main.detach().item()):.8f} | "
            f"loss_ber={float(loss_ber.detach().item()):.8f}"
        )
    print("ESA-v2-BER sanity: PASS")


if __name__ == "__main__":
    main()

