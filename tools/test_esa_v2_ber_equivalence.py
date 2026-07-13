import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import load_config, set_seed  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_esa_ber_loss,
    build_rast_teacher_weight_map,
    get_dabe_pu_despl_schedule,
    is_before_finetune_reset,
    rast_teacher_bce_with_logits,
    set_model_epoch,
    weighted_bce_with_logits,
)


BASE_CONFIG = "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long35_lrfloor_2e5.py"
BER_CONFIG = "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_v2_ber_long35_lrfloor_2e5.py"


def assert_close(name, left, right, tolerance=1e-6):
    difference = float((left - right).abs().max().item())
    print(f"{name}: max_abs_diff={difference:.12g}")
    if difference > tolerance:
        raise RuntimeError(f"{name} differs by {difference}, tolerance={tolerance}.")


def synthetic_batch(batch_size=2):
    feature = torch.randn(batch_size, 384, 37, 37)
    fg_core = torch.zeros(batch_size, 1, 68, 68)
    bg_core = torch.zeros_like(fg_core)
    extent = torch.zeros_like(fg_core)
    unknown = torch.zeros_like(fg_core)
    fg_core[:, :, 8:22, 8:22] = 1.0
    bg_core[:, :, 46:62, 46:62] = 1.0
    extent[:, :, 4:64, 4:64] = 1.0
    unknown[:, :, 30:38, 2:8] = 1.0
    return {
        "feature": feature,
        "pu_fg_core": fg_core,
        "pu_bg_core": bg_core,
        "pu_extent": extent,
        "pu_unknown": unknown,
        "pu_target_soft": torch.rand_like(fg_core),
        "pu_weight_map": torch.rand_like(fg_core),
        "dataset": ["TR-CAMO", "TR-COD10K"],
        "stem": ["synthetic_camo", "synthetic_cod10k"],
    }


def original_group_loss(cfg, epoch, output, static_target, static_map, teacher_binary, teacher_map, routing_scale):
    final_logits = output["logits"]
    coarse_logits = output["coarse_logits_68"]
    base_logits = F.interpolate(output["base_logits"], size=(68, 68), mode="bilinear", align_corners=False)
    base_lambda = (
        float(getattr(cfg, "LAMBDA_BASE_AUX", 0.5))
        if is_before_finetune_reset(cfg, epoch)
        else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.3))
    )
    weights = (1.0, float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)), base_lambda)
    logits_list = (final_logits, coarse_logits, base_logits)
    static_terms = [weighted_bce_with_logits(x, static_target, static_map) for x in logits_list]
    teacher_terms = [
        rast_teacher_bce_with_logits(x, teacher_binary, teacher_map, cfg, routing_scale)
        for x in logits_list
    ]
    denom = sum(weights)
    static_group = sum(w * x for w, x in zip(weights, static_terms)) / denom
    teacher_group = sum(w * x for w, x in zip(weights, teacher_terms)) / denom
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
    return static_weight * static_group + teacher_weight * teacher_group


def main():
    parser = argparse.ArgumentParser(description="Verify ESA-v2-BER pre-reset numerical equivalence.")
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--config", default=BER_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    base_cfg = load_config(args.base_config)
    ber_cfg = load_config(args.config)

    set_seed(20260711)
    base_model = build_seg_head(384, base_cfg).to(device)
    base_rng = torch.get_rng_state().clone()
    set_seed(20260711)
    ber_model = build_seg_head(384, ber_cfg).to(device)
    ber_rng = torch.get_rng_state().clone()
    if set(base_model.state_dict()) != set(ber_model.state_dict()):
        raise RuntimeError("ESA-v2-BER changed model state_dict keys.")
    if sum(p.numel() for p in base_model.parameters()) != sum(p.numel() for p in ber_model.parameters()):
        raise RuntimeError("ESA-v2-BER changed model parameter count.")
    for key, value in base_model.state_dict().items():
        if not torch.equal(value, ber_model.state_dict()[key]):
            raise RuntimeError(f"ESA-v2-BER changed initialized state at {key}.")
    if not torch.equal(base_rng, ber_rng):
        raise RuntimeError("ESA-v2-BER changed model-construction RNG state.")
    print(f"state_dict_keys_equal=True | parameter_count={sum(p.numel() for p in base_model.parameters())}")
    print("post_build_rng_state_equal=True")

    set_seed(19)
    batch = synthetic_batch()
    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    image_68 = torch.rand(2, 3, 68, 68, device=device)
    static_target = batch["pu_target_soft"]
    static_map = batch["pu_weight_map"]
    teacher_binary = (torch.rand_like(static_target) >= 0.5).float()

    for epoch in (1, 7, 15, 20):
        set_model_epoch(base_model, epoch)
        set_model_epoch(ber_model, epoch)
        base_output = base_model(batch["feature"], image_68=image_68, return_aux=True)
        ber_output = ber_model(batch["feature"], image_68=image_68, return_aux=True)
        for key in ("logits", "coarse_logits_68", "base_logits"):
            assert_close(f"epoch{epoch}_{key}", base_output[key], ber_output[key])
        if "dagp_topk_idx" in ber_output:
            raise RuntimeError(f"ESA-v2-BER returned graph auxiliary before activation at epoch {epoch}.")

        base_map, base_stats = build_rast_teacher_weight_map(
            base_cfg, batch, teacher_binary, epoch, device
        )
        ber_map, ber_stats = build_rast_teacher_weight_map(
            ber_cfg, batch, teacher_binary, epoch, device
        )
        assert_close(f"epoch{epoch}_teacher_map", base_map, ber_map)
        base_loss = original_group_loss(
            base_cfg,
            epoch,
            base_output,
            static_target,
            static_map,
            teacher_binary,
            base_map,
            base_stats["teacher_routing_scale"],
        )
        ber_main_loss = original_group_loss(
            ber_cfg,
            epoch,
            ber_output,
            static_target,
            static_map,
            teacher_binary,
            ber_map,
            ber_stats["teacher_routing_scale"],
        )
        teacher_prob = teacher_binary.detach()
        ber_weighted, ber_loss_stats, _ = build_esa_ber_loss(
            ber_cfg,
            epoch,
            ber_output,
            ber_output["logits"],
            batch,
            teacher_prob,
            teacher_binary,
            ber_map,
            ber_stats,
            device,
        )
        if float(ber_weighted.detach().item()) != 0.0 or ber_loss_stats["ber_scale"] != 0.0:
            raise RuntimeError(f"ESA-v2-BER must be a strict no-op at epoch {epoch}.")
        assert_close(f"epoch{epoch}_loss_total", base_loss, ber_main_loss + ber_weighted)

    print("ESA-v2-BER pre-reset equivalence: PASS")


if __name__ == "__main__":
    main()

