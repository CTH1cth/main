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
    build_rast_region_masks,
    build_rast_teacher_weight_map,
    compute_dino_core_margin_68,
    get_dabe_pu_despl_schedule,
    get_esa_post_reset_scale,
    rast_teacher_bce_with_logits,
    set_model_epoch,
    weighted_bce_with_logits,
)


BASE_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_"
    "rast_v12_esa_asym_long35_lrfloor_2e5.py"
)
POST_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_dagp_uncgate_ndr_"
    "rast_v12_esa_asym_postreset35_lrfloor_2e5.py"
)


def assert_close(name, left, right, tolerance=1e-6):
    difference = float((left - right).abs().max().item())
    print(f"{name}: max_abs_diff={difference:.12g}")
    if difference > tolerance:
        raise RuntimeError(f"{name} differs by {difference}, tolerance={tolerance}.")


def compare_model_state(base_model, post_model):
    base_state = base_model.state_dict()
    post_state = post_model.state_dict()
    if set(base_state) != set(post_state):
        raise RuntimeError("Post-reset config changed the model state-dict schema.")
    max_difference = 0.0
    max_key = None
    for key, value in base_state.items():
        difference = float((value - post_state[key]).abs().max().item())
        if difference > max_difference:
            max_difference = difference
            max_key = key
    print(f"state_dict_max_abs_diff={max_difference:.12g} | key={max_key}")
    if max_difference != 0.0:
        raise RuntimeError("Post-reset config changed model initialization.")


def build_synthetic_batch():
    height = width = 68
    fg_core = torch.zeros(1, 1, height, width)
    bg_core = torch.zeros_like(fg_core)
    extent = torch.zeros_like(fg_core)
    unknown = torch.zeros_like(fg_core)
    fg_core[:, :, 2:14, 2:16] = 1.0
    bg_core[:, :, 2:14, 52:66] = 1.0
    extent[:, :, 20:56, 2:66] = 1.0
    unknown[:, :, 58:66, 24:44] = 1.0

    teacher_binary = torch.zeros_like(fg_core)
    teacher_binary[bg_core.bool()] = 1.0
    teacher_binary[:, :, 20:28, 2:66] = 1.0

    feature = torch.zeros(1, 384, 37, 37)
    fg_37 = F.interpolate(fg_core, size=(37, 37), mode="nearest").bool()
    bg_37 = F.interpolate(bg_core, size=(37, 37), mode="nearest").bool()
    feature[:, 0][fg_37[:, 0]] = 1.0
    feature[:, 0][bg_37[:, 0]] = -1.0
    feature[:, 1, :, :] = 1.0
    feature[:, 1][fg_37[:, 0] | bg_37[:, 0]] = 0.0
    feature[:, 0, :, :12] = torch.where(
        (fg_37[:, 0] | bg_37[:, 0])[:, :, :12],
        feature[:, 0, :, :12],
        torch.ones_like(feature[:, 0, :, :12]),
    )
    feature[:, 1, :, :12] = 0.0
    feature[:, 0, :, 25:] = torch.where(
        (fg_37[:, 0] | bg_37[:, 0])[:, :, 25:],
        feature[:, 0, :, 25:],
        -torch.ones_like(feature[:, 0, :, 25:]),
    )
    feature[:, 1, :, 25:] = 0.0

    batch = {
        "feature": feature,
        "pu_fg_core": fg_core,
        "pu_bg_core": bg_core,
        "pu_extent": extent,
        "pu_unknown": unknown,
    }
    return batch, teacher_binary


def compute_group_loss(cfg, epoch, logits_list, static_target, static_map, teacher_target, teacher_map, routing_scale):
    aux_weights = (1.0, 0.5, 0.5 if epoch <= 20 else 0.3)
    static_terms = [
        weighted_bce_with_logits(logits, static_target, static_map)
        for logits in logits_list
    ]
    teacher_terms = [
        rast_teacher_bce_with_logits(
            logits,
            teacher_target,
            teacher_map,
            cfg,
            routing_scale,
        )
        for logits in logits_list
    ]
    weight_sum = sum(aux_weights)
    static_group = sum(w * loss for w, loss in zip(aux_weights, static_terms)) / weight_sum
    teacher_group = sum(w * loss for w, loss in zip(aux_weights, teacher_terms)) / weight_sum
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
    return static_weight * static_group + teacher_weight * teacher_group


def assert_region_value(name, teacher_map, mask, expected):
    if not bool(mask.any().item()):
        raise RuntimeError(f"Synthetic region {name} is unexpectedly empty.")
    values = teacher_map[mask]
    max_error = float((values - expected).abs().max().item())
    print(f"{name}: pixels={int(mask.sum().item())} | mean={float(values.mean().item()):.6f}")
    if max_error > 1e-5:
        raise RuntimeError(f"{name} expected {expected}, max_error={max_error}.")


def main():
    parser = argparse.ArgumentParser(description="Test ESA-Asym PostReset35 equivalence and region weights.")
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--config", default=POST_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base_cfg = load_config(args.base_config)
    post_cfg = load_config(args.config)
    set_seed(20260711)
    base_model = build_seg_head(384, base_cfg).to(device)
    base_rng_state = torch.get_rng_state().clone()
    set_seed(20260711)
    post_model = build_seg_head(384, post_cfg).to(device)
    post_rng_state = torch.get_rng_state().clone()
    compare_model_state(base_model, post_model)
    if not torch.equal(base_rng_state, post_rng_state):
        raise RuntimeError("Post-reset config changed model-construction RNG state.")
    print("post_build_rng_state_equal=True")

    set_seed(73)
    feature = torch.randn(2, 384, 37, 37, device=device)
    image_68 = torch.rand(2, 3, 68, 68, device=device)
    output_keys = ("logits", "coarse_logits_68", "base_logits")
    for epoch in (1, 7, 15, 20):
        set_model_epoch(base_model, epoch)
        set_model_epoch(post_model, epoch)
        with torch.no_grad():
            base_output = base_model(feature, image_68=image_68, return_aux=True)
            post_output = post_model(feature, image_68=image_68, return_aux=True)
        for key in output_keys:
            assert_close(f"epoch{epoch}_{key}", base_output[key], post_output[key])

    batch, teacher_binary = build_synthetic_batch()
    batch = {key: value.to(device) for key, value in batch.items()}
    teacher_binary = teacher_binary.to(device)
    set_seed(91)
    logits_list = [torch.randn_like(teacher_binary) for _ in range(3)]
    static_target = torch.rand_like(teacher_binary)
    static_map = torch.rand_like(teacher_binary).clamp_min(0.05)
    for epoch in (1, 7, 15, 20):
        base_map, base_stats = build_rast_teacher_weight_map(
            base_cfg, batch, teacher_binary, epoch, device
        )
        post_map, post_stats = build_rast_teacher_weight_map(
            post_cfg, batch, teacher_binary, epoch, device
        )
        assert_close(f"epoch{epoch}_teacher_map", base_map, post_map)
        base_loss = compute_group_loss(
            base_cfg,
            epoch,
            logits_list,
            static_target,
            static_map,
            teacher_binary,
            base_map,
            base_stats["teacher_routing_scale"],
        )
        post_loss = compute_group_loss(
            post_cfg,
            epoch,
            logits_list,
            static_target,
            static_map,
            teacher_binary,
            post_map,
            post_stats["teacher_routing_scale"],
        )
        assert_close(f"epoch{epoch}_loss_total", base_loss, post_loss)

    if [get_esa_post_reset_scale(post_cfg, epoch) for epoch in (20, 21, 35, 36)] != [
        0.0,
        1.0,
        1.0,
        0.0,
    ]:
        raise RuntimeError("ESA post-reset schedule mismatch.")
    teacher_map, stats = build_rast_teacher_weight_map(
        post_cfg, batch, teacher_binary, 21, device
    )
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(21, post_cfg)
    if (static_weight, teacher_weight) != (0.0, 1.0):
        raise RuntimeError(f"Epoch21 is not teacher-only: {(static_weight, teacher_weight)}")
    if stats["rast_pre_reset_scale"] != 0.0 or stats["rast_post_reset_scale"] != 0.0:
        raise RuntimeError(f"RAST unexpectedly active at epoch21: {stats}")
    if stats["esa_post_reset_scale"] != 1.0 or stats["teacher_routing_scale"] != 1.0:
        raise RuntimeError(f"ESA post-reset routing is not fully active: {stats}")

    masks = build_rast_region_masks(batch, device)
    margin, _ = compute_dino_core_margin_68(
        post_cfg,
        batch,
        masks["fg_core"],
        masks["bg_core"],
        teacher_binary.shape[-2:],
        device,
        prefix="ESA",
    )
    teacher_fg = teacher_binary >= 0.5
    teacher_bg = ~teacher_fg
    extent_teacher_fg = masks["extent"] & teacher_fg
    extent_teacher_bg = masks["extent"] & teacher_bg
    fg_like = extent_teacher_bg & (margin >= 0.05)
    bg_like = extent_teacher_bg & (margin <= -0.05)
    ambig = extent_teacher_bg & (~fg_like) & (~bg_like)
    assert_region_value("fg_core", teacher_map, masks["fg_core"], 1.0)
    assert_region_value("bg_core", teacher_map, masks["bg_core"], 1.0)
    assert_region_value("unknown", teacher_map, masks["unknown"], 1.0)
    assert_region_value("extent_teacher_fg", teacher_map, extent_teacher_fg, 1.0)
    assert_region_value("extent_bg_fg_like", teacher_map, fg_like, 0.25)
    assert_region_value("extent_bg_ambig", teacher_map, ambig, 0.50)
    assert_region_value("extent_bg_bg_like", teacher_map, bg_like, 1.0)
    print(
        "ESA PostReset35: PASS | "
        f"map_min={float(teacher_map.min().item()):.6f} | "
        f"map_max={float(teacher_map.max().item()):.6f}"
    )


if __name__ == "__main__":
    main()
