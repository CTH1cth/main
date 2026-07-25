"""Minimal evidence-protect + ring ECST router for DABE-Clean."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.ecst import (
    build_asymmetric_negative_verified_weight,
    build_past_only_temporal_reliability,
)


ECST_MINIMAL_VERSION = "evidence_protect_ring_asym_v1"


def validate_ecst_minimal_config(cfg):
    enabled = bool(getattr(cfg, "USE_ECST_MINIMAL", False))
    if not enabled:
        return False
    if bool(getattr(cfg, "USE_ECST", False)):
        raise RuntimeError("Minimal ECST and full ECST cannot be enabled together.")
    if not bool(getattr(cfg, "USE_DABE_CLEAN", False)):
        raise RuntimeError("Minimal ECST requires USE_DABE_CLEAN=True.")
    if bool(getattr(cfg, "DABE_CLEAN_USE_LEGACY_ECST_REGIONS", False)):
        raise RuntimeError("Minimal ECST must not read legacy ECST regions.")
    version = str(getattr(cfg, "ECST_MINIMAL_VERSION", ""))
    if version != ECST_MINIMAL_VERSION:
        raise RuntimeError(
            f"Unsupported ECST_MINIMAL_VERSION={version!r}; "
            f"expected={ECST_MINIMAL_VERSION!r}."
        )
    radius = int(getattr(cfg, "ECST_MINIMAL_RING_RADIUS", 1))
    if radius not in {1, 2}:
        raise RuntimeError("ECST_MINIMAL_RING_RADIUS must be 1 or 2.")
    conflict_floor = float(getattr(cfg, "ECST_MINIMAL_CONFLICT_FLOOR", 0.20))
    if not 0.0 <= conflict_floor <= 1.0:
        raise RuntimeError("ECST_MINIMAL_CONFLICT_FLOOR must be in [0,1].")
    if int(getattr(cfg, "ECST_START_EPOCH", 7)) != 7:
        raise RuntimeError("Minimal ECST preserves ECST_START_EPOCH=7.")
    if int(getattr(cfg, "ECST_RAMP_END_EPOCH", 15)) != 15:
        raise RuntimeError("Minimal ECST preserves ECST_RAMP_END_EPOCH=15.")
    if int(getattr(cfg, "ECST_STOP_EPOCH", 21)) != 21:
        raise RuntimeError("Minimal ECST preserves ECST_STOP_EPOCH=21.")
    return True


def get_ecst_minimal_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_ECST_MINIMAL", False)):
        return 0.0
    start = int(getattr(cfg, "ECST_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "ECST_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "ECST_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(max(1, ramp_end - start + 1))
    return 1.0


def build_minimal_support_ring(foreground_evidence, radius):
    foreground_evidence = foreground_evidence.detach().float()
    if foreground_evidence.ndim != 4 or int(foreground_evidence.shape[1]) != 1:
        raise RuntimeError(
            "Minimal ECST foreground evidence must be [B,1,H,W], got "
            f"{list(foreground_evidence.shape)}."
        )
    radius = int(radius)
    if radius not in {1, 2}:
        raise RuntimeError("Minimal ECST ring radius must be 1 or 2.")
    support = foreground_evidence > 0.5
    dilated = support.float()
    for _ in range(radius):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
    ring = (dilated > 0.5) & (~support)
    if bool((support & ring).any().item()):
        raise RuntimeError("Minimal ECST support/ring must be mutually exclusive.")
    return support.detach(), ring.detach()


def _continuous_prototype_margin(feature, foreground, background):
    feature = feature.detach().float()
    foreground = foreground.detach().float()
    background = background.detach().float()
    if feature.ndim != 4 or tuple(feature.shape[-2:]) != tuple(
        foreground.shape[-2:]
    ):
        raise RuntimeError(
            "Minimal ECST feature/evidence shape mismatch: "
            f"{list(feature.shape)} vs {list(foreground.shape)}."
        )
    if tuple(foreground.shape) != tuple(background.shape):
        raise RuntimeError("Minimal ECST foreground/background shape mismatch.")
    if int(foreground.shape[1]) != 1:
        raise RuntimeError("Minimal ECST evidence must have one channel.")
    feature_similarity = F.normalize(feature, dim=1)
    margins = []
    fg_fallback = 0
    bg_fallback = 0
    eps = 1e-6
    for index in range(int(feature.shape[0])):
        full_mean = feature[index].mean(dim=(1, 2))

        def prototype(weight, is_fg):
            nonlocal fg_fallback, bg_fallback
            denominator = weight.sum()
            if float(denominator.detach().item()) < eps:
                if is_fg:
                    fg_fallback += 1
                else:
                    bg_fallback += 1
                mean = full_mean
            else:
                mean = (feature[index] * weight).sum(dim=(1, 2)) / denominator
            return F.normalize(mean, dim=0, eps=1e-12)

        fg_proto = prototype(foreground[index], True)
        bg_proto = prototype(background[index], False)
        sim_fg = (
            feature_similarity[index] * fg_proto.view(-1, 1, 1)
        ).sum(dim=0, keepdim=True)
        sim_bg = (
            feature_similarity[index] * bg_proto.view(-1, 1, 1)
        ).sum(dim=0, keepdim=True)
        margins.append(sim_fg - sim_bg)
    margin = torch.stack(margins, dim=0).detach()
    return margin, {
        "fg_prototype_fallback_count": fg_fallback,
        "bg_prototype_fallback_count": bg_fallback,
    }


def _masked_mean(value, mask, empty=0.0):
    if not bool(mask.any().item()):
        return float(empty)
    return float(value[mask].mean().detach().item())


def build_ecst_minimal_teacher_weight_map(
    cfg,
    batch,
    teacher_prob,
    temporal_mean,
    temporal_second,
    history_count,
    epoch,
    device,
    return_states=False,
):
    validate_ecst_minimal_config(cfg)
    teacher_prob = teacher_prob.detach().to(device=device).float()
    if teacher_prob.ndim != 4 or int(teacher_prob.shape[1]) != 1:
        raise RuntimeError(
            f"Minimal ECST teacher probability must be [B,1,H,W], got {list(teacher_prob.shape)}."
        )
    required = (
        "dabe_clean_target_37",
        "dabe_clean_fg_evidence_37",
        "dabe_clean_bg_evidence_37",
        "feature",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Minimal ECST batch is missing fields: {missing}.")
    static_target = batch["dabe_clean_target_37"].to(device).float().detach()
    foreground = batch["dabe_clean_fg_evidence_37"].to(device).float().detach()
    background = batch["dabe_clean_bg_evidence_37"].to(device).float().detach()
    feature = batch["feature"].to(device).float().detach()
    expected_37 = (int(teacher_prob.shape[0]), 1, 37, 37)
    for name, tensor in (
        ("static_target", static_target),
        ("foreground", foreground),
        ("background", background),
    ):
        if tuple(tensor.shape) != expected_37:
            raise RuntimeError(
                f"Minimal ECST {name} shape mismatch: {list(tensor.shape)} != {list(expected_37)}."
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"Minimal ECST {name} contains NaN/Inf.")
        if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
            raise RuntimeError(f"Minimal ECST {name} is outside [0,1].")
    if feature.ndim != 4 or int(feature.shape[0]) != expected_37[0] or tuple(
        feature.shape[-2:]
    ) != (37, 37):
        raise RuntimeError(f"Minimal ECST feature must be [B,C,37,37], got {list(feature.shape)}.")

    support, ring = build_minimal_support_ring(
        foreground, int(getattr(cfg, "ECST_MINIMAL_RING_RADIUS", 1))
    )
    teacher_37 = F.interpolate(
        teacher_prob,
        size=(37, 37),
        mode="bilinear",
        align_corners=False,
    ).detach()
    teacher_fg = teacher_37 >= 0.5
    teacher_bg = ~teacher_fg
    static_binary = static_target >= 0.5
    conflict = static_binary != teacher_fg
    static_strength = (2.0 * (static_target - 0.5).abs()).clamp(0.0, 1.0)
    conflict_floor = float(getattr(cfg, "ECST_MINIMAL_CONFLICT_FLOOR", 0.20))
    protect_weight = (
        1.0 - (1.0 - conflict_floor) * static_strength
    ).clamp(conflict_floor, 1.0)

    margin_37, prototype_stats = _continuous_prototype_margin(
        feature, foreground, background
    )
    temporal_mean_37 = F.interpolate(
        temporal_mean.detach().to(device).float(),
        size=(37, 37),
        mode="bilinear",
        align_corners=False,
    )
    temporal_second_37 = F.interpolate(
        temporal_second.detach().to(device).float(),
        size=(37, 37),
        mode="bilinear",
        align_corners=False,
    )
    temporal = build_past_only_temporal_reliability(
        cfg, temporal_mean_37, temporal_second_37, history_count.to(device)
    )
    negative = build_asymmetric_negative_verified_weight(
        cfg, margin_37, temporal["bg_reliability"]
    )
    negative_weight = negative["negative_verified_weight"]

    raw_37 = torch.ones_like(static_target)
    raw_37 = torch.where(conflict, protect_weight, raw_37)
    raw_37 = torch.where(ring & teacher_fg, torch.ones_like(raw_37), raw_37)
    raw_37 = torch.where(ring & teacher_bg, negative_weight, raw_37)
    raw_68 = F.interpolate(
        raw_37,
        size=teacher_prob.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    weight_min = float(getattr(cfg, "ECST_WEIGHT_MIN", 0.20))
    raw_68 = raw_68.clamp(weight_min, 1.0)
    scale = get_ecst_minimal_scale(cfg, epoch)
    effective = ((1.0 - scale) + scale * raw_68).clamp(weight_min, 1.0).detach()
    if effective.requires_grad or not bool(torch.isfinite(effective).all().item()):
        raise RuntimeError("Minimal ECST output must be finite and detached.")

    ring_fg = ring & teacher_fg
    ring_bg = ring & teacher_bg
    stats = {
        "ecst_scale": float(scale),
        "support_area": float(support.float().mean().item()),
        "ring_area": float(ring.float().mean().item()),
        "static_strength_mean": float(static_strength.mean().item()),
        "conflict_ratio": float(conflict.float().mean().item()),
        "ring_teacher_fg_ratio": float(ring_fg.float().mean().item()),
        "ring_teacher_bg_ratio": float(ring_bg.float().mean().item()),
        "protect_weight_mean": _masked_mean(protect_weight, conflict, empty=1.0),
        "negative_verified_weight_mean": _masked_mean(
            negative_weight, ring_bg, empty=1.0
        ),
        "teacher_map_min": float(effective.min().item()),
        "teacher_map_mean": float(effective.mean().item()),
        "teacher_map_max": float(effective.max().item()),
        "history_count_min": int(history_count.min().item()),
        "history_count_mean": float(history_count.float().mean().item()),
        "history_count_max": int(history_count.max().item()),
        "history_valid_ratio": float(temporal["history_valid"].float().mean().item()),
        "memory_active": True,
        **prototype_stats,
    }
    states = {
        "support": support,
        "ring": ring,
        "teacher_fg_37": teacher_fg.detach(),
        "teacher_bg_37": teacher_bg.detach(),
        "static_binary": static_binary.detach(),
        "conflict": conflict.detach(),
        "static_strength": static_strength.detach(),
        "protect_weight": protect_weight.detach(),
        "margin_37": margin_37.detach(),
        "negative_verified_weight": negative_weight.detach(),
        "raw_37": raw_37.detach(),
        "raw_68": raw_68.detach(),
        "temporal": temporal,
    }
    if return_states:
        return effective, stats, states
    return effective, stats
