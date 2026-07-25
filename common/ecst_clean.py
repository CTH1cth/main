"""Clean-DP-native evidence-conditioned EMA-Teacher routing.

This router is deliberately independent from the PU-v1.1 region protocol.  It
consumes only the continuous Clean cache, cached DINO features and past-only
EMA-Teacher moments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


ECST_CLEAN_V1_VERSION = "v1_signed_conf_ring_asym"
ECST_CLEAN_V2_VERSION = "v2_early_r2_strength15"
ECST_CLEAN_V3_VERSION = "v3_instant_full_r2_strength15_ablation"
ECST_CLEAN_V4_VERSION = "v4_directional_asymmetric_strength"
ECST_CLEAN_V5_VERSION = "v5_asym_continuous_recoverability"
# Backward-compatible public alias used by the existing v1 tests/config.
ECST_CLEAN_VERSION = ECST_CLEAN_V1_VERSION
ECST_CLEAN_VERSION_CONTRACTS = {
    ECST_CLEAN_V1_VERSION: {
        "start_epoch": 7,
        "ramp_end_epoch": 15,
        "ring_radius": 1,
        "strength_mode": "uniform",
        "strength_max": 1.0,
    },
    ECST_CLEAN_V2_VERSION: {
        "start_epoch": 5,
        "ramp_end_epoch": 10,
        "ring_radius": 2,
        "strength_mode": "uniform",
        "strength_max": 1.5,
    },
    ECST_CLEAN_V3_VERSION: {
        "start_epoch": 2,
        "ramp_end_epoch": 2,
        "ring_radius": 2,
        "strength_mode": "uniform",
        "strength_max": 1.5,
    },
    ECST_CLEAN_V4_VERSION: {
        "start_epoch": 5,
        "ramp_end_epoch": 10,
        "ring_radius": 2,
        "strength_mode": "directional",
        "erase_strength": 2.5,
        "add_strength": 1.0,
        "ring_bg_strength": 2.5,
    },
    ECST_CLEAN_V5_VERSION: {
        "start_epoch": 5,
        "ramp_end_epoch": 10,
        "strength_mode": "directional_continuous",
        "erase_strength": 2.5,
        "add_strength": 1.0,
        "recovery_strength": 2.5,
        "use_hard_ring": False,
        "recovery_mode": "latent_rw_geometric_mean",
    },
}
LEGACY_REGION_KEYS = {
    "legacy_ecst_fg_core",
    "legacy_ecst_fg_fallback",
    "legacy_ecst_bg_core",
    "legacy_ecst_extent",
    "legacy_ecst_unknown",
    "fg_core_pu_68",
    "fg_core_fallback_68",
    "bg_core_pu_68",
    "extent_candidate_68",
    "unknown_68",
}


def _finite_float(cfg, name):
    value = float(getattr(cfg, name))
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got {value}.")
    return value


def _require_close(cfg, name, expected, atol=1e-12):
    value = _finite_float(cfg, name)
    if abs(value - float(expected)) > float(atol):
        raise RuntimeError(f"{name} must equal {expected}, got {value}.")
    return value


def validate_ecst_clean_config(cfg):
    """Validate an isolated, explicitly versioned Clean-ECST protocol."""

    enabled = bool(getattr(cfg, "USE_ECST_CLEAN", False))
    if not enabled:
        return False
    if str(getattr(cfg, "TEACHER_ROUTING_MODE", "")).strip().lower() != "clean_ecst":
        raise RuntimeError(
            "USE_ECST_CLEAN=True requires TEACHER_ROUTING_MODE='clean_ecst'."
        )
    if bool(getattr(cfg, "USE_ECST", False)) or bool(
        getattr(cfg, "USE_ECST_MINIMAL", False)
    ):
        raise RuntimeError("Clean-ECST cannot enable legacy full/minimal ECST.")
    if not bool(getattr(cfg, "USE_DABE_CLEAN", False)):
        raise RuntimeError("Clean-ECST requires USE_DABE_CLEAN=True.")
    if str(getattr(cfg, "DABE_CLEAN_TARGET_MODE", "")).strip().lower() != "dp":
        raise RuntimeError("Clean-ECST requires DABE_CLEAN_TARGET_MODE='dp'.")
    if bool(getattr(cfg, "USE_DABE_PU", False)):
        raise RuntimeError("Clean-ECST must not enable or load DABE-PU.")
    if bool(getattr(cfg, "DABE_CLEAN_USE_LEGACY_ECST_REGIONS", False)) or bool(
        getattr(cfg, "DABE_CLEAN_LEGACY_ROUTING_ONLY", False)
    ):
        raise RuntimeError("Clean-ECST must not load legacy ECST regions.")
    if str(getattr(cfg, "DABE_CLEAN_LEGACY_REGION_ROOT", "")).strip():
        raise RuntimeError("Clean-ECST requires DABE_CLEAN_LEGACY_REGION_ROOT=''.")
    if str(getattr(cfg, "DABE_PU_ROOT", "")).strip():
        raise RuntimeError("Clean-ECST requires the inactive DABE_PU_ROOT to be empty.")
    version = str(getattr(cfg, "ECST_CLEAN_VERSION", ""))
    if version not in ECST_CLEAN_VERSION_CONTRACTS:
        raise RuntimeError(
            f"Unsupported ECST_CLEAN_VERSION={version!r}; expected one of "
            f"{sorted(ECST_CLEAN_VERSION_CONTRACTS)}."
        )
    version_contract = ECST_CLEAN_VERSION_CONTRACTS[version]

    integer_contract = {
        "ECST_CLEAN_START_EPOCH": version_contract["start_epoch"],
        "ECST_CLEAN_RAMP_END_EPOCH": version_contract["ramp_end_epoch"],
        "ECST_CLEAN_STOP_EPOCH": 21,
        "ECST_CLEAN_MEMORY_UPDATE_START_EPOCH": 1,
        "ECST_CLEAN_MEMORY_UPDATE_END_EPOCH": 20,
        "ECST_CLEAN_MIN_HISTORY": 3,
        "MAX_EPOCH": 45,
        "STOP_AFTER_EPOCH": 0,
        "SAVE_INTERVAL": 1,
    }
    if bool(version_contract.get("use_hard_ring", True)):
        integer_contract["ECST_CLEAN_RING_RADIUS"] = version_contract[
            "ring_radius"
        ]
    elif bool(getattr(cfg, "ECST_CLEAN_USE_HARD_RING", True)):
        raise RuntimeError(f"{version} requires ECST_CLEAN_USE_HARD_RING=False.")
    for name, expected in integer_contract.items():
        value = int(getattr(cfg, name, -1))
        if value != expected:
            raise RuntimeError(f"{name} must equal {expected}, got {value}.")
    float_contract = {
        "ECST_CLEAN_TEMPORAL_RHO": 0.90,
        "ECST_CLEAN_VARIANCE_TAU": 0.02,
        "ECST_CLEAN_CONFLICT_FLOOR": 0.20,
        "ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR": 0.25,
        "ECST_CLEAN_MARGIN_TAU": 0.05,
        "ECST_CLEAN_WEIGHT_MIN": 0.20,
        "ECST_CLEAN_WEIGHT_MAX": 1.00,
    }
    for name, expected in float_contract.items():
        _require_close(cfg, name, expected)
    strength_mode = str(
        getattr(cfg, "ECST_CLEAN_STRENGTH_MODE", "uniform")
    ).strip().lower()
    if strength_mode != version_contract["strength_mode"]:
        raise RuntimeError(
            f"ECST_CLEAN_STRENGTH_MODE must equal "
            f"{version_contract['strength_mode']!r} for {version}, got "
            f"{strength_mode!r}."
        )
    if strength_mode == "uniform":
        strength_max = float(getattr(cfg, "ECST_CLEAN_STRENGTH_MAX", 1.0))
        if not math.isfinite(strength_max) or abs(
            strength_max - float(version_contract["strength_max"])
        ) > 1e-12:
            raise RuntimeError(
                "ECST_CLEAN_STRENGTH_MAX must equal "
                f"{version_contract['strength_max']} for {version}, got "
                f"{strength_max}."
            )
    elif strength_mode == "directional":
        if hasattr(cfg, "ECST_CLEAN_STRENGTH_MAX"):
            raise RuntimeError(
                f"{version} must not define the legacy unified "
                "ECST_CLEAN_STRENGTH_MAX field."
            )
        for name, contract_key in (
            ("ECST_CLEAN_ERASE_STRENGTH", "erase_strength"),
            ("ECST_CLEAN_ADD_STRENGTH", "add_strength"),
            ("ECST_CLEAN_RING_BG_STRENGTH", "ring_bg_strength"),
        ):
            value = _finite_float(cfg, name)
            expected = float(version_contract[contract_key])
            if value <= 0.0 or abs(value - expected) > 1e-12:
                raise RuntimeError(
                    f"{name} must equal {expected} for {version}, got {value}."
                )
    else:
        if hasattr(cfg, "ECST_CLEAN_STRENGTH_MAX"):
            raise RuntimeError(
                f"{version} must not define ECST_CLEAN_STRENGTH_MAX."
            )
        if hasattr(cfg, "ECST_CLEAN_RING_BG_STRENGTH"):
            raise RuntimeError(
                f"{version} must not define ECST_CLEAN_RING_BG_STRENGTH."
            )
        for name, contract_key in (
            ("ECST_CLEAN_ERASE_STRENGTH", "erase_strength"),
            ("ECST_CLEAN_ADD_STRENGTH", "add_strength"),
            ("ECST_CLEAN_RECOVERY_STRENGTH", "recovery_strength"),
        ):
            value = _finite_float(cfg, name)
            expected = float(version_contract[contract_key])
            if value <= 0.0 or abs(value - expected) > 1e-12:
                raise RuntimeError(
                    f"{name} must equal {expected} for {version}, got {value}."
                )
        recovery_mode = str(
            getattr(cfg, "ECST_CLEAN_RECOVERY_MODE", "")
        ).strip().lower()
        if recovery_mode != version_contract["recovery_mode"]:
            raise RuntimeError(
                f"ECST_CLEAN_RECOVERY_MODE must equal "
                f"{version_contract['recovery_mode']!r}, got {recovery_mode!r}."
            )
    if str(getattr(cfg, "ECST_CLEAN_MEMORY_DTYPE", "")).lower() != "float16":
        raise RuntimeError("ECST_CLEAN_MEMORY_DTYPE must equal 'float16'.")
    for name in (
        "ECST_CLEAN_USE_PREUPDATE_STATS",
        "ECST_CLEAN_RESET_MEMORY_AT_FINETUNE_RESET",
        "ECST_CLEAN_APPLY_TO_FINAL",
        "ECST_CLEAN_APPLY_TO_COARSE_AUX",
        "ECST_CLEAN_APPLY_TO_BASE_AUX",
        "SAVE_EVERY_EPOCH",
    ):
        if not bool(getattr(cfg, name, False)):
            raise RuntimeError(f"Clean-ECST requires {name}=True.")
    if str(getattr(cfg, "DABE_CLEAN_STATIC_WEIGHT_MODE", "")).lower() != "ones":
        raise RuntimeError("Clean-ECST must retain unweighted Clean static BCE.")
    return True


def get_ecst_clean_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_ECST_CLEAN", False)):
        return 0.0
    start = int(getattr(cfg, "ECST_CLEAN_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "ECST_CLEAN_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "ECST_CLEAN_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(max(1, ramp_end - start + 1))
    return 1.0


def get_ecst_clean_strength_max(cfg):
    strength = float(getattr(cfg, "ECST_CLEAN_STRENGTH_MAX", 1.0))
    if not math.isfinite(strength) or strength <= 0.0:
        raise RuntimeError(
            f"ECST_CLEAN_STRENGTH_MAX must be finite and positive, got {strength}."
        )
    return strength


def get_ecst_clean_strength_mode(cfg):
    mode = str(
        getattr(cfg, "ECST_CLEAN_STRENGTH_MODE", "uniform")
    ).strip().lower()
    if mode not in {"uniform", "directional", "directional_continuous"}:
        raise RuntimeError(
            f"Unsupported ECST_CLEAN_STRENGTH_MODE={mode!r}."
        )
    return mode


def get_ecst_clean_directional_strengths(cfg):
    if get_ecst_clean_strength_mode(cfg) != "directional":
        raise RuntimeError(
            "Directional strengths require ECST_CLEAN_STRENGTH_MODE='directional'."
        )
    strengths = {
        "erase": float(getattr(cfg, "ECST_CLEAN_ERASE_STRENGTH")),
        "add": float(getattr(cfg, "ECST_CLEAN_ADD_STRENGTH")),
        "ring_bg": float(getattr(cfg, "ECST_CLEAN_RING_BG_STRENGTH")),
    }
    for name, value in strengths.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"Clean-ECST {name} strength must be finite and positive, got {value}."
            )
    return strengths


def get_ecst_clean_continuous_strengths(cfg):
    if get_ecst_clean_strength_mode(cfg) != "directional_continuous":
        raise RuntimeError(
            "Continuous strengths require "
            "ECST_CLEAN_STRENGTH_MODE='directional_continuous'."
        )
    strengths = {
        "erase": float(getattr(cfg, "ECST_CLEAN_ERASE_STRENGTH")),
        "add": float(getattr(cfg, "ECST_CLEAN_ADD_STRENGTH")),
        "recovery": float(getattr(cfg, "ECST_CLEAN_RECOVERY_STRENGTH")),
    }
    for name, value in strengths.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"Clean-ECST {name} strength must be finite and positive, got {value}."
            )
    return strengths


def apply_ecst_clean_strength(
    raw_map,
    schedule_scale,
    strength_max,
    weight_min=0.20,
    weight_max=1.00,
):
    """Scale only the raw routing-map displacement from the identity map."""

    raw = raw_map.detach().float()
    if raw.ndim != 4 or int(raw.shape[1]) != 1:
        raise RuntimeError(
            f"Clean-ECST raw map must be [B,1,H,W], got {list(raw.shape)}."
        )
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError("Clean-ECST raw map contains NaN/Inf.")
    scale = float(schedule_scale)
    strength = float(strength_max)
    minimum = float(weight_min)
    maximum = float(weight_max)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise RuntimeError(f"Clean-ECST schedule scale must be in [0,1], got {scale}.")
    if not math.isfinite(strength) or strength <= 0.0:
        raise RuntimeError(
            f"Clean-ECST strength_max must be finite and positive, got {strength}."
        )
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise RuntimeError(
            f"Invalid Clean-ECST map range [{minimum},{maximum}]."
        )
    return (
        1.0 + scale * strength * (raw - 1.0)
    ).clamp(minimum, maximum).detach()


def apply_directional_strength(
    raw_weight,
    schedule_scale,
    direction_strength,
    weight_min,
):
    """Apply one directional strength without changing its raw weight."""

    return apply_ecst_clean_strength(
        raw_weight,
        schedule_scale=schedule_scale,
        strength_max=direction_strength,
        weight_min=weight_min,
        weight_max=1.0,
    )


def compose_ecst_clean_directional_map(
    conflict_weight,
    negative_weight,
    erase_conflict_effective,
    add_conflict_effective,
    ring_teacher_bg,
    ring_teacher_fg,
    schedule_scale,
    erase_strength,
    add_strength,
    ring_bg_strength,
    weight_min,
):
    """Compose mutually exclusive directional maps in the prescribed order."""

    conflict = conflict_weight.detach().float()
    negative = negative_weight.detach().float()
    expected_shape = tuple(conflict.shape)
    if tuple(negative.shape) != expected_shape:
        raise RuntimeError("Clean-ECST directional raw-weight shape mismatch.")
    masks = {
        "erase_conflict_effective": erase_conflict_effective.detach().bool(),
        "add_conflict_effective": add_conflict_effective.detach().bool(),
        "ring_teacher_bg": ring_teacher_bg.detach().bool(),
        "ring_teacher_fg": ring_teacher_fg.detach().bool(),
    }
    for name, mask in masks.items():
        if tuple(mask.shape) != expected_shape:
            raise RuntimeError(
                f"Clean-ECST directional mask {name} shape mismatch: "
                f"{list(mask.shape)} != {list(expected_shape)}."
            )
    mask_values = list(masks.items())
    for left_index, (left_name, left) in enumerate(mask_values):
        for right_name, right in mask_values[left_index + 1 :]:
            if bool((left & right).any().item()):
                raise RuntimeError(
                    "Clean-ECST directional regions overlap: "
                    f"{left_name}/{right_name}."
                )

    erase_map = apply_directional_strength(
        conflict,
        schedule_scale,
        erase_strength,
        weight_min,
    )
    add_map = apply_directional_strength(
        conflict,
        schedule_scale,
        add_strength,
        weight_min,
    )
    ring_bg_map = apply_directional_strength(
        negative,
        schedule_scale,
        ring_bg_strength,
        weight_min,
    )
    effective_map = torch.ones_like(conflict)
    effective_map = torch.where(
        masks["erase_conflict_effective"], erase_map, effective_map
    )
    effective_map = torch.where(
        masks["add_conflict_effective"], add_map, effective_map
    )
    effective_map = torch.where(
        masks["ring_teacher_bg"], ring_bg_map, effective_map
    )
    effective_map = torch.where(
        masks["ring_teacher_fg"], torch.ones_like(effective_map), effective_map
    )
    effective_map = effective_map.clamp(float(weight_min), 1.0).detach()
    return effective_map, {
        "erase_map": erase_map.detach(),
        "add_map": add_map.detach(),
        "ring_bg_map": ring_bg_map.detach(),
        **masks,
    }


def _validate_probability(name, value, expected_shape):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected_shape):
        actual = list(value.shape) if torch.is_tensor(value) else type(value)
        raise RuntimeError(f"Clean-ECST {name} must be {list(expected_shape)}, got {actual}.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"Clean-ECST {name} contains NaN/Inf.")
    value_min = float(value.detach().min().item())
    value_max = float(value.detach().max().item())
    if value_min < -1e-6 or value_max > 1.0 + 1e-6:
        raise RuntimeError(
            f"Clean-ECST {name} is outside [0,1]: {value_min}/{value_max}."
        )


def build_clean_support_ring(foreground_evidence_68, radius=1):
    foreground = foreground_evidence_68.detach().float()
    if foreground.ndim != 4 or int(foreground.shape[1]) != 1:
        raise RuntimeError(
            "Clean-ECST foreground evidence must be [B,1,H,W], got "
            f"{list(foreground.shape)}."
        )
    radius = int(radius)
    if radius not in {1, 2}:
        raise RuntimeError("Clean-ECST supports only ring radius 1 or 2.")
    support = foreground > 0.5
    dilated = support.float()
    for _ in range(radius):
        dilated = F.max_pool2d(
            dilated, kernel_size=3, stride=1, padding=1
        )
    dilated = dilated > 0.5
    ring = dilated & (~support)
    if bool((support & ring).any().item()):
        raise RuntimeError("Clean-ECST support and ring must be mutually exclusive.")
    return support.detach(), ring.detach()


def build_clean_semantic_fg_tendency(
    feature_37,
    foreground_evidence_37,
    background_evidence_37,
    margin_tau,
    output_size=(68, 68),
):
    """Build continuous DINO foreground tendency from weighted prototypes."""

    feature = feature_37.detach().float()
    foreground = foreground_evidence_37.detach().float()
    background = background_evidence_37.detach().float()
    if feature.ndim != 4 or int(feature.shape[1]) != 384:
        raise RuntimeError(
            f"Clean-ECST DINO feature must be [B,384,H,W], got {list(feature.shape)}."
        )
    expected = (int(feature.shape[0]), 1, int(feature.shape[2]), int(feature.shape[3]))
    _validate_probability("foreground_evidence_37", foreground, expected)
    _validate_probability("background_evidence_37", background, expected)
    tau = float(margin_tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise RuntimeError(f"Clean-ECST margin_tau must be positive, got {tau}.")

    normalized = F.normalize(feature, dim=1, eps=1e-12).detach()
    tendencies = []
    fg_fallback_count = 0
    bg_fallback_count = 0
    eps = 1e-6
    for index in range(int(feature.shape[0])):
        tokens = normalized[index]
        global_proto = F.normalize(tokens.mean(dim=(1, 2)), dim=0, eps=1e-12)

        def prototype(weight, is_foreground):
            nonlocal fg_fallback_count, bg_fallback_count
            weight_sum = weight.sum()
            if float(weight_sum.detach().item()) < eps:
                if is_foreground:
                    fg_fallback_count += 1
                else:
                    bg_fallback_count += 1
                return global_proto
            weighted = (tokens * weight).sum(dim=(1, 2)) / (weight_sum + eps)
            return F.normalize(weighted, dim=0, eps=1e-12)

        fg_proto = prototype(foreground[index], True).detach()
        bg_proto = prototype(background[index], False).detach()
        similarity_fg = (tokens * fg_proto.view(-1, 1, 1)).sum(
            dim=0, keepdim=True
        )
        similarity_bg = (tokens * bg_proto.view(-1, 1, 1)).sum(
            dim=0, keepdim=True
        )
        tendencies.append(torch.sigmoid((similarity_fg - similarity_bg) / tau))

    tendency_37 = torch.stack(tendencies, dim=0).clamp(0.0, 1.0).detach()
    tendency_68 = F.interpolate(
        tendency_37,
        size=tuple(int(value) for value in output_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()
    return tendency_37, tendency_68, {
        "fg_prototype_fallback_count": int(fg_fallback_count),
        "bg_prototype_fallback_count": int(bg_fallback_count),
    }


def build_clean_history_bg_reliability(
    history_mean,
    history_second,
    history_count,
    min_history,
    variance_tau,
):
    mean = history_mean.detach().float()
    second = history_second.detach().float()
    count = history_count.detach().long()
    if tuple(mean.shape) != tuple(second.shape) or mean.ndim != 4 or int(mean.shape[1]) != 1:
        raise RuntimeError("Clean-ECST temporal mean/second shape mismatch.")
    if count.ndim != 1 or int(count.shape[0]) != int(mean.shape[0]):
        raise RuntimeError("Clean-ECST history_count must be [B].")
    if not bool(torch.isfinite(mean).all().item()) or not bool(
        torch.isfinite(second).all().item()
    ):
        raise RuntimeError("Clean-ECST temporal moments contain NaN/Inf.")
    tau = float(variance_tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise RuntimeError("ECST_CLEAN_VARIANCE_TAU must be positive.")
    variance = (second - mean.square()).clamp_min(0.0)
    valid = (count >= int(min_history)).view(-1, 1, 1, 1)
    reliability = (
        (2.0 * (0.5 - mean)).clamp(0.0, 1.0)
        * torch.exp(-variance / tau)
    ).clamp(0.0, 1.0)
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    return {
        "mean": mean.detach(),
        "variance": variance.detach(),
        "history_valid": valid.detach(),
        "history_bg_reliability": reliability.detach(),
    }


def build_clean_negative_weight(
    semantic_fg_tendency,
    history_bg_reliability,
    negative_floor,
):
    semantic = semantic_fg_tendency.detach().float()
    history = history_bg_reliability.detach().float()
    if tuple(semantic.shape) != tuple(history.shape):
        raise RuntimeError("Clean-ECST semantic/history shape mismatch.")
    floor = float(negative_floor)
    if not 0.0 <= floor <= 1.0:
        raise RuntimeError("Clean-ECST negative floor must be in [0,1].")
    return (
        1.0 - (1.0 - floor) * semantic * (1.0 - history)
    ).clamp(floor, 1.0).detach()


def _masked_mean(value, mask, empty=1.0):
    if not bool(mask.any().item()):
        return float(empty)
    return float(value[mask].mean().detach().item())


def build_ecst_clean_continuous_teacher_weight_map(
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
    """Build v5's threshold-free static evidence and continuous recovery route."""

    if bool(getattr(cfg, "ECST_CLEAN_USE_HARD_RING", True)):
        raise RuntimeError("Clean-ECST v5 forbids Hard Ring routing.")
    leaked_ring = sorted(key for key in batch if "ring" in str(key).lower())
    if leaked_ring:
        raise RuntimeError(
            f"Clean-ECST v5 batch leaked Hard Ring fields: {leaked_ring}."
        )
    required = ("dabe_clean_target_68", "dabe_clean_recoverability_68")
    missing = sorted(key for key in required if key not in batch)
    if missing:
        raise KeyError(f"Clean-ECST v5 batch is missing fields: {missing}.")

    teacher = teacher_prob.detach().to(device=device).float()
    if teacher.ndim != 4 or int(teacher.shape[1]) != 1:
        raise RuntimeError(
            f"Clean-ECST v5 Teacher probability must be [B,1,H,W], got "
            f"{list(teacher.shape)}."
        )
    expected = tuple(teacher.shape)
    target = batch["dabe_clean_target_68"].detach().to(device).float()
    recoverability = (
        batch["dabe_clean_recoverability_68"].detach().to(device).float()
    )
    _validate_probability("target_dp_68", target, expected)
    _validate_probability("recoverability_68", recoverability, expected)
    temporal = build_clean_history_bg_reliability(
        temporal_mean.detach().to(device).float(),
        temporal_second.detach().to(device).float(),
        history_count.detach().to(device),
        min_history=int(getattr(cfg, "ECST_CLEAN_MIN_HISTORY", 3)),
        variance_tau=float(getattr(cfg, "ECST_CLEAN_VARIANCE_TAU", 0.02)),
    )

    signed_static = (2.0 * target - 1.0).detach()
    positive_static_evidence = signed_static.clamp_min(0.0).detach()
    negative_static_evidence = (-signed_static).clamp_min(0.0).detach()
    teacher_fg = (teacher > 0.5).detach()
    teacher_bg = (~teacher_fg).detach()
    scale = float(get_ecst_clean_scale(cfg, epoch))
    strengths = get_ecst_clean_continuous_strengths(cfg)
    weight_min = float(getattr(cfg, "ECST_CLEAN_WEIGHT_MIN", 0.20))
    suppression_max = 1.0 - weight_min
    conflict_floor = float(getattr(cfg, "ECST_CLEAN_CONFLICT_FLOOR", 0.20))
    negative_floor = float(
        getattr(cfg, "ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR", 0.25)
    )

    erase_suppression_raw = (
        (1.0 - conflict_floor) * positive_static_evidence
    ).detach()
    erase_suppression = (
        scale * strengths["erase"] * erase_suppression_raw
    ).clamp(0.0, suppression_max).detach()
    recovery_suppression_raw = (
        (1.0 - negative_floor)
        * recoverability
        * (1.0 - temporal["history_bg_reliability"])
    ).detach()
    recovery_suppression = (
        scale * strengths["recovery"] * recovery_suppression_raw
    ).clamp(0.0, suppression_max).detach()
    teacher_bg_suppression = (
        1.0
        - (1.0 - erase_suppression)
        * (1.0 - recovery_suppression)
    ).clamp(0.0, suppression_max).detach()
    teacher_bg_weight = (1.0 - teacher_bg_suppression).clamp(
        weight_min, 1.0
    ).detach()

    add_suppression_raw = (
        (1.0 - conflict_floor) * negative_static_evidence
    ).detach()
    add_suppression = (
        scale * strengths["add"] * add_suppression_raw
    ).clamp(0.0, suppression_max).detach()
    teacher_fg_weight = (1.0 - add_suppression).clamp(
        weight_min, 1.0
    ).detach()
    effective_map = torch.where(
        teacher_fg, teacher_fg_weight, teacher_bg_weight
    ).clamp(weight_min, 1.0).detach()
    if effective_map.requires_grad or not bool(
        torch.isfinite(effective_map).all().item()
    ):
        raise RuntimeError("Clean-ECST v5 route must be finite and detached.")

    erase_weight = (1.0 - erase_suppression).clamp(weight_min, 1.0)
    recovery_weight = (1.0 - recovery_suppression).clamp(weight_min, 1.0)
    static_strength = signed_static.abs()
    semantic = batch.get("dabe_clean_semantic_fg_tendency_37")
    semantic_mean = (
        float(semantic.detach().float().mean().item())
        if torch.is_tensor(semantic)
        else 0.0
    )
    stats = {
        "routing_mode": "clean_ecst",
        "ecst_scale": scale,
        "schedule_scale": scale,
        "strength_mode": "directional_continuous",
        "strength_max": None,
        "effective_strength": None,
        "erase_strength": float(strengths["erase"]),
        "add_strength": float(strengths["add"]),
        "recovery_strength": float(strengths["recovery"]),
        "ring_bg_strength": None,
        "hard_ring_used": False,
        "memory_active": True,
        "history_count_min": int(history_count.min().detach().item()),
        "history_count_mean": float(history_count.float().mean().detach().item()),
        "history_count_max": int(history_count.max().detach().item()),
        "history_valid_ratio": float(
            temporal["history_valid"].float().mean().detach().item()
        ),
        "positive_static_evidence_mean": float(
            positive_static_evidence.mean().item()
        ),
        "negative_static_evidence_mean": float(
            negative_static_evidence.mean().item()
        ),
        "recoverability_mean": float(recoverability.mean().item()),
        "teacher_fg_ratio": float(teacher_fg.float().mean().item()),
        "teacher_bg_ratio": float(teacher_bg.float().mean().item()),
        "erase_suppression_mass": float(
            (erase_suppression * teacher_bg.float()).mean().item()
        ),
        "recovery_suppression_mass": float(
            (recovery_suppression * teacher_bg.float()).mean().item()
        ),
        "add_suppression_mass": float(
            (add_suppression * teacher_fg.float()).mean().item()
        ),
        "combined_teacher_bg_suppression_mass": float(
            (teacher_bg_suppression * teacher_bg.float()).mean().item()
        ),
        "erase_recovery_overlap_mass": float(
            (
                erase_suppression
                * recovery_suppression
                * teacher_bg.float()
            ).mean().item()
        ),
        "erase_effective_weight_mean": _masked_mean(
            erase_weight, teacher_bg, empty=1.0
        ),
        "recovery_effective_weight_mean": _masked_mean(
            recovery_weight, teacher_bg, empty=1.0
        ),
        "teacher_bg_effective_weight_mean": _masked_mean(
            teacher_bg_weight, teacher_bg, empty=1.0
        ),
        "teacher_fg_effective_weight_mean": _masked_mean(
            teacher_fg_weight, teacher_fg, empty=1.0
        ),
        "erase_floor_saturation_ratio": _masked_mean(
            (erase_weight <= weight_min + 1e-7).float(), teacher_bg, empty=0.0
        ),
        "recovery_floor_saturation_ratio": _masked_mean(
            (recovery_weight <= weight_min + 1e-7).float(),
            teacher_bg,
            empty=0.0,
        ),
        "teacher_bg_floor_saturation_ratio": _masked_mean(
            (teacher_bg_weight <= weight_min + 1e-7).float(),
            teacher_bg,
            empty=0.0,
        ),
        "teacher_fg_floor_saturation_ratio": _masked_mean(
            (teacher_fg_weight <= weight_min + 1e-7).float(),
            teacher_fg,
            empty=0.0,
        ),
        "teacher_map_raw_min": float(effective_map.min().item()),
        "teacher_map_raw_mean": float(effective_map.mean().item()),
        "teacher_map_raw_max": float(effective_map.max().item()),
        "teacher_map_min": float(effective_map.min().item()),
        "teacher_map_mean": float(effective_map.mean().item()),
        "teacher_map_max": float(effective_map.max().item()),
        # Compatibility-only continuous diagnostics for the shared logger.
        "support_area": 0.0,
        "ring_area": 0.0,
        "static_strength_min": float(static_strength.min().item()),
        "static_strength_mean": float(static_strength.mean().item()),
        "static_strength_max": float(static_strength.max().item()),
        "teacher_fg_area": float(teacher_fg.float().mean().item()),
        "conflict_ratio": 0.0,
        "fg_erase_conflict_ratio": 0.0,
        "bg_add_conflict_ratio": 0.0,
        "erase_conflict_ratio": 0.0,
        "add_conflict_ratio": 0.0,
        "ring_teacher_fg_ratio": 0.0,
        "ring_teacher_bg_ratio": 0.0,
        "semantic_fg_tendency_min": semantic_mean,
        "semantic_fg_tendency_mean": semantic_mean,
        "semantic_fg_tendency_max": semantic_mean,
        "history_bg_reliability_min": float(
            temporal["history_bg_reliability"].min().item()
        ),
        "history_bg_reliability_mean": float(
            temporal["history_bg_reliability"].mean().item()
        ),
        "history_bg_reliability_max": float(
            temporal["history_bg_reliability"].max().item()
        ),
        "negative_weight_mean": float(recovery_weight.mean().item()),
        "conflict_weight_mean": float(erase_weight.mean().item()),
        "add_effective_weight_mean": _masked_mean(
            teacher_fg_weight, teacher_fg, empty=1.0
        ),
        "ring_bg_effective_weight_mean": 1.0,
        "ring_bg_suppression_mass": 0.0,
        "add_floor_saturation_ratio": _masked_mean(
            (teacher_fg_weight <= weight_min + 1e-7).float(),
            teacher_fg,
            empty=0.0,
        ),
        "ring_bg_floor_saturation_ratio": 0.0,
        "map_on_support": 1.0,
        "map_on_ring_teacher_fg": 1.0,
        "map_on_ring_teacher_bg": 1.0,
        "map_outside_support_ring": float(effective_map.mean().item()),
        "fg_prototype_fallback_count": 0,
        "bg_prototype_fallback_count": 0,
    }
    states = {
        "signed_static": signed_static.detach(),
        "positive_static_evidence": positive_static_evidence.detach(),
        "negative_static_evidence": negative_static_evidence.detach(),
        "teacher_fg": teacher_fg.detach(),
        "teacher_bg": teacher_bg.detach(),
        "recoverability_68": recoverability.detach(),
        "history_mean": temporal["mean"].detach(),
        "history_variance": temporal["variance"].detach(),
        "history_valid": temporal["history_valid"].detach(),
        "history_bg_reliability": temporal["history_bg_reliability"].detach(),
        "erase_suppression_raw": erase_suppression_raw.detach(),
        "erase_suppression": erase_suppression.detach(),
        "recovery_suppression_raw": recovery_suppression_raw.detach(),
        "recovery_suppression": recovery_suppression.detach(),
        "teacher_bg_suppression": teacher_bg_suppression.detach(),
        "add_suppression_raw": add_suppression_raw.detach(),
        "add_suppression": add_suppression.detach(),
        "teacher_bg_weight": teacher_bg_weight.detach(),
        "teacher_fg_weight": teacher_fg_weight.detach(),
        "effective_map": effective_map.detach(),
    }
    if return_states:
        return effective_map, stats, states
    return effective_map, stats


def build_ecst_clean_teacher_weight_map(
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
    validate_ecst_clean_config(cfg)
    leaked = sorted(LEGACY_REGION_KEYS.intersection(batch))
    if leaked:
        raise RuntimeError(f"Clean-ECST batch leaked legacy PU regions: {leaked}.")
    leaked_weight = sorted(
        key
        for key in batch
        if key in {"weight_map", "pu_weight_map", "dabe_clean_static_weight_map"}
        or "static_weight_map" in key
    )
    if leaked_weight:
        raise RuntimeError(f"Clean-ECST batch leaked static weight maps: {leaked_weight}.")
    if get_ecst_clean_strength_mode(cfg) == "directional_continuous":
        return build_ecst_clean_continuous_teacher_weight_map(
            cfg=cfg,
            batch=batch,
            teacher_prob=teacher_prob,
            temporal_mean=temporal_mean,
            temporal_second=temporal_second,
            history_count=history_count,
            epoch=epoch,
            device=device,
            return_states=return_states,
        )
    required = (
        "dabe_clean_target_68",
        "dabe_clean_fg_evidence_37",
        "dabe_clean_fg_evidence_68",
        "dabe_clean_bg_evidence_37",
        "dabe_clean_bg_evidence_68",
        "feature",
    )
    missing = sorted(key for key in required if key not in batch)
    if missing:
        raise KeyError(f"Clean-ECST batch is missing fields: {missing}.")

    teacher = teacher_prob.detach().to(device=device).float()
    if teacher.ndim != 4 or int(teacher.shape[1]) != 1:
        raise RuntimeError(
            f"Clean-ECST Teacher probability must be [B,1,H,W], got {list(teacher.shape)}."
        )
    expected_68 = tuple(teacher.shape)
    target = batch["dabe_clean_target_68"].detach().to(device).float()
    foreground_68 = batch["dabe_clean_fg_evidence_68"].detach().to(device).float()
    background_68 = batch["dabe_clean_bg_evidence_68"].detach().to(device).float()
    for name, value in (
        ("target_dp_68", target),
        ("foreground_evidence_68", foreground_68),
        ("background_evidence_68", background_68),
        ("teacher_prob", teacher),
    ):
        _validate_probability(name, value, expected_68)

    batch_size = int(teacher.shape[0])
    expected_37 = (batch_size, 1, 37, 37)
    foreground_37 = batch["dabe_clean_fg_evidence_37"].detach().to(device).float()
    background_37 = batch["dabe_clean_bg_evidence_37"].detach().to(device).float()
    _validate_probability("foreground_evidence_37", foreground_37, expected_37)
    _validate_probability("background_evidence_37", background_37, expected_37)
    feature = batch["feature"].detach().to(device).float()
    if tuple(feature.shape) != (batch_size, 384, 37, 37):
        raise RuntimeError(
            "Clean-ECST DINO feature must be [B,384,37,37], got "
            f"{list(feature.shape)}."
        )

    support, ring = build_clean_support_ring(
        foreground_68,
        radius=int(getattr(cfg, "ECST_CLEAN_RING_RADIUS", 1)),
    )
    signed_evidence = (2.0 * target - 1.0).detach()
    static_strength = signed_evidence.abs().clamp(0.0, 1.0).detach()
    static_fg = (target > 0.5).detach()
    teacher_fg = (teacher > 0.5).detach()
    conflict = (teacher_fg != static_fg).detach()
    fg_erase_conflict = (static_fg & (~teacher_fg)).detach()
    bg_add_conflict = ((~static_fg) & teacher_fg).detach()

    conflict_floor = float(getattr(cfg, "ECST_CLEAN_CONFLICT_FLOOR", 0.20))
    conflict_weight = (
        1.0 - (1.0 - conflict_floor) * static_strength
    ).clamp(conflict_floor, 1.0).detach()
    semantic_37, semantic_68, prototype_stats = build_clean_semantic_fg_tendency(
        feature,
        foreground_37,
        background_37,
        margin_tau=float(getattr(cfg, "ECST_CLEAN_MARGIN_TAU", 0.05)),
        output_size=teacher.shape[-2:],
    )
    temporal = build_clean_history_bg_reliability(
        temporal_mean.detach().to(device).float(),
        temporal_second.detach().to(device).float(),
        history_count.detach().to(device),
        min_history=int(getattr(cfg, "ECST_CLEAN_MIN_HISTORY", 3)),
        variance_tau=float(getattr(cfg, "ECST_CLEAN_VARIANCE_TAU", 0.02)),
    )
    negative_weight = build_clean_negative_weight(
        semantic_68,
        temporal["history_bg_reliability"],
        negative_floor=float(
            getattr(cfg, "ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR", 0.25)
        ),
    )

    ring_teacher_fg = (ring & teacher_fg).detach()
    ring_teacher_bg = (ring & (~teacher_fg)).detach()
    erase_conflict_effective = (fg_erase_conflict & (~ring)).detach()
    add_conflict_effective = (bg_add_conflict & (~ring)).detach()
    raw_map = torch.ones_like(target)
    raw_map = torch.where(conflict, conflict_weight, raw_map)
    raw_map = torch.where(ring_teacher_fg, torch.ones_like(raw_map), raw_map)
    raw_map = torch.where(ring_teacher_bg, negative_weight, raw_map)
    weight_min = float(getattr(cfg, "ECST_CLEAN_WEIGHT_MIN", 0.20))
    weight_max = float(getattr(cfg, "ECST_CLEAN_WEIGHT_MAX", 1.00))
    raw_map = raw_map.clamp(weight_min, weight_max).detach()
    scale = get_ecst_clean_scale(cfg, epoch)
    strength_mode = get_ecst_clean_strength_mode(cfg)
    if strength_mode == "directional":
        directional_strengths = get_ecst_clean_directional_strengths(cfg)
        effective_map, directional_states = compose_ecst_clean_directional_map(
            conflict_weight=conflict_weight,
            negative_weight=negative_weight,
            erase_conflict_effective=erase_conflict_effective,
            add_conflict_effective=add_conflict_effective,
            ring_teacher_bg=ring_teacher_bg,
            ring_teacher_fg=ring_teacher_fg,
            schedule_scale=scale,
            erase_strength=directional_strengths["erase"],
            add_strength=directional_strengths["add"],
            ring_bg_strength=directional_strengths["ring_bg"],
            weight_min=weight_min,
        )
        strength_max = None
        effective_strength = None
    else:
        strength_max = get_ecst_clean_strength_max(cfg)
        effective_strength = float(scale) * strength_max
        effective_map = apply_ecst_clean_strength(
            raw_map,
            schedule_scale=scale,
            strength_max=strength_max,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        directional_strengths = {
            "erase": strength_max,
            "add": strength_max,
            "ring_bg": strength_max,
        }
        directional_equivalent, directional_states = (
            compose_ecst_clean_directional_map(
                conflict_weight=conflict_weight,
                negative_weight=negative_weight,
                erase_conflict_effective=erase_conflict_effective,
                add_conflict_effective=add_conflict_effective,
                ring_teacher_bg=ring_teacher_bg,
                ring_teacher_fg=ring_teacher_fg,
                schedule_scale=scale,
                erase_strength=strength_max,
                add_strength=strength_max,
                ring_bg_strength=strength_max,
                weight_min=weight_min,
            )
        )
        if not torch.equal(effective_map, directional_equivalent):
            raise RuntimeError(
                "Uniform and equal-direction Clean-ECST maps diverged."
            )
    if effective_map.requires_grad or not bool(torch.isfinite(effective_map).all().item()):
        raise RuntimeError("Clean-ECST Teacher map must be finite and detached.")
    if tuple(effective_map.shape) != expected_68:
        raise RuntimeError("Clean-ECST Teacher map shape mismatch.")

    outside = (~(support | ring)).detach()
    stats = {
        "routing_mode": "clean_ecst",
        "ecst_scale": float(scale),
        "schedule_scale": float(scale),
        "strength_mode": strength_mode,
        "strength_max": (
            float(strength_max) if strength_max is not None else None
        ),
        "effective_strength": (
            float(effective_strength) if effective_strength is not None else None
        ),
        "erase_strength": float(directional_strengths["erase"]),
        "add_strength": float(directional_strengths["add"]),
        "ring_bg_strength": float(directional_strengths["ring_bg"]),
        "memory_active": True,
        "history_count_min": int(history_count.min().detach().item()),
        "history_count_mean": float(history_count.float().mean().detach().item()),
        "history_count_max": int(history_count.max().detach().item()),
        "history_valid_ratio": float(
            temporal["history_valid"].float().mean().detach().item()
        ),
        "support_area": float(support.float().mean().item()),
        "ring_area": float(ring.float().mean().item()),
        "static_strength_min": float(static_strength.min().item()),
        "static_strength_mean": float(static_strength.mean().item()),
        "static_strength_max": float(static_strength.max().item()),
        "teacher_fg_area": float(teacher_fg.float().mean().item()),
        "conflict_ratio": float(conflict.float().mean().item()),
        "fg_erase_conflict_ratio": float(fg_erase_conflict.float().mean().item()),
        "bg_add_conflict_ratio": float(bg_add_conflict.float().mean().item()),
        "erase_conflict_ratio": float(
            erase_conflict_effective.float().mean().item()
        ),
        "add_conflict_ratio": float(
            add_conflict_effective.float().mean().item()
        ),
        "ring_teacher_fg_ratio": float(ring_teacher_fg.float().mean().item()),
        "ring_teacher_bg_ratio": float(ring_teacher_bg.float().mean().item()),
        "semantic_fg_tendency_min": float(semantic_68.min().item()),
        "semantic_fg_tendency_mean": float(semantic_68.mean().item()),
        "semantic_fg_tendency_max": float(semantic_68.max().item()),
        "history_bg_reliability_min": float(
            temporal["history_bg_reliability"].min().item()
        ),
        "history_bg_reliability_mean": float(
            temporal["history_bg_reliability"].mean().item()
        ),
        "history_bg_reliability_max": float(
            temporal["history_bg_reliability"].max().item()
        ),
        "negative_weight_mean": _masked_mean(
            negative_weight, ring_teacher_bg, empty=1.0
        ),
        "conflict_weight_mean": _masked_mean(
            conflict_weight, conflict, empty=1.0
        ),
        "erase_effective_weight_mean": _masked_mean(
            effective_map, erase_conflict_effective, empty=1.0
        ),
        "add_effective_weight_mean": _masked_mean(
            effective_map, add_conflict_effective, empty=1.0
        ),
        "ring_bg_effective_weight_mean": _masked_mean(
            effective_map, ring_teacher_bg, empty=1.0
        ),
        "erase_suppression_mass": float(
            (
                (1.0 - effective_map)
                * erase_conflict_effective.float()
            ).mean().item()
        ),
        "add_suppression_mass": float(
            (
                (1.0 - effective_map)
                * add_conflict_effective.float()
            ).mean().item()
        ),
        "ring_bg_suppression_mass": float(
            (
                (1.0 - effective_map) * ring_teacher_bg.float()
            ).mean().item()
        ),
        "erase_floor_saturation_ratio": _masked_mean(
            (effective_map <= weight_min + 1e-7).float(),
            erase_conflict_effective,
            empty=0.0,
        ),
        "add_floor_saturation_ratio": _masked_mean(
            (effective_map <= weight_min + 1e-7).float(),
            add_conflict_effective,
            empty=0.0,
        ),
        "ring_bg_floor_saturation_ratio": _masked_mean(
            (effective_map <= weight_min + 1e-7).float(),
            ring_teacher_bg,
            empty=0.0,
        ),
        "teacher_map_raw_min": float(raw_map.min().item()),
        "teacher_map_raw_mean": float(raw_map.mean().item()),
        "teacher_map_raw_max": float(raw_map.max().item()),
        "teacher_map_min": float(effective_map.min().item()),
        "teacher_map_mean": float(effective_map.mean().item()),
        "teacher_map_max": float(effective_map.max().item()),
        "map_on_support": _masked_mean(effective_map, support, empty=1.0),
        "map_on_ring_teacher_fg": _masked_mean(
            effective_map, ring_teacher_fg, empty=1.0
        ),
        "map_on_ring_teacher_bg": _masked_mean(
            effective_map, ring_teacher_bg, empty=1.0
        ),
        "map_outside_support_ring": _masked_mean(
            effective_map, outside, empty=1.0
        ),
        **prototype_stats,
    }
    states = {
        "signed_evidence": signed_evidence.detach(),
        "static_strength": static_strength.detach(),
        "static_fg": static_fg.detach(),
        "support": support.detach(),
        "ring": ring.detach(),
        "teacher_fg": teacher_fg.detach(),
        "conflict": conflict.detach(),
        "fg_erase_conflict": fg_erase_conflict.detach(),
        "bg_add_conflict": bg_add_conflict.detach(),
        "conflict_weight": conflict_weight.detach(),
        "semantic_fg_tendency_37": semantic_37.detach(),
        "semantic_fg_tendency_68": semantic_68.detach(),
        "history_mean": temporal["mean"].detach(),
        "history_variance": temporal["variance"].detach(),
        "history_valid": temporal["history_valid"].detach(),
        "history_bg_reliability": temporal["history_bg_reliability"].detach(),
        "negative_weight": negative_weight.detach(),
        "ring_teacher_fg": ring_teacher_fg.detach(),
        "ring_teacher_bg": ring_teacher_bg.detach(),
        "erase_conflict_effective": erase_conflict_effective.detach(),
        "add_conflict_effective": add_conflict_effective.detach(),
        "raw_map": raw_map.detach(),
        "effective_map": effective_map.detach(),
        "erase_map": directional_states["erase_map"].detach(),
        "add_map": directional_states["add_map"].detach(),
        "ring_bg_map": directional_states["ring_bg_map"].detach(),
    }
    if return_states:
        return effective_map, stats, states
    return effective_map, stats
