import hashlib
import json

import torch

from common.utils import config_to_dict


STATIC_WEIGHT_MODES = {"cache", "known_uniform", "ones"}
STATIC_WEIGHT_REGIONS = (
    "fg_core",
    "fg_fallback",
    "bg_core",
    "extent",
    "unknown",
    "other",
)


def get_static_weight_mode(cfg):
    mode = str(getattr(cfg, "STATIC_WEIGHT_MODE", "cache")).lower()
    if mode not in STATIC_WEIGHT_MODES:
        raise RuntimeError(
            f"Unsupported STATIC_WEIGHT_MODE={mode!r}, "
            f"allowed={sorted(STATIC_WEIGHT_MODES)}"
        )
    return mode


def _validate_unit_map(tensor, name):
    if tensor.ndim != 4 or int(tensor.shape[1]) != 1:
        raise RuntimeError(f"{name} must be [B,1,H,W], got {list(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf")
    min_value = float(tensor.min().detach().item())
    max_value = float(tensor.max().detach().item())
    if min_value < 0.0 or max_value > 1.0:
        raise RuntimeError(
            f"{name} must remain in [0,1], got "
            f"min={min_value:.8f}, max={max_value:.8f}"
        )


def build_effective_static_weight(cfg, batch, raw_weight_map, device):
    mode = get_static_weight_mode(cfg)
    raw_weight_map = raw_weight_map.to(device=device).float()
    _validate_unit_map(raw_weight_map, "raw static weight map")

    if mode == "cache":
        effective = raw_weight_map
    elif mode == "ones":
        effective = torch.ones_like(raw_weight_map)
    elif mode == "known_uniform":
        if "pu_unknown" not in batch:
            raise KeyError(
                "STATIC_WEIGHT_MODE='known_uniform' requires batch['pu_unknown']"
            )
        unknown = batch["pu_unknown"].to(
            device=device,
            non_blocking=True,
        ).float()
        _validate_unit_map(unknown, "pu_unknown")
        if tuple(unknown.shape) != tuple(raw_weight_map.shape):
            raise RuntimeError(
                "pu_unknown shape mismatch: "
                f"{list(unknown.shape)} != {list(raw_weight_map.shape)}"
            )
        effective = (unknown <= 0.5).float()
    else:
        raise AssertionError(mode)

    _validate_unit_map(effective, "effective static weight map")
    return effective.detach(), mode


def build_static_weight_region_masks(batch, device, expected_shape=None):
    required = {
        "fg_core": "pu_fg_core",
        "fg_fallback": "pu_fg_fallback",
        "bg_core": "pu_bg_core",
        "extent": "pu_extent",
        "unknown": "pu_unknown",
    }
    source = {}
    for name, key in required.items():
        if key not in batch:
            raise KeyError(f"Static-weight region audit requires batch[{key!r}]")
        tensor = batch[key].to(device=device, non_blocking=True).float()
        _validate_unit_map(tensor, key)
        if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"{key} shape mismatch: {list(tensor.shape)} != "
                f"{list(expected_shape)}"
            )
        source[name] = tensor > 0.5

    fg_core = source["fg_core"]
    fg_fallback = source["fg_fallback"] & (~fg_core)
    bg_core = source["bg_core"] & (~fg_core) & (~fg_fallback)
    extent = (
        source["extent"]
        & (~fg_core)
        & (~fg_fallback)
        & (~bg_core)
    )
    unknown = (
        source["unknown"]
        & (~fg_core)
        & (~fg_fallback)
        & (~bg_core)
        & (~extent)
    )
    other = ~(fg_core | fg_fallback | bg_core | extent | unknown)
    masks = {
        "fg_core": fg_core,
        "fg_fallback": fg_fallback,
        "bg_core": bg_core,
        "extent": extent,
        "unknown": unknown,
        "other": other,
    }
    coverage = sum(mask.to(torch.int8) for mask in masks.values())
    if not bool(torch.equal(coverage, torch.ones_like(coverage))):
        raise RuntimeError("Static-weight diagnostic regions are not mutually exclusive")
    return masks


def static_weight_protocol_fingerprint(cfg):
    resolved = config_to_dict(cfg)
    resolved.pop("EXP_NAME", None)
    resolved.pop("STATIC_WEIGHT_MODE", None)
    payload = json.dumps(
        resolved,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_static_weight_audit_accumulator():
    return {
        "pixels": 0,
        "raw_weight_sum": 0.0,
        "effective_weight_sum": 0.0,
        "effective_zero_count": 0,
        "effective_full_count": 0,
        "gradient_proxy_sum": 0.0,
        "student_prob_sum": 0.0,
        "teacher_prob_sum": 0.0,
        "student_pred_count": 0,
        "teacher_pred_count": 0,
        "region_pixels": {name: 0 for name in STATIC_WEIGHT_REGIONS},
        "region_weight_sum": {name: 0.0 for name in STATIC_WEIGHT_REGIONS},
        "region_gradient_sum": {name: 0.0 for name in STATIC_WEIGHT_REGIONS},
        "region_student_prob_sum": {
            name: 0.0 for name in STATIC_WEIGHT_REGIONS
        },
        "region_teacher_prob_sum": {
            name: 0.0 for name in STATIC_WEIGHT_REGIONS
        },
    }


def accumulate_static_weight_audit(
    accumulator,
    raw_weight_map,
    effective_weight_map,
    static_target,
    student_logits,
    teacher_prob,
    region_masks,
    static_weight,
):
    with torch.no_grad():
        raw = raw_weight_map.detach().float()
        effective = effective_weight_map.detach().float()
        target = static_target.detach().float()
        student_prob = student_logits.detach().float().sigmoid()
        teacher_prob = teacher_prob.detach().float()
        expected_shape = tuple(raw.shape)
        for name, tensor in (
            ("effective_weight_map", effective),
            ("static_target", target),
            ("student_logits", student_logits),
            ("teacher_prob", teacher_prob),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Static-weight audit {name} shape mismatch: "
                    f"{list(tensor.shape)} != {list(expected_shape)}"
                )
        _validate_unit_map(raw, "raw static weight map")
        _validate_unit_map(effective, "effective static weight map")
        _validate_unit_map(target, "static target")
        _validate_unit_map(teacher_prob, "teacher probability")

        pixels = int(raw.numel())
        accumulator["pixels"] += pixels
        accumulator["raw_weight_sum"] += float(raw.double().sum().item())
        accumulator["effective_weight_sum"] += float(
            effective.double().sum().item()
        )
        accumulator["effective_zero_count"] += int(
            (effective <= 1e-8).sum().item()
        )
        accumulator["effective_full_count"] += int(
            (effective >= 0.99).sum().item()
        )
        accumulator["student_prob_sum"] += float(
            student_prob.double().sum().item()
        )
        accumulator["teacher_prob_sum"] += float(
            teacher_prob.double().sum().item()
        )
        accumulator["student_pred_count"] += int(
            (student_prob >= 0.5).sum().item()
        )
        accumulator["teacher_pred_count"] += int(
            (teacher_prob >= 0.5).sum().item()
        )

        static_active = float(static_weight) > 1e-12
        gradient_proxy = (
            effective * (student_prob - target).abs()
            if static_active
            else torch.zeros_like(effective)
        )
        accumulator["gradient_proxy_sum"] += float(
            gradient_proxy.double().sum().item()
        )
        for name in STATIC_WEIGHT_REGIONS:
            mask = region_masks[name]
            if tuple(mask.shape) != expected_shape or mask.dtype != torch.bool:
                raise RuntimeError(
                    f"Static-weight region {name} must be bool "
                    f"{list(expected_shape)}, got {mask.dtype} {list(mask.shape)}"
                )
            count = int(mask.sum().item())
            accumulator["region_pixels"][name] += count
            if count <= 0:
                continue
            accumulator["region_weight_sum"][name] += float(
                effective[mask].double().sum().item()
            )
            accumulator["region_gradient_sum"][name] += float(
                gradient_proxy[mask].double().sum().item()
            )
            accumulator["region_student_prob_sum"][name] += float(
                student_prob[mask].double().sum().item()
            )
            accumulator["region_teacher_prob_sum"][name] += float(
                teacher_prob[mask].double().sum().item()
            )


def finalize_static_weight_audit(
    accumulator,
    epoch,
    mode,
    global_static_weight,
    global_teacher_weight,
):
    pixels = int(accumulator["pixels"])
    if pixels <= 0:
        raise RuntimeError("Static-weight epoch audit has no pixels")
    effective_sum = float(accumulator["effective_weight_sum"])
    gradient_sum = float(accumulator["gradient_proxy_sum"])
    static_active = float(global_static_weight) > 1e-12
    row = {
        "epoch": int(epoch),
        "static_weight_mode": str(mode),
        "global_static_weight": float(global_static_weight),
        "global_teacher_weight": float(global_teacher_weight),
        "static_gradient_active": bool(static_active),
        "raw_weight_mean": float(accumulator["raw_weight_sum"]) / pixels,
        "effective_weight_mean": effective_sum / pixels,
        "effective_zero_ratio": int(accumulator["effective_zero_count"]) / pixels,
        "effective_full_ratio": int(accumulator["effective_full_count"]) / pixels,
        "student_prob_mean": float(accumulator["student_prob_sum"]) / pixels,
        "teacher_prob_mean": float(accumulator["teacher_prob_sum"]) / pixels,
        "student_pred_area": int(accumulator["student_pred_count"]) / pixels,
        "teacher_pred_area": int(accumulator["teacher_pred_count"]) / pixels,
    }
    for name in STATIC_WEIGHT_REGIONS:
        count = int(accumulator["region_pixels"][name])
        row[f"{name}_area_ratio"] = count / pixels
        row[f"{name}_weight_mass"] = (
            float(accumulator["region_weight_sum"][name]) / effective_sum
            if effective_sum > 0.0
            else 0.0
        )
        row[f"{name}_gradmass"] = (
            float(accumulator["region_gradient_sum"][name]) / gradient_sum
            if static_active and gradient_sum > 0.0
            else 0.0
        )
        row[f"student_prob_{name}"] = (
            float(accumulator["region_student_prob_sum"][name]) / count
            if count > 0
            else 0.0
        )
        row[f"teacher_prob_{name}"] = (
            float(accumulator["region_teacher_prob_sum"][name]) / count
            if count > 0
            else 0.0
        )
    return row
