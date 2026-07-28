"""Strict adapter for the historical FOUND-style fixed pseudo cache."""

from pathlib import Path

import torch
import torch.nn.functional as F


FOUND_STATIC_SOURCE = "found_fixed"
FOUND_STATIC_RESIZE_MODE = "bilinear"
FOUND_STATIC_NATIVE_SHAPE = (1, 28, 28)


def found_static_enabled(cfg):
    return str(
        getattr(cfg, "DABE_CLEAN_TRAINING_TARGET_SOURCE", "target_offline_68")
    ).strip().lower() == FOUND_STATIC_SOURCE


def found_static_root(cfg):
    value = str(getattr(cfg, "FOUND_STATIC_ROOT", "")).strip()
    if not value:
        raise RuntimeError(
            "DABE_CLEAN_TRAINING_TARGET_SOURCE='found_fixed' requires "
            "FOUND_STATIC_ROOT."
        )
    return Path(value).expanduser().resolve()


def found_static_manifest_path(cfg):
    return found_static_root(cfg) / "manifest_train.jsonl"


def build_found_static_target(native, loss_size):
    """Resize a detached binary FOUND mask to the continuous training target."""

    if not torch.is_tensor(native):
        raise TypeError("FOUND static target must be a tensor.")
    value = native.detach().to(device="cpu", dtype=torch.float32)
    if tuple(value.shape) != FOUND_STATIC_NATIVE_SHAPE:
        raise RuntimeError(
            "FOUND static native shape mismatch: "
            f"{list(value.shape)} != {list(FOUND_STATIC_NATIVE_SHAPE)}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("FOUND static target contains NaN/Inf.")
    if not bool(((value == 0.0) | (value == 1.0)).all().item()):
        raise RuntimeError("FOUND static native cache must be exactly binary {0,1}.")
    size = int(loss_size)
    if size <= 0:
        raise ValueError(f"FOUND static loss size must be positive, got {size}.")
    target = F.interpolate(
        value.unsqueeze(0),
        size=(size, size),
        mode=FOUND_STATIC_RESIZE_MODE,
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).detach()
    if tuple(target.shape) != (1, size, size):
        raise RuntimeError(f"FOUND static resized shape mismatch: {list(target.shape)}.")
    if target.dtype != torch.float32 or target.requires_grad:
        raise RuntimeError("FOUND static resized target must be detached float32.")
    if not bool(torch.isfinite(target).all().item()):
        raise RuntimeError("FOUND static resized target contains NaN/Inf.")
    return target
