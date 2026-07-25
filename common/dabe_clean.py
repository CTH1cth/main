"""Continuous DABE targets used by the Bridge/Clean supervision path.

This module is intentionally independent from the PU-v1.1 region-construction
code.  It consumes already cached continuous evidence and never reads GT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DABE_BRIDGE_VERSION = "dabe_bridge_v1"
DABE_CLEAN_VERSION = "dabe_clean_v1"
DABE_CLEAN_CONTREC_VERSION = "dabe_clean_v2_contrec"
DABE_CLEAN_TARGET_MODES = {"dp", "diff", "bridge"}


def expected_dabe_clean_payload_version(cfg, mode=None):
    mode = str(
        mode if mode is not None else getattr(cfg, "DABE_CLEAN_TARGET_MODE", "dp")
    ).strip().lower()
    if mode == "bridge":
        return DABE_BRIDGE_VERSION
    configured = str(
        getattr(cfg, "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION", "")
    ).strip()
    if configured:
        return configured
    version = str(getattr(cfg, "DABE_CLEAN_VERSION", "v1")).strip().lower()
    if version == "v1":
        return DABE_CLEAN_VERSION
    if version == "v2_contrec":
        return DABE_CLEAN_CONTREC_VERSION
    raise RuntimeError(f"Unsupported DABE_CLEAN_VERSION={version!r}.")


def _validate_probability_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if value.ndim not in {3, 4} or int(value.shape[-3]) != 1:
        raise RuntimeError(
            f"{name} must be [1,H,W] or [B,1,H,W], got {list(value.shape)}."
        )
    if value.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32, got {value.dtype}.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")
    value_min = float(value.min().detach().item())
    value_max = float(value.max().detach().item())
    if value_min < -1e-6 or value_max > 1.0 + 1e-6:
        raise RuntimeError(
            f"{name} must remain in [0,1], got {value_min:.8f}/{value_max:.8f}."
        )


def _validate_same_shape(left, right, left_name, right_name):
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(
            f"{left_name}/{right_name} shape mismatch: "
            f"{list(left.shape)} != {list(right.shape)}."
        )


def _finish_target(value, name):
    value = value.clamp(0.0, 1.0).to(dtype=torch.float32).detach()
    _validate_probability_tensor(value, name)
    if value.requires_grad:
        raise RuntimeError(f"{name} must be detached.")
    return value


def build_bridge_target(target_soft, static_weight):
    """Absorb the PU static map into one neutral-centred soft target."""

    _validate_probability_tensor(target_soft, "target_soft")
    _validate_probability_tensor(static_weight, "static_weight")
    _validate_same_shape(target_soft, static_weight, "target_soft", "static_weight")
    target_snapshot = target_soft.detach().clone()
    weight_snapshot = static_weight.detach().clone()
    bridge = 0.5 + static_weight.detach() * (target_soft.detach() - 0.5)
    bridge = _finish_target(bridge, "bridge_target")
    if not torch.equal(target_soft.detach(), target_snapshot):
        raise RuntimeError("build_bridge_target modified target_soft in place.")
    if not torch.equal(static_weight.detach(), weight_snapshot):
        raise RuntimeError("build_bridge_target modified static_weight in place.")
    return bridge


def build_background_evidence(bc_map, residual):
    """Build continuous background evidence B=b*(1-r)."""

    _validate_probability_tensor(bc_map, "bc_map")
    _validate_probability_tensor(residual, "residual")
    _validate_same_shape(bc_map, residual, "bc_map", "residual")
    bc_snapshot = bc_map.detach().clone()
    residual_snapshot = residual.detach().clone()
    background = bc_map.detach().clamp(0.0, 1.0) * (
        1.0 - residual.detach().clamp(0.0, 1.0)
    )
    background = _finish_target(background, "background_evidence")
    if not torch.equal(bc_map.detach(), bc_snapshot):
        raise RuntimeError("build_background_evidence modified bc_map in place.")
    if not torch.equal(residual.detach(), residual_snapshot):
        raise RuntimeError("build_background_evidence modified residual in place.")
    return background


def build_clean_targets(foreground_evidence, background_evidence):
    """Return the direction-preserving and evidence-difference targets."""

    _validate_probability_tensor(foreground_evidence, "foreground_evidence")
    _validate_probability_tensor(background_evidence, "background_evidence")
    _validate_same_shape(
        foreground_evidence,
        background_evidence,
        "foreground_evidence",
        "background_evidence",
    )
    foreground_snapshot = foreground_evidence.detach().clone()
    background_snapshot = background_evidence.detach().clone()
    foreground = foreground_evidence.detach().clamp(0.0, 1.0)
    background = background_evidence.detach().clamp(0.0, 1.0)
    confidence = torch.maximum(foreground, background)
    target_dp = 0.5 + confidence * (foreground - 0.5)
    target_diff = 0.5 + 0.5 * (foreground - background)
    outputs = {
        "foreground_evidence": _finish_target(
            foreground, "foreground_evidence_output"
        ),
        "background_evidence": _finish_target(
            background, "background_evidence_output"
        ),
        "confidence": _finish_target(confidence, "clean_confidence"),
        "target_dp": _finish_target(target_dp, "target_dp"),
        "target_diff": _finish_target(target_diff, "target_diff"),
    }
    if not torch.equal(foreground_evidence.detach(), foreground_snapshot):
        raise RuntimeError("build_clean_targets modified foreground_evidence in place.")
    if not torch.equal(background_evidence.detach(), background_snapshot):
        raise RuntimeError("build_clean_targets modified background_evidence in place.")
    return outputs


def build_continuous_recoverability(
    p_rw_37,
    evidence_gate_37,
    foreground_evidence_37,
    background_evidence_37,
    semantic_fg_tendency_37,
    output_size=(68, 68),
):
    """Build the threshold-free DABE continuous recoverability field."""

    inputs = {
        "p_rw_37": p_rw_37,
        "evidence_gate_37": evidence_gate_37,
        "foreground_evidence_37": foreground_evidence_37,
        "background_evidence_37": background_evidence_37,
        "semantic_fg_tendency_37": semantic_fg_tendency_37,
    }
    snapshots = {}
    reference_shape = None
    for name, value in inputs.items():
        _validate_probability_tensor(value, name)
        snapshots[name] = value.detach().clone()
        if reference_shape is None:
            reference_shape = tuple(value.shape)
        elif tuple(value.shape) != reference_shape:
            raise RuntimeError(
                f"Continuous recoverability shape mismatch for {name}: "
                f"{list(value.shape)} != {list(reference_shape)}."
            )

    p_rw = p_rw_37.detach().float()
    evidence_gate = evidence_gate_37.detach().float()
    foreground = foreground_evidence_37.detach().float()
    background = background_evidence_37.detach().float()
    semantic = semantic_fg_tendency_37.detach().float()
    reconstructed_foreground = (p_rw * evidence_gate).clamp(0.0, 1.0)
    foreground_reconstruction_error = float(
        (reconstructed_foreground - foreground).abs().max().item()
    )
    if foreground_reconstruction_error >= 1e-5:
        raise RuntimeError(
            "DABE foreground evidence regression failed: "
            "max_abs(p_rw_37*evidence_gate_37-foreground_evidence_37)="
            f"{foreground_reconstruction_error:.9g} >= 1e-5."
        )

    latent_rw = (p_rw - foreground).clamp(0.0, 1.0)
    recovery_product = (
        latent_rw * (1.0 - background) * semantic
    ).clamp(0.0, 1.0)
    recoverability_37 = recovery_product.pow(1.0 / 3.0).clamp(0.0, 1.0)
    batched = recoverability_37.unsqueeze(0) if recoverability_37.ndim == 3 else recoverability_37
    recoverability_68 = F.interpolate(
        batched,
        size=tuple(int(value) for value in output_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if recoverability_37.ndim == 3:
        recoverability_68 = recoverability_68.squeeze(0)

    outputs = {
        "latent_rw_37": _finish_target(latent_rw, "latent_rw_37"),
        "recovery_product_37": _finish_target(
            recovery_product, "recovery_product_37"
        ),
        "recoverability_37": _finish_target(
            recoverability_37, "recoverability_37"
        ),
        "recoverability_68": _finish_target(
            recoverability_68, "recoverability_68"
        ),
        "foreground_reconstruction_error": foreground_reconstruction_error,
    }
    for name, value in inputs.items():
        if not torch.equal(value.detach(), snapshots[name]):
            raise RuntimeError(
                f"build_continuous_recoverability modified {name} in place."
            )
    return outputs


def select_clean_target(payload, mode, suffix="68"):
    mode = str(mode).strip().lower()
    if mode not in DABE_CLEAN_TARGET_MODES:
        raise RuntimeError(
            f"Unsupported DABE_CLEAN_TARGET_MODE={mode!r}, "
            f"allowed={sorted(DABE_CLEAN_TARGET_MODES)}."
        )
    suffix = str(suffix)
    key = f"bridge_target_{suffix}" if mode == "bridge" else f"target_{mode}_{suffix}"
    if key not in payload:
        raise KeyError(f"DABE target payload is missing {key!r}.")
    target = payload[key].float().detach()
    _validate_probability_tensor(target, key)
    return target
