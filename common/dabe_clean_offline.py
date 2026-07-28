"""Pure-offline DABE complementary-evidence consolidation.

The functions in this module are deliberately independent from Teacher,
training epoch, temporal memory, and every ECST routing implementation.  They
consume fixed 37x37 evidence maps and return one mode-selected soft target.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F


DABE_CLEAN_OFFLINE_PAYLOAD_VERSION = "dabe_clean_v3_offline_consolidation"
DABE_CLEAN_OFFLINE_MODES = {
    "baseline_dp",
    "complementary_fg",
    "complementary_fg_conflict",
    "direct_consolidated_fg",
}

_FORMULA_SPEC = {
    "primary_fg": "p_rw*evidence",
    "background": "bc*(1-residual)",
    "latent_support": "relu(p_rw-primary_fg)",
    "complementary_fg": "clamp(latent_support*(1-background)*semantic,0,1)^(1/3)",
    "consolidated_fg": "primary_fg+(1-primary_fg)*complementary_fg",
    "baseline_dp": "0.5+max(primary_fg,background)*(primary_fg-0.5)",
    "complementary_fg_mode": "0.5+max(consolidated_fg,background)*(consolidated_fg-0.5)",
    "complementary_fg_conflict": "0.5+max(consolidated_fg,background)*(1-consolidated_fg*background)*(consolidated_fg-0.5)",
    "direct_consolidated_fg": "consolidated_fg",
    "resize": "bilinear evidence 37->68, then reapply selected target formula",
}
DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT = hashlib.sha256(
    json.dumps(_FORMULA_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _validate_probability(name, value, expected_shape=None):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if value.dtype != torch.float32:
        raise RuntimeError(f"{name} must be float32, got {value.dtype}.")
    if value.ndim not in {3, 4} or int(value.shape[-3]) != 1:
        raise RuntimeError(
            f"{name} must have shape [1,H,W] or [B,1,H,W], got {list(value.shape)}."
        )
    if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{name} shape mismatch: {list(value.shape)} != {list(expected_shape)}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")
    minimum = float(value.detach().min().item())
    maximum = float(value.detach().max().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise RuntimeError(
            f"{name} must remain in [0,1], got {minimum:.9g}/{maximum:.9g}."
        )


def _finish(name, value):
    value = value.to(dtype=torch.float32).clamp(0.0, 1.0).detach()
    _validate_probability(name, value)
    if value.requires_grad:
        raise RuntimeError(f"{name} must be detached.")
    return value


def _mode_target(primary_fg, background, complementary_fg, mode):
    """Apply a v3 target mode at any common spatial resolution."""

    mode = str(mode).strip().lower()
    if mode not in DABE_CLEAN_OFFLINE_MODES:
        raise RuntimeError(
            f"Unsupported DABE_CLEAN_OFFLINE_MODE={mode!r}; "
            f"allowed={sorted(DABE_CLEAN_OFFLINE_MODES)}."
        )
    consolidated = (
        primary_fg + (1.0 - primary_fg) * complementary_fg
    ).clamp(0.0, 1.0)
    conflict = (consolidated * background).clamp(0.0, 1.0)
    if mode == "baseline_dp":
        target_fg = primary_fg
        commitment = torch.maximum(primary_fg, background)
        target = 0.5 + commitment * (target_fg - 0.5)
    elif mode == "complementary_fg":
        target_fg = consolidated
        commitment = torch.maximum(consolidated, background)
        target = 0.5 + commitment * (target_fg - 0.5)
    elif mode == "complementary_fg_conflict":
        target_fg = consolidated
        commitment = torch.maximum(consolidated, background) * (1.0 - conflict)
        target = 0.5 + commitment * (target_fg - 0.5)
    else:
        # V3 is a direct-soft control.  An all-one commitment preserves the
        # unified neutral-centred representation without affecting the target.
        target_fg = consolidated
        commitment = torch.ones_like(consolidated)
        target = consolidated
    return {
        "consolidated_fg": _finish("consolidated_fg", consolidated),
        "conflict": _finish("conflict", conflict),
        "commitment": _finish("commitment", commitment),
        "target": _finish("target_offline", target),
    }


def build_offline_consolidation(
    *,
    residual_37,
    bc_map_37,
    p_rw_37,
    evidence_37,
    semantic_fg_tendency_37,
    offline_mode,
    output_size=(68, 68),
):
    """Build all offline diagnostics and the sole 68x68 training target."""

    inputs = {
        "residual_37": residual_37,
        "bc_map_37": bc_map_37,
        "p_rw_37": p_rw_37,
        "evidence_37": evidence_37,
        "semantic_fg_tendency_37": semantic_fg_tendency_37,
    }
    snapshots = {}
    reference_shape = None
    for name, value in inputs.items():
        _validate_probability(name, value)
        snapshots[name] = value.detach().clone()
        if reference_shape is None:
            reference_shape = tuple(value.shape)
        elif tuple(value.shape) != reference_shape:
            raise RuntimeError(
                f"Offline evidence shape mismatch for {name}: "
                f"{list(value.shape)} != {list(reference_shape)}."
            )
    if tuple(reference_shape[-2:]) != (37, 37):
        raise RuntimeError(
            f"Offline evidence must be native 37x37, got {list(reference_shape)}."
        )

    residual = residual_37.detach().float()
    bc_map = bc_map_37.detach().float()
    p_rw = p_rw_37.detach().float()
    evidence = evidence_37.detach().float()
    semantic = semantic_fg_tendency_37.detach().float()
    primary = (p_rw * evidence).clamp(0.0, 1.0)
    background = (bc_map * (1.0 - residual)).clamp(0.0, 1.0)
    latent = torch.relu(p_rw - primary).clamp(0.0, 1.0)
    complementary = (
        latent * (1.0 - background) * semantic
    ).clamp(0.0, 1.0).pow(1.0 / 3.0)
    mode_37 = _mode_target(primary, background, complementary, offline_mode)

    def resize(value):
        batched = value.unsqueeze(0) if value.ndim == 3 else value
        resized = F.interpolate(
            batched,
            size=tuple(int(item) for item in output_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0) if value.ndim == 3 else resized

    # Reapplying the mode after evidence interpolation preserves the exact
    # historical Clean-DP 68x68 definition for V0 while keeping one shared
    # construction for every ablation.
    mode_68 = _mode_target(
        resize(primary), resize(background), resize(complementary), offline_mode
    )
    outputs = {
        "residual_37": _finish("residual_37_output", residual),
        "p_rw_37": _finish("p_rw_37_output", p_rw),
        "primary_fg_37": _finish("primary_fg_37", primary),
        "background_evidence_37": _finish(
            "background_evidence_37", background
        ),
        "semantic_fg_tendency_37": _finish(
            "semantic_fg_tendency_37_output", semantic
        ),
        "latent_support_37": _finish("latent_support_37", latent),
        "complementary_fg_37": _finish(
            "complementary_fg_37", complementary
        ),
        "complementary_increment_37": _finish(
            "complementary_increment_37",
            mode_37["consolidated_fg"] - primary,
        ),
        "consolidated_fg_37": mode_37["consolidated_fg"],
        "conflict_37": mode_37["conflict"],
        "commitment_37": mode_37["commitment"],
        "target_offline_37": mode_37["target"],
        "target_offline_68": mode_68["target"],
    }
    if not bool((outputs["consolidated_fg_37"] + 1e-7 >= primary).all().item()):
        raise RuntimeError("Noisy-OR must not reduce primary foreground evidence.")
    for name, value in inputs.items():
        if not torch.equal(value.detach(), snapshots[name]):
            raise RuntimeError(f"build_offline_consolidation modified {name} in place.")
    return outputs


def offline_mode_target(primary_fg, background, complementary_fg, mode):
    """Public formula helper used by the cache auditor."""

    return _mode_target(primary_fg, background, complementary_fg, mode)
