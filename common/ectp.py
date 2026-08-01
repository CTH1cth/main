"""Evidence-Constrained Teacher Projection (ECTP).

ECTP is deliberately a target transformation, not a pixel-wise loss router.
It contracts a conflicting binary EMA-Teacher target towards the neutral value
``0.5`` only while the static and Teacher supervision sources overlap.  The
module owns no temporal state and exposes no tunable numerical strength or
schedule.
"""

from __future__ import annotations

import math
from typing import Any

import torch


ECTP_VERSION = "ectp_v1_dual_evidence_overlap_projection"

_FORBIDDEN_CONFIG_FIELDS = (
    "ECTP_STRENGTH",
    "ECTP_WEIGHT_MIN",
    "ECTP_WEIGHT_MAX",
    "ECTP_START_EPOCH",
    "ECTP_RAMP_END_EPOCH",
    "ECTP_STOP_EPOCH",
    "ECTP_FG_STRENGTH",
    "ECTP_BG_STRENGTH",
    "ECTP_USE_HISTORY",
    "ECTP_THRESHOLD",
)

_REQUIRED_CONFIG_VALUES = {
    "USE_ECTP": True,
    "TEACHER_ROUTING_MODE": "ectp",
    "ECTP_VERSION": ECTP_VERSION,
    "USE_ECST": False,
    "USE_ECST_MINIMAL": False,
    "USE_ECST_CLEAN": False,
    "USE_BITC": False,
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": False,
    "USE_DABE_CLEAN": True,
    "USE_DABE_PU": False,
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
    "DABE_CLEAN_DABE_V2_VERSION": "v2",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
    "STATIC_WEIGHT_MODE": "ones",
    "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
    "TEACHER_TARGET_MODE": "binary",
    "USE_DABE_CLEAN_DESPL_SCHEDULE": True,
    "USE_TEACHER_BINARY_FULL_LOSS": True,
    "USE_DAGP_SAFE_HEAD": True,
    "USE_NDR_BRANCH": True,
    "FINETUNE_RESET_EPOCH": 20,
    "MAX_EPOCH": 45,
    "STOP_AFTER_EPOCH": 0,
}

# A missing false feature flag is equivalent to the repository-wide default
# (disabled).  Every positive flag and every protocol value remains mandatory.
_OPTIONAL_FALSE_FLAGS = {
    "USE_ECST",
    "USE_ECST_MINIMAL",
    "USE_ECST_CLEAN",
    "USE_BITC",
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS",
    "USE_DABE_PU",
}


def _config_value(cfg: Any, name: str) -> Any:
    if name in _OPTIONAL_FALSE_FLAGS:
        return getattr(cfg, name, False)
    if not hasattr(cfg, name):
        raise RuntimeError(f"ECTP config is missing required field {name}.")
    return getattr(cfg, name)


def _config_value_matches(name: str, actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if name == "DABE_CLEAN_DABE_V2_HARD_THRESHOLD":
        if isinstance(actual, bool):
            return False
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value == 0.5
    return actual == expected


def validate_ectp_config(cfg: Any) -> dict[str, Any]:
    """Validate the parameter-free ECTP protocol before data/model creation.

    The returned dictionary is a small startup-audit record.  Validation is
    intentionally strict: adding even an unused ECTP strength, threshold,
    history, or private schedule field is an error.
    """

    errors: list[str] = []
    for name in _FORBIDDEN_CONFIG_FIELDS:
        if hasattr(cfg, name):
            errors.append(f"forbidden ECTP config field is present: {name}")

    resolved: dict[str, Any] = {}
    for name, expected in _REQUIRED_CONFIG_VALUES.items():
        try:
            actual = _config_value(cfg, name)
        except RuntimeError as error:
            errors.append(str(error))
            continue
        resolved[name] = actual
        if not _config_value_matches(name, actual, expected):
            errors.append(
                f"ECTP config mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    if errors:
        raise RuntimeError("ECTP config validation failed: " + "; ".join(errors))

    allow_foreground_cache_drift = getattr(
        cfg,
        "ECTP_ALLOW_FOREGROUND_CACHE_DRIFT",
        False,
    )
    if not isinstance(allow_foreground_cache_drift, bool):
        raise RuntimeError(
            "ECTP_ALLOW_FOREGROUND_CACHE_DRIFT must be a bool when present."
        )

    return {
        "schema": "ectp_config_audit_v1",
        "status": "PASS",
        "version": ECTP_VERSION,
        "routing_mode": "ectp",
        "extra_numeric_parameters": [],
        "temporal_history": False,
        "legacy_regions": False,
        "pixel_weight_map": "identity",
        "independent_schedule": False,
        "allow_foreground_cache_drift": allow_foreground_cache_drift,
        "resolved": resolved,
    }


def _require_ectp_map(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"ECTP {name} must be a torch.Tensor.")
    if tuple(value.shape[1:]) != (1, 68, 68) or value.ndim != 4:
        raise RuntimeError(
            f"ECTP {name} must have shape [B,1,68,68], "
            f"got {list(value.shape)}."
        )
    if int(value.shape[0]) < 1:
        raise RuntimeError(f"ECTP {name} batch dimension must be positive.")
    if not value.is_floating_point():
        raise RuntimeError(f"ECTP {name} must be floating point.")
    if bool(value.requires_grad):
        raise RuntimeError(f"ECTP {name} must be detached (requires_grad=False).")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"ECTP {name} contains NaN or Inf.")
    return value


def _require_unit_interval(name: str, value: torch.Tensor) -> None:
    minimum = float(value.min().item())
    maximum = float(value.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(
            f"ECTP {name} must be in [0,1], got "
            f"min={minimum:.9g}, max={maximum:.9g}."
        )


def _require_binary(name: str, value: torch.Tensor) -> None:
    if not bool(((value == 0.0) | (value == 1.0)).all().item()):
        raise RuntimeError(f"ECTP {name} must contain exact binary values {{0,1}}.")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return 0.0
    return float(value.masked_select(mask).mean().item())


def _finite_schedule_weights(
    static_weight: float,
    teacher_weight: float,
) -> tuple[float, float, float]:
    try:
        alpha = float(static_weight)
        beta = float(teacher_weight)
    except (TypeError, ValueError) as error:
        raise RuntimeError("ECTP supervision weights must be numeric scalars.") from error
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise RuntimeError(
            f"ECTP supervision weights must be finite, got {alpha}/{beta}."
        )
    if alpha < 0.0 or beta < 0.0:
        raise RuntimeError(
            f"ECTP supervision weights must be non-negative, got {alpha}/{beta}."
        )
    if abs((alpha + beta) - 1.0) > 1e-6:
        raise RuntimeError(
            "ECTP requires the effective global static/Teacher weights to sum "
            f"to one, got {alpha} + {beta} = {alpha + beta}."
        )
    overlap = max(0.0, min(1.0, 4.0 * alpha * beta))
    return alpha, beta, overlap


def build_ectp_projected_target(
    *,
    foreground_evidence: torch.Tensor,
    background_evidence: torch.Tensor,
    static_target: torch.Tensor,
    teacher_binary: torch.Tensor,
    static_weight: float,
    teacher_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the symmetric, evidence-constrained effective Teacher target.

    ``static_target`` must be the exact hardening of ``foreground_evidence`` at
    ``> 0.5``.  All tensor outputs in the diagnostic dictionary are detached
    and may therefore be reused by logging/audit code without creating a graph.
    """

    inputs = {
        "foreground_evidence": _require_ectp_map(
            "foreground_evidence", foreground_evidence
        ),
        "background_evidence": _require_ectp_map(
            "background_evidence", background_evidence
        ),
        "static_target": _require_ectp_map("static_target", static_target),
        "teacher_binary": _require_ectp_map("teacher_binary", teacher_binary),
    }
    reference_shape = tuple(foreground_evidence.shape)
    reference_device = foreground_evidence.device
    snapshots: dict[str, torch.Tensor] = {}
    for name, value in inputs.items():
        if tuple(value.shape) != reference_shape:
            raise RuntimeError(
                f"ECTP input shape mismatch: {name}={list(value.shape)}, "
                f"expected={list(reference_shape)}."
            )
        if value.device != reference_device:
            raise RuntimeError(
                f"ECTP input device mismatch: {name}={value.device}, "
                f"expected={reference_device}."
            )
        snapshots[name] = value.detach().clone()

    _require_unit_interval("foreground_evidence", foreground_evidence)
    _require_unit_interval("background_evidence", background_evidence)
    _require_unit_interval("static_target", static_target)
    _require_unit_interval("teacher_binary", teacher_binary)
    _require_binary("static_target", static_target)
    _require_binary("teacher_binary", teacher_binary)

    expected_static = (foreground_evidence > 0.5).to(dtype=static_target.dtype)
    static_target_consistency_error = float(
        (static_target - expected_static).abs().max().item()
    )
    if not torch.equal(static_target, expected_static):
        raise RuntimeError(
            "ECTP static target is not exactly "
            "(foreground_evidence > 0.5).float(); "
            f"max_abs_error={static_target_consistency_error:.9g}."
        )

    alpha, beta, overlap = _finite_schedule_weights(
        static_weight, teacher_weight
    )

    # Compute in float32 even when cache tensors are stored in float16.  The
    # binary targets remain exact after conversion.
    foreground = foreground_evidence.detach().float()
    background = background_evidence.detach().float()
    static = static_target.detach().float()
    teacher = teacher_binary.detach().float()

    support = (
        static * foreground * (1.0 - background)
        + (1.0 - static) * background * (1.0 - foreground)
    ).clamp(0.0, 1.0)

    conflict_bool = teacher.ne(static)
    conflict = conflict_bool.float()
    fg_to_bg_bool = (static > 0.5) & (teacher < 0.5)
    bg_to_fg_bool = (static < 0.5) & (teacher > 0.5)

    teacher_sign = 2.0 * teacher - 1.0
    projection_factor = (1.0 - overlap * support * conflict).clamp(0.0, 1.0)
    projected_sign = teacher_sign * projection_factor
    projected_target_raw = (0.5 + 0.5 * projected_sign).clamp(0.0, 1.0)
    projected_target = torch.where(
        conflict_bool,
        projected_target_raw,
        teacher,
    ).detach()

    if bool(projected_target.requires_grad):
        raise RuntimeError("ECTP projected target unexpectedly requires gradients.")
    if not bool(torch.isfinite(projected_target).all().item()):
        raise RuntimeError("ECTP projected target contains NaN or Inf.")
    if float(projected_target.min().item()) < -1e-6 or float(
        projected_target.max().item()
    ) > 1.0 + 1e-6:
        raise RuntimeError("ECTP projected target is outside [0,1].")
    if not torch.equal(
        projected_target[~conflict_bool], teacher[~conflict_bool]
    ):
        raise RuntimeError("ECTP changed a non-conflicting Teacher target.")
    if not bool(
        (projected_target[teacher > 0.5] >= 0.5 - 1e-6).all().item()
    ):
        raise RuntimeError("ECTP projected a Teacher-FG target below 0.5.")
    if not bool(
        (projected_target[teacher < 0.5] <= 0.5 + 1e-6).all().item()
    ):
        raise RuntimeError("ECTP projected a Teacher-BG target above 0.5.")
    if overlap == 0.0 and not torch.equal(projected_target, teacher):
        raise RuntimeError("ECTP changed Teacher targets while overlap_rho is zero.")

    for name, value in inputs.items():
        if not torch.equal(value, snapshots[name]):
            raise RuntimeError(f"ECTP modified input tensor in-place: {name}.")

    absolute_shift = (projected_target - teacher).abs()
    signed_shift = projected_target - teacher
    fg_protection = torch.where(
        fg_to_bg_bool, projected_target - teacher, torch.zeros_like(teacher)
    )
    bg_protection = torch.where(
        bg_to_fg_bool, teacher - projected_target, torch.zeros_like(teacher)
    )
    pixel_count = int(projected_target.numel())
    conflict_count = int(conflict_bool.sum().item())
    static_fg_bool = static > 0.5
    static_bg_bool = ~static_fg_bool

    # ``*_mass`` is reported as mass density (sum divided by all pixels), so
    # it is comparable across different batch sizes.  Raw sums/counts remain
    # available to the epoch accumulator for exact dataset-wide aggregation.
    stats: dict[str, Any] = {
        "version": ECTP_VERSION,
        "foreground_evidence": foreground.detach(),
        "background_evidence": background.detach(),
        "static_target": static.detach(),
        "teacher_binary": teacher.detach(),
        "support": support.detach(),
        "conflict": conflict.detach(),
        "conflict_bool": conflict_bool.detach(),
        "fg_to_bg_conflict": fg_to_bg_bool.detach(),
        "bg_to_fg_conflict": bg_to_fg_bool.detach(),
        "projection_factor": projection_factor.detach(),
        "projected_target_raw": projected_target_raw.detach(),
        "projected_target": projected_target,
        "absolute_shift": absolute_shift.detach(),
        "fg_protection": fg_protection.detach(),
        "bg_protection": bg_protection.detach(),
        "alpha": alpha,
        "beta": beta,
        "static_weight": alpha,
        "teacher_weight": beta,
        "overlap": overlap,
        "overlap_rho": overlap,
        "support_min": float(support.min().item()),
        "support_mean": float(support.mean().item()),
        "support_max": float(support.max().item()),
        "support_on_static_fg_mean": _masked_mean(support, static_fg_bool),
        "support_on_static_bg_mean": _masked_mean(support, static_bg_bool),
        "static_fg_ratio": float(static.mean().item()),
        "static_fg_area": float(static.mean().item()),
        "teacher_binary_fg_ratio": float(teacher.mean().item()),
        "teacher_fg_ratio": float(teacher.mean().item()),
        "teacher_binary_area": float(teacher.mean().item()),
        "conflict_ratio": float(conflict.mean().item()),
        "fg_to_bg_conflict_ratio": float(fg_to_bg_bool.float().mean().item()),
        "bg_to_fg_conflict_ratio": float(bg_to_fg_bool.float().mean().item()),
        "support_on_conflict_mean": _masked_mean(support, conflict_bool),
        "conflict_support_mean": _masked_mean(support, conflict_bool),
        "projected_target_min": float(projected_target.min().item()),
        "projected_target_mean": float(projected_target.mean().item()),
        "projected_target_max": float(projected_target.max().item()),
        "projected_target_shift_mean": float(absolute_shift.mean().item()),
        "projected_target_shift_abs_mean": float(absolute_shift.mean().item()),
        "projected_shift_abs_mean": float(absolute_shift.mean().item()),
        "projected_target_minus_binary_mean": float(signed_shift.mean().item()),
        "projection_mass": float(absolute_shift.mean().item()),
        "neutralized_conflict_mean": _masked_mean(
            absolute_shift, conflict_bool
        ),
        "fg_protection_mass": float(fg_protection.mean().item()),
        "bg_protection_mass": float(bg_protection.mean().item()),
        "nonconflict_exact_match": bool(
            torch.equal(
                projected_target[~conflict_bool], teacher[~conflict_bool]
            )
        ),
        "static_target_consistency_error": static_target_consistency_error,
        "temporal_memory_initialized": False,
        "legacy_regions_used": False,
        "pixel_weight_map": "identity",
        "pixel_count": pixel_count,
        "conflict_count": conflict_count,
    }
    return projected_target, stats


def new_ectp_epoch_accumulator() -> dict[str, Any]:
    """Create a state-free, pixel-weighted ECTP epoch accumulator."""

    return {
        "schema": "ectp_epoch_accumulator_v1",
        "batch_count": 0,
        "pixel_count": 0,
        "conflict_count": 0,
        "fg_to_bg_conflict_count": 0,
        "bg_to_fg_conflict_count": 0,
        "sums": {
            "static_weight": 0.0,
            "teacher_weight": 0.0,
            "overlap_rho": 0.0,
            "static_fg": 0.0,
            "teacher_fg": 0.0,
            "support": 0.0,
            "conflict_support": 0.0,
            "projected_target": 0.0,
            "projected_shift_abs": 0.0,
            "fg_protection": 0.0,
            "bg_protection": 0.0,
            "student_pred_area": 0.0,
            "teacher_pred_area": 0.0,
        },
        "student_area_pixel_count": 0,
        "teacher_area_pixel_count": 0,
    }


def _prediction_area(value: Any, name: str) -> float:
    if torch.is_tensor(value):
        detached = value.detach()
        if detached.numel() == 1:
            result = float(detached.item())
        else:
            if detached.ndim != 4 or int(detached.shape[1]) != 1:
                raise RuntimeError(
                    f"ECTP {name} tensor must be scalar or [B,1,H,W]."
                )
            if not bool(torch.isfinite(detached).all().item()):
                raise RuntimeError(f"ECTP {name} tensor contains NaN or Inf.")
            result = float((detached > 0.5).float().mean().item())
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"ECTP {name} must be numeric.") from error
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise RuntimeError(f"ECTP {name} must be finite and in [0,1].")
    return result


def accumulate_ectp_epoch(
    accumulator: dict[str, Any] | None,
    stats: dict[str, Any],
    student_pred_area: Any | None = None,
    teacher_pred_area: Any | None = None,
) -> None:
    """Accumulate one ECTP batch without retaining GPU tensors.

    Prediction areas may be scalar ratios or probability maps.  Maps are
    binarized at ``> 0.5``, matching the repository's area diagnostics.
    ``teacher_pred_area`` defaults to the original binary Teacher area and
    never to the projected-target mean.
    """

    if accumulator is None:
        return
    required_tensors = (
        "static_target",
        "teacher_binary",
        "support",
        "conflict_bool",
        "fg_to_bg_conflict",
        "bg_to_fg_conflict",
        "projected_target",
        "absolute_shift",
        "fg_protection",
        "bg_protection",
    )
    missing = [name for name in required_tensors if name not in stats]
    if missing:
        raise RuntimeError(f"ECTP epoch stats are missing fields: {missing}")

    tensors = {name: stats[name].detach().float() for name in required_tensors}
    shape = tuple(tensors["projected_target"].shape)
    for name, value in tensors.items():
        if tuple(value.shape) != shape:
            raise RuntimeError(f"ECTP epoch tensor shape mismatch for {name}.")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"ECTP epoch tensor is not finite: {name}.")

    pixel_count = int(tensors["projected_target"].numel())
    conflict_count = int(tensors["conflict_bool"].sum().item())
    fg_to_bg_count = int(tensors["fg_to_bg_conflict"].sum().item())
    bg_to_fg_count = int(tensors["bg_to_fg_conflict"].sum().item())
    alpha, beta, overlap = _finite_schedule_weights(
        stats.get("static_weight", stats.get("alpha")),
        stats.get("teacher_weight", stats.get("beta")),
    )
    sums = accumulator["sums"]
    sums["static_weight"] += alpha * pixel_count
    sums["teacher_weight"] += beta * pixel_count
    sums["overlap_rho"] += overlap * pixel_count
    sums["static_fg"] += float(tensors["static_target"].sum().item())
    sums["teacher_fg"] += float(tensors["teacher_binary"].sum().item())
    sums["support"] += float(tensors["support"].sum().item())
    sums["conflict_support"] += float(
        tensors["support"].masked_select(
            tensors["conflict_bool"].bool()
        ).sum().item()
    )
    sums["projected_target"] += float(tensors["projected_target"].sum().item())
    sums["projected_shift_abs"] += float(tensors["absolute_shift"].sum().item())
    sums["fg_protection"] += float(tensors["fg_protection"].sum().item())
    sums["bg_protection"] += float(tensors["bg_protection"].sum().item())

    if student_pred_area is not None:
        value = _prediction_area(student_pred_area, "student_pred_area")
        sums["student_pred_area"] += value * pixel_count
        accumulator["student_area_pixel_count"] += pixel_count
    if teacher_pred_area is None:
        teacher_pred_area = float(tensors["teacher_binary"].mean().item())
    value = _prediction_area(teacher_pred_area, "teacher_pred_area")
    sums["teacher_pred_area"] += value * pixel_count
    accumulator["teacher_area_pixel_count"] += pixel_count

    accumulator["batch_count"] += 1
    accumulator["pixel_count"] += pixel_count
    accumulator["conflict_count"] += conflict_count
    accumulator["fg_to_bg_conflict_count"] += fg_to_bg_count
    accumulator["bg_to_fg_conflict_count"] += bg_to_fg_count


def finalize_ectp_epoch(accumulator: dict[str, Any]) -> dict[str, Any]:
    """Finalize pixel-weighted ECTP epoch diagnostics."""

    batch_count = int(accumulator.get("batch_count", 0))
    pixel_count = int(accumulator.get("pixel_count", 0))
    if batch_count < 1 or pixel_count < 1:
        raise RuntimeError("ECTP epoch accumulator is empty.")
    conflict_count = int(accumulator.get("conflict_count", 0))
    fg_to_bg_count = int(accumulator.get("fg_to_bg_conflict_count", 0))
    bg_to_fg_count = int(accumulator.get("bg_to_fg_conflict_count", 0))
    sums = accumulator["sums"]
    student_pixels = int(accumulator.get("student_area_pixel_count", 0))
    teacher_pixels = int(accumulator.get("teacher_area_pixel_count", 0))

    row = {
        "batch_count": batch_count,
        "pixel_count": pixel_count,
        "static_weight": sums["static_weight"] / pixel_count,
        "teacher_weight": sums["teacher_weight"] / pixel_count,
        "overlap_rho": sums["overlap_rho"] / pixel_count,
        "static_fg_ratio": sums["static_fg"] / pixel_count,
        "static_fg_area": sums["static_fg"] / pixel_count,
        "teacher_fg_ratio": sums["teacher_fg"] / pixel_count,
        "teacher_binary_area": sums["teacher_fg"] / pixel_count,
        "conflict_ratio": conflict_count / pixel_count,
        "fg_to_bg_conflict_ratio": fg_to_bg_count / pixel_count,
        "bg_to_fg_conflict_ratio": bg_to_fg_count / pixel_count,
        "support_mean": sums["support"] / pixel_count,
        "conflict_support_mean": (
            sums["conflict_support"] / conflict_count
            if conflict_count > 0
            else 0.0
        ),
        "projected_target_mean": sums["projected_target"] / pixel_count,
        "projected_shift_abs_mean": sums["projected_shift_abs"] / pixel_count,
        "projected_target_shift_mean": sums["projected_shift_abs"]
        / pixel_count,
        "projection_mass": sums["projected_shift_abs"] / pixel_count,
        "neutralized_conflict_mean": (
            sums["projected_shift_abs"] / conflict_count
            if conflict_count > 0
            else 0.0
        ),
        "fg_protection_mass": sums["fg_protection"] / pixel_count,
        "bg_protection_mass": sums["bg_protection"] / pixel_count,
        "student_pred_area": (
            sums["student_pred_area"] / student_pixels
            if student_pixels > 0
            else None
        ),
        "teacher_pred_area": (
            sums["teacher_pred_area"] / teacher_pixels
            if teacher_pixels > 0
            else None
        ),
    }
    row["alpha"] = row["static_weight"]
    row["beta"] = row["teacher_weight"]
    row["overlap"] = row["overlap_rho"]
    return row
