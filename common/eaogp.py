"""Evidence-Anchored Online Graph Projection (EAOGP-v1).

EAOGP is a detached Teacher-target transformation.  It reuses the sparse raw
DINO Top-K graph already built by DAGP-Safe, validates those candidate edges
with the current EMA-Teacher projection embedding, and anchors the resulting
online graph target with evidence from the same DABE-v2 payload.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


EAOGP_VERSION = "eaogp_v1_dabe_anchor_dualgraph_projection"
EAOGP_EPS = 1e-6
EAOGP_RESIDUAL_SOURCE_KEY = "residual_37"

_FORBIDDEN_CONFIG_FIELDS = (
    "EAOGP_STRENGTH",
    "EAOGP_WEIGHT",
    "EAOGP_FG_STRENGTH",
    "EAOGP_BG_STRENGTH",
    "EAOGP_START_EPOCH",
    "EAOGP_RAMP_END_EPOCH",
    "EAOGP_STOP_EPOCH",
    "EAOGP_TAU",
    "EAOGP_TOPK",
    "EAOGP_THRESHOLD",
    "EAOGP_HISTORY",
    "EAOGP_MEMORY",
    "EAOGP_AREA_TARGET",
)

_REQUIRED_CONFIG_VALUES = {
    "USE_EAOGP": True,
    "TEACHER_ROUTING_MODE": "eaogp",
    "EAOGP_VERSION": EAOGP_VERSION,
    "USE_ECTP": False,
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
    "HEAD_TYPE": "dagp_safe",
    "USE_DAGP_SAFE_HEAD": True,
    "USE_NDR_BRANCH": True,
    "FINETUNE_RESET_EPOCH": 20,
    "FINETUNE_RESET_TEACHER": True,
    "MAX_EPOCH": 45,
    "STOP_AFTER_EPOCH": 0,
    "DAGP_SAFE_TOPK": 12,
    "DAGP_SAFE_HIDDEN": 64,
    "DAGP_SAFE_TAU": 0.07,
    "DAGP_SAFE_WARMUP_EPOCH": 6,
    "DAGP_SAFE_RAMP_START_EPOCH": 7,
    "DAGP_SAFE_RAMP_END_EPOCH": 15,
}

_OPTIONAL_FALSE_FLAGS = {
    "USE_ECTP",
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
        raise RuntimeError(f"EAOGP config is missing required field {name}.")
    return getattr(cfg, name)


def _config_value_matches(name: str, actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value == expected
    if isinstance(expected, int):
        if isinstance(actual, bool):
            return False
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value.is_integer() and int(value) == expected
    return str(actual).strip().lower() == str(expected).strip().lower()


def validate_eaogp_config(cfg: Any) -> dict[str, Any]:
    """Validate the parameter-free EAOGP-v1 protocol."""

    errors: list[str] = []
    for name in _FORBIDDEN_CONFIG_FIELDS:
        if hasattr(cfg, name):
            errors.append(f"forbidden EAOGP config field is present: {name}")
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
                f"EAOGP config mismatch: {name}={actual!r}, expected={expected!r}"
            )
    if errors:
        raise RuntimeError("EAOGP config validation failed: " + "; ".join(errors))
    return {
        "schema": "eaogp_config_audit_v1",
        "status": "PASS",
        "version": EAOGP_VERSION,
        "routing_mode": "eaogp",
        "new_trainable_parameters": 0,
        "temporal_history": False,
        "legacy_regions": False,
        "future_teacher": False,
        "pixel_weight_map": "identity",
        "independent_schedule": False,
        "dabe_residual_source_key": EAOGP_RESIDUAL_SOURCE_KEY,
        "resolved": resolved,
    }


def get_eaogp_scale(cfg: Any, epoch: int) -> float:
    """Return the scale expected from the existing DAGP-Safe ramp."""

    warmup = int(getattr(cfg, "DAGP_SAFE_WARMUP_EPOCH"))
    ramp_start = int(getattr(cfg, "DAGP_SAFE_RAMP_START_EPOCH"))
    ramp_end = int(getattr(cfg, "DAGP_SAFE_RAMP_END_EPOCH"))
    epoch = int(epoch)
    if epoch <= warmup or epoch < ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    denominator = max(1, ramp_end - ramp_start + 1)
    return float(epoch - ramp_start + 1) / float(denominator)


def build_same_source_background_evidence(
    bc_map_37: torch.Tensor,
    residual_norm_37: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build B37 and bilinear B68 from one DABE-v2 payload."""

    bc_map_37 = _require_float_map(
        "bc_map_37", bc_map_37, channels=1, height=37, width=37
    )
    residual_norm_37 = _require_float_map(
        "residual_norm_37",
        residual_norm_37,
        channels=1,
        height=37,
        width=37,
    )
    _require_unit_interval("bc_map_37", bc_map_37)
    _require_unit_interval("residual_norm_37", residual_norm_37)
    background_37 = (
        bc_map_37.clamp(0.0, 1.0)
        * (1.0 - residual_norm_37.clamp(0.0, 1.0))
    ).clamp(0.0, 1.0).detach()
    background_68 = F.interpolate(
        background_37,
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()
    return background_37, background_68


def _require_float_map(
    name: str,
    value: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"EAOGP {name} must be a tensor.")
    if value.ndim != 4 or tuple(value.shape[1:]) != (channels, height, width):
        raise RuntimeError(
            f"EAOGP {name} must be [B,{channels},{height},{width}], "
            f"got {list(value.shape)}."
        )
    if int(value.shape[0]) < 1 or not value.is_floating_point():
        raise RuntimeError(f"EAOGP {name} must be a non-empty floating tensor.")
    if value.requires_grad:
        raise RuntimeError(f"EAOGP {name} must be detached.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"EAOGP {name} contains NaN/Inf.")
    return value


def _require_unit_interval(name: str, value: torch.Tensor) -> None:
    minimum = float(value.min().item())
    maximum = float(value.max().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise RuntimeError(
            f"EAOGP {name} must be in [0,1], got {minimum:.9g}/{maximum:.9g}."
        )


def _gather_neighbors(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_nodes, channels = nodes.shape
    k = int(indices.shape[-1])
    offsets = (
        torch.arange(
            batch_size,
            device=indices.device,
            dtype=indices.dtype,
        ).view(batch_size, 1, 1)
        * num_nodes
    )
    flat_indices = (indices + offsets).reshape(-1)
    return (
        nodes.reshape(batch_size * num_nodes, channels)
        .index_select(0, flat_indices)
        .reshape(batch_size, num_nodes, k, channels)
    )


def build_dual_graph_weights(
    dino_topk_idx: torch.Tensor,
    dino_topk_weight: torch.Tensor,
    teacher_embedding_37: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validate DINO candidate edges with detached EMA-Teacher semantics."""

    if not torch.is_tensor(dino_topk_idx) or dino_topk_idx.ndim != 3:
        raise RuntimeError("EAOGP dino_topk_idx must be [B,1369,K].")
    if dino_topk_idx.requires_grad:
        raise RuntimeError("EAOGP dino_topk_idx must be detached.")
    if dino_topk_idx.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("EAOGP dino_topk_idx must use an integer dtype.")
    if not torch.is_tensor(dino_topk_weight):
        raise TypeError("EAOGP dino_topk_weight must be a tensor.")
    if tuple(dino_topk_weight.shape) != tuple(dino_topk_idx.shape):
        raise RuntimeError("EAOGP DINO Top-K index/weight shapes differ.")
    if dino_topk_weight.requires_grad or not dino_topk_weight.is_floating_point():
        raise RuntimeError("EAOGP DINO Top-K weight must be detached floating point.")
    if not bool(torch.isfinite(dino_topk_weight).all().item()):
        raise RuntimeError("EAOGP DINO Top-K weight contains NaN/Inf.")
    teacher_embedding_37 = _require_float_map(
        "teacher_embedding_37",
        teacher_embedding_37,
        channels=64,
        height=37,
        width=37,
    )
    batch_size, num_nodes, k = dino_topk_idx.shape
    if num_nodes != 37 * 37 or k != 12:
        raise RuntimeError(
            "EAOGP requires the existing DAGP graph [B,1369,12], got "
            f"{list(dino_topk_idx.shape)}."
        )
    if int(teacher_embedding_37.shape[0]) != batch_size:
        raise RuntimeError("EAOGP Teacher embedding batch size differs from graph.")
    if int(dino_topk_idx.min().item()) < 0 or int(dino_topk_idx.max().item()) >= num_nodes:
        raise RuntimeError("EAOGP DINO Top-K index is out of range.")
    if float(dino_topk_weight.min().item()) < 0.0:
        raise RuntimeError("EAOGP DINO Top-K weights must be non-negative.")
    dino_row_sum_error = (
        dino_topk_weight.float().sum(dim=-1) - 1.0
    ).abs().max()
    if float(dino_row_sum_error.item()) > 1e-5:
        raise RuntimeError(
            "EAOGP raw DINO weights are not row-normalized: "
            f"max_error={float(dino_row_sum_error.item()):.9g}."
        )

    teacher_nodes = teacher_embedding_37.flatten(2).transpose(1, 2).float()
    teacher_norm = torch.linalg.vector_norm(
        teacher_nodes,
        ord=2,
        dim=-1,
        keepdim=True,
    )
    teacher_nodes = teacher_nodes / (teacher_norm + EAOGP_EPS)
    teacher_neighbors = _gather_neighbors(teacher_nodes, dino_topk_idx.long())
    cosine = (teacher_nodes.unsqueeze(2) * teacher_neighbors).sum(dim=-1)
    teacher_task_gate = ((1.0 + cosine) * 0.5).clamp(0.0, 1.0)
    unnormalized = dino_topk_weight.float() * teacher_task_gate
    unnormalized_sum = unnormalized.sum(dim=-1, keepdim=True)
    fallback = unnormalized_sum < EAOGP_EPS
    normalized = unnormalized / unnormalized_sum.clamp_min(EAOGP_EPS)
    dual_weight = torch.where(
        fallback,
        dino_topk_weight.float(),
        normalized,
    ).detach()
    dual_row_sum_error = (dual_weight.sum(dim=-1) - 1.0).abs().max()
    if float(dual_row_sum_error.item()) > 1e-5:
        raise RuntimeError(
            "EAOGP dual graph weights are not row-normalized: "
            f"max_error={float(dual_row_sum_error.item()):.9g}."
        )
    return {
        "dual_graph_weight": dual_weight,
        "teacher_task_gate": teacher_task_gate.detach(),
        "fallback_row_mask": fallback.squeeze(-1).detach(),
        "dino_row_sum_error": dino_row_sum_error.detach(),
        "dual_row_sum_error": dual_row_sum_error.detach(),
    }


def _graph_propagate(
    probability_nodes: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    neighbors = _gather_neighbors(probability_nodes, topk_idx.long())
    return (topk_weight.float().unsqueeze(-1) * neighbors.float()).sum(dim=2)


def _entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.float().clamp_min(EAOGP_EPS)
    return -(weights.float() * safe.log()).sum(dim=-1)


def build_eaogp_target(
    foreground_response_68: torch.Tensor,
    background_evidence_68: torch.Tensor,
    static_target_68: torch.Tensor,
    teacher_prob_68: torch.Tensor,
    teacher_binary_68: torch.Tensor,
    teacher_embedding_37: torch.Tensor,
    dino_topk_idx: torch.Tensor,
    dino_topk_weight: torch.Tensor,
    scale: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build one detached EAOGP Teacher target for all prediction branches."""

    foreground_response_68 = _require_float_map(
        "foreground_response_68",
        foreground_response_68,
        channels=1,
        height=68,
        width=68,
    )
    background_evidence_68 = _require_float_map(
        "background_evidence_68",
        background_evidence_68,
        channels=1,
        height=68,
        width=68,
    )
    static_target_68 = _require_float_map(
        "static_target_68", static_target_68, channels=1, height=68, width=68
    )
    teacher_prob_68 = _require_float_map(
        "teacher_prob_68", teacher_prob_68, channels=1, height=68, width=68
    )
    teacher_binary_68 = _require_float_map(
        "teacher_binary_68", teacher_binary_68, channels=1, height=68, width=68
    )
    shapes = {
        tuple(value.shape)
        for value in (
            foreground_response_68,
            background_evidence_68,
            static_target_68,
            teacher_prob_68,
            teacher_binary_68,
        )
    }
    if len(shapes) != 1:
        raise RuntimeError(f"EAOGP 68-grid input shapes differ: {sorted(shapes)}")
    for name, value in (
        ("foreground_response_68", foreground_response_68),
        ("background_evidence_68", background_evidence_68),
        ("static_target_68", static_target_68),
        ("teacher_prob_68", teacher_prob_68),
        ("teacher_binary_68", teacher_binary_68),
    ):
        _require_unit_interval(name, value)
    expected_static = (foreground_response_68 > 0.5).float()
    if not torch.equal(static_target_68, expected_static):
        raise RuntimeError(
            "EAOGP static target must equal strict 1[DABE-v2 soft > 0.5]."
        )
    expected_teacher = (teacher_prob_68 >= 0.5).float()
    if not torch.equal(teacher_binary_68, expected_teacher):
        raise RuntimeError(
            "EAOGP binary Teacher must equal 1[Teacher probability >= 0.5]."
        )
    scale_value = float(scale.detach().item()) if torch.is_tensor(scale) else float(scale)
    if not math.isfinite(scale_value) or scale_value < 0.0 or scale_value > 1.0:
        raise RuntimeError(f"EAOGP scale must be finite in [0,1], got {scale_value}.")

    graph = build_dual_graph_weights(
        dino_topk_idx,
        dino_topk_weight,
        teacher_embedding_37,
    )
    teacher_prob_37 = F.interpolate(
        teacher_prob_68,
        size=(37, 37),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()
    probability_nodes = teacher_prob_37.flatten(2).transpose(1, 2)
    dino_nodes = _graph_propagate(
        probability_nodes,
        dino_topk_idx,
        dino_topk_weight,
    )
    dual_nodes = _graph_propagate(
        probability_nodes,
        dino_topk_idx,
        graph["dual_graph_weight"],
    )
    batch_size = int(teacher_prob_68.shape[0])
    dino_consensus_37 = dino_nodes.transpose(1, 2).reshape(
        batch_size, 1, 37, 37
    ).clamp(0.0, 1.0).detach()
    dual_consensus_37 = dual_nodes.transpose(1, 2).reshape(
        batch_size, 1, 37, 37
    ).clamp(0.0, 1.0).detach()
    dino_consensus_68 = F.interpolate(
        dino_consensus_37,
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()
    dual_consensus_68 = F.interpolate(
        dual_consensus_37,
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()

    anchor_confidence = (
        static_target_68 * foreground_response_68 * (1.0 - background_evidence_68)
        + (1.0 - static_target_68)
        * background_evidence_68
        * (1.0 - foreground_response_68)
    ).clamp(0.0, 1.0).detach()
    teacher_uncertainty = (
        4.0 * teacher_prob_68 * (1.0 - teacher_prob_68)
    ).clamp(0.0, 1.0).detach()
    online_target = (
        (1.0 - teacher_uncertainty) * teacher_binary_68
        + teacher_uncertainty * dual_consensus_68
    ).clamp(0.0, 1.0).detach()
    raw_target = (
        anchor_confidence * foreground_response_68
        + (1.0 - anchor_confidence) * online_target
    ).clamp(0.0, 1.0).detach()
    if scale_value == 0.0:
        effective_target = teacher_binary_68.detach().clone()
    else:
        effective_target = (
            teacher_binary_68
            + teacher_binary_68.new_tensor(scale_value)
            * (raw_target - teacher_binary_68)
        ).clamp(0.0, 1.0).detach()
    if effective_target.requires_grad or not bool(torch.isfinite(effective_target).all().item()):
        raise RuntimeError("EAOGP effective Teacher target must be finite and detached.")
    _require_unit_interval("effective_teacher_target", effective_target)
    if scale_value == 0.0 and not torch.equal(effective_target, teacher_binary_68):
        raise RuntimeError("EAOGP scale=0 must exactly preserve binary Teacher target.")

    anchor_delta = (
        effective_target.new_tensor(scale_value)
        * anchor_confidence
        * (foreground_response_68 - teacher_binary_68)
    ).detach()
    graph_delta = (
        effective_target.new_tensor(scale_value)
        * (1.0 - anchor_confidence)
        * teacher_uncertainty
        * (dual_consensus_68 - teacher_binary_68)
    ).detach()
    effective_hard = (effective_target >= 0.5).float()
    new_foreground = (
        (effective_hard > 0.5) & (teacher_binary_68 < 0.5)
    ).float().detach()
    new_background = (
        (effective_hard < 0.5) & (teacher_binary_68 > 0.5)
    ).float().detach()
    static_fg = static_target_68 > 0.5
    static_bg = ~static_fg
    result: dict[str, Any] = {
        "scale": scale_value,
        "foreground_response_68": foreground_response_68.detach(),
        "background_evidence_68": background_evidence_68.detach(),
        "anchor_confidence_68": anchor_confidence,
        "teacher_uncertainty_68": teacher_uncertainty,
        "dino_graph_consensus_37": dino_consensus_37,
        "dino_graph_consensus_68": dino_consensus_68,
        "dual_graph_consensus_37": dual_consensus_37,
        "dual_graph_consensus_68": dual_consensus_68,
        "online_target_68": online_target,
        "raw_eaogp_target_68": raw_target,
        "effective_teacher_target_68": effective_target,
        "new_foreground_68": new_foreground,
        "new_background_68": new_background,
        "anchor_delta_68": anchor_delta,
        "graph_delta_68": graph_delta,
        "teacher_task_gate": graph["teacher_task_gate"],
        "dual_graph_weight": graph["dual_graph_weight"],
        "dino_row_sum_error": float(graph["dino_row_sum_error"].item()),
        "dual_row_sum_error": float(graph["dual_row_sum_error"].item()),
        "fallback_row_ratio": float(graph["fallback_row_mask"].float().mean().item()),
    }
    anchor_mass = float(
        (effective_target.new_tensor(scale_value) * anchor_confidence).mean().item()
    )
    graph_mass = float(
        (
            effective_target.new_tensor(scale_value)
            * (1.0 - anchor_confidence)
            * teacher_uncertainty
        ).mean().item()
    )
    stats = {
        "eaogp_scale": scale_value,
        "static_fg_area": float(static_target_68.mean().item()),
        "teacher_binary_area": float(teacher_binary_68.mean().item()),
        "effective_target_mean": float(effective_target.mean().item()),
        "effective_target_hard_area": float(effective_hard.mean().item()),
        "eaogp_target_mean": float(effective_target.mean().item()),
        "eaogp_target_hard_area": float(effective_hard.mean().item()),
        "anchor_confidence_mean": float(anchor_confidence.mean().item()),
        "anchor_confidence_fg_mean": (
            float(anchor_confidence[static_fg].mean().item())
            if bool(static_fg.any().item())
            else 0.0
        ),
        "anchor_confidence_bg_mean": (
            float(anchor_confidence[static_bg].mean().item())
            if bool(static_bg.any().item())
            else 0.0
        ),
        "teacher_uncertainty_mean": float(teacher_uncertainty.mean().item()),
        "dino_graph_consensus_mean": float(dino_consensus_68.mean().item()),
        "dual_graph_consensus_mean": float(dual_consensus_68.mean().item()),
        "dual_vs_dino_shift_abs_mean": float(
            (dual_consensus_68 - dino_consensus_68).abs().mean().item()
        ),
        "task_gate_mean": float(graph["teacher_task_gate"].mean().item()),
        "task_gate_min": float(graph["teacher_task_gate"].min().item()),
        "task_gate_max": float(graph["teacher_task_gate"].max().item()),
        "dual_graph_entropy": float(_entropy(graph["dual_graph_weight"]).mean().item()),
        "dino_graph_entropy": float(_entropy(dino_topk_weight).mean().item()),
        "anchor_delta_abs_mean": float(anchor_delta.abs().mean().item()),
        "graph_delta_abs_mean": float(graph_delta.abs().mean().item()),
        "new_fg_ratio": float(new_foreground.mean().item()),
        "new_bg_ratio": float(new_background.mean().item()),
        "eaogp_new_fg_mass": float(new_foreground.mean().item()),
        "eaogp_new_bg_mass": float(new_background.mean().item()),
        "raw_eaogp_target_mean": float(raw_target.mean().item()),
        "anchor_mass": anchor_mass,
        "graph_mass": graph_mass,
        "eaogp_anchor_mass": anchor_mass,
        "eaogp_graph_mass": graph_mass,
    }
    result["stats"] = stats
    return effective_target, result


_EPOCH_MEAN_FIELDS = (
    "static_weight",
    "teacher_weight",
    "eaogp_scale",
    "static_fg_area",
    "teacher_binary_area",
    "student_pred_area",
    "teacher_pred_area",
    "effective_target_mean",
    "effective_target_hard_area",
    "eaogp_target_mean",
    "eaogp_target_hard_area",
    "anchor_confidence_mean",
    "anchor_confidence_fg_mean",
    "anchor_confidence_bg_mean",
    "teacher_uncertainty_mean",
    "dino_graph_consensus_mean",
    "dual_graph_consensus_mean",
    "dual_vs_dino_shift_abs_mean",
    "task_gate_mean",
    "dual_graph_entropy",
    "anchor_delta_abs_mean",
    "graph_delta_abs_mean",
    "new_fg_ratio",
    "new_bg_ratio",
    "eaogp_anchor_mass",
    "eaogp_graph_mass",
    "eaogp_new_fg_mass",
    "eaogp_new_bg_mass",
    "loss_static_final",
    "loss_teacher_final",
    "loss_total",
)


def new_eaogp_epoch_accumulator() -> dict[str, Any]:
    return {
        "sample_weight": 0.0,
        "sums": {name: 0.0 for name in _EPOCH_MEAN_FIELDS},
    }


def accumulate_eaogp_epoch(
    accumulator: dict[str, Any],
    metrics: Mapping[str, float],
    *,
    batch_size: int,
) -> None:
    weight = float(batch_size)
    if weight <= 0.0:
        raise RuntimeError("EAOGP batch_size must be positive.")
    accumulator["sample_weight"] += weight
    for name in _EPOCH_MEAN_FIELDS:
        if name not in metrics:
            continue
        value = float(metrics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"EAOGP epoch metric {name} is not finite: {value}.")
        accumulator["sums"][name] += weight * value


def finalize_eaogp_epoch(accumulator: Mapping[str, Any]) -> dict[str, float]:
    weight = float(accumulator.get("sample_weight", 0.0))
    if weight <= 0.0:
        return {name: 0.0 for name in _EPOCH_MEAN_FIELDS}
    sums = accumulator["sums"]
    return {name: float(sums[name]) / weight for name in _EPOCH_MEAN_FIELDS}
