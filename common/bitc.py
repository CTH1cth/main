"""BITC-v1 background-intervened teacher calibration.

This module is training-only.  It never reads GT and all teacher-side signals
are constructed under no-grad from cached DINO features, cached DABE
background retrieval indices, and the current binary EMA-Teacher target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


BITC_CACHE_SCHEMA = "dabe_bg_intervention_v1"
BITC_SUBSTITUTION_MODES = {
    "weighted_bg_query_norm",
    "random_bg_query_norm",
    "nearest_bg_query_norm",
    "identity_query_norm",
}
BITC_GRID = 37


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"BITC {name} contains NaN/Inf or is not a tensor")


def _quantile(tensor: torch.Tensor, q: float) -> float:
    return float(torch.quantile(tensor.detach().float().reshape(-1), float(q)).item())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, fallback: float = 0.0) -> float:
    selected = value.detach().float()[mask]
    return float(selected.mean().item()) if int(selected.numel()) else float(fallback)


def _stable_seed(seed: int, sample_key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def bitc_cache_protocol_payload(
    *,
    backbone_key: str,
    dabe_version: str,
    params: dict,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": BITC_CACHE_SCHEMA,
        "backbone_key": str(backbone_key),
        "dabe_version": str(dabe_version),
        "grid": BITC_GRID,
        "seed": int(seed),
        "k_recon": int(params.get("K_RECON", 32)),
        "tau_recon": float(params.get("TAU_RECON", 0.07)),
        "sigma_color_recon": float(params.get("SIGMA_COLOR_RECON", 0.05)),
        "lambda_color_recon": float(params.get("LAMBDA_COLOR_RECON", 0.20)),
        "index_space": "absolute_flat_37x37",
        "weight_dtype": "float16",
    }


def bitc_cache_fingerprint(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(
        protocol,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def build_bitc_retrieval_cache(
    feature: torch.Tensor,
    rgb_grid: torch.Tensor,
    background_anchor_37: torch.Tensor,
    params: dict,
    *,
    seed: int,
    sample_key: str,
) -> dict[str, torch.Tensor | int | float]:
    """Reproduce DABE top-K retrieval and return absolute 37x37 indices."""
    if feature.ndim != 3 or list(feature.shape[-2:]) != [BITC_GRID, BITC_GRID]:
        raise RuntimeError(f"BITC feature must be [C,37,37], got {list(feature.shape)}")
    if list(rgb_grid.shape) != [3, BITC_GRID, BITC_GRID]:
        raise RuntimeError(f"BITC rgb_grid must be [3,37,37], got {list(rgb_grid.shape)}")
    if background_anchor_37.ndim == 3 and int(background_anchor_37.shape[0]) == 1:
        background_anchor_37 = background_anchor_37[0]
    if list(background_anchor_37.shape) != [BITC_GRID, BITC_GRID]:
        raise RuntimeError(
            f"BITC background anchor must be [1,37,37] or [37,37], got "
            f"{list(background_anchor_37.shape)}"
        )
    channels = int(feature.shape[0])
    raw = feature.permute(1, 2, 0).reshape(BITC_GRID * BITC_GRID, channels).float()
    feature_normalized = F.normalize(raw, dim=1, p=2)
    rgb = rgb_grid.permute(1, 2, 0).reshape(BITC_GRID * BITC_GRID, 3).float()
    anchor_indices = torch.where(background_anchor_37.reshape(-1) > 0.5)[0]
    if int(anchor_indices.numel()) < 1:
        raise RuntimeError("BITC DABE background anchor set is empty")
    anchor_feature = feature_normalized.index_select(0, anchor_indices)
    anchor_rgb = rgb.index_select(0, anchor_indices)
    k = min(int(params.get("K_RECON", 32)), int(anchor_indices.numel()))
    tau = float(params.get("TAU_RECON", 0.07))
    sigma_color = float(params.get("SIGMA_COLOR_RECON", 0.05))
    lambda_color = float(params.get("LAMBDA_COLOR_RECON", 0.20))
    if k < 1 or tau <= 0.0 or sigma_color <= 0.0:
        raise RuntimeError(
            f"Invalid BITC DABE retrieval parameters: k/tau/sigma={k}/{tau}/{sigma_color}"
        )

    index_chunks = []
    weight_chunks = []
    for start in range(0, int(raw.shape[0]), 512):
        end = min(start + 512, int(raw.shape[0]))
        query_feature = feature_normalized[start:end]
        query_rgb = rgb[start:end]
        feature_similarity = query_feature @ anchor_feature.t()
        color_distance2 = torch.cdist(query_rgb, anchor_rgb, p=2.0).square()
        color_similarity = torch.exp(-color_distance2 / sigma_color)
        similarity = feature_similarity + lambda_color * color_similarity
        top_values, top_local = torch.topk(similarity, k=k, dim=1)
        index_chunks.append(anchor_indices[top_local])
        weight_chunks.append(torch.softmax(top_values / tau, dim=1))
    topk_indices = torch.cat(index_chunks, dim=0)
    topk_weights = torch.cat(weight_chunks, dim=0)
    nearest_bg_index = topk_indices[:, 0]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, sample_key))
    random_local = torch.randint(
        0,
        int(anchor_indices.numel()),
        (BITC_GRID * BITC_GRID,),
        generator=generator,
        device="cpu",
    ).to(anchor_indices.device)
    fixed_random_bg_index = anchor_indices[random_local]
    membership = torch.zeros(BITC_GRID * BITC_GRID, dtype=torch.bool, device=feature.device)
    membership[anchor_indices] = True
    if not bool(membership[topk_indices].all().item()):
        raise RuntimeError("BITC top-K contains a non-background-anchor index")
    if not bool(membership[nearest_bg_index].all().item()):
        raise RuntimeError("BITC nearest index is outside the background anchors")
    if not bool(membership[fixed_random_bg_index].all().item()):
        raise RuntimeError("BITC random index is outside the background anchors")
    _finite(topk_weights, "topk_weights")
    sum_error = float((topk_weights.sum(dim=1) - 1.0).abs().max().item())
    if sum_error > 1e-5:
        raise RuntimeError(f"BITC top-K weights are not normalized: {sum_error:.9g}")
    return {
        "topk_indices": topk_indices.to(torch.int16).cpu(),
        "topk_weights": topk_weights.to(torch.float16).cpu(),
        "background_anchor_indices": anchor_indices.to(torch.int16).cpu(),
        "nearest_bg_index": nearest_bg_index.to(torch.int16).cpu(),
        "fixed_random_bg_index": fixed_random_bg_index.to(torch.int16).cpu(),
        "background_anchor_count": int(anchor_indices.numel()),
        "topk": int(k),
        "weight_sum_error_fp32": sum_error,
    }


def validate_bitc_config(cfg) -> bool:
    enabled = bool(getattr(cfg, "USE_BITC", False))
    routing_mode = str(getattr(cfg, "TEACHER_ROUTING_MODE", "")).strip().lower()
    if not enabled:
        if routing_mode == "bitc_v1":
            raise RuntimeError("TEACHER_ROUTING_MODE='bitc_v1' requires USE_BITC=True")
        return False
    expected = {
        "BITC_VERSION": "v1",
        "BITC_NUM_GROUPS": 2,
        "BITC_GROUP_MODE": "checkerboard_2",
        "BITC_BATCH_INTERVENTIONS": True,
        "BITC_USE_COARSE_ONLY": True,
        "BITC_CENTER_MODE": "non_intervened_median",
        "BITC_RESPONSE_NORMALIZATION": "per_image_mad",
        "BITC_RESPONSE_CLIP": 6.0,
        "BITC_WEIGHT_FLOOR": 0.20,
        "BITC_APPLY_TO_FINAL": True,
        "BITC_APPLY_TO_COARSE": True,
        "BITC_APPLY_TO_BASE": True,
        "BITC_WEIGHTED_NORMALIZE": True,
        "BITC_DETACH_RESPONSE": True,
        "BITC_DETACH_MAP": True,
        "USE_ECST": False,
        "USE_TEPR_LITE": False,
        "TEACHER_ROUTING_MODE": "bitc_v1",
        "STATIC_WEIGHT_MODE": "ones",
        "DABE_PU_STATIC_SOURCE": "target_soft_68",
        "DABE_PU_STATIC_TARGET_MODE": "soft",
        "TEACHER_TARGET_MODE": "binary",
        "DABE_PU_VERSION": "pu_v11",
        "P_INIT_MODE": "dabe_pu_v11_desplsched",
        "TEACHER_FUSION_MODE": "dabe_pu_despl_sched",
        "USE_DABE_PU": True,
        "USE_DABE_PU_DESPL_SCHEDULE": True,
        "USE_DABE_PU_STATIC_LOSS": True,
        "USE_TEACHER_BINARY_FULL_LOSS": True,
        "USE_TEACHER_SOFT_FULL_LOSS": False,
        "HEAD_TYPE": "dagp_safe",
        "USE_DAGP_SAFE_HEAD": True,
        "USE_NDR_BRANCH": True,
        "USE_NDR_COARSE_AUX": True,
        "USE_BASE_AUX_LOSS": True,
        "MAX_EPOCH": 45,
        "DABE_PU_DESPL_TEACHER_ONLY_START": 21,
        "FINETUNE_RESET_EPOCH": 20,
        "FINETUNE_RESET_TIMING": "after_epoch",
        "LOSS_SIZE": 68,
    }
    mismatched = {
        name: getattr(cfg, name, None)
        for name, value in expected.items()
        if getattr(cfg, name, None) != value
    }
    if mismatched:
        raise RuntimeError(f"BITC-v1 baseline/config contract mismatch: {mismatched}")
    mode = str(getattr(cfg, "BITC_SUBSTITUTION_MODE", "")).strip().lower()
    if mode not in BITC_SUBSTITUTION_MODES - {"identity_query_norm"}:
        raise RuntimeError(
            f"Unsupported BITC_SUBSTITUTION_MODE={mode!r}; "
            f"expected one of {sorted(BITC_SUBSTITUTION_MODES - {'identity_query_norm'})}"
        )
    if not str(getattr(cfg, "BITC_CACHE_ROOT", "")).strip():
        raise RuntimeError("BITC_CACHE_ROOT must be configured")
    forbidden = (
        "USE_RAST",
        "USE_ESA_ASYM",
        "ESA_POST_RESET_ENABLE",
        "USE_ESA_BER",
        "USE_SOURCE_ARBITER",
        "USE_EGSA",
        "USE_TADR_ROUTER",
        "USE_TCE",
        "USE_LCEG",
        "USE_HBNS_LITE",
        "USE_EPR_POS",
        "USE_AP_STCR",
        "USE_CVSA",
        "USE_PSSF",
        "USE_GKD_LITE",
        "USE_MULTI_LEVEL_FEATURE",
        "USE_MULTI_VIEW_FEATURE",
        "USE_VIEW_CONSISTENCY",
        "USE_CSSD",
        "USE_CACD",
        "USE_CSD_DECODER",
        "USE_CSD_V1R",
    )
    active = [name for name in forbidden if bool(getattr(cfg, name, False))]
    if active:
        raise RuntimeError(f"BITC-v1 cannot be combined with historical routers: {active}")
    static_source = str(
        getattr(cfg, "DABE_PU_STATIC_SOURCE", "target_soft_68")
    ).strip().lower()
    if static_source != "target_soft_68":
        raise RuntimeError(
            "BITC-v1 must preserve DABE-PU target_soft_68 static supervision, "
            f"got {static_source!r}"
        )
    debug_max = int(getattr(cfg, "BITC_DEBUG_MAX_IMAGES", 5))
    log_interval = int(getattr(cfg, "BITC_LOG_INTERVAL", 1))
    if not 1 <= debug_max <= 5:
        raise RuntimeError(
            f"BITC_DEBUG_MAX_IMAGES must be in [1,5], got {debug_max}"
        )
    if log_interval < 1:
        raise RuntimeError(f"BITC_LOG_INTERVAL must be positive, got {log_interval}")
    return True


@torch.no_grad()
def build_query_norm_substitute(
    feature: torch.Tensor,
    batch: dict[str, Any],
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Construct a query-norm matched substitute from collated BITC cache fields."""
    mode = str(mode).strip().lower()
    if mode not in BITC_SUBSTITUTION_MODES:
        raise ValueError(f"Unsupported BITC substitute mode: {mode}")
    if feature.ndim != 4 or list(feature.shape[-2:]) != [BITC_GRID, BITC_GRID]:
        raise RuntimeError(f"BITC feature must be [B,C,37,37], got {list(feature.shape)}")
    batch_size, channels, height, width = feature.shape
    nodes = height * width
    flat = feature.detach().float().flatten(2).transpose(1, 2).contiguous()
    original_norm = torch.linalg.vector_norm(flat, dim=2)
    normalized = F.normalize(flat, dim=2, p=2)
    anchor_mask = batch["bitc_background_anchor_mask"].to(feature.device).bool()
    if list(anchor_mask.shape) != [batch_size, nodes]:
        raise RuntimeError(
            f"BITC anchor mask must be [B,1369], got {list(anchor_mask.shape)}"
        )

    if mode == "identity_query_norm":
        substitute_flat = flat
    elif mode == "weighted_bg_query_norm":
        indices = batch["bitc_topk_indices"].to(feature.device).long()
        weights = batch["bitc_topk_weights"].to(feature.device).float()
        if indices.ndim != 3 or list(indices.shape[:2]) != [batch_size, nodes]:
            raise RuntimeError(f"BITC top-K indices shape mismatch: {list(indices.shape)}")
        if list(weights.shape) != list(indices.shape):
            raise RuntimeError(
                f"BITC top-K weight shape mismatch: {list(weights.shape)} != {list(indices.shape)}"
            )
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= nodes:
            raise RuntimeError("BITC top-K index is outside [0,1369)")
        if not bool(torch.gather(anchor_mask, 1, indices.reshape(batch_size, -1)).all().item()):
            raise RuntimeError("BITC top-K index is not a background anchor")
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
        offsets = torch.arange(batch_size, device=feature.device).view(-1, 1, 1) * nodes
        selected = normalized.reshape(batch_size * nodes, channels).index_select(
            0, (indices + offsets).reshape(-1)
        ).reshape(batch_size, nodes, int(indices.shape[2]), channels)
        direction = F.normalize(
            (weights.unsqueeze(-1) * selected).sum(dim=2), dim=2, p=2
        )
        substitute_flat = direction * original_norm.unsqueeze(2)
    else:
        field = (
            "bitc_fixed_random_bg_index"
            if mode == "random_bg_query_norm"
            else "bitc_nearest_bg_index"
        )
        indices = batch[field].to(feature.device).long()
        if list(indices.shape) != [batch_size, nodes]:
            raise RuntimeError(f"BITC {field} shape mismatch: {list(indices.shape)}")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= nodes:
            raise RuntimeError(f"BITC {field} is outside [0,1369)")
        if not bool(torch.gather(anchor_mask, 1, indices).all().item()):
            raise RuntimeError(f"BITC {field} contains a non-background anchor")
        offsets = torch.arange(batch_size, device=feature.device).view(-1, 1) * nodes
        direction = normalized.reshape(batch_size * nodes, channels).index_select(
            0, (indices + offsets).reshape(-1)
        ).reshape(batch_size, nodes, channels)
        substitute_flat = direction * original_norm.unsqueeze(2)

    substitute_norm = torch.linalg.vector_norm(substitute_flat, dim=2)
    ratio = substitute_norm / original_norm.clamp_min(1e-12)
    relative_error = (substitute_norm - original_norm).abs() / original_norm.clamp_min(1e-12)
    if float(relative_error.max().item()) > 1e-4:
        raise RuntimeError(
            "BITC query-norm invariant failed: "
            f"max_relative_error={float(relative_error.max().item()):.9g}"
        )
    substitute = substitute_flat.transpose(1, 2).reshape(
        batch_size, channels, height, width
    ).contiguous().detach()
    _finite(substitute, "substitute")
    stats = {
        "bitc_original_token_norm_mean": float(original_norm.mean().item()),
        "bitc_substitute_token_norm_mean": float(substitute_norm.mean().item()),
        "bitc_norm_ratio_mean": float(ratio.mean().item()),
        "bitc_norm_ratio_p10": _quantile(ratio, 0.10),
        "bitc_norm_ratio_p50": _quantile(ratio, 0.50),
        "bitc_norm_ratio_p90": _quantile(ratio, 0.90),
        "bitc_norm_relative_error_max": float(relative_error.max().item()),
    }
    return substitute, stats


def checkerboard_group_masks(
    height: int = BITC_GRID,
    width: int = BITC_GRID,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    yy = torch.arange(int(height), device=device).view(height, 1)
    xx = torch.arange(int(width), device=device).view(1, width)
    group_ids = (yy + xx) % 2
    masks = torch.stack((group_ids == 0, group_ids == 1), dim=0).unsqueeze(1)
    coverage = masks.to(torch.int16).sum(dim=0)
    if not torch.equal(coverage, torch.ones_like(coverage)):
        raise RuntimeError("BITC checkerboard groups do not cover every token exactly once")
    return masks


@torch.no_grad()
def build_grouped_counterfactuals(
    feature: torch.Tensor,
    substitute: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature.shape != substitute.shape:
        raise RuntimeError(
            f"BITC feature/substitute mismatch: {list(feature.shape)}/{list(substitute.shape)}"
        )
    batch_size, channels, height, width = feature.shape
    masks = checkerboard_group_masks(height, width, device=feature.device)
    grouped = torch.where(
        masks.unsqueeze(1),
        substitute.unsqueeze(0),
        feature.detach().unsqueeze(0),
    )
    grouped = grouped.reshape(2 * batch_size, channels, height, width).contiguous().detach()
    return grouped, masks.detach()


@torch.no_grad()
def build_bitc_teacher_map(
    *,
    original_coarse_logits_37: torch.Tensor,
    counterfactual_coarse_logits_37: torch.Tensor,
    group_masks: torch.Tensor,
    teacher_binary_68: torch.Tensor,
    weight_floor: float = 0.20,
    response_clip: float = 6.0,
    norm_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    if original_coarse_logits_37.ndim != 4 or list(
        original_coarse_logits_37.shape[1:]
    ) != [1, BITC_GRID, BITC_GRID]:
        raise RuntimeError(
            "BITC original coarse logits must be [B,1,37,37], got "
            f"{list(original_coarse_logits_37.shape)}"
        )
    batch_size = int(original_coarse_logits_37.shape[0])
    if list(counterfactual_coarse_logits_37.shape) != [
        2 * batch_size,
        1,
        BITC_GRID,
        BITC_GRID,
    ]:
        raise RuntimeError(
            "BITC counterfactual coarse logits must be [2B,1,37,37], got "
            f"{list(counterfactual_coarse_logits_37.shape)}"
        )
    if list(group_masks.shape) != [2, 1, BITC_GRID, BITC_GRID]:
        raise RuntimeError(f"BITC group mask shape mismatch: {list(group_masks.shape)}")
    if teacher_binary_68.ndim != 4 or int(teacher_binary_68.shape[1]) != 1:
        raise RuntimeError(
            f"BITC Teacher binary target must be [B,1,H,W], got {list(teacher_binary_68.shape)}"
        )
    floor = float(weight_floor)
    clip = float(response_clip)
    if not 0.0 <= floor < 1.0 or clip <= 0.0:
        raise RuntimeError(f"Invalid BITC floor/clip: {floor}/{clip}")

    counterfactual = counterfactual_coarse_logits_37.reshape(
        2, batch_size, 1, BITC_GRID, BITC_GRID
    )
    difference = original_coarse_logits_37.detach().unsqueeze(0) - counterfactual.detach()
    centered_groups = []
    group_medians = []
    spill_ratios = []
    raw_selected = torch.zeros_like(original_coarse_logits_37, dtype=torch.float32)
    centered_selected = torch.zeros_like(original_coarse_logits_37, dtype=torch.float32)
    for group in range(2):
        mask = group_masks[group : group + 1].bool()
        unmasked_values = difference[group, :, 0][:, ~mask[0, 0]]
        median = unmasked_values.median(dim=1).values.view(batch_size, 1, 1, 1)
        centered = difference[group] - median
        centered_groups.append(centered)
        group_medians.append(median)
        expanded_mask = mask.expand(batch_size, -1, -1, -1)
        raw_selected = torch.where(expanded_mask, difference[group], raw_selected)
        centered_selected = torch.where(expanded_mask, centered, centered_selected)
        inside = difference[group].abs()[expanded_mask].reshape(batch_size, -1).mean(dim=1)
        outside = difference[group].abs()[~expanded_mask].reshape(batch_size, -1).mean(dim=1)
        spill_ratios.append(outside / inside.clamp_min(1e-6))

    centered_groups_tensor = torch.stack(centered_groups, dim=0)
    group_medians_tensor = torch.stack(group_medians, dim=0)
    flat_response = centered_selected.flatten(1)
    response_median = flat_response.median(dim=1).values.view(batch_size, 1, 1, 1)
    mad = (centered_selected - response_median).abs().flatten(1).median(dim=1).values
    robust_scale = (1.4826 * mad + 1e-6).view(batch_size, 1, 1, 1)
    normalized = torch.clamp(
        (centered_selected - response_median) / robust_scale,
        min=-clip,
        max=clip,
    )
    evidence = torch.sigmoid(normalized)
    teacher_binary_37 = F.interpolate(
        teacher_binary_68.detach().float(),
        size=(BITC_GRID, BITC_GRID),
        mode="nearest",
    )
    agreement = (
        teacher_binary_37 * evidence
        + (1.0 - teacher_binary_37) * (1.0 - evidence)
    )
    teacher_map_37 = floor + (1.0 - floor) * agreement
    teacher_map_68 = F.interpolate(
        teacher_map_37,
        size=teacher_binary_68.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(floor, 1.0)
    tensors = {
        "difference_groups": difference,
        "centered_difference_groups": centered_groups_tensor,
        "group_unintervened_median": group_medians_tensor,
        "response_raw_37": raw_selected,
        "response_centered_37": centered_selected,
        "response_normalized_37": normalized,
        "response_mad": mad,
        "cf_evidence_37": evidence,
        "teacher_binary_37": teacher_binary_37,
        "teacher_agreement_37": agreement,
        "teacher_map_37": teacher_map_37,
        "teacher_map_68": teacher_map_68,
        "spill_ratio_per_group": torch.stack(spill_ratios, dim=1),
        "group_masks": group_masks,
    }
    for name, tensor in tensors.items():
        if torch.is_tensor(tensor):
            _finite(tensor.float(), name)
            if tensor.requires_grad:
                raise RuntimeError(f"BITC {name} must be detached")
    map_min = float(teacher_map_68.min().item())
    map_max = float(teacher_map_68.max().item())
    if map_min < floor - 1e-6 or map_max > 1.0 + 1e-6:
        raise RuntimeError(f"BITC Teacher map outside [{floor},1]: {map_min}/{map_max}")
    teacher_fg = teacher_binary_37 >= 0.5
    teacher_bg = ~teacher_fg
    spill = tensors["spill_ratio_per_group"]
    stats = {
        "bitc_response_raw_mean": float(raw_selected.mean().item()),
        "bitc_response_raw_std": float(raw_selected.std(unbiased=False).item()),
        "bitc_response_centered_mean": float(centered_selected.mean().item()),
        "bitc_response_centered_median": _quantile(centered_selected, 0.50),
        "bitc_response_centered_p10": _quantile(centered_selected, 0.10),
        "bitc_response_centered_p90": _quantile(centered_selected, 0.90),
        "bitc_response_positive_ratio": float((centered_selected > 0.0).float().mean().item()),
        "bitc_response_mad": float(mad.mean().item()),
        "bitc_cf_evidence_mean": float(evidence.mean().item()),
        "bitc_cf_evidence_std": float(evidence.std(unbiased=False).item()),
        "bitc_cf_evidence_p10": _quantile(evidence, 0.10),
        "bitc_cf_evidence_p50": _quantile(evidence, 0.50),
        "bitc_cf_evidence_p90": _quantile(evidence, 0.90),
        "bitc_teacher_agreement_mean": float(agreement.mean().item()),
        "bitc_teacher_map_mean": float(teacher_map_68.mean().item()),
        "bitc_teacher_map_min": map_min,
        "bitc_teacher_map_max": map_max,
        "bitc_teacher_map_p10": _quantile(teacher_map_68, 0.10),
        "bitc_teacher_map_p50": _quantile(teacher_map_68, 0.50),
        "bitc_teacher_map_p90": _quantile(teacher_map_68, 0.90),
        "bitc_map_teacher_fg_mean": _masked_mean(teacher_map_37, teacher_fg, 1.0),
        "bitc_map_teacher_bg_mean": _masked_mean(teacher_map_37, teacher_bg, 1.0),
        "bitc_response_teacher_fg_mean": _masked_mean(centered_selected, teacher_fg, 0.0),
        "bitc_response_teacher_bg_mean": _masked_mean(centered_selected, teacher_bg, 0.0),
        "bitc_spill_ratio_mean": float(spill.mean().item()),
        "bitc_spill_ratio_median": _quantile(spill, 0.50),
        "bitc_spill_ratio_p90": _quantile(spill, 0.90),
        "bitc_group_coverage_ratio": float(group_masks.float().sum(dim=0).mean().item()),
    }
    if norm_stats:
        stats.update({name: float(value) for name, value in norm_stats.items()})
    return {**tensors, "stats": stats}


BITC_EPOCH_MEAN_KEYS = (
    "bitc_original_token_norm_mean",
    "bitc_substitute_token_norm_mean",
    "bitc_norm_ratio_mean",
    "bitc_norm_ratio_p10",
    "bitc_norm_ratio_p50",
    "bitc_norm_ratio_p90",
    "bitc_norm_relative_error_max",
    "bitc_response_raw_mean",
    "bitc_response_raw_std",
    "bitc_response_centered_mean",
    "bitc_response_centered_median",
    "bitc_response_centered_p10",
    "bitc_response_centered_p90",
    "bitc_response_positive_ratio",
    "bitc_response_mad",
    "bitc_cf_evidence_mean",
    "bitc_cf_evidence_std",
    "bitc_cf_evidence_p10",
    "bitc_cf_evidence_p50",
    "bitc_cf_evidence_p90",
    "bitc_teacher_agreement_mean",
    "bitc_teacher_map_mean",
    "bitc_teacher_map_min",
    "bitc_teacher_map_max",
    "bitc_teacher_map_p10",
    "bitc_teacher_map_p50",
    "bitc_teacher_map_p90",
    "bitc_map_teacher_fg_mean",
    "bitc_map_teacher_bg_mean",
    "bitc_response_teacher_fg_mean",
    "bitc_response_teacher_bg_mean",
    "bitc_spill_ratio_mean",
    "bitc_spill_ratio_median",
    "bitc_spill_ratio_p90",
    "bitc_group_coverage_ratio",
)
BITC_EPOCH_MIN_KEYS = {"bitc_teacher_map_min"}
BITC_EPOCH_MAX_KEYS = {
    "bitc_teacher_map_max",
    "bitc_norm_relative_error_max",
}


def new_bitc_epoch_accumulator() -> dict[str, Any]:
    return {
        "batches": 0,
        "sums": {name: 0.0 for name in BITC_EPOCH_MEAN_KEYS},
        "mins": {name: None for name in BITC_EPOCH_MIN_KEYS},
        "maxs": {name: None for name in BITC_EPOCH_MAX_KEYS},
    }


def accumulate_bitc_epoch(accumulator: dict[str, Any], stats: dict[str, float]) -> None:
    if accumulator is None:
        return
    missing = [name for name in BITC_EPOCH_MEAN_KEYS if name not in stats]
    if missing:
        raise RuntimeError(f"BITC epoch stats are missing fields: {missing}")
    accumulator["batches"] += 1
    for name in BITC_EPOCH_MEAN_KEYS:
        value = float(stats[name])
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"BITC epoch stat is not finite: {name}={value}")
        accumulator["sums"][name] += value
        if name in BITC_EPOCH_MIN_KEYS:
            current = accumulator["mins"][name]
            accumulator["mins"][name] = value if current is None else min(current, value)
        if name in BITC_EPOCH_MAX_KEYS:
            current = accumulator["maxs"][name]
            accumulator["maxs"][name] = value if current is None else max(current, value)


def finalize_bitc_epoch(accumulator: dict[str, Any]) -> dict[str, float]:
    batches = int(accumulator.get("batches", 0))
    if batches < 1:
        raise RuntimeError("BITC epoch accumulator is empty")
    result = {
        name: float(accumulator["sums"][name]) / float(batches)
        for name in BITC_EPOCH_MEAN_KEYS
    }
    result.update(
        {name: float(accumulator["mins"][name]) for name in BITC_EPOCH_MIN_KEYS}
    )
    result.update(
        {name: float(accumulator["maxs"][name]) for name in BITC_EPOCH_MAX_KEYS}
    )
    return result


def _debug_scalar_image(
    tensor: torch.Tensor,
    *,
    signed: bool = False,
    fixed_range: tuple[float, float] | None = None,
    size: int = 180,
) -> Image.Image:
    array = tensor.detach().float().squeeze().cpu().numpy()
    if array.ndim != 2:
        raise RuntimeError(f"BITC debug panel must be 2-D, got {array.shape}")
    if fixed_range is None:
        if signed:
            scale = float(np.quantile(np.abs(array), 0.98))
            low, high = -max(scale, 1e-6), max(scale, 1e-6)
        else:
            low, high = float(array.min()), float(array.max())
            if high - low < 1e-6:
                low, high = low - 0.5, high + 0.5
    else:
        low, high = (float(fixed_range[0]), float(fixed_range[1]))
    unit = np.clip((array - low) / max(high - low, 1e-12), 0.0, 1.0)
    if signed:
        red = np.clip(2.0 * unit - 1.0, 0.0, 1.0)
        blue = np.clip(1.0 - 2.0 * unit, 0.0, 1.0)
        green = 1.0 - np.abs(2.0 * unit - 1.0)
        rgb = np.stack((red + green, green, blue + green), axis=-1)
    else:
        rgb = np.stack((unit, unit, unit), axis=-1)
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _debug_rgb_image(tensor: torch.Tensor, size: int = 180) -> Image.Image:
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255.0).astype(np.uint8))
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


@torch.no_grad()
def export_bitc_debug_batch(
    output_dir: str | Path,
    *,
    image_68: torch.Tensor,
    dabe_soft_68: torch.Tensor,
    teacher_binary_68: torch.Tensor,
    teacher_coarse_logits_37: torch.Tensor,
    substitute_norm_ratio_37: torch.Tensor,
    response_centered_37: torch.Tensor,
    cf_evidence_37: torch.Tensor,
    teacher_map_68: torch.Tensor,
    student_final_prob_68: torch.Tensor,
    group_masks: torch.Tensor,
    sample_names: list[str],
    start_index: int,
    max_images: int = 5,
) -> int:
    """Write the optional ten-panel BITC diagnostic, capped by the caller."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    batch_size = int(image_68.shape[0])
    remaining = max(0, int(max_images) - int(start_index))
    count = min(batch_size, remaining)
    panel_size = 180
    title_height = 24
    for batch_index in range(count):
        group0 = group_masks[0].float()
        panels = [
            ("RGB", _debug_rgb_image(image_68[batch_index], panel_size)),
            (
                "DABE soft target",
                _debug_scalar_image(
                    dabe_soft_68[batch_index], fixed_range=(0.0, 1.0), size=panel_size
                ),
            ),
            (
                "Teacher binary",
                _debug_scalar_image(
                    teacher_binary_68[batch_index], fixed_range=(0.0, 1.0), size=panel_size
                ),
            ),
            (
                "Teacher coarse prob",
                _debug_scalar_image(
                    teacher_coarse_logits_37[batch_index].sigmoid(),
                    fixed_range=(0.0, 1.0),
                    size=panel_size,
                ),
            ),
            (
                "Substitute norm ratio",
                _debug_scalar_image(
                    substitute_norm_ratio_37[batch_index],
                    fixed_range=(0.98, 1.02),
                    size=panel_size,
                ),
            ),
            (
                "Centered response",
                _debug_scalar_image(
                    response_centered_37[batch_index], signed=True, size=panel_size
                ),
            ),
            (
                "Intervention evidence",
                _debug_scalar_image(
                    cf_evidence_37[batch_index], fixed_range=(0.0, 1.0), size=panel_size
                ),
            ),
            (
                "BITC Teacher map",
                _debug_scalar_image(
                    teacher_map_68[batch_index], fixed_range=(0.2, 1.0), size=panel_size
                ),
            ),
            (
                "Student final prediction",
                _debug_scalar_image(
                    student_final_prob_68[batch_index],
                    fixed_range=(0.0, 1.0),
                    size=panel_size,
                ),
            ),
            (
                "Checkerboard group 0",
                _debug_scalar_image(group0, fixed_range=(0.0, 1.0), size=panel_size),
            ),
        ]
        canvas = Image.new(
            "RGB",
            (5 * panel_size, 2 * (panel_size + title_height)),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)
        for panel_index, (title, panel) in enumerate(panels):
            row, column = divmod(panel_index, 5)
            x = column * panel_size
            y = row * (panel_size + title_height)
            draw.text((x + 4, y + 4), title, fill=(0, 0, 0))
            canvas.paste(panel, (x, y + title_height))
        name = sample_names[batch_index] if batch_index < len(sample_names) else str(batch_index)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        destination = root / f"{int(start_index) + batch_index:03d}_{safe_name}.png"
        canvas.save(destination)
    return count
