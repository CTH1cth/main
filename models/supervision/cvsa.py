"""CVSA-v1 cross-view source-risk supervision.

CVSA never estimates ground-truth correctness.  Its two-view target is only a
training-time proxy for the relative cross-view risk of the fixed DABE-PU and
pre-update EMA-teacher sources.  The learnable router consumes canonical-view
inputs only; hflip evidence is used exclusively to supervise that router.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common.utils import build_image_items, read_jsonl, torch_load


CVSA_CACHE_VERSION = "cvsa_v1_hflip_cache_v1"
CVSA_DIAGNOSTIC_SCHEMA = "cvsa_v1_diagnostic_v1"


def _require_map(name, value, channels=1, shape=None):
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError(f"{name} must be a 4D tensor, got {type(value)!r}.")
    if int(value.shape[1]) != int(channels):
        raise RuntimeError(
            f"{name} must have {channels} channels, got {list(value.shape)}."
        )
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise RuntimeError(
            f"{name} shape mismatch: {list(value.shape)} != {list(shape)}."
        )
    value = value.float()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")
    return value


def _require_unit_map(name, value, shape=None):
    value = _require_map(name, value, channels=1, shape=shape)
    if float(value.min().item()) < -1e-6 or float(value.max().item()) > 1.0 + 1e-6:
        raise RuntimeError(f"{name} must stay in [0,1].")
    return value.clamp(0.0, 1.0)


def resize_probability(value, resolution):
    value = _require_unit_map("probability", value)
    resolution = int(resolution)
    if tuple(value.shape[-2:]) == (resolution, resolution):
        return value
    return F.interpolate(
        value,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def align_hflip_view(value):
    if not torch.is_tensor(value) or value.ndim < 2:
        raise RuntimeError("HFlip alignment requires a tensor with spatial axes.")
    return torch.flip(value, dims=[-1])


def build_source_views(source_a_68, source_b_68, evidence_resolution=37):
    source_a_37 = resize_probability(source_a_68.detach(), evidence_resolution)
    source_b_37 = resize_probability(source_b_68.detach(), evidence_resolution)
    return source_a_37.detach(), align_hflip_view(source_b_37).detach()


def compute_equivariance_risk(source_a_37, source_b_aligned_37):
    source_a_37 = _require_unit_map("source_a_37", source_a_37)
    source_b_aligned_37 = _require_unit_map(
        "source_b_aligned_37", source_b_aligned_37, source_a_37.shape
    )
    return (source_a_37 - source_b_aligned_37).abs().clamp(0.0, 1.0).detach()


def _source_mass(source):
    foreground = source.flatten(1).sum(dim=1)
    background = (1.0 - source).flatten(1).sum(dim=1)
    return foreground, background


def validate_source_mass(source_a, source_b_aligned, min_proto_mass_patches=4.0):
    source_a = _require_unit_map("source_a", source_a)
    source_b_aligned = _require_unit_map(
        "source_b_aligned", source_b_aligned, source_a.shape
    )
    minimum = float(min_proto_mass_patches)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise RuntimeError("min_proto_mass_patches must be positive and finite.")
    fg_a, bg_a = _source_mass(source_a)
    fg_b, bg_b = _source_mass(source_b_aligned)
    invalid_fg = (fg_a < minimum) | (fg_b < minimum)
    invalid_bg = (bg_a < minimum) | (bg_b < minimum)
    valid = ~(invalid_fg | invalid_bg)
    return {
        "valid": valid.detach(),
        "invalid_fg": invalid_fg.detach(),
        "invalid_bg": invalid_bg.detach(),
        "fg_mass_a": fg_a.detach(),
        "bg_mass_a": bg_a.detach(),
        "fg_mass_b": fg_b.detach(),
        "bg_mass_b": bg_b.detach(),
    }


def build_soft_prototypes(feature, source, valid, eps=1e-6):
    """Build source-conditioned prototypes only for valid images."""
    feature = _require_map("feature", feature, channels=feature.shape[1])
    source = _require_unit_map("source", source)
    if tuple(feature.shape[0:1] + feature.shape[-2:]) != tuple(
        source.shape[0:1] + source.shape[-2:]
    ):
        raise RuntimeError(
            f"Feature/source spatial mismatch: {list(feature.shape)} vs {list(source.shape)}."
        )
    valid = valid.to(device=feature.device, dtype=torch.bool)
    batch, channels = int(feature.shape[0]), int(feature.shape[1])
    fg_proto = torch.zeros((batch, channels), device=feature.device, dtype=torch.float32)
    bg_proto = torch.zeros_like(fg_proto)
    if bool(valid.any().item()):
        feature_valid = feature[valid]
        source_valid = source[valid]
        fg_mass = source_valid.flatten(2).sum(dim=2).clamp_min(float(eps))
        bg_weight = 1.0 - source_valid
        bg_mass = bg_weight.flatten(2).sum(dim=2).clamp_min(float(eps))
        fg = (feature_valid * source_valid).flatten(2).sum(dim=2) / fg_mass
        bg = (feature_valid * bg_weight).flatten(2).sum(dim=2) / bg_mass
        fg_proto[valid] = F.normalize(fg, dim=1, eps=float(eps))
        bg_proto[valid] = F.normalize(bg, dim=1, eps=float(eps))
    return fg_proto.detach(), bg_proto.detach()


def _semantic_transfer_bce(feature, fg_proto, bg_proto, target, temperature):
    fg_similarity = (feature * fg_proto[:, :, None, None]).sum(dim=1, keepdim=True)
    bg_similarity = (feature * bg_proto[:, :, None, None]).sum(dim=1, keepdim=True)
    probability = torch.sigmoid(
        (fg_similarity - bg_similarity) / float(temperature)
    ).clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(probability, target, reduction="none")


@torch.no_grad()
def compute_cross_view_semantic_risk(
    feature_a,
    feature_b_aligned,
    source_a,
    source_b_aligned,
    valid,
    semantic_temperature=0.20,
    eps=1e-6,
):
    temperature = float(semantic_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("semantic_temperature must be positive and finite.")
    feature_a = F.normalize(feature_a.detach().float(), dim=1, eps=float(eps))
    feature_b_aligned = F.normalize(
        feature_b_aligned.detach().float(), dim=1, eps=float(eps)
    )
    source_a = source_a.detach().float()
    source_b_aligned = source_b_aligned.detach().float()
    fg_a, bg_a = build_soft_prototypes(feature_a, source_a, valid, eps=eps)
    fg_b, bg_b = build_soft_prototypes(
        feature_b_aligned, source_b_aligned, valid, eps=eps
    )
    forward_bce = _semantic_transfer_bce(
        feature_b_aligned, fg_a, bg_a, source_b_aligned, temperature
    )
    backward_bce = _semantic_transfer_bce(
        feature_a, fg_b, bg_b, source_a, temperature
    )
    semantic_bce = 0.5 * (forward_bce + backward_bce)
    risk = (1.0 - torch.exp(-semantic_bce)).clamp(0.0, 1.0)
    valid_map = valid[:, None, None, None]
    return torch.where(valid_map, risk, torch.ones_like(risk)).detach()


@torch.no_grad()
def compute_source_risk(
    feature_a,
    feature_b_aligned,
    source_a,
    source_b_aligned,
    risk_eq_weight=0.5,
    risk_sem_weight=0.5,
    semantic_temperature=0.20,
    min_proto_mass_patches=4.0,
    eps=1e-6,
):
    eq_weight = float(risk_eq_weight)
    sem_weight = float(risk_sem_weight)
    if (
        not math.isfinite(eq_weight)
        or not math.isfinite(sem_weight)
        or eq_weight < 0.0
        or sem_weight < 0.0
        or abs(eq_weight + sem_weight - 1.0) > 1e-12
    ):
        raise RuntimeError("CVSA risk weights must be nonnegative and sum to 1.")
    validity = validate_source_mass(
        source_a, source_b_aligned, min_proto_mass_patches
    )
    risk_eq = compute_equivariance_risk(source_a, source_b_aligned)
    risk_sem = compute_cross_view_semantic_risk(
        feature_a=feature_a,
        feature_b_aligned=feature_b_aligned,
        source_a=source_a,
        source_b_aligned=source_b_aligned,
        valid=validity["valid"],
        semantic_temperature=semantic_temperature,
        eps=eps,
    )
    total = (eq_weight * risk_eq + sem_weight * risk_sem).clamp(0.0, 1.0)
    total = torch.where(
        validity["valid"][:, None, None, None], total, torch.ones_like(total)
    )
    return {
        "risk_eq_37": risk_eq.detach(),
        "risk_sem_37": risk_sem.detach(),
        "risk_total_37": total.detach(),
        **validity,
    }


@torch.no_grad()
def build_route_target(
    fixed_risk_37,
    teacher_risk_37,
    fixed_valid,
    teacher_valid,
    route_temperature=0.10,
):
    fixed_risk_37 = _require_unit_map("fixed_risk_37", fixed_risk_37)
    teacher_risk_37 = _require_unit_map(
        "teacher_risk_37", teacher_risk_37, fixed_risk_37.shape
    )
    temperature = float(route_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("route_temperature must be positive and finite.")
    fixed_valid = fixed_valid.to(fixed_risk_37.device, dtype=torch.bool)
    teacher_valid = teacher_valid.to(fixed_risk_37.device, dtype=torch.bool)
    target = torch.sigmoid((fixed_risk_37 - teacher_risk_37) / temperature)
    fixed_only = fixed_valid & ~teacher_valid
    teacher_only = ~fixed_valid & teacher_valid
    both_invalid = ~fixed_valid & ~teacher_valid
    target = torch.where(fixed_only[:, None, None, None], torch.zeros_like(target), target)
    target = torch.where(teacher_only[:, None, None, None], torch.ones_like(target), target)
    target = torch.where(
        both_invalid[:, None, None, None], torch.full_like(target, 0.5), target
    ).detach()
    weight = (2.0 * (target - 0.5).abs()).clamp(0.0, 1.0)
    weight = torch.where(
        both_invalid[:, None, None, None], torch.zeros_like(weight), weight
    ).detach()
    return {
        "route_target_37": target,
        "route_target_weight_37": weight,
        "both_valid": (fixed_valid & teacher_valid).detach(),
        "both_invalid": both_invalid.detach(),
    }


def bernoulli_entropy(probability, eps=1e-6):
    probability = _require_unit_map("entropy_probability", probability)
    value = -(
        probability * torch.log(probability + float(eps))
        + (1.0 - probability) * torch.log(1.0 - probability + float(eps))
    ) / math.log(2.0)
    return value.clamp(0.0, 1.0)


def build_router_inputs(feature_a, fixed_a_37, teacher_a_37, eps=1e-6):
    feature_a = _require_map("feature_a", feature_a, channels=384).detach().float()
    fixed_a_37 = _require_unit_map("fixed_a_37", fixed_a_37).detach()
    teacher_a_37 = _require_unit_map(
        "teacher_a_37", teacher_a_37, fixed_a_37.shape
    ).detach()
    if tuple(feature_a.shape[-2:]) != tuple(fixed_a_37.shape[-2:]):
        raise RuntimeError("CVSA router feature and scalar grids must match.")
    scalar = torch.cat(
        [
            fixed_a_37,
            teacher_a_37,
            (teacher_a_37 - fixed_a_37).abs(),
            bernoulli_entropy(fixed_a_37, eps=eps),
            bernoulli_entropy(teacher_a_37, eps=eps),
        ],
        dim=1,
    ).detach()
    return feature_a, scalar


class CVSAPatchRouter(nn.Module):
    def __init__(
        self,
        feature_channels=384,
        feature_dim=32,
        hidden_dim=32,
        gn_groups=4,
        init_teacher_prob=0.05,
        eps=1e-6,
    ):
        super().__init__()
        init_probability = float(init_teacher_prob)
        if not 0.0 < init_probability < 1.0:
            raise RuntimeError("router_init_teacher_prob must be in (0,1).")
        self.eps = float(eps)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(int(feature_channels), int(feature_dim), 1, bias=False),
            nn.GroupNorm(int(gn_groups), int(feature_dim)),
            nn.GELU(),
        )
        fusion_channels = int(feature_dim) + 5
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, int(hidden_dim), 3, padding=1, bias=False),
            nn.GroupNorm(int(gn_groups), int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1, bias=False),
            nn.GroupNorm(int(gn_groups), int(hidden_dim)),
            nn.GELU(),
        )
        self.router_out = nn.Conv2d(int(hidden_dim), 1, 1, bias=True)
        nn.init.zeros_(self.router_out.weight)
        nn.init.constant_(
            self.router_out.bias,
            math.log(init_probability / (1.0 - init_probability)),
        )

    def forward(self, feature_a, fixed_a_37, teacher_a_37):
        feature, scalar = build_router_inputs(
            feature_a, fixed_a_37, teacher_a_37, eps=self.eps
        )
        projected = self.feature_proj(feature)
        return torch.sigmoid(self.router_out(self.fusion(torch.cat([projected, scalar], 1))))


def cvsa_router_parameter_count(router):
    if router is None:
        return 0
    return sum(parameter.numel() for parameter in router.parameters())


def cvsa_router_macs(
    feature_dim=32, hidden_dim=32, resolution=37, feature_channels=384
):
    pixels = int(resolution) * int(resolution)
    return int(
        pixels * int(feature_channels) * int(feature_dim)
        + pixels * (int(feature_dim) + 5) * int(hidden_dim) * 9
        + pixels * int(hidden_dim) * int(hidden_dim) * 9
        + pixels * int(hidden_dim)
    )


def compute_router_loss(router_gate_37, route_target_37, route_target_weight_37, eps=1e-6):
    router_gate_37 = _require_unit_map("router_gate_37", router_gate_37)
    route_target_37 = _require_unit_map(
        "route_target_37", route_target_37, router_gate_37.shape
    )
    route_target_weight_37 = _require_unit_map(
        "route_target_weight_37", route_target_weight_37, router_gate_37.shape
    )
    route_bce = F.binary_cross_entropy(
        router_gate_37, route_target_37, reduction="none"
    )
    return (route_bce * route_target_weight_37).sum() / (
        route_target_weight_37.sum() + float(eps)
    )


def build_mixed_target(
    fixed_soft_68, teacher_binary_68, gate_37, loss_resolution=68
):
    fixed_soft_68 = _require_unit_map("fixed_soft_68", fixed_soft_68).detach()
    teacher_binary_68 = _require_unit_map(
        "teacher_binary_68", teacher_binary_68, fixed_soft_68.shape
    ).detach()
    if not bool(
        ((teacher_binary_68 == 0.0) | (teacher_binary_68 == 1.0)).all().item()
    ):
        raise RuntimeError("CVSA teacher_binary_68 must contain only 0/1 values.")
    gate_37 = _require_unit_map("gate_37", gate_37)
    gate_68 = F.interpolate(
        gate_37,
        size=(int(loss_resolution), int(loss_resolution)),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    gate_68_detached = gate_68.detach()
    mixed = (
        (1.0 - gate_68_detached) * fixed_soft_68
        + gate_68_detached * teacher_binary_68
    ).detach().clamp(0.0, 1.0)
    return gate_68_detached, mixed


def build_cvsa_batch(
    config,
    feature_a,
    feature_b,
    fixed_a_68,
    fixed_b_68,
    teacher_a_68,
    teacher_b_68,
    teacher_binary_a_68,
    router=None,
):
    """Build current-iteration risk target, router gate and mixed target."""
    config = dict(config)
    evidence_resolution = int(config.get("evidence_resolution", 37))
    feature_a = feature_a.detach().float()
    feature_b_aligned = align_hflip_view(feature_b.detach().float())
    fixed_a_37, fixed_b_aligned_37 = build_source_views(
        fixed_a_68, fixed_b_68, evidence_resolution
    )
    teacher_a_37, teacher_b_aligned_37 = build_source_views(
        teacher_a_68, teacher_b_68, evidence_resolution
    )
    with torch.no_grad():
        risk_kwargs = {
            "risk_eq_weight": float(config.get("risk_eq_weight", 0.5)),
            "risk_sem_weight": float(config.get("risk_sem_weight", 0.5)),
            "semantic_temperature": float(config.get("semantic_temperature", 0.20)),
            "min_proto_mass_patches": float(
                config.get("min_proto_mass_patches", 4.0)
            ),
            "eps": float(config.get("eps", 1e-6)),
        }
        fixed = compute_source_risk(
            feature_a,
            feature_b_aligned,
            fixed_a_37,
            fixed_b_aligned_37,
            **risk_kwargs,
        )
        teacher = compute_source_risk(
            feature_a,
            feature_b_aligned,
            teacher_a_37,
            teacher_b_aligned_37,
            **risk_kwargs,
        )
        route = build_route_target(
            fixed["risk_total_37"],
            teacher["risk_total_37"],
            fixed["valid"],
            teacher["valid"],
            route_temperature=float(config.get("route_temperature", 0.10)),
        )
    route_mode = str(config.get("route_mode", "learnable")).lower()
    if route_mode == "learnable":
        if router is None:
            raise RuntimeError("CVSA learnable mode requires a patch router.")
        gate_37 = router(feature_a, fixed_a_37, teacher_a_37)
        loss_route = compute_router_loss(
            gate_37,
            route["route_target_37"],
            route["route_target_weight_37"],
            eps=float(config.get("eps", 1e-6)),
        )
    elif route_mode == "direct":
        if router is not None:
            raise RuntimeError("CVSA direct mode must not construct a router.")
        gate_37 = route["route_target_37"].detach()
        loss_route = gate_37.sum() * 0.0
    else:
        raise RuntimeError(f"Unsupported CVSA route_mode={route_mode!r}.")
    gate_68, mixed_target = build_mixed_target(
        fixed_a_68,
        teacher_binary_a_68,
        gate_37,
        loss_resolution=int(config.get("loss_resolution", 68)),
    )
    result = {
        "route_mode": route_mode,
        "feature_b_aligned_37": feature_b_aligned.detach(),
        "fixed_a_37": fixed_a_37,
        "fixed_b_aligned_37": fixed_b_aligned_37,
        "teacher_a_37": teacher_a_37,
        "teacher_b_aligned_37": teacher_b_aligned_37,
        "fixed_a_68": fixed_a_68.detach(),
        "fixed_b_aligned_68": align_hflip_view(fixed_b_68.detach()),
        "teacher_a_68": teacher_a_68.detach(),
        "teacher_b_aligned_68": align_hflip_view(teacher_b_68.detach()),
        "teacher_binary_a_68": teacher_binary_a_68.detach(),
        "fixed_eq_risk_37": fixed["risk_eq_37"],
        "fixed_sem_risk_37": fixed["risk_sem_37"],
        "fixed_total_risk_37": fixed["risk_total_37"],
        "teacher_eq_risk_37": teacher["risk_eq_37"],
        "teacher_sem_risk_37": teacher["risk_sem_37"],
        "teacher_total_risk_37": teacher["risk_total_37"],
        "risk_gap_37": (
            fixed["risk_total_37"] - teacher["risk_total_37"]
        ).detach(),
        "fixed_valid": fixed["valid"],
        "teacher_valid": teacher["valid"],
        "fixed_invalid_fg": fixed["invalid_fg"],
        "fixed_invalid_bg": fixed["invalid_bg"],
        "teacher_invalid_fg": teacher["invalid_fg"],
        "teacher_invalid_bg": teacher["invalid_bg"],
        **route,
        "router_gate_37": gate_37,
        "gate_68": gate_68,
        "mixed_target_68": mixed_target,
        "loss_route": loss_route,
        "global_schedule_used": False,
        "epoch_ratio_used": False,
        "future_teacher_used": False,
        "temporal_history_allocated": False,
        "training_gt_used": False,
    }
    return result


class _RunningDistribution:
    def __init__(self, low=0.0, high=1.0, bins=1001):
        self.low = float(low)
        self.high = float(high)
        self.bins = int(bins)
        self.count = 0
        self.total = 0.0
        self.square = 0.0
        self.hist = torch.zeros(self.bins, dtype=torch.float64)

    def update(self, value, mask=None):
        value = value.detach().float()
        if mask is not None:
            mask = mask.to(value.device, dtype=torch.bool)
            if mask.shape != value.shape:
                mask = torch.broadcast_to(mask, value.shape)
            value = value[mask]
        else:
            value = value.reshape(-1)
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            return
        clipped = value.clamp(self.low, self.high).cpu()
        self.count += int(clipped.numel())
        self.total += float(clipped.double().sum().item())
        self.square += float(clipped.double().square().sum().item())
        self.hist += torch.histc(
            clipped.float(), bins=self.bins, min=self.low, max=self.high
        ).double()

    def finalize(self):
        if self.count <= 0:
            return {name: 0.0 for name in ("mean", "std", "p10", "p50", "p90")}
        mean = self.total / self.count
        variance = max(0.0, self.square / self.count - mean * mean)
        cumulative = self.hist.cumsum(0)
        values = {}
        for name, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            threshold = quantile * self.count
            index = int(torch.searchsorted(cumulative, torch.tensor(threshold)).item())
            index = max(0, min(self.bins - 1, index))
            values[name] = self.low + (self.high - self.low) * index / max(
                self.bins - 1, 1
            )
        return {"mean": mean, "std": math.sqrt(variance), **values}


def new_cvsa_epoch_accumulator():
    distributions = {
        "fixed_eq_risk": _RunningDistribution(),
        "teacher_eq_risk": _RunningDistribution(),
        "fixed_sem_risk": _RunningDistribution(),
        "teacher_sem_risk": _RunningDistribution(),
        "fixed_total_risk": _RunningDistribution(),
        "teacher_total_risk": _RunningDistribution(),
        "risk_gap": _RunningDistribution(-1.0, 1.0, 2001),
        "route_target": _RunningDistribution(),
        "router_gate": _RunningDistribution(),
    }
    return {
        "distributions": distributions,
        "sample_count": 0,
        "pixel_count": 0,
        "scalar_sums": {},
        "calibration": {
            "weight": 0.0,
            "abs": 0.0,
            "square": 0.0,
            "x": 0.0,
            "y": 0.0,
            "xx": 0.0,
            "yy": 0.0,
            "xy": 0.0,
        },
    }


@torch.no_grad()
def accumulate_cvsa_epoch(accumulator, result, student_prob_68, loss_seg, loss_total):
    distributions = accumulator["distributions"]
    field_map = {
        "fixed_eq_risk": "fixed_eq_risk_37",
        "teacher_eq_risk": "teacher_eq_risk_37",
        "fixed_sem_risk": "fixed_sem_risk_37",
        "teacher_sem_risk": "teacher_sem_risk_37",
        "fixed_total_risk": "fixed_total_risk_37",
        "teacher_total_risk": "teacher_total_risk_37",
        "risk_gap": "risk_gap_37",
        "route_target": "route_target_37",
        "router_gate": "router_gate_37",
    }
    for name, field in field_map.items():
        distributions[name].update(result[field])
    batch = int(result["fixed_valid"].numel())
    pixels = int(result["route_target_37"].numel())
    accumulator["sample_count"] += batch
    accumulator["pixel_count"] += pixels
    fixed_valid = result["fixed_valid"].float()
    teacher_valid = result["teacher_valid"].float()
    both_valid = result["both_valid"].float()
    both_invalid = result["both_invalid"].float()
    target = result["route_target_37"].detach().float()
    gate = result["router_gate_37"].detach().float()
    valid_route = (~result["both_invalid"])[:, None, None, None].float()
    weight = result["route_target_weight_37"].detach().float() * valid_route
    scalar_values = {
        "fixed_valid": float(fixed_valid.sum().item()),
        "teacher_valid": float(teacher_valid.sum().item()),
        "both_valid": float(both_valid.sum().item()),
        "both_invalid": float(both_invalid.sum().item()),
        "fixed_invalid_fg": float(result["fixed_invalid_fg"].float().sum().item()),
        "fixed_invalid_bg": float(result["fixed_invalid_bg"].float().sum().item()),
        "teacher_invalid_fg": float(result["teacher_invalid_fg"].float().sum().item()),
        "teacher_invalid_bg": float(result["teacher_invalid_bg"].float().sum().item()),
        "route_target_weight": float(result["route_target_weight_37"].sum().item()),
        "target_fixed_pref": float(((target < 0.25).float() * valid_route).sum().item()),
        "target_teacher_pref": float(((target > 0.75).float() * valid_route).sum().item()),
        "target_ambiguous": float((((target >= 0.4) & (target <= 0.6)).float() * valid_route).sum().item()),
        "router_fixed": float(((gate < 0.25).float() * valid_route).sum().item()),
        "router_teacher": float(((gate > 0.75).float() * valid_route).sum().item()),
        "router_ambiguous": float((((gate >= 0.4) & (gate <= 0.6)).float() * valid_route).sum().item()),
        "valid_route_pixels": float(valid_route.sum().item() * target.shape[-1] * target.shape[-2] / max(valid_route.shape[-1] * valid_route.shape[-2], 1)),
        "fixed_area": float(result["fixed_a_68"].mean(dim=(1, 2, 3)).sum().item()),
        "teacher_area": float(result["teacher_binary_a_68"].mean(dim=(1, 2, 3)).sum().item()),
        "mixed_target_area": float(result["mixed_target_68"].mean(dim=(1, 2, 3)).sum().item()),
        "student_area": float((student_prob_68 > 0.5).float().mean(dim=(1, 2, 3)).sum().item()),
        "teacher_prob_mean": float(result["teacher_a_68"].mean(dim=(1, 2, 3)).sum().item()),
        "student_prob_mean": float(student_prob_68.mean(dim=(1, 2, 3)).sum().item()),
        "loss_route": float(result["loss_route"].detach().item()),
        "loss_seg": float(loss_seg.detach().item()),
        "loss_total": float(loss_total.detach().item()),
        "router_gradient_norm": float(result.get("router_gradient_norm", 0.0)),
        "batch_count": 1.0,
    }
    for name, value in scalar_values.items():
        accumulator["scalar_sums"][name] = accumulator["scalar_sums"].get(name, 0.0) + value
    calibration = accumulator["calibration"]
    calibration["weight"] += float(weight.sum().item())
    calibration["abs"] += float((weight * (gate - target).abs()).sum().item())
    calibration["square"] += float((weight * (gate - target).square()).sum().item())
    calibration["x"] += float((weight * gate).sum().item())
    calibration["y"] += float((weight * target).sum().item())
    calibration["xx"] += float((weight * gate.square()).sum().item())
    calibration["yy"] += float((weight * target.square()).sum().item())
    calibration["xy"] += float((weight * gate * target).sum().item())


def finalize_cvsa_epoch(accumulator):
    result = {}
    for name, distribution in accumulator["distributions"].items():
        for statistic, value in distribution.finalize().items():
            result[f"{name}_{statistic}"] = value
    sample_count = max(int(accumulator["sample_count"]), 1)
    pixel_count = max(int(accumulator["pixel_count"]), 1)
    sums = accumulator["scalar_sums"]
    for name in (
        "fixed_valid", "teacher_valid", "both_valid", "both_invalid",
        "fixed_invalid_fg", "fixed_invalid_bg", "teacher_invalid_fg", "teacher_invalid_bg",
    ):
        result[f"{name}_ratio"] = sums.get(name, 0.0) / sample_count
    valid_pixels = max(sums.get("valid_route_pixels", 0.0), 1.0)
    result["route_target_weight_mean"] = sums.get("route_target_weight", 0.0) / pixel_count
    for name in (
        "target_fixed_pref", "target_teacher_pref", "target_ambiguous",
        "router_fixed", "router_teacher", "router_ambiguous",
    ):
        result[f"{name}_ratio"] = sums.get(name, 0.0) / valid_pixels
    for name in (
        "fixed_area", "teacher_area", "mixed_target_area", "student_area",
        "teacher_prob_mean", "student_prob_mean",
    ):
        result[name] = sums.get(name, 0.0) / sample_count
    batch_count = max(sums.get("batch_count", 0.0), 1.0)
    for name in ("loss_route", "loss_seg", "loss_total", "router_gradient_norm"):
        result[name] = sums.get(name, 0.0) / batch_count
    cal = accumulator["calibration"]
    weight = max(cal["weight"], 1e-12)
    result["router_target_mae"] = cal["abs"] / weight
    result["router_target_brier"] = cal["square"] / weight
    mean_x, mean_y = cal["x"] / weight, cal["y"] / weight
    covariance = cal["xy"] / weight - mean_x * mean_y
    variance_x = max(0.0, cal["xx"] / weight - mean_x * mean_x)
    variance_y = max(0.0, cal["yy"] / weight - mean_y * mean_y)
    denominator = math.sqrt(variance_x * variance_y)
    result["router_target_correlation"] = covariance / denominator if denominator > 1e-12 else 0.0
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cvsa_protocol_fingerprint(config):
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_map(path):
    rows = read_jsonl(path)
    mapping = {}
    for row in rows:
        key = (row.get("dataset"), row.get("stem"))
        if None in key or key in mapping:
            raise RuntimeError(f"Invalid or duplicate CVSA manifest key {key}: {path}")
        mapping[key] = row
    return mapping


def validate_cvsa_cache_manifests(cfg, max_samples=None):
    config = dict(getattr(cfg, "CVSA"))
    feature_manifest = Path(config["feature_cache_hflip_root"]) / "manifest_train.jsonl"
    fixed_manifest = Path(config["fixed_cache_hflip_root"]) / "manifest_train.jsonl"
    for path in (feature_manifest, fixed_manifest):
        if not path.is_file():
            raise RuntimeError(f"CVSA cache manifest is missing: {path}")
    feature_map = _manifest_map(feature_manifest)
    fixed_map = _manifest_map(fixed_manifest)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    keys = [(item["dataset"], item["stem"]) for item in items]
    item_map = {(item["dataset"], item["stem"]): item for item in items}
    missing_feature = [key for key in keys if key not in feature_map]
    missing_fixed = [key for key in keys if key not in fixed_map]
    if missing_feature or missing_fixed:
        raise RuntimeError(
            f"CVSA cache is incomplete: feature={missing_feature[:5]}, fixed={missing_fixed[:5]}."
        )
    for key in keys:
        feature_row = feature_map[key]
        fixed_row = fixed_map[key]
        required_feature = {
            "view": "hflip",
            "backbone": cfg.BACKBONE_KEY,
            "cache_version": CVSA_CACHE_VERSION,
            "independently_generated": True,
            "training_gt_used": False,
            "generation_call_chain": "source_rgb->horizontal_flip->DINO_key",
        }
        required_fixed = {
            "view": "hflip",
            "dabe_version": "pu_v11",
            "source_feature_view": "hflip",
            "cache_version": CVSA_CACHE_VERSION,
            "independently_generated": True,
            "training_gt_used": False,
            "generation_call_chain": (
                "source_rgb->horizontal_flip + "
                "independent_hflip_DINO->DABE-PU-v1.1"
            ),
        }
        for name, expected in required_feature.items():
            if feature_row.get(name) != expected:
                raise RuntimeError(
                    f"CVSA hflip feature provenance mismatch for {key}: {name}={feature_row.get(name)!r}."
                )
        for name, expected in required_fixed.items():
            if fixed_row.get(name) != expected:
                raise RuntimeError(
                    f"CVSA hflip fixed provenance mismatch for {key}: {name}={fixed_row.get(name)!r}."
                )
        feature_path = feature_row.get("feature_path", feature_row.get("cache_path"))
        fixed_path = fixed_row.get("fixed_path", fixed_row.get("cache_path"))
        if not feature_path or not Path(feature_path).is_file():
            raise RuntimeError(f"CVSA hflip feature file missing for {key}: {feature_path}")
        if not fixed_path or not Path(fixed_path).is_file():
            raise RuntimeError(f"CVSA hflip fixed file missing for {key}: {fixed_path}")
        if feature_row.get("checksum") != _sha256_file(feature_path):
            raise RuntimeError(f"CVSA hflip feature checksum mismatch for {key}.")
        if fixed_row.get("checksum") != _sha256_file(fixed_path):
            raise RuntimeError(f"CVSA hflip fixed checksum mismatch for {key}.")
        source_path = Path(item_map[key]["image_path"]).resolve()
        for row_name, row in (("feature", feature_row), ("fixed", fixed_row)):
            if Path(row.get("source_image_path", "")).resolve() != source_path:
                raise RuntimeError(
                    f"CVSA hflip {row_name} source path mismatch for {key}."
                )
            if row.get("source_image_checksum") != _sha256_file(source_path):
                raise RuntimeError(
                    f"CVSA hflip {row_name} source checksum mismatch for {key}."
                )
            if "gt_path" in row or row.get("training_gt_used") is not False:
                raise RuntimeError(
                    f"CVSA hflip {row_name} manifest leaks training GT for {key}."
                )
        if Path(fixed_row.get("source_feature_path", "")).resolve() != Path(
            feature_path
        ).resolve():
            raise RuntimeError(f"CVSA DABE source feature path mismatch for {key}.")
        if fixed_row.get("source_feature_checksum") != feature_row.get("checksum"):
            raise RuntimeError(
                f"CVSA DABE source feature checksum mismatch for {key}."
            )
        feature_payload = torch_load(feature_path, map_location="cpu")
        fixed_payload = torch_load(fixed_path, map_location="cpu")
        feature = feature_payload.get("tensor") if isinstance(feature_payload, dict) else None
        target = fixed_payload.get("target_soft_68") if isinstance(fixed_payload, dict) else None
        if not torch.is_tensor(feature) or list(feature.shape) != [384, 37, 37]:
            raise RuntimeError(f"CVSA hflip feature shape mismatch for {key}.")
        if not torch.is_tensor(target) or list(target.shape) != [1, 68, 68]:
            raise RuntimeError(f"CVSA hflip DABE-PU target shape mismatch for {key}.")
        for payload_name, payload in (
            ("feature", feature_payload),
            ("fixed", fixed_payload),
        ):
            if not isinstance(payload, dict):
                raise RuntimeError(f"CVSA {payload_name} payload must be a dict for {key}.")
            if "gt_path" in payload or payload.get("training_gt_used") is not False:
                raise RuntimeError(
                    f"CVSA {payload_name} payload leaks training GT for {key}."
                )
    return True, (
        f"feature={feature_manifest} | fixed={fixed_manifest} | samples={len(keys)} | "
        "provenance=independent_hflip_rgb->DINO->DABE-PU"
    )


def build_cvsa_diagnostic_payload(
    epoch, local_index, batch, image_68, student_prob_68, result
):
    index = int(local_index)
    resize = lambda value: F.interpolate(
        value[index : index + 1].detach().float(),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )[0].cpu()
    return {
        "schema_version": CVSA_DIAGNOSTIC_SCHEMA,
        "epoch": int(epoch),
        "route_mode": str(result["route_mode"]),
        "sample_index": int(batch["sample_index"][index]),
        "dataset": str(batch["dataset"][index]),
        "stem": str(batch["stem"][index]),
        "image_path": str(batch["image_path"][index]),
        "rgb": image_68[index].detach().float().cpu().clamp(0.0, 1.0),
        "fixed_original": result["fixed_a_68"][index].cpu(),
        "fixed_hflip_aligned": result["fixed_b_aligned_68"][index].cpu(),
        "teacher_original": result["teacher_a_68"][index].cpu(),
        "teacher_hflip_aligned": result["teacher_b_aligned_68"][index].cpu(),
        "student": student_prob_68[index].detach().float().cpu(),
        "fixed_eq_risk": resize(result["fixed_eq_risk_37"]),
        "teacher_eq_risk": resize(result["teacher_eq_risk_37"]),
        "fixed_sem_risk": resize(result["fixed_sem_risk_37"]),
        "teacher_sem_risk": resize(result["teacher_sem_risk_37"]),
        "fixed_total_risk": resize(result["fixed_total_risk_37"]),
        "teacher_total_risk": resize(result["teacher_total_risk_37"]),
        "route_target": resize(result["route_target_37"]),
        "router_gate": resize(result["router_gate_37"]),
        "mixed_target": result["mixed_target_68"][index].cpu(),
        "future_teacher_used": False,
        "global_schedule_used": False,
        "training_gt_used": False,
    }


def export_cvsa_visualization(payload, output_path, gt=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def array(tensor):
        value = tensor.detach().float().cpu()
        if value.ndim == 3 and value.shape[0] == 3:
            return value.permute(1, 2, 0).numpy()
        return value.squeeze().numpy()

    rgb = payload["rgb"]
    rgb_hflip = torch.flip(rgb, dims=[-1])
    gt_tensor = gt.detach().float().cpu() if torch.is_tensor(gt) else None
    first = [
        ("RGB original", rgb, None),
        ("RGB hflip", rgb_hflip, None),
        ("Fixed original", payload["fixed_original"], "gray"),
        ("Aligned fixed hflip", payload["fixed_hflip_aligned"], "gray"),
        ("Teacher original", payload["teacher_original"], "gray"),
        ("Aligned teacher hflip", payload["teacher_hflip_aligned"], "gray"),
        ("Student", payload["student"], "gray"),
        ("GT (diagnostic only)" if gt_tensor is not None else "GT not read in training", gt_tensor, "gray"),
    ]
    second = [
        ("Fixed equivariance risk", payload["fixed_eq_risk"], "magma"),
        ("Teacher equivariance risk", payload["teacher_eq_risk"], "magma"),
        ("Fixed semantic risk", payload["fixed_sem_risk"], "magma"),
        ("Teacher semantic risk", payload["teacher_sem_risk"], "magma"),
        ("Fixed total risk", payload["fixed_total_risk"], "magma"),
        ("Teacher total risk", payload["teacher_total_risk"], "magma"),
        ("Route target", payload["route_target"], "viridis"),
        ("Router gate", payload["router_gate"], "viridis"),
        ("Mixed target", payload["mixed_target"], "gray"),
    ]
    figure, axes = plt.subplots(2, 9, figsize=(27, 6.5))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (title, tensor, cmap) in zip(axes[0], first):
        axis.set_title(title, fontsize=8)
        if tensor is not None:
            axis.imshow(array(tensor), cmap=cmap, vmin=0.0, vmax=1.0)
    for axis, (title, tensor, cmap) in zip(axes[1], second):
        axis.set_title(title, fontsize=8)
        axis.imshow(array(tensor), cmap=cmap, vmin=0.0, vmax=1.0)
    figure.suptitle(
        "CVSA-v1: Cross-View Source Arbitration\n"
        f"route_mode={payload['route_mode']} | future_teacher=False | global_schedule=False",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
