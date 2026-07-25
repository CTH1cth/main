"""DABE-faithful background substitutes and sparse grouped interventions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn.functional as F


SUPPORTED_SUBSTITUTION_MODES = (
    "weighted_bg",
    "nearest_bg",
    "random_bg",
    "weighted_bg_angle_matched",
    "random_bg_angle_matched",
    "identity",
)

SUPPORTED_SUBSTITUTE_NORM_MODES = (
    "dabe_unit",
    "query_match",
)


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf")


def _rotate_to_angle(
    query_normalized: torch.Tensor,
    candidate_normalized: torch.Tensor,
    target_angle: torch.Tensor,
) -> torch.Tensor:
    """Rotate query toward candidate while imposing a per-token angle.

    The result follows the candidate's great-circle direction, but its angular
    displacement from the query is exactly ``target_angle``.  A deterministic
    orthogonal direction handles the numerically degenerate antipodal case.
    """
    cosine = (query_normalized * candidate_normalized).sum(dim=1).clamp(-1.0, 1.0)
    tangent = candidate_normalized - cosine.unsqueeze(1) * query_normalized
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)

    basis_index = query_normalized.abs().argmin(dim=1)
    basis = torch.zeros_like(query_normalized)
    basis.scatter_(1, basis_index.unsqueeze(1), 1.0)
    fallback = basis - (basis * query_normalized).sum(dim=1, keepdim=True) * query_normalized
    fallback = F.normalize(fallback, dim=1, p=2)
    tangent_direction = torch.where(
        tangent_norm > 1e-7,
        tangent / tangent_norm.clamp_min(1e-12),
        fallback,
    )
    rotated = (
        torch.cos(target_angle).unsqueeze(1) * query_normalized
        + torch.sin(target_angle).unsqueeze(1) * tangent_direction
    )
    return F.normalize(rotated, dim=1, p=2)


def coordinate_group_ids(
    height: int,
    width: int,
    group_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return g(y,x)=group_size*(y mod group_size)+(x mod group_size)."""
    height = int(height)
    width = int(width)
    group_size = int(group_size)
    if height < 1 or width < 1:
        raise ValueError(f"Invalid grid: {height}x{width}")
    if group_size not in {3, 4}:
        raise ValueError("BISA-v0 supports only group_size 3 or 4")
    yy = torch.arange(height, device=device).view(height, 1)
    xx = torch.arange(width, device=device).view(1, width)
    group_ids = group_size * (yy % group_size) + (xx % group_size)
    expected = set(range(group_size * group_size))
    actual = set(int(value) for value in torch.unique(group_ids).cpu().tolist())
    if actual != expected:
        raise RuntimeError(f"Grouped intervention coverage mismatch: {actual} != {expected}")
    return group_ids.long()


def apply_group_interventions(
    feature: torch.Tensor,
    substitute: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Build one counterfactual feature map per coordinate-colour group."""
    if feature.ndim != 3 or substitute.shape != feature.shape:
        raise RuntimeError(
            "feature/substitute must share [C,H,W], got "
            f"{list(feature.shape)}/{list(substitute.shape)}"
        )
    channels, height, width = feature.shape
    if tuple(group_ids.shape) != (height, width):
        raise RuntimeError(
            f"group_ids shape {list(group_ids.shape)} != {[height, width]}"
        )
    num_groups = int(group_ids.max().item()) + 1
    stacked = feature.unsqueeze(0).expand(num_groups, channels, height, width).clone()
    substitute_expanded = substitute.unsqueeze(0).expand_as(stacked)
    masks = torch.stack([group_ids == group for group in range(num_groups)], dim=0)
    stacked = torch.where(masks.unsqueeze(1), substitute_expanded, stacked)
    coverage = masks.to(torch.int16).sum(dim=0)
    if not torch.equal(coverage, torch.ones_like(coverage)):
        raise RuntimeError("Every token must be replaced by exactly one intervention group")
    return stacked.detach()


def _local_mean_3x3(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise RuntimeError(f"Expected [G,H,W], got {list(values.shape)}")
    values4 = values.unsqueeze(1)
    kernel = values.new_ones((1, 1, 3, 3))
    numerator = F.conv2d(values4, kernel, padding=1)
    denominator = F.conv2d(torch.ones_like(values4), kernel, padding=1)
    return (numerator / denominator.clamp_min(1.0)).squeeze(1)


def assemble_group_responses(
    original_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Assemble token self/local responses and group spillover statistics."""
    if original_logits.ndim == 3 and int(original_logits.shape[0]) == 1:
        original_logits = original_logits[0]
    if counterfactual_logits.ndim == 4 and int(counterfactual_logits.shape[1]) == 1:
        counterfactual_logits = counterfactual_logits[:, 0]
    if original_logits.ndim != 2 or counterfactual_logits.ndim != 3:
        raise RuntimeError(
            "Expected original [H,W] and counterfactual [G,H,W], got "
            f"{list(original_logits.shape)}/{list(counterfactual_logits.shape)}"
        )
    num_groups, height, width = counterfactual_logits.shape
    if tuple(original_logits.shape) != (height, width):
        raise RuntimeError("Original/counterfactual logit shapes do not align")
    if tuple(group_ids.shape) != (height, width):
        raise RuntimeError("group_ids/logit shapes do not align")
    if num_groups != int(group_ids.max().item()) + 1:
        raise RuntimeError("Counterfactual batch does not cover every group")

    diff = original_logits.unsqueeze(0) - counterfactual_logits
    _finite(diff, "group logit response")
    local = _local_mean_3x3(diff)
    medians = diff.flatten(1).median(dim=1).values
    gather_index = group_ids.unsqueeze(0)
    self_raw = torch.gather(diff, 0, gather_index).squeeze(0)
    local_raw = torch.gather(local, 0, gather_index).squeeze(0)
    center_map = medians[group_ids]

    spill = []
    in_abs = []
    out_abs = []
    for group in range(num_groups):
        inside = group_ids == group
        outside = ~inside
        response_abs = diff[group].abs()
        d_in = response_abs[inside].mean()
        d_out = response_abs[outside].mean()
        in_abs.append(d_in)
        out_abs.append(d_out)
        spill.append(d_out / (d_in + float(eps)))

    result = {
        "group_response": diff.detach(),
        "group_median": medians.detach(),
        "c_self_raw": self_raw.detach(),
        "c_local_raw": local_raw.detach(),
        "c_self_centered": (self_raw - center_map).detach(),
        "c_local_centered": (local_raw - center_map).detach(),
        "spill_ratio_per_group": torch.stack(spill).detach(),
        "spill_in_abs_per_group": torch.stack(in_abs).detach(),
        "spill_out_abs_per_group": torch.stack(out_abs).detach(),
        "spill_ratio_map": torch.stack(spill)[group_ids].detach(),
    }
    for name, tensor in result.items():
        _finite(tensor, name)
    return result


def _stable_seed(seed: int, sample_key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _nearest_anchor_distance(
    values: torch.Tensor,
    anchors: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    chunks = []
    for start in range(0, int(values.shape[0]), int(chunk_size)):
        distance = torch.cdist(values[start : start + chunk_size].float(), anchors.float())
        chunks.append(distance.min(dim=1).values)
    return torch.cat(chunks, dim=0)


@dataclass(frozen=True)
class BackgroundSubstitutionResult:
    substitutes: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor | float | int | str]


@torch.no_grad()
def build_dabe_background_substitutes(
    feature: torch.Tensor,
    rgb_grid: torch.Tensor,
    bg_anchor_mask: torch.Tensor,
    params: dict,
    modes: Iterable[str],
    *,
    seed: int,
    sample_key: str,
    norm_mode: str = "dabe_unit",
) -> BackgroundSubstitutionResult:
    """Recompute the DABE top-K background reconstruction without GT.

    Similarity, colour term, top-K, temperature and direction reconstruction
    are identical to ``common.dabe_pseudo._background_residual``.

    ``dabe_unit`` preserves the unit-norm DABE reconstruction. ``query_match``
    is an explicit intervention adapter: it preserves the exact DABE direction
    but multiplies every weighted/nearest/random substitute by the norm of its
    corresponding raw query token.  Thus all controls change direction while
    holding per-token feature magnitude fixed.
    """
    modes = tuple(dict.fromkeys(str(mode).strip().lower() for mode in modes))
    invalid = sorted(set(modes).difference(SUPPORTED_SUBSTITUTION_MODES))
    if invalid:
        raise ValueError(f"Unsupported substitution mode(s): {invalid}")
    norm_mode = str(norm_mode).strip().lower()
    if norm_mode not in SUPPORTED_SUBSTITUTE_NORM_MODES:
        raise ValueError(
            f"Unsupported substitute norm mode {norm_mode!r}; "
            f"expected one of {SUPPORTED_SUBSTITUTE_NORM_MODES}"
        )
    if feature.ndim != 3:
        raise RuntimeError(f"feature must be [C,H,W], got {list(feature.shape)}")
    channels, height, width = feature.shape
    if tuple(rgb_grid.shape) != (3, height, width):
        raise RuntimeError(
            f"rgb_grid must be [3,H,W], got {list(rgb_grid.shape)}"
        )
    if bg_anchor_mask.ndim == 3 and int(bg_anchor_mask.shape[0]) == 1:
        bg_anchor_mask = bg_anchor_mask[0]
    if tuple(bg_anchor_mask.shape) != (height, width):
        raise RuntimeError("bg_anchor_mask does not align with feature grid")

    raw = feature.permute(1, 2, 0).reshape(height * width, channels).float()
    feat_n = F.normalize(raw, dim=1, p=2)
    rgb = rgb_grid.permute(1, 2, 0).reshape(height * width, 3).float()
    anchor_indices = torch.where(bg_anchor_mask.reshape(-1) > 0.5)[0]
    if int(anchor_indices.numel()) < 1:
        raise RuntimeError("DABE background anchor set is empty")
    anchor_feat_n = feat_n.index_select(0, anchor_indices)
    anchor_rgb = rgb.index_select(0, anchor_indices)

    k = min(int(params.get("K_RECON", 32)), int(anchor_indices.numel()))
    if k < 1:
        raise RuntimeError("DABE K_RECON resolved to zero")
    tau = float(params.get("TAU_RECON", 0.07))
    sigma_color = float(params.get("SIGMA_COLOR_RECON", 0.05))
    lambda_color = float(params.get("LAMBDA_COLOR_RECON", 0.20))
    if tau <= 0.0 or sigma_color <= 0.0:
        raise RuntimeError("Invalid DABE reconstruction temperature")

    weighted_chunks = []
    nearest_chunks = []
    nearest_local_chunks = []
    entropy_chunks = []
    for start in range(0, int(raw.shape[0]), 512):
        end = min(start + 512, int(raw.shape[0]))
        query_n = feat_n[start:end]
        query_rgb = rgb[start:end]
        sim_feat = query_n @ anchor_feat_n.t()
        color_dist2 = torch.cdist(query_rgb, anchor_rgb, p=2.0).square()
        sim_color = torch.exp(-color_dist2 / sigma_color)
        similarity = sim_feat + lambda_color * sim_color
        top_value, top_local = torch.topk(similarity, k=k, dim=1)
        weights = torch.softmax(top_value / tau, dim=1)
        selected_normalized = anchor_feat_n[top_local]
        weighted_chunks.append(
            F.normalize(
                (weights.unsqueeze(-1) * selected_normalized).sum(dim=1),
                dim=1,
                p=2,
            )
        )
        nearest_chunks.append(anchor_feat_n[top_local[:, 0]])
        nearest_local_chunks.append(top_local[:, 0])
        entropy_chunks.append(-(weights * torch.log(weights + 1e-12)).sum(dim=1))

    weighted_normalized = torch.cat(weighted_chunks, dim=0)
    nearest_normalized = torch.cat(nearest_chunks, dim=0)
    nearest_local_index = torch.cat(nearest_local_chunks, dim=0)
    retrieval_entropy = torch.cat(entropy_chunks, dim=0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, sample_key))
    random_local_cpu = torch.randint(
        low=0,
        high=int(anchor_indices.numel()),
        size=(height * width,),
        generator=generator,
        device="cpu",
    )
    random_local = random_local_cpu.to(device=raw.device)
    random_normalized = anchor_feat_n[random_local]

    weighted_angle = torch.acos(
        (feat_n * weighted_normalized).sum(dim=1).clamp(-1.0, 1.0)
    )
    random_angle = torch.acos(
        (feat_n * random_normalized).sum(dim=1).clamp(-1.0, 1.0)
    )
    matched_target_angle = torch.minimum(weighted_angle, random_angle)
    weighted_angle_matched_normalized = _rotate_to_angle(
        feat_n, weighted_normalized, matched_target_angle
    )
    random_angle_matched_normalized = _rotate_to_angle(
        feat_n, random_normalized, matched_target_angle
    )

    original_norm = torch.linalg.vector_norm(raw, dim=1)
    if norm_mode == "query_match":
        query_scale = original_norm.unsqueeze(1)
        weighted_value = weighted_normalized * query_scale
        nearest_value = nearest_normalized * query_scale
        random_value = random_normalized * query_scale
        weighted_angle_matched_value = weighted_angle_matched_normalized * query_scale
        random_angle_matched_value = random_angle_matched_normalized * query_scale
    else:
        weighted_value = weighted_normalized
        nearest_value = nearest_normalized
        random_value = random_normalized
        weighted_angle_matched_value = weighted_angle_matched_normalized
        random_angle_matched_value = random_angle_matched_normalized

    flattened = {
        "weighted_bg": weighted_value,
        "nearest_bg": nearest_value,
        "random_bg": random_value,
        "weighted_bg_angle_matched": weighted_angle_matched_value,
        "random_bg_angle_matched": random_angle_matched_value,
        "identity": raw,
    }
    substitutes = {
        mode: flattened[mode]
        .reshape(height, width, channels)
        .permute(2, 0, 1)
        .contiguous()
        .detach()
        for mode in modes
    }
    for mode, tensor in substitutes.items():
        _finite(tensor, f"{mode} substitute")
        if tensor.requires_grad:
            raise RuntimeError(f"{mode} substitute must be detached")

    if norm_mode == "query_match":
        tolerance = 1e-5 * original_norm.clamp_min(1.0)
        for mode in (
            "weighted_bg",
            "nearest_bg",
            "random_bg",
            "weighted_bg_angle_matched",
            "random_bg_angle_matched",
        ):
            substituted_norm = torch.linalg.vector_norm(flattened[mode], dim=1)
            error = (substituted_norm - original_norm).abs()
            if not bool((error <= tolerance).all().item()):
                raise RuntimeError(
                    f"query_match norm invariant failed for {mode}: "
                    f"max_abs_error={float(error.max().item()):.9g}"
                )
    matched_weighted_gap = 1.0 - F.cosine_similarity(
        raw, weighted_angle_matched_value, dim=1
    )
    matched_random_gap = 1.0 - F.cosine_similarity(
        raw, random_angle_matched_value, dim=1
    )
    if not bool(
        torch.allclose(matched_weighted_gap, matched_random_gap, atol=5e-6, rtol=5e-6)
    ):
        raise RuntimeError(
            "Angle-matched weighted/random substitutes do not share the same angle: "
            f"max_gap_error={float((matched_weighted_gap - matched_random_gap).abs().max().item()):.9g}"
        )

    diagnostics: Dict[str, torch.Tensor | float | int | str] = {
        "substitute_norm_mode": norm_mode,
        "background_anchor_count": int(anchor_indices.numel()),
        "anchor_indices": anchor_indices.detach(),
        "nearest_anchor_indices": anchor_indices[nearest_local_index].detach(),
        "random_anchor_indices": anchor_indices[random_local].detach(),
        "retrieval_weight_entropy": retrieval_entropy.detach(),
        "original_feature_norm": original_norm.detach(),
        "angle_matched_target_rad": matched_target_angle.detach(),
        "bg_cosine_gap": (
            1.0 - F.cosine_similarity(raw, weighted_value, dim=1)
        ).detach(),
        "bg_feature_l2_distance": torch.linalg.vector_norm(
            raw - weighted_value, dim=1
        ).detach(),
    }
    for mode in flattened:
        values = flattened[mode]
        cosine = F.cosine_similarity(raw, values, dim=1).clamp(-1.0, 1.0)
        diagnostics[f"{mode}_feature_norm"] = torch.linalg.vector_norm(
            values, dim=1
        ).detach()
        diagnostics[f"{mode}_cosine_gap"] = (1.0 - cosine).detach()
        diagnostics[f"{mode}_angle_rad"] = torch.acos(cosine).detach()
        diagnostics[f"{mode}_angle_deg"] = torch.rad2deg(torch.acos(cosine)).detach()
        diagnostics[f"{mode}_feature_l2_distance"] = torch.linalg.vector_norm(
            raw - values, dim=1
        ).detach()
        diagnostics[f"{mode}_nearest_anchor_distance"] = (
            _nearest_anchor_distance(
                F.normalize(values, dim=1, p=2), anchor_feat_n
            ).detach()
        )
    for name, value in diagnostics.items():
        if torch.is_tensor(value):
            _finite(value.float(), name)
    return BackgroundSubstitutionResult(substitutes=substitutes, diagnostics=diagnostics)
