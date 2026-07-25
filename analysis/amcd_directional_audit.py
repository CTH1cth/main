"""Core math and statistics for the AMCD-v0 directional-mimicry audit.

This module is deliberately independent from the training pipeline.  It consumes
already frozen patch features and a soft spatial partition, never ground truth.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


@dataclass(frozen=True)
class SlotExtraction:
    slots: torch.Tensor
    slot_mass: torch.Tensor
    assignment_map: torch.Tensor
    slot_validity: torch.Tensor
    initialization_indices: torch.Tensor
    reconstruction_diagnostic: float


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN/Inf.")


def _flatten_features(features: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _require_finite("features", features)
    if features.requires_grad:
        raise ValueError("AMCD-v0 audit inputs must have requires_grad=False.")
    if features.ndim == 3:
        channels, height, width = features.shape
        flat = features.reshape(channels, height * width).transpose(0, 1)
    elif features.ndim == 2:
        flat = features
        height, width = 1, int(features.shape[0])
    else:
        raise ValueError(
            f"features must be [C,H,W] or [N,C], got {list(features.shape)}."
        )
    if flat.shape[0] < 2 or flat.shape[1] < 1:
        raise ValueError(f"Invalid flattened feature shape: {list(flat.shape)}.")
    return F.normalize(flat.float(), dim=-1), int(height), int(width)


def _flatten_weights(
    spatial_weights: torch.Tensor,
    num_tokens: int,
    height: int,
    width: int,
) -> torch.Tensor:
    _require_finite("spatial_weights", spatial_weights)
    if spatial_weights.requires_grad:
        raise ValueError("AMCD-v0 audit inputs must have requires_grad=False.")
    weights = spatial_weights.float().reshape(-1)
    if weights.numel() != num_tokens:
        raise ValueError(
            "Feature/weight token mismatch: "
            f"N={num_tokens}, weights={weights.numel()}, grid={height}x{width}."
        )
    if float(weights.min().item()) < -1e-7 or float(weights.max().item()) > 1.0 + 1e-7:
        raise ValueError(
            f"spatial_weights must be in [0,1], got "
            f"[{float(weights.min().item())}, {float(weights.max().item())}]."
        )
    return weights.clamp(0.0, 1.0)


def _weighted_farthest_index(
    features: torch.Tensor,
    weights: torch.Tensor,
    centers: Sequence[torch.Tensor],
    unavailable: set[int],
) -> int:
    if centers:
        center_tensor = torch.stack(list(centers), dim=0)
        min_distance = (1.0 - features @ center_tensor.transpose(0, 1)).amin(dim=1)
        score = weights * min_distance.clamp_min(0.0)
    else:
        score = weights.clone()
    if unavailable:
        blocked = torch.tensor(
            sorted(unavailable), device=score.device, dtype=torch.long
        )
        score = score.clone()
        score.index_fill_(0, blocked, -float("inf"))
    # torch.argmax returns the first index for ties, which is the required rule.
    index = int(torch.argmax(score).item())
    if not math.isfinite(float(score[index].item())):
        raise RuntimeError("No token remains for deterministic slot initialization.")
    return index


@torch.no_grad()
def extract_weighted_slots(
    features: torch.Tensor,
    spatial_weights: torch.Tensor,
    num_slots: int,
    temperature: float,
    num_iterations: int = 10,
    eps: float = 1e-6,
    min_slot_mass: float = 1e-4,
) -> SlotExtraction:
    """Compress a softly weighted patch set into deterministic equal-capacity slots."""

    num_slots = int(num_slots)
    num_iterations = int(num_iterations)
    temperature = float(temperature)
    eps = float(eps)
    min_slot_mass = float(min_slot_mass)
    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if eps <= 0.0 or min_slot_mass < 0.0:
        raise ValueError("eps must be positive and min_slot_mass non-negative.")

    normalized, height, width = _flatten_features(features)
    weights = _flatten_weights(
        spatial_weights, normalized.shape[0], height, width
    )
    if num_slots > normalized.shape[0]:
        raise ValueError(
            f"num_slots={num_slots} exceeds token count {normalized.shape[0]}."
        )
    if float(weights.sum().item()) <= eps:
        raise ValueError("Weighted slot extraction received zero effective mass.")

    chosen: list[int] = []
    centers_list: list[torch.Tensor] = []
    for _ in range(num_slots):
        index = _weighted_farthest_index(
            normalized, weights, centers_list, set(chosen)
        )
        chosen.append(index)
        centers_list.append(normalized[index])
    centers = torch.stack(centers_list, dim=0)

    for _ in range(num_iterations):
        probabilities = torch.softmax(
            (normalized @ centers.transpose(0, 1)) / temperature, dim=-1
        )
        weighted_assignment = weights[:, None] * probabilities
        slot_mass = weighted_assignment.sum(dim=0)
        updated = weighted_assignment.transpose(0, 1) @ normalized
        updated = updated / slot_mass[:, None].clamp_min(eps)
        updated = F.normalize(updated, dim=-1)

        empty_slots = torch.nonzero(
            slot_mass < min_slot_mass, as_tuple=False
        ).flatten()
        if empty_slots.numel():
            active_centers = [
                updated[index]
                for index in range(num_slots)
                if float(slot_mass[index].item()) >= min_slot_mass
            ]
            unavailable = set(chosen)
            for slot_index_tensor in empty_slots:
                slot_index = int(slot_index_tensor.item())
                replacement = _weighted_farthest_index(
                    normalized, weights, active_centers, unavailable
                )
                chosen[slot_index] = replacement
                unavailable.add(replacement)
                updated[slot_index] = normalized[replacement]
                active_centers.append(normalized[replacement])
        centers = F.normalize(updated, dim=-1)

    probabilities = torch.softmax(
        (normalized @ centers.transpose(0, 1)) / temperature, dim=-1
    )
    weighted_assignment = weights[:, None] * probabilities
    slot_mass = weighted_assignment.sum(dim=0)
    slot_validity = slot_mass >= min_slot_mass
    hard_assignment = probabilities.argmax(dim=-1)
    hard_assignment = torch.where(
        weights > eps, hard_assignment, torch.full_like(hard_assignment, -1)
    )
    reconstruction = F.normalize(probabilities @ centers, dim=-1)
    residual = 1.0 - (normalized * reconstruction).sum(dim=-1)
    reconstruction_diagnostic = float(
        ((residual * weights).sum() / weights.sum().clamp_min(eps)).item()
    )
    for name, tensor in (
        ("slots", centers),
        ("slot_mass", slot_mass),
        ("assignment", weighted_assignment),
    ):
        _require_finite(name, tensor)
    return SlotExtraction(
        slots=centers.detach(),
        slot_mass=slot_mass.detach(),
        assignment_map=hard_assignment.reshape(height, width).detach(),
        slot_validity=slot_validity.detach(),
        initialization_indices=torch.tensor(
            chosen, device=features.device, dtype=torch.long
        ),
        reconstruction_diagnostic=reconstruction_diagnostic,
    )


@torch.no_grad()
def compute_bidirectional_coverage(
    object_slots: torch.Tensor,
    background_slots: torch.Tensor,
    temperature: float,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor | float]:
    """Compute equal-slot bidirectional conditional coverage residuals."""

    _require_finite("object_slots", object_slots)
    _require_finite("background_slots", background_slots)
    if object_slots.requires_grad or background_slots.requires_grad:
        raise ValueError("AMCD-v0 audit inputs must have requires_grad=False.")
    if object_slots.ndim != 2 or background_slots.ndim != 2:
        raise ValueError("Slots must be rank-2 tensors [K,C].")
    if object_slots.shape != background_slots.shape:
        raise ValueError(
            "Object/background slots must have equal capacity and shape, got "
            f"{list(object_slots.shape)} and {list(background_slots.shape)}."
        )
    temperature = float(temperature)
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("coverage temperature must be finite and positive.")
    object_slots = F.normalize(object_slots.float(), dim=-1)
    background_slots = F.normalize(background_slots.float(), dim=-1)
    cosine = object_slots @ background_slots.transpose(0, 1)

    background_to_object_weights = torch.softmax(cosine / temperature, dim=1)
    reconstructed_object = F.normalize(
        background_to_object_weights @ background_slots, dim=-1
    )
    residual_background_to_object = 1.0 - (
        object_slots * reconstructed_object
    ).sum(dim=-1)

    object_to_background_weights = torch.softmax(
        cosine.transpose(0, 1) / temperature, dim=1
    )
    reconstructed_background = F.normalize(
        object_to_background_weights @ object_slots, dim=-1
    )
    residual_object_to_background = 1.0 - (
        background_slots * reconstructed_background
    ).sum(dim=-1)

    e_background_to_object = residual_background_to_object.mean()
    e_object_to_background = residual_object_to_background.mean()
    asymmetry_raw = e_object_to_background - e_background_to_object
    asymmetry_normalized = asymmetry_raw / (
        e_object_to_background + e_background_to_object + float(eps)
    )
    result = {
        "cosine_matrix": cosine.detach(),
        "background_to_object_weights": background_to_object_weights.detach(),
        "object_to_background_weights": object_to_background_weights.detach(),
        "residual_background_to_object_slots": residual_background_to_object.detach(),
        "residual_object_to_background_slots": residual_object_to_background.detach(),
        "e_background_to_object": float(e_background_to_object.item()),
        "e_object_to_background": float(e_object_to_background.item()),
        "asymmetry_raw": float(asymmetry_raw.item()),
        "asymmetry_normalized": float(asymmetry_normalized.item()),
    }
    if not all(
        math.isfinite(float(result[key]))
        for key in (
            "e_background_to_object",
            "e_object_to_background",
            "asymmetry_raw",
            "asymmetry_normalized",
        )
    ):
        raise RuntimeError("Bidirectional coverage produced NaN/Inf.")
    return result


@torch.no_grad()
def compute_asymmetry(
    features: torch.Tensor,
    soft_mask: torch.Tensor,
    num_slots: int,
    slot_temperature: float,
    coverage_temperature: float,
    num_iterations: int = 10,
    eps: float = 1e-6,
    min_mass: float = 4.0,
) -> dict:
    """Extract equal slots from a soft partition and calculate directional asymmetry."""

    normalized, height, width = _flatten_features(features)
    del normalized
    mask = soft_mask.float().reshape(height, width)
    _require_finite("soft_mask", mask)
    if soft_mask.requires_grad:
        raise ValueError("AMCD-v0 audit inputs must have requires_grad=False.")
    if float(mask.min().item()) < -1e-7 or float(mask.max().item()) > 1.0 + 1e-7:
        raise ValueError("soft_mask must be in [0,1].")
    mask = mask.clamp(0.0, 1.0)
    object_mass = float(mask.sum().item())
    background_mass = float((1.0 - mask).sum().item())
    invalid_reasons = []
    if object_mass < float(min_mass):
        invalid_reasons.append("invalid object mass")
    if background_mass < float(min_mass):
        invalid_reasons.append("invalid background mass")
    if invalid_reasons:
        return {
            "valid": False,
            "invalid_reason": "; ".join(invalid_reasons),
            "object_mass": object_mass,
            "background_mass": background_mass,
        }

    object_result = extract_weighted_slots(
        features,
        mask,
        num_slots=num_slots,
        temperature=slot_temperature,
        num_iterations=num_iterations,
        eps=eps,
    )
    background_result = extract_weighted_slots(
        features,
        1.0 - mask,
        num_slots=num_slots,
        temperature=slot_temperature,
        num_iterations=num_iterations,
        eps=eps,
    )
    coverage = compute_bidirectional_coverage(
        object_result.slots,
        background_result.slots,
        temperature=coverage_temperature,
        eps=eps,
    )
    return {
        "valid": True,
        "invalid_reason": "",
        "object_mass": object_mass,
        "background_mass": background_mass,
        "object_slots": object_result,
        "background_slots": background_result,
        **coverage,
    }


def mask_geometry_statistics(mask: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    mask = mask.detach().float().squeeze()
    _require_finite("mask", mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be spatial [H,W], got {list(mask.shape)}.")
    mask = mask.clamp(0.0, 1.0)
    entropy = -(
        mask * torch.log(mask.clamp_min(eps))
        + (1.0 - mask) * torch.log((1.0 - mask).clamp_min(eps))
    ).mean()
    horizontal = (mask[:, 1:] - mask[:, :-1]).abs().reshape(-1)
    vertical = (mask[1:, :] - mask[:-1, :]).abs().reshape(-1)
    boundary_density = torch.cat((horizontal, vertical)).mean()
    return {
        "soft_object_area": float(mask.mean().item()),
        "soft_background_area": float((1.0 - mask).mean().item()),
        "mask_entropy": float(entropy.item()),
        "mask_boundary_density": float(boundary_density.item()),
        "connectedness_proxy": float((1.0 - boundary_density).item()),
    }


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def permuted_mask(mask: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    flat = mask.detach().cpu().float().reshape(-1)
    order = torch.randperm(flat.numel(), generator=generator)
    return flat[order].reshape(mask.shape).to(mask.device)


def rolled_mask(mask: torch.Tensor, seed: int, min_patch_shift: int = 4) -> tuple[torch.Tensor, int, int]:
    if mask.ndim != 2:
        raise ValueError("rolled_mask expects [H,W].")
    height, width = (int(value) for value in mask.shape)
    min_patch_shift = int(min_patch_shift)
    y_choices = [
        value
        for value in range(1, height)
        if min(value, height - value) >= min_patch_shift
    ]
    x_choices = [
        value
        for value in range(1, width)
        if min(value, width - value) >= min_patch_shift
    ]
    if not y_choices or not x_choices:
        raise ValueError(
            f"Grid {height}x{width} cannot support min shift {min_patch_shift}."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    y_index = int(torch.randint(len(y_choices), (1,), generator=generator).item())
    x_index = int(torch.randint(len(x_choices), (1,), generator=generator).item())
    dy, dx = y_choices[y_index], x_choices[x_index]
    return torch.roll(mask, shifts=(dy, dx), dims=(0, 1)), dy, dx


def descriptive_statistics(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in (
            "mean", "std", "median", "p10", "p25", "p75", "p90", "positive_ratio"
        )} | {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "positive_ratio": float((array > 0.0).mean()),
    }


def bootstrap_confidence_interval(
    values: Iterable[float],
    statistic: str,
    num_bootstrap: int,
    seed: int,
    chunk_size: int = 256,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic!r}.")
    num_bootstrap = int(num_bootstrap)
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(num_bootstrap, dtype=np.float64)
    for start in range(0, num_bootstrap, int(chunk_size)):
        stop = min(num_bootstrap, start + int(chunk_size))
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        samples = array[indices]
        if statistic == "mean":
            estimates[start:stop] = samples.mean(axis=1)
        else:
            estimates[start:stop] = np.median(samples, axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def paired_rank_biserial(differences: Iterable[float]) -> float:
    values = np.asarray(list(differences), dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0.0)]
    if not values.size:
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0.0].sum())
    negative = float(ranks[values < 0.0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def paired_comparison_statistics(
    differences: Iterable[float],
    num_bootstrap: int,
    seed: int,
    eps: float = 1e-12,
) -> dict[str, float | int | list[float]]:
    values = np.asarray(list(differences), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "mean_ci95": [float("nan"), float("nan")],
            "median_ci95": [float("nan"), float("nan")],
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
            "rank_biserial": float("nan"),
            "d_z": float("nan"),
        }
    if np.all(values == 0.0):
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        test = stats.wilcoxon(
            values, zero_method="wilcox", correction=False, alternative="two-sided"
        )
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_pvalue = float(test.pvalue)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    mean_ci = bootstrap_confidence_interval(
        values,
        statistic="mean",
        num_bootstrap=num_bootstrap,
        seed=stable_seed(seed, "mean"),
    )
    median_ci = bootstrap_confidence_interval(
        values,
        statistic="median",
        num_bootstrap=num_bootstrap,
        seed=stable_seed(seed, "median"),
    )
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_ci95": [mean_ci[0], mean_ci[1]],
        "median_ci95": [median_ci[0], median_ci[1]],
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "rank_biserial": float(paired_rank_biserial(values)),
        "d_z": float(values.mean() / (std + float(eps))),
    }


def spearman_statistics(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[valid], y_array[valid]
    if x_array.size < 2 or np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return {"count": int(x_array.size), "rho": float("nan"), "pvalue": float("nan")}
    result = stats.spearmanr(x_array, y_array)
    return {
        "count": int(x_array.size),
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def geometry_linear_explanation(
    asymmetry: Iterable[float],
    area: Iterable[float],
    entropy: Iterable[float],
    boundary_density: Iterable[float],
    eps: float = 1e-12,
) -> dict[str, float | int | dict[str, float]]:
    y = np.asarray(list(asymmetry), dtype=np.float64)
    predictors = np.column_stack(
        [
            np.asarray(list(area), dtype=np.float64),
            np.asarray(list(entropy), dtype=np.float64),
            np.asarray(list(boundary_density), dtype=np.float64),
        ]
    )
    valid = np.isfinite(y) & np.isfinite(predictors).all(axis=1)
    y, predictors = y[valid], predictors[valid]
    names = ["intercept", "area", "entropy", "boundary_density"]
    if y.size < len(names) + 1:
        return {
            "count": int(y.size),
            "r_squared": float("nan"),
            "adjusted_r_squared": float("nan"),
            "coefficients": {name: float("nan") for name in names},
        }
    design = np.column_stack((np.ones(y.size), predictors))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual_sum = float(np.square(y - prediction).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - residual_sum / (total_sum + float(eps))
    num_predictors = predictors.shape[1]
    adjusted = 1.0 - (1.0 - r_squared) * (y.size - 1) / max(
        1, y.size - num_predictors - 1
    )
    return {
        "count": int(y.size),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted),
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
    }


@torch.no_grad()
def run_analytic_assertions(device: torch.device | str = "cpu") -> dict[str, float | bool]:
    """Run formula-level invariants before touching real cache samples."""

    device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260722)
    features = torch.randn(16, 6, generator=generator).to(device)
    slots = F.normalize(features[:8], dim=-1)
    identical = compute_bidirectional_coverage(slots, slots.clone(), 0.10)
    identical_abs = abs(float(identical["asymmetry_raw"]))
    if identical_abs >= 1e-6:
        raise AssertionError(f"Identical-set asymmetry failed: {identical_abs}.")

    permutation = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4], device=device)
    before = compute_bidirectional_coverage(slots, F.normalize(features[8:], dim=-1), 0.10)
    after = compute_bidirectional_coverage(
        slots[permutation], F.normalize(features[8:], dim=-1)[permutation.flip(0)], 0.10
    )
    permutation_error = abs(
        float(before["asymmetry_raw"]) - float(after["asymmetry_raw"])
    )
    if permutation_error >= 1e-6:
        raise AssertionError(f"Slot-order invariance failed: {permutation_error}.")

    spatial_features = torch.randn(12, 7, 7, generator=generator).to(device)
    mask = torch.rand(7, 7, generator=generator).to(device)
    real = compute_asymmetry(spatial_features, mask, 4, 0.10, 0.10, 4)
    inverse = compute_asymmetry(spatial_features, 1.0 - mask, 4, 0.10, 0.10, 4)
    inverse_error = abs(
        float(real["asymmetry_raw"]) + float(inverse["asymmetry_raw"])
    )
    if inverse_error >= 1e-5:
        raise AssertionError(f"Inverse sign-flip failed: {inverse_error}.")
    return {
        "identical_set_abs_asymmetry": identical_abs,
        "slot_order_abs_error": permutation_error,
        "inverse_sign_flip_abs_error": inverse_error,
        "all_inputs_require_grad_false": not any(
            tensor.requires_grad for tensor in (features, slots, spatial_features, mask)
        ),
    }
