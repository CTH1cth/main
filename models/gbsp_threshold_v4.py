"""GT-free GBSP V4 tail purification and residual-coherence methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from scipy.ndimage import median_filter
from scipy.stats import rankdata
from skimage.filters import threshold_multiotsu, threshold_otsu


@dataclass(frozen=True)
class ThresholdV4Result:
    mask: torch.Tensor
    maps: dict[str, torch.Tensor]
    threshold_high: float | None
    threshold_low: float | None
    numerical_failure: bool
    diagnostics: dict[str, Any]


def _score(value: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...]]:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise ValueError("score must be a non-empty tensor")
    array = value.detach().cpu().double().numpy().copy()
    if not np.isfinite(array).all() or array.min() < -1e-7 or array.max() > 1.0 + 1e-7:
        raise ValueError("score must be finite and inside [0,1]")
    return np.clip(array.reshape(-1), 0.0, 1.0), tuple(array.shape)


def _indices(value: torch.Tensor, size: int) -> np.ndarray:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0:
        raise ValueError("background_indices must be a non-empty vector")
    indices = value.detach().cpu().long().numpy().copy()
    if indices.min() < 0 or indices.max() >= size or np.unique(indices).size != indices.size:
        raise ValueError("background_indices are invalid or duplicated")
    return indices


def _tensor(value: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(shape).copy())


def _result(
    mask: np.ndarray,
    shape: tuple[int, ...],
    maps: dict[str, np.ndarray],
    high: float | None,
    low: float | None,
    failure: bool,
    diagnostics: dict[str, Any],
) -> ThresholdV4Result:
    return ThresholdV4Result(
        mask=_tensor(np.asarray(mask, dtype=np.float32), shape),
        maps={name: _tensor(value, shape) for name, value in maps.items()},
        threshold_high=None if high is None else float(high),
        threshold_low=None if low is None else float(low),
        numerical_failure=bool(failure),
        diagnostics=diagnostics,
    )


def empirical_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("rank input contains NaN/Inf")
    return (rankdata(values, method="average") - 0.5) / values.size


class SeededHysteresis:
    """Retain only 8-connected support components containing a seed."""

    @staticmethod
    def apply(seed: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        seed = np.asarray(seed, dtype=bool)
        support = np.asarray(support, dtype=bool)
        if seed.shape != support.shape or seed.ndim != 2:
            raise ValueError("seed/support must be matching 2-D maps")
        labels, count = connected_components(support, structure=np.ones((3, 3), np.uint8))
        kept = np.unique(labels[seed])
        kept = kept[kept != 0]
        final = np.isin(labels, kept)
        seed_labels, seed_count = connected_components(seed, structure=np.ones((3, 3), np.uint8))
        del seed_labels
        diagnostics = {
            "num_seed_components": int(seed_count),
            "num_support_components": int(count),
            "num_retained_components": int(kept.size),
            "seed_area": float(seed.mean()),
            "support_area": float(support.mean()),
            "final_area": float(final.mean()),
            "seed_outside_support_count": int(np.sum(seed & ~support)),
        }
        return final, diagnostics


class HierarchicalUpperTailOtsu:
    """Two-level Otsu tail core plus optional adaptive seeded recovery."""

    def __init__(
        self,
        variant: str = "core",
        classes: int = 3,
        nbins: int = 256,
        upper_nbins: int = 256,
        min_tail_count: int = 16,
        min_tail_unique: int = 8,
        min_bridge_count: int = 8,
    ) -> None:
        if variant not in {"core", "mid_h", "med_h"}:
            raise ValueError(variant)
        if (classes, nbins, upper_nbins) != (3, 256, 256):
            raise ValueError("V4 Otsu settings are frozen")
        self.variant = variant
        self.classes = classes
        self.nbins = nbins
        self.upper_nbins = upper_nbins
        self.min_tail_count = min_tail_count
        self.min_tail_unique = min_tail_unique
        self.min_bridge_count = min_bridge_count

    def apply(self, minmax_score: torch.Tensor, background_indices: torch.Tensor) -> ThresholdV4Result:
        score, shape = _score(minmax_score)
        bc = _indices(background_indices, score.size)
        if np.unique(score).size < 3:
            return _result(
                np.zeros(score.size, bool), shape, {}, None, None, True,
                {"failure_reason": "first_level_unique_values_below_three"},
            )
        t1, t2 = map(float, threshold_multiotsu(score, classes=3, nbins=256))
        upper = score[score > t2]
        invalid = upper.size < self.min_tail_count or np.unique(upper).size < self.min_tail_unique
        if invalid:
            diagnostics = {
                "t1": t1, "t2": t2, "t3": None,
                "upper_tail_patch_count": int(upper.size),
                "upper_tail_unique_values": int(np.unique(upper).size),
                "huto_upper_tail_invalid": True,
                "failure_reason": "invalid_upper_tail",
            }
            return _result(np.zeros(score.size, bool), shape, {}, None, None, True, diagnostics)
        t3 = float(threshold_otsu(upper.astype(np.float64), nbins=self.upper_nbins))
        if not t3 > t2:
            return _result(
                np.zeros(score.size, bool), shape, {}, None, None, True,
                {"t1": t1, "t2": t2, "t3": t3, "huto_upper_tail_invalid": True,
                 "failure_reason": "second_threshold_not_above_first_tail_threshold"},
            )
        seed = score > t3
        bridge = score[(score > t2) & (score <= t3)]
        threshold_low = None
        support = seed.copy()
        connectivity = {}
        if self.variant == "mid_h":
            threshold_low = float((t2 + t3) / 2.0)
            support = score > threshold_low
        elif self.variant == "med_h":
            if bridge.size < self.min_bridge_count:
                return _result(
                    np.zeros(score.size, bool), shape, {}, t3, None, True,
                    {"t1": t1, "t2": t2, "t3": t3, "bridge_patch_count": int(bridge.size),
                     "failure_reason": "bridge_patch_count_below_eight"},
                )
            threshold_low = float(np.median(bridge))
            support = score > threshold_low
        if self.variant == "core":
            final = seed
            seed_grid = seed.reshape(shape[-2:])
            _, seed_count = connected_components(seed_grid, structure=np.ones((3, 3), np.uint8))
            connectivity = {
                "num_seed_components": int(seed_count),
                "num_support_components": int(seed_count),
                "num_retained_components": int(seed_count),
            }
        else:
            final_grid, connectivity = SeededHysteresis.apply(
                seed.reshape(shape[-2:]), support.reshape(shape[-2:])
            )
            final = final_grid.reshape(-1)
        state = np.zeros(score.size, np.int64)
        state[score > t1] = 1
        state[score > t2] = 2
        diagnostics = {
            "t1": t1, "t2": t2, "t3": t3,
            "class0_area": float(np.mean(score <= t1)),
            "class1_area": float(np.mean((score > t1) & (score <= t2))),
            "class2_area": float(np.mean(score > t2)),
            "core_area": float(seed.mean()),
            "upper_tail_patch_count": int(upper.size),
            "upper_tail_unique_values": int(np.unique(upper).size),
            "bridge_patch_count": int(bridge.size),
            "huto_upper_tail_invalid": False,
            "threshold_high": t3,
            "threshold_low": threshold_low,
            "threshold_low_rule": self.variant,
            "seed_area": float(seed.mean()),
            "support_area": float(support.mean()),
            "final_area": float(final.mean()),
            "bc_above_t2_ratio": float(np.mean(score[bc] > t2)),
            "bc_above_t3_ratio": float(np.mean(seed[bc])),
            "bc_seed_ratio": float(np.mean(seed[bc])),
            "bc_support_ratio": float(np.mean(support[bc])),
            "bc_final_ratio": float(np.mean(final[bc])),
            **connectivity,
            "failure_reason": None,
        }
        return _result(
            final, shape,
            {"state_first_level": state, "mask_seed": seed, "mask_support": support},
            t3, threshold_low, False, diagnostics,
        )


def _multiotsu_seed(score: np.ndarray) -> tuple[np.ndarray, float, float]:
    if np.unique(score).size < 3:
        raise ValueError("seed score has fewer than three unique values")
    low, high = map(float, threshold_multiotsu(score, classes=3, nbins=256))
    return score > high, low, high


def _huto_median_support(
    score_tensor: torch.Tensor, background_indices: torch.Tensor
) -> tuple[np.ndarray, tuple[int, ...], dict[str, Any]]:
    huto = HierarchicalUpperTailOtsu("med_h").apply(score_tensor, background_indices)
    if huto.numerical_failure:
        raise RuntimeError(f"HUTO median support invalid: {huto.diagnostics}")
    support = huto.maps["mask_support"].numpy().astype(bool).reshape(-1)
    return support, tuple(huto.mask.shape), huto.diagnostics


class OrthogonalResidualCoherence:
    """Rank-fused residual magnitude/direction-coherence seeds with HUTO support."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)

    @staticmethod
    def coherence(residual_vectors: np.ndarray, height: int, width: int, eps: float) -> np.ndarray:
        residual = np.asarray(residual_vectors, dtype=np.float64)
        if residual.shape[0] != height * width or residual.ndim != 2:
            raise ValueError("residual_vectors shape mismatch")
        norm = np.linalg.norm(residual, axis=1)
        direction = np.zeros_like(residual)
        valid = norm >= eps
        direction[valid] = residual[valid] / norm[valid, None]
        direction = direction.reshape(height, width, -1)
        valid = valid.reshape(height, width)
        total = np.zeros((height, width), np.float64)
        count = np.zeros((height, width), np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, -dy), min(height, height - dy)
                x0, x1 = max(0, -dx), min(width, width - dx)
                neighbor_y0, neighbor_y1 = y0 + dy, y1 + dy
                neighbor_x0, neighbor_x1 = x0 + dx, x1 + dx
                cosine = np.sum(
                    direction[y0:y1, x0:x1] * direction[neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1],
                    axis=2,
                )
                both = valid[y0:y1, x0:x1] & valid[neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1]
                total[y0:y1, x0:x1] += np.maximum(cosine, 0.0) * both
                count[y0:y1, x0:x1] += 1.0
        output = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        output[~valid] = 0.0
        return np.clip(output, 0.0, 1.0)

    def apply(
        self,
        minmax_score: torch.Tensor,
        raw_residual: torch.Tensor,
        residual_vectors: torch.Tensor,
        background_indices: torch.Tensor,
    ) -> ThresholdV4Result:
        score, shape = _score(minmax_score)
        raw = raw_residual.detach().cpu().double().reshape(-1).numpy()
        if residual_vectors.shape[0] != score.size:
            raise ValueError("all 1369 residual vectors must be queried")
        bc = _indices(background_indices, score.size)
        coherence = self.coherence(
            residual_vectors.detach().cpu().double().numpy(), shape[-2], shape[-1], self.epsilon
        ).reshape(-1)
        residual_rank = empirical_rank(raw)
        coherence_rank = empirical_rank(coherence)
        orc_score = np.sqrt(residual_rank * coherence_rank)
        seed, u1, u2 = _multiotsu_seed(orc_score)
        support, _, huto = _huto_median_support(minmax_score, background_indices)
        final_grid, connectivity = SeededHysteresis.apply(
            seed.reshape(shape[-2:]), support.reshape(shape[-2:])
        )
        final = final_grid.reshape(-1)
        diagnostics = {
            "orc_t1": u1, "orc_t2": u2,
            "orc_seed_area": float(seed.mean()),
            "support_threshold": float(huto["threshold_low"]),
            "support_area": float(support.mean()),
            "final_area": float(final.mean()),
            "bc_seed_ratio": float(seed[bc].mean()),
            "bc_support_ratio": float(support[bc].mean()),
            "bc_final_ratio": float(final[bc].mean()),
            **connectivity,
            "failure_reason": None,
        }
        return _result(
            final, shape,
            {
                "residual_norm_map": np.sqrt(np.maximum(raw, 0.0)),
                "residual_coherence_map": coherence,
                "residual_rank_map": residual_rank,
                "coherence_rank_map": coherence_rank,
                "orc_score_map": orc_score,
                "mask_seed": seed,
                "mask_support": support,
            },
            u2, float(huto["threshold_low"]), False, diagnostics,
        )


class LocalResidualContrast:
    """Rank-fused local residual-contrast seeds with original-residual support."""

    def apply(self, minmax_score: torch.Tensor, background_indices: torch.Tensor) -> ThresholdV4Result:
        score, shape = _score(minmax_score)
        bc = _indices(background_indices, score.size)
        grid = score.reshape(shape[-2:])
        median3 = median_filter(grid, size=3, mode="reflect")
        median7 = median_filter(grid, size=7, mode="reflect")
        contrast = np.maximum(np.maximum(grid - median3, 0.0), np.maximum(grid - median7, 0.0))
        score_rank = empirical_rank(score)
        contrast_rank = empirical_rank(contrast.reshape(-1))
        lrc_score = np.sqrt(score_rank * contrast_rank)
        seed, v1, v2 = _multiotsu_seed(lrc_score)
        support, _, huto = _huto_median_support(minmax_score, background_indices)
        final_grid, connectivity = SeededHysteresis.apply(
            seed.reshape(shape[-2:]), support.reshape(shape[-2:])
        )
        final = final_grid.reshape(-1)
        diagnostics = {
            "lrc_t1": v1, "lrc_t2": v2,
            "lrc_seed_area": float(seed.mean()),
            "support_threshold": float(huto["threshold_low"]),
            "support_area": float(support.mean()),
            "final_area": float(final.mean()),
            "bc_seed_ratio": float(seed[bc].mean()),
            "bc_support_ratio": float(support[bc].mean()),
            "bc_final_ratio": float(final[bc].mean()),
            **connectivity,
            "failure_reason": None,
        }
        return _result(
            final, shape,
            {
                "median3_map": median3,
                "median7_map": median7,
                "local_contrast_map": contrast,
                "residual_rank_map": score_rank,
                "contrast_rank_map": contrast_rank,
                "lrc_score_map": lrc_score,
                "mask_seed": seed,
                "mask_support": support,
            },
            v2, float(huto["threshold_low"]), False, diagnostics,
        )
