"""GT-free GBSP V5 seeded structural recovery methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from skimage.filters import threshold_multiotsu, threshold_otsu


@dataclass(frozen=True)
class ThresholdV5Result:
    mask: torch.Tensor
    maps: dict[str, torch.Tensor]
    threshold_high: float | None
    threshold_low: float | None
    numerical_failure: bool
    diagnostics: dict[str, Any]


def _score(value: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int]]:
    if not torch.is_tensor(value) or value.numel() == 0 or value.ndim < 2:
        raise ValueError("score must be a non-empty tensor map")
    shape = tuple(value.shape)
    spatial = tuple(value.shape[-2:])
    array = value.detach().cpu().double().numpy().reshape(-1).copy()
    if array.size != spatial[0] * spatial[1]:
        raise ValueError("score must contain exactly one spatial map")
    if not np.isfinite(array).all() or array.min() < -1e-7 or array.max() > 1.0 + 1e-7:
        raise ValueError("score must be finite and inside [0,1]")
    return np.clip(array, 0.0, 1.0), shape, spatial


def _indices(value: torch.Tensor, size: int, name: str) -> np.ndarray:
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    array = value.detach().cpu().long().numpy().copy()
    if array.size and (array.min() < 0 or array.max() >= size or np.unique(array).size != array.size):
        raise ValueError(f"{name} contains invalid or duplicate indices")
    return array


def _tensor(value: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(shape).copy())


def _result(mask, shape, maps, high, low, failure, diagnostics) -> ThresholdV5Result:
    return ThresholdV5Result(
        mask=_tensor(np.asarray(mask, dtype=np.float32), shape),
        maps={key: _tensor(value, shape) for key, value in maps.items()},
        threshold_high=None if high is None else float(high),
        threshold_low=None if low is None else float(low),
        numerical_failure=bool(failure), diagnostics=diagnostics,
    )


def _seeded_region(active: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, int, int]:
    labels, count = connected_components(active, structure=np.ones((3, 3), np.uint8))
    kept = np.unique(labels[seed])
    kept = kept[kept != 0]
    return np.isin(labels, kept), int(count), int(kept.size)


class HierarchicalThresholdSeeds:
    """Frozen Multi-Otsu-3 plus upper-tail Otsu seed/support construction."""

    def __init__(self, min_tail_count: int = 16, min_tail_unique: int = 8) -> None:
        self.min_tail_count = int(min_tail_count)
        self.min_tail_unique = int(min_tail_unique)

    def apply(self, score_tensor: torch.Tensor, background_indices: torch.Tensor) -> ThresholdV5Result:
        score, shape, spatial = _score(score_tensor)
        bc = _indices(background_indices, score.size, "background_indices")
        if np.unique(score).size < 3:
            return _result(np.zeros(score.size, bool), shape, {}, None, None, True,
                           {"failure_reason": "first_level_unique_values_below_three"})
        t1, t2 = map(float, threshold_multiotsu(score, classes=3, nbins=256))
        upper = score[score > t2]
        if upper.size < self.min_tail_count or np.unique(upper).size < self.min_tail_unique:
            return _result(np.zeros(score.size, bool), shape, {}, None, t2, True, {
                "t1": t1, "t2": t2, "t3": None, "invalid_upper_tail": True,
                "upper_tail_patch_count": int(upper.size),
                "upper_tail_unique_values": int(np.unique(upper).size),
                "failure_reason": "invalid_upper_tail",
            })
        t3 = float(threshold_otsu(upper.astype(np.float64), nbins=256))
        if not t3 > t2:
            return _result(np.zeros(score.size, bool), shape, {}, t3, t2, True, {
                "t1": t1, "t2": t2, "t3": t3, "invalid_upper_tail": True,
                "failure_reason": "second_threshold_not_above_t2",
            })
        seed, support = score > t3, score > t2
        state = np.zeros(score.size, np.float32)
        state[score > t1] = 1.0; state[score > t2] = 2.0
        seed_labels, seed_count = connected_components(seed.reshape(spatial), structure=np.ones((3, 3), np.uint8))
        support_labels, support_count = connected_components(support.reshape(spatial), structure=np.ones((3, 3), np.uint8))
        del seed_labels, support_labels
        diagnostics = {
            "t1": t1, "t2": t2, "t3": t3, "invalid_upper_tail": False,
            "upper_tail_patch_count": int(upper.size),
            "upper_tail_unique_values": int(np.unique(upper).size),
            "seed_area": float(seed.mean()), "support_area": float(support.mean()),
            "num_seed_components": int(seed_count), "num_support_components": int(support_count),
            "bc_seed_ratio": float(seed[bc].mean()) if bc.size else 0.0,
            "bc_support_ratio": float(support[bc].mean()) if bc.size else 0.0,
            "failure_reason": None,
        }
        return _result(seed, shape, {"threshold_state": state, "seed_mask": seed, "support_mask": support},
                       t3, t2, False, diagnostics)


class SeededMultiOtsuHysteresis:
    """Retain exactly the t2 support components intersecting a t3 seed."""

    def __init__(self) -> None:
        self.thresholds = HierarchicalThresholdSeeds()

    def apply(self, score_tensor: torch.Tensor, background_indices: torch.Tensor) -> ThresholdV5Result:
        common = self.thresholds.apply(score_tensor, background_indices)
        if common.numerical_failure:
            return common
        score, shape, spatial = _score(score_tensor)
        bc = _indices(background_indices, score.size, "background_indices")
        seed = common.maps["seed_mask"].numpy().reshape(spatial).astype(bool)
        support = common.maps["support_mask"].numpy().reshape(spatial).astype(bool)
        final, support_count, retained_count = _seeded_region(support, seed)
        flat = final.reshape(-1)
        diagnostics = {
            **common.diagnostics,
            "num_support_components": support_count,
            "num_retained_components": retained_count,
            "retained_support_ratio": float(flat.sum() / max(int(support.sum()), 1)),
            "final_area": float(flat.mean()),
            "bc_final_ratio": float(flat[bc].mean()) if bc.size else 0.0,
        }
        return _result(flat, shape, {
            "threshold_state": common.maps["threshold_state"].numpy(),
            "seed_mask": seed, "support_mask": support, "final_mask": final,
        }, common.threshold_high, common.threshold_low, False, diagnostics)


class SeededPersistentComponentGrowth:
    """Per-support-component threshold descent stopped by an online robust growth jump."""

    def __init__(self, min_history: int = 5, mad_multiplier: float = 3.0, min_events: int = 6) -> None:
        self.min_history = int(min_history)
        self.mad_multiplier = float(mad_multiplier)
        self.min_events = int(min_events)
        self.thresholds = HierarchicalThresholdSeeds()

    def detect_jump(self, areas: list[int] | np.ndarray) -> dict[str, Any]:
        values = np.asarray(areas, dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) < 0):
            raise ValueError("areas must be a monotone one-dimensional sequence")
        growth = np.diff(values) / (values[:-1] + 1.0)
        result = {"jump_index": None, "jump_value": None, "jump_baseline": None,
                  "jump_mad_scale": None, "growth_values": growth.tolist()}
        # min_history counts the initial area state plus prior growth events.  Thus
        # the sixth area event (the fifth growth) has four historical growth rates.
        for index, value in enumerate(growth):
            history = growth[:index]
            if index + 2 < self.min_events or history.size < self.min_history - 1:
                continue
            center = float(np.median(history))
            scale = float(1.4826 * np.median(np.abs(history - center)))
            if value > center + self.mad_multiplier * scale and value > center + 1e-6:
                result.update({"jump_index": int(index + 1), "jump_value": float(value),
                               "jump_baseline": center, "jump_mad_scale": scale})
                break
        return result

    def apply(self, score_tensor: torch.Tensor, background_indices: torch.Tensor) -> ThresholdV5Result:
        common = self.thresholds.apply(score_tensor, background_indices)
        if common.numerical_failure:
            return common
        score, shape, spatial = _score(score_tensor)
        bc = _indices(background_indices, score.size, "background_indices")
        score_grid = score.reshape(spatial)
        seed = common.maps["seed_mask"].numpy().reshape(spatial).astype(bool)
        support = common.maps["support_mask"].numpy().reshape(spatial).astype(bool)
        labels, support_count = connected_components(support, structure=np.ones((3, 3), np.uint8))
        final = np.zeros(spatial, bool)
        component_rows: list[dict[str, Any]] = []
        insufficient = 0; seeded_count = 0
        for component_id in range(1, support_count + 1):
            component = labels == component_id
            component_seed = component & seed
            if not component_seed.any():
                continue
            seeded_count += 1
            interior_values = np.unique(score_grid[component & (score_grid > common.threshold_low) &
                                                   (score_grid < common.threshold_high)])
            thresholds = [float(common.threshold_high), *sorted(map(float, interior_values), reverse=True),
                          float(common.threshold_low)]
            regions=[]; areas=[]
            for threshold in thresholds:
                active = component & (score_grid >= threshold)
                region, _, _ = _seeded_region(active, component_seed)
                regions.append(region); areas.append(int(region.sum()))
            event_count = len(thresholds)
            jump = self.detect_jump(areas) if event_count >= self.min_events else {
                "jump_index": None, "jump_value": None, "jump_baseline": None,
                "jump_mad_scale": None, "growth_values": [],
            }
            if event_count < self.min_events:
                insufficient += 1; chosen = component; stop_threshold = float(common.threshold_low)
            elif jump["jump_index"] is None:
                chosen = component; stop_threshold = float(common.threshold_low)
            else:
                stop_index = int(jump["jump_index"]) - 1
                chosen = regions[stop_index]; stop_threshold = thresholds[stop_index]
            final |= chosen
            component_rows.append({
                "component_id": int(component_id), "component_seed_area": int(component_seed.sum()),
                "component_support_area": int(component.sum()), "component_event_count": int(event_count),
                "component_stop_threshold": float(stop_threshold), "component_final_area": int(chosen.sum()),
                "component_thresholds": thresholds, "component_growth_curve": areas,
                "component_growth_values": jump["growth_values"],
                "component_jump_index": jump["jump_index"], "component_jump_value": jump["jump_value"],
                "component_jump_baseline": jump["jump_baseline"],
                "component_jump_mad_scale": jump["jump_mad_scale"],
                "insufficient_growth_events": bool(event_count < self.min_events),
            })
        flat = final.reshape(-1)
        diagnostics = {
            **common.diagnostics, "num_support_components": int(support_count),
            "num_seeded_support_components": int(seeded_count),
            "component_diagnostics": component_rows,
            "insufficient_growth_events_count": int(insufficient),
            "jump_detected_component_count": int(sum(row["component_jump_index"] is not None for row in component_rows)),
            "final_area": float(flat.mean()),
            "bc_final_ratio": float(flat[bc].mean()) if bc.size else 0.0,
        }
        return _result(flat, shape, {
            "threshold_state": common.maps["threshold_state"].numpy(),
            "seed_mask": seed, "support_mask": support, "final_mask": final,
        }, common.threshold_high, common.threshold_low, False, diagnostics)


class BackgroundConditionedMarkerPropagation:
    """Harmonic foreground probability on the frozen semantic/color/edge patch graph."""

    def __init__(self, probability_threshold: float = 0.5, epsilon: float = 1e-8) -> None:
        self.probability_threshold = float(probability_threshold)
        self.epsilon = float(epsilon)
        self.thresholds = HierarchicalThresholdSeeds()

    @staticmethod
    def _adjacency(indices: np.ndarray, valid: np.ndarray, values: np.ndarray, is_cost: bool,
                   epsilon: float) -> tuple[Any, np.ndarray]:
        count, degree = indices.shape
        if valid.shape != indices.shape or values.shape != indices.shape:
            raise ValueError("graph tensors must have matching [N,K] shapes")
        src = np.broadcast_to(np.arange(count)[:, None], (count, degree))[valid]
        dst = indices[valid]; edge_values = values[valid].astype(np.float64)
        if edge_values.size == 0 or not np.isfinite(edge_values).all() or np.any(edge_values < 0):
            raise ValueError("graph has no valid finite nonnegative edges")
        if is_cost:
            positive = edge_values[edge_values > 0]
            scale = float(np.median(positive)) if positive.size else epsilon
            weights = np.exp(-edge_values / (scale + epsilon))
        else:
            weights = edge_values
        adjacency = coo_matrix((weights, (src, dst)), shape=(count, count)).tocsr()
        adjacency = (adjacency + adjacency.T) * 0.5
        adjacency.setdiag(0); adjacency.eliminate_zeros()
        return adjacency, adjacency.data

    @staticmethod
    def harmonic_probability(adjacency, foreground: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, str, float]:
        foreground = np.asarray(foreground, dtype=bool).reshape(-1)
        background = np.asarray(background, dtype=bool).reshape(-1)
        if foreground.size != adjacency.shape[0] or background.size != foreground.size:
            raise ValueError("marker size does not match graph")
        if np.any(foreground & background) or not foreground.any() or not background.any():
            raise ValueError("foreground/background markers must be nonempty and disjoint")
        marked = foreground | background; unknown = ~marked
        probability = np.zeros(foreground.size, np.float64); probability[foreground] = 1.0
        status, residual = "no_unknown_nodes", 0.0
        if unknown.any():
            degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
            laplacian = diags(degree) - adjacency
            unknown_idx, marked_idx = np.flatnonzero(unknown), np.flatnonzero(marked)
            lhs = laplacian[unknown_idx][:, unknown_idx].tocsc()
            rhs = -(laplacian[unknown_idx][:, marked_idx] @ probability[marked_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solved = np.asarray(spsolve(lhs, rhs), dtype=np.float64).reshape(-1)
            if not np.isfinite(solved).all():
                raise FloatingPointError("harmonic solver returned NaN/Inf")
            residual = float(np.max(np.abs(lhs @ solved - rhs))) if solved.size else 0.0
            probability[unknown_idx] = solved; status = "solved"
        return probability, status, residual

    def apply(self, score_tensor: torch.Tensor, background_indices: torch.Tensor,
              boundary_indices: torch.Tensor, neighbor_indices: torch.Tensor,
              neighbor_valid: torch.Tensor, graph_values: torch.Tensor,
              graph_is_cost: bool = False) -> ThresholdV5Result:
        common = self.thresholds.apply(score_tensor, background_indices)
        if common.numerical_failure:
            return common
        score, shape, spatial = _score(score_tensor); count = score.size
        bc = _indices(background_indices, count, "background_indices")
        boundary = _indices(boundary_indices, count, "boundary_indices")
        indices = neighbor_indices.detach().cpu().long().numpy()
        valid = neighbor_valid.detach().cpu().bool().numpy()
        values = graph_values.detach().cpu().double().numpy()
        if indices.shape[0] != count or indices.ndim != 2 or np.any(indices[valid] < 0) or np.any(indices[valid] >= count):
            raise ValueError("neighbor_indices are invalid")
        foreground = common.maps["seed_mask"].numpy().reshape(-1).astype(bool)
        low = score <= float(common.diagnostics["t1"])
        bc_low = np.zeros(count, bool); bc_low[bc] = low[bc]
        low_labels, _ = connected_components(low.reshape(spatial), structure=np.ones((3, 3), np.uint8))
        boundary_labels = np.unique(low_labels.reshape(-1)[boundary]); boundary_labels = boundary_labels[boundary_labels != 0]
        boundary_low = np.isin(low_labels, boundary_labels).reshape(-1)
        background = bc_low | boundary_low
        if not foreground.any() or not background.any():
            return _result(np.zeros(count, bool), shape, {
                "seed_mask_fg": foreground, "seed_mask_bg": background,
            }, common.threshold_high, common.threshold_low, True, {
                **common.diagnostics, "foreground_seed_count": int(foreground.sum()),
                "background_seed_count": int(background.sum()),
                "failure_reason": "missing_foreground_or_background_markers",
            })
        adjacency, edge_weights = self._adjacency(indices, valid, values, graph_is_cost, self.epsilon)
        probability, solver_status, solver_residual = self.harmonic_probability(
            adjacency, foreground, background
        )
        if probability.min() < -1e-6 or probability.max() > 1.0 + 1e-6:
            raise FloatingPointError("harmonic probabilities outside numerical tolerance")
        probability = np.clip(probability, 0.0, 1.0)
        final = probability > self.probability_threshold
        diagnostics = {
            **common.diagnostics, "foreground_seed_count": int(foreground.sum()),
            "background_seed_count": int(background.sum()),
            "bc_background_seed_count": int(bc_low.sum()),
            "boundary_background_seed_count": int(boundary_low.sum()),
            "graph_edge_count": int(adjacency.nnz // 2),
            "graph_weight_min": float(edge_weights.min()), "graph_weight_max": float(edge_weights.max()),
            "graph_weight_median": float(np.median(edge_weights)), "graph_values_are_costs": bool(graph_is_cost),
            "solver_status": solver_status, "solver_residual": solver_residual,
            "probability_min": float(probability.min()), "probability_max": float(probability.max()),
            "probability_mean": float(probability.mean()), "equivalent_area": float(final.mean()),
            "final_area": float(final.mean()),
            "bc_selected_as_foreground_ratio": float(final[bc].mean()) if bc.size else 0.0,
            "bc_final_ratio": float(final[bc].mean()) if bc.size else 0.0,
            "failure_reason": None,
        }
        return _result(final, shape, {
            "threshold_state": common.maps["threshold_state"].numpy(),
            "seed_mask_fg": foreground, "seed_mask_bg": background,
            "probability_map": probability, "final_mask": final,
        }, common.threshold_high, common.threshold_low, False, diagnostics)
