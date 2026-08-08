"""GT-free precision-first tri-state calibration for cached GBSP residuals.

The implementations in this module consume only one image's 37x37 GBSP
Min-Max residual and Full-BC indices.  They never read GT, R1 predictions,
the fixed-0.58 reference, or a target foreground-area statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from skimage.filters import threshold_multiotsu


@dataclass(frozen=True)
class ThresholdV3Result:
    mask_core: torch.Tensor
    continuous_maps: dict[str, torch.Tensor]
    threshold_high: float | None
    threshold_low: float | None
    numerical_failure: bool
    diagnostics: dict[str, Any]


def _vector(value: torch.Tensor, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty tensor")
    array = value.detach().cpu().double().numpy().copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array.reshape(-1), tuple(array.shape)


def _indices(value: torch.Tensor, size: int) -> np.ndarray:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0:
        raise ValueError("background_indices must be a non-empty vector")
    indices = value.detach().cpu().long().numpy().copy()
    if indices.min() < 0 or indices.max() >= size or np.unique(indices).size != indices.size:
        raise ValueError("background_indices are invalid or duplicated")
    return indices


def _tensor_map(value: np.ndarray, shape: tuple[int, ...], dtype=np.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=dtype).reshape(shape).copy())


def _result(
    mask: np.ndarray,
    shape: tuple[int, ...],
    continuous: dict[str, np.ndarray],
    threshold_high: float | None,
    threshold_low: float | None,
    numerical_failure: bool,
    diagnostics: dict[str, Any],
) -> ThresholdV3Result:
    return ThresholdV3Result(
        mask_core=_tensor_map(np.asarray(mask, dtype=np.float32), shape),
        continuous_maps={key: _tensor_map(value, shape) for key, value in continuous.items()},
        threshold_high=None if threshold_high is None else float(threshold_high),
        threshold_low=None if threshold_low is None else float(threshold_low),
        numerical_failure=bool(numerical_failure),
        diagnostics=diagnostics,
    )


class MultiOtsuThreeState:
    """Three-level Multi-Otsu; only the highest residual state is foreground."""

    def __init__(self, classes: int = 3, nbins: int = 256) -> None:
        if int(classes) != 3 or int(nbins) != 256:
            raise ValueError("V3 Multi-Otsu is frozen to classes=3, nbins=256")
        self.classes = int(classes)
        self.nbins = int(nbins)

    def apply(
        self, minmax_score: torch.Tensor, background_indices: torch.Tensor
    ) -> ThresholdV3Result:
        score, shape = _vector(minmax_score, "minmax_score")
        bc = _indices(background_indices, score.size)
        invalid = np.unique(score).size < self.classes
        if invalid:
            diagnostics = {
                "t_low": None,
                "t_high": None,
                "class0_area": 1.0,
                "class1_area": 0.0,
                "class2_area": 0.0,
                "foreground_area": 0.0,
                "bc_in_class0_ratio": 1.0,
                "bc_in_class1_ratio": 0.0,
                "bc_in_class2_ratio": 0.0,
                "multiotsu_invalid_unique_values": True,
                "invalid_unique_values": True,
                "failure_reason": "fewer_than_three_unique_scores",
            }
            return _result(
                np.zeros_like(score, dtype=bool), shape, {}, None, None, True, diagnostics
            )
        thresholds = threshold_multiotsu(
            image=score.astype(np.float64, copy=False), classes=3, nbins=256
        )
        t_low, t_high = map(float, thresholds)
        state = np.digitize(score, thresholds, right=True)
        mask = state == 2
        diagnostics = {
            "t_low": t_low,
            "t_high": t_high,
            "class0_area": float(np.mean(state == 0)),
            "class1_area": float(np.mean(state == 1)),
            "class2_area": float(np.mean(state == 2)),
            "foreground_area": float(mask.mean()),
            "bc_in_class0_ratio": float(np.mean(state[bc] == 0)),
            "bc_in_class1_ratio": float(np.mean(state[bc] == 1)),
            "bc_in_class2_ratio": float(np.mean(state[bc] == 2)),
            "multiotsu_invalid_unique_values": False,
            "invalid_unique_values": False,
            "failure_reason": None,
        }
        return _result(mask, shape, {"state": state}, t_high, t_low, False, diagnostics)


def _inv_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return float(value + np.log(-np.expm1(-value)))


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-8, 1.0 - 1e-8))
    return float(np.log(value) - np.log1p(-value))


def _normal_logpdf_torch(
    values: torch.Tensor, means: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    return (
        -0.5 * ((values[:, None] - means[None, :]) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
    )


def _normal_logpdf_numpy(values: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    return (
        -0.5 * ((values[:, None] - means[None, :]) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
    )


def _shared_gaussian_em(y: np.ndarray, components: int) -> tuple[float, dict[str, Any]]:
    """Deterministic unanchored shared-variance mixture fit used only for BIC."""
    if components == 1:
        mean = float(y.mean())
        sigma = max(0.05, float(np.sqrt(np.mean((y - mean) ** 2))))
        ll = float(_normal_logpdf_numpy(y, np.asarray([mean]), sigma).sum())
        return ll, {"means": [mean], "weights": [1.0], "sigma": sigma}
    quantile_sets = {
        2: ((0.25, 0.75), (0.10, 0.90)),
        3: ((0.20, 0.55, 0.85), (0.10, 0.50, 0.90)),
    }[components]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for quantiles in quantile_sets:
        means = np.asarray(np.quantile(y, quantiles), dtype=np.float64)
        weights = np.full(components, 1.0 / components, dtype=np.float64)
        sigma = max(0.05, float(np.std(y)))
        previous = -np.inf
        for iteration in range(500):
            log_joint = np.log(np.clip(weights, 1e-12, None))[None, :] + _normal_logpdf_numpy(
                y, means, sigma
            )
            log_norm = logsumexp(log_joint, axis=1)
            responsibilities = np.exp(log_joint - log_norm[:, None])
            counts = np.maximum(responsibilities.sum(axis=0), 1e-8)
            weights = counts / counts.sum()
            means = (responsibilities * y[:, None]).sum(axis=0) / counts
            order = np.argsort(means, kind="stable")
            means, weights = means[order], weights[order]
            variance = float(
                (responsibilities[:, order] * (y[:, None] - means[None, :]) ** 2).sum()
                / y.size
            )
            sigma = float(np.clip(np.sqrt(max(variance, 1e-12)), 0.05, 10.0))
            log_joint = np.log(np.clip(weights, 1e-12, None))[None, :] + _normal_logpdf_numpy(
                y, means, sigma
            )
            likelihood = float(logsumexp(log_joint, axis=1).sum())
            if np.isfinite(previous) and abs(likelihood - previous) <= 1e-10 * (1.0 + abs(previous)):
                break
            previous = likelihood
        candidates.append(
            (
                likelihood,
                {
                    "means": means.tolist(),
                    "weights": weights.tolist(),
                    "sigma": sigma,
                    "iterations": iteration + 1,
                },
            )
        )
    return max(candidates, key=lambda item: item[0])


class OrderedTriGaussianCore:
    """Ordered, shared-variance, BC-superclass-anchored tri-Gaussian model."""

    def __init__(
        self,
        anchor_strength: float = 1.0,
        min_mean_gap: float = 0.05,
        sigma_floor: float = 0.05,
        sigma_cap: float = 10.0,
        foreground_mixture_floor: float = 0.001,
        foreground_mixture_cap: float = 0.5,
        score_clip: float = 1e-4,
        max_iter: int = 300,
        history_size: int = 50,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
    ) -> None:
        if anchor_strength < 0:
            raise ValueError("anchor_strength must be nonnegative")
        if min_mean_gap != 0.05 or sigma_floor != 0.05 or sigma_cap != 10.0:
            raise ValueError("OTGC V3 mean-gap and sigma limits are frozen")
        if foreground_mixture_floor != 0.001 or foreground_mixture_cap != 0.5:
            raise ValueError("OTGC V3 foreground-mixture limits are frozen")
        self.anchor_strength = float(anchor_strength)
        self.min_mean_gap = float(min_mean_gap)
        self.sigma_floor = float(sigma_floor)
        self.sigma_cap = float(sigma_cap)
        self.pi2_floor = float(foreground_mixture_floor)
        self.pi2_span = float(foreground_mixture_cap - foreground_mixture_floor)
        self.score_clip = float(score_clip)
        self.max_iter = int(max_iter)
        self.history_size = int(history_size)
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)

    def _decode(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0 = theta[0]
        mu1 = mu0 + self.min_mean_gap + torch.nn.functional.softplus(theta[1])
        mu2 = mu1 + self.min_mean_gap + torch.nn.functional.softplus(theta[2])
        sigma = self.sigma_floor + torch.nn.functional.softplus(theta[3])
        pi2 = self.pi2_floor + self.pi2_span * torch.sigmoid(theta[4])
        bg_split = torch.sigmoid(theta[5])
        pi0 = (1.0 - pi2) * bg_split
        pi1 = (1.0 - pi2) * (1.0 - bg_split)
        return torch.stack((mu0, mu1, mu2)), sigma, torch.stack((pi0, pi1, pi2))

    def _initializations(self, y: np.ndarray, bc: np.ndarray) -> list[tuple[str, np.ndarray]]:
        definitions = (
            ("A_q20_q55_q85", np.quantile(y, (0.20, 0.55, 0.85))),
            ("B_q10_q50_q90", np.quantile(y, (0.10, 0.50, 0.90))),
            (
                "C_bc25_bc75_q90",
                np.asarray((np.quantile(y[bc], 0.25), np.quantile(y[bc], 0.75), np.quantile(y, 0.90))),
            ),
        )
        sigma = max(0.20, float(np.subtract(*np.percentile(y, [75, 25])) / 1.349))
        output = []
        for name, raw_means in definitions:
            means = np.asarray(raw_means, dtype=np.float64).copy()
            for index in (1, 2):
                means[index] = max(means[index], means[index - 1] + self.min_mean_gap + 1e-4)
            gaps = np.diff(means) - self.min_mean_gap
            theta = np.asarray(
                (
                    means[0],
                    _inv_softplus(gaps[0]),
                    _inv_softplus(gaps[1]),
                    _inv_softplus(max(sigma - self.sigma_floor, 1e-5)),
                    _logit((0.15 - self.pi2_floor) / self.pi2_span),
                    _logit(0.60 / (0.60 + 0.25)),
                ),
                dtype=np.float64,
            )
            output.append((name, theta))
        return output

    def _fit_one(
        self, y: torch.Tensor, bc: torch.Tensor, name: str, initial: np.ndarray
    ) -> dict[str, Any]:
        theta = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=self.max_iter,
            history_size=self.history_size,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            line_search_fn="strong_wolfe",
        )
        history: list[float] = []

        def terms() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            means, sigma, weights = self._decode(theta)
            log_joint = torch.log(weights)[None, :] + _normal_logpdf_torch(y, means, sigma)
            log_norm = torch.logsumexp(log_joint, dim=1)
            log_p_bg = torch.logsumexp(log_joint.index_select(0, bc)[:, :2], dim=1) - log_norm.index_select(0, bc)
            mixture_mean = log_norm.mean()
            anchor_mean = log_p_bg.mean()
            objective = mixture_mean + self.anchor_strength * anchor_mean
            return objective, mixture_mean, anchor_mean, log_joint, log_norm

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            objective, _, _, _, _ = terms()
            loss = -objective
            loss.backward()
            history.append(float(objective.detach()))
            return loss

        error = None
        try:
            optimizer.step(closure)
        except (RuntimeError, ValueError, FloatingPointError) as exception:
            error = repr(exception)
        optimizer.zero_grad(set_to_none=True)
        objective, mixture_mean, anchor_mean, log_joint, log_norm = terms()
        (-objective).backward()
        grad_norm = float(theta.grad.detach().abs().max()) if theta.grad is not None else float("inf")
        means_t, sigma_t, weights_t = self._decode(theta)
        posterior = torch.exp(log_joint - log_norm[:, None])
        state = optimizer.state.get(theta, {})
        iterations = int(state.get("n_iter", 0))
        finite = bool(
            torch.isfinite(theta).all()
            and torch.isfinite(means_t).all()
            and torch.isfinite(sigma_t)
            and torch.isfinite(weights_t).all()
            and torch.isfinite(posterior).all()
            and torch.isfinite(objective)
        )
        converged = bool(
            error is None
            and finite
            and iterations > 0
            and (iterations < self.max_iter or grad_norm <= max(self.tolerance_grad * 10.0, 1e-6))
        )
        return {
            "initialization": name,
            "theta": theta.detach().numpy().copy(),
            "means": means_t.detach().numpy().copy(),
            "sigma": float(sigma_t.detach()),
            "weights": weights_t.detach().numpy().copy(),
            "posterior": posterior.detach().numpy().copy(),
            "objective": float(objective.detach()),
            "mixture_mean": float(mixture_mean.detach()),
            "mixture_log_likelihood": float(mixture_mean.detach()) * y.numel(),
            "anchor_loss": float(anchor_mean.detach()),
            "iterations": iterations,
            "function_evaluations": int(state.get("func_evals", len(history))),
            "gradient_max_abs": grad_norm,
            "optimizer_converged": converged,
            "finite": finite,
            "error": error,
            "objective_history": history,
        }

    @staticmethod
    def _posterior_at(value: float, means: np.ndarray, sigma: float, weights: np.ndarray) -> float:
        log_joint = np.log(weights) + _normal_logpdf_numpy(
            np.asarray([value], dtype=np.float64), means, sigma
        )[0]
        return float(np.exp(log_joint[2] - logsumexp(log_joint)))

    def _audit_candidate(self, candidate: dict[str, Any], score: np.ndarray) -> dict[str, Any]:
        means = candidate["means"]
        sigma = candidate["sigma"]
        weights = candidate["weights"]
        posterior = candidate["posterior"]
        order = np.argsort(score, kind="stable")
        violations = int(np.sum(np.diff(posterior[order, 2]) < -1e-8))
        probability_error = float(np.max(np.abs(posterior.sum(axis=1) - 1.0)))
        low, high = float(means[1]), float(means[2] + 8.0 * sigma)
        f_low = self._posterior_at(low, means, sigma, weights) - 0.5
        f_high = self._posterior_at(high, means, sigma, weights) - 0.5
        threshold_y = None
        if np.isfinite((f_low, f_high)).all() and f_low <= 0.0 <= f_high:
            threshold_y = low if f_low == 0.0 else float(
                brentq(
                    lambda point: self._posterior_at(point, means, sigma, weights) - 0.5,
                    low,
                    high,
                    xtol=1e-12,
                    rtol=1e-12,
                )
            )
        threshold = None if threshold_y is None else float(1.0 / (1.0 + np.exp(-threshold_y)))
        gaps = np.diff(means) / sigma
        collapse = bool(np.min(weights) < 1e-6 or np.min(np.diff(means)) <= 0.0)
        legal = bool(
            candidate["optimizer_converged"]
            and candidate["finite"]
            and means[0] < means[1] < means[2]
            and self.sigma_floor <= sigma <= self.sigma_cap
            and self.pi2_floor <= weights[2] <= self.pi2_floor + self.pi2_span
            and probability_error <= 1e-8
            and violations == 0
            and threshold is not None
            and not collapse
        )
        return {
            **candidate,
            "posterior_monotonic_violation_count": violations,
            "monotonicity_passed": violations == 0,
            "posterior_sum_max_abs_error": probability_error,
            "threshold_y": threshold_y,
            "equivalent_threshold_high": threshold,
            "otgc_no_high_intersection": threshold is None,
            "gap01_std": float(gaps[0]),
            "gap12_std": float(gaps[1]),
            "weak_high_component": bool(gaps[1] < 0.25),
            "component_collapse": collapse,
            "legal": legal,
        }

    def apply(
        self, minmax_score: torch.Tensor, background_indices: torch.Tensor
    ) -> ThresholdV3Result:
        original, shape = _vector(minmax_score, "minmax_score")
        score = np.clip(original, self.score_clip, 1.0 - self.score_clip)
        y_np = np.log(score) - np.log1p(-score)
        bc_np = _indices(background_indices, score.size)
        y = torch.from_numpy(y_np.copy()).double()
        bc = torch.from_numpy(bc_np.copy()).long()
        candidates = [
            self._audit_candidate(self._fit_one(y, bc, name, initial), original)
            for name, initial in self._initializations(y_np, bc_np)
        ]
        legal = [candidate for candidate in candidates if candidate["legal"]]
        selected = max(legal or candidates, key=lambda row: (row["objective"], row["initialization"]))

        ll1, fit1 = _shared_gaussian_em(y_np, 1)
        ll2, fit2 = _shared_gaussian_em(y_np, 2)
        ll3, fit3 = _shared_gaussian_em(y_np, 3)
        n = y_np.size
        bic1 = 2.0 * np.log(n) - 2.0 * ll1
        bic2 = 4.0 * np.log(n) - 2.0 * ll2
        bic3 = 6.0 * np.log(n) - 2.0 * ll3

        posterior = selected["posterior"]
        valid = bool(selected["legal"])
        mask = posterior[:, 2] > 0.5 if valid else np.zeros(n, dtype=bool)
        diagnostics = {
            "mu0": float(selected["means"][0]),
            "mu1": float(selected["means"][1]),
            "mu2": float(selected["means"][2]),
            "sigma_shared": float(selected["sigma"]),
            "pi0": float(selected["weights"][0]),
            "pi1": float(selected["weights"][1]),
            "pi2": float(selected["weights"][2]),
            "gap01_std": selected["gap01_std"],
            "gap12_std": selected["gap12_std"],
            "anchor_loss": selected["anchor_loss"],
            "mixture_log_likelihood": selected["mixture_log_likelihood"],
            "objective": selected["objective"],
            "selected_initialization": selected["initialization"],
            "optimizer_converged": selected["optimizer_converged"],
            "optimizer_iterations": selected["iterations"],
            "optimizer_function_evaluations": selected["function_evaluations"],
            "optimizer_gradient_max_abs": selected["gradient_max_abs"],
            "bic1": float(bic1),
            "bic2": float(bic2),
            "bic3": float(bic3),
            "delta_bic_3_vs_2": float(bic2 - bic3),
            "bic1_fit": fit1,
            "bic2_fit": fit2,
            "bic3_fit": fit3,
            "equivalent_threshold_high": selected["equivalent_threshold_high"],
            "foreground_area": float(mask.mean()),
            "bc_in_c2_ratio": float(np.mean(mask[bc_np])),
            "weak_high_component": selected["weak_high_component"],
            "otgc_weak_high_component": selected["weak_high_component"],
            "monotonicity_passed": selected["monotonicity_passed"],
            "posterior_monotonic_violation_count": selected["posterior_monotonic_violation_count"],
            "posterior_sum_max_abs_error": selected["posterior_sum_max_abs_error"],
            "otgc_no_high_intersection": selected["otgc_no_high_intersection"],
            "component_collapse": selected["component_collapse"],
            "legal_fit": valid,
            "all_initializations": [
                {
                    "initialization": item["initialization"],
                    "objective": item["objective"],
                    "optimizer_converged": item["optimizer_converged"],
                    "legal": item["legal"],
                    "error": item["error"],
                }
                for item in candidates
            ],
            "failure_reason": None if valid else "no_legal_initialization",
        }
        return _result(
            mask,
            shape,
            {
                "posterior_c0": posterior[:, 0],
                "posterior_c1": posterior[:, 1],
                "posterior_c2": posterior[:, 2],
            },
            selected["equivalent_threshold_high"],
            None,
            not valid,
            diagnostics,
        )


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))


def _segment_fit(
    px: np.ndarray,
    py: np.ndarray,
    pxx: np.ndarray,
    pyy: np.ndarray,
    pxy: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, float, float]:
    n = end - start
    if n < 2:
        return 0.0, 0.0, float("inf")
    sx, sy = px[end] - px[start], py[end] - py[start]
    sxx, syy = pxx[end] - pxx[start], pyy[end] - pyy[start]
    sxy = pxy[end] - pxy[start]
    denominator = n * sxx - sx * sx
    slope = 0.0 if abs(denominator) < 1e-15 else (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n
    sse = syy - 2.0 * slope * sxy - 2.0 * intercept * sy + slope * slope * sxx + 2.0 * slope * intercept * sx + n * intercept * intercept
    return float(slope), float(intercept), float(max(sse, 0.0))


def _segment_sse_array(
    px: np.ndarray,
    py: np.ndarray,
    pxx: np.ndarray,
    pyy: np.ndarray,
    pxy: np.ndarray,
    start,
    end,
) -> np.ndarray:
    """Vectorized linear-regression SSE for broadcastable [start, end) pairs."""
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    count = end - start
    sx, sy = px[end] - px[start], py[end] - py[start]
    sxx, syy = pxx[end] - pxx[start], pyy[end] - pyy[start]
    sxy = pxy[end] - pxy[start]
    denominator = count * sxx - sx * sx
    slope = np.divide(
        count * sxy - sx * sy,
        denominator,
        out=np.zeros(np.broadcast(count, denominator).shape, dtype=np.float64),
        where=np.abs(denominator) >= 1e-15,
    )
    intercept = (sy - slope * sx) / count
    sse = syy - 2.0 * slope * sxy - 2.0 * intercept * sy + slope * slope * sxx + 2.0 * slope * intercept * sx + count * intercept * intercept
    return np.maximum(sse, 0.0)


class UpperTailThreeSegmentCP:
    """Three-segment change points on the smoothed descending score curve."""

    def __init__(
        self,
        smooth_window: int = 21,
        smooth_polyorder: int = 2,
        min_high_length: int = 8,
        min_middle_length: int = 24,
        min_background_length: int = 64,
        eps: float = 1e-8,
    ) -> None:
        frozen = (smooth_window, smooth_polyorder, min_high_length, min_middle_length, min_background_length)
        if frozen != (21, 2, 8, 24, 64):
            raise ValueError("UT-3CP V3 parameters are frozen by protocol")
        self.window = int(smooth_window)
        self.order = int(smooth_polyorder)
        self.min_high = int(min_high_length)
        self.min_middle = int(min_middle_length)
        self.min_background = int(min_background_length)
        self.eps = float(eps)

    def apply(
        self, minmax_score: torch.Tensor, background_indices: torch.Tensor
    ) -> ThresholdV3Result:
        score, shape = _vector(minmax_score, "minmax_score")
        bc = _indices(background_indices, score.size)
        n = score.size
        if n < max(self.window, self.min_high + self.min_middle + self.min_background):
            diagnostics = {"failure_reason": "curve_too_short", "curve_length": n}
            return _result(np.zeros(n, bool), shape, {}, None, None, True, diagnostics)
        order = np.argsort(-score, kind="stable")
        sorted_score = score[order]
        x = np.log((np.arange(1, n + 1, dtype=np.float64) - 0.5) / n)
        smooth = savgol_filter(sorted_score, self.window, self.order, mode="interp")

        def optimize_segments(curve: np.ndarray):
            px, py = _prefix(x), _prefix(curve)
            pxx, pyy, pxy = _prefix(x * x), _prefix(curve * curve), _prefix(x * curve)
            one = _segment_fit(px, py, pxx, pyy, pxy, 0, n)
            two_breaks = np.arange(self.min_high, n - self.min_background + 1)
            two_sse = _segment_sse_array(
                px, py, pxx, pyy, pxy, 0, two_breaks
            ) + _segment_sse_array(px, py, pxx, pyy, pxy, two_breaks, n)
            best2_index = int(np.argmin(two_sse))
            best2 = (float(two_sse[best2_index]), int(two_breaks[best2_index]))
            best3 = (float("inf"), -1, -1)
            for first_break in range(
                self.min_high, n - self.min_middle - self.min_background + 1
            ):
                first_sse = _segment_fit(px, py, pxx, pyy, pxy, 0, first_break)[2]
                second_breaks = np.arange(
                    first_break + self.min_middle, n - self.min_background + 1
                )
                total = first_sse + _segment_sse_array(
                    px, py, pxx, pyy, pxy, first_break, second_breaks
                ) + _segment_sse_array(
                    px, py, pxx, pyy, pxy, second_breaks, n
                )
                local_index = int(np.argmin(total))
                local_sse = float(total[local_index])
                if local_sse < best3[0]:
                    best3 = (local_sse, first_break, int(second_breaks[local_index]))
            return one, best2, best3, (px, py, pxx, pyy, pxy)

        # The smoothed copy is used only to locate the operational change
        # points.  Model-order BIC is deliberately fitted on the untouched
        # sorted curve so smoothing cannot manufacture a third transition
        # segment around a genuine two-segment breakpoint.
        search_one, search_two, search_three, search_prefixes = optimize_segments(smooth)
        bic_one, bic_two, bic_three, _ = optimize_segments(sorted_score)
        k1, k2 = search_three[1], search_three[2]
        if k1 < 1 or k2 <= k1 or k2 >= n:
            diagnostics = {"failure_reason": "no_valid_three_segment_solution"}
            return _result(np.zeros(n, bool), shape, {}, None, None, True, diagnostics)
        px, py, pxx, pyy, pxy = search_prefixes
        high_fit = _segment_fit(px, py, pxx, pyy, pxy, 0, k1)
        middle_fit = _segment_fit(px, py, pxx, pyy, pxy, k1, k2)
        background_fit = _segment_fit(px, py, pxx, pyy, pxy, k2, n)
        t_high = float((sorted_score[k1 - 1] + sorted_score[k1]) / 2.0)
        t_low = float((sorted_score[k2 - 1] + sorted_score[k2]) / 2.0)
        mask = score > t_high
        state = np.zeros(n, dtype=np.int64)
        state[score > t_low] = 1
        state[score > t_high] = 2
        sse1, sse2, sse3 = bic_one[2], bic_two[0], bic_three[0]
        bic1 = n * np.log(sse1 / n + self.eps) + 2.0 * np.log(n)
        bic2 = n * np.log(sse2 / n + self.eps) + 5.0 * np.log(n)
        bic3 = n * np.log(sse3 / n + self.eps) + 8.0 * np.log(n)
        diagnostics = {
            "k1": int(k1),
            "k2": int(k2),
            "high_segment_area": float(k1 / n),
            "middle_segment_area": float((k2 - k1) / n),
            "background_segment_area": float((n - k2) / n),
            "threshold_high": t_high,
            "threshold_low": t_low,
            "slope_high": high_fit[0],
            "intercept_high": high_fit[1],
            "slope_middle": middle_fit[0],
            "intercept_middle": middle_fit[1],
            "slope_background": background_fit[0],
            "intercept_background": background_fit[1],
            "high_mean_score": float(sorted_score[:k1].mean()),
            "middle_mean_score": float(sorted_score[k1:k2].mean()),
            "background_mean_score": float(sorted_score[k2:].mean()),
            "high_score_min": float(sorted_score[k1 - 1]),
            "high_score_max": float(sorted_score[0]),
            "middle_score_min": float(sorted_score[k2 - 1]),
            "middle_score_max": float(sorted_score[k1]),
            "background_score_min": float(sorted_score[-1]),
            "background_score_max": float(sorted_score[k2]),
            "sse1": float(sse1),
            "sse2": float(sse2),
            "sse3": float(sse3),
            "search_sse1_smoothed": float(search_one[2]),
            "search_sse2_smoothed": float(search_two[0]),
            "search_sse3_smoothed": float(search_three[0]),
            "bic_two_segment_change_point": int(bic_two[1]),
            "bic_three_segment_k1": int(bic_three[1]),
            "bic_three_segment_k2": int(bic_three[2]),
            "bic_curve": "raw_sorted_score",
            "change_point_search_curve": "savgol_smoothed_copy",
            "bic1": float(bic1),
            "bic2": float(bic2),
            "bic3": float(bic3),
            "delta_bic_3_vs_2": float(bic2 - bic3),
            "weak_three_segment_support": bool(bic2 - bic3 <= 10.0),
            "ut3cp_weak_three_segment_support": bool(bic2 - bic3 <= 10.0),
            "k1_at_minimum": bool(k1 == self.min_high),
            "foreground_area": float(mask.mean()),
            "bc_above_high_threshold_ratio": float(mask[bc].mean()),
            "failure_reason": None,
        }
        sorted_curve = np.empty(n, dtype=np.float64)
        smoothed_curve = np.empty(n, dtype=np.float64)
        rank_map = np.empty(n, dtype=np.float64)
        sorted_curve[order] = sorted_score
        smoothed_curve[order] = smooth
        rank_map[order] = np.arange(1, n + 1, dtype=np.float64) / n
        return _result(
            mask,
            shape,
            {
                "state": state,
                "rank_fraction": rank_map,
                "sorted_value_at_patch": sorted_curve,
                "smoothed_value_at_patch": smoothed_curve,
            },
            t_high,
            t_low,
            False,
            diagnostics,
        )


class TriStateHysteresis:
    """Keep only 8-connected low-support components containing a high seed."""

    @staticmethod
    def apply(high_seed: torch.Tensor, low_support: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not torch.is_tensor(high_seed) or not torch.is_tensor(low_support):
            raise TypeError("high_seed and low_support must be tensors")
        if tuple(high_seed.shape) != tuple(low_support.shape) or high_seed.numel() == 0:
            raise ValueError("high_seed and low_support shapes must match and be non-empty")
        shape = tuple(high_seed.shape)
        if len(shape) < 2:
            raise ValueError("hysteresis expects a spatial tensor")
        high = high_seed.detach().cpu().numpy().astype(bool).reshape(shape[-2:])
        low = low_support.detach().cpu().numpy().astype(bool).reshape(shape[-2:])
        if np.any(high & ~low):
            raise ValueError("low support must contain every high-confidence seed")
        labels, count = connected_components(low, structure=np.ones((3, 3), dtype=np.uint8))
        seeded_labels = np.unique(labels[high])
        seeded_labels = seeded_labels[seeded_labels != 0]
        output = np.isin(labels, seeded_labels)
        component_sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
        diagnostics = {
            "support_component_count": int(count),
            "seeded_component_count": int(seeded_labels.size),
            "high_seed_area": float(high.mean()),
            "low_support_area": float(low.mean()),
            "foreground_area": float(output.mean()),
            "hysteresis_large_support_component": bool(
                component_sizes.size and component_sizes.max() / low.size > 0.5
            ),
        }
        return _tensor_map(output.astype(np.float32), shape), diagnostics
