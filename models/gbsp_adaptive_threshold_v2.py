"""Deterministic GT-free adaptive thresholding for cached GBSP residuals.

The classes in this module consume only one image's cached GBSP raw/Min-Max
scores and Full-BC indices.  They never read GT, R1, target-area statistics,
or the fixed-0.58 reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import betaln, logsumexp


@dataclass(frozen=True)
class AdaptiveThresholdResult:
    binary_mask: torch.Tensor
    continuous_map: torch.Tensor | None
    equivalent_minmax_threshold: float | None
    numerical_failure: bool
    diagnostics: dict[str, Any]


def _vector(value: torch.Tensor, name: str) -> np.ndarray:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty tensor")
    array = value.detach().cpu().double().reshape(-1).numpy().copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _indices(value: torch.Tensor, size: int) -> np.ndarray:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0:
        raise ValueError("background_indices must be a non-empty vector")
    indices = value.detach().cpu().long().numpy().copy()
    if indices.min() < 0 or indices.max() >= size or np.unique(indices).size != indices.size:
        raise ValueError("background_indices are invalid or duplicated")
    return indices


def _mask_result(
    mask: np.ndarray,
    continuous: np.ndarray | None,
    score: np.ndarray,
    failure: bool,
    diagnostics: dict[str, Any],
) -> AdaptiveThresholdResult:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    selected = score[mask]
    threshold = float(selected.min()) if selected.size else None
    return AdaptiveThresholdResult(
        binary_mask=torch.from_numpy(mask.copy()).reshape(1, 37, 37),
        continuous_map=(
            torch.from_numpy(np.asarray(continuous, dtype=np.float32).copy()).reshape(1, 37, 37)
            if continuous is not None
            else None
        ),
        equivalent_minmax_threshold=threshold,
        numerical_failure=bool(failure),
        diagnostics=diagnostics,
    )


def _beta_logpdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (
        (alpha - 1.0) * np.log(x)
        + (beta - 1.0) * np.log1p(-x)
        - betaln(alpha, beta)
    )


def _beta_moments(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return fallback
    mean = float(values.mean())
    variance = float(values.var())
    common = mean * (1.0 - mean) / (variance + 1e-8) - 1.0
    alpha, beta = mean * common, (1.0 - mean) * common
    if not np.isfinite([alpha, beta]).all() or alpha < 0.2 or beta < 0.2:
        return fallback
    return float(np.clip(alpha, 0.2, 200.0)), float(np.clip(beta, 0.2, 200.0))


def _beta_one_component(score: np.ndarray) -> tuple[float, float, float, bool]:
    initial = _beta_moments(score, (2.0, 2.0))

    def objective(theta: np.ndarray) -> float:
        return float(-_beta_logpdf(score, theta[0], theta[1]).sum())

    fitted = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        bounds=((0.2, 200.0), (0.2, 200.0)),
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-7},
    )
    theta = fitted.x if np.isfinite(fitted.fun) else np.asarray(initial)
    log_likelihood = -objective(theta)
    return float(theta[0]), float(theta[1]), float(log_likelihood), bool(fitted.success)


class BCBetaMixtureCalibrator:
    """BA-BMC: full-query two-Beta mixture with a soft Full-BC anchor."""

    def __init__(
        self,
        anchor_strength: float = 1.0,
        score_clip: float = 1e-4,
        max_iter: int = 300,
        tolerance: float = 1e-7,
    ) -> None:
        if float(anchor_strength) not in {0.5, 1.0}:
            raise ValueError("BA-BMC anchor_strength must be 0.5 or 1.0")
        self.anchor_strength = float(anchor_strength)
        self.score_clip = float(score_clip)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def _initializations(self, score: np.ndarray, bc: np.ndarray) -> list[tuple[str, np.ndarray]]:
        ordered = np.sort(score)
        n = score.size
        top15 = ordered[max(0, int(np.floor(0.85 * n))):]
        top20 = ordered[max(0, int(np.floor(0.80 * n))):]
        low60 = ordered[: max(2, int(np.ceil(0.60 * n)))]
        median = float(np.median(score))
        low_med, high_med = score[score <= median], score[score > median]
        definitions = (
            ("A_bc_top15", score[bc], top15, 0.15),
            ("B_low60_top20", low60, top20, 0.20),
            ("C_median_split", low_med, high_med, min(0.30, high_med.size / n)),
        )
        output = []
        for name, background, foreground, pi in definitions:
            ab, bb = _beta_moments(background, (2.0, 5.0))
            af, bf = _beta_moments(foreground, (5.0, 2.0))
            output.append((name, np.asarray([ab, bb, af, bf, np.clip(pi, 0.001, 0.5)])))
        return output

    def apply(self, minmax_score: torch.Tensor, background_indices: torch.Tensor) -> AdaptiveThresholdResult:
        original = _vector(minmax_score, "minmax_score")
        score = np.clip(original, self.score_clip, 1.0 - self.score_clip)
        bc = _indices(background_indices, score.size)

        def objective(theta: np.ndarray) -> float:
            ab, bb, af, bf, pi = theta
            log_bg = _beta_logpdf(score, ab, bb)
            log_fg = _beta_logpdf(score, af, bf)
            mixture = logsumexp(
                np.stack((np.log1p(-pi) + log_bg, np.log(pi) + log_fg)), axis=0
            )
            value = mixture.mean() + self.anchor_strength * log_bg[bc].mean()
            return float(-value)

        attempts = []
        for name, initial in self._initializations(score, bc):
            fit = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=((0.2, 200.0),) * 4 + ((0.001, 0.5),),
                options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": self.tolerance},
            )
            attempts.append((float(fit.fun), name, fit))
        finite = [item for item in attempts if np.isfinite(item[0]) and np.isfinite(item[2].x).all()]
        if not finite:
            diagnostics = {
                "bmc_fit_invalid": True,
                "failure_reason": "all_initializations_nonfinite",
                "anchor_strength": self.anchor_strength,
                "optimizer_converged": False,
            }
            return _mask_result(np.zeros(score.size, bool), None, original, True, diagnostics)
        _, selected_name, fit = min(finite, key=lambda item: (item[0], item[1]))
        ab, bb, af, bf, pi = map(float, fit.x)
        mean_bg, mean_fg = ab / (ab + bb), af / (af + bf)
        labels_swapped = False
        if mean_bg > mean_fg:
            ab, bb, af, bf = af, bf, ab, bb
            pi = 1.0 - pi
            mean_bg, mean_fg = mean_fg, mean_bg
            labels_swapped = True
        log_bg = _beta_logpdf(score, ab, bb)
        log_fg = _beta_logpdf(score, af, bf)
        log_weight_bg = np.log1p(-pi) + log_bg if 0.0 < pi < 1.0 else np.full_like(score, -np.inf)
        log_weight_fg = np.log(pi) + log_fg if 0.0 < pi < 1.0 else np.full_like(score, -np.inf)
        normalizer = logsumexp(np.stack((log_weight_bg, log_weight_fg)), axis=0)
        posterior = np.exp(log_weight_fg - normalizer)
        ordered_posterior = posterior[np.argsort(score, kind="stable")]
        monotonic_violations = int(np.sum(np.diff(ordered_posterior) < -1e-9))
        mix_log_likelihood = float(normalizer.sum())
        one_a, one_b, one_ll, one_converged = _beta_one_component(score)
        bic_one = 2.0 * np.log(score.size) - 2.0 * one_ll
        bic_two = 5.0 * np.log(score.size) - 2.0 * mix_log_likelihood
        mean_gap = mean_fg - mean_bg
        finite_output = np.isfinite(
            [ab, bb, af, bf, pi, mean_bg, mean_fg, mix_log_likelihood, bic_one, bic_two]
        ).all() and np.isfinite(posterior).all()
        valid = bool(
            fit.success
            and finite_output
            and 0.001 <= pi <= 0.5
            and mean_gap >= 0.05
            and monotonic_violations == 0
        )
        mask = posterior > 0.5 if valid else np.zeros(score.size, dtype=bool)
        diagnostics = {
            "alpha_bg": ab,
            "beta_bg": bb,
            "alpha_fg": af,
            "beta_fg": bf,
            "pi_fg": pi,
            "mean_bg": mean_bg,
            "mean_fg": mean_fg,
            "mean_gap": mean_gap,
            "bic_one_component": float(bic_one),
            "bic_two_component": float(bic_two),
            "delta_bic": float(bic_one - bic_two),
            "bic_two_component_supported": bool(bic_one - bic_two > 10.0),
            "one_component_alpha": one_a,
            "one_component_beta": one_b,
            "one_component_converged": one_converged,
            "optimizer_converged": bool(fit.success),
            "optimizer_status": int(fit.status),
            "optimizer_iterations": int(getattr(fit, "nit", -1)),
            "initialization_selected": selected_name,
            "labels_swapped": labels_swapped,
            "posterior_monotonic_violation_count": monotonic_violations,
            "anchor_strength": self.anchor_strength,
            "bmc_fit_invalid": not valid,
            "failure_reason": None if valid else "legality_check_failed",
        }
        return _mask_result(mask, posterior, original, not valid, diagnostics)


def _normal_logpdf(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return -0.5 * ((values - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


class BCLogitGaussianMixtureCalibrator:
    """BA-LGMC: deterministic anchored EM in clipped-score logit space."""

    def __init__(
        self,
        anchor_strength: float = 1.0,
        score_clip: float = 1e-4,
        max_iter: int = 200,
        tolerance: float = 1e-7,
    ) -> None:
        if float(anchor_strength) not in {0.5, 1.0}:
            raise ValueError("BA-LGMC anchor_strength must be 0.5 or 1.0")
        self.anchor_strength = float(anchor_strength)
        self.score_clip = float(score_clip)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    @staticmethod
    def _sigma(values: np.ndarray) -> float:
        if values.size < 2:
            return 1.0
        return float(np.clip(values.std(), 0.05, 10.0))

    def _objective(self, y: np.ndarray, bc: np.ndarray, params: np.ndarray) -> tuple[float, np.ndarray]:
        mb, sb, mf, sf, pi = map(float, params)
        log_bg = np.log1p(-pi) + _normal_logpdf(y, mb, sb)
        log_fg = np.log(pi) + _normal_logpdf(y, mf, sf)
        mixture = logsumexp(np.stack((log_bg, log_fg)), axis=0)
        anchored = mixture.mean() + self.anchor_strength * _normal_logpdf(y[bc], mb, sb).mean()
        posterior = np.exp(log_fg - mixture)
        return float(anchored), posterior

    def _run_em(self, y: np.ndarray, bc: np.ndarray, initial: np.ndarray) -> dict[str, Any]:
        params = initial.astype(np.float64).copy()
        objective, posterior = self._objective(y, bc, params)
        history = [objective]
        converged = False
        for _ in range(self.max_iter):
            rb = 1.0 - posterior
            adjusted_bg = rb.copy()
            adjusted_bg[bc] = (adjusted_bg[bc] + self.anchor_strength) / (1.0 + self.anchor_strength)
            adjusted_fg = 1.0 - adjusted_bg
            sum_bg, sum_fg = adjusted_bg.sum(), adjusted_fg.sum()
            if sum_bg <= 1e-12 or sum_fg <= 1e-12:
                break
            proposal = np.asarray(
                [
                    np.sum(adjusted_bg * y) / sum_bg,
                    1.0,
                    np.sum(adjusted_fg * y) / sum_fg,
                    1.0,
                    np.clip(adjusted_fg.mean(), 0.001, 0.5),
                ],
                dtype=np.float64,
            )
            proposal[1] = np.clip(
                np.sqrt(np.sum(adjusted_bg * (y - proposal[0]) ** 2) / sum_bg + 1e-4),
                0.05,
                10.0,
            )
            proposal[3] = np.clip(
                np.sqrt(np.sum(adjusted_fg * (y - proposal[2]) ** 2) / sum_fg + 1e-4),
                0.05,
                10.0,
            )
            new_objective, new_posterior = self._objective(y, bc, proposal)
            # Deterministic damping preserves a monotone anchored objective.
            if new_objective < objective - 1e-12:
                accepted = False
                for power in range(1, 21):
                    weight = 0.5**power
                    candidate = params + weight * (proposal - params)
                    candidate[1] = np.clip(candidate[1], 0.05, 10.0)
                    candidate[3] = np.clip(candidate[3], 0.05, 10.0)
                    candidate[4] = np.clip(candidate[4], 0.001, 0.5)
                    candidate_objective, candidate_posterior = self._objective(y, bc, candidate)
                    if candidate_objective >= objective - 1e-12:
                        proposal, new_objective, new_posterior = candidate, candidate_objective, candidate_posterior
                        accepted = True
                        break
                if not accepted:
                    converged = True
                    break
            delta = new_objective - objective
            params, objective, posterior = proposal, new_objective, new_posterior
            history.append(objective)
            if abs(delta) < self.tolerance:
                converged = True
                break
        return {
            "params": params,
            "objective": objective,
            "posterior": posterior,
            "history": history,
            "converged": converged,
            "iterations": len(history) - 1,
        }

    def apply(self, minmax_score: torch.Tensor, background_indices: torch.Tensor) -> AdaptiveThresholdResult:
        original = _vector(minmax_score, "minmax_score")
        score = np.clip(original, self.score_clip, 1.0 - self.score_clip)
        y = np.log(score / (1.0 - score))
        bc = _indices(background_indices, score.size)
        top = y[y >= np.quantile(y, 0.85)]
        bc_values = y[bc]
        mad = 1.4826 * np.median(np.abs(bc_values - np.median(bc_values)))
        initial_a = np.asarray(
            [
                np.median(bc_values), np.clip(mad, 0.05, 10.0),
                top.mean(), self._sigma(top), 0.15,
            ]
        )
        q30, q70 = np.quantile(y, [0.30, 0.70])
        initial_b = np.asarray(
            [q30, self._sigma(y[y <= np.median(y)]), q70, self._sigma(y[y > np.median(y)]), 0.20]
        )
        attempts = [
            ("A_bc_top15", self._run_em(y, bc, initial_a)),
            ("B_q30_q70", self._run_em(y, bc, initial_b)),
        ]
        selected_name, selected = max(attempts, key=lambda item: (item[1]["objective"], item[0]))
        mb, sb, mf, sf, pi = map(float, selected["params"])
        labels_swapped = False
        if mb > mf:
            mb, sb, mf, sf, pi = mf, sf, mb, sb, 1.0 - pi
            labels_swapped = True
        log_bg = np.log1p(-pi) + _normal_logpdf(y, mb, sb) if 0.0 < pi < 1.0 else np.full_like(y, -np.inf)
        log_fg = np.log(pi) + _normal_logpdf(y, mf, sf) if 0.0 < pi < 1.0 else np.full_like(y, -np.inf)
        normalizer = logsumexp(np.stack((log_bg, log_fg)), axis=0)
        posterior = np.exp(log_fg - normalizer)
        mix_ll = float(normalizer.sum())
        one_mean, one_sigma = float(y.mean()), float(np.clip(y.std(), 0.05, 10.0))
        one_ll = float(_normal_logpdf(y, one_mean, one_sigma).sum())
        bic_one = 2.0 * np.log(y.size) - 2.0 * one_ll
        bic_two = 5.0 * np.log(y.size) - 2.0 * mix_ll
        monotonic_violations = int(np.sum(np.diff(posterior[np.argsort(score, kind="stable")]) < -1e-9))
        mean_gap = mf - mb
        finite = np.isfinite([mb, sb, mf, sf, pi, mix_ll, bic_one, bic_two]).all() and np.isfinite(posterior).all()
        valid = bool(
            selected["converged"]
            and finite
            and 0.001 <= pi <= 0.5
            and 0.05 <= sb <= 10.0
            and 0.05 <= sf <= 10.0
            and mean_gap >= 0.25
        )
        mask = posterior > 0.5 if valid else np.zeros(score.size, dtype=bool)
        diagnostics = {
            "mu_bg": mb,
            "sigma_bg": sb,
            "mu_fg": mf,
            "sigma_fg": sf,
            "pi_fg": pi,
            "mean_gap": mean_gap,
            "bic_one_component": float(bic_one),
            "bic_two_component": float(bic_two),
            "delta_bic": float(bic_one - bic_two),
            "bic_two_component_supported": bool(bic_one - bic_two > 10.0),
            "em_iterations": int(selected["iterations"]),
            "log_likelihood": mix_ll,
            "anchored_objective": float(selected["objective"]),
            "em_objective_history": [float(value) for value in selected["history"]],
            "optimizer_converged": bool(selected["converged"]),
            "initialization_selected": selected_name,
            "labels_swapped": labels_swapped,
            "posterior_monotonic_violation_count": monotonic_violations,
            "anchor_strength": self.anchor_strength,
            "lgmc_fit_invalid": not valid,
            "failure_reason": None if valid else "legality_check_failed",
        }
        return _mask_result(mask, posterior, original, not valid, diagnostics)


class ExpandedBackgroundEVTCalibrator:
    """EB-EVT with a fixed median-expanded reference and GPD upper tail."""

    def __init__(
        self,
        evt_q: float = 0.025,
        reference_quantile: float = 0.80,
        min_tail_count: int = 30,
        eps: float = 1e-8,
    ) -> None:
        if float(evt_q) not in {0.01, 0.025, 0.05}:
            raise ValueError("evt_q must be one of 0.01, 0.025, 0.05")
        self.evt_q = float(evt_q)
        self.reference_quantile = float(reference_quantile)
        self.min_tail_count = int(min_tail_count)
        self.eps = float(eps)

    @staticmethod
    def _gpd_nll(theta: np.ndarray, excess: np.ndarray) -> float:
        xi, log_scale = float(theta[0]), float(theta[1])
        scale = np.exp(log_scale)
        support = 1.0 + xi * excess / scale
        if scale <= 1e-6 or np.any(support <= 0.0):
            return 1e100
        if abs(xi) < 1e-7:
            return float(excess.size * log_scale + np.sum(excess / scale))
        return float(excess.size * log_scale + (1.0 / xi + 1.0) * np.log(support).sum())

    def _fit_gpd(self, excess: np.ndarray) -> tuple[float, float, bool, int]:
        scale0 = max(float(excess.mean()), 1e-5)
        bounds = ((-0.5, 0.8), (np.log(1.000001e-6), np.log(max(scale0 * 100.0, 1e-4))))
        attempts = []
        for xi0 in (-0.1, 0.0, 0.1, 0.2):
            fit = minimize(
                self._gpd_nll,
                np.asarray([xi0, np.log(scale0)]),
                args=(excess,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-7},
            )
            attempts.append(fit)
        finite = [fit for fit in attempts if np.isfinite(fit.fun) and np.isfinite(fit.x).all() and fit.fun < 1e90]
        if not finite:
            return float("nan"), float("nan"), False, -1
        fit = min(finite, key=lambda item: float(item.fun))
        return float(fit.x[0]), float(np.exp(fit.x[1])), bool(fit.success), int(getattr(fit, "nit", -1))

    def apply(
        self,
        raw_residual: torch.Tensor,
        minmax_score: torch.Tensor,
        background_indices: torch.Tensor,
    ) -> AdaptiveThresholdResult:
        raw = _vector(raw_residual, "raw_residual")
        score = np.clip(_vector(minmax_score, "minmax_score"), 1e-4, 1.0 - 1e-4)
        bc = _indices(background_indices, score.size)
        log_raw = np.log(np.maximum(raw, 0.0) + self.eps)
        low = np.where(score <= np.median(score))[0]
        reference_indices = np.union1d(bc, low)
        reference = log_raw[reference_indices]
        threshold_u = float(np.quantile(reference, self.reference_quantile))
        excess = reference[reference > threshold_u] - threshold_u
        insufficient = excess.size < self.min_tail_count
        if insufficient:
            diagnostics = {
                "reference_pool_size": int(reference.size),
                "tail_threshold_u": threshold_u,
                "tail_sample_count": int(excess.size),
                "evt_q": self.evt_q,
                "evt_tail_insufficient": True,
                "evt_no_discovery": True,
                "gpd_fit_converged": False,
                "failure_reason": "tail_insufficient",
            }
            return _mask_result(np.zeros(score.size, bool), None, score, True, diagnostics)
        xi, scale, converged, iterations = self._fit_gpd(excess)
        if not converged or not np.isfinite([xi, scale]).all() or not (-0.5 <= xi <= 0.8) or scale <= 1e-6:
            diagnostics = {
                "reference_pool_size": int(reference.size),
                "tail_threshold_u": threshold_u,
                "tail_sample_count": int(excess.size),
                "evt_q": self.evt_q,
                "evt_tail_insufficient": False,
                "evt_no_discovery": True,
                "gpd_shape": xi,
                "gpd_scale": scale,
                "gpd_fit_converged": converged,
                "failure_reason": "gpd_fit_invalid",
            }
            return _mask_result(np.zeros(score.size, bool), None, score, True, diagnostics)
        sorted_reference = np.sort(reference)
        pvalue = np.empty_like(log_raw)
        above = log_raw > threshold_u
        empirical = (1.0 + reference.size - np.searchsorted(sorted_reference, log_raw[~above], side="left")) / (reference.size + 1.0)
        pvalue[~above] = empirical
        pu = float(excess.size / reference.size)
        query_excess = log_raw[above] - threshold_u
        support = 1.0 + xi * query_excess / scale
        if abs(xi) < 1e-7:
            survival = np.exp(-query_excess / scale)
        else:
            survival = np.where(support > 0.0, support ** (-1.0 / xi), 0.0)
        pvalue[above] = pu * survival
        pvalue = np.clip(pvalue, 1e-30, 1.0)
        order = np.argsort(pvalue, kind="stable")
        sorted_p = pvalue[order]
        positions = np.arange(1, pvalue.size + 1)
        discoveries = np.where(sorted_p <= positions / pvalue.size * self.evt_q)[0]
        no_discovery = discoveries.size == 0
        if no_discovery:
            selected_k, p_threshold = 0, None
            mask = np.zeros(score.size, dtype=bool)
        else:
            selected_k = int(discoveries[-1] + 1)
            p_threshold = float(sorted_p[selected_k - 1])
            mask = pvalue <= p_threshold
        diagnostics = {
            "reference_pool_size": int(reference.size),
            "reference_pool_indices": torch.from_numpy(reference_indices.astype(np.int64)),
            "tail_threshold_u": threshold_u,
            "tail_sample_count": int(excess.size),
            "tail_probability_at_u": pu,
            "gpd_shape": xi,
            "gpd_scale": scale,
            "gpd_fit_converged": converged,
            "gpd_fit_iterations": iterations,
            "evt_q": self.evt_q,
            "selected_k": selected_k,
            "pvalue_threshold": p_threshold,
            "evt_tail_insufficient": False,
            "evt_no_discovery": no_discovery,
            "failure_reason": None,
        }
        return _mask_result(mask, pvalue, score, False, diagnostics)


def _segment_sse(prefix: dict[str, np.ndarray], left: int, right: int) -> float:
    count = right - left
    if count < 2:
        return float("inf")
    sums = {key: value[right] - value[left] for key, value in prefix.items()}
    centered_x = sums["x2"] - sums["x"] ** 2 / count
    centered_xy = sums["xy"] - sums["x"] * sums["y"] / count
    centered_y = sums["y2"] - sums["y"] ** 2 / count
    if centered_x <= 1e-20:
        return max(float(centered_y), 0.0)
    return max(float(centered_y - centered_xy**2 / centered_x), 0.0)


class QueryDistributionChangePoint:
    """QDCP-PL or QDCP-K on the full descending query-score curve."""

    def __init__(
        self,
        method: str = "pl",
        min_segment: int = 32,
        savgol_window: int = 31,
        savgol_order: int = 2,
        score_clip: float = 1e-4,
    ) -> None:
        if method not in {"pl", "kneedle"}:
            raise ValueError("QDCP method must be pl or kneedle")
        self.method = method
        self.min_segment = int(min_segment)
        self.savgol_window = int(savgol_window)
        self.savgol_order = int(savgol_order)
        self.score_clip = float(score_clip)

    def apply(self, minmax_score: torch.Tensor) -> AdaptiveThresholdResult:
        original = _vector(minmax_score, "minmax_score")
        score = np.clip(original, self.score_clip, 1.0 - self.score_clip)
        ordered = np.sort(score)[::-1].copy()
        n = ordered.size
        x = np.linspace(0.0, 1.0, n)
        if self.method == "pl":
            prefix = {}
            for key, values in (
                ("x", x), ("y", ordered), ("x2", x * x),
                ("xy", x * ordered), ("y2", ordered * ordered),
            ):
                prefix[key] = np.concatenate(([0.0], np.cumsum(values)))
            candidates = np.arange(self.min_segment, n - self.min_segment + 1)
            sse = np.asarray(
                [_segment_sse(prefix, 0, int(k)) + _segment_sse(prefix, int(k), n) for k in candidates]
            )
            winner = int(np.argmin(sse))
            selected_k = int(candidates[winner])
            sse_two = float(sse[winner])
            sse_one = _segment_sse(prefix, 0, n)
            bic_one = n * np.log(max(sse_one / n, 1e-30)) + 2.0 * np.log(n)
            bic_two = n * np.log(max(sse_two / n, 1e-30)) + 4.0 * np.log(n)
            delta_bic = float(bic_one - bic_two)
            knee_strength = None
        else:
            smoothed = savgol_filter(
                ordered, window_length=self.savgol_window, polyorder=self.savgol_order, mode="interp"
            )
            denominator = smoothed[0] - smoothed[-1]
            normalized = (smoothed - smoothed[-1]) / (denominator + 1e-12)
            deviation = (1.0 - x) - normalized
            winner = int(np.argmax(deviation[:-1]))
            selected_k = winner + 1
            knee_strength = float(deviation[winner])
            sse_one = sse_two = delta_bic = None
        threshold = float((ordered[selected_k - 1] + ordered[selected_k]) / 2.0)
        mask = score > threshold
        diagnostics = {
            "change_point_index": selected_k,
            "equivalent_area": float(mask.mean()),
            "threshold": threshold,
            "sse_one_line": sse_one,
            "sse_two_line": sse_two,
            "delta_bic": delta_bic,
            "knee_strength": knee_strength,
            "knee_index": selected_k if self.method == "kneedle" else None,
            "knee_area": float(mask.mean()) if self.method == "kneedle" else None,
            "knee_threshold": threshold if self.method == "kneedle" else None,
            "weak_change_point": bool(delta_bic is not None and delta_bic <= 10.0),
            "method": "qdcp_pl" if self.method == "pl" else "qdcp_k",
            "failure_reason": None,
        }
        return _mask_result(mask, None, original, False, diagnostics)


class BootstrapSubspaceConsensus:
    """Reserved BSC gate; intentionally unavailable before Pilot200 qualifies."""

    def __init__(self) -> None:
        raise RuntimeError(
            "BSC is gated off until a Pilot200 base method reaches the predeclared range"
        )
