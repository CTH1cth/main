"""PCA squared-prediction-error (SPE) control-limit calibration.

The module is deliberately image-local and GT-free.  It consumes the
singular spectrum and raw orthogonal residual produced by the frozen,
single-global GBSP PCA model; it does not refit PCA or alter residual ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import gamma as gamma_distribution


@dataclass(frozen=True)
class PCASpectrum:
    background_candidate_count: int
    feature_dimension: int
    effective_spectrum_dimension: int
    pca_rank: int
    singular_values: torch.Tensor
    covariance_eigenvalues: torch.Tensor
    retained_eigenvalues: torch.Tensor
    discarded_eigenvalues: torch.Tensor
    theta1: float
    theta2: float
    theta3: float
    numerical_validity: bool
    failure_reason: str | None


@dataclass(frozen=True)
class SPELimit:
    method: str
    control_level: float
    tau: float | None
    numerical_validity: bool
    failure_reason: str | None
    h0: float | None = None
    bracket: float | None = None
    gamma_shape: float | None = None
    gamma_scale: float | None = None


@dataclass(frozen=True)
class CalibratedResidual:
    score: torch.Tensor | None
    mask: torch.Tensor | None
    tau: float | None
    numerical_validity: bool
    failure_reason: str | None


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


class PCASpectrumExtractor:
    """Recover covariance eigenvalues from the cached centered-data SVD."""

    def extract(
        self,
        singular_values: torch.Tensor,
        background_candidate_count: int,
        feature_dimension: int,
        pca_rank: int,
    ) -> PCASpectrum:
        count = int(background_candidate_count)
        dimension = int(feature_dimension)
        rank = int(pca_rank)
        if not torch.is_tensor(singular_values) or singular_values.ndim != 1:
            raise ValueError("singular_values must be a one-dimensional tensor")
        singular = singular_values.detach().cpu().to(torch.float64).contiguous()
        if not bool(torch.isfinite(singular).all()) or bool((singular < 0).any()):
            raise ValueError("singular_values must be finite and non-negative")
        if count < 1 or dimension < 1 or rank < 0:
            raise ValueError("background count, feature dimension and rank are invalid")

        effective = min(dimension, max(count - 1, 0))
        available = min(effective, int(singular.numel()))
        covariance = (
            singular[:available].square() / float(count - 1)
            if count >= 2
            else singular.new_empty((0,))
        )
        retained = covariance[: min(rank, available)].clone()
        discarded = covariance[rank:available].clone() if rank < available else covariance.new_empty((0,))

        failure = None
        if count < 3:
            failure = "background_candidate_count_below_three"
        elif effective < rank + 1:
            failure = "discarded_spectrum_empty"
        elif int(singular.numel()) < effective:
            failure = "singular_spectrum_shorter_than_effective_dimension"
        elif discarded.numel() == 0:
            failure = "discarded_spectrum_empty"

        theta1 = float(discarded.sum()) if discarded.numel() else 0.0
        theta2 = float(discarded.square().sum()) if discarded.numel() else 0.0
        theta3 = float(discarded.pow(3).sum()) if discarded.numel() else 0.0
        if failure is None and not _finite_positive(theta1):
            failure = "theta1_nonpositive_or_nonfinite"
        if failure is None and not _finite_positive(theta2):
            failure = "theta2_nonpositive_or_nonfinite"
        if failure is None and (not math.isfinite(theta3) or theta3 < 0.0):
            failure = "theta3_negative_or_nonfinite"

        return PCASpectrum(
            background_candidate_count=count,
            feature_dimension=dimension,
            effective_spectrum_dimension=effective,
            pca_rank=rank,
            singular_values=singular.clone(),
            covariance_eigenvalues=covariance,
            retained_eigenvalues=retained,
            discarded_eigenvalues=discarded,
            theta1=theta1,
            theta2=theta2,
            theta3=theta3,
            numerical_validity=failure is None,
            failure_reason=failure,
        )


class JacksonMudholkarSPELimit:
    """Jackson--Mudholkar upper control limit for discarded PCA energy."""

    @staticmethod
    def _normal_quantile(control_level: float) -> float:
        alpha = float(control_level)
        if not 0.0 < alpha < 1.0:
            raise ValueError("control_level must be in (0,1)")
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        return float(normal.icdf(torch.tensor(alpha, dtype=torch.float64)))

    def apply(self, spectrum: PCASpectrum, control_level: float = 0.95) -> SPELimit:
        alpha = float(control_level)
        if not spectrum.numerical_validity:
            return SPELimit("jm_spe", alpha, None, False, spectrum.failure_reason)
        t1, t2, t3 = spectrum.theta1, spectrum.theta2, spectrum.theta3
        h0 = 1.0 - (2.0 * t1 * t3) / (3.0 * t2 * t2)
        if not _finite_positive(h0):
            return SPELimit("jm_spe", alpha, None, False, "h0_nonpositive_or_nonfinite", h0=h0)
        z = self._normal_quantile(alpha)
        bracket = (
            1.0
            + z * math.sqrt(2.0 * t2 * h0 * h0) / t1
            + t2 * h0 * (h0 - 1.0) / (t1 * t1)
        )
        if not _finite_positive(bracket):
            return SPELimit(
                "jm_spe", alpha, None, False, "jm_bracket_nonpositive_or_nonfinite", h0=h0, bracket=bracket
            )
        try:
            tau = t1 * math.exp(math.log(bracket) / h0)
        except (OverflowError, ValueError):
            tau = float("nan")
        if not _finite_positive(tau):
            return SPELimit(
                "jm_spe", alpha, None, False, "jm_tau_nonpositive_or_nonfinite", h0=h0, bracket=bracket
            )
        return SPELimit("jm_spe", alpha, float(tau), True, None, h0=h0, bracket=bracket)


class GammaSPELimit:
    """Moment-matched Gamma control limit, retained as a numerical control."""

    def apply(self, spectrum: PCASpectrum, control_level: float = 0.95) -> SPELimit:
        alpha = float(control_level)
        if not 0.0 < alpha < 1.0:
            raise ValueError("control_level must be in (0,1)")
        if not spectrum.numerical_validity:
            return SPELimit("gamma_spe", alpha, None, False, spectrum.failure_reason)
        t1, t2 = spectrum.theta1, spectrum.theta2
        shape = t1 * t1 / (2.0 * t2)
        scale = 2.0 * t2 / t1
        if not _finite_positive(shape) or not _finite_positive(scale):
            return SPELimit(
                "gamma_spe", alpha, None, False, "gamma_parameters_invalid",
                gamma_shape=shape, gamma_scale=scale,
            )
        tau = float(gamma_distribution.ppf(alpha, a=shape, scale=scale))
        if not _finite_positive(tau):
            return SPELimit(
                "gamma_spe", alpha, None, False, "gamma_tau_nonpositive_or_nonfinite",
                gamma_shape=shape, gamma_scale=scale,
            )
        return SPELimit(
            "gamma_spe", alpha, tau, True, None,
            gamma_shape=float(shape), gamma_scale=float(scale),
        )


class BoundedResidualCalibrator:
    """Map a non-negative raw residual to Q/(Q+tau+eps)."""

    def __init__(self, eps: float = 1e-12) -> None:
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def apply(self, raw_residual: torch.Tensor, limit: SPELimit) -> CalibratedResidual:
        if not torch.is_tensor(raw_residual):
            raise TypeError("raw_residual must be a tensor")
        residual = raw_residual.detach().cpu().to(torch.float64).contiguous()
        if not bool(torch.isfinite(residual).all()) or bool((residual < 0).any()):
            raise ValueError("raw_residual must be finite and non-negative")
        if not limit.numerical_validity or limit.tau is None:
            return CalibratedResidual(None, None, limit.tau, False, limit.failure_reason)
        tau = float(limit.tau)
        score = residual / (residual + tau + self.eps)
        # Use the mathematically equivalent raw-domain comparison for the hard
        # label, so no floating-point epsilon can move a value across the limit.
        mask = residual > tau
        if not bool(torch.isfinite(score).all()) or float(score.min()) < 0.0 or float(score.max()) > 1.0:
            return CalibratedResidual(None, None, tau, False, "calibrated_score_invalid")
        return CalibratedResidual(score, mask, tau, True, None)


def equivalent_minmax_threshold(raw_residual: torch.Tensor, tau: float, eps: float = 1e-12) -> float:
    residual = raw_residual.detach().cpu().to(torch.float64)
    if residual.numel() == 0 or not bool(torch.isfinite(residual).all()):
        raise ValueError("raw_residual must be finite and non-empty")
    return float((float(tau) - float(residual.min())) / (float(residual.max() - residual.min()) + float(eps)))

