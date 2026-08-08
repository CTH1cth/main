from __future__ import annotations

import numpy as np
import torch
from scipy.stats import genpareto

from models.gbsp_adaptive_threshold_v2 import (
    BCBetaMixtureCalibrator,
    BCLogitGaussianMixtureCalibrator,
    ExpandedBackgroundEVTCalibrator,
    QueryDistributionChangePoint,
)


def _auc(labels: np.ndarray, score: np.ndarray) -> float:
    positive = score[labels == 1]
    negative = score[labels == 0]
    return float(
        ((positive[:, None] > negative[None, :]).mean())
        + 0.5 * ((positive[:, None] == negative[None, :]).mean())
    )


def _grid(values: np.ndarray) -> torch.Tensor:
    padded = np.resize(values.astype(np.float32), 37 * 37)
    return torch.from_numpy(padded).reshape(1, 37, 37)


def _assert_diagnostics_equal(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for key in first:
        if torch.is_tensor(first[key]):
            assert torch.equal(first[key], second[key])
        else:
            assert first[key] == second[key]


def test_ba_bmc_recovers_deterministic_beta_components():
    rng = np.random.default_rng(71)
    background = rng.beta(2.0, 8.0, 500)
    foreground = rng.beta(8.0, 2.0, 100)
    values = np.concatenate((background, foreground))
    score = _grid(values)
    bc = torch.arange(300)
    method = BCBetaMixtureCalibrator(anchor_strength=1.0)
    first = method.apply(score, bc)
    second = method.apply(score, bc)
    posterior = first.continuous_map.reshape(-1).numpy()[:600]
    labels = np.concatenate((np.zeros(500), np.ones(100)))
    assert not first.numerical_failure
    assert first.diagnostics["mean_bg"] < first.diagnostics["mean_fg"]
    assert _auc(labels, posterior) > 0.95
    assert first.diagnostics["mean_bg"] < first.equivalent_minmax_threshold < first.diagnostics["mean_fg"]
    _assert_diagnostics_equal(first.diagnostics, second.diagnostics)
    assert torch.equal(first.binary_mask, second.binary_mask)


def test_ba_lgmc_em_is_ordered_finite_monotone_and_deterministic():
    rng = np.random.default_rng(73)
    bg = rng.normal(-2.0, 0.6, 500)
    fg = rng.normal(1.5, 0.8, 100)
    score = _grid(1.0 / (1.0 + np.exp(-np.concatenate((bg, fg)))))
    bc = torch.arange(300)
    method = BCLogitGaussianMixtureCalibrator(anchor_strength=1.0)
    first = method.apply(score, bc)
    second = method.apply(score, bc)
    history = np.asarray(first.diagnostics["em_objective_history"])
    assert first.diagnostics["mu_bg"] < first.diagnostics["mu_fg"]
    assert first.continuous_map is not None
    assert bool(torch.isfinite(first.continuous_map).all())
    assert bool(np.all(np.diff(history) >= -1e-10))
    _assert_diagnostics_equal(first.diagnostics, second.diagnostics)
    assert torch.equal(first.binary_mask, second.binary_mask)


def test_evt_gpd_fit_and_continuous_extrapolated_tail_probabilities():
    rng = np.random.default_rng(79)
    excess = genpareto.rvs(c=0.2, scale=1.0, size=4000, random_state=rng)
    method = ExpandedBackgroundEVTCalibrator(evt_q=0.025)
    shape, scale, converged, _ = method._fit_gpd(excess)
    assert converged
    assert abs(shape - 0.2) < 0.12
    assert abs(scale - 1.0) < 0.15

    raw = np.exp(np.linspace(-4.0, 5.0, 37 * 37))
    score = (raw - raw.min()) / (raw.max() - raw.min())
    result = method.apply(_grid(raw), _grid(score), torch.arange(411))
    pvalue = result.continuous_map.reshape(-1).numpy()
    assert pvalue[-1] < pvalue[-2] < pvalue[-3]
    assert len(np.unique(pvalue[-10:])) > 5


def test_qdcp_piecewise_break_and_single_line_bic():
    n = 37 * 37
    first = np.linspace(1.0, 0.72, 150, endpoint=False)
    second = np.linspace(0.72, 0.0, n - 150)
    result = QueryDistributionChangePoint(method="pl").apply(_grid(np.concatenate((first, second))))
    assert abs(result.diagnostics["change_point_index"] - 150) <= 5

    line = np.linspace(1.0, 0.0, n)
    single = QueryDistributionChangePoint(method="pl").apply(_grid(line))
    assert single.diagnostics["delta_bic"] <= 10.0


def test_all_v2_methods_are_exactly_deterministic_on_same_input():
    values = torch.linspace(0.0, 1.0, 37 * 37).reshape(1, 37, 37)
    raw = torch.exp(values * 4.0)
    bc = torch.arange(411)
    methods = (
        BCBetaMixtureCalibrator(0.5),
        BCLogitGaussianMixtureCalibrator(0.5),
        ExpandedBackgroundEVTCalibrator(0.01),
        QueryDistributionChangePoint("pl"),
        QueryDistributionChangePoint("kneedle"),
    )
    for method in methods:
        first = method.apply(raw, values, bc) if isinstance(method, ExpandedBackgroundEVTCalibrator) else (
            method.apply(values) if isinstance(method, QueryDistributionChangePoint) else method.apply(values, bc)
        )
        second = method.apply(raw, values, bc) if isinstance(method, ExpandedBackgroundEVTCalibrator) else (
            method.apply(values) if isinstance(method, QueryDistributionChangePoint) else method.apply(values, bc)
        )
        _assert_diagnostics_equal(first.diagnostics, second.diagnostics)
        assert first.equivalent_minmax_threshold == second.equivalent_minmax_threshold
        assert torch.equal(first.binary_mask, second.binary_mask)
