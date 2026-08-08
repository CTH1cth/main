from __future__ import annotations

import numpy as np
import torch

from models.gbsp_threshold_v3 import (
    MultiOtsuThreeState,
    OrderedTriGaussianCore,
    TriStateHysteresis,
    UpperTailThreeSegmentCP,
)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _grid(values: np.ndarray) -> torch.Tensor:
    assert values.size == 37 * 37
    return torch.from_numpy(values.astype(np.float32)).reshape(1, 37, 37)


def _nested_equal(first, second) -> None:
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _nested_equal(left, right)
    else:
        assert first == second


def test_multi_otsu_three_state_recovers_highest_peak() -> None:
    rng = np.random.default_rng(301)
    counts = (600, 500, 269)
    values = np.concatenate(
        (
            rng.normal(0.15, 0.03, counts[0]),
            rng.normal(0.40, 0.04, counts[1]),
            rng.normal(0.80, 0.04, counts[2]),
        )
    ).clip(0.0, 1.0)
    labels = np.concatenate((np.zeros(sum(counts[:2]), dtype=int), np.ones(counts[2], dtype=int)))
    result = MultiOtsuThreeState().apply(_grid(values), torch.arange(411))
    prediction = result.mask_core.reshape(-1).numpy().astype(int)
    assert not result.numerical_failure
    assert 0.20 < result.threshold_low < 0.36
    assert 0.48 < result.threshold_high < 0.70
    assert float((prediction == labels).mean()) > 0.95


def test_multi_otsu_invalid_unique_values_is_explicit() -> None:
    score = torch.zeros(1, 37, 37)
    score[..., 10:, :] = 1.0
    result = MultiOtsuThreeState().apply(score, torch.arange(411))
    assert result.numerical_failure
    assert result.diagnostics["multiotsu_invalid_unique_values"]
    assert int(result.mask_core.sum()) == 0


def _tri_gaussian_sample() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(307)
    counts = (821, 342, 206)
    y = np.concatenate(
        (
            rng.normal(-3.0, 0.7, counts[0]),
            rng.normal(-0.5, 0.7, counts[1]),
            rng.normal(2.0, 0.7, counts[2]),
        )
    )
    score = 1.0 / (1.0 + np.exp(-y))
    labels = np.concatenate((np.zeros(sum(counts[:2]), dtype=int), np.ones(counts[2], dtype=int)))
    # BC contains only the two non-foreground states.
    bc = torch.from_numpy(np.concatenate((np.arange(300), np.arange(counts[0], counts[0] + 111)))).long()
    return _grid(score), bc, labels


def test_otgc_is_ordered_monotone_accurate_anchored_and_deterministic() -> None:
    score, bc, labels = _tri_gaussian_sample()
    anchored = OrderedTriGaussianCore(anchor_strength=1.0)
    first = anchored.apply(score, bc)
    second = anchored.apply(score, bc)
    posterior = first.continuous_maps["posterior_c2"].reshape(-1).numpy()
    assert not first.numerical_failure
    assert first.diagnostics["mu0"] < first.diagnostics["mu1"] < first.diagnostics["mu2"]
    assert first.diagnostics["monotonicity_passed"]
    assert _auc(labels, posterior) > 0.95
    threshold_y = np.log(first.threshold_high) - np.log1p(-first.threshold_high)
    assert first.diagnostics["mu1"] < threshold_y < first.diagnostics["mu2"]
    _nested_equal(first.diagnostics, second.diagnostics)
    assert torch.equal(first.mask_core, second.mask_core)
    for key in first.continuous_maps:
        assert torch.equal(first.continuous_maps[key], second.continuous_maps[key])

    unanchored = OrderedTriGaussianCore(anchor_strength=0.0).apply(score, bc)
    anchored_bc = float(first.continuous_maps["posterior_c2"].reshape(-1).index_select(0, bc).mean())
    unanchored_bc = float(unanchored.continuous_maps["posterior_c2"].reshape(-1).index_select(0, bc).mean())
    assert anchored_bc <= unanchored_bc + 1e-8


def test_otgc_two_state_input_is_finite_and_bic_or_weak_flag_detects_it() -> None:
    rng = np.random.default_rng(311)
    y = np.concatenate((rng.normal(-2.0, 0.65, 1000), rng.normal(1.0, 0.65, 369)))
    score = _grid(1.0 / (1.0 + np.exp(-y)))
    result = OrderedTriGaussianCore().apply(score, torch.arange(411))
    for tensor in result.continuous_maps.values():
        assert bool(torch.isfinite(tensor).all())
    assert np.isfinite(result.diagnostics["delta_bic_3_vs_2"])
    assert result.diagnostics["delta_bic_3_vs_2"] <= 10.0 or result.diagnostics["weak_high_component"]


def _piecewise_on_log_rank(n: int, breaks: tuple[int, ...], levels: tuple[float, ...]) -> np.ndarray:
    x = np.log((np.arange(1, n + 1, dtype=np.float64) - 0.5) / n)
    output = np.empty(n, dtype=np.float64)
    starts = (0, *breaks)
    ends = (*breaks, n)
    for start, end, high, low in zip(starts, ends, levels[:-1], levels[1:]):
        output[start:end] = np.interp(x[start:end], (x[start], x[end - 1]), (high, low))
    return output


def test_ut3cp_recovers_three_segments_and_rejects_extra_break_for_two_segments() -> None:
    n = 37 * 37
    three = _piecewise_on_log_rank(n, (120, 480), (1.0, 0.78, 0.40, 0.03))
    three += np.sin(np.arange(n) * 0.13) * 2e-4
    result = UpperTailThreeSegmentCP().apply(_grid(three), torch.arange(411))
    assert not result.numerical_failure
    assert abs(result.diagnostics["k1"] - 120) <= 8
    assert abs(result.diagnostics["k2"] - 480) <= 15

    two = _piecewise_on_log_rank(n, (360,), (1.0, 0.62, 0.05))
    two += np.sin(np.arange(n) * 0.17) * 2e-4
    two_result = UpperTailThreeSegmentCP().apply(_grid(two), torch.arange(411))
    assert two_result.diagnostics["delta_bic_3_vs_2"] <= 10.0


def test_hysteresis_uses_eight_connectivity_and_removes_unseeded_support() -> None:
    high = torch.zeros(1, 7, 7)
    low = torch.zeros_like(high)
    high[0, 1, 1] = 1
    low[0, 1, 1] = 1
    low[0, 2, 2] = 1  # diagonal: retained only with 8-connectivity
    low[0, 2, 3] = 1
    low[0, 5, 5] = 1  # isolated support: removed
    output, diagnostics = TriStateHysteresis.apply(high, low)
    assert output[0, 1, 1] == 1
    assert output[0, 2, 2] == 1
    assert output[0, 2, 3] == 1
    assert output[0, 5, 5] == 0
    assert torch.all(output >= high)
    assert diagnostics["support_component_count"] == 2
