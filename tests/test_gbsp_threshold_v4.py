from __future__ import annotations

import numpy as np
import torch
from skimage.filters import threshold_multiotsu

from models.gbsp_threshold_v4 import (
    HierarchicalUpperTailOtsu,
    LocalResidualContrast,
    OrthogonalResidualCoherence,
    SeededHysteresis,
    empirical_rank,
)


def test_huto_second_level_purifies_upper_tail() -> None:
    rng = np.random.default_rng(401)
    groups = (
        rng.normal(0.15, 0.03, 300), rng.normal(0.45, 0.04, 120),
        rng.normal(0.70, 0.04, 60), rng.normal(0.88, 0.02, 30),
    )
    values = np.concatenate(groups).clip(0, 1)
    score = torch.from_numpy(values.astype(np.float32)).reshape(1, 1, -1)
    result = HierarchicalUpperTailOtsu("core").apply(score, torch.arange(200))
    assert not result.numerical_failure
    d = result.diagnostics
    assert d["t1"] < d["t2"] < d["t3"]
    direct = values > d["t2"]
    core = result.mask.reshape(-1).numpy() > 0.5
    truth = np.arange(values.size) >= 480
    direct_precision = float((truth & direct).sum() / direct.sum())
    core_precision = float((truth & core).sum() / core.sum())
    assert direct[420:480].mean() > 0.8 and core[480:].mean() > 0.8
    assert core_precision > direct_precision


def test_seeded_hysteresis_is_eight_connected_without_morphology() -> None:
    seed = np.zeros((7, 7), bool); support = np.zeros((7, 7), bool)
    seed[1, 1] = True; support[1, 1] = True; support[2, 2] = True; support[2, 3] = True
    support[5, 5] = True
    final, diagnostics = SeededHysteresis.apply(seed, support)
    assert final[1, 1] and final[2, 2] and final[2, 3]
    assert not final[5, 5]
    assert diagnostics["num_support_components"] == 2
    assert diagnostics["num_retained_components"] == 1


def test_orc_coherence_separates_equal_norm_coherent_and_random_vectors() -> None:
    rng = np.random.default_rng(409); height = width = 15; dim = 32
    vectors = rng.normal(size=(height * width, dim)); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    target = np.zeros((height, width), bool); target[4:11, 4:11] = True
    base = rng.normal(size=dim); base /= np.linalg.norm(base)
    for index in np.flatnonzero(target.reshape(-1)):
        value = base + rng.normal(scale=0.02, size=dim); vectors[index] = value / np.linalg.norm(value)
    coherence = OrthogonalResidualCoherence.coherence(vectors, height, width, 1e-8)
    assert coherence[target].mean() > coherence[~target].mean() + 0.35
    magnitude_rank = empirical_rank(np.ones(height * width))
    score = np.sqrt(magnitude_rank * empirical_rank(coherence.reshape(-1)))
    assert score[target.reshape(-1)].mean() > score[~target.reshape(-1)].mean() + 0.15
    assert np.array_equal(coherence, OrthogonalResidualCoherence.coherence(vectors, height, width, 1e-8))


def test_lrc_seed_prefers_local_peak_over_uniform_high_background() -> None:
    y, x = np.mgrid[:37, :37]; grid = 0.08 + 0.001 * x + 0.0007 * y
    grid[4:15, 4:15] += 0.42  # broad high background: little local contrast
    grid[24:31, 24:31] += 0.55
    grid[26:29, 26:29] -= 0.03  # still high residual, but little local contrast
    score = torch.from_numpy(grid.astype(np.float32)).reshape(1, 37, 37)
    result = LocalResidualContrast().apply(score, torch.arange(411))
    assert not result.numerical_failure
    seed = result.maps["mask_seed"][0].numpy() > 0.5
    final = result.mask[0].numpy() > 0.5
    assert seed[24:31, 24:31].mean() > seed[4:15, 4:15].mean()
    assert final[26:29, 26:29].mean() > seed[26:29, 26:29].mean()
    assert final[4:15, 4:15].mean() < 0.5


def test_average_rank_ties_and_huto_determinism() -> None:
    values = np.asarray([0.0, 1.0, 1.0, 3.0])
    rank = empirical_rank(values)
    assert rank[1] == rank[2]
    assert np.array_equal(rank, empirical_rank(values))
    score = torch.linspace(0, 1, 37 * 37).reshape(1, 37, 37)
    first = HierarchicalUpperTailOtsu("med_h").apply(score, torch.arange(411))
    second = HierarchicalUpperTailOtsu("med_h").apply(score, torch.arange(411))
    assert torch.equal(first.mask, second.mask)
    assert first.threshold_high == second.threshold_high
    assert first.threshold_low == second.threshold_low
    assert first.diagnostics == second.diagnostics
    assert tuple(first.mask.shape) == (1, 37, 37)
