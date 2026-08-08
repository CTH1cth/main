from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix

from models.gbsp_threshold_v5 import (
    BackgroundConditionedMarkerPropagation,
    SeededMultiOtsuHysteresis,
    SeededPersistentComponentGrowth,
)


def _grid_graph(height: int, width: int, weak_cut: bool = False):
    count = height * width; neighbors = torch.zeros((count, 8), dtype=torch.long)
    valid = torch.zeros((count, 8), dtype=torch.bool); weights = torch.zeros((count, 8))
    for y in range(height):
        for x in range(width):
            source = y * width + x; slot = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dy, dx) == (0, 0): continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        target = ny * width + nx; neighbors[source, slot] = target; valid[source, slot] = True
                        cut = weak_cut and ((x < width // 2 <= nx) or (nx < width // 2 <= x))
                        weights[source, slot] = 1e-4 if cut else 1.0; slot += 1
    return neighbors, valid, weights


def test_smoh_recovers_seeded_support_and_rejects_unseeded_component() -> None:
    rng = np.random.default_rng(501); score = rng.normal(0.08, 0.01, (37, 37))
    score[19:37, 0:20] = rng.uniform(0.27, 0.33, (18, 20))
    score[5:18, 5:18] = rng.uniform(0.56, 0.68, (13, 13))
    score[9:13, 9:13] = rng.uniform(0.88, 0.98, (4, 4))
    score[27:33, 27:33] = rng.uniform(0.57, 0.63, (6, 6))
    result = SeededMultiOtsuHysteresis().apply(torch.tensor(score, dtype=torch.float32)[None], torch.arange(200))
    assert not result.numerical_failure
    seed = result.maps["seed_mask"][0].bool(); support = result.maps["support_mask"][0].bool(); final = result.mask[0].bool()
    assert torch.all(final[seed])
    assert final[5:18, 5:18].sum() == support[5:18, 5:18].sum()
    assert final[27:33, 27:33].sum() == 0
    assert result.diagnostics["num_retained_components"] < result.diagnostics["num_support_components"]


def test_spcg_smooth_growth_uses_full_support() -> None:
    detector = SeededPersistentComponentGrowth()
    result = detector.detect_jump([10, 12, 14, 17, 20, 23, 26])
    assert result["jump_index"] is None


def test_spcg_abrupt_growth_stops_before_20_to_45() -> None:
    detector = SeededPersistentComponentGrowth()
    result = detector.detect_jump([10, 12, 14, 17, 20, 45, 70])
    assert result["jump_index"] == 5
    assert result["jump_value"] > result["jump_baseline"]


def test_spcg_components_can_receive_different_stop_decisions() -> None:
    detector = SeededPersistentComponentGrowth()
    smooth = detector.detect_jump([10, 12, 14, 17, 20, 23, 26])
    abrupt = detector.detect_jump([8, 10, 12, 15, 18, 42, 55])
    assert smooth["jump_index"] is None and abrupt["jump_index"] == 5


def test_bcmp_linear_chain_is_harmonic_and_monotone() -> None:
    rows = np.array([0, 1, 1, 2, 2, 3]); cols = np.array([1, 0, 2, 1, 3, 2])
    adjacency = coo_matrix((np.ones(6), (rows, cols)), shape=(4, 4)).tocsr()
    fg = np.array([False, False, False, True]); bg = np.array([True, False, False, False])
    probability, status, residual = BackgroundConditionedMarkerPropagation.harmonic_probability(adjacency, fg, bg)
    assert status == "solved" and residual < 1e-10
    assert np.all(np.diff(probability) > 0) and np.all((probability[1:3] > 0) & (probability[1:3] < 1))
    assert np.array_equal(probability > 0.5, np.array([False, False, True, True]))


def test_bcmp_strong_edge_blocks_propagation() -> None:
    # Two unknown nodes see the same foreground marker; the second has a weak
    # foreground edge and a strong background edge.
    rows=np.array([0,1,0,2,3,1,3,2]); cols=np.array([1,0,2,0,1,3,2,3])
    weights=np.array([1,1,1e-3,1e-3,1e-3,1e-3,1,1],dtype=float)
    adjacency=coo_matrix((weights,(rows,cols)),shape=(4,4)).tocsr()
    fg=np.array([True,False,False,False]); bg=np.array([False,False,False,True])
    probability,_,_=BackgroundConditionedMarkerPropagation.harmonic_probability(adjacency,fg,bg)
    assert probability[1] > 0.9 and probability[2] < 0.1


def test_bcmp_does_not_hard_lock_high_residual_bc_and_is_deterministic() -> None:
    score=torch.linspace(0,1,37*37).reshape(1,37,37)
    high_index=torch.tensor([1368],dtype=torch.long)
    boundary=[]
    for y in range(37):
        for x in range(37):
            if y in (0,36) or x in (0,36): boundary.append(y*37+x)
    neighbors,valid,weights=_grid_graph(37,37)
    method=BackgroundConditionedMarkerPropagation()
    first=method.apply(score,high_index,torch.tensor(boundary),neighbors,valid,weights)
    second=method.apply(score,high_index,torch.tensor(boundary),neighbors,valid,weights)
    assert not first.numerical_failure
    assert first.maps["seed_mask_fg"].reshape(-1)[1368] == 1
    assert first.mask.reshape(-1)[1368] == 1
    assert torch.equal(first.mask,second.mask)
    assert torch.equal(first.maps["probability_map"],second.maps["probability_map"])
    assert first.diagnostics == second.diagnostics
