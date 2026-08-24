import numpy as np
import torch

from models.gbsp_similarity_baselines import compute_similarity_baselines
from models.reference_contamination import (
    controlled_clustered_candidate_set,
    controlled_candidate_set,
    local_wrong_reference_counts,
    matched_size_clean_candidate_set,
    rounded_contamination_count,
    score_gbsp_fixed_rank,
    score_knn8,
)


def _features(channels: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260808)
    return torch.randn(1369, channels, generator=generator)


def test_knn8_and_fixed_rank_gbsp_are_finite_and_complete():
    feature = _features()
    candidate = torch.arange(80)
    knn = score_knn8(feature, candidate)
    gbsp = score_gbsp_fixed_rank(feature, candidate, rank=8)
    assert knn.score.shape == (1369,)
    assert knn.neighbor_indices.shape == (1369, 8)
    assert knn.self_match_violation_count == 0
    assert torch.isfinite(knn.score).all()
    assert gbsp.score.shape == (1369,) and torch.isfinite(gbsp.score).all()
    assert gbsp.selected_rank == 8 and gbsp.orthonormal_error < 1e-4


def test_knn8_exactly_reuses_the_frozen_baseline_definition():
    feature = _features()
    candidate = torch.arange(80)
    current = score_knn8(feature, candidate)
    frozen = compute_similarity_baselines(feature, candidate, k=8)
    assert torch.allclose(current.score, frozen.scores["knn8_cos"], atol=0.0, rtol=0.0)


def test_local_wrong_count_matches_neighbor_labels():
    neighbors = torch.arange(8).repeat(1369, 1)
    labels = np.zeros(1369, dtype=np.uint8)
    labels[[1, 3, 7]] = 1
    count = local_wrong_reference_counts(neighbors, labels)
    assert count.dtype == torch.uint8
    assert count.shape == (1369,)
    assert torch.equal(count, torch.full((1369,), 3, dtype=torch.uint8))


def test_controlled_contamination_is_fixed_size_clean_and_nested():
    labels = np.zeros(1369, dtype=np.uint8)
    labels[500:700] = 1
    clean = torch.arange(100)
    one = controlled_candidate_set(clean, labels, fraction=.01, seed=2, identity="CAMO/x")
    three = controlled_candidate_set(clean, labels, fraction=.03, seed=2, identity="CAMO/x")
    repeat = controlled_candidate_set(clean, labels, fraction=.03, seed=2, identity="CAMO/x")
    assert one.valid and three.valid
    assert one.indices.numel() == clean.numel() == three.indices.numel()
    assert one.injected_foreground_indices.numel() == 1
    assert three.injected_foreground_indices.numel() == 3
    assert torch.isin(one.injected_foreground_indices, three.injected_foreground_indices).all()
    assert torch.isin(one.removed_background_indices, three.removed_background_indices).all()
    assert torch.equal(three.indices, repeat.indices)
    assert int(labels[three.indices].sum()) == 3


def test_controlled_contamination_reports_invalid_when_foreground_is_insufficient():
    labels = np.zeros(1369, dtype=np.uint8)
    labels[500] = 1
    result = controlled_candidate_set(
        torch.arange(100), labels, fraction=.05, seed=0, identity="COD10K/x"
    )
    assert not result.valid
    assert result.indices.numel() == 100
    assert "needs 5 FG patches" in result.reason


def test_matched_size_clean_removes_fg_and_replenishes_true_bg():
    labels = np.zeros(1369, dtype=np.uint8)
    labels[[3, 8, 11]] = 1
    natural = torch.arange(100)
    result = matched_size_clean_candidate_set(
        natural, labels, seed=4, identity="NC4K/x"
    )
    assert result.valid and result.indices.numel() == natural.numel()
    assert not labels[result.indices].any()
    assert set(result.removed_background_indices.tolist()) == {3, 8, 11}
    assert result.injected_foreground_indices.numel() == 3
    assert not torch.isin(result.injected_foreground_indices, natural).any()


def test_matched_random_contamination_is_nested_and_uses_the_same_dictionary_size():
    labels = np.zeros(1369, dtype=np.uint8)
    labels[500:800] = 1
    natural = torch.arange(411)
    labels[[3, 8, 11]] = 1
    matched = matched_size_clean_candidate_set(
        natural, labels, seed=1, identity="CAMO/random-matched"
    )
    assert matched.valid and matched.indices.numel() == natural.numel()
    five = controlled_candidate_set(
        matched.indices, labels, fraction=.05, seed=1,
        identity="CAMO/random-matched",
    )
    ten = controlled_candidate_set(
        matched.indices, labels, fraction=.10, seed=1,
        identity="CAMO/random-matched",
    )
    twenty = controlled_candidate_set(
        matched.indices, labels, fraction=.20, seed=1,
        identity="CAMO/random-matched",
    )
    assert five.valid and ten.valid and twenty.valid
    assert five.indices.numel() == ten.indices.numel() == twenty.indices.numel() == 411
    assert torch.isin(five.injected_foreground_indices, ten.injected_foreground_indices).all()
    assert torch.isin(ten.injected_foreground_indices, twenty.injected_foreground_indices).all()
    assert torch.isin(five.removed_background_indices, ten.removed_background_indices).all()
    assert torch.isin(ten.removed_background_indices, twenty.removed_background_indices).all()
    assert int(labels[five.indices].sum()) == 21
    assert int(labels[ten.indices].sum()) == 41
    assert int(labels[twenty.indices].sum()) == 82


def test_random_and_clustered_contamination_can_share_background_removal():
    labels = np.zeros((37, 37), dtype=np.uint8)
    labels[12:24, 13:27] = 1
    labels = labels.reshape(-1)
    clean = torch.arange(300)
    random = controlled_candidate_set(
        clean,
        labels,
        fraction=.05,
        seed=1,
        identity="CAMO/shared-removal",
        removal_namespace="remove_bg_cluster",
    )
    clustered = controlled_clustered_candidate_set(
        clean,
        labels,
        fraction=.05,
        seed=1,
        identity="CAMO/shared-removal",
    )
    assert random.valid and clustered.valid
    assert torch.equal(
        random.removed_background_indices,
        clustered.removed_background_indices,
    )


def test_round_half_up_is_explicit():
    assert rounded_contamination_count(.005, 100) == 1
    assert rounded_contamination_count(.005, 411) == 2
    assert rounded_contamination_count(.05, 411) == 21


def _connected(indices: torch.Tensor) -> bool:
    selected = set(torch.as_tensor(indices).reshape(-1).tolist())
    if len(selected) <= 1:
        return True
    reached = {next(iter(selected))}
    queue = list(reached)
    for patch in queue:
        row, column = divmod(patch, 37)
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == delta_column == 0:
                    continue
                neighbor_row, neighbor_column = row + delta_row, column + delta_column
                if 0 <= neighbor_row < 37 and 0 <= neighbor_column < 37:
                    neighbor = neighbor_row * 37 + neighbor_column
                    if neighbor in selected and neighbor not in reached:
                        reached.add(neighbor)
                        queue.append(neighbor)
    return reached == selected


def test_clustered_contamination_is_connected_nested_and_fixed_size():
    labels = np.zeros((37, 37), dtype=np.uint8)
    labels[12:24, 13:27] = 1
    labels = labels.reshape(-1)
    clean = torch.arange(300)
    five = controlled_clustered_candidate_set(
        clean, labels, fraction=.05, seed=1, identity="CAMO/cluster"
    )
    ten = controlled_clustered_candidate_set(
        clean, labels, fraction=.10, seed=1, identity="CAMO/cluster"
    )
    repeat = controlled_clustered_candidate_set(
        clean, labels, fraction=.10, seed=1, identity="CAMO/cluster"
    )
    assert five.valid and ten.valid
    assert five.injection_mode == ten.injection_mode == "spatial_cluster_8n"
    assert five.indices.numel() == clean.numel() == ten.indices.numel()
    assert five.injected_connected and ten.injected_connected
    assert _connected(five.injected_foreground_indices)
    assert _connected(ten.injected_foreground_indices)
    assert torch.isin(five.injected_foreground_indices, ten.injected_foreground_indices).all()
    assert torch.isin(five.removed_background_indices, ten.removed_background_indices).all()
    assert torch.equal(ten.indices, repeat.indices)


def test_clustered_contamination_rejects_an_insufficient_connected_component():
    labels = np.zeros((37, 37), dtype=np.uint8)
    labels[10:12, 10:12] = 1
    result = controlled_clustered_candidate_set(
        torch.arange(100), labels.reshape(-1),
        fraction=.05, seed=0, identity="COD10K/small",
    )
    assert not result.valid
    assert result.selected_component_size == 4
    assert "connected FG cluster of 5" in result.reason
