from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from common.eval_dinov2_final_patch_r1_pilot import (
    _local_distances,
    final_hidden_to_feature,
    select_all_samples,
    select_uniform_per_dataset,
)


def _cfg():
    return SimpleNamespace(
        DINO={"embed_dim": 4, "feature_input_size": 6, "patch_size": 2}
    )


def test_final_hidden_to_feature_removes_only_cls_and_preserves_patch_order():
    hidden = torch.arange(1 * 10 * 4, dtype=torch.float32).reshape(1, 10, 4)
    feature = final_hidden_to_feature(hidden, _cfg())
    assert feature.shape == (4, 3, 3)
    torch.testing.assert_close(feature[:, 0, 0], hidden[0, 1])
    torch.testing.assert_close(feature[:, 2, 2], hidden[0, 9])


@pytest.mark.parametrize("shape", [(1, 9, 4), (1, 11, 4), (1, 10, 5)])
def test_final_hidden_to_feature_rejects_wrong_contract(shape):
    with pytest.raises(RuntimeError, match="Expected final hidden state"):
        final_hidden_to_feature(torch.zeros(shape), _cfg())


def test_final_hidden_to_feature_rejects_nonfinite():
    hidden = torch.zeros(1, 10, 4)
    hidden[0, 1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        final_hidden_to_feature(hidden, _cfg())


def test_uniform_selection_is_balanced_and_round_robin_truncates():
    datasets = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
    rows = [
        {"dataset": dataset, "stem": f"{dataset}-{index}"}
        for dataset in datasets
        for index in range(10)
    ]
    selected = select_uniform_per_dataset(rows, per_dataset=3, max_samples=5)
    assert [row["dataset"] for row in selected] == [*datasets, "CHAMELEON"]
    assert len({(row["dataset"], row["stem"]) for row in selected}) == 5


def test_all_sample_selection_validates_official_counts():
    counts = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
    rows = [
        {"dataset": dataset, "stem": f"{dataset}-{index}"}
        for dataset, count in counts.items()
        for index in range(count)
    ]
    selected = select_all_samples(rows)
    assert len(selected) == 6473


def test_local_cosine_distance_is_zero_for_constant_direction():
    feature = torch.ones(8, 4, 4)
    all_edges, top_left = _local_distances(feature)
    assert float(abs(all_edges).max()) < 1e-6
    assert float(abs(top_left).max()) < 1e-6
