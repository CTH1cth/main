from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from tools.gbsp_teaser_analysis.common import load_settings
from tools.gbsp_teaser_max_similarity.collect import load_formal_inputs
from tools.gbsp_teaser_max_similarity.compute import brute_force_query, max_valid_background_similarity


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gbsp_teaser_max_similarity.yaml"


def test_frozen_settings_and_bins() -> None:
    settings = load_settings(CONFIG)
    assert settings.dataset == "CAMO"
    assert settings.gbsp_rank == 8
    assert np.array_equal(settings.similarity_bins, np.linspace(-1, 1, 81))
    assert np.array_equal(settings.residual_bins, np.linspace(0, 1, 51))
    assert settings.descriptive_thresholds == (.5, .6, .7, .8, .9)


def test_bg_self_match_is_excluded() -> None:
    feature = torch.nn.functional.normalize(torch.tensor([
        [1., 0.], [.99, .01], [0., 1.], [.1, .9],
    ]), dim=1)
    label = np.array([0, 0, 1, 1], dtype=np.uint8)
    valid = np.ones(4, dtype=bool)
    result = max_valid_background_similarity(feature, label, valid)
    assert result.self_match_violations == 0
    assert result.matched_index[0] == 1
    assert result.matched_index[1] == 0
    assert np.all(result.matched_index[:2] != np.arange(2))


def test_cosine_and_maximum_match_brute_force() -> None:
    generator = torch.Generator().manual_seed(2026)
    feature = torch.nn.functional.normalize(torch.randn(31, 12, generator=generator), dim=1)
    label = np.zeros(31, dtype=np.uint8); label[20:] = 1
    valid = np.ones(31, dtype=bool)
    result = max_valid_background_similarity(feature, label, valid)
    background = np.arange(20)
    for query in range(31):
        score, index = brute_force_query(feature, background, query)
        assert abs(float(result.score[query]) - score) < 1e-6
        selected_score = float(torch.dot(feature[query], feature[result.matched_index[query]]))
        assert abs(selected_score - score) < 1e-6
        if query < 20:
            assert result.matched_index[query] != query


def test_formal_gbsp_loader_cannot_accept_gt() -> None:
    parameters = inspect.signature(load_formal_inputs).parameters
    assert set(parameters) == {"core_path", "rank", "device"}
    assert not any("gt" in name or "occupancy" in name for name in parameters)
