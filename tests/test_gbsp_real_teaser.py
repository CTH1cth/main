from pathlib import Path

import numpy as np
import torch

from models.gbsp_similarity_baselines import compute_similarity_baselines
from tools.teaser_real_data.common import (
    NUM_PATCHES,
    extract_scores,
    labels_from_occupancy,
    load_core_rows,
    load_settings,
    load_torch,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gbsp_real_teaser.yaml"


def test_frozen_settings_are_loaded_from_production_configs() -> None:
    settings = load_settings(CONFIG)
    assert settings.knn_k == 8
    assert settings.gbsp_rank == 8
    assert settings.hard_threshold == 0.58
    assert settings.main_dataset == "CAMO"


def test_core_and_allpatch_labels_follow_task_protocol() -> None:
    settings = load_settings(CONFIG)
    occupancy = np.linspace(0, 1, NUM_PATCHES, dtype=np.float32)
    core, binary, valid = labels_from_occupancy(occupancy, settings)
    assert np.all(core[occupancy <= 0.2] == 0)
    assert np.all(core[occupancy >= 0.8] == 1)
    assert np.all(core[(occupancy > 0.2) & (occupancy < 0.8)] == -1)
    assert np.array_equal(valid, core >= 0)
    assert np.array_equal(binary, occupancy >= 0.5)


def test_production_knn8_excludes_candidate_self_match() -> None:
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn(NUM_PATCHES, 16, generator=generator)
    background = torch.arange(20)
    result = compute_similarity_baselines(features, background, k=8)
    assert result.self_match_violation_count == 0
    assert result.scores["knn8_cos"].shape == (NUM_PATCHES,)
    assert torch.isfinite(result.scores["knn8_cos"]).all()


def test_one_real_core_cache_reproduces_official_minmax() -> None:
    settings = load_settings(CONFIG)
    row = next(row for row in load_core_rows(settings) if row["dataset"] == "CAMO")
    core = load_torch(row["cache_path"])
    scores = extract_scores(
        core, torch.device("cpu"), knn_k=settings.knn_k, gbsp_rank=settings.gbsp_rank,
    )
    assert scores["self_match_violation_count"] == 0
    assert scores["gbsp_minmax_max_abs_error"] <= 1e-6
    assert np.array_equal(scores["background_indices"], core["results"]["r8"]["background_indices"].numpy())
