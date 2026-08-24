from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS
from models.gbsp_candidate_path_sweep import (
    build_candidate_path_variants,
    candidate_cache_version,
    candidate_variant_tag,
)
from models.gbsp_resolution import prepare_resolution_graph
from tools.cache_gbsp_candidate_path_sweep import (
    CACHE_ROOT,
    _cache_version,
    _validate_cache_root,
)
from common.utils import load_config


def test_registered_variant_names_and_cache_boundary():
    assert candidate_variant_tag(1, 25) == "bw1-p25"
    assert candidate_variant_tag(2, 27.5) == "bw2-p27p5"
    assert candidate_cache_version(2, 30, 32) == "gbsp_r32_pathcand_bw2_p30_v1"
    assert _cache_version(1, 30, 32, "sigmaf085") == (
        "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1"
    )
    with pytest.raises(ValueError):
        _cache_version(1, 30, 32, "bad-label")
    assert _validate_cache_root(CACHE_ROOT / "candidate-sweep") == CACHE_ROOT / "candidate-sweep"
    with pytest.raises(ValueError):
        _validate_cache_root(CACHE_ROOT)
    with pytest.raises(ValueError):
        _validate_cache_root(Path(__file__).resolve().parents[1] / "workdir/candidate-sweep")


def test_sigma_e_diagnostic_configs_change_only_registered_graph_scale():
    root = Path(__file__).resolve().parents[1]
    low = load_config(
        root / "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_sigmae024.py"
    )
    high = load_config(
        root / "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_sigmae0375.py"
    )
    assert low.DABE_SIGMA_E == 0.24
    assert high.DABE_SIGMA_E == 0.375
    assert low.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert high.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == (
        low.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET
    )


def test_six_variants_keep_dijkstra_and_control_path_participation(tmp_path):
    generator = torch.Generator().manual_seed(20260820)
    feature = torch.randn(40, 37, 37, generator=generator)
    image = (np.random.default_rng(20260820).random((111, 97, 3)) * 255).astype(np.uint8)
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(image).save(image_path)
    params = dict(DABE_V2_DEFAULT_PARAMS)
    prepared = prepare_resolution_graph(feature, str(image_path), params)
    results = build_candidate_path_variants(
        prepared,
        params,
        border_widths=(1, 2),
        top_percents=(25, 27.5, 30),
        pca_rank=32,
    )

    assert tuple(results) == (
        "bw1-p25",
        "bw1-p27p5",
        "bw1-p30",
        "bw2-p25",
        "bw2-p27p5",
        "bw2-p30",
    )
    expected_counts = {25.0: 343, 27.5: 377, 30.0: 411}
    expected_seeds = {1: 144, 2: 280}
    for result in results.values():
        assert result.background_indices.numel() == expected_counts[result.top_percent]
        assert result.boundary_seed_count == expected_seeds[result.border_width]
        assert result.boundary_candidate_count == result.boundary_seed_count
        assert result.interior_candidate_count == (
            expected_counts[result.top_percent] - expected_seeds[result.border_width]
        )
        assert result.interior_candidate_count > 0
        assert result.selected_rank == 32
        assert result.raw_residual.shape == (1, 37, 37)
        assert result.minmax_residual.shape == (1, 37, 37)
        assert torch.isfinite(result.minmax_residual).all()
        assert 0.0 <= float(result.minmax_residual.min())
        assert float(result.minmax_residual.max()) <= 1.0

    # With one border ring, a much larger part of the selected dictionary is
    # admitted by non-zero Dijkstra paths instead of being a zero-distance seed.
    assert (
        results["bw1-p30"].interior_candidate_ratio
        > results["bw2-p30"].interior_candidate_ratio
    )
    for width in (1, 2):
        small = set(results[f"bw{width}-p25"].background_indices.tolist())
        medium = set(results[f"bw{width}-p27p5"].background_indices.tolist())
        large = set(results[f"bw{width}-p30"].background_indices.tolist())
        assert small < medium < large
