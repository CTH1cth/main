from pathlib import Path

import torch

from common.cache_dabe_pseudo import _params_from_cfg
from common.cache_features import key_to_feature_tensor
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS
from common.utils import build_image_items, load_config
from models.gbsp_resolution import (
    fit_gbsp_from_prepared,
    hard_mask_at_native,
    prepare_resolution_graph,
)


MAIN = Path(__file__).resolve().parents[1]


def test_key_projection_4096_patch_tokens_becomes_native_64() -> None:
    key = torch.randn(1, 4097, 384)
    feature = key_to_feature_tensor(key)
    assert tuple(feature.shape) == (384, 64, 64)


def test_resolution_configs_freeze_everything_except_grid_and_border() -> None:
    cfg296 = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_296_reference.py")
    cfg_a = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py")
    cfg_b = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw3.py")
    cfg_c = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r12.py")
    assert (cfg296.DINO["feature_input_size"], cfg296.GRID_SIZE, cfg296.GBSP_BORDER_WIDTH) == (296, 37, 2)
    assert (cfg_a.DINO["feature_input_size"], cfg_a.GRID_SIZE, cfg_a.GBSP_BORDER_WIDTH) == (512, 64, 2)
    assert (cfg_b.DINO["feature_input_size"], cfg_b.GRID_SIZE, cfg_b.GBSP_BORDER_WIDTH) == (512, 64, 3)
    assert (cfg_c.DINO["feature_input_size"], cfg_c.GRID_SIZE, cfg_c.GBSP_BORDER_WIDTH) == (512, 64, 2)
    assert cfg296.USE_LEGACY_68_INTERPOLATION is True
    assert cfg_a.USE_LEGACY_68_INTERPOLATION is False
    assert cfg_b.USE_LEGACY_68_INTERPOLATION is False
    assert cfg_c.USE_LEGACY_68_INTERPOLATION is False
    assert (
        cfg296.GBSP_PCA_MAX_RANK,
        cfg_a.GBSP_PCA_MAX_RANK,
        cfg_b.GBSP_PCA_MAX_RANK,
        cfg_c.GBSP_PCA_MAX_RANK,
    ) == (8, 8, 8, 12)
    frozen = (
        "GBSP_BG_RATIO", "GBSP_PCA_ENERGY",
        "GBSP_PCA_MIN_RANK", "GBSP_THRESHOLD", "DABE_SIGMA_F",
        "DABE_SIGMA_C", "DABE_SIGMA_E", "DABE_TAU_BC",
    )
    for field in frozen:
        assert (
            getattr(cfg296, field)
            == getattr(cfg_a, field)
            == getattr(cfg_b, field)
            == getattr(cfg_c, field)
        )


def test_dynamic_graph_pca_and_native_threshold_have_no_fixed_grid() -> None:
    cfg = load_config(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py")
    image = build_image_items(cfg.DATA_ROOT, ["TE-CAMO"], require_gt=False)[0]["image_path"]
    generator = torch.Generator().manual_seed(7)
    feature = torch.randn(384, 9, 9, generator=generator)
    params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg), "GRID": 9, "BORDER_WIDTH": 2}
    prepared = prepare_resolution_graph(feature, image, params)
    assert (prepared.grid_h, prepared.grid_w, prepared.feature_dim) == (9, 9, 384)
    assert tuple(prepared.rgb.shape) == (3, 9, 9)
    assert tuple(prepared.sobel.shape) == (9, 9)
    assert set(prepared.graph_terms) == {
        "semantic_distance", "rgb_distance", "sobel_term", "graph_affinity"
    }
    result = fit_gbsp_from_prepared(
        prepared,
        params,
        pca_energy=.90,
        pca_min_rank=1,
        pca_max_rank=8,
    )
    assert tuple(result.minmax_residual.shape) == (1, 9, 9)
    assert tuple(result.background_anchor.shape) == (1, 9, 9)
    assert 0 < result.background_indices.numel() <= 81
    assert 1 <= result.selected_rank <= 8
    native, original = hard_mask_at_native(result.minmax_residual, .58, (31, 47))
    assert tuple(native.shape) == (1, 9, 9)
    assert tuple(original.shape) == (1, 31, 47)
    legacy, _ = hard_mask_at_native(
        result.minmax_residual, .58, (31, 47), legacy_intermediate_size=11
    )
    assert tuple(legacy.shape) == (1, 11, 11)
