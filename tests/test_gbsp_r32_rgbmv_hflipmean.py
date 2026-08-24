from pathlib import Path

import pytest
import torch

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.utils import load_config, torch_load
from tools.materialize_gbsp_r32_hflip_softmean_cache import (
    OUTPUT_VERSION,
    build_training_payload,
    fuse_identity_hflip,
    validate_output_root,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/dinov1_s8_gbsp_r32_rgbmv_hflipmean_lcic_ssboc.py"
)
LINEAR_CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/dinov1_s8_gbsp_r32_rgbmv_hflipmean_1x1_ssboc.py"
)


def _source() -> dict:
    scores = torch.zeros(6, 1, 37, 37)
    scores[0].fill_(0.2)
    scores[1].fill_(0.8)
    return {
        "dataset": "TR-CAMO",
        "stem": "sample",
        "image_path": "/home/dell01/CTH/sample.jpg",
        "backbone_key": "dinov1-s8",
        "version": "gbsp_r32_true_rgb_d4_multiview_v1",
        "settings_fingerprint": "test",
        "view_order": [
            "identity", "hflip", "vflip", "rot90", "rot180", "rot270"
        ],
        "aligned_scores_37": scores,
        "transformed_rgb_dino_forward_used": True,
        "feature_only_transform_used": False,
        "graph_path_used": True,
        "graph_path_method": "multi_source_dijkstra_neglog_affinity",
        "candidate_border_width": 2,
        "candidate_top_percent": 30.0,
        "pca_rank_mode": "fixed",
        "fixed_pca_rank": 32,
        "per_view_minmax_used": True,
        "gt_used_for_generation": False,
    }


def test_soft_mean_is_equal_view_mean_without_post_minmax():
    fused = fuse_identity_hflip(_source(), "/home/dell01/CTH/source.pt")
    assert tuple(fused.shape) == (1, 37, 37)
    assert torch.equal(fused, torch.full_like(fused, 0.5))


def test_training_payload_satisfies_fixed_r32_provenance():
    source_path = Path("/home/dell01/CTH/source.pt")
    payload = build_training_payload(_source(), source_path)
    assert payload["gbsp_version"] == OUTPUT_VERSION
    assert payload["source_augs"] == ["identity", "hflip"]
    assert payload["source_num_views"] == 2
    assert payload["selected_ranks"].tolist() == [32]
    assert payload["post_fusion_minmax_used"] is False
    assert payload["gt_used_for_generation"] is False


def test_output_must_remain_in_dataset_cache():
    with pytest.raises(ValueError):
        validate_output_root("/home/dell01/CTH/MY-baseline/workdir/bad-cache")


def test_training_config_is_strictly_audited():
    cfg = load_config(CONFIG_PATH)
    assert cfg.EXP_NAME == "36-gbsp-r32-rgbmv-hflipmean-lcic-ssboc"
    assert cfg.DABE_CLEAN_GBSP_AUGS == ["identity", "hflip"]
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.LCIC_SSBOC_VARIANT and cfg.GBSP_SSBOC_VARIANT
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"


def test_linear_training_config_changes_only_pseudo_source():
    cfg = load_config(LINEAR_CONFIG_PATH)
    assert cfg.EXP_NAME == "36-gbsp-r32-rgbmv-hflipmean-1x1-ssboc"
    assert cfg.HEAD_TYPE == "simple"
    assert not cfg.GBSP_LCIC_V1
    assert cfg.GBSP_SSBOC_VARIANT
    assert cfg.DABE_CLEAN_GBSP_AUGS == ["identity", "hflip"]
    assert cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET == {
        "TR-CAMO": 0.50,
        "TR-COD10K": 0.50,
    }
    assert cfg.LR == 0.0006
    assert cfg.SEED == 2027
    report = validate_dabev2hard_static_only_config(cfg)
    assert report["status"] == "PASS"
    assert report["contract"]["student"] == "single_1x1_conv_384_to_1"
    assert report["contract"]["pseudo_views"] == (
        "true_RGB_identity_hflip_inverse_aligned_soft_mean"
    )
