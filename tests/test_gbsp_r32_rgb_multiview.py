from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from models.gbsp_rgb_multiview import (
    METHOD_ORDER,
    VIEW_ORDER,
    apply_rgb_view,
    build_method_probabilities_68,
    inverse_align_tensor,
    key_batch_to_feature_tensor,
    stack_aligned_scores,
    unpack_aligned_scores,
)
from tools.compare_gbsp_r32_rgb_multiview import (
    CACHE_ROOT,
    _settings_fingerprint,
    _validate_output_root,
)


def _score_map(value: float) -> torch.Tensor:
    return torch.full((1, 37, 37), float(value), dtype=torch.float32)


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_rgb_view_inverse_alignment_roundtrip(view):
    array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image = Image.fromarray(array, mode="RGB")
    transformed = np.asarray(apply_rgb_view(image, view))
    tensor = torch.from_numpy(np.array(transformed, copy=True)).permute(2, 0, 1)
    restored = inverse_align_tensor(tensor, view)
    assert torch.equal(restored, torch.from_numpy(array).permute(2, 0, 1))


def test_six_view_fusions_and_strict_tie_rule():
    scores = {
        "identity": _score_map(0.9),
        "hflip": _score_map(0.7),
        "vflip": _score_map(0.8),
        "rot90": _score_map(0.6),
        "rot180": _score_map(0.1),
        "rot270": _score_map(0.2),
    }
    output = build_method_probabilities_68(scores, threshold=0.5)
    assert tuple(output) == METHOD_ORDER
    assert torch.allclose(output["identity"], _score_map(0.9).new_full((1, 68, 68), 0.9))
    assert torch.allclose(
        output["id_hflip_soft_mean"],
        _score_map(0.8).new_full((1, 68, 68), 0.8),
    )
    assert torch.allclose(
        output["six_view_soft_mean"],
        _score_map(0.55).new_full((1, 68, 68), 0.55),
    )
    assert float(output["six_view_hard_majority"].max()) == 1.0

    tied = dict(scores)
    tied["rot90"] = _score_map(0.4)
    tied_output = build_method_probabilities_68(tied, threshold=0.5)
    assert float(tied_output["six_view_hard_majority"].max()) == 0.0


def test_stack_unpack_is_ordered_and_lossless():
    scores = {view: _score_map(index / 10.0) for index, view in enumerate(VIEW_ORDER)}
    stacked = stack_aligned_scores(scores)
    restored = unpack_aligned_scores(stacked)
    assert tuple(restored) == VIEW_ORDER
    for view in VIEW_ORDER:
        assert torch.equal(restored[view], scores[view])


def test_batched_key_conversion():
    key = torch.arange(2 * (37 * 37 + 1) * 4, dtype=torch.float32).reshape(
        2, 37 * 37 + 1, 4
    )
    feature = key_batch_to_feature_tensor(key)
    assert tuple(feature.shape) == (2, 4, 37, 37)
    assert torch.equal(feature[0, :, 0, 0], key[0, 1])


def test_output_root_cannot_escape_cache():
    accepted = _validate_output_root(CACHE_ROOT / "unit-test/multiview")
    assert CACHE_ROOT in accepted.parents
    with pytest.raises(ValueError):
        _validate_output_root(CACHE_ROOT.parent / "outside-cache")


def test_fingerprint_changes_with_threshold():
    class Config:
        BACKBONE_KEY = "dinov1-s8"
        DINO = {
            "model_path": CACHE_ROOT / "fake-local-model",
            "feature_input_size": 296,
            "patch_size": 8,
            "embed_dim": 384,
        }

    params = {
        "SIGMA_F": 0.1,
        "SIGMA_C": 0.05,
        "SIGMA_E": 0.3,
        "TAU_BC": 0.3,
        "BORDER_WIDTH": 2,
        "BG_ANCHOR_TOP_PERCENT": 30.0,
        "BG_ANCHOR_MIN_RATIO": 0.05,
        "BG_ANCHOR_FALLBACK_TOP_PERCENT": 40.0,
    }
    assert _settings_fingerprint(Config, params, 0.5) != _settings_fingerprint(
        Config, params, 0.58
    )
