from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from common.cache_dabe_pseudo import _expected_feature_shape, _load_feature
from common.dabe_pseudo import generate_dabe_pseudo
from common.eval_dinov2_r1 import _hard_mask, _resize_r1_to_gt
from common.utils import load_config


def _cfg(embed_dim: int, input_size: int, patch_size: int):
    return SimpleNamespace(
        DINO={
            "embed_dim": embed_dim,
            "feature_input_size": input_size,
            "patch_size": patch_size,
        }
    )


def _save_feature(tmp_path, tensor: torch.Tensor, dataset="D", stem="s"):
    path = tmp_path / f"{stem}.pt"
    torch.save({"dataset": dataset, "stem": stem, "tensor": tensor}, path)
    return {"cache_path": str(path)}


def test_expected_feature_shape_is_configuration_driven():
    assert _expected_feature_shape(_cfg(384, 296, 8)) == [384, 37, 37]
    assert _expected_feature_shape(_cfg(768, 518, 14)) == [768, 37, 37]


def test_expected_feature_shape_rejects_nondivisible_input():
    with pytest.raises(RuntimeError, match="not divisible"):
        _expected_feature_shape(_cfg(768, 517, 14))


@pytest.mark.parametrize("channels", [384, 768])
def test_load_feature_accepts_configured_channel_count(tmp_path, channels):
    row = _save_feature(tmp_path, torch.randn(channels, 37, 37))
    loaded = _load_feature(row, "D", "s", _cfg(channels, 37, 1))
    assert loaded.shape == (channels, 37, 37)
    assert loaded.dtype == torch.float32
    assert loaded.device.type == "cpu"
    assert loaded.is_contiguous()


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((767, 37, 37), "Expected feature"), ((768, 36, 37), "Expected feature")],
)
def test_load_feature_rejects_wrong_channel_or_grid(tmp_path, shape, expected):
    row = _save_feature(tmp_path, torch.randn(*shape))
    with pytest.raises(RuntimeError, match=expected):
        _load_feature(row, "D", "s", _cfg(768, 518, 14))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_load_feature_rejects_nonfinite(tmp_path, bad):
    tensor = torch.zeros(768, 37, 37)
    tensor[0, 0, 0] = bad
    row = _save_feature(tmp_path, tensor)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _load_feature(row, "D", "s", _cfg(768, 518, 14))


@pytest.mark.parametrize("channels", [384, 768])
def test_r1_is_channel_count_independent(tmp_path, channels):
    generator = torch.Generator().manual_seed(1000 + channels)
    feature = torch.randn(channels, 37, 37, generator=generator)
    image = (np.random.default_rng(42).random((80, 96, 3)) * 255).astype(np.uint8)
    image_path = tmp_path / f"image_{channels}.png"
    Image.fromarray(image).save(image_path)
    result = generate_dabe_pseudo(
        feature,
        str(image_path),
        params={"VERSION": "v2"},
        augs="identity",
    )
    r1 = result["residual_pass1_37"]
    assert r1.shape == (1, 37, 37)
    assert bool(torch.isfinite(r1).all())
    assert float(r1.min()) >= 0.0
    assert float(r1.max()) <= 1.0


def test_formal_resize_is_37_to_68_to_original_with_align_corners_false():
    source = torch.linspace(0.0, 1.0, 37 * 37).reshape(1, 37, 37)
    expected = F.interpolate(
        source.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )
    expected = F.interpolate(
        expected, size=(91, 117), mode="bilinear", align_corners=False
    ).squeeze(0)
    actual = _resize_r1_to_gt(source, (91, 117))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_hard_protocol_is_strict_greater_than_half():
    probability = torch.zeros(1, 37, 37)
    probability[0, 0, 0] = 0.5
    probability[0, 0, 1] = torch.nextafter(
        torch.tensor(0.5), torch.tensor(1.0)
    )
    hard = _hard_mask(probability, 0.5)
    assert not bool(hard[0, 0, 0])
    assert bool(hard[0, 0, 1])


def test_dinov2_r1_config_contract():
    cfg = load_config("configs/dinov2_b14_dabe_r1_direct.py")
    assert cfg.BACKBONE_KEY == "dinov2-b14"
    assert cfg.DINO["model_name"] == "facebook/dinov2-base"
    assert cfg.DINO["patch_size"] == 14
    assert cfg.DINO["feature_input_size"] == 518
    assert cfg.DINO["embed_dim"] == 768
    assert _expected_feature_shape(cfg) == [768, 37, 37]
