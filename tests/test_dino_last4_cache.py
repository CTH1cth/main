from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from common.cache_features import load_dino, preprocess_image
from common.utils import build_image_items, load_config
from tools.cache_dinov1_last4 import (
    LAST4_KEYS,
    _manifest_map,
    _reference_f12,
    _validated_resume_row,
    capture_last4,
    final_feature_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dinov1_s8_r1hard_hsd_v1_sem_148.py"


class _Attention(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def forward(self, inputs):
        return inputs + self.offset


class _Layer(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.attention = nn.Module()
        self.attention.attention = nn.Module()
        self.attention.attention.key = _Attention(offset)


class _FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Layer(index) for index in range(12)])

    def forward(self, inputs):
        tokens = torch.zeros(inputs.shape[0], 37 * 37 + 1, 384, device=inputs.device)
        for layer in self.encoder.layer:
            layer.attention.attention.key(tokens)
        return tokens


def test_capture_last4_cache_keys_shapes_and_values():
    features = capture_last4(_FakeDino(), torch.zeros(1, 3, 296, 296))
    assert tuple(features) == LAST4_KEYS
    for key in LAST4_KEYS:
        assert features[key].shape == (384, 37, 37)
        assert features[key].dtype == torch.float32
        assert features[key].requires_grad is False
    assert float(features["f9"].mean()) == 8.0
    assert float(features["f12"].mean()) == 11.0


def test_resume_validation_accepts_complete_payload_and_rejects_partial_payload():
    features = {
        key: torch.full((384, 37, 37), float(index), dtype=torch.float32)
        for index, key in enumerate(LAST4_KEYS)
    }
    item = {
        "dataset": "TE-CAMO",
        "stem": "sample",
        "image_path": "/home/dell01/CTH/MY-baseline/datasets/COD/sample.jpg",
    }
    cfg = SimpleNamespace(
        BACKBONE_KEY="dinov1-s8",
        DINO={"model_name": "facebook/dino-vits8"},
    )
    payload = {
        "version": "dinov1_s8_last4_key_v1",
        "dataset": item["dataset"],
        "stem": item["stem"],
        "backbone_key": cfg.BACKBONE_KEY,
        "model_key": cfg.DINO["model_name"],
        "input_size": 296,
        "patch_size": 8,
        "resize_interpolation": "bicubic",
        "feature_type": "attention_key_projection",
        "layer_indices_0based": [8, 9, 10, 11],
        "feature_keys": list(LAST4_KEYS),
        "dtype": "float32",
        "features": features,
        "tensor": features["f12"],
        "f12_max_abs_error": 0.0,
    }
    row = _validated_resume_row(
        payload,
        Path("/home/dell01/CTH/MY-baseline/workdir/sample.pt"),
        item,
        cfg,
    )
    assert row["feature_keys"] == list(LAST4_KEYS)
    assert row["shape"]["f12"] == [384, 37, 37]

    partial = dict(payload)
    partial["features"] = dict(features)
    del partial["features"]["f11"]
    with pytest.raises(RuntimeError, match="f11"):
        _validated_resume_row(
            partial,
            Path("/home/dell01/CTH/MY-baseline/workdir/sample.pt"),
            item,
            cfg,
        )


def test_actual_dinov1_f12_matches_existing_final_cache():
    cfg = load_config(CONFIG)
    model_path = (ROOT / cfg.DINO["model_path"]).resolve()
    manifest = final_feature_manifest(cfg, "train")
    if not model_path.is_dir() or not manifest.is_file():
        pytest.skip("Local DINOv1-S/8 weights or existing final cache are unavailable.")

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    item = items[0]
    reference_map = _manifest_map(manifest)
    reference = _reference_f12(
        reference_map[(item["dataset"], item["stem"])],
        item["dataset"],
        item["stem"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    model.eval()
    inputs, _ = preprocess_image(item["image_path"], 296, interpolation="bicubic")
    features = capture_last4(model, inputs.to(device))
    assert all(features[key].shape == (384, 37, 37) for key in LAST4_KEYS)
    error = float((features["f12"] - reference).abs().max().item())
    assert error < 1e-6
