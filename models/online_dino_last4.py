"""Batched frozen DINOv1-S/8 f9--f12 attention-key extraction."""

from __future__ import annotations

import math

import torch
from torch import nn

from common.cache_features import (
    call_dino,
    load_dino,
    resolve_key_projection_at_layer,
)


LAST4_INDICES_0BASED = (8, 9, 10, 11)
LAST4_KEYS = ("f9", "f10", "f11", "f12")


def _batched_key_to_feature(key: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(key) or key.ndim != 3:
        shape = list(key.shape) if torch.is_tensor(key) else None
        raise RuntimeError(f"Online DINO key must be [B,N,C], got {shape}.")
    batch, tokens, channels = map(int, key.shape)
    grid = int(math.sqrt(tokens - 1))
    if grid * grid != tokens - 1:
        raise RuntimeError(
            f"Online DINO patch-token count is not square: {tokens - 1}."
        )
    return (
        key[:, 1:, :]
        .reshape(batch, grid, grid, channels)
        .permute(0, 3, 1, 2)
        .contiguous()
        .detach()
        .float()
    )


class FrozenDINOv1Last4Extractor(nn.Module):
    """Extract last-four attention-key maps in one batched frozen forward."""

    def __init__(self, cfg):
        super().__init__()
        if str(cfg.BACKBONE_KEY) != "dinov1-s8":
            raise RuntimeError(
                "Online last-four extraction requires BACKBONE_KEY='dinov1-s8'."
            )
        if int(cfg.DINO["feature_input_size"]) != 296:
            raise RuntimeError("Online DINOv1-S/8 requires feature_input_size=296.")
        if int(cfg.DINO["patch_size"]) != 8:
            raise RuntimeError("Online DINOv1-S/8 requires patch_size=8.")

        self.model = load_dino(cfg, torch.device("cpu"))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self._holders = {key: None for key in LAST4_KEYS}
        self.key_paths = {}
        self._hook_handles = []

        for index, key in zip(LAST4_INDICES_0BASED, LAST4_KEYS):
            module, path = resolve_key_projection_at_layer(self.model, index)
            self.key_paths[key] = path
            self._hook_handles.append(
                module.register_forward_hook(self._make_hook(key))
            )

    def _make_hook(self, key):
        def hook(_module, _inputs, output):
            self._holders[key] = output

        return hook

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(inputs) or inputs.ndim != 4:
            shape = list(inputs.shape) if torch.is_tensor(inputs) else None
            raise RuntimeError(
                f"Online DINO input must be [B,3,296,296], got {shape}."
            )
        if list(inputs.shape[1:]) != [3, 296, 296]:
            raise RuntimeError(
                "Online DINO input shape mismatch: "
                f"{list(inputs.shape)} != [B,3,296,296]."
            )
        if inputs.dtype != torch.float32:
            raise RuntimeError(
                f"Online DINO input must be float32, got {inputs.dtype}."
            )
        for key in LAST4_KEYS:
            self._holders[key] = None

        # no_grad creates ordinary detached tensors that remain valid inputs to
        # trainable decoder convolutions.  inference_mode tensors cannot be
        # saved for the decoder weight backward pass.
        with torch.no_grad():
            call_dino(self.model, inputs)

        missing = [key for key in LAST4_KEYS if self._holders[key] is None]
        if missing:
            raise RuntimeError(f"Online DINO hooks did not capture: {missing}.")
        features = {
            key: _batched_key_to_feature(self._holders[key])
            for key in LAST4_KEYS
        }
        expected = [int(inputs.shape[0]), 384, 37, 37]
        for key, feature in features.items():
            if list(feature.shape) != expected:
                raise RuntimeError(
                    f"Online DINO {key} shape mismatch: "
                    f"{list(feature.shape)} != {expected}."
                )
        return features

