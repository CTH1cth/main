"""R1-HSD v1 hierarchical semantic/detail decoder heads.

These heads consume frozen DINOv1-S/8 attention-key maps.  They deliberately
contain no backbone, Teacher, EMA, pseudo-label generation, or GT-dependent
logic; supervision remains the responsibility of the training entry point.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


LAST4_KEYS = ("f9", "f10", "f11", "f12")


def _validate_last4(features, in_channels: int) -> dict[str, torch.Tensor]:
    if not isinstance(features, Mapping):
        raise TypeError(
            "R1-HSD last-four input must be a mapping with keys "
            f"{list(LAST4_KEYS)}, got {type(features).__name__}."
        )
    missing = [key for key in LAST4_KEYS if key not in features]
    if missing:
        raise KeyError(f"R1-HSD last-four input is missing keys: {missing}")

    checked = {}
    reference_shape = None
    for key in LAST4_KEYS:
        value = features[key]
        if not torch.is_tensor(value) or value.ndim != 4:
            shape = list(value.shape) if torch.is_tensor(value) else None
            raise RuntimeError(f"R1-HSD {key} must be [B,C,H,W], got {shape}.")
        if int(value.shape[1]) != int(in_channels):
            raise RuntimeError(
                f"R1-HSD {key} channel mismatch: {int(value.shape[1])} "
                f"!= {int(in_channels)}."
            )
        if tuple(value.shape[-2:]) != (37, 37):
            raise RuntimeError(
                f"R1-HSD {key} spatial shape must be 37x37, got "
                f"{list(value.shape[-2:])}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"R1-HSD {key} contains NaN/Inf.")
        shape = (int(value.shape[0]), *map(int, value.shape[-2:]))
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise RuntimeError(
                f"R1-HSD last-four batch/spatial mismatch: {key} has {shape}, "
                f"expected {reference_shape}."
            )
        checked[key] = value
    return checked


def _group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class Last4LinearProbe(nn.Module):
    """Exactly one trainable 1x1 convolution over concatenated f9--f12."""

    def __init__(self, in_channels: int = 384):
        super().__init__()
        self.in_channels = int(in_channels)
        self.proj = nn.Conv2d(4 * self.in_channels, 1, kernel_size=1)

    def forward(self, features):
        features = _validate_last4(features, self.in_channels)
        return self.proj(torch.cat([features[key] for key in LAST4_KEYS], dim=1))

    def estimated_conv_macs_per_image(self) -> int:
        return 37 * 37 * (4 * self.in_channels)


class F12ScaleLiftHead(nn.Module):
    """F12-only 37->74->148 scale-lift isolation decoder."""

    def __init__(
        self,
        in_channels: int = 384,
        channels: int = 64,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError(f"Scale-Lift channels must be positive, got {channels}.")

        # Deliberately no GN/activation here: the requested isolation path is
        # exactly one initial 1x1 projection followed by the two Scale-Lifts.
        self.feature_projection = nn.Conv2d(
            self.in_channels, self.channels, kernel_size=1
        )
        self.base_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.lift_37_to_74 = _ScaleLift(self.channels, gn_groups)
        self.coarse_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.lift_74_to_148 = _ScaleLift(self.channels, gn_groups)
        self.final_head = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(self, features):
        if not isinstance(features, Mapping):
            raise TypeError(
                "F12 Scale-Lift input must be a mapping containing only the "
                "required f12 tensor."
            )
        if "f12" not in features:
            raise KeyError("F12 Scale-Lift input is missing f12.")
        f12 = features["f12"]
        if not torch.is_tensor(f12) or f12.ndim != 4:
            shape = list(f12.shape) if torch.is_tensor(f12) else None
            raise RuntimeError(f"F12 Scale-Lift f12 must be [B,C,37,37], got {shape}.")
        if list(f12.shape[1:]) != [self.in_channels, 37, 37]:
            raise RuntimeError(
                "F12 Scale-Lift f12 shape mismatch: "
                f"{list(f12.shape)} != [B,{self.in_channels},37,37]."
            )
        if not bool(torch.isfinite(f12).all().item()):
            raise RuntimeError("F12 Scale-Lift f12 contains NaN/Inf.")

        feature_37 = self.feature_projection(f12)
        base_logits_37 = self.base_head(feature_37)
        feature_74 = self.lift_37_to_74(feature_37)
        coarse_logits_74 = self.coarse_head(feature_74)
        feature_148 = self.lift_74_to_148(feature_74)
        final_logits_148 = self.final_head(feature_148)
        return {
            "logits": final_logits_148,
            "final_logits": final_logits_148,
            "coarse_logits": coarse_logits_74,
            "coarse_logits_74": coarse_logits_74,
            "base_logits": base_logits_37,
            "base_logits_37": base_logits_37,
            "feature_37": feature_37,
            "feature_74": feature_74,
            "feature_148": feature_148,
        }

    def estimated_conv_macs_per_image(self) -> int:
        c = self.channels
        n37, n74, n148 = 37 * 37, 74 * 74, 148 * 148
        macs = n37 * self.in_channels * c
        macs += n37 * c
        macs += n74 * (9 * c + c * c)
        macs += n74 * c
        macs += n148 * (9 * c + c * c)
        macs += n148 * c
        return int(macs)


class _Adapter(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, gn_groups: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            _group_norm(out_channels, gn_groups),
            nn.GELU(),
        )


class _HierarchicalGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # [U_l, H_next, |U_l-H_next|, cosine agreement].
        self.gate = nn.Conv2d(3 * channels + 1, 1, kernel_size=1)
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )

    def forward(self, current: torch.Tensor, deeper: torch.Tensor):
        cosine = F.cosine_similarity(current, deeper, dim=1, eps=1e-6).unsqueeze(1)
        agreement = (1.0 + cosine).mul(0.5).clamp(0.0, 1.0)
        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [current, deeper, torch.abs(current - deeper), agreement],
                    dim=1,
                )
            )
        )
        return deeper + gate * self.local(current), gate, agreement


class _ScaleLift(nn.Sequential):
    def __init__(self, channels: int, gn_groups: int):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            _group_norm(channels, gn_groups),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = F.interpolate(
            feature, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return super().forward(feature)


class _DetailEncoder(nn.Sequential):
    def __init__(self, out_channels: int, gn_groups: int):
        super().__init__(
            nn.Conv2d(4, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _group_norm(out_channels, gn_groups),
            nn.GELU(),
        )


def _normalized_sobel(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or int(image.shape[1]) != 3:
        raise RuntimeError(f"HSD detail RGB must be [B,3,H,W], got {list(image.shape)}.")
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    sobel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = image.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-6)
    maximum = magnitude.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    return (magnitude / (maximum + 1e-6)).clamp(0.0, 1.0)


class HSDV1Head(nn.Module):
    """Hierarchical semantic decoder with an optional uncertainty detail path."""

    def __init__(
        self,
        in_channels: int = 384,
        semantic_channels: int = 64,
        detail_channels: int = 32,
        gn_groups: int = 8,
        use_detail: bool = False,
        output_size: int = 148,
        coarse_size: int = 37,
    ):
        super().__init__()
        if int(output_size) != 148:
            raise ValueError(f"R1-HSD v1 requires output_size=148, got {output_size}.")
        self.in_channels = int(in_channels)
        self.semantic_channels = int(semantic_channels)
        self.detail_channels = int(detail_channels)
        self.use_detail = bool(use_detail)
        self.output_size = int(output_size)
        self.coarse_size = int(coarse_size)
        if self.coarse_size not in {37, 74}:
            raise ValueError(
                f"R1-HSD coarse_size must be 37 or 74, got {coarse_size}."
            )

        self.adapters = nn.ModuleDict(
            {
                key: _Adapter(self.in_channels, self.semantic_channels, gn_groups)
                for key in LAST4_KEYS
            }
        )
        self.cross_gates = nn.ModuleDict(
            {
                key: _HierarchicalGate(self.semantic_channels)
                for key in ("f11", "f10", "f9")
            }
        )
        self.lift_37_to_74 = _ScaleLift(self.semantic_channels, gn_groups)
        self.lift_74_to_148 = _ScaleLift(self.semantic_channels, gn_groups)

        self.base_head = nn.Conv2d(self.semantic_channels, 1, kernel_size=1)
        self.coarse_head = nn.Conv2d(self.semantic_channels, 1, kernel_size=1)
        self.final_head = nn.Conv2d(self.semantic_channels, 1, kernel_size=1)

        if self.use_detail:
            self.detail_encoder = _DetailEncoder(self.detail_channels, gn_groups)
            self.detail_gate = nn.Conv2d(
                self.semantic_channels + self.detail_channels + 2,
                1,
                kernel_size=1,
            )
            self.detail_to_semantic = nn.Conv2d(
                self.detail_channels, self.semantic_channels, kernel_size=1
            )
            # HSD-Full starts from exactly the semantic path for a shared state.
            nn.init.zeros_(self.detail_to_semantic.weight)
            if self.detail_to_semantic.bias is not None:
                nn.init.zeros_(self.detail_to_semantic.bias)

    def estimated_conv_macs_per_image(self) -> int:
        c = self.semantic_channels
        d = self.detail_channels
        n37, n74, n148 = 37 * 37, 74 * 74, 148 * 148
        macs = 4 * n37 * self.in_channels * c
        # Three cross-layer 1x1 gates and three depthwise local refiners.
        macs += 3 * n37 * ((3 * c + 1) + 9 * c)
        # Two serial scale-lift blocks: DW3x3 + PW1x1.
        macs += n74 * (9 * c + c * c)
        macs += n148 * (9 * c + c * c)
        # base, native-resolution coarse, and final predictors.
        coarse_pixels = n74 if self.coarse_size == 74 else n37
        macs += n37 * c + coarse_pixels * c + n148 * c
        if self.use_detail:
            # RGB+Sobel detail encoder, detail gate, and zero-init projection.
            macs += n148 * (4 * d * 9 + d * d * 9)
            macs += n148 * (c + d + 2)
            macs += n148 * d * c
        return int(macs)

    def _semantic_forward(self, features):
        features = _validate_last4(features, self.in_channels)
        adapted = {key: self.adapters[key](features[key]) for key in LAST4_KEYS}

        hierarchy = adapted["f12"]
        gates = {}
        agreements = {}
        for key in ("f11", "f10", "f9"):
            hierarchy, gates[key], agreements[key] = self.cross_gates[key](
                adapted[key], hierarchy
            )

        semantic_37 = hierarchy
        semantic_74 = self.lift_37_to_74(semantic_37)
        semantic_148 = self.lift_74_to_148(semantic_74)
        base_logits_37 = self.base_head(adapted["f12"])
        coarse_feature = (
            semantic_74 if self.coarse_size == 74 else semantic_37
        )
        coarse_logits = self.coarse_head(coarse_feature)
        return (
            semantic_37,
            semantic_74,
            semantic_148,
            base_logits_37,
            coarse_logits,
            gates,
            agreements,
        )

    def forward(self, features, image_148: torch.Tensor | None = None):
        (
            semantic_37,
            semantic_74,
            semantic_148,
            base_logits_37,
            coarse_logits,
            cross_gates,
            agreements,
        ) = self._semantic_forward(features)

        coarse_logits_148 = F.interpolate(
            coarse_logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        coarse_prob_148 = torch.sigmoid(coarse_logits_148)
        uncertainty_148 = 4.0 * coarse_prob_148 * (1.0 - coarse_prob_148)

        detail_feat_148 = None
        detail_gate_148 = None
        detail_residual_148 = torch.zeros_like(semantic_148)
        sobel_148 = None
        if self.use_detail:
            if image_148 is None:
                raise KeyError("HSD-Full requires image_148; HSD-Semantic does not.")
            if tuple(image_148.shape[-2:]) != (self.output_size, self.output_size):
                raise RuntimeError(
                    "HSD-Full image_148 must be 148x148, got "
                    f"{list(image_148.shape)}."
                )
            image_148 = image_148.to(dtype=semantic_148.dtype)
            sobel_148 = _normalized_sobel(image_148)
            detail_feat_148 = self.detail_encoder(torch.cat([image_148, sobel_148], dim=1))
            detail_gate_148 = uncertainty_148 * torch.sigmoid(
                self.detail_gate(
                    torch.cat(
                        [
                            semantic_148,
                            detail_feat_148,
                            coarse_prob_148,
                            sobel_148,
                        ],
                        dim=1,
                    )
                )
            )
            detail_residual_148 = self.detail_to_semantic(
                detail_gate_148 * detail_feat_148
            )

        semantic_logits_148 = self.final_head(semantic_148)
        detail_logits_residual_148 = F.conv2d(
            detail_residual_148,
            self.final_head.weight,
            bias=None,
        )
        final_logits = semantic_logits_148 + detail_logits_residual_148
        output = {
            "logits": final_logits,
            "final_logits": final_logits,
            "coarse_logits": coarse_logits,
            f"coarse_logits_{self.coarse_size}": coarse_logits,
            "base_logits_37": base_logits_37,
            "base_logits": base_logits_37,
            "semantic_feat_37": semantic_37,
            "semantic_feat_74": semantic_74,
            "semantic_feat_148": semantic_148,
            "coarse_logits_148": coarse_logits_148,
            "coarse_prob_148": coarse_prob_148,
            "uncertainty_148": uncertainty_148,
            "detail_residual_148": detail_residual_148,
            "semantic_logits_148": semantic_logits_148,
            "detail_logits_residual_148": detail_logits_residual_148,
        }
        for key in ("f11", "f10", "f9"):
            suffix = key[1:]
            output[f"cross_gate_{suffix}"] = cross_gates[key]
            output[f"cross_agreement_{suffix}"] = agreements[key]
        if self.use_detail:
            output.update(
                {
                    "sobel_148": sobel_148,
                    "detail_feat_148": detail_feat_148,
                    "detail_gate_148": detail_gate_148,
                }
            )
        return output
