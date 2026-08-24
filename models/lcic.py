"""Minimal single-layer DINO readouts for the GBSP LCIC ablation.

LCIC intentionally consumes only the frozen, final-layer DINO patch feature.
The local innovation below is a 3x3 DINO-graph high-pass component; it is not
the GBSP background-subspace reconstruction residual used to build labels.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


LOCAL_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SELF_OFFSET_INDEX = LOCAL_OFFSETS.index((0, 0))


def _validate_feature_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 4:
        raise TypeError(f"{name} must be a [B,C,H,W] tensor.")
    if min(int(value.shape[-2]), int(value.shape[-1])) < 1:
        raise ValueError(f"{name} must have non-empty spatial dimensions.")


def _shifted_views(value: torch.Tensor):
    """Yield each valid 3x3 neighbor aligned to its center location."""

    height, width = (int(value.shape[-2]), int(value.shape[-1]))
    padded = F.pad(value, (1, 1, 1, 1), mode="constant", value=0.0)
    for dy, dx in LOCAL_OFFSETS:
        yield padded[
            ...,
            1 + dy : 1 + dy + height,
            1 + dx : 1 + dx + width,
        ]


@torch.no_grad()
def build_local_dino_affinity(
    feature: torch.Tensor,
    eps: float = 1e-6,
    return_diagnostics: bool = False,
):
    """Build detached ReLU-cosine weights for valid 8-neighbors plus self.

    The returned tensor is ``[B,9,H,W]`` rather than a dense ``[B,HW,HW]``
    matrix. Invalid border neighbors receive exactly zero weight. The self
    score is set to one explicitly, which is the cosine self-similarity for
    every non-zero DINO feature and also keeps synthetic zero-input tests safe.
    """

    _validate_feature_tensor(feature, "LCIC feature")
    if not float(eps) > 0.0:
        raise ValueError(f"LCIC affinity eps must be positive, got {eps}.")

    detached_feature = feature.detach()
    normalized = F.normalize(detached_feature, p=2, dim=1, eps=float(eps))
    valid_source = torch.ones(
        (int(feature.shape[0]), 1, int(feature.shape[2]), int(feature.shape[3])),
        device=feature.device,
        dtype=feature.dtype,
    )
    neighbor_features = _shifted_views(normalized)
    neighbor_validity = _shifted_views(valid_source)
    scores = []
    self_similarity = None
    for index, (neighbor, valid) in enumerate(
        zip(neighbor_features, neighbor_validity)
    ):
        similarity = (normalized * neighbor).sum(dim=1, keepdim=True)
        similarity = torch.relu(similarity) * valid
        if index == SELF_OFFSET_INDEX:
            similarity = valid
            self_similarity = similarity
        scores.append(similarity)

    score_tensor = torch.cat(scores, dim=1)
    weights = score_tensor / score_tensor.sum(dim=1, keepdim=True).clamp_min(
        float(eps)
    )
    weights = weights.detach()
    if not return_diagnostics:
        return weights

    row_sum = weights.sum(dim=1)
    diagnostics = {
        "row_sum_max_abs_error": (row_sum - 1.0).abs().max(),
        "self_similarity_min": self_similarity.min(),
        "self_similarity_mean": self_similarity.mean(),
        "self_similarity_max": self_similarity.max(),
        "affinity_requires_grad": bool(weights.requires_grad),
        "dense_affinity_materialized": False,
    }
    return weights, diagnostics


def local_affinity_propagate(
    affinity: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Apply one local propagation step without materializing dense affinity."""

    _validate_feature_tensor(value, "LCIC propagation value")
    if not torch.is_tensor(affinity) or affinity.ndim != 4:
        raise TypeError("LCIC affinity must be a [B,9,H,W] tensor.")
    expected = (
        int(value.shape[0]),
        len(LOCAL_OFFSETS),
        int(value.shape[2]),
        int(value.shape[3]),
    )
    if tuple(affinity.shape) != expected:
        raise RuntimeError(
            f"LCIC affinity/value shape mismatch: {tuple(affinity.shape)} != {expected}."
        )

    propagated = torch.zeros_like(value)
    for index, neighbor in enumerate(_shifted_views(value)):
        propagated = propagated + affinity[:, index : index + 1] * neighbor
    return propagated


class LCICHead(nn.Module):
    """Linear anchor with optional one-step consensus and innovation correction."""

    _DIAGNOSTIC_NAMES = (
        "anchor_abs_mean",
        "consensus_delta_abs_mean",
        "innovation_logit_abs_mean",
        "weighted_consensus_abs_mean",
        "weighted_innovation_abs_mean",
    )

    def __init__(
        self,
        in_channels: int = 384,
        use_consensus: bool = True,
        use_innovation: bool = True,
        affinity_eps: float = 1e-6,
        consensus_gain: float = 1.0,
        innovation_gain: float = 1.0,
        adaptive_gate: bool = False,
        adaptive_gate_hidden: int = 8,
        adaptive_gate_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_consensus = bool(use_consensus)
        self.use_innovation = bool(use_innovation)
        self.affinity_eps = float(affinity_eps)
        self.consensus_gain = float(consensus_gain)
        self.innovation_gain = float(innovation_gain)
        self.adaptive_gate_enabled = bool(adaptive_gate)
        self.adaptive_gate_hidden = int(adaptive_gate_hidden)
        self.adaptive_gate_eps = float(adaptive_gate_eps)
        if self.in_channels <= 0:
            raise ValueError("LCIC in_channels must be positive.")
        if not math.isfinite(self.consensus_gain) or self.consensus_gain < 0.0:
            raise ValueError("LCIC consensus gain must be finite and non-negative.")
        if not math.isfinite(self.innovation_gain) or self.innovation_gain < 0.0:
            raise ValueError("LCIC innovation gain must be finite and non-negative.")
        if not (self.use_consensus or self.use_innovation):
            raise ValueError("LCICHead requires at least one correction branch.")
        if self.adaptive_gate_enabled and not (
            self.use_consensus and self.use_innovation
        ):
            raise ValueError(
                "LCIC adaptive gating requires both consensus and innovation."
            )
        if self.adaptive_gate_enabled and (
            self.consensus_gain != 1.0 or self.innovation_gain != 1.0
        ):
            raise ValueError(
                "LCIC adaptive gating forbids fixed consensus/innovation gains."
            )
        if self.adaptive_gate_hidden <= 0:
            raise ValueError("LCIC adaptive gate hidden width must be positive.")
        if not math.isfinite(self.adaptive_gate_eps) or self.adaptive_gate_eps <= 0.0:
            raise ValueError("LCIC adaptive gate eps must be finite and positive.")

        # Anchor is constructed first so a fixed seed matches SimpleConvSegHead.
        self.anchor = nn.Conv2d(self.in_channels, 1, kernel_size=1, bias=True)
        if self.use_innovation:
            self.innovation_head = nn.Conv2d(
                self.in_channels, 1, kernel_size=1, bias=False
            )
            if self.adaptive_gate_enabled:
                self.register_parameter("beta", None)
            else:
                self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.innovation_head = None
            self.register_parameter("beta", None)
        if self.use_consensus:
            if self.adaptive_gate_enabled:
                self.register_parameter("alpha", None)
            else:
                self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("alpha", None)

        if self.adaptive_gate_enabled:
            # Four detached image descriptors (entropy, foreground mass and
            # the two raw residual/anchor RMS ratios) produce one consensus
            # and one innovation coefficient per image.  The zero-initialized
            # last layer preserves exact equality with the linear anchor while
            # leaving both coefficients directly learnable and unbounded.
            self.adaptive_gate = nn.Sequential(
                nn.Linear(4, self.adaptive_gate_hidden),
                nn.GELU(),
                nn.Linear(self.adaptive_gate_hidden, 2),
            )
            nn.init.zeros_(self.adaptive_gate[-1].weight)
            nn.init.zeros_(self.adaptive_gate[-1].bias)
        else:
            self.adaptive_gate = None

        self.register_buffer(
            "_epoch_diagnostic_sums", torch.zeros(len(self._DIAGNOSTIC_NAMES)),
            persistent=False,
        )
        self.register_buffer(
            "_epoch_diagnostic_batches", torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_epoch_gate_sum", torch.zeros(2), persistent=False,
        )
        self.register_buffer(
            "_epoch_gate_square_sum", torch.zeros(2), persistent=False,
        )
        self.register_buffer(
            "_epoch_gate_count", torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_epoch_gate_min", torch.full((2,), float("inf")), persistent=False,
        )
        self.register_buffer(
            "_epoch_gate_max", torch.full((2,), float("-inf")), persistent=False,
        )

    @property
    def variant_name(self) -> str:
        if self.adaptive_gate_enabled:
            return "LCIC_D_FULL_ADAPTIVE_GATE"
        if self.use_consensus and self.use_innovation:
            return "LCIC_D_FULL"
        if self.use_consensus:
            return "LCIC_B_CONSENSUS"
        return "LCIC_C_INNOVATION"

    def reset_epoch_diagnostics(self) -> None:
        self._epoch_diagnostic_sums.zero_()
        self._epoch_diagnostic_batches.zero_()
        self._epoch_gate_sum.zero_()
        self._epoch_gate_square_sum.zero_()
        self._epoch_gate_count.zero_()
        self._epoch_gate_min.fill_(float("inf"))
        self._epoch_gate_max.fill_(float("-inf"))

    def _normalize_adaptive_residual(
        self, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a unit-RMS residual with a detached per-image scale."""

        rms = value.detach().float().square().mean(
            dim=(1, 2, 3), keepdim=True
        ).sqrt()
        valid = rms > self.adaptive_gate_eps
        inverse_rms = torch.where(
            valid,
            rms.clamp_min(self.adaptive_gate_eps).reciprocal(),
            torch.zeros_like(rms),
        )
        normalized = value * inverse_rms.to(device=value.device, dtype=value.dtype)
        return normalized, rms

    def _adaptive_gate_values(
        self,
        anchor_logits: torch.Tensor,
        consensus_rms: torch.Tensor,
        innovation_rms: torch.Tensor,
    ) -> torch.Tensor:
        """Predict detached-statistic, image-specific residual strengths."""

        detached_anchor = anchor_logits.detach().float()
        probability = detached_anchor.sigmoid().clamp(
            self.adaptive_gate_eps, 1.0 - self.adaptive_gate_eps
        )
        entropy = -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()
        ).mean(dim=(1, 2, 3)) / math.log(2.0)
        foreground_mass = probability.mean(dim=(1, 2, 3))
        anchor_rms = detached_anchor.square().mean(
            dim=(1, 2, 3), keepdim=True
        ).sqrt()
        denominator = anchor_rms.clamp_min(self.adaptive_gate_eps)
        consensus_ratio = torch.log1p(consensus_rms / denominator).flatten()
        innovation_ratio = torch.log1p(innovation_rms / denominator).flatten()
        descriptors = torch.stack(
            (entropy, foreground_mass, consensus_ratio, innovation_ratio), dim=1
        ).detach()
        descriptors = torch.nan_to_num(
            descriptors, nan=0.0, posinf=20.0, neginf=-20.0
        )
        gates = self.adaptive_gate(
            descriptors.to(device=anchor_logits.device, dtype=anchor_logits.dtype)
        )
        return gates.reshape(int(anchor_logits.shape[0]), 2, 1, 1)

    @torch.no_grad()
    def _accumulate_gate_diagnostics(self, gates: torch.Tensor) -> None:
        flattened = gates.detach().reshape(int(gates.shape[0]), 2).float()
        self._epoch_gate_sum.add_(flattened.sum(dim=0).to(self._epoch_gate_sum))
        self._epoch_gate_square_sum.add_(
            flattened.square().sum(dim=0).to(self._epoch_gate_square_sum)
        )
        self._epoch_gate_count.add_(int(flattened.shape[0]))
        self._epoch_gate_min.copy_(
            torch.minimum(
                self._epoch_gate_min,
                flattened.amin(dim=0).to(self._epoch_gate_min),
            )
        )
        self._epoch_gate_max.copy_(
            torch.maximum(
                self._epoch_gate_max,
                flattened.amax(dim=0).to(self._epoch_gate_max),
            )
        )

    @torch.no_grad()
    def epoch_diagnostics(self) -> Dict[str, float]:
        count = int(self._epoch_diagnostic_batches.item())
        divisor = max(count, 1)
        means = self._epoch_diagnostic_sums / divisor
        result = {
            name: float(means[index].item())
            for index, name in enumerate(self._DIAGNOSTIC_NAMES)
        }
        gate_count = int(self._epoch_gate_count.item())
        if gate_count > 0:
            gate_mean = self._epoch_gate_sum / gate_count
            gate_variance = (
                self._epoch_gate_square_sum / gate_count - gate_mean.square()
            ).clamp_min(0.0)
            gate_std = gate_variance.sqrt()
            gate_min = self._epoch_gate_min
            gate_max = self._epoch_gate_max
        else:
            gate_mean = torch.zeros_like(self._epoch_gate_sum)
            gate_std = torch.zeros_like(self._epoch_gate_sum)
            gate_min = torch.zeros_like(self._epoch_gate_sum)
            gate_max = torch.zeros_like(self._epoch_gate_sum)
        result.update(
            {
                "batches": count,
                "adaptive_gate_enabled": self.adaptive_gate_enabled,
                "gate_samples": gate_count,
                "alpha": float(self.alpha.item()) if self.alpha is not None else 0.0,
                "beta": float(self.beta.item()) if self.beta is not None else 0.0,
                "consensus_gain": self.consensus_gain,
                "innovation_gain": self.innovation_gain,
                "effective_alpha": (
                    self.consensus_gain * float(self.alpha.item())
                    if self.alpha is not None
                    else 0.0
                ),
                "effective_beta": (
                    self.innovation_gain * float(self.beta.item())
                    if self.beta is not None
                    else 0.0
                ),
                "consensus_gate_mean": float(gate_mean[0].item()),
                "consensus_gate_std": float(gate_std[0].item()),
                "consensus_gate_min": float(gate_min[0].item()),
                "consensus_gate_max": float(gate_max[0].item()),
                "innovation_gate_mean": float(gate_mean[1].item()),
                "innovation_gate_std": float(gate_std[1].item()),
                "innovation_gate_min": float(gate_min[1].item()),
                "innovation_gate_max": float(gate_max[1].item()),
            }
        )
        return result

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(feature, "LCIC feature")
        if int(feature.shape[1]) != self.in_channels:
            raise RuntimeError(
                f"LCIC input channels mismatch: {int(feature.shape[1])} != "
                f"{self.in_channels}."
            )

        anchor_logits = self.anchor(feature)
        affinity = build_local_dino_affinity(
            feature.detach(), eps=self.affinity_eps
        )
        output = anchor_logits
        consensus_delta = torch.zeros_like(anchor_logits)
        innovation_logits = torch.zeros_like(anchor_logits)
        weighted_consensus = torch.zeros_like(anchor_logits)
        weighted_innovation = torch.zeros_like(anchor_logits)

        if self.use_consensus:
            consensus_logits = local_affinity_propagate(affinity, anchor_logits)
            consensus_delta = consensus_logits - anchor_logits

        if self.use_innovation:
            detached_feature = feature.detach()
            local_consensus_feature = local_affinity_propagate(
                affinity, detached_feature
            )
            local_innovation_feature = detached_feature - local_consensus_feature
            innovation_logits = self.innovation_head(local_innovation_feature)

        adaptive_gates = None
        if self.adaptive_gate_enabled:
            normalized_consensus, consensus_rms = self._normalize_adaptive_residual(
                consensus_delta
            )
            normalized_innovation, innovation_rms = self._normalize_adaptive_residual(
                innovation_logits
            )
            adaptive_gates = self._adaptive_gate_values(
                anchor_logits, consensus_rms, innovation_rms
            )
            weighted_consensus = adaptive_gates[:, 0:1] * normalized_consensus
            weighted_innovation = adaptive_gates[:, 1:2] * normalized_innovation
            output = output + weighted_consensus + weighted_innovation
        else:
            if self.use_consensus:
                weighted_consensus = (
                    self.consensus_gain * self.alpha * consensus_delta
                )
                output = output + weighted_consensus
            if self.use_innovation:
                weighted_innovation = (
                    self.innovation_gain * self.beta * innovation_logits
                )
                output = output + weighted_innovation

        if self.training:
            with torch.no_grad():
                batch_diagnostics = torch.stack(
                    (
                        anchor_logits.detach().abs().mean(),
                        consensus_delta.detach().abs().mean(),
                        innovation_logits.detach().abs().mean(),
                        weighted_consensus.detach().abs().mean(),
                        weighted_innovation.detach().abs().mean(),
                    )
                ).to(self._epoch_diagnostic_sums)
                self._epoch_diagnostic_sums.add_(batch_diagnostics)
                self._epoch_diagnostic_batches.add_(1)
                if adaptive_gates is not None:
                    self._accumulate_gate_diagnostics(adaptive_gates)
        return output


class DWLiteHead(nn.Module):
    """Capacity control with ordinary local convolutions and no DINO graph."""

    def __init__(self, in_channels: int = 384, hidden_channels: int = 16):
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("DW-Lite channels must be positive.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act2 = nn.GELU()
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(feature, "DW-Lite feature")
        if int(feature.shape[1]) != self.in_channels:
            raise RuntimeError(
                f"DW-Lite input channels mismatch: {int(feature.shape[1])} != "
                f"{self.in_channels}."
            )
        value = self.act1(self.reduce(feature))
        value = self.act2(self.depthwise(value))
        return self.out(value)


class Conv3x3LiteHead(nn.Module):
    """Traditional lightweight decoder with an ordinary dense 3x3 conv."""

    def __init__(self, in_channels: int = 384, hidden_channels: int = 16):
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Conv3x3-Lite channels must be positive.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.reduce = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.conv3x3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.act2 = nn.GELU()
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _validate_feature_tensor(feature, "Conv3x3-Lite feature")
        if int(feature.shape[1]) != self.in_channels:
            raise RuntimeError(
                "Conv3x3-Lite input channels mismatch: "
                f"{int(feature.shape[1])} != {self.in_channels}."
            )
        value = self.act1(self.reduce(feature))
        value = self.act2(self.conv3x3(value))
        return self.out(value)
