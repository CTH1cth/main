"""Bounded cross-layer residual decoder for static Hard-R1 supervision."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


LAST4_KEYS = ("f9", "f10", "f11", "f12")


def _group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and int(channels) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def _validate_features(features, in_channels: int) -> dict[str, torch.Tensor]:
    if not isinstance(features, Mapping):
        raise TypeError(
            "BCRD-Sem input must be a mapping with f9/f10/f11/f12."
        )
    missing = [key for key in LAST4_KEYS if key not in features]
    if missing:
        raise KeyError(f"BCRD-Sem input is missing keys: {missing}")
    checked = {}
    for key in LAST4_KEYS:
        value = features[key]
        if not torch.is_tensor(value) or value.ndim != 4:
            shape = list(value.shape) if torch.is_tensor(value) else None
            raise RuntimeError(f"BCRD-Sem {key} must be [B,C,37,37], got {shape}.")
        if list(value.shape[1:]) != [int(in_channels), 37, 37]:
            raise RuntimeError(
                f"BCRD-Sem {key} shape mismatch: {list(value.shape)} != "
                f"[B,{int(in_channels)},37,37]."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"BCRD-Sem {key} contains NaN/Inf.")
        checked[key] = value
    return checked


class _Projection(nn.Sequential):
    def __init__(self, in_channels: int, dim: int, gn_groups: int):
        super().__init__(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            _group_norm(dim, gn_groups),
            nn.GELU(),
        )


class _ResidualProposal(nn.Sequential):
    def __init__(self, dim: int, gn_groups: int):
        super().__init__(
            nn.Conv2d(3 * dim + 1, dim, kernel_size=1),
            _group_norm(dim, gn_groups),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self[-1].weight)
        if self[-1].bias is not None:
            nn.init.zeros_(self[-1].bias)


class BCRDSemV1Head(nn.Module):
    """A strict F12 1x1 anchor with bounded bidirectional corrections."""

    def __init__(
        self,
        in_channels: int = 384,
        dim: int = 32,
        gn_groups: int = 4,
        consistency_tau: float = 0.10,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.dim = int(dim)
        self.gn_groups = int(gn_groups)
        self.consistency_tau = float(consistency_tau)
        self.alpha = float(alpha)
        if self.dim <= 0:
            raise ValueError(f"BCRD_DIM must be positive, got {dim}.")
        if self.consistency_tau <= 0.0:
            raise ValueError(
                "BCRD_CONSISTENCY_TAU must be positive, got "
                f"{consistency_tau}."
            )
        if self.alpha < 0.0:
            raise ValueError(f"BCRD_ALPHA must be non-negative, got {alpha}.")

        # This is deliberately a single plain 1x1 Conv over f12.
        self.base_head = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        self.projections = nn.ModuleDict(
            {
                key: _Projection(self.in_channels, self.dim, self.gn_groups)
                for key in LAST4_KEYS
            }
        )
        self.proposal_heads = nn.ModuleDict(
            {
                key: _ResidualProposal(self.dim, self.gn_groups)
                for key in ("f9", "f10", "f11")
            }
        )

    def forward(self, features):
        features = _validate_features(features, self.in_channels)
        base_logits_37 = self.base_head(features["f12"])
        base_prob_detached = torch.sigmoid(base_logits_37).detach()
        base_uncertainty = 4.0 * base_prob_detached * (1.0 - base_prob_detached)

        projected = {
            key: self.projections[key](features[key]) for key in LAST4_KEYS
        }
        deep = projected["f12"]
        proposals = {}
        for key in ("f9", "f10", "f11"):
            shallow = projected[key]
            cosine = F.cosine_similarity(
                shallow, deep, dim=1, eps=1e-6
            ).unsqueeze(1)
            proposal_input = torch.cat(
                [shallow, deep, torch.abs(shallow - deep), cosine], dim=1
            )
            proposals[key] = self.proposal_heads[key](proposal_input)

        proposal_stack = torch.stack(
            [proposals[key] for key in ("f9", "f10", "f11")], dim=1
        )
        proposal_mean = proposal_stack.mean(dim=1)
        proposal_variance = proposal_stack.var(dim=1, unbiased=False)
        consistency_gate = torch.exp(
            -proposal_variance.detach() / self.consistency_tau
        )
        applied_residual = (
            self.alpha
            * base_uncertainty
            * consistency_gate
            * torch.tanh(proposal_mean)
        )
        final_logits_37 = base_logits_37 + applied_residual

        return {
            "logits": final_logits_37,
            "final_logits": final_logits_37,
            "final_logits_37": final_logits_37,
            "base_logits": base_logits_37,
            "base_logits_37": base_logits_37,
            "proposal_f9": proposals["f9"],
            "proposal_f10": proposals["f10"],
            "proposal_f11": proposals["f11"],
            "proposal_mean": proposal_mean,
            "proposal_variance": proposal_variance,
            "consistency_gate": consistency_gate,
            "base_uncertainty": base_uncertainty,
            "applied_residual": applied_residual,
        }

    def estimated_conv_macs_per_image(self) -> int:
        n = 37 * 37
        d = self.dim
        macs = n * self.in_channels  # F12 base 1x1.
        macs += 4 * n * self.in_channels * d  # Four projections.
        macs += 3 * n * ((3 * d + 1) * d + d)  # Proposal heads.
        return int(macs)
