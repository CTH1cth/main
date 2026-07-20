import math

import torch
from torch import nn


class PredictiveSupervisionStateFilter(nn.Module):
    """Patch-wise training-only predictor for gain or horizon retention."""

    def __init__(
        self,
        feature_channels=384,
        feature_proj_dim=32,
        state_channels=9,
        hidden_dim=32,
        gn_groups=4,
        init_gain=0.01,
        output_semantics="single_step_gain",
    ):
        super().__init__()
        feature_channels = int(feature_channels)
        feature_proj_dim = int(feature_proj_dim)
        state_channels = int(state_channels)
        hidden_dim = int(hidden_dim)
        gn_groups = int(gn_groups)
        init_gain = float(init_gain)
        if not 0.0 < init_gain < 1.0:
            raise ValueError(f"PSSF init_gain must be in (0,1), got {init_gain}.")
        if hidden_dim % gn_groups != 0:
            raise ValueError(
                f"PSSF hidden_dim must be divisible by gn_groups: "
                f"{hidden_dim} vs {gn_groups}."
            )

        self.feature_channels = feature_channels
        self.feature_proj_dim = feature_proj_dim
        self.state_channels = state_channels
        self.hidden_dim = hidden_dim
        self.gn_groups = gn_groups
        self.init_gain = init_gain
        self.output_semantics = str(output_semantics).strip().lower()
        if self.output_semantics not in {
            "single_step_gain",
            "horizon_innovation_retention",
        }:
            raise ValueError(
                "Unsupported PSSF output semantics: "
                f"{self.output_semantics!r}."
            )

        self.feature_proj = nn.Conv2d(
            feature_channels, feature_proj_dim, kernel_size=1
        )
        self.local = nn.Sequential(
            nn.Conv2d(
                feature_proj_dim + state_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(gn_groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.GroupNorm(gn_groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 16, kernel_size=1),
            nn.GELU(),
        )
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.constant_(
            self.final_conv.bias,
            math.log(init_gain / (1.0 - init_gain)),
        )

    @property
    def input_channels(self):
        return self.feature_proj_dim + self.state_channels

    def protocol(self):
        protocol = {
            "feature_channels": self.feature_channels,
            "feature_proj_dim": self.feature_proj_dim,
            "state_channels": self.state_channels,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "gn_groups": self.gn_groups,
            "init_gain": self.init_gain,
        }
        # Keep the protected v1 protocol payload byte-for-byte compatible.
        if self.output_semantics != "single_step_gain":
            protocol["output_semantics"] = self.output_semantics
        return protocol

    def forward(self, feature_37, state_37):
        if feature_37.ndim != 4 or int(feature_37.shape[1]) != self.feature_channels:
            raise RuntimeError(
                "PSSF feature must be "
                f"[B,{self.feature_channels},H,W], got {list(feature_37.shape)}."
            )
        if state_37.ndim != 4 or int(state_37.shape[1]) != self.state_channels:
            raise RuntimeError(
                "PSSF state must be "
                f"[B,{self.state_channels},H,W], got {list(state_37.shape)}."
            )
        if tuple(feature_37.shape[-2:]) != tuple(state_37.shape[-2:]):
            raise RuntimeError(
                "PSSF feature/state spatial mismatch: "
                f"{list(feature_37.shape)} vs {list(state_37.shape)}."
            )
        if not bool(torch.isfinite(feature_37).all().item()):
            raise RuntimeError("PSSF feature contains NaN/Inf.")
        if not bool(torch.isfinite(state_37).all().item()):
            raise RuntimeError("PSSF dynamic state contains NaN/Inf.")

        feature_37 = feature_37.detach().float()
        state_37 = state_37.detach().float()
        projected = self.feature_proj(feature_37)
        output = torch.sigmoid(
            self.final_conv(self.local(torch.cat([projected, state_37], dim=1)))
        )
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("PSSF output contains NaN/Inf.")
        return output
