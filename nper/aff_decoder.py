import torch
import torch.nn.functional as F
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // int(reduction))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        weighted = (avg + mx) * 0.5
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx) + self.mlp(weighted))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return torch.sigmoid(self.conv(torch.cat([avg, mx, std], dim=1)))


class AdaptiveFrequencyFusion(nn.Module):
    def __init__(self, semantic_channels, detail_channels, decoder_channels=256, use_aff=True):
        super().__init__()
        self.use_aff = bool(use_aff)
        self.semantic_proj = nn.ModuleList(
            [nn.Conv2d(int(ch), int(decoder_channels), kernel_size=1) for ch in semantic_channels]
        )
        self.detail_proj = nn.Conv2d(int(detail_channels), int(decoder_channels), kernel_size=1)
        in_channels = int(decoder_channels) * (len(semantic_channels) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, int(decoder_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(decoder_channels)),
            nn.GELU(),
            nn.Conv2d(int(decoder_channels), int(decoder_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(decoder_channels)),
            nn.GELU(),
        )
        self.high_freq = nn.Conv2d(int(decoder_channels), int(decoder_channels), kernel_size=3, padding=1)
        self.channel_attn = ChannelAttention(int(decoder_channels))
        self.spatial_attn = SpatialAttention()

    def forward(self, semantic_feats, detail_feat):
        target_size = detail_feat.shape[-2:]
        feats = []
        for feat, proj in zip(semantic_feats, self.semantic_proj):
            z = proj(feat)
            z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
            feats.append(z)
        detail = self.detail_proj(detail_feat)
        fused = self.fuse(torch.cat(feats + [detail], dim=1))
        if not self.use_aff:
            return fused
        low = F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
        high = self.high_freq(fused - low)
        fused = fused + high
        return fused * self.channel_attn(fused) * self.spatial_attn(fused) + fused
