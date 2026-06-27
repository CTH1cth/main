import torch
import torch.nn.functional as F
from torch import nn

from nper.aff_decoder import AdaptiveFrequencyFusion
from nper.cnn_detail_branch import CNNDetailBranch
from nper.dino_lora import FrozenDINOWithAdapters
from nper.online_dino_key import OnlineDINOKeyExtractor


class LinearProbeHead(nn.Module):
    def __init__(self, in_channels, head_type="conv1x1"):
        super().__init__()
        if head_type == "conv3x3":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            )
        elif head_type == "conv1x1":
            self.net = nn.Conv2d(in_channels, 1, kernel_size=1)
        else:
            raise ValueError(f"Unknown LINEAR_PROBE_HEAD: {head_type}")

    def forward(self, x):
        return self.net(x)


class NPERUCOD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.decoder_type = str(getattr(cfg, "DECODER", "aff_dual"))
        if self.decoder_type == "linear_probe":
            self.dino_key = OnlineDINOKeyExtractor(cfg)
            self.linear_head = LinearProbeHead(
                in_channels=int(self.dino_key.model.config.hidden_size),
                head_type=str(getattr(cfg, "LINEAR_PROBE_HEAD", "conv1x1")),
            )
            self.dino = self.dino_key
            self.detail = None
            return
        self.dino = FrozenDINOWithAdapters(cfg)
        self.detail = CNNDetailBranch(cfg)
        decoder_channels = int(cfg.DECODER_CHANNELS)
        semantic_channels = [self.dino.hidden_size] * len(getattr(cfg, "DINO_OUT_LAYERS", [3, 6, 9, 12]))
        self.decoder = AdaptiveFrequencyFusion(
            semantic_channels=semantic_channels,
            detail_channels=int(cfg.DETAIL_OUT_CHANNELS),
            decoder_channels=decoder_channels,
            use_aff=bool(getattr(cfg, "USE_AFF", True)),
        )
        self.fg_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.bg_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.final_head = nn.Sequential(
            nn.Conv2d(decoder_channels + 3, decoder_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def forward(self, image):
        if self.decoder_type == "linear_probe":
            raise RuntimeError("linear_probe forward requires forward_with_paths(image, image_paths).")
        return self.forward_with_paths(image, None)

    def forward_with_paths(self, image, image_paths=None):
        if self.decoder_type == "linear_probe":
            if image_paths is None:
                raise RuntimeError("linear_probe requires image_paths for cache-aligned DINO preprocessing.")
            feature = self.dino_key(image_paths, device=image.device)
            logits = self.linear_head(feature)
            logits = F.interpolate(
                logits,
                size=(int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE)),
                mode="bilinear",
                align_corners=False,
            )
            zeros = torch.zeros_like(logits)
            return {
                "logits": logits,
                "prob": logits.sigmoid(),
                "fg_logits": logits,
                "bg_logits": -logits,
                "boundary_logits": zeros,
                "features": {"fused": feature, "semantic": [feature], "detail": None},
                "attn_maps": [],
            }
        dino_out = self.dino(image)
        detail = self.detail(image)
        fused = self.decoder(dino_out["semantic_feats"], detail)
        fg_logits = self.fg_head(fused)
        bg_logits = self.bg_head(fused)
        boundary_logits = self.boundary_head(fused)
        logits = self.final_head(torch.cat([fused, fg_logits, -bg_logits, boundary_logits], dim=1))
        logits = F.interpolate(
            logits,
            size=(int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE)),
            mode="bilinear",
            align_corners=False,
        )
        fg_logits = F.interpolate(fg_logits, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        bg_logits = F.interpolate(bg_logits, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(
            boundary_logits, size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        return {
            "logits": logits,
            "prob": logits.sigmoid(),
            "fg_logits": fg_logits,
            "bg_logits": bg_logits,
            "boundary_logits": boundary_logits,
            "features": {
                "semantic": dino_out["semantic_feats"],
                "detail": detail,
                "fused": fused,
            },
            "attn_maps": dino_out["attn_maps"],
        }


@torch.no_grad()
def update_ema_model(student, teacher, momentum=0.99):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(float(momentum)).add_(student_param.data, alpha=1.0 - float(momentum))
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
