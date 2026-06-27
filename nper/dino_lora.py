import inspect
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _call_dino(model, pixel_values):
    params = inspect.signature(model.forward).parameters
    kwargs = {
        "output_hidden_states": True,
        "output_attentions": True,
        "return_dict": True,
    }
    if "interpolate_pos_encoding" in params:
        kwargs["interpolate_pos_encoding"] = True
    return model(pixel_values, **kwargs)


class MSMRAdapter(nn.Module):
    def __init__(self, channels, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        rank = max(1, int(rank))
        self.scale = float(alpha) / float(rank)
        self.down = nn.Conv2d(channels, rank, kernel_size=1)
        self.ms1 = nn.Conv2d(rank, rank, kernel_size=1)
        self.ms3 = nn.Conv2d(rank, rank, kernel_size=3, padding=1)
        self.ms5 = nn.Conv2d(rank, rank, kernel_size=5, padding=2)
        self.ms_fuse = nn.Conv2d(rank * 3, rank, kernel_size=1)
        self.mr = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(rank, rank, kernel_size=3, padding=1),
        )
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.up = nn.Conv2d(rank, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(x)
        ms = torch.cat([self.ms1(z), self.ms3(z), self.ms5(z)], dim=1)
        ms = self.ms_fuse(ms)
        small = F.avg_pool2d(z, kernel_size=2, stride=2, ceil_mode=True)
        small = self.mr(small)
        small = F.interpolate(small, size=z.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.up(self.dropout(F.gelu(ms + small)))
        return x + self.scale * delta


class FrozenDINOWithAdapters(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_path = Path(cfg.DINO_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Local DINO weight path not found: {model_path}")
        try:
            self.dino = AutoModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                add_pooling_layer=False,
                output_attentions=True,
                output_hidden_states=True,
            )
        except TypeError:
            self.dino = AutoModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                output_attentions=True,
                output_hidden_states=True,
            )
        self.hidden_size = int(self.dino.config.hidden_size)
        self.out_layers = [int(v) for v in getattr(cfg, "DINO_OUT_LAYERS", [3, 6, 9, 12])]
        self.attn_layers = [int(v) for v in getattr(cfg, "DINO_ATTN_LAYERS", [6, 9, 12])]
        self.freeze_base = bool(getattr(cfg, "DINO_FREEZE_BASE", True))
        if self.freeze_base:
            for param in self.dino.parameters():
                param.requires_grad_(False)
            self.dino.eval()

        self.use_adapters = bool(getattr(cfg, "DINO_USE_LORA", True))
        self.adapters = nn.ModuleDict()
        if self.use_adapters:
            for layer in self.out_layers:
                self.adapters[str(layer)] = MSMRAdapter(
                    self.hidden_size,
                    rank=int(getattr(cfg, "DINO_LORA_RANK", 8)),
                    alpha=float(getattr(cfg, "DINO_LORA_ALPHA", 16)),
                    dropout=float(getattr(cfg, "DINO_LORA_DROPOUT", 0.0)),
                )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.dino.eval()
        return self

    def _base_forward(self, image):
        mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
        std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
        pixel_values = (image - mean) / std
        if self.freeze_base:
            with torch.no_grad():
                return _call_dino(self.dino, pixel_values)
        return _call_dino(self.dino, pixel_values)

    def forward(self, image):
        outputs = self._base_forward(image)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("DINO did not return hidden states.")
        last_tokens = hidden_states[-1]
        patch_count = int(last_tokens.shape[1] - 1)
        grid = int(math.sqrt(patch_count))
        if grid * grid != patch_count:
            raise RuntimeError(f"Non-square DINO patch token count: {patch_count}")

        semantic_feats = []
        for layer in self.out_layers:
            index = max(0, min(int(layer), len(hidden_states) - 1))
            tokens = hidden_states[index][:, 1:, :]
            feat = tokens.transpose(1, 2).reshape(image.shape[0], self.hidden_size, grid, grid)
            adapter_key = str(layer)
            adapter = self.adapters[adapter_key] if self.use_adapters and adapter_key in self.adapters else None
            if adapter is not None:
                feat = adapter(feat)
            semantic_feats.append(feat)

        attn_maps = []
        if outputs.attentions is not None:
            for layer in self.attn_layers:
                index = max(0, min(int(layer) - 1, len(outputs.attentions) - 1))
                attn_maps.append(outputs.attentions[index].detach())

        return {
            "semantic_feats": semantic_feats,
            "attn_maps": attn_maps,
            "grid": grid,
        }
