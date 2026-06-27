from pathlib import Path

import torch
from torch import nn

from common.cache_features import (
    call_dino,
    key_to_feature_tensor,
    load_dino,
    preprocess_image,
    resolve_key_projection,
)


class OnlineDINOKeyExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = load_dino(cfg, torch.device("cpu"))
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.key_holder = {"tensor": None}
        key_module, key_path = resolve_key_projection(self.model)
        self.key_path = key_path
        self.handle = key_module.register_forward_hook(self._hook_key)

    def _hook_key(self, _module, _inputs, output):
        self.key_holder["tensor"] = output

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _load_batch_inputs(self, image_paths, device):
        size = int(self.cfg.DINO["feature_input_size"])
        tensors = []
        for path in image_paths:
            tensor, _ = preprocess_image(str(Path(path)), size)
            tensors.append(tensor.squeeze(0))
        return torch.stack(tensors, dim=0).to(device)

    @torch.no_grad()
    def forward(self, image_paths, device=None):
        if device is None:
            device = next(self.model.parameters()).device
        if next(self.model.parameters()).device != device:
            self.model.to(device)
        inputs = self._load_batch_inputs(image_paths, device)
        self.key_holder["tensor"] = None
        call_dino(self.model, inputs)
        key = self.key_holder["tensor"]
        if key is None:
            raise RuntimeError("DINO key hook did not capture a tensor.")
        features = []
        for index in range(key.shape[0]):
            features.append(key_to_feature_tensor(key[index : index + 1]))
        return torch.stack(features, dim=0).to(device)
