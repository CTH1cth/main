import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nper.resnet_utils import load_resnet18_backbone


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rgb_to_gray_tensor(image):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image [3,H,W], got {list(image.shape)}")
    r, g, b = image[0], image[1], image[2]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(0)


def sobel_tensor(image):
    gray = rgb_to_gray_tensor(image).unsqueeze(0)
    sx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy).squeeze(0)
    return edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-8)


def dog_tensor(image, small=3, large=9):
    gray = rgb_to_gray_tensor(image).detach().cpu().numpy().squeeze().astype(np.float32)
    first = cv2.GaussianBlur(gray, (int(small), int(small)), 0)
    second = cv2.GaussianBlur(gray, (int(large), int(large)), 0)
    dog = np.abs(first - second)
    if float(dog.max()) > 0:
        dog = dog / float(dog.max())
    return torch.from_numpy(dog).to(device=image.device, dtype=image.dtype).unsqueeze(0)


def lbp_tensor(image):
    gray = rgb_to_gray_tensor(image).detach().cpu().numpy().squeeze().astype(np.float32)
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    code = np.zeros_like(center, dtype=np.float32)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        code += (neighbor >= center).astype(np.float32) * float(2**bit)
    code = code / 255.0
    return torch.from_numpy(code).to(device=image.device, dtype=image.dtype).unsqueeze(0)


class NativeCueExtractor:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.use_resnet = bool(getattr(cfg, "MNP_USE_RESNET18", False)) if cfg is not None else False
        self.resnet_mid = None
        self.pretrained = False
        if self.use_resnet:
            backbone, self.pretrained, self.weight_source = load_resnet18_backbone(
                cfg,
                pretrained=True,
                allow_random_init=bool(getattr(cfg, "MNP_ALLOW_RANDOM_RESNET", False)),
                component="NativeCueExtractor",
            )
            self.resnet_mid = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
            )
            if bool(getattr(cfg, "MNP_RESNET18_FREEZE", True)):
                self.resnet_mid.eval()
                for param in self.resnet_mid.parameters():
                    param.requires_grad_(False)

    @torch.no_grad()
    def _resnet18_mid(self, image):
        if self.resnet_mid is None:
            return None
        device = next(self.resnet_mid.parameters()).device
        tensor = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        mean = IMAGENET_MEAN.to(device=device, dtype=tensor.dtype)
        std = IMAGENET_STD.to(device=device, dtype=tensor.dtype)
        feat = self.resnet_mid((tensor - mean) / std)
        feat = F.interpolate(feat, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return feat.squeeze(0).detach().cpu()

    def extract(self, image):
        cues = {
            "lbp": lbp_tensor(image),
            "dog": dog_tensor(image),
            "sobel": sobel_tensor(image),
        }
        resnet_mid = self._resnet18_mid(image) if self.use_resnet else None
        if resnet_mid is not None:
            cues["resnet18_mid"] = resnet_mid
        return cues
