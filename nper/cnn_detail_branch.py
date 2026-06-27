import torch.nn.functional as F
from torch import nn

from nper.resnet_utils import load_resnet18_backbone


class CNNDetailBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone, self.pretrained, self.weight_source = load_resnet18_backbone(
            cfg,
            pretrained=bool(getattr(cfg, "DETAIL_PRETRAINED", True)),
            allow_random_init=bool(getattr(cfg, "DETAIL_ALLOW_RANDOM_INIT", False)),
            component="CNNDetailBranch",
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.project = nn.Sequential(
            nn.Conv2d(128, int(cfg.DETAIL_OUT_CHANNELS), kernel_size=1),
            nn.BatchNorm2d(int(cfg.DETAIL_OUT_CHANNELS)),
            nn.GELU(),
        )
        if bool(getattr(cfg, "DETAIL_FREEZE", False)):
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, image):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.project(x)
