"""DINOv1-B/8 feature protocol matching released UCOD-DPL preprocessing."""

from configs.dinov1_b8 import *  # noqa: F401,F403


EXP_NAME = "dinov1_b8_ucod_dpl"
FEATURE_CACHE_KEY = "dinov1-b8-ucod-dpl"

# UCOD-DPL uses torchvision.transforms.Resize with its default PIL bilinear
# interpolation for the 296x296 feature input.
FEATURE_RESIZE_INTERPOLATION = "bilinear"
