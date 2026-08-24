"""512-A: pure resolution/native-grid transfer with the original BW=2."""

from configs.dinov1_s8_gbsp_resolution_512_native64_base import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_resolution_512_native64_bw2"
GBSP_RESOLUTION_VARIANT = "512-A"
DABE_BORDER_WIDTH = 2
GBSP_BORDER_WIDTH = 2

