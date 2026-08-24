"""512-B: only the relative border-width mismatch is corrected to BW=3."""

from configs.dinov1_s8_gbsp_resolution_512_native64_base import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_resolution_512_native64_bw3"
GBSP_RESOLUTION_VARIANT = "512-B"
DABE_BORDER_WIDTH = 3
GBSP_BORDER_WIDTH = 3

