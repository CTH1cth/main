"""512-C: native-64/BW2 with the PCA capacity cap raised from 8 to 12."""

from configs.dinov1_s8_gbsp_resolution_512_native64_bw2 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_resolution_512_native64_bw2_r12"
GBSP_RESOLUTION_VARIANT = "512-C"
GBSP_PCA_MAX_RANK = 12

