"""Offline graph diagnostic: BW2-P30 with stronger Sobel edge penalty."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050 import *  # noqa: F401,F403


EXP_NAME = "diag-gbsp-r32-bw2-p30-sigmae024"
DABE_SIGMA_E = 0.24
GBSP_GRAPH_WEIGHT_DIAGNOSTIC = True
