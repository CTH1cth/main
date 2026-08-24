"""Offline graph-weight diagnostic: BW1-P30 with weaker semantic penalty."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050 import *  # noqa: F401,F403


EXP_NAME = "diag-gbsp-r32-bw1-p30-sigmaf115"
DABE_SIGMA_F = 0.115
GBSP_GRAPH_WEIGHT_DIAGNOSTIC = True
