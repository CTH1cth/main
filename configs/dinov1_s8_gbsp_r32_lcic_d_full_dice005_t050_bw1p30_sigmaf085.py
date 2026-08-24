"""Unified BW1-P30/SIGMA_F=0.085 GBSP target with the established R32 LCIC.

Both training datasets use the same graph, candidate, PCA, threshold, decoder,
loss, optimizer, and schedule settings.  Only the frozen pseudo-target cache is
replaced relative to the numbered 26 uniform-t0.50 control.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050 import *  # noqa: F401,F403


EXP_NAME = "30-gbsp-r32-lcic-d-full-dice005-t050-bw1-p30-sf085"

LCIC_R32_GRAPH_CANDIDATE_VARIANT = True
GBSP_VERSION = "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1"
GBSP_CANDIDATE_BORDER_WIDTH = 1
GBSP_CANDIDATE_TOP_PERCENT = 30.0
GBSP_GRAPH_SIGMA_F = 0.085
DABE_SIGMA_F = 0.085

DABE_CLEAN_DABE_V2_ROOT = (
    "../datasets/cache/gbsp_r32_graph_sigmaf_sweep_v1/"
    "sigmaf085/bw1-p30/dinov1-s8"
)
DABE_CLEAN_GBSP_VERSION = "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1"
DABE_CLEAN_GBSP_GRAPH_CANDIDATE_DIRECT = True

GBSP_THRESHOLD_SELECTION_SPLIT = (
    "train4040_oracle_gt68_r32_bw1_p30_sigmaf085_uniform050"
)
