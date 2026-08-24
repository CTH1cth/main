"""Uniform-t0.50 control for R32 LCIC-D-FULL plus 0.05 Soft-Dice."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_camo050_cod10k058 import *  # noqa: F401,F403


EXP_NAME = "26-gbsp-r32-lcic-d-full-dice005-uniform050"

LCIC_UNIFORM_T050_VARIANT = True
DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET = {
    "TR-CAMO": 0.50,
    "TR-COD10K": 0.50,
}

GBSP_THRESHOLD_SELECTION_SPLIT = "train4040_oracle_gt68_r32_uniform050"
