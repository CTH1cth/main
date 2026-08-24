"""Uniform-t0.54 pseudo-label ablation for the matched R32 1x1 control."""

from configs.dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "33-gbsp-r32-1x1-dice005-t054-seed2027"

# Keep every training/evaluation setting from the t0.50 linear control and
# change only the training pseudo-label threshold for both source datasets.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = False
GBSP_R32_LINEAR_DICE005_T054_SEED2027 = True
DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET = {
    "TR-CAMO": 0.54,
    "TR-COD10K": 0.54,
}
GBSP_THRESHOLD_SELECTION_SPLIT = "train4040_oracle_gt68_r32_uniform054"
