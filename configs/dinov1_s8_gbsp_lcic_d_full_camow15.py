"""LCIC-D with uniform GBSP t=0.58 and normalized CAMO loss weight 1.5."""

from configs.dinov1_s8_gbsp_lcic_d_full import *  # noqa: F401,F403


EXP_NAME = "24-lcic-d-full-camow15"
LCIC_DATASET_LOSS_WEIGHT_VARIANT = True

# The pseudo-label threshold remains identical for both training datasets.
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.58

# Per-image BCE is averaged with these weights and divided by their batch sum.
TR_CAMO_LOSS_WEIGHT = 1.5
TR_COD10K_LOSS_WEIGHT = 1.0
