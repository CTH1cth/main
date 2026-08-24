"""Fixed-R32 GBSP with LCIC-D-FULL and the validated 0.05 Soft-Dice term."""

from configs.dinov1_s8_gbsp_r32_absmm_t058_hard_linear import *  # noqa: F401,F403


EXP_NAME = "25-gbsp-r32-lcic-d-full-dice005-t058"

GBSP_LCIC_V1 = True
LCIC_R32_VARIANT = True
LCIC_VARIANT = "d_full"
LCIC_USE_CONSENSUS = True
LCIC_USE_INNOVATION = True
LCIC_AFFINITY_EPS = 1e-6
LCIC_DWLITE_CHANNELS = 16
HEAD_TYPE = "lcic"

LCIC_SOFT_DICE_VARIANT = True
LCIC_SOFT_DICE_WEIGHT = 0.05
