"""No-Soft-Dice ablation for the seed-2027 R32 uniform-t0.50 LCIC-D run."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "26-gbsp-r32-lcic-d-full-nodice-uniform050-seed2027"

LCIC_NO_SOFT_DICE_VARIANT = True
LCIC_SOFT_DICE_VARIANT = False
LCIC_SOFT_DICE_WEIGHT = 0.0
