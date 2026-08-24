"""Unified R32 LCIC control with a 0.10 Soft-Dice loss weight.

The established BW2-P30, uniform-t0.50 pseudo target, LCIC-D-FULL decoder,
optimizer, learning-rate schedule, and seed are unchanged.  Only the weak
region-overlap loss weight is increased from 0.05 to 0.10.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050 import *  # noqa: F401,F403


EXP_NAME = "32-gbsp-r32-lcic-d-full-dice010-t050"

LCIC_SOFT_DICE_010_VARIANT = True
LCIC_SOFT_DICE_WEIGHT = 0.10
