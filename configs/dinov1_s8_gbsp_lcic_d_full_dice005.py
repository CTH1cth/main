"""LCIC-D with uniform GBSP t=0.58 and a weak soft-Dice loss term."""

from configs.dinov1_s8_gbsp_lcic_d_full import *  # noqa: F401,F403


EXP_NAME = "24-lcic-d-full-dice005"
LCIC_SOFT_DICE_VARIANT = True

# BCE remains the primary loss; this weak term only adds region-overlap signal.
LCIC_SOFT_DICE_WEIGHT = 0.05
