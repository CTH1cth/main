"""Unified R32 LCIC control with a lower decoder learning rate.

This experiment inherits the established BW2-P30, uniform-t0.50 pseudo target,
LCIC-D-FULL decoder, and 0.05 Soft-Dice loss.  Only the initial optimizer
learning rate is reduced from 6e-4 to 3e-4.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050 import *  # noqa: F401,F403


EXP_NAME = "31-gbsp-r32-lcic-d-full-dice005-t050-lr0003"

LCIC_LR_0003_VARIANT = True
LR = 3e-4
# The optimizer builder reads DINO["lr"], while LR is the scheduler/audit
# base value.  Keep both representations synchronized for this one-variable
# learning-rate ablation.
DINO = {**DINO, "lr": 3e-4}
