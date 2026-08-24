"""LCIC-D with uniform GBSP t=0.58 and a slower iteration StepLR."""

from configs.dinov1_s8_gbsp_lcic_d_full import *  # noqa: F401,F403


EXP_NAME = "24-lcic-d-full-lrstep50"
LCIC_LR_STEP_VARIANT = True

# Keep gamma and every training protocol field unchanged; only delay each
# per-iteration decay from once per 25 optimizer steps to once per 50 steps.
LR_STEP_SIZE = 50
