"""Epoch-3 fork: freeze the LCIC anchor and update corrections for epoch 4."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "26-gbsp-r32-lcic-d-full-dice005-uniform050-seed2027-"
    "structonly-e3"
)
LCIC_STRUCTURAL_ONLY_E3 = True
LCIC_STRUCTURAL_ONLY_REQUIRED_RESUME_EPOCH = 3
LCIC_STRUCTURAL_ONLY_REQUIRED_SOURCE_EXP = (
    "26-gbsp-r32-lcic-d-full-dice005-uniform050-seed2027"
)
STOP_AFTER_EPOCH = 4
