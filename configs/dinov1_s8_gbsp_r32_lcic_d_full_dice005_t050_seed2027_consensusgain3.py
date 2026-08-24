"""R32 LCIC-D seed-2027 run with a fixed 3x consensus branch gain."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "26-gbsp-r32-lcic-d-full-dice005-uniform050-seed2027-"
    "consensusgain3"
)
LCIC_CONSENSUS_GAIN3_VARIANT = True
LCIC_CONSENSUS_GAIN = 3.0
LCIC_INNOVATION_GAIN = 1.0
