"""No-Soft-Dice ablation for the seed-2027 R32 uniform-t0.50 1x1 run."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_t050_nodice_seed2027 import *  # noqa: F401,F403


EXP_NAME = "33-gbsp-r32-1x1-nodice-t050-seed2027"

# Decoder-only control matched to the audited LCIC No-Dice configuration.
GBSP_R32_LINEAR_NODICE_T050_SEED2027 = True
HEAD_TYPE = "simple"
GBSP_LCIC_V1 = False
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False
