"""Clean-ECST v3 instant-full schedule ablation."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v2_r2_s15_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v3_instantfull_r2_s15_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

ECST_CLEAN_VERSION = "v3_instant_full_r2_strength15_ablation"

ECST_CLEAN_START_EPOCH = 2
ECST_CLEAN_RAMP_END_EPOCH = 2
ECST_CLEAN_STOP_EPOCH = 21

ECST_CLEAN_RING_RADIUS = 2
ECST_CLEAN_STRENGTH_MAX = 1.5
