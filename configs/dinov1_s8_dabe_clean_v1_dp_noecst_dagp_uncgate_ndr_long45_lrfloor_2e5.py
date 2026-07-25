"""DABE-Clean DP Long45 with identity teacher routing."""

from configs.dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_noecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)
USE_ECST = False
TEACHER_ROUTING_MODE = "none"
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False
STOP_AFTER_EPOCH = 0

