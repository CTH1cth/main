"""Six-epoch diagnostic run for the Clean-DP target; MAX_EPOCH stays 45."""

from configs.dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_"
    "stop6_lrfloor_2e5"
)
STOP_AFTER_EPOCH = 6

