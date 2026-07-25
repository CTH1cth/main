"""Six-epoch diagnostic run for the Bridge target; MAX_EPOCH stays 45."""

from configs.dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_bridge_v1_ecst_dagp_uncgate_ndr_"
    "stop6_lrfloor_2e5"
)
DABE_CLEAN_TARGET_MODE = "bridge"
DABE_CLEAN_ROOT = "../datasets/cache/dabe_bridge_v1_pseudo_cache/dinov1-s8"
STOP_AFTER_EPOCH = 6

