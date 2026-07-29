"""Clean-DP + full ECST with only the static target replaced by DABE-v2 hard."""

from configs.dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)

# Single-variable static-target ablation.  Keep the Clean data/loss path and
# legacy full-ECST routing unchanged; read the exact independent DABE-v2 cache
# used by offline quality audits and harden at the native 68x68 loss grid.
DABE_CLEAN_STATIC_TARGET_SOURCE = "dabe_v2_hard_68"
DABE_CLEAN_DABE_V2_ROOT = "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
DABE_CLEAN_DABE_V2_VERSION = "v2"
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.5
