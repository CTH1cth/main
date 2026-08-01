"""DABE-v2 Hard static supervision with unchanged A1 Clean-ECST v5."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)

# Strict single-variable ablation: only the target consumed by the static BCE
# changes.  The inherited A1 v2-contrec cache remains the source of
# dabe_clean_target_68 and recoverability_68 for Clean-ECST routing.
DABE_CLEAN_STATIC_TARGET_SOURCE = "dabe_v2_hard_68"
DABE_CLEAN_DABE_V2_ROOT = "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
DABE_CLEAN_DABE_V2_VERSION = "v2"
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.5
