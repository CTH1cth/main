"""A1: remove only the weak BC multiplier from DABE-v2 foreground scores."""

from configs.dinov1_s8_dabe_clean_v1_dp_noecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

# Strict A1 single-variable ablation:
#   fg_score = residual * (1 - BC_LAMBDA * BC) -> residual
# All other DABE-v2, Clean-DP, teacher and optimization definitions are inherited.
DABE_BC_LAMBDA = 0.0
DABE_CLEAN_ABLATION = "a1_residual_only_no_weak_bc_multiplier"

# A1 uses one enriched Clean cache containing both the final target and the
# source-side maps required by the direct baseline-vs-A1 audit.
DABE_CLEAN_ROOT = (
    "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_SOURCE_ROOT = DABE_CLEAN_ROOT

STOP_AFTER_EPOCH = 0
