"""Clean-ECST v5 A1: remove only DABE's weak BC foreground multiplier."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)

# A1 changes both DABE foreground-score passes from
# residual * (1 - 0.5 * BC) to residual.  Every Clean-ECST v5 definition is
# inherited unchanged from the authoritative v5 configuration.
DABE_BC_LAMBDA = 0.0
DABE_CLEAN_ABLATION = "a1_residual_only_no_weak_bc_multiplier"

# Reuse the already generated A1 source maps, then derive a separate v2
# continuous-recoverability cache required by Clean-ECST v5.
DABE_CLEAN_SOURCE_ROOT = (
    "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_ROOT = (
    "../datasets/cache/dabe_clean_v2_contrec_a1_residual_only_pseudo_cache/"
    "dinov1-s8"
)

