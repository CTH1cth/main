"""C2 ablation: signed static plus latent support, without Teacher history."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_c2_nohist_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)

ECST_CLEAN_VERSION = "c2_signed_latent_no_history"
ECST_CLEAN_USE_HISTORY = False
