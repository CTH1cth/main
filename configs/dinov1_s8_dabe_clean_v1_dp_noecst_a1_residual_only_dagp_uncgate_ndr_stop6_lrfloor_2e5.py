"""Six-epoch diagnostic configuration for the A1 residual-only ablation."""

from configs.dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_"
    "dagp_uncgate_ndr_stop6_lrfloor_2e5"
)
STOP_AFTER_EPOCH = 6
