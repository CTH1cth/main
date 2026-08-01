"""Full-ECST strict global gradient-magnitude control (stop at epoch 20)."""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_gmg_stop20_"
    "dagp_uncgate_ndr_lrfloor_2e5"
)

ECST_CAUSAL_CONTROL_MODE = "gradient_magnitude_global"
ECST_CAUSAL_GRADIENT_NORM = "l1"
STOP_AFTER_EPOCH = 20
