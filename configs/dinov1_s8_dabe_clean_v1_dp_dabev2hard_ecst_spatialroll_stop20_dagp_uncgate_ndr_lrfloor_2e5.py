"""Full-ECST strict deterministic Spatial-Roll control (stop at epoch 20)."""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_spatialroll_stop20_"
    "dagp_uncgate_ndr_lrfloor_2e5"
)

ECST_CAUSAL_CONTROL_MODE = "spatial_roll"
ECST_CAUSAL_CONTROL_SEED = 20260731
STOP_AFTER_EPOCH = 20
