"""Clean-ECST v4 directional asymmetric-strength ablation."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v2_r2_s15_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v4_asym_e25_a10_r25_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

ECST_CLEAN_VERSION = "v4_directional_asymmetric_strength"

ECST_CLEAN_START_EPOCH = 5
ECST_CLEAN_RAMP_END_EPOCH = 10
ECST_CLEAN_STOP_EPOCH = 21

ECST_CLEAN_RING_RADIUS = 2

ECST_CLEAN_STRENGTH_MODE = "directional"
ECST_CLEAN_ERASE_STRENGTH = 2.5
ECST_CLEAN_ADD_STRENGTH = 1.0
ECST_CLEAN_RING_BG_STRENGTH = 2.5

# The v4 path must not expose or consume v2's unified strength field.
del ECST_CLEAN_STRENGTH_MAX
