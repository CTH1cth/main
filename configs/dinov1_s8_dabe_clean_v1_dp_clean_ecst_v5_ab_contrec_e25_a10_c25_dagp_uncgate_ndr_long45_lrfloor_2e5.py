"""Clean-ECST v5: asymmetric routing plus continuous recoverability."""

from configs.dinov1_s8_dabe_clean_v1_dp_clean_ecst_v4_asym_e25_a10_r25_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
    "e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)

DABE_CLEAN_VERSION = "v2_contrec"
DABE_CLEAN_EXPECTED_PAYLOAD_VERSION = "dabe_clean_v2_contrec"
DABE_CLEAN_SOURCE_ROOT = (
    "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_ROOT = (
    "../datasets/cache/dabe_clean_v2_contrec_pseudo_cache/dinov1-s8"
)

ECST_CLEAN_VERSION = "v5_asym_continuous_recoverability"
ECST_CLEAN_START_EPOCH = 5
ECST_CLEAN_RAMP_END_EPOCH = 10
ECST_CLEAN_STOP_EPOCH = 21

ECST_CLEAN_STRENGTH_MODE = "directional_continuous"
ECST_CLEAN_ERASE_STRENGTH = 2.5
ECST_CLEAN_ADD_STRENGTH = 1.0
ECST_CLEAN_RECOVERY_STRENGTH = 2.5

ECST_CLEAN_USE_HARD_RING = False
ECST_CLEAN_RECOVERY_MODE = "latent_rw_geometric_mean"

# v5 does not expose or consume any geometric Ring parameter.
del ECST_CLEAN_RING_RADIUS
del ECST_CLEAN_RING_BG_STRENGTH
