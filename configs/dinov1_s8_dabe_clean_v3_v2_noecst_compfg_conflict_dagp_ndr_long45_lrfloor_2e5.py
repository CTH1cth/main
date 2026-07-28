"""V2: complementary foreground plus conflict-calibrated commitment."""

from configs.dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_dabe_clean_v3_v2_noecst_compfg_conflict_dagp_ndr_long45_lrfloor_2e5"
DABE_CLEAN_VERSION = "v3_offline_consolidation"
DABE_CLEAN_EXPECTED_PAYLOAD_VERSION = "dabe_clean_v3_offline_consolidation"
DABE_CLEAN_OFFLINE_MODE = "complementary_fg_conflict"
DABE_CLEAN_ROOT = (
    "../datasets/cache/dabe_clean_v3_v2_compfg_conflict_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_OFFLINE_SOURCE_ROOT = (
    "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_OFFLINE_SEMANTIC_ROOT = (
    "../datasets/cache/dabe_clean_v2_contrec_a1_residual_only_pseudo_cache/"
    "dinov1-s8"
)
DABE_BC_LAMBDA = 0.0
DABE_CLEAN_SOURCE_ROOT = DABE_CLEAN_ROOT
DABE_PU_ROOT = ""
DABE_CLEAN_LEGACY_REGION_ROOT = ""
DABE_CLEAN_LEGACY_ROUTING_ONLY = False
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False
USE_DABE_PU = False
USE_ECST = False
USE_ECST_MINIMAL = False
USE_ECST_CLEAN = False
TEACHER_ROUTING_MODE = "none"
USE_RAST = False
USE_ESA_ASYM = False
STOP_AFTER_EPOCH = 0
