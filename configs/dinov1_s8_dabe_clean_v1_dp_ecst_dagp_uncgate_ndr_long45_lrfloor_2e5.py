"""DABE-Clean DP with the preserved full-ECST Long45 protocol."""

from configs.dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)

# Independent continuous static-supervision cache.
USE_DABE_PU = False
USE_DABE_CLEAN = True
DABE_CLEAN_VERSION = "v1"
DABE_CLEAN_TARGET_MODE = "dp"
DABE_CLEAN_ROOT = "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8"
DABE_CLEAN_STATIC_WEIGHT_MODE = "ones"

P_INIT_MODE = "dabe_clean_v1_desplsched"
TEACHER_FUSION_MODE = "dabe_clean_despl_sched"
USE_DABE_PU_DESPL_SCHEDULE = False
USE_DABE_CLEAN_DESPL_SCHEDULE = True
STATIC_WEIGHT_MODE = "ones"

# Full ECST reads only the five namespaced legacy routing regions.
USE_ECST = True
USE_ECST_MINIMAL = False
TEACHER_ROUTING_MODE = "ecst"
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = True
DABE_CLEAN_LEGACY_REGION_ROOT = (
    "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
)

STOP_AFTER_EPOCH = 0

