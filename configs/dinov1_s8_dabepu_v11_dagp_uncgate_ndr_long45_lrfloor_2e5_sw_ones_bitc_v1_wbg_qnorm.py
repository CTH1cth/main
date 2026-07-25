from configs.dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long45_"
    "lrfloor_2e5_sw_ones_bitc_v1_wbg_qnorm"
)

# BITC is the only Teacher pixel router; inherited ECST fields remain inert.
USE_ECST = False
USE_TEPR_LITE = False
USE_BITC = True
TEACHER_ROUTING_MODE = "bitc_v1"
STATIC_WEIGHT_MODE = "ones"
DABE_PU_STATIC_SOURCE = "target_soft_68"

BITC_VERSION = "v1"
BITC_SUBSTITUTION_MODE = "weighted_bg_query_norm"
BITC_CACHE_ROOT = "../datasets/cache/dabe_bg_intervention_v1/dinov1-s8"
BITC_CACHE_SEED = 20260724
BITC_NUM_GROUPS = 2
BITC_GROUP_MODE = "checkerboard_2"
BITC_BATCH_INTERVENTIONS = True
BITC_USE_COARSE_ONLY = True
BITC_CENTER_MODE = "non_intervened_median"
BITC_RESPONSE_NORMALIZATION = "per_image_mad"
BITC_RESPONSE_CLIP = 6.0
BITC_WEIGHT_FLOOR = 0.20
BITC_APPLY_TO_FINAL = True
BITC_APPLY_TO_COARSE = True
BITC_APPLY_TO_BASE = True
BITC_WEIGHTED_NORMALIZE = True
BITC_DETACH_RESPONSE = True
BITC_DETACH_MAP = True
BITC_DEBUG = False
BITC_DEBUG_MAX_IMAGES = 5
BITC_LOG_INTERVAL = 1
