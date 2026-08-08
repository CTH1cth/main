"""CF-BRC-HC hard pseudo-label -> one 1x1 convolution, no Teacher."""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_cf_brc_hc_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

GBSP_CF_BRC_HC_LINEAR = True
GBSP_THRESHOLD_VERSION = "gbsp_cf_brc_hc_v1"
GBSP_THRESHOLD_GT_FREE = True
GBSP_THRESHOLD_METHOD = "cf_brc_hc"

DABE_CLEAN_DABE_V2_ROOT = "../workdir/gbsp_threshold/crossfit_train4040"
DABE_CLEAN_STATIC_TARGET_SOURCE = "gbsp_cf_brc_hc_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "hc_mask"
DABE_CLEAN_GBSP_THRESHOLD_VERSION = "gbsp_cf_brc_hc_v1"
DABE_CLEAN_GBSP_AUGS = ["identity"]
R1_HARD_SOURCE_KEY = "hc_mask"
R1_HARD_RESIZE_MODE = "bilinear_align_corners_false"
R1_HARD_THRESHOLD = 0.5
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.5

# Exactly one trainable 1x1 convolution; no Teacher/DAGP/NDR/auxiliary branch.
DABEV2HARD_R1_LINEAR = True
DABEV2HARD_PURE_STUDENT = True
R1_ONLY_CACHE_IO = True
HEAD_TYPE = "simple"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False
FINETUNE_RESET_TEACHER = False
