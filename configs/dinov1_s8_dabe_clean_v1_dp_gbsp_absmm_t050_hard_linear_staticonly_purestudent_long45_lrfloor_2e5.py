"""GBSP Absolute-MinMax t=0.50 -> one 1x1 convolution, no Teacher.

This is the canonical midpoint-threshold control for the image-wise Min-Max
response.  It reuses the immutable continuous 37x37 GBSP cache, upsamples the
response to 68x68 with bilinear interpolation, and then applies strict > 0.50.
No GT is used to select the threshold.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t050_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

GBSP_ABSMM_T050_LINEAR = True
GBSP_VERSION = "gbsp_pca_absmm_v1"
GBSP_ORACLE_THRESHOLD_VERIFICATION = False
GBSP_THRESHOLD_SELECTION_SPLIT = "predefined_minmax_midpoint"
GBSP_THRESHOLD_GT_FREE = True
GBSP_SOURCE_DABE_ROOT = (
    "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
)

# Reuse the immutable continuous response.  The directory name records the
# historical cache build threshold, but training reads only gbsp_abs_minmax_37.
DABE_CLEAN_DABE_V2_ROOT = (
    "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
)
DABE_CLEAN_STATIC_TARGET_SOURCE = "gbsp_abs_minmax_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "gbsp_abs_minmax_37"
DABE_CLEAN_GBSP_VERSION = "gbsp_pca_absmm_v1"
DABE_CLEAN_GBSP_AUGS = ["identity"]
R1_HARD_SOURCE_KEY = "gbsp_abs_minmax_37"
R1_HARD_RESIZE_MODE = "bilinear_align_corners_false"
R1_HARD_THRESHOLD = 0.50
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.50

# Exactly one trainable 1x1 convolution; no Teacher/DAGP/NDR/auxiliary branch.
DABEV2HARD_R1_LINEAR = True
DABEV2HARD_PURE_STUDENT = True
R1_ONLY_CACHE_IO = True
HEAD_TYPE = "simple"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False
FINETUNE_RESET_TEACHER = False
