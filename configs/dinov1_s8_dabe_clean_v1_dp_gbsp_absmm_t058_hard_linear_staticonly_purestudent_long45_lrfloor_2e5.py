"""GBSP Absolute-MinMax t=0.58 -> one 1x1 convolution, no Teacher.

The threshold is an oracle diagnostic selected from the 4040-image training
GT scan.  It must not be presented as GT-free model selection.  The existing
continuous 37x37 GBSP cache is reused; only resize-first hard thresholding is
changed from the t=0.63 control.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

GBSP_ABSMM_T058_LINEAR = True
GBSP_VERSION = "gbsp_pca_absmm_v1"
GBSP_ORACLE_THRESHOLD_VERIFICATION = True
GBSP_THRESHOLD_SELECTION_SPLIT = "train4040_oracle_gt68"
GBSP_SOURCE_DABE_ROOT = (
    "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
)

# Reuse the immutable continuous response.  The loader performs bilinear
# 37->68 first, then applies strict > 0.58.
DABE_CLEAN_DABE_V2_ROOT = (
    "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
)
DABE_CLEAN_STATIC_TARGET_SOURCE = "gbsp_abs_minmax_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "gbsp_abs_minmax_37"
DABE_CLEAN_GBSP_VERSION = "gbsp_pca_absmm_v1"
DABE_CLEAN_GBSP_AUGS = ["identity"]
R1_HARD_SOURCE_KEY = "gbsp_abs_minmax_37"
R1_HARD_RESIZE_MODE = "bilinear_align_corners_false"
R1_HARD_THRESHOLD = 0.58
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.58

# Exactly one trainable 1x1 convolution; no Teacher/DAGP/NDR/auxiliary branch.
DABEV2HARD_R1_LINEAR = True
DABEV2HARD_PURE_STUDENT = True
R1_ONLY_CACHE_IO = True
HEAD_TYPE = "simple"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False
FINETUNE_RESET_TEACHER = False
