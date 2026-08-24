"""512-C GBSP t=0.60 -> native-64 one-layer 1x1 pure Student.

The frozen DINOv1-S/8 key feature is extracted at 512x512, so both the
decoder input and the static hard target live on the native 64x64 patch grid.
No Teacher, DAGP, NDR, graph propagation, or legacy DABE-v2 cache is used.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t060_hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp512c_t060_native64_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

GBSP512C_T060_NATIVE64_LINEAR = True
GBSP_VERSION = "gbsp_resolution_512c_native64_v1"
GBSP_THRESHOLD_SELECTION_SPLIT = "full512c_threshold_sensitivity_t060"
GBSP_SOURCE_DABE_ROOT = ""

# Frozen 512-input / native-64 DINO key features.
CACHE_ROOT = "../workdir/gbsp_resolution_512/train512c/cache"
DINO = dict(DINO)
DINO["feature_input_size"] = 512
FEATURE_CACHE_KEY = BACKBONE_KEY
FEATURE_INPUT_SIZE = 512
GRID_SIZE = 64
SUPERVISION_GRID = 64
LOSS_SIZE = 64
USE_LEGACY_68_INTERPOLATION = False
DABE_GRID = 64
DABE_LOSS_SIZE = 64

# Direct 512-C construction: Full-BC/BW2/R30/PCA-energy90/cap12.
GBSP_RESOLUTION_VARIANT = "512-C"
GBSP_BORDER_WIDTH = 2
GBSP_BG_RATIO = 0.30
GBSP_PCA_ENERGY = 0.90
GBSP_PCA_MIN_RANK = 1
GBSP_PCA_MAX_RANK = 12
GBSP_THRESHOLD = 0.60
GBSP_GT_USED_FOR_GENERATION = False

# The pseudo cache is produced directly from the DINO512 feature cache.
DABE_CLEAN_DABE_V2_ROOT = (
    "../workdir/gbsp_resolution_512/train512c/pseudo/dinov1-s8"
)
DABE_CLEAN_DABE_V2_SOURCE_KEY = "gbsp_abs_minmax_64"
DABE_CLEAN_GBSP_SOURCE_KEY = "gbsp_abs_minmax_64"
DABE_CLEAN_GBSP_SOURCE_GRID = 64
DABE_CLEAN_GBSP_VERSION = "gbsp_resolution_512c_native64_v1"
DABE_CLEAN_GBSP_DIRECT_FROM_FEATURES = True
DABE_CLEAN_GBSP_AUGS = ["identity"]
R1_HARD_SOURCE_KEY = "gbsp_abs_minmax_64"
R1_HARD_RESIZE_MODE = "native64_no_resize"
R1_HARD_THRESHOLD = 0.60
DABE_CLEAN_DABE_V2_HARD_THRESHOLD = 0.60

# One trainable 1x1 convolution only.
DABEV2HARD_R1_LINEAR = True
DABEV2HARD_PURE_STUDENT = True
R1_ONLY_CACHE_IO = True
HEAD_TYPE = "simple"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False
FINETUNE_RESET_TEACHER = False
