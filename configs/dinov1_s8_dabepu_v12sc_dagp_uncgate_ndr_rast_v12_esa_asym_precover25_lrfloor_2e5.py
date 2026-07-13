from configs.dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long45_lceg_v1_cc_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_dabepu_v12sc_dagp_uncgate_ndr_rast_v12_esa_asym_precover25_lrfloor_2e5"

MAX_EPOCH = 25
max_epoch = 25

SAVE_EVERY_EPOCH = True
SAVE_EPOCH_CKPT = True
CKPT_SAVE_EVERY_EPOCH = True
SAVE_ALL_EPOCHS = True
SAVE_INTERVAL = 1

USE_DABE_PU = True
DABE_PU_VERSION = "pu_v12_shape_complete"
DABE_PU_ROOT = "../datasets/cache/dabe_pu_v12_shape_cache/dinov1-s8/clean45_cover35_v1"
P_INIT_MODE = "dabe_pu_v12_shape_desplsched"

USE_DABE_PU_DESPL_SCHEDULE = True
USE_DABE_PU_STATIC_LOSS = True
DABE_PU_STATIC_TARGET_MODE = "soft"
USE_DABE_PU_HARD_STATIC_TARGET = False
USE_TEACHER_BINARY_FULL_LOSS = True
USE_TEACHER_SOFT_FULL_LOSS = False
TEACHER_TARGET_MODE = "binary"

TEACHER_FUSION_MODE = "dabe_pu_despl_sched"
DABE_PU_DESPL_STAGE_START = 1
DABE_PU_DESPL_STAGE_END = 20
DABE_PU_DESPL_STATIC_START = 1.0
DABE_PU_DESPL_STATIC_END = 0.05
DABE_PU_DESPL_TEACHER_START = 0.0
DABE_PU_DESPL_TEACHER_END = 0.95
DABE_PU_DESPL_TEACHER_ONLY_START = 21

USE_RAST = True
USE_ESA_ASYM = True
RAST_START_EPOCH = 7
RAST_RAMP_END_EPOCH = 15
RAST_STOP_EPOCH = 21
RAST_POST_RESET_ENABLE = False
ESA_ASYM_START_EPOCH = 7
ESA_ASYM_RAMP_END_EPOCH = 15
ESA_ASYM_STOP_EPOCH = 21

# Precover stage does not use LCEG. It only produces epoch_025 for the new coverage cache.
USE_LCEG = False
LCEG_LAMBDA_MAX = 0.0
LCEG_START_EPOCH = 26
LCEG_RAMP_END_EPOCH = 28
LCEG_STOP_EPOCH = 36

# Keep the base model as DAGP-Safe + NDR-v1; no v2 patch branch in this source-pseudo experiment.
USE_NDR_BRANCH = True
USE_NDR_V2 = False
NDR_VERSION = "v1_native_detail_residual"

USE_TCE = False
TCE_LAMBDA_MAX = 0.0
USE_HBNS_LITE = False
HBNS_LAMBDA_MAX = 0.0
USE_EPR_POS = False
EPR_LAMBDA_MAX = 0.0
USE_DARE = False
USE_PROTO_CONTRAST = False
LAMBDA_PROTO_MAX = 0.0
USE_DABE_AWARE_LOSS = False
USE_DABE_TVERSKY_LOSS = False
USE_DABE_AREA_GUARD = False
GKD_MODE = "off"
USE_GKD_LITE = False
USE_TADR_ROUTER = False
