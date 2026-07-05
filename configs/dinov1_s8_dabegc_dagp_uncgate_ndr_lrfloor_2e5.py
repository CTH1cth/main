from configs.dinov1_s8_despl_dagp_uncgate_ndr_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_dabegc_dagp_uncgate_ndr_lrfloor_2e5"

USE_DABE_PSEUDO = True
DABE_VERSION = "gc"
DABE_PSEUDO_ROOT = "../datasets/cache/dabe_gc_pseudo_cache/dinov1-s8"

P_INIT_MODE = "dabe_gc_only"
P_INIT_DABE_WEIGHT = 1.0
P_INIT_DESPL_WEIGHT = 0.0
P_INIT_FIXED_WEIGHT = 0.0

# Keep DESPL light cache only for pseudo_fixed/pseudo_despl diagnostics.
USE_DESPL_PSEUDO = True
USE_DESPL_LIGHT_CACHE = True

# Preserve the original DESPL-best training protocol.
TEACHER_FUSION_MODE = "default"
MAX_EPOCH = 25
FINETUNE_RESET_EPOCH = 21
FINETUNE_RESET_REBUILD_OPTIMIZER = True
FINETUNE_RESET_REBUILD_SCHEDULER = True
FINETUNE_RESET_GLOBAL_STEP = True

LR_POLICY = "step_floor"
USE_LR_FLOOR = True
LR_FLOOR = 2e-5

# Explicitly disable later experimental branches for this ablation.
USE_DABE_AWARE_LOSS = False
DABE_AWARE_USE_TRIMAP = False
DABE_AWARE_CORE_LOCK = False
DABE_AWARE_WEIGHTED_BCE = False
USE_DABE_TVERSKY_LOSS = False
USE_DABE_AREA_GUARD = False

USE_PROTO_CONTRAST = False
LAMBDA_PROTO_MAX = 0.0
USE_MULTI_VIEW_FEATURE = False
MULTI_VIEW_TYPES = []
USE_VIEW_CONSISTENCY = False
LAMBDA_VIEW_MAX = 0.0

GKD_MODE = "off"
USE_GKD_LITE = False
USE_TADR_ROUTER = False
