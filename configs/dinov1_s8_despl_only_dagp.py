from configs.dinov1_s8_despl_only_teacher_cache import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_dagp"

HEAD_TYPE = "dagp"
USE_DAGP_HEAD = True

DAGP_HIDDEN = 64
DAGP_TOPK = 24
DAGP_TAU = 0.10
DAGP_ALPHA_INIT = 0.05
DAGP_GAMMA_INIT = 0.10
DAGP_NUM_LAYERS = 1
DAGP_AFFINITY_DETACH = True
DAGP_USE_FFN = False
DAGP_USE_DWCONV = False
DAGP_DEBUG_FIRST_BATCH = True
