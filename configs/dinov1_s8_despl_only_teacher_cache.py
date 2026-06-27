from configs.dinov1_s8_despl_teacher_cache import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_teacher_cache"

USE_DESPL_PSEUDO = True
NPER_PSEUDO_BANK_ROOT = "../datasets/cache/nper_pseudo_bank"
USE_DESPL_LIGHT_CACHE = True
DESPL_LIGHT_CACHE_ROOT = "../datasets/cache/despl_blend_pseudo_cache"
DESPL_PSEUDO_SOURCE = "nper_pseudo_bank"

P_INIT_MODE = "despl_only"
P_INIT_DESPL_WEIGHT = 1.0
P_INIT_FIXED_WEIGHT = 0.0

USE_QRA = False
USE_CCR = False
USE_DREPP = False
USE_DRE_SAFE_PRIOR = False
PSEUDO_USE_GCM = False
PSEUDO_USE_TEACHER = False
PSEUDO_USE_LOCAL_ATTN = False

USE_PSTA = False
USE_MNP = False
DINO_USE_LORA = False
USE_CNN_DETAIL_BRANCH = False
USE_AFF = False
USE_DUAL_FG_BG_HEAD = False
USE_BOUNDARY_HEAD = False

LAMBDA_MNP = 0.0
LAMBDA_LOCAL = 0.0
LAMBDA_PSTA = 0.0
LAMBDA_CONTRAST = 0.0
LAMBDA_BOUNDARY = 0.0
LAMBDA_ENTROPY = 0.0
LAMBDA_ANCHOR = 0.0

USE_LATE_DESPL_ANCHOR_LOSS = False
HEAD_TYPE = "simple"
