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

# ------------------------------------------------------------
# GKD-Lite: DESPL Quality-Graded Distillation
# default: disabled, should not affect original DESPL-only run
# ------------------------------------------------------------
GKD_MODE = "off"
USE_GKD_LITE = False

# area quality
GKD_AREA_MIN_LOW = 0.005
GKD_AREA_MIN_HIGH = 0.01
GKD_AREA_MAX_HIGH = 0.60
GKD_AREA_MAX_LOW = 0.80

# connected component quality
GKD_CC_HIGH_MAX = 4
GKD_CC_MID_MAX = 8
GKD_LCC_HIGH_MIN = 0.55
GKD_LCC_MID_MIN = 0.35

# edge-touch quality
GKD_EDGE_HIGH_MAX = 0.15
GKD_EDGE_MID_MAX = 0.30

# DESPL-fixed agreement quality
GKD_AGREE_HIGH_MIN = 0.50
GKD_AGREE_MID_MIN = 0.25

# total quality thresholds
GKD_Q_HIGH = 0.75
GKD_Q_NORMAL = 0.45

# sample-level weights
GKD_SAMPLE_W_HIGH = 1.15
GKD_SAMPLE_W_NORMAL = 1.00
GKD_SAMPLE_W_LOW = 0.60

# pixel-level weight
GKD_PIXEL_W_MIN = 0.60

# after epoch21, keep teacher-only target but weaken GKD weighting
GKD_LATE_STRENGTH = 0.30

# logging
GKD_LOG_FIRST_BATCH = True

# audit logging
GKD_AUDIT_WRITE_CSV = True
GKD_AUDIT_LOG_FIRST_BATCH = True
GKD_AUDIT_LOG_LOSS_DELTA = True
GKD_AUDIT_LOG_PIXEL_HIST = True

# branch-loss settings
GKD_BRANCH_LOW_MODE = "teacher_l1"
GKD_BRANCH_HIGH_EXTRA_TARGET = "despl"
GKD_BRANCH_L1_WEIGHT = 1.0
GKD_BRANCH_MSE_WEIGHT = 1.0
GKD_BRANCH_LOW_WEIGHT = 1.0
GKD_BRANCH_NORMAL_WEIGHT = 1.0
GKD_BRANCH_HIGH_WEIGHT = 1.0
GKD_BRANCH_LATE_STRENGTH = 0.30

# entropy pixel weight for branch mode
GKD_USE_ENTROPY_PIXEL_WEIGHT = True
GKD_ENTROPY_START_EPOCH = 2
GKD_ENTROPY_CANDIDATES = ["despl", "teacher", "fixed"]
GKD_ENTROPY_WEIGHT_MIN = 0.50

# reserved pseudo-box background prior, disabled by default
GKD_USE_PSEUDO_BOX_BG = False
GKD_BOX_BG_WEIGHT = 0.05
GKD_BOX_EXPAND_RATIO = 1.20

# ------------------------------------------------------------
# GKD Branch v2-safe grading
# ------------------------------------------------------------
GKD_ENABLE_STRICT_HIGH_GATE = False
GKD_ENABLE_HARD_DOWNGRADE = False
GKD_ENABLE_DYNAMIC_LOW = False

GKD_V2_Q_HIGH = 0.90
GKD_V2_Q_NORMAL = 0.60

GKD_HIGH_AREA_MIN = 0.01
GKD_HIGH_AREA_MAX = 0.60
GKD_HIGH_CC_MAX = 4
GKD_HIGH_LCC_MIN = 0.70
GKD_HIGH_EDGE_MAX = 0.15
GKD_HIGH_AGREE_MIN = 0.50

GKD_DOWNGRADE_CC_MAX = 6
GKD_DOWNGRADE_AGREE_MIN = 0.45
GKD_DOWNGRADE_AREA_MAX = 0.70
GKD_DOWNGRADE_EDGE_MAX = 0.20

GKD_STATIC_LOW_AREA_MIN = 0.003
GKD_STATIC_LOW_AREA_MAX = 0.85
GKD_STATIC_LOW_CC_MAX = 15
GKD_STATIC_LOW_LCC_MIN = 0.20
GKD_STATIC_LOW_EDGE_MAX = 0.45
GKD_STATIC_LOW_AGREE_MIN = 0.15

GKD_DYNAMIC_LOW_START_EPOCH = 6
GKD_TEACHER_DESPL_LOW_IOU = 0.35
GKD_TEACHER_DESPL_NORMAL_IOU = 0.55
GKD_TEACHER_AREA_MIN = 0.005
GKD_TEACHER_AREA_MAX = 0.80

GKD_ENABLE_ENTROPY_GRADE = False
GKD_ENTROPY_LOW_MEAN = 0.65
GKD_ENTROPY_NORMAL_MEAN = 0.50

GKD_LOW_TEACHER_BCE_WEIGHT = 0.3
GKD_LOG_GRADE_TRANSITIONS = True
GKD_LOG_DYNAMIC_LOW_REASONS = True

# ------------------------------------------------------------
# GKD-v3 Conservative, disabled by default
# ------------------------------------------------------------
GKD_ENABLE_DYNAMIC_HIGH_CAP = False
GKD_DISABLE_AFTER_EPOCH = -1

GKD_DYNAMIC_HIGH_CAP_START_EPOCH = 6
GKD_TEACHER_DESPL_HIGH_CAP_IOU = 0.55
GKD_DYNAMIC_LOW_TO_NORMAL_ONLY = True
GKD_V3_LOW_ONLY_STATIC = True

GKD_V3_Q_HIGH = 0.90
GKD_V3_Q_NORMAL = 0.60

GKD_LOW_TEACHER_L1_WEIGHT = 0.3
GKD_LOG_DYNAMIC_HIGH_CAP = True
GKD_LOG_DISABLE_AFTER_EPOCH = True
