from configs.dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_rast_v12_esa_asym_clean35_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_dabepu_v11_dagp_uncgate_csd_v1r_hr_bfr_v1_rast_v12_esa_asym_clean35_lrfloor_2e5"

# Keep the CSD-v1R coarse anchor protocol unchanged.
HEAD_TYPE = "dagp_safe_csd_v1r"
head_type = "dagp_safe_csd_v1r"
USE_DAGP_SAFE_HEAD = True
USE_CSD_V1R = True
USE_CSD_DECODER = False

MAX_EPOCH = 35
max_epoch = 35
SAVE_EVERY_EPOCH = True
SAVE_INTERVAL = 1

# HR-BFR v1: narrow-band high-resolution boundary refinement.
USE_HR_BFR = True
HR_BFR_VERSION = "v1_narrow_band_136"
HR_BFR_USE_HR_LOGITS_FOR_EVAL = True
# Eval-only probe multiplier. Training forward always forces this value to 1.0.
HR_BFR_EVAL_RES_SCALE = 1.0

HR_BFR_SIZE = 136
HR_BFR_SCALE = 2

HR_BFR_USE_RGB = True
HR_BFR_USE_SOBEL = True
HR_BFR_USE_ANCHOR_PROB = True
HR_BFR_USE_ANCHOR_UNCERT = True
HR_BFR_USE_DINO_SEM = True

HR_BFR_SEM_DIM = 32
HR_BFR_DETAIL_DIM = 32
HR_BFR_HIDDEN_DIM = 32

HR_BFR_RESIDUAL_MODE = True
HR_BFR_RESIDUAL_CLIP = 2.0
HR_BFR_BETA_MAX = 0.05

HR_BFR_WARMUP_EPOCH = 15
HR_BFR_RAMP_START_EPOCH = 16
HR_BFR_RAMP_END_EPOCH = 20

HR_BFR_USE_RES_ZERO_INIT = True
HR_BFR_USE_GATE_BIAS_INIT = True
HR_BFR_GATE_BIAS_INIT = -2.0

HR_BFR_BOUNDARY_SOURCE = "anchor_prob_68"
HR_BFR_BOUNDARY_THRESH = 0.5
HR_BFR_BOUNDARY_RADIUS_68 = 1
HR_BFR_BOUNDARY_DILATE_136 = 1
HR_BFR_BOUNDARY_BAND_ONLY = True
HR_BFR_DETACH_ANCHOR = True
HR_BFR_MAX_BAND_RATIO = 0.35

HR_BFR_USE_BAND_BCE = True
HR_BFR_BAND_BCE_WEIGHT_MAX = 0.05

HR_BFR_USE_OUTBAND_ANCHOR = True
HR_BFR_OUTBAND_ANCHOR_WEIGHT_MAX = 0.10

HR_BFR_USE_BG_PROB_LOCK = True
HR_BFR_BG_PROB_LOCK_WEIGHT_MAX = 0.02
HR_BFR_BG_PROB_LOCK_DELTA = 0.02

HR_BFR_USE_AREA_NEUTRAL = True
HR_BFR_AREA_NEUTRAL_WEIGHT_MAX = 0.02
HR_BFR_AREA_TOL = 0.005

HR_BFR_USE_EDGE_ALIGN = True
HR_BFR_EDGE_ALIGN_WEIGHT_MAX = 0.01
HR_BFR_EDGE_Q = 0.60

# Explicitly keep unrelated branches disabled.
USE_LCEG = False
LCEG_LAMBDA_MAX = 0.0
USE_TCE = False
TCE_LAMBDA_MAX = 0.0
USE_HBNS_LITE = False
HBNS_LAMBDA_MAX = 0.0
USE_EPR_POS = False
EPR_LAMBDA_MAX = 0.0
USE_DARE = False
USE_NDR_BRANCH = False
USE_NDR_V2 = False
USE_PROTO_CONTRAST = False
LAMBDA_PROTO_MAX = 0.0
USE_MULTI_VIEW_FEATURE = False
use_multi_view_feature = False
USE_VIEW_CONSISTENCY = False
use_view_consistency = False
lambda_view_max = 0.0
GKD_MODE = "off"
USE_GKD_LITE = False
USE_TADR_ROUTER = False
