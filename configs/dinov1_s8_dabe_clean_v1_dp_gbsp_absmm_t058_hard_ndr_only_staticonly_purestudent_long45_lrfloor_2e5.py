"""GBSP-r8 t=0.58 hard supervision -> NDR-v1 only, no DAGP.

The pseudo-label and pure-Student protocol are identical to the audited
GBSP-r8/t=0.58 1x1 control.  The decoder contains a native 1x1 coarse head
plus NDR-v1.  DINO graph propagation is structurally bypassed.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_ndr_only_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

GBSP_ABSMM_T058_LINEAR = False
GBSP_ABSMM_T058_DAGP_ONLY = False
GBSP_ABSMM_T058_NDR_ONLY = True
DABEV2HARD_R1_LINEAR = False
DABEV2HARD_R1_DAGP_NDR = False
DAGP_NDR_FULL_FROM_EPOCH1 = False
DAGP_ONLY_FULL_FROM_EPOCH1 = False
NDR_ONLY_FULL_FROM_EPOCH1 = True

# Dedicated NDR-only construction.  No DAGP graph is executed or trained.
HEAD_TYPE = "ndr_only"
USE_DAGP_HEAD = False
USE_DAGP_SAFE_HEAD = False
DAGP_SAFE_ALPHA_MAX = 0.0
DAGP_SAFE_GAMMA_MAX = 0.0
DAGP_SAFE_USE_PROB_GATE = False
DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE = False

# Original NDR-v1, fully active from epoch one.
USE_NDR_BRANCH = True
USE_NDR_V2 = False
NDR_VERSION = "v1_native_detail_residual"
NDR_WARMUP_EPOCH = 0
NDR_RAMP_START_EPOCH = 1
NDR_RAMP_END_EPOCH = 1
USE_NDR_COARSE_AUX = True
LAMBDA_NDR_COARSE_AUX = 0.5
USE_BASE_AUX_LOSS = True

# Pure Student: no Teacher construction/forward/EMA supervision.
FINETUNE_RESET_TEACHER = False
