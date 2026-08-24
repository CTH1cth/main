"""GBSP-r8 t=0.58 hard supervision -> DAGP-Safe only, no NDR.

This decoder ablation keeps the completed GBSP-r8 pseudo-label protocol and
all pure-Student training hyperparameters unchanged.  DAGP-Safe is fully
active from epoch one.  The native-detail residual (NDR) branch and its
auxiliary loss are explicitly disabled.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_dagp_only_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

# Dedicated audited selector: the pseudo-label source remains GBSP-r8/t=0.58,
# while the decoder is no longer the linear 1x1 control.
GBSP_ABSMM_T058_LINEAR = False
GBSP_ABSMM_T058_DAGP_ONLY = True
DABEV2HARD_R1_LINEAR = False
DABEV2HARD_R1_DAGP_NDR = False
DAGP_NDR_FULL_FROM_EPOCH1 = False
DAGP_ONLY_FULL_FROM_EPOCH1 = True

# DAGP-Safe is the only decoder refinement.  Its base auxiliary prediction is
# an internal part of the established DAGP head, not an NDR branch.
HEAD_TYPE = "dagp_safe"
USE_DAGP_HEAD = False
USE_DAGP_SAFE_HEAD = True
USE_BASE_AUX_LOSS = True

# Full DAGP strength from the first epoch.
DAGP_SAFE_WARMUP_EPOCH = 0
DAGP_SAFE_RAMP_START_EPOCH = 1
DAGP_SAFE_RAMP_END_EPOCH = 1

# Defense-in-depth: no NDR module, NDR-v2 path, or NDR auxiliary supervision.
USE_NDR_BRANCH = False
USE_NDR_V2 = False
USE_NDR_COARSE_AUX = False
LAMBDA_NDR_COARSE_AUX = 0.0

# Pure Student: no Teacher construction/forward/EMA supervision.
FINETUNE_RESET_TEACHER = False
