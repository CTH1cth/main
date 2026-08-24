"""Hard-R1 static-only supervision for the existing DAGP-Safe + NDR-v1 head.

Only the Student is instantiated.  ``residual_pass1_37`` is bilinearly resized
to 68x68 and thresholded strictly above 0.5; this fixed binary target has full
weight for all 45 epochs.  There is no Teacher forward or EMA update.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

DABEV2HARD_R1_DAGP_NDR = True
DABEV2HARD_PURE_STUDENT = True
DAGP_NDR_FULL_FROM_EPOCH1 = True

DABE_CLEAN_STATIC_TARGET_SOURCE = "dabe_v2_r1_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "residual_pass1_37"
R1_HARD_SOURCE_KEY = "residual_pass1_37"
R1_HARD_RESIZE_MODE = "bilinear_align_corners_false"
R1_HARD_THRESHOLD = 0.5

# Explicitly preserve the current DAGP-Safe + NDR-v1 Student decoder.
HEAD_TYPE = "dagp_safe"
USE_DAGP_SAFE_HEAD = True
USE_NDR_BRANCH = True
USE_NDR_COARSE_AUX = True
USE_BASE_AUX_LOSS = True

# No warmup/ramp: epoch 1 already uses the configured maximum DAGP/NDR scales.
DAGP_SAFE_WARMUP_EPOCH = 0
DAGP_SAFE_RAMP_START_EPOCH = 1
DAGP_SAFE_RAMP_END_EPOCH = 1
NDR_WARMUP_EPOCH = 0
NDR_RAMP_START_EPOCH = 1
NDR_RAMP_END_EPOCH = 1

# Pure-Student training has no Teacher handover and therefore no reset phase.
# All downstream HSD/Last4/DAGP variants inherit this permanent contract.
PURE_STUDENT_RESET_PERMANENTLY_DISABLED = True
FINETUNE_RESET_EPOCH = 0
FINETUNE_RESET_REBUILD_OPTIMIZER = False
FINETUNE_RESET_REBUILD_SCHEDULER = False
FINETUNE_RESET_GLOBAL_STEP = False
FINETUNE_RESET_FORCE_LR_FLOOR = False
FINETUNE_RESET_TEACHER = False
LR_FLOOR_APPLY_AFTER_FINETUNE_RESET = False
