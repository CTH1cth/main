"""Hard-R1 -> one 1x1 convolution, with no Teacher.

The cached first-pass residual is resized from 37x37 to 68x68 and thresholded
strictly at 0.5.  That binary mask is the fixed full-weight target for all 45
epochs.  The model is exactly ``SimpleConvSegHead`` (one 1x1 convolution).
"""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

DABEV2HARD_R1_LINEAR = True
DABEV2HARD_PURE_STUDENT = True

# This pure-Student experiment only consumes the independent R1 cache.  Keep
# USE_DABE_CLEAN=True for the established static-BCE training protocol, but do
# not audit/load the legacy DABE-Clean payloads that cannot enter its loss.
R1_ONLY_CACHE_IO = True

# Static label: Hard(bilinear(residual_pass1_37, 68), threshold > 0.5).
DABE_CLEAN_STATIC_TARGET_SOURCE = "dabe_v2_r1_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "residual_pass1_37"
R1_HARD_SOURCE_KEY = "residual_pass1_37"
R1_HARD_RESIZE_MODE = "bilinear_align_corners_false"
R1_HARD_THRESHOLD = 0.5

# Exactly one trainable 1x1 convolution; no DAGP/NDR/auxiliary decoder branch.
HEAD_TYPE = "simple"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False

# There is no Teacher module/forward/EMA on this experiment path.
FINETUNE_RESET_TEACHER = False
