"""R1-hard + R1-HSD v1 semantic-only decoder at 148x148."""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_hsd_v1_sem_148"
R1_HSD_V1 = True
R1_HSD_VARIANT = "semantic"
DABEV2HARD_PURE_STUDENT = True

PSEUDO_LABEL_MODE = "r1_hard"
PSEUDO_LABEL_THRESHOLD = 0.5
R1_HARD_SOURCE_KEY = "residual_pass1_37"
R1_HARD_THRESHOLD = 0.5

DINO_FEATURE_MODE = "last4"
DINO_FEATURE_KEYS = ["f9", "f10", "f11", "f12"]
USE_MULTI_LEVEL_FEATURE = True
MULTI_LEVEL_LAYERS = [9, 10, 11, 12]
MULTI_LEVEL_FEATURE_TYPE = "attention_key_projection"
MULTI_LEVEL_FEATURE_DTYPE = "float32"
LAST4_FEATURE_CACHE_ROOT = "../datasets/cache/dinov1_s8_last4_296"
ML_FEATURE_PREFLIGHT_MODE = "sample"
ML_FEATURE_PREFLIGHT_SAMPLES = 32

DECODER_TYPE = "hsd_v1"
HEAD_TYPE = "hsd_v1"
HSD_SEMANTIC_CHANNELS = 64
HSD_DETAIL_CHANNELS = 32
HSD_GN_GROUPS = 8
HSD_OUTPUT_SIZE = 148
USE_DETAIL = False

# Keep the established DAGP-NDR three-way weights and normalized weighted sum.
USE_NDR_COARSE_AUX = True
USE_BASE_AUX_LOSS = True
LAMBDA_NDR_COARSE_AUX = 0.5

# The old decoder family is inactive.
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_NDR_V2 = False
USE_CSD_DECODER = False
USE_CSD_V1R = False
USE_CACD = False
FINETUNE_RESET_TEACHER = False

