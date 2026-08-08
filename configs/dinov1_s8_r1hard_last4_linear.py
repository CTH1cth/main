"""R1-hard + frozen DINOv1-S/8 last-four single-linear probe."""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_last4_linear"
R1_LAST4_EXPERIMENT = True
R1_HSD_V1 = False
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

DECODER_TYPE = "last4_linear"
HEAD_TYPE = "last4_linear_probe"
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_NDR_V2 = False
USE_BASE_AUX_LOSS = False
USE_DETAIL = False

FINETUNE_RESET_TEACHER = False

