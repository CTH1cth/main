from configs.dinov1_s8_despl_only_dagp_safe_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_dagp_safe_uncgate_lrfloor_2e5"

HEAD_TYPE = "dagp_safe"
USE_DAGP_SAFE_HEAD = True

USE_LR_FLOOR = True
LR_FLOOR = 2e-5

DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE = True
DAGP_SAFE_UNCERTAINTY_POWER = 1.0
DAGP_SAFE_UNCERTAINTY_MIN = 0.0
DAGP_SAFE_UNCERTAINTY_MAX = 1.0
DAGP_SAFE_UNCERTAINTY_DETACH = True
