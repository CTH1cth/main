"""Oracle diagnostic: Last4 Linear supervised by strict training GT_37."""

from configs.dinov1_s8_r1hard_last4_linear_b16 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gt37_last4_linear_b16"
GT_DIAGNOSTIC_SUPERVISION = True
GT_DIAGNOSTIC_REFERENCE_CONFIG = "configs/dinov1_s8_r1hard_last4_linear_b16.py"
DECODER_SUPERVISION_SOURCE = "gt_hard_37"
GT_DIAGNOSTIC_RESIZE_MODE = "nearest_to_37_then_nearest_lift"
GT_DIAGNOSTIC_STRICT_BINARY = True
