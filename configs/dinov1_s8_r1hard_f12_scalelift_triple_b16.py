"""F12-only 37->74->148 Scale-Lift with triple R1 supervision."""

from configs.dinov1_s8_r1hard_last4_linear_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_f12_scalelift_triple_b16"
R1_DECODER_ISOLATION_V1 = True
R1_DECODER_DIRECT_R1_37 = True
R1_DECODER_REFERENCE_CONFIG = "configs/dinov1_s8_r1hard_last4_linear_online.py"
BATCH_SIZE = 16

R1_LAST4_EXPERIMENT = False
R1_HSD_V1 = False
DECODER_TYPE = "f12_scalelift"
HEAD_TYPE = "f12_scalelift"
DINO_FEATURE_KEYS = ["f12"]

SCALE_LIFT_DIM = 64
SCALE_LIFT_GN_GROUPS = 8
SCALE_LIFT_SUPERVISION_MODE = "triple"
USE_BASE_AUX_LOSS = True
USE_NDR_COARSE_AUX = True
LAMBDA_NDR_COARSE_AUX = 0.5
LAMBDA_BASE_AUX = 0.5
LAMBDA_BASE_AUX_AFTER_RESET = 0.5
USE_DETAIL = False

