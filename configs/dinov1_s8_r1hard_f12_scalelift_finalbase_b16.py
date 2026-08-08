"""F12-only 37->74->148 Scale-Lift with final/base R1 supervision."""

from configs.dinov1_s8_r1hard_f12_scalelift_triple_b16 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_f12_scalelift_finalbase_b16"
SCALE_LIFT_SUPERVISION_MODE = "final_base"
USE_NDR_COARSE_AUX = False
LAMBDA_NDR_COARSE_AUX = 0.0
LAMBDA_BASE_AUX = 0.25
LAMBDA_BASE_AUX_AFTER_RESET = 0.25

