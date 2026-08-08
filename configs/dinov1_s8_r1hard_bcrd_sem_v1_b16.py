"""BCRD-Sem v1 over online DINOv1-S/8 f9--f12 and direct Hard-R1_37."""

from configs.dinov1_s8_r1hard_last4_linear_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_bcrd_sem_v1_b16"
R1_DECODER_ISOLATION_V1 = True
R1_DECODER_DIRECT_R1_37 = True
R1_DECODER_REFERENCE_CONFIG = "configs/dinov1_s8_r1hard_last4_linear_online.py"
BATCH_SIZE = 16

R1_LAST4_EXPERIMENT = False
R1_HSD_V1 = False
DECODER_TYPE = "bcrd_sem_v1"
HEAD_TYPE = "bcrd_sem_v1"
DINO_FEATURE_KEYS = ["f9", "f10", "f11", "f12"]

BCRD_DIM = 32
BCRD_GN_GROUPS = 4
BCRD_CONSISTENCY_TAU = 0.10
BCRD_ALPHA = 0.25
BCRD_SUPERVISION_MODE = "final_base"
USE_BASE_AUX_LOSS = True
USE_NDR_COARSE_AUX = False
LAMBDA_NDR_COARSE_AUX = 0.0
LAMBDA_BASE_AUX = 0.25
LAMBDA_BASE_AUX_AFTER_RESET = 0.25
USE_DETAIL = False

