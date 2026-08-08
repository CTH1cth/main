"""Existing HSD-Sem 37->74->148 decoder with final/base R1 supervision."""

from configs.dinov1_s8_r1hard_last4_linear_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_hsd_v1_sem_finalbase_b16"
R1_DECODER_ISOLATION_V1 = True
R1_DECODER_DIRECT_R1_37 = True
R1_DECODER_REFERENCE_CONFIG = "configs/dinov1_s8_r1hard_last4_linear_online.py"
BATCH_SIZE = 16

R1_LAST4_EXPERIMENT = False
R1_HSD_V1 = True
R1_HSD_VARIANT = "semantic"
DECODER_TYPE = "hsd_v1"
HEAD_TYPE = "hsd_v1"

HSD_SEMANTIC_CHANNELS = 64
HSD_DETAIL_CHANNELS = 32
HSD_GN_GROUPS = 8
HSD_OUTPUT_SIZE = 148
HSD_COARSE_SIZE = 74
HSD_SPATIAL_SUPERVISION_SIZES = [37, 74, 148]
HSD_SUPERVISION_MODE = "final_base"
USE_DETAIL = False

USE_BASE_AUX_LOSS = True
USE_NDR_COARSE_AUX = False
LAMBDA_NDR_COARSE_AUX = 0.0
LAMBDA_BASE_AUX = 0.25
LAMBDA_BASE_AUX_AFTER_RESET = 0.25

