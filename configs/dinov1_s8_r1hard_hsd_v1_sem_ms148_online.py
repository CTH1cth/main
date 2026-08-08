"""Formal online R1-HSD v1 semantic decoder with 37/74/148 supervision."""

from configs.dinov1_s8_r1hard_hsd_v1_sem_148 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_hsd_v1_sem_ms148_online"
R1_FORMAL_ONLINE_V1 = True
ONLINE_DINO_LAST4 = True
DINO_FEATURE_MODE = "online_last4"

HSD_COARSE_SIZE = 74
HSD_SPATIAL_SUPERVISION_SIZES = [37, 74, 148]

# Static R1 never changes, so optimizer state and auxiliary weights never
# switch at an inherited Teacher-handover boundary.
FINETUNE_RESET_EPOCH = 0
FINETUNE_RESET_REBUILD_OPTIMIZER = False
FINETUNE_RESET_REBUILD_SCHEDULER = False
FINETUNE_RESET_GLOBAL_STEP = False
FINETUNE_RESET_FORCE_LR_FLOOR = False
FINETUNE_RESET_TEACHER = False
LR_FLOOR_APPLY_AFTER_FINETUNE_RESET = False
LR_LINEAR_STAGE1_EPOCHS = MAX_EPOCH
LAMBDA_BASE_AUX_AFTER_RESET = LAMBDA_BASE_AUX

