"""Formal online DINOv1-S/8 Last4 single-linear R1 baseline."""

from configs.dinov1_s8_r1hard_last4_linear import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_last4_linear_online"
R1_FORMAL_ONLINE_V1 = True
ONLINE_DINO_LAST4 = True
DINO_FEATURE_MODE = "online_last4"

# One immutable offline target and one continuous optimization phase.
FINETUNE_RESET_EPOCH = 0
FINETUNE_RESET_REBUILD_OPTIMIZER = False
FINETUNE_RESET_REBUILD_SCHEDULER = False
FINETUNE_RESET_GLOBAL_STEP = False
FINETUNE_RESET_FORCE_LR_FLOOR = False
FINETUNE_RESET_TEACHER = False
LR_FLOOR_APPLY_AFTER_FINETUNE_RESET = False
LR_LINEAR_STAGE1_EPOCHS = MAX_EPOCH

