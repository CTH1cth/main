from configs.dinov1_s8_despl_only_teacher_cache import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_lr_floor_2e5"

HEAD_TYPE = "simple"
USE_DAGP_HEAD = False

USE_LR_FLOOR = True
LR_FLOOR = 2e-5
LR_FLOOR_MODE = "global"
LR_FLOOR_APPLY_AFTER_SCHEDULER_STEP = True
LR_FLOOR_APPLY_AFTER_FINETUNE_RESET = True

LR_POLICY = "step_floor"
