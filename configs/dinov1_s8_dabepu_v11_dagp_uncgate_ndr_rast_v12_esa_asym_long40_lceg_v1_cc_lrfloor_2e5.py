from configs.dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long35_lceg_v1_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_long40_lceg_v1_cc_lrfloor_2e5"

MAX_EPOCH = 40
max_epoch = 40

SAVE_EVERY_EPOCH = True
SAVE_EPOCH_CKPT = True
CKPT_SAVE_EVERY_EPOCH = True
SAVE_ALL_EPOCHS = True
SAVE_INTERVAL = 1

# Lock Long35/LCEG-v1 protocol. Epoch36-40 are clean teacher-only.
DINO = dict(DINO)
DINO["lr"] = 0.0006
LR = 0.0006
lr = 0.0006
lr0 = 0.0006
LR_POLICY = "step_floor"
lr_policy = "step_floor"
USE_LR_FLOOR = True
use_lr_floor = True
LR_FLOOR = 2e-5
lr_floor = 2e-5
LR_LINEAR_STAGE1_EPOCHS = 19
lr_linear_stage1_epochs = 19
LR_LINEAR_STAGE2_EPOCHS = 0
lr_linear_stage2_epochs = 0
LR_FLOOR_MODE = "global"
lr_floor_mode = "global"
LR_FLOOR_APPLY_AFTER_SCHEDULER_STEP = True
lr_floor_apply_after_scheduler_step = True
LR_FLOOR_APPLY_AFTER_FINETUNE_RESET = True
lr_floor_apply_after_finetune_reset = True

FINETUNE_RESET_EPOCH = 20
finetune_reset_epoch = 20
FINETUNE_RESET_TIMING = "after_epoch"
finetune_reset_timing = "after_epoch"
FINETUNE_RESET_REBUILD_OPTIMIZER = True
finetune_reset_rebuild_optimizer = True
FINETUNE_RESET_REBUILD_SCHEDULER = True
finetune_reset_rebuild_scheduler = True
FINETUNE_RESET_GLOBAL_STEP = True
finetune_reset_global_step = True
FINETUNE_RESET_TEACHER = True
finetune_reset_teacher = True
FINETUNE_RESET_FORCE_LR_FLOOR = True
finetune_reset_force_lr_floor = True
FINETUNE_RESET_LR = 2e-5
finetune_reset_lr = 2e-5

TEACHER_FUSION_MODE = "dabe_pu_despl_sched"
teacher_fusion_mode = "dabe_pu_despl_sched"
FUSION_ORIG_DECAY_EPOCHS = 20
fusion_orig_decay_epochs = 20
FUSION_HOLD_FIXED_WEIGHT = 0.05
fusion_hold_fixed_weight = 0.05
TEACHER_FUSION_PRE_RESET_EPOCHS = 19
teacher_fusion_pre_reset_epochs = 19
FUSION_MIN_FIXED_WEIGHT = 0.05
fusion_min_fixed_weight = 0.05
TEACHER_FUSION_MAX_WEIGHT = 1.0
teacher_fusion_max_weight = 1.0

DABE_PU_DESPL_STAGE_START = 1
DABE_PU_DESPL_STAGE_END = 20
DABE_PU_DESPL_STATIC_START = 1.0
DABE_PU_DESPL_STATIC_END = 0.05
DABE_PU_DESPL_TEACHER_START = 0.0
DABE_PU_DESPL_TEACHER_END = 0.95
DABE_PU_DESPL_TEACHER_ONLY_START = 21

RAST_START_EPOCH = 7
RAST_RAMP_END_EPOCH = 15
RAST_STOP_EPOCH = 21
RAST_POST_RESET_ENABLE = False
RAST_POST_RESET_SCALE = 0.0
RAST_POST_RESET_CONFLICT_ONLY = False
RAST_POST_RESET_USE_STATIC_LOSS = False

ESA_ASYM_START_EPOCH = 7
ESA_ASYM_RAMP_END_EPOCH = 15
ESA_ASYM_STOP_EPOCH = 21

USE_TCE = False
TCE_LAMBDA_MAX = 0.0
USE_HBNS_LITE = False
HBNS_LAMBDA_MAX = 0.0
USE_EPR_POS = False
EPR_LAMBDA_MAX = 0.0
USE_DARE = False
USE_PROTO_CONTRAST = False
LAMBDA_PROTO_MAX = 0.0

# LCEG-v1 stays unchanged, just stops before epoch36 clean consolidation.
USE_LCEG = True
LCEG_VERSION = "v1_late_core_extent_guard_cc40"
LCEG_START_EPOCH = 26
LCEG_RAMP_END_EPOCH = 28
LCEG_STOP_EPOCH = 36

LCEG_COVER_CACHE_ROOT = "../datasets/cache/lceg_cover_cache/dinov1-s8/long35_epoch025"
LCEG_COVER_MODEL = "student"
LCEG_COVER_EPOCH = 25

USE_CLEAN_CONSOLIDATION_LOG = True
