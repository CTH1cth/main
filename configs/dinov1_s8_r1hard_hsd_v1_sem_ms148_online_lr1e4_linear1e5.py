"""Low-LR control for online R1-HSD-Sem: 1e-4 -> 1e-5 over 45 epochs."""

from configs.dinov1_s8_r1hard_hsd_v1_sem_ms148_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_hsd_v1_sem_ms148_online_lr1e4_linear1e5"
R1_HSD_LR_ABLATION = True
R1_HSD_LR_REFERENCE_CONFIG = (
    "configs/dinov1_s8_r1hard_hsd_v1_sem_ms148_online.py"
)
R1_ONLY_CACHE_IO = True

# Keep the formal BS=16 protocol and change only the optimizer LR trajectory.
LR = 1e-4
DINO = dict(DINO)
DINO["lr"] = LR
LR_POLICY = "linear_floor_two_stage"
USE_LR_FLOOR = True
LR_FLOOR = 1e-5
LR_LINEAR_STAGE1_EPOCHS = MAX_EPOCH
LR_LINEAR_STAGE2_EPOCHS = 0
LR_FLOOR_MODE = "global"
LR_FLOOR_APPLY_AFTER_SCHEDULER_STEP = True
