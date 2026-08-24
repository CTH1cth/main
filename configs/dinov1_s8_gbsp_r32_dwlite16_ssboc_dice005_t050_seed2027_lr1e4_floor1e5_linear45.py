"""DW-Lite16 + SSBOC with a full-run 1e-4 -> 1e-5 LR decay."""

from configs.dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027-"
    "lr1e4-floor1e5-linear45"
)

# The LR changes linearly over all 45 epochs.  FINETUNE_RESET_EPOCH=0 means
# this is one uninterrupted full-run schedule rather than a reset stage.
GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027 = False
GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_DICE005_T050_SEED2027 = True
LR = 1e-4
LR_FLOOR = 1e-5
LR_POLICY = "linear_floor_two_stage"
LR_LINEAR_STAGE1_EPOCHS = MAX_EPOCH
LR_LINEAR_STAGE2_EPOCHS = 0
DINO = {**DINO, "lr": LR}
