"""Half-LR control for the R32 DW-Lite16 + SSBOC seed-2027 run."""

from configs.dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027-lr0003"

# Single intervention: halve the initial decoder LR.  The iteration StepLR,
# gamma, global 2e-5 floor, optimizer, labels, loss, and decoder are unchanged.
GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027 = False
GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027 = True
LR = 3e-4
DINO = {**DINO, "lr": LR}
