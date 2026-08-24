"""Low-LR control for R32 DW-Lite16 + SSBOC: 1e-4 down to 1e-5."""

from configs.dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027-lr1e4-floor1e5"
)

# Only the LR envelope changes.  Keep the established iteration StepLR
# (step_size=25, gamma=0.95), optimizer, decoder, loss, labels, and seed.
GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027 = False
GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_DICE005_T050_SEED2027 = True
LR = 1e-4
LR_FLOOR = 1e-5
DINO = {**DINO, "lr": LR}
