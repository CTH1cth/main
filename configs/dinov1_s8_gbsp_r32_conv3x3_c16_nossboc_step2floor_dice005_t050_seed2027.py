"""Traditional full-3x3 decoder without SSBOC and with early-fit LR."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "35-gbsp-r32-conv3x3-c16-nossboc-step2floor-"
    "dice005-t050-seed2027"
)

# Decoder: cached DINO F12 -> 1x1 (384->16) -> GELU -> ordinary dense 3x3
# (16->16, groups=1) -> GELU -> 1x1 (16->1).
GBSP_R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_DICE005_T050_SEED2027 = True
LCIC_VARIANT = "f_conv3x3"
HEAD_TYPE = "lcic_conv3x3"
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False
LCIC_CONV3X3_CHANNELS = 16

# Loss: explicitly retain only the matched hard-target BCE + 0.05 Soft-Dice.
LCIC_SSBOC_VARIANT = False
GBSP_SSBOC_VARIANT = False

# Let the randomly initialized decoder fit rapidly for two epochs using the
# established per-iteration StepLR, then hold the existing 2e-5 floor.
LR_POLICY = "step_then_floor"
LR_STEP_THEN_FLOOR_EPOCHS = 2
