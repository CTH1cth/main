"""Linear 1x1 control matched to the R32 LCIC-D seed-2027 run.

The cached GBSP response, effective per-dataset pseudo threshold, loss,
optimizer, schedule, seed, and evaluation threshold are inherited unchanged.
Only the decoder path is replaced by the established 68-grid single 1x1
convolution control.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "33-gbsp-r32-1x1-dice005-t050-seed2027"

# Decoder-only control: interpolate the cached F12 feature to LOSS_SIZE and
# apply exactly one 384->1 convolution, as in the established linear baseline.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = True
HEAD_TYPE = "simple"
GBSP_LCIC_V1 = False
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False

# Soft-Dice is a supervision term, not an LCIC operation.  Keep the same 0.05
# term so the comparison changes the decoder rather than decoder plus loss.
GBSP_LINEAR_SOFT_DICE_CONTROL = True
LCIC_SOFT_DICE_VARIANT = True
LCIC_SOFT_DICE_WEIGHT = 0.05

