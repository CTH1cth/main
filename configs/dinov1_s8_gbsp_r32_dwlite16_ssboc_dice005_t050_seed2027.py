"""Traditional feature-space DW-Lite decoder with parameter-free SSBOC."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_ssboc_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027"

# Decoder-only intervention relative to the matched LCIC+SSBOC run.  DW-Lite
# performs 384->16 channel reduction, a local 3x3 depthwise convolution, and a
# final 1x1 classifier.  It never propagates or refines an intermediate logit.
GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027 = True
LCIC_SSBOC_VARIANT = False
LCIC_VARIANT = "e_dwlite"
HEAD_TYPE = "lcic_dwlite"
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False
LCIC_DWLITE_CHANNELS = 16
