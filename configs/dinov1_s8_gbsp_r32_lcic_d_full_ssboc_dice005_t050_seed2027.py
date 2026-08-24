"""Parameter-free SSBOC loss on the matched LCIC-D-FULL seed-2027 run."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "34-gbsp-r32-lcic-d-full-ssboc-dice005-t050-seed2027"

# Decoder, pseudo target, optimizer, schedule, seed, and evaluation protocol
# are inherited unchanged.  SSBOC consumes the same detached continuous R32
# GBSP score whose strict-greater-than-0.50 mask supplies the BCE/Dice target.
# The self-scaled correlation term introduces no tunable coefficient.
LCIC_SSBOC_VARIANT = True
GBSP_SSBOC_VARIANT = True
