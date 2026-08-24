"""Parameter-free SSBOC loss on the matched R32 t0.50 1x1 control."""

from configs.dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "34-gbsp-r32-1x1-ssboc-dice005-t050-seed2027"

# The only intervention is the self-scaled, bias-orthogonal correlation term.
# It consumes the same cached continuous R32 GBSP score that generated the
# unchanged strict-greater-than-0.50 hard target.  It has no lambda, margin,
# confidence cutoff, top-k fraction, temperature, or epoch schedule.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = False
GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027 = True
GBSP_SSBOC_VARIANT = True
