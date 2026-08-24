"""Pixel-confidence BCE ablation for the matched R32 t0.50 1x1 control."""

from configs.dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "34-gbsp-r32-1x1-confbce-f050-g1-dice005-t050-seed2027"

# Preserve the exact R32 hard target and every baseline training setting.  The
# sole intervention is a detached pixel weight computed from the cached
# continuous GBSP score.  Raw confidence has a 0.50 floor and gamma 1.0, then
# is normalized within foreground/background separately for every image so the
# aggregate class contribution is unchanged.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = False
GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027 = True
GBSP_CONFIDENCE_BCE_VARIANT = True
GBSP_CONFIDENCE_BCE_FLOOR = 0.50
GBSP_CONFIDENCE_BCE_GAMMA = 1.0
GBSP_CONFIDENCE_BCE_CLASSWISE_NORMALIZE = True
