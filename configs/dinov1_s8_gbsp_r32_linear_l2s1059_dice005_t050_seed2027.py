"""Scaled-L2 F12 input ablation for the matched R32 t0.50 1x1 control."""

from configs.dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "33-gbsp-r32-1x1-l2s1059-dice005-t050-seed2027"

# Preserve the average raw-F12 patch norm while removing per-patch norm
# variation. Normalization is applied on the cached 37-grid before resize.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = False
GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027 = True
SIMPLE_FEATURE_L2_NORMALIZE = True
SIMPLE_FEATURE_L2_SCALE = 10.59
SIMPLE_FEATURE_L2_EPS = 1e-12
