"""LCIC-A: native-grid linear anchor under the frozen GBSP t=0.58 protocol."""

from configs.dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "24-lcic-a-linear"
GBSP_LCIC_V1 = True
LCIC_VARIANT = "a_linear"
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False
LCIC_AFFINITY_EPS = 1e-6
LCIC_DWLITE_CHANNELS = 16

# Reuse SimpleConvSegHead unchanged, but read F12 at its native 37x37 grid.
HEAD_TYPE = "lcic_linear"
