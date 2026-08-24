"""LCIC-D: full consensus plus graph-innovation lightweight decoder."""

from configs.dinov1_s8_gbsp_lcic_a_linear import *  # noqa: F401,F403


EXP_NAME = "24-lcic-d-full"
LCIC_VARIANT = "d_full"
LCIC_USE_CONSENSUS = True
LCIC_USE_INNOVATION = True
HEAD_TYPE = "lcic"
