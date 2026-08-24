"""LCIC-E: ordinary depthwise-local capacity control without DINO affinity."""

from configs.dinov1_s8_gbsp_lcic_a_linear import *  # noqa: F401,F403


EXP_NAME = "24-lcic-e-dwlite"
LCIC_VARIANT = "e_dwlite"
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = False
LCIC_DWLITE_CHANNELS = 16
HEAD_TYPE = "lcic_dwlite"
