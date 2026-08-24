"""LCIC-C: linear anchor plus local DINO graph-innovation readout."""

from configs.dinov1_s8_gbsp_lcic_a_linear import *  # noqa: F401,F403


EXP_NAME = "24-lcic-c-innovation"
LCIC_VARIANT = "c_innovation"
LCIC_USE_CONSENSUS = False
LCIC_USE_INNOVATION = True
HEAD_TYPE = "lcic"
