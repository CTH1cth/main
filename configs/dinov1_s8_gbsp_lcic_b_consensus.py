"""LCIC-B: linear anchor plus one detached-affinity consensus correction."""

from configs.dinov1_s8_gbsp_lcic_a_linear import *  # noqa: F401,F403


EXP_NAME = "24-lcic-b-consensus"
LCIC_VARIANT = "b_consensus"
LCIC_USE_CONSENSUS = True
LCIC_USE_INNOVATION = False
HEAD_TYPE = "lcic"
