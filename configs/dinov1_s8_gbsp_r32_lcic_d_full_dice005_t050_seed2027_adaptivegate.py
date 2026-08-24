"""R32 LCIC-D with normalized image-adaptive correction strengths."""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = (
    "26-gbsp-r32-lcic-d-full-dice005-uniform050-seed2027-"
    "adaptivegate"
)

# The two correction branches are normalized per image.  A 58-parameter MLP
# learns their signed logit amplitudes directly from detached image statistics;
# no hand-selected consensus/innovation multiplier is applied.
LCIC_ADAPTIVE_GATE_VARIANT = True
LCIC_ADAPTIVE_GATE_HIDDEN = 8
LCIC_ADAPTIVE_GATE_EPS = 1e-6
LCIC_CONSENSUS_GAIN = 1.0
LCIC_INNOVATION_GAIN = 1.0
