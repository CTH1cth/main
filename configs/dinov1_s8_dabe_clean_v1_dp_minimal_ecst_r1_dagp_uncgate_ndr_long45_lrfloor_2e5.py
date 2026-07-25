"""DABE-Clean DP with isolated Minimal-ECST radius 1."""

from configs.dinov1_s8_dabe_clean_v1_dp_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_minimal_ecst_r1_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)
USE_ECST = False
USE_ECST_MINIMAL = True
TEACHER_ROUTING_MODE = "minimal_ecst"
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False
ECST_MINIMAL_VERSION = "evidence_protect_ring_asym_v1"
ECST_MINIMAL_RING_RADIUS = 1
ECST_MINIMAL_CONFLICT_FLOOR = 0.20
STOP_AFTER_EPOCH = 0

