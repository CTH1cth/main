"""DABE-v2-hard static supervision with Evidence-Anchored Online Graph Projection."""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_eaogp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

USE_EAOGP = True
TEACHER_ROUTING_MODE = "eaogp"
EAOGP_VERSION = "eaogp_v1_dabe_anchor_dualgraph_projection"

USE_ECTP = False
USE_ECST = False
USE_ECST_MINIMAL = False
USE_ECST_CLEAN = False
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False

STOP_AFTER_EPOCH = 0
