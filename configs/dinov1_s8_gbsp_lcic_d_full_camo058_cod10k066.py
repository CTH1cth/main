"""LCIC-D with dataset-specific oracle GBSP hard-target thresholds.

TR-CAMO uses 0.58 and TR-COD10K uses 0.66.  All decoder, optimizer,
schedule, cache, and evaluation settings remain identical to LCIC-D-FULL.
"""

from configs.dinov1_s8_gbsp_lcic_d_full import *  # noqa: F401,F403


EXP_NAME = "24-lcic-d-full-camo058-cod10k066"
LCIC_DATASET_THRESHOLD_VARIANT = True
DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET = {
    "TR-CAMO": 0.58,
    "TR-COD10K": 0.66,
}
DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD = True
