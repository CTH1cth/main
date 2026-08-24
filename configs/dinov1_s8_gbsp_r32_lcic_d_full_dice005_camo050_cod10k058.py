"""R32 LCIC-D-FULL with Soft-Dice and dataset-specific hard targets.

The full 4040-image oracle diagnostic shows that fixed-R32 needs a larger
foreground target on TR-CAMO while TR-COD10K benefits from retaining the
stricter 0.58 threshold.  The decoder, optimizer, schedule, cache, and all
other training settings are inherited unchanged from the validated R32
LCIC-D-FULL + 0.05 Soft-Dice configuration.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_dice005 import *  # noqa: F401,F403


EXP_NAME = "26-gbsp-r32-lcic-d-full-dice005-camo050-cod058"

LCIC_DATASET_THRESHOLD_VARIANT = True
DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET = {
    "TR-CAMO": 0.50,
    "TR-COD10K": 0.58,
}
DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD = True

# Both constants were selected with training GT and are oracle diagnostics.
GBSP_THRESHOLD_SELECTION_SPLIT = (
    "train4040_oracle_gt68_r32_dataset_specific_camo050_cod10k058"
)
