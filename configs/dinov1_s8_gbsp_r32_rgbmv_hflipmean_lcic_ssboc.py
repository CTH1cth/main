"""True-RGB identity+hflip soft-mean pseudo target with LCIC-D-FULL+SSBOC.

This is a strict pseudo-target-only variant of the matched seed-2027 control:
decoder, SSBOC, 0.05 Soft-Dice, optimizer, schedule, and threshold stay fixed.
"""

from configs.dinov1_s8_gbsp_r32_lcic_d_full_ssboc_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "36-gbsp-r32-rgbmv-hflipmean-lcic-ssboc"
GBSP_RGBMV_HFLIP_SOFTMEAN_VARIANT = True

GBSP_VERSION = "gbsp_r32_rgbmv_id_hflip_softmean_v1"
DABE_CLEAN_DABE_V2_ROOT = (
    "../datasets/cache/gbsp_r32_rgbmv_id_hflip_softmean_v1/dinov1-s8"
)
DABE_CLEAN_GBSP_VERSION = "gbsp_r32_rgbmv_id_hflip_softmean_v1"
DABE_CLEAN_GBSP_AUGS = ["identity", "hflip"]

# The fusion itself is GT-free.  The retained t=0.50 is inherited from the
# existing training-GT threshold diagnostic and is not re-selected here.
GBSP_THRESHOLD_SELECTION_SPLIT = (
    "train4040_oracle_gt68_r32_rgbmv_id_hflip_softmean_t050"
)
