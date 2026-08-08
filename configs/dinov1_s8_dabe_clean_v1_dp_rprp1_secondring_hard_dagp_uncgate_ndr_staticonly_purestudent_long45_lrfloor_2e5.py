"""Hard P1-RPR-SecondRing with DAGP-Safe + NDR-v1 from epoch one.

The static target and cache are identical to the P1-RPR linear experiment:
``p1_rpr_secondring_37`` is bilinearly resized to 68x68 and thresholded with
strict ``> 0.5``.  Only the Student decoder changes to DAGP-Safe + NDR-v1;
there is no Teacher forward or EMA update.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

RPR_P1_SECOND_RING_DAGP_NDR = True
R1_ONLY_CACHE_IO = True
RPR_TRAIN_CACHE_MODE = "p1_only"
RPR_TRAIN_CACHE_WORKERS = 2
RPR_TRAIN_CACHE_TORCH_THREADS = 12

DABE_CLEAN_DABE_V2_ROOT = (
    "../workdir/dabe_rpr_p1_train_singleview/dinov1-s8"
)
DABE_CLEAN_STATIC_TARGET_SOURCE = "rpr_p1_second_ring_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "p1_rpr_secondring_37"
DABE_CLEAN_CVBR_VERSION = "dabe_cvbr_v1"
DABE_CLEAN_RPR_VERSION = "dabe_rpr_v1"
DABE_CLEAN_RPR_AUGS = ("identity",)
R1_HARD_SOURCE_KEY = "p1_rpr_secondring_37"
