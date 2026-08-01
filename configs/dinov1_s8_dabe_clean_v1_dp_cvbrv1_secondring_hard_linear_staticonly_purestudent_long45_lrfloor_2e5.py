"""Hard V1-CVBR-SecondRing -> one 1x1 convolution, with no Teacher.

This is the exact linear Hard-R1 pure-Student control with only the immutable
static pseudo target changed.  ``v1_cvbr_second_ring_37`` is bilinearly resized
from 37x37 to 68x68 (``align_corners=False``), then thresholded with strict
``> 0.5``.  The resulting binary mask has full loss weight for all 45 epochs.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_cvbrv1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

CVBR_V1_SECOND_RING_LINEAR = True

# Independent GT-free CVBR training cache.  The cache builder writes
# manifest_train.jsonl here without changing DABE-v2 or DINO feature caches.
DABE_CLEAN_DABE_V2_ROOT = (
    "../workdir/dabe_cvbr_v1_train_singleview/dinov1-s8"
)
DABE_CLEAN_STATIC_TARGET_SOURCE = "cvbr_v1_second_ring_hard_68"
DABE_CLEAN_DABE_V2_SOURCE_KEY = "v1_cvbr_second_ring_37"
DABE_CLEAN_CVBR_VERSION = "dabe_cvbr_v1"
DABE_CLEAN_CVBR_AUGS = ("identity",)
R1_HARD_SOURCE_KEY = "v1_cvbr_second_ring_37"
