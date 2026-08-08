"""Hard P1-RPR-SecondRing -> one 1x1 convolution, with no Teacher.

This is the exact V1-CVBR/R1 linear pure-Student protocol with only the
immutable 37x37 static pseudo target changed to ``p1_rpr_secondring_37``.
The response is bilinearly resized to 68x68 with ``align_corners=False`` and
thresholded with strict ``> 0.5``.  The binary mask keeps full loss weight for
all 45 epochs.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

RPR_P1_SECOND_RING_LINEAR = True
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
