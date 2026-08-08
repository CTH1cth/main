"""Oracle diagnostic: legacy cached DINO feature + one 1x1 conv + GT_68.

This inherits the exact Hard-R1 linear pure-Student protocol and changes only
the effective supervision source.  The original training GT is resized to
68x68 with nearest-neighbour interpolation and is the sole BCE target.  The
legacy single-layer DINO feature cache is reused; no Last4 cache and no online
DINO forward are involved.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gt68_linear_cached_b16"

GT68_LINEAR_CACHED_DIAGNOSTIC = True
GT_DIAGNOSTIC_SUPERVISION = True
GT_DIAGNOSTIC_REFERENCE_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
DECODER_SUPERVISION_SOURCE = "gt_hard_68"
GT_DIAGNOSTIC_RESIZE_MODE = "nearest_to_68"
GT_DIAGNOSTIC_STRICT_BINARY = True
