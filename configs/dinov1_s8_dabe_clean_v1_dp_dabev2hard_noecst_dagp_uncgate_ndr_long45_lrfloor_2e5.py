"""Strict no-ECST control for DABE-v2-hard static supervision.

Only the pixel-wise ECST teacher routing is disabled.
Static target, global static-to-teacher schedule, binary EMA teacher,
finetune reset, optimizer, learning rate, DAGP, NDR, data loading,
random seed and checkpoint protocol must remain unchanged.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

# The only conceptual ablation: disable spatially varying ECST teacher
# admission and use an exact all-one teacher route.
USE_ECST = False
USE_ECST_MINIMAL = False
TEACHER_ROUTING_MODE = "none"
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False

STOP_AFTER_EPOCH = 0
