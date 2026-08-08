"""DINOv1-S/8 Hard-R1 + pure DBA decoder upper-bound ablation.

The complete data, target, optimizer, schedule, epoch, batch-size and seed
protocol is inherited from the established S/8 Hard-R1 single-1x1 baseline.
Only the segmentation head and its intrinsic DBA loss group are changed.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_r1hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_dba_staticonly_purestudent_long45_lrfloor_2e5"

DABEV2HARD_R1_LINEAR = False
DABEV2HARD_R1_DBA = True
HEAD_TYPE = "dba"

DBA_EMBED_DIM = 64
DBA_OUTPUT_SIZE = 68
DBA_ORTHOGONAL_WEIGHT = 1.0

# Explicit exclusions for this decoder-only ablation.
USE_DAGP_SAFE_HEAD = False
USE_NDR_BRANCH = False
USE_BASE_AUX_LOSS = False
FINETUNE_RESET_TEACHER = False
LOOK_TWICE = False
