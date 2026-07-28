"""Teacher-only ablation with no offline pseudo-label cache access."""

from configs.dinov1_s8_dabe_clean_v3_v1_noecst_found_fixed_ablation_dagp_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_teacher_only_no_offline_pseudo_"
    "dagp_ndr_long45_lrfloor_2e5"
)

# Keep the frozen DINOv1-S8 feature cache and the complete DAGP-NDR/EMA/
# Long45/reset protocol, but forbid every offline pseudo-label cache.
DABE_CLEAN_TRAINING_TARGET_SOURCE = "teacher_only_no_offline_pseudo"
FOUND_STATIC_ROOT = ""
DABE_CLEAN_ROOT = ""
DABE_CLEAN_SOURCE_ROOT = ""
DABE_CLEAN_OFFLINE_SOURCE_ROOT = ""
DABE_CLEAN_OFFLINE_SEMANTIC_ROOT = ""
DABE_PU_ROOT = ""
DABE_CLEAN_LEGACY_REGION_ROOT = ""

# The historical epoch-21 milestone remains for protocol/reset compatibility,
# while the weights are Teacher-only from the very first optimizer step.
DABE_PU_DESPL_STATIC_START = 0.0
DABE_PU_DESPL_STATIC_END = 0.0
DABE_PU_DESPL_TEACHER_START = 1.0
DABE_PU_DESPL_TEACHER_END = 1.0
