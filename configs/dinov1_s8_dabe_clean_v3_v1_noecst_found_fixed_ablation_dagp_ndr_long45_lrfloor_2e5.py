"""V1-protocol ablation: replace only the static target with FOUND fixed pseudo."""

from configs.dinov1_s8_dabe_clean_v3_v1_noecst_compfg_dagp_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v3_v1_noecst_found_fixed_ablation_"
    "dagp_ndr_long45_lrfloor_2e5"
)

# Keep the complete V1 Long45/Teacher/DAGP/NDR protocol, but replace its static
# supervision tensor with the historical DINOv1-S8 FOUND-style fixed pseudo.
DABE_CLEAN_TRAINING_TARGET_SOURCE = "found_fixed"
FOUND_STATIC_ROOT = "../datasets/cache/pseudo_label_cache/dinov1-s8"
FOUND_STATIC_RESIZE_MODE = "bilinear"

# These V1-formula roots are deliberately empty so this ablation cannot read
# complementary/recoverability payloads while training.
DABE_CLEAN_ROOT = ""
DABE_CLEAN_SOURCE_ROOT = ""
DABE_CLEAN_OFFLINE_SOURCE_ROOT = ""
DABE_CLEAN_OFFLINE_SEMANTIC_ROOT = ""
