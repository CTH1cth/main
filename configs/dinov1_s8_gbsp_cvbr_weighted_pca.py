"""Frozen first-phase protocol for CVBR-guided soft-weighted GBSP PCA."""

from configs.dinov1_s8_gbsp_core_ablation import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_cvbr_weighted_pca"
GBSP_CVBR_WEIGHTED_VERSION = "gbsp_cvbr_weighted_pca_v1"
GBSP_CVBR_SOURCE_ROOT = "../workdir/gbsp_core_optimization/full_rank"
GBSP_CVBR_CACHE_ROOT = "../workdir/dabe_cvbr_v1_identity/dinov1-s8"
GBSP_CVBR_ORACLE_ROOT = "../workdir/gbsp_oracle_clean_matched_full6473/full6473"
GBSP_CVBR_OUTPUT_ROOT = "../workdir/gbsp_cvbr_weighted_pca"

# Everything except the PCA estimator is frozen.
GBSP_CVBR_GRID = 37
GBSP_CVBR_FEATURE_DIM = 384
GBSP_CVBR_RANK = 8
GBSP_CVBR_SCOPE = "existing_ring2_cv_error"
GBSP_CVBR_SCORE_FIELD = "boundary_cv_error_37"
GBSP_CVBR_VALID_MASK_FIELD = "border_ring2_only_37"
GBSP_CVBR_VALIDATED_VERSION = "dabe_cvbr_v1"
GBSP_CVBR_THRESHOLD = 0.58
GBSP_CVBR_RESIZE = "bilinear_37_to_68_to_original"
GBSP_CVBR_GT_PATCH_RULE = "nearest_37_gt_gt_0.5"

# No extra variants may be appended during this registered first phase.
GBSP_CVBR_VARIANTS = {
    "R0_uniform": {"cache_key": "gbsp_uniform_r8", "top_frac": 0.00, "low_weight": 1.00},
    "R1_p005_w025": {"cache_key": "gbsp_cvbrw_r8_p005_w025", "top_frac": 0.05, "low_weight": 0.25},
    "R2_p010_w025": {"cache_key": "gbsp_cvbrw_r8_p010_w025", "top_frac": 0.10, "low_weight": 0.25},
    "R3_p010_w010": {"cache_key": "gbsp_cvbrw_r8_p010_w010", "top_frac": 0.10, "low_weight": 0.10},
    "R4_p015_w025": {"cache_key": "gbsp_cvbrw_r8_p015_w025", "top_frac": 0.15, "low_weight": 0.25},
    "R5_p010_w000": {"cache_key": "gbsp_cvbrtrim_r8_p010", "top_frac": 0.10, "low_weight": 0.00},
}

GBSP_CVBR_EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "TE-CAMO": 250,
    "TE-COD10K": 2026,
    "NC4K": 4121,
}
GBSP_CVBR_UNIFORM_RESIDUAL_TOLERANCE = 1e-5
GBSP_CVBR_REFERENCE_AUROC = 0.8372494310380951
GBSP_CVBR_REFERENCE_TOP10_ENRICHMENT = 4.863112944925015
GBSP_CVBR_BOOTSTRAP_REPETITIONS = 10000
GBSP_CVBR_BOOTSTRAP_SEED = 20260814

