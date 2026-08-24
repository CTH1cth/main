"""Frozen protocol for the GBSP candidate harmful-influence diagnosis."""

from configs.dinov1_s8_gbsp_cvbr_weighted_pca import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_candidate_influence"
GBSP_INFLUENCE_VERSION = "gbsp_candidate_influence_v1"
GBSP_INFLUENCE_CORE_ROOT = "../workdir/gbsp_core_optimization/full_rank"
GBSP_INFLUENCE_R0_ROOT = "../workdir/gbsp_cvbr_weighted_pca/R0_uniform"
GBSP_INFLUENCE_ORACLE_ROOT = "../workdir/gbsp_oracle_clean_matched_full6473/full6473"
GBSP_INFLUENCE_OUTPUT_ROOT = "../workdir/gbsp_candidate_influence_analysis"

GBSP_INFLUENCE_GRID = 37
GBSP_INFLUENCE_FEATURE_DIM = 384
GBSP_INFLUENCE_RANK = 8
GBSP_INFLUENCE_CVBR_SCORED_PER_IMAGE = 136
GBSP_INFLUENCE_GT_PATCH_RULE = "nearest_37_gt_gt_0.5"
GBSP_INFLUENCE_ORACLE_SEEDS = (0, 1, 2)
GBSP_INFLUENCE_TOP_FRACTION = 0.10
GBSP_INFLUENCE_EPS = 1e-12

# Fixed before looking at any exact-LOO result.
GBSP_INFLUENCE_SUBSET_SIZE = 600
GBSP_INFLUENCE_SUBSET_SEED = 20260814
GBSP_INFLUENCE_SUBSET_MIN_CONTAMINATED_RATIO = 0.60
GBSP_INFLUENCE_INCLUDE_ALL_CHAMELEON = True
GBSP_INFLUENCE_INCLUDE_ALL_CAMO = True

# Formal correctness gate required by the task book.
GBSP_INFLUENCE_VALIDATION_IMAGES = 100
GBSP_INFLUENCE_VALIDATION_CANDIDATES = 10
GBSP_INFLUENCE_VALIDATION_SEED = 20260814
GBSP_INFLUENCE_PROJECTOR_TOLERANCE = 1e-5
GBSP_INFLUENCE_MEAN_TOLERANCE = 1e-6

# Query-removal is diagnostic only and never changes a deployed prediction.
GBSP_INFLUENCE_TOP_H_PER_IMAGE = 3
GBSP_INFLUENCE_RANDOM_BG_PER_IMAGE = 1
GBSP_INFLUENCE_REMOVAL_SEED = 20260815

GBSP_INFLUENCE_EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "CAMO": 250,
    "COD10K": 2026,
    "NC4K": 4121,
}
