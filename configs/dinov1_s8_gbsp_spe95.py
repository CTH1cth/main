"""Frozen settings for GT-free GBSP PCA-SPE residual calibration."""

GBSP_SPE_VERSION = "gbsp_spe_calibration_v1"
GBSP_SPE_GRID = 37
GBSP_SPE_FEATURE_DIM = 384
GBSP_SPE_EPS = 1e-12
GBSP_SPE_PRIMARY_CONTROL_LEVEL = 0.95
GBSP_SPE_CONTROL_LEVELS = (0.90, 0.95, 0.99)
GBSP_SPE_METHODS = ("jm_spe", "gamma_spe")

# Historical directory name aside, this is the complete frozen M=1 GBSP
# cache: 76 + 250 + 2026 + 4121 = 6473 images.
GBSP_SPE_GBSP_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_SPE_TEST20_LIST = "../workdir/gbsp_threshold_v2/test20_ids.txt"
GBSP_SPE_DIAGNOSTIC200_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_SPE_OUTPUT_ROOT = "../workdir/gbsp_spe_calibration"

GBSP_SPE_EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "TE-CAMO": 250,
    "TE-COD10K": 2026,
    "NC4K": 4121,
}

GBSP_SPE_FORMAL_GBSP = {
    "num_subspaces": 1,
    "pca_energy": 0.90,
    "pca_max_rank": 8,
    "pca_min_rank": 1,
    "feature_source": "frozen_identity_dinov1_s8",
}

GBSP_SPE_INDEPENDENCE = {
    "gt_used": False,
    "r1_used": False,
    "fixed_058_used": False,
    "target_area_used": False,
    "dino_forward_used": False,
    "pca_refit": False,
    "pca_rank_changed": False,
    "raw_residual_changed": False,
    "morphology_used": False,
    "graph_propagation_used": False,
    "background_candidates_forced": False,
}

