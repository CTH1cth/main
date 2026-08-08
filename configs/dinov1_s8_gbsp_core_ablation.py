"""Frozen protocol for the GBSP continuous-score core audit."""

from configs.dinov1_s8_mbsp import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_core_ablation"
GBSP_CORE_VERSION = "gbsp_core_optimization_v1"
GBSP_CORE_SOURCE_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_CORE_OUTPUT_ROOT = "../workdir/gbsp_core_optimization"
GBSP_CORE_TEST20_LIST = "../workdir/gbsp_threshold_v2/test20_ids.txt"

# Frozen formal representation and current GBSP baseline.
GBSP_CORE_GRID = 37
GBSP_CORE_FEATURE_DIM = 384
GBSP_CORE_PCA_ENERGY = 0.90
GBSP_CORE_PCA_MAX_RANK = 8
GBSP_CORE_PCA_MIN_RANK = 1
GBSP_CORE_EPS = 1e-8
GBSP_CORE_RESIZE = "bilinear_37_to_68_to_original"
GBSP_CORE_THRESHOLDS = (0.50, 0.58)

GBSP_CORE_RANK_VARIANTS = ("r0", "r4", "r8", "r16", "r32", "current", "ev90", "ev95")
GBSP_CORE_BACKGROUND_VARIANTS = ("boundary280", "fullbc_matched280", "fullbc")
GBSP_CORE_WEIGHT_VARIANTS = ("equal", "path_inverse")
GBSP_CORE_EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "TE-CAMO": 250,
    "TE-COD10K": 2026,
    "NC4K": 4121,
}

# Existing 6473-image formal-cache reference.  It is an audit gate, not a
# threshold-selection target and is never used while producing new responses.
GBSP_CORE_BASELINE_REFERENCE = {
    "pixel_AP": 0.7726448043,
    "pixel_AUROC": 0.9524009755,
    "S_m": 0.7236768642,
    "F_beta_w": 0.6093427908,
    "E_mean": 0.8088151742,
    "MAE": 0.0897199697,
}
GBSP_CORE_BASELINE_TOLERANCE = 1e-4
