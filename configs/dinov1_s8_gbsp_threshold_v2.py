"""Frozen settings for GBSP Adaptive Thresholding V2."""

GBSP_V2_VERSION = "gbsp_adaptive_threshold_v2"
GBSP_V2_BACKBONE = "dinov1-s8"
GBSP_V2_GRID = 37
GBSP_V2_NUM_PATCHES = 1369
GBSP_V2_EPS = 1e-8
GBSP_V2_SCORE_CLIP = 1e-4

# Existing immutable M=1 GBSP cache.  Its directory name is historical; the
# manifest contains the complete formal 6473-image test set.
GBSP_V2_TEST_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_V2_TEST20_LIST = "../workdir/gbsp_threshold_v2/test20_ids.txt"
GBSP_V2_PILOT200_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_V2_OUTPUT_ROOT = "../workdir/gbsp_threshold_v2"
GBSP_V2_BASELINE_AUDIT = (
    "../workdir/gbsp_threshold_v2/baseline_test20/numerical_failure_summary.json"
)

# Reference-only baseline.  This value is forbidden from every V2 adaptive
# formula and is consumed only by the evaluator's fixed baseline branch.
GBSP_V2_REFERENCE_FIXED_THRESHOLD = 0.58
GBSP_V2_BASELINE_TEST20_TARGET = {
    "S_m": 0.7458224526724683,
    "F_beta_w": 0.658548512255012,
    "F_beta_mean": 0.7119134420296271,
    "E_mean": 0.8245257796660878,
    "MAE": 0.07604126462976965,
}
GBSP_V2_BASELINE_TOLERANCE = 1e-4

GBSP_V2_BMC_ANCHOR_STRENGTHS = (0.5, 1.0)
GBSP_V2_BMC_PRIMARY_ANCHOR = 1.0
GBSP_V2_BMC_ALPHA_BETA_MIN = 0.2
GBSP_V2_BMC_ALPHA_BETA_MAX = 200.0
GBSP_V2_BMC_PI_MIN = 0.001
GBSP_V2_BMC_PI_MAX = 0.5
GBSP_V2_BMC_MIN_MEAN_GAP = 0.05
GBSP_V2_BMC_MAX_ITER = 300

GBSP_V2_LGMC_ANCHOR_STRENGTHS = (0.5, 1.0)
GBSP_V2_LGMC_PRIMARY_ANCHOR = 1.0
GBSP_V2_LGMC_SIGMA_MIN = 0.05
GBSP_V2_LGMC_SIGMA_MAX = 10.0
GBSP_V2_LGMC_MIN_MEAN_GAP = 0.25
GBSP_V2_LGMC_MAX_ITER = 200

GBSP_V2_OPT_TOLERANCE = 1e-7

GBSP_V2_EVT_REFERENCE_LOW_QUANTILE = 0.50
GBSP_V2_EVT_TAIL_START_QUANTILE = 0.80
GBSP_V2_EVT_MIN_TAIL_COUNT = 30
GBSP_V2_EVT_Q_VALUES = (0.01, 0.025, 0.05)
GBSP_V2_EVT_PRIMARY_Q = 0.025
GBSP_V2_EVT_SHAPE_MIN = -0.5
GBSP_V2_EVT_SHAPE_MAX = 0.8
GBSP_V2_EVT_SCALE_MIN = 1e-6

GBSP_V2_QDCP_MIN_SEGMENT = 32
GBSP_V2_QDCP_SAVGOL_WINDOW = 31
GBSP_V2_QDCP_SAVGOL_ORDER = 2
GBSP_V2_BIC_SUPPORT = 10.0

GBSP_V2_METHODS = ("ba_bmc", "ba_lgmc", "eb_evt", "qdcp_pl", "qdcp_k")

GBSP_V2_TEST20_GATE = {
    "F_beta_w_min": 0.6485,
    "Precision_min": 0.7299,
    "Area_min": 0.08,
    "Area_max": 0.1510,
    "empty_ratio_max": 0.10,
    "large_ratio_max": 0.0,
    "numerical_failure_ratio_max": 0.0,
}
GBSP_V2_PILOT200_GATE = {
    "F_beta_w_min": 0.6255,
    "MAE_max": 0.0819,
    "Precision_min": 0.6907,
    "Area_min": 0.09,
    "Area_max": 0.155,
    "empty_ratio_max": 0.02,
    "large_ratio_max": 0.0,
    "COD10K_F_beta_w_drop_max": 0.01,
    "NC4K_F_beta_w_drop_max": 0.01,
    "per_image_noninferior_min": 90,
}

GBSP_V2_BSC_VOTE_THRESHOLD = 0.625
GBSP_V2_BSC_NUM_GROUPS = 8
