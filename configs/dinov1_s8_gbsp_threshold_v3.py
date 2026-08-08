"""Frozen settings for GBSP Adaptive Thresholding V3."""

GBSP_V3_VERSION = "gbsp_adaptive_threshold_v3"
GBSP_V3_BACKBONE = "dinov1-s8"
GBSP_V3_GRID = 37
GBSP_V3_NUM_PATCHES = 1369
GBSP_V3_EPS = 1e-8
GBSP_V3_SCORE_CLIP = 1e-4

# Immutable, already-generated M=1 GBSP cache.  The historical directory name
# does not imply a 200-image subset; its manifest contains all 6473 test images.
GBSP_V3_TEST_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_V3_TEST20_LIST = "../workdir/gbsp_threshold_v2/test20_ids.txt"
GBSP_V3_PILOT200_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_V3_OUTPUT_ROOT = "../workdir/gbsp_threshold_v3"
GBSP_V3_BASELINE_AUDIT = (
    "../workdir/gbsp_threshold_v3/baseline_test20/numerical_failure_summary.json"
)

# Evaluation-only reference.  This value is never passed to an adaptive method.
GBSP_V3_REFERENCE_FIXED_THRESHOLD = 0.58
GBSP_V3_BASELINE_TEST20_TARGET = {
    "S_m": 0.7458224526724683,
    "F_beta_w": 0.658548512255012,
    "F_beta_mean": 0.7119134420296271,
    "E_mean": 0.8245257796660878,
    "MAE": 0.07604126462976965,
    "Precision": 0.759875,
    "Recall": 0.687049,
    "Area": 0.125833,
}
GBSP_V3_BASELINE_TOLERANCE = 1e-4

GBSP_V3_CORE_METHODS = ("multi_otsu_3", "otgc", "ut_3cp")
GBSP_V3_HYSTERESIS_METHODS = ("multi_otsu_3_h", "otgc_h", "ut_3cp_h")

# Frozen OTGC Core settings.
GBSP_V3_OTGC_ANCHOR_STRENGTH = 1.0
GBSP_V3_OTGC_MIN_MEAN_GAP = 0.05
GBSP_V3_OTGC_SIGMA_FLOOR = 0.05
GBSP_V3_OTGC_SIGMA_CAP = 10.0
GBSP_V3_OTGC_FOREGROUND_MIXTURE_FLOOR = 0.001
GBSP_V3_OTGC_FOREGROUND_MIXTURE_CAP = 0.5
GBSP_V3_OTGC_MAX_ITER = 300
GBSP_V3_OTGC_HISTORY_SIZE = 50
GBSP_V3_OTGC_TOLERANCE_GRAD = 1e-7
GBSP_V3_OTGC_TOLERANCE_CHANGE = 1e-9

# Frozen UT-3CP Core settings.
GBSP_V3_UT3CP_SMOOTH_WINDOW = 21
GBSP_V3_UT3CP_SMOOTH_POLYORDER = 2
GBSP_V3_UT3CP_MIN_HIGH_LENGTH = 8
GBSP_V3_UT3CP_MIN_MIDDLE_LENGTH = 24
GBSP_V3_UT3CP_MIN_BACKGROUND_LENGTH = 64

GBSP_V3_TEST20_DIRECT_GATE = {
    "F_beta_w_min": 0.6485,
    "Precision_min": 0.7299,
    "Area_min": 0.08,
    "Area_max": 0.1510,
    "empty_ratio_max": 0.05,
    "large_ratio_max": 0.0,
    "numerical_failure_ratio_max": 0.0,
}
GBSP_V3_HYSTERESIS_TRIGGER = {
    "F_beta_w_min": 0.630,
    "Precision_min": 0.7599,
    "Recall_max": 0.62,
    "Area_max": 0.09,
    "forbid_area_over": 0.16,
    "forbid_precision_below": 0.70,
}
GBSP_V3_HYSTERESIS_GATE = {
    "Precision_drop_max": 0.02,
    "Area_max": 0.1510,
    "hard_stop_area": 0.16,
    "hard_stop_precision_drop": 0.03,
}
GBSP_V3_PILOT200_GATE = {
    "F_beta_w_min": 0.6255,
    "MAE_max": 0.0819,
    "Precision_min": 0.6907,
    "Area_min": 0.09,
    "Area_max": 0.155,
    "empty_ratio_max": 0.02,
    "large_ratio_max": 0.0,
    "COD10K_F_beta_w_drop_max": 0.01,
    "NC4K_F_beta_w_drop_max": 0.01,
}

# This contract is included in every generated run_config/frozen config.
GBSP_V3_INDEPENDENCE_CONTRACT = {
    "gt_used": False,
    "r1_used": False,
    "fixed_058_used": False,
    "target_area_prior_used": False,
    "fixed_topk_used": False,
    "dino_forward_used": False,
    "pca_refit_used": False,
    "morphology_used": False,
}
