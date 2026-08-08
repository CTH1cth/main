"""Frozen settings for GBSP Adaptive Thresholding V5."""

GBSP_V5_VERSION = "gbsp_adaptive_threshold_v5"
GBSP_V5_GRID = 37
GBSP_V5_EPS = 1e-8
GBSP_V5_GBSP_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_V5_GRAPH_ROOT = "../workdir/dabe_r1_design_v1_identity/dinov1-s8"
GBSP_V5_DIAGNOSTIC200_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_V5_OUTPUT_ROOT = "../workdir/gbsp_threshold_v5"
GBSP_V5_BASELINE_AUDIT = "../workdir/gbsp_threshold_v5/baseline_diagnostic200/numerical_failure_summary.json"

GBSP_V5_METHODS = ("smoh", "spcg", "bcmp")
GBSP_V5_FAMILIES = {"smoh": "smoh", "spcg": "spcg", "bcmp": "bcmp"}
GBSP_V5_REFERENCES = ("fixed_058", "multi_otsu_3", "huto_mid_h")
GBSP_V5_BASELINE_METHODS = ("fixed_058", "multi_otsu_3", "huto_core", "huto_mid_h")

GBSP_V5_MULTI_OTSU_CLASSES = 3
GBSP_V5_MULTI_OTSU_BINS = 256
GBSP_V5_UPPER_TAIL_OTSU_BINS = 256
GBSP_V5_CONNECTIVITY = 8
GBSP_V5_SPCG_MIN_HISTORY = 5
GBSP_V5_SPCG_MAD_MULTIPLIER = 3.0
GBSP_V5_SPCG_MIN_EVENTS = 6
GBSP_V5_GRAPH_KIND = "similarity"
GBSP_V5_GRAPH_WEIGHT_FIELD = "graph_weight_node"
GBSP_V5_GRAPH_PROBABILITY_THRESHOLD = 0.5
GBSP_V5_FIXED_THRESHOLD_REFERENCE_ONLY = 0.58
GBSP_V5_BASELINE_TOLERANCE = 1e-4

# V4 Diagnostic200 frozen targets.  These values are audit-only and never enter
# any V5 prediction formula.
GBSP_V5_BASELINE_TARGETS = {
    "fixed_058": {"S_m": 0.7346198640920132, "F_beta_w": 0.6305235716610487,
                  "F_beta_mean": 0.680264240881445, "E_mean": 0.8234602987913578,
                  "MAE": 0.07891534837916059, "Precision": 0.7207470012433115,
                  "Recall": 0.6921307623975742, "Area": 0.12922998821828513},
    "multi_otsu_3": {"S_m": 0.7255891531691699, "F_beta_w": 0.6185861471235419,
                     "F_beta_mean": 0.6626725338037653, "E_mean": 0.8147345520300472,
                     "MAE": 0.08666119875609554, "Precision": 0.6814539258676371,
                     "Recall": 0.7372218303604904, "Area": 0.1430966732604429},
    "huto_core": {"S_m": 0.6672134231123222, "F_beta_w": 0.5515156121067157,
                  "F_beta_mean": 0.6402544284797662, "E_mean": 0.7576377299110948,
                  "MAE": 0.09697742740444391, "Precision": 0.7832308564977304,
                  "Recall": 0.4791075209302066, "Area": 0.07283607933670283},
    "huto_mid_h": {"S_m": 0.7192650757260788, "F_beta_w": 0.6159944524909458,
                   "F_beta_mean": 0.6771869926518183, "E_mean": 0.8111532093633138,
                   "MAE": 0.08448518298329391, "Precision": 0.7423460446106029,
                   "Recall": 0.6303888488356914, "Area": 0.10758201047312468},
}

GBSP_V5_DIAGNOSTIC_MARGINS = {
    "S_m": -0.008, "F_beta_w": -0.008, "E_mean": -0.008, "MAE": 0.005,
    "Precision": -0.03, "Area_min": 0.09, "Area_max": 0.155,
}
GBSP_V5_HARD_FAILURE = {
    "triple_decline": -0.01, "MAE_increase": 0.01, "Precision_drop": 0.05,
    "Area_min": 0.06, "Area_max": 0.17, "empty_rate_max": 0.02,
    "area_over_50_rate_max": 0.0,
}
GBSP_V5_STRICT_MARGINS = {
    "S_m": -0.005, "F_beta_w": -0.005, "E_mean": -0.005, "MAE": 0.003,
    "F_beta_mean": -0.010, "Precision": -0.020, "Area_min": 0.09, "Area_max": 0.15,
}
GBSP_V5_DATASET_MARGINS = {"S_m": -0.015, "F_beta_w": -0.015, "E_mean": -0.015, "MAE": 0.010}
GBSP_V5_BOOTSTRAP_MARGINS = {"S_m": -0.005, "F_beta_w": -0.005, "E_mean": -0.005, "MAE": 0.003}
GBSP_V5_BOOTSTRAP_REPETITIONS = 2000
GBSP_V5_BOOTSTRAP_SEED = 20260806

GBSP_V5_INDEPENDENCE = {
    "gt_used": False, "r1_used": False, "fixed_058_used": False,
    "target_area_used": False, "fixed_topk_used": False,
    "dino_forward_used": False, "pca_changed": False,
    "morphology_used": False, "all_bc_hard_background": False,
}
