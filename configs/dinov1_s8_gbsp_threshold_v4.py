"""Frozen settings for GBSP Adaptive Thresholding V4."""

GBSP_V4_VERSION = "gbsp_adaptive_threshold_v4"
GBSP_V4_GRID = 37
GBSP_V4_DIM = 384
GBSP_V4_EPS = 1e-8
GBSP_V4_TEST_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_V4_TEST20_LIST = "../workdir/gbsp_threshold_v2/test20_ids.txt"
GBSP_V4_PILOT200_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_V4_OUTPUT_ROOT = "../workdir/gbsp_threshold_v4"
GBSP_V4_BASELINE_AUDIT = "../workdir/gbsp_threshold_v4/baseline_test20/numerical_failure_summary.json"

GBSP_V4_METHODS = ("huto_core", "huto_mid_h", "huto_med_h", "orc_h", "lrc_h")
GBSP_V4_FAMILIES = {
    "huto_core": "huto", "huto_mid_h": "huto", "huto_med_h": "huto",
    "orc_h": "orc", "lrc_h": "lrc",
}
GBSP_V4_FIXED_THRESHOLD_REFERENCE_ONLY = 0.58
GBSP_V4_BASELINE_TOLERANCE = 1e-4
GBSP_V4_BASELINE_TARGETS = {
    "fixed_058": {
        "S_m": 0.7458224526724683, "F_beta_w": 0.658548512255012,
        "F_beta_mean": 0.7119134420296271, "E_mean": 0.8245257796660878,
        "MAE": 0.07604126462976965, "Precision": 0.7598750500938152,
        "Recall": 0.6870491265021301, "Area": 0.12583255814388394,
    },
    "multi_otsu_3": {
        "S_m": 0.7413144572606218, "F_beta_w": 0.6545402886032569,
        "F_beta_mean": 0.7034737331204619, "E_mean": 0.8365482375429132,
        "MAE": 0.08381385305907081, "Precision": 0.7303415899509997,
        "Recall": 0.7285906815082573, "Area": 0.13890930572524668,
    },
}

GBSP_V4_TEST20_MARGINS = {
    "S_m": -0.008, "F_beta_w": -0.010, "E_mean": -0.010, "MAE": 0.006,
    "Precision": -0.035, "Area_min": 0.07, "Area_max": 0.16,
}
GBSP_V4_TEST20_HARD_FAILURE = {
    "MAE_increase": 0.010, "Precision_drop": 0.05, "Area_max": 0.17, "Area_min": 0.04,
}
GBSP_V4_PILOT_GATE = {
    "S_m": 0.7296, "F_beta_w": 0.6255, "E_mean": 0.8185, "MAE": 0.0819,
    "F_beta_mean": 0.6703, "Precision": 0.7007, "Area_min": 0.09, "Area_max": 0.15,
}
GBSP_V4_DATASET_MARGINS = {
    "S_m": -0.015, "F_beta_w": -0.015, "E_mean": -0.015, "MAE": 0.010,
}
GBSP_V4_BOOTSTRAP_MARGINS = {
    "S_m": -0.005, "F_beta_w": -0.005, "E_mean": -0.005, "MAE": 0.003,
}
GBSP_V4_BOOTSTRAP_REPETITIONS = 2000
GBSP_V4_BOOTSTRAP_SEED = 20260806

GBSP_V4_INDEPENDENCE = {
    "gt_used": False, "r1_used": False, "fixed_058_used": False,
    "target_area_used": False, "fixed_topk_used": False, "dino_forward_used": False,
    "pca_changed": False, "morphology_used": False,
}
