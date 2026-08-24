"""Median-ratio graph-scale calibration derived once from fixed sample500."""

from configs.dinov1_s8_gbsp_resolution_512_native64_bw2 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_resolution_512_native64_graphcal"
GBSP_RESOLUTION_VARIANT = "512-D-GraphCal"

# Frozen Stage-1 derivation: Q50(term_512) / Q50(term_296).
DABE_SIGMA_F = 0.08407949442700827
DABE_SIGMA_C = 0.042238552325553784
DABE_SIGMA_E = 0.2522818568827296
GBSP_GRAPH_CALIBRATION_SOURCE = "fixed_sample500_seed42_median_ratio"

