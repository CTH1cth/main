import heapq
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


EPS = 1e-8

DABE_V1_DEFAULT_PARAMS = {
    "VERSION": "v1",
    "GRID": 37,
    "LOSS_SIZE": 68,
    "SIGMA_F": 0.10,
    "SIGMA_C": 0.05,
    "SIGMA_E": 0.30,
    "BORDER_WIDTH": 2,
    "TAU_BC": 0.30,
    "BG_ANCHOR_TOP_PERCENT": 30.0,
    "BG_ANCHOR_MIN_RATIO": 0.05,
    "BG_ANCHOR_FALLBACK_TOP_PERCENT": 40.0,
    "K_RECON": 32,
    "LAMBDA_COLOR_RECON": 0.20,
    "SIGMA_COLOR_RECON": 0.05,
    "TAU_RECON": 0.07,
    "GAMMA_BC_SUPPRESS": 1.0,
    "FG_PERCENTILE": 85.0,
    "FG_BC_MAX": 0.50,
    "FG_MIN_COMPONENT_RATIO": 0.005,
    "FG_MAX_COMPONENT_RATIO": 0.60,
    "FG_TOP_COMPONENTS": 3,
    "FG_FALLBACK_PERCENTILE": 90.0,
    "FG_FALLBACK_BC_MAX": 0.70,
    "BG_CORE_BC_MIN": 0.70,
    "BG_CORE_FG_MAX": 0.30,
    "BG_CORE_MIN_RATIO": 0.05,
    "PROP_ITER": 200,
    "PROP_TOL": 1e-5,
    "VIEW_AGREEMENT_VAR_DENOM": 0.25,
}

DABE_V2_DEFAULT_PARAMS = {
    **DABE_V1_DEFAULT_PARAMS,
    "VERSION": "v2",
    "BC_PENALTY_MODE": "weak",
    "BC_LAMBDA": 0.50,
    "BC_SUPPRESS_GAMMA": 1.0,
    "FG_PERCENTILE": 92.0,
    "FG_BC_MAX": 0.40,
    "FG_KEEP_TOPK": 3,
    "FG_MIN_COMPONENT_RATIO": 0.005,
    "FG_MAX_COMPONENT_RATIO": 0.60,
    "MIN_COMP_AREA": 0.005,
    "MAX_COMP_AREA": 0.60,
    "USE_COMPONENT_SCORE": True,
    "COMPONENT_SCORE_MODE": "compact_area_border",
    "USE_TWOPASS_BG": True,
    "BG_BC_MIN": 0.65,
    "BG_RESIDUAL_PERCENTILE": 30.0,
    "BG_SCORE_MAX": 0.30,
    "USE_EVIDENCE_GATE": True,
    "EVIDENCE_MODE": "multiply",
    "EVIDENCE_PERCENTILE": 70.0,
    "EVIDENCE_TAU": 0.08,
    "RW_ITER": 200,
    "RW_TOL": 1e-5,
}

DABE_V3_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "v3",
    "DABE_V3_EVIDENCE_FLOOR": 0.50,
    "DABE_V3_EXPAND_WEIGHT": 0.35,
    "DABE_V3_EXPAND_PERCENTILE": 82.0,
    "DABE_V3_EXPAND_TAU": 0.08,
    "DABE_V3_CORE_AFF_TOPK": 32,
    "DABE_V3_CORE_AFF_MIN": 0.45,
    "DABE_V3_CORE_AFF_USE_MAX": True,
    "DABE_V3_RESIDUAL_EXPAND_POWER": 0.5,
    "DABE_V3_EVIDENCE_EXPAND_POWER": 0.5,
    "DABE_V3_BG_SUPPRESS_VALUE": 0.03,
    "DABE_V3_FG_CORE_MIN_VALUE": 0.90,
    "DABE_V3_AREA_PRIOR_LOW": 0.10,
    "DABE_V3_AREA_PRIOR_HIGH": 0.22,
    "DABE_V3_AREA_PRIOR_SOFT": True,
    "DABE_V3_COMPONENT_TOPK": 5,
    "DABE_V3_COMPONENT_MIN_AREA": 4,
    "DABE_V3_COMPONENT_SEM_KEEP": 0.50,
    "DABE_V3_COMPONENT_EDGE_KEEP": 0.35,
    "DABE_V3_COMPONENT_ELONGATION_KEEP": 2.5,
    "DABE_V3_COMPONENT_WEAK_KEEP_SCALE": 0.35,
    "DABE_V3_OUTPUT_SOFT": True,
}

DABE_V31_DEFAULT_PARAMS = {
    **DABE_V3_DEFAULT_PARAMS,
    "VERSION": "v3_1",
    "DABE_V31_BASE_EVIDENCE_MODE": "strict",
    "DABE_V31_BASE_MASK_THRESH": 0.35,
    "DABE_V31_USE_LOCAL_BAND": True,
    "DABE_V31_CANDIDATE_DILATE_RADIUS": 3,
    "DABE_V31_CANDIDATE_EXCLUDE_BG_CORE": True,
    "DABE_V31_CORE_AFF_MIN": 0.58,
    "DABE_V31_RESIDUAL_PERCENTILE": 60.0,
    "DABE_V31_EVIDENCE_MIN": 0.35,
    "DABE_V31_EXPAND_PERCENTILE": 88.0,
    "DABE_V31_EXPAND_TAU": 0.05,
    "DABE_V31_EXPAND_WEIGHT": 0.20,
    "DABE_V31_USE_ADAPTIVE_EXPAND": True,
    "DABE_V31_AREA_TINY": 0.08,
    "DABE_V31_AREA_NORMAL": 0.16,
    "DABE_V31_AREA_LARGE": 0.20,
    "DABE_V31_EXPAND_WEIGHT_TINY": 0.25,
    "DABE_V31_EXPAND_WEIGHT_NORMAL": 0.20,
    "DABE_V31_EXPAND_WEIGHT_LARGE": 0.10,
    "DABE_V31_EXPAND_WEIGHT_XLARGE": 0.05,
    "DABE_V31_DILATE_RADIUS_TINY": 3,
    "DABE_V31_DILATE_RADIUS_NORMAL": 3,
    "DABE_V31_DILATE_RADIUS_LARGE": 2,
    "DABE_V31_DILATE_RADIUS_XLARGE": 1,
    "DABE_V31_FG_CORE_MIN_VALUE": 0.90,
    "DABE_V31_BG_SUPPRESS_VALUE": 0.03,
    "DABE_V31_COMPONENT_TOPK": 5,
    "DABE_V31_COMPONENT_MIN_AREA": 4,
    "DABE_V31_COMPONENT_SEM_KEEP": 0.58,
    "DABE_V31_COMPONENT_EDGE_KEEP": 0.35,
    "DABE_V31_COMPONENT_ELONGATION_KEEP": 2.5,
    "DABE_V31_COMPONENT_MAX_DIST_TO_BASE": 3,
    "DABE_V31_COMPONENT_WEAK_KEEP_SCALE": 0.0,
}

DABE_GC_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "gc",
    "DABE_GC_VERSION": "gc",
    "DABE_GC_BASE_MODE": "strict",
    "DABE_GC_BASE_THRESH": 0.35,
    "DABE_GC_FG_SEED_FROM_FG_CORE": True,
    "DABE_GC_FG_SEED_SCORE_PERCENTILE": 92.0,
    "DABE_GC_FG_SEED_EVIDENCE_MIN": 0.45,
    "DABE_GC_FG_SEED_RESIDUAL_PERCENTILE": 70.0,
    "DABE_GC_FG_SEED_MIN_PIXELS": 4,
    "DABE_GC_BG_SEED_FROM_BG_CORE": True,
    "DABE_GC_BG_SEED_LOW_RESIDUAL_PERCENTILE": 25.0,
    "DABE_GC_BG_SEED_BC_MIN": 0.60,
    "DABE_GC_BG_SEED_BORDER_WIDTH": 1,
    "DABE_GC_BG_SEED_MIN_PIXELS": 16,
    "DABE_GC_GRAPH_RADIUS": 1,
    "DABE_GC_USE_RADIUS2": False,
    "DABE_GC_USE_SEM_AFF": True,
    "DABE_GC_USE_COLOR_AFF": True,
    "DABE_GC_USE_EDGE_STOP": True,
    "DABE_GC_SEM_POWER": 1.0,
    "DABE_GC_COLOR_SIGMA": 0.12,
    "DABE_GC_EDGE_SIGMA": 0.25,
    "DABE_GC_SPATIAL_SIGMA": 1.5,
    "DABE_GC_USE_LOCAL_TOPK": False,
    "DABE_GC_LOCAL_TOPK": 4,
    "DABE_GC_LOCAL_TOPK_RADIUS": 3,
    "DABE_GC_LOCAL_TOPK_WEIGHT": 0.25,
    "DABE_GC_DIFFUSE_ITERS": 25,
    "DABE_GC_DIFFUSE_MU": 0.85,
    "DABE_GC_SEED_LOCK": True,
    "DABE_GC_USE_BASE_AS_PRIOR": True,
    "DABE_GC_PRIOR_WEIGHT": 0.10,
    "DABE_GC_COMPLETION_MODE": "positive_delta",
    "DABE_GC_BLEND_WEIGHT": 0.65,
    "DABE_GC_CONF_EVIDENCE_POWER": 0.5,
    "DABE_GC_CONF_AFF_POWER": 0.5,
    "DABE_GC_DELTA_CLAMP_MIN": 0.0,
    "DABE_GC_FG_SEED_VALUE": 0.95,
    "DABE_GC_BG_SEED_VALUE": 0.02,
    "DABE_GC_BG_CORE_SUPPRESS": 0.03,
    "DABE_GC_AREA_LOW": 0.08,
    "DABE_GC_AREA_HIGH": 0.22,
    "DABE_GC_AREA_SUPPRESS_HIGH": True,
    "DABE_GC_AREA_SUPPRESS_MIN_FACTOR": 0.75,
    "DABE_GC_COMPONENT_MIN_AREA": 4,
    "DABE_GC_COMPONENT_TOPK": 5,
    "DABE_GC_COMPONENT_KEEP_SEED": True,
    "DABE_GC_COMPONENT_SEM_KEEP": 0.55,
    "DABE_GC_COMPONENT_EDGE_KEEP": 0.35,
    "DABE_GC_COMPONENT_ELONGATION_KEEP": 2.5,
    "DABE_GC_COMPONENT_MAX_DIST_TO_FG": 3,
    "DABE_GC_COMPONENT_WEAK_KEEP_SCALE": 0.0,
    "DABE_GC_VIEW_AGREEMENT_POWER": 1.0,
    "DABE_GC_OUTPUT_SOFT": True,
}

DABE_RAC_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "rac",
    "DABE_RAC_VERSION": "rac",
    "DABE_RAC_BASE_MODE": "strict",
    "DABE_RAC_BASE_MASK_THRESH": 0.35,
    "DABE_RAC_FG_SEED_FROM_FG_CORE": True,
    "DABE_RAC_FG_SEED_SCORE_PERCENTILE": 92.0,
    "DABE_RAC_FG_SEED_RESIDUAL_PERCENTILE": 70.0,
    "DABE_RAC_FG_SEED_EVIDENCE_MIN": 0.45,
    "DABE_RAC_FG_SEED_MIN_PIXELS": 4,
    "DABE_RAC_BG_SEED_FROM_BG_CORE": True,
    "DABE_RAC_BG_SEED_BC_MIN": 0.60,
    "DABE_RAC_BG_SEED_LOW_RESIDUAL_PERCENTILE": 25.0,
    "DABE_RAC_BG_SEED_BORDER_WIDTH": 1,
    "DABE_RAC_BG_SEED_MIN_PIXELS": 16,
    "DABE_RAC_REGION_RADIUS": 2,
    "DABE_RAC_MAX_REGION_STEPS": 6,
    "DABE_RAC_REGION_SEM_MIN": 0.50,
    "DABE_RAC_REGION_COLOR_MIN": 0.35,
    "DABE_RAC_REGION_EDGE_MAX": 0.65,
    "DABE_RAC_REGION_BG_BLOCK": True,
    "DABE_RAC_COLOR_SIGMA": 0.12,
    "DABE_RAC_EDGE_SIGMA": 0.25,
    "DABE_RAC_SEM_POWER": 1.0,
    "DABE_RAC_COLOR_POWER": 1.0,
    "DABE_RAC_EDGE_POWER": 1.0,
    "DABE_RAC_REGION_MIN_AREA": 4,
    "DABE_RAC_REGION_MAX_AREA_RATIO": 0.35,
    "DABE_RAC_REGION_TOPK": 8,
    "DABE_RAC_CONF_SEED_WEIGHT": 1.00,
    "DABE_RAC_CONF_SEM_WEIGHT": 1.00,
    "DABE_RAC_CONF_RESIDUAL_WEIGHT": 0.80,
    "DABE_RAC_CONF_EVIDENCE_WEIGHT": 0.80,
    "DABE_RAC_CONF_COLOR_WEIGHT": 0.50,
    "DABE_RAC_CONF_EDGE_WEIGHT": 0.50,
    "DABE_RAC_CONF_BG_PENALTY": 1.50,
    "DABE_RAC_CONF_BORDER_PENALTY": 0.80,
    "DABE_RAC_REGION_KEEP_THRESH": 0.42,
    "DABE_RAC_REGION_STRONG_KEEP_THRESH": 0.60,
    "DABE_RAC_REGION_WEAK_KEEP_SCALE": 0.35,
    "DABE_RAC_COMPLETION_WEIGHT": 0.55,
    "DABE_RAC_WEAK_COMPLETION_WEIGHT": 0.25,
    "DABE_RAC_POSITIVE_ONLY": True,
    "DABE_RAC_FG_SEED_VALUE": 0.95,
    "DABE_RAC_BG_SEED_VALUE": 0.02,
    "DABE_RAC_BG_CORE_SUPPRESS": 0.03,
    "DABE_RAC_AREA_LOW": 0.10,
    "DABE_RAC_AREA_TARGET_LOW": 0.14,
    "DABE_RAC_AREA_TARGET_HIGH": 0.20,
    "DABE_RAC_AREA_HIGH": 0.24,
    "DABE_RAC_AREA_SUPPRESS_HIGH": True,
    "DABE_RAC_AREA_SUPPRESS_MIN_FACTOR": 0.75,
    "DABE_RAC_COMPONENT_MIN_AREA": 4,
    "DABE_RAC_COMPONENT_TOPK": 5,
    "DABE_RAC_COMPONENT_KEEP_SEED": True,
    "DABE_RAC_COMPONENT_SEM_KEEP": 0.55,
    "DABE_RAC_COMPONENT_EDGE_KEEP": 0.35,
    "DABE_RAC_COMPONENT_ELONGATION_KEEP": 2.5,
    "DABE_RAC_COMPONENT_MAX_DIST_TO_FG": 3,
    "DABE_RAC_COMPONENT_WEAK_KEEP_SCALE": 0.0,
    "DABE_RAC_VIEW_AGREEMENT_POWER": 1.0,
    "DABE_RAC_OUTPUT_SOFT": True,
}

DABE_RAC_SAFE_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "rac_safe",
    "DABE_RAC_SAFE_VERSION": "rac_safe",
    "DABE_RAC_SAFE_BASE_MODE": "strict",
    "DABE_RAC_SAFE_BASE_MASK_THRESH": 0.30,
    "DABE_RAC_SAFE_USE_LOCAL_BAND": True,
    "DABE_RAC_SAFE_LOCAL_BAND_RADIUS": 3,
    "DABE_RAC_SAFE_LOCAL_BAND_EXCLUDE_BG_CORE": True,
    "DABE_RAC_SAFE_FG_SEED_FROM_FG_CORE": True,
    "DABE_RAC_SAFE_FG_SEED_SCORE_PERCENTILE": 92.0,
    "DABE_RAC_SAFE_FG_SEED_RESIDUAL_PERCENTILE": 70.0,
    "DABE_RAC_SAFE_FG_SEED_EVIDENCE_MIN": 0.45,
    "DABE_RAC_SAFE_FG_SEED_MIN_PIXELS": 4,
    "DABE_RAC_SAFE_BG_SEED_FROM_BG_CORE": True,
    "DABE_RAC_SAFE_BG_SEED_BC_MIN": 0.60,
    "DABE_RAC_SAFE_BG_SEED_LOW_RESIDUAL_PERCENTILE": 25.0,
    "DABE_RAC_SAFE_BG_SEED_BORDER_WIDTH": 1,
    "DABE_RAC_SAFE_BG_SEED_MIN_PIXELS": 16,
    "DABE_RAC_SAFE_REGION_RADIUS": 1,
    "DABE_RAC_SAFE_MAX_REGION_STEPS": 3,
    "DABE_RAC_SAFE_REGION_SEM_MIN": 0.58,
    "DABE_RAC_SAFE_REGION_COLOR_MIN": 0.45,
    "DABE_RAC_SAFE_REGION_EDGE_MAX": 0.45,
    "DABE_RAC_SAFE_REGION_BG_BLOCK": True,
    "DABE_RAC_SAFE_REGION_MAX_AREA_RATIO": 0.18,
    "DABE_RAC_SAFE_REGION_MAX_REL_TO_BASE_COMP": 1.5,
    "DABE_RAC_SAFE_COLOR_SIGMA": 0.12,
    "DABE_RAC_SAFE_EDGE_SIGMA": 0.25,
    "DABE_RAC_SAFE_SEM_POWER": 1.0,
    "DABE_RAC_SAFE_COLOR_POWER": 1.0,
    "DABE_RAC_SAFE_EDGE_POWER": 1.0,
    "DABE_RAC_SAFE_REJECT_BG_OVERLAP": 0.02,
    "DABE_RAC_SAFE_REJECT_BORDER_TOUCH": 0.35,
    "DABE_RAC_SAFE_REJECT_EVIDENCE_MIN": 0.30,
    "DABE_RAC_SAFE_REJECT_REQUIRE_FG_SEED": True,
    "DABE_RAC_SAFE_CONF_SEM_POWER": 1.5,
    "DABE_RAC_SAFE_CONF_BG_POWER": 2.0,
    "DABE_RAC_SAFE_CONF_BORDER_POWER": 1.5,
    "DABE_RAC_SAFE_REGION_KEEP_THRESH": 0.45,
    "DABE_RAC_SAFE_PIXEL_RESIDUAL_PERCENTILE": 55.0,
    "DABE_RAC_SAFE_PIXEL_EVIDENCE_MIN": 0.30,
    "DABE_RAC_SAFE_PIXEL_AFF_MIN": 0.50,
    "DABE_RAC_SAFE_PIXEL_EDGE_MAX": 0.60,
    "DABE_RAC_SAFE_PIXEL_GATE_POWER": 1.0,
    "DABE_RAC_SAFE_COMPLETION_WEIGHT": 0.45,
    "DABE_RAC_SAFE_POSITIVE_ONLY": True,
    "DABE_RAC_SAFE_DISABLE_WEAK_REGION": True,
    "DABE_RAC_SAFE_DELTA_BUDGET_ABS": 0.03,
    "DABE_RAC_SAFE_DELTA_BUDGET_REL": 0.25,
    "DABE_RAC_SAFE_TARGET_AREA_LOW": 0.13,
    "DABE_RAC_SAFE_TARGET_AREA_HIGH": 0.155,
    "DABE_RAC_SAFE_USE_TOP_DELTA_BUDGET": True,
    "DABE_RAC_SAFE_FG_SEED_VALUE": 0.95,
    "DABE_RAC_SAFE_BG_SEED_VALUE": 0.02,
    "DABE_RAC_SAFE_BG_CORE_SUPPRESS": 0.03,
    "DABE_RAC_SAFE_COMPONENT_MIN_AREA": 4,
    "DABE_RAC_SAFE_COMPONENT_TOPK": 5,
    "DABE_RAC_SAFE_COMPONENT_KEEP_SEED": True,
    "DABE_RAC_SAFE_COMPONENT_SEM_KEEP": 0.58,
    "DABE_RAC_SAFE_COMPONENT_EDGE_KEEP": 0.35,
    "DABE_RAC_SAFE_COMPONENT_ELONGATION_KEEP": 2.5,
    "DABE_RAC_SAFE_COMPONENT_MAX_DIST_TO_FG": 3,
    "DABE_RAC_SAFE_COMPONENT_WEAK_KEEP_SCALE": 0.0,
    "DABE_RAC_SAFE_VIEW_AGREEMENT_POWER": 1.0,
    "DABE_RAC_SAFE_OUTPUT_SOFT": True,
}

DABE_PU_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "pu",
    "DABE_PU_VERSION": "pu",
    "DABE_PU_BASE_MODE": "strict",
    "DABE_PU_BASE_FG_THRESH": 0.35,
    "DABE_PU_BASE_LOW_THRESH": 0.10,
    "DABE_PU_FG_CORE_FROM_V2": True,
    "DABE_PU_FG_CORE_P_BASE_THRESH": 0.55,
    "DABE_PU_FG_CORE_EVIDENCE_MIN": 0.50,
    "DABE_PU_FG_CORE_RESIDUAL_PERCENTILE": 70.0,
    "DABE_PU_FG_CORE_FG_SCORE_PERCENTILE": 88.0,
    "DABE_PU_FG_CORE_MIN_PIXELS": 4,
    "DABE_PU_BG_CORE_FROM_V2": True,
    "DABE_PU_BG_CORE_BC_MIN": 0.65,
    "DABE_PU_BG_CORE_RESIDUAL_PERCENTILE": 25.0,
    "DABE_PU_BG_CORE_P_BASE_MAX": 0.15,
    "DABE_PU_BG_CORE_BORDER_WIDTH": 1,
    "DABE_PU_BG_CORE_MIN_PIXELS": 16,
    "DABE_PU_EXTENT_USE_LOCAL_BAND": True,
    "DABE_PU_EXTENT_BAND_BASE_THRESH": 0.18,
    "DABE_PU_EXTENT_BAND_RADIUS": 3,
    "DABE_PU_EXTENT_EXCLUDE_BG_CORE": True,
    "DABE_PU_EXTENT_EVIDENCE_MIN": 0.20,
    "DABE_PU_EXTENT_RESIDUAL_PERCENTILE": 45.0,
    "DABE_PU_EXTENT_FG_SCORE_PERCENTILE": 45.0,
    "DABE_PU_EXTENT_EDGE_MAX": 0.75,
    "DABE_PU_USE_CORE_AFFINITY": True,
    "DABE_PU_CORE_AFF_MIN": 0.45,
    "DABE_PU_CORE_AFF_WEIGHT": 0.35,
    "DABE_PU_TARGET_FG_CORE": 1.00,
    "DABE_PU_TARGET_BG_CORE": 0.00,
    "DABE_PU_TARGET_EXTENT_MIN": 0.35,
    "DABE_PU_TARGET_EXTENT_MAX": 0.55,
    "DABE_PU_TARGET_UNKNOWN": 0.50,
    "DABE_PU_WEIGHT_FG_CORE": 1.00,
    "DABE_PU_WEIGHT_BG_CORE": 1.00,
    "DABE_PU_WEIGHT_EXTENT": 0.20,
    "DABE_PU_WEIGHT_UNKNOWN": 0.02,
    "DABE_PU_WEIGHT_SOFT_BASE": 0.10,
    "DABE_PU_EXTENT_MIN_COMPONENT_AREA": 3,
    "DABE_PU_REMOVE_ISOLATED_EXTENT": True,
    "DABE_PU_EXTENT_MAX_AREA_RATIO": 0.35,
    "DABE_PU_OUTPUT_SOFT": True,
}

DABE_PU_V11_DEFAULT_PARAMS = {
    **DABE_V2_DEFAULT_PARAMS,
    "VERSION": "pu_v11",
    "DABE_PU_V11_VERSION": "pu_v11",
    "DABE_PU_V11_BASE_MODE": "strict",
    "DABE_PU_V11_BASE_FG_THRESH": 0.35,
    "DABE_PU_V11_BASE_LOW_THRESH": 0.10,
    "DABE_PU_V11_USE_V2_FG_CORE_AS_CANDIDATE": True,
    "DABE_PU_V11_FG_CORE_P_BASE_THRESH": 0.55,
    "DABE_PU_V11_FG_CORE_EVIDENCE_MIN": 0.50,
    "DABE_PU_V11_FG_CORE_RESIDUAL_PERCENTILE": 70.0,
    "DABE_PU_V11_FG_CORE_FG_SCORE_PERCENTILE": 88.0,
    "DABE_PU_V11_FG_CORE_BGCORE_MAX": 0.0,
    "DABE_PU_V11_FG_CORE_MIN_PIXELS": 4,
    "DABE_PU_V11_USE_FG_CORE_RELIABILITY": True,
    "DABE_PU_V11_FG_REL_EVIDENCE_WEIGHT": 0.30,
    "DABE_PU_V11_FG_REL_RESIDUAL_WEIGHT": 0.30,
    "DABE_PU_V11_FG_REL_FG_SCORE_WEIGHT": 0.25,
    "DABE_PU_V11_FG_REL_COMPACTNESS_WEIGHT": 0.15,
    "DABE_PU_V11_FG_CORE_WEIGHT_MIN": 0.45,
    "DABE_PU_V11_FG_CORE_WEIGHT_MAX": 1.00,
    "DABE_PU_V11_FG_CORE_TARGET": 1.00,
    "DABE_PU_V11_FG_CORE_FALLBACK_TARGET": 0.85,
    "DABE_PU_V11_FG_CORE_FALLBACK_WEIGHT": 0.35,
    "DABE_PU_V11_BG_CORE_FROM_V2": True,
    "DABE_PU_V11_BG_CORE_BC_MIN": 0.65,
    "DABE_PU_V11_BG_CORE_RESIDUAL_PERCENTILE": 25.0,
    "DABE_PU_V11_BG_CORE_P_BASE_MAX": 0.15,
    "DABE_PU_V11_BG_CORE_BORDER_WIDTH": 1,
    "DABE_PU_V11_BG_CORE_MIN_PIXELS": 16,
    "DABE_PU_V11_BG_CORE_TARGET": 0.00,
    "DABE_PU_V11_BG_CORE_WEIGHT": 1.00,
    "DABE_PU_V11_EXTENT_USE_LOCAL_BAND": True,
    "DABE_PU_V11_EXTENT_BAND_BASE_THRESH": 0.18,
    "DABE_PU_V11_EXTENT_BAND_RADIUS": 3,
    "DABE_PU_V11_EXTENT_EXCLUDE_BG_CORE": True,
    "DABE_PU_V11_EXTENT_EVIDENCE_MIN": 0.20,
    "DABE_PU_V11_EXTENT_RESIDUAL_PERCENTILE": 45.0,
    "DABE_PU_V11_EXTENT_FG_SCORE_PERCENTILE": 45.0,
    "DABE_PU_V11_EXTENT_EDGE_MAX": 0.75,
    "DABE_PU_V11_USE_CORE_AFFINITY": True,
    "DABE_PU_V11_CORE_AFF_MIN": 0.45,
    "DABE_PU_V11_CORE_AFF_WEIGHT": 0.35,
    "DABE_PU_V11_TARGET_EXTENT": 0.50,
    "DABE_PU_V11_WEIGHT_EXTENT": 0.08,
    "DABE_PU_V11_WEIGHT_EXTENT_MIN": 0.05,
    "DABE_PU_V11_WEIGHT_EXTENT_MAX": 0.10,
    "DABE_PU_V11_TARGET_UNKNOWN": 0.50,
    "DABE_PU_V11_WEIGHT_UNKNOWN": 0.00,
    "DABE_PU_V11_USE_SOFT_BASE_WEIGHT": True,
    "DABE_PU_V11_WEIGHT_SOFT_BASE": 0.05,
    "DABE_PU_V11_EXTENT_MIN_COMPONENT_AREA": 3,
    "DABE_PU_V11_REMOVE_ISOLATED_EXTENT": True,
    "DABE_PU_V11_EXTENT_MAX_AREA_RATIO": 0.30,
    "DABE_PU_V11_EXTENT_REQUIRE_NEAR_BASE": True,
    "DABE_PU_V11_EXTENT_MAX_DIST_TO_BASE": 2,
    "DABE_PU_V11_OUTPUT_SOFT": True,
}

DABE_DEFAULT_PARAMS = dict(DABE_V2_DEFAULT_PARAMS)

VALID_AUGS = {"identity", "hflip", "vflip", "rot180"}

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def _merge_params(params):
    version = "v2"
    if params:
        version = str(params.get("VERSION", params.get("DABE_VERSION", "v2"))).lower()
    if version == "v1":
        merged = dict(DABE_V1_DEFAULT_PARAMS)
    elif version == "v2":
        merged = dict(DABE_V2_DEFAULT_PARAMS)
    elif version == "v3":
        merged = dict(DABE_V3_DEFAULT_PARAMS)
    elif version == "v3_1":
        merged = dict(DABE_V31_DEFAULT_PARAMS)
    elif version == "gc":
        merged = dict(DABE_GC_DEFAULT_PARAMS)
    elif version == "rac":
        merged = dict(DABE_RAC_DEFAULT_PARAMS)
    elif version == "rac_safe":
        merged = dict(DABE_RAC_SAFE_DEFAULT_PARAMS)
    elif version == "pu":
        merged = dict(DABE_PU_DEFAULT_PARAMS)
    elif version == "pu_v11":
        merged = dict(DABE_PU_V11_DEFAULT_PARAMS)
    else:
        raise ValueError(f"Unsupported DABE VERSION: {version}")
    if params:
        for key, value in params.items():
            if key.isupper():
                merged[key] = value
    merged["VERSION"] = str(merged.get("VERSION", version)).lower()
    return merged


def _parse_augs(augs):
    if augs is None:
        parsed = ["identity", "hflip", "vflip", "rot180"]
    elif isinstance(augs, str):
        parsed = [item.strip().lower() for item in augs.split(",") if item.strip()]
    else:
        parsed = [str(item).strip().lower() for item in augs if str(item).strip()]
    if not parsed:
        raise ValueError("DABE augs must not be empty.")
    bad = [aug for aug in parsed if aug not in VALID_AUGS]
    if bad:
        raise ValueError(f"Unsupported DABE augmentation(s): {bad}")
    return parsed


def _validate_feature(feature, grid):
    if not torch.is_tensor(feature):
        raise TypeError("DABE feature must be a torch.Tensor.")
    feature = feature.detach().cpu().float()
    if feature.ndim != 3 or int(feature.shape[0]) <= 0:
        raise RuntimeError(
            "DABE expects feature shape [C,grid,grid] with C>0, "
            f"got {list(feature.shape)}"
        )
    expected_spatial = [int(grid), int(grid)]
    if list(feature.shape[-2:]) != expected_spatial:
        raise RuntimeError(
            f"DABE expects feature spatial shape {expected_spatial}, "
            f"got {list(feature.shape[-2:])} from {list(feature.shape)}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError("DABE feature contains NaN/Inf.")
    return feature.contiguous()


def _load_rgb_grid(image_path, grid):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    resized = image.resize((int(grid), int(grid)), RESAMPLE_BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _apply_aug(tensor, aug):
    if aug == "identity":
        return tensor
    if aug == "hflip":
        return torch.flip(tensor, dims=[-1])
    if aug == "vflip":
        return torch.flip(tensor, dims=[-2])
    if aug == "rot180":
        return torch.flip(tensor, dims=[-2, -1])
    raise ValueError(f"Unsupported DABE augmentation: {aug}")


def _minmax(tensor):
    tensor = tensor.float()
    tmin = tensor.min()
    tmax = tensor.max()
    if float((tmax - tmin).abs().item()) < EPS:
        return torch.zeros_like(tensor)
    return ((tensor - tmin) / (tmax - tmin + EPS)).clamp(0.0, 1.0)


def _sobel_magnitude(rgb):
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    gray4 = gray.view(1, 1, *gray.shape)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    padded = F.pad(gray4, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    mag = torch.sqrt(gx.square() + gy.square() + EPS).squeeze(0).squeeze(0)
    return _minmax(mag)


def _border_mask(grid, width):
    mask = torch.zeros((int(grid), int(grid)), dtype=torch.bool)
    width = max(1, int(width))
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask.reshape(-1)


def _build_local_graph(feat_n, rgb_n, edge_n, grid, params):
    grid = int(grid)
    num_nodes = grid * grid
    neigh_idx = torch.zeros((num_nodes, 8), dtype=torch.long)
    neigh_weight = torch.zeros((num_nodes, 8), dtype=torch.float32)
    sigma_f = float(params["SIGMA_F"])
    sigma_c = float(params["SIGMA_C"])
    sigma_e = float(params["SIGMA_E"])

    for y in range(grid):
        for x in range(grid):
            src = y * grid + x
            slot = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                        continue
                    dst = ny * grid + nx
                    df = 1.0 - float(torch.dot(feat_n[src], feat_n[dst]).item())
                    dc = float(torch.sum((rgb_n[src] - rgb_n[dst]).square()).item())
                    de = float(torch.maximum(edge_n[src], edge_n[dst]).item())
                    weight = np.exp(-df / sigma_f - dc / sigma_c - de / sigma_e)
                    neigh_idx[src, slot] = dst
                    neigh_weight[src, slot] = max(float(weight), EPS)
                    slot += 1
    return neigh_idx, neigh_weight


def _background_connectivity(neigh_idx, neigh_weight, grid, params):
    num_nodes = int(grid) * int(grid)
    border = _border_mask(grid, int(params["BORDER_WIDTH"]))
    distances = np.full(num_nodes, np.inf, dtype=np.float64)
    heap = []
    for node in torch.where(border)[0].tolist():
        distances[node] = 0.0
        heapq.heappush(heap, (0.0, int(node)))

    idx_np = neigh_idx.numpy()
    weight_np = neigh_weight.numpy()
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for slot in range(weight_np.shape[1]):
            weight = float(weight_np[node, slot])
            if weight <= 0.0:
                continue
            nxt = int(idx_np[node, slot])
            new_dist = dist - np.log(weight + EPS)
            if new_dist < distances[nxt]:
                distances[nxt] = new_dist
                heapq.heappush(heap, (new_dist, nxt))

    finite = np.isfinite(distances)
    if not finite.all():
        fill = float(np.max(distances[finite])) if finite.any() else 0.0
        distances[~finite] = fill
    max_dist = float(np.max(distances))
    norm_dist = distances / (max_dist + EPS) if max_dist > EPS else distances
    bc = np.exp(-norm_dist / float(params["TAU_BC"]))
    return torch.from_numpy(bc.astype(np.float32)).clamp(0.0, 1.0), border


def _top_percent_mask(values, top_percent):
    threshold = torch.quantile(values.float(), max(0.0, min(1.0, 1.0 - float(top_percent) / 100.0)))
    return values >= threshold


def _background_anchor(bc, border, params):
    num_nodes = int(bc.numel())
    anchor = _top_percent_mask(bc, float(params["BG_ANCHOR_TOP_PERCENT"]))
    min_count = max(1, int(round(float(params["BG_ANCHOR_MIN_RATIO"]) * num_nodes)))
    if int(anchor.sum().item()) < min_count:
        anchor = border | _top_percent_mask(bc, float(params["BG_ANCHOR_FALLBACK_TOP_PERCENT"]))
    return anchor.bool()


def _background_residual(feat_n, rgb_n, anchor, params):
    anchor_idx = torch.where(anchor)[0]
    if anchor_idx.numel() == 0:
        return torch.ones((feat_n.shape[0],), dtype=torch.float32)

    feat_anchor = feat_n[anchor_idx]
    rgb_anchor = rgb_n[anchor_idx]
    k = min(int(params["K_RECON"]), int(anchor_idx.numel()))
    chunk_size = 512
    feat_hat_chunks = []
    rgb_hat_chunks = []
    for start in range(0, feat_n.shape[0], chunk_size):
        end = min(start + chunk_size, feat_n.shape[0])
        feat_chunk = feat_n[start:end]
        rgb_chunk = rgb_n[start:end]
        sim_feat = feat_chunk @ feat_anchor.t()
        color_dist2 = torch.cdist(rgb_chunk, rgb_anchor, p=2.0).square()
        sim_color = torch.exp(-color_dist2 / float(params["SIGMA_COLOR_RECON"]))
        sim = sim_feat + float(params["LAMBDA_COLOR_RECON"]) * sim_color
        top_val, top_idx = torch.topk(sim, k=k, dim=1)
        weights = torch.softmax(top_val / float(params["TAU_RECON"]), dim=1)
        feat_top = feat_anchor[top_idx]
        rgb_top = rgb_anchor[top_idx]
        feat_hat_chunks.append((weights.unsqueeze(-1) * feat_top).sum(dim=1))
        rgb_hat_chunks.append((weights.unsqueeze(-1) * rgb_top).sum(dim=1))

    feat_hat = F.normalize(torch.cat(feat_hat_chunks, dim=0), dim=1, p=2)
    rgb_hat = torch.cat(rgb_hat_chunks, dim=0)
    r_f = (1.0 - (feat_n * feat_hat).sum(dim=1)).clamp_min(0.0)
    r_c = torch.linalg.norm(rgb_n - rgb_hat, dim=1)
    return _minmax(r_f + 0.2 * r_c)


def _select_fg_core(fg_score, bc, grid, params):
    num_nodes = int(fg_score.numel())
    fallback_flag = False
    fallback_reason = "none"
    threshold = torch.quantile(fg_score.float(), float(params["FG_PERCENTILE"]) / 100.0)
    candidate = ((fg_score > threshold) & (bc < float(params["FG_BC_MAX"]))).reshape(grid, grid)

    labels, num_labels = ndimage.label(
        candidate.cpu().numpy().astype(np.uint8),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    components = []
    min_area = max(1, int(round(float(params["FG_MIN_COMPONENT_RATIO"]) * num_nodes)))
    max_area = max(1, int(round(float(params["FG_MAX_COMPONENT_RATIO"]) * num_nodes)))
    score_np = fg_score.reshape(grid, grid).cpu().numpy()
    for label_id in range(1, int(num_labels) + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_area or area > max_area:
            continue
        components.append((float(score_np[component].mean()), label_id))

    fg_core_np = np.zeros((grid, grid), dtype=bool)
    components.sort(reverse=True)
    for _score, label_id in components[: int(params["FG_TOP_COMPONENTS"])]:
        fg_core_np |= labels == label_id
    fg_core = torch.from_numpy(fg_core_np.reshape(-1))

    if int(fg_core.sum().item()) == 0:
        fallback_flag = True
        fallback_reason = "empty_fg_core"
        weak_threshold = torch.quantile(
            fg_score.float(),
            float(params["FG_FALLBACK_PERCENTILE"]) / 100.0,
        )
        fg_core = (fg_score > weak_threshold) & (bc < float(params["FG_FALLBACK_BC_MAX"]))
        if int(fg_core.sum().item()) == 0:
            fallback_reason = "empty_fg_core_all_zero"

    return fg_core.bool(), bool(fallback_flag), fallback_reason


def _select_bg_core(bc, fg_score, fg_core, bg_anchor, border, params):
    bg_core = (
        (bc > float(params["BG_CORE_BC_MIN"]))
        & (fg_score < float(params["BG_CORE_FG_MAX"]))
        & ~fg_core
    )
    min_count = max(1, int(round(float(params["BG_CORE_MIN_RATIO"]) * int(bc.numel()))))
    if int(bg_core.sum().item()) < min_count:
        bg_core = bg_anchor & ~fg_core
    if int(bg_core.sum().item()) == 0:
        bg_core = border & ~fg_core
    return bg_core.bool()


def _component_area_bounds(params):
    min_ratio = float(params.get("MIN_COMP_AREA", params.get("FG_MIN_COMPONENT_RATIO", 0.005)))
    max_ratio = float(params.get("MAX_COMP_AREA", params.get("FG_MAX_COMPONENT_RATIO", 0.60)))
    return min_ratio, max_ratio


def _num_components(mask, grid):
    labels, num_labels = ndimage.label(
        mask.reshape(grid, grid).detach().cpu().numpy().astype(np.uint8),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    del labels
    return int(num_labels)


def _area_prior(area_ratio, min_ratio, max_ratio):
    midpoint = 0.5 * (min_ratio + max_ratio)
    half_span = max(0.5 * (max_ratio - min_ratio), EPS)
    return max(0.05, min(1.0, 1.0 - abs(float(area_ratio) - midpoint) / half_span))


def _component_border_touch_ratio(component):
    area = max(int(component.sum()), 1)
    border_count = (
        int(component[0, :].sum())
        + int(component[-1, :].sum())
        + int(component[:, 0].sum())
        + int(component[:, -1].sum())
    )
    return min(1.0, float(border_count) / float(area))


def _component_dino_compactness(feat_n, component_flat):
    indices = torch.where(component_flat)[0]
    if indices.numel() <= 1:
        return 1.0
    comp_feat = feat_n[indices]
    mean_feat = F.normalize(comp_feat.mean(dim=0, keepdim=True), dim=1, p=2).squeeze(0)
    compact = torch.mv(comp_feat, mean_feat).mean()
    return float(((compact + 1.0) * 0.5).clamp(0.0, 1.0).item())


def _select_fg_core_v2(feat_n, fg_score, bc, grid, params):
    num_nodes = int(fg_score.numel())
    fallback_flag = False
    fallback_reason = "none"
    threshold = torch.quantile(fg_score.float(), float(params["FG_PERCENTILE"]) / 100.0)
    candidate = ((fg_score > threshold) & (bc < float(params["FG_BC_MAX"]))).reshape(grid, grid)

    labels, num_labels = ndimage.label(
        candidate.cpu().numpy().astype(np.uint8),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    min_ratio, max_ratio = _component_area_bounds(params)
    min_area = max(1, int(round(min_ratio * num_nodes)))
    max_area = max(1, int(round(max_ratio * num_nodes)))
    score_np = fg_score.reshape(grid, grid).cpu().numpy()
    components = []

    for label_id in range(1, int(num_labels) + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_area or area > max_area:
            continue
        component_flat = torch.from_numpy(component.reshape(-1)).bool()
        mean_score = float(score_np[component].mean())
        if bool(params.get("USE_COMPONENT_SCORE", True)):
            compactness = _component_dino_compactness(feat_n, component_flat)
            border_score = 1.0 - _component_border_touch_ratio(component)
            area_score = _area_prior(area / float(num_nodes), min_ratio, max_ratio)
            score = mean_score * compactness * max(0.0, border_score) * area_score
        else:
            score = mean_score
        components.append((float(score), label_id))

    fg_core_np = np.zeros((grid, grid), dtype=bool)
    components.sort(reverse=True)
    keep_topk = int(params.get("FG_KEEP_TOPK", params.get("FG_TOP_COMPONENTS", 3)))
    for _score, label_id in components[:keep_topk]:
        fg_core_np |= labels == label_id
    fg_core = torch.from_numpy(fg_core_np.reshape(-1))

    if int(fg_core.sum().item()) == 0:
        fallback_flag = True
        fallback_reason = "empty_fg_core"
        weak_threshold = torch.quantile(fg_score.float(), 0.90)
        fg_core = (fg_score > weak_threshold) & (bc < float(params.get("FG_FALLBACK_BC_MAX", 0.70)))
        if int(fg_core.sum().item()) == 0:
            fallback_reason = "empty_fg_core_all_zero"

    component_count = _num_components(fg_core.bool(), grid) if int(fg_core.sum().item()) > 0 else 0
    return fg_core.bool(), bool(fallback_flag), fallback_reason, int(component_count)


def _weak_bc_fg_score(residual, bc, params):
    mode = str(params.get("BC_PENALTY_MODE", "weak")).lower()
    if mode != "weak":
        raise ValueError(f"Unsupported DABE v2 BC_PENALTY_MODE: {mode}")
    penalty = 1.0 - float(params.get("BC_LAMBDA", 0.5)) * bc
    penalty = penalty.clamp(0.0, 1.0)
    gamma = float(params.get("BC_SUPPRESS_GAMMA", 1.0))
    return (residual * penalty.pow(gamma)).clamp(0.0, 1.0)


def _build_bg_core_pass2(bc, residual_pass1, fg_core_pre, bg_anchor, params):
    residual_threshold = torch.quantile(
        residual_pass1.float(),
        float(params.get("BG_RESIDUAL_PERCENTILE", 30.0)) / 100.0,
    )
    bg_core = (
        ((bc > float(params.get("BG_BC_MIN", 0.65))) | (residual_pass1 < residual_threshold))
        & ~fg_core_pre
    )
    min_count = max(1, int(round(float(params.get("BG_CORE_MIN_RATIO", 0.05)) * int(bc.numel()))))
    if int(bg_core.sum().item()) < min_count:
        bg_core = bg_anchor & ~fg_core_pre
    if int(bg_core.sum().item()) == 0:
        bg_core = bg_anchor
    return bg_core.bool()


def _select_bg_core_v2(bc, fg_score, fg_core, bg_core_pass2, bg_anchor, border, params):
    bg_core = (
        (bc > float(params.get("BG_CORE_BC_MIN", 0.70)))
        & (fg_score < float(params.get("BG_SCORE_MAX", 0.30)))
        & ~fg_core
    )
    min_count = max(1, int(round(float(params.get("BG_CORE_MIN_RATIO", 0.05)) * int(bc.numel()))))
    if int(bg_core.sum().item()) < min_count:
        bg_core = bg_core_pass2 & ~fg_core
    if int(bg_core.sum().item()) < min_count:
        bg_core = bg_anchor & ~fg_core
    if int(bg_core.sum().item()) == 0:
        bg_core = border & ~fg_core
    return bg_core.bool()


def _evidence_gate(fg_score, params):
    threshold = torch.quantile(
        fg_score.float(),
        float(params.get("EVIDENCE_PERCENTILE", 70.0)) / 100.0,
    )
    return torch.sigmoid((fg_score - threshold) / float(params.get("EVIDENCE_TAU", 0.08))).clamp(0.0, 1.0)


def _label_propagation(neigh_idx, neigh_weight, fg_core, bg_core, fg_score, params):
    if int(fg_core.sum().item()) == 0:
        return torch.zeros_like(fg_score)
    p = fg_score.float().clone().clamp(0.0, 1.0)
    p[fg_core] = 1.0
    p[bg_core] = 0.0
    seed = fg_core | bg_core
    denom = neigh_weight.sum(dim=1).clamp_min(EPS)

    max_iter = int(params.get("RW_ITER", params.get("PROP_ITER", 200)))
    tol = float(params.get("RW_TOL", params.get("PROP_TOL", 1e-5)))
    for _ in range(max_iter):
        gathered = p[neigh_idx]
        updated = (gathered * neigh_weight).sum(dim=1) / denom
        new_p = p.clone()
        new_p[~seed] = updated[~seed]
        new_p[fg_core] = 1.0
        new_p[bg_core] = 0.0
        delta = float((new_p - p).abs().max().item())
        p = new_p
        if delta < tol:
            break
    return p.clamp(0.0, 1.0)


def _view_result_to_maps(result, grid):
    out = {}
    for key in (
        "bc_map_37",
        "bg_anchor_37",
        "bg_core_37",
        "fg_score_37",
        "fg_core_37",
        "residual_37",
        "p_dabe_37",
    ):
        out[key] = result[key].reshape(1, grid, grid).float()
    return out


def _run_single_view_v1(feature, rgb, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge = _sobel_magnitude(rgb)
    edge_n = edge.reshape(-1).float()
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, grid, params)
    bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, params)
    bg_anchor = _background_anchor(bc, border, params)
    residual = _background_residual(feat_n, rgb_n, bg_anchor, params)
    fg_score = (residual * (1.0 - bc).clamp(0.0, 1.0).pow(float(params["GAMMA_BC_SUPPRESS"]))).clamp(0.0, 1.0)
    fg_core, fallback_flag, fallback_reason = _select_fg_core(fg_score, bc, grid, params)
    bg_core = _select_bg_core(bc, fg_score, fg_core, bg_anchor, border, params)
    p_dabe = _label_propagation(neigh_idx, neigh_weight, fg_core, bg_core, fg_score, params)

    maps = {
        "bc_map_37": bc.reshape(1, grid, grid),
        "bg_anchor_37": bg_anchor.float().reshape(1, grid, grid),
        "bg_core_37": bg_core.float().reshape(1, grid, grid),
        "fg_score_37": fg_score.reshape(1, grid, grid),
        "fg_core_37": fg_core.float().reshape(1, grid, grid),
        "residual_37": residual.reshape(1, grid, grid),
        "residual_pass1_37": residual.reshape(1, grid, grid),
        "p_rw_37": p_dabe.reshape(1, grid, grid),
        "evidence_37": torch.ones((1, grid, grid), dtype=torch.float32),
        "p_dabe_37": p_dabe.reshape(1, grid, grid),
        "num_components": _num_components(fg_core.bool(), grid) if int(fg_core.sum().item()) > 0 else 0,
        "fallback_flag": bool(fallback_flag),
        "fallback_reason": fallback_reason,
    }
    return maps


def _run_single_view_v2(feature, rgb, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge = _sobel_magnitude(rgb)
    edge_n = edge.reshape(-1).float()
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, grid, params)
    bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, params)
    bg_anchor = _background_anchor(bc, border, params)

    residual_pass1 = _background_residual(feat_n, rgb_n, bg_anchor, params)
    fg_score_1 = _weak_bc_fg_score(residual_pass1, bc, params)
    fg_core_pre, fallback_pre, fallback_reason_pre, _ = _select_fg_core_v2(
        feat_n,
        fg_score_1,
        bc,
        grid,
        params,
    )

    if bool(params.get("USE_TWOPASS_BG", True)):
        bg_core_pass2 = _build_bg_core_pass2(bc, residual_pass1, fg_core_pre, bg_anchor, params)
        residual = _background_residual(feat_n, rgb_n, bg_core_pass2, params)
    else:
        bg_core_pass2 = bg_anchor & ~fg_core_pre
        residual = residual_pass1

    fg_score = _weak_bc_fg_score(residual, bc, params)
    fg_core, fallback_flag, fallback_reason, component_count = _select_fg_core_v2(
        feat_n,
        fg_score,
        bc,
        grid,
        params,
    )
    if fallback_pre and not fallback_flag:
        fallback_reason = fallback_reason_pre
    fallback_flag = bool(fallback_pre or fallback_flag)
    bg_core = _select_bg_core_v2(bc, fg_score, fg_core, bg_core_pass2, bg_anchor, border, params)

    p_rw = _label_propagation(neigh_idx, neigh_weight, fg_core, bg_core, fg_score, params)
    if bool(params.get("USE_EVIDENCE_GATE", True)):
        evidence = _evidence_gate(fg_score, params)
        mode = str(params.get("EVIDENCE_MODE", "multiply")).lower()
        if mode == "multiply":
            p_dabe = p_rw * evidence
        elif mode == "min":
            p_dabe = torch.minimum(p_rw, evidence)
        else:
            raise ValueError(f"Unsupported DABE v2 EVIDENCE_MODE: {mode}")
    else:
        evidence = torch.ones_like(p_rw)
        p_dabe = p_rw

    return {
        "bc_map_37": bc.reshape(1, grid, grid),
        "bg_anchor_37": bg_anchor.float().reshape(1, grid, grid),
        "bg_core_37": bg_core.float().reshape(1, grid, grid),
        "fg_score_37": fg_score.reshape(1, grid, grid),
        "fg_core_37": fg_core.float().reshape(1, grid, grid),
        "residual_37": residual.reshape(1, grid, grid),
        "residual_pass1_37": residual_pass1.reshape(1, grid, grid),
        "p_rw_37": p_rw.reshape(1, grid, grid),
        "evidence_37": evidence.reshape(1, grid, grid),
        "p_dabe_37": p_dabe.reshape(1, grid, grid).clamp(0.0, 1.0),
        "num_components": int(component_count),
        "fallback_flag": bool(fallback_flag),
        "fallback_reason": fallback_reason if fallback_flag else "none",
    }


def _compute_core_affinity(feature, fg_core, fg_score, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    fg_flat = (fg_core.reshape(-1).float() > 0.5)
    fg_idx = torch.where(fg_flat)[0]
    if fg_idx.numel() < 4:
        return torch.zeros((1, grid, grid), dtype=torch.float32), True

    max_core = min(128, int(fg_idx.numel()))
    if fg_idx.numel() > max_core:
        fg_scores = fg_score.reshape(-1).float().index_select(0, fg_idx)
        top_idx = torch.topk(fg_scores, k=max_core, largest=True).indices
        fg_idx = fg_idx.index_select(0, top_idx)
    core_feat = feat_n.index_select(0, fg_idx)
    sim = feat_n @ core_feat.t()
    if bool(params.get("DABE_V3_CORE_AFF_USE_MAX", True)):
        core_sim = sim.max(dim=1).values
    else:
        topk = min(int(params.get("DABE_V3_CORE_AFF_TOPK", 32)), int(sim.shape[1]))
        core_sim = torch.topk(sim, k=max(1, topk), dim=1, largest=True).values.mean(dim=1)
    core_affinity = ((core_sim + 1.0) * 0.5).clamp(0.0, 1.0)
    return core_affinity.reshape(1, grid, grid).float(), False


def _component_bbox(component):
    ys, xs = np.where(component)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _component_protection(p_v3, fg_core, bg_core, fg_score, residual, evidence, core_affinity, edge, params):
    grid = int(params["GRID"])
    tmp_mask = (p_v3.reshape(grid, grid) > 0.5).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(tmp_mask, structure=np.ones((3, 3), dtype=np.uint8))
    p_map = p_v3.reshape(grid, grid).float().clone()
    fg_map = (fg_core.reshape(grid, grid).float() > 0.5)
    bg_map = (bg_core.reshape(grid, grid).float() > 0.5)
    fg_score_map = fg_score.reshape(grid, grid).float()
    residual_map = residual.reshape(grid, grid).float()
    evidence_map = evidence.reshape(grid, grid).float()
    affinity_map = core_affinity.reshape(grid, grid).float()
    edge_map = edge.reshape(grid, grid).float()

    keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    components = []
    residual_high = float(torch.quantile(residual_map.flatten(), 0.60).item())

    for label_id in range(1, int(num_labels) + 1):
        component_np = labels == label_id
        area = int(component_np.sum())
        if area <= 0:
            continue
        component = torch.from_numpy(component_np).bool()
        y0, y1, x0, x1 = _component_bbox(component_np)
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        border_touch = _component_border_touch_ratio(component_np)
        mean_p = float(p_map[component].mean().item())
        mean_fg = float(fg_score_map[component].mean().item())
        mean_residual = float(residual_map[component].mean().item())
        mean_evidence = float(evidence_map[component].mean().item())
        mean_affinity = float(affinity_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        contains_fg_core = bool((fg_map & component).any().item())
        score = (
            mean_p
            * mean_fg
            * mean_residual
            * mean_evidence
            * mean_affinity
            * max(0.0, 1.0 - border_touch)
        )
        components.append(
            {
                "label_id": int(label_id),
                "component": component,
                "area": area,
                "score": float(score),
                "mean_residual": mean_residual,
                "mean_affinity": mean_affinity,
                "mean_edge": mean_edge,
                "elongation": elongation,
                "contains_fg_core": contains_fg_core,
            }
        )

    topk = int(params.get("DABE_V3_COMPONENT_TOPK", 5))
    min_area = int(params.get("DABE_V3_COMPONENT_MIN_AREA", 4))
    sem_keep = float(params.get("DABE_V3_COMPONENT_SEM_KEEP", 0.50))
    edge_keep = float(params.get("DABE_V3_COMPONENT_EDGE_KEEP", 0.35))
    elongation_keep = float(params.get("DABE_V3_COMPONENT_ELONGATION_KEEP", 2.5))
    weak_scale = float(params.get("DABE_V3_COMPONENT_WEAK_KEEP_SCALE", 0.35))
    top_labels = {
        comp["label_id"]
        for comp in sorted(components, key=lambda item: item["score"], reverse=True)[: max(0, topk)]
    }

    num_kept = 0
    num_weak = 0
    num_removed = 0
    for comp in components:
        component = comp["component"]
        keep = (
            comp["label_id"] in top_labels
            or comp["mean_affinity"] >= sem_keep
            or (comp["elongation"] >= elongation_keep and comp["mean_edge"] >= edge_keep)
            or (comp["mean_affinity"] >= sem_keep and comp["mean_residual"] >= residual_high)
            or comp["contains_fg_core"]
        )
        if keep:
            keep_mask |= component
            num_kept += 1
        elif comp["area"] < min_area:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
        else:
            p_map[component] = p_map[component] * weak_scale
            weak_mask |= component
            num_weak += 1

    p_map = torch.where(bg_map, torch.minimum(p_map, torch.full_like(p_map, 0.5 - 1e-6)), p_map)
    del removed_mask
    return {
        "p_refined_37": p_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "component_keep_mask_37": keep_mask.float().reshape(1, grid, grid),
        "component_weak_mask_37": weak_mask.float().reshape(1, grid, grid),
        "num_components": int(num_labels),
        "num_kept_components": int(num_kept),
        "num_weak_components": int(num_weak),
        "num_removed_components": int(num_removed),
    }


def _compute_core_affinity_max(feature, fg_core, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    fg_flat = (fg_core.reshape(-1).float() > 0.5)
    fg_idx = torch.where(fg_flat)[0]
    if fg_idx.numel() < 4:
        return torch.zeros((1, grid, grid), dtype=torch.float32), True
    core_feat = feat_n.index_select(0, fg_idx)
    core_sim = (feat_n @ core_feat.t()).max(dim=1).values
    core_affinity = ((core_sim + 1.0) * 0.5).clamp(0.0, 1.0)
    return core_affinity.reshape(1, grid, grid).float(), False


def _dilate_mask(mask, radius):
    radius = max(0, int(radius))
    mask_np = mask.detach().cpu().bool().squeeze().numpy()
    if radius == 0:
        return torch.from_numpy(mask_np).bool().reshape_as(mask.bool())
    dilated = ndimage.binary_dilation(
        mask_np,
        structure=np.ones((3, 3), dtype=np.uint8),
        iterations=radius,
    )
    return torch.from_numpy(dilated).bool().reshape_as(mask.bool())


def _v31_adaptive_expand(base_area, params):
    if not bool(params.get("DABE_V31_USE_ADAPTIVE_EXPAND", True)):
        return (
            int(params.get("DABE_V31_CANDIDATE_DILATE_RADIUS", 3)),
            float(params.get("DABE_V31_EXPAND_WEIGHT", 0.20)),
        )
    if base_area < float(params.get("DABE_V31_AREA_TINY", 0.08)):
        return (
            int(params.get("DABE_V31_DILATE_RADIUS_TINY", 3)),
            float(params.get("DABE_V31_EXPAND_WEIGHT_TINY", 0.25)),
        )
    if base_area <= float(params.get("DABE_V31_AREA_NORMAL", 0.16)):
        return (
            int(params.get("DABE_V31_DILATE_RADIUS_NORMAL", 3)),
            float(params.get("DABE_V31_EXPAND_WEIGHT_NORMAL", 0.20)),
        )
    if base_area <= float(params.get("DABE_V31_AREA_LARGE", 0.20)):
        return (
            int(params.get("DABE_V31_DILATE_RADIUS_LARGE", 2)),
            float(params.get("DABE_V31_EXPAND_WEIGHT_LARGE", 0.10)),
        )
    return (
        int(params.get("DABE_V31_DILATE_RADIUS_XLARGE", 1)),
        float(params.get("DABE_V31_EXPAND_WEIGHT_XLARGE", 0.05)),
    )


def _component_protection_v31(p_v31, base_mask, fg_core, bg_core, residual_norm, evidence, core_affinity, edge, params):
    grid = int(params["GRID"])
    tmp_mask = (p_v31.reshape(grid, grid) > 0.5).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(tmp_mask, structure=np.ones((3, 3), dtype=np.uint8))
    p_map = p_v31.reshape(grid, grid).float().clone()
    base_map = (base_mask.reshape(grid, grid).float() > 0.5)
    fg_map = (fg_core.reshape(grid, grid).float() > 0.5)
    bg_map = (bg_core.reshape(grid, grid).float() > 0.5)
    residual_map = residual_norm.reshape(grid, grid).float()
    evidence_map = evidence.reshape(grid, grid).float()
    affinity_map = core_affinity.reshape(grid, grid).float()
    edge_map = edge.reshape(grid, grid).float()

    base_np = base_map.detach().cpu().numpy().astype(bool)
    if base_np.any():
        dist_to_base = torch.from_numpy(ndimage.distance_transform_edt(~base_np)).float()
    else:
        dist_to_base = torch.full((grid, grid), float(grid), dtype=torch.float32)

    keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    components = []
    for label_id in range(1, int(num_labels) + 1):
        component_np = labels == label_id
        area = int(component_np.sum())
        if area <= 0:
            continue
        component = torch.from_numpy(component_np).bool()
        y0, y1, x0, x1 = _component_bbox(component_np)
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        border_touch = _component_border_touch_ratio(component_np)
        mean_p = float(p_map[component].mean().item())
        mean_residual = float(residual_map[component].mean().item())
        mean_evidence = float(evidence_map[component].mean().item())
        mean_affinity = float(affinity_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        distance_to_base = float(dist_to_base[component].min().item())
        contains_fg_core = bool((fg_map & component).any().item())
        score = mean_p * mean_affinity * mean_residual * mean_evidence * max(0.0, 1.0 - border_touch)
        components.append(
            {
                "label_id": int(label_id),
                "component": component,
                "area": area,
                "score": float(score),
                "mean_affinity": mean_affinity,
                "mean_edge": mean_edge,
                "elongation": elongation,
                "distance_to_base": distance_to_base,
                "contains_fg_core": contains_fg_core,
            }
        )

    topk = int(params.get("DABE_V31_COMPONENT_TOPK", 5))
    min_area = int(params.get("DABE_V31_COMPONENT_MIN_AREA", 4))
    sem_keep = float(params.get("DABE_V31_COMPONENT_SEM_KEEP", 0.58))
    edge_keep = float(params.get("DABE_V31_COMPONENT_EDGE_KEEP", 0.35))
    elongation_keep = float(params.get("DABE_V31_COMPONENT_ELONGATION_KEEP", 2.5))
    max_dist = float(params.get("DABE_V31_COMPONENT_MAX_DIST_TO_BASE", 3))
    weak_scale = float(params.get("DABE_V31_COMPONENT_WEAK_KEEP_SCALE", 0.0))
    top_labels = {
        comp["label_id"]
        for comp in sorted(components, key=lambda item: item["score"], reverse=True)[: max(0, topk)]
    }

    num_kept = 0
    num_weak = 0
    num_removed = 0
    for comp in components:
        component = comp["component"]
        if comp["area"] < min_area:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
            continue
        near_base = comp["distance_to_base"] <= max_dist
        keep = (
            comp["contains_fg_core"]
            or comp["label_id"] in top_labels
            or comp["mean_affinity"] >= sem_keep
            or (comp["elongation"] >= elongation_keep and comp["mean_edge"] >= edge_keep and near_base)
            or (near_base and comp["mean_affinity"] >= sem_keep)
        )
        if keep:
            keep_mask |= component
            num_kept += 1
        elif weak_scale <= 0.0:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
        else:
            p_map[component] = p_map[component] * weak_scale
            weak_mask |= component
            num_weak += 1

    p_map = torch.where(bg_map, torch.minimum(p_map, torch.full_like(p_map, 0.5 - 1e-6)), p_map)
    return {
        "p_refined_37": p_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "component_keep_mask_37": keep_mask.float().reshape(1, grid, grid),
        "component_weak_mask_37": weak_mask.float().reshape(1, grid, grid),
        "component_removed_mask_37": removed_mask.float().reshape(1, grid, grid),
        "num_components": int(num_labels),
        "num_kept_components": int(num_kept),
        "num_weak_components": int(num_weak),
        "num_removed_components": int(num_removed),
    }


def _gc_construct_seeds(p_base, fg_core, bg_core, fg_score, residual, evidence, bc_map, params):
    grid = int(params["GRID"])
    fallback_reasons = []
    fg_seed = torch.zeros_like(p_base, dtype=torch.bool)
    bg_seed = torch.zeros_like(p_base, dtype=torch.bool)

    if bool(params.get("DABE_GC_FG_SEED_FROM_FG_CORE", True)):
        fg_seed |= fg_core > 0.5
    score_thr = torch.quantile(
        fg_score.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_GC_FG_SEED_SCORE_PERCENTILE", 92.0)) / 100.0)),
    )
    res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_GC_FG_SEED_RESIDUAL_PERCENTILE", 70.0)) / 100.0)),
    )
    fg_seed |= (
        (fg_score >= score_thr)
        & (residual >= res_thr)
        & (evidence >= float(params.get("DABE_GC_FG_SEED_EVIDENCE_MIN", 0.45)))
        & (bg_core <= 0.5)
    )
    fg_min = max(1, int(params.get("DABE_GC_FG_SEED_MIN_PIXELS", 4)))
    if int(fg_seed.sum().item()) < fg_min:
        fallback_reasons.append("too_few_fg_seed")
        topk = min(fg_min, int(p_base.numel()))
        top_idx = torch.topk(p_base.reshape(-1).float(), k=topk, largest=True).indices
        fg_seed_flat = fg_seed.reshape(-1)
        fg_seed_flat[top_idx] = True
        fg_seed = fg_seed_flat.reshape_as(fg_seed)

    if bool(params.get("DABE_GC_BG_SEED_FROM_BG_CORE", True)):
        bg_seed |= bg_core > 0.5
    low_res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_GC_BG_SEED_LOW_RESIDUAL_PERCENTILE", 25.0)) / 100.0)),
    )
    bg_seed |= (
        (residual <= low_res_thr)
        & (bc_map >= float(params.get("DABE_GC_BG_SEED_BC_MIN", 0.60)))
    )
    border = _border_mask(grid, int(params.get("DABE_GC_BG_SEED_BORDER_WIDTH", 1))).reshape(1, grid, grid)
    bg_seed |= border & (p_base < 0.2)
    bg_seed &= ~fg_seed

    bg_min = max(1, int(params.get("DABE_GC_BG_SEED_MIN_PIXELS", 16)))
    if int(bg_seed.sum().item()) < bg_min:
        fallback_reasons.append("too_few_bg_seed")
        residual_norm = _minmax(residual.reshape(-1)).reshape_as(residual)
        bg_score = ((1.0 - residual_norm) * bc_map).reshape(-1).float()
        bg_score[fg_seed.reshape(-1)] = -1.0
        topk = min(bg_min, int(bg_score.numel()))
        top_idx = torch.topk(bg_score, k=topk, largest=True).indices
        bg_seed_flat = bg_seed.reshape(-1)
        bg_seed_flat[top_idx] = True
        bg_seed = bg_seed_flat.reshape_as(bg_seed) & ~fg_seed

    return fg_seed, bg_seed, fallback_reasons


def _gc_edge_weight(src, dst, dx, dy, feat_n, rgb_n, edge_n, fg_flat, bg_flat, params, multiplier=1.0):
    weight = float(multiplier)
    spatial_sigma = max(float(params.get("DABE_GC_SPATIAL_SIGMA", 1.5)), EPS)
    spatial_dist2 = float(dx * dx + dy * dy)
    weight *= float(np.exp(-spatial_dist2 / (2.0 * spatial_sigma * spatial_sigma)))
    if bool(params.get("DABE_GC_USE_SEM_AFF", True)):
        sem = ((float(torch.dot(feat_n[src], feat_n[dst]).item()) + 1.0) * 0.5)
        sem = max(0.0, min(1.0, sem))
        weight *= sem ** float(params.get("DABE_GC_SEM_POWER", 1.0))
    if bool(params.get("DABE_GC_USE_COLOR_AFF", True)):
        sigma = max(float(params.get("DABE_GC_COLOR_SIGMA", 0.12)), EPS)
        color_dist2 = float(torch.sum((rgb_n[src] - rgb_n[dst]).square()).item())
        weight *= float(np.exp(-color_dist2 / (2.0 * sigma * sigma)))
    if bool(params.get("DABE_GC_USE_EDGE_STOP", True)):
        sigma = max(float(params.get("DABE_GC_EDGE_SIGMA", 0.25)), EPS)
        edge_between = float(torch.maximum(edge_n[src], edge_n[dst]).item())
        weight *= float(np.exp(-edge_between / sigma))
    if (bool(fg_flat[src]) and bool(bg_flat[dst])) or (bool(bg_flat[src]) and bool(fg_flat[dst])):
        weight *= 0.1
    return max(float(weight), EPS)


def _gc_build_graph(feature, rgb, edge, fg_seed, bg_seed, bg_core, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = edge.reshape(-1).float()
    fg_flat = (fg_seed.reshape(-1).bool()).tolist()
    bg_flat = (bg_seed.reshape(-1).bool()).tolist()
    bg_core_flat = (bg_core.reshape(-1).float() > 0.5)

    radius = max(1, int(params.get("DABE_GC_GRAPH_RADIUS", 1)))
    if not bool(params.get("DABE_GC_USE_RADIUS2", False)):
        radius = min(radius, 1)
    rows_idx = []
    rows_weight = []
    for y in range(grid):
        for x in range(grid):
            src = y * grid + x
            idxs = []
            weights = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                        continue
                    dst = ny * grid + nx
                    idxs.append(dst)
                    weights.append(_gc_edge_weight(src, dst, dx, dy, feat_n, rgb_n, edge_n, fg_flat, bg_flat, params))

            if bool(params.get("DABE_GC_USE_LOCAL_TOPK", False)):
                local_radius = max(radius + 1, int(params.get("DABE_GC_LOCAL_TOPK_RADIUS", 3)))
                topk_candidates = []
                for dy in range(-local_radius, local_radius + 1):
                    for dx in range(-local_radius, local_radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) <= radius and abs(dy) <= radius:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                            continue
                        dst = ny * grid + nx
                        if bool(bg_core_flat[src]) or bool(bg_core_flat[dst]):
                            continue
                        sem = float(torch.dot(feat_n[src], feat_n[dst]).item())
                        topk_candidates.append((sem, dst, dx, dy))
                topk_candidates.sort(key=lambda item: item[0], reverse=True)
                for _sem, dst, dx, dy in topk_candidates[: max(0, int(params.get("DABE_GC_LOCAL_TOPK", 4)))]:
                    idxs.append(dst)
                    weights.append(
                        _gc_edge_weight(
                            src,
                            dst,
                            dx,
                            dy,
                            feat_n,
                            rgb_n,
                            edge_n,
                            fg_flat,
                            bg_flat,
                            params,
                            multiplier=float(params.get("DABE_GC_LOCAL_TOPK_WEIGHT", 0.25)),
                        )
                    )

            if not idxs:
                idxs = [src]
                weights = [1.0]
            rows_idx.append(torch.tensor(idxs, dtype=torch.long))
            rows_weight.append(torch.tensor(weights, dtype=torch.float32))

    max_slots = max(int(row.numel()) for row in rows_idx)
    neigh_idx = torch.zeros((grid * grid, max_slots), dtype=torch.long)
    neigh_weight = torch.zeros((grid * grid, max_slots), dtype=torch.float32)
    for row_id, (idxs, weights) in enumerate(zip(rows_idx, rows_weight)):
        slots = int(idxs.numel())
        neigh_idx[row_id, :slots] = idxs
        neigh_weight[row_id, :slots] = weights
    degree = neigh_weight.sum(dim=1).clamp_min(EPS)
    neigh_weight_norm = neigh_weight / degree.unsqueeze(1)
    graph_degree = _minmax(degree).reshape(1, grid, grid)
    return neigh_idx, neigh_weight_norm, graph_degree


def _gc_diffuse(p_base, fg_seed, bg_seed, neigh_idx, neigh_weight_norm, params):
    y = p_base.reshape(-1).float().clone()
    p_base_flat = p_base.reshape(-1).float()
    fg_flat = fg_seed.reshape(-1).bool()
    bg_flat = bg_seed.reshape(-1).bool()
    y[fg_flat] = 1.0
    y[bg_flat] = 0.0
    anchor = p_base_flat.clone()
    anchor[fg_flat] = 1.0
    anchor[bg_flat] = 0.0
    mu = float(params.get("DABE_GC_DIFFUSE_MU", 0.85))
    prior_weight = float(params.get("DABE_GC_PRIOR_WEIGHT", 0.10))
    for _ in range(int(params.get("DABE_GC_DIFFUSE_ITERS", 25))):
        y_prop = (y[neigh_idx] * neigh_weight_norm).sum(dim=1)
        if bool(params.get("DABE_GC_USE_BASE_AS_PRIOR", True)):
            prior = prior_weight * p_base_flat + (1.0 - prior_weight) * anchor
            y_new = mu * y_prop + (1.0 - mu) * prior
        else:
            y_new = mu * y_prop + (1.0 - mu) * anchor
        if bool(params.get("DABE_GC_SEED_LOCK", True)):
            y_new[fg_flat] = 1.0
            y_new[bg_flat] = 0.0
        y = y_new.clamp(0.0, 1.0)
    grid = int(params["GRID"])
    return y.reshape(1, grid, grid)


def _gc_area_control(p_final, fg_seed, evidence, bg_core, params):
    area_before = float(p_final.mean().item())
    low = float(params.get("DABE_GC_AREA_LOW", 0.08))
    high = float(params.get("DABE_GC_AREA_HIGH", 0.22))
    small_area_flag = bool(area_before < low)
    large_area_flag = bool(area_before > high)
    factor = 1.0
    if large_area_flag and bool(params.get("DABE_GC_AREA_SUPPRESS_HIGH", True)):
        over = max(0.0, area_before - high)
        factor = 1.0 - over / max(high, EPS)
        factor = max(float(params.get("DABE_GC_AREA_SUPPRESS_MIN_FACTOR", 0.75)), min(1.0, factor))
        suppress_mask = (fg_seed <= 0.5) & (evidence < 0.75) & (bg_core <= 0.5)
        p_final = torch.where(suppress_mask, p_final * factor, p_final)
    area_after = float(p_final.mean().item())
    return p_final.clamp(0.0, 1.0), small_area_flag, large_area_flag, area_before, area_after, float(factor)


def _component_protection_gc(p_final, feature, fg_seed, bg_seed, evidence, fg_score, edge, params):
    grid = int(params["GRID"])
    tmp_mask = (p_final.reshape(grid, grid) > 0.5).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(tmp_mask, structure=np.ones((3, 3), dtype=np.uint8))
    p_map = p_final.reshape(grid, grid).float().clone()
    fg_map = fg_seed.reshape(grid, grid).bool()
    bg_map = bg_seed.reshape(grid, grid).bool()
    evidence_map = evidence.reshape(grid, grid).float()
    fg_score_map = fg_score.reshape(grid, grid).float()
    edge_map = edge.reshape(grid, grid).float()
    sem_to_fg, too_few = _compute_core_affinity_max(feature, fg_seed.float(), params)
    sem_map = sem_to_fg.reshape(grid, grid).float()
    fg_np = fg_map.detach().cpu().numpy().astype(bool)
    if fg_np.any():
        dist_to_fg_map = torch.from_numpy(ndimage.distance_transform_edt(~fg_np)).float()
    else:
        dist_to_fg_map = torch.full((grid, grid), float(grid), dtype=torch.float32)

    keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    components = []
    for label_id in range(1, int(num_labels) + 1):
        component_np = labels == label_id
        area = int(component_np.sum())
        if area <= 0:
            continue
        component = torch.from_numpy(component_np).bool()
        y0, y1, x0, x1 = _component_bbox(component_np)
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        border_touch = _component_border_touch_ratio(component_np)
        mean_p = float(p_map[component].mean().item())
        mean_evidence = float(evidence_map[component].mean().item())
        mean_fg_score = float(fg_score_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        mean_sem = float(sem_map[component].mean().item()) if not too_few else 0.0
        distance_to_fg = float(dist_to_fg_map[component].min().item())
        contains_fg_seed = bool((fg_map & component).any().item())
        contains_bg_seed = bool((bg_map & component).any().item())
        score = mean_p * mean_evidence * mean_fg_score * mean_sem * max(0.0, 1.0 - border_touch)
        components.append(
            {
                "label_id": int(label_id),
                "component": component,
                "area": area,
                "score": float(score),
                "mean_evidence": mean_evidence,
                "mean_sem": mean_sem,
                "mean_edge": mean_edge,
                "elongation": elongation,
                "distance_to_fg": distance_to_fg,
                "contains_fg_seed": contains_fg_seed,
                "contains_bg_seed": contains_bg_seed,
            }
        )

    topk = int(params.get("DABE_GC_COMPONENT_TOPK", 5))
    min_area = int(params.get("DABE_GC_COMPONENT_MIN_AREA", 4))
    sem_keep = float(params.get("DABE_GC_COMPONENT_SEM_KEEP", 0.55))
    edge_keep = float(params.get("DABE_GC_COMPONENT_EDGE_KEEP", 0.35))
    elongation_keep = float(params.get("DABE_GC_COMPONENT_ELONGATION_KEEP", 2.5))
    max_dist = float(params.get("DABE_GC_COMPONENT_MAX_DIST_TO_FG", 3))
    weak_scale = float(params.get("DABE_GC_COMPONENT_WEAK_KEEP_SCALE", 0.0))
    top_labels = {
        comp["label_id"]
        for comp in sorted(components, key=lambda item: item["score"], reverse=True)[: max(0, topk)]
    }

    num_kept = 0
    num_weak = 0
    num_removed = 0
    for comp in components:
        component = comp["component"]
        if comp["area"] < min_area or (comp["contains_bg_seed"] and not comp["contains_fg_seed"]):
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
            continue
        near_fg = comp["distance_to_fg"] <= max_dist
        keep = (
            (bool(params.get("DABE_GC_COMPONENT_KEEP_SEED", True)) and comp["contains_fg_seed"])
            or comp["label_id"] in top_labels
            or comp["mean_sem"] >= sem_keep
            or (comp["elongation"] >= elongation_keep and comp["mean_edge"] >= edge_keep and near_fg)
            or (near_fg and comp["mean_evidence"] >= 0.4)
        )
        if keep:
            keep_mask |= component
            num_kept += 1
        elif weak_scale <= 0.0:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
        else:
            p_map[component] = p_map[component] * weak_scale
            weak_mask |= component
            num_weak += 1

    return {
        "p_refined_37": p_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "component_keep_mask_37": keep_mask.float().reshape(1, grid, grid),
        "component_weak_mask_37": weak_mask.float().reshape(1, grid, grid),
        "component_removed_mask_37": removed_mask.float().reshape(1, grid, grid),
        "num_components": int(num_labels),
        "num_kept_components": int(num_kept),
        "num_weak_components": int(num_weak),
        "num_removed_components": int(num_removed),
    }


def _rac_construct_seeds(p_base, fg_core, bg_core, fg_score, residual, evidence, bc_map, params):
    grid = int(params["GRID"])
    fallback_reasons = []
    fg_seed = torch.zeros_like(p_base, dtype=torch.bool)
    bg_seed = torch.zeros_like(p_base, dtype=torch.bool)

    if bool(params.get("DABE_RAC_FG_SEED_FROM_FG_CORE", True)):
        fg_seed |= fg_core > 0.5
    score_thr = torch.quantile(
        fg_score.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_FG_SEED_SCORE_PERCENTILE", 92.0)) / 100.0)),
    )
    res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_FG_SEED_RESIDUAL_PERCENTILE", 70.0)) / 100.0)),
    )
    fg_seed |= (
        (fg_score >= score_thr)
        & (residual >= res_thr)
        & (evidence >= float(params.get("DABE_RAC_FG_SEED_EVIDENCE_MIN", 0.45)))
        & (bg_core <= 0.5)
    )
    fg_min = max(1, int(params.get("DABE_RAC_FG_SEED_MIN_PIXELS", 4)))
    if int(fg_seed.sum().item()) < fg_min:
        fallback_reasons.append("too_few_fg_seed")
        score = p_base.reshape(-1).float().clone()
        score[(bg_core.reshape(-1) > 0.5)] = -1.0
        topk = min(fg_min, int(score.numel()))
        top_idx = torch.topk(score, k=topk, largest=True).indices
        fg_seed_flat = fg_seed.reshape(-1)
        fg_seed_flat[top_idx] = True
        fg_seed = fg_seed_flat.reshape_as(fg_seed) & (bg_core <= 0.5)

    if bool(params.get("DABE_RAC_BG_SEED_FROM_BG_CORE", True)):
        bg_seed |= bg_core > 0.5
    low_res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_BG_SEED_LOW_RESIDUAL_PERCENTILE", 25.0)) / 100.0)),
    )
    bg_seed |= (
        (bc_map >= float(params.get("DABE_RAC_BG_SEED_BC_MIN", 0.60)))
        & (residual <= low_res_thr)
    )
    border = _border_mask(grid, int(params.get("DABE_RAC_BG_SEED_BORDER_WIDTH", 1))).reshape(1, grid, grid)
    bg_seed |= border & (p_base < 0.2)
    bg_seed &= ~fg_seed

    bg_min = max(1, int(params.get("DABE_RAC_BG_SEED_MIN_PIXELS", 16)))
    if int(bg_seed.sum().item()) < bg_min:
        fallback_reasons.append("too_few_bg_seed")
        residual_norm = _minmax(residual.reshape(-1)).reshape_as(residual)
        bg_score = ((1.0 - residual_norm) * bc_map).reshape(-1).float()
        bg_score[fg_seed.reshape(-1)] = -1.0
        topk = min(bg_min, int(bg_score.numel()))
        top_idx = torch.topk(bg_score, k=topk, largest=True).indices
        bg_seed_flat = bg_seed.reshape(-1)
        bg_seed_flat[top_idx] = True
        bg_seed = bg_seed_flat.reshape_as(bg_seed) & ~fg_seed

    return fg_seed.bool(), bg_seed.bool(), fallback_reasons


def _rac_pair_affinity(src, dst, feat_n, rgb_n, edge_n, bg_seed_flat, bg_core_flat, params):
    sem = ((float(torch.dot(feat_n[src], feat_n[dst]).item()) + 1.0) * 0.5)
    sem = max(0.0, min(1.0, sem)) ** float(params.get("DABE_RAC_SEM_POWER", 1.0))
    color_sigma = max(float(params.get("DABE_RAC_COLOR_SIGMA", 0.12)), EPS)
    color_dist2 = float(torch.sum((rgb_n[src] - rgb_n[dst]).square()).item())
    color = float(np.exp(-color_dist2 / (2.0 * color_sigma * color_sigma)))
    color = max(0.0, min(1.0, color)) ** float(params.get("DABE_RAC_COLOR_POWER", 1.0))
    edge_sigma = max(float(params.get("DABE_RAC_EDGE_SIGMA", 0.25)), EPS)
    edge_between = float(torch.maximum(edge_n[src], edge_n[dst]).item())
    edge_stop = float(np.exp(-edge_between / edge_sigma))
    edge_stop = max(0.0, min(1.0, edge_stop)) ** float(params.get("DABE_RAC_EDGE_POWER", 1.0))
    weight = sem * color * edge_stop
    if bool(params.get("DABE_RAC_REGION_BG_BLOCK", True)) and (bool(bg_seed_flat[dst]) or bool(bg_core_flat[dst])):
        weight *= 0.1
    return float(max(weight, EPS)), float(sem), float(color), float(edge_stop), float(edge_between)


def _rac_region_aff_degree(feature, rgb, edge, bg_seed, bg_core, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = edge.reshape(-1).float()
    bg_seed_flat = bg_seed.reshape(-1).bool()
    bg_core_flat = bg_core.reshape(-1).float() > 0.5
    radius = max(1, int(params.get("DABE_RAC_REGION_RADIUS", 2)))
    degree = torch.zeros(grid * grid, dtype=torch.float32)
    for y in range(grid):
        for x in range(grid):
            src = y * grid + x
            weight_sum = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                        continue
                    dst = ny * grid + nx
                    weight, _sem, _color, _edge_stop, _edge_between = _rac_pair_affinity(
                        src,
                        dst,
                        feat_n,
                        rgb_n,
                        edge_n,
                        bg_seed_flat,
                        bg_core_flat,
                        params,
                    )
                    weight_sum += weight
            degree[src] = float(weight_sum)
    return _minmax(degree).reshape(1, grid, grid), feat_n, rgb_n, edge_n


def _rac_region_candidate_indices(frontier, region, grid, radius):
    frontier_np = frontier.detach().cpu().numpy().astype(bool)
    region_np = region.detach().cpu().numpy().astype(bool)
    candidates = set()
    ys, xs = np.where(frontier_np)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                    continue
                if region_np[ny, nx]:
                    continue
                candidates.add(ny * grid + nx)
    return sorted(candidates)


def _rac_region_support(candidate_idx, region_flat, feat_n, rgb_n, edge_n, params):
    region_idx = torch.where(region_flat)[0]
    if region_idx.numel() == 0:
        return 0.0, 0.0, 1.0
    cand_feat = feat_n[candidate_idx].unsqueeze(0)
    sem = ((cand_feat @ feat_n.index_select(0, region_idx).t()).max().item() + 1.0) * 0.5
    sem = max(0.0, min(1.0, float(sem))) ** float(params.get("DABE_RAC_SEM_POWER", 1.0))
    color_sigma = max(float(params.get("DABE_RAC_COLOR_SIGMA", 0.12)), EPS)
    color_dist2 = torch.sum((rgb_n.index_select(0, region_idx) - rgb_n[candidate_idx]).square(), dim=1)
    color = torch.exp(-color_dist2 / (2.0 * color_sigma * color_sigma)).max().item()
    color = max(0.0, min(1.0, float(color))) ** float(params.get("DABE_RAC_COLOR_POWER", 1.0))
    edge = float(edge_n[candidate_idx].item())
    return float(sem), float(color), float(edge)


def _rac_grow_regions(feature, rgb, edge, fg_seed, bg_seed, bg_core, params):
    grid = int(params["GRID"])
    total = grid * grid
    _degree, feat_n, rgb_n, edge_n = _rac_region_aff_degree(feature, rgb, edge, bg_seed, bg_core, params)
    fg_np = fg_seed.reshape(grid, grid).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(fg_np, structure=np.ones((3, 3), dtype=np.uint8))
    bg_block = ((bg_seed | (bg_core > 0.5)).reshape(grid, grid)).bool()
    radius = max(1, int(params.get("DABE_RAC_REGION_RADIUS", 2)))
    max_steps = max(0, int(params.get("DABE_RAC_MAX_REGION_STEPS", 6)))
    max_area = max(1, int(round(float(params.get("DABE_RAC_REGION_MAX_AREA_RATIO", 0.35)) * total)))
    sem_min = float(params.get("DABE_RAC_REGION_SEM_MIN", 0.50))
    color_min = float(params.get("DABE_RAC_REGION_COLOR_MIN", 0.35))
    edge_max = float(params.get("DABE_RAC_REGION_EDGE_MAX", 0.65))
    regions = []
    for label_id in range(1, int(num_labels) + 1):
        region = torch.from_numpy(labels == label_id).bool()
        if not bool(region.any().item()):
            continue
        region &= ~bg_block
        frontier = region.clone()
        for _step in range(max_steps):
            if int(region.sum().item()) >= max_area:
                break
            new_region = torch.zeros_like(region)
            region_flat = region.reshape(-1).bool()
            for cand in _rac_region_candidate_indices(frontier, region, grid, radius):
                cy, cx = divmod(int(cand), grid)
                if bool(bg_block[cy, cx].item()):
                    continue
                sem, color, edge_value = _rac_region_support(cand, region_flat, feat_n, rgb_n, edge_n, params)
                if sem >= sem_min and color >= color_min and edge_value <= edge_max:
                    new_region[cy, cx] = True
                    if int(region.sum().item() + new_region.sum().item()) >= max_area:
                        break
            if not bool(new_region.any().item()):
                break
            new_region &= ~region
            region |= new_region
            frontier = new_region
        regions.append({"seed_label": int(label_id), "mask": region})
    return regions, _degree, feat_n, rgb_n, edge_n


def _rac_color_to_seed_map(rgb, fg_seed, params):
    grid = int(params["GRID"])
    rgb_map = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    fg_flat = fg_seed.reshape(-1).bool()
    if int(fg_flat.sum().item()) == 0:
        return torch.zeros((1, grid, grid), dtype=torch.float32)
    seed_color = rgb_map[fg_flat].mean(dim=0, keepdim=True)
    color_sigma = max(float(params.get("DABE_RAC_COLOR_SIGMA", 0.12)), EPS)
    dist2 = torch.sum((rgb_map - seed_color).square(), dim=1)
    color_aff = torch.exp(-dist2 / (2.0 * color_sigma * color_sigma)).clamp(0.0, 1.0)
    return color_aff.reshape(1, grid, grid)


def _rac_score_regions(regions, feature, rgb, p_base, fg_seed, bg_seed, fg_score, residual, evidence, edge, params):
    grid = int(params["GRID"])
    total = grid * grid
    sem_to_fg, too_few = _compute_core_affinity_max(feature, fg_seed.float(), params)
    color_to_fg = _rac_color_to_seed_map(rgb, fg_seed, params)
    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    p_base_map = p_base.reshape(grid, grid).float()
    fg_map = fg_seed.reshape(grid, grid).bool()
    bg_map = bg_seed.reshape(grid, grid).bool()
    fg_score_map = fg_score.reshape(grid, grid).float()
    residual_map = residual_norm.reshape(grid, grid).float()
    evidence_map = evidence.reshape(grid, grid).float()
    sem_map = sem_to_fg.reshape(grid, grid).float()
    color_map = color_to_fg.reshape(grid, grid).float()
    edge_map = edge.reshape(grid, grid).float()
    min_area = int(params.get("DABE_RAC_REGION_MIN_AREA", 4))
    max_area = max(1, int(round(float(params.get("DABE_RAC_REGION_MAX_AREA_RATIO", 0.35)) * total)))
    stats = []
    for idx, item in enumerate(regions, 1):
        component = item["mask"].reshape(grid, grid).bool()
        area = int(component.sum().item())
        if area <= 0:
            continue
        y0, y1, x0, x1 = _component_bbox(component.detach().cpu().numpy().astype(bool))
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        bbox_area = max(1, width * height)
        compactness = float(area / bbox_area)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        border_touch = _component_border_touch_ratio(component.detach().cpu().numpy().astype(bool))
        seed_support = float((component & fg_map).float().sum().item() / max(area, 1))
        bg_overlap = float((component & bg_map).float().sum().item() / max(area, 1))
        sem_support = float(sem_map[component].mean().item()) if not too_few else 0.0
        color_support = float(color_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        edge_support = max(0.0, 1.0 - mean_edge)
        raw_conf = (
            float(params.get("DABE_RAC_CONF_SEED_WEIGHT", 1.0)) * seed_support
            + float(params.get("DABE_RAC_CONF_SEM_WEIGHT", 1.0)) * sem_support
            + float(params.get("DABE_RAC_CONF_RESIDUAL_WEIGHT", 0.8)) * float(residual_map[component].mean().item())
            + float(params.get("DABE_RAC_CONF_EVIDENCE_WEIGHT", 0.8)) * float(evidence_map[component].mean().item())
            + float(params.get("DABE_RAC_CONF_COLOR_WEIGHT", 0.5)) * color_support
            + float(params.get("DABE_RAC_CONF_EDGE_WEIGHT", 0.5)) * edge_support
            - float(params.get("DABE_RAC_CONF_BG_PENALTY", 1.5)) * bg_overlap
            - float(params.get("DABE_RAC_CONF_BORDER_PENALTY", 0.8)) * border_touch
        )
        conf = float(torch.sigmoid(torch.tensor(raw_conf, dtype=torch.float32)).item())
        stats.append(
            {
                "region_id": int(idx),
                "seed_label": int(item.get("seed_label", idx)),
                "area": int(area),
                "area_ratio": float(area / total),
                "fg_seed_overlap": seed_support,
                "bg_seed_overlap": bg_overlap,
                "mean_p_base": float(p_base_map[component].mean().item()),
                "mean_fg_score": float(fg_score_map[component].mean().item()),
                "mean_residual": float(residual_map[component].mean().item()),
                "mean_evidence": float(evidence_map[component].mean().item()),
                "mean_sem_to_fg_seed": sem_support,
                "mean_color_consistency": color_support,
                "mean_edge_boundary": mean_edge,
                "border_touch_ratio": float(border_touch),
                "compactness": compactness,
                "elongation": elongation,
                "raw_confidence": float(raw_conf),
                "confidence": conf,
                "valid_area": bool(area >= min_area and area <= max_area),
                "mask": component,
            }
        )
    return stats, sem_to_fg, color_to_fg, residual_norm


def _rac_region_completion(region_stats, p_base, params):
    grid = int(params["GRID"])
    p_base_map = p_base.reshape(grid, grid).float()
    p_region = torch.zeros((grid, grid), dtype=torch.float32)
    p_region_strong = torch.zeros((grid, grid), dtype=torch.float32)
    p_region_weak = torch.zeros((grid, grid), dtype=torch.float32)
    strong_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    rejected_mask = torch.zeros((grid, grid), dtype=torch.bool)
    region_id_map = torch.zeros((grid, grid), dtype=torch.float32)
    keep_thr = float(params.get("DABE_RAC_REGION_KEEP_THRESH", 0.42))
    strong_thr = float(params.get("DABE_RAC_REGION_STRONG_KEEP_THRESH", 0.60))
    weak_scale = float(params.get("DABE_RAC_REGION_WEAK_KEEP_SCALE", 0.35))
    topk = max(0, int(params.get("DABE_RAC_REGION_TOPK", 8)))
    ranked_ids = {
        item["region_id"]
        for item in sorted(region_stats, key=lambda value: value["confidence"], reverse=True)[:topk]
    }
    num_strong = 0
    num_weak = 0
    num_rejected = 0
    num_regions = max(1, len(region_stats))
    for item in region_stats:
        component = item["mask"]
        conf = float(item["confidence"])
        region_id_map[component] = max(region_id_map[component].max().item(), float(item["region_id"]) / float(num_regions))
        is_candidate = bool(item["valid_area"]) and item["region_id"] in ranked_ids
        if is_candidate and conf >= strong_thr:
            strong_mask |= component
            p_region_strong[component] = torch.maximum(p_region_strong[component], torch.full_like(p_region_strong[component], conf))
            p_region[component] = torch.maximum(p_region[component], torch.full_like(p_region[component], conf))
            item["status"] = "strong"
            num_strong += 1
        elif is_candidate and conf >= keep_thr:
            weak_mask |= component
            weak_value = weak_scale * conf
            p_region_weak[component] = torch.maximum(p_region_weak[component], torch.full_like(p_region_weak[component], conf))
            p_region[component] = torch.maximum(p_region[component], torch.full_like(p_region[component], weak_value))
            item["status"] = "weak"
            num_weak += 1
        else:
            rejected_mask |= component
            item["status"] = "rejected"
            num_rejected += 1

    if bool(params.get("DABE_RAC_POSITIVE_ONLY", True)):
        delta_strong = (p_region_strong - p_base_map).clamp_min(0.0)
        delta_weak = (p_region_weak - p_base_map).clamp_min(0.0)
        p_rac = p_base_map + float(params.get("DABE_RAC_COMPLETION_WEIGHT", 0.55)) * delta_strong
        p_rac = torch.maximum(
            p_rac,
            p_base_map + float(params.get("DABE_RAC_WEAK_COMPLETION_WEIGHT", 0.25)) * delta_weak,
        )
        delta_pos = torch.maximum(delta_strong, delta_weak)
    else:
        p_rac = p_base_map + float(params.get("DABE_RAC_COMPLETION_WEIGHT", 0.55)) * (p_region - p_base_map)
        delta_pos = (p_region - p_base_map).clamp_min(0.0)
    public_stats = []
    for item in region_stats:
        public_stats.append({key: value for key, value in item.items() if key != "mask"})
    return {
        "p_rac": p_rac.reshape(1, grid, grid).clamp(0.0, 1.0),
        "p_region_37": p_region.reshape(1, grid, grid).clamp(0.0, 1.0),
        "region_id_map_37": region_id_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "strong_region_mask_37": strong_mask.float().reshape(1, grid, grid),
        "weak_region_mask_37": weak_mask.float().reshape(1, grid, grid),
        "rejected_region_mask_37": rejected_mask.float().reshape(1, grid, grid),
        "delta_pos_37": delta_pos.reshape(1, grid, grid).clamp(0.0, 1.0),
        "region_stats": public_stats,
        "num_candidate_regions": int(len(region_stats)),
        "num_strong_regions": int(num_strong),
        "num_weak_regions": int(num_weak),
        "num_rejected_regions": int(num_rejected),
        "region_area": float((p_region > 0.0).float().mean().item()),
    }


def _rac_area_control(p_final, fg_seed, evidence, bg_core, params):
    area_before = float(p_final.mean().item())
    low = float(params.get("DABE_RAC_AREA_LOW", 0.10))
    high = float(params.get("DABE_RAC_AREA_HIGH", 0.24))
    small_area_flag = bool(area_before < low)
    large_area_flag = bool(area_before > high)
    factor = 1.0
    if large_area_flag and bool(params.get("DABE_RAC_AREA_SUPPRESS_HIGH", True)):
        over = max(0.0, area_before - high)
        factor = 1.0 - over / max(high, EPS)
        factor = max(float(params.get("DABE_RAC_AREA_SUPPRESS_MIN_FACTOR", 0.75)), min(1.0, factor))
        suppress_mask = (fg_seed <= 0.5) & (evidence < 0.75) & (bg_core <= 0.5)
        p_final = torch.where(suppress_mask, p_final * factor, p_final)
    area_after = float(p_final.mean().item())
    return p_final.clamp(0.0, 1.0), small_area_flag, large_area_flag, area_before, area_after, float(factor)


def _component_protection_rac(p_final, feature, fg_seed, bg_seed, edge, params):
    grid = int(params["GRID"])
    tmp_mask = (p_final.reshape(grid, grid) > 0.5).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(tmp_mask, structure=np.ones((3, 3), dtype=np.uint8))
    p_map = p_final.reshape(grid, grid).float().clone()
    fg_map = fg_seed.reshape(grid, grid).bool()
    bg_map = bg_seed.reshape(grid, grid).bool()
    edge_map = edge.reshape(grid, grid).float()
    sem_to_fg, too_few = _compute_core_affinity_max(feature, fg_seed.float(), params)
    sem_map = sem_to_fg.reshape(grid, grid).float()
    fg_np = fg_map.detach().cpu().numpy().astype(bool)
    if fg_np.any():
        dist_to_fg_map = torch.from_numpy(ndimage.distance_transform_edt(~fg_np)).float()
    else:
        dist_to_fg_map = torch.full((grid, grid), float(grid), dtype=torch.float32)

    keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    components = []
    for label_id in range(1, int(num_labels) + 1):
        component_np = labels == label_id
        area = int(component_np.sum())
        if area <= 0:
            continue
        component = torch.from_numpy(component_np).bool()
        y0, y1, x0, x1 = _component_bbox(component_np)
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        mean_p = float(p_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        mean_sem = float(sem_map[component].mean().item()) if not too_few else 0.0
        distance_to_fg = float(dist_to_fg_map[component].min().item())
        contains_fg_seed = bool((fg_map & component).any().item())
        contains_bg_seed = bool((bg_map & component).any().item())
        border_touch = _component_border_touch_ratio(component_np)
        score = mean_p * mean_sem * max(0.0, 1.0 - border_touch)
        components.append(
            {
                "label_id": int(label_id),
                "component": component,
                "area": int(area),
                "score": float(score),
                "mean_p": mean_p,
                "mean_sem": mean_sem,
                "mean_edge": mean_edge,
                "elongation": elongation,
                "distance_to_fg": distance_to_fg,
                "contains_fg_seed": contains_fg_seed,
                "contains_bg_seed": contains_bg_seed,
            }
        )

    topk = int(params.get("DABE_RAC_COMPONENT_TOPK", 5))
    min_area = int(params.get("DABE_RAC_COMPONENT_MIN_AREA", 4))
    sem_keep = float(params.get("DABE_RAC_COMPONENT_SEM_KEEP", 0.55))
    edge_keep = float(params.get("DABE_RAC_COMPONENT_EDGE_KEEP", 0.35))
    elongation_keep = float(params.get("DABE_RAC_COMPONENT_ELONGATION_KEEP", 2.5))
    max_dist = float(params.get("DABE_RAC_COMPONENT_MAX_DIST_TO_FG", 3))
    weak_scale = float(params.get("DABE_RAC_COMPONENT_WEAK_KEEP_SCALE", 0.0))
    high_p = float(params.get("DABE_RAC_REGION_STRONG_KEEP_THRESH", 0.60))
    top_labels = {
        comp["label_id"]
        for comp in sorted(components, key=lambda item: item["score"], reverse=True)[: max(0, topk)]
    }

    num_kept = 0
    num_weak = 0
    num_removed = 0
    for comp in components:
        component = comp["component"]
        if comp["area"] < min_area or (comp["contains_bg_seed"] and not comp["contains_fg_seed"]):
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
            continue
        near_fg = comp["distance_to_fg"] <= max_dist
        keep = (
            (bool(params.get("DABE_RAC_COMPONENT_KEEP_SEED", True)) and comp["contains_fg_seed"])
            or comp["label_id"] in top_labels
            or comp["mean_sem"] >= sem_keep
            or (comp["elongation"] >= elongation_keep and comp["mean_edge"] >= edge_keep and near_fg)
            or (near_fg and comp["mean_p"] >= high_p)
        )
        if keep:
            keep_mask |= component
            num_kept += 1
        elif weak_scale <= 0.0:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
        else:
            p_map[component] = p_map[component] * weak_scale
            weak_mask |= component
            num_weak += 1

    return {
        "p_refined_37": p_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "component_keep_mask_37": keep_mask.float().reshape(1, grid, grid),
        "component_weak_mask_37": weak_mask.float().reshape(1, grid, grid),
        "component_removed_mask_37": removed_mask.float().reshape(1, grid, grid),
        "num_components": int(num_labels),
        "num_kept_components": int(num_kept),
        "num_weak_components": int(num_weak),
        "num_removed_components": int(num_removed),
    }


def _rac_safe_construct_seeds(p_base, fg_core, bg_core, fg_score, residual, evidence, bc_map, params):
    grid = int(params["GRID"])
    fallback_reasons = []
    fg_seed = torch.zeros_like(p_base, dtype=torch.bool)
    bg_seed = torch.zeros_like(p_base, dtype=torch.bool)

    if bool(params.get("DABE_RAC_SAFE_FG_SEED_FROM_FG_CORE", True)):
        fg_seed |= fg_core > 0.5
    score_thr = torch.quantile(
        fg_score.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_SAFE_FG_SEED_SCORE_PERCENTILE", 92.0)) / 100.0)),
    )
    res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_SAFE_FG_SEED_RESIDUAL_PERCENTILE", 70.0)) / 100.0)),
    )
    fg_seed |= (
        (fg_score >= score_thr)
        & (residual >= res_thr)
        & (evidence >= float(params.get("DABE_RAC_SAFE_FG_SEED_EVIDENCE_MIN", 0.45)))
        & (bg_core <= 0.5)
    )
    fg_min = max(1, int(params.get("DABE_RAC_SAFE_FG_SEED_MIN_PIXELS", 4)))
    if int(fg_seed.sum().item()) < fg_min:
        fallback_reasons.append("too_few_fg_seed")
        score = p_base.reshape(-1).float().clone()
        score[bg_core.reshape(-1) > 0.5] = -1.0
        topk = min(fg_min, int(score.numel()))
        top_idx = torch.topk(score, k=topk, largest=True).indices
        fg_seed_flat = fg_seed.reshape(-1)
        fg_seed_flat[top_idx] = True
        fg_seed = fg_seed_flat.reshape_as(fg_seed) & (bg_core <= 0.5)

    if bool(params.get("DABE_RAC_SAFE_BG_SEED_FROM_BG_CORE", True)):
        bg_seed |= bg_core > 0.5
    low_res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_RAC_SAFE_BG_SEED_LOW_RESIDUAL_PERCENTILE", 25.0)) / 100.0)),
    )
    bg_seed |= (
        (bc_map >= float(params.get("DABE_RAC_SAFE_BG_SEED_BC_MIN", 0.60)))
        & (residual <= low_res_thr)
    )
    border = _border_mask(grid, int(params.get("DABE_RAC_SAFE_BG_SEED_BORDER_WIDTH", 1))).reshape(1, grid, grid)
    bg_seed |= border & (p_base < 0.2)
    bg_seed &= ~fg_seed

    bg_min = max(1, int(params.get("DABE_RAC_SAFE_BG_SEED_MIN_PIXELS", 16)))
    if int(bg_seed.sum().item()) < bg_min:
        fallback_reasons.append("too_few_bg_seed")
        residual_norm = _minmax(residual.reshape(-1)).reshape_as(residual)
        bg_score = ((1.0 - residual_norm) * bc_map).reshape(-1).float()
        bg_score[fg_seed.reshape(-1)] = -1.0
        topk = min(bg_min, int(bg_score.numel()))
        top_idx = torch.topk(bg_score, k=topk, largest=True).indices
        bg_seed_flat = bg_seed.reshape(-1)
        bg_seed_flat[top_idx] = True
        bg_seed = bg_seed_flat.reshape_as(bg_seed) & ~fg_seed

    return fg_seed.bool(), bg_seed.bool(), fallback_reasons


def _rac_safe_pair_affinity(src, dst, feat_n, rgb_n, edge_n, bg_seed_flat, bg_core_flat, params):
    sem = ((float(torch.dot(feat_n[src], feat_n[dst]).item()) + 1.0) * 0.5)
    sem = max(0.0, min(1.0, sem)) ** float(params.get("DABE_RAC_SAFE_SEM_POWER", 1.0))
    color_sigma = max(float(params.get("DABE_RAC_SAFE_COLOR_SIGMA", 0.12)), EPS)
    color_dist2 = float(torch.sum((rgb_n[src] - rgb_n[dst]).square()).item())
    color = float(np.exp(-color_dist2 / (2.0 * color_sigma * color_sigma)))
    color = max(0.0, min(1.0, color)) ** float(params.get("DABE_RAC_SAFE_COLOR_POWER", 1.0))
    edge_sigma = max(float(params.get("DABE_RAC_SAFE_EDGE_SIGMA", 0.25)), EPS)
    edge_between = float(torch.maximum(edge_n[src], edge_n[dst]).item())
    edge_stop = float(np.exp(-edge_between / edge_sigma))
    edge_stop = max(0.0, min(1.0, edge_stop)) ** float(params.get("DABE_RAC_SAFE_EDGE_POWER", 1.0))
    weight = sem * color * edge_stop
    if bool(params.get("DABE_RAC_SAFE_REGION_BG_BLOCK", True)) and (bool(bg_seed_flat[dst]) or bool(bg_core_flat[dst])):
        weight *= 0.1
    return float(max(weight, EPS)), float(sem), float(color), float(edge_stop), float(edge_between)


def _rac_safe_region_aff_degree(feature, rgb, edge, bg_seed, bg_core, params):
    grid = int(params["GRID"])
    feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
    rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    edge_n = edge.reshape(-1).float()
    bg_seed_flat = bg_seed.reshape(-1).bool()
    bg_core_flat = bg_core.reshape(-1).float() > 0.5
    radius = max(1, int(params.get("DABE_RAC_SAFE_REGION_RADIUS", 1)))
    degree = torch.zeros(grid * grid, dtype=torch.float32)
    for y in range(grid):
        for x in range(grid):
            src = y * grid + x
            weight_sum = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= grid or nx < 0 or nx >= grid:
                        continue
                    dst = ny * grid + nx
                    weight, _sem, _color, _edge_stop, _edge_between = _rac_safe_pair_affinity(
                        src,
                        dst,
                        feat_n,
                        rgb_n,
                        edge_n,
                        bg_seed_flat,
                        bg_core_flat,
                        params,
                    )
                    weight_sum += weight
            degree[src] = float(weight_sum)
    return _minmax(degree).reshape(1, grid, grid), feat_n, rgb_n, edge_n


def _rac_safe_region_support(candidate_idx, region_flat, feat_n, rgb_n, edge_n, params):
    region_idx = torch.where(region_flat)[0]
    if region_idx.numel() == 0:
        return 0.0, 0.0, 1.0
    cand_feat = feat_n[candidate_idx].unsqueeze(0)
    sem = ((cand_feat @ feat_n.index_select(0, region_idx).t()).max().item() + 1.0) * 0.5
    sem = max(0.0, min(1.0, float(sem))) ** float(params.get("DABE_RAC_SAFE_SEM_POWER", 1.0))
    color_sigma = max(float(params.get("DABE_RAC_SAFE_COLOR_SIGMA", 0.12)), EPS)
    color_dist2 = torch.sum((rgb_n.index_select(0, region_idx) - rgb_n[candidate_idx]).square(), dim=1)
    color = torch.exp(-color_dist2 / (2.0 * color_sigma * color_sigma)).max().item()
    color = max(0.0, min(1.0, float(color))) ** float(params.get("DABE_RAC_SAFE_COLOR_POWER", 1.0))
    edge = float(edge_n[candidate_idx].item())
    return float(sem), float(color), float(edge)


def _rac_safe_grow_regions(feature, rgb, edge, fg_seed, bg_seed, bg_core, base_mask, local_band, params):
    grid = int(params["GRID"])
    total = grid * grid
    _degree, feat_n, rgb_n, edge_n = _rac_safe_region_aff_degree(feature, rgb, edge, bg_seed, bg_core, params)
    fg_np = fg_seed.reshape(grid, grid).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(fg_np, structure=np.ones((3, 3), dtype=np.uint8))
    base_labels, _num_base = ndimage.label(
        base_mask.reshape(grid, grid).detach().cpu().numpy().astype(np.uint8),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    bg_block = ((bg_seed | (bg_core > 0.5)).reshape(grid, grid)).bool()
    local_map = local_band.reshape(grid, grid).bool()
    radius = max(1, int(params.get("DABE_RAC_SAFE_REGION_RADIUS", 1)))
    max_steps = max(0, int(params.get("DABE_RAC_SAFE_MAX_REGION_STEPS", 3)))
    max_area_ratio = float(params.get("DABE_RAC_SAFE_REGION_MAX_AREA_RATIO", 0.18))
    max_rel = float(params.get("DABE_RAC_SAFE_REGION_MAX_REL_TO_BASE_COMP", 1.5))
    sem_min = float(params.get("DABE_RAC_SAFE_REGION_SEM_MIN", 0.58))
    color_min = float(params.get("DABE_RAC_SAFE_REGION_COLOR_MIN", 0.45))
    edge_max = float(params.get("DABE_RAC_SAFE_REGION_EDGE_MAX", 0.45))
    regions = []
    for label_id in range(1, int(num_labels) + 1):
        region = torch.from_numpy(labels == label_id).bool()
        if not bool(region.any().item()):
            continue
        region = region & local_map & ~bg_block
        if not bool(region.any().item()):
            continue

        overlap_base_labels = base_labels[region.detach().cpu().numpy().astype(bool)]
        overlap_base_labels = overlap_base_labels[overlap_base_labels > 0]
        if overlap_base_labels.size:
            base_comp_area = max(int((base_labels == int(base_id)).sum()) for base_id in np.unique(overlap_base_labels))
        else:
            base_comp_area = max(1, int(region.sum().item()))
        max_area_abs = max(1, int(round(max_area_ratio * total)))
        max_area_rel = max(1, int(round(max_rel * base_comp_area)))
        max_area = max(int(region.sum().item()), min(max_area_abs, max_area_rel))

        frontier = region.clone()
        for _step in range(max_steps):
            if int(region.sum().item()) >= max_area:
                break
            new_region = torch.zeros_like(region)
            region_flat = region.reshape(-1).bool()
            for cand in _rac_region_candidate_indices(frontier, region, grid, radius):
                cy, cx = divmod(int(cand), grid)
                if bool(bg_block[cy, cx].item()) or not bool(local_map[cy, cx].item()):
                    continue
                sem, color, edge_value = _rac_safe_region_support(cand, region_flat, feat_n, rgb_n, edge_n, params)
                if sem >= sem_min and color >= color_min and edge_value <= edge_max:
                    new_region[cy, cx] = True
                    if int(region.sum().item() + new_region.sum().item()) >= max_area:
                        break
            if not bool(new_region.any().item()):
                break
            new_region &= ~region
            region |= new_region
            frontier = new_region
        regions.append(
            {
                "seed_label": int(label_id),
                "mask": region,
                "base_component_area": int(base_comp_area),
                "max_region_area": int(max_area),
            }
        )
    return regions, _degree, feat_n, rgb_n, edge_n


def _rac_safe_color_to_seed_map(rgb, fg_seed, params):
    grid = int(params["GRID"])
    rgb_map = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
    fg_flat = fg_seed.reshape(-1).bool()
    if int(fg_flat.sum().item()) == 0:
        return torch.zeros((1, grid, grid), dtype=torch.float32)
    seed_color = rgb_map[fg_flat].mean(dim=0, keepdim=True)
    color_sigma = max(float(params.get("DABE_RAC_SAFE_COLOR_SIGMA", 0.12)), EPS)
    dist2 = torch.sum((rgb_map - seed_color).square(), dim=1)
    color_aff = torch.exp(-dist2 / (2.0 * color_sigma * color_sigma)).clamp(0.0, 1.0)
    return color_aff.reshape(1, grid, grid)


def _rac_safe_score_regions(regions, feature, rgb, p_base, fg_seed, bg_seed, fg_score, residual, evidence, edge, params):
    grid = int(params["GRID"])
    total = grid * grid
    sem_to_fg, too_few = _compute_core_affinity_max(feature, fg_seed.float(), params)
    color_to_fg = _rac_safe_color_to_seed_map(rgb, fg_seed, params)
    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    fg_score_norm = _minmax(fg_score.reshape(-1)).reshape(1, grid, grid)
    p_base_map = p_base.reshape(grid, grid).float()
    fg_map = fg_seed.reshape(grid, grid).bool()
    bg_map = bg_seed.reshape(grid, grid).bool()
    fg_score_map = fg_score_norm.reshape(grid, grid).float()
    residual_map = residual_norm.reshape(grid, grid).float()
    evidence_map = evidence.reshape(grid, grid).float()
    sem_map = sem_to_fg.reshape(grid, grid).float()
    color_map = color_to_fg.reshape(grid, grid).float()
    edge_map = edge.reshape(grid, grid).float()
    median_residual = float(torch.quantile(residual_map.reshape(-1).float(), 0.50).item())
    max_area = max(1, int(round(float(params.get("DABE_RAC_SAFE_REGION_MAX_AREA_RATIO", 0.18)) * total)))
    stats = []
    for idx, item in enumerate(regions, 1):
        component = item["mask"].reshape(grid, grid).bool()
        area = int(component.sum().item())
        if area <= 0:
            continue
        y0, y1, x0, x1 = _component_bbox(component.detach().cpu().numpy().astype(bool))
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        bbox_area = max(1, width * height)
        compactness = float(area / bbox_area)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        border_touch = _component_border_touch_ratio(component.detach().cpu().numpy().astype(bool))
        seed_support = float((component & fg_map).float().sum().item() / max(area, 1))
        bg_overlap = float((component & bg_map).float().sum().item() / max(area, 1))
        sem_support = float(sem_map[component].mean().item()) if not too_few else 0.0
        color_support = float(color_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        edge_support = max(0.0, 1.0 - mean_edge)
        mean_residual = float(residual_map[component].mean().item())
        mean_evidence = float(evidence_map[component].mean().item())
        bg_suppress = max(0.0, min(1.0, 1.0 - bg_overlap))
        border_suppress = max(0.0, min(1.0, 1.0 - border_touch))
        conf = (
            max(0.0, min(1.0, seed_support))
            * max(0.0, min(1.0, sem_support)) ** float(params.get("DABE_RAC_SAFE_CONF_SEM_POWER", 1.5))
            * max(0.0, min(1.0, mean_residual))
            * max(0.0, min(1.0, mean_evidence))
            * max(0.0, min(1.0, color_support))
            * max(0.0, min(1.0, edge_support))
            * bg_suppress ** float(params.get("DABE_RAC_SAFE_CONF_BG_POWER", 2.0))
            * border_suppress ** float(params.get("DABE_RAC_SAFE_CONF_BORDER_POWER", 1.5))
        )
        reject_reason = "none"
        valid_area = bool(area <= max_area and area <= int(item.get("max_region_area", max_area)))
        if bg_overlap > float(params.get("DABE_RAC_SAFE_REJECT_BG_OVERLAP", 0.02)):
            reject_reason = "bg_overlap"
        elif border_touch > float(params.get("DABE_RAC_SAFE_REJECT_BORDER_TOUCH", 0.35)):
            reject_reason = "border_touch"
        elif bool(params.get("DABE_RAC_SAFE_REJECT_REQUIRE_FG_SEED", True)) and seed_support <= 0.0:
            reject_reason = "no_fg_seed"
        elif mean_evidence < float(params.get("DABE_RAC_SAFE_REJECT_EVIDENCE_MIN", 0.30)):
            reject_reason = "low_evidence"
        elif mean_residual < median_residual:
            reject_reason = "low_residual"
        elif not valid_area:
            reject_reason = "invalid_area"
        stats.append(
            {
                "region_id": int(idx),
                "seed_label": int(item.get("seed_label", idx)),
                "area": int(area),
                "area_ratio": float(area / total),
                "base_component_area": int(item.get("base_component_area", 0)),
                "max_region_area": int(item.get("max_region_area", max_area)),
                "fg_seed_overlap": seed_support,
                "bg_seed_overlap": bg_overlap,
                "mean_p_base": float(p_base_map[component].mean().item()),
                "mean_fg_score": float(fg_score_map[component].mean().item()),
                "mean_residual": mean_residual,
                "mean_evidence": mean_evidence,
                "mean_sem_to_fg_seed": sem_support,
                "mean_color_consistency": color_support,
                "mean_edge_boundary": mean_edge,
                "border_touch_ratio": float(border_touch),
                "compactness": compactness,
                "elongation": elongation,
                "confidence": float(conf),
                "valid_area": bool(valid_area),
                "reject_reason": reject_reason,
                "mask": component,
            }
        )
    return stats, sem_to_fg, color_to_fg, residual_norm, fg_score_norm


def _rac_safe_region_support_map(region_stats, params):
    grid = int(params["GRID"])
    p_support = torch.zeros((grid, grid), dtype=torch.float32)
    region_id_map = torch.zeros((grid, grid), dtype=torch.float32)
    kept_mask = torch.zeros((grid, grid), dtype=torch.bool)
    rejected_mask = torch.zeros((grid, grid), dtype=torch.bool)
    keep_thr = float(params.get("DABE_RAC_SAFE_REGION_KEEP_THRESH", 0.45))
    num_kept = 0
    num_rejected = 0
    num_regions = max(1, len(region_stats))
    for item in region_stats:
        component = item["mask"]
        conf = float(item["confidence"])
        region_id_map[component] = max(region_id_map[component].max().item(), float(item["region_id"]) / float(num_regions))
        keep = item["reject_reason"] == "none" and conf >= keep_thr
        if keep:
            kept_mask |= component
            p_support[component] = torch.maximum(p_support[component], torch.full_like(p_support[component], conf))
            item["status"] = "kept"
            num_kept += 1
        else:
            rejected_mask |= component
            item["status"] = "rejected"
            num_rejected += 1
    public_stats = []
    for item in region_stats:
        public_stats.append({key: value for key, value in item.items() if key != "mask"})
    return {
        "p_region_support_37": p_support.reshape(1, grid, grid).clamp(0.0, 1.0),
        "region_id_map_37": region_id_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "kept_region_mask_37": kept_mask.float().reshape(1, grid, grid),
        "rejected_region_mask_37": rejected_mask.float().reshape(1, grid, grid),
        "region_stats": public_stats,
        "num_candidate_regions": int(len(region_stats)),
        "num_kept_regions": int(num_kept),
        "num_rejected_regions": int(num_rejected),
        "region_area": float((p_support > 0.0).float().mean().item()),
    }


def _rac_safe_pixel_completion(p_base, local_band, p_region_support, residual, fg_score, evidence, region_aff_degree, edge, bg_core, params):
    grid = int(params["GRID"])
    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    fg_score_norm = _minmax(fg_score.reshape(-1)).reshape(1, grid, grid)
    aff_norm = _minmax(region_aff_degree.reshape(-1)).reshape(1, grid, grid)
    edge_allow = (1.0 - edge).clamp(0.0, 1.0)
    res_percentile = max(0.0, min(1.0, float(params.get("DABE_RAC_SAFE_PIXEL_RESIDUAL_PERCENTILE", 55.0)) / 100.0))
    res_thr = torch.quantile(residual_norm.reshape(-1).float(), res_percentile)
    valid_pixel = (
        local_band.bool()
        & (residual_norm >= res_thr)
        & (evidence >= float(params.get("DABE_RAC_SAFE_PIXEL_EVIDENCE_MIN", 0.30)))
        & (aff_norm >= float(params.get("DABE_RAC_SAFE_PIXEL_AFF_MIN", 0.50)))
        & (edge <= float(params.get("DABE_RAC_SAFE_PIXEL_EDGE_MAX", 0.60)))
        & (bg_core <= 0.5)
    )
    pixel_gate = (
        residual_norm
        * evidence.clamp(0.0, 1.0)
        * aff_norm
        * edge_allow
        * valid_pixel.float()
    ).clamp(0.0, 1.0)
    pixel_gate = pixel_gate.pow(float(params.get("DABE_RAC_SAFE_PIXEL_GATE_POWER", 1.0))).clamp(0.0, 1.0)
    pixel_completion = (p_region_support * pixel_gate).clamp(0.0, 1.0)
    if bool(params.get("DABE_RAC_SAFE_POSITIVE_ONLY", True)):
        delta_pos = (pixel_completion - p_base).clamp_min(0.0)
    else:
        delta_pos = pixel_completion - p_base
    return {
        "residual_norm_37": residual_norm,
        "fg_score_norm_37": fg_score_norm,
        "aff_norm_37": aff_norm,
        "pixel_gate_37": pixel_gate,
        "pixel_completion_37": pixel_completion,
        "delta_pos_37": delta_pos.clamp(0.0, 1.0),
    }


def _rac_safe_apply_delta_budget(p_base, delta_pos, params):
    grid = int(params["GRID"])
    total = grid * grid
    base_area = float(p_base.mean().item())
    delta_budget = min(
        float(params.get("DABE_RAC_SAFE_DELTA_BUDGET_ABS", 0.03)),
        float(params.get("DABE_RAC_SAFE_DELTA_BUDGET_REL", 0.25)) * base_area,
    )
    delta_budget = max(0.0, float(delta_budget))
    max_final_area = float(base_area + delta_budget)
    completion_weight = float(params.get("DABE_RAC_SAFE_COMPLETION_WEIGHT", 0.45))
    p_safe_before_budget = (p_base + completion_weight * delta_pos).clamp(0.0, 1.0)
    top_delta_mask = torch.zeros_like(delta_pos, dtype=torch.bool)
    if bool(params.get("DABE_RAC_SAFE_USE_TOP_DELTA_BUDGET", True)):
        num_keep = int(delta_budget * total)
        score = delta_pos.reshape(-1).float()
        if num_keep > 0 and float(score.sum().item()) > 0.0:
            num_keep = min(num_keep, int(score.numel()))
            top_idx = torch.topk(score, k=num_keep, largest=True).indices
            top_delta_mask_flat = top_delta_mask.reshape(-1)
            top_delta_mask_flat[top_idx] = True
            top_delta_mask = top_delta_mask_flat.reshape_as(delta_pos)
            delta_pos_budgeted = delta_pos * top_delta_mask.float()
        else:
            delta_pos_budgeted = torch.zeros_like(delta_pos)
    else:
        top_delta_mask = delta_pos > 0.0
        delta_pos_budgeted = delta_pos
    p_safe = (p_base + completion_weight * delta_pos_budgeted).clamp(0.0, 1.0)
    return {
        "p_safe_37": p_safe,
        "p_safe_before_budget_37": p_safe_before_budget,
        "delta_budget": float(delta_budget),
        "max_final_area": float(max_final_area),
        "top_delta_mask_37": top_delta_mask.float(),
        "delta_pos_budgeted_37": delta_pos_budgeted.clamp(0.0, 1.0),
        "area_before_budget": float(p_safe_before_budget.mean().item()),
        "area_after_budget": float(p_safe.mean().item()),
    }


def _component_protection_rac_safe(p_final, feature, fg_seed, bg_seed, edge, params):
    grid = int(params["GRID"])
    tmp_mask = (p_final.reshape(grid, grid) > 0.5).detach().cpu().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(tmp_mask, structure=np.ones((3, 3), dtype=np.uint8))
    p_map = p_final.reshape(grid, grid).float().clone()
    fg_map = fg_seed.reshape(grid, grid).bool()
    bg_map = bg_seed.reshape(grid, grid).bool()
    edge_map = edge.reshape(grid, grid).float()
    sem_to_fg, too_few = _compute_core_affinity_max(feature, fg_seed.float(), params)
    sem_map = sem_to_fg.reshape(grid, grid).float()
    fg_np = fg_map.detach().cpu().numpy().astype(bool)
    if fg_np.any():
        dist_to_fg_map = torch.from_numpy(ndimage.distance_transform_edt(~fg_np)).float()
    else:
        dist_to_fg_map = torch.full((grid, grid), float(grid), dtype=torch.float32)

    keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
    weak_mask = torch.zeros((grid, grid), dtype=torch.bool)
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    components = []
    for label_id in range(1, int(num_labels) + 1):
        component_np = labels == label_id
        area = int(component_np.sum())
        if area <= 0:
            continue
        component = torch.from_numpy(component_np).bool()
        y0, y1, x0, x1 = _component_bbox(component_np)
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)
        elongation = float(max(width, height) / (min(width, height) + EPS))
        mean_p = float(p_map[component].mean().item())
        mean_edge = float(edge_map[component].mean().item())
        mean_sem = float(sem_map[component].mean().item()) if not too_few else 0.0
        distance_to_fg = float(dist_to_fg_map[component].min().item())
        contains_fg_seed = bool((fg_map & component).any().item())
        contains_bg_seed = bool((bg_map & component).any().item())
        border_touch = _component_border_touch_ratio(component_np)
        score = mean_p * mean_sem * max(0.0, 1.0 - border_touch)
        components.append(
            {
                "label_id": int(label_id),
                "component": component,
                "area": int(area),
                "score": float(score),
                "mean_p": mean_p,
                "mean_sem": mean_sem,
                "mean_edge": mean_edge,
                "elongation": elongation,
                "distance_to_fg": distance_to_fg,
                "contains_fg_seed": contains_fg_seed,
                "contains_bg_seed": contains_bg_seed,
            }
        )

    topk = int(params.get("DABE_RAC_SAFE_COMPONENT_TOPK", 5))
    min_area = int(params.get("DABE_RAC_SAFE_COMPONENT_MIN_AREA", 4))
    sem_keep = float(params.get("DABE_RAC_SAFE_COMPONENT_SEM_KEEP", 0.58))
    edge_keep = float(params.get("DABE_RAC_SAFE_COMPONENT_EDGE_KEEP", 0.35))
    elongation_keep = float(params.get("DABE_RAC_SAFE_COMPONENT_ELONGATION_KEEP", 2.5))
    max_dist = float(params.get("DABE_RAC_SAFE_COMPONENT_MAX_DIST_TO_FG", 3))
    weak_scale = float(params.get("DABE_RAC_SAFE_COMPONENT_WEAK_KEEP_SCALE", 0.0))
    high_p = float(params.get("DABE_RAC_SAFE_REGION_KEEP_THRESH", 0.45))
    top_labels = {
        comp["label_id"]
        for comp in sorted(components, key=lambda item: item["score"], reverse=True)[: max(0, topk)]
    }

    num_kept = 0
    num_weak = 0
    num_removed = 0
    for comp in components:
        component = comp["component"]
        if comp["area"] < min_area or (comp["contains_bg_seed"] and not comp["contains_fg_seed"]):
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
            continue
        near_fg = comp["distance_to_fg"] <= max_dist
        keep = (
            (bool(params.get("DABE_RAC_SAFE_COMPONENT_KEEP_SEED", True)) and comp["contains_fg_seed"])
            or comp["label_id"] in top_labels
            or comp["mean_sem"] >= sem_keep
            or (comp["elongation"] >= elongation_keep and comp["mean_edge"] >= edge_keep and near_fg)
            or (near_fg and comp["mean_p"] >= high_p)
        )
        if keep:
            keep_mask |= component
            num_kept += 1
        elif weak_scale <= 0.0:
            p_map[component] = 0.0
            removed_mask |= component
            num_removed += 1
        else:
            p_map[component] = p_map[component] * weak_scale
            weak_mask |= component
            num_weak += 1

    return {
        "p_refined_37": p_map.reshape(1, grid, grid).clamp(0.0, 1.0),
        "component_keep_mask_37": keep_mask.float().reshape(1, grid, grid),
        "component_weak_mask_37": weak_mask.float().reshape(1, grid, grid),
        "component_removed_mask_37": removed_mask.float().reshape(1, grid, grid),
        "num_components": int(num_labels),
        "num_kept_components": int(num_kept),
        "num_weak_components": int(num_weak),
        "num_removed_components": int(num_removed),
    }


def _apply_v3_area_prior(p_v3, fg_core, evidence, params):
    area_before = float(p_v3.mean().item())
    low = float(params.get("DABE_V3_AREA_PRIOR_LOW", 0.10))
    high = float(params.get("DABE_V3_AREA_PRIOR_HIGH", 0.22))
    small_area_flag = bool(area_before < low)
    large_area_flag = bool(area_before > high)
    if large_area_flag and bool(params.get("DABE_V3_AREA_PRIOR_SOFT", True)):
        over = max(0.0, area_before - high)
        suppress = max(0.75, min(1.0, 1.0 - over / max(high, EPS)))
        protect = (fg_core > 0.5) | (evidence > 0.75)
        p_v3 = torch.where(protect, p_v3, p_v3 * suppress)
    area_after = float(p_v3.mean().item())
    return p_v3.clamp(0.0, 1.0), small_area_flag, large_area_flag, area_before, area_after


def _run_single_view_v3(feature, rgb, params):
    grid = int(params["GRID"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    evidence_floor = float(params.get("DABE_V3_EVIDENCE_FLOOR", 0.50))
    evidence_soft = (evidence_floor + (1.0 - evidence_floor) * evidence).clamp(0.0, 1.0)
    p_rw_evid = (p_rw * evidence_soft).clamp(0.0, 1.0)

    core_affinity, too_few_core = _compute_core_affinity(feature, fg_core, fg_score, params)
    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    evidence_norm = evidence.clamp(0.0, 1.0)
    expand_score = core_affinity.clone()
    expand_score = expand_score * residual_norm.pow(float(params.get("DABE_V3_RESIDUAL_EXPAND_POWER", 0.5)))
    expand_score = expand_score * evidence_norm.pow(float(params.get("DABE_V3_EVIDENCE_EXPAND_POWER", 0.5)))
    expand_score = expand_score * (1.0 - bg_core)
    if too_few_core:
        p_expand = torch.zeros_like(expand_score)
        fallback_flag = True
        fallback_reason = "too_few_fg_core_for_v3_expand"
    else:
        valid = ((1.0 - bg_core) > 0.5).reshape(-1)
        valid_values = expand_score.reshape(-1)[valid]
        if valid_values.numel() == 0:
            valid_values = expand_score.reshape(-1)
        percentile = float(params.get("DABE_V3_EXPAND_PERCENTILE", 82.0)) / 100.0
        threshold = torch.quantile(valid_values.float(), max(0.0, min(1.0, percentile)))
        p_expand = torch.sigmoid((expand_score - threshold) / float(params.get("DABE_V3_EXPAND_TAU", 0.08)))
        p_expand = p_expand * (core_affinity >= float(params.get("DABE_V3_CORE_AFF_MIN", 0.45))).float()
        p_expand = p_expand * (1.0 - bg_core)
        p_expand = p_expand.clamp(0.0, 1.0)
        fallback_flag = False
        fallback_reason = "none"

    p_v3 = p_rw_evid + float(params.get("DABE_V3_EXPAND_WEIGHT", 0.35)) * (1.0 - p_rw_evid) * p_expand
    p_v3 = torch.where(
        fg_core > 0.5,
        torch.maximum(p_v3, torch.full_like(p_v3, float(params.get("DABE_V3_FG_CORE_MIN_VALUE", 0.90)))),
        p_v3,
    )
    p_v3 = torch.where(
        bg_core > 0.5,
        torch.minimum(p_v3, torch.full_like(p_v3, float(params.get("DABE_V3_BG_SUPPRESS_VALUE", 0.03)))),
        p_v3,
    ).clamp(0.0, 1.0)

    component = _component_protection(
        p_v3,
        fg_core,
        bg_core,
        fg_score,
        residual,
        evidence,
        core_affinity,
        edge,
        params,
    )
    p_v3 = component["p_refined_37"]
    p_v3, small_area_flag, large_area_flag, area_before, area_after = _apply_v3_area_prior(
        p_v3,
        fg_core,
        evidence,
        params,
    )
    uncertain = (1.0 - torch.clamp(fg_core + bg_core, 0.0, 1.0)).float()

    base_fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        base_fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    if fallback_flag:
        base_fallback_reasons.append(fallback_reason)

    out = dict(base)
    out.update(
        {
            "p_dabe_37": p_v3,
            "p_dabe_v3_37": p_v3,
            "p_rw_evid_37": p_rw_evid,
            "evidence_soft_37": evidence_soft,
            "core_affinity_37": core_affinity,
            "expand_score_37": expand_score.clamp(0.0, 1.0),
            "p_expand_37": p_expand,
            "uncertain_37": uncertain,
            "component_keep_mask_37": component["component_keep_mask_37"],
            "component_weak_mask_37": component["component_weak_mask_37"],
            "num_components": int(component["num_components"]),
            "num_kept_components": int(component["num_kept_components"]),
            "num_weak_components": int(component["num_weak_components"]),
            "num_removed_components": int(component["num_removed_components"]),
            "small_area_flag": bool(small_area_flag),
            "large_area_flag": bool(large_area_flag),
            "area_before_area_prior": float(area_before),
            "area_after_area_prior": float(area_after),
            "fallback_flag": bool(base_fallback_reasons),
            "fallback_reason": "|".join(sorted(set(base_fallback_reasons))) if base_fallback_reasons else "none",
        }
    )
    return out


def _run_single_view_v31(feature, rgb, params):
    grid = int(params["GRID"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_V31_BASE_EVIDENCE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_V31_BASE_EVIDENCE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)
    base_area = float(p_base.mean().item())
    base_mask = p_base > float(params.get("DABE_V31_BASE_MASK_THRESH", 0.35))
    adaptive_radius, adaptive_expand_weight = _v31_adaptive_expand(base_area, params)

    if bool(params.get("DABE_V31_USE_LOCAL_BAND", True)):
        candidate_band = _dilate_mask(base_mask, adaptive_radius)
    else:
        candidate_band = torch.ones_like(base_mask, dtype=torch.bool)
    if bool(params.get("DABE_V31_CANDIDATE_EXCLUDE_BG_CORE", True)):
        candidate_band = candidate_band & (bg_core <= 0.5)

    core_affinity, too_few_core = _compute_core_affinity_max(feature, fg_core, params)
    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    evidence_norm = evidence.clamp(0.0, 1.0)
    expand_score = (
        core_affinity
        * residual_norm.sqrt()
        * evidence_norm.sqrt()
    ).clamp(0.0, 1.0)

    residual_percentile = max(0.0, min(1.0, float(params.get("DABE_V31_RESIDUAL_PERCENTILE", 60.0)) / 100.0))
    residual_thr = torch.quantile(residual_norm.reshape(-1).float(), residual_percentile)
    valid_expand = (
        candidate_band
        & (core_affinity >= float(params.get("DABE_V31_CORE_AFF_MIN", 0.58)))
        & (residual_norm >= residual_thr)
        & (evidence >= float(params.get("DABE_V31_EVIDENCE_MIN", 0.35)))
        & (bg_core <= 0.5)
    )

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    if too_few_core:
        p_expand = torch.zeros_like(p_base)
        expand_thr = 0.0
        fallback_reasons.append("too_few_fg_core_for_v31_expand")
    else:
        valid_values = expand_score[valid_expand]
        if valid_values.numel() > 0:
            expand_percentile = max(0.0, min(1.0, float(params.get("DABE_V31_EXPAND_PERCENTILE", 88.0)) / 100.0))
            expand_thr_tensor = torch.quantile(valid_values.float(), expand_percentile)
            expand_thr = float(expand_thr_tensor.item())
            p_expand = torch.sigmoid(
                (expand_score - expand_thr_tensor) / float(params.get("DABE_V31_EXPAND_TAU", 0.05))
            )
            p_expand = (p_expand * valid_expand.float()).clamp(0.0, 1.0)
        else:
            p_expand = torch.zeros_like(p_base)
            expand_thr = 0.0

    p_v31 = p_base + float(adaptive_expand_weight) * (1.0 - p_base) * p_expand
    p_v31 = torch.where(
        fg_core > 0.5,
        torch.maximum(p_v31, torch.full_like(p_v31, float(params.get("DABE_V31_FG_CORE_MIN_VALUE", 0.90)))),
        p_v31,
    )
    p_v31 = torch.where(
        bg_core > 0.5,
        torch.minimum(p_v31, torch.full_like(p_v31, float(params.get("DABE_V31_BG_SUPPRESS_VALUE", 0.03)))),
        p_v31,
    ).clamp(0.0, 1.0)

    component = _component_protection_v31(
        p_v31,
        base_mask.float(),
        fg_core,
        bg_core,
        residual_norm,
        evidence,
        core_affinity,
        edge,
        params,
    )
    p_v31 = component["p_refined_37"]
    uncertain = (1.0 - torch.clamp(fg_core + bg_core, 0.0, 1.0)).float()

    out = dict(base)
    out.update(
        {
            "p_dabe_37": p_v31,
            "p_dabe_v31_37": p_v31,
            "p_base_37": p_base,
            "base_mask_37": base_mask.float(),
            "candidate_band_37": candidate_band.float(),
            "core_affinity_37": core_affinity,
            "residual_norm_37": residual_norm,
            "expand_score_37": expand_score,
            "valid_expand_37": valid_expand.float(),
            "p_expand_37": p_expand,
            "uncertain_37": uncertain,
            "component_keep_mask_37": component["component_keep_mask_37"],
            "component_weak_mask_37": component["component_weak_mask_37"],
            "component_removed_mask_37": component["component_removed_mask_37"],
            "num_components": int(component["num_components"]),
            "num_kept_components": int(component["num_kept_components"]),
            "num_weak_components": int(component["num_weak_components"]),
            "num_removed_components": int(component["num_removed_components"]),
            "base_area": float(base_area),
            "expand_area": float((p_expand > 0.5).float().mean().item()),
            "valid_expand_area": float(valid_expand.float().mean().item()),
            "adaptive_radius": int(adaptive_radius),
            "adaptive_expand_weight": float(adaptive_expand_weight),
            "expand_thr": float(expand_thr),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _run_single_view_rac(feature, rgb, params):
    grid = int(params["GRID"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    bc_map = base["bc_map_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_RAC_BASE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_RAC_BASE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)
    fg_seed, bg_seed, seed_fallback_reasons = _rac_construct_seeds(
        p_base,
        fg_core,
        bg_core,
        fg_score,
        residual,
        evidence,
        bc_map,
        params,
    )
    regions, region_aff_degree, _feat_n, _rgb_n, _edge_n = _rac_grow_regions(
        feature,
        rgb,
        edge,
        fg_seed,
        bg_seed,
        bg_core,
        params,
    )
    region_stats, core_affinity, color_affinity, residual_norm = _rac_score_regions(
        regions,
        feature,
        rgb,
        p_base,
        fg_seed,
        bg_seed,
        fg_score,
        residual,
        evidence,
        edge,
        params,
    )
    region = _rac_region_completion(region_stats, p_base, params)
    p_rac = region["p_rac"]
    p_rac = torch.where(
        fg_seed,
        torch.maximum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_FG_SEED_VALUE", 0.95)))),
        p_rac,
    )
    p_rac = torch.where(
        bg_seed,
        torch.minimum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_BG_SEED_VALUE", 0.02)))),
        p_rac,
    )
    p_rac = torch.where(
        bg_core > 0.5,
        torch.minimum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_BG_CORE_SUPPRESS", 0.03)))),
        p_rac,
    ).clamp(0.0, 1.0)
    p_rac_before_area = p_rac.clone()
    p_rac, small_area_flag, large_area_flag, area_before, area_after, suppress_factor = _rac_area_control(
        p_rac,
        fg_seed.float(),
        evidence,
        bg_core,
        params,
    )
    component = _component_protection_rac(
        p_rac,
        feature,
        fg_seed.float(),
        bg_seed.float(),
        edge,
        params,
    )
    p_rac = component["p_refined_37"]
    p_rac = torch.where(
        fg_seed,
        torch.maximum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_FG_SEED_VALUE", 0.95)))),
        p_rac,
    )
    p_rac = torch.where(
        bg_seed,
        torch.minimum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_BG_SEED_VALUE", 0.02)))),
        p_rac,
    )
    p_rac = torch.where(
        bg_core > 0.5,
        torch.minimum(p_rac, torch.full_like(p_rac, float(params.get("DABE_RAC_BG_CORE_SUPPRESS", 0.03)))),
        p_rac,
    ).clamp(0.0, 1.0)
    uncertain = (1.0 - torch.clamp(fg_core + bg_core, 0.0, 1.0)).float()

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    fallback_reasons.extend(seed_fallback_reasons)

    out = dict(base)
    out.update(
        {
            "p_dabe_37": p_rac,
            "p_dabe_rac_37": p_rac,
            "p_base_37": p_base,
            "fg_seed_37": fg_seed.float(),
            "bg_seed_37": bg_seed.float(),
            "fg_seed_area": float(fg_seed.float().mean().item()),
            "bg_seed_area": float(bg_seed.float().mean().item()),
            "region_id_map_37": region["region_id_map_37"],
            "p_region_37": region["p_region_37"],
            "strong_region_mask_37": region["strong_region_mask_37"],
            "weak_region_mask_37": region["weak_region_mask_37"],
            "rejected_region_mask_37": region["rejected_region_mask_37"],
            "delta_pos_37": region["delta_pos_37"],
            "p_rac_before_area_37": p_rac_before_area,
            "edge_37": edge,
            "region_aff_degree_37": region_aff_degree,
            "core_affinity_37": core_affinity,
            "color_affinity_37": color_affinity,
            "residual_norm_37": residual_norm,
            "fg_core_37": fg_core,
            "bg_core_37": bg_core,
            "uncertain_37": uncertain,
            "component_keep_mask_37": component["component_keep_mask_37"],
            "component_weak_mask_37": component["component_weak_mask_37"],
            "component_removed_mask_37": component["component_removed_mask_37"],
            "num_components": int(component["num_components"]),
            "num_kept_components": int(component["num_kept_components"]),
            "num_weak_components": int(component["num_weak_components"]),
            "num_removed_components": int(component["num_removed_components"]),
            "num_candidate_regions": int(region["num_candidate_regions"]),
            "num_strong_regions": int(region["num_strong_regions"]),
            "num_weak_regions": int(region["num_weak_regions"]),
            "num_rejected_regions": int(region["num_rejected_regions"]),
            "region_stats": region["region_stats"],
            "base_area": float(p_base.mean().item()),
            "region_area": float(region["region_area"]),
            "area_before_area_control": float(area_before),
            "area_after_area_control": float(area_after),
            "small_area_flag": bool(small_area_flag),
            "large_area_flag": bool(large_area_flag),
            "area_suppress_factor": float(suppress_factor),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _run_single_view_rac_safe(feature, rgb, params):
    grid = int(params["GRID"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    bc_map = base["bc_map_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_RAC_SAFE_BASE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_RAC_SAFE_BASE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)
    base_area = float(p_base.mean().item())
    base_mask = p_base > float(params.get("DABE_RAC_SAFE_BASE_MASK_THRESH", 0.30))
    if bool(params.get("DABE_RAC_SAFE_USE_LOCAL_BAND", True)):
        local_band = _dilate_mask(base_mask, int(params.get("DABE_RAC_SAFE_LOCAL_BAND_RADIUS", 3)))
    else:
        local_band = torch.ones_like(base_mask, dtype=torch.bool)
    if bool(params.get("DABE_RAC_SAFE_LOCAL_BAND_EXCLUDE_BG_CORE", True)):
        local_band = local_band & (bg_core <= 0.5)

    fg_seed, bg_seed, seed_fallback_reasons = _rac_safe_construct_seeds(
        p_base,
        fg_core,
        bg_core,
        fg_score,
        residual,
        evidence,
        bc_map,
        params,
    )
    fg_seed = fg_seed & local_band
    fg_min = max(1, int(params.get("DABE_RAC_SAFE_FG_SEED_MIN_PIXELS", 4)))
    if int(fg_seed.sum().item()) < fg_min:
        seed_fallback_reasons.append("too_few_fg_seed_local_band")
        score = p_base.reshape(-1).float().clone()
        score[(bg_core.reshape(-1) > 0.5) | (~local_band.reshape(-1).bool())] = -1.0
        topk = min(fg_min, int(score.numel()))
        top_idx = torch.topk(score, k=topk, largest=True).indices
        fg_seed_flat = fg_seed.reshape(-1)
        fg_seed_flat[top_idx] = True
        fg_seed = fg_seed_flat.reshape_as(fg_seed) & local_band & (bg_core <= 0.5)
        bg_seed &= ~fg_seed

    regions, region_aff_degree, _feat_n, _rgb_n, _edge_n = _rac_safe_grow_regions(
        feature,
        rgb,
        edge,
        fg_seed,
        bg_seed,
        bg_core,
        base_mask,
        local_band,
        params,
    )
    region_stats, core_affinity, color_affinity, residual_norm, fg_score_norm = _rac_safe_score_regions(
        regions,
        feature,
        rgb,
        p_base,
        fg_seed,
        bg_seed,
        fg_score,
        residual,
        evidence,
        edge,
        params,
    )
    region = _rac_safe_region_support_map(region_stats, params)
    pixel = _rac_safe_pixel_completion(
        p_base,
        local_band,
        region["p_region_support_37"],
        residual,
        fg_score,
        evidence,
        region_aff_degree,
        edge,
        bg_core,
        params,
    )
    budget = _rac_safe_apply_delta_budget(p_base, pixel["delta_pos_37"], params)
    p_safe = budget["p_safe_37"]
    p_safe = torch.where(
        fg_seed,
        torch.maximum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_FG_SEED_VALUE", 0.95)))),
        p_safe,
    )
    p_safe = torch.where(
        bg_seed,
        torch.minimum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_BG_SEED_VALUE", 0.02)))),
        p_safe,
    )
    p_safe = torch.where(
        bg_core > 0.5,
        torch.minimum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_BG_CORE_SUPPRESS", 0.03)))),
        p_safe,
    ).clamp(0.0, 1.0)
    p_safe_after_core_lock = p_safe.clone()

    component = _component_protection_rac_safe(
        p_safe,
        feature,
        fg_seed.float(),
        bg_seed.float(),
        edge,
        params,
    )
    p_safe = component["p_refined_37"]
    p_safe = torch.where(
        fg_seed,
        torch.maximum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_FG_SEED_VALUE", 0.95)))),
        p_safe,
    )
    p_safe = torch.where(
        bg_seed,
        torch.minimum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_BG_SEED_VALUE", 0.02)))),
        p_safe,
    )
    p_safe = torch.where(
        bg_core > 0.5,
        torch.minimum(p_safe, torch.full_like(p_safe, float(params.get("DABE_RAC_SAFE_BG_CORE_SUPPRESS", 0.03)))),
        p_safe,
    ).clamp(0.0, 1.0)
    uncertain = (1.0 - torch.clamp(fg_core + bg_core, 0.0, 1.0)).float()

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    fallback_reasons.extend(seed_fallback_reasons)
    safe_area = float(p_safe.mean().item())

    out = dict(base)
    out.update(
        {
            "p_dabe_37": p_safe,
            "p_dabe_rac_safe_37": p_safe,
            "p_base_37": p_base,
            "base_mask_37": base_mask.float(),
            "local_band_37": local_band.float(),
            "local_band_area": float(local_band.float().mean().item()),
            "fg_seed_37": fg_seed.float(),
            "bg_seed_37": bg_seed.float(),
            "fg_seed_area": float(fg_seed.float().mean().item()),
            "bg_seed_area": float(bg_seed.float().mean().item()),
            "region_id_map_37": region["region_id_map_37"],
            "p_region_support_37": region["p_region_support_37"],
            "kept_region_mask_37": region["kept_region_mask_37"],
            "rejected_region_mask_37": region["rejected_region_mask_37"],
            "region_aff_degree_37": region_aff_degree,
            "edge_37": edge,
            "core_affinity_37": core_affinity,
            "color_affinity_37": color_affinity,
            "residual_norm_37": pixel["residual_norm_37"],
            "fg_score_norm_37": pixel["fg_score_norm_37"],
            "aff_norm_37": pixel["aff_norm_37"],
            "pixel_gate_37": pixel["pixel_gate_37"],
            "pixel_completion_37": pixel["pixel_completion_37"],
            "delta_pos_37": pixel["delta_pos_37"],
            "p_safe_before_budget_37": budget["p_safe_before_budget_37"],
            "top_delta_mask_37": budget["top_delta_mask_37"],
            "delta_pos_budgeted_37": budget["delta_pos_budgeted_37"],
            "p_safe_after_core_lock_37": p_safe_after_core_lock,
            "fg_core_37": fg_core,
            "bg_core_37": bg_core,
            "uncertain_37": uncertain,
            "component_keep_mask_37": component["component_keep_mask_37"],
            "component_weak_mask_37": component["component_weak_mask_37"],
            "component_removed_mask_37": component["component_removed_mask_37"],
            "num_components": int(component["num_components"]),
            "num_kept_components": int(component["num_kept_components"]),
            "num_weak_components": int(component["num_weak_components"]),
            "num_removed_components": int(component["num_removed_components"]),
            "num_candidate_regions": int(region["num_candidate_regions"]),
            "num_kept_regions": int(region["num_kept_regions"]),
            "num_rejected_regions": int(region["num_rejected_regions"]),
            "region_stats": region["region_stats"],
            "base_area": float(base_area),
            "safe_area": float(safe_area),
            "region_area": float(region["region_area"]),
            "delta_budget": float(budget["delta_budget"]),
            "max_final_area": float(budget["max_final_area"]),
            "area_before_budget": float(budget["area_before_budget"]),
            "area_after_budget": float(budget["area_after_budget"]),
            "small_area_flag": bool(safe_area < float(params.get("DABE_RAC_SAFE_TARGET_AREA_LOW", 0.13))),
            "large_area_flag": bool(safe_area > float(params.get("DABE_RAC_SAFE_TARGET_AREA_HIGH", 0.155))),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _pu_construct_fg_core(p_base, fg_core, bg_core, fg_score, residual, evidence, params):
    fallback_reasons = []
    fg_core_pu = torch.zeros_like(p_base, dtype=torch.bool)
    if bool(params.get("DABE_PU_FG_CORE_FROM_V2", True)):
        fg_core_pu |= fg_core > 0.5
    res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_PU_FG_CORE_RESIDUAL_PERCENTILE", 70.0)) / 100.0)),
    )
    score_thr = torch.quantile(
        fg_score.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_PU_FG_CORE_FG_SCORE_PERCENTILE", 88.0)) / 100.0)),
    )
    fg_core_pu |= (
        (p_base >= float(params.get("DABE_PU_FG_CORE_P_BASE_THRESH", 0.55)))
        & (evidence >= float(params.get("DABE_PU_FG_CORE_EVIDENCE_MIN", 0.50)))
        & (residual >= res_thr)
        & (fg_score >= score_thr)
        & (bg_core <= 0.5)
    )
    fg_min = max(1, int(params.get("DABE_PU_FG_CORE_MIN_PIXELS", 4)))
    if int(fg_core_pu.sum().item()) < fg_min:
        fallback_reasons.append("too_few_pu_fg_core")
        score = p_base.reshape(-1).float().clone()
        score[bg_core.reshape(-1) > 0.5] = -1.0
        topk = min(fg_min, int(score.numel()))
        top_idx = torch.topk(score, k=topk, largest=True).indices
        fg_flat = fg_core_pu.reshape(-1)
        fg_flat[top_idx] = True
        fg_core_pu = fg_flat.reshape_as(fg_core_pu) & (bg_core <= 0.5)
    return fg_core_pu.bool(), fallback_reasons


def _pu_construct_bg_core(p_base, fg_core_pu, bg_core, bc_map, residual, params):
    grid = int(params["GRID"])
    fallback_reasons = []
    bg_core_pu = torch.zeros_like(p_base, dtype=torch.bool)
    if bool(params.get("DABE_PU_BG_CORE_FROM_V2", True)):
        bg_core_pu |= bg_core > 0.5
    low_res_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_PU_BG_CORE_RESIDUAL_PERCENTILE", 25.0)) / 100.0)),
    )
    bg_core_pu |= (
        (bc_map >= float(params.get("DABE_PU_BG_CORE_BC_MIN", 0.65)))
        & (residual <= low_res_thr)
        & (p_base <= float(params.get("DABE_PU_BG_CORE_P_BASE_MAX", 0.15)))
    )
    border = _border_mask(grid, int(params.get("DABE_PU_BG_CORE_BORDER_WIDTH", 1))).reshape(1, grid, grid)
    bg_core_pu |= border & (p_base <= float(params.get("DABE_PU_BG_CORE_P_BASE_MAX", 0.15)))
    bg_core_pu &= ~fg_core_pu

    bg_min = max(1, int(params.get("DABE_PU_BG_CORE_MIN_PIXELS", 16)))
    if int(bg_core_pu.sum().item()) < bg_min:
        fallback_reasons.append("too_few_pu_bg_core")
        residual_norm = _minmax(residual.reshape(-1)).reshape_as(residual)
        bg_score = ((1.0 - residual_norm) * bc_map * (1.0 - p_base)).reshape(-1).float()
        bg_score[fg_core_pu.reshape(-1)] = -1.0
        topk = min(bg_min, int(bg_score.numel()))
        top_idx = torch.topk(bg_score, k=topk, largest=True).indices
        bg_flat = bg_core_pu.reshape(-1)
        bg_flat[top_idx] = True
        bg_core_pu = bg_flat.reshape_as(bg_core_pu) & ~fg_core_pu
    return bg_core_pu.bool(), fallback_reasons


def _pu_cleanup_extent(extent_candidate, extent_score, p_base, params):
    if not bool(params.get("DABE_PU_REMOVE_ISOLATED_EXTENT", True)):
        return extent_candidate.clamp(0.0, 1.0)
    grid = int(params["GRID"])
    extent_np = (extent_candidate.detach().cpu().float().squeeze().numpy() > 0.0).astype(np.uint8)
    labels, num_labels = ndimage.label(extent_np, structure=np.ones((3, 3), dtype=np.uint8))
    if int(num_labels) == 0:
        return torch.zeros_like(extent_candidate)
    base_mask = (p_base > float(params.get("DABE_PU_BASE_FG_THRESH", 0.35)))
    base_band = _dilate_mask(base_mask, 1).reshape(grid, grid).bool()
    extent_map = extent_candidate.reshape(grid, grid).float().clone()
    score_map = extent_score.reshape(grid, grid).float()
    min_area = max(1, int(params.get("DABE_PU_EXTENT_MIN_COMPONENT_AREA", 3)))
    components = []
    for label_id in range(1, int(num_labels) + 1):
        component = torch.from_numpy(labels == label_id).bool()
        area = int(component.sum().item())
        if area < min_area or not bool((component & base_band).any().item()):
            extent_map[component] = 0.0
            continue
        components.append(
            {
                "component": component,
                "area": area,
                "score": float(score_map[component].mean().item()),
            }
        )
    max_area = max(0, int(round(float(params.get("DABE_PU_EXTENT_MAX_AREA_RATIO", 0.35)) * grid * grid)))
    current_area = int((extent_map > 0.0).sum().item())
    if current_area > max_area and max_area > 0:
        keep_mask = torch.zeros((grid, grid), dtype=torch.bool)
        used = 0
        for comp in sorted(components, key=lambda item: item["score"], reverse=True):
            if used + comp["area"] > max_area and used > 0:
                continue
            keep_mask |= comp["component"]
            used += comp["area"]
            if used >= max_area:
                break
        extent_map = torch.where(keep_mask, extent_map, torch.zeros_like(extent_map))
    return extent_map.reshape(1, grid, grid).clamp(0.0, 1.0)


def _run_single_view_pu(feature, rgb, params):
    grid = int(params["GRID"])
    loss_size = int(params["LOSS_SIZE"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    bc_map = base["bc_map_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_PU_BASE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_PU_BASE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)
    fg_core_pu, fg_fallback = _pu_construct_fg_core(p_base, fg_core, bg_core, fg_score, residual, evidence, params)
    bg_core_pu, bg_fallback = _pu_construct_bg_core(p_base, fg_core_pu, bg_core, bc_map, residual, params)

    base_band_seed = p_base > float(params.get("DABE_PU_EXTENT_BAND_BASE_THRESH", 0.18))
    if bool(params.get("DABE_PU_EXTENT_USE_LOCAL_BAND", True)):
        extent_band = _dilate_mask(base_band_seed, int(params.get("DABE_PU_EXTENT_BAND_RADIUS", 3)))
    else:
        extent_band = torch.ones_like(base_band_seed, dtype=torch.bool)
    if bool(params.get("DABE_PU_EXTENT_EXCLUDE_BG_CORE", True)):
        extent_band = extent_band & ~bg_core_pu
    extent_candidate_region = extent_band & ~fg_core_pu & ~bg_core_pu

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    fallback_reasons.extend(fg_fallback)
    fallback_reasons.extend(bg_fallback)
    if bool(params.get("DABE_PU_USE_CORE_AFFINITY", True)):
        core_affinity, too_few_core = _compute_core_affinity_max(feature, fg_core_pu.float(), params)
        if too_few_core:
            core_affinity = torch.ones_like(p_base)
            fallback_reasons.append("too_few_pu_fg_core_for_affinity")
    else:
        core_affinity = torch.ones_like(p_base)

    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    fg_score_norm = _minmax(fg_score.reshape(-1)).reshape(1, grid, grid)
    evidence_norm = evidence.clamp(0.0, 1.0)
    edge_allow = (1.0 - edge).clamp(0.0, 1.0)
    extent_score = (
        0.35 * residual_norm
        + 0.30 * fg_score_norm
        + 0.20 * evidence_norm
        + 0.15 * edge_allow
    ).clamp(0.0, 1.0)
    if bool(params.get("DABE_PU_USE_CORE_AFFINITY", True)):
        core_weight = float(params.get("DABE_PU_CORE_AFF_WEIGHT", 0.35))
        extent_score = ((1.0 - core_weight) * extent_score + core_weight * core_affinity).clamp(0.0, 1.0)

    res_extent_thr = torch.quantile(
        residual.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_PU_EXTENT_RESIDUAL_PERCENTILE", 45.0)) / 100.0)),
    )
    score_extent_thr = torch.quantile(
        fg_score.reshape(-1).float(),
        max(0.0, min(1.0, float(params.get("DABE_PU_EXTENT_FG_SCORE_PERCENTILE", 45.0)) / 100.0)),
    )
    valid_extent = (
        extent_candidate_region
        & (evidence >= float(params.get("DABE_PU_EXTENT_EVIDENCE_MIN", 0.20)))
        & (residual >= res_extent_thr)
        & (fg_score >= score_extent_thr)
        & (edge <= float(params.get("DABE_PU_EXTENT_EDGE_MAX", 0.75)))
    )
    if bool(params.get("DABE_PU_USE_CORE_AFFINITY", True)):
        valid_extent &= core_affinity >= float(params.get("DABE_PU_CORE_AFF_MIN", 0.45))
    extent_candidate = (extent_score * valid_extent.float()).clamp(0.0, 1.0)
    extent_candidate = _pu_cleanup_extent(extent_candidate, extent_score, p_base, params)

    extent_mask = extent_candidate > 0.0
    known_union = torch.clamp(fg_core_pu.float() + bg_core_pu.float() + extent_mask.float(), 0.0, 1.0)
    unknown = (1.0 - known_union).clamp(0.0, 1.0)

    target_soft = torch.full_like(p_base, float(params.get("DABE_PU_TARGET_UNKNOWN", 0.50)))
    target_soft = torch.where(bg_core_pu, torch.full_like(target_soft, float(params.get("DABE_PU_TARGET_BG_CORE", 0.0))), target_soft)
    target_soft = torch.where(fg_core_pu, torch.full_like(target_soft, float(params.get("DABE_PU_TARGET_FG_CORE", 1.0))), target_soft)
    extent_value = float(params.get("DABE_PU_TARGET_EXTENT_MIN", 0.35)) + (
        float(params.get("DABE_PU_TARGET_EXTENT_MAX", 0.55)) - float(params.get("DABE_PU_TARGET_EXTENT_MIN", 0.35))
    ) * extent_candidate
    target_soft = torch.where(extent_mask & ~fg_core_pu & ~bg_core_pu, extent_value, target_soft).clamp(0.0, 1.0)

    weight_map = torch.full_like(p_base, float(params.get("DABE_PU_WEIGHT_UNKNOWN", 0.02)))
    weight_map = torch.maximum(weight_map, float(params.get("DABE_PU_WEIGHT_SOFT_BASE", 0.10)) * p_base)
    weight_map = torch.where(extent_mask, torch.full_like(weight_map, float(params.get("DABE_PU_WEIGHT_EXTENT", 0.20))), weight_map)
    weight_map = torch.where(bg_core_pu, torch.full_like(weight_map, float(params.get("DABE_PU_WEIGHT_BG_CORE", 1.0))), weight_map)
    weight_map = torch.where(fg_core_pu, torch.full_like(weight_map, float(params.get("DABE_PU_WEIGHT_FG_CORE", 1.0))), weight_map)
    weight_map = weight_map.clamp(0.0, 1.0)

    fg_core_pu_68 = _resize_to_loss_nearest(fg_core_pu.float(), loss_size)
    bg_core_pu_68 = _resize_to_loss_nearest(bg_core_pu.float(), loss_size)
    extent_candidate_68 = _resize_to_loss(extent_candidate, loss_size)
    target_soft_68 = _resize_to_loss(target_soft, loss_size)
    weight_map_68 = _resize_to_loss(weight_map, loss_size)
    known_68 = torch.clamp(
        fg_core_pu_68 + bg_core_pu_68 + (extent_candidate_68 > 0.0).float(),
        0.0,
        1.0,
    )
    unknown_68 = (1.0 - known_68).clamp(0.0, 1.0)

    out = dict(base)
    out.update(
        {
            "p_dabe_37": target_soft,
            "p_dabe_68": target_soft_68,
            "p_base_37": p_base,
            "p_base_68": _resize_to_loss(p_base, loss_size),
            "edge_37": edge,
            "fg_core_pu_37": fg_core_pu.float(),
            "bg_core_pu_37": bg_core_pu.float(),
            "extent_band_37": extent_band.float(),
            "extent_candidate_37": extent_candidate,
            "valid_extent_37": valid_extent.float(),
            "unknown_37": unknown,
            "target_soft_37": target_soft,
            "weight_map_37": weight_map,
            "fg_core_pu_68": fg_core_pu_68,
            "bg_core_pu_68": bg_core_pu_68,
            "extent_candidate_68": extent_candidate_68,
            "unknown_68": unknown_68,
            "target_soft_68": target_soft_68,
            "weight_map_68": weight_map_68,
            "core_affinity_37": core_affinity,
            "residual_norm_37": residual_norm,
            "fg_score_norm_37": fg_score_norm,
            "fg_core_pu_area": float(fg_core_pu.float().mean().item()),
            "bg_core_pu_area": float(bg_core_pu.float().mean().item()),
            "extent_area": float((extent_candidate > 0.0).float().mean().item()),
            "unknown_area": float(unknown.mean().item()),
            "target_soft_area": float(target_soft.mean().item()),
            "weight_mean": float(weight_map.mean().item()),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _pu_v11_quantile(tensor, percentile):
    q = max(0.0, min(1.0, float(percentile) / 100.0))
    return torch.quantile(tensor.reshape(-1).float(), q)


def _pu_v11_construct_fg_core(p_base, fg_core, bg_core, fg_score, residual, evidence, params):
    fallback_reasons = []
    fg_candidate = torch.zeros_like(p_base, dtype=torch.bool)
    if bool(params.get("DABE_PU_V11_USE_V2_FG_CORE_AS_CANDIDATE", True)):
        fg_candidate |= fg_core > 0.5
    fg_candidate |= p_base >= float(params.get("DABE_PU_V11_FG_CORE_P_BASE_THRESH", 0.55))
    res_thr = _pu_v11_quantile(residual, params.get("DABE_PU_V11_FG_CORE_RESIDUAL_PERCENTILE", 70.0))
    score_thr = _pu_v11_quantile(fg_score, params.get("DABE_PU_V11_FG_CORE_FG_SCORE_PERCENTILE", 88.0))
    bgcore_max = float(params.get("DABE_PU_V11_FG_CORE_BGCORE_MAX", 0.0))
    fg_core_pu = (
        fg_candidate
        & (p_base >= float(params.get("DABE_PU_V11_FG_CORE_P_BASE_THRESH", 0.55)))
        & (evidence >= float(params.get("DABE_PU_V11_FG_CORE_EVIDENCE_MIN", 0.50)))
        & (residual >= res_thr)
        & (fg_score >= score_thr)
        & (bg_core <= bgcore_max)
    )
    fg_core_fallback = torch.zeros_like(fg_core_pu)
    fg_min = max(1, int(params.get("DABE_PU_V11_FG_CORE_MIN_PIXELS", 4)))
    if int(fg_core_pu.sum().item()) < fg_min:
        fallback_reasons.append("too_few_pu_v11_fg_core")
        score = p_base.reshape(-1).float().clone()
        invalid = ((bg_core > bgcore_max) | fg_core_pu).reshape(-1)
        score[invalid] = -1.0
        valid_count = int((score >= 0.0).sum().item())
        if valid_count > 0:
            topk = min(fg_min, valid_count)
            top_idx = torch.topk(score, k=topk, largest=True).indices
            fallback_flat = fg_core_fallback.reshape(-1)
            fallback_flat[top_idx] = True
            fg_core_fallback = fallback_flat.reshape_as(fg_core_fallback) & (bg_core <= bgcore_max) & ~fg_core_pu
    return fg_candidate.bool(), fg_core_pu.bool(), fg_core_fallback.bool(), fallback_reasons


def _pu_v11_construct_bg_core(p_base, fg_core_pu, fg_core_fallback, bg_core, bc_map, residual, params):
    grid = int(params["GRID"])
    fallback_reasons = []
    bg_core_pu = torch.zeros_like(p_base, dtype=torch.bool)
    if bool(params.get("DABE_PU_V11_BG_CORE_FROM_V2", True)):
        bg_core_pu |= bg_core > 0.5
    low_res_thr = _pu_v11_quantile(residual, params.get("DABE_PU_V11_BG_CORE_RESIDUAL_PERCENTILE", 25.0))
    p_base_max = float(params.get("DABE_PU_V11_BG_CORE_P_BASE_MAX", 0.15))
    bg_core_pu |= (
        (bc_map >= float(params.get("DABE_PU_V11_BG_CORE_BC_MIN", 0.65)))
        & (residual <= low_res_thr)
        & (p_base <= p_base_max)
    )
    border = _border_mask(grid, int(params.get("DABE_PU_V11_BG_CORE_BORDER_WIDTH", 1))).reshape(1, grid, grid)
    bg_core_pu |= border & (p_base <= p_base_max)
    bg_core_pu &= ~fg_core_pu & ~fg_core_fallback

    bg_min = max(1, int(params.get("DABE_PU_V11_BG_CORE_MIN_PIXELS", 16)))
    if int(bg_core_pu.sum().item()) < bg_min:
        fallback_reasons.append("too_few_pu_v11_bg_core")
        residual_norm = _minmax(residual.reshape(-1)).reshape_as(residual)
        bg_score = ((1.0 - residual_norm) * bc_map * (1.0 - p_base)).reshape(-1).float()
        bg_score[(fg_core_pu | fg_core_fallback).reshape(-1)] = -1.0
        valid_count = int((bg_score >= 0.0).sum().item())
        if valid_count > 0:
            topk = min(bg_min, valid_count)
            top_idx = torch.topk(bg_score, k=topk, largest=True).indices
            bg_flat = bg_core_pu.reshape(-1)
            bg_flat[top_idx] = True
            bg_core_pu = bg_flat.reshape_as(bg_core_pu) & ~fg_core_pu & ~fg_core_fallback
    return bg_core_pu.bool(), fallback_reasons


def _pu_v11_compactness_map(mask, fallback_mask):
    compactness = torch.zeros_like(mask, dtype=torch.float32)
    mask_np = mask.detach().cpu().bool().squeeze().numpy().astype(np.uint8)
    labels, num_labels = ndimage.label(mask_np, structure=np.ones((3, 3), dtype=np.uint8))
    for label_id in range(1, int(num_labels) + 1):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        area = float(ys.size)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        value = max(0.0, min(1.0, area / max(bbox_area, 1.0)))
        component = torch.from_numpy(labels == label_id).bool().reshape_as(mask)
        compactness = torch.where(component, torch.full_like(compactness, value), compactness)
    compactness = torch.where(fallback_mask.bool(), torch.full_like(compactness, 0.5), compactness)
    return compactness.clamp(0.0, 1.0)


def _pu_v11_cleanup_extent(extent_candidate, extent_score, p_base, params):
    if not bool(params.get("DABE_PU_V11_REMOVE_ISOLATED_EXTENT", True)):
        return extent_candidate.clamp(0.0, 1.0), torch.zeros_like(extent_candidate)
    grid = int(params["GRID"])
    extent_np = (extent_candidate.detach().cpu().float().squeeze().numpy() > 0.0).astype(np.uint8)
    labels, num_labels = ndimage.label(extent_np, structure=np.ones((3, 3), dtype=np.uint8))
    if int(num_labels) == 0:
        return torch.zeros_like(extent_candidate), torch.zeros_like(extent_candidate)

    base_mask = p_base > float(params.get("DABE_PU_V11_BASE_FG_THRESH", 0.35))
    base_band = _dilate_mask(base_mask, 1).reshape(grid, grid).bool()
    base_np = base_mask.detach().cpu().bool().squeeze().numpy()
    distance_to_base = ndimage.distance_transform_edt(~base_np)
    require_near_base = bool(params.get("DABE_PU_V11_EXTENT_REQUIRE_NEAR_BASE", True))
    max_dist = float(params.get("DABE_PU_V11_EXTENT_MAX_DIST_TO_BASE", 2))
    min_area = max(1, int(params.get("DABE_PU_V11_EXTENT_MIN_COMPONENT_AREA", 3)))

    extent_map = extent_candidate.reshape(grid, grid).float().clone()
    score_map = extent_score.reshape(grid, grid).float()
    removed_mask = torch.zeros((grid, grid), dtype=torch.bool)
    for label_id in range(1, int(num_labels) + 1):
        component = torch.from_numpy(labels == label_id).bool()
        area = int(component.sum().item())
        near_base = bool((component & base_band).any().item())
        if require_near_base:
            near_base = near_base and float(distance_to_base[labels == label_id].min()) <= max_dist
        if area < min_area or not near_base:
            extent_map[component] = 0.0
            removed_mask |= component

    max_area = max(0, int(round(float(params.get("DABE_PU_V11_EXTENT_MAX_AREA_RATIO", 0.30)) * grid * grid)))
    current_mask = extent_map > 0.0
    current_area = int(current_mask.sum().item())
    if current_area > max_area and max_area > 0:
        keep_flat = torch.zeros(grid * grid, dtype=torch.bool)
        score_flat = score_map.reshape(-1).clone()
        score_flat[~current_mask.reshape(-1)] = -1.0
        top_idx = torch.topk(score_flat, k=max_area, largest=True).indices
        keep_flat[top_idx] = True
        keep_mask = keep_flat.reshape(grid, grid)
        removed_mask |= current_mask & ~keep_mask
        extent_map = torch.where(keep_mask, extent_map, torch.zeros_like(extent_map))
    return extent_map.reshape(1, grid, grid).clamp(0.0, 1.0), removed_mask.reshape(1, grid, grid).float()


def _run_single_view_pu_v11(feature, rgb, params):
    grid = int(params["GRID"])
    loss_size = int(params["LOSS_SIZE"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    bc_map = base["bc_map_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_PU_V11_BASE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_PU_V11_BASE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)

    fg_candidate, fg_core_pu, fg_core_fallback, fg_fallback_reasons = _pu_v11_construct_fg_core(
        p_base,
        fg_core,
        bg_core,
        fg_score,
        residual,
        evidence,
        params,
    )
    bg_core_pu, bg_fallback_reasons = _pu_v11_construct_bg_core(
        p_base,
        fg_core_pu,
        fg_core_fallback,
        bg_core,
        bc_map,
        residual,
        params,
    )

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    fallback_reasons.extend(fg_fallback_reasons)
    fallback_reasons.extend(bg_fallback_reasons)

    core_mask = fg_core_pu | fg_core_fallback
    if bool(params.get("DABE_PU_V11_USE_CORE_AFFINITY", True)):
        core_affinity, too_few_core = _compute_core_affinity_max(feature, core_mask.float(), params)
        if too_few_core:
            core_affinity = torch.ones_like(p_base)
            fallback_reasons.append("too_few_core_for_affinity")
    else:
        core_affinity = torch.ones_like(p_base)

    residual_norm = _minmax(residual.reshape(-1)).reshape(1, grid, grid)
    fg_score_norm = _minmax(fg_score.reshape(-1)).reshape(1, grid, grid)
    evidence_norm = evidence.clamp(0.0, 1.0)
    compactness = _pu_v11_compactness_map(fg_core_pu, fg_core_fallback)
    if bool(params.get("DABE_PU_V11_USE_FG_CORE_RELIABILITY", True)):
        fg_reliability = (
            float(params.get("DABE_PU_V11_FG_REL_EVIDENCE_WEIGHT", 0.30)) * evidence_norm
            + float(params.get("DABE_PU_V11_FG_REL_RESIDUAL_WEIGHT", 0.30)) * residual_norm
            + float(params.get("DABE_PU_V11_FG_REL_FG_SCORE_WEIGHT", 0.25)) * fg_score_norm
            + float(params.get("DABE_PU_V11_FG_REL_COMPACTNESS_WEIGHT", 0.15)) * compactness
        ).clamp(0.0, 1.0)
    else:
        fg_reliability = torch.ones_like(p_base)
    fg_reliability = (fg_reliability * fg_core_pu.float()).clamp(0.0, 1.0)
    fg_weight_min = float(params.get("DABE_PU_V11_FG_CORE_WEIGHT_MIN", 0.45))
    fg_weight_max = float(params.get("DABE_PU_V11_FG_CORE_WEIGHT_MAX", 1.0))
    fg_core_weight = (fg_weight_min + (fg_weight_max - fg_weight_min) * fg_reliability).clamp(0.0, 1.0)
    fg_core_weight = fg_core_weight * fg_core_pu.float()
    fg_core_weight = torch.where(
        fg_core_fallback,
        torch.full_like(fg_core_weight, float(params.get("DABE_PU_V11_FG_CORE_FALLBACK_WEIGHT", 0.35))),
        fg_core_weight,
    )

    base_band_seed = p_base > float(params.get("DABE_PU_V11_EXTENT_BAND_BASE_THRESH", 0.18))
    if bool(params.get("DABE_PU_V11_EXTENT_USE_LOCAL_BAND", True)):
        extent_band = _dilate_mask(base_band_seed, int(params.get("DABE_PU_V11_EXTENT_BAND_RADIUS", 3)))
    else:
        extent_band = torch.ones_like(base_band_seed, dtype=torch.bool)
    if bool(params.get("DABE_PU_V11_EXTENT_EXCLUDE_BG_CORE", True)):
        extent_band = extent_band & ~bg_core_pu
    extent_candidate_region = extent_band & ~fg_core_pu & ~fg_core_fallback & ~bg_core_pu

    edge_allow = (1.0 - edge).clamp(0.0, 1.0)
    extent_score = (
        0.35 * residual_norm
        + 0.30 * fg_score_norm
        + 0.20 * evidence_norm
        + 0.15 * edge_allow
    ).clamp(0.0, 1.0)
    if bool(params.get("DABE_PU_V11_USE_CORE_AFFINITY", True)):
        core_weight = float(params.get("DABE_PU_V11_CORE_AFF_WEIGHT", 0.35))
        extent_score = ((1.0 - core_weight) * extent_score + core_weight * core_affinity).clamp(0.0, 1.0)

    res_extent_thr = _pu_v11_quantile(residual, params.get("DABE_PU_V11_EXTENT_RESIDUAL_PERCENTILE", 45.0))
    score_extent_thr = _pu_v11_quantile(fg_score, params.get("DABE_PU_V11_EXTENT_FG_SCORE_PERCENTILE", 45.0))
    valid_extent = (
        extent_candidate_region
        & (evidence >= float(params.get("DABE_PU_V11_EXTENT_EVIDENCE_MIN", 0.20)))
        & (residual >= res_extent_thr)
        & (fg_score >= score_extent_thr)
        & (edge <= float(params.get("DABE_PU_V11_EXTENT_EDGE_MAX", 0.75)))
    )
    if bool(params.get("DABE_PU_V11_USE_CORE_AFFINITY", True)):
        valid_extent &= core_affinity >= float(params.get("DABE_PU_V11_CORE_AFF_MIN", 0.45))
    extent_candidate = (extent_score * valid_extent.float()).clamp(0.0, 1.0)
    extent_candidate, extent_removed_mask = _pu_v11_cleanup_extent(extent_candidate, extent_score, p_base, params)

    extent_mask = extent_candidate > 0.0
    known_union = torch.clamp(
        fg_core_pu.float() + fg_core_fallback.float() + bg_core_pu.float() + extent_mask.float(),
        0.0,
        1.0,
    )
    unknown = (1.0 - known_union).clamp(0.0, 1.0)

    target_soft = torch.full_like(p_base, float(params.get("DABE_PU_V11_TARGET_UNKNOWN", 0.50)))
    target_soft = torch.where(bg_core_pu, torch.full_like(target_soft, float(params.get("DABE_PU_V11_BG_CORE_TARGET", 0.0))), target_soft)
    target_soft = torch.where(fg_core_pu, torch.full_like(target_soft, float(params.get("DABE_PU_V11_FG_CORE_TARGET", 1.0))), target_soft)
    target_soft = torch.where(
        fg_core_fallback,
        torch.full_like(target_soft, float(params.get("DABE_PU_V11_FG_CORE_FALLBACK_TARGET", 0.85))),
        target_soft,
    )
    extent_target = float(params.get("DABE_PU_V11_TARGET_EXTENT", 0.50))
    target_soft = torch.where(
        extent_mask & ~fg_core_pu & ~fg_core_fallback & ~bg_core_pu,
        torch.full_like(target_soft, extent_target),
        target_soft,
    ).clamp(0.0, 1.0)

    weight_map = torch.full_like(p_base, float(params.get("DABE_PU_V11_WEIGHT_UNKNOWN", 0.0)))
    if bool(params.get("DABE_PU_V11_USE_SOFT_BASE_WEIGHT", True)):
        weight_map = torch.maximum(weight_map, float(params.get("DABE_PU_V11_WEIGHT_SOFT_BASE", 0.05)) * p_base)
    extent_weight = float(params.get("DABE_PU_V11_WEIGHT_EXTENT", 0.08))
    extent_weight = max(
        float(params.get("DABE_PU_V11_WEIGHT_EXTENT_MIN", 0.05)),
        min(float(params.get("DABE_PU_V11_WEIGHT_EXTENT_MAX", 0.10)), extent_weight),
    )
    weight_map = torch.where(extent_mask, torch.full_like(weight_map, extent_weight), weight_map)
    weight_map = torch.where(bg_core_pu, torch.full_like(weight_map, float(params.get("DABE_PU_V11_BG_CORE_WEIGHT", 1.0))), weight_map)
    weight_map = torch.where(fg_core_pu, fg_core_weight, weight_map)
    weight_map = torch.where(
        fg_core_fallback,
        torch.full_like(weight_map, float(params.get("DABE_PU_V11_FG_CORE_FALLBACK_WEIGHT", 0.35))),
        weight_map,
    )
    weight_map = torch.where(unknown > 0.5, torch.full_like(weight_map, float(params.get("DABE_PU_V11_WEIGHT_UNKNOWN", 0.0))), weight_map)
    weight_map = weight_map.clamp(0.0, 1.0)

    fg_core_pu_68 = _resize_to_loss_nearest(fg_core_pu.float(), loss_size)
    fg_core_fallback_68 = _resize_to_loss_nearest(fg_core_fallback.float(), loss_size)
    bg_core_pu_68 = _resize_to_loss_nearest(bg_core_pu.float(), loss_size)
    extent_candidate_68 = _resize_to_loss(extent_candidate, loss_size)
    target_soft_68 = _resize_to_loss(target_soft, loss_size)
    weight_map_68 = _resize_to_loss(weight_map, loss_size)
    known_68 = torch.clamp(
        fg_core_pu_68 + fg_core_fallback_68 + bg_core_pu_68 + (extent_candidate_68 > 0.0).float(),
        0.0,
        1.0,
    )
    unknown_68 = (1.0 - known_68).clamp(0.0, 1.0)

    out = dict(base)
    out.update(
        {
            "p_dabe_37": target_soft,
            "p_dabe_68": target_soft_68,
            "p_base_37": p_base,
            "p_base_68": _resize_to_loss(p_base, loss_size),
            "edge_37": edge,
            "fg_core_candidate_37": fg_candidate.float(),
            "fg_core_pu_37": fg_core_pu.float(),
            "fg_core_fallback_37": fg_core_fallback.float(),
            "fg_reliability_37": fg_reliability,
            "fg_core_weight_37": fg_core_weight,
            "bg_core_pu_37": bg_core_pu.float(),
            "extent_band_37": extent_band.float(),
            "extent_candidate_37": extent_candidate,
            "extent_score_37": extent_score,
            "valid_extent_37": valid_extent.float(),
            "extent_removed_mask_37": extent_removed_mask,
            "unknown_37": unknown,
            "target_soft_37": target_soft,
            "weight_map_37": weight_map,
            "fg_core_pu_68": fg_core_pu_68,
            "fg_core_fallback_68": fg_core_fallback_68,
            "bg_core_pu_68": bg_core_pu_68,
            "extent_candidate_68": extent_candidate_68,
            "unknown_68": unknown_68,
            "target_soft_68": target_soft_68,
            "weight_map_68": weight_map_68,
            "core_affinity_37": core_affinity,
            "residual_norm_37": residual_norm,
            "fg_score_norm_37": fg_score_norm,
            "fg_core_pu_area": float(fg_core_pu.float().mean().item()),
            "fg_core_fallback_area": float(fg_core_fallback.float().mean().item()),
            "fg_core_reliability_mean": float(fg_reliability[fg_core_pu].mean().item()) if bool(fg_core_pu.any().item()) else 0.0,
            "bg_core_pu_area": float(bg_core_pu.float().mean().item()),
            "extent_area": float((extent_candidate > 0.0).float().mean().item()),
            "unknown_area": float(unknown.mean().item()),
            "target_soft_area": float(target_soft.mean().item()),
            "weight_mean": float(weight_map.mean().item()),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _run_single_view_gc(feature, rgb, params):
    grid = int(params["GRID"])
    base = _run_single_view_v2(feature, rgb, params)
    p_rw = base["p_rw_37"].float().clamp(0.0, 1.0)
    evidence = base["evidence_37"].float().clamp(0.0, 1.0)
    fg_core = base["fg_core_37"].float().clamp(0.0, 1.0)
    bg_core = base["bg_core_37"].float().clamp(0.0, 1.0)
    fg_score = base["fg_score_37"].float().clamp(0.0, 1.0)
    residual = base["residual_37"].float().clamp(0.0, 1.0)
    bc_map = base["bc_map_37"].float().clamp(0.0, 1.0)
    edge = _sobel_magnitude(rgb).reshape(1, grid, grid).float().clamp(0.0, 1.0)

    base_mode = str(params.get("DABE_GC_BASE_MODE", "strict")).lower()
    if base_mode != "strict":
        raise ValueError(f"Unsupported DABE_GC_BASE_MODE: {base_mode}")
    p_base = (p_rw * evidence).clamp(0.0, 1.0)
    fg_seed, bg_seed, seed_fallback_reasons = _gc_construct_seeds(
        p_base,
        fg_core,
        bg_core,
        fg_score,
        residual,
        evidence,
        bc_map,
        params,
    )
    neigh_idx, neigh_weight_norm, graph_degree = _gc_build_graph(
        feature,
        rgb,
        edge,
        fg_seed,
        bg_seed,
        bg_core,
        params,
    )
    p_gc = _gc_diffuse(p_base, fg_seed, bg_seed, neigh_idx, neigh_weight_norm, params)

    completion_mode = str(params.get("DABE_GC_COMPLETION_MODE", "positive_delta")).lower()
    if completion_mode != "positive_delta":
        raise ValueError(f"Unsupported DABE_GC_COMPLETION_MODE: {completion_mode}")
    delta_pos = (p_gc - p_base).clamp_min(float(params.get("DABE_GC_DELTA_CLAMP_MIN", 0.0)))
    gc_conf = (
        evidence.clamp(0.0, 1.0).pow(float(params.get("DABE_GC_CONF_EVIDENCE_POWER", 0.5)))
        * graph_degree.clamp(0.0, 1.0).pow(float(params.get("DABE_GC_CONF_AFF_POWER", 0.5)))
    ).clamp(0.0, 1.0)
    p_final = p_base + float(params.get("DABE_GC_BLEND_WEIGHT", 0.65)) * gc_conf * delta_pos
    p_final = torch.where(
        fg_seed,
        torch.maximum(p_final, torch.full_like(p_final, float(params.get("DABE_GC_FG_SEED_VALUE", 0.95)))),
        p_final,
    )
    p_final = torch.where(
        bg_seed,
        torch.minimum(p_final, torch.full_like(p_final, float(params.get("DABE_GC_BG_SEED_VALUE", 0.02)))),
        p_final,
    )
    p_final = torch.where(
        bg_core > 0.5,
        torch.minimum(p_final, torch.full_like(p_final, float(params.get("DABE_GC_BG_CORE_SUPPRESS", 0.03)))),
        p_final,
    ).clamp(0.0, 1.0)

    p_final, small_area_flag, large_area_flag, area_before, area_after, suppress_factor = _gc_area_control(
        p_final,
        fg_seed.float(),
        evidence,
        bg_core,
        params,
    )
    component = _component_protection_gc(
        p_final,
        feature,
        fg_seed.float(),
        bg_seed.float(),
        evidence,
        fg_score,
        edge,
        params,
    )
    p_final = component["p_refined_37"]
    uncertain = (1.0 - torch.clamp(fg_core + bg_core, 0.0, 1.0)).float()

    fallback_reasons = []
    if bool(base.get("fallback_flag", False)):
        fallback_reasons.append(str(base.get("fallback_reason", "fallback")))
    fallback_reasons.extend(seed_fallback_reasons)

    out = dict(base)
    out.update(
        {
            "p_dabe_37": p_final,
            "p_dabe_gc_37": p_final,
            "p_base_37": p_base,
            "fg_seed_37": fg_seed.float(),
            "bg_seed_37": bg_seed.float(),
            "fg_seed_area": float(fg_seed.float().mean().item()),
            "bg_seed_area": float(bg_seed.float().mean().item()),
            "graph_degree_37": graph_degree,
            "edge_37": edge,
            "p_gc_37": p_gc,
            "delta_pos_37": delta_pos,
            "gc_conf_37": gc_conf,
            "gc_conf_source": "graph_degree",
            "fg_core_37": fg_core,
            "bg_core_37": bg_core,
            "uncertain_37": uncertain,
            "component_keep_mask_37": component["component_keep_mask_37"],
            "component_weak_mask_37": component["component_weak_mask_37"],
            "component_removed_mask_37": component["component_removed_mask_37"],
            "num_components": int(component["num_components"]),
            "num_kept_components": int(component["num_kept_components"]),
            "num_weak_components": int(component["num_weak_components"]),
            "num_removed_components": int(component["num_removed_components"]),
            "base_area": float(p_base.mean().item()),
            "gc_area": float((p_gc > 0.5).float().mean().item()),
            "area_before_area_control": float(area_before),
            "area_after_area_control": float(area_after),
            "small_area_flag": bool(small_area_flag),
            "large_area_flag": bool(large_area_flag),
            "area_suppress_factor": float(suppress_factor),
            "fallback_flag": bool(fallback_reasons),
            "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        }
    )
    return out


def _run_single_view(feature, rgb, params):
    version = str(params.get("VERSION", "v2")).lower()
    if version == "v1":
        return _run_single_view_v1(feature, rgb, params)
    if version == "v2":
        return _run_single_view_v2(feature, rgb, params)
    if version == "v3":
        return _run_single_view_v3(feature, rgb, params)
    if version == "v3_1":
        return _run_single_view_v31(feature, rgb, params)
    if version == "gc":
        return _run_single_view_gc(feature, rgb, params)
    if version == "rac":
        return _run_single_view_rac(feature, rgb, params)
    if version == "rac_safe":
        return _run_single_view_rac_safe(feature, rgb, params)
    if version == "pu":
        return _run_single_view_pu(feature, rgb, params)
    if version == "pu_v11":
        return _run_single_view_pu_v11(feature, rgb, params)
    raise ValueError(f"Unsupported DABE VERSION: {version}")


def _resize_to_loss(tensor, loss_size):
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(int(loss_size), int(loss_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).contiguous()


def _resize_to_loss_nearest(tensor, loss_size):
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(int(loss_size), int(loss_size)),
        mode="nearest",
    ).squeeze(0).clamp(0.0, 1.0).contiguous()


def _count_components_tensor(tensor, threshold=0.5):
    array = (tensor.detach().cpu().float().squeeze().numpy() > float(threshold)).astype(np.uint8)
    _labels, num_labels = ndimage.label(array, structure=np.ones((3, 3), dtype=np.uint8))
    return int(num_labels)


def generate_dabe_pseudo(feature, image_path, fixed_pseudo_path=None, params=None, augs=None):
    """Generate training-free DABE pseudo labels from one cached DINO feature.

    fixed_pseudo_path is accepted only for diagnostic API compatibility and is
    intentionally not read by the DABE algorithm.
    """
    del fixed_pseudo_path
    params = _merge_params(params)
    grid = int(params["GRID"])
    loss_size = int(params["LOSS_SIZE"])
    feature = _validate_feature(feature, grid)
    rgb = _load_rgb_grid(image_path, grid)
    augs = _parse_augs(augs)
    version = str(params.get("VERSION", "v2")).lower()

    aligned_views = []
    aligned_map_keys = [
        "bc_map_37",
        "bg_anchor_37",
        "bg_core_37",
        "fg_score_37",
        "fg_core_37",
        "residual_37",
        "residual_pass1_37",
        "p_rw_37",
        "evidence_37",
    ]
    if version == "v3":
        aligned_map_keys.extend(
            [
                "p_rw_evid_37",
                "evidence_soft_37",
                "core_affinity_37",
                "expand_score_37",
                "p_expand_37",
                "uncertain_37",
                "component_keep_mask_37",
                "component_weak_mask_37",
                "p_dabe_v3_37",
            ]
        )
    elif version == "v3_1":
        aligned_map_keys.extend(
            [
                "p_dabe_v31_37",
                "p_base_37",
                "base_mask_37",
                "candidate_band_37",
                "core_affinity_37",
                "residual_norm_37",
                "expand_score_37",
                "valid_expand_37",
                "p_expand_37",
                "uncertain_37",
                "component_keep_mask_37",
                "component_weak_mask_37",
                "component_removed_mask_37",
            ]
        )
    elif version == "gc":
        aligned_map_keys.extend(
            [
                "p_dabe_gc_37",
                "p_base_37",
                "fg_seed_37",
                "bg_seed_37",
                "graph_degree_37",
                "edge_37",
                "p_gc_37",
                "delta_pos_37",
                "gc_conf_37",
                "uncertain_37",
                "component_keep_mask_37",
                "component_weak_mask_37",
                "component_removed_mask_37",
            ]
        )
    elif version == "rac":
        aligned_map_keys.extend(
            [
                "p_dabe_rac_37",
                "p_base_37",
                "fg_seed_37",
                "bg_seed_37",
                "region_id_map_37",
                "p_region_37",
                "strong_region_mask_37",
                "weak_region_mask_37",
                "rejected_region_mask_37",
                "delta_pos_37",
                "p_rac_before_area_37",
                "region_aff_degree_37",
                "edge_37",
                "core_affinity_37",
                "color_affinity_37",
                "residual_norm_37",
                "uncertain_37",
                "component_keep_mask_37",
                "component_weak_mask_37",
                "component_removed_mask_37",
            ]
        )
    elif version == "rac_safe":
        aligned_map_keys.extend(
            [
                "p_dabe_rac_safe_37",
                "p_base_37",
                "base_mask_37",
                "local_band_37",
                "fg_seed_37",
                "bg_seed_37",
                "region_id_map_37",
                "p_region_support_37",
                "kept_region_mask_37",
                "rejected_region_mask_37",
                "region_aff_degree_37",
                "edge_37",
                "core_affinity_37",
                "color_affinity_37",
                "residual_norm_37",
                "fg_score_norm_37",
                "aff_norm_37",
                "pixel_gate_37",
                "pixel_completion_37",
                "delta_pos_37",
                "p_safe_before_budget_37",
                "top_delta_mask_37",
                "delta_pos_budgeted_37",
                "p_safe_after_core_lock_37",
                "uncertain_37",
                "component_keep_mask_37",
                "component_weak_mask_37",
                "component_removed_mask_37",
            ]
        )
    elif version == "pu":
        aligned_map_keys.extend(
            [
                "p_base_37",
                "p_base_68",
                "edge_37",
                "fg_core_pu_37",
                "bg_core_pu_37",
                "extent_band_37",
                "extent_candidate_37",
                "valid_extent_37",
                "unknown_37",
                "target_soft_37",
                "weight_map_37",
                "fg_core_pu_68",
                "bg_core_pu_68",
                "extent_candidate_68",
                "unknown_68",
                "target_soft_68",
                "weight_map_68",
                "core_affinity_37",
                "residual_norm_37",
                "fg_score_norm_37",
            ]
        )
    elif version == "pu_v11":
        aligned_map_keys.extend(
            [
                "p_base_37",
                "p_base_68",
                "edge_37",
                "fg_core_candidate_37",
                "fg_core_pu_37",
                "fg_core_fallback_37",
                "fg_reliability_37",
                "fg_core_weight_37",
                "bg_core_pu_37",
                "extent_band_37",
                "extent_candidate_37",
                "extent_score_37",
                "valid_extent_37",
                "extent_removed_mask_37",
                "unknown_37",
                "target_soft_37",
                "weight_map_37",
                "fg_core_pu_68",
                "fg_core_fallback_68",
                "bg_core_pu_68",
                "extent_candidate_68",
                "unknown_68",
                "target_soft_68",
                "weight_map_68",
                "core_affinity_37",
                "residual_norm_37",
                "fg_score_norm_37",
            ]
        )
    aligned_maps = {key: [] for key in aligned_map_keys}
    fallback_reasons = []
    fallback_flag = False
    component_counts = []
    kept_counts = []
    weak_counts = []
    removed_counts = []
    small_area_flags = []
    large_area_flags = []
    area_before_values = []
    area_after_values = []
    base_area_values = []
    expand_area_values = []
    valid_expand_area_values = []
    adaptive_radius_values = []
    adaptive_expand_weight_values = []
    expand_thr_values = []
    fg_seed_area_values = []
    bg_seed_area_values = []
    gc_area_values = []
    region_area_values = []
    safe_area_values = []
    local_band_area_values = []
    kept_region_counts = []
    candidate_region_counts = []
    strong_region_counts = []
    weak_region_counts = []
    rejected_region_counts = []
    region_stats_views = []
    delta_budget_values = []
    max_final_area_values = []
    area_before_budget_values = []
    area_after_budget_values = []
    area_before_control_values = []
    area_after_control_values = []
    area_suppress_factor_values = []
    gc_conf_sources = []
    fg_core_pu_area_values = []
    fg_core_fallback_area_values = []
    fg_core_reliability_mean_values = []
    bg_core_pu_area_values = []
    extent_area_values = []
    unknown_area_values = []
    target_soft_area_values = []
    weight_mean_values = []

    for aug in augs:
        feature_aug = _apply_aug(feature, aug)
        rgb_aug = _apply_aug(rgb, aug)
        result = _run_single_view(feature_aug, rgb_aug, params)
        p_aligned = _apply_aug(result["p_dabe_37"], aug).float().clamp(0.0, 1.0)
        aligned_views.append(p_aligned)
        for key in aligned_maps:
            aligned_maps[key].append(_apply_aug(result[key], aug).float().clamp(0.0, 1.0))
        component_counts.append(int(result.get("num_components", 0)))
        if result["fallback_flag"]:
            fallback_flag = True
            fallback_reasons.append(str(result["fallback_reason"]))
        if version in {"v3", "v3_1", "gc", "rac", "rac_safe"}:
            kept_counts.append(int(result.get("num_kept_components", 0)))
            weak_counts.append(int(result.get("num_weak_components", 0)))
            removed_counts.append(int(result.get("num_removed_components", 0)))
        if version == "v3":
            small_area_flags.append(bool(result.get("small_area_flag", False)))
            large_area_flags.append(bool(result.get("large_area_flag", False)))
            area_before_values.append(float(result.get("area_before_area_prior", p_aligned.mean().item())))
            area_after_values.append(float(result.get("area_after_area_prior", p_aligned.mean().item())))
        elif version == "v3_1":
            base_area_values.append(float(result.get("base_area", 0.0)))
            expand_area_values.append(float(result.get("expand_area", 0.0)))
            valid_expand_area_values.append(float(result.get("valid_expand_area", 0.0)))
            adaptive_radius_values.append(float(result.get("adaptive_radius", 0.0)))
            adaptive_expand_weight_values.append(float(result.get("adaptive_expand_weight", 0.0)))
            expand_thr_values.append(float(result.get("expand_thr", 0.0)))
        elif version == "gc":
            base_area_values.append(float(result.get("base_area", 0.0)))
            fg_seed_area_values.append(float(result.get("fg_seed_area", 0.0)))
            bg_seed_area_values.append(float(result.get("bg_seed_area", 0.0)))
            gc_area_values.append(float(result.get("gc_area", 0.0)))
            area_before_control_values.append(float(result.get("area_before_area_control", p_aligned.mean().item())))
            area_after_control_values.append(float(result.get("area_after_area_control", p_aligned.mean().item())))
            area_suppress_factor_values.append(float(result.get("area_suppress_factor", 1.0)))
            small_area_flags.append(bool(result.get("small_area_flag", False)))
            large_area_flags.append(bool(result.get("large_area_flag", False)))
            gc_conf_sources.append(str(result.get("gc_conf_source", "graph_degree")))
        elif version == "rac":
            base_area_values.append(float(result.get("base_area", 0.0)))
            fg_seed_area_values.append(float(result.get("fg_seed_area", 0.0)))
            bg_seed_area_values.append(float(result.get("bg_seed_area", 0.0)))
            region_area_values.append(float(result.get("region_area", 0.0)))
            candidate_region_counts.append(int(result.get("num_candidate_regions", 0)))
            strong_region_counts.append(int(result.get("num_strong_regions", 0)))
            weak_region_counts.append(int(result.get("num_weak_regions", 0)))
            rejected_region_counts.append(int(result.get("num_rejected_regions", 0)))
            area_before_control_values.append(float(result.get("area_before_area_control", p_aligned.mean().item())))
            area_after_control_values.append(float(result.get("area_after_area_control", p_aligned.mean().item())))
            area_suppress_factor_values.append(float(result.get("area_suppress_factor", 1.0)))
            small_area_flags.append(bool(result.get("small_area_flag", False)))
            large_area_flags.append(bool(result.get("large_area_flag", False)))
            for stat in result.get("region_stats", []):
                row = dict(stat)
                row["view"] = aug
                region_stats_views.append(row)
        elif version == "rac_safe":
            base_area_values.append(float(result.get("base_area", 0.0)))
            safe_area_values.append(float(result.get("safe_area", 0.0)))
            local_band_area_values.append(float(result.get("local_band_area", 0.0)))
            fg_seed_area_values.append(float(result.get("fg_seed_area", 0.0)))
            bg_seed_area_values.append(float(result.get("bg_seed_area", 0.0)))
            region_area_values.append(float(result.get("region_area", 0.0)))
            candidate_region_counts.append(int(result.get("num_candidate_regions", 0)))
            kept_region_counts.append(int(result.get("num_kept_regions", 0)))
            rejected_region_counts.append(int(result.get("num_rejected_regions", 0)))
            delta_budget_values.append(float(result.get("delta_budget", 0.0)))
            max_final_area_values.append(float(result.get("max_final_area", 0.0)))
            area_before_budget_values.append(float(result.get("area_before_budget", p_aligned.mean().item())))
            area_after_budget_values.append(float(result.get("area_after_budget", p_aligned.mean().item())))
            small_area_flags.append(bool(result.get("small_area_flag", False)))
            large_area_flags.append(bool(result.get("large_area_flag", False)))
            for stat in result.get("region_stats", []):
                row = dict(stat)
                row["view"] = aug
                region_stats_views.append(row)
        elif version in {"pu", "pu_v11"}:
            base_area_values.append(float(result.get("p_base_37", p_aligned).mean().item()))
            fg_core_pu_area_values.append(float(result.get("fg_core_pu_area", 0.0)))
            fg_core_fallback_area_values.append(float(result.get("fg_core_fallback_area", 0.0)))
            fg_core_reliability_mean_values.append(float(result.get("fg_core_reliability_mean", 0.0)))
            bg_core_pu_area_values.append(float(result.get("bg_core_pu_area", 0.0)))
            extent_area_values.append(float(result.get("extent_area", 0.0)))
            unknown_area_values.append(float(result.get("unknown_area", 0.0)))
            target_soft_area_values.append(float(result.get("target_soft_area", p_aligned.mean().item())))
            weight_mean_values.append(float(result.get("weight_mean", 0.0)))

    view_stack = torch.stack(aligned_views, dim=0)
    p_dabe_37 = view_stack.mean(dim=0).clamp(0.0, 1.0).contiguous()
    if view_stack.shape[0] == 1:
        view_agreement = torch.ones_like(p_dabe_37)
    elif version in {"v3", "v3_1", "rac", "rac_safe"}:
        view_agreement = (1.0 - view_stack.std(dim=0, unbiased=False)).clamp(0.0, 1.0)
    else:
        var = view_stack.var(dim=0, unbiased=False)
        view_agreement = (1.0 - var / float(params["VIEW_AGREEMENT_VAR_DENOM"])).clamp(0.0, 1.0)

    output = {
        "p_dabe_37": p_dabe_37,
        "p_dabe_68": _resize_to_loss(p_dabe_37, loss_size),
        "view_agreement_37": view_agreement.contiguous(),
        "dabe_version": version,
        "fallback_flag": bool(fallback_flag),
        "fallback_reason": "|".join(sorted(set(fallback_reasons))) if fallback_reasons else "none",
        "params": dict(params),
        "augs": list(augs),
        "num_views": int(len(augs)),
    }
    for key, values in aligned_maps.items():
        output[key] = torch.stack(values, dim=0).mean(dim=0).clamp(0.0, 1.0).contiguous()
    if version == "gc":
        agreement = output["view_agreement_37"].pow(float(params.get("DABE_GC_VIEW_AGREEMENT_POWER", 1.0)))
        output["p_dabe_37"] = (
            output["p_base_37"] + agreement * (output["p_dabe_37"] - output["p_base_37"])
        ).clamp(0.0, 1.0).contiguous()
        output["p_dabe_68"] = _resize_to_loss(output["p_dabe_37"], loss_size)
    elif version == "rac":
        agreement = output["view_agreement_37"].pow(float(params.get("DABE_RAC_VIEW_AGREEMENT_POWER", 1.0)))
        output["p_dabe_37"] = (
            output["p_base_37"] + agreement * (output["p_dabe_37"] - output["p_base_37"])
        ).clamp(0.0, 1.0).contiguous()
        output["p_dabe_68"] = _resize_to_loss(output["p_dabe_37"], loss_size)
    elif version == "rac_safe":
        agreement = output["view_agreement_37"].pow(float(params.get("DABE_RAC_SAFE_VIEW_AGREEMENT_POWER", 1.0)))
        output["p_dabe_37"] = (
            output["p_base_37"] + agreement * (output["p_dabe_37"] - output["p_base_37"])
        ).clamp(0.0, 1.0).contiguous()
        output["p_dabe_68"] = _resize_to_loss(output["p_dabe_37"], loss_size)
    if version == "v3":
        output["p_dabe_v3_37"] = output["p_dabe_37"]
        output["p_dabe_v3_68"] = output["p_dabe_68"]
        output["fg_core_68"] = _resize_to_loss_nearest(output["fg_core_37"], loss_size)
        output["bg_core_68"] = _resize_to_loss_nearest(output["bg_core_37"], loss_size)
        output["uncertain_37"] = (1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).clamp(0.0, 1.0)
        output["uncertain_68"] = _resize_to_loss_nearest(output["uncertain_37"], loss_size)
        output["num_kept_components"] = int(round(float(np.mean(kept_counts)))) if kept_counts else 0
        output["num_weak_components"] = int(round(float(np.mean(weak_counts)))) if weak_counts else 0
        output["num_removed_components"] = int(round(float(np.mean(removed_counts)))) if removed_counts else 0
        output["small_area_flag"] = bool(any(small_area_flags))
        output["large_area_flag"] = bool(any(large_area_flags))
        output["area_before_area_prior"] = float(np.mean(area_before_values)) if area_before_values else float(p_dabe_37.mean().item())
        output["area_after_area_prior"] = float(np.mean(area_after_values)) if area_after_values else float(p_dabe_37.mean().item())
    elif version == "v3_1":
        output["p_dabe_v31_37"] = output["p_dabe_37"]
        output["p_dabe_v31_68"] = output["p_dabe_68"]
        output["fg_core_68"] = _resize_to_loss_nearest(output["fg_core_37"], loss_size)
        output["bg_core_68"] = _resize_to_loss_nearest(output["bg_core_37"], loss_size)
        output["uncertain_37"] = (1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).clamp(0.0, 1.0)
        output["uncertain_68"] = _resize_to_loss_nearest(output["uncertain_37"], loss_size)
        output["num_kept_components"] = int(round(float(np.mean(kept_counts)))) if kept_counts else 0
        output["num_weak_components"] = int(round(float(np.mean(weak_counts)))) if weak_counts else 0
        output["num_removed_components"] = int(round(float(np.mean(removed_counts)))) if removed_counts else 0
        output["base_area"] = float(np.mean(base_area_values)) if base_area_values else 0.0
        output["expand_area"] = float(np.mean(expand_area_values)) if expand_area_values else 0.0
        output["valid_expand_area"] = float(np.mean(valid_expand_area_values)) if valid_expand_area_values else 0.0
        output["adaptive_radius"] = int(round(float(np.mean(adaptive_radius_values)))) if adaptive_radius_values else 0
        output["adaptive_expand_weight"] = float(np.mean(adaptive_expand_weight_values)) if adaptive_expand_weight_values else 0.0
        output["expand_thr"] = float(np.mean(expand_thr_values)) if expand_thr_values else 0.0
    elif version == "gc":
        output["p_dabe_gc_37"] = output["p_dabe_37"]
        output["p_dabe_gc_68"] = output["p_dabe_68"]
        output["fg_core_68"] = _resize_to_loss_nearest(output["fg_core_37"], loss_size)
        output["bg_core_68"] = _resize_to_loss_nearest(output["bg_core_37"], loss_size)
        output["uncertain_37"] = (1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).clamp(0.0, 1.0)
        output["uncertain_68"] = _resize_to_loss_nearest(output["uncertain_37"], loss_size)
        output["num_kept_components"] = int(round(float(np.mean(kept_counts)))) if kept_counts else 0
        output["num_weak_components"] = int(round(float(np.mean(weak_counts)))) if weak_counts else 0
        output["num_removed_components"] = int(round(float(np.mean(removed_counts)))) if removed_counts else 0
        output["base_area"] = float(np.mean(base_area_values)) if base_area_values else 0.0
        output["fg_seed_area"] = float(np.mean(fg_seed_area_values)) if fg_seed_area_values else 0.0
        output["bg_seed_area"] = float(np.mean(bg_seed_area_values)) if bg_seed_area_values else 0.0
        output["gc_area"] = float(np.mean(gc_area_values)) if gc_area_values else 0.0
        output["area_before_area_control"] = float(np.mean(area_before_control_values)) if area_before_control_values else float(p_dabe_37.mean().item())
        output["area_after_area_control"] = float(np.mean(area_after_control_values)) if area_after_control_values else float(p_dabe_37.mean().item())
        output["area_suppress_factor"] = float(np.mean(area_suppress_factor_values)) if area_suppress_factor_values else 1.0
        output["gc_conf_source"] = "|".join(sorted(set(gc_conf_sources))) if gc_conf_sources else "graph_degree"
        output["small_area_flag"] = bool(any(small_area_flags))
        output["large_area_flag"] = bool(any(large_area_flags))
    elif version == "rac":
        output["p_dabe_rac_37"] = output["p_dabe_37"]
        output["p_dabe_rac_68"] = output["p_dabe_68"]
        output["fg_core_68"] = _resize_to_loss_nearest(output["fg_core_37"], loss_size)
        output["bg_core_68"] = _resize_to_loss_nearest(output["bg_core_37"], loss_size)
        output["uncertain_37"] = (1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).clamp(0.0, 1.0)
        output["uncertain_68"] = _resize_to_loss_nearest(output["uncertain_37"], loss_size)
        output["num_kept_components"] = int(round(float(np.mean(kept_counts)))) if kept_counts else 0
        output["num_weak_components"] = int(round(float(np.mean(weak_counts)))) if weak_counts else 0
        output["num_removed_components"] = int(round(float(np.mean(removed_counts)))) if removed_counts else 0
        output["base_area"] = float(np.mean(base_area_values)) if base_area_values else 0.0
        output["region_area"] = float(np.mean(region_area_values)) if region_area_values else 0.0
        output["fg_seed_area"] = float(np.mean(fg_seed_area_values)) if fg_seed_area_values else 0.0
        output["bg_seed_area"] = float(np.mean(bg_seed_area_values)) if bg_seed_area_values else 0.0
        output["num_candidate_regions"] = int(round(float(np.mean(candidate_region_counts)))) if candidate_region_counts else 0
        output["num_strong_regions"] = int(round(float(np.mean(strong_region_counts)))) if strong_region_counts else 0
        output["num_weak_regions"] = int(round(float(np.mean(weak_region_counts)))) if weak_region_counts else 0
        output["num_rejected_regions"] = int(round(float(np.mean(rejected_region_counts)))) if rejected_region_counts else 0
        output["region_stats"] = region_stats_views
        output["area_before_area_control"] = float(np.mean(area_before_control_values)) if area_before_control_values else float(p_dabe_37.mean().item())
        output["area_after_area_control"] = float(np.mean(area_after_control_values)) if area_after_control_values else float(p_dabe_37.mean().item())
        output["area_suppress_factor"] = float(np.mean(area_suppress_factor_values)) if area_suppress_factor_values else 1.0
        output["small_area_flag"] = bool(any(small_area_flags))
        output["large_area_flag"] = bool(any(large_area_flags))
    elif version == "rac_safe":
        output["p_dabe_rac_safe_37"] = output["p_dabe_37"]
        output["p_dabe_rac_safe_68"] = output["p_dabe_68"]
        output["fg_core_68"] = _resize_to_loss_nearest(output["fg_core_37"], loss_size)
        output["bg_core_68"] = _resize_to_loss_nearest(output["bg_core_37"], loss_size)
        output["uncertain_37"] = (1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).clamp(0.0, 1.0)
        output["uncertain_68"] = _resize_to_loss_nearest(output["uncertain_37"], loss_size)
        output["num_kept_components"] = int(round(float(np.mean(kept_counts)))) if kept_counts else 0
        output["num_weak_components"] = int(round(float(np.mean(weak_counts)))) if weak_counts else 0
        output["num_removed_components"] = int(round(float(np.mean(removed_counts)))) if removed_counts else 0
        output["base_area"] = float(np.mean(base_area_values)) if base_area_values else 0.0
        output["safe_area"] = float(output["p_dabe_37"].mean().item())
        output["local_band_area"] = float(np.mean(local_band_area_values)) if local_band_area_values else 0.0
        output["region_area"] = float(np.mean(region_area_values)) if region_area_values else 0.0
        output["fg_seed_area"] = float(np.mean(fg_seed_area_values)) if fg_seed_area_values else 0.0
        output["bg_seed_area"] = float(np.mean(bg_seed_area_values)) if bg_seed_area_values else 0.0
        output["num_candidate_regions"] = int(round(float(np.mean(candidate_region_counts)))) if candidate_region_counts else 0
        output["num_kept_regions"] = int(round(float(np.mean(kept_region_counts)))) if kept_region_counts else 0
        output["num_rejected_regions"] = int(round(float(np.mean(rejected_region_counts)))) if rejected_region_counts else 0
        output["region_stats"] = region_stats_views
        output["delta_budget"] = float(np.mean(delta_budget_values)) if delta_budget_values else 0.0
        output["max_final_area"] = float(np.mean(max_final_area_values)) if max_final_area_values else 0.0
        output["area_before_budget"] = float(np.mean(area_before_budget_values)) if area_before_budget_values else float(p_dabe_37.mean().item())
        output["area_after_budget"] = float(np.mean(area_after_budget_values)) if area_after_budget_values else float(p_dabe_37.mean().item())
        output["small_area_flag"] = bool(any(small_area_flags))
        output["large_area_flag"] = bool(any(large_area_flags))
    elif version in {"pu", "pu_v11"}:
        output["target_soft_37"] = output["p_dabe_37"].clamp(0.0, 1.0)
        output["target_soft_68"] = output["p_dabe_68"].clamp(0.0, 1.0)
        output["p_base_68"] = _resize_to_loss(output["p_base_37"], loss_size)
        output["fg_core_pu_68"] = _resize_to_loss_nearest(output["fg_core_pu_37"], loss_size)
        if version == "pu_v11":
            output["fg_core_fallback_68"] = _resize_to_loss_nearest(output["fg_core_fallback_37"], loss_size)
        output["bg_core_pu_68"] = _resize_to_loss_nearest(output["bg_core_pu_37"], loss_size)
        output["extent_candidate_68"] = _resize_to_loss(output["extent_candidate_37"], loss_size)
        output["weight_map_68"] = _resize_to_loss(output["weight_map_37"], loss_size)
        fg_known_68 = output["fg_core_pu_68"]
        if version == "pu_v11":
            fg_known_68 = torch.clamp(fg_known_68 + output["fg_core_fallback_68"], 0.0, 1.0)
        known_68 = torch.clamp(
            fg_known_68 + output["bg_core_pu_68"] + (output["extent_candidate_68"] > 0.0).float(),
            0.0,
            1.0,
        )
        output["unknown_68"] = (1.0 - known_68).clamp(0.0, 1.0)
        output["fg_core_pu_area"] = float(np.mean(fg_core_pu_area_values)) if fg_core_pu_area_values else float(output["fg_core_pu_37"].mean().item())
        output["fg_core_fallback_area"] = float(np.mean(fg_core_fallback_area_values)) if fg_core_fallback_area_values else float(output.get("fg_core_fallback_37", torch.zeros_like(output["fg_core_pu_37"])).mean().item())
        output["fg_core_reliability_mean"] = float(np.mean(fg_core_reliability_mean_values)) if fg_core_reliability_mean_values else 0.0
        output["bg_core_pu_area"] = float(np.mean(bg_core_pu_area_values)) if bg_core_pu_area_values else float(output["bg_core_pu_37"].mean().item())
        output["extent_area"] = float(np.mean(extent_area_values)) if extent_area_values else float((output["extent_candidate_37"] > 0.0).float().mean().item())
        output["unknown_area"] = float(np.mean(unknown_area_values)) if unknown_area_values else float(output["unknown_37"].mean().item())
        output["target_soft_area"] = float(np.mean(target_soft_area_values)) if target_soft_area_values else float(output["target_soft_37"].mean().item())
        output["weight_mean"] = float(np.mean(weight_mean_values)) if weight_mean_values else float(output["weight_map_37"].mean().item())
        output["base_area"] = float(output["p_base_37"].mean().item())
        output["large_area_flag"] = bool(output["target_soft_area"] > float(params.get("MAX_COMP_AREA", 0.60)))
        output["small_area_flag"] = False

    output["area"] = float((output["p_dabe_68"] > 0.5).float().mean().item())
    if version == "v3":
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output.get("large_area_flag", False) or output["area"] > float(params.get("DABE_V3_AREA_PRIOR_HIGH", 0.22)))
        output["small_area_flag"] = bool(output.get("small_area_flag", False) or output["area"] < float(params.get("DABE_V3_AREA_PRIOR_LOW", 0.10)))
    elif version == "v3_1":
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output["area"] > float(params.get("MAX_COMP_AREA", 0.60)))
        output["small_area_flag"] = False
    elif version == "gc":
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output.get("large_area_flag", False) or output["area"] > float(params.get("DABE_GC_AREA_HIGH", 0.22)))
        output["small_area_flag"] = bool(output.get("small_area_flag", False) or output["area"] < float(params.get("DABE_GC_AREA_LOW", 0.08)))
    elif version == "rac":
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output.get("large_area_flag", False) or output["area"] > float(params.get("DABE_RAC_AREA_HIGH", 0.24)))
        output["small_area_flag"] = bool(output.get("small_area_flag", False) or output["area"] < float(params.get("DABE_RAC_AREA_LOW", 0.10)))
    elif version == "rac_safe":
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output.get("large_area_flag", False) or output["area"] > float(params.get("DABE_RAC_SAFE_TARGET_AREA_HIGH", 0.155)))
        output["small_area_flag"] = bool(output.get("small_area_flag", False) or output["area"] < float(params.get("DABE_RAC_SAFE_TARGET_AREA_LOW", 0.13)))
    elif version in {"pu", "pu_v11"}:
        output["num_components"] = int(_count_components_tensor(output["target_soft_68"], threshold=0.5))
        output["large_area_flag"] = bool(output.get("large_area_flag", False))
        output["small_area_flag"] = bool(output.get("small_area_flag", False))
    else:
        output["num_components"] = int(_count_components_tensor(output["p_dabe_68"], threshold=0.5))
        output["large_area_flag"] = bool(output["area"] > float(params.get("MAX_COMP_AREA", 0.60)))
    output["fg_core_area"] = float(output["fg_core_37"].mean().item())
    output["bg_core_area"] = float(output["bg_core_37"].mean().item())
    output["uncertain_area"] = float(output.get("uncertain_37", 1.0 - torch.clamp(output["fg_core_37"] + output["bg_core_37"], 0.0, 1.0)).mean().item())
    output["residual_mean"] = float(output["residual_37"].mean().item())
    output["fg_score_mean"] = float(output["fg_score_37"].mean().item())
    output["bc_mean"] = float(output["bc_map_37"].mean().item())
    return output
