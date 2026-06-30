from configs.dinov1_s8_despl_only_sap_rcim_adapt import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_sap_caff_min"

SAP_RCIM_MODE = "caff_min"

SAP_RCIM_USE_CAFF = True
SAP_RCIM_USE_FUSION = False
SAP_RCIM_USE_RECEPTIVE_CONV = False
SAP_RCIM_USE_GAP_GUIDE = False

USE_BASE_AUX_LOSS = True
LAMBDA_BASE_AUX = 0.3
LAMBDA_BASE_AUX_AFTER_RESET = 0.1
