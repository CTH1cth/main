from configs.dinov1_s8_despl_only_sap_rcim_adapt import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_sap_rcim_full_longwarm"

MAX_EPOCH = 40
FINETUNE_RESET_EPOCH = 31

TEACHER_FUSION_MODE = "linear_to_095_before_reset"
TEACHER_FUSION_MAX_WEIGHT = 0.95

COMPLEX_HEAD_LR_POLICY = "hold_cosine_before_reset"
COMPLEX_HEAD_PRE_RESET_LR = 3e-4
COMPLEX_HEAD_LR_HOLD_EPOCHS = 10
COMPLEX_HEAD_LR_MIN = 3e-5

COMPLEX_HEAD_POST_RESET_LR = 6e-4
COMPLEX_HEAD_POST_RESET_SCHEDULER = "original_iter_steplr"

SAP_RCIM_MODE = "full"
SAP_RCIM_USE_CAFF = True
SAP_RCIM_USE_FUSION = True
SAP_RCIM_USE_RECEPTIVE_CONV = True
SAP_RCIM_USE_GAP_GUIDE = True
