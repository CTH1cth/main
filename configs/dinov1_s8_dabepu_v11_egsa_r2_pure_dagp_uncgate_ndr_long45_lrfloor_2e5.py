from configs.dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_egsa_r2_pure_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5"
)

SOURCE_ARBITER_VERSION = "egsa_r2_pure_delayed_cv_v1"
SOURCE_ARBITER_MODE = "pure_loss_space"
SOURCE_ARBITER_USE_ECST_TEACHER_MAP = False
SOURCE_ARBITER_USE_STRUCTURED_EVIDENCE = True
SOURCE_ARBITER_ECST_AUDIT_ONLY = True

SOURCE_ARBITER_DABE_SOURCE = "dabe_pu_weighted_soft"
SOURCE_ARBITER_TEACHER_SOURCE = "ema_binary_raw_bce"
SOURCE_ARBITER_COMPUTE_R1_SHADOW_LOSS = True
SOURCE_ARBITER_LOG_PLAIN_VS_ECST_TEACHER = True
SOURCE_ARBITER_LOG_SIGNED_CORRECTION = True
SOURCE_ARBITER_LOG_UTILITY_CLASS_BALANCE = True

# ECST remains an evidence provider and an audit-only R1 shadow in R2.
ECST_APPLY_TO_FINAL = False
ECST_APPLY_TO_COARSE_AUX = False
ECST_APPLY_TO_BASE_AUX = False
