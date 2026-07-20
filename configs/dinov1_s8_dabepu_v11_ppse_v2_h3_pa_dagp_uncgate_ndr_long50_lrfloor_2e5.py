from configs.dinov1_s8_dabepu_v11_pssf_h3_dagp_uncgate_ndr_long50_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_ppse_v2_h3_pa_dagp_uncgate_ndr_"
    "long50_lrfloor_2e5"
)

SUPERVISION_MODE = "ppse_v2_state"
TEACHER_FUSION_MODE = "ppse_v2_state"

USE_PSSF = True
PPSE_VERSION = "horizon_normalized_prior_anchored_v2"
PSSF_VERSION = "horizon_retention_actor_learner_v2"
PSSF_SOURCE_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_pssf_h3_dagp_uncgate_ndr_"
    "long50_lrfloor_2e5.py"
)

PSSF_HORIZON = 3
PSSF_HISTORY_WINDOW = 3

PPSE_STATE_STEP_MODE = "inverse_horizon"
PPSE_STATE_STEP = 1.0 / PSSF_HORIZON

PPSE_USE_PRIOR_ANCHOR = True
PPSE_PRIOR_SOURCE = "dabe_pu_target_soft_68"

PPSE_USE_ACTOR_LEARNER = True
PPSE_ACTOR_SYNC_MODE = "epoch_start_hard_copy"
PPSE_ACTOR_TRAINABLE = False
PPSE_ACTOR_UPDATE_WITHIN_EPOCH = False

PSSF_INIT_RETENTION = 0.03
PSSF_RETENTION_TARGET = "future_binary_innovation_retention"
PSSF_TEACHER_OBSERVATION = "binary_68"

PSSF_KEEP_STATE_ACROSS_RESET = True
PSSF_CLEAR_HISTORY_AFTER_RESET = True
PSSF_KEEP_LEARNER_ACROSS_RESET = True
PSSF_KEEP_ACTOR_ACROSS_RESET = True
PSSF_KEEP_OPTIMIZER_ACROSS_RESET = True

PSSF_RUNTIME_FILENAME = "ppse_v2_runtime_latest.pt"
