"""Contract for the Hard-R1, linear-head, pure-Student experiment.

This module only audits resolved Python configurations.  It never constructs a
dataset/model and never starts training or evaluation.
"""

import math
from pathlib import Path

from common.dabev2hard_noecst_control import resolved_control_values
from common.utils import load_config


MAIN_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
DAGP_NDR_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
CVBR_V1_SECOND_RING_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_cvbrv1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
RPR_P1_SECOND_RING_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
RPR_P1_SECOND_RING_DAGP_NDR_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_"
    "dagp_uncgate_ndr_staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_ABSMM_T063_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t063_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_ABSMM_T050_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t050_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_ABSMM_T060_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t060_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP512C_T060_NATIVE64_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp512c_t060_native64_hard_"
    "linear_staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_ABSMM_T058_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_R32_ABSMM_T058_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_absmm_t058_hard_linear.py"
)
GBSP_ABSMM_T058_DAGP_ONLY_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_dagp_only_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_ABSMM_T058_NDR_ONLY_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_ndr_only_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GBSP_CF_BRC_HC_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_cf_brc_hc_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
GT68_LINEAR_CACHED_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gt68_linear_cached_b16.py"
)
DBA_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_r1hard_dba_staticonly_purestudent_"
    "long45_lrfloor_2e5.py"
)
LCIC_CONFIG_PATHS = {
    "a_linear": MAIN_ROOT / "configs/dinov1_s8_gbsp_lcic_a_linear.py",
    "b_consensus": MAIN_ROOT / "configs/dinov1_s8_gbsp_lcic_b_consensus.py",
    "c_innovation": MAIN_ROOT / "configs/dinov1_s8_gbsp_lcic_c_innovation.py",
    "d_full": MAIN_ROOT / "configs/dinov1_s8_gbsp_lcic_d_full.py",
    "e_dwlite": MAIN_ROOT / "configs/dinov1_s8_gbsp_lcic_e_dwlite.py",
}
LCIC_DATASET_THRESHOLD_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_camo058_cod10k066.py"
)
LCIC_DATASET_LOSS_WEIGHT_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_camow15.py"
)
LCIC_LR_STEP_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_lrstep50.py"
)
LCIC_SOFT_DICE_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_lcic_d_full_dice005.py"
)
LCIC_R32_D_FULL_SOFT_DICE_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005.py"
)
LCIC_R32_D_FULL_SOFT_DICE_DATASET_THRESHOLD_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_"
    "camo050_cod10k058.py"
)
LCIC_R32_D_FULL_SOFT_DICE_T050_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050.py"
)
LCIC_R32_D_FULL_SOFT_DICE_010_T050_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice010_t050.py"
)
LCIC_R32_D_FULL_SOFT_DICE_010_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice010_t050_seed2027.py"
)
LCIC_R32_D_FULL_SOFT_DICE_T050_LR0003_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_lr0003.py"
)
LCIC_R32_D_FULL_SOFT_DICE_T050_BW1P30_SIGMAF085_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "bw1p30_sigmaf085.py"
)
LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS = {
    2026: MAIN_ROOT / (
        "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2026.py"
    ),
    2027: MAIN_ROOT / (
        "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027.py"
    ),
}
LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_STRUCTONLY_E3_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
    "seed2027_structonly_e3.py"
)
LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_CONSENSUS_GAIN3_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
        "seed2027_consensusgain3.py"
    )
)
LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_ADAPTIVE_GATE_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_"
        "seed2027_adaptivegate.py"
    )
)
LCIC_R32_D_FULL_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_ssboc_dice005_t050_"
    "seed2027.py"
)
LCIC_R32_RGBMV_HFLIPMEAN_SSBOC_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_rgbmv_hflipmean_lcic_ssboc.py"
)
GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_seed2027.py"
)
GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
        "seed2027_lr0003.py"
    )
)
GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_DICE005_T050_SEED2027_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
        "seed2027_lr1e4_floor1e5.py"
    )
)
GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_DICE005_T050_SEED2027_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_dwlite16_ssboc_dice005_t050_"
        "seed2027_lr1e4_floor1e5_linear45.py"
    )
)
GBSP_R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_DICE005_T050_SEED2027_CONFIG_PATH = (
    MAIN_ROOT
    / (
        "configs/dinov1_s8_gbsp_r32_conv3x3_c16_nossboc_"
        "step2floor_dice005_t050_seed2027.py"
    )
)
LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_t050_nodice_seed2027.py"
)
GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027.py"
)
GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_l2s1059_dice005_t050_seed2027.py"
)
GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_confbce_dice005_t050_seed2027.py"
)
GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_ssboc_dice005_t050_seed2027.py"
)
GBSP_R32_LINEAR_RGBMV_HFLIPMEAN_SSBOC_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_rgbmv_hflipmean_1x1_ssboc.py"
)
GBSP_R16_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r16_linear_dice005_t050_seed2027.py"
)
GBSP_R32_LINEAR_DICE005_T054_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_dice005_t054_seed2027.py"
)
GBSP_R32_LINEAR_NODICE_T050_SEED2027_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_linear_nodice_t050_seed2027.py"
)

EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
DAGP_NDR_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
CVBR_V1_SECOND_RING_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_cvbrv1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
RPR_P1_SECOND_RING_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
RPR_P1_SECOND_RING_DAGP_NDR_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_rprp1_secondring_hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_ABSMM_T063_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t063_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_ABSMM_T050_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t050_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_ABSMM_T060_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t060_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP512C_T060_NATIVE64_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp512c_t060_native64_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_ABSMM_T058_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_R32_ABSMM_T058_EXP_NAME = "25-gbsp-r32-absmm-t058-1×1"
GBSP_R32_LINEAR_DICE005_T050_SEED2027_EXP_NAME = (
    "33-gbsp-r32-1x1-dice005-t050-seed2027"
)
GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_EXP_NAME = (
    "33-gbsp-r32-1x1-l2s1059-dice005-t050-seed2027"
)
GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_EXP_NAME = (
    "34-gbsp-r32-1x1-confbce-f050-g1-dice005-t050-seed2027"
)
GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_EXP_NAME = (
    "34-gbsp-r32-1x1-ssboc-dice005-t050-seed2027"
)
GBSP_R16_LINEAR_DICE005_T050_SEED2027_EXP_NAME = (
    "33-gbsp-r16-1x1-dice005-t050-seed2027"
)
GBSP_R32_LINEAR_DICE005_T054_SEED2027_EXP_NAME = (
    "33-gbsp-r32-1x1-dice005-t054-seed2027"
)
GBSP_R32_LINEAR_NODICE_T050_SEED2027_EXP_NAME = (
    "33-gbsp-r32-1x1-nodice-t050-seed2027"
)
GBSP_ABSMM_T058_DAGP_ONLY_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_dagp_only_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_ABSMM_T058_NDR_ONLY_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_ndr_only_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GBSP_CF_BRC_HC_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_cf_brc_hc_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
GT68_LINEAR_CACHED_EXP_NAME = "dinov1_s8_gt68_linear_cached_b16"
DBA_EXP_NAME = (
    "dinov1_s8_r1hard_dba_staticonly_purestudent_long45_lrfloor_2e5"
)
LCIC_VARIANT_SPECS = {
    "a_linear": {
        "EXP_NAME": "24-lcic-a-linear",
        "HEAD_TYPE": "lcic_linear",
        "LCIC_USE_CONSENSUS": False,
        "LCIC_USE_INNOVATION": False,
        "trainable_params": 385,
    },
    "b_consensus": {
        "EXP_NAME": "24-lcic-b-consensus",
        "HEAD_TYPE": "lcic",
        "LCIC_USE_CONSENSUS": True,
        "LCIC_USE_INNOVATION": False,
        "trainable_params": 386,
    },
    "c_innovation": {
        "EXP_NAME": "24-lcic-c-innovation",
        "HEAD_TYPE": "lcic",
        "LCIC_USE_CONSENSUS": False,
        "LCIC_USE_INNOVATION": True,
        "trainable_params": 770,
    },
    "d_full": {
        "EXP_NAME": "24-lcic-d-full",
        "HEAD_TYPE": "lcic",
        "LCIC_USE_CONSENSUS": True,
        "LCIC_USE_INNOVATION": True,
        "trainable_params": 771,
    },
    "e_dwlite": {
        "EXP_NAME": "24-lcic-e-dwlite",
        "HEAD_TYPE": "lcic_dwlite",
        "LCIC_USE_CONSENSUS": False,
        "LCIC_USE_INNOVATION": False,
        "trainable_params": 6337,
    },
    "f_conv3x3": {
        "EXP_NAME": "35-gbsp-r32-conv3x3-c16-nossboc-step2floor-dice005-t050-seed2027",
        "HEAD_TYPE": "lcic_conv3x3",
        "LCIC_USE_CONSENSUS": False,
        "LCIC_USE_INNOVATION": False,
        "trainable_params": 8497,
    },
}

COMMON_EXPECTED_OVERRIDES = {
    "DABEV2HARD_PURE_STUDENT": True,
    "R1_ONLY_CACHE_IO": True,
    "R1_HARD_RESIZE_MODE": "bilinear_align_corners_false",
    "R1_HARD_THRESHOLD": 0.5,
    "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
    "FINETUNE_RESET_EPOCH": 0,
    "FINETUNE_RESET_REBUILD_OPTIMIZER": False,
    "FINETUNE_RESET_REBUILD_SCHEDULER": False,
    "FINETUNE_RESET_GLOBAL_STEP": False,
    "FINETUNE_RESET_FORCE_LR_FLOOR": False,
    "FINETUNE_RESET_TEACHER": False,
    "LR_FLOOR_APPLY_AFTER_FINETUNE_RESET": False,
}

R1_SOURCE_OVERRIDES = {
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_r1_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "residual_pass1_37",
    "R1_HARD_SOURCE_KEY": "residual_pass1_37",
}

EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    **R1_SOURCE_OVERRIDES,
    "EXP_NAME": EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

DAGP_NDR_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    **R1_SOURCE_OVERRIDES,
    "EXP_NAME": DAGP_NDR_EXP_NAME,
    "DABEV2HARD_R1_DAGP_NDR": True,
    "DAGP_NDR_FULL_FROM_EPOCH1": True,
    "DAGP_SAFE_WARMUP_EPOCH": 0,
    "DAGP_SAFE_RAMP_START_EPOCH": 1,
    "DAGP_SAFE_RAMP_END_EPOCH": 1,
    "NDR_WARMUP_EPOCH": 0,
    "NDR_RAMP_START_EPOCH": 1,
    "NDR_RAMP_END_EPOCH": 1,
}

CVBR_V1_SECOND_RING_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": CVBR_V1_SECOND_RING_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "CVBR_V1_SECOND_RING_LINEAR": True,
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/dabe_cvbr_v1_train_singleview/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "cvbr_v1_second_ring_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "v1_cvbr_second_ring_37",
    "DABE_CLEAN_CVBR_VERSION": "dabe_cvbr_v1",
    "DABE_CLEAN_CVBR_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "v1_cvbr_second_ring_37",
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

COMMON_REQUIRED_VALUES = {
    "DABEV2HARD_STATIC_ONLY": True,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
    "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
    "STATIC_WEIGHT_MODE": "ones",
    "DABE_PU_DESPL_STATIC_START": 1.0,
    "DABE_PU_DESPL_STATIC_END": 1.0,
    "DABE_PU_DESPL_TEACHER_START": 0.0,
    "DABE_PU_DESPL_TEACHER_END": 0.0,
    "USE_TEACHER_BINARY_FULL_LOSS": False,
    "USE_TEACHER_SOFT_FULL_LOSS": False,
    "USE_TEACHER_CONF_LOSS": False,
    "USE_DABE_CLEAN": True,
    "USE_DABE_PU": False,
    "USE_DABE_PU_STATIC_LOSS": True,
    "MAX_EPOCH": 45,
    "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
    "FINETUNE_RESET_EPOCH": 0,
    "FINETUNE_RESET_REBUILD_OPTIMIZER": False,
    "FINETUNE_RESET_REBUILD_SCHEDULER": False,
    "FINETUNE_RESET_GLOBAL_STEP": False,
    "FINETUNE_RESET_FORCE_LR_FLOOR": False,
    "FINETUNE_RESET_TEACHER": False,
    "LR_FLOOR_APPLY_AFTER_FINETUNE_RESET": False,
    "STOP_AFTER_EPOCH": 0,
}

REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **EXPECTED_OVERRIDES,
}

DAGP_NDR_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **DAGP_NDR_EXPECTED_OVERRIDES,
    "HEAD_TYPE": "dagp_safe",
    "USE_DAGP_SAFE_HEAD": True,
    "USE_NDR_BRANCH": True,
    "USE_NDR_COARSE_AUX": True,
    "USE_BASE_AUX_LOSS": True,
}

CVBR_V1_SECOND_RING_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **CVBR_V1_SECOND_RING_EXPECTED_OVERRIDES,
}

RPR_P1_SECOND_RING_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": RPR_P1_SECOND_RING_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "RPR_P1_SECOND_RING_LINEAR": True,
    "RPR_TRAIN_CACHE_MODE": "p1_only",
    "RPR_TRAIN_CACHE_WORKERS": 2,
    "RPR_TRAIN_CACHE_TORCH_THREADS": 12,
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/dabe_rpr_p1_train_singleview/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "rpr_p1_second_ring_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "p1_rpr_secondring_37",
    "DABE_CLEAN_CVBR_VERSION": "dabe_cvbr_v1",
    "DABE_CLEAN_RPR_VERSION": "dabe_rpr_v1",
    "DABE_CLEAN_RPR_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "p1_rpr_secondring_37",
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

RPR_P1_SECOND_RING_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **RPR_P1_SECOND_RING_EXPECTED_OVERRIDES,
}

RPR_P1_SECOND_RING_DAGP_NDR_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": RPR_P1_SECOND_RING_DAGP_NDR_EXP_NAME,
    "DABEV2HARD_R1_DAGP_NDR": True,
    "RPR_P1_SECOND_RING_DAGP_NDR": True,
    "RPR_TRAIN_CACHE_MODE": "p1_only",
    "RPR_TRAIN_CACHE_WORKERS": 2,
    "RPR_TRAIN_CACHE_TORCH_THREADS": 12,
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/dabe_rpr_p1_train_singleview/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "rpr_p1_second_ring_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "p1_rpr_secondring_37",
    "DABE_CLEAN_CVBR_VERSION": "dabe_cvbr_v1",
    "DABE_CLEAN_RPR_VERSION": "dabe_rpr_v1",
    "DABE_CLEAN_RPR_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "p1_rpr_secondring_37",
    "DAGP_NDR_FULL_FROM_EPOCH1": True,
    "DAGP_SAFE_WARMUP_EPOCH": 0,
    "DAGP_SAFE_RAMP_START_EPOCH": 1,
    "DAGP_SAFE_RAMP_END_EPOCH": 1,
    "NDR_WARMUP_EPOCH": 0,
    "NDR_RAMP_START_EPOCH": 1,
    "NDR_RAMP_END_EPOCH": 1,
}

RPR_P1_SECOND_RING_DAGP_NDR_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **RPR_P1_SECOND_RING_DAGP_NDR_EXPECTED_OVERRIDES,
    "HEAD_TYPE": "dagp_safe",
    "USE_DAGP_SAFE_HEAD": True,
    "USE_NDR_BRANCH": True,
    "USE_NDR_COARSE_AUX": True,
    "USE_BASE_AUX_LOSS": True,
}

GBSP_ABSMM_T063_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T063_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T063_LINEAR": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": True,
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.63,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.63,
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

GBSP_ABSMM_T063_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T063_EXPECTED_OVERRIDES,
}

GBSP_ABSMM_T050_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T050_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T050_LINEAR": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": False,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "predefined_minmax_midpoint",
    "GBSP_THRESHOLD_GT_FREE": True,
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.50,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.50,
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

GBSP_ABSMM_T050_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T050_EXPECTED_OVERRIDES,
}

GBSP_ABSMM_T060_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T060_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T060_LINEAR": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": True,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "threshold_sensitivity_control_t060",
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.60,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.60,
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

GBSP_ABSMM_T060_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T060_EXPECTED_OVERRIDES,
}

GBSP_ABSMM_T058_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T058_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T058_LINEAR": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": True,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "train4040_oracle_gt68",
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.58,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.58,
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

GBSP_ABSMM_T058_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T058_EXPECTED_OVERRIDES,
}

GBSP_R32_ABSMM_T058_EXPECTED_OVERRIDES = {
    **GBSP_ABSMM_T058_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_R32_ABSMM_T058_EXP_NAME,
    "GBSP_VERSION": "gbsp_pca_absmm_r32_v1",
    "GBSP_PCA_RANK_MODE": "fixed",
    "GBSP_FIXED_PCA_RANK": 32,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "train4040_pilot200_even_oracle_gt68",
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../datasets/cache/gbsp_pca_absmm_r32_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_r32_v1",
    "DABE_CLEAN_GBSP_PCA_RANK_MODE": "fixed",
    "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
    "LEAN_PURE_STUDENT_LOGGING": True,
}

GBSP_R32_ABSMM_T058_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_R32_ABSMM_T058_EXPECTED_OVERRIDES,
}

GBSP_ABSMM_T058_DAGP_ONLY_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T058_DAGP_ONLY_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": False,
    "DABEV2HARD_R1_DAGP_NDR": False,
    "DAGP_NDR_FULL_FROM_EPOCH1": False,
    "DAGP_ONLY_FULL_FROM_EPOCH1": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T058_LINEAR": False,
    "GBSP_ABSMM_T058_DAGP_ONLY": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": True,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "train4040_oracle_gt68",
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.58,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.58,
    "HEAD_TYPE": "dagp_safe",
    "USE_DAGP_HEAD": False,
    "USE_DAGP_SAFE_HEAD": True,
    "DAGP_SAFE_WARMUP_EPOCH": 0,
    "DAGP_SAFE_RAMP_START_EPOCH": 1,
    "DAGP_SAFE_RAMP_END_EPOCH": 1,
    "USE_NDR_BRANCH": False,
    "USE_NDR_V2": False,
    "USE_NDR_COARSE_AUX": False,
    "LAMBDA_NDR_COARSE_AUX": 0.0,
    "USE_BASE_AUX_LOSS": True,
}

GBSP_ABSMM_T058_DAGP_ONLY_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T058_DAGP_ONLY_EXPECTED_OVERRIDES,
}

GBSP_ABSMM_T058_NDR_ONLY_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_ABSMM_T058_NDR_ONLY_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": False,
    "DABEV2HARD_R1_DAGP_NDR": False,
    "DAGP_NDR_FULL_FROM_EPOCH1": False,
    "DAGP_ONLY_FULL_FROM_EPOCH1": False,
    "NDR_ONLY_FULL_FROM_EPOCH1": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_ABSMM_T058_LINEAR": False,
    "GBSP_ABSMM_T058_DAGP_ONLY": False,
    "GBSP_ABSMM_T058_NDR_ONLY": True,
    "GBSP_VERSION": "gbsp_pca_absmm_v1",
    "GBSP_ORACLE_THRESHOLD_VERIFICATION": True,
    "GBSP_THRESHOLD_SELECTION_SPLIT": "train4040_oracle_gt68",
    "GBSP_SOURCE_DABE_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
    ),
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_abs_minmax_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_37",
    "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_37",
    "R1_HARD_THRESHOLD": 0.58,
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.58,
    "HEAD_TYPE": "ndr_only",
    "USE_DAGP_HEAD": False,
    "USE_DAGP_SAFE_HEAD": False,
    "DAGP_SAFE_ALPHA_MAX": 0.0,
    "DAGP_SAFE_GAMMA_MAX": 0.0,
    "DAGP_SAFE_USE_PROB_GATE": False,
    "DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE": False,
    "USE_NDR_BRANCH": True,
    "USE_NDR_V2": False,
    "NDR_VERSION": "v1_native_detail_residual",
    "NDR_WARMUP_EPOCH": 0,
    "NDR_RAMP_START_EPOCH": 1,
    "NDR_RAMP_END_EPOCH": 1,
    "USE_NDR_COARSE_AUX": True,
    "LAMBDA_NDR_COARSE_AUX": 0.5,
    "USE_BASE_AUX_LOSS": True,
}

GBSP_ABSMM_T058_NDR_ONLY_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_ABSMM_T058_NDR_ONLY_EXPECTED_OVERRIDES,
}

GBSP_CF_BRC_HC_EXPECTED_OVERRIDES = {
    **COMMON_EXPECTED_OVERRIDES,
    "EXP_NAME": GBSP_CF_BRC_HC_EXP_NAME,
    "DABEV2HARD_R1_LINEAR": True,
    "DABEV2HARD_PURE_STUDENT": True,
    "GBSP_CF_BRC_HC_LINEAR": True,
    "GBSP_THRESHOLD_VERSION": "gbsp_cf_brc_hc_v1",
    "GBSP_THRESHOLD_GT_FREE": True,
    "GBSP_THRESHOLD_METHOD": "cf_brc_hc",
    "DABE_CLEAN_DABE_V2_ROOT": "../workdir/gbsp_threshold/crossfit_train4040",
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "gbsp_cf_brc_hc_hard_68",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY": "hc_mask",
    "DABE_CLEAN_GBSP_THRESHOLD_VERSION": "gbsp_cf_brc_hc_v1",
    "DABE_CLEAN_GBSP_AUGS": ["identity"],
    "R1_HARD_SOURCE_KEY": "hc_mask",
    "HEAD_TYPE": "simple",
    "USE_DAGP_SAFE_HEAD": False,
    "USE_NDR_BRANCH": False,
    "USE_BASE_AUX_LOSS": False,
}

GBSP_CF_BRC_HC_REQUIRED_VALUES = {
    **COMMON_REQUIRED_VALUES,
    **GBSP_CF_BRC_HC_EXPECTED_OVERRIDES,
}


def is_r1hard_linear_pure_student_config(cfg):
    """Identify either named Hard-R1 pure-Student experiment.

    The historical function name is retained because the training entry point
    already imports it as the pure-Student selector.
    """

    return bool(
        getattr(cfg, "GBSP_LCIC_V1", False)
        or getattr(cfg, "GBSP_R32_LINEAR_DICE005_T050_SEED2027", False)
        or getattr(
            cfg,
            "GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027",
            False,
        )
        or getattr(
            cfg,
            "GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027",
            False,
        )
        or getattr(
            cfg,
            "GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027",
            False,
        )
        or getattr(cfg, "GBSP_R16_LINEAR_DICE005_T050_SEED2027", False)
        or getattr(cfg, "GBSP_R32_LINEAR_DICE005_T054_SEED2027", False)
        or getattr(cfg, "GBSP_R32_LINEAR_NODICE_T050_SEED2027", False)
        or
        getattr(cfg, "GT68_LINEAR_CACHED_DIAGNOSTIC", False)
        or getattr(cfg, "R1_DECODER_ISOLATION_V1", False)
        or getattr(cfg, "R1_HSD_V1", False)
        or getattr(cfg, "R1_LAST4_EXPERIMENT", False)
        or getattr(cfg, "DABEV2HARD_R1_DBA", False)
    ) or str(getattr(cfg, "EXP_NAME", "")) in {
        EXP_NAME,
        DAGP_NDR_EXP_NAME,
        CVBR_V1_SECOND_RING_EXP_NAME,
        RPR_P1_SECOND_RING_EXP_NAME,
        RPR_P1_SECOND_RING_DAGP_NDR_EXP_NAME,
        GBSP_ABSMM_T063_EXP_NAME,
        GBSP_ABSMM_T050_EXP_NAME,
        GBSP_ABSMM_T060_EXP_NAME,
        GBSP512C_T060_NATIVE64_EXP_NAME,
        GBSP_ABSMM_T058_EXP_NAME,
        GBSP_R32_ABSMM_T058_EXP_NAME,
        GBSP_R32_LINEAR_DICE005_T050_SEED2027_EXP_NAME,
        GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_EXP_NAME,
        GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_EXP_NAME,
        GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_EXP_NAME,
        GBSP_R16_LINEAR_DICE005_T050_SEED2027_EXP_NAME,
        GBSP_R32_LINEAR_DICE005_T054_SEED2027_EXP_NAME,
        GBSP_R32_LINEAR_NODICE_T050_SEED2027_EXP_NAME,
        GBSP_ABSMM_T058_DAGP_ONLY_EXP_NAME,
        GBSP_ABSMM_T058_NDR_ONLY_EXP_NAME,
        GBSP_CF_BRC_HC_EXP_NAME,
        DBA_EXP_NAME,
    }


def is_r1hard_dagp_ndr_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) in {
        DAGP_NDR_EXP_NAME,
        RPR_P1_SECOND_RING_DAGP_NDR_EXP_NAME,
    }


def is_cvbr_v1_second_ring_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == CVBR_V1_SECOND_RING_EXP_NAME


def is_rpr_p1_second_ring_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == RPR_P1_SECOND_RING_EXP_NAME


def is_rpr_p1_second_ring_dagp_ndr_pure_student_config(cfg):
    return (
        str(getattr(cfg, "EXP_NAME", ""))
        == RPR_P1_SECOND_RING_DAGP_NDR_EXP_NAME
    )


def is_gbsp_absmm_t063_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_ABSMM_T063_EXP_NAME


def is_gbsp_absmm_t050_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_ABSMM_T050_EXP_NAME


def is_gbsp_absmm_t060_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_ABSMM_T060_EXP_NAME


def is_gbsp512c_t060_native64_pure_student_config(cfg):
    return (
        str(getattr(cfg, "EXP_NAME", ""))
        == GBSP512C_T060_NATIVE64_EXP_NAME
    )


def is_gbsp_absmm_t058_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_ABSMM_T058_EXP_NAME


def is_gbsp_r32_absmm_t058_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_R32_ABSMM_T058_EXP_NAME


def is_gbsp_absmm_t058_dagp_only_pure_student_config(cfg):
    return (
        str(getattr(cfg, "EXP_NAME", ""))
        == GBSP_ABSMM_T058_DAGP_ONLY_EXP_NAME
    )


def is_gbsp_absmm_t058_ndr_only_pure_student_config(cfg):
    return (
        str(getattr(cfg, "EXP_NAME", ""))
        == GBSP_ABSMM_T058_NDR_ONLY_EXP_NAME
    )


def is_gbsp_cf_brc_hc_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_CF_BRC_HC_EXP_NAME


def _matches(actual, expected):
    if isinstance(expected, float):
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and abs(value - expected) < 1e-12
    return actual == expected


def validate_gbsp_lcic_config(cfg):
    """Audit LCIC and its declared single-variable training variant."""

    if not bool(getattr(cfg, "GBSP_LCIC_V1", False)):
        return None
    variant = str(getattr(cfg, "LCIC_VARIANT", "")).strip().lower()
    if variant not in LCIC_VARIANT_SPECS:
        raise RuntimeError(f"Unsupported LCIC_VARIANT={variant!r}.")
    spec = LCIC_VARIANT_SPECS[variant]
    dwlite_ssboc_control = bool(
        getattr(
            cfg,
            "GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027",
            False,
        )
    )
    dwlite_ssboc_lr0003_variant = bool(
        getattr(
            cfg,
            "GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027",
            False,
        )
    )
    dwlite_ssboc_lr1e4_floor1e5_variant = bool(
        getattr(
            cfg,
            (
                "GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_"
                "DICE005_T050_SEED2027"
            ),
            False,
        )
    )
    dwlite_ssboc_lr1e4_floor1e5_linear45_variant = bool(
        getattr(
            cfg,
            (
                "GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_"
                "DICE005_T050_SEED2027"
            ),
            False,
        )
    )
    dwlite_ssboc_variant = bool(
        dwlite_ssboc_control
        or dwlite_ssboc_lr0003_variant
        or dwlite_ssboc_lr1e4_floor1e5_variant
        or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
    )
    conv3x3_nossboc_step2floor_variant = bool(
        getattr(
            cfg,
            (
                "GBSP_R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_"
                "DICE005_T050_SEED2027"
            ),
            False,
        )
    )
    r32_variant = bool(getattr(cfg, "LCIC_R32_VARIANT", False))
    reference_config_path = (
        GBSP_R32_ABSMM_T058_CONFIG_PATH
        if r32_variant
        else GBSP_ABSMM_T058_CONFIG_PATH
    )
    reference_cfg = load_config(reference_config_path)
    reference = resolved_control_values(reference_cfg)
    candidate = resolved_control_values(cfg)
    missing = object()
    differences = [
        {
            "field": name,
            "reference": (
                "<MISSING>" if reference.get(name, missing) is missing
                else reference.get(name)
            ),
            "candidate": (
                "<MISSING>" if candidate.get(name, missing) is missing
                else candidate.get(name)
            ),
        }
        for name in sorted(set(reference).union(candidate))
        if reference.get(name, missing) != candidate.get(name, missing)
    ]
    expected_overrides = {
        "EXP_NAME": spec["EXP_NAME"],
        "GBSP_LCIC_V1": True,
        "LCIC_VARIANT": variant,
        "LCIC_USE_CONSENSUS": spec["LCIC_USE_CONSENSUS"],
        "LCIC_USE_INNOVATION": spec["LCIC_USE_INNOVATION"],
        "LCIC_AFFINITY_EPS": 1e-6,
        "LCIC_DWLITE_CHANNELS": 16,
        "HEAD_TYPE": spec["HEAD_TYPE"],
    }
    if r32_variant:
        if variant != "d_full" and not (
            variant == "e_dwlite" and dwlite_ssboc_variant
        ) and not (
            variant == "f_conv3x3" and conv3x3_nossboc_step2floor_variant
        ):
            raise RuntimeError(
                "Fixed-R32 decoder experiments require LCIC-D-FULL or the "
                "declared DW-Lite+SSBOC control."
            )
        expected_overrides.update(
            {
                "EXP_NAME": "25-gbsp-r32-lcic-d-full-dice005-t058",
                "LCIC_R32_VARIANT": True,
            }
        )
    dataset_threshold_variant = bool(
        getattr(cfg, "LCIC_DATASET_THRESHOLD_VARIANT", False)
    )
    dataset_loss_weight_variant = bool(
        getattr(cfg, "LCIC_DATASET_LOSS_WEIGHT_VARIANT", False)
    )
    lr_step_variant = bool(
        getattr(cfg, "LCIC_LR_STEP_VARIANT", False)
    )
    lr_0003_variant = bool(
        getattr(cfg, "LCIC_LR_0003_VARIANT", False)
    )
    soft_dice_variant = bool(
        getattr(cfg, "LCIC_SOFT_DICE_VARIANT", False)
    )
    soft_dice_010_variant = bool(
        getattr(cfg, "LCIC_SOFT_DICE_010_VARIANT", False)
    )
    graph_candidate_variant = bool(
        getattr(cfg, "LCIC_R32_GRAPH_CANDIDATE_VARIANT", False)
    )
    enabled_training_variants = sum(
        (
            dataset_threshold_variant,
            dataset_loss_weight_variant,
            lr_step_variant,
            lr_0003_variant,
            soft_dice_variant,
        )
    )
    r32_dataset_threshold_variant = bool(
        r32_variant
        and (
            variant == "d_full"
            or dwlite_ssboc_variant
            or conv3x3_nossboc_step2floor_variant
        )
        and dataset_threshold_variant
        and not dataset_loss_weight_variant
        and not lr_step_variant
    )
    r32_dice_dataset_threshold_variant = bool(
        r32_dataset_threshold_variant and soft_dice_variant
    )
    r32_uniform_t050_variant = bool(
        r32_dataset_threshold_variant
        and getattr(cfg, "LCIC_UNIFORM_T050_VARIANT", False)
    )
    r32_dice_uniform_t050_variant = bool(
        r32_uniform_t050_variant and soft_dice_variant
    )
    no_soft_dice_variant = bool(
        getattr(cfg, "LCIC_NO_SOFT_DICE_VARIANT", False)
    )
    seed_variant = bool(getattr(cfg, "LCIC_SEED_VARIANT", False))
    structural_only_e3_variant = bool(
        getattr(cfg, "LCIC_STRUCTURAL_ONLY_E3", False)
    )
    consensus_gain3_variant = bool(
        getattr(cfg, "LCIC_CONSENSUS_GAIN3_VARIANT", False)
    )
    adaptive_gate_variant = bool(
        getattr(cfg, "LCIC_ADAPTIVE_GATE_VARIANT", False)
    )
    ssboc_variant = bool(getattr(cfg, "LCIC_SSBOC_VARIANT", False))
    rgbmv_hflipmean_variant = bool(
        getattr(cfg, "GBSP_RGBMV_HFLIP_SOFTMEAN_VARIANT", False)
    )
    if enabled_training_variants > 1 and not r32_dataset_threshold_variant:
        raise RuntimeError(
            "LCIC training variants must be tested independently."
        )
    if dataset_threshold_variant:
        if (
            variant != "d_full"
            and not dwlite_ssboc_variant
            and not conv3x3_nossboc_step2floor_variant
        ):
            raise RuntimeError(
                "Dataset-specific LCIC thresholds are currently restricted "
                "to LCIC_VARIANT='d_full'."
            )
        if r32_dataset_threshold_variant:
            r32_thresholds = (
                {"TR-CAMO": 0.50, "TR-COD10K": 0.50}
                if r32_uniform_t050_variant
                else {"TR-CAMO": 0.50, "TR-COD10K": 0.58}
            )
            r32_exp_name = (
                "26-gbsp-r32-lcic-d-full-dice005-uniform050"
                if r32_uniform_t050_variant
                else "26-gbsp-r32-lcic-d-full-dice005-camo050-cod058"
            )
            r32_selection_split = (
                "train4040_oracle_gt68_r32_uniform050"
                if r32_uniform_t050_variant
                else (
                    "train4040_oracle_gt68_r32_dataset_specific_"
                    "camo050_cod10k058"
                )
            )
            expected_overrides.update(
                {
                    "EXP_NAME": r32_exp_name,
                    "LCIC_DATASET_THRESHOLD_VARIANT": True,
                    "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": r32_thresholds,
                    "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
                    "GBSP_THRESHOLD_SELECTION_SPLIT": r32_selection_split,
                }
            )
            if r32_uniform_t050_variant:
                expected_overrides["LCIC_UNIFORM_T050_VARIANT"] = True
        else:
            expected_overrides.update(
                {
                    "EXP_NAME": "24-lcic-d-full-camo058-cod10k066",
                    "LCIC_DATASET_THRESHOLD_VARIANT": True,
                    "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                        "TR-CAMO": 0.58,
                        "TR-COD10K": 0.66,
                    },
                    "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
                }
            )
    if dataset_loss_weight_variant:
        if variant != "d_full":
            raise RuntimeError(
                "Dataset-specific LCIC loss weights are currently restricted "
                "to LCIC_VARIANT='d_full'."
            )
        expected_overrides.update(
            {
                "EXP_NAME": "24-lcic-d-full-camow15",
                "LCIC_DATASET_LOSS_WEIGHT_VARIANT": True,
                "TR_CAMO_LOSS_WEIGHT": 1.5,
                "TR_COD10K_LOSS_WEIGHT": 1.0,
            }
        )
    if lr_step_variant:
        if variant != "d_full":
            raise RuntimeError(
                "LCIC LR-step variants are currently restricted to "
                "LCIC_VARIANT='d_full'."
            )
        expected_overrides.update(
            {
                "EXP_NAME": "24-lcic-d-full-lrstep50",
                "LCIC_LR_STEP_VARIANT": True,
                "LR_STEP_SIZE": 50,
            }
        )
    if soft_dice_variant:
        if (
            variant != "d_full"
            and not dwlite_ssboc_variant
            and not conv3x3_nossboc_step2floor_variant
        ):
            raise RuntimeError(
                "LCIC soft-Dice variants are currently restricted to "
                "LCIC_VARIANT='d_full'."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    (
                        "26-gbsp-r32-lcic-d-full-dice005-uniform050"
                        if r32_dice_uniform_t050_variant
                        else "26-gbsp-r32-lcic-d-full-dice005-camo050-cod058"
                    )
                    if r32_dice_dataset_threshold_variant
                    else (
                        "25-gbsp-r32-lcic-d-full-dice005-t058"
                        if r32_variant
                        else "24-lcic-d-full-dice005"
                    )
                ),
                "LCIC_SOFT_DICE_VARIANT": True,
                "LCIC_SOFT_DICE_WEIGHT": 0.05,
            }
        )
    if graph_candidate_variant:
        if not r32_dice_uniform_t050_variant or seed_variant:
            raise RuntimeError(
                "The graph-candidate cache variant requires the seed-3407 "
                "R32 LCIC-D Soft-Dice uniform-t0.50 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "30-gbsp-r32-lcic-d-full-dice005-t050-bw1-p30-sf085"
                ),
                "LCIC_R32_GRAPH_CANDIDATE_VARIANT": True,
                "GBSP_VERSION": "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1",
                "GBSP_CANDIDATE_BORDER_WIDTH": 1,
                "GBSP_CANDIDATE_TOP_PERCENT": 30.0,
                "GBSP_GRAPH_SIGMA_F": 0.085,
                "DABE_SIGMA_F": 0.085,
                "DABE_CLEAN_DABE_V2_ROOT": (
                    "../datasets/cache/gbsp_r32_graph_sigmaf_sweep_v1/"
                    "sigmaf085/bw1-p30/dinov1-s8"
                ),
                "DABE_CLEAN_GBSP_VERSION": (
                    "gbsp_r32_pathcand_bw1_p30_sigmaf085_v1"
                ),
                "DABE_CLEAN_GBSP_GRAPH_CANDIDATE_DIRECT": True,
                "GBSP_THRESHOLD_SELECTION_SPLIT": (
                    "train4040_oracle_gt68_r32_bw1_p30_sigmaf085_uniform050"
                ),
            }
        )
    if no_soft_dice_variant:
        if not r32_uniform_t050_variant:
            raise RuntimeError(
                "The no-Soft-Dice ablation is restricted to the R32 "
                "uniform-t0.50 LCIC-D configuration."
            )
        if soft_dice_variant:
            raise RuntimeError(
                "LCIC_NO_SOFT_DICE_VARIANT requires "
                "LCIC_SOFT_DICE_VARIANT=False."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "26-gbsp-r32-lcic-d-full-nodice-uniform050-"
                    f"seed{int(getattr(cfg, 'SEED', -1))}"
                ),
                "LCIC_NO_SOFT_DICE_VARIANT": True,
                "LCIC_SOFT_DICE_VARIANT": False,
                "LCIC_SOFT_DICE_WEIGHT": 0.0,
            }
        )
    if seed_variant:
        if not r32_uniform_t050_variant:
            raise RuntimeError(
                "LCIC seed repeats are restricted to the validated R32 "
                "uniform-t0.50 LCIC-D configurations."
            )
        seed = int(getattr(cfg, "SEED", -1))
        if seed not in LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS:
            raise RuntimeError(f"Unsupported LCIC repeat seed: {seed}.")
        expected_overrides.update(
            {
                "EXP_NAME": (
                    (
                        "26-gbsp-r32-lcic-d-full-nodice-uniform050-"
                        if no_soft_dice_variant
                        else (
                            "32-gbsp-r32-lcic-d-full-dice010-t050-"
                            if soft_dice_010_variant
                            else "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                        )
                    )
                    + f"seed{seed}"
                ),
                "LCIC_SEED_VARIANT": True,
                "SEED": seed,
            }
        )
    if structural_only_e3_variant:
        if (
            not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or adaptive_gate_variant
        ):
            raise RuntimeError(
                "LCIC structural-only epoch-3 fork requires the seed-2027 "
                "R32 LCIC-D Soft-Dice=0.05 uniform-t0.50 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                    "seed2027-structonly-e3"
                ),
                "LCIC_STRUCTURAL_ONLY_E3": True,
                "LCIC_STRUCTURAL_ONLY_REQUIRED_RESUME_EPOCH": 3,
                "LCIC_STRUCTURAL_ONLY_REQUIRED_SOURCE_EXP": (
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                    "seed2027"
                ),
                "STOP_AFTER_EPOCH": 4,
            }
        )
    if consensus_gain3_variant:
        if (
            not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or structural_only_e3_variant
            or adaptive_gate_variant
        ):
            raise RuntimeError(
                "LCIC consensus-gain3 training requires the seed-2027 R32 "
                "LCIC-D Soft-Dice=0.05 uniform-t0.50 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                    "seed2027-consensusgain3"
                ),
                "LCIC_CONSENSUS_GAIN3_VARIANT": True,
                "LCIC_CONSENSUS_GAIN": 3.0,
                "LCIC_INNOVATION_GAIN": 1.0,
            }
        )
    if adaptive_gate_variant:
        if (
            not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or structural_only_e3_variant
            or consensus_gain3_variant
        ):
            raise RuntimeError(
                "LCIC adaptive-gate training requires the seed-2027 R32 "
                "LCIC-D Soft-Dice=0.05 uniform-t0.50 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "26-gbsp-r32-lcic-d-full-dice005-uniform050-"
                    "seed2027-adaptivegate"
                ),
                "LCIC_ADAPTIVE_GATE_VARIANT": True,
                "LCIC_ADAPTIVE_GATE_HIDDEN": 8,
                "LCIC_ADAPTIVE_GATE_EPS": 1e-6,
                "LCIC_CONSENSUS_GAIN": 1.0,
                "LCIC_INNOVATION_GAIN": 1.0,
            }
        )
    if ssboc_variant:
        if (
            not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or structural_only_e3_variant
            or consensus_gain3_variant
            or adaptive_gate_variant
            or bool(getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False))
        ):
            raise RuntimeError(
                "LCIC SSBOC requires the seed-2027 R32 LCIC-D-FULL "
                "Soft-Dice=0.05 uniform-t0.50 control protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "34-gbsp-r32-lcic-d-full-ssboc-dice005-t050-"
                    "seed2027"
                ),
                "LCIC_SSBOC_VARIANT": True,
                "GBSP_SSBOC_VARIANT": True,
            }
        )
    if rgbmv_hflipmean_variant:
        if (
            not ssboc_variant
            or variant != "d_full"
            or not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or graph_candidate_variant
            or bool(getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False))
        ):
            raise RuntimeError(
                "RGB multi-view identity+hflip soft mean requires the "
                "matched seed-2027 R32 LCIC-D-FULL+SSBOC control."
            )
        expected_overrides.update(
            {
                "EXP_NAME": "36-gbsp-r32-rgbmv-hflipmean-lcic-ssboc",
                "GBSP_RGBMV_HFLIP_SOFTMEAN_VARIANT": True,
                "GBSP_VERSION": "gbsp_r32_rgbmv_id_hflip_softmean_v1",
                "DABE_CLEAN_DABE_V2_ROOT": (
                    "../datasets/cache/"
                    "gbsp_r32_rgbmv_id_hflip_softmean_v1/dinov1-s8"
                ),
                "DABE_CLEAN_GBSP_VERSION": (
                    "gbsp_r32_rgbmv_id_hflip_softmean_v1"
                ),
                "DABE_CLEAN_GBSP_AUGS": ["identity", "hflip"],
                "GBSP_THRESHOLD_SELECTION_SPLIT": (
                    "train4040_oracle_gt68_r32_rgbmv_"
                    "id_hflip_softmean_t050"
                ),
            }
        )
    if dwlite_ssboc_variant:
        if (
            variant != "e_dwlite"
            or not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or bool(getattr(cfg, "LCIC_SSBOC_VARIANT", False))
            or not bool(getattr(cfg, "GBSP_SSBOC_VARIANT", False))
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or structural_only_e3_variant
            or consensus_gain3_variant
            or adaptive_gate_variant
            or bool(getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False))
        ):
            raise RuntimeError(
                "DW-Lite+SSBOC requires the seed-2027 R32 uniform-t0.50 "
                "Soft-Dice=0.05 protocol and feature-space e_dwlite head."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027-"
                    "lr1e4-floor1e5-linear45"
                    if dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                    else
                    "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027-"
                    "lr1e4-floor1e5"
                    if dwlite_ssboc_lr1e4_floor1e5_variant
                    else "34-gbsp-r32-dwlite16-ssboc-dice005-t050-"
                    "seed2027-lr0003"
                    if dwlite_ssboc_lr0003_variant
                    else "34-gbsp-r32-dwlite16-ssboc-dice005-t050-seed2027"
                ),
                "GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027": (
                    dwlite_ssboc_control
                ),
                "LCIC_SSBOC_VARIANT": False,
                "GBSP_SSBOC_VARIANT": True,
                "LCIC_VARIANT": "e_dwlite",
                "HEAD_TYPE": "lcic_dwlite",
                "LCIC_USE_CONSENSUS": False,
                "LCIC_USE_INNOVATION": False,
                "LCIC_DWLITE_CHANNELS": 16,
            }
        )
        if dwlite_ssboc_lr0003_variant:
            expected_overrides.update(
                {
                    "GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027": True,
                    "LR": 3e-4,
                    "DINO": {**reference["DINO"], "lr": 3e-4},
                }
            )
        if dwlite_ssboc_lr1e4_floor1e5_variant:
            expected_overrides.update(
                {
                    (
                        "GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_"
                        "DICE005_T050_SEED2027"
                    ): True,
                    "LR": 1e-4,
                    "LR_FLOOR": 1e-5,
                    "DINO": {**reference["DINO"], "lr": 1e-4},
                }
            )
        if dwlite_ssboc_lr1e4_floor1e5_linear45_variant:
            expected_overrides.update(
                {
                    (
                        "GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_"
                        "DICE005_T050_SEED2027"
                    ): True,
                    "LR": 1e-4,
                    "LR_FLOOR": 1e-5,
                    "LR_POLICY": "linear_floor_two_stage",
                    "LR_LINEAR_STAGE1_EPOCHS": 45,
                    "LR_LINEAR_STAGE2_EPOCHS": 0,
                    "DINO": {**reference["DINO"], "lr": 1e-4},
                }
            )
    if conv3x3_nossboc_step2floor_variant:
        if (
            variant != "f_conv3x3"
            or not r32_dice_uniform_t050_variant
            or not seed_variant
            or int(getattr(cfg, "SEED", -1)) != 2027
            or bool(getattr(cfg, "LCIC_SSBOC_VARIANT", False))
            or bool(getattr(cfg, "GBSP_SSBOC_VARIANT", False))
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or soft_dice_010_variant
            or structural_only_e3_variant
            or consensus_gain3_variant
            or adaptive_gate_variant
            or bool(getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False))
        ):
            raise RuntimeError(
                "Conv3x3-Lite no-SSBOC step2-floor experiment requires the "
                "seed-2027 R32 uniform-t0.50 Soft-Dice=0.05 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "35-gbsp-r32-conv3x3-c16-nossboc-step2floor-"
                    "dice005-t050-seed2027"
                ),
                (
                    "GBSP_R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_"
                    "DICE005_T050_SEED2027"
                ): True,
                "LCIC_VARIANT": "f_conv3x3",
                "HEAD_TYPE": "lcic_conv3x3",
                "LCIC_USE_CONSENSUS": False,
                "LCIC_USE_INNOVATION": False,
                "LCIC_CONV3X3_CHANNELS": 16,
                "LCIC_SSBOC_VARIANT": False,
                "GBSP_SSBOC_VARIANT": False,
                "LR_POLICY": "step_then_floor",
                "LR_STEP_THEN_FLOOR_EPOCHS": 2,
            }
        )
    if lr_0003_variant:
        if (
            not r32_dice_uniform_t050_variant
            or graph_candidate_variant
            or no_soft_dice_variant
            or seed_variant
            or soft_dice_010_variant
        ):
            raise RuntimeError(
                "LCIC LR=3e-4 ablation requires the seed-42 R32 LCIC-D "
                "Soft-Dice uniform-t0.50 control protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "31-gbsp-r32-lcic-d-full-dice005-t050-lr0003"
                ),
                "LCIC_LR_0003_VARIANT": True,
                "LR": 3e-4,
                "DINO": {**reference["DINO"], "lr": 3e-4},
            }
        )
    if soft_dice_010_variant:
        if (
            not r32_dice_uniform_t050_variant
            or graph_candidate_variant
            or no_soft_dice_variant
            or lr_0003_variant
            or (
                seed_variant
                and int(getattr(cfg, "SEED", -1)) != 2027
            )
        ):
            raise RuntimeError(
                "LCIC Soft-Dice=0.10 ablation requires the seed-42 control "
                "or its declared seed-2027 repeat under the R32 LCIC-D "
                "uniform-t0.50 protocol."
            )
        expected_overrides.update(
            {
                "EXP_NAME": (
                    "32-gbsp-r32-lcic-d-full-dice010-t050-seed2027"
                    if seed_variant
                    else "32-gbsp-r32-lcic-d-full-dice010-t050"
                ),
                "LCIC_SOFT_DICE_010_VARIANT": True,
                "LCIC_SOFT_DICE_WEIGHT": 0.10,
            }
        )
    expected_difference_names = {
        name
        for name, value in expected_overrides.items()
        if reference.get(name, missing) != value
    }
    difference_names = {record["field"] for record in differences}
    errors = []
    if difference_names != expected_difference_names:
        errors.append(
            "LCIC difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(expected_difference_names)}"
        )
    for name, value in expected_overrides.items():
        actual = candidate.get(name, "<MISSING>")
        if not _matches(actual, value):
            errors.append(
                f"LCIC field mismatch: {name}={actual!r}, expected={value!r}"
            )

    protected_names = (
        "BACKBONE_KEY",
        "DINO",
        "DINO_FEATURE_TYPE",
        "DINO_FEATURE_INPUT_SIZE",
        "FEATURE_INPUT_SIZE",
        "GRID_SIZE",
        "LOSS_SIZE",
        "TRAIN_DATASETS",
        "VAL_DATASETS",
        "TEST_DATASETS",
        "BATCH_SIZE",
        "NUM_WORKERS",
        "SEED",
        "LR",
        "LR_FLOOR",
        "LR_POLICY",
        "MAX_EPOCH",
        "FINETUNE_RESET_EPOCH",
        "FINETUNE_RESET_LR",
        "DABE_CLEAN_DABE_V2_ROOT",
        "DABE_CLEAN_DABE_V2_SOURCE_KEY",
        "DABE_CLEAN_STATIC_TARGET_SOURCE",
        "DABE_CLEAN_DABE_V2_HARD_THRESHOLD",
        "R1_HARD_SOURCE_KEY",
        "R1_HARD_RESIZE_MODE",
        "R1_HARD_THRESHOLD",
        "BEST_DATASET",
        "BEST_METRIC",
        "THRESHOLD",
    )
    protected = {}
    for name in protected_names:
        reference_value = reference.get(name, "<MISSING>")
        candidate_value = candidate.get(name, "<MISSING>")
        expected_protected_value = (
            expected_overrides["SEED"]
            if seed_variant and name == "SEED"
            else (
                expected_overrides["LR"]
                if (
                    lr_0003_variant
                    or dwlite_ssboc_lr0003_variant
                    or dwlite_ssboc_lr1e4_floor1e5_variant
                    or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                )
                and name == "LR"
                else (
                    expected_overrides["DINO"]
                    if (
                        lr_0003_variant
                        or dwlite_ssboc_lr0003_variant
                        or dwlite_ssboc_lr1e4_floor1e5_variant
                        or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                    )
                    and name == "DINO"
                    else (
                        expected_overrides["LR_FLOOR"]
                        if (
                            dwlite_ssboc_lr1e4_floor1e5_variant
                            or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                        )
                        and name == "LR_FLOOR"
                        else (
                            expected_overrides["LR_POLICY"]
                            if (
                                dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                                or conv3x3_nossboc_step2floor_variant
                            )
                            and name == "LR_POLICY"
                            else reference_value
                        )
                    )
                )
            )
        )
        if (
            graph_candidate_variant or rgbmv_hflipmean_variant
        ) and name == "DABE_CLEAN_DABE_V2_ROOT":
            expected_protected_value = expected_overrides[
                "DABE_CLEAN_DABE_V2_ROOT"
            ]
        matches = expected_protected_value == candidate_value
        protected[name] = {
            "reference": reference_value,
            "candidate": candidate_value,
            "expected": expected_protected_value,
            "matches": matches,
        }
        if not matches:
            errors.append(f"LCIC changed protected protocol field: {name}")

    forbidden_true = (
        "USE_DAGP_SAFE_HEAD",
        "USE_NDR_BRANCH",
        "USE_BASE_AUX_LOSS",
        "USE_MULTI_LEVEL_FEATURE",
        "ONLINE_DINO_LAST4",
        "USE_TEACHER_BINARY_FULL_LOSS",
        "USE_TEACHER_SOFT_FULL_LOSS",
        "USE_TEACHER_CONF_LOSS",
        "LOOK_TWICE",
    )
    enabled_forbidden = [
        name for name in forbidden_true if bool(candidate.get(name, False))
    ]
    if enabled_forbidden:
        errors.append(f"LCIC forbidden branches/options enabled: {enabled_forbidden}")

    report = {
        "schema": (
            "gbsp_r32_lcic_decoder_config_audit_v1"
            if r32_variant
            else "gbsp_lcic_decoder_only_config_audit_v1"
        ),
        "status": "PASS" if not errors else "FAIL",
        "reference_config": str(reference_config_path),
        "candidate_config": str(
            LCIC_R32_RGBMV_HFLIPMEAN_SSBOC_CONFIG_PATH
            if rgbmv_hflipmean_variant
            else GBSP_R32_CONV3X3_C16_NOSSBOC_STEP2FLOOR_DICE005_T050_SEED2027_CONFIG_PATH
            if conv3x3_nossboc_step2floor_variant
            else GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_LINEAR45_DICE005_T050_SEED2027_CONFIG_PATH
            if dwlite_ssboc_lr1e4_floor1e5_linear45_variant
            else GBSP_R32_DWLITE16_SSBOC_LR1E4_FLOOR1E5_DICE005_T050_SEED2027_CONFIG_PATH
            if dwlite_ssboc_lr1e4_floor1e5_variant
            else GBSP_R32_DWLITE16_SSBOC_LR0003_DICE005_T050_SEED2027_CONFIG_PATH
            if dwlite_ssboc_lr0003_variant
            else GBSP_R32_DWLITE16_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
            if dwlite_ssboc_variant
            else LCIC_R32_D_FULL_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
            if ssboc_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_ADAPTIVE_GATE_CONFIG_PATH
            if adaptive_gate_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_CONSENSUS_GAIN3_CONFIG_PATH
            if consensus_gain3_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_SEED2027_STRUCTONLY_E3_CONFIG_PATH
            if structural_only_e3_variant
            else LCIC_R32_D_FULL_SOFT_DICE_010_T050_SEED2027_CONFIG_PATH
            if soft_dice_010_variant and seed_variant
            else LCIC_R32_D_FULL_SOFT_DICE_010_T050_CONFIG_PATH
            if soft_dice_010_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_LR0003_CONFIG_PATH
            if lr_0003_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_BW1P30_SIGMAF085_CONFIG_PATH
            if graph_candidate_variant
            else LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH
            if no_soft_dice_variant
            else LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS[
                int(getattr(cfg, "SEED"))
            ]
            if seed_variant
            else (
                LCIC_R32_D_FULL_SOFT_DICE_T050_CONFIG_PATH
                if r32_dice_uniform_t050_variant
                else (
                    LCIC_R32_D_FULL_SOFT_DICE_DATASET_THRESHOLD_CONFIG_PATH
                    if r32_dice_dataset_threshold_variant
                    else (
                        LCIC_R32_D_FULL_SOFT_DICE_CONFIG_PATH
                        if r32_variant
                        else (
                            LCIC_DATASET_THRESHOLD_CONFIG_PATH
                            if dataset_threshold_variant
                            else (
                                LCIC_DATASET_LOSS_WEIGHT_CONFIG_PATH
                                if dataset_loss_weight_variant
                                else (
                                    LCIC_LR_STEP_CONFIG_PATH
                                    if lr_step_variant
                                    else (
                                        LCIC_SOFT_DICE_CONFIG_PATH
                                        if soft_dice_variant
                                        else LCIC_CONFIG_PATHS[variant]
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "variant": variant,
        "actual_differences": differences,
        "required_effective_differences": sorted(expected_difference_names),
        "protected_protocol_fields": protected,
        "trainable_params": (
            827 if adaptive_gate_variant else int(spec["trainable_params"])
        ),
        "active_trainable_params": (
            386
            if structural_only_e3_variant
            else 827
            if adaptive_gate_variant
            else int(spec["trainable_params"])
        ),
        "errors": errors,
        "contract": {
            "backbone": "frozen_DINOv1_ViT-S8",
            "feature": "single_cached_F12_384x37x37",
            "decoder_input": "DINO_feature_only",
            "decoder_operation_space": (
                "feature_space_only"
                if dwlite_ssboc_variant or conv3x3_nossboc_step2floor_variant
                else "anchor_logit_residual"
                if variant == "d_full"
                else "feature_space"
            ),
            "intermediate_logit_refinement": (
                False
                if dwlite_ssboc_variant or conv3x3_nossboc_step2floor_variant
                else None
            ),
            "consensus_gain": float(getattr(cfg, "LCIC_CONSENSUS_GAIN", 1.0)),
            "innovation_gain": float(getattr(cfg, "LCIC_INNOVATION_GAIN", 1.0)),
            "adaptive_gate": adaptive_gate_variant,
            "adaptive_gate_hidden": int(
                getattr(cfg, "LCIC_ADAPTIVE_GATE_HIDDEN", 0)
            ),
            "adaptive_gate_input": (
                "detached_entropy_foreground_mass_consensus_ratio_"
                "innovation_ratio"
                if adaptive_gate_variant
                else "none"
            ),
            "adaptive_residual_normalization": (
                "per_image_unit_rms_detached_scale"
                if adaptive_gate_variant
                else "none"
            ),
            "gbsp_residual_decoder_input": False,
            "rgb_decoder_input": False,
            "sobel_decoder_input": False,
            "target": (
                (
                    "GBSP_R32_BW1_P30_SIGMAF085_abs_minmax_"
                    "strict_gt_uniform_0.50"
                )
                if graph_candidate_variant
                else "GBSP_R32_abs_minmax_strict_gt_uniform_0.50"
                if r32_uniform_t050_variant
                else (
                    "GBSP_R32_abs_minmax_strict_gt_CAMO_0.50_COD10K_0.58"
                    if r32_dice_dataset_threshold_variant
                    else (
                        "GBSP_abs_minmax_strict_gt_CAMO_0.58_COD10K_0.66"
                        if dataset_threshold_variant
                        else "unchanged_GBSP_abs_minmax_strict_gt_0.58"
                    )
                )
            ),
            "loss": (
                (
                    "per_image_BCEWithLogits_plus_0.05_soft_Dice_plus_"
                    "parameter_free_self_scaled_bias_orthogonal_GBSP_"
                    "correlation"
                )
                if ssboc_variant or dwlite_ssboc_variant
                else "per_sample_BCEWithLogits_weighted_CAMO_1.5_COD10K_1.0_"
                "normalized_by_batch_weight_sum"
                if dataset_loss_weight_variant
                else (
                    (
                        "mean_BCEWithLogits_plus_0.10_soft_Dice"
                        if soft_dice_010_variant
                        else "mean_BCEWithLogits_plus_0.05_soft_Dice"
                    )
                    if soft_dice_variant
                    else "unchanged_single_mean_BCEWithLogits"
                )
            ),
            "pca_rank_mode": "fixed" if r32_variant else "legacy_ev90_capped_r8",
            "fixed_pca_rank": 32 if r32_variant else None,
            "scheduler": (
                "iteration_StepLR_step25_gamma0.95_first2_then_floor2e-5"
                if conv3x3_nossboc_step2floor_variant
                else "manual_linear_1e-4_to_1e-5_over_45_epochs"
                if dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                else
                "iteration_StepLR_step50_gamma0.95_with_lr_floor"
                if lr_step_variant
                else "unchanged_iteration_StepLR_step25_gamma0.95_with_lr_floor"
            ),
            "learning_rate": (
                1e-4
                if (
                    dwlite_ssboc_lr1e4_floor1e5_variant
                    or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                )
                else 3e-4
                if lr_0003_variant or dwlite_ssboc_lr0003_variant
                else reference["LR"]
            ),
            "learning_rate_floor": (
                1e-5
                if (
                    dwlite_ssboc_lr1e4_floor1e5_variant
                    or dwlite_ssboc_lr1e4_floor1e5_linear45_variant
                )
                else reference["LR_FLOOR"]
            ),
            "optimizer_reset": "permanently_disabled",
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
        },
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }
    if errors:
        raise RuntimeError(
            "GBSP LCIC decoder-only config audit failed: " + "; ".join(errors)
        )
    return report


def validate_r1hard_dba_pure_student_config(cfg):
    """Audit DBA as the only change from the established S/8 1x1 run."""

    reference_cfg = load_config(CONFIG_PATH)
    reference = resolved_control_values(reference_cfg)
    candidate = resolved_control_values(cfg)
    missing = object()
    differences = [
        {
            "field": name,
            "reference": (
                "<MISSING>"
                if reference.get(name, missing) is missing
                else reference.get(name)
            ),
            "candidate": (
                "<MISSING>"
                if candidate.get(name, missing) is missing
                else candidate.get(name)
            ),
        }
        for name in sorted(set(reference).union(candidate))
        if reference.get(name, missing) != candidate.get(name, missing)
    ]
    expected_overrides = {
        "EXP_NAME": DBA_EXP_NAME,
        "DABEV2HARD_R1_LINEAR": False,
        "DABEV2HARD_R1_DBA": True,
        "HEAD_TYPE": "dba",
        "DBA_EMBED_DIM": 64,
        "DBA_OUTPUT_SIZE": 68,
        "DBA_ORTHOGONAL_WEIGHT": 1.0,
        "LOOK_TWICE": False,
    }
    expected_difference_names = {
        name
        for name, value in expected_overrides.items()
        if reference.get(name, missing) != value
    }
    difference_names = {record["field"] for record in differences}
    errors = []
    if difference_names != expected_difference_names:
        errors.append(
            "DBA difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(expected_difference_names)}"
        )
    for name, value in expected_overrides.items():
        actual = candidate.get(name, "<MISSING>")
        if not _matches(actual, value):
            errors.append(
                f"DBA field mismatch: {name}={actual!r}, expected={value!r}"
            )

    protected = (
        "BACKBONE_KEY",
        "DINO_FEATURE_TYPE",
        "DINO_FEATURE_INPUT_SIZE",
        "FEATURE_INPUT_SIZE",
        "LOSS_SIZE",
        "BATCH_SIZE",
        "NUM_WORKERS",
        "SEED",
        "LR",
        "LR_FLOOR",
        "MAX_EPOCH",
        "FINETUNE_RESET_EPOCH",
        "FINETUNE_RESET_LR",
        "DABE_CLEAN_DABE_V2_ROOT",
        "DABE_CLEAN_DABE_V2_SOURCE_KEY",
        "DABE_CLEAN_STATIC_TARGET_SOURCE",
        "R1_HARD_SOURCE_KEY",
        "R1_HARD_RESIZE_MODE",
        "R1_HARD_THRESHOLD",
    )
    protected_values = {}
    for name in protected:
        reference_value = reference.get(name, "<MISSING>")
        candidate_value = candidate.get(name, "<MISSING>")
        matches = reference_value == candidate_value
        protected_values[name] = {
            "reference": reference_value,
            "candidate": candidate_value,
            "matches": matches,
        }
        if not matches:
            errors.append(f"DBA changed protected protocol field: {name}")

    required = {
        **COMMON_REQUIRED_VALUES,
        **COMMON_EXPECTED_OVERRIDES,
        **R1_SOURCE_OVERRIDES,
        "DABEV2HARD_PURE_STUDENT": True,
        "USE_DAGP_SAFE_HEAD": False,
        "USE_NDR_BRANCH": False,
        "USE_BASE_AUX_LOSS": False,
        "FINETUNE_RESET_TEACHER": False,
    }
    for name, value in required.items():
        actual = candidate.get(name, "<MISSING>")
        if not _matches(actual, value):
            errors.append(
                f"DBA pure-Student field mismatch: {name}={actual!r}, "
                f"expected={value!r}"
            )
    dino = candidate.get("DINO", {})
    expected_dino = {
        "model_name": "facebook/dino-vits8",
        "patch_size": 8,
        "embed_dim": 384,
        "feature_input_size": 296,
        "lr": 6e-4,
    }
    for name, value in expected_dino.items():
        actual = dino.get(name, "<MISSING>") if isinstance(dino, dict) else "<MISSING>"
        if not _matches(actual, value):
            errors.append(
                f"DBA DINOv1-S/8 field mismatch: DINO[{name!r}]="
                f"{actual!r}, expected={value!r}"
            )
    if errors:
        raise RuntimeError(
            "Hard-R1 DBA pure-Student config audit failed: "
            + "; ".join(errors)
        )
    return {
        "schema": "r1hard_dba_pure_student_config_audit_v1",
        "status": "PASS",
        "reference_config": str(CONFIG_PATH),
        "candidate_config": str(DBA_CONFIG_PATH),
        "actual_differences": differences,
        "required_effective_differences": sorted(
            expected_difference_names
        ),
        "protected_protocol_fields": protected_values,
        "errors": [],
        "contract": {
            "backbone": "DINOv1-S/8",
            "feature": "single_cached_F12_384x37x37",
            "feature_input_size": 296,
            "decoder_input": "bilinear_F12_to_384x68x68",
            "target": "strict_Hard_R1_68_from_residual_pass1_37",
            "student": "DBA_384_to_2x64_fg_bg",
            "loss": "foreground_BCE+reverse_background_BCE+orthogonal",
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
            "apm": False,
            "look_twice": False,
        },
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def build_r1hard_linear_audit_report(reference_cfg, candidate_cfg):
    reference = resolved_control_values(reference_cfg)
    candidate = resolved_control_values(candidate_cfg)
    missing = object()
    differences = []
    for name in sorted(set(reference).union(candidate)):
        left = reference.get(name, missing)
        right = candidate.get(name, missing)
        if left == right:
            continue
        differences.append(
            {
                "field": name,
                "reference": "<MISSING>" if left is missing else left,
                "candidate": "<MISSING>" if right is missing else right,
            }
        )

    difference_names = {record["field"] for record in differences}
    dagp_ndr = is_r1hard_dagp_ndr_pure_student_config(candidate_cfg)
    cvbr_v1 = is_cvbr_v1_second_ring_pure_student_config(candidate_cfg)
    rpr_p1 = is_rpr_p1_second_ring_pure_student_config(candidate_cfg)
    rpr_p1_dagp_ndr = (
        is_rpr_p1_second_ring_dagp_ndr_pure_student_config(candidate_cfg)
    )
    gbsp_absmm_t063 = is_gbsp_absmm_t063_pure_student_config(candidate_cfg)
    gbsp_absmm_t050 = is_gbsp_absmm_t050_pure_student_config(candidate_cfg)
    gbsp_absmm_t060 = is_gbsp_absmm_t060_pure_student_config(candidate_cfg)
    gbsp_absmm_t058 = is_gbsp_absmm_t058_pure_student_config(candidate_cfg)
    gbsp_r32_absmm_t058 = is_gbsp_r32_absmm_t058_pure_student_config(
        candidate_cfg
    )
    gbsp_absmm_t058_dagp_only = (
        is_gbsp_absmm_t058_dagp_only_pure_student_config(candidate_cfg)
    )
    gbsp_absmm_t058_ndr_only = (
        is_gbsp_absmm_t058_ndr_only_pure_student_config(candidate_cfg)
    )
    gbsp_cf_brc_hc = is_gbsp_cf_brc_hc_pure_student_config(candidate_cfg)
    if gbsp_r32_absmm_t058:
        expected_overrides = GBSP_R32_ABSMM_T058_EXPECTED_OVERRIDES
        required_values = GBSP_R32_ABSMM_T058_REQUIRED_VALUES
    elif gbsp_cf_brc_hc:
        expected_overrides = GBSP_CF_BRC_HC_EXPECTED_OVERRIDES
        required_values = GBSP_CF_BRC_HC_REQUIRED_VALUES
    elif gbsp_absmm_t058_ndr_only:
        expected_overrides = GBSP_ABSMM_T058_NDR_ONLY_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T058_NDR_ONLY_REQUIRED_VALUES
    elif gbsp_absmm_t058_dagp_only:
        expected_overrides = GBSP_ABSMM_T058_DAGP_ONLY_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T058_DAGP_ONLY_REQUIRED_VALUES
    elif gbsp_absmm_t058:
        expected_overrides = GBSP_ABSMM_T058_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T058_REQUIRED_VALUES
    elif gbsp_absmm_t050:
        expected_overrides = GBSP_ABSMM_T050_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T050_REQUIRED_VALUES
    elif gbsp_absmm_t060:
        expected_overrides = GBSP_ABSMM_T060_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T060_REQUIRED_VALUES
    elif gbsp_absmm_t063:
        expected_overrides = GBSP_ABSMM_T063_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T063_REQUIRED_VALUES
    elif rpr_p1_dagp_ndr:
        expected_overrides = RPR_P1_SECOND_RING_DAGP_NDR_EXPECTED_OVERRIDES
        required_values = RPR_P1_SECOND_RING_DAGP_NDR_REQUIRED_VALUES
    elif dagp_ndr:
        expected_overrides = DAGP_NDR_EXPECTED_OVERRIDES
        required_values = DAGP_NDR_REQUIRED_VALUES
    elif cvbr_v1:
        expected_overrides = CVBR_V1_SECOND_RING_EXPECTED_OVERRIDES
        required_values = CVBR_V1_SECOND_RING_REQUIRED_VALUES
    elif rpr_p1:
        expected_overrides = RPR_P1_SECOND_RING_EXPECTED_OVERRIDES
        required_values = RPR_P1_SECOND_RING_REQUIRED_VALUES
    else:
        expected_overrides = EXPECTED_OVERRIDES
        required_values = REQUIRED_VALUES
    expected_difference_names = {
        name
        for name, expected in expected_overrides.items()
        if reference.get(name, missing) != expected
    }
    errors = []
    if difference_names != expected_difference_names:
        errors.append(
            "effective difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(expected_difference_names)}"
        )
    for name, expected in required_values.items():
        actual = candidate.get(name, "<MISSING>")
        if not _matches(actual, expected):
            errors.append(
                f"Hard-R1 pure-Student field mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    return {
        "schema": (
            "gbsp_r32_absmm_t058_linear_pure_student_config_audit_v1"
            if gbsp_r32_absmm_t058
            else (
                "gbsp_absmm_t058_ndr_only_pure_student_config_audit_v1"
                if gbsp_absmm_t058_ndr_only
                else (
                    "gbsp_cf_brc_hc_linear_pure_student_config_audit_v1"
                    if gbsp_cf_brc_hc
                    else (
                        "gbsp_absmm_t058_dagp_only_pure_student_config_audit_v1"
                        if gbsp_absmm_t058_dagp_only
                        else (
                            "gbsp_absmm_t058_linear_pure_student_config_audit_v1"
                            if gbsp_absmm_t058
                            else (
                                "gbsp_absmm_t050_linear_pure_student_config_audit_v1"
                                if gbsp_absmm_t050
                                else (
                                    "gbsp_absmm_t060_linear_pure_student_config_audit_v1"
                                    if gbsp_absmm_t060
                                    else (
                                        "gbsp_absmm_t063_linear_pure_student_config_audit_v1"
                                        if gbsp_absmm_t063
                                        else (
                                            "rpr_p1_second_ring_dagp_ndr_pure_student_config_audit_v1"
                                            if rpr_p1_dagp_ndr
                                            else (
                                                "r1hard_dagp_ndr_pure_student_config_audit_v1"
                                                if dagp_ndr
                                                else (
                                                    "cvbr_v1_second_ring_linear_pure_student_config_audit_v1"
                                                    if cvbr_v1
                                                    else (
                                                        "rpr_p1_second_ring_linear_pure_student_config_audit_v1"
                                                        if rpr_p1
                                                        else "r1hard_linear_pure_student_config_audit_v1"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "status": "PASS" if not errors else "FAIL",
        "reference_config": str(REFERENCE_CONFIG_PATH),
        "candidate_config": str(
            GBSP_R32_ABSMM_T058_CONFIG_PATH
            if gbsp_r32_absmm_t058
            else (
                GBSP_ABSMM_T058_NDR_ONLY_CONFIG_PATH
                if gbsp_absmm_t058_ndr_only
                else (
                    GBSP_CF_BRC_HC_CONFIG_PATH
                    if gbsp_cf_brc_hc
                    else (
                        GBSP_ABSMM_T058_DAGP_ONLY_CONFIG_PATH
                        if gbsp_absmm_t058_dagp_only
                        else (
                            GBSP_ABSMM_T058_CONFIG_PATH
                            if gbsp_absmm_t058
                            else (
                                GBSP_ABSMM_T050_CONFIG_PATH
                                if gbsp_absmm_t050
                                else (
                                    GBSP_ABSMM_T060_CONFIG_PATH
                                    if gbsp_absmm_t060
                                    else (
                                        GBSP_ABSMM_T063_CONFIG_PATH
                                        if gbsp_absmm_t063
                                        else (
                                            RPR_P1_SECOND_RING_DAGP_NDR_CONFIG_PATH
                                            if rpr_p1_dagp_ndr
                                            else (
                                                DAGP_NDR_CONFIG_PATH
                                                if dagp_ndr
                                                else (
                                                    CVBR_V1_SECOND_RING_CONFIG_PATH if cvbr_v1 else CONFIG_PATH
                                                    if not rpr_p1
                                                    else RPR_P1_SECOND_RING_CONFIG_PATH
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "actual_differences": differences,
        "required_effective_differences": sorted(expected_difference_names),
        "errors": errors,
        "contract": {
            "source": (
                "hc_mask"
                if gbsp_cf_brc_hc
                else (
                    "gbsp_abs_minmax_37"
                    if (
                        gbsp_absmm_t058
                        or gbsp_r32_absmm_t058
                        or gbsp_absmm_t058_dagp_only
                        or gbsp_absmm_t058_ndr_only
                        or gbsp_absmm_t050
                        or gbsp_absmm_t060
                        or gbsp_absmm_t063
                    )
                    else (
                        "v1_cvbr_second_ring_37"
                        if cvbr_v1
                        else (
                            "p1_rpr_secondring_37"
                            if (rpr_p1 or rpr_p1_dagp_ndr)
                            else "residual_pass1_37"
                        )
                    )
                )
            ),
            "resize": "bilinear_37_to_68_align_corners_false",
            "target": (
                "strict_greater_than_0.58"
                if (
                    gbsp_absmm_t058
                    or gbsp_r32_absmm_t058
                    or gbsp_absmm_t058_dagp_only
                    or gbsp_absmm_t058_ndr_only
                )
                else (
                    "strict_greater_than_0.60"
                    if gbsp_absmm_t060
                    else (
                        "strict_greater_than_0.63"
                        if gbsp_absmm_t063
                        else "strict_greater_than_0.5"
                    )
                )
            ),
            "student": (
                "ndr_v1_only"
                if gbsp_absmm_t058_ndr_only
                else (
                    "dagp_safe_only"
                    if gbsp_absmm_t058_dagp_only
                    else (
                        "dagp_safe_plus_ndr_v1"
                        if dagp_ndr
                        else "single_1x1_conv"
                    )
                )
            ),
            "pca_rank_mode": "fixed" if gbsp_r32_absmm_t058 else None,
            "fixed_pca_rank": 32 if gbsp_r32_absmm_t058 else None,
            "dagp_ndr_scale_epoch1": 1.0 if dagp_ndr else None,
            "dagp_scale_epoch1": (
                1.0 if gbsp_absmm_t058_dagp_only else None
            ),
            "ndr_scale_epoch1": (
                1.0 if gbsp_absmm_t058_ndr_only else None
            ),
            "dagp_graph_executed": (
                False if gbsp_absmm_t058_ndr_only else None
            ),
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
            "optimizer_reset": "permanently_disabled",
            "static_loss_weight": 1.0,
            "teacher_loss_weight": 0.0,
        },
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def validate_gbsp512c_t060_native64_pure_student_config(cfg):
    """Audit the intentional 296/37/68 -> 512/64/64 protocol change."""

    candidate = resolved_control_values(cfg)
    required = {
        **GBSP_ABSMM_T060_REQUIRED_VALUES,
        "EXP_NAME": GBSP512C_T060_NATIVE64_EXP_NAME,
        "GBSP512C_T060_NATIVE64_LINEAR": True,
        "GBSP_VERSION": "gbsp_resolution_512c_native64_v1",
        "GBSP_THRESHOLD_SELECTION_SPLIT": (
            "full512c_threshold_sensitivity_t060"
        ),
        "GBSP_SOURCE_DABE_ROOT": "",
        "CACHE_ROOT": "../workdir/gbsp_resolution_512/train512c/cache",
        "FEATURE_CACHE_KEY": "dinov1-s8",
        "FEATURE_INPUT_SIZE": 512,
        "GRID_SIZE": 64,
        "SUPERVISION_GRID": 64,
        "LOSS_SIZE": 64,
        "USE_LEGACY_68_INTERPOLATION": False,
        "DABE_GRID": 64,
        "DABE_LOSS_SIZE": 64,
        "GBSP_RESOLUTION_VARIANT": "512-C",
        "GBSP_BORDER_WIDTH": 2,
        "GBSP_BG_RATIO": 0.30,
        "GBSP_PCA_ENERGY": 0.90,
        "GBSP_PCA_MIN_RANK": 1,
        "GBSP_PCA_MAX_RANK": 12,
        "GBSP_THRESHOLD": 0.60,
        "GBSP_GT_USED_FOR_GENERATION": False,
        "DABE_CLEAN_DABE_V2_ROOT": (
            "../workdir/gbsp_resolution_512/train512c/pseudo/dinov1-s8"
        ),
        "DABE_CLEAN_DABE_V2_SOURCE_KEY": "gbsp_abs_minmax_64",
        "DABE_CLEAN_GBSP_SOURCE_KEY": "gbsp_abs_minmax_64",
        "DABE_CLEAN_GBSP_SOURCE_GRID": 64,
        "DABE_CLEAN_GBSP_VERSION": "gbsp_resolution_512c_native64_v1",
        "DABE_CLEAN_GBSP_DIRECT_FROM_FEATURES": True,
        "DABE_CLEAN_GBSP_AUGS": ["identity"],
        "R1_HARD_SOURCE_KEY": "gbsp_abs_minmax_64",
        "R1_HARD_RESIZE_MODE": "native64_no_resize",
        "R1_HARD_THRESHOLD": 0.60,
        "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.60,
    }
    errors = []
    for name, expected in required.items():
        actual = candidate.get(name, "<MISSING>")
        if not _matches(actual, expected):
            errors.append(
                f"512-C field mismatch: {name}={actual!r}, expected={expected!r}"
            )
    dino = candidate.get("DINO", {})
    expected_dino = {
        "model_name": "facebook/dino-vits8",
        "patch_size": 8,
        "embed_dim": 384,
        "feature_input_size": 512,
        "lr": 6e-4,
    }
    for name, expected in expected_dino.items():
        actual = dino.get(name, "<MISSING>") if isinstance(dino, dict) else "<MISSING>"
        if not _matches(actual, expected):
            errors.append(
                f"512-C DINO mismatch: DINO[{name!r}]={actual!r}, "
                f"expected={expected!r}"
            )
    if errors:
        raise RuntimeError(
            "GBSP512-C native-64 pure-Student config audit failed: "
            + "; ".join(errors)
        )
    return {
        "schema": "gbsp512c_t060_native64_pure_student_config_audit_v1",
        "status": "PASS",
        "candidate_config": str(GBSP512C_T060_NATIVE64_CONFIG_PATH),
        "actual_differences": [
            {"field": name} for name in (
                "DINO.feature_input_size",
                "FEATURE_INPUT_SIZE",
                "GRID_SIZE",
                "LOSS_SIZE",
                "GBSP_BORDER_WIDTH",
                "GBSP_PCA_MAX_RANK",
                "DABE_CLEAN_DABE_V2_SOURCE_KEY",
            )
        ],
        "required_effective_differences": [
            "DINO.feature_input_size",
            "FEATURE_INPUT_SIZE",
            "GRID_SIZE",
            "LOSS_SIZE",
            "GBSP_BORDER_WIDTH",
            "GBSP_PCA_MAX_RANK",
            "DABE_CLEAN_DABE_V2_SOURCE_KEY",
        ],
        "errors": [],
        "contract": {
            "feature": "single_cached_DINOv1-S8_key_384x64x64_at_input512",
            "background": "Full-BC_BW2_R30",
            "reconstruction": "global_PCA_energy90_cap12_absolute_residual",
            "calibration": "per_image_minmax",
            "target": "strict_greater_than_0.60_native64",
            "student": "single_1x1_conv_384_to_1_at_native64",
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
        },
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def validate_r1hard_linear_pure_student_config(cfg):
    if not is_r1hard_linear_pure_student_config(cfg):
        return None
    if bool(
        getattr(
            cfg,
            "GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027",
            False,
        )
    ):
        rgbmv_hflipmean_variant = bool(
            getattr(
                cfg,
                "GBSP_R32_LINEAR_RGBMV_HFLIPMEAN_SSBOC_VARIANT",
                False,
            )
        )
        reference_cfg = load_config(
            GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "EXP_NAME",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
            "GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027",
            "GBSP_SSBOC_VARIANT",
        }
        if rgbmv_hflipmean_variant:
            allowed_differences.update(
                {
                    "GBSP_R32_LINEAR_RGBMV_HFLIPMEAN_SSBOC_VARIANT",
                    "GBSP_VERSION",
                    "DABE_CLEAN_DABE_V2_ROOT",
                    "DABE_CLEAN_GBSP_VERSION",
                    "DABE_CLEAN_GBSP_AUGS",
                    "GBSP_THRESHOLD_SELECTION_SPLIT",
                }
            )
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear SSBOC control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": (
                "36-gbsp-r32-rgbmv-hflipmean-1x1-ssboc"
                if rgbmv_hflipmean_variant
                else GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_EXP_NAME
            ),
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": False,
            "GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027": True,
            "GBSP_SSBOC_VARIANT": True,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "USE_NDR_BRANCH": False,
            "USE_BASE_AUX_LOSS": False,
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        if rgbmv_hflipmean_variant:
            expected.update(
                {
                    "GBSP_R32_LINEAR_RGBMV_HFLIPMEAN_SSBOC_VARIANT": True,
                    "GBSP_VERSION": "gbsp_r32_rgbmv_id_hflip_softmean_v1",
                    "DABE_CLEAN_DABE_V2_ROOT": (
                        "../datasets/cache/"
                        "gbsp_r32_rgbmv_id_hflip_softmean_v1/dinov1-s8"
                    ),
                    "DABE_CLEAN_GBSP_VERSION": (
                        "gbsp_r32_rgbmv_id_hflip_softmean_v1"
                    ),
                    "DABE_CLEAN_GBSP_AUGS": ["identity", "hflip"],
                    "GBSP_THRESHOLD_SELECTION_SPLIT": (
                        "train4040_oracle_gt68_r32_rgbmv_"
                        "id_hflip_softmean_t050"
                    ),
                }
            )
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R32 linear SSBOC control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        if bool(getattr(cfg, "GBSP_CONFIDENCE_BCE_VARIANT", False)):
            errors.append("SSBOC cannot enable GBSP confidence-weighted BCE.")
        if bool(getattr(cfg, "LCIC_DATASET_LOSS_WEIGHT_VARIANT", False)):
            errors.append("SSBOC must retain uniform per-image dataset weighting.")
        report = {
            "schema": "gbsp_r32_linear_ssboc_dice005_t050_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_RGBMV_HFLIPMEAN_SSBOC_CONFIG_PATH
                if rgbmv_hflipmean_variant
                else GBSP_R32_LINEAR_SSBOC_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.50_both_datasets",
                "pseudo_views": (
                    "true_RGB_identity_hflip_inverse_aligned_soft_mean"
                    if rgbmv_hflipmean_variant
                    else "identity"
                ),
                "loss": (
                    "BCE_mean_plus_0.05_SoftDice_plus_parameter_free_"
                    "self_scaled_bias_orthogonal_GBSP_correlation"
                ),
                "continuous_supervision": (
                    "same_detached_R32_GBSP_score_used_to_build_hard_target"
                ),
                "new_tunable_hyperparameters": 0,
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear SSBOC t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(
            cfg,
            "GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027",
            False,
        )
    ):
        reference_cfg = load_config(
            GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "EXP_NAME",
            "GBSP_CONFIDENCE_BCE_CLASSWISE_NORMALIZE",
            "GBSP_CONFIDENCE_BCE_FLOOR",
            "GBSP_CONFIDENCE_BCE_GAMMA",
            "GBSP_CONFIDENCE_BCE_VARIANT",
            "GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear confidence-BCE control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": (
                GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_EXP_NAME
            ),
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": False,
            "GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027": True,
            "GBSP_CONFIDENCE_BCE_VARIANT": True,
            "GBSP_CONFIDENCE_BCE_FLOOR": 0.50,
            "GBSP_CONFIDENCE_BCE_GAMMA": 1.0,
            "GBSP_CONFIDENCE_BCE_CLASSWISE_NORMALIZE": True,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R32 linear confidence-BCE control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        report = {
            "schema": (
                "gbsp_r32_linear_confbce_dice005_t050_seed2027_audit_v1"
            ),
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_CONFBCE_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.50_both_datasets",
                "loss": (
                    "classwise_normalized_GBSP_confidence_BCE_"
                    "floor0.50_gamma1.0_plus_0.05_SoftDice"
                ),
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear confidence-BCE Dice005 t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(
            cfg,
            "GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027",
            False,
        )
    ):
        reference_cfg = load_config(
            GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "EXP_NAME",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
            "GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027",
            "SIMPLE_FEATURE_L2_EPS",
            "SIMPLE_FEATURE_L2_NORMALIZE",
            "SIMPLE_FEATURE_L2_SCALE",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear scaled-L2 control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": (
                GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_EXP_NAME
            ),
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": False,
            "GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027": True,
            "SIMPLE_FEATURE_L2_NORMALIZE": True,
            "SIMPLE_FEATURE_L2_SCALE": 10.59,
            "SIMPLE_FEATURE_L2_EPS": 1e-12,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R32 linear scaled-L2 control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        report = {
            "schema": "gbsp_r32_linear_l2s1059_dice005_t050_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_L2S1059_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": (
                    "cached_DINOv1-S8_F12_per_patch_L2_fixed_norm10.59_"
                    "then_interpolated_37_to_68"
                ),
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.50_both_datasets",
                "loss": "BCEWithLogits_mean_plus_0.05_SoftDice",
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear scaled-L2 Dice005 t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(cfg, "GBSP_R16_LINEAR_DICE005_T050_SEED2027", False)
    ):
        reference_cfg = load_config(
            GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "DABE_CLEAN_DABE_V2_ROOT",
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK",
            "DABE_CLEAN_GBSP_VERSION",
            "EXP_NAME",
            "GBSP_FIXED_PCA_RANK",
            "GBSP_R16_LINEAR_DICE005_T050_SEED2027",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
            "GBSP_THRESHOLD_SELECTION_SPLIT",
            "GBSP_VERSION",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R16 linear t0.50 control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": GBSP_R16_LINEAR_DICE005_T050_SEED2027_EXP_NAME,
            "GBSP_R16_LINEAR_DICE005_T050_SEED2027": True,
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": False,
            "GBSP_VERSION": "gbsp_pca_absmm_r16_v1",
            "GBSP_PCA_RANK_MODE": "fixed",
            "GBSP_FIXED_PCA_RANK": 16,
            "DABE_CLEAN_DABE_V2_ROOT": (
                "../datasets/cache/gbsp_pca_absmm_r16_pseudo_cache/dinov1-s8"
            ),
            "DABE_CLEAN_GBSP_VERSION": "gbsp_pca_absmm_r16_v1",
            "DABE_CLEAN_GBSP_PCA_RANK_MODE": "fixed",
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 16,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R16 linear t0.50 control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        report = {
            "schema": "gbsp_r16_linear_dice005_t050_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R16_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R16_minmax_strict_gt_0.50_both_datasets",
                "loss": "BCEWithLogits_mean_plus_0.05_SoftDice",
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R16 linear Dice005 t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(cfg, "GBSP_R32_LINEAR_DICE005_T054_SEED2027", False)
    ):
        reference_cfg = load_config(
            GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET",
            "EXP_NAME",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
            "GBSP_R32_LINEAR_DICE005_T054_SEED2027",
            "GBSP_THRESHOLD_SELECTION_SPLIT",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear t0.54 control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": GBSP_R32_LINEAR_DICE005_T054_SEED2027_EXP_NAME,
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": False,
            "GBSP_R32_LINEAR_DICE005_T054_SEED2027": True,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.54,
                "TR-COD10K": 0.54,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R32 linear t0.54 control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        report = {
            "schema": "gbsp_r32_linear_dice005_t054_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_DICE005_T054_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.54_both_datasets",
                "loss": "BCEWithLogits_mean_plus_0.05_SoftDice",
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear Dice005 t0.54 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(cfg, "GBSP_R32_LINEAR_NODICE_T050_SEED2027", False)
    ):
        reference_cfg = load_config(
            LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "EXP_NAME",
            "GBSP_LCIC_V1",
            "GBSP_R32_LINEAR_NODICE_T050_SEED2027",
            "HEAD_TYPE",
            "LCIC_USE_CONSENSUS",
            "LCIC_USE_INNOVATION",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear No-Dice control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": GBSP_R32_LINEAR_NODICE_T050_SEED2027_EXP_NAME,
            "GBSP_R32_LINEAR_NODICE_T050_SEED2027": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": False,
            "LCIC_SOFT_DICE_WEIGHT": 0.0,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    "R32 linear No-Dice control field mismatch: "
                    f"{name}={actual!r}, expected={value!r}"
                )
        report = {
            "schema": "gbsp_r32_linear_nodice_t050_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                LCIC_R32_D_FULL_T050_NO_DICE_SEED2027_CONFIG_PATH
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_NODICE_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.50_both_datasets",
                "loss": "BCEWithLogits_mean",
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear No-Dice t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(
        getattr(cfg, "GBSP_R32_LINEAR_DICE005_T050_SEED2027", False)
    ):
        reference_cfg = load_config(
            LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS[2027]
        )
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "EXP_NAME",
            "GBSP_LCIC_V1",
            "GBSP_LINEAR_SOFT_DICE_CONTROL",
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027",
            "HEAD_TYPE",
            "LCIC_USE_CONSENSUS",
            "LCIC_USE_INNOVATION",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "R32 linear control difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": GBSP_R32_LINEAR_DICE005_T050_SEED2027_EXP_NAME,
            "GBSP_R32_LINEAR_DICE005_T050_SEED2027": True,
            "GBSP_LINEAR_SOFT_DICE_CONTROL": True,
            "GBSP_LCIC_V1": False,
            "HEAD_TYPE": "simple",
            "LCIC_USE_CONSENSUS": False,
            "LCIC_USE_INNOVATION": False,
            "LCIC_SOFT_DICE_VARIANT": True,
            "LCIC_SOFT_DICE_WEIGHT": 0.05,
            "GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_GBSP_FIXED_PCA_RANK": 32,
            "DABE_CLEAN_DABE_V2_HARD_THRESHOLDS_BY_DATASET": {
                "TR-CAMO": 0.50,
                "TR-COD10K": 0.50,
            },
            "DABE_CLEAN_DABE_V2_REQUIRE_DATASET_THRESHOLD": True,
            "THRESHOLD": 0.50,
            "SEED": 2027,
            "LR": 0.0006,
            "MAX_EPOCH": 45,
            "BATCH_SIZE": 16,
            "LOSS_SIZE": 68,
            "PURE_STUDENT_RESET_PERMANENTLY_DISABLED": True,
            "FINETUNE_RESET_EPOCH": 0,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    f"R32 linear control field mismatch: {name}={actual!r}, "
                    f"expected={value!r}"
                )
        report = {
            "schema": "gbsp_r32_linear_dice005_t050_seed2027_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(
                LCIC_R32_D_FULL_SOFT_DICE_T050_SEED_CONFIG_PATHS[2027]
            ),
            "candidate_config": str(
                GBSP_R32_LINEAR_DICE005_T050_SEED2027_CONFIG_PATH
            ),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature": "cached_DINOv1-S8_F12_interpolated_37_to_68",
                "student": "single_1x1_conv_384_to_1",
                "pseudo": "fixed_R32_minmax_strict_gt_0.50_both_datasets",
                "loss": "BCEWithLogits_mean_plus_0.05_SoftDice",
                "prediction_threshold": 0.50,
                "seed": 2027,
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R32 linear Dice005 t0.50 control audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(getattr(cfg, "GBSP_LCIC_V1", False)):
        return validate_gbsp_lcic_config(cfg)
    if bool(getattr(cfg, "DABEV2HARD_R1_DBA", False)) or str(
        getattr(cfg, "EXP_NAME", "")
    ) == DBA_EXP_NAME:
        return validate_r1hard_dba_pure_student_config(cfg)
    if is_gbsp512c_t060_native64_pure_student_config(cfg):
        return validate_gbsp512c_t060_native64_pure_student_config(cfg)
    if bool(getattr(cfg, "GT68_LINEAR_CACHED_DIAGNOSTIC", False)):
        reference_cfg = load_config(CONFIG_PATH)
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": (
                    "<MISSING>"
                    if reference.get(name, missing) is missing
                    else reference.get(name)
                ),
                "candidate": (
                    "<MISSING>"
                    if candidate.get(name, missing) is missing
                    else candidate.get(name)
                ),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        allowed_differences = {
            "DECODER_SUPERVISION_SOURCE",
            "EXP_NAME",
            "GT68_LINEAR_CACHED_DIAGNOSTIC",
            "GT_DIAGNOSTIC_REFERENCE_CONFIG",
            "GT_DIAGNOSTIC_RESIZE_MODE",
            "GT_DIAGNOSTIC_STRICT_BINARY",
            "GT_DIAGNOSTIC_SUPERVISION",
        }
        difference_names = {record["field"] for record in differences}
        errors = []
        if difference_names != allowed_differences:
            errors.append(
                "GT_68 cached-linear difference set mismatch: "
                f"actual={sorted(difference_names)}, "
                f"required={sorted(allowed_differences)}"
            )
        expected = {
            "EXP_NAME": GT68_LINEAR_CACHED_EXP_NAME,
            "GT68_LINEAR_CACHED_DIAGNOSTIC": True,
            "GT_DIAGNOSTIC_SUPERVISION": True,
            "GT_DIAGNOSTIC_REFERENCE_CONFIG": (
                "configs/dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
                "staticonly_purestudent_long45_lrfloor_2e5.py"
            ),
            "DECODER_SUPERVISION_SOURCE": "gt_hard_68",
            "GT_DIAGNOSTIC_RESIZE_MODE": "nearest_to_68",
            "GT_DIAGNOSTIC_STRICT_BINARY": True,
            "HEAD_TYPE": "simple",
            "USE_MULTI_LEVEL_FEATURE": "<MISSING>",
            "ONLINE_DINO_LAST4": "<MISSING>",
            "LOSS_SIZE": 68,
            "BATCH_SIZE": 16,
        }
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    f"GT_68 cached-linear field mismatch: {name}={actual!r}, "
                    f"expected={value!r}"
                )
        reference_path = Path(
            str(candidate.get("GT_DIAGNOSTIC_REFERENCE_CONFIG", ""))
        )
        if reference_path.is_absolute() or ".." in reference_path.parts:
            errors.append(
                "GT_DIAGNOSTIC_REFERENCE_CONFIG must be a repository-relative path."
            )
        report = {
            "schema": "gt68_cached_linear_oracle_config_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(CONFIG_PATH),
            "candidate_config": str(GT68_LINEAR_CACHED_CONFIG_PATH),
            "actual_differences": differences,
            "required_effective_differences": sorted(allowed_differences),
            "errors": errors,
            "contract": {
                "feature_source": "legacy_single_layer_feature_cache",
                "student": "single_1x1_conv",
                "target": "strict_binary_training_gt_nearest_68",
                "loss": "single_mean_bce_with_logits_at_68",
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
                "online_dino_forward": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "GT_68 cached-linear config audit failed: "
                + "; ".join(errors)
            )
        return report
    if bool(getattr(cfg, "R1_DECODER_ISOLATION_V1", False)):
        from common.r1_decoder_isolation import (
            validate_r1_decoder_isolation_config,
        )

        return validate_r1_decoder_isolation_config(cfg)
    if bool(getattr(cfg, "R1_HSD_V1", False)) or bool(
        getattr(cfg, "R1_LAST4_EXPERIMENT", False)
    ):
        formal_online = bool(getattr(cfg, "R1_FORMAL_ONLINE_V1", False))
        reference_cfg = load_config(DAGP_NDR_CONFIG_PATH)
        reference = resolved_control_values(reference_cfg)
        candidate = resolved_control_values(cfg)
        missing = object()
        differences = [
            {
                "field": name,
                "reference": "<MISSING>" if reference.get(name, missing) is missing else reference.get(name),
                "candidate": "<MISSING>" if candidate.get(name, missing) is missing else candidate.get(name),
            }
            for name in sorted(set(reference).union(candidate))
            if reference.get(name, missing) != candidate.get(name, missing)
        ]
        expected = {
            **COMMON_REQUIRED_VALUES,
            **R1_SOURCE_OVERRIDES,
            **COMMON_EXPECTED_OVERRIDES,
            "DABEV2HARD_PURE_STUDENT": True,
            "PSEUDO_LABEL_MODE": "r1_hard",
            "PSEUDO_LABEL_THRESHOLD": 0.5,
            "DINO_FEATURE_MODE": (
                "online_last4" if formal_online else "last4"
            ),
            "DINO_FEATURE_KEYS": ["f9", "f10", "f11", "f12"],
            "USE_MULTI_LEVEL_FEATURE": True,
            "MULTI_LEVEL_LAYERS": [9, 10, 11, 12],
            "MULTI_LEVEL_FEATURE_TYPE": "attention_key_projection",
            "MULTI_LEVEL_FEATURE_DTYPE": "float32",
            "USE_DAGP_SAFE_HEAD": False,
            "USE_NDR_BRANCH": False,
            "USE_NDR_V2": False,
            "FINETUNE_RESET_TEACHER": False,
        }
        if formal_online:
            expected.update(
                {
                    "R1_FORMAL_ONLINE_V1": True,
                    "ONLINE_DINO_LAST4": True,
                    "FINETUNE_RESET_EPOCH": 0,
                    "FINETUNE_RESET_REBUILD_OPTIMIZER": False,
                    "FINETUNE_RESET_REBUILD_SCHEDULER": False,
                    "FINETUNE_RESET_GLOBAL_STEP": False,
                    "FINETUNE_RESET_FORCE_LR_FLOOR": False,
                    "LR_FLOOR_APPLY_AFTER_FINETUNE_RESET": False,
                    "LR_LINEAR_STAGE1_EPOCHS": 45,
                }
            )
        if bool(getattr(cfg, "R1_HSD_V1", False)):
            expected.update(
                {
                    "DECODER_TYPE": "hsd_v1",
                    "HEAD_TYPE": "hsd_v1",
                    "HSD_OUTPUT_SIZE": 148,
                    "HSD_SEMANTIC_CHANNELS": 64,
                    "HSD_DETAIL_CHANNELS": 32,
                    "USE_BASE_AUX_LOSS": True,
                    "USE_NDR_COARSE_AUX": True,
                    "LAMBDA_NDR_COARSE_AUX": 0.5,
                }
            )
            if formal_online:
                expected.update(
                    {
                        "HSD_COARSE_SIZE": 74,
                        "HSD_SPATIAL_SUPERVISION_SIZES": [37, 74, 148],
                        "LAMBDA_BASE_AUX_AFTER_RESET": 0.5,
                    }
                )
        else:
            expected.update(
                {
                    "DECODER_TYPE": "last4_linear",
                    "HEAD_TYPE": "last4_linear_probe",
                    "USE_BASE_AUX_LOSS": False,
                    "USE_DETAIL": False,
                }
            )
        errors = []
        for name, value in expected.items():
            actual = candidate.get(name, "<MISSING>")
            if not _matches(actual, value):
                errors.append(
                    f"R1 last-four field mismatch: {name}={actual!r}, expected={value!r}"
                )
        for name in (
            "USE_TEACHER_BINARY_FULL_LOSS",
            "USE_TEACHER_SOFT_FULL_LOSS",
            "USE_TEACHER_CONF_LOSS",
        ):
            if bool(candidate.get(name, False)):
                errors.append(f"R1 last-four forbids Teacher supervision: {name}=True")
        lr_ablation = bool(candidate.get("R1_HSD_LR_ABLATION", False))
        if lr_ablation:
            expected_lr_ablation = {
                "R1_HSD_V1": True,
                "R1_FORMAL_ONLINE_V1": True,
                "R1_HSD_LR_REFERENCE_CONFIG": (
                    "configs/dinov1_s8_r1hard_hsd_v1_sem_ms148_online.py"
                ),
                "LR": 1e-4,
                "LR_FLOOR": 1e-5,
                "LR_POLICY": "linear_floor_two_stage",
                "LR_LINEAR_STAGE1_EPOCHS": 45,
                "LR_LINEAR_STAGE2_EPOCHS": 0,
            }
            for name, value in expected_lr_ablation.items():
                actual = candidate.get(name, "<MISSING>")
                if not _matches(actual, value):
                    errors.append(
                        f"R1 HSD LR-ablation field mismatch: {name}="
                        f"{actual!r}, expected={value!r}"
                    )
            reference_dino = reference.get("DINO", {})
            candidate_dino = candidate.get("DINO", {})
            if not isinstance(reference_dino, dict) or not isinstance(candidate_dino, dict):
                errors.append("R1 HSD LR-ablation requires dictionary DINO configs")
            else:
                expected_dino = dict(reference_dino)
                expected_dino["lr"] = 1e-4
                if candidate_dino != expected_dino:
                    errors.append(
                        "R1 HSD LR-ablation may change only DINO['lr']: "
                        f"actual={candidate_dino!r}, expected={expected_dino!r}"
                    )
        for name in ("LR", "LR_FLOOR", "MAX_EPOCH", "LOSS_SIZE"):
            if lr_ablation and name in {"LR", "LR_FLOOR"}:
                continue
            if candidate.get(name) != reference.get(name):
                errors.append(
                    f"R1 last-four changed protected protocol field {name}: "
                    f"{candidate.get(name)!r} != {reference.get(name)!r}"
                )
        batch_speed_probe = bool(candidate.get("R1_BATCH_SPEED_PROBE", False))
        reference_batch_size = int(reference.get("BATCH_SIZE", 16))
        candidate_batch_size = int(candidate.get("BATCH_SIZE", -1))
        if batch_speed_probe:
            declared_reference_batch_size = int(
                candidate.get("R1_REFERENCE_BATCH_SIZE", -1)
            )
            if declared_reference_batch_size != reference_batch_size:
                errors.append(
                    "R1 batch-speed probe reference mismatch: "
                    f"{declared_reference_batch_size} != {reference_batch_size}"
                )
            if candidate_batch_size <= reference_batch_size:
                errors.append(
                    "R1 batch-speed probe requires BATCH_SIZE greater than the "
                    f"reference: {candidate_batch_size} <= {reference_batch_size}"
                )
        elif candidate_batch_size != reference_batch_size:
            errors.append(
                "R1 last-four changed protected protocol field BATCH_SIZE: "
                f"{candidate_batch_size!r} != {reference_batch_size!r}"
            )
        report = {
            "schema": "r1_last4_pure_student_config_audit_v1",
            "status": "PASS" if not errors else "FAIL",
            "reference_config": str(DAGP_NDR_CONFIG_PATH),
            "candidate_config": str(getattr(cfg, "EXP_NAME", "")),
            "actual_differences": differences,
            "required_effective_differences": [record["field"] for record in differences],
            "errors": errors,
            "contract": {
                "source": "residual_pass1_37",
                "target": "strict_greater_than_0.5",
                "teacher_instantiated": False,
                "teacher_forward": False,
                "ema_update": False,
                "feature_keys": ["f9", "f10", "f11", "f12"],
                "batch_speed_probe": batch_speed_probe,
                "reference_batch_size": reference_batch_size,
                "effective_batch_size": candidate_batch_size,
                "lr_ablation": lr_ablation,
                "lr_scheduler_adjusted_for_batch": False,
            },
            "safety": {
                "creates_model": False,
                "creates_dataset": False,
                "starts_training": False,
                "starts_evaluation": False,
            },
        }
        if errors:
            raise RuntimeError(
                "R1 last-four pure-Student config audit failed: "
                + "; ".join(errors)
            )
        return report
    reference_cfg = load_config(REFERENCE_CONFIG_PATH)
    report = build_r1hard_linear_audit_report(reference_cfg, cfg)
    if report["status"] != "PASS":
        raise RuntimeError(
            "Hard-R1 pure-Student config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
