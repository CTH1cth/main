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
GBSP_ABSMM_T058_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_"
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
GBSP_ABSMM_T058_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_"
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

COMMON_EXPECTED_OVERRIDES = {
    "DABEV2HARD_PURE_STUDENT": True,
    "R1_ONLY_CACHE_IO": True,
    "R1_HARD_RESIZE_MODE": "bilinear_align_corners_false",
    "R1_HARD_THRESHOLD": 0.5,
    "FINETUNE_RESET_TEACHER": False,
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
    "FINETUNE_RESET_EPOCH": 20,
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
        GBSP_ABSMM_T058_EXP_NAME,
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


def is_gbsp_absmm_t058_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == GBSP_ABSMM_T058_EXP_NAME


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
    gbsp_absmm_t058 = is_gbsp_absmm_t058_pure_student_config(candidate_cfg)
    gbsp_cf_brc_hc = is_gbsp_cf_brc_hc_pure_student_config(candidate_cfg)
    if gbsp_cf_brc_hc:
        expected_overrides = GBSP_CF_BRC_HC_EXPECTED_OVERRIDES
        required_values = GBSP_CF_BRC_HC_REQUIRED_VALUES
    elif gbsp_absmm_t058:
        expected_overrides = GBSP_ABSMM_T058_EXPECTED_OVERRIDES
        required_values = GBSP_ABSMM_T058_REQUIRED_VALUES
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
            "gbsp_cf_brc_hc_linear_pure_student_config_audit_v1"
            if gbsp_cf_brc_hc
            else (
                "gbsp_absmm_t058_linear_pure_student_config_audit_v1"
                if gbsp_absmm_t058
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
        ),
        "status": "PASS" if not errors else "FAIL",
        "reference_config": str(REFERENCE_CONFIG_PATH),
        "candidate_config": str(
            GBSP_CF_BRC_HC_CONFIG_PATH
            if gbsp_cf_brc_hc
            else (
                GBSP_ABSMM_T058_CONFIG_PATH
                if gbsp_absmm_t058
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
                    if (gbsp_absmm_t058 or gbsp_absmm_t063)
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
                if gbsp_absmm_t058
                else (
                    "strict_greater_than_0.63"
                    if gbsp_absmm_t063
                    else "strict_greater_than_0.5"
                )
            ),
            "student": (
                "dagp_safe_plus_ndr_v1"
                if dagp_ndr
                else "single_1x1_conv"
            ),
            "dagp_ndr_scale_epoch1": 1.0 if dagp_ndr else None,
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
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


def validate_r1hard_linear_pure_student_config(cfg):
    if not is_r1hard_linear_pure_student_config(cfg):
        return None
    if bool(getattr(cfg, "DABEV2HARD_R1_DBA", False)) or str(
        getattr(cfg, "EXP_NAME", "")
    ) == DBA_EXP_NAME:
        return validate_r1hard_dba_pure_student_config(cfg)
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
