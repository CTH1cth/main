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

COMMON_EXPECTED_OVERRIDES = {
    "DABEV2HARD_PURE_STUDENT": True,
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


def is_r1hard_linear_pure_student_config(cfg):
    """Identify either named Hard-R1 pure-Student experiment.

    The historical function name is retained because the training entry point
    already imports it as the pure-Student selector.
    """

    return str(getattr(cfg, "EXP_NAME", "")) in {
        EXP_NAME,
        DAGP_NDR_EXP_NAME,
        CVBR_V1_SECOND_RING_EXP_NAME,
    }


def is_r1hard_dagp_ndr_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == DAGP_NDR_EXP_NAME


def is_cvbr_v1_second_ring_pure_student_config(cfg):
    return str(getattr(cfg, "EXP_NAME", "")) == CVBR_V1_SECOND_RING_EXP_NAME


def _matches(actual, expected):
    if isinstance(expected, float):
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and abs(value - expected) < 1e-12
    return actual == expected


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
    if dagp_ndr:
        expected_overrides = DAGP_NDR_EXPECTED_OVERRIDES
        required_values = DAGP_NDR_REQUIRED_VALUES
    elif cvbr_v1:
        expected_overrides = CVBR_V1_SECOND_RING_EXPECTED_OVERRIDES
        required_values = CVBR_V1_SECOND_RING_REQUIRED_VALUES
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
            "r1hard_dagp_ndr_pure_student_config_audit_v1"
            if dagp_ndr
            else (
                "cvbr_v1_second_ring_linear_pure_student_config_audit_v1"
                if cvbr_v1
                else "r1hard_linear_pure_student_config_audit_v1"
            )
        ),
        "status": "PASS" if not errors else "FAIL",
        "reference_config": str(REFERENCE_CONFIG_PATH),
        "candidate_config": str(
            DAGP_NDR_CONFIG_PATH
            if dagp_ndr
            else (
                CVBR_V1_SECOND_RING_CONFIG_PATH if cvbr_v1 else CONFIG_PATH
            )
        ),
        "actual_differences": differences,
        "required_effective_differences": sorted(expected_difference_names),
        "errors": errors,
        "contract": {
            "source": (
                "v1_cvbr_second_ring_37" if cvbr_v1 else "residual_pass1_37"
            ),
            "resize": "bilinear_37_to_68_align_corners_false",
            "target": "strict_greater_than_0.5",
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
    reference_cfg = load_config(REFERENCE_CONFIG_PATH)
    report = build_r1hard_linear_audit_report(reference_cfg, cfg)
    if report["status"] != "PASS":
        raise RuntimeError(
            "Hard-R1 pure-Student config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
