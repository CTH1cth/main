"""Static configuration contract for the DABE-v2 Hard static-only ablation.

The checks in this module resolve and compare Python configurations only.  No
dataset, model, optimizer, training process, or evaluation process is created.
"""

import math
from pathlib import Path

from common.dabev2hard_noecst_control import resolved_control_values
from common.utils import load_config


MAIN_ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
STATIC_ONLY_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)

PARENT_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)
STATIC_ONLY_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)
R1HARD_LINEAR_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
R1HARD_DAGP_NDR_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_r1hard_dagp_uncgate_ndr_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)
CVBR_V1_SECOND_RING_LINEAR_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_cvbrv1_secondring_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5"
)

STATIC_ONLY_ENDPOINTS = {
    "DABE_PU_DESPL_STATIC_START": 1.0,
    "DABE_PU_DESPL_STATIC_END": 1.0,
    "DABE_PU_DESPL_TEACHER_START": 0.0,
    "DABE_PU_DESPL_TEACHER_END": 0.0,
}

EXPECTED_CHILD_OVERRIDES = {
    "EXP_NAME": STATIC_ONLY_EXP_NAME,
    "DABEV2HARD_STATIC_ONLY": True,
    **STATIC_ONLY_ENDPOINTS,
    "USE_TEACHER_BINARY_FULL_LOSS": False,
}

# USE_TEACHER_SOFT_FULL_LOSS is explicitly restated as False in the child but
# is already False in the parent, so it is protected rather than an effective
# difference.  The four endpoint names remain whitelisted even when a parent
# endpoint already equals its static-only value.
ALLOWED_DIFFS = frozenset(EXPECTED_CHILD_OVERRIDES)

STATIC_ONLY_REQUIRED_VALUES = {
    **EXPECTED_CHILD_OVERRIDES,
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
    "DABE_CLEAN_DABE_V2_VERSION": "v2",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
    "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
    "STATIC_WEIGHT_MODE": "ones",
    "USE_DABE_CLEAN": True,
    "USE_DABE_PU": False,
    "USE_DABE_PU_STATIC_LOSS": True,
    "USE_DABE_CLEAN_DESPL_SCHEDULE": True,
    "USE_DABE_PU_DESPL_SCHEDULE": False,
    "LAMBDA_DABE_PU_STATIC": 1.0,
    "TEACHER_FUSION_MODE": "dabe_clean_despl_sched",
    "USE_ECST": False,
    "USE_ECST_MINIMAL": False,
    "TEACHER_ROUTING_MODE": "none",
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": False,
    "USE_TEACHER_BINARY_FULL_LOSS": False,
    "USE_TEACHER_SOFT_FULL_LOSS": False,
    "USE_TEACHER_CONF_LOSS": False,
    "MAX_EPOCH": 45,
    "FINETUNE_RESET_EPOCH": 20,
    "STOP_AFTER_EPOCH": 0,
}

# These fields make the preserved comparison protocol visible in the report.
# Every one must remain byte-for-byte equal to the strict no-ECST parent.
PROTECTED_KEY_FIELDS = (
    "DABE_CLEAN_DABE_V2_ROOT",
    "DABE_CLEAN_STATIC_TARGET_SOURCE",
    "DABE_CLEAN_DABE_V2_VERSION",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD",
    "DABE_CLEAN_STATIC_WEIGHT_MODE",
    "STATIC_WEIGHT_MODE",
    "TEACHER_FUSION_MODE",
    "TEACHER_TARGET_MODE",
    "P_INIT_MODE",
    "EMA_WEIGHT",
    "FINETUNE_RESET_EPOCH",
    "FINETUNE_RESET_TEACHER",
    "FINETUNE_RESET_LR",
    "LR",
    "LR_FLOOR",
    "MAX_EPOCH",
    "HEAD_TYPE",
    "DAGP_SAFE_TOPK",
    "DAGP_SAFE_TAU",
    "DAGP_SAFE_ALPHA_MAX",
    "DAGP_SAFE_GAMMA_MAX",
    "NDR_VERSION",
    "NDR_BETA_MAX",
    "LAMBDA_BASE_AUX",
    "LAMBDA_BASE_AUX_AFTER_RESET",
)


def is_dabev2hard_static_only_config(cfg):
    """Identify only the named static-only experiment.

    Identification is intentionally based on the exact experiment name.  This
    ensures a named config whose selector was accidentally removed still enters
    validation and fails, instead of silently falling back to Teacher handover.
    """

    return str(getattr(cfg, "EXP_NAME", "")) in {
        STATIC_ONLY_EXP_NAME,
        R1HARD_LINEAR_EXP_NAME,
        R1HARD_DAGP_NDR_EXP_NAME,
        CVBR_V1_SECOND_RING_LINEAR_EXP_NAME,
    }


def _difference_records(parent_values, static_only_values):
    missing = object()
    records = []
    for name in sorted(set(parent_values).union(static_only_values)):
        parent_value = parent_values.get(name, missing)
        static_only_value = static_only_values.get(name, missing)
        if parent_value == static_only_value:
            continue
        records.append(
            {
                "field": name,
                "parent": "<MISSING>" if parent_value is missing else parent_value,
                "static_only": (
                    "<MISSING>"
                    if static_only_value is missing
                    else static_only_value
                ),
            }
        )
    return records


def _matches_expected(actual, expected):
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            return False
        return math.isfinite(actual_float) and abs(actual_float - expected) < 1e-12
    return actual == expected


def build_static_only_audit_report(parent_cfg, static_only_cfg):
    """Compare the resolved child against its strict no-ECST parent."""

    parent_values = resolved_control_values(parent_cfg)
    static_only_values = resolved_control_values(static_only_cfg)
    differences = _difference_records(parent_values, static_only_values)
    difference_names = {record["field"] for record in differences}
    errors = []

    if str(parent_values.get("EXP_NAME", "")) != PARENT_EXP_NAME:
        errors.append(
            "parent EXP_NAME mismatch: "
            f"{parent_values.get('EXP_NAME')!r} != {PARENT_EXP_NAME!r}"
        )

    unexpected = sorted(difference_names - ALLOWED_DIFFS)
    if unexpected:
        errors.append(f"unexpected effective config differences: {unexpected}")

    # Derive the exact expected effective-difference set from the resolved
    # parent.  At present STATIC_START and TEACHER_START already equal the
    # requested values, while the other two endpoints change effectively.
    missing = object()
    expected_effective_differences = {
        name
        for name, expected in EXPECTED_CHILD_OVERRIDES.items()
        if parent_values.get(name, missing) != expected
    }
    if difference_names != expected_effective_differences:
        errors.append(
            "effective difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(expected_effective_differences)}"
        )

    for name, expected in STATIC_ONLY_REQUIRED_VALUES.items():
        actual = static_only_values.get(name, "<MISSING>")
        if not _matches_expected(actual, expected):
            errors.append(
                f"static-only field mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    dabe_v2_root = str(
        static_only_values.get("DABE_CLEAN_DABE_V2_ROOT", "")
    ).strip()
    if not dabe_v2_root:
        errors.append("DABE_CLEAN_DABE_V2_ROOT must be explicit and non-empty")

    protected_fields = {}
    for name in PROTECTED_KEY_FIELDS:
        parent_value = parent_values.get(name, "<MISSING>")
        static_only_value = static_only_values.get(name, "<MISSING>")
        matches = parent_value == static_only_value
        protected_fields[name] = {
            "parent": parent_value,
            "static_only": static_only_value,
            "matches": matches,
        }
        if not matches:
            errors.append(f"protected key field differs: {name}")

    # Disable only Teacher supervision switches already present in the parent.
    # EMA_WEIGHT is deliberately protected above, because retaining an EMA
    # shadow for diagnostics is not Teacher supervision when all Teacher loss
    # coefficients and branches are exactly zero.
    teacher_loss_switches = {}
    for name, parent_value in sorted(parent_values.items()):
        if not (name.startswith("USE_TEACHER_") and name.endswith("_LOSS")):
            continue
        actual = static_only_values.get(name, "<MISSING>")
        teacher_loss_switches[name] = {
            "parent": parent_value,
            "static_only": actual,
            "disabled": actual is False,
        }
        if actual is not False:
            errors.append(f"Teacher loss switch must be False: {name}={actual!r}")

    return {
        "schema": "dabev2hard_static_only_config_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "parent_config": str(PARENT_CONFIG_PATH),
        "static_only_config": str(STATIC_ONLY_CONFIG_PATH),
        "compared_uppercase_and_lowercase_field_count": len(
            set(parent_values).union(static_only_values)
        ),
        "allowed_differences": sorted(ALLOWED_DIFFS),
        "required_effective_differences": sorted(
            expected_effective_differences
        ),
        "actual_differences": differences,
        "static_only_endpoints": dict(STATIC_ONLY_ENDPOINTS),
        "teacher_loss_switches": teacher_loss_switches,
        "protected_key_fields": protected_fields,
        "errors": errors,
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def validate_dabev2hard_static_only_config(cfg):
    """Validate the named config, returning ``None`` for every other config."""

    if not is_dabev2hard_static_only_config(cfg):
        return None
    if str(getattr(cfg, "EXP_NAME", "")) in {
        R1HARD_LINEAR_EXP_NAME,
        R1HARD_DAGP_NDR_EXP_NAME,
        CVBR_V1_SECOND_RING_LINEAR_EXP_NAME,
    }:
        from common.r1hard_linear_pure_student import (
            validate_r1hard_linear_pure_student_config,
        )

        return validate_r1hard_linear_pure_student_config(cfg)
    parent_cfg = load_config(PARENT_CONFIG_PATH)
    report = build_static_only_audit_report(parent_cfg, cfg)
    if report["status"] != "PASS":
        raise RuntimeError(
            "DABE-v2 Hard static-only config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
