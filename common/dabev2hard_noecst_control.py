"""Contract shared by the DABE-v2-hard ECST/no-ECST control audit.

This module only compares resolved Python configurations.  It does not create
datasets, models, optimizers, training processes, or evaluation processes.
"""

from pathlib import Path

from common.utils import config_to_dict, load_config, make_jsonable


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
CONTROL_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)

BASELINE_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)
CONTROL_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

ALLOWED_DIFFS = frozenset(
    {
        "EXP_NAME",
        "USE_ECST",
        "USE_ECST_MINIMAL",
        "TEACHER_ROUTING_MODE",
        "DABE_CLEAN_USE_LEGACY_ECST_REGIONS",
    }
)

# USE_ECST_MINIMAL is deliberately restated as False by the control config, so
# it belongs to the whitelist but must not be an effective difference.
REQUIRED_EFFECTIVE_DIFFS = frozenset(
    {
        "EXP_NAME",
        "USE_ECST",
        "TEACHER_ROUTING_MODE",
        "DABE_CLEAN_USE_LEGACY_ECST_REGIONS",
    }
)

# Historical configs sometimes retain lowercase aliases.  Compare these in
# addition to every uppercase resolved field so a silent legacy override cannot
# bypass the single-variable audit.
TRAINING_RELATED_LOWERCASE_FIELDS = (
    "batch_size",
    "num_workers",
    "seed",
    "random_seed",
    "optimizer",
    "scheduler",
    "lr",
    "lr_floor",
    "max_epoch",
    "ema_weight",
    "head_type",
    "finetune_reset_epoch",
    "finetune_reset_teacher",
    "finetune_reset_lr",
    "fusion_orig_decay_epochs",
    "fusion_hold_fixed_weight",
    "teacher_fusion_pre_reset_epochs",
    "fusion_min_fixed_weight",
    "fusion_max_weight",
)

CONTROL_REQUIRED_VALUES = {
    "EXP_NAME": CONTROL_EXP_NAME,
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
    "DABE_CLEAN_DABE_V2_VERSION": "v2",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
    "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
    "STATIC_WEIGHT_MODE": "ones",
    "USE_ECST": False,
    "USE_ECST_MINIMAL": False,
    "TEACHER_ROUTING_MODE": "none",
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": False,
    "USE_DABE_CLEAN": True,
    "USE_DABE_PU": False,
    "STOP_AFTER_EPOCH": 0,
}

KEY_MATCH_FIELDS = (
    "DABE_CLEAN_STATIC_TARGET_SOURCE",
    "DABE_CLEAN_DABE_V2_ROOT",
    "DABE_CLEAN_DABE_V2_VERSION",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD",
    "DABE_CLEAN_STATIC_WEIGHT_MODE",
    "STATIC_WEIGHT_MODE",
    "TEACHER_FUSION_MODE",
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


def is_strict_control_config(cfg):
    """Identify only the named strict control; other no-ECST configs are inert."""

    return str(getattr(cfg, "EXP_NAME", "")) == CONTROL_EXP_NAME


def resolved_control_values(cfg):
    """Return all uppercase fields plus known lowercase training aliases."""

    values = config_to_dict(cfg)
    for name in TRAINING_RELATED_LOWERCASE_FIELDS:
        if hasattr(cfg, name):
            values[name] = make_jsonable(getattr(cfg, name))
    return values


def _difference_records(baseline_values, control_values):
    missing = object()
    records = []
    for name in sorted(set(baseline_values).union(control_values)):
        baseline_value = baseline_values.get(name, missing)
        control_value = control_values.get(name, missing)
        if baseline_value == control_value:
            continue
        records.append(
            {
                "field": name,
                "baseline": (
                    "<MISSING>" if baseline_value is missing else baseline_value
                ),
                "control": (
                    "<MISSING>" if control_value is missing else control_value
                ),
            }
        )
    return records


def build_control_audit_report(baseline_cfg, control_cfg):
    """Build a deterministic report without mutating either config module."""

    baseline_values = resolved_control_values(baseline_cfg)
    control_values = resolved_control_values(control_cfg)
    differences = _difference_records(baseline_values, control_values)
    difference_names = {record["field"] for record in differences}
    errors = []

    if str(baseline_values.get("EXP_NAME", "")) != BASELINE_EXP_NAME:
        errors.append(
            "baseline EXP_NAME mismatch: "
            f"{baseline_values.get('EXP_NAME')!r} != {BASELINE_EXP_NAME!r}"
        )
    unexpected = sorted(difference_names - ALLOWED_DIFFS)
    if unexpected:
        errors.append(f"unexpected effective config differences: {unexpected}")
    if difference_names != REQUIRED_EFFECTIVE_DIFFS:
        errors.append(
            "effective difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(REQUIRED_EFFECTIVE_DIFFS)}"
        )

    for name, expected in CONTROL_REQUIRED_VALUES.items():
        actual = control_values.get(name, "<MISSING>")
        if name == "DABE_CLEAN_DABE_V2_HARD_THRESHOLD":
            try:
                matches = abs(float(actual) - float(expected)) < 1e-12
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(
                f"control field mismatch: {name}={actual!r}, expected={expected!r}"
            )

    dabe_v2_root = str(
        control_values.get("DABE_CLEAN_DABE_V2_ROOT", "")
    ).strip()
    if not dabe_v2_root:
        errors.append("DABE_CLEAN_DABE_V2_ROOT must be explicit and non-empty")

    key_fields = {}
    for name in KEY_MATCH_FIELDS:
        baseline_value = baseline_values.get(name, "<MISSING>")
        control_value = control_values.get(name, "<MISSING>")
        matches = baseline_value == control_value
        key_fields[name] = {
            "baseline": baseline_value,
            "control": control_value,
            "matches": matches,
        }
        if not matches:
            errors.append(f"protected key field differs: {name}")

    return {
        "schema": "dabev2hard_ecst_vs_noecst_config_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "baseline_config": str(BASELINE_CONFIG_PATH),
        "control_config": str(CONTROL_CONFIG_PATH),
        "compared_uppercase_and_lowercase_field_count": len(
            set(baseline_values).union(control_values)
        ),
        "allowed_differences": sorted(ALLOWED_DIFFS),
        "required_effective_differences": sorted(REQUIRED_EFFECTIVE_DIFFS),
        "actual_differences": differences,
        "key_fields": key_fields,
        "errors": errors,
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def validate_strict_control_config(cfg):
    """Fail before dataset/model creation if the named control drifts."""

    if not is_strict_control_config(cfg):
        return None
    baseline_cfg = load_config(BASELINE_CONFIG_PATH)
    report = build_control_audit_report(baseline_cfg, cfg)
    if report["status"] != "PASS":
        raise RuntimeError(
            "DABE-v2-hard no-ECST strict-control config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
