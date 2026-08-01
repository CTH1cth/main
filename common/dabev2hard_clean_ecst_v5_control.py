"""Static contract for DABE-v2 Hard + unchanged A1 Clean-ECST v5.

The contract resolves and compares Python configuration modules only.  It
does not create a dataset, model, optimizer, training process, or evaluation
process.
"""

import math
from pathlib import Path

from common.utils import config_to_dict, load_config, make_jsonable


MAIN_ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
CONTROL_CONFIG_PATH = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_dabev2hard_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)

PARENT_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)
CONTROL_EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5"
)

EXPECTED_CONTROL_OVERRIDES = {
    "EXP_NAME": CONTROL_EXP_NAME,
    "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
    "DABE_CLEAN_DABE_V2_ROOT": (
        "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    ),
    "DABE_CLEAN_DABE_V2_VERSION": "v2",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
}
ALLOWED_DIFFS = frozenset(EXPECTED_CONTROL_OVERRIDES)
REQUIRED_EFFECTIVE_DIFFS = ALLOWED_DIFFS

# Anchor the comparison to the authoritative A1/v2-contrec experiment instead
# of merely trusting any module passed through --parent-config.
PARENT_REQUIRED_VALUES = {
    "EXP_NAME": PARENT_EXP_NAME,
    "USE_DABE_CLEAN": True,
    "USE_DABE_PU": False,
    "DABE_CLEAN_VERSION": "v2_contrec",
    "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION": "dabe_clean_v2_contrec",
    "DABE_CLEAN_TARGET_MODE": "dp",
    "DABE_CLEAN_SOURCE_ROOT": (
        "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/"
        "dinov1-s8"
    ),
    "DABE_CLEAN_ROOT": (
        "../datasets/cache/dabe_clean_v2_contrec_a1_residual_only_"
        "pseudo_cache/dinov1-s8"
    ),
    "DABE_BC_LAMBDA": 0.0,
    "DABE_CLEAN_ABLATION": "a1_residual_only_no_weak_bc_multiplier",
    "TEACHER_ROUTING_MODE": "clean_ecst",
    "USE_ECST": False,
    "USE_ECST_MINIMAL": False,
    "USE_ECST_CLEAN": True,
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": False,
    "ECST_CLEAN_VERSION": "v5_asym_continuous_recoverability",
    "ECST_CLEAN_START_EPOCH": 5,
    "ECST_CLEAN_RAMP_END_EPOCH": 10,
    "ECST_CLEAN_STOP_EPOCH": 21,
    "ECST_CLEAN_STRENGTH_MODE": "directional_continuous",
    "ECST_CLEAN_ERASE_STRENGTH": 2.5,
    "ECST_CLEAN_ADD_STRENGTH": 1.0,
    "ECST_CLEAN_RECOVERY_STRENGTH": 2.5,
    "ECST_CLEAN_USE_HARD_RING": False,
    "ECST_CLEAN_RECOVERY_MODE": "latent_rw_geometric_mean",
    "ECST_CLEAN_MEMORY_UPDATE_START_EPOCH": 1,
    "ECST_CLEAN_MEMORY_UPDATE_END_EPOCH": 20,
    "ECST_CLEAN_TEMPORAL_RHO": 0.9,
    "ECST_CLEAN_MIN_HISTORY": 3,
    "ECST_CLEAN_VARIANCE_TAU": 0.02,
    "ECST_CLEAN_USE_PREUPDATE_STATS": True,
    "ECST_CLEAN_MEMORY_DTYPE": "float16",
    "ECST_CLEAN_RESET_MEMORY_AT_FINETUNE_RESET": True,
    "TEACHER_FUSION_MODE": "dabe_clean_despl_sched",
    "USE_DABE_CLEAN_DESPL_SCHEDULE": True,
    "USE_DABE_PU_DESPL_SCHEDULE": False,
    "DABE_PU_DESPL_STAGE_START": 1,
    "DABE_PU_DESPL_STAGE_END": 20,
    "DABE_PU_DESPL_STATIC_START": 1.0,
    "DABE_PU_DESPL_STATIC_END": 0.05,
    "DABE_PU_DESPL_TEACHER_START": 0.0,
    "DABE_PU_DESPL_TEACHER_END": 0.95,
    "DABE_PU_DESPL_TEACHER_ONLY_START": 21,
    "EMA_WEIGHT": 0.99,
    "FINETUNE_RESET_EPOCH": 20,
    "FINETUNE_RESET_TIMING": "after_epoch",
    "FINETUNE_RESET_REBUILD_OPTIMIZER": True,
    "FINETUNE_RESET_REBUILD_SCHEDULER": True,
    "FINETUNE_RESET_GLOBAL_STEP": True,
    "FINETUNE_RESET_TEACHER": True,
    "FINETUNE_RESET_FORCE_LR_FLOOR": True,
    "FINETUNE_RESET_LR": 2e-5,
    "LR": 6e-4,
    "LR_FLOOR": 2e-5,
    "MAX_EPOCH": 45,
    "STOP_AFTER_EPOCH": 0,
    "HEAD_TYPE": "dagp_safe",
    "USE_DAGP_SAFE_HEAD": True,
    "USE_NDR_BRANCH": True,
    "USE_NDR_COARSE_AUX": True,
}

# The global difference check already protects every resolved field.  These
# fields/prefixes are surfaced separately so the audit makes the promised
# invariants (Clean evidence, history, schedule, EMA/reset, DAGP and NDR)
# directly inspectable.
PROTECTED_EXPLICIT_FIELDS = (
    "DABE_CLEAN_VERSION",
    "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION",
    "DABE_CLEAN_TARGET_MODE",
    "DABE_CLEAN_SOURCE_ROOT",
    "DABE_CLEAN_ROOT",
    "DABE_BC_LAMBDA",
    "DABE_CLEAN_ABLATION",
    "TEACHER_ROUTING_MODE",
    "USE_ECST",
    "USE_ECST_MINIMAL",
    "USE_ECST_CLEAN",
    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS",
    "TEACHER_FUSION_MODE",
    "USE_DABE_CLEAN_DESPL_SCHEDULE",
    "USE_DABE_PU_DESPL_SCHEDULE",
    "EMA_WEIGHT",
    "LR",
    "LR_FLOOR",
    "MAX_EPOCH",
    "STOP_AFTER_EPOCH",
    "HEAD_TYPE",
    "USE_DAGP_SAFE_HEAD",
    "USE_NDR_BRANCH",
    "USE_NDR_COARSE_AUX",
    "LAMBDA_BASE_AUX",
    "LAMBDA_BASE_AUX_AFTER_RESET",
)
PROTECTED_PREFIXES = (
    "ECST_CLEAN_",
    "DABE_PU_DESPL_",
    "FINETUNE_RESET_",
    "DAGP_SAFE_",
    "NDR_",
)

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
)


def is_dabev2hard_clean_ecst_v5_control(cfg):
    """Identify only the exact named DABE-v2 Hard/Clean-ECST v5 control."""

    return str(getattr(cfg, "EXP_NAME", "")) == CONTROL_EXP_NAME


def resolved_control_values(cfg):
    """Return uppercase config fields plus known lowercase training aliases."""

    values = config_to_dict(cfg)
    for name in TRAINING_RELATED_LOWERCASE_FIELDS:
        if hasattr(cfg, name):
            values[name] = make_jsonable(getattr(cfg, name))
    return values


def _difference_records(parent_values, control_values):
    missing = object()
    records = []
    for name in sorted(set(parent_values).union(control_values)):
        parent_value = parent_values.get(name, missing)
        control_value = control_values.get(name, missing)
        if parent_value == control_value:
            continue
        records.append(
            {
                "field": name,
                "parent": "<MISSING>" if parent_value is missing else parent_value,
                "control": (
                    "<MISSING>" if control_value is missing else control_value
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


def _protected_field_names(parent_values, control_values):
    names = set(PROTECTED_EXPLICIT_FIELDS)
    for name in set(parent_values).union(control_values):
        if any(name.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            names.add(name)
    return sorted(names - ALLOWED_DIFFS)


def build_control_audit_report(parent_cfg, control_cfg):
    """Compare the resolved child to the authoritative A1 parent."""

    parent_values = resolved_control_values(parent_cfg)
    control_values = resolved_control_values(control_cfg)
    differences = _difference_records(parent_values, control_values)
    difference_names = {record["field"] for record in differences}
    errors = []

    unexpected = sorted(difference_names - ALLOWED_DIFFS)
    if unexpected:
        errors.append(f"unexpected effective config differences: {unexpected}")
    if difference_names != REQUIRED_EFFECTIVE_DIFFS:
        errors.append(
            "effective difference set mismatch: "
            f"actual={sorted(difference_names)}, "
            f"required={sorted(REQUIRED_EFFECTIVE_DIFFS)}"
        )

    for name, expected in PARENT_REQUIRED_VALUES.items():
        actual = parent_values.get(name, "<MISSING>")
        if not _matches_expected(actual, expected):
            errors.append(
                f"authoritative parent field mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    for name, expected in EXPECTED_CONTROL_OVERRIDES.items():
        actual = control_values.get(name, "<MISSING>")
        if not _matches_expected(actual, expected):
            errors.append(
                f"control override mismatch: {name}={actual!r}, "
                f"expected={expected!r}"
            )

    protected_fields = {}
    for name in _protected_field_names(parent_values, control_values):
        parent_value = parent_values.get(name, "<MISSING>")
        control_value = control_values.get(name, "<MISSING>")
        matches = parent_value == control_value
        protected_fields[name] = {
            "parent": parent_value,
            "control": control_value,
            "matches": matches,
        }
        if not matches:
            errors.append(f"protected field differs: {name}")

    clean_root = str(control_values.get("DABE_CLEAN_ROOT", "")).strip()
    dabe_v2_root = str(
        control_values.get("DABE_CLEAN_DABE_V2_ROOT", "")
    ).strip()
    if not clean_root:
        errors.append("A1 v2-contrec DABE_CLEAN_ROOT must remain non-empty")
    if not dabe_v2_root:
        errors.append("DABE_CLEAN_DABE_V2_ROOT must be explicit and non-empty")

    return {
        "schema": "dabev2hard_clean_ecst_v5_config_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "parent_config": str(PARENT_CONFIG_PATH),
        "control_config": str(CONTROL_CONFIG_PATH),
        "compared_uppercase_and_lowercase_field_count": len(
            set(parent_values).union(control_values)
        ),
        "allowed_differences": sorted(ALLOWED_DIFFS),
        "required_effective_differences": sorted(REQUIRED_EFFECTIVE_DIFFS),
        "actual_differences": differences,
        "protected_fields": protected_fields,
        "supervision_semantics": {
            "static_bce_source": "dabe_v2_hard_68",
            "static_bce_formula": "1[p_dabe_68 > 0.5]",
            "clean_ecst_routing_target": "dabe_clean_target_68",
            "clean_ecst_routing_recoverability": (
                "dabe_clean_recoverability_68"
            ),
            "routing_evidence_cache": clean_root,
            "routing_evidence_unchanged_from_parent": True,
            "static_cache": dabe_v2_root,
            "single_variable_statement": (
                "Only static BCE changes to DABE-v2 Hard; A1 Clean target and "
                "continuous recoverability remain the Clean-ECST evidence."
            ),
        },
        "errors": errors,
        "safety": {
            "creates_model": False,
            "creates_dataset": False,
            "starts_training": False,
            "starts_evaluation": False,
        },
    }


def validate_dabev2hard_clean_ecst_v5_control(cfg):
    """Validate the named config; return ``None`` for every other config."""

    if not is_dabev2hard_clean_ecst_v5_control(cfg):
        return None
    parent_cfg = load_config(PARENT_CONFIG_PATH)
    report = build_control_audit_report(parent_cfg, cfg)
    if report["status"] != "PASS":
        raise RuntimeError(
            "DABE-v2 Hard + A1 Clean-ECST v5 config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
