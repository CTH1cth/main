"""Fail-fast configuration audit for the R1 decoder-isolation experiments."""

from __future__ import annotations

from pathlib import Path

from common.utils import config_to_dict, load_config


MAIN_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG_PATH = (
    MAIN_ROOT / "configs/dinov1_s8_r1hard_last4_linear_online.py"
)

ALLOWED_EXACT_FIELDS = frozenset(
    {
        "EXP_NAME",
        "R1_DECODER_ISOLATION_V1",
        "R1_DECODER_DIRECT_R1_37",
        "R1_DECODER_REFERENCE_CONFIG",
        "R1_LAST4_EXPERIMENT",
        "R1_HSD_V1",
        "R1_HSD_VARIANT",
        "DECODER_TYPE",
        "HEAD_TYPE",
        "DINO_FEATURE_MODE",
        "DINO_FEATURE_KEYS",
        "USE_MULTI_LEVEL_FEATURE",
        "USE_BASE_AUX_LOSS",
        "USE_NDR_COARSE_AUX",
        "LAMBDA_NDR_COARSE_AUX",
        "LAMBDA_BASE_AUX",
        "LAMBDA_BASE_AUX_AFTER_RESET",
        "USE_DETAIL",
        "DECODER_SUPERVISION_SOURCE",
        "GT_DIAGNOSTIC_SUPERVISION",
        "GT_DIAGNOSTIC_RESIZE_MODE",
        "GT_DIAGNOSTIC_STRICT_BINARY",
        "GT_DIAGNOSTIC_REFERENCE_CONFIG",
    }
)
ALLOWED_FIELD_PREFIXES = ("HSD_", "SCALE_LIFT_", "BCRD_", "R1_DECODER_")

PROTECTED_FIELDS = (
    "BATCH_SIZE",
    "VAL_BATCH_SIZE",
    "NUM_WORKERS",
    "SEED",
    "DINO",
    "BACKBONE_KEY",
    "DINO_FEATURE_MODE",
    "ONLINE_DINO_LAST4",
    "MULTI_LEVEL_LAYERS",
    "MULTI_LEVEL_FEATURE_TYPE",
    "MULTI_LEVEL_FEATURE_DTYPE",
    "R1_HARD_SOURCE_KEY",
    "R1_HARD_THRESHOLD",
    "DABE_CLEAN_DABE_V2_ROOT",
    "DABE_CLEAN_DABE_V2_SOURCE_KEY",
    "DABE_CLEAN_DABE_V2_HARD_THRESHOLD",
    "DABE_PU_DESPL_STATIC_START",
    "DABE_PU_DESPL_STATIC_END",
    "DABE_PU_DESPL_TEACHER_START",
    "DABE_PU_DESPL_TEACHER_END",
    "USE_TEACHER_BINARY_FULL_LOSS",
    "USE_TEACHER_SOFT_FULL_LOSS",
    "USE_TEACHER_CONF_LOSS",
    "FINETUNE_RESET_EPOCH",
    "FINETUNE_RESET_TEACHER",
    "FINETUNE_RESET_REBUILD_OPTIMIZER",
    "FINETUNE_RESET_REBUILD_SCHEDULER",
    "LR",
    "LR_FLOOR",
    "LR_LINEAR_STAGE1_EPOCHS",
    "MAX_EPOCH",
    "STOP_AFTER_EPOCH",
    "LOSS_SIZE",
)


def is_r1_decoder_isolation_config(cfg) -> bool:
    return bool(getattr(cfg, "R1_DECODER_ISOLATION_V1", False))


def _allowed_difference(name: str) -> bool:
    return name in ALLOWED_EXACT_FIELDS or name.startswith(ALLOWED_FIELD_PREFIXES)


def build_r1_decoder_isolation_audit_report(reference_cfg, candidate_cfg):
    reference = config_to_dict(reference_cfg)
    candidate = config_to_dict(candidate_cfg)
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

    errors = []
    unexpected = sorted(
        record["field"]
        for record in differences
        if not _allowed_difference(record["field"])
    )
    if unexpected:
        errors.append(f"unexpected effective config differences: {unexpected}")

    for name in PROTECTED_FIELDS:
        if candidate.get(name, missing) != reference.get(name, missing):
            errors.append(
                f"protected protocol field differs: {name}="
                f"{candidate.get(name, '<MISSING>')!r} != "
                f"{reference.get(name, '<MISSING>')!r}"
            )

    gt_diagnostic = bool(candidate.get("GT_DIAGNOSTIC_SUPERVISION", False))
    required = {
        "R1_DECODER_ISOLATION_V1": True,
        "R1_DECODER_DIRECT_R1_37": True,
        "BATCH_SIZE": 16,
        "R1_FORMAL_ONLINE_V1": True,
        "ONLINE_DINO_LAST4": True,
        "DINO_FEATURE_MODE": "online_last4",
        "DABEV2HARD_PURE_STUDENT": True,
        "DABEV2HARD_STATIC_ONLY": True,
        "R1_HARD_SOURCE_KEY": "residual_pass1_37",
        "R1_HARD_THRESHOLD": 0.5,
        "DABE_PU_DESPL_STATIC_START": 1.0,
        "DABE_PU_DESPL_STATIC_END": 1.0,
        "DABE_PU_DESPL_TEACHER_START": 0.0,
        "DABE_PU_DESPL_TEACHER_END": 0.0,
        "FINETUNE_RESET_EPOCH": 0,
        "FINETUNE_RESET_TEACHER": False,
        "FINETUNE_RESET_REBUILD_OPTIMIZER": False,
        "FINETUNE_RESET_REBUILD_SCHEDULER": False,
    }
    if gt_diagnostic:
        required.update(
            {
                "DECODER_SUPERVISION_SOURCE": "gt_hard_37",
                "GT_DIAGNOSTIC_RESIZE_MODE": "nearest_to_37_then_nearest_lift",
                "GT_DIAGNOSTIC_STRICT_BINARY": True,
            }
        )
        gt_reference_rel = str(
            candidate.get("GT_DIAGNOSTIC_REFERENCE_CONFIG", "")
        ).strip()
        if not gt_reference_rel:
            errors.append("GT_DIAGNOSTIC_REFERENCE_CONFIG must be explicit.")
            gt_pair_differences = []
        else:
            gt_reference_path = (MAIN_ROOT / gt_reference_rel).resolve()
            try:
                gt_reference_path.relative_to(MAIN_ROOT.resolve())
            except ValueError:
                errors.append(
                    "GT_DIAGNOSTIC_REFERENCE_CONFIG must stay inside MAIN_ROOT."
                )
                gt_pair_differences = []
            else:
                gt_reference = config_to_dict(load_config(gt_reference_path))
                gt_missing = object()
                gt_pair_differences = []
                for name in sorted(set(gt_reference).union(candidate)):
                    left = gt_reference.get(name, gt_missing)
                    right = candidate.get(name, gt_missing)
                    if left == right:
                        continue
                    gt_pair_differences.append(
                        {
                            "field": name,
                            "r1_reference": "<MISSING>" if left is gt_missing else left,
                            "gt_candidate": "<MISSING>" if right is gt_missing else right,
                        }
                    )
                allowed_gt_pair = {
                    "EXP_NAME",
                    "GT_DIAGNOSTIC_SUPERVISION",
                    "GT_DIAGNOSTIC_REFERENCE_CONFIG",
                    "DECODER_SUPERVISION_SOURCE",
                    "GT_DIAGNOSTIC_RESIZE_MODE",
                    "GT_DIAGNOSTIC_STRICT_BINARY",
                }
                unexpected_gt_pair = sorted(
                    record["field"]
                    for record in gt_pair_differences
                    if record["field"] not in allowed_gt_pair
                )
                if unexpected_gt_pair:
                    errors.append(
                        "GT-vs-R1 paired config differences are not label-only: "
                        f"{unexpected_gt_pair}"
                    )
    elif str(candidate.get("DECODER_SUPERVISION_SOURCE", "r1_hard_37")) != "r1_hard_37":
        gt_pair_differences = []
        errors.append(
            "Non-GT isolation config requires DECODER_SUPERVISION_SOURCE='r1_hard_37'."
        )
    else:
        gt_pair_differences = []
    for name, expected in required.items():
        if candidate.get(name, missing) != expected:
            errors.append(
                f"required isolation field mismatch: {name}="
                f"{candidate.get(name, '<MISSING>')!r}, expected={expected!r}"
            )
    for name in (
        "USE_TEACHER_BINARY_FULL_LOSS",
        "USE_TEACHER_SOFT_FULL_LOSS",
        "USE_TEACHER_CONF_LOSS",
    ):
        if bool(candidate.get(name, False)):
            errors.append(f"Teacher supervision must be disabled: {name}=True")

    decoder_type = str(candidate.get("DECODER_TYPE", "")).lower()
    expected_heads = {
        "last4_linear": "last4_linear_probe",
        "f12_scalelift": "f12_scalelift",
        "hsd_v1": "hsd_v1",
        "bcrd_sem_v1": "bcrd_sem_v1",
    }
    expected_head = expected_heads.get(decoder_type)
    if expected_head is None:
        errors.append(f"unsupported isolation DECODER_TYPE={decoder_type!r}")
    elif str(candidate.get("HEAD_TYPE", "")).lower() != expected_head:
        errors.append(
            f"decoder/head mismatch: {decoder_type!r}/"
            f"{candidate.get('HEAD_TYPE')!r}, expected head={expected_head!r}"
        )

    return {
        "schema": "r1_decoder_isolation_config_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "reference_config": str(REFERENCE_CONFIG_PATH),
        "candidate_config": str(candidate.get("EXP_NAME", "")),
        "allowed_differences": sorted(ALLOWED_EXACT_FIELDS)
        + [prefix + "*" for prefix in ALLOWED_FIELD_PREFIXES],
        "actual_differences": differences,
        "protected_fields": list(PROTECTED_FIELDS),
        "gt_pair_differences": gt_pair_differences,
        "errors": errors,
        "contract": {
            "target": (
                "strict_binary_training_gt_nearest_37_direct"
                if gt_diagnostic
                else "strict_binary_residual_pass1_37_direct"
            ),
            "gt_diagnostic": gt_diagnostic,
            "teacher_instantiated": False,
            "teacher_forward": False,
            "ema_update": False,
            "batch_size": 16,
            "optimizer_scheduler_epochs": "identical_to_reference",
        },
    }


def validate_r1_decoder_isolation_config(cfg):
    if not is_r1_decoder_isolation_config(cfg):
        return None
    report = build_r1_decoder_isolation_audit_report(
        load_config(REFERENCE_CONFIG_PATH), cfg
    )
    if report["status"] != "PASS":
        raise RuntimeError(
            "R1 decoder-isolation config audit failed: "
            + "; ".join(report["errors"])
        )
    return report
