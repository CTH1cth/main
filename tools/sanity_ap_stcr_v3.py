import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKDIR_ROOT = (PROJECT_ROOT / "workdir").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ap_stcr import (  # noqa: E402
    AnchorPropagatedSemanticTemporalCorrection,
    build_ap_stcr_diagnostic_payload,
    render_ap_stcr_diagnostic,
    strict_teacher_binary,
)
from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import load_config  # noqa: E402
from model import build_seg_head  # noqa: E402
from tools.render_ap_stcr_visual import _validate_payload  # noqa: E402
from train import (  # noqa: E402
    build_ap_stcr_segmentation_group,
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    validate_ap_stcr_config,
)


V3_VERSION = "ap_stcr_v3_soft_disagreement_bounded_continuation"
REQUIRED_BATCH_FIELDS = {
    "sample_index",
    "dataset",
    "stem",
    "image_path",
    "feature",
    "pu_target_soft",
    "pu_target_soft_37",
    "pu_bg_anchor_37",
    "image_68",
}
FORBIDDEN_GT_FIELDS = {
    "gt",
    "gt_path",
    "ground_truth",
    "ground_truth_path",
    "mask",
    "mask_path",
}
REQUIRED_V3_RESULT_FIELDS = {
    "soft_correction_37",
    "soft_deviation_37",
    "fused_support_37",
    "support_deficiency_37",
    "soft_inertia_37",
    "teacher_soft_37",
    "semantic_support_37",
    "temporal_support_37",
    "local_acceptance_37",
    "effective_teacher_weight_37",
    "effective_teacher_weight_68",
    "mixed_target_68",
    "source_conflict_37",
    "history_mean_37",
    "history_count",
    "history_valid",
}


def _resolve_config_path(value):
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"--config must be inside {PROJECT_ROOT}, got {path}."
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"--config not found: {path}")
    return path


def _prepare_output_dir(value):
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(WORKDIR_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"--out must be inside {WORKDIR_ROOT}, got {path}."
        ) from error
    if path == WORKDIR_ROOT:
        raise RuntimeError("--out must name a dedicated directory under workdir.")
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"--out is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty --out directory: {path}"
            )
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested, but CUDA is unavailable.")
    return torch.device(value)


def _assert_close(name, actual, expected, atol=1e-6, rtol=1e-6):
    actual = torch.as_tensor(actual).detach().float().cpu()
    expected = torch.as_tensor(expected).detach().float().cpu()
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape mismatch: {list(actual.shape)} != "
            f"{list(expected.shape)}."
        )
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        maximum = float((actual - expected).abs().max().item())
        raise RuntimeError(
            f"{name} mismatch: max_abs_error={maximum:.8g}, "
            f"atol={atol}, rtol={rtol}."
        )
    return float((actual - expected).abs().max().item())


def _assert_finite_range(name, value, lower, upper, tolerance=1e-6):
    tensor = torch.as_tensor(value).detach().float()
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} must be non-empty and finite.")
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if minimum < lower - tolerance or maximum > upper + tolerance:
        raise RuntimeError(
            f"{name} outside [{lower},{upper}]: min={minimum}, max={maximum}."
        )


def _tensor_stats(value):
    tensor = torch.as_tensor(value).detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
    }


def _history_snapshot(ap_stcr):
    state = ap_stcr.history_bank.state_dict()
    return {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in state.items()
    }


def _assert_history_unchanged(name, before, after):
    if before.keys() != after.keys():
        raise RuntimeError(f"{name}: history schema changed during read.")
    for field in before:
        left = before[field]
        right = after[field]
        if torch.is_tensor(left):
            if not torch.equal(left, right):
                raise RuntimeError(
                    f"{name}: history field {field} mutated during build_batch."
                )
        elif left != right:
            raise RuntimeError(
                f"{name}: history field {field} mutated during build_batch."
            )


def _schedule_endpoint_audit(cfg):
    ap_config = dict(cfg.AP_STCR)
    stage_start = int(cfg.DABE_PU_DESPL_STAGE_START)
    stage_end = int(cfg.DABE_PU_DESPL_STAGE_END)
    continuation_start = int(ap_config["post_reset_teacher_start_epoch"])
    continuation_end = int(ap_config["post_reset_teacher_end_epoch"])
    max_epoch = int(cfg.MAX_EPOCH)
    epochs = sorted(
        {
            stage_start,
            stage_end,
            continuation_start,
            continuation_end,
            max_epoch,
            *range(20, 27),
        }
    )
    rows = []
    for epoch in epochs:
        fixed, teacher = get_dabe_pu_despl_schedule(epoch, cfg)
        if abs(float(fixed) + float(teacher) - 1.0) > 1e-8:
            raise RuntimeError(
                f"Schedule weights do not sum to one at epoch={epoch}."
            )
        if epoch <= stage_end:
            progress = float(epoch - stage_start) / float(
                stage_end - stage_start
            )
            progress = max(0.0, min(1.0, progress))
            expected_ratio = float(cfg.DABE_PU_DESPL_TEACHER_START) + (
                float(cfg.DABE_PU_DESPL_TEACHER_END)
                - float(cfg.DABE_PU_DESPL_TEACHER_START)
            ) * progress
        else:
            progress = float(epoch - continuation_start) / float(
                continuation_end - continuation_start
            )
            progress = max(0.0, min(1.0, progress))
            expected_ratio = float(
                ap_config["post_reset_teacher_start_ratio"]
            ) + (
                float(ap_config["post_reset_teacher_end_ratio"])
                - float(ap_config["post_reset_teacher_start_ratio"])
            ) * progress
        error = abs(float(teacher) - expected_ratio)
        if error > 1e-12:
            raise RuntimeError(
                f"Schedule endpoint mismatch at epoch={epoch}: "
                f"{teacher} != {expected_ratio}."
            )
        rows.append(
            {
                "epoch": epoch,
                "fixed_ratio": float(fixed),
                "teacher_ratio": float(teacher),
                "expected_teacher_ratio": expected_ratio,
                "abs_error": error,
            }
        )
    audited_epochs = {row["epoch"] for row in rows}
    missing_continuation_epochs = sorted(set(range(20, 27)) - audited_epochs)
    if missing_continuation_epochs:
        raise RuntimeError(
            "Schedule audit omitted continuation epochs: "
            f"{missing_continuation_epochs}."
        )
    return {
        "rows": rows,
        "required_continuation_epochs": list(range(20, 27)),
        "all_required_continuation_epochs_audited": True,
    }


def _require_real_batch(batch, max_samples):
    missing = sorted(REQUIRED_BATCH_FIELDS.difference(batch))
    if missing:
        raise RuntimeError(f"Real cached batch is missing fields: {missing}")
    lower_names = {str(name).lower() for name in batch}
    forbidden = sorted(FORBIDDEN_GT_FIELDS.intersection(lower_names))
    forbidden.extend(sorted(name for name in lower_names if name.startswith("gt_")))
    if forbidden:
        raise RuntimeError(
            f"Sanity batch must not contain training GT fields: {forbidden}"
        )
    batch_size = int(torch.as_tensor(batch["sample_index"]).numel())
    if not 1 <= batch_size <= max_samples:
        raise RuntimeError(
            f"Unexpected real batch size={batch_size}, max_samples={max_samples}."
        )
    return batch_size


def _require_v3_result(result, fixed_37, fixed_68):
    if str(result.get("version", "")) != V3_VERSION:
        raise RuntimeError(
            f"Expected v3 result version={V3_VERSION!r}, "
            f"got {result.get('version')!r}."
        )
    missing = sorted(REQUIRED_V3_RESULT_FIELDS.difference(result))
    if missing:
        raise RuntimeError(f"AP-STCR v3 result is missing fields: {missing}")
    expected_37 = tuple(fixed_37.shape)
    expected_68 = tuple(fixed_68.shape)
    for name in (
        "soft_correction_37",
        "soft_deviation_37",
        "fused_support_37",
        "support_deficiency_37",
        "soft_inertia_37",
        "teacher_soft_37",
        "semantic_support_37",
        "temporal_support_37",
        "local_acceptance_37",
        "effective_teacher_weight_37",
        "source_conflict_37",
        "history_mean_37",
    ):
        value = result[name]
        if tuple(value.shape) != expected_37:
            raise RuntimeError(
                f"{name} shape mismatch: {list(value.shape)} != "
                f"{list(expected_37)}."
            )
        if value.requires_grad:
            raise RuntimeError(f"{name} must be detached.")
    for name in ("effective_teacher_weight_68", "mixed_target_68"):
        value = result[name]
        if tuple(value.shape) != expected_68:
            raise RuntimeError(
                f"{name} shape mismatch: {list(value.shape)} != "
                f"{list(expected_68)}."
            )
        if value.requires_grad:
            raise RuntimeError(f"{name} must be detached.")

    _assert_finite_range("soft_correction_37", result["soft_correction_37"], -1.0, 1.0)
    for name in (
        "soft_deviation_37",
        "fused_support_37",
        "support_deficiency_37",
        "soft_inertia_37",
        "teacher_soft_37",
        "semantic_support_37",
        "temporal_support_37",
        "local_acceptance_37",
        "effective_teacher_weight_37",
        "effective_teacher_weight_68",
        "mixed_target_68",
        "source_conflict_37",
    ):
        _assert_finite_range(name, result[name], 0.0, 1.0)


def _v3_formula_audit(result, fixed_37, ap_config):
    teacher_soft = result["teacher_soft_37"]
    expected_correction = teacher_soft - fixed_37
    expected_deviation = expected_correction.abs()
    expected_deficiency = 1.0 - result["fused_support_37"]
    expected_inertia = expected_deviation * expected_deficiency
    strength = float(ap_config["soft_rejection_strength"])
    min_acceptance = float(ap_config["min_local_acceptance"])
    expected_acceptance = (1.0 - strength * expected_inertia).clamp(
        min=min_acceptance,
        max=1.0,
    )
    errors = {
        "soft_correction": _assert_close(
            "v3 soft correction",
            result["soft_correction_37"],
            expected_correction,
        ),
        "soft_deviation": _assert_close(
            "v3 soft deviation",
            result["soft_deviation_37"],
            expected_deviation,
        ),
        "support_deficiency": _assert_close(
            "v3 support deficiency",
            result["support_deficiency_37"],
            expected_deficiency,
        ),
        "soft_inertia": _assert_close(
            "v3 soft inertia",
            result["soft_inertia_37"],
            expected_inertia,
        ),
        "local_acceptance": _assert_close(
            "v3 bounded local acceptance",
            result["local_acceptance_37"],
            expected_acceptance,
        ),
    }
    observed_min = float(result["local_acceptance_37"].min().item())
    if observed_min < min_acceptance - 1e-6:
        raise RuntimeError(
            f"v3 acceptance floor violated: {observed_min} < {min_acceptance}."
        )
    return {
        "max_abs_errors": errors,
        "formula_dependencies": [
            "teacher_soft_37",
            "fixed_pseudo_37",
            "fused_support_37",
            "soft_rejection_strength",
            "min_local_acceptance",
        ],
        "source_conflict_usage": "diagnostic_only",
        "analytic_endpoints": {
            "zero_deviation_acceptance": 1.0,
            "zero_deficiency_acceptance": 1.0,
            "unit_deviation_unit_deficiency_acceptance": max(
                min_acceptance,
                1.0 - strength,
            ),
            "configured_min_local_acceptance": min_acceptance,
        },
    }


def _alpha_endpoint_audit(ap_stcr, fixed_68, teacher_binary_68, acceptance_37):
    at_zero = ap_stcr.build_target(
        fixed_68,
        teacher_binary_68,
        0.0,
        acceptance_37,
    )
    at_one = ap_stcr.build_target(
        fixed_68,
        teacher_binary_68,
        1.0,
        acceptance_37,
    )
    zero_target_error = _assert_close(
        "alpha=0 mixed target",
        at_zero["mixed_target_68"],
        fixed_68,
    )
    zero_weight_error = _assert_close(
        "alpha=0 effective weight",
        at_zero["effective_teacher_weight_68"],
        torch.zeros_like(fixed_68),
    )
    acceptance_68 = F.interpolate(
        acceptance_37,
        size=fixed_68.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    expected_at_one = (
        (1.0 - acceptance_68) * fixed_68
        + acceptance_68 * teacher_binary_68
    ).clamp(0.0, 1.0)
    one_weight_error = _assert_close(
        "alpha=1 effective weight",
        at_one["effective_teacher_weight_68"],
        acceptance_68,
    )
    one_target_error = _assert_close(
        "alpha=1 mixed target",
        at_one["mixed_target_68"],
        expected_at_one,
    )
    full_acceptance = ap_stcr.build_target(
        fixed_68,
        teacher_binary_68,
        1.0,
        torch.ones_like(acceptance_37),
    )
    full_acceptance_weight_error = _assert_close(
        "alpha=1 full-acceptance effective weight",
        full_acceptance["effective_teacher_weight_68"],
        torch.ones_like(fixed_68),
    )
    full_acceptance_target_error = _assert_close(
        "alpha=1 full-acceptance target",
        full_acceptance["mixed_target_68"],
        teacher_binary_68,
    )
    return {
        "alpha_zero_target_max_abs_error": zero_target_error,
        "alpha_zero_weight_max_abs_error": zero_weight_error,
        "alpha_one_weight_max_abs_error": one_weight_error,
        "alpha_one_target_max_abs_error": one_target_error,
        "alpha_one_full_acceptance_weight_max_abs_error": (
            full_acceptance_weight_error
        ),
        "alpha_one_full_acceptance_teacher_binary_max_abs_error": (
            full_acceptance_target_error
        ),
    }


def _analytic_acceptance_endpoint_audit(ap_stcr, template, history_valid):
    zeros = torch.zeros_like(template)
    ones = torch.ones_like(template)
    zero_deviation = ap_stcr.compute_soft_disagreement_acceptance(
        semantic_support=zeros,
        temporal_support=zeros,
        history_valid=history_valid,
        soft_deviation=zeros,
    )
    unit_deviation_zero_support = (
        ap_stcr.compute_soft_disagreement_acceptance(
            semantic_support=zeros,
            temporal_support=zeros,
            history_valid=history_valid,
            soft_deviation=ones,
        )
    )
    unit_deviation_unit_support = (
        ap_stcr.compute_soft_disagreement_acceptance(
            semantic_support=ones,
            temporal_support=ones,
            history_valid=history_valid,
            soft_deviation=ones,
        )
    )
    min_acceptance = float(ap_stcr.min_local_acceptance)
    errors = {
        "deviation_zero_acceptance_one": _assert_close(
            "analytic endpoint deviation=0",
            zero_deviation["local_acceptance_37"],
            ones,
        ),
        "deviation_one_support_zero_acceptance_floor": _assert_close(
            "analytic endpoint deviation=1 support=0",
            unit_deviation_zero_support["local_acceptance_37"],
            torch.full_like(template, min_acceptance),
        ),
        "deviation_one_support_one_acceptance_one": _assert_close(
            "analytic endpoint deviation=1 support=1",
            unit_deviation_unit_support["local_acceptance_37"],
            ones,
        ),
    }
    return {
        "compute_api": "compute_soft_disagreement_acceptance",
        "uses_real_batch_shape_and_device": True,
        "min_local_acceptance": min_acceptance,
        "max_abs_errors": errors,
    }


def _sample_rows(batch):
    indices = torch.as_tensor(batch["sample_index"]).flatten().tolist()
    return [
        {
            "sample_index": int(indices[index]),
            "dataset": str(batch["dataset"][index]),
            "stem": str(batch["stem"][index]),
            "image_path": str(batch["image_path"][index]),
        }
        for index in range(len(indices))
    ]


@torch.no_grad()
def _run_sanity(config_path, max_samples, device, out_root):
    if torch.is_grad_enabled():
        raise RuntimeError("AP-STCR v3 sanity must run with gradients disabled.")

    cfg = load_config(config_path)
    validate_ap_stcr_config(cfg)
    ap_config = dict(cfg.AP_STCR)
    if str(ap_config.get("version", "")) != V3_VERSION:
        raise RuntimeError(
            f"This command accepts AP-STCR v3 only; got "
            f"{ap_config.get('version')!r}."
        )
    schedule_audit = _schedule_endpoint_audit(cfg)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    dataset = CachedTrainDataset(cfg, max_samples=max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Cached training dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=min(max_samples, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch_size = _require_real_batch(batch, max_samples)

    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = copy.deepcopy(student).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.eval()
    teacher.eval()
    set_model_epoch(student, 1)
    set_model_epoch(teacher, 1)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    if image_68 is None:
        raise RuntimeError("AP-STCR v3 real forward requires image_68.")
    student_out = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
    )
    teacher_out = forward_seg_head(
        teacher,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=False,
    )
    student_logits = resize_logits_for_loss(extract_logits(student_out), cfg)
    teacher_logits = resize_logits_for_loss(extract_logits(teacher_out), cfg)
    student_prob = student_logits.sigmoid()
    teacher_soft_68 = teacher_logits.sigmoid()
    teacher_binary_68 = strict_teacher_binary(
        teacher_soft_68,
        threshold=float(ap_config["teacher_binary_threshold"]),
    )
    for name, value in {
        "student_logits": student_logits,
        "teacher_logits": teacher_logits,
        "student_prob": student_prob,
        "teacher_soft_68": teacher_soft_68,
    }.items():
        if value.requires_grad or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"{name} must be finite and detached.")

    fixed_37 = batch["pu_target_soft_37"].to(device).float()
    fixed_68 = batch["pu_target_soft"].to(device).float()
    background_37 = batch["pu_bg_anchor_37"].to(device).float()
    dino_features = batch["feature"].to(device).float()
    sample_indices = batch["sample_index"]
    datasets = batch["dataset"]
    stems = batch["stem"]

    ap_stcr = AnchorPropagatedSemanticTemporalCorrection(
        ap_config,
        dataset.keys,
    )
    _, alpha_epoch_1 = get_dabe_pu_despl_schedule(1, cfg)
    before_first = _history_snapshot(ap_stcr)
    first_result = ap_stcr.build_batch(
        dino_features=dino_features,
        fixed_pseudo_37=fixed_37,
        fixed_pseudo_68=fixed_68,
        dabe_background_seed_37=background_37,
        teacher_soft_68=teacher_soft_68,
        teacher_binary_68=teacher_binary_68,
        global_teacher_ratio=alpha_epoch_1,
        sample_indices=sample_indices,
        datasets=datasets,
        stems=stems,
    )
    after_first = _history_snapshot(ap_stcr)
    _assert_history_unchanged("first build_batch", before_first, after_first)
    _require_v3_result(first_result, fixed_37, fixed_68)
    first_fused_error = _assert_close(
        "first-pass fused support",
        first_result["fused_support_37"],
        first_result["semantic_support_37"],
    )
    if bool(first_result["history_valid"].any().item()):
        raise RuntimeError("First AP-STCR pass unexpectedly consumed current history.")
    if bool((first_result["history_count"] != 0).any().item()):
        raise RuntimeError("First AP-STCR pass must observe zero history count.")

    ap_stcr.update_history(
        sample_indices,
        datasets,
        stems,
        first_result["teacher_soft_37"],
        epoch=1,
    )
    before_second = _history_snapshot(ap_stcr)
    set_model_epoch(student, 2)
    set_model_epoch(teacher, 2)
    _, alpha_epoch_2 = get_dabe_pu_despl_schedule(2, cfg)
    second_result = ap_stcr.build_batch(
        dino_features=dino_features,
        fixed_pseudo_37=fixed_37,
        fixed_pseudo_68=fixed_68,
        dabe_background_seed_37=background_37,
        teacher_soft_68=teacher_soft_68,
        teacher_binary_68=teacher_binary_68,
        global_teacher_ratio=alpha_epoch_2,
        sample_indices=sample_indices,
        datasets=datasets,
        stems=stems,
    )
    after_second = _history_snapshot(ap_stcr)
    _assert_history_unchanged("second build_batch", before_second, after_second)
    _require_v3_result(second_result, fixed_37, fixed_68)
    second_expected_fused = 0.5 * (
        second_result["semantic_support_37"]
        + second_result["temporal_support_37"]
    )
    second_fused_error = _assert_close(
        "second-pass fused support",
        second_result["fused_support_37"],
        second_expected_fused,
    )
    if not bool(second_result["history_valid"].all().item()):
        raise RuntimeError("Second AP-STCR pass did not consume the prior snapshot.")
    if not bool((second_result["history_count"] == 1).all().item()):
        raise RuntimeError("Second AP-STCR pass must observe exactly one past item.")
    history_error = _assert_close(
        "past-only history mean",
        second_result["history_mean_37"],
        first_result["teacher_soft_37"],
        atol=5e-4,
        rtol=0.0,
    )

    formula_audit = _v3_formula_audit(second_result, fixed_37, ap_config)
    analytic_endpoint_audit = _analytic_acceptance_endpoint_audit(
        ap_stcr,
        second_result["semantic_support_37"],
        second_result["history_valid"],
    )
    alpha_audit = _alpha_endpoint_audit(
        ap_stcr,
        fixed_68,
        teacher_binary_68,
        second_result["local_acceptance_37"],
    )
    segmentation = build_ap_stcr_segmentation_group(
        cfg,
        2,
        student_out,
        student_logits,
        second_result["mixed_target_68"],
    )
    loss_fields = {
        name: float(value.detach().cpu().item())
        for name, value in segmentation.items()
        if torch.is_tensor(value)
    }
    if any(not torch.isfinite(torch.tensor(value)).item() for value in loss_fields.values()):
        raise RuntimeError("Segmentation loss audit contains NaN/Inf.")

    diagnostic = build_ap_stcr_diagnostic_payload(
        epoch=2,
        local_index=0,
        batch=batch,
        image_68=image_68,
        student_prob_68=student_prob,
        result=second_result,
    )
    diagnostic_path = out_root / "ap_stcr_v3_diagnostic.pt"
    png_path = out_root / "ap_stcr_v3_diagnostic.png"
    json_path = out_root / "sanity_ap_stcr_v3.json"
    _validate_payload(diagnostic, diagnostic_path)
    torch.save(diagnostic, diagnostic_path)
    render_ap_stcr_diagnostic(diagnostic, png_path, gt=None)

    result_stats = {
        name: _tensor_stats(second_result[name])
        for name in sorted(REQUIRED_V3_RESULT_FIELDS)
        if torch.is_tensor(second_result[name])
    }
    summary = {
        "status": "ok",
        "schema_version": "ap_stcr_v3_sanity_v1",
        "config": str(config_path),
        "ap_stcr_version": V3_VERSION,
        "device": str(device),
        "max_samples": int(max_samples),
        "batch_size": int(batch_size),
        "samples": _sample_rows(batch),
        "execution_contract": {
            "real_cached_batch": True,
            "real_student_forward": True,
            "real_teacher_forward": True,
            "torch_no_grad": True,
            "optimizer_created": False,
            "backward_called": False,
            "ema_update_called": False,
            "checkpoint_loaded": False,
            "checkpoint_written": False,
            "validation_called": False,
            "training_gt_used": False,
        },
        "schedule_endpoint_audit": schedule_audit,
        "v3_formula_audit": formula_audit,
        "analytic_acceptance_endpoint_audit": analytic_endpoint_audit,
        "alpha_endpoint_audit": alpha_audit,
        "past_only_audit": {
            "first_pass_history_valid": False,
            "first_pass_history_count": 0,
            "history_updated_after_first_pass_only": True,
            "second_pass_history_valid": True,
            "second_pass_history_count": 1,
            "history_mean_max_abs_error_vs_prior_teacher": history_error,
            "first_fused_equals_semantic_max_abs_error": first_fused_error,
            "second_fused_equals_semantic_temporal_mean_max_abs_error": (
                second_fused_error
            ),
            "build_batch_mutated_history": False,
        },
        "losses": loss_fields,
        "result_stats": result_stats,
        "source_conflict": {
            "usage": "diagnostic_only",
            "included_in_formula_dependencies": False,
            "stats": _tensor_stats(second_result["source_conflict_37"]),
        },
        "outputs": {
            "json": str(json_path),
            "diagnostic_pt": str(diagnostic_path),
            "diagnostic_png": str(png_path),
            "diagnostic_pt_kind": "offline_diagnostic_not_checkpoint",
        },
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded AP-STCR v3 real-cache/real-forward sanity audit "
            "without training, validation, checkpoints, EMA updates, or gradients."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Number of real cached samples; must be in [1,5].",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Dedicated empty output directory under MY-baseline/workdir.",
    )
    args = parser.parse_args()

    if not 1 <= int(args.max_samples) <= 5:
        parser.error("--max_samples must be in [1,5].")
    config_path = _resolve_config_path(args.config)
    out_root = _prepare_output_dir(args.out)
    device = _resolve_device(args.device)
    summary = _run_sanity(
        config_path=config_path,
        max_samples=int(args.max_samples),
        device=device,
        out_root=out_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
