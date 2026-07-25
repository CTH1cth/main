import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
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
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    validate_ap_stcr_config,
)


V4_VERSION = "ap_stcr_v4_transition_envelope_non_compensatory"
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
REQUIRED_V4_RESULT_FIELDS = {
    "soft_correction_37",
    "soft_deviation_37",
    "semantic_contradiction_37",
    "temporal_instability_37",
    "combined_negative_evidence_37",
    "transition_penalty_37",
    "transition_envelope",
    "teacher_soft_37",
    "semantic_support_37",
    "temporal_support_37",
    "local_acceptance_37",
    "effective_teacher_weight_37",
    "effective_teacher_weight_68",
    "mixed_target_68",
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


def _require_real_batch(batch, max_samples):
    missing = sorted(REQUIRED_BATCH_FIELDS.difference(batch))
    if missing:
        raise RuntimeError(f"Real cached batch is missing fields: {missing}")
    lower_names = {str(name).lower() for name in batch}
    forbidden = sorted(FORBIDDEN_GT_FIELDS.intersection(lower_names))
    forbidden.extend(
        sorted(name for name in lower_names if name.startswith("gt_"))
    )
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


def _require_v4_result(result, fixed_37, fixed_68):
    if str(result.get("version", "")) != V4_VERSION:
        raise RuntimeError(
            f"Expected v4 result version={V4_VERSION!r}, "
            f"got {result.get('version')!r}."
        )
    missing = sorted(REQUIRED_V4_RESULT_FIELDS.difference(result))
    if missing:
        raise RuntimeError(f"AP-STCR v4 result is missing fields: {missing}")
    expected_37 = tuple(fixed_37.shape)
    expected_68 = tuple(fixed_68.shape)
    fields_37 = (
        "soft_correction_37",
        "soft_deviation_37",
        "semantic_contradiction_37",
        "temporal_instability_37",
        "combined_negative_evidence_37",
        "transition_penalty_37",
        "teacher_soft_37",
        "semantic_support_37",
        "temporal_support_37",
        "local_acceptance_37",
        "effective_teacher_weight_37",
        "history_mean_37",
    )
    for name in fields_37:
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
    envelope = result["transition_envelope"]
    if torch.is_tensor(envelope):
        if envelope.numel() != 1 or envelope.requires_grad:
            raise RuntimeError("transition_envelope must be a detached scalar.")
        envelope = float(envelope.detach().item())
    else:
        envelope = float(envelope)
    if not math.isfinite(envelope) or not 0.0 <= envelope <= 1.0:
        raise RuntimeError(
            f"transition_envelope outside [0,1]: {envelope}."
        )

    _assert_finite_range(
        "soft_correction_37", result["soft_correction_37"], -1.0, 1.0
    )
    for name in fields_37:
        if name != "soft_correction_37":
            _assert_finite_range(name, result[name], 0.0, 1.0)
    for name in ("effective_teacher_weight_68", "mixed_target_68"):
        _assert_finite_range(name, result[name], 0.0, 1.0)


def _build_batch(ap_stcr, inputs, alpha):
    return ap_stcr.build_batch(
        dino_features=inputs["dino_features"],
        fixed_pseudo_37=inputs["fixed_37"],
        fixed_pseudo_68=inputs["fixed_68"],
        dabe_background_seed_37=inputs["background_37"],
        teacher_soft_68=inputs["teacher_soft_68"],
        teacher_binary_68=inputs["teacher_binary_68"],
        global_teacher_ratio=float(alpha),
        sample_indices=inputs["sample_indices"],
        datasets=inputs["datasets"],
        stems=inputs["stems"],
    )


def _analytic_evidence_audit(ap_stcr, template, history_valid):
    zeros = torch.zeros_like(template)
    ones = torch.ones_like(template)
    valid = torch.ones_like(history_valid, dtype=torch.bool)
    invalid = torch.zeros_like(history_valid, dtype=torch.bool)

    semantic_only = ap_stcr.compute_non_compensatory_acceptance(
        semantic_contradiction=torch.full_like(template, 0.8),
        temporal_support=ones,
        history_valid=valid,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    temporal_only = ap_stcr.compute_non_compensatory_acceptance(
        semantic_contradiction=zeros,
        temporal_support=torch.full_like(template, 0.3),
        history_valid=valid,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    no_negative = ap_stcr.compute_non_compensatory_acceptance(
        semantic_contradiction=zeros,
        temporal_support=ones,
        history_valid=valid,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    no_history = ap_stcr.compute_non_compensatory_acceptance(
        semantic_contradiction=zeros,
        temporal_support=zeros,
        history_valid=invalid,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    errors = {
        "semantic_0p8_temporal_0_soft_or_0p8": _assert_close(
            "soft-OR semantic-only endpoint",
            semantic_only["combined_negative_evidence_37"],
            torch.full_like(template, 0.8),
        ),
        "semantic_0_temporal_0p7_soft_or_0p7": _assert_close(
            "soft-OR temporal-only endpoint",
            temporal_only["combined_negative_evidence_37"],
            torch.full_like(template, 0.7),
        ),
        "semantic_0_temporal_0_soft_or_0": _assert_close(
            "soft-OR zero endpoint",
            no_negative["combined_negative_evidence_37"],
            zeros,
        ),
        "zero_negative_evidence_acceptance_one": _assert_close(
            "zero evidence acceptance",
            no_negative["local_acceptance_37"],
            ones,
        ),
        "no_history_temporal_instability_zero": _assert_close(
            "no-history temporal instability",
            no_history["temporal_instability_37"],
            zeros,
        ),
    }

    expected_envelopes = {
        0.0: 0.0,
        0.25: 0.75,
        0.5: 1.0,
        0.75: 0.75,
        0.95: 0.19,
        1.0: 0.0,
    }
    envelope_rows = []
    for alpha, expected in expected_envelopes.items():
        result = ap_stcr.compute_non_compensatory_acceptance(
            semantic_contradiction=ones,
            temporal_support=ones,
            history_valid=valid,
            soft_deviation=ones,
            global_teacher_ratio=alpha,
        )
        error = _assert_close(
            f"transition envelope alpha={alpha}",
            result["transition_envelope"],
            expected,
            atol=1e-7,
            rtol=0.0,
        )
        expected_acceptance = (
            1.0 - float(ap_stcr.evidence_rejection_strength) * expected
        )
        acceptance_error = _assert_close(
            f"enveloped acceptance alpha={alpha}",
            result["local_acceptance_37"],
            torch.full_like(template, expected_acceptance),
        )
        envelope_rows.append(
            {
                "alpha": alpha,
                "transition_envelope": float(
                    result["transition_envelope"]
                ),
                "local_acceptance": float(
                    result["local_acceptance_37"].mean().item()
                ),
                "expected_local_acceptance": expected_acceptance,
                "expected": expected,
                "abs_error": error,
                "acceptance_abs_error": acceptance_error,
            }
        )
    return {
        "compute_api": "compute_non_compensatory_acceptance",
        "soft_or_max_abs_errors": errors,
        "transition_envelope_rows": envelope_rows,
        "no_history_temporal_penalty": False,
    }


def _formula_audit(result, fixed_37, ap_config, global_teacher_ratio):
    soft_correction = result["teacher_soft_37"] - fixed_37
    soft_deviation = soft_correction.abs().clamp(0.0, 1.0)
    semantic_direction = torch.tanh(
        result["semantic_margin_37"] / float(ap_config["tau_margin"])
    )
    correction_direction = torch.tanh(
        soft_correction / float(ap_config["tau_delta"])
    )
    semantic_contradiction = torch.relu(
        -semantic_direction * correction_direction
    ).clamp(0.0, 1.0)
    history_mask = result["history_valid"].to(
        device=result["temporal_support_37"].device,
        dtype=torch.bool,
    )
    if history_mask.ndim == 1:
        history_mask = history_mask[:, None, None, None]
    temporal_instability = torch.where(
        history_mask,
        1.0 - result["temporal_support_37"],
        torch.zeros_like(result["temporal_support_37"]),
    ).clamp(0.0, 1.0)
    combined = (
        1.0
        - (1.0 - semantic_contradiction)
        * (1.0 - temporal_instability)
    ).clamp(0.0, 1.0)
    envelope = max(
        0.0,
        min(
            1.0,
            4.0 * global_teacher_ratio * (1.0 - global_teacher_ratio),
        ),
    )
    penalty = (envelope * soft_deviation * combined).clamp(0.0, 1.0)
    strength = float(ap_config["evidence_rejection_strength"])
    acceptance = (1.0 - strength * penalty).clamp(
        min=1.0 - strength,
        max=1.0,
    )
    observed_acceptance_min = float(
        result["local_acceptance_37"].min().item()
    )
    if observed_acceptance_min < 1.0 - strength - 1e-6:
        raise RuntimeError(
            "v4 local acceptance floor violated: "
            f"{observed_acceptance_min} < {1.0 - strength}."
        )
    errors = {
        "soft_correction": _assert_close(
            "v4 soft correction", result["soft_correction_37"], soft_correction
        ),
        "soft_deviation": _assert_close(
            "v4 soft deviation", result["soft_deviation_37"], soft_deviation
        ),
        "semantic_contradiction": _assert_close(
            "v4 semantic contradiction",
            result["semantic_contradiction_37"],
            semantic_contradiction,
        ),
        "temporal_instability": _assert_close(
            "v4 temporal instability",
            result["temporal_instability_37"],
            temporal_instability,
        ),
        "combined_negative_evidence": _assert_close(
            "v4 negative soft-OR",
            result["combined_negative_evidence_37"],
            combined,
        ),
        "transition_envelope": _assert_close(
            "v4 transition envelope",
            result["transition_envelope"],
            envelope,
        ),
        "transition_penalty": _assert_close(
            "v4 transition penalty",
            result["transition_penalty_37"],
            penalty,
        ),
        "local_acceptance": _assert_close(
            "v4 local acceptance",
            result["local_acceptance_37"],
            acceptance,
        ),
    }
    return {
        "global_teacher_ratio": float(global_teacher_ratio),
        "transition_envelope": float(envelope),
        "configured_acceptance_bounds": [1.0 - strength, 1.0],
        "observed_acceptance_min": observed_acceptance_min,
        "max_abs_errors": errors,
        "formula_dependencies": [
            "semantic_contradiction_37",
            "temporal_instability_37",
            "soft_deviation_37",
            "global_teacher_ratio",
            "evidence_rejection_strength",
        ],
        "fused_support_used": False,
    }


def _natural_endpoint_audit(ap_stcr, inputs):
    at_zero = _build_batch(ap_stcr, inputs, 0.0)
    at_one = _build_batch(ap_stcr, inputs, 1.0)
    _require_v4_result(at_zero, inputs["fixed_37"], inputs["fixed_68"])
    _require_v4_result(at_one, inputs["fixed_37"], inputs["fixed_68"])
    errors = {
        "alpha_zero_envelope_zero": _assert_close(
            "alpha=0 envelope", at_zero["transition_envelope"], 0.0
        ),
        "alpha_zero_acceptance_one": _assert_close(
            "alpha=0 acceptance",
            at_zero["local_acceptance_37"],
            torch.ones_like(inputs["fixed_37"]),
        ),
        "alpha_zero_effective_weight_zero": _assert_close(
            "alpha=0 effective weight",
            at_zero["effective_teacher_weight_68"],
            torch.zeros_like(inputs["fixed_68"]),
        ),
        "alpha_zero_target_fixed": _assert_close(
            "alpha=0 target",
            at_zero["mixed_target_68"],
            inputs["fixed_68"],
        ),
        "alpha_one_envelope_zero": _assert_close(
            "alpha=1 envelope", at_one["transition_envelope"], 0.0
        ),
        "alpha_one_acceptance_one": _assert_close(
            "alpha=1 acceptance",
            at_one["local_acceptance_37"],
            torch.ones_like(inputs["fixed_37"]),
        ),
        "alpha_one_effective_weight_one": _assert_close(
            "alpha=1 effective weight",
            at_one["effective_teacher_weight_68"],
            torch.ones_like(inputs["fixed_68"]),
        ),
        "alpha_one_target_teacher": _assert_close(
            "alpha=1 target",
            at_one["mixed_target_68"],
            inputs["teacher_binary_68"],
        ),
    }
    return {
        "max_abs_errors": errors,
        "teacher_only_epoch_branch_used": False,
        "ap_stcr_start_or_stop_epoch_used": False,
        "endpoint_control": "global_teacher_ratio_only",
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
        raise RuntimeError("AP-STCR v4 sanity must run with gradients disabled.")

    cfg = load_config(config_path)
    validate_ap_stcr_config(cfg)
    ap_config = dict(cfg.AP_STCR)
    if str(ap_config.get("version", "")) != V4_VERSION:
        raise RuntimeError(
            f"This command accepts AP-STCR v4 only; got "
            f"{ap_config.get('version')!r}."
        )

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
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        set_model_epoch(model, 1)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    if image_68 is None:
        raise RuntimeError("AP-STCR v4 real forward requires image_68.")
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
        "teacher_binary_68": teacher_binary_68,
    }.items():
        if value.requires_grad or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"{name} must be finite and detached.")

    inputs = {
        "dino_features": batch["feature"].to(device).float(),
        "fixed_37": batch["pu_target_soft_37"].to(device).float(),
        "fixed_68": batch["pu_target_soft"].to(device).float(),
        "background_37": batch["pu_bg_anchor_37"].to(device).float(),
        "teacher_soft_68": teacher_soft_68,
        "teacher_binary_68": teacher_binary_68,
        "sample_indices": batch["sample_index"],
        "datasets": batch["dataset"],
        "stems": batch["stem"],
    }
    ap_stcr = AnchorPropagatedSemanticTemporalCorrection(
        ap_config,
        dataset.keys,
    )
    analytic_audit = _analytic_evidence_audit(
        ap_stcr,
        inputs["fixed_37"],
        torch.zeros(
            (batch_size, 1, 1, 1),
            dtype=torch.bool,
            device=device,
        ),
    )
    endpoint_audit = _natural_endpoint_audit(ap_stcr, inputs)

    before_first = _history_snapshot(ap_stcr)
    first_result = _build_batch(ap_stcr, inputs, 0.5)
    after_first = _history_snapshot(ap_stcr)
    _assert_history_unchanged("first build_batch", before_first, after_first)
    _require_v4_result(first_result, inputs["fixed_37"], inputs["fixed_68"])
    first_instability_error = _assert_close(
        "first-pass temporal instability",
        first_result["temporal_instability_37"],
        torch.zeros_like(inputs["fixed_37"]),
    )
    if bool(first_result["history_valid"].any().item()):
        raise RuntimeError("First AP-STCR pass unexpectedly consumed current history.")
    if bool((first_result["history_count"] != 0).any().item()):
        raise RuntimeError("First AP-STCR pass must observe zero history count.")

    ap_stcr.update_history(
        inputs["sample_indices"],
        inputs["datasets"],
        inputs["stems"],
        first_result["teacher_soft_37"],
        epoch=1,
    )
    set_model_epoch(student, 2)
    set_model_epoch(teacher, 2)
    before_second = _history_snapshot(ap_stcr)
    second_result = _build_batch(ap_stcr, inputs, 0.5)
    after_second = _history_snapshot(ap_stcr)
    _assert_history_unchanged("second build_batch", before_second, after_second)
    _require_v4_result(second_result, inputs["fixed_37"], inputs["fixed_68"])
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
    formula_audit = _formula_audit(
        second_result,
        inputs["fixed_37"],
        ap_config,
        global_teacher_ratio=0.5,
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
    if any(not math.isfinite(value) for value in loss_fields.values()):
        raise RuntimeError("Segmentation loss audit contains NaN/Inf.")

    diagnostic = build_ap_stcr_diagnostic_payload(
        epoch=2,
        local_index=0,
        batch=batch,
        image_68=image_68,
        student_prob_68=student_prob,
        result=second_result,
    )
    diagnostic_path = out_root / "ap_stcr_v4_diagnostic.pt"
    png_path = out_root / "ap_stcr_v4_diagnostic.png"
    json_path = out_root / "sanity_ap_stcr_v4.json"
    _validate_payload(diagnostic, diagnostic_path)
    torch.save(diagnostic, diagnostic_path)
    render_ap_stcr_diagnostic(diagnostic, png_path, gt=None)

    result_stats = {
        name: _tensor_stats(second_result[name])
        for name in sorted(REQUIRED_V4_RESULT_FIELDS)
        if torch.is_tensor(second_result[name])
    }
    summary = {
        "status": "ok",
        "schema_version": "ap_stcr_v4_sanity_v1",
        "config": str(config_path),
        "ap_stcr_version": V4_VERSION,
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
        "analytic_evidence_audit": analytic_audit,
        "natural_endpoint_audit": endpoint_audit,
        "v4_formula_audit": formula_audit,
        "past_only_audit": {
            "first_pass_history_valid": False,
            "first_pass_history_count": 0,
            "first_pass_temporal_instability_max_abs_error": (
                first_instability_error
            ),
            "history_updated_after_first_pass_only": True,
            "second_pass_history_valid": True,
            "second_pass_history_count": 1,
            "history_mean_max_abs_error_vs_prior_teacher": history_error,
            "build_batch_mutated_history": False,
        },
        "losses": loss_fields,
        "result_stats": result_stats,
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
            "Run the AP-STCR v4 real-cache/real-forward sanity audit without "
            "training, validation, checkpoints, EMA updates, or gradients."
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
