import argparse
import copy
import json
import math
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
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    validate_ap_stcr_config,
)


A1_VERSION = "ap_stcr_v4_semantic_only_ablation"
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
REQUIRED_A1_RESULT_FIELDS = {
    "soft_correction_37",
    "soft_deviation_37",
    "semantic_contradiction_37",
    "combined_negative_evidence_37",
    "transition_envelope",
    "transition_penalty_37",
    "teacher_soft_37",
    "semantic_margin_37",
    "local_acceptance_37",
    "effective_teacher_weight_37",
    "effective_teacher_weight_68",
    "mixed_target_68",
    "global_teacher_ratio",
}
FORBIDDEN_A1_RESULT_FIELDS = {
    "temporal_support_37",
    "temporal_instability_37",
    "history_mean_37",
    "history_count",
    "history_valid",
    "direction_support_37",
    "deviation_support_37",
    "fused_support_37",
    "support_deficiency_37",
    "soft_inertia_37",
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


def _require_a1_result(result, fixed_37, fixed_68):
    if str(result.get("version", "")) != A1_VERSION:
        raise RuntimeError(
            f"Expected A1 result version={A1_VERSION!r}, "
            f"got {result.get('version')!r}."
        )
    missing = sorted(REQUIRED_A1_RESULT_FIELDS.difference(result))
    if missing:
        raise RuntimeError(f"AP-STCR A1 result is missing fields: {missing}")
    forbidden = sorted(FORBIDDEN_A1_RESULT_FIELDS.intersection(result))
    forbidden.extend(
        sorted(
            name
            for name in result
            if str(name).startswith("history_") and name not in forbidden
        )
    )
    if forbidden:
        raise RuntimeError(
            "AP-STCR A1 result contains temporal/history fields: "
            f"{forbidden}"
        )

    expected_37 = tuple(fixed_37.shape)
    expected_68 = tuple(fixed_68.shape)
    fields_37 = (
        "soft_correction_37",
        "soft_deviation_37",
        "semantic_contradiction_37",
        "combined_negative_evidence_37",
        "transition_penalty_37",
        "teacher_soft_37",
        "semantic_margin_37",
        "local_acceptance_37",
        "effective_teacher_weight_37",
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

    _assert_finite_range(
        "soft_correction_37", result["soft_correction_37"], -1.0, 1.0
    )
    _assert_finite_range(
        "semantic_margin_37", result["semantic_margin_37"], -1.0, 1.0
    )
    for name in fields_37:
        if name not in {"soft_correction_37", "semantic_margin_37"}:
            _assert_finite_range(name, result[name], 0.0, 1.0)
    for name in ("effective_teacher_weight_68", "mixed_target_68"):
        _assert_finite_range(name, result[name], 0.0, 1.0)
    for name in ("transition_envelope", "global_teacher_ratio"):
        value = float(result[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{name} outside [0,1]: {value}.")


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


def _analytic_endpoint_audit(ap_stcr, template):
    zeros = torch.zeros_like(template)
    ones = torch.ones_like(template)
    zero_contradiction = ap_stcr.compute_semantic_only_acceptance(
        semantic_contradiction=zeros,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    maximal_contradiction = ap_stcr.compute_semantic_only_acceptance(
        semantic_contradiction=ones,
        soft_deviation=ones,
        global_teacher_ratio=0.5,
    )
    expected_floor = 1.0 - float(ap_stcr.evidence_rejection_strength)
    errors = {
        "zero_contradiction_combined_zero": _assert_close(
            "A1 zero contradiction evidence",
            zero_contradiction["combined_negative_evidence_37"],
            zeros,
        ),
        "zero_contradiction_penalty_zero": _assert_close(
            "A1 zero contradiction penalty",
            zero_contradiction["transition_penalty_37"],
            zeros,
        ),
        "zero_contradiction_acceptance_one": _assert_close(
            "A1 zero contradiction acceptance",
            zero_contradiction["local_acceptance_37"],
            ones,
        ),
        "max_contradiction_combined_one": _assert_close(
            "A1 maximal contradiction evidence",
            maximal_contradiction["combined_negative_evidence_37"],
            ones,
        ),
        "max_contradiction_envelope_one": _assert_close(
            "A1 maximal contradiction envelope",
            maximal_contradiction["transition_envelope"],
            1.0,
        ),
        "max_contradiction_penalty_one": _assert_close(
            "A1 maximal contradiction penalty",
            maximal_contradiction["transition_penalty_37"],
            ones,
        ),
        "max_contradiction_acceptance_floor": _assert_close(
            "A1 maximal contradiction acceptance",
            maximal_contradiction["local_acceptance_37"],
            torch.full_like(template, expected_floor),
        ),
    }
    envelope_rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0):
        expected_envelope = 4.0 * alpha * (1.0 - alpha)
        output = ap_stcr.compute_semantic_only_acceptance(
            semantic_contradiction=ones,
            soft_deviation=ones,
            global_teacher_ratio=alpha,
        )
        envelope_error = _assert_close(
            f"A1 envelope alpha={alpha}",
            output["transition_envelope"],
            expected_envelope,
            atol=1e-7,
            rtol=0.0,
        )
        envelope_rows.append(
            {
                "alpha": alpha,
                "transition_envelope": float(output["transition_envelope"]),
                "expected_transition_envelope": expected_envelope,
                "abs_error": envelope_error,
            }
        )
    return {
        "compute_api": "compute_semantic_only_acceptance",
        "max_abs_errors": errors,
        "transition_envelope_rows": envelope_rows,
    }


def _formula_audit(result, fixed_37, ap_config, alpha):
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
    combined = semantic_contradiction
    envelope = max(0.0, min(1.0, 4.0 * alpha * (1.0 - alpha)))
    penalty = (envelope * soft_deviation * combined).clamp(0.0, 1.0)
    strength = float(ap_config["evidence_rejection_strength"])
    acceptance = (1.0 - strength * penalty).clamp(
        min=1.0 - strength,
        max=1.0,
    )
    errors = {
        "soft_correction": _assert_close(
            "A1 soft correction", result["soft_correction_37"], soft_correction
        ),
        "soft_deviation": _assert_close(
            "A1 soft deviation", result["soft_deviation_37"], soft_deviation
        ),
        "semantic_contradiction": _assert_close(
            "A1 semantic contradiction",
            result["semantic_contradiction_37"],
            semantic_contradiction,
        ),
        "combined_equals_semantic": _assert_close(
            "A1 combined evidence",
            result["combined_negative_evidence_37"],
            combined,
        ),
        "transition_envelope": _assert_close(
            "A1 transition envelope",
            result["transition_envelope"],
            envelope,
        ),
        "transition_penalty": _assert_close(
            "A1 transition penalty",
            result["transition_penalty_37"],
            penalty,
        ),
        "local_acceptance": _assert_close(
            "A1 local acceptance",
            result["local_acceptance_37"],
            acceptance,
        ),
    }
    return {
        "global_teacher_ratio": float(alpha),
        "transition_envelope": float(envelope),
        "max_abs_errors": errors,
        "combined_evidence_is_semantic_contradiction": True,
        "temporal_formula_dependency": False,
    }


def _natural_endpoint_audit(ap_stcr, inputs):
    at_zero = _build_batch(ap_stcr, inputs, 0.0)
    at_one = _build_batch(ap_stcr, inputs, 1.0)
    _require_a1_result(at_zero, inputs["fixed_37"], inputs["fixed_68"])
    _require_a1_result(at_one, inputs["fixed_37"], inputs["fixed_68"])
    errors = {
        "alpha_zero_envelope_zero": _assert_close(
            "A1 alpha=0 envelope", at_zero["transition_envelope"], 0.0
        ),
        "alpha_zero_acceptance_one": _assert_close(
            "A1 alpha=0 acceptance",
            at_zero["local_acceptance_37"],
            torch.ones_like(inputs["fixed_37"]),
        ),
        "alpha_zero_effective_weight_zero": _assert_close(
            "A1 alpha=0 effective weight",
            at_zero["effective_teacher_weight_68"],
            torch.zeros_like(inputs["fixed_68"]),
        ),
        "alpha_zero_target_fixed": _assert_close(
            "A1 alpha=0 mixed target",
            at_zero["mixed_target_68"],
            inputs["fixed_68"],
        ),
        "alpha_one_envelope_zero": _assert_close(
            "A1 alpha=1 envelope", at_one["transition_envelope"], 0.0
        ),
        "alpha_one_acceptance_one": _assert_close(
            "A1 alpha=1 acceptance",
            at_one["local_acceptance_37"],
            torch.ones_like(inputs["fixed_37"]),
        ),
        "alpha_one_effective_weight_one": _assert_close(
            "A1 alpha=1 effective weight",
            at_one["effective_teacher_weight_68"],
            torch.ones_like(inputs["fixed_68"]),
        ),
        "alpha_one_target_binary_teacher": _assert_close(
            "A1 alpha=1 mixed target",
            at_one["mixed_target_68"],
            inputs["teacher_binary_68"],
        ),
    }
    return {
        "max_abs_errors": errors,
        "endpoint_control": "global_teacher_ratio_only",
        "teacher_only_epoch_branch_used": False,
        "ap_stcr_start_or_stop_epoch_used": False,
    }


def _branch_target_audit(cfg, student_out, student_logits, mixed_target):
    if not bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
        raise RuntimeError("A1 requires the inherited coarse supervision branch.")
    if not bool(getattr(cfg, "USE_BASE_AUX_LOSS", True)):
        raise RuntimeError("A1 requires the inherited base supervision branch.")
    segmentation = build_ap_stcr_segmentation_group(
        cfg,
        2,
        student_out,
        student_logits,
        mixed_target,
    )
    expected = {
        "final": F.binary_cross_entropy_with_logits(
            student_logits,
            mixed_target,
            reduction="mean",
        )
    }
    coarse_logits = resize_logits_for_loss(
        student_out["coarse_logits_68"], cfg
    )
    expected["coarse"] = F.binary_cross_entropy_with_logits(
        coarse_logits,
        mixed_target,
        reduction="mean",
    )
    base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
    expected["base"] = F.binary_cross_entropy_with_logits(
        base_logits,
        mixed_target,
        reduction="mean",
    )
    errors = {
        name: _assert_close(
            f"A1 {name} shared-target BCE",
            segmentation[f"loss_{name}"],
            value,
        )
        for name, value in expected.items()
    }
    losses = {
        name: float(value.detach().cpu().item())
        for name, value in segmentation.items()
        if torch.is_tensor(value)
    }
    if any(not math.isfinite(value) for value in losses.values()):
        raise RuntimeError("A1 segmentation loss audit contains NaN/Inf.")
    return {
        "same_mixed_target_for_final_coarse_base": True,
        "branch_bce_max_abs_errors": errors,
        "losses": losses,
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
        raise RuntimeError("AP-STCR A1 sanity must run with gradients disabled.")

    cfg = load_config(config_path)
    validate_ap_stcr_config(cfg)
    ap_config = dict(cfg.AP_STCR)
    if str(ap_config.get("version", "")) != A1_VERSION:
        raise RuntimeError(
            f"This command accepts AP-STCR A1 only; got "
            f"{ap_config.get('version')!r}."
        )
    if ap_config.get("use_temporal_evidence") is not False:
        raise RuntimeError("A1 requires use_temporal_evidence=False.")
    if ap_config.get("temporal_history_enabled") is not False:
        raise RuntimeError("A1 requires temporal_history_enabled=False.")
    if str(ap_config.get("evidence_fusion", "")) != "semantic_only":
        raise RuntimeError("A1 requires evidence_fusion='semantic_only'.")
    if ap_config.get("use_transition_envelope") is not True:
        raise RuntimeError("A1 requires use_transition_envelope=True.")
    if not math.isclose(
        float(ap_config.get("evidence_rejection_strength", float("nan"))),
        0.35,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("A1 requires evidence_rejection_strength=0.35.")
    if not math.isclose(
        float(ap_config.get("min_local_acceptance", float("nan"))),
        0.65,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("A1 requires min_local_acceptance=0.65.")

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
        raise RuntimeError("AP-STCR A1 real forward requires image_68.")
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
    if ap_stcr.history_bank is not None:
        raise RuntimeError("A1 must not allocate a temporal history bank.")

    analytic_audit = _analytic_endpoint_audit(ap_stcr, inputs["fixed_37"])
    endpoint_audit = _natural_endpoint_audit(ap_stcr, inputs)
    result = _build_batch(ap_stcr, inputs, 0.5)
    _require_a1_result(result, inputs["fixed_37"], inputs["fixed_68"])
    if ap_stcr.history_bank is not None:
        raise RuntimeError("A1 build_batch allocated temporal history state.")
    formula_audit = _formula_audit(
        result,
        inputs["fixed_37"],
        ap_config,
        alpha=0.5,
    )
    branch_audit = _branch_target_audit(
        cfg,
        student_out,
        student_logits,
        result["mixed_target_68"],
    )

    diagnostic = build_ap_stcr_diagnostic_payload(
        epoch=1,
        local_index=0,
        batch=batch,
        image_68=image_68,
        student_prob_68=student_prob,
        result=result,
    )
    diagnostic_path = out_root / "ap_stcr_v4_semantic_only_diagnostic.pt"
    png_path = out_root / "ap_stcr_v4_semantic_only_diagnostic.png"
    json_path = out_root / "sanity_ap_stcr_v4_semantic_only.json"
    _validate_payload(diagnostic, diagnostic_path)
    torch.save(diagnostic, diagnostic_path)
    render_ap_stcr_diagnostic(diagnostic, png_path, gt=None)

    result_stats = {
        name: _tensor_stats(result[name])
        for name in sorted(REQUIRED_A1_RESULT_FIELDS)
        if torch.is_tensor(result[name])
    }
    summary = {
        "status": "ok",
        "schema_version": "ap_stcr_v4_semantic_only_sanity_v1",
        "config": str(config_path),
        "ap_stcr_version": A1_VERSION,
        "device": str(device),
        "max_samples": int(max_samples),
        "batch_size": int(batch_size),
        "samples": _sample_rows(batch),
        "execution_contract": {
            "real_cached_dino_features": True,
            "real_cached_dabe_targets_and_anchors": True,
            "real_cached_rgb": True,
            "real_student_forward": True,
            "real_teacher_forward": True,
            "torch_no_grad": True,
            "optimizer_created": False,
            "backward_called": False,
            "ema_update_called": False,
            "checkpoint_loaded": False,
            "checkpoint_written": False,
            "validation_called": False,
            "evaluation_called": False,
            "training_gt_used": False,
        },
        "temporal_ablation_audit": {
            "use_temporal_evidence": False,
            "temporal_history_enabled": False,
            "history_bank_allocated": False,
            "temporal_result_fields_present": False,
            "temporal_formula_dependency": False,
        },
        "analytic_endpoint_audit": analytic_audit,
        "natural_endpoint_audit": endpoint_audit,
        "semantic_only_formula_audit": formula_audit,
        "branch_target_audit": branch_audit,
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
            "Run the AP-STCR v4 semantic-only A1 real-cache/real-forward "
            "sanity audit without training, validation, evaluation, "
            "checkpoints, EMA updates, or gradients."
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
