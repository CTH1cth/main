"""Pure helpers for the Full-ECST causal/selectivity audit.

The functions in this module are deliberately independent from the training
dataset and never read GT.  Post-hoc tools may pass training GT tensors into
the correction helpers, but the training path imports only the loss/control
functions near the top of this file.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


ECST_CAUSAL_CONTROL_MODES = frozenset(
    {"none", "gradient_magnitude_global", "spatial_roll"}
)
ECST_CAUSAL_CONTROL_SEED = 20260731
ECST_CAUSAL_EPS = 1e-6
ECST_CAUSAL_COVERAGES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)


def _as_float_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor.")
    if value.numel() < 1:
        raise RuntimeError(f"{name} cannot be empty.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf.")
    return value


def _same_shape(*values: torch.Tensor) -> None:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise RuntimeError(f"Tensor shapes differ: {sorted(shapes)}")


def get_ecst_causal_control_mode(cfg: Any) -> str:
    mode = str(getattr(cfg, "ECST_CAUSAL_CONTROL_MODE", "none")).strip().lower()
    if mode not in ECST_CAUSAL_CONTROL_MODES:
        raise RuntimeError(
            f"Unsupported ECST_CAUSAL_CONTROL_MODE={mode!r}; "
            f"expected one of {sorted(ECST_CAUSAL_CONTROL_MODES)}."
        )
    return mode


def validate_ecst_causal_audit_config(
    cfg: Any,
    baseline_cfg: Any | None = None,
) -> dict[str, Any]:
    """Validate a causal-control config without mutating either config."""

    mode = get_ecst_causal_control_mode(cfg)
    report: dict[str, Any] = {
        "schema": "full_ecst_causal_control_config_v1",
        "status": "PASS",
        "mode": mode,
        "errors": [],
        "effective_differences": [],
    }
    if mode == "none":
        return report

    required = {
        "USE_ECST": True,
        "TEACHER_ROUTING_MODE": "ecst",
        "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
        "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
        "STATIC_WEIGHT_MODE": "ones",
        "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
        "STOP_AFTER_EPOCH": 20,
    }
    for name, expected in required.items():
        actual = getattr(cfg, name, None)
        if isinstance(expected, float):
            try:
                matches = math.isfinite(float(actual)) and abs(
                    float(actual) - expected
                ) <= 1e-12
            except (TypeError, ValueError):
                matches = False
        elif isinstance(expected, str):
            matches = str(actual).strip().lower() == expected.lower()
        else:
            matches = actual is expected
        if not matches:
            report["errors"].append(
                f"{name}={actual!r}, expected={expected!r}"
            )

    if mode == "gradient_magnitude_global":
        norm = str(
            getattr(cfg, "ECST_CAUSAL_GRADIENT_NORM", "")
        ).strip().lower()
        if norm != "l1":
            report["errors"].append(
                "gradient_magnitude_global requires "
                "ECST_CAUSAL_GRADIENT_NORM='l1'"
            )
    elif mode == "spatial_roll":
        try:
            seed = int(getattr(cfg, "ECST_CAUSAL_CONTROL_SEED"))
        except (AttributeError, TypeError, ValueError):
            seed = -1
        if seed != ECST_CAUSAL_CONTROL_SEED:
            report["errors"].append(
                "spatial_roll requires ECST_CAUSAL_CONTROL_SEED=20260731"
            )

    if baseline_cfg is not None:
        allowed = {
            "EXP_NAME",
            "ECST_CAUSAL_CONTROL_MODE",
            "ECST_CAUSAL_CONTROL_SEED",
            "ECST_CAUSAL_GRADIENT_NORM",
            "STOP_AFTER_EPOCH",
        }
        baseline_values = {
            name: getattr(baseline_cfg, name)
            for name in dir(baseline_cfg)
            if name.isupper()
        }
        control_values = {
            name: getattr(cfg, name)
            for name in dir(cfg)
            if name.isupper()
        }
        missing = object()
        differences = []
        for name in sorted(set(baseline_values) | set(control_values)):
            before = baseline_values.get(name, missing)
            after = control_values.get(name, missing)
            if before == after:
                continue
            differences.append(
                {
                    "field": name,
                    "baseline": "<MISSING>" if before is missing else before,
                    "control": "<MISSING>" if after is missing else after,
                }
            )
        report["effective_differences"] = differences
        unexpected = sorted(
            record["field"] for record in differences if record["field"] not in allowed
        )
        if unexpected:
            report["errors"].append(
                f"unexpected effective config differences: {unexpected}"
            )

    if report["errors"]:
        report["status"] = "FAIL"
        raise RuntimeError(
            "ECST causal-control config validation failed: "
            + "; ".join(report["errors"])
        )
    return report


def normalized_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    eps: float = ECST_CAUSAL_EPS,
) -> torch.Tensor:
    """The authoritative Full-ECST batch-global normalized Teacher BCE."""

    logits = _as_float_tensor("logits", logits)
    target = _as_float_tensor("target", target)
    weight_map = _as_float_tensor("weight_map", weight_map)
    _same_shape(logits, target, weight_map)
    if weight_map.requires_grad:
        raise RuntimeError("ECST audit weight_map must be detached.")
    if float(weight_map.min().item()) < 0.0:
        raise RuntimeError("weight_map must be non-negative.")
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    return (pixel_loss * weight_map).sum() / (
        weight_map.sum() + float(eps)
    )


def plain_mean_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = _as_float_tensor("logits", logits)
    target = _as_float_tensor("target", target)
    _same_shape(logits, target)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def compute_logit_gradient_field(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    *,
    mode: str = "plain",
    global_scalar: float | torch.Tensor | None = None,
    eps: float = ECST_CAUSAL_EPS,
) -> torch.Tensor:
    """Return the analytic BCE gradient with respect to each logit."""

    logits = _as_float_tensor("logits", logits)
    target = _as_float_tensor("target", target)
    _same_shape(logits, target)
    residual = (torch.sigmoid(logits.detach()) - target.detach()).float()
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "plain":
        result = residual / float(residual.numel())
    elif normalized_mode == "normalized_weighted":
        if weight_map is None:
            raise RuntimeError("normalized_weighted gradient requires weight_map.")
        weight_map = _as_float_tensor("weight_map", weight_map).detach().float()
        _same_shape(residual, weight_map)
        result = weight_map * residual / (weight_map.sum() + float(eps))
    elif normalized_mode == "gmg":
        if global_scalar is None:
            raise RuntimeError("GMG gradient requires global_scalar.")
        scalar = (
            float(global_scalar.detach().item())
            if torch.is_tensor(global_scalar)
            else float(global_scalar)
        )
        if not math.isfinite(scalar) or scalar < 0.0:
            raise RuntimeError(f"Invalid GMG scalar: {scalar}")
        result = scalar * residual / float(residual.numel())
    else:
        raise RuntimeError(f"Unknown gradient mode: {mode!r}")
    return result.detach()


def compute_gradient_magnitude_match_scalar(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    *,
    norm: str = "l1",
    eps: float = ECST_CAUSAL_EPS,
) -> torch.Tensor:
    weighted = compute_logit_gradient_field(
        logits,
        target,
        weight_map,
        mode="normalized_weighted",
        eps=eps,
    )
    plain = compute_logit_gradient_field(logits, target, mode="plain", eps=eps)
    norm = str(norm).strip().lower()
    if norm == "l1":
        numerator = weighted.abs().sum()
        denominator = plain.abs().sum()
    elif norm == "l2":
        numerator = torch.linalg.vector_norm(weighted.reshape(-1), ord=2)
        denominator = torch.linalg.vector_norm(plain.reshape(-1), ord=2)
    else:
        raise RuntimeError(f"Unsupported gradient matching norm: {norm!r}")
    # ``+ eps`` would systematically undershoot the requested magnitude by
    # approximately eps even when the denominator is well conditioned.  Use
    # eps only as a zero guard so GMG remains an actual magnitude match.
    safe_denominator = denominator.clamp_min(float(eps))
    return (numerator / safe_denominator).detach()


def gradient_magnitude_matched_global_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    original_weight_map: torch.Tensor,
    *,
    eps: float = ECST_CAUSAL_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    scalar = compute_gradient_magnitude_match_scalar(
        logits, target, original_weight_map, norm="l1", eps=eps
    )
    return scalar * plain_mean_bce(logits, target), scalar


def build_constant_mean_map(
    weight_map: torch.Tensor,
    *,
    per_image: bool = False,
) -> torch.Tensor:
    """Build a global constant map, or the task-defined per-image shadow map.

    Full ECST normalizes over the complete batch.  Only the global constant map
    is mathematically equivalent to ordinary mean BCE.  A per-image constant
    map can reweight images when their means differ and is therefore exposed
    only as an explicit shadow variant.
    """

    weight_map = _as_float_tensor("weight_map", weight_map)
    if weight_map.ndim != 4:
        raise RuntimeError("weight_map must be [B,C,H,W].")
    if per_image:
        mean = weight_map.detach().mean(dim=(1, 2, 3), keepdim=True)
    else:
        mean = weight_map.detach().mean().reshape(1, 1, 1, 1)
    return torch.ones_like(weight_map) * mean


def _stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _identity_list(value: Sequence[str] | str, batch_size: int, name: str) -> list[str]:
    if isinstance(value, str):
        if batch_size != 1:
            raise RuntimeError(f"Scalar {name} is valid only for batch size 1.")
        return [value]
    result = [str(item) for item in value]
    if len(result) != batch_size:
        raise RuntimeError(
            f"{name} identity count {len(result)} != batch size {batch_size}."
        )
    return result


def deterministic_roll_shift(
    dataset: str,
    stem: str,
    *,
    seed: int = ECST_CAUSAL_CONTROL_SEED,
    height: int = 68,
    width: int = 68,
) -> tuple[int, int]:
    if height != 68 or width != 68:
        raise RuntimeError("Registered Spatial Roll is defined for 68x68 maps.")
    digest = hashlib.sha256(
        f"{dataset}\x1f{stem}\x1f{int(seed)}".encode("utf-8")
    ).digest()
    magnitude_y = 17 + int.from_bytes(digest[0:2], "big") % 35
    magnitude_x = 17 + int.from_bytes(digest[2:4], "big") % 35
    shift_y = magnitude_y if digest[4] & 1 else -magnitude_y
    shift_x = magnitude_x if digest[5] & 1 else -magnitude_x
    return shift_y, shift_x


def build_spatial_roll_map(
    weight_map: torch.Tensor,
    datasets: Sequence[str] | str,
    stems: Sequence[str] | str,
    *,
    seed: int = ECST_CAUSAL_CONTROL_SEED,
) -> torch.Tensor:
    weight_map = _as_float_tensor("weight_map", weight_map)
    if weight_map.ndim != 4 or tuple(weight_map.shape[-2:]) != (68, 68):
        raise RuntimeError("Spatial Roll requires [B,C,68,68] weight maps.")
    batch_size = int(weight_map.shape[0])
    dataset_values = _identity_list(datasets, batch_size, "dataset")
    stem_values = _identity_list(stems, batch_size, "stem")
    rolled = []
    for index, (dataset, stem) in enumerate(zip(dataset_values, stem_values)):
        shift = deterministic_roll_shift(
            dataset, stem, seed=int(seed), height=68, width=68
        )
        rolled.append(
            torch.roll(weight_map[index], shifts=shift, dims=(-2, -1))
        )
    return torch.stack(rolled, dim=0).detach()


def build_block_shuffle_map(
    weight_map: torch.Tensor,
    datasets: Sequence[str] | str,
    stems: Sequence[str] | str,
    *,
    seed: int = ECST_CAUSAL_CONTROL_SEED,
    block_size: int = 4,
) -> torch.Tensor:
    weight_map = _as_float_tensor("weight_map", weight_map)
    if weight_map.ndim != 4:
        raise RuntimeError("Block Shuffle requires [B,C,H,W].")
    batch_size, channels, height, width = weight_map.shape
    block_size = int(block_size)
    if height % block_size or width % block_size:
        raise RuntimeError("Map size must be divisible by block_size.")
    dataset_values = _identity_list(datasets, batch_size, "dataset")
    stem_values = _identity_list(stems, batch_size, "stem")
    rows = height // block_size
    cols = width // block_size
    outputs = []
    for index, (dataset, stem) in enumerate(zip(dataset_values, stem_values)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_int("block", dataset, stem, int(seed)) % (2**63 - 1)
        )
        permutation = torch.randperm(rows * cols, generator=generator).to(
            weight_map.device
        )
        blocks = (
            weight_map[index]
            .reshape(channels, rows, block_size, cols, block_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(rows * cols, channels, block_size, block_size)
        )
        shuffled = blocks.index_select(0, permutation)
        shuffled = (
            shuffled.reshape(rows, cols, channels, block_size, block_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(channels, height, width)
        )
        outputs.append(shuffled)
    return torch.stack(outputs, dim=0).detach()


def build_pixel_permutation_map(
    weight_map: torch.Tensor,
    datasets: Sequence[str] | str,
    stems: Sequence[str] | str,
    *,
    seed: int = ECST_CAUSAL_CONTROL_SEED,
) -> torch.Tensor:
    weight_map = _as_float_tensor("weight_map", weight_map)
    if weight_map.ndim != 4:
        raise RuntimeError("Pixel permutation requires [B,C,H,W].")
    batch_size, channels, height, width = weight_map.shape
    dataset_values = _identity_list(datasets, batch_size, "dataset")
    stem_values = _identity_list(stems, batch_size, "stem")
    outputs = []
    for index, (dataset, stem) in enumerate(zip(dataset_values, stem_values)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_int("pixel", dataset, stem, int(seed)) % (2**63 - 1)
        )
        permutation = torch.randperm(height * width, generator=generator).to(
            weight_map.device
        )
        flat = weight_map[index].reshape(channels, height * width)
        outputs.append(flat.index_select(1, permutation).reshape(channels, height, width))
    return torch.stack(outputs, dim=0).detach()


def compute_correction_masks(
    static_target: torch.Tensor,
    teacher_binary: torch.Tensor,
    gt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    static = static_target.detach() > 0.5
    teacher = teacher_binary.detach() > 0.5
    _same_shape(static, teacher)
    add_fg = (~static) & teacher
    erase_fg = static & (~teacher)
    conflict = add_fg | erase_fg
    result = {
        "add_fg": add_fg.detach(),
        "erase_fg": erase_fg.detach(),
        "conflict": conflict.detach(),
        "agreement": (~conflict).detach(),
    }
    if gt is not None:
        ground_truth = gt.detach() > 0.5
        _same_shape(static, ground_truth)
        correct = conflict & (teacher == ground_truth)
        wrong = conflict & (~correct)
        result.update(
            {
                "correct": correct.detach(),
                "wrong": wrong.detach(),
                "add_fg_correct": (add_fg & ground_truth).detach(),
                "add_fg_wrong": (add_fg & (~ground_truth)).detach(),
                "erase_fg_correct": (erase_fg & (~ground_truth)).detach(),
                "erase_fg_wrong": (erase_fg & ground_truth).detach(),
                "static_error": (static != ground_truth).detach(),
                "teacher_error": (teacher != ground_truth).detach(),
            }
        )
    return result


def _binary_overlap_metrics(prediction: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    pred = prediction.bool()
    truth = gt.bool()
    tp = float((pred & truth).sum().item())
    fp = float((pred & (~truth)).sum().item())
    fn = float(((~pred) & truth).sum().item())
    iou_denominator = tp + fp + fn
    f1_denominator = 2.0 * tp + fp + fn
    return {
        "iou": tp / iou_denominator if iou_denominator > 0.0 else 0.0,
        "f1": 2.0 * tp / f1_denominator if f1_denominator > 0.0 else 0.0,
    }


def compute_correction_quality(
    static_target: torch.Tensor,
    teacher_binary: torch.Tensor,
    gt: torch.Tensor,
) -> dict[str, float | int | None]:
    masks = compute_correction_masks(static_target, teacher_binary, gt)
    static = static_target.detach() > 0.5
    teacher = teacher_binary.detach() > 0.5
    truth = gt.detach() > 0.5
    conflict_count = int(masks["conflict"].sum().item())
    correct_count = int(masks["correct"].sum().item())
    wrong_count = int(masks["wrong"].sum().item())
    static_error_count = int(masks["static_error"].sum().item())
    before = _binary_overlap_metrics(static, truth)
    after = _binary_overlap_metrics(teacher, truth)
    fp_corrected = int((static & (~truth) & (~teacher)).sum().item())
    fp_introduced = int(((~static) & (~truth) & teacher).sum().item())
    fn_corrected = int(((~static) & truth & teacher).sum().item())
    fn_introduced = int((static & truth & (~teacher)).sum().item())
    result: dict[str, float | int | None] = {
        "correction_pixel_count": conflict_count,
        "correct_correction_count": correct_count,
        "wrong_correction_count": wrong_count,
        "correction_precision": (
            correct_count / conflict_count if conflict_count else None
        ),
        "correction_recall_relative_to_static_errors": (
            correct_count / static_error_count if static_error_count else None
        ),
        "add_fg_correct_count": int(masks["add_fg_correct"].sum().item()),
        "add_fg_wrong_count": int(masks["add_fg_wrong"].sum().item()),
        "erase_fg_correct_count": int(masks["erase_fg_correct"].sum().item()),
        "erase_fg_wrong_count": int(masks["erase_fg_wrong"].sum().item()),
        "fp_corrected": fp_corrected,
        "fp_introduced": fp_introduced,
        "fn_corrected": fn_corrected,
        "fn_introduced": fn_introduced,
        "net_pixel_error_change": int(masks["teacher_error"].sum().item())
        - static_error_count,
        "net_iou_change": after["iou"] - before["iou"],
        "net_f1_change": after["f1"] - before["f1"],
    }
    add_count = int(masks["add_fg"].sum().item())
    erase_count = int(masks["erase_fg"].sum().item())
    result.update(
        {
            "add_fg_correct_ratio": (
                result["add_fg_correct_count"] / add_count if add_count else None
            ),
            "add_fg_wrong_ratio": (
                result["add_fg_wrong_count"] / add_count if add_count else None
            ),
            "erase_fg_correct_ratio": (
                result["erase_fg_correct_count"] / erase_count if erase_count else None
            ),
            "erase_fg_wrong_ratio": (
                result["erase_fg_wrong_count"] / erase_count if erase_count else None
            ),
        }
    )
    return result


def _ranking_inputs(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    if score.numel() != label.numel():
        raise RuntimeError("ranking score/label lengths differ.")
    finite = torch.isfinite(score)
    score = score[finite]
    label = label[finite]
    if score.numel() == 0:
        return score, label, "EMPTY"
    positives = int(label.sum().item())
    if positives == 0:
        return score, label, "ALL_NEGATIVE"
    if positives == int(label.numel()):
        return score, label, "ALL_POSITIVE"
    if float(score.max().item()) == float(score.min().item()):
        return score, label, "ALL_SCORES_EQUAL"
    return score, label, None


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < int(values.numel()):
        end = start + 1
        while end < int(values.numel()) and bool(
            sorted_values[end] == sorted_values[start]
        ):
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def compute_score_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
    ece_bins: int = 10,
) -> dict[str, Any]:
    score, label, reason = _ranking_inputs(scores, labels)
    if reason is not None:
        return {
            "status": "N/A",
            "reason": reason,
            "count": int(score.numel()),
            "positive_count": int(label.sum().item()),
            "auroc": None,
            "auprc": None,
            "average_precision": None,
            "brier": None,
            "ece": None,
            "balanced_accuracy": {},
        }
    label_f = label.float()
    positive_count = int(label.sum().item())
    negative_count = int(label.numel()) - positive_count
    ranks = _average_ranks(score)
    positive_rank_sum = float(ranks[label].sum().item())
    auroc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)

    order = torch.argsort(score, descending=True, stable=True)
    sorted_label = label_f[order]
    tp = torch.cumsum(sorted_label, dim=0)
    fp = torch.cumsum(1.0 - sorted_label, dim=0)
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / float(positive_count)
    previous_recall = torch.cat([torch.zeros(1), recall[:-1]])
    average_precision = float(
        ((recall - previous_recall) * precision).sum().item()
    )
    recall_curve = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    precision_curve = torch.cat([torch.ones(1), precision, torch.zeros(1)])
    auprc = float(torch.trapz(precision_curve, recall_curve).item())

    balanced = {}
    for threshold in thresholds:
        prediction = score >= float(threshold)
        tpr = float((prediction & label).sum().item()) / positive_count
        tnr = float(((~prediction) & (~label)).sum().item()) / negative_count
        balanced[f"{float(threshold):.3f}"] = 0.5 * (tpr + tnr)

    clipped = score.clamp(0.0, 1.0)
    brier = float((clipped - label_f).square().mean().item())
    ece = 0.0
    for index in range(int(ece_bins)):
        lower = index / float(ece_bins)
        upper = (index + 1) / float(ece_bins)
        in_bin = (clipped >= lower) & (
            clipped <= upper if index == ece_bins - 1 else clipped < upper
        )
        if bool(in_bin.any().item()):
            fraction = float(in_bin.float().mean().item())
            confidence = float(clipped[in_bin].mean().item())
            accuracy = float(label_f[in_bin].mean().item())
            ece += fraction * abs(confidence - accuracy)
    return {
        "status": "OK",
        "reason": None,
        "count": int(label.numel()),
        "positive_count": positive_count,
        "positive_ratio": positive_count / float(label.numel()),
        "auroc": auroc,
        "auprc": auprc,
        "average_precision": average_precision,
        "brier": brier,
        "ece": ece,
        "balanced_accuracy": balanced,
    }


def compute_precision_at_coverage(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    coverages: Sequence[float] = ECST_CAUSAL_COVERAGES,
) -> list[dict[str, Any]]:
    score = scores.detach().float().reshape(-1).cpu()
    label = labels.detach().bool().reshape(-1).cpu()
    if score.numel() != label.numel():
        raise RuntimeError("coverage score/label lengths differ.")
    finite = torch.isfinite(score)
    score, label = score[finite], label[finite]
    if score.numel() == 0:
        return [
            {
                "coverage": float(coverage),
                "status": "N/A",
                "reason": "EMPTY",
            }
            for coverage in coverages
        ]
    order = torch.argsort(score, descending=True, stable=True)
    ranked = label[order]
    total_correct = int(label.sum().item())
    total_wrong = int(label.numel()) - total_correct
    random_precision = total_correct / float(label.numel())
    rows = []
    for coverage in coverages:
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            raise ValueError(f"Invalid coverage: {coverage}")
        selected_count = max(1, int(math.ceil(coverage * int(label.numel()))))
        selected = ranked[:selected_count]
        correct = int(selected.sum().item())
        wrong = selected_count - correct
        precision = correct / float(selected_count)
        rows.append(
            {
                "coverage": coverage,
                "status": "OK",
                "selected_count": selected_count,
                "precision": precision,
                "wrong_correction_rate": wrong / float(selected_count),
                "correct_correction_retention": (
                    correct / float(total_correct) if total_correct else None
                ),
                "wrong_correction_retention": (
                    wrong / float(total_wrong) if total_wrong else None
                ),
                "random_precision": random_precision,
                "selectivity_gain_over_random": precision - random_precision,
            }
        )
    return rows


def compute_gradient_retention(
    logits: torch.Tensor,
    teacher_target: torch.Tensor,
    static_target: torch.Tensor,
    gt: torch.Tensor,
    weight_map: torch.Tensor,
    *,
    eps: float = ECST_CAUSAL_EPS,
) -> dict[str, float | None]:
    _same_shape(logits, teacher_target, static_target, gt, weight_map)
    masks = compute_correction_masks(static_target, teacher_target, gt)
    magnitude = (
        torch.sigmoid(logits.detach()) - teacher_target.detach()
    ).abs().float()
    weight = weight_map.detach().float()

    def retention(mask: torch.Tensor) -> float | None:
        denominator = float((magnitude * mask.float()).sum().item())
        if denominator <= 0.0:
            return None
        numerator = float((weight * magnitude * mask.float()).sum().item())
        return numerator / denominator

    correct = retention(masks["correct"])
    wrong = retention(masks["wrong"])
    return {
        "correct_gradient_retention": correct,
        "wrong_gradient_retention": wrong,
        "selectivity_gap": (
            correct - wrong if correct is not None and wrong is not None else None
        ),
    }


def compute_map_spatial_autocorrelation(weight_map: torch.Tensor) -> float | None:
    value = weight_map.detach().float()
    if value.ndim < 2:
        raise RuntimeError("Spatial map must have at least two dimensions.")
    centered = value - value.mean(dim=(-2, -1), keepdim=True)
    pairs = []
    for lhs, rhs in (
        (centered[..., :, :-1], centered[..., :, 1:]),
        (centered[..., :-1, :], centered[..., 1:, :]),
    ):
        denominator = torch.sqrt(lhs.square().sum() * rhs.square().sum())
        if float(denominator.item()) > 0.0:
            pairs.append(float((lhs * rhs).sum().item() / denominator.item()))
    return sum(pairs) / len(pairs) if pairs else None


def compute_gradient_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    foreground_mask: torch.Tensor | None = None,
    conflict_mask: torch.Tensor | None = None,
    eps: float = ECST_CAUSAL_EPS,
) -> dict[str, float | None]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    _same_shape(reference, candidate)
    ref_flat, candidate_flat = reference.reshape(-1), candidate.reshape(-1)
    denominator = torch.linalg.vector_norm(ref_flat) * torch.linalg.vector_norm(
        candidate_flat
    )
    denominator_value = float(denominator.item())
    cosine = (
        float(torch.dot(ref_flat, candidate_flat).item() / denominator_value)
        if denominator_value > 0.0
        else None
    )
    sign_agreement = float(
        (torch.sign(reference) == torch.sign(candidate)).float().mean().item()
    )
    top_count = max(1, int(math.ceil(0.10 * reference.numel())))
    ref_top = torch.topk(reference.abs().reshape(-1), top_count).indices
    candidate_top = torch.topk(candidate.abs().reshape(-1), top_count).indices
    overlap = len(set(ref_top.tolist()) & set(candidate_top.tolist())) / float(top_count)
    result: dict[str, float | None] = {
        "gradient_l1": float(candidate.abs().sum().item()),
        "gradient_l2": float(torch.linalg.vector_norm(candidate_flat).item()),
        "gradient_cosine_to_original": cosine,
        "gradient_sign_agreement": sign_agreement,
        "top10_abs_gradient_overlap": overlap,
    }

    def mass(mask: torch.Tensor | None) -> float | None:
        if mask is None:
            return None
        mask = mask.detach().bool()
        _same_shape(candidate, mask)
        total = float(candidate.abs().sum().item())
        if total <= 0.0:
            return None
        return float(candidate.abs()[mask].sum().item()) / total

    fg_mass = mass(foreground_mask)
    conflict_mass = mass(conflict_mask)
    result.update(
        {
            "foreground_gradient_mass": fg_mass,
            "background_gradient_mass": (
                1.0 - fg_mass if fg_mass is not None else None
            ),
            "conflict_gradient_mass": conflict_mass,
            "non_conflict_gradient_mass": (
                1.0 - conflict_mass if conflict_mass is not None else None
            ),
        }
    )
    return result


def compute_bootstrap_ci(
    image_records: Sequence[Any],
    metric_fn: Callable[[Sequence[Any]], float | None],
    *,
    repetitions: int = 1000,
    seed: int = ECST_CAUSAL_CONTROL_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    records = list(image_records)
    if not records:
        return {"status": "N/A", "reason": "NO_IMAGES", "value": None}
    value = metric_fn(records)
    if value is None or not math.isfinite(float(value)):
        return {"status": "N/A", "reason": "METRIC_UNDEFINED", "value": None}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    estimates = []
    count = len(records)
    for _ in range(int(repetitions)):
        indices = torch.randint(0, count, (count,), generator=generator).tolist()
        estimate = metric_fn([records[index] for index in indices])
        if estimate is not None and math.isfinite(float(estimate)):
            estimates.append(float(estimate))
    if not estimates:
        return {"status": "N/A", "reason": "ALL_REPLICATES_UNDEFINED", "value": value}
    tensor = torch.tensor(estimates, dtype=torch.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "status": "OK",
        "reason": None,
        "value": float(value),
        "ci_low": float(torch.quantile(tensor, alpha).item()),
        "ci_high": float(torch.quantile(tensor, 1.0 - alpha).item()),
        "valid_repetitions": len(estimates),
        "requested_repetitions": int(repetitions),
        "bootstrap_unit": "image",
        "seed": int(seed),
        "confidence": float(confidence),
    }


def new_ecst_causal_accumulator() -> dict[str, Any]:
    return {
        "batch_count": 0,
        "sample_count": 0,
        "scalar_sums": defaultdict(float),
        "scalar_counts": defaultdict(int),
    }


def accumulate_ecst_causal_batch(
    accumulator: dict[str, Any],
    batch_size: int,
    metrics: Mapping[str, Any],
) -> None:
    accumulator["batch_count"] += 1
    accumulator["sample_count"] += int(batch_size)
    for name, value in metrics.items():
        if value is None or isinstance(value, (dict, list, tuple, str, bool)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            accumulator["scalar_sums"][name] += numeric * int(batch_size)
            accumulator["scalar_counts"][name] += int(batch_size)


def finalize_ecst_causal_audit(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    scalar_sums = accumulator["scalar_sums"]
    scalar_counts = accumulator["scalar_counts"]
    means = {
        name: scalar_sums[name] / scalar_counts[name]
        for name in sorted(scalar_sums)
        if scalar_counts[name] > 0
    }
    return {
        "batch_count": int(accumulator["batch_count"]),
        "sample_count": int(accumulator["sample_count"]),
        **means,
    }
