"""Token labelling and metric helpers for BISA-v0."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


RESPONSE_FIELDS = (
    "c_self_raw",
    "c_local_raw",
    "c_self_centered",
    "c_local_centered",
)


def downsample_gt_occupancy(gt: torch.Tensor, height: int = 37, width: int = 37) -> torch.Tensor:
    """Area-pool a GT mask to token occupancy without using it in a forward path."""
    gt = torch.as_tensor(gt).float()
    if gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    elif gt.ndim == 3:
        gt = gt.unsqueeze(0)
    if gt.ndim != 4 or int(gt.shape[1]) != 1:
        raise RuntimeError(f"GT must be [H,W], [1,H,W], or [B,1,H,W], got {list(gt.shape)}")
    if not bool(torch.isfinite(gt).all().item()):
        raise RuntimeError("GT contains NaN/Inf")
    gt = gt.clamp(0.0, 1.0)
    return F.adaptive_avg_pool2d(gt, output_size=(int(height), int(width))).squeeze(0).squeeze(0)


def strict_gt_labels(
    occupancy: torch.Tensor,
    *,
    foreground_threshold: float = 0.75,
    background_threshold: float = 0.25,
) -> torch.Tensor:
    if float(background_threshold) >= float(foreground_threshold):
        raise ValueError("Strict GT background threshold must be below foreground threshold")
    labels = torch.full_like(occupancy, -1, dtype=torch.int8)
    labels[occupancy >= float(foreground_threshold)] = 1
    labels[occupancy <= float(background_threshold)] = 0
    return labels


def classify_teacher_errors(
    teacher_prob: torch.Tensor,
    gt_labels: torch.Tensor,
) -> torch.Tensor:
    """Encode ignore=-1, TN=0, FP=1, FN=2, TP=3."""
    if teacher_prob.shape != gt_labels.shape:
        raise RuntimeError("Teacher probability and GT labels do not align")
    teacher_binary = teacher_prob > 0.5
    valid = gt_labels >= 0
    result = torch.full_like(gt_labels, -1, dtype=torch.int8)
    result[valid & (~teacher_binary) & (gt_labels == 0)] = 0
    result[valid & teacher_binary & (gt_labels == 0)] = 1
    result[valid & (~teacher_binary) & (gt_labels == 1)] = 2
    result[valid & teacher_binary & (gt_labels == 1)] = 3
    return result


def teacher_entropy(probability: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = probability.float().clamp(float(eps), 1.0 - float(eps))
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def safe_binary_metrics(labels, scores, threshold: float = 0.0) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "sample_count": int(labels.size),
        }
    prediction = scores > float(threshold)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sample_count": int(labels.size),
    }


def response_task_arrays(frame, task: str, score_column: str):
    task = str(task).lower()
    if task == "tp_fp":
        subset = frame[frame["teacher_error_type"].isin(["TP", "FP"])]
        labels = (subset["teacher_error_type"] == "TP").astype(np.int8).to_numpy()
    elif task == "fn_tn":
        subset = frame[frame["teacher_error_type"].isin(["FN", "TN"])]
        labels = (subset["teacher_error_type"] == "FN").astype(np.int8).to_numpy()
    else:
        raise ValueError(f"Unsupported BISA task: {task}")
    return subset, labels, subset[score_column].to_numpy(dtype=np.float64)


def json_finite(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
