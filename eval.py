import argparse
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label as connected_component_label
from torch.utils.data import DataLoader

from common.dataset import CachedEvalDataset
from common.metrics import CODMetrics
from common.utils import (
    Logger,
    check_cacd_feature_cache,
    check_ml_feature_cache,
    config_to_dict,
    ensure_dir,
    ensure_cache_available,
    format_metric_table,
    load_config,
    torch_load,
    write_yaml,
)
from model import build_seg_head
from models.online_dino_last4 import FrozenDINOv1Last4Extractor


PROBABILITY_VALLEY_BINS = 256
PROBABILITY_VALLEY_SMOOTH_SIGMA = 2.0
PROBABILITY_VALLEY_SEARCH_RANGE = (0.40, 0.65)
PROBABILITY_VALLEY_CLIP_RANGE = (0.45, 0.60)
PROBABILITY_VALLEY_SHRINK = 0.50
PROBABILITY_VALLEY_MIN_PEAK_SEPARATION = 0.15
PROBABILITY_VALLEY_MAX_DEPTH_RATIO = 0.80
PROBABILITY_VALLEY_MIN_FOREGROUND_MASS = 0.002
FLOAT_OTSU_BINS = 256
LOGIT_TRIANGLE_BINS = 256
LOGIT_TRIANGLE_PROBABILITY_EPS = 1e-4
LOGIT_GMM_BINS = 512
LOGIT_GMM_PROBABILITY_EPS = 1e-4
LOGIT_GMM_MAX_ITERATIONS = 64
LOGIT_GMM_TOLERANCE = 1e-6
LOGIT_GMM_REG_COVAR = 1e-3
LOGIT_GMM_MIN_COMPONENT_WEIGHT = 0.002
LOGIT_GMM_MIN_STANDARDIZED_SEPARATION = 0.50


def infer_in_channels(student_state):
    # 从 checkpoint 的 head 权重反推 feature channel 数，兼容 simple/DAGP/context_residual。
    if "adapters.f9.0.weight" in student_state:
        weight = student_state["adapters.f9.0.weight"]
    elif "consensus_encoder.proj12.0.weight" in student_state:
        weight = student_state["consensus_encoder.proj12.0.weight"]
    elif "coarse_path.base_head.weight" in student_state:
        weight = student_state["coarse_path.base_head.weight"]
    elif "csd_residual.csd_feat_proj.0.weight" in student_state:
        weight = student_state["csd_residual.csd_feat_proj.0.weight"]
    elif "sem_proj.0.weight" in student_state:
        weight = student_state["sem_proj.0.weight"]
    elif "anchor.weight" in student_state:
        weight = student_state["anchor.weight"]
    elif "reduce.weight" in student_state:
        weight = student_state["reduce.weight"]
    elif "feature_projection.weight" in student_state:
        weight = student_state["feature_projection.weight"]
    elif "base_head.weight" in student_state:
        weight = student_state["base_head.weight"]
    elif "proj.weight" in student_state:
        weight = student_state["proj.weight"]
    elif "base.weight" in student_state:
        weight = student_state["base.weight"]
    elif "base_head.proj.weight" in student_state:
        weight = student_state["base_head.proj.weight"]
    elif "decoupling.weight" in student_state:
        # DBA: the first 1x1 projection is [2 * embed_dim, in_channels, 1, 1].
        weight = student_state["decoupling.weight"]
    elif "proj12.conv.weight" in student_state:
        weight = student_state["proj12.conv.weight"]
    else:
        raise KeyError("Cannot infer in_channels from checkpoint student state.")
    channels = int(weight.shape[1])
    if "proj.weight" in student_state and channels > 1024 and channels % 4 == 0:
        # Last4LinearProbe concatenates four equal-width DINO features.
        channels //= 4
    return channels


def use_multi_level_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))


def use_dagp_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp"


def use_csd_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "csd_v1"


def use_csd_v1r_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe_csd_v1r"


def use_hr_bfr(cfg):
    return bool(getattr(cfg, "USE_HR_BFR", False))


def use_cssd(cfg):
    return bool(getattr(cfg, "USE_CSSD", False))


def use_cacd(cfg):
    return bool(getattr(cfg, "USE_CACD", False)) or str(
        getattr(cfg, "HEAD_TYPE", "simple")
    ).lower() == "cacd_v1_base"


def use_raw_feature_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {
        "dagp",
        "dagp_safe",
        "csd_v1",
        "dagp_safe_csd_v1r",
        "cacd_v1_base",
        "lcic_linear",
        "lcic",
        "lcic_dwlite",
        "lcic_conv3x3",
    }


def apply_lcic_inference_scales(
    cfg,
    student,
    consensus_scale=1.0,
    innovation_scale=1.0,
):
    """Apply one-run LCIC branch gains to the loaded in-memory checkpoint."""

    consensus_scale = float(consensus_scale)
    innovation_scale = float(innovation_scale)
    for name, value in (
        ("consensus_scale", consensus_scale),
        ("innovation_scale", innovation_scale),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"LCIC {name} must be finite and non-negative, got {value}."
            )
    lcic_enabled = bool(getattr(cfg, "GBSP_LCIC_V1", False))
    if not lcic_enabled:
        if consensus_scale != 1.0 or innovation_scale != 1.0:
            raise RuntimeError(
                "LCIC inference scales can only be used with an LCIC config."
            )
        return None
    lcic_variant = str(getattr(cfg, "LCIC_VARIANT", "")).lower()
    if lcic_variant != "d_full":
        if consensus_scale != 1.0 or innovation_scale != 1.0:
            raise RuntimeError(
                "Manual LCIC inference scales require "
                "LCIC_VARIANT='d_full'; feature-space decoders such as "
                f"{lcic_variant!r} have no consensus/innovation logit branches."
            )
        # GBSP_LCIC_V1 is also the legacy routing flag for the feature-space
        # DW-Lite control.  Default 1x/1x scales are a no-op for that decoder.
        return None
    target = student.module if hasattr(student, "module") else student
    if bool(getattr(target, "adaptive_gate_enabled", False)):
        if consensus_scale != 1.0 or innovation_scale != 1.0:
            raise RuntimeError(
                "LCIC adaptive-gate checkpoints forbid manual inference scales."
            )
        return None
    if getattr(target, "alpha", None) is None or getattr(
        target, "beta", None
    ) is None:
        raise RuntimeError("LCIC-D checkpoint is missing alpha or beta.")
    original_alpha = float(target.alpha.detach().item())
    original_beta = float(target.beta.detach().item())
    intrinsic_consensus_gain = float(
        getattr(target, "consensus_gain", 1.0)
    )
    intrinsic_innovation_gain = float(
        getattr(target, "innovation_gain", 1.0)
    )
    with torch.no_grad():
        target.alpha.mul_(consensus_scale)
        target.beta.mul_(innovation_scale)
    return {
        "consensus_scale": consensus_scale,
        "innovation_scale": innovation_scale,
        "original_alpha": original_alpha,
        "original_beta": original_beta,
        "intrinsic_consensus_gain": intrinsic_consensus_gain,
        "intrinsic_innovation_gain": intrinsic_innovation_gain,
        "effective_alpha": (
            intrinsic_consensus_gain * float(target.alpha.detach().item())
        ),
        "effective_beta": (
            intrinsic_innovation_gain * float(target.beta.detach().item())
        ),
    }


def use_ndr_branch(cfg):
    return bool(getattr(cfg, "USE_NDR_BRANCH", False))


def use_hsd_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "hsd_v1"


def use_last4_linear_probe(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "last4_linear"


def use_f12_scalelift_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "f12_scalelift"


def use_bcrd_sem_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "bcrd_sem_v1"


def use_last4_feature_decoder(cfg):
    return (
        use_hsd_decoder(cfg)
        or use_last4_linear_probe(cfg)
        or use_f12_scalelift_decoder(cfg)
        or use_bcrd_sem_decoder(cfg)
    )


def use_online_dino_last4(cfg):
    return bool(getattr(cfg, "ONLINE_DINO_LAST4", False))


def make_model_input(cfg, batch, device, online_dino=None):
    if use_last4_feature_decoder(cfg):
        if use_online_dino_last4(cfg):
            if online_dino is None:
                raise RuntimeError(
                    "ONLINE_DINO_LAST4 evaluation requires a frozen extractor."
                )
            if "dino_input_296" not in batch:
                raise KeyError("Online DINO eval batch is missing dino_input_296.")
            inputs = batch["dino_input_296"].to(
                device, non_blocking=True
            ).float()
            return online_dino(inputs)
        required = tuple(f"feature_l{layer}" for layer in (9, 10, 11, 12))
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"DINO last-four eval batch is missing: {missing}")
        return {
            f"f{layer}": batch[f"feature_l{layer}"].to(
                device, non_blocking=True
            ).float()
            for layer in (9, 10, 11, 12)
        }
    if use_cacd(cfg):
        required = ("feature_l10", "feature_l11", "feature")
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"CACD eval batch missing feature fields: {missing}")
        return {
            "f10": batch["feature_l10"].to(device, non_blocking=True).float(),
            "f11": batch["feature_l11"].to(device, non_blocking=True).float(),
            "f12": batch["feature"].to(device, non_blocking=True).float(),
        }
    if use_multi_level_feature(cfg):
        return {
            f"l{int(layer)}": batch[f"feature_l{int(layer)}"].to(device, non_blocking=True).float()
            for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])
        }
    feature = batch["feature"].to(device, non_blocking=True).float()
    if use_raw_feature_head(cfg):
        return feature
    return F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")


def make_image_68(cfg, batch, device):
    if not (use_ndr_branch(cfg) or use_csd_head(cfg) or use_csd_v1r_head(cfg) or use_cacd(cfg)):
        return None
    if "image_68" not in batch:
        raise KeyError("NDR/CSD decoder requires batch['image_68'].")
    return batch["image_68"].to(device, non_blocking=True).float()


def make_sobel_68(cfg, batch, device):
    if not use_cacd(cfg):
        return None
    if "sobel_68" not in batch:
        raise KeyError("CACD eval requires batch['sobel_68'].")
    return batch["sobel_68"].to(device, non_blocking=True).float()


def make_image_136(cfg, batch, device):
    if not use_hr_bfr(cfg):
        return None
    if "image_136" not in batch:
        raise KeyError("HR-BFR eval requires batch['image_136'].")
    return batch["image_136"].to(device, non_blocking=True).float()


def make_image_148(cfg, batch, device):
    if not (use_hsd_decoder(cfg) and bool(getattr(cfg, "USE_DETAIL", False))):
        return None
    if "image_148" not in batch:
        raise KeyError("HSD-Full eval requires batch['image_148'].")
    return batch["image_148"].to(device, non_blocking=True).float()


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def extract_logits_for_eval(output, cfg):
    if isinstance(output, dict) and use_hr_bfr(cfg) and bool(getattr(cfg, "HR_BFR_USE_HR_LOGITS_FOR_EVAL", True)):
        if "hr_logits" not in output:
            raise KeyError("HR_BFR_USE_HR_LOGITS_FOR_EVAL=True but model output has no hr_logits.")
        return output["hr_logits"], "hr_logits"
    if isinstance(output, dict):
        if "final_logits" in output:
            return output["final_logits"], "final_logits"
        return output["logits"], "logits"
    return output, "tensor"


def output_scalar(output, key, default=0.0):
    if not isinstance(output, dict) or key not in output:
        return float(default)
    value = output[key]
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def dataloader_worker_kwargs(cfg):
    num_workers = int(cfg.NUM_WORKERS)
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "DATALOADER_PERSISTENT_WORKERS", False))
        prefetch_factor = int(getattr(cfg, "DATALOADER_PREFETCH_FACTOR", 2))
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def save_pred_png(path, pred):
    ensure_dir(Path(path).parent)
    array = pred.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


def save_probability_png(path, probability):
    """Save a continuous probability map as an 8-bit grayscale PNG."""

    ensure_dir(Path(path).parent)
    array = probability.detach().float().cpu().squeeze().numpy()
    array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def parse_dataset_threshold_overrides(items, allowed_datasets):
    """Parse repeated DATASET=THRESHOLD CLI overrides."""

    allowed = {str(name) for name in allowed_datasets}
    overrides = {}
    for item in items or ():
        if "=" not in str(item):
            raise ValueError(
                "--dataset_threshold must use DATASET=THRESHOLD syntax, "
                f"got {item!r}."
            )
        dataset_name, raw_threshold = str(item).split("=", 1)
        dataset_name = dataset_name.strip()
        if dataset_name not in allowed:
            raise ValueError(
                f"Unknown dataset in --dataset_threshold: {dataset_name!r}."
            )
        if dataset_name in overrides:
            raise ValueError(
                f"Duplicate --dataset_threshold for {dataset_name}."
            )
        try:
            threshold = float(raw_threshold)
        except ValueError as error:
            raise ValueError(
                f"Invalid threshold for {dataset_name}: {raw_threshold!r}."
            ) from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid evaluation threshold for {dataset_name}: {threshold}."
            )
        overrides[dataset_name] = threshold
    return overrides


def _smooth_histogram(histogram, sigma):
    radius = max(1, int(round(3.0 * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(histogram.astype(np.float64), kernel, mode="same")


def probability_valley_threshold(probability, fallback_threshold=0.50):
    """Select a conservative per-image threshold from a probability valley."""

    fallback = float(fallback_threshold)
    values = torch.as_tensor(probability).detach().float().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    diagnostics = {
        "fallback": True,
        "reason": "insufficient_values",
        "raw_threshold": fallback,
        "threshold": fallback,
    }
    if values.size < 16:
        return fallback, diagnostics
    values = np.clip(values, 0.0, 1.0)
    background_mass = float(np.mean(values <= 0.50))
    foreground_mass = float(np.mean(values > 0.50))
    if (
        background_mass < PROBABILITY_VALLEY_MIN_FOREGROUND_MASS
        or foreground_mass < PROBABILITY_VALLEY_MIN_FOREGROUND_MASS
    ):
        diagnostics["reason"] = "missing_probability_class"
        return fallback, diagnostics

    histogram, edges = np.histogram(
        values,
        bins=PROBABILITY_VALLEY_BINS,
        range=(0.0, 1.0),
    )
    smoothed = _smooth_histogram(
        histogram, PROBABILITY_VALLEY_SMOOTH_SIGMA
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    left_indices = np.flatnonzero(centers < 0.50)
    right_indices = np.flatnonzero(centers >= 0.50)
    if left_indices.size == 0 or right_indices.size == 0:
        diagnostics["reason"] = "missing_histogram_side"
        return fallback, diagnostics

    left_peak = int(left_indices[np.argmax(smoothed[left_indices])])
    right_peak = int(right_indices[np.argmax(smoothed[right_indices])])
    peak_separation = float(centers[right_peak] - centers[left_peak])
    if peak_separation < PROBABILITY_VALLEY_MIN_PEAK_SEPARATION:
        diagnostics["reason"] = "peaks_too_close"
        return fallback, diagnostics

    search_low, search_high = PROBABILITY_VALLEY_SEARCH_RANGE
    valley_indices = np.flatnonzero(
        (centers > centers[left_peak])
        & (centers < centers[right_peak])
        & (centers >= search_low)
        & (centers <= search_high)
    )
    if valley_indices.size == 0:
        diagnostics["reason"] = "no_valley_in_search_range"
        return fallback, diagnostics

    valley_density = smoothed[valley_indices]
    minimum_density = float(valley_density.min())
    tolerance = max(1e-12, 0.02 * minimum_density)
    minimum_candidates = valley_indices[
        valley_density <= minimum_density + tolerance
    ]
    peak_midpoint = 0.5 * (centers[left_peak] + centers[right_peak])
    valley_index = int(
        minimum_candidates[
            np.argmin(np.abs(centers[minimum_candidates] - peak_midpoint))
        ]
    )
    smaller_peak = float(min(smoothed[left_peak], smoothed[right_peak]))
    depth_ratio = (
        float(smoothed[valley_index]) / smaller_peak
        if smaller_peak > 0.0
        else float("inf")
    )
    if depth_ratio > PROBABILITY_VALLEY_MAX_DEPTH_RATIO:
        diagnostics["reason"] = "valley_not_deep_enough"
        return fallback, diagnostics

    raw_threshold = float(centers[valley_index])
    threshold = fallback + PROBABILITY_VALLEY_SHRINK * (
        raw_threshold - fallback
    )
    clip_low, clip_high = PROBABILITY_VALLEY_CLIP_RANGE
    threshold = float(np.clip(threshold, clip_low, clip_high))
    diagnostics.update(
        {
            "fallback": False,
            "reason": "ok",
            "raw_threshold": raw_threshold,
            "threshold": threshold,
            "left_peak": float(centers[left_peak]),
            "right_peak": float(centers[right_peak]),
            "depth_ratio": depth_ratio,
        }
    )
    return threshold, diagnostics


def float_otsu_threshold(probability, fallback_threshold=0.50):
    """Compute per-image Otsu directly from float probabilities.

    Unlike the EReCu inference implementation, this path never multiplies by
    255, rounds, or casts the prediction to uint8. Float values are accumulated
    into a 256-bin histogram over the observed per-image probability range.
    """

    fallback = float(fallback_threshold)
    values = torch.as_tensor(probability).detach().float().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    diagnostics = {
        "fallback": True,
        "reason": "insufficient_values",
        "raw_threshold": fallback,
        "threshold": fallback,
    }
    if values.size < 2:
        return fallback, diagnostics

    values = np.clip(values.astype(np.float64, copy=False), 0.0, 1.0)
    value_min = float(values.min())
    value_max = float(values.max())
    diagnostics.update({"value_min": value_min, "value_max": value_max})
    if value_max <= value_min:
        diagnostics["reason"] = "constant_probability_map"
        return fallback, diagnostics

    histogram, edges = np.histogram(
        values,
        bins=FLOAT_OTSU_BINS,
        range=(value_min, value_max),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = histogram.astype(np.float64)
    left_weight = np.cumsum(counts)
    right_weight = np.cumsum(counts[::-1])[::-1]
    left_sum = np.cumsum(counts * centers)
    right_sum = np.cumsum((counts * centers)[::-1])[::-1]

    valid = (left_weight[:-1] > 0.0) & (right_weight[1:] > 0.0)
    if not np.any(valid):
        diagnostics["reason"] = "missing_otsu_partition"
        return fallback, diagnostics

    scores = np.full(FLOAT_OTSU_BINS - 1, -np.inf, dtype=np.float64)
    left_mean = left_sum[:-1][valid] / left_weight[:-1][valid]
    right_mean = right_sum[1:][valid] / right_weight[1:][valid]
    scores[valid] = (
        left_weight[:-1][valid]
        * right_weight[1:][valid]
        * (left_mean - right_mean) ** 2
    )
    threshold = float(centers[int(np.argmax(scores))])
    diagnostics.update(
        {
            "fallback": False,
            "reason": "ok",
            "raw_threshold": threshold,
            "threshold": threshold,
        }
    )
    return threshold, diagnostics


def binary_hysteresis_mask(probability, low_threshold, high_threshold):
    """Keep low-threshold components that contain a high-threshold seed."""

    low = float(low_threshold)
    high = float(high_threshold)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(
            f"Invalid hysteresis thresholds: low={low}, high={high}."
        )
    tensor = torch.as_tensor(probability).detach().float()
    if tensor.numel() == 0:
        raise ValueError("Hysteresis probability map must not be empty.")
    original_shape = tensor.shape
    spatial = tensor.squeeze().cpu().numpy()
    if spatial.ndim != 2:
        raise ValueError(
            "Hysteresis expects one 2D probability map, got "
            f"shape={list(original_shape)}."
        )

    candidate = spatial > low
    seed = spatial > high
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = connected_component_label(
        candidate, structure=structure
    )
    seed_labels = np.unique(labels[seed])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size:
        output = np.isin(labels, seed_labels)
    else:
        output = np.zeros_like(candidate, dtype=bool)
    mask = torch.as_tensor(output, dtype=torch.float32, device=tensor.device).reshape(
        original_shape
    )
    diagnostics = {
        "low_threshold": low,
        "high_threshold": high,
        "candidate_area": float(candidate.mean()),
        "seed_area": float(seed.mean()),
        "output_area": float(output.mean()),
        "component_count": int(component_count),
        "kept_component_count": int(seed_labels.size),
        "missing_seed": bool(seed_labels.size == 0),
    }
    return mask, diagnostics


def otsu_anchor_hysteresis_mask(probability, anchor_threshold=0.50):
    """Combine per-image float Otsu with a stable anchor via hysteresis."""

    anchor = float(anchor_threshold)
    otsu, otsu_diagnostics = float_otsu_threshold(
        probability, fallback_threshold=anchor
    )
    low, high = min(otsu, anchor), max(otsu, anchor)
    mask, hysteresis_diagnostics = binary_hysteresis_mask(
        probability, low_threshold=low, high_threshold=high
    )
    fallback = bool(otsu_diagnostics["fallback"])
    reason = str(otsu_diagnostics["reason"])
    if hysteresis_diagnostics["missing_seed"]:
        mask = (torch.as_tensor(probability).detach().float() > anchor).float()
        fallback = True
        reason = "missing_hysteresis_seed"
    diagnostics = {
        **hysteresis_diagnostics,
        "fallback": fallback,
        "reason": reason,
        "raw_threshold": float(otsu),
        "threshold": float(otsu),
        "otsu_threshold": float(otsu),
        "anchor_threshold": anchor,
    }
    return mask, diagnostics


def logit_triangle_threshold(probability, fallback_threshold=0.50):
    """Compute a per-image Triangle threshold on float logits without GT."""

    fallback = float(fallback_threshold)
    values = torch.as_tensor(probability).detach().float().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    diagnostics = {
        "fallback": True,
        "reason": "insufficient_values",
        "raw_threshold": fallback,
        "threshold": fallback,
    }
    if values.size < 2:
        return fallback, diagnostics

    probability_eps = float(LOGIT_TRIANGLE_PROBABILITY_EPS)
    values = np.clip(
        values.astype(np.float64, copy=False),
        probability_eps,
        1.0 - probability_eps,
    )
    logits = np.log(values) - np.log1p(-values)
    logit_min = float(logits.min())
    logit_max = float(logits.max())
    diagnostics.update({"logit_min": logit_min, "logit_max": logit_max})
    if (
        not math.isfinite(logit_min)
        or not math.isfinite(logit_max)
        or logit_max <= logit_min
    ):
        diagnostics["reason"] = "constant_probability_map"
        return fallback, diagnostics

    histogram, edges = np.histogram(
        logits,
        bins=LOGIT_TRIANGLE_BINS,
        range=(logit_min, logit_max),
    )
    nonzero_indices = np.flatnonzero(histogram)
    if nonzero_indices.size < 2:
        diagnostics["reason"] = "insufficient_histogram_support"
        return fallback, diagnostics

    peak_index = int(np.argmax(histogram))
    low_index = int(nonzero_indices[0])
    high_index = int(nonzero_indices[-1])
    original_peak_index = peak_index
    flipped = peak_index - low_index < high_index - peak_index
    working_histogram = histogram
    if flipped:
        working_histogram = histogram[::-1]
        low_index = LOGIT_TRIANGLE_BINS - high_index - 1
        peak_index = LOGIT_TRIANGLE_BINS - peak_index - 1

    peak_width = int(peak_index - low_index)
    if peak_width <= 0:
        diagnostics["reason"] = "missing_triangle_tail"
        return fallback, diagnostics
    peak_height = float(working_histogram[peak_index])
    normalizer = math.hypot(peak_height, float(peak_width))
    if normalizer <= 0.0:
        diagnostics["reason"] = "degenerate_triangle"
        return fallback, diagnostics

    offsets = np.arange(peak_width, dtype=np.float64)
    tail_heights = working_histogram[
        low_index : low_index + peak_width
    ].astype(np.float64)
    distances = (
        (peak_height / normalizer) * offsets
        - (float(peak_width) / normalizer) * tail_heights
    )
    threshold_index = int(np.argmax(distances)) + low_index
    if flipped:
        threshold_index = LOGIT_TRIANGLE_BINS - threshold_index - 1

    centers = 0.5 * (edges[:-1] + edges[1:])
    threshold_logit = float(centers[threshold_index])
    threshold = float(1.0 / (1.0 + math.exp(-threshold_logit)))
    diagnostics.update(
        {
            "fallback": False,
            "reason": "ok",
            "raw_threshold": threshold,
            "threshold": threshold,
            "threshold_logit": threshold_logit,
            "peak_logit": float(centers[original_peak_index]),
            "tail_direction": "right" if flipped else "left",
        }
    )
    return threshold, diagnostics


def logit_gmm_threshold(probability, fallback_threshold=0.50):
    """Fit a deterministic two-component GMM to one logit probability map."""

    fallback = float(fallback_threshold)
    values = torch.as_tensor(probability).detach().float().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    diagnostics = {
        "fallback": True,
        "reason": "insufficient_values",
        "raw_threshold": fallback,
        "threshold": fallback,
    }
    if values.size < 16:
        return fallback, diagnostics

    probability_eps = float(LOGIT_GMM_PROBABILITY_EPS)
    values = np.clip(values.astype(np.float64, copy=False), probability_eps, 1.0 - probability_eps)
    logits = np.log(values) - np.log1p(-values)
    logit_min = float(logits.min())
    logit_max = float(logits.max())
    if not math.isfinite(logit_min) or not math.isfinite(logit_max) or logit_max <= logit_min:
        diagnostics["reason"] = "constant_probability_map"
        return fallback, diagnostics

    histogram, edges = np.histogram(
        logits,
        bins=LOGIT_GMM_BINS,
        range=(logit_min, logit_max),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = histogram.astype(np.float64)
    total_count = float(counts.sum())
    initial_background = centers < 0.0
    initial_foreground = ~initial_background
    initial_masses = np.asarray(
        [counts[initial_background].sum(), counts[initial_foreground].sum()],
        dtype=np.float64,
    )
    if np.any(initial_masses < LOGIT_GMM_MIN_COMPONENT_WEIGHT * total_count):
        diagnostics["reason"] = "missing_initial_probability_class"
        return fallback, diagnostics

    means = np.asarray(
        [
            np.sum(counts[initial_background] * centers[initial_background]) / initial_masses[0],
            np.sum(counts[initial_foreground] * centers[initial_foreground]) / initial_masses[1],
        ],
        dtype=np.float64,
    )
    variances = np.asarray(
        [
            np.sum(counts[initial_background] * (centers[initial_background] - means[0]) ** 2)
            / initial_masses[0],
            np.sum(counts[initial_foreground] * (centers[initial_foreground] - means[1]) ** 2)
            / initial_masses[1],
        ],
        dtype=np.float64,
    )
    variances = np.maximum(variances, LOGIT_GMM_REG_COVAR)
    weights = initial_masses / total_count
    previous_average_log_likelihood = None
    converged = False
    iterations = 0

    for iteration in range(LOGIT_GMM_MAX_ITERATIONS):
        log_density = (
            np.log(np.maximum(weights[:, None], np.finfo(np.float64).tiny))
            - 0.5 * np.log(2.0 * np.pi * variances[:, None])
            - 0.5 * (centers[None, :] - means[:, None]) ** 2 / variances[:, None]
        )
        max_log_density = np.max(log_density, axis=0)
        exp_density = np.exp(log_density - max_log_density[None, :])
        density_sum = np.maximum(exp_density.sum(axis=0), np.finfo(np.float64).tiny)
        responsibilities = exp_density / density_sum[None, :]
        weighted_responsibilities = responsibilities * counts[None, :]
        component_counts = weighted_responsibilities.sum(axis=1)
        if np.any(component_counts <= 0.0):
            diagnostics["reason"] = "empty_gmm_component"
            return fallback, diagnostics

        weights = component_counts / total_count
        means = (weighted_responsibilities * centers[None, :]).sum(axis=1) / component_counts
        variances = (
            weighted_responsibilities * (centers[None, :] - means[:, None]) ** 2
        ).sum(axis=1) / component_counts
        variances = np.maximum(variances, LOGIT_GMM_REG_COVAR)
        order = np.argsort(means)
        weights, means, variances = weights[order], means[order], variances[order]

        average_log_likelihood = float(
            np.sum(counts * (max_log_density + np.log(density_sum))) / total_count
        )
        iterations = iteration + 1
        if (
            previous_average_log_likelihood is not None
            and abs(average_log_likelihood - previous_average_log_likelihood)
            <= LOGIT_GMM_TOLERANCE
        ):
            converged = True
            break
        previous_average_log_likelihood = average_log_likelihood

    if np.any(weights < LOGIT_GMM_MIN_COMPONENT_WEIGHT):
        diagnostics["reason"] = "small_gmm_component"
        return fallback, diagnostics
    standardized_separation = float(
        (means[1] - means[0]) / math.sqrt(variances[0] + variances[1])
    )
    if standardized_separation < LOGIT_GMM_MIN_STANDARDIZED_SEPARATION:
        diagnostics["reason"] = "gmm_components_not_separated"
        return fallback, diagnostics

    candidates = np.linspace(means[0], means[1], 4097, dtype=np.float64)
    log_background = (
        math.log(weights[0])
        - 0.5 * math.log(2.0 * math.pi * variances[0])
        - 0.5 * (candidates - means[0]) ** 2 / variances[0]
    )
    log_foreground = (
        math.log(weights[1])
        - 0.5 * math.log(2.0 * math.pi * variances[1])
        - 0.5 * (candidates - means[1]) ** 2 / variances[1]
    )
    threshold_logit = float(candidates[int(np.argmin(np.abs(log_foreground - log_background)))])
    threshold = float(1.0 / (1.0 + math.exp(-threshold_logit)))
    diagnostics.update(
        {
            "fallback": False,
            "reason": "ok",
            "raw_threshold": threshold,
            "threshold": threshold,
            "background_weight": float(weights[0]),
            "foreground_weight": float(weights[1]),
            "background_logit_mean": float(means[0]),
            "foreground_logit_mean": float(means[1]),
            "background_logit_std": float(math.sqrt(variances[0])),
            "foreground_logit_std": float(math.sqrt(variances[1])),
            "standardized_separation": standardized_separation,
            "iterations": iterations,
            "converged": converged,
        }
    )
    return threshold, diagnostics


def resolve_eval_binary_threshold(
    cfg,
    dataset_name,
    threshold_overrides=None,
    default_threshold=None,
):
    """Return one dataset threshold, with optional one-run CLI overrides."""

    default = float(
        cfg.THRESHOLD if default_threshold is None else default_threshold
    )
    if not math.isfinite(default) or not 0.0 <= default <= 1.0:
        raise ValueError(f"Invalid default evaluation threshold: {default}.")
    raw_mapping = threshold_overrides or {}
    if not isinstance(raw_mapping, Mapping):
        raise TypeError("threshold_overrides must be a mapping.")
    threshold = float(raw_mapping.get(str(dataset_name), default))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Invalid evaluation threshold for {dataset_name}: {threshold}."
        )
    return threshold


def finalize_eval_metric_results(metrics, binary_reference_metrics=None):
    """Finalize main metrics and preserve hard-mask-only ACC/mIoU semantics."""

    result = metrics.get_result()
    binary_reference_result = None
    if binary_reference_metrics is not None:
        binary_reference_result = binary_reference_metrics.get_result()
        result["ACC"] = binary_reference_result["ACC"]
        result["mIOU"] = binary_reference_result["mIOU"]
    return result, binary_reference_result


@torch.no_grad()
def eval_dataset(
    cfg,
    student,
    dataset_name,
    device,
    out_dir,
    logger,
    max_samples=-1,
    online_dino=None,
    threshold_overrides=None,
    default_threshold=None,
    adaptive_threshold="none",
    metric_input="binary",
):
    dataset = CachedEvalDataset(
        cfg,
        split="test",
        datasets=[dataset_name],
        max_samples=int(max_samples),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(cfg),
    )
    metric_input = str(metric_input).strip().lower()
    if metric_input not in {"binary", "probability"}:
        raise ValueError(f"Unknown metric input mode: {metric_input!r}.")
    metrics = CODMetrics()
    binary_reference_metrics = CODMetrics() if metric_input == "probability" else None
    pred_dir = Path(out_dir) / "pred" / dataset_name
    student.eval()
    hr_bfr_eval_batches = 0
    hr_bfr_skip_ratio_sum = 0.0
    hr_bfr_valid_ratio_sum = 0.0
    hr_bfr_active_pixel_ratio_sum = 0.0
    hr_bfr_band_ratio_sum = 0.0
    hr_bfr_band_ratio_max = 0.0
    cssd_single_view_logged = False
    cacd_logged = False
    binary_threshold = resolve_eval_binary_threshold(
        cfg,
        dataset_name,
        threshold_overrides,
        default_threshold=default_threshold,
    )
    adaptive_threshold = str(adaptive_threshold).strip().lower()
    logger.log(
        f"[EvalThreshold] Dataset: {dataset_name} | "
        f"mode={adaptive_threshold} | fallback_threshold={binary_threshold:.6f}"
    )
    threshold_values = []
    raw_threshold_values = []
    threshold_fallbacks = 0
    logit_gmm_diagnostics = []
    otsu_hysteresis_diagnostics = []
    prediction_area_sum = 0.0
    probability_mean_sum = 0.0
    soft_hard_gap_sum = 0.0
    ambiguous_ratio_01_09_sum = 0.0
    ambiguous_ratio_04_06_sum = 0.0
    evaluated_images = 0

    for batch in loader:
        if bool(getattr(cfg, "USE_AP_STCR", False)) or str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).strip().lower() == "ap_stcr":
            forbidden_training_fields = {
                "pu_target_soft_37",
                "pu_bg_anchor_37",
                "sample_index",
            }.intersection(batch)
            if forbidden_training_fields:
                raise RuntimeError(
                    "AP-STCR training-only state appeared in eval batch: "
                    f"{sorted(forbidden_training_fields)}"
                )
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = batch["stem"][0]
        model_input = make_model_input(
            cfg, batch, device, online_dino=online_dino
        )
        image_68 = make_image_68(cfg, batch, device)
        sobel_68 = make_sobel_68(cfg, batch, device)
        image_136 = make_image_136(cfg, batch, device)
        image_148 = make_image_148(cfg, batch, device)
        if use_cssd(cfg):
            if not bool(getattr(cfg, "CSSD_STRICT_SINGLE_VIEW_EVAL", True)):
                raise RuntimeError("CSSD-v1a eval requires CSSD_STRICT_SINGLE_VIEW_EVAL=True.")
            cssd_field = str(getattr(cfg, "CSSD_HR_FEATURE_FIELD", "feature_cssd_hr"))
            if cssd_field in batch:
                raise RuntimeError(
                    f"CSSD train-only high feature {cssd_field!r} appeared in CachedEvalDataset."
                )
            if not torch.is_tensor(model_input) or list(model_input.shape[1:]) != [384, 37, 37]:
                raise RuntimeError(
                    "CSSD eval must use only normal cached DINO [B,384,37,37], got "
                    f"{list(model_input.shape) if torch.is_tensor(model_input) else type(model_input).__name__}."
                )
            if not cssd_single_view_logged:
                logger.log(
                    f"[Eval CSSD] Dataset: {dataset_name} | "
                    "high_resolution_view_used=False | high_cache_read=False | "
                    f"feature_source=normal_cache | feature_shape={list(model_input.shape)} | "
                    "logits_source=normal_final_logits | test_time_scale_fusion=False"
                )
                cssd_single_view_logged = True
        if use_hsd_decoder(cfg):
            output = student(model_input, image_148=image_148)
            logits, _ = extract_logits_for_eval(output, cfg)
        elif str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {
            "dagp_safe",
            "csd_v1",
            "dagp_safe_csd_v1r",
            "cacd_v1_base",
        }:
            if use_cacd(cfg):
                output = student(
                    model_input,
                    image_68=image_68,
                    sobel_68=sobel_68,
                    return_aux=False,
                )
                if not cacd_logged:
                    logger.log(
                        f"[Eval CACD] Dataset: {dataset_name} | USE_CACD=True | "
                        f"CACD_VERSION={getattr(cfg, 'CACD_VERSION')} | "
                        f"feature_l10 shape={list(model_input['f10'].shape)} | "
                        f"feature_l11 shape={list(model_input['f11'].shape)} | "
                        f"feature_l12 shape={list(model_input['f12'].shape)} | "
                        f"final_logits shape={list(output['final_logits'].shape)} | "
                        "logits_source=final_logits | extra_inference_branch=False | post_processing=False"
                    )
                    cacd_logged = True
            elif use_csd_v1r_head(cfg):
                output = student(model_input, image_68=image_68, image_136=image_136, return_aux=False)
            else:
                output = student(model_input, image_68=image_68, return_aux=False)
            logits, logits_name = extract_logits_for_eval(output, cfg)
            if use_hr_bfr(cfg) and isinstance(output, dict):
                hr_bfr_eval_batches += 1
                hr_bfr_skip_ratio_sum += output_scalar(output, "hr_skip_img_ratio", 0.0)
                hr_bfr_valid_ratio_sum += output_scalar(output, "hr_valid_img_ratio", 1.0)
                hr_bfr_active_pixel_ratio_sum += output_scalar(output, "hr_active_pixel_ratio", 0.0)
                hr_bfr_band_ratio_sum += output_scalar(output, "hr_band_ratio_raw_136_mean", 0.0)
                hr_bfr_band_ratio_max = max(
                    hr_bfr_band_ratio_max,
                    output_scalar(output, "hr_band_ratio_raw_136_max", 0.0),
                )
            if stem == batch["stem"][0] and not hasattr(eval_dataset, "_hr_bfr_logged"):
                logger.log(f"[Eval] USE_HR_BFR={bool(getattr(cfg, 'USE_HR_BFR', False))}")
                logger.log(
                    "[Eval] HR_BFR_USE_HR_LOGITS_FOR_EVAL="
                    f"{bool(getattr(cfg, 'HR_BFR_USE_HR_LOGITS_FOR_EVAL', True))}"
                )
                logger.log(
                    "[Eval] HR_BFR_EVAL_RES_SCALE="
                    f"{float(getattr(cfg, 'HR_BFR_EVAL_RES_SCALE', 1.0)):.6f}"
                )
                logger.log(f"[Eval] logits_for_eval={logits_name}")
                logger.log(f"[Eval] logits shape={list(logits.shape)}")
                eval_dataset._hr_bfr_logged = True
        else:
            logits = extract_logits(student(model_input))
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear")
        probability = logits.sigmoid()
        image_threshold = binary_threshold
        adaptive_prediction = None
        if adaptive_threshold in {
            "probability_valley",
            "otsu_float",
            "triangle_logit",
            "logit_gmm",
        }:
            threshold_functions = {
                "probability_valley": probability_valley_threshold,
                "otsu_float": float_otsu_threshold,
                "triangle_logit": logit_triangle_threshold,
                "logit_gmm": logit_gmm_threshold,
            }
            threshold_fn = threshold_functions[adaptive_threshold]
            image_threshold, threshold_diagnostics = threshold_fn(
                probability, fallback_threshold=binary_threshold
            )
            raw_threshold_values.append(
                float(threshold_diagnostics["raw_threshold"])
            )
            threshold_fallbacks += int(threshold_diagnostics["fallback"])
            if adaptive_threshold == "logit_gmm" and not threshold_diagnostics["fallback"]:
                logit_gmm_diagnostics.append(threshold_diagnostics)
        elif adaptive_threshold == "otsu_hysteresis":
            adaptive_prediction, threshold_diagnostics = otsu_anchor_hysteresis_mask(
                probability, anchor_threshold=binary_threshold
            )
            image_threshold = float(threshold_diagnostics["otsu_threshold"])
            raw_threshold_values.append(image_threshold)
            threshold_fallbacks += int(threshold_diagnostics["fallback"])
            otsu_hysteresis_diagnostics.append(threshold_diagnostics)
        threshold_values.append(float(image_threshold))
        pred = (
            adaptive_prediction
            if adaptive_prediction is not None
            else (probability > image_threshold).float()
        )
        prediction_area_sum += float(pred.mean().item())
        probability_mean_sum += float(probability.mean().item())
        soft_hard_gap_sum += float(torch.abs(probability - pred).mean().item())
        ambiguous_ratio_01_09_sum += float(
            ((probability > 0.10) & (probability < 0.90)).float().mean().item()
        )
        ambiguous_ratio_04_06_sum += float(
            ((probability > 0.40) & (probability < 0.60)).float().mean().item()
        )
        evaluated_images += 1
        if metric_input == "probability":
            save_probability_png(pred_dir / f"{stem}.png", probability)
            metrics.step(gt, probability)
            binary_reference_metrics.step(gt, pred)
        else:
            save_pred_png(pred_dir / f"{stem}.png", pred)
            metrics.step(gt, pred)

    result, binary_reference_result = finalize_eval_metric_results(
        metrics, binary_reference_metrics
    )
    logger.log(f"[Eval] Dataset: {dataset_name}")
    if threshold_values:
        threshold_array = np.asarray(threshold_values, dtype=np.float64)
        threshold_label = (
            "adaptive" if adaptive_threshold != "none" else f"{binary_threshold:g}"
        )
        logger.log(
            f"[EvalThresholdSummary] Dataset: {dataset_name} | "
            f"mode={adaptive_threshold} | mean={threshold_array.mean():.6f} | "
            f"std={threshold_array.std():.6f} | min={threshold_array.min():.6f} | "
            f"max={threshold_array.max():.6f} | "
            f"fallback_ratio={threshold_fallbacks / max(evaluated_images, 1):.6f} | "
            f"probability_mean={probability_mean_sum / max(evaluated_images, 1):.6f} | "
            f"prediction_area={prediction_area_sum / max(evaluated_images, 1):.6f} | "
            f"soft_hard_gap={soft_hard_gap_sum / max(evaluated_images, 1):.6f} | "
            f"ambiguous_ratio_01_09={ambiguous_ratio_01_09_sum / max(evaluated_images, 1):.6f} | "
            f"ambiguous_ratio_04_06={ambiguous_ratio_04_06_sum / max(evaluated_images, 1):.6f}"
        )
        if logit_gmm_diagnostics:
            logger.log(
                f"[EvalLogitGMMSummary] Dataset: {dataset_name} | "
                f"background_weight={np.mean([item['background_weight'] for item in logit_gmm_diagnostics]):.6f} | "
                f"foreground_weight={np.mean([item['foreground_weight'] for item in logit_gmm_diagnostics]):.6f} | "
                f"background_logit_mean={np.mean([item['background_logit_mean'] for item in logit_gmm_diagnostics]):.6f} | "
                f"foreground_logit_mean={np.mean([item['foreground_logit_mean'] for item in logit_gmm_diagnostics]):.6f} | "
                f"standardized_separation={np.mean([item['standardized_separation'] for item in logit_gmm_diagnostics]):.6f} | "
                f"converged_ratio={np.mean([float(item['converged']) for item in logit_gmm_diagnostics]):.6f} | "
                f"mean_iterations={np.mean([item['iterations'] for item in logit_gmm_diagnostics]):.3f}"
            )
        if otsu_hysteresis_diagnostics:
            logger.log(
                f"[EvalOtsuHysteresisSummary] Dataset: {dataset_name} | "
                f"low_threshold={np.mean([item['low_threshold'] for item in otsu_hysteresis_diagnostics]):.6f} | "
                f"high_threshold={np.mean([item['high_threshold'] for item in otsu_hysteresis_diagnostics]):.6f} | "
                f"candidate_area={np.mean([item['candidate_area'] for item in otsu_hysteresis_diagnostics]):.6f} | "
                f"seed_area={np.mean([item['seed_area'] for item in otsu_hysteresis_diagnostics]):.6f} | "
                f"output_area={np.mean([item['output_area'] for item in otsu_hysteresis_diagnostics]):.6f} | "
                f"component_count={np.mean([item['component_count'] for item in otsu_hysteresis_diagnostics]):.3f} | "
                f"kept_component_count={np.mean([item['kept_component_count'] for item in otsu_hysteresis_diagnostics]):.3f}"
            )
    else:
        threshold_label = f"{binary_threshold:g}"
    if use_hr_bfr(cfg):
        denom = max(hr_bfr_eval_batches, 1)
        logger.log(
            f"[Eval HR-BFR] Dataset: {dataset_name} | "
            f"fallback_ratio={hr_bfr_skip_ratio_sum / denom:.8f} | "
            f"valid_img_ratio={hr_bfr_valid_ratio_sum / denom:.8f} | "
            f"band_ratio_mean={hr_bfr_band_ratio_sum / denom:.8f} | "
            f"band_ratio_max={hr_bfr_band_ratio_max:.8f} | "
            f"hr_active_pixel_ratio={hr_bfr_active_pixel_ratio_sum / denom:.8f}"
        )
    logger.log(format_metric_table(result))
    logger.log(
        f"F_MAX={float(result['F_MAX']):.6f} | "
        f"E_ADP={float(result['E_ADP']):.6f} | "
        f"E_MAX={float(result['E_MAX']):.6f} | "
        f"ACC@{threshold_label}={float(result['ACC']):.6f} | "
        f"mIoU@{threshold_label}={float(result['mIOU']):.6f}"
    )
    if binary_reference_result is not None:
        logger.log(
            f"[EvalBinaryReference] Dataset: {dataset_name} | "
            f"threshold={threshold_label} | main_metric_input=binary_prediction"
        )
        logger.log(format_metric_table(binary_reference_result))
        logger.log(
            f"F_MAX={float(binary_reference_result['F_MAX']):.6f} | "
            f"E_ADP={float(binary_reference_result['E_ADP']):.6f} | "
            f"E_MAX={float(binary_reference_result['E_MAX']):.6f} | "
            f"ACC@{threshold_label}={float(binary_reference_result['ACC']):.6f} | "
            f"mIoU@{threshold_label}={float(binary_reference_result['mIOU']):.6f}"
        )
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate clean cached-DINO EMA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval_tag", type=str, default=None)
    parser.add_argument("--eval_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="One-run global output threshold override.",
    )
    parser.add_argument(
        "--adaptive_threshold",
        choices=(
            "none",
            "probability_valley",
            "otsu_float",
            "triangle_logit",
            "logit_gmm",
            "otsu_hysteresis",
        ),
        default="none",
        help="One-run per-image adaptive output threshold mode.",
    )
    parser.add_argument(
        "--metric_input",
        choices=("binary", "probability"),
        default="binary",
        help="Input used by S/Fw/Fm/E/MAE; ACC/mIoU always use a binary mask.",
    )
    parser.add_argument(
        "--lcic_consensus_scale",
        type=float,
        default=1.0,
        help="One-run multiplier for the learned LCIC consensus contribution.",
    )
    parser.add_argument(
        "--lcic_innovation_scale",
        type=float,
        default=1.0,
        help="One-run multiplier for the learned LCIC innovation contribution.",
    )
    parser.add_argument(
        "--dataset_threshold",
        action="append",
        default=[],
        metavar="DATASET=THRESHOLD",
        help="One-run output threshold override; may be repeated.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Explicit evaluation output directory.",
    )
    args = parser.parse_args()
    if args.eval_tag is not None and args.eval_name is not None:
        raise ValueError("--eval_tag and deprecated --eval_name cannot be used together.")
    if int(args.max_samples) == 0 or int(args.max_samples) < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")

    cfg = load_config(args.config)
    default_threshold = float(
        cfg.THRESHOLD if args.threshold is None else args.threshold
    )
    if not math.isfinite(default_threshold) or not 0.0 <= default_threshold <= 1.0:
        raise ValueError(
            f"Invalid --threshold value: {default_threshold}."
        )
    threshold_overrides = parse_dataset_threshold_overrides(
        args.dataset_threshold, cfg.TEST_DATASETS
    )
    if args.adaptive_threshold != "none" and threshold_overrides:
        raise ValueError(
            "--adaptive_threshold cannot be combined with --dataset_threshold."
        )
    if use_cacd(cfg):
        if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "cacd_v1_base" or not bool(
            getattr(cfg, "USE_CACD", False)
        ):
            raise RuntimeError("CACD eval requires HEAD_TYPE='cacd_v1_base' and USE_CACD=True.")
        if any(
            bool(getattr(cfg, name, False))
            for name in ("USE_NDR_BRANCH", "USE_CSD_V1R", "USE_DAGP_SAFE_HEAD", "USE_HR_BFR", "USE_CSSD")
        ):
            raise RuntimeError("CACD eval forbids DAGP/NDR/CSD/HR-BFR/CSSD branches.")
    if use_cssd(cfg):
        if not bool(getattr(cfg, "CSSD_TRAIN_ONLY", True)):
            raise RuntimeError("CSSD-v1a eval requires CSSD_TRAIN_ONLY=True.")
        if not bool(getattr(cfg, "CSSD_STRICT_SINGLE_VIEW_EVAL", True)):
            raise RuntimeError("CSSD-v1a eval requires CSSD_STRICT_SINGLE_VIEW_EVAL=True.")
        if not use_csd_v1r_head(cfg) or bool(getattr(cfg, "USE_HR_BFR", False)):
            raise RuntimeError(
                "CSSD-v1a eval supports only the normal dagp_safe_csd_v1r output without HR-BFR."
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_eval_tag = f"eval_{Path(args.ckpt).stem}"
    eval_tag = args.eval_tag or args.eval_name or default_eval_tag
    eval_tag_path = Path(eval_tag)
    if eval_tag_path.is_absolute() or len(eval_tag_path.parts) != 1 or eval_tag in {"", ".", ".."}:
        raise ValueError(f"--eval_tag must be a single directory name, got {eval_tag!r}.")
    out_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(cfg.WORK_ROOT) / cfg.EXP_NAME / eval_tag
    )
    ensure_dir(out_dir)
    write_yaml(out_dir / "config.yaml", config_to_dict(cfg))

    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    student_state = checkpoint["student"]
    student = build_seg_head(infer_in_channels(student_state), cfg).to(device)
    missing_keys = sorted(set(student.state_dict()) - set(student_state))
    if str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe" and missing_keys == ["current_epoch_tensor"]:
        load_result = student.load_state_dict(student_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")
        student.set_epoch(int(getattr(cfg, "MAX_EPOCH", 25)))
    else:
        student.load_state_dict(student_state)
    lcic_scale_report = apply_lcic_inference_scales(
        cfg,
        student,
        consensus_scale=args.lcic_consensus_scale,
        innovation_scale=args.lcic_innovation_scale,
    )
    online_dino = None
    if use_online_dino_last4(cfg):
        online_dino = FrozenDINOv1Last4Extractor(cfg).to(device).eval()

    with Logger(out_dir / "eval.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"ckpt = {args.ckpt}")
        logger.log(f"eval_tag = {eval_tag}")
        logger.log("model_for_eval = student")
        logger.log("look_twice = false")
        logger.log("teacher_or_apm_at_inference = false")
        logger.log(
            "evaluation_protocol = "
            + (
                "continuous_probability_cod_metrics"
                if args.metric_input == "probability"
                else "original_binary"
            )
        )
        logger.log("model_output_resize = bilinear_to_gt_size")
        logger.log("prediction_activation = sigmoid")
        logger.log("threshold_operator = >")
        logger.log(f"binary_threshold_config_default = {float(cfg.THRESHOLD):.6f}")
        logger.log(f"binary_threshold_run_default = {default_threshold:.6f}")
        logger.log(f"adaptive_threshold_mode = {args.adaptive_threshold}")
        if lcic_scale_report is not None:
            logger.log(
                "[Eval LCICScale] "
                f"consensus_scale="
                f"{lcic_scale_report['consensus_scale']:.6f} | "
                f"innovation_scale="
                f"{lcic_scale_report['innovation_scale']:.6f} | "
                f"intrinsic_consensus_gain="
                f"{lcic_scale_report['intrinsic_consensus_gain']:.6f} | "
                f"intrinsic_innovation_gain="
                f"{lcic_scale_report['intrinsic_innovation_gain']:.6f} | "
                f"alpha_original="
                f"{lcic_scale_report['original_alpha']:.9f} | "
                f"alpha_effective="
                f"{lcic_scale_report['effective_alpha']:.9f} | "
                f"beta_original="
                f"{lcic_scale_report['original_beta']:.9f} | "
                f"beta_effective="
                f"{lcic_scale_report['effective_beta']:.9f} | "
                "application=pre_sigmoid_internal_logit_branch | "
                "checkpoint_file_modified=False"
            )
        if args.adaptive_threshold == "probability_valley":
            logger.log(
                "probability_valley_protocol = "
                f"bins:{PROBABILITY_VALLEY_BINS},"
                f"smooth_sigma:{PROBABILITY_VALLEY_SMOOTH_SIGMA:g},"
                f"search:[{PROBABILITY_VALLEY_SEARCH_RANGE[0]:g},"
                f"{PROBABILITY_VALLEY_SEARCH_RANGE[1]:g}],"
                f"shrink:{PROBABILITY_VALLEY_SHRINK:g},"
                f"clip:[{PROBABILITY_VALLEY_CLIP_RANGE[0]:g},"
                f"{PROBABILITY_VALLEY_CLIP_RANGE[1]:g}],"
                f"fallback:{default_threshold:g}"
            )
        elif args.adaptive_threshold == "otsu_float":
            logger.log(
                "float_otsu_protocol = "
                "source:sigmoid_float32,"
                f"bins:{FLOAT_OTSU_BINS},"
                "histogram_range:per_image_min_max,"
                "uint8_quantization:false,"
                "gt_used:false,"
                f"fallback:{default_threshold:g}"
            )
        elif args.adaptive_threshold == "triangle_logit":
            logger.log(
                "logit_triangle_protocol = "
                "source:sigmoid_float32_then_logit,"
                f"bins:{LOGIT_TRIANGLE_BINS},"
                f"probability_eps:{LOGIT_TRIANGLE_PROBABILITY_EPS:g},"
                "histogram_range:per_image_logit_min_max,"
                "tail:auto_longer_side,"
                "operator:strict_greater_than,"
                "uint8_quantization:false,"
                "gt_used:false,"
                f"fallback:{default_threshold:g}"
            )
        elif args.adaptive_threshold == "logit_gmm":
            logger.log(
                "logit_gmm_protocol = "
                f"bins:{LOGIT_GMM_BINS},"
                f"probability_eps:{LOGIT_GMM_PROBABILITY_EPS:g},"
                f"max_iterations:{LOGIT_GMM_MAX_ITERATIONS},"
                f"tolerance:{LOGIT_GMM_TOLERANCE:g},"
                f"reg_covar:{LOGIT_GMM_REG_COVAR:g},"
                f"min_component_weight:{LOGIT_GMM_MIN_COMPONENT_WEIGHT:g},"
                f"min_standardized_separation:{LOGIT_GMM_MIN_STANDARDIZED_SEPARATION:g},"
                "decision:equal_weighted_posterior_density,"
                "uint8_quantization:false,"
                "gt_used:false,"
                f"fallback:{default_threshold:g}"
            )
        elif args.adaptive_threshold == "otsu_hysteresis":
            logger.log(
                "otsu_hysteresis_protocol = "
                "otsu_source:sigmoid_float32,"
                f"anchor:{default_threshold:g},"
                "low:min(otsu,anchor),"
                "high:max(otsu,anchor),"
                "connectivity:8,"
                "operator:strict_greater_than,"
                "uint8_quantization:false,"
                "gt_used:false,"
                f"fallback:{default_threshold:g}"
            )
        eval_thresholds = {
            str(dataset_name): resolve_eval_binary_threshold(
                cfg,
                dataset_name,
                threshold_overrides,
                default_threshold=default_threshold,
            )
            for dataset_name in cfg.TEST_DATASETS
        }
        logger.log(
            "dataset_threshold_overrides = "
            + (
                ", ".join(
                    f"{name}:{threshold:.6f}"
                    for name, threshold in threshold_overrides.items()
                )
                if threshold_overrides
                else "none"
            )
        )
        logger.log(
            "binary_thresholds_by_dataset = "
            + ", ".join(
                f"{name}:{threshold:.6f}"
                for name, threshold in eval_thresholds.items()
            )
        )
        logger.log(
            "pred_save_format = "
            + (
                "probability_uint8_0_255"
                if args.metric_input == "probability"
                else "binary_0_255"
            )
        )
        logger.log(f"main_metric_input = {args.metric_input}")
        if args.metric_input == "probability":
            logger.log("metric_probability_source = sigmoid_float32")
            logger.log("metric_probability_preprocess = per_image_minmax_in_CODMetrics")
            logger.log("binary_reference_metrics = enabled")
            logger.log("acc_miou_input = thresholded_binary_prediction")
        logger.log(f"prediction_dir = {out_dir / 'pred'}")
        logger.log(f"max_samples = {int(args.max_samples)}")
        if online_dino is not None:
            logger.log("feature_source = online frozen DINOv1-S/8 f9-f12")
            logger.log("feature_cache_used = false")
            logger.log(f"online_dino_key_paths = {online_dino.key_paths}")
        if bool(getattr(cfg, "USE_ECST", False)):
            logger.log("[Eval ECST] training_only=True")
            logger.log("[Eval ECST] temporal_memory_used=False")
            logger.log("[Eval ECST] teacher_weight_map_used=False")
            logger.log("[Eval ECST] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_SOURCE_ARBITER", False)):
            arbiter_mode = str(
                getattr(cfg, "SOURCE_ARBITER_MODE", "residual_over_ecst")
            ).lower()
            logger.log(
                f"[Eval SourceArbiter] mode={arbiter_mode} | training_only=True"
            )
            logger.log("[Eval SourceArbiter] router_loaded=False")
            logger.log("[Eval SourceArbiter] utility_evaluator_loaded=False")
            logger.log("[Eval SourceArbiter] route_memory_used=False")
            logger.log("[Eval SourceArbiter] temporal_memory_used=False")
            logger.log("[Eval SourceArbiter] ecst_map_used=False")
            logger.log("[Eval SourceArbiter] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_TEPR_LITE", False)):
            logger.log("[Eval TEPR-Lite] temporal_memory_used=False")
            logger.log("[Eval TEPR-Lite] teacher_weight_map_used=False")
            logger.log("[Eval TEPR-Lite] logits_source=student_final_logits")
        supervision_mode = str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).strip().lower()
        if bool(getattr(cfg, "USE_AP_STCR", False)) or (
            supervision_mode == "ap_stcr"
        ):
            logger.log("[Eval AP-STCR] training_only=True")
            logger.log("[Eval AP-STCR] semantic_cache_used=False")
            logger.log("[Eval AP-STCR] temporal_history_used=False")
            logger.log("[Eval AP-STCR] mixed_target_used=False")
            logger.log("[Eval AP-STCR] teacher_loaded=False")
            logger.log("[Eval AP-STCR] teacher_forward=False")
            logger.log("[Eval AP-STCR] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_PSSF", False)) or supervision_mode in {
            "pssf_state",
            "ppse_v2_state",
        }:
            if supervision_mode == "ppse_v2_state":
                logger.log("[Eval PPSE-v2] training_only=True")
                logger.log("[Eval PPSE-v2] model_for_eval=student")
                logger.log(
                    "[Eval PPSE-v2] use_ppse_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_pssf_actor_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_pssf_learner_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_teacher_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] "
                    "use_supervision_state_at_inference=False"
                )
                logger.log("[Eval PPSE-v2] q_state_used=False")
                logger.log("[Eval PPSE-v2] history_used=False")
                logger.log(
                    "[Eval PPSE-v2] logits_source=student_final_logits"
                )
            else:
                logger.log("[Eval PSSF] training_only=True")
                logger.log("[Eval PSSF] use_pssf_at_inference=False")
                logger.log("[Eval PSSF] use_teacher_at_inference=False")
                logger.log(
                    "[Eval PSSF] use_supervision_state_at_inference=False"
                )
                logger.log("[Eval PSSF] pssf_network_loaded=False")
                logger.log("[Eval PSSF] teacher_loaded=False")
                logger.log("[Eval PSSF] q_state_used=False")
                logger.log("[Eval PSSF] history_used=False")
                logger.log(
                    "[Eval PSSF] logits_source=student_final_logits"
                )
        if bool(getattr(cfg, "USE_ESA_BER", False)):
            logger.log("[Eval ESA-BER] USE_ESA_BER=True")
            logger.log("[Eval ESA-BER] training_only=True")
            logger.log("[Eval ESA-BER] candidate_selection_used=False")
            logger.log("[Eval ESA-BER] extra_inference_branch=False")
        if use_cssd(cfg):
            logger.log(f"[Eval CSSD] USE_CSSD={bool(getattr(cfg, 'USE_CSSD', False))}")
            logger.log(f"[Eval CSSD] CSSD_TRAIN_ONLY={bool(getattr(cfg, 'CSSD_TRAIN_ONLY', True))}")
            logger.log("[Eval CSSD] high_resolution_view_used=False")
            logger.log("[Eval CSSD] high_cache_read=False")
            logger.log("[Eval CSSD] feature_source=normal_cache")
            logger.log("[Eval CSSD] logits_source=normal_final_logits")
            logger.log("[Eval CSSD] test_time_scale_fusion=False")
        if use_cacd(cfg):
            logger.log("[Eval CACD] USE_CACD=True")
            logger.log(f"[Eval CACD] CACD_VERSION={getattr(cfg, 'CACD_VERSION')}")
            logger.log("[Eval CACD] dabe_pu_read=False")
            logger.log("[Eval CACD] teacher_forward=False")
            logger.log("[Eval CACD] extra_inference_branch=False")
            logger.log("[Eval CACD] post_processing=False")
        if use_online_dino_last4(cfg):
            logger.log(
                "[Online DINO] test cache preflight skipped | "
                "source=original JPEG"
            )
        elif use_multi_level_feature(cfg):
            _, ml_reason = check_ml_feature_cache(cfg, "test")
            logger.log(f"[Cache] feature_ml:test ready | {ml_reason}")
        else:
            ensure_cache_available(cfg, "feature", split="test", logger=logger.log)
        if use_cacd(cfg):
            _, cacd_reason = check_cacd_feature_cache(cfg, "test")
            logger.log(f"[Cache] CACD F10/F11:test ready | {cacd_reason}")
        for dataset_name in cfg.TEST_DATASETS:
            eval_dataset(
                cfg,
                student,
                dataset_name,
                device,
                out_dir,
                logger,
                max_samples=args.max_samples,
                online_dino=online_dino,
                threshold_overrides=threshold_overrides,
                default_threshold=default_threshold,
                adaptive_threshold=args.adaptive_threshold,
                metric_input=args.metric_input,
            )


if __name__ == "__main__":
    main()
