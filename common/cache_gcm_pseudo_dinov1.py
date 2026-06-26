import argparse
import copy
import csv
import hashlib
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_pseudo import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    call_dino,
    load_dino,
    resolve_key_projection,
)
from common.metrics import CODMetrics
from common.utils import (
    build_image_items,
    ensure_dir,
    find_gt_path,
    load_config,
    torch_load,
)


EPS = 1e-8
COD_METRIC_NAMES = ("S_m", "F_beta_w", "F_beta_m", "E_phi_m", "M")
EXTRA_METRIC_NAMES = ("Precision", "Recall", "IoU", "area_ratio")
METRIC_NAMES = COD_METRIC_NAMES + EXTRA_METRIC_NAMES
METHODS = ("fixed", "cgs", "sar", "gcm")
CGS_HEAD_STRATEGIES = (
    "top1",
    "topk_weighted",
    "topk_union",
    "mean_all",
    "max_all",
)

STATS_FIELDS = [
    "dataset",
    "image_name",
    "selected_head",
    "stability_score",
    "hard_sample",
    "valid",
    "area_fixed",
    "area_cgs",
    "area_sar",
    "area_gcm",
]
for _method in METHODS:
    STATS_FIELDS.extend(f"{metric}_{_method}" for metric in METRIC_NAMES)

DEBUG_FIELDS = [
    "dataset",
    "image_name",
    "selected_head",
    "patch_grid",
    "input_size",
    "patch_size",
    "attention_shape",
    "key_shape",
    "total_tokens",
    "key_tokens",
    "attention_prefix",
    "key_prefix",
    "patch_start",
    "patch_end",
    "patch_count",
    "num_heads",
    "key_channels",
    "patch0_value",
    "patch0_rank",
    "patch0_is_max",
    "selected_head_fg_patches",
    "patch0_key_norm_rank",
    "patch0_key_norm_is_top1",
    "mean_head_patch0_rank",
    "mean_head_patch0_is_top1",
    "attn_vs_key_norm_corr",
    "mean_attn_vs_key_norm_corr",
    "empty_cgs",
    "empty_sar",
    "empty_gcm",
    "stability_score",
    "hard_sample",
    "valid",
    "missing_fixed",
    "fixed_shape",
    "fixed_shape_mismatch",
    "missing_gt",
    "cgs_head_strategy",
    "cgs_topk",
    "selected_head_ids",
    "selected_head_scores",
    "cgs_area",
    "cgs_area_ratio",
    "selected_head_patch0_is_max",
    "selected_head_patch0_rank",
    "thr_method",
    "thr_alpha",
    "thr_quantile",
    "thr_cgs",
    "thr_sar",
    "sbre",
    "scon",
    "cgs_score",
    "sar_radius",
    "sar_iters",
    "sar_area",
    "sar_area_ratio",
    "mcf_K",
    "mcf_tau",
    "mcf_vote_thr",
    "mcf_scales",
    "mcf_use_flip",
    "gcm_area",
    "gcm_area_ratio",
    "iou_cgs_sar",
    "iou_sar_gcm",
    "error_msg",
]


def validate_config(cfg):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError(
            "This GCM implementation only supports the existing DINOv1-S/8 "
            "config / BACKBONE_KEY=dinov1-s8."
        )
    input_size = int(cfg.DINO["pseudo_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    if input_size <= 0 or patch_size <= 0 or input_size % patch_size != 0:
        raise RuntimeError(
            f"Invalid pseudo input / patch size: {input_size} / {patch_size}"
        )


def normalize_01(array):
    array = np.asarray(array, dtype=np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if not np.isfinite(amin) or not np.isfinite(amax) or amax - amin < EPS:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - amin) / (amax - amin)).astype(np.float32)


def stable_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / max(float(exponent.sum()), EPS)


def adaptive_threshold(array, method, alpha, quantile):
    array = np.asarray(array, dtype=np.float32)
    if method == "otsu":
        if float(array.max() - array.min()) < EPS:
            return float(array.mean())
        array_u8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(
            array_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return float(threshold) / 255.0
    if method == "mean_std":
        return float(array.mean() + float(alpha) * array.std())
    if method == "quantile":
        return float(np.quantile(array, float(quantile)))
    raise ValueError(f"Unknown threshold method: {method}")


def descending_rank(array, index=0):
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return 1 + int(np.count_nonzero(flat > flat[index]))


def pearson_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size != second.size:
        raise ValueError(f"Correlation size mismatch: {first.size} != {second.size}")
    if float(first.std()) < 1e-12 or float(second.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def image_to_tensor(image):
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def build_view_image(base_image, input_size, scale, flip):
    image = base_image
    if abs(float(scale) - 1.0) > EPS:
        if scale <= 0:
            raise ValueError(f"MCF scale must be positive, got {scale}")
        scaled_size = max(input_size, int(round(input_size * float(scale))))
        scaled = image.resize(
            (scaled_size, scaled_size), Image.Resampling.BICUBIC
        )
        left = (scaled_size - input_size) // 2
        top = (scaled_size - input_size) // 2
        image = scaled.crop((left, top, left + input_size, top + input_size))
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def extract_attention_and_feature(
    model,
    key_holder,
    view_image,
    device,
    input_size,
    patch_size,
):
    inputs = image_to_tensor(view_image).to(device)
    key_holder["tensor"] = None
    with torch.no_grad():
        outputs = call_dino(model, inputs)
    if outputs.attentions is None:
        raise RuntimeError("DINO did not return attentions.")
    attention = outputs.attentions[-1]
    key = key_holder["tensor"]
    if key is None:
        raise RuntimeError("DINO key hook did not capture a tensor.")
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise RuntimeError(f"Unexpected attention shape: {list(attention.shape)}")
    if key.ndim != 3 or key.shape[0] != 1:
        raise RuntimeError(f"Unexpected key shape: {list(key.shape)}")

    expected_patch_count = (input_size // patch_size) ** 2
    attention_tokens = int(attention.shape[-1])
    key_tokens = int(key.shape[1])
    attention_prefix = attention_tokens - expected_patch_count
    key_prefix = key_tokens - expected_patch_count
    if attention_prefix != key_prefix:
        raise RuntimeError(
            f"Attention/key prefix mismatch: {attention_prefix} != {key_prefix}"
        )
    if attention_prefix < 1:
        raise RuntimeError(f"Expected at least one prefix token, got {attention_prefix}")

    patch_start = attention_prefix
    patch_end = patch_start + expected_patch_count
    grid = int(math.sqrt(expected_patch_count))
    if grid * grid != expected_patch_count:
        raise RuntimeError(
            f"Expected patch count is not square: {expected_patch_count}"
        )
    if patch_end != attention_tokens or patch_end != key_tokens:
        raise RuntimeError(
            f"Patch slice does not end at token count: end={patch_end}, "
            f"attention={attention_tokens}, key={key_tokens}"
        )

    num_heads = int(attention.shape[1])
    key_channels = int(key.shape[-1])
    attn_heads = (
        attention[0, :, 0, patch_start:patch_end]
        .reshape(num_heads, grid, grid)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    patch_feat = (
        key[0, patch_start:patch_end, :]
        .reshape(grid, grid, key_channels)
        .permute(2, 0, 1)
        .detach()
    )
    return attn_heads, patch_feat, {
        "attention_shape": list(attention.shape),
        "key_shape": list(key.shape),
        "total_tokens": attention_tokens,
        "key_tokens": key_tokens,
        "attention_prefix": attention_prefix,
        "key_prefix": key_prefix,
        "patch_start": patch_start,
        "patch_end": patch_end,
        "patch_count": expected_patch_count,
        "grid": grid,
        "num_heads": num_heads,
        "key_channels": key_channels,
    }


def run_cgs(
    attn_heads,
    grid,
    patch_count,
    method,
    alpha,
    quantile,
    head_strategy,
    topk,
):
    if attn_heads.shape[1:] != (grid, grid):
        raise RuntimeError(
            f"CGS attention shape mismatch: {attn_heads.shape} vs grid={grid}"
        )
    if grid * grid != patch_count:
        raise RuntimeError(f"CGS grid/count mismatch: {grid} / {patch_count}")

    normalized = np.stack([normalize_01(head) for head in attn_heads], axis=0)
    thresholds = []
    masks = []
    breadths = []
    concentrations = []
    foreground_counts = []
    valid_heads = []

    for head in normalized:
        threshold = adaptive_threshold(head, method, alpha, quantile)
        mask = head > threshold
        foreground_count = int(mask.sum())
        breadth = foreground_count / float(patch_count)
        concentration = 0.0
        valid = foreground_count >= 2
        if valid:
            coordinates = np.argwhere(mask).astype(np.float64)
            covariance = np.cov(coordinates, rowvar=False, bias=True)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            concentration = 1.0 / (float(np.sqrt(eigenvalues).sum()) + EPS)
        thresholds.append(float(threshold))
        masks.append(mask.astype(np.uint8))
        breadths.append(float(breadth))
        concentrations.append(float(concentration))
        foreground_counts.append(foreground_count)
        valid_heads.append(valid)

    scores = stable_softmax(breadths) * stable_softmax(concentrations)
    scores = np.where(np.asarray(valid_heads, dtype=bool), scores, 0.0)
    valid_indices = np.flatnonzero(np.asarray(valid_heads, dtype=bool))
    if head_strategy in {"top1", "topk_weighted", "topk_union"} and not len(
        valid_indices
    ):
        return {
            "attn_map": np.zeros((grid, grid), dtype=np.float32),
            "mask": np.zeros((grid, grid), dtype=np.uint8),
            "selected_head": -1,
            "selected_head_ids": [],
            "selected_head_scores": [],
            "threshold": float("nan"),
            "sbre": breadths,
            "scon": concentrations,
            "scores": scores.tolist(),
            "foreground_counts": foreground_counts,
            "patch0_value": float("nan"),
            "patch0_rank": -1,
            "patch0_is_max": False,
            "empty": True,
        }

    if head_strategy == "top1":
        selected_ids = [int(np.argmax(scores))]
        selected_map = normalized[selected_ids[0]]
        selected_mask = masks[selected_ids[0]]
        selected_threshold = thresholds[selected_ids[0]]
    elif head_strategy in {"topk_weighted", "topk_union"}:
        if topk < 1:
            raise ValueError(f"--cgs_topk must be >= 1, got {topk}")
        ranked = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        selected_ids = [int(value) for value in ranked[: min(topk, len(ranked))]]
        if head_strategy == "topk_weighted":
            selected_scores = scores[selected_ids].astype(np.float64)
            score_sum = float(selected_scores.sum())
            weights = (
                selected_scores / score_sum
                if score_sum > EPS
                else np.full(len(selected_ids), 1.0 / len(selected_ids))
            )
            selected_map = np.sum(
                normalized[selected_ids] * weights[:, None, None], axis=0
            ).astype(np.float32)
            selected_threshold = adaptive_threshold(
                selected_map, method, alpha, quantile
            )
            selected_mask = (selected_map > selected_threshold).astype(np.uint8)
        else:
            selected_mask = np.any(
                np.asarray(masks, dtype=bool)[selected_ids], axis=0
            ).astype(np.uint8)
            # The task defines the union itself as the CGS seed consumed by SAR.
            selected_map = selected_mask.astype(np.float32)
            selected_threshold = float("nan")
    elif head_strategy == "mean_all":
        selected_ids = list(range(normalized.shape[0]))
        selected_map = normalized.mean(axis=0).astype(np.float32)
        selected_threshold = adaptive_threshold(
            selected_map, method, alpha, quantile
        )
        selected_mask = (selected_map > selected_threshold).astype(np.uint8)
    elif head_strategy == "max_all":
        selected_ids = list(range(normalized.shape[0]))
        selected_map = normalized.max(axis=0).astype(np.float32)
        selected_threshold = adaptive_threshold(
            selected_map, method, alpha, quantile
        )
        selected_mask = (selected_map > selected_threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown CGS head strategy: {head_strategy}")

    selected_head = selected_ids[0] if len(selected_ids) == 1 else -1
    patch0_rank = descending_rank(selected_map)
    return {
        "attn_map": selected_map,
        "mask": selected_mask,
        "selected_head": selected_head,
        "selected_head_ids": selected_ids,
        "selected_head_scores": [float(scores[index]) for index in selected_ids],
        "threshold": float(selected_threshold),
        "sbre": breadths,
        "scon": concentrations,
        "scores": scores.tolist(),
        "foreground_counts": foreground_counts,
        "patch0_value": float(selected_map[0, 0]),
        "patch0_rank": patch0_rank,
        "patch0_is_max": patch0_rank == 1,
        "empty": int(selected_mask.sum()) == 0,
    }


def run_sar(
    attn_map,
    patch_feat,
    grid,
    patch_count,
    radius,
    iterations,
    method,
    alpha,
    quantile,
):
    if radius < 0:
        raise ValueError(f"--sar_radius must be >= 0, got {radius}")
    if iterations < 1:
        raise ValueError(f"--sar_iters must be >= 1, got {iterations}")
    if grid * grid != patch_count:
        raise RuntimeError(f"SAR grid/count mismatch: {grid} / {patch_count}")
    if attn_map.shape != (grid, grid) or patch_feat.shape[1:] != (grid, grid):
        raise RuntimeError(
            f"SAR input mismatch: attention={attn_map.shape}, "
            f"feature={list(patch_feat.shape)}, grid={grid}"
        )

    feature = F.normalize(patch_feat.float(), dim=0, p=2)
    source = torch.as_tensor(attn_map, dtype=torch.float32, device=feature.device)
    refined = source
    for _ in range(iterations):
        numerator = torch.zeros_like(source)
        denominator = torch.zeros_like(source)
        for dy in range(-radius, radius + 1):
            target_y0 = max(0, -dy)
            target_y1 = min(grid, grid - dy)
            neighbor_y0 = target_y0 + dy
            neighbor_y1 = target_y1 + dy
            for dx in range(-radius, radius + 1):
                target_x0 = max(0, -dx)
                target_x1 = min(grid, grid - dx)
                neighbor_x0 = target_x0 + dx
                neighbor_x1 = target_x1 + dx
                center = feature[
                    :, target_y0:target_y1, target_x0:target_x1
                ]
                neighbor = feature[
                    :, neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1
                ]
                affinity = (center * neighbor).sum(dim=0).clamp_min(0.0)
                numerator[
                    target_y0:target_y1, target_x0:target_x1
                ] += affinity * refined[
                    neighbor_y0:neighbor_y1, neighbor_x0:neighbor_x1
                ]
                denominator[
                    target_y0:target_y1, target_x0:target_x1
                ] += affinity
        refined = torch.where(
            denominator > EPS, numerator / denominator.clamp_min(EPS), refined
        )

    refined_np = refined.detach().cpu().numpy().astype(np.float32)
    threshold = adaptive_threshold(refined_np, method, alpha, quantile)
    mask = (refined_np > threshold).astype(np.uint8)
    return {
        "attn_map": refined_np,
        "mask": mask,
        "threshold": float(threshold),
        "empty": int(mask.sum()) == 0,
    }


def extract_single_view(
    model,
    key_holder,
    base_image,
    device,
    input_size,
    patch_size,
    scale,
    flip,
):
    view_image = build_view_image(base_image, input_size, scale, flip)
    attn_heads, patch_feat, token_info = extract_attention_and_feature(
        model,
        key_holder,
        view_image,
        device,
        input_size,
        patch_size,
    )
    return {
        "attn_heads": attn_heads,
        "patch_feat": patch_feat,
        "token_info": token_info,
        "scale": float(scale),
        "flip": bool(flip),
    }


def process_extracted_view(extracted, args):
    token_info = extracted["token_info"]
    grid = token_info["grid"]
    patch_count = token_info["patch_count"]
    cgs = run_cgs(
        extracted["attn_heads"],
        grid,
        patch_count,
        args.thr_method,
        args.thr_alpha,
        args.thr_quantile,
        args.cgs_head_strategy,
        args.cgs_topk,
    )
    sar = run_sar(
        cgs["attn_map"],
        extracted["patch_feat"],
        grid,
        patch_count,
        args.sar_radius,
        args.sar_iters,
        args.thr_method,
        args.thr_alpha,
        args.thr_quantile,
    )
    return {
        "cgs": cgs,
        "sar": sar,
        "attn_heads": extracted["attn_heads"],
        "patch_feat": extracted["patch_feat"],
        "token_info": token_info,
        "scale": extracted["scale"],
        "flip": extracted["flip"],
    }


def run_single_view(
    model,
    key_holder,
    base_image,
    device,
    input_size,
    patch_size,
    args,
    scale,
    flip,
):
    extracted = extract_single_view(
        model,
        key_holder,
        base_image,
        device,
        input_size,
        patch_size,
        scale,
        flip,
    )
    return process_extracted_view(extracted, args)


def inverse_view_mask(mask, grid, scale, flip):
    if mask.shape != (grid, grid):
        raise RuntimeError(
            f"Inverse mask shape mismatch: {mask.shape} vs {(grid, grid)}"
        )
    mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).view(
        1, 1, grid, grid
    )
    coordinates = (torch.arange(grid, dtype=torch.float32) + 0.5) * (
        2.0 / grid
    ) - 1.0
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    source_x = xx * float(scale)
    source_y = yy * float(scale)
    if flip:
        source_x = -source_x
    valid = (
        (source_x >= -1.0)
        & (source_x <= 1.0)
        & (source_y >= -1.0)
        & (source_y <= 1.0)
    )
    sample_grid = torch.stack([source_x, source_y], dim=-1).unsqueeze(0)
    restored = F.grid_sample(
        mask_tensor,
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    valid_np = valid.numpy().astype(np.uint8)
    restored_np = (restored.numpy() > 0.5).astype(np.uint8) * valid_np
    return restored_np, valid_np


def masked_iou(first, second, valid_region):
    valid = np.asarray(valid_region, dtype=bool)
    if not np.any(valid):
        return None
    first = np.asarray(first, dtype=bool)[valid]
    second = np.asarray(second, dtype=bool)[valid]
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def binary_iou(first, second, grid):
    return masked_iou(first, second, np.ones((grid, grid), dtype=np.uint8))


def combine_mcf_views(processed_views, grid, args):
    original_result = processed_views[0]
    if not args.enable_mcf:
        mask = original_result["sar"]["mask"].copy()
        return {
            "mask": mask,
            "stability_score": 1.0,
            "hard_sample": False,
            "valid": True,
            "mcf_K": 1,
            "empty": int(mask.sum()) == 0,
        }

    if not 0.0 <= float(args.mcf_vote_thr) <= 1.0:
        raise ValueError("--mcf_vote_thr must be in [0, 1].")
    views = []
    for result in processed_views:
        view_info = result["token_info"]
        if view_info["grid"] != grid:
            raise RuntimeError(
                f"MCF view grid changed: {view_info['grid']} != {grid}"
            )
        restored, valid_region = inverse_view_mask(
            result["sar"]["mask"],
            grid,
            result["scale"],
            result["flip"],
        )
        views.append(
            {
                "mask": restored,
                "valid_region": valid_region,
            }
        )

    foreground_votes = np.zeros((grid, grid), dtype=np.float32)
    valid_votes = np.zeros((grid, grid), dtype=np.float32)
    for view in views:
        foreground_votes += (
            view["mask"].astype(np.float32)
            * view["valid_region"].astype(np.float32)
        )
        valid_votes += view["valid_region"].astype(np.float32)
    foreground_ratio = np.divide(
        foreground_votes,
        valid_votes,
        out=np.zeros_like(foreground_votes),
        where=valid_votes > 0,
    )
    consensus = (foreground_ratio > float(args.mcf_vote_thr)).astype(np.uint8)
    consensus[valid_votes == 0] = 0

    view_ious = [
        value
        for value in (
            masked_iou(view["mask"], consensus, view["valid_region"])
            for view in views
        )
        if value is not None
    ]
    stability_score = float(np.mean(view_ious)) if view_ious else 0.0
    hard_sample = not view_ious or stability_score < float(args.mcf_tau)
    return {
        "mask": consensus,
        "stability_score": stability_score,
        "hard_sample": bool(hard_sample),
        "valid": not bool(hard_sample),
        "mcf_K": len(views),
        "empty": int(consensus.sum()) == 0,
    }


def run_mcf(
    model,
    key_holder,
    base_image,
    device,
    input_size,
    patch_size,
    grid,
    patch_count,
    args,
    original_result,
):
    if grid * grid != patch_count:
        raise RuntimeError(f"MCF grid/count mismatch: {grid} / {patch_count}")
    view_specs = [("original", 1.0, False)]
    if args.enable_mcf and args.mcf_use_flip:
        view_specs.append(("horizontal_flip", 1.0, True))
    for scale in (args.mcf_scales if args.enable_mcf else []):
        if scale <= 0:
            raise ValueError(f"MCF scale must be positive, got {scale}")
        view_specs.append((f"scale_{scale:g}", float(scale), False))
        if args.mcf_use_flip:
            view_specs.append((f"scale_{scale:g}_horizontal_flip", float(scale), True))

    processed_views = []
    for index, (name, scale, flip) in enumerate(view_specs):
        result = (
            original_result
            if index == 0
            else run_single_view(
                model,
                key_holder,
                base_image,
                device,
                input_size,
                patch_size,
                args,
                scale,
                flip,
            )
        )
        view_info = result["token_info"]
        if view_info["grid"] != grid or view_info["patch_count"] != patch_count:
            raise RuntimeError(
                f"MCF view grid changed for {name}: "
                f"{view_info['grid']} / {view_info['patch_count']}"
            )
        processed_views.append(result)
    return combine_mcf_views(processed_views, grid, args)


def read_fixed_pseudo(cfg, dataset, stem, grid):
    path = (
        Path(cfg.CACHE_ROOT)
        / "pseudo_label_cache"
        / cfg.BACKBONE_KEY
        / dataset
        / f"{stem}.pt"
    )
    if not path.exists():
        return None, {
            "missing_fixed": True,
            "fixed_shape": [],
            "fixed_shape_mismatch": False,
        }
    try:
        payload = torch_load(path, map_location="cpu")
        tensor = payload.get("tensor") if isinstance(payload, dict) else None
        actual_shape = list(tensor.shape) if isinstance(tensor, torch.Tensor) else []
        expected_shape = [1, grid, grid]
        if tensor is None or actual_shape != expected_shape:
            return None, {
                "missing_fixed": True,
                "fixed_shape": actual_shape,
                "fixed_shape_mismatch": True,
            }
        mask = (tensor.squeeze(0).float().numpy() > 0.5).astype(np.uint8)
        return mask, {
            "missing_fixed": False,
            "fixed_shape": actual_shape,
            "fixed_shape_mismatch": False,
        }
    except Exception:
        return None, {
            "missing_fixed": True,
            "fixed_shape": [],
            "fixed_shape_mismatch": True,
        }


def read_gt(cfg, dataset, stem):
    try:
        gt_path = find_gt_path(cfg.DATA_ROOT, dataset, stem)
    except FileNotFoundError:
        return None
    image = Image.open(gt_path).convert("L")
    return (np.asarray(image, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


def resize_grid_mask(mask, grid, size_hw):
    if mask.shape != (grid, grid):
        raise RuntimeError(
            f"Metric/visual mask shape mismatch: {mask.shape} vs {(grid, grid)}"
        )
    height, width = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def evaluate_mask(mask, grid, gt):
    prediction = resize_grid_mask(mask, grid, gt.shape)
    metrics = CODMetrics()
    gt_tensor = torch.from_numpy(gt.astype(np.float32)).view(1, 1, *gt.shape)
    pred_tensor = torch.from_numpy(prediction.astype(np.float32)).view(
        1, 1, *prediction.shape
    )
    metrics.step(gt_tensor, pred_tensor)
    result = metrics.get_result()
    pred_bool = prediction.astype(bool)
    gt_bool = gt.astype(bool)
    true_positive = float(np.logical_and(pred_bool, gt_bool).sum())
    false_positive = float(np.logical_and(pred_bool, ~gt_bool).sum())
    false_negative = float(np.logical_and(~pred_bool, gt_bool).sum())
    return {
        "S_m": float(result["SMeasure"]),
        "F_beta_w": float(result["WFM"]),
        "F_beta_m": float(result["F_MEAN"]),
        "E_phi_m": float(result["E_MEAN"]),
        "M": float(result["MAE"]),
        "Precision": true_positive / (true_positive + false_positive + EPS),
        "Recall": true_positive / (true_positive + false_negative + EPS),
        "IoU": true_positive
        / (true_positive + false_positive + false_negative + EPS),
        "area_ratio": float(pred_bool.mean()),
    }


def blank_stats_row(dataset, stem):
    row = {field: float("nan") for field in STATS_FIELDS}
    row.update(
        {
            "dataset": dataset,
            "image_name": stem,
            "selected_head": -1,
            "hard_sample": True,
            "valid": False,
        }
    )
    return row


def build_stats_row(dataset, stem, original_result, mcf, fixed, gt, grid):
    row = blank_stats_row(dataset, stem)
    masks = {
        "fixed": fixed,
        "cgs": original_result["cgs"]["mask"],
        "sar": original_result["sar"]["mask"],
        "gcm": mcf["mask"],
    }
    row.update(
        {
            "selected_head": original_result["cgs"]["selected_head"],
            "stability_score": mcf["stability_score"],
            "hard_sample": mcf["hard_sample"],
            "valid": mcf["valid"],
        }
    )
    for method, mask in masks.items():
        row[f"area_{method}"] = (
            float(mask.mean()) if mask is not None else float("nan")
        )
        metrics = (
            evaluate_mask(mask, grid, gt)
            if mask is not None and gt is not None
            else None
        )
        for metric in METRIC_NAMES:
            row[f"{metric}_{method}"] = (
                metrics[metric] if metrics is not None else float("nan")
            )
    return row


def build_debug_row(
    dataset,
    stem,
    original_result,
    mcf,
    fixed_info,
    gt,
    input_size,
    patch_size,
    args,
):
    token_info = original_result["token_info"]
    grid = token_info["grid"]
    patch_count = token_info["patch_count"]
    cgs = original_result["cgs"]
    sar = original_result["sar"]
    key_norm = (
        torch.linalg.vector_norm(original_result["patch_feat"], dim=0)
        .detach()
        .cpu()
        .numpy()
    )
    patch0_key_norm_rank = descending_rank(key_norm)
    mean_head_attention = original_result["attn_heads"].mean(axis=0)
    mean_head_patch0_rank = descending_rank(mean_head_attention)
    selected_corr = pearson_correlation(cgs["attn_map"], key_norm)
    mean_corr = pearson_correlation(mean_head_attention, key_norm)
    selected_head_fg_patches = (
        cgs["foreground_counts"][cgs["selected_head"]]
        if cgs["selected_head"] >= 0
        else int(cgs["mask"].sum())
    )
    return {
        "dataset": dataset,
        "image_name": stem,
        "selected_head": cgs["selected_head"],
        "patch_grid": [grid, grid],
        "input_size": input_size,
        "patch_size": patch_size,
        "attention_shape": token_info["attention_shape"],
        "key_shape": token_info["key_shape"],
        "total_tokens": token_info["total_tokens"],
        "key_tokens": token_info["key_tokens"],
        "attention_prefix": token_info["attention_prefix"],
        "key_prefix": token_info["key_prefix"],
        "patch_start": token_info["patch_start"],
        "patch_end": token_info["patch_end"],
        "patch_count": patch_count,
        "num_heads": token_info["num_heads"],
        "key_channels": token_info["key_channels"],
        "patch0_value": cgs["patch0_value"],
        "patch0_rank": cgs["patch0_rank"],
        "patch0_is_max": cgs["patch0_is_max"],
        "selected_head_fg_patches": selected_head_fg_patches,
        "patch0_key_norm_rank": patch0_key_norm_rank,
        "patch0_key_norm_is_top1": patch0_key_norm_rank == 1,
        "mean_head_patch0_rank": mean_head_patch0_rank,
        "mean_head_patch0_is_top1": mean_head_patch0_rank == 1,
        "attn_vs_key_norm_corr": selected_corr,
        "mean_attn_vs_key_norm_corr": mean_corr,
        "empty_cgs": cgs["empty"],
        "empty_sar": sar["empty"],
        "empty_gcm": mcf["empty"],
        "stability_score": mcf["stability_score"],
        "hard_sample": mcf["hard_sample"],
        "valid": mcf["valid"],
        "missing_fixed": fixed_info["missing_fixed"],
        "fixed_shape": fixed_info["fixed_shape"],
        "fixed_shape_mismatch": fixed_info["fixed_shape_mismatch"],
        "missing_gt": gt is None,
        "cgs_head_strategy": args.cgs_head_strategy,
        "cgs_topk": args.cgs_topk,
        "selected_head_ids": cgs["selected_head_ids"],
        "selected_head_scores": cgs["selected_head_scores"],
        "cgs_area": int(cgs["mask"].sum()),
        "cgs_area_ratio": float(cgs["mask"].mean()),
        "selected_head_patch0_is_max": cgs["patch0_is_max"],
        "selected_head_patch0_rank": cgs["patch0_rank"],
        "thr_method": args.thr_method,
        "thr_alpha": args.thr_alpha,
        "thr_quantile": args.thr_quantile,
        "thr_cgs": cgs["threshold"],
        "thr_sar": sar["threshold"],
        "sbre": cgs["sbre"],
        "scon": cgs["scon"],
        "cgs_score": cgs["scores"],
        "sar_radius": args.sar_radius,
        "sar_iters": args.sar_iters,
        "sar_area": int(sar["mask"].sum()),
        "sar_area_ratio": float(sar["mask"].mean()),
        "mcf_K": mcf["mcf_K"],
        "mcf_tau": args.mcf_tau,
        "mcf_vote_thr": args.mcf_vote_thr,
        "mcf_scales": args.mcf_scales,
        "mcf_use_flip": args.mcf_use_flip,
        "gcm_area": int(mcf["mask"].sum()),
        "gcm_area_ratio": float(mcf["mask"].mean()),
        "iou_cgs_sar": binary_iou(cgs["mask"], sar["mask"], grid),
        "iou_sar_gcm": binary_iou(sar["mask"], mcf["mask"], grid),
        "error_msg": "",
    }


def format_csv_value(value):
    if isinstance(value, (list, tuple)):
        return ";".join(str(format_csv_value(item)) for item in value)
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (float, np.floating)):
        return "nan" if not np.isfinite(value) else f"{float(value):.10g}"
    return value


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: format_csv_value(row.get(field, ""))
                    for field in fieldnames
                }
            )


def panel(image, title, size=224, resample=Image.Resampling.NEAREST):
    image = image.convert("RGB").resize((size, size), resample)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), title, fill="black")
    return canvas


def mask_panel(mask, title, grid, size=224):
    if mask is None:
        return panel(Image.new("L", (size, size), 0), f"{title} (missing)", size)
    display = resize_grid_mask(mask, grid, (size, size))
    return panel(Image.fromarray(display * 255), title, size)


def save_visualization(
    path,
    image_path,
    gt,
    fixed,
    cgs,
    sar,
    gcm,
    info,
    grid,
    input_size,
):
    image = Image.open(image_path).convert("RGB")
    gt_image = (
        Image.fromarray(gt.astype(np.uint8) * 255)
        if gt is not None
        else Image.new("L", (input_size, input_size), 0)
    )
    info_image = Image.new("RGB", (input_size, input_size), "white")
    draw = ImageDraw.Draw(info_image)
    lines = [
        f"head: {info['selected_head']}",
        f"stability: {info['stability_score']:.4f}",
        f"hard: {info['hard_sample']}",
        f"valid: {info['valid']}",
        f"grid: {grid}x{grid}",
        f"patch0 rank: {info['patch0_rank']}",
        f"patch0 max: {info['patch0_is_max']}",
        f"empty GCM: {info['empty_gcm']}",
    ]
    line_step = max(18, (input_size - 12) // len(lines))
    for index, line in enumerate(lines):
        draw.text((8, 8 + index * line_step), line, fill="black")

    panels = [
        panel(image, "Image", input_size, Image.Resampling.BICUBIC),
        panel(gt_image, "GT", input_size),
        mask_panel(fixed, "Original fixed", grid, input_size),
        mask_panel(cgs, "CGS", grid, input_size),
        mask_panel(sar, "CGS+SAR", grid, input_size),
        mask_panel(gcm, "GCM/MCF", grid, input_size),
        panel(info_image, "Info", input_size),
    ]
    canvas = Image.new(
        "RGB",
        (sum(item.width for item in panels), max(item.height for item in panels)),
        "white",
    )
    x = 0
    for item in panels:
        canvas.paste(item, (x, 0))
        x += item.width
    ensure_dir(Path(path).parent)
    canvas.save(path)


def prepare_dataset_dir(out_root, dataset, overwrite):
    dataset_root = Path(out_root) / dataset
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset directory exists; pass --overwrite: {dataset_root}"
            )
        shutil.rmtree(dataset_root)
    for name in ("gcm_cache", "candidate_cache", "vis"):
        ensure_dir(dataset_root / name)
    return dataset_root


def save_outputs(
    dataset_root,
    cfg,
    args,
    item,
    original_size,
    original_result,
    mcf,
    input_size,
    patch_size,
    grid,
):
    stem = item["stem"]
    cgs_mask = original_result["cgs"]["mask"]
    sar_mask = original_result["sar"]["mask"]
    gcm_mask = mcf["mask"]
    token_info = original_result["token_info"]
    payload = {
        "dataset": item["dataset"],
        "stem": stem,
        "image_path": item["image_path"],
        "original_size": tuple(int(value) for value in original_size),
        "tensor": torch.from_numpy(gcm_mask).unsqueeze(0).float(),
        "source": "icme_gcm_dinov1",
        "backbone_key": cfg.BACKBONE_KEY,
        "input_size": input_size,
        "patch_size": patch_size,
        "patch_grid": [grid, grid],
        "total_tokens": token_info["total_tokens"],
        "key_tokens": token_info["key_tokens"],
        "patch_start": token_info["patch_start"],
        "patch_end": token_info["patch_end"],
        "patch_count": token_info["patch_count"],
        "valid": mcf["valid"],
        "hard_sample": mcf["hard_sample"],
        "stability_score": mcf["stability_score"],
    }
    torch.save(payload, dataset_root / "gcm_cache" / f"{stem}.pt")
    candidate_mask = (
        sar_mask if args.final_candidate == "sar" else gcm_mask
    )
    candidate_payload = dict(payload)
    candidate_payload.update(
        {
            "tensor": torch.from_numpy(candidate_mask).unsqueeze(0).float(),
            "source": f"icme_{args.final_candidate}_dinov1",
            "final_candidate": args.final_candidate,
        }
    )
    torch.save(
        candidate_payload, dataset_root / "candidate_cache" / f"{stem}.pt"
    )


def process_item(
    cfg,
    model,
    key_holder,
    device,
    args,
    item,
    dataset_root,
    input_size,
    patch_size,
):
    source_image = Image.open(item["image_path"]).convert("RGB")
    original_size = (source_image.height, source_image.width)
    base_image = source_image.resize(
        (input_size, input_size), Image.Resampling.BICUBIC
    )
    original_result = run_single_view(
        model,
        key_holder,
        base_image,
        device,
        input_size,
        patch_size,
        args,
        scale=1.0,
        flip=False,
    )
    token_info = original_result["token_info"]
    grid = token_info["grid"]
    patch_count = token_info["patch_count"]
    mcf = run_mcf(
        model,
        key_holder,
        base_image,
        device,
        input_size,
        patch_size,
        grid,
        patch_count,
        args,
        original_result,
    )
    fixed, fixed_info = read_fixed_pseudo(
        cfg, item["dataset"], item["stem"], grid
    )
    gt = read_gt(cfg, item["dataset"], item["stem"])
    save_outputs(
        dataset_root,
        cfg,
        args,
        item,
        original_size,
        original_result,
        mcf,
        input_size,
        patch_size,
        grid,
    )
    stats_row = build_stats_row(
        item["dataset"],
        item["stem"],
        original_result,
        mcf,
        fixed,
        gt,
        grid,
    )

    cgs = original_result["cgs"]
    sar = original_result["sar"]
    debug_row = build_debug_row(
        item["dataset"],
        item["stem"],
        original_result,
        mcf,
        fixed_info,
        gt,
        input_size,
        patch_size,
        args,
    )
    if args.save_vis:
        save_visualization(
            dataset_root / "vis" / f"{item['stem']}.png",
            item["image_path"],
            gt,
            fixed,
            cgs["mask"],
            sar["mask"],
            mcf["mask"],
            debug_row,
            grid,
            input_size,
        )
    return stats_row, debug_row


def finite_mean(rows, field):
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def print_dataset_summary(dataset, stats_rows, debug_rows):
    complete_debug = [
        row for row in debug_rows if not row.get("error_msg")
    ]
    count = len(complete_debug)
    if not count:
        print(f"[Summary] dataset={dataset} | num_samples=0")
        return
    first = complete_debug[0]
    print(f"[Summary] dataset={dataset}")
    print(f"num_samples = {count}")
    print(f"grid size = {first['patch_grid']}")
    for field, label in (
        ("patch0_is_max", "selected head patch0_is_max"),
        ("patch0_key_norm_is_top1", "patch0_key_norm_is_top1"),
        ("empty_cgs", "empty_cgs"),
        ("empty_sar", "empty_sar"),
        ("empty_gcm", "empty_gcm"),
        ("hard_sample", "hard_sample"),
        ("missing_fixed", "missing_fixed"),
    ):
        hit_count = sum(bool(row[field]) for row in complete_debug)
        print(f"{label} = {hit_count}/{count} ({hit_count / count:.6f})")
    for method in METHODS:
        values = {
            metric: finite_mean(stats_rows, f"{metric}_{method}")
            for metric in METRIC_NAMES
        }
        formatted = " | ".join(
            f"{metric}={value:.6f}" if np.isfinite(value) else f"{metric}=nan"
            for metric, value in values.items()
        )
        print(f"mean metrics {method}: {formatted}")


def run_generation(cfg, args):
    validate_config(cfg)
    unknown = sorted(set(args.datasets) - set(cfg.TRAIN_DATASETS))
    if unknown:
        raise ValueError(
            f"--datasets must be selected from {cfg.TRAIN_DATASETS}, got {unknown}"
        )
    existing = [
        str(Path(args.out_root) / dataset)
        for dataset in args.datasets
        if (Path(args.out_root) / dataset).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output dataset directory exists; pass --overwrite: "
            + ", ".join(existing)
        )
    ensure_dir(args.out_root)

    input_size = int(cfg.DINO["pseudo_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    key_module, key_path = resolve_key_projection(model)
    key_holder = {"tensor": None}

    def hook_key(_module, _inputs, output):
        key_holder["tensor"] = output.detach()

    handle = key_module.register_forward_hook(hook_key)
    print(f"device = {device}")
    print(f"backbone_key = {cfg.BACKBONE_KEY}")
    print(f"model_path = {cfg.DINO['model_path']}")
    print(f"key_hook = {key_path}")
    print(f"input_size = {input_size}")
    print(f"patch_size = {patch_size}")
    total_errors = 0
    try:
        for dataset in args.datasets:
            dataset_root = prepare_dataset_dir(
                args.out_root, dataset, args.overwrite
            )
            items = build_image_items(
                cfg.DATA_ROOT, [dataset], require_gt=False
            )
            if args.max_samples >= 0:
                items = items[: args.max_samples]
            stats_rows = []
            debug_rows = []
            for item in tqdm(items, desc=f"DINOv1 GCM {dataset}"):
                try:
                    stats_row, debug_row = process_item(
                        cfg,
                        model,
                        key_holder,
                        device,
                        args,
                        item,
                        dataset_root,
                        input_size,
                        patch_size,
                    )
                except Exception as exc:
                    total_errors += 1
                    stats_row = blank_stats_row(dataset, item["stem"])
                    debug_row = {field: "" for field in DEBUG_FIELDS}
                    debug_row.update(
                        {
                            "dataset": dataset,
                            "image_name": item["stem"],
                            "input_size": input_size,
                            "patch_size": patch_size,
                            "error_msg": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"[Error] {dataset}/{item['stem']} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                stats_rows.append(stats_row)
                debug_rows.append(debug_row)
            write_csv(dataset_root / "stats.csv", stats_rows, STATS_FIELDS)
            write_csv(
                dataset_root / "debug_stats.csv", debug_rows, DEBUG_FIELDS
            )
            print_dataset_summary(dataset, stats_rows, debug_rows)
            print(f"wrote = {dataset_root / 'stats.csv'}")
            print(f"wrote = {dataset_root / 'debug_stats.csv'}")
    finally:
        handle.remove()
    if total_errors:
        raise RuntimeError(
            f"DINOv1 GCM completed with {total_errors} sample error(s); "
            "see debug_stats.csv."
        )


def build_stage_a_combos(args):
    threshold_settings = [
        ("otsu", 0.0, 0.5),
        ("quantile", 0.0, 0.5),
        ("quantile", 0.0, 0.6),
        ("mean_std", -0.5, 0.5),
    ]
    sar_settings = [(1, 1), (2, 1)]
    mcf_settings = [(False, 0.5), (True, 0.33), (True, 0.5)]
    combos = []
    for strategy in CGS_HEAD_STRATEGIES:
        for method, alpha, quantile in threshold_settings:
            for radius, iterations in sar_settings:
                for enable_mcf, vote_threshold in mcf_settings:
                    combo_args = copy.copy(args)
                    combo_args.cgs_head_strategy = strategy
                    combo_args.cgs_topk = 3
                    combo_args.thr_method = method
                    combo_args.thr_alpha = alpha
                    combo_args.thr_quantile = quantile
                    combo_args.sar_radius = radius
                    combo_args.sar_iters = iterations
                    combo_args.enable_mcf = enable_mcf
                    combo_args.mcf_vote_thr = vote_threshold
                    # Stage A uses the canonical four-view MCF setup.
                    combo_args.mcf_scales = [1.25]
                    combo_args.mcf_use_flip = True
                    combos.append(
                        {
                            "combo_id": f"combo_{len(combos):03d}",
                            "args": combo_args,
                        }
                    )
    if len(combos) != 120:
        raise RuntimeError(f"Stage A must contain 120 combos, got {len(combos)}")
    return combos


def combo_postprocess_key(combo_args):
    return (
        combo_args.cgs_head_strategy,
        combo_args.cgs_topk,
        combo_args.thr_method,
        combo_args.thr_alpha,
        combo_args.thr_quantile,
        combo_args.sar_radius,
        combo_args.sar_iters,
    )


def mask_metric_key(mask, grid, gt):
    return (grid, gt.shape, np.asarray(mask, dtype=np.uint8).tobytes())


def build_stats_row_cached(
    dataset,
    stem,
    original_result,
    mcf,
    fixed,
    gt,
    grid,
    metric_cache,
):
    row = blank_stats_row(dataset, stem)
    masks = {
        "fixed": fixed,
        "cgs": original_result["cgs"]["mask"],
        "sar": original_result["sar"]["mask"],
        "gcm": mcf["mask"],
    }
    row.update(
        {
            "selected_head": original_result["cgs"]["selected_head"],
            "stability_score": mcf["stability_score"],
            "hard_sample": mcf["hard_sample"],
            "valid": mcf["valid"],
        }
    )
    for method, mask in masks.items():
        row[f"area_{method}"] = (
            float(mask.mean()) if mask is not None else float("nan")
        )
        metrics = None
        if mask is not None and gt is not None:
            key = mask_metric_key(mask, grid, gt)
            if key not in metric_cache:
                metric_cache[key] = evaluate_mask(mask, grid, gt)
            metrics = metric_cache[key]
        for metric in METRIC_NAMES:
            row[f"{metric}_{method}"] = (
                metrics[metric] if metrics is not None else float("nan")
            )
    return row


def fixed_cache_path(cfg, dataset, stem):
    return (
        Path(cfg.CACHE_ROOT)
        / "pseudo_label_cache"
        / cfg.BACKBONE_KEY
        / dataset
        / f"{stem}.pt"
    )


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = Path(path).stat()
    return (stat.st_mtime_ns, stat.st_size, digest.hexdigest())


def snapshot_fixed_files(cfg, items):
    snapshot = {}
    for item in items:
        path = fixed_cache_path(cfg, item["dataset"], item["stem"])
        snapshot[str(path)] = file_digest(path) if path.exists() else None
    return snapshot


def assert_fixed_snapshot_unchanged(before):
    changed = []
    for path_string, expected in before.items():
        path = Path(path_string)
        actual = file_digest(path) if path.exists() else None
        if actual != expected:
            changed.append(path_string)
    if changed:
        raise RuntimeError(
            "Fixed pseudo cache changed during sweep: " + ", ".join(changed)
        )


def paired_delta_mean(rows, method, metric):
    paired = []
    for row in rows:
        values = []
        for candidate in COD_METRIC_NAMES:
            try:
                fixed_value = float(row[f"{candidate}_fixed"])
                method_value = float(row[f"{candidate}_{method}"])
            except (KeyError, TypeError, ValueError):
                values = []
                break
            if not np.isfinite(fixed_value) or not np.isfinite(method_value):
                values = []
                break
            values.append((fixed_value, method_value))
        if values:
            index = COD_METRIC_NAMES.index(metric)
            paired.append(values[index][1] - values[index][0])
    return float(np.mean(paired)) if paired else float("nan")


SWEEP_SUMMARY_FIELDS = [
    "combo_id",
    "dataset",
    "num_samples",
    "cgs_head_strategy",
    "cgs_topk",
    "thr_method",
    "thr_alpha",
    "thr_quantile",
    "sar_radius",
    "sar_iters",
    "enable_mcf",
    "mcf_vote_thr",
    "hard_sample_ratio",
    "empty_cgs_ratio",
    "empty_sar_ratio",
    "empty_gcm_ratio",
]
for _method in METHODS:
    SWEEP_SUMMARY_FIELDS.extend(
        f"{_method}_{metric}" for metric in METRIC_NAMES
    )
for _method in ("sar", "gcm"):
    SWEEP_SUMMARY_FIELDS.extend(
        f"delta_{_method}_{metric}_vs_fixed" for metric in COD_METRIC_NAMES
    )
SWEEP_SUMMARY_FIELDS.extend(["score_sar", "score_gcm"])


def build_sweep_summary_row(combo, dataset, stats_rows, debug_rows):
    combo_args = combo["args"]
    complete = [row for row in debug_rows if not row.get("error_msg")]
    count = len(complete)
    row = {
        "combo_id": combo["combo_id"],
        "dataset": dataset,
        "num_samples": count,
        "cgs_head_strategy": combo_args.cgs_head_strategy,
        "cgs_topk": combo_args.cgs_topk,
        "thr_method": combo_args.thr_method,
        "thr_alpha": combo_args.thr_alpha,
        "thr_quantile": combo_args.thr_quantile,
        "sar_radius": combo_args.sar_radius,
        "sar_iters": combo_args.sar_iters,
        "enable_mcf": combo_args.enable_mcf,
        "mcf_vote_thr": combo_args.mcf_vote_thr,
    }
    for field in ("hard_sample", "empty_cgs", "empty_sar", "empty_gcm"):
        row[f"{field}_ratio"] = (
            sum(bool(item[field]) for item in complete) / count
            if count
            else float("nan")
        )
    for method in METHODS:
        for metric in METRIC_NAMES:
            row[f"{method}_{metric}"] = finite_mean(
                stats_rows, f"{metric}_{method}"
            )
    for method in ("sar", "gcm"):
        for metric in COD_METRIC_NAMES:
            row[f"delta_{method}_{metric}_vs_fixed"] = paired_delta_mean(
                stats_rows, method, metric
            )
        row[f"score_{method}"] = (
            row[f"{method}_S_m"]
            + row[f"{method}_F_beta_w"]
            + row[f"{method}_F_beta_m"]
            + row[f"{method}_E_phi_m"]
            - row[f"{method}_M"]
        )
    return row


def format_threshold_setting(row):
    method = row["thr_method"]
    if method == "quantile":
        return f"quantile={float(row['thr_quantile']):g}"
    if method == "mean_std":
        return f"mean_std={float(row['thr_alpha']):g}"
    return "otsu"


def print_sweep_ranking(summary_rows, field, descending, method):
    finite_rows = [
        row for row in summary_rows if np.isfinite(float(row.get(field, np.nan)))
    ]
    ranked = sorted(
        finite_rows, key=lambda row: float(row[field]), reverse=descending
    )[:10]
    direction = "desc" if descending else "asc"
    print(f"\n[Top 10] {field} ({direction})")
    for rank, row in enumerate(ranked, 1):
        print(
            f"{rank:02d} {row['combo_id']} | "
            f"head={row['cgs_head_strategy']} topk={row['cgs_topk']} | "
            f"thr={format_threshold_setting(row)} | "
            f"sar=r{row['sar_radius']}_i{row['sar_iters']} | "
            f"mcf={'on' if row['enable_mcf'] else 'off'}"
            f"@{float(row['mcf_vote_thr']):g} | "
            f"S={float(row[f'{method}_S_m']):.6f} "
            f"Fw={float(row[f'{method}_F_beta_w']):.6f} "
            f"Fm={float(row[f'{method}_F_beta_m']):.6f} "
            f"E={float(row[f'{method}_E_phi_m']):.6f} "
            f"M={float(row[f'{method}_M']):.6f} "
            f"P={float(row[f'{method}_Precision']):.6f} "
            f"R={float(row[f'{method}_Recall']):.6f} "
            f"IoU={float(row[f'{method}_IoU']):.6f} "
            f"area={float(row[f'{method}_area_ratio']):.6f} "
            f"hard={float(row['hard_sample_ratio']):.6f}"
        )


def run_sweep(cfg, args):
    validate_config(cfg)
    unknown = sorted(set(args.datasets) - set(cfg.TRAIN_DATASETS))
    if unknown:
        raise ValueError(
            f"--datasets must be selected from {cfg.TRAIN_DATASETS}, got {unknown}"
        )
    out_root = Path(args.out_root)
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Sweep output exists; pass --overwrite: {out_root}"
            )
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    combos = build_stage_a_combos(args)
    combo_stats = {
        (combo["combo_id"], dataset): []
        for combo in combos
        for dataset in args.datasets
    }
    combo_debug = {
        (combo["combo_id"], dataset): []
        for combo in combos
        for dataset in args.datasets
    }
    for combo in combos:
        for dataset in args.datasets:
            ensure_dir(out_root / combo["combo_id"] / dataset / "vis")

    input_size = int(cfg.DINO["pseudo_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    key_module, key_path = resolve_key_projection(model)
    key_holder = {"tensor": None}

    def hook_key(_module, _inputs, output):
        key_holder["tensor"] = output.detach()

    handle = key_module.register_forward_hook(hook_key)
    print(f"device = {device}")
    print(f"backbone_key = {cfg.BACKBONE_KEY}")
    print(f"key_hook = {key_path}")
    print(f"Stage A combos = {len(combos)}")
    print("Stage A MCF views = original, flip, scale_1.25, scale_1.25+flip")
    view_specs = [
        (1.0, False),
        (1.0, True),
        (1.25, False),
        (1.25, True),
    ]
    summary_rows = []
    total_errors = 0
    try:
        for dataset in args.datasets:
            items = build_image_items(
                cfg.DATA_ROOT, [dataset], require_gt=False
            )
            if args.max_samples >= 0:
                items = items[: args.max_samples]
            fixed_before = snapshot_fixed_files(cfg, items)
            for item in tqdm(items, desc=f"DINOv1 Stage A sweep {dataset}"):
                try:
                    source_image = Image.open(item["image_path"]).convert("RGB")
                    base_image = source_image.resize(
                        (input_size, input_size), Image.Resampling.BICUBIC
                    )
                    extracted_views = [
                        extract_single_view(
                            model,
                            key_holder,
                            base_image,
                            device,
                            input_size,
                            patch_size,
                            scale,
                            flip,
                        )
                        for scale, flip in view_specs
                    ]
                    token_info = extracted_views[0]["token_info"]
                    grid = token_info["grid"]
                    patch_count = token_info["patch_count"]
                    for extracted in extracted_views[1:]:
                        other = extracted["token_info"]
                        if (
                            other["grid"] != grid
                            or other["patch_count"] != patch_count
                        ):
                            raise RuntimeError("Sweep view token grid mismatch.")
                    fixed, fixed_info = read_fixed_pseudo(
                        cfg, dataset, item["stem"], grid
                    )
                    gt = read_gt(cfg, dataset, item["stem"])
                    metric_cache = {}
                    processed_cache = {}
                    for combo in combos:
                        combo_args = combo["args"]
                        postprocess_key = combo_postprocess_key(combo_args)
                        if postprocess_key not in processed_cache:
                            processed_cache[postprocess_key] = [
                                process_extracted_view(extracted, combo_args)
                                for extracted in extracted_views
                            ]
                        processed_views = processed_cache[postprocess_key]
                        original_result = processed_views[0]
                        mcf = combine_mcf_views(
                            processed_views, grid, combo_args
                        )
                        stats_row = build_stats_row_cached(
                            dataset,
                            item["stem"],
                            original_result,
                            mcf,
                            fixed,
                            gt,
                            grid,
                            metric_cache,
                        )
                        debug_row = build_debug_row(
                            dataset,
                            item["stem"],
                            original_result,
                            mcf,
                            fixed_info,
                            gt,
                            input_size,
                            patch_size,
                            combo_args,
                        )
                        key = (combo["combo_id"], dataset)
                        combo_stats[key].append(stats_row)
                        combo_debug[key].append(debug_row)
                        if args.save_vis:
                            save_visualization(
                                out_root
                                / combo["combo_id"]
                                / dataset
                                / "vis"
                                / f"{item['stem']}.png",
                                item["image_path"],
                                gt,
                                fixed,
                                original_result["cgs"]["mask"],
                                original_result["sar"]["mask"],
                                mcf["mask"],
                                debug_row,
                                grid,
                                input_size,
                            )
                except Exception as exc:
                    total_errors += 1
                    print(
                        f"[Error] {dataset}/{item['stem']} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                    for combo in combos:
                        key = (combo["combo_id"], dataset)
                        combo_stats[key].append(
                            blank_stats_row(dataset, item["stem"])
                        )
                        debug_row = {field: "" for field in DEBUG_FIELDS}
                        debug_row.update(
                            {
                                "dataset": dataset,
                                "image_name": item["stem"],
                                "input_size": input_size,
                                "patch_size": patch_size,
                                "error_msg": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        combo_debug[key].append(debug_row)
            assert_fixed_snapshot_unchanged(fixed_before)
            print(f"fixed cache invariant verified: {dataset} ({len(items)} files)")

            for combo in combos:
                key = (combo["combo_id"], dataset)
                dataset_root = out_root / combo["combo_id"] / dataset
                write_csv(
                    dataset_root / "stats.csv",
                    combo_stats[key],
                    STATS_FIELDS,
                )
                write_csv(
                    dataset_root / "debug_stats.csv",
                    combo_debug[key],
                    DEBUG_FIELDS,
                )
                summary_rows.append(
                    build_sweep_summary_row(
                        combo,
                        dataset,
                        combo_stats[key],
                        combo_debug[key],
                    )
                )
    finally:
        handle.remove()

    write_csv(
        out_root / "sweep_summary.csv",
        summary_rows,
        SWEEP_SUMMARY_FIELDS,
    )
    print(f"wrote = {out_root / 'sweep_summary.csv'}")
    for dataset in args.datasets:
        rows = [row for row in summary_rows if row["dataset"] == dataset]
        print(f"\n===== Sweep ranking: {dataset} =====")
        print_sweep_ranking(rows, "sar_F_beta_w", True, "sar")
        print_sweep_ranking(rows, "gcm_F_beta_w", True, "gcm")
        print_sweep_ranking(rows, "score_sar", True, "sar")
        print_sweep_ranking(rows, "score_gcm", True, "gcm")
        print_sweep_ranking(rows, "gcm_M", False, "gcm")
    if total_errors:
        raise RuntimeError(
            f"Stage A sweep completed with {total_errors} sample error(s); "
            "see each combo debug_stats.csv."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate dynamic-grid ICME-GCM pseudo labels with DINOv1-S/8."
    )
    parser.add_argument("--config", default="configs/dinov1_s8.py")
    parser.add_argument(
        "--datasets", nargs="+", default=["TR-CAMO", "TR-COD10K"]
    )
    parser.add_argument(
        "--out_root", default="../workdir/gcm_icme_repro_dinov1"
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--thr_method",
        choices=["otsu", "mean_std", "quantile"],
        default="otsu",
    )
    parser.add_argument("--thr_alpha", type=float, default=0.0)
    parser.add_argument("--thr_quantile", type=float, default=0.75)
    parser.add_argument(
        "--cgs_head_strategy",
        choices=CGS_HEAD_STRATEGIES,
        default="top1",
    )
    parser.add_argument("--cgs_topk", type=int, default=3)
    parser.add_argument("--sar_radius", type=int, default=1)
    parser.add_argument("--sar_iters", type=int, default=1)
    parser.add_argument("--enable_mcf", action="store_true")
    parser.add_argument("--mcf_tau", type=float, default=0.55)
    parser.add_argument("--mcf_vote_thr", type=float, default=0.5)
    parser.add_argument("--mcf_scales", nargs="+", type=float, default=[1.25])
    parser.add_argument("--mcf_use_flip", action="store_true")
    parser.add_argument(
        "--final_candidate", choices=["sar", "gcm"], default="gcm"
    )
    parser.add_argument("--run_sweep", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not 0.0 <= args.thr_quantile <= 1.0:
        raise ValueError("--thr_quantile must be in [0, 1].")
    if args.cgs_topk < 1:
        raise ValueError("--cgs_topk must be >= 1.")
    if args.sar_radius < 0:
        raise ValueError("--sar_radius must be >= 0.")
    if args.sar_iters < 1:
        raise ValueError("--sar_iters must be >= 1.")
    if not 0.0 <= args.mcf_vote_thr <= 1.0:
        raise ValueError("--mcf_vote_thr must be in [0, 1].")
    cfg = load_config(args.config)
    if args.run_sweep:
        run_sweep(cfg, args)
    else:
        run_generation(cfg, args)


if __name__ == "__main__":
    main()
