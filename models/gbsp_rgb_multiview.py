"""True-RGB multi-view helpers for the fixed-R32 GBSP diagnostic.

The geometric view is applied to the RGB image before the frozen DINO
forward.  Each resulting GBSP score is inverse-warped to the identity frame
before fusion.  No feature-only augmentation is implemented here.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.cache_features import IMAGENET_MEAN, IMAGENET_STD
from common.dabe_pseudo import _build_local_graph, _sobel_magnitude
from models.gbsp_candidate_path_sweep import (
    build_candidate_path_variants,
    candidate_variant_tag,
)
from models.gbsp_resolution import PreparedResolutionGraph


VIEW_ORDER = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")
GENERATED_VIEW_ORDER = VIEW_ORDER[1:]
METHOD_ORDER = (
    "identity",
    "id_hflip_soft_mean",
    "six_view_soft_mean",
    "six_view_hard_majority",
)


def _transpose_namespace():
    return getattr(Image, "Transpose", Image)


def apply_rgb_view(image: Image.Image, view: str) -> Image.Image:
    """Apply one registered D4 view to an RGB image."""
    name = str(view).lower()
    transpose = _transpose_namespace()
    operations = {
        "hflip": transpose.FLIP_LEFT_RIGHT,
        "vflip": transpose.FLIP_TOP_BOTTOM,
        "rot90": transpose.ROTATE_90,
        "rot180": transpose.ROTATE_180,
        "rot270": transpose.ROTATE_270,
    }
    if name == "identity":
        return image.copy()
    if name not in operations:
        raise ValueError(f"unsupported RGB view: {view!r}")
    return image.transpose(operations[name])


def inverse_align_tensor(value: torch.Tensor, view: str) -> torch.Tensor:
    """Undo a registered RGB view on the last two tensor dimensions."""
    if not torch.is_tensor(value) or value.ndim < 2:
        raise ValueError("value must be a tensor with at least two dimensions")
    name = str(view).lower()
    if name == "identity":
        return value
    if name == "hflip":
        return torch.flip(value, dims=(-1,))
    if name == "vflip":
        return torch.flip(value, dims=(-2,))
    if name == "rot90":
        return torch.rot90(value, k=-1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(value, k=2, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(value, k=1, dims=(-2, -1))
    raise ValueError(f"unsupported RGB view: {view!r}")


def pil_to_dino_tensor(
    image: Image.Image,
    size: int,
    *,
    interpolation: str = "bicubic",
) -> torch.Tensor:
    """Match the repository's frozen-DINO square resize and normalization."""
    modes = {
        "bilinear": getattr(Image, "Resampling", Image).BILINEAR,
        "bicubic": getattr(Image, "Resampling", Image).BICUBIC,
    }
    mode = str(interpolation).lower()
    if mode not in modes:
        raise ValueError(f"unsupported interpolation: {interpolation!r}")
    resized = image.convert("RGB").resize((int(size), int(size)), modes[mode])
    array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).contiguous()


def pil_to_rgb_grid(image: Image.Image, grid: int) -> torch.Tensor:
    """Build the RGB grid used by the GBSP graph from the same transformed RGB."""
    resampling = getattr(Image, "Resampling", Image)
    resized = image.convert("RGB").resize(
        (int(grid), int(grid)), resampling.BICUBIC
    )
    array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def key_batch_to_feature_tensor(key: torch.Tensor) -> torch.Tensor:
    """Convert a batched final-layer DINO key tensor to ``[B,C,H,W]``."""
    if not torch.is_tensor(key) or key.ndim != 3:
        raise ValueError(f"DINO key must be [B,N,C], got {type(key)!r}")
    batch, tokens, channels = map(int, key.shape)
    grid = int(math.sqrt(tokens - 1))
    if grid * grid != tokens - 1:
        raise RuntimeError(f"non-square DINO patch-token count: {tokens - 1}")
    return (
        key[:, 1:, :]
        .reshape(batch, grid, grid, channels)
        .permute(0, 3, 1, 2)
        .detach()
        .cpu()
        .float()
        .contiguous()
    )


def prepare_graph_from_rgb(
    feature: torch.Tensor,
    rgb: torch.Tensor,
    graph_params: dict,
) -> PreparedResolutionGraph:
    """Prepare the unchanged GBSP graph without writing a temporary RGB file."""
    if not torch.is_tensor(feature) or feature.ndim != 3:
        raise ValueError("feature must be Tensor[C,H,W]")
    value = feature.detach().cpu().float().contiguous()
    channels, grid_h, grid_w = map(int, value.shape)
    if grid_h != grid_w or channels < 1 or grid_h < 2:
        raise ValueError(f"invalid feature shape: {list(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("feature contains NaN/Inf")
    rgb_value = rgb.detach().cpu().float().contiguous()
    if tuple(rgb_value.shape) != (3, grid_h, grid_w):
        raise ValueError(
            f"RGB grid must be [3,{grid_h},{grid_w}], got {list(rgb_value.shape)}"
        )
    if not bool(torch.isfinite(rgb_value).all()):
        raise ValueError("RGB grid contains NaN/Inf")

    rgb_started = time.perf_counter()
    sobel = _sobel_magnitude(rgb_value).float().contiguous()
    rgb_seconds = time.perf_counter() - rgb_started
    normalized = F.normalize(
        value.permute(1, 2, 0).reshape(grid_h * grid_w, channels), p=2, dim=1
    ).contiguous()
    rgb_flat = rgb_value.permute(1, 2, 0).reshape(grid_h * grid_w, 3)
    sobel_flat = sobel.reshape(-1)
    graph_started = time.perf_counter()
    neighbor_indices, neighbor_affinity = _build_local_graph(
        normalized, rgb_flat, sobel_flat, grid_h, graph_params
    )
    valid = neighbor_affinity > 0
    source = (
        torch.arange(grid_h * grid_w)
        .view(-1, 1)
        .expand_as(neighbor_indices)[valid]
    )
    target = neighbor_indices[valid]
    terms = {
        "semantic_distance": (
            1.0 - (normalized[source] * normalized[target]).sum(1)
        ).float(),
        "rgb_distance": (
            rgb_flat[source] - rgb_flat[target]
        ).square().sum(1).float(),
        "sobel_term": torch.maximum(
            sobel_flat[source], sobel_flat[target]
        ).float(),
        "graph_affinity": neighbor_affinity[valid].float(),
    }
    graph_seconds = time.perf_counter() - graph_started
    for name, tensor in terms.items():
        if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"invalid graph term: {name}")
    return PreparedResolutionGraph(
        feature=value,
        normalized_feature=normalized,
        rgb=rgb_value,
        sobel=sobel,
        neighbor_indices=neighbor_indices,
        neighbor_affinity=neighbor_affinity,
        graph_terms={name: tensor.contiguous() for name, tensor in terms.items()},
        grid_h=grid_h,
        grid_w=grid_w,
        feature_dim=channels,
        rgb_seconds=rgb_seconds,
        graph_seconds=graph_seconds,
    )


def fixed_r32_score_from_rgb(
    feature: torch.Tensor,
    rgb: torch.Tensor,
    graph_params: dict,
    *,
    border_width: int = 2,
    top_percent: float = 30.0,
) -> tuple[torch.Tensor, dict]:
    """Run the unchanged graph path and fixed-R32 reconstruction for one view."""
    prepared = prepare_graph_from_rgb(feature, rgb, graph_params)
    tag = candidate_variant_tag(int(border_width), float(top_percent))
    result = build_candidate_path_variants(
        prepared,
        graph_params,
        border_widths=(int(border_width),),
        top_percents=(float(top_percent),),
        pca_rank=32,
    )[tag]
    metadata = {
        "candidate_count": int(result.background_indices.numel()),
        "boundary_seed_count": int(result.boundary_seed_count),
        "interior_candidate_count": int(result.interior_candidate_count),
        "selected_rank": int(result.selected_rank),
    }
    return result.minmax_residual.float().contiguous(), metadata


def validate_aligned_scores(aligned_scores: dict[str, torch.Tensor]) -> None:
    if tuple(aligned_scores) != VIEW_ORDER:
        raise ValueError(
            f"aligned score order must be {VIEW_ORDER}, got {tuple(aligned_scores)}"
        )
    for view, score in aligned_scores.items():
        if not torch.is_tensor(score) or tuple(score.shape) != (1, 37, 37):
            raise ValueError(f"{view} score must be Tensor[1,37,37]")
        if (
            not bool(torch.isfinite(score).all())
            or float(score.min()) < -1e-6
            or float(score.max()) > 1.0 + 1e-6
        ):
            raise ValueError(f"{view} score must be finite in [0,1]")


def stack_aligned_scores(aligned_scores: dict[str, torch.Tensor]) -> torch.Tensor:
    validate_aligned_scores(aligned_scores)
    return torch.stack(
        [aligned_scores[view].detach().cpu().float() for view in VIEW_ORDER],
        dim=0,
    ).contiguous()


def unpack_aligned_scores(stacked: torch.Tensor) -> dict[str, torch.Tensor]:
    if not torch.is_tensor(stacked) or tuple(stacked.shape) != (6, 1, 37, 37):
        raise ValueError("stacked aligned scores must be Tensor[6,1,37,37]")
    value = stacked.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("stacked aligned scores contain NaN/Inf")
    return {view: value[index] for index, view in enumerate(VIEW_ORDER)}


def build_method_probabilities_68(
    aligned_scores: dict[str, torch.Tensor],
    *,
    threshold: float = 0.50,
) -> dict[str, torch.Tensor]:
    """Build all registered fusion methods from one shared six-view score set."""
    validate_aligned_scores(aligned_scores)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    stack = stack_aligned_scores(aligned_scores)
    score68 = F.interpolate(
        stack,
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )
    identity = score68[0]
    pair_mean = F.interpolate(
        stack[:2].mean(0, keepdim=True),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    six_mean = F.interpolate(
        stack.mean(0, keepdim=True),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    # Six views can tie 3:3.  Strict majority deliberately maps a tie to BG.
    majority = ((score68 > float(threshold)).sum(0) > 3).float()
    return {
        "identity": identity.contiguous(),
        "id_hflip_soft_mean": pair_mean.contiguous(),
        "six_view_soft_mean": six_mean.contiguous(),
        "six_view_hard_majority": majority.contiguous(),
    }
