"""Controlled Full-BC candidate-count and path-participation sweep for GBSP.

Every variant keeps the same eight-neighbour graph and multi-source Dijkstra
mechanism.  Only the border seed width and the top-BC candidate percentage are
changed.  The PCA branch is fixed-rank and otherwise identical to the current
GBSP absolute-residual/Min-Max target construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common.dabe_pseudo import _background_anchor, _background_connectivity, _minmax
from models.gbsp_core_variants import (
    PCARankSelector,
    decompose_pca,
    pca_fit_from_decomposition,
    score_all_patches,
)
from models.gbsp_resolution import PreparedResolutionGraph


@dataclass(frozen=True)
class CandidatePathVariant:
    tag: str
    border_width: int
    top_percent: float
    bc: torch.Tensor
    border: torch.Tensor
    background_anchor: torch.Tensor
    background_indices: torch.Tensor
    raw_residual: torch.Tensor
    minmax_residual: torch.Tensor
    selected_rank: int
    singular_values: torch.Tensor
    subspace_mean: torch.Tensor
    retained_variance_ratio: float
    boundary_seed_count: int
    boundary_candidate_count: int
    interior_candidate_count: int
    interior_candidate_ratio: float


def _percent_tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def candidate_variant_tag(border_width: int, top_percent: float) -> str:
    width = int(border_width)
    percent = float(top_percent)
    if width < 1:
        raise ValueError("border_width must be positive")
    if not 0.0 < percent < 100.0:
        raise ValueError("top_percent must be in (0,100)")
    return f"bw{width}-p{_percent_tag(percent)}"


def candidate_cache_version(border_width: int, top_percent: float, rank: int) -> str:
    tag = candidate_variant_tag(border_width, top_percent).replace("-", "_")
    return f"gbsp_r{int(rank)}_pathcand_{tag}_v1"


def build_candidate_path_variants(
    prepared: PreparedResolutionGraph,
    base_params: dict,
    *,
    border_widths: tuple[int, ...] = (1, 2),
    top_percents: tuple[float, ...] = (25.0, 27.5, 30.0),
    pca_rank: int = 32,
) -> dict[str, CandidatePathVariant]:
    """Build the registered BW x top-percent sweep from one prepared graph."""
    if prepared.grid_h != 37 or prepared.grid_w != 37:
        raise ValueError("the formal candidate-path sweep requires a 37x37 graph")
    if int(pca_rank) <= 0:
        raise ValueError("pca_rank must be positive")
    widths = tuple(dict.fromkeys(int(value) for value in border_widths))
    percents = tuple(dict.fromkeys(float(value) for value in top_percents))
    if not widths or not percents:
        raise ValueError("border_widths and top_percents must not be empty")
    for width in widths:
        candidate_variant_tag(width, percents[0])
    for percent in percents:
        candidate_variant_tag(widths[0], percent)

    selector = PCARankSelector(0.90, int(pca_rank), 1)
    output: dict[str, CandidatePathVariant] = {}
    for width in widths:
        width_params = {**base_params, "BORDER_WIDTH": int(width)}
        # Connectivity is recomputed once per seed width.  It is never replaced
        # by a geometric Boundary-only selector.
        bc, border = _background_connectivity(
            prepared.neighbor_indices,
            prepared.neighbor_affinity,
            prepared.grid_h,
            width_params,
        )
        for percent in percents:
            params = {
                **width_params,
                "BG_ANCHOR_TOP_PERCENT": float(percent),
            }
            anchor = _background_anchor(bc, border, params)
            indices = torch.where(anchor)[0].long().contiguous()
            if int(indices.numel()) <= int(pca_rank) + 1:
                raise RuntimeError(
                    f"candidate dictionary is too small for fixed R{pca_rank}: "
                    f"BW={width}, top={percent}, count={indices.numel()}"
                )
            background = prepared.normalized_feature.index_select(0, indices)
            decomposition = decompose_pca(background)
            fit = pca_fit_from_decomposition(
                decomposition,
                "fixed",
                selector,
                int(pca_rank),
            )
            raw = score_all_patches(prepared.normalized_feature, fit).reshape(
                1, prepared.grid_h, prepared.grid_w
            )
            calibrated = _minmax(raw).float().contiguous()
            boundary_candidates = int((anchor & border).sum().item())
            interior_candidates = int(indices.numel()) - boundary_candidates
            tag = candidate_variant_tag(width, percent)
            output[tag] = CandidatePathVariant(
                tag=tag,
                border_width=int(width),
                top_percent=float(percent),
                bc=bc.reshape(1, 37, 37).float().contiguous(),
                border=border.reshape(1, 37, 37).bool().contiguous(),
                background_anchor=anchor.reshape(1, 37, 37).bool().contiguous(),
                background_indices=indices,
                raw_residual=raw.float().contiguous(),
                minmax_residual=calibrated,
                selected_rank=int(fit.selected_rank),
                singular_values=decomposition.singular_values.float().contiguous(),
                subspace_mean=fit.mean.float().contiguous(),
                retained_variance_ratio=float(fit.retained_variance_ratio),
                boundary_seed_count=int(border.sum().item()),
                boundary_candidate_count=boundary_candidates,
                interior_candidate_count=interior_candidates,
                interior_candidate_ratio=(
                    float(interior_candidates) / float(indices.numel())
                ),
            )
    return output
