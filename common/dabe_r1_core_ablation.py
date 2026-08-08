"""Shared formulas for the four frozen R1 core-mechanism ablations.

This module deliberately reuses the production DABE-v2 graph, dictionary,
reconstruction, Min-Max and RGB-grid implementations.  It contains no GT,
upsampling, thresholding, propagation, morphology, or learned component.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common.dabe_background_null import (
    BackgroundReconstructionResult,
    reconstruct_from_background_atoms,
)
from common.dabe_pseudo import (
    _background_anchor,
    _background_connectivity,
    _border_mask,
    _build_local_graph,
    _minmax,
    _sobel_magnitude,
    _validate_feature,
)


ABLATION_VERSION = "dabe_r1_core_ablation_v1"
GRID = 37
BOUNDARY_WIDTH = 2
MATCHED_CAPACITY = 280
MODES = (
    "soft_similarity",
    "feature_only_reconstruction",
    "boundary280_full_r1",
    "bc280_full_r1",
)


@dataclass(frozen=True)
class DictionaryBundle:
    full_bc: torch.Tensor
    boundary280: torch.Tensor
    bc280: torch.Tensor
    connectivity: torch.Tensor
    full_bc_size: int
    boundary280_size: int
    bc280_size: int
    boundary_bc280_equal: bool


@dataclass(frozen=True)
class ScoreBundle:
    raw: dict[str, torch.Tensor]
    normalized: dict[str, torch.Tensor]
    dictionaries: DictionaryBundle
    full_reconstruction: BackgroundReconstructionResult


def _assert_params(params: dict) -> None:
    if int(params["GRID"]) != GRID:
        raise ValueError(f"R1 core ablations require GRID={GRID}, got {params['GRID']}")
    if int(params["BORDER_WIDTH"]) != BOUNDARY_WIDTH:
        raise ValueError(
            f"Boundary-280 requires BORDER_WIDTH={BOUNDARY_WIDTH}, "
            f"got {params['BORDER_WIDTH']}"
        )
    if int(params["K_RECON"]) > MATCHED_CAPACITY:
        raise ValueError(
            f"K_RECON={params['K_RECON']} exceeds matched dictionary size "
            f"{MATCHED_CAPACITY}"
        )


def build_full_bc_dictionary(
    connectivity: torch.Tensor,
    border: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Use the unchanged production Full-BC anchor selector."""
    mask = _background_anchor(connectivity, border, params).detach().cpu().bool().reshape(-1)
    if int(mask.numel()) != GRID * GRID or int(mask.sum()) == 0:
        raise RuntimeError("invalid Full-BC dictionary")
    return mask.contiguous()


def build_boundary280_dictionary() -> torch.Tensor:
    """The complete two-patch boundary ring of a 37x37 grid."""
    mask = _border_mask(GRID, BOUNDARY_WIDTH).detach().cpu().bool().reshape(-1)
    count = int(mask.sum())
    if count != MATCHED_CAPACITY:
        raise RuntimeError(f"Boundary-280 must contain 280 atoms, got {count}")
    return mask.contiguous()


def build_bc280_dictionary(
    full_bc: torch.Tensor,
    connectivity: torch.Tensor,
) -> torch.Tensor:
    """Top-280 Full-BC atoms by BC reliability, then flat patch index.

    ``torch.argsort(..., stable=True)`` preserves the flattened-index order for
    equal reliability because the source candidates are already in that order.
    """
    full_bc = full_bc.detach().cpu().bool().reshape(-1)
    connectivity = connectivity.detach().cpu().float().reshape(-1)
    candidate_index = torch.where(full_bc)[0]
    if int(candidate_index.numel()) < MATCHED_CAPACITY:
        raise RuntimeError(
            f"Full-BC dictionary has only {candidate_index.numel()} atoms; "
            f"cannot form BC-{MATCHED_CAPACITY}"
        )
    reliability = connectivity.index_select(0, candidate_index)
    order = torch.argsort(reliability, descending=True, stable=True)
    selected = candidate_index.index_select(0, order[:MATCHED_CAPACITY])
    mask = torch.zeros(GRID * GRID, dtype=torch.bool)
    mask[selected] = True
    if int(mask.sum()) != MATCHED_CAPACITY:
        raise RuntimeError(f"BC-280 must contain 280 unique atoms, got {int(mask.sum())}")
    return mask.contiguous()


def build_dictionary_bundle(
    feature: torch.Tensor,
    rgb_grid: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, DictionaryBundle]:
    """Build normalized patch tensors and all three frozen dictionaries once."""
    _assert_params(params)
    feature = _validate_feature(feature, GRID)
    if tuple(feature.shape[-2:]) != (GRID, GRID):
        raise RuntimeError(f"feature grid must be {(GRID, GRID)}")
    if not torch.is_tensor(rgb_grid) or tuple(rgb_grid.shape) != (3, GRID, GRID):
        raise ValueError(f"rgb_grid must be Tensor[3,{GRID},{GRID}]")
    if not bool(torch.isfinite(rgb_grid).all()):
        raise ValueError("rgb_grid contains NaN/Inf")

    feat_n = F.normalize(feature.reshape(feature.shape[0], -1).t(), dim=1, p=2)
    rgb_n = rgb_grid.detach().cpu().float().reshape(3, -1).t().contiguous()
    edge_n = _sobel_magnitude(rgb_grid.detach().cpu().float()).reshape(-1)
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, GRID, params)
    connectivity, border = _background_connectivity(neigh_idx, neigh_weight, GRID, params)
    full_bc = build_full_bc_dictionary(connectivity, border, params)
    boundary280 = build_boundary280_dictionary()
    bc280 = build_bc280_dictionary(full_bc, connectivity)
    bundle = DictionaryBundle(
        full_bc=full_bc,
        boundary280=boundary280,
        bc280=bc280,
        connectivity=connectivity.detach().cpu().float().contiguous(),
        full_bc_size=int(full_bc.sum()),
        boundary280_size=int(boundary280.sum()),
        bc280_size=int(bc280.sum()),
        boundary_bc280_equal=bool(torch.equal(boundary280, bc280)),
    )
    return feat_n.contiguous(), rgb_n, bundle


def compute_soft_similarity_score(
    feat_n: torch.Tensor,
    full_result: BackgroundReconstructionResult,
) -> torch.Tensor:
    """Direct weighted query-to-background cosine score.

    The selected atoms and ``a_ij`` are exactly those of production Full R1;
    only the scoring operation changes.  Thus RGB is not added to this score,
    although the frozen production retrieval weights retain their original
    color-aware definition.  This is the only interpretation that holds both
    the neighbors and weights fixed, as required by the controlled ablation.
    """
    indices = full_result.topk_anchor_global_index
    weights = full_result.topk_weight
    atoms = feat_n.index_select(0, indices.reshape(-1)).reshape(
        indices.shape[0], indices.shape[1], feat_n.shape[1]
    )
    similarity = (atoms * feat_n[:, None, :]).sum(dim=2)
    return (1.0 - (weights * similarity).sum(dim=1)).float().contiguous()


def compute_feature_reconstruction_score(
    feat_n: torch.Tensor,
    full_result: BackgroundReconstructionResult,
) -> torch.Tensor:
    """Production Full-R1 feature residual with the RGB residual removed."""
    return (
        1.0 - (feat_n * full_result.reconstructed_feature).sum(dim=1)
    ).clamp_min(0.0).float().contiguous()


def compute_full_r1_score(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    dictionary: torch.Tensor,
    params: dict,
) -> BackgroundReconstructionResult:
    """Call the unchanged production Full-R1 reconstruction formula."""
    if int(dictionary.sum()) < int(params["K_RECON"]):
        raise RuntimeError("top-k exceeds dictionary size")
    return reconstruct_from_background_atoms(
        feat_n,
        rgb_n,
        dictionary,
        params,
        exclude_chebyshev_radius=None,
    )


def _score_map(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu().float().reshape(1, GRID, GRID).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("score contains NaN/Inf")
    return value


def build_r1_core_ablation_scores(
    feature: torch.Tensor,
    rgb_grid: torch.Tensor,
    params: dict,
) -> ScoreBundle:
    """Compute all four modes and the untouched Full-R1 reference in one pass."""
    feat_n, rgb_n, dictionaries = build_dictionary_bundle(feature, rgb_grid, params)
    full = compute_full_r1_score(feat_n, rgb_n, dictionaries.full_bc, params)
    boundary = compute_full_r1_score(feat_n, rgb_n, dictionaries.boundary280, params)
    # Avoid a redundant reconstruction when the exact dictionaries coincide.
    bc280 = (
        boundary
        if dictionaries.boundary_bc280_equal
        else compute_full_r1_score(feat_n, rgb_n, dictionaries.bc280, params)
    )
    raw_flat = {
        "full_bc_full_r1": full.raw_residual,
        "soft_similarity": compute_soft_similarity_score(feat_n, full),
        "feature_only_reconstruction": compute_feature_reconstruction_score(feat_n, full),
        "boundary280_full_r1": boundary.raw_residual,
        "bc280_full_r1": bc280.raw_residual,
    }
    raw = {name: _score_map(value) for name, value in raw_flat.items()}
    normalized = {
        name: _score_map(_minmax(value.reshape(-1))) for name, value in raw.items()
    }
    for name in (*MODES, "full_bc_full_r1"):
        if not bool(torch.isfinite(raw[name]).all()):
            raise RuntimeError(f"non-finite raw score for {name}")
        if not bool(torch.isfinite(normalized[name]).all()):
            raise RuntimeError(f"non-finite normalized score for {name}")
    return ScoreBundle(
        raw=raw,
        normalized=normalized,
        dictionaries=dictionaries,
        full_reconstruction=full,
    )

