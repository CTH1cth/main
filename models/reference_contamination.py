"""Mechanism-only interventions on a frozen Full-BC reference dictionary.

The functions in this module never alter DINO features or the graph that
created Full-BC.  GT labels are accepted only to diagnose, clean, or
deliberately contaminate an already frozen candidate set.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from models.local_background_subspace import retrieve_background_neighbors


GRID = 37
NUM_PATCHES = GRID * GRID


@dataclass(frozen=True)
class KNN8ReferenceScore:
    score: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_similarities: torch.Tensor
    self_match_violation_count: int


@dataclass(frozen=True)
class GBSPReferenceScore:
    score: torch.Tensor
    mean: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    selected_rank: int
    orthonormal_error: float


@dataclass(frozen=True)
class ControlledCandidateSet:
    indices: torch.Tensor
    removed_background_indices: torch.Tensor
    injected_foreground_indices: torch.Tensor
    requested_fraction: float
    realized_fraction: float
    valid: bool
    reason: str
    injection_mode: str = "random_scattered"
    selected_component_size: int = 0
    cluster_anchor_index: int = -1
    injected_connected: bool = False


def normalize_patch_features(features: torch.Tensor) -> torch.Tensor:
    """Return the frozen per-patch L2-normalized [1369,D] representation."""
    if not torch.is_tensor(features):
        raise TypeError("features must be a tensor")
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1).T
    if features.ndim != 2 or int(features.shape[0]) != NUM_PATCHES:
        raise ValueError(f"expected [C,37,37] or [1369,D], got {tuple(features.shape)}")
    value = F.normalize(features.float(), dim=1, eps=1e-12).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("features contain NaN/Inf")
    return value


def validate_candidate_indices(indices: torch.Tensor, *, minimum: int = 9) -> torch.Tensor:
    value = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
    if value.numel() < int(minimum):
        raise ValueError(f"candidate set needs at least {minimum} patches")
    if value.numel() != torch.unique(value).numel():
        raise ValueError("candidate indices contain duplicates")
    if int(value.min()) < 0 or int(value.max()) >= NUM_PATCHES:
        raise ValueError("candidate index out of range")
    return value.contiguous()


def score_knn8(features: torch.Tensor, candidate_indices: torch.Tensor) -> KNN8ReferenceScore:
    """Exact frozen KNN8 anomaly with candidate-query leave-one-out."""
    candidate = validate_candidate_indices(candidate_indices, minimum=9).to(features.device)
    retrieval = retrieve_background_neighbors(features, candidate, max_k=8)
    return KNN8ReferenceScore(
        score=(1.0 - retrieval.cosine_similarities.mean(dim=1)).float().contiguous(),
        neighbor_indices=retrieval.neighbor_indices.long().contiguous(),
        neighbor_similarities=retrieval.cosine_similarities.float().contiguous(),
        self_match_violation_count=int(retrieval.self_match_violation_count),
    )


def score_gbsp_fixed_rank(
    features: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    rank: int = 8,
) -> GBSPReferenceScore:
    """Equal-weight global PCA squared residual under a fixed rank."""
    x = normalize_patch_features(features)
    candidate = validate_candidate_indices(candidate_indices, minimum=int(rank) + 2).to(x.device)
    background = x.index_select(0, candidate)
    mean = background.mean(dim=0)
    centered_background = background - mean
    _, singular_values, vh = torch.linalg.svd(centered_background, full_matrices=False)
    effective = min(int(background.shape[0]) - 1, int(background.shape[1]))
    selected_rank = min(int(rank), effective - 1)
    if selected_rank != int(rank):
        raise ValueError(
            f"fixed rank {rank} is unsupported by candidate set of size {background.shape[0]}"
        )
    # CUDA float32 SVD can leave the selected vectors with ~1e-4 numerical
    # orthogonality drift.  Reduced QR preserves their span exactly while
    # making the residual projection a proper orthogonal projection.
    basis, _ = torch.linalg.qr(vh[:selected_rank].T, mode="reduced")
    basis = basis.contiguous()
    identity = torch.eye(selected_rank, dtype=basis.dtype, device=basis.device)
    orthonormal_error = float((basis.T @ basis - identity).abs().max().item())
    if orthonormal_error >= 1e-4:
        raise RuntimeError(f"GBSP basis is not orthonormal: {orthonormal_error}")
    centered = x - mean
    coefficients = centered @ basis
    score = (centered.square().sum(dim=1) - coefficients.square().sum(dim=1)).clamp_min(0.0)
    if score.shape != (NUM_PATCHES,) or not bool(torch.isfinite(score).all()):
        raise RuntimeError("GBSP produced an invalid score")
    return GBSPReferenceScore(
        score=score.float().contiguous(),
        mean=mean.float().contiguous(),
        basis=basis.float().contiguous(),
        singular_values=singular_values.float().contiguous(),
        selected_rank=selected_rank,
        orthonormal_error=orthonormal_error,
    )


def local_wrong_reference_counts(
    neighbor_indices: torch.Tensor,
    foreground_labels: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    labels = torch.as_tensor(foreground_labels, dtype=torch.bool, device=neighbor_indices.device).reshape(-1)
    if labels.numel() != NUM_PATCHES:
        raise ValueError("foreground labels must contain 1369 values")
    neighbors = torch.as_tensor(neighbor_indices, dtype=torch.long, device=labels.device)
    if neighbors.ndim != 2 or neighbors.shape[0] != NUM_PATCHES:
        raise ValueError("neighbor indices must be [1369,K]")
    return labels.index_select(0, neighbors.reshape(-1)).reshape_as(neighbors).sum(dim=1).to(torch.uint8)


def _identity_seed(identity: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}|{identity}|{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _permutation(values: torch.Tensor, *, identity: str, seed: int, namespace: str) -> torch.Tensor:
    array = torch.as_tensor(values, dtype=torch.long).cpu().numpy()
    rng = np.random.default_rng(_identity_seed(identity, seed, namespace))
    return torch.from_numpy(array[rng.permutation(array.size)].copy()).long()


def rounded_contamination_count(fraction: float, candidate_count: int) -> int:
    """Round-half-up implementation of round(p*|B0|), independent of bankers rounding."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("contamination fraction must be in [0,1]")
    return int(math.floor(float(fraction) * int(candidate_count) + 0.5))


def controlled_candidate_set(
    clean_candidate_indices: torch.Tensor,
    foreground_labels: torch.Tensor | np.ndarray,
    *,
    fraction: float,
    seed: int,
    identity: str,
    removal_namespace: str = "remove_bg",
) -> ControlledCandidateSet:
    """Replace true-BG clean references by same-image true-FG patches.

    For a fixed image and seed, all contamination levels are nested prefixes of
    the same two deterministic permutations.
    """
    clean = validate_candidate_indices(clean_candidate_indices, minimum=9).cpu()
    labels = torch.as_tensor(foreground_labels, dtype=torch.bool).reshape(-1).cpu()
    if labels.numel() != NUM_PATCHES:
        raise ValueError("foreground labels must contain 1369 values")
    if bool(labels.index_select(0, clean).any()):
        raise ValueError("clean candidate set still contains foreground patches")
    requested = rounded_contamination_count(float(fraction), int(clean.numel()))
    eligible_foreground = torch.where(labels)[0]
    if requested > int(eligible_foreground.numel()):
        return ControlledCandidateSet(
            indices=clean.clone(),
            removed_background_indices=torch.empty(0, dtype=torch.long),
            injected_foreground_indices=torch.empty(0, dtype=torch.long),
            requested_fraction=float(fraction),
            realized_fraction=0.0,
            valid=False,
            reason=f"needs {requested} FG patches but only {eligible_foreground.numel()} exist",
        )
    removal_order = _permutation(
        clean, identity=identity, seed=seed, namespace=str(removal_namespace)
    )
    injection_order = _permutation(
        eligible_foreground, identity=identity, seed=seed, namespace="inject_fg"
    )
    removed = removal_order[:requested]
    injected = injection_order[:requested]
    if requested:
        retained = clean[~torch.isin(clean, removed)]
        candidate = torch.cat([retained, injected]).sort().values
    else:
        candidate, removed, injected = clean.clone(), removed.clone(), injected.clone()
    validate_candidate_indices(candidate, minimum=9)
    if int(candidate.numel()) != int(clean.numel()):
        raise RuntimeError("controlled contamination changed dictionary size")
    actual_wrong = int(labels.index_select(0, candidate).sum())
    if actual_wrong != requested:
        raise RuntimeError("controlled contamination did not realize the requested wrong-reference count")
    return ControlledCandidateSet(
        indices=candidate,
        removed_background_indices=removed,
        injected_foreground_indices=injected,
        requested_fraction=float(fraction),
        realized_fraction=actual_wrong / float(clean.numel()),
        valid=True,
        reason="",
    )


def _foreground_components(labels: torch.Tensor) -> list[torch.Tensor]:
    """Return deterministic 8-connected foreground components on the 37x37 grid."""
    mask = torch.as_tensor(labels, dtype=torch.bool).reshape(GRID, GRID).cpu()
    visited = torch.zeros_like(mask)
    components: list[torch.Tensor] = []
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for row in range(GRID):
        for column in range(GRID):
            if not bool(mask[row, column]) or bool(visited[row, column]):
                continue
            queue = [(row, column)]
            visited[row, column] = True
            component = []
            for current_row, current_column in queue:
                component.append(current_row * GRID + current_column)
                for delta_row, delta_column in offsets:
                    next_row, next_column = current_row + delta_row, current_column + delta_column
                    if not (0 <= next_row < GRID and 0 <= next_column < GRID):
                        continue
                    if bool(mask[next_row, next_column]) and not bool(visited[next_row, next_column]):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            components.append(torch.tensor(component, dtype=torch.long))
    return sorted(components, key=lambda value: (-int(value.numel()), int(value.min())))


def _cluster_bfs_order(
    component: torch.Tensor,
    *,
    identity: str,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Build a seeded, connected and nested traversal of one foreground component."""
    component = torch.as_tensor(component, dtype=torch.long).reshape(-1).cpu()
    component_set = set(component.tolist())
    priority_order = _permutation(
        component, identity=identity, seed=seed, namespace="cluster_priority"
    ).tolist()
    priority = {patch: rank for rank, patch in enumerate(priority_order)}
    anchor = int(_permutation(
        component, identity=identity, seed=seed, namespace="cluster_anchor"
    )[0])
    queue = [anchor]
    visited = {anchor}
    ordered = []
    for patch in queue:
        ordered.append(patch)
        row, column = divmod(patch, GRID)
        neighbors = []
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == 0 and delta_column == 0:
                    continue
                next_row, next_column = row + delta_row, column + delta_column
                if not (0 <= next_row < GRID and 0 <= next_column < GRID):
                    continue
                neighbor = next_row * GRID + next_column
                if neighbor in component_set and neighbor not in visited:
                    neighbors.append(neighbor)
        for neighbor in sorted(neighbors, key=lambda value: priority[value]):
            visited.add(neighbor)
            queue.append(neighbor)
    if len(ordered) != int(component.numel()):
        raise RuntimeError("foreground component traversal is incomplete")
    return torch.tensor(ordered, dtype=torch.long), anchor


def _is_eight_connected(indices: torch.Tensor) -> bool:
    value = torch.as_tensor(indices, dtype=torch.long).reshape(-1).cpu()
    if value.numel() <= 1:
        return True
    selected = set(value.tolist())
    reached = {next(iter(selected))}
    queue = list(reached)
    for patch in queue:
        row, column = divmod(patch, GRID)
        for delta_row in (-1, 0, 1):
            for delta_column in (-1, 0, 1):
                if delta_row == 0 and delta_column == 0:
                    continue
                next_row, next_column = row + delta_row, column + delta_column
                if not (0 <= next_row < GRID and 0 <= next_column < GRID):
                    continue
                neighbor = next_row * GRID + next_column
                if neighbor in selected and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
    return reached == selected


def controlled_clustered_candidate_set(
    clean_candidate_indices: torch.Tensor,
    foreground_labels: torch.Tensor | np.ndarray,
    *,
    fraction: float,
    seed: int,
    identity: str,
) -> ControlledCandidateSet:
    """Inject a nested 8-connected foreground cluster while preserving dictionary size.

    The largest foreground connected component is frozen for every contamination
    level.  The seed changes only the anchor/traversal and the removed clean
    references, so 5/10/20 percent conditions remain nested for a fixed image
    and seed.
    """
    clean = validate_candidate_indices(clean_candidate_indices, minimum=9).cpu()
    labels = torch.as_tensor(foreground_labels, dtype=torch.bool).reshape(-1).cpu()
    if labels.numel() != NUM_PATCHES:
        raise ValueError("foreground labels must contain 1369 values")
    if bool(labels.index_select(0, clean).any()):
        raise ValueError("clean candidate set still contains foreground patches")
    requested = rounded_contamination_count(float(fraction), int(clean.numel()))
    components = _foreground_components(labels)
    largest = components[0] if components else torch.empty(0, dtype=torch.long)
    if requested > int(largest.numel()):
        return ControlledCandidateSet(
            indices=clean.clone(),
            removed_background_indices=torch.empty(0, dtype=torch.long),
            injected_foreground_indices=torch.empty(0, dtype=torch.long),
            requested_fraction=float(fraction),
            realized_fraction=0.0,
            valid=False,
            reason=(
                f"needs a connected FG cluster of {requested} patches but largest component "
                f"contains {largest.numel()}"
            ),
            injection_mode="spatial_cluster_8n",
            selected_component_size=int(largest.numel()),
        )
    if requested:
        cluster_order, anchor = _cluster_bfs_order(
            largest, identity=identity, seed=seed
        )
        injected = cluster_order[:requested]
    else:
        injected = torch.empty(0, dtype=torch.long)
        anchor = -1
    removal_order = _permutation(
        clean, identity=identity, seed=seed, namespace="remove_bg_cluster"
    )
    removed = removal_order[:requested]
    if requested:
        retained = clean[~torch.isin(clean, removed)]
        candidate = torch.cat([retained, injected]).sort().values
    else:
        candidate = clean.clone()
    validate_candidate_indices(candidate, minimum=9)
    if int(candidate.numel()) != int(clean.numel()):
        raise RuntimeError("clustered contamination changed dictionary size")
    actual_wrong = int(labels.index_select(0, candidate).sum())
    connected = _is_eight_connected(injected)
    if actual_wrong != requested or not connected:
        raise RuntimeError("clustered contamination did not realize a connected requested set")
    return ControlledCandidateSet(
        indices=candidate,
        removed_background_indices=removed,
        injected_foreground_indices=injected,
        requested_fraction=float(fraction),
        realized_fraction=actual_wrong / float(clean.numel()),
        valid=True,
        reason="",
        injection_mode="spatial_cluster_8n",
        selected_component_size=int(largest.numel()),
        cluster_anchor_index=anchor,
        injected_connected=connected,
    )


def matched_size_clean_candidate_set(
    natural_candidate_indices: torch.Tensor,
    foreground_labels: torch.Tensor | np.ndarray,
    *,
    seed: int,
    identity: str,
) -> ControlledCandidateSet:
    """Oracle-clean dictionary replenished to the natural dictionary size."""
    natural = validate_candidate_indices(natural_candidate_indices, minimum=9).cpu()
    labels = torch.as_tensor(foreground_labels, dtype=torch.bool).reshape(-1).cpu()
    erroneous = natural[labels.index_select(0, natural)]
    clean = natural[~labels.index_select(0, natural)]
    background_non_candidates = torch.where((~labels) & (~torch.isin(torch.arange(NUM_PATCHES), natural)))[0]
    required = int(erroneous.numel())
    if required > int(background_non_candidates.numel()):
        return ControlledCandidateSet(
            indices=clean,
            removed_background_indices=erroneous,
            injected_foreground_indices=torch.empty(0, dtype=torch.long),
            requested_fraction=0.0,
            realized_fraction=0.0,
            valid=False,
            reason=f"needs {required} replacement BG patches but only {background_non_candidates.numel()} exist",
        )
    order = _permutation(
        background_non_candidates, identity=identity, seed=seed, namespace="matched_clean_bg"
    )
    replacement = order[:required]
    candidate = torch.cat([clean, replacement]).sort().values
    validate_candidate_indices(candidate, minimum=9)
    if candidate.numel() != natural.numel() or bool(labels.index_select(0, candidate).any()):
        raise RuntimeError("matched-size clean intervention is invalid")
    return ControlledCandidateSet(
        indices=candidate,
        removed_background_indices=erroneous,
        injected_foreground_indices=replacement,
        requested_fraction=0.0,
        realized_fraction=0.0,
        valid=True,
        reason="",
    )
