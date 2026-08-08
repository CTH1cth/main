"""Background discovery primitives for GBSP-V2.

This module is deliberately independent from the formal GBSP implementation.
It reuses the frozen local DINO/RGB/Sobel edge cost, but replaces cumulative
shortest-path selection with deterministic random walk propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


GRID = 37
NUM_PATCHES = GRID * GRID
FEATURE_DIM = 384
BOUNDARY_WIDTH = 2
BOUNDARY_COUNT = 280
EPS = 1e-8


class GBSPV2NumericalError(RuntimeError):
    """A numerical failure that must be recorded rather than hidden."""


@dataclass(frozen=True)
class LocalGraph:
    source: torch.Tensor
    target: torch.Tensor
    cost: torch.Tensor
    affinity: torch.Tensor
    transition: torch.Tensor
    degree: torch.Tensor
    edge_scale: float
    row_sum_error: float
    num_nodes: int = NUM_PATCHES

    def transpose_step(self, value: torch.Tensor) -> torch.Tensor:
        vector = value.detach().cpu().double().reshape(-1)
        if vector.numel() != self.num_nodes:
            raise ValueError(f"random-walk vector must have {self.num_nodes} entries")
        output = torch.zeros(self.num_nodes, dtype=torch.float64)
        output.index_add_(
            0,
            self.target,
            self.transition.double() * vector.index_select(0, self.source),
        )
        return output

    def dense_transition(self) -> torch.Tensor:
        """Materialize P for small diagnostic/unit-test use only."""
        value = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float64)
        value.index_put_((self.source, self.target), self.transition.double(), accumulate=True)
        return value


@dataclass(frozen=True)
class RandomWalkResult:
    confidence: torch.Tensor
    iterations: int
    convergence_error: float
    converged: bool
    seed_sum: float


@dataclass(frozen=True)
class SoftSeedResult:
    seed: torch.Tensor
    inward_consistency: torch.Tensor
    global_dino_density: torch.Tensor
    inward_rank: torch.Tensor
    density_rank: torch.Tensor


@dataclass(frozen=True)
class CoresetResult:
    indices: torch.Tensor
    pool_indices: torch.Tensor
    final_min_distance: torch.Tensor


def _matrix(value: torch.Tensor, name: str, rows: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2 or not value.numel():
        raise ValueError(f"{name} must be a non-empty Tensor[N,D]")
    result = value.detach().cpu().float().contiguous()
    if rows is not None and result.shape[0] != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def boundary_two_ring_mask(grid: int = GRID) -> torch.Tensor:
    if int(grid) != GRID:
        raise ValueError(f"formal GBSP-V2 requires grid={GRID}")
    mask = torch.zeros(GRID, GRID, dtype=torch.bool)
    mask[:BOUNDARY_WIDTH] = True
    mask[-BOUNDARY_WIDTH:] = True
    mask[:, :BOUNDARY_WIDTH] = True
    mask[:, -BOUNDARY_WIDTH:] = True
    flat = mask.reshape(-1)
    if int(flat.sum()) != BOUNDARY_COUNT:
        raise RuntimeError(f"two-ring boundary must contain {BOUNDARY_COUNT} patches")
    return flat


def load_rgb_grid(path: str | Path, grid: int = GRID) -> torch.Tensor:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((int(grid), int(grid)), Image.Resampling.BICUBIC)
        value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value.copy()).permute(2, 0, 1).contiguous()


def minmax(value: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    result = value.detach().cpu().float()
    low, high = result.min(), result.max()
    if float(high - low) <= eps:
        return torch.zeros_like(result)
    return ((result - low) / (high - low + eps)).clamp(0.0, 1.0)


def sobel_magnitude(rgb: torch.Tensor) -> torch.Tensor:
    if tuple(rgb.shape) != (3, GRID, GRID):
        raise ValueError(f"RGB grid must be (3,{GRID},{GRID})")
    gray = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)
    padded = F.pad(gray.reshape(1, 1, GRID, GRID), (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)
    return minmax(torch.sqrt(gx.square() + gy.square() + EPS).reshape(-1))


def build_local_graph(
    normalized_features: torch.Tensor,
    rgb: torch.Tensor,
    *,
    sigma_f: float = .10,
    sigma_c: float = .05,
    sigma_e: float = .30,
    eps: float = EPS,
) -> LocalGraph:
    """Build the frozen symmetric eight-neighbour graph and row-stochastic P."""
    features = _matrix(normalized_features, "normalized_features", NUM_PATCHES)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"feature dimension must be {FEATURE_DIM}")
    norms = torch.linalg.vector_norm(features, dim=1)
    if float((norms - 1.0).abs().max()) > 1e-4:
        raise ValueError("GBSP-V2 graph requires per-patch L2-normalized features")
    if tuple(rgb.shape) != (3, GRID, GRID) or not bool(torch.isfinite(rgb).all()):
        raise ValueError(f"rgb must be finite Tensor[3,{GRID},{GRID}]")
    rgb_flat = rgb.permute(1, 2, 0).reshape(NUM_PATCHES, 3).float()
    edge = sobel_magnitude(rgb)

    # Four canonical offsets are duplicated to obtain a symmetric directed graph.
    sources: list[int] = []
    targets: list[int] = []
    costs: list[float] = []
    for row in range(GRID):
        for col in range(GRID):
            source = row * GRID + col
            for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
                other_row, other_col = row + dy, col + dx
                if not (0 <= other_row < GRID and 0 <= other_col < GRID):
                    continue
                target = other_row * GRID + other_col
                feature_cost = 1.0 - float(torch.dot(features[source], features[target]))
                color_cost = float((rgb_flat[source] - rgb_flat[target]).square().sum())
                edge_cost = float(torch.maximum(edge[source], edge[target]))
                cost = max(0.0, feature_cost / sigma_f + color_cost / sigma_c + edge_cost / sigma_e)
                sources.extend((source, target))
                targets.extend((target, source))
                costs.extend((cost, cost))

    source_tensor = torch.tensor(sources, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    cost_tensor = torch.tensor(costs, dtype=torch.float64)
    positive = cost_tensor[cost_tensor > 0]
    if not positive.numel():
        raise GBSPV2NumericalError("positive graph-edge cost set is empty")
    scale = float(positive.median())
    if scale < 1e-8:
        scale = float(positive.mean()) + 1e-8
    affinity = torch.exp(-cost_tensor / (scale + eps))
    if not bool(torch.isfinite(affinity).all()) or bool((affinity < 0).any()):
        raise GBSPV2NumericalError("graph affinity contains invalid values")
    degree = torch.zeros(NUM_PATCHES, dtype=torch.float64)
    degree.index_add_(0, source_tensor, affinity)
    if bool((degree <= 0).any()) or not bool(torch.isfinite(degree).all()):
        raise GBSPV2NumericalError("random-walk graph contains an isolated node")
    transition = affinity / degree.index_select(0, source_tensor)
    row_sum = torch.zeros(NUM_PATCHES, dtype=torch.float64)
    row_sum.index_add_(0, source_tensor, transition)
    row_error = float((row_sum - 1.0).abs().max())
    if row_error > 1e-10:
        raise GBSPV2NumericalError(f"random-walk row normalization failed: {row_error}")
    return LocalGraph(
        source_tensor,
        target_tensor,
        cost_tensor.float().contiguous(),
        affinity.float().contiguous(),
        transition.double().contiguous(),
        degree.float().contiguous(),
        scale,
        row_error,
        NUM_PATCHES,
    )


def average_rank(value: torch.Tensor) -> torch.Tensor:
    """Return deterministic one-based average ranks (ascending, ties averaged)."""
    vector = value.detach().cpu().double().reshape(-1)
    if not vector.numel() or not bool(torch.isfinite(vector).all()):
        raise ValueError("rank input must be non-empty and finite")
    order = torch.argsort(vector, stable=True)
    sorted_value = vector.index_select(0, order)
    ranks = torch.empty_like(vector)
    start = 0
    while start < vector.numel():
        stop = start + 1
        while stop < vector.numel() and bool(sorted_value[stop] == sorted_value[start]):
            stop += 1
        average = .5 * ((start + 1) + stop)
        ranks.index_fill_(0, order[start:stop], average)
        start = stop
    return ranks.float()


def hard_boundary_seed() -> torch.Tensor:
    return boundary_two_ring_mask().float()


def soft_boundary_seed(
    graph: LocalGraph,
    normalized_features: torch.Tensor,
    *,
    top_k: int = 16,
) -> SoftSeedResult:
    features = _matrix(normalized_features, "normalized_features", NUM_PATCHES)
    boundary_indices = torch.where(boundary_two_ring_mask())[0]
    if not 0 < int(top_k) < NUM_PATCHES:
        raise ValueError("soft-seed top_k must be in [1,1368]")
    center = .5 * (GRID - 1)
    rows = torch.div(torch.arange(NUM_PATCHES), GRID, rounding_mode="floor").float()
    cols = (torch.arange(NUM_PATCHES) % GRID).float()
    center_distance = (rows - center).square() + (cols - center).square()
    inward = torch.empty(boundary_indices.numel(), dtype=torch.float32)
    for position, node in enumerate(boundary_indices.tolist()):
        edge_mask = graph.source == node
        neighbours = graph.target[edge_mask]
        affinities = graph.affinity[edge_mask]
        is_inward = center_distance.index_select(0, neighbours) < center_distance[node]
        selected = affinities[is_inward] if bool(is_inward.any()) else affinities
        if not selected.numel():
            raise GBSPV2NumericalError(f"boundary node {node} has no graph neighbour")
        inward[position] = selected.mean()

    similarity = features.index_select(0, boundary_indices) @ features.t()
    similarity[torch.arange(boundary_indices.numel()), boundary_indices] = -torch.inf
    density = torch.topk(similarity, k=int(top_k), dim=1, largest=True, sorted=False).values.mean(1)
    inward_rank = (average_rank(inward) - .5) / boundary_indices.numel()
    density_rank = (average_rank(density) - .5) / boundary_indices.numel()
    boundary_weight = torch.sqrt((inward_rank * density_rank).clamp_min(0.0))
    seed = torch.zeros(NUM_PATCHES, dtype=torch.float32)
    seed.index_copy_(0, boundary_indices, boundary_weight)

    def expand(boundary_value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(NUM_PATCHES, dtype=torch.float32)
        result.index_copy_(0, boundary_indices, boundary_value.float())
        return result

    return SoftSeedResult(
        seed,
        expand(inward),
        expand(density),
        expand(inward_rank),
        expand(density_rank),
    )


def random_walk_with_restart(
    graph: LocalGraph,
    seed: torch.Tensor,
    *,
    alpha: float = .85,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> RandomWalkResult:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0,1)")
    initial = seed.detach().cpu().double().reshape(-1)
    if initial.numel() != graph.num_nodes or not bool(torch.isfinite(initial).all()) or bool((initial < 0).any()):
        raise ValueError(f"random-walk seed must contain {graph.num_nodes} finite nonnegative values")
    current = initial.clone()
    error = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        updated = (1.0 - alpha) * initial + alpha * graph.transpose_step(current)
        error = float((updated - current).abs().sum())
        current = updated
        if error < tolerance:
            converged = True
            break
    if not bool(torch.isfinite(current).all()) or bool((current < -1e-12).any()):
        raise GBSPV2NumericalError("random-walk confidence is invalid")
    return RandomWalkResult(
        current.clamp_min(0.0).float().contiguous(),
        iteration,
        error,
        converged,
        float(initial.sum()),
    )


def top_confidence_indices(confidence: torch.Tensor, count: int) -> torch.Tensor:
    value = confidence.detach().cpu().float().reshape(-1)
    if value.numel() != NUM_PATCHES or not bool(torch.isfinite(value).all()):
        raise ValueError("confidence must contain 1369 finite values")
    if not 0 < int(count) <= NUM_PATCHES:
        raise ValueError("invalid background candidate count")
    # Stable sorting preserves ascending linear patch index for exact ties.
    return torch.argsort(value, descending=True, stable=True)[: int(count)].long().contiguous()


def confidence_diversity_coreset(
    confidence: torch.Tensor,
    normalized_features: torch.Tensor,
    count: int,
) -> CoresetResult:
    value = confidence.detach().cpu().float().reshape(-1)
    features = _matrix(normalized_features, "normalized_features", NUM_PATCHES)
    count = int(count)
    if not 0 < count <= NUM_PATCHES:
        raise ValueError("invalid coreset size")
    pool_size = min(2 * count, NUM_PATCHES)
    pool = top_confidence_indices(value, pool_size)
    selected = [int(pool[0])]
    remaining = torch.ones(pool_size, dtype=torch.bool)
    remaining[0] = False
    first_distance = 1.0 - features.index_select(0, pool) @ features[selected[0]]
    minimum_distance = first_distance.clamp_min(0.0)

    while len(selected) < count:
        positions = torch.where(remaining)[0]
        patch_indices = pool.index_select(0, positions)
        confidence_rank = average_rank(value.index_select(0, patch_indices))
        distance_rank = average_rank(minimum_distance.index_select(0, positions))
        size = positions.numel()
        confidence_ecdf = confidence_rank / size
        distance_ecdf = distance_rank / size
        joint = torch.sqrt((confidence_ecdf * distance_ecdf).clamp_min(0.0))
        maximum = joint.max()
        tied_patches = patch_indices[joint == maximum]
        chosen_patch = int(tied_patches.min())
        chosen_position = int(torch.where(pool == chosen_patch)[0].item())
        selected.append(chosen_patch)
        remaining[chosen_position] = False
        new_distance = 1.0 - features.index_select(0, pool) @ features[chosen_patch]
        minimum_distance = torch.minimum(minimum_distance, new_distance.clamp_min(0.0))

    indices = torch.tensor(selected, dtype=torch.long)
    if indices.numel() != count or torch.unique(indices).numel() != count:
        raise GBSPV2NumericalError("confidence-diversity selection is incomplete")
    return CoresetResult(indices, pool, minimum_distance.float().contiguous())


def candidate_diagnostics(
    indices: torch.Tensor,
    normalized_features: torch.Tensor,
    singular_values: torch.Tensor | None = None,
) -> dict[str, float | int]:
    selected = indices.detach().cpu().long().reshape(-1)
    features = _matrix(normalized_features, "normalized_features", NUM_PATCHES)
    if not selected.numel() or selected.numel() != torch.unique(selected).numel():
        raise ValueError("candidate indices must be non-empty and unique")
    boundary = boundary_two_ring_mask().index_select(0, selected)
    rows = torch.div(selected, GRID, rounding_mode="floor")
    cols = selected % GRID
    depth = torch.minimum(torch.minimum(rows, GRID - 1 - rows), torch.minimum(cols, GRID - 1 - cols)).float()
    selected_features = features.index_select(0, selected)
    pair_distance = (1.0 - selected_features @ selected_features.t()).clamp_min(0.0)
    pair_distance.fill_diagonal_(torch.inf)
    nearest = pair_distance.min(1).values
    finite_pair = pair_distance[torch.isfinite(pair_distance)]
    singular = (
        singular_values.detach().cpu().double().reshape(-1)
        if singular_values is not None
        else torch.linalg.svdvals(selected_features - selected_features.mean(0)).double()
    )
    energy = singular.square()
    probability = energy / energy.sum() if float(energy.sum()) > 0 else energy
    positive = probability > 0
    effective_rank = float(torch.exp(-(probability[positive] * torch.log(probability[positive])).sum())) if bool(positive.any()) else 0.0
    coverage_similarity = features @ selected_features.t()
    return {
        "candidate_count": int(selected.numel()),
        "boundary_candidate_count": int(boundary.sum()),
        "interior_candidate_count": int((~boundary).sum()),
        "boundary_candidate_ratio": float(boundary.float().mean()),
        "candidate_spatial_depth_mean": float(depth.mean()),
        "candidate_spatial_depth_median": float(depth.median()),
        "candidate_pairwise_cosine_distance": float(finite_pair.mean()) if finite_pair.numel() else 0.0,
        "nearest_neighbor_distance": float(nearest.mean()) if nearest.numel() > 1 else 0.0,
        "effective_rank": effective_rank,
        "feature_coverage": float(coverage_similarity.max(1).values.mean()),
    }
