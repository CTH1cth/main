"""Query-conditioned convex reconstruction with batched simplex PGD."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .retrieval import NUM_PATCHES, RetrievalResult, flatten_feature


@dataclass(frozen=True)
class SimplexPGDResult:
    coefficients: torch.Tensor
    residual: torch.Tensor
    iteration_count: torch.Tensor
    converged: torch.Tensor
    simplex_sum_error: torch.Tensor
    minimum_alpha: torch.Tensor
    objective_initial: torch.Tensor
    objective_final: torch.Tensor
    objective_increase_max: float


@dataclass(frozen=True)
class ConvexReconstructionResult:
    k: int
    feature_geometry: str
    residual: torch.Tensor
    coefficients: torch.Tensor
    iteration_count: torch.Tensor
    converged: torch.Tensor
    simplex_sum_error: torch.Tensor
    minimum_alpha: torch.Tensor
    objective_initial: torch.Tensor
    objective_final: torch.Tensor
    objective_increase_max: float
    effective_active_atoms: torch.Tensor
    largest_alpha: torch.Tensor
    top2_alpha_sum: torch.Tensor
    top4_alpha_sum: torch.Tensor
    alpha_entropy: torch.Tensor


def project_probability_simplex(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of the final dimension onto the probability simplex."""
    if not torch.is_tensor(value) or value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("simplex input must have a non-empty final dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("simplex input contains NaN/Inf")
    original_shape = value.shape
    flat = value.reshape(-1, original_shape[-1])
    ordered, _ = torch.sort(flat, dim=1, descending=True)
    cumulative = torch.cumsum(ordered, dim=1) - 1.0
    divisor = torch.arange(
        1, flat.shape[1] + 1, dtype=flat.dtype, device=flat.device
    ).unsqueeze(0)
    valid = ordered - cumulative / divisor > 0
    rho = valid.sum(dim=1).clamp_min(1) - 1
    theta = cumulative.gather(1, rho.unsqueeze(1)).squeeze(1) / (rho.to(flat.dtype) + 1.0)
    projected = torch.clamp(flat - theta.unsqueeze(1), min=0.0)
    return projected.reshape(original_shape).contiguous()


def _objective(gram: torch.Tensor, correlation: torch.Tensor, x_norm2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    quadratic = torch.einsum("bi,bij,bj->b", alpha, gram, alpha)
    return quadratic - 2.0 * (correlation * alpha).sum(dim=1) + x_norm2


def simplex_projected_gradient(
    query: torch.Tensor,
    atoms: torch.Tensor,
    *,
    max_iterations: int = 64,
    tolerance: float = 1e-6,
) -> SimplexPGDResult:
    """Solve ``min ||x-Ba||²`` under ``a>=0, sum(a)=1`` in batch."""
    if query.ndim != 2 or atoms.ndim != 3 or query.shape[0] != atoms.shape[0] or query.shape[1] != atoms.shape[2]:
        raise ValueError("query/atoms must be [B,D] and [B,K,D]")
    if atoms.shape[1] < 2:
        raise ValueError("convex reconstruction requires at least two atoms")
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(atoms).all()):
        raise ValueError("query/atoms contain NaN/Inf")
    max_iterations, tolerance = int(max_iterations), float(tolerance)
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid PGD stopping parameters")

    # On the probability simplex sum(alpha)=1, subtracting the same local
    # centroid from query and every atom leaves the residual exactly unchanged:
    #   x - sum(a_i b_i) == (x-mu) - sum(a_i (b_i-mu)).
    # It removes the irrelevant common-feature direction from G and avoids an
    # excessively small Lipschitz step, especially for raw DINO keys.
    atom_mean = atoms.mean(dim=1, keepdim=True)
    optimization_atoms = atoms - atom_mean
    optimization_query = query - atom_mean.squeeze(1)
    gram = optimization_atoms @ optimization_atoms.transpose(1, 2)
    correlation = torch.einsum("bkd,bd->bk", optimization_atoms, optimization_query)
    x_norm2 = optimization_query.square().sum(dim=1)
    # G is PSD. The prescribed Lipschitz constant is 2*lambda_max(G).
    largest = torch.linalg.eigvalsh(gram).select(1, gram.shape[1] - 1).clamp_min(0.0)
    step = 1.0 / (2.0 * largest + 1e-12)
    uniform = torch.full(
        (query.shape[0], atoms.shape[1]),
        1.0 / float(atoms.shape[1]),
        dtype=query.dtype,
        device=query.device,
    )
    # Equality-constrained least-squares warm start, followed by an exact
    # simplex projection.  A tiny scale-relative ridge only stabilizes nearly
    # duplicate local atoms; failed KKT systems fall back to uniform weights.
    k = atoms.shape[1]
    gram_scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)
    regularized = gram + torch.eye(k, dtype=query.dtype, device=query.device).unsqueeze(0) * (
        gram_scale * 1e-7
    ).reshape(-1, 1, 1)
    kkt = torch.zeros(query.shape[0], k + 1, k + 1, dtype=query.dtype, device=query.device)
    kkt[:, :k, :k] = regularized
    kkt[:, :k, k] = 1.0
    kkt[:, k, :k] = 1.0
    right = torch.cat(
        (correlation, torch.ones(query.shape[0], 1, dtype=query.dtype, device=query.device)),
        dim=1,
    )
    warm, info = torch.linalg.solve_ex(kkt, right.unsqueeze(2), check_errors=False)
    warm = warm[:, :k, 0]
    valid_warm = (info == 0) & torch.isfinite(warm).all(dim=1)
    alpha = project_probability_simplex(
        torch.where(valid_warm.unsqueeze(1), warm, uniform)
    )
    objective_initial = _objective(gram, correlation, x_norm2, alpha)
    previous_objective = objective_initial
    extrapolated = alpha.clone()
    momentum = torch.ones(query.shape[0], dtype=query.dtype, device=query.device)
    iteration_count = torch.full(
        (query.shape[0],), max_iterations, dtype=torch.int16, device=query.device
    )
    converged = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
    objective_increase_max = 0.0
    for iteration in range(1, max_iterations + 1):
        already_converged = converged.clone()
        # Monotone accelerated projected gradient.  The fallback step is the
        # original PGD update, so acceleration can never increase the stored
        # objective relative to the feasible current iterate.
        gradient_fast = 2.0 * (
            torch.einsum("bij,bj->bi", gram, extrapolated) - correlation
        )
        candidate_fast = project_probability_simplex(
            extrapolated - step.unsqueeze(1) * gradient_fast
        )
        objective_fast = _objective(gram, correlation, x_norm2, candidate_fast)
        restart = objective_fast > previous_objective
        if bool(restart.any()):
            gradient_plain = 2.0 * (
                torch.einsum("bij,bj->bi", gram, alpha) - correlation
            )
            candidate_plain = project_probability_simplex(
                alpha - step.unsqueeze(1) * gradient_plain
            )
            candidate = torch.where(restart.unsqueeze(1), candidate_plain, candidate_fast)
        else:
            candidate = candidate_fast
        delta = (candidate - alpha).abs().amax(dim=1)
        objective = _objective(gram, correlation, x_norm2, candidate)
        active = ~already_converged
        if bool(active.any()):
            objective_increase_max = max(
                objective_increase_max,
                float((objective[active] - previous_objective[active]).max().item()),
            )
        newly = (~converged) & (delta < tolerance)
        iteration_count[newly] = int(iteration)
        next_momentum = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * momentum.square()))
        acceleration = ((momentum - 1.0) / next_momentum).unsqueeze(1)
        next_extrapolated = candidate + acceleration * (candidate - alpha)
        # A monotonicity restart clears momentum for affected cases.
        next_extrapolated = torch.where(restart.unsqueeze(1), candidate, next_extrapolated)
        next_momentum = torch.where(restart, torch.ones_like(next_momentum), next_momentum)
        alpha = torch.where(already_converged.unsqueeze(1), alpha, candidate)
        previous_objective = torch.where(already_converged, previous_objective, objective)
        extrapolated = torch.where(already_converged.unsqueeze(1), alpha, next_extrapolated)
        momentum = torch.where(already_converged, momentum, next_momentum)
        converged |= newly
        if bool(converged.all()):
            break

    reconstruction = torch.einsum("bk,bkd->bd", alpha, atoms)
    residual = (query - reconstruction).square().sum(dim=1)
    sum_error = (alpha.sum(dim=1) - 1.0).abs()
    minimum = alpha.min(dim=1).values
    if float(sum_error.max()) >= 1e-5 or float(minimum.min()) < -1e-7:
        raise RuntimeError(
            f"invalid simplex solution: sum_error={float(sum_error.max())}, min={float(minimum.min())}"
        )
    if not bool(torch.isfinite(residual).all()) or bool((residual < -1e-7).any()):
        raise RuntimeError("convex reconstruction returned an invalid residual")
    return SimplexPGDResult(
        coefficients=alpha.contiguous(),
        residual=residual.clamp_min(0).float().contiguous(),
        iteration_count=iteration_count,
        converged=converged,
        simplex_sum_error=sum_error.float().contiguous(),
        minimum_alpha=minimum.float().contiguous(),
        objective_initial=objective_initial.float().contiguous(),
        objective_final=previous_objective.float().contiguous(),
        objective_increase_max=objective_increase_max,
    )


def local_convex_reconstruct(
    raw_features: torch.Tensor,
    retrieval: RetrievalResult,
    *,
    k: int,
    feature_geometry: str = "l2",
    query_batch_size: int = 256,
    max_iterations: int = 64,
    tolerance: float = 1e-6,
) -> ConvexReconstructionResult:
    """Run LCBR for all 1369 queries using frozen retrieved indices."""
    feature_geometry = str(feature_geometry).lower()
    if feature_geometry not in {"l2", "raw"}:
        raise ValueError("feature_geometry must be 'l2' or 'raw'")
    feature = (
        retrieval.normalized_features
        if feature_geometry == "l2"
        else flatten_feature(raw_features, normalize=False)
    )
    k, query_batch_size = int(k), int(query_batch_size)
    if k < 2 or k > retrieval.neighbor_indices.shape[1]:
        raise ValueError(f"invalid LCBR K={k}")
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be positive")

    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "coefficients", "residual", "iteration_count", "converged",
            "simplex_sum_error", "minimum_alpha", "objective_initial", "objective_final",
        )
    }
    increase = 0.0
    indices = retrieval.neighbor_indices[:, :k]
    for start in range(0, NUM_PATCHES, query_batch_size):
        stop = min(start + query_batch_size, NUM_PATCHES)
        query = feature[start:stop]
        local_index = indices[start:stop]
        atoms = feature.index_select(0, local_index.reshape(-1)).reshape(
            stop - start, k, feature.shape[1]
        )
        solved = simplex_projected_gradient(
            query, atoms, max_iterations=max_iterations, tolerance=tolerance
        )
        for name in parts:
            parts[name].append(getattr(solved, name))
        increase = max(increase, solved.objective_increase_max)
    merged = {name: torch.cat(value, dim=0).contiguous() for name, value in parts.items()}
    alpha = merged["coefficients"]
    ordered = torch.sort(alpha, dim=1, descending=True).values
    entropy = -(alpha * alpha.clamp_min(1e-24).log()).sum(dim=1)
    effective = 1.0 / alpha.square().sum(dim=1).clamp_min(1e-24)
    return ConvexReconstructionResult(
        k=k,
        feature_geometry=feature_geometry,
        residual=merged["residual"],
        coefficients=alpha,
        iteration_count=merged["iteration_count"],
        converged=merged["converged"],
        simplex_sum_error=merged["simplex_sum_error"],
        minimum_alpha=merged["minimum_alpha"],
        objective_initial=merged["objective_initial"],
        objective_final=merged["objective_final"],
        objective_increase_max=increase,
        effective_active_atoms=effective.float().contiguous(),
        largest_alpha=ordered[:, 0].float().contiguous(),
        top2_alpha_sum=ordered[:, : min(2, k)].sum(dim=1).float().contiguous(),
        top4_alpha_sum=ordered[:, : min(4, k)].sum(dim=1).float().contiguous(),
        alpha_entropy=entropy.float().contiguous(),
    )
