from __future__ import annotations

import torch

from models.reconstruction.local_convex_reconstruction import simplex_projected_gradient
from models.reconstruction.retrieval import retrieve_fullbc_neighbors


def test_convex_solver_recovers_feasible_reconstruction() -> None:
    generator = torch.Generator().manual_seed(7)
    atoms = torch.randn(32, 8, 24, generator=generator)
    truth = torch.rand(32, 8, generator=generator)
    truth /= truth.sum(dim=1, keepdim=True)
    query = torch.einsum("bk,bkd->bd", truth, atoms)
    result = simplex_projected_gradient(query, atoms, max_iterations=256, tolerance=1e-7)
    assert torch.max(result.simplex_sum_error) < 1e-6
    assert torch.min(result.minimum_alpha) >= -1e-7
    assert torch.max(result.residual) < 1e-6
    assert result.objective_increase_max < 1e-5


def test_fullbc_retrieval_strictly_excludes_self() -> None:
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(384, 37, 37, generator=generator)
    background = torch.arange(1369)
    result = retrieve_fullbc_neighbors(features, background, max_k=32)
    query = torch.arange(1369).unsqueeze(1)
    assert result.self_match_violation_count == 0
    assert not torch.any(result.neighbor_indices.cpu() == query)
    assert torch.all(result.cosine_similarities[:, :-1] >= result.cosine_similarities[:, 1:])
