from __future__ import annotations

import torch

from models.reconstruction.local_convex_reconstruction import project_probability_simplex


def test_simplex_projection_is_feasible_and_idempotent() -> None:
    generator = torch.Generator().manual_seed(20260807)
    value = torch.randn(128, 16, generator=generator)
    projected = project_probability_simplex(value)
    assert torch.all(projected >= 0)
    assert torch.max(torch.abs(projected.sum(dim=1) - 1)) < 1e-6
    assert torch.max(torch.abs(project_probability_simplex(projected) - projected)) < 1e-6


def test_simplex_projection_preserves_valid_points() -> None:
    value = torch.tensor([[.1, .2, .3, .4], [0., 0., 1., 0.]], dtype=torch.float32)
    assert torch.allclose(project_probability_simplex(value), value, atol=1e-7, rtol=0)
