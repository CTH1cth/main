from __future__ import annotations

import numpy as np
import torch

from models.gbsp_candidate_influence import (
    candidate_influence_proxy,
    exact_loo_batch,
    projector_frobenius_error,
)
from tools.gbsp_candidate_influence_common import high_is_one_percentile, top_mask


def _basis(value: torch.Tensor, rank: int) -> torch.Tensor:
    centered = value - value.mean(0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank].t().contiguous()


def test_exact_rank_one_downdate_matches_literal_delete() -> None:
    generator = torch.Generator().manual_seed(17)
    background = torch.randn(31, 12, generator=generator, dtype=torch.float64)
    rank = 4
    original = _basis(background, rank)
    positions = torch.tensor([0, 3, 11, 30])
    fast = exact_loo_batch(background, positions, original, original.unsqueeze(0))
    for offset, position in enumerate(positions.tolist()):
        keep = torch.ones(background.shape[0], dtype=torch.bool)
        keep[position] = False
        reduced = background[keep]
        brute_mean = reduced.mean(0)
        brute_basis = _basis(reduced, rank)
        assert float((fast.mean[offset] - brute_mean).abs().max()) < 1e-12
        assert float(projector_frobenius_error(fast.basis[offset], brute_basis)) < 1e-6


def test_proxy_has_requested_raw_and_gap_forms() -> None:
    generator = torch.Generator().manual_seed(23)
    background = torch.randn(40, 9, generator=generator)
    mean = background.mean(0)
    _, singular, vh = torch.linalg.svd(background - mean, full_matrices=False)
    basis = vh[:3].t()
    result = candidate_influence_proxy(background, mean, basis, singular)
    assert torch.allclose(
        result.raw,
        result.in_subspace_energy * result.orthogonal_energy,
    )
    assert result.spectral_gap >= 0
    assert torch.allclose(result.gap_normalized, result.raw / (result.spectral_gap + 1e-12))


def test_percentile_and_top_mask_are_stable_and_high_is_one() -> None:
    values = np.asarray([1.0, 3.0, 3.0, 2.0])
    percentile = high_is_one_percentile(values)
    assert np.allclose(percentile, [0.25, 1.0, 0.75, 0.5])
    assert top_mask(values, 0.5).tolist() == [False, True, True, False]
