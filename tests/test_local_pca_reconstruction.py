from __future__ import annotations

import torch

from models.reconstruction.local_pca_reconstruction import local_pca_reconstruct
from models.reconstruction.retrieval import RetrievalResult


def test_local_rank4_subspace_reconstructs_rank4_queries() -> None:
    generator = torch.Generator().manual_seed(11)
    mean = torch.randn(12, generator=generator)
    basis, _ = torch.linalg.qr(torch.randn(12, 4, generator=generator))
    coefficient = torch.randn(1369, 4, generator=generator)
    feature = mean + coefficient @ basis.T
    neighbor = torch.stack([
        (torch.arange(16) + offset * 17) % 1369 for offset in range(1369)
    ])
    retrieval = RetrievalResult(
        normalized_features=torch.nn.functional.normalize(feature, dim=1),
        background_indices=torch.arange(1369),
        neighbor_indices=neighbor,
        cosine_similarities=torch.zeros(1369, 16),
        self_match_violation_count=0,
    )
    result = local_pca_reconstruct(feature, retrieval, k=16, rank=4, feature_geometry="raw")
    assert result.residual.shape == (1369,)
    assert torch.quantile(result.residual, .99) < 1e-8
    assert result.orthonormal_error_max < 1e-4
