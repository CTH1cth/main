"""Global rank-r PCA reconstruction on a frozen background memory."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .retrieval import NUM_PATCHES, flatten_feature


@dataclass(frozen=True)
class GlobalPCAResult:
    residual: torch.Tensor
    mean: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    selected_rank: int
    retained_variance_ratio: float
    svd_orthonormal_error_before_qr: float
    orthonormal_error: float


def global_pca_reconstruct(
    features: torch.Tensor,
    background_indices: torch.Tensor,
    *,
    rank: int = 8,
    normalize_features: bool = False,
) -> GlobalPCAResult:
    query = flatten_feature(features, normalize=bool(normalize_features))
    background_index = torch.as_tensor(
        background_indices, dtype=torch.long, device=query.device
    ).reshape(-1)
    if background_index.numel() < 2 or background_index.numel() != torch.unique(background_index).numel():
        raise ValueError("global PCA needs at least two unique background atoms")
    memory = query.index_select(0, background_index)
    mean = memory.mean(dim=0)
    centered_memory = memory - mean
    _, singular, vh = torch.linalg.svd(centered_memory, full_matrices=False)
    effective = min(int(memory.shape[0]) - 1, int(memory.shape[1]))
    rank = int(rank)
    if rank < 1 or rank > effective:
        raise ValueError(f"rank={rank} is invalid for effective dimension {effective}")
    raw_basis = vh[:rank].T.contiguous()
    raw_gram = raw_basis.T @ raw_basis
    raw_error = float(
        (raw_gram - torch.eye(rank, device=query.device, dtype=query.dtype)).abs().max()
    )
    # CUDA float32 SVD can exceed a strict 1e-4 orthogonality audit by a few
    # 1e-5.  Reduced QR only re-orthogonalizes the same selected span; it does
    # not alter the learned background subspace.
    basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
    basis = basis.contiguous()
    gram = basis.T @ basis
    error = float((gram - torch.eye(rank, device=query.device, dtype=query.dtype)).abs().max())
    if error >= 1e-4:
        raise RuntimeError(f"global PCA basis is not orthonormal: {error}")
    centered_query = query - mean
    projection = (centered_query @ basis) @ basis.T
    residual = (centered_query - projection).square().sum(dim=1)
    if residual.shape != (NUM_PATCHES,) or not bool(torch.isfinite(residual).all()):
        raise RuntimeError("global PCA returned incomplete/non-finite scores")
    energy = singular[:effective].double().square()
    ratio = float(energy[:rank].sum() / energy.sum()) if float(energy.sum()) > 0 else 0.0
    return GlobalPCAResult(
        residual=residual.float().contiguous(),
        mean=mean.float().contiguous(),
        basis=basis.float().contiguous(),
        singular_values=singular.float().contiguous(),
        selected_rank=rank,
        retained_variance_ratio=ratio,
        svd_orthonormal_error_before_qr=raw_error,
        orthonormal_error=error,
    )
