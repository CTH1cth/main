"""Conditional K16 affine reconstruction fallback.

This operator is implemented for completeness but must not be executed unless
LCBR and LSR are both within 0.002 of KNN8, as required by the frozen task.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .retrieval import NUM_PATCHES, RetrievalResult, flatten_feature


@dataclass(frozen=True)
class LocalAffineResult:
    residual: torch.Tensor
    coefficients: torch.Tensor
    affine_sum_error: torch.Tensor
    rcond: float


def local_affine_reconstruct(
    raw_features: torch.Tensor,
    retrieval: RetrievalResult,
    *,
    k: int = 16,
    feature_geometry: str = "l2",
    rcond: float = 1e-6,
    query_batch_size: int = 128,
) -> LocalAffineResult:
    if int(k) != 16:
        raise ValueError("the frozen affine fallback only permits K=16")
    feature_geometry = str(feature_geometry).lower()
    if feature_geometry not in {"l2", "raw"}:
        raise ValueError("feature_geometry must be 'l2' or 'raw'")
    feature = retrieval.normalized_features if feature_geometry == "l2" else flatten_feature(raw_features)
    coefficients, residuals = [], []
    index_all = retrieval.neighbor_indices[:, :16]
    for start in range(0, NUM_PATCHES, int(query_batch_size)):
        stop = min(start + int(query_batch_size), NUM_PATCHES)
        query = feature[start:stop]
        index = index_all[start:stop]
        atoms = feature.index_select(0, index.reshape(-1)).reshape(stop - start, 16, feature.shape[1])
        # Affine constraint: B*a=x and 1^T*a=1 in least-squares form.
        design = torch.cat(
            (atoms.transpose(1, 2), torch.ones(stop - start, 1, 16, device=feature.device, dtype=feature.dtype)),
            dim=1,
        )
        target = torch.cat(
            (query, torch.ones(stop - start, 1, device=feature.device, dtype=feature.dtype)), dim=1
        ).unsqueeze(2)
        # Explicit batched SVD keeps this path CUDA-compatible.  LAPACK's
        # ``gelsd`` driver is CPU-only in PyTorch.
        u, singular, vh = torch.linalg.svd(design, full_matrices=False)
        cutoff = singular.amax(dim=1, keepdim=True) * float(rcond)
        singular_inv = torch.where(singular > cutoff, singular.reciprocal(), 0.0)
        pseudo_inverse = (
            vh.transpose(1, 2)
            @ torch.diag_embed(singular_inv)
            @ u.transpose(1, 2)
        )
        solution = (pseudo_inverse @ target).squeeze(2)
        # Numerically restore the exact affine constraint without a lambda search.
        solution = solution + (1.0 - solution.sum(dim=1, keepdim=True)) / 16.0
        reconstruction = torch.einsum("bk,bkd->bd", solution, atoms)
        coefficients.append(solution)
        residuals.append((query - reconstruction).square().sum(dim=1))
    alpha = torch.cat(coefficients).contiguous()
    residual = torch.cat(residuals).contiguous()
    error = (alpha.sum(dim=1) - 1.0).abs()
    if float(error.max()) >= 1e-5 or not bool(torch.isfinite(residual).all()):
        raise RuntimeError("affine reconstruction failed its numerical contract")
    return LocalAffineResult(residual.float(), alpha.float(), error.float(), float(rcond))
