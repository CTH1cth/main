"""GT-free local background scale normalization."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalScaleResult:
    raw_residual: torch.Tensor
    centroid_scale: torch.Tensor
    pairwise_scale: torch.Tensor
    centroid_normalized: torch.Tensor
    pairwise_normalized: torch.Tensor
    centroid_floor: torch.Tensor
    pairwise_floor: torch.Tensor


def normalize_local_residual(residual: torch.Tensor, local_atoms: torch.Tensor) -> LocalScaleResult:
    residual = torch.as_tensor(residual).float().reshape(-1)
    atoms = torch.as_tensor(local_atoms).float()
    if atoms.ndim != 3 or atoms.shape[0] != residual.numel() or atoms.shape[1] < 2:
        raise ValueError("local_atoms must be [N,K,D] aligned with residual")
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(atoms).all()):
        raise ValueError("scale inputs contain NaN/Inf")
    mean = atoms.mean(dim=1, keepdim=True)
    centroid_distance = (atoms - mean).square().sum(dim=2)
    centroid_scale = centroid_distance.median(dim=1).values
    # Use the Gram identity instead of materialising [N,K,K,D].  For the
    # formal N=1369, K=32 setting this avoids a multi-gigabyte temporary.
    atom_norm2 = atoms.square().sum(dim=2)
    pair_distance = (
        atom_norm2.unsqueeze(2)
        + atom_norm2.unsqueeze(1)
        - 2.0 * (atoms @ atoms.transpose(1, 2))
    ).clamp_min_(0.0)
    upper = torch.triu_indices(atoms.shape[1], atoms.shape[1], offset=1, device=atoms.device)
    pairwise_scale = pair_distance[:, upper[0], upper[1]].median(dim=1).values
    centroid_floor = 1e-6 * centroid_scale + 1e-12
    pairwise_floor = 1e-6 * pairwise_scale + 1e-12
    return LocalScaleResult(
        raw_residual=residual,
        centroid_scale=centroid_scale.float(),
        pairwise_scale=pairwise_scale.float(),
        centroid_normalized=(residual / (centroid_scale + centroid_floor)).float(),
        pairwise_normalized=(residual / (pairwise_scale + pairwise_floor)).float(),
        centroid_floor=centroid_floor.float(),
        pairwise_floor=pairwise_floor.float(),
    )
