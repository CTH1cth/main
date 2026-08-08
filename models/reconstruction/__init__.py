"""Training-free background reconstruction operators.

The package is intentionally independent from the frozen GBSP implementation.
All operators consume an already selected Full-BC memory; none of them reads
ground truth or changes the background candidate set.
"""

from .global_pca_reconstruction import GlobalPCAResult, global_pca_reconstruct
from .local_affine_reconstruction import LocalAffineResult, local_affine_reconstruct
from .local_convex_reconstruction import (
    ConvexReconstructionResult,
    SimplexPGDResult,
    local_convex_reconstruct,
    project_probability_simplex,
    simplex_projected_gradient,
)
from .local_pca_reconstruction import LocalPCAResult, local_pca_reconstruct
from .local_scale_normalization import LocalScaleResult, normalize_local_residual
from .retrieval import RetrievalResult, flatten_feature, retrieve_fullbc_neighbors

__all__ = [
    "ConvexReconstructionResult",
    "GlobalPCAResult",
    "LocalAffineResult",
    "LocalPCAResult",
    "LocalScaleResult",
    "RetrievalResult",
    "SimplexPGDResult",
    "flatten_feature",
    "global_pca_reconstruct",
    "local_affine_reconstruct",
    "local_convex_reconstruct",
    "local_pca_reconstruct",
    "normalize_local_residual",
    "project_probability_simplex",
    "retrieve_fullbc_neighbors",
    "simplex_projected_gradient",
]
