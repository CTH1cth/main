"""Closed-form multi-background PCA subspace projection.

The implementation is deliberately independent from the trainable model
stack.  It consumes frozen patch descriptors and contains no parameters that
are learned across images.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MBSPScore:
    relative_residual: torch.Tensor
    absolute_residual: torch.Tensor
    best_subspace_index: torch.Tensor
    per_subspace_relative: torch.Tensor
    per_subspace_absolute: torch.Tensor


def _require_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2:
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must be Tensor[N,D], got {shape}")
    value = value.detach().to(dtype=torch.float32).contiguous()
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    norm = torch.linalg.vector_norm(value, dim=1)
    if bool((norm <= 0).any()):
        raise ValueError(f"{name} contains a zero vector")
    return F.normalize(value, p=2, dim=1)


class MultiBackgroundSubspaceProjector:
    """Fit image-specific affine PCA subspaces and score patch residuals."""

    def __init__(
        self,
        num_subspaces: int = 4,
        min_cluster_size: int = 16,
        pca_energy: float = 0.90,
        pca_max_rank: int = 8,
        pca_min_rank: int = 1,
        seed: int = 0,
        eps: float = 1e-8,
        kmeans_n_init: int = 10,
        kmeans_max_iter: int = 100,
    ) -> None:
        if int(num_subspaces) < 1:
            raise ValueError("num_subspaces must be >= 1")
        if int(min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if not 0.0 < float(pca_energy) <= 1.0:
            raise ValueError("pca_energy must be in (0,1]")
        if int(pca_max_rank) < 0:
            raise ValueError("pca_max_rank must be >= 0")
        if int(pca_min_rank) < 0 or int(pca_min_rank) > int(pca_max_rank):
            raise ValueError("pca_min_rank must be in [0,pca_max_rank]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if int(kmeans_n_init) < 1 or int(kmeans_max_iter) < 1:
            raise ValueError("K-means iteration counts must be positive")

        self.num_subspaces = int(num_subspaces)
        self.min_cluster_size = int(min_cluster_size)
        self.pca_energy = float(pca_energy)
        self.pca_max_rank = int(pca_max_rank)
        self.pca_min_rank = int(pca_min_rank)
        self.seed = int(seed)
        self.eps = float(eps)
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self._is_fitted = False

    @staticmethod
    def _centers(features: torch.Tensor, assignments: torch.Tensor, count: int) -> torch.Tensor:
        centers = []
        for cluster_index in range(count):
            members = features[assignments == cluster_index]
            if members.shape[0] == 0:
                raise RuntimeError("empty cluster survived center recomputation")
            center = members.mean(dim=0, keepdim=True)
            centers.append(F.normalize(center, p=2, dim=1).squeeze(0))
        return torch.stack(centers, dim=0)

    def _initial_centers(
        self,
        features: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        num_samples = int(features.shape[0])
        first = int(torch.randint(num_samples, (1,), generator=generator).item())
        selected = [first]
        closest_distance = 1.0 - features @ features[first]
        for _ in range(1, count):
            probability = closest_distance.clamp_min(0.0)
            probability[selected] = 0.0
            total = float(probability.sum())
            if total <= self.eps:
                remaining = [index for index in range(num_samples) if index not in selected]
                chosen = remaining[int(torch.randint(len(remaining), (1,), generator=generator).item())]
            else:
                chosen = int(torch.multinomial(probability / probability.sum(), 1, generator=generator).item())
            selected.append(chosen)
            distance = 1.0 - features @ features[chosen]
            closest_distance = torch.minimum(closest_distance, distance)
        return features[selected].clone()

    def _one_kmeans(
        self,
        features: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        centers = self._initial_centers(features, count, generator)
        previous = None
        iterations = 0
        for iteration in range(self.kmeans_max_iter):
            similarity = features @ centers.t()
            assignments = similarity.argmax(dim=1)
            sizes = torch.bincount(assignments, minlength=count)
            empty = torch.where(sizes == 0)[0]
            if empty.numel():
                nearest = similarity.max(dim=1).values
                candidates = torch.argsort(nearest, descending=False, stable=True)
                used = set()
                for cluster_index in empty.tolist():
                    sample_index = next(
                        int(index)
                        for index in candidates.tolist()
                        if int(index) not in used
                        and int(sizes[int(assignments[int(index)])]) > 1
                    )
                    sizes[int(assignments[sample_index])] -= 1
                    assignments[sample_index] = cluster_index
                    sizes[cluster_index] += 1
                    used.add(sample_index)
            centers = self._centers(features, assignments, count)
            iterations = iteration + 1
            if previous is not None and torch.equal(assignments, previous):
                break
            previous = assignments.clone()
        objective = float((features * centers.index_select(0, assignments)).sum(dim=1).sum())
        return assignments, centers, objective, iterations

    def _spherical_kmeans(
        self, features: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        best = None
        objectives = []
        iterations = []
        for init_index in range(self.kmeans_n_init):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + init_index * 1_000_003)
            result = self._one_kmeans(features, count, generator)
            objectives.append(result[2])
            iterations.append(result[3])
            if best is None or result[2] > best[2] + self.eps:
                best = result
        assert best is not None
        return best[0], best[1], {
            "objectives": objectives,
            "iterations": iterations,
            "best_initialization": int(max(range(len(objectives)), key=objectives.__getitem__)),
        }

    def _merge_small_clusters(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        centers: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        original_count = int(centers.shape[0])
        original_sizes = torch.bincount(assignments, minlength=original_count)
        legal = torch.where(original_sizes >= self.min_cluster_size)[0]
        small = torch.where(original_sizes < self.min_cluster_size)[0]
        merged_samples = int(original_sizes.index_select(0, small).sum()) if small.numel() else 0
        if legal.numel() == 0:
            final = torch.zeros_like(assignments)
            return final, {
                "original_cluster_sizes": original_sizes.tolist(),
                "num_original_clusters": original_count,
                "num_small_clusters": int(small.numel()),
                "num_merged_samples": merged_samples,
                "fallback_no_legal_cluster": True,
            }

        final = assignments.clone()
        legal_centers = centers.index_select(0, legal)
        for small_index in small.tolist():
            members = torch.where(assignments == small_index)[0]
            if members.numel() == 0:
                continue
            member_center = F.normalize(
                features.index_select(0, members).mean(dim=0, keepdim=True), p=2, dim=1
            )
            target = int((member_center @ legal_centers.t()).argmax())
            final[members] = legal[target]

        remap = {int(old): new for new, old in enumerate(legal.tolist())}
        for old, new in remap.items():
            final[final == old] = new
        return final, {
            "original_cluster_sizes": original_sizes.tolist(),
            "num_original_clusters": original_count,
            "num_small_clusters": int(small.numel()),
            "num_merged_samples": merged_samples,
            "fallback_no_legal_cluster": False,
        }

    def _fit_subspaces(self, features: torch.Tensor, assignments: torch.Tensor) -> None:
        subspace_count = int(assignments.max()) + 1
        means = []
        bases = []
        singular_values = []
        ranks = []
        cluster_sizes = []
        zero_variance = []
        retained_energy = []
        representative_indices = []
        svd_start = time.perf_counter()

        for cluster_index in range(subspace_count):
            member_index = torch.where(assignments == cluster_index)[0]
            members = features.index_select(0, member_index)
            mean = members.mean(dim=0)
            centered = members - mean
            # Rank-0 is the explicit background-mean prototype ablation.  It
            # must not fit PCA and therefore must not call SVD at all.
            if self.pca_max_rank == 0:
                singular = features.new_zeros((0,))
                total_energy = float(centered.square().sum())
                max_rank = 0
                right = None
            else:
                _, singular, right = torch.linalg.svd(centered, full_matrices=False)
                total_energy = float(singular.square().sum())
                max_rank = min(
                    self.pca_max_rank,
                    int(members.shape[0]) - 1,
                    int(features.shape[1]),
                )
            if total_energy <= self.eps or max_rank <= 0:
                rank = 0
                basis = features.new_zeros((features.shape[1], 0))
                captured = 0.0
                is_zero = total_energy <= self.eps
            else:
                assert right is not None
                energy = singular.square()
                cumulative = torch.cumsum(energy, dim=0) / (energy.sum() + self.eps)
                energy_rank = int(torch.searchsorted(cumulative, self.pca_energy).item()) + 1
                rank = min(max_rank, max(self.pca_min_rank, energy_rank))
                basis = right[:rank].t().contiguous()
                gram_error = float((basis.t() @ basis - torch.eye(rank)).abs().max())
                if gram_error >= 1e-4:
                    raise RuntimeError(f"PCA basis is not orthonormal: error={gram_error}")
                captured = float(energy[:rank].sum() / (energy.sum() + self.eps))
                is_zero = False

            mean_distance = torch.linalg.vector_norm(members - mean, dim=1)
            representative_indices.append(int(member_index[int(mean_distance.argmin())]))
            means.append(mean)
            bases.append(basis)
            singular_values.append(singular.contiguous())
            ranks.append(rank)
            cluster_sizes.append(int(members.shape[0]))
            zero_variance.append(is_zero)
            retained_energy.append(captured)

        self.timing["svd_seconds"] = time.perf_counter() - svd_start
        self.subspace_means = torch.stack(means, dim=0).contiguous()
        self.subspace_bases = tuple(bases)
        self.singular_values = tuple(singular_values)
        self.selected_ranks = torch.tensor(ranks, dtype=torch.long)
        self.cluster_sizes = torch.tensor(cluster_sizes, dtype=torch.long)
        self.zero_variance_clusters = torch.tensor(zero_variance, dtype=torch.bool)
        self.retained_energy = torch.tensor(retained_energy, dtype=torch.float32)
        self.representative_background_indices = torch.tensor(
            representative_indices, dtype=torch.long
        )

    def fit(self, background_features: torch.Tensor) -> "MultiBackgroundSubspaceProjector":
        features = _require_matrix(background_features, "background_features").cpu()
        self.timing = {"kmeans_seconds": 0.0, "svd_seconds": 0.0, "score_seconds": 0.0}
        count = int(features.shape[0])
        fallback = {
            "insufficient_background_atoms": count < self.min_cluster_size,
            "fallback_single_center": False,
            "fallback_single_subspace": False,
            "requested_subspaces_reduced": False,
        }

        kmeans_meta: dict[str, Any] = {}
        if count < self.min_cluster_size:
            assignments = torch.zeros(count, dtype=torch.long)
            fallback["fallback_single_center"] = True
            merge_meta = {
                "original_cluster_sizes": [count],
                "num_original_clusters": 1,
                "num_small_clusters": 1,
                "num_merged_samples": 0,
                "fallback_no_legal_cluster": False,
            }
        elif self.num_subspaces == 1:
            assignments = torch.zeros(count, dtype=torch.long)
            merge_meta = {
                "original_cluster_sizes": [count],
                "num_original_clusters": 1,
                "num_small_clusters": 0,
                "num_merged_samples": 0,
                "fallback_no_legal_cluster": False,
            }
        else:
            requested = min(self.num_subspaces, count)
            fallback["requested_subspaces_reduced"] = requested != self.num_subspaces
            start = time.perf_counter()
            initial, centers, kmeans_meta = self._spherical_kmeans(features, requested)
            self.timing["kmeans_seconds"] = time.perf_counter() - start
            assignments, merge_meta = self._merge_small_clusters(features, initial, centers)

        effective_count = int(assignments.max()) + 1
        if effective_count == 1 and self.num_subspaces > 1:
            fallback["fallback_single_subspace"] = True
        if fallback["fallback_single_center"]:
            mean = features.mean(dim=0, keepdim=True)
            is_zero_variance = bool((features - mean).square().sum() <= self.eps)
            self.subspace_means = mean.contiguous()
            self.subspace_bases = (features.new_zeros((features.shape[1], 0)),)
            self.singular_values = (features.new_zeros((0,)),)
            self.selected_ranks = torch.zeros(1, dtype=torch.long)
            self.cluster_sizes = torch.tensor([count], dtype=torch.long)
            self.zero_variance_clusters = torch.tensor([is_zero_variance], dtype=torch.bool)
            self.retained_energy = torch.zeros(1, dtype=torch.float32)
            distance = torch.linalg.vector_norm(features - mean, dim=1)
            self.representative_background_indices = torch.tensor(
                [int(distance.argmin())], dtype=torch.long
            )
        else:
            self._fit_subspaces(features, assignments)

        self.cluster_assignments = assignments.contiguous()
        self.cluster_centers = self._centers(features, assignments, effective_count)
        self.num_background_atoms = count
        self.feature_dim = int(features.shape[1])
        self.num_effective_subspaces = effective_count
        self.fallback_flags = fallback
        self.clustering_diagnostics = {**merge_meta, **kmeans_meta}
        self._is_fitted = True
        return self

    def score(self, query_features: torch.Tensor) -> MBSPScore:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before score")
        query = _require_matrix(query_features, "query_features").cpu()
        if int(query.shape[1]) != self.feature_dim:
            raise ValueError(
                f"query feature dimension {query.shape[1]} != fitted dimension {self.feature_dim}"
            )
        start = time.perf_counter()
        relative = []
        absolute = []
        for mean, basis in zip(self.subspace_means, self.subspace_bases):
            centered = query - mean
            denominator = centered.square().sum(dim=1)
            if basis.shape[1] == 0:
                residual_squared = denominator
            else:
                coefficient = centered @ basis
                residual_squared = (centered - coefficient @ basis.t()).square().sum(dim=1)
            absolute.append(residual_squared.clamp_min(0.0))
            relative.append((residual_squared / (denominator + self.eps)).clamp(0.0, 1.0))

        per_relative = torch.stack(relative, dim=1).contiguous()
        per_absolute = torch.stack(absolute, dim=1).contiguous()
        relative_min, best = per_relative.min(dim=1)
        absolute_min = per_absolute.min(dim=1).values
        self.timing["score_seconds"] = time.perf_counter() - start
        return MBSPScore(
            relative_residual=relative_min.contiguous(),
            absolute_residual=absolute_min.contiguous(),
            best_subspace_index=best.to(dtype=torch.long).contiguous(),
            per_subspace_relative=per_relative,
            per_subspace_absolute=per_absolute,
        )

    def diagnostics(self) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("fit must be called before diagnostics")
        return {
            "num_background_atoms": self.num_background_atoms,
            "num_requested_subspaces": self.num_subspaces,
            "num_effective_subspaces": self.num_effective_subspaces,
            "cluster_sizes": self.cluster_sizes.tolist(),
            "selected_ranks": self.selected_ranks.tolist(),
            "zero_variance_clusters": self.zero_variance_clusters.tolist(),
            "retained_energy": self.retained_energy.tolist(),
            "representative_background_indices": self.representative_background_indices.tolist(),
            "fallback_flags": dict(self.fallback_flags),
            "clustering": dict(self.clustering_diagnostics),
            "timing": dict(self.timing),
        }
