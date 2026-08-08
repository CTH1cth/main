"""GT-free cross-fitted calibration for single-subspace GBSP residuals.

This module contains only deterministic per-image statistics.  It never reads
ground truth, R1 masks, target areas, or dataset-specific thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.mbsp_reconstruction import MultiBackgroundSubspaceProjector


@dataclass(frozen=True)
class CrossFittedResidualResult:
    fold_ids: torch.Tensor
    fold_ranks: torch.Tensor
    fold_mu: torch.Tensor
    fold_scale_median: torch.Tensor
    fold_scale_mad: torch.Tensor
    fold_scale_method: tuple[str, ...]
    fold_basis_orthonormal_max_error: torch.Tensor
    fold_fit_background_positions: tuple[torch.Tensor, ...]
    fold_heldout_background_positions: tuple[torch.Tensor, ...]
    query_raw_residual_each_fold: torch.Tensor
    query_z_each_fold: torch.Tensor
    query_raw_median: torch.Tensor
    query_z_median: torch.Tensor
    background_oof_raw_residual: torch.Tensor
    background_oof_z: torch.Tensor
    numerical_fallback: bool
    numerical_fallback_reasons: tuple[str, ...]


@dataclass(frozen=True)
class BackgroundTailResult:
    p_value: torch.Tensor
    anomaly_score: torch.Tensor
    monotonicity_violation_count: int


@dataclass(frozen=True)
class ThresholdResult:
    binary_mask: torch.Tensor
    threshold: float
    fallback: bool = False
    k_star: int = -1
    sorted_p: torch.Tensor | None = None
    score_curve: torch.Tensor | None = None
    candidate_mask: torch.Tensor | None = None
    selected_count: int = 0
    boundary_tie_count: int = 0


@dataclass(frozen=True)
class BackgroundTailShrinkageResult:
    """Auditable BTS threshold after mapping the background tail to Min-Max."""

    binary_mask: torch.Tensor
    threshold: float
    background_z_quantile: float
    fold_raw_thresholds: torch.Tensor
    background_raw_threshold: float
    background_minmax_threshold_unclipped: float
    background_minmax_threshold: float
    minmax_mapping_clipped: bool
    selected_count: int
    boundary_tie_count: int


def deterministic_spatial_folds(
    background_indices: torch.Tensor,
    grid: int = 37,
    num_folds: int = 5,
) -> torch.Tensor:
    """Assign spatially interleaved, exactly balanced deterministic folds.

    ``(row + 2 * col) % K`` is used first.  Only the minimum number of items
    needed to make fold sizes differ by at most one is deterministically moved.
    """

    if not torch.is_tensor(background_indices) or background_indices.ndim != 1:
        raise ValueError("background_indices must be a 1-D tensor")
    indices = background_indices.detach().cpu().long().contiguous()
    if indices.numel() < int(num_folds):
        raise ValueError(
            f"at least {num_folds} background candidates are required, got {indices.numel()}"
        )
    if int(grid) < 1 or int(num_folds) < 2:
        raise ValueError("grid must be positive and num_folds must be >= 2")
    if int(indices.min()) < 0 or int(indices.max()) >= int(grid) * int(grid):
        raise ValueError("background index lies outside the patch grid")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("background_indices must be unique")

    rows = torch.div(indices, int(grid), rounding_mode="floor")
    cols = indices.remainder(int(grid))
    fold_ids = (rows + 2 * cols).remainder(int(num_folds)).long()
    counts = torch.bincount(fold_ids, minlength=int(num_folds))
    base, remainder = divmod(int(indices.numel()), int(num_folds))

    # Give the larger target sizes to folds that already contain the most
    # formula-assigned samples.  This minimizes deterministic reassignments.
    target = torch.full((int(num_folds),), base, dtype=torch.long)
    preference = sorted(range(int(num_folds)), key=lambda k: (-int(counts[k]), k))
    for fold in preference[:remainder]:
        target[fold] += 1

    deficits = []
    for fold in range(int(num_folds)):
        deficits.extend([fold] * max(0, int(target[fold] - counts[fold])))
    deficit_cursor = 0
    for source in range(int(num_folds)):
        surplus = max(0, int(counts[source] - target[source]))
        if surplus == 0:
            continue
        positions = torch.where(fold_ids == source)[0]
        # Linear patch order is stable and independent of input enumeration.
        ordered = positions[torch.argsort(indices.index_select(0, positions), stable=True)]
        for position in ordered[-surplus:].tolist():
            fold_ids[position] = deficits[deficit_cursor]
            deficit_cursor += 1
    if deficit_cursor != len(deficits):
        raise RuntimeError("deterministic fold balancing did not close all deficits")
    final_counts = torch.bincount(fold_ids, minlength=int(num_folds))
    if int(final_counts.max() - final_counts.min()) > 1 or bool((final_counts == 0).any()):
        raise RuntimeError(f"invalid balanced fold sizes: {final_counts.tolist()}")
    return fold_ids.contiguous()


def _robust_log_location_scale(
    residual: torch.Tensor,
    eps: float,
    scale_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, str, bool]:
    if residual.ndim != 1 or residual.numel() == 0:
        raise ValueError("held-out residual must be a non-empty vector")
    log_residual = torch.log(residual.clamp_min(0.0) + float(eps))
    location = torch.median(log_residual)
    scale = 1.4826 * torch.median(torch.abs(log_residual - location))
    method = "mad"
    numerical_fallback = False
    if float(scale) < float(scale_floor):
        q1 = torch.quantile(log_residual, 0.25)
        q3 = torch.quantile(log_residual, 0.75)
        scale = (q3 - q1) / 1.349
        method = "iqr"
    if float(scale) < float(scale_floor):
        scale = log_residual.std(unbiased=False) + float(scale_floor)
        method = "std_plus_floor"
        numerical_fallback = True
    if not bool(torch.isfinite(location)) or not bool(torch.isfinite(scale)):
        raise RuntimeError("non-finite robust background scale")
    return location, scale, method, numerical_fallback


class CrossFittedBackgroundResidual:
    """Five-fold out-of-fold background residual and query calibration."""

    def __init__(
        self,
        num_folds: int = 5,
        grid: int = 37,
        pca_energy: float = 0.90,
        pca_max_rank: int = 8,
        pca_min_rank: int = 1,
        min_cluster_size: int = 16,
        eps: float = 1e-12,
        scale_floor: float = 1e-6,
    ) -> None:
        if int(num_folds) < 2:
            raise ValueError("num_folds must be >= 2")
        if int(pca_max_rank) < 1 or not 0 <= int(pca_min_rank) <= int(pca_max_rank):
            raise ValueError("cross-fitted PCA ranks are invalid")
        if float(eps) <= 0 or float(scale_floor) <= 0:
            raise ValueError("eps and scale_floor must be positive")
        self.num_folds = int(num_folds)
        self.grid = int(grid)
        self.pca_energy = float(pca_energy)
        self.pca_max_rank = int(pca_max_rank)
        self.pca_min_rank = int(pca_min_rank)
        self.min_cluster_size = int(min_cluster_size)
        self.eps = float(eps)
        self.scale_floor = float(scale_floor)

    @staticmethod
    def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim != 2 or value.numel() == 0:
            raise ValueError(f"{name} must be a non-empty Tensor[N,D]")
        value = value.detach().cpu().float().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        if bool((torch.linalg.vector_norm(value, dim=1) <= 0).any()):
            raise ValueError(f"{name} contains a zero vector")
        return F.normalize(value, p=2, dim=1)

    def fit_score(
        self,
        background_features: torch.Tensor,
        background_indices: torch.Tensor,
        query_features: torch.Tensor,
    ) -> CrossFittedResidualResult:
        background = self._matrix(background_features, "background_features")
        query = self._matrix(query_features, "query_features")
        if background.shape[1] != query.shape[1]:
            raise ValueError("background/query feature dimensions differ")
        if int(background.shape[0]) != int(background_indices.numel()):
            raise ValueError("background feature/index counts differ")

        fold_ids = deterministic_spatial_folds(
            background_indices, grid=self.grid, num_folds=self.num_folds
        )
        background_count = int(background.shape[0])
        query_count = int(query.shape[0])
        oof_raw = torch.empty(background_count, dtype=torch.float32)
        oof_z = torch.empty(background_count, dtype=torch.float32)
        query_raw_folds = []
        query_z_folds = []
        ranks = []
        means = []
        locations = []
        scales = []
        scale_methods = []
        gram_errors = []
        fit_positions_all = []
        heldout_positions_all = []
        fallback_reasons = []

        all_positions = torch.arange(background_count, dtype=torch.long)
        for fold in range(self.num_folds):
            heldout = torch.where(fold_ids == fold)[0]
            fit_positions = torch.where(fold_ids != fold)[0]
            if heldout.numel() == 0 or fit_positions.numel() < 2:
                raise RuntimeError(
                    f"fold {fold} has invalid fit/held-out sizes: "
                    f"{fit_positions.numel()}/{heldout.numel()}"
                )
            # Explicit no-leakage assertion required by the protocol.
            if bool(torch.isin(heldout, fit_positions).any()):
                raise RuntimeError(f"fold {fold} leaks held-out background candidates")
            if not torch.equal(
                torch.sort(torch.cat((heldout, fit_positions))).values, all_positions
            ):
                raise RuntimeError(f"fold {fold} does not partition the background set")

            projector = MultiBackgroundSubspaceProjector(
                num_subspaces=1,
                min_cluster_size=self.min_cluster_size,
                pca_energy=self.pca_energy,
                pca_max_rank=self.pca_max_rank,
                pca_min_rank=self.pca_min_rank,
                seed=0,
                eps=self.eps,
                kmeans_n_init=1,
                kmeans_max_iter=1,
            ).fit(background.index_select(0, fit_positions))
            basis = projector.subspace_bases[0]
            gram_error = 0.0
            if basis.shape[1]:
                identity = torch.eye(basis.shape[1], dtype=basis.dtype)
                gram_error = float((basis.t() @ basis - identity).abs().max())
            if gram_error >= 1e-4:
                raise RuntimeError(f"fold {fold} PCA basis error={gram_error}")

            heldout_raw = projector.score(background.index_select(0, heldout)).absolute_residual
            query_raw = projector.score(query).absolute_residual
            location, scale, method, numerical = _robust_log_location_scale(
                heldout_raw, self.eps, self.scale_floor
            )
            heldout_z = (torch.log(heldout_raw + self.eps) - location) / (
                scale + self.eps
            )
            query_z = (torch.log(query_raw + self.eps) - location) / (scale + self.eps)
            if not bool(torch.isfinite(heldout_z).all()) or not bool(
                torch.isfinite(query_z).all()
            ):
                raise RuntimeError(f"fold {fold} produced non-finite standardized residuals")

            oof_raw.index_copy_(0, heldout, heldout_raw)
            oof_z.index_copy_(0, heldout, heldout_z)
            query_raw_folds.append(query_raw)
            query_z_folds.append(query_z)
            ranks.append(int(projector.selected_ranks[0]))
            means.append(projector.subspace_means[0])
            locations.append(location)
            scales.append(scale)
            scale_methods.append(method)
            gram_errors.append(gram_error)
            fit_positions_all.append(fit_positions.contiguous())
            heldout_positions_all.append(heldout.contiguous())
            if numerical:
                fallback_reasons.append(f"fold_{fold}:std_plus_floor")

        query_raw_each = torch.stack(query_raw_folds, dim=0).contiguous()
        query_z_each = torch.stack(query_z_folds, dim=0).contiguous()
        if tuple(query_raw_each.shape) != (self.num_folds, query_count):
            raise RuntimeError("unexpected cross-fitted query residual shape")
        return CrossFittedResidualResult(
            fold_ids=fold_ids,
            fold_ranks=torch.tensor(ranks, dtype=torch.long),
            fold_mu=torch.stack(means, dim=0).float().contiguous(),
            fold_scale_median=torch.stack(locations).float().contiguous(),
            fold_scale_mad=torch.stack(scales).float().contiguous(),
            fold_scale_method=tuple(scale_methods),
            fold_basis_orthonormal_max_error=torch.tensor(gram_errors),
            fold_fit_background_positions=tuple(fit_positions_all),
            fold_heldout_background_positions=tuple(heldout_positions_all),
            query_raw_residual_each_fold=query_raw_each,
            query_z_each_fold=query_z_each,
            query_raw_median=torch.median(query_raw_each, dim=0).values.contiguous(),
            query_z_median=torch.median(query_z_each, dim=0).values.contiguous(),
            background_oof_raw_residual=oof_raw.contiguous(),
            background_oof_z=oof_z.contiguous(),
            numerical_fallback=bool(fallback_reasons),
            numerical_fallback_reasons=tuple(fallback_reasons),
        )


class BackgroundTailCalibrator:
    """Convert query robust z-scores to empirical background upper-tail p-values."""

    def __init__(self, eps: float = 1e-12) -> None:
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def calibrate(
        self, query_z: torch.Tensor, background_oof_z: torch.Tensor
    ) -> BackgroundTailResult:
        query = query_z.detach().cpu().float().reshape(-1).contiguous()
        background = background_oof_z.detach().cpu().float().reshape(-1).contiguous()
        if query.numel() == 0 or background.numel() == 0:
            raise ValueError("query/background z-score vectors must be non-empty")
        if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(background).all()):
            raise ValueError("query/background z-scores must be finite")
        sorted_background = torch.sort(background, stable=True).values
        first_ge = torch.searchsorted(sorted_background, query, right=False)
        count_ge = int(background.numel()) - first_ge
        p_value = (1.0 + count_ge.float()) / (int(background.numel()) + 1.0)
        p_value = p_value.clamp(1.0 / (int(background.numel()) + 1.0), 1.0)
        anomaly = -torch.log10(p_value + self.eps)
        order = torch.argsort(query, descending=False, stable=True)
        ordered_p = p_value.index_select(0, order)
        # As Z increases, p must not increase.
        violations = int((ordered_p[1:] > ordered_p[:-1] + 1e-7).sum())
        if violations:
            raise RuntimeError(f"background-tail p-value monotonicity violations={violations}")
        return BackgroundTailResult(
            p_value=p_value.contiguous(),
            anomaly_score=anomaly.contiguous(),
            monotonicity_violation_count=violations,
        )


class HigherCriticismThreshold:
    """Predeclared HC threshold on empirical background-tail p-values."""

    def __init__(self, p_max: float = 0.20, fallback_p: float = 0.05, eps: float = 1e-12):
        if not 0 < float(fallback_p) <= float(p_max) < 1:
            raise ValueError("require 0 < fallback_p <= p_max < 1")
        self.p_max = float(p_max)
        self.fallback_p = float(fallback_p)
        self.eps = float(eps)

    def apply(self, p_value: torch.Tensor, background_count: int) -> ThresholdResult:
        p = p_value.detach().cpu().float().reshape(-1).contiguous()
        if p.numel() == 0 or not bool(torch.isfinite(p).all()):
            raise ValueError("p_value must be non-empty and finite")
        if float(p.min()) <= 0 or float(p.max()) > 1:
            raise ValueError("p_value must lie in (0,1]")
        if int(background_count) < 1:
            raise ValueError("background_count must be positive")
        sorted_p = torch.sort(p, stable=True).values
        positions = torch.arange(1, p.numel() + 1, dtype=torch.float32)
        lower = 1.0 / (int(background_count) + 1.0)
        candidate = (sorted_p >= lower - 1e-12) & (sorted_p <= self.p_max)
        curve = torch.zeros_like(sorted_p)
        if bool(candidate.any()):
            selected_p = sorted_p[candidate]
            selected_k = positions[candidate]
            curve[candidate] = (p.numel() ** 0.5) * (
                selected_k / p.numel() - selected_p
            ) / torch.sqrt(selected_p * (1.0 - selected_p) + self.eps)
            candidate_positions = torch.where(candidate)[0]
            winner_position = int(candidate_positions[int(torch.argmax(curve[candidate]))])
            threshold = float(sorted_p[winner_position])
            k_star = winner_position + 1
            fallback = False
        else:
            threshold = self.fallback_p
            k_star = -1
            fallback = True
        mask = p <= threshold
        tie_count = int((p == threshold).sum())
        return ThresholdResult(
            binary_mask=mask.contiguous(),
            threshold=threshold,
            fallback=fallback,
            k_star=k_star,
            sorted_p=sorted_p,
            score_curve=curve,
            candidate_mask=candidate,
            selected_count=int(mask.sum()),
            boundary_tie_count=tie_count,
        )


class BackgroundQuantileThreshold:
    """CF-BQ95: fixed empirical background upper-tail probability 0.05."""

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < float(alpha) < 1:
            raise ValueError("alpha must lie in (0,1)")
        self.alpha = float(alpha)

    def apply(self, p_value: torch.Tensor) -> ThresholdResult:
        p = p_value.detach().cpu().float()
        mask = p <= self.alpha
        return ThresholdResult(
            binary_mask=mask.contiguous(),
            threshold=self.alpha,
            selected_count=int(mask.sum()),
            boundary_tie_count=int((p == self.alpha).sum()),
        )


class RobustMADThreshold:
    """CF-RMAD: fixed robust standardized residual threshold."""

    def __init__(self, kappa: float = 3.0) -> None:
        if float(kappa) <= 0:
            raise ValueError("kappa must be positive")
        self.kappa = float(kappa)

    def apply(self, query_z: torch.Tensor) -> ThresholdResult:
        z = query_z.detach().cpu().float()
        mask = z > self.kappa
        return ThresholdResult(
            binary_mask=mask.contiguous(),
            threshold=self.kappa,
            selected_count=int(mask.sum()),
            boundary_tie_count=int((z == self.kappa).sum()),
        )


class BackgroundTailShrinkageThreshold:
    """BTS: shrink an image-specific OOF background tail toward fixed 0.58.

    The task specification defines the background tail in robust-z space but
    the stable global work point in image-wise Min-Max space.  Directly adding
    those quantities is dimensionally invalid.  We therefore invert the
    per-fold log-MAD transforms, median-fuse the resulting raw thresholds, and
    map that raw work point through the same image-wise Min-Max transform used
    by the fixed baseline.  Shrinkage then happens entirely in Min-Max space.
    """

    def __init__(
        self,
        background_quantile: float = 0.95,
        global_minmax_threshold: float = 0.58,
        shrinkage: float = 0.5,
        eps: float = 1e-12,
    ) -> None:
        if not 0.0 < float(background_quantile) < 1.0:
            raise ValueError("background_quantile must lie in (0,1)")
        if not 0.0 <= float(global_minmax_threshold) <= 1.0:
            raise ValueError("global_minmax_threshold must lie in [0,1]")
        if not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0,1]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        self.background_quantile = float(background_quantile)
        self.global_minmax_threshold = float(global_minmax_threshold)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)

    @staticmethod
    def _finite_vector(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.numel() == 0:
            raise ValueError(f"{name} must be a non-empty tensor")
        value = value.detach().cpu().float().reshape(-1).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN/Inf")
        return value

    def apply(
        self,
        background_oof_z: torch.Tensor,
        fold_log_median: torch.Tensor,
        fold_log_scale: torch.Tensor,
        gbsp_raw: torch.Tensor,
        gbsp_minmax: torch.Tensor,
    ) -> BackgroundTailShrinkageResult:
        background_z = self._finite_vector(background_oof_z, "background_oof_z")
        locations = self._finite_vector(fold_log_median, "fold_log_median")
        scales = self._finite_vector(fold_log_scale, "fold_log_scale")
        raw = self._finite_vector(gbsp_raw, "gbsp_raw")
        minmax = self._finite_vector(gbsp_minmax, "gbsp_minmax")
        if locations.numel() != scales.numel() or locations.numel() < 2:
            raise ValueError("fold location/scale counts must match and be >= 2")
        if raw.numel() != minmax.numel():
            raise ValueError("GBSP raw/Min-Max shapes differ")
        if bool((scales <= 0.0).any()):
            raise ValueError("fold log scales must be positive")
        if float(minmax.min()) < -1e-6 or float(minmax.max()) > 1.0 + 1e-6:
            raise ValueError("gbsp_minmax must lie in [0,1]")

        background_z_threshold = torch.quantile(
            background_z, self.background_quantile
        )
        fold_raw_thresholds = torch.exp(
            locations + scales * background_z_threshold
        ) - self.eps
        if not bool(torch.isfinite(fold_raw_thresholds).all()):
            raise RuntimeError("BTS inverse log-MAD mapping produced NaN/Inf")
        background_raw_threshold = torch.median(fold_raw_thresholds)
        raw_min, raw_max = raw.min(), raw.max()
        raw_range = raw_max - raw_min
        if float(raw_range) <= self.eps:
            raise RuntimeError("GBSP raw residual has a degenerate Min-Max range")
        background_mm_unclipped = (background_raw_threshold - raw_min) / raw_range
        background_mm = background_mm_unclipped.clamp(0.0, 1.0)
        threshold = (
            (1.0 - self.shrinkage) * self.global_minmax_threshold
            + self.shrinkage * float(background_mm)
        )
        mask = minmax > threshold
        return BackgroundTailShrinkageResult(
            binary_mask=mask.contiguous(),
            threshold=float(threshold),
            background_z_quantile=float(background_z_threshold),
            fold_raw_thresholds=fold_raw_thresholds.contiguous(),
            background_raw_threshold=float(background_raw_threshold),
            background_minmax_threshold_unclipped=float(background_mm_unclipped),
            background_minmax_threshold=float(background_mm),
            minmax_mapping_clipped=bool(
                float(background_mm_unclipped) < 0.0
                or float(background_mm_unclipped) > 1.0
            ),
            selected_count=int(mask.sum()),
            boundary_tie_count=int((minmax == threshold).sum()),
        )
