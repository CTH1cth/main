from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MaxBackgroundResult:
    score: np.ndarray
    matched_index: np.ndarray
    num_background: int
    self_match_violations: int


def max_valid_background_similarity(
    feature: torch.Tensor,
    label: np.ndarray,
    valid: np.ndarray,
) -> MaxBackgroundResult:
    """Return one maximum valid-background cosine score per valid query.

    ``feature`` must already be L2 normalized.  A background query is forbidden
    from matching itself. Mixed/invalid queries are stored as NaN/-1.
    """
    n = int(feature.shape[0])
    label = np.asarray(label, dtype=np.uint8).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if feature.ndim != 2 or label.shape != (n,) or valid.shape != (n,):
        raise ValueError("feature/label/valid shape mismatch")
    background = np.flatnonzero(valid & (label == 0))
    query = np.flatnonzero(valid)
    if background.size < 2:
        raise ValueError("at least two valid background patches are required")
    similarity = feature[torch.as_tensor(query, device=feature.device)] @ feature[
        torch.as_tensor(background, device=feature.device)
    ].T

    # Remove BG self matches. query is sorted, so searchsorted gives the row.
    query_row = np.searchsorted(query, background)
    if not np.array_equal(query[query_row], background):
        raise AssertionError("background queries are missing from valid query set")
    similarity[
        torch.as_tensor(query_row, device=feature.device),
        torch.arange(background.size, device=feature.device),
    ] = -torch.inf
    maximum, local_index = similarity.max(dim=1)
    matched = background[local_index.detach().cpu().numpy()]

    score = np.full(n, np.nan, dtype=np.float32)
    matched_index = np.full(n, -1, dtype=np.int32)
    score[query] = maximum.detach().cpu().float().numpy()
    matched_index[query] = matched.astype(np.int32, copy=False)
    bg_query = valid & (label == 0)
    violations = int(np.count_nonzero(matched_index[bg_query] == np.flatnonzero(bg_query)))
    if violations:
        raise AssertionError(f"BG self-match exclusion failed: {violations}")
    if not np.isfinite(score[valid]).all():
        raise RuntimeError("non-finite maximum similarity")
    return MaxBackgroundResult(score, matched_index, int(background.size), violations)


def brute_force_query(
    feature: torch.Tensor,
    background: np.ndarray,
    query_index: int,
) -> tuple[float, int]:
    """Independent small-query reference used only by correctness audits."""
    background = np.asarray(background, dtype=np.int64)
    candidates = background[background != int(query_index)]
    if candidates.size == 0:
        raise ValueError("empty brute-force background candidate set")
    query = feature[int(query_index)].unsqueeze(0)
    refs = feature[torch.as_tensor(candidates, device=feature.device)]
    values = torch.nn.functional.cosine_similarity(query, refs, dim=1, eps=1e-12)
    local = int(values.argmax().item())
    return float(values[local].item()), int(candidates[local])
