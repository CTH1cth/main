from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sample_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    lines = set()
    aliases = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) >= 2:
            raw_dataset, stem = parts[0], parts[1]
            dataset = aliases.get(raw_dataset, raw_dataset)
            lines.add(f"{dataset}/{stem}")
            lines.add(f"{raw_dataset}/{stem}")
        else:
            lines.add(raw)
    return lines or None


def sample_selected(row: dict, ids: set[str] | None) -> bool:
    if ids is None:
        return True
    stem, dataset = row["stem"], row["dataset"]
    aliases = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K", "CAMO": "TE-CAMO", "COD10K": "TE-COD10K"}
    candidates = {stem, f"{dataset}/{stem}", f"{dataset}:{stem}"}
    if dataset in aliases:
        candidates.add(f"{aliases[dataset]}/{stem}")
    return bool(candidates & ids)


def patch_gt(gt_path: str, grid: int = 37) -> np.ndarray:
    gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(gt)[None, None]
    return F.interpolate(t, size=(grid, grid), mode="area")[0, 0].numpy().reshape(-1)


def labels_from_area(area: np.ndarray, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if strict:
        valid = (area <= 0.2) | (area >= 0.8)
        return (area >= 0.8).astype(np.uint8), valid
    return (area >= 0.5).astype(np.uint8), np.ones(area.shape, dtype=bool)


def precision_at_recall(y: np.ndarray, score: np.ndarray, targets=(0.5, 0.6, 0.7)) -> dict:
    precision, recall, _ = precision_recall_curve(y, score)
    return {f"p_at_r{int(t * 100)}": float(precision[recall >= t].max()) for t in targets}


def binary_metrics(y: np.ndarray, score: np.ndarray, targets=(0.5, 0.6, 0.7)) -> dict:
    y = np.asarray(y).astype(np.uint8)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2 or not np.isfinite(score).all():
        return {"ap": np.nan, "auroc": np.nan, **{f"p_at_r{int(t*100)}": np.nan for t in targets}}
    return {
        "ap": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        **precision_at_recall(y, score, targets),
    }


def minmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return np.zeros_like(x) if span <= 1e-12 else (x - x.min()) / span


def quantile_mask(similarity: np.ndarray, quantile: float) -> np.ndarray:
    # This function intentionally accepts no GT.
    return similarity >= np.quantile(similarity, quantile)


def equal_frequency_bins(similarity: np.ndarray, n_bins: int = 5) -> np.ndarray:
    order = np.argsort(similarity, kind="mergesort")
    bins = np.empty(similarity.size, dtype=np.int64)
    for b, idx in enumerate(np.array_split(order, n_bins), 1):
        bins[idx] = b
    return bins


def similarity_matched_pairs(
    similarity: np.ndarray, labels: np.ndarray, valid: np.ndarray, tolerance: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    fg = np.flatnonzero(valid & (labels == 1))
    bg = np.flatnonzero(valid & (labels == 0))
    if not len(fg) or not len(bg):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    bg_order = bg[np.argsort(similarity[bg])]
    bg_values = similarity[bg_order]
    positions = np.searchsorted(bg_values, similarity[fg])
    selected, kept_fg = [], []
    for f, p in zip(fg, positions):
        candidates = [min(max(int(p), 0), len(bg_order) - 1), min(max(int(p) - 1, 0), len(bg_order) - 1)]
        j = min(candidates, key=lambda z: abs(float(bg_values[z] - similarity[f])))
        if abs(float(bg_values[j] - similarity[f])) <= tolerance:
            kept_fg.append(f); selected.append(bg_order[j])
    return np.asarray(kept_fg, dtype=np.int64), np.asarray(selected, dtype=np.int64)
