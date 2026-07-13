import hashlib
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


VARIANT_DIMS = {
    "V0": 7,
    "V1": 1,
    "V2": 384,
    "V3": 43,
    "V4": 427,
    "V5": 433,
    "V6": 433,
}


def stable_hash_int(*parts):
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def stable_hash_float(*parts):
    return stable_hash_int(*parts) / float(16**16 - 1)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def stratified_image_split(items, seed=3407):
    by_dataset = {}
    for item in items:
        by_dataset.setdefault(item["dataset"], []).append(item)
    manifest = []
    for dataset, rows in sorted(by_dataset.items()):
        rows = list(rows)
        rows.sort(key=lambda row: row["stem"])
        rng = random.Random(int(seed) + stable_hash_int(dataset))
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for idx, row in enumerate(rows):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            manifest.append({"dataset": row["dataset"], "stem": row["stem"], "split": split})
    manifest.sort(key=lambda row: (row["dataset"], row["stem"]))
    return manifest


def build_split_lookup(split_manifest):
    return {(row["dataset"], row["stem"]): row["split"] for row in split_manifest}


def select_stratified_indices(dataset, max_images_per_source=-1, seed=3407):
    wanted = {"TR-CAMO", "TR-COD10K"}
    by_dataset = {name: [] for name in wanted}
    for idx, item in enumerate(dataset.items):
        name = item["dataset"]
        if name in by_dataset:
            by_dataset[name].append((idx, item))
    selected = []
    for name in sorted(by_dataset):
        rows = by_dataset[name]
        if not rows:
            raise RuntimeError(f"No training items found for {name}.")
        rows = sorted(rows, key=lambda pair: pair[1]["stem"])
        rng = random.Random(int(seed) + stable_hash_int("select", name))
        rng.shuffle(rows)
        if int(max_images_per_source) > 0:
            rows = rows[: int(max_images_per_source)]
        selected.extend(rows)
    selected.sort(key=lambda pair: (pair[1]["dataset"], pair[1]["stem"]))
    return selected


def stable_take_indices(mask, dataset, stem, seed, max_count):
    coords = np.flatnonzero(mask.reshape(-1))
    if coords.size == 0:
        return coords.astype(np.int64)
    order = np.array(
        [stable_hash_float(dataset, stem, int(seed), int(coord)) for coord in coords],
        dtype=np.float64,
    )
    coords = coords[np.argsort(order)]
    if max_count is not None and int(max_count) > 0:
        coords = coords[: int(max_count)]
    return coords.astype(np.int64)


def block_support_query_masks(dataset, stem, height, width, seed, block_size=4):
    block_h = int(math.ceil(float(height) / float(block_size)))
    support = np.zeros((height, width), dtype=bool)
    for y in range(height):
        by = y // int(block_size)
        for x in range(width):
            bx = x // int(block_size)
            block_id = by * block_h + bx
            support[y, x] = stable_hash_float(dataset, stem, block_id, seed) < 0.5
    return support, ~support


def build_episode_masks(fg_core, bg_core, dataset, stem, seed):
    height, width = fg_core.shape
    support_blocks, query_blocks = block_support_query_masks(dataset, stem, height, width, seed)
    fg_support = fg_core & support_blocks
    bg_support = bg_core & support_blocks & (~fg_support)
    fg_query = fg_core & query_blocks
    bg_query = bg_core & query_blocks & (~fg_query)
    reason = []
    if int(fg_support.sum()) < 8:
        reason.append("fg_support_lt8")
    if int(bg_support.sum()) < 16:
        reason.append("bg_support_lt16")
    if int(fg_query.sum()) < 8:
        reason.append("fg_query_lt8")
    if int(bg_query.sum()) < 16:
        reason.append("bg_query_lt16")
    if reason:
        return None, ",".join(reason)
    fg_support_idx = stable_take_indices(fg_support, dataset, stem, seed, 128)
    bg_support_idx = stable_take_indices(bg_support, dataset, stem, seed, 256)
    fg_query_idx_all = stable_take_indices(fg_query, dataset, stem, seed + 100000, None)
    bg_query_idx_all = stable_take_indices(bg_query, dataset, stem, seed + 200000, None)
    n_query = min(64, int(fg_query_idx_all.size), int(bg_query_idx_all.size))
    if n_query < 8:
        return None, "balanced_query_lt8"
    fg_query_idx = fg_query_idx_all[:n_query]
    bg_query_idx = bg_query_idx_all[:n_query]
    return {
        "fg_support_idx": fg_support_idx,
        "bg_support_idx": bg_support_idx,
        "fg_query_idx": fg_query_idx,
        "bg_query_idx": bg_query_idx,
        "query_idx": np.concatenate([fg_query_idx, bg_query_idx]).astype(np.int64),
        "labels": np.concatenate(
            [np.ones(n_query, dtype=np.float32), np.zeros(n_query, dtype=np.float32)]
        ),
        "query_mask": np.isin(
            np.arange(height * width, dtype=np.int64),
            np.concatenate([fg_query_idx, bg_query_idx]),
        ).reshape(height, width),
    }, ""


def normalize_channels(arr):
    norm = np.linalg.norm(arr, axis=0, keepdims=True)
    return arr / np.maximum(norm, 1e-8)


def prototype_features(feature_chw, fg_idx, bg_idx, query_idx):
    channels = feature_chw.shape[0]
    flat = feature_chw.reshape(channels, -1)
    fg_proto = flat[:, fg_idx].mean(axis=1)
    bg_proto = flat[:, bg_idx].mean(axis=1)
    fg_proto = fg_proto / max(float(np.linalg.norm(fg_proto)), 1e-8)
    bg_proto = bg_proto / max(float(np.linalg.norm(bg_proto)), 1e-8)
    query = flat[:, query_idx].T
    sim_fg = query @ fg_proto
    sim_bg = query @ bg_proto
    return np.stack([sim_fg, sim_bg, sim_fg - sim_bg], axis=1).astype(np.float32)


def self_support_indices(final_prob_hw, query_mask, dataset, stem, seed):
    non_query = ~query_mask
    flat_prob = final_prob_hw.reshape(-1)
    flat_non_query = non_query.reshape(-1)
    fg_mask = (flat_prob >= 0.80) & flat_non_query
    bg_mask = (flat_prob <= 0.20) & flat_non_query
    fg_fallback = False
    bg_fallback = False
    fg_idx = np.flatnonzero(fg_mask)
    bg_idx = np.flatnonzero(bg_mask)
    if fg_idx.size < 8:
        pool = np.flatnonzero(flat_non_query)
        if pool.size < 8:
            return None, None, True, True, "self_fg_pool_lt8"
        order = np.lexsort(
            (
                np.array([stable_hash_float(dataset, stem, seed, "fg", int(i)) for i in pool]),
                -flat_prob[pool],
            )
        )
        fg_idx = pool[order[:8]]
        fg_fallback = True
    else:
        fg_idx = stable_take_indices(fg_mask.reshape(final_prob_hw.shape), dataset, stem, seed, 128)
    if bg_idx.size < 32:
        pool = np.flatnonzero(flat_non_query)
        if pool.size < 32:
            return None, None, fg_fallback, True, "self_bg_pool_lt32"
        order = np.lexsort(
            (
                np.array([stable_hash_float(dataset, stem, seed, "bg", int(i)) for i in pool]),
                flat_prob[pool],
            )
        )
        bg_idx = pool[order[:32]]
        bg_fallback = True
    else:
        bg_idx = stable_take_indices(bg_mask.reshape(final_prob_hw.shape), dataset, stem, seed + 1, 256)
    return fg_idx.astype(np.int64), bg_idx.astype(np.int64), fg_fallback, bg_fallback, ""


def variant_columns(variant):
    variant = str(variant).upper()
    v0 = list(range(0, 7))
    v1 = list(range(7, 8))
    v2 = list(range(8, 392))
    v3 = list(range(392, 435))
    dabe = list(range(435, 441))
    self_sup = list(range(441, 447))
    if variant == "V0":
        return v0
    if variant == "V1":
        return v1
    if variant == "V2":
        return v2
    if variant == "V3":
        return v3
    if variant == "V4":
        return v2 + v3
    if variant == "V5":
        return v2 + v3 + dabe
    if variant == "V6":
        return v2 + v3 + self_sup
    raise ValueError(f"Unsupported ICAR variant: {variant}")


def make_probe(probe_type, input_dim):
    probe_type = str(probe_type).lower()
    if probe_type == "linear":
        return nn.Linear(int(input_dim), 1)
    if probe_type in {"mlp", "tinymlp"}:
        return nn.Sequential(
            nn.Linear(int(input_dim), 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
    raise ValueError(f"Unsupported probe type: {probe_type}")


def standardize_fit(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def standardize_apply(x, mean, std):
    return (x.astype(np.float32) - mean[None, :]) / std[None, :]


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))).astype(np.float64)


def binary_metrics_from_scores(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    fg_recall = tp / max(tp + fn, 1)
    bg_recall = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (fg_recall + bg_recall)
    f1 = 2.0 * precision * fg_recall / max(precision + fg_recall, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {
        "precision": precision,
        "fg_recall": fg_recall,
        "bg_recall": bg_recall,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_samples": int(y_true.size),
    }


def auroc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def auprc_score(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def ece_score(y_true, probs, num_bins=10):
    y_true = np.asarray(y_true).astype(np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = (probs >= 0.5).astype(np.float64)
    conf = np.maximum(probs, 1.0 - probs)
    correct = (pred == y_true).astype(np.float64)
    ece = 0.0
    for idx in range(int(num_bins)):
        lo = idx / float(num_bins)
        hi = (idx + 1) / float(num_bins)
        mask = (conf >= lo) & (conf < hi if idx < num_bins - 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return ece


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x, y):
    return correlation(rankdata_average(x), rankdata_average(y))


def r2_score_1d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.var(y) < 1e-12:
        return float("nan")
    a, b = np.polyfit(x, y, deg=1)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def evaluate_scores(y_true, logits, threshold=0.5):
    probs = sigmoid_np(logits)
    metrics = binary_metrics_from_scores(y_true, probs, threshold=threshold)
    metrics.update(
        {
            "auroc": auroc_score(y_true, logits),
            "auprc": auprc_score(y_true, logits),
            "ece": ece_score(y_true, probs),
        }
    )
    return metrics


def macro_metrics(y_true, logits, image_ids, threshold=0.5):
    rows = []
    for image_id in sorted(set(image_ids)):
        mask = image_ids == image_id
        if int(mask.sum()) <= 0:
            continue
        if len(np.unique(y_true[mask])) < 2:
            continue
        rows.append(evaluate_scores(y_true[mask], logits[mask], threshold=threshold))
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys if key not in {"tp", "tn", "fp", "fn"}}


def choose_threshold(y_true, logits):
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    probs = sigmoid_np(logits)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = binary_metrics_from_scores(y_true, probs, threshold=threshold)["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def zip_report(out_dir, zip_name, include_names):
    out_dir = Path(out_dir)
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = out_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=name)
        plots_dir = out_dir / "plots"
        if plots_dir.exists():
            for path in sorted(plots_dir.glob("*.png")):
                zf.write(path, arcname=str(path.relative_to(out_dir)))
    return zip_path
