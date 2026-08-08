"""Shared, frozen-protocol utilities for the GBSP/KNN8/LSR task."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from skimage.filters import threshold_multiotsu, threshold_otsu
from sklearn.metrics import average_precision_score, roc_auc_score

from common.eval_dabe_rank_calibration import FastCODContext


GRID = 37
NUM_PATCHES = GRID * GRID
DATASETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "CAMO": 250, "COD10K": 2026, "NC4K": 4121}
COD_METRICS = (
    "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE",
    "Precision", "Recall", "Area", "IoU", "Dice",
)
DATASET_ALIASES = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}
METHOD_ALIASES = {
    "knn8": "knn8_cos", "gbsp": "gbsp_r8", "global_gbsp": "gbsp_r8",
    "mean_prototype": "rank0_global_reference", "nn": "nn_cos",
}


def normalize_dataset(value: str) -> str:
    return DATASET_ALIASES.get(str(value), str(value))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_manifest(root: str | Path, *, score: bool = False, split: str = "test") -> Path:
    root = Path(root).resolve()
    names = (
        ("score_manifest.jsonl", f"score_manifest_{split}.jsonl")
        if score else
        (f"manifest_{split}.jsonl", "manifest.jsonl")
    )
    for name in names:
        path = root / name
        if path.is_file():
            return path
    matches = sorted(root.glob("*manifest*.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot resolve {'score ' if score else ''}manifest under {root}")
    return matches[0]


def load_manifest(
    root: str | Path,
    *,
    score: bool = False,
    split: str = "test",
    max_samples: int = -1,
) -> list[dict]:
    rows = read_jsonl(resolve_manifest(root, score=score, split=split))
    seen = set()
    output = []
    for line, raw in enumerate(rows, 1):
        row = dict(raw)
        row["dataset"] = normalize_dataset(row.get("dataset", ""))
        key = (row["dataset"], str(row.get("stem", "")))
        if not all(key) or key in seen:
            raise RuntimeError(f"invalid/duplicate identity at manifest line {line}: {key}")
        seen.add(key)
        output.append(row)
    output.sort(key=lambda row: (DATASETS.index(row["dataset"]) if row["dataset"] in DATASETS else 99, row["stem"]))
    return output if int(max_samples) < 0 else output[: int(max_samples)]


def index_manifest(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    return {(normalize_dataset(row["dataset"]), str(row["stem"])): row for row in rows}


def load_torch(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict payload: {path}")
    return value


def score_path(row: dict) -> Path:
    for field in ("score_path", "cache_path"):
        if row.get(field):
            return Path(row[field])
    raise KeyError("manifest row has neither score_path nor cache_path")


def score_from_payload(payload: dict, method: str) -> torch.Tensor:
    requested = str(method)
    aliased = METHOD_ALIASES.get(requested, requested)
    # New experiment caches may deliberately use the concise canonical name
    # (for example ``knn8``), whereas historical payloads use ``knn8_cos``.
    # Prefer an exact payload match and only then fall back to the old alias.
    if isinstance(payload.get("scores"), dict) and torch.is_tensor(payload["scores"].get(requested)):
        name = requested
    else:
        name = aliased
    if name == "rank0_global_reference" and torch.is_tensor(payload.get(name)):
        value = payload[name]
    elif isinstance(payload.get("scores"), dict) and torch.is_tensor(payload["scores"].get(name)):
        value = payload["scores"][name]
    elif isinstance(payload.get("variants"), dict) and isinstance(payload["variants"].get(name), dict):
        value = payload["variants"][name].get("local_residual")
    elif torch.is_tensor(payload.get(name)):
        value = payload[name]
    else:
        raise KeyError(f"score {name!r} not found; keys={sorted(payload)}")
    value = value.detach().cpu().float().reshape(-1)
    if value.numel() != NUM_PATCHES or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain 1369 finite values")
    return value


def minmax(value: torch.Tensor | np.ndarray, eps: float = 1e-12) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    low, high = tensor.min(), tensor.max()
    if float(high - low) <= eps:
        return torch.zeros_like(tensor)
    return ((tensor - low) / (high - low)).clamp(0.0, 1.0)


def load_native_gt(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy((value > 0.5).astype(np.float32)).unsqueeze(0)


def load_patch_area(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(value).unsqueeze(0).unsqueeze(0)
    return F.interpolate(tensor, size=(GRID, GRID), mode="area")[0, 0].numpy().reshape(-1)


def patch_labels(area: np.ndarray, rule: str = "main_0.5") -> tuple[np.ndarray, np.ndarray]:
    area = np.asarray(area, dtype=np.float64).reshape(-1)
    if rule == "main_0.5":
        return (area >= 0.5).astype(np.uint8), np.ones(area.shape, dtype=bool)
    if rule == "strict_0.8_0.2":
        return (area >= 0.8).astype(np.uint8), (area <= 0.2) | (area >= 0.8)
    raise ValueError(f"unknown patch label rule: {rule}")


def resize_score_to_native(score37: torch.Tensor | np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
    value = torch.as_tensor(score37).detach().cpu().float().reshape(1, 1, GRID, GRID)
    value = F.interpolate(value, size=(68, 68), mode="bilinear", align_corners=False)
    return F.interpolate(value, size=shape, mode="bilinear", align_corners=False).squeeze(0)


def rank_metrics(score: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> dict[str, float]:
    value = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(labels).reshape(-1) > 0.5
    if target.sum() == 0 or (~target).sum() == 0 or not np.isfinite(value).all():
        return {"AP": float("nan"), "AUROC": float("nan")}
    return {
        "AP": float(average_precision_score(target, value)),
        "AUROC": float(roc_auc_score(target, value)),
    }


def cod_metrics(context: FastCODContext, probability: torch.Tensor, threshold: float) -> dict[str, float]:
    values = context.evaluate_many([("candidate", "hard", probability)], float(threshold))[("candidate", "hard")]
    precision, recall = float(values["Precision"]), float(values["Recall"])
    return {
        "S_m": float(values["S_m"]), "F_beta_w": float(values["F_beta_w"]),
        "F_beta_mean": float(values["F_beta_mean"]), "E_mean": float(values["E_mean"]),
        "MAE": float(values["MAE"]), "Precision": precision, "Recall": recall,
        "Area": float(values["Area"]), "IoU": float(values["IoU"]),
        "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
    }


def adaptive_threshold(score37: torch.Tensor | np.ndarray, method: str) -> tuple[float, bool]:
    value = np.asarray(score37, dtype=np.float64).reshape(-1)
    unique = np.unique(value)
    if unique.size < 2:
        return 0.5, True
    if method == "otsu":
        return float(threshold_otsu(value, nbins=256)), False
    if method == "multi_otsu_3":
        if unique.size < 3:
            return float(threshold_otsu(value, nbins=256)), True
        return float(threshold_multiotsu(value, classes=3, nbins=256)[-1]), False
    raise ValueError(f"unknown adaptive threshold: {method}")


def right_ecdf(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    ordered = np.sort(value, kind="mergesort")
    return np.searchsorted(ordered, value, side="right").astype(np.float64) / float(value.size)


def correlation_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return {"Pearson": float("nan"), "Spearman": float("nan"), "Kendall_tau": float("nan")}
    return {
        "Pearson": float(pearsonr(a, b).statistic),
        "Spearman": float(spearmanr(a, b).statistic),
        "Kendall_tau": float(kendalltau(a, b).statistic),
    }


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_per_dataset(
    rows: Sequence[dict],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple([normalize_dataset(row["dataset"]), *[row[field] for field in group_fields]])].append(row)
    output = []
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output.append({
            "scope": "dataset", "dataset": key[0],
            **dict(zip(group_fields, key[1:])), "num_samples": len(subset),
            **{metric: finite_mean(row[metric] for row in subset) for metric in metric_fields},
        })
    combinations = sorted({tuple(row[field] for field in group_fields) for row in rows}, key=lambda key: tuple(map(str, key)))
    for combination in combinations:
        selected = [row for row in output if tuple(row[field] for field in group_fields) == combination]
        if selected:
            output.append({
                "scope": "dataset_macro", "dataset": "ALL",
                **dict(zip(group_fields, combination)),
                "num_samples": sum(int(row["num_samples"]) for row in selected),
                **{metric: finite_mean(row[metric] for row in selected) for metric in metric_fields},
            })
            source = [row for row in rows if tuple(row[field] for field in group_fields) == combination]
            output.append({
                "scope": "image_macro", "dataset": "ALL",
                **dict(zip(group_fields, combination)), "num_samples": len(source),
                **{metric: finite_mean(row[metric] for row in source) for metric in metric_fields},
            })
    return output


def plateau_rows(curve: Sequence[dict], method_field: str = "method") -> list[dict]:
    output = []
    methods = sorted({row[method_field] for row in curve})
    for method in methods:
        rows = sorted(
            (row for row in curve if row[method_field] == method and row.get("scope") == "dataset_macro"),
            key=lambda row: float(row["threshold"]),
        )
        if not rows:
            continue
        best = {
            "S_m": max(float(row["S_m"]) for row in rows),
            "F_beta_w": max(float(row["F_beta_w"]) for row in rows),
            "E_mean": max(float(row["E_mean"]) for row in rows),
            "MAE": min(float(row["MAE"]) for row in rows),
        }
        stable = [
            row for row in rows
            if float(row["S_m"]) >= best["S_m"] - 0.005
            and float(row["F_beta_w"]) >= best["F_beta_w"] - 0.005
            and float(row["E_mean"]) >= best["E_mean"] - 0.005
            and float(row["MAE"]) <= best["MAE"] + 0.003
        ]
        low = min(float(row["threshold"]) for row in stable) if stable else float("nan")
        high = max(float(row["threshold"]) for row in stable) if stable else float("nan")
        output.append({
            method_field: method, "plateau_min": low, "plateau_max": high,
            "plateau_width": high - low if stable else 0.0,
            "stable_threshold_count": len(stable),
            **{f"best_{metric}": value for metric, value in best.items()},
        })
    return output


def stratified_paired_bootstrap(
    per_image: Sequence[dict],
    *,
    candidate: str,
    baseline: str,
    method_field: str,
    metrics: Sequence[str],
    repetitions: int,
    seed: int,
    protocol_fields: Sequence[str] = (),
) -> list[dict]:
    if int(repetitions) <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    output = []
    protocol_values = sorted({tuple(row[field] for field in protocol_fields) for row in per_image}) or [()]
    for protocol in protocol_values:
        subset = [row for row in per_image if tuple(row[field] for field in protocol_fields) == protocol]
        lookup = {(normalize_dataset(row["dataset"]), row["stem"], row[method_field]): row for row in subset}
        for metric in metrics:
            arrays = []
            for dataset in DATASETS:
                keys = sorted({
                    (key[0], key[1]) for key in lookup
                    if key[0] == dataset
                    and (*key[:2], candidate) in lookup
                    and (*key[:2], baseline) in lookup
                })
                values = np.asarray([
                    float(lookup[(*key, candidate)][metric]) - float(lookup[(*key, baseline)][metric])
                    for key in keys
                ], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    arrays.append((dataset, values))
            if not arrays:
                continue
            draws = np.empty(int(repetitions), dtype=np.float64)
            per_dataset_draws = {dataset: np.empty(int(repetitions), dtype=np.float64) for dataset, _ in arrays}
            for repetition in range(int(repetitions)):
                means = []
                for dataset, values in arrays:
                    draw = float(values[rng.integers(0, values.size, values.size)].mean())
                    per_dataset_draws[dataset][repetition] = draw
                    means.append(draw)
                draws[repetition] = float(np.mean(means))
            base = {
                "candidate": candidate, "baseline": baseline,
                **dict(zip(protocol_fields, protocol)), "metric": metric,
                "repetitions": int(repetitions), "seed": int(seed),
            }
            output.append({
                **base, "dataset": "DATASET_MACRO",
                "delta": float(np.mean([values.mean() for _, values in arrays])),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            })
            for dataset, values in arrays:
                distribution = per_dataset_draws[dataset]
                output.append({
                    **base, "dataset": dataset, "delta": float(values.mean()),
                    "ci95_low": float(np.quantile(distribution, 0.025)),
                    "ci95_high": float(np.quantile(distribution, 0.975)),
                })
    return output
