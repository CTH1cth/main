#!/usr/bin/env python3
"""Find samples where L2 DINO PCA overlaps but GBSP residual PCA separates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score


DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rank", default="r8")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shortlist_per_dataset", type=int, default=250)
    parser.add_argument("--refine_top", type=int, default=80)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--min_foreground_patches", type=int, default=20)
    parser.add_argument("--min_background_patches", type=int, default=100)
    parser.add_argument("--background_occupancy_max", type=float, default=0.20)
    parser.add_argument("--foreground_occupancy_min", type=float, default=0.80)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = None
    if torch.is_tensor(payload):
        value = payload
    elif isinstance(payload, dict):
        for key in ("patch_tokens", "features", "tensor"):
            if torch.is_tensor(payload.get(key)):
                value = payload[key]
                break
    if value is None:
        raise KeyError(f"Cannot resolve DINO feature tensor from {path}")
    value = value.detach().cpu().float().squeeze()
    if value.ndim == 3:
        channels, height, width = value.shape
        value = value.permute(1, 2, 0).reshape(height * width, channels)
    elif value.ndim == 2:
        if value.shape[0] != 37 * 37 and value.shape[1] == 37 * 37:
            value = value.t().contiguous()
    if tuple(value.shape) != (37 * 37, 384):
        raise ValueError(f"Unexpected feature shape in {path}: {tuple(value.shape)}")
    return F.normalize(value, p=2, dim=1)


def occupancy_labels(
    path: str | Path,
    background_max: float,
    foreground_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array)[None, None]
    resized = F.interpolate(tensor, size=(296, 296), mode="nearest")
    occupancy = F.avg_pool2d(resized, kernel_size=8, stride=8)[0, 0].numpy().reshape(-1)
    background = occupancy <= background_max
    foreground = occupancy >= foreground_min
    valid = background | foreground
    labels = foreground[valid].astype(np.uint8)
    return valid, labels, occupancy


def safe_auc(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, score)),
        float(average_precision_score(labels, score)),
    )


def stage1_one(path: Path, args: argparse.Namespace) -> dict | None:
    try:
        core = torch.load(path, map_location="cpu", weights_only=False)
        if args.rank not in core["results"]:
            return None
        valid, labels, occupancy = occupancy_labels(
            core["gt_path"],
            args.background_occupancy_max,
            args.foreground_occupancy_min,
        )
        foreground_count = int(labels.sum())
        background_count = int(labels.size - labels.sum())
        if foreground_count < args.min_foreground_patches or background_count < args.min_background_patches:
            return None
        score = torch.as_tensor(core["results"][args.rank]["absolute_raw"]).float().numpy().reshape(-1)
        auc, ap = safe_auc(labels, score[valid])
        return {
            "dataset": str(core["dataset"]),
            "stem": str(core["stem"]),
            "core_path": str(path.resolve()),
            "feature_path": str(Path(core["source_feature_path"]).resolve()),
            "image_path": str(Path(core["image_path"]).resolve()),
            "gt_path": str(Path(core["gt_path"]).resolve()),
            "foreground_patches": foreground_count,
            "background_patches": background_count,
            "ambiguous_patches": int((~valid).sum()),
            "foreground_fraction_confident": float(labels.mean()),
            "foreground_fraction_all": float((occupancy >= 0.5).mean()),
            "residual_energy_auc": auc,
            "residual_energy_ap": ap,
            "stage1_quality": float(auc + 0.25 * ap),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "core_path": str(path)}


def pca_lowrank(value: torch.Tensor, seed: int) -> tuple[np.ndarray, float]:
    centered = value - value.mean(dim=0, keepdim=True)
    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if value.is_cuda else None
    torch.manual_seed(seed)
    if value.is_cuda:
        torch.cuda.manual_seed_all(seed)
    _, singular, vectors = torch.pca_lowrank(centered, q=6, center=False, niter=3)
    embedding = centered @ vectors[:, :2]
    explained = float(singular[:2].square().sum() / centered.square().sum().clamp_min(1e-12))
    torch.random.set_rng_state(generator_state)
    if value.is_cuda and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    return embedding.detach().cpu().numpy(), explained


def pca_refined(value: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    pca = PCA(
        n_components=2,
        svd_solver="randomized",
        n_oversamples=24,
        iterated_power=7,
        random_state=seed,
    )
    embedding = pca.fit_transform(value)
    return embedding, float(pca.explained_variance_ratio_.sum())


def embedding_metrics(
    embedding: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    knn: int,
    with_silhouette: bool,
) -> dict[str, float]:
    points = embedding[valid]
    foreground = labels.astype(bool)
    background = ~foreground
    prevalence = float(labels.mean())
    query_k = min(int(knn) + 1, len(points))
    neighbors = cKDTree(points).query(points, k=query_k)[1]
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    neighbors = neighbors[:, 1:]
    neighbor_labels = labels[neighbors]
    foreground_purity = float(neighbor_labels[foreground].mean())
    background_purity = float((1 - neighbor_labels[background]).mean())
    foreground_lift = float((foreground_purity - prevalence) / max(1.0 - prevalence, 1e-12))
    background_prevalence = 1.0 - prevalence
    background_lift = float(
        (background_purity - background_prevalence) / max(prevalence, 1e-12)
    )
    mean_foreground = points[foreground].mean(axis=0)
    mean_background = points[background].mean(axis=0)
    separation = float(np.linalg.norm(mean_foreground - mean_background))
    within = float(
        np.sqrt(
            0.5
            * (
                np.mean(np.sum((points[foreground] - mean_foreground) ** 2, axis=1))
                + np.mean(np.sum((points[background] - mean_background) ** 2, axis=1))
            )
        )
    )
    result = {
        "fg_knn_purity": foreground_purity,
        "bg_knn_purity": background_purity,
        "fg_knn_lift": foreground_lift,
        "bg_knn_lift": background_lift,
        "balanced_knn_lift": 0.5 * (foreground_lift + background_lift),
        "centroid_separation_over_within": separation / max(within, 1e-12),
    }
    if with_silhouette:
        result["silhouette"] = float(silhouette_score(points, labels))
    return result


def feature_and_residual(row: dict, rank: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    core = torch.load(row["core_path"], map_location="cpu", weights_only=False)
    feature = load_feature(row["feature_path"]).to(device)
    result = core["results"][rank]
    mean = torch.as_tensor(result["mean"]).float().reshape(1, -1).to(device)
    basis = torch.as_tensor(result["basis"]).float().to(device)
    centered = feature - mean
    residual = centered - (centered @ basis) @ basis.t()
    cached = torch.as_tensor(result["absolute_raw"]).float().reshape(-1).to(device)
    error = float((residual.square().sum(dim=1) - cached).abs().max())
    if error > 2e-6:
        raise RuntimeError(f"Residual reproduction error {error} for {row['dataset']}/{row['stem']}")
    return feature, residual


def stage2_one(row: dict, args: argparse.Namespace, index: int) -> dict:
    valid, labels, _ = occupancy_labels(
        row["gt_path"],
        args.background_occupancy_max,
        args.foreground_occupancy_min,
    )
    feature, residual = feature_and_residual(row, args.rank, args.device)
    raw_pca, raw_variance = pca_lowrank(feature, args.seed + index * 2)
    residual_pca, residual_variance = pca_lowrank(residual, args.seed + index * 2 + 1)
    raw_metrics = embedding_metrics(raw_pca, valid, labels, args.knn, False)
    residual_metrics = embedding_metrics(residual_pca, valid, labels, args.knn, False)
    result = dict(row)
    result.update({f"raw_{key}": value for key, value in raw_metrics.items()})
    result.update({f"residual_{key}": value for key, value in residual_metrics.items()})
    result["raw_pca_variance"] = raw_variance
    result["residual_pca_variance"] = residual_variance
    result["fg_knn_lift_gain"] = residual_metrics["fg_knn_lift"] - raw_metrics["fg_knn_lift"]
    result["balanced_knn_lift_gain"] = (
        residual_metrics["balanced_knn_lift"] - raw_metrics["balanced_knn_lift"]
    )
    result["centroid_ratio_gain"] = (
        residual_metrics["centroid_separation_over_within"]
        - raw_metrics["centroid_separation_over_within"]
    )
    return result


def stage3_one(row: dict, args: argparse.Namespace, index: int) -> dict:
    valid, labels, _ = occupancy_labels(
        row["gt_path"],
        args.background_occupancy_max,
        args.foreground_occupancy_min,
    )
    feature, residual = feature_and_residual(row, args.rank, args.device)
    raw_np = feature.cpu().numpy()
    residual_np = residual.cpu().numpy()
    raw_pca, raw_variance = pca_refined(raw_np, args.seed + index * 2)
    residual_pca, residual_variance = pca_refined(residual_np, args.seed + index * 2 + 1)
    raw_metrics = embedding_metrics(raw_pca, valid, labels, args.knn, True)
    residual_metrics = embedding_metrics(residual_pca, valid, labels, args.knn, True)
    result = dict(row)
    for key in list(result):
        if key.startswith("raw_") or key.startswith("residual_") or key.endswith("_gain"):
            if key not in ("residual_energy_auc", "residual_energy_ap"):
                result.pop(key)
    result.update({f"raw_{key}": value for key, value in raw_metrics.items()})
    result.update({f"residual_{key}": value for key, value in residual_metrics.items()})
    result["raw_pca_variance"] = raw_variance
    result["residual_pca_variance"] = residual_variance
    result["fg_knn_lift_gain"] = residual_metrics["fg_knn_lift"] - raw_metrics["fg_knn_lift"]
    result["balanced_knn_lift_gain"] = (
        residual_metrics["balanced_knn_lift"] - raw_metrics["balanced_knn_lift"]
    )
    result["centroid_ratio_gain"] = (
        residual_metrics["centroid_separation_over_within"]
        - raw_metrics["centroid_separation_over_within"]
    )
    result["silhouette_gain"] = residual_metrics["silhouette"] - raw_metrics["silhouette"]
    return result


def percentile(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(values if higher_is_better else -values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    return ranks / (len(values) - 1)


def add_composite(rows: list[dict], refined: bool) -> None:
    raw_purity = np.asarray([row["raw_fg_knn_lift"] for row in rows])
    residual_purity = np.asarray([row["residual_fg_knn_lift"] for row in rows])
    raw_centroid = np.asarray([row["raw_centroid_separation_over_within"] for row in rows])
    residual_centroid = np.asarray([row["residual_centroid_separation_over_within"] for row in rows])
    gain = np.asarray([row["fg_knn_lift_gain"] for row in rows])
    energy_auc = np.asarray([row["residual_energy_auc"] for row in rows])
    score = (
        0.22 * percentile(raw_purity, higher_is_better=False)
        + 0.13 * percentile(raw_centroid, higher_is_better=False)
        + 0.22 * percentile(residual_purity, higher_is_better=True)
        + 0.13 * percentile(residual_centroid, higher_is_better=True)
        + 0.20 * percentile(gain, higher_is_better=True)
        + 0.10 * percentile(energy_auc, higher_is_better=True)
    )
    if refined:
        raw_silhouette = np.asarray([row["raw_silhouette"] for row in rows])
        residual_silhouette = np.asarray([row["residual_silhouette"] for row in rows])
        silhouette_gain = np.asarray([row["silhouette_gain"] for row in rows])
        score = (
            0.75 * score
            + 0.08 * percentile(raw_silhouette, higher_is_better=False)
            + 0.08 * percentile(residual_silhouette, higher_is_better=True)
            + 0.09 * percentile(silhouette_gain, higher_is_better=True)
        )
    for row, value in zip(rows, score):
        row["composite_score"] = float(value)


def main() -> None:
    args = parse_args()
    started = time.time()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    core_root = Path(args.core_root).resolve()
    paths: list[Path] = []
    for dataset in args.datasets:
        paths.extend(sorted((core_root / dataset).glob("*.pt")))
    if args.max_samples > 0 and len(paths) > args.max_samples:
        indices = np.linspace(0, len(paths) - 1, args.max_samples, dtype=int)
        paths = [paths[index] for index in indices]
    print(f"stage1 paths={len(paths)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        stage1_raw = list(executor.map(lambda path: stage1_one(path, args), paths))
    errors = [row for row in stage1_raw if row is not None and "error" in row]
    stage1 = [row for row in stage1_raw if row is not None and "error" not in row]
    if not stage1:
        raise RuntimeError("Stage 1 produced no eligible samples")
    stage1.sort(key=lambda row: row["stage1_quality"], reverse=True)
    write_csv(output / "stage1_residual_energy.csv", stage1)
    if errors:
        write_csv(output / "errors.csv", errors)
    print(f"stage1 eligible={len(stage1)} errors={len(errors)} elapsed={time.time()-started:.1f}s", flush=True)

    shortlist: list[dict] = []
    for dataset in args.datasets:
        candidates = [row for row in stage1 if row["dataset"] == dataset]
        shortlist.extend(candidates[: args.shortlist_per_dataset])
    print(f"stage2 shortlist={len(shortlist)}", flush=True)
    stage2 = []
    for index, row in enumerate(shortlist):
        stage2.append(stage2_one(row, args, index))
        if (index + 1) % 50 == 0 or index + 1 == len(shortlist):
            print(f"stage2 {index+1}/{len(shortlist)} elapsed={time.time()-started:.1f}s", flush=True)
    add_composite(stage2, refined=False)
    stage2.sort(key=lambda row: row["composite_score"], reverse=True)
    write_csv(output / "stage2_pca_approx.csv", stage2)

    refine_count = min(args.refine_top, len(stage2))
    print(f"stage3 refine={refine_count}", flush=True)
    stage3 = []
    for index, row in enumerate(stage2[:refine_count]):
        stage3.append(stage3_one(row, args, index))
        if (index + 1) % 10 == 0 or index + 1 == refine_count:
            print(f"stage3 {index+1}/{refine_count} elapsed={time.time()-started:.1f}s", flush=True)
    add_composite(stage3, refined=True)
    stage3.sort(key=lambda row: row["composite_score"], reverse=True)
    write_csv(output / "stage3_pca_refined.csv", stage3)

    summary = {
        "schema": "gbsp_feature_separability_scan_v1",
        "objective": "L2 DINOv1-S/8 PCA overlap with GBSP residual-vector PCA separation",
        "rank": args.rank,
        "datasets": args.datasets,
        "counts": {
            "paths": len(paths),
            "eligible": len(stage1),
            "shortlist": len(stage2),
            "refined": len(stage3),
            "errors": len(errors),
        },
        "label_protocol": {
            "background_occupancy_max": args.background_occupancy_max,
            "foreground_occupancy_min": args.foreground_occupancy_min,
            "min_foreground_patches": args.min_foreground_patches,
            "min_background_patches": args.min_background_patches,
            "gt_role": "GT is used only to score and rank unsupervised embeddings, never to fit PCA or GBSP.",
        },
        "selection_note": (
            "This is a GT-assisted qualitative case-mining diagnostic. The selected case must not be "
            "presented as representative or average behavior."
        ),
        "best": stage3[0],
        "top10": stage3[:10],
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
