#!/usr/bin/env python3
"""Render one auditable direct-DINO versus GBSP reconstruction case."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from models.gbsp_similarity_baselines import (  # noqa: E402
    compute_similarity_baselines,
    minmax_per_image,
)


METHODS = ("mean_l2", "proto_cos", "nn_cos", "knn8_cos", "gbsp_r8")
DATASET_ALIASES = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("patch_tokens", "features", "tensor"):
        value = payload.get(key)
        if torch.is_tensor(value):
            return value.squeeze().float()
    candidates = [
        value
        for value in payload.values()
        if torch.is_tensor(value) and value.numel() == 384 * 37 * 37
    ]
    if len(candidates) != 1:
        raise KeyError(f"Cannot resolve one DINO feature tensor from {path}")
    return candidates[0].squeeze().float()


def load_metrics(path: str | Path, dataset: str, stem: str) -> dict[str, dict[str, float]]:
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["stem"] == stem and row["method"] in METHODS:
                result[row["method"]] = {
                    key: float(row[key])
                    for key in ("ap", "auroc", "p_at_r50", "p_at_r60", "p_at_r70")
                }
    missing = sorted(set(METHODS).difference(result))
    if missing:
        raise RuntimeError(f"Missing per-image metrics for {dataset}/{stem}: {missing}")
    return result


def native_response(score: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    value = score.detach().float().reshape(1, 1, 37, 37)
    value = F.interpolate(value, size=(68, 68), mode="bilinear", align_corners=False)
    value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    return value[0, 0].cpu().numpy()


def save_gray(path: Path, value: np.ndarray) -> None:
    array = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_heatmap(path: Path, value: np.ndarray) -> None:
    plt.imsave(path, value, cmap="magma", vmin=0.0, vmax=1.0)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def hide_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    args = parse_args()
    core_root = Path(args.core_root).resolve()
    core_path = core_root / args.dataset / f"{args.stem}.pt"
    output = Path(args.out_dir).resolve()
    if MAIN_ROOT.resolve() in output.parents or output == MAIN_ROOT.resolve():
        raise ValueError("Visualization output must stay outside the source repository main directory")
    output.mkdir(parents=True, exist_ok=True)

    core = torch.load(core_path, map_location="cpu", weights_only=False)
    if str(core["dataset"]) != args.dataset or str(core["stem"]) != args.stem:
        raise RuntimeError(f"Core identity mismatch: {core_path}")
    r8 = core["results"]["r8"]
    feature = load_feature(core["source_feature_path"])
    direct = compute_similarity_baselines(
        feature,
        torch.as_tensor(r8["background_indices"]),
        k=8,
    ).scores
    raw_scores = {**direct, "gbsp_r8": torch.as_tensor(r8["absolute_raw"]).float().reshape(-1)}
    normalized_scores = {
        method: minmax_per_image(raw_scores[method]).reshape(-1)
        for method in METHODS
    }

    metric_dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    metrics = load_metrics(args.metrics_csv, metric_dataset, args.stem)
    with Image.open(core["image_path"]) as handle:
        rgb_image = handle.convert("RGB").copy()
    with Image.open(core["gt_path"]) as handle:
        gt_image = handle.convert("L").copy()
    rgb = np.asarray(rgb_image)
    gt = np.asarray(gt_image, dtype=np.float32) / 255.0
    height, width = gt.shape
    if rgb.shape[:2] != (height, width):
        rgb = np.asarray(rgb_image.resize((width, height), Image.Resampling.BICUBIC))
    responses = {
        method: native_response(normalized_scores[method], (height, width))
        for method in METHODS
    }

    rgb_image.save(output / "rgb.png")
    gt_image.save(output / "gt.png")
    for method in METHODS:
        save_gray(output / f"{method}_response_gray.png", responses[method])
        save_heatmap(output / f"{method}_response_heatmap.png", responses[method])

    response_titles = {
        method: f"{method}\nAP={metrics[method]['ap']:.3f}, AUC={metrics[method]['auroc']:.3f}"
        for method in METHODS
    }
    comparison_values = (rgb, gt, responses["knn8_cos"], responses["gbsp_r8"])
    comparison_titles = (
        "RGB",
        "GT",
        response_titles["knn8_cos"].replace("knn8_cos", "Direct DINO KNN8"),
        response_titles["gbsp_r8"].replace("gbsp_r8", "GBSP-r8 residual"),
    )
    for style, cmap in (("gray", "gray"), ("heatmap", "magma")):
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2))
        for index, (axis, value, title) in enumerate(zip(axes, comparison_values, comparison_titles)):
            if index == 0:
                axis.imshow(value)
            elif index == 1:
                axis.imshow(value, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axis.imshow(value, cmap=cmap, vmin=0.0, vmax=1.0)
                axis.contour(gt > 0.5, levels=[0.5], colors=["cyan"], linewidths=1.0)
            axis.set_title(title, fontsize=10)
            hide_axis(axis)
        fig.suptitle(f"{metric_dataset}/{args.stem}: direct matching vs reconstruction residual", fontsize=12)
        fig.tight_layout()
        save_figure(fig, output / f"response_comparison_{style}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 7, figsize=(18.0, 3.0))
    values = (rgb, gt, *(responses[method] for method in METHODS))
    titles = ("RGB", "GT", *(response_titles[method] for method in METHODS))
    for index, (axis, value, title) in enumerate(zip(axes, values, titles)):
        if index == 0:
            axis.imshow(value)
        else:
            axis.imshow(value, cmap="gray", vmin=0.0, vmax=1.0)
            if index >= 2:
                axis.contour(gt > 0.5, levels=[0.5], colors=["cyan"], linewidths=0.8)
        axis.set_title(title, fontsize=8)
        hide_axis(axis)
    fig.suptitle("All direct DINO responses versus GBSP-r8", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output / "all_direct_vs_gbsp_gray")
    plt.close(fig)

    knn = responses["knn8_cos"]
    gbsp = responses["gbsp_r8"]
    delta = gbsp - knn
    delta_limit = max(float(np.abs(delta).max()), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for axis, value, title in zip(
        axes,
        (knn, gbsp, delta),
        ("Direct DINO KNN8", "GBSP-r8 residual", "GBSP - KNN8"),
    ):
        if title == "GBSP - KNN8":
            image = axis.imshow(value, cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit)
        else:
            image = axis.imshow(value, cmap="magma", vmin=0.0, vmax=1.0)
        axis.contour(gt > 0.5, levels=[0.5], colors=["cyan"], linewidths=1.0)
        axis.set_title(title, fontsize=10)
        hide_axis(axis)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle("Response change after background-subspace reconstruction", fontsize=12)
    fig.tight_layout()
    save_figure(fig, output / "response_delta_heatmap")
    plt.close(fig)

    best_raw_ap = max(metrics[method]["ap"] for method in METHODS[:-1])
    best_raw_auc = max(metrics[method]["auroc"] for method in METHODS[:-1])
    metadata = {
        "schema": "gbsp_reconstruction_case_visualization_v1",
        "dataset": metric_dataset,
        "core_dataset": args.dataset,
        "stem": args.stem,
        "selection": "offline GT-aware mechanism-case audit",
        "core_path": str(core_path),
        "feature_path": str(core["source_feature_path"]),
        "image_path": str(core["image_path"]),
        "gt_path": str(core["gt_path"]),
        "background_candidates": int(torch.as_tensor(r8["background_indices"]).numel()),
        "direct_baseline": "1 - mean top-8 cosine similarity to the same background candidates, leave-one-out",
        "gbsp_response": "rank-8 background-subspace absolute reconstruction residual",
        "normalization": "per-image Min-Max at 37x37",
        "visualization_resize": "37 -> 68 -> native, bilinear, align_corners=False",
        "gt_contour_color": "cyan",
        "metrics": metrics,
        "best_raw_vs_gbsp": {
            "ap": {"raw": best_raw_ap, "gbsp": metrics["gbsp_r8"]["ap"], "delta": metrics["gbsp_r8"]["ap"] - best_raw_ap},
            "auroc": {"raw": best_raw_auc, "gbsp": metrics["gbsp_r8"]["auroc"], "delta": metrics["gbsp_r8"]["auroc"] - best_raw_auc},
        },
        "training_run": False,
        "dino_forward_run": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "files": sorted(path.name for path in output.iterdir())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
