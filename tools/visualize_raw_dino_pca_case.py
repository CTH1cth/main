#!/usr/bin/env python3
"""Compare paper-style PCA plots for one cached DINO patch-feature sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BACKGROUND_COLOR = "#65B5DC"
FOREGROUND_COLOR = "#E85B48"
AMBIGUOUS_COLOR = "#BDBDBD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--paper_pca_path")
    parser.add_argument("--paper_title", default="Paper: DINOv2 PCA")
    parser.add_argument("--feature_title", default="DINOv1-S/8")
    parser.add_argument(
        "--paper_align_flip_both",
        action="store_true",
        help="Also save a comparison with both PCA coordinate signs flipped.",
    )
    parser.add_argument("--background_occupancy_max", type=float, default=0.20)
    parser.add_argument("--foreground_occupancy_min", type=float, default=0.80)
    return parser.parse_args()


def load_feature(path: str | Path) -> tuple[torch.Tensor, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        value = payload
    elif isinstance(payload, dict):
        value = None
        for key in ("patch_tokens", "features", "tensor"):
            candidate = payload.get(key)
            if torch.is_tensor(candidate):
                value = candidate
                break
        if value is None:
            raise KeyError(f"Cannot resolve a feature tensor from {path}")
    else:
        raise TypeError(f"Unsupported feature payload: {type(payload)!r}")

    value = value.detach().cpu().float().squeeze()
    if value.ndim == 3:
        channels, height, width = value.shape
        value = value.permute(1, 2, 0).reshape(height * width, channels)
    elif value.ndim == 2:
        patch_count = min(value.shape)
        side = int(round(patch_count**0.5))
        if side * side != patch_count:
            raise ValueError(f"Cannot infer a square patch grid from {tuple(value.shape)}")
        if value.shape[0] != patch_count:
            value = value.t().contiguous()
        height = width = side
    else:
        raise ValueError(f"Unexpected feature shape: {tuple(value.shape)}")
    return value, int(height), int(width)


def patch_occupancy(path: str | Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        gt = torch.from_numpy(
            np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        )[None, None]
    occupancy = F.interpolate(gt, size=(height, width), mode="area")[0, 0]
    return occupancy.numpy().reshape(-1)


def nearest_patch_labels(path: str | Path, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        gt = torch.from_numpy(
            np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        )[None, None]
    labels = F.interpolate(gt, size=(height, width), mode="nearest")[0, 0]
    return (labels.numpy().reshape(-1) >= 0.5)


def pca_embedding(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, svd_solver="full")
    embedding = pca.fit_transform(features)
    return embedding, pca.explained_variance_ratio_


def class_metrics(features: np.ndarray, embedding: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    def centroid_ratio(values: np.ndarray) -> float:
        background = values[~labels]
        foreground = values[labels]
        c_background = background.mean(axis=0)
        c_foreground = foreground.mean(axis=0)
        separation = np.linalg.norm(c_background - c_foreground)
        within = np.sqrt(
            0.5
            * (
                np.mean(np.sum((background - c_background) ** 2, axis=1))
                + np.mean(np.sum((foreground - c_foreground) ** 2, axis=1))
            )
        )
        return float(separation / (within + 1e-12))

    return {
        "silhouette_full": float(silhouette_score(features, labels)),
        "silhouette_pca2": float(silhouette_score(embedding, labels)),
        "centroid_separation_over_within_full": centroid_ratio(features),
        "centroid_separation_over_within_pca2": centroid_ratio(embedding),
    }


def scatter_paper_style(axis: plt.Axes, embedding: np.ndarray, labels: np.ndarray, title: str) -> None:
    background = ~labels
    foreground = labels
    axis.scatter(
        embedding[background, 0],
        embedding[background, 1],
        s=14,
        c=BACKGROUND_COLOR,
        edgecolors="#1C3745",
        linewidths=0.45,
        alpha=0.90,
        label="BACK.",
        rasterized=True,
    )
    axis.scatter(
        embedding[foreground, 0],
        embedding[foreground, 1],
        s=18,
        c=FOREGROUND_COLOR,
        edgecolors="#5D211A",
        linewidths=0.45,
        alpha=0.95,
        label="FORE.",
        rasterized=True,
    )
    axis.set_title(title, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="best", frameon=True, fontsize=8)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature, grid_h, grid_w = load_feature(args.feature_path)
    raw = feature.numpy()
    normalized = F.normalize(feature, p=2, dim=1).numpy()
    labels_nearest = nearest_patch_labels(args.gt_path, grid_h, grid_w)
    occupancy = patch_occupancy(args.gt_path, grid_h, grid_w)
    confident_background = occupancy <= float(args.background_occupancy_max)
    confident_foreground = occupancy >= float(args.foreground_occupancy_min)
    ambiguous = ~(confident_background | confident_foreground)

    raw_pca, raw_ratio = pca_embedding(raw)
    normalized_pca, normalized_ratio = pca_embedding(normalized)

    panels = 4 if args.paper_pca_path else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.0 * panels, 3.8))
    axis_index = 0
    if args.paper_pca_path:
        with Image.open(args.paper_pca_path) as paper_image:
            axes[0].imshow(paper_image.convert("RGB"))
        axes[0].set_title(args.paper_title, fontsize=11)
        axes[0].axis("off")
        axis_index = 1
    scatter_paper_style(
        axes[axis_index],
        raw_pca,
        labels_nearest,
        f"Same sample: {args.feature_title} PCA\nraw cached features; variance={raw_ratio.sum():.1%}",
    )
    scatter_paper_style(
        axes[axis_index + 1],
        normalized_pca,
        labels_nearest,
        f"Same sample: {args.feature_title} PCA\nL2-normalized; variance={normalized_ratio.sum():.1%}",
    )
    with Image.open(args.image_path) as input_image:
        axes[axis_index + 2].imshow(input_image.convert("RGB"))
    axes[axis_index + 2].set_title("Exact source image", fontsize=11)
    axes[axis_index + 2].axis("off")
    fig.suptitle(
        "PCA protocol comparison — GT controls point colors only",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(fig, output / "paper_vs_feature_pca")
    plt.close(fig)

    if args.paper_pca_path and args.paper_align_flip_both:
        fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8))
        with Image.open(args.paper_pca_path) as paper_image:
            axes[0].imshow(paper_image.convert("RGB"))
        axes[0].set_title(args.paper_title, fontsize=11)
        axes[0].axis("off")
        scatter_paper_style(
            axes[1],
            -raw_pca,
            labels_nearest,
            f"{args.feature_title}: raw PCA\nboth signs flipped; variance={raw_ratio.sum():.1%}",
        )
        scatter_paper_style(
            axes[2],
            -normalized_pca,
            labels_nearest,
            f"{args.feature_title}: L2 PCA\nboth signs flipped; variance={normalized_ratio.sum():.1%}",
        )
        with Image.open(args.image_path) as input_image:
            axes[3].imshow(input_image.convert("RGB"))
        axes[3].set_title("Exact source image", fontsize=11)
        axes[3].axis("off")
        fig.suptitle(
            "Paper-aligned view — PCA axis signs are mathematically arbitrary",
            fontsize=13,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        save_figure(fig, output / "paper_vs_feature_pca_sign_flipped")
        plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(6.4, 5.4))
    axis.scatter(
        normalized_pca[ambiguous, 0],
        normalized_pca[ambiguous, 1],
        s=10,
        c=AMBIGUOUS_COLOR,
        alpha=0.28,
        linewidths=0,
        label="Ambiguous boundary patch",
        rasterized=True,
    )
    axis.scatter(
        normalized_pca[confident_background, 0],
        normalized_pca[confident_background, 1],
        s=15,
        c=BACKGROUND_COLOR,
        edgecolors="#1C3745",
        linewidths=0.35,
        alpha=0.82,
        label="Confident background",
        rasterized=True,
    )
    axis.scatter(
        normalized_pca[confident_foreground, 0],
        normalized_pca[confident_foreground, 1],
        s=21,
        c=FOREGROUND_COLOR,
        edgecolors="#5D211A",
        linewidths=0.4,
        alpha=0.94,
        label="Confident foreground",
        rasterized=True,
    )
    axis.set_title(
        f"{args.feature_title} PCA with boundary ambiguity shown\n"
        f"PC1+PC2 variance={normalized_ratio.sum():.1%}",
        fontsize=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    save_figure(fig, output / "feature_pca_with_ambiguous_patches")
    plt.close(fig)

    metadata = {
        "feature_path": str(Path(args.feature_path).resolve()),
        "image_path": str(Path(args.image_path).resolve()),
        "gt_path": str(Path(args.gt_path).resolve()),
        "paper_pca_path": str(Path(args.paper_pca_path).resolve()) if args.paper_pca_path else None,
        "feature_title": args.feature_title,
        "grid": [grid_h, grid_w],
        "feature_dimension": int(feature.shape[1]),
        "patch_count": int(feature.shape[0]),
        "nearest_label_counts": {
            "background": int((~labels_nearest).sum()),
            "foreground": int(labels_nearest.sum()),
        },
        "confident_label_counts": {
            "background": int(confident_background.sum()),
            "foreground": int(confident_foreground.sum()),
            "ambiguous": int(ambiguous.sum()),
        },
        "raw_feature_norm": {
            "mean": float(feature.norm(dim=1).mean()),
            "std": float(feature.norm(dim=1).std()),
            "min": float(feature.norm(dim=1).min()),
            "max": float(feature.norm(dim=1).max()),
        },
        "pca": {
            "raw_explained_variance_ratio": raw_ratio.tolist(),
            "normalized_explained_variance_ratio": normalized_ratio.tolist(),
        },
        "separability": {
            "raw": class_metrics(raw, raw_pca, labels_nearest),
            "l2_normalized": class_metrics(normalized, normalized_pca, labels_nearest),
        },
        "protocol_note": (
            "PCA is fitted without GT. GT is used only after projection to color points. "
            "Paper-style panels label every patch by nearest-neighbor resized GT; the audit "
            "panel exposes boundary-ambiguous patches using area occupancy thresholds."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
