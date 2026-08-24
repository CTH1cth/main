#!/usr/bin/env python3
"""Visualize raw DINO patches and exact GBSP residual vectors for one image."""

from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from models.gbsp_similarity_baselines import (  # noqa: E402
    compute_similarity_baselines,
)


GRID = 37
PATCH_COUNT = GRID * GRID
BACKGROUND_COLOR = "#2B8CBE"
FOREGROUND_COLOR = "#E34A33"
AMBIGUOUS_COLOR = "#BDBDBD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--background_occupancy_max", type=float, default=0.20)
    parser.add_argument("--foreground_occupancy_min", type=float, default=0.80)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def load_feature(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = None
    for key in ("patch_tokens", "features", "tensor"):
        if torch.is_tensor(payload.get(key)):
            value = payload[key]
            break
    if value is None:
        candidates = [
            item
            for item in payload.values()
            if torch.is_tensor(item) and item.numel() == 384 * GRID * GRID
        ]
        if len(candidates) != 1:
            raise KeyError(f"Cannot resolve one DINO feature tensor from {path}")
        value = candidates[0]
    value = value.detach().cpu().float().squeeze()
    if value.ndim == 3 and tuple(value.shape[-2:]) == (GRID, GRID):
        value = value.permute(1, 2, 0).reshape(PATCH_COUNT, -1)
    elif value.ndim == 2 and value.shape[0] == PATCH_COUNT:
        value = value.contiguous()
    elif value.ndim == 2 and value.shape[1] == PATCH_COUNT:
        value = value.t().contiguous()
    else:
        raise ValueError(f"Unexpected DINO feature shape: {tuple(value.shape)}")
    return F.normalize(value, p=2, dim=1)


def patch_occupancy(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        gt = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(gt).reshape(1, 1, *gt.shape)
    resized = F.interpolate(tensor, size=(296, 296), mode="nearest")
    occupancy = F.avg_pool2d(resized, kernel_size=8, stride=8)[0, 0]
    if tuple(occupancy.shape) != (GRID, GRID):
        raise RuntimeError(f"Unexpected patch occupancy shape: {tuple(occupancy.shape)}")
    return occupancy.numpy().reshape(-1)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def fit_embeddings(value: np.ndarray, seed: int, perplexity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, svd_solver="full")
    pca_embedding = pca.fit_transform(value)
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate="auto",
        init="pca",
        max_iter=1500,
        random_state=int(seed),
        method="barnes_hut",
    ).fit_transform(value)
    return pca_embedding, tsne_embedding, pca.explained_variance_ratio_


def scatter_classes(
    axis: plt.Axes,
    embedding: np.ndarray,
    background: np.ndarray,
    foreground: np.ndarray,
    ambiguous: np.ndarray,
) -> None:
    axis.scatter(
        embedding[ambiguous, 0],
        embedding[ambiguous, 1],
        s=7,
        c=AMBIGUOUS_COLOR,
        alpha=0.20,
        linewidths=0,
        label="Ambiguous",
        rasterized=True,
    )
    axis.scatter(
        embedding[background, 0],
        embedding[background, 1],
        s=11,
        c=BACKGROUND_COLOR,
        alpha=0.68,
        edgecolors="#163B50",
        linewidths=0.18,
        label="Background",
        rasterized=True,
    )
    axis.scatter(
        embedding[foreground, 0],
        embedding[foreground, 1],
        s=18,
        c=FOREGROUND_COLOR,
        alpha=0.88,
        edgecolors="#6E1B12",
        linewidths=0.25,
        label="Foreground",
        rasterized=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def binary_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "ap": float(average_precision_score(labels, score)),
        "auroc": float(roc_auc_score(labels, score)),
    }


def main() -> None:
    args = parse_args()
    core_path = Path(args.core_root).resolve() / args.dataset / f"{args.stem}.pt"
    output = Path(args.out_dir).resolve()
    if MAIN_ROOT.resolve() in output.parents or output == MAIN_ROOT.resolve():
        raise ValueError("Visualization output must stay outside the source repository main directory")
    output.mkdir(parents=True, exist_ok=True)

    core = torch.load(core_path, map_location="cpu", weights_only=False)
    if str(core["dataset"]) != args.dataset or str(core["stem"]) != args.stem:
        raise RuntimeError(f"Core identity mismatch: {core_path}")
    feature = load_feature(core["source_feature_path"])
    r8 = core["results"]["r8"]
    mean = torch.as_tensor(r8["mean"]).detach().cpu().float().reshape(1, -1)
    basis = torch.as_tensor(r8["basis"]).detach().cpu().float()
    if tuple(basis.shape) != (feature.shape[1], 8):
        raise ValueError(f"Expected rank-8 basis, got {tuple(basis.shape)}")
    centered = feature - mean
    projected = (centered @ basis) @ basis.t()
    residual = centered - projected
    projection_energy = projected.square().sum(dim=1)
    residual_energy = residual.square().sum(dim=1)
    cached_residual = torch.as_tensor(r8["absolute_raw"]).detach().cpu().float().reshape(-1)
    reproduction_error = float((residual_energy - cached_residual).abs().max())
    if reproduction_error > 2e-6:
        raise RuntimeError(f"GBSP residual reproduction failed: {reproduction_error}")

    occupancy = patch_occupancy(core["gt_path"])
    background = occupancy <= float(args.background_occupancy_max)
    foreground = occupancy >= float(args.foreground_occupancy_min)
    ambiguous = ~(background | foreground)
    valid = background | foreground
    labels = foreground[valid].astype(np.uint8)
    if not labels.any() or labels.all():
        raise RuntimeError("Confident patch labels must contain both classes")

    raw_np = feature.numpy()
    residual_np = residual.numpy()
    raw_pca, raw_tsne, raw_ratio = fit_embeddings(
        raw_np, args.seed, args.tsne_perplexity
    )
    residual_pca, residual_tsne, residual_ratio = fit_embeddings(
        residual_np, args.seed, args.tsne_perplexity
    )
    pca_separability = {
        "raw_silhouette": float(silhouette_score(raw_pca[valid], labels)),
        "residual_silhouette": float(silhouette_score(residual_pca[valid], labels)),
    }
    pca_separability["silhouette_gain"] = (
        pca_separability["residual_silhouette"]
        - pca_separability["raw_silhouette"]
    )

    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.2))
    with Image.open(core["image_path"]) as image:
        axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")
    with Image.open(core["gt_path"]) as image:
        axes[1].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("GT (coloring/ranking only)", fontsize=11)
    axes[1].axis("off")
    scatter_classes(axes[2], raw_pca, background, foreground, ambiguous)
    axes[2].set_title(
        "L2-normalized DINOv1-S/8 — PCA\n"
        f"silhouette={pca_separability['raw_silhouette']:.3f}; "
        f"variance={raw_ratio.sum():.1%}",
        fontsize=11,
    )
    scatter_classes(axes[3], residual_pca, background, foreground, ambiguous)
    axes[3].set_title(
        "GBSP-R8 residual vector — PCA\n"
        f"silhouette={pca_separability['residual_silhouette']:.3f}; "
        f"variance={residual_ratio.sum():.1%}",
        fontsize=11,
    )
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{args.dataset}/{args.stem}: raw-feature overlap versus residual separation\n"
        "PCA is unsupervised; GT is used only after projection for colors and metrics",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.91))
    save_figure(fig, output / "input_gt_raw_residual_pca")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    scatter_classes(axes[0], raw_pca, background, foreground, ambiguous)
    axes[0].set_title(
        "L2-normalized DINOv1-S/8 — PCA\n"
        f"silhouette={pca_separability['raw_silhouette']:.3f}; "
        f"variance={raw_ratio.sum():.1%}",
        fontsize=11,
    )
    scatter_classes(axes[1], residual_pca, background, foreground, ambiguous)
    axes[1].set_title(
        "GBSP-R8 residual vector — PCA\n"
        f"silhouette={pca_separability['residual_silhouette']:.3f}; "
        f"variance={residual_ratio.sum():.1%}",
        fontsize=11,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{args.dataset}/{args.stem}\n"
        "GT-colored patches; GT is not used to fit either PCA",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.91))
    save_figure(fig, output / "raw_vs_residual_pca")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.0))
    panels = (
        (raw_pca, f"Raw DINO — PCA\nPC1+PC2 variance={raw_ratio.sum():.1%}"),
        (residual_pca, f"Residual vector — PCA\nPC1+PC2 variance={residual_ratio.sum():.1%}"),
        (raw_tsne, "Raw DINO — t-SNE"),
        (residual_tsne, "Residual vector — t-SNE"),
    )
    for axis, (embedding, title) in zip(axes.reshape(-1), panels):
        scatter_classes(axis, embedding, background, foreground, ambiguous)
        axis.set_title(title, fontsize=11)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{args.dataset}/{args.stem}: GT-colored patch embeddings\n"
        "Embedding is unsupervised; GT is used only for point colors",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.95))
    save_figure(fig, output / "raw_vs_residual_pca_tsne")
    plt.close(fig)

    projection_np = projection_energy.numpy()
    residual_np_energy = residual_energy.numpy()
    fig, axis = plt.subplots(1, 1, figsize=(6.6, 5.4))
    mechanism_embedding = np.column_stack((projection_np, residual_np_energy))
    scatter_classes(axis, mechanism_embedding, background, foreground, ambiguous)
    axis.set_xlabel(r"Background-subspace energy  $\|UU^T(x-\mu)\|_2^2$")
    axis.set_ylabel(r"GBSP residual energy  $\|(I-UU^T)(x-\mu)\|_2^2$")
    axis.set_title("Deterministic GBSP mechanism plane", fontsize=12)
    handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(handles, legend_labels, loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    save_figure(fig, output / "gbsp_mechanism_plane")
    plt.close(fig)

    direct = compute_similarity_baselines(
        feature,
        torch.as_tensor(r8["background_indices"]),
        k=8,
    ).scores["knn8_cos"].detach().cpu().numpy()
    patch_metrics = {
        "direct_dino_knn8": binary_metrics(labels, direct[valid]),
        "gbsp_r8_residual": binary_metrics(labels, residual_np_energy[valid]),
    }
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 4.8))
    bins = np.linspace(
        float(residual_np_energy[valid].min()),
        float(residual_np_energy[valid].max()),
        45,
    )
    axis.hist(
        residual_np_energy[background],
        bins=bins,
        density=True,
        alpha=0.60,
        color=BACKGROUND_COLOR,
        label=f"Background (n={int(background.sum())})",
    )
    axis.hist(
        residual_np_energy[foreground],
        bins=bins,
        density=True,
        alpha=0.72,
        color=FOREGROUND_COLOR,
        label=f"Foreground (n={int(foreground.sum())})",
    )
    axis.set_xlabel(r"GBSP residual energy  $\|e_i\|_2^2$")
    axis.set_ylabel("Density")
    axis.set_title(
        "Full-dimensional residual-score separation\n"
        f"Patch AP={patch_metrics['gbsp_r8_residual']['ap']:.3f}, "
        f"AUROC={patch_metrics['gbsp_r8_residual']['auroc']:.3f}"
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "gbsp_residual_score_distribution")
    plt.close(fig)

    metadata = {
        "schema": "gbsp_feature_separability_case_v1",
        "dataset": args.dataset,
        "stem": args.stem,
        "core_path": str(core_path),
        "feature_path": str(core["source_feature_path"]),
        "gt_path": str(core["gt_path"]),
        "raw_representation": "per-patch L2-normalized DINOv1-S/8 descriptor, 384-D",
        "residual_representation": "e=(I-UU^T)(x-mu), 384-D",
        "formal_score": "squared L2 norm of residual vector",
        "formal_score_reproduction_max_abs_error": reproduction_error,
        "rank": int(r8["selected_rank"]),
        "background_candidate_count": int(torch.as_tensor(r8["background_indices"]).numel()),
        "patch_label_protocol": {
            "gt_resize": "native nearest to 296, average-pool 8 to 37x37 occupancy",
            "background_occupancy_max": float(args.background_occupancy_max),
            "foreground_occupancy_min": float(args.foreground_occupancy_min),
            "background_count": int(background.sum()),
            "foreground_count": int(foreground.sum()),
            "ambiguous_count": int(ambiguous.sum()),
            "gt_role": "visual coloring and post-hoc metrics only",
        },
        "embedding": {
            "fit_points": PATCH_COUNT,
            "gt_used_in_fit": False,
            "pca_fit": "separate unsupervised PCA per representation",
            "raw_pca_variance_ratio": raw_ratio.tolist(),
            "residual_pca_variance_ratio": residual_ratio.tolist(),
            "tsne_fit": "separate t-SNE per representation",
            "tsne_perplexity": float(args.tsne_perplexity),
            "tsne_seed": int(args.seed),
            "warning": "separate t-SNE coordinate systems are qualitative and not metrically comparable",
        },
        "pca_separability_confident_gt_only": pca_separability,
        "patch_metrics_confident_gt_only": patch_metrics,
        "training_run": False,
        "dino_forward_run": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "residual_reproduction_error": reproduction_error,
                "patch_metrics": patch_metrics,
                "files": sorted(path.name for path in output.iterdir()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
