#!/usr/bin/env python3
"""Deterministically render the four required pair-ranking case types."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gbsp_knn_lsr_common import (
    index_manifest,
    load_manifest,
    load_patch_area,
    load_torch,
    minmax,
    normalize_dataset,
    patch_labels,
    score_from_payload,
    score_path,
    write_csv,
)


CASES = {
    "A": lambda k, g, l: k & ~g & l,
    "B": lambda k, g, l: ~k & g & l,
    "C": lambda k, g, l: k & ~g & ~l,
    "D": lambda k, g, l: ~k & ~g & ~l,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_root", required=True)
    parser.add_argument("--knn8_root", required=True)
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _feature(payload: dict, path: Path) -> torch.Tensor:
    for key in ("patch_tokens", "features", "tensor"):
        if torch.is_tensor(payload.get(key)):
            value = payload[key]
            break
    else:
        values = [value for value in payload.values() if torch.is_tensor(value) and value.numel() == 384 * 1369]
        if len(values) != 1:
            raise KeyError(f"cannot resolve feature tensor: {path}")
        value = values[0]
    value = value.squeeze()
    if value.ndim == 3:
        value = value.reshape(value.shape[0], -1).T
    return F.normalize(value.float(), dim=1, eps=1e-12)


def _find_pair(knn: np.ndarray, gbsp: np.ndarray, local: np.ndarray, labels: np.ndarray, case: str):
    fg, bg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    for foreground in fg:
        k = knn[foreground] > knn[bg]
        g = gbsp[foreground] > gbsp[bg]
        l = local[foreground] > local[bg]
        matches = np.flatnonzero(CASES[case](k, g, l))
        if matches.size:
            return int(foreground), int(bg[matches[0]])
    return None


def _render_maps(item: dict, output: Path, threshold: float) -> None:
    rgb = np.asarray(Image.open(item["image_path"]).convert("RGB"))
    gt = np.asarray(Image.open(item["gt_path"]).convert("L")) / 255.0
    bc = np.zeros(1369, dtype=np.float32)
    bc[item["background_indices"]] = 1.0
    scores = item["scores"]
    panels = [
        (rgb, "RGB"), (gt, "GT"), (bc.reshape(37, 37), "Full BC"),
        (scores["knn8"].reshape(37, 37), "KNN8 score"),
        (scores["gbsp"].reshape(37, 37), "Global GBSP residual"),
        (scores["local"].reshape(37, 37), "LSR-K16-R4 residual"),
        ((minmax(scores["knn8"]).numpy() > threshold).reshape(37, 37), f"KNN8 fixed-{threshold:.2f}"),
        ((minmax(scores["gbsp"]).numpy() > threshold).reshape(37, 37), f"GBSP fixed-{threshold:.2f}"),
        ((minmax(scores["local"]).numpy() > threshold).reshape(37, 37), f"LSR fixed-{threshold:.2f}"),
    ]
    figure, axes = plt.subplots(2, 5, figsize=(16, 7))
    for axis, panel in zip(axes.flat, panels):
        axis.imshow(panel[0], cmap=None if panel[0].ndim == 3 else "gray")
        axis.set_title(panel[1]); axis.axis("off")
    axes.flat[-1].axis("off")
    figure.suptitle(
        f"Case {item['case']} | {item['dataset']}/{item['stem']} | "
        f"FG={item['foreground_query']} BG={item['background_query']}"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _crop(image: Image.Image, index: int) -> Image.Image:
    row, col = divmod(int(index), 37)
    width, height = image.size
    left, right = int(np.floor(col * width / 37)), int(np.ceil((col + 1) * width / 37))
    top, bottom = int(np.floor(row * height / 37)), int(np.ceil((row + 1) * height / 37))
    return image.crop((left, top, max(right, left + 1), max(bottom, top + 1))).resize((96, 96))


def _render_features(item: dict, output: Path) -> None:
    local_payload = item["local_payload"]
    core = load_torch(local_payload["source_core_path"])
    feature_path = Path(local_payload["source_feature_path"])
    feature = _feature(load_torch(feature_path), feature_path)
    query_index = item["foreground_query"]
    neighbors = local_payload["variants"]["k16_r4"]["knn_indices"][query_index].long()
    local_feature = feature[neighbors]
    local_mean = local_feature.mean(0)
    _, _, vh = torch.linalg.svd(local_feature - local_mean, full_matrices=False)
    local_basis = vh[:4].T
    global_mean = core["results"]["r8"]["mean"].float()
    global_basis = core["results"]["r8"]["basis"].float()
    query = feature[query_index]
    global_projection = global_mean + global_basis @ (global_basis.T @ (query - global_mean))
    local_projection = local_mean + local_basis @ (local_basis.T @ (query - local_mean))

    image = Image.open(item["image_path"]).convert("RGB")
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(4, 6)
    axis = figure.add_subplot(grid[0, 0])
    axis.imshow(_crop(image, query_index)); axis.set_title(f"Query {query_index}"); axis.axis("off")
    for offset, neighbor in enumerate(neighbors.tolist()[:16], 1):
        axis = figure.add_subplot(grid[offset // 6, offset % 6])
        axis.imshow(_crop(image, neighbor)); axis.set_title(f"N{offset}: {neighbor}"); axis.axis("off")

    global_axis = figure.add_subplot(grid[3, 0:2])
    global_points = (local_feature - global_mean) @ global_basis[:, :2]
    global_query = (query - global_mean) @ global_basis[:, :2]
    global_axis.scatter(global_points[:, 0], global_points[:, 1], s=24, label="Top-K BG")
    global_axis.scatter([global_query[0]], [global_query[1]], marker="*", s=160, label="Query")
    global_axis.set_title("Global PCA first two directions"); global_axis.legend()

    local_axis = figure.add_subplot(grid[3, 2:4])
    local_points = (local_feature - local_mean) @ local_basis[:, :2]
    local_query = (query - local_mean) @ local_basis[:, :2]
    local_axis.scatter(local_points[:, 0], local_points[:, 1], s=24, label="Top-K BG")
    local_axis.scatter([local_query[0]], [local_query[1]], marker="*", s=160, label="Query")
    local_axis.set_title("Local PCA first two directions"); local_axis.legend()

    norm_axis = figure.add_subplot(grid[3, 4:6])
    names = ("Global proj", "Local proj", "Global residual", "Local residual")
    values = (
        float((global_projection - global_mean).norm()), float((local_projection - local_mean).norm()),
        float((query - global_projection).norm()), float((query - local_projection).norm()),
    )
    norm_axis.bar(np.arange(4), values)
    norm_axis.set_xticks(np.arange(4), names, rotation=30, ha="right")
    norm_axis.set_title("Projection / residual vector norms")
    figure.suptitle(f"Case {item['case']} feature audit | {item['dataset']}/{item['stem']}")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = Path(args.out_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    local_rows = load_manifest(args.local_root, split=args.split, max_samples=args.max_samples)
    knn_index = index_manifest(load_manifest(args.knn8_root, score=True, split=args.split))
    gbsp_index = index_manifest(load_manifest(args.gbsp_root, score=True, split=args.split))
    selected = {}
    for row in local_rows:
        dataset, stem = normalize_dataset(row["dataset"]), row["stem"]
        key = (dataset, stem)
        local_payload = load_torch(score_path(row))
        knn_payload = load_torch(score_path(knn_index[key]))
        gbsp_row = gbsp_index[key]
        gbsp_payload = knn_payload if score_path(knn_index[key]).resolve() == score_path(gbsp_row).resolve() else load_torch(score_path(gbsp_row))
        scores = {
            "knn8": score_from_payload(knn_payload, "knn8").numpy(),
            "gbsp": score_from_payload(gbsp_payload, "gbsp").numpy(),
            "local": score_from_payload(local_payload, "k16_r4").numpy(),
        }
        labels, _ = patch_labels(load_patch_area(row["gt_path"]), "main_0.5")
        for case in CASES:
            if case in selected:
                continue
            match = _find_pair(scores["knn8"], scores["gbsp"], scores["local"], labels, case)
            if match is not None:
                selected[case] = {
                    "case": case, "dataset": dataset, "stem": stem,
                    "foreground_query": match[0], "background_query": match[1],
                    "image_path": row["image_path"], "gt_path": row["gt_path"],
                    "background_indices": local_payload["full_bc_background_indices"].long().numpy(),
                    "scores": scores, "local_payload": local_payload,
                }
        if len(selected) == len(CASES):
            break
    rows = []
    for case, item in selected.items():
        _render_maps(item, output / f"case_{case}_maps.png", args.fixed_threshold)
        _render_features(item, output / f"case_{case}_feature_space.png")
        rows.append({key: item[key] for key in ("case", "dataset", "stem", "foreground_query", "background_query", "image_path", "gt_path")})
    write_csv(output / "visual_cases.csv", rows)
    print(f"rendered {len(rows)}/4 deterministic case types")


if __name__ == "__main__":
    main()
