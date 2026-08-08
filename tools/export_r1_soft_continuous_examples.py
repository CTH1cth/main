#!/usr/bin/env python3
"""Export examples where R1 reconstruction improves continuous pixel AP.

The comparison isolates feature-only reconstruction from soft background
similarity.  No binary mask or fixed threshold is used anywhere in selection
or visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import torch_load  # noqa: E402


DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest(path: Path) -> dict[tuple[str, str], dict]:
    rows = _read_jsonl(path)
    return {(str(row["dataset"]), str(row["stem"])): row for row in rows}


def _metric_pairs(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["stem"])][row["method"]] = row
    output = []
    for (dataset, stem), methods in grouped.items():
        if not {"feature_only_reconstruction", "soft_similarity"} <= methods.keys():
            continue
        reconstruction_ap = float(methods["feature_only_reconstruction"]["pixel_AP"])
        soft_ap = float(methods["soft_similarity"]["pixel_AP"])
        output.append(
            {
                "dataset": dataset,
                "stem": stem,
                "r1_reconstruction_ap": reconstruction_ap,
                "soft_similarity_ap": soft_ap,
                "ap_gain": reconstruction_ap - soft_ap,
            }
        )
    return output


def _correlations(path: Path) -> dict[tuple[str, str], dict]:
    output = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if (
            row["candidate"] == "feature_only_reconstruction"
            and row["reference"] == "soft_similarity"
        ):
            output[(row["dataset"], row["stem"])] = {
                "pearson": float(row["pearson_37"]),
                "spearman": float(row["spearman_37"]),
            }
    return output


def _load_score(path: str, height: int, width: int) -> np.ndarray:
    payload = torch_load(Path(path), map_location="cpu")
    score = payload.get("normalized_score_37")
    if not torch.is_tensor(score) or tuple(score.shape) != (1, 37, 37):
        raise ValueError(f"normalized_score_37 missing or malformed: {path}")
    score = F.interpolate(
        score.float().unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    return np.clip(score, 0.0, 1.0)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _gray(value: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    return _fit(image, size)


def _signed_color(value: np.ndarray, size: tuple[int, int], positive: tuple[int, int, int], negative: tuple[int, int, int]) -> Image.Image:
    scale = float(np.quantile(np.abs(value), 0.99))
    scale = max(scale, 1e-8)
    normalized = np.clip(value / scale, -1.0, 1.0)
    rgb = np.zeros((*value.shape, 3), dtype=np.float32)
    positive_mask = normalized >= 0
    for channel in range(3):
        rgb[..., channel] = np.where(
            positive_mask,
            normalized * positive[channel],
            -normalized * negative[channel],
        )
    return _fit(Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB"), size)


def _percentile_rank(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(-1)
    ranks = rankdata(flat, method="average")
    if flat.size <= 1:
        return np.zeros_like(value, dtype=np.float32)
    return ((ranks - 1.0) / (flat.size - 1.0)).reshape(value.shape).astype(np.float32)


def _labeled(panel: Image.Image, label: str, width: int, label_height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, panel.height + label_height), "white")
    canvas.paste(panel, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), label, fill="black", font=ImageFont.load_default())
    return canvas


def _render(row: dict, feature_row: dict, soft_row: dict, tile: int) -> tuple[Image.Image, dict]:
    with Image.open(feature_row["image_path"]) as opened:
        original = opened.convert("RGB")
    with Image.open(feature_row["gt_path"]) as opened:
        gt_image = opened.convert("L")
    height, width = gt_image.height, gt_image.width
    gt = np.asarray(gt_image, dtype=np.float32) / 255.0 > 0.5
    reconstruction = _load_score(feature_row["score_path"], height, width)
    soft = _load_score(soft_row["score_path"], height, width)
    rank_gain = _percentile_rank(reconstruction) - _percentile_rank(soft)
    # Positive means a GT foreground pixel was raised, or a GT background
    # pixel was lowered, by reconstruction.  This is evaluation-only display.
    beneficial_rank_change = np.where(gt, rank_gain, -rank_gain)
    size = (tile, tile)
    panels = (
        _fit(original, size),
        _fit(gt_image, size),
        _gray(soft, size),
        _gray(reconstruction, size),
        _signed_color(reconstruction - soft, size, (255, 55, 30), (30, 120, 255)),
        _signed_color(beneficial_rank_change, size, (30, 220, 70), (220, 30, 180)),
    )
    labels = (
        "Image",
        "GT (evaluation only)",
        f"Soft similarity | AP {row['soft_similarity_ap']:.4f}",
        f"R1 feature reconstruction | AP {row['r1_reconstruction_ap']:.4f}",
        "Score delta | red: R1 higher, blue: Soft higher",
        "Rank effect | green: beneficial, magenta: harmful",
    )
    label_height = 30
    labeled = [_labeled(panel, label, tile, label_height) for panel, label in zip(panels, labels)]
    header_height = 36
    canvas = Image.new("RGB", (tile * len(labeled), labeled[0].height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 10),
        f"{row['dataset']} / {row['stem']} | continuous Pixel AP gain = {row['ap_gain']:+.4f}",
        fill="black",
        font=ImageFont.load_default(),
    )
    for index, panel in enumerate(labeled):
        canvas.paste(panel, (index * tile, header_height))
    diagnostics = {
        **row,
        "mean_abs_score_delta": float(np.mean(np.abs(reconstruction - soft))),
        "mean_abs_rank_delta": float(np.mean(np.abs(rank_gain))),
        "gt_mean_beneficial_rank_change": float(np.mean(beneficial_rank_change)),
        "image_path": feature_row["image_path"],
        "gt_path": feature_row["gt_path"],
        "r1_score_path": feature_row["score_path"],
        "soft_score_path": soft_row["score_path"],
    }
    return canvas, diagnostics


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = _manifest(Path(args.feature_manifest))
    soft_manifest = _manifest(Path(args.soft_manifest))
    rows = _metric_pairs(Path(args.metrics))
    correlations = _correlations(Path(args.correlations))
    for row in rows:
        correlation = correlations[(row["dataset"], row["stem"])]
        row.update(correlation)
        row["visible_gain_score"] = row["ap_gain"] * (1.0 - row["spearman"])
    selected = []
    for dataset in DATASETS:
        candidates = sorted(
            (row for row in rows if row["dataset"] == dataset and row["ap_gain"] > 0),
            key=lambda row: (
                row["visible_gain_score"] if args.prefer_visible else row["ap_gain"]
            ),
            reverse=True,
        )
        selected.extend(candidates[: args.per_dataset])
    if not selected:
        raise RuntimeError("no positive continuous AP examples found")

    rendered = []
    diagnostics = []
    for index, row in enumerate(selected, 1):
        key = (row["dataset"], row["stem"])
        panel, diagnostic = _render(row, feature_manifest[key], soft_manifest[key], args.tile)
        filename = f"{index:02d}_{row['dataset']}_{row['stem']}.png"
        panel.save(output_dir / filename)
        diagnostic["comparison_path"] = str(output_dir / filename)
        diagnostics.append(diagnostic)
        rendered.append(panel)

    overview = Image.new(
        "RGB",
        (max(panel.width for panel in rendered), sum(panel.height for panel in rendered)),
        "white",
    )
    y = 0
    for panel in rendered:
        overview.paste(panel, (0, y))
        y += panel.height
    overview.save(output_dir / "R1_SOFT_CONTINUOUS_TOP8.png")

    fields = list(diagnostics[0])
    with (output_dir / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    lines = [
        "# R1 reconstruction vs Soft Similarity: continuous-response examples",
        "",
        "Selection uses per-image continuous Pixel AP gain only; no binary threshold is used.",
        "The final panel is GT-aware and is for interpretation/evaluation only.",
        "",
        "| Dataset | Image | Soft AP | R1 reconstruction AP | Gain |",
        "|---|---|---:|---:|---:|",
        *(
            f"| {row['dataset']} | {row['stem']} | {row['soft_similarity_ap']:.4f} | "
            f"{row['r1_reconstruction_ap']:.4f} | {row['ap_gain']:+.4f} |"
            for row in diagnostics
        ),
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_examples": len(diagnostics)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="../workdir/22/diagnostics/per_image_metrics.csv",
    )
    parser.add_argument(
        "--feature_manifest",
        default="../workdir/22/feature_only_reconstruction/manifest.jsonl",
    )
    parser.add_argument(
        "--soft_manifest",
        default="../workdir/22/soft_similarity/manifest.jsonl",
    )
    parser.add_argument(
        "--correlations",
        default="../workdir/22/diagnostics/score_correlations.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="../workdir/r1_soft_continuous_examples",
    )
    parser.add_argument("--per_dataset", type=int, default=2)
    parser.add_argument("--tile", type=int, default=260)
    parser.add_argument(
        "--prefer_visible",
        action="store_true",
        help="prefer AP gains accompanied by larger continuous rank changes",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
