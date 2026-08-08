#!/usr/bin/env python3
"""Create 20 transparent GBSP-vs-R1 grayscale response comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_GBSP_MANIFEST = (
    REPO_ROOT
    / "workdir/mbsp_pca_v1/ablations/M1_pilot200/manifest_test.jsonl"
)
DEFAULT_METRICS = REPO_ROOT / "workdir/gbsp_pca_v1_eval/per_sample_metrics.csv"
DEFAULT_OUT = REPO_ROOT / "workdir/gbsp_r1_gray20"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
GBSP_METHOD = "mbsp_abs_minmax"
R1_METHOD = "Current Full R1"
GBSP_THRESHOLD = 0.63
R1_THRESHOLD = 0.55
PANEL_SIZE = 220
TITLE_HEIGHT = 38


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _manifest_map(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    mapping = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapping:
            raise RuntimeError(f"duplicate manifest key: {key}")
        mapping[key] = row
    return rows, mapping


def _metric_pairs(path: Path) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row["method"])
            if method in {GBSP_METHOD, R1_METHOD}:
                grouped[(str(row["dataset"]), str(row["stem"]))][method] = row
    output = {}
    for key, methods in grouped.items():
        if set(methods) != {GBSP_METHOD, R1_METHOD}:
            continue
        gbsp_ap = float(methods[GBSP_METHOD]["pixel_AP"])
        r1_ap = float(methods[R1_METHOD]["pixel_AP"])
        output[key] = {
            "gbsp_ap": gbsp_ap,
            "r1_ap": r1_ap,
            "ap_delta": gbsp_ap - r1_ap,
        }
    return output


def _select_stratified(
    manifest_rows: list[dict], metrics: dict[tuple[str, str], dict]
) -> list[dict]:
    selected = []
    for dataset in DATASETS:
        candidates = []
        for row in manifest_rows:
            key = (str(row["dataset"]), str(row["stem"]))
            if key[0] == dataset and key in metrics:
                candidates.append({**row, **metrics[key]})
        candidates.sort(key=lambda row: (float(row["ap_delta"]), str(row["stem"])))
        if len(candidates) < 5:
            raise RuntimeError(f"{dataset} has fewer than five paired candidates")
        positions = (0, 1, len(candidates) // 2, len(candidates) - 2, len(candidates) - 1)
        categories = (
            "largest_loss_1",
            "largest_loss_2",
            "median_delta",
            "largest_gain_2",
            "largest_gain_1",
        )
        for position, category in zip(positions, categories):
            selected.append({**candidates[position], "selection_category": category})
    return selected


def _load_response(path: str | Path, field: str) -> torch.Tensor:
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"response payload must be a dict: {path}")
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise ValueError(f"{field} escaped [0,1]: {path}")
    return value.clamp(0.0, 1.0)


def _resize_formal(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    lifted = F.interpolate(
        value.unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(
        lifted,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _gray(value: torch.Tensor) -> Image.Image:
    array = value.squeeze(0).numpy()
    return Image.fromarray(
        np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )


def _panel(content: Image.Image, title: str, nearest: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    resized = content.resize((PANEL_SIZE, PANEL_SIZE), resampling).convert("RGB")
    panel = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + TITLE_HEIGHT), "white")
    panel.paste(resized, (0, TITLE_HEIGHT))
    ImageDraw.Draw(panel).multiline_text(
        (4, 3), title, fill="black", font=ImageFont.load_default(), spacing=1
    )
    return panel


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _render_one(index: int, row: dict, output_dir: Path) -> tuple[dict, Image.Image]:
    gbsp_payload = torch_load(row["cache_path"], map_location="cpu")
    source_dabe_path = row.get("source_dabe_path") or gbsp_payload.get(
        "source_dabe_path"
    )
    if not source_dabe_path:
        raise KeyError(f"source_dabe_path missing: {row['dataset']}/{row['stem']}")

    with Image.open(row["image_path"]) as handle:
        image = handle.convert("RGB").copy()
    with Image.open(row["gt_path"]) as handle:
        gt = handle.convert("L").copy()
    width, height = gt.size
    r1 = _resize_formal(
        _load_response(source_dabe_path, "residual_pass1_37"), height, width
    )
    gbsp = _resize_formal(
        _load_response(row["cache_path"], "absolute_minmax"), height, width
    )
    r1_hard = _gray((r1 > R1_THRESHOLD).float())
    gbsp_hard = _gray((gbsp > GBSP_THRESHOLD).float())
    ap_delta = float(row["ap_delta"])
    panels = (
        _panel(image, f"Image\n{row['dataset']}/{row['stem']}"),
        _panel(gt, "GT", nearest=True),
        _panel(_gray(r1), f"R1 gray\nAP={float(row['r1_ap']):.4f}"),
        _panel(r1_hard, f"R1 hard\nstrict > {R1_THRESHOLD:.2f}", nearest=True),
        _panel(
            _gray(gbsp),
            f"GBSP gray\nAP={float(row['gbsp_ap']):.4f} d={ap_delta:+.4f}",
        ),
        _panel(
            gbsp_hard,
            f"GBSP hard\nstrict > {GBSP_THRESHOLD:.2f}",
            nearest=True,
        ),
    )
    comparison = Image.new(
        "RGB",
        (PANEL_SIZE * len(panels), PANEL_SIZE + TITLE_HEIGHT),
        "white",
    )
    for panel_index, panel in enumerate(panels):
        comparison.paste(panel, (panel_index * PANEL_SIZE, 0))
    filename = (
        f"{index:02d}_{_safe_name(str(row['dataset']))}_"
        f"{_safe_name(str(row['stem']))}_{row['selection_category']}.png"
    )
    path = output_dir / "individual" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(path)
    record = {
        "index": index,
        "dataset": row["dataset"],
        "stem": row["stem"],
        "selection_category": row["selection_category"],
        "r1_pixel_AP": float(row["r1_ap"]),
        "gbsp_pixel_AP": float(row["gbsp_ap"]),
        "ap_delta_gbsp_minus_r1": ap_delta,
        "r1_threshold": R1_THRESHOLD,
        "gbsp_threshold": GBSP_THRESHOLD,
        "image_path": row["image_path"],
        "gt_path": row["gt_path"],
        "r1_cache_path": str(source_dabe_path),
        "gbsp_cache_path": row["cache_path"],
        "comparison_path": str(path),
    }
    return record, comparison


def _stack(images: list[Image.Image], path: Path) -> None:
    if not images:
        return
    canvas = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images)),
        "white",
    )
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_outputs(output_dir: Path, records: list[dict]) -> None:
    fields = list(records[0])
    with (output_dir / "selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "selection": (
                    "per dataset: two largest AP losses, median AP delta, "
                    "two largest AP gains"
                ),
                "datasets": list(DATASETS),
                "num_images": len(records),
                "gbsp_field": "absolute_minmax",
                "r1_field": "residual_pass1_37",
                "resize": "37->68->original GT, bilinear, align_corners=False",
                "r1_hard": f"strict > {R1_THRESHOLD}",
                "gbsp_hard": f"strict > {GBSP_THRESHOLD}",
                "training_used": False,
                "dino_forward_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GBSP vs R1：20 张灰度响应对比",
        "",
        "- 每数据集固定选择 AP 差值最差 2 张、中位数 1 张、最好 2 张。",
        "- 响应统一执行 37→68→原始 GT 尺寸双线性上采样。",
        f"- Hard 阈值：R1 strict > {R1_THRESHOLD}；GBSP strict > {GBSP_THRESHOLD}。",
        "- 未执行训练、DINO forward 或伪标签重建。",
        "",
        "| # | dataset | stem | category | R1 AP | GBSP AP | delta | file |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        path = Path(record["comparison_path"])
        lines.append(
            f"| {record['index']} | {record['dataset']} | {record['stem']} | "
            f"{record['selection_category']} | {record['r1_pixel_AP']:.4f} | "
            f"{record['gbsp_pixel_AP']:.4f} | "
            f"{record['ap_delta_gbsp_minus_r1']:+.4f} | "
            f"[image](individual/{path.name}) |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gbsp_manifest", default=str(DEFAULT_GBSP_MANIFEST))
    parser.add_argument("--metrics_csv", default=str(DEFAULT_METRICS))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    manifest_path = _resolve(args.gbsp_manifest)
    metrics_path = _resolve(args.metrics_csv)
    output_dir = _resolve(args.out_dir)
    rows, _ = _manifest_map(manifest_path)
    if len(rows) != 6473:
        raise RuntimeError(f"expected 6473 GBSP manifest rows, got {len(rows)}")
    metrics = _metric_pairs(metrics_path)
    selected = _select_stratified(rows, metrics)
    if len(selected) != 20:
        raise RuntimeError(f"expected exactly 20 selected rows, got {len(selected)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records, comparisons = [], []
    by_dataset: dict[str, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(selected, 1):
        record, comparison = _render_one(index, row, output_dir)
        records.append(record)
        comparisons.append(comparison)
        by_dataset[str(row["dataset"])].append(comparison)
        print(
            f"[{index:02d}/20] {row['dataset']}/{row['stem']} "
            f"AP_delta={float(row['ap_delta']):+.6f}",
            flush=True,
        )
    for dataset in DATASETS:
        _stack(
            by_dataset[dataset],
            output_dir / f"contact_{_safe_name(dataset)}.png",
        )
    _stack(comparisons, output_dir / "contact_all_20.png")
    _write_outputs(output_dir, records)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
