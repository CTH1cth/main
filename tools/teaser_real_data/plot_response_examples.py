#!/usr/bin/env python3
"""Create seed-frozen real response examples without performance-based selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import (  # noqa: E402
    load_native_images, load_npz, load_score_rows, load_settings,
    native_hard_mask, native_response, save_vector_figure, write_json,
)

CMAP = "magma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=8)
    return parser.parse_args()


def _frozen_ids(config: Path) -> list[str] | None:
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    value = raw.get("frozen_example_ids")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = config.resolve().parents[2] / path
    if not path.is_file():
        return None
    return [line.strip().split("/")[-1] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_examples(config: Path, rows: list[dict], dataset: str, seed: int, count: int) -> tuple[list[dict], str]:
    candidates = sorted([row for row in rows if row["dataset"] == dataset], key=lambda row: row["stem"])
    ids = _frozen_ids(config)
    if ids:
        lookup = {row["stem"]: row for row in candidates}
        selected = [lookup[stem] for stem in ids if stem in lookup][:count]
        if len(selected) == count:
            return selected, "pre-existing frozen Fig.3 ID list; no score/GT-based selection"
    rng = np.random.default_rng(seed)
    take = min(count, len(candidates))
    positions = rng.choice(len(candidates), size=take, replace=False)
    return [candidates[int(index)] for index in positions], f"uniform random CAMO sample, seed={seed}; no score/GT-based selection"


def _save_gray(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8), mode="L").save(path)


def _save_response(path: Path, value: np.ndarray) -> None:
    plt.imsave(path, value, cmap=CMAP, vmin=0.0, vmax=1.0)


def plot(config: str | Path, score_root: str | Path, out_dir: str | Path, count: int = 8) -> list[dict]:
    config = Path(config).resolve()
    settings = load_settings(config)
    rows = load_score_rows(score_root)
    selected, protocol = select_examples(config, rows, settings.main_dataset, settings.random_seed, count)
    out = Path(out_dir) / "response_examples"
    out.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in selected:
        with load_npz(row) as payload:
            rgb, gt = load_native_images(row["image_path"], row["gt_path"])
            height, width = rgb.shape[:2]
            if gt.shape != (height, width):
                gt = np.asarray(Image.fromarray(gt).resize((width, height), Image.Resampling.NEAREST))
            knn37 = payload["knn8_foreground_score_normalized"]
            gbsp37 = payload["gbsp_normalized_score"]
            knn = native_response(knn37, (height, width))
            gbsp = native_response(gbsp37, (height, width))
            knn_mask = native_hard_mask(knn37, (height, width), settings.hard_threshold)
            gbsp_mask = native_hard_mask(gbsp37, (height, width), settings.hard_threshold)
            sample_dir = out / row["stem"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(sample_dir / "rgb.png")
            _save_gray(sample_dir / "gt.png", gt)
            _save_response(sample_dir / "knn8_response.png", knn)
            _save_response(sample_dir / "gbsp_response.png", gbsp)
            _save_gray(sample_dir / "knn8_mask.png", knn_mask)
            _save_gray(sample_dir / "gbsp_mask.png", gbsp_mask)
            pair = {
                "fg_index": int(payload["pair_fg_index"]),
                "bg_index": int(payload["pair_bg_index"]),
                "cosine_similarity": float(payload["pair_cosine_similarity"]),
            }
            write_json(sample_dir / "patch_pair.json", pair)
            rendered.append({
                "dataset": row["dataset"], "stem": row["stem"], "directory": str(sample_dir),
                "rgb": rgb, "gt": gt, "knn": knn, "gbsp": gbsp,
                "knn_mask": knn_mask, "gbsp_mask": gbsp_mask,
                "pair_cosine_similarity": pair["cosine_similarity"],
            })
    metadata = {
        "dataset": settings.main_dataset, "selection_protocol": protocol,
        "seed": settings.random_seed, "requested": count,
        "selected": [item["stem"] for item in rendered],
        "performance_cherry_picking": False,
        "response_cmap": CMAP, "response_vmin": 0.0, "response_vmax": 1.0,
        "mask_threshold": settings.hard_threshold,
    }
    write_json(out / "selection_protocol.json", metadata)
    (out / "SELECTION_PROTOCOL.md").write_text(
        "# Response Example Selection\n\n"
        f"- Dataset: `{settings.main_dataset}`\n"
        f"- Protocol: {protocol}\n"
        f"- Selected IDs: {', '.join(metadata['selected'])}\n"
        "- GT and method scores were not used to rank or choose examples.\n",
        encoding="utf-8",
    )
    if rendered:
        fig, axes = plt.subplots(len(rendered), 6, figsize=(12.0, 2.0 * len(rendered)), squeeze=False)
        titles = ("RGB", "GT", "KNN8 response", "GBSP response", "KNN8 mask", "GBSP mask")
        for row_index, item in enumerate(rendered):
            values = (item["rgb"], item["gt"], item["knn"], item["gbsp"], item["knn_mask"], item["gbsp_mask"])
            for col_index, value in enumerate(values):
                ax = axes[row_index, col_index]
                if col_index in (2, 3):
                    ax.imshow(value, cmap=CMAP, vmin=0.0, vmax=1.0)
                elif col_index in (1, 4, 5):
                    ax.imshow(value, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(value)
                if row_index == 0:
                    ax.set_title(titles[col_index], fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(item["stem"], fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Seed-frozen CAMO response examples", fontsize=12)
        fig.tight_layout()
        save_vector_figure(fig, out / "contact_sheet")
        plt.close(fig)

        item = rendered[0]
        fig, axes = plt.subplots(1, 6, figsize=(13.2, 2.5))
        values = (item["rgb"], item["gt"], item["knn"], item["gbsp"], item["knn_mask"], item["gbsp_mask"])
        for index, value in enumerate(values):
            if index in (2, 3):
                axes[index].imshow(value, cmap=CMAP, vmin=0.0, vmax=1.0)
            elif index in (1, 4, 5):
                axes[index].imshow(value, cmap="gray", vmin=0, vmax=255)
            else:
                axes[index].imshow(value)
            axes[index].set_title(titles[index], fontsize=9)
            axes[index].axis("off")
        fig.suptitle(f"Frozen high-resolution example: {item['stem']}", fontsize=11)
        fig.tight_layout()
        save_vector_figure(fig, out / "selected_example_high_resolution")
        plt.close(fig)
    return [{key: value for key, value in item.items() if not isinstance(value, np.ndarray)} for item in rendered]


def main() -> None:
    args = parse_args()
    print(json.dumps(plot(args.config, args.score_root, args.out_dir, args.num_examples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
