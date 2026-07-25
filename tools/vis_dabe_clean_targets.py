#!/usr/bin/env python3
"""Render compact DABE Bridge/Clean target comparisons."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_cache import parse_bool  # noqa: E402
from common.utils import find_gt_path, manifest_to_map, read_jsonl, torch_load  # noqa: E402


DEFAULT_SOURCE = "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
DEFAULT_BRIDGE = "../datasets/cache/dabe_bridge_v1_pseudo_cache/dinov1-s8"
DEFAULT_CLEAN = "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8"
DEFAULT_DATA = "/home/dell01/CTH/MY-baseline/datasets/COD"
DEFAULT_OUTPUT = "../workdir/dabe_clean_v1_audit/vis"


def _array(payload, key):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Visualization payload is missing {key!r}.")
    value = value.detach().cpu().float()
    if tuple(value.shape) != (1, 68, 68):
        raise RuntimeError(f"{key} must be [1,68,68], got {list(value.shape)}.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} contains NaN/Inf.")
    return value[0].clamp(0.0, 1.0).numpy()


def _signed_rgb(value):
    value = np.asarray(value, dtype=np.float32)
    scale = max(float(np.max(np.abs(value))), 1e-8)
    strength = np.clip(np.abs(value) / scale, 0.0, 1.0)
    rgb = np.full((*value.shape, 3), 0.5, dtype=np.float32)
    positive = value > 0
    negative = value < 0
    rgb[positive, 0] = 0.5 + 0.5 * strength[positive]
    rgb[positive, 1] = 0.5 * (1.0 - strength[positive])
    rgb[positive, 2] = 0.5 * (1.0 - strength[positive])
    rgb[negative, 2] = 0.5 + 0.5 * strength[negative]
    rgb[negative, 0] = 0.5 * (1.0 - strength[negative])
    rgb[negative, 1] = 0.5 * (1.0 - strength[negative])
    return rgb


def _prepare_output(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--bridge-root", default=DEFAULT_BRIDGE)
    parser.add_argument("--clean-root", default=DEFAULT_CLEAN)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--stems", default="")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive.")

    output_root = _prepare_output(args.output_root, args.overwrite)
    source_rows = read_jsonl(Path(args.source_root) / "manifest_train.jsonl")
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
    source_rows = [row for row in source_rows if str(row.get("dataset")) in datasets]
    requested_stems = {item.strip() for item in args.stems.split(",") if item.strip()}
    if requested_stems:
        source_rows = [row for row in source_rows if str(row.get("stem")) in requested_stems]
        found = {str(row["stem"]) for row in source_rows}
        missing = sorted(requested_stems - found)
        if missing:
            raise RuntimeError(f"Requested stems were not found: {missing}.")
    if args.random:
        generator = random.Random(args.seed)
        generator.shuffle(source_rows)
    source_rows = source_rows[: args.max_samples]
    if not source_rows:
        raise RuntimeError("No visualization samples selected.")
    bridge_manifest = Path(args.bridge_root) / "manifest_train.jsonl"
    clean_manifest = Path(args.clean_root) / "manifest_train.jsonl"
    bridge_map = manifest_to_map(read_jsonl(bridge_manifest), bridge_manifest)
    clean_map = manifest_to_map(read_jsonl(clean_manifest), clean_manifest)

    for row in source_rows:
        dataset = str(row["dataset"])
        stem = str(row["stem"])
        key = (dataset, stem)
        if key not in bridge_map or key not in clean_map:
            raise RuntimeError(f"Bridge/Clean cache is missing {key}.")
        source = torch_load(row["cache_path"], map_location="cpu")
        bridge = torch_load(bridge_map[key]["cache_path"], map_location="cpu")
        clean = torch_load(clean_map[key]["cache_path"], map_location="cpu")
        image_path = Path(args.data_root) / dataset / "im"
        candidates = sorted(image_path.glob(f"{stem}.*"))
        if not candidates:
            raise FileNotFoundError(f"RGB image not found for {dataset}/{stem}.")
        rgb = np.asarray(Image.open(candidates[0]).convert("RGB"))
        gt = np.asarray(Image.open(find_gt_path(args.data_root, dataset, stem)).convert("L"))
        p_base = _array(source, "p_base_68")
        background = _array(clean, "background_evidence_68")
        target_soft = _array(source, "target_soft_68")
        weight = _array(source, "weight_map_68")
        old_gradient = weight * (target_soft - 0.5)
        bridge_target = _array(bridge, "bridge_target_68")
        target_dp = _array(clean, "target_dp_68")
        target_diff = _array(clean, "target_diff_68")
        hard = target_dp > 0.5
        neutral = np.abs(target_dp - 0.5) <= 0.1
        panels = [
            ("RGB", rgb, None),
            ("GT", gt, "gray"),
            ("DABE-v2 F", p_base, "gray"),
            ("Background B", background, "gray"),
            ("PU target", target_soft, "gray"),
            ("PU weight", weight, "gray"),
            ("Old grad +", _signed_rgb(np.maximum(old_gradient, 0.0)), None),
            ("Old grad -", _signed_rgb(np.minimum(old_gradient, 0.0)), None),
            ("Bridge", bridge_target, "gray"),
            ("Clean DP", target_dp, "gray"),
            ("Clean Diff", target_diff, "gray"),
            ("DP > .5", hard, "gray"),
            ("Neutral", neutral, "gray"),
        ]
        figure, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.45))
        for axis, (title, value, cmap) in zip(axes, panels):
            axis.imshow(value, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        figure.tight_layout(pad=0.35)
        figure.savefig(output_root / f"{dataset}_{stem}_dabe_clean.png", dpi=150)
        plt.close(figure)

    print("TRAIN_GT_USED_FOR_VISUALIZATION_ONLY=True")
    print("TRAIN_GT_USED_FOR_TRAINING=False")
    print(f"samples={len(source_rows)}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
