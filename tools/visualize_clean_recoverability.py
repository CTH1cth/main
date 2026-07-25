#!/usr/bin/env python3
"""Offline GT-allowed visualization of cached continuous recoverability fields."""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.utils import (  # noqa: E402
    build_image_items,
    dabe_clean_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


FIELDS = (
    ("Clean-DP", "target_dp_68"),
    ("Foreground evidence F", "foreground_evidence_37"),
    ("p_rw", "p_rw_37"),
    ("latent_rw", "latent_rw_37"),
    ("Background evidence B", "background_evidence_37"),
    ("DINO semantic tendency H", "semantic_fg_tendency_37"),
    ("Recoverability A_rec", "recoverability_37"),
)


def _prepare_output(path):
    path = Path(path).resolve()
    if path == MAIN_ROOT or MAIN_ROOT in path.parents:
        raise RuntimeError("Visualization output must remain outside the repository.")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Visualization output is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _field(payload, key, size=(256, 256)):
    value = payload.get(key)
    if not torch.is_tensor(value) or value.ndim != 3:
        raise RuntimeError(f"Missing visualization tensor {key!r}.")
    value = value.detach().cpu().float().unsqueeze(0)
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"Visualization tensor {key} contains NaN/Inf.")
    value = F.interpolate(
        value, size=size, mode="bilinear", align_corners=False
    ).squeeze().clamp(0.0, 1.0)
    return value.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive.")

    cfg = load_config(args.config)
    if str(getattr(cfg, "DABE_CLEAN_VERSION", "")) != "v2_contrec":
        raise RuntimeError("This visualizer requires DABE_CLEAN_VERSION='v2_contrec'.")
    output_root = args.output_root or str(
        Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "recoverability_vis_32"
    )
    output_root = _prepare_output(output_root)
    manifest = dabe_clean_manifest_path(cfg)
    row_map = manifest_to_map(read_jsonl(manifest), manifest)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=True)
    if args.num_samples > len(items):
        parser.error(
            f"--num-samples={args.num_samples} exceeds dataset size {len(items)}."
        )
    rng = random.Random(int(args.seed))
    selected = rng.sample(items, args.num_samples)
    outputs = []
    for item in selected:
        key = (item["dataset"], item["stem"])
        if key not in row_map:
            raise RuntimeError(f"Recoverability manifest is missing {key}.")
        payload = torch_load(row_map[key]["cache_path"], map_location="cpu")
        if str(payload.get("version")) != "dabe_clean_v2_contrec":
            raise RuntimeError(f"Unexpected cache version for {key}.")
        rgb = np.asarray(
            Image.open(item["image_path"]).convert("RGB").resize(
                (256, 256), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        gt = np.asarray(
            Image.open(item["gt_path"]).convert("L").resize(
                (256, 256), Image.Resampling.NEAREST
            ),
            dtype=np.float32,
        ) / 255.0
        maps = [(title, _field(payload, field)) for title, field in FIELDS]
        recovery = _field(payload, "recoverability_68")
        heat = plt.get_cmap("turbo")(recovery)[..., :3]
        overlay = np.clip(0.55 * rgb + 0.45 * heat, 0.0, 1.0)
        panels = [("RGB", rgb, None), ("GT (audit only)", gt, "gray")]
        panels.extend((title, value, "gray") for title, value in maps)
        panels.append(("Recoverability overlay", overlay, None))

        figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        for axis, (title, value, cmap) in zip(axes.flat, panels):
            axis.imshow(value, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"{item['dataset']} / {item['stem']}")
        output = output_root / f"{item['dataset']}_{item['stem']}_contrec.png"
        figure.savefig(output, dpi=140)
        plt.close(figure)
        outputs.append(str(output))

    summary = {
        "schema": "dabe_clean_v2_contrec_visualization",
        "config": str(Path(args.config).resolve()),
        "seed": int(args.seed),
        "num_samples": len(outputs),
        "gt_usage": "offline_visualization_only",
        "outputs": outputs,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
