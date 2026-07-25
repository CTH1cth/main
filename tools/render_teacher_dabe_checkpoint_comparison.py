#!/usr/bin/env python3
"""Render fixed random training samples across EMA-Teacher checkpoints.

This is a read-only visualization utility: it consumes cached DINO/DABE data
and checkpoint Teacher weights, never reads GT, and never performs training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from common.dataset import CachedTrainDataset
from common.utils import load_config, torch_load
from model import build_seg_head
from train import (
    extract_logits,
    forward_seg_head,
    make_image_136,
    make_image_68,
    make_model_input,
    make_sobel_68,
    resize_logits_for_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare binary EMA-Teacher predictions with DABE pseudo labels"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def rgb_image(tensor: torch.Tensor, size: int) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def mask_image(tensor: torch.Tensor, size: int) -> Image.Image:
    array = tensor.detach().float().squeeze().clamp(0.0, 1.0).cpu().numpy()
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")
    return image.resize((size, size), resample=Image.Resampling.NEAREST).convert("RGB")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def render_row(
    panels: list[tuple[str, Image.Image]],
    *,
    sample_name: str,
    panel_size: int,
) -> Image.Image:
    title_height = 25
    canvas = Image.new(
        "RGB",
        (len(panels) * panel_size, panel_size + title_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = index * panel_size
        draw.text((x + 4, 5), title, fill=(0, 0, 0))
        canvas.paste(panel, (x, title_height))
    draw.rectangle((0, title_height, panel_size - 1, title_height + 18), fill=(255, 255, 255))
    draw.text((4, title_height + 2), sample_name[:25], fill=(0, 0, 0))
    return canvas


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 1 <= int(args.num_samples) <= 10:
        raise ValueError("--num_samples must be in [1,10]")
    epochs = tuple(dict.fromkeys(int(epoch) for epoch in args.epochs))
    if any(epoch < 1 for epoch in epochs):
        raise ValueError("--epochs must contain positive integers")

    output_dir = resolve(args.output_dir)
    checkpoint_dir = resolve(args.checkpoint_dir)
    prepare_output(output_dir)
    checkpoint_paths = {
        epoch: checkpoint_dir / f"epoch_{epoch:03d}.pth" for epoch in epochs
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    cfg = load_config(resolve(args.config))
    # BITC cache fields are unnecessary for this read-only comparison.  DABE
    # soft targets and the cached DINO feature remain the authoritative inputs.
    cfg.USE_BITC = False
    dataset = CachedTrainDataset(cfg, max_samples=-1)
    if len(dataset) < int(args.num_samples):
        raise RuntimeError(
            f"Dataset has only {len(dataset)} samples, requested {args.num_samples}"
        )
    rng = random.Random(int(args.seed))
    selected_indices = sorted(rng.sample(range(len(dataset)), int(args.num_samples)))
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=int(args.num_samples),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))
    leaked_gt = sorted(
        key
        for key in batch
        if str(key).lower() in {"gt", "mask", "ground_truth", "gt_path", "mask_path"}
    )
    if leaked_gt:
        raise RuntimeError(f"Visualization batch unexpectedly contains GT: {leaked_gt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    image_136 = make_image_136(cfg, batch, device)
    sobel_68 = make_sobel_68(cfg, batch, device)
    teacher_binary_by_epoch: dict[int, torch.Tensor] = {}
    teacher = build_seg_head(dataset.in_channels, cfg).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for epoch, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        if int(checkpoint.get("epoch", -1)) != epoch:
            raise RuntimeError(
                f"Checkpoint epoch mismatch: {checkpoint_path} has {checkpoint.get('epoch')}"
            )
        teacher.load_state_dict(checkpoint["teacher"], strict=True)
        teacher.eval()
        output = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            image_136=image_136,
            sobel_68=sobel_68,
            return_aux=False,
        )
        logits = resize_logits_for_loss(extract_logits(output), cfg)
        teacher_binary_by_epoch[epoch] = (logits.sigmoid() >= 0.5).float().cpu()

    dabe_soft = batch["pu_target_soft"].float().cpu()
    dabe_binary = (dabe_soft >= 0.5).float()
    rgb = batch["image_68"].float().cpu()
    datasets = list(batch["dataset"])
    stems = list(batch["stem"])
    panel_size = 170
    overview_panel_size = 112
    overview_label_width = 220
    overview_header_height = 28
    overview_row_height = overview_panel_size + 22
    titles = ["RGB", "DABE soft", "DABE >= 0.5"] + [
        f"Teacher e{epoch}" for epoch in epochs
    ]
    overview = Image.new(
        "RGB",
        (
            overview_label_width + len(titles) * overview_panel_size,
            overview_header_height + len(selected_indices) * overview_row_height,
        ),
        color=(255, 255, 255),
    )
    overview_draw = ImageDraw.Draw(overview)
    for column, title in enumerate(titles):
        overview_draw.text(
            (overview_label_width + column * overview_panel_size + 4, 7),
            title,
            fill=(0, 0, 0),
        )

    manifest_samples = []
    for batch_index, dataset_index in enumerate(selected_indices):
        sample_name = f"{datasets[batch_index]}__{stems[batch_index]}"
        sample_dir = output_dir / f"{batch_index:02d}_{safe_name(sample_name)}"
        sample_dir.mkdir(parents=True, exist_ok=False)
        raw_panels: list[tuple[str, Image.Image]] = [
            ("RGB", rgb_image(rgb[batch_index], panel_size)),
            ("DABE soft", mask_image(dabe_soft[batch_index], panel_size)),
            ("DABE >= 0.5", mask_image(dabe_binary[batch_index], panel_size)),
        ]
        rgb_image(rgb[batch_index], 256).save(sample_dir / "rgb.png")
        mask_image(dabe_soft[batch_index], 256).save(sample_dir / "dabe_soft.png")
        mask_image(dabe_binary[batch_index], 256).save(sample_dir / "dabe_binary.png")
        for epoch in epochs:
            binary = teacher_binary_by_epoch[epoch][batch_index]
            raw_panels.append(
                (f"Teacher e{epoch}", mask_image(binary, panel_size))
            )
            mask_image(binary, 256).save(
                sample_dir / f"teacher_epoch_{epoch:03d}_binary.png"
            )
        comparison = render_row(
            raw_panels,
            sample_name=sample_name,
            panel_size=panel_size,
        )
        comparison.save(sample_dir / "comparison.png")

        y = overview_header_height + batch_index * overview_row_height
        overview_draw.text((4, y + 4), f"{batch_index:02d} {sample_name[:30]}", fill=(0, 0, 0))
        overview_panels = [
            rgb_image(rgb[batch_index], overview_panel_size),
            mask_image(dabe_soft[batch_index], overview_panel_size),
            mask_image(dabe_binary[batch_index], overview_panel_size),
        ] + [
            mask_image(teacher_binary_by_epoch[epoch][batch_index], overview_panel_size)
            for epoch in epochs
        ]
        for column, panel in enumerate(overview_panels):
            overview.paste(
                panel,
                (overview_label_width + column * overview_panel_size, y + 20),
            )
        manifest_samples.append(
            {
                "order": batch_index,
                "dataset_index": int(dataset_index),
                "dataset": datasets[batch_index],
                "stem": stems[batch_index],
                "directory": str(sample_dir.resolve()),
            }
        )

    overview.save(output_dir / "overview_10.png")
    manifest = {
        "config": str(resolve(args.config)),
        "checkpoint_dir": str(checkpoint_dir),
        "epochs": list(epochs),
        "num_samples": len(selected_indices),
        "random_seed": int(args.seed),
        "selected_indices": selected_indices,
        "teacher_threshold": 0.5,
        "dabe_binary_threshold": 0.5,
        "dabe_source": "pu_target_soft",
        "gt_read": False,
        "training_triggered": False,
        "samples": manifest_samples,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
