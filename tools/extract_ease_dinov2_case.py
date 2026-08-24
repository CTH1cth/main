#!/usr/bin/env python3
"""Extract one DINOv2 patch-feature tensor using the official EASE protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ease_root", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--image_size", type=int, default=476)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ease_root = Path(args.ease_root).resolve()
    if str(ease_root) not in sys.path:
        sys.path.insert(0, str(ease_root))

    from hubconf import dinov2_vitl14

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    with Image.open(args.image_path) as image:
        original_size = list(image.size)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(args.device)

    model = dinov2_vitl14().to(args.device).eval()
    with torch.inference_mode():
        feature = model.get_intermediate_layers(tensor, reshape=True)[0][0].cpu()

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tensor": feature,
            "model": "dinov2_vitl14",
            "protocol": "EASE official default: Resize(476,476), ImageNet normalization, get_intermediate_layers(...)[0]",
            "image_path": str(Path(args.image_path).resolve()),
            "original_size": original_size,
            "input_size": [args.image_size, args.image_size],
        },
        output_path,
    )
    print(f"saved={output_path}")
    print(f"shape={tuple(feature.shape)} dtype={feature.dtype}")


if __name__ == "__main__":
    main()
