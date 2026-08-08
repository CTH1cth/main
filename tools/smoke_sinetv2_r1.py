#!/usr/bin/env python3
"""Run one SINet-V2 optimization step on the exported R1 supervision."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets/cache/sinetv2_r1_v1s_train"
DEFAULT_SINET_ROOT = Path("/home/dell01/CTH/0base/RISE-master/SINet-V2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sinet-root", type=Path, default=DEFAULT_SINET_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    sinet_root = args.sinet_root.resolve()
    if not (sinet_root / "MyTrain_Val.py").is_file():
        raise FileNotFoundError(f"Official SINet-V2 source not found: {sinet_root}")
    if not (dataset_root / "protocol.json").is_file():
        raise FileNotFoundError(f"R1 export is incomplete: {dataset_root}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(sinet_root))
    from lib.Network_Res2Net_GRA_NCD import Network  # noqa: PLC0415
    from MyTrain_Val import structure_loss  # noqa: PLC0415
    from utils.data_val import get_loader  # noqa: PLC0415

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested but CUDA is unavailable")

    loader = get_loader(
        image_root=str(dataset_root / "TrainDataset/Imgs") + "/",
        gt_root=str(dataset_root / "TrainDataset/PseudoMask") + "/",
        batchsize=1,
        trainsize=352,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    images, masks = next(iter(loader))
    if tuple(images.shape) != (1, 3, 352, 352):
        raise RuntimeError(f"Unexpected image batch shape: {tuple(images.shape)}")
    if tuple(masks.shape) != (1, 1, 352, 352):
        raise RuntimeError(f"Unexpected mask batch shape: {tuple(masks.shape)}")
    if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
        raise RuntimeError("Non-finite smoke-test inputs")

    model = Network(channel=32, imagenet_pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    images = images.to(device)
    masks = masks.to(device)
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)
    if len(predictions) != 4:
        raise RuntimeError(f"Expected four SINet-V2 predictions, got {len(predictions)}")
    loss_init = sum(structure_loss(prediction, masks) for prediction in predictions[:3])
    loss_final = structure_loss(predictions[3], masks)
    loss = loss_init + loss_final
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {float(loss)}")
    loss.backward()
    optimizer.step()

    print(
        "SINet-V2 R1 smoke test passed: "
        f"device={device}, batch={tuple(images.shape)}, "
        f"mask_range=({float(masks.min()):.6f}, {float(masks.max()):.6f}), "
        f"loss={float(loss.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
