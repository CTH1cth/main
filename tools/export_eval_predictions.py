import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedEvalDataset  # noqa: E402
from common.utils import ensure_cache_available, ensure_dir, load_config, torch_load, write_jsonl  # noqa: E402
from eval import infer_in_channels  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    dataloader_worker_kwargs,
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
)


def parse_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def save_gray_png(path, array):
    ensure_dir(Path(path).parent)
    clipped = np.clip(array, 0.0, 1.0)
    Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_bin_png(path, mask):
    ensure_dir(Path(path).parent)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def load_student(cfg, ckpt_path, device):
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    if "student" not in checkpoint:
        raise KeyError(f"Checkpoint missing student state: {ckpt_path}")
    state = checkpoint["student"]
    student = build_seg_head(infer_in_channels(state), cfg).to(device)
    missing_keys = sorted(set(student.state_dict()) - set(state))
    if str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe" and missing_keys == ["current_epoch_tensor"]:
        result = student.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    else:
        student.load_state_dict(state)
    if hasattr(student, "set_epoch"):
        student.set_epoch(int(checkpoint.get("epoch", getattr(cfg, "MAX_EPOCH", 25))))
    student.eval()
    return student


@torch.no_grad()
def export_dataset(cfg, model, dataset_name, args, device):
    dataset = CachedEvalDataset(
        cfg,
        split="test",
        datasets=[dataset_name],
        max_samples=int(args.max_samples),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(cfg),
    )

    rows = []
    threshold = float(getattr(cfg, "THRESHOLD", 0.5))
    out_root = Path(args.out).expanduser()
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True).float()
        dataset_value = str(batch["dataset"][0])
        stem = str(batch["stem"][0])
        image_path = str(batch["image_path"][0])
        gt_path = str(batch["gt_path"][0])
        if dataset_value != dataset_name:
            raise RuntimeError(f"Dataset mismatch: expected {dataset_name}, got {dataset_value}/{stem}")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not Path(gt_path).exists():
            raise FileNotFoundError(f"GT not found: {gt_path}")

        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        output = forward_seg_head(model, model_input, cfg, image_68=image_68, return_aux=False)
        logits = extract_logits(output)
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits).detach().cpu().squeeze().numpy().astype(np.float32)
        prob = np.clip(prob, 0.0, 1.0)
        binary = prob > threshold

        dataset_dir = out_root / dataset_name
        ensure_dir(dataset_dir)
        prob_path = dataset_dir / f"{stem}_prob.npy"
        prob_png_path = dataset_dir / f"{stem}_prob.png"
        bin_path = dataset_dir / f"{stem}_bin.png"
        if args.save_prob:
            np.save(prob_path, prob)
            save_gray_png(prob_png_path, prob)
        if args.save_binary:
            save_bin_png(bin_path, binary)

        rows.append(
            {
                "dataset": dataset_name,
                "stem": stem,
                "image_path": image_path,
                "gt_path": gt_path,
                "prob_path": str(prob_path.resolve()) if args.save_prob else "",
                "prob_png_path": str(prob_png_path.resolve()) if args.save_prob else "",
                "bin_path": str(bin_path.resolve()) if args.save_binary else "",
                "tag": str(args.tag),
                "config": str(Path(args.config).expanduser()),
                "ckpt": str(Path(args.ckpt).expanduser()),
                "model_for_eval": "student",
                "threshold": threshold,
                "shape": list(prob.shape),
            }
        )
        print(f"[Export] {args.tag} | {dataset_name}/{stem} | shape={list(prob.shape)}", flush=True)
    if not rows:
        raise RuntimeError(f"No samples exported for dataset {dataset_name}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export eval probability and binary predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--save-binary", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    if not args.save_prob and not args.save_binary:
        raise ValueError("At least one of --save-prob or --save-binary must be set.")
    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config(args.config)
    datasets = parse_csv(args.datasets)
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset name.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_cache_available(cfg, "feature", split="test")
    model = load_student(cfg, ckpt_path, device)

    out_root = Path(args.out).expanduser()
    ensure_dir(out_root)
    rows = []
    for dataset_name in datasets:
        rows.extend(export_dataset(cfg, model, dataset_name, args, device))

    manifest_path = out_root / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    meta = {
        "tag": str(args.tag),
        "config": str(Path(args.config).expanduser()),
        "ckpt": str(ckpt_path),
        "datasets": datasets,
        "num_samples": len(rows),
        "save_prob": bool(args.save_prob),
        "save_binary": bool(args.save_binary),
        "threshold": float(getattr(cfg, "THRESHOLD", 0.5)),
    }
    with (out_root / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Export] done | rows={len(rows)} | manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
