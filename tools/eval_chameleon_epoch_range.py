import argparse
import csv
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
from common.metrics import CODMetrics  # noqa: E402
from common.utils import (  # noqa: E402
    Logger,
    config_to_dict,
    ensure_cache_available,
    ensure_dir,
    format_metric_table,
    load_config,
    torch_load,
    write_yaml,
)
from eval import infer_in_channels  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    dataloader_worker_kwargs,
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
)


METRIC_COLUMNS = [
    "epoch",
    "ckpt",
    "S_m",
    "F_beta_w",
    "F_beta_m",
    "E_phi_m",
    "MAE",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one dataset over a checkpoint epoch range and export VIS masks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--start_epoch", type=int, required=True)
    parser.add_argument("--end_epoch", type=int, required=True)
    parser.add_argument("--vis_start_epoch", type=int, required=True)
    parser.add_argument("--vis_end_epoch", type=int, required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dataset", default="CHAMELEON")
    return parser.parse_args()


def checkpoint_path(ckpt_dir, epoch):
    return Path(ckpt_dir).expanduser() / f"epoch_{int(epoch):03d}.pth"


def validate_args(args):
    if int(args.start_epoch) > int(args.end_epoch):
        raise ValueError("--start_epoch must be <= --end_epoch.")
    if int(args.vis_start_epoch) > int(args.vis_end_epoch):
        raise ValueError("--vis_start_epoch must be <= --vis_end_epoch.")
    if int(args.vis_start_epoch) < int(args.start_epoch) or int(args.vis_end_epoch) > int(args.end_epoch):
        raise ValueError("VIS epoch range must be inside the evaluation epoch range.")
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    missing = [
        str(checkpoint_path(ckpt_dir, epoch))
        for epoch in range(int(args.start_epoch), int(args.end_epoch) + 1)
        if not checkpoint_path(ckpt_dir, epoch).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint file(s): {missing[:10]}")


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
    head_type = str(getattr(cfg, "HEAD_TYPE", "simple")).lower()
    if head_type == "dagp_safe" and missing_keys == ["current_epoch_tensor"]:
        result = student.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    else:
        student.load_state_dict(state)
    if hasattr(student, "set_epoch"):
        student.set_epoch(int(checkpoint.get("epoch", getattr(cfg, "MAX_EPOCH", 25))))
    student.eval()
    return student


def save_binary_png(path, tensor):
    ensure_dir(Path(path).parent)
    array = tensor.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


def save_gt_png(path, gt_tensor):
    ensure_dir(Path(path).parent)
    gt = (gt_tensor.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(gt).save(path)


def save_rgb_png(path, image_path):
    ensure_dir(Path(path).parent)
    Image.open(image_path).convert("RGB").save(path)


def result_row(epoch, ckpt, result):
    return {
        "epoch": int(epoch),
        "ckpt": str(ckpt),
        "S_m": float(result["SMeasure"]),
        "F_beta_w": float(result["WFM"]),
        "F_beta_m": float(result["F_MEAN"]),
        "E_phi_m": float(result["E_MEAN"]),
        "MAE": float(result["MAE"]),
    }


def write_metric_csv(path, rows):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def eval_one_epoch(cfg, student, loader, device, epoch, vis_root, save_vis):
    metrics = CODMetrics()
    threshold = float(cfg.THRESHOLD)
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = str(batch["stem"][0])
        image_path = str(batch["image_path"][0])
        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        output = forward_seg_head(student, model_input, cfg, image_68=image_68, return_aux=False)
        logits = extract_logits(output)
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = (logits.sigmoid() > threshold).float()
        metrics.step(gt, pred)

        if save_vis:
            sample_dir = Path(vis_root) / stem
            rgb_path = sample_dir / "rgb.png"
            gt_path = sample_dir / "gt.png"
            if not rgb_path.exists():
                save_rgb_png(rgb_path, image_path)
            if not gt_path.exists():
                save_gt_png(gt_path, gt)
            save_binary_png(sample_dir / f"pred_epoch_{int(epoch):03d}.png", pred)

    return metrics.get_result()


def main():
    args = parse_args()
    validate_args(args)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root).expanduser()
    ensure_dir(out_root)
    write_yaml(out_root / "config.yaml", config_to_dict(cfg))

    vis_root = out_root.parent / "VIS" / f"chameleon_epoch_{int(args.vis_start_epoch):03d}_{int(args.vis_end_epoch):03d}"
    ensure_dir(vis_root)

    ensure_cache_available(cfg, "feature", split="test")
    dataset = CachedEvalDataset(cfg, split="test", datasets=[str(args.dataset)])
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
    csv_path = out_root / "chameleon_epoch_metrics.csv"
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    with Logger(out_root / "chameleon_epoch_eval.log") as logger:
        logger.log(f"config = {args.config}")
        logger.log(f"ckpt_dir = {ckpt_dir}")
        logger.log(f"epoch_range = {int(args.start_epoch):03d}-{int(args.end_epoch):03d}")
        logger.log(f"vis_epoch_range = {int(args.vis_start_epoch):03d}-{int(args.vis_end_epoch):03d}")
        logger.log(f"dataset = {args.dataset}")
        logger.log(f"threshold = {float(cfg.THRESHOLD):.6f}")
        logger.log(f"device = {device}")
        logger.log(f"out_root = {out_root}")
        logger.log(f"vis_root = {vis_root}")
        logger.log(f"num_samples = {len(dataset)}")

        for epoch in range(int(args.start_epoch), int(args.end_epoch) + 1):
            ckpt = checkpoint_path(ckpt_dir, epoch)
            logger.log(f"[Epoch {epoch:03d}] ckpt = {ckpt}")
            student = load_student(cfg, ckpt, device)
            result = eval_one_epoch(
                cfg,
                student,
                loader,
                device,
                epoch,
                vis_root,
                save_vis=int(args.vis_start_epoch) <= epoch <= int(args.vis_end_epoch),
            )
            row = result_row(epoch, ckpt, result)
            rows.append(row)
            write_metric_csv(csv_path, rows)
            logger.log(f"[Eval] Dataset: {args.dataset} | Epoch: {epoch:03d}")
            logger.log(format_metric_table(result))
            logger.log(
                f"[CSV] epoch={epoch:03d} | S_m={row['S_m']:.6f} | "
                f"F_beta_w={row['F_beta_w']:.6f} | F_beta_m={row['F_beta_m']:.6f} | "
                f"E_phi_m={row['E_phi_m']:.6f} | MAE={row['MAE']:.6f}"
            )
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.log(f"[Done] metrics_csv = {csv_path}")
        logger.log(f"[Done] vis_root = {vis_root}")


if __name__ == "__main__":
    main()
