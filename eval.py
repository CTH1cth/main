import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from common.dataset import CachedEvalDataset
from common.metrics import CODMetrics
from common.utils import (
    Logger,
    config_to_dict,
    ensure_dir,
    ensure_cache_available,
    format_metric_table,
    load_config,
    torch_load,
    write_yaml,
)
from model import build_seg_head


def infer_in_channels(student_state):
    # 从 checkpoint 的 head 权重反推 feature channel 数，兼容 simple/context_residual。
    if "proj.weight" in student_state:
        weight = student_state["proj.weight"]
    elif "base.weight" in student_state:
        weight = student_state["base.weight"]
    else:
        raise KeyError("Cannot infer in_channels from checkpoint student state.")
    return int(weight.shape[1])


def save_pred_png(path, pred):
    # 保存 0/255 二值 PNG，路径目录不存在时自动创建。
    ensure_dir(Path(path).parent)
    array = pred.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


@torch.no_grad()
def eval_dataset(cfg, student, dataset_name, device, out_dir, logger):
    # 单个测试集独立累计指标并保存 pred/{dataset} 下的预测图。
    dataset = CachedEvalDataset(cfg, split="test", datasets=[dataset_name])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    metrics = CODMetrics()
    pred_dir = Path(out_dir) / "pred" / dataset_name
    student.eval()

    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = batch["stem"][0]
        feature_68 = F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")
        logits = student(feature_68)
        # 当前实现保存的 pred 与计算指标用的是同一张原图尺寸二值 mask。
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear")
        pred = (logits.sigmoid() > float(cfg.THRESHOLD)).float()
        save_pred_png(pred_dir / f"{stem}.png", pred)
        metrics.step(gt, pred)

    result = metrics.get_result()
    logger.log(f"[Eval] Dataset: {dataset_name}")
    logger.log(format_metric_table(result))
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate clean cached-DINO EMA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "eval"
    ensure_dir(out_dir)
    write_yaml(out_dir / "config.yaml", config_to_dict(cfg))

    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    student_state = checkpoint["student"]
    student = build_seg_head(infer_in_channels(student_state), cfg).to(device)
    student.load_state_dict(student_state)

    with Logger(out_dir / "eval.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"ckpt = {args.ckpt}")
        logger.log("model_for_eval = student")
        logger.log("pred_save_format = binary_0_255")
        ensure_cache_available(cfg, "feature", split="test", logger=logger.log)
        for dataset_name in cfg.TEST_DATASETS:
            eval_dataset(cfg, student, dataset_name, device, out_dir, logger)


if __name__ == "__main__":
    main()
