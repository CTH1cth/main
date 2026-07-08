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
    check_ml_feature_cache,
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
    # 从 checkpoint 的 head 权重反推 feature channel 数，兼容 simple/DAGP/context_residual。
    if "base_head.weight" in student_state:
        weight = student_state["base_head.weight"]
    elif "proj.weight" in student_state:
        weight = student_state["proj.weight"]
    elif "base.weight" in student_state:
        weight = student_state["base.weight"]
    elif "base_head.proj.weight" in student_state:
        weight = student_state["base_head.proj.weight"]
    elif "proj12.conv.weight" in student_state:
        weight = student_state["proj12.conv.weight"]
    else:
        raise KeyError("Cannot infer in_channels from checkpoint student state.")
    return int(weight.shape[1])


def use_multi_level_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))


def use_dagp_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp"


def use_raw_feature_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {"dagp", "dagp_safe"}


def use_ndr_branch(cfg):
    return bool(getattr(cfg, "USE_NDR_BRANCH", False))


def make_model_input(cfg, batch, device):
    if use_multi_level_feature(cfg):
        return {
            f"l{int(layer)}": batch[f"feature_l{int(layer)}"].to(device, non_blocking=True).float()
            for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])
        }
    feature = batch["feature"].to(device, non_blocking=True).float()
    if use_raw_feature_head(cfg):
        return feature
    return F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")


def make_image_68(cfg, batch, device):
    if not use_ndr_branch(cfg):
        return None
    if "image_68" not in batch:
        raise KeyError("USE_NDR_BRANCH=True requires batch['image_68'].")
    return batch["image_68"].to(device, non_blocking=True).float()


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def dataloader_worker_kwargs(cfg):
    num_workers = int(cfg.NUM_WORKERS)
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "DATALOADER_PERSISTENT_WORKERS", False))
        prefetch_factor = int(getattr(cfg, "DATALOADER_PREFETCH_FACTOR", 2))
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


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
        **dataloader_worker_kwargs(cfg),
    )
    metrics = CODMetrics()
    pred_dir = Path(out_dir) / "pred" / dataset_name
    student.eval()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = batch["stem"][0]
        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        if str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe":
            logits = extract_logits(student(model_input, image_68=image_68, return_aux=False))
        else:
            logits = extract_logits(student(model_input))
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
    missing_keys = sorted(set(student.state_dict()) - set(student_state))
    if str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe" and missing_keys == ["current_epoch_tensor"]:
        load_result = student.load_state_dict(student_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")
        student.set_epoch(int(getattr(cfg, "MAX_EPOCH", 25)))
    else:
        student.load_state_dict(student_state)

    with Logger(out_dir / "eval45.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"ckpt = {args.ckpt}")
        logger.log("model_for_eval = student")
        logger.log("pred_save_format = binary_0_255")
        if use_multi_level_feature(cfg):
            _, ml_reason = check_ml_feature_cache(cfg, "test")
            logger.log(f"[Cache] feature_ml:test ready | {ml_reason}")
        else:
            ensure_cache_available(cfg, "feature", split="test", logger=logger.log)
        for dataset_name in cfg.TEST_DATASETS:
            eval_dataset(cfg, student, dataset_name, device, out_dir, logger)


if __name__ == "__main__":
    main()
