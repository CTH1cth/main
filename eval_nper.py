import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.metrics import CODMetrics
from common.utils import Logger, config_to_dict, ensure_dir, format_metric_table, load_config, torch_load, write_yaml
from nper.model_nper import NPERUCOD
from nper.train_utils import build_eval_loader


def model_forward(model, image, batch):
    if hasattr(model, "forward_with_paths"):
        return model.forward_with_paths(image, batch.get("image_path"))
    return model(image)


def save_pred_png(path, pred):
    ensure_dir(Path(path).parent)
    array = pred.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


@torch.no_grad()
def eval_dataset(cfg, model, dataset_name, device, out_dir, logger, max_samples=-1):
    _, loader = build_eval_loader(cfg, split="test", datasets=[dataset_name], max_samples=max_samples)
    metrics = CODMetrics()
    pred_dir = Path(out_dir) / "pred" / dataset_name
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = batch["stem"][0]
        out = model_forward(model, image, batch)
        logits = F.interpolate(out["logits"], size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = (logits.sigmoid() > float(cfg.THRESHOLD)).float()
        save_pred_png(pred_dir / f"{stem}.png", pred)
        metrics.step(gt, pred)
    result = metrics.get_result()
    logger.log(f"[Eval] Dataset: {dataset_name}")
    logger.log(format_metric_table(result))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate NPER-UCOD-V1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
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
    model = NPERUCOD(cfg).to(device)
    model.load_state_dict(checkpoint["student"])
    with Logger(out_dir / "eval.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"ckpt = {args.ckpt}")
        logger.log("model_for_eval = student")
        logger.log("pseudo_bank_used = false")
        for dataset_name in cfg.TEST_DATASETS:
            eval_dataset(cfg, model, dataset_name, device, out_dir, logger, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
