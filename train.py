import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import CachedEvalDataset, CachedTrainDataset
from common.metrics import CODMetrics
from common.utils import (
    Logger,
    config_to_dict,
    current_lr,
    ensure_dir,
    ensure_cache_available,
    format_metric_table,
    load_config,
    metric_value,
    set_seed,
    write_yaml,
)
from model import SimpleConvSegHead, update_ema


def get_teacher_weight(epoch):
    # 1-based epoch: 前 20 轮从 fixed pseudo 平滑过渡到 teacher pseudo。
    if epoch <= 20:
        return (epoch - 1) / 20.0
    return 1.0


def build_optimizer_scheduler(cfg, student):
    # 与 UCOD-DPL 对齐：AdamW + 每 iteration StepLR。
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.DINO["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.95,
    )
    return optimizer, scheduler


def build_loaders(cfg):
    # 构建训练 DataLoader；shuffle 的随机性由固定 generator 控制。
    train_dataset = CachedTrainDataset(cfg)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.SEED))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.BATCH_SIZE),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return train_dataset, train_loader


@torch.no_grad()
def validate_one_dataset(cfg, student, dataset_name, device):
    # 每轮验证只用 student，验证阶段不保存预测图。
    dataset = CachedEvalDataset(cfg, split="val", datasets=[dataset_name])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.VAL_BATCH_SIZE),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    metrics = CODMetrics()
    student.eval()
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()
        feature_68 = F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")
        logits = student(feature_68)
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear")
        pred = (logits.sigmoid() > float(cfg.THRESHOLD)).float()
        metrics.step(gt, pred)
    return metrics.get_result()


def save_checkpoint(path, epoch, cfg, student, teacher, optimizer, scheduler, best_metric, best_epoch):
    # checkpoint 同时保存 teacher 和优化器状态，便于后续接续训练。
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": epoch,
            "backbone_key": cfg.BACKBONE_KEY,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": config_to_dict(cfg),
        },
        path,
    )


def log_cache_summary(logger, cfg, train_dataset):
    # 训练日志开头记录 cache 规模和 shape，方便确认读到的是正确 backbone。
    logger.log(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger.log(f"train datasets = {' + '.join(cfg.TRAIN_DATASETS)}")
    logger.log(f"num_train_samples = {len(train_dataset)}")
    logger.log(f"feature cache = {Path(cfg.CACHE_ROOT) / 'features_cache' / cfg.BACKBONE_KEY}")
    logger.log(f"pseudo cache = {Path(cfg.CACHE_ROOT) / 'pseudo_label_cache' / cfg.BACKBONE_KEY}")
    logger.log(f"feature shape example = {train_dataset.feature_shape}")
    logger.log(f"pseudo shape example = {train_dataset.pseudo_shape}")
    logger.log(f"loss size = {cfg.LOSS_SIZE}x{cfg.LOSS_SIZE}")


def main():
    parser = argparse.ArgumentParser(description="Train clean cached-DINO EMA baseline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "train"
    ckpt_dir = train_dir / "ckpt"
    ensure_dir(ckpt_dir)
    write_yaml(train_dir / "config.yaml", config_to_dict(cfg))

    with Logger(train_dir / "train.log") as logger:
        logger.log(f"device = {device}")
        logger.log("optimizer = AdamW")
        logger.log(f"lr = {cfg.DINO['lr']}")
        logger.log("scheduler = StepLR(step_size=25, gamma=0.95)")
        logger.log("scheduler_step = iter")
        logger.log(f"ema_weight = {cfg.EMA_WEIGHT}")
        logger.log("finetune_reset_epoch = 21")
        logger.log("finetune_reset_rebuild_optimizer = true")
        logger.log("finetune_reset_rebuild_scheduler = true")
        logger.log("finetune_reset_global_step = true")
        logger.log("finetune_reset_teacher = false")
        logger.log("DINO_in_training_loop = false")
        logger.log("train_gt_in_train = false")

        # 训练前只补齐 cache；正式训练循环不加载 DINO，也不读训练集 GT。
        ensure_cache_available(cfg, "feature", split="train", logger=logger.log)
        ensure_cache_available(cfg, "feature", split="val", logger=logger.log)
        ensure_cache_available(cfg, "pseudo", logger=logger.log)

        train_dataset, train_loader = build_loaders(cfg)
        log_cache_summary(logger, cfg, train_dataset)

        in_channels = train_dataset.in_channels
        student = SimpleConvSegHead(in_channels).to(device)
        teacher = SimpleConvSegHead(in_channels).to(device)
        # 初始 teacher 与 student 对齐；第 21 轮 finetune reset 不显式重置 teacher。
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer, scheduler = build_optimizer_scheduler(cfg, student)

        best_metric = float("inf")
        best_epoch = 0
        global_step = 0

        for epoch in range(1, int(cfg.MAX_EPOCH) + 1):
            # 对齐 UCOD-DPL：teacher-only 阶段首轮第一个 batch 前重置优化器状态和 EMA 步数。
            if epoch == 21:
                optimizer, scheduler = build_optimizer_scheduler(cfg, student)
                global_step = 0
                logger.log(
                    "[Finetune Reset] epoch=021 | "
                    "rebuild_optimizer=True | "
                    "rebuild_scheduler=True | "
                    "reset_global_step=True | "
                    "reset_teacher=False | "
                    f"lr={current_lr(optimizer):.8f}"
                )

            student.train()
            teacher.eval()
            teacher_weight = get_teacher_weight(epoch)
            fixed_weight = 1.0 - teacher_weight
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                feature = batch["feature"].to(device, non_blocking=True).float()
                pseudo = batch["pseudo"].to(device, non_blocking=True).float()
                feature_68 = F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")
                pseudo_68 = F.interpolate(pseudo, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear").float()

                student_logits = student(feature_68)
                with torch.no_grad():
                    teacher_logits = teacher(feature_68)
                    teacher_binary = (teacher_logits.sigmoid() > float(cfg.THRESHOLD)).float()

                mixed_target = fixed_weight * pseudo_68 + teacher_weight * teacher_binary
                loss = criterion(student_logits, mixed_target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # teacher pseudo 已在本轮 EMA 更新前生成，避免当前 student 更新泄漏进 target。
                update_ema(student, teacher, global_step, ema_weight=float(cfg.EMA_WEIGHT))
                global_step += 1

                total_loss += float(loss.item())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            logger.log(
                f"[Train] Epoch {epoch:03d}/{cfg.MAX_EPOCH:03d} | "
                f"avg_train_loss={avg_loss:.6f} | lr={current_lr(optimizer):.8f} | "
                f"fixed_weight={fixed_weight:.2f} | teacher_weight={teacher_weight:.2f}"
            )

            val_results = {}
            for dataset_name in cfg.VAL_DATASETS:
                result = validate_one_dataset(cfg, student, dataset_name, device)
                val_results[dataset_name] = result
                logger.log(f"[Validation] Epoch {epoch:03d}/{cfg.MAX_EPOCH:03d} | Dataset: {dataset_name}")
                logger.log(format_metric_table(result))

            current_best_candidate = metric_value(val_results[cfg.BEST_DATASET], cfg.BEST_METRIC)
            improved = current_best_candidate < best_metric if cfg.BEST_MODE == "min" else current_best_candidate > best_metric
            if improved:
                best_metric = current_best_candidate
                best_epoch = epoch
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch,
                    cfg,
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    best_metric,
                    best_epoch,
                )

            logger.log(f"best MAE so far = {best_metric:.6f}")
            logger.log(f"best epoch = {best_epoch}")

            if epoch % int(cfg.SAVE_INTERVAL) == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pth",
                    epoch,
                    cfg,
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    best_metric,
                    best_epoch,
                )


if __name__ == "__main__":
    main()
