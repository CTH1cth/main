import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import CachedEvalDataset, CachedTrainDataset
from common.metrics import CODMetrics
from common.utils import (
    Logger,
    cache_status,
    check_ccr_cache,
    check_despl_light_cache,
    check_despl_pseudo_bank,
    check_qra_cache,
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
from model import build_seg_head, update_ema


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


def build_loaders(cfg, max_train_samples=-1):
    # 构建训练 DataLoader；shuffle 的随机性由固定 generator 控制。
    train_dataset = CachedTrainDataset(cfg, max_samples=max_train_samples)
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
def validate_one_dataset(cfg, student, dataset_name, device, max_samples=-1):
    # 每轮验证只用 student，验证阶段不保存预测图。
    dataset = CachedEvalDataset(
        cfg,
        split="val",
        datasets=[dataset_name],
        max_samples=max_samples,
    )
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
    logger.log(
        "feature cache path = "
        f"{(Path(cfg.CACHE_ROOT) / 'features_cache' / cfg.BACKBONE_KEY).resolve()}"
    )
    logger.log(
        "original pseudo cache path = "
        f"{train_dataset.original_pseudo_cache_root}"
    )
    logger.log(
        f"pseudo_cache_override = {train_dataset.pseudo_cache_override}"
    )
    logger.log(
        "actual pseudo cache path used = "
        f"{train_dataset.actual_pseudo_cache_pattern}"
    )
    logger.log(
        "first pseudo cache file = "
        f"{train_dataset.first_pseudo_cache_path}"
    )
    logger.log(f"pseudo source = {train_dataset.pseudo_source}")
    logger.log(
        f"pseudo final candidate = {train_dataset.pseudo_final_candidate}"
    )
    logger.log(f"feature shape example = {train_dataset.feature_shape}")
    logger.log(f"pseudo shape example = {train_dataset.pseudo_shape}")
    if getattr(cfg, "USE_QRA", False):
        logger.log(f"QRA cache path = {train_dataset.qra_cache_root}")
        logger.log(f"first QRA cache file = {train_dataset.qra_first_cache_path}")
        logger.log(f"first QRA quality = {train_dataset.qra_first_quality}")
        logger.log(f"first QRA sim = {train_dataset.qra_first_sim:.6f}")
        logger.log(f"first QRA anchor ratio = {train_dataset.qra_first_anchor_ratio:.6f}")
    if getattr(cfg, "USE_CCR", False):
        logger.log(f"CCR cache path = {train_dataset.ccr_cache_root}")
        logger.log(f"first CCR cache file = {train_dataset.ccr_first_cache_path}")
        logger.log(f"first CCR quality = {train_dataset.ccr_first_quality}")
        logger.log(f"first CCR IoU = {train_dataset.ccr_first_iou:.6f}")
        logger.log(f"first CCR anchor ratio = {train_dataset.ccr_first_anchor_ratio:.6f}")
    if getattr(cfg, "USE_DESPL_PSEUDO", False):
        logger.log(f"DESPL pseudo cache path = {train_dataset.despl_cache_root}")
        logger.log(f"first DESPL pseudo file = {train_dataset.despl_first_cache_path}")
        logger.log(f"use_despl_pseudo = true")
        logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
        logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', 'despl_fixed_blend')}")
        logger.log(
            "p_init_formula = "
            f"{float(getattr(cfg, 'P_INIT_DESPL_WEIGHT', 0.8)):.3f}*p_despl + "
            f"{float(getattr(cfg, 'P_INIT_FIXED_WEIGHT', 0.2)):.3f}*p_fixed"
        )
    logger.log(f"loss size = {cfg.LOSS_SIZE}x{cfg.LOSS_SIZE}")


def log_first_batch_pseudo(logger, cfg, train_dataset):
    # 使用独立、无 shuffle 的 loader 做诊断，不推进正式训练 loader 的随机状态。
    diagnostic_loader = DataLoader(
        train_dataset,
        batch_size=min(int(cfg.BATCH_SIZE), len(train_dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(diagnostic_loader))
    pseudo = batch["pseudo"].float()
    logger.log(f"first batch pseudo tensor shape = {list(pseudo.shape)}")
    logger.log(f"first sample pseudo tensor shape = {list(pseudo[0].shape)}")
    logger.log(f"pseudo min = {float(pseudo.min().item()):.6f}")
    logger.log(f"pseudo max = {float(pseudo.max().item()):.6f}")
    logger.log(f"pseudo sum = {float(pseudo.sum().item()):.6f}")
    logger.log(
        "first batch datasets = "
        + ",".join(str(value) for value in batch["dataset"])
    )
    logger.log(
        "first batch stems = "
        + ",".join(str(value) for value in batch["stem"])
    )
    if getattr(cfg, "USE_QRA", False):
        logger.log(f"first batch QRA quality = {batch['qra_quality'].tolist()}")
        logger.log(
            "first batch QRA sim = "
            + ",".join(f"{float(value):.4f}" for value in batch["qra_sim"])
        )
    if getattr(cfg, "USE_CCR", False):
        logger.log(f"first batch CCR quality = {batch['ccr_quality'].tolist()}")
        logger.log(
            "first batch CCR IoU = "
            + ",".join(f"{float(value):.4f}" for value in batch["ccr_iou_fixed_despl"])
        )
    if getattr(cfg, "USE_DESPL_PSEUDO", False):
        logger.log(f"first batch pseudo_fixed tensor shape = {list(batch['pseudo_fixed'].shape)}")
        logger.log(f"first batch pseudo_despl tensor shape = {list(batch['pseudo_despl'].shape)}")
        logger.log(
            "first batch p_init_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["p_init_area"])
        )
        logger.log(
            "first batch p_fixed_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["p_fixed_area"])
        )
        logger.log(
            "first batch p_despl_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["p_despl_area"])
        )


def qra_quality_weights(quality, q0, q1, q2):
    quality = quality.long()
    weights = torch.full_like(quality, float(q0), dtype=torch.float32)
    weights = torch.where(
        quality == 1,
        torch.full_like(weights, float(q1)),
        weights,
    )
    weights = torch.where(
        quality == 2,
        torch.full_like(weights, float(q2)),
        weights,
    )
    return weights


def qra_fixed_blend(cfg, quality, device):
    if not bool(getattr(cfg, "QRA_REPLACE_FIXED", True)):
        return torch.zeros_like(quality, dtype=torch.float32, device=device)
    return qra_quality_weights(
        quality,
        getattr(cfg, "QRA_FIXED_BLEND_Q0", 0.0),
        getattr(cfg, "QRA_FIXED_BLEND_Q1", 0.5),
        getattr(cfg, "QRA_FIXED_BLEND_Q2", 1.0),
    ).to(device)


def compute_qra_losses(cfg, epoch, student_logits, batch, device, criterion_none):
    quality = batch["qra_quality"].to(device, non_blocking=True).long()
    anchor_fg = batch["qra_anchor_fg"].to(device, non_blocking=True).bool()
    anchor_bg = batch["qra_anchor_bg"].to(device, non_blocking=True).bool()
    anchor_mask = anchor_fg | anchor_bg
    anchor_label = anchor_fg.float()

    anchor_loss_map = criterion_none(student_logits, anchor_label)
    anchor_weight = anchor_mask.float()
    anchor_den = anchor_weight.flatten(1).sum(dim=1).clamp_min(1.0)
    anchor_sample = (anchor_loss_map * anchor_weight).flatten(1).sum(dim=1) / anchor_den
    if epoch <= 20:
        lambda_anchor = qra_quality_weights(
            quality,
            getattr(cfg, "QRA_LAMBDA_ANCHOR_Q0", 0.0),
            getattr(cfg, "QRA_LAMBDA_ANCHOR_Q1", 0.0),
            getattr(cfg, "QRA_LAMBDA_ANCHOR_Q2", 0.0),
        ).to(device)
    else:
        lambda_anchor = torch.where(
            quality > 0,
            torch.full_like(quality, float(getattr(cfg, "QRA_LAMBDA_LATE_ANCHOR", 0.0)), dtype=torch.float32),
            torch.zeros_like(quality, dtype=torch.float32),
        ).to(device)
    loss_anchor = (lambda_anchor * anchor_sample).mean()

    if epoch <= 20:
        soft_target = batch["qra_p_fused"].to(device, non_blocking=True).float()
        pixel_weight = batch["qra_pixel_weight"].to(device, non_blocking=True).float()
        soft_loss_map = criterion_none(student_logits, soft_target)
        soft_sample = (soft_loss_map * pixel_weight).flatten(1).mean(dim=1)
        lambda_soft = qra_quality_weights(
            quality,
            getattr(cfg, "QRA_LAMBDA_SOFT_Q0", 0.0),
            getattr(cfg, "QRA_LAMBDA_SOFT_Q1", 0.0),
            getattr(cfg, "QRA_LAMBDA_SOFT_Q2", 0.0),
        ).to(device)
        loss_soft = (lambda_soft * soft_sample).mean()
    else:
        loss_soft = student_logits.sum() * 0.0

    return loss_anchor, loss_soft


def ccr_quality_weights(quality, q0, q1, q2):
    return qra_quality_weights(quality, q0, q1, q2)


def compute_ccr_anchor_loss(cfg, student_logits, batch, device, criterion_none):
    quality = batch["ccr_quality"].to(device, non_blocking=True).long()
    anchor_fg = batch["ccr_anchor_fg"].to(device, non_blocking=True).bool()
    anchor_bg = batch["ccr_anchor_bg"].to(device, non_blocking=True).bool()
    anchor_mask = anchor_fg | anchor_bg
    anchor_label = anchor_fg.float()
    loss_map = criterion_none(student_logits, anchor_label)
    anchor_weight = anchor_mask.float()
    anchor_den = anchor_weight.flatten(1).sum(dim=1).clamp_min(1.0)
    anchor_sample = (loss_map * anchor_weight).flatten(1).sum(dim=1) / anchor_den
    lambdas = ccr_quality_weights(
        quality,
        getattr(cfg, "CCR_LAMBDA_ANCHOR_Q0", 0.0),
        getattr(cfg, "CCR_LAMBDA_ANCHOR_Q1", 0.0),
        getattr(cfg, "CCR_LAMBDA_ANCHOR_Q2", 0.0),
    ).to(device)
    return (lambdas * anchor_sample).mean()


def apply_ccr_late_override(cfg, mixed_target, teacher_binary, batch, device):
    if not bool(getattr(cfg, "CCR_LATE_OVERRIDE", False)):
        return mixed_target, 0.0
    quality = batch["ccr_quality"].to(device, non_blocking=True).long()
    rho = ccr_quality_weights(
        quality,
        getattr(cfg, "CCR_LATE_RHO_Q0", 0.0),
        getattr(cfg, "CCR_LATE_RHO_Q1", 0.0),
        getattr(cfg, "CCR_LATE_RHO_Q2", 0.0),
    ).to(device).view(-1, 1, 1, 1)
    anchor_fg = batch["ccr_anchor_fg"].to(device, non_blocking=True).bool()
    anchor_bg = batch["ccr_anchor_bg"].to(device, non_blocking=True).bool()
    anchor_mask = anchor_fg | anchor_bg
    active_mask = anchor_mask & (rho > 0)
    if not bool(active_mask.any().item()):
        return mixed_target, 0.0
    anchor_label = anchor_fg.float()
    override_target = (1.0 - rho) * teacher_binary + rho * anchor_label
    mixed_target = torch.where(active_mask, override_target, mixed_target)
    return mixed_target, float(active_mask.float().mean().item())


def head_gamma_value(model):
    gamma = getattr(model, "gamma", None)
    if gamma is None:
        return None
    return float(gamma.detach().cpu().item())


def main():
    parser = argparse.ArgumentParser(description="Train clean cached-DINO EMA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo_cache_override", default=None)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--debug_loader_only", action="store_true")
    parser.add_argument("--work_dir", default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_train_samples != -1:
        raise ValueError("--max_samples and --max_train_samples cannot be used together.")
    sample_limit = args.max_samples if args.max_samples is not None else args.max_train_samples
    if sample_limit == 0 or sample_limit < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max_epochs must be a positive integer.")

    cfg = load_config(args.config)
    if bool(getattr(cfg, "USE_QRA", False)) and bool(getattr(cfg, "USE_CCR", False)):
        raise RuntimeError("USE_QRA=True and USE_CCR=True cannot be combined.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and (
        bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False))
    ):
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
    if bool(getattr(cfg, "USE_QRA", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_QRA=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_CCR", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_CCR=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with --pseudo_cache_override.")
    cfg.PSEUDO_CACHE_OVERRIDE = args.pseudo_cache_override
    max_epoch = int(args.max_epochs) if args.max_epochs is not None else int(cfg.MAX_EPOCH)
    set_seed(int(cfg.SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.work_dir:
        train_dir = Path(args.work_dir)
    elif args.debug_loader_only:
        train_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "debug_loader"
    else:
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
        logger.log(f"max_epoch = {max_epoch}")
        logger.log(f"max_samples = {sample_limit}")
        logger.log(f"ema_weight = {cfg.EMA_WEIGHT}")
        logger.log("finetune_reset_epoch = 21")
        logger.log("finetune_reset_rebuild_optimizer = true")
        logger.log("finetune_reset_rebuild_scheduler = true")
        logger.log("finetune_reset_global_step = true")
        logger.log("finetune_reset_teacher = false")
        logger.log("DINO_in_training_loop = false")
        logger.log("train_gt_in_train = false")
        logger.log(f"head_type = {getattr(cfg, 'HEAD_TYPE', 'simple')}")
        logger.log(f"use_despl_pseudo = {bool(getattr(cfg, 'USE_DESPL_PSEUDO', False))}")
        logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
        logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', 'original_fixed')}")

        use_qra = bool(getattr(cfg, "USE_QRA", False))
        use_ccr = bool(getattr(cfg, "USE_CCR", False))
        use_despl = bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        use_despl_light = use_despl and bool(getattr(cfg, "USE_DESPL_LIGHT_CACHE", False))

        # 正式训练循环不加载 DINO，也不读训练集 GT。DESPL 实验只检查现有 cache，不自动生成。
        if use_despl:
            for split in ("train", "val"):
                if split == "val" and args.debug_loader_only:
                    continue
                complete, reason = cache_status(cfg, "feature", split=split)
                if not complete:
                    raise RuntimeError(f"feature {split} cache missing or incomplete: {reason}")
                logger.log(f"[Cache] feature:{split} ready | {reason}")
        else:
            ensure_cache_available(cfg, "feature", split="train", logger=logger.log)
            if not args.debug_loader_only:
                ensure_cache_available(cfg, "feature", split="val", logger=logger.log)
        if cfg.PSEUDO_CACHE_OVERRIDE:
            logger.log(
                "[Cache] pseudo override enabled | "
                "skip original pseudo cache generation/check"
            )
        elif use_despl:
            if use_despl_light:
                _, despl_reason = check_despl_light_cache(
                    cfg,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] DESPL light cache ready | {despl_reason}")
            else:
                _, despl_reason = check_despl_pseudo_bank(
                    cfg,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] DESPL pseudo bank ready | {despl_reason}")
        else:
            ensure_cache_available(cfg, "pseudo", logger=logger.log)
        if use_qra:
            _, qra_reason = check_qra_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] QRA ready | {qra_reason}")
        if use_ccr:
            _, ccr_reason = check_ccr_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] CCR ready | {ccr_reason}")

        train_dataset, train_loader = build_loaders(
            cfg, max_train_samples=sample_limit
        )
        log_cache_summary(logger, cfg, train_dataset)
        log_first_batch_pseudo(logger, cfg, train_dataset)
        if args.debug_loader_only:
            logger.log(
                "[Debug Loader] completed successfully; training loop not entered."
            )
            return

        in_channels = train_dataset.in_channels
        student = build_seg_head(in_channels, cfg).to(device)
        teacher = build_seg_head(in_channels, cfg).to(device)
        # 初始 teacher 与 student 对齐；第 21 轮 finetune reset 不显式重置 teacher。
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

        criterion = torch.nn.BCEWithLogitsLoss()
        criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer, scheduler = build_optimizer_scheduler(cfg, student)

        best_metric = float("inf")
        best_epoch = 0
        global_step = 0

        for epoch in range(1, max_epoch + 1):
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
            total_base_loss = 0.0
            total_anchor_loss = 0.0
            total_soft_loss = 0.0
            num_batches = 0
            qra_q0 = 0
            qra_q1 = 0
            qra_q2 = 0
            qra_num_samples = 0
            qra_anchor_ratio_sum = 0.0
            qra_sim_sum = 0.0
            ccr_q0 = 0
            ccr_q1 = 0
            ccr_q2 = 0
            ccr_num_samples = 0
            ccr_iou_sum = 0.0
            ccr_corr_area_sum = 0.0
            ccr_expand_area_sum = 0.0
            ccr_shrink_area_sum = 0.0
            ccr_anchor_ratio_sum = 0.0
            ccr_late_override_sum = 0.0
            despl_num_samples = 0
            despl_p_init_area_sum = 0.0
            despl_p_fixed_area_sum = 0.0
            despl_p_despl_area_sum = 0.0

            for batch in train_loader:
                feature = batch["feature"].to(device, non_blocking=True).float()
                pseudo = batch["pseudo"].to(device, non_blocking=True).float()
                feature_68 = F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")
                pseudo_68 = F.interpolate(pseudo, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear").float()

                student_logits = student(feature_68)
                with torch.no_grad():
                    teacher_logits = teacher(feature_68)
                    teacher_binary = (teacher_logits.sigmoid() > float(cfg.THRESHOLD)).float()

                fixed_target = pseudo_68
                if use_ccr:
                    fixed_target = batch["ccr_p_corr"].to(device, non_blocking=True).float()
                elif use_qra and epoch <= 20:
                    qra_quality = batch["qra_quality"].to(device, non_blocking=True).long()
                    qra_fused = batch["qra_p_fused"].to(device, non_blocking=True).float()
                    blend = qra_fixed_blend(cfg, qra_quality, device).view(-1, 1, 1, 1)
                    fixed_target = (1.0 - blend) * pseudo_68 + blend * qra_fused

                late_override_ratio = 0.0
                if epoch <= 20:
                    mixed_target = fixed_weight * fixed_target + teacher_weight * teacher_binary
                else:
                    mixed_target = teacher_binary
                    if use_ccr:
                        mixed_target, late_override_ratio = apply_ccr_late_override(
                            cfg,
                            mixed_target,
                            teacher_binary,
                            batch,
                            device,
                        )
                loss_base = criterion(student_logits, mixed_target)
                if use_qra:
                    loss_anchor, loss_soft = compute_qra_losses(
                        cfg,
                        epoch,
                        student_logits,
                        batch,
                        device,
                        criterion_none,
                    )
                    loss = loss_base + loss_anchor + loss_soft
                elif use_ccr and epoch <= 20 and bool(getattr(cfg, "CCR_USE_ANCHOR_LOSS", False)):
                    loss_anchor = compute_ccr_anchor_loss(
                        cfg,
                        student_logits,
                        batch,
                        device,
                        criterion_none,
                    )
                    loss_soft = student_logits.sum() * 0.0
                    loss = loss_base + loss_anchor
                else:
                    loss_anchor = student_logits.sum() * 0.0
                    loss_soft = student_logits.sum() * 0.0
                    loss = loss_base

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # teacher pseudo 已在本轮 EMA 更新前生成，避免当前 student 更新泄漏进 target。
                update_ema(student, teacher, global_step, ema_weight=float(cfg.EMA_WEIGHT))
                global_step += 1

                total_loss += float(loss.item())
                total_base_loss += float(loss_base.item())
                total_anchor_loss += float(loss_anchor.item())
                total_soft_loss += float(loss_soft.item())
                num_batches += 1
                if use_qra:
                    quality_cpu = batch["qra_quality"]
                    qra_q0 += int((quality_cpu == 0).sum().item())
                    qra_q1 += int((quality_cpu == 1).sum().item())
                    qra_q2 += int((quality_cpu == 2).sum().item())
                    qra_num_samples += int(quality_cpu.numel())
                    qra_anchor_ratio_sum += float(batch["qra_anchor_ratio"].sum().item())
                    qra_sim_sum += float(batch["qra_sim"].sum().item())
                if use_ccr:
                    quality_cpu = batch["ccr_quality"]
                    ccr_q0 += int((quality_cpu == 0).sum().item())
                    ccr_q1 += int((quality_cpu == 1).sum().item())
                    ccr_q2 += int((quality_cpu == 2).sum().item())
                    ccr_num_samples += int(quality_cpu.numel())
                    ccr_iou_sum += float(batch["ccr_iou_fixed_despl"].sum().item())
                    ccr_corr_area_sum += float(batch["ccr_corr_area"].sum().item())
                    ccr_expand_area_sum += float(batch["ccr_trusted_expand_area"].sum().item())
                    ccr_shrink_area_sum += float(batch["ccr_trusted_shrink_area"].sum().item())
                    ccr_anchor_ratio_sum += float(batch["ccr_anchor_ratio"].sum().item())
                    ccr_late_override_sum += late_override_ratio
                if use_despl:
                    despl_num_samples += int(batch["p_init_area"].numel())
                    despl_p_init_area_sum += float(batch["p_init_area"].sum().item())
                    despl_p_fixed_area_sum += float(batch["p_fixed_area"].sum().item())
                    despl_p_despl_area_sum += float(batch["p_despl_area"].sum().item())

            avg_loss = total_loss / max(num_batches, 1)
            logger.log(
                f"[Train] Epoch {epoch:03d}/{max_epoch:03d} | "
                f"avg_train_loss={avg_loss:.6f} | lr={current_lr(optimizer):.8f} | "
                f"fixed_weight={fixed_weight:.2f} | teacher_weight={teacher_weight:.2f}"
            )
            if use_qra:
                logger.log(
                    f"[QRA] epoch={epoch:03d} | "
                    f"q0={qra_q0} | q1={qra_q1} | q2={qra_q2} | "
                    f"anchor_ratio={qra_anchor_ratio_sum / max(qra_num_samples, 1):.6f} | "
                    f"sim={qra_sim_sum / max(qra_num_samples, 1):.6f} | "
                    f"loss_base={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_anchor={total_anchor_loss / max(num_batches, 1):.6f} | "
                    f"loss_soft={total_soft_loss / max(num_batches, 1):.6f}"
                )
            if use_ccr:
                gamma = head_gamma_value(student)
                gamma_text = "NA" if gamma is None else f"{gamma:.8f}"
                logger.log(
                    f"[CCR] epoch={epoch:03d} | "
                    f"q0={ccr_q0} | q1={ccr_q1} | q2={ccr_q2} | "
                    f"iou={ccr_iou_sum / max(ccr_num_samples, 1):.6f} | "
                    f"corr_area={ccr_corr_area_sum / max(ccr_num_samples, 1):.6f} | "
                    f"trusted_expand={ccr_expand_area_sum / max(ccr_num_samples, 1):.6f} | "
                    f"trusted_shrink={ccr_shrink_area_sum / max(ccr_num_samples, 1):.6f} | "
                    f"anchor_ratio={ccr_anchor_ratio_sum / max(ccr_num_samples, 1):.6f} | "
                    f"late_override_ratio={ccr_late_override_sum / max(num_batches, 1):.6f} | "
                    f"loss_base={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_anchor={total_anchor_loss / max(num_batches, 1):.6f} | "
                    f"head_gamma={gamma_text}"
                )
            if use_despl:
                logger.log(
                    f"[DESPL] epoch={epoch:03d} | "
                    "use_despl_pseudo=True | "
                    f"use_despl_light_cache={use_despl_light} | "
                    f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'despl_fixed_blend')} | "
                    f"p_init_area_mean={despl_p_init_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"p_fixed_area_mean={despl_p_fixed_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"p_despl_area_mean={despl_p_despl_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"fixed_weight={fixed_weight:.2f} | "
                    f"teacher_weight={teacher_weight:.2f}"
                )

            val_results = {}
            for dataset_name in cfg.VAL_DATASETS:
                result = validate_one_dataset(
                    cfg,
                    student,
                    dataset_name,
                    device,
                    max_samples=sample_limit,
                )
                val_results[dataset_name] = result
                logger.log(f"[Validation] Epoch {epoch:03d}/{max_epoch:03d} | Dataset: {dataset_name}")
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
