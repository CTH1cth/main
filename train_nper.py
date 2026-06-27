import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common.metrics import CODMetrics
from common.utils import (
    Logger,
    check_nper_pseudo_bank,
    config_to_dict,
    ensure_dir,
    format_metric_table,
    load_config,
    metric_value,
    set_seed,
    torch_load,
    write_yaml,
)
from nper.losses import (
    boundary_loss,
    contrast_loss,
    dice_loss,
    entropy_loss,
    local_pseudo_loss,
    mnp_loss,
    partial_bce,
    weighted_bce_with_logits,
    weighted_iou_loss,
)
from nper.model_nper import NPERUCOD, update_ema_model
from nper.pseudo_evolution import QualityAwarePseudoEvolution
from nper.train_utils import build_eval_loader, build_train_loader, make_psta_view


def model_forward(model, images, batch):
    if hasattr(model, "forward_with_paths"):
        return model.forward_with_paths(images, batch.get("image_path"))
    return model(images)


def build_optimizer_scheduler(cfg, model, steps_per_epoch, max_epochs):
    lora_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "dino.adapters" in name:
            lora_params.append(param)
        else:
            decoder_params.append(param)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": float(cfg.LR_LORA), "name": "lora"})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": float(cfg.LR_DECODER), "name": "decoder"})
    if not groups:
        raise RuntimeError("No trainable NPER parameters found.")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.WEIGHT_DECAY))
    total_steps = max(1, int(steps_per_epoch) * int(max_epochs))
    warmup_steps = max(1, int(steps_per_epoch) * int(getattr(cfg, "WARMUP_EPOCHS", 0)))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def group_lrs(optimizer):
    out = {}
    for group in optimizer.param_groups:
        out[group.get("name", "group")] = float(group["lr"])
    return out


@torch.no_grad()
def validate_one_dataset(cfg, model, dataset_name, device, max_samples=-1):
    _, loader = build_eval_loader(cfg, split="val", datasets=[dataset_name], max_samples=max_samples)
    metrics = CODMetrics()
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True).float()
        gt = batch["gt"].to(device, non_blocking=True).float()
        out = model_forward(model, image, batch)
        logits = F.interpolate(out["logits"], size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = (logits.sigmoid() > float(cfg.THRESHOLD)).float()
        metrics.step(gt, pred)
    return metrics.get_result()


def save_checkpoint(path, epoch, cfg, student, teacher, optimizer, scheduler, best_metric, best_epoch):
    ensure_dir(Path(path).parent)
    payload = {
        "epoch": int(epoch),
        "method": "nper_ucod_v1",
        "backbone_key": cfg.BACKBONE_KEY,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config": config_to_dict(cfg),
    }
    if teacher is not None:
        payload["teacher"] = teacher.state_dict()
    torch.save(payload, path)


def anchor_loss(logits, anchor_fg, anchor_bg):
    target = torch.zeros_like(logits)
    target[anchor_fg] = 1.0
    target[anchor_bg] = 0.0
    mask = anchor_fg | anchor_bg
    return partial_bce(logits, target, mask)


def compute_losses(cfg, out, evo, batch):
    logits = out["logits"]
    prob = out["prob"]
    zero = logits.sum() * 0.0
    target = evo["p_evo"].detach()
    weight = evo["pixel_weight"].detach()
    loss_bce = weighted_bce_with_logits(logits, target, weight)
    loss_iou = weighted_iou_loss(prob, target, weight)
    loss_dice = dice_loss(prob, target, weight)
    loss_pseudo = loss_bce + loss_iou + loss_dice
    loss_anchor = anchor_loss(logits, evo["anchor_fg"], evo["anchor_bg"]) if float(cfg.LAMBDA_ANCHOR) != 0.0 else zero
    loss_local = (
        local_pseudo_loss(logits, evo["local_pseudo"], evo["local_mask"])
        if float(cfg.LAMBDA_LOCAL) != 0.0
        else zero
    )
    loss_mnp = (
        mnp_loss(prob, batch["sobel"].to(logits.device).float(), weight)
        if float(cfg.LAMBDA_MNP) != 0.0 and bool(getattr(cfg, "USE_MNP", True))
        else zero
    )
    loss_boundary = boundary_loss(out["boundary_logits"], target) if float(cfg.LAMBDA_BOUNDARY) != 0.0 else zero
    loss_entropy = entropy_loss(prob) if float(cfg.LAMBDA_ENTROPY) != 0.0 else zero
    loss_contrast = contrast_loss(out["features"], target) if float(cfg.LAMBDA_CONTRAST) != 0.0 else zero
    total = (
        float(cfg.LAMBDA_PSEUDO) * loss_pseudo
        + float(cfg.LAMBDA_ANCHOR) * loss_anchor
        + float(cfg.LAMBDA_MNP) * loss_mnp
        + float(cfg.LAMBDA_LOCAL) * loss_local
        + float(cfg.LAMBDA_CONTRAST) * loss_contrast
        + float(cfg.LAMBDA_BOUNDARY) * loss_boundary
        + float(cfg.LAMBDA_ENTROPY) * loss_entropy
    )
    return {
        "total": total,
        "pseudo": loss_pseudo,
        "bce": loss_bce,
        "iou": loss_iou,
        "dice": loss_dice,
        "anchor": loss_anchor,
        "local": loss_local,
        "mnp": loss_mnp,
        "contrast": loss_contrast,
        "boundary": loss_boundary,
        "entropy": loss_entropy,
    }


def maybe_psta_loss(cfg, model, images, epoch):
    if not bool(getattr(cfg, "USE_PSTA", True)) or int(epoch) < int(getattr(cfg, "PSTA_START_EPOCH", 6)):
        return images.sum() * 0.0
    scales = list(getattr(cfg, "PSTA_SCALES", [1.25, 1.0, 0.75]))
    views = [
        make_psta_view(
            images,
            scale=scale,
            color_jitter=bool(getattr(cfg, "PSTA_USE_COLOR_JITTER", True)),
            gaussian_blur=bool(getattr(cfg, "PSTA_USE_GAUSSIAN_BLUR", True)),
        )
        for scale in scales
    ]
    probs = [model(view)["prob"] for view in views]
    from nper.losses import psta_loss

    return psta_loss(probs[0], probs[1], probs[2])


def train_one_epoch(cfg, student, teacher, loader, optimizer, scheduler, scaler, evolver, device, epoch):
    student.train()
    if teacher is not None:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(1, int(getattr(cfg, "ACCUM_STEPS", 1)))
    use_amp = bool(getattr(cfg, "USE_AMP", True)) and device.type == "cuda"
    sums = {}
    count = 0
    for step, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model_forward(student, images, batch)
            teacher_out = None
            if (
                teacher is not None
                and bool(getattr(cfg, "PSEUDO_USE_TEACHER", True))
                and int(epoch) >= int(getattr(cfg, "EVOLUTION_START_EPOCH", 4))
            ):
                with torch.no_grad():
                    teacher_out = model_forward(teacher, images, batch)
            evo = evolver.evolve(batch, out, teacher_out, epoch)
            losses = compute_losses(cfg, out, evo, batch)
            loss_psta = maybe_psta_loss(cfg, student, images, epoch)
            losses["psta"] = loss_psta
            loss = losses["total"] + float(cfg.LAMBDA_PSTA) * loss_psta
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if step % accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if teacher is not None:
                update_ema_model(student, teacher, momentum=float(cfg.EMA_MOMENTUM))

        batch_size = int(images.shape[0])
        count += batch_size
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().item()) * batch_size
        sums["quality"] = sums.get("quality", 0.0) + float(batch["quality_score"].float().mean().item()) * batch_size
        sums["hard"] = sums.get("hard", 0.0) + float(batch["hard_score"].float().mean().item()) * batch_size
        sums["teacher_weight"] = sums.get("teacher_weight", 0.0) + float(evo["teacher_weight_mean"].detach().item()) * batch_size
        sums["local_ratio"] = sums.get("local_ratio", 0.0) + float(evo["local_pseudo_ratio"].detach().item()) * batch_size
        sums["anchor_ratio"] = sums.get("anchor_ratio", 0.0) + float(evo["anchor_ratio"].detach().item()) * batch_size

    return {name: value / max(1, count) for name, value in sums.items()}


def main():
    parser = argparse.ArgumentParser(description="Train NPER-UCOD-V1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.SEED))
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(cfg.MAX_EPOCH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / "train"
    ckpt_dir = out_dir / "ckpt"
    ensure_dir(out_dir)
    write_yaml(out_dir / "config.yaml", config_to_dict(cfg))

    check_nper_pseudo_bank(cfg, max_samples=args.max_samples)
    train_dataset, train_loader = build_train_loader(cfg, max_samples=args.max_samples)
    student = NPERUCOD(cfg).to(device)
    teacher = None
    if bool(getattr(cfg, "USE_EMA_TEACHER", True)):
        teacher = NPERUCOD(cfg).to(device)
        teacher.load_state_dict(student.state_dict())
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
    optimizer, scheduler = build_optimizer_scheduler(cfg, student, len(train_loader), max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(cfg, "USE_AMP", True)) and device.type == "cuda")
    evolver = QualityAwarePseudoEvolution(cfg)

    start_epoch = 1
    best_metric = None
    best_epoch = 0
    if args.resume:
        checkpoint = torch_load(args.resume, map_location="cpu")
        student.load_state_dict(checkpoint["student"])
        if teacher is not None and "teacher" in checkpoint:
            teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = checkpoint.get("best_metric")
        best_epoch = int(checkpoint.get("best_epoch", 0))

    with Logger(out_dir / "train.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"exp_name = {cfg.EXP_NAME}")
        logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', 'quality_fusion')}")
        logger.log(f"use_gcm = {str(bool(getattr(cfg, 'PSEUDO_USE_GCM', True))).lower()}")
        logger.log(f"use_despl = {str(bool(getattr(cfg, 'PSEUDO_USE_DESPL', True))).lower()}")
        logger.log(f"use_teacher = {str(bool(getattr(cfg, 'PSEUDO_USE_TEACHER', True) and teacher is not None)).lower()}")
        logger.log(f"use_psta = {str(bool(getattr(cfg, 'USE_PSTA', True))).lower()}")
        logger.log(f"use_mnp = {str(bool(getattr(cfg, 'USE_MNP', True))).lower()}")
        logger.log(f"num_train_samples = {len(train_dataset)}")
        logger.log(f"first_pseudo_bank = {train_dataset.first_cache_path}")
        logger.log(f"pseudo_shape = {train_dataset.pseudo_shape}")
        logger.log(f"max_epochs = {max_epochs}")
        logger.log(f"resnet18_detail_pretrained = {str(bool(getattr(student.detail, 'pretrained', False))).lower()}")
        logger.log(f"resnet18_weight_source = {getattr(student.detail, 'weight_source', 'unknown')}")
        logger.log("train_gt_used = false")
        for epoch in range(start_epoch, max_epochs + 1):
            stats = train_one_epoch(
                cfg, student, teacher, train_loader, optimizer, scheduler, scaler, evolver, device, epoch
            )
            lrs = group_lrs(optimizer)
            logger.log(
                "[NPER][Epoch {epoch:03d}] "
                "lr_lora={lr_lora:.6g} lr_decoder={lr_decoder:.6g} "
                "loss_total={total:.6f} loss_pseudo={pseudo:.6f} loss_anchor={anchor:.6f} "
                "loss_mnp={mnp:.6f} loss_local={local:.6f} loss_psta={psta:.6f} "
                "loss_boundary={boundary:.6f} loss_entropy={entropy:.6f} "
                "mean_quality={quality:.6f} mean_hard={hard:.6f} "
                "teacher_weight_mean={teacher_weight:.6f} local_pseudo_ratio={local_ratio:.6f} "
                "anchor_ratio={anchor_ratio:.6f}".format(
                    epoch=epoch,
                    lr_lora=lrs.get("lora", 0.0),
                    lr_decoder=lrs.get("decoder", 0.0),
                    **stats,
                )
            )
            metrics = validate_one_dataset(cfg, student, cfg.BEST_DATASET, device, max_samples=args.max_samples)
            logger.log(f"[Val][Epoch {epoch:03d}] Dataset: {cfg.BEST_DATASET}")
            logger.log(format_metric_table(metrics))
            current = metric_value(metrics, cfg.BEST_METRIC)
            improved = False
            if best_metric is None:
                improved = True
            elif cfg.BEST_MODE == "min":
                improved = current < float(best_metric)
            else:
                improved = current > float(best_metric)
            if improved:
                best_metric = current
                best_epoch = epoch
                save_checkpoint(ckpt_dir / "best.pth", epoch, cfg, student, teacher, optimizer, scheduler, best_metric, best_epoch)
                logger.log(f"[Checkpoint] best updated | epoch={epoch} {cfg.BEST_METRIC}={current:.6f}")
            if epoch % int(cfg.SAVE_INTERVAL) == 0 or epoch == max_epochs:
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
        logger.log(f"best_epoch = {best_epoch}")
        logger.log(f"best_metric = {best_metric}")


if __name__ == "__main__":
    main()
