import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.dataset import CachedEvalDataset, CachedTrainDataset
from common.gkd_lite import (
    apply_gkd_strength,
    compute_gkd_branch_loss,
    compute_gkd_quality,
    compute_pixel_weight,
    compute_plain_and_reweight_loss_for_audit,
    get_gkd_mode,
    is_gkd_enabled,
    is_gkd_v3_enabled,
)
from common.metrics import CODMetrics
from common.utils import (
    Logger,
    cache_status,
    check_ccr_cache,
    check_despl_light_cache,
    check_despl_paper_cache,
    check_despl_pseudo_bank,
    check_drepp_cache,
    check_ml_feature_cache,
    check_qra_cache,
    config_to_dict,
    current_lr,
    ensure_dir,
    ensure_cache_available,
    format_metric_table,
    load_config,
    metric_value,
    ml_feature_cache_dir,
    set_seed,
    write_yaml,
)
from model import build_seg_head, update_ema


def get_teacher_weight(epoch):
    # 1-based epoch: 前 20 轮从 fixed pseudo 平滑过渡到 teacher pseudo。
    if epoch <= 20:
        return (epoch - 1) / 20.0
    return 1.0


def get_reset_epoch(cfg):
    return int(getattr(cfg, "FINETUNE_RESET_EPOCH", 21))


def get_fusion_weights(epoch, cfg):
    reset_epoch = get_reset_epoch(cfg)
    mode = str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower()
    if mode == "orig20_hold_until_reset":
        if epoch >= reset_epoch:
            return 0.0, 1.0
        decay_epochs = int(getattr(cfg, "FUSION_ORIG_DECAY_EPOCHS", 20))
        hold_fixed = float(getattr(cfg, "FUSION_HOLD_FIXED_WEIGHT", 0.05))
        hold_fixed = max(0.0, min(1.0, hold_fixed))
        if epoch <= decay_epochs:
            fixed_weight = 1.0 - 0.05 * float(epoch - 1)
            fixed_weight = max(hold_fixed, fixed_weight)
        else:
            fixed_weight = hold_fixed
        fixed_weight = max(0.0, min(1.0, float(fixed_weight)))
        return fixed_weight, 1.0 - fixed_weight

    if mode == "linear_to_095_before_reset":
        if epoch >= reset_epoch:
            return 0.0, 1.0
        pre_epochs = int(getattr(cfg, "TEACHER_FUSION_PRE_RESET_EPOCHS", reset_epoch - 1))
        if hasattr(cfg, "FUSION_MIN_FIXED_WEIGHT"):
            min_fixed = float(getattr(cfg, "FUSION_MIN_FIXED_WEIGHT"))
        else:
            max_teacher = float(getattr(cfg, "TEACHER_FUSION_MAX_WEIGHT", 0.95))
            min_fixed = 1.0 - max_teacher
        min_fixed = max(0.0, min(1.0, min_fixed))
        if pre_epochs <= 1:
            fixed_weight = min_fixed
        else:
            progress = float(epoch - 1) / float(max(1, pre_epochs - 1))
            fixed_weight = 1.0 - (1.0 - min_fixed) * progress
        fixed_weight = max(min_fixed, min(1.0, float(fixed_weight)))
        return fixed_weight, 1.0 - fixed_weight

    teacher_weight = get_teacher_weight(epoch)
    return 1.0 - teacher_weight, teacher_weight


def get_fixed_teacher_weights(cfg, epoch):
    use_fast = bool(getattr(cfg, "USE_FAST_TEACHER_FUSION", False))
    mode = str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower()
    if use_fast and mode == "fast_t10":
        max_teacher = float(getattr(cfg, "FAST_T10_MAX_TEACHER_WEIGHT", 0.95))
        start = int(getattr(cfg, "FAST_T10_START_EPOCH", 2))
        end = int(getattr(cfg, "FAST_T10_END_EPOCH", 10))
        hold_end = int(getattr(cfg, "FAST_T10_HOLD_END_EPOCH", 20))
        if epoch < start:
            teacher_weight = 0.0
        elif epoch <= end:
            denom = max(1, end - start + 1)
            teacher_weight = ((epoch - start + 1) / denom) * max_teacher
        elif epoch <= hold_end:
            teacher_weight = max_teacher
        else:
            teacher_weight = 1.0
        teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
        return 1.0 - teacher_weight, teacher_weight, "fast_t10"

    fixed_weight, teacher_weight = get_fusion_weights(epoch, cfg)
    if mode in {"linear_to_095_before_reset", "orig20_hold_until_reset"}:
        return fixed_weight, teacher_weight, mode
    return fixed_weight, teacher_weight, "default"


def get_hold_cosine_lr(epoch, cfg):
    base_lr = float(getattr(cfg, "COMPLEX_HEAD_PRE_RESET_LR", 3e-4))
    min_lr = float(getattr(cfg, "COMPLEX_HEAD_LR_MIN", 3e-5))
    hold_epochs = int(getattr(cfg, "COMPLEX_HEAD_LR_HOLD_EPOCHS", 10))
    reset_epoch = get_reset_epoch(cfg)
    pre_epochs = reset_epoch - 1
    if epoch <= hold_epochs:
        return base_lr
    progress = float(epoch - hold_epochs) / float(max(1, pre_epochs - hold_epochs))
    progress = max(0.0, min(1.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def use_complex_head_lr_policy(cfg):
    return str(getattr(cfg, "COMPLEX_HEAD_LR_POLICY", "")).lower() == "hold_cosine_before_reset"


def apply_complex_head_lr_policy(optimizer, epoch, cfg):
    if not use_complex_head_lr_policy(cfg) or epoch >= get_reset_epoch(cfg):
        return None
    lr = get_hold_cosine_lr(epoch, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def should_step_iter_scheduler(epoch, cfg):
    return not (
        use_linear_floor_two_stage_lr(cfg)
        or (use_complex_head_lr_policy(cfg) and epoch < get_reset_epoch(cfg))
    )


def complex_head_post_reset_lr(cfg):
    return float(getattr(cfg, "COMPLEX_HEAD_POST_RESET_LR", cfg.DINO["lr"]))


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def get_base_lr(cfg):
    return float(getattr(cfg, "LR", cfg.DINO["lr"]))


def use_linear_floor_two_stage_lr(cfg):
    return str(getattr(cfg, "LR_POLICY", "")).lower() == "linear_floor_two_stage"


def compute_linear_floor_two_stage_lr(epoch, iter_idx, num_iters_per_epoch, cfg):
    lr0 = get_base_lr(cfg)
    lr_floor = float(getattr(cfg, "LR_FLOOR", 2e-5))
    reset_epoch = get_reset_epoch(cfg)
    num_iters_per_epoch = max(1, int(num_iters_per_epoch))

    if epoch < reset_epoch:
        stage_epochs = int(getattr(cfg, "LR_LINEAR_STAGE1_EPOCHS", reset_epoch - 1))
        stage_epoch_idx = epoch - 1
    else:
        stage_epochs = int(getattr(cfg, "LR_LINEAR_STAGE2_EPOCHS", 10))
        stage_epoch_idx = epoch - reset_epoch

    stage_epochs = max(1, stage_epochs)
    total_steps = max(1, stage_epochs * num_iters_per_epoch - 1)
    stage_step = stage_epoch_idx * num_iters_per_epoch + int(iter_idx)
    progress = min(1.0, max(0.0, float(stage_step) / float(total_steps)))
    lr = lr_floor + (lr0 - lr_floor) * (1.0 - progress)
    return max(lr_floor, float(lr))


def build_optimizer_scheduler(cfg, student, lr=None):
    # 与 UCOD-DPL 对齐：AdamW + 每 iteration StepLR。
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.DINO["lr"] if lr is None else lr))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.95,
    )
    return optimizer, scheduler


def apply_lr_floor(optimizer, cfg):
    if not bool(getattr(cfg, "USE_LR_FLOOR", False)):
        return False, None, None

    mode = str(getattr(cfg, "LR_FLOOR_MODE", "global")).lower()
    if mode != "global":
        raise RuntimeError(f"Unsupported LR_FLOOR_MODE: {mode}. Only 'global' is implemented.")

    lr_floor = float(getattr(cfg, "LR_FLOOR", 0.0))
    if lr_floor <= 0.0:
        return False, None, None

    old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    clamped = False
    for group in optimizer.param_groups:
        if float(group["lr"]) < lr_floor:
            group["lr"] = lr_floor
            clamped = True
    new_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return clamped, min(old_lrs), min(new_lrs)


def use_multi_level_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))


def use_dagp_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp"


def use_dagp_safe_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe"


def use_raw_feature_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {"dagp", "dagp_safe"}


def use_ndr_branch(cfg):
    return bool(getattr(cfg, "USE_NDR_BRANCH", False))


def set_model_epoch(model, epoch):
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_epoch"):
        target.set_epoch(epoch)


def make_image_68(cfg, batch, device):
    if not use_ndr_branch(cfg):
        return None
    if "image_68" not in batch:
        raise KeyError("USE_NDR_BRANCH=True requires batch['image_68'].")
    return batch["image_68"].to(device, non_blocking=True).float()


def forward_seg_head(model, model_input, cfg, image_68=None, return_aux=False):
    if use_dagp_safe_head(cfg):
        return model(model_input, image_68=image_68, return_aux=return_aux)
    return model(model_input)


SAP_DEBUG_KEYS = (
    "x10_abs_mean",
    "x11_abs_mean",
    "x12_abs_mean",
    "caff_10_11_abs_mean",
    "caff_11_12_abs_mean",
    "fusion_abs_mean",
    "final_logits_abs_mean",
)


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


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def resize_logits_for_loss(logits, cfg):
    target_size = (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    if tuple(logits.shape[-2:]) == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def _dagp_scalar(model, name):
    value = getattr(model, name, None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def log_dagp_first_batch(logger, student, model_input, raw_logits, loss_logits, pseudo):
    logger.log(f"[DAGP FirstBatch] feature shape = {list(model_input.shape)}")
    logger.log(f"[DAGP FirstBatch] raw logits shape = {list(raw_logits.shape)}")
    logger.log(f"[DAGP FirstBatch] loss logits shape = {list(loss_logits.shape)}")
    logger.log(f"[DAGP FirstBatch] pseudo shape = {list(pseudo.shape)}")
    logger.log(f"[DAGP FirstBatch] alpha = {_dagp_scalar(student, 'alpha'):.8f}")
    logger.log(f"[DAGP FirstBatch] gamma = {_dagp_scalar(student, 'gamma'):.8f}")
    logger.log(f"[DAGP FirstBatch] topk = {int(getattr(student, 'topk'))}")
    logger.log(f"[DAGP FirstBatch] tau = {float(getattr(student, 'tau')):.6f}")


def output_scalar(output, name, default=0.0):
    if not isinstance(output, dict) or name not in output:
        return float(default)
    value = output[name]
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def log_dagp_safe_first_batch(logger, student, model_input, raw_logits, loss_logits, pseudo, output):
    logger.log(f"[DAGP-Safe FirstBatch] feature shape = {list(model_input.shape)}")
    logger.log(f"[DAGP-Safe FirstBatch] raw logits shape = {list(raw_logits.shape)}")
    logger.log(f"[DAGP-Safe FirstBatch] loss logits shape = {list(loss_logits.shape)}")
    logger.log(f"[DAGP-Safe FirstBatch] pseudo shape = {list(pseudo.shape)}")
    logger.log(f"[DAGP-Safe FirstBatch] epoch = {int(getattr(student, 'current_epoch_tensor').item())}")
    logger.log(f"[DAGP-Safe FirstBatch] scale = {output_scalar(output, 'dagp_scale'):.8f}")
    logger.log(f"[DAGP-Safe FirstBatch] alpha_eff = {output_scalar(output, 'dagp_alpha_eff'):.8f}")
    logger.log(f"[DAGP-Safe FirstBatch] gamma_eff = {output_scalar(output, 'dagp_gamma_eff'):.8f}")
    logger.log(f"[DAGP-Safe FirstBatch] topk = {int(getattr(student, 'topk'))}")
    logger.log(f"[DAGP-Safe FirstBatch] tau = {float(getattr(student, 'tau')):.6f}")
    logger.log(f"[DAGP-Safe FirstBatch] use_prob_gate = {bool(getattr(student, 'use_prob_gate'))}")
    logger.log(
        f"[DAGP-Safe FirstBatch] use_uncertainty_output_gate = "
        f"{bool(getattr(student, 'use_uncertainty_output_gate', False))}"
    )
    unc_gate_mean = output_scalar(output, "uncertainty_gate_mean", -1.0)
    unc_gate_min = output_scalar(output, "uncertainty_gate_min", -1.0)
    unc_gate_max = output_scalar(output, "uncertainty_gate_max", -1.0)
    logger.log(
        f"[DAGP-Safe FirstBatch] uncertainty_gate_mean/min/max = "
        f"{unc_gate_mean:.8f}/{unc_gate_min:.8f}/{unc_gate_max:.8f}"
    )


def log_ndr_first_batch(logger, image_68, output, pseudo):
    logger.log(f"[NDR FirstBatch] image_68 shape = {list(image_68.shape)}")
    logger.log(f"[NDR FirstBatch] sobel_68 shape = {list(output['sobel_68'].shape)}")
    logger.log(f"[NDR FirstBatch] coarse_logits_37 shape = {list(output['coarse_logits_37'].shape)}")
    logger.log(f"[NDR FirstBatch] coarse_logits_68 shape = {list(output['coarse_logits_68'].shape)}")
    logger.log(f"[NDR FirstBatch] residual_logits_68 shape = {list(output['residual_logits_68'].shape)}")
    logger.log(f"[NDR FirstBatch] final_logits_68 shape = {list(output['logits'].shape)}")
    logger.log(f"[NDR FirstBatch] pseudo shape = {list(pseudo.shape)}")
    logger.log(f"[NDR FirstBatch] beta_eff = {output_scalar(output, 'ndr_beta_eff'):.8f}")
    gate_mean = output_scalar(output, "ndr_detail_gate_mean")
    gate_min = output_scalar(output, "ndr_detail_gate_min")
    gate_max = output_scalar(output, "ndr_detail_gate_max")
    logger.log(
        f"[NDR FirstBatch] detail_gate mean/min/max = "
        f"{gate_mean:.8f}/{gate_min:.8f}/{gate_max:.8f}"
    )
    residual_abs_mean = output_scalar(output, "ndr_residual_abs_mean")
    residual_abs_max = output_scalar(output, "ndr_residual_abs_max")
    logger.log(
        f"[NDR FirstBatch] residual abs mean/max = "
        f"{residual_abs_mean:.8f}/{residual_abs_max:.8f}"
    )


def output_tensor_abs_mean(output, name):
    if not isinstance(output, dict) or name not in output:
        return 0.0
    return float(output[name].detach().abs().mean().cpu().item())


def output_context_scale(output):
    if not isinstance(output, dict) or "context_scale" not in output:
        return None
    return float(output["context_scale"].detach().cpu().item())


def output_debug_value(output, name):
    if not isinstance(output, dict):
        return 0.0
    debug = output.get("debug")
    if not isinstance(debug, dict) or name not in debug:
        return 0.0
    value = debug[name]
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def dataloader_worker_kwargs(cfg):
    num_workers = int(cfg.NUM_WORKERS)
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "DATALOADER_PERSISTENT_WORKERS", False))
        prefetch_factor = int(getattr(cfg, "DATALOADER_PREFETCH_FACTOR", 2))
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


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
        **dataloader_worker_kwargs(cfg),
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
        **dataloader_worker_kwargs(cfg),
    )
    metrics = CODMetrics()
    student.eval()
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True).float()
        model_input = make_model_input(cfg, batch, device)
        image_68 = make_image_68(cfg, batch, device)
        logits = extract_logits(forward_seg_head(student, model_input, cfg, image_68=image_68, return_aux=False))
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


def current_time_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def should_save_epoch_checkpoint(epoch, max_epoch, save_interval, reset_epoch=None):
    if reset_epoch is not None and int(epoch) == int(reset_epoch):
        return True
    last_checkpoint_start = max(1, int(max_epoch) - 4)
    interval = int(save_interval)
    interval_due = interval > 0 and int(epoch) % interval == 0
    return interval_due or int(epoch) >= last_checkpoint_start


def log_cache_summary(logger, cfg, train_dataset):
    # 训练日志开头记录 cache 规模和 shape，方便确认读到的是正确 backbone。
    logger.log(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger.log(f"train datasets = {' + '.join(cfg.TRAIN_DATASETS)}")
    logger.log(f"num_train_samples = {len(train_dataset)}")
    feature_root = (
        ml_feature_cache_dir(cfg)
        if use_multi_level_feature(cfg)
        else Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY
    )
    logger.log(
        "feature cache path = "
        f"{feature_root.resolve()}"
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
    if getattr(cfg, "USE_DREPP", False):
        logger.log(f"DRE++ cache path = {train_dataset.drepp_cache_root}")
        logger.log(f"first DRE++ cache file = {train_dataset.drepp_first_cache_path}")
        logger.log("USE_DREPP=True")
        logger.log("DESPL-core active = true")
        logger.log("memory initialized = true")
        logger.log("teacher_uncertain_only=true")
        logger.log("fixed_local_only=true")
        logger.log(f"local_refine={bool(getattr(cfg, 'USE_LOCAL_REFINE', False))}")
        logger.log("global_blend=false")
    if getattr(cfg, "USE_DESPL_PSEUDO", False):
        fixed_weight_in_init = float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.2))
        use_fixed_in_pseudo = (
            str(getattr(cfg, "P_INIT_MODE", "")) not in {"despl_only", "despl_paper_only"}
            and abs(fixed_weight_in_init) > 0.0
        )
        logger.log(f"DESPL pseudo cache path = {train_dataset.despl_cache_root}")
        logger.log(f"first DESPL pseudo file = {train_dataset.despl_first_cache_path}")
        logger.log(f"use_despl_pseudo = true")
        logger.log(f"use_despl_paper_cache = {bool(getattr(cfg, 'USE_DESPL_PAPER_CACHE', False))}")
        logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
        logger.log(f"use_dre_safe_prior = {bool(getattr(cfg, 'USE_DRE_SAFE_PRIOR', False))}")
        logger.log(f"use_late_despl_anchor_loss = {bool(getattr(cfg, 'USE_LATE_DESPL_ANCHOR_LOSS', False))}")
        logger.log(f"use_gcm = {bool(getattr(cfg, 'PSEUDO_USE_GCM', False))}")
        logger.log(f"use_pure_despl_supervision = {bool(getattr(cfg, 'USE_PURE_DESPL_SUPERVISION', False))}")
        logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', 'despl_fixed_blend')}")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "despl_paper_only":
            logger.log("p_init_formula = p_despl_paper_soft")
        elif str(getattr(cfg, "P_INIT_MODE", "")) == "despl_only":
            logger.log("p_init_formula = p_despl")
        else:
            logger.log(
                "p_init_formula = "
                f"{float(getattr(cfg, 'P_INIT_DESPL_WEIGHT', 0.8)):.3f}*p_despl + "
                f"{float(getattr(cfg, 'P_INIT_FIXED_WEIGHT', 0.2)):.3f}*p_fixed"
            )
        logger.log(f"use_fixed_in_pseudo = {use_fixed_in_pseudo}")
        logger.log(f"fixed_used_for_training = {use_fixed_in_pseudo}")
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
    if use_multi_level_feature(cfg):
        for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12]):
            name = f"feature_l{int(layer)}"
            tensor = batch[name].float()
            logger.log(f"first batch {name} tensor shape = {list(tensor.shape)}")
            logger.log(
                f"first batch {name} mean/std = "
                f"{float(tensor.mean().item()):.6f}/{float(tensor.std(unbiased=False).item()):.6f}"
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
    if getattr(cfg, "USE_DREPP", False):
        logger.log(
            "first batch DRE++ p_despl_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_p_despl_area"])
        )
        logger.log(
            "first batch DRE++ fixed_local_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_fixed_local_area"])
        )
        logger.log(
            "first batch DRE++ core_fg_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_core_fg_area"])
        )
        logger.log(
            "first batch DRE++ core_bg_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_core_bg_area"])
        )
        logger.log(
            "first batch DRE++ uncertain_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_uncertain_area"])
        )
        logger.log(
            "first batch DRE++ boundary_band_area = "
            + ",".join(f"{float(value):.6f}" for value in batch["drepp_boundary_band_area"])
        )
        logger.log(f"first batch DRE++ global_blend = {batch['drepp_global_blend'].tolist()}")
    if getattr(cfg, "USE_DESPL_PSEUDO", False):
        logger.log(f"first batch pseudo_fixed tensor shape = {list(batch['pseudo_fixed'].shape)}")
        logger.log(f"first batch pseudo_despl tensor shape = {list(batch['pseudo_despl'].shape)}")
        if bool(getattr(cfg, "USE_DESPL_ANCHOR_PBCE", False)):
            theta_fg = float(getattr(cfg, "ANCHOR_PBCE_THETA_FG", 0.70))
            theta_bg = float(getattr(cfg, "ANCHOR_PBCE_THETA_BG", 0.30))
            if theta_fg <= theta_bg:
                raise RuntimeError(f"ANCHOR_PBCE_THETA_FG must be > THETA_BG, got {theta_fg} <= {theta_bg}")
            anchor_source = batch.get("pseudo_despl", batch["pseudo"]).float().clamp(0.0, 1.0)
            if anchor_source.ndim != 4 or anchor_source.shape[1] != 1:
                raise RuntimeError(
                    "DESPL Anchor-PBCE first batch source must be [B,1,H,W], "
                    f"got {list(anchor_source.shape)}"
                )
            anchor_fg = anchor_source >= theta_fg
            anchor_bg = anchor_source <= theta_bg
            anchor_valid = anchor_fg | anchor_bg
            logger.log(f"first batch anchor source shape = {list(anchor_source.shape)}")
            logger.log(f"first batch anchor valid_ratio = {float(anchor_valid.float().mean().item()):.6f}")
            logger.log(f"first batch anchor fg_ratio = {float(anchor_fg.float().mean().item()):.6f}")
            logger.log(f"first batch anchor bg_ratio = {float(anchor_bg.float().mean().item()):.6f}")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "despl_paper_only":
            logger.log(f"first batch pseudo_despl_paper tensor shape = {list(batch['pseudo_despl_paper'].shape)}")
            logger.log(
                "first batch p_despl_paper_area = "
                + ",".join(f"{float(value):.6f}" for value in batch["p_despl_paper_area"])
            )
            logger.log(
                "first batch p_despl_paper_view_consistency = "
                + ",".join(f"{float(value):.6f}" for value in batch["p_despl_paper_view_consistency"])
            )
        if getattr(cfg, "USE_DRE_SAFE_PRIOR", False):
            logger.log(f"first batch pseudo_base tensor shape = {list(batch['pseudo_base'].shape)}")
            logger.log(f"first batch pseudo_safe tensor shape = {list(batch['pseudo_safe'].shape)}")
            logger.log(
                "first batch dre_safe_candidate_ratio = "
                + ",".join(f"{float(value):.6f}" for value in batch["dre_safe_candidate_ratio"])
            )
            logger.log(
                "first batch dre_safe_fallback = "
                + ",".join(str(bool(value)) for value in batch["dre_safe_fallback"])
            )
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


def init_gkd_epoch_stats():
    return {
        "num_samples": 0,
        "num_high": 0,
        "num_normal": 0,
        "num_low": 0,
        "q_score_sum": 0.0,
        "sample_weight_sum": 0.0,
        "pixel_weight_sum": 0.0,
        "area_sum": 0.0,
        "num_cc_sum": 0.0,
        "largest_cc_ratio_sum": 0.0,
        "edge_touch_ratio_sum": 0.0,
        "despl_fixed_iou_sum": 0.0,
        "despl_fixed_iou_count": 0,
        "last_strength": 1.0,
    }


def update_gkd_epoch_stats(epoch_stats, q_info, sample_weight, pixel_weight, strength):
    stats = q_info["stats"]
    num_samples = int(stats.get("num_samples", 0))
    if num_samples <= 0:
        return
    epoch_stats["num_samples"] += num_samples
    epoch_stats["num_high"] += int(stats.get("num_high", 0))
    epoch_stats["num_normal"] += int(stats.get("num_normal", 0))
    epoch_stats["num_low"] += int(stats.get("num_low", 0))
    epoch_stats["q_score_sum"] += float(stats.get("q_score_mean", 0.0)) * num_samples
    epoch_stats["sample_weight_sum"] += float(sample_weight.detach().mean().item()) * num_samples
    epoch_stats["pixel_weight_sum"] += float(pixel_weight.detach().mean().item()) * num_samples
    epoch_stats["area_sum"] += float(stats.get("area_mean", 0.0)) * num_samples
    epoch_stats["num_cc_sum"] += float(stats.get("num_cc_mean", 0.0)) * num_samples
    epoch_stats["largest_cc_ratio_sum"] += (
        float(stats.get("largest_cc_ratio_mean", 0.0)) * num_samples
    )
    epoch_stats["edge_touch_ratio_sum"] += (
        float(stats.get("edge_touch_ratio_mean", 0.0)) * num_samples
    )
    iou_count = int(stats.get("despl_fixed_iou_count", 0))
    if iou_count > 0:
        epoch_stats["despl_fixed_iou_sum"] += (
            float(stats.get("despl_fixed_iou_mean", 0.0)) * iou_count
        )
        epoch_stats["despl_fixed_iou_count"] += iou_count
    epoch_stats["last_strength"] = float(strength)


def log_gkd_first_batch(logger, batch, q_info, prefix="[GKD-Audit FirstBatch]"):
    logger.log(prefix)
    logger.log("idx dataset stem area num_cc lcc_ratio edge_touch iou_df q_score grade sample_w")
    datasets = list(batch.get("dataset", []))
    stems = list(batch.get("stem", []))
    for idx, item in enumerate(q_info["per_sample"][:16]):
        dataset = str(datasets[idx]) if idx < len(datasets) else "NA"
        stem = str(stems[idx]) if idx < len(stems) else "NA"
        logger.log(
            f"{idx} {dataset} {stem} "
            f"{float(item['area']):.4f} "
            f"{int(item['num_cc'])} "
            f"{float(item['largest_cc_ratio']):.4f} "
            f"{float(item['edge_touch_ratio']):.4f} "
            f"{float(item['despl_fixed_iou']):.4f} "
            f"{float(item['q_score']):.4f} "
            f"{int(item['grade'])} "
            f"{float(item['sample_weight']):.4f}"
        )


def format_gkd_epoch_log(epoch, epoch_stats):
    num_samples = max(int(epoch_stats["num_samples"]), 1)
    iou_count = int(epoch_stats["despl_fixed_iou_count"])
    iou_mean = (
        epoch_stats["despl_fixed_iou_sum"] / iou_count
        if iou_count > 0
        else -1.0
    )
    return (
        f"[GKD-Lite] epoch={epoch:03d} | "
        "enabled=True | "
        f"strength={epoch_stats['last_strength']:.3f} | "
        f"high={epoch_stats['num_high']} | "
        f"normal={epoch_stats['num_normal']} | "
        f"low={epoch_stats['num_low']} | "
        f"q_score_mean={epoch_stats['q_score_sum'] / num_samples:.6f} | "
        f"sample_w_mean={epoch_stats['sample_weight_sum'] / num_samples:.6f} | "
        f"pixel_w_mean={epoch_stats['pixel_weight_sum'] / num_samples:.6f} | "
        f"area_mean={epoch_stats['area_sum'] / num_samples:.6f} | "
        f"cc_mean={epoch_stats['num_cc_sum'] / num_samples:.6f} | "
        f"lcc_ratio_mean={epoch_stats['largest_cc_ratio_sum'] / num_samples:.6f} | "
        f"edge_touch_mean={epoch_stats['edge_touch_ratio_sum'] / num_samples:.6f} | "
        f"despl_fixed_iou_mean={iou_mean:.6f}"
    )


GKD_AUDIT_HEADERS = [
    "epoch",
    "mode",
    "strength",
    "batch_calls",
    "sample_calls",
    "high_count",
    "normal_count",
    "low_count",
    "high_ratio",
    "normal_ratio",
    "low_ratio",
    "plain_loss",
    "reweight_loss",
    "relative_delta",
    "q_score_mean",
    "sample_weight_mean",
    "sample_weight_std",
    "pixel_weight_mean",
    "pixel_weight_std",
    "pixel_weight_p10",
    "pixel_weight_p50",
    "pixel_weight_p90",
    "pixel_weight_lt_0_7_ratio",
    "pixel_weight_gt_0_95_ratio",
    "area_mean",
    "num_cc_mean",
    "largest_cc_ratio_mean",
    "edge_touch_ratio_mean",
    "despl_fixed_iou_mean",
]


GKD_BRANCH_HEADERS = [
    "epoch",
    "mode",
    "strength",
    "batch_calls",
    "sample_calls",
    "high_count",
    "normal_count",
    "low_count",
    "high_ratio",
    "normal_ratio",
    "low_ratio",
    "plain_bce",
    "branch_loss",
    "branch_delta",
    "low_loss",
    "normal_loss",
    "high_loss",
    "high_bce",
    "high_l1",
    "high_mse",
    "teacher_l1",
    "teacher_bce",
    "high_extra_ratio",
    "pixel_weight_mean",
    "pixel_weight_std",
    "entropy_mean",
    "entropy_std",
    "entropy_high_ratio",
]


GKD_BRANCH_V2_HEADERS = [
    "epoch",
    "mode",
    "strength",
    "batch_calls",
    "sample_calls",
    "old_high_count",
    "old_normal_count",
    "old_low_count",
    "old_high_ratio",
    "old_normal_ratio",
    "old_low_ratio",
    "final_high_count",
    "final_normal_count",
    "final_low_count",
    "final_high_ratio",
    "final_normal_ratio",
    "final_low_ratio",
    "strict_high_block_count",
    "hard_downgrade_count",
    "static_low_count",
    "teacher_low_count",
    "teacher_normal_cap_count",
    "dynamic_low_count",
    "teacher_despl_iou_mean",
    "teacher_despl_iou_p10",
    "teacher_despl_iou_p50",
    "teacher_despl_iou_p90",
    "teacher_area_mean",
    "high_extra_ratio",
    "teacher_bce",
]


GKD_BRANCH_V3_HEADERS = [
    "epoch",
    "mode",
    "disabled_after_epoch",
    "loss_used",
    "batch_calls",
    "sample_calls",
    "old_high_count",
    "old_normal_count",
    "old_low_count",
    "old_high_ratio",
    "old_normal_ratio",
    "old_low_ratio",
    "final_high_count",
    "final_normal_count",
    "final_low_count",
    "final_high_ratio",
    "final_normal_ratio",
    "final_low_ratio",
    "strict_high_block_count",
    "hard_downgrade_count",
    "static_low_count",
    "dynamic_high_cap_count",
    "teacher_low_count",
    "dynamic_low_count",
    "teacher_despl_iou_mean",
    "teacher_despl_iou_p10",
    "teacher_despl_iou_p50",
    "teacher_despl_iou_p90",
    "teacher_area_mean",
    "plain_bce",
    "branch_loss",
    "branch_delta",
    "low_loss",
    "normal_loss",
    "high_loss",
    "high_bce",
    "high_l1",
    "high_mse",
    "teacher_l1",
    "teacher_bce",
    "high_extra_ratio",
    "pixel_weight_mean",
    "pixel_weight_std",
    "entropy_weight_used",
]


GKD_FIRST_BATCH_HEADERS = [
    "idx",
    "dataset",
    "stem",
    "area",
    "num_cc",
    "largest_cc_ratio",
    "edge_touch_ratio",
    "despl_fixed_iou",
    "q_score",
    "grade",
    "sample_weight",
]


GKD_BRANCH_V2_FIRST_BATCH_HEADERS = [
    "idx",
    "dataset",
    "stem",
    "area",
    "num_cc",
    "largest_cc_ratio",
    "edge_touch_ratio",
    "despl_fixed_iou",
    "teacher_area",
    "teacher_despl_iou",
    "q_score",
    "raw_grade",
    "final_grade",
    "grade_reason",
    "branch_type",
]


GKD_BRANCH_V3_FIRST_BATCH_HEADERS = [
    "idx",
    "dataset",
    "stem",
    "area",
    "num_cc",
    "largest_cc_ratio",
    "edge_touch_ratio",
    "despl_fixed_iou",
    "teacher_area",
    "teacher_despl_iou",
    "q_score",
    "raw_grade",
    "final_grade",
    "grade_reason",
    "branch_type",
]


def init_gkd_audit_accumulator(mode):
    return {
        "mode": mode,
        "batch_calls": 0,
        "sample_calls": 0,
        "num_high": 0,
        "num_normal": 0,
        "num_low": 0,
        "num_raw_high": 0,
        "num_raw_normal": 0,
        "num_raw_low": 0,
        "num_strict_high_block": 0,
        "num_high_downgrade": 0,
        "num_static_low": 0,
        "num_teacher_low": 0,
        "num_teacher_normal_cap": 0,
        "num_dynamic_low": 0,
        "num_dynamic_high_cap": 0,
        "disabled_after_epoch_count": 0,
        "plain_loss_sum": 0.0,
        "reweight_loss_sum": 0.0,
        "relative_delta_sum": 0.0,
        "q_score_sum": 0.0,
        "sample_weight_mean_sum": 0.0,
        "sample_weight_std_sum": 0.0,
        "pixel_weight_mean_sum": 0.0,
        "pixel_weight_std_sum": 0.0,
        "pixel_weight_p10_sum": 0.0,
        "pixel_weight_p50_sum": 0.0,
        "pixel_weight_p90_sum": 0.0,
        "pixel_weight_lt_0_7_ratio_sum": 0.0,
        "pixel_weight_gt_0_95_ratio_sum": 0.0,
        "area_sum": 0.0,
        "num_cc_sum": 0.0,
        "largest_cc_ratio_sum": 0.0,
        "edge_touch_ratio_sum": 0.0,
        "despl_fixed_iou_sum": 0.0,
        "despl_fixed_iou_valid_batches": 0,
        "teacher_despl_iou_mean_sum": 0.0,
        "teacher_despl_iou_p10_sum": 0.0,
        "teacher_despl_iou_p50_sum": 0.0,
        "teacher_despl_iou_p90_sum": 0.0,
        "teacher_despl_iou_valid_batches": 0,
        "teacher_area_mean_sum": 0.0,
        "teacher_area_valid_batches": 0,
        "teacher_bce_sum": 0.0,
        "high_extra_ratio_sum": 0.0,
        "entropy_weight_used_sum": 0.0,
        "last_strength": 1.0,
    }


def update_gkd_audit_accumulator(acc, info):
    batch_size = int(info.get("num_samples", 0))
    acc["batch_calls"] += 1
    acc["sample_calls"] += batch_size
    acc["num_high"] += int(info.get("num_high", 0))
    acc["num_normal"] += int(info.get("num_normal", 0))
    acc["num_low"] += int(info.get("num_low", 0))
    acc["num_raw_high"] += int(info.get("num_raw_high", 0))
    acc["num_raw_normal"] += int(info.get("num_raw_normal", 0))
    acc["num_raw_low"] += int(info.get("num_raw_low", 0))
    acc["num_strict_high_block"] += int(info.get("num_strict_high_block", 0))
    acc["num_high_downgrade"] += int(info.get("num_high_downgrade", 0))
    acc["num_static_low"] += int(info.get("num_static_low", 0))
    acc["num_teacher_low"] += int(info.get("num_teacher_low", 0))
    acc["num_teacher_normal_cap"] += int(info.get("num_teacher_normal_cap", 0))
    acc["num_dynamic_low"] += int(info.get("num_dynamic_low", 0))
    acc["num_dynamic_high_cap"] += int(info.get("num_dynamic_high_cap", 0))
    acc["disabled_after_epoch_count"] += int(bool(info.get("disabled_after_epoch", False)))
    acc["plain_loss_sum"] += float(info.get("plain_loss", info.get("plain_bce", 0.0)))
    acc["reweight_loss_sum"] += float(info.get("reweight_loss", 0.0))
    acc["relative_delta_sum"] += float(info.get("relative_delta", 0.0))
    acc["q_score_sum"] += float(info.get("q_score_mean", 0.0)) * batch_size
    acc["sample_weight_mean_sum"] += float(info.get("sample_weight_mean", 0.0))
    acc["sample_weight_std_sum"] += float(info.get("sample_weight_std", 0.0))
    acc["pixel_weight_mean_sum"] += float(info.get("pixel_weight_mean", 0.0))
    acc["pixel_weight_std_sum"] += float(info.get("pixel_weight_std", 0.0))
    acc["pixel_weight_p10_sum"] += float(info.get("pixel_weight_p10", 0.0))
    acc["pixel_weight_p50_sum"] += float(info.get("pixel_weight_p50", 0.0))
    acc["pixel_weight_p90_sum"] += float(info.get("pixel_weight_p90", 0.0))
    acc["pixel_weight_lt_0_7_ratio_sum"] += float(info.get("pixel_weight_lt_0_7_ratio", 0.0))
    acc["pixel_weight_gt_0_95_ratio_sum"] += float(info.get("pixel_weight_gt_0_95_ratio", 0.0))
    acc["area_sum"] += float(info.get("area_mean", 0.0)) * batch_size
    acc["num_cc_sum"] += float(info.get("num_cc_mean", 0.0)) * batch_size
    acc["largest_cc_ratio_sum"] += float(info.get("largest_cc_ratio_mean", 0.0)) * batch_size
    acc["edge_touch_ratio_sum"] += float(info.get("edge_touch_ratio_mean", 0.0)) * batch_size
    if float(info.get("despl_fixed_iou_mean", -1.0)) >= 0:
        acc["despl_fixed_iou_sum"] += float(info.get("despl_fixed_iou_mean", 0.0))
        acc["despl_fixed_iou_valid_batches"] += 1
    if float(info.get("teacher_despl_iou_mean", -1.0)) >= 0:
        acc["teacher_despl_iou_mean_sum"] += float(info.get("teacher_despl_iou_mean", 0.0))
        acc["teacher_despl_iou_p10_sum"] += float(info.get("teacher_despl_iou_p10", 0.0))
        acc["teacher_despl_iou_p50_sum"] += float(info.get("teacher_despl_iou_p50", 0.0))
        acc["teacher_despl_iou_p90_sum"] += float(info.get("teacher_despl_iou_p90", 0.0))
        acc["teacher_despl_iou_valid_batches"] += 1
    if float(info.get("teacher_area_mean", -1.0)) >= 0:
        acc["teacher_area_mean_sum"] += float(info.get("teacher_area_mean", 0.0))
        acc["teacher_area_valid_batches"] += 1
    acc["teacher_bce_sum"] += float(info.get("teacher_bce", 0.0))
    acc["high_extra_ratio_sum"] += float(info.get("high_extra_ratio", 0.0))
    acc["entropy_weight_used_sum"] += float(bool(info.get("entropy_weight_used", False)))
    acc["last_strength"] = float(info.get("strength", info.get("branch_strength", 1.0)))


def finalize_gkd_audit_row(epoch, acc):
    batch_calls = max(int(acc["batch_calls"]), 1)
    sample_calls = max(int(acc["sample_calls"]), 1)
    high = int(acc["num_high"])
    normal = int(acc["num_normal"])
    low = int(acc["num_low"])
    iou_batches = int(acc["despl_fixed_iou_valid_batches"])
    teacher_iou_batches = int(acc["teacher_despl_iou_valid_batches"])
    teacher_area_batches = int(acc["teacher_area_valid_batches"])
    return {
        "epoch": int(epoch),
        "mode": acc["mode"],
        "strength": float(acc["last_strength"]),
        "batch_calls": int(acc["batch_calls"]),
        "sample_calls": int(acc["sample_calls"]),
        "high_count": high,
        "normal_count": normal,
        "low_count": low,
        "high_ratio": high / sample_calls,
        "normal_ratio": normal / sample_calls,
        "low_ratio": low / sample_calls,
        "plain_loss": acc["plain_loss_sum"] / batch_calls,
        "reweight_loss": acc["reweight_loss_sum"] / batch_calls,
        "relative_delta": acc["relative_delta_sum"] / batch_calls,
        "q_score_mean": acc["q_score_sum"] / sample_calls,
        "sample_weight_mean": acc["sample_weight_mean_sum"] / batch_calls,
        "sample_weight_std": acc["sample_weight_std_sum"] / batch_calls,
        "pixel_weight_mean": acc["pixel_weight_mean_sum"] / batch_calls,
        "pixel_weight_std": acc["pixel_weight_std_sum"] / batch_calls,
        "pixel_weight_p10": acc["pixel_weight_p10_sum"] / batch_calls,
        "pixel_weight_p50": acc["pixel_weight_p50_sum"] / batch_calls,
        "pixel_weight_p90": acc["pixel_weight_p90_sum"] / batch_calls,
        "pixel_weight_lt_0_7_ratio": acc["pixel_weight_lt_0_7_ratio_sum"] / batch_calls,
        "pixel_weight_gt_0_95_ratio": acc["pixel_weight_gt_0_95_ratio_sum"] / batch_calls,
        "area_mean": acc["area_sum"] / sample_calls,
        "num_cc_mean": acc["num_cc_sum"] / sample_calls,
        "largest_cc_ratio_mean": acc["largest_cc_ratio_sum"] / sample_calls,
        "edge_touch_ratio_mean": acc["edge_touch_ratio_sum"] / sample_calls,
        "despl_fixed_iou_mean": (
            acc["despl_fixed_iou_sum"] / iou_batches if iou_batches > 0 else -1.0
        ),
        "old_high_count": int(acc["num_raw_high"]),
        "old_normal_count": int(acc["num_raw_normal"]),
        "old_low_count": int(acc["num_raw_low"]),
        "old_high_ratio": int(acc["num_raw_high"]) / sample_calls,
        "old_normal_ratio": int(acc["num_raw_normal"]) / sample_calls,
        "old_low_ratio": int(acc["num_raw_low"]) / sample_calls,
        "final_high_count": high,
        "final_normal_count": normal,
        "final_low_count": low,
        "final_high_ratio": high / sample_calls,
        "final_normal_ratio": normal / sample_calls,
        "final_low_ratio": low / sample_calls,
        "strict_high_block_count": int(acc["num_strict_high_block"]),
        "hard_downgrade_count": int(acc["num_high_downgrade"]),
        "static_low_count": int(acc["num_static_low"]),
        "teacher_low_count": int(acc["num_teacher_low"]),
        "teacher_normal_cap_count": int(acc["num_teacher_normal_cap"]),
        "dynamic_low_count": int(acc["num_dynamic_low"]),
        "dynamic_high_cap_count": int(acc["num_dynamic_high_cap"]),
        "disabled_after_epoch": bool(acc["disabled_after_epoch_count"] > 0),
        "loss_used": "plain_bce" if acc["mode"] == "audit" else "reweight_bce",
        "teacher_despl_iou_mean": (
            acc["teacher_despl_iou_mean_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p10": (
            acc["teacher_despl_iou_p10_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p50": (
            acc["teacher_despl_iou_p50_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p90": (
            acc["teacher_despl_iou_p90_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_area_mean": (
            acc["teacher_area_mean_sum"] / teacher_area_batches
            if teacher_area_batches > 0
            else -1.0
        ),
        "high_extra_ratio": acc["high_extra_ratio_sum"] / batch_calls,
        "teacher_bce": acc["teacher_bce_sum"] / batch_calls,
        "plain_bce": acc["plain_loss_sum"] / batch_calls,
        "branch_loss": acc["plain_loss_sum"] / batch_calls,
        "branch_delta": 0.0,
        "low_loss": 0.0,
        "normal_loss": 0.0,
        "high_loss": 0.0,
        "high_bce": 0.0,
        "high_l1": 0.0,
        "high_mse": 0.0,
        "teacher_l1": 0.0,
        "entropy_weight_used": bool(acc["entropy_weight_used_sum"] > 0),
    }


def init_gkd_branch_accumulator():
    return {
        "batch_calls": 0,
        "sample_calls": 0,
        "num_high": 0,
        "num_normal": 0,
        "num_low": 0,
        "num_raw_high": 0,
        "num_raw_normal": 0,
        "num_raw_low": 0,
        "num_strict_high_block": 0,
        "num_high_downgrade": 0,
        "num_static_low": 0,
        "num_teacher_low": 0,
        "num_teacher_normal_cap": 0,
        "num_dynamic_low": 0,
        "num_dynamic_high_cap": 0,
        "disabled_after_epoch_count": 0,
        "plain_bce_sum": 0.0,
        "branch_loss_sum": 0.0,
        "branch_delta_sum": 0.0,
        "low_loss_sum": 0.0,
        "normal_loss_sum": 0.0,
        "high_loss_sum": 0.0,
        "high_bce_sum": 0.0,
        "high_l1_sum": 0.0,
        "high_mse_sum": 0.0,
        "teacher_l1_sum": 0.0,
        "teacher_bce_sum": 0.0,
        "high_extra_ratio_sum": 0.0,
        "pixel_weight_mean_sum": 0.0,
        "pixel_weight_std_sum": 0.0,
        "entropy_mean_sum": 0.0,
        "entropy_std_sum": 0.0,
        "entropy_high_ratio_sum": 0.0,
        "teacher_despl_iou_mean_sum": 0.0,
        "teacher_despl_iou_p10_sum": 0.0,
        "teacher_despl_iou_p50_sum": 0.0,
        "teacher_despl_iou_p90_sum": 0.0,
        "teacher_despl_iou_valid_batches": 0,
        "teacher_area_mean_sum": 0.0,
        "teacher_area_valid_batches": 0,
        "entropy_weight_used_sum": 0.0,
        "last_strength": 1.0,
    }


def update_gkd_branch_accumulator(acc, info):
    batch_size = int(info.get("num_samples", 0))
    acc["batch_calls"] += 1
    acc["sample_calls"] += batch_size
    acc["num_high"] += int(info.get("num_high", 0))
    acc["num_normal"] += int(info.get("num_normal", 0))
    acc["num_low"] += int(info.get("num_low", 0))
    acc["num_raw_high"] += int(info.get("num_raw_high", 0))
    acc["num_raw_normal"] += int(info.get("num_raw_normal", 0))
    acc["num_raw_low"] += int(info.get("num_raw_low", 0))
    acc["num_strict_high_block"] += int(info.get("num_strict_high_block", 0))
    acc["num_high_downgrade"] += int(info.get("num_high_downgrade", 0))
    acc["num_static_low"] += int(info.get("num_static_low", 0))
    acc["num_teacher_low"] += int(info.get("num_teacher_low", 0))
    acc["num_teacher_normal_cap"] += int(info.get("num_teacher_normal_cap", 0))
    acc["num_dynamic_low"] += int(info.get("num_dynamic_low", 0))
    acc["num_dynamic_high_cap"] += int(info.get("num_dynamic_high_cap", 0))
    acc["disabled_after_epoch_count"] += int(bool(info.get("disabled_after_epoch", False)))
    for key, info_key in (
        ("plain_bce_sum", "plain_bce"),
        ("branch_loss_sum", "branch_loss"),
        ("branch_delta_sum", "branch_delta"),
        ("low_loss_sum", "low_loss"),
        ("normal_loss_sum", "normal_loss"),
        ("high_loss_sum", "high_loss"),
        ("high_bce_sum", "high_bce"),
        ("high_l1_sum", "high_l1"),
        ("high_mse_sum", "high_mse"),
        ("teacher_l1_sum", "teacher_l1"),
        ("teacher_bce_sum", "teacher_bce"),
        ("high_extra_ratio_sum", "high_extra_ratio"),
        ("pixel_weight_mean_sum", "pixel_weight_mean"),
        ("pixel_weight_std_sum", "pixel_weight_std"),
        ("entropy_mean_sum", "entropy_mean"),
        ("entropy_std_sum", "entropy_std"),
        ("entropy_high_ratio_sum", "entropy_high_ratio"),
    ):
        acc[key] += float(info.get(info_key, 0.0))
    acc["entropy_weight_used_sum"] += float(bool(info.get("entropy_weight_used", False)))
    if float(info.get("teacher_despl_iou_mean", -1.0)) >= 0:
        acc["teacher_despl_iou_mean_sum"] += float(info.get("teacher_despl_iou_mean", 0.0))
        acc["teacher_despl_iou_p10_sum"] += float(info.get("teacher_despl_iou_p10", 0.0))
        acc["teacher_despl_iou_p50_sum"] += float(info.get("teacher_despl_iou_p50", 0.0))
        acc["teacher_despl_iou_p90_sum"] += float(info.get("teacher_despl_iou_p90", 0.0))
        acc["teacher_despl_iou_valid_batches"] += 1
    if float(info.get("teacher_area_mean", -1.0)) >= 0:
        acc["teacher_area_mean_sum"] += float(info.get("teacher_area_mean", 0.0))
        acc["teacher_area_valid_batches"] += 1
    acc["last_strength"] = float(info.get("branch_strength", info.get("strength", 1.0)))


def finalize_gkd_branch_row(epoch, acc):
    batch_calls = max(int(acc["batch_calls"]), 1)
    sample_calls = max(int(acc["sample_calls"]), 1)
    high = int(acc["num_high"])
    normal = int(acc["num_normal"])
    low = int(acc["num_low"])
    teacher_iou_batches = int(acc["teacher_despl_iou_valid_batches"])
    teacher_area_batches = int(acc["teacher_area_valid_batches"])
    return {
        "epoch": int(epoch),
        "mode": "branch",
        "strength": float(acc["last_strength"]),
        "batch_calls": int(acc["batch_calls"]),
        "sample_calls": int(acc["sample_calls"]),
        "high_count": high,
        "normal_count": normal,
        "low_count": low,
        "high_ratio": high / sample_calls,
        "normal_ratio": normal / sample_calls,
        "low_ratio": low / sample_calls,
        "plain_bce": acc["plain_bce_sum"] / batch_calls,
        "branch_loss": acc["branch_loss_sum"] / batch_calls,
        "branch_delta": acc["branch_delta_sum"] / batch_calls,
        "low_loss": acc["low_loss_sum"] / batch_calls,
        "normal_loss": acc["normal_loss_sum"] / batch_calls,
        "high_loss": acc["high_loss_sum"] / batch_calls,
        "high_bce": acc["high_bce_sum"] / batch_calls,
        "high_l1": acc["high_l1_sum"] / batch_calls,
        "high_mse": acc["high_mse_sum"] / batch_calls,
        "teacher_l1": acc["teacher_l1_sum"] / batch_calls,
        "teacher_bce": acc["teacher_bce_sum"] / batch_calls,
        "high_extra_ratio": acc["high_extra_ratio_sum"] / batch_calls,
        "pixel_weight_mean": acc["pixel_weight_mean_sum"] / batch_calls,
        "pixel_weight_std": acc["pixel_weight_std_sum"] / batch_calls,
        "entropy_mean": acc["entropy_mean_sum"] / batch_calls,
        "entropy_std": acc["entropy_std_sum"] / batch_calls,
        "entropy_high_ratio": acc["entropy_high_ratio_sum"] / batch_calls,
        "old_high_count": int(acc["num_raw_high"]),
        "old_normal_count": int(acc["num_raw_normal"]),
        "old_low_count": int(acc["num_raw_low"]),
        "old_high_ratio": int(acc["num_raw_high"]) / sample_calls,
        "old_normal_ratio": int(acc["num_raw_normal"]) / sample_calls,
        "old_low_ratio": int(acc["num_raw_low"]) / sample_calls,
        "final_high_count": high,
        "final_normal_count": normal,
        "final_low_count": low,
        "final_high_ratio": high / sample_calls,
        "final_normal_ratio": normal / sample_calls,
        "final_low_ratio": low / sample_calls,
        "strict_high_block_count": int(acc["num_strict_high_block"]),
        "hard_downgrade_count": int(acc["num_high_downgrade"]),
        "static_low_count": int(acc["num_static_low"]),
        "teacher_low_count": int(acc["num_teacher_low"]),
        "teacher_normal_cap_count": int(acc["num_teacher_normal_cap"]),
        "dynamic_low_count": int(acc["num_dynamic_low"]),
        "dynamic_high_cap_count": int(acc["num_dynamic_high_cap"]),
        "disabled_after_epoch": bool(acc["disabled_after_epoch_count"] > 0),
        "loss_used": "plain_bce" if acc["disabled_after_epoch_count"] > 0 else "branch",
        "teacher_despl_iou_mean": (
            acc["teacher_despl_iou_mean_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p10": (
            acc["teacher_despl_iou_p10_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p50": (
            acc["teacher_despl_iou_p50_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_despl_iou_p90": (
            acc["teacher_despl_iou_p90_sum"] / teacher_iou_batches
            if teacher_iou_batches > 0
            else -1.0
        ),
        "teacher_area_mean": (
            acc["teacher_area_mean_sum"] / teacher_area_batches
            if teacher_area_batches > 0
            else -1.0
        ),
        "entropy_weight_used": bool(acc["entropy_weight_used_sum"] > 0),
    }


def write_csv_row(path, headers, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in headers})


def write_gkd_first_batch_csv(path, batch, q_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(batch.get("dataset", []))
    stems = list(batch.get("stem", []))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GKD_FIRST_BATCH_HEADERS)
        writer.writeheader()
        for idx, item in enumerate(q_info["per_sample"][:16]):
            writer.writerow(
                {
                    "idx": idx,
                    "dataset": str(datasets[idx]) if idx < len(datasets) else "NA",
                    "stem": str(stems[idx]) if idx < len(stems) else "NA",
                    "area": float(item["area"]),
                    "num_cc": int(item["num_cc"]),
                    "largest_cc_ratio": float(item["largest_cc_ratio"]),
                    "edge_touch_ratio": float(item["edge_touch_ratio"]),
                    "despl_fixed_iou": float(item["despl_fixed_iou"]),
                    "q_score": float(item["q_score"]),
                    "grade": int(item["grade"]),
                    "sample_weight": float(item["sample_weight"]),
                }
            )


def write_gkd_branch_v2_first_batch_csv(path, batch, q_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(batch.get("dataset", []))
    stems = list(batch.get("stem", []))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GKD_BRANCH_V2_FIRST_BATCH_HEADERS)
        writer.writeheader()
        for idx, item in enumerate(q_info["per_sample"][:16]):
            writer.writerow(
                {
                    "idx": idx,
                    "dataset": str(datasets[idx]) if idx < len(datasets) else "NA",
                    "stem": str(stems[idx]) if idx < len(stems) else "NA",
                    "area": float(item["area"]),
                    "num_cc": int(item["num_cc"]),
                    "largest_cc_ratio": float(item["largest_cc_ratio"]),
                    "edge_touch_ratio": float(item["edge_touch_ratio"]),
                    "despl_fixed_iou": float(item["despl_fixed_iou"]),
                    "teacher_area": float(item.get("teacher_area", -1.0)),
                    "teacher_despl_iou": float(item.get("teacher_despl_iou", -1.0)),
                    "q_score": float(item["q_score"]),
                    "raw_grade": int(item.get("raw_grade", item["grade"])),
                    "final_grade": int(item.get("final_grade", item["grade"])),
                    "grade_reason": str(item.get("grade_reason", "none")),
                    "branch_type": str(item.get("branch_type", "")),
                }
            )


def write_gkd_branch_v3_first_batch_csv(path, batch, q_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(batch.get("dataset", []))
    stems = list(batch.get("stem", []))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GKD_BRANCH_V3_FIRST_BATCH_HEADERS)
        writer.writeheader()
        for idx, item in enumerate(q_info["per_sample"][:16]):
            writer.writerow(
                {
                    "idx": idx,
                    "dataset": str(datasets[idx]) if idx < len(datasets) else "NA",
                    "stem": str(stems[idx]) if idx < len(stems) else "NA",
                    "area": float(item["area"]),
                    "num_cc": int(item["num_cc"]),
                    "largest_cc_ratio": float(item["largest_cc_ratio"]),
                    "edge_touch_ratio": float(item["edge_touch_ratio"]),
                    "despl_fixed_iou": float(item["despl_fixed_iou"]),
                    "teacher_area": float(item.get("teacher_area", -1.0)),
                    "teacher_despl_iou": float(item.get("teacher_despl_iou", -1.0)),
                    "q_score": float(item["q_score"]),
                    "raw_grade": int(item.get("raw_grade", item["grade"])),
                    "final_grade": int(item.get("final_grade", item["grade"])),
                    "grade_reason": str(item.get("grade_reason", "none")),
                    "branch_type": str(item.get("branch_type", "")),
                }
            )


def format_gkd_audit_log(epoch, row, loss_used):
    return (
        f"[GKD-Audit] epoch={epoch:03d} | "
        f"mode={row['mode']} | "
        f"loss_used={loss_used} | "
        f"strength={row['strength']:.3f} | "
        f"batch_calls={row['batch_calls']} | "
        f"sample_calls={row['sample_calls']} | "
        f"high={row['high_count']} ({row['high_ratio']:.3f}) | "
        f"normal={row['normal_count']} ({row['normal_ratio']:.3f}) | "
        f"low={row['low_count']} ({row['low_ratio']:.3f}) | "
        f"plain_loss={row['plain_loss']:.6f} | "
        f"reweight_loss={row['reweight_loss']:.6f} | "
        f"rel_delta={row['relative_delta']:.6f} | "
        f"q_score_mean={row['q_score_mean']:.6f} | "
        f"sample_w_mean={row['sample_weight_mean']:.6f} | "
        f"sample_w_std={row['sample_weight_std']:.6f} | "
        f"pixel_w_mean={row['pixel_weight_mean']:.6f} | "
        f"pixel_w_std={row['pixel_weight_std']:.6f} | "
        f"pixel_w_p10={row['pixel_weight_p10']:.6f} | "
        f"pixel_w_p50={row['pixel_weight_p50']:.6f} | "
        f"pixel_w_p90={row['pixel_weight_p90']:.6f} | "
        f"pixel_w_lt_0.7={row['pixel_weight_lt_0_7_ratio']:.6f} | "
        f"pixel_w_gt_0.95={row['pixel_weight_gt_0_95_ratio']:.6f} | "
        f"area_mean={row['area_mean']:.6f} | "
        f"cc_mean={row['num_cc_mean']:.6f} | "
        f"lcc_ratio_mean={row['largest_cc_ratio_mean']:.6f} | "
        f"edge_touch_mean={row['edge_touch_ratio_mean']:.6f} | "
        f"despl_fixed_iou_mean={row['despl_fixed_iou_mean']:.6f}"
    )


def format_gkd_grade_v2_log(epoch, row):
    return (
        f"[GKD-GradeV2] epoch={epoch:03d} | "
        f"old_high={row['old_high_count']} ({row['old_high_ratio']:.3f}) | "
        f"old_normal={row['old_normal_count']} ({row['old_normal_ratio']:.3f}) | "
        f"old_low={row['old_low_count']} ({row['old_low_ratio']:.3f}) | "
        f"final_high={row['final_high_count']} ({row['final_high_ratio']:.3f}) | "
        f"final_normal={row['final_normal_count']} ({row['final_normal_ratio']:.3f}) | "
        f"final_low={row['final_low_count']} ({row['final_low_ratio']:.3f}) | "
        f"strict_high_block={row['strict_high_block_count']} | "
        f"hard_downgrade={row['hard_downgrade_count']} | "
        f"static_low={row['static_low_count']} | "
        f"teacher_low={row['teacher_low_count']} | "
        f"teacher_normal_cap={row['teacher_normal_cap_count']} | "
        f"dynamic_low={row['dynamic_low_count']} | "
        f"teacher_despl_iou_mean={row['teacher_despl_iou_mean']:.6f} | "
        f"teacher_area_mean={row['teacher_area_mean']:.6f}"
    )


def format_gkd_grade_v3_log(epoch, row):
    return (
        f"[GKD-GradeV3] epoch={epoch:03d} | "
        f"old_high={row['old_high_count']} ({row['old_high_ratio']:.3f}) | "
        f"old_normal={row['old_normal_count']} ({row['old_normal_ratio']:.3f}) | "
        f"old_low={row['old_low_count']} ({row['old_low_ratio']:.3f}) | "
        f"final_high={row['final_high_count']} ({row['final_high_ratio']:.3f}) | "
        f"final_normal={row['final_normal_count']} ({row['final_normal_ratio']:.3f}) | "
        f"final_low={row['final_low_count']} ({row['final_low_ratio']:.3f}) | "
        f"strict_high_block={row['strict_high_block_count']} | "
        f"hard_downgrade={row['hard_downgrade_count']} | "
        f"static_low={row['static_low_count']} | "
        f"dynamic_high_cap={row['dynamic_high_cap_count']} | "
        f"teacher_low={row['teacher_low_count']} | "
        f"dynamic_low={row['dynamic_low_count']} | "
        f"teacher_despl_iou_mean={row['teacher_despl_iou_mean']:.6f} | "
        f"teacher_area_mean={row['teacher_area_mean']:.6f}"
    )


def format_gkd_branch_log(epoch, row):
    return (
        f"[GKD-Branch] epoch={epoch:03d} | "
        f"mode=branch | "
        f"strength={row['strength']:.3f} | "
        f"batch_calls={row['batch_calls']} | "
        f"sample_calls={row['sample_calls']} | "
        f"high={row['high_count']} ({row['high_ratio']:.3f}) | "
        f"normal={row['normal_count']} ({row['normal_ratio']:.3f}) | "
        f"low={row['low_count']} ({row['low_ratio']:.3f}) | "
        f"plain_bce={row['plain_bce']:.6f} | "
        f"branch_loss={row['branch_loss']:.6f} | "
        f"branch_delta={row['branch_delta']:.6f} | "
        f"low_loss={row['low_loss']:.6f} | "
        f"normal_loss={row['normal_loss']:.6f} | "
        f"high_loss={row['high_loss']:.6f} | "
        f"high_bce={row['high_bce']:.6f} | "
        f"high_l1={row['high_l1']:.6f} | "
        f"high_mse={row['high_mse']:.6f} | "
        f"teacher_l1={row['teacher_l1']:.6f} | "
        f"teacher_bce={row['teacher_bce']:.6f} | "
        f"high_extra_ratio={row['high_extra_ratio']:.6f} | "
        f"pixel_w_mean={row['pixel_weight_mean']:.6f} | "
        f"pixel_w_std={row['pixel_weight_std']:.6f} | "
        f"entropy_mean={row['entropy_mean']:.6f} | "
        f"entropy_std={row['entropy_std']:.6f} | "
        f"entropy_high_ratio={row['entropy_high_ratio']:.6f}"
    )


def format_gkd_branch_v3_log(epoch, row):
    return (
        f"[GKD-BranchV3] epoch={epoch:03d} | "
        f"disabled_after_epoch={bool(row['disabled_after_epoch'])} | "
        f"loss_used={row['loss_used']} | "
        f"plain_bce={row['plain_bce']:.6f} | "
        f"branch_loss={row['branch_loss']:.6f} | "
        f"branch_delta={row['branch_delta']:.6f} | "
        f"low_loss={row['low_loss']:.6f} | "
        f"normal_loss={row['normal_loss']:.6f} | "
        f"high_loss={row['high_loss']:.6f} | "
        f"high_bce={row['high_bce']:.6f} | "
        f"high_l1={row['high_l1']:.6f} | "
        f"high_mse={row['high_mse']:.6f} | "
        f"teacher_l1={row['teacher_l1']:.6f} | "
        f"teacher_bce={row['teacher_bce']:.6f} | "
        f"high_extra_ratio={row['high_extra_ratio']:.6f} | "
        f"pixel_w_mean={row['pixel_weight_mean']:.6f} | "
        f"pixel_w_std={row['pixel_weight_std']:.6f} | "
        f"entropy_weight_used={bool(row['entropy_weight_used'])}"
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


def compute_late_despl_anchor_loss(cfg, student_logits, batch, device, criterion_none):
    if not bool(getattr(cfg, "USE_LATE_DESPL_ANCHOR_LOSS", False)):
        return student_logits.sum() * 0.0
    pseudo_despl = batch["pseudo_despl"].to(device, non_blocking=True).float()
    if list(pseudo_despl.shape[-2:]) != list(student_logits.shape[-2:]):
        pseudo_despl = F.interpolate(
            pseudo_despl,
            size=student_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    anchor_fg = pseudo_despl >= 0.75
    anchor_bg = pseudo_despl <= 0.10
    anchor_mask = anchor_fg | anchor_bg
    if not bool(anchor_mask.any().item()):
        return student_logits.sum() * 0.0
    anchor_label = anchor_fg.float()
    loss_map = criterion_none(student_logits, anchor_label)
    den = anchor_mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    loss_sample = (loss_map * anchor_mask.float()).flatten(1).sum(dim=1) / den
    return float(getattr(cfg, "LATE_DESPL_ANCHOR_LAMBDA", 0.02)) * loss_sample.mean()


def get_anchor_pbce_lambda(cfg, epoch):
    if epoch <= int(getattr(cfg, "ANCHOR_PBCE_EARLY_END_EPOCH", 20)):
        return float(getattr(cfg, "ANCHOR_PBCE_LAMBDA_EARLY", 0.0))
    late_start = int(getattr(cfg, "ANCHOR_PBCE_LATE_START_EPOCH", 10**9))
    late_end = int(getattr(cfg, "ANCHOR_PBCE_LATE_END_EPOCH", -1))
    if late_start <= epoch <= late_end:
        return float(getattr(cfg, "ANCHOR_PBCE_LAMBDA_LATE", 0.0))
    return 0.0


def prepare_despl_anchor_source(batch, logits, device):
    if "pseudo_despl" in batch:
        anchor_source = batch["pseudo_despl"]
    elif "pseudo" in batch:
        anchor_source = batch["pseudo"]
    else:
        raise RuntimeError("DESPL Anchor-PBCE requires batch['pseudo_despl'] or batch['pseudo'].")
    anchor_source = anchor_source.to(device, non_blocking=True).float()
    if anchor_source.ndim != 4 or anchor_source.shape[1] != 1:
        raise RuntimeError(
            "DESPL Anchor-PBCE source must be [B,1,H,W], "
            f"got {list(anchor_source.shape)}"
        )
    anchor_source = anchor_source.clamp(0.0, 1.0)
    if list(anchor_source.shape[-2:]) != list(logits.shape[-2:]):
        anchor_source = F.interpolate(
            anchor_source,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
    return anchor_source


def compute_despl_anchor_pbce(logits, p_despl, theta_fg, theta_bg, min_valid_pixels=16):
    if float(theta_fg) <= float(theta_bg):
        raise RuntimeError(f"ANCHOR_PBCE_THETA_FG must be > THETA_BG, got {theta_fg} <= {theta_bg}")
    if list(logits.shape) != list(p_despl.shape):
        raise RuntimeError(
            "DESPL Anchor-PBCE logits/source shape mismatch: "
            f"{list(logits.shape)} != {list(p_despl.shape)}"
        )

    fg = p_despl >= float(theta_fg)
    bg = p_despl <= float(theta_bg)
    valid = fg | bg
    label = fg.float()
    loss_map = F.binary_cross_entropy_with_logits(logits, label, reduction="none")

    batch_size = int(logits.shape[0])
    loss_flat = loss_map.flatten(1)
    valid_flat = valid.flatten(1).float()
    valid_count = valid_flat.sum(dim=1)
    has_valid = valid_count >= int(min_valid_pixels)
    sample_loss = (loss_flat * valid_flat).sum(dim=1) / valid_count.clamp_min(1.0)
    if bool(has_valid.any().item()):
        loss_anchor = sample_loss[has_valid].mean()
    else:
        loss_anchor = logits.sum() * 0.0

    numel = float(valid_flat.shape[1])
    fg_flat = fg.flatten(1).float()
    bg_flat = bg.flatten(1).float()
    stats = {
        "valid_ratio_mean": float((valid_count.mean() / numel).detach().cpu().item()),
        "fg_ratio_mean": float((fg_flat.sum(dim=1).mean() / numel).detach().cpu().item()),
        "bg_ratio_mean": float((bg_flat.sum(dim=1).mean() / numel).detach().cpu().item()),
        "skipped_samples": int((~has_valid).sum().detach().cpu().item()),
        "num_samples": batch_size,
    }
    return loss_anchor, stats


def head_gamma_value(model):
    gamma = getattr(model, "gamma", None)
    if gamma is None:
        return None
    return float(gamma.detach().cpu().item())


def drepp_restore_core(tensor, core_fg, core_bg):
    tensor = torch.where(core_fg, torch.ones_like(tensor), tensor)
    tensor = torch.where(core_bg, torch.zeros_like(tensor), tensor)
    return tensor.clamp(0.0, 1.0)


def drepp_entropy(prob):
    eps = 1e-6
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / torch.log(
        torch.tensor(2.0, device=prob.device)
    )


def drepp_quality(prob, mask):
    active = mask.bool()
    if not bool(active.any().item()):
        return 0.0
    reliability = 1.0 - drepp_entropy(prob)
    return float(reliability[active].mean().item())


def drepp_binary_iou(first, second, mask):
    active = mask.bool()
    if not bool(active.any().item()):
        return 1.0
    first = first.bool() & active
    second = second.bool() & active
    union = (first | second).float().sum().item()
    if union <= 0:
        return 1.0
    return float((first & second).float().sum().item() / union)


def drepp_batch_keys(batch):
    return [(str(dataset), str(stem)) for dataset, stem in zip(batch["dataset"], batch["stem"])]


def drepp_memory_batch(memory_bank, batch, device):
    tensors = []
    for index, key in enumerate(drepp_batch_keys(batch)):
        if key not in memory_bank:
            memory_bank[key] = batch["drepp_memory_init"][index].detach().cpu().float().clone()
        tensors.append(memory_bank[key])
    return torch.stack(tensors, dim=0).to(device, non_blocking=True).float()


def drepp_apply_fixed_local(memory, fixed_local, core_fg, core_bg, cfg):
    value = float(getattr(cfg, "DREPP_FIXED_RECALL_VALUE", 0.65))
    recall = torch.full_like(memory, value)
    memory = torch.where(fixed_local.bool(), torch.maximum(memory, recall), memory)
    return drepp_restore_core(memory, core_fg, core_bg)


@torch.no_grad()
def drepp_update_memory(cfg, epoch, memory_bank, batch, teacher_prob, current_memory, device):
    if epoch <= 5:
        return current_memory, 0, 0.0, 0.0, 0.0
    core_fg = batch["drepp_core_fg"].to(device, non_blocking=True).bool()
    core_bg = batch["drepp_core_bg"].to(device, non_blocking=True).bool()
    uncertain = batch["drepp_uncertain"].to(device, non_blocking=True).bool()
    alpha = float(getattr(cfg, "DREPP_MEMORY_ALPHA_LATE", 0.35)) if epoch >= 21 else float(
        getattr(cfg, "DREPP_MEMORY_ALPHA", 0.20)
    )
    margin = float(getattr(cfg, "DREPP_MEMORY_MARGIN", 0.03))
    iou_th = float(getattr(cfg, "DREPP_MEMORY_IOU_TH", 0.30))
    conf_th = float(getattr(cfg, "DREPP_MEMORY_CONF_TH", 0.45))
    updated_memory = current_memory.clone()
    accepted = 0
    quality_teacher_sum = 0.0
    quality_memory_sum = 0.0
    iou_sum = 0.0
    keys = drepp_batch_keys(batch)
    for index, key in enumerate(keys):
        mask = uncertain[index]
        teacher_i = teacher_prob[index]
        memory_i = current_memory[index]
        q_teacher = drepp_quality(teacher_i, mask)
        q_memory = drepp_quality(memory_i, mask)
        iou = drepp_binary_iou(
            teacher_i > float(cfg.THRESHOLD),
            memory_i > float(cfg.THRESHOLD),
            mask,
        )
        quality_teacher_sum += q_teacher
        quality_memory_sum += q_memory
        iou_sum += iou
        if q_teacher >= q_memory - margin and iou >= iou_th and q_teacher >= conf_th:
            proposed = memory_i + alpha * (teacher_i - memory_i)
            proposed = torch.where(mask, proposed, memory_i)
            proposed = drepp_restore_core(proposed, core_fg[index], core_bg[index])
            updated_memory[index] = proposed
            memory_bank[key] = proposed.detach().cpu().float()
            accepted += 1
    num = max(len(keys), 1)
    return (
        updated_memory,
        accepted,
        quality_teacher_sum / num,
        quality_memory_sum / num,
        iou_sum / num,
    )


def compute_drepp_local_loss(cfg, epoch, student_logits, teacher_prob, batch, device, criterion_none):
    if not bool(getattr(cfg, "USE_LOCAL_REFINE", False)):
        return student_logits.sum() * 0.0, 0.0
    if epoch < int(getattr(cfg, "DREPP_LOCAL_START_EPOCH", 6)):
        return student_logits.sum() * 0.0, 0.0
    lambda_local = float(getattr(cfg, "DREPP_LAMBDA_LOCAL", 0.0))
    if lambda_local <= 0.0:
        return student_logits.sum() * 0.0, 0.0
    band = batch["drepp_boundary_band"].to(device, non_blocking=True).bool()
    uncertain = batch["drepp_uncertain"].to(device, non_blocking=True).bool()
    sim = batch["drepp_feature_sim"].to(device, non_blocking=True).float()
    fg = (teacher_prob >= float(getattr(cfg, "DREPP_LOCAL_FG_TH", 0.70))) & (
        sim >= float(getattr(cfg, "DREPP_LOCAL_SIM_TH", 0.35))
    )
    bg = (teacher_prob <= float(getattr(cfg, "DREPP_LOCAL_BG_TH", 0.30))) & (
        sim <= float(getattr(cfg, "DREPP_LOCAL_BG_SIM_TH", 0.25))
    )
    mask = band & uncertain & (fg | bg)
    if not bool(mask.any().item()):
        return student_logits.sum() * 0.0, 0.0
    label = fg.float()
    loss_map = criterion_none(student_logits, label)
    den = mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    loss_sample = (loss_map * mask.float()).flatten(1).sum(dim=1) / den
    return lambda_local * loss_sample.mean(), float(mask.float().mean().item())


def compute_drepp_anchor_loss(cfg, student_logits, batch, device, criterion_none):
    lambda_anchor = float(getattr(cfg, "DREPP_LAMBDA_ANCHOR", 0.0))
    if lambda_anchor <= 0.0:
        return student_logits.sum() * 0.0
    core_fg = batch["drepp_core_fg"].to(device, non_blocking=True).bool()
    core_bg = batch["drepp_core_bg"].to(device, non_blocking=True).bool()
    mask = core_fg | core_bg
    if not bool(mask.any().item()):
        return student_logits.sum() * 0.0
    label = core_fg.float()
    loss_map = criterion_none(student_logits, label)
    den = mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    loss_sample = (loss_map * mask.float()).flatten(1).sum(dim=1) / den
    return lambda_anchor * loss_sample.mean()


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
    if bool(getattr(cfg, "USE_DREPP", False)) and (
        bool(getattr(cfg, "USE_QRA", False))
        or bool(getattr(cfg, "USE_CCR", False))
        or bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
    ):
        raise RuntimeError("USE_DREPP=True cannot be combined with USE_QRA, USE_CCR, or USE_DESPL_PSEUDO.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and (
        bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False))
    ):
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
    if bool(getattr(cfg, "USE_QRA", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_QRA=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_CCR", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_CCR=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DREPP", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DREPP=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with --pseudo_cache_override.")
    cfg.PSEUDO_CACHE_OVERRIDE = args.pseudo_cache_override
    max_epoch = int(args.max_epochs) if args.max_epochs is not None else int(cfg.MAX_EPOCH)
    reset_epoch = get_reset_epoch(cfg)
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
        gkd_mode = get_gkd_mode(cfg)
        logger.log(f"train_start_time = {current_time_text()}")
        logger.log(f"EXP_NAME = {cfg.EXP_NAME}")
        logger.log(f"device = {device}")
        logger.log("optimizer = AdamW")
        logger.log(f"lr = {cfg.DINO['lr']}")
        logger.log(f"lr0 = {get_base_lr(cfg):.8f}")
        if use_linear_floor_two_stage_lr(cfg):
            logger.log("scheduler = manual linear_floor_two_stage (StepLR object retained for checkpoint)")
            logger.log("scheduler_step = manual_iter")
        else:
            logger.log("scheduler = StepLR(step_size=25, gamma=0.95)")
            logger.log("scheduler_step = iter")
        logger.log(f"lr_policy = {getattr(cfg, 'LR_POLICY', 'step')}")
        logger.log(f"use_lr_floor = {bool(getattr(cfg, 'USE_LR_FLOOR', False))}")
        logger.log(f"lr_floor = {float(getattr(cfg, 'LR_FLOOR', 0.0)):.8f}")
        logger.log(f"lr_linear_stage1_epochs = {int(getattr(cfg, 'LR_LINEAR_STAGE1_EPOCHS', reset_epoch - 1))}")
        logger.log(f"lr_linear_stage2_epochs = {int(getattr(cfg, 'LR_LINEAR_STAGE2_EPOCHS', 0))}")
        logger.log(f"lr_floor_mode = {getattr(cfg, 'LR_FLOOR_MODE', 'global')}")
        logger.log(
            f"lr_floor_apply_after_scheduler_step = "
            f"{bool(getattr(cfg, 'LR_FLOOR_APPLY_AFTER_SCHEDULER_STEP', True))}"
        )
        logger.log(
            f"lr_floor_apply_after_finetune_reset = "
            f"{bool(getattr(cfg, 'LR_FLOOR_APPLY_AFTER_FINETUNE_RESET', True))}"
        )
        logger.log(f"max_epoch = {max_epoch}")
        logger.log(f"max_samples = {sample_limit}")
        logger.log(f"ema_weight = {cfg.EMA_WEIGHT}")
        logger.log(f"finetune_reset_epoch = {reset_epoch}")
        logger.log("finetune_reset_rebuild_optimizer = true")
        logger.log("finetune_reset_rebuild_scheduler = true")
        logger.log("finetune_reset_global_step = true")
        logger.log("finetune_reset_teacher = false")
        logger.log(f"teacher_fusion_mode = {getattr(cfg, 'TEACHER_FUSION_MODE', 'default')}")
        logger.log(f"fusion_orig_decay_epochs = {int(getattr(cfg, 'FUSION_ORIG_DECAY_EPOCHS', 20))}")
        logger.log(f"fusion_hold_fixed_weight = {float(getattr(cfg, 'FUSION_HOLD_FIXED_WEIGHT', 0.05)):.6f}")
        logger.log(
            f"teacher_fusion_pre_reset_epochs = "
            f"{int(getattr(cfg, 'TEACHER_FUSION_PRE_RESET_EPOCHS', reset_epoch - 1))}"
        )
        logger.log(
            f"fusion_min_fixed_weight = "
            f"{float(getattr(cfg, 'FUSION_MIN_FIXED_WEIGHT', 1.0 - float(getattr(cfg, 'TEACHER_FUSION_MAX_WEIGHT', 0.95)))):.6f}"
        )
        logger.log(f"teacher_fusion_max_weight = {float(getattr(cfg, 'TEACHER_FUSION_MAX_WEIGHT', 1.0)):.6f}")
        logger.log(f"complex_head_lr_policy = {getattr(cfg, 'COMPLEX_HEAD_LR_POLICY', 'none')}")
        logger.log(f"complex_head_pre_reset_lr = {float(getattr(cfg, 'COMPLEX_HEAD_PRE_RESET_LR', 0.0)):.6f}")
        logger.log(f"complex_head_lr_hold_epochs = {int(getattr(cfg, 'COMPLEX_HEAD_LR_HOLD_EPOCHS', 0))}")
        logger.log(f"complex_head_lr_min = {float(getattr(cfg, 'COMPLEX_HEAD_LR_MIN', 0.0)):.6f}")
        logger.log(f"complex_head_post_reset_lr = {complex_head_post_reset_lr(cfg):.6f}")
        logger.log(
            f"complex_head_post_reset_scheduler = "
            f"{getattr(cfg, 'COMPLEX_HEAD_POST_RESET_SCHEDULER', 'original_iter_steplr')}"
        )
        logger.log("DINO_in_training_loop = false")
        logger.log("train_gt_in_train = false")
        logger.log(f"head_type = {getattr(cfg, 'HEAD_TYPE', 'simple')}")
        if use_dagp_head(cfg):
            logger.log(f"use_dagp_head = {bool(getattr(cfg, 'USE_DAGP_HEAD', True))}")
            logger.log(f"DAGP_HIDDEN = {int(getattr(cfg, 'DAGP_HIDDEN', 64))}")
            logger.log(f"DAGP_TOPK = {int(getattr(cfg, 'DAGP_TOPK', 24))}")
            logger.log(f"DAGP_TAU = {float(getattr(cfg, 'DAGP_TAU', 0.10)):.6f}")
            logger.log(f"DAGP_ALPHA_INIT = {float(getattr(cfg, 'DAGP_ALPHA_INIT', 0.05)):.6f}")
            logger.log(f"DAGP_GAMMA_INIT = {float(getattr(cfg, 'DAGP_GAMMA_INIT', 0.10)):.6f}")
            logger.log(f"DAGP_NUM_LAYERS = {int(getattr(cfg, 'DAGP_NUM_LAYERS', 1))}")
            logger.log(f"DAGP_AFFINITY_DETACH = {bool(getattr(cfg, 'DAGP_AFFINITY_DETACH', True))}")
            logger.log(f"DAGP_USE_FFN = {bool(getattr(cfg, 'DAGP_USE_FFN', False))}")
            logger.log(f"DAGP_USE_DWCONV = {bool(getattr(cfg, 'DAGP_USE_DWCONV', False))}")
        if use_dagp_safe_head(cfg):
            logger.log(f"use_dagp_safe_head = {bool(getattr(cfg, 'USE_DAGP_SAFE_HEAD', True))}")
            logger.log(f"DAGP_SAFE_HIDDEN = {int(getattr(cfg, 'DAGP_SAFE_HIDDEN', 64))}")
            logger.log(f"DAGP_SAFE_TOPK = {int(getattr(cfg, 'DAGP_SAFE_TOPK', 12))}")
            logger.log(f"DAGP_SAFE_TAU = {float(getattr(cfg, 'DAGP_SAFE_TAU', 0.07)):.6f}")
            logger.log(f"DAGP_SAFE_ALPHA_MAX = {float(getattr(cfg, 'DAGP_SAFE_ALPHA_MAX', 0.05)):.6f}")
            logger.log(f"DAGP_SAFE_GAMMA_MAX = {float(getattr(cfg, 'DAGP_SAFE_GAMMA_MAX', 0.03)):.6f}")
            logger.log(f"DAGP_SAFE_WARMUP_EPOCH = {int(getattr(cfg, 'DAGP_SAFE_WARMUP_EPOCH', 6))}")
            logger.log(f"DAGP_SAFE_RAMP_START_EPOCH = {int(getattr(cfg, 'DAGP_SAFE_RAMP_START_EPOCH', 7))}")
            logger.log(f"DAGP_SAFE_RAMP_END_EPOCH = {int(getattr(cfg, 'DAGP_SAFE_RAMP_END_EPOCH', 15))}")
            logger.log(f"DAGP_SAFE_USE_PROB_GATE = {bool(getattr(cfg, 'DAGP_SAFE_USE_PROB_GATE', True))}")
            logger.log(
                f"DAGP_SAFE_PROB_GATE_SIGMA = {float(getattr(cfg, 'DAGP_SAFE_PROB_GATE_SIGMA', 0.25)):.6f}"
            )
            logger.log(
                f"DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE = "
                f"{bool(getattr(cfg, 'DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE', False))}"
            )
            logger.log(
                f"DAGP_SAFE_UNCERTAINTY_POWER = "
                f"{float(getattr(cfg, 'DAGP_SAFE_UNCERTAINTY_POWER', 1.0)):.6f}"
            )
            logger.log(
                f"DAGP_SAFE_UNCERTAINTY_MIN = "
                f"{float(getattr(cfg, 'DAGP_SAFE_UNCERTAINTY_MIN', 0.0)):.6f}"
            )
            logger.log(
                f"DAGP_SAFE_UNCERTAINTY_MAX = "
                f"{float(getattr(cfg, 'DAGP_SAFE_UNCERTAINTY_MAX', 1.0)):.6f}"
            )
            logger.log(
                f"DAGP_SAFE_UNCERTAINTY_DETACH = "
                f"{bool(getattr(cfg, 'DAGP_SAFE_UNCERTAINTY_DETACH', True))}"
            )
            logger.log(f"USE_BASE_AUX_LOSS = {bool(getattr(cfg, 'USE_BASE_AUX_LOSS', False))}")
            logger.log(f"LAMBDA_BASE_AUX = {float(getattr(cfg, 'LAMBDA_BASE_AUX', 0.0)):.6f}")
            logger.log(
                f"LAMBDA_BASE_AUX_AFTER_RESET = "
                f"{float(getattr(cfg, 'LAMBDA_BASE_AUX_AFTER_RESET', 0.0)):.6f}"
            )
            logger.log(f"USE_LR_FLOOR = {bool(getattr(cfg, 'USE_LR_FLOOR', False))}")
            logger.log(f"LR_FLOOR = {float(getattr(cfg, 'LR_FLOOR', 0.0)):.8f}")
        if use_ndr_branch(cfg):
            logger.log(f"USE_NDR_BRANCH = {bool(getattr(cfg, 'USE_NDR_BRANCH', False))}")
            logger.log(f"NDR_INPUT_RGB = {bool(getattr(cfg, 'NDR_INPUT_RGB', True))}")
            logger.log(f"NDR_INPUT_SOBEL = {bool(getattr(cfg, 'NDR_INPUT_SOBEL', True))}")
            logger.log(f"NDR_INPUT_COARSE_PROB = {bool(getattr(cfg, 'NDR_INPUT_COARSE_PROB', True))}")
            logger.log(f"NDR_IN_CHANNELS = {int(getattr(cfg, 'NDR_IN_CHANNELS', 5))}")
            logger.log(f"NDR_HIDDEN = {int(getattr(cfg, 'NDR_HIDDEN', 32))}")
            logger.log(f"NDR_NUM_LAYERS = {int(getattr(cfg, 'NDR_NUM_LAYERS', 3))}")
            logger.log(f"NDR_USE_GN = {bool(getattr(cfg, 'NDR_USE_GN', True))}")
            logger.log(f"NDR_GN_GROUPS = {int(getattr(cfg, 'NDR_GN_GROUPS', 4))}")
            logger.log(f"NDR_ACT = {getattr(cfg, 'NDR_ACT', 'gelu')}")
            logger.log(f"NDR_BETA_MAX = {float(getattr(cfg, 'NDR_BETA_MAX', 0.10)):.6f}")
            logger.log(f"NDR_WARMUP_EPOCH = {int(getattr(cfg, 'NDR_WARMUP_EPOCH', 6))}")
            logger.log(f"NDR_RAMP_START_EPOCH = {int(getattr(cfg, 'NDR_RAMP_START_EPOCH', 7))}")
            logger.log(f"NDR_RAMP_END_EPOCH = {int(getattr(cfg, 'NDR_RAMP_END_EPOCH', 15))}")
            logger.log(f"NDR_RESIDUAL_CLIP = {float(getattr(cfg, 'NDR_RESIDUAL_CLIP', 2.0)):.6f}")
            logger.log(f"NDR_USE_UNCERTAINTY_GATE = {bool(getattr(cfg, 'NDR_USE_UNCERTAINTY_GATE', True))}")
            logger.log(f"NDR_USE_EDGE_GATE = {bool(getattr(cfg, 'NDR_USE_EDGE_GATE', True))}")
            logger.log(f"NDR_GATE_MODE = {getattr(cfg, 'NDR_GATE_MODE', 'uncertainty_edge_boost')}")
            logger.log(f"USE_NDR_COARSE_AUX = {bool(getattr(cfg, 'USE_NDR_COARSE_AUX', False))}")
            logger.log(f"LAMBDA_NDR_COARSE_AUX = {float(getattr(cfg, 'LAMBDA_NDR_COARSE_AUX', 0.0)):.6f}")
            logger.log(f"NDR_USE_RES_REG = {bool(getattr(cfg, 'NDR_USE_RES_REG', False))}")
            logger.log(f"NDR_RES_REG_WEIGHT = {float(getattr(cfg, 'NDR_RES_REG_WEIGHT', 0.0)):.6f}")
        logger.log(f"use_multi_level_feature = {use_multi_level_feature(cfg)}")
        logger.log(f"multi_level_layers = {list(getattr(cfg, 'MULTI_LEVEL_LAYERS', []))}")
        logger.log(f"multi_level_feature_dtype = {getattr(cfg, 'MULTI_LEVEL_FEATURE_DTYPE', 'float32')}")
        logger.log(f"ml_feature_preflight_mode = {getattr(cfg, 'ML_FEATURE_PREFLIGHT_MODE', 'sample')}")
        logger.log(f"ml_feature_preflight_samples = {int(getattr(cfg, 'ML_FEATURE_PREFLIGHT_SAMPLES', 32))}")
        logger.log(f"mlc_hidden = {int(getattr(cfg, 'MLC_HIDDEN', 0))}")
        logger.log(f"mlc_use_semantic_gate = {bool(getattr(cfg, 'MLC_USE_SEMANTIC_GATE', False))}")
        logger.log(f"mlc_use_sce_lite = {bool(getattr(cfg, 'MLC_USE_SCE_LITE', False))}")
        logger.log(f"mlc_use_pcf = {bool(getattr(cfg, 'MLC_USE_PCF', False))}")
        logger.log(f"mlc_res_scale_init = {float(getattr(cfg, 'MLC_RES_SCALE_INIT', 0.0)):.6f}")
        if str(getattr(cfg, "HEAD_TYPE", "simple")) == "sap_rcim":
            logger.log(f"sap_rcim_mode = {getattr(cfg, 'SAP_RCIM_MODE', 'full')}")
            logger.log(f"sap_rcim_channel = {int(getattr(cfg, 'SAP_RCIM_CHANNEL', 64))}")
            logger.log(f"sap_rcim_width = {int(getattr(cfg, 'SAP_RCIM_WIDTH', 32))}")
            logger.log(f"sap_rcim_use_caff = {bool(getattr(cfg, 'SAP_RCIM_USE_CAFF', True))}")
            logger.log(f"sap_rcim_use_fusion = {bool(getattr(cfg, 'SAP_RCIM_USE_FUSION', True))}")
            logger.log(f"sap_rcim_use_receptive_conv = {bool(getattr(cfg, 'SAP_RCIM_USE_RECEPTIVE_CONV', True))}")
            logger.log(f"sap_rcim_use_gap_guide = {bool(getattr(cfg, 'SAP_RCIM_USE_GAP_GUIDE', True))}")
            logger.log(f"sap_rcim_out_size = {int(getattr(cfg, 'SAP_RCIM_OUT_SIZE', cfg.LOSS_SIZE))}")
        logger.log(f"use_base_aux_loss = {bool(getattr(cfg, 'USE_BASE_AUX_LOSS', False))}")
        logger.log(f"lambda_base_aux = {float(getattr(cfg, 'LAMBDA_BASE_AUX', 0.0)):.6f}")
        logger.log(
            f"lambda_base_aux_after_reset = {float(getattr(cfg, 'LAMBDA_BASE_AUX_AFTER_RESET', 0.0)):.6f}"
        )
        logger.log(f"base_aux_normalize = {bool(getattr(cfg, 'BASE_AUX_NORMALIZE', False))}")
        logger.log(f"num_workers = {int(cfg.NUM_WORKERS)}")
        logger.log(f"dataloader_persistent_workers = {bool(getattr(cfg, 'DATALOADER_PERSISTENT_WORKERS', False))}")
        logger.log(f"dataloader_prefetch_factor = {int(getattr(cfg, 'DATALOADER_PREFETCH_FACTOR', 2))}")
        logger.log(f"USE_DREPP={bool(getattr(cfg, 'USE_DREPP', False))}")
        logger.log(f"use_despl_pseudo = {bool(getattr(cfg, 'USE_DESPL_PSEUDO', False))}")
        logger.log(f"use_despl_paper_cache = {bool(getattr(cfg, 'USE_DESPL_PAPER_CACHE', False))}")
        logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
        logger.log(f"use_dre_safe_prior = {bool(getattr(cfg, 'USE_DRE_SAFE_PRIOR', False))}")
        logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', 'original_fixed')}")
        logger.log(f"use_gcm = {bool(getattr(cfg, 'PSEUDO_USE_GCM', False))}")
        logger.log(f"use_despl_anchor_pbce = {bool(getattr(cfg, 'USE_DESPL_ANCHOR_PBCE', False))}")
        logger.log(f"use_pure_despl_supervision = {bool(getattr(cfg, 'USE_PURE_DESPL_SUPERVISION', False))}")
        logger.log(f"USE_FAST_TEACHER_FUSION = {bool(getattr(cfg, 'USE_FAST_TEACHER_FUSION', False))}")
        logger.log(f"TEACHER_FUSION_MODE = {getattr(cfg, 'TEACHER_FUSION_MODE', 'default')}")
        logger.log(f"GKD_MODE = {gkd_mode}")
        logger.log(f"USE_GKD_LITE = {bool(getattr(cfg, 'USE_GKD_LITE', False))}")
        logger.log(f"GKD_AUDIT_WRITE_CSV = {bool(getattr(cfg, 'GKD_AUDIT_WRITE_CSV', True))}")
        logger.log(f"GKD_AUDIT_LOG_LOSS_DELTA = {bool(getattr(cfg, 'GKD_AUDIT_LOG_LOSS_DELTA', True))}")
        logger.log(f"GKD_AUDIT_LOG_PIXEL_HIST = {bool(getattr(cfg, 'GKD_AUDIT_LOG_PIXEL_HIST', True))}")
        logger.log(f"GKD_SAMPLE_W_HIGH = {float(getattr(cfg, 'GKD_SAMPLE_W_HIGH', 1.15)):.2f}")
        logger.log(f"GKD_SAMPLE_W_NORMAL = {float(getattr(cfg, 'GKD_SAMPLE_W_NORMAL', 1.00)):.2f}")
        logger.log(f"GKD_SAMPLE_W_LOW = {float(getattr(cfg, 'GKD_SAMPLE_W_LOW', 0.60)):.2f}")
        logger.log(f"GKD_PIXEL_W_MIN = {float(getattr(cfg, 'GKD_PIXEL_W_MIN', 0.60)):.2f}")
        logger.log(f"GKD_LATE_STRENGTH = {float(getattr(cfg, 'GKD_LATE_STRENGTH', 0.30)):.2f}")
        logger.log(f"GKD_BRANCH_L1_WEIGHT = {float(getattr(cfg, 'GKD_BRANCH_L1_WEIGHT', 1.0)):.2f}")
        logger.log(f"GKD_BRANCH_MSE_WEIGHT = {float(getattr(cfg, 'GKD_BRANCH_MSE_WEIGHT', 1.0)):.2f}")
        logger.log(f"GKD_BRANCH_LATE_STRENGTH = {float(getattr(cfg, 'GKD_BRANCH_LATE_STRENGTH', 0.30)):.2f}")
        logger.log(f"GKD_BRANCH_LOW_MODE = {getattr(cfg, 'GKD_BRANCH_LOW_MODE', 'teacher_l1')}")
        logger.log(f"GKD_LOW_TEACHER_BCE_WEIGHT = {float(getattr(cfg, 'GKD_LOW_TEACHER_BCE_WEIGHT', 0.3)):.2f}")
        logger.log(f"GKD_USE_ENTROPY_PIXEL_WEIGHT = {bool(getattr(cfg, 'GKD_USE_ENTROPY_PIXEL_WEIGHT', True))}")
        logger.log(f"GKD_USE_PSEUDO_BOX_BG = {bool(getattr(cfg, 'GKD_USE_PSEUDO_BOX_BG', False))}")
        logger.log(f"GKD_ENABLE_STRICT_HIGH_GATE = {bool(getattr(cfg, 'GKD_ENABLE_STRICT_HIGH_GATE', False))}")
        logger.log(f"GKD_ENABLE_HARD_DOWNGRADE = {bool(getattr(cfg, 'GKD_ENABLE_HARD_DOWNGRADE', False))}")
        logger.log(f"GKD_ENABLE_DYNAMIC_LOW = {bool(getattr(cfg, 'GKD_ENABLE_DYNAMIC_LOW', False))}")
        logger.log(f"GKD_ENABLE_DYNAMIC_HIGH_CAP = {bool(getattr(cfg, 'GKD_ENABLE_DYNAMIC_HIGH_CAP', False))}")
        logger.log(f"GKD_V3_LOW_ONLY_STATIC = {bool(getattr(cfg, 'GKD_V3_LOW_ONLY_STATIC', True))}")
        logger.log(f"GKD_V2_Q_HIGH = {float(getattr(cfg, 'GKD_V2_Q_HIGH', getattr(cfg, 'GKD_Q_HIGH', 0.75))):.2f}")
        logger.log(f"GKD_V2_Q_NORMAL = {float(getattr(cfg, 'GKD_V2_Q_NORMAL', getattr(cfg, 'GKD_Q_NORMAL', 0.45))):.2f}")
        logger.log(f"GKD_V3_Q_HIGH = {float(getattr(cfg, 'GKD_V3_Q_HIGH', getattr(cfg, 'GKD_V2_Q_HIGH', getattr(cfg, 'GKD_Q_HIGH', 0.75)))):.2f}")
        logger.log(f"GKD_V3_Q_NORMAL = {float(getattr(cfg, 'GKD_V3_Q_NORMAL', getattr(cfg, 'GKD_V2_Q_NORMAL', getattr(cfg, 'GKD_Q_NORMAL', 0.45)))):.2f}")
        logger.log(f"GKD_DISABLE_AFTER_EPOCH = {int(getattr(cfg, 'GKD_DISABLE_AFTER_EPOCH', -1))}")
        logger.log(f"GKD_LOW_TEACHER_L1_WEIGHT = {float(getattr(cfg, 'GKD_LOW_TEACHER_L1_WEIGHT', 0.3)):.2f}")
        logger.log(f"loss size = {int(cfg.LOSS_SIZE)}x{int(cfg.LOSS_SIZE)}")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "despl_only":
            logger.log("pseudo final candidate = p_despl")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")

        use_qra = bool(getattr(cfg, "USE_QRA", False))
        use_ccr = bool(getattr(cfg, "USE_CCR", False))
        use_drepp = bool(getattr(cfg, "USE_DREPP", False))
        use_despl = bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        use_despl_paper = use_despl and bool(getattr(cfg, "USE_DESPL_PAPER_CACHE", False))
        use_dre_safe = use_despl and bool(getattr(cfg, "USE_DRE_SAFE_PRIOR", False))
        use_despl_light = use_despl and bool(getattr(cfg, "USE_DESPL_LIGHT_CACHE", False))
        use_anchor_pbce = use_despl and bool(getattr(cfg, "USE_DESPL_ANCHOR_PBCE", False))
        use_pure_despl = use_despl and bool(getattr(cfg, "USE_PURE_DESPL_SUPERVISION", False))
        use_fast_teacher_fusion = bool(getattr(cfg, "USE_FAST_TEACHER_FUSION", False))
        use_ml_feature = use_multi_level_feature(cfg)
        use_gkd_lite = use_despl and is_gkd_enabled(cfg)
        use_gkd_v3 = use_gkd_lite and is_gkd_v3_enabled(cfg)
        teacher_fusion_mode = str(getattr(cfg, "TEACHER_FUSION_MODE", "default")).lower()
        complex_post_reset_scheduler = str(
            getattr(cfg, "COMPLEX_HEAD_POST_RESET_SCHEDULER", "original_iter_steplr")
        ).lower()
        if use_complex_head_lr_policy(cfg) and complex_post_reset_scheduler != "original_iter_steplr":
            raise RuntimeError(
                "COMPLEX_HEAD_POST_RESET_SCHEDULER currently supports only "
                f"'original_iter_steplr', got {complex_post_reset_scheduler}."
            )
        if use_anchor_pbce:
            if str(getattr(cfg, "ANCHOR_PBCE_SOURCE", "p_despl")) != "p_despl":
                raise RuntimeError("DESPL Anchor-PBCE currently supports ANCHOR_PBCE_SOURCE='p_despl' only.")
            theta_fg = float(getattr(cfg, "ANCHOR_PBCE_THETA_FG", 0.70))
            theta_bg = float(getattr(cfg, "ANCHOR_PBCE_THETA_BG", 0.30))
            if theta_fg <= theta_bg:
                raise RuntimeError(f"ANCHOR_PBCE_THETA_FG must be > THETA_BG, got {theta_fg} <= {theta_bg}")
        if use_pure_despl:
            if str(getattr(cfg, "PURE_DESPL_SUPERVISION_SOURCE", "p_despl")) != "p_despl":
                raise RuntimeError("Pure DESPL supervision currently supports source='p_despl' only.")
            if str(getattr(cfg, "P_INIT_MODE", "")) != "despl_only":
                raise RuntimeError("USE_PURE_DESPL_SUPERVISION=True requires P_INIT_MODE='despl_only'.")
            if use_qra or use_ccr or use_drepp:
                raise RuntimeError("USE_PURE_DESPL_SUPERVISION=True cannot be combined with QRA, CCR, or DRE++.")
        if use_fast_teacher_fusion and teacher_fusion_mode != "fast_t10":
            raise RuntimeError("USE_FAST_TEACHER_FUSION=True currently supports TEACHER_FUSION_MODE='fast_t10' only.")
        if gkd_mode != "off":
            if not use_despl:
                raise RuntimeError("GKD_MODE requires USE_DESPL_PSEUDO=True.")
            if str(getattr(cfg, "P_INIT_MODE", "")) != "despl_only":
                raise RuntimeError("GKD_MODE requires P_INIT_MODE='despl_only'.")
            if str(getattr(cfg, "HEAD_TYPE", "simple")) != "simple":
                raise RuntimeError("GKD_MODE requires HEAD_TYPE='simple'.")
            if abs(float(getattr(cfg, "P_INIT_DESPL_WEIGHT", 1.0)) - 1.0) > 1e-8:
                raise RuntimeError("GKD_MODE requires P_INIT_DESPL_WEIGHT=1.0.")
            if abs(float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.0))) > 1e-8:
                raise RuntimeError("GKD_MODE requires P_INIT_FIXED_WEIGHT=0.0.")
            if use_qra or use_ccr or use_drepp or use_dre_safe or use_despl_paper:
                raise RuntimeError("GKD_MODE cannot be combined with QRA, CCR, DRE++, DRE-SAFE, or DESPL-paper.")
            if use_anchor_pbce or use_pure_despl or use_fast_teacher_fusion:
                raise RuntimeError("GKD_MODE cannot be combined with Anchor-PBCE, Pure DESPL, or Fast Teacher Fusion.")
            if bool(getattr(cfg, "USE_LATE_DESPL_ANCHOR_LOSS", False)):
                raise RuntimeError("GKD_MODE cannot be combined with late DESPL anchor loss.")

        # 正式训练循环不加载 DINO，也不读训练集 GT。DESPL 实验只检查现有 cache，不自动生成。
        if use_ml_feature:
            for split in ("train", "val"):
                if split == "val" and args.debug_loader_only:
                    continue
                _, ml_reason = check_ml_feature_cache(
                    cfg,
                    split,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] feature_ml:{split} ready | {ml_reason}")
        elif use_despl or use_drepp:
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
        elif use_drepp:
            complete, reason = cache_status(cfg, "pseudo")
            if not complete:
                raise RuntimeError(f"fixed pseudo cache missing or incomplete: {reason}")
            logger.log(f"[Cache] pseudo ready | {reason}")
            _, drepp_reason = check_drepp_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] DRE++ ready | {drepp_reason}")
        elif use_despl:
            if use_despl_paper:
                _, despl_reason = check_despl_paper_cache(
                    cfg,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] DESPL-paper cache ready | {despl_reason}")
            elif use_despl_light:
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
        # 初始 teacher 与 student 对齐；finetune reset 不显式重置 teacher。
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

        criterion = torch.nn.BCEWithLogitsLoss()
        criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer, scheduler = build_optimizer_scheduler(cfg, student)

        best_metric = float("inf")
        best_epoch = 0
        global_step = 0
        drepp_memory_bank = {}
        gkd_first_batch_logged = False
        dagp_first_batch_logged = False
        dagp_safe_first_batch_logged = False
        ndr_first_batch_logged = False
        lr_floor_activated_logged = False
        gkd_first_batch_path = train_dir / "gkd_first_batch.csv"
        gkd_audit_csv_path = train_dir / "gkd_audit_epoch.csv"
        gkd_branch_csv_path = train_dir / "gkd_branch_epoch.csv"
        gkd_branch_v2_csv_path = train_dir / "gkd_branch_v2_epoch.csv"
        gkd_branch_v2_first_batch_path = train_dir / "gkd_branch_v2_first_batch.csv"
        gkd_branch_v3_csv_path = train_dir / "gkd_branch_v3_epoch.csv"
        gkd_branch_v3_first_batch_path = train_dir / "gkd_branch_v3_first_batch.csv"

        for epoch in range(1, max_epoch + 1):
            # 对齐 UCOD-DPL：teacher-only 阶段首轮第一个 batch 前重置优化器状态和 EMA 步数。
            if epoch == reset_epoch:
                optimizer, scheduler = build_optimizer_scheduler(cfg, student, lr=complex_head_post_reset_lr(cfg))
                global_step = 0
                if bool(getattr(cfg, "LR_FLOOR_APPLY_AFTER_FINETUNE_RESET", True)):
                    lr_floor_clamped, scheduler_lr, clamped_lr = apply_lr_floor(optimizer, cfg)
                    if lr_floor_clamped and not lr_floor_activated_logged:
                        logger.log(
                            "[LR Floor] activated | "
                            f"global_step={global_step} | "
                            f"scheduler_lr={scheduler_lr:.8f} | "
                            f"clamped_lr={clamped_lr:.8f}"
                        )
                        lr_floor_activated_logged = True
                logger.log(
                    f"[Finetune Reset] epoch={epoch:03d} | "
                    "rebuild_optimizer=True | "
                    "rebuild_scheduler=True | "
                    "reset_global_step=True | "
                    "reset_teacher=False | "
                    f"lr={current_lr(optimizer):.8f}"
                )
            apply_complex_head_lr_policy(optimizer, epoch, cfg)

            set_model_epoch(student, epoch)
            set_model_epoch(teacher, epoch)
            student.train()
            teacher.eval()
            fixed_weight, teacher_weight, fusion_mode = get_fixed_teacher_weights(cfg, epoch)
            effective_despl_weight = 1.0 if use_pure_despl else fixed_weight
            effective_teacher_weight = 0.0 if use_pure_despl else teacher_weight
            target_mode = "pure_despl" if use_pure_despl else fusion_mode
            teacher_binary_used = False if use_pure_despl else bool(effective_teacher_weight > 0.0)
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
            dre_safe_p_base_area_sum = 0.0
            dre_safe_p_safe_area_sum = 0.0
            dre_safe_candidate_ratio_sum = 0.0
            dre_safe_fallback_sum = 0.0
            dre_safe_cc_base_sum = 0.0
            dre_safe_cc_safe_sum = 0.0
            dre_safe_delta_sum = 0.0
            dre_safe_changed_ratio_sum = 0.0
            drepp_num_samples = 0
            drepp_p_despl_area_sum = 0.0
            drepp_p_fixed_area_sum = 0.0
            drepp_core_fg_area_sum = 0.0
            drepp_core_bg_area_sum = 0.0
            drepp_uncertain_area_sum = 0.0
            drepp_fixed_local_area_sum = 0.0
            drepp_boundary_band_area_sum = 0.0
            drepp_fixed_local_ratio_sum = 0.0
            drepp_memory_accept_count = 0
            drepp_teacher_quality_sum = 0.0
            drepp_memory_quality_sum = 0.0
            drepp_iou_sum = 0.0
            drepp_local_ratio_sum = 0.0
            drepp_beta_epoch = 0.0
            total_local_loss = 0.0
            total_aux_base_loss = 0.0
            mlc_context_scale_sum = 0.0
            mlc_base_abs_sum = 0.0
            mlc_context_abs_sum = 0.0
            mlc_final_abs_sum = 0.0
            mlc_stat_batches = 0
            sap_base_abs_sum = 0.0
            sap_logits_abs_sum = 0.0
            sap_final_abs_sum = 0.0
            sap_debug_sums = {name: 0.0 for name in SAP_DEBUG_KEYS}
            sap_stat_batches = 0
            student_prob_mean_sum = 0.0
            teacher_prob_mean_sum = 0.0
            student_pred_area_sum = 0.0
            teacher_pred_area_sum = 0.0
            mixed_target_area_sum = 0.0
            dagp_safe_scale_sum = 0.0
            dagp_safe_alpha_sum = 0.0
            dagp_safe_gamma_sum = 0.0
            dagp_safe_unc_gate_mean_sum = 0.0
            dagp_safe_unc_gate_min = None
            dagp_safe_unc_gate_max = None
            dagp_safe_stat_batches = 0
            ndr_beta_sum = 0.0
            ndr_gate_mean_sum = 0.0
            ndr_gate_min = None
            ndr_gate_max = None
            ndr_residual_abs_mean_sum = 0.0
            ndr_residual_abs_max = None
            ndr_stat_batches = 0
            total_ndr_coarse_aux_loss = 0.0
            total_ndr_res_reg_loss = 0.0
            anchor_pbce_lambda_epoch = get_anchor_pbce_lambda(cfg, epoch) if use_anchor_pbce else 0.0
            anchor_pbce_loss_sum = 0.0
            anchor_pbce_valid_ratio_sum = 0.0
            anchor_pbce_fg_ratio_sum = 0.0
            anchor_pbce_bg_ratio_sum = 0.0
            anchor_pbce_skipped_samples = 0
            anchor_pbce_batches = 0
            gkd_audit_epoch = (
                init_gkd_audit_accumulator(gkd_mode)
                if gkd_mode in {"audit", "reweight"}
                else None
            )
            gkd_branch_epoch = init_gkd_branch_accumulator() if gkd_mode == "branch" else None

            for iter_idx, batch in enumerate(train_loader):
                pseudo = batch["pseudo"].to(device, non_blocking=True).float()
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                pseudo_68 = F.interpolate(pseudo, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear").float()

                student_out = forward_seg_head(
                    student,
                    model_input,
                    cfg,
                    image_68=image_68,
                    return_aux=use_dagp_safe_head(cfg),
                )
                raw_student_logits = extract_logits(student_out)
                student_logits = resize_logits_for_loss(raw_student_logits, cfg)
                if (
                    use_dagp_head(cfg)
                    and bool(getattr(cfg, "DAGP_DEBUG_FIRST_BATCH", True))
                    and not dagp_first_batch_logged
                ):
                    log_dagp_first_batch(logger, student, model_input, raw_student_logits, student_logits, pseudo_68)
                    dagp_first_batch_logged = True
                if (
                    use_dagp_safe_head(cfg)
                    and bool(getattr(cfg, "DAGP_SAFE_DEBUG_FIRST_BATCH", True))
                    and not dagp_safe_first_batch_logged
                ):
                    dagp_safe_raw_logits_for_log = (
                        student_out.get("coarse_logits_37", raw_student_logits)
                        if isinstance(student_out, dict)
                        else raw_student_logits
                    )
                    log_dagp_safe_first_batch(
                        logger,
                        student,
                        model_input,
                        dagp_safe_raw_logits_for_log,
                        student_logits,
                        pseudo_68,
                        student_out,
                    )
                    dagp_safe_first_batch_logged = True
                if (
                    use_ndr_branch(cfg)
                    and bool(getattr(cfg, "NDR_DEBUG_FIRST_BATCH", True))
                    and not ndr_first_batch_logged
                    and isinstance(student_out, dict)
                ):
                    log_ndr_first_batch(logger, image_68, student_out, pseudo_68)
                    ndr_first_batch_logged = True
                with torch.no_grad():
                    teacher_out = forward_seg_head(
                        teacher,
                        model_input,
                        cfg,
                        image_68=image_68,
                        return_aux=False,
                    )
                    teacher_logits = resize_logits_for_loss(extract_logits(teacher_out), cfg)
                    teacher_prob = teacher_logits.sigmoid()
                    teacher_binary = (teacher_prob > float(cfg.THRESHOLD)).float()

                fixed_target = pseudo_68
                if use_ccr:
                    fixed_target = batch["ccr_p_corr"].to(device, non_blocking=True).float()
                elif use_qra and epoch < reset_epoch:
                    qra_quality = batch["qra_quality"].to(device, non_blocking=True).long()
                    qra_fused = batch["qra_p_fused"].to(device, non_blocking=True).float()
                    blend = qra_fixed_blend(cfg, qra_quality, device).view(-1, 1, 1, 1)
                    fixed_target = (1.0 - blend) * pseudo_68 + blend * qra_fused

                late_override_ratio = 0.0
                if use_drepp:
                    core_fg = batch["drepp_core_fg"].to(device, non_blocking=True).bool()
                    core_bg = batch["drepp_core_bg"].to(device, non_blocking=True).bool()
                    uncertain = batch["drepp_uncertain"].to(device, non_blocking=True).bool()
                    current_memory = drepp_memory_batch(drepp_memory_bank, batch, device)
                    (
                        current_memory,
                        accepted,
                        q_teacher,
                        q_memory,
                        stable_iou,
                    ) = drepp_update_memory(
                        cfg,
                        epoch,
                        drepp_memory_bank,
                        batch,
                        teacher_prob,
                        current_memory,
                        device,
                    )
                    if epoch <= 5:
                        mixed_target = pseudo_68
                        drepp_beta = 0.0
                    else:
                        fixed_local = batch["drepp_fixed_local_recall"].to(device, non_blocking=True).bool()
                        memory_target = drepp_apply_fixed_local(current_memory, fixed_local, core_fg, core_bg, cfg)
                        if epoch < reset_epoch:
                            drepp_beta = min(
                                teacher_weight,
                                float(getattr(cfg, "DREPP_TEACHER_BETA_MAX", 0.35)),
                            )
                        else:
                            drepp_beta = float(getattr(cfg, "DREPP_TEACHER_BETA_LATE", 0.65))
                        uncertain_target = (1.0 - drepp_beta) * memory_target + drepp_beta * teacher_prob
                        mixed_target = torch.where(uncertain, uncertain_target, memory_target)
                    mixed_target = drepp_restore_core(mixed_target, core_fg, core_bg)
                    batch_size_drepp = int(batch["drepp_p_despl_area"].numel())
                    drepp_num_samples += batch_size_drepp
                    drepp_memory_accept_count += int(accepted)
                    drepp_teacher_quality_sum += float(q_teacher) * batch_size_drepp
                    drepp_memory_quality_sum += float(q_memory) * batch_size_drepp
                    drepp_iou_sum += float(stable_iou) * batch_size_drepp
                    drepp_beta_epoch = drepp_beta
                elif use_pure_despl:
                    mixed_target = fixed_target.float()
                elif epoch < reset_epoch:
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
                with torch.no_grad():
                    student_prob_for_area = student_logits.detach().sigmoid()
                    student_prob_mean_sum += float(student_prob_for_area.mean().item())
                    teacher_prob_mean_sum += float(teacher_prob.mean().item())
                    student_pred_area_sum += float(
                        (student_prob_for_area > float(cfg.THRESHOLD)).float().mean().item()
                    )
                    teacher_pred_area_sum += float(teacher_binary.mean().item())
                    mixed_target_area_sum += float(mixed_target.detach().mean().item())
                if gkd_mode != "off":
                    p_despl_for_quality = batch.get("pseudo_despl", fixed_target)
                    p_fixed_for_quality = batch.get("pseudo_fixed", None)
                    p_despl_for_quality = p_despl_for_quality.to(device, non_blocking=True).float()
                    if p_despl_for_quality.shape[-2:] != student_logits.shape[-2:]:
                        p_despl_for_quality = F.interpolate(
                            p_despl_for_quality,
                            size=student_logits.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    p_despl_for_quality = p_despl_for_quality.clamp(0.0, 1.0)

                    if p_fixed_for_quality is not None:
                        p_fixed_for_quality = p_fixed_for_quality.to(device, non_blocking=True).float()
                        if p_fixed_for_quality.ndim != 4 or p_fixed_for_quality.shape[1] != 1:
                            p_fixed_for_quality = None
                        else:
                            if p_fixed_for_quality.shape[-2:] != student_logits.shape[-2:]:
                                p_fixed_for_quality = F.interpolate(
                                    p_fixed_for_quality,
                                    size=student_logits.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            p_fixed_for_quality = p_fixed_for_quality.clamp(0.0, 1.0)

                    if gkd_mode in {"audit", "reweight"}:
                        gkd_info = compute_plain_and_reweight_loss_for_audit(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits,
                            mixed_target=mixed_target,
                            p_despl=p_despl_for_quality,
                            p_fixed=p_fixed_for_quality,
                            epoch=epoch,
                            cfg=cfg,
                        )
                        update_gkd_audit_accumulator(gkd_audit_epoch, gkd_info)
                        if gkd_mode == "audit":
                            loss_base = criterion(student_logits, mixed_target)
                        else:
                            loss_base = gkd_info["reweight_loss_tensor"]
                        if not bool(torch.isfinite(loss_base).item()):
                            raise RuntimeError(f"GKD {gkd_mode} loss is not finite.")
                    elif gkd_mode == "branch":
                        loss_base, gkd_info = compute_gkd_branch_loss(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits,
                            mixed_target=mixed_target,
                            p_despl=p_despl_for_quality,
                            p_fixed=p_fixed_for_quality,
                            epoch=epoch,
                            cfg=cfg,
                        )
                        update_gkd_branch_accumulator(gkd_branch_epoch, gkd_info)
                        if not bool(torch.isfinite(loss_base).item()):
                            raise RuntimeError("GKD branch loss is not finite.")
                    else:
                        raise RuntimeError(f"Unknown GKD_MODE: {gkd_mode}")

                    if (
                        epoch == 1
                        and not gkd_first_batch_logged
                        and bool(getattr(cfg, "GKD_AUDIT_LOG_FIRST_BATCH", True))
                    ):
                        first_prefix = (
                            "[GKD-Branch FirstBatch]"
                            if gkd_mode == "branch"
                            else "[GKD-Audit FirstBatch]"
                        )
                        log_gkd_first_batch(logger, batch, gkd_info, prefix=first_prefix)
                        if bool(getattr(cfg, "GKD_AUDIT_WRITE_CSV", True)):
                            write_gkd_first_batch_csv(gkd_first_batch_path, batch, gkd_info)
                            if use_gkd_v3:
                                write_gkd_branch_v3_first_batch_csv(
                                    gkd_branch_v3_first_batch_path,
                                    batch,
                                    gkd_info,
                                )
                            else:
                                write_gkd_branch_v2_first_batch_csv(
                                    gkd_branch_v2_first_batch_path,
                                    batch,
                                    gkd_info,
                                )
                        gkd_first_batch_logged = True
                else:
                    loss_base = criterion(student_logits, mixed_target)
                loss_aux_base = student_logits.sum() * 0.0
                loss_ndr_coarse_aux = student_logits.sum() * 0.0
                loss_ndr_res_reg = student_logits.sum() * 0.0
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
                elif use_ccr and epoch < reset_epoch and bool(getattr(cfg, "CCR_USE_ANCHOR_LOSS", False)):
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
                if use_ndr_branch(cfg):
                    if not isinstance(student_out, dict) or "coarse_logits_68" not in student_out:
                        raise RuntimeError("USE_NDR_BRANCH=True requires coarse_logits_68 in student output.")
                    ndr_terms = [loss_base]
                    ndr_weights = [1.0]
                    if bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
                        coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                        loss_ndr_coarse_aux = criterion(coarse_logits, mixed_target)
                        ndr_terms.append(loss_ndr_coarse_aux)
                        ndr_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                    if (
                        bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                        and "base_logits" in student_out
                    ):
                        reset_epoch = int(getattr(cfg, "FINETUNE_RESET_EPOCH", 21))
                        aux_lambda = (
                            float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                            if epoch < reset_epoch
                            else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                        )
                        base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                        loss_aux_base = criterion(base_logits, mixed_target)
                        ndr_terms.append(loss_aux_base)
                        ndr_weights.append(aux_lambda)
                    weight_sum = max(1e-12, sum(ndr_weights))
                    loss = sum(w * term for w, term in zip(ndr_weights, ndr_terms)) / weight_sum
                    if bool(getattr(cfg, "NDR_USE_RES_REG", False)):
                        detail_gate = student_out["detail_gate"].detach()
                        residual_logits = student_out["residual_logits_68"]
                        loss_ndr_res_reg = torch.mean(torch.abs(detail_gate * residual_logits))
                        loss = loss + float(getattr(cfg, "NDR_RES_REG_WEIGHT", 0.001)) * loss_ndr_res_reg
                elif (
                    bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                    and isinstance(student_out, dict)
                    and "base_logits" in student_out
                ):
                    reset_epoch = int(getattr(cfg, "FINETUNE_RESET_EPOCH", 21))
                    aux_lambda = (
                        float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                        if epoch < reset_epoch
                        else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                    )
                    base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                    loss_aux_base = criterion(base_logits, mixed_target)
                    if bool(getattr(cfg, "BASE_AUX_NORMALIZE", False)):
                        loss = (loss + aux_lambda * loss_aux_base) / (1.0 + aux_lambda)
                    else:
                        loss = loss + aux_lambda * loss_aux_base
                if (
                    use_despl
                    and bool(getattr(cfg, "USE_LATE_DESPL_ANCHOR_LOSS", False))
                    and epoch >= reset_epoch
                ):
                    loss_anchor = loss_anchor + compute_late_despl_anchor_loss(
                        cfg,
                        student_logits,
                        batch,
                        device,
                        criterion_none,
                    )
                    loss = loss + loss_anchor
                if use_anchor_pbce:
                    anchor_source = prepare_despl_anchor_source(batch, student_logits, device)
                    anchor_pbce_loss, anchor_pbce_stats = compute_despl_anchor_pbce(
                        student_logits,
                        anchor_source,
                        float(getattr(cfg, "ANCHOR_PBCE_THETA_FG", 0.70)),
                        float(getattr(cfg, "ANCHOR_PBCE_THETA_BG", 0.30)),
                        int(getattr(cfg, "ANCHOR_PBCE_MIN_VALID_PIXELS", 16)),
                    )
                    if not bool(torch.isfinite(anchor_pbce_loss).item()):
                        raise RuntimeError("DESPL Anchor-PBCE loss is not finite.")
                    weighted_anchor_pbce = anchor_pbce_lambda_epoch * anchor_pbce_loss
                    loss_anchor = loss_anchor + weighted_anchor_pbce
                    loss = loss + weighted_anchor_pbce
                    anchor_pbce_loss_sum += float(anchor_pbce_loss.item())
                    anchor_pbce_valid_ratio_sum += float(anchor_pbce_stats["valid_ratio_mean"])
                    anchor_pbce_fg_ratio_sum += float(anchor_pbce_stats["fg_ratio_mean"])
                    anchor_pbce_bg_ratio_sum += float(anchor_pbce_stats["bg_ratio_mean"])
                    anchor_pbce_skipped_samples += int(anchor_pbce_stats["skipped_samples"])
                    anchor_pbce_batches += 1
                loss_local = student_logits.sum() * 0.0
                if use_drepp:
                    loss_local, local_ratio = compute_drepp_local_loss(
                        cfg,
                        epoch,
                        student_logits,
                        teacher_prob,
                        batch,
                        device,
                        criterion_none,
                    )
                    loss_anchor = loss_anchor + compute_drepp_anchor_loss(
                        cfg,
                        student_logits,
                        batch,
                        device,
                        criterion_none,
                    )
                    loss = loss + loss_local + loss_anchor
                    drepp_local_ratio_sum += float(local_ratio)

                if use_linear_floor_two_stage_lr(cfg):
                    set_optimizer_lr(
                        optimizer,
                        compute_linear_floor_two_stage_lr(
                            epoch,
                            iter_idx,
                            len(train_loader),
                            cfg,
                        ),
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if should_step_iter_scheduler(epoch, cfg):
                    scheduler.step()
                    if bool(getattr(cfg, "LR_FLOOR_APPLY_AFTER_SCHEDULER_STEP", True)):
                        lr_floor_clamped, scheduler_lr, clamped_lr = apply_lr_floor(optimizer, cfg)
                        if lr_floor_clamped and not lr_floor_activated_logged:
                            logger.log(
                                "[LR Floor] activated | "
                                f"global_step={global_step} | "
                                f"scheduler_lr={scheduler_lr:.8f} | "
                                f"clamped_lr={clamped_lr:.8f}"
                            )
                            lr_floor_activated_logged = True
                # teacher pseudo 已在本轮 EMA 更新前生成，避免当前 student 更新泄漏进 target。
                update_ema(student, teacher, global_step, ema_weight=float(cfg.EMA_WEIGHT))
                global_step += 1

                total_loss += float(loss.item())
                total_base_loss += float(loss_base.item())
                total_anchor_loss += float(loss_anchor.item())
                total_soft_loss += float(loss_soft.item())
                total_local_loss += float(loss_local.item())
                total_aux_base_loss += float(loss_aux_base.item())
                total_ndr_coarse_aux_loss += float(loss_ndr_coarse_aux.item())
                total_ndr_res_reg_loss += float(loss_ndr_res_reg.item())
                if isinstance(student_out, dict):
                    scale_value = output_context_scale(student_out)
                    if scale_value is not None:
                        mlc_context_scale_sum += scale_value
                        mlc_base_abs_sum += output_tensor_abs_mean(student_out, "base_logits")
                        mlc_context_abs_sum += output_tensor_abs_mean(student_out, "context_logits")
                        mlc_final_abs_sum += output_tensor_abs_mean(student_out, "logits")
                        mlc_stat_batches += 1
                    if str(getattr(cfg, "HEAD_TYPE", "simple")) == "sap_rcim":
                        sap_base_abs_sum += output_tensor_abs_mean(student_out, "base_logits")
                        sap_logits_abs_sum += output_tensor_abs_mean(student_out, "sap_logits")
                        sap_final_abs_sum += output_tensor_abs_mean(student_out, "logits")
                        for name in SAP_DEBUG_KEYS:
                            sap_debug_sums[name] += output_debug_value(student_out, name)
                        sap_stat_batches += 1
                    if use_dagp_safe_head(cfg):
                        dagp_safe_scale_sum += output_scalar(student_out, "dagp_scale")
                        dagp_safe_alpha_sum += output_scalar(student_out, "dagp_alpha_eff")
                        dagp_safe_gamma_sum += output_scalar(student_out, "dagp_gamma_eff")
                        unc_gate_mean = output_scalar(student_out, "uncertainty_gate_mean", -1.0)
                        unc_gate_min = output_scalar(student_out, "uncertainty_gate_min", -1.0)
                        unc_gate_max = output_scalar(student_out, "uncertainty_gate_max", -1.0)
                        dagp_safe_unc_gate_mean_sum += unc_gate_mean
                        dagp_safe_unc_gate_min = (
                            unc_gate_min
                            if dagp_safe_unc_gate_min is None
                            else min(dagp_safe_unc_gate_min, unc_gate_min)
                        )
                        dagp_safe_unc_gate_max = (
                            unc_gate_max
                            if dagp_safe_unc_gate_max is None
                            else max(dagp_safe_unc_gate_max, unc_gate_max)
                        )
                        dagp_safe_stat_batches += 1
                    if use_ndr_branch(cfg):
                        ndr_beta_sum += output_scalar(student_out, "ndr_beta_eff")
                        ndr_gate_mean = output_scalar(student_out, "ndr_detail_gate_mean")
                        ndr_gate_min_value = output_scalar(student_out, "ndr_detail_gate_min")
                        ndr_gate_max_value = output_scalar(student_out, "ndr_detail_gate_max")
                        ndr_residual_abs_mean_sum += output_scalar(student_out, "ndr_residual_abs_mean")
                        ndr_residual_abs_max_value = output_scalar(student_out, "ndr_residual_abs_max")
                        ndr_gate_mean_sum += ndr_gate_mean
                        ndr_gate_min = (
                            ndr_gate_min_value
                            if ndr_gate_min is None
                            else min(ndr_gate_min, ndr_gate_min_value)
                        )
                        ndr_gate_max = (
                            ndr_gate_max_value
                            if ndr_gate_max is None
                            else max(ndr_gate_max, ndr_gate_max_value)
                        )
                        ndr_residual_abs_max = (
                            ndr_residual_abs_max_value
                            if ndr_residual_abs_max is None
                            else max(ndr_residual_abs_max, ndr_residual_abs_max_value)
                        )
                        ndr_stat_batches += 1
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
                    if use_dre_safe:
                        dre_safe_p_base_area_sum += float(batch["dre_safe_area_base"].sum().item())
                        dre_safe_p_safe_area_sum += float(batch["dre_safe_area_safe"].sum().item())
                        dre_safe_candidate_ratio_sum += float(batch["dre_safe_candidate_ratio"].sum().item())
                        dre_safe_fallback_sum += float(batch["dre_safe_fallback"].float().sum().item())
                        dre_safe_cc_base_sum += float(batch["dre_safe_cc_base"].float().sum().item())
                        dre_safe_cc_safe_sum += float(batch["dre_safe_cc_safe"].float().sum().item())
                        dre_safe_delta_sum += float(batch["dre_safe_positive_delta_mean"].sum().item())
                        dre_safe_changed_ratio_sum += float(batch["dre_safe_changed_ratio"].sum().item())
                if use_drepp:
                    drepp_p_despl_area_sum += float(batch["drepp_p_despl_area"].sum().item())
                    drepp_p_fixed_area_sum += float(batch["drepp_p_fixed_area"].sum().item())
                    drepp_core_fg_area_sum += float(batch["drepp_core_fg_area"].sum().item())
                    drepp_core_bg_area_sum += float(batch["drepp_core_bg_area"].sum().item())
                    drepp_uncertain_area_sum += float(batch["drepp_uncertain_area"].sum().item())
                    drepp_fixed_local_area_sum += float(batch["drepp_fixed_local_area"].sum().item())
                    drepp_boundary_band_area_sum += float(batch["drepp_boundary_band_area"].sum().item())
                    drepp_fixed_local_ratio_sum += float(batch["drepp_fixed_local_ratio"].sum().item())

            avg_loss = total_loss / max(num_batches, 1)
            logger.log(
                f"[Train] Epoch {epoch:03d}/{max_epoch:03d} | "
                f"avg_train_loss={avg_loss:.6f} | lr={current_lr(optimizer):.8f} | "
                f"fixed_weight={effective_despl_weight:.2f} | "
                f"teacher_weight={effective_teacher_weight:.2f} | "
                f"schedule_fixed_weight={fixed_weight:.2f} | "
                f"schedule_teacher_weight={teacher_weight:.2f}"
            )
            stat_batches = max(num_batches, 1)
            logger.log(
                f"[PredArea] epoch={epoch:03d} | "
                f"student_prob_mean={student_prob_mean_sum / stat_batches:.6f} | "
                f"teacher_prob_mean={teacher_prob_mean_sum / stat_batches:.6f} | "
                f"student_pred_area_mean={student_pred_area_sum / stat_batches:.6f} | "
                f"teacher_pred_area_mean={teacher_pred_area_sum / stat_batches:.6f} | "
                f"mixed_target_area_mean={mixed_target_area_sum / stat_batches:.6f}"
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
                fixed_weight_in_init = float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.2))
                use_fixed_in_pseudo = (
                    str(getattr(cfg, "P_INIT_MODE", "")) not in {"despl_only", "despl_paper_only"}
                    and abs(fixed_weight_in_init) > 0.0
                )
                logger.log(
                    f"[DESPL] epoch={epoch:03d} | "
                    "use_despl_pseudo=True | "
                    f"use_despl_paper_cache={bool(getattr(cfg, 'USE_DESPL_PAPER_CACHE', False))} | "
                    f"use_despl_light_cache={use_despl_light} | "
                    f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'despl_fixed_blend')} | "
                    f"use_fixed_in_pseudo={use_fixed_in_pseudo} | "
                    f"fixed_used_for_training={use_fixed_in_pseudo} | "
                    f"use_gcm={bool(getattr(cfg, 'PSEUDO_USE_GCM', False))} | "
                    f"p_init_area_mean={despl_p_init_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"p_fixed_area_mean={despl_p_fixed_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"p_despl_area_mean={despl_p_despl_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"fixed_weight={fixed_weight:.2f} | "
                    f"teacher_weight={teacher_weight:.2f}"
                )
            if gkd_mode in {"audit", "reweight"}:
                gkd_row = finalize_gkd_audit_row(epoch, gkd_audit_epoch)
                loss_used = "plain_bce" if gkd_mode == "audit" else "reweight_bce"
                logger.log(format_gkd_audit_log(epoch, gkd_row, loss_used=loss_used))
                if use_gkd_v3:
                    logger.log(format_gkd_grade_v3_log(epoch, gkd_row))
                else:
                    logger.log(format_gkd_grade_v2_log(epoch, gkd_row))
                if bool(getattr(cfg, "GKD_AUDIT_WRITE_CSV", True)):
                    write_csv_row(gkd_audit_csv_path, GKD_AUDIT_HEADERS, gkd_row)
                    if use_gkd_v3:
                        write_csv_row(gkd_branch_v3_csv_path, GKD_BRANCH_V3_HEADERS, gkd_row)
                    else:
                        write_csv_row(gkd_branch_v2_csv_path, GKD_BRANCH_V2_HEADERS, gkd_row)
            if gkd_mode == "branch":
                branch_row = finalize_gkd_branch_row(epoch, gkd_branch_epoch)
                if use_gkd_v3:
                    logger.log(format_gkd_branch_v3_log(epoch, branch_row))
                    logger.log(format_gkd_grade_v3_log(epoch, branch_row))
                else:
                    logger.log(format_gkd_branch_log(epoch, branch_row))
                    logger.log(format_gkd_grade_v2_log(epoch, branch_row))
                if epoch > 20 and float(branch_row["high_extra_ratio"]) > 1e-8:
                    logger.log(
                        "[GKD-Branch Warning] epoch>20 high_extra_ratio should be 0. "
                        f"got {float(branch_row['high_extra_ratio']):.8f}"
                    )
                if bool(getattr(cfg, "GKD_AUDIT_WRITE_CSV", True)):
                    if use_gkd_v3:
                        write_csv_row(gkd_branch_v3_csv_path, GKD_BRANCH_V3_HEADERS, branch_row)
                    else:
                        write_csv_row(gkd_branch_csv_path, GKD_BRANCH_HEADERS, branch_row)
                        write_csv_row(gkd_branch_v2_csv_path, GKD_BRANCH_V2_HEADERS, branch_row)
            if use_pure_despl:
                logger.log(
                    f"[PURE_DESPL] epoch={epoch:03d} | "
                    "use=True | "
                    f"target_mode={target_mode} | "
                    f"effective_despl_weight={effective_despl_weight:.2f} | "
                    f"effective_teacher_weight={effective_teacher_weight:.2f} | "
                    f"teacher_binary_used={teacher_binary_used} | "
                    "reset_kept=True"
                )
            if use_fast_teacher_fusion and teacher_fusion_mode == "fast_t10" and not use_pure_despl:
                logger.log(
                    f"[FAST_T10] epoch={epoch:03d} | "
                    "use=True | "
                    f"fusion_mode={fusion_mode} | "
                    f"effective_despl_weight={effective_despl_weight:.4f} | "
                    f"effective_teacher_weight={effective_teacher_weight:.4f} | "
                    f"teacher_binary_used={teacher_binary_used} | "
                    f"reset_epoch={reset_epoch}"
                )
            if use_anchor_pbce:
                logger.log(
                    f"[ANCHOR_PBCE] epoch={epoch:03d} | "
                    "use=True | "
                    f"source={getattr(cfg, 'ANCHOR_PBCE_SOURCE', 'p_despl')} | "
                    f"theta_fg={float(getattr(cfg, 'ANCHOR_PBCE_THETA_FG', 0.70)):.3f} | "
                    f"theta_bg={float(getattr(cfg, 'ANCHOR_PBCE_THETA_BG', 0.30)):.3f} | "
                    f"lambda={anchor_pbce_lambda_epoch:.4f} | "
                    f"loss_base_mean={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_anchor_mean={anchor_pbce_loss_sum / max(anchor_pbce_batches, 1):.6f} | "
                    f"valid_ratio_mean={anchor_pbce_valid_ratio_sum / max(anchor_pbce_batches, 1):.6f} | "
                    f"fg_ratio_mean={anchor_pbce_fg_ratio_sum / max(anchor_pbce_batches, 1):.6f} | "
                    f"bg_ratio_mean={anchor_pbce_bg_ratio_sum / max(anchor_pbce_batches, 1):.6f} | "
                    f"skipped_samples={anchor_pbce_skipped_samples}"
                )
            if str(getattr(cfg, "HEAD_TYPE", "simple")) == "gated_context":
                gamma = head_gamma_value(student)
                gamma_text = "NA" if gamma is None else f"{gamma:.8f}"
                gamma_abs_text = "NA" if gamma is None else f"{abs(gamma):.8f}"
                logger.log(
                    f"[GatedContext] epoch={epoch:03d} | "
                    "head_type=gated_context | "
                    f"gated_gamma={gamma_text} | "
                    f"gated_gamma_abs={gamma_abs_text}"
                )
            if str(getattr(cfg, "HEAD_TYPE", "simple")) == "ml_context":
                stat_batches = max(mlc_stat_batches, 1)
                logger.log(
                    f"[MLCFD] epoch={epoch:03d} | "
                    "head_type=ml_context | "
                    f"context_scale={mlc_context_scale_sum / stat_batches:.8f} | "
                    f"loss_main={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_aux_base={total_aux_base_loss / max(num_batches, 1):.6f} | "
                    f"base_logits_abs_mean={mlc_base_abs_sum / stat_batches:.6f} | "
                    f"context_logits_abs_mean={mlc_context_abs_sum / stat_batches:.6f} | "
                    f"final_logits_abs_mean={mlc_final_abs_sum / stat_batches:.6f}"
                )
            if str(getattr(cfg, "HEAD_TYPE", "simple")) == "sap_rcim":
                stat_batches = max(sap_stat_batches, 1)
                logger.log(
                    f"[SAP-RCIM] epoch={epoch:03d} | "
                    f"mode={getattr(cfg, 'SAP_RCIM_MODE', 'full')} | "
                    f"loss_main={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_aux_base={total_aux_base_loss / max(num_batches, 1):.6f} | "
                    f"base_logits_abs_mean={sap_base_abs_sum / stat_batches:.6f} | "
                    f"sap_logits_abs_mean={sap_logits_abs_sum / stat_batches:.6f} | "
                    f"final_logits_abs_mean={sap_final_abs_sum / stat_batches:.6f} | "
                    f"x10_abs_mean={sap_debug_sums['x10_abs_mean'] / stat_batches:.6f} | "
                    f"x11_abs_mean={sap_debug_sums['x11_abs_mean'] / stat_batches:.6f} | "
                    f"x12_abs_mean={sap_debug_sums['x12_abs_mean'] / stat_batches:.6f} | "
                    f"caff_10_11_abs_mean={sap_debug_sums['caff_10_11_abs_mean'] / stat_batches:.6f} | "
                    f"caff_11_12_abs_mean={sap_debug_sums['caff_11_12_abs_mean'] / stat_batches:.6f} | "
                    f"fusion_abs_mean={sap_debug_sums['fusion_abs_mean'] / stat_batches:.6f}"
                )
            if use_dagp_safe_head(cfg):
                stat_batches = max(dagp_safe_stat_batches, 1)
                logger.log(
                    f"[DAGP-Safe] epoch={epoch:03d} | "
                    f"dagp_scale={dagp_safe_scale_sum / stat_batches:.8f} | "
                    f"dagp_alpha_eff={dagp_safe_alpha_sum / stat_batches:.8f} | "
                    f"dagp_gamma_eff={dagp_safe_gamma_sum / stat_batches:.8f} | "
                    f"unc_gate_mean={dagp_safe_unc_gate_mean_sum / stat_batches:.8f} | "
                    f"unc_gate_min={(dagp_safe_unc_gate_min if dagp_safe_unc_gate_min is not None else -1.0):.8f} | "
                    f"unc_gate_max={(dagp_safe_unc_gate_max if dagp_safe_unc_gate_max is not None else -1.0):.8f} | "
                    f"loss_main={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_aux_base={total_aux_base_loss / max(num_batches, 1):.6f}"
                )
            if use_ndr_branch(cfg):
                stat_batches = max(ndr_stat_batches, 1)
                logger.log(
                    f"[NDR] epoch={epoch:03d} | "
                    f"ndr_beta_eff={ndr_beta_sum / stat_batches:.8f} | "
                    f"ndr_gate_mean={ndr_gate_mean_sum / stat_batches:.8f} | "
                    f"ndr_gate_min={(ndr_gate_min if ndr_gate_min is not None else -1.0):.8f} | "
                    f"ndr_gate_max={(ndr_gate_max if ndr_gate_max is not None else -1.0):.8f} | "
                    f"ndr_residual_abs_mean={ndr_residual_abs_mean_sum / stat_batches:.8f} | "
                    f"ndr_residual_abs_max={(ndr_residual_abs_max if ndr_residual_abs_max is not None else -1.0):.8f} | "
                    f"loss_final={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_coarse_aux={total_ndr_coarse_aux_loss / max(num_batches, 1):.6f} | "
                    f"loss_base_aux={total_aux_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_res_reg={total_ndr_res_reg_loss / max(num_batches, 1):.6f}"
                )
            if use_dre_safe:
                logger.log(
                    f"[DRE_SAFE] epoch={epoch:03d} | "
                    "use_dre_safe_prior=True | "
                    f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'safe_despl_residual')} | "
                    f"p_base_area_mean={dre_safe_p_base_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"p_safe_area_mean={dre_safe_p_safe_area_sum / max(despl_num_samples, 1):.6f} | "
                    f"safe_candidate_ratio_mean={dre_safe_candidate_ratio_sum / max(despl_num_samples, 1):.6f} | "
                    f"safe_fallback_ratio={dre_safe_fallback_sum / max(despl_num_samples, 1):.6f} | "
                    f"cc_base_mean={dre_safe_cc_base_sum / max(despl_num_samples, 1):.6f} | "
                    f"cc_safe_mean={dre_safe_cc_safe_sum / max(despl_num_samples, 1):.6f} | "
                    f"safe_positive_delta_mean={dre_safe_delta_sum / max(despl_num_samples, 1):.6f} | "
                    f"safe_changed_ratio_mean={dre_safe_changed_ratio_sum / max(despl_num_samples, 1):.6f} | "
                    f"fixed_weight={fixed_weight:.2f} | "
                    f"teacher_weight={teacher_weight:.2f} | "
                    f"use_late_despl_anchor_loss={bool(getattr(cfg, 'USE_LATE_DESPL_ANCHOR_LOSS', False))}"
                )
            if use_drepp:
                memory_update_ratio = drepp_memory_accept_count / max(drepp_num_samples, 1)
                logger.log(
                    f"[DREPP] epoch={epoch:03d} | "
                    "USE_DREPP=True | "
                    "DESPL-core active=true | "
                    "memory initialized=true | "
                    "teacher_uncertain_only=true | "
                    "fixed_local_only=true | "
                    f"local_refine={bool(getattr(cfg, 'USE_LOCAL_REFINE', False))} | "
                    "anchor_protected=true | "
                    "global_blend=false | "
                    f"memory_update_ratio={memory_update_ratio:.6f} | "
                    f"teacher_accept_ratio={memory_update_ratio:.6f} | "
                    f"teacher_q={drepp_teacher_quality_sum / max(drepp_num_samples, 1):.6f} | "
                    f"memory_q={drepp_memory_quality_sum / max(drepp_num_samples, 1):.6f} | "
                    f"stable_iou={drepp_iou_sum / max(drepp_num_samples, 1):.6f} | "
                    f"uncertain_ratio={drepp_uncertain_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"despl_area={drepp_p_despl_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"fixed_area={drepp_p_fixed_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"core_fg_area={drepp_core_fg_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"core_bg_area={drepp_core_bg_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"fixed_local_area={drepp_fixed_local_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"fixed_local_ratio={drepp_fixed_local_ratio_sum / max(drepp_num_samples, 1):.6f} | "
                    f"boundary_band_area={drepp_boundary_band_area_sum / max(drepp_num_samples, 1):.6f} | "
                    f"local_mask_ratio={drepp_local_ratio_sum / max(num_batches, 1):.6f} | "
                    f"beta={drepp_beta_epoch:.2f} | "
                    f"loss_base={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_local={total_local_loss / max(num_batches, 1):.6f} | "
                    f"loss_anchor={total_anchor_loss / max(num_batches, 1):.6f}"
                )

            val_results = {}
            for dataset_name in cfg.VAL_DATASETS:
                set_model_epoch(student, epoch)
                result = validate_one_dataset(
                    cfg,
                    student,
                    dataset_name,
                    device,
                    max_samples=sample_limit,
                )
                val_results[dataset_name] = result
                logger.log(
                    f"[Validation] Epoch {epoch:03d}/{max_epoch:03d} | "
                    f"Dataset: {dataset_name} | time={current_time_text()}"
                )
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

            if should_save_epoch_checkpoint(epoch, max_epoch, cfg.SAVE_INTERVAL, reset_epoch=reset_epoch):
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
