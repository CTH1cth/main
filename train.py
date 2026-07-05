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
    check_dabe_pu_cache,
    check_dabe_pseudo_cache,
    check_ccr_cache,
    check_despl_light_cache,
    check_despl_paper_cache,
    check_despl_pseudo_bank,
    check_drepp_cache,
    check_hflip_feature_cache,
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


def is_finetune_reset_enabled(cfg):
    return get_reset_epoch(cfg) > 0


def finetune_reset_timing(cfg):
    return str(getattr(cfg, "FINETUNE_RESET_TIMING", "before_epoch")).lower()


def is_after_epoch_finetune_reset(cfg):
    return finetune_reset_timing(cfg) == "after_epoch"


def is_before_finetune_reset(cfg, epoch):
    if not is_finetune_reset_enabled(cfg):
        return True
    if is_after_epoch_finetune_reset(cfg):
        return int(epoch) <= get_reset_epoch(cfg)
    return int(epoch) < get_reset_epoch(cfg)


def is_at_or_after_finetune_reset(cfg, epoch):
    if not is_finetune_reset_enabled(cfg):
        return False
    if is_after_epoch_finetune_reset(cfg):
        return int(epoch) > get_reset_epoch(cfg)
    return int(epoch) >= get_reset_epoch(cfg)


def _linear_schedule_value(epoch, start_epoch, end_epoch, start_value, end_value):
    start_epoch = int(start_epoch)
    end_epoch = int(end_epoch)
    start_value = float(start_value)
    end_value = float(end_value)
    if end_epoch <= start_epoch:
        return end_value
    progress = float(int(epoch) - start_epoch) / float(end_epoch - start_epoch)
    progress = max(0.0, min(1.0, progress))
    return start_value + (end_value - start_value) * progress


def get_dabe_pu_schedule(epoch, cfg):
    epoch = int(epoch)
    if epoch <= 6:
        return (
            float(getattr(cfg, "DABE_PU_STATIC_E1_E6", 1.0)),
            float(getattr(cfg, "DABE_PU_TEACHER_E1_E6", 0.0)),
        )
    stage2_start = int(getattr(cfg, "DABE_PU_STAGE2_START", 7))
    stage2_end = int(getattr(cfg, "DABE_PU_STAGE2_END", 20))
    after_epoch = int(getattr(cfg, "DABE_PU_AFTER_EPOCH", 21))
    if stage2_start <= epoch <= stage2_end:
        static_weight = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "DABE_PU_STATIC_STAGE2_START", 1.0)),
            float(getattr(cfg, "DABE_PU_STATIC_STAGE2_END", 0.40)),
        )
        teacher_weight = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "DABE_PU_TEACHER_STAGE2_START", 0.0)),
            float(getattr(cfg, "DABE_PU_TEACHER_STAGE2_END", 0.60)),
        )
    elif epoch < after_epoch:
        static_weight = float(getattr(cfg, "DABE_PU_STATIC_STAGE2_END", 0.40))
        teacher_weight = float(getattr(cfg, "DABE_PU_TEACHER_STAGE2_END", 0.60))
    else:
        static_weight = float(getattr(cfg, "DABE_PU_STATIC_AFTER", 0.30))
        teacher_weight = float(getattr(cfg, "DABE_PU_TEACHER_AFTER", 0.70))
    static_weight = max(0.0, min(1.0, float(static_weight)))
    teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
    return static_weight, teacher_weight


def get_dabe_pu_balanced_v2_schedule(epoch, cfg):
    epoch = int(epoch)
    stage1_end = int(getattr(cfg, "DABE_PU_V2_STAGE1_END", 3))
    if epoch <= stage1_end:
        return 1.0, 0.0
    stage2_start = int(getattr(cfg, "DABE_PU_V2_STAGE2_START", 4))
    stage2_end = int(getattr(cfg, "DABE_PU_V2_STAGE2_END", 15))
    if stage2_start <= epoch <= stage2_end:
        static_weight = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "DABE_PU_V2_STATIC_STAGE2_START", 0.85)),
            float(getattr(cfg, "DABE_PU_V2_STATIC_STAGE2_END", 0.45)),
        )
        teacher_weight = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "DABE_PU_V2_TEACHER_STAGE2_START", 0.15)),
            float(getattr(cfg, "DABE_PU_V2_TEACHER_STAGE2_END", 0.55)),
        )
    else:
        static_weight = float(getattr(cfg, "DABE_PU_V2_STATIC_STAGE3", 0.30))
        teacher_weight = float(getattr(cfg, "DABE_PU_V2_TEACHER_STAGE3", 0.70))
    static_weight = max(0.0, min(1.0, float(static_weight)))
    teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
    return static_weight, teacher_weight


def get_dabe_pu_despl_schedule(epoch, cfg):
    epoch = int(epoch)
    teacher_only_start = int(
        getattr(cfg, "DABE_PU_DESPL_TEACHER_ONLY_START", get_reset_epoch(cfg) + 1)
    )
    if epoch >= teacher_only_start:
        return 0.0, 1.0
    stage_start = int(getattr(cfg, "DABE_PU_DESPL_STAGE_START", 1))
    stage_end = int(getattr(cfg, "DABE_PU_DESPL_STAGE_END", max(1, teacher_only_start - 1)))
    static_weight = _linear_schedule_value(
        epoch,
        stage_start,
        stage_end,
        float(getattr(cfg, "DABE_PU_DESPL_STATIC_START", 1.0)),
        float(getattr(cfg, "DABE_PU_DESPL_STATIC_END", 0.05)),
    )
    teacher_weight = _linear_schedule_value(
        epoch,
        stage_start,
        stage_end,
        float(getattr(cfg, "DABE_PU_DESPL_TEACHER_START", 0.0)),
        float(getattr(cfg, "DABE_PU_DESPL_TEACHER_END", 0.95)),
    )
    static_weight = max(0.0, min(1.0, float(static_weight)))
    teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
    return static_weight, teacher_weight


def get_dabe_pu_despl_teacher_target_mode(cfg):
    mode = str(getattr(cfg, "TEACHER_TARGET_MODE", "binary")).lower()
    if bool(getattr(cfg, "USE_TEACHER_SOFT_FULL_LOSS", False)):
        mode = "soft_prob"
    if mode not in {"binary", "soft_prob"}:
        raise RuntimeError(f"Unsupported TEACHER_TARGET_MODE for dabe_pu_despl_sched: {mode}")
    return mode


def get_dabe_pu_despl_static_target_mode(cfg):
    mode = str(getattr(cfg, "DABE_PU_STATIC_TARGET_MODE", "soft")).lower()
    if bool(getattr(cfg, "USE_DABE_PU_HARD_STATIC_TARGET", False)):
        mode = "hard_from_target_soft"
    if mode in {"target_soft", "soft_target"}:
        mode = "soft"
    if mode not in {"soft", "hard_from_target_soft"}:
        raise RuntimeError(f"Unsupported DABE_PU_STATIC_TARGET_MODE for dabe_pu_despl_sched: {mode}")
    if mode == "hard_from_target_soft" and not bool(getattr(cfg, "DABE_PU_HARD_KEEP_WEIGHT_MAP", True)):
        raise RuntimeError("DABE_PU_HARD_KEEP_WEIGHT_MAP=False is not supported for hard static target.")
    return mode


def build_dabe_pu_despl_static_target(cfg, pu_target_soft, pu_weight_map):
    mode = get_dabe_pu_despl_static_target_mode(cfg)
    if mode == "hard_from_target_soft":
        threshold = float(getattr(cfg, "DABE_PU_HARD_THRESH", 0.5))
        return (pu_target_soft > threshold).float(), pu_weight_map, mode
    return pu_target_soft, pu_weight_map, mode


def get_rast_pre_reset_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_RAST", False)):
        return 0.0
    if bool(getattr(cfg, "RAST_DISABLE_AFTER_RESET", True)) and is_at_or_after_finetune_reset(cfg, epoch):
        return 0.0
    start = int(getattr(cfg, "RAST_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "RAST_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "RAST_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_rast_post_reset_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_RAST", False)):
        return 0.0
    if not bool(getattr(cfg, "RAST_POST_RESET_ENABLE", False)):
        return 0.0
    start = int(getattr(cfg, "RAST_POST_RESET_START_EPOCH", 21))
    end = int(getattr(cfg, "RAST_POST_RESET_END_EPOCH", 25))
    epoch = int(epoch)
    if epoch < start or epoch > end:
        return 0.0
    return float(getattr(cfg, "RAST_POST_RESET_SCALE", 0.30))


def get_rast_scale(cfg, epoch):
    return max(
        float(get_rast_pre_reset_scale(cfg, epoch)),
        float(get_rast_post_reset_scale(cfg, epoch)),
    )


def get_esa_asym_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_ESA_ASYM", False)):
        return 0.0
    start = int(getattr(cfg, "ESA_ASYM_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "ESA_ASYM_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "ESA_ASYM_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_dabe_oem_schedule(epoch, cfg):
    epoch = int(epoch)
    stage1_end = int(getattr(cfg, "OEM_STAGE1_END", 3))
    if epoch <= stage1_end:
        return 0.0, 0.0
    stage2_start = int(getattr(cfg, "OEM_STAGE2_START", 4))
    stage2_end = int(getattr(cfg, "OEM_STAGE2_END", 15))
    if stage2_start <= epoch <= stage2_end:
        lambda_dyn_pos = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "OEM_DYN_POS_STAGE2_START", 0.0)),
            float(getattr(cfg, "OEM_DYN_POS_STAGE2_END", 0.35)),
        )
        lambda_dyn_bg = _linear_schedule_value(
            epoch,
            stage2_start,
            stage2_end,
            float(getattr(cfg, "OEM_DYN_BG_STAGE2_START", 0.0)),
            float(getattr(cfg, "OEM_DYN_BG_STAGE2_END", 0.12)),
        )
    else:
        lambda_dyn_pos = float(getattr(cfg, "OEM_DYN_POS_STAGE3", 0.35))
        lambda_dyn_bg = float(getattr(cfg, "OEM_DYN_BG_STAGE3", 0.12))
    return max(0.0, float(lambda_dyn_pos)), max(0.0, float(lambda_dyn_bg))


def get_fusion_weights(epoch, cfg):
    reset_epoch = get_reset_epoch(cfg)
    reset_enabled = is_finetune_reset_enabled(cfg)
    mode = str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower()
    if mode == "dabe_sticky":
        stage2_start = int(getattr(cfg, "DABE_STICKY_STAGE2_START", 7))
        stage2_end = int(getattr(cfg, "DABE_STICKY_STAGE2_END", 15))
        stage3_start = int(getattr(cfg, "DABE_STICKY_STAGE3_START", 16))
        stage3_end = int(getattr(cfg, "DABE_STICKY_STAGE3_END", 20))
        after_epoch = int(getattr(cfg, "DABE_STICKY_AFTER_EPOCH", 21))

        if int(epoch) < stage2_start:
            dabe_weight = float(getattr(cfg, "DABE_STICKY_E1_E6_DABE_WEIGHT", 1.0))
            teacher_weight = 1.0 - dabe_weight
        elif int(epoch) <= stage2_end:
            dabe_weight = _linear_schedule_value(
                epoch,
                stage2_start,
                stage2_end,
                float(getattr(cfg, "DABE_STICKY_STAGE2_DABE_START", 0.85)),
                float(getattr(cfg, "DABE_STICKY_STAGE2_DABE_END", 0.55)),
            )
            teacher_weight = 1.0 - dabe_weight
        elif int(epoch) < stage3_start:
            dabe_weight = float(getattr(cfg, "DABE_STICKY_STAGE2_DABE_END", 0.55))
            teacher_weight = 1.0 - dabe_weight
        elif int(epoch) <= stage3_end:
            dabe_weight = _linear_schedule_value(
                epoch,
                stage3_start,
                stage3_end,
                float(getattr(cfg, "DABE_STICKY_STAGE3_DABE_START", 0.55)),
                float(getattr(cfg, "DABE_STICKY_STAGE3_DABE_END", 0.25)),
            )
            teacher_weight = 1.0 - dabe_weight
        elif int(epoch) < after_epoch:
            dabe_weight = float(getattr(cfg, "DABE_STICKY_STAGE3_DABE_END", 0.25))
            teacher_weight = 1.0 - dabe_weight
        else:
            dabe_weight = float(getattr(cfg, "DABE_STICKY_AFTER_DABE_WEIGHT", 0.15))
            teacher_weight = float(getattr(cfg, "DABE_STICKY_AFTER_TEACHER_WEIGHT", 0.85))
            if abs((dabe_weight + teacher_weight) - 1.0) > 1e-6:
                raise RuntimeError(
                    "DABE sticky after weights must sum to 1.0, got "
                    f"{dabe_weight:.6f} + {teacher_weight:.6f}."
                )

        dabe_weight = max(0.0, min(1.0, float(dabe_weight)))
        teacher_weight = max(0.0, min(1.0, float(teacher_weight)))
        return dabe_weight, teacher_weight

    if mode == "orig20_hold_until_reset":
        if reset_enabled and epoch >= reset_epoch:
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
        if reset_enabled and epoch >= reset_epoch:
            return 0.0, 1.0
        default_pre_epochs = reset_epoch - 1 if reset_enabled else int(getattr(cfg, "MAX_EPOCH", 20))
        pre_epochs = int(getattr(cfg, "TEACHER_FUSION_PRE_RESET_EPOCHS", default_pre_epochs))
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
    if mode == "dabe_pu_oem":
        return 1.0, 0.0, mode
    if mode == "dabe_pu_balanced_v2":
        static_weight, teacher_weight = get_dabe_pu_balanced_v2_schedule(epoch, cfg)
        return static_weight, teacher_weight, mode
    if mode == "dabe_pu_despl_sched":
        static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
        return static_weight, teacher_weight, mode
    if mode == "dabe_pu_conf":
        static_weight, teacher_weight = get_dabe_pu_schedule(epoch, cfg)
        return static_weight, teacher_weight, mode
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
    if mode in {"linear_to_095_before_reset", "orig20_hold_until_reset", "dabe_sticky"}:
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
    if (
        not use_complex_head_lr_policy(cfg)
        or not is_finetune_reset_enabled(cfg)
        or epoch >= get_reset_epoch(cfg)
    ):
        return None
    lr = get_hold_cosine_lr(epoch, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def should_step_iter_scheduler(epoch, cfg):
    return not (
        use_linear_floor_two_stage_lr(cfg)
        or (
            use_complex_head_lr_policy(cfg)
            and is_finetune_reset_enabled(cfg)
            and epoch < get_reset_epoch(cfg)
        )
    )


def complex_head_post_reset_lr(cfg):
    return float(getattr(cfg, "COMPLEX_HEAD_POST_RESET_LR", getattr(cfg, "FINETUNE_RESET_LR", cfg.DINO["lr"])))


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


def apply_finetune_reset(
    logger,
    cfg,
    epoch,
    student,
    teacher,
    optimizer,
    scheduler,
    global_step,
    lr_floor_activated_logged,
):
    rebuild_optimizer = bool(getattr(cfg, "FINETUNE_RESET_REBUILD_OPTIMIZER", True))
    rebuild_scheduler = bool(getattr(cfg, "FINETUNE_RESET_REBUILD_SCHEDULER", True))
    reset_global_step = bool(getattr(cfg, "FINETUNE_RESET_GLOBAL_STEP", True))
    reset_teacher = bool(getattr(cfg, "FINETUNE_RESET_TEACHER", False))
    if rebuild_optimizer:
        optimizer, scheduler = build_optimizer_scheduler(cfg, student, lr=complex_head_post_reset_lr(cfg))
    elif rebuild_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=25,
            gamma=0.95,
        )
    if reset_global_step:
        global_step = 0
    if reset_teacher:
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)
    if bool(getattr(cfg, "FINETUNE_RESET_FORCE_LR_FLOOR", False)):
        reset_lr = float(getattr(cfg, "FINETUNE_RESET_LR", getattr(cfg, "LR_FLOOR", current_lr(optimizer))))
        set_optimizer_lr(optimizer, reset_lr)
    elif bool(getattr(cfg, "LR_FLOOR_APPLY_AFTER_FINETUNE_RESET", True)):
        lr_floor_clamped, scheduler_lr, clamped_lr = apply_lr_floor(optimizer, cfg)
        if lr_floor_clamped and not lr_floor_activated_logged:
            logger.log(
                "[LR Floor] activated | "
                f"global_step={global_step} | "
                f"scheduler_lr={scheduler_lr:.8f} | "
                f"clamped_lr={clamped_lr:.8f}"
            )
            lr_floor_activated_logged = True
    force_lr_floor = bool(getattr(cfg, "FINETUNE_RESET_FORCE_LR_FLOOR", False))
    logger.log(
        f"[FinetuneReset] epoch={int(epoch):03d} | "
        f"timing={finetune_reset_timing(cfg)} | "
        f"rebuild_optimizer={rebuild_optimizer} | "
        f"rebuild_scheduler={rebuild_scheduler} | "
        f"reset_global_step={reset_global_step} | "
        f"reset_teacher={reset_teacher} | "
        f"force_lr_floor={force_lr_floor} | "
        f"lr_after_reset={current_lr(optimizer):.8f}"
    )
    return optimizer, scheduler, global_step, lr_floor_activated_logged


def use_multi_level_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))


def use_multi_view_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False))


def multi_view_types(cfg):
    return [str(view).lower() for view in getattr(cfg, "MULTI_VIEW_TYPES", [])]


def use_hflip_view(cfg):
    return use_multi_view_feature(cfg) and "hflip" in multi_view_types(cfg)


def use_view_consistency(cfg):
    return use_hflip_view(cfg) and bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False))


def use_proto_contrast(cfg):
    return use_hflip_view(cfg) and bool(getattr(cfg, "USE_PROTO_CONTRAST", False))


def use_hflip_training_view(cfg):
    return use_view_consistency(cfg) or use_proto_contrast(cfg)


def use_dagp_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp"


def use_dagp_safe_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe"


def use_raw_feature_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {"dagp", "dagp_safe"}


def use_ndr_branch(cfg):
    return bool(getattr(cfg, "USE_NDR_BRANCH", False))


def use_tadr_router(cfg):
    return bool(getattr(cfg, "USE_TADR_ROUTER", False))


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


def make_hflip_image_68(cfg, batch, device):
    if not use_ndr_branch(cfg):
        return None
    if "image_hflip_68" not in batch:
        raise KeyError("HFlip view with USE_NDR_BRANCH=True requires batch['image_hflip_68'].")
    return batch["image_hflip_68"].to(device, non_blocking=True).float()


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
    return make_single_feature_model_input(cfg, feature)


def make_single_feature_model_input(cfg, feature):
    if use_raw_feature_head(cfg):
        return feature
    return F.interpolate(feature, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear")


def make_hflip_model_input(cfg, batch, device):
    if use_multi_level_feature(cfg):
        raise RuntimeError("HFlip multi-view consistency currently supports single-level cached DINO features only.")
    if "feature_hflip" not in batch:
        raise KeyError("USE_MULTI_VIEW_FEATURE=True with hflip requires batch['feature_hflip'].")
    feature = batch["feature_hflip"].to(device, non_blocking=True).float()
    return make_single_feature_model_input(cfg, feature)


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def view_consistency_lambda(cfg, epoch):
    if not use_view_consistency(cfg):
        return 0.0
    max_lambda = float(getattr(cfg, "LAMBDA_VIEW_MAX", 0.0))
    if max_lambda <= 0.0:
        return 0.0
    warmup_epoch = int(getattr(cfg, "VIEW_WARMUP_EPOCH", 6))
    ramp_start = int(getattr(cfg, "VIEW_RAMP_START_EPOCH", warmup_epoch + 1))
    ramp_end = int(getattr(cfg, "VIEW_RAMP_END_EPOCH", ramp_start))
    if int(epoch) <= warmup_epoch or int(epoch) < ramp_start:
        scale = 0.0
    elif int(epoch) >= ramp_end:
        scale = 1.0
    else:
        denom = max(1, ramp_end - ramp_start + 1)
        scale = float(int(epoch) - ramp_start + 1) / float(denom)
    if is_at_or_after_finetune_reset(cfg, epoch):
        scale *= float(getattr(cfg, "VIEW_AFTER_RESET_SCALE", 0.0))
    return max_lambda * max(0.0, min(1.0, scale))


def proto_contrast_lambda(cfg, epoch):
    if not use_proto_contrast(cfg):
        return 0.0
    max_lambda = float(getattr(cfg, "LAMBDA_PROTO_MAX", 0.0))
    if max_lambda <= 0.0:
        return 0.0
    warmup_epoch = int(getattr(cfg, "PROTO_WARMUP_EPOCH", 6))
    ramp_start = int(getattr(cfg, "PROTO_RAMP_START_EPOCH", warmup_epoch + 1))
    ramp_end = int(getattr(cfg, "PROTO_RAMP_END_EPOCH", ramp_start))
    if int(epoch) <= warmup_epoch or int(epoch) < ramp_start:
        scale = 0.0
    elif int(epoch) >= ramp_end:
        scale = 1.0
    else:
        denom = max(1, ramp_end - ramp_start + 1)
        scale = float(int(epoch) - ramp_start + 1) / float(denom)
    if is_at_or_after_finetune_reset(cfg, epoch):
        scale *= float(getattr(cfg, "PROTO_AFTER_RESET_SCALE", 0.0))
    return max_lambda * max(0.0, min(1.0, scale))


def compute_view_consistency_loss(normal_logits, hflip_logits, pseudo_68, cfg):
    if str(getattr(cfg, "VIEW_CONF_SOURCE", "despl_core")).lower() != "despl_core":
        raise RuntimeError("VIEW_CONF_SOURCE currently supports only 'despl_core'.")
    loss_type = str(getattr(cfg, "VIEW_CONSISTENCY_TYPE", "l1")).lower()
    normal_prob = normal_logits.sigmoid()
    hflip_prob_inv = torch.flip(hflip_logits.sigmoid(), dims=[-1])
    diff = normal_prob - hflip_prob_inv
    if loss_type == "l1":
        loss_map = diff.abs()
    elif loss_type == "mse":
        loss_map = diff.square()
    else:
        raise RuntimeError(f"Unsupported VIEW_CONSISTENCY_TYPE: {loss_type}")

    fg = pseudo_68 > float(getattr(cfg, "VIEW_FG_THRESH", 0.8))
    bg = pseudo_68 < float(getattr(cfg, "VIEW_BG_THRESH", 0.2))
    core = fg | bg
    weight = core.float()
    boundary_weight = float(getattr(cfg, "VIEW_BOUNDARY_WEIGHT", 0.0))
    if boundary_weight > 0.0:
        weight = torch.where(core, weight, torch.full_like(weight, boundary_weight))
    weight = weight.detach()
    denom = weight.sum().clamp_min(1.0)
    loss = (loss_map * weight).sum() / denom
    with torch.no_grad():
        stats = {
            "loss_raw": float(loss.detach().item()),
            "core_ratio": float(core.float().mean().item()),
            "mean_abs_diff": float(diff.detach().abs().mean().item()),
            "weight_mean": float(weight.mean().item()),
            "hflip_prob_inv": hflip_prob_inv.detach(),
        }
    return loss, stats


def build_proto_core_masks(pseudo, prob_n, prob_f_inv, cfg):
    mode = str(getattr(cfg, "PROTO_CORE_MODE", "despl_pred_agree")).lower()
    fg = pseudo > float(getattr(cfg, "PROTO_FG_THRESH", 0.90))
    bg = pseudo < float(getattr(cfg, "PROTO_BG_THRESH", 0.10))
    if mode == "despl_pred_agree":
        fg = (
            fg
            & (prob_n.detach() > float(getattr(cfg, "PROTO_PRED_FG_THRESH", 0.60)))
            & (prob_f_inv.detach() > float(getattr(cfg, "PROTO_PRED_FG_THRESH", 0.60)))
        )
        bg = (
            bg
            & (prob_n.detach() < float(getattr(cfg, "PROTO_PRED_BG_THRESH", 0.40)))
            & (prob_f_inv.detach() < float(getattr(cfg, "PROTO_PRED_BG_THRESH", 0.40)))
        )
    elif mode != "despl_only":
        raise RuntimeError(f"Unsupported PROTO_CORE_MODE: {mode}")
    if bool(getattr(cfg, "PROTO_DETACH_MASK", True)):
        fg = fg.detach()
        bg = bg.detach()
    return fg.bool(), bg.bool()


def _proto_zero_stats():
    return {
        "proto_mode": "global",
        "loss_proto": 0.0,
        "align_loss": 0.0,
        "sep_loss": 0.0,
        "pixel_loss": 0.0,
        "pixel_fg_loss": 0.0,
        "pixel_bg_loss": 0.0,
        "valid_ratio": 0.0,
        "fg_core_ratio": 0.0,
        "bg_core_ratio": 0.0,
        "bg_hard_ratio": 0.0,
        "bg_ring_ratio": 0.0,
        "bg_disagree_ratio": 0.0,
        "bg_residual_ratio": 0.0,
        "hard_fg_ratio": 0.0,
        "hard_bg_ratio": 0.0,
        "sep_active_ratio": 0.0,
        "fg_fallback_ratio": 0.0,
        "cos_fg_view": 0.0,
        "cos_bg_view": 0.0,
        "cos_fg_bg": 0.0,
    }


def _masked_mean_proto(z, mask):
    z_flat = z.flatten(1)
    mask_flat = mask.flatten()
    proto = z_flat[:, mask_flat].mean(dim=1)
    return F.normalize(proto, dim=0)


def _reliable_indices(mask, reliability, max_pixels):
    mask_flat = mask.flatten()
    idx = torch.nonzero(mask_flat, as_tuple=False).flatten()
    if int(max_pixels) > 0 and idx.numel() > int(max_pixels):
        scores = reliability.flatten().index_select(0, idx)
        top_idx = torch.topk(scores, k=int(max_pixels), largest=True).indices
        idx = idx.index_select(0, top_idx)
    return idx


def _pixel_proto_loss(z_flat, fg_idx, bg_idx, p_fg, p_bg, tau):
    losses = []
    if fg_idx.numel() > 0:
        z_fg = z_flat.index_select(1, fg_idx).transpose(0, 1)
        sim_fg = torch.matmul(z_fg, p_fg)
        sim_bg = torch.matmul(z_fg, p_bg)
        losses.append(F.softplus((sim_bg - sim_fg) / tau).mean())
    if bg_idx.numel() > 0:
        z_bg = z_flat.index_select(1, bg_idx).transpose(0, 1)
        sim_bg = torch.matmul(z_bg, p_bg)
        sim_fg = torch.matmul(z_bg, p_fg)
        losses.append(F.softplus((sim_fg - sim_bg) / tau).mean())
    if not losses:
        return z_flat.sum() * 0.0
    return torch.stack(losses).mean()


def compute_proto_contrast_loss(out_n, out_f, pseudo_68, cfg):
    mode = str(getattr(cfg, "PROTO_MODE", "global")).lower()
    if mode in {"", "global"}:
        return compute_proto_contrast_loss_global(out_n, out_f, pseudo_68, cfg)
    if mode == "hard_selective":
        return compute_proto_contrast_loss_hard_selective(out_n, out_f, pseudo_68, cfg)
    raise RuntimeError(f"Unsupported PROTO_MODE: {mode}")


def compute_proto_contrast_loss_global(out_n, out_f, pseudo_68, cfg):
    if not isinstance(out_n, dict) or not isinstance(out_f, dict):
        raise RuntimeError("USE_PROTO_CONTRAST=True requires dict outputs from normal and hflip forwards.")
    if "proto_feat" not in out_n or "proto_feat" not in out_f:
        raise RuntimeError("USE_PROTO_CONTRAST=True requires output['proto_feat'].")
    logits_n = resize_logits_for_loss(extract_logits(out_n), cfg)
    logits_f = resize_logits_for_loss(extract_logits(out_f), cfg)
    prob_n = logits_n.sigmoid()
    prob_f_inv = torch.flip(logits_f.sigmoid(), dims=[-1])
    z_n = F.normalize(out_n["proto_feat"], dim=1)
    z_f = F.normalize(torch.flip(out_f["proto_feat"], dims=[-1]), dim=1)
    target_size = (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    if tuple(z_n.shape[-2:]) != target_size:
        z_n = F.interpolate(z_n, size=target_size, mode="bilinear", align_corners=False)
        z_n = F.normalize(z_n, dim=1)
    if tuple(z_f.shape[-2:]) != target_size:
        z_f = F.interpolate(z_f, size=target_size, mode="bilinear", align_corners=False)
        z_f = F.normalize(z_f, dim=1)

    fg_core, bg_core = build_proto_core_masks(pseudo_68, prob_n, prob_f_inv, cfg)
    stats = _proto_zero_stats()
    stats["proto_mode"] = "global"
    stats["fg_core_ratio"] = float(fg_core.float().mean().item())
    stats["bg_core_ratio"] = float(bg_core.float().mean().item())

    min_fg = int(getattr(cfg, "PROTO_MIN_FG_PIXELS", 16))
    min_bg = int(getattr(cfg, "PROTO_MIN_BG_PIXELS", 128))
    max_pixels = int(getattr(cfg, "PROTO_MAX_PIXELS_PER_CLASS", 256))
    tau = float(getattr(cfg, "PROTO_TAU", 0.10))
    if tau <= 0.0:
        raise RuntimeError(f"PROTO_TAU must be positive, got {tau}")
    margin = float(getattr(cfg, "PROTO_SEP_MARGIN", 0.20))
    align_w = float(getattr(cfg, "PROTO_ALIGN_WEIGHT", 1.0))
    sep_w = float(getattr(cfg, "PROTO_SEP_WEIGHT", 0.5))
    pixel_w = float(getattr(cfg, "PROTO_PIXEL_WEIGHT", 1.0))
    if not bool(getattr(cfg, "PROTO_SKIP_INVALID", True)):
        raise RuntimeError("PROTO_SKIP_INVALID=False is not implemented for MVFlip-Proto.")

    losses = []
    align_losses = []
    sep_losses = []
    pixel_losses = []
    cos_fg_values = []
    cos_bg_values = []
    cos_sep_values = []
    batch_size = int(z_n.shape[0])
    for index in range(batch_size):
        fg_mask = fg_core[index, 0]
        bg_mask = bg_core[index, 0]
        fg_count = int(fg_mask.sum().item())
        bg_count = int(bg_mask.sum().item())
        if fg_count < min_fg or bg_count < min_bg:
            continue

        z_n_i = z_n[index]
        z_f_i = z_f[index]
        pseudo_i = pseudo_68[index, 0]
        p_fg_n = _masked_mean_proto(z_n_i, fg_mask)
        p_bg_n = _masked_mean_proto(z_n_i, bg_mask)
        p_fg_f = _masked_mean_proto(z_f_i, fg_mask)
        p_bg_f = _masked_mean_proto(z_f_i, bg_mask)

        cos_fg = F.cosine_similarity(p_fg_n, p_fg_f, dim=0)
        cos_bg = F.cosine_similarity(p_bg_n, p_bg_f, dim=0)
        loss_align = (1.0 - cos_fg) + (1.0 - cos_bg)
        p_fg = F.normalize(0.5 * (p_fg_n + p_fg_f), dim=0)
        p_bg = F.normalize(0.5 * (p_bg_n + p_bg_f), dim=0)
        cos_fg_bg = F.cosine_similarity(p_fg, p_bg, dim=0)
        loss_sep = F.relu(cos_fg_bg - margin)

        fg_idx = _reliable_indices(fg_mask, pseudo_i, max_pixels)
        bg_idx = _reliable_indices(bg_mask, 1.0 - pseudo_i, max_pixels)
        p_fg_pix = p_fg.detach() if bool(getattr(cfg, "PROTO_DETACH_PIXEL_PROTOTYPE", True)) else p_fg
        p_bg_pix = p_bg.detach() if bool(getattr(cfg, "PROTO_DETACH_PIXEL_PROTOTYPE", True)) else p_bg
        z_n_flat = z_n_i.flatten(1)
        z_f_flat = z_f_i.flatten(1)
        loss_pixel = 0.5 * (
            _pixel_proto_loss(z_n_flat, fg_idx, bg_idx, p_fg_pix, p_bg_pix, tau)
            + _pixel_proto_loss(z_f_flat, fg_idx, bg_idx, p_fg_pix, p_bg_pix, tau)
        )
        loss_i = align_w * loss_align + sep_w * loss_sep + pixel_w * loss_pixel
        losses.append(loss_i)
        align_losses.append(loss_align.detach())
        sep_losses.append(loss_sep.detach())
        pixel_losses.append(loss_pixel.detach())
        cos_fg_values.append(cos_fg.detach())
        cos_bg_values.append(cos_bg.detach())
        cos_sep_values.append(cos_fg_bg.detach())

    if not losses:
        return logits_n.sum() * 0.0, stats

    loss = torch.stack(losses).mean()
    stats.update(
        {
            "loss_proto": float(loss.detach().item()),
            "align_loss": float(torch.stack(align_losses).mean().item()),
            "sep_loss": float(torch.stack(sep_losses).mean().item()),
            "pixel_loss": float(torch.stack(pixel_losses).mean().item()),
            "valid_ratio": float(len(losses) / max(batch_size, 1)),
            "cos_fg_view": float(torch.stack(cos_fg_values).mean().item()),
            "cos_bg_view": float(torch.stack(cos_bg_values).mean().item()),
            "cos_fg_bg": float(torch.stack(cos_sep_values).mean().item()),
        }
    )
    return loss, stats


def _resize_proto_map(value, target_size):
    if value is None:
        return None
    if tuple(value.shape[-2:]) == target_size:
        return value
    return F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)


def _dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)
    return dilated > 0.5


def _erode_mask(mask, radius):
    return ~_dilate_mask(~mask.bool(), radius)


def _topk_mask(mask, score, max_pixels):
    mask = mask.bool()
    idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    max_pixels = int(max_pixels)
    if max_pixels > 0 and idx.numel() > max_pixels:
        scores = score.detach().flatten().index_select(0, idx)
        top_idx = torch.topk(scores, k=max_pixels, largest=True).indices
        idx = idx.index_select(0, top_idx)
    out = torch.zeros_like(mask, dtype=torch.bool).flatten()
    out.index_fill_(0, idx, True)
    return out.view_as(mask)


def _hard_margin_pixel_proto_losses(z_flat, fg_idx, bg_idx, p_fg, p_bg, margin):
    zero = z_flat.sum() * 0.0
    loss_fg = zero
    loss_bg = zero
    if fg_idx.numel() > 0:
        z_fg = z_flat.index_select(1, fg_idx).transpose(0, 1)
        sim_fg = torch.matmul(z_fg, p_fg)
        sim_bg = torch.matmul(z_fg, p_bg)
        loss_fg = F.relu(sim_bg - sim_fg + margin).mean()
    if bg_idx.numel() > 0:
        z_bg = z_flat.index_select(1, bg_idx).transpose(0, 1)
        sim_bg = torch.matmul(z_bg, p_bg)
        sim_fg = torch.matmul(z_bg, p_fg)
        loss_bg = F.relu(sim_fg - sim_bg + margin).mean()
    return loss_fg, loss_bg


def build_hard_selective_masks(pseudo, prob_n, prob_f_inv, coarse_prob=None, residual_logits=None, cfg=None):
    if cfg is None:
        raise RuntimeError("build_hard_selective_masks requires cfg.")
    if not bool(getattr(cfg, "PROTO_USE_HARD_BG_ONLY", True)):
        raise RuntimeError("MVProto-HS currently requires PROTO_USE_HARD_BG_ONLY=True.")

    target_size = tuple(pseudo.shape[-2:])
    coarse_prob = _resize_proto_map(coarse_prob, target_size)
    residual_logits = _resize_proto_map(residual_logits, target_size)

    pred_mean = 0.5 * (prob_n.detach() + prob_f_inv.detach())
    pseudo_detached = pseudo.detach()
    pseudo_fg = pseudo_detached > float(getattr(cfg, "PROTO_FG_THRESH", 0.90))
    pseudo_bg = pseudo_detached < float(getattr(cfg, "PROTO_BG_THRESH", 0.10))

    radius = int(getattr(cfg, "PROTO_BG_RING_RADIUS", 3))
    fg_dilate = _dilate_mask(pseudo_fg, radius)
    fg_erode = _erode_mask(pseudo_fg, radius)
    boundary_ring = fg_dilate & ~fg_erode
    use_bg_ring = bool(getattr(cfg, "PROTO_USE_BG_RING", True))
    bg_ring = pseudo_bg & fg_dilate & ~pseudo_fg if use_bg_ring else torch.zeros_like(pseudo_bg)

    bg_low = float(getattr(cfg, "PROTO_PRED_BG_LOW", 0.20))
    bg_high = float(getattr(cfg, "PROTO_PRED_BG_HIGH", 0.60))
    bg_pred_hard = pseudo_bg & (pred_mean >= bg_low) & (pred_mean <= bg_high)

    if bool(getattr(cfg, "PROTO_USE_DISAGREE_MAP", True)) and coarse_prob is not None:
        coarse_prob = coarse_prob.detach()
        disagree = (prob_n.detach() - coarse_prob).abs()
        bg_disagree = (
            pseudo_bg
            & (disagree > float(getattr(cfg, "PROTO_DISAGREE_THRESH", 0.15)))
            & (pred_mean > 0.15)
        )
    else:
        bg_disagree = torch.zeros_like(pseudo_bg)

    if bool(getattr(cfg, "PROTO_USE_NDR_RESIDUAL_FOR_HARD", True)) and residual_logits is not None:
        res_abs = residual_logits.detach().abs()
        quantile = float(getattr(cfg, "PROTO_NDR_RESIDUAL_Q", 0.80))
        res_thr = torch.quantile(res_abs.flatten(2), quantile, dim=2, keepdim=True).view(
            res_abs.shape[0],
            res_abs.shape[1],
            1,
            1,
        )
        bg_residual_hard = pseudo_bg & (res_abs > res_thr) & (pred_mean > 0.15)
    else:
        bg_residual_hard = torch.zeros_like(pseudo_bg)

    bg_hard = pseudo_bg & (bg_pred_hard | bg_ring | bg_disagree | bg_residual_hard)
    bg_hard_score = (
        pred_mean
        + 0.5 * bg_ring.float()
        + 0.5 * bg_disagree.float()
        + 0.5 * bg_residual_hard.float()
    ).detach()

    fg_pred_agree = (
        pseudo_fg
        & (prob_n.detach() > float(getattr(cfg, "PROTO_PRED_FG_THRESH", 0.60)))
        & (prob_f_inv.detach() > float(getattr(cfg, "PROTO_PRED_FG_THRESH", 0.60)))
    )
    fg_inner = pseudo_fg & ~boundary_ring
    fg_core_raw = fg_pred_agree & fg_inner
    fg_core = fg_core_raw.clone()
    fg_fallback = torch.zeros(pseudo.shape[0], device=pseudo.device, dtype=torch.bool)
    min_fg = int(getattr(cfg, "PROTO_MIN_FG_PIXELS", 16))
    fg_raw_count = fg_core_raw.flatten(1).sum(dim=1)
    fg_agree_count = fg_pred_agree.flatten(1).sum(dim=1)
    fallback_mask = (fg_raw_count < min_fg) & (fg_agree_count >= min_fg)
    if bool(fallback_mask.any().item()):
        fg_core[fallback_mask] = fg_pred_agree[fallback_mask]
        fg_fallback[fallback_mask] = True

    fg_confident = pseudo_fg & (pred_mean > 0.70)
    fg_boundary_hard = fg_confident & boundary_ring
    fg_low_margin = pseudo_fg & (pred_mean >= 0.60) & (pred_mean <= 0.85)
    hard_fg = fg_boundary_hard | fg_low_margin
    fg_hard_score = (1.0 - (pred_mean - 0.70).abs()).detach()

    if bool(getattr(cfg, "PROTO_DETACH_MASK", True)):
        fg_core = fg_core.detach()
        bg_hard = bg_hard.detach()
        bg_ring = bg_ring.detach()
        bg_disagree = bg_disagree.detach()
        bg_residual_hard = bg_residual_hard.detach()
        hard_fg = hard_fg.detach()

    return {
        "fg_core": fg_core.bool(),
        "bg_hard": bg_hard.bool(),
        "bg_ring": bg_ring.bool(),
        "bg_disagree": bg_disagree.bool(),
        "bg_residual_hard": bg_residual_hard.bool(),
        "hard_fg": hard_fg.bool(),
        "bg_hard_score": bg_hard_score,
        "fg_hard_score": fg_hard_score,
        "fg_fallback": fg_fallback,
    }


def compute_proto_contrast_loss_hard_selective(out_n, out_f, pseudo_68, cfg):
    if not isinstance(out_n, dict) or not isinstance(out_f, dict):
        raise RuntimeError("USE_PROTO_CONTRAST=True requires dict outputs from normal and hflip forwards.")
    if "proto_feat" not in out_n or "proto_feat" not in out_f:
        raise RuntimeError("USE_PROTO_CONTRAST=True requires output['proto_feat'].")
    if str(getattr(cfg, "PROTO_CORE_MODE", "despl_pred_agree_hard")).lower() != "despl_pred_agree_hard":
        raise RuntimeError("PROTO_MODE='hard_selective' requires PROTO_CORE_MODE='despl_pred_agree_hard'.")
    if str(getattr(cfg, "PROTO_PIXEL_LOSS_MODE", "hard_margin")).lower() != "hard_margin":
        raise RuntimeError("MVProto-HS currently supports PROTO_PIXEL_LOSS_MODE='hard_margin' only.")
    if not bool(getattr(cfg, "PROTO_SKIP_INVALID", True)):
        raise RuntimeError("PROTO_SKIP_INVALID=False is not implemented for MVProto-HS.")

    logits_n = resize_logits_for_loss(extract_logits(out_n), cfg)
    logits_f = resize_logits_for_loss(extract_logits(out_f), cfg)
    prob_n = logits_n.sigmoid()
    prob_f_inv = torch.flip(logits_f.sigmoid(), dims=[-1])
    z_n = F.normalize(out_n["proto_feat"], dim=1)
    z_f = F.normalize(torch.flip(out_f["proto_feat"], dims=[-1]), dim=1)
    target_size = (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    if tuple(z_n.shape[-2:]) != target_size:
        z_n = F.interpolate(z_n, size=target_size, mode="bilinear", align_corners=False)
        z_n = F.normalize(z_n, dim=1)
    if tuple(z_f.shape[-2:]) != target_size:
        z_f = F.interpolate(z_f, size=target_size, mode="bilinear", align_corners=False)
        z_f = F.normalize(z_f, dim=1)

    coarse_prob = out_n.get("coarse_prob", out_n.get("coarse_prob_68", None))
    residual_logits = out_n.get("residual_logits_68", out_n.get("ndr_residual", None))
    masks = build_hard_selective_masks(
        pseudo_68,
        prob_n,
        prob_f_inv,
        coarse_prob=coarse_prob,
        residual_logits=residual_logits,
        cfg=cfg,
    )

    stats = _proto_zero_stats()
    stats["proto_mode"] = "hard_selective"
    stats["fg_core_ratio"] = float(masks["fg_core"].float().mean().item())
    stats["bg_core_ratio"] = float(masks["bg_hard"].float().mean().item())
    stats["bg_hard_ratio"] = stats["bg_core_ratio"]
    stats["bg_ring_ratio"] = float(masks["bg_ring"].float().mean().item())
    stats["bg_disagree_ratio"] = float(masks["bg_disagree"].float().mean().item())
    stats["bg_residual_ratio"] = float(masks["bg_residual_hard"].float().mean().item())
    stats["fg_fallback_ratio"] = float(masks["fg_fallback"].float().mean().item())

    min_fg = int(getattr(cfg, "PROTO_MIN_FG_PIXELS", 16))
    min_bg = int(getattr(cfg, "PROTO_MIN_BG_PIXELS", 32))
    max_fg = int(getattr(cfg, "PROTO_MAX_FG_PIXELS", getattr(cfg, "PROTO_MAX_PIXELS_PER_CLASS", 128)))
    max_bg = int(getattr(cfg, "PROTO_MAX_BG_PIXELS", getattr(cfg, "PROTO_MAX_PIXELS_PER_CLASS", 256)))
    margin = float(getattr(cfg, "PROTO_SEP_MARGIN", 0.20))
    pixel_margin = float(getattr(cfg, "PROTO_PIXEL_MARGIN", 0.20))
    align_w = float(getattr(cfg, "PROTO_ALIGN_WEIGHT", 1.0))
    sep_w = float(getattr(cfg, "PROTO_SEP_WEIGHT", 0.25))
    pixel_fg_w = float(getattr(cfg, "PROTO_PIXEL_FG_WEIGHT", 0.30))
    pixel_bg_w = float(getattr(cfg, "PROTO_PIXEL_BG_WEIGHT", 1.0))

    losses = []
    align_losses = []
    sep_losses = []
    pixel_losses = []
    pixel_fg_losses = []
    pixel_bg_losses = []
    cos_fg_values = []
    cos_bg_values = []
    cos_sep_values = []
    sep_active_values = []
    selected_hard_fg = torch.zeros_like(masks["hard_fg"])
    selected_hard_bg = torch.zeros_like(masks["bg_hard"])
    batch_size = int(z_n.shape[0])
    for index in range(batch_size):
        fg_mask = masks["fg_core"][index, 0]
        bg_hard = masks["bg_hard"][index, 0]
        bg_proto_mask = _topk_mask(bg_hard, masks["bg_hard_score"][index, 0], max_bg)
        fg_count = int(fg_mask.sum().item())
        bg_count = int(bg_proto_mask.sum().item())
        if fg_count < min_fg or bg_count < min_bg:
            continue

        hard_fg_mask = masks["hard_fg"][index, 0]
        fg_idx = _reliable_indices(hard_fg_mask, masks["fg_hard_score"][index, 0], max_fg)
        if fg_idx.numel() == 0:
            fg_idx = _reliable_indices(fg_mask, pseudo_68[index, 0], max_fg)
        bg_idx = torch.nonzero(bg_proto_mask.flatten(), as_tuple=False).flatten()
        if fg_idx.numel() > 0:
            selected_hard_fg[index, 0].flatten().index_fill_(0, fg_idx, True)
        if bg_idx.numel() > 0:
            selected_hard_bg[index, 0].flatten().index_fill_(0, bg_idx, True)

        z_n_i = z_n[index]
        z_f_i = z_f[index]
        p_fg_n = _masked_mean_proto(z_n_i, fg_mask)
        p_bg_n = _masked_mean_proto(z_n_i, bg_proto_mask)
        p_fg_f = _masked_mean_proto(z_f_i, fg_mask)
        p_bg_f = _masked_mean_proto(z_f_i, bg_proto_mask)

        cos_fg = F.cosine_similarity(p_fg_n, p_fg_f, dim=0)
        cos_bg = F.cosine_similarity(p_bg_n, p_bg_f, dim=0)
        loss_align = (1.0 - cos_fg) + (1.0 - cos_bg)
        p_fg = F.normalize(0.5 * (p_fg_n + p_fg_f), dim=0)
        p_bg = F.normalize(0.5 * (p_bg_n + p_bg_f), dim=0)
        cos_fg_bg = F.cosine_similarity(p_fg, p_bg, dim=0)
        loss_sep = F.relu(cos_fg_bg - margin)

        p_fg_pix = p_fg.detach() if bool(getattr(cfg, "PROTO_DETACH_PIXEL_PROTOTYPE", True)) else p_fg
        p_bg_pix = p_bg.detach() if bool(getattr(cfg, "PROTO_DETACH_PIXEL_PROTOTYPE", True)) else p_bg
        z_n_flat = z_n_i.flatten(1)
        z_f_flat = z_f_i.flatten(1)
        loss_fg_n, loss_bg_n = _hard_margin_pixel_proto_losses(
            z_n_flat,
            fg_idx,
            bg_idx,
            p_fg_pix,
            p_bg_pix,
            pixel_margin,
        )
        loss_fg_f, loss_bg_f = _hard_margin_pixel_proto_losses(
            z_f_flat,
            fg_idx,
            bg_idx,
            p_fg_pix,
            p_bg_pix,
            pixel_margin,
        )
        loss_pixel_fg = 0.5 * (loss_fg_n + loss_fg_f)
        loss_pixel_bg = 0.5 * (loss_bg_n + loss_bg_f)
        loss_pixel = pixel_fg_w * loss_pixel_fg + pixel_bg_w * loss_pixel_bg
        loss_i = align_w * loss_align + sep_w * loss_sep + loss_pixel
        losses.append(loss_i)
        align_losses.append(loss_align.detach())
        sep_losses.append(loss_sep.detach())
        pixel_losses.append(loss_pixel.detach())
        pixel_fg_losses.append(loss_pixel_fg.detach())
        pixel_bg_losses.append(loss_pixel_bg.detach())
        cos_fg_values.append(cos_fg.detach())
        cos_bg_values.append(cos_bg.detach())
        cos_sep_values.append(cos_fg_bg.detach())
        sep_active_values.append((cos_fg_bg.detach() > margin).float())

    stats["hard_fg_ratio"] = float(selected_hard_fg.float().mean().item())
    stats["hard_bg_ratio"] = float(selected_hard_bg.float().mean().item())
    if not losses:
        return logits_n.sum() * 0.0, stats

    loss = torch.stack(losses).mean()
    stats.update(
        {
            "loss_proto": float(loss.detach().item()),
            "align_loss": float(torch.stack(align_losses).mean().item()),
            "sep_loss": float(torch.stack(sep_losses).mean().item()),
            "pixel_loss": float(torch.stack(pixel_losses).mean().item()),
            "pixel_fg_loss": float(torch.stack(pixel_fg_losses).mean().item()),
            "pixel_bg_loss": float(torch.stack(pixel_bg_losses).mean().item()),
            "valid_ratio": float(len(losses) / max(batch_size, 1)),
            "sep_active_ratio": float(torch.stack(sep_active_values).mean().item()),
            "cos_fg_view": float(torch.stack(cos_fg_values).mean().item()),
            "cos_bg_view": float(torch.stack(cos_bg_values).mean().item()),
            "cos_fg_bg": float(torch.stack(cos_sep_values).mean().item()),
        }
    )
    return loss, stats


def resize_logits_for_loss(logits, cfg):
    target_size = (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    if tuple(logits.shape[-2:]) == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def use_dabe_aware_loss(cfg):
    return bool(getattr(cfg, "USE_DABE_AWARE_LOSS", False))


def use_dabe_pu_loss(cfg):
    return bool(getattr(cfg, "USE_DABE_PU", False))


def weighted_bce_with_logits(logits, target, weight_map, eps=1e-6):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * weight_map
    return loss.sum() / (weight_map.sum() + float(eps))


def masked_bce_with_logits(logits, target, mask, eps=1e-6):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = mask.float()
    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    return (loss * mask).sum() / (denom + float(eps))


def combine_group_losses(loss_items, eps=1e-6):
    total = None
    weight_sum = 0.0
    zero_ref = None
    for weight, loss_tensor, valid in loss_items:
        if zero_ref is None:
            zero_ref = loss_tensor
        is_valid = bool(valid.detach().item()) if torch.is_tensor(valid) else bool(valid)
        weight = float(weight)
        if is_valid and weight > 0.0:
            total = weight * loss_tensor if total is None else total + weight * loss_tensor
            weight_sum += weight
    if total is None or weight_sum <= 0.0:
        return zero_ref * 0.0
    return total / (weight_sum + float(eps))


def _batch_pu_tensor(batch, key, device):
    return batch[key].to(device, non_blocking=True).float()


def build_rast_region_masks(batch, device):
    fg_core = _batch_pu_tensor(batch, "pu_fg_core", device) > 0.5
    bg_core = _batch_pu_tensor(batch, "pu_bg_core", device) > 0.5
    extent = _batch_pu_tensor(batch, "pu_extent", device) > 0.5
    unknown = _batch_pu_tensor(batch, "pu_unknown", device) > 0.5

    bg_core = bg_core & (~fg_core)
    extent = extent & (~fg_core) & (~bg_core)
    unknown = unknown & (~fg_core) & (~bg_core) & (~extent)
    other = ~(fg_core | bg_core | extent | unknown)
    return {
        "fg_core": fg_core,
        "bg_core": bg_core,
        "extent": extent,
        "unknown": unknown,
        "other": other,
    }


def _rast_mask_mean(value, mask):
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom.detach().item()) <= 0.0:
        return 0.0
    return float((value * mask_f).sum().detach().item() / (denom.detach().item() + 1e-6))


def compute_dino_core_margin_68(cfg, batch, fg_core, bg_core, output_size, device, prefix="ESA"):
    feat = batch["feature"].to(device, non_blocking=True).float()
    if feat.ndim != 4 or feat.shape[1] != 384:
        raise RuntimeError(f"{prefix} expects cached DINO feature [B,384,H,W], got {list(feat.shape)}.")
    feature_size = int(getattr(cfg, f"{prefix}_FEATURE_SIZE", feat.shape[-1]))
    if feat.shape[-2:] != (feature_size, feature_size):
        raise RuntimeError(
            f"{prefix} expects feature spatial {feature_size}x{feature_size}, got {list(feat.shape[-2:])}."
        )

    feat_norm = F.normalize(feat, dim=1)
    fg_core_feat = F.interpolate(fg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5
    bg_core_feat = F.interpolate(bg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5

    margin_feat = torch.zeros(
        (feat.shape[0], 1, feat.shape[-2], feat.shape[-1]),
        device=device,
        dtype=feat.dtype,
    )
    skipped_no_fg = 0
    skipped_no_bg = 0
    for idx in range(int(feat.shape[0])):
        fg_mask = fg_core_feat[idx, 0]
        bg_mask = bg_core_feat[idx, 0]
        if int(fg_mask.sum().detach().item()) <= 0:
            skipped_no_fg += 1
            continue
        if int(bg_mask.sum().detach().item()) <= 0:
            skipped_no_bg += 1
            continue
        fg_proto = F.normalize(feat_norm[idx, :, fg_mask].mean(dim=1), dim=0)
        bg_proto = F.normalize(feat_norm[idx, :, bg_mask].mean(dim=1), dim=0)
        if bool(getattr(cfg, f"{prefix}_DETACH_PROTO", True)):
            fg_proto = fg_proto.detach()
            bg_proto = bg_proto.detach()
        sim_fg = (feat_norm[idx] * fg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        sim_bg = (feat_norm[idx] * bg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        margin_feat[idx] = sim_fg - sim_bg

    margin_68 = F.interpolate(
        margin_feat,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    if bool(getattr(cfg, f"{prefix}_DETACH_MASK", True)):
        margin_68 = margin_68.detach()
    stats = {
        f"{prefix.lower()}_skipped_no_fg_proto": skipped_no_fg,
        f"{prefix.lower()}_skipped_no_bg_proto": skipped_no_bg,
    }
    return margin_68, stats


def build_rast_teacher_weight_map(cfg, batch, teacher_binary, epoch, device):
    masks = build_rast_region_masks(batch, device)
    fg_core = masks["fg_core"]
    bg_core = masks["bg_core"]
    extent = masks["extent"]
    unknown = masks["unknown"]

    teacher_fg = teacher_binary >= 0.5
    teacher_bg = teacher_binary < 0.5
    fg_conflict = fg_core & teacher_bg
    bg_conflict = bg_core & teacher_fg

    teacher_map_region = torch.ones_like(teacher_binary, dtype=torch.float32, device=device)
    teacher_map_region = teacher_map_region * float(getattr(cfg, "RAST_OTHER_TEACHER_MULT", 1.0))
    teacher_map_region = torch.where(
        extent,
        teacher_map_region * float(getattr(cfg, "RAST_EXTENT_TEACHER_MULT", 0.75)),
        teacher_map_region,
    )
    teacher_map_region = torch.where(
        unknown,
        teacher_map_region * float(getattr(cfg, "RAST_UNKNOWN_TEACHER_MULT", 0.25)),
        teacher_map_region,
    )
    conflict_mult = float(getattr(cfg, "RAST_CONFLICT_TEACHER_MULT", 0.20))
    teacher_map_region = torch.where(fg_conflict, teacher_map_region * conflict_mult, teacher_map_region)
    teacher_map_region = torch.where(bg_conflict, teacher_map_region * conflict_mult, teacher_map_region)

    use_esa_asym = bool(getattr(cfg, "USE_ESA_ASYM", False))
    esa_margin_68 = torch.zeros_like(teacher_binary, dtype=torch.float32, device=device)
    esa_margin_stats = {
        "esa_skipped_no_fg_proto": 0,
        "esa_skipped_no_bg_proto": 0,
    }
    extent_teacher_fg = extent & teacher_fg
    extent_teacher_bg = extent & teacher_bg
    extent_teacher_bg_fg_like = torch.zeros_like(extent_teacher_bg)
    extent_teacher_bg_bg_like = torch.zeros_like(extent_teacher_bg)
    extent_teacher_bg_ambig = torch.zeros_like(extent_teacher_bg)
    if use_esa_asym:
        if not bool(getattr(cfg, "ESA_USE_DINO_MARGIN", True)):
            raise RuntimeError("USE_ESA_ASYM=True currently requires ESA_USE_DINO_MARGIN=True.")
        esa_margin_68, esa_margin_stats = compute_dino_core_margin_68(
            cfg,
            batch,
            fg_core,
            bg_core,
            teacher_binary.shape[-2:],
            device,
            prefix="ESA",
        )
        margin_fg_like = float(getattr(cfg, "ESA_MARGIN_FG_LIKE", 0.05))
        margin_bg_like = float(getattr(cfg, "ESA_MARGIN_BG_LIKE", -0.05))
        extent_teacher_bg_fg_like = extent_teacher_bg & (esa_margin_68 >= margin_fg_like)
        extent_teacher_bg_bg_like = extent_teacher_bg & (esa_margin_68 <= margin_bg_like)
        extent_teacher_bg_ambig = extent_teacher_bg & (~extent_teacher_bg_fg_like) & (~extent_teacher_bg_bg_like)

        teacher_map_esa = torch.ones_like(teacher_binary, dtype=torch.float32, device=device)
        teacher_map_esa = teacher_map_esa * float(getattr(cfg, "RAST_OTHER_TEACHER_MULT", 1.0))
        unknown_mult = float(getattr(cfg, "RAST_UNKNOWN_TEACHER_MULT", 0.50))
        if bool(getattr(cfg, "ESA_TOUCH_UNKNOWN", False)):
            unknown_mult = float(getattr(cfg, "ESA_UNKNOWN_TEACHER_MULT", unknown_mult))
        teacher_map_esa = torch.where(unknown, teacher_map_esa * unknown_mult, teacher_map_esa)
        teacher_map_esa = torch.where(
            extent_teacher_fg,
            teacher_map_esa * float(getattr(cfg, "ESA_EXTENT_TEACHER_FG_MULT", 1.0)),
            teacher_map_esa,
        )
        teacher_map_esa = torch.where(
            extent_teacher_bg_fg_like,
            teacher_map_esa * float(getattr(cfg, "ESA_EXTENT_TEACHER_BG_FG_LIKE_MULT", 0.25)),
            teacher_map_esa,
        )
        teacher_map_esa = torch.where(
            extent_teacher_bg_ambig,
            teacher_map_esa * float(getattr(cfg, "ESA_EXTENT_TEACHER_BG_AMBIG_MULT", 0.50)),
            teacher_map_esa,
        )
        teacher_map_esa = torch.where(
            extent_teacher_bg_bg_like,
            teacher_map_esa * float(getattr(cfg, "ESA_EXTENT_TEACHER_BG_BG_LIKE_MULT", 1.0)),
            teacher_map_esa,
        )
        teacher_map_esa = torch.where(fg_conflict, teacher_map_esa * conflict_mult, teacher_map_esa)
        teacher_map_esa = torch.where(bg_conflict, teacher_map_esa * conflict_mult, teacher_map_esa)
        teacher_map_region = teacher_map_esa

    teacher_map_post = torch.ones_like(teacher_binary, dtype=torch.float32, device=device)
    post_conflict_mult = float(getattr(cfg, "RAST_POST_RESET_CONFLICT_TEACHER_MULT", 0.30))
    teacher_map_post = torch.where(fg_conflict, teacher_map_post * post_conflict_mult, teacher_map_post)
    teacher_map_post = torch.where(bg_conflict, teacher_map_post * post_conflict_mult, teacher_map_post)

    ones = torch.ones_like(teacher_map_region)
    rast_pre_reset_scale = float(get_rast_pre_reset_scale(cfg, epoch))
    rast_post_reset_scale = float(get_rast_post_reset_scale(cfg, epoch))
    if rast_pre_reset_scale > 0.0:
        rast_scale_effective = rast_pre_reset_scale
        rast_phase = "pre_reset"
        teacher_map_eff = (1.0 - rast_pre_reset_scale) * ones + rast_pre_reset_scale * teacher_map_region
    elif rast_post_reset_scale > 0.0:
        rast_scale_effective = rast_post_reset_scale
        rast_phase = "post_reset_conflict_only"
        teacher_map_eff = (1.0 - rast_post_reset_scale) * ones + rast_post_reset_scale * teacher_map_post
    else:
        rast_scale_effective = 0.0
        rast_phase = "off"
        teacher_map_eff = ones

    eps = 1e-6
    fg_area = float(fg_core.float().mean().detach().item())
    bg_area = float(bg_core.float().mean().detach().item())
    extent_area = float(extent.float().mean().detach().item())
    unknown_area = float(unknown.float().mean().detach().item())
    fg_conflict_ratio = float(
        fg_conflict.float().sum().detach().item() / (fg_core.float().sum().detach().item() + eps)
    )
    bg_conflict_ratio = float(
        bg_conflict.float().sum().detach().item() / (bg_core.float().sum().detach().item() + eps)
    )
    stats = {
        "rast_scale": rast_scale_effective,
        "rast_pre_reset_scale": rast_pre_reset_scale,
        "rast_post_reset_scale": rast_post_reset_scale,
        "rast_scale_effective": rast_scale_effective,
        "rast_phase": rast_phase,
        "rast_post_reset_enable": bool(getattr(cfg, "RAST_POST_RESET_ENABLE", False)),
        "rast_post_reset_conflict_only": bool(getattr(cfg, "RAST_POST_RESET_CONFLICT_ONLY", False)),
        "fg_core_area": fg_area,
        "bg_core_area": bg_area,
        "extent_area": extent_area,
        "unknown_area": unknown_area,
        "fg_conflict_ratio": fg_conflict_ratio,
        "bg_conflict_ratio": bg_conflict_ratio,
        "teacher_map_mean": float(teacher_map_eff.mean().detach().item()),
        "teacher_map_min": float(teacher_map_eff.min().detach().item()),
        "teacher_map_max": float(teacher_map_eff.max().detach().item()),
        "teacher_map_fg_core_mean": _rast_mask_mean(teacher_map_eff, fg_core),
        "teacher_map_bg_core_mean": _rast_mask_mean(teacher_map_eff, bg_core),
        "teacher_map_extent_mean": _rast_mask_mean(teacher_map_eff, extent),
        "teacher_map_unknown_mean": _rast_mask_mean(teacher_map_eff, unknown),
        "esa_asym_enable": use_esa_asym,
        "esa_asym_scale": float(get_esa_asym_scale(cfg, epoch)),
        "esa_margin_shape": list(esa_margin_68.shape),
        "esa_margin_mean": float(esa_margin_68.mean().detach().item()),
        "esa_margin_min": float(esa_margin_68.min().detach().item()),
        "esa_margin_max": float(esa_margin_68.max().detach().item()),
        "esa_margin_extent_mean": _rast_mask_mean(esa_margin_68, extent),
        "esa_margin_extent_teacher_fg_mean": _rast_mask_mean(esa_margin_68, extent_teacher_fg),
        "esa_margin_extent_teacher_bg_mean": _rast_mask_mean(esa_margin_68, extent_teacher_bg),
        "esa_extent_teacher_fg_ratio": float(extent_teacher_fg.float().mean().detach().item()),
        "esa_extent_teacher_bg_ratio": float(extent_teacher_bg.float().mean().detach().item()),
        "esa_extent_teacher_bg_fg_like_ratio": float(extent_teacher_bg_fg_like.float().mean().detach().item()),
        "esa_extent_teacher_bg_ambig_ratio": float(extent_teacher_bg_ambig.float().mean().detach().item()),
        "esa_extent_teacher_bg_bg_like_ratio": float(extent_teacher_bg_bg_like.float().mean().detach().item()),
        "esa_teacher_map_extent_mean": _rast_mask_mean(teacher_map_eff, extent),
        "esa_teacher_map_extent_teacher_fg_mean": _rast_mask_mean(teacher_map_eff, extent_teacher_fg),
        "esa_teacher_map_extent_teacher_bg_mean": _rast_mask_mean(teacher_map_eff, extent_teacher_bg),
        "esa_teacher_map_extent_teacher_bg_fg_like_mean": _rast_mask_mean(
            teacher_map_eff, extent_teacher_bg_fg_like
        ),
        "esa_teacher_map_extent_teacher_bg_ambig_mean": _rast_mask_mean(
            teacher_map_eff, extent_teacher_bg_ambig
        ),
        "esa_teacher_map_extent_teacher_bg_bg_like_mean": _rast_mask_mean(
            teacher_map_eff, extent_teacher_bg_bg_like
        ),
        "esa_skipped_no_fg_proto": int(esa_margin_stats.get("esa_skipped_no_fg_proto", 0)),
        "esa_skipped_no_bg_proto": int(esa_margin_stats.get("esa_skipped_no_bg_proto", 0)),
    }
    return teacher_map_eff, stats


def rast_teacher_bce_with_logits(logits, target, teacher_map_eff, cfg, rast_scale, apply_to_loss=True, eps=1e-6):
    if (
        not bool(getattr(cfg, "USE_RAST", False))
        or not bool(apply_to_loss)
        or teacher_map_eff is None
        or float(rast_scale) <= 0.0
    ):
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    if bool(getattr(cfg, "RAST_WEIGHTED_LOSS_NORMALIZE", True)):
        return weighted_bce_with_logits(
            logits,
            target,
            teacher_map_eff.to(device=logits.device, dtype=logits.dtype),
            eps=eps,
        )
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * teacher_map_eff.to(device=logits.device, dtype=logits.dtype)).mean()


def get_hbns_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_HBNS_LITE", False)):
        return 0.0
    start = int(getattr(cfg, "HBNS_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "HBNS_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "HBNS_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_epr_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_EPR_POS", False)):
        return 0.0
    start = int(getattr(cfg, "EPR_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "EPR_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "EPR_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def cap_mask_by_score_per_image(mask, score, max_ratio, min_pixels):
    capped = torch.zeros_like(mask, dtype=torch.bool)
    batch_size = int(mask.shape[0])
    num_pixels = int(mask.shape[-2] * mask.shape[-1])
    max_pixels = max(1, int(float(max_ratio) * float(num_pixels)))
    min_pixels = max(0, int(min_pixels))
    for idx in range(batch_size):
        flat_mask = mask[idx].flatten()
        candidate_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
        num_candidates = int(candidate_idx.numel())
        if num_candidates < min_pixels or num_candidates <= 0:
            continue
        if num_candidates <= max_pixels:
            keep_idx = candidate_idx
        else:
            flat_score = score[idx].flatten()
            topk = torch.topk(flat_score.index_select(0, candidate_idx), k=max_pixels, largest=True)
            keep_idx = candidate_idx[topk.indices]
        capped[idx].flatten().index_fill_(0, keep_idx, True)
    return capped


def build_hbns_hard_bg_mask(cfg, batch, logits, teacher_binary, region_masks, device):
    student_prob_source = logits.detach() if bool(getattr(cfg, "HBNS_DETACH_MASK", True)) else logits
    student_prob = torch.sigmoid(student_prob_source)
    student_fg = student_prob > float(getattr(cfg, "HBNS_STUDENT_PROB_THRESH", 0.50))

    pu_target_soft = _batch_pu_tensor(batch, "pu_target_soft", device)
    pu_weight_map = _batch_pu_tensor(batch, "pu_weight_map", device)
    bg_core = region_masks["bg_core"]
    unknown = region_masks["unknown"]

    bg_core_region = bg_core if bool(getattr(cfg, "HBNS_USE_BG_CORE", True)) else torch.zeros_like(bg_core)
    low_target_bg = (
        (pu_target_soft < float(getattr(cfg, "HBNS_LOW_TARGET_THRESH", 0.15)))
        & (pu_weight_map > float(getattr(cfg, "HBNS_LOW_TARGET_WEIGHT_THRESH", 0.50)))
    )
    if not bool(getattr(cfg, "HBNS_USE_LOW_TARGET_BG", True)):
        low_target_bg = torch.zeros_like(bg_core)
    unknown_bg_like = unknown & (teacher_binary < 0.5) & student_fg
    if not bool(getattr(cfg, "HBNS_USE_UNKNOWN_BG_LIKE", True)):
        unknown_bg_like = torch.zeros_like(bg_core)

    hard_bg_raw = student_fg & (bg_core_region | low_target_bg | unknown_bg_like)
    hard_bg = cap_mask_by_score_per_image(
        hard_bg_raw,
        student_prob.detach(),
        float(getattr(cfg, "HBNS_MAX_RATIO_PER_IMAGE", 0.05)),
        int(getattr(cfg, "HBNS_MIN_PIXELS_PER_IMAGE", 8)),
    )

    num_pixels = float(hard_bg.shape[-2] * hard_bg.shape[-1])
    raw_pixels_per_image = hard_bg_raw.float().flatten(1).sum(dim=1)
    capped_pixels_per_image = hard_bg.float().flatten(1).sum(dim=1)
    stats = {
        "hard_bg_ratio": float(hard_bg.float().mean().detach().item()),
        "hard_bg_raw_ratio": float(hard_bg_raw.float().mean().detach().item()),
        "hard_bg_ratio_bg_core": float((student_fg & bg_core_region).float().mean().detach().item()),
        "hard_bg_ratio_low_target": float((student_fg & low_target_bg).float().mean().detach().item()),
        "hard_bg_ratio_unknown_bg_like": float(unknown_bg_like.float().mean().detach().item()),
        "hard_bg_pixels_mean": float(capped_pixels_per_image.mean().detach().item()),
        "hard_bg_raw_pixels_mean": float(raw_pixels_per_image.mean().detach().item()),
        "hard_bg_skipped_images": int((capped_pixels_per_image <= 0).sum().detach().item()),
        "hard_bg_max_pixels_per_image": float(
            max(1, int(float(getattr(cfg, "HBNS_MAX_RATIO_PER_IMAGE", 0.05)) * num_pixels))
        ),
    }
    return hard_bg, stats


def build_epr_pos_mask(cfg, batch, region_masks, teacher_prob, teacher_binary, device):
    extent = region_masks["extent"]
    fg_core = region_masks["fg_core"]
    bg_core = region_masks["bg_core"]
    unknown = region_masks["unknown"]
    if str(getattr(cfg, "EPR_REGION", "extent")).lower() != "extent":
        raise RuntimeError("EPR_REGION currently supports only 'extent'.")
    if bool(getattr(cfg, "EPR_USE_UNKNOWN", False)):
        raise RuntimeError("EPR_USE_UNKNOWN=True is not supported for EPR-pos-lite.")
    if not bool(getattr(cfg, "EPR_POSITIVE_ONLY", True)):
        raise RuntimeError("EPR_POSITIVE_ONLY must be True.")
    if str(getattr(cfg, "EPR_FEATURE_SOURCE", "cached_dino")).lower() != "cached_dino":
        raise RuntimeError("EPR_FEATURE_SOURCE currently supports only 'cached_dino'.")

    feat = batch["feature"].to(device, non_blocking=True).float()
    if feat.ndim != 4 or feat.shape[1] != 384:
        raise RuntimeError(f"EPR expects cached DINO feature [B,384,H,W], got {list(feat.shape)}.")
    feature_size = int(getattr(cfg, "EPR_FEATURE_SIZE", feat.shape[-1]))
    if feat.shape[-2:] != (feature_size, feature_size):
        raise RuntimeError(
            f"EPR expects feature spatial {feature_size}x{feature_size}, got {list(feat.shape[-2:])}."
        )
    loss_size = int(getattr(cfg, "EPR_LOSS_SIZE", teacher_prob.shape[-1]))
    if teacher_prob.shape[-2:] != (loss_size, loss_size):
        raise RuntimeError(
            f"EPR expects teacher/logit spatial {loss_size}x{loss_size}, got {list(teacher_prob.shape[-2:])}."
        )

    feat_norm = F.normalize(feat, dim=1)
    fg_core_37 = F.interpolate(fg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5
    bg_core_37 = F.interpolate(bg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5

    margin_37 = torch.zeros(
        (feat.shape[0], 1, feat.shape[-2], feat.shape[-1]),
        device=device,
        dtype=feat.dtype,
    )
    min_pixels = int(getattr(cfg, "EPR_MIN_PIXELS_PER_IMAGE", 8))
    skipped_no_fg = 0
    skipped_no_bg = 0
    for idx in range(int(feat.shape[0])):
        fg_mask = fg_core_37[idx, 0]
        bg_mask = bg_core_37[idx, 0]
        fg_count = int(fg_mask.sum().detach().item())
        bg_count = int(bg_mask.sum().detach().item())
        if fg_count < min_pixels:
            skipped_no_fg += 1
            continue
        if bg_count < min_pixels:
            skipped_no_bg += 1
            continue
        fg_pixels = feat_norm[idx, :, fg_mask]
        bg_pixels = feat_norm[idx, :, bg_mask]
        fg_proto = F.normalize(fg_pixels.mean(dim=1), dim=0)
        bg_proto = F.normalize(bg_pixels.mean(dim=1), dim=0)
        if bool(getattr(cfg, "EPR_DETACH_PROTO", True)):
            fg_proto = fg_proto.detach()
            bg_proto = bg_proto.detach()
        sim_fg = (feat_norm[idx] * fg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        sim_bg = (feat_norm[idx] * bg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        margin_37[idx] = sim_fg - sim_bg

    margin_68 = F.interpolate(
        margin_37,
        size=teacher_prob.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    if bool(getattr(cfg, "EPR_DETACH_MASK", True)):
        margin_68 = margin_68.detach()
    teacher_prob_detached = teacher_prob.detach()
    teacher_binary_detached = teacher_binary.detach()
    teacher_conf = 2.0 * torch.abs(teacher_prob_detached - 0.5)
    teacher_fg_cond = teacher_binary_detached >= 0.5
    if not bool(getattr(cfg, "EPR_REQUIRE_TEACHER_FG", True)):
        teacher_fg_cond = torch.ones_like(teacher_fg_cond, dtype=torch.bool)
    teacher_conf_cond = teacher_conf >= float(getattr(cfg, "EPR_TEACHER_CONF_THRESH", 0.75))
    if bool(getattr(cfg, "EPR_USE_DINO_PROTO_MARGIN", True)):
        margin_cond = margin_68 >= float(getattr(cfg, "EPR_MARGIN_THRESH", 0.05))
    else:
        margin_cond = torch.ones_like(extent, dtype=torch.bool)
    raw_mask = extent & teacher_fg_cond & teacher_conf_cond & margin_cond
    raw_mask = raw_mask & (~fg_core) & (~bg_core) & (~unknown)
    score = teacher_conf + margin_68
    epr_pos_mask = cap_mask_by_score_per_image(
        raw_mask,
        score.detach(),
        float(getattr(cfg, "EPR_MAX_RATIO_PER_IMAGE", 0.03)),
        min_pixels,
    )

    raw_pixels = raw_mask.float().flatten(1).sum(dim=1)
    capped_pixels = epr_pos_mask.float().flatten(1).sum(dim=1)
    pos_count = float(epr_pos_mask.float().sum().detach().item())
    raw_count = float(raw_mask.float().sum().detach().item())
    margin_pos_mean = _rast_mask_mean(margin_68, epr_pos_mask) if pos_count > 0.0 else 0.0
    teacher_conf_pos_mean = _rast_mask_mean(teacher_conf, epr_pos_mask) if pos_count > 0.0 else 0.0
    stats = {
        "epr_pos_ratio": float(epr_pos_mask.float().mean().detach().item()),
        "epr_pos_raw_ratio": float(raw_mask.float().mean().detach().item()),
        "epr_pos_pixels_mean": float(capped_pixels.mean().detach().item()),
        "epr_pos_raw_pixels_mean": float(raw_pixels.mean().detach().item()),
        "epr_valid_image_ratio": float((capped_pixels > 0).float().mean().detach().item()),
        "epr_margin_mean": float(margin_68.mean().detach().item()),
        "epr_margin_min": float(margin_68.min().detach().item()),
        "epr_margin_max": float(margin_68.max().detach().item()),
        "epr_margin_pos_mean": margin_pos_mean,
        "epr_teacher_conf_pos_mean": teacher_conf_pos_mean,
        "epr_extent_area": float(extent.float().mean().detach().item()),
        "epr_unknown_overlap_ratio": float((epr_pos_mask & unknown).float().mean().detach().item()),
        "epr_skipped_no_fg_proto": skipped_no_fg,
        "epr_skipped_no_bg_proto": skipped_no_bg,
        "epr_max_pixels_per_image": float(
            max(1, int(float(getattr(cfg, "EPR_MAX_RATIO_PER_IMAGE", 0.03)) * float(extent.shape[-2] * extent.shape[-1])))
        ),
        "epr_raw_count": raw_count,
        "epr_pos_count": pos_count,
    }
    return epr_pos_mask, margin_68, stats


def hbns_lite_loss_for_logits(cfg, logits, hard_bg_mask):
    if hard_bg_mask is None or float(hard_bg_mask.float().sum().detach().item()) <= 0.0:
        return logits.sum() * 0.0
    target_bg = torch.zeros_like(logits)
    loss_map = F.binary_cross_entropy_with_logits(logits, target_bg, reduction="none")
    weight = hard_bg_mask.to(device=logits.device, dtype=logits.dtype)
    if bool(getattr(cfg, "HBNS_WEIGHTED_NORMALIZE", True)):
        return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)
    return (loss_map * weight).mean()


def epr_pos_loss_for_logits(cfg, logits, epr_pos_mask):
    if epr_pos_mask is None or float(epr_pos_mask.float().sum().detach().item()) <= 0.0:
        return logits.sum() * 0.0
    target_pos = torch.ones_like(logits)
    loss_map = F.binary_cross_entropy_with_logits(logits, target_pos, reduction="none")
    weight = epr_pos_mask.to(device=logits.device, dtype=logits.dtype)
    if bool(getattr(cfg, "EPR_WEIGHTED_NORMALIZE", True)):
        return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)
    return (loss_map * weight).mean()


def compute_esa_region_diagnostics(batch, region_masks, student_logits, teacher_prob, teacher_binary):
    student_prob = torch.sigmoid(student_logits.detach())
    teacher_prob_detached = teacher_prob.detach()
    teacher_binary_detached = teacher_binary.detach()
    teacher_conf = 2.0 * torch.abs(teacher_prob_detached - 0.5)
    teacher_loss_map = F.binary_cross_entropy_with_logits(
        student_logits.detach(),
        teacher_binary_detached,
        reduction="none",
    )
    stats = {}
    for name in ("fg_core", "bg_core", "extent", "unknown"):
        mask = region_masks[name]
        stats[f"student_prob_{name}"] = _masked_mean_for_log(student_prob, mask)
        stats[f"teacher_fg_{name}"] = _masked_mean_for_log(teacher_binary_detached, mask)
        stats[f"teacher_conf_{name}"] = _masked_mean_for_log(teacher_conf, mask)
        stats[f"teacher_loss_{name}"] = _masked_mean_for_log(teacher_loss_map, mask)
        stats[f"area_{name}"] = float(mask.float().mean().detach().item())
    return stats


def build_pu_static_group_loss(logits, batch, cfg):
    device = logits.device
    fg_mask = (_batch_pu_tensor(batch, "pu_fg_core", device) > 0.5).float()
    fg_fallback_mask = (_batch_pu_tensor(batch, "pu_fg_fallback", device) > 0.5).float()
    bg_mask = (_batch_pu_tensor(batch, "pu_bg_core", device) > 0.5).float()
    extent_mask = (_batch_pu_tensor(batch, "pu_extent", device) > 1e-6).float()
    unknown_mask = (_batch_pu_tensor(batch, "pu_unknown", device) > 0.5).float()

    fg_fallback_mask = fg_fallback_mask * (1.0 - fg_mask)
    bg_mask = bg_mask * (1.0 - fg_mask) * (1.0 - fg_fallback_mask)
    extent_mask = extent_mask * (1.0 - fg_mask) * (1.0 - fg_fallback_mask) * (1.0 - bg_mask)
    unknown_mask = unknown_mask * (1.0 - fg_mask) * (1.0 - fg_fallback_mask) * (1.0 - bg_mask) * (1.0 - extent_mask)

    loss_fg = masked_bce_with_logits(
        logits,
        torch.ones_like(logits),
        fg_mask,
        eps=float(getattr(cfg, "PU_STATIC_GROUP_EPS", 1e-6)),
    )
    loss_fg_fallback = masked_bce_with_logits(
        logits,
        torch.full_like(logits, 0.85),
        fg_fallback_mask,
        eps=float(getattr(cfg, "PU_STATIC_GROUP_EPS", 1e-6)),
    )
    loss_bg = masked_bce_with_logits(
        logits,
        torch.zeros_like(logits),
        bg_mask,
        eps=float(getattr(cfg, "PU_STATIC_GROUP_EPS", 1e-6)),
    )
    loss_extent = masked_bce_with_logits(
        logits,
        torch.full_like(logits, 0.5),
        extent_mask,
        eps=float(getattr(cfg, "PU_STATIC_GROUP_EPS", 1e-6)),
    )
    loss_static = combine_group_losses(
        [
            (float(getattr(cfg, "PU_STATIC_LAMBDA_FG", 1.0)), loss_fg, fg_mask.sum() > 0),
            (float(getattr(cfg, "PU_STATIC_LAMBDA_FG_FALLBACK", 0.35)), loss_fg_fallback, fg_fallback_mask.sum() > 0),
            (float(getattr(cfg, "PU_STATIC_LAMBDA_BG", 0.50)), loss_bg, bg_mask.sum() > 0),
            (float(getattr(cfg, "PU_STATIC_LAMBDA_EXTENT", 0.10)), loss_extent, extent_mask.sum() > 0),
            (float(getattr(cfg, "PU_STATIC_LAMBDA_UNKNOWN", 0.0)), logits.sum() * 0.0, unknown_mask.sum() > 0),
        ],
        eps=float(getattr(cfg, "PU_STATIC_GROUP_EPS", 1e-6)),
    )
    stats = {
        "loss_static_fg": float(loss_fg.detach().item()),
        "loss_static_fg_fallback": float(loss_fg_fallback.detach().item()),
        "loss_static_bg": float(loss_bg.detach().item()),
        "loss_static_extent": float(loss_extent.detach().item()),
        "fg_core_area": float(fg_mask.mean().detach().item()),
        "fg_fallback_area": float(fg_fallback_mask.mean().detach().item()),
        "bg_core_area": float(bg_mask.mean().detach().item()),
        "extent_area": float(extent_mask.mean().detach().item()),
        "unknown_area": float(unknown_mask.mean().detach().item()),
    }
    return loss_static, stats


def build_dabe_pu_teacher_conf_target_weight(cfg, teacher_prob, pu_fg_core, pu_bg_core):
    teacher_fg = teacher_prob >= float(getattr(cfg, "TEACHER_CONF_FG_THRESH", 0.75))
    teacher_bg = teacher_prob <= float(getattr(cfg, "TEACHER_CONF_BG_THRESH", 0.25))
    if bool(getattr(cfg, "TEACHER_CONF_IGNORE_PU_CORE", True)):
        core_mask = (pu_fg_core > 0.5) | (pu_bg_core > 0.5)
        teacher_fg = teacher_fg & (~core_mask)
        teacher_bg = teacher_bg & (~core_mask)
    conf_mask = teacher_fg | teacher_bg
    valid_weight = torch.ones_like(teacher_prob).clamp(
        min=float(getattr(cfg, "TEACHER_CONF_WEIGHT_MIN", 0.0)),
        max=float(getattr(cfg, "TEACHER_CONF_WEIGHT_MAX", 1.0)),
    )
    weight_map = torch.where(conf_mask, valid_weight, torch.zeros_like(valid_weight))
    target = teacher_fg.float()
    stats = {
        "teacher_conf_ratio": float(conf_mask.float().mean().item()),
        "teacher_fg_ratio": float(teacher_fg.float().mean().item()),
        "teacher_bg_ratio": float(teacher_bg.float().mean().item()),
    }
    return target, weight_map, stats


def _cap_teacher_bg_mask(teacher_bg_mask, teacher_fg_mask, teacher_prob, cfg):
    capped = torch.zeros_like(teacher_bg_mask, dtype=torch.bool)
    batch_size = int(teacher_bg_mask.shape[0])
    num_pixels = int(teacher_bg_mask.shape[-2] * teacher_bg_mask.shape[-1])
    ratio_cap = int(float(getattr(cfg, "TEACHER_CONF_BG_MAX_RATIO", 0.15)) * num_pixels)
    bg_min_pixels = int(getattr(cfg, "TEACHER_CONF_BG_MIN_PIXELS", 128))
    no_fg_ratio = float(getattr(cfg, "TEACHER_CONF_BG_CAP_IF_NO_FG_RATIO", 0.05))
    bg_to_fg_max = float(getattr(cfg, "TEACHER_CONF_BG_TO_FG_MAX", 5.0))
    for idx in range(batch_size):
        bg_flat = teacher_bg_mask[idx].flatten()
        fg_count = int(teacher_fg_mask[idx].sum().item())
        bg_count = int(bg_flat.sum().item())
        if fg_count > 0:
            fg_cap = int(bg_to_fg_max * fg_count)
            max_bg = min(ratio_cap, max(fg_cap, bg_min_pixels))
        else:
            max_bg = int(no_fg_ratio * num_pixels)
        max_bg = max(0, min(int(max_bg), bg_count))
        if max_bg <= 0:
            continue
        if bg_count <= max_bg:
            capped[idx] = teacher_bg_mask[idx]
            continue
        bg_indices = torch.nonzero(bg_flat, as_tuple=False).flatten()
        bg_scores = (1.0 - teacher_prob[idx].flatten())[bg_indices]
        keep_indices = bg_indices[torch.topk(bg_scores, k=max_bg, largest=True).indices]
        capped_flat = torch.zeros_like(bg_flat, dtype=torch.bool)
        capped_flat[keep_indices] = True
        capped[idx] = capped_flat.view_as(teacher_bg_mask[idx])
    return capped


def build_teacher_conf_balanced_loss(logits, teacher_prob, batch, cfg):
    device = logits.device
    teacher_fg_mask = teacher_prob >= float(getattr(cfg, "TEACHER_CONF_FG_THRESH", 0.70))
    teacher_bg_mask = teacher_prob <= float(getattr(cfg, "TEACHER_CONF_BG_THRESH", 0.20))
    if bool(getattr(cfg, "TEACHER_CONF_IGNORE_PU_CORE", True)):
        pu_core = (
            (_batch_pu_tensor(batch, "pu_fg_core", device) > 0.5)
            | (_batch_pu_tensor(batch, "pu_fg_fallback", device) > 0.5)
            | (_batch_pu_tensor(batch, "pu_bg_core", device) > 0.5)
        )
        teacher_fg_mask = teacher_fg_mask & (~pu_core)
        teacher_bg_mask = teacher_bg_mask & (~pu_core)
    teacher_bg_mask_capped = _cap_teacher_bg_mask(teacher_bg_mask, teacher_fg_mask, teacher_prob, cfg)

    loss_teacher_fg = masked_bce_with_logits(
        logits,
        torch.ones_like(logits),
        teacher_fg_mask.float(),
        eps=float(getattr(cfg, "TEACHER_CONF_GROUP_EPS", 1e-6)),
    )
    loss_teacher_bg = masked_bce_with_logits(
        logits,
        torch.zeros_like(logits),
        teacher_bg_mask_capped.float(),
        eps=float(getattr(cfg, "TEACHER_CONF_GROUP_EPS", 1e-6)),
    )
    loss_teacher = combine_group_losses(
        [
            (float(getattr(cfg, "TEACHER_CONF_LAMBDA_FG", 1.0)), loss_teacher_fg, teacher_fg_mask.sum() > 0),
            (float(getattr(cfg, "TEACHER_CONF_LAMBDA_BG", 0.30)), loss_teacher_bg, teacher_bg_mask_capped.sum() > 0),
        ],
        eps=float(getattr(cfg, "TEACHER_CONF_GROUP_EPS", 1e-6)),
    )
    stats = {
        "teacher_fg_ratio_raw": float(teacher_fg_mask.float().mean().detach().item()),
        "teacher_bg_ratio_raw": float(teacher_bg_mask.float().mean().detach().item()),
        "teacher_bg_ratio_capped": float(teacher_bg_mask_capped.float().mean().detach().item()),
        "teacher_conf_ratio_capped": float((teacher_fg_mask | teacher_bg_mask_capped).float().mean().detach().item()),
        "loss_teacher_fg": float(loss_teacher_fg.detach().item()),
        "loss_teacher_bg": float(loss_teacher_bg.detach().item()),
    }
    return loss_teacher, stats


def build_dabe_oem_seed_loss(logits, batch, cfg):
    device = logits.device
    fg_mask = (_batch_pu_tensor(batch, "pu_fg_core", device) > 0.5).float()
    fg_fallback_mask = (_batch_pu_tensor(batch, "pu_fg_fallback", device) > 0.5).float()
    bg_mask = (_batch_pu_tensor(batch, "pu_bg_core", device) > 0.5).float()

    fg_fallback_mask = fg_fallback_mask * (1.0 - fg_mask)
    bg_mask = bg_mask * (1.0 - fg_mask) * (1.0 - fg_fallback_mask)

    eps = float(getattr(cfg, "OEM_SEED_GROUP_EPS", 1e-6))
    loss_fg = masked_bce_with_logits(logits, torch.ones_like(logits), fg_mask, eps=eps)
    loss_fg_fallback = masked_bce_with_logits(
        logits,
        torch.full_like(logits, 0.85),
        fg_fallback_mask,
        eps=eps,
    )
    loss_bg = masked_bce_with_logits(logits, torch.zeros_like(logits), bg_mask, eps=eps)
    loss_seed = combine_group_losses(
        [
            (float(getattr(cfg, "OEM_SEED_LAMBDA_FG", 1.0)), loss_fg, fg_mask.sum() > 0),
            (
                float(getattr(cfg, "OEM_SEED_LAMBDA_FG_FALLBACK", 0.35)),
                loss_fg_fallback,
                fg_fallback_mask.sum() > 0,
            ),
            (float(getattr(cfg, "OEM_SEED_LAMBDA_BG", 1.0)), loss_bg, bg_mask.sum() > 0),
        ],
        eps=eps,
    )
    stats = {
        "loss_seed_fg": float(loss_fg.detach().item()),
        "loss_seed_fg_fallback": float(loss_fg_fallback.detach().item()),
        "loss_seed_bg": float(loss_bg.detach().item()),
        "seed_fg_area": float(fg_mask.detach().mean().item()),
        "seed_fg_fallback_area": float(fg_fallback_mask.detach().mean().item()),
        "seed_bg_area": float(bg_mask.detach().mean().item()),
    }
    return loss_seed, stats


def _normalize_map_01(values):
    min_value = values.min()
    max_value = values.max()
    denom = (max_value - min_value).clamp_min(1e-6)
    return (values - min_value) / denom


def _masked_mean_for_log(values, mask):
    mask = mask.float()
    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        return 0.0
    return float(((values * mask).sum() / denom).detach().item())


def _select_topk_mask(candidate, score, k):
    selected = torch.zeros_like(candidate, dtype=torch.bool)
    idx = torch.nonzero(candidate.flatten(), as_tuple=False).flatten()
    if idx.numel() == 0 or int(k) <= 0:
        return selected
    k = min(int(k), int(idx.numel()))
    keep = idx[torch.topk(score.flatten().index_select(0, idx), k=k, largest=True).indices]
    selected_flat = selected.flatten()
    selected_flat[keep] = True
    return selected_flat.view_as(candidate)


def _empty_oem_masks(feature_37, teacher_prob_68, teacher_prob_37):
    b, _, h, w = feature_37.shape
    device = feature_37.device
    mask37 = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
    mask68 = torch.zeros_like(teacher_prob_68)
    proto_delta = torch.zeros((b, 1, h, w), device=device, dtype=torch.float32)
    return {
        "pos_mask_37": mask37,
        "bg_mask_37": mask37.clone(),
        "pos_mask_68": mask68,
        "bg_mask_68": mask68.clone(),
        "pos_target_68": teacher_prob_68,
        "bg_target_68": torch.zeros_like(teacher_prob_68),
        "proto_delta_37": proto_delta,
        "teacher_prob_37": teacher_prob_37,
    }


def build_dabe_oem_dynamic_masks(feature_37, teacher_logits_68, batch, cfg, epoch):
    if str(getattr(cfg, "OEM_PROTO_FEATURE_SOURCE", "dino37")).lower() != "dino37":
        raise RuntimeError("DABE-OEM currently supports OEM_PROTO_FEATURE_SOURCE='dino37' only.")

    feature_37 = feature_37.float()
    teacher_prob_68 = torch.sigmoid(teacher_logits_68).detach()
    teacher_prob_37 = F.interpolate(
        teacher_prob_68,
        size=feature_37.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).detach()

    fg_core_37 = (_batch_pu_tensor(batch, "pu_fg_core_37", feature_37.device) > 0.5)
    fg_fallback_37 = (_batch_pu_tensor(batch, "pu_fg_fallback_37", feature_37.device) > 0.5)
    bg_core_37 = (_batch_pu_tensor(batch, "pu_bg_core_37", feature_37.device) > 0.5)
    extent_37 = (_batch_pu_tensor(batch, "pu_extent_37", feature_37.device) > 1e-6)
    unknown_37 = (_batch_pu_tensor(batch, "pu_unknown_37", feature_37.device) > 0.5)
    fg_seed_37 = fg_core_37 | fg_fallback_37
    bg_seed_37 = bg_core_37

    masks = _empty_oem_masks(feature_37, teacher_prob_68, teacher_prob_37)
    lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
    base_stats = {
        "oem_pos_raw_ratio": 0.0,
        "oem_pos_capped_ratio": 0.0,
        "oem_bg_raw_ratio": 0.0,
        "oem_bg_capped_ratio": 0.0,
        "oem_skip_no_fg_proto": 0,
        "oem_skip_no_bg_proto": 0,
        "oem_skip_no_pos_region": 0,
        "proto_delta_mean": 0.0,
        "proto_delta_min": 0.0,
        "proto_delta_max": 0.0,
        "teacher_prob_37_mean": float(teacher_prob_37.mean().detach().item()),
        "teacher_prob_37_fg_seed_mean": _masked_mean_for_log(teacher_prob_37, fg_seed_37),
        "teacher_prob_37_bg_seed_mean": _masked_mean_for_log(teacher_prob_37, bg_seed_37),
        "teacher_prob_37_extent_mean": _masked_mean_for_log(teacher_prob_37, extent_37),
    }
    if lambda_dyn_pos <= 0.0 and lambda_dyn_bg <= 0.0:
        masks["stats"] = base_stats
        return masks

    feat = F.normalize(feature_37, dim=1) if bool(getattr(cfg, "OEM_PROTO_L2_NORM", True)) else feature_37
    b, c, h, w = feat.shape
    n = h * w
    pos_mask_37 = torch.zeros((b, 1, h, w), device=feat.device, dtype=torch.bool)
    bg_mask_37 = torch.zeros_like(pos_mask_37)
    proto_delta_37 = torch.zeros((b, 1, h, w), device=feat.device, dtype=torch.float32)

    fg_min = int(getattr(cfg, "OEM_PROTO_MIN_FG_PIXELS", 4))
    bg_min = int(getattr(cfg, "OEM_PROTO_MIN_BG_PIXELS", 16))
    pos_raw_count = 0
    pos_cap_count = 0
    bg_raw_count = 0
    bg_cap_count = 0
    skip_no_fg = 0
    skip_no_bg = 0
    skip_no_pos_region = 0

    for i in range(b):
        feat_i = feat[i].flatten(1)
        fg_seed = fg_seed_37[i, 0].flatten()
        bg_seed = bg_seed_37[i, 0].flatten()
        if int(fg_seed.sum().item()) < fg_min:
            skip_no_fg += 1
            continue
        if int(bg_seed.sum().item()) < bg_min:
            skip_no_bg += 1
            continue

        fg_proto = F.normalize(feat_i[:, fg_seed].mean(dim=1), dim=0)
        bg_proto = F.normalize(feat_i[:, bg_seed].mean(dim=1), dim=0)
        sim_fg = (feat_i * fg_proto[:, None]).sum(dim=0)
        sim_bg = (feat_i * bg_proto[:, None]).sum(dim=0)
        proto_delta = sim_fg - sim_bg
        proto_delta_37[i, 0] = proto_delta.view(h, w)

        extent = extent_37[i, 0].flatten()
        unknown = unknown_37[i, 0].flatten()
        pos_region = extent.clone()
        if bool(getattr(cfg, "OEM_USE_UNKNOWN_FOR_POS", False)):
            pos_region = pos_region | unknown
        pos_region = pos_region & (~bg_seed)
        if int(pos_region.sum().item()) <= 0:
            skip_no_pos_region += 1
            continue
        pos_values = proto_delta[pos_region]
        proto_pos_thr = torch.quantile(pos_values, float(getattr(cfg, "OEM_PROTO_POS_QUANTILE", 0.80)))
        proto_pos_thr = torch.maximum(
            proto_pos_thr,
            torch.tensor(float(getattr(cfg, "OEM_PROTO_POS_MIN", 0.05)), device=feat.device),
        )
        teacher_i = teacher_prob_37[i, 0].flatten()
        pos_candidate = (
            pos_region
            & (teacher_i >= float(getattr(cfg, "OEM_TEACHER_POS_THRESH", 0.60)))
            & (proto_delta >= proto_pos_thr)
        )
        pos_raw = int(pos_candidate.sum().item())
        pos_raw_count += pos_raw
        if pos_raw >= int(getattr(cfg, "OEM_POS_MIN_PIXELS", 4)):
            fg_count = int(fg_seed.sum().item())
            cap_by_ratio = int(float(getattr(cfg, "OEM_POS_CAP_RATIO", 0.05)) * n)
            cap_by_fg = int(float(getattr(cfg, "OEM_POS_CAP_TO_FG", 0.75)) * fg_count)
            pos_cap = max(int(getattr(cfg, "OEM_POS_MIN_PIXELS", 4)), min(cap_by_ratio, cap_by_fg))
            proto_norm = _normalize_map_01(proto_delta)
            pos_score = 0.5 * teacher_i + 0.5 * proto_norm
            selected_pos = _select_topk_mask(pos_candidate.view(h, w), pos_score.view(h, w), pos_cap)
            pos_mask_37[i, 0] = selected_pos
            pos_cap_count += int(selected_pos.sum().item())
        else:
            skip_no_pos_region += 1

        bg_region = (~fg_seed)
        if bool(getattr(cfg, "OEM_USE_UNKNOWN_FOR_BG", True)):
            bg_region = bg_region & (unknown | (~extent))
        else:
            bg_region = bg_region & (~extent)
        bg_region = bg_region & (~pos_mask_37[i, 0].flatten())
        if int(bg_region.sum().item()) <= 0:
            continue
        bg_values = proto_delta[bg_region]
        proto_bg_thr = torch.quantile(bg_values, float(getattr(cfg, "OEM_PROTO_BG_QUANTILE", 0.20)))
        proto_bg_thr = torch.minimum(
            proto_bg_thr,
            torch.tensor(float(getattr(cfg, "OEM_PROTO_BG_MAX", -0.05)), device=feat.device),
        )
        bg_candidate = (
            bg_region
            & (teacher_i <= float(getattr(cfg, "OEM_TEACHER_BG_THRESH", 0.20)))
            & (proto_delta <= proto_bg_thr)
        )
        bg_raw_count += int(bg_candidate.sum().item())
        pos_count = int(pos_mask_37[i, 0].sum().item())
        if pos_count > 0:
            cap_by_ratio = int(float(getattr(cfg, "OEM_BG_CAP_RATIO", 0.10)) * n)
            cap_by_pos = int(float(getattr(cfg, "OEM_BG_TO_POS_MAX", 3.0)) * pos_count)
            bg_cap = min(cap_by_ratio, max(cap_by_pos, int(getattr(cfg, "OEM_BG_MIN_PIXELS", 32))))
        else:
            bg_cap = int(float(getattr(cfg, "OEM_BG_CAP_IF_NO_POS_RATIO", 0.03)) * n)
        bg_score = (1.0 - teacher_i) + torch.clamp(-proto_delta, min=0.0)
        selected_bg = _select_topk_mask(bg_candidate.view(h, w), bg_score.view(h, w), bg_cap)
        bg_mask_37[i, 0] = selected_bg
        bg_cap_count += int(selected_bg.sum().item())

    pos_mask_68 = F.interpolate(pos_mask_37.float(), size=teacher_prob_68.shape[-2:], mode="nearest")
    bg_mask_68 = F.interpolate(bg_mask_37.float(), size=teacher_prob_68.shape[-2:], mode="nearest")
    bg_mask_68 = bg_mask_68 * (1.0 - pos_mask_68)
    if str(getattr(cfg, "OEM_DYN_POS_TARGET_MODE", "teacher_soft")).lower() == "teacher_soft":
        pos_target_68 = teacher_prob_68.clamp(
            min=float(getattr(cfg, "OEM_DYN_POS_TARGET_MIN", 0.65)),
            max=float(getattr(cfg, "OEM_DYN_POS_TARGET_MAX", 0.95)),
        )
    else:
        pos_target_68 = torch.full_like(teacher_prob_68, float(getattr(cfg, "OEM_DYN_POS_TARGET_VALUE", 0.75)))

    masks.update(
        {
            "pos_mask_37": pos_mask_37.float(),
            "bg_mask_37": bg_mask_37.float(),
            "pos_mask_68": pos_mask_68,
            "bg_mask_68": bg_mask_68,
            "pos_target_68": pos_target_68,
            "bg_target_68": torch.full_like(teacher_prob_68, float(getattr(cfg, "OEM_DYN_BG_TARGET", 0.0))),
            "proto_delta_37": proto_delta_37,
            "teacher_prob_37": teacher_prob_37,
        }
    )
    proto_flat = proto_delta_37.flatten()
    base_stats.update(
        {
            "oem_pos_raw_ratio": float(pos_raw_count) / float(max(1, b * n)),
            "oem_pos_capped_ratio": float(pos_cap_count) / float(max(1, b * n)),
            "oem_bg_raw_ratio": float(bg_raw_count) / float(max(1, b * n)),
            "oem_bg_capped_ratio": float(bg_cap_count) / float(max(1, b * n)),
            "oem_skip_no_fg_proto": skip_no_fg,
            "oem_skip_no_bg_proto": skip_no_bg,
            "oem_skip_no_pos_region": skip_no_pos_region,
            "proto_delta_mean": float(proto_flat.mean().detach().item()),
            "proto_delta_min": float(proto_flat.min().detach().item()),
            "proto_delta_max": float(proto_flat.max().detach().item()),
        }
    )
    masks["stats"] = base_stats
    return masks


def build_dabe_oem_dynamic_loss(logits, oem_masks, cfg):
    eps = float(getattr(cfg, "OEM_DYNAMIC_LOSS_EPS", getattr(cfg, "OEM_SEED_GROUP_EPS", 1e-6)))
    loss_dyn_pos = masked_bce_with_logits(
        logits,
        oem_masks["pos_target_68"],
        oem_masks["pos_mask_68"],
        eps=eps,
    )
    loss_dyn_bg = masked_bce_with_logits(
        logits,
        oem_masks["bg_target_68"],
        oem_masks["bg_mask_68"],
        eps=eps,
    )
    stats = {
        "loss_dyn_pos": float(loss_dyn_pos.detach().item()),
        "loss_dyn_bg": float(loss_dyn_bg.detach().item()),
    }
    return loss_dyn_pos, loss_dyn_bg, stats


def soft_tversky_loss(logits, target, alpha_fp=0.3, beta_fn=0.7, eps=1e-6):
    prob = torch.sigmoid(logits)
    reduce_dims = (1, 2, 3)
    tp = (prob * target).sum(dim=reduce_dims)
    fp = (prob * (1.0 - target)).sum(dim=reduce_dims)
    fn = ((1.0 - prob) * target).sum(dim=reduce_dims)
    tversky = (tp + float(eps)) / (tp + float(alpha_fp) * fp + float(beta_fn) * fn + float(eps))
    return (1.0 - tversky).mean()


def dabe_area_guard_loss(cfg, epoch, logits, pseudo_68):
    if not bool(getattr(cfg, "USE_DABE_AREA_GUARD", False)):
        return logits.sum() * 0.0
    if int(epoch) < int(getattr(cfg, "DABE_AREA_GUARD_START_EPOCH", 7)):
        return logits.sum() * 0.0
    prob = torch.sigmoid(logits)
    if bool(getattr(cfg, "DABE_AREA_GUARD_USE_SOFT_AREA", True)):
        pred_area = prob.mean(dim=(1, 2, 3))
    else:
        pred_area = (prob > 0.5).float().mean(dim=(1, 2, 3))
    dabe_area = pseudo_68.mean(dim=(1, 2, 3))
    lower_bound = float(getattr(cfg, "DABE_AREA_GUARD_RATIO", 0.85)) * dabe_area
    loss_area = F.relu(lower_bound - pred_area).pow(2).mean()
    return float(getattr(cfg, "DABE_AREA_GUARD_WEIGHT", 0.02)) * loss_area


def build_dabe_aware_target_and_weight(cfg, batch, pseudo_68, teacher_binary, dabe_weight, teacher_weight):
    required = ("dabe_fg_core_68", "dabe_bg_core_68", "dabe_evidence_68", "dabe_uncertain_68")
    missing = [name for name in required if name not in batch]
    if missing:
        raise RuntimeError(f"USE_DABE_AWARE_LOSS=True requires batch fields: missing={missing}")

    device = pseudo_68.device
    fg_core = batch["dabe_fg_core_68"].to(device, non_blocking=True).float()
    bg_core = batch["dabe_bg_core_68"].to(device, non_blocking=True).float()
    evidence = batch["dabe_evidence_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    uncertain = batch["dabe_uncertain_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    fg_core = (fg_core > 0.5).float()
    bg_core = (bg_core > 0.5).float()

    mixed_target = (float(dabe_weight) * pseudo_68 + float(teacher_weight) * teacher_binary).clamp(0.0, 1.0)
    target = mixed_target
    if bool(getattr(cfg, "DABE_AWARE_CORE_LOCK", True)):
        target = torch.where(fg_core > 0.5, torch.ones_like(target), target)
        target = torch.where(bg_core > 0.5, torch.zeros_like(target), target)
    target = target.clamp(0.0, 1.0)

    if bool(getattr(cfg, "DABE_AWARE_WEIGHTED_BCE", True)):
        weight_map = float(getattr(cfg, "DABE_UNCERTAIN_WEIGHT", 0.20)) + (
            float(getattr(cfg, "DABE_EVIDENCE_WEIGHT_SCALE", 0.40)) * evidence
        )
        weight_map = torch.clamp(weight_map, min=float(getattr(cfg, "DABE_UNCERTAIN_WEIGHT", 0.20)), max=1.0)
        weight_map = torch.where(
            fg_core > 0.5,
            torch.full_like(weight_map, float(getattr(cfg, "DABE_FG_CORE_WEIGHT", 2.0))),
            weight_map,
        )
        weight_map = torch.where(
            bg_core > 0.5,
            torch.full_like(weight_map, float(getattr(cfg, "DABE_BG_CORE_WEIGHT", 1.2))),
            weight_map,
        )
    else:
        weight_map = torch.ones_like(target)
    weight_map = weight_map.clamp(min=1e-6)

    stats = {
        "fg_core_area": float(fg_core.detach().mean().item()),
        "bg_core_area": float(bg_core.detach().mean().item()),
        "uncertain_area": float(uncertain.detach().mean().item()),
        "evidence_mean": float(evidence.detach().mean().item()),
        "target_area": float(target.detach().mean().item()),
        "weight_map_mean": float(weight_map.detach().mean().item()),
        "weight_map_min": float(weight_map.detach().min().item()),
        "weight_map_max": float(weight_map.detach().max().item()),
    }
    return target, weight_map, stats


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


def shape_text(value):
    if torch.is_tensor(value):
        return str(list(value.shape))
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}:{shape_text(val)}" for key, val in value.items()) + "}"
    return str(type(value).__name__)


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


def log_mvflip_first_batch(
    logger,
    model_input,
    hflip_model_input,
    raw_student_logits,
    raw_hflip_logits,
    student_logits,
    hflip_logits,
    pseudo,
    hflip_prob_inv,
    lambda_view,
    stats,
):
    logger.log(f"[MVFlip FirstBatch] normal feature shape = {shape_text(model_input)}")
    logger.log(f"[MVFlip FirstBatch] hflip feature shape = {shape_text(hflip_model_input)}")
    logger.log(f"[MVFlip FirstBatch] normal raw logits shape = {list(raw_student_logits.shape)}")
    logger.log(f"[MVFlip FirstBatch] hflip raw logits shape = {list(raw_hflip_logits.shape)}")
    logger.log(f"[MVFlip FirstBatch] normal loss logits shape = {list(student_logits.shape)}")
    logger.log(f"[MVFlip FirstBatch] hflip loss logits shape = {list(hflip_logits.shape)}")
    logger.log(f"[MVFlip FirstBatch] hflip inverted prob shape = {list(hflip_prob_inv.shape)}")
    logger.log(f"[MVFlip FirstBatch] pseudo shape = {list(pseudo.shape)}")
    logger.log(
        f"[MVFlip FirstBatch] lambda_view = {float(lambda_view):.8f} | "
        f"loss_view_raw = {float(stats['loss_raw']):.8f} | "
        f"core_ratio = {float(stats['core_ratio']):.8f} | "
        f"mean_abs_diff = {float(stats['mean_abs_diff']):.8f}"
    )


def log_mvproto_first_batch(logger, model_input, hflip_model_input, student_out, hflip_out, pseudo, lambda_proto, stats):
    is_hs = str(stats.get("proto_mode", "global")) == "hard_selective"
    tag = "[MVProto-HS FirstBatch]" if is_hs else "[MVProto FirstBatch]"
    logger.log(f"{tag} normal feature shape = {shape_text(model_input)}")
    logger.log(f"{tag} hflip feature shape = {shape_text(hflip_model_input)}")
    logger.log(f"{tag} normal logits shape = {list(extract_logits(student_out).shape)}")
    logger.log(f"{tag} hflip logits shape = {list(extract_logits(hflip_out).shape)}")
    logger.log(f"{tag} normal proto_feat shape = {list(student_out['proto_feat'].shape)}")
    logger.log(f"{tag} hflip proto_feat shape = {list(hflip_out['proto_feat'].shape)}")
    logger.log(f"{tag} pseudo shape = {list(pseudo.shape)}")
    logger.log(
        f"{tag} lambda_proto = {float(lambda_proto):.8f} | "
        f"loss_proto = {float(stats['loss_proto']):.8f} | "
        f"proto_valid_ratio = {float(stats['valid_ratio']):.8f} | "
        f"fg_core_ratio = {float(stats['fg_core_ratio']):.8f} | "
        f"bg_core_ratio = {float(stats['bg_core_ratio']):.8f}"
    )
    if is_hs:
        logger.log(
            f"{tag} bg_hard/ring/disagree/residual = "
            f"{float(stats['bg_hard_ratio']):.8f}/"
            f"{float(stats['bg_ring_ratio']):.8f}/"
            f"{float(stats['bg_disagree_ratio']):.8f}/"
            f"{float(stats['bg_residual_ratio']):.8f} | "
            f"hard_fg/hard_bg = {float(stats['hard_fg_ratio']):.8f}/"
            f"{float(stats['hard_bg_ratio']):.8f} | "
            f"fg_fallback_ratio = {float(stats['fg_fallback_ratio']):.8f}"
        )
    logger.log(
        f"{tag} align/sep/pixel = "
        f"{float(stats['align_loss']):.8f}/"
        f"{float(stats['sep_loss']):.8f}/"
        f"{float(stats['pixel_loss']):.8f} | "
        f"pixel_fg/pixel_bg = {float(stats['pixel_fg_loss']):.8f}/"
        f"{float(stats['pixel_bg_loss']):.8f} | "
        f"sep_active = {float(stats['sep_active_ratio']):.8f} | "
        f"cos_fg_view/cos_bg_view/cos_fg_bg = "
        f"{float(stats['cos_fg_view']):.8f}/"
        f"{float(stats['cos_bg_view']):.8f}/"
        f"{float(stats['cos_fg_bg']):.8f}"
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


def log_tadr_first_batch(logger, output):
    logger.log(f"[TADR FirstBatch] coarse_prob_68 shape = {list(output['coarse_prob_68'].shape)}")
    logger.log(f"[TADR FirstBatch] uncertainty_68 shape = {list(output['uncertainty_68'].shape)}")
    logger.log(f"[TADR FirstBatch] sobel_68 shape = {list(output['sobel_68'].shape)}")
    logger.log(f"[TADR FirstBatch] coarse_boundary_68 shape = {list(output['coarse_boundary_68'].shape)}")
    logger.log(f"[TADR FirstBatch] router_input shape = {list(output['router_input'].shape)}")
    logger.log(f"[TADR FirstBatch] router_map_68 shape = {list(output['router_map_68'].shape)}")
    logger.log(
        f"[TADR FirstBatch] router_map mean/min/max = "
        f"{output_scalar(output, 'tadr_router_mean'):.8f}/"
        f"{output_scalar(output, 'tadr_router_min'):.8f}/"
        f"{output_scalar(output, 'tadr_router_max'):.8f}"
    )
    logger.log(
        f"[TADR FirstBatch] base_gate mean/min/max = "
        f"{output_scalar(output, 'tadr_base_gate_mean'):.8f}/"
        f"{output_scalar(output, 'tadr_base_gate_min'):.8f}/"
        f"{output_scalar(output, 'tadr_base_gate_max'):.8f}"
    )
    logger.log(
        f"[TADR FirstBatch] final_detail_gate mean/min/max = "
        f"{output_scalar(output, 'tadr_final_gate_mean'):.8f}/"
        f"{output_scalar(output, 'tadr_final_gate_min'):.8f}/"
        f"{output_scalar(output, 'tadr_final_gate_max'):.8f}"
    )
    logger.log(
        f"[TADR FirstBatch] residual abs mean/max = "
        f"{output_scalar(output, 'ndr_residual_abs_mean'):.8f}/"
        f"{output_scalar(output, 'ndr_residual_abs_max'):.8f}"
    )
    logger.log(f"[TADR FirstBatch] beta_eff = {output_scalar(output, 'ndr_beta_eff'):.8f}")


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
    if use_hflip_view(cfg):
        logger.log(f"hflip feature cache path = {train_dataset.hflip_feature_root}")
        logger.log(f"first hflip feature cache file = {train_dataset.hflip_first_cache_path}")
        logger.log(f"hflip feature shape example = {train_dataset.hflip_feature_shape}")
    logger.log(f"pseudo shape example = {train_dataset.pseudo_shape}")
    if getattr(cfg, "USE_DABE_PSEUDO", False):
        logger.log(f"DABE pseudo cache path = {train_dataset.dabe_cache_root}")
        logger.log(f"first DABE pseudo file = {train_dataset.dabe_first_cache_path}")
        logger.log(f"first DABE pseudo tensor key = {train_dataset.dabe_first_source_key}")
        logger.log(f"first DABE pseudo resized_from_37 = {train_dataset.dabe_first_resized_from_37}")
        if train_dataset.dabe_first_resized_from_37:
            logger.log(
                "first DABE pseudo resize = "
                f"{train_dataset.dabe_first_resized_from} -> {train_dataset.dabe_first_resized_to}"
            )
        logger.log(f"use_dabe_pseudo = true")
        logger.log(f"dabe_version = {getattr(cfg, 'DABE_VERSION', 'v2')}")
        logger.log(f"dabe_pseudo_root = {getattr(cfg, 'DABE_PSEUDO_ROOT', '')}")
    if getattr(cfg, "USE_DABE_PU", False):
        logger.log(f"DABE-PU cache path = {train_dataset.dabe_pu_cache_root}")
        logger.log(f"first DABE-PU file = {train_dataset.dabe_pu_first_cache_path}")
        logger.log("use_dabe_pu = true")
        logger.log(f"dabe_pu_version = {getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')}")
        logger.log(f"dabe_pu_root = {getattr(cfg, 'DABE_PU_ROOT', '')}")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "dabe_pu_v11_oem":
            logger.log("p_init_mode = dabe_pu_v11_oem")
            logger.log("p_init_formula = DABE-PU fg/bg seeds + OEM teacher-prototype dynamic extent")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
        if str(getattr(cfg, "P_INIT_MODE", "")) in {
            "dabe_pu_v11_desplsched",
            "dabe_pu_v11_desplsched_exactreset",
            "dabe_pu_v11_desplsched_A1_keepteacher_lowlr",
            "dabe_pu_v11_desplsched_A2_resetteacher_highlr",
            "dabe_pu_v11_desplsched_softteacher",
            "dabe_pu_v11_desplsched_dabehard",
        }:
            logger.log(f"p_init_mode = {getattr(cfg, 'P_INIT_MODE', '')}")
            if get_dabe_pu_despl_teacher_target_mode(cfg) == "soft_prob":
                logger.log("p_init_formula = target_soft_68 weighted BCE + DESPL-style full teacher soft BCE")
            elif get_dabe_pu_despl_static_target_mode(cfg) == "hard_from_target_soft":
                logger.log("p_init_formula = target_hard_from_target_soft_68 weighted BCE + DESPL-style full teacher binary BCE")
            else:
                logger.log("p_init_formula = target_soft_68 weighted BCE + DESPL-style full teacher binary BCE")
            logger.log("use_despl_pseudo = False")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
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
        if bool(getattr(cfg, "USE_DABE_PSEUDO", False)) or bool(getattr(cfg, "USE_DABE_PU", False)):
            use_fixed_in_pseudo = False
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
        elif str(getattr(cfg, "P_INIT_MODE", "")) in {"dabe_only", "dabe_gc_only"}:
            logger.log("p_init_formula = p_dabe_68")
        elif str(getattr(cfg, "P_INIT_MODE", "")) == "dabe_pu_v11":
            logger.log("p_init_formula = target_soft_68 + weight_map_68")
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
    logger.log(f"first batch pseudo source = {train_dataset.pseudo_source}")
    logger.log(f"first batch pseudo final candidate = {train_dataset.pseudo_final_candidate}")
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
        logger.log(
            "first batch p_fixed_area mean = "
            f"{float(batch['p_fixed_area'].float().mean().item()):.6f}"
        )
        logger.log(
            "first batch p_despl_area mean = "
            f"{float(batch['p_despl_area'].float().mean().item()):.6f}"
        )
        if getattr(cfg, "USE_DABE_PSEUDO", False):
            pseudo_dabe = batch["pseudo_dabe"].float()
            logger.log(f"first batch pseudo_dabe tensor shape = {list(pseudo_dabe.shape)}")
            logger.log(f"first batch pseudo equals pseudo_dabe = {bool(torch.allclose(pseudo, pseudo_dabe))}")
            logger.log(f"first batch dabe tensor keys = {batch['dabe_source_key']}")
            logger.log(
                "first batch dabe resized_from_37 count = "
                f"{int(batch['dabe_resized_from_37'].bool().sum().item())}"
            )
            logger.log(
                "first batch dabe area mean = "
                f"{float(batch['p_dabe_area'].float().mean().item()):.6f}"
            )
            if use_dabe_aware_loss(cfg):
                logger.log(f"first batch dabe_fg_core_68 shape = {list(batch['dabe_fg_core_68'].shape)}")
                logger.log(f"first batch dabe_bg_core_68 shape = {list(batch['dabe_bg_core_68'].shape)}")
                logger.log(f"first batch dabe_evidence_68 shape = {list(batch['dabe_evidence_68'].shape)}")
                logger.log(f"first batch dabe_uncertain_68 shape = {list(batch['dabe_uncertain_68'].shape)}")
                first_dabe_weight, first_teacher_weight, _ = get_fixed_teacher_weights(cfg, 1)
                aware_target, aware_weight, aware_stats = build_dabe_aware_target_and_weight(
                    cfg,
                    batch,
                    pseudo,
                    torch.zeros_like(pseudo),
                    first_dabe_weight,
                    first_teacher_weight,
                )
                logger.log(
                    "first batch dabe_fg_core_area_mean = "
                    f"{aware_stats['fg_core_area']:.6f}"
                )
                logger.log(
                    "first batch dabe_bg_core_area_mean = "
                    f"{aware_stats['bg_core_area']:.6f}"
                )
                logger.log(
                    "first batch dabe_uncertain_area_mean = "
                    f"{aware_stats['uncertain_area']:.6f}"
                )
                logger.log(
                    "first batch dabe_evidence_mean = "
                    f"{aware_stats['evidence_mean']:.6f}"
                )
                logger.log(
                    "first batch dabe_weight_map mean/min/max = "
                    f"{float(aware_weight.mean().item()):.6f}/"
                    f"{float(aware_weight.min().item()):.6f}/"
                    f"{float(aware_weight.max().item()):.6f}"
                )
                logger.log(
                    "first batch dabe_aware_target mean/min/max = "
                    f"{float(aware_target.mean().item()):.6f}/"
                    f"{float(aware_target.min().item()):.6f}/"
                    f"{float(aware_target.max().item()):.6f}"
                )
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
        if getattr(cfg, "USE_DABE_PSEUDO", False):
            logger.log(
                "first batch p_dabe_area = "
                + ",".join(f"{float(value):.6f}" for value in batch["p_dabe_area"])
            )
    if getattr(cfg, "USE_DABE_PU", False):
        pu_target = batch["pu_target_soft"].float()
        pu_weight = batch["pu_weight_map"].float()
        pu_hard_thresh = float(getattr(cfg, "DABE_PU_HARD_THRESH", 0.5))
        pu_target_hard = (pu_target > pu_hard_thresh).float()
        logger.log(f"first batch pu_target_soft shape = {list(pu_target.shape)}")
        logger.log(
            "first batch pu_target_soft min/max/mean = "
            f"{float(pu_target.min().item()):.6f}/"
            f"{float(pu_target.max().item()):.6f}/"
            f"{float(pu_target.mean().item()):.6f}"
        )
        logger.log(
            "first batch pu_target_hard min/max/mean = "
            f"{float(pu_target_hard.min().item()):.6f}/"
            f"{float(pu_target_hard.max().item()):.6f}/"
            f"{float(pu_target_hard.mean().item()):.6f}"
        )
        logger.log(f"first batch pu_target_hard_area_mean = {float(pu_target_hard.mean().item()):.6f}")
        logger.log(f"first batch pu_weight_map shape = {list(pu_weight.shape)}")
        logger.log(
            "first batch pu_weight_map min/max/mean = "
            f"{float(pu_weight.min().item()):.6f}/"
            f"{float(pu_weight.max().item()):.6f}/"
            f"{float(pu_weight.mean().item()):.6f}"
        )
        logger.log(f"first batch pseudo equals pu_target_soft = {bool(torch.allclose(pseudo, pu_target))}")
        logger.log(f"first batch pu_fg_core shape = {list(batch['pu_fg_core'].shape)}")
        logger.log(f"first batch pu_fg_fallback shape = {list(batch['pu_fg_fallback'].shape)}")
        logger.log(f"first batch pu_bg_core shape = {list(batch['pu_bg_core'].shape)}")
        logger.log(f"first batch pu_extent shape = {list(batch['pu_extent'].shape)}")
        logger.log(f"first batch pu_unknown shape = {list(batch['pu_unknown'].shape)}")
        if "pu_fg_core_37" in batch and batch["pu_fg_core_37"].numel() > 0:
            logger.log(f"first batch pu_fg_core_37 shape = {list(batch['pu_fg_core_37'].shape)}")
            logger.log(f"first batch pu_fg_fallback_37 shape = {list(batch['pu_fg_fallback_37'].shape)}")
            logger.log(f"first batch pu_bg_core_37 shape = {list(batch['pu_bg_core_37'].shape)}")
            logger.log(f"first batch pu_extent_37 shape = {list(batch['pu_extent_37'].shape)}")
            logger.log(f"first batch pu_unknown_37 shape = {list(batch['pu_unknown_37'].shape)}")
            logger.log(
                "first batch pu37 fg_core/fallback/bg/extent/unknown area mean = "
                f"{float(batch['pu_fg_core_37'].float().mean().item()):.6f}/"
                f"{float(batch['pu_fg_fallback_37'].float().mean().item()):.6f}/"
                f"{float(batch['pu_bg_core_37'].float().mean().item()):.6f}/"
                f"{float(batch['pu_extent_37'].float().mean().item()):.6f}/"
                f"{float(batch['pu_unknown_37'].float().mean().item()):.6f}"
            )
        logger.log(
            "first batch pu fg_core/fallback/bg/extent/unknown area mean = "
            f"{float(batch['pu_fg_core'].float().mean().item()):.6f}/"
            f"{float(batch['pu_fg_fallback'].float().mean().item()):.6f}/"
            f"{float(batch['pu_bg_core'].float().mean().item()):.6f}/"
            f"{float(batch['pu_extent'].float().mean().item()):.6f}/"
            f"{float(batch['pu_unknown'].float().mean().item()):.6f}"
        )
        logger.log(f"first batch pu effective weight sum = {float(pu_weight.sum().item()):.6f}")


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
        or bool(getattr(cfg, "USE_DABE_PSEUDO", False))
        or bool(getattr(cfg, "USE_DABE_PU", False))
    ):
        raise RuntimeError("USE_DREPP=True cannot be combined with USE_QRA, USE_CCR, USE_DESPL_PSEUDO, USE_DABE_PSEUDO, or USE_DABE_PU.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and (
        bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False))
    ):
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
    if bool(getattr(cfg, "USE_DABE_PSEUDO", False)):
        if str(getattr(cfg, "P_INIT_MODE", "")) not in {"dabe_only", "dabe_gc_only"}:
            raise RuntimeError("USE_DABE_PSEUDO=True requires P_INIT_MODE in {'dabe_only', 'dabe_gc_only'}.")
        if bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False)):
            raise RuntimeError("USE_DABE_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
    if bool(getattr(cfg, "USE_DABE_PU", False)):
        if bool(getattr(cfg, "USE_DABE_PSEUDO", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with USE_DABE_PSEUDO=True.")
        if str(getattr(cfg, "P_INIT_MODE", "")) not in {
            "dabe_pu_v11",
            "dabe_pu_v11_oem",
            "dabe_pu_v11_desplsched",
            "dabe_pu_v11_desplsched_exactreset",
            "dabe_pu_v11_desplsched_A1_keepteacher_lowlr",
            "dabe_pu_v11_desplsched_A2_resetteacher_highlr",
            "dabe_pu_v11_desplsched_softteacher",
            "dabe_pu_v11_desplsched_dabehard",
        }:
            raise RuntimeError(
                "USE_DABE_PU=True requires P_INIT_MODE in "
                "{'dabe_pu_v11', 'dabe_pu_v11_oem', "
                "'dabe_pu_v11_desplsched', 'dabe_pu_v11_desplsched_exactreset', "
                "'dabe_pu_v11_desplsched_A1_keepteacher_lowlr', "
                "'dabe_pu_v11_desplsched_A2_resetteacher_highlr', "
                "'dabe_pu_v11_desplsched_softteacher', "
                "'dabe_pu_v11_desplsched_dabehard'}."
            )
        if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() != "pu_v11":
            raise RuntimeError("USE_DABE_PU=True requires DABE_PU_VERSION='pu_v11'.")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() not in {
            "dabe_pu_conf",
            "dabe_pu_balanced_v2",
            "dabe_pu_oem",
            "dabe_pu_despl_sched",
        }:
            raise RuntimeError(
                "USE_DABE_PU=True requires TEACHER_FUSION_MODE in "
                "{'dabe_pu_conf', 'dabe_pu_balanced_v2', 'dabe_pu_oem', 'dabe_pu_despl_sched'}."
            )
        if bool(getattr(cfg, "USE_TEACHER_SOFT_FULL_LOSS", False)):
            if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
                raise RuntimeError("USE_TEACHER_SOFT_FULL_LOSS=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_pu_despl_sched":
            teacher_target_mode = get_dabe_pu_despl_teacher_target_mode(cfg)
            static_target_mode = get_dabe_pu_despl_static_target_mode(cfg)
            if teacher_target_mode == "soft_prob" and bool(getattr(cfg, "USE_TEACHER_BINARY_FULL_LOSS", False)):
                raise RuntimeError("TEACHER_TARGET_MODE='soft_prob' cannot be combined with USE_TEACHER_BINARY_FULL_LOSS=True.")
            if bool(getattr(cfg, "USE_RAST", False)):
                if teacher_target_mode != "binary":
                    raise RuntimeError("USE_RAST=True requires binary teacher target in dabe_pu_despl_sched.")
                if static_target_mode != "soft":
                    raise RuntimeError("USE_RAST=True requires DABE-PU static target mode 'soft'.")
            if bool(getattr(cfg, "USE_HBNS_LITE", False)):
                if teacher_target_mode != "binary":
                    raise RuntimeError("USE_HBNS_LITE=True requires binary teacher target in dabe_pu_despl_sched.")
                if static_target_mode != "soft":
                    raise RuntimeError("USE_HBNS_LITE=True requires DABE-PU static target mode 'soft'.")
            if bool(getattr(cfg, "USE_EPR_POS", False)):
                if bool(getattr(cfg, "USE_HBNS_LITE", False)):
                    raise RuntimeError("USE_EPR_POS=True cannot be combined with USE_HBNS_LITE=True.")
                if teacher_target_mode != "binary":
                    raise RuntimeError("USE_EPR_POS=True requires binary teacher target in dabe_pu_despl_sched.")
                if static_target_mode != "soft":
                    raise RuntimeError("USE_EPR_POS=True requires DABE-PU static target mode 'soft'.")
                if str(getattr(cfg, "EPR_REGION", "extent")).lower() != "extent":
                    raise RuntimeError("USE_EPR_POS=True currently requires EPR_REGION='extent'.")
                if bool(getattr(cfg, "EPR_USE_UNKNOWN", False)):
                    raise RuntimeError("USE_EPR_POS=True requires EPR_USE_UNKNOWN=False.")
                if not bool(getattr(cfg, "EPR_POSITIVE_ONLY", True)):
                    raise RuntimeError("USE_EPR_POS=True requires EPR_POSITIVE_ONLY=True.")
                if str(getattr(cfg, "EPR_FEATURE_SOURCE", "cached_dino")).lower() != "cached_dino":
                    raise RuntimeError("USE_EPR_POS=True currently requires EPR_FEATURE_SOURCE='cached_dino'.")
            if bool(getattr(cfg, "USE_ESA_ASYM", False)):
                if bool(getattr(cfg, "USE_HBNS_LITE", False)) or bool(getattr(cfg, "USE_EPR_POS", False)):
                    raise RuntimeError("USE_ESA_ASYM=True cannot be combined with HBNS-lite or EPR-pos.")
                if teacher_target_mode != "binary":
                    raise RuntimeError("USE_ESA_ASYM=True requires binary teacher target in dabe_pu_despl_sched.")
                if static_target_mode != "soft":
                    raise RuntimeError("USE_ESA_ASYM=True requires DABE-PU static target mode 'soft'.")
                if not bool(getattr(cfg, "USE_RAST", False)):
                    raise RuntimeError("USE_ESA_ASYM=True requires USE_RAST=True.")
                if bool(getattr(cfg, "ESA_TOUCH_UNKNOWN", False)):
                    raise RuntimeError("USE_ESA_ASYM=True currently requires ESA_TOUCH_UNKNOWN=False.")
                if not bool(getattr(cfg, "ESA_USE_DINO_MARGIN", True)):
                    raise RuntimeError("USE_ESA_ASYM=True requires ESA_USE_DINO_MARGIN=True.")
        elif bool(getattr(cfg, "USE_RAST", False)):
            raise RuntimeError("USE_RAST=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_HBNS_LITE", False)):
            raise RuntimeError("USE_HBNS_LITE=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_EPR_POS", False)):
            raise RuntimeError("USE_EPR_POS=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_ESA_ASYM", False)):
            raise RuntimeError("USE_ESA_ASYM=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with USE_QRA=True or USE_CCR=True.")
        if bool(getattr(cfg, "USE_DABE_AWARE_LOSS", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with USE_DABE_AWARE_LOSS=True.")
        if bool(getattr(cfg, "USE_DABE_TVERSKY_LOSS", False)) or bool(getattr(cfg, "USE_DABE_AREA_GUARD", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with DABE Tversky or area guard losses.")
        if str(getattr(cfg, "GKD_MODE", "off")).lower() != "off" or bool(getattr(cfg, "USE_GKD_LITE", False)):
            raise RuntimeError("USE_DABE_PU=True requires GKD_MODE='off' and USE_GKD_LITE=False.")
        if bool(getattr(cfg, "USE_PROTO_CONTRAST", False)) or bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with proto contrast or multi-view feature.")
        if bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)) or bool(getattr(cfg, "USE_TADR_ROUTER", False)):
            raise RuntimeError("USE_DABE_PU=True cannot be combined with view consistency or TADR router.")
    elif bool(getattr(cfg, "USE_HBNS_LITE", False)):
        raise RuntimeError("USE_HBNS_LITE=True requires USE_DABE_PU=True.")
    elif bool(getattr(cfg, "USE_EPR_POS", False)):
        raise RuntimeError("USE_EPR_POS=True requires USE_DABE_PU=True.")
    elif bool(getattr(cfg, "USE_ESA_ASYM", False)):
        raise RuntimeError("USE_ESA_ASYM=True requires USE_DABE_PU=True.")
    if bool(getattr(cfg, "USE_DABE_AWARE_LOSS", False)):
        if not bool(getattr(cfg, "USE_DABE_PSEUDO", False)):
            raise RuntimeError("USE_DABE_AWARE_LOSS=True requires USE_DABE_PSEUDO=True.")
        if str(getattr(cfg, "P_INIT_MODE", "")) != "dabe_only":
            raise RuntimeError("USE_DABE_AWARE_LOSS=True requires P_INIT_MODE='dabe_only'.")
        if bool(getattr(cfg, "USE_QRA", False)) or bool(getattr(cfg, "USE_CCR", False)):
            raise RuntimeError("USE_DABE_AWARE_LOSS=True cannot be combined with USE_QRA=True or USE_CCR=True.")
        if bool(getattr(cfg, "USE_DREPP", False)):
            raise RuntimeError("USE_DABE_AWARE_LOSS=True cannot be combined with USE_DREPP=True.")
    if bool(getattr(cfg, "USE_QRA", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_QRA=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_CCR", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_CCR=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DREPP", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DREPP=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DESPL_PSEUDO", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DABE_PSEUDO", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DABE_PSEUDO=True cannot be combined with --pseudo_cache_override.")
    if bool(getattr(cfg, "USE_DABE_PU", False)) and args.pseudo_cache_override:
        raise RuntimeError("USE_DABE_PU=True cannot be combined with --pseudo_cache_override.")
    cfg.PSEUDO_CACHE_OVERRIDE = args.pseudo_cache_override
    max_epoch = int(args.max_epochs) if args.max_epochs is not None else int(cfg.MAX_EPOCH)
    reset_epoch = get_reset_epoch(cfg)
    reset_enabled = is_finetune_reset_enabled(cfg)
    default_pre_reset_epochs = reset_epoch - 1 if reset_enabled else max_epoch
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
        logger.log(
            f"lr_linear_stage1_epochs = {int(getattr(cfg, 'LR_LINEAR_STAGE1_EPOCHS', default_pre_reset_epochs))}"
        )
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
        logger.log(f"finetune_reset_enabled = {reset_enabled}")
        logger.log(
            f"finetune_reset_rebuild_optimizer = "
            f"{bool(getattr(cfg, 'FINETUNE_RESET_REBUILD_OPTIMIZER', True))}"
        )
        logger.log(
            f"finetune_reset_rebuild_scheduler = "
            f"{bool(getattr(cfg, 'FINETUNE_RESET_REBUILD_SCHEDULER', True))}"
        )
        logger.log(
            f"finetune_reset_global_step = {bool(getattr(cfg, 'FINETUNE_RESET_GLOBAL_STEP', True))}"
        )
        logger.log(f"finetune_reset_teacher = {bool(getattr(cfg, 'FINETUNE_RESET_TEACHER', False))}")
        logger.log(f"finetune_reset_timing = {finetune_reset_timing(cfg)}")
        logger.log(f"finetune_reset_force_lr_floor = {bool(getattr(cfg, 'FINETUNE_RESET_FORCE_LR_FLOOR', False))}")
        logger.log(f"finetune_reset_lr = {float(getattr(cfg, 'FINETUNE_RESET_LR', getattr(cfg, 'LR_FLOOR', 0.0))):.8f}")
        logger.log(f"teacher_fusion_mode = {getattr(cfg, 'TEACHER_FUSION_MODE', 'default')}")
        logger.log(f"fusion_orig_decay_epochs = {int(getattr(cfg, 'FUSION_ORIG_DECAY_EPOCHS', 20))}")
        logger.log(f"fusion_hold_fixed_weight = {float(getattr(cfg, 'FUSION_HOLD_FIXED_WEIGHT', 0.05)):.6f}")
        logger.log(
            f"teacher_fusion_pre_reset_epochs = "
            f"{int(getattr(cfg, 'TEACHER_FUSION_PRE_RESET_EPOCHS', default_pre_reset_epochs))}"
        )
        logger.log(
            f"fusion_min_fixed_weight = "
            f"{float(getattr(cfg, 'FUSION_MIN_FIXED_WEIGHT', 1.0 - float(getattr(cfg, 'TEACHER_FUSION_MAX_WEIGHT', 0.95)))):.6f}"
        )
        logger.log(f"teacher_fusion_max_weight = {float(getattr(cfg, 'TEACHER_FUSION_MAX_WEIGHT', 1.0)):.6f}")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_sticky":
            logger.log(f"DABE_STICKY_E1_E6_DABE_WEIGHT = {float(getattr(cfg, 'DABE_STICKY_E1_E6_DABE_WEIGHT', 1.0)):.6f}")
            logger.log(f"DABE_STICKY_STAGE2_START = {int(getattr(cfg, 'DABE_STICKY_STAGE2_START', 7))}")
            logger.log(f"DABE_STICKY_STAGE2_END = {int(getattr(cfg, 'DABE_STICKY_STAGE2_END', 15))}")
            logger.log(f"DABE_STICKY_STAGE2_DABE_START = {float(getattr(cfg, 'DABE_STICKY_STAGE2_DABE_START', 0.85)):.6f}")
            logger.log(f"DABE_STICKY_STAGE2_DABE_END = {float(getattr(cfg, 'DABE_STICKY_STAGE2_DABE_END', 0.55)):.6f}")
            logger.log(f"DABE_STICKY_STAGE3_START = {int(getattr(cfg, 'DABE_STICKY_STAGE3_START', 16))}")
            logger.log(f"DABE_STICKY_STAGE3_END = {int(getattr(cfg, 'DABE_STICKY_STAGE3_END', 20))}")
            logger.log(f"DABE_STICKY_STAGE3_DABE_START = {float(getattr(cfg, 'DABE_STICKY_STAGE3_DABE_START', 0.55)):.6f}")
            logger.log(f"DABE_STICKY_STAGE3_DABE_END = {float(getattr(cfg, 'DABE_STICKY_STAGE3_DABE_END', 0.25)):.6f}")
            logger.log(f"DABE_STICKY_AFTER_EPOCH = {int(getattr(cfg, 'DABE_STICKY_AFTER_EPOCH', 21))}")
            logger.log(f"DABE_STICKY_AFTER_DABE_WEIGHT = {float(getattr(cfg, 'DABE_STICKY_AFTER_DABE_WEIGHT', 0.15)):.6f}")
            logger.log(f"DABE_STICKY_AFTER_TEACHER_WEIGHT = {float(getattr(cfg, 'DABE_STICKY_AFTER_TEACHER_WEIGHT', 0.85)):.6f}")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_pu_conf":
            logger.log(f"USE_DABE_PU = {bool(getattr(cfg, 'USE_DABE_PU', False))}")
            logger.log(f"DABE_PU_VERSION = {getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')}")
            logger.log(f"DABE_PU_ROOT = {getattr(cfg, 'DABE_PU_ROOT', '')}")
            logger.log(f"USE_DABE_PU_STATIC_LOSS = {bool(getattr(cfg, 'USE_DABE_PU_STATIC_LOSS', True))}")
            logger.log(f"USE_TEACHER_CONF_LOSS = {bool(getattr(cfg, 'USE_TEACHER_CONF_LOSS', True))}")
            logger.log(f"DABE_PU_STATIC_E1_E6 = {float(getattr(cfg, 'DABE_PU_STATIC_E1_E6', 1.0)):.6f}")
            logger.log(f"DABE_PU_TEACHER_E1_E6 = {float(getattr(cfg, 'DABE_PU_TEACHER_E1_E6', 0.0)):.6f}")
            logger.log(f"DABE_PU_STAGE2_START = {int(getattr(cfg, 'DABE_PU_STAGE2_START', 7))}")
            logger.log(f"DABE_PU_STAGE2_END = {int(getattr(cfg, 'DABE_PU_STAGE2_END', 20))}")
            logger.log(f"DABE_PU_STATIC_STAGE2_START = {float(getattr(cfg, 'DABE_PU_STATIC_STAGE2_START', 1.0)):.6f}")
            logger.log(f"DABE_PU_STATIC_STAGE2_END = {float(getattr(cfg, 'DABE_PU_STATIC_STAGE2_END', 0.40)):.6f}")
            logger.log(f"DABE_PU_TEACHER_STAGE2_START = {float(getattr(cfg, 'DABE_PU_TEACHER_STAGE2_START', 0.0)):.6f}")
            logger.log(f"DABE_PU_TEACHER_STAGE2_END = {float(getattr(cfg, 'DABE_PU_TEACHER_STAGE2_END', 0.60)):.6f}")
            logger.log(f"DABE_PU_AFTER_EPOCH = {int(getattr(cfg, 'DABE_PU_AFTER_EPOCH', 21))}")
            logger.log(f"DABE_PU_STATIC_AFTER = {float(getattr(cfg, 'DABE_PU_STATIC_AFTER', 0.30)):.6f}")
            logger.log(f"DABE_PU_TEACHER_AFTER = {float(getattr(cfg, 'DABE_PU_TEACHER_AFTER', 0.70)):.6f}")
            logger.log(f"TEACHER_CONF_FG_THRESH = {float(getattr(cfg, 'TEACHER_CONF_FG_THRESH', 0.75)):.6f}")
            logger.log(f"TEACHER_CONF_BG_THRESH = {float(getattr(cfg, 'TEACHER_CONF_BG_THRESH', 0.25)):.6f}")
            logger.log(f"TEACHER_CONF_WEIGHT_MIN = {float(getattr(cfg, 'TEACHER_CONF_WEIGHT_MIN', 0.0)):.6f}")
            logger.log(f"TEACHER_CONF_WEIGHT_MAX = {float(getattr(cfg, 'TEACHER_CONF_WEIGHT_MAX', 1.0)):.6f}")
            logger.log(f"TEACHER_CONF_IGNORE_PU_CORE = {bool(getattr(cfg, 'TEACHER_CONF_IGNORE_PU_CORE', True))}")
            logger.log(f"DABE_PU_WEIGHTED_BCE_EPS = {float(getattr(cfg, 'DABE_PU_WEIGHTED_BCE_EPS', 1e-6)):.8f}")
            logger.log(f"LAMBDA_DABE_PU_STATIC = {float(getattr(cfg, 'LAMBDA_DABE_PU_STATIC', 1.0)):.6f}")
            logger.log(f"LAMBDA_TEACHER_CONF = {float(getattr(cfg, 'LAMBDA_TEACHER_CONF', 1.0)):.6f}")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_pu_despl_sched":
            logger.log(f"USE_DABE_PU = {bool(getattr(cfg, 'USE_DABE_PU', False))}")
            logger.log(f"DABE_PU_VERSION = {getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')}")
            logger.log(f"DABE_PU_ROOT = {getattr(cfg, 'DABE_PU_ROOT', '')}")
            logger.log(f"P_INIT_MODE = {getattr(cfg, 'P_INIT_MODE', '')}")
            logger.log(f"USE_DABE_PU_DESPL_SCHEDULE = {bool(getattr(cfg, 'USE_DABE_PU_DESPL_SCHEDULE', False))}")
            logger.log(f"USE_DABE_PU_STATIC_LOSS = {bool(getattr(cfg, 'USE_DABE_PU_STATIC_LOSS', True))}")
            logger.log(f"DABE_PU_STATIC_TARGET_MODE = {get_dabe_pu_despl_static_target_mode(cfg)}")
            logger.log(f"DABE_PU_HARD_THRESH = {float(getattr(cfg, 'DABE_PU_HARD_THRESH', 0.5)):.6f}")
            logger.log(f"USE_DABE_PU_HARD_STATIC_TARGET = {bool(getattr(cfg, 'USE_DABE_PU_HARD_STATIC_TARGET', False))}")
            logger.log(f"DABE_PU_HARD_KEEP_WEIGHT_MAP = {bool(getattr(cfg, 'DABE_PU_HARD_KEEP_WEIGHT_MAP', True))}")
            logger.log(f"USE_TEACHER_SOFT_FULL_LOSS = {bool(getattr(cfg, 'USE_TEACHER_SOFT_FULL_LOSS', False))}")
            logger.log(f"USE_TEACHER_BINARY_FULL_LOSS = {bool(getattr(cfg, 'USE_TEACHER_BINARY_FULL_LOSS', True))}")
            logger.log(f"TEACHER_TARGET_MODE = {get_dabe_pu_despl_teacher_target_mode(cfg)}")
            logger.log(f"USE_TEACHER_CONF_LOSS = {bool(getattr(cfg, 'USE_TEACHER_CONF_LOSS', False))}")
            logger.log(f"USE_DABE_PU_GROUP_BALANCED_STATIC = {bool(getattr(cfg, 'USE_DABE_PU_GROUP_BALANCED_STATIC', False))}")
            logger.log(f"USE_DABE_OEM = {bool(getattr(cfg, 'USE_DABE_OEM', False))}")
            logger.log(f"use_despl_pseudo = {bool(getattr(cfg, 'USE_DESPL_PSEUDO', False))}")
            logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
            logger.log(f"use_despl_paper_cache = {bool(getattr(cfg, 'USE_DESPL_PAPER_CACHE', False))}")
            logger.log(f"use_fixed_in_pseudo = {bool(getattr(cfg, 'USE_FIXED_IN_PSEUDO', False))}")
            logger.log("fixed_used_for_training = False")
            logger.log(f"DABE_PU_DESPL_STAGE_START = {int(getattr(cfg, 'DABE_PU_DESPL_STAGE_START', 1))}")
            logger.log(f"DABE_PU_DESPL_STAGE_END = {int(getattr(cfg, 'DABE_PU_DESPL_STAGE_END', 20))}")
            logger.log(f"DABE_PU_DESPL_STATIC_START = {float(getattr(cfg, 'DABE_PU_DESPL_STATIC_START', 1.0)):.6f}")
            logger.log(f"DABE_PU_DESPL_STATIC_END = {float(getattr(cfg, 'DABE_PU_DESPL_STATIC_END', 0.05)):.6f}")
            logger.log(f"DABE_PU_DESPL_TEACHER_START = {float(getattr(cfg, 'DABE_PU_DESPL_TEACHER_START', 0.0)):.6f}")
            logger.log(f"DABE_PU_DESPL_TEACHER_END = {float(getattr(cfg, 'DABE_PU_DESPL_TEACHER_END', 0.95)):.6f}")
            logger.log(f"DABE_PU_DESPL_TEACHER_ONLY_START = {int(getattr(cfg, 'DABE_PU_DESPL_TEACHER_ONLY_START', 21))}")
            logger.log(f"DABE_PU_WEIGHTED_BCE_EPS = {float(getattr(cfg, 'DABE_PU_WEIGHTED_BCE_EPS', 1e-6)):.8f}")
            logger.log(f"USE_RAST = {bool(getattr(cfg, 'USE_RAST', False))}")
            logger.log(f"RAST_VERSION = {getattr(cfg, 'RAST_VERSION', 'v1')}")
            logger.log(f"RAST_START_EPOCH = {int(getattr(cfg, 'RAST_START_EPOCH', 7))}")
            logger.log(f"RAST_RAMP_END_EPOCH = {int(getattr(cfg, 'RAST_RAMP_END_EPOCH', 15))}")
            logger.log(f"RAST_STOP_EPOCH = {int(getattr(cfg, 'RAST_STOP_EPOCH', 21))}")
            logger.log(f"RAST_CONFLICT_TEACHER_MULT = {float(getattr(cfg, 'RAST_CONFLICT_TEACHER_MULT', 0.20)):.6f}")
            logger.log(f"RAST_EXTENT_TEACHER_MULT = {float(getattr(cfg, 'RAST_EXTENT_TEACHER_MULT', 0.75)):.6f}")
            logger.log(f"RAST_UNKNOWN_TEACHER_MULT = {float(getattr(cfg, 'RAST_UNKNOWN_TEACHER_MULT', 0.25)):.6f}")
            logger.log(f"RAST_OTHER_TEACHER_MULT = {float(getattr(cfg, 'RAST_OTHER_TEACHER_MULT', 1.0)):.6f}")
            logger.log(f"RAST_STATIC_FG_CORE_MULT = {float(getattr(cfg, 'RAST_STATIC_FG_CORE_MULT', 1.0)):.6f}")
            logger.log(f"RAST_STATIC_BG_CORE_MULT = {float(getattr(cfg, 'RAST_STATIC_BG_CORE_MULT', 1.0)):.6f}")
            logger.log(f"RAST_STATIC_EXTENT_MULT = {float(getattr(cfg, 'RAST_STATIC_EXTENT_MULT', 1.0)):.6f}")
            logger.log(f"RAST_STATIC_UNKNOWN_MULT = {float(getattr(cfg, 'RAST_STATIC_UNKNOWN_MULT', 1.0)):.6f}")
            logger.log(f"RAST_APPLY_TO_FINAL = {bool(getattr(cfg, 'RAST_APPLY_TO_FINAL', True))}")
            logger.log(f"RAST_APPLY_TO_COARSE_AUX = {bool(getattr(cfg, 'RAST_APPLY_TO_COARSE_AUX', True))}")
            logger.log(f"RAST_APPLY_TO_BASE_AUX = {bool(getattr(cfg, 'RAST_APPLY_TO_BASE_AUX', True))}")
            logger.log(f"RAST_DISABLE_AFTER_RESET = {bool(getattr(cfg, 'RAST_DISABLE_AFTER_RESET', True))}")
            logger.log(f"RAST_POST_RESET_ENABLE = {bool(getattr(cfg, 'RAST_POST_RESET_ENABLE', False))}")
            logger.log(f"RAST_POST_RESET_START_EPOCH = {int(getattr(cfg, 'RAST_POST_RESET_START_EPOCH', 21))}")
            logger.log(f"RAST_POST_RESET_END_EPOCH = {int(getattr(cfg, 'RAST_POST_RESET_END_EPOCH', 25))}")
            logger.log(f"RAST_POST_RESET_SCALE = {float(getattr(cfg, 'RAST_POST_RESET_SCALE', 0.30)):.6f}")
            logger.log(f"RAST_POST_RESET_CONFLICT_TEACHER_MULT = {float(getattr(cfg, 'RAST_POST_RESET_CONFLICT_TEACHER_MULT', 0.30)):.6f}")
            logger.log(f"RAST_POST_RESET_EXTENT_TEACHER_MULT = {float(getattr(cfg, 'RAST_POST_RESET_EXTENT_TEACHER_MULT', 1.0)):.6f}")
            logger.log(f"RAST_POST_RESET_UNKNOWN_TEACHER_MULT = {float(getattr(cfg, 'RAST_POST_RESET_UNKNOWN_TEACHER_MULT', 1.0)):.6f}")
            logger.log(f"RAST_POST_RESET_CONFLICT_ONLY = {bool(getattr(cfg, 'RAST_POST_RESET_CONFLICT_ONLY', False))}")
            logger.log(f"RAST_POST_RESET_USE_STATIC_LOSS = {bool(getattr(cfg, 'RAST_POST_RESET_USE_STATIC_LOSS', False))}")
            logger.log(f"RAST_WEIGHTED_LOSS_NORMALIZE = {bool(getattr(cfg, 'RAST_WEIGHTED_LOSS_NORMALIZE', True))}")
            logger.log(f"USE_HBNS_LITE = {bool(getattr(cfg, 'USE_HBNS_LITE', False))}")
            logger.log(f"HBNS_VERSION = {getattr(cfg, 'HBNS_VERSION', 'lite_v1')}")
            logger.log(f"HBNS_START_EPOCH = {int(getattr(cfg, 'HBNS_START_EPOCH', 7))}")
            logger.log(f"HBNS_RAMP_END_EPOCH = {int(getattr(cfg, 'HBNS_RAMP_END_EPOCH', 15))}")
            logger.log(f"HBNS_STOP_EPOCH = {int(getattr(cfg, 'HBNS_STOP_EPOCH', 21))}")
            logger.log(f"HBNS_LAMBDA_MAX = {float(getattr(cfg, 'HBNS_LAMBDA_MAX', 0.01)):.8f}")
            logger.log(f"HBNS_APPLY_TO_FINAL = {bool(getattr(cfg, 'HBNS_APPLY_TO_FINAL', True))}")
            logger.log(f"HBNS_APPLY_TO_COARSE_AUX = {bool(getattr(cfg, 'HBNS_APPLY_TO_COARSE_AUX', True))}")
            logger.log(f"HBNS_APPLY_TO_BASE_AUX = {bool(getattr(cfg, 'HBNS_APPLY_TO_BASE_AUX', False))}")
            logger.log(f"HBNS_STUDENT_PROB_THRESH = {float(getattr(cfg, 'HBNS_STUDENT_PROB_THRESH', 0.50)):.6f}")
            logger.log(f"HBNS_LOW_TARGET_THRESH = {float(getattr(cfg, 'HBNS_LOW_TARGET_THRESH', 0.15)):.6f}")
            logger.log(f"HBNS_LOW_TARGET_WEIGHT_THRESH = {float(getattr(cfg, 'HBNS_LOW_TARGET_WEIGHT_THRESH', 0.50)):.6f}")
            logger.log(f"HBNS_USE_BG_CORE = {bool(getattr(cfg, 'HBNS_USE_BG_CORE', True))}")
            logger.log(f"HBNS_USE_LOW_TARGET_BG = {bool(getattr(cfg, 'HBNS_USE_LOW_TARGET_BG', True))}")
            logger.log(f"HBNS_USE_UNKNOWN_BG_LIKE = {bool(getattr(cfg, 'HBNS_USE_UNKNOWN_BG_LIKE', True))}")
            logger.log(f"HBNS_USE_BG_DILATION_RING = {bool(getattr(cfg, 'HBNS_USE_BG_DILATION_RING', False))}")
            logger.log(f"HBNS_BG_RING_RADIUS = {int(getattr(cfg, 'HBNS_BG_RING_RADIUS', 3))}")
            logger.log(f"HBNS_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'HBNS_MAX_RATIO_PER_IMAGE', 0.05)):.6f}")
            logger.log(f"HBNS_MIN_PIXELS_PER_IMAGE = {int(getattr(cfg, 'HBNS_MIN_PIXELS_PER_IMAGE', 8))}")
            logger.log(f"HBNS_DETACH_MASK = {bool(getattr(cfg, 'HBNS_DETACH_MASK', True))}")
            logger.log(f"HBNS_WEIGHTED_NORMALIZE = {bool(getattr(cfg, 'HBNS_WEIGHTED_NORMALIZE', True))}")
            logger.log(f"USE_EPR_POS = {bool(getattr(cfg, 'USE_EPR_POS', False))}")
            logger.log(f"EPR_VERSION = {getattr(cfg, 'EPR_VERSION', 'pos_lite_v1')}")
            logger.log(f"EPR_START_EPOCH = {int(getattr(cfg, 'EPR_START_EPOCH', 7))}")
            logger.log(f"EPR_RAMP_END_EPOCH = {int(getattr(cfg, 'EPR_RAMP_END_EPOCH', 15))}")
            logger.log(f"EPR_STOP_EPOCH = {int(getattr(cfg, 'EPR_STOP_EPOCH', 21))}")
            logger.log(f"EPR_LAMBDA_MAX = {float(getattr(cfg, 'EPR_LAMBDA_MAX', 0.005)):.8f}")
            logger.log(f"EPR_APPLY_TO_FINAL = {bool(getattr(cfg, 'EPR_APPLY_TO_FINAL', True))}")
            logger.log(f"EPR_APPLY_TO_COARSE_AUX = {bool(getattr(cfg, 'EPR_APPLY_TO_COARSE_AUX', True))}")
            logger.log(f"EPR_APPLY_TO_BASE_AUX = {bool(getattr(cfg, 'EPR_APPLY_TO_BASE_AUX', False))}")
            logger.log(f"EPR_REGION = {getattr(cfg, 'EPR_REGION', 'extent')}")
            logger.log(f"EPR_USE_UNKNOWN = {bool(getattr(cfg, 'EPR_USE_UNKNOWN', False))}")
            logger.log(f"EPR_POSITIVE_ONLY = {bool(getattr(cfg, 'EPR_POSITIVE_ONLY', True))}")
            logger.log(f"EPR_REQUIRE_TEACHER_FG = {bool(getattr(cfg, 'EPR_REQUIRE_TEACHER_FG', True))}")
            logger.log(f"EPR_TEACHER_CONF_THRESH = {float(getattr(cfg, 'EPR_TEACHER_CONF_THRESH', 0.75)):.6f}")
            logger.log(f"EPR_USE_DINO_PROTO_MARGIN = {bool(getattr(cfg, 'EPR_USE_DINO_PROTO_MARGIN', True))}")
            logger.log(f"EPR_MARGIN_THRESH = {float(getattr(cfg, 'EPR_MARGIN_THRESH', 0.05)):.6f}")
            logger.log(f"EPR_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'EPR_MAX_RATIO_PER_IMAGE', 0.03)):.6f}")
            logger.log(f"EPR_MIN_PIXELS_PER_IMAGE = {int(getattr(cfg, 'EPR_MIN_PIXELS_PER_IMAGE', 8))}")
            logger.log(f"EPR_FEATURE_SOURCE = {getattr(cfg, 'EPR_FEATURE_SOURCE', 'cached_dino')}")
            logger.log(f"EPR_FEATURE_SIZE = {int(getattr(cfg, 'EPR_FEATURE_SIZE', 37))}")
            logger.log(f"EPR_LOSS_SIZE = {int(getattr(cfg, 'EPR_LOSS_SIZE', int(getattr(cfg, 'LOSS_SIZE', 68))))}")
            logger.log(f"EPR_DETACH_MASK = {bool(getattr(cfg, 'EPR_DETACH_MASK', True))}")
            logger.log(f"EPR_DETACH_PROTO = {bool(getattr(cfg, 'EPR_DETACH_PROTO', True))}")
            logger.log(f"EPR_WEIGHTED_NORMALIZE = {bool(getattr(cfg, 'EPR_WEIGHTED_NORMALIZE', True))}")
            logger.log(f"USE_EPR_DIAGNOSTIC = {bool(getattr(cfg, 'USE_EPR_DIAGNOSTIC', False))}")
            logger.log(f"USE_ESA_ASYM = {bool(getattr(cfg, 'USE_ESA_ASYM', False))}")
            logger.log(f"ESA_ASYM_VERSION = {getattr(cfg, 'ESA_ASYM_VERSION', 'extent_teacher_bg_routing_v1')}")
            logger.log(f"ESA_ASYM_START_EPOCH = {int(getattr(cfg, 'ESA_ASYM_START_EPOCH', 7))}")
            logger.log(f"ESA_ASYM_RAMP_END_EPOCH = {int(getattr(cfg, 'ESA_ASYM_RAMP_END_EPOCH', 15))}")
            logger.log(f"ESA_ASYM_STOP_EPOCH = {int(getattr(cfg, 'ESA_ASYM_STOP_EPOCH', 21))}")
            logger.log(f"ESA_EXTENT_TEACHER_FG_MULT = {float(getattr(cfg, 'ESA_EXTENT_TEACHER_FG_MULT', 1.0)):.6f}")
            logger.log(f"ESA_EXTENT_TEACHER_BG_MULT = {float(getattr(cfg, 'ESA_EXTENT_TEACHER_BG_MULT', 0.50)):.6f}")
            logger.log(f"ESA_USE_DINO_MARGIN = {bool(getattr(cfg, 'ESA_USE_DINO_MARGIN', True))}")
            logger.log(f"ESA_MARGIN_FG_LIKE = {float(getattr(cfg, 'ESA_MARGIN_FG_LIKE', 0.05)):.6f}")
            logger.log(f"ESA_MARGIN_BG_LIKE = {float(getattr(cfg, 'ESA_MARGIN_BG_LIKE', -0.05)):.6f}")
            logger.log(
                "ESA_EXTENT_TEACHER_BG_FG_LIKE_MULT = "
                f"{float(getattr(cfg, 'ESA_EXTENT_TEACHER_BG_FG_LIKE_MULT', 0.25)):.6f}"
            )
            logger.log(
                "ESA_EXTENT_TEACHER_BG_AMBIG_MULT = "
                f"{float(getattr(cfg, 'ESA_EXTENT_TEACHER_BG_AMBIG_MULT', 0.50)):.6f}"
            )
            logger.log(
                "ESA_EXTENT_TEACHER_BG_BG_LIKE_MULT = "
                f"{float(getattr(cfg, 'ESA_EXTENT_TEACHER_BG_BG_LIKE_MULT', 1.0)):.6f}"
            )
            logger.log(f"ESA_TOUCH_UNKNOWN = {bool(getattr(cfg, 'ESA_TOUCH_UNKNOWN', False))}")
            logger.log(f"ESA_UNKNOWN_TEACHER_MULT = {float(getattr(cfg, 'ESA_UNKNOWN_TEACHER_MULT', 0.50)):.6f}")
            logger.log(f"ESA_FEATURE_SIZE = {int(getattr(cfg, 'ESA_FEATURE_SIZE', 37))}")
            logger.log(f"ESA_DETACH_MASK = {bool(getattr(cfg, 'ESA_DETACH_MASK', True))}")
            logger.log(f"ESA_DETACH_PROTO = {bool(getattr(cfg, 'ESA_DETACH_PROTO', True))}")
            logger.log(f"USE_ESA_DIAGNOSTIC = {bool(getattr(cfg, 'USE_ESA_DIAGNOSTIC', False))}")
            logger.log(f"ESA_DIAG_LOG_INTERVAL_EPOCH = {int(getattr(cfg, 'ESA_DIAG_LOG_INTERVAL_EPOCH', 1))}")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_pu_balanced_v2":
            logger.log(f"USE_DABE_PU = {bool(getattr(cfg, 'USE_DABE_PU', False))}")
            logger.log(f"DABE_PU_VERSION = {getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')}")
            logger.log(f"DABE_PU_ROOT = {getattr(cfg, 'DABE_PU_ROOT', '')}")
            logger.log(f"USE_DABE_PU_GROUP_BALANCED_STATIC = {bool(getattr(cfg, 'USE_DABE_PU_GROUP_BALANCED_STATIC', False))}")
            logger.log(f"USE_TEACHER_CONF_BALANCED = {bool(getattr(cfg, 'USE_TEACHER_CONF_BALANCED', False))}")
            logger.log(f"DABE_PU_V2_STAGE1_END = {int(getattr(cfg, 'DABE_PU_V2_STAGE1_END', 3))}")
            logger.log(f"DABE_PU_V2_STAGE2_START = {int(getattr(cfg, 'DABE_PU_V2_STAGE2_START', 4))}")
            logger.log(f"DABE_PU_V2_STAGE2_END = {int(getattr(cfg, 'DABE_PU_V2_STAGE2_END', 15))}")
            logger.log(f"DABE_PU_V2_STATIC_STAGE2_START = {float(getattr(cfg, 'DABE_PU_V2_STATIC_STAGE2_START', 0.85)):.6f}")
            logger.log(f"DABE_PU_V2_STATIC_STAGE2_END = {float(getattr(cfg, 'DABE_PU_V2_STATIC_STAGE2_END', 0.45)):.6f}")
            logger.log(f"DABE_PU_V2_TEACHER_STAGE2_START = {float(getattr(cfg, 'DABE_PU_V2_TEACHER_STAGE2_START', 0.15)):.6f}")
            logger.log(f"DABE_PU_V2_TEACHER_STAGE2_END = {float(getattr(cfg, 'DABE_PU_V2_TEACHER_STAGE2_END', 0.55)):.6f}")
            logger.log(f"DABE_PU_V2_STAGE3_START = {int(getattr(cfg, 'DABE_PU_V2_STAGE3_START', 16))}")
            logger.log(f"DABE_PU_V2_STATIC_STAGE3 = {float(getattr(cfg, 'DABE_PU_V2_STATIC_STAGE3', 0.30)):.6f}")
            logger.log(f"DABE_PU_V2_TEACHER_STAGE3 = {float(getattr(cfg, 'DABE_PU_V2_TEACHER_STAGE3', 0.70)):.6f}")
            logger.log(f"PU_STATIC_LAMBDA_FG = {float(getattr(cfg, 'PU_STATIC_LAMBDA_FG', 1.0)):.6f}")
            logger.log(f"PU_STATIC_LAMBDA_FG_FALLBACK = {float(getattr(cfg, 'PU_STATIC_LAMBDA_FG_FALLBACK', 0.35)):.6f}")
            logger.log(f"PU_STATIC_LAMBDA_BG = {float(getattr(cfg, 'PU_STATIC_LAMBDA_BG', 0.50)):.6f}")
            logger.log(f"PU_STATIC_LAMBDA_EXTENT = {float(getattr(cfg, 'PU_STATIC_LAMBDA_EXTENT', 0.10)):.6f}")
            logger.log(f"PU_STATIC_LAMBDA_UNKNOWN = {float(getattr(cfg, 'PU_STATIC_LAMBDA_UNKNOWN', 0.0)):.6f}")
            logger.log(f"PU_STATIC_GROUP_EPS = {float(getattr(cfg, 'PU_STATIC_GROUP_EPS', 1e-6)):.8f}")
            logger.log(f"TEACHER_CONF_FG_THRESH = {float(getattr(cfg, 'TEACHER_CONF_FG_THRESH', 0.70)):.6f}")
            logger.log(f"TEACHER_CONF_BG_THRESH = {float(getattr(cfg, 'TEACHER_CONF_BG_THRESH', 0.20)):.6f}")
            logger.log(f"TEACHER_CONF_IGNORE_PU_CORE = {bool(getattr(cfg, 'TEACHER_CONF_IGNORE_PU_CORE', True))}")
            logger.log(f"TEACHER_CONF_LAMBDA_FG = {float(getattr(cfg, 'TEACHER_CONF_LAMBDA_FG', 1.0)):.6f}")
            logger.log(f"TEACHER_CONF_LAMBDA_BG = {float(getattr(cfg, 'TEACHER_CONF_LAMBDA_BG', 0.30)):.6f}")
            logger.log(f"TEACHER_CONF_GROUP_EPS = {float(getattr(cfg, 'TEACHER_CONF_GROUP_EPS', 1e-6)):.8f}")
            logger.log(f"TEACHER_CONF_BG_MAX_RATIO = {float(getattr(cfg, 'TEACHER_CONF_BG_MAX_RATIO', 0.15)):.6f}")
            logger.log(f"TEACHER_CONF_BG_TO_FG_MAX = {float(getattr(cfg, 'TEACHER_CONF_BG_TO_FG_MAX', 5.0)):.6f}")
            logger.log(f"TEACHER_CONF_BG_MIN_PIXELS = {int(getattr(cfg, 'TEACHER_CONF_BG_MIN_PIXELS', 128))}")
            logger.log(f"TEACHER_CONF_BG_CAP_IF_NO_FG_RATIO = {float(getattr(cfg, 'TEACHER_CONF_BG_CAP_IF_NO_FG_RATIO', 0.05)):.6f}")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_pu_oem":
            logger.log(f"USE_DABE_OEM = {bool(getattr(cfg, 'USE_DABE_OEM', False))}")
            logger.log(f"USE_DABE_PU = {bool(getattr(cfg, 'USE_DABE_PU', False))}")
            logger.log(f"DABE_PU_VERSION = {getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')}")
            logger.log(f"DABE_PU_ROOT = {getattr(cfg, 'DABE_PU_ROOT', '')}")
            logger.log(f"USE_DABE_PU_SEED_STATIC_LOSS = {bool(getattr(cfg, 'USE_DABE_PU_SEED_STATIC_LOSS', True))}")
            logger.log(f"USE_DABE_OEM_DYNAMIC_EXTENT = {bool(getattr(cfg, 'USE_DABE_OEM_DYNAMIC_EXTENT', True))}")
            logger.log(f"use_despl_pseudo = {bool(getattr(cfg, 'USE_DESPL_PSEUDO', False))}")
            logger.log(f"use_despl_light_cache = {bool(getattr(cfg, 'USE_DESPL_LIGHT_CACHE', False))}")
            logger.log(f"use_despl_paper_cache = {bool(getattr(cfg, 'USE_DESPL_PAPER_CACHE', False))}")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
            logger.log(f"OEM_STAGE1_END = {int(getattr(cfg, 'OEM_STAGE1_END', 3))}")
            logger.log(f"OEM_STAGE2_START = {int(getattr(cfg, 'OEM_STAGE2_START', 4))}")
            logger.log(f"OEM_STAGE2_END = {int(getattr(cfg, 'OEM_STAGE2_END', 15))}")
            logger.log(f"OEM_DYN_POS_STAGE2_START = {float(getattr(cfg, 'OEM_DYN_POS_STAGE2_START', 0.0)):.6f}")
            logger.log(f"OEM_DYN_POS_STAGE2_END = {float(getattr(cfg, 'OEM_DYN_POS_STAGE2_END', 0.35)):.6f}")
            logger.log(f"OEM_DYN_BG_STAGE2_START = {float(getattr(cfg, 'OEM_DYN_BG_STAGE2_START', 0.0)):.6f}")
            logger.log(f"OEM_DYN_BG_STAGE2_END = {float(getattr(cfg, 'OEM_DYN_BG_STAGE2_END', 0.12)):.6f}")
            logger.log(f"OEM_STAGE3_START = {int(getattr(cfg, 'OEM_STAGE3_START', 16))}")
            logger.log(f"OEM_DYN_POS_STAGE3 = {float(getattr(cfg, 'OEM_DYN_POS_STAGE3', 0.35)):.6f}")
            logger.log(f"OEM_DYN_BG_STAGE3 = {float(getattr(cfg, 'OEM_DYN_BG_STAGE3', 0.12)):.6f}")
            logger.log(f"OEM_SEED_LAMBDA_FG = {float(getattr(cfg, 'OEM_SEED_LAMBDA_FG', 1.0)):.6f}")
            logger.log(f"OEM_SEED_LAMBDA_BG = {float(getattr(cfg, 'OEM_SEED_LAMBDA_BG', 1.0)):.6f}")
            logger.log(f"OEM_SEED_LAMBDA_FG_FALLBACK = {float(getattr(cfg, 'OEM_SEED_LAMBDA_FG_FALLBACK', 0.35)):.6f}")
            logger.log(f"OEM_PROTO_FEATURE_SOURCE = {getattr(cfg, 'OEM_PROTO_FEATURE_SOURCE', 'dino37')}")
            logger.log(f"OEM_PROTO_L2_NORM = {bool(getattr(cfg, 'OEM_PROTO_L2_NORM', True))}")
            logger.log(f"OEM_PROTO_MIN_FG_PIXELS = {int(getattr(cfg, 'OEM_PROTO_MIN_FG_PIXELS', 4))}")
            logger.log(f"OEM_PROTO_MIN_BG_PIXELS = {int(getattr(cfg, 'OEM_PROTO_MIN_BG_PIXELS', 16))}")
            logger.log(f"OEM_TEACHER_POS_THRESH = {float(getattr(cfg, 'OEM_TEACHER_POS_THRESH', 0.60)):.6f}")
            logger.log(f"OEM_TEACHER_BG_THRESH = {float(getattr(cfg, 'OEM_TEACHER_BG_THRESH', 0.20)):.6f}")
            logger.log(f"OEM_PROTO_POS_MIN = {float(getattr(cfg, 'OEM_PROTO_POS_MIN', 0.05)):.6f}")
            logger.log(f"OEM_PROTO_POS_QUANTILE = {float(getattr(cfg, 'OEM_PROTO_POS_QUANTILE', 0.80)):.6f}")
            logger.log(f"OEM_PROTO_BG_MAX = {float(getattr(cfg, 'OEM_PROTO_BG_MAX', -0.05)):.6f}")
            logger.log(f"OEM_PROTO_BG_QUANTILE = {float(getattr(cfg, 'OEM_PROTO_BG_QUANTILE', 0.20)):.6f}")
            logger.log(f"OEM_POS_CAP_RATIO = {float(getattr(cfg, 'OEM_POS_CAP_RATIO', 0.05)):.6f}")
            logger.log(f"OEM_POS_CAP_TO_FG = {float(getattr(cfg, 'OEM_POS_CAP_TO_FG', 0.75)):.6f}")
            logger.log(f"OEM_BG_CAP_RATIO = {float(getattr(cfg, 'OEM_BG_CAP_RATIO', 0.10)):.6f}")
            logger.log(f"OEM_BG_TO_POS_MAX = {float(getattr(cfg, 'OEM_BG_TO_POS_MAX', 3.0)):.6f}")
            logger.log(f"OEM_BG_CAP_IF_NO_POS_RATIO = {float(getattr(cfg, 'OEM_BG_CAP_IF_NO_POS_RATIO', 0.03)):.6f}")
            logger.log(f"OEM_DYN_POS_TARGET_MODE = {getattr(cfg, 'OEM_DYN_POS_TARGET_MODE', 'teacher_soft')}")
            logger.log(f"OEM_USE_DYNAMIC_ON_FINAL = {bool(getattr(cfg, 'OEM_USE_DYNAMIC_ON_FINAL', True))}")
            logger.log(f"OEM_USE_DYNAMIC_ON_COARSE = {bool(getattr(cfg, 'OEM_USE_DYNAMIC_ON_COARSE', False))}")
            logger.log(f"OEM_USE_DYNAMIC_ON_BASE = {bool(getattr(cfg, 'OEM_USE_DYNAMIC_ON_BASE', False))}")
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
            logger.log(f"USE_TADR_ROUTER = {bool(getattr(cfg, 'USE_TADR_ROUTER', False))}")
            if use_tadr_router(cfg):
                logger.log(f"TADR_ROUTER_IN_CHANNELS = {int(getattr(cfg, 'TADR_ROUTER_IN_CHANNELS', 4))}")
                logger.log(f"TADR_ROUTER_HIDDEN = {int(getattr(cfg, 'TADR_ROUTER_HIDDEN', 16))}")
                logger.log(f"TADR_ROUTER_NUM_LAYERS = {int(getattr(cfg, 'TADR_ROUTER_NUM_LAYERS', 2))}")
                logger.log(f"TADR_ROUTER_ACT = {getattr(cfg, 'TADR_ROUTER_ACT', 'gelu')}")
                logger.log(f"TADR_USE_COARSE_PROB = {bool(getattr(cfg, 'TADR_USE_COARSE_PROB', True))}")
                logger.log(f"TADR_USE_UNCERTAINTY = {bool(getattr(cfg, 'TADR_USE_UNCERTAINTY', True))}")
                logger.log(f"TADR_USE_SOBEL = {bool(getattr(cfg, 'TADR_USE_SOBEL', True))}")
                logger.log(f"TADR_USE_COARSE_BOUNDARY = {bool(getattr(cfg, 'TADR_USE_COARSE_BOUNDARY', True))}")
                logger.log(f"TADR_ROUTER_INIT_BIAS = {float(getattr(cfg, 'TADR_ROUTER_INIT_BIAS', 2.0)):.6f}")
                logger.log(f"TADR_ROUTER_ZERO_INIT_OUT = {bool(getattr(cfg, 'TADR_ROUTER_ZERO_INIT_OUT', True))}")
                logger.log(f"TADR_ROUTER_DETACH_INPUTS = {bool(getattr(cfg, 'TADR_ROUTER_DETACH_INPUTS', True))}")
                logger.log(f"TADR_ROUTER_MIN = {float(getattr(cfg, 'TADR_ROUTER_MIN', 0.0)):.6f}")
                logger.log(f"TADR_ROUTER_MAX = {float(getattr(cfg, 'TADR_ROUTER_MAX', 1.0)):.6f}")
        logger.log(f"use_multi_level_feature = {use_multi_level_feature(cfg)}")
        logger.log(f"multi_level_layers = {list(getattr(cfg, 'MULTI_LEVEL_LAYERS', []))}")
        logger.log(f"multi_level_feature_dtype = {getattr(cfg, 'MULTI_LEVEL_FEATURE_DTYPE', 'float32')}")
        logger.log(f"ml_feature_preflight_mode = {getattr(cfg, 'ML_FEATURE_PREFLIGHT_MODE', 'sample')}")
        logger.log(f"ml_feature_preflight_samples = {int(getattr(cfg, 'ML_FEATURE_PREFLIGHT_SAMPLES', 32))}")
        logger.log(f"use_multi_view_feature = {use_multi_view_feature(cfg)}")
        logger.log(f"multi_view_types = {multi_view_types(cfg)}")
        logger.log(f"use_view_consistency = {bool(getattr(cfg, 'USE_VIEW_CONSISTENCY', False))}")
        logger.log(f"hflip_feature_cache_root = {getattr(cfg, 'HFLIP_FEATURE_CACHE_ROOT', '')}")
        logger.log(f"lambda_view_max = {float(getattr(cfg, 'LAMBDA_VIEW_MAX', 0.0)):.6f}")
        logger.log(f"view_consistency_type = {getattr(cfg, 'VIEW_CONSISTENCY_TYPE', 'l1')}")
        logger.log(f"view_conf_source = {getattr(cfg, 'VIEW_CONF_SOURCE', 'despl_core')}")
        logger.log(f"view_fg_thresh = {float(getattr(cfg, 'VIEW_FG_THRESH', 0.8)):.6f}")
        logger.log(f"view_bg_thresh = {float(getattr(cfg, 'VIEW_BG_THRESH', 0.2)):.6f}")
        logger.log(f"view_boundary_weight = {float(getattr(cfg, 'VIEW_BOUNDARY_WEIGHT', 0.0)):.6f}")
        logger.log(f"view_warmup_epoch = {int(getattr(cfg, 'VIEW_WARMUP_EPOCH', 6))}")
        logger.log(f"view_ramp_start_epoch = {int(getattr(cfg, 'VIEW_RAMP_START_EPOCH', 7))}")
        logger.log(f"view_ramp_end_epoch = {int(getattr(cfg, 'VIEW_RAMP_END_EPOCH', 15))}")
        logger.log(f"view_after_reset_scale = {float(getattr(cfg, 'VIEW_AFTER_RESET_SCALE', 0.0)):.6f}")
        logger.log(f"mv_loss_debug = {bool(getattr(cfg, 'MV_LOSS_DEBUG', False))}")
        logger.log(f"USE_PROTO_CONTRAST = {bool(getattr(cfg, 'USE_PROTO_CONTRAST', False))}")
        logger.log(f"LAMBDA_PROTO_MAX = {float(getattr(cfg, 'LAMBDA_PROTO_MAX', 0.0)):.6f}")
        logger.log(f"PROTO_MODE = {getattr(cfg, 'PROTO_MODE', 'global')}")
        logger.log(f"PROTO_FEATURE_SOURCE = {getattr(cfg, 'PROTO_FEATURE_SOURCE', 'dagp_semantic')}")
        logger.log(f"PROTO_USE_PROJ_HEAD = {bool(getattr(cfg, 'PROTO_USE_PROJ_HEAD', True))}")
        logger.log(f"PROTO_PROJ_HIDDEN = {int(getattr(cfg, 'PROTO_PROJ_HIDDEN', 64))}")
        logger.log(f"PROTO_PROJ_DIM = {int(getattr(cfg, 'PROTO_PROJ_DIM', 32))}")
        logger.log(f"PROTO_PROJ_ACT = {getattr(cfg, 'PROTO_PROJ_ACT', 'gelu')}")
        logger.log(f"PROTO_CORE_MODE = {getattr(cfg, 'PROTO_CORE_MODE', 'despl_pred_agree')}")
        logger.log(f"PROTO_FG_THRESH = {float(getattr(cfg, 'PROTO_FG_THRESH', 0.90)):.6f}")
        logger.log(f"PROTO_BG_THRESH = {float(getattr(cfg, 'PROTO_BG_THRESH', 0.10)):.6f}")
        logger.log(f"PROTO_PRED_FG_THRESH = {float(getattr(cfg, 'PROTO_PRED_FG_THRESH', 0.60)):.6f}")
        logger.log(f"PROTO_PRED_BG_THRESH = {float(getattr(cfg, 'PROTO_PRED_BG_THRESH', 0.40)):.6f}")
        logger.log(f"PROTO_PRED_BG_LOW = {float(getattr(cfg, 'PROTO_PRED_BG_LOW', 0.20)):.6f}")
        logger.log(f"PROTO_PRED_BG_HIGH = {float(getattr(cfg, 'PROTO_PRED_BG_HIGH', 0.60)):.6f}")
        logger.log(f"PROTO_EASY_BG_THRESH = {float(getattr(cfg, 'PROTO_EASY_BG_THRESH', 0.20)):.6f}")
        logger.log(f"PROTO_USE_HARD_BG_ONLY = {bool(getattr(cfg, 'PROTO_USE_HARD_BG_ONLY', True))}")
        logger.log(f"PROTO_USE_BG_RING = {bool(getattr(cfg, 'PROTO_USE_BG_RING', True))}")
        logger.log(f"PROTO_BG_RING_RADIUS = {int(getattr(cfg, 'PROTO_BG_RING_RADIUS', 3))}")
        logger.log(f"PROTO_USE_DISAGREE_MAP = {bool(getattr(cfg, 'PROTO_USE_DISAGREE_MAP', True))}")
        logger.log(f"PROTO_DISAGREE_THRESH = {float(getattr(cfg, 'PROTO_DISAGREE_THRESH', 0.15)):.6f}")
        logger.log(
            f"PROTO_USE_NDR_RESIDUAL_FOR_HARD = {bool(getattr(cfg, 'PROTO_USE_NDR_RESIDUAL_FOR_HARD', True))}"
        )
        logger.log(f"PROTO_NDR_RESIDUAL_Q = {float(getattr(cfg, 'PROTO_NDR_RESIDUAL_Q', 0.80)):.6f}")
        logger.log(f"PROTO_MIN_FG_PIXELS = {int(getattr(cfg, 'PROTO_MIN_FG_PIXELS', 16))}")
        logger.log(f"PROTO_MIN_BG_PIXELS = {int(getattr(cfg, 'PROTO_MIN_BG_PIXELS', 128))}")
        logger.log(f"PROTO_MAX_PIXELS_PER_CLASS = {int(getattr(cfg, 'PROTO_MAX_PIXELS_PER_CLASS', 256))}")
        logger.log(f"PROTO_MAX_FG_PIXELS = {int(getattr(cfg, 'PROTO_MAX_FG_PIXELS', 128))}")
        logger.log(f"PROTO_MAX_BG_PIXELS = {int(getattr(cfg, 'PROTO_MAX_BG_PIXELS', 256))}")
        logger.log(f"PROTO_TAU = {float(getattr(cfg, 'PROTO_TAU', 0.10)):.6f}")
        logger.log(f"PROTO_PIXEL_LOSS_MODE = {getattr(cfg, 'PROTO_PIXEL_LOSS_MODE', 'softplus')}")
        logger.log(f"PROTO_PIXEL_MARGIN = {float(getattr(cfg, 'PROTO_PIXEL_MARGIN', 0.20)):.6f}")
        logger.log(f"PROTO_SEP_MARGIN = {float(getattr(cfg, 'PROTO_SEP_MARGIN', 0.20)):.6f}")
        logger.log(f"PROTO_ALIGN_WEIGHT = {float(getattr(cfg, 'PROTO_ALIGN_WEIGHT', 1.0)):.6f}")
        logger.log(f"PROTO_SEP_WEIGHT = {float(getattr(cfg, 'PROTO_SEP_WEIGHT', 0.5)):.6f}")
        logger.log(f"PROTO_PIXEL_WEIGHT = {float(getattr(cfg, 'PROTO_PIXEL_WEIGHT', 1.0)):.6f}")
        logger.log(f"PROTO_PIXEL_FG_WEIGHT = {float(getattr(cfg, 'PROTO_PIXEL_FG_WEIGHT', 0.30)):.6f}")
        logger.log(f"PROTO_PIXEL_BG_WEIGHT = {float(getattr(cfg, 'PROTO_PIXEL_BG_WEIGHT', 1.0)):.6f}")
        logger.log(f"PROTO_WARMUP_EPOCH = {int(getattr(cfg, 'PROTO_WARMUP_EPOCH', 6))}")
        logger.log(f"PROTO_RAMP_START_EPOCH = {int(getattr(cfg, 'PROTO_RAMP_START_EPOCH', 7))}")
        logger.log(f"PROTO_RAMP_END_EPOCH = {int(getattr(cfg, 'PROTO_RAMP_END_EPOCH', 15))}")
        logger.log(f"PROTO_AFTER_RESET_SCALE = {float(getattr(cfg, 'PROTO_AFTER_RESET_SCALE', 0.0)):.6f}")
        logger.log(f"PROTO_DETACH_MASK = {bool(getattr(cfg, 'PROTO_DETACH_MASK', True))}")
        logger.log(
            f"PROTO_DETACH_PIXEL_PROTOTYPE = {bool(getattr(cfg, 'PROTO_DETACH_PIXEL_PROTOTYPE', True))}"
        )
        logger.log(f"PROTO_SKIP_INVALID = {bool(getattr(cfg, 'PROTO_SKIP_INVALID', True))}")
        logger.log(f"MV_PROTO_DEBUG = {bool(getattr(cfg, 'MV_PROTO_DEBUG', False))}")
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
        logger.log(f"use_dabe_pseudo = {bool(getattr(cfg, 'USE_DABE_PSEUDO', False))}")
        logger.log(f"dabe_version = {getattr(cfg, 'DABE_VERSION', 'v2')}")
        logger.log(f"dabe_pseudo_root = {getattr(cfg, 'DABE_PSEUDO_ROOT', '')}")
        logger.log(f"use_dabe_pu = {bool(getattr(cfg, 'USE_DABE_PU', False))}")
        logger.log(f"dabe_pu_version = {getattr(cfg, 'DABE_PU_VERSION', '')}")
        logger.log(f"dabe_pu_root = {getattr(cfg, 'DABE_PU_ROOT', '')}")
        logger.log(f"USE_DABE_AWARE_LOSS = {bool(getattr(cfg, 'USE_DABE_AWARE_LOSS', False))}")
        logger.log(f"DABE_AWARE_USE_TRIMAP = {bool(getattr(cfg, 'DABE_AWARE_USE_TRIMAP', False))}")
        logger.log(f"DABE_AWARE_CORE_LOCK = {bool(getattr(cfg, 'DABE_AWARE_CORE_LOCK', False))}")
        logger.log(f"DABE_AWARE_WEIGHTED_BCE = {bool(getattr(cfg, 'DABE_AWARE_WEIGHTED_BCE', False))}")
        logger.log(f"DABE_FG_CORE_WEIGHT = {float(getattr(cfg, 'DABE_FG_CORE_WEIGHT', 2.0)):.6f}")
        logger.log(f"DABE_BG_CORE_WEIGHT = {float(getattr(cfg, 'DABE_BG_CORE_WEIGHT', 1.2)):.6f}")
        logger.log(f"DABE_UNCERTAIN_WEIGHT = {float(getattr(cfg, 'DABE_UNCERTAIN_WEIGHT', 0.20)):.6f}")
        logger.log(f"DABE_EVIDENCE_WEIGHT_SCALE = {float(getattr(cfg, 'DABE_EVIDENCE_WEIGHT_SCALE', 0.40)):.6f}")
        logger.log(f"DABE_CORE_THRESH = {float(getattr(cfg, 'DABE_CORE_THRESH', 0.5)):.6f}")
        logger.log(f"USE_DABE_TVERSKY_LOSS = {bool(getattr(cfg, 'USE_DABE_TVERSKY_LOSS', False))}")
        logger.log(f"DABE_TVERSKY_WEIGHT = {float(getattr(cfg, 'DABE_TVERSKY_WEIGHT', 0.30)):.6f}")
        logger.log(f"DABE_TVERSKY_ALPHA_FP = {float(getattr(cfg, 'DABE_TVERSKY_ALPHA_FP', 0.30)):.6f}")
        logger.log(f"DABE_TVERSKY_BETA_FN = {float(getattr(cfg, 'DABE_TVERSKY_BETA_FN', 0.70)):.6f}")
        logger.log(f"USE_DABE_AREA_GUARD = {bool(getattr(cfg, 'USE_DABE_AREA_GUARD', False))}")
        logger.log(f"DABE_AREA_GUARD_WEIGHT = {float(getattr(cfg, 'DABE_AREA_GUARD_WEIGHT', 0.02)):.6f}")
        logger.log(f"DABE_AREA_GUARD_RATIO = {float(getattr(cfg, 'DABE_AREA_GUARD_RATIO', 0.85)):.6f}")
        logger.log(f"DABE_AREA_GUARD_START_EPOCH = {int(getattr(cfg, 'DABE_AREA_GUARD_START_EPOCH', 7))}")
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
        if str(getattr(cfg, "P_INIT_MODE", "")) in {"dabe_only", "dabe_gc_only"}:
            logger.log("pseudo final candidate = p_dabe_68")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "dabe_pu_v11":
            logger.log("pseudo final candidate = target_soft_68")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
        if str(getattr(cfg, "P_INIT_MODE", "")) in {
            "dabe_pu_v11_desplsched",
            "dabe_pu_v11_desplsched_exactreset",
            "dabe_pu_v11_desplsched_A1_keepteacher_lowlr",
            "dabe_pu_v11_desplsched_A2_resetteacher_highlr",
            "dabe_pu_v11_desplsched_softteacher",
            "dabe_pu_v11_desplsched_dabehard",
        }:
            if get_dabe_pu_despl_static_target_mode(cfg) == "hard_from_target_soft":
                logger.log("pseudo final candidate = target_hard_from_target_soft_68")
            else:
                logger.log("pseudo final candidate = target_soft_68")
            if get_dabe_pu_despl_teacher_target_mode(cfg) == "soft_prob":
                logger.log("p_init_formula = static weighted BCE(target_soft_68, weight_map_68) + full teacher soft BCE")
            elif get_dabe_pu_despl_static_target_mode(cfg) == "hard_from_target_soft":
                logger.log("p_init_formula = static weighted BCE(target_hard_from_target_soft_68, weight_map_68) + full teacher binary BCE")
            else:
                logger.log("p_init_formula = static weighted BCE(target_soft_68, weight_map_68) + full teacher binary BCE")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")
        if str(getattr(cfg, "P_INIT_MODE", "")) == "dabe_pu_v11_oem":
            logger.log("pseudo final candidate = DABE-PU seed masks + OEM dynamic extent")
            logger.log("use_fixed_in_pseudo = False")
            logger.log("fixed_used_for_training = False")

        use_qra = bool(getattr(cfg, "USE_QRA", False))
        use_ccr = bool(getattr(cfg, "USE_CCR", False))
        use_drepp = bool(getattr(cfg, "USE_DREPP", False))
        use_dabe = bool(getattr(cfg, "USE_DABE_PSEUDO", False))
        use_dabe_pu = bool(getattr(cfg, "USE_DABE_PU", False))
        use_dabe_aware = use_dabe_aware_loss(cfg)
        use_despl = bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        use_despl_paper = use_despl and bool(getattr(cfg, "USE_DESPL_PAPER_CACHE", False))
        use_dre_safe = use_despl and bool(getattr(cfg, "USE_DRE_SAFE_PRIOR", False))
        use_despl_light = use_despl and bool(getattr(cfg, "USE_DESPL_LIGHT_CACHE", False))
        use_anchor_pbce = use_despl and bool(getattr(cfg, "USE_DESPL_ANCHOR_PBCE", False))
        use_pure_despl = use_despl and bool(getattr(cfg, "USE_PURE_DESPL_SUPERVISION", False))
        use_fast_teacher_fusion = bool(getattr(cfg, "USE_FAST_TEACHER_FUSION", False))
        use_ml_feature = use_multi_level_feature(cfg)
        use_hflip_mv = use_hflip_view(cfg)
        use_proto = use_proto_contrast(cfg)
        use_gkd_lite = use_despl and is_gkd_enabled(cfg)
        use_gkd_v3 = use_gkd_lite and is_gkd_v3_enabled(cfg)
        teacher_fusion_mode = str(getattr(cfg, "TEACHER_FUSION_MODE", "default")).lower()
        use_dabe_oem = use_dabe_pu and (
            teacher_fusion_mode == "dabe_pu_oem" or bool(getattr(cfg, "USE_DABE_OEM", False))
        )
        use_dabe_pu_balanced_v2 = use_dabe_pu and teacher_fusion_mode == "dabe_pu_balanced_v2"
        use_dabe_pu_despl_sched = use_dabe_pu and teacher_fusion_mode == "dabe_pu_despl_sched"
        use_rast = bool(getattr(cfg, "USE_RAST", False)) and use_dabe_pu_despl_sched
        use_hbns_lite = bool(getattr(cfg, "USE_HBNS_LITE", False)) and use_dabe_pu_despl_sched
        use_epr_pos = bool(getattr(cfg, "USE_EPR_POS", False)) and use_dabe_pu_despl_sched
        use_esa_asym = bool(getattr(cfg, "USE_ESA_ASYM", False)) and use_dabe_pu_despl_sched
        use_esa_diagnostic = bool(getattr(cfg, "USE_ESA_DIAGNOSTIC", False)) and use_dabe_pu_despl_sched
        dabe_pu_despl_teacher_target_mode = (
            get_dabe_pu_despl_teacher_target_mode(cfg) if use_dabe_pu_despl_sched else "binary"
        )
        dabe_pu_despl_static_target_mode = (
            get_dabe_pu_despl_static_target_mode(cfg) if use_dabe_pu_despl_sched else "soft"
        )
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
        if use_hflip_mv and use_ml_feature:
            raise RuntimeError("HFlip multi-view consistency currently supports single-level cached DINO features only.")
        if use_hflip_mv and not (bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)) or use_proto):
            raise RuntimeError(
                "USE_MULTI_VIEW_FEATURE with hflip requires USE_VIEW_CONSISTENCY=True or USE_PROTO_CONTRAST=True."
            )
        if use_hflip_mv and str(getattr(cfg, "VIEW_CONF_SOURCE", "despl_core")).lower() != "despl_core":
            raise RuntimeError("HFlip view consistency currently supports VIEW_CONF_SOURCE='despl_core' only.")
        if use_proto:
            proto_mode = str(getattr(cfg, "PROTO_MODE", "global")).lower()
            if proto_mode not in {"", "global", "hard_selective"}:
                raise RuntimeError(f"Unsupported PROTO_MODE: {proto_mode}")
            if str(getattr(cfg, "PROTO_FEATURE_SOURCE", "dagp_semantic")) != "dagp_semantic":
                raise RuntimeError("MVFlip-Proto currently supports PROTO_FEATURE_SOURCE='dagp_semantic' only.")
            if not bool(getattr(cfg, "PROTO_USE_PROJ_HEAD", True)):
                raise RuntimeError("MVFlip-Proto currently requires PROTO_USE_PROJ_HEAD=True.")
            if proto_mode == "hard_selective":
                if str(getattr(cfg, "PROTO_CORE_MODE", "despl_pred_agree_hard")).lower() != "despl_pred_agree_hard":
                    raise RuntimeError("PROTO_MODE='hard_selective' requires PROTO_CORE_MODE='despl_pred_agree_hard'.")
                if str(getattr(cfg, "PROTO_PIXEL_LOSS_MODE", "hard_margin")).lower() != "hard_margin":
                    raise RuntimeError("PROTO_MODE='hard_selective' requires PROTO_PIXEL_LOSS_MODE='hard_margin'.")
                if not bool(getattr(cfg, "PROTO_USE_HARD_BG_ONLY", True)):
                    raise RuntimeError("PROTO_MODE='hard_selective' requires PROTO_USE_HARD_BG_ONLY=True.")
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
        if use_dabe_aware and gkd_mode != "off":
            raise RuntimeError("USE_DABE_AWARE_LOSS=True requires GKD_MODE=off.")

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
        if use_hflip_mv:
            _, hflip_reason = check_hflip_feature_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] feature_hflip:train ready | {hflip_reason}")
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
            if use_dabe:
                _, dabe_reason = check_dabe_pseudo_cache(
                    cfg,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] DABE pseudo cache ready | {dabe_reason}")
            if use_dabe_pu:
                _, dabe_pu_reason = check_dabe_pu_cache(
                    cfg,
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] DABE-PU cache ready | {dabe_pu_reason}")
        elif use_dabe_pu:
            _, dabe_pu_reason = check_dabe_pu_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] DABE-PU cache ready | {dabe_pu_reason}")
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
        tadr_first_batch_logged = False
        mvflip_first_batch_logged = False
        mvproto_first_batch_logged = False
        rast_first_batch_logged = False
        hbns_first_batch_logged = False
        epr_first_batch_logged = False
        esa_asym_first_batch_logged = False
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
            if reset_enabled and not is_after_epoch_finetune_reset(cfg) and epoch == reset_epoch:
                optimizer, scheduler, global_step, lr_floor_activated_logged = apply_finetune_reset(
                    logger,
                    cfg,
                    epoch,
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    global_step,
                    lr_floor_activated_logged,
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
            dabe_p_area_sum = 0.0
            dabe_num_samples = 0
            dabe_aware_fg_core_area_sum = 0.0
            dabe_aware_bg_core_area_sum = 0.0
            dabe_aware_uncertain_area_sum = 0.0
            dabe_aware_evidence_sum = 0.0
            dabe_aware_target_area_sum = 0.0
            dabe_aware_weight_map_sum = 0.0
            dabe_aware_loss_final_bce_sum = 0.0
            dabe_aware_loss_tversky_sum = 0.0
            dabe_aware_loss_area_guard_sum = 0.0
            dabe_aware_stat_batches = 0
            dabe_pu_target_mean_sum = 0.0
            dabe_pu_weight_mean_sum = 0.0
            dabe_pu_fg_core_mean_sum = 0.0
            dabe_pu_fg_fallback_mean_sum = 0.0
            dabe_pu_bg_core_mean_sum = 0.0
            dabe_pu_extent_mean_sum = 0.0
            dabe_pu_unknown_mean_sum = 0.0
            dabe_pu_static_final_loss_sum = 0.0
            dabe_pu_static_coarse_loss_sum = 0.0
            dabe_pu_static_base_loss_sum = 0.0
            dabe_pu_static_group_loss_sum = 0.0
            dabe_pu_teacher_final_loss_sum = 0.0
            dabe_pu_teacher_coarse_loss_sum = 0.0
            dabe_pu_teacher_base_loss_sum = 0.0
            dabe_pu_teacher_group_loss_sum = 0.0
            dabe_pu_teacher_conf_ratio_sum = 0.0
            dabe_pu_teacher_fg_ratio_sum = 0.0
            dabe_pu_teacher_bg_ratio_sum = 0.0
            dabe_pu_stat_batches = 0
            dabe_pu_target_hard_area_sum = 0.0
            dabe_pu_bal_static_fg_loss_sum = 0.0
            dabe_pu_bal_static_fg_fallback_loss_sum = 0.0
            dabe_pu_bal_static_bg_loss_sum = 0.0
            dabe_pu_bal_static_extent_loss_sum = 0.0
            dabe_pu_bal_teacher_fg_loss_sum = 0.0
            dabe_pu_bal_teacher_bg_loss_sum = 0.0
            dabe_pu_bal_teacher_fg_raw_ratio_sum = 0.0
            dabe_pu_bal_teacher_bg_raw_ratio_sum = 0.0
            dabe_pu_bal_teacher_bg_capped_ratio_sum = 0.0
            dabe_pu_bal_teacher_conf_capped_ratio_sum = 0.0
            dabe_pu_bal_stat_batches = 0
            oem_lambda_dyn_pos_sum = 0.0
            oem_lambda_dyn_bg_sum = 0.0
            oem_seed_fg_area_sum = 0.0
            oem_seed_bg_area_sum = 0.0
            oem_extent_area_sum = 0.0
            oem_unknown_area_sum = 0.0
            oem_loss_seed_fg_sum = 0.0
            oem_loss_seed_fg_fallback_sum = 0.0
            oem_loss_seed_bg_sum = 0.0
            oem_loss_seed_final_sum = 0.0
            oem_loss_seed_coarse_sum = 0.0
            oem_loss_seed_base_sum = 0.0
            oem_loss_seed_group_sum = 0.0
            oem_loss_dyn_pos_sum = 0.0
            oem_loss_dyn_bg_sum = 0.0
            oem_pos_raw_ratio_sum = 0.0
            oem_pos_capped_ratio_sum = 0.0
            oem_bg_raw_ratio_sum = 0.0
            oem_bg_capped_ratio_sum = 0.0
            oem_skip_no_fg_proto_sum = 0
            oem_skip_no_bg_proto_sum = 0
            oem_skip_no_pos_region_sum = 0
            oem_proto_delta_mean_sum = 0.0
            oem_proto_delta_min = None
            oem_proto_delta_max = None
            oem_teacher_prob_37_mean_sum = 0.0
            oem_teacher_prob_37_fg_seed_sum = 0.0
            oem_teacher_prob_37_bg_seed_sum = 0.0
            oem_teacher_prob_37_extent_sum = 0.0
            oem_stat_batches = 0
            rast_scale_sum = 0.0
            rast_pre_reset_scale_sum = 0.0
            rast_post_reset_scale_sum = 0.0
            rast_scale_effective_sum = 0.0
            rast_fg_core_area_sum = 0.0
            rast_bg_core_area_sum = 0.0
            rast_extent_area_sum = 0.0
            rast_unknown_area_sum = 0.0
            rast_fg_conflict_ratio_sum = 0.0
            rast_bg_conflict_ratio_sum = 0.0
            rast_teacher_map_mean_sum = 0.0
            rast_teacher_map_min = None
            rast_teacher_map_max = None
            rast_teacher_map_fg_core_mean_sum = 0.0
            rast_teacher_map_bg_core_mean_sum = 0.0
            rast_teacher_map_extent_mean_sum = 0.0
            rast_teacher_map_unknown_mean_sum = 0.0
            rast_stat_batches = 0
            hbns_scale_sum = 0.0
            hbns_lambda_sum = 0.0
            hbns_hard_bg_ratio_sum = 0.0
            hbns_hard_bg_raw_ratio_sum = 0.0
            hbns_hard_bg_ratio_bg_core_sum = 0.0
            hbns_hard_bg_ratio_low_target_sum = 0.0
            hbns_hard_bg_ratio_unknown_bg_like_sum = 0.0
            hbns_hard_bg_pixels_mean_sum = 0.0
            hbns_loss_final_sum = 0.0
            hbns_loss_coarse_sum = 0.0
            hbns_loss_base_sum = 0.0
            hbns_loss_sum = 0.0
            hbns_stat_batches = 0
            epr_scale_sum = 0.0
            epr_lambda_sum = 0.0
            epr_pos_ratio_sum = 0.0
            epr_pos_raw_ratio_sum = 0.0
            epr_pos_pixels_mean_sum = 0.0
            epr_valid_image_ratio_sum = 0.0
            epr_margin_mean_sum = 0.0
            epr_margin_min = None
            epr_margin_max = None
            epr_margin_pos_mean_sum = 0.0
            epr_teacher_conf_pos_mean_sum = 0.0
            epr_extent_area_sum = 0.0
            epr_unknown_overlap_ratio_sum = 0.0
            epr_loss_final_sum = 0.0
            epr_loss_coarse_sum = 0.0
            epr_loss_base_sum = 0.0
            epr_loss_sum = 0.0
            epr_stat_batches = 0
            esa_asym_scale_sum = 0.0
            esa_margin_mean_sum = 0.0
            esa_margin_min = None
            esa_margin_max = None
            esa_margin_extent_mean_sum = 0.0
            esa_margin_extent_teacher_fg_mean_sum = 0.0
            esa_margin_extent_teacher_bg_mean_sum = 0.0
            esa_extent_teacher_fg_ratio_sum = 0.0
            esa_extent_teacher_bg_ratio_sum = 0.0
            esa_extent_teacher_bg_fg_like_ratio_sum = 0.0
            esa_extent_teacher_bg_ambig_ratio_sum = 0.0
            esa_extent_teacher_bg_bg_like_ratio_sum = 0.0
            esa_teacher_map_extent_mean_sum = 0.0
            esa_teacher_map_extent_teacher_fg_mean_sum = 0.0
            esa_teacher_map_extent_teacher_bg_mean_sum = 0.0
            esa_teacher_map_extent_teacher_bg_fg_like_mean_sum = 0.0
            esa_teacher_map_extent_teacher_bg_ambig_mean_sum = 0.0
            esa_teacher_map_extent_teacher_bg_bg_like_mean_sum = 0.0
            esa_skipped_no_fg_proto_sum = 0
            esa_skipped_no_bg_proto_sum = 0
            esa_asym_stat_batches = 0
            esa_student_prob_fg_core_sum = 0.0
            esa_student_prob_bg_core_sum = 0.0
            esa_student_prob_extent_sum = 0.0
            esa_student_prob_unknown_sum = 0.0
            esa_teacher_fg_fg_core_sum = 0.0
            esa_teacher_fg_bg_core_sum = 0.0
            esa_teacher_fg_extent_sum = 0.0
            esa_teacher_fg_unknown_sum = 0.0
            esa_teacher_conf_fg_core_sum = 0.0
            esa_teacher_conf_bg_core_sum = 0.0
            esa_teacher_conf_extent_sum = 0.0
            esa_teacher_conf_unknown_sum = 0.0
            esa_teacher_loss_fg_core_sum = 0.0
            esa_teacher_loss_bg_core_sum = 0.0
            esa_teacher_loss_extent_sum = 0.0
            esa_teacher_loss_unknown_sum = 0.0
            esa_stat_batches = 0
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
            teacher_prob_min = None
            teacher_prob_max = None
            teacher_soft_target_mean_sum = 0.0
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
            tadr_router_mean_sum = 0.0
            tadr_router_min = None
            tadr_router_max = None
            tadr_base_gate_mean_sum = 0.0
            tadr_base_gate_min = None
            tadr_base_gate_max = None
            tadr_final_gate_mean_sum = 0.0
            tadr_final_gate_min = None
            tadr_final_gate_max = None
            tadr_stat_batches = 0
            mv_lambda_sum = 0.0
            mv_loss_sum = 0.0
            mv_core_ratio_sum = 0.0
            mv_mean_abs_diff_sum = 0.0
            mv_stat_batches = 0
            proto_lambda_sum = 0.0
            proto_loss_sum = 0.0
            proto_align_sum = 0.0
            proto_sep_sum = 0.0
            proto_pixel_sum = 0.0
            proto_pixel_fg_sum = 0.0
            proto_pixel_bg_sum = 0.0
            proto_valid_ratio_sum = 0.0
            proto_fg_core_ratio_sum = 0.0
            proto_bg_core_ratio_sum = 0.0
            proto_bg_hard_ratio_sum = 0.0
            proto_bg_ring_ratio_sum = 0.0
            proto_bg_disagree_ratio_sum = 0.0
            proto_bg_residual_ratio_sum = 0.0
            proto_hard_fg_ratio_sum = 0.0
            proto_hard_bg_ratio_sum = 0.0
            proto_sep_active_ratio_sum = 0.0
            proto_fg_fallback_ratio_sum = 0.0
            proto_cos_fg_view_sum = 0.0
            proto_cos_bg_view_sum = 0.0
            proto_cos_fg_bg_sum = 0.0
            proto_stat_batches = 0
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
            if use_hflip_training_view(cfg) and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            for iter_idx, batch in enumerate(train_loader):
                pseudo = batch["pseudo"].to(device, non_blocking=True).float()
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                pseudo_68 = F.interpolate(pseudo, size=(cfg.LOSS_SIZE, cfg.LOSS_SIZE), mode="bilinear").float()
                pu_target_soft = None
                pu_weight_map = None
                pu_static_target = None
                pu_static_weight_map = None
                pu_target_hard = None
                pu_fg_core = None
                pu_bg_core = None
                if use_dabe_pu:
                    pu_target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
                    pu_weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
                    hard_thresh = float(getattr(cfg, "DABE_PU_HARD_THRESH", 0.5))
                    pu_target_hard = (pu_target_soft > hard_thresh).float()
                    pu_static_target = pu_target_soft
                    pu_static_weight_map = pu_weight_map
                    if use_dabe_pu_despl_sched:
                        pu_static_target, pu_static_weight_map, _ = build_dabe_pu_despl_static_target(
                            cfg,
                            pu_target_soft,
                            pu_weight_map,
                        )
                    pu_fg_core = batch["pu_fg_core"].to(device, non_blocking=True).float()
                    pu_bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float()

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
                if (
                    use_tadr_router(cfg)
                    and bool(getattr(cfg, "TADR_DEBUG_FIRST_BATCH", True))
                    and not tadr_first_batch_logged
                    and isinstance(student_out, dict)
                ):
                    log_tadr_first_batch(logger, student_out)
                    tadr_first_batch_logged = True
                lambda_view = view_consistency_lambda(cfg, epoch)
                lambda_proto = proto_contrast_lambda(cfg, epoch)
                loss_view = student_logits.sum() * 0.0
                loss_proto = student_logits.sum() * 0.0
                mv_stats = {
                    "loss_raw": 0.0,
                    "core_ratio": 0.0,
                    "mean_abs_diff": 0.0,
                    "weight_mean": 0.0,
                    "hflip_prob_inv": None,
                }
                proto_stats = _proto_zero_stats()
                hflip_model_input = None
                raw_hflip_logits = None
                hflip_logits = None
                hflip_out = None
                should_run_hflip_view = use_hflip_training_view(cfg) and (
                    (
                        use_view_consistency(cfg)
                        and (
                            lambda_view > 0.0
                            or (
                                bool(getattr(cfg, "MV_LOSS_DEBUG", False))
                                and not mvflip_first_batch_logged
                            )
                        )
                    )
                    or (
                        use_proto
                        and (
                            lambda_proto > 0.0
                            or (
                                bool(getattr(cfg, "MV_PROTO_DEBUG", False))
                                and not mvproto_first_batch_logged
                            )
                        )
                    )
                )
                if should_run_hflip_view:
                    hflip_model_input = make_hflip_model_input(cfg, batch, device)
                    hflip_image_68 = make_hflip_image_68(cfg, batch, device)
                    hflip_out = forward_seg_head(
                        student,
                        hflip_model_input,
                        cfg,
                        image_68=hflip_image_68,
                        return_aux=use_proto,
                    )
                    raw_hflip_logits = extract_logits(hflip_out)
                    hflip_logits = resize_logits_for_loss(raw_hflip_logits, cfg)
                    if use_view_consistency(cfg):
                        loss_view, mv_stats = compute_view_consistency_loss(
                            student_logits,
                            hflip_logits,
                            pseudo_68,
                            cfg,
                        )
                        if not bool(torch.isfinite(loss_view).item()):
                            raise RuntimeError("MVFlip view consistency loss is not finite.")
                    if use_proto:
                        loss_proto, proto_stats = compute_proto_contrast_loss(
                            student_out,
                            hflip_out,
                            pseudo_68,
                            cfg,
                        )
                        if not bool(torch.isfinite(loss_proto).item()):
                            raise RuntimeError("MVFlip proto contrast loss is not finite.")
                    if (
                        use_view_consistency(cfg)
                        and bool(getattr(cfg, "MV_LOSS_DEBUG", False))
                        and not mvflip_first_batch_logged
                    ):
                        log_mvflip_first_batch(
                            logger,
                            model_input,
                            hflip_model_input,
                            raw_student_logits,
                            raw_hflip_logits,
                            student_logits,
                            hflip_logits,
                            pseudo_68,
                            mv_stats["hflip_prob_inv"],
                            lambda_view,
                            mv_stats,
                        )
                        mvflip_first_batch_logged = True
                    if (
                        use_proto
                        and bool(getattr(cfg, "MV_PROTO_DEBUG", False))
                        and not mvproto_first_batch_logged
                    ):
                        log_mvproto_first_batch(
                            logger,
                            model_input,
                            hflip_model_input,
                            student_out,
                            hflip_out,
                            pseudo_68,
                            lambda_proto,
                            proto_stats,
                        )
                        mvproto_first_batch_logged = True
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
                    teacher_binary_thresh = 0.5 if use_dabe_pu_despl_sched else float(cfg.THRESHOLD)
                    teacher_binary = (teacher_prob >= teacher_binary_thresh).float()
                    if use_dabe_pu_despl_sched and dabe_pu_despl_teacher_target_mode == "soft_prob":
                        teacher_full_target = teacher_prob.detach()
                    else:
                        teacher_full_target = teacher_binary

                rast_teacher_map_eff = None
                rast_stats = {
                    "rast_scale": 0.0,
                    "rast_pre_reset_scale": 0.0,
                    "rast_post_reset_scale": 0.0,
                    "rast_scale_effective": 0.0,
                    "rast_phase": "off",
                    "rast_post_reset_enable": bool(getattr(cfg, "RAST_POST_RESET_ENABLE", False)),
                    "rast_post_reset_conflict_only": bool(getattr(cfg, "RAST_POST_RESET_CONFLICT_ONLY", False)),
                    "fg_core_area": 0.0,
                    "bg_core_area": 0.0,
                    "extent_area": 0.0,
                    "unknown_area": 0.0,
                    "fg_conflict_ratio": 0.0,
                    "bg_conflict_ratio": 0.0,
                    "teacher_map_mean": 1.0,
                    "teacher_map_min": 1.0,
                    "teacher_map_max": 1.0,
                    "teacher_map_fg_core_mean": 1.0,
                    "teacher_map_bg_core_mean": 1.0,
                    "teacher_map_extent_mean": 1.0,
                    "teacher_map_unknown_mean": 1.0,
                }
                if use_rast:
                    rast_teacher_map_eff, rast_stats = build_rast_teacher_weight_map(
                        cfg,
                        batch,
                        teacher_binary,
                        epoch,
                        device,
                    )
                    if (
                        not rast_first_batch_logged
                        and int(epoch) % max(1, int(getattr(cfg, "RAST_LOG_INTERVAL_EPOCH", 1))) == 0
                    ):
                        logger.log(f"[RAST FirstBatch] USE_RAST = {bool(getattr(cfg, 'USE_RAST', False))}")
                        logger.log(f"[RAST FirstBatch] RAST_VERSION = {getattr(cfg, 'RAST_VERSION', 'v1')}")
                        logger.log(f"[RAST FirstBatch] RAST_POST_RESET_ENABLE = {bool(getattr(cfg, 'RAST_POST_RESET_ENABLE', False))}")
                        logger.log(f"[RAST FirstBatch] RAST_POST_RESET_SCALE = {float(getattr(cfg, 'RAST_POST_RESET_SCALE', 0.30)):.6f}")
                        logger.log(
                            "[RAST FirstBatch] RAST_POST_RESET_CONFLICT_TEACHER_MULT = "
                            f"{float(getattr(cfg, 'RAST_POST_RESET_CONFLICT_TEACHER_MULT', 0.30)):.6f}"
                        )
                        logger.log(f"[RAST FirstBatch] rast_scale = {float(rast_stats['rast_scale']):.8f}")
                        logger.log(
                            "[RAST FirstBatch] rast_pre/post/effective = "
                            f"{float(rast_stats['rast_pre_reset_scale']):.8f}/"
                            f"{float(rast_stats['rast_post_reset_scale']):.8f}/"
                            f"{float(rast_stats['rast_scale_effective']):.8f}"
                        )
                        logger.log(f"[RAST FirstBatch] teacher_map_eff shape = {list(rast_teacher_map_eff.shape)}")
                        logger.log(
                            "[RAST FirstBatch] teacher_map_eff min/mean/max = "
                            f"{float(rast_stats['teacher_map_min']):.6f}/"
                            f"{float(rast_stats['teacher_map_mean']):.6f}/"
                            f"{float(rast_stats['teacher_map_max']):.6f}"
                        )
                        logger.log(
                            "[RAST FirstBatch] fg/bg/extent/unknown area = "
                            f"{float(rast_stats['fg_core_area']):.6f}/"
                            f"{float(rast_stats['bg_core_area']):.6f}/"
                            f"{float(rast_stats['extent_area']):.6f}/"
                            f"{float(rast_stats['unknown_area']):.6f}"
                        )
                        logger.log(
                            "[RAST FirstBatch] fg_conflict/bg_conflict ratio = "
                            f"{float(rast_stats['fg_conflict_ratio']):.6f}/"
                            f"{float(rast_stats['bg_conflict_ratio']):.6f}"
                        )
                        rast_first_batch_logged = True
                    if use_esa_asym and not esa_asym_first_batch_logged:
                        logger.log(f"[ESA-Asym FirstBatch] USE_ESA_ASYM = {bool(getattr(cfg, 'USE_ESA_ASYM', False))}")
                        logger.log(
                            "[ESA-Asym FirstBatch] ESA_ASYM_VERSION = "
                            f"{getattr(cfg, 'ESA_ASYM_VERSION', 'extent_teacher_bg_routing_v1')}"
                        )
                        logger.log(f"[ESA-Asym FirstBatch] esa_asym_scale = {float(rast_stats['esa_asym_scale']):.8f}")
                        logger.log(f"[ESA-Asym FirstBatch] margin_68 shape = {list(rast_stats['esa_margin_shape'])}")
                        logger.log(
                            "[ESA-Asym FirstBatch] margin_68 min/mean/max = "
                            f"{float(rast_stats['esa_margin_min']):.6f}/"
                            f"{float(rast_stats['esa_margin_mean']):.6f}/"
                            f"{float(rast_stats['esa_margin_max']):.6f}"
                        )
                        logger.log(
                            "[ESA-Asym FirstBatch] extent teacher_fg/bg ratio = "
                            f"{float(rast_stats['esa_extent_teacher_fg_ratio']):.6f}/"
                            f"{float(rast_stats['esa_extent_teacher_bg_ratio']):.6f}"
                        )
                        logger.log(
                            "[ESA-Asym FirstBatch] extent teacher_bg fg-like/ambig/bg-like ratio = "
                            f"{float(rast_stats['esa_extent_teacher_bg_fg_like_ratio']):.6f}/"
                            f"{float(rast_stats['esa_extent_teacher_bg_ambig_ratio']):.6f}/"
                            f"{float(rast_stats['esa_extent_teacher_bg_bg_like_ratio']):.6f}"
                        )
                        logger.log(
                            "[ESA-Asym FirstBatch] teacher_map extent fg/bg/fg-like/ambig/bg-like mean = "
                            f"{float(rast_stats['esa_teacher_map_extent_teacher_fg_mean']):.6f}/"
                            f"{float(rast_stats['esa_teacher_map_extent_teacher_bg_mean']):.6f}/"
                            f"{float(rast_stats['esa_teacher_map_extent_teacher_bg_fg_like_mean']):.6f}/"
                            f"{float(rast_stats['esa_teacher_map_extent_teacher_bg_ambig_mean']):.6f}/"
                            f"{float(rast_stats['esa_teacher_map_extent_teacher_bg_bg_like_mean']):.6f}"
                        )
                        esa_asym_first_batch_logged = True

                fixed_target = pseudo_68
                if use_ccr:
                    fixed_target = batch["ccr_p_corr"].to(device, non_blocking=True).float()
                elif use_qra and is_before_finetune_reset(cfg, epoch):
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
                        if is_before_finetune_reset(cfg, epoch):
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
                elif use_dabe_oem:
                    mixed_target = pu_fg_core
                elif use_dabe_pu:
                    mixed_target = pu_target_soft
                elif str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() == "dabe_sticky":
                    mixed_target = fixed_weight * fixed_target + teacher_weight * teacher_binary
                elif is_before_finetune_reset(cfg, epoch):
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
                dabe_aware_target = None
                dabe_aware_weight_map = None
                dabe_aware_stats = None
                if use_dabe_aware:
                    dabe_aware_target, dabe_aware_weight_map, dabe_aware_stats = build_dabe_aware_target_and_weight(
                        cfg,
                        batch,
                        pseudo_68,
                        teacher_binary,
                        fixed_weight,
                        teacher_weight,
                    )
                zero_loss = student_logits.sum() * 0.0
                loss_pu_static_final = zero_loss
                loss_pu_static_coarse = zero_loss
                loss_pu_static_base = zero_loss
                loss_pu_static_group = zero_loss
                loss_pu_teacher_final = zero_loss
                loss_pu_teacher_coarse = zero_loss
                loss_pu_teacher_base = zero_loss
                loss_pu_teacher_group = zero_loss
                pu_teacher_target = None
                pu_teacher_weight_map = None
                pu_static_final_stats = {}
                pu_teacher_final_stats = {
                    "teacher_fg_ratio_raw": 0.0,
                    "teacher_bg_ratio_raw": 0.0,
                    "teacher_bg_ratio_capped": 0.0,
                    "teacher_conf_ratio_capped": 0.0,
                    "loss_teacher_fg": 0.0,
                    "loss_teacher_bg": 0.0,
                }
                pu_teacher_stats = {
                    "teacher_conf_ratio": 0.0,
                    "teacher_fg_ratio": 0.0,
                    "teacher_bg_ratio": 0.0,
                }
                loss_oem_seed_final = zero_loss
                loss_oem_seed_coarse = zero_loss
                loss_oem_seed_base = zero_loss
                loss_oem_seed_group = zero_loss
                loss_oem_dyn_pos = zero_loss
                loss_oem_dyn_bg = zero_loss
                oem_seed_final_stats = {
                    "loss_seed_fg": 0.0,
                    "loss_seed_fg_fallback": 0.0,
                    "loss_seed_bg": 0.0,
                    "seed_fg_area": 0.0,
                    "seed_fg_fallback_area": 0.0,
                    "seed_bg_area": 0.0,
                }
                oem_dynamic_stats = {
                    "loss_dyn_pos": 0.0,
                    "loss_dyn_bg": 0.0,
                    "oem_pos_raw_ratio": 0.0,
                    "oem_pos_capped_ratio": 0.0,
                    "oem_bg_raw_ratio": 0.0,
                    "oem_bg_capped_ratio": 0.0,
                    "oem_skip_no_fg_proto": 0,
                    "oem_skip_no_bg_proto": 0,
                    "oem_skip_no_pos_region": 0,
                    "proto_delta_mean": 0.0,
                    "proto_delta_min": 0.0,
                    "proto_delta_max": 0.0,
                    "teacher_prob_37_mean": 0.0,
                    "teacher_prob_37_fg_seed_mean": 0.0,
                    "teacher_prob_37_bg_seed_mean": 0.0,
                    "teacher_prob_37_extent_mean": 0.0,
                }
                if use_dabe_pu and not use_dabe_pu_balanced_v2 and not use_dabe_oem and not use_dabe_pu_despl_sched:
                    pu_teacher_target, pu_teacher_weight_map, pu_teacher_stats = build_dabe_pu_teacher_conf_target_weight(
                        cfg,
                        teacher_prob.detach(),
                        pu_fg_core,
                        pu_bg_core,
                    )
                if use_dabe_pu_despl_sched:
                    teacher_binary_area = float(teacher_binary.detach().mean().item())
                    pu_teacher_stats = {
                        "teacher_conf_ratio": 1.0,
                        "teacher_fg_ratio": teacher_binary_area,
                        "teacher_bg_ratio": 1.0 - teacher_binary_area,
                    }
                with torch.no_grad():
                    student_prob_for_area = student_logits.detach().sigmoid()
                    student_prob_mean_sum += float(student_prob_for_area.mean().item())
                    teacher_prob_mean_sum += float(teacher_prob.mean().item())
                    teacher_prob_min_batch = float(teacher_prob.min().item())
                    teacher_prob_max_batch = float(teacher_prob.max().item())
                    teacher_prob_min = (
                        teacher_prob_min_batch
                        if teacher_prob_min is None
                        else min(teacher_prob_min, teacher_prob_min_batch)
                    )
                    teacher_prob_max = (
                        teacher_prob_max_batch
                        if teacher_prob_max is None
                        else max(teacher_prob_max, teacher_prob_max_batch)
                    )
                    if use_dabe_pu_despl_sched:
                        teacher_soft_target_mean_sum += float(teacher_full_target.mean().item())
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
                    if use_dabe_oem:
                        loss_oem_seed_final, oem_seed_final_stats = build_dabe_oem_seed_loss(
                            student_logits,
                            batch,
                            cfg,
                        )
                        oem_masks = build_dabe_oem_dynamic_masks(
                            batch["feature"].to(device, non_blocking=True).float(),
                            teacher_logits,
                            batch,
                            cfg,
                            epoch,
                        )
                        loss_oem_dyn_pos, loss_oem_dyn_bg, dyn_loss_stats = build_dabe_oem_dynamic_loss(
                            student_logits,
                            oem_masks,
                            cfg,
                        )
                        oem_dynamic_stats = {**oem_masks["stats"], **dyn_loss_stats}
                        lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
                        loss_oem_seed_group = loss_oem_seed_final
                        loss_pu_static_final = loss_oem_seed_final
                        loss_pu_static_group = loss_oem_seed_group
                        loss_final_bce = loss_oem_seed_final
                        loss_tversky = student_logits.sum() * 0.0
                        loss_base = (
                            loss_oem_seed_group
                            + lambda_dyn_pos * loss_oem_dyn_pos
                            + lambda_dyn_bg * loss_oem_dyn_bg
                        )
                    elif use_dabe_pu_balanced_v2:
                        loss_pu_static_final, pu_static_final_stats = build_pu_static_group_loss(
                            student_logits,
                            batch,
                            cfg,
                        )
                        loss_pu_teacher_final, pu_teacher_final_stats = build_teacher_conf_balanced_loss(
                            student_logits,
                            teacher_prob.detach(),
                            batch,
                            cfg,
                        )
                        pu_teacher_stats = {
                            "teacher_conf_ratio": float(pu_teacher_final_stats["teacher_conf_ratio_capped"]),
                            "teacher_fg_ratio": float(pu_teacher_final_stats["teacher_fg_ratio_raw"]),
                            "teacher_bg_ratio": float(pu_teacher_final_stats["teacher_bg_ratio_capped"]),
                        }
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_balanced_v2_schedule(epoch, cfg)
                        loss_pu_static_group = loss_pu_static_final
                        loss_pu_teacher_group = loss_pu_teacher_final
                        loss_final_bce = loss_pu_static_final
                        loss_tversky = student_logits.sum() * 0.0
                        loss_base = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                    elif use_dabe_pu_despl_sched:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        loss_pu_static_final = weighted_bce_with_logits(
                            student_logits,
                            pu_static_target,
                            pu_static_weight_map,
                            eps=eps,
                        )
                        loss_pu_teacher_final = rast_teacher_bce_with_logits(
                            student_logits,
                            teacher_full_target,
                            rast_teacher_map_eff,
                            cfg,
                            rast_stats["rast_scale"],
                            apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_FINAL", True)),
                            eps=eps,
                        )
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_despl_schedule(epoch, cfg)
                        loss_pu_static_group = loss_pu_static_final
                        loss_pu_teacher_group = loss_pu_teacher_final
                        loss_final_bce = loss_pu_static_final
                        loss_tversky = student_logits.sum() * 0.0
                        loss_base = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                    elif use_dabe_pu:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        loss_pu_static_final = weighted_bce_with_logits(
                            student_logits,
                            pu_target_soft,
                            pu_weight_map,
                            eps=eps,
                        )
                        loss_pu_teacher_final = weighted_bce_with_logits(
                            student_logits,
                            pu_teacher_target,
                            pu_teacher_weight_map,
                            eps=eps,
                        )
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_schedule(epoch, cfg)
                        pu_static_loss_weight *= float(getattr(cfg, "LAMBDA_DABE_PU_STATIC", 1.0))
                        pu_teacher_loss_weight *= float(getattr(cfg, "LAMBDA_TEACHER_CONF", 1.0))
                        loss_pu_static_group = loss_pu_static_final
                        loss_pu_teacher_group = loss_pu_teacher_final
                        loss_final_bce = loss_pu_static_final
                        loss_tversky = student_logits.sum() * 0.0
                        loss_base = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                    elif use_dabe_aware:
                        loss_final_bce = weighted_bce_with_logits(
                            student_logits,
                            dabe_aware_target,
                            dabe_aware_weight_map,
                            eps=float(getattr(cfg, "DABE_TVERSKY_EPS", 1e-6)),
                        )
                        if bool(getattr(cfg, "USE_DABE_TVERSKY_LOSS", True)):
                            loss_tversky = soft_tversky_loss(
                                student_logits,
                                dabe_aware_target,
                                alpha_fp=float(getattr(cfg, "DABE_TVERSKY_ALPHA_FP", 0.30)),
                                beta_fn=float(getattr(cfg, "DABE_TVERSKY_BETA_FN", 0.70)),
                                eps=float(getattr(cfg, "DABE_TVERSKY_EPS", 1e-6)),
                            )
                        else:
                            loss_tversky = student_logits.sum() * 0.0
                        loss_base = loss_final_bce + float(getattr(cfg, "DABE_TVERSKY_WEIGHT", 0.30)) * loss_tversky
                    else:
                        loss_final_bce = criterion(student_logits, mixed_target)
                        loss_tversky = student_logits.sum() * 0.0
                        loss_base = loss_final_bce
                loss_aux_base = student_logits.sum() * 0.0
                loss_ndr_coarse_aux = student_logits.sum() * 0.0
                loss_ndr_res_reg = student_logits.sum() * 0.0
                loss_area_guard = student_logits.sum() * 0.0
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
                elif (
                    use_ccr
                    and is_before_finetune_reset(cfg, epoch)
                    and bool(getattr(cfg, "CCR_USE_ANCHOR_LOSS", False))
                ):
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
                    if use_dabe_oem:
                        lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
                        seed_terms = [loss_oem_seed_final]
                        seed_weights = [1.0]
                        dyn_pos_terms = [loss_oem_dyn_pos]
                        dyn_bg_terms = [loss_oem_dyn_bg]
                        dyn_weights = [1.0]
                        if (
                            bool(getattr(cfg, "USE_NDR_COARSE_AUX", True))
                            and bool(getattr(cfg, "OEM_USE_SEED_LOSS_ON_COARSE", True))
                        ):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            loss_oem_seed_coarse, _ = build_dabe_oem_seed_loss(coarse_logits, batch, cfg)
                            seed_terms.append(loss_oem_seed_coarse)
                            seed_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                            if bool(getattr(cfg, "OEM_USE_DYNAMIC_ON_COARSE", False)):
                                dyn_pos_coarse, dyn_bg_coarse, _ = build_dabe_oem_dynamic_loss(
                                    coarse_logits,
                                    oem_masks,
                                    cfg,
                                )
                                dyn_pos_terms.append(dyn_pos_coarse)
                                dyn_bg_terms.append(dyn_bg_coarse)
                                dyn_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                                loss_ndr_coarse_aux = (
                                    loss_oem_seed_coarse
                                    + lambda_dyn_pos * dyn_pos_coarse
                                    + lambda_dyn_bg * dyn_bg_coarse
                                )
                            else:
                                loss_ndr_coarse_aux = loss_oem_seed_coarse
                        if (
                            bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                            and bool(getattr(cfg, "OEM_USE_SEED_LOSS_ON_BASE", True))
                            and "base_logits" in student_out
                        ):
                            aux_lambda = (
                                float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                                if is_before_finetune_reset(cfg, epoch)
                                else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                            )
                            base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                            loss_oem_seed_base, _ = build_dabe_oem_seed_loss(base_logits, batch, cfg)
                            seed_terms.append(loss_oem_seed_base)
                            seed_weights.append(aux_lambda)
                            if bool(getattr(cfg, "OEM_USE_DYNAMIC_ON_BASE", False)):
                                dyn_pos_base, dyn_bg_base, _ = build_dabe_oem_dynamic_loss(
                                    base_logits,
                                    oem_masks,
                                    cfg,
                                )
                                dyn_pos_terms.append(dyn_pos_base)
                                dyn_bg_terms.append(dyn_bg_base)
                                dyn_weights.append(aux_lambda)
                                loss_aux_base = (
                                    loss_oem_seed_base
                                    + lambda_dyn_pos * dyn_pos_base
                                    + lambda_dyn_bg * dyn_bg_base
                                )
                            else:
                                loss_aux_base = loss_oem_seed_base
                        seed_weight_sum = max(1e-12, sum(seed_weights))
                        loss_oem_seed_group = sum(
                            w * term for w, term in zip(seed_weights, seed_terms)
                        ) / seed_weight_sum
                        dyn_weight_sum = max(1e-12, sum(dyn_weights))
                        dyn_pos_group = sum(
                            w * term for w, term in zip(dyn_weights, dyn_pos_terms)
                        ) / dyn_weight_sum
                        dyn_bg_group = sum(
                            w * term for w, term in zip(dyn_weights, dyn_bg_terms)
                        ) / dyn_weight_sum
                        loss_pu_static_group = loss_oem_seed_group
                        loss = loss_oem_seed_group + lambda_dyn_pos * dyn_pos_group + lambda_dyn_bg * dyn_bg_group
                    elif use_dabe_pu_balanced_v2:
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_balanced_v2_schedule(epoch, cfg)
                        static_terms = [loss_pu_static_final]
                        teacher_terms = [loss_pu_teacher_final]
                        pu_aux_weights = [1.0]
                        if bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            loss_pu_static_coarse, _ = build_pu_static_group_loss(
                                coarse_logits,
                                batch,
                                cfg,
                            )
                            loss_pu_teacher_coarse, _ = build_teacher_conf_balanced_loss(
                                coarse_logits,
                                teacher_prob.detach(),
                                batch,
                                cfg,
                            )
                            static_terms.append(loss_pu_static_coarse)
                            teacher_terms.append(loss_pu_teacher_coarse)
                            pu_aux_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                        if (
                            bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                            and "base_logits" in student_out
                        ):
                            aux_lambda = (
                                float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                                if is_before_finetune_reset(cfg, epoch)
                                else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                            )
                            base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                            loss_pu_static_base, _ = build_pu_static_group_loss(
                                base_logits,
                                batch,
                                cfg,
                            )
                            loss_pu_teacher_base, _ = build_teacher_conf_balanced_loss(
                                base_logits,
                                teacher_prob.detach(),
                                batch,
                                cfg,
                            )
                            static_terms.append(loss_pu_static_base)
                            teacher_terms.append(loss_pu_teacher_base)
                            pu_aux_weights.append(aux_lambda)
                        weight_sum = max(1e-12, sum(pu_aux_weights))
                        loss_pu_static_group = sum(
                            w * term for w, term in zip(pu_aux_weights, static_terms)
                        ) / weight_sum
                        loss_pu_teacher_group = sum(
                            w * term for w, term in zip(pu_aux_weights, teacher_terms)
                        ) / weight_sum
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_ndr_coarse_aux = (
                            pu_static_loss_weight * loss_pu_static_coarse
                            + pu_teacher_loss_weight * loss_pu_teacher_coarse
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    elif use_dabe_pu_despl_sched:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_despl_schedule(epoch, cfg)
                        static_terms = [loss_pu_static_final]
                        teacher_terms = [loss_pu_teacher_final]
                        pu_aux_weights = [1.0]
                        if bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            loss_pu_static_coarse = weighted_bce_with_logits(
                                coarse_logits,
                                pu_static_target,
                                pu_static_weight_map,
                                eps=eps,
                            )
                            loss_pu_teacher_coarse = rast_teacher_bce_with_logits(
                                coarse_logits,
                                teacher_full_target,
                                rast_teacher_map_eff,
                                cfg,
                                rast_stats["rast_scale"],
                                apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_COARSE_AUX", True)),
                                eps=eps,
                            )
                            static_terms.append(loss_pu_static_coarse)
                            teacher_terms.append(loss_pu_teacher_coarse)
                            pu_aux_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                        if (
                            bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                            and "base_logits" in student_out
                        ):
                            aux_lambda = (
                                float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                                if is_before_finetune_reset(cfg, epoch)
                                else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                            )
                            base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                            loss_pu_static_base = weighted_bce_with_logits(
                                base_logits,
                                pu_static_target,
                                pu_static_weight_map,
                                eps=eps,
                            )
                            loss_pu_teacher_base = rast_teacher_bce_with_logits(
                                base_logits,
                                teacher_full_target,
                                rast_teacher_map_eff,
                                cfg,
                                rast_stats["rast_scale"],
                                apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_BASE_AUX", True)),
                                eps=eps,
                            )
                            static_terms.append(loss_pu_static_base)
                            teacher_terms.append(loss_pu_teacher_base)
                            pu_aux_weights.append(aux_lambda)
                        weight_sum = max(1e-12, sum(pu_aux_weights))
                        loss_pu_static_group = sum(
                            w * term for w, term in zip(pu_aux_weights, static_terms)
                        ) / weight_sum
                        loss_pu_teacher_group = sum(
                            w * term for w, term in zip(pu_aux_weights, teacher_terms)
                        ) / weight_sum
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_ndr_coarse_aux = (
                            pu_static_loss_weight * loss_pu_static_coarse
                            + pu_teacher_loss_weight * loss_pu_teacher_coarse
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    elif use_dabe_pu:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_schedule(epoch, cfg)
                        pu_static_loss_weight *= float(getattr(cfg, "LAMBDA_DABE_PU_STATIC", 1.0))
                        pu_teacher_loss_weight *= float(getattr(cfg, "LAMBDA_TEACHER_CONF", 1.0))
                        static_terms = [loss_pu_static_final]
                        teacher_terms = [loss_pu_teacher_final]
                        pu_aux_weights = [1.0]
                        if bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            loss_pu_static_coarse = weighted_bce_with_logits(
                                coarse_logits,
                                pu_target_soft,
                                pu_weight_map,
                                eps=eps,
                            )
                            loss_pu_teacher_coarse = weighted_bce_with_logits(
                                coarse_logits,
                                pu_teacher_target,
                                pu_teacher_weight_map,
                                eps=eps,
                            )
                            static_terms.append(loss_pu_static_coarse)
                            teacher_terms.append(loss_pu_teacher_coarse)
                            pu_aux_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                        if (
                            bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                            and "base_logits" in student_out
                        ):
                            aux_lambda = (
                                float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                                if is_before_finetune_reset(cfg, epoch)
                                else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                            )
                            base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                            loss_pu_static_base = weighted_bce_with_logits(
                                base_logits,
                                pu_target_soft,
                                pu_weight_map,
                                eps=eps,
                            )
                            loss_pu_teacher_base = weighted_bce_with_logits(
                                base_logits,
                                pu_teacher_target,
                                pu_teacher_weight_map,
                                eps=eps,
                            )
                            static_terms.append(loss_pu_static_base)
                            teacher_terms.append(loss_pu_teacher_base)
                            pu_aux_weights.append(aux_lambda)
                        weight_sum = max(1e-12, sum(pu_aux_weights))
                        loss_pu_static_group = sum(
                            w * term for w, term in zip(pu_aux_weights, static_terms)
                        ) / weight_sum
                        loss_pu_teacher_group = sum(
                            w * term for w, term in zip(pu_aux_weights, teacher_terms)
                        ) / weight_sum
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_ndr_coarse_aux = (
                            pu_static_loss_weight * loss_pu_static_coarse
                            + pu_teacher_loss_weight * loss_pu_teacher_coarse
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    else:
                        ndr_terms = [loss_base]
                        ndr_weights = [1.0]
                        if bool(getattr(cfg, "USE_NDR_COARSE_AUX", True)):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            if use_dabe_aware:
                                loss_ndr_coarse_aux = weighted_bce_with_logits(
                                    coarse_logits,
                                    dabe_aware_target,
                                    dabe_aware_weight_map,
                                    eps=float(getattr(cfg, "DABE_TVERSKY_EPS", 1e-6)),
                                )
                            else:
                                loss_ndr_coarse_aux = criterion(coarse_logits, mixed_target)
                            ndr_terms.append(loss_ndr_coarse_aux)
                            ndr_weights.append(float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5)))
                        if (
                            bool(getattr(cfg, "USE_BASE_AUX_LOSS", False))
                            and "base_logits" in student_out
                        ):
                            aux_lambda = (
                                float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                                if is_before_finetune_reset(cfg, epoch)
                                else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                            )
                            base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                            if use_dabe_aware:
                                loss_aux_base = weighted_bce_with_logits(
                                    base_logits,
                                    dabe_aware_target,
                                    dabe_aware_weight_map,
                                    eps=float(getattr(cfg, "DABE_TVERSKY_EPS", 1e-6)),
                                )
                            else:
                                loss_aux_base = criterion(base_logits, mixed_target)
                            ndr_terms.append(loss_aux_base)
                            ndr_weights.append(aux_lambda)
                        weight_sum = max(1e-12, sum(ndr_weights))
                        loss = sum(w * term for w, term in zip(ndr_weights, ndr_terms)) / weight_sum
                        if use_dabe_aware:
                            loss_area_guard = dabe_area_guard_loss(cfg, epoch, student_logits, pseudo_68)
                            loss = loss + loss_area_guard
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
                    aux_lambda = (
                        float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
                        if is_before_finetune_reset(cfg, epoch)
                        else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
                    )
                    base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                    if use_dabe_oem:
                        lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
                        if bool(getattr(cfg, "OEM_USE_SEED_LOSS_ON_BASE", True)):
                            loss_oem_seed_base, _ = build_dabe_oem_seed_loss(base_logits, batch, cfg)
                            loss_oem_seed_group = (loss_oem_seed_final + aux_lambda * loss_oem_seed_base) / (
                                1.0 + aux_lambda
                            )
                            loss_aux_base = loss_oem_seed_base
                        else:
                            loss_oem_seed_group = loss_oem_seed_final
                        if bool(getattr(cfg, "OEM_USE_DYNAMIC_ON_BASE", False)):
                            dyn_pos_base, dyn_bg_base, _ = build_dabe_oem_dynamic_loss(base_logits, oem_masks, cfg)
                            dyn_pos_group = (loss_oem_dyn_pos + aux_lambda * dyn_pos_base) / (1.0 + aux_lambda)
                            dyn_bg_group = (loss_oem_dyn_bg + aux_lambda * dyn_bg_base) / (1.0 + aux_lambda)
                            loss_aux_base = (
                                loss_aux_base
                                + lambda_dyn_pos * dyn_pos_base
                                + lambda_dyn_bg * dyn_bg_base
                            )
                        else:
                            dyn_pos_group = loss_oem_dyn_pos
                            dyn_bg_group = loss_oem_dyn_bg
                        loss_pu_static_group = loss_oem_seed_group
                        loss = loss_oem_seed_group + lambda_dyn_pos * dyn_pos_group + lambda_dyn_bg * dyn_bg_group
                    elif use_dabe_pu_balanced_v2:
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_balanced_v2_schedule(epoch, cfg)
                        loss_pu_static_base, _ = build_pu_static_group_loss(
                            base_logits,
                            batch,
                            cfg,
                        )
                        loss_pu_teacher_base, _ = build_teacher_conf_balanced_loss(
                            base_logits,
                            teacher_prob.detach(),
                            batch,
                            cfg,
                        )
                        loss_pu_static_group = (loss_pu_static_final + aux_lambda * loss_pu_static_base) / (1.0 + aux_lambda)
                        loss_pu_teacher_group = (loss_pu_teacher_final + aux_lambda * loss_pu_teacher_base) / (1.0 + aux_lambda)
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    elif use_dabe_pu_despl_sched:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_despl_schedule(epoch, cfg)
                        loss_pu_static_base = weighted_bce_with_logits(
                            base_logits,
                            pu_static_target,
                            pu_static_weight_map,
                            eps=eps,
                        )
                        loss_pu_teacher_base = rast_teacher_bce_with_logits(
                            base_logits,
                            teacher_full_target,
                            rast_teacher_map_eff,
                            cfg,
                            rast_stats["rast_scale"],
                            apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_BASE_AUX", True)),
                            eps=eps,
                        )
                        loss_pu_static_group = (loss_pu_static_final + aux_lambda * loss_pu_static_base) / (1.0 + aux_lambda)
                        loss_pu_teacher_group = (loss_pu_teacher_final + aux_lambda * loss_pu_teacher_base) / (1.0 + aux_lambda)
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    elif use_dabe_pu:
                        eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                        pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_schedule(epoch, cfg)
                        pu_static_loss_weight *= float(getattr(cfg, "LAMBDA_DABE_PU_STATIC", 1.0))
                        pu_teacher_loss_weight *= float(getattr(cfg, "LAMBDA_TEACHER_CONF", 1.0))
                        loss_pu_static_base = weighted_bce_with_logits(
                            base_logits,
                            pu_target_soft,
                            pu_weight_map,
                            eps=eps,
                        )
                        loss_pu_teacher_base = weighted_bce_with_logits(
                            base_logits,
                            pu_teacher_target,
                            pu_teacher_weight_map,
                            eps=eps,
                        )
                        loss_pu_static_group = (loss_pu_static_final + aux_lambda * loss_pu_static_base) / (1.0 + aux_lambda)
                        loss_pu_teacher_group = (loss_pu_teacher_final + aux_lambda * loss_pu_teacher_base) / (1.0 + aux_lambda)
                        loss = (
                            pu_static_loss_weight * loss_pu_static_group
                            + pu_teacher_loss_weight * loss_pu_teacher_group
                        )
                        loss_aux_base = (
                            pu_static_loss_weight * loss_pu_static_base
                            + pu_teacher_loss_weight * loss_pu_teacher_base
                        )
                    elif use_dabe_aware:
                        loss_aux_base = weighted_bce_with_logits(
                            base_logits,
                            dabe_aware_target,
                            dabe_aware_weight_map,
                            eps=float(getattr(cfg, "DABE_TVERSKY_EPS", 1e-6)),
                        )
                    else:
                        loss_aux_base = criterion(base_logits, mixed_target)
                    if use_dabe_pu:
                        pass
                    elif bool(getattr(cfg, "BASE_AUX_NORMALIZE", False)):
                        loss = (loss + aux_lambda * loss_aux_base) / (1.0 + aux_lambda)
                    else:
                        loss = loss + aux_lambda * loss_aux_base
                if use_dabe_aware and not use_ndr_branch(cfg):
                    loss_area_guard = dabe_area_guard_loss(cfg, epoch, student_logits, pseudo_68)
                    loss = loss + loss_area_guard
                if (
                    use_despl
                    and bool(getattr(cfg, "USE_LATE_DESPL_ANCHOR_LOSS", False))
                    and is_at_or_after_finetune_reset(cfg, epoch)
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

                if use_view_consistency(cfg):
                    loss = loss + float(lambda_view) * loss_view
                if use_proto:
                    loss = loss + float(lambda_proto) * loss_proto

                loss_hbns_final = student_logits.sum() * 0.0
                loss_hbns_coarse = student_logits.sum() * 0.0
                loss_hbns_base = student_logits.sum() * 0.0
                loss_hbns = student_logits.sum() * 0.0
                hbns_scale = 0.0
                lambda_hbns_eff = 0.0
                loss_epr_final = student_logits.sum() * 0.0
                loss_epr_coarse = student_logits.sum() * 0.0
                loss_epr_base = student_logits.sum() * 0.0
                loss_epr = student_logits.sum() * 0.0
                epr_scale = 0.0
                lambda_epr_eff = 0.0
                hbns_stats = {
                    "hard_bg_ratio": 0.0,
                    "hard_bg_raw_ratio": 0.0,
                    "hard_bg_ratio_bg_core": 0.0,
                    "hard_bg_ratio_low_target": 0.0,
                    "hard_bg_ratio_unknown_bg_like": 0.0,
                    "hard_bg_pixels_mean": 0.0,
                }
                epr_stats = {
                    "epr_pos_ratio": 0.0,
                    "epr_pos_raw_ratio": 0.0,
                    "epr_pos_pixels_mean": 0.0,
                    "epr_valid_image_ratio": 0.0,
                    "epr_margin_mean": 0.0,
                    "epr_margin_min": 0.0,
                    "epr_margin_max": 0.0,
                    "epr_margin_pos_mean": 0.0,
                    "epr_teacher_conf_pos_mean": 0.0,
                    "epr_extent_area": 0.0,
                    "epr_unknown_overlap_ratio": 0.0,
                }
                esa_stats = {}
                if use_hbns_lite or use_epr_pos or use_esa_diagnostic:
                    aux_region_masks = build_rast_region_masks(batch, device)
                else:
                    aux_region_masks = None
                if use_epr_pos:
                    epr_scale = float(get_epr_scale(cfg, epoch))
                    lambda_epr_eff = float(getattr(cfg, "EPR_LAMBDA_MAX", 0.005)) * epr_scale
                    epr_pos_mask, epr_margin_68, epr_stats = build_epr_pos_mask(
                        cfg,
                        batch,
                        aux_region_masks,
                        teacher_prob,
                        teacher_binary,
                        device,
                    )
                    epr_terms = []
                    if bool(getattr(cfg, "EPR_APPLY_TO_FINAL", True)):
                        loss_epr_final = epr_pos_loss_for_logits(cfg, student_logits, epr_pos_mask)
                        epr_terms.append(loss_epr_final)
                    if (
                        bool(getattr(cfg, "EPR_APPLY_TO_COARSE_AUX", True))
                        and isinstance(student_out, dict)
                        and "coarse_logits_68" in student_out
                    ):
                        epr_coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                        loss_epr_coarse = epr_pos_loss_for_logits(cfg, epr_coarse_logits, epr_pos_mask)
                        epr_terms.append(loss_epr_coarse)
                    if (
                        bool(getattr(cfg, "EPR_APPLY_TO_BASE_AUX", False))
                        and isinstance(student_out, dict)
                        and "base_logits" in student_out
                    ):
                        epr_base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                        loss_epr_base = epr_pos_loss_for_logits(cfg, epr_base_logits, epr_pos_mask)
                        epr_terms.append(loss_epr_base)
                    if epr_terms:
                        loss_epr = sum(epr_terms) / float(len(epr_terms))
                    loss = loss + lambda_epr_eff * loss_epr
                    if not epr_first_batch_logged:
                        logger.log(f"[EPR-pos FirstBatch] USE_EPR_POS = {bool(getattr(cfg, 'USE_EPR_POS', False))}")
                        logger.log(f"[EPR-pos FirstBatch] EPR_VERSION = {getattr(cfg, 'EPR_VERSION', 'pos_lite_v1')}")
                        logger.log(f"[EPR-pos FirstBatch] epr_scale = {epr_scale:.8f}")
                        logger.log(f"[EPR-pos FirstBatch] lambda_epr_eff = {lambda_epr_eff:.8f}")
                        logger.log(f"[EPR-pos FirstBatch] epr_pos_mask shape = {list(epr_pos_mask.shape)}")
                        logger.log(
                            "[EPR-pos FirstBatch] epr_pos_ratio/raw_ratio/pixels_mean = "
                            f"{float(epr_stats['epr_pos_ratio']):.6f}/"
                            f"{float(epr_stats['epr_pos_raw_ratio']):.6f}/"
                            f"{float(epr_stats['epr_pos_pixels_mean']):.6f}"
                        )
                        logger.log(
                            "[EPR-pos FirstBatch] margin_68 min/mean/max = "
                            f"{float(epr_stats['epr_margin_min']):.6f}/"
                            f"{float(epr_stats['epr_margin_mean']):.6f}/"
                            f"{float(epr_stats['epr_margin_max']):.6f}"
                        )
                        epr_first_batch_logged = True
                if use_hbns_lite:
                    hbns_scale = float(get_hbns_scale(cfg, epoch))
                    lambda_hbns_eff = float(getattr(cfg, "HBNS_LAMBDA_MAX", 0.01)) * hbns_scale
                    hard_bg_mask, hbns_stats = build_hbns_hard_bg_mask(
                        cfg,
                        batch,
                        student_logits,
                        teacher_binary,
                        aux_region_masks,
                        device,
                    )
                    hbns_terms = []
                    if bool(getattr(cfg, "HBNS_APPLY_TO_FINAL", True)):
                        loss_hbns_final = hbns_lite_loss_for_logits(cfg, student_logits, hard_bg_mask)
                        hbns_terms.append(loss_hbns_final)
                    if (
                        bool(getattr(cfg, "HBNS_APPLY_TO_COARSE_AUX", True))
                        and isinstance(student_out, dict)
                        and "coarse_logits_68" in student_out
                    ):
                        hbns_coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                        loss_hbns_coarse = hbns_lite_loss_for_logits(cfg, hbns_coarse_logits, hard_bg_mask)
                        hbns_terms.append(loss_hbns_coarse)
                    if (
                        bool(getattr(cfg, "HBNS_APPLY_TO_BASE_AUX", False))
                        and isinstance(student_out, dict)
                        and "base_logits" in student_out
                    ):
                        hbns_base_logits = resize_logits_for_loss(student_out["base_logits"], cfg)
                        loss_hbns_base = hbns_lite_loss_for_logits(cfg, hbns_base_logits, hard_bg_mask)
                        hbns_terms.append(loss_hbns_base)
                    if hbns_terms:
                        loss_hbns = sum(hbns_terms) / float(len(hbns_terms))
                    loss = loss + lambda_hbns_eff * loss_hbns
                    if not hbns_first_batch_logged:
                        logger.log(f"[HBNS-lite FirstBatch] USE_HBNS_LITE = {bool(getattr(cfg, 'USE_HBNS_LITE', False))}")
                        logger.log(f"[HBNS-lite FirstBatch] HBNS_VERSION = {getattr(cfg, 'HBNS_VERSION', 'lite_v1')}")
                        logger.log(f"[HBNS-lite FirstBatch] hbns_scale = {hbns_scale:.8f}")
                        logger.log(f"[HBNS-lite FirstBatch] lambda_hbns_eff = {lambda_hbns_eff:.8f}")
                        logger.log(f"[HBNS-lite FirstBatch] hard_bg_mask shape = {list(hard_bg_mask.shape)}")
                        logger.log(
                            "[HBNS-lite FirstBatch] hard_bg_ratio/raw_ratio/pixels_mean = "
                            f"{float(hbns_stats['hard_bg_ratio']):.6f}/"
                            f"{float(hbns_stats['hard_bg_raw_ratio']):.6f}/"
                            f"{float(hbns_stats['hard_bg_pixels_mean']):.6f}"
                        )
                        hbns_first_batch_logged = True
                if use_esa_diagnostic:
                    esa_stats = compute_esa_region_diagnostics(
                        batch,
                        aux_region_masks,
                        student_logits,
                        teacher_prob,
                        teacher_binary,
                    )

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
                if use_dabe_aware:
                    dabe_aware_fg_core_area_sum += float(dabe_aware_stats["fg_core_area"])
                    dabe_aware_bg_core_area_sum += float(dabe_aware_stats["bg_core_area"])
                    dabe_aware_uncertain_area_sum += float(dabe_aware_stats["uncertain_area"])
                    dabe_aware_evidence_sum += float(dabe_aware_stats["evidence_mean"])
                    dabe_aware_target_area_sum += float(dabe_aware_stats["target_area"])
                    dabe_aware_weight_map_sum += float(dabe_aware_stats["weight_map_mean"])
                    dabe_aware_loss_final_bce_sum += float(loss_final_bce.detach().item())
                    dabe_aware_loss_tversky_sum += float(loss_tversky.detach().item())
                    dabe_aware_loss_area_guard_sum += float(loss_area_guard.detach().item())
                    dabe_aware_stat_batches += 1
                if use_dabe_pu:
                    dabe_pu_target_mean_sum += float(pu_target_soft.detach().mean().item())
                    dabe_pu_weight_mean_sum += float(pu_weight_map.detach().mean().item())
                    dabe_pu_target_hard_area_sum += float(pu_target_hard.detach().mean().item())
                    dabe_pu_fg_core_mean_sum += float(batch["pu_fg_core"].float().mean().item())
                    dabe_pu_fg_fallback_mean_sum += float(batch["pu_fg_fallback"].float().mean().item())
                    dabe_pu_bg_core_mean_sum += float(batch["pu_bg_core"].float().mean().item())
                    dabe_pu_extent_mean_sum += float(batch["pu_extent"].float().mean().item())
                    dabe_pu_unknown_mean_sum += float(batch["pu_unknown"].float().mean().item())
                    dabe_pu_static_final_loss_sum += float(loss_pu_static_final.detach().item())
                    dabe_pu_static_coarse_loss_sum += float(loss_pu_static_coarse.detach().item())
                    dabe_pu_static_base_loss_sum += float(loss_pu_static_base.detach().item())
                    dabe_pu_static_group_loss_sum += float(loss_pu_static_group.detach().item())
                    dabe_pu_teacher_final_loss_sum += float(loss_pu_teacher_final.detach().item())
                    dabe_pu_teacher_coarse_loss_sum += float(loss_pu_teacher_coarse.detach().item())
                    dabe_pu_teacher_base_loss_sum += float(loss_pu_teacher_base.detach().item())
                    dabe_pu_teacher_group_loss_sum += float(loss_pu_teacher_group.detach().item())
                    dabe_pu_teacher_conf_ratio_sum += float(pu_teacher_stats["teacher_conf_ratio"])
                    dabe_pu_teacher_fg_ratio_sum += float(pu_teacher_stats["teacher_fg_ratio"])
                    dabe_pu_teacher_bg_ratio_sum += float(pu_teacher_stats["teacher_bg_ratio"])
                    dabe_pu_stat_batches += 1
                    if use_rast:
                        rast_scale_sum += float(rast_stats["rast_scale"])
                        rast_pre_reset_scale_sum += float(rast_stats["rast_pre_reset_scale"])
                        rast_post_reset_scale_sum += float(rast_stats["rast_post_reset_scale"])
                        rast_scale_effective_sum += float(rast_stats["rast_scale_effective"])
                        rast_fg_core_area_sum += float(rast_stats["fg_core_area"])
                        rast_bg_core_area_sum += float(rast_stats["bg_core_area"])
                        rast_extent_area_sum += float(rast_stats["extent_area"])
                        rast_unknown_area_sum += float(rast_stats["unknown_area"])
                        rast_fg_conflict_ratio_sum += float(rast_stats["fg_conflict_ratio"])
                        rast_bg_conflict_ratio_sum += float(rast_stats["bg_conflict_ratio"])
                        rast_teacher_map_mean_sum += float(rast_stats["teacher_map_mean"])
                        rast_teacher_map_min = (
                            float(rast_stats["teacher_map_min"])
                            if rast_teacher_map_min is None
                            else min(rast_teacher_map_min, float(rast_stats["teacher_map_min"]))
                        )
                        rast_teacher_map_max = (
                            float(rast_stats["teacher_map_max"])
                            if rast_teacher_map_max is None
                            else max(rast_teacher_map_max, float(rast_stats["teacher_map_max"]))
                        )
                        rast_teacher_map_fg_core_mean_sum += float(rast_stats["teacher_map_fg_core_mean"])
                        rast_teacher_map_bg_core_mean_sum += float(rast_stats["teacher_map_bg_core_mean"])
                        rast_teacher_map_extent_mean_sum += float(rast_stats["teacher_map_extent_mean"])
                        rast_teacher_map_unknown_mean_sum += float(rast_stats["teacher_map_unknown_mean"])
                        rast_stat_batches += 1
                        if use_esa_asym:
                            esa_asym_scale_sum += float(rast_stats["esa_asym_scale"])
                            esa_margin_mean_sum += float(rast_stats["esa_margin_mean"])
                            esa_margin_min_value = float(rast_stats["esa_margin_min"])
                            esa_margin_max_value = float(rast_stats["esa_margin_max"])
                            esa_margin_min = (
                                esa_margin_min_value
                                if esa_margin_min is None
                                else min(esa_margin_min, esa_margin_min_value)
                            )
                            esa_margin_max = (
                                esa_margin_max_value
                                if esa_margin_max is None
                                else max(esa_margin_max, esa_margin_max_value)
                            )
                            esa_margin_extent_mean_sum += float(rast_stats["esa_margin_extent_mean"])
                            esa_margin_extent_teacher_fg_mean_sum += float(
                                rast_stats["esa_margin_extent_teacher_fg_mean"]
                            )
                            esa_margin_extent_teacher_bg_mean_sum += float(
                                rast_stats["esa_margin_extent_teacher_bg_mean"]
                            )
                            esa_extent_teacher_fg_ratio_sum += float(rast_stats["esa_extent_teacher_fg_ratio"])
                            esa_extent_teacher_bg_ratio_sum += float(rast_stats["esa_extent_teacher_bg_ratio"])
                            esa_extent_teacher_bg_fg_like_ratio_sum += float(
                                rast_stats["esa_extent_teacher_bg_fg_like_ratio"]
                            )
                            esa_extent_teacher_bg_ambig_ratio_sum += float(
                                rast_stats["esa_extent_teacher_bg_ambig_ratio"]
                            )
                            esa_extent_teacher_bg_bg_like_ratio_sum += float(
                                rast_stats["esa_extent_teacher_bg_bg_like_ratio"]
                            )
                            esa_teacher_map_extent_mean_sum += float(rast_stats["esa_teacher_map_extent_mean"])
                            esa_teacher_map_extent_teacher_fg_mean_sum += float(
                                rast_stats["esa_teacher_map_extent_teacher_fg_mean"]
                            )
                            esa_teacher_map_extent_teacher_bg_mean_sum += float(
                                rast_stats["esa_teacher_map_extent_teacher_bg_mean"]
                            )
                            esa_teacher_map_extent_teacher_bg_fg_like_mean_sum += float(
                                rast_stats["esa_teacher_map_extent_teacher_bg_fg_like_mean"]
                            )
                            esa_teacher_map_extent_teacher_bg_ambig_mean_sum += float(
                                rast_stats["esa_teacher_map_extent_teacher_bg_ambig_mean"]
                            )
                            esa_teacher_map_extent_teacher_bg_bg_like_mean_sum += float(
                                rast_stats["esa_teacher_map_extent_teacher_bg_bg_like_mean"]
                            )
                            esa_skipped_no_fg_proto_sum += int(rast_stats["esa_skipped_no_fg_proto"])
                            esa_skipped_no_bg_proto_sum += int(rast_stats["esa_skipped_no_bg_proto"])
                            esa_asym_stat_batches += 1
                    if use_hbns_lite:
                        hbns_scale_sum += float(hbns_scale)
                        hbns_lambda_sum += float(lambda_hbns_eff)
                        hbns_hard_bg_ratio_sum += float(hbns_stats["hard_bg_ratio"])
                        hbns_hard_bg_raw_ratio_sum += float(hbns_stats["hard_bg_raw_ratio"])
                        hbns_hard_bg_ratio_bg_core_sum += float(hbns_stats["hard_bg_ratio_bg_core"])
                        hbns_hard_bg_ratio_low_target_sum += float(hbns_stats["hard_bg_ratio_low_target"])
                        hbns_hard_bg_ratio_unknown_bg_like_sum += float(hbns_stats["hard_bg_ratio_unknown_bg_like"])
                        hbns_hard_bg_pixels_mean_sum += float(hbns_stats["hard_bg_pixels_mean"])
                        hbns_loss_final_sum += float(loss_hbns_final.detach().item())
                        hbns_loss_coarse_sum += float(loss_hbns_coarse.detach().item())
                        hbns_loss_base_sum += float(loss_hbns_base.detach().item())
                        hbns_loss_sum += float(loss_hbns.detach().item())
                        hbns_stat_batches += 1
                    if use_epr_pos:
                        epr_scale_sum += float(epr_scale)
                        epr_lambda_sum += float(lambda_epr_eff)
                        epr_pos_ratio_sum += float(epr_stats["epr_pos_ratio"])
                        epr_pos_raw_ratio_sum += float(epr_stats["epr_pos_raw_ratio"])
                        epr_pos_pixels_mean_sum += float(epr_stats["epr_pos_pixels_mean"])
                        epr_valid_image_ratio_sum += float(epr_stats["epr_valid_image_ratio"])
                        epr_margin_mean_sum += float(epr_stats["epr_margin_mean"])
                        epr_margin_min_value = float(epr_stats["epr_margin_min"])
                        epr_margin_max_value = float(epr_stats["epr_margin_max"])
                        epr_margin_min = (
                            epr_margin_min_value
                            if epr_margin_min is None
                            else min(epr_margin_min, epr_margin_min_value)
                        )
                        epr_margin_max = (
                            epr_margin_max_value
                            if epr_margin_max is None
                            else max(epr_margin_max, epr_margin_max_value)
                        )
                        epr_margin_pos_mean_sum += float(epr_stats["epr_margin_pos_mean"])
                        epr_teacher_conf_pos_mean_sum += float(epr_stats["epr_teacher_conf_pos_mean"])
                        epr_extent_area_sum += float(epr_stats["epr_extent_area"])
                        epr_unknown_overlap_ratio_sum += float(epr_stats["epr_unknown_overlap_ratio"])
                        epr_loss_final_sum += float(loss_epr_final.detach().item())
                        epr_loss_coarse_sum += float(loss_epr_coarse.detach().item())
                        epr_loss_base_sum += float(loss_epr_base.detach().item())
                        epr_loss_sum += float(loss_epr.detach().item())
                        epr_stat_batches += 1
                    if use_esa_diagnostic:
                        esa_student_prob_fg_core_sum += float(esa_stats["student_prob_fg_core"])
                        esa_student_prob_bg_core_sum += float(esa_stats["student_prob_bg_core"])
                        esa_student_prob_extent_sum += float(esa_stats["student_prob_extent"])
                        esa_student_prob_unknown_sum += float(esa_stats["student_prob_unknown"])
                        esa_teacher_fg_fg_core_sum += float(esa_stats["teacher_fg_fg_core"])
                        esa_teacher_fg_bg_core_sum += float(esa_stats["teacher_fg_bg_core"])
                        esa_teacher_fg_extent_sum += float(esa_stats["teacher_fg_extent"])
                        esa_teacher_fg_unknown_sum += float(esa_stats["teacher_fg_unknown"])
                        esa_teacher_conf_fg_core_sum += float(esa_stats["teacher_conf_fg_core"])
                        esa_teacher_conf_bg_core_sum += float(esa_stats["teacher_conf_bg_core"])
                        esa_teacher_conf_extent_sum += float(esa_stats["teacher_conf_extent"])
                        esa_teacher_conf_unknown_sum += float(esa_stats["teacher_conf_unknown"])
                        esa_teacher_loss_fg_core_sum += float(esa_stats["teacher_loss_fg_core"])
                        esa_teacher_loss_bg_core_sum += float(esa_stats["teacher_loss_bg_core"])
                        esa_teacher_loss_extent_sum += float(esa_stats["teacher_loss_extent"])
                        esa_teacher_loss_unknown_sum += float(esa_stats["teacher_loss_unknown"])
                        esa_stat_batches += 1
                    if use_dabe_oem:
                        lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
                        oem_lambda_dyn_pos_sum += float(lambda_dyn_pos)
                        oem_lambda_dyn_bg_sum += float(lambda_dyn_bg)
                        oem_seed_fg_area_sum += float(oem_seed_final_stats.get("seed_fg_area", 0.0))
                        oem_seed_bg_area_sum += float(oem_seed_final_stats.get("seed_bg_area", 0.0))
                        oem_extent_area_sum += float(batch["pu_extent"].float().mean().item())
                        oem_unknown_area_sum += float(batch["pu_unknown"].float().mean().item())
                        oem_loss_seed_fg_sum += float(oem_seed_final_stats.get("loss_seed_fg", 0.0))
                        oem_loss_seed_fg_fallback_sum += float(oem_seed_final_stats.get("loss_seed_fg_fallback", 0.0))
                        oem_loss_seed_bg_sum += float(oem_seed_final_stats.get("loss_seed_bg", 0.0))
                        oem_loss_seed_final_sum += float(loss_oem_seed_final.detach().item())
                        oem_loss_seed_coarse_sum += float(loss_oem_seed_coarse.detach().item())
                        oem_loss_seed_base_sum += float(loss_oem_seed_base.detach().item())
                        oem_loss_seed_group_sum += float(loss_oem_seed_group.detach().item())
                        oem_loss_dyn_pos_sum += float(loss_oem_dyn_pos.detach().item())
                        oem_loss_dyn_bg_sum += float(loss_oem_dyn_bg.detach().item())
                        oem_pos_raw_ratio_sum += float(oem_dynamic_stats.get("oem_pos_raw_ratio", 0.0))
                        oem_pos_capped_ratio_sum += float(oem_dynamic_stats.get("oem_pos_capped_ratio", 0.0))
                        oem_bg_raw_ratio_sum += float(oem_dynamic_stats.get("oem_bg_raw_ratio", 0.0))
                        oem_bg_capped_ratio_sum += float(oem_dynamic_stats.get("oem_bg_capped_ratio", 0.0))
                        oem_skip_no_fg_proto_sum += int(oem_dynamic_stats.get("oem_skip_no_fg_proto", 0))
                        oem_skip_no_bg_proto_sum += int(oem_dynamic_stats.get("oem_skip_no_bg_proto", 0))
                        oem_skip_no_pos_region_sum += int(oem_dynamic_stats.get("oem_skip_no_pos_region", 0))
                        proto_delta_min = float(oem_dynamic_stats.get("proto_delta_min", 0.0))
                        proto_delta_max = float(oem_dynamic_stats.get("proto_delta_max", 0.0))
                        oem_proto_delta_mean_sum += float(oem_dynamic_stats.get("proto_delta_mean", 0.0))
                        oem_proto_delta_min = (
                            proto_delta_min
                            if oem_proto_delta_min is None
                            else min(oem_proto_delta_min, proto_delta_min)
                        )
                        oem_proto_delta_max = (
                            proto_delta_max
                            if oem_proto_delta_max is None
                            else max(oem_proto_delta_max, proto_delta_max)
                        )
                        oem_teacher_prob_37_mean_sum += float(oem_dynamic_stats.get("teacher_prob_37_mean", 0.0))
                        oem_teacher_prob_37_fg_seed_sum += float(
                            oem_dynamic_stats.get("teacher_prob_37_fg_seed_mean", 0.0)
                        )
                        oem_teacher_prob_37_bg_seed_sum += float(
                            oem_dynamic_stats.get("teacher_prob_37_bg_seed_mean", 0.0)
                        )
                        oem_teacher_prob_37_extent_sum += float(
                            oem_dynamic_stats.get("teacher_prob_37_extent_mean", 0.0)
                        )
                        oem_stat_batches += 1
                    if use_dabe_pu_balanced_v2:
                        dabe_pu_bal_static_fg_loss_sum += float(pu_static_final_stats.get("loss_static_fg", 0.0))
                        dabe_pu_bal_static_fg_fallback_loss_sum += float(pu_static_final_stats.get("loss_static_fg_fallback", 0.0))
                        dabe_pu_bal_static_bg_loss_sum += float(pu_static_final_stats.get("loss_static_bg", 0.0))
                        dabe_pu_bal_static_extent_loss_sum += float(pu_static_final_stats.get("loss_static_extent", 0.0))
                        dabe_pu_bal_teacher_fg_loss_sum += float(pu_teacher_final_stats.get("loss_teacher_fg", 0.0))
                        dabe_pu_bal_teacher_bg_loss_sum += float(pu_teacher_final_stats.get("loss_teacher_bg", 0.0))
                        dabe_pu_bal_teacher_fg_raw_ratio_sum += float(pu_teacher_final_stats.get("teacher_fg_ratio_raw", 0.0))
                        dabe_pu_bal_teacher_bg_raw_ratio_sum += float(pu_teacher_final_stats.get("teacher_bg_ratio_raw", 0.0))
                        dabe_pu_bal_teacher_bg_capped_ratio_sum += float(pu_teacher_final_stats.get("teacher_bg_ratio_capped", 0.0))
                        dabe_pu_bal_teacher_conf_capped_ratio_sum += float(pu_teacher_final_stats.get("teacher_conf_ratio_capped", 0.0))
                        dabe_pu_bal_stat_batches += 1
                if use_view_consistency(cfg):
                    mv_lambda_sum += float(lambda_view)
                    mv_loss_sum += float(loss_view.detach().item())
                    mv_core_ratio_sum += float(mv_stats["core_ratio"])
                    mv_mean_abs_diff_sum += float(mv_stats["mean_abs_diff"])
                    mv_stat_batches += 1
                if use_proto:
                    proto_lambda_sum += float(lambda_proto)
                    proto_loss_sum += float(loss_proto.detach().item())
                    proto_align_sum += float(proto_stats["align_loss"])
                    proto_sep_sum += float(proto_stats["sep_loss"])
                    proto_pixel_sum += float(proto_stats["pixel_loss"])
                    proto_pixel_fg_sum += float(proto_stats["pixel_fg_loss"])
                    proto_pixel_bg_sum += float(proto_stats["pixel_bg_loss"])
                    proto_valid_ratio_sum += float(proto_stats["valid_ratio"])
                    proto_fg_core_ratio_sum += float(proto_stats["fg_core_ratio"])
                    proto_bg_core_ratio_sum += float(proto_stats["bg_core_ratio"])
                    proto_bg_hard_ratio_sum += float(proto_stats["bg_hard_ratio"])
                    proto_bg_ring_ratio_sum += float(proto_stats["bg_ring_ratio"])
                    proto_bg_disagree_ratio_sum += float(proto_stats["bg_disagree_ratio"])
                    proto_bg_residual_ratio_sum += float(proto_stats["bg_residual_ratio"])
                    proto_hard_fg_ratio_sum += float(proto_stats["hard_fg_ratio"])
                    proto_hard_bg_ratio_sum += float(proto_stats["hard_bg_ratio"])
                    proto_sep_active_ratio_sum += float(proto_stats["sep_active_ratio"])
                    proto_fg_fallback_ratio_sum += float(proto_stats["fg_fallback_ratio"])
                    proto_cos_fg_view_sum += float(proto_stats["cos_fg_view"])
                    proto_cos_bg_view_sum += float(proto_stats["cos_bg_view"])
                    proto_cos_fg_bg_sum += float(proto_stats["cos_fg_bg"])
                    proto_stat_batches += 1
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
                        if use_tadr_router(cfg):
                            router_mean = output_scalar(student_out, "tadr_router_mean")
                            router_min = output_scalar(student_out, "tadr_router_min")
                            router_max = output_scalar(student_out, "tadr_router_max")
                            base_gate_mean = output_scalar(student_out, "tadr_base_gate_mean")
                            base_gate_min = output_scalar(student_out, "tadr_base_gate_min")
                            base_gate_max = output_scalar(student_out, "tadr_base_gate_max")
                            final_gate_mean = output_scalar(student_out, "tadr_final_gate_mean")
                            final_gate_min = output_scalar(student_out, "tadr_final_gate_min")
                            final_gate_max = output_scalar(student_out, "tadr_final_gate_max")
                            tadr_router_mean_sum += router_mean
                            tadr_router_min = router_min if tadr_router_min is None else min(tadr_router_min, router_min)
                            tadr_router_max = router_max if tadr_router_max is None else max(tadr_router_max, router_max)
                            tadr_base_gate_mean_sum += base_gate_mean
                            tadr_base_gate_min = (
                                base_gate_min
                                if tadr_base_gate_min is None
                                else min(tadr_base_gate_min, base_gate_min)
                            )
                            tadr_base_gate_max = (
                                base_gate_max
                                if tadr_base_gate_max is None
                                else max(tadr_base_gate_max, base_gate_max)
                            )
                            tadr_final_gate_mean_sum += final_gate_mean
                            tadr_final_gate_min = (
                                final_gate_min
                                if tadr_final_gate_min is None
                                else min(tadr_final_gate_min, final_gate_min)
                            )
                            tadr_final_gate_max = (
                                final_gate_max
                                if tadr_final_gate_max is None
                                else max(tadr_final_gate_max, final_gate_max)
                            )
                            tadr_stat_batches += 1
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
                    if use_dabe:
                        dabe_num_samples += int(batch["p_dabe_area"].numel())
                        dabe_p_area_sum += float(batch["p_dabe_area"].sum().item())
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
                f"teacher_fusion_mode={fusion_mode} | "
                f"fixed_weight={effective_despl_weight:.2f} | "
                f"teacher_weight={effective_teacher_weight:.2f} | "
                f"schedule_fixed_weight={fixed_weight:.2f} | "
                f"schedule_teacher_weight={teacher_weight:.2f} | "
                f"dabe_weight={effective_despl_weight:.2f} | "
                f"schedule_dabe_weight={fixed_weight:.2f}"
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
                if use_dabe or use_dabe_pu:
                    use_fixed_in_pseudo = False
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
            if use_dabe:
                logger.log(
                    f"[DABE] epoch={epoch:03d} | "
                    "use_dabe_pseudo=True | "
                    f"dabe_version={getattr(cfg, 'DABE_VERSION', 'v2')} | "
                    f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'dabe_only')} | "
                    f"p_dabe_area_mean={dabe_p_area_sum / max(dabe_num_samples, 1):.6f} | "
                    f"dabe_weight={effective_despl_weight:.2f} | "
                    f"teacher_weight={effective_teacher_weight:.2f} | "
                    f"schedule_dabe_weight={fixed_weight:.2f} | "
                    f"schedule_teacher_weight={teacher_weight:.2f} | "
                    "fixed_used_for_training=False"
                )
                if str(getattr(cfg, "DABE_VERSION", "v2")).lower() == "gc":
                    logger.log(
                        f"[DABE-GC] epoch={epoch:03d} | "
                        "pseudo_source=dabe_gc_cache | "
                        f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'dabe_gc_only')} | "
                        f"p_dabe_area_mean={dabe_p_area_sum / max(dabe_num_samples, 1):.6f} | "
                        f"p_despl_area_mean={despl_p_despl_area_sum / max(despl_num_samples, 1):.6f} | "
                        f"p_fixed_area_mean={despl_p_fixed_area_sum / max(despl_num_samples, 1):.6f} | "
                        f"fixed_weight={fixed_weight:.2f} | "
                        f"teacher_weight={teacher_weight:.2f} | "
                        "fixed_used_for_training=False"
                    )
            if use_dabe_aware:
                stat_batches = max(dabe_aware_stat_batches, 1)
                logger.log(
                    f"[DABE-Aware] epoch={epoch:03d} | "
                    f"dabe_weight={effective_despl_weight:.2f} | "
                    f"teacher_weight={effective_teacher_weight:.2f} | "
                    f"p_dabe_area_mean={dabe_p_area_sum / max(dabe_num_samples, 1):.6f} | "
                    f"fg_core_area_mean={dabe_aware_fg_core_area_sum / stat_batches:.6f} | "
                    f"bg_core_area_mean={dabe_aware_bg_core_area_sum / stat_batches:.6f} | "
                    f"uncertain_area_mean={dabe_aware_uncertain_area_sum / stat_batches:.6f} | "
                    f"evidence_mean={dabe_aware_evidence_sum / stat_batches:.6f} | "
                    f"target_area_mean={dabe_aware_target_area_sum / stat_batches:.6f} | "
                    f"weight_map_mean={dabe_aware_weight_map_sum / stat_batches:.6f} | "
                    f"loss_final_bce={dabe_aware_loss_final_bce_sum / stat_batches:.6f} | "
                    f"loss_tversky={dabe_aware_loss_tversky_sum / stat_batches:.6f} | "
                    f"loss_area_guard={dabe_aware_loss_area_guard_sum / stat_batches:.6f} | "
                    f"loss_coarse_aux={total_ndr_coarse_aux_loss / max(num_batches, 1):.6f} | "
                    f"loss_base_aux={total_aux_base_loss / max(num_batches, 1):.6f}"
                )
            if use_dabe_pu:
                stat_batches = max(dabe_pu_stat_batches, 1)
                if use_dabe_oem:
                    static_weight_log, teacher_weight_log = 1.0, 0.0
                elif use_dabe_pu_balanced_v2:
                    static_weight_log, teacher_weight_log = get_dabe_pu_balanced_v2_schedule(epoch, cfg)
                elif use_dabe_pu_despl_sched:
                    static_weight_log, teacher_weight_log = get_dabe_pu_despl_schedule(epoch, cfg)
                else:
                    static_weight_log, teacher_weight_log = get_dabe_pu_schedule(epoch, cfg)
                logger.log(
                    f"[DABE-PU] epoch={epoch:03d} | "
                    "use_dabe_pu=True | "
                    f"dabe_pu_version={getattr(cfg, 'DABE_PU_VERSION', 'pu_v11')} | "
                    f"p_init_mode={getattr(cfg, 'P_INIT_MODE', 'dabe_pu_v11')} | "
                    f"static_weight={static_weight_log:.2f} | "
                    f"teacher_weight={teacher_weight_log:.2f} | "
                    f"target_soft_mean={dabe_pu_target_mean_sum / stat_batches:.6f} | "
                    f"weight_map_mean={dabe_pu_weight_mean_sum / stat_batches:.6f} | "
                    f"fg_core_mean={dabe_pu_fg_core_mean_sum / stat_batches:.6f} | "
                    f"fg_fallback_mean={dabe_pu_fg_fallback_mean_sum / stat_batches:.6f} | "
                    f"bg_core_mean={dabe_pu_bg_core_mean_sum / stat_batches:.6f} | "
                    f"extent_mean={dabe_pu_extent_mean_sum / stat_batches:.6f} | "
                    f"unknown_mean={dabe_pu_unknown_mean_sum / stat_batches:.6f} | "
                    f"loss_static_final={dabe_pu_static_final_loss_sum / stat_batches:.6f} | "
                    f"loss_static_coarse={dabe_pu_static_coarse_loss_sum / stat_batches:.6f} | "
                    f"loss_static_base={dabe_pu_static_base_loss_sum / stat_batches:.6f} | "
                    f"loss_static_group={dabe_pu_static_group_loss_sum / stat_batches:.6f} | "
                    f"loss_teacher_final={dabe_pu_teacher_final_loss_sum / stat_batches:.6f} | "
                    f"loss_teacher_coarse={dabe_pu_teacher_coarse_loss_sum / stat_batches:.6f} | "
                    f"loss_teacher_base={dabe_pu_teacher_base_loss_sum / stat_batches:.6f} | "
                    f"loss_teacher_group={dabe_pu_teacher_group_loss_sum / stat_batches:.6f} | "
                    f"teacher_conf_ratio={dabe_pu_teacher_conf_ratio_sum / stat_batches:.6f} | "
                    f"teacher_fg_ratio={dabe_pu_teacher_fg_ratio_sum / stat_batches:.6f} | "
                    f"teacher_bg_ratio={dabe_pu_teacher_bg_ratio_sum / stat_batches:.6f} | "
                    "fixed_used_for_training=False"
                )
                if use_dabe_pu_despl_sched:
                    logger.log(
                        f"[DABE-PU-DesplSched] epoch={epoch:03d} | "
                        f"static_target_mode={dabe_pu_despl_static_target_mode} | "
                        f"teacher_target_mode={dabe_pu_despl_teacher_target_mode} | "
                        f"static_weight={static_weight_log:.2f} | "
                        f"teacher_weight={teacher_weight_log:.2f} | "
                        f"is_teacher_only={bool(static_weight_log <= 1e-8 and teacher_weight_log >= 1.0 - 1e-8)} | "
                        f"target_soft_mean={dabe_pu_target_mean_sum / stat_batches:.6f} | "
                        f"target_hard_area_mean={dabe_pu_target_hard_area_sum / stat_batches:.6f} | "
                        f"weight_map_mean={dabe_pu_weight_mean_sum / stat_batches:.6f} | "
                        f"teacher_prob_mean={teacher_prob_mean_sum / max(num_batches, 1):.6f} | "
                        f"teacher_prob_min={(teacher_prob_min if teacher_prob_min is not None else 0.0):.6f} | "
                        f"teacher_prob_max={(teacher_prob_max if teacher_prob_max is not None else 0.0):.6f} | "
                        f"teacher_soft_target_mean={teacher_soft_target_mean_sum / max(num_batches, 1):.6f} | "
                        f"teacher_binary_area_mean={teacher_pred_area_sum / max(num_batches, 1):.6f} | "
                        f"teacher_binary_area_mean_for_debug_only={teacher_pred_area_sum / max(num_batches, 1):.6f} | "
                        f"loss_static_final={dabe_pu_static_final_loss_sum / stat_batches:.6f} | "
                        f"loss_static_coarse={dabe_pu_static_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_static_base={dabe_pu_static_base_loss_sum / stat_batches:.6f} | "
                        f"loss_static_group={dabe_pu_static_group_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_final={dabe_pu_teacher_final_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_coarse={dabe_pu_teacher_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_base={dabe_pu_teacher_base_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_group={dabe_pu_teacher_group_loss_sum / stat_batches:.6f} | "
                        f"loss_total={total_loss / max(num_batches, 1):.6f}"
                    )
                if use_rast:
                    rast_batches = max(rast_stat_batches, 1)
                    logger.log(
                        f"[RAST] epoch={epoch:03d} | "
                        f"rast_scale={rast_scale_sum / rast_batches:.6f} | "
                        f"rast_pre_reset_scale={rast_pre_reset_scale_sum / rast_batches:.6f} | "
                        f"rast_post_reset_scale={rast_post_reset_scale_sum / rast_batches:.6f} | "
                        f"rast_scale_effective={rast_scale_effective_sum / rast_batches:.6f} | "
                        f"rast_post_reset_enable={bool(getattr(cfg, 'RAST_POST_RESET_ENABLE', False))} | "
                        f"rast_post_reset_conflict_only={bool(getattr(cfg, 'RAST_POST_RESET_CONFLICT_ONLY', False))} | "
                        f"rast_fg_core_area={rast_fg_core_area_sum / rast_batches:.6f} | "
                        f"rast_bg_core_area={rast_bg_core_area_sum / rast_batches:.6f} | "
                        f"rast_extent_area={rast_extent_area_sum / rast_batches:.6f} | "
                        f"rast_unknown_area={rast_unknown_area_sum / rast_batches:.6f} | "
                        f"rast_fg_conflict_ratio={rast_fg_conflict_ratio_sum / rast_batches:.6f} | "
                        f"rast_bg_conflict_ratio={rast_bg_conflict_ratio_sum / rast_batches:.6f} | "
                        f"rast_teacher_map_mean={rast_teacher_map_mean_sum / rast_batches:.6f} | "
                        f"rast_teacher_map_min={(rast_teacher_map_min if rast_teacher_map_min is not None else 1.0):.6f} | "
                        f"rast_teacher_map_max={(rast_teacher_map_max if rast_teacher_map_max is not None else 1.0):.6f} | "
                        f"rast_teacher_map_fg_core_mean={rast_teacher_map_fg_core_mean_sum / rast_batches:.6f} | "
                        f"rast_teacher_map_bg_core_mean={rast_teacher_map_bg_core_mean_sum / rast_batches:.6f} | "
                        f"rast_teacher_map_extent_mean={rast_teacher_map_extent_mean_sum / rast_batches:.6f} | "
                        f"rast_teacher_map_unknown_mean={rast_teacher_map_unknown_mean_sum / rast_batches:.6f} | "
                        f"loss_static_final={dabe_pu_static_final_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_final={dabe_pu_teacher_final_loss_sum / stat_batches:.6f} | "
                        f"loss_static_coarse={dabe_pu_static_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_coarse={dabe_pu_teacher_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_static_base={dabe_pu_static_base_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_base={dabe_pu_teacher_base_loss_sum / stat_batches:.6f} | "
                        f"student_pred_area_mean={student_pred_area_sum / stat_batches:.6f} | "
                        f"teacher_pred_area_mean={teacher_pred_area_sum / stat_batches:.6f}"
                    )
                if use_esa_asym:
                    esa_asym_batches = max(esa_asym_stat_batches, 1)
                    logger.log(
                        f"[ESA-Asym] epoch={epoch:03d} | "
                        f"esa_asym_scale={esa_asym_scale_sum / esa_asym_batches:.6f} | "
                        f"esa_margin_mean={esa_margin_mean_sum / esa_asym_batches:.6f} | "
                        f"esa_margin_min={(esa_margin_min if esa_margin_min is not None else 0.0):.6f} | "
                        f"esa_margin_max={(esa_margin_max if esa_margin_max is not None else 0.0):.6f} | "
                        f"esa_margin_extent_mean={esa_margin_extent_mean_sum / esa_asym_batches:.6f} | "
                        f"esa_margin_extent_teacher_fg_mean={esa_margin_extent_teacher_fg_mean_sum / esa_asym_batches:.6f} | "
                        f"esa_margin_extent_teacher_bg_mean={esa_margin_extent_teacher_bg_mean_sum / esa_asym_batches:.6f} | "
                        f"extent_teacher_fg_ratio={esa_extent_teacher_fg_ratio_sum / esa_asym_batches:.6f} | "
                        f"extent_teacher_bg_ratio={esa_extent_teacher_bg_ratio_sum / esa_asym_batches:.6f} | "
                        f"extent_teacher_bg_fg_like_ratio={esa_extent_teacher_bg_fg_like_ratio_sum / esa_asym_batches:.6f} | "
                        f"extent_teacher_bg_ambig_ratio={esa_extent_teacher_bg_ambig_ratio_sum / esa_asym_batches:.6f} | "
                        f"extent_teacher_bg_bg_like_ratio={esa_extent_teacher_bg_bg_like_ratio_sum / esa_asym_batches:.6f} | "
                        f"teacher_map_extent_mean={esa_teacher_map_extent_mean_sum / esa_asym_batches:.6f} | "
                        f"teacher_map_extent_teacher_fg_mean={esa_teacher_map_extent_teacher_fg_mean_sum / esa_asym_batches:.6f} | "
                        f"teacher_map_extent_teacher_bg_mean={esa_teacher_map_extent_teacher_bg_mean_sum / esa_asym_batches:.6f} | "
                        "teacher_map_extent_teacher_bg_fg_like_mean="
                        f"{esa_teacher_map_extent_teacher_bg_fg_like_mean_sum / esa_asym_batches:.6f} | "
                        "teacher_map_extent_teacher_bg_ambig_mean="
                        f"{esa_teacher_map_extent_teacher_bg_ambig_mean_sum / esa_asym_batches:.6f} | "
                        "teacher_map_extent_teacher_bg_bg_like_mean="
                        f"{esa_teacher_map_extent_teacher_bg_bg_like_mean_sum / esa_asym_batches:.6f} | "
                        f"skipped_no_fg_proto={esa_skipped_no_fg_proto_sum} | "
                        f"skipped_no_bg_proto={esa_skipped_no_bg_proto_sum}"
                    )
                if use_hbns_lite:
                    hbns_batches = max(hbns_stat_batches, 1)
                    logger.log(
                        f"[HBNS-lite] epoch={epoch:03d} | "
                        f"hbns_scale={hbns_scale_sum / hbns_batches:.6f} | "
                        f"lambda_hbns_eff={hbns_lambda_sum / hbns_batches:.8f} | "
                        f"hard_bg_ratio={hbns_hard_bg_ratio_sum / hbns_batches:.6f} | "
                        f"hard_bg_raw_ratio={hbns_hard_bg_raw_ratio_sum / hbns_batches:.6f} | "
                        f"hard_bg_ratio_bg_core={hbns_hard_bg_ratio_bg_core_sum / hbns_batches:.6f} | "
                        f"hard_bg_ratio_low_target={hbns_hard_bg_ratio_low_target_sum / hbns_batches:.6f} | "
                        f"hard_bg_ratio_unknown_bg_like={hbns_hard_bg_ratio_unknown_bg_like_sum / hbns_batches:.6f} | "
                        f"hard_bg_pixels_mean={hbns_hard_bg_pixels_mean_sum / hbns_batches:.6f} | "
                        f"loss_hbns_final={hbns_loss_final_sum / hbns_batches:.6f} | "
                        f"loss_hbns_coarse={hbns_loss_coarse_sum / hbns_batches:.6f} | "
                        f"loss_hbns_base={hbns_loss_base_sum / hbns_batches:.6f} | "
                        f"loss_hbns={hbns_loss_sum / hbns_batches:.6f}"
                    )
                if use_epr_pos:
                    epr_batches = max(epr_stat_batches, 1)
                    logger.log(
                        f"[EPR-pos] epoch={epoch:03d} | "
                        f"epr_scale={epr_scale_sum / epr_batches:.6f} | "
                        f"lambda_epr_eff={epr_lambda_sum / epr_batches:.8f} | "
                        f"epr_pos_ratio={epr_pos_ratio_sum / epr_batches:.6f} | "
                        f"epr_pos_raw_ratio={epr_pos_raw_ratio_sum / epr_batches:.6f} | "
                        f"epr_pos_pixels_mean={epr_pos_pixels_mean_sum / epr_batches:.6f} | "
                        f"epr_valid_image_ratio={epr_valid_image_ratio_sum / epr_batches:.6f} | "
                        f"epr_extent_area={epr_extent_area_sum / epr_batches:.6f} | "
                        f"epr_unknown_overlap_ratio={epr_unknown_overlap_ratio_sum / epr_batches:.6f} | "
                        f"epr_margin_mean={epr_margin_mean_sum / epr_batches:.6f} | "
                        f"epr_margin_min={(epr_margin_min if epr_margin_min is not None else 0.0):.6f} | "
                        f"epr_margin_max={(epr_margin_max if epr_margin_max is not None else 0.0):.6f} | "
                        f"epr_margin_pos_mean={epr_margin_pos_mean_sum / epr_batches:.6f} | "
                        f"epr_teacher_conf_pos_mean={epr_teacher_conf_pos_mean_sum / epr_batches:.6f} | "
                        f"loss_epr_final={epr_loss_final_sum / epr_batches:.6f} | "
                        f"loss_epr_coarse={epr_loss_coarse_sum / epr_batches:.6f} | "
                        f"loss_epr_base={epr_loss_base_sum / epr_batches:.6f} | "
                        f"loss_epr={epr_loss_sum / epr_batches:.6f}"
                    )
                if (
                    use_esa_diagnostic
                    and int(epoch) % max(1, int(getattr(cfg, "ESA_DIAG_LOG_INTERVAL_EPOCH", 1))) == 0
                ):
                    esa_batches = max(esa_stat_batches, 1)
                    logger.log(
                        f"[ESA-Diag] epoch={epoch:03d} | "
                        f"student_prob_fg_core={esa_student_prob_fg_core_sum / esa_batches:.6f} | "
                        f"student_prob_bg_core={esa_student_prob_bg_core_sum / esa_batches:.6f} | "
                        f"student_prob_extent={esa_student_prob_extent_sum / esa_batches:.6f} | "
                        f"student_prob_unknown={esa_student_prob_unknown_sum / esa_batches:.6f} | "
                        f"teacher_fg_fg_core={esa_teacher_fg_fg_core_sum / esa_batches:.6f} | "
                        f"teacher_fg_bg_core={esa_teacher_fg_bg_core_sum / esa_batches:.6f} | "
                        f"teacher_fg_extent={esa_teacher_fg_extent_sum / esa_batches:.6f} | "
                        f"teacher_fg_unknown={esa_teacher_fg_unknown_sum / esa_batches:.6f} | "
                        f"teacher_conf_fg_core={esa_teacher_conf_fg_core_sum / esa_batches:.6f} | "
                        f"teacher_conf_bg_core={esa_teacher_conf_bg_core_sum / esa_batches:.6f} | "
                        f"teacher_conf_extent={esa_teacher_conf_extent_sum / esa_batches:.6f} | "
                        f"teacher_conf_unknown={esa_teacher_conf_unknown_sum / esa_batches:.6f} | "
                        f"teacher_loss_fg_core={esa_teacher_loss_fg_core_sum / esa_batches:.6f} | "
                        f"teacher_loss_bg_core={esa_teacher_loss_bg_core_sum / esa_batches:.6f} | "
                        f"teacher_loss_extent={esa_teacher_loss_extent_sum / esa_batches:.6f} | "
                        f"teacher_loss_unknown={esa_teacher_loss_unknown_sum / esa_batches:.6f}"
                    )
                if use_dabe_oem:
                    oem_batches = max(oem_stat_batches, 1)
                    logger.log(
                        f"[DABE-OEM] epoch={epoch:03d} | "
                        f"lambda_dyn_pos={oem_lambda_dyn_pos_sum / oem_batches:.6f} | "
                        f"lambda_dyn_bg={oem_lambda_dyn_bg_sum / oem_batches:.6f} | "
                        f"seed_fg_area_mean={oem_seed_fg_area_sum / oem_batches:.6f} | "
                        f"seed_bg_area_mean={oem_seed_bg_area_sum / oem_batches:.6f} | "
                        f"extent_area_mean={oem_extent_area_sum / oem_batches:.6f} | "
                        f"unknown_area_mean={oem_unknown_area_sum / oem_batches:.6f} | "
                        f"loss_seed_fg={oem_loss_seed_fg_sum / oem_batches:.6f} | "
                        f"loss_seed_fg_fallback={oem_loss_seed_fg_fallback_sum / oem_batches:.6f} | "
                        f"loss_seed_bg={oem_loss_seed_bg_sum / oem_batches:.6f} | "
                        f"loss_seed_final={oem_loss_seed_final_sum / oem_batches:.6f} | "
                        f"loss_seed_coarse={oem_loss_seed_coarse_sum / oem_batches:.6f} | "
                        f"loss_seed_base={oem_loss_seed_base_sum / oem_batches:.6f} | "
                        f"loss_seed_group={oem_loss_seed_group_sum / oem_batches:.6f} | "
                        f"oem_pos_raw_ratio={oem_pos_raw_ratio_sum / oem_batches:.6f} | "
                        f"oem_pos_capped_ratio={oem_pos_capped_ratio_sum / oem_batches:.6f} | "
                        f"oem_bg_raw_ratio={oem_bg_raw_ratio_sum / oem_batches:.6f} | "
                        f"oem_bg_capped_ratio={oem_bg_capped_ratio_sum / oem_batches:.6f} | "
                        f"oem_skip_no_fg_proto={oem_skip_no_fg_proto_sum} | "
                        f"oem_skip_no_bg_proto={oem_skip_no_bg_proto_sum} | "
                        f"oem_skip_no_pos_region={oem_skip_no_pos_region_sum} | "
                        f"proto_delta_mean={oem_proto_delta_mean_sum / oem_batches:.6f} | "
                        f"proto_delta_min={(oem_proto_delta_min if oem_proto_delta_min is not None else 0.0):.6f} | "
                        f"proto_delta_max={(oem_proto_delta_max if oem_proto_delta_max is not None else 0.0):.6f} | "
                        f"teacher_prob_37_mean={oem_teacher_prob_37_mean_sum / oem_batches:.6f} | "
                        f"teacher_prob_37_fg_seed_mean={oem_teacher_prob_37_fg_seed_sum / oem_batches:.6f} | "
                        f"teacher_prob_37_bg_seed_mean={oem_teacher_prob_37_bg_seed_sum / oem_batches:.6f} | "
                        f"teacher_prob_37_extent_mean={oem_teacher_prob_37_extent_sum / oem_batches:.6f} | "
                        f"loss_dyn_pos={oem_loss_dyn_pos_sum / oem_batches:.6f} | "
                        f"loss_dyn_bg={oem_loss_dyn_bg_sum / oem_batches:.6f} | "
                        f"loss_total={total_loss / max(num_batches, 1):.6f}"
                    )
                if use_dabe_pu_balanced_v2:
                    bal_batches = max(dabe_pu_bal_stat_batches, 1)
                    logger.log(
                        f"[DABE-PU-BalV2] epoch={epoch:03d} | "
                        f"static_weight={static_weight_log:.2f} | "
                        f"teacher_weight={teacher_weight_log:.2f} | "
                        f"fg_core_area_mean={dabe_pu_fg_core_mean_sum / stat_batches:.6f} | "
                        f"fg_fallback_area_mean={dabe_pu_fg_fallback_mean_sum / stat_batches:.6f} | "
                        f"bg_core_area_mean={dabe_pu_bg_core_mean_sum / stat_batches:.6f} | "
                        f"extent_area_mean={dabe_pu_extent_mean_sum / stat_batches:.6f} | "
                        f"unknown_area_mean={dabe_pu_unknown_mean_sum / stat_batches:.6f} | "
                        f"loss_static_fg={dabe_pu_bal_static_fg_loss_sum / bal_batches:.6f} | "
                        f"loss_static_fg_fallback={dabe_pu_bal_static_fg_fallback_loss_sum / bal_batches:.6f} | "
                        f"loss_static_bg={dabe_pu_bal_static_bg_loss_sum / bal_batches:.6f} | "
                        f"loss_static_extent={dabe_pu_bal_static_extent_loss_sum / bal_batches:.6f} | "
                        f"loss_static_final={dabe_pu_static_final_loss_sum / stat_batches:.6f} | "
                        f"loss_static_coarse={dabe_pu_static_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_static_base={dabe_pu_static_base_loss_sum / stat_batches:.6f} | "
                        f"loss_static_group={dabe_pu_static_group_loss_sum / stat_batches:.6f} | "
                        f"teacher_fg_ratio_raw={dabe_pu_bal_teacher_fg_raw_ratio_sum / bal_batches:.6f} | "
                        f"teacher_bg_ratio_raw={dabe_pu_bal_teacher_bg_raw_ratio_sum / bal_batches:.6f} | "
                        f"teacher_bg_ratio_capped={dabe_pu_bal_teacher_bg_capped_ratio_sum / bal_batches:.6f} | "
                        f"teacher_conf_ratio_capped={dabe_pu_bal_teacher_conf_capped_ratio_sum / bal_batches:.6f} | "
                        f"loss_teacher_fg={dabe_pu_bal_teacher_fg_loss_sum / bal_batches:.6f} | "
                        f"loss_teacher_bg={dabe_pu_bal_teacher_bg_loss_sum / bal_batches:.6f} | "
                        f"loss_teacher_final={dabe_pu_teacher_final_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_coarse={dabe_pu_teacher_coarse_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_base={dabe_pu_teacher_base_loss_sum / stat_batches:.6f} | "
                        f"loss_teacher_group={dabe_pu_teacher_group_loss_sum / stat_batches:.6f}"
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
            if use_view_consistency(cfg):
                stat_batches = max(mv_stat_batches, 1)
                if torch.cuda.is_available():
                    mv_peak_mem = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                else:
                    mv_peak_mem = 0.0
                logger.log(
                    f"[MVFlip] epoch={epoch:03d} | "
                    f"lambda_view={mv_lambda_sum / stat_batches:.8f} | "
                    f"loss_view={mv_loss_sum / stat_batches:.8f} | "
                    f"core_ratio={mv_core_ratio_sum / stat_batches:.8f} | "
                    f"mean_abs_diff={mv_mean_abs_diff_sum / stat_batches:.8f} | "
                    f"peak_memory_mb={mv_peak_mem:.2f}"
                )
            if use_proto:
                stat_batches = max(proto_stat_batches, 1)
                if torch.cuda.is_available():
                    proto_peak_mem = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                else:
                    proto_peak_mem = 0.0
                proto_mode = str(getattr(cfg, "PROTO_MODE", "global")).lower()
                proto_valid_avg = proto_valid_ratio_sum / stat_batches
                proto_fg_core_avg = proto_fg_core_ratio_sum / stat_batches
                proto_pixel_avg = proto_pixel_sum / stat_batches
                if proto_mode == "hard_selective":
                    proto_bg_hard_avg = proto_bg_hard_ratio_sum / stat_batches
                    proto_sep_active_avg = proto_sep_active_ratio_sum / stat_batches
                    logger.log(
                        f"[MVProto-HS] epoch={epoch:03d} | "
                        f"lambda_proto={proto_lambda_sum / stat_batches:.8f} | "
                        f"loss_proto={proto_loss_sum / stat_batches:.8f} | "
                        f"align_loss={proto_align_sum / stat_batches:.8f} | "
                        f"sep_loss={proto_sep_sum / stat_batches:.8f} | "
                        f"pixel_loss={proto_pixel_avg:.8f} | "
                        f"pixel_fg_loss={proto_pixel_fg_sum / stat_batches:.8f} | "
                        f"pixel_bg_loss={proto_pixel_bg_sum / stat_batches:.8f} | "
                        f"valid_ratio={proto_valid_avg:.8f} | "
                        f"fg_core_ratio={proto_fg_core_avg:.8f} | "
                        f"bg_hard_ratio={proto_bg_hard_avg:.8f} | "
                        f"bg_ring_ratio={proto_bg_ring_ratio_sum / stat_batches:.8f} | "
                        f"bg_disagree_ratio={proto_bg_disagree_ratio_sum / stat_batches:.8f} | "
                        f"bg_residual_ratio={proto_bg_residual_ratio_sum / stat_batches:.8f} | "
                        f"hard_fg_ratio={proto_hard_fg_ratio_sum / stat_batches:.8f} | "
                        f"hard_bg_ratio={proto_hard_bg_ratio_sum / stat_batches:.8f} | "
                        f"fg_fallback_ratio={proto_fg_fallback_ratio_sum / stat_batches:.8f} | "
                        f"sep_active_ratio={proto_sep_active_avg:.8f} | "
                        f"cos_fg_view={proto_cos_fg_view_sum / stat_batches:.8f} | "
                        f"cos_bg_view={proto_cos_bg_view_sum / stat_batches:.8f} | "
                        f"cos_fg_bg={proto_cos_fg_bg_sum / stat_batches:.8f} | "
                        f"peak_memory_mb={proto_peak_mem:.2f}"
                    )
                    if proto_valid_avg < 0.30:
                        logger.log(f"[MVProto-HS Warning] proto_valid_ratio low: {proto_valid_avg:.8f}")
                    if proto_bg_hard_avg < 0.005:
                        logger.log(f"[MVProto-HS Warning] bg_hard_ratio low: {proto_bg_hard_avg:.8f}")
                    if proto_fg_core_avg < 0.005:
                        logger.log(f"[MVProto-HS Warning] fg_core_ratio low: {proto_fg_core_avg:.8f}")
                    if proto_sep_active_avg == 0.0 and proto_pixel_avg < 1e-8:
                        logger.log("[MVProto-HS Warning] sep_active_ratio=0 and pixel_loss is near zero.")
                else:
                    logger.log(
                        f"[MVProto] epoch={epoch:03d} | "
                        f"lambda_proto={proto_lambda_sum / stat_batches:.8f} | "
                        f"loss_proto={proto_loss_sum / stat_batches:.8f} | "
                        f"align_loss={proto_align_sum / stat_batches:.8f} | "
                        f"sep_loss={proto_sep_sum / stat_batches:.8f} | "
                        f"pixel_loss={proto_pixel_avg:.8f} | "
                        f"valid_ratio={proto_valid_avg:.8f} | "
                        f"fg_core_ratio={proto_fg_core_avg:.8f} | "
                        f"bg_core_ratio={proto_bg_core_ratio_sum / stat_batches:.8f} | "
                        f"cos_fg_view={proto_cos_fg_view_sum / stat_batches:.8f} | "
                        f"cos_bg_view={proto_cos_bg_view_sum / stat_batches:.8f} | "
                        f"cos_fg_bg={proto_cos_fg_bg_sum / stat_batches:.8f} | "
                        f"peak_memory_mb={proto_peak_mem:.2f}"
                    )
            if use_tadr_router(cfg):
                stat_batches = max(tadr_stat_batches, 1)
                logger.log(
                    f"[TADR] epoch={epoch:03d} | "
                    f"tadr_router_mean={tadr_router_mean_sum / stat_batches:.8f} | "
                    f"tadr_router_min={(tadr_router_min if tadr_router_min is not None else -1.0):.8f} | "
                    f"tadr_router_max={(tadr_router_max if tadr_router_max is not None else -1.0):.8f} | "
                    f"tadr_base_gate_mean={tadr_base_gate_mean_sum / stat_batches:.8f} | "
                    f"tadr_base_gate_min={(tadr_base_gate_min if tadr_base_gate_min is not None else -1.0):.8f} | "
                    f"tadr_base_gate_max={(tadr_base_gate_max if tadr_base_gate_max is not None else -1.0):.8f} | "
                    f"tadr_final_gate_mean={tadr_final_gate_mean_sum / stat_batches:.8f} | "
                    f"tadr_final_gate_min={(tadr_final_gate_min if tadr_final_gate_min is not None else -1.0):.8f} | "
                    f"tadr_final_gate_max={(tadr_final_gate_max if tadr_final_gate_max is not None else -1.0):.8f} | "
                    f"ndr_beta_eff={ndr_beta_sum / max(ndr_stat_batches, 1):.8f} | "
                    f"ndr_residual_abs_mean={ndr_residual_abs_mean_sum / max(ndr_stat_batches, 1):.8f} | "
                    f"ndr_residual_abs_max={(ndr_residual_abs_max if ndr_residual_abs_max is not None else -1.0):.8f} | "
                    f"loss_final={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_coarse_aux={total_ndr_coarse_aux_loss / max(num_batches, 1):.6f} | "
                    f"loss_base_aux={total_aux_base_loss / max(num_batches, 1):.6f}"
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

            if reset_enabled and is_after_epoch_finetune_reset(cfg) and epoch == reset_epoch:
                optimizer, scheduler, global_step, lr_floor_activated_logged = apply_finetune_reset(
                    logger,
                    cfg,
                    epoch,
                    student,
                    teacher,
                    optimizer,
                    scheduler,
                    global_step,
                    lr_floor_activated_logged,
                )


if __name__ == "__main__":
    main()
