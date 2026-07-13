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
from common.tepr import (
    TemporalTeacherMemory,
    build_teacher_region_masks,
    build_tepr_lite_teacher_weight_map,
    build_tepr_lite_v11_teacher_weight_map,
    compute_dino_core_margin_68 as compute_dino_core_margin_68_shared,
    get_tepr_scale,
    histogram_quantile_from_hist,
    temporal_variance_p90_from_hist,
)
from common.utils import (
    Logger,
    cache_status,
    check_cacd_feature_cache,
    check_dabe_pu_cache,
    check_dabe_pseudo_cache,
    check_ccr_cache,
    check_cssd_hr_feature_cache,
    check_despl_light_cache,
    check_despl_paper_cache,
    check_despl_pseudo_bank,
    check_drepp_cache,
    check_hflip_feature_cache,
    check_ml_feature_cache,
    check_qra_cache,
    check_tce_cover_cache,
    check_lceg_cover_cache,
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


def get_linear_scale(epoch, start, ramp_end, stop):
    epoch = int(epoch)
    start = int(start)
    ramp_end = int(ramp_end)
    stop = int(stop)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_ndr_v2_shape_lb_scale(cfg, epoch):
    if not bool(getattr(cfg, "NDR_V2_USE_SHAPE_LOWER_BOUND", False)):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, "NDR_V2_SHAPE_LB_START_EPOCH", 36)),
        int(getattr(cfg, "NDR_V2_SHAPE_LB_RAMP_END_EPOCH", 38)),
        int(getattr(cfg, "NDR_V2_SHAPE_LB_STOP_EPOCH", 46)),
    )


def get_ndr_v2_bg_res_lock_scale(cfg, epoch):
    if not bool(getattr(cfg, "NDR_V2_USE_BG_RES_LOCK", False)):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, "NDR_V2_BG_RES_LOCK_START_EPOCH", 1)),
        int(getattr(cfg, "NDR_V2_BG_RES_LOCK_RAMP_END_EPOCH", 1)),
        int(getattr(cfg, "NDR_V2_BG_RES_LOCK_STOP_EPOCH", int(getattr(cfg, "MAX_EPOCH", 25)) + 1)),
    )


def get_ndr_v2_bg_prob_lock_scale(cfg, epoch):
    if not bool(getattr(cfg, "NDR_V2_USE_BG_PROB_LOCK", False)):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, "NDR_V2_BG_PROB_LOCK_START_EPOCH", 36)),
        int(getattr(cfg, "NDR_V2_BG_PROB_LOCK_RAMP_END_EPOCH", 38)),
        int(getattr(cfg, "NDR_V2_BG_PROB_LOCK_STOP_EPOCH", 46)),
    )


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


def get_cssd_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_CSSD", False)):
        return 0.0
    epoch = int(epoch)
    start = int(getattr(cfg, "CSSD_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "CSSD_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "CSSD_STOP_EPOCH", 36))
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_pa_dagp_edge_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_PA_DAGP", False)):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, "PA_DAGP_START_EPOCH", 7)),
        int(getattr(cfg, "PA_DAGP_RAMP_END_EPOCH", 15)),
        int(getattr(cfg, "PA_DAGP_EDGE_STOP_EPOCH", 36)),
    )


def get_pa_dagp_aux_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_PA_DAGP", False)) or not bool(
        getattr(cfg, "PA_DAGP_USE_AUX_LOSS", True)
    ):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, "PA_DAGP_START_EPOCH", 7)),
        int(getattr(cfg, "PA_DAGP_RAMP_END_EPOCH", 15)),
        int(getattr(cfg, "PA_DAGP_AUX_LOSS_STOP_EPOCH", 21)),
    )


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


def get_esa_post_reset_scale(cfg, epoch):
    if not bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)):
        return 0.0
    if bool(getattr(cfg, "ESA_POST_RESET_RAMP", False)):
        raise RuntimeError("ESA PostReset Keep35 requires ESA_POST_RESET_RAMP=False.")
    start = int(getattr(cfg, "ESA_POST_RESET_START_EPOCH", 21))
    stop = int(getattr(cfg, "ESA_POST_RESET_STOP_EPOCH", 36))
    scale = float(getattr(cfg, "ESA_POST_RESET_SCALE", 1.0))
    if start <= 0 or stop <= start:
        raise RuntimeError(
            f"Invalid ESA post-reset window: start={start}, stop={stop}."
        )
    if not 0.0 <= scale <= 1.0:
        raise RuntimeError(f"ESA_POST_RESET_SCALE must be in [0,1], got {scale}.")
    epoch = int(epoch)
    return scale if start <= epoch < stop else 0.0


def teacher_routing_apply_flag(cfg, branch):
    suffixes = {"final": "FINAL", "coarse": "COARSE_AUX", "base": "BASE_AUX"}
    branch = str(branch).lower()
    if branch not in suffixes:
        raise ValueError(f"Unsupported teacher routing branch: {branch}")
    prefix = "TEPR" if bool(getattr(cfg, "USE_TEPR_LITE", False)) else "RAST"
    return bool(getattr(cfg, f"{prefix}_APPLY_TO_{suffixes[branch]}", True))


def validate_tepr_lite_config(cfg):
    if not bool(getattr(cfg, "USE_TEPR_LITE", False)):
        return
    required = {
        "USE_DABE_PU": bool(getattr(cfg, "USE_DABE_PU", False)),
        "USE_TEACHER_BINARY_FULL_LOSS": bool(getattr(cfg, "USE_TEACHER_BINARY_FULL_LOSS", False)),
        "TEPR_USE_PREUPDATE_STATS": bool(getattr(cfg, "TEPR_USE_PREUPDATE_STATS", False)),
        "TEPR_RESET_MEMORY_AT_FINETUNE_RESET": bool(
            getattr(cfg, "TEPR_RESET_MEMORY_AT_FINETUNE_RESET", False)
        ),
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing:
        raise RuntimeError(f"TEPR-Lite required flags are disabled: {missing}.")
    if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() != "pu_v11":
        raise RuntimeError("TEPR-Lite requires DABE_PU_VERSION='pu_v11'.")
    if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
        raise RuntimeError("TEPR-Lite requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
    if get_dabe_pu_despl_teacher_target_mode(cfg) != "binary":
        raise RuntimeError("TEPR-Lite requires binary teacher target.")
    if get_dabe_pu_despl_static_target_mode(cfg) != "soft":
        raise RuntimeError("TEPR-Lite requires soft DABE-PU static target.")
    forbidden = {
        name: bool(getattr(cfg, name, False))
        for name in (
            "USE_RAST",
            "USE_ESA_ASYM",
            "ESA_POST_RESET_ENABLE",
            "USE_ESA_BER",
            "USE_HBNS_LITE",
            "USE_EPR_POS",
            "USE_TCE",
            "USE_LCEG",
            "USE_CSSD",
            "USE_HR_BFR",
            "USE_PA_DAGP",
        )
    }
    enabled_forbidden = [name for name, enabled in forbidden.items() if enabled]
    if enabled_forbidden:
        raise RuntimeError(f"TEPR-Lite cannot be combined with: {enabled_forbidden}.")
    routing_mode = str(getattr(cfg, "TEPR_ROUTING_MODE", "legacy_v1")).lower()
    if routing_mode not in {"legacy_v1", "state_conditional_asymneg"}:
        raise RuntimeError(f"Unsupported TEPR_ROUTING_MODE={routing_mode!r}.")
    use_v11 = routing_mode == "state_conditional_asymneg"
    if use_v11:
        expected_strings = {
            "TEPR_VERSION": "lite_v1_1_state_conditional_asymneg",
            "TEPR_TEMPORAL_SCOPE": "extent_teacher_bg_only",
            "TEPR_INSUFFICIENT_HISTORY_MODE": "dino_only",
            "TEPR_CORE_MODE": "current_binary_exact",
            "TEPR_UNKNOWN_MODE": "fixed",
        }
        mismatched = {
            name: getattr(cfg, name, None)
            for name, expected in expected_strings.items()
            if str(getattr(cfg, name, "")).lower() != expected
        }
        if mismatched:
            raise RuntimeError(
                f"TEPR-v1.1 string configuration mismatch: {mismatched}; "
                f"expected={expected_strings}."
            )
        expected_flags = {
            "TEPR_USE_TEMPORAL_ON_CORE": False,
            "TEPR_USE_TEMPORAL_ON_EXTENT_FG": False,
            "TEPR_USE_TEMPORAL_ON_EXTENT_BG": True,
            "TEPR_USE_TEMPORAL_ON_UNKNOWN": False,
        }
        flag_mismatch = {
            name: bool(getattr(cfg, name, not expected))
            for name, expected in expected_flags.items()
            if bool(getattr(cfg, name, not expected)) != expected
        }
        if flag_mismatch:
            raise RuntimeError(
                f"TEPR-v1.1 temporal scope flags mismatch: {flag_mismatch}; "
                f"expected={expected_flags}."
            )
        expected_values = {
            "TEPR_CORE_CONFLICT_WEIGHT": 0.20,
            "TEPR_EXTENT_FG_WEIGHT": 1.00,
            "TEPR_EXTENT_BG_WEIGHT_FLOOR": 0.25,
            "TEPR_EXTENT_DINO_LAMBDA": 1.386294,
            "TEPR_UNKNOWN_WEIGHT": 0.50,
            "TEPR_OTHER_WEIGHT": 1.00,
            "TEPR_WEIGHT_MIN": 0.20,
            "TEPR_WEIGHT_MAX": 1.00,
        }
        value_mismatch = {}
        for name, expected in expected_values.items():
            actual = float(getattr(cfg, name, float("nan")))
            if not math.isfinite(actual) or abs(actual - expected) > 1e-8:
                value_mismatch[name] = actual
        if value_mismatch:
            raise RuntimeError(
                f"TEPR-v1.1 numeric configuration mismatch: {value_mismatch}; "
                f"expected={expected_values}."
            )
    start = int(getattr(cfg, "TEPR_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "TEPR_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "TEPR_STOP_EPOCH", 21))
    teacher_only_start = int(getattr(cfg, "DABE_PU_DESPL_TEACHER_ONLY_START", 21))
    if not 0 < start <= ramp_end < stop <= teacher_only_start:
        raise RuntimeError(
            f"Invalid TEPR window: start={start}, ramp_end={ramp_end}, "
            f"stop={stop}, teacher_only={teacher_only_start}."
        )
    update_start = int(getattr(cfg, "TEPR_MEMORY_UPDATE_START_EPOCH", 1))
    update_end = int(getattr(cfg, "TEPR_MEMORY_UPDATE_END_EPOCH", 20))
    if update_start != 1 or update_end != stop - 1:
        raise RuntimeError(
            f"TEPR memory update window must be [1,{stop - 1}], got [{update_start},{update_end}]."
        )
    rho = float(getattr(cfg, "TEPR_TEMPORAL_RHO", 0.90))
    if not 0.0 < rho < 1.0:
        raise RuntimeError(f"TEPR_TEMPORAL_RHO must be in (0,1), got {rho}.")
    if float(getattr(cfg, "TEPR_VARIANCE_TAU", 0.02)) <= 0.0:
        raise RuntimeError("TEPR_VARIANCE_TAU must be positive.")
    if float(getattr(cfg, "TEPR_MARGIN_TAU", 0.05)) <= 0.0:
        raise RuntimeError("TEPR_MARGIN_TAU must be positive.")
    if float(getattr(cfg, "TEPR_CONF_GAMMA", 1.0)) <= 0.0:
        raise RuntimeError("TEPR_CONF_GAMMA must be positive.")
    if not use_v11 and (
        float(getattr(cfg, "TEPR_CORE_LAMBDA", math.log(5.0))) <= 0.0
        or float(getattr(cfg, "TEPR_EXTENT_LAMBDA", math.log(4.0))) <= 0.0
    ):
        raise RuntimeError("TEPR core/extent lambdas must be positive.")
    weight_min = float(getattr(cfg, "TEPR_WEIGHT_MIN", 0.20))
    weight_max = float(getattr(cfg, "TEPR_WEIGHT_MAX", 1.00))
    if not 0.0 <= weight_min <= weight_max <= 1.0:
        raise RuntimeError(f"Invalid TEPR weight range: [{weight_min},{weight_max}].")
    if int(getattr(cfg, "TEPR_MIN_HISTORY", 3)) <= 0:
        raise RuntimeError("TEPR_MIN_HISTORY must be positive.")
    if str(getattr(cfg, "TEPR_MEMORY_DTYPE", "float16")).lower() not in {"float16", "float32"}:
        raise RuntimeError("TEPR_MEMORY_DTYPE must be float16 or float32.")
    if not use_v11:
        unknown_min = float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT_MIN", 0.50))
        unknown_max = float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT_MAX", 0.80))
        extent_min = float(getattr(cfg, "TEPR_EXTENT_TEMPORAL_MIN", 0.60))
        if not weight_min <= unknown_min <= unknown_max <= weight_max:
            raise RuntimeError("Invalid TEPR unknown weight range.")
        if not weight_min <= extent_min <= weight_max:
            raise RuntimeError("Invalid TEPR extent temporal minimum.")
    if not all(teacher_routing_apply_flag(cfg, branch) for branch in ("final", "coarse", "base")):
        raise RuntimeError("TEPR-Lite requires final/coarse/base teacher routing enabled.")
    if int(getattr(cfg, "LOSS_SIZE", -1)) != 68:
        raise RuntimeError("TEPR-Lite requires LOSS_SIZE=68.")
    if int(getattr(cfg, "MAX_EPOCH", -1)) not in {35, 40, 45}:
        raise RuntimeError("TEPR-Lite requires MAX_EPOCH in {35, 40, 45}.")
    if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "dagp_safe" or not bool(
        getattr(cfg, "USE_NDR_BRANCH", False)
    ) or bool(getattr(cfg, "USE_NDR_V2", False)):
        raise RuntimeError("TEPR-Lite requires the original DAGP-Safe + NDR-v1 head.")
    if get_reset_epoch(cfg) != 20 or finetune_reset_timing(cfg) != "after_epoch":
        raise RuntimeError("TEPR-Lite requires epoch20 after-epoch finetune reset.")
    if not bool(getattr(cfg, "FINETUNE_RESET_TEACHER", False)):
        raise RuntimeError("TEPR-Lite requires FINETUNE_RESET_TEACHER=True.")


def build_configured_tepr_teacher_weight_map(
    cfg,
    batch,
    teacher_prob,
    temporal_mean,
    temporal_second,
    history_count,
    epoch,
    device,
):
    routing_mode = str(getattr(cfg, "TEPR_ROUTING_MODE", "legacy_v1")).lower()
    if routing_mode == "state_conditional_asymneg":
        return build_tepr_lite_v11_teacher_weight_map(
            cfg,
            batch,
            teacher_prob,
            temporal_mean,
            temporal_second,
            history_count,
            epoch,
            device,
        )
    if routing_mode == "legacy_v1":
        return build_tepr_lite_teacher_weight_map(
            cfg,
            batch,
            teacher_prob,
            temporal_mean,
            temporal_second,
            history_count,
            epoch,
            device,
        )
    raise RuntimeError(f"Unsupported TEPR_ROUTING_MODE={routing_mode!r}.")


def get_tce_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_TCE", False)):
        return 0.0
    start = int(getattr(cfg, "TCE_START_EPOCH", 31))
    ramp_end = int(getattr(cfg, "TCE_RAMP_END_EPOCH", 32))
    stop = int(getattr(cfg, "TCE_STOP_EPOCH", 36))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        denom = max(1, ramp_end - start + 1)
        return float(epoch - start + 1) / float(denom)
    return 1.0


def get_lceg_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_LCEG", False)):
        return 0.0
    start = int(getattr(cfg, "LCEG_START_EPOCH", 26))
    ramp_end = int(getattr(cfg, "LCEG_RAMP_END_EPOCH", 28))
    stop = int(getattr(cfg, "LCEG_STOP_EPOCH", 36))
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
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {"dagp_safe", "dagp_safe_csd_v1r"}


def use_csd_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "csd_v1"


def use_csd_v1r_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp_safe_csd_v1r"


def use_hr_bfr(cfg):
    return bool(getattr(cfg, "USE_HR_BFR", False))


def use_cssd(cfg):
    return bool(getattr(cfg, "USE_CSSD", False))


def use_cacd(cfg):
    return bool(getattr(cfg, "USE_CACD", False)) or str(
        getattr(cfg, "HEAD_TYPE", "simple")
    ).lower() == "cacd_v1_base"


def use_pa_dagp(cfg):
    return bool(getattr(cfg, "USE_PA_DAGP", False))


def use_raw_feature_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {
        "dagp",
        "dagp_safe",
        "csd_v1",
        "dagp_safe_csd_v1r",
        "cacd_v1_base",
    }


def use_ndr_branch(cfg):
    return bool(getattr(cfg, "USE_NDR_BRANCH", False))


def use_ndr_v2(cfg):
    return bool(getattr(cfg, "USE_NDR_V2", False))


def use_tadr_router(cfg):
    return bool(getattr(cfg, "USE_TADR_ROUTER", False))


def set_model_epoch(model, epoch):
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "set_epoch"):
        target.set_epoch(epoch)


def make_image_68(cfg, batch, device):
    if not (use_ndr_branch(cfg) or use_csd_head(cfg) or use_csd_v1r_head(cfg) or use_cacd(cfg)):
        return None
    if "image_68" not in batch:
        raise KeyError("NDR/CSD decoder requires batch['image_68'].")
    return batch["image_68"].to(device, non_blocking=True).float()


def make_sobel_68(cfg, batch, device):
    if not use_cacd(cfg):
        return None
    if "sobel_68" not in batch:
        raise KeyError("CACD-v1-Base requires batch['sobel_68'] computed from image_68.")
    sobel = batch["sobel_68"].to(device, non_blocking=True).float()
    if sobel.ndim != 4 or list(sobel.shape[1:]) != [1, 68, 68]:
        raise RuntimeError(f"CACD sobel_68 shape mismatch: {list(sobel.shape)}")
    return sobel


def make_image_136(cfg, batch, device):
    if not use_hr_bfr(cfg):
        return None
    if "image_136" not in batch:
        raise KeyError("HR-BFR requires batch['image_136'] loaded from original image resize.")
    return batch["image_136"].to(device, non_blocking=True).float()


def make_hflip_image_68(cfg, batch, device):
    if not use_ndr_branch(cfg):
        return None
    if "image_hflip_68" not in batch:
        raise KeyError("HFlip view with USE_NDR_BRANCH=True requires batch['image_hflip_68'].")
    return batch["image_hflip_68"].to(device, non_blocking=True).float()


def forward_seg_head(
    model,
    model_input,
    cfg,
    image_68=None,
    image_136=None,
    sobel_68=None,
    return_aux=False,
    bg_reliable_68=None,
    pa_compare_original=False,
):
    if use_cacd(cfg):
        return model(
            model_input,
            image_68=image_68,
            sobel_68=sobel_68,
            return_aux=return_aux,
        )
    if use_csd_v1r_head(cfg):
        return model(
            model_input,
            image_68=image_68,
            image_136=image_136,
            return_aux=return_aux,
            bg_reliable_68=bg_reliable_68,
            pa_compare_original=pa_compare_original,
        )
    if use_csd_head(cfg):
        return model(model_input, image_68=image_68, return_aux=return_aux, bg_reliable_68=bg_reliable_68)
    if use_dagp_safe_head(cfg):
        return model(
            model_input,
            image_68=image_68,
            return_aux=return_aux,
            pa_compare_original=pa_compare_original,
        )
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
    if use_cacd(cfg):
        required = ("feature_l10", "feature_l11", "feature")
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"CACD batch missing feature fields: {missing}")
        features = {
            "f10": batch["feature_l10"].to(device, non_blocking=True).float(),
            "f11": batch["feature_l11"].to(device, non_blocking=True).float(),
            "f12": batch["feature"].to(device, non_blocking=True).float(),
        }
        expected = [384, int(getattr(cfg, "CACD_FEATURE_SIZE", 37)), int(getattr(cfg, "CACD_FEATURE_SIZE", 37))]
        for name, value in features.items():
            if list(value.shape[1:]) != expected or not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(
                    f"CACD {name} input invalid: shape={list(value.shape)}, expected=[B,{expected}], "
                    f"finite={bool(torch.isfinite(value).all().item())}"
                )
        return features
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


def extract_logits_for_eval(output, cfg):
    if isinstance(output, dict) and use_hr_bfr(cfg) and bool(getattr(cfg, "HR_BFR_USE_HR_LOGITS_FOR_EVAL", True)):
        if "hr_logits" not in output:
            raise KeyError("HR_BFR_USE_HR_LOGITS_FOR_EVAL=True but model output has no hr_logits.")
        return output["hr_logits"]
    if isinstance(output, dict):
        if "final_logits" in output:
            return output["final_logits"]
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


def teacher_weighted_bce_with_logits(
    logits,
    target,
    weight_map,
    enabled,
    apply_to_loss=True,
    eps=1e-6,
):
    if not bool(enabled) or not bool(apply_to_loss) or weight_map is None:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    return weighted_bce_with_logits(
        logits,
        target,
        weight_map.to(device=logits.device, dtype=logits.dtype),
        eps=eps,
    )


def build_cacd_anchor_partial_ce(anchor_logits, batch, cfg, device=None):
    """Per-image/per-class balanced partial CE on mutually exclusive PU cores."""
    if anchor_logits.ndim != 4 or int(anchor_logits.shape[1]) != 3:
        raise RuntimeError(f"CACD anchor logits must be [B,3,H,W], got {list(anchor_logits.shape)}")
    target_device = anchor_logits.device if device is None else device
    fg_core_68 = batch["pu_fg_core"].to(target_device, non_blocking=True).float() > 0.5
    bg_core_68 = batch["pu_bg_core"].to(target_device, non_blocking=True).float() > 0.5
    overlap = fg_core_68 & bg_core_68
    overlap_ratio = float(overlap.float().mean().detach().item())
    if overlap_ratio > 1e-6:
        raise RuntimeError(
            f"CACD fg/bg core overlap ratio exceeds 1e-6: {overlap_ratio:.9g}"
        )
    native_size = tuple(int(v) for v in anchor_logits.shape[-2:])
    fg = F.interpolate(fg_core_68.float(), size=native_size, mode="nearest") > 0.5
    bg = F.interpolate(bg_core_68.float(), size=native_size, mode="nearest") > 0.5
    log_prob = F.log_softmax(anchor_logits, dim=1)
    loss_fg_map = -log_prob[:, 0:1]
    loss_bg_map = -log_prob[:, 1:2]
    min_pixels = int(getattr(cfg, "CACD_ANCHOR_MIN_PIXELS_PER_CLASS", 4))
    image_losses = []
    fg_losses = []
    bg_losses = []
    valid_fg_images = 0
    valid_bg_images = 0
    valid_images = 0
    for index in range(int(anchor_logits.shape[0])):
        class_losses = []
        if int(fg[index].sum().item()) >= min_pixels:
            value = loss_fg_map[index][fg[index]].mean()
            class_losses.append(value)
            fg_losses.append(value)
            valid_fg_images += 1
        if int(bg[index].sum().item()) >= min_pixels:
            value = loss_bg_map[index][bg[index]].mean()
            class_losses.append(value)
            bg_losses.append(value)
            valid_bg_images += 1
        if class_losses:
            image_losses.append(torch.stack(class_losses).mean())
            valid_images += 1
    zero = anchor_logits.sum() * 0.0
    loss = torch.stack(image_losses).mean() if image_losses else zero
    loss_fg = torch.stack(fg_losses).mean() if fg_losses else zero
    loss_bg = torch.stack(bg_losses).mean() if bg_losses else zero
    anchor_prob = torch.softmax(anchor_logits.detach(), dim=1)

    def masked_mean(value, mask):
        return float(value[mask].mean().item()) if bool(mask.any().item()) else 0.0

    fg_prediction = anchor_prob.argmax(dim=1, keepdim=True) == 0
    bg_prediction = anchor_prob.argmax(dim=1, keepdim=True) == 1
    fg_acc = masked_mean(fg_prediction.float(), fg)
    bg_acc = masked_mean(bg_prediction.float(), bg)
    valid_acc = [
        value
        for value, count in ((fg_acc, int(fg.sum())), (bg_acc, int(bg.sum())))
        if count > 0
    ]
    stats = {
        "loss_anchor_fg": float(loss_fg.detach().item()),
        "loss_anchor_bg": float(loss_bg.detach().item()),
        "loss_anchor": float(loss.detach().item()),
        "overlap_ratio": overlap_ratio,
        "fg_core_ratio_37": float(fg.float().mean().item()),
        "bg_core_ratio_37": float(bg.float().mean().item()),
        "valid_fg_image_ratio": valid_fg_images / max(1, int(anchor_logits.shape[0])),
        "valid_bg_image_ratio": valid_bg_images / max(1, int(anchor_logits.shape[0])),
        "valid_image_ratio": valid_images / max(1, int(anchor_logits.shape[0])),
        "anchor_fg_core_prob": masked_mean(anchor_prob[:, 0:1], fg),
        "anchor_bg_core_prob": masked_mean(anchor_prob[:, 1:2], bg),
        "anchor_fg_core_acc": fg_acc,
        "anchor_bg_core_acc": bg_acc,
        "anchor_balanced_acc": sum(valid_acc) / len(valid_acc) if valid_acc else 0.0,
    }
    if "pu_extent" in batch:
        extent = F.interpolate(
            batch["pu_extent"].to(target_device, non_blocking=True).float(),
            size=native_size,
            mode="nearest",
        ) > 0.5
        stats.update(
            {
                "anchor_fg_extent_mean": masked_mean(anchor_prob[:, 0:1], extent),
                "anchor_bg_extent_mean": masked_mean(anchor_prob[:, 1:2], extent),
                "anchor_amb_extent_mean": masked_mean(anchor_prob[:, 2:3], extent),
            }
        )
    else:
        stats.update(
            {
                "anchor_fg_extent_mean": 0.0,
                "anchor_bg_extent_mean": 0.0,
                "anchor_amb_extent_mean": 0.0,
            }
        )
    return loss, stats


def masked_mean_per_valid_image(loss_map, mask, min_pixels):
    if loss_map.shape != mask.shape:
        raise RuntimeError(
            f"masked mean shape mismatch: loss={list(loss_map.shape)}, mask={list(mask.shape)}"
        )
    mask_float = mask.float()
    counts = mask_float.flatten(1).sum(dim=1)
    valid = counts >= int(min_pixels)
    per_image = (loss_map * mask_float).flatten(1).sum(dim=1) / counts.clamp_min(1.0)
    if bool(valid.any().item()):
        return per_image[valid].mean(), valid
    return loss_map.sum() * 0.0, valid


def _balanced_valid_losses(loss_a, valid_a, loss_b, valid_b, zero):
    has_a = bool(valid_a.any().item())
    has_b = bool(valid_b.any().item())
    if has_a and has_b:
        return 0.5 * (loss_a + loss_b)
    if has_a:
        return loss_a
    if has_b:
        return loss_b
    return zero


def _masked_scalar_mean(values, mask):
    mask = mask.bool()
    if bool(mask.any().item()):
        return float(values.detach()[mask].float().mean().item())
    return 0.0


def build_pa_dagp_aux_loss(cfg, epoch, student_out, batch, device):
    if not bool(getattr(cfg, "USE_PA_DAGP", False)):
        if isinstance(student_out, dict):
            zero_source = student_out.get("logits")
        else:
            zero_source = student_out
        zero = zero_source.sum() * 0.0
        return zero, {"edge_scale": 0.0, "aux_scale": 0.0, "lambda_pa_eff": 0.0}
    if not isinstance(student_out, dict):
        raise RuntimeError("USE_PA_DAGP=True requires dict student output.")
    required = {
        "pa_raw_polarity",
        "pa_signed_polarity",
        "pa_base_prob",
        "pa_anchor_valid",
        "pa_rho",
        "pa_diag",
    }
    missing = sorted(required - set(student_out))
    if missing:
        raise RuntimeError(f"PA-DAGP student output is missing fields: {missing}")
    if "pu_fg_core" not in batch or "pu_bg_core" not in batch:
        raise RuntimeError("PA-DAGP auxiliary loss requires DABE-PU fg/bg core fields.")

    raw_polarity = student_out["pa_raw_polarity"]
    signed_polarity = student_out["pa_signed_polarity"]
    base_prob = student_out["pa_base_prob"].detach()
    anchor_valid = student_out["pa_anchor_valid"].detach().bool()
    if raw_polarity.ndim != 4 or raw_polarity.shape[1] != 1:
        raise RuntimeError(f"PA-DAGP raw polarity must be [B,1,H,W], got {list(raw_polarity.shape)}")
    if signed_polarity.shape != raw_polarity.shape or base_prob.shape != raw_polarity.shape:
        raise RuntimeError(
            "PA-DAGP polarity/base probability shape mismatch: "
            f"raw={list(raw_polarity.shape)}, signed={list(signed_polarity.shape)}, "
            f"base_prob={list(base_prob.shape)}"
        )
    if anchor_valid.shape != (raw_polarity.shape[0],):
        raise RuntimeError(f"PA-DAGP anchor_valid must be [B], got {list(anchor_valid.shape)}")
    if not bool(torch.isfinite(raw_polarity).all().item()) or not bool(
        torch.isfinite(signed_polarity).all().item()
    ):
        raise RuntimeError("PA-DAGP polarity output contains NaN/Inf.")

    native_size = tuple(raw_polarity.shape[-2:])
    fg_core = batch["pu_fg_core"].to(device, non_blocking=True).float()
    bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float()
    fg_core = F.interpolate(fg_core, size=native_size, mode="nearest") > 0.5
    bg_core = (F.interpolate(bg_core, size=native_size, mode="nearest") > 0.5) & (~fg_core)
    valid_image_mask = anchor_valid.view(-1, 1, 1, 1)
    fg_core = fg_core & valid_image_mask
    bg_core = bg_core & valid_image_mask

    core_margin = float(getattr(cfg, "PA_DAGP_CORE_MARGIN", 0.50))
    hard_margin = float(getattr(cfg, "PA_DAGP_HARD_MARGIN", 0.75))
    min_core = int(getattr(cfg, "PA_DAGP_MIN_CORE_PIXELS_PER_IMAGE", 4))
    min_hard = int(getattr(cfg, "PA_DAGP_MIN_HARD_PIXELS_PER_IMAGE", 1))
    loss_fg, valid_fg = masked_mean_per_valid_image(
        F.softplus(core_margin - raw_polarity), fg_core, min_core
    )
    loss_bg, valid_bg = masked_mean_per_valid_image(
        F.softplus(core_margin + raw_polarity), bg_core, min_core
    )
    zero = raw_polarity.sum() * 0.0
    loss_core = _balanced_valid_losses(loss_fg, valid_fg, loss_bg, valid_bg, zero)

    hard_fg = fg_core & (
        base_prob < float(getattr(cfg, "PA_DAGP_HARD_FG_PROB_THRESH", 0.50))
    )
    hard_bg = bg_core & (
        base_prob > float(getattr(cfg, "PA_DAGP_HARD_BG_PROB_THRESH", 0.50))
    )
    loss_hfg, valid_hfg = masked_mean_per_valid_image(
        F.softplus(hard_margin - raw_polarity), hard_fg, min_hard
    )
    loss_hbg, valid_hbg = masked_mean_per_valid_image(
        F.softplus(hard_margin + raw_polarity), hard_bg, min_hard
    )
    loss_hard = _balanced_valid_losses(loss_hfg, valid_hfg, loss_hbg, valid_hbg, zero)
    loss_raw = loss_core + float(getattr(cfg, "PA_DAGP_HARD_LOSS_MULT", 0.50)) * loss_hard
    aux_scale = float(get_pa_dagp_aux_scale(epoch, cfg))
    lambda_pa_eff = float(getattr(cfg, "PA_DAGP_AUX_LOSS_WEIGHT_MAX", 0.020)) * aux_scale
    # Keep PA parameters completely untouched during auxiliary warmup. Using
    # `0 * loss_raw` would create zero gradients and let AdamW weight decay the
    # optional branch before its configured start epoch.
    loss_weighted = (
        loss_raw * lambda_pa_eff
        if lambda_pa_eff > 0.0
        else raw_polarity.new_zeros(())
    )
    if not bool(torch.isfinite(loss_weighted).item()):
        raise RuntimeError("PA-DAGP auxiliary loss is NaN/Inf.")

    fg_mean = _masked_scalar_mean(raw_polarity, fg_core)
    bg_mean = _masked_scalar_mean(raw_polarity, bg_core)
    diag = {
        key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
        for key, value in student_out["pa_diag"].items()
    }
    stats = {
        **diag,
        "edge_scale": float(get_pa_dagp_edge_scale(epoch, cfg)),
        "aux_scale": aux_scale,
        "lambda_pa_eff": lambda_pa_eff,
        "fg_core_pol_mean": fg_mean,
        "bg_core_pol_mean": bg_mean,
        "core_gap": fg_mean - bg_mean,
        "hard_fg_ratio": float(hard_fg.float().mean().item()),
        "hard_bg_ratio": float(hard_bg.float().mean().item()),
        "hard_fg_pol_mean": _masked_scalar_mean(raw_polarity, hard_fg),
        "hard_bg_pol_mean": _masked_scalar_mean(raw_polarity, hard_bg),
        "loss_pa_fg": float(loss_fg.detach().item()),
        "loss_pa_bg": float(loss_bg.detach().item()),
        "loss_pa_core": float(loss_core.detach().item()),
        "loss_pa_hfg": float(loss_hfg.detach().item()),
        "loss_pa_hbg": float(loss_hbg.detach().item()),
        "loss_pa_hard": float(loss_hard.detach().item()),
        "loss_pa_raw": float(loss_raw.detach().item()),
        "loss_pa_weighted": float(loss_weighted.detach().item()),
        "fg_core_valid_image_ratio": float(valid_fg.float().mean().item()),
        "bg_core_valid_image_ratio": float(valid_bg.float().mean().item()),
    }
    return loss_weighted, stats


def build_csd_bg_reliable_mask(batch, cfg, target_shape, device):
    bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float()
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    if bg_core.shape[-2:] != target_shape:
        bg_core = F.interpolate(bg_core, size=target_shape, mode="nearest")
    if target_soft.shape[-2:] != target_shape:
        target_soft = F.interpolate(target_soft, size=target_shape, mode="bilinear", align_corners=False)
    if weight_map.shape[-2:] != target_shape:
        weight_map = F.interpolate(weight_map, size=target_shape, mode="bilinear", align_corners=False)
    prefix = "CSD_V1R" if use_csd_v1r_head(cfg) else "CSD"
    bg_core_mask = bg_core > float(getattr(cfg, f"{prefix}_BG_CORE_THRESH", 0.5))
    low_target_bg = (
        target_soft < float(getattr(cfg, f"{prefix}_BG_LOW_TARGET_THRESH", 0.15))
    ) & (
        weight_map > float(getattr(cfg, f"{prefix}_BG_LOW_TARGET_WEIGHT_THRESH", 0.50))
    )
    return (bg_core_mask | low_target_bg).detach()


def soft_morph_boundary(prob, radius=2):
    radius = int(radius)
    if radius <= 0:
        return torch.zeros_like(prob)
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=radius)
    return torch.clamp(dilated - eroded, 0.0, 1.0)


def per_image_quantile_map(x, q):
    flat = x.flatten(1)
    numel = flat.shape[1]
    kth = int(math.ceil(max(0.0, min(1.0, float(q))) * float(max(numel - 1, 1)))) + 1
    kth = max(1, min(numel, kth))
    return flat.kthvalue(kth, dim=1).values.view(-1, 1, 1, 1)


def get_csd_boundary_scale(cfg, epoch):
    prefix = "CSD_V1R" if use_csd_v1r_head(cfg) else "CSD"
    if not bool(getattr(cfg, f"{prefix}_USE_BOUNDARY_AUX", False)):
        return 0.0
    return get_linear_scale(
        epoch,
        int(getattr(cfg, f"{prefix}_BOUNDARY_AUX_START_EPOCH", 7)),
        int(getattr(cfg, f"{prefix}_BOUNDARY_AUX_RAMP_END_EPOCH", 15)),
        int(getattr(cfg, f"{prefix}_BOUNDARY_AUX_STOP_EPOCH", 21)),
    )


def build_csd_boundary_target(student_out, batch, cfg):
    if not isinstance(student_out, dict) or "sobel_68" not in student_out:
        raise RuntimeError("CSD boundary aux requires sobel_68 in student output.")
    target_soft = batch["pu_target_soft"].to(student_out["sobel_68"].device, non_blocking=True).float()
    if target_soft.shape[-2:] != student_out["sobel_68"].shape[-2:]:
        target_soft = F.interpolate(target_soft, size=student_out["sobel_68"].shape[-2:], mode="bilinear", align_corners=False)
    boundary_band = soft_morph_boundary(
        target_soft.detach(),
        radius=int(getattr(cfg, "CSD_V1R_BOUNDARY_RADIUS" if use_csd_v1r_head(cfg) else "CSD_BOUNDARY_RADIUS", 2)),
    )
    edge_norm = student_out["sobel_68"].detach()
    edge_q_key = "CSD_V1R_BOUNDARY_EDGE_Q" if use_csd_v1r_head(cfg) else "CSD_BOUNDARY_EDGE_Q"
    edge_thresh = per_image_quantile_map(edge_norm, float(getattr(cfg, edge_q_key, 0.60)))
    edge_support = (edge_norm >= edge_thresh).float()
    return torch.clamp(boundary_band * edge_support, 0.0, 1.0), boundary_band, edge_support


def build_csd_aux_losses(student_out, batch, cfg, epoch, device):
    if not isinstance(student_out, dict):
        zero = torch.tensor(0.0, device=device)
        return zero, zero, {
            "bg_reliable_ratio": 0.0,
            "loss_bg_detail": 0.0,
            "loss_bg_detail_weighted": 0.0,
            "boundary_scale": 0.0,
            "boundary_target_mean": 0.0,
            "boundary_target_max": 0.0,
            "loss_boundary": 0.0,
            "loss_boundary_weighted": 0.0,
        }
    logits = student_out["logits"]
    zero = logits.sum() * 0.0
    csd_scale = output_scalar(student_out, "csd_scale", 0.0)
    bg_loss = zero
    bg_weighted = zero
    bg_reliable_ratio = 0.0
    prefix = "CSD_V1R" if use_csd_v1r_head(cfg) else "CSD"
    if bool(getattr(cfg, f"{prefix}_USE_BG_DETAIL_LOCK", False)) and "csd_detail_injected_68" in student_out:
        bg_reliable = build_csd_bg_reliable_mask(batch, cfg, logits.shape[-2:], device)
        bg_float = bg_reliable.to(dtype=logits.dtype)
        detail_abs = student_out["csd_detail_injected_68"].abs()
        bg_loss = (detail_abs * bg_float).sum() / bg_float.sum().clamp_min(1.0)
        bg_weighted = (
            float(getattr(cfg, f"{prefix}_BG_DETAIL_LOCK_WEIGHT", 0.005))
            * float(csd_scale)
            * bg_loss
        )
        bg_reliable_ratio = float(bg_float.detach().mean().item())
    boundary_loss = zero
    boundary_weighted = zero
    boundary_scale = get_csd_boundary_scale(cfg, epoch)
    boundary_target_mean = 0.0
    boundary_target_max = 0.0
    if bool(getattr(cfg, f"{prefix}_USE_BOUNDARY_AUX", False)) and "boundary_logits" in student_out:
        boundary_target, _, _ = build_csd_boundary_target(student_out, batch, cfg)
        boundary_logits = resize_logits_for_loss(student_out["boundary_logits"], cfg)
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        boundary_weighted = (
            float(getattr(cfg, f"{prefix}_BOUNDARY_AUX_WEIGHT_MAX", 0.020))
            * float(boundary_scale)
            * boundary_loss
        )
        boundary_target_mean = float(boundary_target.detach().mean().item())
        boundary_target_max = float(boundary_target.detach().max().item())
    stats = {
        "bg_reliable_ratio": bg_reliable_ratio,
        "loss_bg_detail": float(bg_loss.detach().item()),
        "loss_bg_detail_weighted": float(bg_weighted.detach().item()),
        "boundary_scale": float(boundary_scale),
        "boundary_target_mean": boundary_target_mean,
        "boundary_target_max": boundary_target_max,
        "loss_boundary": float(boundary_loss.detach().item()),
        "loss_boundary_weighted": float(boundary_weighted.detach().item()),
    }
    return bg_weighted, boundary_weighted, stats


def compute_sobel_mag_1ch(x):
    if x.ndim != 4 or x.shape[1] != 1:
        raise RuntimeError(f"Expected single-channel tensor [B,1,H,W], got {list(x.shape)}.")
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    dx = F.conv2d(x, sobel_x, padding=1)
    dy = F.conv2d(x, sobel_y, padding=1)
    sobel = torch.sqrt(dx * dx + dy * dy + 1e-6)
    sobel = sobel / (sobel.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return sobel.clamp(0.0, 1.0)


def get_hr_bfr_scale(cfg, epoch):
    if not use_hr_bfr(cfg):
        return 0.0
    epoch = int(epoch)
    warmup = int(getattr(cfg, "HR_BFR_WARMUP_EPOCH", 6))
    start = int(getattr(cfg, "HR_BFR_RAMP_START_EPOCH", 7))
    end = int(getattr(cfg, "HR_BFR_RAMP_END_EPOCH", 15))
    if epoch <= warmup or epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    denom = max(1, end - start + 1)
    return float(epoch - start + 1) / float(denom)


def _hr_masked_mean(value, weight, eps=1e-6):
    denom = weight.sum().clamp_min(float(eps))
    return (value * weight).sum() / denom


def build_hr_bfr_losses(student_out, batch, cfg, epoch, static_weight, teacher_weight, teacher_binary, device):
    if not use_hr_bfr(cfg):
        zero = teacher_binary.sum() * 0.0
        return zero, {}
    if not isinstance(student_out, dict) or "hr_logits" not in student_out:
        raise RuntimeError("USE_HR_BFR=True requires model output dict with hr_logits.")

    hr_logits = student_out["hr_logits"]
    anchor_logits = student_out["hr_anchor_logits"].detach()
    band = student_out["hr_band_gate_136"].detach().float()
    if hr_logits.shape[-2:] != band.shape[-2:]:
        raise RuntimeError("HR-BFR logits and band shape mismatch.")

    b = int(hr_logits.shape[0])
    raw_band = student_out.get("hr_band_gate_raw_136", band).detach().float()
    band_per_image = raw_band.flatten(1).mean(dim=1)
    band_ratio = float(band_per_image.mean().detach().item())
    band_ratio_max = float(band_per_image.max().detach().item())
    max_band_ratio = float(getattr(cfg, "HR_BFR_MAX_BAND_RATIO", 0.35))
    valid_img = student_out.get("hr_valid_img_mask", None)
    if valid_img is None:
        valid_img = (band_per_image <= max_band_ratio).to(dtype=band.dtype, device=band.device).view(b, 1, 1, 1)
    else:
        valid_img = valid_img.detach().to(device=band.device, dtype=band.dtype)
    valid_img_ratio = float(valid_img.mean().detach().item())
    skip_img_ratio = 1.0 - valid_img_ratio
    active_pixel_ratio = float(band.detach().mean().item())

    hr_scale = float(output_scalar(student_out, "hr_scale", get_hr_bfr_scale(cfg, epoch)))
    beta_hr = float(output_scalar(student_out, "hr_beta_eff", float(getattr(cfg, "HR_BFR_BETA_MAX", 0.05)) * hr_scale))
    eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    target_mixed_68 = (
        float(static_weight) * target_soft
        + float(teacher_weight) * teacher_binary.detach().float()
    ).clamp(0.0, 1.0)
    target_mixed_136 = F.interpolate(
        target_mixed_68.detach(),
        size=hr_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    weight_136 = F.interpolate(
        weight_map.detach(),
        size=hr_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    zero = hr_logits.sum() * 0.0
    loss_band_bce = zero
    loss_outband_anchor = zero
    loss_bg_prob_lock = zero
    loss_area_neutral = zero
    loss_edge_align = zero

    if bool(getattr(cfg, "HR_BFR_USE_BAND_BCE", True)):
        bce_map = F.binary_cross_entropy_with_logits(hr_logits, target_mixed_136, reduction="none")
        loss_band_bce = _hr_masked_mean(bce_map, band * weight_136 * valid_img, eps=eps)

    hr_prob = torch.sigmoid(hr_logits)
    anchor_prob = torch.sigmoid(anchor_logits)
    out_band = ((1.0 - band) * valid_img).detach()
    if bool(getattr(cfg, "HR_BFR_USE_OUTBAND_ANCHOR", True)):
        smooth_l1 = F.smooth_l1_loss(hr_prob, anchor_prob, reduction="none")
        loss_outband_anchor = _hr_masked_mean(smooth_l1, out_band, eps=eps)

    bg_reliable_ratio = 0.0
    if bool(getattr(cfg, "HR_BFR_USE_BG_PROB_LOCK", True)):
        bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float()
        low_target_bg = (
            (target_soft < float(getattr(cfg, "CSD_V1R_BG_LOW_TARGET_THRESH", 0.15)))
            & (weight_map > float(getattr(cfg, "CSD_V1R_BG_LOW_TARGET_WEIGHT_THRESH", 0.50)))
        ).float()
        bg_reliable = ((bg_core > 0.5).float() + low_target_bg).clamp(0.0, 1.0)
        bg_reliable_136 = F.interpolate(bg_reliable, size=hr_logits.shape[-2:], mode="nearest").detach() * valid_img
        bg_reliable_ratio = float(bg_reliable_136.mean().detach().item())
        delta = float(getattr(cfg, "HR_BFR_BG_PROB_LOCK_DELTA", 0.02))
        bg_over_anchor = F.relu(hr_prob - anchor_prob - delta).pow(2)
        loss_bg_prob_lock = _hr_masked_mean(bg_over_anchor, bg_reliable_136, eps=eps)

    area_delta = (hr_prob.mean(dim=(1, 2, 3)) - anchor_prob.mean(dim=(1, 2, 3))).abs()
    if bool(getattr(cfg, "HR_BFR_USE_AREA_NEUTRAL", True)):
        tol = float(getattr(cfg, "HR_BFR_AREA_TOL", 0.005))
        valid_flat = valid_img.view(-1)
        denom = valid_flat.sum().clamp_min(1.0)
        loss_area_neutral = (F.relu(area_delta - tol).pow(2) * valid_flat).sum() / denom

    edge_support_mean = 0.0
    if bool(getattr(cfg, "HR_BFR_USE_EDGE_ALIGN", True)):
        if "hr_sobel_136" not in student_out:
            raise RuntimeError("HR_BFR_USE_EDGE_ALIGN=True requires hr_sobel_136 in model output.")
        image_edge = student_out["hr_sobel_136"].detach().clamp(0.0, 1.0)
        edge_thr = per_image_quantile_map(image_edge, float(getattr(cfg, "HR_BFR_EDGE_Q", 0.60)))
        edge_support = (image_edge >= edge_thr).float()
        edge_support_mean = float((edge_support * band).sum().detach().item() / (band.sum().detach().item() + 1e-6))
        prob_edge = compute_sobel_mag_1ch(hr_prob)
        loss_edge_align = _hr_masked_mean(prob_edge * (1.0 - edge_support), band * valid_img, eps=eps)

    lambda_band = (
        float(getattr(cfg, "HR_BFR_BAND_BCE_WEIGHT_MAX", 0.05))
        * hr_scale
        if bool(getattr(cfg, "HR_BFR_USE_BAND_BCE", True))
        else 0.0
    )
    lambda_out = (
        float(getattr(cfg, "HR_BFR_OUTBAND_ANCHOR_WEIGHT_MAX", 0.10))
        * hr_scale
        if bool(getattr(cfg, "HR_BFR_USE_OUTBAND_ANCHOR", True))
        else 0.0
    )
    lambda_bg = (
        float(getattr(cfg, "HR_BFR_BG_PROB_LOCK_WEIGHT_MAX", 0.02))
        * hr_scale
        if bool(getattr(cfg, "HR_BFR_USE_BG_PROB_LOCK", True))
        else 0.0
    )
    lambda_area = (
        float(getattr(cfg, "HR_BFR_AREA_NEUTRAL_WEIGHT_MAX", 0.02))
        * hr_scale
        if bool(getattr(cfg, "HR_BFR_USE_AREA_NEUTRAL", True))
        else 0.0
    )
    lambda_edge = (
        float(getattr(cfg, "HR_BFR_EDGE_ALIGN_WEIGHT_MAX", 0.01))
        * hr_scale
        if bool(getattr(cfg, "HR_BFR_USE_EDGE_ALIGN", True))
        else 0.0
    )
    loss_hr = (
        lambda_band * loss_band_bce
        + lambda_out * loss_outband_anchor
        + lambda_bg * loss_bg_prob_lock
        + lambda_area * loss_area_neutral
        + lambda_edge * loss_edge_align
    )
    stats = {
        "hr_scale": hr_scale,
        "hr_beta_eff": beta_hr,
        "band_ratio": band_ratio,
        "band_ratio_max": band_ratio_max,
        "valid_img_ratio": valid_img_ratio,
        "skip_img_ratio": skip_img_ratio,
        "hr_active_pixel_ratio": active_pixel_ratio,
        "anchor_area": float(anchor_prob.detach().mean().item()),
        "hr_area": float(hr_prob.detach().mean().item()),
        "area_delta": float(area_delta.detach().mean().item()),
        "hr_minus_anchor_abs_mean": output_scalar(student_out, "hr_minus_anchor_abs_mean", 0.0),
        "hr_residual_abs_mean": output_scalar(student_out, "hr_residual_abs_mean", 0.0),
        "hr_residual_abs_max": output_scalar(student_out, "hr_residual_abs_max", 0.0),
        "bg_reliable_ratio": bg_reliable_ratio,
        "edge_support_mean": edge_support_mean,
        "target_mixed_mean": float(target_mixed_136.detach().mean().item()),
        "loss_band_bce": float(loss_band_bce.detach().item()),
        "loss_outband_anchor": float(loss_outband_anchor.detach().item()),
        "loss_bg_prob_lock": float(loss_bg_prob_lock.detach().item()),
        "loss_area_neutral": float(loss_area_neutral.detach().item()),
        "loss_edge_align": float(loss_edge_align.detach().item()),
        "lambda_band": lambda_band,
        "lambda_out": lambda_out,
        "lambda_bg": lambda_bg,
        "lambda_area": lambda_area,
        "lambda_edge": lambda_edge,
        "loss_hr_bfr": float(loss_hr.detach().item()),
    }
    return loss_hr, stats


def log_hr_bfr_first_batch(logger, image_136, output, stats):
    logger.log(f"[HR-BFR FirstBatch] USE_HR_BFR = True")
    logger.log(f"[HR-BFR FirstBatch] image_136 shape = {list(image_136.shape)}")
    logger.log(f"[HR-BFR FirstBatch] hr_anchor_logits shape = {list(output['hr_anchor_logits'].shape)}")
    logger.log(f"[HR-BFR FirstBatch] hr_logits shape = {list(output['hr_logits'].shape)}")
    logger.log(f"[HR-BFR FirstBatch] hr_residual_logits shape = {list(output['hr_residual_logits'].shape)}")
    logger.log(f"[HR-BFR FirstBatch] hr_band_gate shape = {list(output['hr_band_gate_136'].shape)}")
    logger.log(
        "[HR-BFR FirstBatch] hr_scale/beta = "
        f"{float(stats.get('hr_scale', 0.0)):.8f}/"
        f"{float(stats.get('hr_beta_eff', 0.0)):.8f}"
    )
    logger.log(
        "[HR-BFR FirstBatch] band ratio mean/max = "
        f"{float(stats.get('band_ratio', 0.0)):.8f}/"
        f"{float(stats.get('band_ratio_max', 0.0)):.8f}"
    )
    logger.log(
        "[HR-BFR FirstBatch] valid/skip image ratio | active_pixel_ratio = "
        f"{float(stats.get('valid_img_ratio', 0.0)):.8f}/"
        f"{float(stats.get('skip_img_ratio', 0.0)):.8f} | "
        f"{float(stats.get('hr_active_pixel_ratio', 0.0)):.8f}"
    )
    logger.log(
        "[HR-BFR FirstBatch] hr_minus_anchor/residual_abs/bg_reliable = "
        f"{float(stats.get('hr_minus_anchor_abs_mean', 0.0)):.8f}/"
        f"{float(stats.get('hr_residual_abs_mean', 0.0)):.8f}/"
        f"{float(stats.get('bg_reliable_ratio', 0.0)):.8f}"
    )
    logger.log(
        "[HR-BFR FirstBatch] loss band/out_anchor/bg_lock/area/edge/total = "
        f"{float(stats.get('loss_band_bce', 0.0)):.8f}/"
        f"{float(stats.get('loss_outband_anchor', 0.0)):.8f}/"
        f"{float(stats.get('loss_bg_prob_lock', 0.0)):.8f}/"
        f"{float(stats.get('loss_area_neutral', 0.0)):.8f}/"
        f"{float(stats.get('loss_edge_align', 0.0)):.8f}/"
        f"{float(stats.get('loss_hr_bfr', 0.0)):.8f}"
    )


def build_ndr_v2_bg_lock_mask(batch, cfg, target_shape, device):
    bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float()
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    if bg_core.shape[-2:] != target_shape:
        bg_core = F.interpolate(bg_core, size=target_shape, mode="nearest")
    if target_soft.shape[-2:] != target_shape:
        target_soft = F.interpolate(target_soft, size=target_shape, mode="bilinear", align_corners=False)
    if weight_map.shape[-2:] != target_shape:
        weight_map = F.interpolate(weight_map, size=target_shape, mode="bilinear", align_corners=False)

    use_bg_core = bool(getattr(cfg, "NDR_V2_BG_LOCK_USE_BG_CORE", True))
    use_low_target = bool(
        getattr(
            cfg,
            "NDR_V2_BG_LOCK_USE_LOW_TARGET_BG",
            getattr(cfg, "NDR_V2_BG_LOCK_USE_LOW_TARGET", True),
        )
    )
    bg_core_mask = (
        bg_core > float(getattr(cfg, "NDR_V2_BG_CORE_THRESH", 0.5))
        if use_bg_core
        else torch.zeros_like(bg_core, dtype=torch.bool)
    )
    low_target_thresh = float(
        getattr(cfg, "NDR_V2_LOW_TARGET_THRESH", getattr(cfg, "NDR_V2_BG_LOCK_TARGET_THRESH", 0.15))
    )
    low_target_weight_thresh = float(
        getattr(cfg, "NDR_V2_LOW_TARGET_WEIGHT_THRESH", getattr(cfg, "NDR_V2_BG_LOCK_WEIGHT_THRESH", 0.50))
    )
    low_target_bg = (
        (target_soft < low_target_thresh)
        & (weight_map > low_target_weight_thresh)
        if use_low_target
        else torch.zeros_like(bg_core_mask)
    )
    bg_lock = bg_core_mask | low_target_bg
    if bool(getattr(cfg, "NDR_V2_BG_LOCK_DETACH_MASK", True)):
        bg_lock = bg_lock.detach()
    stats = {
        "bg_lock_area": float(bg_lock.float().detach().mean().item()),
        "bg_core_area": float(bg_core_mask.float().detach().mean().item()),
        "low_target_bg_area": float(low_target_bg.float().detach().mean().item()),
    }
    return bg_lock, stats


def masked_bce_with_logits(logits, target, mask, eps=1e-6):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask = mask.float()
    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    return (loss * mask).sum() / (denom + float(eps))


def build_ndr_v2_bg_lock_loss(student_out, batch, cfg):
    if not isinstance(student_out, dict):
        raise RuntimeError("USE_NDR_V2=True requires dict output from NDR branch.")
    if "ndr_delta_logits_68" in student_out:
        delta_logits = student_out["ndr_delta_logits_68"]
    elif "detail_gate" in student_out and "residual_logits_68" in student_out:
        beta_eff = output_scalar(student_out, "ndr_beta_eff", 0.0)
        delta_logits = student_out["detail_gate"] * student_out["residual_logits_68"] * float(beta_eff)
    else:
        raise RuntimeError("USE_NDR_V2=True requires ndr_delta_logits_68 or detail_gate/residual_logits_68.")
    if "pu_bg_core" not in batch or "pu_target_soft" not in batch or "pu_weight_map" not in batch:
        raise RuntimeError("NDR-v2 bg lock requires pu_bg_core, pu_target_soft, and pu_weight_map in batch.")

    device = delta_logits.device
    bg_lock, bg_stats = build_ndr_v2_bg_lock_mask(batch, cfg, delta_logits.shape[-2:], device)
    bg_weight = bg_lock.float()
    denom = bg_weight.sum()
    positive_delta = F.relu(delta_logits)
    if float(denom.detach().item()) <= 0.0:
        loss = delta_logits.sum() * 0.0
        pos_bg_mean = 0.0
    else:
        masked_positive_delta = positive_delta * bg_weight
        loss = (masked_positive_delta.square()).sum() / (
            denom + float(getattr(cfg, "NDR_V2_BG_LOCK_EPS", 1e-6))
        )
        pos_bg_mean = float((masked_positive_delta.sum() / denom).detach().item())
    stats = {
        **bg_stats,
        "positive_delta_bg_mean": pos_bg_mean,
        "positive_delta_bg_max": float((positive_delta * bg_weight).detach().max().item()),
        "positive_delta_mean": float(positive_delta.detach().mean().item()),
        "loss_ndr_bg_lock": float(loss.detach().item()),
    }
    return loss, stats


def ndr_v2_shape_lower_bound_loss(logits, shape_candidate, floor):
    mask = shape_candidate.to(device=logits.device).float()
    if int(mask.sum().detach().item()) < 1:
        return logits.sum() * 0.0
    prob = torch.sigmoid(logits)
    loss_map = F.relu(float(floor) - prob).pow(2)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-6)


def build_ndr_v2_shape_lower_bound(
    cfg,
    batch,
    student_logits,
    student_out,
    teacher_binary,
    epoch,
    device,
):
    scale = float(get_ndr_v2_shape_lb_scale(cfg, epoch))
    lambda_eff = float(getattr(cfg, "NDR_V2_SHAPE_LB_WEIGHT_MAX", 0.0)) * scale
    zero_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    stats = {
        "shape_lb_scale": scale,
        "lambda_shape_lb_eff": lambda_eff,
        "shape_candidate_raw_ratio": 0.0,
        "shape_candidate_capped_ratio": 0.0,
        "shape_valid_image_ratio": 0.0,
        "shape_margin_mean": 0.0,
        "shape_edge_mean": 0.0,
        "shape_under_floor_mean": 0.0,
        "shape_lb_skipped_no_fg_proto": 0,
        "shape_lb_skipped_no_bg_proto": 0,
    }
    if scale <= 0.0:
        return zero_mask, stats
    if not isinstance(student_out, dict) or "coarse_logits_68" not in student_out:
        raise RuntimeError("NDR-v2 shape lower-bound requires coarse_logits_68 in student output.")

    final_prob = torch.sigmoid(student_logits.detach())
    coarse_prob = torch.sigmoid(student_out["coarse_logits_68"].detach())
    if coarse_prob.shape[-2:] != student_logits.shape[-2:]:
        coarse_prob = F.interpolate(coarse_prob, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
    teacher_bg = (teacher_binary.detach() < 0.5)
    masks = build_rast_region_masks(batch, device)
    fg_core = masks["fg_core"]
    bg_core = masks["bg_core"]
    extent = masks["extent"]
    unknown = masks["unknown"]

    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_EDGE", True)):
        if isinstance(student_out, dict) and "edge_norm_68" in student_out:
            edge_norm = student_out["edge_norm_68"].detach().to(device=device, dtype=student_logits.dtype)
        elif isinstance(student_out, dict) and "sobel_68" in student_out:
            edge_norm = student_out["sobel_68"].detach().to(device=device, dtype=student_logits.dtype)
        else:
            edge_norm = compute_sobel_mag_68(batch["image_68"].to(device=device, dtype=student_logits.dtype))
        q = float(getattr(cfg, "NDR_V2_SHAPE_LB_EDGE_Q", 0.60))
        edge_thresh = torch.quantile(edge_norm.flatten(1), q, dim=1).view(-1, 1, 1, 1)
        edge_support = edge_norm >= edge_thresh
    else:
        edge_norm = torch.zeros_like(student_logits)
        edge_support = torch.ones_like(student_logits, dtype=torch.bool)

    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_DINO_MARGIN", True)):
        margin_68, margin_stats = compute_dino_core_margin_68(
            cfg,
            batch,
            fg_core,
            bg_core,
            student_logits.shape[-2:],
            device,
            prefix="NDR_V2_SHAPE_LB",
        )
        margin_68 = margin_68.detach()
        stats["shape_lb_skipped_no_fg_proto"] = int(margin_stats.get("ndr_v2_shape_lb_skipped_no_fg_proto", 0))
        stats["shape_lb_skipped_no_bg_proto"] = int(margin_stats.get("ndr_v2_shape_lb_skipped_no_bg_proto", 0))
    else:
        margin_68 = torch.zeros_like(student_logits)

    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_NEAR_FG", True)):
        near_prob_thresh = float(getattr(cfg, "NDR_V2_SHAPE_LB_NEAR_PROB_THRESH", 0.45))
        near_seed = (coarse_prob >= near_prob_thresh) | (final_prob >= near_prob_thresh)
        radius = int(getattr(cfg, "NDR_V2_SHAPE_LB_NEAR_RADIUS", 3))
        near_fg = F.max_pool2d(
            near_seed.float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ) > 0.5
    else:
        near_fg = torch.ones_like(student_logits, dtype=torch.bool)

    if isinstance(student_out, dict) and "boundary_band_68" in student_out:
        boundary_band = student_out["boundary_band_68"].detach().to(device=device, dtype=student_logits.dtype)
    elif isinstance(student_out, dict) and "boundary_band" in student_out:
        boundary_band = student_out["boundary_band"].detach().to(device=device, dtype=student_logits.dtype)
    else:
        radius = int(getattr(cfg, "NDR_V2_BOUNDARY_RADIUS", 2))
        k = 2 * radius + 1
        boundary_band = F.max_pool2d(coarse_prob, kernel_size=k, stride=1, padding=radius)
        boundary_band = boundary_band + F.max_pool2d(-coarse_prob, kernel_size=k, stride=1, padding=radius)
        boundary_band = boundary_band.clamp(0.0, 1.0)
    boundary_candidate = boundary_band >= float(getattr(cfg, "NDR_V2_SHAPE_LB_BOUNDARY_THRESH", 0.05))

    region_candidate = torch.zeros_like(student_logits, dtype=torch.bool)
    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_EXTENT", True)):
        region_candidate = region_candidate | extent
    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_UNKNOWN", False)):
        region_candidate = region_candidate | unknown
    non_bg = ~bg_core if bool(getattr(cfg, "NDR_V2_SHAPE_LB_EXCLUDE_BG_CORE", True)) else torch.ones_like(bg_core)
    non_unknown = ~unknown if not bool(getattr(cfg, "NDR_V2_SHAPE_LB_USE_UNKNOWN", False)) else torch.ones_like(unknown)
    shape_candidate_raw = (
        boundary_candidate
        & region_candidate
        & edge_support
        & (margin_68 >= float(getattr(cfg, "NDR_V2_SHAPE_LB_MARGIN_THRESH", 0.00)))
        & near_fg
        & non_bg
        & non_unknown
    )
    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_REQUIRE_TEACHER_BG", True)):
        shape_candidate_raw = shape_candidate_raw & teacher_bg

    under_floor = F.relu(float(getattr(cfg, "NDR_V2_SHAPE_LB_FLOOR", 0.35)) - final_prob)
    margin_pos = F.relu(margin_68)
    shape_score = 0.4 * boundary_band + 0.3 * edge_norm + 0.2 * margin_pos + 0.3 * under_floor
    shape_candidate = cap_mask_by_score_per_image(
        shape_candidate_raw,
        shape_score,
        float(getattr(cfg, "NDR_V2_SHAPE_LB_MAX_RATIO_PER_IMAGE", 0.005)),
        int(getattr(cfg, "NDR_V2_SHAPE_LB_MIN_PIXELS_PER_IMAGE", 4)),
    )
    if bool(getattr(cfg, "NDR_V2_SHAPE_LB_DETACH_MASK", True)):
        shape_candidate = shape_candidate.detach()
        shape_candidate_raw = shape_candidate_raw.detach()

    stats.update(
        {
            "shape_candidate_raw_ratio": float(shape_candidate_raw.float().detach().mean().item()),
            "shape_candidate_capped_ratio": float(shape_candidate.float().detach().mean().item()),
            "shape_valid_image_ratio": float(
                (shape_candidate.float().flatten(1).sum(dim=1) > 0).float().detach().mean().item()
            ),
            "shape_margin_mean": _mean_on_mask(margin_68, shape_candidate),
            "shape_edge_mean": _mean_on_mask(edge_norm, shape_candidate),
            "shape_under_floor_mean": _mean_on_mask(under_floor, shape_candidate),
        }
    )
    return shape_candidate, stats


def build_ndr_v2_bg_lock_losses(final_logits, student_out, batch, cfg, epoch):
    loss_res, base_stats = build_ndr_v2_bg_lock_loss(student_out, batch, cfg)
    bg_res_scale = float(get_ndr_v2_bg_res_lock_scale(cfg, epoch))
    bg_prob_scale = float(get_ndr_v2_bg_prob_lock_scale(cfg, epoch))
    lambda_res = float(
        getattr(cfg, "NDR_V2_BG_RES_LOCK_WEIGHT_MAX", getattr(cfg, "NDR_V2_BG_LOCK_WEIGHT", 0.005))
    ) * bg_res_scale
    lambda_prob = float(getattr(cfg, "NDR_V2_BG_PROB_LOCK_WEIGHT_MAX", 0.0)) * bg_prob_scale

    device = final_logits.device
    bg_lock, _ = build_ndr_v2_bg_lock_mask(batch, cfg, final_logits.shape[-2:], device)
    bg_weight = bg_lock.float()
    denom = bg_weight.sum()
    if (
        "coarse_logits_68" in student_out
        and student_out["coarse_logits_68"].shape[-2:] == final_logits.shape[-2:]
    ):
        p_anchor = torch.sigmoid(student_out["coarse_logits_68"].detach())
    elif "coarse_logits_68" in student_out:
        p_anchor = torch.sigmoid(
            F.interpolate(
                student_out["coarse_logits_68"].detach(),
                size=final_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
    else:
        p_anchor = torch.zeros_like(final_logits)
    p_final = torch.sigmoid(final_logits)
    if float(denom.detach().item()) <= 0.0 or not bool(getattr(cfg, "NDR_V2_USE_BG_PROB_LOCK", False)):
        loss_prob = final_logits.sum() * 0.0
    else:
        delta = float(getattr(cfg, "NDR_V2_BG_PROB_LOCK_DELTA", 0.02))
        loss_prob_map = F.relu(p_final - p_anchor - delta).pow(2)
        loss_prob = (loss_prob_map * bg_weight).sum() / (
            denom + float(getattr(cfg, "NDR_V2_BG_LOCK_EPS", 1e-6))
        )
    stats = {
        **base_stats,
        "bg_res_lock_scale": bg_res_scale,
        "lambda_bg_res_lock_eff": lambda_res,
        "bg_prob_lock_scale": bg_prob_scale,
        "lambda_bg_prob_lock_eff": lambda_prob,
        "loss_bg_res_lock": float(loss_res.detach().item()),
        "loss_bg_prob_lock": float(loss_prob.detach().item()),
    }
    return loss_res, loss_prob, stats


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
    return build_teacher_region_masks(batch, device)


def _rast_mask_mean(value, mask, empty_value=0.0):
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom.detach().item()) <= 0.0:
        return float(empty_value)
    return float((value * mask_f).sum().detach().item() / (denom.detach().item() + 1e-6))


def compute_dino_core_margin_68(cfg, batch, fg_core, bg_core, output_size, device, prefix="ESA"):
    return compute_dino_core_margin_68_shared(
        cfg,
        batch,
        fg_core,
        bg_core,
        output_size,
        device,
        prefix=prefix,
    )


def get_esa_ber_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_ESA_BER", False)):
        return 0.0
    start = int(getattr(cfg, "ESA_BER_START_EPOCH", 21))
    ramp_end = int(getattr(cfg, "ESA_BER_RAMP_END_EPOCH", 23))
    stop = int(getattr(cfg, "ESA_BER_STOP_EPOCH", 36))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if ramp_end < start:
        raise RuntimeError(
            f"ESA_BER_RAMP_END_EPOCH must be >= start epoch, got {ramp_end} < {start}."
        )
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(ramp_end - start + 1)
    return 1.0


def validate_esa_ber_config(cfg):
    if not bool(getattr(cfg, "USE_ESA_BER", False)):
        return
    required_true = (
        "USE_DAGP_SAFE_HEAD",
        "USE_NDR_BRANCH",
        "USE_DABE_PU",
        "USE_RAST",
        "USE_ESA_ASYM",
        "ESA_BER_AFTER_RESET_ONLY",
        "ESA_BER_APPLY_TO_FINAL_ONLY",
        "ESA_BER_DETACH_CANDIDATES",
        "ESA_BER_DETACH_TEACHER",
        "ESA_BER_DETACH_GRAPH_EVIDENCE",
        "ESA_BER_DETACH_MARGIN",
        "DAGP_SAFE_RETURN_GRAPH_AUX_FOR_BER",
    )
    missing_true = [name for name in required_true if not bool(getattr(cfg, name, False))]
    if missing_true:
        raise RuntimeError(f"ESA-v2-BER requires enabled flags: {missing_true}.")
    if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "dagp_safe":
        raise RuntimeError("ESA-v2-BER requires HEAD_TYPE='dagp_safe'.")
    if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
        raise RuntimeError("ESA-v2-BER requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
    if get_dabe_pu_despl_teacher_target_mode(cfg) != "binary":
        raise RuntimeError("ESA-v2-BER requires the unchanged binary EMA teacher target.")
    if get_dabe_pu_despl_static_target_mode(cfg) != "soft":
        raise RuntimeError("ESA-v2-BER requires the original soft DABE-PU static target.")
    if bool(getattr(cfg, "ESA_BER_APPLY_TO_COARSE", False)) or bool(
        getattr(cfg, "ESA_BER_APPLY_TO_BASE", False)
    ):
        raise RuntimeError("ESA-v2-BER ranking must apply to final logits only.")
    if bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)) or bool(
        getattr(cfg, "RAST_POST_RESET_ENABLE", False)
    ):
        raise RuntimeError("ESA-v2-BER forbids post-reset teacher-map routing.")
    forbidden = (
        "USE_NDR_V2",
        "USE_CSD_V1R",
        "USE_PA_DAGP",
        "USE_LCEG",
        "USE_TCE",
        "USE_HR_BFR",
        "USE_CSSD",
        "USE_HBNS_LITE",
        "USE_EPR_POS",
        "USE_PROTO_CONTRAST",
        "USE_MULTI_VIEW_FEATURE",
        "USE_TADR_ROUTER",
    )
    enabled_forbidden = [name for name in forbidden if bool(getattr(cfg, name, False))]
    if enabled_forbidden:
        raise RuntimeError(f"ESA-v2-BER incompatible branches are enabled: {enabled_forbidden}.")
    if str(getattr(cfg, "GKD_MODE", "off")).lower() != "off" or bool(
        getattr(cfg, "USE_GKD_LITE", False)
    ):
        raise RuntimeError("ESA-v2-BER requires GKD disabled.")
    start = int(getattr(cfg, "ESA_BER_START_EPOCH", 21))
    stop = int(getattr(cfg, "ESA_BER_STOP_EPOCH", 36))
    teacher_only_start = int(getattr(cfg, "DABE_PU_DESPL_TEACHER_ONLY_START", 21))
    if start != teacher_only_start:
        raise RuntimeError("ESA_BER_START_EPOCH must equal the teacher-only start epoch.")
    if stop != int(getattr(cfg, "MAX_EPOCH", 35)) + 1:
        raise RuntimeError("ESA_BER_STOP_EPOCH must equal MAX_EPOCH + 1.")
    if int(getattr(cfg, "RAST_STOP_EPOCH", 21)) != start or int(
        getattr(cfg, "ESA_ASYM_STOP_EPOCH", 21)
    ) != start:
        raise RuntimeError("ESA-v2-BER requires pre-reset RAST/ESA to stop at BER start.")
    if str(getattr(cfg, "ESA_BER_GRAPH_WEIGHT_SOURCE", "semantic_topk")) != "semantic_topk":
        raise RuntimeError("ESA-v2-BER requires semantic_topk graph evidence.")
    if float(getattr(cfg, "ESA_BER_MAX_RATIO_PER_CLASS", 0.005)) <= 0.0:
        raise RuntimeError("ESA_BER_MAX_RATIO_PER_CLASS must be positive.")
    if int(getattr(cfg, "ESA_BER_MIN_PAIRS_PER_IMAGE", 4)) <= 0:
        raise RuntimeError("ESA_BER_MIN_PAIRS_PER_IMAGE must be positive.")
    if int(getattr(cfg, "ESA_BER_MAX_PAIRS_PER_IMAGE", 24)) < int(
        getattr(cfg, "ESA_BER_MIN_PAIRS_PER_IMAGE", 4)
    ):
        raise RuntimeError("ESA_BER_MAX_PAIRS_PER_IMAGE must be >= minimum pairs.")


def _ber_gather_scalar_neighbors(values, topk_idx):
    if values.ndim != 2 or topk_idx.ndim != 3:
        raise RuntimeError(
            "ESA-BER scalar gather expects values [B,N] and indices [B,N,K], got "
            f"{list(values.shape)} and {list(topk_idx.shape)}."
        )
    batch_size, num_nodes = values.shape
    if topk_idx.shape[:2] != (batch_size, num_nodes):
        raise RuntimeError(
            "ESA-BER graph index shape does not match core labels: "
            f"values={list(values.shape)}, indices={list(topk_idx.shape)}."
        )
    batch_offset = (
        torch.arange(batch_size, device=topk_idx.device, dtype=topk_idx.dtype).view(-1, 1, 1)
        * num_nodes
    )
    flat_idx = (topk_idx + batch_offset).reshape(-1)
    return values.reshape(-1).index_select(0, flat_idx).reshape_as(topk_idx)


def compute_ber_graph_connectivity(
    topk_idx,
    topk_sem_weight,
    fg_core_native,
    bg_core_native,
    output_size=(68, 68),
):
    topk_idx = topk_idx.detach()
    topk_sem_weight = topk_sem_weight.detach().float()
    fg_core_native = fg_core_native.detach().bool()
    bg_core_native = bg_core_native.detach().bool() & (~fg_core_native)
    if topk_idx.requires_grad or topk_sem_weight.requires_grad:
        raise RuntimeError("ESA-BER graph evidence must be detached.")
    if topk_idx.shape != topk_sem_weight.shape or topk_idx.ndim != 3:
        raise RuntimeError(
            "ESA-BER top-k auxiliary must have matching [B,N,K] shapes, got "
            f"idx={list(topk_idx.shape)}, weight={list(topk_sem_weight.shape)}."
        )
    if fg_core_native.ndim != 4 or bg_core_native.shape != fg_core_native.shape:
        raise RuntimeError(
            "ESA-BER native core masks must be matching [B,1,H,W] tensors, got "
            f"fg={list(fg_core_native.shape)}, bg={list(bg_core_native.shape)}."
        )
    batch_size, _, height, width = fg_core_native.shape
    num_nodes = height * width
    if topk_idx.shape[:2] != (batch_size, num_nodes):
        raise RuntimeError(
            "ESA-BER top-k node count does not match native core mask: "
            f"idx={list(topk_idx.shape)}, core={list(fg_core_native.shape)}."
        )
    if int(topk_idx.min().item()) < 0 or int(topk_idx.max().item()) >= num_nodes:
        raise RuntimeError("ESA-BER top-k indices are out of range.")
    if not bool(torch.isfinite(topk_sem_weight).all().item()):
        raise RuntimeError("ESA-BER top-k semantic weights contain NaN/Inf.")
    weight_sum_error = (topk_sem_weight.sum(dim=-1) - 1.0).abs().max()
    if float(weight_sum_error.item()) > 1e-5:
        raise RuntimeError(
            "ESA-BER top-k semantic weights are not normalized: "
            f"max_error={float(weight_sum_error.item()):.8g}."
        )

    fg_flat = fg_core_native.flatten(2).squeeze(1)
    bg_flat = bg_core_native.flatten(2).squeeze(1)
    fg_neighbor = _ber_gather_scalar_neighbors(fg_flat, topk_idx).float()
    bg_neighbor = _ber_gather_scalar_neighbors(bg_flat, topk_idx).float()
    fg_conn = (topk_sem_weight * fg_neighbor).sum(dim=-1)
    bg_conn = (topk_sem_weight * bg_neighbor).sum(dim=-1)
    conn_support = fg_conn + bg_conn
    conn_delta = (fg_conn - bg_conn) / (conn_support + 1e-6)
    conn_delta_native = conn_delta.view(batch_size, 1, height, width)
    conn_support_native = conn_support.view(batch_size, 1, height, width)
    conn_delta_out = F.interpolate(
        conn_delta_native,
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    ).detach()
    conn_support_out = F.interpolate(
        conn_support_native,
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    ).detach()
    graph_valid = (
        (fg_conn.max(dim=1).values > 0.0)
        & (bg_conn.max(dim=1).values > 0.0)
    ).detach()
    if not bool(torch.isfinite(conn_delta_out).all().item()) or not bool(
        torch.isfinite(conn_support_out).all().item()
    ):
        raise RuntimeError("ESA-BER graph connectivity contains NaN/Inf.")
    if float(conn_delta_out.min().item()) < -1.0 - 1e-5 or float(
        conn_delta_out.max().item()
    ) > 1.0 + 1e-5:
        raise RuntimeError("ESA-BER normalized graph connectivity is outside [-1,1].")
    return conn_delta_out, conn_support_out, {
        "conn_delta_native": conn_delta_native.detach(),
        "conn_support_native": conn_support_native.detach(),
        "graph_valid": graph_valid,
        "topk_sem_weight_sum_error": float(weight_sum_error.item()),
    }


def ber_positive_strength(value, threshold):
    denominator = max(1.0 - float(threshold), 1e-6)
    return ((value - float(threshold)) / denominator).clamp(0.0, 1.0)


def select_balanced_ber_candidates(
    pos_raw,
    pos_score,
    neg_raw,
    neg_score,
    min_pairs,
    max_pairs,
    max_ratio,
):
    pos_raw = pos_raw.detach().bool()
    neg_raw = neg_raw.detach().bool()
    pos_score = pos_score.detach()
    neg_score = neg_score.detach()
    if pos_raw.shape != neg_raw.shape or pos_raw.shape != pos_score.shape or pos_raw.shape != neg_score.shape:
        raise RuntimeError("ESA-BER candidate masks and scores must have identical shapes.")
    if pos_raw.ndim != 4 or pos_raw.shape[1] != 1:
        raise RuntimeError(f"ESA-BER candidates must be [B,1,H,W], got {list(pos_raw.shape)}.")
    if bool((pos_raw & neg_raw).any().item()):
        raise RuntimeError("ESA-BER positive and negative raw candidates overlap.")
    if pos_raw.requires_grad or neg_raw.requires_grad or pos_score.requires_grad or neg_score.requires_grad:
        raise RuntimeError("ESA-BER candidate selection inputs must be detached.")

    batch_size, _, height, width = pos_raw.shape
    max_by_ratio = int(math.floor(height * width * float(max_ratio)))
    k_cap = min(int(max_pairs), max_by_ratio)
    if k_cap < int(min_pairs):
        raise RuntimeError(
            "ESA-BER pair cap is smaller than ESA_BER_MIN_PAIRS_PER_IMAGE: "
            f"cap={k_cap}, min={int(min_pairs)}."
        )
    selected_pos = torch.zeros_like(pos_raw)
    selected_neg = torch.zeros_like(neg_raw)
    selected_counts = torch.zeros(batch_size, device=pos_raw.device, dtype=torch.long)
    for image_idx in range(batch_size):
        pos_idx = torch.nonzero(pos_raw[image_idx].flatten(), as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(neg_raw[image_idx].flatten(), as_tuple=False).squeeze(1)
        pair_count = min(int(pos_idx.numel()), int(neg_idx.numel()), int(k_cap))
        if pair_count < int(min_pairs):
            continue
        pos_values = pos_score[image_idx].flatten().index_select(0, pos_idx)
        neg_values = neg_score[image_idx].flatten().index_select(0, neg_idx)
        pos_keep = pos_idx.index_select(0, torch.topk(pos_values, pair_count, largest=True).indices)
        neg_keep = neg_idx.index_select(0, torch.topk(neg_values, pair_count, largest=True).indices)
        selected_pos[image_idx].view(-1).index_fill_(0, pos_keep, True)
        selected_neg[image_idx].view(-1).index_fill_(0, neg_keep, True)
        selected_counts[image_idx] = pair_count

    pos_counts = selected_pos.flatten(1).sum(dim=1)
    neg_counts = selected_neg.flatten(1).sum(dim=1)
    if not bool(torch.equal(pos_counts, neg_counts)):
        raise RuntimeError(
            "ESA-BER selected positive/negative counts are not balanced: "
            f"pos={pos_counts.tolist()}, neg={neg_counts.tolist()}."
        )
    if not bool(torch.equal(pos_counts.to(selected_counts.dtype), selected_counts)):
        raise RuntimeError("ESA-BER selected count bookkeeping mismatch.")
    tolerance = 1e-6
    selected_ratio = selected_pos.float().flatten(1).mean(dim=1)
    if bool((selected_ratio > float(max_ratio) + tolerance).any().item()):
        raise RuntimeError(
            "ESA-BER selected candidate ratio exceeds ESA_BER_MAX_RATIO_PER_CLASS: "
            f"max={float(selected_ratio.max().item()):.8f}, cap={float(max_ratio):.8f}."
        )
    return selected_pos.detach(), selected_neg.detach(), selected_counts.detach()


def _ber_masked_mean_tensor(value, mask):
    mask = mask.bool()
    if not bool(mask.any().item()):
        return value.detach().new_tensor(0.0)
    return value.detach()[mask].float().mean()


def build_esa_ber_loss(
    cfg,
    epoch,
    student_out,
    final_logits,
    batch,
    teacher_prob,
    teacher_binary,
    teacher_map_eff,
    rast_stats,
    device,
):
    zero = final_logits.sum() * 0.0
    ber_scale = float(get_esa_ber_scale(cfg, epoch))
    lambda_eff = float(getattr(cfg, "ESA_BER_LAMBDA_MAX", 0.005)) * ber_scale
    empty_stats = {
        "ber_scale": ber_scale,
        "lambda_ber_eff": lambda_eff,
        "loss_ber_raw": 0.0,
        "loss_ber_weighted": 0.0,
        "proto_valid_ratio": 0.0,
        "graph_valid_ratio": 0.0,
        "valid_image_ratio": 0.0,
        "pos_raw_ratio": 0.0,
        "neg_extent_raw_ratio": 0.0,
        "neg_hard_bg_raw_ratio": 0.0,
        "neg_raw_ratio": 0.0,
        "selected_pos_ratio": 0.0,
        "selected_neg_ratio": 0.0,
        "selected_pairs_mean": 0.0,
        "selected_pairs_min": 0.0,
        "selected_pairs_max": 0.0,
        "pos_margin_mean": 0.0,
        "pos_conn_mean": 0.0,
        "pos_student_prob_mean": 0.0,
        "pos_teacher_prob_mean": 0.0,
        "neg_margin_mean": 0.0,
        "neg_conn_mean": 0.0,
        "neg_student_prob_mean": 0.0,
        "neg_teacher_prob_mean": 0.0,
        "pos_logit_mean": 0.0,
        "neg_logit_mean": 0.0,
        "logit_gap": 0.0,
        "rank_violation_ratio": 0.0,
        "topk_sem_weight_sum_error": 0.0,
        "student_prob_fg_core": 0.0,
        "student_prob_bg_core": 0.0,
        "student_prob_extent": 0.0,
        "teacher_fg_extent": 0.0,
        "source_stats": {},
        "per_image": [],
    }
    if ber_scale <= 0.0:
        return zero, empty_stats, {}

    if not bool(getattr(cfg, "ESA_BER_APPLY_TO_FINAL_ONLY", True)):
        raise RuntimeError("ESA-v2-BER requires ESA_BER_APPLY_TO_FINAL_ONLY=True.")
    if bool(getattr(cfg, "ESA_BER_APPLY_TO_COARSE", False)) or bool(
        getattr(cfg, "ESA_BER_APPLY_TO_BASE", False)
    ):
        raise RuntimeError("ESA-v2-BER must not apply ranking loss to coarse/base logits.")
    if str(getattr(cfg, "ESA_BER_GRAPH_WEIGHT_SOURCE", "semantic_topk")) != "semantic_topk":
        raise RuntimeError("ESA-v2-BER requires semantic_topk graph weights.")
    if not isinstance(student_out, dict):
        raise RuntimeError("ESA-v2-BER requires DAGP-Safe auxiliary dict output.")
    required_aux = {"dagp_topk_idx", "dagp_topk_sem_weight"}
    missing_aux = sorted(required_aux - set(student_out))
    if missing_aux:
        raise RuntimeError(f"ESA-v2-BER missing DAGP semantic graph auxiliary: {missing_aux}.")
    if teacher_prob.requires_grad or teacher_binary.requires_grad:
        raise RuntimeError("ESA-v2-BER teacher probability and binary target must be detached.")

    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
    if bool(getattr(cfg, "ESA_BER_AFTER_RESET_ONLY", True)):
        if abs(float(static_weight)) > 1e-8 or abs(float(teacher_weight) - 1.0) > 1e-8:
            raise RuntimeError(
                "ESA-v2-BER active epochs require teacher-only supervision: "
                f"static={static_weight}, teacher={teacher_weight}."
            )
    if bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)) or bool(
        getattr(cfg, "RAST_POST_RESET_ENABLE", False)
    ):
        raise RuntimeError("ESA-v2-BER forbids post-reset ESA/RAST teacher-map routing.")
    teacher_map_error = float((teacher_map_eff.detach() - 1.0).abs().max().item())
    if teacher_map_error > 1e-5:
        raise RuntimeError(
            "ESA-v2-BER requires an all-one teacher map after reset: "
            f"max_error={teacher_map_error:.8g}."
        )

    region_masks = build_rast_region_masks(batch, device)
    fg_core = region_masks["fg_core"]
    bg_core = region_masks["bg_core"]
    extent = region_masks["extent"]
    unknown = region_masks["unknown"]
    topk_idx = student_out["dagp_topk_idx"].detach()
    topk_sem_weight = student_out["dagp_topk_sem_weight"].detach()
    num_nodes = int(topk_idx.shape[1])
    native_size = int(math.isqrt(num_nodes))
    if native_size * native_size != num_nodes:
        raise RuntimeError(f"ESA-v2-BER expects a square native graph, got N={num_nodes}.")
    fg_core_native = F.interpolate(fg_core.float(), size=(native_size, native_size), mode="nearest") > 0.5
    bg_core_native = F.interpolate(bg_core.float(), size=(native_size, native_size), mode="nearest") > 0.5
    bg_core_native = bg_core_native & (~fg_core_native)
    conn_delta, conn_support, graph_stats = compute_ber_graph_connectivity(
        topk_idx,
        topk_sem_weight,
        fg_core_native,
        bg_core_native,
        output_size=final_logits.shape[-2:],
    )

    margin_68 = rast_stats.get("esa_margin_68_tensor")
    proto_valid = rast_stats.get("esa_proto_valid")
    if margin_68 is None or proto_valid is None:
        margin_68, margin_stats = compute_dino_core_margin_68(
            cfg,
            batch,
            fg_core,
            bg_core,
            final_logits.shape[-2:],
            device,
            prefix="ESA",
        )
        proto_valid = margin_stats["esa_proto_valid"]
    margin_68 = margin_68.detach()
    proto_valid = proto_valid.detach().bool()
    graph_valid = graph_stats["graph_valid"].detach().bool()
    evidence_valid = (proto_valid & graph_valid).view(-1, 1, 1, 1)
    if margin_68.requires_grad or conn_delta.requires_grad or conn_support.requires_grad:
        raise RuntimeError("ESA-v2-BER margin and graph evidence must be detached.")

    student_prob = torch.sigmoid(final_logits.detach())
    teacher_prob = teacher_prob.detach()
    teacher_binary_bool = teacher_binary.detach() >= 0.5
    teacher_bg = ~teacher_binary_bool
    if student_prob.requires_grad or teacher_prob.requires_grad:
        raise RuntimeError("ESA-v2-BER candidate probabilities must be detached.")

    pos_raw = extent & evidence_valid
    if bool(getattr(cfg, "ESA_BER_POS_REQUIRE_TEACHER_BG", True)):
        pos_raw = pos_raw & teacher_bg
    pos_raw = (
        pos_raw
        & (margin_68 >= float(getattr(cfg, "ESA_BER_POS_MARGIN_MIN", 0.05)))
        & (conn_delta >= float(getattr(cfg, "ESA_BER_POS_CONN_MIN", 0.05)))
        & (conn_support >= float(getattr(cfg, "ESA_BER_POS_CONN_SUPPORT_MIN", 0.05)))
        & (student_prob >= float(getattr(cfg, "ESA_BER_POS_STUDENT_PROB_MIN", 0.15)))
    )
    if bool(getattr(cfg, "ESA_BER_POS_EXCLUDE_FG_CORE", True)):
        pos_raw = pos_raw & (~fg_core)
    if bool(getattr(cfg, "ESA_BER_POS_EXCLUDE_BG_CORE", True)):
        pos_raw = pos_raw & (~bg_core)
    if bool(getattr(cfg, "ESA_BER_POS_EXCLUDE_UNKNOWN", True)):
        pos_raw = pos_raw & (~unknown)

    pos_margin_score = ber_positive_strength(
        margin_68, float(getattr(cfg, "ESA_BER_POS_MARGIN_MIN", 0.05))
    )
    pos_conn_score = ber_positive_strength(
        conn_delta, float(getattr(cfg, "ESA_BER_POS_CONN_MIN", 0.05))
    )
    pos_dynamic_score = (
        0.5 * student_prob + 0.5 * F.relu(student_prob - teacher_prob)
    ).clamp(0.0, 1.0)
    pos_score = (
        float(getattr(cfg, "ESA_BER_POS_SCORE_MARGIN_WEIGHT", 0.40)) * pos_margin_score
        + float(getattr(cfg, "ESA_BER_POS_SCORE_CONN_WEIGHT", 0.40)) * pos_conn_score
        + float(getattr(cfg, "ESA_BER_POS_SCORE_DYNAMIC_WEIGHT", 0.20)) * pos_dynamic_score
    ).detach()

    neg_extent_raw = torch.zeros_like(pos_raw)
    neg_extent_score = torch.zeros_like(pos_score)
    if bool(getattr(cfg, "ESA_BER_USE_NEG_EXTENT", True)):
        neg_extent_raw = extent & evidence_valid
        if bool(getattr(cfg, "ESA_BER_NEG_EXTENT_REQUIRE_TEACHER_BG", True)):
            neg_extent_raw = neg_extent_raw & teacher_bg
        neg_extent_raw = (
            neg_extent_raw
            & (margin_68 >= float(getattr(cfg, "ESA_BER_NEG_EXTENT_MARGIN_MIN", -0.05)))
            & (conn_delta <= float(getattr(cfg, "ESA_BER_NEG_EXTENT_CONN_MAX", -0.05)))
            & (
                conn_support
                >= float(getattr(cfg, "ESA_BER_NEG_EXTENT_CONN_SUPPORT_MIN", 0.05))
            )
            & (
                student_prob
                >= float(getattr(cfg, "ESA_BER_NEG_EXTENT_STUDENT_PROB_MIN", 0.15))
            )
            & (~fg_core)
            & (~bg_core)
            & (~unknown)
        )
        targetlike_score = ber_positive_strength(
            margin_68,
            float(getattr(cfg, "ESA_BER_NEG_EXTENT_MARGIN_MIN", -0.05)),
        )
        bg_conn_score = ber_positive_strength(
            -conn_delta,
            abs(float(getattr(cfg, "ESA_BER_NEG_EXTENT_CONN_MAX", -0.05))),
        )
        neg_extent_score = (
            float(getattr(cfg, "ESA_BER_NEG_EXTENT_SCORE_TARGETLIKE_WEIGHT", 0.25))
            * targetlike_score
            + float(getattr(cfg, "ESA_BER_NEG_EXTENT_SCORE_BG_CONN_WEIGHT", 0.45))
            * bg_conn_score
            + float(getattr(cfg, "ESA_BER_NEG_EXTENT_SCORE_HARDNESS_WEIGHT", 0.30))
            * student_prob
        ).detach()

    neg_hard_bg_raw = torch.zeros_like(pos_raw)
    neg_hard_score = torch.zeros_like(pos_score)
    if bool(getattr(cfg, "ESA_BER_USE_NEG_HARD_BG", True)):
        reliable_bg = torch.zeros_like(bg_core)
        if bool(getattr(cfg, "ESA_BER_NEG_HARD_BG_USE_BG_CORE", True)):
            reliable_bg = reliable_bg | bg_core
        if bool(getattr(cfg, "ESA_BER_NEG_HARD_BG_USE_LOW_TARGET_BG", True)):
            low_target_bg = (
                (_batch_pu_tensor(batch, "pu_target_soft", device) < float(getattr(cfg, "ESA_BER_LOW_TARGET_THRESH", 0.15)))
                & (_batch_pu_tensor(batch, "pu_weight_map", device) > float(getattr(cfg, "ESA_BER_LOW_TARGET_WEIGHT_THRESH", 0.50)))
            )
            reliable_bg = reliable_bg | low_target_bg
        foreground_active = student_prob >= float(
            getattr(cfg, "ESA_BER_NEG_HARD_BG_STUDENT_PROB_MIN", 0.30)
        )
        if bool(getattr(cfg, "ESA_BER_NEG_HARD_BG_ALLOW_TEACHER_FG", True)):
            foreground_active = foreground_active | teacher_binary_bool
        neg_hard_bg_raw = (
            reliable_bg
            & evidence_valid
            & foreground_active
            & (~fg_core)
            & (margin_68 <= float(getattr(cfg, "ESA_BER_NEG_HARD_BG_MARGIN_MAX", -0.05)))
            & (conn_delta <= float(getattr(cfg, "ESA_BER_NEG_HARD_BG_CONN_MAX", -0.05)))
            & (
                conn_support
                >= float(getattr(cfg, "ESA_BER_NEG_HARD_BG_CONN_SUPPORT_MIN", 0.05))
            )
        )
        neg_margin_score = ber_positive_strength(
            -margin_68,
            abs(float(getattr(cfg, "ESA_BER_NEG_HARD_BG_MARGIN_MAX", -0.05))),
        )
        neg_conn_score = ber_positive_strength(
            -conn_delta,
            abs(float(getattr(cfg, "ESA_BER_NEG_HARD_BG_CONN_MAX", -0.05))),
        )
        neg_activation_score = torch.maximum(student_prob, teacher_prob)
        neg_hard_score = (
            float(getattr(cfg, "ESA_BER_NEG_HARD_SCORE_MARGIN_WEIGHT", 0.30))
            * neg_margin_score
            + float(getattr(cfg, "ESA_BER_NEG_HARD_SCORE_CONN_WEIGHT", 0.30))
            * neg_conn_score
            + float(getattr(cfg, "ESA_BER_NEG_HARD_SCORE_HARDNESS_WEIGHT", 0.40))
            * neg_activation_score
        ).detach()

    neg_raw = (neg_extent_raw | neg_hard_bg_raw) & (~pos_raw)
    if bool((pos_raw & neg_raw).any().item()):
        raise RuntimeError("ESA-v2-BER positive and negative candidates overlap after exclusion.")
    negative_fill = torch.full_like(neg_extent_score, -1e9)
    neg_score = torch.maximum(
        torch.where(neg_extent_raw, neg_extent_score, negative_fill),
        torch.where(neg_hard_bg_raw, neg_hard_score, negative_fill),
    ).detach()
    pos_raw = pos_raw.detach()
    neg_extent_raw = neg_extent_raw.detach()
    neg_hard_bg_raw = neg_hard_bg_raw.detach()
    neg_raw = neg_raw.detach()
    for name, tensor in (
        ("pos_raw", pos_raw),
        ("neg_extent_raw", neg_extent_raw),
        ("neg_hard_bg_raw", neg_hard_bg_raw),
        ("neg_raw", neg_raw),
        ("pos_score", pos_score),
        ("neg_score", neg_score),
    ):
        if tensor.requires_grad:
            raise RuntimeError(f"ESA-v2-BER detached candidate tensor {name} requires grad.")

    selected_pos, selected_neg, selected_counts = select_balanced_ber_candidates(
        pos_raw,
        pos_score,
        neg_raw,
        neg_score,
        int(getattr(cfg, "ESA_BER_MIN_PAIRS_PER_IMAGE", 4)),
        int(getattr(cfg, "ESA_BER_MAX_PAIRS_PER_IMAGE", 24)),
        float(getattr(cfg, "ESA_BER_MAX_RATIO_PER_CLASS", 0.005)),
    )
    valid_images = selected_counts >= int(getattr(cfg, "ESA_BER_MIN_PAIRS_PER_IMAGE", 4))
    rank_margin = float(getattr(cfg, "ESA_BER_RANK_MARGIN", 0.20))
    rank_tau = float(getattr(cfg, "ESA_BER_RANK_TAU", 0.10))
    if rank_tau <= 0.0:
        raise RuntimeError(f"ESA_BER_RANK_TAU must be positive, got {rank_tau}.")
    image_losses = []
    rank_violation_count = 0
    rank_pair_count = 0
    per_image = []
    dataset_names = [str(name) for name in batch.get("dataset", [""] * final_logits.shape[0])]
    stems = [str(name) for name in batch.get("stem", [""] * final_logits.shape[0])]
    for image_idx in range(final_logits.shape[0]):
        count = int(selected_counts[image_idx].item())
        image_record = {
            "dataset": dataset_names[image_idx],
            "stem": stems[image_idx],
            "valid": count >= int(getattr(cfg, "ESA_BER_MIN_PAIRS_PER_IMAGE", 4)),
            "selected_pairs": count,
            "pos_raw_ratio": float(pos_raw[image_idx].float().mean().item()),
            "neg_extent_raw_ratio": float(neg_extent_raw[image_idx].float().mean().item()),
            "neg_hard_bg_raw_ratio": float(neg_hard_bg_raw[image_idx].float().mean().item()),
            "neg_raw_ratio": float(neg_raw[image_idx].float().mean().item()),
            "logit_gap": 0.0,
            "rank_violation_ratio": 0.0,
            "pos_margin_mean": 0.0,
            "pos_conn_mean": 0.0,
            "neg_margin_mean": 0.0,
            "neg_conn_mean": 0.0,
        }
        if image_record["valid"]:
            z_pos = final_logits[image_idx][selected_pos[image_idx]]
            z_neg = final_logits[image_idx][selected_neg[image_idx]]
            if z_pos.numel() != z_neg.numel() or z_pos.numel() != count:
                raise RuntimeError("ESA-v2-BER selected logit counts are not balanced.")
            pair_diff = z_pos[:, None] - z_neg[None, :]
            loss_image = rank_tau * F.softplus((rank_margin - pair_diff) / rank_tau).mean()
            image_losses.append(loss_image)
            violation = pair_diff.detach() < rank_margin
            rank_violation_count += int(violation.sum().item())
            rank_pair_count += int(violation.numel())
            image_record["logit_gap"] = float((z_pos.detach().mean() - z_neg.detach().mean()).item())
            image_record["rank_violation_ratio"] = float(violation.float().mean().item())
            image_record["pos_margin_mean"] = float(
                margin_68[image_idx][selected_pos[image_idx]].mean().item()
            )
            image_record["pos_conn_mean"] = float(
                conn_delta[image_idx][selected_pos[image_idx]].mean().item()
            )
            image_record["neg_margin_mean"] = float(
                margin_68[image_idx][selected_neg[image_idx]].mean().item()
            )
            image_record["neg_conn_mean"] = float(
                conn_delta[image_idx][selected_neg[image_idx]].mean().item()
            )
        per_image.append(image_record)
    loss_raw = torch.stack(image_losses).mean() if image_losses else zero
    loss_weighted = lambda_eff * loss_raw
    if not bool(torch.isfinite(loss_raw).item()) or not bool(torch.isfinite(loss_weighted).item()):
        raise RuntimeError("ESA-v2-BER ranking loss contains NaN/Inf.")

    selected_counts_float = selected_counts.float()
    selected_pos_ratio = float(selected_pos.float().mean().item())
    selected_neg_ratio = float(selected_neg.float().mean().item())
    source_stats = {}
    for source in sorted(set(dataset_names)):
        records = [record for record in per_image if record["dataset"] == source]
        if not records:
            continue
        source_stats[source] = {
            "images": len(records),
            "valid_image_ratio": sum(float(record["valid"]) for record in records) / len(records),
            "pos_raw_ratio": sum(record["pos_raw_ratio"] for record in records) / len(records),
            "neg_raw_ratio": sum(record["neg_raw_ratio"] for record in records) / len(records),
            "selected_pairs_mean": sum(record["selected_pairs"] for record in records) / len(records),
            "logit_gap": sum(record["logit_gap"] for record in records) / len(records),
            "rank_violation_ratio": sum(record["rank_violation_ratio"] for record in records) / len(records),
        }
    stats = {
        **empty_stats,
        "loss_ber_raw": float(loss_raw.detach().item()),
        "loss_ber_weighted": float(loss_weighted.detach().item()),
        "proto_valid_ratio": float(proto_valid.float().mean().item()),
        "graph_valid_ratio": float(graph_valid.float().mean().item()),
        "valid_image_ratio": float(valid_images.float().mean().item()),
        "pos_raw_ratio": float(pos_raw.float().mean().item()),
        "neg_extent_raw_ratio": float(neg_extent_raw.float().mean().item()),
        "neg_hard_bg_raw_ratio": float(neg_hard_bg_raw.float().mean().item()),
        "neg_raw_ratio": float(neg_raw.float().mean().item()),
        "selected_pos_ratio": selected_pos_ratio,
        "selected_neg_ratio": selected_neg_ratio,
        "selected_pairs_mean": float(selected_counts_float.mean().item()),
        "selected_pairs_min": float(selected_counts_float.min().item()),
        "selected_pairs_max": float(selected_counts_float.max().item()),
        "pos_margin_mean": float(_ber_masked_mean_tensor(margin_68, selected_pos).item()),
        "pos_conn_mean": float(_ber_masked_mean_tensor(conn_delta, selected_pos).item()),
        "pos_student_prob_mean": float(_ber_masked_mean_tensor(student_prob, selected_pos).item()),
        "pos_teacher_prob_mean": float(_ber_masked_mean_tensor(teacher_prob, selected_pos).item()),
        "neg_margin_mean": float(_ber_masked_mean_tensor(margin_68, selected_neg).item()),
        "neg_conn_mean": float(_ber_masked_mean_tensor(conn_delta, selected_neg).item()),
        "neg_student_prob_mean": float(_ber_masked_mean_tensor(student_prob, selected_neg).item()),
        "neg_teacher_prob_mean": float(_ber_masked_mean_tensor(teacher_prob, selected_neg).item()),
        "pos_logit_mean": float(_ber_masked_mean_tensor(final_logits, selected_pos).item()),
        "neg_logit_mean": float(_ber_masked_mean_tensor(final_logits, selected_neg).item()),
        "topk_sem_weight_sum_error": float(graph_stats["topk_sem_weight_sum_error"]),
        "student_prob_fg_core": float(_ber_masked_mean_tensor(student_prob, fg_core).item()),
        "student_prob_bg_core": float(_ber_masked_mean_tensor(student_prob, bg_core).item()),
        "student_prob_extent": float(_ber_masked_mean_tensor(student_prob, extent).item()),
        "teacher_fg_extent": float(_ber_masked_mean_tensor(teacher_binary.float(), extent).item()),
        "source_stats": source_stats,
        "per_image": per_image,
    }
    stats["logit_gap"] = stats["pos_logit_mean"] - stats["neg_logit_mean"]
    stats["rank_violation_ratio"] = (
        float(rank_violation_count) / float(rank_pair_count) if rank_pair_count > 0 else 0.0
    )
    aux = {
        "margin_68": margin_68,
        "conn_delta_68": conn_delta,
        "conn_support_68": conn_support,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "student_prob": student_prob,
        "teacher_prob": teacher_prob,
        "pos_raw": pos_raw,
        "neg_extent_raw": neg_extent_raw,
        "neg_hard_bg_raw": neg_hard_bg_raw,
        "neg_raw": neg_raw,
        "selected_pos": selected_pos,
        "selected_neg": selected_neg,
        "selected_counts": selected_counts,
        "proto_valid": proto_valid,
        "graph_valid": graph_valid,
    }
    return loss_weighted, stats, aux


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
    use_esa_post_reset = bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False))
    esa_margin_68 = torch.zeros_like(teacher_binary, dtype=torch.float32, device=device)
    esa_margin_stats = {
        "esa_skipped_no_fg_proto": 0,
        "esa_skipped_no_bg_proto": 0,
        "esa_proto_valid": torch.zeros(
            teacher_binary.shape[0], device=device, dtype=torch.bool
        ),
    }
    extent_teacher_fg = extent & teacher_fg
    extent_teacher_bg = extent & teacher_bg
    extent_teacher_bg_fg_like = torch.zeros_like(extent_teacher_bg)
    extent_teacher_bg_bg_like = torch.zeros_like(extent_teacher_bg)
    extent_teacher_bg_ambig = torch.zeros_like(extent_teacher_bg)
    if use_esa_asym or use_esa_post_reset:
        if not bool(getattr(cfg, "ESA_USE_DINO_MARGIN", True)):
            raise RuntimeError("ESA asymmetric routing requires ESA_USE_DINO_MARGIN=True.")
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

        if use_esa_asym:
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

    teacher_map_post_esa = torch.ones_like(teacher_binary, dtype=torch.float32, device=device)
    teacher_map_post_esa = torch.where(
        extent_teacher_fg,
        teacher_map_post_esa
        * float(getattr(cfg, "ESA_POST_RESET_EXTENT_TEACHER_FG_MULT", 1.0)),
        teacher_map_post_esa,
    )
    teacher_map_post_esa = torch.where(
        extent_teacher_bg_fg_like,
        teacher_map_post_esa
        * float(getattr(cfg, "ESA_POST_RESET_EXTENT_TEACHER_BG_FG_LIKE_MULT", 0.25)),
        teacher_map_post_esa,
    )
    teacher_map_post_esa = torch.where(
        extent_teacher_bg_ambig,
        teacher_map_post_esa
        * float(getattr(cfg, "ESA_POST_RESET_EXTENT_TEACHER_BG_AMBIG_MULT", 0.50)),
        teacher_map_post_esa,
    )
    teacher_map_post_esa = torch.where(
        extent_teacher_bg_bg_like,
        teacher_map_post_esa
        * float(getattr(cfg, "ESA_POST_RESET_EXTENT_TEACHER_BG_BG_LIKE_MULT", 1.0)),
        teacher_map_post_esa,
    )

    ones = torch.ones_like(teacher_map_region)
    rast_pre_reset_scale = float(get_rast_pre_reset_scale(cfg, epoch))
    rast_post_reset_scale = float(get_rast_post_reset_scale(cfg, epoch))
    esa_post_reset_scale = float(get_esa_post_reset_scale(cfg, epoch))
    esa_post_map_eff = (
        (1.0 - esa_post_reset_scale) * ones
        + esa_post_reset_scale * teacher_map_post_esa
    )
    if rast_pre_reset_scale > 0.0:
        rast_scale_effective = rast_pre_reset_scale
        teacher_routing_scale = rast_pre_reset_scale
        rast_phase = "pre_reset"
        teacher_map_eff = (1.0 - rast_pre_reset_scale) * ones + rast_pre_reset_scale * teacher_map_region
    elif esa_post_reset_scale > 0.0:
        rast_scale_effective = 0.0
        teacher_routing_scale = esa_post_reset_scale
        rast_phase = "esa_post_reset_extent_only"
        teacher_map_eff = esa_post_map_eff
    elif rast_post_reset_scale > 0.0:
        rast_scale_effective = rast_post_reset_scale
        teacher_routing_scale = rast_post_reset_scale
        rast_phase = "post_reset_conflict_only"
        teacher_map_eff = (1.0 - rast_post_reset_scale) * ones + rast_post_reset_scale * teacher_map_post
    else:
        rast_scale_effective = 0.0
        teacher_routing_scale = 0.0
        rast_phase = "off"
        teacher_map_eff = ones

    if esa_post_reset_scale > 0.0:
        if rast_pre_reset_scale > 0.0 or rast_post_reset_scale > 0.0:
            raise RuntimeError("ESA post-reset routing must run with both RAST scales disabled.")
        static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
        if abs(float(static_weight)) > 1e-8 or abs(float(teacher_weight) - 1.0) > 1e-8:
            raise RuntimeError(
                "ESA post-reset routing requires teacher-only schedule: "
                f"static={static_weight}, teacher={teacher_weight}."
            )
        if not bool(torch.isfinite(teacher_map_eff).all().item()):
            raise RuntimeError("ESA post-reset teacher map contains NaN/Inf.")
        map_min = float(teacher_map_eff.min().detach().item())
        map_max = float(teacher_map_eff.max().detach().item())
        if map_min < 0.25 - 1e-5 or map_max > 1.0 + 1e-5:
            raise RuntimeError(
                f"ESA post-reset map out of [0.25,1]: min={map_min}, max={map_max}."
            )
        expected_regions = (
            (fg_core, 1.0, "fg_core"),
            (bg_core, 1.0, "bg_core"),
            (unknown, 1.0, "unknown"),
            (extent_teacher_fg, 1.0, "extent_teacher_fg"),
            (extent_teacher_bg_fg_like, 0.25, "extent_bg_fg_like"),
            (extent_teacher_bg_ambig, 0.50, "extent_bg_ambig"),
            (extent_teacher_bg_bg_like, 1.0, "extent_bg_bg_like"),
        )
        for region_mask, expected, region_name in expected_regions:
            if bool(region_mask.any().item()):
                max_error = float(
                    (teacher_map_eff[region_mask] - expected).abs().max().detach().item()
                )
                if max_error > 1e-5:
                    raise RuntimeError(
                        f"ESA post-reset {region_name} weight mismatch: "
                        f"expected={expected}, max_error={max_error}."
                    )

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
        "teacher_routing_scale": teacher_routing_scale,
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
        "esa_post_reset_enable": use_esa_post_reset,
        "esa_post_reset_active": esa_post_reset_scale > 0.0,
        "esa_post_reset_scale": esa_post_reset_scale,
        "esa_post_map_mean": float(esa_post_map_eff.mean().detach().item()),
        "esa_post_map_min": float(esa_post_map_eff.min().detach().item()),
        "esa_post_map_max": float(esa_post_map_eff.max().detach().item()),
        "esa_post_map_fg_core_mean": _rast_mask_mean(esa_post_map_eff, fg_core, empty_value=1.0),
        "esa_post_map_bg_core_mean": _rast_mask_mean(esa_post_map_eff, bg_core, empty_value=1.0),
        "esa_post_map_unknown_mean": _rast_mask_mean(esa_post_map_eff, unknown, empty_value=1.0),
        "esa_post_map_extent_teacher_fg_mean": _rast_mask_mean(
            esa_post_map_eff, extent_teacher_fg, empty_value=1.0
        ),
        "esa_post_map_extent_bg_fg_like_mean": _rast_mask_mean(
            esa_post_map_eff, extent_teacher_bg_fg_like, empty_value=1.0
        ),
        "esa_post_map_extent_bg_ambig_mean": _rast_mask_mean(
            esa_post_map_eff, extent_teacher_bg_ambig, empty_value=1.0
        ),
        "esa_post_map_extent_bg_bg_like_mean": _rast_mask_mean(
            esa_post_map_eff, extent_teacher_bg_bg_like, empty_value=1.0
        ),
        "esa_post_fg_core_valid": bool(fg_core.any().item()),
        "esa_post_bg_core_valid": bool(bg_core.any().item()),
        "esa_post_unknown_valid": bool(unknown.any().item()),
        "esa_post_extent_teacher_fg_valid": bool(extent_teacher_fg.any().item()),
        "esa_post_extent_bg_fg_like_valid": bool(extent_teacher_bg_fg_like.any().item()),
        "esa_post_extent_bg_ambig_valid": bool(extent_teacher_bg_ambig.any().item()),
        "esa_post_extent_bg_bg_like_valid": bool(extent_teacher_bg_bg_like.any().item()),
        "esa_margin_shape": list(esa_margin_68.shape),
        "esa_margin_68_tensor": esa_margin_68.detach(),
        "esa_proto_valid": esa_margin_stats["esa_proto_valid"].detach(),
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


def rast_teacher_bce_with_logits(
    logits,
    target,
    teacher_map_eff,
    cfg,
    teacher_routing_scale,
    apply_to_loss=True,
    eps=1e-6,
):
    enabled = (
        bool(getattr(cfg, "USE_RAST", False))
        or bool(getattr(cfg, "USE_TEPR_LITE", False))
    ) and float(teacher_routing_scale) > 0.0
    if bool(getattr(cfg, "USE_TEPR_LITE", False)) or bool(
        getattr(cfg, "RAST_WEIGHTED_LOSS_NORMALIZE", True)
    ):
        return teacher_weighted_bce_with_logits(
            logits,
            target,
            teacher_map_eff,
            enabled=enabled,
            apply_to_loss=apply_to_loss,
            eps=eps,
        )
    if not enabled or not bool(apply_to_loss) or teacher_map_eff is None:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * teacher_map_eff.to(device=logits.device, dtype=logits.dtype)).mean()


TEPR_V11_CONDITIONAL_KEYS = (
    "core_conflict_map",
    "core_no_conflict_map",
    "extent_teacher_fg_map",
    "extent_teacher_bg_map",
    "extent_bg_temporal_reliability",
    "extent_bg_dino_ceiling",
    "extent_bg_final_weight",
    "extent_bg_fg_like_weight",
    "extent_bg_ambiguous_weight",
    "extent_bg_bg_like_weight",
    "unknown_map",
    "other_map",
    "student_prob_fg_core",
    "student_prob_extent",
    "teacher_fg_fg_core",
    "teacher_fg_extent",
)


def _tepr_add_masked_conditional(stats, name, value, mask):
    mask_f = mask.float()
    count = float(mask_f.sum().detach().item())
    value_sum = float((value * mask_f).sum().detach().item()) if count > 0.0 else 0.0
    stats.setdefault("conditional_sums", {})[name] = value_sum
    stats.setdefault("conditional_counts", {})[name] = count
    stats.setdefault("conditional_means", {})[name] = value_sum / count if count > 0.0 else 0.0


def add_tepr_prediction_stats(stats, batch, student_logits, teacher_prob, device):
    masks = build_rast_region_masks(batch, device)
    if tuple(student_logits.shape[-2:]) != tuple(teacher_prob.shape[-2:]):
        student_logits = F.interpolate(
            student_logits,
            size=teacher_prob.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    student_prob = torch.sigmoid(student_logits.detach())
    teacher_fg = teacher_prob.detach() >= 0.5
    _tepr_add_masked_conditional(
        stats, "student_prob_fg_core", student_prob, masks["fg_core"]
    )
    _tepr_add_masked_conditional(
        stats, "student_prob_extent", student_prob, masks["extent"]
    )
    _tepr_add_masked_conditional(
        stats, "teacher_fg_fg_core", teacher_fg.float(), masks["fg_core"]
    )
    _tepr_add_masked_conditional(
        stats, "teacher_fg_extent", teacher_fg.float(), masks["extent"]
    )
    stats["prediction_pixel_count"] = int(student_prob.numel())
    stats["student_pred_fg_count"] = int((student_prob >= 0.5).sum().detach().item())
    stats["teacher_pred_fg_count"] = int(teacher_fg.sum().detach().item())


def make_tepr_inactive_stats(batch, teacher_prob, device):
    masks = build_rast_region_masks(batch, device)
    region_counts = {
        name: float(mask.float().sum().detach().item()) for name, mask in masks.items()
    }
    teacher_fg = teacher_prob.detach() >= 0.5
    fg_conflict = masks["fg_core"] & (~teacher_fg)
    bg_conflict = masks["bg_core"] & teacher_fg
    core_conflict = fg_conflict | bg_conflict
    core_no_conflict = (masks["fg_core"] | masks["bg_core"]) & (~core_conflict)
    extent_teacher_fg = masks["extent"] & teacher_fg
    extent_teacher_bg = masks["extent"] & (~teacher_fg)
    ones = torch.ones_like(teacher_prob, dtype=torch.float32, device=device)
    conditional_values = {
        "core_conflict_map": (ones, core_conflict),
        "core_no_conflict_map": (ones, core_no_conflict),
        "extent_teacher_fg_map": (ones, extent_teacher_fg),
        "extent_teacher_bg_map": (ones, extent_teacher_bg),
        "unknown_map": (ones, masks["unknown"]),
        "other_map": (ones, masks["other"]),
        "teacher_fg_fg_core": (teacher_fg.float(), masks["fg_core"]),
        "teacher_fg_extent": (teacher_fg.float(), masks["extent"]),
    }
    conditional_sums = {name: 0.0 for name in TEPR_V11_CONDITIONAL_KEYS}
    conditional_counts = {name: 0.0 for name in TEPR_V11_CONDITIONAL_KEYS}
    conditional_means = {name: 0.0 for name in TEPR_V11_CONDITIONAL_KEYS}
    for name, (value, mask) in conditional_values.items():
        count = float(mask.float().sum().detach().item())
        value_sum = float((value * mask.float()).sum().detach().item()) if count > 0.0 else 0.0
        conditional_sums[name] = value_sum
        conditional_counts[name] = count
        conditional_means[name] = value_sum / count if count > 0.0 else 0.0
    pixel_count = int(teacher_prob.numel())
    return {
        "tepr_scale": 0.0,
        "memory_active": False,
        "history_count_min": 0,
        "history_count_mean": 0.0,
        "history_count_max": 0,
        "history_valid_ratio": 0.0,
        "temporal_mean_min": 0.0,
        "temporal_mean_mean": 0.0,
        "temporal_mean_max": 0.0,
        "temporal_var_min": 0.0,
        "temporal_var_mean": 0.0,
        "temporal_var_max": 0.0,
        "temporal_conf_mean": 0.0,
        "temporal_stability_mean": 0.0,
        "temporal_reliability_mean": 0.0,
        "core_conflict_mean": 0.0,
        "core_conflict_max": 0.0,
        "extent_conflict_mean": 0.0,
        "extent_conflict_max": 0.0,
        "dino_margin_min": 0.0,
        "dino_margin_mean": 0.0,
        "dino_margin_max": 0.0,
        "fg_tendency_mean": 0.0,
        "teacher_map_raw_min": 1.0,
        "teacher_map_raw_mean": 1.0,
        "teacher_map_raw_max": 1.0,
        "teacher_map_min": 1.0,
        "teacher_map_mean": 1.0,
        "teacher_map_max": 1.0,
        "teacher_map_lt_030_ratio": 0.0,
        "teacher_map_lt_050_ratio": 0.0,
        "teacher_map_gt_090_ratio": 1.0,
        "region_means": {name: 1.0 for name in masks},
        "region_sums": dict(region_counts),
        "region_counts": region_counts,
        "conditional_means": conditional_means,
        "conditional_sums": conditional_sums,
        "conditional_counts": conditional_counts,
        "temporal_var_hist": torch.zeros(4096, dtype=torch.float32),
        "extent_bg_reliability_hist": torch.zeros(4096, dtype=torch.float32),
        "fg_core_conflict_count": int(fg_conflict.sum().detach().item()),
        "fg_core_count": int(masks["fg_core"].sum().detach().item()),
        "bg_core_conflict_count": int(bg_conflict.sum().detach().item()),
        "bg_core_count": int(masks["bg_core"].sum().detach().item()),
        "extent_teacher_fg_count": int(extent_teacher_fg.sum().detach().item()),
        "extent_teacher_bg_count": int(extent_teacher_bg.sum().detach().item()),
        "extent_count": int(masks["extent"].sum().detach().item()),
        "map_pixel_count": pixel_count,
        "map_sum": float(pixel_count),
        "map_lt_030_count": 0,
        "map_lt_050_count": 0,
        "map_gt_090_count": pixel_count,
    }


def new_tepr_epoch_accumulator():
    region_names = ("fg_core", "bg_core", "extent", "unknown", "other")
    return {
        "batches": 0,
        "history_count_mean_sum": 0.0,
        "temporal_var_sum": 0.0,
        "temporal_reliability_sum": 0.0,
        "core_conflict_sum": 0.0,
        "extent_conflict_sum": 0.0,
        "map_min": None,
        "map_max": None,
        "map_sum": 0.0,
        "map_pixels": 0,
        "map_lt_030": 0,
        "map_lt_050": 0,
        "map_gt_090": 0,
        "region_sums": {name: 0.0 for name in region_names},
        "region_counts": {name: 0.0 for name in region_names},
        "conditional_sums": {name: 0.0 for name in TEPR_V11_CONDITIONAL_KEYS},
        "conditional_counts": {name: 0.0 for name in TEPR_V11_CONDITIONAL_KEYS},
        "fg_core_conflict_count": 0,
        "fg_core_count": 0,
        "bg_core_conflict_count": 0,
        "bg_core_count": 0,
        "extent_teacher_fg_count": 0,
        "extent_teacher_bg_count": 0,
        "extent_count": 0,
        "prediction_pixels": 0,
        "student_pred_fg_count": 0,
        "teacher_pred_fg_count": 0,
        "variance_hist": torch.zeros(4096, dtype=torch.float64),
        "extent_bg_reliability_hist": torch.zeros(4096, dtype=torch.float64),
        "memory_active_batches": 0,
    }


def update_tepr_epoch_accumulator(accumulator, stats):
    accumulator["batches"] += 1
    accumulator["history_count_mean_sum"] += float(stats["history_count_mean"])
    accumulator["temporal_var_sum"] += float(stats["temporal_var_mean"])
    accumulator["temporal_reliability_sum"] += float(stats["temporal_reliability_mean"])
    accumulator["core_conflict_sum"] += float(stats["core_conflict_mean"])
    accumulator["extent_conflict_sum"] += float(stats["extent_conflict_mean"])
    map_min = float(stats["teacher_map_min"])
    map_max = float(stats["teacher_map_max"])
    accumulator["map_min"] = map_min if accumulator["map_min"] is None else min(accumulator["map_min"], map_min)
    accumulator["map_max"] = map_max if accumulator["map_max"] is None else max(accumulator["map_max"], map_max)
    accumulator["map_sum"] += float(stats["map_sum"])
    accumulator["map_pixels"] += int(stats["map_pixel_count"])
    accumulator["map_lt_030"] += int(stats["map_lt_030_count"])
    accumulator["map_lt_050"] += int(stats["map_lt_050_count"])
    accumulator["map_gt_090"] += int(stats["map_gt_090_count"])
    for name in accumulator["region_sums"]:
        accumulator["region_sums"][name] += float(stats["region_sums"][name])
        accumulator["region_counts"][name] += float(stats["region_counts"][name])
    for name in accumulator["conditional_sums"]:
        accumulator["conditional_sums"][name] += float(
            stats.get("conditional_sums", {}).get(name, 0.0)
        )
        accumulator["conditional_counts"][name] += float(
            stats.get("conditional_counts", {}).get(name, 0.0)
        )
    for name in (
        "fg_core_conflict_count",
        "fg_core_count",
        "bg_core_conflict_count",
        "bg_core_count",
        "extent_teacher_fg_count",
        "extent_teacher_bg_count",
        "extent_count",
    ):
        accumulator[name] += int(stats.get(name, 0))
    accumulator["prediction_pixels"] += int(stats.get("prediction_pixel_count", 0))
    accumulator["student_pred_fg_count"] += int(stats.get("student_pred_fg_count", 0))
    accumulator["teacher_pred_fg_count"] += int(stats.get("teacher_pred_fg_count", 0))
    accumulator["variance_hist"] += stats["temporal_var_hist"].double().cpu()
    accumulator["extent_bg_reliability_hist"] += stats.get(
        "extent_bg_reliability_hist", torch.zeros(4096)
    ).double().cpu()
    accumulator["memory_active_batches"] += int(bool(stats.get("memory_active", True)))


def log_tepr_first_batch(logger, cfg, epoch, sample_indices, stats):
    regions = stats["region_means"]
    logger.log(
        f"[TEPR-Lite FirstBatch] USE_TEPR_LITE={bool(getattr(cfg, 'USE_TEPR_LITE', False))} | "
        f"TEPR_VERSION={getattr(cfg, 'TEPR_VERSION', '')} | epoch={int(epoch):03d} | "
        f"TEPR_ROUTING_MODE={getattr(cfg, 'TEPR_ROUTING_MODE', 'legacy_v1')} | "
        f"tepr_scale={float(stats['tepr_scale']):.8f} | memory_active={bool(stats.get('memory_active', True))}"
    )
    logger.log(
        f"[TEPR-Lite FirstBatch] sample_index shape/min/max={list(sample_indices.shape)}/"
        f"{int(sample_indices.min().item())}/{int(sample_indices.max().item())} | "
        f"history_count min/mean/max={int(stats['history_count_min'])}/"
        f"{float(stats['history_count_mean']):.4f}/{int(stats['history_count_max'])}"
    )
    logger.log(
        "[TEPR-Lite FirstBatch] temporal mean min/mean/max="
        f"{stats['temporal_mean_min']:.6f}/{stats['temporal_mean_mean']:.6f}/{stats['temporal_mean_max']:.6f} | "
        "var min/mean/max="
        f"{stats['temporal_var_min']:.6f}/{stats['temporal_var_mean']:.6f}/{stats['temporal_var_max']:.6f} | "
        f"conf/stability/reliability={stats['temporal_conf_mean']:.6f}/"
        f"{stats['temporal_stability_mean']:.6f}/{stats['temporal_reliability_mean']:.6f}"
    )
    logger.log(
        "[TEPR-Lite FirstBatch] core conflict mean/max="
        f"{stats['core_conflict_mean']:.6f}/{stats['core_conflict_max']:.6f} | "
        "extent conflict mean/max="
        f"{stats['extent_conflict_mean']:.6f}/{stats['extent_conflict_max']:.6f} | "
        "dino margin min/mean/max="
        f"{stats['dino_margin_min']:.6f}/{stats['dino_margin_mean']:.6f}/{stats['dino_margin_max']:.6f} | "
        f"fg_tendency_mean={stats['fg_tendency_mean']:.6f}"
    )
    logger.log(
        "[TEPR-Lite FirstBatch] map raw min/mean/max="
        f"{stats['teacher_map_raw_min']:.6f}/{stats['teacher_map_raw_mean']:.6f}/"
        f"{stats['teacher_map_raw_max']:.6f} | eff min/mean/max="
        f"{stats['teacher_map_min']:.6f}/{stats['teacher_map_mean']:.6f}/{stats['teacher_map_max']:.6f}"
    )
    logger.log(
        "[TEPR-Lite FirstBatch] map fg/bg/extent/unknown/other="
        f"{regions['fg_core']:.6f}/{regions['bg_core']:.6f}/{regions['extent']:.6f}/"
        f"{regions['unknown']:.6f}/{regions['other']:.6f}"
    )
    if str(getattr(cfg, "TEPR_ROUTING_MODE", "legacy_v1")).lower() == "state_conditional_asymneg":
        conditional = stats.get("conditional_means", {})
        fg_count = max(int(stats.get("fg_core_count", 0)), 1)
        bg_count = max(int(stats.get("bg_core_count", 0)), 1)
        extent_count = max(int(stats.get("extent_count", 0)), 1)
        reliability_hist = stats.get("extent_bg_reliability_hist", torch.zeros(4096))
        logger.log(
            "[TEPR-Lite-v1.1 FirstBatch] core conflict ratio fg/bg="
            f"{int(stats.get('fg_core_conflict_count', 0)) / fg_count:.6f}/"
            f"{int(stats.get('bg_core_conflict_count', 0)) / bg_count:.6f} | "
            "map conflict/no-conflict="
            f"{conditional.get('core_conflict_map', 0.0):.6f}/"
            f"{conditional.get('core_no_conflict_map', 0.0):.6f} | "
            "valid conflict/no-conflict="
            f"{int(stats.get('conditional_counts', {}).get('core_conflict_map', 0.0) > 0.0)}/"
            f"{int(stats.get('conditional_counts', {}).get('core_no_conflict_map', 0.0) > 0.0)}"
        )
        logger.log(
            "[TEPR-Lite-v1.1 FirstBatch] extent teacher fg/bg ratio="
            f"{int(stats.get('extent_teacher_fg_count', 0)) / extent_count:.6f}/"
            f"{int(stats.get('extent_teacher_bg_count', 0)) / extent_count:.6f} | "
            "map fg/bg="
            f"{conditional.get('extent_teacher_fg_map', 0.0):.6f}/"
            f"{conditional.get('extent_teacher_bg_map', 0.0):.6f} | "
            "valid fg/bg="
            f"{int(stats.get('conditional_counts', {}).get('extent_teacher_fg_map', 0.0) > 0.0)}/"
            f"{int(stats.get('conditional_counts', {}).get('extent_teacher_bg_map', 0.0) > 0.0)}"
        )
        logger.log(
            "[TEPR-Lite-v1.1 FirstBatch] extent-bg reliability p10/p50/p90="
            f"{histogram_quantile_from_hist(reliability_hist, 0.10):.6f}/"
            f"{histogram_quantile_from_hist(reliability_hist, 0.50):.6f}/"
            f"{histogram_quantile_from_hist(reliability_hist, 0.90):.6f} | "
            f"dino_ceiling={conditional.get('extent_bg_dino_ceiling', 0.0):.6f} | "
            f"raw_final_weight={conditional.get('extent_bg_final_weight', 0.0):.6f}"
        )
        logger.log(
            "[TEPR-Lite-v1.1 FirstBatch] extent-bg map fg-like/ambiguous/bg-like="
            f"{conditional.get('extent_bg_fg_like_weight', 0.0):.6f}/"
            f"{conditional.get('extent_bg_ambiguous_weight', 0.0):.6f}/"
            f"{conditional.get('extent_bg_bg_like_weight', 0.0):.6f} | "
            "unknown/other="
            f"{conditional.get('unknown_map', 0.0):.6f}/"
            f"{conditional.get('other_map', 0.0):.6f} | "
            "valid unknown/other="
            f"{int(stats.get('conditional_counts', {}).get('unknown_map', 0.0) > 0.0)}/"
            f"{int(stats.get('conditional_counts', {}).get('other_map', 0.0) > 0.0)}"
        )


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


def masked_smooth_l1(prediction, target, mask, eps=1e-6):
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    denom = mask.sum()
    if float(denom.detach().item()) <= 0.0:
        return prediction.sum() * 0.0
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (loss * mask).sum() / (denom + float(eps))


def masked_soft_bce_with_logits(logits, target, mask, pixel_weight=None, eps=1e-6):
    weight = mask.to(device=logits.device, dtype=logits.dtype)
    if pixel_weight is not None:
        weight = weight * pixel_weight.to(device=logits.device, dtype=logits.dtype)
    denom = weight.sum()
    if float(denom.detach().item()) <= 0.0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight).sum() / (denom + float(eps))


def _cssd_decoder_coarse_enabled(cfg):
    if use_csd_v1r_head(cfg):
        return bool(getattr(cfg, "CSD_V1R_USE_COARSE_AUX", False)), float(
            getattr(cfg, "LAMBDA_CSD_V1R_COARSE_AUX", 0.5)
        )
    if use_csd_head(cfg):
        return bool(getattr(cfg, "CSD_USE_COARSE_AUX", False)), float(
            getattr(cfg, "LAMBDA_CSD_COARSE_AUX", 0.5)
        )
    return False, 0.0


def compute_cssd_original_group_loss(
    output,
    static_target,
    static_weight_map,
    teacher_target,
    teacher_map_eff,
    teacher_routing_scale,
    epoch,
    cfg,
    apply_final=True,
    apply_coarse=True,
    apply_base=True,
):
    if not isinstance(output, dict):
        raise RuntimeError("CSSD original group loss requires dict decoder output.")
    eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
    static_schedule, teacher_schedule = get_dabe_pu_despl_schedule(epoch, cfg)
    static_terms = []
    teacher_terms = []
    term_weights = []
    zero = extract_logits(output).sum() * 0.0
    result = {
        "loss_static_final": zero,
        "loss_teacher_final": zero,
        "loss_static_coarse": zero,
        "loss_teacher_coarse": zero,
        "loss_static_base": zero,
        "loss_teacher_base": zero,
    }

    if bool(apply_final):
        final_logits = resize_logits_for_loss(extract_logits(output), cfg)
        result["loss_static_final"] = weighted_bce_with_logits(
            final_logits, static_target, static_weight_map, eps=eps
        )
        result["loss_teacher_final"] = rast_teacher_bce_with_logits(
            final_logits,
            teacher_target,
            teacher_map_eff,
            cfg,
            teacher_routing_scale,
            apply_to_loss=teacher_routing_apply_flag(cfg, "final"),
            eps=eps,
        )
        static_terms.append(result["loss_static_final"])
        teacher_terms.append(result["loss_teacher_final"])
        term_weights.append(1.0)

    coarse_enabled, coarse_lambda = _cssd_decoder_coarse_enabled(cfg)
    if bool(apply_coarse) and coarse_enabled:
        if "coarse_logits_68" not in output:
            raise RuntimeError("CSSD high supervised coarse loss requires coarse_logits_68.")
        coarse_logits = resize_logits_for_loss(output["coarse_logits_68"], cfg)
        result["loss_static_coarse"] = weighted_bce_with_logits(
            coarse_logits, static_target, static_weight_map, eps=eps
        )
        result["loss_teacher_coarse"] = rast_teacher_bce_with_logits(
            coarse_logits,
            teacher_target,
            teacher_map_eff,
            cfg,
            teacher_routing_scale,
            apply_to_loss=teacher_routing_apply_flag(cfg, "coarse"),
            eps=eps,
        )
        static_terms.append(result["loss_static_coarse"])
        teacher_terms.append(result["loss_teacher_coarse"])
        term_weights.append(coarse_lambda)

    if bool(apply_base) and bool(getattr(cfg, "USE_BASE_AUX_LOSS", False)):
        if "base_logits" not in output:
            raise RuntimeError("CSSD high supervised base loss requires base_logits.")
        base_lambda = (
            float(getattr(cfg, "LAMBDA_BASE_AUX", 0.3))
            if is_before_finetune_reset(cfg, epoch)
            else float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.1))
        )
        base_logits = resize_logits_for_loss(output["base_logits"], cfg)
        result["loss_static_base"] = weighted_bce_with_logits(
            base_logits, static_target, static_weight_map, eps=eps
        )
        result["loss_teacher_base"] = rast_teacher_bce_with_logits(
            base_logits,
            teacher_target,
            teacher_map_eff,
            cfg,
            teacher_routing_scale,
            apply_to_loss=teacher_routing_apply_flag(cfg, "base"),
            eps=eps,
        )
        static_terms.append(result["loss_static_base"])
        teacher_terms.append(result["loss_teacher_base"])
        term_weights.append(base_lambda)

    if not term_weights:
        raise RuntimeError("CSSD high supervised loss has no enabled final/coarse/base terms.")
    weight_sum = max(1e-12, sum(term_weights))
    result["loss_static_group"] = sum(
        weight * term for weight, term in zip(term_weights, static_terms)
    ) / weight_sum
    result["loss_teacher_group"] = sum(
        weight * term for weight, term in zip(term_weights, teacher_terms)
    ) / weight_sum
    result["loss_group"] = (
        float(static_schedule) * result["loss_static_group"]
        + float(teacher_schedule) * result["loss_teacher_group"]
    )
    result["static_weight"] = float(static_schedule)
    result["teacher_weight"] = float(teacher_schedule)
    return result


def forward_cssd_high_microbatches(
    student,
    feature_hr,
    image_68,
    bg_reliable_68,
    cfg,
):
    if feature_hr.ndim != 4 or list(feature_hr.shape[1:]) != [
        int(getattr(cfg, "CSSD_HR_FEATURE_CHANNELS", 384)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
    ]:
        raise RuntimeError(f"CSSD HR feature shape mismatch: {list(feature_hr.shape)}")
    microbatch = int(getattr(cfg, "CSSD_HR_MICROBATCH", 4))
    if microbatch <= 0:
        raise RuntimeError(f"CSSD_HR_MICROBATCH must be positive, got {microbatch}")
    outputs = []
    for start in range(0, int(feature_hr.shape[0]), microbatch):
        end = min(int(feature_hr.shape[0]), start + microbatch)
        chunk_out = forward_seg_head(
            student,
            make_single_feature_model_input(cfg, feature_hr[start:end]),
            cfg,
            image_68=image_68[start:end],
            return_aux=True,
            bg_reliable_68=bg_reliable_68[start:end] if bg_reliable_68 is not None else None,
        )
        if not isinstance(chunk_out, dict):
            raise RuntimeError("CSSD shared high forward requires dict output.")
        outputs.append(chunk_out)
    required = ("logits", "final_logits", "coarse_logits_68", "base_logits", "coarse_logits_native")
    merged = {}
    for key in required:
        if any(key not in output for output in outputs):
            raise RuntimeError(f"CSSD high forward output missing {key!r}")
        merged[key] = torch.cat([output[key] for output in outputs], dim=0)
        if not bool(torch.isfinite(merged[key]).all().item()):
            raise RuntimeError(f"CSSD high forward output {key!r} contains NaN/Inf.")
    batch_size = int(feature_hr.shape[0])
    expected_68 = [batch_size, 1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    for key in ("logits", "final_logits", "coarse_logits_68", "base_logits"):
        if list(merged[key].shape) != expected_68:
            raise RuntimeError(
                f"CSSD high output {key} shape mismatch: {list(merged[key].shape)} != {expected_68}."
            )
    native_size = int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48))
    if list(merged["coarse_logits_native"].shape) != [batch_size, 1, native_size, native_size]:
        raise RuntimeError(
            "CSSD high coarse native shape mismatch: "
            f"{list(merged['coarse_logits_native'].shape)} != {[batch_size, 1, native_size, native_size]}."
        )
    return merged, len(outputs)


def _cssd_normalize_per_image(value, eps=1e-6):
    flat = value.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return torch.clamp((value - minimum) / (maximum - minimum + float(eps)), 0.0, 1.0)


def _cssd_probability_sobel(probability):
    sobel_x = probability.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = probability.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    dx = F.conv2d(probability, sobel_x, padding=1)
    dy = F.conv2d(probability, sobel_y, padding=1)
    return _cssd_normalize_per_image(torch.sqrt(dx * dx + dy * dy + 1e-6))


def build_cssd_distillation_losses(normal_output, high_output, batch, image_68, cfg, cssd_scale):
    normal_logits = resize_logits_for_loss(extract_logits(normal_output), cfg)
    high_logits = resize_logits_for_loss(extract_logits(high_output), cfg)
    p_normal = torch.sigmoid(normal_logits)
    p_high = torch.sigmoid(high_logits)
    p_high_target = p_high.detach() if bool(getattr(cfg, "CSSD_DETACH_HR_TARGET", True)) else p_high
    conf_normal = (2.0 * torch.abs(p_normal.detach() - 0.5)).clamp(0.0, 1.0)
    conf_high = (2.0 * torch.abs(p_high_target - 0.5)).clamp(0.0, 1.0)
    conf_adv = F.relu(conf_high - conf_normal - float(getattr(cfg, "CSSD_CONF_ADV_MARGIN", 0.05)))

    device = normal_logits.device
    fg_core = batch["pu_fg_core"].to(device, non_blocking=True).float() > 0.5
    bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float() > 0.5
    extent = batch["pu_extent"].to(device, non_blocking=True).float() > 0.5
    unknown = batch["pu_unknown"].to(device, non_blocking=True).float() > 0.5
    target_soft = batch["pu_target_soft"].to(device, non_blocking=True).float()
    weight_map = batch["pu_weight_map"].to(device, non_blocking=True).float()
    low_target_bg = (target_soft < 0.15) & (weight_map > 0.50)
    if bool(getattr(cfg, "CSSD_USE_UNKNOWN", False)):
        raise RuntimeError("CSSD-v1a forbids unknown-region transfer.")

    if bool(getattr(cfg, "CSSD_EXCLUDE_FG_BG_CONFLICT", True)):
        positive_conflict = (p_high_target >= 0.5) & (bg_core | low_target_bg)
        negative_conflict = (p_high_target < 0.5) & fg_core
        allowed = ~(positive_conflict | negative_conflict)
    else:
        allowed = torch.ones_like(fg_core)
    core_mask = (
        (fg_core | bg_core)
        & (conf_high >= float(getattr(cfg, "CSSD_CORE_CONF_THRESH", 0.60)))
        & allowed
    ).detach()
    prob_diff = torch.abs(p_high_target - p_normal.detach())
    transfer_raw = (
        extent
        & (~unknown)
        & allowed
        & (conf_high >= float(getattr(cfg, "CSSD_TRANSFER_CONF_THRESH", 0.70)))
        & (conf_adv > 0.0)
        & (prob_diff >= float(getattr(cfg, "CSSD_MIN_PROB_DIFF", 0.05)))
    ).detach()
    transfer_score = (conf_adv + 0.5 * prob_diff).detach()
    transfer_mask = cap_mask_by_score_per_image(
        transfer_raw,
        transfer_score,
        float(getattr(cfg, "CSSD_TRANSFER_MAX_RATIO_PER_IMAGE", 0.03)),
        int(getattr(cfg, "CSSD_TRANSFER_MIN_PIXELS_PER_IMAGE", 4)),
    ).detach()
    transfer_weight = (0.5 + conf_adv).detach()
    loss_core = masked_smooth_l1(p_normal, p_high_target, core_mask)
    loss_transfer = masked_soft_bce_with_logits(
        normal_logits, p_high_target, transfer_mask, pixel_weight=transfer_weight
    )
    loss_pred = (
        float(getattr(cfg, "CSSD_CORE_LOSS_MULT", 0.25)) * loss_core
        + float(getattr(cfg, "CSSD_TRANSFER_LOSS_MULT", 1.0)) * loss_transfer
    )

    boundary_normal = _cssd_probability_sobel(p_normal)
    boundary_high = _cssd_probability_sobel(p_high_target).detach()
    image_edge = compute_sobel_mag_68(image_68.detach()).detach()
    teacher_threshold = per_image_quantile_map(
        boundary_high, float(getattr(cfg, "CSSD_BOUNDARY_TEACHER_Q", 0.70))
    )
    edge_threshold = per_image_quantile_map(
        image_edge, float(getattr(cfg, "CSSD_BOUNDARY_IMAGE_EDGE_Q", 0.50))
    )
    boundary_raw = (
        (boundary_high >= teacher_threshold)
        & (image_edge >= edge_threshold)
        & (~low_target_bg)
    ).detach()
    boundary_score = (boundary_high * image_edge).detach()
    boundary_mask = cap_mask_by_score_per_image(
        boundary_raw,
        boundary_score,
        float(getattr(cfg, "CSSD_BOUNDARY_MAX_RATIO_PER_IMAGE", 0.10)),
        int(getattr(cfg, "CSSD_BOUNDARY_MIN_PIXELS_PER_IMAGE", 8)),
    ).detach()
    loss_boundary = masked_smooth_l1(
        boundary_normal, boundary_high.detach(), boundary_mask
    )

    transfer_per_image = transfer_mask.float().flatten(1).mean(dim=1)
    boundary_per_image = boundary_mask.float().flatten(1).mean(dim=1)
    transfer_cap = float(getattr(cfg, "CSSD_TRANSFER_MAX_RATIO_PER_IMAGE", 0.03))
    boundary_cap = float(getattr(cfg, "CSSD_BOUNDARY_MAX_RATIO_PER_IMAGE", 0.10))
    if float(transfer_per_image.max().item()) > transfer_cap + 1e-6:
        raise RuntimeError("CSSD transfer mask exceeded per-image cap.")
    if float(boundary_per_image.max().item()) > boundary_cap + 1e-6:
        raise RuntimeError("CSSD boundary mask exceeded per-image cap.")

    normal_binary = p_normal.detach() >= 0.5
    high_binary = p_high_target >= 0.5
    stats = {
        "cssd_scale": float(cssd_scale),
        "normal_prob_mean": float(p_normal.detach().mean().item()),
        "high_prob_mean": float(p_high_target.mean().item()),
        "normal_pred_area": float(normal_binary.float().mean().item()),
        "high_pred_area": float(high_binary.float().mean().item()),
        "normal_high_prob_abs_diff": float(torch.abs(p_normal.detach() - p_high_target).mean().item()),
        "normal_high_binary_agreement": float((normal_binary == high_binary).float().mean().item()),
        "normal_conf_mean": float(conf_normal.mean().item()),
        "high_conf_mean": float(conf_high.mean().item()),
        "high_more_conf_ratio": float((conf_high > conf_normal).float().mean().item()),
        "conf_adv_mean": float(conf_adv.mean().item()),
        "core_mask_ratio": float(core_mask.float().mean().item()),
        "transfer_raw_ratio": float(transfer_raw.float().mean().item()),
        "transfer_capped_ratio": float(transfer_mask.float().mean().item()),
        "transfer_valid_image_ratio": float((transfer_per_image > 0).float().mean().item()),
        "boundary_mask_ratio": float(boundary_mask.float().mean().item()),
        "boundary_valid_image_ratio": float((boundary_per_image > 0).float().mean().item()),
    }
    return loss_core, loss_transfer, loss_pred, loss_boundary, stats


def compute_sobel_mag_68(image_68):
    if image_68 is None:
        raise RuntimeError("TCE edge support requires image_68 from USE_NDR_BRANCH=True.")
    if image_68.ndim != 4 or image_68.shape[1] != 3:
        raise RuntimeError(f"TCE expects image_68 [B,3,H,W], got {list(image_68.shape)}.")
    gray = 0.299 * image_68[:, 0:1] + 0.587 * image_68[:, 1:2] + 0.114 * image_68[:, 2:3]
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=image_68.device,
        dtype=image_68.dtype,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=image_68.device,
        dtype=image_68.dtype,
    ).unsqueeze(0)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    sobel = torch.sqrt(dx * dx + dy * dy + 1e-6)
    sobel = sobel / (sobel.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return sobel.clamp(0.0, 1.0)


def _mean_on_mask(value, mask):
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom.detach().item()) <= 0.0:
        return 0.0
    return float((value * mask_f).sum().detach().item() / (denom.detach().item() + 1e-6))


def tce_lower_bound_loss_for_logits(logits, lost_cover_mask, new_boundary_mask, cfg):
    mask = (lost_cover_mask | new_boundary_mask).to(device=logits.device)
    if int(mask.sum().detach().item()) <= 0:
        return logits.sum() * 0.0
    prob = torch.sigmoid(logits)
    floor = torch.zeros_like(prob)
    floor = torch.where(
        lost_cover_mask.to(device=logits.device),
        torch.full_like(floor, float(getattr(cfg, "TCE_LOST_TARGET_FLOOR", 0.45))),
        floor,
    )
    floor = torch.where(
        new_boundary_mask.to(device=logits.device),
        torch.full_like(floor, float(getattr(cfg, "TCE_NEW_TARGET_FLOOR", 0.35))),
        floor,
    )
    loss_map = F.relu(floor - prob).pow(2)
    mask_f = mask.float()
    return (loss_map * mask_f).sum() / (mask_f.sum() + 1e-6)


def tce_teacher_bce_with_logits(logits, target, weight_map, cfg, eps=1e-6):
    if weight_map is None:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    weight_map = weight_map.to(device=logits.device, dtype=logits.dtype)
    if bool(getattr(cfg, "TCE_WEIGHTED_NORMALIZE", True)):
        return weighted_bce_with_logits(logits, target, weight_map, eps=eps)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight_map).mean()


def build_tce_masks_and_teacher_map(
    cfg,
    batch,
    student_logits,
    teacher_prob,
    teacher_binary,
    rast_teacher_map_eff,
    epoch,
    device,
    image_68=None,
):
    if image_68 is None:
        image_68 = batch.get("image_68")
    base_teacher_map = (
        rast_teacher_map_eff.to(device=device, dtype=student_logits.dtype)
        if rast_teacher_map_eff is not None
        else torch.ones_like(student_logits)
    )
    cover_prob = batch["tce_cover_prob_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    cover_binary = batch["tce_cover_binary_68"].to(device, non_blocking=True).float() >= 0.5
    cover_conf = batch["tce_cover_conf_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    if bool(getattr(cfg, "TCE_DETACH_COVER", True)):
        cover_prob = cover_prob.detach()
        cover_binary = cover_binary.detach()
        cover_conf = cover_conf.detach()
    cover_area_batch = batch["tce_cover_area"].to(device, non_blocking=True).float()
    if cover_area_batch.ndim == 0:
        cover_area_batch = cover_area_batch.view(1)
    tce_scale = float(get_tce_scale(cfg, epoch))
    lambda_tce_eff = float(getattr(cfg, "TCE_LAMBDA_MAX", 0.010)) * tce_scale
    zero_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    stats = {
        "tce_scale": tce_scale,
        "lambda_tce_eff": lambda_tce_eff,
        "cover_area_mean": float(cover_area_batch.mean().detach().item()),
        "current_area_mean": 0.0,
        "shrink_gate_ratio": 0.0,
        "bg_safe_gate_ratio": 0.0,
        "image_gate_ratio": 0.0,
        "lost_raw_ratio": 0.0,
        "lost_capped_ratio": 0.0,
        "new_raw_ratio": 0.0,
        "new_capped_ratio": 0.0,
        "tce_total_ratio": 0.0,
        "tce_valid_image_ratio": 0.0,
        "lost_margin_mean": 0.0,
        "new_margin_mean": 0.0,
        "lost_cover_conf_mean": 0.0,
        "current_teacher_conf_lost_mean": 0.0,
        "current_teacher_conf_new_mean": 0.0,
        "teacher_map_final_mean": float(base_teacher_map.detach().mean().item()),
        "teacher_map_final_tce_mean": 1.0,
        "loss_tce": 0.0,
        "tce_skipped_no_fg_proto": 0,
        "tce_skipped_no_bg_proto": 0,
    }
    if tce_scale <= 0.0:
        student_prob_for_area = torch.sigmoid(student_logits.detach())
        stats["current_area_mean"] = float((student_prob_for_area >= 0.5).float().mean().detach().item())
        return None, zero_mask, zero_mask, stats

    region_masks = build_rast_region_masks(batch, device)
    fg_core = region_masks["fg_core"]
    bg_core = region_masks["bg_core"]
    extent = region_masks["extent"]
    if bool(getattr(cfg, "TCE_USE_UNKNOWN", False)):
        extent = extent | region_masks["unknown"]

    student_prob_src = student_logits.detach() if bool(getattr(cfg, "TCE_DETACH_MASK", True)) else student_logits
    student_prob = torch.sigmoid(student_prob_src)
    teacher_prob = teacher_prob.detach()
    teacher_binary = teacher_binary.detach() >= 0.5
    current_teacher_bg = ~teacher_binary
    current_teacher_conf = (2.0 * (teacher_prob - 0.5).abs()).clamp(0.0, 1.0)

    if bool(getattr(cfg, "TCE_USE_DINO_MARGIN", True)):
        margin_68, margin_stats = compute_dino_core_margin_68(
            cfg,
            batch,
            fg_core,
            bg_core,
            student_logits.shape[-2:],
            device,
            prefix="TCE",
        )
        if bool(getattr(cfg, "TCE_DETACH_MARGIN", True)):
            margin_68 = margin_68.detach()
        stats["tce_skipped_no_fg_proto"] = int(margin_stats.get("tce_skipped_no_fg_proto", 0))
        stats["tce_skipped_no_bg_proto"] = int(margin_stats.get("tce_skipped_no_bg_proto", 0))
    else:
        margin_68 = torch.zeros_like(student_logits)

    if bool(getattr(cfg, "TCE_USE_NEAR_CURRENT_FG", True)):
        current_fg = student_prob >= float(getattr(cfg, "TCE_NEAR_FG_THRESH", 0.50))
        radius = int(getattr(cfg, "TCE_NEAR_FG_RADIUS", 3))
        near_current_fg = F.max_pool2d(
            current_fg.float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ) > 0.5
    else:
        near_current_fg = torch.ones_like(student_logits, dtype=torch.bool)

    if bool(getattr(cfg, "TCE_USE_EDGE_SUPPORT", True)):
        if image_68 is None:
            raise RuntimeError("TCE_USE_EDGE_SUPPORT=True requires image_68 in batch or helper input.")
        edge_score = compute_sobel_mag_68(image_68.to(device=device, dtype=student_logits.dtype))
        q = float(getattr(cfg, "TCE_EDGE_Q", 0.60))
        edge_thresh = torch.quantile(edge_score.flatten(1), q, dim=1).view(-1, 1, 1, 1)
        edge_support = edge_score >= edge_thresh
    else:
        edge_score = torch.zeros_like(student_logits)
        edge_support = torch.ones_like(student_logits, dtype=torch.bool)

    current_pred = student_prob >= 0.5
    current_area = current_pred.float().flatten(1).mean(dim=1)
    extent_f = extent.float()
    extent_sum = extent_f.flatten(1).sum(dim=1).clamp_min(1e-6)
    current_fg_extent_ratio = (current_pred & extent).float().flatten(1).sum(dim=1) / extent_sum
    cover_fg_extent_ratio = (cover_binary & extent).float().flatten(1).sum(dim=1) / extent_sum
    shrink_risk = (
        current_area < cover_area_batch * float(getattr(cfg, "TCE_SHRINK_AREA_RATIO", 0.90))
    ) | (
        current_fg_extent_ratio
        < cover_fg_extent_ratio * float(getattr(cfg, "TCE_SHRINK_EXTENT_RATIO", 0.85))
    )
    if not bool(getattr(cfg, "TCE_USE_IMAGE_GATE", True)):
        shrink_risk = torch.ones_like(shrink_risk, dtype=torch.bool)

    bg_sum = bg_core.float().flatten(1).sum(dim=1).clamp_min(1e-6)
    bg_core_teacher_fg_ratio = (bg_core & teacher_binary).float().flatten(1).sum(dim=1) / bg_sum
    background_safe = bg_core_teacher_fg_ratio <= float(getattr(cfg, "TCE_BG_CORE_TEACHER_FG_MAX", 0.015))
    if not bool(getattr(cfg, "TCE_USE_BG_RISK_GATE", True)):
        background_safe = torch.ones_like(background_safe, dtype=torch.bool)
    image_gate = (shrink_risk & background_safe).view(-1, 1, 1, 1)

    non_bg = ~bg_core if bool(getattr(cfg, "TCE_EXCLUDE_BG_CORE", True)) else torch.ones_like(bg_core)
    if bool(getattr(cfg, "TCE_USE_LOST_COVER", True)):
        lost_raw = (
            extent
            & cover_binary
            & current_teacher_bg
            & (cover_conf >= float(getattr(cfg, "TCE_COVER_CONF_THRESH", 0.60)))
            & (current_teacher_conf <= float(getattr(cfg, "TCE_CURRENT_BG_CONF_MAX", 0.90)))
            & (margin_68 >= float(getattr(cfg, "TCE_LOST_MARGIN_THRESH", 0.05)))
            & near_current_fg
            & non_bg
            & image_gate
        )
    else:
        lost_raw = torch.zeros_like(student_logits, dtype=torch.bool)
    if bool(getattr(cfg, "TCE_USE_NEW_BOUNDARY", True)):
        new_raw = (
            extent
            & (~cover_binary)
            & current_teacher_bg
            & (margin_68 >= float(getattr(cfg, "TCE_NEW_MARGIN_THRESH", 0.10)))
            & (student_prob >= float(getattr(cfg, "TCE_NEW_STUDENT_PROB_THRESH", 0.25)))
            & near_current_fg
            & edge_support
            & non_bg
            & image_gate
        )
    else:
        new_raw = torch.zeros_like(student_logits, dtype=torch.bool)
    lost_score = margin_68 + 0.5 * cover_conf - 0.3 * current_teacher_conf
    new_score = margin_68 + 0.3 * student_prob + 0.2 * edge_score - 0.3 * current_teacher_conf
    lost_capped = cap_mask_by_score_per_image(
        lost_raw,
        lost_score,
        float(getattr(cfg, "TCE_LOST_MAX_RATIO_PER_IMAGE", 0.004)),
        0,
    )
    new_capped = cap_mask_by_score_per_image(
        new_raw,
        new_score,
        float(getattr(cfg, "TCE_NEW_MAX_RATIO_PER_IMAGE", 0.002)),
        0,
    )
    total_raw = lost_capped | new_capped
    total_score = torch.maximum(lost_score, new_score)
    total_capped = cap_mask_by_score_per_image(
        total_raw,
        total_score,
        float(getattr(cfg, "TCE_TOTAL_MAX_RATIO_PER_IMAGE", 0.005)),
        int(getattr(cfg, "TCE_MIN_PIXELS_PER_IMAGE", 4)),
    )
    lost_final = lost_capped & total_capped
    new_final = new_capped & total_capped & (~lost_final)

    teacher_map_final = base_teacher_map.clone()
    if bool(getattr(cfg, "TCE_USE_TEACHER_BG_REWEIGHT", True)) and bool(getattr(cfg, "TCE_APPLY_TO_FINAL", True)):
        teacher_map_final = torch.where(
            lost_final & current_teacher_bg,
            teacher_map_final * float(getattr(cfg, "TCE_LOST_BG_TEACHER_MULT", 0.60)),
            teacher_map_final,
        )
        teacher_map_final = torch.where(
            new_final & current_teacher_bg,
            teacher_map_final * float(getattr(cfg, "TCE_NEW_BG_TEACHER_MULT", 0.75)),
            teacher_map_final,
        )

    total_final = lost_final | new_final
    valid_images = total_final.float().flatten(1).sum(dim=1) >= int(getattr(cfg, "TCE_MIN_PIXELS_PER_IMAGE", 4))
    stats.update(
        {
            "current_area_mean": float(current_area.detach().mean().item()),
            "shrink_gate_ratio": float(shrink_risk.float().detach().mean().item()),
            "bg_safe_gate_ratio": float(background_safe.float().detach().mean().item()),
            "image_gate_ratio": float(image_gate.float().detach().mean().item()),
            "lost_raw_ratio": float(lost_raw.float().detach().mean().item()),
            "lost_capped_ratio": float(lost_final.float().detach().mean().item()),
            "new_raw_ratio": float(new_raw.float().detach().mean().item()),
            "new_capped_ratio": float(new_final.float().detach().mean().item()),
            "tce_total_ratio": float(total_final.float().detach().mean().item()),
            "tce_valid_image_ratio": float(valid_images.float().detach().mean().item()),
            "lost_margin_mean": _mean_on_mask(margin_68, lost_final),
            "new_margin_mean": _mean_on_mask(margin_68, new_final),
            "lost_cover_conf_mean": _mean_on_mask(cover_conf, lost_final),
            "current_teacher_conf_lost_mean": _mean_on_mask(current_teacher_conf, lost_final),
            "current_teacher_conf_new_mean": _mean_on_mask(current_teacher_conf, new_final),
            "teacher_map_final_mean": float(teacher_map_final.detach().mean().item()),
            "teacher_map_final_tce_mean": _mean_on_mask(teacher_map_final, total_final),
        }
    )
    return teacher_map_final, lost_final.detach(), new_final.detach(), stats


def lceg_teacher_bce_with_logits(logits, target, weight_map, cfg, eps=1e-6):
    if weight_map is None:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    weight_map = weight_map.to(device=logits.device, dtype=logits.dtype)
    if bool(getattr(cfg, "LCEG_WEIGHTED_NORMALIZE", True)):
        return weighted_bce_with_logits(logits, target, weight_map, eps=eps)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weight_map).mean()


def _lceg_branch_lower_bound_loss(logits, mask, floor_value, eps=1e-6):
    mask = mask.to(device=logits.device)
    if int(mask.sum().detach().item()) <= 0:
        return logits.sum() * 0.0
    prob = torch.sigmoid(logits)
    loss_map = F.relu(float(floor_value) - prob).pow(2)
    mask_f = mask.float()
    return (loss_map * mask_f).sum() / (mask_f.sum().clamp_min(float(eps)))


def lceg_lower_bound_loss_for_logits(logits, core_mask, lost_mask, new_mask, cfg):
    loss_core = _lceg_branch_lower_bound_loss(
        logits,
        core_mask,
        float(getattr(cfg, "LCEG_CORE_TARGET_FLOOR", 0.55)),
    )
    loss_lost = _lceg_branch_lower_bound_loss(
        logits,
        lost_mask,
        float(getattr(cfg, "LCEG_LOST_TARGET_FLOOR", 0.45)),
    )
    loss_new = _lceg_branch_lower_bound_loss(
        logits,
        new_mask,
        float(getattr(cfg, "LCEG_NEW_TARGET_FLOOR", 0.35)),
    )
    loss_total = (
        float(getattr(cfg, "LCEG_BRANCH_WEIGHT_CORE", 1.0)) * loss_core
        + float(getattr(cfg, "LCEG_BRANCH_WEIGHT_LOST", 0.8)) * loss_lost
        + float(getattr(cfg, "LCEG_BRANCH_WEIGHT_NEW", 0.5)) * loss_new
    )
    return loss_total, {
        "core": loss_core,
        "lost": loss_lost,
        "new": loss_new,
    }


def build_lceg_masks_and_teacher_maps(
    cfg,
    batch,
    student_logits,
    teacher_prob,
    teacher_binary,
    rast_teacher_map_eff,
    epoch,
    device,
    image_68=None,
):
    if image_68 is None:
        image_68 = batch.get("image_68")
    base_teacher_map = (
        rast_teacher_map_eff.to(device=device, dtype=student_logits.dtype)
        if rast_teacher_map_eff is not None
        else torch.ones_like(student_logits)
    )
    cover_prob = batch["lceg_cover_prob_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    cover_binary = batch["lceg_cover_binary_68"].to(device, non_blocking=True).float() >= 0.5
    cover_conf = batch["lceg_cover_conf_68"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
    if bool(getattr(cfg, "LCEG_DETACH_COVER", True)):
        cover_prob = cover_prob.detach()
        cover_binary = cover_binary.detach()
        cover_conf = cover_conf.detach()
    cover_area_batch = batch["lceg_cover_area"].to(device, non_blocking=True).float()
    if cover_area_batch.ndim == 0:
        cover_area_batch = cover_area_batch.view(1)

    lceg_scale = float(get_lceg_scale(cfg, epoch))
    lambda_lceg_eff = float(getattr(cfg, "LCEG_LAMBDA_MAX", 0.020)) * lceg_scale
    zero_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    stats = {
        "lceg_scale": lceg_scale,
        "lambda_lceg_eff": lambda_lceg_eff,
        "cover_area_mean": float(cover_area_batch.mean().detach().item()),
        "cover_binary_area_mean": float(cover_binary.float().detach().mean().item()),
        "cover_conf_mean": float(cover_conf.detach().mean().item()),
        "current_area_mean": 0.0,
        "current_fg_extent_ratio": 0.0,
        "cover_fg_extent_ratio": 0.0,
        "shrink_gate_ratio": 0.0,
        "bg_safe_gate_ratio": 0.0,
        "core_raw_ratio": 0.0,
        "core_capped_ratio": 0.0,
        "lost_raw_ratio": 0.0,
        "lost_capped_ratio": 0.0,
        "new_raw_ratio": 0.0,
        "new_capped_ratio": 0.0,
        "lceg_total_ratio": 0.0,
        "lceg_valid_image_ratio": 0.0,
        "core_conf_mean": 0.0,
        "lost_cover_prob_mean": 0.0,
        "lost_margin_mean": 0.0,
        "new_margin_mean": 0.0,
        "teacher_map_final_mean": float(base_teacher_map.detach().mean().item()),
        "teacher_map_final_lceg_mean": 1.0,
        "teacher_map_coarse_mean": float(base_teacher_map.detach().mean().item()),
        "teacher_map_coarse_lceg_mean": 1.0,
        "lceg_skipped_no_fg_proto": 0,
        "lceg_skipped_no_bg_proto": 0,
    }
    if lceg_scale <= 0.0:
        student_prob_for_area = torch.sigmoid(student_logits.detach())
        stats["current_area_mean"] = float((student_prob_for_area >= 0.5).float().mean().detach().item())
        return None, None, zero_mask, zero_mask, zero_mask, stats

    region_masks = build_rast_region_masks(batch, device)
    fg_core = region_masks["fg_core"]
    bg_core = region_masks["bg_core"]
    extent = region_masks["extent"]

    student_prob_src = student_logits.detach() if bool(getattr(cfg, "LCEG_DETACH_MASK", True)) else student_logits
    student_prob = torch.sigmoid(student_prob_src)
    teacher_prob = teacher_prob.detach()
    teacher_binary = teacher_binary.detach() >= 0.5
    teacher_bg = ~teacher_binary
    teacher_conf = (2.0 * (teacher_prob - 0.5).abs()).clamp(0.0, 1.0)

    if bool(getattr(cfg, "LCEG_USE_DINO_MARGIN", True)):
        margin_68, margin_stats = compute_dino_core_margin_68(
            cfg,
            batch,
            fg_core,
            bg_core,
            student_logits.shape[-2:],
            device,
            prefix="LCEG",
        )
        if bool(getattr(cfg, "LCEG_DETACH_MARGIN", True)):
            margin_68 = margin_68.detach()
        stats["lceg_skipped_no_fg_proto"] = int(margin_stats.get("lceg_skipped_no_fg_proto", 0))
        stats["lceg_skipped_no_bg_proto"] = int(margin_stats.get("lceg_skipped_no_bg_proto", 0))
    else:
        margin_68 = torch.zeros_like(student_logits)

    if bool(getattr(cfg, "LCEG_USE_NEAR_FG", True)):
        current_fg_for_near = student_prob >= float(getattr(cfg, "LCEG_NEAR_CURRENT_FG_THRESH", 0.50))
        cover_fg_for_near = cover_prob >= float(getattr(cfg, "LCEG_NEAR_COVER_PROB_THRESH", 0.40))
        near_seed = current_fg_for_near | cover_fg_for_near
        radius = int(getattr(cfg, "LCEG_NEAR_FG_RADIUS", 3))
        near_fg = F.max_pool2d(
            near_seed.float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ) > 0.5
    else:
        near_fg = torch.ones_like(student_logits, dtype=torch.bool)

    if bool(getattr(cfg, "LCEG_USE_EDGE_SUPPORT", True)):
        if image_68 is None:
            raise RuntimeError("LCEG_USE_EDGE_SUPPORT=True requires image_68 in batch or helper input.")
        edge_score = compute_sobel_mag_68(image_68.to(device=device, dtype=student_logits.dtype))
        q = float(getattr(cfg, "LCEG_EDGE_Q", 0.60))
        edge_thresh = torch.quantile(edge_score.flatten(1), q, dim=1).view(-1, 1, 1, 1)
        edge_support = edge_score >= edge_thresh
    else:
        edge_score = torch.zeros_like(student_logits)
        edge_support = torch.ones_like(student_logits, dtype=torch.bool)

    current_pred = student_prob >= 0.5
    current_area = current_pred.float().flatten(1).mean(dim=1)
    extent_f = extent.float()
    extent_sum = extent_f.flatten(1).sum(dim=1).clamp_min(1e-6)
    current_fg_extent_ratio = (current_pred & extent).float().flatten(1).sum(dim=1) / extent_sum
    cover_extent_fg = cover_prob >= float(getattr(cfg, "LCEG_LOST_COVER_PROB_THRESH", 0.40))
    cover_fg_extent_ratio = (cover_extent_fg & extent).float().flatten(1).sum(dim=1) / extent_sum
    shrink_gate = (
        current_area < cover_area_batch * float(getattr(cfg, "LCEG_SHRINK_AREA_RATIO", 0.98))
    ) | (
        current_fg_extent_ratio
        < cover_fg_extent_ratio * float(getattr(cfg, "LCEG_SHRINK_EXTENT_RATIO", 0.95))
    )
    if not bool(getattr(cfg, "LCEG_USE_IMAGE_GATE_FOR_EXTENT", True)):
        shrink_gate = torch.ones_like(shrink_gate, dtype=torch.bool)

    bg_sum = bg_core.float().flatten(1).sum(dim=1).clamp_min(1e-6)
    bg_core_teacher_fg_ratio = (bg_core & teacher_binary).float().flatten(1).sum(dim=1) / bg_sum
    bg_safe = bg_core_teacher_fg_ratio <= float(getattr(cfg, "LCEG_BG_CORE_TEACHER_FG_MAX", 0.020))
    if not bool(getattr(cfg, "LCEG_USE_BG_RISK_GATE", True)):
        bg_safe = torch.ones_like(bg_safe, dtype=torch.bool)
    bg_safe_gate = bg_safe.view(-1, 1, 1, 1)
    extent_image_gate = (shrink_gate & bg_safe).view(-1, 1, 1, 1)

    non_bg = ~bg_core if bool(getattr(cfg, "LCEG_EXCLUDE_BG_CORE", True)) else torch.ones_like(bg_core)
    if bool(getattr(cfg, "LCEG_USE_CORE_GUARD", True)):
        core_raw = (
            fg_core
            & teacher_bg
            & (teacher_conf <= float(getattr(cfg, "LCEG_CURRENT_BG_CONF_MAX_CORE", 0.98)))
            & (student_prob >= float(getattr(cfg, "LCEG_CORE_STUDENT_PROB_MIN", 0.0)))
            & bg_safe_gate
        )
    else:
        core_raw = torch.zeros_like(student_logits, dtype=torch.bool)
    if bool(getattr(cfg, "LCEG_USE_LOST_EXTENT", True)):
        lost_raw = (
            extent
            & (cover_prob >= float(getattr(cfg, "LCEG_LOST_COVER_PROB_THRESH", 0.40)))
            & teacher_bg
            & (teacher_conf <= float(getattr(cfg, "LCEG_CURRENT_BG_CONF_MAX_EXTENT", 0.95)))
            & (student_prob >= float(getattr(cfg, "LCEG_LOST_STUDENT_PROB_MIN", 0.05)))
            & (margin_68 >= float(getattr(cfg, "LCEG_LOST_MARGIN_THRESH", -0.02)))
            & near_fg
            & non_bg
            & extent_image_gate
        )
    else:
        lost_raw = torch.zeros_like(student_logits, dtype=torch.bool)
    if bool(getattr(cfg, "LCEG_USE_NEW_BOUNDARY", True)):
        new_raw = (
            extent
            & (cover_prob < float(getattr(cfg, "LCEG_NEW_COVER_PROB_MAX", 0.40)))
            & teacher_bg
            & (teacher_conf <= float(getattr(cfg, "LCEG_CURRENT_BG_CONF_MAX_EXTENT", 0.95)))
            & (student_prob >= float(getattr(cfg, "LCEG_NEW_STUDENT_PROB_MIN", 0.15)))
            & (margin_68 >= float(getattr(cfg, "LCEG_NEW_MARGIN_THRESH", 0.05)))
            & near_fg
            & edge_support
            & non_bg
            & extent_image_gate
        )
    else:
        new_raw = torch.zeros_like(student_logits, dtype=torch.bool)

    core_score = (1.0 - teacher_conf) + 0.3 * cover_prob + 0.2 * student_prob
    lost_score = margin_68 + 0.5 * cover_prob + 0.3 * student_prob - 0.3 * teacher_conf
    new_score = margin_68 + 0.3 * student_prob + 0.2 * edge_score - 0.3 * teacher_conf
    min_pixels = int(getattr(cfg, "LCEG_MIN_PIXELS_PER_IMAGE", 4))
    core_capped = cap_mask_by_score_per_image(
        core_raw,
        core_score,
        float(getattr(cfg, "LCEG_CORE_MAX_RATIO_PER_IMAGE", 0.010)),
        min_pixels,
    )
    lost_capped = cap_mask_by_score_per_image(
        lost_raw,
        lost_score,
        float(getattr(cfg, "LCEG_LOST_MAX_RATIO_PER_IMAGE", 0.010)),
        min_pixels,
    )
    new_capped = cap_mask_by_score_per_image(
        new_raw,
        new_score,
        float(getattr(cfg, "LCEG_NEW_MAX_RATIO_PER_IMAGE", 0.003)),
        min_pixels,
    )

    neg = torch.full_like(student_logits, -1e6)
    total_raw = core_capped | lost_capped | new_capped
    total_score = torch.maximum(
        torch.where(core_capped, core_score, neg),
        torch.maximum(
            torch.where(lost_capped, lost_score, neg),
            torch.where(new_capped, new_score, neg),
        ),
    )
    total_capped = cap_mask_by_score_per_image(
        total_raw,
        total_score,
        float(getattr(cfg, "LCEG_TOTAL_MAX_RATIO_PER_IMAGE", 0.020)),
        min_pixels,
    )
    core_final = core_capped & total_capped
    lost_final = lost_capped & total_capped
    new_final = new_capped & total_capped
    total_final = core_final | lost_final | new_final

    teacher_map_final = base_teacher_map.clone()
    teacher_map_coarse = base_teacher_map.clone()
    if bool(getattr(cfg, "LCEG_USE_TEACHER_BG_REWEIGHT", True)):
        for apply_map, allow_apply in (
            (teacher_map_final, bool(getattr(cfg, "LCEG_APPLY_TO_FINAL", True))),
            (teacher_map_coarse, bool(getattr(cfg, "LCEG_APPLY_TO_COARSE_AUX", True))),
        ):
            if allow_apply:
                apply_map.copy_(
                    torch.where(
                        core_final & teacher_bg,
                        apply_map * float(getattr(cfg, "LCEG_CORE_BG_TEACHER_MULT", 0.50)),
                        apply_map,
                    )
                )
                apply_map.copy_(
                    torch.where(
                        lost_final & teacher_bg,
                        apply_map * float(getattr(cfg, "LCEG_LOST_BG_TEACHER_MULT", 0.60)),
                        apply_map,
                    )
                )
                apply_map.copy_(
                    torch.where(
                        new_final & teacher_bg,
                        apply_map * float(getattr(cfg, "LCEG_NEW_BG_TEACHER_MULT", 0.75)),
                        apply_map,
                    )
                )

    valid_images = total_final.float().flatten(1).sum(dim=1) >= min_pixels
    stats.update(
        {
            "current_area_mean": float(current_area.detach().mean().item()),
            "current_fg_extent_ratio": float(current_fg_extent_ratio.detach().mean().item()),
            "cover_fg_extent_ratio": float(cover_fg_extent_ratio.detach().mean().item()),
            "shrink_gate_ratio": float(shrink_gate.float().detach().mean().item()),
            "bg_safe_gate_ratio": float(bg_safe.float().detach().mean().item()),
            "core_raw_ratio": float(core_raw.float().detach().mean().item()),
            "core_capped_ratio": float(core_final.float().detach().mean().item()),
            "lost_raw_ratio": float(lost_raw.float().detach().mean().item()),
            "lost_capped_ratio": float(lost_final.float().detach().mean().item()),
            "new_raw_ratio": float(new_raw.float().detach().mean().item()),
            "new_capped_ratio": float(new_final.float().detach().mean().item()),
            "lceg_total_ratio": float(total_final.float().detach().mean().item()),
            "lceg_valid_image_ratio": float(valid_images.float().detach().mean().item()),
            "core_conf_mean": _mean_on_mask(teacher_conf, core_final),
            "lost_cover_prob_mean": _mean_on_mask(cover_prob, lost_final),
            "lost_margin_mean": _mean_on_mask(margin_68, lost_final),
            "new_margin_mean": _mean_on_mask(margin_68, new_final),
            "teacher_map_final_mean": float(teacher_map_final.detach().mean().item()),
            "teacher_map_final_lceg_mean": _mean_on_mask(teacher_map_final, total_final),
            "teacher_map_coarse_mean": float(teacher_map_coarse.detach().mean().item()),
            "teacher_map_coarse_lceg_mean": _mean_on_mask(teacher_map_coarse, total_final),
        }
    )
    return (
        teacher_map_final,
        teacher_map_coarse,
        core_final.detach(),
        lost_final.detach(),
        new_final.detach(),
        stats,
    )


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


def _cacd_aux_float(output, name, default=0.0):
    aux = output.get("cacd_aux", {}) if isinstance(output, dict) else {}
    value = aux.get(name, default)
    return float(value.detach().item()) if torch.is_tensor(value) else float(value)


def log_cacd_first_batch(logger, model_input, image_68, sobel_68, output, anchor_stats, loss_seg, loss_anchor, loss_total, cfg):
    logger.log("[CACD FirstBatch]")
    logger.log(
        "[CACD FirstBatch] f10/f11/f12 shape = "
        f"{list(model_input['f10'].shape)}/{list(model_input['f11'].shape)}/{list(model_input['f12'].shape)}"
    )
    logger.log(
        "[CACD FirstBatch] z10/z11/z12 shape = "
        f"[B,{int(getattr(cfg, 'CACD_DIM', 96))},37,37]"
    )
    logger.log(f"[CACD FirstBatch] image_68/sobel_68 shape = {list(image_68.shape)}/{list(sobel_68.shape)}")
    logger.log(
        "[CACD FirstBatch] alpha10/alpha11 = "
        f"{_cacd_aux_float(output, 'alpha10'):.8f}/{_cacd_aux_float(output, 'alpha11'):.8f}"
    )
    for name in ("gate10", "gate11", "cos10", "cos11", "q_consensus"):
        logger.log(
            f"[CACD FirstBatch] {name} mean/min/max = "
            f"{_cacd_aux_float(output, name + '_mean'):.8f}/"
            f"{_cacd_aux_float(output, name + '_min'):.8f}/"
            f"{_cacd_aux_float(output, name + '_max'):.8f}"
        )
    anchor_prob = output["anchor_prob"]
    logger.log(f"[CACD FirstBatch] anchor_logits shape = {list(output['anchor_logits'].shape)}")
    logger.log(
        "[CACD FirstBatch] anchor fg/bg/amb mean = "
        f"{float(anchor_prob[:, 0:1].mean()):.8f}/"
        f"{float(anchor_prob[:, 1:2].mean()):.8f}/"
        f"{float(anchor_prob[:, 2:3].mean()):.8f}"
    )
    logger.log(
        "[CACD FirstBatch] fg/bg slot shape = "
        f"[B,{int(getattr(cfg, 'CACD_NUM_FG_SLOTS', 4))},{int(getattr(cfg, 'CACD_SLOT_DIM', 96))}]/"
        f"[B,{int(getattr(cfg, 'CACD_NUM_BG_SLOTS', 4))},{int(getattr(cfg, 'CACD_SLOT_DIM', 96))}]"
    )
    logger.log(
        "[CACD FirstBatch] fg/bg slot cosine mean/max = "
        f"{_cacd_aux_float(output, 'fg_slot_cos_mean'):.8f}/{_cacd_aux_float(output, 'fg_slot_cos_max'):.8f} | "
        f"{_cacd_aux_float(output, 'bg_slot_cos_mean'):.8f}/{_cacd_aux_float(output, 'bg_slot_cos_max'):.8f}"
    )
    logger.log(
        "[CACD FirstBatch] fg/bg attention overlap mean/max = "
        f"{_cacd_aux_float(output, 'fg_attn_overlap_mean'):.8f}/{_cacd_aux_float(output, 'fg_attn_overlap_max'):.8f} | "
        f"{_cacd_aux_float(output, 'bg_attn_overlap_mean'):.8f}/{_cacd_aux_float(output, 'bg_attn_overlap_max'):.8f}"
    )
    logger.log(
        "[CACD FirstBatch] context gate mean/min/max, relation/delta abs mean = "
        f"{_cacd_aux_float(output, 'context_gate_mean'):.8f}/"
        f"{_cacd_aux_float(output, 'context_gate_min'):.8f}/"
        f"{_cacd_aux_float(output, 'context_gate_max'):.8f} | "
        f"{_cacd_aux_float(output, 'context_relation_abs_mean'):.8f}/"
        f"{_cacd_aux_float(output, 'context_delta_abs_mean'):.8f}"
    )
    logger.log(
        "[CACD FirstBatch] base/coarse/final logits shape = "
        f"{list(output['base_logits'].shape)}/{list(output['coarse_logits'].shape)}/{list(output['final_logits'].shape)}"
    )
    logger.log(
        "[CACD FirstBatch] detail feature/gate = "
        f"[B,{int(getattr(cfg, 'CACD_DETAIL_DIM', 32))},68,68] | "
        f"{_cacd_aux_float(output, 'detail_gate_mean'):.8f}/"
        f"{_cacd_aux_float(output, 'detail_gate_min'):.8f}/"
        f"{_cacd_aux_float(output, 'detail_gate_max'):.8f}"
    )
    weighted = float(getattr(cfg, "CACD_ANCHOR_LOSS_WEIGHT", 0.05)) * float(loss_anchor.detach().item())
    logger.log(
        "[CACD FirstBatch] loss seg/anchor_fg/anchor_bg/anchor/weighted/total = "
        f"{float(loss_seg.detach().item()):.8f}/"
        f"{anchor_stats['loss_anchor_fg']:.8f}/{anchor_stats['loss_anchor_bg']:.8f}/"
        f"{float(loss_anchor.detach().item()):.8f}/{weighted:.8f}/{float(loss_total.detach().item()):.8f}"
    )


def log_csd_first_batch(logger, model_input, image_68, output, csd_stats):
    logger.log(f"[CSD FirstBatch] feature shape = {list(model_input.shape)}")
    logger.log(f"[CSD FirstBatch] image_68 shape = {list(image_68.shape)}")
    logger.log(f"[CSD FirstBatch] semantic_feat_37 shape = {list(output['semantic_feat_37'].shape)}")
    logger.log(f"[CSD FirstBatch] semantic_feat_68 shape = {list(output['semantic_feat_68'].shape)}")
    logger.log(f"[CSD FirstBatch] detail_feat_68 shape = {list(output['detail_feat_68'].shape)}")
    logger.log(f"[CSD FirstBatch] coarse_logits_37 shape = {list(output['coarse_logits_37'].shape)}")
    logger.log(f"[CSD FirstBatch] coarse_logits_68 shape = {list(output['coarse_logits_68'].shape)}")
    logger.log(f"[CSD FirstBatch] residual_logits shape = {list(output['residual_logits_68'].shape)}")
    logger.log(f"[CSD FirstBatch] final_logits shape = {list(output['logits'].shape)}")
    logger.log(f"[CSD FirstBatch] boundary_logits shape = {list(output['boundary_logits'].shape)}")
    logger.log(f"[CSD FirstBatch] csd_scale = {output_scalar(output, 'csd_scale'):.8f}")
    logger.log(f"[CSD FirstBatch] beta_eff = {output_scalar(output, 'csd_beta_eff'):.8f}")
    logger.log(
        "[CSD FirstBatch] detail_gate mean/min/max = "
        f"{output_scalar(output, 'csd_detail_gate_mean'):.8f}/"
        f"{output_scalar(output, 'csd_detail_gate_min'):.8f}/"
        f"{output_scalar(output, 'csd_detail_gate_max'):.8f}"
    )
    logger.log(f"[CSD FirstBatch] bg_reliable area = {float(csd_stats['bg_reliable_ratio']):.8f}")
    logger.log(
        "[CSD FirstBatch] boundary_target mean/max = "
        f"{float(csd_stats['boundary_target_mean']):.8f}/"
        f"{float(csd_stats['boundary_target_max']):.8f}"
    )


def log_csd_v1r_first_batch(logger, model_input, image_68, output, csd_stats):
    logger.log(f"[CSD-v1R FirstBatch] feature shape = {list(model_input.shape)}")
    logger.log(f"[CSD-v1R FirstBatch] image_68 shape = {list(image_68.shape)}")
    logger.log(f"[CSD-v1R FirstBatch] old coarse 37 shape = {list(output['old_coarse_logits_37'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] old coarse 68 shape = {list(output['old_coarse_logits_68'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] semantic_feat_37 shape = {list(output['semantic_feat_37'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] semantic_feat_68 shape = {list(output['semantic_feat_68'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] detail_feat_68 shape = {list(output['detail_feat_68'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] residual_logits shape = {list(output['residual_logits_68'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] final_logits shape = {list(output['logits'].shape)}")
    logger.log(f"[CSD-v1R FirstBatch] csd_scale = {output_scalar(output, 'csd_scale'):.8f}")
    logger.log(f"[CSD-v1R FirstBatch] beta_eff = {output_scalar(output, 'csd_beta_eff'):.8f}")
    logger.log(
        "[CSD-v1R FirstBatch] final_minus_coarse_abs_mean = "
        f"{output_scalar(output, 'csd_final_minus_coarse_abs_mean', 0.0):.8f}"
    )
    logger.log(
        "[CSD-v1R FirstBatch] detail_gate mean/min/max = "
        f"{output_scalar(output, 'csd_detail_gate_mean'):.8f}/"
        f"{output_scalar(output, 'csd_detail_gate_min'):.8f}/"
        f"{output_scalar(output, 'csd_detail_gate_max'):.8f}"
    )
    logger.log(f"[CSD-v1R FirstBatch] bg_reliable area = {float(csd_stats['bg_reliable_ratio']):.8f}")
    logger.log(
        "[CSD-v1R FirstBatch] bg detail loss/weighted = "
        f"{float(csd_stats['loss_bg_detail']):.8f}/"
        f"{float(csd_stats['loss_bg_detail_weighted']):.8f}"
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
    if "ndr_v2_shape_alpha_eff" in output:
        logger.log(f"[NDR-v2 FirstBatch] shape_alpha_eff = {output_scalar(output, 'ndr_v2_shape_alpha_eff'):.8f}")
        logger.log(f"[NDR-v2 FirstBatch] boundary_band shape = {list(output['boundary_band_68'].shape)}")
        logger.log(f"[NDR-v2 FirstBatch] edge_norm shape = {list(output['edge_norm_68'].shape)}")
        logger.log(f"[NDR-v2 FirstBatch] shape_boost shape = {list(output['shape_boost_68'].shape)}")
        logger.log(
            "[NDR-v2 FirstBatch] detail_gate_v1/v2 mean = "
            f"{output_scalar(output, 'ndr_v2_detail_gate_v1_mean'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_detail_gate_v2_mean'):.8f}"
        )
        logger.log(
            "[NDR-v2 FirstBatch] boundary mean/min/max = "
            f"{output_scalar(output, 'ndr_v2_boundary_mean'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_boundary_min'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_boundary_max'):.8f}"
        )
        logger.log(
            "[NDR-v2 FirstBatch] edge_norm mean/min/max = "
            f"{output_scalar(output, 'ndr_v2_edge_norm_mean'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_edge_norm_min'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_edge_norm_max'):.8f}"
        )
        logger.log(
            "[NDR-v2 FirstBatch] shape_boost mean/min/max = "
            f"{output_scalar(output, 'ndr_v2_shape_boost_mean'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_shape_boost_min'):.8f}/"
            f"{output_scalar(output, 'ndr_v2_shape_boost_max'):.8f}"
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
        sobel_68 = make_sobel_68(cfg, batch, device)
        image_136 = make_image_136(cfg, batch, device)
        output = forward_seg_head(
            student,
            model_input,
            cfg,
            image_68=image_68,
            image_136=image_136,
            sobel_68=sobel_68,
            return_aux=False,
        )
        logits = extract_logits_for_eval(
            output,
            cfg,
        )
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


def should_save_epoch_checkpoint(epoch, max_epoch, save_interval, reset_epoch=None, save_every_epoch=False):
    if bool(save_every_epoch):
        return True
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
    if use_cssd(cfg):
        logger.log(
            "CSSD HR feature cache path = "
            f"{Path(getattr(cfg, 'CSSD_HR_CACHE_ROOT')).resolve()}"
        )
        logger.log(f"CSSD HR feature shape example = {train_dataset.cssd_hr_feature_shape}")
        logger.log(f"first CSSD HR feature file = {train_dataset.cssd_hr_first_cache_path}")
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
            "dabe_pu_v12_shape_desplsched",
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
    if getattr(cfg, "USE_TCE", False):
        logger.log(f"TCE cover cache path = {train_dataset.tce_cover_cache_root}")
        logger.log(f"first TCE cover file = {train_dataset.tce_cover_first_cache_path}")
        logger.log(f"use_tce = {bool(getattr(cfg, 'USE_TCE', False))}")
        logger.log(f"tce_version = {getattr(cfg, 'TCE_VERSION', 'v1_temporal_coverage_expansion')}")
        logger.log(f"tce_cover_epoch = {int(getattr(cfg, 'TCE_COVER_EPOCH', 30))}")
        logger.log(f"tce_cover_model = {getattr(cfg, 'TCE_COVER_MODEL', 'student')}")
    if getattr(cfg, "USE_LCEG", False):
        logger.log(f"LCEG cover cache path = {train_dataset.lceg_cover_cache_root}")
        logger.log(f"first LCEG cover file = {train_dataset.lceg_cover_first_cache_path}")
        logger.log(f"use_lceg = {bool(getattr(cfg, 'USE_LCEG', False))}")
        logger.log(f"lceg_version = {getattr(cfg, 'LCEG_VERSION', 'v1_late_core_extent_guard')}")
        logger.log(f"lceg_cover_epoch = {int(getattr(cfg, 'LCEG_COVER_EPOCH', 25))}")
        logger.log(f"lceg_cover_model = {getattr(cfg, 'LCEG_COVER_MODEL', 'student')}")
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
        if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() == "pu_v12_shape_complete":
            logger.log(f"first batch target_base_mean = {float(batch['pu_target_base'].float().mean().item()):.6f}")
            logger.log(f"first batch target_v12_mean = {float(pu_target.mean().item()):.6f}")
            logger.log(
                "first batch target_delta_mean = "
                f"{float((pu_target - batch['pu_target_base'].float()).mean().item()):.6f}"
            )
            logger.log(f"first batch weight_base_mean = {float(batch['pu_weight_base'].float().mean().item()):.6f}")
            logger.log(
                "first batch sc_bg_lock/sc_extent_agree_fg/sc_lost_extent/sc_new_boundary area = "
                f"{float(batch['pu_sc_bg_lock'].float().mean().item()):.6f}/"
                f"{float(batch['pu_sc_extent_agree_fg'].float().mean().item()):.6f}/"
                f"{float(batch['pu_sc_lost_extent'].float().mean().item()):.6f}/"
                f"{float(batch['pu_sc_new_boundary'].float().mean().item()):.6f}"
            )
    if getattr(cfg, "USE_TCE", False):
        cover_prob = batch["tce_cover_prob_68"].float()
        cover_binary = batch["tce_cover_binary_68"].float()
        cover_conf = batch["tce_cover_conf_68"].float()
        logger.log(f"[TCE FirstBatch] USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
        logger.log(f"[TCE FirstBatch] TCE_VERSION = {getattr(cfg, 'TCE_VERSION', 'v1_temporal_coverage_expansion')}")
        logger.log(f"[TCE FirstBatch] tce_scale = {get_tce_scale(cfg, 1):.6f}")
        logger.log(f"[TCE FirstBatch] TCE_COVER_CACHE_ROOT = {getattr(cfg, 'TCE_COVER_CACHE_ROOT', '')}")
        logger.log(f"[TCE FirstBatch] cover_prob shape = {list(cover_prob.shape)}")
        logger.log(
            "[TCE FirstBatch] cover_prob min/mean/max = "
            f"{float(cover_prob.min().item()):.6f}/"
            f"{float(cover_prob.mean().item()):.6f}/"
            f"{float(cover_prob.max().item()):.6f}"
        )
        logger.log(f"[TCE FirstBatch] cover_binary area = {float(cover_binary.mean().item()):.6f}")
        logger.log(f"[TCE FirstBatch] cover_conf mean = {float(cover_conf.mean().item()):.6f}")
        logger.log("[TCE FirstBatch] lost/new candidate ratio = 0.000000/0.000000 because tce_scale=0")
    if getattr(cfg, "USE_LCEG", False):
        cover_prob = batch["lceg_cover_prob_68"].float()
        cover_binary = batch["lceg_cover_binary_68"].float()
        cover_conf = batch["lceg_cover_conf_68"].float()
        cover_area = batch["lceg_cover_area"].float()
        logger.log(f"[LCEG FirstBatch] USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
        logger.log(f"[LCEG FirstBatch] LCEG_VERSION = {getattr(cfg, 'LCEG_VERSION', 'v1_late_core_extent_guard')}")
        logger.log(f"[LCEG FirstBatch] lceg_scale = {get_lceg_scale(cfg, 1):.6f}")
        logger.log(f"[LCEG FirstBatch] LCEG_COVER_CACHE_ROOT = {getattr(cfg, 'LCEG_COVER_CACHE_ROOT', '')}")
        logger.log(f"[LCEG FirstBatch] cover_prob shape = {list(cover_prob.shape)}")
        logger.log(
            "[LCEG FirstBatch] cover_prob min/mean/max = "
            f"{float(cover_prob.min().item()):.6f}/"
            f"{float(cover_prob.mean().item()):.6f}/"
            f"{float(cover_prob.max().item()):.6f}"
        )
        logger.log(f"[LCEG FirstBatch] cover_binary area = {float(cover_binary.mean().item()):.6f}")
        logger.log(f"[LCEG FirstBatch] cover_conf mean = {float(cover_conf.mean().item()):.6f}")
        logger.log(f"[LCEG FirstBatch] cover_area mean = {float(cover_area.mean().item()):.6f}")
        logger.log("[LCEG FirstBatch] core/lost/new candidate ratio = 0.000000/0.000000/0.000000 because lceg_scale=0")


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
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_train_samples != -1:
        raise ValueError("--max_samples and --max_train_samples cannot be used together.")
    sample_limit = args.max_samples if args.max_samples is not None else args.max_train_samples
    if sample_limit == 0 or sample_limit < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max_epochs must be a positive integer.")
    if args.resume and args.debug_loader_only:
        raise ValueError("--resume cannot be combined with --debug_loader_only.")

    cfg = load_config(args.config)
    validate_esa_ber_config(cfg)
    validate_tepr_lite_config(cfg)
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
            "dabe_pu_v12_shape_desplsched",
        }:
            raise RuntimeError(
                "USE_DABE_PU=True requires P_INIT_MODE in "
                "{'dabe_pu_v11', 'dabe_pu_v11_oem', "
                "'dabe_pu_v11_desplsched', 'dabe_pu_v11_desplsched_exactreset', "
                "'dabe_pu_v11_desplsched_A1_keepteacher_lowlr', "
                "'dabe_pu_v11_desplsched_A2_resetteacher_highlr', "
                "'dabe_pu_v11_desplsched_softteacher', "
                "'dabe_pu_v11_desplsched_dabehard', "
                "'dabe_pu_v12_shape_desplsched'}."
            )
        if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() not in {"pu_v11", "pu_v12_shape_complete"}:
            raise RuntimeError("USE_DABE_PU=True requires DABE_PU_VERSION in {'pu_v11', 'pu_v12_shape_complete'}.")
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
            if bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)):
                if not bool(getattr(cfg, "USE_ESA_ASYM", False)) or not bool(
                    getattr(cfg, "USE_RAST", False)
                ):
                    raise RuntimeError(
                        "ESA_POST_RESET_ENABLE=True requires USE_ESA_ASYM=True and USE_RAST=True."
                    )
                if str(getattr(cfg, "ESA_POST_RESET_VERSION", "")).lower() != "extent_only_keep_v1":
                    raise RuntimeError(
                        "ESA PostReset Keep35 requires ESA_POST_RESET_VERSION='extent_only_keep_v1'."
                    )
                if str(getattr(cfg, "ESA_POST_RESET_MODE", "")).lower() != "extent_only":
                    raise RuntimeError(
                        "ESA PostReset Keep35 requires ESA_POST_RESET_MODE='extent_only'."
                    )
                if bool(getattr(cfg, "ESA_POST_RESET_RAMP", False)):
                    raise RuntimeError("ESA PostReset Keep35 forbids a second post-reset ramp.")
                if abs(float(getattr(cfg, "ESA_POST_RESET_SCALE", 1.0)) - 1.0) > 1e-8:
                    raise RuntimeError("ESA PostReset Keep35 requires ESA_POST_RESET_SCALE=1.0.")
                teacher_only_start = int(getattr(cfg, "DABE_PU_DESPL_TEACHER_ONLY_START", 21))
                if int(getattr(cfg, "ESA_POST_RESET_START_EPOCH", 21)) != teacher_only_start:
                    raise RuntimeError(
                        "ESA post-reset start must equal DABE-PU teacher-only start."
                    )
                if int(getattr(cfg, "ESA_POST_RESET_STOP_EPOCH", 36)) != int(
                    getattr(cfg, "MAX_EPOCH", 35)
                ) + 1:
                    raise RuntimeError(
                        "ESA post-reset stop must be MAX_EPOCH + 1 for Keep35."
                    )
                if int(getattr(cfg, "RAST_STOP_EPOCH", 21)) != teacher_only_start or int(
                    getattr(cfg, "ESA_ASYM_STOP_EPOCH", 21)
                ) != teacher_only_start:
                    raise RuntimeError(
                        "ESA PostReset Keep35 requires pre-reset RAST/ESA to stop at teacher-only start."
                    )
                if not bool(getattr(cfg, "RAST_DISABLE_AFTER_RESET", True)):
                    raise RuntimeError("ESA PostReset Keep35 requires RAST_DISABLE_AFTER_RESET=True.")
                if bool(getattr(cfg, "RAST_POST_RESET_ENABLE", False)) or abs(
                    float(getattr(cfg, "RAST_POST_RESET_SCALE", 0.0))
                ) > 1e-8:
                    raise RuntimeError("ESA PostReset Keep35 cannot enable RAST post-reset routing.")
                forbidden_post_flags = {
                    "ESA_POST_RESET_TOUCH_UNKNOWN": bool(
                        getattr(cfg, "ESA_POST_RESET_TOUCH_UNKNOWN", False)
                    ),
                    "ESA_POST_RESET_USE_CORE_CONFLICT": bool(
                        getattr(cfg, "ESA_POST_RESET_USE_CORE_CONFLICT", False)
                    ),
                    "ESA_POST_RESET_USE_RAST_UNKNOWN": bool(
                        getattr(cfg, "ESA_POST_RESET_USE_RAST_UNKNOWN", False)
                    ),
                }
                enabled_post_flags = [
                    name for name, enabled in forbidden_post_flags.items() if enabled
                ]
                if enabled_post_flags:
                    raise RuntimeError(
                        f"ESA PostReset Keep35 forbids: {enabled_post_flags}."
                    )
                required_multipliers = {
                    "ESA_POST_RESET_EXTENT_TEACHER_FG_MULT": 1.0,
                    "ESA_POST_RESET_EXTENT_TEACHER_BG_FG_LIKE_MULT": 0.25,
                    "ESA_POST_RESET_EXTENT_TEACHER_BG_AMBIG_MULT": 0.50,
                    "ESA_POST_RESET_EXTENT_TEACHER_BG_BG_LIKE_MULT": 1.0,
                }
                for name, expected in required_multipliers.items():
                    if abs(float(getattr(cfg, name, expected)) - expected) > 1e-8:
                        raise RuntimeError(f"{name} must equal {expected} for Keep35.")
                for name in (
                    "ESA_POST_RESET_APPLY_TO_FINAL",
                    "ESA_POST_RESET_APPLY_TO_COARSE_AUX",
                    "ESA_POST_RESET_APPLY_TO_BASE_AUX",
                    "RAST_APPLY_TO_FINAL",
                    "RAST_APPLY_TO_COARSE_AUX",
                    "RAST_APPLY_TO_BASE_AUX",
                ):
                    if not bool(getattr(cfg, name, True)):
                        raise RuntimeError(f"ESA PostReset Keep35 requires {name}=True.")
                if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "dagp_safe" or not bool(
                    getattr(cfg, "USE_NDR_BRANCH", False)
                ) or bool(getattr(cfg, "USE_NDR_V2", False)):
                    raise RuntimeError(
                        "ESA PostReset Keep35 requires the original DAGP-Safe + NDR-v1 head."
                    )
                forbidden_branches = {
                    "USE_PA_DAGP": bool(getattr(cfg, "USE_PA_DAGP", False)),
                    "USE_CSD_V1R": bool(getattr(cfg, "USE_CSD_V1R", False)),
                    "USE_CSD_DECODER": bool(getattr(cfg, "USE_CSD_DECODER", False)),
                    "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
                    "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
                    "USE_CSSD": bool(getattr(cfg, "USE_CSSD", False)),
                    "USE_HR_BFR": bool(getattr(cfg, "USE_HR_BFR", False)),
                    "USE_HBNS_LITE": bool(getattr(cfg, "USE_HBNS_LITE", False)),
                    "USE_EPR_POS": bool(getattr(cfg, "USE_EPR_POS", False)),
                    "USE_PROTO_CONTRAST": bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
                    "USE_MULTI_VIEW_FEATURE": bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)),
                    "USE_VIEW_CONSISTENCY": bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)),
                    "USE_TADR_ROUTER": bool(getattr(cfg, "USE_TADR_ROUTER", False)),
                }
                enabled_branches = [
                    name for name, enabled in forbidden_branches.items() if enabled
                ]
                if enabled_branches:
                    raise RuntimeError(
                        f"ESA PostReset Keep35 cannot be combined with: {enabled_branches}."
                    )
                if not bool(getattr(cfg, "USE_ESA_DIAGNOSTIC", False)):
                    raise RuntimeError(
                        "ESA PostReset Keep35 requires USE_ESA_DIAGNOSTIC=True for region logging."
                    )
            if bool(getattr(cfg, "USE_LCEG", False)):
                if bool(getattr(cfg, "USE_TCE", False)):
                    raise RuntimeError("USE_LCEG=True cannot be combined with USE_TCE=True.")
                if bool(getattr(cfg, "USE_HBNS_LITE", False)) or bool(getattr(cfg, "USE_EPR_POS", False)):
                    raise RuntimeError("USE_LCEG=True cannot be combined with HBNS-lite or EPR-pos.")
                if bool(getattr(cfg, "USE_PROTO_CONTRAST", False)):
                    raise RuntimeError("USE_LCEG=True cannot be combined with proto contrast.")
                if teacher_target_mode != "binary":
                    raise RuntimeError("USE_LCEG=True requires binary teacher target in dabe_pu_despl_sched.")
                if static_target_mode != "soft":
                    raise RuntimeError("USE_LCEG=True requires DABE-PU static target mode 'soft'.")
                if bool(getattr(cfg, "LCEG_USE_UNKNOWN", False)):
                    raise RuntimeError("LCEG-v1 requires LCEG_USE_UNKNOWN=False.")
                if bool(getattr(cfg, "LCEG_APPLY_TO_BASE_AUX", False)):
                    raise RuntimeError("LCEG-v1 must not apply to base aux.")
        elif bool(getattr(cfg, "USE_RAST", False)):
            raise RuntimeError("USE_RAST=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_HBNS_LITE", False)):
            raise RuntimeError("USE_HBNS_LITE=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_EPR_POS", False)):
            raise RuntimeError("USE_EPR_POS=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_ESA_ASYM", False)):
            raise RuntimeError("USE_ESA_ASYM=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        elif bool(getattr(cfg, "USE_LCEG", False)):
            raise RuntimeError("USE_LCEG=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if bool(getattr(cfg, "USE_CSD_DECODER", False)):
            if not use_csd_head(cfg):
                raise RuntimeError("USE_CSD_DECODER=True requires HEAD_TYPE='csd_v1'.")
            forbidden = {
                "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
                "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
                "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
                "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
            }
            enabled = [name for name, value in forbidden.items() if value]
            if enabled:
                raise RuntimeError(f"CSD-v1 clean35 cannot be combined with: {enabled}")
        if bool(getattr(cfg, "USE_CSD_V1R", False)) or use_csd_v1r_head(cfg):
            if not use_csd_v1r_head(cfg):
                raise RuntimeError("USE_CSD_V1R=True requires HEAD_TYPE='dagp_safe_csd_v1r'.")
            forbidden = {
                "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
                "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
                "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
                "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
                "USE_HBNS_LITE": bool(getattr(cfg, "USE_HBNS_LITE", False)),
                "USE_EPR_POS": bool(getattr(cfg, "USE_EPR_POS", False)),
                "USE_PROTO_CONTRAST": bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
                "USE_MULTI_VIEW_FEATURE": bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)),
                "USE_VIEW_CONSISTENCY": bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)),
                "USE_TADR_ROUTER": bool(getattr(cfg, "USE_TADR_ROUTER", False)),
            }
            enabled = [name for name, value in forbidden.items() if value]
            if enabled:
                raise RuntimeError(f"CSD-v1R cannot be combined with: {enabled}")
        if use_hr_bfr(cfg):
            if not use_csd_v1r_head(cfg) or not bool(getattr(cfg, "USE_CSD_V1R", False)):
                raise RuntimeError("USE_HR_BFR=True requires HEAD_TYPE='dagp_safe_csd_v1r' and USE_CSD_V1R=True.")
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
        if use_ndr_v2(cfg):
            if not use_ndr_branch(cfg):
                raise RuntimeError("USE_NDR_V2=True requires USE_NDR_BRANCH=True.")
            if not bool(getattr(cfg, "USE_DABE_PU", False)):
                raise RuntimeError("USE_NDR_V2=True currently requires USE_DABE_PU=True.")
            if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
                raise RuntimeError("USE_NDR_V2=True currently requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
            if bool(getattr(cfg, "NDR_V2_USE_SHAPE_LOWER_BOUND", False)):
                version = str(getattr(cfg, "NDR_VERSION", "")).lower()
                if not version.startswith("v2b"):
                    raise RuntimeError("NDR_V2_USE_SHAPE_LOWER_BOUND=True is reserved for NDR-v2b configs.")
            if float(getattr(cfg, "NDR_V2_SHAPE_LOWER_BOUND_WEIGHT", 0.0)) != 0.0:
                raise RuntimeError("Use NDR_V2_SHAPE_LB_WEIGHT_MAX instead of NDR_V2_SHAPE_LOWER_BOUND_WEIGHT.")
    elif bool(getattr(cfg, "USE_HBNS_LITE", False)):
        raise RuntimeError("USE_HBNS_LITE=True requires USE_DABE_PU=True.")
    elif bool(getattr(cfg, "USE_EPR_POS", False)):
        raise RuntimeError("USE_EPR_POS=True requires USE_DABE_PU=True.")
    elif bool(getattr(cfg, "USE_ESA_ASYM", False)):
        raise RuntimeError("USE_ESA_ASYM=True requires USE_DABE_PU=True.")
    elif use_ndr_v2(cfg):
        raise RuntimeError("USE_NDR_V2=True requires USE_DABE_PU=True.")
    elif use_hr_bfr(cfg):
        raise RuntimeError("USE_HR_BFR=True requires USE_DABE_PU=True.")
    if bool(getattr(cfg, "USE_CSD_DECODER", False)):
        if not use_csd_head(cfg):
            raise RuntimeError("USE_CSD_DECODER=True requires HEAD_TYPE='csd_v1'.")
        forbidden = {
            "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
            "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
            "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
            "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise RuntimeError(f"CSD-v1 clean35 cannot be combined with: {enabled}")
    if bool(getattr(cfg, "USE_CSD_V1R", False)) or use_csd_v1r_head(cfg):
        if not use_csd_v1r_head(cfg):
            raise RuntimeError("USE_CSD_V1R=True requires HEAD_TYPE='dagp_safe_csd_v1r'.")
        forbidden = {
            "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
            "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
            "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
            "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
            "USE_HBNS_LITE": bool(getattr(cfg, "USE_HBNS_LITE", False)),
            "USE_EPR_POS": bool(getattr(cfg, "USE_EPR_POS", False)),
            "USE_PROTO_CONTRAST": bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
            "USE_MULTI_VIEW_FEATURE": bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)),
            "USE_VIEW_CONSISTENCY": bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)),
            "USE_TADR_ROUTER": bool(getattr(cfg, "USE_TADR_ROUTER", False)),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise RuntimeError(f"CSD-v1R cannot be combined with: {enabled}")
    if str(getattr(cfg, "PA_DAGP_VERSION", "")).lower() == "v1_cross_polarity_edge_cut" and not use_pa_dagp(cfg):
        raise RuntimeError("PA_DAGP_VERSION=v1_cross_polarity_edge_cut requires USE_PA_DAGP=True.")
    if use_pa_dagp(cfg):
        if str(getattr(cfg, "PA_DAGP_VERSION", "")).lower() != "v1_cross_polarity_edge_cut":
            raise RuntimeError("USE_PA_DAGP=True requires PA_DAGP_VERSION='v1_cross_polarity_edge_cut'.")
        if not use_csd_v1r_head(cfg) or not bool(getattr(cfg, "USE_CSD_V1R", False)):
            raise RuntimeError("PA-DAGP-v1 requires HEAD_TYPE='dagp_safe_csd_v1r' and USE_CSD_V1R=True.")
        if not bool(getattr(cfg, "USE_DABE_PU", False)) or not bool(getattr(cfg, "USE_RAST", False)):
            raise RuntimeError("PA-DAGP-v1 requires DABE-PU and RAST.")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
            raise RuntimeError("PA-DAGP-v1 requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if get_dabe_pu_despl_teacher_target_mode(cfg) != "binary":
            raise RuntimeError("PA-DAGP-v1 requires the binary EMA teacher target.")
        if get_dabe_pu_despl_static_target_mode(cfg) != "soft":
            raise RuntimeError("PA-DAGP-v1 requires the soft DABE-PU static target.")
        forbidden = {
            "USE_ESA_ASYM": bool(getattr(cfg, "USE_ESA_ASYM", False)),
            "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
            "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
            "USE_HR_BFR": bool(getattr(cfg, "USE_HR_BFR", False)),
            "USE_CSSD": bool(getattr(cfg, "USE_CSSD", False)),
            "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
            "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
            "USE_HBNS_LITE": bool(getattr(cfg, "USE_HBNS_LITE", False)),
            "USE_EPR_POS": bool(getattr(cfg, "USE_EPR_POS", False)),
            "USE_PROTO_CONTRAST": bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
            "USE_MULTI_VIEW_FEATURE": bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)),
            "USE_VIEW_CONSISTENCY": bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)),
            "USE_TADR_ROUTER": bool(getattr(cfg, "USE_TADR_ROUTER", False)),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise RuntimeError(f"PA-DAGP-v1 cannot be combined with: {enabled}")
        if bool(getattr(cfg, "PA_DAGP_USE_SAME_POLARITY_BOOST", False)):
            raise RuntimeError("PA-DAGP-v1 forbids same-polarity edge boost.")
        if not bool(getattr(cfg, "PA_DAGP_RENORMALIZE_EDGE", True)):
            raise RuntimeError("PA-DAGP-v1 requires edge renormalization.")
        for name in (
            "PA_DAGP_DETACH_BASE_PROB",
            "PA_DAGP_DETACH_ANCHORS",
            "PA_DAGP_DETACH_RHO",
        ):
            if not bool(getattr(cfg, name, True)):
                raise RuntimeError(f"PA-DAGP-v1 requires {name}=True.")
        forbidden_losses = (
            "PA_DAGP_USE_EXTENT_LOSS",
            "PA_DAGP_USE_UNKNOWN_LOSS",
            "PA_DAGP_USE_SEPARATION_LOSS",
            "PA_DAGP_USE_AREA_LOSS",
            "PA_DAGP_USE_CONTRASTIVE_LOSS",
        )
        enabled_losses = [name for name in forbidden_losses if bool(getattr(cfg, name, False))]
        if enabled_losses:
            raise RuntimeError(f"PA-DAGP-v1 forbidden losses enabled: {enabled_losses}")
        if int(getattr(cfg, "PA_DAGP_START_EPOCH", 7)) != int(
            getattr(cfg, "DAGP_SAFE_RAMP_START_EPOCH", 7)
        ) or int(getattr(cfg, "PA_DAGP_RAMP_END_EPOCH", 15)) != int(
            getattr(cfg, "DAGP_SAFE_RAMP_END_EPOCH", 15)
        ):
            raise RuntimeError("PA-DAGP-v1 edge schedule must align with DAGP-Safe ramp epochs.")
    if use_cssd(cfg):
        if not bool(getattr(cfg, "CSSD_TRAIN_ONLY", True)):
            raise RuntimeError("CSSD-v1a requires CSSD_TRAIN_ONLY=True.")
        if not use_csd_v1r_head(cfg) or not bool(getattr(cfg, "USE_CSD_V1R", False)):
            raise RuntimeError(
                "CSSD-v1a requires HEAD_TYPE='dagp_safe_csd_v1r' and USE_CSD_V1R=True."
            )
        if not bool(getattr(cfg, "USE_DABE_PU", False)):
            raise RuntimeError("CSSD-v1a requires USE_DABE_PU=True.")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
            raise RuntimeError("CSSD-v1a requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if get_dabe_pu_despl_teacher_target_mode(cfg) != "binary":
            raise RuntimeError("CSSD-v1a requires the normal-view binary EMA teacher target.")
        if get_dabe_pu_despl_static_target_mode(cfg) != "soft":
            raise RuntimeError("CSSD-v1a requires DABE_PU_STATIC_TARGET_MODE='soft'.")
        if not bool(getattr(cfg, "CSSD_USE_SHARED_MODEL", True)):
            raise RuntimeError("CSSD-v1a requires CSSD_USE_SHARED_MODEL=True.")
        if bool(getattr(cfg, "CSSD_USE_SEPARATE_HR_HEAD", False)):
            raise RuntimeError("CSSD-v1a forbids a separate high-resolution head.")
        if int(getattr(cfg, "CSSD_HR_FORWARD_EVERY", 1)) != 1:
            raise RuntimeError("CSSD-v1a requires CSSD_HR_FORWARD_EVERY=1.")
        if not bool(getattr(cfg, "CSSD_SKIP_HR_FORWARD_WHEN_SCALE_ZERO", True)):
            raise RuntimeError("CSSD-v1a requires zero high forwards while cssd_scale=0.")
        expected_values = {
            "CSSD_NORMAL_INPUT_SIZE": 296,
            "CSSD_NORMAL_FEATURE_SIZE": 37,
            "CSSD_HR_INPUT_SIZE": 384,
            "CSSD_HR_FEATURE_SIZE": 48,
            "CSSD_HR_FEATURE_CHANNELS": 384,
        }
        for name, expected in expected_values.items():
            actual = int(getattr(cfg, name, expected))
            if actual != expected:
                raise RuntimeError(f"CSSD-v1a requires {name}={expected}, got {actual}.")
        if int(getattr(cfg, "BATCH_SIZE", 16)) != 16 or int(getattr(cfg, "LOSS_SIZE", 68)) != 68:
            raise RuntimeError("CSSD-v1a locks BATCH_SIZE=16 and LOSS_SIZE=68.")
        if int(cfg.DINO.get("feature_input_size", -1)) != 296 or int(cfg.DINO.get("patch_size", -1)) != 8:
            raise RuntimeError("CSSD-v1a normal cache must remain DINO input296/patch8.")
        if str(getattr(cfg, "CSSD_HR_CACHE_DTYPE", "float32")).lower() != "float32":
            raise RuntimeError("CSSD-v1a requires float32 high-resolution feature cache.")
        if str(getattr(cfg, "CSSD_HR_CACHE_KEY", "feature")) != "feature":
            raise RuntimeError("CSSD-v1a requires CSSD_HR_CACHE_KEY='feature'.")
        if str(getattr(cfg, "CSSD_TRANSFER_REGION", "extent")).lower() != "extent":
            raise RuntimeError("CSSD-v1a supports extent-only transfer.")
        if bool(getattr(cfg, "CSSD_USE_UNKNOWN", False)):
            raise RuntimeError("CSSD-v1a forbids unknown-region transfer.")
        if not bool(getattr(cfg, "CSSD_DETACH_HR_TARGET", True)):
            raise RuntimeError("CSSD-v1a requires detached high-view probability targets.")
        if not bool(getattr(cfg, "CSSD_HR_SUP_USE_ORIGINAL_PIPELINE", True)):
            raise RuntimeError("CSSD-v1a requires the original DABE-PU/teacher/RAST/ESA group for high supervision.")
        if not bool(getattr(cfg, "CSSD_BOUNDARY_USE_SOFT_GRADIENT", True)) or bool(
            getattr(cfg, "CSSD_BOUNDARY_USE_HARD_CONTOUR", False)
        ):
            raise RuntimeError("CSSD-v1a requires soft Sobel boundary consistency and forbids hard contours.")
        forbidden = {
            "USE_HR_BFR": bool(getattr(cfg, "USE_HR_BFR", False)),
            "USE_LCEG": bool(getattr(cfg, "USE_LCEG", False)),
            "USE_TCE": bool(getattr(cfg, "USE_TCE", False)),
            "USE_HBNS_LITE": bool(getattr(cfg, "USE_HBNS_LITE", False)),
            "USE_EPR_POS": bool(getattr(cfg, "USE_EPR_POS", False)),
            "USE_DARE": bool(getattr(cfg, "USE_DARE", False)),
            "USE_PROTO_CONTRAST": bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
            "USE_MULTI_VIEW_FEATURE": bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False)),
            "USE_VIEW_CONSISTENCY": bool(getattr(cfg, "USE_VIEW_CONSISTENCY", False)),
            "USE_NDR_BRANCH": bool(getattr(cfg, "USE_NDR_BRANCH", False)),
            "USE_NDR_V2": bool(getattr(cfg, "USE_NDR_V2", False)),
            "USE_TADR_ROUTER": bool(getattr(cfg, "USE_TADR_ROUTER", False)),
            "USE_GKD_LITE": bool(getattr(cfg, "USE_GKD_LITE", False)),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled or str(getattr(cfg, "GKD_MODE", "off")).lower() != "off":
            raise RuntimeError(f"CSSD-v1a cannot be combined with disabled branches: {enabled or ['GKD_MODE']}.")
    if use_hr_bfr(cfg):
        if not use_csd_v1r_head(cfg) or not bool(getattr(cfg, "USE_CSD_V1R", False)):
            raise RuntimeError("USE_HR_BFR=True requires HEAD_TYPE='dagp_safe_csd_v1r' and USE_CSD_V1R=True.")
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
    if use_cacd(cfg):
        if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "cacd_v1_base" or not bool(
            getattr(cfg, "USE_CACD", False)
        ):
            raise RuntimeError("CACD-v1-Base requires HEAD_TYPE='cacd_v1_base' and USE_CACD=True.")
        if str(getattr(cfg, "CACD_VERSION", "")) != "v1_base_last3_consensus_anchor_context_68":
            raise RuntimeError("Unexpected CACD_VERSION for CACD-v1-Base.")
        expected_ints = {
            "CACD_FEATURE_SIZE": 37,
            "CACD_OUTPUT_SIZE": 68,
            "CACD_DINO_CHANNELS": 384,
            "CACD_DIM": 96,
            "CACD_ANCHOR_CLASSES": 3,
            "CACD_NUM_FG_SLOTS": 4,
            "CACD_NUM_BG_SLOTS": 4,
            "CACD_SLOT_DIM": 96,
        }
        for name, expected in expected_ints.items():
            if int(getattr(cfg, name, expected)) != expected:
                raise RuntimeError(f"CACD-v1-Base requires {name}={expected}.")
        if list(getattr(cfg, "CACD_EXTRA_LAYER_INDICES", [])) != [9, 10] or int(
            getattr(cfg, "CACD_FINAL_LAYER_INDEX", -1)
        ) != 11:
            raise RuntimeError("CACD-v1-Base requires extra layers [9,10] and final layer 11.")
        if str(getattr(cfg, "CACD_EXTRA_FEATURE_DTYPE", "")).lower() != "float32":
            raise RuntimeError("CACD-v1-Base requires float32 extra feature cache.")
        if not bool(getattr(cfg, "CACD_REUSE_EXISTING_FINAL_FEATURE", True)):
            raise RuntimeError("CACD-v1-Base must reuse the existing final F12 cache.")
        required_true = (
            "CACD_USE_CROSS_LAYER_CONSENSUS",
            "CACD_QC_DETACH",
            "CACD_USE_ANCHOR_HEAD",
            "CACD_USE_MULTI_ANCHOR_CONTEXT",
            "CACD_CONTEXT_FINAL_ZERO_INIT",
            "CACD_USE_RGB_DETAIL",
            "CACD_USE_SOBEL_DETAIL",
            "CACD_USE_ANCHOR_LOSS",
            "CACD_ANCHOR_LOSS_ALL_EPOCHS",
            "CACD_KEEP_ORIGINAL_TEACHER_SCHEDULE",
            "CACD_KEEP_ORIGINAL_RESET",
            "USE_RAST",
            "USE_ESA_ASYM",
            "USE_DABE_PU",
        )
        disabled_required = (
            "CACD_USE_HARD_WARMUP",
            "CACD_ANCHOR_USE_EXTENT_LABEL",
            "CACD_ANCHOR_USE_UNKNOWN_LABEL",
            "CACD_USE_136_DECODER",
            "CACD_USE_BOUNDARY_POST_REFINE",
            "CACD_USE_SLOT_DIVERSITY_LOSS",
            "CACD_USE_RELATION_LOSS",
            "CACD_USE_LOCAL_STRUCTURE_LOSS",
            "CACD_USE_VIEW_LOSS",
            "CACD_USE_ADAPTIVE_TEACHER",
            "USE_DAGP_SAFE_HEAD",
            "USE_NDR_BRANCH",
            "USE_NDR_V2",
            "USE_CSD_DECODER",
            "USE_CSD_V1R",
            "USE_PA_DAGP",
            "USE_ESA_BER",
            "USE_HR_BFR",
            "USE_CSSD",
            "USE_LCEG",
            "USE_TCE",
            "USE_HBNS_LITE",
            "USE_EPR_POS",
            "USE_DARE",
            "USE_DABE_OEM",
            "USE_DABE_PU_GROUP_BALANCED_STATIC",
            "USE_TEACHER_CONF_LOSS",
            "USE_TEACHER_SOFT_FULL_LOSS",
            "USE_PROTO_CONTRAST",
            "USE_MULTI_VIEW_FEATURE",
            "USE_VIEW_CONSISTENCY",
            "USE_TADR_ROUTER",
            "USE_GKD_LITE",
        )
        missing_true = [name for name in required_true if not bool(getattr(cfg, name, False))]
        enabled_forbidden = [name for name in disabled_required if bool(getattr(cfg, name, False))]
        if missing_true or enabled_forbidden:
            raise RuntimeError(
                f"CACD-v1-Base guard failed: required_true_missing={missing_true}, "
                f"forbidden_enabled={enabled_forbidden}"
            )
        if str(getattr(cfg, "GKD_MODE", "off")).lower() != "off":
            raise RuntimeError("CACD-v1-Base requires GKD_MODE='off'.")
        if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() != "pu_v11":
            raise RuntimeError("CACD-v1-Base requires DABE-PU v11.")
        if str(getattr(cfg, "TEACHER_FUSION_MODE", "")).lower() != "dabe_pu_despl_sched":
            raise RuntimeError("CACD-v1-Base preserves TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
        if get_dabe_pu_despl_teacher_target_mode(cfg) != "binary" or get_dabe_pu_despl_static_target_mode(cfg) != "soft":
            raise RuntimeError("CACD-v1-Base requires binary teacher and soft static target.")
        if int(getattr(cfg, "FINETUNE_RESET_EPOCH", -1)) != 20 or finetune_reset_timing(cfg) != "after_epoch":
            raise RuntimeError("CACD-v1-Base must preserve epoch20 after-epoch reset.")
        if int(getattr(cfg, "DABE_PU_DESPL_TEACHER_ONLY_START", -1)) != 21:
            raise RuntimeError("CACD-v1-Base must preserve epoch21 teacher-only start.")
        if int(getattr(cfg, "RAST_STOP_EPOCH", -1)) != 21 or int(
            getattr(cfg, "ESA_ASYM_STOP_EPOCH", -1)
        ) != 21 or bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)):
            raise RuntimeError("CACD-v1-Base must preserve pre-reset-only RAST/ESA routing.")
        if int(getattr(cfg, "LOSS_SIZE", -1)) != 68 or int(getattr(cfg, "MAX_EPOCH", -1)) != 35:
            raise RuntimeError("CACD-v1-Base requires LOSS_SIZE=68 and MAX_EPOCH=35.")
    elif str(getattr(cfg, "HEAD_TYPE", "")).lower() == "cacd_v1_base" or bool(
        getattr(cfg, "USE_CACD", False)
    ):
        raise RuntimeError("HEAD_TYPE='cacd_v1_base' and USE_CACD must be enabled together.")

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
        if use_cacd(cfg):
            logger.log("HEAD_TYPE = cacd_v1_base")
            logger.log("USE_CACD = True")
            for field in (
                "CACD_VERSION",
                "CACD_DIM",
                "CACD_EXTRA_LAYER_INDICES",
                "CACD_FINAL_LAYER_INDEX",
                "CACD_EXTRA_FEATURE_CACHE_ROOT",
                "CACD_NUM_FG_SLOTS",
                "CACD_NUM_BG_SLOTS",
                "CACD_ANCHOR_LOSS_WEIGHT",
                "CACD_USE_136_DECODER",
                "CACD_KEEP_ORIGINAL_TEACHER_SCHEDULE",
                "CACD_KEEP_ORIGINAL_RESET",
            ):
                logger.log(f"{field} = {getattr(cfg, field)}")
            for field in (
                "USE_DAGP_SAFE_HEAD",
                "USE_NDR_BRANCH",
                "USE_CSD_V1R",
                "USE_PA_DAGP",
                "USE_RAST",
                "USE_ESA_ASYM",
                "ESA_POST_RESET_ENABLE",
            ):
                logger.log(f"{field} = {bool(getattr(cfg, field, False))}")
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
        logger.log(f"SAVE_EVERY_EPOCH = {bool(getattr(cfg, 'SAVE_EVERY_EPOCH', False))}")
        logger.log(f"SAVE_INTERVAL = {int(getattr(cfg, 'SAVE_INTERVAL', 0))}")
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
            logger.log(f"USE_TEPR_LITE = {bool(getattr(cfg, 'USE_TEPR_LITE', False))}")
            if bool(getattr(cfg, "USE_TEPR_LITE", False)):
                tepr_routing_mode = str(getattr(cfg, "TEPR_ROUTING_MODE", "legacy_v1"))
                logger.log(f"TEPR_ROUTING_MODE = {tepr_routing_mode}")
                for field in (
                    "TEPR_VERSION",
                    "TEPR_START_EPOCH",
                    "TEPR_RAMP_END_EPOCH",
                    "TEPR_STOP_EPOCH",
                    "TEPR_MEMORY_UPDATE_START_EPOCH",
                    "TEPR_MEMORY_UPDATE_END_EPOCH",
                    "TEPR_TEMPORAL_RHO",
                    "TEPR_MIN_HISTORY",
                    "TEPR_MEMORY_DTYPE",
                    "TEPR_USE_PREUPDATE_STATS",
                    "TEPR_VARIANCE_TAU",
                    "TEPR_CONF_GAMMA",
                    "TEPR_CORE_LAMBDA",
                    "TEPR_EXTENT_LAMBDA",
                    "TEPR_MARGIN_TAU",
                    "TEPR_UNKNOWN_WEIGHT_MIN",
                    "TEPR_UNKNOWN_WEIGHT_MAX",
                    "TEPR_EXTENT_TEMPORAL_MIN",
                    "TEPR_WEIGHT_MIN",
                    "TEPR_WEIGHT_MAX",
                    "TEPR_APPLY_TO_FINAL",
                    "TEPR_APPLY_TO_COARSE_AUX",
                    "TEPR_APPLY_TO_BASE_AUX",
                    "TEPR_RESET_MEMORY_AT_FINETUNE_RESET",
                    "TEPR_LOG_INTERVAL_EPOCH",
                    "TEPR_DEBUG_FIRST_BATCH",
                ):
                    logger.log(f"{field} = {getattr(cfg, field)}")
                if tepr_routing_mode.lower() == "state_conditional_asymneg":
                    for field in (
                        "TEPR_TEMPORAL_SCOPE",
                        "TEPR_INSUFFICIENT_HISTORY_MODE",
                        "TEPR_CORE_MODE",
                        "TEPR_CORE_CONFLICT_WEIGHT",
                        "TEPR_USE_TEMPORAL_ON_CORE",
                        "TEPR_EXTENT_FG_WEIGHT",
                        "TEPR_EXTENT_BG_WEIGHT_FLOOR",
                        "TEPR_EXTENT_DINO_LAMBDA",
                        "TEPR_USE_TEMPORAL_ON_EXTENT_FG",
                        "TEPR_USE_TEMPORAL_ON_EXTENT_BG",
                        "TEPR_UNKNOWN_MODE",
                        "TEPR_UNKNOWN_WEIGHT",
                        "TEPR_USE_TEMPORAL_ON_UNKNOWN",
                        "TEPR_OTHER_WEIGHT",
                    ):
                        logger.log(f"{field} = {getattr(cfg, field)}")
                    logger.log(
                        "TEPR ignored_in_v11 = "
                        "['TEPR_UNKNOWN_WEIGHT_MIN', 'TEPR_UNKNOWN_WEIGHT_MAX', "
                        "'TEPR_EXTENT_TEMPORAL_MIN', 'TEPR_CORE_LAMBDA', "
                        "'TEPR_EXTENT_LAMBDA']"
                    )
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
            logger.log(f"ESA_POST_RESET_ENABLE = {bool(getattr(cfg, 'ESA_POST_RESET_ENABLE', False))}")
            logger.log(
                f"ESA_POST_RESET_VERSION = {getattr(cfg, 'ESA_POST_RESET_VERSION', 'off')}"
            )
            logger.log(
                f"ESA_POST_RESET_START_EPOCH = {int(getattr(cfg, 'ESA_POST_RESET_START_EPOCH', 21))}"
            )
            logger.log(
                f"ESA_POST_RESET_STOP_EPOCH = {int(getattr(cfg, 'ESA_POST_RESET_STOP_EPOCH', 36))}"
            )
            logger.log(
                f"ESA_POST_RESET_SCALE = {float(getattr(cfg, 'ESA_POST_RESET_SCALE', 0.0)):.6f}"
            )
            logger.log(
                f"ESA_POST_RESET_RAMP = {bool(getattr(cfg, 'ESA_POST_RESET_RAMP', False))}"
            )
            logger.log(
                f"ESA_POST_RESET_MODE = {getattr(cfg, 'ESA_POST_RESET_MODE', 'off')}"
            )
            logger.log(
                "ESA_POST_RESET_EXTENT_TEACHER_FG_MULT = "
                f"{float(getattr(cfg, 'ESA_POST_RESET_EXTENT_TEACHER_FG_MULT', 1.0)):.6f}"
            )
            logger.log(
                "ESA_POST_RESET_EXTENT_TEACHER_BG_FG_LIKE_MULT = "
                f"{float(getattr(cfg, 'ESA_POST_RESET_EXTENT_TEACHER_BG_FG_LIKE_MULT', 0.25)):.6f}"
            )
            logger.log(
                "ESA_POST_RESET_EXTENT_TEACHER_BG_AMBIG_MULT = "
                f"{float(getattr(cfg, 'ESA_POST_RESET_EXTENT_TEACHER_BG_AMBIG_MULT', 0.50)):.6f}"
            )
            logger.log(
                "ESA_POST_RESET_EXTENT_TEACHER_BG_BG_LIKE_MULT = "
                f"{float(getattr(cfg, 'ESA_POST_RESET_EXTENT_TEACHER_BG_BG_LIKE_MULT', 1.0)):.6f}"
            )
            logger.log(
                f"ESA_POST_RESET_TOUCH_UNKNOWN = {bool(getattr(cfg, 'ESA_POST_RESET_TOUCH_UNKNOWN', False))}"
            )
            logger.log(
                "ESA_POST_RESET_USE_CORE_CONFLICT = "
                f"{bool(getattr(cfg, 'ESA_POST_RESET_USE_CORE_CONFLICT', False))}"
            )
            logger.log(
                "ESA_POST_RESET_USE_RAST_UNKNOWN = "
                f"{bool(getattr(cfg, 'ESA_POST_RESET_USE_RAST_UNKNOWN', False))}"
            )
            logger.log(
                f"ESA_POST_RESET_APPLY_TO_FINAL = {bool(getattr(cfg, 'ESA_POST_RESET_APPLY_TO_FINAL', True))}"
            )
            logger.log(
                "ESA_POST_RESET_APPLY_TO_COARSE_AUX = "
                f"{bool(getattr(cfg, 'ESA_POST_RESET_APPLY_TO_COARSE_AUX', True))}"
            )
            logger.log(
                "ESA_POST_RESET_APPLY_TO_BASE_AUX = "
                f"{bool(getattr(cfg, 'ESA_POST_RESET_APPLY_TO_BASE_AUX', True))}"
            )
            if bool(getattr(cfg, "USE_ESA_BER", False)):
                logger.log(f"USE_ESA_BER = {bool(getattr(cfg, 'USE_ESA_BER', False))}")
                logger.log(f"ESA_BER_VERSION = {getattr(cfg, 'ESA_BER_VERSION', '')}")
                for field in (
                    "ESA_BER_START_EPOCH",
                    "ESA_BER_RAMP_END_EPOCH",
                    "ESA_BER_STOP_EPOCH",
                    "ESA_BER_MIN_PAIRS_PER_IMAGE",
                    "ESA_BER_MAX_PAIRS_PER_IMAGE",
                ):
                    logger.log(f"{field} = {int(getattr(cfg, field))}")
                for field in (
                    "ESA_BER_LAMBDA_MAX",
                    "ESA_BER_RANK_MARGIN",
                    "ESA_BER_RANK_TAU",
                    "ESA_BER_MAX_RATIO_PER_CLASS",
                ):
                    logger.log(f"{field} = {float(getattr(cfg, field)):.8f}")
                for field in (
                    "ESA_BER_APPLY_TO_FINAL_ONLY",
                    "ESA_BER_APPLY_TO_COARSE",
                    "ESA_BER_APPLY_TO_BASE",
                    "ESA_BER_DETACH_CANDIDATES",
                    "ESA_BER_DETACH_TEACHER",
                    "ESA_BER_DETACH_GRAPH_EVIDENCE",
                    "ESA_BER_DETACH_MARGIN",
                    "DAGP_SAFE_RETURN_GRAPH_AUX_FOR_BER",
                ):
                    logger.log(f"{field} = {bool(getattr(cfg, field))}")
                logger.log(
                    f"ESA_BER_GRAPH_WEIGHT_SOURCE = {getattr(cfg, 'ESA_BER_GRAPH_WEIGHT_SOURCE', '')}"
                )
                logger.log(f"USE_NDR_BRANCH = {bool(getattr(cfg, 'USE_NDR_BRANCH', False))}")
                logger.log(f"USE_CSD_V1R = {bool(getattr(cfg, 'USE_CSD_V1R', False))}")
                logger.log(f"USE_PA_DAGP = {bool(getattr(cfg, 'USE_PA_DAGP', False))}")
            logger.log(f"USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
            logger.log(f"TCE_VERSION = {getattr(cfg, 'TCE_VERSION', 'v1_temporal_coverage_expansion')}")
            logger.log(f"TCE_COVER_CACHE_ROOT = {getattr(cfg, 'TCE_COVER_CACHE_ROOT', '')}")
            logger.log(f"TCE_COVER_MODEL = {getattr(cfg, 'TCE_COVER_MODEL', 'student')}")
            logger.log(f"TCE_COVER_EPOCH = {int(getattr(cfg, 'TCE_COVER_EPOCH', 30))}")
            logger.log(f"TCE_START_EPOCH = {int(getattr(cfg, 'TCE_START_EPOCH', 31))}")
            logger.log(f"TCE_RAMP_END_EPOCH = {int(getattr(cfg, 'TCE_RAMP_END_EPOCH', 32))}")
            logger.log(f"TCE_STOP_EPOCH = {int(getattr(cfg, 'TCE_STOP_EPOCH', 36))}")
            logger.log(f"TCE_REGION = {getattr(cfg, 'TCE_REGION', 'extent')}")
            logger.log(f"TCE_USE_UNKNOWN = {bool(getattr(cfg, 'TCE_USE_UNKNOWN', False))}")
            logger.log(f"TCE_USE_DINO_MARGIN = {bool(getattr(cfg, 'TCE_USE_DINO_MARGIN', True))}")
            logger.log(f"TCE_LOST_MARGIN_THRESH = {float(getattr(cfg, 'TCE_LOST_MARGIN_THRESH', 0.05)):.6f}")
            logger.log(f"TCE_NEW_MARGIN_THRESH = {float(getattr(cfg, 'TCE_NEW_MARGIN_THRESH', 0.10)):.6f}")
            logger.log(f"TCE_COVER_CONF_THRESH = {float(getattr(cfg, 'TCE_COVER_CONF_THRESH', 0.60)):.6f}")
            logger.log(f"TCE_CURRENT_BG_CONF_MAX = {float(getattr(cfg, 'TCE_CURRENT_BG_CONF_MAX', 0.90)):.6f}")
            logger.log(f"TCE_NEAR_FG_THRESH = {float(getattr(cfg, 'TCE_NEAR_FG_THRESH', 0.50)):.6f}")
            logger.log(f"TCE_NEAR_FG_RADIUS = {int(getattr(cfg, 'TCE_NEAR_FG_RADIUS', 3))}")
            logger.log(f"TCE_EDGE_Q = {float(getattr(cfg, 'TCE_EDGE_Q', 0.60)):.6f}")
            logger.log(f"TCE_SHRINK_AREA_RATIO = {float(getattr(cfg, 'TCE_SHRINK_AREA_RATIO', 0.90)):.6f}")
            logger.log(f"TCE_SHRINK_EXTENT_RATIO = {float(getattr(cfg, 'TCE_SHRINK_EXTENT_RATIO', 0.85)):.6f}")
            logger.log(f"TCE_BG_CORE_TEACHER_FG_MAX = {float(getattr(cfg, 'TCE_BG_CORE_TEACHER_FG_MAX', 0.015)):.6f}")
            logger.log(f"TCE_LOST_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'TCE_LOST_MAX_RATIO_PER_IMAGE', 0.004)):.6f}")
            logger.log(f"TCE_NEW_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'TCE_NEW_MAX_RATIO_PER_IMAGE', 0.002)):.6f}")
            logger.log(f"TCE_TOTAL_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'TCE_TOTAL_MAX_RATIO_PER_IMAGE', 0.005)):.6f}")
            logger.log(f"TCE_MIN_PIXELS_PER_IMAGE = {int(getattr(cfg, 'TCE_MIN_PIXELS_PER_IMAGE', 4))}")
            logger.log(f"TCE_LAMBDA_MAX = {float(getattr(cfg, 'TCE_LAMBDA_MAX', 0.010)):.8f}")
            logger.log(f"TCE_LOST_TARGET_FLOOR = {float(getattr(cfg, 'TCE_LOST_TARGET_FLOOR', 0.45)):.6f}")
            logger.log(f"TCE_NEW_TARGET_FLOOR = {float(getattr(cfg, 'TCE_NEW_TARGET_FLOOR', 0.35)):.6f}")
            logger.log(f"TCE_LOST_BG_TEACHER_MULT = {float(getattr(cfg, 'TCE_LOST_BG_TEACHER_MULT', 0.60)):.6f}")
            logger.log(f"TCE_NEW_BG_TEACHER_MULT = {float(getattr(cfg, 'TCE_NEW_BG_TEACHER_MULT', 0.75)):.6f}")
            logger.log(f"TCE_APPLY_TO_FINAL = {bool(getattr(cfg, 'TCE_APPLY_TO_FINAL', True))}")
            logger.log(f"TCE_APPLY_TO_COARSE_AUX = {bool(getattr(cfg, 'TCE_APPLY_TO_COARSE_AUX', False))}")
            logger.log(f"TCE_APPLY_TO_BASE_AUX = {bool(getattr(cfg, 'TCE_APPLY_TO_BASE_AUX', False))}")
            logger.log(f"TCE_WEIGHTED_NORMALIZE = {bool(getattr(cfg, 'TCE_WEIGHTED_NORMALIZE', True))}")
            logger.log(f"USE_TCE_DIAGNOSTIC = {bool(getattr(cfg, 'USE_TCE_DIAGNOSTIC', False))}")
            logger.log(f"TCE_DIAG_LOG_INTERVAL_EPOCH = {int(getattr(cfg, 'TCE_DIAG_LOG_INTERVAL_EPOCH', 1))}")
            logger.log(f"USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
            logger.log(f"LCEG_VERSION = {getattr(cfg, 'LCEG_VERSION', 'v1_late_core_extent_guard')}")
            logger.log(f"LCEG_COVER_CACHE_ROOT = {getattr(cfg, 'LCEG_COVER_CACHE_ROOT', '')}")
            logger.log(f"LCEG_COVER_MODEL = {getattr(cfg, 'LCEG_COVER_MODEL', 'student')}")
            logger.log(f"LCEG_COVER_EPOCH = {int(getattr(cfg, 'LCEG_COVER_EPOCH', 25))}")
            logger.log(f"LCEG_START_EPOCH = {int(getattr(cfg, 'LCEG_START_EPOCH', 26))}")
            logger.log(f"LCEG_RAMP_END_EPOCH = {int(getattr(cfg, 'LCEG_RAMP_END_EPOCH', 28))}")
            logger.log(f"LCEG_STOP_EPOCH = {int(getattr(cfg, 'LCEG_STOP_EPOCH', 36))}")
            logger.log(f"LCEG_USE_CORE_GUARD = {bool(getattr(cfg, 'LCEG_USE_CORE_GUARD', True))}")
            logger.log(f"LCEG_USE_LOST_EXTENT = {bool(getattr(cfg, 'LCEG_USE_LOST_EXTENT', True))}")
            logger.log(f"LCEG_USE_NEW_BOUNDARY = {bool(getattr(cfg, 'LCEG_USE_NEW_BOUNDARY', True))}")
            logger.log(f"LCEG_USE_UNKNOWN = {bool(getattr(cfg, 'LCEG_USE_UNKNOWN', False))}")
            logger.log(f"LCEG_USE_DINO_MARGIN = {bool(getattr(cfg, 'LCEG_USE_DINO_MARGIN', True))}")
            logger.log(f"LCEG_LOST_MARGIN_THRESH = {float(getattr(cfg, 'LCEG_LOST_MARGIN_THRESH', -0.02)):.6f}")
            logger.log(f"LCEG_NEW_MARGIN_THRESH = {float(getattr(cfg, 'LCEG_NEW_MARGIN_THRESH', 0.05)):.6f}")
            logger.log(f"LCEG_LOST_COVER_PROB_THRESH = {float(getattr(cfg, 'LCEG_LOST_COVER_PROB_THRESH', 0.40)):.6f}")
            logger.log(f"LCEG_NEW_COVER_PROB_MAX = {float(getattr(cfg, 'LCEG_NEW_COVER_PROB_MAX', 0.40)):.6f}")
            logger.log(f"LCEG_CURRENT_BG_CONF_MAX_CORE = {float(getattr(cfg, 'LCEG_CURRENT_BG_CONF_MAX_CORE', 0.98)):.6f}")
            logger.log(f"LCEG_CURRENT_BG_CONF_MAX_EXTENT = {float(getattr(cfg, 'LCEG_CURRENT_BG_CONF_MAX_EXTENT', 0.95)):.6f}")
            logger.log(f"LCEG_CORE_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'LCEG_CORE_MAX_RATIO_PER_IMAGE', 0.010)):.6f}")
            logger.log(f"LCEG_LOST_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'LCEG_LOST_MAX_RATIO_PER_IMAGE', 0.010)):.6f}")
            logger.log(f"LCEG_NEW_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'LCEG_NEW_MAX_RATIO_PER_IMAGE', 0.003)):.6f}")
            logger.log(f"LCEG_TOTAL_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'LCEG_TOTAL_MAX_RATIO_PER_IMAGE', 0.020)):.6f}")
            logger.log(f"LCEG_MIN_PIXELS_PER_IMAGE = {int(getattr(cfg, 'LCEG_MIN_PIXELS_PER_IMAGE', 4))}")
            logger.log(f"LCEG_CORE_TARGET_FLOOR = {float(getattr(cfg, 'LCEG_CORE_TARGET_FLOOR', 0.55)):.6f}")
            logger.log(f"LCEG_LOST_TARGET_FLOOR = {float(getattr(cfg, 'LCEG_LOST_TARGET_FLOOR', 0.45)):.6f}")
            logger.log(f"LCEG_NEW_TARGET_FLOOR = {float(getattr(cfg, 'LCEG_NEW_TARGET_FLOOR', 0.35)):.6f}")
            logger.log(f"LCEG_LAMBDA_MAX = {float(getattr(cfg, 'LCEG_LAMBDA_MAX', 0.020)):.8f}")
            logger.log(f"LCEG_APPLY_TO_FINAL = {bool(getattr(cfg, 'LCEG_APPLY_TO_FINAL', True))}")
            logger.log(f"LCEG_APPLY_TO_COARSE_AUX = {bool(getattr(cfg, 'LCEG_APPLY_TO_COARSE_AUX', True))}")
            logger.log(f"LCEG_APPLY_TO_BASE_AUX = {bool(getattr(cfg, 'LCEG_APPLY_TO_BASE_AUX', False))}")
            logger.log(f"LCEG_COARSE_LOSS_WEIGHT = {float(getattr(cfg, 'LCEG_COARSE_LOSS_WEIGHT', 0.50)):.6f}")
            logger.log(f"USE_LCEG_DIAGNOSTIC = {bool(getattr(cfg, 'USE_LCEG_DIAGNOSTIC', False))}")
            logger.log(f"LCEG_DIAG_LOG_INTERVAL_EPOCH = {int(getattr(cfg, 'LCEG_DIAG_LOG_INTERVAL_EPOCH', 1))}")
            if use_csd_head(cfg) or bool(getattr(cfg, "USE_CSD_DECODER", False)):
                logger.log(f"HEAD_TYPE = {getattr(cfg, 'HEAD_TYPE', 'simple')}")
                logger.log(f"USE_CSD_DECODER = {bool(getattr(cfg, 'USE_CSD_DECODER', False))}")
                logger.log(f"CSD_VERSION = {getattr(cfg, 'CSD_VERSION', 'v1')}")
                logger.log(f"CSD_SEM_DIM = {int(getattr(cfg, 'CSD_SEM_DIM', 64))}")
                logger.log(f"CSD_DETAIL_DIM = {int(getattr(cfg, 'CSD_DETAIL_DIM', 32))}")
                logger.log(f"CSD_FUSION_DIM = {int(getattr(cfg, 'CSD_FUSION_DIM', 64))}")
                logger.log(f"CSD_USE_DAGP_SEMANTIC = {bool(getattr(cfg, 'CSD_USE_DAGP_SEMANTIC', True))}")
                logger.log(f"CSD_DAGP_TOPK = {int(getattr(cfg, 'CSD_DAGP_TOPK', getattr(cfg, 'DAGP_SAFE_TOPK', 12)))}")
                logger.log(f"CSD_DAGP_TAU = {float(getattr(cfg, 'CSD_DAGP_TAU', getattr(cfg, 'DAGP_SAFE_TAU', 0.07))):.6f}")
                logger.log(f"CSD_DAGP_ALPHA_MAX = {float(getattr(cfg, 'CSD_DAGP_ALPHA_MAX', getattr(cfg, 'DAGP_SAFE_ALPHA_MAX', 0.05))):.6f}")
                logger.log(f"CSD_DAGP_GAMMA_MAX = {float(getattr(cfg, 'CSD_DAGP_GAMMA_MAX', getattr(cfg, 'DAGP_SAFE_GAMMA_MAX', 0.03))):.6f}")
                logger.log(f"CSD_WARMUP_EPOCH = {int(getattr(cfg, 'CSD_WARMUP_EPOCH', 6))}")
                logger.log(f"CSD_RAMP_START_EPOCH = {int(getattr(cfg, 'CSD_RAMP_START_EPOCH', 7))}")
                logger.log(f"CSD_RAMP_END_EPOCH = {int(getattr(cfg, 'CSD_RAMP_END_EPOCH', 15))}")
                logger.log(f"CSD_BETA_MAX = {float(getattr(cfg, 'CSD_BETA_MAX', 0.10)):.6f}")
                logger.log(f"CSD_RESIDUAL_CLIP = {float(getattr(cfg, 'CSD_RESIDUAL_CLIP', 2.0)):.6f}")
                logger.log(f"CSD_USE_COARSE_AUX = {bool(getattr(cfg, 'CSD_USE_COARSE_AUX', False))}")
                logger.log(f"LAMBDA_CSD_COARSE_AUX = {float(getattr(cfg, 'LAMBDA_CSD_COARSE_AUX', 0.5)):.6f}")
                logger.log(f"CSD_USE_BG_DETAIL_LOCK = {bool(getattr(cfg, 'CSD_USE_BG_DETAIL_LOCK', False))}")
                logger.log(f"CSD_BG_DETAIL_LOCK_WEIGHT = {float(getattr(cfg, 'CSD_BG_DETAIL_LOCK_WEIGHT', 0.005)):.8f}")
                logger.log(f"CSD_USE_BOUNDARY_AUX = {bool(getattr(cfg, 'CSD_USE_BOUNDARY_AUX', False))}")
                logger.log(f"CSD_BOUNDARY_AUX_WEIGHT_MAX = {float(getattr(cfg, 'CSD_BOUNDARY_AUX_WEIGHT_MAX', 0.020)):.8f}")
                logger.log(f"CSD_BOUNDARY_AUX_START_EPOCH = {int(getattr(cfg, 'CSD_BOUNDARY_AUX_START_EPOCH', 7))}")
                logger.log(f"CSD_BOUNDARY_AUX_RAMP_END_EPOCH = {int(getattr(cfg, 'CSD_BOUNDARY_AUX_RAMP_END_EPOCH', 15))}")
                logger.log(f"CSD_BOUNDARY_AUX_STOP_EPOCH = {int(getattr(cfg, 'CSD_BOUNDARY_AUX_STOP_EPOCH', 21))}")
                logger.log(f"USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
                logger.log(f"USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
                logger.log(f"USE_NDR_BRANCH = {bool(getattr(cfg, 'USE_NDR_BRANCH', False))}")
                logger.log(f"USE_NDR_V2 = {bool(getattr(cfg, 'USE_NDR_V2', False))}")
            if use_csd_v1r_head(cfg) or bool(getattr(cfg, "USE_CSD_V1R", False)):
                logger.log(f"HEAD_TYPE = {getattr(cfg, 'HEAD_TYPE', 'simple')}")
                logger.log(f"USE_DAGP_SAFE_HEAD = {bool(getattr(cfg, 'USE_DAGP_SAFE_HEAD', True))}")
                logger.log(f"USE_CSD_V1R = {bool(getattr(cfg, 'USE_CSD_V1R', False))}")
                logger.log(f"CSD_V1R_VERSION = {getattr(cfg, 'CSD_V1R_VERSION', 'v1r')}")
                logger.log(f"CSD_V1R_SEM_DIM = {int(getattr(cfg, 'CSD_V1R_SEM_DIM', 64))}")
                logger.log(f"CSD_V1R_DETAIL_DIM = {int(getattr(cfg, 'CSD_V1R_DETAIL_DIM', 32))}")
                logger.log(f"CSD_V1R_FUSION_DIM = {int(getattr(cfg, 'CSD_V1R_FUSION_DIM', 64))}")
                logger.log(f"CSD_V1R_WARMUP_EPOCH = {int(getattr(cfg, 'CSD_V1R_WARMUP_EPOCH', 6))}")
                logger.log(f"CSD_V1R_RAMP_START_EPOCH = {int(getattr(cfg, 'CSD_V1R_RAMP_START_EPOCH', 7))}")
                logger.log(f"CSD_V1R_RAMP_END_EPOCH = {int(getattr(cfg, 'CSD_V1R_RAMP_END_EPOCH', 15))}")
                logger.log(f"CSD_V1R_BETA_MAX = {float(getattr(cfg, 'CSD_V1R_BETA_MAX', 0.05)):.6f}")
                logger.log(f"CSD_V1R_RESIDUAL_CLIP = {float(getattr(cfg, 'CSD_V1R_RESIDUAL_CLIP', 2.0)):.6f}")
                logger.log(f"CSD_V1R_USE_COARSE_AUX = {bool(getattr(cfg, 'CSD_V1R_USE_COARSE_AUX', True))}")
                logger.log(f"LAMBDA_CSD_V1R_COARSE_AUX = {float(getattr(cfg, 'LAMBDA_CSD_V1R_COARSE_AUX', 0.5)):.6f}")
                logger.log(f"CSD_V1R_USE_BG_DETAIL_LOCK = {bool(getattr(cfg, 'CSD_V1R_USE_BG_DETAIL_LOCK', True))}")
                logger.log(f"CSD_V1R_BG_DETAIL_LOCK_WEIGHT = {float(getattr(cfg, 'CSD_V1R_BG_DETAIL_LOCK_WEIGHT', 0.005)):.8f}")
                logger.log(f"CSD_V1R_BG_SUPPRESS_STRENGTH = {float(getattr(cfg, 'CSD_V1R_BG_SUPPRESS_STRENGTH', 0.70)):.6f}")
                logger.log(f"USE_NDR_BRANCH = {bool(getattr(cfg, 'USE_NDR_BRANCH', False))}")
                logger.log(f"USE_NDR_V2 = {bool(getattr(cfg, 'USE_NDR_V2', False))}")
                logger.log(f"USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
                logger.log(f"USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
                logger.log(f"USE_HBNS_LITE = {bool(getattr(cfg, 'USE_HBNS_LITE', False))}")
                logger.log(f"USE_EPR_POS = {bool(getattr(cfg, 'USE_EPR_POS', False))}")
                logger.log(f"USE_HR_BFR = {bool(getattr(cfg, 'USE_HR_BFR', False))}")
                if use_hr_bfr(cfg):
                    logger.log(f"HR_BFR_VERSION = {getattr(cfg, 'HR_BFR_VERSION', 'v1_narrow_band_136')}")
                    logger.log(f"HR_BFR_SIZE = {int(getattr(cfg, 'HR_BFR_SIZE', 136))}")
                    logger.log(f"HR_BFR_USE_HR_LOGITS_FOR_EVAL = {bool(getattr(cfg, 'HR_BFR_USE_HR_LOGITS_FOR_EVAL', True))}")
                    logger.log(f"HR_BFR_SEM_DIM = {int(getattr(cfg, 'HR_BFR_SEM_DIM', 32))}")
                    logger.log(f"HR_BFR_DETAIL_DIM = {int(getattr(cfg, 'HR_BFR_DETAIL_DIM', 32))}")
                    logger.log(f"HR_BFR_HIDDEN_DIM = {int(getattr(cfg, 'HR_BFR_HIDDEN_DIM', 32))}")
                    logger.log(f"HR_BFR_BETA_MAX = {float(getattr(cfg, 'HR_BFR_BETA_MAX', 0.05)):.6f}")
                    logger.log(f"HR_BFR_RESIDUAL_CLIP = {float(getattr(cfg, 'HR_BFR_RESIDUAL_CLIP', 2.0)):.6f}")
                    logger.log(f"HR_BFR_WARMUP_EPOCH = {int(getattr(cfg, 'HR_BFR_WARMUP_EPOCH', 6))}")
                    logger.log(f"HR_BFR_RAMP_START_EPOCH = {int(getattr(cfg, 'HR_BFR_RAMP_START_EPOCH', 7))}")
                    logger.log(f"HR_BFR_RAMP_END_EPOCH = {int(getattr(cfg, 'HR_BFR_RAMP_END_EPOCH', 15))}")
                    logger.log(f"HR_BFR_BOUNDARY_THRESH = {float(getattr(cfg, 'HR_BFR_BOUNDARY_THRESH', 0.5)):.6f}")
                    logger.log(f"HR_BFR_BOUNDARY_RADIUS_68 = {int(getattr(cfg, 'HR_BFR_BOUNDARY_RADIUS_68', 2))}")
                    logger.log(f"HR_BFR_BOUNDARY_DILATE_136 = {int(getattr(cfg, 'HR_BFR_BOUNDARY_DILATE_136', 2))}")
                    logger.log(f"HR_BFR_MAX_BAND_RATIO = {float(getattr(cfg, 'HR_BFR_MAX_BAND_RATIO', 0.35)):.6f}")
                    logger.log(f"HR_BFR_BAND_BCE_WEIGHT_MAX = {float(getattr(cfg, 'HR_BFR_BAND_BCE_WEIGHT_MAX', 0.05)):.8f}")
                    logger.log(f"HR_BFR_OUTBAND_ANCHOR_WEIGHT_MAX = {float(getattr(cfg, 'HR_BFR_OUTBAND_ANCHOR_WEIGHT_MAX', 0.10)):.8f}")
                    logger.log(f"HR_BFR_BG_PROB_LOCK_WEIGHT_MAX = {float(getattr(cfg, 'HR_BFR_BG_PROB_LOCK_WEIGHT_MAX', 0.02)):.8f}")
                    logger.log(f"HR_BFR_AREA_NEUTRAL_WEIGHT_MAX = {float(getattr(cfg, 'HR_BFR_AREA_NEUTRAL_WEIGHT_MAX', 0.02)):.8f}")
                    logger.log(f"HR_BFR_EDGE_ALIGN_WEIGHT_MAX = {float(getattr(cfg, 'HR_BFR_EDGE_ALIGN_WEIGHT_MAX', 0.01)):.8f}")
            if use_cssd(cfg):
                logger.log(f"USE_CSSD = {bool(getattr(cfg, 'USE_CSSD', False))}")
                cssd_log_fields = (
                    "CSSD_VERSION",
                    "CSSD_TRAIN_ONLY",
                    "CSSD_USE_SHARED_MODEL",
                    "CSSD_USE_SEPARATE_HR_HEAD",
                    "CSSD_NORMAL_INPUT_SIZE",
                    "CSSD_NORMAL_FEATURE_SIZE",
                    "CSSD_HR_INPUT_SIZE",
                    "CSSD_HR_FEATURE_SIZE",
                    "CSSD_HR_FEATURE_CHANNELS",
                    "CSSD_HR_CACHE_ROOT",
                    "CSSD_HR_CACHE_DTYPE",
                    "CSSD_HR_FEATURE_FIELD",
                    "CSSD_WARMUP_EPOCH",
                    "CSSD_START_EPOCH",
                    "CSSD_RAMP_END_EPOCH",
                    "CSSD_STOP_EPOCH",
                    "CSSD_USE_HR_SUPERVISED_LOSS",
                    "CSSD_HR_SUP_WEIGHT_MAX",
                    "CSSD_HR_SUP_USE_ORIGINAL_PIPELINE",
                    "CSSD_HR_SUP_APPLY_TO_FINAL",
                    "CSSD_HR_SUP_APPLY_TO_COARSE",
                    "CSSD_HR_SUP_APPLY_TO_BASE",
                    "CSSD_USE_PRED_DISTILL",
                    "CSSD_PRED_WEIGHT_MAX",
                    "CSSD_CORE_LOSS_MULT",
                    "CSSD_TRANSFER_LOSS_MULT",
                    "CSSD_CORE_CONF_THRESH",
                    "CSSD_TRANSFER_CONF_THRESH",
                    "CSSD_CONF_ADV_MARGIN",
                    "CSSD_MIN_PROB_DIFF",
                    "CSSD_TRANSFER_REGION",
                    "CSSD_USE_UNKNOWN",
                    "CSSD_TRANSFER_MAX_RATIO_PER_IMAGE",
                    "CSSD_TRANSFER_MIN_PIXELS_PER_IMAGE",
                    "CSSD_USE_BOUNDARY_CONSISTENCY",
                    "CSSD_BOUNDARY_WEIGHT_MAX",
                    "CSSD_BOUNDARY_TEACHER_Q",
                    "CSSD_BOUNDARY_IMAGE_EDGE_Q",
                    "CSSD_BOUNDARY_MAX_RATIO_PER_IMAGE",
                    "CSSD_BOUNDARY_MIN_PIXELS_PER_IMAGE",
                    "CSSD_HR_MICROBATCH",
                    "CSSD_HR_FORWARD_EVERY",
                    "CSSD_SKIP_HR_FORWARD_WHEN_SCALE_ZERO",
                    "CSSD_STRICT_SINGLE_VIEW_EVAL",
                )
                logger.log("[CSSD-v1a] privileged_view=train_only | eval_view=normal_37_only")
                for field in cssd_log_fields:
                    logger.log(f"{field} = {getattr(cfg, field)}")
                logger.log(f"USE_HR_BFR = {bool(getattr(cfg, 'USE_HR_BFR', False))}")
                logger.log(f"USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
                logger.log(f"USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
                logger.log(f"USE_HBNS_LITE = {bool(getattr(cfg, 'USE_HBNS_LITE', False))}")
                logger.log(f"USE_EPR_POS = {bool(getattr(cfg, 'USE_EPR_POS', False))}")
                logger.log(f"USE_PROTO_CONTRAST = {bool(getattr(cfg, 'USE_PROTO_CONTRAST', False))}")
                logger.log(f"USE_MULTI_VIEW_FEATURE = {bool(getattr(cfg, 'USE_MULTI_VIEW_FEATURE', False))}")
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
        logger.log(f"USE_PA_DAGP = {use_pa_dagp(cfg)}")
        if use_pa_dagp(cfg):
            logger.log(f"PA_DAGP_VERSION = {getattr(cfg, 'PA_DAGP_VERSION', '')}")
            logger.log(f"PA_DAGP_POL_DIM = {int(getattr(cfg, 'PA_DAGP_POL_DIM', 32))}")
            logger.log(f"PA_DAGP_POL_GN_GROUPS = {int(getattr(cfg, 'PA_DAGP_POL_GN_GROUPS', 4))}")
            logger.log(f"PA_DAGP_POL_ACT = {getattr(cfg, 'PA_DAGP_POL_ACT', 'gelu')}")
            logger.log(
                f"PA_DAGP_ANCHOR_WEIGHT_POWER = {float(getattr(cfg, 'PA_DAGP_ANCHOR_WEIGHT_POWER', 2.0)):.6f}"
            )
            logger.log(f"PA_DAGP_ANCHOR_EPS = {float(getattr(cfg, 'PA_DAGP_ANCHOR_EPS', 1e-6)):.8f}")
            logger.log(f"PA_DAGP_DETACH_BASE_PROB = {bool(getattr(cfg, 'PA_DAGP_DETACH_BASE_PROB', True))}")
            logger.log(f"PA_DAGP_DETACH_ANCHORS = {bool(getattr(cfg, 'PA_DAGP_DETACH_ANCHORS', True))}")
            logger.log(f"PA_DAGP_DETACH_RHO = {bool(getattr(cfg, 'PA_DAGP_DETACH_RHO', True))}")
            logger.log(f"PA_DAGP_USE_CALIB_HEAD = {bool(getattr(cfg, 'PA_DAGP_USE_CALIB_HEAD', True))}")
            logger.log(f"PA_DAGP_CALIB_HIDDEN = {int(getattr(cfg, 'PA_DAGP_CALIB_HIDDEN', 32))}")
            logger.log(f"PA_DAGP_CALIB_ZERO_INIT = {bool(getattr(cfg, 'PA_DAGP_CALIB_ZERO_INIT', True))}")
            logger.log(f"PA_DAGP_POLARITY_TAU = {float(getattr(cfg, 'PA_DAGP_POLARITY_TAU', 0.50)):.6f}")
            logger.log(f"PA_DAGP_EDGE_CUT_MAX = {float(getattr(cfg, 'PA_DAGP_EDGE_CUT_MAX', 0.50)):.6f}")
            logger.log(f"PA_DAGP_EDGE_GATE_MIN = {float(getattr(cfg, 'PA_DAGP_EDGE_GATE_MIN', 0.50)):.6f}")
            logger.log(
                f"PA_DAGP_USE_SAME_POLARITY_BOOST = {bool(getattr(cfg, 'PA_DAGP_USE_SAME_POLARITY_BOOST', False))}"
            )
            logger.log(f"PA_DAGP_RENORMALIZE_EDGE = {bool(getattr(cfg, 'PA_DAGP_RENORMALIZE_EDGE', True))}")
            logger.log(f"PA_DAGP_START_EPOCH = {int(getattr(cfg, 'PA_DAGP_START_EPOCH', 7))}")
            logger.log(f"PA_DAGP_RAMP_END_EPOCH = {int(getattr(cfg, 'PA_DAGP_RAMP_END_EPOCH', 15))}")
            logger.log(f"PA_DAGP_EDGE_STOP_EPOCH = {int(getattr(cfg, 'PA_DAGP_EDGE_STOP_EPOCH', 36))}")
            logger.log(f"PA_DAGP_USE_AUX_LOSS = {bool(getattr(cfg, 'PA_DAGP_USE_AUX_LOSS', True))}")
            logger.log(
                f"PA_DAGP_AUX_LOSS_WEIGHT_MAX = {float(getattr(cfg, 'PA_DAGP_AUX_LOSS_WEIGHT_MAX', 0.020)):.8f}"
            )
            logger.log(f"PA_DAGP_AUX_LOSS_STOP_EPOCH = {int(getattr(cfg, 'PA_DAGP_AUX_LOSS_STOP_EPOCH', 21))}")
            logger.log(f"PA_DAGP_CORE_MARGIN = {float(getattr(cfg, 'PA_DAGP_CORE_MARGIN', 0.50)):.6f}")
            logger.log(f"PA_DAGP_HARD_MARGIN = {float(getattr(cfg, 'PA_DAGP_HARD_MARGIN', 0.75)):.6f}")
            logger.log(f"PA_DAGP_HARD_LOSS_MULT = {float(getattr(cfg, 'PA_DAGP_HARD_LOSS_MULT', 0.50)):.6f}")
            logger.log(f"USE_PA_DAGP_DIAGNOSTIC = {bool(getattr(cfg, 'USE_PA_DAGP_DIAGNOSTIC', True))}")
            for branch_name in (
                "USE_ESA_ASYM",
                "USE_RAST",
                "USE_CSD_V1R",
                "USE_LCEG",
                "USE_TCE",
                "USE_HR_BFR",
                "USE_CSSD",
                "USE_NDR_BRANCH",
                "USE_NDR_V2",
            ):
                logger.log(f"{branch_name} = {bool(getattr(cfg, branch_name, False))}")
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
            logger.log(f"USE_NDR_V2 = {bool(getattr(cfg, 'USE_NDR_V2', False))}")
            logger.log(f"NDR_VERSION = {getattr(cfg, 'NDR_VERSION', 'v1')}")
            logger.log(f"NDR_V2_USE_SHAPE_GATE = {bool(getattr(cfg, 'NDR_V2_USE_SHAPE_GATE', False))}")
            logger.log(f"NDR_V2_SHAPE_GATE_MODE = {getattr(cfg, 'NDR_V2_SHAPE_GATE_MODE', 'soft_boundary_edge_boost')}")
            logger.log(f"NDR_V2_BOUNDARY_SOURCE = {getattr(cfg, 'NDR_V2_BOUNDARY_SOURCE', 'coarse_prob')}")
            logger.log(f"NDR_V2_BOUNDARY_RADIUS = {int(getattr(cfg, 'NDR_V2_BOUNDARY_RADIUS', 2))}")
            logger.log(f"NDR_V2_BOUNDARY_DETACH = {bool(getattr(cfg, 'NDR_V2_BOUNDARY_DETACH', True))}")
            logger.log(f"NDR_V2_BOUNDARY_SOFT = {bool(getattr(cfg, 'NDR_V2_BOUNDARY_SOFT', True))}")
            logger.log(f"NDR_V2_SHAPE_ALPHA_MAX = {float(getattr(cfg, 'NDR_V2_SHAPE_ALPHA_MAX', 0.20)):.6f}")
            logger.log(f"NDR_V2_SHAPE_EDGE_MIX = {float(getattr(cfg, 'NDR_V2_SHAPE_EDGE_MIX', 0.50)):.6f}")
            logger.log(f"NDR_V2_SHAPE_UNCERT_MIX = {float(getattr(cfg, 'NDR_V2_SHAPE_UNCERT_MIX', 0.50)):.6f}")
            logger.log(f"NDR_V2_GATE_COMBINE = {getattr(cfg, 'NDR_V2_GATE_COMBINE', 'add_clamp')}")
            logger.log(f"NDR_V2_USE_SHAPE_LOWER_BOUND = {bool(getattr(cfg, 'NDR_V2_USE_SHAPE_LOWER_BOUND', False))}")
            logger.log(f"NDR_V2_SHAPE_LB_START_EPOCH = {int(getattr(cfg, 'NDR_V2_SHAPE_LB_START_EPOCH', 36))}")
            logger.log(f"NDR_V2_SHAPE_LB_RAMP_END_EPOCH = {int(getattr(cfg, 'NDR_V2_SHAPE_LB_RAMP_END_EPOCH', 38))}")
            logger.log(f"NDR_V2_SHAPE_LB_STOP_EPOCH = {int(getattr(cfg, 'NDR_V2_SHAPE_LB_STOP_EPOCH', 46))}")
            logger.log(f"NDR_V2_SHAPE_LB_WEIGHT_MAX = {float(getattr(cfg, 'NDR_V2_SHAPE_LB_WEIGHT_MAX', 0.0)):.8f}")
            logger.log(f"NDR_V2_SHAPE_LB_FLOOR = {float(getattr(cfg, 'NDR_V2_SHAPE_LB_FLOOR', 0.35)):.6f}")
            logger.log(f"NDR_V2_SHAPE_LB_MAX_RATIO_PER_IMAGE = {float(getattr(cfg, 'NDR_V2_SHAPE_LB_MAX_RATIO_PER_IMAGE', 0.005)):.6f}")
            logger.log(f"NDR_V2_SHAPE_LB_WEIGHTED_NORMALIZE = {bool(getattr(cfg, 'NDR_V2_SHAPE_LB_WEIGHTED_NORMALIZE', True))}")
            logger.log(f"NDR_V2_SHAPE_LB_USE_EXTENT = {bool(getattr(cfg, 'NDR_V2_SHAPE_LB_USE_EXTENT', True))}")
            logger.log(f"NDR_V2_SHAPE_LB_USE_UNKNOWN = {bool(getattr(cfg, 'NDR_V2_SHAPE_LB_USE_UNKNOWN', False))}")
            logger.log(f"NDR_V2_SHAPE_LB_REQUIRE_TEACHER_BG = {bool(getattr(cfg, 'NDR_V2_SHAPE_LB_REQUIRE_TEACHER_BG', True))}")
            logger.log(f"NDR_V2_USE_BG_RES_LOCK = {bool(getattr(cfg, 'NDR_V2_USE_BG_RES_LOCK', False))}")
            logger.log(f"NDR_V2_BG_LOCK_WEIGHT = {float(getattr(cfg, 'NDR_V2_BG_LOCK_WEIGHT', 0.0)):.8f}")
            logger.log(f"NDR_V2_BG_RES_LOCK_START_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_RES_LOCK_START_EPOCH', 1))}")
            logger.log(f"NDR_V2_BG_RES_LOCK_RAMP_END_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_RES_LOCK_RAMP_END_EPOCH', 1))}")
            logger.log(f"NDR_V2_BG_RES_LOCK_STOP_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_RES_LOCK_STOP_EPOCH', int(getattr(cfg, 'MAX_EPOCH', 25)) + 1))}")
            logger.log(f"NDR_V2_BG_RES_LOCK_WEIGHT_MAX = {float(getattr(cfg, 'NDR_V2_BG_RES_LOCK_WEIGHT_MAX', getattr(cfg, 'NDR_V2_BG_LOCK_WEIGHT', 0.0))):.8f}")
            logger.log(f"NDR_V2_USE_BG_PROB_LOCK = {bool(getattr(cfg, 'NDR_V2_USE_BG_PROB_LOCK', False))}")
            logger.log(f"NDR_V2_BG_PROB_LOCK_START_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_PROB_LOCK_START_EPOCH', 36))}")
            logger.log(f"NDR_V2_BG_PROB_LOCK_RAMP_END_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_PROB_LOCK_RAMP_END_EPOCH', 38))}")
            logger.log(f"NDR_V2_BG_PROB_LOCK_STOP_EPOCH = {int(getattr(cfg, 'NDR_V2_BG_PROB_LOCK_STOP_EPOCH', 46))}")
            logger.log(f"NDR_V2_BG_PROB_LOCK_WEIGHT_MAX = {float(getattr(cfg, 'NDR_V2_BG_PROB_LOCK_WEIGHT_MAX', 0.0)):.8f}")
            logger.log(f"NDR_V2_BG_PROB_LOCK_DELTA = {float(getattr(cfg, 'NDR_V2_BG_PROB_LOCK_DELTA', 0.02)):.6f}")
            logger.log(f"NDR_V2_BG_CORE_THRESH = {float(getattr(cfg, 'NDR_V2_BG_CORE_THRESH', 0.5)):.6f}")
            logger.log(f"NDR_V2_LOW_TARGET_THRESH = {float(getattr(cfg, 'NDR_V2_LOW_TARGET_THRESH', getattr(cfg, 'NDR_V2_BG_LOCK_TARGET_THRESH', 0.15))):.6f}")
            logger.log(f"NDR_V2_LOW_TARGET_WEIGHT_THRESH = {float(getattr(cfg, 'NDR_V2_LOW_TARGET_WEIGHT_THRESH', getattr(cfg, 'NDR_V2_BG_LOCK_WEIGHT_THRESH', 0.50))):.6f}")
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
            "dabe_pu_v12_shape_desplsched",
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
        use_tepr_lite = bool(getattr(cfg, "USE_TEPR_LITE", False)) and use_dabe_pu_despl_sched
        use_rast = bool(getattr(cfg, "USE_RAST", False)) and use_dabe_pu_despl_sched
        use_hbns_lite = bool(getattr(cfg, "USE_HBNS_LITE", False)) and use_dabe_pu_despl_sched
        use_epr_pos = bool(getattr(cfg, "USE_EPR_POS", False)) and use_dabe_pu_despl_sched
        use_esa_asym = bool(getattr(cfg, "USE_ESA_ASYM", False)) and use_dabe_pu_despl_sched
        use_esa_post_reset = bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)) and use_dabe_pu_despl_sched
        use_esa_ber = bool(getattr(cfg, "USE_ESA_BER", False)) and use_dabe_pu_despl_sched
        use_esa_diagnostic = bool(getattr(cfg, "USE_ESA_DIAGNOSTIC", False)) and use_dabe_pu_despl_sched
        use_tce = bool(getattr(cfg, "USE_TCE", False)) and use_dabe_pu_despl_sched
        use_lceg = bool(getattr(cfg, "USE_LCEG", False)) and use_dabe_pu_despl_sched
        use_cssd_train = use_cssd(cfg)
        use_pa_dagp_train = use_pa_dagp(cfg)
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
        if bool(getattr(cfg, "USE_TCE", False)):
            if not use_dabe_pu_despl_sched:
                raise RuntimeError("USE_TCE=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
            if dabe_pu_despl_teacher_target_mode != "binary":
                raise RuntimeError("USE_TCE=True requires binary full teacher target.")
            if dabe_pu_despl_static_target_mode != "soft":
                raise RuntimeError("USE_TCE=True requires DABE_PU_STATIC_TARGET_MODE='soft'.")
            if use_hbns_lite or use_epr_pos or use_proto:
                raise RuntimeError("USE_TCE=True cannot be combined with HBNS-lite, EPR-pos, or proto contrast.")
            if str(getattr(cfg, "TCE_REGION", "extent")).lower() != "extent":
                raise RuntimeError("TCE-v1 currently supports TCE_REGION='extent' only.")
        if bool(getattr(cfg, "USE_LCEG", False)):
            if not use_dabe_pu_despl_sched:
                raise RuntimeError("USE_LCEG=True requires TEACHER_FUSION_MODE='dabe_pu_despl_sched'.")
            if dabe_pu_despl_teacher_target_mode != "binary":
                raise RuntimeError("USE_LCEG=True requires binary full teacher target.")
            if dabe_pu_despl_static_target_mode != "soft":
                raise RuntimeError("USE_LCEG=True requires DABE_PU_STATIC_TARGET_MODE='soft'.")
            if bool(getattr(cfg, "USE_TCE", False)) or use_hbns_lite or use_epr_pos or use_proto:
                raise RuntimeError("USE_LCEG=True cannot be combined with TCE, HBNS-lite, EPR-pos, or proto contrast.")
            if bool(getattr(cfg, "LCEG_USE_UNKNOWN", False)):
                raise RuntimeError("LCEG-v1 requires LCEG_USE_UNKNOWN=False.")
            if bool(getattr(cfg, "LCEG_APPLY_TO_BASE_AUX", False)):
                raise RuntimeError("LCEG-v1 must not apply to base aux.")
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
        if use_cacd(cfg):
            _, cacd_train_reason = check_cacd_feature_cache(
                cfg,
                "train",
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] CACD F10/F11:train ready | {cacd_train_reason}")
            if not args.debug_loader_only:
                _, cacd_val_reason = check_cacd_feature_cache(
                    cfg,
                    "val",
                    max_samples=sample_limit if sample_limit >= 0 else None,
                )
                logger.log(f"[Cache] CACD F10/F11:val ready | {cacd_val_reason}")
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
        if use_cssd_train:
            _, cssd_reason = check_cssd_hr_feature_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] CSSD HR feature:train ready | {cssd_reason}")
        if use_tce:
            _, tce_reason = check_tce_cover_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] TCE cover cache ready | {tce_reason}")
        if use_lceg:
            _, lceg_reason = check_lceg_cover_cache(
                cfg,
                max_samples=sample_limit if sample_limit >= 0 else None,
            )
            logger.log(f"[Cache] LCEG cover cache ready | {lceg_reason}")
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
        if use_cacd(cfg):
            student_keys = list(student.state_dict().keys())
            teacher_keys = list(teacher.state_dict().keys())
            if student_keys != teacher_keys:
                raise RuntimeError("CACD student/teacher state_dict keys differ.")
            forbidden_state_tokens = (
                "ndr_branch",
                "csd_residual",
                "pa_dagp",
                "graph_pred",
                "hr_bfr",
            )
            bad_keys = [key for key in student_keys if any(token in key.lower() for token in forbidden_state_tokens)]
            if bad_keys:
                raise RuntimeError(f"CACD state_dict contains forbidden old-head parameters: {bad_keys[:20]}")
            with torch.no_grad():
                max_init_diff = max(
                    (
                        float((left - right).abs().max().item())
                        for left, right in zip(student.state_dict().values(), teacher.state_dict().values())
                        if torch.is_tensor(left) and left.is_floating_point()
                    ),
                    default=0.0,
                )
            if max_init_diff != 0.0:
                raise RuntimeError(f"CACD student/teacher initial diff must be zero, got {max_init_diff:.9g}.")
            count = lambda module: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            total_params = count(student)
            logger.log(f"total_trainable_params = {total_params}")
            logger.log(f"cacd_trainable_params = {total_params}")
            logger.log(f"anchor_head_params = {count(student.anchor_estimator)}")
            logger.log(f"context_reasoner_params = {count(student.context_reasoner)}")
            logger.log(f"decoder_params = {count(student.decoder)}")
            logger.log("DAGP module instantiated = False")
            logger.log("NDR module instantiated = False")
            logger.log("CSD module instantiated = False")
            logger.log("CACD module instantiated = True")
            logger.log(f"student_teacher_state_keys_equal = True")
            logger.log(f"student_teacher_initial_max_diff = {max_init_diff:.9g}")

        criterion = torch.nn.BCEWithLogitsLoss()
        criterion_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer, scheduler = build_optimizer_scheduler(cfg, student)

        best_metric = float("inf")
        best_epoch = 0
        global_step = 0
        start_epoch = 1
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            required_keys = {
                "epoch",
                "backbone_key",
                "student",
                "teacher",
                "optimizer",
                "scheduler",
            }
            missing_keys = sorted(required_keys.difference(checkpoint))
            if missing_keys:
                raise RuntimeError(
                    f"Resume checkpoint is missing required keys: {missing_keys}"
                )
            if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
                raise RuntimeError(
                    "Resume backbone mismatch: "
                    f"{checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )

            student.load_state_dict(checkpoint["student"], strict=True)
            teacher.load_state_dict(checkpoint["teacher"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

            best_metric = float(checkpoint.get("best_metric", best_metric))
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            saved_epoch = int(checkpoint["epoch"])
            start_epoch = saved_epoch + 1
            if start_epoch > max_epoch:
                raise RuntimeError(
                    f"Resume checkpoint epoch {saved_epoch} leaves no epochs to run "
                    f"with max_epoch={max_epoch}."
                )

            if "global_step" in checkpoint:
                global_step = int(checkpoint["global_step"])
                global_step_source = "checkpoint"
            else:
                global_step = int(scheduler.state_dict().get("last_epoch", 0))
                global_step_source = "scheduler.last_epoch"

            logger.log(
                f"[Resume] checkpoint={resume_path} | "
                f"saved_epoch={saved_epoch} | "
                f"start_epoch={start_epoch} | "
                f"global_step={global_step} | "
                f"global_step_source={global_step_source} | "
                f"lr={current_lr(optimizer):.8f} | "
                "student_restored=True | teacher_restored=True | "
                "optimizer_restored=True | scheduler_restored=True"
            )

        tepr_memory = None
        if use_tepr_lite:
            memory_update_end = int(
                getattr(cfg, "TEPR_MEMORY_UPDATE_END_EPOCH", 20)
            )
            if args.resume and start_epoch <= memory_update_end:
                raise RuntimeError(
                    "Cannot resume TEPR-Lite inside its temporal-memory update window: "
                    f"start_epoch={start_epoch}, update_end={memory_update_end}. "
                    "Legacy checkpoints do not contain TEPR temporal memory."
                )
            if start_epoch <= memory_update_end:
                tepr_memory = TemporalTeacherMemory(
                    num_samples=len(train_dataset),
                    height=int(cfg.LOSS_SIZE),
                    width=int(cfg.LOSS_SIZE),
                    dtype=str(getattr(cfg, "TEPR_MEMORY_DTYPE", "float16")),
                )
                memory_bytes = (
                    tepr_memory.mean.numel() * tepr_memory.mean.element_size()
                    + tepr_memory.second.numel() * tepr_memory.second.element_size()
                    + tepr_memory.count.numel() * tepr_memory.count.element_size()
                )
                logger.log(
                    "[TEPR-Lite] temporal memory initialized | "
                    f"num_samples={len(train_dataset)} | "
                    f"shape=[1,{int(cfg.LOSS_SIZE)},{int(cfg.LOSS_SIZE)}] | "
                    f"dtype={getattr(cfg, 'TEPR_MEMORY_DTYPE', 'float16')} | "
                    f"bytes={memory_bytes}"
                )
            else:
                logger.log(
                    "[TEPR-Lite] temporal memory inactive | "
                    f"start_epoch={start_epoch} | update_end={memory_update_end} | "
                    "memory_active=False"
                )
        drepp_memory_bank = {}
        gkd_first_batch_logged = False
        dagp_first_batch_logged = False
        dagp_safe_first_batch_logged = False
        csd_first_batch_logged = False
        cssd_first_active_batch_logged = False
        cssd_area_ratio_warning_streak = 0
        cssd_transfer_noop_warning_streak = 0
        cacd_first_batch_logged = False
        cacd_warning_streaks = {}
        hr_bfr_first_batch_logged = False
        ndr_first_batch_logged = False
        ndr_v2_bg_lock_first_batch_logged = False
        tadr_first_batch_logged = False
        mvflip_first_batch_logged = False
        mvproto_first_batch_logged = False
        rast_first_batch_logged = False
        tepr_first_batch_logged_epoch = None
        hbns_first_batch_logged = False
        epr_first_batch_logged = False
        esa_asym_first_batch_logged = False
        esa_ber_first_active_batch_logged = False
        esa_ber_noop_warning_streak = 0
        esa_ber_area_warning_streak = 0
        esa_ber_pre_active_area_reference = None
        tce_train_first_batch_logged = False
        lceg_train_first_batch_logged = False
        pa_dagp_first_active_batch_logged = False
        pa_dagp_warning_streaks = {
            "positive_ratio": 0,
            "negative_ratio": 0,
            "ambiguous_ratio": 0,
            "signed_pol_std": 0,
            "edge_suppressed_ratio": 0,
            "anchor_valid_ratio": 0,
        }
        lr_floor_activated_logged = False
        gkd_first_batch_path = train_dir / "gkd_first_batch.csv"
        gkd_audit_csv_path = train_dir / "gkd_audit_epoch.csv"
        gkd_branch_csv_path = train_dir / "gkd_branch_epoch.csv"
        gkd_branch_v2_csv_path = train_dir / "gkd_branch_v2_epoch.csv"
        gkd_branch_v2_first_batch_path = train_dir / "gkd_branch_v2_first_batch.csv"
        gkd_branch_v3_csv_path = train_dir / "gkd_branch_v3_epoch.csv"
        gkd_branch_v3_first_batch_path = train_dir / "gkd_branch_v3_first_batch.csv"

        for epoch in range(start_epoch, max_epoch + 1):
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
            cacd_aux_sums = {}
            cacd_anchor_sums = {}
            cacd_loss_seg_sum = 0.0
            cacd_loss_anchor_sum = 0.0
            cacd_loss_anchor_weighted_sum = 0.0
            cacd_loss_total_sum = 0.0
            cacd_stat_batches = 0
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
            dabe_pu_v12_target_base_mean_sum = 0.0
            dabe_pu_v12_weight_base_mean_sum = 0.0
            dabe_pu_v12_target_delta_mean_sum = 0.0
            dabe_pu_v12_sc_bg_lock_sum = 0.0
            dabe_pu_v12_sc_extent_agree_sum = 0.0
            dabe_pu_v12_sc_lost_extent_sum = 0.0
            dabe_pu_v12_sc_new_boundary_sum = 0.0
            tepr_epoch_accumulator = new_tepr_epoch_accumulator() if use_tepr_lite else None
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
            teacher_routing_scale_sum = 0.0
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
            esa_post_stat_keys = (
                "esa_post_reset_scale",
                "teacher_routing_scale",
                "esa_post_map_mean",
                "esa_post_map_fg_core_mean",
                "esa_post_map_bg_core_mean",
                "esa_post_map_unknown_mean",
                "esa_post_map_extent_teacher_fg_mean",
                "esa_post_map_extent_bg_fg_like_mean",
                "esa_post_map_extent_bg_ambig_mean",
                "esa_post_map_extent_bg_bg_like_mean",
            )
            esa_post_sums = {key: 0.0 for key in esa_post_stat_keys}
            esa_post_valid_keys = (
                "esa_post_fg_core_valid",
                "esa_post_bg_core_valid",
                "esa_post_unknown_valid",
                "esa_post_extent_teacher_fg_valid",
                "esa_post_extent_bg_fg_like_valid",
                "esa_post_extent_bg_ambig_valid",
                "esa_post_extent_bg_bg_like_valid",
            )
            esa_post_valid_counts = {key: 0 for key in esa_post_valid_keys}
            esa_post_active_batches = 0
            esa_post_map_min = None
            esa_post_map_max = None
            esa_post_stat_batches = 0
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
            esa_ber_stat_keys = (
                "ber_scale",
                "lambda_ber_eff",
                "proto_valid_ratio",
                "graph_valid_ratio",
                "valid_image_ratio",
                "pos_raw_ratio",
                "neg_extent_raw_ratio",
                "neg_hard_bg_raw_ratio",
                "neg_raw_ratio",
                "selected_pos_ratio",
                "selected_neg_ratio",
                "selected_pairs_mean",
                "pos_margin_mean",
                "pos_conn_mean",
                "pos_student_prob_mean",
                "pos_teacher_prob_mean",
                "neg_margin_mean",
                "neg_conn_mean",
                "neg_student_prob_mean",
                "neg_teacher_prob_mean",
                "pos_logit_mean",
                "neg_logit_mean",
                "logit_gap",
                "rank_violation_ratio",
                "loss_ber_raw",
                "loss_ber_weighted",
                "ber_to_main_loss_ratio",
                "student_prob_fg_core",
                "student_prob_bg_core",
                "student_prob_extent",
                "teacher_fg_extent",
            )
            esa_ber_sums = {key: 0.0 for key in esa_ber_stat_keys}
            esa_ber_pairs_min = None
            esa_ber_pairs_max = None
            esa_ber_topk_sum_error_max = 0.0
            esa_ber_stat_batches = 0
            esa_ber_source_sums = {}
            esa_ber_loss_ratio_warning = False
            tce_scale_sum = 0.0
            tce_lambda_sum = 0.0
            tce_cover_area_sum = 0.0
            tce_current_area_sum = 0.0
            tce_shrink_gate_ratio_sum = 0.0
            tce_bg_safe_gate_ratio_sum = 0.0
            tce_image_gate_ratio_sum = 0.0
            tce_lost_raw_ratio_sum = 0.0
            tce_lost_capped_ratio_sum = 0.0
            tce_new_raw_ratio_sum = 0.0
            tce_new_capped_ratio_sum = 0.0
            tce_total_ratio_sum = 0.0
            tce_valid_image_ratio_sum = 0.0
            tce_lost_margin_mean_sum = 0.0
            tce_new_margin_mean_sum = 0.0
            tce_lost_cover_conf_mean_sum = 0.0
            tce_current_teacher_conf_lost_mean_sum = 0.0
            tce_current_teacher_conf_new_mean_sum = 0.0
            tce_teacher_map_final_mean_sum = 0.0
            tce_teacher_map_final_tce_mean_sum = 0.0
            tce_loss_sum = 0.0
            tce_skipped_no_fg_proto_sum = 0
            tce_skipped_no_bg_proto_sum = 0
            tce_stat_batches = 0
            lceg_scale_sum = 0.0
            lceg_lambda_sum = 0.0
            lceg_cover_area_sum = 0.0
            lceg_current_area_sum = 0.0
            lceg_current_fg_extent_ratio_sum = 0.0
            lceg_cover_fg_extent_ratio_sum = 0.0
            lceg_shrink_gate_ratio_sum = 0.0
            lceg_bg_safe_gate_ratio_sum = 0.0
            lceg_core_raw_ratio_sum = 0.0
            lceg_core_capped_ratio_sum = 0.0
            lceg_lost_raw_ratio_sum = 0.0
            lceg_lost_capped_ratio_sum = 0.0
            lceg_new_raw_ratio_sum = 0.0
            lceg_new_capped_ratio_sum = 0.0
            lceg_total_ratio_sum = 0.0
            lceg_valid_image_ratio_sum = 0.0
            lceg_core_conf_mean_sum = 0.0
            lceg_lost_cover_prob_mean_sum = 0.0
            lceg_lost_margin_mean_sum = 0.0
            lceg_new_margin_mean_sum = 0.0
            lceg_teacher_map_final_mean_sum = 0.0
            lceg_teacher_map_final_lceg_mean_sum = 0.0
            lceg_teacher_map_coarse_mean_sum = 0.0
            lceg_teacher_map_coarse_lceg_mean_sum = 0.0
            lceg_loss_core_sum = 0.0
            lceg_loss_lost_sum = 0.0
            lceg_loss_new_sum = 0.0
            lceg_loss_final_sum = 0.0
            lceg_loss_coarse_sum = 0.0
            lceg_loss_sum = 0.0
            lceg_skipped_no_fg_proto_sum = 0
            lceg_skipped_no_bg_proto_sum = 0
            lceg_stat_batches = 0
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
            total_ndr_v2_bg_lock_loss = 0.0
            total_ndr_v2_weighted_loss = 0.0
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
            pa_dagp_stat_keys = (
                "edge_scale",
                "aux_scale",
                "edge_cut_eff",
                "lambda_pa_eff",
                "anchor_valid_ratio",
                "anchor_cosine_mean",
                "rho_mean",
                "raw_pol_mean",
                "raw_pol_std",
                "signed_pol_mean",
                "signed_pol_std",
                "positive_ratio",
                "negative_ratio",
                "ambiguous_ratio",
                "fg_core_pol_mean",
                "bg_core_pol_mean",
                "core_gap",
                "hard_fg_ratio",
                "hard_bg_ratio",
                "hard_fg_pol_mean",
                "hard_bg_pol_mean",
                "cross_edge_ratio",
                "gate_mean",
                "edge_suppressed_ratio",
                "loss_pa_fg",
                "loss_pa_bg",
                "loss_pa_core",
                "loss_pa_hfg",
                "loss_pa_hbg",
                "loss_pa_hard",
                "loss_pa_raw",
                "loss_pa_weighted",
            )
            pa_dagp_epoch_sums = {key: 0.0 for key in pa_dagp_stat_keys}
            pa_dagp_gate_min = None
            pa_dagp_gate_max = None
            pa_dagp_compare_coarse_abs_diff = 0.0
            pa_dagp_compare_coarse_area_delta = 0.0
            pa_dagp_compare_final_area_delta = 0.0
            pa_dagp_stat_batches = 0
            csd_scale_sum = 0.0
            csd_beta_sum = 0.0
            csd_gate_mean_sum = 0.0
            csd_gate_min = None
            csd_gate_max = None
            csd_residual_abs_mean_sum = 0.0
            csd_residual_abs_max = None
            csd_final_minus_coarse_abs_mean_sum = 0.0
            csd_bg_reliable_ratio_sum = 0.0
            csd_loss_bg_detail_sum = 0.0
            csd_loss_bg_detail_weighted_sum = 0.0
            csd_boundary_scale_sum = 0.0
            csd_boundary_target_mean_sum = 0.0
            csd_loss_boundary_sum = 0.0
            csd_loss_boundary_weighted_sum = 0.0
            csd_stat_batches = 0
            cssd_scale_epoch = float(get_cssd_scale(epoch, cfg)) if use_cssd_train else 0.0
            cssd_high_forward_batches = 0
            cssd_high_forward_calls = 0
            cssd_stat_batches = 0
            cssd_epoch_sums = {
                "lambda_hr": 0.0,
                "lambda_pred": 0.0,
                "lambda_boundary": 0.0,
                "loss_normal_group": 0.0,
                "loss_high_group": 0.0,
                "loss_dual_group": 0.0,
                "loss_high_static_group": 0.0,
                "loss_high_teacher_group": 0.0,
                "loss_core": 0.0,
                "loss_transfer": 0.0,
                "loss_pred": 0.0,
                "loss_boundary": 0.0,
                "loss_cssd_weighted": 0.0,
                "normal_prob_mean": 0.0,
                "high_prob_mean": 0.0,
                "normal_pred_area": 0.0,
                "high_pred_area": 0.0,
                "normal_high_prob_abs_diff": 0.0,
                "normal_high_binary_agreement": 0.0,
                "normal_conf_mean": 0.0,
                "high_conf_mean": 0.0,
                "high_more_conf_ratio": 0.0,
                "conf_adv_mean": 0.0,
                "core_mask_ratio": 0.0,
                "transfer_raw_ratio": 0.0,
                "transfer_capped_ratio": 0.0,
                "transfer_valid_image_ratio": 0.0,
                "boundary_mask_ratio": 0.0,
                "boundary_valid_image_ratio": 0.0,
            }
            hr_bfr_scale_sum = 0.0
            hr_bfr_beta_sum = 0.0
            hr_bfr_band_ratio_sum = 0.0
            hr_bfr_band_ratio_max = None
            hr_bfr_valid_img_ratio_sum = 0.0
            hr_bfr_skip_img_ratio_sum = 0.0
            hr_bfr_active_pixel_ratio_sum = 0.0
            hr_bfr_anchor_area_sum = 0.0
            hr_bfr_hr_area_sum = 0.0
            hr_bfr_area_delta_sum = 0.0
            hr_bfr_minus_anchor_sum = 0.0
            hr_bfr_residual_abs_mean_sum = 0.0
            hr_bfr_residual_abs_max = None
            hr_bfr_bg_reliable_ratio_sum = 0.0
            hr_bfr_edge_support_sum = 0.0
            hr_bfr_loss_band_sum = 0.0
            hr_bfr_loss_outband_sum = 0.0
            hr_bfr_loss_bg_sum = 0.0
            hr_bfr_loss_area_sum = 0.0
            hr_bfr_loss_edge_sum = 0.0
            hr_bfr_loss_total_sum = 0.0
            hr_bfr_stat_batches = 0
            ndr_beta_sum = 0.0
            ndr_gate_mean_sum = 0.0
            ndr_gate_min = None
            ndr_gate_max = None
            ndr_residual_abs_mean_sum = 0.0
            ndr_residual_abs_max = None
            ndr_stat_batches = 0
            ndr_v2_shape_alpha_sum = 0.0
            ndr_v2_gate_v1_mean_sum = 0.0
            ndr_v2_gate_v2_mean_sum = 0.0
            ndr_v2_boundary_mean_sum = 0.0
            ndr_v2_boundary_min = None
            ndr_v2_boundary_max = None
            ndr_v2_edge_norm_mean_sum = 0.0
            ndr_v2_shape_boost_mean_sum = 0.0
            ndr_v2_shape_boost_max = None
            ndr_v2_shape_lb_scale_sum = 0.0
            ndr_v2_shape_lb_lambda_sum = 0.0
            ndr_v2_shape_candidate_raw_ratio_sum = 0.0
            ndr_v2_shape_candidate_capped_ratio_sum = 0.0
            ndr_v2_shape_valid_image_ratio_sum = 0.0
            ndr_v2_shape_margin_mean_sum = 0.0
            ndr_v2_shape_edge_mean_sum = 0.0
            ndr_v2_shape_under_floor_mean_sum = 0.0
            ndr_v2_loss_shape_lb_sum = 0.0
            ndr_v2_bg_res_lock_scale_sum = 0.0
            ndr_v2_bg_res_lock_lambda_sum = 0.0
            ndr_v2_bg_prob_lock_scale_sum = 0.0
            ndr_v2_bg_prob_lock_lambda_sum = 0.0
            ndr_v2_bg_lock_area_sum = 0.0
            ndr_v2_bg_core_area_sum = 0.0
            ndr_v2_low_target_bg_area_sum = 0.0
            ndr_v2_positive_delta_bg_mean_sum = 0.0
            ndr_v2_positive_delta_bg_max = None
            ndr_v2_loss_bg_lock_sum = 0.0
            ndr_v2_loss_bg_prob_lock_sum = 0.0
            ndr_v2_weighted_loss_sum = 0.0
            ndr_v2_stat_batches = 0
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
                if use_cssd_train:
                    optimizer.zero_grad(set_to_none=True)
                pseudo = batch["pseudo"].to(device, non_blocking=True).float()
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                sobel_68 = make_sobel_68(cfg, batch, device)
                image_136 = make_image_136(cfg, batch, device)
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
                csd_bg_reliable_68 = None
                if (
                    (use_csd_head(cfg) or use_csd_v1r_head(cfg))
                    and use_dabe_pu
                    and bool(
                        getattr(
                            cfg,
                            "CSD_V1R_USE_BG_DETAIL_LOCK" if use_csd_v1r_head(cfg) else "CSD_USE_BG_DETAIL_LOCK",
                            True,
                        )
                    )
                ):
                    csd_bg_reliable_68 = build_csd_bg_reliable_mask(
                        batch,
                        cfg,
                        (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                        device,
                    ).float()

                pa_compare_original = bool(
                    use_pa_dagp_train
                    and bool(getattr(cfg, "PA_DAGP_DIAG_COMPARE_ORIGINAL_FIRST_BATCH", True))
                    and iter_idx == 0
                )
                student_out = forward_seg_head(
                    student,
                    model_input,
                    cfg,
                    image_68=image_68,
                    image_136=image_136,
                    sobel_68=sobel_68,
                    return_aux=use_dagp_safe_head(cfg) or use_csd_head(cfg) or use_csd_v1r_head(cfg) or use_cacd(cfg),
                    bg_reliable_68=csd_bg_reliable_68,
                    pa_compare_original=pa_compare_original,
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
                        image_136=image_136,
                        sobel_68=sobel_68,
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
                tepr_sample_indices = None
                tepr_stats = None
                rast_stats = {
                    "rast_scale": 0.0,
                    "rast_pre_reset_scale": 0.0,
                    "rast_post_reset_scale": 0.0,
                    "rast_scale_effective": 0.0,
                    "teacher_routing_scale": 0.0,
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
                    "esa_post_reset_enable": bool(getattr(cfg, "ESA_POST_RESET_ENABLE", False)),
                    "esa_post_reset_active": False,
                    "esa_post_reset_scale": 0.0,
                }
                if use_tepr_lite:
                    if "sample_index" not in batch:
                        raise RuntimeError("TEPR-Lite batch is missing stable sample_index.")
                    tepr_sample_indices = batch["sample_index"].long()
                    if tepr_sample_indices.ndim != 1 or int(tepr_sample_indices.shape[0]) != int(
                        teacher_prob.shape[0]
                    ):
                        raise RuntimeError(
                            f"TEPR sample_index must be [B], got {list(tepr_sample_indices.shape)}."
                        )
                    index_min = int(tepr_sample_indices.min().item())
                    index_max = int(tepr_sample_indices.max().item())
                    if index_min < 0 or index_max >= len(train_dataset):
                        raise RuntimeError(
                            f"TEPR sample_index out of range: {index_min}/{index_max}, "
                            f"dataset_size={len(train_dataset)}."
                        )
                    update_end = int(getattr(cfg, "TEPR_MEMORY_UPDATE_END_EPOCH", 20))
                    if int(epoch) <= update_end:
                        if tepr_memory is None:
                            raise RuntimeError("TEPR temporal memory was released before its update window ended.")
                        temporal_mean, temporal_second, history_count = tepr_memory.fetch(
                            tepr_sample_indices,
                            device,
                        )
                        rast_teacher_map_eff, tepr_stats = build_configured_tepr_teacher_weight_map(
                            cfg=cfg,
                            batch=batch,
                            teacher_prob=teacher_prob.detach(),
                            temporal_mean=temporal_mean,
                            temporal_second=temporal_second,
                            history_count=history_count,
                            epoch=epoch,
                            device=device,
                        )
                        tepr_stats["memory_active"] = True
                    else:
                        rast_teacher_map_eff = torch.ones_like(teacher_prob, dtype=torch.float32)
                        tepr_stats = make_tepr_inactive_stats(batch, teacher_prob, device)
                    rast_stats["teacher_routing_scale"] = float(tepr_stats["tepr_scale"])
                    if (
                        bool(getattr(cfg, "TEPR_DEBUG_FIRST_BATCH", True))
                        and tepr_first_batch_logged_epoch != int(epoch)
                        and int(epoch) % max(1, int(getattr(cfg, "TEPR_LOG_INTERVAL_EPOCH", 1))) == 0
                    ):
                        log_tepr_first_batch(
                            logger,
                            cfg,
                            epoch,
                            tepr_sample_indices,
                            tepr_stats,
                        )
                        tepr_first_batch_logged_epoch = int(epoch)
                elif use_rast:
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
                        logger.log(
                            "[RAST FirstBatch] teacher_routing_scale = "
                            f"{float(rast_stats['teacher_routing_scale']):.8f}"
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

                tce_teacher_map_final = None
                tce_lost_mask = torch.zeros_like(student_logits, dtype=torch.bool)
                tce_new_mask = torch.zeros_like(student_logits, dtype=torch.bool)
                tce_stats = {
                    "tce_scale": 0.0,
                    "lambda_tce_eff": 0.0,
                    "cover_area_mean": 0.0,
                    "current_area_mean": 0.0,
                    "shrink_gate_ratio": 0.0,
                    "bg_safe_gate_ratio": 0.0,
                    "image_gate_ratio": 0.0,
                    "lost_raw_ratio": 0.0,
                    "lost_capped_ratio": 0.0,
                    "new_raw_ratio": 0.0,
                    "new_capped_ratio": 0.0,
                    "tce_total_ratio": 0.0,
                    "tce_valid_image_ratio": 0.0,
                    "lost_margin_mean": 0.0,
                    "new_margin_mean": 0.0,
                    "lost_cover_conf_mean": 0.0,
                    "current_teacher_conf_lost_mean": 0.0,
                    "current_teacher_conf_new_mean": 0.0,
                    "teacher_map_final_mean": 1.0,
                    "teacher_map_final_tce_mean": 1.0,
                    "loss_tce": 0.0,
                    "tce_skipped_no_fg_proto": 0,
                    "tce_skipped_no_bg_proto": 0,
                }
                loss_tce_final = student_logits.sum() * 0.0
                if use_tce:
                    tce_teacher_map_final, tce_lost_mask, tce_new_mask, tce_stats = build_tce_masks_and_teacher_map(
                        cfg,
                        batch,
                        student_logits,
                        teacher_prob,
                        teacher_binary,
                        rast_teacher_map_eff,
                        epoch,
                        device,
                        image_68=image_68,
                    )
                    loss_tce_final = tce_lower_bound_loss_for_logits(
                        student_logits,
                        tce_lost_mask,
                        tce_new_mask,
                        cfg,
                    )
                    tce_stats["loss_tce"] = float(loss_tce_final.detach().item())
                    if not tce_train_first_batch_logged:
                        logger.log(f"[TCE Train FirstBatch] USE_TCE = {bool(getattr(cfg, 'USE_TCE', False))}")
                        logger.log(f"[TCE Train FirstBatch] TCE_VERSION = {getattr(cfg, 'TCE_VERSION', 'v1_temporal_coverage_expansion')}")
                        logger.log(f"[TCE Train FirstBatch] tce_scale = {float(tce_stats['tce_scale']):.8f}")
                        logger.log(f"[TCE Train FirstBatch] lambda_tce_eff = {float(tce_stats['lambda_tce_eff']):.8f}")
                        logger.log(f"[TCE Train FirstBatch] tce_candidate shape = {list(tce_lost_mask.shape)}")
                        logger.log(
                            "[TCE Train FirstBatch] lost/new candidate ratio = "
                            f"{float(tce_stats['lost_capped_ratio']):.6f}/"
                            f"{float(tce_stats['new_capped_ratio']):.6f}"
                        )
                        logger.log(
                            "[TCE Train FirstBatch] teacher_map_final mean/tce_mean = "
                            f"{float(tce_stats['teacher_map_final_mean']):.6f}/"
                            f"{float(tce_stats['teacher_map_final_tce_mean']):.6f}"
                        )
                        tce_train_first_batch_logged = True

                lceg_teacher_map_final = None
                lceg_teacher_map_coarse = None
                lceg_core_mask = torch.zeros_like(student_logits, dtype=torch.bool)
                lceg_lost_mask = torch.zeros_like(student_logits, dtype=torch.bool)
                lceg_new_mask = torch.zeros_like(student_logits, dtype=torch.bool)
                lceg_stats = {
                    "lceg_scale": 0.0,
                    "lambda_lceg_eff": 0.0,
                    "cover_area_mean": 0.0,
                    "cover_binary_area_mean": 0.0,
                    "cover_conf_mean": 0.0,
                    "current_area_mean": 0.0,
                    "current_fg_extent_ratio": 0.0,
                    "cover_fg_extent_ratio": 0.0,
                    "shrink_gate_ratio": 0.0,
                    "bg_safe_gate_ratio": 0.0,
                    "core_raw_ratio": 0.0,
                    "core_capped_ratio": 0.0,
                    "lost_raw_ratio": 0.0,
                    "lost_capped_ratio": 0.0,
                    "new_raw_ratio": 0.0,
                    "new_capped_ratio": 0.0,
                    "lceg_total_ratio": 0.0,
                    "lceg_valid_image_ratio": 0.0,
                    "core_conf_mean": 0.0,
                    "lost_cover_prob_mean": 0.0,
                    "lost_margin_mean": 0.0,
                    "new_margin_mean": 0.0,
                    "teacher_map_final_mean": 1.0,
                    "teacher_map_final_lceg_mean": 1.0,
                    "teacher_map_coarse_mean": 1.0,
                    "teacher_map_coarse_lceg_mean": 1.0,
                    "lceg_skipped_no_fg_proto": 0,
                    "lceg_skipped_no_bg_proto": 0,
                }
                loss_lceg_final = student_logits.sum() * 0.0
                loss_lceg_coarse = student_logits.sum() * 0.0
                loss_lceg = student_logits.sum() * 0.0
                loss_lceg_final_parts = {
                    "core": student_logits.sum() * 0.0,
                    "lost": student_logits.sum() * 0.0,
                    "new": student_logits.sum() * 0.0,
                }
                if use_lceg:
                    (
                        lceg_teacher_map_final,
                        lceg_teacher_map_coarse,
                        lceg_core_mask,
                        lceg_lost_mask,
                        lceg_new_mask,
                        lceg_stats,
                    ) = build_lceg_masks_and_teacher_maps(
                        cfg,
                        batch,
                        student_logits,
                        teacher_prob,
                        teacher_binary,
                        rast_teacher_map_eff,
                        epoch,
                        device,
                        image_68=image_68,
                    )
                    loss_lceg_final, loss_lceg_final_parts = lceg_lower_bound_loss_for_logits(
                        student_logits,
                        lceg_core_mask,
                        lceg_lost_mask,
                        lceg_new_mask,
                        cfg,
                    )
                    if not lceg_train_first_batch_logged:
                        logger.log(f"[LCEG Train FirstBatch] USE_LCEG = {bool(getattr(cfg, 'USE_LCEG', False))}")
                        logger.log(f"[LCEG Train FirstBatch] LCEG_VERSION = {getattr(cfg, 'LCEG_VERSION', 'v1_late_core_extent_guard')}")
                        logger.log(f"[LCEG Train FirstBatch] lceg_scale = {float(lceg_stats['lceg_scale']):.8f}")
                        logger.log(f"[LCEG Train FirstBatch] lambda_lceg_eff = {float(lceg_stats['lambda_lceg_eff']):.8f}")
                        logger.log(f"[LCEG Train FirstBatch] lceg_candidate shape = {list(lceg_core_mask.shape)}")
                        logger.log(
                            "[LCEG Train FirstBatch] core/lost/new candidate ratio = "
                            f"{float(lceg_stats['core_capped_ratio']):.6f}/"
                            f"{float(lceg_stats['lost_capped_ratio']):.6f}/"
                            f"{float(lceg_stats['new_capped_ratio']):.6f}"
                        )
                        logger.log(
                            "[LCEG Train FirstBatch] teacher_map final/coarse lceg mean = "
                            f"{float(lceg_stats['teacher_map_final_lceg_mean']):.6f}/"
                            f"{float(lceg_stats['teacher_map_coarse_lceg_mean']):.6f}"
                        )
                        lceg_train_first_batch_logged = True

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
                        if lceg_teacher_map_final is not None and bool(getattr(cfg, "LCEG_APPLY_TO_FINAL", True)):
                            loss_pu_teacher_final = lceg_teacher_bce_with_logits(
                                student_logits,
                                teacher_full_target,
                                lceg_teacher_map_final,
                                cfg,
                                eps=eps,
                            )
                        elif tce_teacher_map_final is not None and bool(getattr(cfg, "TCE_APPLY_TO_FINAL", True)):
                            loss_pu_teacher_final = tce_teacher_bce_with_logits(
                                student_logits,
                                teacher_full_target,
                                tce_teacher_map_final,
                                cfg,
                                eps=eps,
                            )
                        else:
                            loss_pu_teacher_final = rast_teacher_bce_with_logits(
                                student_logits,
                                teacher_full_target,
                                rast_teacher_map_eff,
                                cfg,
                                rast_stats["teacher_routing_scale"],
                                apply_to_loss=teacher_routing_apply_flag(cfg, "final"),
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
                loss_ndr_v2_shape_lb = student_logits.sum() * 0.0
                loss_ndr_v2_bg_lock = student_logits.sum() * 0.0
                loss_ndr_v2_bg_prob_lock = student_logits.sum() * 0.0
                loss_ndr_v2_weighted = student_logits.sum() * 0.0
                loss_csd_bg_detail = student_logits.sum() * 0.0
                loss_csd_boundary = student_logits.sum() * 0.0
                loss_hr_bfr = student_logits.sum() * 0.0
                csd_stats = {
                    "bg_reliable_ratio": 0.0,
                    "loss_bg_detail": 0.0,
                    "loss_bg_detail_weighted": 0.0,
                    "boundary_scale": 0.0,
                    "boundary_target_mean": 0.0,
                    "boundary_target_max": 0.0,
                    "loss_boundary": 0.0,
                    "loss_boundary_weighted": 0.0,
                }
                hr_bfr_stats = {
                    "hr_scale": 0.0,
                    "hr_beta_eff": 0.0,
                    "band_ratio": 0.0,
                    "band_ratio_max": 0.0,
                    "valid_img_ratio": 0.0,
                    "skip_img_ratio": 0.0,
                    "hr_active_pixel_ratio": 0.0,
                    "anchor_area": 0.0,
                    "hr_area": 0.0,
                    "area_delta": 0.0,
                    "hr_minus_anchor_abs_mean": 0.0,
                    "hr_residual_abs_mean": 0.0,
                    "hr_residual_abs_max": 0.0,
                    "bg_reliable_ratio": 0.0,
                    "edge_support_mean": 0.0,
                    "loss_band_bce": 0.0,
                    "loss_outband_anchor": 0.0,
                    "loss_bg_prob_lock": 0.0,
                    "loss_area_neutral": 0.0,
                    "loss_edge_align": 0.0,
                    "loss_hr_bfr": 0.0,
                }
                ndr_v2_shape_stats = {
                    "shape_lb_scale": 0.0,
                    "lambda_shape_lb_eff": 0.0,
                    "shape_candidate_raw_ratio": 0.0,
                    "shape_candidate_capped_ratio": 0.0,
                    "shape_valid_image_ratio": 0.0,
                    "shape_margin_mean": 0.0,
                    "shape_edge_mean": 0.0,
                    "shape_under_floor_mean": 0.0,
                    "shape_lb_skipped_no_fg_proto": 0,
                    "shape_lb_skipped_no_bg_proto": 0,
                }
                ndr_v2_shape_candidate = torch.zeros_like(student_logits, dtype=torch.bool)
                ndr_v2_bg_lock_stats = {
                    "bg_lock_area": 0.0,
                    "bg_core_area": 0.0,
                    "low_target_bg_area": 0.0,
                    "positive_delta_bg_mean": 0.0,
                    "positive_delta_bg_max": 0.0,
                    "positive_delta_mean": 0.0,
                    "bg_res_lock_scale": 0.0,
                    "lambda_bg_res_lock_eff": 0.0,
                    "bg_prob_lock_scale": 0.0,
                    "lambda_bg_prob_lock_eff": 0.0,
                    "loss_bg_res_lock": 0.0,
                    "loss_bg_prob_lock": 0.0,
                }
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
                if use_ndr_branch(cfg) or use_csd_head(cfg) or use_csd_v1r_head(cfg) or use_cacd(cfg):
                    if not isinstance(student_out, dict) or "coarse_logits_68" not in student_out:
                        raise RuntimeError("Decoder aux path requires coarse_logits_68 in student output.")
                    decoder_coarse_aux_enabled = (
                        bool(getattr(cfg, "CACD_USE_COARSE_AUX", True))
                        if use_cacd(cfg)
                        else (
                            bool(getattr(cfg, "CSD_V1R_USE_COARSE_AUX", True))
                            if use_csd_v1r_head(cfg)
                            else (
                                bool(getattr(cfg, "CSD_USE_COARSE_AUX", False))
                                if use_csd_head(cfg)
                                else bool(getattr(cfg, "USE_NDR_COARSE_AUX", True))
                            )
                        )
                    )
                    decoder_coarse_aux_lambda = (
                        float(getattr(cfg, "LAMBDA_CACD_COARSE_AUX", 0.5))
                        if use_cacd(cfg)
                        else (
                            float(getattr(cfg, "LAMBDA_CSD_V1R_COARSE_AUX", 0.5))
                            if use_csd_v1r_head(cfg)
                            else (
                                float(getattr(cfg, "LAMBDA_CSD_COARSE_AUX", 0.5))
                                if use_csd_head(cfg)
                                else float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5))
                            )
                        )
                    )
                    if use_dabe_oem:
                        lambda_dyn_pos, lambda_dyn_bg = get_dabe_oem_schedule(epoch, cfg)
                        seed_terms = [loss_oem_seed_final]
                        seed_weights = [1.0]
                        dyn_pos_terms = [loss_oem_dyn_pos]
                        dyn_bg_terms = [loss_oem_dyn_bg]
                        dyn_weights = [1.0]
                        if (
                            decoder_coarse_aux_enabled
                            and bool(getattr(cfg, "OEM_USE_SEED_LOSS_ON_COARSE", True))
                        ):
                            coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                            loss_oem_seed_coarse, _ = build_dabe_oem_seed_loss(coarse_logits, batch, cfg)
                            seed_terms.append(loss_oem_seed_coarse)
                            seed_weights.append(decoder_coarse_aux_lambda)
                            if bool(getattr(cfg, "OEM_USE_DYNAMIC_ON_COARSE", False)):
                                dyn_pos_coarse, dyn_bg_coarse, _ = build_dabe_oem_dynamic_loss(
                                    coarse_logits,
                                    oem_masks,
                                    cfg,
                                )
                                dyn_pos_terms.append(dyn_pos_coarse)
                                dyn_bg_terms.append(dyn_bg_coarse)
                                dyn_weights.append(decoder_coarse_aux_lambda)
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
                        if decoder_coarse_aux_enabled:
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
                            pu_aux_weights.append(decoder_coarse_aux_lambda)
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
                        if use_cssd_train:
                            normal_group_parts = compute_cssd_original_group_loss(
                                student_out,
                                pu_static_target,
                                pu_static_weight_map,
                                teacher_full_target,
                                rast_teacher_map_eff,
                                rast_stats["teacher_routing_scale"],
                                epoch,
                                cfg,
                                apply_final=True,
                                apply_coarse=True,
                                apply_base=True,
                            )
                            pu_static_loss_weight = float(normal_group_parts["static_weight"])
                            pu_teacher_loss_weight = float(normal_group_parts["teacher_weight"])
                            loss_pu_static_final = normal_group_parts["loss_static_final"]
                            loss_pu_teacher_final = normal_group_parts["loss_teacher_final"]
                            loss_pu_static_coarse = normal_group_parts["loss_static_coarse"]
                            loss_pu_teacher_coarse = normal_group_parts["loss_teacher_coarse"]
                            loss_pu_static_base = normal_group_parts["loss_static_base"]
                            loss_pu_teacher_base = normal_group_parts["loss_teacher_base"]
                            loss_pu_static_group = normal_group_parts["loss_static_group"]
                            loss_pu_teacher_group = normal_group_parts["loss_teacher_group"]
                            loss = normal_group_parts["loss_group"]
                            loss_ndr_coarse_aux = (
                                pu_static_loss_weight * loss_pu_static_coarse
                                + pu_teacher_loss_weight * loss_pu_teacher_coarse
                            )
                            loss_aux_base = (
                                pu_static_loss_weight * loss_pu_static_base
                                + pu_teacher_loss_weight * loss_pu_teacher_base
                            )
                        else:
                            eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
                            pu_static_loss_weight, pu_teacher_loss_weight = get_dabe_pu_despl_schedule(epoch, cfg)
                            static_terms = [loss_pu_static_final]
                            teacher_terms = [loss_pu_teacher_final]
                            pu_aux_weights = [1.0]
                            if decoder_coarse_aux_enabled:
                                coarse_logits = resize_logits_for_loss(student_out["coarse_logits_68"], cfg)
                                loss_pu_static_coarse = weighted_bce_with_logits(
                                    coarse_logits,
                                    pu_static_target,
                                    pu_static_weight_map,
                                    eps=eps,
                                )
                                if lceg_teacher_map_coarse is not None and bool(getattr(cfg, "LCEG_APPLY_TO_COARSE_AUX", True)):
                                    loss_pu_teacher_coarse = lceg_teacher_bce_with_logits(
                                        coarse_logits,
                                        teacher_full_target,
                                        lceg_teacher_map_coarse,
                                        cfg,
                                        eps=eps,
                                    )
                                    loss_lceg_coarse, _ = lceg_lower_bound_loss_for_logits(
                                        coarse_logits,
                                        lceg_core_mask,
                                        lceg_lost_mask,
                                        lceg_new_mask,
                                        cfg,
                                    )
                                else:
                                    loss_pu_teacher_coarse = rast_teacher_bce_with_logits(
                                        coarse_logits,
                                        teacher_full_target,
                                        rast_teacher_map_eff,
                                        cfg,
                                        rast_stats["teacher_routing_scale"],
                                        apply_to_loss=teacher_routing_apply_flag(cfg, "coarse"),
                                        eps=eps,
                                    )
                                static_terms.append(loss_pu_static_coarse)
                                teacher_terms.append(loss_pu_teacher_coarse)
                                pu_aux_weights.append(decoder_coarse_aux_lambda)
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
                                    rast_stats["teacher_routing_scale"],
                                    apply_to_loss=teacher_routing_apply_flag(cfg, "base"),
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
                        if decoder_coarse_aux_enabled:
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
                            pu_aux_weights.append(decoder_coarse_aux_lambda)
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
                        if decoder_coarse_aux_enabled:
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
                            ndr_weights.append(decoder_coarse_aux_lambda)
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
                    if use_ndr_branch(cfg) and bool(getattr(cfg, "NDR_USE_RES_REG", False)):
                        detail_gate = student_out["detail_gate"].detach()
                        residual_logits = student_out["residual_logits_68"]
                        loss_ndr_res_reg = torch.mean(torch.abs(detail_gate * residual_logits))
                        loss = loss + float(getattr(cfg, "NDR_RES_REG_WEIGHT", 0.001)) * loss_ndr_res_reg
                    if use_ndr_v2(cfg):
                        ndr_v2_shape_candidate, ndr_v2_shape_stats = build_ndr_v2_shape_lower_bound(
                            cfg,
                            batch,
                            student_logits,
                            student_out,
                            teacher_binary,
                            epoch,
                            device,
                        )
                        loss_ndr_v2_shape_lb = ndr_v2_shape_lower_bound_loss(
                            student_logits,
                            ndr_v2_shape_candidate,
                            float(getattr(cfg, "NDR_V2_SHAPE_LB_FLOOR", 0.35)),
                        )
                        loss_ndr_v2_bg_lock, loss_ndr_v2_bg_prob_lock, ndr_v2_bg_lock_stats = (
                            build_ndr_v2_bg_lock_losses(
                                student_logits,
                                student_out,
                                batch,
                                cfg,
                                epoch,
                            )
                        )
                        loss_ndr_v2_weighted = (
                            float(ndr_v2_shape_stats["lambda_shape_lb_eff"]) * loss_ndr_v2_shape_lb
                            + float(ndr_v2_bg_lock_stats["lambda_bg_res_lock_eff"]) * loss_ndr_v2_bg_lock
                            + float(ndr_v2_bg_lock_stats["lambda_bg_prob_lock_eff"]) * loss_ndr_v2_bg_prob_lock
                        )
                        loss = loss + loss_ndr_v2_weighted
                        if not ndr_v2_bg_lock_first_batch_logged:
                            logger.log(f"[NDR-v2 FirstBatch] USE_NDR_V2 = {bool(getattr(cfg, 'USE_NDR_V2', False))}")
                            logger.log(f"[NDR-v2 FirstBatch] NDR_VERSION = {getattr(cfg, 'NDR_VERSION', 'v2a_shape_gate_bg_lock')}")
                            if str(getattr(cfg, "NDR_VERSION", "")).lower().startswith("v2b"):
                                logger.log(f"[NDR-v2b FirstBatch] USE_NDR_V2 = {bool(getattr(cfg, 'USE_NDR_V2', False))}")
                                logger.log(f"[NDR-v2b FirstBatch] NDR_VERSION = {getattr(cfg, 'NDR_VERSION', 'v2b_shape_lb_bg_prob_lock')}")
                                logger.log(f"[NDR-v2b FirstBatch] shape_lb_scale = {float(ndr_v2_shape_stats['shape_lb_scale']):.8f}")
                                logger.log(f"[NDR-v2b FirstBatch] lambda_shape_lb_eff = {float(ndr_v2_shape_stats['lambda_shape_lb_eff']):.8f}")
                                logger.log(f"[NDR-v2b FirstBatch] bg_res_lock_scale = {float(ndr_v2_bg_lock_stats['bg_res_lock_scale']):.8f}")
                                logger.log(f"[NDR-v2b FirstBatch] bg_prob_lock_scale = {float(ndr_v2_bg_lock_stats['bg_prob_lock_scale']):.8f}")
                                logger.log(f"[NDR-v2b FirstBatch] shape_candidate shape = {list(ndr_v2_shape_candidate.shape)}")
                                logger.log(
                                    "[NDR-v2b FirstBatch] shape_candidate raw/capped ratio = "
                                    f"{float(ndr_v2_shape_stats['shape_candidate_raw_ratio']):.8f}/"
                                    f"{float(ndr_v2_shape_stats['shape_candidate_capped_ratio']):.8f}"
                                )
                            logger.log(
                                "[NDR-v2 FirstBatch] lambda shape/bg_res/bg_prob = "
                                f"{float(ndr_v2_shape_stats['lambda_shape_lb_eff']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['lambda_bg_res_lock_eff']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['lambda_bg_prob_lock_eff']):.8f}"
                            )
                            logger.log(
                                "[NDR-v2 FirstBatch] bg_lock/bg_core/low_target_bg area = "
                                f"{float(ndr_v2_bg_lock_stats['bg_lock_area']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['bg_core_area']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['low_target_bg_area']):.8f}"
                            )
                            logger.log(
                                "[NDR-v2 FirstBatch] positive_delta bg_mean/bg_max/all_mean = "
                                f"{float(ndr_v2_bg_lock_stats['positive_delta_bg_mean']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['positive_delta_bg_max']):.8f}/"
                                f"{float(ndr_v2_bg_lock_stats['positive_delta_mean']):.8f}"
                            )
                            logger.log(
                                "[NDR-v2 FirstBatch] loss shape_lb/bg_res/bg_prob/weighted = "
                                f"{float(loss_ndr_v2_shape_lb.detach().item()):.8f}/"
                                f"{float(loss_ndr_v2_bg_lock.detach().item()):.8f}/"
                                f"{float(loss_ndr_v2_bg_prob_lock.detach().item()):.8f}/"
                                f"{float(loss_ndr_v2_weighted.detach().item()):.8f}"
                            )
                            ndr_v2_bg_lock_first_batch_logged = True
                    loss_cssd_normal_group = loss
                    loss_cssd_high_group = student_logits.sum() * 0.0
                    loss_cssd_dual_group = loss_cssd_normal_group
                    loss_cssd_core = student_logits.sum() * 0.0
                    loss_cssd_transfer = student_logits.sum() * 0.0
                    loss_cssd_pred = student_logits.sum() * 0.0
                    loss_cssd_boundary_consistency = student_logits.sum() * 0.0
                    loss_cssd_weighted = student_logits.sum() * 0.0
                    cssd_lambda_hr = 0.0
                    cssd_lambda_pred = 0.0
                    cssd_lambda_boundary = 0.0
                    cssd_batch_stats = {}
                    if use_csd_head(cfg) or use_csd_v1r_head(cfg):
                        loss_csd_bg_detail, loss_csd_boundary, csd_stats = build_csd_aux_losses(
                            student_out,
                            batch,
                            cfg,
                            epoch,
                            device,
                        )
                        loss = loss + loss_csd_bg_detail + loss_csd_boundary
                        if not csd_first_batch_logged:
                            if use_csd_v1r_head(cfg):
                                log_csd_v1r_first_batch(logger, model_input, image_68, student_out, csd_stats)
                            else:
                                log_csd_first_batch(logger, model_input, image_68, student_out, csd_stats)
                            csd_first_batch_logged = True
                    if use_cssd_train and cssd_scale_epoch > 0.0:
                        cssd_feature_field = str(getattr(cfg, "CSSD_HR_FEATURE_FIELD", "feature_cssd_hr"))
                        if cssd_feature_field not in batch:
                            raise RuntimeError(
                                f"CSSD active batch is missing {cssd_feature_field!r}; cache preflight/dataset mismatch."
                            )
                        feature_cssd_hr = batch[cssd_feature_field]
                        if feature_cssd_hr.dtype != torch.float32:
                            raise RuntimeError(
                                f"CSSD high cache must remain float32, got {feature_cssd_hr.dtype}."
                            )
                        feature_cssd_hr = feature_cssd_hr.to(device, non_blocking=True)
                        if not torch.is_tensor(model_input) or list(model_input.shape[1:]) != [384, 37, 37]:
                            raise RuntimeError(
                                "CSSD normal student input must be a single [B,384,37,37] tensor, got "
                                f"{type(model_input).__name__} "
                                f"{list(model_input.shape) if torch.is_tensor(model_input) else ''}."
                            )
                        high_output, high_forward_calls = forward_cssd_high_microbatches(
                            student,
                            feature_cssd_hr,
                            image_68,
                            csd_bg_reliable_68,
                            cfg,
                        )
                        cssd_high_forward_batches += 1
                        cssd_high_forward_calls += int(high_forward_calls)
                        high_group_parts = compute_cssd_original_group_loss(
                            high_output,
                            pu_static_target,
                            pu_static_weight_map,
                            teacher_full_target,
                            rast_teacher_map_eff,
                            rast_stats["teacher_routing_scale"],
                            epoch,
                            cfg,
                            apply_final=bool(getattr(cfg, "CSSD_HR_SUP_APPLY_TO_FINAL", True)),
                            apply_coarse=bool(getattr(cfg, "CSSD_HR_SUP_APPLY_TO_COARSE", True)),
                            apply_base=bool(getattr(cfg, "CSSD_HR_SUP_APPLY_TO_BASE", True)),
                        )
                        loss_cssd_high_group = high_group_parts["loss_group"]
                        cssd_lambda_hr = (
                            float(getattr(cfg, "CSSD_HR_SUP_WEIGHT_MAX", 0.25)) * cssd_scale_epoch
                            if bool(getattr(cfg, "CSSD_USE_HR_SUPERVISED_LOSS", True))
                            else 0.0
                        )
                        loss_cssd_dual_group = (
                            loss_cssd_normal_group + cssd_lambda_hr * loss_cssd_high_group
                        ) / (1.0 + cssd_lambda_hr)
                        (
                            loss_cssd_core,
                            loss_cssd_transfer,
                            loss_cssd_pred,
                            loss_cssd_boundary_consistency,
                            cssd_batch_stats,
                        ) = build_cssd_distillation_losses(
                            student_out,
                            high_output,
                            batch,
                            image_68,
                            cfg,
                            cssd_scale_epoch,
                        )
                        cssd_lambda_pred = (
                            float(getattr(cfg, "CSSD_PRED_WEIGHT_MAX", 0.05)) * cssd_scale_epoch
                            if bool(getattr(cfg, "CSSD_USE_PRED_DISTILL", True))
                            else 0.0
                        )
                        cssd_lambda_boundary = (
                            float(getattr(cfg, "CSSD_BOUNDARY_WEIGHT_MAX", 0.02)) * cssd_scale_epoch
                            if bool(getattr(cfg, "CSSD_USE_BOUNDARY_CONSISTENCY", True))
                            else 0.0
                        )
                        loss_cssd_weighted = (
                            cssd_lambda_pred * loss_cssd_pred
                            + cssd_lambda_boundary * loss_cssd_boundary_consistency
                        )
                        # The normal CSD auxiliary remains outside dual-group normalization;
                        # the privileged view deliberately receives no CSD bg-detail auxiliary.
                        loss = (
                            loss_cssd_dual_group
                            + loss_csd_bg_detail
                            + loss_csd_boundary
                            + loss_cssd_weighted
                        )
                        cssd_stat_batches += 1
                        cssd_epoch_sums["lambda_hr"] += cssd_lambda_hr
                        cssd_epoch_sums["lambda_pred"] += cssd_lambda_pred
                        cssd_epoch_sums["lambda_boundary"] += cssd_lambda_boundary
                        cssd_epoch_sums["loss_normal_group"] += float(loss_cssd_normal_group.detach().item())
                        cssd_epoch_sums["loss_high_group"] += float(loss_cssd_high_group.detach().item())
                        cssd_epoch_sums["loss_dual_group"] += float(loss_cssd_dual_group.detach().item())
                        cssd_epoch_sums["loss_high_static_group"] += float(
                            high_group_parts["loss_static_group"].detach().item()
                        )
                        cssd_epoch_sums["loss_high_teacher_group"] += float(
                            high_group_parts["loss_teacher_group"].detach().item()
                        )
                        cssd_epoch_sums["loss_core"] += float(loss_cssd_core.detach().item())
                        cssd_epoch_sums["loss_transfer"] += float(loss_cssd_transfer.detach().item())
                        cssd_epoch_sums["loss_pred"] += float(loss_cssd_pred.detach().item())
                        cssd_epoch_sums["loss_boundary"] += float(
                            loss_cssd_boundary_consistency.detach().item()
                        )
                        cssd_epoch_sums["loss_cssd_weighted"] += float(loss_cssd_weighted.detach().item())
                        for stat_name in (
                            "normal_prob_mean",
                            "high_prob_mean",
                            "normal_pred_area",
                            "high_pred_area",
                            "normal_high_prob_abs_diff",
                            "normal_high_binary_agreement",
                            "normal_conf_mean",
                            "high_conf_mean",
                            "high_more_conf_ratio",
                            "conf_adv_mean",
                            "core_mask_ratio",
                            "transfer_raw_ratio",
                            "transfer_capped_ratio",
                            "transfer_valid_image_ratio",
                            "boundary_mask_ratio",
                            "boundary_valid_image_ratio",
                        ):
                            cssd_epoch_sums[stat_name] += float(cssd_batch_stats[stat_name])
                        if not cssd_first_active_batch_logged:
                            logger.log("[CSSD FirstActiveBatch] shared_model=True | separate_hr_head=False")
                            logger.log(
                                "[CSSD FirstActiveBatch] epoch/scale/lambda_hr/lambda_pred/lambda_boundary = "
                                f"{epoch}/{cssd_scale_epoch:.8f}/{cssd_lambda_hr:.8f}/"
                                f"{cssd_lambda_pred:.8f}/{cssd_lambda_boundary:.8f}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] normal/high feature shape = "
                                f"{list(model_input.shape)}/{list(feature_cssd_hr.shape)}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] normal native/final/coarse/base shape = "
                                f"{list(student_out['coarse_logits_native'].shape)}/"
                                f"{list(extract_logits(student_out).shape)}/"
                                f"{list(student_out['coarse_logits_68'].shape)}/"
                                f"{list(student_out['base_logits'].shape)}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] high native/final/coarse/base shape = "
                                f"{list(high_output['coarse_logits_native'].shape)}/"
                                f"{list(extract_logits(high_output).shape)}/"
                                f"{list(high_output['coarse_logits_68'].shape)}/"
                                f"{list(high_output['base_logits'].shape)}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] high_forward_microbatches/teacher_high_forward_count = "
                                f"{high_forward_calls}/0"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] normal/high probability mean/abs_diff = "
                                f"{cssd_batch_stats['normal_prob_mean']:.6f}/"
                                f"{cssd_batch_stats['high_prob_mean']:.6f}/"
                                f"{cssd_batch_stats['normal_high_prob_abs_diff']:.6f}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] confidence normal/high/adv = "
                                f"{cssd_batch_stats['normal_conf_mean']:.6f}/"
                                f"{cssd_batch_stats['high_conf_mean']:.6f}/"
                                f"{cssd_batch_stats['conf_adv_mean']:.6f}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] core/transfer_raw/transfer_cap/boundary ratio = "
                                f"{cssd_batch_stats['core_mask_ratio']:.6f}/"
                                f"{cssd_batch_stats['transfer_raw_ratio']:.6f}/"
                                f"{cssd_batch_stats['transfer_capped_ratio']:.6f}/"
                                f"{cssd_batch_stats['boundary_mask_ratio']:.6f}"
                            )
                            logger.log(
                                "[CSSD FirstActiveBatch] loss normal/high/dual/core/transfer/boundary/weighted/total = "
                                f"{float(loss_cssd_normal_group.detach().item()):.8f}/"
                                f"{float(loss_cssd_high_group.detach().item()):.8f}/"
                                f"{float(loss_cssd_dual_group.detach().item()):.8f}/"
                                f"{float(loss_cssd_core.detach().item()):.8f}/"
                                f"{float(loss_cssd_transfer.detach().item()):.8f}/"
                                f"{float(loss_cssd_boundary_consistency.detach().item()):.8f}/"
                                f"{float(loss_cssd_weighted.detach().item()):.8f}/"
                                f"{float(loss.detach().item()):.8f}"
                            )
                            cssd_first_active_batch_logged = True
                    elif use_cssd_train:
                        if cssd_high_forward_batches != 0 or cssd_high_forward_calls != 0:
                            raise RuntimeError("CSSD high forward occurred while cssd_scale=0.")
                        normal_prob_inactive = torch.sigmoid(student_logits.detach())
                        cssd_stat_batches += 1
                        cssd_epoch_sums["normal_prob_mean"] += float(
                            normal_prob_inactive.mean().item()
                        )
                        cssd_epoch_sums["normal_pred_area"] += float(
                            (normal_prob_inactive >= 0.5).float().mean().item()
                        )
                        cssd_epoch_sums["normal_conf_mean"] += float(
                            (2.0 * torch.abs(normal_prob_inactive - 0.5)).mean().item()
                        )
                        cssd_epoch_sums["loss_normal_group"] += float(
                            loss_cssd_normal_group.detach().item()
                        )
                        cssd_epoch_sums["loss_dual_group"] += float(
                            loss_cssd_normal_group.detach().item()
                        )
                    if use_hr_bfr(cfg):
                        loss_hr_bfr, hr_bfr_stats = build_hr_bfr_losses(
                            student_out,
                            batch,
                            cfg,
                            epoch,
                            pu_static_loss_weight,
                            pu_teacher_loss_weight,
                            teacher_binary,
                            device,
                        )
                        loss = loss + loss_hr_bfr
                        if not hr_bfr_first_batch_logged:
                            log_hr_bfr_first_batch(logger, image_136, student_out, hr_bfr_stats)
                            hr_bfr_first_batch_logged = True
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
                            rast_stats["teacher_routing_scale"],
                            apply_to_loss=teacher_routing_apply_flag(cfg, "base"),
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
                if use_tce:
                    lambda_tce_eff = float(tce_stats["lambda_tce_eff"])
                    if bool(getattr(cfg, "TCE_USE_LOWER_BOUND_LOSS", True)) and bool(getattr(cfg, "TCE_APPLY_TO_FINAL", True)):
                        loss = loss + lambda_tce_eff * loss_tce_final
                if use_lceg:
                    lambda_lceg_eff = float(lceg_stats["lambda_lceg_eff"])
                    loss_lceg = student_logits.sum() * 0.0
                    if bool(getattr(cfg, "LCEG_APPLY_TO_FINAL", True)):
                        loss_lceg = loss_lceg + loss_lceg_final
                    if bool(getattr(cfg, "LCEG_APPLY_TO_COARSE_AUX", True)):
                        loss_lceg = loss_lceg + float(getattr(cfg, "LCEG_COARSE_LOSS_WEIGHT", 0.50)) * loss_lceg_coarse
                    loss = loss + lambda_lceg_eff * loss_lceg

                loss_pa_weighted = student_logits.sum() * 0.0
                pa_stats = None
                if use_pa_dagp_train:
                    loss_pa_weighted, pa_stats = build_pa_dagp_aux_loss(
                        cfg,
                        epoch,
                        student_out,
                        batch,
                        device,
                    )
                    loss = loss + loss_pa_weighted
                    if pa_compare_original:
                        compare_required = {
                            "pa_original_coarse_logits_native",
                            "pa_original_coarse_logits_68",
                            "pa_original_final_logits_68",
                        }
                        missing_compare = sorted(compare_required - set(student_out))
                        if missing_compare:
                            raise RuntimeError(
                                f"PA-DAGP original-edge comparison output missing: {missing_compare}"
                            )
                        current_coarse_native = student_out["coarse_logits_native"].detach()
                        original_coarse_native = student_out["pa_original_coarse_logits_native"].detach()
                        pa_dagp_compare_coarse_abs_diff = float(
                            (current_coarse_native - original_coarse_native).abs().mean().item()
                        )
                        current_coarse_prob = student_out["coarse_logits_68"].detach().sigmoid()
                        original_coarse_prob = student_out["pa_original_coarse_logits_68"].detach().sigmoid()
                        current_final_prob = student_out["final_logits"].detach().sigmoid()
                        original_final_prob = student_out["pa_original_final_logits_68"].detach().sigmoid()
                        pa_dagp_compare_coarse_area_delta = float(
                            ((current_coarse_prob >= 0.5).float().mean() - (original_coarse_prob >= 0.5).float().mean()).item()
                        )
                        pa_dagp_compare_final_area_delta = float(
                            ((current_final_prob >= 0.5).float().mean() - (original_final_prob >= 0.5).float().mean()).item()
                        )
                    if (
                        not pa_dagp_first_active_batch_logged
                        and float(pa_stats["edge_scale"]) > 0.0
                    ):
                        logger.log("[PA-DAGP FirstActiveBatch]")
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] epoch/edge_scale/aux_scale/edge_cut_eff/lambda_pa_eff = "
                            f"{epoch}/{pa_stats['edge_scale']:.8f}/{pa_stats['aux_scale']:.8f}/"
                            f"{pa_stats['edge_cut_eff']:.8f}/{pa_stats['lambda_pa_eff']:.8f}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] feature/base/pol_feat/raw/signed shape = "
                            f"{list(model_input.shape)}/{list(student_out['base_logits_native'].shape)}/"
                            f"{[model_input.shape[0], int(getattr(cfg, 'PA_DAGP_POL_DIM', 32)), model_input.shape[-2], model_input.shape[-1]]}/"
                            f"{list(student_out['pa_raw_polarity'].shape)}/"
                            f"{list(student_out['pa_signed_polarity'].shape)}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] anchor_valid/fg_norm(mean,min,max)/bg_norm(mean,min,max) = "
                            f"{pa_stats['anchor_valid_ratio']:.6f}/"
                            f"({pa_stats['anchor_fg_norm_mean']:.6f},{pa_stats['anchor_fg_norm_min']:.6f},{pa_stats['anchor_fg_norm_max']:.6f})/"
                            f"({pa_stats['anchor_bg_norm_mean']:.6f},{pa_stats['anchor_bg_norm_min']:.6f},{pa_stats['anchor_bg_norm_max']:.6f})"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] anchor_cosine/rho mean,min,max = "
                            f"({pa_stats['anchor_cosine_mean']:.6f},{pa_stats['anchor_cosine_min']:.6f},{pa_stats['anchor_cosine_max']:.6f})/"
                            f"({pa_stats['rho_mean']:.6f},{pa_stats['rho_min']:.6f},{pa_stats['rho_max']:.6f})"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] raw_pol mean/std/min/max = "
                            f"{pa_stats['raw_pol_mean']:.6f}/{pa_stats['raw_pol_std']:.6f}/"
                            f"{pa_stats['raw_pol_min']:.6f}/{pa_stats['raw_pol_max']:.6f}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] signed_pol mean/std/min/max and pos/neg/ambig = "
                            f"{pa_stats['signed_pol_mean']:.6f}/{pa_stats['signed_pol_std']:.6f}/"
                            f"{pa_stats['signed_pol_min']:.6f}/{pa_stats['signed_pol_max']:.6f} | "
                            f"{pa_stats['positive_ratio']:.6f}/{pa_stats['negative_ratio']:.6f}/{pa_stats['ambiguous_ratio']:.6f}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] cross_edge/gate(mean,min,max)/suppressed = "
                            f"{pa_stats['cross_edge_ratio']:.6f}/"
                            f"({pa_stats['gate_mean']:.6f},{pa_stats['gate_min']:.6f},{pa_stats['gate_max']:.6f})/"
                            f"{pa_stats['edge_suppressed_ratio']:.6f}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] fg/bg/hard_fg/hard_bg polarity mean = "
                            f"{pa_stats['fg_core_pol_mean']:.6f}/{pa_stats['bg_core_pol_mean']:.6f}/"
                            f"{pa_stats['hard_fg_pol_mean']:.6f}/{pa_stats['hard_bg_pol_mean']:.6f}"
                        )
                        logger.log(
                            "[PA-DAGP FirstActiveBatch] loss fg/bg/core/hfg/hbg/hard/raw/weighted = "
                            f"{pa_stats['loss_pa_fg']:.8f}/{pa_stats['loss_pa_bg']:.8f}/"
                            f"{pa_stats['loss_pa_core']:.8f}/{pa_stats['loss_pa_hfg']:.8f}/"
                            f"{pa_stats['loss_pa_hbg']:.8f}/{pa_stats['loss_pa_hard']:.8f}/"
                            f"{pa_stats['loss_pa_raw']:.8f}/{pa_stats['loss_pa_weighted']:.8f}"
                        )
                        pa_dagp_first_active_batch_logged = True

                loss_cacd_anchor = student_logits.sum() * 0.0
                loss_cacd_anchor_weighted = student_logits.sum() * 0.0
                cacd_anchor_stats = None
                if use_cacd(cfg):
                    if not isinstance(student_out, dict) or "anchor_logits" not in student_out:
                        raise RuntimeError("CACD student output is missing anchor_logits.")
                    loss_cacd_seg = loss
                    loss_cacd_anchor, cacd_anchor_stats = build_cacd_anchor_partial_ce(
                        student_out["anchor_logits"], batch, cfg, device=device
                    )
                    loss_cacd_anchor_weighted = (
                        float(getattr(cfg, "CACD_ANCHOR_LOSS_WEIGHT", 0.05))
                        * loss_cacd_anchor
                    )
                    loss = loss_cacd_seg + loss_cacd_anchor_weighted
                    if not bool(torch.isfinite(loss_cacd_anchor).item()):
                        raise RuntimeError("CACD anchor partial CE is NaN/Inf.")
                    cacd_aux = student_out.get("cacd_aux")
                    if not isinstance(cacd_aux, dict):
                        raise RuntimeError("CACD output is missing detached scalar cacd_aux diagnostics.")
                    for name, value in cacd_aux.items():
                        scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
                        if not math.isfinite(scalar):
                            raise RuntimeError(f"CACD diagnostic {name} is non-finite.")
                        cacd_aux_sums[name] = cacd_aux_sums.get(name, 0.0) + scalar
                    for name, value in cacd_anchor_stats.items():
                        scalar = float(value)
                        if not math.isfinite(scalar):
                            raise RuntimeError(f"CACD anchor diagnostic {name} is non-finite.")
                        cacd_anchor_sums[name] = cacd_anchor_sums.get(name, 0.0) + scalar
                    cacd_loss_seg_sum += float(loss_cacd_seg.detach().item())
                    cacd_loss_anchor_sum += float(loss_cacd_anchor.detach().item())
                    cacd_loss_anchor_weighted_sum += float(loss_cacd_anchor_weighted.detach().item())
                    cacd_loss_total_sum += float(loss.detach().item())
                    cacd_stat_batches += 1
                    if not cacd_first_batch_logged:
                        log_cacd_first_batch(
                            logger,
                            model_input,
                            image_68,
                            sobel_68,
                            student_out,
                            cacd_anchor_stats,
                            loss_cacd_seg,
                            loss_cacd_anchor,
                            loss,
                            cfg,
                        )
                        cacd_first_batch_logged = True

                loss_ber_weighted = student_logits.sum() * 0.0
                ber_stats = None
                ber_aux = None
                if use_esa_ber:
                    loss_current_before_ber = loss
                    loss_ber_weighted, ber_stats, ber_aux = build_esa_ber_loss(
                        cfg,
                        epoch,
                        student_out,
                        student_logits,
                        batch,
                        teacher_prob,
                        teacher_binary,
                        rast_teacher_map_eff,
                        rast_stats,
                        device,
                    )
                    main_loss_abs = abs(float(loss_current_before_ber.detach().item()))
                    ber_ratio = float(loss_ber_weighted.detach().item()) / max(main_loss_abs, 1e-12)
                    ber_stats["ber_to_main_loss_ratio"] = ber_ratio
                    if ber_ratio > 0.30:
                        raise RuntimeError(
                            "ESA-v2-BER weighted loss exceeds 30% of the current main loss: "
                            f"epoch={epoch}, iter={iter_idx}, ratio={ber_ratio:.8f}."
                        )
                    if ber_ratio > 0.15:
                        esa_ber_loss_ratio_warning = True
                    loss = loss_current_before_ber + loss_ber_weighted

                    if (
                        not esa_ber_first_active_batch_logged
                        and float(ber_stats["ber_scale"]) > 0.0
                    ):
                        logger.log("[ESA-BER FirstActiveBatch]")
                        logger.log(
                            "[ESA-BER FirstActiveBatch] "
                            f"epoch={epoch} | ber_scale={ber_stats['ber_scale']:.6f} | "
                            f"lambda_ber_eff={ber_stats['lambda_ber_eff']:.6f}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] shapes | "
                            f"final_logits={list(student_logits.shape)} | "
                            f"teacher_prob={list(teacher_prob.shape)} | "
                            f"margin_68={list(ber_aux['margin_68'].shape)} | "
                            f"conn_delta_68={list(ber_aux['conn_delta_68'].shape)} | "
                            f"conn_support_68={list(ber_aux['conn_support_68'].shape)} | "
                            f"topk_idx={list(student_out['dagp_topk_idx'].shape)} | "
                            f"topk_sem_weight={list(student_out['dagp_topk_sem_weight'].shape)}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] evidence | "
                            f"topk_sem_weight_sum_error={ber_stats['topk_sem_weight_sum_error']:.8g} | "
                            f"proto_valid_ratio={ber_stats['proto_valid_ratio']:.6f} | "
                            f"graph_valid_ratio={ber_stats['graph_valid_ratio']:.6f}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] candidates | "
                            f"pos_raw_ratio={ber_stats['pos_raw_ratio']:.8f} | "
                            f"neg_extent_raw_ratio={ber_stats['neg_extent_raw_ratio']:.8f} | "
                            f"neg_hard_bg_raw_ratio={ber_stats['neg_hard_bg_raw_ratio']:.8f} | "
                            f"neg_raw_ratio={ber_stats['neg_raw_ratio']:.8f} | "
                            f"valid_image_ratio={ber_stats['valid_image_ratio']:.6f} | "
                            f"selected_pairs_mean/min/max={ber_stats['selected_pairs_mean']:.4f}/"
                            f"{ber_stats['selected_pairs_min']:.0f}/{ber_stats['selected_pairs_max']:.0f}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] selected positive | "
                            f"margin={ber_stats['pos_margin_mean']:.6f} | "
                            f"conn={ber_stats['pos_conn_mean']:.6f} | "
                            f"student_prob={ber_stats['pos_student_prob_mean']:.6f} | "
                            f"teacher_prob={ber_stats['pos_teacher_prob_mean']:.6f}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] selected negative | "
                            f"margin={ber_stats['neg_margin_mean']:.6f} | "
                            f"conn={ber_stats['neg_conn_mean']:.6f} | "
                            f"student_prob={ber_stats['neg_student_prob_mean']:.6f} | "
                            f"teacher_prob={ber_stats['neg_teacher_prob_mean']:.6f}"
                        )
                        logger.log(
                            "[ESA-BER FirstActiveBatch] ranking | "
                            f"pos_logit={ber_stats['pos_logit_mean']:.6f} | "
                            f"neg_logit={ber_stats['neg_logit_mean']:.6f} | "
                            f"gap={ber_stats['logit_gap']:.6f} | "
                            f"violation={ber_stats['rank_violation_ratio']:.6f} | "
                            f"loss_raw={ber_stats['loss_ber_raw']:.8f} | "
                            f"loss_weighted={ber_stats['loss_ber_weighted']:.8f} | "
                            f"ber_to_main={ber_stats['ber_to_main_loss_ratio']:.8f}"
                        )
                        esa_ber_first_active_batch_logged = True

                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(
                        f"Training loss is NaN/Inf at epoch={epoch}, iter={iter_idx}; "
                        f"cssd_scale={cssd_scale_epoch:.8f}."
                    )

                if use_tepr_lite:
                    update_start = int(getattr(cfg, "TEPR_MEMORY_UPDATE_START_EPOCH", 1))
                    update_end = int(getattr(cfg, "TEPR_MEMORY_UPDATE_END_EPOCH", 20))
                    if update_start <= int(epoch) <= update_end:
                        if tepr_memory is None or tepr_sample_indices is None:
                            raise RuntimeError("TEPR memory/index is unavailable during its update window.")
                        with torch.no_grad():
                            tepr_memory.update(
                                indices=tepr_sample_indices,
                                teacher_prob=teacher_prob.detach(),
                                rho=float(getattr(cfg, "TEPR_TEMPORAL_RHO", 0.90)),
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

                if not use_cssd_train:
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
                total_ndr_v2_bg_lock_loss += float(loss_ndr_v2_bg_lock.item())
                total_ndr_v2_weighted_loss += float(loss_ndr_v2_weighted.item())
                if use_esa_ber:
                    if ber_stats is None:
                        raise RuntimeError("ESA-v2-BER stats are missing for an enabled batch.")
                    for key in esa_ber_stat_keys:
                        esa_ber_sums[key] += float(ber_stats[key])
                    pair_min_value = float(ber_stats["selected_pairs_min"])
                    pair_max_value = float(ber_stats["selected_pairs_max"])
                    esa_ber_pairs_min = (
                        pair_min_value
                        if esa_ber_pairs_min is None
                        else min(esa_ber_pairs_min, pair_min_value)
                    )
                    esa_ber_pairs_max = (
                        pair_max_value
                        if esa_ber_pairs_max is None
                        else max(esa_ber_pairs_max, pair_max_value)
                    )
                    esa_ber_topk_sum_error_max = max(
                        esa_ber_topk_sum_error_max,
                        float(ber_stats["topk_sem_weight_sum_error"]),
                    )
                    for source, source_stats in ber_stats["source_stats"].items():
                        accumulator = esa_ber_source_sums.setdefault(
                            source,
                            {
                                "images": 0,
                                "valid_images": 0.0,
                                "pos_raw_ratio": 0.0,
                                "neg_raw_ratio": 0.0,
                                "selected_pairs": 0.0,
                                "logit_gap": 0.0,
                                "rank_violation_ratio": 0.0,
                            },
                        )
                        image_count = int(source_stats["images"])
                        accumulator["images"] += image_count
                        accumulator["valid_images"] += float(source_stats["valid_image_ratio"]) * image_count
                        for key in (
                            "pos_raw_ratio",
                            "neg_raw_ratio",
                            "selected_pairs_mean",
                            "logit_gap",
                            "rank_violation_ratio",
                        ):
                            target_key = "selected_pairs" if key == "selected_pairs_mean" else key
                            accumulator[target_key] += float(source_stats[key]) * image_count
                    esa_ber_stat_batches += 1
                if use_pa_dagp_train:
                    if pa_stats is None:
                        raise RuntimeError("PA-DAGP stats are missing for an enabled batch.")
                    for key in pa_dagp_stat_keys:
                        pa_dagp_epoch_sums[key] += float(pa_stats[key])
                    gate_min_value = float(pa_stats["gate_min"])
                    gate_max_value = float(pa_stats["gate_max"])
                    pa_dagp_gate_min = (
                        gate_min_value
                        if pa_dagp_gate_min is None
                        else min(pa_dagp_gate_min, gate_min_value)
                    )
                    pa_dagp_gate_max = (
                        gate_max_value
                        if pa_dagp_gate_max is None
                        else max(pa_dagp_gate_max, gate_max_value)
                    )
                    pa_dagp_stat_batches += 1
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
                    if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() == "pu_v12_shape_complete":
                        dabe_pu_v12_target_base_mean_sum += float(batch["pu_target_base"].float().mean().item())
                        dabe_pu_v12_weight_base_mean_sum += float(batch["pu_weight_base"].float().mean().item())
                        dabe_pu_v12_target_delta_mean_sum += float(
                            (batch["pu_target_soft"].float() - batch["pu_target_base"].float()).mean().item()
                        )
                        dabe_pu_v12_sc_bg_lock_sum += float(batch["pu_sc_bg_lock"].float().mean().item())
                        dabe_pu_v12_sc_extent_agree_sum += float(
                            batch["pu_sc_extent_agree_fg"].float().mean().item()
                        )
                        dabe_pu_v12_sc_lost_extent_sum += float(batch["pu_sc_lost_extent"].float().mean().item())
                        dabe_pu_v12_sc_new_boundary_sum += float(batch["pu_sc_new_boundary"].float().mean().item())
                    dabe_pu_stat_batches += 1
                    if use_tepr_lite:
                        if tepr_stats is None or tepr_epoch_accumulator is None:
                            raise RuntimeError("TEPR-Lite stats are missing for an enabled batch.")
                        add_tepr_prediction_stats(
                            tepr_stats,
                            batch,
                            student_logits,
                            teacher_prob,
                            device,
                        )
                        update_tepr_epoch_accumulator(tepr_epoch_accumulator, tepr_stats)
                    if use_rast:
                        rast_scale_sum += float(rast_stats["rast_scale"])
                        rast_pre_reset_scale_sum += float(rast_stats["rast_pre_reset_scale"])
                        rast_post_reset_scale_sum += float(rast_stats["rast_post_reset_scale"])
                        rast_scale_effective_sum += float(rast_stats["rast_scale_effective"])
                        teacher_routing_scale_sum += float(rast_stats["teacher_routing_scale"])
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
                        if use_esa_post_reset:
                            for key in esa_post_stat_keys:
                                esa_post_sums[key] += float(rast_stats[key])
                            for key in esa_post_valid_keys:
                                esa_post_valid_counts[key] += int(bool(rast_stats[key]))
                            esa_post_active_batches += int(bool(rast_stats["esa_post_reset_active"]))
                            post_map_min_value = float(rast_stats["esa_post_map_min"])
                            post_map_max_value = float(rast_stats["esa_post_map_max"])
                            esa_post_map_min = (
                                post_map_min_value
                                if esa_post_map_min is None
                                else min(esa_post_map_min, post_map_min_value)
                            )
                            esa_post_map_max = (
                                post_map_max_value
                                if esa_post_map_max is None
                                else max(esa_post_map_max, post_map_max_value)
                            )
                            esa_post_stat_batches += 1
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
                    if use_tce:
                        tce_scale_sum += float(tce_stats["tce_scale"])
                        tce_lambda_sum += float(tce_stats["lambda_tce_eff"])
                        tce_cover_area_sum += float(tce_stats["cover_area_mean"])
                        tce_current_area_sum += float(tce_stats["current_area_mean"])
                        tce_shrink_gate_ratio_sum += float(tce_stats["shrink_gate_ratio"])
                        tce_bg_safe_gate_ratio_sum += float(tce_stats["bg_safe_gate_ratio"])
                        tce_image_gate_ratio_sum += float(tce_stats["image_gate_ratio"])
                        tce_lost_raw_ratio_sum += float(tce_stats["lost_raw_ratio"])
                        tce_lost_capped_ratio_sum += float(tce_stats["lost_capped_ratio"])
                        tce_new_raw_ratio_sum += float(tce_stats["new_raw_ratio"])
                        tce_new_capped_ratio_sum += float(tce_stats["new_capped_ratio"])
                        tce_total_ratio_sum += float(tce_stats["tce_total_ratio"])
                        tce_valid_image_ratio_sum += float(tce_stats["tce_valid_image_ratio"])
                        tce_lost_margin_mean_sum += float(tce_stats["lost_margin_mean"])
                        tce_new_margin_mean_sum += float(tce_stats["new_margin_mean"])
                        tce_lost_cover_conf_mean_sum += float(tce_stats["lost_cover_conf_mean"])
                        tce_current_teacher_conf_lost_mean_sum += float(tce_stats["current_teacher_conf_lost_mean"])
                        tce_current_teacher_conf_new_mean_sum += float(tce_stats["current_teacher_conf_new_mean"])
                        tce_teacher_map_final_mean_sum += float(tce_stats["teacher_map_final_mean"])
                        tce_teacher_map_final_tce_mean_sum += float(tce_stats["teacher_map_final_tce_mean"])
                        tce_loss_sum += float(loss_tce_final.detach().item())
                        tce_skipped_no_fg_proto_sum += int(tce_stats.get("tce_skipped_no_fg_proto", 0))
                        tce_skipped_no_bg_proto_sum += int(tce_stats.get("tce_skipped_no_bg_proto", 0))
                        tce_stat_batches += 1
                    if use_lceg:
                        lceg_scale_sum += float(lceg_stats["lceg_scale"])
                        lceg_lambda_sum += float(lceg_stats["lambda_lceg_eff"])
                        lceg_cover_area_sum += float(lceg_stats["cover_area_mean"])
                        lceg_current_area_sum += float(lceg_stats["current_area_mean"])
                        lceg_current_fg_extent_ratio_sum += float(lceg_stats["current_fg_extent_ratio"])
                        lceg_cover_fg_extent_ratio_sum += float(lceg_stats["cover_fg_extent_ratio"])
                        lceg_shrink_gate_ratio_sum += float(lceg_stats["shrink_gate_ratio"])
                        lceg_bg_safe_gate_ratio_sum += float(lceg_stats["bg_safe_gate_ratio"])
                        lceg_core_raw_ratio_sum += float(lceg_stats["core_raw_ratio"])
                        lceg_core_capped_ratio_sum += float(lceg_stats["core_capped_ratio"])
                        lceg_lost_raw_ratio_sum += float(lceg_stats["lost_raw_ratio"])
                        lceg_lost_capped_ratio_sum += float(lceg_stats["lost_capped_ratio"])
                        lceg_new_raw_ratio_sum += float(lceg_stats["new_raw_ratio"])
                        lceg_new_capped_ratio_sum += float(lceg_stats["new_capped_ratio"])
                        lceg_total_ratio_sum += float(lceg_stats["lceg_total_ratio"])
                        lceg_valid_image_ratio_sum += float(lceg_stats["lceg_valid_image_ratio"])
                        lceg_core_conf_mean_sum += float(lceg_stats["core_conf_mean"])
                        lceg_lost_cover_prob_mean_sum += float(lceg_stats["lost_cover_prob_mean"])
                        lceg_lost_margin_mean_sum += float(lceg_stats["lost_margin_mean"])
                        lceg_new_margin_mean_sum += float(lceg_stats["new_margin_mean"])
                        lceg_teacher_map_final_mean_sum += float(lceg_stats["teacher_map_final_mean"])
                        lceg_teacher_map_final_lceg_mean_sum += float(lceg_stats["teacher_map_final_lceg_mean"])
                        lceg_teacher_map_coarse_mean_sum += float(lceg_stats["teacher_map_coarse_mean"])
                        lceg_teacher_map_coarse_lceg_mean_sum += float(lceg_stats["teacher_map_coarse_lceg_mean"])
                        lceg_loss_core_sum += float(loss_lceg_final_parts["core"].detach().item())
                        lceg_loss_lost_sum += float(loss_lceg_final_parts["lost"].detach().item())
                        lceg_loss_new_sum += float(loss_lceg_final_parts["new"].detach().item())
                        lceg_loss_final_sum += float(loss_lceg_final.detach().item())
                        lceg_loss_coarse_sum += float(loss_lceg_coarse.detach().item())
                        lceg_loss_sum += float(loss_lceg.detach().item())
                        lceg_skipped_no_fg_proto_sum += int(lceg_stats.get("lceg_skipped_no_fg_proto", 0))
                        lceg_skipped_no_bg_proto_sum += int(lceg_stats.get("lceg_skipped_no_bg_proto", 0))
                        lceg_stat_batches += 1
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
                    if use_csd_head(cfg) or use_csd_v1r_head(cfg):
                        csd_scale_sum += output_scalar(student_out, "csd_scale")
                        csd_beta_sum += output_scalar(student_out, "csd_beta_eff")
                        csd_gate_mean = output_scalar(student_out, "csd_detail_gate_mean")
                        csd_gate_min_value = output_scalar(student_out, "csd_detail_gate_min")
                        csd_gate_max_value = output_scalar(student_out, "csd_detail_gate_max")
                        csd_residual_abs_mean_sum += output_scalar(student_out, "csd_residual_abs_mean")
                        csd_residual_abs_max_value = output_scalar(student_out, "csd_residual_abs_max")
                        csd_final_minus_coarse_abs_mean_sum += output_scalar(
                            student_out,
                            "csd_final_minus_coarse_abs_mean",
                            0.0,
                        )
                        csd_gate_mean_sum += csd_gate_mean
                        csd_gate_min = (
                            csd_gate_min_value
                            if csd_gate_min is None
                            else min(csd_gate_min, csd_gate_min_value)
                        )
                        csd_gate_max = (
                            csd_gate_max_value
                            if csd_gate_max is None
                            else max(csd_gate_max, csd_gate_max_value)
                        )
                        csd_residual_abs_max = (
                            csd_residual_abs_max_value
                            if csd_residual_abs_max is None
                            else max(csd_residual_abs_max, csd_residual_abs_max_value)
                        )
                        csd_bg_reliable_ratio_sum += float(csd_stats["bg_reliable_ratio"])
                        csd_loss_bg_detail_sum += float(csd_stats["loss_bg_detail"])
                        csd_loss_bg_detail_weighted_sum += float(csd_stats["loss_bg_detail_weighted"])
                        csd_boundary_scale_sum += float(csd_stats["boundary_scale"])
                        csd_boundary_target_mean_sum += float(csd_stats["boundary_target_mean"])
                        csd_loss_boundary_sum += float(csd_stats["loss_boundary"])
                        csd_loss_boundary_weighted_sum += float(csd_stats["loss_boundary_weighted"])
                        csd_stat_batches += 1
                    if use_hr_bfr(cfg):
                        hr_bfr_scale_sum += float(hr_bfr_stats["hr_scale"])
                        hr_bfr_beta_sum += float(hr_bfr_stats["hr_beta_eff"])
                        hr_bfr_band_ratio_sum += float(hr_bfr_stats["band_ratio"])
                        hr_bfr_band_ratio_max = (
                            float(hr_bfr_stats["band_ratio_max"])
                            if hr_bfr_band_ratio_max is None
                            else max(hr_bfr_band_ratio_max, float(hr_bfr_stats["band_ratio_max"]))
                        )
                        hr_bfr_valid_img_ratio_sum += float(hr_bfr_stats["valid_img_ratio"])
                        hr_bfr_skip_img_ratio_sum += float(hr_bfr_stats["skip_img_ratio"])
                        hr_bfr_active_pixel_ratio_sum += float(hr_bfr_stats["hr_active_pixel_ratio"])
                        hr_bfr_anchor_area_sum += float(hr_bfr_stats["anchor_area"])
                        hr_bfr_hr_area_sum += float(hr_bfr_stats["hr_area"])
                        hr_bfr_area_delta_sum += float(hr_bfr_stats["area_delta"])
                        hr_bfr_minus_anchor_sum += float(hr_bfr_stats["hr_minus_anchor_abs_mean"])
                        hr_bfr_residual_abs_mean_sum += float(hr_bfr_stats["hr_residual_abs_mean"])
                        hr_residual_abs_max = float(hr_bfr_stats["hr_residual_abs_max"])
                        hr_bfr_residual_abs_max = (
                            hr_residual_abs_max
                            if hr_bfr_residual_abs_max is None
                            else max(hr_bfr_residual_abs_max, hr_residual_abs_max)
                        )
                        hr_bfr_bg_reliable_ratio_sum += float(hr_bfr_stats["bg_reliable_ratio"])
                        hr_bfr_edge_support_sum += float(hr_bfr_stats["edge_support_mean"])
                        hr_bfr_loss_band_sum += float(hr_bfr_stats["loss_band_bce"])
                        hr_bfr_loss_outband_sum += float(hr_bfr_stats["loss_outband_anchor"])
                        hr_bfr_loss_bg_sum += float(hr_bfr_stats["loss_bg_prob_lock"])
                        hr_bfr_loss_area_sum += float(hr_bfr_stats["loss_area_neutral"])
                        hr_bfr_loss_edge_sum += float(hr_bfr_stats["loss_edge_align"])
                        hr_bfr_loss_total_sum += float(hr_bfr_stats["loss_hr_bfr"])
                        hr_bfr_stat_batches += 1
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
                        if use_ndr_v2(cfg):
                            ndr_v2_shape_alpha_sum += output_scalar(student_out, "ndr_v2_shape_alpha_eff")
                            ndr_v2_gate_v1_mean_sum += output_scalar(student_out, "ndr_v2_detail_gate_v1_mean")
                            ndr_v2_gate_v2_mean_sum += output_scalar(student_out, "ndr_v2_detail_gate_v2_mean")
                            ndr_v2_boundary_mean = output_scalar(student_out, "ndr_v2_boundary_mean")
                            ndr_v2_boundary_min_value = output_scalar(student_out, "ndr_v2_boundary_min")
                            ndr_v2_boundary_max_value = output_scalar(student_out, "ndr_v2_boundary_max")
                            ndr_v2_edge_norm_mean_sum += output_scalar(student_out, "ndr_v2_edge_norm_mean")
                            ndr_v2_shape_boost_mean_sum += output_scalar(student_out, "ndr_v2_shape_boost_mean")
                            ndr_v2_shape_boost_max_value = output_scalar(student_out, "ndr_v2_shape_boost_max")
                            ndr_v2_boundary_mean_sum += ndr_v2_boundary_mean
                            ndr_v2_boundary_min = (
                                ndr_v2_boundary_min_value
                                if ndr_v2_boundary_min is None
                                else min(ndr_v2_boundary_min, ndr_v2_boundary_min_value)
                            )
                            ndr_v2_boundary_max = (
                                ndr_v2_boundary_max_value
                                if ndr_v2_boundary_max is None
                                else max(ndr_v2_boundary_max, ndr_v2_boundary_max_value)
                            )
                            ndr_v2_shape_boost_max = (
                                ndr_v2_shape_boost_max_value
                                if ndr_v2_shape_boost_max is None
                                else max(ndr_v2_shape_boost_max, ndr_v2_shape_boost_max_value)
                            )
                            ndr_v2_shape_lb_scale_sum += float(ndr_v2_shape_stats["shape_lb_scale"])
                            ndr_v2_shape_lb_lambda_sum += float(ndr_v2_shape_stats["lambda_shape_lb_eff"])
                            ndr_v2_shape_candidate_raw_ratio_sum += float(
                                ndr_v2_shape_stats["shape_candidate_raw_ratio"]
                            )
                            ndr_v2_shape_candidate_capped_ratio_sum += float(
                                ndr_v2_shape_stats["shape_candidate_capped_ratio"]
                            )
                            ndr_v2_shape_valid_image_ratio_sum += float(
                                ndr_v2_shape_stats["shape_valid_image_ratio"]
                            )
                            ndr_v2_shape_margin_mean_sum += float(ndr_v2_shape_stats["shape_margin_mean"])
                            ndr_v2_shape_edge_mean_sum += float(ndr_v2_shape_stats["shape_edge_mean"])
                            ndr_v2_shape_under_floor_mean_sum += float(
                                ndr_v2_shape_stats["shape_under_floor_mean"]
                            )
                            ndr_v2_bg_lock_area_sum += float(ndr_v2_bg_lock_stats["bg_lock_area"])
                            ndr_v2_bg_core_area_sum += float(ndr_v2_bg_lock_stats["bg_core_area"])
                            ndr_v2_low_target_bg_area_sum += float(ndr_v2_bg_lock_stats["low_target_bg_area"])
                            ndr_v2_bg_res_lock_scale_sum += float(ndr_v2_bg_lock_stats["bg_res_lock_scale"])
                            ndr_v2_bg_res_lock_lambda_sum += float(
                                ndr_v2_bg_lock_stats["lambda_bg_res_lock_eff"]
                            )
                            ndr_v2_bg_prob_lock_scale_sum += float(ndr_v2_bg_lock_stats["bg_prob_lock_scale"])
                            ndr_v2_bg_prob_lock_lambda_sum += float(
                                ndr_v2_bg_lock_stats["lambda_bg_prob_lock_eff"]
                            )
                            ndr_v2_positive_delta_bg_mean_sum += float(ndr_v2_bg_lock_stats["positive_delta_bg_mean"])
                            ndr_v2_positive_delta_bg_max_value = float(ndr_v2_bg_lock_stats["positive_delta_bg_max"])
                            ndr_v2_positive_delta_bg_max = (
                                ndr_v2_positive_delta_bg_max_value
                                if ndr_v2_positive_delta_bg_max is None
                                else max(ndr_v2_positive_delta_bg_max, ndr_v2_positive_delta_bg_max_value)
                            )
                            ndr_v2_loss_shape_lb_sum += float(loss_ndr_v2_shape_lb.detach().item())
                            ndr_v2_loss_bg_lock_sum += float(loss_ndr_v2_bg_lock.detach().item())
                            ndr_v2_loss_bg_prob_lock_sum += float(loss_ndr_v2_bg_prob_lock.detach().item())
                            ndr_v2_weighted_loss_sum += float(loss_ndr_v2_weighted.detach().item())
                            ndr_v2_stat_batches += 1
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
            if use_cacd(cfg) and cacd_stat_batches > 0:
                cb = float(cacd_stat_batches)
                aux_avg = {name: value / cb for name, value in cacd_aux_sums.items()}
                anchor_avg = {name: value / cb for name, value in cacd_anchor_sums.items()}
                seg_avg = cacd_loss_seg_sum / cb
                anchor_loss_avg = cacd_loss_anchor_sum / cb
                anchor_weighted_avg = cacd_loss_anchor_weighted_sum / cb
                anchor_ratio = anchor_weighted_avg / max(abs(seg_avg), 1e-12)
                logger.log(
                    f"[CACD] epoch={epoch:03d} | "
                    f"alpha10={aux_avg.get('alpha10', 0.0):.8f} | alpha11={aux_avg.get('alpha11', 0.0):.8f} | "
                    f"gate10_mean={aux_avg.get('gate10_mean', 0.0):.6f} | gate11_mean={aux_avg.get('gate11_mean', 0.0):.6f} | "
                    f"cos10_mean={aux_avg.get('cos10_mean', 0.0):.6f} | cos11_mean={aux_avg.get('cos11_mean', 0.0):.6f} | "
                    f"qc_mean={aux_avg.get('q_consensus_mean', 0.0):.6f} | "
                    f"anchor_fg_mean={aux_avg.get('anchor_fg_mean', 0.0):.6f} | "
                    f"anchor_bg_mean={aux_avg.get('anchor_bg_mean', 0.0):.6f} | "
                    f"anchor_amb_mean={aux_avg.get('anchor_amb_mean', 0.0):.6f} | "
                    f"anchor_fg_core_prob={anchor_avg.get('anchor_fg_core_prob', 0.0):.6f} | "
                    f"anchor_bg_core_prob={anchor_avg.get('anchor_bg_core_prob', 0.0):.6f} | "
                    f"anchor_fg_core_acc={anchor_avg.get('anchor_fg_core_acc', 0.0):.6f} | "
                    f"anchor_bg_core_acc={anchor_avg.get('anchor_bg_core_acc', 0.0):.6f} | "
                    f"anchor_balanced_acc={anchor_avg.get('anchor_balanced_acc', 0.0):.6f} | "
                    f"anchor_fg_extent_mean={anchor_avg.get('anchor_fg_extent_mean', 0.0):.6f} | "
                    f"anchor_bg_extent_mean={anchor_avg.get('anchor_bg_extent_mean', 0.0):.6f} | "
                    f"anchor_amb_extent_mean={anchor_avg.get('anchor_amb_extent_mean', 0.0):.6f} | "
                    f"fg_slot_cos_mean={aux_avg.get('fg_slot_cos_mean', 0.0):.6f} | "
                    f"fg_slot_cos_max={aux_avg.get('fg_slot_cos_max', 0.0):.6f} | "
                    f"bg_slot_cos_mean={aux_avg.get('bg_slot_cos_mean', 0.0):.6f} | "
                    f"bg_slot_cos_max={aux_avg.get('bg_slot_cos_max', 0.0):.6f} | "
                    f"fg_attn_overlap_mean={aux_avg.get('fg_attn_overlap_mean', 0.0):.6f} | "
                    f"bg_attn_overlap_mean={aux_avg.get('bg_attn_overlap_mean', 0.0):.6f} | "
                    f"context_gate_mean={aux_avg.get('context_gate_mean', 0.0):.6f} | "
                    f"context_delta_abs_mean={aux_avg.get('context_delta_abs_mean', 0.0):.8f} | "
                    f"context_delta_norm_ratio={aux_avg.get('context_delta_norm_ratio', 0.0):.6f} | "
                    f"base_coarse_abs_diff={aux_avg.get('base_coarse_abs_diff', 0.0):.6f} | "
                    f"coarse_final_abs_diff={aux_avg.get('coarse_final_abs_diff', 0.0):.6f} | "
                    f"detail_gate_mean={aux_avg.get('detail_gate_mean', 0.0):.6f} | "
                    f"loss_seg={seg_avg:.8f} | loss_anchor={anchor_loss_avg:.8f} | "
                    f"loss_anchor_weighted={anchor_weighted_avg:.8f} | "
                    f"anchor_to_seg_loss_ratio={anchor_ratio:.6f} | "
                    f"loss_total={cacd_loss_total_sum / cb:.8f} | "
                    f"student_pred_area={student_pred_area_sum / stat_batches:.6f} | "
                    f"teacher_pred_area={teacher_pred_area_sum / stat_batches:.6f} | "
                    f"student_prob_fg_core={esa_student_prob_fg_core_sum / max(esa_stat_batches, 1):.6f} | "
                    f"student_prob_bg_core={esa_student_prob_bg_core_sum / max(esa_stat_batches, 1):.6f} | "
                    f"student_prob_extent={esa_student_prob_extent_sum / max(esa_stat_batches, 1):.6f} | "
                    f"teacher_fg_extent={esa_teacher_fg_extent_sum / max(esa_stat_batches, 1):.6f}"
                )
                warning_conditions = {
                    "fusion_noop": aux_avg.get("alpha10", 0.0) < 0.005 and aux_avg.get("alpha11", 0.0) < 0.005,
                    "gate10_open": aux_avg.get("gate10_mean", 0.0) > 0.95,
                    "gate11_open": aux_avg.get("gate11_mean", 0.0) > 0.95,
                    "gate10_closed": aux_avg.get("gate10_mean", 0.0) < 0.05,
                    "gate11_closed": aux_avg.get("gate11_mean", 0.0) < 0.05,
                    "anchor_fg_collapse": aux_avg.get("anchor_fg_mean", 0.0) > 0.90,
                    "anchor_bg_collapse": aux_avg.get("anchor_bg_mean", 0.0) > 0.95,
                    "anchor_amb_collapse": aux_avg.get("anchor_amb_mean", 0.0) > 0.95,
                    "anchor_learning_failed": epoch > 10 and anchor_avg.get("anchor_balanced_acc", 1.0) < 0.70,
                    "slot_collapse": max(aux_avg.get("fg_slot_cos_max", 0.0), aux_avg.get("bg_slot_cos_max", 0.0)) > 0.98,
                    "attention_collapse": max(aux_avg.get("fg_attn_overlap_mean", 0.0), aux_avg.get("bg_attn_overlap_mean", 0.0)) > 0.90,
                    "context_noop": epoch > 10 and aux_avg.get("context_delta_abs_mean", 1.0) < 1e-4,
                    "context_explode": aux_avg.get("context_delta_norm_ratio", 0.0) > 1.0,
                    "anchor_loss_strong": anchor_ratio > 0.25,
                }
                for name, active in warning_conditions.items():
                    cacd_warning_streaks[name] = cacd_warning_streaks.get(name, 0) + 1 if active else 0
                    if cacd_warning_streaks[name] == 3:
                        logger.log(f"[CACD WARNING] condition={name} persisted for 3 epochs; no automatic adjustment applied.")
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
                    if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() == "pu_v12_shape_complete":
                        logger.log(
                            f"[DABE-PU++] epoch={epoch:03d} | "
                            f"target_v12_mean={dabe_pu_target_mean_sum / stat_batches:.6f} | "
                            f"target_base_mean={dabe_pu_v12_target_base_mean_sum / stat_batches:.6f} | "
                            f"target_delta_mean={dabe_pu_v12_target_delta_mean_sum / stat_batches:.6f} | "
                            f"weight_v12_mean={dabe_pu_weight_mean_sum / stat_batches:.6f} | "
                            f"weight_base_mean={dabe_pu_v12_weight_base_mean_sum / stat_batches:.6f} | "
                            f"sc_bg_lock_ratio={dabe_pu_v12_sc_bg_lock_sum / stat_batches:.6f} | "
                            f"sc_extent_agree_ratio={dabe_pu_v12_sc_extent_agree_sum / stat_batches:.6f} | "
                            f"sc_lost_extent_ratio={dabe_pu_v12_sc_lost_extent_sum / stat_batches:.6f} | "
                            f"sc_new_boundary_ratio={dabe_pu_v12_sc_new_boundary_sum / stat_batches:.6f} | "
                            f"fg_core_ratio={dabe_pu_fg_core_mean_sum / stat_batches:.6f} | "
                            f"bg_core_ratio={dabe_pu_bg_core_mean_sum / stat_batches:.6f} | "
                            f"extent_ratio={dabe_pu_extent_mean_sum / stat_batches:.6f} | "
                            f"unknown_ratio={dabe_pu_unknown_mean_sum / stat_batches:.6f}"
                        )
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
                if use_tepr_lite:
                    tepr_batches = max(int(tepr_epoch_accumulator["batches"]), 1)
                    tepr_pixels = max(int(tepr_epoch_accumulator["map_pixels"]), 1)
                    tepr_region_means = {}
                    for region_name in tepr_epoch_accumulator["region_sums"]:
                        region_count = float(tepr_epoch_accumulator["region_counts"][region_name])
                        tepr_region_means[region_name] = (
                            float(tepr_epoch_accumulator["region_sums"][region_name]) / region_count
                            if region_count > 0.0
                            else 1.0
                        )
                    logger.log(
                        f"[TEPR-Lite] epoch={epoch:03d} | "
                        f"tepr_scale={float(get_tepr_scale(cfg, epoch)):.8f} | "
                        f"memory_active_ratio={tepr_epoch_accumulator['memory_active_batches'] / tepr_batches:.6f} | "
                        f"history_count_mean={tepr_epoch_accumulator['history_count_mean_sum'] / tepr_batches:.6f} | "
                        f"temporal_var_mean={tepr_epoch_accumulator['temporal_var_sum'] / tepr_batches:.8f} | "
                        f"temporal_var_p90={temporal_variance_p90_from_hist(tepr_epoch_accumulator['variance_hist']):.8f} | "
                        f"temporal_reliability_mean={tepr_epoch_accumulator['temporal_reliability_sum'] / tepr_batches:.6f} | "
                        f"core_conflict_mean={tepr_epoch_accumulator['core_conflict_sum'] / tepr_batches:.6f} | "
                        f"extent_conflict_mean={tepr_epoch_accumulator['extent_conflict_sum'] / tepr_batches:.6f} | "
                        f"teacher_map_mean={tepr_epoch_accumulator['map_sum'] / tepr_pixels:.6f} | "
                        f"teacher_map_min={(tepr_epoch_accumulator['map_min'] if tepr_epoch_accumulator['map_min'] is not None else 1.0):.6f} | "
                        f"teacher_map_max={(tepr_epoch_accumulator['map_max'] if tepr_epoch_accumulator['map_max'] is not None else 1.0):.6f} | "
                        f"map_fg/bg/extent/unknown/other={tepr_region_means['fg_core']:.6f}/"
                        f"{tepr_region_means['bg_core']:.6f}/{tepr_region_means['extent']:.6f}/"
                        f"{tepr_region_means['unknown']:.6f}/{tepr_region_means['other']:.6f} | "
                        f"map_lt_0.30_ratio={tepr_epoch_accumulator['map_lt_030'] / tepr_pixels:.8f} | "
                        f"map_lt_0.50_ratio={tepr_epoch_accumulator['map_lt_050'] / tepr_pixels:.8f} | "
                        f"map_gt_0.90_ratio={tepr_epoch_accumulator['map_gt_090'] / tepr_pixels:.8f} | "
                        f"teacher_loss_final={dabe_pu_teacher_final_loss_sum / stat_batches:.6f} | "
                        f"teacher_loss_coarse={dabe_pu_teacher_coarse_loss_sum / stat_batches:.6f} | "
                        f"teacher_loss_base={dabe_pu_teacher_base_loss_sum / stat_batches:.6f}"
                    )
                    if str(getattr(cfg, "TEPR_ROUTING_MODE", "legacy_v1")).lower() == "state_conditional_asymneg":
                        conditional_means = {}
                        conditional_valid = {}
                        for name in tepr_epoch_accumulator["conditional_sums"]:
                            count = float(tepr_epoch_accumulator["conditional_counts"][name])
                            conditional_valid[name] = count > 0.0
                            conditional_means[name] = (
                                float(tepr_epoch_accumulator["conditional_sums"][name]) / count
                                if count > 0.0
                                else 0.0
                            )
                        fg_core_count = int(tepr_epoch_accumulator["fg_core_count"])
                        bg_core_count = int(tepr_epoch_accumulator["bg_core_count"])
                        extent_count = int(tepr_epoch_accumulator["extent_count"])
                        prediction_pixels = max(
                            int(tepr_epoch_accumulator["prediction_pixels"]), 1
                        )
                        reliability_hist = tepr_epoch_accumulator[
                            "extent_bg_reliability_hist"
                        ]
                        logger.log(
                            f"[TEPR-Lite-v1.1] epoch={epoch:03d} | "
                            f"fg_core_conflict_ratio={tepr_epoch_accumulator['fg_core_conflict_count'] / max(fg_core_count, 1):.6f} | "
                            f"bg_core_conflict_ratio={tepr_epoch_accumulator['bg_core_conflict_count'] / max(bg_core_count, 1):.6f} | "
                            f"core_conflict_map_mean={conditional_means['core_conflict_map']:.6f} | "
                            f"core_no_conflict_map_mean={conditional_means['core_no_conflict_map']:.6f} | "
                            f"extent_teacher_fg_ratio={tepr_epoch_accumulator['extent_teacher_fg_count'] / max(extent_count, 1):.6f} | "
                            f"extent_teacher_bg_ratio={tepr_epoch_accumulator['extent_teacher_bg_count'] / max(extent_count, 1):.6f} | "
                            f"extent_teacher_fg_map_mean={conditional_means['extent_teacher_fg_map']:.6f} | "
                            f"extent_teacher_bg_map_mean={conditional_means['extent_teacher_bg_map']:.6f} | "
                            f"extent_bg_temporal_reliability_mean={conditional_means['extent_bg_temporal_reliability']:.6f} | "
                            "extent_bg_temporal_reliability_p10/p50/p90="
                            f"{histogram_quantile_from_hist(reliability_hist, 0.10):.6f}/"
                            f"{histogram_quantile_from_hist(reliability_hist, 0.50):.6f}/"
                            f"{histogram_quantile_from_hist(reliability_hist, 0.90):.6f} | "
                            f"extent_bg_dino_ceiling_mean={conditional_means['extent_bg_dino_ceiling']:.6f} | "
                            f"extent_bg_final_weight_mean={conditional_means['extent_bg_final_weight']:.6f} | "
                            "extent_bg_fg_like/ambiguous/bg_like_weight_mean="
                            f"{conditional_means['extent_bg_fg_like_weight']:.6f}/"
                            f"{conditional_means['extent_bg_ambiguous_weight']:.6f}/"
                            f"{conditional_means['extent_bg_bg_like_weight']:.6f} | "
                            f"unknown_map_mean={conditional_means['unknown_map']:.6f} | "
                            f"other_map_mean={conditional_means['other_map']:.6f} | "
                            f"student_pred_area_mean={tepr_epoch_accumulator['student_pred_fg_count'] / prediction_pixels:.6f} | "
                            f"teacher_pred_area_mean={tepr_epoch_accumulator['teacher_pred_fg_count'] / prediction_pixels:.6f} | "
                            f"student_prob_fg_core={conditional_means['student_prob_fg_core']:.6f} | "
                            f"student_prob_extent={conditional_means['student_prob_extent']:.6f} | "
                            f"teacher_fg_fg_core={conditional_means['teacher_fg_fg_core']:.6f} | "
                            f"teacher_fg_extent={conditional_means['teacher_fg_extent']:.6f} | "
                            "valid core_conflict/core_no_conflict/extent_fg/extent_bg/unknown/other="
                            f"{int(conditional_valid['core_conflict_map'])}/"
                            f"{int(conditional_valid['core_no_conflict_map'])}/"
                            f"{int(conditional_valid['extent_teacher_fg_map'])}/"
                            f"{int(conditional_valid['extent_teacher_bg_map'])}/"
                            f"{int(conditional_valid['unknown_map'])}/"
                            f"{int(conditional_valid['other_map'])}"
                        )
                if use_rast:
                    rast_batches = max(rast_stat_batches, 1)
                    logger.log(
                        f"[RAST] epoch={epoch:03d} | "
                        f"rast_scale={rast_scale_sum / rast_batches:.6f} | "
                        f"rast_pre_reset_scale={rast_pre_reset_scale_sum / rast_batches:.6f} | "
                        f"rast_post_reset_scale={rast_post_reset_scale_sum / rast_batches:.6f} | "
                        f"rast_scale_effective={rast_scale_effective_sum / rast_batches:.6f} | "
                        f"teacher_routing_scale={teacher_routing_scale_sum / rast_batches:.6f} | "
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
                if use_esa_post_reset:
                    post_batches = max(esa_post_stat_batches, 1)
                    diag_batches = max(esa_stat_batches, 1)
                    logger.log(
                        f"[ESA-PostReset] epoch={epoch:03d} | "
                        f"active={bool(esa_post_active_batches > 0)} | "
                        f"post_scale={esa_post_sums['esa_post_reset_scale'] / post_batches:.6f} | "
                        f"teacher_routing_scale={esa_post_sums['teacher_routing_scale'] / post_batches:.6f} | "
                        f"static_weight={static_weight_log:.2f} | "
                        f"teacher_weight={teacher_weight_log:.2f} | "
                        f"rast_pre_reset_scale={rast_pre_reset_scale_sum / max(rast_stat_batches, 1):.6f} | "
                        f"rast_post_reset_scale={rast_post_reset_scale_sum / max(rast_stat_batches, 1):.6f} | "
                        f"extent_teacher_fg_ratio={esa_extent_teacher_fg_ratio_sum / max(esa_asym_stat_batches, 1):.6f} | "
                        f"extent_bg_fg_like_ratio={esa_extent_teacher_bg_fg_like_ratio_sum / max(esa_asym_stat_batches, 1):.6f} | "
                        f"extent_bg_ambig_ratio={esa_extent_teacher_bg_ambig_ratio_sum / max(esa_asym_stat_batches, 1):.6f} | "
                        f"extent_bg_bg_like_ratio={esa_extent_teacher_bg_bg_like_ratio_sum / max(esa_asym_stat_batches, 1):.6f} | "
                        f"map_mean={esa_post_sums['esa_post_map_mean'] / post_batches:.6f} | "
                        f"map_min={(esa_post_map_min if esa_post_map_min is not None else 1.0):.6f} | "
                        f"map_max={(esa_post_map_max if esa_post_map_max is not None else 1.0):.6f} | "
                        f"map_fg_core_mean={esa_post_sums['esa_post_map_fg_core_mean'] / post_batches:.6f} | "
                        f"map_bg_core_mean={esa_post_sums['esa_post_map_bg_core_mean'] / post_batches:.6f} | "
                        f"map_unknown_mean={esa_post_sums['esa_post_map_unknown_mean'] / post_batches:.6f} | "
                        "map_extent_teacher_fg_mean="
                        f"{esa_post_sums['esa_post_map_extent_teacher_fg_mean'] / post_batches:.6f} | "
                        "map_extent_bg_fg_like_mean="
                        f"{esa_post_sums['esa_post_map_extent_bg_fg_like_mean'] / post_batches:.6f} | "
                        "map_extent_bg_ambig_mean="
                        f"{esa_post_sums['esa_post_map_extent_bg_ambig_mean'] / post_batches:.6f} | "
                        "map_extent_bg_bg_like_mean="
                        f"{esa_post_sums['esa_post_map_extent_bg_bg_like_mean'] / post_batches:.6f} | "
                        f"fg_core_valid_ratio={esa_post_valid_counts['esa_post_fg_core_valid'] / post_batches:.6f} | "
                        f"bg_core_valid_ratio={esa_post_valid_counts['esa_post_bg_core_valid'] / post_batches:.6f} | "
                        f"unknown_valid_ratio={esa_post_valid_counts['esa_post_unknown_valid'] / post_batches:.6f} | "
                        f"student_prob_fg_core={esa_student_prob_fg_core_sum / diag_batches:.6f} | "
                        f"student_prob_bg_core={esa_student_prob_bg_core_sum / diag_batches:.6f} | "
                        f"student_prob_extent={esa_student_prob_extent_sum / diag_batches:.6f} | "
                        f"student_prob_unknown={esa_student_prob_unknown_sum / diag_batches:.6f} | "
                        f"teacher_fg_fg_core={esa_teacher_fg_fg_core_sum / diag_batches:.6f} | "
                        f"teacher_fg_bg_core={esa_teacher_fg_bg_core_sum / diag_batches:.6f} | "
                        f"teacher_fg_extent={esa_teacher_fg_extent_sum / diag_batches:.6f} | "
                        f"teacher_fg_unknown={esa_teacher_fg_unknown_sum / diag_batches:.6f} | "
                        f"student_pred_area={student_pred_area_sum / max(num_batches, 1):.6f} | "
                        f"teacher_pred_area={teacher_pred_area_sum / max(num_batches, 1):.6f}"
                    )
                if use_esa_ber:
                    ber_batches = max(esa_ber_stat_batches, 1)
                    ber_avg = {
                        key: esa_ber_sums[key] / ber_batches for key in esa_ber_stat_keys
                    }
                    logger.log(
                        f"[ESA-BER] epoch={epoch:03d} | "
                        f"ber_scale={ber_avg['ber_scale']:.6f} | "
                        f"lambda_ber_eff={ber_avg['lambda_ber_eff']:.8f} | "
                        f"proto_valid_ratio={ber_avg['proto_valid_ratio']:.6f} | "
                        f"graph_valid_ratio={ber_avg['graph_valid_ratio']:.6f} | "
                        f"valid_image_ratio={ber_avg['valid_image_ratio']:.6f} | "
                        f"pos_raw_ratio={ber_avg['pos_raw_ratio']:.8f} | "
                        f"neg_extent_raw_ratio={ber_avg['neg_extent_raw_ratio']:.8f} | "
                        f"neg_hard_bg_raw_ratio={ber_avg['neg_hard_bg_raw_ratio']:.8f} | "
                        f"neg_raw_ratio={ber_avg['neg_raw_ratio']:.8f} | "
                        f"selected_pos_ratio={ber_avg['selected_pos_ratio']:.8f} | "
                        f"selected_neg_ratio={ber_avg['selected_neg_ratio']:.8f} | "
                        f"selected_pairs_mean={ber_avg['selected_pairs_mean']:.4f} | "
                        f"selected_pairs_min={(esa_ber_pairs_min if esa_ber_pairs_min is not None else 0.0):.0f} | "
                        f"selected_pairs_max={(esa_ber_pairs_max if esa_ber_pairs_max is not None else 0.0):.0f} | "
                        f"topk_sem_weight_sum_error_max={esa_ber_topk_sum_error_max:.8g} | "
                        f"pos_margin_mean={ber_avg['pos_margin_mean']:.6f} | "
                        f"pos_conn_mean={ber_avg['pos_conn_mean']:.6f} | "
                        f"pos_student_prob_mean={ber_avg['pos_student_prob_mean']:.6f} | "
                        f"pos_teacher_prob_mean={ber_avg['pos_teacher_prob_mean']:.6f} | "
                        f"neg_margin_mean={ber_avg['neg_margin_mean']:.6f} | "
                        f"neg_conn_mean={ber_avg['neg_conn_mean']:.6f} | "
                        f"neg_student_prob_mean={ber_avg['neg_student_prob_mean']:.6f} | "
                        f"neg_teacher_prob_mean={ber_avg['neg_teacher_prob_mean']:.6f} | "
                        f"pos_logit_mean={ber_avg['pos_logit_mean']:.6f} | "
                        f"neg_logit_mean={ber_avg['neg_logit_mean']:.6f} | "
                        f"logit_gap={ber_avg['logit_gap']:.6f} | "
                        f"rank_violation_ratio={ber_avg['rank_violation_ratio']:.6f} | "
                        f"loss_ber_raw={ber_avg['loss_ber_raw']:.8f} | "
                        f"loss_ber_weighted={ber_avg['loss_ber_weighted']:.8f} | "
                        f"ber_to_main_loss_ratio={ber_avg['ber_to_main_loss_ratio']:.8f} | "
                        f"student_pred_area={student_pred_area_sum / max(num_batches, 1):.6f} | "
                        f"teacher_pred_area={teacher_pred_area_sum / max(num_batches, 1):.6f} | "
                        f"student_prob_fg_core={ber_avg['student_prob_fg_core']:.6f} | "
                        f"student_prob_bg_core={ber_avg['student_prob_bg_core']:.6f} | "
                        f"student_prob_extent={ber_avg['student_prob_extent']:.6f} | "
                        f"teacher_fg_extent={ber_avg['teacher_fg_extent']:.6f}"
                    )
                    for source, source_accumulator in sorted(esa_ber_source_sums.items()):
                        source_images = max(int(source_accumulator["images"]), 1)
                        logger.log(
                            f"[ESA-BER-Source] source={source} | "
                            f"candidate_pos_ratio={source_accumulator['pos_raw_ratio'] / source_images:.8f} | "
                            f"candidate_neg_ratio={source_accumulator['neg_raw_ratio'] / source_images:.8f} | "
                            f"valid_image_ratio={source_accumulator['valid_images'] / source_images:.6f} | "
                            f"selected_pairs={source_accumulator['selected_pairs'] / source_images:.4f} | "
                            f"logit_gap={source_accumulator['logit_gap'] / source_images:.6f} | "
                            f"rank_violation={source_accumulator['rank_violation_ratio'] / source_images:.6f}"
                        )
                    if ber_avg["ber_scale"] > 0.0 and ber_avg["valid_image_ratio"] < 0.20:
                        esa_ber_noop_warning_streak += 1
                    else:
                        esa_ber_noop_warning_streak = 0
                    if esa_ber_noop_warning_streak >= 3:
                        logger.log(
                            "[ESA-BER WARNING] candidate selection is nearly inactive for "
                            f"{esa_ber_noop_warning_streak} consecutive epochs."
                        )
                    if esa_ber_loss_ratio_warning:
                        logger.log(
                            "[ESA-BER WARNING] weighted BER loss exceeded 15% of main loss "
                            "for at least one batch in this epoch."
                        )
                    current_area = student_pred_area_sum / max(num_batches, 1)
                    if epoch == int(getattr(cfg, "ESA_BER_START_EPOCH", 21)) - 1:
                        esa_ber_pre_active_area_reference = current_area
                    if (
                        ber_avg["ber_scale"] > 0.0
                        and ber_avg["selected_pairs_mean"] > 0.0
                        and esa_ber_pre_active_area_reference is not None
                    ):
                        area_change = abs(current_area - esa_ber_pre_active_area_reference) / max(
                            abs(esa_ber_pre_active_area_reference), 1e-6
                        )
                        esa_ber_area_warning_streak = (
                            esa_ber_area_warning_streak + 1 if area_change > 0.05 else 0
                        )
                        if esa_ber_area_warning_streak >= 3:
                            logger.log(
                                "[ESA-BER WARNING] ranking may be degenerating into area bias; "
                                f"student area differs from the pre-BER epoch reference by {area_change:.2%}."
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
                    use_tce
                    and bool(getattr(cfg, "USE_TCE_DIAGNOSTIC", True))
                    and int(epoch) % max(1, int(getattr(cfg, "TCE_DIAG_LOG_INTERVAL_EPOCH", 1))) == 0
                ):
                    tce_batches = max(tce_stat_batches, 1)
                    logger.log(
                        f"[TCE] epoch={epoch:03d} | "
                        f"tce_scale={tce_scale_sum / tce_batches:.6f} | "
                        f"lambda_tce_eff={tce_lambda_sum / tce_batches:.8f} | "
                        f"cover_area_mean={tce_cover_area_sum / tce_batches:.6f} | "
                        f"current_area_mean={tce_current_area_sum / tce_batches:.6f} | "
                        f"shrink_gate_ratio={tce_shrink_gate_ratio_sum / tce_batches:.6f} | "
                        f"bg_safe_gate_ratio={tce_bg_safe_gate_ratio_sum / tce_batches:.6f} | "
                        f"image_gate_ratio={tce_image_gate_ratio_sum / tce_batches:.6f} | "
                        f"lost_raw_ratio={tce_lost_raw_ratio_sum / tce_batches:.6f} | "
                        f"lost_capped_ratio={tce_lost_capped_ratio_sum / tce_batches:.6f} | "
                        f"new_raw_ratio={tce_new_raw_ratio_sum / tce_batches:.6f} | "
                        f"new_capped_ratio={tce_new_capped_ratio_sum / tce_batches:.6f} | "
                        f"tce_total_ratio={tce_total_ratio_sum / tce_batches:.6f} | "
                        f"tce_valid_image_ratio={tce_valid_image_ratio_sum / tce_batches:.6f} | "
                        f"lost_margin_mean={tce_lost_margin_mean_sum / tce_batches:.6f} | "
                        f"new_margin_mean={tce_new_margin_mean_sum / tce_batches:.6f} | "
                        f"lost_cover_conf_mean={tce_lost_cover_conf_mean_sum / tce_batches:.6f} | "
                        f"current_teacher_conf_lost_mean={tce_current_teacher_conf_lost_mean_sum / tce_batches:.6f} | "
                        f"current_teacher_conf_new_mean={tce_current_teacher_conf_new_mean_sum / tce_batches:.6f} | "
                        f"teacher_map_final_mean={tce_teacher_map_final_mean_sum / tce_batches:.6f} | "
                        f"teacher_map_final_tce_mean={tce_teacher_map_final_tce_mean_sum / tce_batches:.6f} | "
                        f"skipped_no_fg_proto={tce_skipped_no_fg_proto_sum} | "
                        f"skipped_no_bg_proto={tce_skipped_no_bg_proto_sum} | "
                        f"loss_tce={tce_loss_sum / tce_batches:.6f}"
                    )
                if (
                    use_lceg
                    and bool(getattr(cfg, "USE_LCEG_DIAGNOSTIC", True))
                    and int(epoch) % max(1, int(getattr(cfg, "LCEG_DIAG_LOG_INTERVAL_EPOCH", 1))) == 0
                ):
                    lceg_batches = max(lceg_stat_batches, 1)
                    logger.log(
                        f"[LCEG] epoch={epoch:03d} | "
                        f"lceg_scale={lceg_scale_sum / lceg_batches:.6f} | "
                        f"lambda_lceg_eff={lceg_lambda_sum / lceg_batches:.8f} | "
                        f"cover_area_mean={lceg_cover_area_sum / lceg_batches:.6f} | "
                        f"current_area_mean={lceg_current_area_sum / lceg_batches:.6f} | "
                        f"current_fg_extent_ratio={lceg_current_fg_extent_ratio_sum / lceg_batches:.6f} | "
                        f"cover_fg_extent_ratio={lceg_cover_fg_extent_ratio_sum / lceg_batches:.6f} | "
                        f"shrink_gate_ratio={lceg_shrink_gate_ratio_sum / lceg_batches:.6f} | "
                        f"bg_safe_gate_ratio={lceg_bg_safe_gate_ratio_sum / lceg_batches:.6f} | "
                        f"core_raw_ratio={lceg_core_raw_ratio_sum / lceg_batches:.6f} | "
                        f"core_capped_ratio={lceg_core_capped_ratio_sum / lceg_batches:.6f} | "
                        f"lost_raw_ratio={lceg_lost_raw_ratio_sum / lceg_batches:.6f} | "
                        f"lost_capped_ratio={lceg_lost_capped_ratio_sum / lceg_batches:.6f} | "
                        f"new_raw_ratio={lceg_new_raw_ratio_sum / lceg_batches:.6f} | "
                        f"new_capped_ratio={lceg_new_capped_ratio_sum / lceg_batches:.6f} | "
                        f"lceg_total_ratio={lceg_total_ratio_sum / lceg_batches:.6f} | "
                        f"lceg_valid_image_ratio={lceg_valid_image_ratio_sum / lceg_batches:.6f} | "
                        f"core_conf_mean={lceg_core_conf_mean_sum / lceg_batches:.6f} | "
                        f"lost_cover_prob_mean={lceg_lost_cover_prob_mean_sum / lceg_batches:.6f} | "
                        f"lost_margin_mean={lceg_lost_margin_mean_sum / lceg_batches:.6f} | "
                        f"new_margin_mean={lceg_new_margin_mean_sum / lceg_batches:.6f} | "
                        f"teacher_map_final_mean={lceg_teacher_map_final_mean_sum / lceg_batches:.6f} | "
                        f"teacher_map_final_lceg_mean={lceg_teacher_map_final_lceg_mean_sum / lceg_batches:.6f} | "
                        f"teacher_map_coarse_mean={lceg_teacher_map_coarse_mean_sum / lceg_batches:.6f} | "
                        f"teacher_map_coarse_lceg_mean={lceg_teacher_map_coarse_lceg_mean_sum / lceg_batches:.6f} | "
                        f"skipped_no_fg_proto={lceg_skipped_no_fg_proto_sum} | "
                        f"skipped_no_bg_proto={lceg_skipped_no_bg_proto_sum} | "
                        f"loss_lceg_core={lceg_loss_core_sum / lceg_batches:.6f} | "
                        f"loss_lceg_lost={lceg_loss_lost_sum / lceg_batches:.6f} | "
                        f"loss_lceg_new={lceg_loss_new_sum / lceg_batches:.6f} | "
                        f"loss_lceg_final={lceg_loss_final_sum / lceg_batches:.6f} | "
                        f"loss_lceg_coarse={lceg_loss_coarse_sum / lceg_batches:.6f} | "
                        f"loss_lceg={lceg_loss_sum / lceg_batches:.6f}"
                    )
                if (
                    bool(getattr(cfg, "USE_CLEAN_CONSOLIDATION_LOG", False))
                    and use_dabe_pu_despl_sched
                    and int(epoch) >= int(getattr(cfg, "LCEG_STOP_EPOCH", max_epoch + 1))
                ):
                    cc_batches = max(num_batches, 1)
                    cc_rast_batches = max(rast_stat_batches, 1)
                    cc_esa_batches = max(esa_asym_stat_batches, 1)
                    cc_lceg_batches = max(lceg_stat_batches, 1)
                    logger.log(
                        f"[CleanConsolidation] epoch={epoch:03d} | "
                        "is_clean_consolidation=True | "
                        f"static_weight={static_weight_log:.2f} | "
                        f"teacher_weight={teacher_weight_log:.2f} | "
                        f"rast_scale={rast_scale_effective_sum / cc_rast_batches:.6f} | "
                        f"esa_asym_scale={esa_asym_scale_sum / cc_esa_batches:.6f} | "
                        f"lceg_scale={lceg_scale_sum / cc_lceg_batches:.6f} | "
                        f"lambda_lceg_eff={lceg_lambda_sum / cc_lceg_batches:.8f} | "
                        f"student_pred_area_mean={student_pred_area_sum / cc_batches:.6f} | "
                        f"teacher_pred_area_mean={teacher_pred_area_sum / cc_batches:.6f} | "
                        f"fg_conflict_ratio={rast_fg_conflict_ratio_sum / cc_rast_batches:.6f} | "
                        f"extent_teacher_fg_ratio={esa_extent_teacher_fg_ratio_sum / cc_esa_batches:.6f} | "
                        f"teacher_map_final_lceg_mean={lceg_teacher_map_final_lceg_mean_sum / cc_lceg_batches:.6f} | "
                        f"teacher_map_coarse_lceg_mean={lceg_teacher_map_coarse_lceg_mean_sum / cc_lceg_batches:.6f} | "
                        f"loss_lceg={lceg_loss_sum / cc_lceg_batches:.6f} | "
                        f"teacher_loss_final={dabe_pu_teacher_final_loss_sum / max(dabe_pu_stat_batches, 1):.6f} | "
                        f"teacher_loss_coarse={dabe_pu_teacher_coarse_loss_sum / max(dabe_pu_stat_batches, 1):.6f} | "
                        f"teacher_loss_base={dabe_pu_teacher_base_loss_sum / max(dabe_pu_stat_batches, 1):.6f} | "
                        f"lr={current_lr(optimizer):.8f}"
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
            if (
                use_pa_dagp_train
                and int(epoch) % max(1, int(getattr(cfg, "PA_DAGP_DIAG_LOG_INTERVAL_EPOCH", 1))) == 0
            ):
                pa_batches = max(pa_dagp_stat_batches, 1)
                pa_avg = {key: pa_dagp_epoch_sums[key] / pa_batches for key in pa_dagp_stat_keys}
                logger.log(
                    f"[PA-DAGP] epoch={epoch:03d} | "
                    f"edge_scale={pa_avg['edge_scale']:.8f} | "
                    f"aux_scale={pa_avg['aux_scale']:.8f} | "
                    f"edge_cut_eff={pa_avg['edge_cut_eff']:.8f} | "
                    f"lambda_pa_eff={pa_avg['lambda_pa_eff']:.8f} | "
                    f"anchor_valid_ratio={pa_avg['anchor_valid_ratio']:.6f} | "
                    f"anchor_cosine_mean={pa_avg['anchor_cosine_mean']:.6f} | "
                    f"rho_mean={pa_avg['rho_mean']:.6f} | "
                    f"raw_pol_mean={pa_avg['raw_pol_mean']:.6f} | "
                    f"raw_pol_std={pa_avg['raw_pol_std']:.6f} | "
                    f"signed_pol_mean={pa_avg['signed_pol_mean']:.6f} | "
                    f"signed_pol_std={pa_avg['signed_pol_std']:.6f} | "
                    f"positive_ratio={pa_avg['positive_ratio']:.6f} | "
                    f"negative_ratio={pa_avg['negative_ratio']:.6f} | "
                    f"ambiguous_ratio={pa_avg['ambiguous_ratio']:.6f} | "
                    f"fg_core_pol_mean={pa_avg['fg_core_pol_mean']:.6f} | "
                    f"bg_core_pol_mean={pa_avg['bg_core_pol_mean']:.6f} | "
                    f"core_gap={pa_avg['core_gap']:.6f} | "
                    f"hard_fg_ratio={pa_avg['hard_fg_ratio']:.6f} | "
                    f"hard_bg_ratio={pa_avg['hard_bg_ratio']:.6f} | "
                    f"hard_fg_pol_mean={pa_avg['hard_fg_pol_mean']:.6f} | "
                    f"hard_bg_pol_mean={pa_avg['hard_bg_pol_mean']:.6f} | "
                    f"cross_edge_ratio={pa_avg['cross_edge_ratio']:.6f} | "
                    f"gate_mean={pa_avg['gate_mean']:.6f} | "
                    f"gate_min={(pa_dagp_gate_min if pa_dagp_gate_min is not None else 1.0):.6f} | "
                    f"gate_max={(pa_dagp_gate_max if pa_dagp_gate_max is not None else 1.0):.6f} | "
                    f"edge_suppressed_ratio={pa_avg['edge_suppressed_ratio']:.6f} | "
                    f"loss_pa_fg={pa_avg['loss_pa_fg']:.8f} | "
                    f"loss_pa_bg={pa_avg['loss_pa_bg']:.8f} | "
                    f"loss_pa_core={pa_avg['loss_pa_core']:.8f} | "
                    f"loss_pa_hfg={pa_avg['loss_pa_hfg']:.8f} | "
                    f"loss_pa_hbg={pa_avg['loss_pa_hbg']:.8f} | "
                    f"loss_pa_hard={pa_avg['loss_pa_hard']:.8f} | "
                    f"loss_pa_weighted={pa_avg['loss_pa_weighted']:.8f} | "
                    f"coarse_pa_vs_original_abs_diff={pa_dagp_compare_coarse_abs_diff:.8f} | "
                    f"coarse_pred_area_delta={pa_dagp_compare_coarse_area_delta:.8f} | "
                    f"final_pred_area_delta={pa_dagp_compare_final_area_delta:.8f}"
                )
                if pa_avg["edge_scale"] > 0.0:
                    warning_conditions = {
                        "positive_ratio": pa_avg["positive_ratio"] > 0.90,
                        "negative_ratio": pa_avg["negative_ratio"] > 0.90,
                        "ambiguous_ratio": pa_avg["ambiguous_ratio"] > 0.95,
                        "signed_pol_std": pa_avg["signed_pol_std"] < 0.05,
                        "edge_suppressed_ratio": pa_avg["edge_suppressed_ratio"] > 0.40,
                        "anchor_valid_ratio": pa_avg["anchor_valid_ratio"] < 0.80,
                    }
                    for warning_name, active in warning_conditions.items():
                        if active:
                            pa_dagp_warning_streaks[warning_name] += 1
                        else:
                            pa_dagp_warning_streaks[warning_name] = 0
                        if pa_dagp_warning_streaks[warning_name] == 3:
                            logger.log(
                                f"[PA-DAGP WARNING] {warning_name} condition persisted for 3 epochs; "
                                "parameters are not changed automatically."
                            )
            if use_csd_head(cfg) or use_csd_v1r_head(cfg):
                stat_batches = max(csd_stat_batches, 1)
                csd_log_tag = "CSD-v1R" if use_csd_v1r_head(cfg) else "CSD"
                logger.log(
                    f"[{csd_log_tag}] epoch={epoch:03d} | "
                    f"csd_scale={csd_scale_sum / stat_batches:.8f} | "
                    f"beta_eff={csd_beta_sum / stat_batches:.8f} | "
                    f"final_minus_coarse_abs_mean={csd_final_minus_coarse_abs_mean_sum / stat_batches:.8f} | "
                    f"detail_gate_mean={csd_gate_mean_sum / stat_batches:.8f} | "
                    f"detail_gate_min={(csd_gate_min if csd_gate_min is not None else -1.0):.8f} | "
                    f"detail_gate_max={(csd_gate_max if csd_gate_max is not None else -1.0):.8f} | "
                    f"residual_abs_mean={csd_residual_abs_mean_sum / stat_batches:.8f} | "
                    f"residual_abs_max={(csd_residual_abs_max if csd_residual_abs_max is not None else -1.0):.8f} | "
                    f"bg_reliable_ratio={csd_bg_reliable_ratio_sum / stat_batches:.8f} | "
                    f"loss_bg_detail={csd_loss_bg_detail_sum / stat_batches:.8f} | "
                    f"loss_bg_detail_weighted={csd_loss_bg_detail_weighted_sum / stat_batches:.8f} | "
                    f"boundary_scale={csd_boundary_scale_sum / stat_batches:.8f} | "
                    f"boundary_target_mean={csd_boundary_target_mean_sum / stat_batches:.8f} | "
                    f"loss_boundary={csd_loss_boundary_sum / stat_batches:.8f} | "
                    f"loss_boundary_weighted={csd_loss_boundary_weighted_sum / stat_batches:.8f} | "
                    f"loss_final={total_base_loss / max(num_batches, 1):.6f} | "
                    f"loss_coarse={total_ndr_coarse_aux_loss / max(num_batches, 1):.6f} | "
                    f"loss_base={total_aux_base_loss / max(num_batches, 1):.6f}"
                )
                if use_cssd_train:
                    cssd_batches = max(cssd_stat_batches, 1)
                    logger.log(
                        f"[CSSD] epoch={epoch:03d} | "
                        f"cssd_scale={cssd_scale_epoch:.8f} | "
                        f"high_forward_batches={cssd_high_forward_batches} | "
                        f"high_forward_microbatches={cssd_high_forward_calls} | "
                        "teacher_high_forward_count=0 | optimizer_steps_per_batch=1 | "
                        f"lambda_hr={cssd_epoch_sums['lambda_hr'] / cssd_batches:.8f} | "
                        f"lambda_pred={cssd_epoch_sums['lambda_pred'] / cssd_batches:.8f} | "
                        f"lambda_boundary={cssd_epoch_sums['lambda_boundary'] / cssd_batches:.8f} | "
                        f"normal_prob_mean={cssd_epoch_sums['normal_prob_mean'] / cssd_batches:.8f} | "
                        f"high_prob_mean={cssd_epoch_sums['high_prob_mean'] / cssd_batches:.8f} | "
                        f"normal_area={cssd_epoch_sums['normal_pred_area'] / cssd_batches:.8f} | "
                        f"high_area={cssd_epoch_sums['high_pred_area'] / cssd_batches:.8f} | "
                        f"prob_abs_diff={cssd_epoch_sums['normal_high_prob_abs_diff'] / cssd_batches:.8f} | "
                        f"binary_agreement={cssd_epoch_sums['normal_high_binary_agreement'] / cssd_batches:.8f} | "
                        f"normal_conf={cssd_epoch_sums['normal_conf_mean'] / cssd_batches:.8f} | "
                        f"high_conf={cssd_epoch_sums['high_conf_mean'] / cssd_batches:.8f} | "
                        f"high_more_conf_ratio={cssd_epoch_sums['high_more_conf_ratio'] / cssd_batches:.8f} | "
                        f"conf_adv_mean={cssd_epoch_sums['conf_adv_mean'] / cssd_batches:.8f} | "
                        f"core_ratio={cssd_epoch_sums['core_mask_ratio'] / cssd_batches:.8f} | "
                        f"transfer_raw_ratio={cssd_epoch_sums['transfer_raw_ratio'] / cssd_batches:.8f} | "
                        f"transfer_ratio={cssd_epoch_sums['transfer_capped_ratio'] / cssd_batches:.8f} | "
                        f"transfer_valid_image_ratio={cssd_epoch_sums['transfer_valid_image_ratio'] / cssd_batches:.8f} | "
                        f"boundary_ratio={cssd_epoch_sums['boundary_mask_ratio'] / cssd_batches:.8f} | "
                        f"boundary_valid_image_ratio={cssd_epoch_sums['boundary_valid_image_ratio'] / cssd_batches:.8f} | "
                        f"loss_normal_group={cssd_epoch_sums['loss_normal_group'] / cssd_batches:.8f} | "
                        f"loss_high_group={cssd_epoch_sums['loss_high_group'] / cssd_batches:.8f} | "
                        f"loss_high_static={cssd_epoch_sums['loss_high_static_group'] / cssd_batches:.8f} | "
                        f"loss_high_teacher={cssd_epoch_sums['loss_high_teacher_group'] / cssd_batches:.8f} | "
                        f"loss_dual_group={cssd_epoch_sums['loss_dual_group'] / cssd_batches:.8f} | "
                        f"loss_core={cssd_epoch_sums['loss_core'] / cssd_batches:.8f} | "
                        f"loss_transfer={cssd_epoch_sums['loss_transfer'] / cssd_batches:.8f} | "
                        f"loss_pred={cssd_epoch_sums['loss_pred'] / cssd_batches:.8f} | "
                        f"loss_boundary={cssd_epoch_sums['loss_boundary'] / cssd_batches:.8f} | "
                        f"loss_cssd_total={cssd_epoch_sums['loss_cssd_weighted'] / cssd_batches:.8f} | "
                        f"loss_total={avg_loss:.8f}"
                    )
                    if cssd_scale_epoch <= 0.0:
                        if cssd_high_forward_batches != 0 or cssd_high_forward_calls != 0:
                            raise RuntimeError("CSSD inactive epoch performed a high-resolution forward.")
                    elif cssd_stat_batches > 0:
                        normal_area = cssd_epoch_sums["normal_pred_area"] / cssd_stat_batches
                        high_area = cssd_epoch_sums["high_pred_area"] / cssd_stat_batches
                        area_ratio = high_area / max(normal_area, 1e-8)
                        if area_ratio < 0.5 or area_ratio > 2.0:
                            cssd_area_ratio_warning_streak += 1
                        else:
                            cssd_area_ratio_warning_streak = 0
                        transfer_ratio = cssd_epoch_sums["transfer_capped_ratio"] / cssd_stat_batches
                        if transfer_ratio < 0.001:
                            cssd_transfer_noop_warning_streak += 1
                        else:
                            cssd_transfer_noop_warning_streak = 0
                        if cssd_area_ratio_warning_streak == 3:
                            logger.log(
                                f"[CSSD WARNING] high-resolution prediction distribution mismatch | "
                                f"epoch={epoch:03d} | high_normal_area_ratio={area_ratio:.8f} | "
                                f"out_of_range_streak={cssd_area_ratio_warning_streak}"
                            )
                        if cssd_transfer_noop_warning_streak == 5:
                            logger.log(
                                f"[CSSD WARNING] privileged transfer is nearly inactive | "
                                f"epoch={epoch:03d} | transfer_ratio={transfer_ratio:.8f} | "
                                f"low_transfer_streak={cssd_transfer_noop_warning_streak}"
                            )
                if use_hr_bfr(cfg):
                    hr_batches = max(hr_bfr_stat_batches, 1)
                    logger.log(
                        f"[HR-BFR] epoch={epoch:03d} | "
                        f"hr_scale={hr_bfr_scale_sum / hr_batches:.8f} | "
                        f"hr_beta_eff={hr_bfr_beta_sum / hr_batches:.8f} | "
                        f"band_ratio_mean={hr_bfr_band_ratio_sum / hr_batches:.8f} | "
                        f"band_ratio_max={(hr_bfr_band_ratio_max if hr_bfr_band_ratio_max is not None else 0.0):.8f} | "
                        f"valid_img_ratio={hr_bfr_valid_img_ratio_sum / hr_batches:.8f} | "
                        f"skip_img_ratio={hr_bfr_skip_img_ratio_sum / hr_batches:.8f} | "
                        f"hr_active_pixel_ratio={hr_bfr_active_pixel_ratio_sum / hr_batches:.8f} | "
                        f"anchor_area={hr_bfr_anchor_area_sum / hr_batches:.8f} | "
                        f"hr_area={hr_bfr_hr_area_sum / hr_batches:.8f} | "
                        f"area_delta={hr_bfr_area_delta_sum / hr_batches:.8f} | "
                        f"hr_minus_anchor_abs_mean={hr_bfr_minus_anchor_sum / hr_batches:.8f} | "
                        f"residual_abs_mean={hr_bfr_residual_abs_mean_sum / hr_batches:.8f} | "
                        f"residual_abs_max={(hr_bfr_residual_abs_max if hr_bfr_residual_abs_max is not None else 0.0):.8f} | "
                        f"bg_reliable_ratio={hr_bfr_bg_reliable_ratio_sum / hr_batches:.8f} | "
                        f"edge_support_mean={hr_bfr_edge_support_sum / hr_batches:.8f} | "
                        f"loss_band_bce={hr_bfr_loss_band_sum / hr_batches:.8f} | "
                        f"loss_outband_anchor={hr_bfr_loss_outband_sum / hr_batches:.8f} | "
                        f"loss_bg_prob_lock={hr_bfr_loss_bg_sum / hr_batches:.8f} | "
                        f"loss_area_neutral={hr_bfr_loss_area_sum / hr_batches:.8f} | "
                        f"loss_edge_align={hr_bfr_loss_edge_sum / hr_batches:.8f} | "
                        f"loss_hr_bfr={hr_bfr_loss_total_sum / hr_batches:.8f}"
                    )
                    if hr_bfr_skip_img_ratio_sum / hr_batches > 0.5:
                        logger.log(
                            f"[HR-BFR Warning] epoch={epoch:03d} | "
                            f"skip_img_ratio={hr_bfr_skip_img_ratio_sum / hr_batches:.8f} > 0.50000000 | "
                            "HR-BFR skipped invalid images and kept anchor logits; training continues."
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
                if use_ndr_v2(cfg):
                    v2_batches = max(ndr_v2_stat_batches, 1)
                    logger.log(
                        f"[NDR-v2] epoch={epoch:03d} | "
                        f"shape_alpha_eff={ndr_v2_shape_alpha_sum / v2_batches:.8f} | "
                        f"gate_v1_mean={ndr_v2_gate_v1_mean_sum / v2_batches:.8f} | "
                        f"gate_v2_mean={ndr_v2_gate_v2_mean_sum / v2_batches:.8f} | "
                        f"boundary_mean={ndr_v2_boundary_mean_sum / v2_batches:.8f} | "
                        f"boundary_min={(ndr_v2_boundary_min if ndr_v2_boundary_min is not None else 0.0):.8f} | "
                        f"boundary_max={(ndr_v2_boundary_max if ndr_v2_boundary_max is not None else 0.0):.8f} | "
                        f"edge_norm_mean={ndr_v2_edge_norm_mean_sum / v2_batches:.8f} | "
                        f"shape_boost_mean={ndr_v2_shape_boost_mean_sum / v2_batches:.8f} | "
                        f"shape_boost_max={(ndr_v2_shape_boost_max if ndr_v2_shape_boost_max is not None else 0.0):.8f} | "
                        f"shape_lb_scale={ndr_v2_shape_lb_scale_sum / v2_batches:.8f} | "
                        f"lambda_shape_lb_eff={ndr_v2_shape_lb_lambda_sum / v2_batches:.8f} | "
                        f"shape_candidate_raw_ratio={ndr_v2_shape_candidate_raw_ratio_sum / v2_batches:.8f} | "
                        f"shape_candidate_capped_ratio={ndr_v2_shape_candidate_capped_ratio_sum / v2_batches:.8f} | "
                        f"shape_valid_image_ratio={ndr_v2_shape_valid_image_ratio_sum / v2_batches:.8f} | "
                        f"shape_margin_mean={ndr_v2_shape_margin_mean_sum / v2_batches:.8f} | "
                        f"shape_edge_mean={ndr_v2_shape_edge_mean_sum / v2_batches:.8f} | "
                        f"shape_under_floor_mean={ndr_v2_shape_under_floor_mean_sum / v2_batches:.8f} | "
                        f"bg_lock_area={ndr_v2_bg_lock_area_sum / v2_batches:.8f} | "
                        f"bg_core_area={ndr_v2_bg_core_area_sum / v2_batches:.8f} | "
                        f"low_target_bg_area={ndr_v2_low_target_bg_area_sum / v2_batches:.8f} | "
                        f"bg_res_lock_scale={ndr_v2_bg_res_lock_scale_sum / v2_batches:.8f} | "
                        f"lambda_bg_res_lock_eff={ndr_v2_bg_res_lock_lambda_sum / v2_batches:.8f} | "
                        f"bg_prob_lock_scale={ndr_v2_bg_prob_lock_scale_sum / v2_batches:.8f} | "
                        f"lambda_bg_prob_lock_eff={ndr_v2_bg_prob_lock_lambda_sum / v2_batches:.8f} | "
                        f"positive_delta_bg_mean={ndr_v2_positive_delta_bg_mean_sum / v2_batches:.8f} | "
                        f"positive_delta_bg_max={(ndr_v2_positive_delta_bg_max if ndr_v2_positive_delta_bg_max is not None else 0.0):.8f} | "
                        f"loss_shape_lb={ndr_v2_loss_shape_lb_sum / v2_batches:.8f} | "
                        f"loss_bg_res_lock={ndr_v2_loss_bg_lock_sum / v2_batches:.8f} | "
                        f"loss_bg_prob_lock={ndr_v2_loss_bg_prob_lock_sum / v2_batches:.8f} | "
                        f"loss_ndr_v2b={ndr_v2_weighted_loss_sum / v2_batches:.8f}"
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

            if should_save_epoch_checkpoint(
                epoch,
                max_epoch,
                cfg.SAVE_INTERVAL,
                reset_epoch=reset_epoch,
                save_every_epoch=bool(getattr(cfg, "SAVE_EVERY_EPOCH", False)),
            ):
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
                if use_tepr_lite and bool(
                    getattr(cfg, "TEPR_RESET_MEMORY_AT_FINETUNE_RESET", True)
                ):
                    if tepr_memory is not None:
                        tepr_memory.clear()
                    tepr_memory = None
                    logger.log(
                        f"[TEPR-Lite] temporal memory cleared and released at finetune reset | "
                        f"epoch={epoch:03d}"
                    )


if __name__ == "__main__":
    main()
