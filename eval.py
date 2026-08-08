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
    check_cacd_feature_cache,
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
from models.online_dino_last4 import FrozenDINOv1Last4Extractor


def infer_in_channels(student_state):
    # 从 checkpoint 的 head 权重反推 feature channel 数，兼容 simple/DAGP/context_residual。
    if "adapters.f9.0.weight" in student_state:
        weight = student_state["adapters.f9.0.weight"]
    elif "consensus_encoder.proj12.0.weight" in student_state:
        weight = student_state["consensus_encoder.proj12.0.weight"]
    elif "coarse_path.base_head.weight" in student_state:
        weight = student_state["coarse_path.base_head.weight"]
    elif "csd_residual.csd_feat_proj.0.weight" in student_state:
        weight = student_state["csd_residual.csd_feat_proj.0.weight"]
    elif "sem_proj.0.weight" in student_state:
        weight = student_state["sem_proj.0.weight"]
    elif "feature_projection.weight" in student_state:
        weight = student_state["feature_projection.weight"]
    elif "base_head.weight" in student_state:
        weight = student_state["base_head.weight"]
    elif "proj.weight" in student_state:
        weight = student_state["proj.weight"]
    elif "base.weight" in student_state:
        weight = student_state["base.weight"]
    elif "base_head.proj.weight" in student_state:
        weight = student_state["base_head.proj.weight"]
    elif "decoupling.weight" in student_state:
        # DBA: the first 1x1 projection is [2 * embed_dim, in_channels, 1, 1].
        weight = student_state["decoupling.weight"]
    elif "proj12.conv.weight" in student_state:
        weight = student_state["proj12.conv.weight"]
    else:
        raise KeyError("Cannot infer in_channels from checkpoint student state.")
    channels = int(weight.shape[1])
    if "proj.weight" in student_state and channels > 1024 and channels % 4 == 0:
        # Last4LinearProbe concatenates four equal-width DINO features.
        channels //= 4
    return channels


def use_multi_level_feature(cfg):
    return bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))


def use_dagp_head(cfg):
    return str(getattr(cfg, "HEAD_TYPE", "simple")).lower() == "dagp"


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


def use_hsd_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "hsd_v1"


def use_last4_linear_probe(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "last4_linear"


def use_f12_scalelift_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "f12_scalelift"


def use_bcrd_sem_decoder(cfg):
    return str(getattr(cfg, "DECODER_TYPE", "")).strip().lower() == "bcrd_sem_v1"


def use_last4_feature_decoder(cfg):
    return (
        use_hsd_decoder(cfg)
        or use_last4_linear_probe(cfg)
        or use_f12_scalelift_decoder(cfg)
        or use_bcrd_sem_decoder(cfg)
    )


def use_online_dino_last4(cfg):
    return bool(getattr(cfg, "ONLINE_DINO_LAST4", False))


def make_model_input(cfg, batch, device, online_dino=None):
    if use_last4_feature_decoder(cfg):
        if use_online_dino_last4(cfg):
            if online_dino is None:
                raise RuntimeError(
                    "ONLINE_DINO_LAST4 evaluation requires a frozen extractor."
                )
            if "dino_input_296" not in batch:
                raise KeyError("Online DINO eval batch is missing dino_input_296.")
            inputs = batch["dino_input_296"].to(
                device, non_blocking=True
            ).float()
            return online_dino(inputs)
        required = tuple(f"feature_l{layer}" for layer in (9, 10, 11, 12))
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"DINO last-four eval batch is missing: {missing}")
        return {
            f"f{layer}": batch[f"feature_l{layer}"].to(
                device, non_blocking=True
            ).float()
            for layer in (9, 10, 11, 12)
        }
    if use_cacd(cfg):
        required = ("feature_l10", "feature_l11", "feature")
        missing = [field for field in required if field not in batch]
        if missing:
            raise KeyError(f"CACD eval batch missing feature fields: {missing}")
        return {
            "f10": batch["feature_l10"].to(device, non_blocking=True).float(),
            "f11": batch["feature_l11"].to(device, non_blocking=True).float(),
            "f12": batch["feature"].to(device, non_blocking=True).float(),
        }
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
    if not (use_ndr_branch(cfg) or use_csd_head(cfg) or use_csd_v1r_head(cfg) or use_cacd(cfg)):
        return None
    if "image_68" not in batch:
        raise KeyError("NDR/CSD decoder requires batch['image_68'].")
    return batch["image_68"].to(device, non_blocking=True).float()


def make_sobel_68(cfg, batch, device):
    if not use_cacd(cfg):
        return None
    if "sobel_68" not in batch:
        raise KeyError("CACD eval requires batch['sobel_68'].")
    return batch["sobel_68"].to(device, non_blocking=True).float()


def make_image_136(cfg, batch, device):
    if not use_hr_bfr(cfg):
        return None
    if "image_136" not in batch:
        raise KeyError("HR-BFR eval requires batch['image_136'].")
    return batch["image_136"].to(device, non_blocking=True).float()


def make_image_148(cfg, batch, device):
    if not (use_hsd_decoder(cfg) and bool(getattr(cfg, "USE_DETAIL", False))):
        return None
    if "image_148" not in batch:
        raise KeyError("HSD-Full eval requires batch['image_148'].")
    return batch["image_148"].to(device, non_blocking=True).float()


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def extract_logits_for_eval(output, cfg):
    if isinstance(output, dict) and use_hr_bfr(cfg) and bool(getattr(cfg, "HR_BFR_USE_HR_LOGITS_FOR_EVAL", True)):
        if "hr_logits" not in output:
            raise KeyError("HR_BFR_USE_HR_LOGITS_FOR_EVAL=True but model output has no hr_logits.")
        return output["hr_logits"], "hr_logits"
    if isinstance(output, dict):
        if "final_logits" in output:
            return output["final_logits"], "final_logits"
        return output["logits"], "logits"
    return output, "tensor"


def output_scalar(output, key, default=0.0):
    if not isinstance(output, dict) or key not in output:
        return float(default)
    value = output[key]
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
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


def save_pred_png(path, pred):
    ensure_dir(Path(path).parent)
    array = pred.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


@torch.no_grad()
def eval_dataset(
    cfg,
    student,
    dataset_name,
    device,
    out_dir,
    logger,
    max_samples=-1,
    online_dino=None,
):
    dataset = CachedEvalDataset(
        cfg,
        split="test",
        datasets=[dataset_name],
        max_samples=int(max_samples),
    )
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
    hr_bfr_eval_batches = 0
    hr_bfr_skip_ratio_sum = 0.0
    hr_bfr_valid_ratio_sum = 0.0
    hr_bfr_active_pixel_ratio_sum = 0.0
    hr_bfr_band_ratio_sum = 0.0
    hr_bfr_band_ratio_max = 0.0
    cssd_single_view_logged = False
    cacd_logged = False

    for batch in loader:
        if bool(getattr(cfg, "USE_AP_STCR", False)) or str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).strip().lower() == "ap_stcr":
            forbidden_training_fields = {
                "pu_target_soft_37",
                "pu_bg_anchor_37",
                "sample_index",
            }.intersection(batch)
            if forbidden_training_fields:
                raise RuntimeError(
                    "AP-STCR training-only state appeared in eval batch: "
                    f"{sorted(forbidden_training_fields)}"
                )
        gt = batch["gt"].to(device, non_blocking=True).float()
        stem = batch["stem"][0]
        model_input = make_model_input(
            cfg, batch, device, online_dino=online_dino
        )
        image_68 = make_image_68(cfg, batch, device)
        sobel_68 = make_sobel_68(cfg, batch, device)
        image_136 = make_image_136(cfg, batch, device)
        image_148 = make_image_148(cfg, batch, device)
        if use_cssd(cfg):
            if not bool(getattr(cfg, "CSSD_STRICT_SINGLE_VIEW_EVAL", True)):
                raise RuntimeError("CSSD-v1a eval requires CSSD_STRICT_SINGLE_VIEW_EVAL=True.")
            cssd_field = str(getattr(cfg, "CSSD_HR_FEATURE_FIELD", "feature_cssd_hr"))
            if cssd_field in batch:
                raise RuntimeError(
                    f"CSSD train-only high feature {cssd_field!r} appeared in CachedEvalDataset."
                )
            if not torch.is_tensor(model_input) or list(model_input.shape[1:]) != [384, 37, 37]:
                raise RuntimeError(
                    "CSSD eval must use only normal cached DINO [B,384,37,37], got "
                    f"{list(model_input.shape) if torch.is_tensor(model_input) else type(model_input).__name__}."
                )
            if not cssd_single_view_logged:
                logger.log(
                    f"[Eval CSSD] Dataset: {dataset_name} | "
                    "high_resolution_view_used=False | high_cache_read=False | "
                    f"feature_source=normal_cache | feature_shape={list(model_input.shape)} | "
                    "logits_source=normal_final_logits | test_time_scale_fusion=False"
                )
                cssd_single_view_logged = True
        if use_hsd_decoder(cfg):
            output = student(model_input, image_148=image_148)
            logits, _ = extract_logits_for_eval(output, cfg)
        elif str(getattr(cfg, "HEAD_TYPE", "simple")).lower() in {
            "dagp_safe",
            "csd_v1",
            "dagp_safe_csd_v1r",
            "cacd_v1_base",
        }:
            if use_cacd(cfg):
                output = student(
                    model_input,
                    image_68=image_68,
                    sobel_68=sobel_68,
                    return_aux=False,
                )
                if not cacd_logged:
                    logger.log(
                        f"[Eval CACD] Dataset: {dataset_name} | USE_CACD=True | "
                        f"CACD_VERSION={getattr(cfg, 'CACD_VERSION')} | "
                        f"feature_l10 shape={list(model_input['f10'].shape)} | "
                        f"feature_l11 shape={list(model_input['f11'].shape)} | "
                        f"feature_l12 shape={list(model_input['f12'].shape)} | "
                        f"final_logits shape={list(output['final_logits'].shape)} | "
                        "logits_source=final_logits | extra_inference_branch=False | post_processing=False"
                    )
                    cacd_logged = True
            elif use_csd_v1r_head(cfg):
                output = student(model_input, image_68=image_68, image_136=image_136, return_aux=False)
            else:
                output = student(model_input, image_68=image_68, return_aux=False)
            logits, logits_name = extract_logits_for_eval(output, cfg)
            if use_hr_bfr(cfg) and isinstance(output, dict):
                hr_bfr_eval_batches += 1
                hr_bfr_skip_ratio_sum += output_scalar(output, "hr_skip_img_ratio", 0.0)
                hr_bfr_valid_ratio_sum += output_scalar(output, "hr_valid_img_ratio", 1.0)
                hr_bfr_active_pixel_ratio_sum += output_scalar(output, "hr_active_pixel_ratio", 0.0)
                hr_bfr_band_ratio_sum += output_scalar(output, "hr_band_ratio_raw_136_mean", 0.0)
                hr_bfr_band_ratio_max = max(
                    hr_bfr_band_ratio_max,
                    output_scalar(output, "hr_band_ratio_raw_136_max", 0.0),
                )
            if stem == batch["stem"][0] and not hasattr(eval_dataset, "_hr_bfr_logged"):
                logger.log(f"[Eval] USE_HR_BFR={bool(getattr(cfg, 'USE_HR_BFR', False))}")
                logger.log(
                    "[Eval] HR_BFR_USE_HR_LOGITS_FOR_EVAL="
                    f"{bool(getattr(cfg, 'HR_BFR_USE_HR_LOGITS_FOR_EVAL', True))}"
                )
                logger.log(
                    "[Eval] HR_BFR_EVAL_RES_SCALE="
                    f"{float(getattr(cfg, 'HR_BFR_EVAL_RES_SCALE', 1.0)):.6f}"
                )
                logger.log(f"[Eval] logits_for_eval={logits_name}")
                logger.log(f"[Eval] logits shape={list(logits.shape)}")
                eval_dataset._hr_bfr_logged = True
        else:
            logits = extract_logits(student(model_input))
        logits = F.interpolate(logits, size=gt.shape[-2:], mode="bilinear")
        pred = (logits.sigmoid() > float(cfg.THRESHOLD)).float()
        save_pred_png(pred_dir / f"{stem}.png", pred)
        metrics.step(gt, pred)

    result = metrics.get_result()
    logger.log(f"[Eval] Dataset: {dataset_name}")
    if use_hr_bfr(cfg):
        denom = max(hr_bfr_eval_batches, 1)
        logger.log(
            f"[Eval HR-BFR] Dataset: {dataset_name} | "
            f"fallback_ratio={hr_bfr_skip_ratio_sum / denom:.8f} | "
            f"valid_img_ratio={hr_bfr_valid_ratio_sum / denom:.8f} | "
            f"band_ratio_mean={hr_bfr_band_ratio_sum / denom:.8f} | "
            f"band_ratio_max={hr_bfr_band_ratio_max:.8f} | "
            f"hr_active_pixel_ratio={hr_bfr_active_pixel_ratio_sum / denom:.8f}"
        )
    logger.log(format_metric_table(result))
    logger.log(
        f"F_MAX={float(result['F_MAX']):.6f} | "
        f"E_MAX={float(result['E_MAX']):.6f} | "
        f"ACC@0.5={float(result['ACC']):.6f} | "
        f"mIoU@0.5={float(result['mIOU']):.6f}"
    )
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate clean cached-DINO EMA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval_tag", type=str, default=None)
    parser.add_argument("--eval_name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    if args.eval_tag is not None and args.eval_name is not None:
        raise ValueError("--eval_tag and deprecated --eval_name cannot be used together.")
    if int(args.max_samples) == 0 or int(args.max_samples) < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")

    cfg = load_config(args.config)
    if use_cacd(cfg):
        if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "cacd_v1_base" or not bool(
            getattr(cfg, "USE_CACD", False)
        ):
            raise RuntimeError("CACD eval requires HEAD_TYPE='cacd_v1_base' and USE_CACD=True.")
        if any(
            bool(getattr(cfg, name, False))
            for name in ("USE_NDR_BRANCH", "USE_CSD_V1R", "USE_DAGP_SAFE_HEAD", "USE_HR_BFR", "USE_CSSD")
        ):
            raise RuntimeError("CACD eval forbids DAGP/NDR/CSD/HR-BFR/CSSD branches.")
    if use_cssd(cfg):
        if not bool(getattr(cfg, "CSSD_TRAIN_ONLY", True)):
            raise RuntimeError("CSSD-v1a eval requires CSSD_TRAIN_ONLY=True.")
        if not bool(getattr(cfg, "CSSD_STRICT_SINGLE_VIEW_EVAL", True)):
            raise RuntimeError("CSSD-v1a eval requires CSSD_STRICT_SINGLE_VIEW_EVAL=True.")
        if not use_csd_v1r_head(cfg) or bool(getattr(cfg, "USE_HR_BFR", False)):
            raise RuntimeError(
                "CSSD-v1a eval supports only the normal dagp_safe_csd_v1r output without HR-BFR."
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_eval_tag = f"eval_{Path(args.ckpt).stem}"
    eval_tag = args.eval_tag or args.eval_name or default_eval_tag
    eval_tag_path = Path(eval_tag)
    if eval_tag_path.is_absolute() or len(eval_tag_path.parts) != 1 or eval_tag in {"", ".", ".."}:
        raise ValueError(f"--eval_tag must be a single directory name, got {eval_tag!r}.")
    out_dir = Path(cfg.WORK_ROOT) / cfg.EXP_NAME / eval_tag
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
    online_dino = None
    if use_online_dino_last4(cfg):
        online_dino = FrozenDINOv1Last4Extractor(cfg).to(device).eval()

    with Logger(out_dir / "eval.log") as logger:
        logger.log(f"device = {device}")
        logger.log(f"ckpt = {args.ckpt}")
        logger.log(f"eval_tag = {eval_tag}")
        logger.log("model_for_eval = student")
        logger.log("look_twice = false")
        logger.log("teacher_or_apm_at_inference = false")
        logger.log("evaluation_protocol = original_binary")
        logger.log("model_output_resize = bilinear_to_gt_size")
        logger.log("prediction_activation = sigmoid")
        logger.log("threshold_operator = >")
        logger.log(f"binary_threshold = {float(cfg.THRESHOLD):.6f}")
        logger.log("pred_save_format = binary_0_255")
        logger.log("main_metric_input = binary_prediction")
        logger.log(f"prediction_dir = {out_dir / 'pred'}")
        logger.log(f"max_samples = {int(args.max_samples)}")
        if online_dino is not None:
            logger.log("feature_source = online frozen DINOv1-S/8 f9-f12")
            logger.log("feature_cache_used = false")
            logger.log(f"online_dino_key_paths = {online_dino.key_paths}")
        if bool(getattr(cfg, "USE_ECST", False)):
            logger.log("[Eval ECST] training_only=True")
            logger.log("[Eval ECST] temporal_memory_used=False")
            logger.log("[Eval ECST] teacher_weight_map_used=False")
            logger.log("[Eval ECST] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_SOURCE_ARBITER", False)):
            arbiter_mode = str(
                getattr(cfg, "SOURCE_ARBITER_MODE", "residual_over_ecst")
            ).lower()
            logger.log(
                f"[Eval SourceArbiter] mode={arbiter_mode} | training_only=True"
            )
            logger.log("[Eval SourceArbiter] router_loaded=False")
            logger.log("[Eval SourceArbiter] utility_evaluator_loaded=False")
            logger.log("[Eval SourceArbiter] route_memory_used=False")
            logger.log("[Eval SourceArbiter] temporal_memory_used=False")
            logger.log("[Eval SourceArbiter] ecst_map_used=False")
            logger.log("[Eval SourceArbiter] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_TEPR_LITE", False)):
            logger.log("[Eval TEPR-Lite] temporal_memory_used=False")
            logger.log("[Eval TEPR-Lite] teacher_weight_map_used=False")
            logger.log("[Eval TEPR-Lite] logits_source=student_final_logits")
        supervision_mode = str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).strip().lower()
        if bool(getattr(cfg, "USE_AP_STCR", False)) or (
            supervision_mode == "ap_stcr"
        ):
            logger.log("[Eval AP-STCR] training_only=True")
            logger.log("[Eval AP-STCR] semantic_cache_used=False")
            logger.log("[Eval AP-STCR] temporal_history_used=False")
            logger.log("[Eval AP-STCR] mixed_target_used=False")
            logger.log("[Eval AP-STCR] teacher_loaded=False")
            logger.log("[Eval AP-STCR] teacher_forward=False")
            logger.log("[Eval AP-STCR] logits_source=student_final_logits")
        if bool(getattr(cfg, "USE_PSSF", False)) or supervision_mode in {
            "pssf_state",
            "ppse_v2_state",
        }:
            if supervision_mode == "ppse_v2_state":
                logger.log("[Eval PPSE-v2] training_only=True")
                logger.log("[Eval PPSE-v2] model_for_eval=student")
                logger.log(
                    "[Eval PPSE-v2] use_ppse_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_pssf_actor_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_pssf_learner_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] use_teacher_at_inference=False"
                )
                logger.log(
                    "[Eval PPSE-v2] "
                    "use_supervision_state_at_inference=False"
                )
                logger.log("[Eval PPSE-v2] q_state_used=False")
                logger.log("[Eval PPSE-v2] history_used=False")
                logger.log(
                    "[Eval PPSE-v2] logits_source=student_final_logits"
                )
            else:
                logger.log("[Eval PSSF] training_only=True")
                logger.log("[Eval PSSF] use_pssf_at_inference=False")
                logger.log("[Eval PSSF] use_teacher_at_inference=False")
                logger.log(
                    "[Eval PSSF] use_supervision_state_at_inference=False"
                )
                logger.log("[Eval PSSF] pssf_network_loaded=False")
                logger.log("[Eval PSSF] teacher_loaded=False")
                logger.log("[Eval PSSF] q_state_used=False")
                logger.log("[Eval PSSF] history_used=False")
                logger.log(
                    "[Eval PSSF] logits_source=student_final_logits"
                )
        if bool(getattr(cfg, "USE_ESA_BER", False)):
            logger.log("[Eval ESA-BER] USE_ESA_BER=True")
            logger.log("[Eval ESA-BER] training_only=True")
            logger.log("[Eval ESA-BER] candidate_selection_used=False")
            logger.log("[Eval ESA-BER] extra_inference_branch=False")
        if use_cssd(cfg):
            logger.log(f"[Eval CSSD] USE_CSSD={bool(getattr(cfg, 'USE_CSSD', False))}")
            logger.log(f"[Eval CSSD] CSSD_TRAIN_ONLY={bool(getattr(cfg, 'CSSD_TRAIN_ONLY', True))}")
            logger.log("[Eval CSSD] high_resolution_view_used=False")
            logger.log("[Eval CSSD] high_cache_read=False")
            logger.log("[Eval CSSD] feature_source=normal_cache")
            logger.log("[Eval CSSD] logits_source=normal_final_logits")
            logger.log("[Eval CSSD] test_time_scale_fusion=False")
        if use_cacd(cfg):
            logger.log("[Eval CACD] USE_CACD=True")
            logger.log(f"[Eval CACD] CACD_VERSION={getattr(cfg, 'CACD_VERSION')}")
            logger.log("[Eval CACD] dabe_pu_read=False")
            logger.log("[Eval CACD] teacher_forward=False")
            logger.log("[Eval CACD] extra_inference_branch=False")
            logger.log("[Eval CACD] post_processing=False")
        if use_online_dino_last4(cfg):
            logger.log(
                "[Online DINO] test cache preflight skipped | "
                "source=original JPEG"
            )
        elif use_multi_level_feature(cfg):
            _, ml_reason = check_ml_feature_cache(cfg, "test")
            logger.log(f"[Cache] feature_ml:test ready | {ml_reason}")
        else:
            ensure_cache_available(cfg, "feature", split="test", logger=logger.log)
        if use_cacd(cfg):
            _, cacd_reason = check_cacd_feature_cache(cfg, "test")
            logger.log(f"[Cache] CACD F10/F11:test ready | {cacd_reason}")
        for dataset_name in cfg.TEST_DATASETS:
            eval_dataset(
                cfg,
                student,
                dataset_name,
                device,
                out_dir,
                logger,
                max_samples=args.max_samples,
                online_dino=online_dino,
            )


if __name__ == "__main__":
    main()
