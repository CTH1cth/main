import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.icar_probe_utils import (  # noqa: E402
    VARIANT_DIMS,
    auroc_score,
    binary_metrics_from_scores,
    build_episode_masks,
    build_split_lookup,
    choose_threshold,
    correlation,
    ensure_dir,
    evaluate_scores,
    macro_metrics,
    make_probe,
    normalize_channels,
    prototype_features,
    r2_score_1d,
    select_stratified_indices,
    self_support_indices,
    spearman,
    stable_hash_int,
    stable_take_indices,
    standardize_apply,
    standardize_fit,
    stratified_image_split,
    variant_columns,
    write_json,
    zip_report,
)
from common.utils import load_config, set_seed, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    compute_dino_core_margin_68,
    extract_logits,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DATASETS = ("TR-CAMO", "TR-COD10K")
CHECKPOINTS = (15, 20)
RAW_DIM = 447
REPORT_FILES = (
    "summary.md",
    "summary.json",
    "split_manifest.json",
    "normalization_stats.json",
    "metrics_all.csv",
    "metrics_hard.csv",
    "metrics_teacher_conflict.csv",
    "metrics_by_source.csv",
    "metrics_by_checkpoint.csv",
    "correlation_analysis.csv",
    "support_statistics.csv",
    "per_image_metrics.csv",
    "ambiguous_extent_audit.csv",
)


def parse_csv(value, cast=str):
    if value is None or str(value).strip() == "":
        return []
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ICAR-SP-v1 separability probe. This script is read-only for the original model."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt15", required=True)
    parser.add_argument("--ckpt20", required=True)
    parser.add_argument("--max-images-per-source", type=int, default=-1)
    parser.add_argument("--split-seed", type=int, default=3407)
    parser.add_argument("--support-split-seeds", default="3407,3408,3409")
    parser.add_argument("--probe-seeds", default="3407,3408,3409")
    parser.add_argument("--variants", default="V0,V1,V2,V3,V4,V5,V6")
    parser.add_argument("--probe-types", default="linear,mlp")
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--max-probe-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_device(value):
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def infer_in_channels(state):
    for key in ("base_head.weight", "proj.weight", "coarse_path.base_head.weight"):
        if key in state:
            return int(state[key].shape[1])
    raise KeyError("Cannot infer in_channels from checkpoint student state.")


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def max_state_diff(a, b):
    if set(a) != set(b):
        return float("inf")
    max_diff = 0.0
    for key in a:
        va = a[key].detach().cpu()
        vb = b[key].detach().cpu()
        if va.shape != vb.shape:
            return float("inf")
        if torch.is_floating_point(va):
            max_diff = max(max_diff, float((va - vb).abs().max().item()))
        elif not torch.equal(va, vb):
            return float("inf")
    return max_diff


def load_frozen_models(cfg, ckpt_path, epoch, device):
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    if int(checkpoint.get("epoch", epoch)) != int(epoch):
        raise RuntimeError(f"Checkpoint epoch mismatch: {ckpt_path} stores {checkpoint.get('epoch')}, expected {epoch}.")
    if "student" not in checkpoint or "teacher" not in checkpoint:
        raise RuntimeError(f"ICAR-SP requires checkpoint student and teacher states: {ckpt_path}")
    student_state = checkpoint["student"]
    teacher_state = checkpoint["teacher"]
    in_channels = infer_in_channels(student_state)
    student = build_seg_head(in_channels, cfg).to(device)
    teacher = build_seg_head(in_channels, cfg).to(device)
    student.load_state_dict(student_state, strict=True)
    teacher.load_state_dict(teacher_state, strict=True)
    set_model_epoch(student, int(epoch))
    set_model_epoch(teacher, int(epoch))
    student.eval()
    teacher.eval()
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return student, teacher, checkpoint


def to_68(logits, cfg):
    if logits.ndim != 4:
        raise RuntimeError(f"Expected 4D logits, got {list(logits.shape)}")
    if tuple(logits.shape[-2:]) == (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)):
        return logits
    return F.interpolate(logits, size=(int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)), mode="bilinear", align_corners=False)


def output_logits(output, key, cfg, fallback=None):
    if isinstance(output, dict) and key in output:
        return to_68(output[key], cfg)
    if fallback is not None:
        return to_68(fallback, cfg)
    raise RuntimeError(f"Model output missing required key: {key}")


def validate_config(cfg):
    if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "dagp_safe":
        raise RuntimeError("ICAR-SP-v1 expects Long35 HEAD_TYPE='dagp_safe'.")
    if not bool(getattr(cfg, "USE_NDR_BRANCH", False)):
        raise RuntimeError("ICAR-SP-v1 requires USE_NDR_BRANCH=True.")
    if not bool(getattr(cfg, "USE_DABE_PU", False)):
        raise RuntimeError("ICAR-SP-v1 requires DABE-PU fields from CachedTrainDataset.")
    if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() != "pu_v11":
        raise RuntimeError("ICAR-SP-v1 expects DABE_PU_VERSION='pu_v11'.")


def append_sample_rows(
    rows,
    meta,
    raw_features,
    labels,
    split,
    dataset,
    stem,
    epoch,
    support_seed,
    final_prob,
    final_logit,
    esa_margin,
    teacher_prob,
):
    n = int(labels.shape[0])
    rows.append(raw_features.astype(np.float32))
    meta["label"].append(labels.astype(np.float32))
    meta["split"].extend([split] * n)
    meta["dataset"].extend([dataset] * n)
    meta["stem"].extend([stem] * n)
    meta["image_id"].extend([f"{dataset}/{stem}"] * n)
    meta["epoch"].extend([int(epoch)] * n)
    meta["support_seed"].extend([int(support_seed)] * n)
    meta["final_prob"].append(final_prob.astype(np.float32))
    meta["final_logit"].append(final_logit.astype(np.float32))
    meta["esa_margin"].append(esa_margin.astype(np.float32))
    meta["teacher_prob"].append(teacher_prob.astype(np.float32))
    meta["student_hard"].append(((labels == 1.0) & (final_prob < 0.5)) | ((labels == 0.0) & (final_prob >= 0.5)))
    teacher_binary = teacher_prob >= 0.5
    meta["teacher_conflict"].append(((labels == 1.0) & (~teacher_binary)) | ((labels == 0.0) & teacher_binary))


def flatten_meta(meta):
    out = {}
    for key, value in meta.items():
        if key in {"split", "dataset", "stem", "image_id", "epoch", "support_seed"}:
            out[key] = np.asarray(value)
        else:
            out[key] = np.concatenate(value, axis=0) if value else np.asarray([])
    out["combined_hard"] = out["student_hard"] | out["teacher_conflict"]
    return out


def flatten_amb_meta(meta):
    out = {}
    for key, value in meta.items():
        if key in {"split", "dataset", "stem", "image_id", "epoch", "support_seed"}:
            out[key] = np.asarray(value)
        else:
            out[key] = np.concatenate(value, axis=0) if value else np.asarray([])
    return out


def build_raw_feature_rows(
    v0_maps,
    esa_margin_map,
    dino_68,
    detail_68,
    image_68,
    sobel_68,
    dabe_support,
    self_support,
    query_idx,
):
    h, w = v0_maps.shape[-2:]
    flat_idx = query_idx.astype(np.int64)
    v0 = v0_maps.reshape(7, h * w).T[flat_idx]
    v1 = esa_margin_map.reshape(1, h * w).T[flat_idx]
    dino = dino_68.reshape(384, h * w).T[flat_idx]
    detail = detail_68.reshape(32, h * w).T[flat_idx]
    rgb = image_68.reshape(3, h * w).T[flat_idx]
    sobel = sobel_68.reshape(1, h * w).T[flat_idx]
    local = np.concatenate([detail, rgb, sobel, v0], axis=1)
    return np.concatenate([v0, v1, dino, local, dabe_support, self_support], axis=1).astype(np.float32)


def extract_checkpoint_samples(cfg, dataset, selected_pairs, split_lookup, support_seeds, epoch, student, teacher, device, args):
    loader = DataLoader(
        Subset(dataset, [idx for idx, _ in selected_pairs]),
        batch_size=int(args.extract_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    rows = []
    amb_rows = []
    meta = defaultdict(list)
    amb_meta = defaultdict(list)
    support_stats = []
    equivalence = {"final": 0.0, "coarse": 0.0, "base": 0.0, "checked": False}
    model_state_before = {"student": clone_state(student), "teacher": clone_state(teacher)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if any(key.lower().startswith("gt") for key in batch.keys()):
                raise RuntimeError(f"ICAR-SP training batch contains GT-like field: {sorted(batch.keys())}")
            model_input = make_model_input(cfg, batch, device)
            image_68 = make_image_68(cfg, batch, device)
            output = student(model_input, image_68=image_68, return_aux=True, return_probe_aux=True)
            if not isinstance(output, dict) or "probe_ndr_detail_feat" not in output:
                raise RuntimeError("Student output missing probe_ndr_detail_feat. Check return_probe_aux support.")
            if not equivalence["checked"]:
                output_no_probe = student(model_input, image_68=image_68, return_aux=True, return_probe_aux=False)
                for name, key in (("final", "logits"), ("coarse", "coarse_logits_68"), ("base", "base_logits")):
                    lhs = output_logits(output, key if key != "logits" else "logits", cfg)
                    rhs = output_logits(output_no_probe, key if key != "logits" else "logits", cfg)
                    equivalence[name] = float((lhs - rhs).abs().max().item())
                equivalence["checked"] = True
            teacher_output = teacher(model_input, image_68=image_68, return_aux=False)
            final_logits = resize_logits_for_loss(extract_logits(output), cfg).detach()
            teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), cfg).detach()
            base_logits = output_logits(output, "base_logits", cfg).detach()
            coarse_logits = output_logits(output, "coarse_logits_68", cfg, fallback=output.get("coarse_logits")).detach()
            final_prob = final_logits.sigmoid()
            teacher_prob = teacher_logits.sigmoid()
            coarse_prob = coarse_logits.sigmoid()
            coarse_uncert = torch.clamp(1.0 - 2.0 * (coarse_prob - 0.5).abs(), 0.0, 1.0)
            final_uncert = torch.clamp(1.0 - 2.0 * (final_prob - 0.5).abs(), 0.0, 1.0)
            esa_margin, _ = compute_dino_core_margin_68(
                cfg,
                batch,
                batch["pu_fg_core"].to(device, non_blocking=True).float(),
                batch["pu_bg_core"].to(device, non_blocking=True).float(),
                (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                device,
                prefix="ESA",
            )
            dino_68 = F.interpolate(
                batch["feature"].to(device, non_blocking=True).float(),
                size=(int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                mode="bilinear",
                align_corners=False,
            )
            dino_68 = F.normalize(dino_68, dim=1).detach()
            detail_68 = F.normalize(output["probe_ndr_detail_feat"].detach().float(), dim=1)
            sobel_68 = output["sobel_68"].detach().float()
            image_68_det = image_68.detach().float()
            v0_maps = torch.cat(
                [base_logits, coarse_logits, final_logits, coarse_prob, final_prob, coarse_uncert, final_uncert],
                dim=1,
            ).detach()
            batch_size = int(final_logits.shape[0])
            for local_idx in range(batch_size):
                dataset_name = batch["dataset"][local_idx]
                stem = batch["stem"][local_idx]
                split = split_lookup[(dataset_name, stem)]
                fg_core = batch["pu_fg_core"][local_idx, 0].numpy() > 0.5
                bg_core = (batch["pu_bg_core"][local_idx, 0].numpy() > 0.5) & (~fg_core)
                extent = batch["pu_extent"][local_idx, 0].numpy() > 0.5

                v0_np = v0_maps[local_idx].cpu().numpy()
                esa_np = esa_margin[local_idx].detach().cpu().numpy()
                dino_np = dino_68[local_idx].cpu().numpy()
                detail_np = detail_68[local_idx].cpu().numpy()
                image_np = image_68_det[local_idx].cpu().numpy()
                sobel_np = sobel_68[local_idx].cpu().numpy()
                final_prob_np = final_prob[local_idx, 0].detach().cpu().numpy()
                final_logit_np = final_logits[local_idx, 0].detach().cpu().numpy()
                teacher_prob_np = teacher_prob[local_idx, 0].detach().cpu().numpy()
                esa_hw = esa_np[0]

                for support_seed in support_seeds:
                    episode, invalid_reason = build_episode_masks(
                        fg_core, bg_core, dataset_name, stem, int(support_seed)
                    )
                    stat = {
                        "checkpoint_epoch": int(epoch),
                        "dataset": dataset_name,
                        "stem": stem,
                        "split": split,
                        "support_seed": int(support_seed),
                        "valid_episode": episode is not None,
                        "invalid_reason": invalid_reason,
                        "self_fg_fallback": False,
                        "self_bg_fallback": False,
                    }
                    if episode is None:
                        support_stats.append(stat)
                        continue
                    dabe_dino = prototype_features(
                        dino_np, episode["fg_support_idx"], episode["bg_support_idx"], episode["query_idx"]
                    )
                    dabe_detail = prototype_features(
                        detail_np, episode["fg_support_idx"], episode["bg_support_idx"], episode["query_idx"]
                    )
                    dabe_support = np.concatenate([dabe_dino, dabe_detail], axis=1)
                    self_fg_idx, self_bg_idx, fg_fb, bg_fb, self_reason = self_support_indices(
                        final_prob_np,
                        episode["query_mask"],
                        dataset_name,
                        stem,
                        int(support_seed),
                    )
                    stat.update(
                        {
                            "self_fg_fallback": bool(fg_fb),
                            "self_bg_fallback": bool(bg_fb),
                            "self_both_fallback": bool(fg_fb and bg_fb),
                            "self_invalid_reason": self_reason,
                        }
                    )
                    if self_fg_idx is None or self_bg_idx is None:
                        support_stats.append(stat)
                        continue
                    self_dino = prototype_features(dino_np, self_fg_idx, self_bg_idx, episode["query_idx"])
                    self_detail = prototype_features(detail_np, self_fg_idx, self_bg_idx, episode["query_idx"])
                    self_support = np.concatenate([self_dino, self_detail], axis=1)
                    raw = build_raw_feature_rows(
                        v0_np,
                        esa_np,
                        dino_np,
                        detail_np,
                        image_np,
                        sobel_np,
                        dabe_support,
                        self_support,
                        episode["query_idx"],
                    )
                    q_final_prob = final_prob_np.reshape(-1)[episode["query_idx"]]
                    q_final_logit = final_logit_np.reshape(-1)[episode["query_idx"]]
                    q_esa = esa_hw.reshape(-1)[episode["query_idx"]]
                    q_teacher = teacher_prob_np.reshape(-1)[episode["query_idx"]]
                    append_sample_rows(
                        rows,
                        meta,
                        raw,
                        episode["labels"],
                        split,
                        dataset_name,
                        stem,
                        epoch,
                        support_seed,
                        q_final_prob,
                        q_final_logit,
                        q_esa,
                        q_teacher,
                    )

                    amb_mask = extent & (teacher_prob_np < 0.5)
                    amb_idx = stable_take_indices(amb_mask, dataset_name, stem, int(support_seed) + 9999, 128)
                    if amb_idx.size > 0:
                        amb_dabe = np.concatenate(
                            [
                                prototype_features(dino_np, episode["fg_support_idx"], episode["bg_support_idx"], amb_idx),
                                prototype_features(detail_np, episode["fg_support_idx"], episode["bg_support_idx"], amb_idx),
                            ],
                            axis=1,
                        )
                        amb_self = np.concatenate(
                            [
                                prototype_features(dino_np, self_fg_idx, self_bg_idx, amb_idx),
                                prototype_features(detail_np, self_fg_idx, self_bg_idx, amb_idx),
                            ],
                            axis=1,
                        )
                        amb_raw = build_raw_feature_rows(
                            v0_np, esa_np, dino_np, detail_np, image_np, sobel_np, amb_dabe, amb_self, amb_idx
                        )
                        amb_rows.append(amb_raw)
                        n_amb = int(amb_raw.shape[0])
                        amb_meta["split"].extend([split] * n_amb)
                        amb_meta["dataset"].extend([dataset_name] * n_amb)
                        amb_meta["stem"].extend([stem] * n_amb)
                        amb_meta["image_id"].extend([f"{dataset_name}/{stem}"] * n_amb)
                        amb_meta["epoch"].extend([int(epoch)] * n_amb)
                        amb_meta["support_seed"].extend([int(support_seed)] * n_amb)
                        amb_meta["final_prob"].append(final_prob_np.reshape(-1)[amb_idx].astype(np.float32))
                        amb_meta["final_logit"].append(final_logit_np.reshape(-1)[amb_idx].astype(np.float32))
                        amb_meta["esa_margin"].append(esa_hw.reshape(-1)[amb_idx].astype(np.float32))
                    support_stats.append(stat)

    model_state_after = {"student": clone_state(student), "teacher": clone_state(teacher)}
    state_diff = {
        role: max_state_diff(model_state_before[role], model_state_after[role]) for role in ("student", "teacher")
    }
    data = {
        "x": np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, RAW_DIM), dtype=np.float32),
        "meta": flatten_meta(meta),
        "amb_x": np.concatenate(amb_rows, axis=0).astype(np.float32)
        if amb_rows
        else np.zeros((0, RAW_DIM), dtype=np.float32),
        "amb_meta": flatten_amb_meta(amb_meta) if amb_rows else {},
        "support_stats": support_stats,
        "equivalence": equivalence,
        "state_diff": state_diff,
    }
    return data


def subset_mask(meta, split="test", source="Mixed", subset="all_core"):
    mask = meta["split"] == split
    if source != "Mixed":
        mask = mask & (meta["dataset"] == source)
    if subset == "all_core":
        return mask
    if subset == "student_hard":
        return mask & meta["student_hard"]
    if subset == "teacher_conflict":
        return mask & meta["teacher_conflict"]
    if subset == "combined_hard":
        return mask & meta["combined_hard"]
    raise ValueError(f"Unknown subset: {subset}")


def predict_logits(model, x, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = torch.from_numpy(x[start : start + int(batch_size)]).to(device)
            outputs.append(model(xb).squeeze(1).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0,), dtype=np.float32)


def val_score_for_early_stop(y, logits, meta):
    mask = subset_mask(meta, split="val", source="Mixed", subset="combined_hard")
    if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
        mask = subset_mask(meta, split="val", source="Mixed", subset="all_core")
    if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
        return float("nan")
    image_ids = meta["image_id"][mask]
    score = macro_metrics(y[mask], logits[mask], image_ids, threshold=0.5).get("auroc", float("nan"))
    if math.isnan(score):
        score = auroc_score(y[mask], logits[mask])
    return float(score)


def train_probe(x_variant, y, meta, probe_type, seed, args, device):
    set_seed(int(seed))
    train_mask = meta["split"] == "train"
    val_mask = meta["split"] == "val"
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Train/val split has no query pixels.")
    mean, std = standardize_fit(x_variant[train_mask])
    x_train = standardize_apply(x_variant[train_mask], mean, std)
    y_train = y[train_mask].astype(np.float32)
    x_all = standardize_apply(x_variant, mean, std)
    model = make_probe(probe_type, x_variant.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(args.query_batch_size),
        shuffle=True,
        drop_last=False,
    )
    best_state = None
    best_score = -float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, int(args.max_probe_epochs) + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
        logits_all = predict_logits(model, x_all, device)
        score = val_score_for_early_stop(y, logits_all, meta)
        if math.isnan(score):
            score = -float("inf")
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(args.early_stop_patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    logits_all = predict_logits(model, x_all, device)
    val_threshold_mask = subset_mask(meta, split="val", source="Mixed", subset="combined_hard")
    if int(val_threshold_mask.sum()) < 10 or len(np.unique(y[val_threshold_mask])) < 2:
        val_threshold_mask = subset_mask(meta, split="val", source="Mixed", subset="all_core")
    threshold = choose_threshold(y[val_threshold_mask], logits_all[val_threshold_mask])
    return model, logits_all, mean, std, threshold, {"best_epoch": best_epoch, "best_val_macro_auroc": best_score}


def evaluate_run(epoch, variant, probe_type, seed, y, logits, meta, threshold):
    rows = []
    per_image = []
    for subset in ("all_core", "student_hard", "teacher_conflict", "combined_hard"):
        for source in ("Mixed", "TR-CAMO", "TR-COD10K"):
            mask = subset_mask(meta, split="test", source=source, subset=subset)
            if int(mask.sum()) == 0 or len(np.unique(y[mask])) < 2:
                metrics = {"num_samples": int(mask.sum()), "auroc": float("nan")}
                macro = {}
            else:
                metrics = evaluate_scores(y[mask], logits[mask], threshold=threshold)
                macro = macro_metrics(y[mask], logits[mask], meta["image_id"][mask], threshold=threshold)
            row = {
                "checkpoint_epoch": int(epoch),
                "variant": variant,
                "probe_type": probe_type,
                "seed": int(seed),
                "subset": subset,
                "source": source,
                "threshold": float(threshold),
            }
            for key, value in metrics.items():
                row[f"micro_{key}"] = value
            for key, value in macro.items():
                row[f"macro_{key}"] = value
            rows.append(row)
            if source != "Mixed":
                image_ids = sorted(set(meta["image_id"][mask]))
                for image_id in image_ids:
                    imask = mask & (meta["image_id"] == image_id)
                    if int(imask.sum()) == 0 or len(np.unique(y[imask])) < 2:
                        continue
                    im = evaluate_scores(y[imask], logits[imask], threshold=threshold)
                    dataset, stem = image_id.split("/", 1)
                    per_image.append(
                        {
                            "checkpoint_epoch": int(epoch),
                            "variant": variant,
                            "probe_type": probe_type,
                            "seed": int(seed),
                            "subset": subset,
                            "dataset": dataset,
                            "stem": stem,
                            **{f"micro_{k}": v for k, v in im.items()},
                        }
                    )
    return rows, per_image


def evaluate_correlations(epoch, variant, probe_type, seed, y, logits, meta):
    rows = []
    for source in ("Mixed", "TR-CAMO", "TR-COD10K"):
        mask = subset_mask(meta, split="test", source=source, subset="all_core")
        if int(mask.sum()) < 2:
            continue
        rows.append(
            {
                "checkpoint_epoch": int(epoch),
                "variant": variant,
                "probe_type": probe_type,
                "seed": int(seed),
                "source": source,
                "pearson_score_final_prob": correlation(logits[mask], meta["final_prob"][mask]),
                "spearman_score_final_prob": spearman(logits[mask], meta["final_prob"][mask]),
                "pearson_score_esa_margin": correlation(logits[mask], meta["esa_margin"][mask]),
                "spearman_score_esa_margin": spearman(logits[mask], meta["esa_margin"][mask]),
                "r2_score_from_final_logit": r2_score_1d(meta["final_logit"][mask], logits[mask]),
            }
        )
    return rows


def evaluate_ambiguous(epoch, variant, probe_type, seed, model, mean, std, data, device):
    if data["amb_x"].shape[0] == 0:
        return []
    cols = variant_columns(variant)
    x = standardize_apply(data["amb_x"][:, cols], mean, std)
    logits = predict_logits(model, x, device)
    meta = data["amb_meta"]
    rows = []
    for source in ("Mixed", "TR-CAMO", "TR-COD10K"):
        mask = meta["split"] == "test"
        if source != "Mixed":
            mask = mask & (meta["dataset"] == source)
        if int(mask.sum()) == 0:
            continue
        pos_ratio = float((logits[mask] > 0.0).mean())
        rows.append(
            {
                "checkpoint_epoch": int(epoch),
                "variant": variant,
                "probe_type": probe_type,
                "seed": int(seed),
                "source": source,
                "num_pixels": int(mask.sum()),
                "score_mean": float(logits[mask].mean()),
                "score_std": float(logits[mask].std()),
                "positive_score_ratio": pos_ratio,
                "negative_score_ratio": float((logits[mask] < 0.0).mean()),
                "pearson_score_final_prob": correlation(logits[mask], meta["final_prob"][mask]),
                "spearman_score_final_prob": spearman(logits[mask], meta["final_prob"][mask]),
                "pearson_score_esa_margin": correlation(logits[mask], meta["esa_margin"][mask]),
                "spearman_score_esa_margin": spearman(logits[mask], meta["esa_margin"][mask]),
                "one_sided_warning": bool(pos_ratio > 0.90 or pos_ratio < 0.10),
            }
        )
    return rows


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_by(rows, group_keys, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out = []
    for key, group in grouped.items():
        item = {name: value for name, value in zip(group_keys, key)}
        for metric in metric_keys:
            values = [float(row[metric]) for row in group if metric in row and row[metric] not in ("", None)]
            values = [value for value in values if not math.isnan(value)]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values))
        out.append(item)
    return out


def summarize_support(support_stats):
    grouped = defaultdict(list)
    for row in support_stats:
        grouped[(row["checkpoint_epoch"], row["dataset"], row["split"])].append(row)
    out = []
    for (epoch, dataset, split), rows in grouped.items():
        valid = [row for row in rows if row["valid_episode"] and not row.get("self_invalid_reason")]
        out.append(
            {
                "checkpoint_epoch": epoch,
                "dataset": dataset,
                "split": split,
                "num_episode_attempts": len(rows),
                "valid_episode_ratio": len(valid) / max(len(rows), 1),
                "self_fg_fallback_ratio": sum(bool(row.get("self_fg_fallback")) for row in valid) / max(len(valid), 1),
                "self_bg_fallback_ratio": sum(bool(row.get("self_bg_fallback")) for row in valid) / max(len(valid), 1),
                "self_both_fallback_ratio": sum(bool(row.get("self_both_fallback")) for row in valid) / max(len(valid), 1),
            }
        )
    return out


def automatic_verdict(metric_rows, corr_rows, support_rows):
    def mean_metric(epoch, variant, source, metric):
        vals = [
            float(row.get(metric, float("nan")))
            for row in metric_rows
            if int(row.get("checkpoint_epoch", -1)) == int(epoch)
            and row.get("variant") == variant
            and str(row.get("probe_type")).lower() in {"mlp", "tinymlp"}
            and row.get("subset") == "combined_hard"
            and row.get("source") == source
        ]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    v6_ok = True
    for epoch in CHECKPOINTS:
        if mean_metric(epoch, "V6", "Mixed", "macro_auroc") < 0.80:
            v6_ok = False
        if mean_metric(epoch, "V6", "Mixed", "macro_balanced_accuracy") < 0.75:
            v6_ok = False
        for source in DATASETS:
            if mean_metric(epoch, "V6", source, "macro_auroc") < 0.78:
                v6_ok = False
        if abs(mean_metric(epoch, "V6", "TR-CAMO", "macro_auroc") - mean_metric(epoch, "V6", "TR-COD10K", "macro_auroc")) > 0.08:
            v6_ok = False
    baseline_best = -float("inf")
    v6_best = float("inf")
    for epoch in CHECKPOINTS:
        baseline_best = max(
            baseline_best,
            mean_metric(epoch, "V0", "Mixed", "macro_auroc"),
            mean_metric(epoch, "V1", "Mixed", "macro_auroc"),
        )
        v6_best = min(v6_best, mean_metric(epoch, "V6", "Mixed", "macro_auroc"))
    v6_corr_ok = True
    for row in corr_rows:
        if row.get("variant") == "V6" and str(row.get("probe_type")).lower() in {"mlp", "tinymlp"} and row.get("source") == "Mixed":
            if abs(float(row.get("pearson_score_final_prob", 1.0))) >= 0.95:
                v6_corr_ok = False
            if abs(float(row.get("spearman_score_final_prob", 1.0))) >= 0.95:
                v6_corr_ok = False
    v6_seed_std_ok = True
    for epoch in CHECKPOINTS:
        vals = [
            float(row.get("macro_auroc", float("nan")))
            for row in metric_rows
            if int(row.get("checkpoint_epoch", -1)) == int(epoch)
            and row.get("variant") == "V6"
            and str(row.get("probe_type")).lower() in {"mlp", "tinymlp"}
            and row.get("subset") == "combined_hard"
            and row.get("source") == "Mixed"
        ]
        vals = [v for v in vals if not math.isnan(v)]
        if len(vals) >= 2 and float(np.std(vals)) > 0.02:
            v6_seed_std_ok = False
    fallback_ok = True
    for row in support_rows:
        if row["split"] == "test" and float(row.get("self_both_fallback_ratio", 0.0)) > 0.30:
            fallback_ok = False
    if v6_ok and v6_best >= baseline_best + 0.05 and v6_corr_ok and v6_seed_std_ok and fallback_ok:
        return "S1", (
            "S1: Current features support an inference-compatible image-conditioned ambiguity resolver. "
            "A full ICAR-v1 experiment is justified."
        )
    v5_ok = all(
        mean_metric(epoch, "V5", "Mixed", "macro_auroc") >= 0.80
        and mean_metric(epoch, "V5", "Mixed", "macro_balanced_accuracy") >= 0.75
        for epoch in CHECKPOINTS
    )
    if v5_ok and (v6_best < 0.75 or v6_best < min(mean_metric(e, "V5", "Mixed", "macro_auroc") for e in CHECKPOINTS) - 0.07 or not fallback_ok):
        return "S2", (
            "S2: The representation is separable only with DABE support. Do not start full ICAR yet. "
            "The train-inference support gap must be solved first."
        )
    v4_gain = min(mean_metric(e, "V4", "Mixed", "macro_auroc") for e in CHECKPOINTS) - baseline_best
    if v4_gain >= 0.03:
        return "S3", (
            "S3: Local semantic-detail features add useful information, but image-conditioned ambiguity "
            "resolution is not yet reliable."
        )
    return "S4", (
        "S4: Current DINO/NDR/local features do not provide sufficient new ambiguity information. "
        "Stop the ICAR route with the current feature source."
    )


def make_plots(out_dir, metric_rows, corr_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots = ensure_dir(Path(out_dir) / "plots")
    variants = sorted({row["variant"] for row in metric_rows})
    for metric, filename, title in (
        ("macro_auroc", "hard_auroc_by_variant.png", "Combined-hard macro AUROC"),
        ("macro_balanced_accuracy", "hard_balanced_accuracy_by_variant.png", "Combined-hard macro balanced accuracy"),
    ):
        vals = []
        for variant in variants:
            v = [
                float(row.get(metric, float("nan")))
                for row in metric_rows
                if row.get("subset") == "combined_hard" and row.get("source") == "Mixed" and row.get("variant") == variant
            ]
            vals.append(float(np.nanmean(v)) if v else float("nan"))
        plt.figure(figsize=(8, 4))
        plt.bar(variants, vals)
        plt.ylim(0, 1)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plots / filename, dpi=160)
        plt.close()
    gaps = []
    for variant in variants:
        camo = [
            float(row.get("macro_auroc", float("nan")))
            for row in metric_rows
            if row.get("subset") == "combined_hard" and row.get("source") == "TR-CAMO" and row.get("variant") == variant
        ]
        cod = [
            float(row.get("macro_auroc", float("nan")))
            for row in metric_rows
            if row.get("subset") == "combined_hard" and row.get("source") == "TR-COD10K" and row.get("variant") == variant
        ]
        gaps.append(abs(float(np.nanmean(camo)) - float(np.nanmean(cod))) if camo and cod else float("nan"))
    plt.figure(figsize=(8, 4))
    plt.bar(variants, gaps)
    plt.title("Source gap by variant")
    plt.tight_layout()
    plt.savefig(plots / "source_gap_by_variant.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    for variant in variants:
        xs, ys = [], []
        for epoch in CHECKPOINTS:
            vals = [
                float(row.get("macro_auroc", float("nan")))
                for row in metric_rows
                if row.get("subset") == "combined_hard"
                and row.get("source") == "Mixed"
                and row.get("variant") == variant
                and int(row.get("checkpoint_epoch")) == int(epoch)
            ]
            if vals:
                xs.append(epoch)
                ys.append(float(np.nanmean(vals)))
        if xs:
            plt.plot(xs, ys, marker="o", label=variant)
    plt.legend(ncol=2, fontsize=8)
    plt.title("Checkpoint stability")
    plt.tight_layout()
    plt.savefig(plots / "checkpoint_stability.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 4))
    xs, ys = [], []
    for row in corr_rows:
        if row.get("source") == "Mixed":
            xs.append(row.get("variant"))
            ys.append(abs(float(row.get("pearson_score_final_prob", float("nan")))))
    if xs:
        order = sorted(set(xs))
        vals = [float(np.nanmean([y for x, y in zip(xs, ys) if x == key])) for key in order]
        plt.bar(order, vals)
    plt.title("Abs Pearson(score, final_prob)")
    plt.tight_layout()
    plt.savefig(plots / "score_probability_correlation.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    variants = [item.upper() for item in parse_csv(args.variants)]
    probe_types = [item.lower() for item in parse_csv(args.probe_types)]
    probe_seeds = parse_csv(args.probe_seeds, int)
    support_seeds = parse_csv(args.support_split_seeds, int)
    for variant in variants:
        if variant not in VARIANT_DIMS:
            raise RuntimeError(f"Unsupported variant: {variant}")
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out)
    set_seed(int(args.split_seed))

    dataset = CachedTrainDataset(cfg, max_samples=-1)
    selected_pairs = select_stratified_indices(dataset, args.max_images_per_source, args.split_seed)
    split_manifest = stratified_image_split([item for _, item in selected_pairs], seed=args.split_seed)
    write_json(out_dir / "split_manifest.json", split_manifest)
    split_lookup = build_split_lookup(split_manifest)

    ckpts = {15: Path(args.ckpt15), 20: Path(args.ckpt20)}
    all_data = {}
    extraction_summary = {}
    for epoch in CHECKPOINTS:
        print(f"[ICAR-SP] extracting checkpoint epoch {epoch}", flush=True)
        student, teacher, _ = load_frozen_models(cfg, ckpts[epoch], epoch, device)
        data = extract_checkpoint_samples(
            cfg, dataset, selected_pairs, split_lookup, support_seeds, epoch, student, teacher, device, args
        )
        for role in ("student", "teacher"):
            if data["state_diff"][role] != 0.0:
                raise RuntimeError(f"{role} state changed during extraction: max_diff={data['state_diff'][role]}")
        if max(data["equivalence"]["final"], data["equivalence"]["coarse"], data["equivalence"]["base"]) > 1e-6:
            raise RuntimeError(f"return_probe_aux numerical equivalence failed: {data['equivalence']}")
        if data["x"].shape[1] != RAW_DIM:
            raise RuntimeError(f"Raw feature dimension mismatch: {data['x'].shape[1]} != {RAW_DIM}")
        all_data[epoch] = data
        extraction_summary[epoch] = {
            "num_query_pixels": int(data["x"].shape[0]),
            "num_ambiguous_pixels": int(data["amb_x"].shape[0]),
            "equivalence": data["equivalence"],
            "state_diff": data["state_diff"],
        }
        del student, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_rows = []
    per_image_rows = []
    corr_rows = []
    ambiguous_rows = []
    norm_stats = {}
    train_summaries = []
    for epoch, data in all_data.items():
        y = data["meta"]["label"].astype(np.float32)
        for variant in variants:
            cols = variant_columns(variant)
            x_variant = data["x"][:, cols]
            for probe_type in probe_types:
                for seed in probe_seeds:
                    print(f"[ICAR-SP] train epoch={epoch} variant={variant} probe={probe_type} seed={seed}", flush=True)
                    model, logits, mean, std, threshold, train_info = train_probe(
                        x_variant, y, data["meta"], probe_type, seed, args, device
                    )
                    key = f"epoch{epoch}_{variant}_{probe_type}_seed{seed}"
                    norm_stats[key] = {
                        "columns": cols,
                        "mean": mean.astype(float).tolist(),
                        "std": std.astype(float).tolist(),
                    }
                    train_summaries.append(
                        {
                            "checkpoint_epoch": int(epoch),
                            "variant": variant,
                            "probe_type": probe_type,
                            "seed": int(seed),
                            "threshold": float(threshold),
                            **train_info,
                        }
                    )
                    rows, image_rows = evaluate_run(epoch, variant, probe_type, seed, y, logits, data["meta"], threshold)
                    metric_rows.extend(rows)
                    per_image_rows.extend(image_rows)
                    corr_rows.extend(evaluate_correlations(epoch, variant, probe_type, seed, y, logits, data["meta"]))
                    ambiguous_rows.extend(evaluate_ambiguous(epoch, variant, probe_type, seed, model, mean, std, data, device))
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    support_rows = []
    for data in all_data.values():
        support_rows.extend(summarize_support(data["support_stats"]))
    write_json(out_dir / "normalization_stats.json", norm_stats)

    metrics_all = [row for row in metric_rows if row.get("subset") == "all_core"]
    metrics_hard = [row for row in metric_rows if row.get("subset") in {"student_hard", "combined_hard"}]
    metrics_teacher = [row for row in metric_rows if row.get("subset") == "teacher_conflict"]
    metrics_by_source = metric_rows
    metrics_by_checkpoint = aggregate_by(
        metric_rows,
        ["checkpoint_epoch", "variant", "probe_type", "subset", "source"],
        ["macro_auroc", "macro_balanced_accuracy", "micro_auroc", "micro_balanced_accuracy"],
    )
    write_csv(out_dir / "metrics_all.csv", metrics_all)
    write_csv(out_dir / "metrics_hard.csv", metrics_hard)
    write_csv(out_dir / "metrics_teacher_conflict.csv", metrics_teacher)
    write_csv(out_dir / "metrics_by_source.csv", metrics_by_source)
    write_csv(out_dir / "metrics_by_checkpoint.csv", metrics_by_checkpoint)
    write_csv(out_dir / "correlation_analysis.csv", corr_rows)
    write_csv(out_dir / "support_statistics.csv", support_rows)
    write_csv(out_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(out_dir / "ambiguous_extent_audit.csv", ambiguous_rows)

    verdict_code, verdict_text = automatic_verdict(metric_rows, corr_rows, support_rows)
    summary = {
        "verdict": verdict_code,
        "verdict_text": verdict_text,
        "config": str(Path(args.config).resolve()),
        "ckpt15": str(Path(args.ckpt15).resolve()),
        "ckpt20": str(Path(args.ckpt20).resolve()),
        "device": str(device),
        "variants": variants,
        "probe_types": probe_types,
        "probe_seeds": probe_seeds,
        "support_split_seeds": support_seeds,
        "variant_dims": {variant: VARIANT_DIMS[variant] for variant in variants},
        "linear_probe": "nn.Linear(input_dim, 1)",
        "tiny_mlp_probe": "Linear(input_dim,64)-LayerNorm-GELU-Dropout(0.10)-Linear(64,1)",
        "extraction": extraction_summary,
        "train_summaries": train_summaries,
        "num_selected_images": len(selected_pairs),
        "split_counts": {
            key: sum(1 for row in split_manifest if row["split"] == key) for key in ("train", "val", "test")
        },
    }
    write_json(out_dir / "summary.json", summary)
    make_plots(out_dir, metric_rows, corr_rows)
    summary_md = [
        "# ICAR-SP-v1 Separability Probe",
        "",
        f"Verdict: **{verdict_code}**",
        "",
        verdict_text,
        "",
        "DABE-PU core is a proxy label, not ground truth.",
        "",
        "Passing this probe proves only that reliable core proxies are separable on unseen training images.",
        "",
        "It does not prove that extent/unknown pixels can be labeled correctly, and it does not guarantee CHAMELEON improvement.",
        "",
        "## Inputs",
        f"- Config: `{summary['config']}`",
        f"- epoch15 ckpt: `{summary['ckpt15']}`",
        f"- epoch20 ckpt: `{summary['ckpt20']}`",
        f"- Selected images: `{len(selected_pairs)}`",
        f"- Split counts: `{summary['split_counts']}`",
        "",
        "## Safety checks",
    ]
    for epoch in CHECKPOINTS:
        eq = extraction_summary[epoch]["equivalence"]
        diff = extraction_summary[epoch]["state_diff"]
        summary_md.append(
            f"- epoch{epoch}: return_probe_aux diff final/coarse/base = "
            f"{eq['final']:.3g}/{eq['coarse']:.3g}/{eq['base']:.3g}; "
            f"student/teacher state diff = {diff['student']:.3g}/{diff['teacher']:.3g}"
        )
    summary_md.extend(
        [
            "",
            "## Variant dimensions",
            ", ".join(f"{variant}={VARIANT_DIMS[variant]}" for variant in variants),
            "",
            "## Notes",
            "- No training GT or test GT is read.",
            "- Student and teacher are frozen; only probe classifier parameters are optimized.",
            "- SELF support is selected from final probability on non-query pixels and does not use DABE masks.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")
    zip_path = zip_report(out_dir, "icar_separability_probe.zip", REPORT_FILES)
    print(f"verdict = {verdict_code}", flush=True)
    print(f"zip = {zip_path}", flush=True)


if __name__ == "__main__":
    main()
