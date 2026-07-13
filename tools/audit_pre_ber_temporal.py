import argparse
import csv
import json
import random
import sys
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import ensure_dir, load_config, set_seed, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_esa_ber_loss,
    build_rast_teacher_weight_map,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DATASETS = ("TR-CAMO", "TR-COD10K")
EXPECTED_SOURCE_EPOCHS = (7, 10, 15, 20)
EXPECTED_FUTURE_MAP = {7: (10, 15, 20), 10: (15, 20), 15: (20,), 20: ()}
DEFAULT_BER_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_v2_ber_long35_lrfloor_2e5.py"
)
REPORT_FILES = (
    "summary.md",
    "summary.json",
    "temporal_pairs.csv",
    "per_source_epoch.csv",
    "per_dataset_stats.csv",
    "per_image_stats.csv",
    "candidate_tracks.csv",
)

TEMPORAL_FIELDS = [
    "source_epoch",
    "future_epoch",
    "source_dataset",
    "num_images",
    "valid_images",
    "valid_image_ratio",
    "pos_count",
    "neg_count",
    "source_pos_prob",
    "source_neg_prob",
    "source_prob_gap",
    "source_logit_gap",
    "future_pos_prob",
    "future_neg_prob",
    "future_prob_gap",
    "future_logit_gap",
    "delta_pos_prob",
    "delta_neg_prob",
    "delta_logit_gap",
    "pos_future_fg_ratio",
    "neg_future_fg_ratio",
    "fg_ratio_gap",
    "pos_prob_increase_ratio",
    "neg_prob_decrease_ratio",
    "pair_order_correct_ratio",
    "pair_order_flip_ratio",
    "candidate_spatial_persistence",
    "pair_count",
]

SOURCE_FIELDS = [
    "source_epoch",
    "source_dataset",
    "num_images",
    "valid_images",
    "valid_image_ratio",
    "pos_count",
    "neg_count",
    "neg_extent_count",
    "neg_hard_bg_count",
    "neg_both_count",
    "source_pos_prob",
    "source_neg_prob",
    "source_prob_gap",
    "source_pos_teacher_prob",
    "source_neg_teacher_prob",
    "source_logit_gap",
    "pos_score_mean",
    "neg_score_mean",
    "pos_margin_mean",
    "neg_margin_mean",
    "pos_conn_mean",
    "neg_conn_mean",
]

PER_IMAGE_FIELDS = [
    "source_epoch",
    "future_epoch",
    "dataset",
    "stem",
    "valid",
    "pos_count",
    "neg_count",
    "source_pos_prob",
    "source_neg_prob",
    "source_prob_gap",
    "source_logit_gap",
    "future_pos_prob",
    "future_neg_prob",
    "future_prob_gap",
    "future_logit_gap",
    "delta_pos_prob",
    "delta_neg_prob",
    "delta_logit_gap",
    "pos_future_fg_ratio",
    "neg_future_fg_ratio",
    "fg_ratio_gap",
    "pos_prob_increase_ratio",
    "neg_prob_decrease_ratio",
    "pair_order_correct_ratio",
    "pair_order_flip_ratio",
    "candidate_spatial_persistence",
    "pair_count",
    "pair_order_correct_count",
    "pair_order_flip_count",
    "pos_future_fg_count",
    "neg_future_fg_count",
    "pos_prob_increase_count",
    "neg_prob_decrease_count",
    "persistence_count",
]

CANDIDATE_FIELDS = [
    "source_epoch",
    "future_epoch",
    "dataset",
    "stem",
    "candidate_class",
    "negative_subtype",
    "row",
    "col",
    "flat_index",
    "source_score",
    "source_margin",
    "source_conn_delta",
    "source_student_logit",
    "source_student_prob",
    "source_teacher_prob",
    "source_binary",
    "future_student_logit",
    "future_student_prob",
    "future_binary",
    "prob_delta",
    "binary_persistent",
]


class ConfigOverlay:
    def __init__(self, base, overrides):
        self._base = base
        self._overrides = dict(overrides)

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit pre-reset ESA-v2-BER candidate temporal stability without GT or training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--ber-config", default=str(DEFAULT_BER_CONFIG))
    parser.add_argument("--source-epochs", default="7,10,15,20")
    parser.add_argument("--future-map", default="7:10,15,20;10:15,20;15:20")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def parse_source_epochs(value):
    try:
        epochs = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --source-epochs: {value!r}") from exc
    if epochs != EXPECTED_SOURCE_EPOCHS:
        raise ValueError(
            f"Pre-BER v1 requires source epochs {EXPECTED_SOURCE_EPOCHS}, got {epochs}."
        )
    return epochs


def parse_future_map(value, source_epochs):
    mapping = {epoch: [] for epoch in source_epochs}
    text = str(value).strip()
    if text:
        for group in text.split(";"):
            group = group.strip()
            if not group:
                continue
            if ":" not in group:
                raise ValueError(f"Invalid future-map group without ':': {group!r}")
            source_text, future_text = group.split(":", 1)
            try:
                source = int(source_text.strip())
                futures = [int(item.strip()) for item in future_text.split(",") if item.strip()]
            except ValueError as exc:
                raise ValueError(f"Invalid future-map group: {group!r}") from exc
            if source not in mapping:
                raise ValueError(f"Future-map source {source} is not in {source_epochs}.")
            if mapping[source]:
                raise ValueError(f"Duplicate future-map source: {source}")
            if len(futures) != len(set(futures)):
                raise ValueError(f"Duplicate future epochs for source {source}: {futures}")
            if any(future <= source or future not in source_epochs for future in futures):
                raise ValueError(f"Invalid future relation {source}:{futures}")
            mapping[source] = futures
    normalized = {epoch: tuple(mapping[epoch]) for epoch in source_epochs}
    if normalized != EXPECTED_FUTURE_MAP:
        raise ValueError(
            f"Pre-BER v1 requires future map {EXPECTED_FUTURE_MAP}, got {normalized}."
        )
    return normalized


def resolve_device(value):
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def validate_args(args):
    for label, value in (("config", args.config), ("BER config", args.ber_config)):
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f"{label} not found: {value}")
    workdir = Path(args.workdir).expanduser()
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir not found: {workdir}")
    if int(args.max_images) < 2 or int(args.max_images) > 1000:
        raise ValueError("--max-images must be in [2, 1000].")
    if args.batch_size is not None and int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < -1:
        raise ValueError("--num-workers must be -1 or non-negative.")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive.")


def validate_configs(base_cfg, ber_cfg):
    expected_base = {
        "HEAD_TYPE": "dagp_safe",
        "P_INIT_MODE": "dabe_pu_v11_desplsched",
        "DABE_PU_VERSION": "pu_v11",
        "FINETUNE_RESET_EPOCH": 20,
        "FINETUNE_RESET_TIMING": "after_epoch",
    }
    for key, expected in expected_base.items():
        actual = getattr(base_cfg, key, None)
        if str(actual).lower() != str(expected).lower():
            raise RuntimeError(f"Base config mismatch: {key}={actual!r}, expected {expected!r}.")
    required_ber = {
        "USE_ESA_BER": True,
        "DAGP_SAFE_RETURN_GRAPH_AUX_FOR_BER": True,
        "ESA_BER_GRAPH_WEIGHT_SOURCE": "semantic_topk",
        "ESA_BER_DETACH_CANDIDATES": True,
        "ESA_BER_DETACH_TEACHER": True,
        "ESA_BER_DETACH_GRAPH_EVIDENCE": True,
        "ESA_BER_DETACH_MARGIN": True,
    }
    for key, expected in required_ber.items():
        actual = getattr(ber_cfg, key, None)
        if str(actual).lower() != str(expected).lower():
            raise RuntimeError(f"BER config mismatch: {key}={actual!r}, expected {expected!r}.")
    if str(getattr(ber_cfg, "HEAD_TYPE", "")).lower() != "dagp_safe":
        raise RuntimeError("BER candidate config must use HEAD_TYPE='dagp_safe'.")


def build_audit_config(ber_cfg):
    return ConfigOverlay(
        ber_cfg,
        {
            "ESA_BER_AFTER_RESET_ONLY": False,
            "ESA_BER_START_EPOCH": 7,
            "ESA_BER_RAMP_END_EPOCH": 7,
            "ESA_BER_STOP_EPOCH": 21,
            "ESA_BER_LAMBDA_MAX": 0.0,
            "ESA_POST_RESET_ENABLE": False,
            "RAST_POST_RESET_ENABLE": False,
            "RAST_POST_RESET_SCALE": 0.0,
        },
    )


def max_state_diff(state_a, state_b):
    if set(state_a) != set(state_b):
        return float("inf")
    maximum = 0.0
    for name in state_a:
        a = state_a[name].detach().cpu()
        b = state_b[name].detach().cpu()
        if a.shape != b.shape:
            return float("inf")
        if torch.is_floating_point(a) or torch.is_complex(a):
            maximum = max(maximum, float((a - b).abs().max().item()))
        elif not torch.equal(a, b):
            return float("inf")
    return maximum


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def checkpoint_path(workdir, epoch):
    return Path(workdir).expanduser() / "train" / "ckpt" / f"epoch_{int(epoch):03d}.pth"


def load_models(base_cfg, audit_cfg, workdir, source_epochs, device):
    models = {}
    metadata = {}
    for epoch in source_epochs:
        path = checkpoint_path(workdir, epoch)
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {path}")
        checkpoint = torch_load(path, map_location="cpu")
        if int(checkpoint.get("epoch", -1)) != int(epoch):
            raise RuntimeError(f"Checkpoint epoch mismatch for {path}: {checkpoint.get('epoch')}")
        if checkpoint.get("backbone_key") != base_cfg.BACKBONE_KEY:
            raise RuntimeError(
                f"Checkpoint backbone mismatch for epoch {epoch}: "
                f"{checkpoint.get('backbone_key')} != {base_cfg.BACKBONE_KEY}"
            )
        if "student" not in checkpoint or "teacher" not in checkpoint:
            raise RuntimeError(f"Checkpoint must contain student and teacher states: {path}")
        saved_cfg = checkpoint.get("config", {})
        if isinstance(saved_cfg, dict):
            for key in ("HEAD_TYPE", "P_INIT_MODE", "DABE_PU_VERSION"):
                if str(saved_cfg.get(key, "")).lower() != str(getattr(base_cfg, key, "")).lower():
                    raise RuntimeError(
                        f"Checkpoint config mismatch at epoch {epoch}: {key}={saved_cfg.get(key)!r}"
                    )
        student_state = checkpoint["student"]
        teacher_state = checkpoint["teacher"]
        if "base_head.weight" not in student_state:
            raise RuntimeError(f"Checkpoint is not a DAGP-Safe state: {path}")
        in_channels = int(student_state["base_head.weight"].shape[1])
        student = build_seg_head(in_channels, audit_cfg).to(device)
        teacher = build_seg_head(in_channels, audit_cfg).to(device)
        student.load_state_dict(student_state, strict=True)
        teacher.load_state_dict(teacher_state, strict=True)
        set_model_epoch(student, epoch)
        set_model_epoch(teacher, epoch)
        student.eval()
        teacher.eval()
        for model in (student, teacher):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        models[epoch] = {"student": student, "teacher": teacher}
        metadata[epoch] = {
            "checkpoint": str(path.resolve()),
            "teacher_source": "checkpoint_teacher",
            "student_teacher_max_abs_diff": max_state_diff(student_state, teacher_state),
            "checkpoint_saved_before_after_epoch_reset": bool(epoch == 20),
        }
    return models, metadata


def allocate_sample_counts(max_images):
    camo = max(1, int(round(int(max_images) * 0.30)))
    cod = int(max_images) - camo
    if cod < 1:
        cod = 1
        camo = int(max_images) - 1
    camo = min(camo, 300)
    cod = min(cod, 700)
    remaining = int(max_images) - camo - cod
    if remaining > 0:
        add_camo = min(300 - camo, remaining)
        camo += add_camo
        remaining -= add_camo
        cod += min(700 - cod, remaining)
    if camo + cod != int(max_images):
        raise RuntimeError(f"Unable to allocate stratified sample count for max_images={max_images}.")
    return {"TR-CAMO": camo, "TR-COD10K": cod}


def build_sample_loader(base_cfg, max_images, seed, batch_size, num_workers):
    dataset = CachedTrainDataset(base_cfg, max_samples=-1)
    if any("gt" in item or "gt_path" in item for item in dataset.items):
        raise RuntimeError("CachedTrainDataset item metadata must not contain GT.")
    requested = allocate_sample_counts(max_images)
    rng = random.Random(int(seed))
    selected = []
    selected_stems = {}
    for dataset_name in DATASETS:
        indices = [
            index
            for index, item in enumerate(dataset.items)
            if str(item.get("dataset")) == dataset_name
        ]
        if len(indices) < requested[dataset_name]:
            raise RuntimeError(
                f"Not enough samples for {dataset_name}: {len(indices)} < {requested[dataset_name]}"
            )
        chosen = rng.sample(indices, requested[dataset_name])
        chosen.sort(key=lambda index: str(dataset.items[index]["stem"]))
        selected.extend(chosen)
        selected_stems[dataset_name] = [str(dataset.items[index]["stem"]) for index in chosen]
    selected.sort(key=lambda index: (str(dataset.items[index]["dataset"]), str(dataset.items[index]["stem"])))
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        **dataloader_worker_kwargs(base_cfg, num_workers),
    )
    return dataset, loader, requested, selected_stems


def dataloader_worker_kwargs(cfg, num_workers):
    kwargs = {}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "DATALOADER_PERSISTENT_WORKERS", False))
        prefetch = int(getattr(cfg, "DATALOADER_PREFETCH_FACTOR", 2))
        if prefetch > 0:
            kwargs["prefetch_factor"] = prefetch
    return kwargs


def extract_logits(output):
    if isinstance(output, dict):
        return output["logits"]
    return output


def forward_all_epochs(models, audit_cfg, batch, source_epochs, device):
    model_input = make_model_input(audit_cfg, batch, device)
    image_68 = make_image_68(audit_cfg, batch, device)
    outputs = {}
    for epoch in source_epochs:
        student_output = models[epoch]["student"](
            model_input,
            image_68=image_68,
            return_aux=True,
        )
        teacher_output = models[epoch]["teacher"](
            model_input,
            image_68=image_68,
            return_aux=False,
        )
        student_logits = resize_logits_for_loss(extract_logits(student_output), audit_cfg).detach()
        teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), audit_cfg).detach()
        outputs[epoch] = {
            "student_output": student_output,
            "student_logits": student_logits,
            "student_prob": student_logits.sigmoid(),
            "teacher_prob": teacher_logits.sigmoid(),
        }
    return outputs


def build_source_candidates(audit_cfg, source_epoch, batch, epoch_outputs, device):
    output = epoch_outputs[source_epoch]
    teacher_prob = output["teacher_prob"].detach()
    teacher_binary = (teacher_prob >= 0.5).float().detach()
    _, rast_stats = build_rast_teacher_weight_map(
        audit_cfg,
        batch,
        teacher_binary,
        source_epoch,
        device,
    )
    teacher_map_ones = torch.ones_like(teacher_binary)
    _, stats, aux = build_esa_ber_loss(
        audit_cfg,
        source_epoch,
        output["student_output"],
        output["student_logits"],
        batch,
        teacher_prob,
        teacher_binary,
        teacher_map_ones,
        rast_stats,
        device,
    )
    required = {
        "margin_68",
        "conn_delta_68",
        "pos_score",
        "neg_score",
        "student_prob",
        "teacher_prob",
        "neg_extent_raw",
        "neg_hard_bg_raw",
        "selected_pos",
        "selected_neg",
        "selected_counts",
    }
    missing = sorted(required - set(aux))
    if missing:
        raise RuntimeError(f"BER helper did not expose required detached audit fields: {missing}")
    for name in required:
        tensor = aux[name]
        if torch.is_tensor(tensor) and tensor.requires_grad:
            raise RuntimeError(f"BER audit field unexpectedly requires grad: {name}")
    return stats, aux


def masked_values(tensor, image_index, mask):
    return tensor[image_index][mask].detach().float()


def safe_mean(tensor):
    return float(tensor.mean().item()) if int(tensor.numel()) else 0.0


def classify_negative_subtype(aux, image_index, flat_index):
    extent = bool(aux["neg_extent_raw"][image_index].reshape(-1)[flat_index].item())
    hard = bool(aux["neg_hard_bg_raw"][image_index].reshape(-1)[flat_index].item())
    if extent and hard:
        return "both"
    if extent:
        return "extent"
    if hard:
        return "hard_bg"
    raise RuntimeError("Selected negative candidate has no raw negative subtype.")


def candidate_track_rows(
    source_epoch,
    future_epoch,
    dataset_name,
    stem,
    image_index,
    candidate_class,
    mask,
    aux,
    source_logits,
    future_logits,
):
    rows = []
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().tolist()
    height, width = mask.shape[-2:]
    score_map = aux["pos_score"] if candidate_class == "positive" else aux["neg_score"]
    for flat_index in flat_indices:
        row_index, col_index = divmod(int(flat_index), int(width))
        source_logit = float(source_logits[image_index].reshape(-1)[flat_index].item())
        source_prob = float(aux["student_prob"][image_index].reshape(-1)[flat_index].item())
        source_binary = int(source_prob >= 0.5)
        if future_logits is None:
            future_logit = ""
            future_prob = ""
            future_binary = ""
            prob_delta = ""
            binary_persistent = ""
        else:
            future_logit = float(future_logits[image_index].reshape(-1)[flat_index].item())
            future_prob = float(torch.sigmoid(future_logits[image_index].reshape(-1)[flat_index]).item())
            future_binary = int(future_prob >= 0.5)
            prob_delta = future_prob - source_prob
            binary_persistent = int(source_binary == future_binary)
        rows.append(
            {
                "source_epoch": int(source_epoch),
                "future_epoch": "" if future_epoch is None else int(future_epoch),
                "dataset": dataset_name,
                "stem": stem,
                "candidate_class": candidate_class,
                "negative_subtype": (
                    "" if candidate_class == "positive" else classify_negative_subtype(aux, image_index, flat_index)
                ),
                "row": row_index,
                "col": col_index,
                "flat_index": int(flat_index),
                "source_score": float(score_map[image_index].reshape(-1)[flat_index].item()),
                "source_margin": float(aux["margin_68"][image_index].reshape(-1)[flat_index].item()),
                "source_conn_delta": float(aux["conn_delta_68"][image_index].reshape(-1)[flat_index].item()),
                "source_student_logit": source_logit,
                "source_student_prob": source_prob,
                "source_teacher_prob": float(aux["teacher_prob"][image_index].reshape(-1)[flat_index].item()),
                "source_binary": source_binary,
                "future_student_logit": future_logit,
                "future_student_prob": future_prob,
                "future_binary": future_binary,
                "prob_delta": prob_delta,
                "binary_persistent": binary_persistent,
            }
        )
    return rows


def build_source_image_row(source_epoch, dataset_name, stem, image_index, aux, source_logits):
    pos_mask = aux["selected_pos"][image_index]
    neg_mask = aux["selected_neg"][image_index]
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    valid = pos_count >= 4 and pos_count == neg_count
    pos_prob = masked_values(aux["student_prob"], image_index, pos_mask)
    neg_prob = masked_values(aux["student_prob"], image_index, neg_mask)
    pos_logits = masked_values(source_logits, image_index, pos_mask)
    neg_logits = masked_values(source_logits, image_index, neg_mask)
    pos_teacher = masked_values(aux["teacher_prob"], image_index, pos_mask)
    neg_teacher = masked_values(aux["teacher_prob"], image_index, neg_mask)
    pos_score = masked_values(aux["pos_score"], image_index, pos_mask)
    neg_score = masked_values(aux["neg_score"], image_index, neg_mask)
    pos_margin = masked_values(aux["margin_68"], image_index, pos_mask)
    neg_margin = masked_values(aux["margin_68"], image_index, neg_mask)
    pos_conn = masked_values(aux["conn_delta_68"], image_index, pos_mask)
    neg_conn = masked_values(aux["conn_delta_68"], image_index, neg_mask)
    neg_extent = int((neg_mask & aux["neg_extent_raw"][image_index]).sum().item())
    neg_hard = int((neg_mask & aux["neg_hard_bg_raw"][image_index]).sum().item())
    neg_both = int(
        (neg_mask & aux["neg_extent_raw"][image_index] & aux["neg_hard_bg_raw"][image_index]).sum().item()
    )
    return {
        "source_epoch": int(source_epoch),
        "dataset": dataset_name,
        "stem": stem,
        "valid": int(valid),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neg_extent_count": neg_extent - neg_both,
        "neg_hard_bg_count": neg_hard - neg_both,
        "neg_both_count": neg_both,
        "source_pos_prob": safe_mean(pos_prob),
        "source_neg_prob": safe_mean(neg_prob),
        "source_pos_teacher_prob": safe_mean(pos_teacher),
        "source_neg_teacher_prob": safe_mean(neg_teacher),
        "source_pos_logit": safe_mean(pos_logits),
        "source_neg_logit": safe_mean(neg_logits),
        "pos_score_mean": safe_mean(pos_score),
        "neg_score_mean": safe_mean(neg_score),
        "pos_margin_mean": safe_mean(pos_margin),
        "neg_margin_mean": safe_mean(neg_margin),
        "pos_conn_mean": safe_mean(pos_conn),
        "neg_conn_mean": safe_mean(neg_conn),
    }


def build_temporal_image_row(
    source_epoch,
    future_epoch,
    dataset_name,
    stem,
    image_index,
    aux,
    source_logits,
    future_logits,
):
    pos_mask = aux["selected_pos"][image_index]
    neg_mask = aux["selected_neg"][image_index]
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    valid = pos_count >= 4 and pos_count == neg_count
    if not valid:
        return {
            "source_epoch": int(source_epoch),
            "future_epoch": int(future_epoch),
            "dataset": dataset_name,
            "stem": stem,
            "valid": 0,
            "pos_count": 0,
            "neg_count": 0,
            **{field: 0.0 for field in PER_IMAGE_FIELDS[7:]},
        }

    source_pos_logits = source_logits[image_index][pos_mask].detach().float()
    source_neg_logits = source_logits[image_index][neg_mask].detach().float()
    future_pos_logits = future_logits[image_index][pos_mask].detach().float()
    future_neg_logits = future_logits[image_index][neg_mask].detach().float()
    source_pos_prob = source_pos_logits.sigmoid()
    source_neg_prob = source_neg_logits.sigmoid()
    future_pos_prob = future_pos_logits.sigmoid()
    future_neg_prob = future_neg_logits.sigmoid()
    source_pair_diff = source_pos_logits[:, None] - source_neg_logits[None, :]
    future_pair_diff = future_pos_logits[:, None] - future_neg_logits[None, :]
    pair_count = int(source_pair_diff.numel())
    correct_count = int((future_pair_diff > 0.0).sum().item())
    flip_count = int(((source_pair_diff * future_pair_diff) < 0.0).sum().item())
    pos_fg_count = int((future_pos_prob >= 0.5).sum().item())
    neg_fg_count = int((future_neg_prob >= 0.5).sum().item())
    pos_increase_count = int((future_pos_prob > source_pos_prob).sum().item())
    neg_decrease_count = int((future_neg_prob < source_neg_prob).sum().item())
    source_binary = torch.cat((source_pos_prob >= 0.5, source_neg_prob >= 0.5))
    future_binary = torch.cat((future_pos_prob >= 0.5, future_neg_prob >= 0.5))
    persistence_count = int((source_binary == future_binary).sum().item())
    source_pos_prob_mean = safe_mean(source_pos_prob)
    source_neg_prob_mean = safe_mean(source_neg_prob)
    future_pos_prob_mean = safe_mean(future_pos_prob)
    future_neg_prob_mean = safe_mean(future_neg_prob)
    source_logit_gap = safe_mean(source_pos_logits) - safe_mean(source_neg_logits)
    future_logit_gap = safe_mean(future_pos_logits) - safe_mean(future_neg_logits)
    return {
        "source_epoch": int(source_epoch),
        "future_epoch": int(future_epoch),
        "dataset": dataset_name,
        "stem": stem,
        "valid": 1,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "source_pos_prob": source_pos_prob_mean,
        "source_neg_prob": source_neg_prob_mean,
        "source_prob_gap": source_pos_prob_mean - source_neg_prob_mean,
        "source_logit_gap": source_logit_gap,
        "future_pos_prob": future_pos_prob_mean,
        "future_neg_prob": future_neg_prob_mean,
        "future_prob_gap": future_pos_prob_mean - future_neg_prob_mean,
        "future_logit_gap": future_logit_gap,
        "delta_pos_prob": safe_mean(future_pos_prob - source_pos_prob),
        "delta_neg_prob": safe_mean(future_neg_prob - source_neg_prob),
        "delta_logit_gap": future_logit_gap - source_logit_gap,
        "pos_future_fg_ratio": pos_fg_count / max(pos_count, 1),
        "neg_future_fg_ratio": neg_fg_count / max(neg_count, 1),
        "fg_ratio_gap": pos_fg_count / max(pos_count, 1) - neg_fg_count / max(neg_count, 1),
        "pos_prob_increase_ratio": pos_increase_count / max(pos_count, 1),
        "neg_prob_decrease_ratio": neg_decrease_count / max(neg_count, 1),
        "pair_order_correct_ratio": correct_count / max(pair_count, 1),
        "pair_order_flip_ratio": flip_count / max(pair_count, 1),
        "candidate_spatial_persistence": persistence_count / max(pos_count + neg_count, 1),
        "pair_count": pair_count,
        "pair_order_correct_count": correct_count,
        "pair_order_flip_count": flip_count,
        "pos_future_fg_count": pos_fg_count,
        "neg_future_fg_count": neg_fg_count,
        "pos_prob_increase_count": pos_increase_count,
        "neg_prob_decrease_count": neg_decrease_count,
        "persistence_count": persistence_count,
    }


def weighted_mean(rows, value_key, weight_key):
    weight = sum(float(row[weight_key]) for row in rows)
    if weight <= 0.0:
        return 0.0
    return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / weight


def aggregate_temporal_rows(rows, source_epoch=None, future_epoch=None, dataset_name=None):
    selected = [
        row
        for row in rows
        if (source_epoch is None or int(row["source_epoch"]) == int(source_epoch))
        and (future_epoch is None or int(row["future_epoch"]) == int(future_epoch))
        and (dataset_name in (None, "ALL") or row["dataset"] == dataset_name)
    ]
    if not selected:
        raise RuntimeError(
            f"No temporal rows for source={source_epoch}, future={future_epoch}, dataset={dataset_name}"
        )
    valid = [row for row in selected if int(row["valid"]) == 1]
    pos_count = sum(int(row["pos_count"]) for row in valid)
    neg_count = sum(int(row["neg_count"]) for row in valid)
    pair_count = sum(int(row["pair_count"]) for row in valid)
    source_pos_prob = weighted_mean(valid, "source_pos_prob", "pos_count")
    source_neg_prob = weighted_mean(valid, "source_neg_prob", "neg_count")
    future_pos_prob = weighted_mean(valid, "future_pos_prob", "pos_count")
    future_neg_prob = weighted_mean(valid, "future_neg_prob", "neg_count")
    source_logit_gap = weighted_mean(valid, "source_logit_gap", "pos_count")
    future_logit_gap = weighted_mean(valid, "future_logit_gap", "pos_count")
    pos_fg_count = sum(int(row["pos_future_fg_count"]) for row in valid)
    neg_fg_count = sum(int(row["neg_future_fg_count"]) for row in valid)
    persistence_count = sum(int(row["persistence_count"]) for row in valid)
    return {
        "source_epoch": "ALL" if source_epoch is None else int(source_epoch),
        "future_epoch": "ALL" if future_epoch is None else int(future_epoch),
        "source_dataset": dataset_name or "ALL",
        "num_images": len(selected),
        "valid_images": len(valid),
        "valid_image_ratio": len(valid) / max(len(selected), 1),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "source_pos_prob": source_pos_prob,
        "source_neg_prob": source_neg_prob,
        "source_prob_gap": source_pos_prob - source_neg_prob,
        "source_logit_gap": source_logit_gap,
        "future_pos_prob": future_pos_prob,
        "future_neg_prob": future_neg_prob,
        "future_prob_gap": future_pos_prob - future_neg_prob,
        "future_logit_gap": future_logit_gap,
        "delta_pos_prob": weighted_mean(valid, "delta_pos_prob", "pos_count"),
        "delta_neg_prob": weighted_mean(valid, "delta_neg_prob", "neg_count"),
        "delta_logit_gap": future_logit_gap - source_logit_gap,
        "pos_future_fg_ratio": pos_fg_count / max(pos_count, 1),
        "neg_future_fg_ratio": neg_fg_count / max(neg_count, 1),
        "fg_ratio_gap": pos_fg_count / max(pos_count, 1) - neg_fg_count / max(neg_count, 1),
        "pos_prob_increase_ratio": (
            sum(int(row["pos_prob_increase_count"]) for row in valid) / max(pos_count, 1)
        ),
        "neg_prob_decrease_ratio": (
            sum(int(row["neg_prob_decrease_count"]) for row in valid) / max(neg_count, 1)
        ),
        "pair_order_correct_ratio": (
            sum(int(row["pair_order_correct_count"]) for row in valid) / max(pair_count, 1)
        ),
        "pair_order_flip_ratio": (
            sum(int(row["pair_order_flip_count"]) for row in valid) / max(pair_count, 1)
        ),
        "candidate_spatial_persistence": persistence_count / max(pos_count + neg_count, 1),
        "pair_count": pair_count,
    }


def aggregate_source_rows(rows, source_epoch, dataset_name):
    selected = [
        row
        for row in rows
        if int(row["source_epoch"]) == int(source_epoch)
        and (dataset_name == "ALL" or row["dataset"] == dataset_name)
    ]
    if not selected:
        raise RuntimeError(f"No source rows for epoch={source_epoch}, dataset={dataset_name}")
    valid = [row for row in selected if int(row["valid"]) == 1]
    pos_count = sum(int(row["pos_count"]) for row in valid)
    neg_count = sum(int(row["neg_count"]) for row in valid)
    source_pos_prob = weighted_mean(valid, "source_pos_prob", "pos_count")
    source_neg_prob = weighted_mean(valid, "source_neg_prob", "neg_count")
    source_pos_logit = weighted_mean(valid, "source_pos_logit", "pos_count")
    source_neg_logit = weighted_mean(valid, "source_neg_logit", "neg_count")
    return {
        "source_epoch": int(source_epoch),
        "source_dataset": dataset_name,
        "num_images": len(selected),
        "valid_images": len(valid),
        "valid_image_ratio": len(valid) / max(len(selected), 1),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neg_extent_count": sum(int(row["neg_extent_count"]) for row in valid),
        "neg_hard_bg_count": sum(int(row["neg_hard_bg_count"]) for row in valid),
        "neg_both_count": sum(int(row["neg_both_count"]) for row in valid),
        "source_pos_prob": source_pos_prob,
        "source_neg_prob": source_neg_prob,
        "source_prob_gap": source_pos_prob - source_neg_prob,
        "source_pos_teacher_prob": weighted_mean(valid, "source_pos_teacher_prob", "pos_count"),
        "source_neg_teacher_prob": weighted_mean(valid, "source_neg_teacher_prob", "neg_count"),
        "source_logit_gap": source_pos_logit - source_neg_logit,
        "pos_score_mean": weighted_mean(valid, "pos_score_mean", "pos_count"),
        "neg_score_mean": weighted_mean(valid, "neg_score_mean", "neg_count"),
        "pos_margin_mean": weighted_mean(valid, "pos_margin_mean", "pos_count"),
        "neg_margin_mean": weighted_mean(valid, "neg_margin_mean", "neg_count"),
        "pos_conn_mean": weighted_mean(valid, "pos_conn_mean", "pos_count"),
        "neg_conn_mean": weighted_mean(valid, "neg_conn_mean", "neg_count"),
    }


def build_aggregates(source_rows, temporal_rows, source_epochs, future_map):
    source_summary = []
    for epoch in source_epochs:
        for dataset_name in (*DATASETS, "ALL"):
            source_summary.append(aggregate_source_rows(source_rows, epoch, dataset_name))
    temporal_summary = []
    for source_epoch in source_epochs:
        for future_epoch in future_map[source_epoch]:
            for dataset_name in (*DATASETS, "ALL"):
                temporal_summary.append(
                    aggregate_temporal_rows(
                        temporal_rows,
                        source_epoch=source_epoch,
                        future_epoch=future_epoch,
                        dataset_name=dataset_name,
                    )
                )
    dataset_summary = [
        aggregate_temporal_rows(temporal_rows, dataset_name=dataset_name)
        for dataset_name in (*DATASETS, "ALL")
    ]
    return source_summary, temporal_summary, dataset_summary


def classify_verdict(temporal_summary):
    lookup = {
        (int(row["source_epoch"]), int(row["future_epoch"]), row["source_dataset"]): row
        for row in temporal_summary
        if row["source_epoch"] != "ALL" and row["future_epoch"] != "ALL"
    }
    checks = {}
    dataset_pass = {}
    for dataset_name in DATASETS:
        relation_checks = []
        for source_epoch in (7, 10, 15):
            row = lookup[(source_epoch, 20, dataset_name)]
            passed = (
                float(row["future_prob_gap"]) >= 0.10
                and float(row["pair_order_correct_ratio"]) >= 0.65
                and float(row["fg_ratio_gap"]) >= 0.20
            )
            relation_checks.append(passed)
            checks[f"{dataset_name}_{source_epoch}_to_20"] = {
                "passed": passed,
                "future_prob_gap": row["future_prob_gap"],
                "pair_order_correct_ratio": row["pair_order_correct_ratio"],
                "fg_ratio_gap": row["fg_ratio_gap"],
            }
        dataset_pass[dataset_name] = all(relation_checks)
    if all(dataset_pass.values()):
        return {
            "class": "A1",
            "message": (
                "Pre-reset BER candidates show useful temporal discrimination. "
                "A Pre-BER replacement experiment is justified."
            ),
            "dataset_pass": dataset_pass,
            "checks": checks,
        }
    if dataset_pass["TR-CAMO"] and not dataset_pass["TR-COD10K"]:
        return {
            "class": "A2",
            "message": (
                "Pre-reset BER candidates are source-biased and may reproduce the CAMO-COD10K trade-off. "
                "Do not run full Pre-BER."
            ),
            "dataset_pass": dataset_pass,
            "checks": checks,
        }
    overall = aggregate_temporal_rows_from_summary(temporal_summary, "ALL")
    a3_checks = {
        "pair_order_correct_le_0_55": float(overall["pair_order_correct_ratio"]) <= 0.55,
        "pair_order_flip_ge_0_30": float(overall["pair_order_flip_ratio"]) >= 0.30,
        "delta_logit_gap_le_0": float(overall["delta_logit_gap"]) <= 0.0,
        "future_prob_gap_lt_0_05": float(overall["future_prob_gap"]) < 0.05,
    }
    if sum(int(value) for value in a3_checks.values()) >= 3:
        return {
            "class": "A3",
            "message": "BER candidate semantics are not temporally reliable. Stop the BER route.",
            "dataset_pass": dataset_pass,
            "checks": checks,
            "a3_checks": a3_checks,
        }
    return {
        "class": "INCONCLUSIVE",
        "message": "Temporal evidence is mixed; do not start a full Pre-BER experiment yet.",
        "dataset_pass": dataset_pass,
        "checks": checks,
        "a3_checks": a3_checks,
    }


def aggregate_temporal_rows_from_summary(rows, dataset_name):
    selected = [row for row in rows if row["source_dataset"] == dataset_name]
    if not selected:
        raise RuntimeError(f"No temporal summary rows for dataset {dataset_name}")
    pos_count = sum(int(row["pos_count"]) for row in selected)
    neg_count = sum(int(row["neg_count"]) for row in selected)
    pair_count = sum(int(row["pair_count"]) for row in selected)
    source_pos = sum(float(row["source_pos_prob"]) * int(row["pos_count"]) for row in selected) / max(pos_count, 1)
    source_neg = sum(float(row["source_neg_prob"]) * int(row["neg_count"]) for row in selected) / max(neg_count, 1)
    future_pos = sum(float(row["future_pos_prob"]) * int(row["pos_count"]) for row in selected) / max(pos_count, 1)
    future_neg = sum(float(row["future_neg_prob"]) * int(row["neg_count"]) for row in selected) / max(neg_count, 1)
    source_gap = sum(float(row["source_logit_gap"]) * int(row["pair_count"]) for row in selected) / max(pair_count, 1)
    future_gap = sum(float(row["future_logit_gap"]) * int(row["pair_count"]) for row in selected) / max(pair_count, 1)
    return {
        "future_prob_gap": future_pos - future_neg,
        "pair_order_correct_ratio": sum(
            float(row["pair_order_correct_ratio"]) * int(row["pair_count"]) for row in selected
        ) / max(pair_count, 1),
        "pair_order_flip_ratio": sum(
            float(row["pair_order_flip_ratio"]) * int(row["pair_count"]) for row in selected
        ) / max(pair_count, 1),
        "delta_logit_gap": future_gap - source_gap,
        "source_prob_gap": source_pos - source_neg,
    }


def write_csv(path, rows, fields):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    out_dir,
    metadata,
    source_summary,
    temporal_summary,
    dataset_summary,
    per_image_rows,
    candidate_rows,
    verdict,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    write_csv(out_dir / "temporal_pairs.csv", temporal_summary, TEMPORAL_FIELDS)
    write_csv(out_dir / "per_source_epoch.csv", source_summary, SOURCE_FIELDS)
    write_csv(out_dir / "per_dataset_stats.csv", dataset_summary, TEMPORAL_FIELDS)
    write_csv(out_dir / "per_image_stats.csv", per_image_rows, PER_IMAGE_FIELDS)
    write_csv(out_dir / "candidate_tracks.csv", candidate_rows, CANDIDATE_FIELDS)
    summary = {
        "metadata": metadata,
        "verdict": verdict,
        "source_epoch_stats": source_summary,
        "temporal_stats": temporal_summary,
        "dataset_stats": dataset_summary,
        "semantic_correctness_warning": (
            "A positive result only supports candidate stability; it does not prove semantic correctness."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Pre-reset BER Candidate Temporal Audit",
        "",
        "## Audit State",
        "",
        f"- Config: `{metadata['config']}`",
        f"- BER config: `{metadata['ber_config']}`",
        f"- Workdir: `{metadata['workdir']}`",
        f"- Source epochs: `{metadata['source_epochs']}`",
        f"- Future map: `{metadata['future_map']}`",
        f"- Images: `{metadata['num_images']}` "
        f"(TR-CAMO={metadata['sample_counts']['TR-CAMO']}, "
        f"TR-COD10K={metadata['sample_counts']['TR-COD10K']})",
        "- Teacher source: `checkpoint_teacher` for every source epoch",
        "- Training/test GT used: `False`",
        "- Optimizer/backward/EMA update: `False`",
        f"- Model state mutation max diff: `{metadata['model_state_max_abs_diff']}`",
        "",
        "## Source Candidate Validity",
        "",
        "| Epoch | Dataset | Valid ratio | Pairs | Source prob gap | Source logit gap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in source_summary:
        if row["source_dataset"] == "ALL":
            continue
        lines.append(
            f"| {row['source_epoch']} | {row['source_dataset']} | "
            f"{float(row['valid_image_ratio']):.6f} | {row['pos_count']} | "
            f"{float(row['source_prob_gap']):.6f} | {float(row['source_logit_gap']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Temporal Results",
            "",
            "| Relation | Dataset | Future P+ | Future P- | Gap | Delta gap | Correct | Flip | Persistence |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in temporal_summary:
        if row["source_dataset"] == "ALL":
            continue
        lines.append(
            f"| {row['source_epoch']}->{row['future_epoch']} | {row['source_dataset']} | "
            f"{float(row['future_pos_prob']):.6f} | {float(row['future_neg_prob']):.6f} | "
            f"{float(row['future_prob_gap']):.6f} | {float(row['delta_logit_gap']):.6f} | "
            f"{float(row['pair_order_correct_ratio']):.6f} | "
            f"{float(row['pair_order_flip_ratio']):.6f} | "
            f"{float(row['candidate_spatial_persistence']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Automatic Verdict",
            "",
            f"- Verdict: **{verdict['class']}**",
            f"- {verdict['message']}",
            "",
            "## Interpretation Limit",
            "",
            "A positive result only supports candidate stability; it does not prove semantic correctness.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zip_path = out_dir / "pre_ber_temporal_audit.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in REPORT_FILES:
            archive.write(out_dir / filename, arcname=filename)
    return summary, zip_path


def run_audit(args):
    validate_args(args)
    source_epochs = parse_source_epochs(args.source_epochs)
    future_map = parse_future_map(args.future_map, source_epochs)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    base_cfg = load_config(args.config)
    ber_cfg = load_config(args.ber_config)
    validate_configs(base_cfg, ber_cfg)
    audit_cfg = build_audit_config(ber_cfg)
    batch_size = int(args.batch_size or base_cfg.BATCH_SIZE)
    num_workers = int(base_cfg.NUM_WORKERS) if int(args.num_workers) == -1 else int(args.num_workers)
    models, checkpoint_metadata = load_models(
        base_cfg,
        audit_cfg,
        args.workdir,
        source_epochs,
        device,
    )
    before_states = {
        (epoch, role): clone_state(models[epoch][role])
        for epoch in source_epochs
        for role in ("student", "teacher")
    }
    dataset, loader, sample_counts, selected_stems = build_sample_loader(
        base_cfg,
        args.max_images,
        args.seed,
        batch_size,
        num_workers,
    )
    print(f"device = {device}", flush=True)
    print(f"workdir = {Path(args.workdir).resolve()}", flush=True)
    print(f"source_epochs = {source_epochs}", flush=True)
    print(f"future_map = {future_map}", flush=True)
    print(f"sample_counts = {sample_counts}", flush=True)
    print(f"batch_size = {batch_size} | num_batches = {len(loader)}", flush=True)
    print("teacher_source = checkpoint_teacher", flush=True)

    source_image_rows = []
    per_image_rows = []
    candidate_rows = []
    topk_error_max = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if "gt" in batch or "gt_path" in batch:
                raise RuntimeError(f"Temporal audit must not read GT; keys={sorted(batch)}")
            epoch_outputs = forward_all_epochs(models, audit_cfg, batch, source_epochs, device)
            dataset_names = [str(value) for value in batch["dataset"]]
            stems = [str(value) for value in batch["stem"]]
            for source_epoch in source_epochs:
                stats, aux = build_source_candidates(
                    audit_cfg,
                    source_epoch,
                    batch,
                    epoch_outputs,
                    device,
                )
                topk_error_max = max(topk_error_max, float(stats["topk_sem_weight_sum_error"]))
                source_logits = epoch_outputs[source_epoch]["student_logits"]
                for image_index, (dataset_name, stem) in enumerate(zip(dataset_names, stems)):
                    source_row = build_source_image_row(
                        source_epoch,
                        dataset_name,
                        stem,
                        image_index,
                        aux,
                        source_logits,
                    )
                    source_image_rows.append(source_row)
                    if not future_map[source_epoch]:
                        for candidate_class, mask in (
                            ("positive", aux["selected_pos"][image_index]),
                            ("negative", aux["selected_neg"][image_index]),
                        ):
                            candidate_rows.extend(
                                candidate_track_rows(
                                    source_epoch,
                                    None,
                                    dataset_name,
                                    stem,
                                    image_index,
                                    candidate_class,
                                    mask,
                                    aux,
                                    source_logits,
                                    None,
                                )
                            )
                    for future_epoch in future_map[source_epoch]:
                        future_logits = epoch_outputs[future_epoch]["student_logits"]
                        per_image_rows.append(
                            build_temporal_image_row(
                                source_epoch,
                                future_epoch,
                                dataset_name,
                                stem,
                                image_index,
                                aux,
                                source_logits,
                                future_logits,
                            )
                        )
                        for candidate_class, mask in (
                            ("positive", aux["selected_pos"][image_index]),
                            ("negative", aux["selected_neg"][image_index]),
                        ):
                            candidate_rows.extend(
                                candidate_track_rows(
                                    source_epoch,
                                    future_epoch,
                                    dataset_name,
                                    stem,
                                    image_index,
                                    candidate_class,
                                    mask,
                                    aux,
                                    source_logits,
                                    future_logits,
                                )
                            )
            if (batch_index + 1) % int(args.progress_every) == 0 or batch_index + 1 == len(loader):
                print(f"processed_batches = {batch_index + 1}/{len(loader)}", flush=True)

    model_state_max_diff = 0.0
    for epoch in source_epochs:
        for role in ("student", "teacher"):
            after = models[epoch][role].state_dict()
            diff = max_state_diff(before_states[(epoch, role)], after)
            model_state_max_diff = max(model_state_max_diff, diff)
            if any(parameter.grad is not None for parameter in models[epoch][role].parameters()):
                raise RuntimeError(f"Unexpected gradient on {role} epoch {epoch}.")
    if model_state_max_diff != 0.0:
        raise RuntimeError(f"Audit mutated model state: max_abs_diff={model_state_max_diff}")

    source_summary, temporal_summary, dataset_summary = build_aggregates(
        source_image_rows,
        per_image_rows,
        source_epochs,
        future_map,
    )
    verdict = classify_verdict(temporal_summary)
    candidate_config = {
        key: getattr(ber_cfg, key)
        for key in dir(ber_cfg)
        if key.startswith("ESA_BER_") and isinstance(getattr(ber_cfg, key), (str, int, float, bool))
    }
    metadata = {
        "config": str(Path(args.config).resolve()),
        "ber_config": str(Path(args.ber_config).resolve()),
        "workdir": str(Path(args.workdir).resolve()),
        "source_epochs": list(source_epochs),
        "future_map": {str(key): list(value) for key, value in future_map.items()},
        "num_images": int(args.max_images),
        "sample_counts": sample_counts,
        "selected_stems": selected_stems,
        "seed": int(args.seed),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "checkpoint_metadata": checkpoint_metadata,
        "teacher_source": "checkpoint_teacher",
        "epoch20_teacher_note": "checkpoint saved before after_epoch reset; direct teacher is epoch20 training state",
        "training_gt_used": False,
        "test_gt_used": False,
        "optimizer_created": False,
        "backward_called": False,
        "ema_updated": False,
        "future_candidates_reselected": False,
        "candidate_spatial_persistence_definition": "source/future binary agreement at fixed selected coordinates",
        "pair_order_definition": "Cartesian selected-positive x selected-negative logit comparisons",
        "model_state_max_abs_diff": model_state_max_diff,
        "topk_sem_weight_sum_error_max": topk_error_max,
        "candidate_track_rows": len(candidate_rows),
        "candidate_config": candidate_config,
    }
    out_dir = Path(args.out).expanduser()
    summary, zip_path = write_reports(
        out_dir,
        metadata,
        source_summary,
        temporal_summary,
        dataset_summary,
        per_image_rows,
        candidate_rows,
        verdict,
    )
    print(f"verdict = {verdict['class']}", flush=True)
    print(f"candidate_track_rows = {len(candidate_rows)}", flush=True)
    print(f"report_zip = {zip_path.resolve()}", flush=True)
    return summary


def main():
    args = parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
