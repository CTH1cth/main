import argparse
import csv
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import ensure_dir, load_config, set_seed, torch_load  # noqa: E402
from eval import infer_in_channels  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_rast_teacher_weight_map,
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    rast_teacher_bce_with_logits,
    resize_logits_for_loss,
    set_model_epoch,
)


PROBE_EPOCH = 21
RESET_EPOCH = 20
SOURCES = ("TR-CAMO", "TR-COD10K")
SUPPORTED_MODES = ("teacher_only", "postreset_esa")
DEFAULT_POSTRESET_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_rast_v12_esa_asym_postreset35_lrfloor_2e5.py"
)
MODULE_ORDER = (
    "base_head",
    "dagp_projection",
    "dagp_graph_head",
    "ndr_branch",
    "all_decoder",
)
EPS = 1e-12

PER_PAIR_FIELDS = [
    "pair_index",
    "mode",
    "module",
    "batch_size",
    "camo_stems",
    "cod_stems",
    "loss_camo",
    "loss_cod",
    "loss_camo_final",
    "loss_camo_coarse",
    "loss_camo_base",
    "loss_cod_final",
    "loss_cod_coarse",
    "loss_cod_base",
    "dot_product",
    "cosine",
    "norm_camo",
    "norm_cod",
    "norm_ratio_cod_over_camo",
    "cod_norm_gt_camo",
    "conflict",
    "conflict_energy",
    "projection_loss_camo",
    "projection_loss_cod",
    "sign_disagreement_ratio",
    "effective_param_count",
    "parameter_count",
]

MODULE_SUMMARY_FIELDS = [
    "mode",
    "module",
    "num_pairs",
    "mean_cosine",
    "median_cosine",
    "negative_cosine_ratio",
    "p10_cosine",
    "p90_cosine",
    "mean_norm_camo",
    "mean_norm_cod",
    "mean_norm_ratio",
    "median_norm_ratio",
    "cod_norm_dominance_ratio",
    "mean_conflict_energy",
    "mean_projection_loss_camo",
    "mean_projection_loss_cod",
    "mean_sign_disagreement_ratio",
    "mean_loss_camo",
    "mean_loss_cod",
]

MODE_COMPARISON_FIELDS = [
    "module",
    "teacher_only_mean_cosine",
    "postreset_esa_mean_cosine",
    "delta_mean_cosine",
    "teacher_only_negative_cosine_ratio",
    "postreset_esa_negative_cosine_ratio",
    "delta_negative_cosine_ratio",
    "teacher_only_mean_conflict_energy",
    "postreset_esa_mean_conflict_energy",
    "delta_mean_conflict_energy",
    "teacher_only_mean_norm_ratio",
    "postreset_esa_mean_norm_ratio",
    "delta_mean_norm_ratio",
    "esa_effect",
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
        description=(
            "Measure TR-CAMO/TR-COD10K decoder gradient conflict at the exact "
            "Long35 epoch-21 teacher-only start state."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--postreset-config",
        default=str(DEFAULT_POSTRESET_CONFIG),
        help="Existing extent-only post-reset ESA config used only for routing fields.",
    )
    parser.add_argument(
        "--modes",
        default="teacher_only,postreset_esa",
        help="Comma-separated subset of teacher_only,postreset_esa.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:0.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 uses cfg.NUM_WORKERS.",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def parse_modes(value):
    modes = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if not modes:
        raise ValueError("--modes must not be empty.")
    if len(modes) != len(set(modes)):
        raise ValueError(f"Duplicate modes are not allowed: {modes}")
    invalid = sorted(set(modes) - set(SUPPORTED_MODES))
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}; supported={SUPPORTED_MODES}")
    return modes


def resolve_device(value):
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    return device


def validate_args(args, modes):
    for label, value in (
        ("config", args.config),
        ("checkpoint", args.ckpt),
    ):
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if "postreset_esa" in modes and not Path(args.postreset_config).expanduser().exists():
        raise FileNotFoundError(f"post-reset ESA config not found: {args.postreset_config}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_pairs) <= 0:
        raise ValueError("--num-pairs must be positive.")
    if int(args.num_workers) < -1:
        raise ValueError("--num-workers must be -1 or non-negative.")
    if int(args.progress_every) <= 0:
        raise ValueError("--progress-every must be positive.")


def config_value(cfg, name, default=None):
    return getattr(cfg, name, default)


def validate_base_config(cfg):
    expected = {
        "HEAD_TYPE": "dagp_safe",
        "P_INIT_MODE": "dabe_pu_v11_desplsched",
        "DABE_PU_VERSION": "pu_v11",
        "TEACHER_FUSION_MODE": "dabe_pu_despl_sched",
        "TEACHER_TARGET_MODE": "binary",
        "DABE_PU_STATIC_TARGET_MODE": "soft",
        "FINETUNE_RESET_EPOCH": RESET_EPOCH,
        "FINETUNE_RESET_TIMING": "after_epoch",
    }
    for key, expected_value in expected.items():
        actual = config_value(cfg, key)
        if str(actual).lower() != str(expected_value).lower():
            raise RuntimeError(f"Probe config mismatch: {key}={actual!r}, expected {expected_value!r}.")
    bool_expected = {
        "USE_DABE_PU": True,
        "USE_NDR_BRANCH": True,
        "USE_NDR_COARSE_AUX": True,
        "USE_BASE_AUX_LOSS": True,
        "USE_RAST": True,
        "USE_TEACHER_BINARY_FULL_LOSS": True,
        "USE_TEACHER_SOFT_FULL_LOSS": False,
        "FINETUNE_RESET_TEACHER": True,
    }
    for key, expected_value in bool_expected.items():
        actual = bool(config_value(cfg, key, False))
        if actual != expected_value:
            raise RuntimeError(f"Probe config mismatch: {key}={actual}, expected {expected_value}.")
    scalar_expected = {
        "LAMBDA_NDR_COARSE_AUX": 0.5,
        "LAMBDA_BASE_AUX_AFTER_RESET": 0.3,
        "LOSS_SIZE": 68.0,
    }
    for key, expected_value in scalar_expected.items():
        actual = float(config_value(cfg, key, float("nan")))
        if not math.isfinite(actual) or abs(actual - expected_value) > 1e-12:
            raise RuntimeError(f"Probe config mismatch: {key}={actual}, expected {expected_value}.")
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(PROBE_EPOCH, cfg)
    if abs(float(static_weight)) > 1e-12 or abs(float(teacher_weight) - 1.0) > 1e-12:
        raise RuntimeError(
            f"Epoch {PROBE_EPOCH} must be teacher-only, got static={static_weight}, teacher={teacher_weight}."
        )


def build_mode_configs(base_cfg, postreset_cfg, modes):
    mode_configs = {}
    if "teacher_only" in modes:
        mode_configs["teacher_only"] = ConfigOverlay(
            base_cfg,
            {
                "ESA_POST_RESET_ENABLE": False,
                "RAST_POST_RESET_ENABLE": False,
                "RAST_POST_RESET_SCALE": 0.0,
            },
        )
    if "postreset_esa" in modes:
        required = {
            "ESA_POST_RESET_ENABLE": True,
            "ESA_POST_RESET_START_EPOCH": 21,
            "ESA_POST_RESET_STOP_EPOCH": 36,
            "ESA_POST_RESET_SCALE": 1.0,
            "ESA_POST_RESET_RAMP": False,
            "ESA_POST_RESET_MODE": "extent_only",
            "RAST_POST_RESET_ENABLE": False,
        }
        for key, expected in required.items():
            actual = getattr(postreset_cfg, key, None)
            if str(actual).lower() != str(expected).lower():
                raise RuntimeError(
                    f"Post-reset ESA config mismatch: {key}={actual!r}, expected {expected!r}."
                )
        overrides = {
            key: getattr(postreset_cfg, key)
            for key in dir(postreset_cfg)
            if key.startswith("ESA_POST_RESET_")
        }
        overrides.update(
            {
                "ESA_POST_RESET_ENABLE": True,
                "RAST_POST_RESET_ENABLE": False,
                "RAST_POST_RESET_SCALE": 0.0,
            }
        )
        mode_configs["postreset_esa"] = ConfigOverlay(base_cfg, overrides)
    return mode_configs


def max_state_diff(state_a, state_b):
    names = sorted(set(state_a) & set(state_b))
    if set(state_a) != set(state_b):
        return float("inf")
    maximum = 0.0
    for name in names:
        a = state_a[name]
        b = state_b[name]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape:
                return float("inf")
            if torch.is_floating_point(a) or torch.is_complex(a):
                diff = float((a.detach().cpu() - b.detach().cpu()).abs().max().item())
            else:
                diff = 0.0 if torch.equal(a.detach().cpu(), b.detach().cpu()) else float("inf")
            maximum = max(maximum, diff)
        elif a != b:
            return float("inf")
    return maximum


def clone_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def validate_checkpoint_metadata(cfg, checkpoint, ckpt_path):
    if int(checkpoint.get("epoch", -1)) != RESET_EPOCH:
        raise RuntimeError(
            f"Probe requires epoch_{RESET_EPOCH:03d}.pth, got epoch={checkpoint.get('epoch')} from {ckpt_path}."
        )
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}."
        )
    if "student" not in checkpoint:
        raise KeyError(f"Checkpoint has no student state: {ckpt_path}")
    saved_cfg = checkpoint.get("config", {})
    if isinstance(saved_cfg, dict):
        for key in ("HEAD_TYPE", "P_INIT_MODE", "DABE_PU_VERSION", "TEACHER_FUSION_MODE"):
            expected = getattr(cfg, key, None)
            actual = saved_cfg.get(key, None)
            if str(actual).lower() != str(expected).lower():
                raise RuntimeError(
                    f"Checkpoint config mismatch: {key}={actual!r}, runtime config={expected!r}."
                )
        if str(saved_cfg.get("FINETUNE_RESET_TIMING", "")).lower() != "after_epoch":
            raise RuntimeError("Checkpoint was not produced by the required after_epoch reset protocol.")
        if not bool(saved_cfg.get("FINETUNE_RESET_TEACHER", False)):
            raise RuntimeError("Checkpoint protocol does not reset teacher from student after epoch 20.")


def load_epoch21_models(cfg, ckpt_path, device):
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    validate_checkpoint_metadata(cfg, checkpoint, ckpt_path)
    student_state = checkpoint["student"]
    in_channels = infer_in_channels(student_state)
    student = build_seg_head(in_channels, cfg).to(device)
    teacher = build_seg_head(in_channels, cfg).to(device)
    student.load_state_dict(student_state, strict=True)

    saved_teacher_diff = None
    if "teacher" in checkpoint:
        saved_teacher_diff = max_state_diff(student_state, checkpoint["teacher"])

    # epoch_020.pth is saved before the configured after-epoch reset. The real
    # epoch-21 teacher is therefore reset from the saved epoch-20 student.
    teacher.load_state_dict(student.state_dict(), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    set_model_epoch(student, PROBE_EPOCH)
    set_model_epoch(teacher, PROBE_EPOCH)
    student.train()
    teacher.eval()
    reconstructed_diff = max_state_diff(student.state_dict(), teacher.state_dict())
    if reconstructed_diff != 0.0:
        raise RuntimeError(
            f"Failed to reconstruct epoch-21 teacher from student: max_abs_diff={reconstructed_diff}."
        )
    return student, teacher, checkpoint, saved_teacher_diff, reconstructed_diff


def build_parameter_groups(student):
    named = list(student.named_parameters())
    if not named:
        raise RuntimeError("Student model has no trainable parameters.")
    if any(not parameter.requires_grad for _, parameter in named):
        frozen = [name for name, parameter in named if not parameter.requires_grad]
        raise RuntimeError(f"Decoder unexpectedly contains frozen parameters: {frozen}")
    names = [name for name, _ in named]
    params = [parameter for _, parameter in named]

    groups = {
        "base_head": [idx for idx, name in enumerate(names) if name.startswith("base_head.")],
        "dagp_projection": [
            idx for idx, name in enumerate(names) if name.startswith("proj.") or name.startswith("value.")
        ],
        "dagp_graph_head": [idx for idx, name in enumerate(names) if name.startswith("graph_pred.")],
        "ndr_branch": [idx for idx, name in enumerate(names) if name.startswith("ndr_branch.")],
        "all_decoder": list(range(len(names))),
    }
    for group_name in MODULE_ORDER:
        if not groups[group_name]:
            raise RuntimeError(f"Parameter group is empty: {group_name}; model names={names}")
    covered = set().union(*(set(groups[name]) for name in MODULE_ORDER if name != "all_decoder"))
    expected = set(range(len(names)))
    if covered != expected:
        missing = [names[idx] for idx in sorted(expected - covered)]
        extra = [names[idx] for idx in sorted(covered - expected)]
        raise RuntimeError(f"Decoder parameter grouping mismatch: missing={missing}, extra={extra}")
    group_names = {
        group: [names[idx] for idx in indices]
        for group, indices in groups.items()
    }
    return named, params, groups, group_names


def dataloader_worker_kwargs(cfg, num_workers):
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "DATALOADER_PERSISTENT_WORKERS", False))
        prefetch = int(getattr(cfg, "DATALOADER_PREFETCH_FACTOR", 2))
        if prefetch > 0:
            kwargs["prefetch_factor"] = prefetch
    return kwargs


def build_source_loaders(cfg, batch_size, seed, num_workers, num_pairs):
    dataset = CachedTrainDataset(cfg, max_samples=-1)
    source_indices = {
        source: [
            index
            for index, item in enumerate(dataset.items)
            if str(item.get("dataset")) == source
        ]
        for source in SOURCES
    }
    loaders = {}
    metadata = {}
    for source_index, source in enumerate(SOURCES):
        indices = source_indices[source]
        if not indices:
            raise RuntimeError(f"No samples found for source {source}.")
        full_batches = len(indices) // int(batch_size)
        if int(num_pairs) > full_batches:
            raise RuntimeError(
                f"Requested {num_pairs} pairs but {source} has only {full_batches} complete "
                f"batches at batch_size={batch_size} ({len(indices)} samples)."
            )
        generator = torch.Generator()
        generator.manual_seed(int(seed) + source_index * 1_000_003)
        loaders[source] = DataLoader(
            Subset(dataset, indices),
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            **dataloader_worker_kwargs(cfg, num_workers),
        )
        metadata[source] = {
            "num_samples": len(indices),
            "num_full_batches": full_batches,
            "shuffle_seed": int(seed) + source_index * 1_000_003,
        }
    return dataset, loaders, metadata


def assert_training_batch(batch, source, batch_size):
    if "gt" in batch or "gt_path" in batch:
        raise RuntimeError(f"Gradient probe must not read training GT; batch keys={sorted(batch)}")
    datasets = list(batch.get("dataset", []))
    if len(datasets) != int(batch_size) or any(str(name) != source for name in datasets):
        raise RuntimeError(f"Source-pure batch violation for {source}: datasets={datasets}")
    if int(batch["feature"].shape[0]) != int(batch_size):
        raise RuntimeError(
            f"Batch-size mismatch for {source}: feature shape={list(batch['feature'].shape)}"
        )
    if bool(batch["feature"].requires_grad):
        raise RuntimeError("Cached DINO feature unexpectedly requires gradients.")


def compute_teacher_group_loss(output, teacher_target, teacher_map, routing_scale, cfg):
    if not isinstance(output, dict):
        raise RuntimeError("Epoch-21 NDR teacher group requires dict output.")
    missing = [key for key in ("coarse_logits_68", "base_logits") if key not in output]
    if missing:
        raise KeyError(f"Student output missing required epoch-21 teacher group logits: {missing}")
    eps = float(getattr(cfg, "DABE_PU_WEIGHTED_BCE_EPS", 1e-6))
    final_logits = resize_logits_for_loss(extract_logits(output), cfg)
    coarse_logits = resize_logits_for_loss(output["coarse_logits_68"], cfg)
    base_logits = resize_logits_for_loss(output["base_logits"], cfg)
    final_loss = rast_teacher_bce_with_logits(
        final_logits,
        teacher_target,
        teacher_map,
        cfg,
        routing_scale,
        apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_FINAL", True)),
        eps=eps,
    )
    coarse_loss = rast_teacher_bce_with_logits(
        coarse_logits,
        teacher_target,
        teacher_map,
        cfg,
        routing_scale,
        apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_COARSE_AUX", True)),
        eps=eps,
    )
    base_loss = rast_teacher_bce_with_logits(
        base_logits,
        teacher_target,
        teacher_map,
        cfg,
        routing_scale,
        apply_to_loss=bool(getattr(cfg, "RAST_APPLY_TO_BASE_AUX", True)),
        eps=eps,
    )
    coarse_weight = float(getattr(cfg, "LAMBDA_NDR_COARSE_AUX", 0.5))
    base_weight = float(getattr(cfg, "LAMBDA_BASE_AUX_AFTER_RESET", 0.3))
    weights = (1.0, coarse_weight, base_weight)
    total = (final_loss + coarse_weight * coarse_loss + base_weight * base_loss) / sum(weights)
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("Epoch-21 teacher group loss contains NaN/Inf.")
    return total, {
        "final": float(final_loss.detach().item()),
        "coarse": float(coarse_loss.detach().item()),
        "base": float(base_loss.detach().item()),
        "total": float(total.detach().item()),
    }


def flatten_group_gradients(gradients, params, group_indices):
    flattened = []
    for index in group_indices:
        gradient = gradients[index]
        if gradient is None:
            flattened.append(torch.zeros(params[index].numel(), dtype=torch.float64))
        else:
            if not bool(torch.isfinite(gradient).all().item()):
                raise RuntimeError("Gradient contains NaN/Inf.")
            flattened.append(gradient.detach().cpu().to(dtype=torch.float64).reshape(-1))
    return torch.cat(flattened, dim=0)


def compute_source_gradients(
    batch,
    source,
    student,
    teacher,
    params,
    groups,
    mode_configs,
    modes,
    base_cfg,
    device,
    batch_size,
):
    assert_training_batch(batch, source, batch_size)
    if any(parameter.grad is not None for parameter in student.parameters()):
        raise RuntimeError("Student .grad must remain empty before autograd.grad.")
    model_input = make_model_input(base_cfg, batch, device)
    image_68 = make_image_68(base_cfg, batch, device)
    with torch.no_grad():
        teacher_output = forward_seg_head(
            teacher,
            model_input,
            base_cfg,
            image_68=image_68,
            return_aux=False,
        )
        teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), base_cfg)
        teacher_target = (teacher_logits.sigmoid() >= 0.5).float().detach()

    student_output = forward_seg_head(
        student,
        model_input,
        base_cfg,
        image_68=image_68,
        return_aux=True,
    )
    mode_losses = {}
    mode_parts = {}
    mode_maps = {}
    mode_stats = {}
    for mode in modes:
        mode_cfg = mode_configs[mode]
        teacher_map, stats = build_rast_teacher_weight_map(
            mode_cfg,
            batch,
            teacher_target,
            PROBE_EPOCH,
            device,
        )
        if list(teacher_map.shape) != list(teacher_target.shape):
            raise RuntimeError(
                f"Teacher map shape mismatch for {mode}: {list(teacher_map.shape)} "
                f"!= {list(teacher_target.shape)}"
            )
        if not bool(torch.isfinite(teacher_map).all().item()):
            raise RuntimeError(f"Teacher map contains NaN/Inf for mode {mode}.")
        if mode == "teacher_only":
            map_error = float((teacher_map - 1.0).abs().max().item())
            if map_error > 1e-7 or float(stats["teacher_routing_scale"]) != 0.0:
                raise RuntimeError(
                    f"teacher_only must use an all-one map with routing scale 0; "
                    f"map_error={map_error}, scale={stats['teacher_routing_scale']}"
                )
        else:
            if str(stats.get("rast_phase")) != "esa_post_reset_extent_only":
                raise RuntimeError(f"Unexpected post-reset ESA phase: {stats.get('rast_phase')}")
            if abs(float(stats["teacher_routing_scale"]) - 1.0) > 1e-12:
                raise RuntimeError(
                    f"postreset_esa routing scale must be 1, got {stats['teacher_routing_scale']}"
                )
        loss, parts = compute_teacher_group_loss(
            student_output,
            teacher_target,
            teacher_map,
            float(stats["teacher_routing_scale"]),
            mode_cfg,
        )
        mode_losses[mode] = loss
        mode_parts[mode] = parts
        mode_maps[mode] = teacher_map.detach()
        mode_stats[mode] = stats

    mode_gradients = {}
    for mode_index, mode in enumerate(modes):
        gradients = torch.autograd.grad(
            mode_losses[mode],
            params,
            retain_graph=mode_index < len(modes) - 1,
            create_graph=False,
            allow_unused=True,
        )
        mode_gradients[mode] = {
            group: flatten_group_gradients(gradients, params, indices)
            for group, indices in groups.items()
        }
    if any(parameter.grad is not None for parameter in student.parameters()):
        raise RuntimeError("torch.autograd.grad unexpectedly accumulated into student .grad.")
    return {
        "gradients": mode_gradients,
        "loss_parts": mode_parts,
        "map_stats": mode_stats,
        "stems": [str(stem) for stem in batch["stem"]],
    }


def gradient_metrics(g_camo, g_cod, eps=EPS):
    if g_camo.shape != g_cod.shape:
        raise RuntimeError(f"Gradient shape mismatch: {list(g_camo.shape)} != {list(g_cod.shape)}")
    if g_camo.ndim != 1:
        raise RuntimeError(f"Flattened gradients must be 1-D, got {list(g_camo.shape)}")
    dot = float(torch.dot(g_camo, g_cod).item())
    norm_camo = float(torch.linalg.vector_norm(g_camo).item())
    norm_cod = float(torch.linalg.vector_norm(g_cod).item())
    norm_product = norm_camo * norm_cod
    cosine = dot / (norm_product + float(eps)) if norm_product > 0.0 else 0.0
    conflict = bool(cosine < 0.0 and norm_camo > eps and norm_cod > eps)
    conflict_energy = max(0.0, -dot) / (norm_product + float(eps)) if norm_product > 0.0 else 0.0
    if conflict:
        removed_camo_norm = max(0.0, -dot) / (norm_cod + float(eps))
        removed_cod_norm = max(0.0, -dot) / (norm_camo + float(eps))
        projection_loss_camo = removed_camo_norm / (norm_camo + float(eps))
        projection_loss_cod = removed_cod_norm / (norm_cod + float(eps))
    else:
        projection_loss_camo = 0.0
        projection_loss_cod = 0.0
    effective = (g_camo.abs() > eps) & (g_cod.abs() > eps)
    effective_count = int(effective.sum().item())
    if effective_count:
        disagreement = float(((g_camo[effective] * g_cod[effective]) < 0.0).double().mean().item())
    else:
        disagreement = 0.0
    values = {
        "dot_product": dot,
        "cosine": cosine,
        "norm_camo": norm_camo,
        "norm_cod": norm_cod,
        "norm_ratio_cod_over_camo": norm_cod / (norm_camo + float(eps)),
        "cod_norm_gt_camo": int(norm_cod > norm_camo),
        "conflict": int(conflict),
        "conflict_energy": conflict_energy,
        "projection_loss_camo": projection_loss_camo,
        "projection_loss_cod": projection_loss_cod,
        "sign_disagreement_ratio": disagreement,
        "effective_param_count": effective_count,
        "parameter_count": int(g_camo.numel()),
    }
    if any(not math.isfinite(float(value)) for key, value in values.items() if key != "parameter_count"):
        raise RuntimeError(f"Non-finite gradient metric: {values}")
    return values


def run_formula_self_test():
    aligned = gradient_metrics(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([2.0, 4.0], dtype=torch.float64),
    )
    opposite = gradient_metrics(
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, 4.0], dtype=torch.float64),
    )
    orthogonal = gradient_metrics(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    zero = gradient_metrics(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
    )
    if abs(aligned["cosine"] - 1.0) > 1e-10 or aligned["conflict"]:
        raise RuntimeError(f"Aligned gradient formula self-test failed: {aligned}")
    if abs(opposite["cosine"] + 1.0) > 1e-10 or opposite["conflict"] != 1:
        raise RuntimeError(f"Opposite gradient formula self-test failed: {opposite}")
    if abs(opposite["conflict_energy"] - 1.0) > 1e-10:
        raise RuntimeError(f"Conflict-energy self-test failed: {opposite}")
    if abs(opposite["projection_loss_camo"] - 1.0) > 1e-10:
        raise RuntimeError(f"PCGrad projection self-test failed: {opposite}")
    if abs(orthogonal["cosine"]) > 1e-12 or orthogonal["conflict"]:
        raise RuntimeError(f"Orthogonal gradient formula self-test failed: {orthogonal}")
    if any(not math.isfinite(float(value)) for value in zero.values()):
        raise RuntimeError(f"Zero-gradient formula self-test failed: {zero}")
    return True


def compare_gradient_results(first, second, modes):
    maximum = 0.0
    for mode in modes:
        for module in MODULE_ORDER:
            a = first["gradients"][mode][module]
            b = second["gradients"][mode][module]
            maximum = max(maximum, float((a - b).abs().max().item()))
    return maximum


def write_csv(path, rows, fields):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def aggregate_rows(per_pair_rows):
    grouped = defaultdict(list)
    for row in per_pair_rows:
        grouped[(row["mode"], row["module"])].append(row)
    summary_rows = []
    for mode, module in sorted(grouped, key=lambda key: (SUPPORTED_MODES.index(key[0]), MODULE_ORDER.index(key[1]))):
        rows = grouped[(mode, module)]
        cosines = np.asarray([float(row["cosine"]) for row in rows], dtype=np.float64)
        norm_ratios = np.asarray(
            [float(row["norm_ratio_cod_over_camo"]) for row in rows], dtype=np.float64
        )
        summary_rows.append(
            {
                "mode": mode,
                "module": module,
                "num_pairs": len(rows),
                "mean_cosine": float(cosines.mean()),
                "median_cosine": float(np.median(cosines)),
                "negative_cosine_ratio": mean([float(row["conflict"]) for row in rows]),
                "p10_cosine": float(np.quantile(cosines, 0.10)),
                "p90_cosine": float(np.quantile(cosines, 0.90)),
                "mean_norm_camo": mean([float(row["norm_camo"]) for row in rows]),
                "mean_norm_cod": mean([float(row["norm_cod"]) for row in rows]),
                "mean_norm_ratio": float(norm_ratios.mean()),
                "median_norm_ratio": float(np.median(norm_ratios)),
                "cod_norm_dominance_ratio": mean([float(row["cod_norm_gt_camo"]) for row in rows]),
                "mean_conflict_energy": mean([float(row["conflict_energy"]) for row in rows]),
                "mean_projection_loss_camo": mean(
                    [float(row["projection_loss_camo"]) for row in rows]
                ),
                "mean_projection_loss_cod": mean(
                    [float(row["projection_loss_cod"]) for row in rows]
                ),
                "mean_sign_disagreement_ratio": mean(
                    [float(row["sign_disagreement_ratio"]) for row in rows]
                ),
                "mean_loss_camo": mean([float(row["loss_camo"]) for row in rows]),
                "mean_loss_cod": mean([float(row["loss_cod"]) for row in rows]),
            }
        )
    return summary_rows


def summary_lookup(module_summary):
    return {(row["mode"], row["module"]): row for row in module_summary}


def classify_source_conflict(module_summary, num_pairs, primary_mode):
    if int(num_pairs) < 64:
        return {
            "class": "SANITY_ONLY",
            "confidence": "insufficient_pairs",
            "message": "Fewer than 64 batch pairs; no statistically effective B1/B2/B3 verdict is emitted.",
            "criteria": {},
        }
    lookup = summary_lookup(module_summary)
    overall = lookup[(primary_mode, "all_decoder")]
    core_modules = ("base_head", "dagp_projection", "dagp_graph_head", "ndr_branch")
    max_core_negative = max(float(lookup[(primary_mode, module)]["negative_cosine_ratio"]) for module in core_modules)
    criteria = {
        "all_decoder_negative_ratio_ge_0_40": float(overall["negative_cosine_ratio"]) >= 0.40,
        "all_decoder_mean_cosine_lt_0": float(overall["mean_cosine"]) < 0.0,
        "core_module_negative_ratio_ge_0_50": max_core_negative >= 0.50,
        "high_energy_and_stable_median": (
            float(overall["mean_conflict_energy"]) >= 0.10
            and float(overall["median_cosine"]) < 0.0
        ),
    }
    if sum(int(value) for value in criteria.values()) >= 3:
        return {
            "class": "B1",
            "confidence": "high",
            "message": (
                "Strong source-level gradient conflict is present. "
                "A conflict-aware teacher consolidation experiment is justified."
            ),
            "criteria": criteria,
        }
    if max_core_negative >= 0.50:
        return {
            "class": "B2",
            "confidence": "high",
            "message": (
                "Conflict is module-localized. Do not apply full-model PCGrad; "
                "restrict any future intervention to the conflicting module."
            ),
            "criteria": criteria,
        }
    explicit_b3 = float(overall["mean_cosine"]) > 0.0 and float(overall["negative_cosine_ratio"]) < 0.20
    return {
        "class": "B3",
        "confidence": "high" if explicit_b3 else "low_mixed",
        "message": (
            "Source-level gradient conflict is not the main bottleneck. "
            "Do not pursue CATC/PCGrad; return to ambiguous-pixel evidence resolution."
            if explicit_b3
            else "No strong or module-local B1/B2 condition was met, but the result is mixed."
        ),
        "criteria": criteria,
    }


def build_mode_comparison(module_summary, num_pairs, modes):
    if not {"teacher_only", "postreset_esa"}.issubset(set(modes)):
        return [], {
            "class": "NOT_AVAILABLE",
            "message": "Both teacher_only and postreset_esa are required for ESA comparison.",
        }
    lookup = summary_lookup(module_summary)
    rows = []
    all_decoder_effect = "no_clear_material_effect"
    for module in MODULE_ORDER:
        teacher = lookup[("teacher_only", module)]
        post = lookup[("postreset_esa", module)]
        delta_cosine = float(post["mean_cosine"]) - float(teacher["mean_cosine"])
        delta_negative = float(post["negative_cosine_ratio"]) - float(teacher["negative_cosine_ratio"])
        delta_energy = float(post["mean_conflict_energy"]) - float(teacher["mean_conflict_energy"])
        if delta_cosine <= -0.05 and delta_negative >= 0.10 and delta_energy >= 0.05:
            effect = "amplifies"
        elif delta_cosine >= 0.05 and delta_negative <= -0.10 and delta_energy <= -0.05:
            effect = "reduces"
        else:
            effect = "no_clear_material_effect"
        if module == "all_decoder":
            all_decoder_effect = effect
        rows.append(
            {
                "module": module,
                "teacher_only_mean_cosine": teacher["mean_cosine"],
                "postreset_esa_mean_cosine": post["mean_cosine"],
                "delta_mean_cosine": delta_cosine,
                "teacher_only_negative_cosine_ratio": teacher["negative_cosine_ratio"],
                "postreset_esa_negative_cosine_ratio": post["negative_cosine_ratio"],
                "delta_negative_cosine_ratio": delta_negative,
                "teacher_only_mean_conflict_energy": teacher["mean_conflict_energy"],
                "postreset_esa_mean_conflict_energy": post["mean_conflict_energy"],
                "delta_mean_conflict_energy": delta_energy,
                "teacher_only_mean_norm_ratio": teacher["mean_norm_ratio"],
                "postreset_esa_mean_norm_ratio": post["mean_norm_ratio"],
                "delta_mean_norm_ratio": float(post["mean_norm_ratio"]) - float(teacher["mean_norm_ratio"]),
                "esa_effect": effect,
            }
        )
    if int(num_pairs) < 64:
        verdict = {
            "class": "SANITY_ONLY",
            "message": "Fewer than 64 batch pairs; no statistically effective B4/B5 verdict is emitted.",
        }
    elif all_decoder_effect == "amplifies":
        verdict = {
            "class": "B4",
            "message": "Post-reset ESA amplifies CAMO-COD10K gradient conflict.",
        }
    elif all_decoder_effect == "reduces":
        verdict = {
            "class": "B5",
            "message": (
                "Post-reset ESA reduces gradient conflict but may still alter the optimization balance."
            ),
        }
    else:
        verdict = {
            "class": "NO_CLEAR_EFFECT",
            "message": "Post-reset ESA has no clear material effect on all-decoder gradient conflict.",
        }
    return rows, verdict


def strongest_conflict_module(module_summary, mode):
    candidates = [row for row in module_summary if row["mode"] == mode and row["module"] != "all_decoder"]
    return max(
        candidates,
        key=lambda row: (
            float(row["negative_cosine_ratio"]),
            float(row["mean_conflict_energy"]),
            -float(row["mean_cosine"]),
        ),
    )


def markdown_table(module_summary):
    lines = [
        "| Mode | Module | Mean cosine | Negative ratio | Conflict energy | COD/CAMO norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in module_summary:
        lines.append(
            f"| {row['mode']} | {row['module']} | {float(row['mean_cosine']):.6f} | "
            f"{float(row['negative_cosine_ratio']):.6f} | "
            f"{float(row['mean_conflict_energy']):.6f} | {float(row['mean_norm_ratio']):.6f} |"
        )
    return "\n".join(lines)


def write_reports(out_dir, metadata, per_pair_rows, module_summary, comparison_rows, source_verdict, esa_verdict):
    out_dir = Path(out_dir)
    write_csv(out_dir / "per_pair_gradients.csv", per_pair_rows, PER_PAIR_FIELDS)
    write_csv(out_dir / "module_summary.csv", module_summary, MODULE_SUMMARY_FIELDS)
    write_csv(out_dir / "mode_comparison.csv", comparison_rows, MODE_COMPARISON_FIELDS)

    primary_mode = metadata["primary_mode"]
    strongest = strongest_conflict_module(module_summary, primary_mode)
    strongest_conflict_present = (
        float(strongest["negative_cosine_ratio"]) > 0.0
        or float(strongest["mean_conflict_energy"]) > 0.0
    )
    all_decoder = summary_lookup(module_summary)[(primary_mode, "all_decoder")]
    summary = {
        "metadata": metadata,
        "source_conflict_verdict": source_verdict,
        "esa_effect_verdict": esa_verdict,
        "primary_all_decoder": all_decoder,
        "strongest_conflict_module": strongest,
        "strongest_conflict_present": strongest_conflict_present,
        "module_summary": module_summary,
        "mode_comparison": comparison_rows,
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = out_dir / "summary.md"
    lines = [
        "# Training-Source Gradient Conflict Probe",
        "",
        "## Probe State",
        "",
        f"- Config: `{metadata['config']}`",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Probe epoch: `{metadata['probe_epoch']}`",
        f"- Teacher source: `{metadata['teacher_source']}`",
        f"- Saved pre-reset student/teacher max diff: `{metadata['saved_teacher_max_abs_diff']}`",
        f"- Reconstructed student/teacher max diff: `{metadata['reconstructed_teacher_max_abs_diff']}`",
        f"- Batch pairs: `{metadata['num_pairs']}`",
        f"- Batch size: `{metadata['batch_size']}`",
        f"- Modes: `{', '.join(metadata['modes'])}`",
        f"- Parameter mutation max diff: `{metadata['student_parameter_max_abs_diff']}`",
        f"- Teacher mutation max diff: `{metadata['teacher_parameter_max_abs_diff']}`",
        f"- Gradient reproducibility max diff: `{metadata['gradient_reproducibility_max_abs_diff']}`",
        "- Optimizer created: `False`",
        "- DINO model loaded: `False` (cached features only)",
        "- Training GT read: `False`",
        "",
        "## Module Summary",
        "",
        markdown_table(module_summary),
        "",
        "## Automatic Verdict",
        "",
        f"- Source conflict: **{source_verdict['class']}** ({source_verdict['confidence']})",
        f"- {source_verdict['message']}",
        f"- ESA effect: **{esa_verdict['class']}**",
        f"- {esa_verdict['message']}",
        (
            f"- Strongest conflict-prone module: `{strongest['module']}` "
            f"(negative ratio={float(strongest['negative_cosine_ratio']):.6f}, "
            f"mean cosine={float(strongest['mean_cosine']):.6f})"
            if strongest_conflict_present
            else f"- No core module produced a negative cosine pair; lowest-alignment module: "
            f"`{strongest['module']}` (mean cosine={float(strongest['mean_cosine']):.6f})"
        ),
        f"- Primary all-decoder COD/CAMO norm ratio: `{float(all_decoder['mean_norm_ratio']):.6f}`",
        "",
        "## Notes",
        "",
        "- The epoch-20 checkpoint is saved before the configured after-epoch teacher reset.",
        "- Epoch-21 teacher was reconstructed from the saved epoch-20 student.",
        "- final/coarse/base teacher losses use normalized weights 1.0/0.5/0.3.",
        "- PCGrad values are diagnostic only; no projected gradient was applied to the model.",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = out_dir / "source_gradient_conflict_probe.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "summary.md",
            "summary.json",
            "per_pair_gradients.csv",
            "module_summary.csv",
            "mode_comparison.csv",
        ):
            archive.write(out_dir / name, arcname=name)
    return summary, zip_path


def run_probe(args):
    modes = parse_modes(args.modes)
    validate_args(args, modes)
    run_formula_self_test()
    device = resolve_device(args.device)
    set_seed(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg = load_config(args.config)
    validate_base_config(cfg)
    postreset_cfg = load_config(args.postreset_config) if "postreset_esa" in modes else None
    mode_configs = build_mode_configs(cfg, postreset_cfg, modes)
    student, teacher, checkpoint, saved_teacher_diff, reconstructed_diff = load_epoch21_models(
        cfg,
        Path(args.ckpt),
        device,
    )
    named_params, params, groups, group_names = build_parameter_groups(student)
    student_before = clone_state_dict(student)
    teacher_before = clone_state_dict(teacher)
    num_workers = int(cfg.NUM_WORKERS) if int(args.num_workers) == -1 else int(args.num_workers)
    dataset, loaders, source_metadata = build_source_loaders(
        cfg,
        args.batch_size,
        args.seed,
        num_workers,
        args.num_pairs,
    )
    if any("gt" in item or "gt_path" in item for item in dataset.items):
        raise RuntimeError("CachedTrainDataset items unexpectedly contain GT metadata.")

    print(f"device = {device}", flush=True)
    print(f"config = {Path(args.config).resolve()}", flush=True)
    print(f"checkpoint = {Path(args.ckpt).resolve()}", flush=True)
    print(f"teacher_source = reset_from_student_epoch20", flush=True)
    print(f"saved_teacher_max_abs_diff = {saved_teacher_diff}", flush=True)
    print(f"modes = {','.join(modes)}", flush=True)
    print(f"batch_size = {args.batch_size} | num_pairs = {args.num_pairs}", flush=True)
    print(f"parameter_groups = {group_names}", flush=True)
    print(f"source_metadata = {source_metadata}", flush=True)

    iterators = {source: iter(loaders[source]) for source in SOURCES}
    per_pair_rows = []
    gradient_reproducibility_max_diff = 0.0
    first_pair_cache = None
    for pair_index in range(int(args.num_pairs)):
        camo_batch = next(iterators["TR-CAMO"])
        cod_batch = next(iterators["TR-COD10K"])
        camo_result = compute_source_gradients(
            camo_batch,
            "TR-CAMO",
            student,
            teacher,
            params,
            groups,
            mode_configs,
            modes,
            cfg,
            device,
            args.batch_size,
        )
        if pair_index == 0:
            repeated = compute_source_gradients(
                camo_batch,
                "TR-CAMO",
                student,
                teacher,
                params,
                groups,
                mode_configs,
                modes,
                cfg,
                device,
                args.batch_size,
            )
            gradient_reproducibility_max_diff = compare_gradient_results(camo_result, repeated, modes)
            if gradient_reproducibility_max_diff > 1e-6:
                raise RuntimeError(
                    "Repeated-input gradients are not reproducible: "
                    f"max_abs_diff={gradient_reproducibility_max_diff:.12g}"
                )
            first_pair_cache = {
                mode: {
                    "teacher_map_mean": float(camo_result["map_stats"][mode]["teacher_map_mean"]),
                    "teacher_map_min": float(camo_result["map_stats"][mode]["teacher_map_min"]),
                    "teacher_map_max": float(camo_result["map_stats"][mode]["teacher_map_max"]),
                    "teacher_routing_scale": float(
                        camo_result["map_stats"][mode]["teacher_routing_scale"]
                    ),
                }
                for mode in modes
            }
        cod_result = compute_source_gradients(
            cod_batch,
            "TR-COD10K",
            student,
            teacher,
            params,
            groups,
            mode_configs,
            modes,
            cfg,
            device,
            args.batch_size,
        )
        for mode in modes:
            for module in MODULE_ORDER:
                metrics = gradient_metrics(
                    camo_result["gradients"][mode][module],
                    cod_result["gradients"][mode][module],
                )
                camo_parts = camo_result["loss_parts"][mode]
                cod_parts = cod_result["loss_parts"][mode]
                row = {
                    "pair_index": pair_index,
                    "mode": mode,
                    "module": module,
                    "batch_size": int(args.batch_size),
                    "camo_stems": "|".join(camo_result["stems"]),
                    "cod_stems": "|".join(cod_result["stems"]),
                    "loss_camo": camo_parts["total"],
                    "loss_cod": cod_parts["total"],
                    "loss_camo_final": camo_parts["final"],
                    "loss_camo_coarse": camo_parts["coarse"],
                    "loss_camo_base": camo_parts["base"],
                    "loss_cod_final": cod_parts["final"],
                    "loss_cod_coarse": cod_parts["coarse"],
                    "loss_cod_base": cod_parts["base"],
                    **metrics,
                }
                per_pair_rows.append(row)
        if (pair_index + 1) % int(args.progress_every) == 0 or pair_index + 1 == int(args.num_pairs):
            print(f"processed_pairs = {pair_index + 1}/{args.num_pairs}", flush=True)

    student_after = clone_state_dict(student)
    teacher_after = clone_state_dict(teacher)
    student_diff = max_state_diff(student_before, student_after)
    teacher_diff = max_state_diff(teacher_before, teacher_after)
    if student_diff != 0.0 or teacher_diff != 0.0:
        raise RuntimeError(
            f"Probe mutated model state: student_diff={student_diff}, teacher_diff={teacher_diff}."
        )
    if any(parameter.grad is not None for parameter in student.parameters()):
        raise RuntimeError("Student .grad is not empty after probe.")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("Teacher .grad is not empty after probe.")

    module_summary = aggregate_rows(per_pair_rows)
    primary_mode = "teacher_only" if "teacher_only" in modes else modes[0]
    source_verdict = classify_source_conflict(module_summary, args.num_pairs, primary_mode)
    comparison_rows, esa_verdict = build_mode_comparison(module_summary, args.num_pairs, modes)
    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "postreset_config": (
            str(Path(args.postreset_config).resolve()) if "postreset_esa" in modes else None
        ),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "probe_epoch": PROBE_EPOCH,
        "teacher_source": "reset_from_student_epoch20",
        "saved_teacher_max_abs_diff": saved_teacher_diff,
        "reconstructed_teacher_max_abs_diff": reconstructed_diff,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_pairs": int(args.num_pairs),
        "seed": int(args.seed),
        "num_workers": int(num_workers),
        "modes": modes,
        "primary_mode": primary_mode,
        "source_metadata": source_metadata,
        "parameter_groups": group_names,
        "num_trainable_parameters": sum(parameter.numel() for _, parameter in named_params),
        "student_parameter_max_abs_diff": student_diff,
        "teacher_parameter_max_abs_diff": teacher_diff,
        "gradient_reproducibility_max_abs_diff": gradient_reproducibility_max_diff,
        "optimizer_created": False,
        "scheduler_created": False,
        "ema_updated": False,
        "dino_model_loaded": False,
        "training_gt_read": False,
        "formula_self_test_passed": True,
        "first_pair_teacher_map": first_pair_cache,
        "statistically_effective": int(args.num_pairs) >= 64,
    }
    out_dir = Path(args.out).expanduser()
    ensure_dir(out_dir)
    summary, zip_path = write_reports(
        out_dir,
        metadata,
        per_pair_rows,
        module_summary,
        comparison_rows,
        source_verdict,
        esa_verdict,
    )
    all_decoder = summary_lookup(module_summary)[(primary_mode, "all_decoder")]
    strongest = summary["strongest_conflict_module"]
    print(
        "all_decoder | "
        f"mode={primary_mode} | mean_cosine={float(all_decoder['mean_cosine']):.6f} | "
        f"negative_ratio={float(all_decoder['negative_cosine_ratio']):.6f} | "
        f"norm_ratio={float(all_decoder['mean_norm_ratio']):.6f}",
        flush=True,
    )
    print(
        f"most_conflict_prone_module = {strongest['module']} | "
        f"actual_negative_conflict={bool(summary['strongest_conflict_present'])} | "
        f"negative_ratio={float(strongest['negative_cosine_ratio']):.6f}",
        flush=True,
    )
    print(f"source_verdict = {source_verdict['class']}", flush=True)
    print(f"esa_verdict = {esa_verdict['class']}", flush=True)
    print(f"report_zip = {zip_path.resolve()}", flush=True)
    return summary


def main():
    args = parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
