#!/usr/bin/env python3
"""Full-ECST causal, spatial-selectivity, and correction-quality audit.

The export/controls stages are GT-free.  Training GT can only be opened by the
explicit post-hoc stage together with ``--allow-training-gt-posthoc``.  No
optimizer, backward, EMA update, validation, or test-set evaluation is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CTH_ROOT = MAIN_ROOT.parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedTrainDataset, build_image_items  # noqa: E402
from common.ecst import build_ecst_teacher_weight_map  # noqa: E402
from common.ecst_causal_audit import (  # noqa: E402
    ECST_CAUSAL_CONTROL_SEED,
    build_block_shuffle_map,
    build_constant_mean_map,
    build_pixel_permutation_map,
    build_spatial_roll_map,
    compute_bootstrap_ci,
    compute_correction_masks,
    compute_correction_quality,
    compute_gradient_comparison,
    compute_gradient_magnitude_match_scalar,
    compute_gradient_retention,
    compute_logit_gradient_field,
    compute_map_spatial_autocorrelation,
    compute_precision_at_coverage,
    compute_score_ranking_metrics,
    normalized_weighted_bce,
    plain_mean_bce,
    validate_ecst_causal_audit_config,
)
from common.utils import (  # noqa: E402
    config_to_dict,
    load_config,
    make_jsonable,
    set_seed,
    torch_load,
)
from model import build_seg_head  # noqa: E402
from tools.inspect_full_ecst_checkpoints import inspect_checkpoints  # noqa: E402
from tools.replay_ecst_memory import replay_memory_snapshots  # noqa: E402
from train import (  # noqa: E402
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DEFAULT_CONFIG = MAIN_ROOT / "configs" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "workdir" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
) / "train" / "ckpt"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "workdir" / "dabev2hard_full_ecst_causal_audit"
)
PRIMARY_EPOCHS = (7, 10, 15, 19, 20)
EXPECTED_TRAIN_SAMPLES = 4040
PAYLOAD_PREFETCH_BATCHES = 1
FORBIDDEN_GT_KEYS = {
    "gt",
    "gt_path",
    "mask",
    "mask_path",
    "ground_truth",
}
CONTINUOUS_PAYLOAD_KEYS = {
    "dabe_v2_soft_68",
    "teacher_final_logit_68",
    "teacher_coarse_logit_68",
    "teacher_base_logit_68",
    "student_final_logit_68",
    "student_coarse_logit_68",
    "student_base_logit_68",
    "teacher_prob_68",
    "teacher_final_prob_68",
    "teacher_coarse_prob_68",
    "teacher_base_prob_68",
    "teacher_confidence_68",
    "dino_margin_68",
    "fg_tendency_68",
    "dino_ceiling_68",
    "temporal_mean_68",
    "temporal_variance_68",
    "temporal_stability_68",
    "bg_reliability_68",
    "ecst_raw_weight_68",
    "ecst_effective_weight_68",
    "dabe_v2_background_68",
    "dino_local_teacher_consistency_68",
    "teacher_flip_consistency_68",
}
BOOL_PAYLOAD_KEYS = {
    "dabe_v2_hard_68",
    "static_target_68",
    "teacher_binary_68",
    "fg_core_68",
    "bg_core_68",
    "extent_68",
    "unknown_68",
    "other_68",
    "fg_conflict_68",
    "bg_conflict_68",
    "core_conflict_68",
    "core_no_conflict_68",
    "extent_teacher_fg_68",
    "extent_teacher_bg_68",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--epochs", type=str, default=",".join(map(str, PRIMARY_EPOCHS)))
    parser.add_argument(
        "--stage",
        choices=("export", "controls", "correction_gt_audit", "report", "all"),
        required=True,
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-training-gt-posthoc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _inside_cth(path: Path) -> Path:
    value = path.expanduser()
    value = (Path.cwd() / value).resolve() if not value.is_absolute() else value.resolve()
    if value != CTH_ROOT and CTH_ROOT not in value.parents:
        raise RuntimeError(f"Path must stay inside {CTH_ROOT}: {value}")
    return value


def _epochs(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not result or result[0] < 1:
        raise ValueError("--epochs must contain positive integers.")
    return result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(cfg: Any) -> str:
    encoded = json.dumps(
        make_jsonable(config_to_dict(cfg)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _batch_strings(value: Any, size: int, name: str) -> list[str]:
    if isinstance(value, str) and size == 1:
        return [value]
    if isinstance(value, (list, tuple)) and len(value) == size:
        return [str(item) for item in value]
    raise RuntimeError(f"Cannot decode batch identity {name}: {value!r}")


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(make_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        path.write_text("status\nEMPTY\n", encoding="utf-8")
        return
    fields = sorted({str(key) for row in records for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: _json_cell(row.get(field)) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _payload_path(root: Path, epoch: int, dataset: str, stem: str) -> Path:
    return root / "payloads" / f"epoch_{epoch:03d}" / dataset / f"{stem}.pt"


def _epoch_payloads(root: Path, epoch: int) -> list[Path]:
    return sorted((root / "payloads" / f"epoch_{epoch:03d}").glob("*/*.pt"))


def _compact_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()
    if name in BOOL_PAYLOAD_KEYS:
        return (value > 0.5).bool()
    if name in CONTINUOUS_PAYLOAD_KEYS:
        return value.to(torch.float16)
    return value


def _as_4d(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = value.to(device=device).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[1]) != 1:
        raise RuntimeError(f"Expected [B,1,H,W], got {list(tensor.shape)}")
    return tensor


def _branch_logits(output: Any, cfg: Any) -> dict[str, torch.Tensor | None]:
    final = resize_logits_for_loss(extract_logits(output), cfg).detach()
    if not isinstance(output, dict):
        return {"final": final, "coarse": None, "base": None}

    def resolve(keys: Sequence[str]) -> torch.Tensor | None:
        for key in keys:
            value = output.get(key)
            if torch.is_tensor(value):
                if tuple(value.shape[-2:]) != (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)):
                    value = F.interpolate(
                        value,
                        size=(int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                        mode="bilinear",
                        align_corners=False,
                    )
                return value.detach()
        return None

    return {
        "final": final,
        "coarse": resolve(("coarse_logits_68", "coarse_logits")),
        "base": resolve(("base_logits_68", "base_logits")),
    }


def _dino_local_consistency(
    teacher_binary_68: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    binary_37 = F.interpolate(
        teacher_binary_68.float(), size=(37, 37), mode="nearest"
    ).flatten(2).squeeze(1)
    batch_size, nodes = binary_37.shape
    if tuple(topk_idx.shape[:2]) != (batch_size, nodes):
        raise RuntimeError("DAGP Top-K graph and Teacher grid are inconsistent.")
    neighbors = torch.gather(
        binary_37.unsqueeze(1).expand(-1, nodes, -1),
        2,
        topk_idx.detach().long(),
    )
    same = neighbors == binary_37.unsqueeze(-1)
    local_37 = (same.float() * topk_weight.detach().float()).sum(-1)
    local_37 = local_37.reshape(batch_size, 1, 37, 37)
    return F.interpolate(
        local_37, size=(68, 68), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0).detach()


def _same_source_background(
    dataset: CachedTrainDataset,
    datasets: Sequence[str],
    stems: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    if dataset.dabe_clean_dabe_v2_map is None:
        raise RuntimeError("DABE-v2 source manifest is unavailable.")
    values = []
    for dataset_name, stem in zip(datasets, stems):
        row = dataset.dabe_clean_dabe_v2_map[(dataset_name, stem)]
        payload = torch_load(row["cache_path"], map_location="cpu")
        if bool(payload.get("training_gt_read", False)):
            raise RuntimeError(
                f"DABE-v2 source reports training GT use: {row['cache_path']}"
            )
        bc = payload.get("bc_map_37")
        residual = payload.get("residual_norm_37", payload.get("residual_37"))
        if not torch.is_tensor(bc) or not torch.is_tensor(residual):
            raise RuntimeError(
                f"DABE-v2 cache lacks same-source B evidence: {row['cache_path']}"
            )
        background = bc.detach().float() * (1.0 - residual.detach().float())
        values.append(background.clamp(0.0, 1.0))
    stacked = torch.stack(values).to(device)
    return F.interpolate(
        stacked, size=(68, 68), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0).detach()


def _load_hflip_map(cfg: Any, expected_keys: set[tuple[str, str]]) -> dict[tuple[str, str], str] | None:
    configured = str(getattr(cfg, "HFLIP_FEATURE_CACHE_ROOT", "")).strip()
    roots = []
    if configured:
        roots.append(Path(configured))
    roots.append(Path("../datasets/cache/features_cache_hflip") / str(cfg.BACKBONE_KEY))
    for root in roots:
        resolved = _inside_cth(root)
        manifest = resolved / "manifest_train.jsonl"
        if not manifest.is_file():
            continue
        mapping: dict[tuple[str, str], str] = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["stem"]))
            path = row.get("cache_path", row.get("path"))
            if path is not None:
                cache_path = Path(path)
                if not cache_path.is_absolute():
                    cache_path = (manifest.parent / cache_path).resolve()
                mapping[key] = str(cache_path)
        if expected_keys.issubset(mapping) and len(mapping) == EXPECTED_TRAIN_SAMPLES:
            return mapping
    return None


def _hflip_feature(path: str, device: torch.device) -> torch.Tensor:
    payload = torch_load(path, map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = next(
            (
                payload[key]
                for key in ("tensor", "feature", "feat")
                if torch.is_tensor(payload.get(key))
            ),
            None,
        )
    else:
        tensor = None
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"Invalid hflip feature payload: {path}")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device).float()


def _validate_export_config(cfg: Any) -> dict[str, Any]:
    audit = validate_ecst_causal_audit_config(cfg)
    expected = {
        "USE_ECST": True,
        "TEACHER_ROUTING_MODE": "ecst",
        "DABE_CLEAN_STATIC_TARGET_SOURCE": "dabe_v2_hard_68",
        "DABE_CLEAN_DABE_V2_HARD_THRESHOLD": 0.5,
        "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": True,
    }
    bad = {
        name: getattr(cfg, name, None)
        for name, value in expected.items()
        if getattr(cfg, name, None) != value
    }
    if bad:
        raise RuntimeError(f"Full-ECST audit config mismatch: {bad}")
    return audit


def _protocol(
    *, cfg: Any, config_path: Path, epochs: Sequence[int], max_samples: int
) -> dict[str, Any]:
    return {
        "schema": "full_ecst_causal_audit_protocol_v1",
        "config": str(config_path),
        "config_fingerprint": _config_fingerprint(cfg),
        "epochs": list(map(int, epochs)),
        "primary_epochs": list(PRIMARY_EPOCHS),
        "max_samples": int(max_samples),
        "expected_full_manifest_samples": EXPECTED_TRAIN_SAMPLES,
        "teacher_bce_formula": "sum(weight * BCEWithLogits_none) / (sum(weight) + 1e-6)",
        "teacher_bce_normalization_scope": "whole_batch",
        "global_constant_map_equals_plain_mean": True,
        "per_image_constant_map_equals_plain_mean": False,
        "per_image_constant_note": (
            "Under whole-batch normalization, different per-image constants "
            "reweight images; it is retained only as a registered shadow control."
        ),
        "training_gt_used_for_training": False,
        "test_gt_used": False,
        "bootstrap_unit": "image",
        "bootstrap_repetitions": 1000,
        "seed": ECST_CAUSAL_CONTROL_SEED,
    }


def _checkpoint_memory(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in (
        "ecst_temporal_memory",
        "temporal_teacher_memory",
        "ecst_memory",
    ):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _validate_memory_state(
    state: Mapping[str, Any], *, sample_count: int, epoch: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, second, count = state.get("mean"), state.get("second"), state.get("count")
    expected_shape = (int(sample_count), 1, 68, 68)
    if not torch.is_tensor(mean) or tuple(mean.shape) != expected_shape:
        raise RuntimeError(
            f"ECST memory mean shape mismatch for epoch {epoch}: "
            f"{None if not torch.is_tensor(mean) else list(mean.shape)} != {list(expected_shape)}"
        )
    if not torch.is_tensor(second) or tuple(second.shape) != expected_shape:
        raise RuntimeError(f"ECST memory second shape mismatch for epoch {epoch}.")
    if not torch.is_tensor(count) or tuple(count.shape) != (int(sample_count),):
        raise RuntimeError(f"ECST memory count shape mismatch for epoch {epoch}.")
    if not bool(torch.isfinite(mean).all().item()) or not bool(
        torch.isfinite(second).all().item()
    ):
        raise RuntimeError(f"ECST memory contains NaN/Inf for epoch {epoch}.")
    if bool((count < 0).any().item()):
        raise RuntimeError(f"ECST memory has negative counts for epoch {epoch}.")
    return mean.detach().cpu(), second.detach().cpu(), count.detach().cpu().long()


def _ensure_memory_snapshots(
    *,
    cfg: Any,
    checkpoint_root: Path,
    output_root: Path,
    epochs: Sequence[int],
    max_samples: int,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    active_epochs = [
        int(epoch)
        for epoch in epochs
        if int(epoch) <= int(getattr(cfg, "ECST_MEMORY_UPDATE_END_EPOCH", 20))
    ]
    if not active_epochs:
        return {"status": "NOT_REQUIRED", "target_epochs": []}
    replay_targets = []
    exact_checkpoint_epochs = []
    for epoch in active_epochs:
        checkpoint_path = checkpoint_root / f"epoch_{epoch:03d}.pth"
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        if _checkpoint_memory(checkpoint) is None:
            replay_targets.append(epoch)
        else:
            exact_checkpoint_epochs.append(epoch)
    if replay_targets:
        replay = replay_memory_snapshots(
            cfg=cfg,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            target_epochs=replay_targets,
            max_samples=max_samples,
            batch_size=batch_size,
            device=device,
            overwrite=overwrite,
        )
    else:
        replay = {"status": "NOT_REQUIRED", "snapshots": []}
    return {
        "status": "READY",
        "exact_checkpoint_epochs": exact_checkpoint_epochs,
        "replayed_epochs": replay_targets,
        "replay": replay,
    }


def _history_for_epoch(
    *,
    checkpoint: Mapping[str, Any],
    output_root: Path,
    epoch: int,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if int(epoch) > 20:
        shape = (int(sample_count), 1, 68, 68)
        return (
            torch.zeros(shape, dtype=torch.float16),
            torch.zeros(shape, dtype=torch.float16),
            torch.zeros((int(sample_count),), dtype=torch.long),
            "post_reset_cleared",
        )
    direct = _checkpoint_memory(checkpoint)
    if direct is not None:
        mean, second, count = _validate_memory_state(
            direct, sample_count=sample_count, epoch=epoch
        )
        return mean, second, count, "checkpoint_exact"
    snapshot_path = output_root / "memory_snapshots" / f"epoch_{epoch:03d}.pt"
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Exact ECST history is unavailable for epoch {epoch}: {snapshot_path}"
        )
    snapshot = torch_load(snapshot_path, map_location="cpu")
    if int(snapshot.get("epoch", -1)) != int(epoch):
        raise RuntimeError(f"Memory snapshot epoch mismatch: {snapshot_path}")
    if bool(snapshot.get("training_gt_read", True)):
        raise RuntimeError(f"Memory replay reports GT access: {snapshot_path}")
    mean, second, count = _validate_memory_state(
        snapshot, sample_count=sample_count, epoch=epoch
    )
    return mean, second, count, "checkpoint_replay"


@torch.no_grad()
def export_payloads(
    *,
    cfg: Any,
    config_path: Path,
    checkpoint_root: Path,
    output_root: Path,
    epochs: Sequence[int],
    max_samples: int,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    _validate_export_config(cfg)
    inventory = inspect_checkpoints(
        config_path=config_path,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        epochs=sorted(set(map(int, epochs)) | set(PRIMARY_EPOCHS) | {24, 25, 27}),
    )
    missing = [
        int(row["requested_epoch"])
        for row in inventory["rows"]
        if int(row["requested_epoch"]) in set(map(int, epochs))
        and not bool(row["checkpoint_exists"])
    ]
    if missing:
        raise FileNotFoundError(f"Requested Full-ECST checkpoints are missing: {missing}")

    dataset = CachedTrainDataset(cfg, max_samples=int(max_samples))
    if len(dataset) < 1:
        raise RuntimeError("Full-ECST audit dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    memory_summary = _ensure_memory_snapshots(
        cfg=cfg,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        epochs=epochs,
        max_samples=max_samples,
        batch_size=batch_size,
        device=device,
        overwrite=overwrite,
    )
    expected_keys = set(dataset.keys)
    hflip_map = _load_hflip_map(cfg, expected_keys)
    hflip_status = "AVAILABLE_FULL" if hflip_map is not None else "SKIPPED_MISSING_CACHE"

    student = build_seg_head(dataset.in_channels, cfg).to(device).eval()
    teacher = build_seg_head(dataset.in_channels, cfg).to(device).eval()
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    payload_manifest_rows: list[dict[str, Any]] = []
    epoch_summaries = []
    for epoch in map(int, epochs):
        checkpoint_path = checkpoint_root / f"epoch_{epoch:03d}.pth"
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        if int(checkpoint.get("epoch", -1)) != epoch:
            raise RuntimeError(f"Checkpoint epoch mismatch: {checkpoint_path}")
        if not isinstance(checkpoint.get("teacher"), dict) or not isinstance(
            checkpoint.get("student"), dict
        ):
            raise RuntimeError(f"Student/Teacher state missing: {checkpoint_path}")
        student.load_state_dict(checkpoint["student"], strict=True)
        teacher.load_state_dict(checkpoint["teacher"], strict=True)
        set_model_epoch(student, epoch)
        set_model_epoch(teacher, epoch)
        checkpoint_hash = _sha256(checkpoint_path)
        mean_bank, second_bank, count_bank, memory_source = _history_for_epoch(
            checkpoint=checkpoint,
            output_root=output_root,
            epoch=epoch,
            sample_count=len(dataset),
        )
        exported = 0
        for batch in loader:
            leaked = sorted(
                key for key in batch if str(key).strip().lower() in FORBIDDEN_GT_KEYS
            )
            if leaked:
                raise RuntimeError(f"GT leaked into export batch: {leaked}")
            indices = batch["sample_index"].detach().long().reshape(-1)
            batch_size_actual = int(indices.numel())
            datasets = _batch_strings(batch["dataset_name"], batch_size_actual, "dataset")
            stems = _batch_strings(batch["stem"], batch_size_actual, "stem")
            model_input = make_model_input(cfg, batch, device)
            image_68 = make_image_68(cfg, batch, device)
            student_output = forward_seg_head(
                student,
                model_input,
                cfg,
                image_68=image_68,
                return_aux=True,
                return_eaogp_aux=False,
            )
            teacher_output = forward_seg_head(
                teacher,
                model_input,
                cfg,
                image_68=image_68,
                return_aux=True,
                return_eaogp_aux=True,
            )
            student_branches = _branch_logits(student_output, cfg)
            teacher_branches = _branch_logits(teacher_output, cfg)
            teacher_prob = torch.sigmoid(teacher_branches["final"]).detach()
            teacher_binary = (teacher_prob >= 0.5).float().detach()
            temporal_mean = mean_bank.index_select(0, indices).to(device).float()
            temporal_second = second_bank.index_select(0, indices).to(device).float()
            history_count = count_bank.index_select(0, indices).to(device)
            effective, ecst_stats, raw, states = build_ecst_teacher_weight_map(
                cfg=cfg,
                batch=batch,
                teacher_prob=teacher_prob,
                temporal_mean=temporal_mean,
                temporal_second=temporal_second,
                history_count=history_count,
                epoch=epoch,
                device=device,
                return_raw=True,
                return_states=True,
                region_key_prefix="legacy_ecst",
            )
            static = batch["dabe_clean_static_target_68"].to(device).float().detach()
            dabe_soft = batch["dabe_clean_dabe_v2_soft_68"].to(device).float().detach()
            expected_hard = (dabe_soft > 0.5).float()
            if not torch.equal(static, expected_hard):
                raise RuntimeError("Static target is not strict DABE-v2 p_dabe_68 > 0.5.")
            masks = states["masks"]
            partition_sum = sum(value.int() for value in masks.values())
            if not bool(torch.all(partition_sum == 1).item()):
                raise RuntimeError("Full-ECST region partition is not one-hot.")
            if (
                effective.requires_grad
                or not bool(torch.isfinite(effective).all().item())
                or float(effective.min().item()) < 0.2 - 1e-6
                or float(effective.max().item()) > 1.0 + 1e-6
            ):
                raise RuntimeError("Full-ECST effective map invariant failed.")
            local_consistency = _dino_local_consistency(
                teacher_binary,
                teacher_output["eaogp_dino_topk_idx"],
                teacher_output["eaogp_dino_topk_weight"],
            )
            background = _same_source_background(dataset, datasets, stems, device)

            flip_consistency = None
            if hflip_map is not None:
                flip_features = torch.cat(
                    [_hflip_feature(hflip_map[(d, s)], device) for d, s in zip(datasets, stems)],
                    dim=0,
                )
                flip_output = forward_seg_head(
                    teacher,
                    flip_features,
                    cfg,
                    image_68=torch.flip(image_68, dims=(-1,)),
                    return_aux=False,
                )
                flip_prob = torch.sigmoid(
                    resize_logits_for_loss(extract_logits(flip_output), cfg)
                )
                flip_prob = torch.flip(flip_prob, dims=(-1,)).detach()
                flip_consistency = (1.0 - (teacher_prob - flip_prob).abs()).clamp(0.0, 1.0)

            static_global, teacher_global = get_dabe_pu_despl_schedule(epoch, cfg)
            for local_index, (dataset_name, stem) in enumerate(zip(datasets, stems)):
                path = _payload_path(output_root, epoch, dataset_name, stem)
                if path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Audit payload already exists; use --overwrite: {path}"
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                item = dataset.items[int(indices[local_index].item())]
                with Image.open(item["image_path"]) as image:
                    image_size = list(image.size)
                payload: dict[str, Any] = {
                    "schema": "full_ecst_pixel_audit_payload_v1",
                    "dataset": dataset_name,
                    "stem": stem,
                    "sample_index": int(indices[local_index].item()),
                    "epoch": epoch,
                    "image_size": image_size,
                    "image_path": str(item["image_path"]),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "memory_source": memory_source,
                    "dabe_v2_soft_68": dabe_soft[local_index],
                    "dabe_v2_hard_68": expected_hard[local_index],
                    "static_target_68": static[local_index],
                    "static_fg_area": float(static[local_index].mean().item()),
                    "teacher_prob_68": teacher_prob[local_index],
                    "teacher_binary_68": teacher_binary[local_index],
                    "teacher_binary_area": float(teacher_binary[local_index].mean().item()),
                    "teacher_confidence_68": 2.0 * (teacher_prob[local_index] - 0.5).abs(),
                    "teacher_final_prob_68": torch.sigmoid(teacher_branches["final"][local_index]),
                    "teacher_coarse_prob_68": (
                        None
                        if teacher_branches["coarse"] is None
                        else torch.sigmoid(teacher_branches["coarse"][local_index])
                    ),
                    "teacher_base_prob_68": (
                        None
                        if teacher_branches["base"] is None
                        else torch.sigmoid(teacher_branches["base"][local_index])
                    ),
                    "teacher_final_logit_68": teacher_branches["final"][local_index],
                    "teacher_coarse_logit_68": (
                        None if teacher_branches["coarse"] is None else teacher_branches["coarse"][local_index]
                    ),
                    "teacher_base_logit_68": (
                        None if teacher_branches["base"] is None else teacher_branches["base"][local_index]
                    ),
                    "student_final_logit_68": student_branches["final"][local_index],
                    "student_coarse_logit_68": (
                        None if student_branches["coarse"] is None else student_branches["coarse"][local_index]
                    ),
                    "student_base_logit_68": (
                        None if student_branches["base"] is None else student_branches["base"][local_index]
                    ),
                    "dino_margin_68": states["margin_68"][local_index],
                    "fg_tendency_68": states["fg_tendency"][local_index],
                    "dino_ceiling_68": states["dino_ceiling"][local_index],
                    "temporal_mean_68": states["mean"][local_index],
                    "temporal_variance_68": states["variance"][local_index],
                    "temporal_stability_68": states["stability"][local_index],
                    "bg_reliability_68": states["bg_reliability"][local_index],
                    "history_valid": bool(states["history_valid"][local_index].item()),
                    "history_count": int(history_count[local_index].item()),
                    "ecst_raw_weight_68": raw[local_index],
                    "ecst_effective_weight_68": effective[local_index],
                    "ecst_scale": float(ecst_stats["ecst_scale"]),
                    "ecst_margin_tau": float(getattr(cfg, "ECST_MARGIN_TAU", 0.05)),
                    "weight_min": float(effective[local_index].min().item()),
                    "weight_mean": float(effective[local_index].mean().item()),
                    "weight_max": float(effective[local_index].max().item()),
                    "static_global_weight": float(static_global),
                    "teacher_global_weight": float(teacher_global),
                    "training_gt_read": False,
                    "test_gt_read": False,
                    "dabe_v2_background_68": background[local_index],
                    "dino_local_teacher_consistency_68": local_consistency[local_index],
                    "hflip_status": hflip_status,
                }
                for name, value in masks.items():
                    payload[f"{name}_68"] = value[local_index]
                for name in (
                    "fg_conflict",
                    "bg_conflict",
                    "core_conflict",
                    "core_no_conflict",
                    "extent_teacher_fg",
                    "extent_teacher_bg",
                ):
                    payload[f"{name}_68"] = states[name][local_index]
                if flip_consistency is not None:
                    payload["teacher_flip_consistency_68"] = flip_consistency[local_index]
                compact = {
                    name: (_compact_tensor(name, value) if torch.is_tensor(value) else value)
                    for name, value in payload.items()
                }
                torch.save(compact, path)
                payload_manifest_rows.append(
                    {
                        "epoch": epoch,
                        "dataset": dataset_name,
                        "stem": stem,
                        "sample_index": int(indices[local_index].item()),
                        "payload_path": str(path),
                        "checkpoint_sha256": checkpoint_hash,
                        "memory_source": memory_source,
                        "training_gt_read": False,
                    }
                )
                exported += 1
        epoch_summaries.append(
            {
                "epoch": epoch,
                "sample_count": exported,
                "checkpoint_sha256": checkpoint_hash,
                "memory_source": memory_source,
                "hflip_status": hflip_status,
            }
        )

    _write_csv(output_root / "payload_manifest.csv", payload_manifest_rows)
    summary = {
        "schema": "full_ecst_payload_export_v1",
        "status": "PASS",
        "sample_count_per_epoch": len(dataset),
        "full_manifest": int(max_samples) < 0 and len(dataset) == EXPECTED_TRAIN_SAMPLES,
        "epochs": list(map(int, epochs)),
        "epoch_summaries": epoch_summaries,
        "memory": memory_summary,
        "hflip_status": hflip_status,
        "training_gt_read": False,
        "test_gt_read": False,
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(make_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


class _ProgressReporter:
    def __init__(self, label: str, total: int) -> None:
        self.label = str(label)
        self.total = int(total)
        self.completed = 0
        self.started = time.perf_counter()
        self.interval = max(1, self.total // 20)
        print(f"[{self.label}] start total={self.total}", flush=True)

    def update(self, *, detail: str = "") -> None:
        self.completed += 1
        if self.completed != self.total and self.completed % self.interval:
            return
        elapsed = time.perf_counter() - self.started
        rate = self.completed / elapsed if elapsed > 0.0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0.0 else float("inf")
        suffix = f" | {detail}" if detail else ""
        print(
            f"[{self.label}] {self.completed}/{self.total} "
            f"({100.0 * self.completed / max(1, self.total):.1f}%) "
            f"elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta) if math.isfinite(eta) else 'unknown'}"
            f"{suffix}",
            flush=True,
        )


def _load_payload_items(paths: Sequence[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, torch_load(path, map_location="cpu")) for path in paths]


def _payload_item_batches(
    paths: Sequence[Path], batch_size: int
) -> Iterable[list[tuple[Path, dict[str, Any]]]]:
    """Load one future batch while preserving the exact path and batch order."""

    starts = list(range(0, len(paths), int(batch_size)))
    if not starts:
        return
    with ThreadPoolExecutor(
        max_workers=PAYLOAD_PREFETCH_BATCHES,
        thread_name_prefix="ecst-payload-prefetch",
    ) as executor:
        future = executor.submit(
            _load_payload_items,
            paths[starts[0] : starts[0] + int(batch_size)],
        )
        for position, start in enumerate(starts):
            items = future.result()
            if position + 1 < len(starts):
                next_start = starts[position + 1]
                future = executor.submit(
                    _load_payload_items,
                    paths[next_start : next_start + int(batch_size)],
                )
            yield items


def _payload_batches(paths: Sequence[Path], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for items in _payload_item_batches(paths, batch_size):
        yield [payload for _, payload in items]


def _payload_items(
    paths: Sequence[Path], batch_size: int
) -> Iterable[tuple[Path, dict[str, Any]]]:
    for items in _payload_item_batches(paths, batch_size):
        yield from items


def _stack_payload(
    payloads: Sequence[Mapping[str, Any]], key: str, device: torch.device
) -> torch.Tensor:
    values = []
    for payload in payloads:
        value = payload.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"Payload field {key!r} is unavailable.")
        values.append(value.detach().float())
    return torch.stack(values).to(device)


def _aggregate_numeric_rows(
    rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output = []
    for key, records in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(group_fields, key))
        fields = sorted({name for record in records for name in record})
        for field in fields:
            if field in group_fields:
                continue
            values = []
            for record in records:
                value = record.get(field)
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                result[field] = sum(values) / len(values)
        result["record_count"] = len(records)
        output.append(result)
    return output


@torch.no_grad()
def audit_shadow_controls(
    *,
    output_root: Path,
    epochs: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    global_constant_max_error = 0.0
    gmg_l1_max_error = 0.0
    histogram_max_error = 0.0
    original_self_cosine_max_error = 0.0
    epoch_paths: dict[int, list[Path]] = {}
    for epoch in map(int, epochs):
        paths = _epoch_payloads(output_root, epoch)
        if not paths:
            raise FileNotFoundError(f"No exported payloads for epoch {epoch}.")
        epoch_paths[epoch] = paths
    total_batches = sum(
        math.ceil(len(paths) / int(batch_size)) for paths in epoch_paths.values()
    )
    progress = _ProgressReporter("controls", total_batches)
    for epoch in map(int, epochs):
        paths = epoch_paths[epoch]
        epoch_batch_count = math.ceil(len(paths) / int(batch_size))
        for batch_index, payloads in enumerate(_payload_batches(paths, batch_size)):
            datasets = [str(payload["dataset"]) for payload in payloads]
            stems = [str(payload["stem"]) for payload in payloads]
            target = _stack_payload(payloads, "teacher_binary_68", device)
            static = _stack_payload(payloads, "static_target_68", device)
            original = _stack_payload(payloads, "ecst_effective_weight_68", device)
            conflict = target.bool() != static.bool()
            global_constant = build_constant_mean_map(original, per_image=False)
            per_image_constant = build_constant_mean_map(original, per_image=True)
            maps = {
                "original_ecst": original,
                "identity": torch.ones_like(original),
                "global_constant": global_constant,
                "per_image_constant": per_image_constant,
                "spatial_roll": build_spatial_roll_map(original, datasets, stems),
                "block_shuffle": build_block_shuffle_map(original, datasets, stems),
                "pixel_permutation": build_pixel_permutation_map(original, datasets, stems),
            }
            original_sorted = torch.sort(original.flatten(1), dim=1).values
            for variant in ("spatial_roll", "block_shuffle", "pixel_permutation"):
                error = float(
                    (
                        torch.sort(maps[variant].flatten(1), dim=1).values
                        - original_sorted
                    )
                    .abs()
                    .max()
                    .item()
                )
                histogram_max_error = max(histogram_max_error, error)
                if error > 1e-7:
                    raise RuntimeError(
                        f"{variant} changed the ECST histogram: {error:.9g}"
                    )
            for branch in ("final", "coarse", "base"):
                key = f"student_{branch}_logit_68"
                if any(not torch.is_tensor(payload.get(key)) for payload in payloads):
                    rows.append(
                        {
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "branch": branch,
                            "variant": "SKIPPED_BRANCH_UNAVAILABLE",
                            "sample_count": len(payloads),
                        }
                    )
                    continue
                logits = _stack_payload(payloads, key, device)
                plain_loss = plain_mean_bce(logits, target)
                reference_gradient = compute_logit_gradient_field(
                    logits, target, original, mode="normalized_weighted"
                )
                scalar_l1 = compute_gradient_magnitude_match_scalar(
                    logits, target, original, norm="l1"
                )
                scalar_l2 = compute_gradient_magnitude_match_scalar(
                    logits, target, original, norm="l2"
                )
                gmg_gradient = compute_logit_gradient_field(
                    logits, target, mode="gmg", global_scalar=scalar_l1
                )
                gmg_l1_error = abs(
                    float(gmg_gradient.abs().sum().item())
                    - float(reference_gradient.abs().sum().item())
                )
                gmg_l1_max_error = max(gmg_l1_max_error, gmg_l1_error)
                gmg_comparison = compute_gradient_comparison(
                    reference_gradient,
                    gmg_gradient,
                    foreground_mask=target > 0.5,
                    conflict_mask=conflict,
                )
                rows.append(
                    {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "branch": branch,
                        "variant": "gmg_global_l1",
                        "sample_count": len(payloads),
                        "loss": float((scalar_l1 * plain_loss).item()),
                        "plain_loss": float(plain_loss.item()),
                        "gmg_scalar_l1": float(scalar_l1.item()),
                        "gmg_scalar_l2_shadow": float(scalar_l2.item()),
                        "gmg_gradient_l1_abs_error": gmg_l1_error,
                        **gmg_comparison,
                    }
                )
                for variant, weight in maps.items():
                    loss = normalized_weighted_bce(logits, target, weight)
                    gradient = compute_logit_gradient_field(
                        logits,
                        target,
                        weight,
                        mode="normalized_weighted",
                    )
                    comparison = compute_gradient_comparison(
                        reference_gradient,
                        gradient,
                        foreground_mask=target > 0.5,
                        conflict_mask=conflict,
                    )
                    if variant == "original_ecst":
                        cosine = comparison["gradient_cosine_to_original"]
                        if cosine is None:
                            raise RuntimeError(
                                "Original ECST gradient unexpectedly has zero norm."
                            )
                        original_self_cosine_max_error = max(
                            original_self_cosine_max_error,
                            abs(float(cosine) - 1.0),
                        )
                    error = abs(float(loss.item()) - float(plain_loss.item()))
                    if variant == "global_constant":
                        global_constant_max_error = max(global_constant_max_error, error)
                    rows.append(
                        {
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "branch": branch,
                            "variant": variant,
                            "sample_count": len(payloads),
                            "loss": float(loss.item()),
                            "plain_loss": float(plain_loss.item()),
                            "loss_minus_plain": float(loss.item() - plain_loss.item()),
                            "plain_abs_error": error,
                            "meanfield_loss": (
                                float(loss.item())
                                if variant in {"global_constant", "per_image_constant"}
                                else None
                            ),
                            "meanfield_plain_abs_error": (
                                error
                                if variant in {"global_constant", "per_image_constant"}
                                else None
                            ),
                            "meanfield_scope": (
                                "global"
                                if variant == "global_constant"
                                else "per_image"
                                if variant == "per_image_constant"
                                else None
                            ),
                            "weight_min": float(weight.min().item()),
                            "weight_mean": float(weight.mean().item()),
                            "weight_max": float(weight.max().item()),
                            "spatial_autocorrelation": compute_map_spatial_autocorrelation(weight),
                            **comparison,
                        }
                    )
            progress.update(
                detail=f"epoch={epoch} batch={batch_index + 1}/{epoch_batch_count}"
            )
    if global_constant_max_error > 1e-7:
        raise RuntimeError(
            "Global constant normalized map did not equal plain mean BCE: "
            f"max_abs_error={global_constant_max_error:.9g}."
        )
    if gmg_l1_max_error > 1e-6:
        raise RuntimeError(
            f"GMG L1 gradient match failed: max_abs_error={gmg_l1_max_error:.9g}."
        )
    if original_self_cosine_max_error > 1e-6:
        raise RuntimeError(
            "Original ECST self-cosine failed: "
            f"max_abs_error={original_self_cosine_max_error:.9g}."
        )
    _write_csv(output_root / "shadow_control_batches.csv", rows)
    summary_rows = _aggregate_numeric_rows(rows, ("epoch", "branch", "variant"))
    _write_csv(output_root / "map_control_summary.csv", summary_rows)
    summary = {
        "schema": "full_ecst_shadow_controls_v1",
        "status": "PASS",
        "epochs": list(map(int, epochs)),
        "batch_rows": len(rows),
        "global_constant_plain_max_abs_error": global_constant_max_error,
        "normalized_bce_constant_map_equals_plain": global_constant_max_error <= 1e-7,
        "per_image_constant_is_not_claimed_equivalent": True,
        "gmg_l1_gradient_max_abs_error": gmg_l1_max_error,
        "original_ecst_self_cosine_max_abs_error": original_self_cosine_max_error,
        "spatial_controls_histogram_max_abs_error": histogram_max_error,
        "payload_prefetch_batches": PAYLOAD_PREFETCH_BATCHES,
        "progress_reporting": True,
        "backward_called": False,
        "gt_read": False,
    }
    (output_root / "controls_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _training_gt_map(cfg: Any) -> dict[tuple[str, str], str]:
    allowed = {"TR-CAMO", "TR-COD10K"}
    configured = {str(value) for value in cfg.TRAIN_DATASETS}
    if not configured or not configured.issubset(allowed):
        raise RuntimeError(
            "Post-hoc audit permits only TR-CAMO/TR-COD10K training GT; "
            f"configured={sorted(configured)}"
        )
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=True)
    result = {}
    for item in items:
        dataset = str(item["dataset"])
        gt_path = item.get("gt_path")
        if dataset not in allowed or not gt_path:
            raise RuntimeError(f"Invalid training GT item: {item}")
        result[(dataset, str(item["stem"]))] = str(gt_path)
    return result


def _load_gt_68(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        mask = image.convert("L").resize((68, 68), resample=Image.Resampling.NEAREST)
        array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array > 0.5).unsqueeze(0)


def _candidate_scores(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    teacher = payload["teacher_binary_68"].detach().float()
    static = payload["static_target_68"].detach().float()
    dabe_soft = payload["dabe_v2_soft_68"].detach().float()
    background = payload["dabe_v2_background_68"].detach().float()
    confidence = payload["teacher_confidence_68"].detach().float()
    margin = payload["dino_margin_68"].detach().float()
    tau = float(payload.get("ecst_margin_tau", 0.05))
    anchor = static * dabe_soft * (1.0 - background) + (
        1.0 - static
    ) * background * (1.0 - dabe_soft)
    scores = {
        "full_ecst_weight": payload["ecst_effective_weight_68"].detach().float(),
        "teacher_confidence": confidence,
        "dabe_static_uncertainty": 1.0 - 2.0 * (dabe_soft - 0.5).abs(),
        "dabe_anchor_reverse": 1.0 - anchor,
        "dino_margin_teacher_support": torch.sigmoid(
            (2.0 * teacher - 1.0) * margin / tau
        ),
        "dino_local_teacher_consistency": payload[
            "dino_local_teacher_consistency_68"
        ].detach().float(),
    }
    if torch.is_tensor(payload.get("teacher_coarse_prob_68")):
        scores["coarse_final_consistency"] = 1.0 - (
            payload["teacher_final_prob_68"].detach().float()
            - payload["teacher_coarse_prob_68"].detach().float()
        ).abs()
    if torch.is_tensor(payload.get("teacher_base_prob_68")):
        scores["base_final_consistency"] = 1.0 - (
            payload["teacher_final_prob_68"].detach().float()
            - payload["teacher_base_prob_68"].detach().float()
        ).abs()
    temporal = torch.full_like(teacher, float("nan"))
    temporal[teacher < 0.5] = payload["bg_reliability_68"].detach().float()[
        teacher < 0.5
    ]
    scores["temporal_bg_reliability"] = temporal
    flip = payload.get("teacher_flip_consistency_68")
    if torch.is_tensor(flip):
        scores["hflip_consistency"] = flip.detach().float()
        scores["hflip_x_teacher_confidence"] = flip.detach().float() * confidence
    return {name: value.clamp(0.0, 1.0) for name, value in scores.items()}


def _control_maps(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    original = payload["ecst_effective_weight_68"].detach().float().unsqueeze(0)
    dataset, stem = str(payload["dataset"]), str(payload["stem"])
    return {
        "original_ecst": original.squeeze(0),
        "identity": torch.ones_like(original).squeeze(0),
        "global_constant": build_constant_mean_map(original, per_image=False).squeeze(0),
        "per_image_constant": build_constant_mean_map(original, per_image=True).squeeze(0),
        "spatial_roll": build_spatial_roll_map(original, [dataset], [stem]).squeeze(0),
        "block_shuffle": build_block_shuffle_map(original, [dataset], [stem]).squeeze(0),
        "pixel_permutation": build_pixel_permutation_map(original, [dataset], [stem]).squeeze(0),
    }


def _safe_mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _ranking_row(
    *,
    epoch: int,
    dataset: str,
    direction: str,
    candidate: str,
    scores: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
) -> dict[str, Any]:
    score = torch.cat([value.reshape(-1) for value in scores]) if scores else torch.empty(0)
    label = torch.cat([value.reshape(-1) for value in labels]) if labels else torch.empty(0, dtype=torch.bool)
    metrics = compute_score_ranking_metrics(score, label)
    row = {
        "epoch": epoch,
        "dataset": dataset,
        "direction": direction,
        "candidate": candidate,
        **metrics,
    }
    for threshold, value in metrics.get("balanced_accuracy", {}).items():
        row[f"balanced_accuracy_at_{threshold}"] = value
    row.pop("balanced_accuracy", None)
    return row


def _image_macro_metric(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    return _safe_mean([record.get(name) for record in records])


def _mask_panel(mask: torch.Tensor) -> np.ndarray:
    array = (mask.detach().float().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    return array


def _heat_panel(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().squeeze().clamp(0.0, 1.0).numpy()
    return np.uint8(np.round(array * 255.0))


def _render_case(
    *, payload_path: Path, gt_path: str, output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = torch_load(payload_path, map_location="cpu")
    gt = _load_gt_68(gt_path)
    static = payload["static_target_68"].bool()
    teacher = payload["teacher_binary_68"].bool()
    masks = compute_correction_masks(static, teacher, gt)
    correctness = np.full((68, 68, 3), 128, dtype=np.uint8)
    agreement = masks["agreement"].squeeze().numpy()
    correct = masks["correct"].squeeze().numpy()
    wrong = masks["wrong"].squeeze().numpy()
    correctness[agreement] = np.array([60, 120, 220], dtype=np.uint8)
    correctness[correct] = np.array([20, 190, 70], dtype=np.uint8)
    correctness[wrong] = np.array([220, 40, 40], dtype=np.uint8)
    roll = build_spatial_roll_map(
        payload["ecst_effective_weight_68"].float().unsqueeze(0),
        [str(payload["dataset"])],
        [str(payload["stem"])],
    ).squeeze(0)
    with Image.open(payload["image_path"]) as image:
        rgb = np.asarray(
            image.convert("RGB").resize((68, 68), resample=Image.Resampling.BILINEAR)
        )
    panels = (
        ("RGB", rgb, None),
        ("GT (post-hoc)", _mask_panel(gt), "gray"),
        ("DABE-v2 Hard", _mask_panel(static), "gray"),
        ("Teacher Binary", _mask_panel(teacher), "gray"),
        ("Teacher add-FG", _mask_panel(masks["add_fg"]), "gray"),
        ("Teacher erase-FG", _mask_panel(masks["erase_fg"]), "gray"),
        ("Correction correctness", correctness, None),
        ("Full ECST weight", _heat_panel(payload["ecst_effective_weight_68"]), "viridis"),
        ("Spatial Roll weight", _heat_panel(roll), "viridis"),
        ("Teacher confidence", _heat_panel(payload["teacher_confidence_68"]), "viridis"),
        (
            "DINO margin",
            _heat_panel(torch.sigmoid(payload["dino_margin_68"].float() / 0.05)),
            "coolwarm",
        ),
        (
            "Coarse-final consistency",
            _heat_panel(
                1.0
                - (
                    payload["teacher_final_prob_68"].float()
                    - payload["teacher_coarse_prob_68"].float()
                ).abs()
            ),
            "viridis",
        ),
    )
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, (title, panel, cmap) in zip(axes.flat, panels):
        axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(
        f"{payload['dataset']} / {payload['stem']} / epoch {payload['epoch']}",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _heap_add(
    heaps: dict[str, list[tuple[float, str]]], category: str, score: float, path: Path
) -> None:
    if not math.isfinite(float(score)) or float(score) <= 0.0:
        return
    item = (float(score), str(path))
    heap = heaps[category]
    if len(heap) < 20:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _correction_group_summary(
    rows: Sequence[Mapping[str, Any]], *, epoch: int, dataset: str, direction: str
) -> dict[str, Any]:
    if direction == "all":
        correction = sum(int(row["correction_pixel_count"]) for row in rows)
        correct = sum(int(row["correct_correction_count"]) for row in rows)
        wrong = sum(int(row["wrong_correction_count"]) for row in rows)
    elif direction == "add_fg":
        correct = sum(int(row["add_fg_correct_count"]) for row in rows)
        wrong = sum(int(row["add_fg_wrong_count"]) for row in rows)
        correction = correct + wrong
    else:
        correct = sum(int(row["erase_fg_correct_count"]) for row in rows)
        wrong = sum(int(row["erase_fg_wrong_count"]) for row in rows)
        correction = correct + wrong
    static_errors = sum(int(row["static_error_count"]) for row in rows)
    return {
        "epoch": epoch,
        "dataset": dataset,
        "direction": direction,
        "image_count": len(rows),
        "correction_pixel_count": correction,
        "correct_correction_count": correct,
        "wrong_correction_count": wrong,
        "correction_precision": correct / correction if correction else None,
        "correction_recall_relative_to_static_errors": (
            correct / static_errors if static_errors else None
        ),
        "add_fg_correct_ratio": (
            sum(int(row["add_fg_correct_count"]) for row in rows)
            / max(1, sum(int(row["add_fg_count"]) for row in rows))
        ),
        "add_fg_wrong_ratio": (
            sum(int(row["add_fg_wrong_count"]) for row in rows)
            / max(1, sum(int(row["add_fg_count"]) for row in rows))
        ),
        "erase_fg_correct_ratio": (
            sum(int(row["erase_fg_correct_count"]) for row in rows)
            / max(1, sum(int(row["erase_fg_count"]) for row in rows))
        ),
        "erase_fg_wrong_ratio": (
            sum(int(row["erase_fg_wrong_count"]) for row in rows)
            / max(1, sum(int(row["erase_fg_count"]) for row in rows))
        ),
        "fp_corrected": sum(int(row["fp_corrected"]) for row in rows),
        "fp_introduced": sum(int(row["fp_introduced"]) for row in rows),
        "fn_corrected": sum(int(row["fn_corrected"]) for row in rows),
        "fn_introduced": sum(int(row["fn_introduced"]) for row in rows),
        "net_pixel_error_change": sum(int(row["net_pixel_error_change"]) for row in rows),
        "net_iou_change": _safe_mean([row["net_iou_change"] for row in rows]),
        "net_f1_change": _safe_mean([row["net_f1_change"] for row in rows]),
    }


def audit_training_gt_corrections(
    *,
    cfg: Any,
    output_root: Path,
    epochs: Sequence[int],
    allow_training_gt_posthoc: bool,
    batch_size: int = 16,
) -> dict[str, Any]:
    if not allow_training_gt_posthoc:
        raise RuntimeError(
            "Refusing to read training GT without --allow-training-gt-posthoc."
        )
    gt_map = _training_gt_map(cfg)
    per_sample_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    gt_control_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    image_bootstrap: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    retention_bootstrap: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    all_group_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    gt_cache: dict[tuple[str, str], torch.Tensor] = {}
    gt_cache_hits = 0
    gt_cache_misses = 0

    epoch_paths: dict[int, list[Path]] = {}
    for epoch in map(int, epochs):
        paths = _epoch_payloads(output_root, epoch)
        if not paths:
            raise FileNotFoundError(f"No payloads for GT audit epoch {epoch}.")
        epoch_paths[epoch] = paths
    progress = _ProgressReporter(
        "correction_gt_audit",
        sum(len(paths) for paths in epoch_paths.values()),
    )
    for epoch in map(int, epochs):
        paths = epoch_paths[epoch]
        by_dataset: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_dataset[path.parent.name].append(path)
        for dataset_name, group_paths in sorted(by_dataset.items()):
            if dataset_name not in {"TR-CAMO", "TR-COD10K"}:
                raise RuntimeError(f"Test/non-training dataset GT is forbidden: {dataset_name}")
            score_parts: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
            label_parts: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
            region_parts: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
                lambda: defaultdict(list)
            )
            group_sample_rows = []
            heaps: dict[str, list[tuple[float, str]]] = defaultdict(list)
            pixel_shard: dict[str, list[np.ndarray]] = defaultdict(list)
            for payload_path, payload in _payload_items(group_paths, batch_size):
                key = (dataset_name, str(payload["stem"]))
                if key not in gt_map:
                    raise RuntimeError(f"Training GT missing for {key}.")
                gt = gt_cache.get(key)
                if gt is None:
                    gt = _load_gt_68(gt_map[key])
                    gt_cache[key] = gt
                    gt_cache_misses += 1
                else:
                    gt_cache_hits += 1
                static = payload["static_target_68"].bool()
                teacher = payload["teacher_binary_68"].bool()
                masks = compute_correction_masks(static, teacher, gt)
                quality = compute_correction_quality(static, teacher, gt)
                candidates = _candidate_scores(payload)
                controls = _control_maps(payload)
                logits = payload["student_final_logit_68"].float()
                sample_row = {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "stem": str(payload["stem"]),
                    "sample_index": int(payload["sample_index"]),
                    "static_fg_area": float(static.float().mean().item()),
                    "teacher_fg_area": float(teacher.float().mean().item()),
                    "static_error_count": int(masks["static_error"].sum().item()),
                    "add_fg_count": int(masks["add_fg"].sum().item()),
                    "erase_fg_count": int(masks["erase_fg"].sum().item()),
                    **quality,
                    "training_gt_used_for_posthoc_audit": True,
                    "training_gt_used_for_training": False,
                    "test_gt_used": False,
                }
                group_sample_rows.append(sample_row)
                per_sample_rows.append(sample_row)
                directions = {
                    "all": masks["conflict"],
                    "add_fg": masks["add_fg"],
                    "erase_fg": masks["erase_fg"],
                }
                for candidate_name, score in candidates.items():
                    for direction, selection in directions.items():
                        score_parts[(candidate_name, direction)].append(score[selection].cpu())
                        label_parts[(candidate_name, direction)].append(
                            masks["correct"][selection].cpu()
                        )
                    selection = directions["all"] & torch.isfinite(score)
                    ranking = compute_score_ranking_metrics(
                        score[selection], masks["correct"][selection]
                    )
                    p20_rows = compute_precision_at_coverage(
                        score[selection], masks["correct"][selection], coverages=(0.20,)
                    )
                    p20 = p20_rows[0].get("precision") if p20_rows else None
                    image_bootstrap[(epoch, dataset_name, candidate_name, "all")].append(
                        {
                            "auroc": ranking.get("auroc"),
                            "auprc": ranking.get("auprc"),
                            "precision_at_20": p20,
                        }
                    )
                for control_name, weight in controls.items():
                    retention = compute_gradient_retention(
                        logits, teacher.float(), static.float(), gt.float(), weight
                    )
                    retention_rows.append(
                        {
                            "epoch": epoch,
                            "dataset": dataset_name,
                            "stem": str(payload["stem"]),
                            "control": control_name,
                            **retention,
                        }
                    )
                    retention_bootstrap[(epoch, dataset_name, control_name)].append(
                        dict(retention)
                    )
                    selection = masks["conflict"]
                    score_parts[(f"control::{control_name}", "all")].append(weight[selection])
                    label_parts[(f"control::{control_name}", "all")].append(
                        masks["correct"][selection]
                    )
                    gt_control_rows.append(
                        {
                            "analysis": "gt_selectivity_per_image",
                            "epoch": epoch,
                            "dataset": dataset_name,
                            "stem": str(payload["stem"]),
                            "variant": control_name,
                            "weight_mean": float(weight.mean().item()),
                            "weight_min": float(weight.min().item()),
                            "weight_max": float(weight.max().item()),
                            "spatial_autocorrelation": compute_map_spatial_autocorrelation(weight),
                            **retention,
                        }
                    )
                region_masks = {
                    "fg_core_conflict": payload["fg_core_68"].bool() & masks["conflict"],
                    "bg_core_conflict": payload["bg_core_68"].bool() & masks["conflict"],
                    "extent_teacher_fg": payload["extent_teacher_fg_68"].bool(),
                    "extent_teacher_bg": payload["extent_teacher_bg_68"].bool(),
                    "unknown": payload["unknown_68"].bool(),
                    "other": payload["other_68"].bool(),
                }
                ecst_weight = candidates["full_ecst_weight"]
                for region_name, region in region_masks.items():
                    region_conflict = region & masks["conflict"]
                    region_parts[region_name]["all"].append(region.reshape(-1))
                    region_parts[region_name]["conflict"].append(region_conflict.reshape(-1))
                    region_parts[region_name]["correct"].append(
                        masks["correct"][region_conflict].reshape(-1)
                    )
                    region_parts[region_name]["weight"].append(
                        ecst_weight[region_conflict].reshape(-1)
                    )
                    region_parts[region_name]["weight_all"].append(
                        ecst_weight[region].reshape(-1)
                    )
                full_weight = candidates["full_ecst_weight"]
                category_scores = {
                    "correct_add_fg_high": float((full_weight * masks["add_fg_correct"]).sum()),
                    "wrong_add_fg_high": float((full_weight * masks["add_fg_wrong"]).sum()),
                    "correct_erase_fg_high": float((full_weight * masks["erase_fg_correct"]).sum()),
                    "wrong_erase_fg_high": float((full_weight * masks["erase_fg_wrong"]).sum()),
                    "ecst_suppresses_wrong": float(((1.0 - full_weight) * masks["wrong"]).sum()),
                    "ecst_suppresses_correct": float(((1.0 - full_weight) * masks["correct"]).sum()),
                    "ecst_accepts_wrong": float((full_weight * masks["wrong"]).sum()),
                }
                for category, value in category_scores.items():
                    _heap_add(heaps, category, value, payload_path)
                conflict = masks["conflict"]
                pixel_shard["correct"].append(masks["correct"][conflict].numpy().astype(np.uint8))
                pixel_shard["direction_add_fg"].append(masks["add_fg"][conflict].numpy().astype(np.uint8))
                for candidate_name, score in candidates.items():
                    pixel_shard[f"score__{candidate_name}"].append(
                        score[conflict].float().numpy().astype(np.float16)
                    )
                progress.update(detail=f"epoch={epoch} dataset={dataset_name}")

            all_group_rows[(epoch, dataset_name)] = group_sample_rows
            for direction in ("all", "add_fg", "erase_fg"):
                direction_rows.append(
                    _correction_group_summary(
                        group_sample_rows,
                        epoch=epoch,
                        dataset=dataset_name,
                        direction=direction,
                    )
                )
            for (candidate_name, direction), parts in sorted(score_parts.items()):
                ranking = _ranking_row(
                    epoch=epoch,
                    dataset=dataset_name,
                    direction=direction,
                    candidate=candidate_name,
                    scores=parts,
                    labels=label_parts[(candidate_name, direction)],
                )
                candidate_rows.append(ranking)
                score = torch.cat([value.reshape(-1) for value in parts]) if parts else torch.empty(0)
                label = torch.cat(
                    [value.reshape(-1) for value in label_parts[(candidate_name, direction)]]
                ) if parts else torch.empty(0, dtype=torch.bool)
                for coverage in compute_precision_at_coverage(score, label):
                    coverage_rows.append(
                        {
                            "epoch": epoch,
                            "dataset": dataset_name,
                            "direction": direction,
                            "candidate": candidate_name,
                            **coverage,
                        }
                    )
            total_pixels = len(group_paths) * 68 * 68
            for region_name, values in sorted(region_parts.items()):
                region_count = sum(int(value.sum().item()) for value in values["all"])
                conflict_count = sum(int(value.sum().item()) for value in values["conflict"])
                correct = torch.cat(values["correct"]) if values["correct"] else torch.empty(0, dtype=torch.bool)
                weight = torch.cat(values["weight"]) if values["weight"] else torch.empty(0)
                weight_all = (
                    torch.cat(values["weight_all"])
                    if values["weight_all"]
                    else torch.empty(0)
                )
                ranking = compute_score_ranking_metrics(weight, correct)
                correct_weight = weight[correct] if weight.numel() else torch.empty(0)
                wrong_weight = weight[~correct] if weight.numel() else torch.empty(0)
                region_rows.append(
                    {
                        "epoch": epoch,
                        "dataset": dataset_name,
                        "region": region_name,
                        "pixel_ratio": region_count / max(1, total_pixels),
                        "correction_ratio": conflict_count / max(1, region_count),
                        "correct_correction_ratio": (
                            float(correct.float().mean().item()) if correct.numel() else None
                        ),
                        "mean_ecst_weight": (
                            float(weight_all.mean().item()) if weight_all.numel() else None
                        ),
                        "correct_correction_mean_weight": (
                            float(correct_weight.mean().item()) if correct_weight.numel() else None
                        ),
                        "wrong_correction_mean_weight": (
                            float(wrong_weight.mean().item()) if wrong_weight.numel() else None
                        ),
                        "selectivity_gap": (
                            float(correct_weight.mean().item() - wrong_weight.mean().item())
                            if correct_weight.numel() and wrong_weight.numel()
                            else None
                        ),
                        "auroc": ranking.get("auroc"),
                        "auprc": ranking.get("auprc"),
                        "ranking_status": ranking["status"],
                        "ranking_reason": ranking["reason"],
                    }
                )
            shard_root = output_root / "pixel_shards" / f"epoch_{epoch:03d}"
            shard_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                shard_root / f"{dataset_name}.npz",
                **{
                    name: np.concatenate(values) if values else np.empty(0)
                    for name, values in pixel_shard.items()
                },
            )
            for category, heap in heaps.items():
                for rank, (_, payload_value) in enumerate(sorted(heap, reverse=True), start=1):
                    payload_path = Path(payload_value)
                    payload = torch_load(payload_path, map_location="cpu")
                    output_path = (
                        output_root
                        / "visualizations"
                        / f"epoch_{epoch:03d}"
                        / dataset_name
                        / f"{category}_{rank:02d}_{payload['stem']}.png"
                    )
                    _render_case(
                        payload_path=payload_path,
                        gt_path=gt_map[(dataset_name, str(payload["stem"]))],
                        output_path=output_path,
                    )

    macro_keys = sorted(
        {
            (epoch, candidate, direction)
            for epoch, _, candidate, direction in image_bootstrap
        }
    )
    retention_macro_keys = sorted(
        {(epoch, control) for epoch, _, control in retention_bootstrap}
    )
    bootstrap_progress = _ProgressReporter(
        "correction_gt_bootstrap",
        3
        * (
            len(image_bootstrap)
            + len(retention_bootstrap)
            + len(macro_keys)
            + len(retention_macro_keys)
        ),
    )
    for key, records in sorted(image_bootstrap.items()):
        epoch, dataset_name, candidate, direction = key
        for metric in ("auroc", "auprc", "precision_at_20"):
            result = compute_bootstrap_ci(
                records,
                lambda selected, field=metric: _image_macro_metric(selected, field),
                repetitions=1000,
                seed=ECST_CAUSAL_CONTROL_SEED,
            )
            bootstrap_rows.append(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "candidate": candidate,
                    "direction": direction,
                    "metric": metric,
                    **result,
                }
            )
            bootstrap_progress.update(
                detail=f"epoch={epoch} dataset={dataset_name} candidate={candidate}"
            )
    for key, records in sorted(retention_bootstrap.items()):
        epoch, dataset_name, control = key
        for metric in (
            "correct_gradient_retention",
            "wrong_gradient_retention",
            "selectivity_gap",
        ):
            result = compute_bootstrap_ci(
                records,
                lambda selected, field=metric: _image_macro_metric(selected, field),
                repetitions=1000,
                seed=ECST_CAUSAL_CONTROL_SEED,
            )
            bootstrap_rows.append(
                {
                    "epoch": epoch,
                    "dataset": dataset_name,
                    "candidate": f"control::{control}",
                    "direction": "all",
                    "metric": metric,
                    **result,
                }
            )
            bootstrap_progress.update(
                detail=f"epoch={epoch} dataset={dataset_name} control={control}"
            )
    for epoch, candidate, direction in macro_keys:
        records = []
        for dataset_name in ("TR-CAMO", "TR-COD10K"):
            records.extend(image_bootstrap.get((epoch, dataset_name, candidate, direction), []))
        for metric in ("auroc", "auprc", "precision_at_20"):
            result = compute_bootstrap_ci(
                records,
                lambda selected, field=metric: _image_macro_metric(selected, field),
                repetitions=1000,
                seed=ECST_CAUSAL_CONTROL_SEED,
            )
            bootstrap_rows.append(
                {
                    "epoch": epoch,
                    "dataset": "macro",
                    "candidate": candidate,
                    "direction": direction,
                    "metric": metric,
                    **result,
                }
            )
            bootstrap_progress.update(
                detail=f"epoch={epoch} dataset=macro candidate={candidate}"
            )
    for epoch, control in retention_macro_keys:
        records = []
        for dataset_name in ("TR-CAMO", "TR-COD10K"):
            records.extend(retention_bootstrap.get((epoch, dataset_name, control), []))
        for metric in (
            "correct_gradient_retention",
            "wrong_gradient_retention",
            "selectivity_gap",
        ):
            result = compute_bootstrap_ci(
                records,
                lambda selected, field=metric: _image_macro_metric(selected, field),
                repetitions=1000,
                seed=ECST_CAUSAL_CONTROL_SEED,
            )
            bootstrap_rows.append(
                {
                    "epoch": epoch,
                    "dataset": "macro",
                    "candidate": f"control::{control}",
                    "direction": "all",
                    "metric": metric,
                    **result,
                }
            )
            bootstrap_progress.update(
                detail=f"epoch={epoch} dataset=macro control={control}"
            )

    retention_summary = _aggregate_numeric_rows(
        retention_rows, ("epoch", "dataset", "control")
    )
    existing_map_rows = _read_csv(output_root / "map_control_summary.csv")
    map_gt_summary = _aggregate_numeric_rows(
        gt_control_rows, ("analysis", "epoch", "dataset", "variant")
    )
    _write_csv(output_root / "per_sample_summary.csv", per_sample_rows)
    _write_csv(output_root / "correction_direction_summary.csv", direction_rows)
    _write_csv(output_root / "candidate_score_summary.csv", candidate_rows)
    _write_csv(output_root / "precision_at_coverage.csv", coverage_rows)
    _write_csv(output_root / "region_policy_summary.csv", region_rows)
    _write_csv(output_root / "gradient_retention_summary.csv", retention_summary)
    _write_csv(output_root / "bootstrap_ci.csv", bootstrap_rows)
    _write_csv(output_root / "map_control_summary.csv", [*existing_map_rows, *map_gt_summary])

    per_epoch_rows = _aggregate_numeric_rows(per_sample_rows, ("epoch",))
    per_dataset_rows = _aggregate_numeric_rows(per_sample_rows, ("dataset",))
    for row in per_epoch_rows:
        row["training_gt_used_for_posthoc_audit"] = True
        row["training_gt_used_for_training"] = False
        row["test_gt_used"] = False
    for row in per_dataset_rows:
        row["training_gt_used_for_posthoc_audit"] = True
        row["training_gt_used_for_training"] = False
        row["test_gt_used"] = False
    _write_csv(output_root / "per_epoch_summary.csv", per_epoch_rows)
    _write_csv(output_root / "per_dataset_summary.csv", per_dataset_rows)
    summary = {
        "schema": "full_ecst_training_gt_posthoc_audit_v1",
        "status": "PASS",
        "epochs": list(map(int, epochs)),
        "sample_rows": len(per_sample_rows),
        "payload_prefetch_batches": PAYLOAD_PREFETCH_BATCHES,
        "progress_reporting": True,
        "gt_cache_entries": len(gt_cache),
        "gt_cache_hits": gt_cache_hits,
        "gt_cache_misses": gt_cache_misses,
        "training_gt_used_for_posthoc_audit": True,
        "training_gt_used_for_training": False,
        "test_gt_used": False,
        "pixel_shards": str(output_root / "pixel_shards"),
        "visualizations": str(output_root / "visualizations"),
    }
    (output_root / "correction_gt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _float_value(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_field(
    rows: Sequence[Mapping[str, Any]], key: str, predicate=lambda row: True
) -> float | None:
    return _safe_mean(
        [_float_value(row, key) for row in rows if predicate(row)]
    )


def _candidate_metric(
    rows: Sequence[Mapping[str, Any]], candidate: str, metric: str
) -> float | None:
    return _mean_field(
        rows,
        metric,
        lambda row: row.get("candidate") == candidate
        and row.get("direction") == "all"
        and row.get("dataset") in {"TR-CAMO", "TR-COD10K"},
    )


def build_report(
    *,
    output_root: Path,
    epochs: Sequence[int],
    max_samples: int,
) -> dict[str, Any]:
    required = (
        "checkpoint_inventory.json",
        "export_summary.json",
        "controls_summary.json",
        "correction_gt_summary.json",
        "per_sample_summary.csv",
        "candidate_score_summary.csv",
        "gradient_retention_summary.csv",
        "map_control_summary.csv",
        "bootstrap_ci.csv",
    )
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Report stage requires completed prior stages; missing=" + str(missing)
        )
    inventory = json.loads(
        (output_root / "checkpoint_inventory.json").read_text(encoding="utf-8")
    )
    export = json.loads((output_root / "export_summary.json").read_text(encoding="utf-8"))
    controls = json.loads((output_root / "controls_summary.json").read_text(encoding="utf-8"))
    candidate_rows = _read_csv(output_root / "candidate_score_summary.csv")
    retention_rows = _read_csv(output_root / "gradient_retention_summary.csv")
    direction_rows = _read_csv(output_root / "correction_direction_summary.csv")
    region_policy_rows = _read_csv(output_root / "region_policy_summary.csv")
    per_sample_rows = _read_csv(output_root / "per_sample_summary.csv")
    requested_epochs = list(map(int, epochs))
    counts = defaultdict(int)
    for row in per_sample_rows:
        counts[int(row["epoch"])] += 1
    full_epoch_counts = all(
        counts[epoch] == EXPECTED_TRAIN_SAMPLES for epoch in requested_epochs
    )
    audit_complete = bool(
        int(max_samples) < 0
        and set(requested_epochs) == set(PRIMARY_EPOCHS)
        and full_epoch_counts
        and bool(export.get("full_manifest"))
    )

    ecst_auroc = _candidate_metric(candidate_rows, "full_ecst_weight", "auroc")
    ecst_auprc = _candidate_metric(candidate_rows, "full_ecst_weight", "auprc")
    ecst_random = _candidate_metric(
        candidate_rows, "full_ecst_weight", "positive_ratio"
    )
    roll_auroc = _candidate_metric(
        candidate_rows, "control::spatial_roll", "auroc"
    )
    pixel_auroc = _candidate_metric(
        candidate_rows, "control::pixel_permutation", "auroc"
    )
    original_gap = _mean_field(
        retention_rows,
        "selectivity_gap",
        lambda row: row.get("control") == "original_ecst",
    )
    roll_gap = _mean_field(
        retention_rows,
        "selectivity_gap",
        lambda row: row.get("control") == "spatial_roll",
    )
    correct_retention = _mean_field(
        retention_rows,
        "correct_gradient_retention",
        lambda row: row.get("control") == "original_ecst",
    )
    wrong_retention = _mean_field(
        retention_rows,
        "wrong_gradient_retention",
        lambda row: row.get("control") == "original_ecst",
    )
    dataset_aurocs = {
        dataset: _mean_field(
            candidate_rows,
            "auroc",
            lambda row, d=dataset: row.get("candidate") == "full_ecst_weight"
            and row.get("direction") == "all"
            and row.get("dataset") == d,
        )
        for dataset in ("TR-CAMO", "TR-COD10K")
    }
    conditions = {
        "ecst_auroc_gt_055": ecst_auroc is not None and ecst_auroc > 0.55,
        "ecst_auprc_above_random_by_002": (
            ecst_auprc is not None
            and ecst_random is not None
            and ecst_auprc > ecst_random + 0.02
        ),
        "selectivity_gap_gt_003": original_gap is not None and original_gap > 0.03,
        "original_better_than_spatial_roll": (
            ecst_auroc is not None
            and roll_auroc is not None
            and ecst_auroc > roll_auroc + 0.01
        ),
        "correct_retention_gt_wrong": (
            correct_retention is not None
            and wrong_retention is not None
            and correct_retention > wrong_retention
        ),
        "cross_dataset_same_direction": all(
            value is not None and value > 0.5 for value in dataset_aurocs.values()
        ),
    }
    full_ecst_has_spatial_selectivity = bool(
        audit_complete and sum(conditions.values()) >= 4
    )

    signal_names = sorted(
        {
            str(row.get("candidate"))
            for row in candidate_rows
            if not str(row.get("candidate", "")).startswith("control::")
        }
    )
    signal_metrics = []
    for candidate in signal_names:
        auroc = _candidate_metric(candidate_rows, candidate, "auroc")
        auprc = _candidate_metric(candidate_rows, candidate, "auprc")
        if auroc is not None:
            signal_metrics.append((auroc, auprc if auprc is not None else -1.0, candidate))
    signal_metrics.sort(reverse=True)
    best = signal_metrics[0] if signal_metrics else (None, None, None)
    signal_table_lines = ["| candidate | AUROC | AUPRC |", "|---|---:|---:|"]
    for auroc, auprc, candidate in signal_metrics:
        signal_table_lines.append(f"| {candidate} | {auroc:.6f} | {auprc:.6f} |")
    region_names = (
        "fg_core_conflict",
        "bg_core_conflict",
        "extent_teacher_fg",
        "extent_teacher_bg",
        "unknown",
        "other",
    )
    region_table_lines = [
        "| region | correction precision | weight gap | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in region_names:
        values = {
            metric: _mean_field(
                region_policy_rows,
                metric,
                lambda row, name=region: row.get("region") == name,
            )
            for metric in (
                "correct_correction_ratio",
                "selectivity_gap",
                "auroc",
                "auprc",
            )
        }
        region_table_lines.append(
            "| {} | {} | {} | {} | {} |".format(
                region,
                values["correct_correction_ratio"],
                values["selectivity_gap"],
                values["auroc"],
                values["auprc"],
            )
        )
    direction_table_lines = [
        "| dataset | direction | correction precision | net IoU change | net F1 change |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name in ("TR-CAMO", "TR-COD10K"):
        for direction in ("add_fg", "erase_fg"):
            selected = [
                row
                for row in direction_rows
                if row.get("dataset") == dataset_name
                and row.get("direction") == direction
            ]
            direction_table_lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    dataset_name,
                    direction,
                    _mean_field(selected, "correction_precision"),
                    _mean_field(selected, "net_iou_change"),
                    _mean_field(selected, "net_f1_change"),
                )
            )
    flip_auroc = _candidate_metric(candidate_rows, "hflip_consistency", "auroc")
    flip_auprc = _candidate_metric(candidate_rows, "hflip_consistency", "auprc")
    teacher_conf_auprc = _candidate_metric(
        candidate_rows, "teacher_confidence", "auprc"
    )
    cross_view_directions = {
        direction: _mean_field(
            candidate_rows,
            "auroc",
            lambda row, d=direction: row.get("candidate") == "hflip_consistency"
            and row.get("direction") == d,
        )
        for direction in ("add_fg", "erase_fg")
    }
    authorize_cross_view = bool(
        audit_complete
        and flip_auroc is not None
        and flip_auroc >= 0.58
        and flip_auprc is not None
        and teacher_conf_auprc is not None
        and flip_auprc > teacher_conf_auprc
        and all(value is not None and value > 0.5 for value in cross_view_directions.values())
    )

    if not audit_complete:
        recommendation_text = "diagnostic_only_no_mechanism_conclusion"
        reasons = [
            "Full admission requires exactly 4040 training images for all registered epochs.",
            f"observed_counts={dict(sorted(counts.items()))}",
        ]
    elif full_ecst_has_spatial_selectivity:
        recommendation_text = "retain_spatial_selectivity_and_seek_simpler_mechanism"
        reasons = [
            f"{sum(conditions.values())}/6 preregistered spatial-selectivity conditions passed.",
            "Run Spatial Roll stop20 only as the registered causal training control.",
        ]
    elif not signal_metrics or max(value[0] for value in signal_metrics) < 0.55:
        recommendation_text = "stop_second_module_search"
        reasons = [
            "Full ECST spatial selectivity did not pass the majority rule.",
            "No preregistered unlabeled candidate reached mean AUROC 0.55.",
        ]
    else:
        recommendation_text = "prefer_simple_global_handover_control"
        reasons = [
            "Spatial selectivity did not pass the majority rule.",
            "GMG stop20 may test whether optimization magnitude explains the gain.",
        ]

    checkpoint_memory_exact = bool(inventory.get("checkpoint_memory_exact"))
    replayed = export.get("memory", {}).get("replayed_epochs", [])
    recommendation = {
        "audit_complete": audit_complete,
        "checkpoint_memory_exact": checkpoint_memory_exact,
        "memory_replay_exact_protocol_used": bool(replayed),
        "normalized_bce_constant_map_equals_plain": bool(
            controls.get("normalized_bce_constant_map_equals_plain")
        ),
        "full_ecst_has_spatial_selectivity": full_ecst_has_spatial_selectivity,
        "full_ecst_better_than_spatial_roll": (
            None
            if not audit_complete or ecst_auroc is None or roll_auroc is None
            else ecst_auroc > roll_auroc + 0.01
        ),
        "full_ecst_better_than_gradient_matched_global": None,
        "full_ecst_vs_gmg_note": (
            "Shadow gradients cannot establish final trained performance; use the registered stop20 GMG control."
        ),
        "best_candidate_signal": best[2],
        "best_candidate_auroc": best[0],
        "best_candidate_auprc": best[1],
        "cross_dataset_consistent": conditions["cross_dataset_same_direction"],
        "authorize_gradmatch_global_training": bool(
            audit_complete and not full_ecst_has_spatial_selectivity
        ),
        "authorize_spatial_roll_training": bool(
            audit_complete and full_ecst_has_spatial_selectivity
        ),
        "authorize_cross_view_mechanism": authorize_cross_view,
        "recommendation": recommendation_text,
        "reasons": reasons,
        "spatial_selectivity_conditions": conditions,
        "summary_metrics": {
            "ecst_auroc": ecst_auroc,
            "ecst_auprc": ecst_auprc,
            "conflict_correct_random_baseline": ecst_random,
            "spatial_roll_auroc": roll_auroc,
            "pixel_permutation_auroc": pixel_auroc,
            "original_selectivity_gap": original_gap,
            "spatial_roll_selectivity_gap": roll_gap,
            "correct_gradient_retention": correct_retention,
            "wrong_gradient_retention": wrong_retention,
            "dataset_aurocs": dataset_aurocs,
        },
    }
    (output_root / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    add_rows = [row for row in direction_rows if row.get("direction") == "add_fg"]
    erase_rows = [row for row in direction_rows if row.get("direction") == "erase_fg"]
    report = f"""# Full ECST 因果审计报告

## 审计状态

- 完整准入审计：`{audit_complete}`
- 目标轮次：`{requested_epochs}`
- 每轮样本数：`{dict(sorted(counts.items()))}`
- checkpoint 内原生 ECST memory：`{checkpoint_memory_exact}`
- 使用只读顺序重放恢复 history：`{bool(replayed)}`
- 训练 GT 仅用于独立事后审计：`True`
- 测试 GT 使用：`False`

若完整准入审计为 False，本报告只属于 diagnostic-only，不给出机制有效性结论。

## 数学协议

当前 Teacher loss 的权威实现为整个 batch 上的
`sum(W * BCE_none) / (sum(W) + 1e-6)`。因此，全局恒定权重图与普通 mean BCE
严格等价（数值最大误差 `{controls.get('global_constant_plain_max_abs_error')}`）。
任务书中的“逐图均值恒定图”在 batch 内各图均值不同时会重新加权图像，故仅保留为
shadow 变体，不将其误报为严格等价控制。

## 30.1 Full ECST 是否具有空间选择性

- ECST correction AUROC：`{ecst_auroc}`
- ECST correction AUPRC：`{ecst_auprc}`
- Spatial Roll AUROC：`{roll_auroc}`
- Pixel Permutation AUROC：`{pixel_auroc}`
- 正确/错误梯度保留：`{correct_retention}` / `{wrong_retention}`
- 六项预注册条件：`{json.dumps(conditions, ensure_ascii=False)}`
- 准入判定：`{full_ecst_has_spatial_selectivity}`

## 30.2 是否只是全局 Teacher 梯度幅度

GMG shadow 已按实际 batch-global 归一化公式匹配 L1 梯度总幅度，误差上限为
`{controls.get('gmg_l1_gradient_max_abs_error')}`。但 shadow 不能推断训练后的验证性能，
因此 `full_ecst_better_than_gradient_matched_global` 保持 `null`，只能由注册的 stop20
训练控制回答。

## 30.3 哪种 Teacher 校正更危险

- 新增前景平均正确率：`{_mean_field(add_rows, 'correction_precision')}`
- 删除前景平均正确率：`{_mean_field(erase_rows, 'correction_precision')}`

结果同时在 `correction_direction_summary.csv` 中按 epoch/dataset 分开给出，不预设任何方向更危险。

{chr(10).join(direction_table_lines)}

## 30.4 ECST 哪一分支有效

`region_policy_summary.csv` 独立列出 core conflict、extent Teacher-FG、extent Teacher-BG、
unknown 与 other 的正确率、权重差、AUROC 和 AUPRC；History 与 DINO margin 在
`candidate_score_summary.csv` 中独立审计。

{chr(10).join(region_table_lines)}

## 30.5 无标签候选信号

- 最佳候选：`{best[2]}`
- 平均 AUROC/AUPRC：`{best[0]}` / `{best[1]}`
- hflip 全量候选 AUROC：`{flip_auroc}`
- Cross-view 准入：`{authorize_cross_view}`

{chr(10).join(signal_table_lines)}

## 30.6 EAOGP 失败诊断接口

DINO 邻域一致性、分支一致性、DABE 锚定反向量以及可选 hflip 一致性均在相同冲突
像素集合上比较。是否能解释 EAOGP 的面积扩张与错误校正，必须以完整审计 CSV 为准；
小样本结果不得外推。

## 建议

`{recommendation_text}`

原因：{json.dumps(reasons, ensure_ascii=False)}
"""
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    return recommendation


def main() -> int:
    args = parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    config_path = _inside_cth(args.config)
    checkpoint_root = _inside_cth(args.checkpoint_root)
    output_root = _inside_cth(args.output_root)
    epochs = _epochs(args.epochs)
    cfg = load_config(config_path)
    _validate_export_config(cfg)
    set_seed(int(getattr(cfg, "SEED", ECST_CAUSAL_CONTROL_SEED)))
    device = _device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    protocol = _protocol(
        cfg=cfg,
        config_path=config_path,
        epochs=epochs,
        max_samples=int(args.max_samples),
    )
    if protocol_path.exists() and not args.overwrite:
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        comparable = ("config_fingerprint", "epochs", "max_samples")
        conflict = {
            name: (existing.get(name), protocol.get(name))
            for name in comparable
            if existing.get(name) != protocol.get(name)
        }
        if conflict:
            raise FileExistsError(
                f"Audit root contains a conflicting protocol: {conflict}"
            )
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results: dict[str, Any] = {
        "schema": "full_ecst_causal_audit_run_v1",
        "stage": args.stage,
        "output_root": str(output_root),
        "epochs": epochs,
    }
    stages = (
        ("export", "controls", "correction_gt_audit", "report")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "export":
            results[stage] = export_payloads(
                cfg=cfg,
                config_path=config_path,
                checkpoint_root=checkpoint_root,
                output_root=output_root,
                epochs=epochs,
                max_samples=int(args.max_samples),
                batch_size=int(args.batch_size),
                device=device,
                overwrite=bool(args.overwrite),
            )
        elif stage == "controls":
            results[stage] = audit_shadow_controls(
                output_root=output_root,
                epochs=epochs,
                batch_size=int(args.batch_size),
                device=device,
            )
        elif stage == "correction_gt_audit":
            results[stage] = audit_training_gt_corrections(
                cfg=cfg,
                output_root=output_root,
                epochs=epochs,
                allow_training_gt_posthoc=bool(args.allow_training_gt_posthoc),
                batch_size=int(args.batch_size),
            )
        elif stage == "report":
            results[stage] = build_report(
                output_root=output_root,
                epochs=epochs,
                max_samples=int(args.max_samples),
            )
    print(json.dumps(make_jsonable(results), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
