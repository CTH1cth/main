"""Cache helpers for DABE Bridge, Clean and continuous-recovery targets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from common.dabe_clean import (
    DABE_BRIDGE_VERSION,
    DABE_CLEAN_CONTREC_VERSION,
    DABE_CLEAN_VERSION,
    build_background_evidence,
    build_bridge_target,
    build_clean_targets,
    build_continuous_recoverability,
)
from common.ecst_clean import build_clean_semantic_fg_tendency
from common.utils import read_jsonl, torch_load, write_jsonl


SOURCE_SHAPES = {"37": (1, 37, 37), "68": (1, 68, 68)}


def _source_tensor(payload, key, shape):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Source DABE-PU cache is missing tensor {key!r}.")
    value = value.detach().cpu().float()
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(
            f"Source tensor {key} shape mismatch: {list(value.shape)} != {list(shape)}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"Source tensor {key} contains NaN/Inf.")
    value_min = float(value.min().item())
    value_max = float(value.max().item())
    if value_min < -1e-6 or value_max > 1.0 + 1e-6:
        raise RuntimeError(
            f"Source tensor {key} is outside [0,1]: {value_min:.8f}/{value_max:.8f}."
        )
    return value.clamp(0.0, 1.0).detach()


def load_pu_source(row):
    path = Path(row["cache_path"])
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE-PU source cache must be a dict: {path}")
    dataset = str(row["dataset"])
    stem = str(row["stem"])
    if str(payload.get("dataset")) != dataset or str(payload.get("stem")) != stem:
        raise RuntimeError(
            f"DABE-PU source identity mismatch: manifest={dataset}/{stem}, "
            f"payload={payload.get('dataset')}/{payload.get('stem')} | {path}"
        )
    return payload


def build_bridge_payload(row):
    source = load_pu_source(row)
    target_37 = _source_tensor(source, "target_soft_37", SOURCE_SHAPES["37"])
    target_68 = _source_tensor(source, "target_soft_68", SOURCE_SHAPES["68"])
    weight_37 = _source_tensor(source, "weight_map_37", SOURCE_SHAPES["37"])
    weight_68 = _source_tensor(source, "weight_map_68", SOURCE_SHAPES["68"])
    p_base_37 = _source_tensor(source, "p_base_37", SOURCE_SHAPES["37"])
    return {
        "bridge_target_37": build_bridge_target(target_37, weight_37),
        "bridge_target_68": build_bridge_target(target_68, weight_68),
        "source_target_soft_37": target_37,
        "source_target_soft_68": target_68,
        "source_weight_map_37": weight_37,
        "source_weight_map_68": weight_68,
        "source_p_base_37": p_base_37,
        "dataset": str(row["dataset"]),
        "stem": str(row["stem"]),
        "backbone_key": str(source.get("backbone_key", "dinov1-s8")),
        "version": DABE_BRIDGE_VERSION,
        "source_version": str(source.get("dabe_version", "pu_v11")),
    }


def build_clean_payload(row):
    source = load_pu_source(row)
    foreground_37 = _source_tensor(source, "p_base_37", SOURCE_SHAPES["37"])
    foreground_68 = _source_tensor(source, "p_base_68", SOURCE_SHAPES["68"])
    bc_map_37 = _source_tensor(source, "bc_map_37", SOURCE_SHAPES["37"])
    residual_37 = _source_tensor(source, "residual_37", SOURCE_SHAPES["37"])
    background_37 = build_background_evidence(bc_map_37, residual_37)
    background_68 = F.interpolate(
        background_37.unsqueeze(0),
        size=SOURCE_SHAPES["68"][-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).detach()
    clean_37 = build_clean_targets(foreground_37, background_37)
    clean_68 = build_clean_targets(foreground_68, background_68)
    return {
        "foreground_evidence_37": clean_37["foreground_evidence"],
        "foreground_evidence_68": clean_68["foreground_evidence"],
        "background_evidence_37": clean_37["background_evidence"],
        "background_evidence_68": clean_68["background_evidence"],
        "target_dp_37": clean_37["target_dp"],
        "target_dp_68": clean_68["target_dp"],
        "target_diff_37": clean_37["target_diff"],
        "target_diff_68": clean_68["target_diff"],
        "source_p_base_37": foreground_37,
        "source_p_base_68": foreground_68,
        "source_bc_map_37": bc_map_37,
        "source_residual_37": residual_37,
        "dataset": str(row["dataset"]),
        "stem": str(row["stem"]),
        "backbone_key": str(source.get("backbone_key", "dinov1-s8")),
        "version": DABE_CLEAN_VERSION,
        "source_version": str(source.get("dabe_version", "pu_v11")),
    }


def _load_feature_source(row, expected_dataset, expected_stem):
    feature_path = Path(row.get("feature_cache_path", ""))
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"DABE continuous-recovery feature cache is missing: {feature_path}"
        )
    payload = torch_load(feature_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature cache must be a dict: {feature_path}")
    if (
        str(payload.get("dataset")) != str(expected_dataset)
        or str(payload.get("stem")) != str(expected_stem)
    ):
        raise RuntimeError(
            "DABE continuous-recovery feature identity mismatch: "
            f"{feature_path}"
        )
    feature = payload.get("tensor")
    if not torch.is_tensor(feature):
        raise RuntimeError(f"Feature cache is missing tensor: {feature_path}")
    feature = feature.detach().cpu().float()
    if tuple(feature.shape) != (384, 37, 37):
        raise RuntimeError(
            f"Feature shape mismatch: {list(feature.shape)} != [384,37,37] | "
            f"{feature_path}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError(f"Feature contains NaN/Inf: {feature_path}")
    return feature


def build_contrec_payload(row):
    """Build one GT-free v2 cache from existing DABE and DINO outputs."""

    source = load_pu_source(row)
    dataset = str(row["dataset"])
    stem = str(row["stem"])
    p_rw_37 = _source_tensor(source, "p_rw_37", SOURCE_SHAPES["37"])
    evidence_gate_37 = _source_tensor(
        source, "evidence_37", SOURCE_SHAPES["37"]
    )
    foreground_37 = _source_tensor(source, "p_base_37", SOURCE_SHAPES["37"])
    foreground_68 = _source_tensor(source, "p_base_68", SOURCE_SHAPES["68"])
    bc_map_37 = _source_tensor(source, "bc_map_37", SOURCE_SHAPES["37"])
    residual_37 = _source_tensor(source, "residual_37", SOURCE_SHAPES["37"])
    background_37 = build_background_evidence(bc_map_37, residual_37)
    background_68 = F.interpolate(
        background_37.unsqueeze(0),
        size=SOURCE_SHAPES["68"][-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).detach()
    clean_37 = build_clean_targets(foreground_37, background_37)
    clean_68 = build_clean_targets(foreground_68, background_68)

    feature = _load_feature_source(row, dataset, stem)
    semantic_37_batched, _, _ = build_clean_semantic_fg_tendency(
        feature.unsqueeze(0),
        foreground_37.unsqueeze(0),
        background_37.unsqueeze(0),
        margin_tau=float(row.get("margin_tau", 0.05)),
        output_size=(68, 68),
    )
    semantic_37 = semantic_37_batched.squeeze(0).detach().cpu().float()
    recovery = build_continuous_recoverability(
        p_rw_37=p_rw_37,
        evidence_gate_37=evidence_gate_37,
        foreground_evidence_37=foreground_37,
        background_evidence_37=background_37,
        semantic_fg_tendency_37=semantic_37,
        output_size=(68, 68),
    )
    return {
        "target_dp_37": clean_37["target_dp"],
        "target_dp_68": clean_68["target_dp"],
        "foreground_evidence_37": foreground_37,
        "foreground_evidence_68": foreground_68,
        "background_evidence_37": background_37,
        "background_evidence_68": background_68,
        "p_rw_37": p_rw_37,
        "evidence_gate_37": evidence_gate_37,
        "semantic_fg_tendency_37": semantic_37,
        "latent_rw_37": recovery["latent_rw_37"],
        "recoverability_37": recovery["recoverability_37"],
        "recoverability_68": recovery["recoverability_68"],
        "foreground_reconstruction_error": float(
            recovery["foreground_reconstruction_error"]
        ),
        "dataset": dataset,
        "stem": stem,
        "backbone_key": str(source.get("backbone_key", "dinov1-s8")),
        "version": DABE_CLEAN_CONTREC_VERSION,
        "source_version": str(source.get("dabe_version", "pu_v11")),
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "hard_ring_generated": False,
        "static_weight_map_written": False,
    }


def cache_manifest_row(payload, cache_path):
    if payload["version"] == DABE_BRIDGE_VERSION:
        mode = "bridge"
    elif payload["version"] == DABE_CLEAN_CONTREC_VERSION:
        mode = "continuous_recoverability"
    else:
        mode = "clean"
    return {
        "dataset": payload["dataset"],
        "stem": payload["stem"],
        "cache_path": str(Path(cache_path).resolve()),
        "backbone_key": payload["backbone_key"],
        "version": payload["version"],
        "source_version": payload["source_version"],
        "cache_kind": mode,
        "shape_37": [1, 37, 37],
        "shape_68": [1, 68, 68],
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "hard_ring_generated": False,
        "static_weight_map_written": False,
    }


def attach_feature_cache_rows(rows, feature_manifest, margin_tau=0.05):
    feature_manifest = Path(feature_manifest)
    feature_rows = read_jsonl(feature_manifest)
    feature_map = {}
    for feature_row in feature_rows:
        key = (str(feature_row.get("dataset")), str(feature_row.get("stem")))
        if None in key or "None" in key:
            raise RuntimeError(f"Bad feature manifest row: {feature_row}")
        if key in feature_map:
            raise RuntimeError(f"Duplicate feature manifest key: {key}")
        feature_map[key] = feature_row
    attached = []
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if key not in feature_map:
            raise RuntimeError(
                f"Feature manifest is missing DABE source sample: {key}"
            )
        augmented = dict(row)
        augmented["feature_cache_path"] = feature_map[key]["cache_path"]
        augmented["margin_tau"] = float(margin_tau)
        attached.append(augmented)
    return attached


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def select_source_rows(input_root, datasets, max_samples):
    manifest = Path(input_root) / "manifest_train.jsonl"
    rows = read_jsonl(manifest)
    selected_datasets = [
        item.strip() for item in str(datasets).split(",") if item.strip()
    ]
    if not selected_datasets:
        raise ValueError("--datasets must contain at least one dataset name.")
    allowed = set(selected_datasets)
    selected = [row for row in rows if str(row.get("dataset")) in allowed]
    found = {str(row.get("dataset")) for row in selected}
    missing = sorted(allowed - found)
    if missing:
        raise RuntimeError(f"Source manifest has no rows for datasets: {missing}.")
    seen = set()
    for row in selected:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if key in seen:
            raise RuntimeError(f"Duplicate source manifest key: {key}.")
        seen.add(key)
    max_samples = int(max_samples)
    if max_samples == 0 or max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    if max_samples > 0:
        selected = selected[:max_samples]
    if not selected:
        raise RuntimeError("No source rows selected.")
    return selected


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def generate_target_cache(rows, output_root, builder, workers=0, overwrite=False):
    output_root = Path(output_root)
    manifest_path = output_root / "manifest_train.jsonl"
    existing_files = []
    if output_root.exists():
        existing_files = [path for path in output_root.rglob("*") if path.is_file()]
    if existing_files and not bool(overwrite):
        raise FileExistsError(
            f"Output directory is non-empty and overwrite=false: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    def process(row):
        payload = builder(row)
        cache_path = output_root / payload["dataset"] / f"{payload['stem']}.pt"
        if cache_path.exists() and not bool(overwrite):
            raise FileExistsError(f"Refusing to overwrite cache: {cache_path}")
        _atomic_torch_save(payload, cache_path)
        return cache_manifest_row(payload, cache_path)

    workers = int(workers)
    if workers < 0:
        raise ValueError("--workers must be >= 0.")
    if workers <= 1:
        manifest_rows = [process(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            manifest_rows = list(executor.map(process, rows))
    if manifest_path.exists() and not bool(overwrite):
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    write_jsonl(manifest_path, manifest_rows)
    return manifest_path, manifest_rows
