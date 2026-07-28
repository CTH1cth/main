"""Cache I/O for pure-offline DABE v3 consolidation targets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from common.dabe_clean_offline import (
    DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT,
    DABE_CLEAN_OFFLINE_MODES,
    DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
    build_offline_consolidation,
)
from common.utils import read_jsonl, torch_load, write_jsonl


SOURCE_VERSION = "dabe_clean_v1"
SEMANTIC_SOURCE_VERSION = "dabe_clean_v2_contrec"
SHAPE_37 = (1, 37, 37)


def _tensor(payload, key, path):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Offline source is missing tensor {key!r}: {path}")
    value = value.detach().cpu().float().contiguous()
    if tuple(value.shape) != SHAPE_37:
        raise RuntimeError(
            f"Offline source {key} shape mismatch: {list(value.shape)} != "
            f"{list(SHAPE_37)} | {path}"
        )
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"Offline source {key} must be detached and finite: {path}")
    if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"Offline source {key} is outside [0,1]: {path}")
    return value.clamp(0.0, 1.0).detach()


def _manifest_map(root, label):
    root = Path(root)
    manifest = root / "manifest_train.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{label} manifest is missing: {manifest}")
    mapping = {}
    rows = read_jsonl(manifest)
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if "None" in key:
            raise RuntimeError(f"Bad {label} manifest row: {row}")
        if key in mapping:
            raise RuntimeError(f"Duplicate {label} manifest key: {key}")
        mapping[key] = row
    return manifest, mapping, rows


def select_joined_rows(source_root, semantic_root, datasets, max_samples=-1):
    source_manifest, source_map, source_rows = _manifest_map(source_root, "A1 source")
    semantic_manifest, semantic_map, _ = _manifest_map(
        semantic_root, "semantic source"
    )
    selected_names = [item.strip() for item in str(datasets).split(",") if item.strip()]
    if not selected_names:
        raise ValueError("--datasets must contain at least one dataset name.")
    allowed = set(selected_names)
    rows = [row for row in source_rows if str(row.get("dataset")) in allowed]
    found = {str(row.get("dataset")) for row in rows}
    missing_datasets = sorted(allowed - found)
    if missing_datasets:
        raise RuntimeError(
            f"A1 source manifest has no rows for datasets: {missing_datasets}."
        )
    max_samples = int(max_samples)
    if max_samples == 0 or max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    if max_samples > 0:
        rows = rows[:max_samples]
    joined = []
    for source_row in rows:
        key = (str(source_row["dataset"]), str(source_row["stem"]))
        semantic_row = semantic_map.get(key)
        if semantic_row is None:
            raise RuntimeError(f"Semantic source is missing sample {key}.")
        joined.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "source_cache_path": str(source_row["cache_path"]),
                "semantic_cache_path": str(semantic_row["cache_path"]),
                "source_manifest": str(source_manifest.resolve()),
                "semantic_manifest": str(semantic_manifest.resolve()),
            }
        )
    if not joined:
        raise RuntimeError("No offline source rows selected.")
    return joined


def _load_payload(path, expected_dataset, expected_stem, expected_version, label):
    path = Path(path)
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict: {path}")
    if (
        str(payload.get("dataset")) != expected_dataset
        or str(payload.get("stem")) != expected_stem
    ):
        raise RuntimeError(f"{label} identity mismatch: {path}")
    if str(payload.get("version")) != expected_version:
        raise RuntimeError(
            f"{label} version mismatch: {payload.get('version')} != "
            f"{expected_version} | {path}"
        )
    if bool(payload.get("training_gt_read", True)):
        raise RuntimeError(f"{label} reports training GT access: {path}")
    if bool(payload.get("teacher_prediction_read", True)):
        raise RuntimeError(f"{label} reports Teacher access: {path}")
    return payload, path


def build_offline_payload(row, offline_mode):
    mode = str(offline_mode).strip().lower()
    if mode not in DABE_CLEAN_OFFLINE_MODES:
        raise RuntimeError(f"Unsupported offline mode: {mode!r}.")
    dataset, stem = str(row["dataset"]), str(row["stem"])
    source, source_path = _load_payload(
        row["source_cache_path"], dataset, stem, SOURCE_VERSION, "A1 source"
    )
    semantic_source, semantic_path = _load_payload(
        row["semantic_cache_path"],
        dataset,
        stem,
        SEMANTIC_SOURCE_VERSION,
        "semantic source",
    )
    if float(source.get("params", {}).get("BC_LAMBDA", -1.0)) != 0.0:
        raise RuntimeError(f"Pure-offline v3 requires A1 BC_LAMBDA=0: {source_path}")
    residual = _tensor(source, "residual_37", source_path)
    fg_score = _tensor(source, "fg_score_37", source_path)
    if float((fg_score - residual).abs().max().item()) > 1e-6:
        raise RuntimeError(f"A1 fg_score != residual: {source_path}")
    p_rw = _tensor(source, "p_rw_37", source_path)
    evidence = _tensor(source, "evidence_37", source_path)
    bc_map = _tensor(source, "bc_map_37", source_path)
    semantic = _tensor(
        semantic_source, "semantic_fg_tendency_37", semantic_path
    )
    outputs = build_offline_consolidation(
        residual_37=residual,
        bc_map_37=bc_map,
        p_rw_37=p_rw,
        evidence_37=evidence,
        semantic_fg_tendency_37=semantic,
        offline_mode=mode,
        output_size=(68, 68),
    )
    primary_reference = _tensor(source, "p_base_37", source_path)
    primary_error = float(
        (outputs["primary_fg_37"] - primary_reference).abs().max().item()
    )
    if primary_error > 1e-6:
        raise RuntimeError(
            f"primary_fg != source p_base for {dataset}/{stem}: {primary_error:.9g}"
        )
    baseline_reference_error = None
    if mode == "baseline_dp":
        baseline_reference = source.get("target_dp_68")
        if not torch.is_tensor(baseline_reference):
            raise RuntimeError(
                f"A1 source is missing target_dp_68 for V0 regression: {source_path}"
            )
        baseline_reference = baseline_reference.detach().cpu().float()
        if tuple(baseline_reference.shape) != (1, 68, 68):
            raise RuntimeError(
                f"A1 target_dp_68 shape mismatch: {source_path}"
            )
        baseline_reference_error = float(
            (outputs["target_offline_68"] - baseline_reference).abs().max().item()
        )
        if baseline_reference_error > 1e-6:
            raise RuntimeError(
                "V0 failed to reproduce the A1 Clean-DP 68x68 target for "
                f"{dataset}/{stem}: {baseline_reference_error:.9g}"
            )
    payload = {
        **outputs,
        "bc_map_37": bc_map,
        "evidence_37": evidence,
        "payload_version": DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
        "version": DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
        "offline_mode": mode,
        "dataset": dataset,
        "stem": stem,
        "backbone_key": str(source.get("backbone_key", "dinov1-s8")),
        "source_version": str(source.get("version")),
        "semantic_source_version": str(semantic_source.get("version")),
        "source_cache": str(source_path.resolve()),
        "semantic_source_cache": str(semantic_path.resolve()),
        "formula_fingerprint": DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT,
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "teacher_checkpoint_read": False,
        "student_checkpoint_read": False,
        "epoch_dependent": False,
        "history_read": False,
        "static_weight_map_written": False,
        "routing_map_written": False,
        "baseline_reference_max_error": baseline_reference_error,
    }
    return payload


def _manifest_row(payload, cache_path):
    target = payload["target_offline_68"]
    primary = payload["primary_fg_37"]
    complementary = payload["complementary_fg_37"]
    consolidated = payload["consolidated_fg_37"]
    background = payload["background_evidence_37"]
    conflict = payload["conflict_37"]
    commitment = payload["commitment_37"]
    return {
        "dataset": payload["dataset"],
        "stem": payload["stem"],
        "cache_path": str(Path(cache_path).resolve()),
        "backbone_key": payload["backbone_key"],
        "payload_version": payload["payload_version"],
        "version": payload["version"],
        "offline_mode": payload["offline_mode"],
        "source_cache": payload["source_cache"],
        "semantic_source_cache": payload["semantic_source_cache"],
        "target_shape": list(target.shape),
        "shape_37": [1, 37, 37],
        "shape_68": [1, 68, 68],
        "formula_fingerprint": payload["formula_fingerprint"],
        "target_mean": float(target.mean().item()),
        "target_hard_area": float((target > 0.5).float().mean().item()),
        "primary_fg_mean": float(primary.mean().item()),
        "primary_fg_hard_area": float((primary > 0.5).float().mean().item()),
        "complementary_fg_mean": float(complementary.mean().item()),
        "consolidated_fg_mean": float(consolidated.mean().item()),
        "consolidated_fg_hard_area": float(
            (consolidated > 0.5).float().mean().item()
        ),
        "background_mean": float(background.mean().item()),
        "conflict_mean": float(conflict.mean().item()),
        "commitment_mean": float(commitment.mean().item()),
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "teacher_checkpoint_read": False,
        "student_checkpoint_read": False,
        "epoch_dependent": False,
        "history_read": False,
        "static_weight_map_written": False,
        "routing_map_written": False,
        "cache_kind": "offline_consolidation",
        "baseline_reference_max_error": payload["baseline_reference_max_error"],
    }


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def generate_offline_cache(rows, output_root, offline_mode, workers=0, overwrite=False):
    output_root = Path(output_root)
    existing = []
    if output_root.exists():
        existing = [path for path in output_root.rglob("*") if path.is_file()]
    if existing and not bool(overwrite):
        raise FileExistsError(
            f"Output directory is non-empty and overwrite=false: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    def process(row):
        payload = build_offline_payload(row, offline_mode)
        path = output_root / payload["dataset"] / f"{payload['stem']}.pt"
        if path.exists() and not bool(overwrite):
            raise FileExistsError(f"Refusing to overwrite offline cache: {path}")
        _atomic_save(payload, path)
        return _manifest_row(payload, path)

    workers = int(workers)
    if workers < 0:
        raise ValueError("--workers must be >= 0.")
    if workers <= 1:
        manifest_rows = [process(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            manifest_rows = list(executor.map(process, rows))
    manifest = output_root / "manifest_train.jsonl"
    if manifest.exists() and not bool(overwrite):
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    write_jsonl(manifest, manifest_rows)
    return manifest, manifest_rows
