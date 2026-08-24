"""Shared I/O and protocol helpers for candidate harmful-influence analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.gbsp_knn_lsr_common import (
    DATASETS,
    index_manifest,
    load_manifest,
    load_native_gt,
    load_torch,
    normalize_dataset,
)
from models.gbsp_candidate_influence import fit_scatter_basis


GRID = 37
NUM_PATCHES = GRID * GRID
FEATURE_DIMENSION = 384
RANK = 8


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(dataset: str, stem: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{dataset}\0{stem}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def high_is_one_percentile(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    output = np.full(value.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(value)
    if not valid.any():
        return output
    index = np.where(valid)[0]
    order = index[np.argsort(-value[index], kind="stable")]
    output[order] = np.arange(order.size, 0, -1, dtype=np.float64) / float(order.size)
    return output


def top_mask(values: np.ndarray, fraction: float, valid: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    candidate = np.isfinite(value) if valid is None else (np.asarray(valid).reshape(-1).astype(bool) & np.isfinite(value))
    index = np.where(candidate)[0]
    output = np.zeros(value.shape, dtype=bool)
    if index.size:
        count = max(1, int(math.ceil(float(fraction) * index.size)))
        order = index[np.argsort(-value[index], kind="stable")]
        output[order[:count]] = True
    return output


def nearest_patch_labels(gt_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    gt = load_native_gt(gt_path)
    label = F.interpolate(gt.unsqueeze(0), size=(GRID, GRID), mode="nearest").squeeze().reshape(-1) > 0.5
    return gt, label


def tensor_field(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


def load_feature(path: str | Path) -> torch.Tensor:
    source = Path(path)
    payload = load_torch(source)
    feature = tensor_field(payload, "tensor", (FEATURE_DIMENSION, GRID, GRID), source)
    return F.normalize(feature.permute(1, 2, 0).reshape(NUM_PATCHES, FEATURE_DIMENSION), p=2, dim=1)


def build_indices(config) -> tuple[list[dict], dict, dict]:
    core_rows = load_manifest(resolve_path(config.GBSP_INFLUENCE_CORE_ROOT), split="test")
    r0_rows = load_manifest(resolve_path(config.GBSP_INFLUENCE_R0_ROOT), split="test")
    oracle_rows = load_manifest(resolve_path(config.GBSP_INFLUENCE_ORACLE_ROOT), split="test")
    return core_rows, index_manifest(r0_rows), index_manifest(oracle_rows)


def load_aligned(row: dict, r0_index: dict) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    dataset, stem = normalize_dataset(row["dataset"]), str(row["stem"])
    core_path = Path(row["cache_path"])
    core = load_torch(core_path)
    r0_row = r0_index[(dataset, stem)]
    r0_path = Path(r0_row["cache_path"])
    r0 = load_torch(r0_path)
    if core.get("gbsp_core_version") != "gbsp_core_optimization_v1":
        raise RuntimeError(f"unsupported core cache: {core_path}")
    if r0.get("version") != "gbsp_cvbr_weighted_pca_v1" or r0.get("variant") != "R0_uniform":
        raise RuntimeError(f"unsupported R0 cache: {r0_path}")
    if (normalize_dataset(core.get("dataset", "")), str(core.get("stem", ""))) != (dataset, stem):
        raise RuntimeError(f"core identity mismatch: {core_path}")
    if (normalize_dataset(r0.get("dataset", "")), str(r0.get("stem", ""))) != (dataset, stem):
        raise RuntimeError(f"R0 identity mismatch: {r0_path}")
    source = core.get("results", {}).get("r8")
    if not isinstance(source, dict) or int(source.get("selected_rank", -1)) != RANK:
        raise RuntimeError(f"fixed rank-8 result missing: {core_path}")
    core_indices = torch.as_tensor(source["background_indices"]).long().reshape(-1)
    r0_indices = torch.as_tensor(r0["background_indices"]).long().reshape(-1)
    if not torch.equal(core_indices, r0_indices):
        raise RuntimeError(f"Full-BC candidate mismatch: {dataset}/{stem}")
    feature = load_feature(r0["source_feature_path"])
    return core, r0, feature, r0_indices


def oracle_items(
    oracle_row: dict | None,
    feature: torch.Tensor,
    expected_count: int,
    expected_seeds: tuple[int, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[list[torch.Tensor], str]:
    if oracle_row is None:
        return [], "oracle_manifest_missing"
    payload = load_torch(oracle_row["cache_path"])
    items = payload.get("matched_size_clean", [])
    seeds = tuple(sorted(int(item.get("seed", -1)) for item in items))
    if seeds != tuple(expected_seeds):
        return [], f"oracle_seed_mismatch:{seeds}"
    bases = []
    for item in sorted(items, key=lambda value: int(value["seed"])):
        if not bool(item.get("valid")):
            return [], f"oracle_invalid_seed_{item.get('seed')}:{item.get('reason', '')}"
        index = torch.as_tensor(item.get("candidate_indices")).long().reshape(-1)
        if index.numel() != int(expected_count) or index.unique().numel() != index.numel():
            return [], f"oracle_candidate_count_seed_{item.get('seed')}"
        background = feature.index_select(0, index).to(device=device, dtype=dtype)
        _, basis = fit_scatter_basis(background, RANK)
        bases.append(basis)
    return bases, "valid"


def parse_subset(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"invalid subset line: {line!r}")
        rows.append({
            "dataset": normalize_dataset(fields[0]), "stem": fields[1],
            "stratum": fields[2] if len(fields) > 2 else "",
            "contamination_ratio": float(fields[3]) if len(fields) > 3 else float("nan"),
        })
    return rows


def npz_path(root: Path, dataset: str, stem: str) -> Path:
    return root / dataset / f"{stem}.npz"
