"""Shared I/O and protocol guards for reconstruction-rescue experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from tools.gbsp_knn_lsr_common import (
    DATASETS, load_manifest, load_torch, normalize_dataset, read_jsonl,
)


GRID = 37
NUM_PATCHES = GRID * GRID
FEATURE_DIM = 384
MAIN_ROOT = Path(__file__).resolve().parents[1]


def require_output_outside_main(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError(f"output must stay outside the main code tree: {output}")
    return output


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_core_rows(root: str | Path, *, split: str, max_samples: int) -> list[dict]:
    root = Path(root).resolve()
    rows = load_manifest(root, split=split, max_samples=max_samples)
    dataset_directory = {"CAMO": "TE-CAMO", "COD10K": "TE-COD10K"}
    for row in rows:
        current = Path(row["cache_path"])
        if not current.is_file():
            candidates = list(dict.fromkeys([
                root / split / dataset_directory.get(row["dataset"], row["dataset"]) / f"{row['stem']}.pt",
                root / split / row["dataset"] / f"{row['stem']}.pt",
            ]))
            existing = [path for path in candidates if path.is_file()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    f"cannot rebase core payload {row['dataset']}/{row['stem']}; "
                    f"manifest={current}, candidates={candidates}"
                )
            row["cache_path"] = str(existing[0])
    return rows


def apply_feature_manifest_override(rows: list[dict], manifest: str | Path) -> list[dict]:
    """Replace machine-specific feature/image/GT paths by identity."""
    manifest = Path(manifest).resolve()
    feature_index: dict[tuple[str, str], dict] = {}
    for source in read_jsonl(manifest):
        key = (normalize_dataset(source.get("dataset", "")), str(source.get("stem", "")))
        if not all(key) or key in feature_index:
            raise RuntimeError(f"invalid/duplicate feature identity in {manifest}: {key}")
        feature_index[key] = source
    output = []
    for raw in rows:
        row = dict(raw)
        key = (normalize_dataset(row["dataset"]), str(row["stem"]))
        if key not in feature_index:
            raise KeyError(f"feature manifest misses {key}")
        feature_row = feature_index[key]
        feature_path = Path(feature_row["cache_path"])
        image_path = Path(feature_row["image_path"])
        gt_path = Path(feature_row.get("gt_path") or image_path.parent.parent / "gt" / f"{key[1]}.png")
        for label, path in (("feature", feature_path), ("image", image_path), ("GT", gt_path)):
            if not path.is_file():
                raise FileNotFoundError(f"rebased {label} path does not exist for {key}: {path}")
        row["source_feature_path_override"] = str(feature_path)
        row["image_path"] = str(image_path)
        row["gt_path"] = str(gt_path)
        output.append(row)
    return output


def feature_tensor(payload: dict, source: Path) -> torch.Tensor:
    for key in ("tensor", "patch_tokens", "features"):
        value = payload.get(key)
        if torch.is_tensor(value):
            break
    else:
        candidates = [
            value for value in payload.values()
            if torch.is_tensor(value) and value.numel() == FEATURE_DIM * NUM_PATCHES
        ]
        if len(candidates) != 1:
            raise KeyError(f"cannot resolve one feature tensor: {source}")
        value = candidates[0]
    value = value.detach().float().squeeze().contiguous()
    if tuple(value.shape) != (FEATURE_DIM, GRID, GRID):
        raise ValueError(f"feature must be [{FEATURE_DIM},{GRID},{GRID}]: {source}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"feature contains NaN/Inf: {source}")
    return value


def load_core_inputs(row: dict, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    core_path = Path(row["cache_path"])
    core = load_torch(core_path)
    identity = (normalize_dataset(row["dataset"]), str(row["stem"]))
    if (normalize_dataset(core.get("dataset", "")), str(core.get("stem", ""))) != identity:
        raise RuntimeError(f"core identity mismatch: {core_path}")
    r8 = core.get("results", {}).get("r8")
    if not isinstance(r8, dict):
        raise KeyError(f"missing frozen r8 result: {core_path}")
    background = torch.as_tensor(r8.get("background_indices"), dtype=torch.long)
    if background.ndim != 1 or background.numel() < 33:
        raise ValueError(f"Full BC memory is too small: {core_path}")
    feature_path = Path(row.get("source_feature_path_override") or core["source_feature_path"])
    feature_payload = load_torch(feature_path)
    if (
        normalize_dataset(feature_payload.get("dataset", "")),
        str(feature_payload.get("stem", "")),
    ) != identity:
        raise RuntimeError(f"feature identity mismatch: {feature_path}")
    core = dict(core)
    core["source_feature_path"] = str(feature_path)
    return core, feature_tensor(feature_payload, feature_path).to(device), background.to(device)


def output_score_path(out_root: Path, split: str, dataset: str, stem: str) -> Path:
    return out_root / "scores" / split / normalize_dataset(dataset) / f"{stem}.pt"


def settings_fingerprint(settings: dict) -> str:
    raw = json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def expected_full(rows: list[dict]) -> bool:
    counts = {dataset: 0 for dataset in DATASETS}
    for row in rows:
        if row["dataset"] in counts:
            counts[row["dataset"]] += 1
    return counts == {"CHAMELEON": 76, "CAMO": 250, "COD10K": 2026, "NC4K": 4121}
