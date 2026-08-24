from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import yaml

MAIN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import (  # noqa: E402
    feature_tensor,
    load_gt_occupancy,
    load_torch,
    save_vector_figure,
)

GRID = 37
NUM_PATCHES = GRID * GRID
BG_COLOR = "#4C78A8"
FG_COLOR = "#F58518"


@dataclass(frozen=True)
class Settings:
    core_root: Path
    dataset: str
    gbsp_rank: int
    binary_threshold: float
    fg_core_threshold: float
    bg_core_threshold: float
    similarity_bins: np.ndarray
    residual_bins: np.ndarray
    descriptive_thresholds: tuple[float, ...]
    random_seed: int
    official_gbsp_config: Path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_gbsp_teaser_official_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_settings(path: str | Path) -> Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    official = resolve_path(raw["official_gbsp_config"])
    module = _load_module(official)
    settings = Settings(
        core_root=resolve_path(raw["core_root"]),
        dataset=str(raw["dataset"]),
        gbsp_rank=int(module.GBSP_CORE_PCA_MAX_RANK),
        binary_threshold=float(raw["patch_binary_threshold"]),
        fg_core_threshold=float(raw["patch_fg_core_threshold"]),
        bg_core_threshold=float(raw["patch_bg_core_threshold"]),
        similarity_bins=np.linspace(-1.0, 1.0, int(raw["similarity_bins"]) + 1),
        residual_bins=np.linspace(0.0, 1.0, int(raw["residual_bins"]) + 1),
        descriptive_thresholds=tuple(map(float, raw["similarity_descriptive_thresholds"])),
        random_seed=int(raw["random_seed"]),
        official_gbsp_config=official,
    )
    if settings.dataset != "CAMO":
        raise RuntimeError("stage 1 is frozen to CAMO-Test")
    if settings.gbsp_rank != 8:
        raise RuntimeError(f"formal GBSP rank must be 8, got {settings.gbsp_rank}")
    if settings.similarity_bins.size != 81 or not np.allclose(settings.similarity_bins[[0, -1]], [-1, 1]):
        raise RuntimeError("similarity histogram must use 80 fixed bins on [-1,1]")
    if settings.residual_bins.size != 51 or not np.allclose(settings.residual_bins[[0, -1]], [0, 1]):
        raise RuntimeError("residual histogram must use 50 fixed bins on [0,1]")
    return settings


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def camo_rows(settings: Settings) -> list[dict]:
    rows = read_jsonl(settings.core_root / "manifest_test.jsonl")
    output = []
    for raw in rows:
        dataset = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}.get(raw["dataset"], raw["dataset"])
        if dataset == settings.dataset:
            output.append({**raw, "dataset": dataset})
    output.sort(key=lambda row: row["stem"])
    if len(output) != 250:
        raise RuntimeError(f"formal CAMO manifest must contain 250 images, got {len(output)}")
    return output


def labels(occupancy: np.ndarray, settings: Settings, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    occupancy = np.asarray(occupancy).reshape(-1)
    if protocol == "allpatch_0.5":
        return (occupancy >= settings.binary_threshold).astype(np.uint8), np.ones(NUM_PATCHES, dtype=bool)
    if protocol == "core_0.2_0.8":
        label = np.zeros(NUM_PATCHES, dtype=np.uint8)
        label[occupancy >= settings.fg_core_threshold] = 1
        valid = (occupancy <= settings.bg_core_threshold) | (occupancy >= settings.fg_core_threshold)
        return label, valid
    raise ValueError(protocol)


def normalized_feature(core: dict, device: torch.device) -> torch.Tensor:
    payload = load_torch(core["source_feature_path"])
    feature = feature_tensor(payload)
    if feature.shape == (384, GRID, GRID):
        feature = feature.permute(1, 2, 0).reshape(NUM_PATCHES, 384)
    feature = torch.nn.functional.normalize(feature.float(), p=2, dim=1, eps=1e-12)
    if feature.shape != (NUM_PATCHES, 384) or not bool(torch.isfinite(feature).all()):
        raise RuntimeError("invalid formal DINO patch feature")
    return feature.to(device)


def probability_hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    count, _ = np.histogram(values, bins=bins)
    total = int(count.sum())
    if total == 0:
        raise ValueError("empty histogram")
    return count.astype(np.float64) / total


def distribution_stats(values: np.ndarray, prefix: str, thresholds: Sequence[float] = ()) -> dict:
    values = np.asarray(values, dtype=np.float64)
    row = {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q75": float(np.quantile(values, .75)),
        f"{prefix}_q90": float(np.quantile(values, .90)),
    }
    row.update({f"{prefix}_p_gt_{threshold:.1f}": float((values > threshold).mean()) for threshold in thresholds})
    return row


def validate_output(path: str | Path) -> Path:
    path = Path(path).resolve()
    allowed = (PROJECT_ROOT / "workdir").resolve()
    try:
        path.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"output must be under {allowed}: {path}") from error
    return path

