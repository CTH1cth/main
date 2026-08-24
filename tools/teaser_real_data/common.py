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
import torch.nn.functional as F
import yaml
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from models.gbsp_similarity_baselines import (  # noqa: E402
    compute_similarity_baselines,
    minmax_per_image,
)

GRID = 37
NUM_PATCHES = GRID * GRID
DATASET_ALIASES = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}


@dataclass(frozen=True)
class FrozenSettings:
    knn_k: int
    gbsp_rank: int
    hard_threshold: float
    core_root: Path
    main_dataset: str
    diagnostic_datasets: tuple[str, ...]
    fg_core_threshold: float
    bg_core_threshold: float
    binary_threshold: float
    hist_bins: int
    random_seed: int
    official_gbsp_config: Path
    official_similarity_config: Path
    official_mask_config: Path


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_settings(config_path: str | Path) -> FrozenSettings:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("teaser YAML must contain a mapping")
    core_config = resolve_project_path(raw["official_gbsp_config"])
    similarity_config = resolve_project_path(raw["official_similarity_config"])
    mask_config = resolve_project_path(raw["official_mask_config"])
    gbsp = _load_module(core_config, "_teaser_gbsp_config")
    similarity = _load_module(similarity_config, "_teaser_similarity_config")
    mask = _load_module(mask_config, "_teaser_mask_config")
    settings = FrozenSettings(
        knn_k=int(similarity.KNN_K),
        gbsp_rank=int(gbsp.GBSP_CORE_PCA_MAX_RANK),
        hard_threshold=float(mask.DABE_CLEAN_DABE_V2_HARD_THRESHOLD),
        core_root=resolve_project_path(raw["core_root"]),
        main_dataset=str(raw["dataset_for_main_figure"]),
        diagnostic_datasets=tuple(map(str, raw["datasets_for_diagnostic"])),
        fg_core_threshold=float(raw["patch_fg_core_threshold"]),
        bg_core_threshold=float(raw["patch_bg_core_threshold"]),
        binary_threshold=float(raw["patch_binary_threshold"]),
        hist_bins=int(raw["hist_bins"]),
        random_seed=int(raw["random_seed"]),
        official_gbsp_config=core_config,
        official_similarity_config=similarity_config,
        official_mask_config=mask_config,
    )
    if settings.knn_k != 8 or settings.gbsp_rank != 8:
        raise RuntimeError(
            f"frozen teaser requires KNN8 and GBSP-r8, got K={settings.knn_k}, "
            f"rank={settings.gbsp_rank}"
        )
    if settings.main_dataset != "CAMO":
        raise RuntimeError("the main teaser dataset is frozen to CAMO")
    if settings.diagnostic_datasets != ("CAMO", "COD10K", "NC4K"):
        raise RuntimeError("diagnostic datasets must remain CAMO/COD10K/NC4K")
    return settings


def normalize_dataset(value: str) -> str:
    return DATASET_ALIASES.get(str(value), str(value))


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_core_rows(settings: FrozenSettings) -> list[dict]:
    path = settings.core_root / "manifest_test.jsonl"
    rows = read_jsonl(path)
    output = []
    for row in rows:
        item = dict(row)
        item["dataset"] = normalize_dataset(item["dataset"])
        output.append(item)
    return output


def load_torch(path: str | Path) -> dict:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dictionary payload: {path}")
    return value


def feature_tensor(payload: dict) -> torch.Tensor:
    for key in ("tensor", "patch_tokens", "features"):
        value = payload.get(key)
        if torch.is_tensor(value):
            value = value.squeeze()
            if value.shape == (384, GRID, GRID):
                return value
            if value.shape == (NUM_PATCHES, 384):
                return value
    values = [
        value.squeeze() for value in payload.values()
        if torch.is_tensor(value) and value.numel() == 384 * NUM_PATCHES
    ]
    if len(values) != 1:
        raise KeyError("cannot uniquely resolve the 37x37 DINO feature tensor")
    return values[0]


def load_gt_occupancy(gt_path: str | Path) -> np.ndarray:
    """Task-book protocol: original GT -> nearest 296 -> 8x8 average pool."""
    with Image.open(gt_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).reshape(1, 1, *array.shape)
    gt296 = F.interpolate(tensor, size=(296, 296), mode="nearest")
    return F.avg_pool2d(gt296, kernel_size=8, stride=8)[0, 0].numpy().reshape(-1)


def load_native_images(image_path: str | Path, gt_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert("L")).copy()
    return rgb, gt


def labels_from_occupancy(
    occupancy: np.ndarray,
    settings: FrozenSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupancy = np.asarray(occupancy, dtype=np.float32).reshape(-1)
    core_label = np.full(NUM_PATCHES, -1, dtype=np.int8)
    core_label[occupancy <= settings.bg_core_threshold] = 0
    core_label[occupancy >= settings.fg_core_threshold] = 1
    binary = (occupancy >= settings.binary_threshold).astype(np.uint8)
    valid = core_label >= 0
    return core_label, binary, valid


def native_response(score37: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Official visualization resize: 37 -> 68 -> original, bilinear."""
    value = torch.as_tensor(score37, dtype=torch.float32).reshape(1, 1, GRID, GRID)
    value = F.interpolate(value, size=(68, 68), mode="bilinear", align_corners=False)
    value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    return value[0, 0].numpy()


def native_hard_mask(score37: np.ndarray, size: tuple[int, int], threshold: float) -> np.ndarray:
    """Official pseudo-label order: Min-Max 37 -> bilinear 68 -> strict threshold."""
    value = torch.as_tensor(score37, dtype=torch.float32).reshape(1, 1, GRID, GRID)
    value = F.interpolate(value, size=(68, 68), mode="bilinear", align_corners=False)
    hard68 = (value > float(threshold)).float()
    hard = F.interpolate(hard68, size=size, mode="nearest")[0, 0].numpy()
    return (hard * 255.0).astype(np.uint8)


def extract_scores(
    core: dict,
    device: torch.device,
    *,
    knn_k: int,
    gbsp_rank: int,
) -> dict[str, np.ndarray | int | float]:
    key = f"r{int(gbsp_rank)}"
    r8 = core["results"][key]
    indices = torch.as_tensor(r8["background_indices"], dtype=torch.long)
    feature_payload = load_torch(core["source_feature_path"])
    feature = feature_tensor(feature_payload).to(device)
    result = compute_similarity_baselines(feature, indices.to(device), k=int(knn_k))
    score_key = "knn8_cos"
    if int(knn_k) != 8:
        raise RuntimeError("the frozen production payload exposes only the audited KNN8 score")
    knn_fg = result.scores[score_key].detach().cpu().reshape(-1)
    knn_norm = minmax_per_image(knn_fg).reshape(-1)
    gbsp_raw = torch.as_tensor(r8["absolute_raw"]).float().reshape(-1)
    gbsp_norm = torch.as_tensor(r8["absolute_minmax"]).float().reshape(-1)
    reproduced = minmax_per_image(gbsp_raw)
    error = float((gbsp_norm - reproduced).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"GBSP official Min-Max reproduction failed: {error}")
    candidate = torch.zeros(NUM_PATCHES, dtype=torch.uint8)
    candidate[indices] = 1
    return {
        "knn_bg_raw": (1.0 - knn_fg).numpy(),
        "knn_fg_raw": knn_fg.numpy(),
        "knn_fg_normalized": knn_norm.numpy(),
        "gbsp_raw": gbsp_raw.numpy(),
        "gbsp_normalized": gbsp_norm.numpy(),
        "background_candidate": candidate.numpy(),
        "background_indices": indices.numpy(),
        "self_match_violation_count": result.self_match_violation_count,
        "num_background": result.num_background,
        "gbsp_minmax_max_abs_error": error,
        "feature": feature.detach().cpu(),
    }


def save_vector_figure(fig, base: str | Path, *, dpi: int = 600) -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def load_score_rows(score_root: str | Path) -> list[dict]:
    return read_jsonl(Path(score_root) / "score_manifest.jsonl")


def load_npz(row: dict):
    return np.load(row["score_path"], allow_pickle=False)
