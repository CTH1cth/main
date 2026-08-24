from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MAIN_ROOT.parent
GRID = 37
PATCH_SIZE = 8
INPUT_SIZE = GRID * PATCH_SIZE
NUM_PATCHES = GRID * GRID
FEATURE_DIM = 384


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("sample-mining config must contain a mapping")
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def torch_payload(path: str | Path) -> dict:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"expected dictionary payload: {path}")
    return value


def feature_tensor(payload: dict, path: str | Path) -> torch.Tensor:
    for key in ("tensor", "patch_tokens", "features"):
        value = payload.get(key)
        if torch.is_tensor(value):
            value = value.detach().cpu().float().squeeze()
            break
    else:
        candidates = [
            value.detach().cpu().float().squeeze()
            for value in payload.values()
            if torch.is_tensor(value) and value.numel() == FEATURE_DIM * NUM_PATCHES
        ]
        if len(candidates) != 1:
            raise KeyError(f"cannot uniquely resolve DINO features: {path}")
        value = candidates[0]
    if tuple(value.shape) == (FEATURE_DIM, GRID, GRID):
        value = value.permute(1, 2, 0).reshape(NUM_PATCHES, FEATURE_DIM)
    elif tuple(value.shape) == (NUM_PATCHES, FEATURE_DIM):
        value = value.contiguous()
    elif tuple(value.shape) == (FEATURE_DIM, NUM_PATCHES):
        value = value.t().contiguous()
    else:
        raise ValueError(f"unexpected DINO feature shape {tuple(value.shape)}: {path}")
    if not bool(torch.isfinite(value).all()) or bool((value.norm(dim=1) <= 0).any()):
        raise ValueError(f"invalid DINO feature tensor: {path}")
    return F.normalize(value, p=2, dim=1, eps=1e-12)


def gt_occupancy(gt_path: str | Path) -> np.ndarray:
    with Image.open(gt_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    value = torch.from_numpy(array).reshape(1, 1, *array.shape)
    value = F.interpolate(value, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")
    value = F.avg_pool2d(value, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    if tuple(value.shape) != (1, 1, GRID, GRID):
        raise RuntimeError(f"unexpected occupancy shape: {tuple(value.shape)}")
    return value[0, 0].numpy().reshape(-1)


def resized_input(path: str | Path, *, is_mask: bool = False) -> Image.Image:
    with Image.open(path) as image:
        mode = "L" if is_mask else "RGB"
        resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        return image.convert(mode).resize((INPUT_SIZE, INPUT_SIZE), resampling).copy()


def score_to_image(score: np.ndarray, *, hard: bool, threshold: float) -> np.ndarray:
    value = torch.as_tensor(score, dtype=torch.float32).reshape(1, 1, GRID, GRID)
    value = F.interpolate(value, size=(68, 68), mode="bilinear", align_corners=False)
    if hard:
        value = (value > float(threshold)).float()
        value = F.interpolate(value, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")
    else:
        value = F.interpolate(value, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    return value[0, 0].numpy()


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool).reshape(-1)
    gt = np.asarray(target, dtype=bool).reshape(-1)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "mae": float(np.not_equal(pred, gt).mean()),
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.float64)
    rank[order] = np.arange(len(values), dtype=np.float64) / float(len(values) - 1)
    return rank


def patch_box(index: int) -> tuple[int, int, int, int]:
    if not 0 <= int(index) < NUM_PATCHES:
        raise IndexError(index)
    y, x = divmod(int(index), GRID)
    return x * PATCH_SIZE, y * PATCH_SIZE, (x + 1) * PATCH_SIZE, (y + 1) * PATCH_SIZE


def assert_output_root(path: str | Path) -> Path:
    output = Path(path).resolve()
    allowed = (PROJECT_ROOT / "workdir").resolve()
    output.relative_to(allowed)
    if MAIN_ROOT == output or MAIN_ROOT in output.parents:
        raise ValueError("output cannot be inside the source repository")
    return output

