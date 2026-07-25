#!/usr/bin/env python3
"""BISA-v0: offline background-intervention sensitivity audit.

This entry point is deliberately isolated from the training path.  It consumes
frozen DINO and DABE-PU caches, runs a frozen decoder under no_grad, and loads
GT only after every original/counterfactual forward for an image has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MAIN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MAIN_ROOT.parent
LOCAL_DEPENDENCY_ROOT = WORKSPACE_ROOT / "workdir" / "bisa_deps"
if LOCAL_DEPENDENCY_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEPENDENCY_ROOT))
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from analysis.bisa.intervention import (
    SUPPORTED_SUBSTITUTE_NORM_MODES,
    SUPPORTED_SUBSTITUTION_MODES,
    apply_group_interventions,
    assemble_group_responses,
    build_dabe_background_substitutes,
    coordinate_group_ids,
)
from analysis.bisa.metrics import (
    classify_teacher_errors,
    downsample_gt_occupancy,
    json_finite,
    strict_gt_labels,
    teacher_entropy,
)
from analysis.bisa.statistics import (
    build_statistical_outputs,
    build_statistical_outputs_from_parquet,
)
from analysis.bisa.visualization import render_bisa_visualization
from common.dabe_pseudo import _load_rgb_grid
from common.dataset import _load_image_68
from common.utils import config_to_dict, load_config, set_seed
from model import build_seg_head


ERROR_NAMES = {-1: "IGNORE", 0: "TN", 1: "FP", 2: "FN", 3: "TP"}
TOKEN_GRID = 37
REQUIRED_DABE_FIELDS = (
    "target_soft_37",
    "bc_map_37",
    "residual_37",
    "fg_score_37",
    "evidence_37",
    "bg_anchor_37",
    "params",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BISA-v0 background intervention sensitivity offline audit"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group_size", type=int, choices=(3, 4), default=4)
    parser.add_argument(
        "--substitution_modes",
        nargs="+",
        choices=SUPPORTED_SUBSTITUTION_MODES,
        default=["weighted_bg", "nearest_bg", "random_bg", "identity"],
    )
    parser.add_argument(
        "--substitute_norm_mode",
        choices=SUPPORTED_SUBSTITUTE_NORM_MODES,
        default="dabe_unit",
        help=(
            "dabe_unit keeps the exact unit-norm DABE reconstruction; "
            "query_match preserves its direction but matches each raw query norm"
        ),
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch_interventions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--visualization_count", type=int, default=40)
    parser.add_argument(
        "--use_teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--primary_output",
        choices=("coarse_logits_37",),
        default="coarse_logits_37",
    )
    parser.add_argument(
        "--robustness_reference",
        default=None,
        help="response_metrics.csv from the complementary group-size run",
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=10000,
        help="Fixed full-run default is 10000; sanity automatically caps at 200",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _state_signature(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


class _TokenParquetWriter:
    """Append token batches without retaining the full audit in RAM."""

    def __init__(self, path: Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "A real Parquet engine is required. Install pyarrow into the "
                f"isolated path {LOCAL_DEPENDENCY_ROOT}."
            ) from exc
        self._pa = pa
        self._pq = pq
        self.path = path
        self._writer = None
        self._schema = None
        self.row_count = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = self._pa.Table.from_pylist(rows, schema=self._schema)
        if self._writer is None:
            self._schema = table.schema
            self._writer = self._pq.ParquetWriter(
                self.path,
                self._schema,
                compression="zstd",
                use_dictionary=True,
            )
        self._writer.write_table(table)
        self.row_count += int(table.num_rows)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _match_manifests(
    feature_root: Path,
    dabe_root: Path,
    datasets: Iterable[str],
    max_samples: int,
) -> list[dict[str, Any]]:
    feature_manifest = feature_root / "manifest_train.jsonl"
    dabe_manifest = dabe_root / "manifest_train.jsonl"
    if not feature_manifest.is_file() or not dabe_manifest.is_file():
        raise FileNotFoundError(
            "Training manifests are required: "
            f"{feature_manifest}, {dabe_manifest}"
        )
    requested = tuple(dict.fromkeys(str(value) for value in datasets))
    if not requested:
        raise ValueError("At least one dataset is required")
    dabe_by_key = {
        (str(row["dataset"]), str(row["stem"])): row
        for row in _read_jsonl(dabe_manifest)
        if str(row.get("dataset")) in requested
    }
    matched_by_dataset: dict[str, list[dict[str, Any]]] = {
        dataset: [] for dataset in requested
    }
    counts = Counter()
    for feature_row in _read_jsonl(feature_manifest):
        dataset = str(feature_row.get("dataset"))
        if dataset not in requested:
            continue
        key = (dataset, str(feature_row.get("stem")))
        if key not in dabe_by_key:
            raise RuntimeError(f"No DABE cache entry for feature entry {key}")
        dabe_row = dabe_by_key[key]
        if Path(feature_row["image_path"]).resolve() != Path(dabe_row["image_path"]).resolve():
            raise RuntimeError(f"Feature/DABE image mismatch for {key}")
        matched_by_dataset[dataset].append(
            {
                "dataset": dataset,
                "stem": key[1],
                "image_id": f"{dataset}/{key[1]}",
                "image_path": Path(feature_row["image_path"]),
                "gt_path": Path(dabe_row["gt_path"]),
                "feature_path": Path(feature_row["cache_path"]),
                "dabe_path": Path(dabe_row["cache_path"]),
            }
        )
        counts[dataset] += 1
    missing = [dataset for dataset in requested if counts[dataset] == 0]
    if missing:
        raise RuntimeError(f"No matched samples for dataset(s): {missing}")
    if int(max_samples) < 0:
        matched = [
            sample
            for dataset in requested
            for sample in matched_by_dataset[dataset]
        ]
    else:
        # Finite multi-dataset audits are deterministic balanced pilots rather
        # than an accidental prefix of the first dataset in manifest order.
        limit = min(
            int(max_samples),
            sum(len(values) for values in matched_by_dataset.values()),
        )
        base, remainder = divmod(limit, len(requested))
        quotas = {
            dataset: min(
                len(matched_by_dataset[dataset]),
                base + int(index < remainder),
            )
            for index, dataset in enumerate(requested)
        }
        unfilled = limit - sum(quotas.values())
        while unfilled > 0:
            progressed = False
            for dataset in requested:
                if quotas[dataset] < len(matched_by_dataset[dataset]):
                    quotas[dataset] += 1
                    unfilled -= 1
                    progressed = True
                    if unfilled == 0:
                        break
            if not progressed:
                break
        matched = []
        for dataset in requested:
            values = matched_by_dataset[dataset]
            quota = quotas[dataset]
            if len(requested) == 1 or quota >= len(values):
                chosen = values[:quota]
            else:
                # Cover the full per-dataset manifest rather than selecting a
                # potentially category-biased prefix (notably COD10K stems).
                positions = [
                    min(
                        len(values) - 1,
                        int((index + 0.5) * len(values) / quota),
                    )
                    for index in range(quota)
                ]
                chosen = [values[position] for position in positions]
            matched.extend(chosen)
    if not matched:
        raise RuntimeError("No feature/DABE manifest matches")
    return matched


def _extract_feature(payload: Any, expected: dict[str, Any]) -> torch.Tensor:
    if isinstance(payload, dict):
        feature = payload.get("tensor")
        if str(payload.get("dataset")) != expected["dataset"]:
            raise RuntimeError("Feature payload dataset mismatch")
        if str(payload.get("stem")) != expected["stem"]:
            raise RuntimeError("Feature payload stem mismatch")
    else:
        feature = payload
    if not torch.is_tensor(feature) or list(feature.shape) != [384, TOKEN_GRID, TOKEN_GRID]:
        raise RuntimeError(
            f"Expected cached DINO feature [384,37,37], got {getattr(feature, 'shape', None)}"
        )
    feature = feature.detach().cpu().float().contiguous()
    if feature.requires_grad or not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError("Cached DINO feature must be detached and finite")
    return feature


def _field_37(payload: dict[str, Any], name: str) -> torch.Tensor:
    value = payload.get(name)
    if not torch.is_tensor(value):
        raise RuntimeError(f"DABE field {name} is missing or not a tensor")
    value = value.detach().cpu().float()
    while value.ndim > 2 and int(value.shape[0]) == 1:
        value = value.squeeze(0)
    if list(value.shape) != [TOKEN_GRID, TOKEN_GRID]:
        raise RuntimeError(f"DABE field {name} shape is {list(value.shape)}, expected [37,37]")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"DABE field {name} contains NaN/Inf")
    return value.contiguous()


def _load_dabe(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"DABE cache payload is not a mapping: {path}")
    missing = [name for name in REQUIRED_DABE_FIELDS if name not in payload]
    if missing:
        raise RuntimeError(f"DABE cache missing required fields {missing}: {path}")
    if str(payload.get("dataset")) != expected["dataset"] or str(payload.get("stem")) != expected["stem"]:
        raise RuntimeError("DABE payload identity mismatch")
    if str(payload.get("dabe_version")) != "pu_v11":
        raise RuntimeError(f"Expected DABE-PU v1.1, got {payload.get('dabe_version')}")
    return payload


def _infer_in_channels(state: dict[str, torch.Tensor]) -> int:
    for key in ("base_head.weight", "module.base_head.weight"):
        if key in state and state[key].ndim == 4:
            return int(state[key].shape[1])
    for key, value in state.items():
        if key.endswith("base_head.weight") and value.ndim == 4:
            return int(value.shape[1])
    raise RuntimeError("Could not infer decoder input channels from checkpoint")


def _load_decoder(
    checkpoint_path: Path,
    cfg: Any,
    device: torch.device,
    use_teacher: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_hash_before = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    saved_config = checkpoint.get("config", {})
    expected_exp = str(getattr(cfg, "EXP_NAME", ""))
    saved_exp = str(saved_config.get("EXP_NAME", "")) if isinstance(saved_config, dict) else ""
    if expected_exp and saved_exp and expected_exp != saved_exp:
        raise RuntimeError(
            f"Checkpoint/config EXP_NAME mismatch: {saved_exp!r} != {expected_exp!r}"
        )

    source_key = None
    model_source = None
    if use_teacher:
        if "ema_teacher" in checkpoint:
            source_key, model_source = "ema_teacher", "ema_teacher"
        elif "teacher" in checkpoint:
            source_key, model_source = "teacher", "ema_teacher"
        elif "student" in checkpoint:
            source_key, model_source = "student", "student_fallback_no_ema_teacher"
    else:
        if "student" not in checkpoint:
            raise RuntimeError("--no-use_teacher requested but checkpoint has no student state")
        source_key, model_source = "student", "student_requested"
    if source_key is None:
        raise RuntimeError(f"No usable decoder state in checkpoint: {checkpoint_path}")
    state = checkpoint[source_key]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint state {source_key} is not a mapping")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    in_channels = _infer_in_channels(state)
    decoder = build_seg_head(in_channels, cfg).to(device)
    incompat = decoder.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Strict decoder load failed: {incompat}")
    if hasattr(decoder, "set_epoch"):
        decoder.set_epoch(checkpoint_epoch)
    decoder.eval()
    decoder.requires_grad_(False)
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise RuntimeError("Frozen audit decoder unexpectedly has trainable parameters")
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_keys": sorted(str(key) for key in checkpoint),
        "saved_exp_name": saved_exp,
        "state_key": source_key,
        "model_source": model_source,
        "in_channels": in_channels,
        "state_signature_before": _state_signature(decoder),
    }
    del checkpoint
    return decoder, metadata


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=device.type == "cuda",
    )


@torch.no_grad()
def _forward(
    decoder: torch.nn.Module,
    feature_batch: torch.Tensor,
    image_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with _autocast(device):
        output = decoder(feature_batch, image_batch, return_aux=True)
    if not isinstance(output, dict):
        raise RuntimeError("DAGP audit requires return_aux dictionary output")
    if "coarse_logits_37" not in output:
        raise RuntimeError("Decoder output does not expose coarse_logits_37")
    coarse = output["coarse_logits_37"]
    if coarse.ndim != 4 or list(coarse.shape[1:]) != [1, TOKEN_GRID, TOKEN_GRID]:
        raise RuntimeError(f"coarse_logits_37 has invalid shape {list(coarse.shape)}")
    if coarse.requires_grad or not bool(torch.isfinite(coarse).all().item()):
        raise RuntimeError("Decoder coarse logits must be detached and finite")
    return {key: value.detach() if torch.is_tensor(value) else value for key, value in output.items()}


def _counterfactual_forward(
    decoder: torch.nn.Module,
    counterfactual: torch.Tensor,
    image_68: torch.Tensor,
    device: torch.device,
    batch_interventions: bool,
) -> dict[str, torch.Tensor]:
    group_count = int(counterfactual.shape[0])
    if batch_interventions:
        image_batch = image_68.unsqueeze(0).expand(group_count, -1, -1, -1)
        return _forward(decoder, counterfactual, image_batch, device)
    collected: dict[str, list[torch.Tensor]] = {}
    for group in range(group_count):
        result = _forward(
            decoder,
            counterfactual[group : group + 1],
            image_68.unsqueeze(0),
            device,
        )
        for key, value in result.items():
            if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == 1:
                collected.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def _load_gt_after_forwards(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"GT not found: {path}")
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    binary = (array > 0.5).astype(np.float32)
    return binary, torch.from_numpy(binary)


def _load_rgb_original(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _flatten_cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().reshape(-1).numpy()


def _token_rows(
    *,
    sample: dict[str, Any],
    checkpoint_epoch: int,
    group_size: int,
    mode: str,
    substitute_norm_mode: str,
    group_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    gt_occupancy: torch.Tensor,
    gt_strict: torch.Tensor,
    error_map: torch.Tensor,
    dabe: dict[str, Any],
    diagnostics: dict[str, Any],
    responses: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    grid = TOKEN_GRID
    y_coord, x_coord = np.indices((grid, grid))
    teacher_logits_np = _flatten_cpu(teacher_logits)
    teacher_prob_np = _flatten_cpu(teacher_prob)
    entropy_np = _flatten_cpu(teacher_entropy(teacher_prob))
    occupancy_np = _flatten_cpu(gt_occupancy)
    strict_np = _flatten_cpu(gt_strict).astype(np.int8)
    error_np = _flatten_cpu(error_map).astype(np.int8)
    fields = {
        "dabe_target_soft": _flatten_cpu(_field_37(dabe, "target_soft_37")),
        "dabe_background_connectivity": _flatten_cpu(_field_37(dabe, "bc_map_37")),
        "dabe_background_residual": _flatten_cpu(_field_37(dabe, "residual_37")),
        "dabe_foreground_score": _flatten_cpu(_field_37(dabe, "fg_score_37")),
        "dabe_evidence": _flatten_cpu(_field_37(dabe, "evidence_37")),
        "bg_cosine_gap": _flatten_cpu(diagnostics["bg_cosine_gap"]),
        "bg_feature_l2_distance": _flatten_cpu(diagnostics["bg_feature_l2_distance"]),
        "original_feature_norm": _flatten_cpu(diagnostics["original_feature_norm"]),
        "retrieval_weight_entropy": _flatten_cpu(diagnostics["retrieval_weight_entropy"]),
        "angle_matched_target_rad": _flatten_cpu(
            diagnostics["angle_matched_target_rad"]
        ),
        "angle_matched_target_deg": np.rad2deg(
            _flatten_cpu(diagnostics["angle_matched_target_rad"])
        ),
        "substitute_cosine_gap": _flatten_cpu(
            diagnostics[f"{mode}_cosine_gap"]
        ),
        "substitute_angle_rad": _flatten_cpu(
            diagnostics[f"{mode}_angle_rad"]
        ),
        "substitute_angle_deg": _flatten_cpu(
            diagnostics[f"{mode}_angle_deg"]
        ),
        "substitute_feature_norm": _flatten_cpu(
            diagnostics[f"{mode}_feature_norm"]
        ),
        "substitute_feature_l2_distance": _flatten_cpu(
            diagnostics[f"{mode}_feature_l2_distance"]
        ),
        "substitute_nearest_anchor_distance": _flatten_cpu(
            diagnostics[f"{mode}_nearest_anchor_distance"]
        ),
    }
    for substitution in ("weighted_bg", "nearest_bg", "random_bg"):
        fields[f"{substitution}_feature_norm"] = _flatten_cpu(
            diagnostics[f"{substitution}_feature_norm"]
        )
        fields[f"{substitution}_nearest_anchor_distance"] = _flatten_cpu(
            diagnostics[f"{substitution}_nearest_anchor_distance"]
        )
    for response_name in (
        "c_self_raw",
        "c_local_raw",
        "c_self_centered",
        "c_local_centered",
        "spill_ratio_map",
    ):
        fields[response_name] = _flatten_cpu(responses[response_name])
    group_np = _flatten_cpu(group_ids).astype(np.int16)
    target_area = float(gt_occupancy.mean().item())
    teacher_area = float((teacher_prob > 0.5).float().mean().item())
    rows = []
    for index in range(grid * grid):
        error_code = int(error_np[index])
        rows.append(
            {
                "dataset": sample["dataset"],
                "image_id": sample["image_id"],
                "image_stem": sample["stem"],
                "checkpoint_epoch": int(checkpoint_epoch),
                "x": int(x_coord.reshape(-1)[index]),
                "y": int(y_coord.reshape(-1)[index]),
                "gt_occupancy": float(occupancy_np[index]),
                "gt_label_strict": int(strict_np[index]),
                "gt_label_05": int(occupancy_np[index] >= 0.5),
                "teacher_logit": float(teacher_logits_np[index]),
                "teacher_prob": float(teacher_prob_np[index]),
                "teacher_entropy": float(entropy_np[index]),
                "teacher_binary": int(teacher_prob_np[index] > 0.5),
                "teacher_error_type": ERROR_NAMES[error_code],
                "substitution_mode": mode,
                "substitute_norm_mode": substitute_norm_mode,
                "group_size": int(group_size),
                "group_id": int(group_np[index]),
                "spill_ratio": float(fields["spill_ratio_map"][index]),
                "target_area": target_area,
                "teacher_area": teacher_area,
                **{
                    key: float(value[index])
                    for key, value in fields.items()
                    if key != "spill_ratio_map"
                },
            }
        )
    return rows


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_finite(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _safe_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(values)) if values.size else math.nan


def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.median().item()) if int(selected.numel()) else math.nan


def _visualization_category_scores(
    weighted_local: torch.Tensor,
    random_local: torch.Tensor,
    error_map: torch.Tensor,
    spill_ratio_map: torch.Tensor,
) -> dict[str, tuple[float, float]]:
    """Return category -> (ranking score, reported class contribution)."""
    tp = _masked_median(weighted_local, error_map == 3)
    fp = _masked_median(weighted_local, error_map == 1)
    fn = _masked_median(weighted_local, error_map == 2)
    tn = _masked_median(weighted_local, error_map == 0)
    tp_fp = tp - fp if math.isfinite(tp) and math.isfinite(fp) else math.nan
    fn_tn = fn - tn if math.isfinite(fn) and math.isfinite(tn) else math.nan
    finite_contributions = [
        value for value in (tp_fp, fn_tn) if math.isfinite(value)
    ]
    general_contribution = (
        max(finite_contributions) if finite_contributions else math.nan
    )
    disagreement = float((weighted_local - random_local).abs().mean().item())
    spill = float(spill_ratio_map.median().item())
    candidates = {
        "high_TP_response": (tp, tp_fp),
        "low_TP_response": (-tp if math.isfinite(tp) else math.nan, tp_fp),
        "high_FP_response": (fp, tp_fp),
        "high_FN_response": (fn, fn_tn),
        "low_FN_response": (-fn if math.isfinite(fn) else math.nan, fn_tn),
        "high_spillover": (spill, general_contribution),
        "weighted_vs_random_disagreement": (
            disagreement,
            general_contribution,
        ),
    }
    return {
        category: (float(score), float(contribution))
        for category, (score, contribution) in candidates.items()
        if math.isfinite(score)
    }


def _offer_visualization_candidate(
    storage: dict[tuple[int, str], list[dict[str, Any]]],
    *,
    epoch: int,
    category: str,
    score: float,
    contribution: float,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    limit: int = 5,
) -> None:
    key = (int(epoch), str(category))
    candidates = storage.setdefault(key, [])
    candidates.append(
        {
            "score": float(score),
            "contribution": float(contribution),
            "payload": payload,
            **metadata,
        }
    )
    candidates.sort(
        key=lambda item: (-item["score"], str(item["image_id"]))
    )
    del candidates[int(limit) :]


def _sanity_summary(
    token_frame: pd.DataFrame,
    image_frame: pd.DataFrame,
    checkpoint_metadata: list[dict[str, Any]],
    group_size: int,
    feature_norms: list[float],
    anchor_counts: list[int],
    peak_gpu_memory: int,
) -> dict[str, Any]:
    identity = token_frame[token_frame["substitution_mode"] == "identity"]
    weighted = token_frame[token_frame["substitution_mode"] == "weighted_bg"]
    random = token_frame[token_frame["substitution_mode"] == "random_bg"]
    groups = coordinate_group_ids(TOKEN_GRID, TOKEN_GRID, group_size)
    group_counts = torch.bincount(groups.reshape(-1), minlength=group_size * group_size)
    label_reference = (
        weighted
        if not weighted.empty
        else token_frame[
            token_frame["substitution_mode"]
            == sorted(token_frame["substitution_mode"].unique())[0]
        ]
    )
    valid = label_reference[
        label_reference["teacher_error_type"] != "IGNORE"
    ]
    error_counts = valid["teacher_error_type"].value_counts().to_dict()
    first_checkpoint = checkpoint_metadata[0]
    return {
        "resolved_config": first_checkpoint.get("saved_exp_name"),
        "checkpoint_path": first_checkpoint["checkpoint_path"],
        "checkpoint_epoch": first_checkpoint["checkpoint_epoch"],
        "model_source": first_checkpoint["model_source"],
        "dataset": sorted(token_frame["dataset"].unique().tolist()),
        "sample_count": int(token_frame["image_id"].nunique()),
        "feature_shape": [384, 37, 37],
        "feature_norm_mean": _safe_mean(feature_norms),
        "background_anchor_count_mean": _safe_mean(anchor_counts),
        "weighted_bg_norm_mean": float(weighted["weighted_bg_feature_norm"].mean()) if not weighted.empty else None,
        "nearest_bg_norm_mean": float(token_frame.loc[token_frame["substitution_mode"] == "nearest_bg", "nearest_bg_feature_norm"].mean()) if "nearest_bg" in set(token_frame["substitution_mode"]) else None,
        "random_bg_norm_mean": float(random["random_bg_feature_norm"].mean()) if not random.empty else None,
        "group_size": int(group_size),
        "num_groups": int(group_size * group_size),
        "tokens_per_group_min": int(group_counts.min().item()),
        "tokens_per_group_max": int(group_counts.max().item()),
        "coverage_ratio": float((group_counts.sum().item()) / (TOKEN_GRID * TOKEN_GRID)),
        "coarse_logits_shape": [1, 1, 37, 37],
        "identity_response_mean": float(identity["c_self_raw"].abs().mean()) if not identity.empty else None,
        "identity_response_max": float(identity["c_self_raw"].abs().max()) if not identity.empty else None,
        "weighted_self_response_mean": float(weighted["c_self_raw"].mean()) if not weighted.empty else None,
        "weighted_local_response_mean": float(weighted["c_local_raw"].mean()) if not weighted.empty else None,
        "weighted_centered_response_mean": float(weighted["c_local_centered"].mean()) if not weighted.empty else None,
        "random_centered_response_mean": float(random["c_local_centered"].mean()) if not random.empty else None,
        "spill_ratio_mean": float(weighted["spill_ratio"].mean()) if not weighted.empty else None,
        "spill_ratio_median": float(weighted["spill_ratio"].median()) if not weighted.empty else None,
        "gt_valid_fg_tokens": int((valid["gt_label_strict"] == 1).sum()),
        "gt_valid_bg_tokens": int((valid["gt_label_strict"] == 0).sum()),
        "tp_count": int(error_counts.get("TP", 0)),
        "fp_count": int(error_counts.get("FP", 0)),
        "fn_count": int(error_counts.get("FN", 0)),
        "tn_count": int(error_counts.get("TN", 0)),
        "teacher_has_grad": False,
        "dino_has_grad": False,
        "checkpoint_modified": bool(any(item["checkpoint_modified"] for item in checkpoint_metadata)),
        "gt_loaded_after_interventions": bool(image_frame["gt_loaded_after_interventions"].all()),
        "seconds_per_image": float(image_frame["seconds"].mean()),
        "decoder_forwards_per_image": int(1 + group_size * group_size * image_frame["substitution_mode"].nunique()),
        "peak_gpu_memory_bytes": int(peak_gpu_memory),
    }


def _report(
    *,
    path: Path,
    verdict: dict[str, Any],
    cfg_path: Path,
    checkpoint_metadata: list[dict[str, Any]],
    args: argparse.Namespace,
    sample_count: int,
    failed_count: int,
    sanity_summary: dict[str, Any],
) -> None:
    lines = [
        "# BISA-v0 Offline Audit Report",
        "",
        "## 1. Verdict",
        "",
        f"`{verdict.get('verdict')}`",
        "",
        "PASS 仅表示该离线信号值得继续研究，不等同于训练方法有效。",
        "",
        "## 2. Repository facts",
        "",
        f"- Config: `{cfg_path}`",
        "- Frozen DINO cache only; DINO was not instantiated.",
        "- Primary output: native DAGP `coarse_logits_37`.",
        "- DABE-PU v1.1 background anchors and saved reconstruction parameters were used.",
        "- No training, optimizer, EMA update, validation, or checkpoint write occurred.",
        "",
        "## 3. Checkpoints",
        "",
    ]
    for item in checkpoint_metadata:
        lines.append(
            f"- epoch {item['checkpoint_epoch']}: `{item['checkpoint_path']}`; "
            f"state `{item['state_key']}` -> `{item['model_source']}`; modified={item['checkpoint_modified']}"
        )
    lines.extend(
        [
            "",
            "## 4. Datasets",
            "",
            f"- Requested: {', '.join(args.datasets)}",
            f"- Processed unique images: {sample_count}",
            f"- Failed image/checkpoint pairs: {failed_count}",
            "",
            "## 5. Intervention implementation",
            "",
            "DABE feature/color similarity, top-K, softmax temperature, normalized-anchor mixture, and final direction normalization are recomputed exactly from cache parameters. Norm handling is an explicit audited intervention setting rather than an implicit conversion.",
            f"Substitute norm mode: `{args.substitute_norm_mode}`. In `query_match`, DABE directions are retained while every substitute is rescaled to its corresponding raw query-token norm.",
            f"Coordinate colouring uses {args.group_size}×{args.group_size} groups; each token is replaced exactly once.",
            "",
            "## 6. Substitute feature audit",
            "",
            "See `feature_ood_summary.csv` and token-level feature norm/distance columns.",
            "",
            "## 7. Identity verification",
            "",
            f"- median absolute response: {sanity_summary.get('identity_response_mean')}",
            f"- max absolute response: {sanity_summary.get('identity_response_max')}",
            "",
            "## 8. Spillover analysis",
            "",
            "See `spillover_summary.csv`.",
            "",
            "## 9. TP vs FP results",
            "",
            "See `response_metrics.csv`, `incremental_metrics.csv`, and `bootstrap_intervals.csv`.",
            "",
            "## 10. FN vs TN results",
            "",
            "See the same fixed statistical outputs; FN/TN is never assigned a different response formula.",
            "",
            "## 11. Incremental information analysis",
            "",
            "Balanced L2 logistic regression uses image-grouped 5-fold CV with fold-local scaling.",
            "",
            "## 12. Weighted vs random comparison",
            "",
            "See `substitution_comparison.csv`.",
            "",
            "## 13. Group-size robustness",
            "",
            "See `group_robustness.csv`; a complementary run is required for a full PASS verdict.",
            "",
            "## 14. Area-quartile robustness",
            "",
            "See `area_quartile_summary.csv`.",
            "",
            "## 15. Cross-dataset transfer",
            "",
            "See `cross_dataset_transfer.csv`.",
            "",
            "## 16. Failure cases",
            "",
            "See `failed_samples.csv`.",
            "",
            "## 17. Final decision",
            "",
            f"`{verdict.get('verdict')}`",
            "",
            "GT image bytes were loaded only after all original and counterfactual forwards for each image. "
            "GT was used solely for offline labels and stratified statistics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer")
    if args.visualization_count < 0:
        raise ValueError("--visualization_count must be non-negative")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap_replicates must be positive")
    full_run = args.max_samples == -1
    set_seed(args.seed)

    cfg_path = _resolve(args.config)
    checkpoint_paths = [_resolve(path) for path in args.checkpoints]
    feature_root = _resolve(args.feature_root)
    dabe_root = _resolve(args.dabe_root)
    output_dir = _resolve(args.output_dir)
    robustness_reference = _resolve(args.robustness_reference) if args.robustness_reference else None
    for path in [cfg_path, *checkpoint_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (feature_root, dabe_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    _prepare_output(output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu explicitly")
    device = torch.device(args.device)
    cfg = load_config(cfg_path)
    samples = _match_manifests(feature_root, dabe_root, args.datasets, args.max_samples)
    group_ids_cpu = coordinate_group_ids(TOKEN_GRID, TOKEN_GRID, args.group_size)
    group_ids = group_ids_cpu.to(device)

    token_path = output_dir / "token_metrics.parquet"
    token_rows: list[dict[str, Any]] = []
    token_writer = _TokenParquetWriter(token_path) if full_run else None
    image_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    feature_norms: list[float] = []
    anchor_counts: list[int] = []
    visual_count = 0
    visual_candidates: dict[tuple[int, str], list[dict[str, Any]]] = {}
    visualization_manifest_rows: list[dict[str, Any]] = []
    peak_gpu_memory = 0

    for checkpoint_path in checkpoint_paths:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        decoder, checkpoint_info = _load_decoder(
            checkpoint_path, cfg, device, args.use_teacher
        )
        epoch = int(checkpoint_info["checkpoint_epoch"])
        for sample in samples:
            sample_start = time.perf_counter()
            try:
                feature_payload = torch.load(
                    sample["feature_path"], map_location="cpu", weights_only=False
                )
                feature_cpu = _extract_feature(feature_payload, sample)
                dabe = _load_dabe(sample["dabe_path"], sample)
                image_68_cpu = _load_image_68(sample["image_path"], 68).detach().float()
                rgb_grid_cpu = _load_rgb_grid(sample["image_path"], TOKEN_GRID).detach().float()
                feature = feature_cpu.to(device)
                image_68 = image_68_cpu.to(device)
                rgb_grid = rgb_grid_cpu.to(device)
                bg_anchor = _field_37(dabe, "bg_anchor_37").to(device)
                substitution = build_dabe_background_substitutes(
                    feature,
                    rgb_grid,
                    bg_anchor,
                    dabe["params"],
                    args.substitution_modes,
                    seed=args.seed,
                    sample_key=sample["image_id"],
                    norm_mode=args.substitute_norm_mode,
                )
                diagnostics = substitution.diagnostics
                feature_norms.append(float(feature_cpu.norm(dim=0).mean().item()))
                anchor_counts.append(int(diagnostics["background_anchor_count"]))

                # Keep the reference and counterfactual batch geometry identical.
                # FP16/CuDNN may otherwise choose different kernels for B=1 and
                # B=G, which creates a spurious non-zero Identity response.
                if args.batch_interventions:
                    group_count = args.group_size * args.group_size
                    original_feature_batch = (
                        feature.unsqueeze(0)
                        .expand(group_count, -1, -1, -1)
                        .clone()
                    )
                    original_image_batch = image_68.unsqueeze(0).expand(
                        group_count, -1, -1, -1
                    )
                else:
                    original_feature_batch = feature.unsqueeze(0)
                    original_image_batch = image_68.unsqueeze(0)
                original = _forward(
                    decoder,
                    original_feature_batch,
                    original_image_batch,
                    device,
                )
                original_logits = original[args.primary_output][0, 0].float()
                mode_responses: dict[str, dict[str, torch.Tensor]] = {}
                for mode in args.substitution_modes:
                    counterfactual_feature = apply_group_interventions(
                        feature, substitution.substitutes[mode], group_ids
                    )
                    if counterfactual_feature.requires_grad:
                        raise RuntimeError("Counterfactual feature unexpectedly requires grad")
                    if mode == "identity" and not torch.equal(
                        counterfactual_feature,
                        feature.unsqueeze(0).expand_as(counterfactual_feature),
                    ):
                        raise RuntimeError("Identity mode changed one or more feature values")
                    counterfactual_output = _counterfactual_forward(
                        decoder,
                        counterfactual_feature,
                        image_68,
                        device,
                        args.batch_interventions,
                    )
                    counterfactual_logits = counterfactual_output[args.primary_output][:, 0].float()
                    responses = assemble_group_responses(
                        original_logits, counterfactual_logits, group_ids
                    )
                    if mode == "identity":
                        identity_median = float(
                            responses["c_self_raw"].abs().median().item()
                        )
                        identity_max = float(
                            responses["c_self_raw"].abs().max().item()
                        )
                        if identity_median >= 1e-6 or identity_max >= 1e-4:
                            raise RuntimeError(
                                "Identity response invariant failed: "
                                f"median_abs={identity_median:.9g}, "
                                f"max_abs={identity_max:.9g}"
                            )
                    mode_responses[mode] = responses

                # Ordering barrier: no GT bytes are opened before all modes finish.
                all_forwards_finished_at = time.perf_counter()
                gt_array, gt_tensor = _load_gt_after_forwards(sample["gt_path"])
                gt_loaded_at = time.perf_counter()
                gt_occupancy = downsample_gt_occupancy(gt_tensor, TOKEN_GRID, TOKEN_GRID).to(device)
                gt_strict = strict_gt_labels(gt_occupancy)
                teacher_probability = torch.sigmoid(original_logits)
                error_map = classify_teacher_errors(teacher_probability, gt_strict)

                elapsed = time.perf_counter() - sample_start
                error_counts = Counter(
                    ERROR_NAMES[int(value)]
                    for value in error_map.detach().cpu().reshape(-1).tolist()
                )
                completed_sample_token_rows: list[dict[str, Any]] = []
                completed_sample_image_rows: list[dict[str, Any]] = []
                for mode in args.substitution_modes:
                    responses = mode_responses[mode]
                    current_token_rows = _token_rows(
                        sample=sample,
                        checkpoint_epoch=epoch,
                        group_size=args.group_size,
                        mode=mode,
                        substitute_norm_mode=args.substitute_norm_mode,
                        group_ids=group_ids,
                        teacher_logits=original_logits,
                        teacher_prob=teacher_probability,
                        gt_occupancy=gt_occupancy,
                        gt_strict=gt_strict,
                        error_map=error_map,
                        dabe=dabe,
                        diagnostics=diagnostics,
                        responses=responses,
                    )
                    completed_sample_token_rows.extend(current_token_rows)
                    completed_sample_image_rows.append(
                        {
                            "dataset": sample["dataset"],
                            "image_id": sample["image_id"],
                            "image_stem": sample["stem"],
                            "checkpoint_epoch": epoch,
                            "substitution_mode": mode,
                            "substitute_norm_mode": args.substitute_norm_mode,
                            "group_size": args.group_size,
                            "background_anchor_count": int(diagnostics["background_anchor_count"]),
                            "target_area": float(gt_occupancy.mean().item()),
                            "teacher_area": float((teacher_probability > 0.5).float().mean().item()),
                            "tp_count": int(error_counts["TP"]),
                            "fp_count": int(error_counts["FP"]),
                            "fn_count": int(error_counts["FN"]),
                            "tn_count": int(error_counts["TN"]),
                            "ignore_count": int(error_counts["IGNORE"]),
                            "spill_ratio_mean": float(responses["spill_ratio_per_group"].mean().item()),
                            "spill_ratio_median": float(responses["spill_ratio_per_group"].median().item()),
                            "seconds": elapsed,
                            "gt_loaded_after_interventions": bool(gt_loaded_at >= all_forwards_finished_at),
                        }
                    )

                # Commit one image atomically after every mode has produced a
                # valid aligned token table.  A failed mode cannot leave a
                # partial sample in the streaming Parquet output.
                if full_run:
                    token_writer.write(completed_sample_token_rows)
                else:
                    token_rows.extend(completed_sample_token_rows)
                image_rows.extend(completed_sample_image_rows)

                if args.save_visualizations:
                    weighted = mode_responses.get("weighted_bg")
                    random_response = mode_responses.get("random_bg")
                    if weighted is not None and random_response is not None:
                        final_prob = None
                        if torch.is_tensor(original.get("logits")):
                            final_prob = torch.sigmoid(original["logits"][0, 0]).float().cpu().numpy()
                        payload = {
                            "rgb": _load_rgb_original(sample["image_path"]),
                            "gt": gt_array,
                            "dabe_target_soft": _field_37(dabe, "target_soft_37").numpy(),
                            "dabe_residual": _field_37(dabe, "residual_37").numpy(),
                            "teacher_prob": teacher_probability.cpu().numpy(),
                            "teacher_error": error_map.float().cpu().numpy(),
                            "weighted_self": weighted["c_self_centered"].cpu().numpy(),
                            "weighted_local": weighted["c_local_centered"].cpu().numpy(),
                            "random_local": random_response["c_local_centered"].cpu().numpy(),
                            "spillover": weighted["spill_ratio_map"].cpu().numpy(),
                            "group_map": group_ids_cpu.numpy(),
                            "final_prob": final_prob,
                            "title": f"{sample['image_id']} | EMA Teacher epoch {epoch}",
                        }
                        if full_run and args.visualization_count > 0:
                            scores = _visualization_category_scores(
                                weighted["c_local_centered"],
                                random_response["c_local_centered"],
                                error_map,
                                weighted["spill_ratio_map"],
                            )
                            metadata = {
                                "dataset": sample["dataset"],
                                "image_id": sample["image_id"],
                                "image_stem": sample["stem"],
                                "checkpoint_epoch": epoch,
                                "teacher_area": float(
                                    (teacher_probability > 0.5).float().mean().item()
                                ),
                                "gt_area": float(gt_occupancy.mean().item()),
                                "spill_ratio": float(
                                    weighted["spill_ratio_map"].median().item()
                                ),
                            }
                            for category, (score, contribution) in scores.items():
                                _offer_visualization_candidate(
                                    visual_candidates,
                                    epoch=epoch,
                                    category=category,
                                    score=score,
                                    contribution=contribution,
                                    payload=payload,
                                    metadata=metadata,
                                    limit=min(5, max(1, args.visualization_count)),
                                )
                        elif visual_count < args.visualization_count:
                            output_path = (
                                output_dir
                                / "visualizations"
                                / f"e{epoch:03d}_{sample['dataset']}_{sample['stem']}.png"
                            )
                            render_bisa_visualization(payload, output_path)
                            visualization_manifest_rows.append(
                                {
                                    "path": str(output_path),
                                    "dataset": sample["dataset"],
                                    "image_stem": sample["stem"],
                                    "checkpoint_epoch": epoch,
                                    "category": "sanity_manifest_order",
                                    "teacher_area": float(
                                        (teacher_probability > 0.5).float().mean().item()
                                    ),
                                    "gt_area": float(gt_occupancy.mean().item()),
                                    "response_contribution": math.nan,
                                    "spill_ratio": float(
                                        weighted["spill_ratio_map"].median().item()
                                    ),
                                }
                            )
                            visual_count += 1
            except Exception as exc:
                failed_rows.append(
                    {
                        "dataset": sample["dataset"],
                        "image_id": sample["image_id"],
                        "checkpoint_epoch": epoch,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"[BISA][FAILED] epoch={epoch} image={sample['image_id']}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        checkpoint_info["state_signature_after"] = _state_signature(decoder)
        checkpoint_info["checkpoint_sha256_after"] = _sha256(checkpoint_path)
        checkpoint_info["checkpoint_modified"] = bool(
            checkpoint_info["checkpoint_sha256_before"]
            != checkpoint_info["checkpoint_sha256_after"]
        )
        checkpoint_info["model_state_modified"] = bool(
            checkpoint_info["state_signature_before"]
            != checkpoint_info["state_signature_after"]
        )
        if checkpoint_info["checkpoint_modified"] or checkpoint_info["model_state_modified"]:
            raise RuntimeError(f"Read-only invariant failed for {checkpoint_path}")
        if device.type == "cuda":
            peak_gpu_memory = max(peak_gpu_memory, int(torch.cuda.max_memory_allocated(device)))
        checkpoint_rows.append(checkpoint_info)
        del decoder

    if full_run and args.save_visualizations:
        for (epoch, category), candidates in sorted(visual_candidates.items()):
            for rank, candidate in enumerate(candidates, start=1):
                contribution = float(candidate["contribution"])
                contribution_text = (
                    f"{contribution:+.4f}" if math.isfinite(contribution) else "NA"
                )
                filename = (
                    f"e{epoch:03d}_{candidate['dataset']}_{candidate['image_stem']}_"
                    f"{category}_rank{rank}_ta{candidate['teacher_area']:.4f}_"
                    f"ga{candidate['gt_area']:.4f}_c{contribution_text}_"
                    f"spill{candidate['spill_ratio']:.4f}.png"
                )
                output_path = output_dir / "visualizations" / filename
                render_bisa_visualization(candidate["payload"], output_path)
                visualization_manifest_rows.append(
                    {
                        "path": str(output_path),
                        "dataset": candidate["dataset"],
                        "image_stem": candidate["image_stem"],
                        "checkpoint_epoch": epoch,
                        "category": category,
                        "teacher_area": candidate["teacher_area"],
                        "gt_area": candidate["gt_area"],
                        "response_contribution": contribution,
                        "spill_ratio": candidate["spill_ratio"],
                    }
                )
                visual_count += 1

    if token_writer is not None:
        token_writer.close()
    completed_token_rows = token_writer.row_count if token_writer is not None else len(token_rows)
    if completed_token_rows == 0:
        raise RuntimeError(f"No samples completed successfully; failures={failed_rows}")
    image_frame = pd.DataFrame(image_rows)
    failed_frame = pd.DataFrame(
        failed_rows,
        columns=["dataset", "image_id", "checkpoint_epoch", "error_type", "error"],
    )
    if not bool(image_frame["gt_loaded_after_interventions"].all()):
        raise RuntimeError("GT ordering invariant failed")
    if full_run:
        statistics = build_statistical_outputs_from_parquet(
            token_path,
            image_frame,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
            robustness_reference=robustness_reference,
        )
        token_frame = None
    else:
        token_frame = pd.DataFrame(token_rows)
        if not np.isfinite(
            token_frame.select_dtypes(include=[np.number]).to_numpy()
        ).all():
            raise RuntimeError("Token table contains NaN/Inf numeric values")
        statistics = build_statistical_outputs(
            token_frame,
            image_frame,
            seed=args.seed,
            full_run=False,
            bootstrap_replicates=args.bootstrap_replicates,
            robustness_reference=robustness_reference,
        )
    ema_epochs = {
        int(item["checkpoint_epoch"])
        for item in checkpoint_rows
        if item["model_source"] == "ema_teacher"
    }
    verdict = statistics["verdict"]
    if full_run and len(ema_epochs) < 2:
        verdict = {
            "verdict": "INCOMPLETE_NO_EMA_TEACHER",
            "reason": "A complete Teacher verdict requires at least two EMA Teacher checkpoints",
            "fixed_criteria_verdict": statistics["verdict"],
        }

    # Parquet is mandatory; no deceptive CSV-with-parquet-extension fallback.
    if not full_run:
        try:
            token_frame.to_parquet(token_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "A real Parquet engine is required. Install pyarrow into the isolated "
                f"path {LOCAL_DEPENDENCY_ROOT} and rerun."
            ) from exc
    _write_csv(image_frame, output_dir / "image_metrics.csv")
    _write_csv(
        pd.DataFrame(
            visualization_manifest_rows,
            columns=[
                "path",
                "dataset",
                "image_stem",
                "checkpoint_epoch",
                "category",
                "teacher_area",
                "gt_area",
                "response_contribution",
                "spill_ratio",
            ],
        ),
        output_dir / "visualization_manifest.csv",
    )
    _write_csv(failed_frame, output_dir / "failed_samples.csv")
    for name in (
        "response_metrics",
        "incremental_metrics",
        "substitution_comparison",
        "group_robustness",
        "area_quartile_summary",
        "spillover_summary",
        "feature_ood_summary",
        "cross_dataset_transfer",
        "bootstrap_intervals",
        "paired_effects",
    ):
        _write_csv(statistics[name], output_dir / f"{name}.csv")
    _write_csv(statistics["response_metrics"], output_dir / "dataset_summary.csv")
    checkpoint_summary = (
        image_frame.groupby(
            ["checkpoint_epoch", "dataset", "substitution_mode", "group_size"],
            as_index=False,
            observed=True,
        )
        .agg(
            image_count=("image_id", "nunique"),
            seconds_per_image=("seconds", "mean"),
            spill_ratio_median=("spill_ratio_median", "median"),
            background_anchor_count_mean=("background_anchor_count", "mean"),
        )
    )
    _write_csv(checkpoint_summary, output_dir / "checkpoint_summary.csv")

    if full_run:
        sanity = {
            "mode": "full_audit",
            "sample_count": int(image_frame["image_id"].nunique()),
            "model_sources": sorted(
                set(item["model_source"] for item in checkpoint_rows)
            ),
            "identity_response_median_absolute": statistics["aggregate"].get(
                "identity_median_absolute_response"
            ),
            "identity_response_max_absolute": statistics["aggregate"].get(
                "identity_max_absolute_response"
            ),
            "teacher_has_grad": False,
            "dino_has_grad": False,
            "checkpoint_modified": False,
            "gt_loaded_after_interventions": True,
            "seconds_per_image": float(image_frame["seconds"].mean()),
            "peak_gpu_memory_bytes": int(peak_gpu_memory),
        }
    else:
        sanity = _sanity_summary(
            token_frame,
            image_frame,
            checkpoint_rows,
            args.group_size,
            feature_norms,
            anchor_counts,
            peak_gpu_memory,
        )
    sanity["substitute_norm_mode"] = args.substitute_norm_mode
    statistics["aggregate"].update(
        {
            "failed_sample_checkpoint_pairs": int(len(failed_rows)),
            "peak_gpu_memory_bytes": int(peak_gpu_memory),
            "seconds_per_image": float(image_frame["seconds"].mean()),
            "decoder_forwards_per_image": int(
                1 + args.group_size * args.group_size * len(args.substitution_modes)
            ),
            "teacher_has_grad": False,
            "dino_has_grad": False,
            "checkpoint_modified": False,
            "gt_loaded_after_interventions": True,
        }
    )
    run_config = {
        "resolved_config_path": str(cfg_path),
        "resolved_config": config_to_dict(cfg),
        "checkpoints": checkpoint_rows,
        "datasets": args.datasets,
        "feature_root": str(feature_root),
        "dabe_root": str(dabe_root),
        "output_dir": str(output_dir),
        "group_size": args.group_size,
        "substitution_modes": args.substitution_modes,
        "substitute_norm_mode": args.substitute_norm_mode,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "device": str(device),
        "batch_interventions": args.batch_interventions,
        "save_visualizations": args.save_visualizations,
        "visualization_count": args.visualization_count,
        "use_teacher": args.use_teacher,
        "primary_output": args.primary_output,
        "robustness_reference": str(robustness_reference) if robustness_reference else None,
        "bootstrap_replicates": args.bootstrap_replicates,
        "full_run": full_run,
        "manifest_sample_count": len(samples),
        "successful_image_checkpoint_pairs": int(
            image_frame[["image_id", "checkpoint_epoch"]].drop_duplicates().shape[0]
        ),
        "local_parquet_dependency_root": str(LOCAL_DEPENDENCY_ROOT),
    }
    _write_json(statistics["aggregate"], output_dir / "aggregate_metrics.json")
    _write_json(verdict, output_dir / "verdict.json")
    _write_json(run_config, output_dir / "run_config_resolved.json")
    _write_json(sanity, output_dir / "sanity_summary.json")
    _report(
        path=output_dir / "REPORT.md",
        verdict=verdict,
        cfg_path=cfg_path,
        checkpoint_metadata=checkpoint_rows,
        args=args,
        sample_count=int(image_frame["image_id"].nunique()),
        failed_count=len(failed_rows),
        sanity_summary=sanity,
    )
    print(json.dumps(json_finite(sanity), ensure_ascii=False, indent=2), flush=True)
    print(f"[BISA] output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
