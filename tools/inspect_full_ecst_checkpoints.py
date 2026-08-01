#!/usr/bin/env python3
"""Inventory Full-ECST checkpoints without constructing a model or dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CTH_ROOT = MAIN_ROOT.parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.utils import config_to_dict, load_config, make_jsonable, torch_load  # noqa: E402


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
DEFAULT_EPOCHS = (7, 10, 15, 19, 20, 24, 25, 27)


def _inside_cth(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if resolved != CTH_ROOT and CTH_ROOT not in resolved.parents:
        raise RuntimeError(f"Path must stay inside {CTH_ROOT}: {resolved}")
    return resolved


def _epochs(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("--epochs must contain positive integers.")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(config: Any) -> str:
    encoded = json.dumps(
        make_jsonable(config_to_dict(config)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        make_jsonable(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _memory_state(checkpoint: dict[str, Any]) -> tuple[str | None, Any]:
    for key in (
        "ecst_temporal_memory",
        "clean_ecst_temporal_memory",
        "temporal_teacher_memory",
        "ecst_memory",
    ):
        if key in checkpoint and checkpoint[key] is not None:
            return key, checkpoint[key]
    return None, None


def _memory_summary(state: Any) -> dict[str, Any]:
    result = {
        "ecst_memory_present": False,
        "ecst_memory_key": "",
        "ecst_memory_valid": False,
        "ecst_memory_num_samples": None,
        "ecst_memory_shape": "",
        "ecst_memory_count_min": None,
        "ecst_memory_count_mean": None,
        "ecst_memory_count_max": None,
        "ecst_memory_error": "",
    }
    if state is None:
        return result
    result["ecst_memory_present"] = True
    if not isinstance(state, dict):
        result["ecst_memory_error"] = "memory state is not a dict"
        return result
    mean = state.get("mean")
    second = state.get("second")
    count = state.get("count")
    errors = []
    if not torch.is_tensor(mean) or not torch.is_tensor(second):
        errors.append("mean/second missing")
    elif tuple(mean.shape) != tuple(second.shape):
        errors.append("mean/second shape mismatch")
    elif mean.ndim != 4 or tuple(mean.shape[1:]) != (1, 68, 68):
        errors.append(f"invalid mean shape {list(mean.shape)}")
    elif not bool(torch.isfinite(mean).all().item()) or not bool(
        torch.isfinite(second).all().item()
    ):
        errors.append("mean/second non-finite")
    if not torch.is_tensor(count) or count.ndim != 1:
        errors.append("count missing or invalid")
    elif torch.is_tensor(mean) and int(count.numel()) != int(mean.shape[0]):
        errors.append("count/sample mismatch")
    elif bool((count < 0).any().item()):
        errors.append("negative count")
    if torch.is_tensor(mean):
        result["ecst_memory_num_samples"] = int(mean.shape[0])
        result["ecst_memory_shape"] = json.dumps(list(mean.shape))
    if torch.is_tensor(count) and count.numel():
        result.update(
            {
                "ecst_memory_count_min": int(count.min().item()),
                "ecst_memory_count_mean": float(count.float().mean().item()),
                "ecst_memory_count_max": int(count.max().item()),
            }
        )
    result["ecst_memory_valid"] = not errors
    result["ecst_memory_error"] = "; ".join(errors)
    return result


def inspect_checkpoints(
    *,
    config_path: Path,
    checkpoint_root: Path,
    output_root: Path,
    epochs: list[int],
) -> dict[str, Any]:
    cfg = load_config(config_path)
    fingerprint = _config_fingerprint(cfg)
    rows = []
    for epoch in epochs:
        path = checkpoint_root / f"epoch_{epoch:03d}.pth"
        row: dict[str, Any] = {
            "requested_epoch": int(epoch),
            "checkpoint_path": str(path),
            "checkpoint_exists": path.is_file(),
            "saved_epoch": None,
            "student_state_present": False,
            "teacher_state_present": False,
            "optimizer_state_present": False,
            "scheduler_state_present": False,
            "checkpoint_config_present": False,
            "config_fingerprint": fingerprint,
            "checkpoint_config_fingerprint": "",
            "checkpoint_config_fingerprint_matches_requested": None,
            "ecst_memory_expected_sample_count": None,
            "checkpoint_sha256": "",
        }
        if path.is_file():
            checkpoint = torch_load(path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise RuntimeError(f"Checkpoint is not a dict: {path}")
            row.update(
                {
                    "saved_epoch": int(checkpoint.get("epoch", -1)),
                    "student_state_present": isinstance(checkpoint.get("student"), dict),
                    "teacher_state_present": isinstance(checkpoint.get("teacher"), dict),
                    "optimizer_state_present": isinstance(checkpoint.get("optimizer"), dict),
                    "scheduler_state_present": isinstance(checkpoint.get("scheduler"), dict),
                    "checkpoint_config_present": isinstance(checkpoint.get("config"), dict),
                    "checkpoint_sha256": _sha256(path),
                }
            )
            if isinstance(checkpoint.get("config"), dict):
                checkpoint_fingerprint = _mapping_fingerprint(checkpoint["config"])
                row["checkpoint_config_fingerprint"] = checkpoint_fingerprint
                row["checkpoint_config_fingerprint_matches_requested"] = (
                    checkpoint_fingerprint == fingerprint
                )
            memory_key, memory = _memory_state(checkpoint)
            row.update(_memory_summary(memory))
            row["ecst_memory_key"] = memory_key or ""
            row["ecst_memory_expected_sample_count"] = (
                int(row["ecst_memory_num_samples"] or -1) == 4040
                if row["ecst_memory_present"]
                else None
            )
        else:
            row.update(_memory_summary(None))
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "checkpoint_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "full_ecst_checkpoint_inventory_v1",
        "config": str(config_path),
        "checkpoint_root": str(checkpoint_root),
        "config_fingerprint": fingerprint,
        "requested_epochs": epochs,
        "all_requested_checkpoints_present": all(
            bool(row["checkpoint_exists"]) for row in rows
        ),
        "all_checkpoints_have_complete_model_state": all(
            bool(row["student_state_present"])
            and bool(row["teacher_state_present"])
            and bool(row["optimizer_state_present"])
            and bool(row["scheduler_state_present"])
            for row in rows
            if row["checkpoint_exists"]
        ),
        "checkpoint_memory_exact": all(
            bool(row["ecst_memory_present"])
            and bool(row["ecst_memory_valid"])
            and int(row["ecst_memory_num_samples"] or -1) == 4040
            for row in rows
            if row["checkpoint_exists"] and int(row["requested_epoch"]) <= 20
        ),
        "memory_replay_required_epochs": [
            int(row["requested_epoch"])
            for row in rows
            if row["checkpoint_exists"]
            and int(row["requested_epoch"]) <= 20
            and not bool(row["ecst_memory_present"])
        ],
        "rows": rows,
        "csv": str(csv_path),
    }
    json_path = output_root / "checkpoint_inventory.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--epochs",
        type=_epochs,
        default=list(DEFAULT_EPOCHS),
    )
    args = parser.parse_args()
    result = inspect_checkpoints(
        config_path=_inside_cth(args.config),
        checkpoint_root=_inside_cth(args.checkpoint_root),
        output_root=_inside_cth(args.output_root),
        epochs=list(args.epochs),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
