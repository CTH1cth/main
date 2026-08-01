#!/usr/bin/env python3
"""Strict no-GT preflight for the EAOGP DABE-v2 evidence payload."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.utils import load_config, read_jsonl, torch_load  # noqa: E402


REQUIRED_TENSORS = {
    "p_dabe_68": (1, 68, 68),
    "bc_map_37": (1, 37, 37),
    "residual_norm_37": (1, 37, 37),
}


def _resolve_from_main(path_value: str) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = MAIN_ROOT / path
    return path.resolve()


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "workdir" / "eaogp_v1_cache_audit"),
    )
    args = parser.parse_args()

    config_path = _resolve_from_main(args.config)
    cfg = load_config(str(config_path))
    cache_root = _resolve_from_main(cfg.DABE_CLEAN_DABE_V2_ROOT)
    manifest_path = cache_root / "manifest_train.jsonl"
    rows = read_jsonl(manifest_path)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    by_sample_path = output_root / "by_sample.csv"
    summary_path = output_root / "summary.json"

    missing_counter: Counter[str] = Counter()
    identity_mismatch_count = 0
    range_violation_count = 0
    finite_violation_count = 0
    shape_violation_count = 0
    training_gt_report_count = 0
    authorized_residual_alias_count = 0
    static_target_consistency_error = 0.0
    foreground_sum = 0.0
    foreground_count = 0
    background_sum = 0.0
    background_count = 0
    anchor_sum = 0.0
    anchor_count = 0
    anchor_fg_sum = 0.0
    anchor_fg_count = 0
    anchor_bg_sum = 0.0
    anchor_bg_count = 0
    sample_records = []

    expected_backbone = str(cfg.BACKBONE_KEY)
    for row_index, row in enumerate(rows):
        dataset = str(row.get("dataset", ""))
        stem = str(row.get("stem", ""))
        cache_path = Path(str(row.get("cache_path", ""))).expanduser()
        if not cache_path.is_absolute():
            cache_path = manifest_path.parent / cache_path
        cache_path = cache_path.resolve()
        record = {
            "row_index": row_index,
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(cache_path),
            "missing_keys": "",
            "identity_ok": False,
            "shape_ok": False,
            "finite_ok": False,
            "range_ok": False,
            "training_gt_read": False,
            "residual_source_key": "",
            "foreground_response_mean": "",
            "background_evidence_mean": "",
            "anchor_confidence_mean": "",
        }
        try:
            payload = torch_load(cache_path, map_location="cpu")
        except Exception as error:  # pragma: no cover - reported as audit data
            missing_counter["<payload_unreadable>"] += 1
            record["missing_keys"] = "<payload_unreadable>"
            record["error"] = repr(error)
            sample_records.append(record)
            continue
        if not isinstance(payload, dict):
            missing_counter["<payload_not_dict>"] += 1
            record["missing_keys"] = "<payload_not_dict>"
            sample_records.append(record)
            continue

        identity_ok = (
            str(payload.get("dataset", "")) == dataset
            and str(payload.get("stem", "")) == stem
            and str(payload.get("backbone_key", "")) == expected_backbone
            and str(payload.get("dabe_version", "")).strip().lower() == "v2"
        )
        record["identity_ok"] = identity_ok
        if not identity_ok:
            identity_mismatch_count += 1

        training_gt_read = bool(payload.get("training_gt_read", False))
        record["training_gt_read"] = training_gt_read
        if training_gt_read:
            training_gt_report_count += 1

        residual_source_key = (
            "residual_norm_37"
            if torch.is_tensor(payload.get("residual_norm_37"))
            else (
                "residual_37"
                if torch.is_tensor(payload.get("residual_37"))
                else ""
            )
        )
        record["residual_source_key"] = residual_source_key
        if residual_source_key == "residual_37":
            authorized_residual_alias_count += 1
        missing = [
            key
            for key in ("p_dabe_68", "bc_map_37")
            if not torch.is_tensor(payload.get(key))
        ]
        if not residual_source_key:
            missing.append("residual_norm_37")
        for key in missing:
            missing_counter[key] += 1
        record["missing_keys"] = ";".join(missing)
        if missing:
            sample_records.append(record)
            continue

        tensors = {
            "p_dabe_68": payload["p_dabe_68"].detach().cpu().float(),
            "bc_map_37": payload["bc_map_37"].detach().cpu().float(),
            "residual_norm_37": payload[residual_source_key]
            .detach()
            .cpu()
            .float(),
        }
        shape_ok = all(
            tuple(tensors[key].shape) == shape
            for key, shape in REQUIRED_TENSORS.items()
        )
        record["shape_ok"] = shape_ok
        if not shape_ok:
            shape_violation_count += 1

        finite_ok = all(bool(torch.isfinite(value).all().item()) for value in tensors.values())
        record["finite_ok"] = finite_ok
        if not finite_ok:
            finite_violation_count += 1

        range_ok = finite_ok and all(
            float(value.min().item()) >= 0.0 and float(value.max().item()) <= 1.0
            for value in tensors.values()
        )
        record["range_ok"] = range_ok
        if not range_ok:
            range_violation_count += 1
        if not (shape_ok and finite_ok and range_ok):
            sample_records.append(record)
            continue

        foreground = tensors["p_dabe_68"]
        bc_map = tensors["bc_map_37"]
        residual = tensors["residual_norm_37"]
        static_target = (foreground > 0.5).float()
        consistency = float(
            (static_target - (foreground > 0.5).float()).abs().max().item()
        )
        static_target_consistency_error = max(
            static_target_consistency_error,
            consistency,
        )
        background_37 = (bc_map * (1.0 - residual)).clamp(0.0, 1.0)
        background = F.interpolate(
            background_37.unsqueeze(0),
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        anchor = (
            static_target * foreground * (1.0 - background)
            + (1.0 - static_target) * background * (1.0 - foreground)
        ).clamp(0.0, 1.0)

        foreground_sum += float(foreground.double().sum().item())
        foreground_count += int(foreground.numel())
        background_sum += float(background.double().sum().item())
        background_count += int(background.numel())
        anchor_sum += float(anchor.double().sum().item())
        anchor_count += int(anchor.numel())
        fg_mask = static_target > 0.5
        bg_mask = ~fg_mask
        anchor_fg_sum += float(anchor[fg_mask].double().sum().item())
        anchor_fg_count += int(fg_mask.sum().item())
        anchor_bg_sum += float(anchor[bg_mask].double().sum().item())
        anchor_bg_count += int(bg_mask.sum().item())
        record["foreground_response_mean"] = _safe_float(foreground.mean().item())
        record["background_evidence_mean"] = _safe_float(background.mean().item())
        record["anchor_confidence_mean"] = _safe_float(anchor.mean().item())
        sample_records.append(record)

    fieldnames = sorted({key for row in sample_records for key in row})
    with by_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_records)

    missing_key_count = int(sum(missing_counter.values()))
    passed = (
        len(rows) == 4040
        and missing_key_count == 0
        and identity_mismatch_count == 0
        and shape_violation_count == 0
        and range_violation_count == 0
        and finite_violation_count == 0
        and training_gt_report_count == 0
        and static_target_consistency_error == 0.0
    )
    summary = {
        "schema": "eaogp_v1_dabe_v2_cache_preflight",
        "status": "PASS" if passed else "FAIL",
        "config": str(config_path),
        "cache_root": str(cache_root),
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "training_gt_read": training_gt_report_count > 0,
        "training_gt_report_count": training_gt_report_count,
        "residual_field_contract": (
            "residual_norm_37_or_user_authorized_normalized_residual_37_alias"
        ),
        "authorized_residual_alias_count": authorized_residual_alias_count,
        "missing_key_count": missing_key_count,
        "missing_by_key": dict(sorted(missing_counter.items())),
        "identity_mismatch_count": identity_mismatch_count,
        "shape_violation_count": shape_violation_count,
        "range_violation_count": range_violation_count,
        "finite_violation_count": finite_violation_count,
        "static_target_consistency_error": static_target_consistency_error,
        "foreground_response_mean": (
            foreground_sum / foreground_count if foreground_count else None
        ),
        "background_evidence_mean": (
            background_sum / background_count if background_count else None
        ),
        "anchor_confidence_mean": anchor_sum / anchor_count if anchor_count else None,
        "anchor_confidence_on_static_fg": (
            anchor_fg_sum / anchor_fg_count if anchor_fg_count else None
        ),
        "anchor_confidence_on_static_bg": (
            anchor_bg_sum / anchor_bg_count if anchor_bg_count else None
        ),
        "by_sample_csv": str(by_sample_path),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
