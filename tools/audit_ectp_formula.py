#!/usr/bin/env python3
"""Run the GT-free, full-cache formula audit for ECTP v1.

The audit deliberately reads only the independent DABE-v2 cache and the
DABE-Clean evidence cache.  It never opens image, GT, Teacher checkpoint,
legacy-region, or temporal-history files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import numpy as np


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.ectp import (  # noqa: E402
    build_ectp_projected_target,
    validate_ectp_config,
)
from common.dabe_clean import expected_dabe_clean_payload_version  # noqa: E402
from common.utils import (  # noqa: E402
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


DEFAULT_CONFIG = (
    MAIN_ROOT
    / "configs"
    / "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ectp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workdir" / "ectp_v1_offline_audit"
EXPECTED_FULL_SAMPLE_COUNT = 4040
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_NAMES = ("q10", "q25", "q50", "q75", "q90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ECTP v1 on paired DABE-v2/DABE-Clean caches without "
            "reading GT, images, Teacher checkpoints, or legacy ECST data."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dabe-v2-root",
        type=Path,
        default=None,
        help="Override cfg.DABE_CLEAN_DABE_V2_ROOT for diagnostic use.",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=None,
        help="Override cfg.DABE_CLEAN_ROOT for diagnostic use.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=EXPECTED_FULL_SAMPLE_COUNT,
        help="Manifest-order sample count; the default audits all 4040 samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing summary.json and by_sample.csv.",
    )
    return parser.parse_args()


def _resolve_cfg_path(value: Any, *, override: Path | None) -> Path:
    path = override if override is not None else Path(str(value))
    path = path.expanduser()
    if not path.is_absolute():
        path = MAIN_ROOT / path
    return path.resolve()


def _validate_identity(
    payload: dict[str, Any],
    *,
    dataset: str,
    stem: str,
    backbone_key: str,
    cache_path: Path,
) -> None:
    actual = (str(payload.get("dataset")), str(payload.get("stem")))
    expected = (dataset, stem)
    if actual != expected:
        raise RuntimeError(
            f"Cache identity mismatch: {actual} != {expected} | {cache_path}"
        )
    if str(payload.get("backbone_key")) != backbone_key:
        raise RuntimeError(
            "Cache backbone mismatch: "
            f"{payload.get('backbone_key')!r} != {backbone_key!r} | "
            f"{cache_path}"
        )


def _load_tensor(
    payload: dict[str, Any],
    field: str,
    *,
    expected_shape: tuple[int, ...],
    cache_path: Path,
) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Missing tensor {field!r}: {cache_path}")
    value = value.detach().cpu().float()
    if tuple(value.shape) != expected_shape:
        raise RuntimeError(
            f"{field} shape mismatch: {tuple(value.shape)} != "
            f"{expected_shape} | {cache_path}"
        )
    if value.requires_grad:
        raise RuntimeError(f"{field} must be detached: {cache_path}")
    return value


def _range_violation_count(value: torch.Tensor) -> int:
    finite = torch.isfinite(value)
    invalid = (~finite) | (value < 0.0) | (value > 1.0)
    return int(invalid.sum().item())


def _finite_counts(values: Iterable[torch.Tensor]) -> tuple[int, int]:
    finite = 0
    total = 0
    for value in values:
        finite += int(torch.isfinite(value).sum().item())
        total += int(value.numel())
    return finite, total


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    count = int(mask.sum().item())
    if count == 0:
        return None
    return float(value[mask].mean().item())


def _quantile_dict(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().cpu().float().reshape(-1)
    # torch.quantile uses an indexing kernel that rejects the full 4040 x
    # 68 x 68 audit vector on this environment.  NumPy's linear quantile is
    # numerically equivalent here and has no such element-count ceiling.
    if int(flattened.numel()) > 10_000_000:
        result_values = np.quantile(
            flattened.numpy(),
            np.asarray(QUANTILES, dtype=np.float64),
            method="linear",
        ).tolist()
    else:
        result_values = torch.quantile(
            flattened,
            torch.tensor(QUANTILES, dtype=torch.float32),
        ).tolist()
    return {
        name: float(item)
        for name, item in zip(QUANTILE_NAMES, result_values)
    }


def _project(
    *,
    foreground: torch.Tensor,
    background: torch.Tensor,
    static_target: torch.Tensor,
    teacher_binary: torch.Tensor,
) -> torch.Tensor:
    result = build_ectp_projected_target(
        foreground_evidence=foreground,
        background_evidence=background,
        static_target=static_target,
        teacher_binary=teacher_binary,
        static_weight=0.5,
        teacher_weight=0.5,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "build_ectp_projected_target must return "
            "(projected_target, stats)."
        )
    projected, _stats = result
    return projected


def _expected_projection(
    teacher_binary: torch.Tensor,
    static_target: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    conflict = teacher_binary.ne(static_target)
    factor = (1.0 - support * conflict.float()).clamp(0.0, 1.0)
    teacher_sign = 2.0 * teacher_binary - 1.0
    raw = (0.5 + 0.5 * teacher_sign * factor).clamp(0.0, 1.0)
    return torch.where(conflict, raw, teacher_binary).detach()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty by-sample audit.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 1 <= int(args.max_samples) <= EXPECTED_FULL_SAMPLE_COUNT:
        raise ValueError(
            f"--max-samples must be in [1,{EXPECTED_FULL_SAMPLE_COUNT}], "
            f"got {args.max_samples}."
        )

    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    validate_ectp_config(cfg)

    dabe_v2_root = _resolve_cfg_path(
        cfg.DABE_CLEAN_DABE_V2_ROOT,
        override=args.dabe_v2_root,
    )
    clean_root = _resolve_cfg_path(
        cfg.DABE_CLEAN_ROOT,
        override=args.clean_root,
    )
    dabe_v2_manifest = dabe_v2_root / "manifest_train.jsonl"
    clean_manifest = clean_root / "manifest_train.jsonl"
    dabe_v2_rows = read_jsonl(dabe_v2_manifest)
    clean_rows = read_jsonl(clean_manifest)
    dabe_v2_map = manifest_to_map(dabe_v2_rows, dabe_v2_manifest)
    clean_map = manifest_to_map(clean_rows, clean_manifest)

    dabe_keys = set(dabe_v2_map)
    clean_keys = set(clean_map)
    if dabe_keys != clean_keys:
        missing_clean = sorted(dabe_keys - clean_keys)[:10]
        missing_dabe = sorted(clean_keys - dabe_keys)[:10]
        raise RuntimeError(
            "DABE-v2 and DABE-Clean manifests are not key-identical | "
            f"missing_clean={missing_clean} | missing_dabe_v2={missing_dabe}"
        )
    if int(args.max_samples) > len(dabe_v2_rows):
        raise RuntimeError(
            f"Requested {args.max_samples} samples but manifest has "
            f"{len(dabe_v2_rows)}."
        )

    output_root = args.output_root.expanduser().resolve()
    summary_path = output_root / "summary.json"
    by_sample_path = output_root / "by_sample.csv"
    existing = [path for path in (summary_path, by_sample_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "ECTP audit output exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_rows = dabe_v2_rows[: int(args.max_samples)]
    expected_shape = (1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    backbone_key = str(cfg.BACKBONE_KEY)
    threshold = float(cfg.DABE_CLEAN_DABE_V2_HARD_THRESHOLD)
    expected_clean_version = expected_dabe_clean_payload_version(cfg)

    by_sample: list[dict[str, Any]] = []
    all_support: list[torch.Tensor] = []
    support_fg_sum = 0.0
    support_fg_count = 0
    support_bg_sum = 0.0
    support_bg_count = 0
    finite_count = 0
    audited_value_count = 0
    range_violation_count = 0
    crossing_fg_count = 0
    crossing_bg_count = 0
    nonconflict_change_count = 0
    max_static_consistency_error = 0.0
    max_foreground_cache_consistency_error = 0.0
    max_formula_error = 0.0
    max_symmetry_error = 0.0
    max_teacher_fg_symmetry_error = 0.0
    max_teacher_bg_symmetry_error = 0.0

    with torch.no_grad():
        for index, row in enumerate(selected_rows):
            dataset = str(row["dataset"])
            stem = str(row["stem"])
            key = (dataset, stem)
            v2_path = Path(row["cache_path"])
            clean_path = Path(clean_map[key]["cache_path"])
            v2_payload = torch_load(v2_path, map_location="cpu")
            clean_payload = torch_load(clean_path, map_location="cpu")
            if not isinstance(v2_payload, dict) or not isinstance(
                clean_payload, dict
            ):
                raise TypeError(f"Cache payload must be dict: {dataset}/{stem}")
            _validate_identity(
                v2_payload,
                dataset=dataset,
                stem=stem,
                backbone_key=backbone_key,
                cache_path=v2_path,
            )
            _validate_identity(
                clean_payload,
                dataset=dataset,
                stem=stem,
                backbone_key=backbone_key,
                cache_path=clean_path,
            )
            if str(v2_payload.get("dabe_version", "")).lower() != str(
                cfg.DABE_CLEAN_DABE_V2_VERSION
            ).lower():
                raise RuntimeError(
                    f"DABE-v2 version mismatch: {dataset}/{stem} | {v2_path}"
                )
            if str(clean_payload.get("version", "")) != expected_clean_version:
                raise RuntimeError(
                    "DABE-Clean version mismatch: "
                    f"{clean_payload.get('version')!r} != "
                    f"{expected_clean_version!r} | {clean_path}"
                )

            foreground = _load_tensor(
                v2_payload,
                "p_dabe_68",
                expected_shape=expected_shape,
                cache_path=v2_path,
            )
            foreground_clean = _load_tensor(
                clean_payload,
                "foreground_evidence_68",
                expected_shape=expected_shape,
                cache_path=clean_path,
            )
            background = _load_tensor(
                clean_payload,
                "background_evidence_68",
                expected_shape=expected_shape,
                cache_path=clean_path,
            )
            static_target = (foreground > threshold).float().detach()
            canonical_static = (foreground > 0.5).float().detach()

            static_consistency_error = float(
                (static_target - canonical_static).abs().max().item()
            )
            foreground_cache_consistency_error = float(
                (foreground - foreground_clean).abs().max().item()
            )
            max_static_consistency_error = max(
                max_static_consistency_error, static_consistency_error
            )
            max_foreground_cache_consistency_error = max(
                max_foreground_cache_consistency_error,
                foreground_cache_consistency_error,
            )

            support = (
                static_target * foreground * (1.0 - background)
                + (1.0 - static_target)
                * background
                * (1.0 - foreground)
            ).clamp(0.0, 1.0)

            foreground4 = foreground.unsqueeze(0)
            background4 = background.unsqueeze(0)
            static4 = static_target.unsqueeze(0)
            teacher_fg = torch.ones_like(static4)
            teacher_bg = torch.zeros_like(static4)
            projected_fg = _project(
                foreground=foreground4,
                background=background4,
                static_target=static4,
                teacher_binary=teacher_fg,
            )
            projected_bg = _project(
                foreground=foreground4,
                background=background4,
                static_target=static4,
                teacher_binary=teacher_bg,
            )

            expected_fg = _expected_projection(
                teacher_fg, static4, support.unsqueeze(0)
            )
            expected_bg = _expected_projection(
                teacher_bg, static4, support.unsqueeze(0)
            )
            formula_error = max(
                float((projected_fg - expected_fg).abs().max().item()),
                float((projected_bg - expected_bg).abs().max().item()),
            )
            max_formula_error = max(max_formula_error, formula_error)

            # Exact class-complement transformation required by ECTP symmetry.
            # The production builder also enforces the real-data invariant
            # Y0 == (F > 0.5).  A formal F/B class-complement need not remain
            # inside that cache-specific domain, so evaluate the specified
            # closed-form transform here after checking production-vs-formula
            # above on the authoritative inputs.
            complement_static = 1.0 - static4
            complement_support = (
                complement_static * background4 * (1.0 - foreground4)
                + (1.0 - complement_static)
                * foreground4
                * (1.0 - background4)
            ).clamp(0.0, 1.0)
            complement_from_fg = _expected_projection(
                teacher_bg,
                complement_static,
                complement_support,
            )
            complement_from_bg = _expected_projection(
                teacher_fg,
                complement_static,
                complement_support,
            )
            teacher_fg_symmetry_error = float(
                (complement_from_fg - (1.0 - projected_fg))
                .abs()
                .max()
                .item()
            )
            teacher_bg_symmetry_error = float(
                (complement_from_bg - (1.0 - projected_bg))
                .abs()
                .max()
                .item()
            )
            symmetry_error = max(
                teacher_fg_symmetry_error,
                teacher_bg_symmetry_error,
            )
            max_symmetry_error = max(max_symmetry_error, symmetry_error)
            max_teacher_fg_symmetry_error = max(
                max_teacher_fg_symmetry_error,
                teacher_fg_symmetry_error,
            )
            max_teacher_bg_symmetry_error = max(
                max_teacher_bg_symmetry_error,
                teacher_bg_symmetry_error,
            )

            fg_crossings = int((projected_fg < 0.5).sum().item())
            bg_crossings = int((projected_bg > 0.5).sum().item())
            crossing_fg_count += fg_crossings
            crossing_bg_count += bg_crossings
            fg_nonconflict = static4 > 0.5
            bg_nonconflict = static4 < 0.5
            sample_nonconflict_changes = int(
                projected_fg[fg_nonconflict].ne(teacher_fg[fg_nonconflict]).sum().item()
                + projected_bg[bg_nonconflict].ne(teacher_bg[bg_nonconflict]).sum().item()
            )
            nonconflict_change_count += sample_nonconflict_changes

            audited_values = (
                foreground,
                foreground_clean,
                background,
                static_target,
                support,
                projected_fg.squeeze(0),
                projected_bg.squeeze(0),
            )
            sample_finite, sample_total = _finite_counts(audited_values)
            sample_range_violations = sum(
                _range_violation_count(value) for value in audited_values
            )
            finite_count += sample_finite
            audited_value_count += sample_total
            range_violation_count += sample_range_violations

            static_fg = static_target > 0.5
            static_bg = ~static_fg
            fg_count = int(static_fg.sum().item())
            bg_count = int(static_bg.sum().item())
            support_fg_sum += float(support[static_fg].sum().item())
            support_fg_count += fg_count
            support_bg_sum += float(support[static_bg].sum().item())
            support_bg_count += bg_count
            support_quantiles = _quantile_dict(support)
            all_support.append(support.reshape(-1).clone())

            row_out: dict[str, Any] = {
                "sample_index": index,
                "dataset": dataset,
                "stem": stem,
                "static_fg_area": float(static_target.mean().item()),
                "foreground_evidence_mean": float(foreground.mean().item()),
                "background_evidence_mean": float(background.mean().item()),
                "support_mean": float(support.mean().item()),
                "support_on_static_fg_mean": _masked_mean(support, static_fg),
                "support_on_static_bg_mean": _masked_mean(support, static_bg),
                "support_q10": support_quantiles["q10"],
                "support_q25": support_quantiles["q25"],
                "support_q50": support_quantiles["q50"],
                "support_q75": support_quantiles["q75"],
                "support_q90": support_quantiles["q90"],
                "static_target_consistency_error": static_consistency_error,
                "foreground_cache_consistency_error": (
                    foreground_cache_consistency_error
                ),
                "finite_ratio": sample_finite / sample_total,
                "range_violation_count": sample_range_violations,
                "foreground_teacher_crossing_count": fg_crossings,
                "background_teacher_crossing_count": bg_crossings,
                "nonconflict_change_count": sample_nonconflict_changes,
                "formula_max_abs_error": formula_error,
                "symmetry_max_abs_error": symmetry_error,
                "all_foreground_teacher_symmetry_max_abs_error": (
                    teacher_fg_symmetry_error
                ),
                "all_background_teacher_symmetry_max_abs_error": (
                    teacher_bg_symmetry_error
                ),
                "fg_to_bg_conflict_projection_mean": _masked_mean(
                    projected_bg.squeeze(0), static_fg
                ),
                "bg_to_fg_conflict_projection_mean": _masked_mean(
                    projected_fg.squeeze(0), static_bg
                ),
            }
            by_sample.append(row_out)

    support_all = torch.cat(all_support, dim=0)
    global_quantiles = _quantile_dict(support_all)
    total_pixels = int(support_all.numel())
    errors: list[str] = []
    if (
        int(args.max_samples) == EXPECTED_FULL_SAMPLE_COUNT
        and len(dabe_v2_rows) != EXPECTED_FULL_SAMPLE_COUNT
    ):
        errors.append(
            "default full audit requires an exact 4040-row paired manifest"
        )
    if max_static_consistency_error != 0.0:
        errors.append(
            "static target differs from (foreground_evidence > 0.5)"
        )
    if max_foreground_cache_consistency_error > 1e-5:
        errors.append("DABE-v2 and DABE-Clean foreground caches differ > 1e-5")
    if finite_count != audited_value_count:
        errors.append("non-finite values detected")
    if range_violation_count != 0:
        errors.append("[0,1] range violations detected")
    if max_formula_error > 1e-7:
        errors.append("production ECTP output differs from the specified formula")
    if max_symmetry_error > 1e-6:
        errors.append("foreground/background complement symmetry failed")
    if crossing_fg_count or crossing_bg_count:
        errors.append("projected target crossed Teacher's 0.5 boundary")
    if nonconflict_change_count:
        errors.append("non-conflict pixels did not preserve binary Teacher exactly")

    summary: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "audit_scope": "formula_and_data_sources_only_no_performance_claim",
        "config": str(config_path),
        "sample_count": len(by_sample),
        "expected_full_sample_count": EXPECTED_FULL_SAMPLE_COUNT,
        "is_full_4040_audit": (
            len(by_sample) == EXPECTED_FULL_SAMPLE_COUNT
            and len(dabe_v2_rows) == EXPECTED_FULL_SAMPLE_COUNT
        ),
        "manifest_sample_count": len(dabe_v2_rows),
        "pixel_count": total_pixels,
        "dabe_v2_manifest": str(dabe_v2_manifest),
        "dabe_clean_manifest": str(clean_manifest),
        "output_root": str(output_root),
        "inputs_read": [
            "DABE-v2 p_dabe_68",
            "DABE-v2 Hard static target derived as p_dabe_68 > 0.5",
            "DABE-Clean foreground_evidence_68 (consistency audit only)",
            "DABE-Clean background_evidence_68",
        ],
        "forbidden_inputs_read": [],
        "ground_truth_read": False,
        "teacher_checkpoint_read": False,
        "legacy_regions_read": False,
        "history_read": False,
        "simulation": {"static_weight": 0.5, "teacher_weight": 0.5, "overlap": 1.0},
        "static_fg_area": float(
            sum(row["static_fg_area"] for row in by_sample) / len(by_sample)
        ),
        "foreground_evidence_mean": float(
            sum(row["foreground_evidence_mean"] for row in by_sample)
            / len(by_sample)
        ),
        "background_evidence_mean": float(
            sum(row["background_evidence_mean"] for row in by_sample)
            / len(by_sample)
        ),
        "support_mean": float(support_all.mean().item()),
        "support_on_static_fg_mean": (
            support_fg_sum / support_fg_count if support_fg_count else None
        ),
        "support_on_static_bg_mean": (
            support_bg_sum / support_bg_count if support_bg_count else None
        ),
        "support_q10": global_quantiles["q10"],
        "support_q25": global_quantiles["q25"],
        "support_q50": global_quantiles["q50"],
        "support_q75": global_quantiles["q75"],
        "support_q90": global_quantiles["q90"],
        "static_target_consistency_error": max_static_consistency_error,
        "foreground_cache_consistency_error": (
            max_foreground_cache_consistency_error
        ),
        "finite_ratio": finite_count / audited_value_count,
        "range_violation_count": range_violation_count,
        "foreground_teacher_crossing_count": crossing_fg_count,
        "background_teacher_crossing_count": crossing_bg_count,
        "nonconflict_change_count": nonconflict_change_count,
        "formula_max_abs_error": max_formula_error,
        "foreground_background_symmetry_max_abs_error": max_symmetry_error,
        "all_foreground_teacher_symmetry_max_abs_error": (
            max_teacher_fg_symmetry_error
        ),
        "all_background_teacher_symmetry_max_abs_error": (
            max_teacher_bg_symmetry_error
        ),
        "all_foreground_teacher_projection_symmetric": (
            max_teacher_fg_symmetry_error <= 1e-6
        ),
        "all_background_teacher_projection_symmetric": (
            max_teacher_bg_symmetry_error <= 1e-6
        ),
        "all_foreground_teacher_stays_on_original_side": (
            crossing_fg_count == 0
        ),
        "all_background_teacher_stays_on_original_side": (
            crossing_bg_count == 0
        ),
        "errors": errors,
    }
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (
            summary["static_fg_area"],
            summary["foreground_evidence_mean"],
            summary["background_evidence_mean"],
            summary["support_mean"],
            summary["finite_ratio"],
        )
    ):
        raise RuntimeError("Audit summary contains a non-finite scalar.")

    _write_csv(by_sample_path, by_sample)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"summary_json = {summary_path}")
    print(f"by_sample_csv = {by_sample_path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
