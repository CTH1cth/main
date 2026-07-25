#!/usr/bin/env python3
"""Offline AMCD-v0 directional-mimicry hypothesis audit.

The tool reads frozen caches only.  It never reads training GT, never updates a
model, and never participates in the training/evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.amcd_directional_audit import (  # noqa: E402
    bootstrap_confidence_interval,
    compute_asymmetry,
    descriptive_statistics,
    geometry_linear_explanation,
    mask_geometry_statistics,
    paired_comparison_statistics,
    permuted_mask,
    rolled_mask,
    run_analytic_assertions,
    spearman_statistics,
    stable_seed,
)
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long50_lrfloor_2e5_sw_ones_noecst_linear30.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT.parent / "workdir" / "amcd_directional_audit"
)
EXPECTED_TRAIN_SAMPLES = 4040
REQUIRED_COLUMNS = [
    "dataset",
    "stem",
    "sample_id",
    "mask_source",
    "num_slots",
    "slot_temperature",
    "coverage_temperature",
    "soft_object_area",
    "soft_background_area",
    "mask_entropy",
    "mask_boundary_density",
    "connectedness_proxy",
    "object_slot_mass_min",
    "object_slot_mass_mean",
    "background_slot_mass_min",
    "background_slot_mass_mean",
    "e_background_to_object",
    "e_object_to_background",
    "asymmetry_raw",
    "asymmetry_normalized",
    "perm_asymmetry_mean",
    "perm_asymmetry_std",
    "roll_asymmetry_mean",
    "roll_asymmetry_std",
    "cross_asymmetry_mean",
    "cross_asymmetry_std",
    "inverse_asymmetry",
    "real_minus_perm",
    "real_minus_roll",
    "real_minus_cross",
    "valid",
    "invalid_reason",
]


@dataclass(frozen=True)
class Setting:
    num_slots: int
    slot_temperature: float
    coverage_temperature: float
    label: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_repo_relative(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _prepare_output(path: str | Path) -> Path:
    output = _resolve_repo_relative(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for relative in (
        "plots",
        "visualizations/highest_asymmetry",
        "visualizations/lowest_asymmetry",
        "visualizations/control_examples",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    return output


def _manifest_map(rows: list[dict], path: Path) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(f"Manifest row missing dataset/stem: {path}")
        key = (str(row["dataset"]), str(row["stem"]))
        if key in result:
            raise RuntimeError(f"Duplicate manifest key {key}: {path}")
        result[key] = row
    return result


def _cache_paths(cfg: Any, split: str) -> tuple[Path, Path]:
    feature_manifest = _resolve_repo_relative(
        Path(cfg.CACHE_ROOT)
        / "features_cache"
        / str(cfg.BACKBONE_KEY)
        / f"manifest_{split}.jsonl"
    )
    dabe_root = getattr(
        cfg,
        "DABE_PU_ROOT",
        "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8",
    )
    dabe_manifest = _resolve_repo_relative(Path(dabe_root) / f"manifest_{split}.jsonl")
    return feature_manifest, dabe_manifest


def _load_feature(row: dict, key: tuple[str, str], device: torch.device) -> torch.Tensor:
    path = Path(row.get("cache_path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"missing feature: {path}")
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature payload must be a dict: {path}")
    if (payload.get("dataset"), payload.get("stem")) != key:
        raise RuntimeError(f"Feature cache key mismatch: {path}")
    feature = payload.get("tensor")
    if not torch.is_tensor(feature):
        raise TypeError(f"Feature payload missing tensor: {path}")
    feature = feature.detach().float()
    if list(feature.shape) != [384, 37, 37]:
        raise RuntimeError(f"shape mismatch feature={list(feature.shape)}: {path}")
    if feature.requires_grad or not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError(f"Feature must be detached and finite: {path}")
    return feature.to(device)


def _load_dabe_mask(row: dict, key: tuple[str, str], device: torch.device) -> torch.Tensor:
    path = Path(row.get("cache_path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"missing pseudo-label: {path}")
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE-PU payload must be a dict: {path}")
    if (payload.get("dataset"), payload.get("stem")) != key:
        raise RuntimeError(f"DABE-PU cache key mismatch: {path}")
    if str(payload.get("dabe_version", "")).lower() != "pu_v11":
        raise RuntimeError(f"Expected DABE-PU pu_v11, got {payload.get('dabe_version')!r}: {path}")
    pseudo = payload.get("target_soft_68")
    if not torch.is_tensor(pseudo):
        raise KeyError(f"DABE-PU payload missing target_soft_68: {path}")
    pseudo = pseudo.detach().float()
    if list(pseudo.shape) != [1, 68, 68]:
        raise RuntimeError(f"shape mismatch target_soft_68={list(pseudo.shape)}: {path}")
    if pseudo.requires_grad or not bool(torch.isfinite(pseudo).all().item()):
        raise RuntimeError(f"DABE-PU target must be detached and finite: {path}")
    if float(pseudo.min().item()) < -1e-7 or float(pseudo.max().item()) > 1.0 + 1e-7:
        raise RuntimeError(f"DABE-PU target out of [0,1]: {path}")
    # The task explicitly requires deriving the 37x37 partition from target_soft_68.
    pseudo_37 = F.interpolate(
        pseudo.unsqueeze(0), size=(37, 37), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)
    return pseudo_37.clamp(0.0, 1.0).to(device)


def _load_rgb(path: str | Path, size: int = 68) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class StudentMaskProvider:
    def __init__(self, cfg: Any, checkpoint_path: str | Path, device: torch.device):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "student" not in checkpoint:
            raise RuntimeError(f"Checkpoint has no student state: {checkpoint_path}")
        if checkpoint.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError("Checkpoint/config backbone mismatch.")
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = build_seg_head(384, cfg).to(device)
        self.model.load_state_dict(checkpoint["student"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        head_type = str(getattr(cfg, "HEAD_TYPE", "simple")).lower()
        if head_type not in {"simple", "dagp", "dagp_safe"}:
            raise RuntimeError(
                "AMCD-v0 optional student masks currently support the stable "
                f"simple/dagp/dagp_safe heads, got {head_type!r}."
            )
        self.head_type = head_type

    @torch.no_grad()
    def __call__(self, feature: torch.Tensor, image_path: str | Path) -> torch.Tensor:
        model_input = feature.unsqueeze(0)
        if self.head_type == "simple":
            model_input = F.interpolate(
                model_input,
                size=(int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE)),
                mode="bilinear",
                align_corners=False,
            )
            output = self.model(model_input)
        elif self.head_type == "dagp":
            output = self.model(model_input)
        else:
            image_68 = _load_rgb(image_path, int(self.cfg.LOSS_SIZE)).unsqueeze(0).to(self.device)
            output = self.model(model_input, image_68=image_68, return_aux=False)
        logits = output["logits"] if isinstance(output, dict) else output
        logits = F.interpolate(
            logits,
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        )
        probability_68 = torch.sigmoid(logits.detach())
        probability_37 = F.interpolate(
            probability_68,
            size=(37, 37),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        if probability_37.requires_grad or not bool(torch.isfinite(probability_37).all().item()):
            raise RuntimeError("Student soft prediction is not detached and finite.")
        return probability_37.clamp(0.0, 1.0)


def _build_settings(args: argparse.Namespace) -> list[Setting]:
    settings = [
        Setting(
            int(args.num_slots),
            float(args.slot_temperature),
            float(args.coverage_temperature),
            "main",
        )
    ]
    if bool(args.num_slots_list) != bool(args.coverage_temperature_list):
        raise ValueError(
            "--num_slots_list and --coverage_temperature_list must be supplied together."
        )
    for num_slots in args.num_slots_list or []:
        for temperature in args.coverage_temperature_list or []:
            candidate = Setting(
                int(num_slots), float(temperature), float(temperature), "robustness"
            )
            if not any(
                current.num_slots == candidate.num_slots
                and abs(current.slot_temperature - candidate.slot_temperature) < 1e-12
                and abs(current.coverage_temperature - candidate.coverage_temperature) < 1e-12
                for current in settings
            ):
                settings.append(candidate)
    for setting in settings:
        if setting.num_slots <= 0:
            raise ValueError("All slot counts must be positive.")
        if setting.slot_temperature <= 0.0 or setting.coverage_temperature <= 0.0:
            raise ValueError("All temperatures must be positive.")
    return settings


def _select_cross_key(
    current_key: tuple[str, str],
    current_area: float,
    area_catalog: dict[tuple[str, str], float],
) -> tuple[tuple[str, str], float, bool]:
    candidates = [
        (key, area)
        for key, area in area_catalog.items()
        if key != current_key and math.isfinite(float(area))
    ]
    cross_dataset = [item for item in candidates if item[0][0] != current_key[0]]
    pool = cross_dataset or candidates
    if not pool:
        raise RuntimeError(f"No cross-image mask candidate for {current_key}.")
    selected_key, selected_area = min(
        pool,
        key=lambda item: (
            abs(float(item[1]) - float(current_area)),
            item[0][0],
            item[0][1],
        ),
    )
    difference = abs(float(selected_area) - float(current_area))
    return selected_key, difference, bool(difference <= 0.02)


def _finite_or_nan(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _invalid_row(
    key: tuple[str, str], mask_source: str, setting: Setting, reason: str
) -> dict:
    row = {
        "dataset": key[0],
        "stem": key[1],
        "sample_id": f"{key[0]}/{key[1]}",
        "mask_source": mask_source,
        "num_slots": setting.num_slots,
        "slot_temperature": setting.slot_temperature,
        "coverage_temperature": setting.coverage_temperature,
        "valid": False,
        "invalid_reason": reason,
        "setting_label": setting.label,
    }
    for column in REQUIRED_COLUMNS:
        row.setdefault(column, float("nan"))
    row["valid"] = False
    row["invalid_reason"] = reason
    return row


@torch.no_grad()
def _audit_setting(
    feature: torch.Tensor,
    mask: torch.Tensor,
    cross_mask: torch.Tensor,
    key: tuple[str, str],
    cross_key: tuple[str, str],
    cross_catalog_difference: float,
    cross_within_tolerance: bool,
    mask_source: str,
    setting: Setting,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    geometry = mask_geometry_statistics(mask)
    kwargs = {
        "num_slots": setting.num_slots,
        "slot_temperature": setting.slot_temperature,
        "coverage_temperature": setting.coverage_temperature,
        "num_iterations": int(args.slot_iterations),
        "eps": float(args.eps),
        "min_mass": float(args.min_mass),
    }
    real = compute_asymmetry(feature, mask, **kwargs)
    if not real["valid"]:
        return _invalid_row(key, mask_source, setting, real["invalid_reason"]), {}

    perm_results, roll_results = [], []
    first_permuted, first_rolled = None, None
    first_roll_shift = None
    for repeat in range(int(args.num_control_repeats)):
        perm = permuted_mask(
            mask,
            stable_seed(args.seed, key[0], key[1], "perm", repeat),
        )
        roll, dy, dx = rolled_mask(
            mask,
            stable_seed(args.seed, key[0], key[1], "roll", repeat),
            min_patch_shift=4,
        )
        perm_result = compute_asymmetry(feature, perm, **kwargs)
        roll_result = compute_asymmetry(feature, roll, **kwargs)
        if not perm_result["valid"] or not roll_result["valid"]:
            raise RuntimeError("Mass-preserving perm/roll control became invalid.")
        perm_results.append(float(perm_result["asymmetry_normalized"]))
        roll_results.append(float(roll_result["asymmetry_normalized"]))
        if repeat == 0:
            first_permuted = perm
            first_rolled = roll
            first_roll_shift = (dy, dx)

    cross = compute_asymmetry(feature, cross_mask, **kwargs)
    inverse = compute_asymmetry(feature, 1.0 - mask, **kwargs)
    if not inverse["valid"]:
        raise RuntimeError("Inverse mask unexpectedly failed mass validation.")
    inverse_raw_error = abs(
        float(real["asymmetry_raw"]) + float(inverse["asymmetry_raw"])
    )
    if inverse_raw_error >= 1e-5:
        raise AssertionError(
            f"Inverse raw sign flip failed for {key}: {inverse_raw_error:.9g}."
        )
    perm_mean, perm_std = _finite_or_nan(perm_results)
    roll_mean, roll_std = _finite_or_nan(roll_results)
    if cross["valid"]:
        cross_mean, cross_std = float(cross["asymmetry_normalized"]), 0.0
        cross_invalid = ""
    else:
        cross_mean, cross_std = float("nan"), float("nan")
        cross_invalid = f"cross control: {cross['invalid_reason']}"
    real_norm = float(real["asymmetry_normalized"])
    object_mass = real["object_slots"].slot_mass
    background_mass = real["background_slots"].slot_mass
    actual_cross_difference = abs(float(mask.mean().item()) - float(cross_mask.mean().item()))
    row = {
        "dataset": key[0],
        "stem": key[1],
        "sample_id": f"{key[0]}/{key[1]}",
        "mask_source": mask_source,
        "num_slots": setting.num_slots,
        "slot_temperature": setting.slot_temperature,
        "coverage_temperature": setting.coverage_temperature,
        **geometry,
        "object_slot_mass_min": float(object_mass.min().item()),
        "object_slot_mass_mean": float(object_mass.mean().item()),
        "background_slot_mass_min": float(background_mass.min().item()),
        "background_slot_mass_mean": float(background_mass.mean().item()),
        "e_background_to_object": float(real["e_background_to_object"]),
        "e_object_to_background": float(real["e_object_to_background"]),
        "asymmetry_raw": float(real["asymmetry_raw"]),
        "asymmetry_normalized": real_norm,
        "perm_asymmetry_mean": perm_mean,
        "perm_asymmetry_std": perm_std,
        "roll_asymmetry_mean": roll_mean,
        "roll_asymmetry_std": roll_std,
        "cross_asymmetry_mean": cross_mean,
        "cross_asymmetry_std": cross_std,
        "inverse_asymmetry": float(inverse["asymmetry_normalized"]),
        "inverse_asymmetry_raw": float(inverse["asymmetry_raw"]),
        "real_minus_perm": real_norm - perm_mean,
        "real_minus_roll": real_norm - roll_mean,
        "real_minus_cross": real_norm - cross_mean,
        "valid": not bool(cross_invalid),
        "invalid_reason": cross_invalid,
        "setting_label": setting.label,
        "object_mass": float(real["object_mass"]),
        "background_mass": float(real["background_mass"]),
        "inverse_raw_sign_flip_abs_error": inverse_raw_error,
        "cross_dataset": cross_key[0],
        "cross_stem": cross_key[1],
        "cross_sample_id": f"{cross_key[0]}/{cross_key[1]}",
        "cross_catalog_area_difference": float(cross_catalog_difference),
        "cross_actual_area_difference": actual_cross_difference,
        "cross_within_0_02": bool(cross_within_tolerance),
        "object_slot_valid_count": int(real["object_slots"].slot_validity.sum().item()),
        "background_slot_valid_count": int(real["background_slots"].slot_validity.sum().item()),
        "object_reconstruction_diagnostic": float(
            real["object_slots"].reconstruction_diagnostic
        ),
        "background_reconstruction_diagnostic": float(
            real["background_slots"].reconstruction_diagnostic
        ),
    }
    artifact = {
        "real": real,
        "mask": mask,
        "permuted_mask": first_permuted,
        "rolled_mask": first_rolled,
        "roll_shift": first_roll_shift,
        "cross_mask": cross_mask,
        "cross": cross,
        "perm_asymmetry": perm_results[0],
        "roll_asymmetry": roll_results[0],
        "cross_asymmetry": cross_mean,
        "cross_key": cross_key,
    }
    return row, artifact


def _main_filter(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    return (
        (frame["num_slots"] == int(args.num_slots))
        & np.isclose(frame["slot_temperature"], float(args.slot_temperature))
        & np.isclose(frame["coverage_temperature"], float(args.coverage_temperature))
    )


def _aggregate_outputs(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    manifest_count: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = frame[_main_filter(frame, args) & frame["valid"].astype(bool)].copy()
    aggregate: dict[str, Any] = {
        "mask_source": args.mask_source,
        "manifest_count": int(manifest_count),
        "processed_unique_samples": int(frame[["dataset", "stem"]].drop_duplicates().shape[0]),
        "valid_main_samples": int(len(main)),
        "invalid_main_samples": int((_main_filter(frame, args) & ~frame["valid"].astype(bool)).sum()),
        "full_hypothesis_decision_eligible": bool(
            args.max_samples is None
            and args.mask_source == "dabe_pu"
            and manifest_count == EXPECTED_TRAIN_SAMPLES
            and len(main) == EXPECTED_TRAIN_SAMPLES
        ),
        "distribution": {},
    }
    for label, subset in [("all", main), *[(name, main[main["dataset"] == name]) for name in sorted(main["dataset"].unique())]]:
        values = subset["asymmetry_normalized"].to_numpy(dtype=np.float64)
        stats_payload = descriptive_statistics(values)
        stats_payload["mean_ci95"] = list(
            bootstrap_confidence_interval(
                values,
                "mean",
                args.num_bootstrap,
                stable_seed(args.seed, "bootstrap", label, "mean"),
            )
        )
        stats_payload["median_ci95"] = list(
            bootstrap_confidence_interval(
                values,
                "median",
                args.num_bootstrap,
                stable_seed(args.seed, "bootstrap", label, "median"),
            )
        )
        aggregate["distribution"][label] = stats_payload

    control_rows = []
    control_stats = {}
    for name, column in (
        ("perm", "real_minus_perm"),
        ("roll", "real_minus_roll"),
        ("cross", "real_minus_cross"),
    ):
        stats_payload = paired_comparison_statistics(
            main[column].to_numpy(dtype=np.float64),
            args.num_bootstrap,
            stable_seed(args.seed, "paired", name),
        )
        control_stats[name] = stats_payload
        control_rows.append(
            {
                "control": name,
                **stats_payload,
                "mean_ci95_low": stats_payload["mean_ci95"][0],
                "mean_ci95_high": stats_payload["mean_ci95"][1],
                "median_ci95_low": stats_payload["median_ci95"][0],
                "median_ci95_high": stats_payload["median_ci95"][1],
            }
        )
    aggregate["paired_controls"] = control_stats
    control_table = pd.DataFrame(control_rows).drop(
        columns=["mean_ci95", "median_ci95"]
    )

    if len(main):
        ranks = main["soft_object_area"].rank(method="first")
        main["area_quartile"] = pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    area_rows = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        subset = main[main.get("area_quartile", pd.Series(index=main.index, dtype=object)) == quartile]
        base = descriptive_statistics(subset["asymmetry_normalized"].to_numpy(dtype=np.float64))
        cross_stats = paired_comparison_statistics(
            subset["real_minus_cross"].to_numpy(dtype=np.float64),
            args.num_bootstrap,
            stable_seed(args.seed, "quartile", quartile, "cross"),
        )
        perm_stats = paired_comparison_statistics(
            subset["real_minus_perm"].to_numpy(dtype=np.float64),
            args.num_bootstrap,
            stable_seed(args.seed, "quartile", quartile, "perm"),
        )
        area_rows.append(
            {
                "quartile": quartile,
                "count": base["count"],
                "area_min": float(subset["soft_object_area"].min()) if len(subset) else float("nan"),
                "area_max": float(subset["soft_object_area"].max()) if len(subset) else float("nan"),
                "asymmetry_mean": base["mean"],
                "asymmetry_median": base["median"],
                "positive_ratio": base["positive_ratio"],
                "real_minus_cross_mean": cross_stats["mean"],
                "real_minus_cross_median": cross_stats["median"],
                "real_minus_cross_d_z": cross_stats["d_z"],
                "real_minus_perm_mean": perm_stats["mean"],
                "real_minus_perm_median": perm_stats["median"],
                "real_minus_perm_d_z": perm_stats["d_z"],
            }
        )
    area_table = pd.DataFrame(area_rows)

    robustness_rows = []
    for keys, subset in frame[frame["valid"].astype(bool)].groupby(
        ["num_slots", "slot_temperature", "coverage_temperature"], sort=True
    ):
        robustness_rows.append(
            {
                "num_slots": int(keys[0]),
                "slot_temperature": float(keys[1]),
                "coverage_temperature": float(keys[2]),
                "count": int(len(subset)),
                "asymmetry_mean": float(subset["asymmetry_normalized"].mean()),
                "asymmetry_median": float(subset["asymmetry_normalized"].median()),
                "positive_ratio": float((subset["asymmetry_normalized"] > 0).mean()),
                "real_minus_perm_median": float(subset["real_minus_perm"].median()),
                "real_minus_roll_median": float(subset["real_minus_roll"].median()),
                "real_minus_cross_median": float(subset["real_minus_cross"].median()),
                "robustness_pass": bool(
                    subset["asymmetry_normalized"].median() > 0
                    and subset["real_minus_cross"].median() > 0
                    and subset["real_minus_perm"].median() > 0
                ),
            }
        )
    robustness_table = pd.DataFrame(
        robustness_rows,
        columns=[
            "num_slots",
            "slot_temperature",
            "coverage_temperature",
            "count",
            "asymmetry_mean",
            "asymmetry_median",
            "positive_ratio",
            "real_minus_perm_median",
            "real_minus_roll_median",
            "real_minus_cross_median",
            "robustness_pass",
        ],
    )

    aggregate["geometry_correlations"] = {
        "area": spearman_statistics(main["asymmetry_normalized"], main["soft_object_area"]),
        "entropy": spearman_statistics(main["asymmetry_normalized"], main["mask_entropy"]),
        "boundary_density": spearman_statistics(main["asymmetry_normalized"], main["mask_boundary_density"]),
    }
    aggregate["geometry_linear_model"] = geometry_linear_explanation(
        main["asymmetry_normalized"],
        main["soft_object_area"],
        main["mask_entropy"],
        main["mask_boundary_density"],
    )
    if len(main) >= 2:
        inverse_corr_raw = float(
            np.corrcoef(main["inverse_asymmetry_raw"], -main["asymmetry_raw"])[0, 1]
        )
        inverse_corr_normalized = float(
            np.corrcoef(main["inverse_asymmetry"], -main["asymmetry_normalized"])[0, 1]
        )
    else:
        inverse_corr_raw = float("nan")
        inverse_corr_normalized = float("nan")
    inverse_error_normalized = np.abs(
        main["inverse_asymmetry"] + main["asymmetry_normalized"]
    )
    inverse_error_raw = main["inverse_raw_sign_flip_abs_error"].to_numpy(
        dtype=np.float64
    )
    aggregate["inverse_symmetry"] = {
        "correlation_inverse_raw_vs_negative_real_raw": inverse_corr_raw,
        "correlation_inverse_normalized_vs_negative_real_normalized": inverse_corr_normalized,
        "median_abs_raw_sum": float(np.median(inverse_error_raw)) if len(inverse_error_raw) else float("nan"),
        "median_abs_normalized_sum": float(np.median(inverse_error_normalized)) if len(inverse_error_normalized) else float("nan"),
        "max_abs_raw_sum": float(main["inverse_raw_sign_flip_abs_error"].max()) if len(main) else float("nan"),
    }

    distribution = aggregate["distribution"].get("all", {})
    quartile_positive = area_table["positive_ratio"].to_numpy(dtype=np.float64)
    robust_only = robustness_table[
        ~(
            (robustness_table["num_slots"] == int(args.num_slots))
            & np.isclose(robustness_table["coverage_temperature"], float(args.coverage_temperature))
        )
    ]
    conditions = {
        "condition_1_positive_median": bool(
            distribution.get("median", float("nan")) > 0
            and distribution.get("median_ci95", [float("nan")])[0] > 0
        ),
        "condition_2_all_controls": bool(
            all(
                payload["median"] > 0
                and payload["median_ci95"][0] > 0
                and payload["rank_biserial"] > 0.15
                and payload["d_z"] > 0.20
                for payload in control_stats.values()
            )
        ),
        "condition_3_coverage": bool(
            distribution.get("positive_ratio", float("nan")) >= 0.60
            and quartile_positive.size == 4
            and np.all(quartile_positive >= 0.55)
        ),
        "condition_4_robustness": bool(
            len(robust_only) == 6 and int(robust_only["robustness_pass"].sum()) >= 5
        ),
        "condition_5_not_geometry": bool(
            control_stats["cross"]["median"] > 0
            and aggregate["geometry_linear_model"]["adjusted_r_squared"] < 0.35
            and int((area_table["real_minus_cross_d_z"] > 0.15).sum()) >= 3
        ),
        "condition_6_inverse_symmetry": bool(
            inverse_corr_raw > 0.995
            and aggregate["inverse_symmetry"]["median_abs_raw_sum"] < 1e-5
        ),
    }
    aggregate["automated_condition_checks"] = conditions
    aggregate["automated_all_conditions_met"] = bool(all(conditions.values()))
    aggregate["hypothesis_status"] = "REVIEW_REQUIRED" if aggregate["full_hypothesis_decision_eligible"] else "NOT_EVALUATED_SANITY_OR_PARTIAL"
    return aggregate, robustness_table, area_table, control_table


def _save_plots(
    frame: pd.DataFrame,
    robustness: pd.DataFrame,
    area_table: pd.DataFrame,
    args: argparse.Namespace,
    output: Path,
) -> None:
    main = frame[_main_filter(frame, args) & frame["valid"].astype(bool)].copy()
    plot_dir = output / "plots"
    plt.figure(figsize=(7, 5))
    plt.hist(main["asymmetry_normalized"], bins=min(40, max(5, len(main))), color="#4472C4", alpha=0.85)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Normalized directional asymmetry")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(plot_dir / "asymmetry_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    data = [main[column].dropna().to_numpy() for column in ("real_minus_perm", "real_minus_roll", "real_minus_cross")]
    if all(len(values) for values in data):
        plt.boxplot(data, tick_labels=["real-perm", "real-roll", "real-cross"], showfliers=False)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Paired normalized difference")
    plt.tight_layout()
    plt.savefig(plot_dir / "paired_control_differences.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(area_table["quartile"], area_table["asymmetry_median"], color="#70AD47")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Median normalized asymmetry")
    plt.tight_layout()
    plt.savefig(plot_dir / "area_stratified_asymmetry.png", dpi=160)
    plt.close()

    for column, filename, xlabel in (
        ("soft_object_area", "area_correlation.png", "Soft object area"),
        ("mask_entropy", "entropy_correlation.png", "Mask entropy"),
    ):
        plt.figure(figsize=(7, 5))
        plt.scatter(main[column], main["asymmetry_normalized"], s=9, alpha=0.45)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel("Normalized directional asymmetry")
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    if len(robustness):
        pivot = robustness.pivot_table(
            index="num_slots", columns="coverage_temperature", values="asymmetry_median"
        )
        image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
        plt.colorbar(image, label="Median normalized asymmetry")
        plt.xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
        plt.xlabel("Coverage temperature")
        plt.ylabel("Number of slots")
    plt.tight_layout()
    plt.savefig(plot_dir / "robustness_heatmap.png", dpi=160)
    plt.close()


def _pca_image(feature: torch.Tensor) -> np.ndarray:
    channels, height, width = feature.shape
    flat = feature.detach().cpu().float().reshape(channels, -1).transpose(0, 1).numpy()
    projection = PCA(n_components=3, svd_solver="randomized", random_state=20260722).fit_transform(flat)
    projection = projection.reshape(height, width, 3)
    low = np.percentile(projection, 1, axis=(0, 1), keepdims=True)
    high = np.percentile(projection, 99, axis=(0, 1), keepdims=True)
    return np.clip((projection - low) / (high - low + 1e-8), 0.0, 1.0)


def _text_panel(axis: Any, title: str, lines: list[str]) -> None:
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.02, 0.92, "\n".join(lines), va="top", ha="left", fontsize=10)


def _render_visualization(
    path: Path,
    image_path: str | Path,
    feature: torch.Tensor,
    artifact: dict,
) -> None:
    real = artifact["real"]
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.reshape(-1)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (visualization only)")
    axes[1].imshow(artifact["mask"].detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Soft partition")
    axes[2].imshow(_pca_image(feature))
    axes[2].set_title("DINO feature PCA")
    axes[3].imshow(real["object_slots"].assignment_map.detach().cpu(), cmap="tab20")
    axes[3].set_title("Object slot assignment")
    axes[4].imshow(real["background_slots"].assignment_map.detach().cpu(), cmap="tab20")
    axes[4].set_title("Background slot assignment")
    _text_panel(axes[5], "Background conditions object", [f"e_B→O = {real['e_background_to_object']:.6f}"])
    _text_panel(axes[6], "Object conditions background", [f"e_O→B = {real['e_object_to_background']:.6f}"])
    _text_panel(axes[7], "Directional asymmetry", [f"A raw = {real['asymmetry_raw']:.6f}", f"A norm = {real['asymmetry_normalized']:.6f}"])
    axes[8].imshow(artifact["permuted_mask"].detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[8].set_title(f"Permuted | A={artifact['perm_asymmetry']:.5f}")
    axes[9].imshow(artifact["rolled_mask"].detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[9].set_title(f"Rolled {artifact['roll_shift']} | A={artifact['roll_asymmetry']:.5f}")
    axes[10].imshow(artifact["cross_mask"].detach().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[10].set_title(f"Cross {artifact['cross_key'][0]} | A={artifact['cross_asymmetry']:.5f}")
    _text_panel(
        axes[11],
        "Control asymmetry",
        [
            f"real = {real['asymmetry_normalized']:.6f}",
            f"perm = {artifact['perm_asymmetry']:.6f}",
            f"roll = {artifact['roll_asymmetry']:.6f}",
            f"cross = {artifact['cross_asymmetry']:.6f}",
        ],
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_report(output: Path, metadata: dict, aggregate: dict, visual_count: int) -> None:
    distribution = aggregate.get("distribution", {}).get("all", {})
    lines = [
        "# AMCD-v0 Directional Mimicry Offline Audit",
        "",
        "This report uses frozen DINO features and soft partitions only. Training GT was not read.",
        "",
        "## Scope",
        "",
        f"- Mask source: `{metadata['mask_source']}`",
        f"- Manifest samples: {metadata['manifest_count']}",
        f"- Selected samples: {metadata['selected_samples']}",
        f"- Valid main-setting samples: {aggregate['valid_main_samples']}",
        f"- Visualizations: {visual_count}",
        "- Training/model updates: none",
        "- Ground truth use: none",
        "",
        "## Main setting summary",
        "",
        f"- Median normalized asymmetry: {distribution.get('median')}",
        f"- Positive ratio: {distribution.get('positive_ratio')}",
        f"- Median 95% bootstrap CI: {distribution.get('median_ci95')}",
        "",
        "## Decision boundary",
        "",
        f"- Full-decision eligible: {aggregate['full_hypothesis_decision_eligible']}",
        f"- Status: `{aggregate['hypothesis_status']}`",
        "",
        "A sanity/partial run is an implementation check and must not be used to declare the AMCD hypothesis PASS or FAIL.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--mask_source", choices=["dabe_pu", "student"], default="dabe_pu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--slot_temperature", type=float, default=0.10)
    parser.add_argument("--coverage_temperature", type=float, default=0.10)
    parser.add_argument("--num_slots_list", nargs="*", type=int, default=None)
    parser.add_argument("--coverage_temperature_list", nargs="*", type=float, default=None)
    parser.add_argument("--slot_iterations", "--num_iterations", type=int, default=10)
    parser.add_argument("--num_control_repeats", type=int, default=10)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--min_mass", type=float, default=4.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--visualize_top_k", type=int, default=20)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.max_samples is not None and not 1 <= int(args.max_samples) <= 5:
        parser.error("--max_samples, when supplied, is restricted to 1-5.")
    if args.mask_source == "student" and not args.checkpoint:
        parser.error("--mask_source student requires --checkpoint selected by the user.")
    if args.mask_source == "dabe_pu" and args.checkpoint:
        parser.error("--checkpoint is only valid with --mask_source student.")
    if args.num_control_repeats <= 0 or args.num_bootstrap <= 0:
        parser.error("Control repeats and bootstrap count must be positive.")
    if args.slot_iterations <= 0 or args.min_mass <= 0 or args.eps <= 0:
        parser.error("slot_iterations, min_mass and eps must be positive.")
    return args


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    settings = _build_settings(args)
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    feature_manifest, dabe_manifest = _cache_paths(cfg, args.split)
    feature_rows = read_jsonl(feature_manifest)
    dabe_rows = read_jsonl(dabe_manifest)
    feature_map = _manifest_map(feature_rows, feature_manifest)
    dabe_map = _manifest_map(dabe_rows, dabe_manifest)
    train_datasets = set(str(name) for name in cfg.TRAIN_DATASETS)
    ordered_rows = [row for row in feature_rows if str(row["dataset"]) in train_datasets]
    if not ordered_rows:
        raise RuntimeError("No configured training rows found in feature manifest.")
    selected_rows = ordered_rows[: args.max_samples] if args.max_samples is not None else ordered_rows
    output = _prepare_output(args.output_dir)
    analytic = run_analytic_assertions(device)

    provider = StudentMaskProvider(cfg, args.checkpoint, device) if args.mask_source == "student" else None
    mask_cache: dict[tuple[str, str], torch.Tensor] = {}
    feature_cache: dict[tuple[str, str], torch.Tensor] = {}
    image_paths: dict[tuple[str, str], str] = {}

    def load_feature_for(key: tuple[str, str]) -> torch.Tensor:
        if key not in feature_cache:
            if key not in feature_map:
                raise FileNotFoundError(f"missing feature manifest row: {key}")
            loaded = _load_feature(feature_map[key], key, device)
            # Retaining every 384x37x37 tensor would cost roughly 8.5 GB for
            # the full training set.  Cache only the explicitly bounded sanity
            # scope; full analysis streams features from disk.
            if args.max_samples is not None:
                feature_cache[key] = loaded
            return loaded
        return feature_cache[key]

    def load_mask_for(key: tuple[str, str]) -> torch.Tensor:
        if key in mask_cache:
            return mask_cache[key]
        if key not in dabe_map:
            raise FileNotFoundError(f"missing pseudo-label manifest row: {key}")
        if args.mask_source == "dabe_pu":
            mask = _load_dabe_mask(dabe_map[key], key, device)
        else:
            feature = load_feature_for(key)
            image_path = feature_map[key].get("image_path") or dabe_map[key].get("image_path")
            if not image_path or not Path(image_path).is_file():
                raise FileNotFoundError(f"RGB path missing for student mask: {key}")
            mask = provider(feature, image_path)
        mask_cache[key] = mask
        return mask

    if args.mask_source == "dabe_pu":
        area_catalog = {
            key: float(row.get("target_soft_area", row.get("area", float("nan"))))
            for key, row in dabe_map.items()
            if key[0] in train_datasets
        }
    else:
        # Student analysis is optional.  Its cross-image pool uses exactly the
        # user-selected run scope; a full run therefore covers all 4040 masks.
        area_catalog = {}
        for row in selected_rows:
            key = (str(row["dataset"]), str(row["stem"]))
            area_catalog[key] = float(load_mask_for(key).mean().item())

    rows: list[dict] = []
    failures: list[dict] = []
    main_artifacts: dict[tuple[str, str], dict] = {}
    for index, feature_row in enumerate(selected_rows):
        key = (str(feature_row["dataset"]), str(feature_row["stem"]))
        image_path = feature_row.get("image_path") or dabe_map.get(key, {}).get("image_path")
        if image_path:
            image_paths[key] = str(image_path)
        try:
            feature = load_feature_for(key)
            mask = load_mask_for(key)
            current_area = float(mask.mean().item())
            cross_key, catalog_difference, within_tolerance = _select_cross_key(
                key, float(area_catalog.get(key, current_area)), area_catalog
            )
            cross_mask = load_mask_for(cross_key)
            for setting in settings:
                row, artifact = _audit_setting(
                    feature,
                    mask,
                    cross_mask,
                    key,
                    cross_key,
                    catalog_difference,
                    within_tolerance,
                    args.mask_source,
                    setting,
                    args,
                )
                rows.append(row)
                if setting.label == "main" and artifact and args.max_samples is not None:
                    main_artifacts[key] = artifact
                if not bool(row["valid"]):
                    failures.append(
                        {
                            "dataset": key[0],
                            "stem": key[1],
                            "sample_id": f"{key[0]}/{key[1]}",
                            "failure_type": "invalid",
                            "reason": row["invalid_reason"],
                        }
                    )
        except Exception as exc:
            reason = str(exc)
            failures.append(
                {
                    "dataset": key[0],
                    "stem": key[1],
                    "sample_id": f"{key[0]}/{key[1]}",
                    "failure_type": "exception",
                    "reason": reason,
                }
            )
            for setting in settings:
                rows.append(_invalid_row(key, args.mask_source, setting, reason))
        if args.max_samples is None and (index + 1) % 100 == 0:
            print(f"[AMCD-v0] processed {index + 1}/{len(selected_rows)}", flush=True)

    frame = pd.DataFrame(rows)
    for column in REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = float("nan")
    ordered_columns = REQUIRED_COLUMNS + [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    frame = frame[ordered_columns]
    frame.to_csv(output / "per_sample_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    aggregate, robustness, area_table, control_table = _aggregate_outputs(
        frame, args, len(ordered_rows)
    )
    robustness.to_csv(output / "robustness_table.csv", index=False)
    area_table.to_csv(output / "area_stratified_metrics.csv", index=False)
    control_table.to_csv(output / "control_comparison.csv", index=False)

    main_frame = frame[_main_filter(frame, args) & frame["valid"].astype(bool)].copy()
    for _, row in main_frame.iterrows():
        reasons = []
        if float(row["asymmetry_normalized"]) <= 0.0:
            reasons.append("non-positive real asymmetry")
        for column in ("real_minus_perm", "real_minus_roll", "real_minus_cross"):
            if float(row[column]) <= 0.0:
                reasons.append(f"{column} non-positive")
        if reasons:
            failures.append(
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "sample_id": row["sample_id"],
                    "failure_type": "directional_or_control",
                    "reason": "; ".join(reasons),
                }
            )
    pd.DataFrame(
        failures,
        columns=["dataset", "stem", "sample_id", "failure_type", "reason"],
    ).drop_duplicates().to_csv(output / "failure_samples.csv", index=False)
    _write_json(output / "aggregate_metrics.json", aggregate)
    _save_plots(frame, robustness, area_table, args, output)

    visual_count = 0
    if len(main_frame):
        top_k = min(int(args.visualize_top_k), len(main_frame))
        if args.max_samples is not None:
            top_k = min(1, top_k)
        groups = [
            ("highest_asymmetry", main_frame.nlargest(top_k, "asymmetry_normalized"), "highest"),
            ("lowest_asymmetry", main_frame.nsmallest(top_k, "asymmetry_normalized"), "lowest"),
            ("control_examples", main_frame.nlargest(top_k, "real_minus_cross"), "largest_real_minus_cross"),
            ("control_examples", main_frame.nsmallest(top_k, "real_minus_cross"), "smallest_real_minus_cross"),
        ]
        for directory, subset, prefix in groups:
            for rank, (_, row) in enumerate(subset.iterrows(), 1):
                key = (str(row["dataset"]), str(row["stem"]))
                if key not in image_paths or not Path(image_paths[key]).is_file():
                    continue
                artifact = main_artifacts.get(key)
                if artifact is None:
                    cross_key = (str(row["cross_dataset"]), str(row["cross_stem"]))
                    visual_args = argparse.Namespace(**vars(args))
                    visual_args.num_control_repeats = 1
                    _, artifact = _audit_setting(
                        load_feature_for(key),
                        load_mask_for(key),
                        load_mask_for(cross_key),
                        key,
                        cross_key,
                        float(row["cross_catalog_area_difference"]),
                        bool(row["cross_within_0_02"]),
                        args.mask_source,
                        settings[0],
                        visual_args,
                    )
                    if not artifact:
                        continue
                name = f"{prefix}_{rank:02d}_{key[0]}_{key[1]}.png"
                _render_visualization(
                    output / "visualizations" / directory / name,
                    image_paths[key],
                    load_feature_for(key),
                    artifact,
                )
                visual_count += 1

    metadata = {
        "audit_version": "amcd_v0_directional_mimicry_offline_precheck",
        "config_path": str(config_path),
        "feature_manifest": str(feature_manifest),
        "dabe_pu_manifest": str(dabe_manifest),
        "feature_cache_root": str(feature_manifest.parent),
        "dabe_pu_cache_root": str(dabe_manifest.parent),
        "manifest_count": len(ordered_rows),
        "selected_samples": len(selected_rows),
        "sample_id_rule": "dataset/stem",
        "mask_source": args.mask_source,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "settings": [asdict(setting) for setting in settings],
        "slot_iterations": int(args.slot_iterations),
        "num_control_repeats": int(args.num_control_repeats),
        "num_bootstrap": int(args.num_bootstrap),
        "seed": int(args.seed),
        "min_mass": float(args.min_mass),
        "device": str(device),
        "uses_ground_truth": False,
        "runs_training": False,
        "updates_model_or_optimizer": False,
        "analytic_assertions": analytic,
    }
    _write_json(output / "config.json", metadata)
    _save_report(output, metadata, aggregate, visual_count)

    sanity_checks = {
        "feature_pseudo_alignment": bool(len(selected_rows) > 0),
        "slot_shapes_valid": bool(
            len(main_frame) > 0
            and (main_frame["object_slot_valid_count"] == int(args.num_slots)).all()
            and (main_frame["background_slot_valid_count"] == int(args.num_slots)).all()
        ),
        "slot_mass_nonzero": bool(
            len(main_frame) > 0
            and (main_frame["object_slot_mass_min"] > 0).all()
            and (main_frame["background_slot_mass_min"] > 0).all()
        ),
        "bidirectional_residual_range": bool(
            len(main_frame) > 0
            and main_frame["e_background_to_object"].between(0.0, 2.0).all()
            and main_frame["e_object_to_background"].between(0.0, 2.0).all()
        ),
        "inverse_sign_flip": bool(
            len(main_frame) > 0
            and (main_frame["inverse_raw_sign_flip_abs_error"] < 1e-5).all()
        ),
        "random_controls_generated": bool(
            len(main_frame) > 0
            and main_frame[["perm_asymmetry_mean", "roll_asymmetry_mean"]].notna().all().all()
        ),
        "cross_image_area_matching_generated": bool(
            len(main_frame) > 0 and main_frame["cross_sample_id"].notna().all()
        ),
        "all_outputs_finite": bool(
            len(main_frame) > 0
            and np.isfinite(
                main_frame[
                    [
                        "asymmetry_raw",
                        "asymmetry_normalized",
                        "perm_asymmetry_mean",
                        "roll_asymmetry_mean",
                        "cross_asymmetry_mean",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
        "visualization_exported": visual_count > 0,
        "analytic_assertions": analytic,
    }
    sanity_payload = {
        "status": "PASS" if all(bool(value) for key, value in sanity_checks.items() if key != "analytic_assertions") else "FAIL",
        "hypothesis_status": "NOT_EVALUATED",
        "num_selected_samples": len(selected_rows),
        "num_valid_main_samples": int(len(main_frame)),
        "checks": sanity_checks,
        "visualization_count": visual_count,
        "max_inverse_raw_sign_flip_abs_error": float(main_frame["inverse_raw_sign_flip_abs_error"].max()) if len(main_frame) else float("nan"),
        "max_cross_actual_area_difference": float(main_frame["cross_actual_area_difference"].max()) if len(main_frame) else float("nan"),
        "output_dir": str(output),
    }
    if args.max_samples is not None:
        _write_json(output / "sanity_results.json", sanity_payload)
    print(json.dumps(_jsonable(sanity_payload), ensure_ascii=False, indent=2))
    return sanity_payload


def main() -> None:
    args = _parse_args()
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
