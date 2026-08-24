#!/usr/bin/env python3
"""Generate and compare true-RGB multi-view fixed-R32 GBSP pseudo labels.

Generation is GT-free.  The script saves each sample's aligned view scores
before loading its training GT, then reports a clearly labelled oracle
diagnostic over TR-CAMO and TR-COD10K.  It never starts model training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.cache_features import (  # noqa: E402
    call_dino,
    load_dino,
    resolve_key_projection,
)
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.metrics import _prepare_data  # noqa: E402
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    read_jsonl,
    torch_load,
    write_json,
    write_jsonl,
)
from models.gbsp_rgb_multiview import (  # noqa: E402
    GENERATED_VIEW_ORDER,
    METHOD_ORDER,
    VIEW_ORDER,
    apply_rgb_view,
    build_method_probabilities_68,
    fixed_r32_score_from_rgb,
    inverse_align_tensor,
    key_batch_to_feature_tensor,
    pil_to_dino_tensor,
    pil_to_rgb_grid,
    stack_aligned_scores,
    unpack_aligned_scores,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CACHE_ROOT = (PROJECT_ROOT / "datasets/cache").resolve()
DEFAULT_CONFIG = MAIN_ROOT / (
    "configs/dinov1_s8_gbsp_r32_lcic_d_full_dice005_t050_seed2027.py"
)
DEFAULT_IDENTITY_MANIFEST = CACHE_ROOT / (
    "gbsp_pca_absmm_r32_pseudo_cache/dinov1-s8/manifest_train.jsonl"
)
DEFAULT_OUTPUT_ROOT = CACHE_ROOT / "gbsp_r32_rgb_multiview_v1/dinov1-s8"
EXPECTED_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
VERSION = "gbsp_r32_true_rgb_d4_multiview_v1"
METRICS = (
    "S",
    "adp_E",
    "MAE",
    "F_beta_w",
    "IoU",
    "Precision",
    "Recall",
    "Dice",
    "PredArea",
    "GTArea",
    "AreaBias",
    "AreaAbsError",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (MAIN_ROOT / value).resolve()


def _validate_output_root(path: str | Path) -> Path:
    output = _resolve(path)
    if output == CACHE_ROOT or CACHE_ROOT not in output.parents:
        raise ValueError(f"output_root must be a child of {CACHE_ROOT}: {output}")
    return output


def _manifest_index(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path)
    index = {(str(row["dataset"]), str(row["stem"])): row for row in rows}
    if len(index) != len(rows):
        raise RuntimeError(f"duplicate manifest identities: {path}")
    return rows, index


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    quotas = {dataset: 0 for dataset in EXPECTED_COUNTS}
    remaining = int(max_samples)
    while remaining:
        progressed = False
        for dataset in EXPECTED_COUNTS:
            if quotas[dataset] < len(grouped[dataset]):
                quotas[dataset] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = []
    for dataset, count in quotas.items():
        if count:
            indices = np.linspace(
                0, len(grouped[dataset]) - 1, count, dtype=np.int64
            )
            selected.extend(grouped[dataset][int(index)] for index in indices)
    return selected


def _settings_fingerprint(cfg, graph_params: dict, threshold: float) -> str:
    fields = {
        "version": VERSION,
        "backbone": str(cfg.BACKBONE_KEY),
        "dino_model_path": str(Path(cfg.DINO["model_path"]).resolve()),
        "feature_input_size": int(cfg.DINO["feature_input_size"]),
        "patch_size": int(cfg.DINO["patch_size"]),
        "feature_dim": int(cfg.DINO["embed_dim"]),
        "feature_interpolation": str(
            getattr(cfg, "FEATURE_RESIZE_INTERPOLATION", "bicubic")
        ),
        "views": list(VIEW_ORDER),
        "generated_views": list(GENERATED_VIEW_ORDER),
        "graph": {
            key: graph_params[key]
            for key in (
                "SIGMA_F",
                "SIGMA_C",
                "SIGMA_E",
                "TAU_BC",
                "BORDER_WIDTH",
                "BG_ANCHOR_TOP_PERCENT",
                "BG_ANCHOR_MIN_RATIO",
                "BG_ANCHOR_FALLBACK_TOP_PERCENT",
            )
        },
        "pca_rank_mode": "fixed",
        "fixed_pca_rank": 32,
        "score": "squared_affine_pca_residual_per_view_minmax",
        "threshold": float(threshold),
        "strict_majority_tie": "background",
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_identity_score(row: dict) -> tuple[torch.Tensor, Path]:
    path = Path(row["cache_path"]).resolve()
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"identity GBSP payload must be dict: {path}")
    identity = (str(row["dataset"]), str(row["stem"]))
    if (str(payload.get("dataset")), str(payload.get("stem"))) != identity:
        raise RuntimeError(f"identity GBSP payload mismatch: {path}")
    if str(payload.get("gbsp_version")) != "gbsp_pca_absmm_r32_v1":
        raise RuntimeError(f"unexpected identity GBSP version: {path}")
    selected = payload.get("selected_ranks")
    if (
        not torch.is_tensor(selected)
        or selected.numel() != 1
        or int(selected.reshape(-1)[0]) != 32
        or str(payload.get("pca_rank_mode")) != "fixed"
    ):
        raise RuntimeError(f"identity cache is not fixed R32: {path}")
    if bool(payload.get("fallback_used", False)):
        raise RuntimeError(f"identity cache unexpectedly used fallback: {path}")
    score = payload.get("gbsp_abs_minmax_37")
    if not torch.is_tensor(score) or tuple(score.shape) != (1, 37, 37):
        raise ValueError(f"invalid identity GBSP score: {path}")
    score = score.detach().cpu().float().contiguous()
    if (
        not bool(torch.isfinite(score).all())
        or float(score.min()) < -1e-6
        or float(score.max()) > 1.0 + 1e-6
    ):
        raise ValueError(f"identity GBSP score must be finite in [0,1]: {path}")
    return score.clamp(0.0, 1.0), path


def _valid_existing(
    path: Path,
    dataset: str,
    stem: str,
    fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        stacked = payload.get("aligned_scores_37")
        return (
            isinstance(payload, dict)
            and payload.get("dataset") == dataset
            and payload.get("stem") == stem
            and payload.get("version") == VERSION
            and payload.get("settings_fingerprint") == fingerprint
            and tuple(payload.get("view_order", ())) == VIEW_ORDER
            and bool(payload.get("transformed_rgb_dino_forward_used"))
            and not bool(payload.get("feature_only_transform_used", True))
            and torch.is_tensor(stacked)
            and tuple(stacked.shape) == (6, 1, 37, 37)
            and bool(torch.isfinite(stacked).all())
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _gt_path(image_path: Path, stem: str) -> Path:
    path = image_path.parent.parent / "gt" / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_gt68(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    value = torch.from_numpy((array > 0.5).astype(np.float32))[None, None]
    return F.interpolate(value, size=(68, 68), mode="nearest").squeeze(0)


def _evaluate_methods(
    aligned_scores: dict[str, torch.Tensor],
    gt68: torch.Tensor,
    threshold: float,
    dataset: str,
    stem: str,
) -> list[dict]:
    probabilities = build_method_probabilities_68(
        aligned_scores, threshold=threshold
    )
    context = FastCODContext(gt68)
    evaluated = context.evaluate_many(
        [(method, "hard", probabilities[method]) for method in METHOD_ORDER],
        float(threshold),
    )
    rows = []
    gt_area = float(gt68.mean())
    for method in METHOD_ORDER:
        metrics = evaluated[(method, "hard")]
        probability = probabilities[method]
        hard = (probability > float(threshold)).float()
        pred_np, _ = _prepare_data(
            gt=context.gt_float,
            pred=hard.numpy().astype(float).squeeze(),
        )
        context.emeasure.gt_fg_numel = context.gt_fg_numel
        context.emeasure.gt_size = context.gt_size
        adp_e = float(context.emeasure.cal_adaptive_em(pred_np, context.gt))
        precision = float(metrics["Precision"])
        recall = float(metrics["Recall"])
        dice = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 1.0
        )
        pred_area = float(metrics["Area"])
        row = {
            "dataset": dataset,
            "stem": stem,
            "method": method,
            "S": float(metrics["S_m"]),
            "adp_E": adp_e,
            "MAE": float(metrics["MAE"]),
            "F_beta_w": float(metrics["F_beta_w"]),
            "IoU": float(metrics["IoU"]),
            "Precision": precision,
            "Recall": recall,
            "Dice": dice,
            "PredArea": pred_area,
            "GTArea": gt_area,
            "AreaBias": pred_area - gt_area,
            "AreaAbsError": abs(pred_area - gt_area),
        }
        if not all(math.isfinite(float(row[field])) for field in METRICS):
            raise RuntimeError(f"non-finite metric: {dataset}/{stem}/{method}")
        rows.append(row)
    return rows


def _aggregate(per_image: list[dict]) -> list[dict]:
    output = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_image:
        grouped[(row["method"], row["dataset"])].append(row)
    dataset_summary: dict[tuple[str, str], dict] = {}
    for method in METHOD_ORDER:
        for dataset in EXPECTED_COUNTS:
            rows = grouped[(method, dataset)]
            if not rows:
                continue
            values = {
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in METRICS
            }
            item = {
                "scope": "dataset",
                "dataset": dataset,
                "method": method,
                "num_samples": len(rows),
                **values,
                "J_S_adpE_1mMAE": (
                    values["S"] + values["adp_E"] + 1.0 - values["MAE"]
                )
                / 3.0,
            }
            output.append(item)
            dataset_summary[(method, dataset)] = item
        available = [
            dataset
            for dataset in EXPECTED_COUNTS
            if (method, dataset) in dataset_summary
        ]
        if available:
            values = {
                metric: float(
                    np.mean(
                        [dataset_summary[(method, dataset)][metric] for dataset in available]
                    )
                )
                for metric in METRICS
            }
            output.append(
                {
                    "scope": "dataset_equal_macro",
                    "dataset": "|".join(available),
                    "method": method,
                    "num_samples": sum(
                        dataset_summary[(method, dataset)]["num_samples"]
                        for dataset in available
                    ),
                    **values,
                    "J_S_adpE_1mMAE": (
                        values["S"] + values["adp_E"] + 1.0 - values["MAE"]
                    )
                    / 3.0,
                }
            )
    baseline = {
        (row["scope"], row["dataset"]): row
        for row in output
        if row["method"] == "identity"
    }
    for row in output:
        base = baseline.get((row["scope"], row["dataset"]))
        if base:
            for metric in (*METRICS, "J_S_adpE_1mMAE"):
                row[f"delta_{metric}_vs_identity"] = float(row[metric]) - float(
                    base[metric]
                )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: list[dict], threshold: float) -> None:
    lines = [
        "# R32-GBSP 真实 RGB 多视图全量诊断",
        "",
        "> GT 不参与伪标签生成；本报告读取训练 GT，因此只属于 oracle 诊断。",
        "",
        f"- 二值化：先 37→68 双线性插值，再 strict > {threshold:.2f}。",
        "- 六视图：identity、hflip、vflip、rot90、rot180、rot270。",
        "- 每个非 identity 视图均由变换后的原始 RGB 独立经过冻结 DINO。",
        "- 六视图多数票为 strict > 3，3:3 平票归背景。",
        "",
        "| scope | method | S | adp E | MAE | Fβw | IoU | P | R | PredArea | ΔJ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["scope"] not in {"dataset", "dataset_equal_macro"}:
            continue
        scope = row["dataset"]
        lines.append(
            f"| {scope} | {row['method']} | {row['S']:.6f} | "
            f"{row['adp_E']:.6f} | {row['MAE']:.6f} | "
            f"{row['F_beta_w']:.6f} | {row['IoU']:.6f} | "
            f"{row['Precision']:.6f} | {row['Recall']:.6f} | "
            f"{row['PredArea']:.6f} | "
            f"{row['delta_J_S_adpE_1mMAE_vs_identity']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_model_state(cfg, device: torch.device):
    model = load_dino(cfg, device)
    holder = {"tensor": None}

    def hook(_module, _inputs, output):
        holder["tensor"] = output.detach()

    module, key_path = resolve_key_projection(model)
    handle = module.register_forward_hook(hook)
    return model, holder, handle, key_path


def _extract_generated_views(
    image: Image.Image,
    cfg,
    model,
    holder: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    interpolation = str(getattr(cfg, "FEATURE_RESIZE_INTERPOLATION", "bicubic"))
    transformed = [apply_rgb_view(image, view) for view in GENERATED_VIEW_ORDER]
    inputs = torch.stack(
        [
            pil_to_dino_tensor(
                value,
                int(cfg.DINO["feature_input_size"]),
                interpolation=interpolation,
            )
            for value in transformed
        ],
        dim=0,
    ).to(device)
    holder["tensor"] = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        call_dino(model, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    key = holder["tensor"]
    if key is None:
        raise RuntimeError("DINO key hook did not capture a tensor")
    features = key_batch_to_feature_tensor(key)
    if int(features.shape[0]) != len(GENERATED_VIEW_ORDER):
        raise RuntimeError("DINO generated-view batch size mismatch")
    rgb_grids = {
        view: pil_to_rgb_grid(value, int(features.shape[-1]))
        for view, value in zip(GENERATED_VIEW_ORDER, transformed)
    }
    return features, rgb_grids, elapsed


def run_comparison(
    *,
    config: str | Path = DEFAULT_CONFIG,
    identity_manifest: str | Path = DEFAULT_IDENTITY_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    threshold: float = 0.50,
    max_samples: int = -1,
    device_name: str = "cuda",
    torch_threads: int = 4,
    force: bool = False,
    strict_failures: bool = False,
) -> dict:
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if int(torch_threads) <= 0:
        raise ValueError("torch_threads must be positive")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    torch.set_num_threads(int(torch_threads))

    config_path = _resolve(config)
    identity_path = _resolve(identity_manifest)
    output = _validate_output_root(output_root)
    cfg = load_config(config_path)
    if str(getattr(cfg, "BACKBONE_KEY", "")) != "dinov1-s8":
        raise ValueError("this protocol requires BACKBONE_KEY='dinov1-s8'")
    if int(cfg.DINO["feature_input_size"]) != 296:
        raise ValueError("this protocol requires the established 296 input")
    if int(cfg.DINO["patch_size"]) != 8 or int(cfg.DINO["embed_dim"]) != 384:
        raise ValueError("this protocol requires DINOv1-S/8")

    graph_params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg)}
    if (
        int(graph_params["BORDER_WIDTH"]) != 2
        or abs(float(graph_params["BG_ANCHOR_TOP_PERCENT"]) - 30.0) > 1e-12
    ):
        raise ValueError("registered multi-view control requires BW2-P30")
    fingerprint = _settings_fingerprint(cfg, graph_params, float(threshold))
    identity_rows, identity_index = _manifest_index(identity_path)
    feature_path = feature_manifest_path(cfg, "train")
    feature_path = _resolve(feature_path)
    feature_rows, feature_index = _manifest_index(feature_path)
    if set(identity_index) != set(feature_index):
        raise RuntimeError("identity GBSP and DINO feature manifests differ")
    counts = Counter(str(row["dataset"]) for row in identity_rows)
    if len(identity_rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"formal source manifest mismatch: {len(identity_rows)} {dict(counts)}"
        )
    selected = _balanced_subset(identity_rows, int(max_samples))
    output.mkdir(parents=True, exist_ok=True)
    analysis_dir = output / f"analysis_t{int(round(float(threshold) * 100)):03d}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{_now()}] config = {config_path}", flush=True)
    print(f"[{_now()}] identity_manifest = {identity_path}", flush=True)
    print(f"[{_now()}] feature_manifest = {feature_path}", flush=True)
    print(f"[{_now()}] output_root = {output}", flush=True)
    print(f"[{_now()}] samples = {len(selected)}", flush=True)
    print(f"[{_now()}] views = {','.join(VIEW_ORDER)}", flush=True)
    print(f"[{_now()}] training_used = False", flush=True)
    print(f"[{_now()}] gt_used_for_generation = False", flush=True)
    print(f"[{_now()}] gt_used_for_diagnostic = True", flush=True)

    model = holder = handle = None
    key_path = None
    manifest_rows = []
    per_image = []
    failures = []
    generated = reused = 0
    dino_seconds = gbsp_seconds = metric_seconds = 0.0
    started = time.perf_counter()
    try:
        for index, identity_row in enumerate(selected, 1):
            dataset = str(identity_row["dataset"])
            stem = str(identity_row["stem"])
            feature_row = feature_index[(dataset, stem)]
            image_path = Path(feature_row["image_path"]).resolve()
            cache_path = output / "train" / dataset / f"{stem}.pt"
            try:
                if _valid_existing(cache_path, dataset, stem, fingerprint) and not force:
                    payload = torch_load(cache_path, map_location="cpu")
                    aligned = unpack_aligned_scores(payload["aligned_scores_37"])
                    reused += 1
                else:
                    identity_score, identity_cache_path = _load_identity_score(
                        identity_row
                    )
                    if model is None:
                        model, holder, handle, key_path = _load_model_state(cfg, device)
                        print(f"[{_now()}] dino_key_hook = {key_path}", flush=True)
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                        original_size = (int(image.height), int(image.width))
                        features, rgb_grids, dino_elapsed = _extract_generated_views(
                            image, cfg, model, holder, device
                        )
                    dino_seconds += dino_elapsed
                    aligned = {"identity": identity_score}
                    view_metadata = {"identity": {"source": "existing_r32_cache"}}
                    gbsp_started = time.perf_counter()
                    for view_index, view in enumerate(GENERATED_VIEW_ORDER):
                        score, metadata = fixed_r32_score_from_rgb(
                            features[view_index],
                            rgb_grids[view],
                            graph_params,
                            border_width=2,
                            top_percent=30.0,
                        )
                        aligned[view] = inverse_align_tensor(score, view).contiguous()
                        view_metadata[view] = metadata
                    current_gbsp_seconds = time.perf_counter() - gbsp_started
                    gbsp_seconds += current_gbsp_seconds
                    stacked = stack_aligned_scores(aligned)
                    payload = {
                        "dataset": dataset,
                        "stem": stem,
                        "image_path": str(image_path),
                        "original_size": original_size,
                        "backbone_key": "dinov1-s8",
                        "version": VERSION,
                        "settings_fingerprint": fingerprint,
                        "view_order": list(VIEW_ORDER),
                        "generated_view_order": list(GENERATED_VIEW_ORDER),
                        "aligned_scores_37": stacked,
                        "transformed_rgb_dino_forward_used": True,
                        "feature_only_transform_used": False,
                        "identity_dino_recomputed": False,
                        "identity_source_cache_path": str(identity_cache_path),
                        "graph_path_used": True,
                        "graph_path_method": "multi_source_dijkstra_neglog_affinity",
                        "candidate_border_width": 2,
                        "candidate_top_percent": 30.0,
                        "pca_rank_mode": "fixed",
                        "fixed_pca_rank": 32,
                        "per_view_minmax_used": True,
                        "gt_used_for_generation": False,
                        "training_used": False,
                        "view_metadata": view_metadata,
                        "runtime": {
                            "five_view_dino_seconds": dino_elapsed,
                            "five_view_gbsp_seconds": current_gbsp_seconds,
                        },
                    }
                    # This write deliberately happens before any GT path is opened.
                    _atomic_save(payload, cache_path)
                    generated += 1

                metric_started = time.perf_counter()
                gt68 = _load_gt68(_gt_path(image_path, stem))
                rows = _evaluate_methods(
                    aligned, gt68, float(threshold), dataset, stem
                )
                metric_seconds += time.perf_counter() - metric_started
                per_image.extend(rows)
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "image_path": str(image_path),
                        "cache_path": str(cache_path.resolve()),
                        "version": VERSION,
                        "settings_fingerprint": fingerprint,
                        "view_order": list(VIEW_ORDER),
                        "shape": [6, 1, 37, 37],
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            if index % 10 == 0 or index == len(selected):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                eta = (len(selected) - index) / max(rate, 1e-9)
                print(
                    f"[{_now()}] {index}/{len(selected)} generated={generated} "
                    f"reused={reused} failed={len(failures)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.remove()

    wall_seconds = time.perf_counter() - started
    write_jsonl(output / "manifest_train.jsonl", manifest_rows)
    write_json(output / "generation_failures.json", failures)
    _write_csv(analysis_dir / "per_image.csv", per_image)
    summary = _aggregate(per_image)
    _write_csv(analysis_dir / "summary.csv", summary)
    _write_report(analysis_dir / "RESULTS.md", summary, float(threshold))
    valid_counts = Counter(str(row["dataset"]) for row in manifest_rows)
    formal = (
        int(max_samples) == -1
        and len(manifest_rows) == EXPECTED_TOTAL
        and dict(valid_counts) == EXPECTED_COUNTS
        and not failures
    )
    protocol = {
        "version": VERSION,
        "created_at": _now(),
        "config_path": str(config_path),
        "identity_manifest": str(identity_path),
        "feature_manifest": str(feature_path),
        "output_root": str(output),
        "analysis_dir": str(analysis_dir),
        "settings_fingerprint": fingerprint,
        "num_requested": len(selected),
        "num_valid": len(manifest_rows),
        "num_failed": len(failures),
        "dataset_counts": dict(valid_counts),
        "formal_full4040": formal,
        "views": list(VIEW_ORDER),
        "methods": list(METHOD_ORDER),
        "threshold": float(threshold),
        "threshold_rule": "resize_37_to_68_bilinear_then_strict_greater",
        "strict_majority_rule": "votes_greater_than_3_tie_is_background",
        "transformed_rgb_dino_forward_used": True,
        "feature_only_transform_used": False,
        "identity_cache_reused": True,
        "graph_path_used": True,
        "fixed_pca_rank": 32,
        "candidate_border_width": 2,
        "candidate_top_percent": 30.0,
        "gt_used_for_generation": False,
        "gt_used_for_diagnostic": True,
        "training_used": False,
        "generated_samples_this_run": generated,
        "reused_samples_this_run": reused,
        "device": str(device),
        "dino_key_hook": key_path,
        "runtime": {
            "dino_seconds": dino_seconds,
            "gbsp_seconds": gbsp_seconds,
            "metric_seconds": metric_seconds,
            "wall_seconds": wall_seconds,
        },
    }
    write_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if failures and strict_failures:
        raise RuntimeError(f"multi-view comparison failed: {len(failures)} images")
    if int(max_samples) == -1 and not formal:
        raise RuntimeError("formal full-4040 comparison did not complete")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--identity_manifest", default=str(DEFAULT_IDENTITY_MANIFEST))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict_failures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(
        config=args.config,
        identity_manifest=args.identity_manifest,
        output_root=args.output_root,
        threshold=args.threshold,
        max_samples=args.max_samples,
        device_name=args.device,
        torch_threads=args.torch_threads,
        force=args.force,
        strict_failures=args.strict_failures,
    )
