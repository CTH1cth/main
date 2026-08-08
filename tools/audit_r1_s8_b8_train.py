#!/usr/bin/env python3
"""Paired offline quality audit for the frozen S/8 and B/8 training R1 caches.

No backbone or decoder is executed.  Every sample uses the same GT and the
same metric implementation.  Three projection protocols are reported:

1. ``train_grid_68``: the exact strict-binary supervision target against GT
   resized to 68x68 with nearest interpolation;
2. ``hard68_to_original``: the exact strict-binary supervision target,
   bilinearly projected to original GT size and thresholded again;
3. ``response_to_original``: the established repository R1 response audit,
   37 -> 68 -> original GT size with bilinear interpolation, then threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
BASELINE_ROOT = SCRIPT_PATH.parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

AUDIT_HELPER_DIR = (
    BASELINE_ROOT / "workdir" / "dabe_v2_stage_response_audit_identity"
)
if str(AUDIT_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_HELPER_DIR))

from stage_response_audit import (  # noqa: E402
    FastCODContext,
    MeanAccumulator,
    resize_for_stage,
)
from common.eval_dabe_pseudo_quality import _load_gt  # noqa: E402
from common.utils import find_gt_path, torch_load  # noqa: E402


EXPECTED_TOTAL = 4040
EXPECTED_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
METHODS = ("S8_R1", "B8_R1")
PROTOCOLS = (
    "train_grid_68",
    "hard68_to_original",
    "response_to_original",
)
METRIC_FIELDS = (
    "S_m",
    "F_beta^w",
    "F_beta^m",
    "E_phi^m",
    "COD_MAE",
    "IoU",
    "Precision",
    "Recall",
    "F1",
    "area_ratio",
    "num_components",
)
PAIR_FIELDS = tuple(f"delta_{field}" for field in METRIC_FIELDS) + (
    "B8_win_S_m",
    "B8_win_F_beta_w",
    "B8_win_E_phi_m",
    "B8_win_COD_MAE",
    "B8_win_IoU",
    "B8_win_Precision",
    "B8_win_Recall",
    "hard68_disagreement",
    "hard68_IoU_S8_B8",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s8-manifest",
        default=str(
            BASELINE_ROOT
            / "datasets/cache/dabe_cvbr_v1_train_singleview/dinov1-s8"
            / "manifest_train.jsonl"
        ),
    )
    parser.add_argument(
        "--b8-manifest",
        default=(
            "/home/dell01/CTH/UCOD-DPL-artifacts/cache/r1_identity_b8/"
            "manifest_train.jsonl"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(BASELINE_ROOT / "datasets/COD"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(BASELINE_ROOT / "workdir/r1_s8_b8_train_audit"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in ("dataset", "stem", "cache_path"):
                if field not in row:
                    raise KeyError(f"{field} missing at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def manifest_map(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_manifest(path)
    mapped = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in mapped:
            raise RuntimeError(f"Duplicate key in {path}: {key}")
        cache_path = Path(row["cache_path"]).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        mapped[key] = row
    return rows, mapped


def validate_inputs(s8_manifest: Path, b8_manifest: Path, data_root: Path):
    s8_rows, s8_map = manifest_map(s8_manifest)
    b8_rows, b8_map = manifest_map(b8_manifest)
    if len(s8_rows) != EXPECTED_TOTAL or len(b8_rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} rows, got S8={len(s8_rows)}, "
            f"B8={len(b8_rows)}"
        )
    if set(s8_map) != set(b8_map):
        missing_b8 = sorted(set(s8_map) - set(b8_map))
        missing_s8 = sorted(set(b8_map) - set(s8_map))
        raise RuntimeError(
            f"S8/B8 key mismatch: missing_b8={missing_b8[:3]}, "
            f"missing_s8={missing_s8[:3]}"
        )
    counts = defaultdict(int)
    tasks = []
    for dataset, stem in sorted(s8_map):
        counts[dataset] += 1
        gt_path = find_gt_path(data_root, dataset, stem).resolve()
        tasks.append(
            {
                "dataset": dataset,
                "stem": stem,
                "gt_path": str(gt_path),
                "s8_cache_path": str(Path(s8_map[(dataset, stem)]["cache_path"]).resolve()),
                "b8_cache_path": str(Path(b8_map[(dataset, stem)]["cache_path"]).resolve()),
            }
        )
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected dataset counts: {dict(counts)}")
    return tasks, dict(counts)


def load_response(payload: dict, key: str, cache_path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a tensor: {cache_path}")
    value = value.detach().cpu().float().contiguous()
    if tuple(value.shape) != (1, 37, 37):
        raise RuntimeError(f"{key} must be [1,37,37]: {cache_path}")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{key} contains NaN/Inf: {cache_path}")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise RuntimeError(f"{key} is outside [0,1]: {cache_path}")
    return value


def strict_hard68(response: torch.Tensor, threshold: float) -> torch.Tensor:
    resized = F.interpolate(
        response.unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (resized > float(threshold)).float().contiguous()


def worker_init(torch_threads: int):
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _identity_check(payload: dict, dataset: str, stem: str, path: Path):
    if str(payload.get("dataset")) != dataset or str(payload.get("stem")) != stem:
        raise RuntimeError(f"Payload identity mismatch: {path}")
    if bool(payload.get("gt_used_for_generation", True)):
        raise RuntimeError(f"GT leakage flag is true: {path}")
    if list(payload.get("source_augs", [])) != ["identity"]:
        raise RuntimeError(f"Not an identity-only cache: {path}")
    if int(payload.get("source_num_views", 0)) != 1:
        raise RuntimeError(f"Not a single-view cache: {path}")


def _pair_values(s8: dict, b8: dict, hard68_s8: torch.Tensor, hard68_b8: torch.Tensor):
    values = {
        f"delta_{field}": float(b8[field]) - float(s8[field])
        for field in METRIC_FIELDS
    }
    values.update(
        {
            "B8_win_S_m": float(b8["S_m"] > s8["S_m"]),
            "B8_win_F_beta_w": float(b8["F_beta^w"] > s8["F_beta^w"]),
            "B8_win_E_phi_m": float(b8["E_phi^m"] > s8["E_phi^m"]),
            "B8_win_COD_MAE": float(b8["COD_MAE"] < s8["COD_MAE"]),
            "B8_win_IoU": float(b8["IoU"] > s8["IoU"]),
            "B8_win_Precision": float(b8["Precision"] > s8["Precision"]),
            "B8_win_Recall": float(b8["Recall"] > s8["Recall"]),
        }
    )
    left, right = hard68_s8.bool(), hard68_b8.bool()
    intersection = float((left & right).sum().item())
    union = float((left | right).sum().item())
    values["hard68_disagreement"] = float((left != right).float().mean().item())
    values["hard68_IoU_S8_B8"] = 1.0 if union == 0.0 else intersection / union
    return values


def process_one(task):
    dataset, stem = str(task["dataset"]), str(task["stem"])
    s8_path = Path(task["s8_cache_path"])
    b8_path = Path(task["b8_cache_path"])
    s8_payload = torch_load(s8_path, map_location="cpu")
    b8_payload = torch_load(b8_path, map_location="cpu")
    if not isinstance(s8_payload, dict) or not isinstance(b8_payload, dict):
        raise TypeError(f"R1 payload must be dict: {dataset}/{stem}")
    _identity_check(s8_payload, dataset, stem, s8_path)
    _identity_check(b8_payload, dataset, stem, b8_path)

    threshold = float(task["threshold"])
    response_s8 = load_response(s8_payload, "b0_r1_bw2_37", s8_path)
    response_b8 = load_response(b8_payload, "residual_pass1_37", b8_path)
    hard68_s8 = strict_hard68(response_s8, threshold)
    hard68_b8 = strict_hard68(response_b8, threshold)

    stored_b8 = b8_payload.get("r1_hard_68")
    if not torch.is_tensor(stored_b8) or tuple(stored_b8.shape) != (1, 68, 68):
        raise RuntimeError(f"Invalid stored B8 r1_hard_68: {b8_path}")
    b8_hard68_max_error = float((stored_b8.float() - hard68_b8).abs().max().item())
    if b8_hard68_max_error != 0.0:
        raise RuntimeError(f"Stored B8 hard68 mismatch: {b8_path}")

    gt = _load_gt(task["gt_path"]).float()
    gt68 = F.interpolate(
        gt.unsqueeze(0), size=(68, 68), mode="nearest"
    ).squeeze(0)

    grid_context = FastCODContext(gt68)
    grid_eval = grid_context.evaluate_many(
        [
            ("S8_R1", "hard", hard68_s8),
            ("B8_R1", "hard", hard68_b8),
        ],
        threshold,
    )

    hard_original_s8 = F.interpolate(
        hard68_s8.unsqueeze(0),
        size=tuple(gt.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    hard_original_b8 = F.interpolate(
        hard68_b8.unsqueeze(0),
        size=tuple(gt.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    response_original_s8 = resize_for_stage(response_s8, gt, loss_size=68)
    response_original_b8 = resize_for_stage(response_b8, gt, loss_size=68)
    original_context = FastCODContext(gt)
    original_eval = original_context.evaluate_many(
        [
            ("hard68/S8_R1", "hard", hard_original_s8),
            ("hard68/B8_R1", "hard", hard_original_b8),
            ("response/S8_R1", "hard", response_original_s8),
            ("response/B8_R1", "hard", response_original_b8),
        ],
        threshold,
    )

    by_protocol = {
        "train_grid_68": {
            method: grid_eval[(method, "hard")] for method in METHODS
        },
        "hard68_to_original": {
            method: original_eval[(f"hard68/{method}", "hard")]
            for method in METHODS
        },
        "response_to_original": {
            method: original_eval[(f"response/{method}", "hard")]
            for method in METHODS
        },
    }
    metric_rows = []
    pair_rows = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            metric_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "protocol": protocol,
                    "method": method,
                    **by_protocol[protocol][method],
                }
            )
        pair_rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "protocol": protocol,
                **_pair_values(
                    by_protocol[protocol]["S8_R1"],
                    by_protocol[protocol]["B8_R1"],
                    hard68_s8,
                    hard68_b8,
                ),
            }
        )
    return {
        "metric_rows": metric_rows,
        "pair_rows": pair_rows,
        "b8_hard68_max_error": b8_hard68_max_error,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(metric_rows: list[dict]):
    accumulators = defaultdict(lambda: MeanAccumulator(METRIC_FIELDS))
    for row in metric_rows:
        keys = (
            ("dataset", row["dataset"], row["protocol"], row["method"]),
            ("sample_weighted", "ALL", row["protocol"], row["method"]),
        )
        for key in keys:
            accumulators[key].add(row)
    rows = []
    for (scope, dataset, protocol, method), accumulator in sorted(accumulators.items()):
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "protocol": protocol,
                "method": method,
                **accumulator.result(),
            }
        )
    dataset_rows = [row for row in rows if row["scope"] == "dataset"]
    for protocol in PROTOCOLS:
        for method in METHODS:
            selected = [
                row
                for row in dataset_rows
                if row["protocol"] == protocol and row["method"] == method
            ]
            macro = {
                field: float(np.mean([float(row[field]) for row in selected]))
                for field in METRIC_FIELDS
            }
            rows.append(
                {
                    "scope": "dataset_macro",
                    "dataset": "ALL",
                    "protocol": protocol,
                    "method": method,
                    **macro,
                    "num_samples": sum(int(row["num_samples"]) for row in selected),
                }
            )
    return sorted(rows, key=lambda row: (row["protocol"], row["scope"], row["dataset"], row["method"]))


def aggregate_pairs(pair_rows: list[dict]):
    accumulators = defaultdict(lambda: MeanAccumulator(PAIR_FIELDS))
    for row in pair_rows:
        keys = (
            ("dataset", row["dataset"], row["protocol"]),
            ("sample_weighted", "ALL", row["protocol"]),
        )
        for key in keys:
            accumulators[key].add(row)
    rows = []
    for (scope, dataset, protocol), accumulator in sorted(accumulators.items()):
        rows.append(
            {
                "scope": scope,
                "dataset": dataset,
                "protocol": protocol,
                **accumulator.result(),
            }
        )
    dataset_rows = [row for row in rows if row["scope"] == "dataset"]
    for protocol in PROTOCOLS:
        selected = [row for row in dataset_rows if row["protocol"] == protocol]
        macro = {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in PAIR_FIELDS
        }
        rows.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "protocol": protocol,
                **macro,
                "num_samples": sum(int(row["num_samples"]) for row in selected),
            }
        )
    return sorted(rows, key=lambda row: (row["protocol"], row["scope"], row["dataset"]))


def markdown_report(metric_summary: list[dict], pair_summary: list[dict], elapsed: float):
    lines = [
        "# DINOv1-S/8 与 DINOv1-B/8 训练集 R1 离线质量审计",
        "",
        "- 样本：TR-CAMO 1000 + TR-COD10K 3040，共 4040 张。",
        "- 两者：identity 单视图、同一 R1 公式参数、同一 0.5 严格阈值。",
        "- S/8 输入 resize：bicubic（仓库默认）；B/8 输入 resize：bilinear（UCOD-DPL 协议）。",
        "- 不运行 DINO、解码器或训练；不保存预测图。",
        f"- 审计耗时：{elapsed:.3f} 秒。",
        "",
    ]
    protocol_titles = {
        "train_grid_68": "实际训练监督网格（68×68）",
        "hard68_to_original": "实际 Hard-68 投影到原始 GT 尺寸",
        "response_to_original": "既有官方 R1 response 原图协议",
    }
    for protocol in PROTOCOLS:
        lines.extend(
            [
                f"## {protocol_titles[protocol]}",
                "",
                "| 聚合 | 方法 | S_m | F_beta^w | E_phi^m | MAE | IoU | Precision | Recall | Area |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = [
            row
            for row in metric_summary
            if row["protocol"] == protocol
            and row["scope"] in {"dataset", "sample_weighted", "dataset_macro"}
        ]
        order = {"TR-CAMO": 0, "TR-COD10K": 1, "ALL": 2}
        scope_order = {"dataset": 0, "sample_weighted": 1, "dataset_macro": 2}
        selected.sort(
            key=lambda row: (
                order.get(row["dataset"], 9),
                scope_order.get(row["scope"], 9),
                row["method"],
            )
        )
        for row in selected:
            aggregation = row["dataset"]
            if row["scope"] == "sample_weighted":
                aggregation = "ALL（样本加权）"
            elif row["scope"] == "dataset_macro":
                aggregation = "ALL（数据集宏平均）"
            lines.append(
                "| {agg} | {method} | {sm:.4f} | {fw:.4f} | {em:.4f} | "
                "{mae:.4f} | {iou:.4f} | {precision:.4f} | {recall:.4f} | {area:.4f} |".format(
                    agg=aggregation,
                    method=row["method"],
                    sm=row["S_m"],
                    fw=row["F_beta^w"],
                    em=row["E_phi^m"],
                    mae=row["COD_MAE"],
                    iou=row["IoU"],
                    precision=row["Precision"],
                    recall=row["Recall"],
                    area=row["area_ratio"],
                )
            )
        paired = next(
            row
            for row in pair_summary
            if row["protocol"] == protocol and row["scope"] == "sample_weighted"
        )
        lines.extend(
            [
                "",
                "B/8 − S/8（4040 张样本加权）："
                f"S_m {paired['delta_S_m']:+.4f}，"
                f"F_beta^w {paired['delta_F_beta^w']:+.4f}，"
                f"E_phi^m {paired['delta_E_phi^m']:+.4f}，"
                f"MAE {paired['delta_COD_MAE']:+.4f}，"
                f"IoU {paired['delta_IoU']:+.4f}。",
                "",
            ]
        )
    headline = next(
        row
        for row in pair_summary
        if row["protocol"] == "hard68_to_original"
        and row["scope"] == "sample_weighted"
    )
    fw_direction = "更高" if headline["delta_F_beta^w"] > 0 else "更低"
    mae_direction = "更低" if headline["delta_COD_MAE"] < 0 else "更高"
    lines.extend(
        [
            "## 结论",
            "",
            "按实际 Hard-68 伪监督投影到原始 GT 的主协议，"
            f"B/8 的 F_beta^w {fw_direction}，MAE {mae_direction}。"
            "具体结论以表格中的成对差值和两个训练子集的一致性为准。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    started = time.time()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or positive")
    if args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("--workers and --torch-threads must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1]")

    s8_manifest = Path(args.s8_manifest).resolve()
    b8_manifest = Path(args.b8_manifest).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    for path in (s8_manifest, b8_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, full_counts = validate_inputs(s8_manifest, b8_manifest, data_root)
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    for task in tasks:
        task["threshold"] = float(args.threshold)

    print(f"samples={len(tasks)} workers={args.workers}", flush=True)
    metric_rows = []
    pair_rows = []
    max_b8_hard68_error = 0.0
    executor = ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.torch_threads,),
    )
    try:
        results = executor.map(process_one, tasks, chunksize=1)
        for index, result in enumerate(results, 1):
            metric_rows.extend(result["metric_rows"])
            pair_rows.extend(result["pair_rows"])
            max_b8_hard68_error = max(
                max_b8_hard68_error, float(result["b8_hard68_max_error"])
            )
            if index == len(tasks) or index % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"processed={index}/{len(tasks)} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    metric_summary = aggregate_rows(metric_rows)
    pair_summary = aggregate_pairs(pair_rows)
    elapsed = time.time() - started
    metric_row_fields = ("dataset", "stem", "protocol", "method", *METRIC_FIELDS)
    pair_row_fields = ("dataset", "stem", "protocol", *PAIR_FIELDS)
    metric_summary_fields = (
        "scope", "dataset", "protocol", "method", *METRIC_FIELDS, "num_samples"
    )
    pair_summary_fields = (
        "scope", "dataset", "protocol", *PAIR_FIELDS, "num_samples"
    )
    write_csv(out_dir / "per_sample_metrics.csv", metric_rows, metric_row_fields)
    write_csv(out_dir / "per_sample_pairs.csv", pair_rows, pair_row_fields)
    write_csv(out_dir / "metrics_summary.csv", metric_summary, metric_summary_fields)
    write_csv(out_dir / "paired_summary.csv", pair_summary, pair_summary_fields)

    report = {
        "purpose": "paired offline S8/B8 training R1 quality audit",
        "num_samples": len(tasks),
        "full_cache_dataset_counts": full_counts,
        "selected_dataset_counts": dict(
            sorted(
                {
                    dataset: sum(task["dataset"] == dataset for task in tasks)
                    for dataset in EXPECTED_COUNTS
                }.items()
            )
        ),
        "threshold": float(args.threshold),
        "protocols": {
            "train_grid_68": (
                "R1_37 -> bilinear 68 align_corners=False -> strict >0.5; "
                "GT -> nearest 68"
            ),
            "hard68_to_original": (
                "strict Hard-R1_68 -> bilinear original GT size -> strict >0.5"
            ),
            "response_to_original": (
                "R1_37 -> bilinear 68 -> bilinear original GT size -> strict >0.5"
            ),
        },
        "methods": {
            "S8_R1": {
                "backbone": "DINOv1-S/8",
                "channels": 384,
                "feature_input_size": 296,
                "feature_resize_interpolation": "bicubic (repository default)",
                "source_key": "b0_r1_bw2_37",
                "manifest": str(s8_manifest),
                "manifest_sha256": sha256_file(s8_manifest),
            },
            "B8_R1": {
                "backbone": "DINOv1-B/8",
                "channels": 768,
                "feature_input_size": 296,
                "feature_resize_interpolation": "bilinear (UCOD-DPL protocol)",
                "source_key": "residual_pass1_37",
                "manifest": str(b8_manifest),
                "manifest_sha256": sha256_file(b8_manifest),
            },
        },
        "cache_contract": {
            "source_augs": ["identity"],
            "source_num_views": 1,
            "gt_used_for_generation": False,
            "grid": 37,
            "loss_size": 68,
            "same_effective_r1_params": True,
            "b8_stored_hard68_max_abs_error": max_b8_hard68_error,
        },
        "aggregation": {
            "dataset": "per-image mean within each training dataset",
            "sample_weighted": "per-image mean across all 4040 samples",
            "dataset_macro": "equal mean of TR-CAMO and TR-COD10K summaries",
            "paired_delta": "B8 minus S8 on the exact same image",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "workers": int(args.workers),
            "torch_threads_per_worker": int(args.torch_threads),
        },
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "metric_helper": str(AUDIT_HELPER_DIR / "stage_response_audit.py"),
        "metric_helper_sha256": sha256_file(
            AUDIT_HELPER_DIR / "stage_response_audit.py"
        ),
        "elapsed_seconds": elapsed,
        "metrics_summary": metric_summary,
        "paired_summary": pair_summary,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    (out_dir / "RESULTS.md").write_text(
        markdown_report(metric_summary, pair_summary, elapsed),
        encoding="utf-8",
    )
    print(f"elapsed_seconds={elapsed:.3f}", flush=True)
    print(f"wrote={out_dir / 'summary.json'}", flush=True)
    print(f"wrote={out_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
