#!/usr/bin/env python3
"""Train-split oracle diagnostic scan for GBSP Hard68 thresholds.

The score path exactly matches training-label construction:
gbsp_abs_minmax_37 -> bilinear 68 -> strict threshold.  GT is resized to 68
with nearest-neighbor interpolation.  The formal 4040-image training split is
used (TR-CAMO 1000 + TR-COD10K 3040).  No model, checkpoint, or DINO forward
is used.  Training GT is used, so every selected threshold is oracle
diagnostic-only and is not a valid GT-free model-selection result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "workdir/gbsp_absmm_t063_train_identity/dinov1-s8/manifest_train.jsonl"
)
DEFAULT_FEATURE_MANIFEST = (
    REPO_ROOT / "datasets/cache/features_cache/dinov1-s8/manifest_train.jsonl"
)
DEFAULT_OUT = REPO_ROOT / "workdir/gbsp_hard68_train4040_threshold_scan"
DATASETS = ("TR-CAMO", "TR-COD10K")
EXPECTED_COUNTS = {
    "TR-CAMO": 1000,
    "TR-COD10K": 3040,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
METRICS = ("S_m", "E_mean", "MAE", "F_beta_w", "F_beta_mean")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    selected, cursor = [], 0
    while len(selected) < max_samples:
        progressed = False
        for dataset in DATASETS:
            if cursor < len(grouped[dataset]):
                selected.append(grouped[dataset][cursor])
                progressed = True
                if len(selected) == max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _attach_training_gt(
    cache_rows: list[dict], feature_rows: list[dict]
) -> list[dict]:
    feature_map = {
        (str(row["dataset"]), str(row["stem"])): row for row in feature_rows
    }
    if len(feature_map) != len(feature_rows):
        raise RuntimeError("training feature manifest contains duplicate identities")
    cache_keys = {(str(row["dataset"]), str(row["stem"])) for row in cache_rows}
    if len(cache_keys) != len(cache_rows):
        raise RuntimeError("GBSP training manifest contains duplicate identities")
    if cache_keys != set(feature_map):
        raise RuntimeError("GBSP/feature training manifest identities differ")
    attached = []
    for row in cache_rows:
        key = (str(row["dataset"]), str(row["stem"]))
        feature_row = feature_map[key]
        image_path = Path(feature_row["image_path"]).resolve()
        gt_path = image_path.parent.parent / "gt" / f"{key[1]}.png"
        if not image_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(
                f"formal training image/GT pair is incomplete: {image_path} {gt_path}"
            )
        attached.append(
            {
                **row,
                "image_path": str(image_path),
                "gt_path": str(gt_path),
            }
        )
    return attached


def _load_gt68(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    gt = torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)
    return F.interpolate(gt.unsqueeze(0), size=(68, 68), mode="nearest").squeeze(0)


def _load_gbsp68(row: dict) -> torch.Tensor:
    path = Path(row["cache_path"])
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"GBSP payload must be a dict: {path}")
    if str(payload.get("dataset")) != str(row["dataset"]) or str(
        payload.get("stem")
    ) != str(row["stem"]):
        raise RuntimeError(f"GBSP payload identity mismatch: {path}")
    if str(payload.get("gbsp_version")) != "gbsp_pca_absmm_v1":
        raise RuntimeError(f"unexpected GBSP cache version: {path}")
    if int(payload.get("num_subspaces", -1)) != 1:
        raise RuntimeError(f"GBSP cache is not the M=1 experiment: {path}")
    value = payload.get("gbsp_abs_minmax_37")
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"gbsp_abs_minmax_37 must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if (
        not bool(torch.isfinite(value).all())
        or float(value.min()) < -1e-6
        or float(value.max()) > 1.0 + 1e-6
    ):
        raise ValueError(f"gbsp_abs_minmax_37 must remain finite in [0,1]: {path}")
    return F.interpolate(
        value.clamp(0.0, 1.0).unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _scan_one(task: dict) -> dict:
    row = task["row"]
    try:
        gt68 = _load_gt68(row["gt_path"])
        score68 = _load_gbsp68(row)
        thresholds = np.asarray(task["thresholds"], dtype=np.float64)
        context = FastCODContext(gt68)
        candidates = [
            (f"t{index}", "hard", (score68 > float(threshold)).float())
            for index, threshold in enumerate(thresholds)
        ]
        evaluated = context.evaluate_many(candidates, 0.5)
        arrays = {
            metric: np.asarray(
                [float(evaluated[(f"t{index}", "hard")][metric]) for index in range(len(thresholds))],
                dtype=np.float64,
            )
            for metric in METRICS
        }
        return {
            "dataset": str(row["dataset"]),
            "stem": str(row["stem"]),
            "arrays": arrays,
        }
    except Exception as error:
        return {
            "dataset": str(row.get("dataset", "")),
            "stem": str(row.get("stem", "")),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _aggregate(
    results: list[dict], thresholds: np.ndarray, datasets: tuple[str, ...]
) -> list[dict]:
    rows = []
    dataset_arrays = {}
    for dataset in datasets:
        subset = [result for result in results if result["dataset"] == dataset]
        if not subset:
            continue
        arrays = {
            metric: np.mean(
                np.stack([result["arrays"][metric] for result in subset]), axis=0
            )
            for metric in METRICS
        }
        dataset_arrays[dataset] = arrays
        for index, threshold in enumerate(thresholds):
            s_value = float(arrays["S_m"][index])
            e_value = float(arrays["E_mean"][index])
            mae_value = float(arrays["MAE"][index])
            rows.append(
                {
                    "scope": "dataset",
                    "dataset": dataset,
                    "num_samples": len(subset),
                    "threshold": float(threshold),
                    **{metric: float(arrays[metric][index]) for metric in METRICS},
                    "J_S_E_1mMAE": (s_value + e_value + 1.0 - mae_value)
                    / 3.0,
                }
            )

    if results:
        arrays = {
            metric: np.mean(
                np.stack([result["arrays"][metric] for result in results]), axis=0
            )
            for metric in METRICS
        }
        for index, threshold in enumerate(thresholds):
            s_value = float(arrays["S_m"][index])
            e_value = float(arrays["E_mean"][index])
            mae_value = float(arrays["MAE"][index])
            rows.append(
                {
                    "scope": "train_sample_weighted",
                    "dataset": "TR-CAMO|TR-COD10K",
                    "num_samples": len(results),
                    "threshold": float(threshold),
                    **{metric: float(arrays[metric][index]) for metric in METRICS},
                    "J_S_E_1mMAE": (s_value + e_value + 1.0 - mae_value)
                    / 3.0,
                }
            )

    available = [dataset for dataset in DATASETS if dataset in dataset_arrays]
    if available:
        arrays = {
            metric: np.mean(
                np.stack([dataset_arrays[dataset][metric] for dataset in available]),
                axis=0,
            )
            for metric in METRICS
        }
        for index, threshold in enumerate(thresholds):
            s_value = float(arrays["S_m"][index])
            e_value = float(arrays["E_mean"][index])
            mae_value = float(arrays["MAE"][index])
            rows.append(
                {
                    "scope": "train_dataset_equal_macro",
                    "dataset": "|".join(available),
                    "num_samples": sum(
                        sum(result["dataset"] == dataset for result in results)
                        for dataset in available
                    ),
                    "threshold": float(threshold),
                    **{metric: float(arrays[metric][index]) for metric in METRICS},
                    "J_S_E_1mMAE": (s_value + e_value + 1.0 - mae_value) / 3.0,
                }
            )
    return rows


def _run_pass(
    selected_rows: list[dict],
    thresholds: np.ndarray,
    workers: int,
    torch_threads: int,
    failures_path: Path,
) -> tuple[list[dict], list[dict], float]:
    tasks = [
        {"row": row, "thresholds": [float(value) for value in thresholds]}
        for row in selected_rows
    ]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(int(torch_threads),),
    ) as pool:
        for index, result in enumerate(pool.map(_scan_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                print(f"Hard68 threshold scan {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if "error" in result]
    valid = [result for result in results if "error" not in result]
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return valid, failures, time.perf_counter() - started


def _selection_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["scope"] == "train_sample_weighted"]


def _best(rows: list[dict], field: str, maximize: bool = True) -> dict:
    finite = [row for row in rows if math.isfinite(float(row[field]))]
    if not finite:
        raise RuntimeError(f"no finite rows for {field}")
    key = lambda row: float(row[field])
    return max(finite, key=key) if maximize else min(finite, key=key)


def _fine_thresholds(center: float, radius: float, step: float) -> np.ndarray:
    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    count = int(round((upper - lower) / step))
    return np.asarray(
        [round(lower + index * step, 10) for index in range(count + 1)],
        dtype=np.float64,
    )


def _report(
    output_dir: Path,
    coarse_rows: list[dict],
    fine_rows: list[dict],
    selected: dict,
    plateau: list[dict],
    metadata: dict,
) -> None:
    fine_selection = _selection_rows(fine_rows)
    best_s = _best(fine_selection, "S_m")
    best_e = _best(fine_selection, "E_mean")
    best_mae = _best(fine_selection, "MAE", maximize=False)
    at_selected = [
        row
        for row in fine_rows
        if abs(float(row["threshold"]) - float(selected["threshold"])) < 1e-12
    ]
    lines = [
        "# GBSP Hard68 训练集 4040 张伪标签阈值扫描",
        "",
        "> Oracle 诊断：阈值选择读取了训练 GT，不可作为无 GT 调参的正式结果。",
        "",
        "## 协议",
        "",
        "- 样本：TR-CAMO 1000 + TR-COD10K 3040，共 4040 张。",
        "- 响应：训练缓存 `gbsp_abs_minmax_37`。",
        "- 标签：37→68 双线性后执行 `strict > threshold`。",
        "- GT：原始二值 GT 最近邻缩放至 68×68。",
        "- 主选择：4040 张逐图等权，与训练样本分布一致。",
        "- 附加报告：TR-CAMO、TR-COD10K 分项及两数据集等权宏平均。",
        "- 综合分：`J=(S+E+1-MAE)/3`。",
        "",
        "## 最优阈值",
        "",
        f"- 综合 J 最优：**{float(selected['threshold']):.3f}**，J={float(selected['J_S_E_1mMAE']):.6f}。",
        f"- S 最优：{float(best_s['threshold']):.3f}，S={float(best_s['S_m']):.6f}。",
        f"- E 最优：{float(best_e['threshold']):.3f}，E={float(best_e['E_mean']):.6f}。",
        f"- MAE 最优：{float(best_mae['threshold']):.3f}，MAE={float(best_mae['MAE']):.6f}。",
        f"- J 高分平台（距最大值≤{metadata['plateau_tolerance']:.6f}）："
        + (
            f"{min(float(row['threshold']) for row in plateau):.3f}～"
            f"{max(float(row['threshold']) for row in plateau):.3f}。"
            if plateau
            else "无。"
        ),
        "",
        "## 综合最优阈值下的完整指标",
        "",
        "| scope | dataset | S | Fβw | Fm | E | MAE | J |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in at_selected:
        lines.append(
            f"| {row['scope']} | {row['dataset']} | {float(row['S_m']):.6f} | "
            f"{float(row['F_beta_w']):.6f} | {float(row['F_beta_mean']):.6f} | "
            f"{float(row['E_mean']):.6f} | {float(row['MAE']):.6f} | "
            f"{float(row.get('J_S_E_1mMAE', float('nan'))):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 运行统计",
            "",
            f"- 粗扫有效/失败：{metadata['coarse_valid']}/{metadata['coarse_failed']}。",
            f"- 精扫有效/失败：{metadata['fine_valid']}/{metadata['fine_failed']}。",
            f"- 粗扫耗时：{metadata['coarse_seconds']:.3f} 秒。",
            f"- 精扫耗时：{metadata['fine_seconds']:.3f} 秒。",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--feature_manifest", default=str(DEFAULT_FEATURE_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--coarse_step", type=float, default=0.01)
    parser.add_argument("--fine_step", type=float, default=0.001)
    parser.add_argument("--fine_radius", type=float, default=0.02)
    parser.add_argument("--plateau_tolerance", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument("--strict_failures", action="store_true")
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    for name in ("coarse_step", "fine_step", "fine_radius"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("workers/torch_threads must be positive")

    manifest_path = _resolve(args.manifest)
    feature_manifest_path = _resolve(args.feature_manifest)
    output_dir = _resolve(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(manifest_path)
    counts = Counter(str(row["dataset"]) for row in manifest_rows)
    if len(manifest_rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"GBSP training manifest differs from formal protocol: "
            f"{len(manifest_rows)} {dict(counts)}"
        )
    feature_rows = read_jsonl(feature_manifest_path)
    feature_counts = Counter(str(row["dataset"]) for row in feature_rows)
    if len(feature_rows) != EXPECTED_TOTAL or dict(feature_counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"feature training manifest differs from formal protocol: "
            f"{len(feature_rows)} {dict(feature_counts)}"
        )
    paired_rows = _attach_training_gt(manifest_rows, feature_rows)
    selected_rows = _balanced_subset(paired_rows, int(args.max_samples))
    coarse_thresholds = np.arange(
        0.0, 1.0 + float(args.coarse_step) / 2.0, float(args.coarse_step)
    ).round(10)
    coarse_valid, coarse_failures, coarse_seconds = _run_pass(
        selected_rows,
        coarse_thresholds,
        args.workers,
        args.torch_threads,
        output_dir / "coarse_failures.json",
    )
    if not coarse_valid:
        raise RuntimeError("coarse scan produced no valid samples")
    coarse_rows = _aggregate(coarse_valid, coarse_thresholds, DATASETS)
    coarse_selection = _selection_rows(coarse_rows)
    coarse_best = _best(coarse_selection, "J_S_E_1mMAE")
    fine_thresholds = _fine_thresholds(
        float(coarse_best["threshold"]),
        float(args.fine_radius),
        float(args.fine_step),
    )
    fine_valid, fine_failures, fine_seconds = _run_pass(
        selected_rows,
        fine_thresholds,
        args.workers,
        args.torch_threads,
        output_dir / "fine_failures.json",
    )
    if not fine_valid:
        raise RuntimeError("fine scan produced no valid samples")
    fine_rows = _aggregate(fine_valid, fine_thresholds, DATASETS)
    fine_selection = _selection_rows(fine_rows)
    selected = _best(fine_selection, "J_S_E_1mMAE")
    plateau = [
        row
        for row in fine_selection
        if float(selected["J_S_E_1mMAE"])
        - float(row["J_S_E_1mMAE"])
        <= float(args.plateau_tolerance)
    ]

    fields = [
        "scope",
        "dataset",
        "num_samples",
        "threshold",
        "S_m",
        "F_beta_w",
        "F_beta_mean",
        "E_mean",
        "MAE",
        "J_S_E_1mMAE",
    ]
    _write_csv(output_dir / "coarse_scan.csv", coarse_rows, fields)
    _write_csv(output_dir / "fine_scan.csv", fine_rows, fields)
    metadata = {
        "schema": "gbsp_hard68_threshold_scan_v1",
        "manifest": str(manifest_path),
        "feature_manifest": str(feature_manifest_path),
        "output_dir": str(output_dir),
        "num_requested": len(selected_rows),
        "coarse_valid": len(coarse_valid),
        "coarse_failed": len(coarse_failures),
        "fine_valid": len(fine_valid),
        "fine_failed": len(fine_failures),
        "coarse_seconds": coarse_seconds,
        "fine_seconds": fine_seconds,
        "coarse_step": float(args.coarse_step),
        "fine_step": float(args.fine_step),
        "fine_radius": float(args.fine_radius),
        "plateau_tolerance": float(args.plateau_tolerance),
        "selection_datasets": list(DATASETS),
        "selection_dataset_weighting": "sample_weighted_1000_plus_3040",
        "selection_score": "(S_m + E_mean + 1 - MAE) / 3",
        "score_source": "GBSP train cache gbsp_abs_minmax_37",
        "pseudo_label_pipeline": "bilinear_37_to_68_then_strict_threshold",
        "gt_pipeline": "binary_original_to_68_nearest",
        "selected_threshold": float(selected["threshold"]),
        "selected_score": float(selected["J_S_E_1mMAE"]),
        "train_gt_used_for_selection": True,
        "test_gt_used_for_selection": False,
        "num_manifest_fallbacks": sum(
            bool(row.get("fallback_used", False)) for row in selected_rows
        ),
        "oracle_diagnostic_only": True,
        "training_used": False,
        "checkpoint_used": False,
        "dino_forward_used": False,
    }
    _report(output_dir, coarse_rows, fine_rows, selected, plateau, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and (coarse_failures or fine_failures):
        raise RuntimeError(
            f"scan failures: coarse={len(coarse_failures)}, fine={len(fine_failures)}"
        )


if __name__ == "__main__":
    main()
