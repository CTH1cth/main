#!/usr/bin/env python3
"""Evaluate GBSP rank-0 mean prototype against the formal rank-r subspace.

Calibration is fixed and GT-free for both methods: image-wise Min-Max followed
by a strict 0.5 threshold.  Absolute raw residuals are used for threshold-free
Pixel AP/AUROC.  No threshold search is implemented in this evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_rank0_ablation.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED_COUNTS = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
METHODS = ("GBSP-r0", "GBSP-r>0")
METRICS = (
    "S_m",
    "F_beta_w",
    "F_beta_mean",
    "E_mean",
    "MAE",
    "Precision",
    "Recall",
    "Area",
    "IoU",
    "Dice",
    "soft_S_m",
    "soft_F_beta_w",
    "soft_F_beta_mean",
    "soft_E_mean",
    "soft_MAE",
    "pixel_AP",
    "pixel_AUROC",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    mapping = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key):
            raise RuntimeError(f"invalid identity at {path}:{line}")
        if key in mapping:
            raise RuntimeError(f"duplicate identity at {path}:{line}: {key}")
        if not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row["cache_path"])
        mapping[key] = row
    return rows, mapping


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped = {dataset: [] for dataset in DATASETS}
    for row in rows:
        grouped[row["dataset"]].append(row)
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


def _tensor(payload: dict, field: str, path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()) or float(value.min()) < -1e-6:
        raise ValueError(f"{field} must be finite and non-negative: {path}")
    return value.clamp_min(0.0)


def _minmax(value: torch.Tensor, eps: float) -> torch.Tensor:
    minimum, maximum = value.min(), value.max()
    if float(maximum - minimum) <= eps:
        return torch.zeros_like(value)
    return ((value - minimum) / (maximum - minimum + eps)).clamp(0.0, 1.0)


def _resize(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = F.interpolate(
        value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )
    value = F.interpolate(value, size=shape, mode="bilinear", align_corners=False)
    return value.squeeze(0)


def _load_gt(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((value > 0.5).astype(np.float32)).unsqueeze(0)


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict:
    values = score.numpy().astype(np.float64).reshape(-1)
    labels = gt.numpy().reshape(-1) > 0.5
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}
    order = np.argsort(-values, kind="stable")
    ordered_score, ordered_label = values[order], labels[order]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positives
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    starts = np.r_[0, np.flatnonzero(ordered_score[1:] != ordered_score[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    group_pos = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_neg = ends - starts - group_pos
    negative_below = negatives - np.cumsum(group_neg)
    wins = np.sum(group_pos * (negative_below + 0.5 * group_neg), dtype=np.float64)
    return {"pixel_AP": ap, "pixel_AUROC": float(wins / (positives * negatives))}


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    x = rankdata(left.numpy().reshape(-1), method="average")
    y = rankdata(right.numpy().reshape(-1), method="average")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 1.0 if np.array_equal(x, y) else float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cod_metrics(context: FastCODContext, probability: torch.Tensor) -> dict:
    values = context.evaluate_many(
        [("candidate", "hard", probability), ("candidate", "soft", probability)],
        0.5,
    )
    hard = values[("candidate", "hard")]
    soft = values[("candidate", "soft")]
    precision, recall = float(hard["Precision"]), float(hard["Recall"])
    return {
        "S_m": float(hard["S_m"]),
        "F_beta_w": float(hard["F_beta_w"]),
        "F_beta_mean": float(hard["F_beta_mean"]),
        "E_mean": float(hard["E_mean"]),
        "MAE": float(hard["MAE"]),
        "Precision": precision,
        "Recall": recall,
        "Area": float(hard["Area"]),
        "IoU": float(hard["IoU"]),
        "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
        "soft_S_m": float(soft["S_m"]),
        "soft_F_beta_w": float(soft["F_beta_w"]),
        "soft_F_beta_mean": float(soft["F_beta_mean"]),
        "soft_E_mean": float(soft["E_mean"]),
        "soft_MAE": float(soft["MAE"]),
    }


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        rank0_path, rankr_path = Path(task["rank0_path"]), Path(task["rankr_path"])
        rank0 = torch_load(rank0_path, map_location="cpu")
        rankr = torch_load(rankr_path, map_location="cpu")
        for payload, path in ((rank0, rank0_path), (rankr, rankr_path)):
            if not isinstance(payload, dict):
                raise TypeError(f"cache payload must be dict: {path}")
            if (str(payload.get("dataset")), str(payload.get("stem"))) != (
                task["dataset"],
                task["stem"],
            ):
                raise RuntimeError(f"cache identity mismatch: {path}")

        settings0 = rank0.get("settings", {})
        settingsr = rankr.get("settings", {})
        if (
            int(rank0.get("num_requested_subspaces", -1)) != 1
            or int(rank0.get("num_effective_subspaces", -1)) != 1
            or int(settings0.get("pca_max_rank", -1)) != 0
            or int(settings0.get("pca_min_rank", -1)) != 0
            or rank0.get("selected_ranks", torch.tensor([-1])).tolist() != [0]
        ):
            raise RuntimeError(f"invalid formal rank-0 cache: {rank0_path}")
        if (
            int(rankr.get("num_requested_subspaces", -1)) != 1
            or int(rankr.get("num_effective_subspaces", -1)) != 1
            or not math.isclose(
                float(settingsr.get("pca_energy", float("nan"))),
                float(task["rankr_pca_energy"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or int(settingsr.get("pca_max_rank", -1)) != int(task["rankr_pca_max_rank"])
            or int(settingsr.get("pca_min_rank", -1)) != int(task["rankr_pca_min_rank"])
        ):
            raise RuntimeError(f"rank-r reference must be formal M=1 GBSP: {rankr_path}")

        shared_setting_keys = (
            "grid",
            "feature_dim",
            "num_subspaces",
            "min_cluster_size",
            "pca_energy",
            "seed",
            "eps",
            "full_bc_source",
            "bc_confidence_source",
            "feature_source",
        )
        for field in shared_setting_keys:
            if settings0.get(field) != settingsr.get(field):
                raise RuntimeError(
                    f"rank-0/rank-r shared setting differs ({field}): "
                    f"{settings0.get(field)!r} != {settingsr.get(field)!r}"
                )
        for field in ("source_feature_path", "source_dabe_path"):
            source0, sourcer = rank0.get(field), rankr.get(field)
            if not source0 or not sourcer or Path(source0).resolve() != Path(sourcer).resolve():
                raise RuntimeError(f"rank-0/rank-r {field} differs")
        if tuple(rank0.get("patch_grid_size", ())) != (37, 37) or tuple(
            rankr.get("patch_grid_size", ())
        ) != (37, 37):
            raise RuntimeError("rank-0/rank-r must both use the 37x37 patch grid")

        indices0 = rank0.get("background_indices")
        indicesr = rankr.get("background_indices")
        if not torch.is_tensor(indices0) or not torch.is_tensor(indicesr):
            raise ValueError("background_indices missing")
        indices_equal = torch.equal(indices0.long(), indicesr.long())
        if not indices_equal:
            raise RuntimeError("rank-0/rank-r background indices differ")
        mean0 = rank0.get("subspace_means")
        meanr = rankr.get("subspace_means")
        if not torch.is_tensor(mean0) or not torch.is_tensor(meanr) or mean0.shape != meanr.shape:
            raise ValueError("subspace_means missing or shape mismatch")
        mean_error = float((mean0.float() - meanr.float()).abs().max())
        if mean_error > float(task["mean_tolerance"]):
            raise RuntimeError(f"rank-0/rank-r background means differ: {mean_error}")

        q0 = _tensor(rank0, "absolute_raw", rank0_path)
        qr = _tensor(rankr, "absolute_raw", rankr_path)
        excess = qr - q0
        violation_count = int((excess > float(task["residual_tolerance"])).sum())
        max_violation = float(excess.max())
        if max_violation > float(task["fatal_residual_tolerance"]):
            raise RuntimeError(
                "PCA residual exceeds mean residual beyond tolerance: "
                f"max={max_violation}, count={violation_count}"
            )

        # GT-free calibration is completed before GT is opened.
        calibrated = {
            "GBSP-r0": _minmax(q0, float(task["eps"])),
            "GBSP-r>0": _minmax(qr, float(task["eps"])),
        }
        gt = _load_gt(Path(task["gt_path"]))
        shape = tuple(gt.shape[-2:])
        raw_native = {"GBSP-r0": _resize(q0, shape), "GBSP-r>0": _resize(qr, shape)}
        calibrated_native = {
            method: _resize(value, shape) for method, value in calibrated.items()
        }
        context = FastCODContext(gt)
        rows = []
        for method in METHODS:
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    **_cod_metrics(context, calibrated_native[method]),
                    **_rank_metrics(raw_native[method], gt),
                }
            )
        by_method = {row["method"]: row for row in rows}
        paired = {
            "dataset": task["dataset"],
            "stem": task["stem"],
            **{
                f"delta_{metric}": by_method["GBSP-r>0"][metric]
                - by_method["GBSP-r0"][metric]
                for metric in METRICS
            },
            "rankr_wins_pixel_AP": int(
                by_method["GBSP-r>0"]["pixel_AP"] > by_method["GBSP-r0"]["pixel_AP"]
            ),
            "raw_spearman": _spearman(raw_native["GBSP-r0"], raw_native["GBSP-r>0"]),
            "mean_residual_reduction": float((q0 - qr).mean()),
            "fraction_residual_reduced": float((qr < q0 - 1e-7).float().mean()),
            "background_count": int(indices0.numel()),
            "background_indices_equal": int(indices_equal),
            "source_feature_equal": 1,
            "source_dabe_equal": 1,
            "background_mean_max_abs_error": mean_error,
            "rank0_selected_rank": 0,
            "rankr_selected_rank": int(rankr["selected_ranks"].reshape(-1)[0]),
            "pointwise_residual_violation_count": violation_count,
            "pointwise_residual_max_violation": max(0.0, max_violation),
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        }
        return {"rows": rows, "paired": paired}
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""),
            "stem": task.get("stem", ""),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for method in METHODS:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if subset:
                output.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "method": method,
                        "num_samples": len(subset),
                        **{metric: _mean(row[metric] for row in subset) for metric in METRICS},
                    }
                )
    for method in METHODS:
        subset = [row for row in output if row["method"] == method]
        output.append(
            {
                "scope": "dataset_macro",
                "dataset": "ALL",
                "method": method,
                "num_samples": sum(row["num_samples"] for row in subset),
                **{metric: _mean(row[metric] for row in subset) for metric in METRICS},
            }
        )
    return output


def _aggregate_paired(rows: list[dict]) -> list[dict]:
    delta_fields = [key for key in rows[0] if key.startswith("delta_")]
    output = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        if subset:
            output.append(
                {
                    "scope": "dataset",
                    "dataset": dataset,
                    "num_samples": len(subset),
                    **{field: _mean(row[field] for row in subset) for field in delta_fields},
                    "rankr_pixel_AP_win_rate": _mean(row["rankr_wins_pixel_AP"] for row in subset),
                    "raw_spearman": _mean(row["raw_spearman"] for row in subset),
                    "mean_residual_reduction": _mean(row["mean_residual_reduction"] for row in subset),
                }
            )
    dataset_rows = list(output)
    output.append(
        {
            "scope": "dataset_macro",
            "dataset": "ALL",
            "num_samples": sum(row["num_samples"] for row in dataset_rows),
            **{field: _mean(row[field] for row in dataset_rows) for field in delta_fields},
            "rankr_pixel_AP_win_rate": _mean(
                row["rankr_pixel_AP_win_rate"] for row in dataset_rows
            ),
            "raw_spearman": _mean(row["raw_spearman"] for row in dataset_rows),
            "mean_residual_reduction": _mean(
                row["mean_residual_reduction"] for row in dataset_rows
            ),
        }
    )
    return output


def _verdict(summary: list[dict], paired: list[dict], complete: bool) -> dict:
    if not complete:
        return {"status": "INCOMPLETE", "reason": "not a complete failure-free 6473-image run"}
    macro = {row["method"]: row for row in summary if row["scope"] == "dataset_macro"}
    ap_delta = macro["GBSP-r>0"]["pixel_AP"] - macro["GBSP-r0"]["pixel_AP"]
    auc_delta = macro["GBSP-r>0"]["pixel_AUROC"] - macro["GBSP-r0"]["pixel_AUROC"]
    dataset_deltas = [
        row["delta_pixel_AP"] for row in paired if row["scope"] == "dataset"
    ]
    improved = sum(delta > 0.0 for delta in dataset_deltas)
    if ap_delta > 0.0 and auc_delta > 0.0 and improved >= 3:
        status = "PCA_SUBSPACE_SUPPORTED"
        reason = "rank-r improves AP/AUROC and AP on at least three datasets"
    elif ap_delta <= 0.0 and auc_delta <= 0.0:
        status = "MEAN_PROTOTYPE_SUFFICIENT"
        reason = "rank-r does not improve either macro AP or macro AUROC"
    else:
        status = "MIXED"
        reason = "rank-r gains are inconsistent across ranking criteria or datasets"
    return {
        "status": status,
        "reason": reason,
        "macro_pixel_AP_delta": ap_delta,
        "macro_pixel_AUROC_delta": auc_delta,
        "datasets_with_positive_AP_delta": improved,
    }


def _table(rows: list[dict], fields: tuple[str, ...]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" if field in {"method", "dataset"} else "---:" for field in fields) + "|"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                f"{float(row[field]):.6f}" if isinstance(row.get(field), (float, np.floating)) else str(row.get(field, ""))
                for field in fields
            )
            + " |"
        )
    return "\n".join((header, divider, *body))


def _report(output_dir: Path, summary: list[dict], paired: list[dict], metadata: dict) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    per_dataset = [row for row in summary if row["scope"] == "dataset"]
    dataset_delta = [row for row in paired if row["scope"] == "dataset"]
    lines = [
        "# GBSP Rank-0 Ablation Report",
        "",
        "- 唯一变量：是否使用同图 Full BC 拟合得到的 PCA 主方向。",
        "- Rank-0 未调用 SVD，分数为查询 Patch 到同一背景均值的平方欧氏距离。",
        "- 连续排序使用 Absolute Raw Pixel AP/AUROC。",
        "- 硬指标统一使用逐图 Min-Max 后 strict `>0.5`；未进行 GT 阈值搜索。",
        "- 校准阶段不读取 GT；GT 只用于响应生成后的评测。",
        "",
        "## 四数据集连续排序指标",
        "",
        _table(per_dataset, ("dataset", "method", "pixel_AP", "pixel_AUROC")),
        "",
        "## 四数据集固定校准 Hard 指标",
        "",
        _table(
            per_dataset,
            (
                "dataset", "method", "S_m", "F_beta_w", "F_beta_mean",
                "E_mean", "MAE", "Precision", "Recall", "Area",
            ),
        ),
        "",
        "## 四数据集宏平均",
        "",
        _table(
            macro,
            (
                "method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE",
                "Precision", "Recall", "Area", "pixel_AP", "pixel_AUROC",
            ),
        ),
        "",
        "## Rank-r 减 Rank-0 的逐数据集差值",
        "",
        _table(
            dataset_delta,
            (
                "dataset", "delta_pixel_AP", "delta_pixel_AUROC", "delta_F_beta_w",
                "delta_S_m", "delta_E_mean", "delta_MAE", "rankr_pixel_AP_win_rate",
            ),
        ),
        "",
        "## 裁决",
        "",
        f"- 状态：**{metadata['verdict']['status']}**。",
        f"- 原因：{metadata['verdict']['reason']}。",
        "",
        "## 控制变量与完整性",
        "",
        f"- 有效/失败：{metadata['num_valid']}/{metadata['num_failed']}。",
        f"- 背景索引不一致样本：{metadata['invariants']['background_index_mismatch_count']}。",
        f"- 源特征不一致样本：{metadata['invariants']['source_feature_mismatch_count']}。",
        f"- 源 DABE/Full BC 不一致样本：{metadata['invariants']['source_dabe_mismatch_count']}。",
        f"- 背景均值最大绝对误差：{metadata['invariants']['background_mean_max_abs_error']:.9g}。",
        f"- PCA残差大于Rank-0残差的最大数值误差：{metadata['invariants']['pointwise_residual_max_violation']:.9g}。",
        f"- Rank-r 选取秩分布：{metadata['rankr_selected_rank_histogram']}。",
        f"- 耗时：{metadata['wall_seconds']:.3f} 秒。",
    ]
    (output_dir / "GBSP_RANK0_ABLATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("--workers and --torch_threads must be positive")
    cfg = load_config(_resolve(args.config))
    rank0_root = _resolve(args.rank0_root or cfg.GBSP_RANK0_ROOT)
    rankr_root = _resolve(args.rankr_root or cfg.GBSP_RANKR_ROOT)
    output_dir = _resolve(args.out_dir or cfg.GBSP_RANK0_EVAL_ROOT)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankr_rows, rankr_map = _manifest(rankr_root / "manifest_test.jsonl")
    _, rank0_map = _manifest(rank0_root / "manifest_test.jsonl")
    selected = [row for row in rankr_rows if not args.dataset or row["dataset"] in args.dataset]
    selected = _balanced_subset(selected, args.max_samples)
    tasks = []
    for row in selected:
        key = (row["dataset"], row["stem"])
        if key not in rank0_map:
            raise KeyError(f"rank-0 manifest missing {key}")
        gt_path = row.get("gt_path") or rank0_map[key].get("gt_path")
        if not gt_path or not Path(gt_path).is_file():
            raise FileNotFoundError(gt_path or f"GT missing for {key}")
        tasks.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "rank0_path": rank0_map[key]["cache_path"],
                "rankr_path": rankr_map[key]["cache_path"],
                "gt_path": gt_path,
                "eps": float(getattr(cfg, "MBSP_EPS", 1e-8)),
                "mean_tolerance": args.mean_tolerance,
                "residual_tolerance": args.residual_tolerance,
                "fatal_residual_tolerance": args.fatal_residual_tolerance,
                "rankr_pca_energy": float(cfg.GBSP_RANKR_PCA_ENERGY),
                "rankr_pca_max_rank": int(cfg.GBSP_RANKR_PCA_MAX_RANK),
                "rankr_pca_min_rank": int(cfg.GBSP_RANKR_PCA_MIN_RANK),
            }
        )
    if not tasks:
        raise RuntimeError("no samples selected")

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.torch_threads,),
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP rank-0 ablation: {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if "error" in result]
    valid = [result for result in results if "error" not in result]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("rank-0 ablation produced no valid samples")

    per_image = [row for result in valid for row in result["rows"]]
    paired_rows = [result["paired"] for result in valid]
    summary = _aggregate(per_image)
    paired_summary = _aggregate_paired(paired_rows)
    counts = Counter(task["dataset"] for task in tasks)
    complete = (
        len(valid) == len(tasks) == EXPECTED_TOTAL
        and not failures
        and dict(counts) == EXPECTED_COUNTS
    )
    verdict = _verdict(summary, paired_summary, complete)
    metadata = {
        "schema": "gbsp_rank0_ablation_v1",
        "config": str(_resolve(args.config)),
        "rank0_root": str(rank0_root),
        "rankr_root": str(rankr_root),
        "output_dir": str(output_dir),
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "dataset_counts": dict(counts),
        "full_formal_evaluation": complete,
        "feature_extraction_used": False,
        "training_used": False,
        "calibration_gt_used": False,
        "evaluation_gt_used": True,
        "calibration": "per-image Min-Max then strict >0.5",
        "continuous_score": "Absolute Raw residual",
        "threshold_search_used": False,
        "wall_seconds": time.perf_counter() - started,
        "rankr_selected_rank_histogram": dict(
            sorted(Counter(row["rankr_selected_rank"] for row in paired_rows).items())
        ),
        "invariants": {
            "background_index_mismatch_count": sum(
                not bool(row["background_indices_equal"]) for row in paired_rows
            ),
            "source_feature_mismatch_count": sum(
                not bool(row["source_feature_equal"]) for row in paired_rows
            ),
            "source_dabe_mismatch_count": sum(
                not bool(row["source_dabe_equal"]) for row in paired_rows
            ),
            "background_mean_max_abs_error": max(
                float(row["background_mean_max_abs_error"]) for row in paired_rows
            ),
            "pointwise_residual_violation_count": sum(
                int(row["pointwise_residual_violation_count"]) for row in paired_rows
            ),
            "pointwise_residual_max_violation": max(
                float(row["pointwise_residual_max_violation"]) for row in paired_rows
            ),
        },
        "verdict": verdict,
    }
    _write_csv(output_dir / "per_image_metrics.csv", per_image)
    _write_csv(output_dir / "paired_metrics.csv", paired_rows)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "paired_summary.csv", paired_summary)
    _write_json(
        output_dir / "per_dataset_metrics.json",
        {"methods": summary, "paired_rankr_minus_rank0": paired_summary},
    )
    _write_json(output_dir / "metadata.json", metadata)
    _report(output_dir, summary, paired_summary, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and failures:
        raise RuntimeError(f"rank-0 evaluation recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rank0_root", "--rank0-root", dest="rank0_root")
    parser.add_argument("--rankr_root", "--rankr-root", dest="rankr_root")
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir")
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--mean_tolerance", type=float, default=1e-7)
    parser.add_argument("--residual_tolerance", type=float, default=1e-6)
    parser.add_argument("--fatal_residual_tolerance", type=float, default=1e-4)
    parser.add_argument("--strict_failures", "--strict-failures", dest="strict_failures", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
