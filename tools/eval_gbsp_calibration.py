#!/usr/bin/env python3
"""Stage-A GBSP calibration audit: exact R1 foreground-area transfer.

The calibration path is GT-free.  Ground truth is opened only after the R1
area has been transferred onto the GBSP absolute-residual ranking, and is used
solely for evaluation and error-transition diagnostics.
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
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_calibration import exact_area_transfer_mask  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_calibration.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
PILOT_TOTAL = 200
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
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            raise RuntimeError(f"duplicate identity in {path}: {key}")
        mapping[key] = row
    return rows, mapping


def _gbsp_manifest(root: Path) -> Path:
    candidates = (
        root / "manifest_test.jsonl",
        root / "ablations/M1_pilot200/manifest_test.jsonl",
        root / "dinov1-s8/manifest_test.jsonl",
    )
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"GBSP manifest not found below {root}")
    # Prefer the formal M=1 cache when a historical parent root is supplied.
    for path in found:
        if "M1_pilot200" in path.parts:
            return path
    return found[0]


def _materialize_pilot_list(path: Path, source_manifest: Path) -> list[tuple[str, str]]:
    if path.is_file():
        keys = []
        for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.replace("/", "\t", 1).split()
            if len(parts) != 2:
                raise ValueError(f"bad pilot identity at {path}:{line}: {raw!r}")
            keys.append((parts[0], parts[1]))
    else:
        rows, _ = _manifest(source_manifest)
        if len(rows) != PILOT_TOTAL:
            raise RuntimeError(
                f"historical pilot manifest must contain {PILOT_TOTAL} rows, got {len(rows)}"
            )
        keys = [(str(row["dataset"]), str(row["stem"])) for row in rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        text = [
            "# Exact identities reused from the historical MBSP pilot200.",
            "# Format: dataset<TAB>stem",
            *(f"{dataset}\t{stem}" for dataset, stem in keys),
        ]
        path.write_text("\n".join(text) + "\n", encoding="utf-8")
    if len(keys) != len(set(keys)):
        raise RuntimeError("pilot list contains duplicate identities")
    if len(keys) != PILOT_TOTAL:
        raise RuntimeError(f"pilot list must contain exactly {PILOT_TOTAL} identities")
    unknown = sorted({dataset for dataset, _ in keys} - set(DATASETS))
    if unknown:
        raise RuntimeError(f"pilot list contains unknown datasets: {unknown}")
    return keys


def _tensor(payload: dict, key: str, path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{key} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} contains NaN/Inf: {path}")
    return value


def _resize_score(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    value = F.interpolate(
        value.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )
    value = F.interpolate(value, size=shape, mode="bilinear", align_corners=False)
    return value.squeeze(0)


def _resize_binary(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return (_resize_score(value.float(), shape) > 0.5).float()


def _load_gt(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _rank_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict:
    values = score.numpy().astype(np.float64).reshape(-1)
    labels = gt.numpy().reshape(-1) > 0.5
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return {"pixel_AP": float("nan"), "pixel_AUROC": float("nan")}
    order = np.argsort(-values, kind="stable")
    ordered_value, ordered_label = values[order], labels[order]
    tp = np.cumsum(ordered_label, dtype=np.float64)
    fp = np.cumsum(~ordered_label, dtype=np.float64)
    group_end = np.r_[ordered_value[1:] != ordered_value[:-1], True]
    precision = tp[group_end] / (tp[group_end] + fp[group_end])
    recall = tp[group_end] / positives
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    starts = np.r_[0, np.flatnonzero(ordered_value[1:] != ordered_value[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    group_pos = np.add.reduceat(ordered_label.astype(np.int64), starts)
    group_neg = ends - starts - group_pos
    negative_below = negatives - np.cumsum(group_neg)
    wins = np.sum(group_pos * (negative_below + 0.5 * group_neg), dtype=np.float64)
    return {"pixel_AP": ap, "pixel_AUROC": float(wins / (positives * negatives))}


def _binary_metrics(context: FastCODContext, mask: torch.Tensor) -> dict:
    values = context.evaluate_many([("candidate", "hard", mask.float())], 0.5)[
        ("candidate", "hard")
    ]
    precision, recall = float(values["Precision"]), float(values["Recall"])
    dice = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "S_m": float(values["S_m"]),
        "F_beta_w": float(values["F_beta_w"]),
        "F_beta_mean": float(values["F_beta_mean"]),
        "E_mean": float(values["E_mean"]),
        "MAE": float(values["MAE"]),
        "Precision": precision,
        "Recall": recall,
        "Area": float(values["Area"]),
        "IoU": float(values["IoU"]),
        "Dice": dice,
    }


def _transition(candidate: torch.Tensor, r1: torch.Tensor, gt: torch.Tensor) -> dict:
    candidate = candidate.bool()
    r1 = r1.bool()
    gt = gt.bool()
    r1_fp = r1 & ~gt
    r1_tp = r1 & gt
    removed_fp = r1_fp & ~candidate
    lost_tp = r1_tp & ~candidate
    new_fp = ~r1 & candidate & ~gt
    new_tp = ~r1 & candidate & gt
    intersection = int((candidate & r1).sum())
    union = int((candidate | r1).sum())
    return {
        "r1_mask_iou": intersection / union if union else 1.0,
        "r1_fp_count": int(r1_fp.sum()),
        "r1_tp_count": int(r1_tp.sum()),
        "removed_r1_fp_count": int(removed_fp.sum()),
        "lost_r1_tp_count": int(lost_tp.sum()),
        "new_fp_count": int(new_fp.sum()),
        "new_tp_count": int(new_tp.sum()),
        "FPRemoval": float(removed_fp.sum()) / max(int(r1_fp.sum()), 1),
        "TPRetention": float((r1_tp & candidate).sum()) / max(int(r1_tp.sum()), 1),
    }


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        gbsp_path, r1_path = Path(task["gbsp_path"]), Path(task["r1_path"])
        gbsp_payload = torch_load(gbsp_path, map_location="cpu")
        r1_payload = torch_load(r1_path, map_location="cpu")
        for payload, path in ((gbsp_payload, gbsp_path), (r1_payload, r1_path)):
            if not isinstance(payload, dict):
                raise TypeError(f"cache payload must be a dict: {path}")
            if str(payload.get("dataset")) != task["dataset"] or str(
                payload.get("stem")
            ) != task["stem"]:
                raise RuntimeError(f"cache identity mismatch: {path}")
        if int(gbsp_payload.get("num_requested_subspaces", -1)) != 1:
            raise RuntimeError(f"stage A requires the M=1 GBSP cache: {gbsp_path}")

        # Calibration happens before GT is opened.
        absolute_raw = _tensor(gbsp_payload, "absolute_raw", gbsp_path)
        absolute_minmax = _tensor(gbsp_payload, "absolute_minmax", gbsp_path)
        r1_score = _tensor(r1_payload, "residual_pass1_37", r1_path)
        r1_patch_mask = r1_score > float(task["r1_threshold"])
        transferred = exact_area_transfer_mask(absolute_raw, r1_patch_mask)
        if transferred.source_foreground_count != transferred.selected_foreground_count:
            raise RuntimeError("area transfer did not preserve patch count")

        gt = _load_gt(Path(task["gt_path"]))
        shape = tuple(gt.shape[-2:])
        r1_native_score = _resize_score(r1_score, shape)
        gbsp_raw_native = _resize_score(absolute_raw, shape)
        gbsp_minmax_native = _resize_score(absolute_minmax, shape)
        masks = {
            "R1": (r1_native_score > float(task["r1_threshold"])).float(),
            "GBSP-MinMax-0.5": (
                gbsp_minmax_native > float(task["minmax_threshold"])
            ).float(),
            "GBSP-R1-Area": _resize_binary(transferred.binary_mask, shape),
        }
        context = FastCODContext(gt)
        rows = []
        for method, mask in masks.items():
            rows.append(
                {
                    "dataset": task["dataset"],
                    "stem": task["stem"],
                    "method": method,
                    **_binary_metrics(context, mask),
                }
            )
        transitions = _transition(masks["GBSP-R1-Area"], masks["R1"], gt)
        rank = {
            "R1": _rank_metrics(r1_native_score, gt),
            "GBSP-Absolute-Raw": _rank_metrics(gbsp_raw_native, gt),
        }
        diagnostic = {
            "dataset": task["dataset"],
            "stem": task["stem"],
            "r1_patch_foreground_count": transferred.source_foreground_count,
            "gbsp_patch_foreground_count": transferred.selected_foreground_count,
            "patch_count_error": (
                transferred.selected_foreground_count
                - transferred.source_foreground_count
            ),
            "r1_patch_area": transferred.source_area_ratio,
            "gbsp_patch_area": transferred.selected_area_ratio,
            "area_transfer_cutoff_raw": transferred.cutoff_score,
            "cutoff_tie_count": transferred.boundary_tie_count,
            "gt_area": float(gt.mean()),
            **transitions,
            "R1_pixel_AP": rank["R1"]["pixel_AP"],
            "R1_pixel_AUROC": rank["R1"]["pixel_AUROC"],
            "GBSP_pixel_AP": rank["GBSP-Absolute-Raw"]["pixel_AP"],
            "GBSP_pixel_AUROC": rank["GBSP-Absolute-Raw"]["pixel_AUROC"],
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        }
        return {"rows": rows, "diagnostic": diagnostic}
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
    methods = tuple(dict.fromkeys(row["method"] for row in rows))
    for dataset in DATASETS:
        for method in methods:
            subset = [
                row for row in rows if row["dataset"] == dataset and row["method"] == method
            ]
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
    for method in methods:
        subset = [
            row for row in output if row["scope"] == "dataset" and row["method"] == method
        ]
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


def _correlation(left, right) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _decision(summary: list[dict], complete: bool) -> dict:
    if not complete:
        return {"status": "INCOMPLETE", "reason": "pilot200 contains failures"}
    macro = {
        row["method"]: row for row in summary if row["scope"] == "dataset_macro"
    }
    candidate, baseline = macro["GBSP-R1-Area"], macro["R1"]
    fw = float(candidate["F_beta_w"])
    precision_gain = float(candidate["Precision"] - baseline["Precision"])
    by_dataset = [
        row
        for row in summary
        if row["scope"] == "dataset" and row["method"] == "GBSP-R1-Area"
    ]
    baseline_dataset = {
        row["dataset"]: row
        for row in summary
        if row["scope"] == "dataset" and row["method"] == "R1"
    }
    improved = sum(
        float(row["F_beta_w"]) >= float(baseline_dataset[row["dataset"]]["F_beta_w"])
        for row in by_dataset
    )
    if fw >= 0.6243 or (fw >= 0.6220 and precision_gain >= 0.005):
        status = "CONTINUE_STAGE_B"
        reason = "area-transfer continuation criterion satisfied"
    elif fw < 0.6180 and improved <= 1:
        status = "STOP_COMPLEX_CALIBRATION"
        reason = "area-transfer stop criterion satisfied"
    else:
        status = "BORDERLINE_REVIEW_REQUIRED"
        reason = "result lies between predefined continuation/stop regions"
    return {
        "status": status,
        "reason": reason,
        "candidate_macro_F_beta_w": fw,
        "precision_gain_vs_r1": precision_gain,
        "datasets_non_degraded_F_beta_w": improved,
    }


def _report(output_dir: Path, summary: list[dict], diagnostics: list[dict], metadata: dict) -> None:
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    fields = ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area", "IoU")
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" if field == "method" else "---:" for field in fields) + "|"
    body = [
        "| "
        + " | ".join(
            str(row[field]) if field == "method" else f"{float(row[field]):.6f}"
            for field in fields
        )
        + " |"
        for row in macro
    ]
    decision = metadata["decision"]
    lines = [
        "# GBSP 校准阶段 A：R1 面积迁移诊断",
        "",
        "- 校准阶段未读取 GT；GT 仅用于形成掩码之后的指标与误差分析。",
        "- R1 面积未放大、缩小或按数据集调整。",
        "- GBSP 前景位置只由 `absolute_raw` 的稳定降序排名决定。",
        "- Patch 前景数量逐图严格相等。",
        "- 当前结果仅为固定 pilot200 门控，不是6473张最终结论。",
        "",
        "## 数据集宏平均",
        "",
        header,
        divider,
        *body,
        "",
        "## 门控结论",
        "",
        f"- 状态：**{decision['status']}**。",
        f"- 原因：{decision['reason']}。",
        "",
        "## 面积与误差迁移",
        "",
        f"- Patch count 最大绝对误差：{max(abs(int(row['patch_count_error'])) for row in diagnostics)}。",
        f"- 与 R1 掩码平均 IoU：{_mean(row['r1_mask_iou'] for row in diagnostics):.6f}。",
        f"- FP Removal：{_mean(row['FPRemoval'] for row in diagnostics):.6f}。",
        f"- TP Retention：{_mean(row['TPRetention'] for row in diagnostics):.6f}。",
        f"- 预测面积与 R1 面积相关性：{metadata['area_correlations']['gbsp_vs_r1']:.6f}。",
        f"- 预测面积与 GT 面积相关性（仅诊断）：{metadata['area_correlations']['gbsp_vs_gt']:.6f}。",
        "",
        "## 完整性",
        "",
        f"- 有效/失败：{metadata['num_valid']}/{metadata['num_failed']}。",
        f"- 样本分布：`{metadata['dataset_counts']}`。",
        f"- 耗时：{metadata['wall_seconds']:.3f} 秒。",
    ]
    (output_dir / "GBSP_CALIBRATION_STAGE_A_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate(args) -> None:
    if args.methods != ["r1_area_transfer"]:
        raise RuntimeError(
            "stage A only permits --methods r1_area_transfer; later methods remain gated"
        )
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("--workers and --torch_threads must both be positive")
    cfg = load_config(_resolve(args.config))
    gbsp_root = _resolve(args.gbsp_root or cfg.GBSP_CALIBRATION_GBSP_ROOT)
    r1_root = _resolve(args.r1_root or cfg.GBSP_CALIBRATION_R1_ROOT)
    sample_list = _resolve(args.sample_list or cfg.GBSP_CALIBRATION_PILOT_LIST)
    pilot_source = _resolve(cfg.GBSP_CALIBRATION_PILOT_SOURCE_MANIFEST)
    output_dir = _resolve(args.out_dir)
    if output_dir == MAIN_ROOT or MAIN_ROOT in output_dir.parents:
        raise ValueError("output directory must stay outside the code repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = _materialize_pilot_list(sample_list, pilot_source)
    if args.max_samples > 0:
        keys = keys[: args.max_samples]
    _, gbsp_map = _manifest(_gbsp_manifest(gbsp_root))
    _, r1_map = _manifest(r1_root / "manifest_test.jsonl")
    tasks = []
    for dataset, stem in keys:
        key = (dataset, stem)
        if key not in gbsp_map or key not in r1_map:
            raise KeyError(f"GBSP/R1 manifest missing pilot identity: {key}")
        gbsp_row, r1_row = gbsp_map[key], r1_map[key]
        for path in (gbsp_row["cache_path"], r1_row["cache_path"], r1_row["gt_path"]):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        tasks.append(
            {
                "dataset": dataset,
                "stem": stem,
                "gbsp_path": gbsp_row["cache_path"],
                "r1_path": r1_row["cache_path"],
                "gt_path": r1_row["gt_path"],
                "r1_threshold": float(cfg.GBSP_CALIBRATION_R1_THRESHOLD),
                "minmax_threshold": float(cfg.GBSP_CALIBRATION_MINMAX_THRESHOLD),
            }
        )

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
                print(f"GBSP calibration stage A: {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    _write_json(output_dir / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("stage A produced no valid samples")
    per_image = [row for result in valid for row in result["rows"]]
    diagnostics = [result["diagnostic"] for result in valid]
    summary = _aggregate(per_image)
    complete = len(valid) == len(tasks) == PILOT_TOTAL and not failures
    decision = _decision(summary, complete)
    area_correlations = {
        "gbsp_vs_r1": _correlation(
            [row["gbsp_patch_area"] for row in diagnostics],
            [row["r1_patch_area"] for row in diagnostics],
        ),
        "gbsp_vs_gt": _correlation(
            [row["gbsp_patch_area"] for row in diagnostics],
            [row["gt_area"] for row in diagnostics],
        ),
        "r1_vs_gt": _correlation(
            [row["r1_patch_area"] for row in diagnostics],
            [row["gt_area"] for row in diagnostics],
        ),
    }
    metadata = {
        "schema": "gbsp_calibration_stage_a_v1",
        "config": str(_resolve(args.config)),
        "gbsp_root": str(gbsp_root),
        "r1_root": str(r1_root),
        "sample_list": str(sample_list),
        "output_dir": str(output_dir),
        "num_requested": len(tasks),
        "num_valid": len(valid),
        "num_failed": len(failures),
        "dataset_counts": dict(Counter(task["dataset"] for task in tasks)),
        "wall_seconds": time.perf_counter() - started,
        "calibration_gt_used": False,
        "evaluation_gt_used": True,
        "training_used": False,
        "feature_extraction_used": False,
        "r1_area_scale": 1.0,
        "patch_count_max_abs_error": max(
            abs(int(row["patch_count_error"])) for row in diagnostics
        ),
        "area_correlations": area_correlations,
        "decision": decision,
    }

    _write_csv(output_dir / "per_image_metrics.csv", per_image, list(per_image[0]))
    _write_csv(output_dir / "area_transfer_metrics.csv", diagnostics, list(diagnostics[0]))
    _write_csv(output_dir / "error_transition_metrics.csv", diagnostics, list(diagnostics[0]))
    _write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    threshold_rows = []
    for dataset in (*DATASETS, "ALL"):
        subset = diagnostics if dataset == "ALL" else [
            row for row in diagnostics if row["dataset"] == dataset
        ]
        if subset:
            cutoffs = [
                float(row["area_transfer_cutoff_raw"])
                for row in subset
                if math.isfinite(float(row["area_transfer_cutoff_raw"]))
            ]
            threshold_rows.append(
                {
                    "dataset": dataset,
                    "num_samples": len(subset),
                    "mean_cutoff": _mean(cutoffs),
                    "std_cutoff": float(np.std(cutoffs)) if cutoffs else float("nan"),
                    "mean_area": _mean(row["gbsp_patch_area"] for row in subset),
                    "std_area": float(np.std([row["gbsp_patch_area"] for row in subset])),
                }
            )
    _write_csv(output_dir / "threshold_statistics.csv", threshold_rows, list(threshold_rows[0]))
    _write_csv(
        output_dir / "calibration_runtime.csv",
        [
            {
                "dataset": row["dataset"],
                "stem": row["stem"],
                "runtime_seconds": row["runtime_seconds"],
                "worker_peak_rss_mb": row["worker_peak_rss_mb"],
            }
            for row in diagnostics
        ],
        ("dataset", "stem", "runtime_seconds", "worker_peak_rss_mb"),
    )
    _write_json(output_dir / "per_dataset_metrics.json", summary)
    _write_json(
        output_dir / "fallback_summary.json",
        {"fallback_count": 0, "fallback_ratio": 0.0, "fallbacks": []},
    )
    _write_json(output_dir / "metadata.json", metadata)
    _report(output_dir, summary, diagnostics, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and failures:
        raise RuntimeError(f"stage A recorded {len(failures)} failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gbsp_root", "--gbsp-root", dest="gbsp_root")
    parser.add_argument("--r1_root", "--r1-root", dest="r1_root")
    parser.add_argument("--crossfit_root", "--crossfit-root", dest="crossfit_root")
    parser.add_argument("--sample_list", "--sample-list", dest="sample_list")
    parser.add_argument("--methods", nargs="+", default=["r1_area_transfer"])
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", "--torch-threads", dest="torch_threads", type=int, default=1)
    parser.add_argument("--strict_failures", "--strict-failures", dest="strict_failures", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
