#!/usr/bin/env python3
"""Stage 3 formal evaluation for the three predeclared LSR variants."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext
from tools.gbsp_knn_lsr_common import (
    COD_METRICS,
    adaptive_threshold,
    aggregate_per_dataset,
    index_manifest,
    load_manifest,
    load_native_gt,
    load_patch_area,
    load_torch,
    minmax,
    normalize_dataset,
    patch_labels,
    plateau_rows,
    rank_metrics,
    resize_score_to_native,
    score_from_payload,
    score_path,
    stratified_paired_bootstrap,
    write_csv,
    write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VARIANTS = ("k8_r2", "k16_r4", "k32_r8")
BASELINES = ("mean_prototype", "nn", "knn8", "global_gbsp")
ALL_METHODS = (*BASELINES, *LOCAL_VARIANTS)
SCORE_NAMES = {
    "mean_prototype": "rank0_global_reference", "nn": "nn_cos",
    "knn8": "knn8_cos", "global_gbsp": "gbsp_r8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_root", required=True)
    parser.add_argument("--knn8_root", required=True)
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--image_minmax", action="store_true")
    parser.add_argument("--fixed_thresholds", nargs="+", type=float, default=(0.50, 0.58))
    parser.add_argument("--threshold_start", type=float, default=0.30)
    parser.add_argument("--threshold_end", type=float, default=0.70)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--adaptive_thresholds", nargs="+", choices=("otsu", "multi_otsu_3"), default=("otsu", "multi_otsu_3"))
    parser.add_argument("--high_similarity_quantiles", nargs="+", type=float, default=(0.80, 0.70, 0.60))
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260807)
    parser.add_argument("--numerical_smoke_only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _cod_many(context: FastCODContext, probabilities: dict[str, object], threshold: float) -> dict[str, dict]:
    candidates = [(method, "hard", value) for method, value in probabilities.items()]
    evaluated = context.evaluate_many(candidates, float(threshold))
    output = {}
    for method in probabilities:
        value = evaluated[(method, "hard")]
        precision, recall = float(value["Precision"]), float(value["Recall"])
        output[method] = {
            "S_m": float(value["S_m"]), "F_beta_w": float(value["F_beta_w"]),
            "F_beta_mean": float(value["F_beta_mean"]), "E_mean": float(value["E_mean"]),
            "MAE": float(value["MAE"]), "Precision": precision, "Recall": recall,
            "Area": float(value["Area"]), "IoU": float(value["IoU"]),
            "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
        }
    return output


def _curve_summary(accumulator: dict) -> list[dict]:
    rows = []
    for (dataset, method, threshold), state in sorted(accumulator.items()):
        rows.append({
            "scope": "dataset", "dataset": dataset, "method": method,
            "threshold": threshold, "num_samples": state["count"],
            **{metric: state[metric] / state["count"] for metric in COD_METRICS},
        })
    for method in ALL_METHODS:
        for threshold in sorted({row["threshold"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["threshold"] == threshold]
            rows.append({
                "scope": "dataset_macro", "dataset": "ALL", "method": method,
                "threshold": threshold, "num_samples": sum(row["num_samples"] for row in selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in COD_METRICS},
            })
    return rows


def _summarize_h(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}) + ["ALL"]:
        for subset in sorted({row["subset"] for row in rows}):
            for method in ("knn8", "global_gbsp", "k16_r4"):
                selected = [
                    row for row in rows if row["subset"] == subset and row["method"] == method
                    and (dataset == "ALL" or row["dataset"] == dataset)
                    and np.isfinite(row["AP"])
                ]
                if selected:
                    output.append({
                        "dataset": dataset, "subset": subset, "method": method,
                        "protocol": "per_image_macro_patch37_main_0.5",
                        "valid_images": len(selected),
                        "AP": float(np.mean([row["AP"] for row in selected])),
                        "AUROC": float(np.mean([row["AUROC"] for row in selected])),
                    })
    return output


def _formal(summary: list[dict], method: str) -> dict:
    return next(row for row in summary if row["scope"] == "dataset_macro" and row["method"] == method)


def _selection(continuous: list[dict], binary050: list[dict], plateaus: list[dict], complementarity_path: Path) -> dict:
    knn, local, global_gbsp = _formal(continuous, "knn8"), _formal(continuous, "k16_r4"), _formal(continuous, "global_gbsp")
    delta_ap = float(local["AP"] - knn["AP"])
    delta_auroc = float(local["AUROC"] - knn["AUROC"])
    primary = "AP" if delta_ap >= delta_auroc else "AUROC"
    local_ds = {row["dataset"]: row for row in continuous if row["scope"] == "dataset" and row["method"] == "k16_r4"}
    knn_ds = {row["dataset"]: row for row in continuous if row["scope"] == "dataset" and row["method"] == "knn8"}
    dataset_delta = {dataset: float(local_ds[dataset][primary] - knn_ds[dataset][primary]) for dataset in local_ds if dataset in knn_ds}
    direction_count = sum(value >= 0.0 for value in dataset_delta.values())
    no_major_cod_nc = all(dataset_delta.get(dataset, -1.0) >= -0.001 for dataset in ("COD10K", "NC4K"))
    strong = (
        (delta_ap >= 0.003 and delta_auroc >= -0.001)
        or (delta_auroc >= 0.003 and delta_ap >= -0.001)
    ) and direction_count >= 3 and no_major_cod_nc
    medium_continuous = max(delta_ap, delta_auroc) >= 0.001 and min(delta_ap, delta_auroc) >= -0.001 and direction_count >= 3 and no_major_cod_nc
    binary = {row["method"]: row for row in binary050 if row["scope"] == "dataset_macro"}
    binary_better = "k16_r4" in binary and "knn8" in binary and float(binary["k16_r4"]["F_beta_w"]) > float(binary["knn8"]["F_beta_w"])
    plateau = {row["method"]: row for row in plateaus}
    plateau_wider = "k16_r4" in plateau and "knn8" in plateau and float(plateau["k16_r4"]["plateau_width"]) > float(plateau["knn8"]["plateau_width"])
    medium = medium_continuous and binary_better and plateau_wider
    complementarity = {}
    if complementarity_path.is_file():
        complementarity = json.loads(complementarity_path.read_text(encoding="utf-8"))
    if strong:
        route, recommendation = "Route_A", "LSR-K16-R4进入离线伪标签生成与后续学生训练候选"
    elif max(delta_ap, delta_auroc) > 0 and not binary_better:
        route, recommendation = "Route_D", "连续排序改善但阈值仍失败；下一阶段研究局部残差标定"
    elif bool(complementarity.get("allow_fusion", False)):
        route, recommendation = "Route_B_pending_fusion", "互补性允许无参数rank fusion；先完成Stage 4"
    else:
        route, recommendation = "Route_C", "停止增加PCA机制；优先保留KNN8伪标签生成路线"
    if not strong and not medium and max(delta_ap, delta_auroc) <= 0 and float(knn["AP"]) >= float(global_gbsp["AP"]):
        next_stage = "Local affine reconstruction only as a future recommendation; not implemented in this round"
    else:
        next_stage = "No local-affine trigger"
    return {
        "selected_route": route, "recommendation": recommendation,
        "LSR_main_delta_AP_vs_KNN8": delta_ap,
        "LSR_main_delta_AUROC_vs_KNN8": delta_auroc,
        "primary_direction_metric": primary, "per_dataset_delta": dataset_delta,
        "non_degraded_dataset_count": direction_count,
        "COD10K_NC4K_no_major_drop": no_major_cod_nc,
        "strong_success": strong, "medium_success_without_downstream": medium,
        "binary_050_Fw_improved": binary_better, "threshold_plateau_wider": plateau_wider,
        "downstream_1x1_run": False,
        "NEXT_STAGE_RECOMMENDATION": next_stage,
    }


def _report(selection: dict, continuous: list[dict], h_rows: list[dict]) -> str:
    formal = [row for row in continuous if row["scope"] == "dataset_macro"]
    lines = [
        "# GBSP / KNN8 / Query-Adaptive LSR Report", "",
        "本报告严格区分连续排序、阈值化质量和后续伪标签学生训练。本轮没有训练学生模型。", "",
        "## 连续指标（四数据集等权宏平均）", "",
        "| Method | AP | AUROC |", "|---|---:|---:|",
    ]
    for row in formal:
        lines.append(f"| {row['method']} | {float(row['AP']):.6f} | {float(row['AUROC']):.6f} |")
    lines += ["", "## H20/H30/H40", "", "| Subset | Method | AP | AUROC |", "|---|---|---:|---:|"]
    for row in h_rows:
        if row["dataset"] == "ALL":
            lines.append(f"| {row['subset']} | {row['method']} | {float(row['AP']):.6f} | {float(row['AUROC']):.6f} |")
    lines += ["", "## 路线裁决", "", f"- 选择：**{selection['selected_route']}**"]
    if "LSR_main_delta_AP_vs_KNN8" in selection:
        lines.append(
            f"- LSR-K16-R4 相对 KNN8：AP {selection['LSR_main_delta_AP_vs_KNN8']:+.6f}，"
            f"AUROC {selection['LSR_main_delta_AUROC_vs_KNN8']:+.6f}。"
        )
    lines += [
        f"- 建议：{selection['recommendation']}。", "",
        "该裁决用于选择更好的离线伪标签生成机制；通过后才进入保持训练协议一致的学生模型验证。", "",
    ]
    return "\n".join(lines)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _process_image(task: dict) -> dict:
    local_row, knn_row, gbsp_row = task["local_row"], task["knn_row"], task["gbsp_row"]
    dataset, stem = normalize_dataset(local_row["dataset"]), local_row["stem"]
    try:
        local_payload = load_torch(score_path(local_row))
        similarity_payload = load_torch(score_path(knn_row))
        gbsp_payload = similarity_payload if score_path(knn_row).resolve() == score_path(gbsp_row).resolve() else load_torch(score_path(gbsp_row))
        scores = {
            "mean_prototype": score_from_payload(similarity_payload, SCORE_NAMES["mean_prototype"]),
            "nn": score_from_payload(similarity_payload, SCORE_NAMES["nn"]),
            "knn8": score_from_payload(similarity_payload, SCORE_NAMES["knn8"]),
            "global_gbsp": score_from_payload(gbsp_payload, SCORE_NAMES["global_gbsp"]),
            **{variant: score_from_payload(local_payload, variant) for variant in LOCAL_VARIANTS},
        }
        if int(local_payload.get("query_count", -1)) != 1369 or not bool(local_payload.get("all_queries_scored", False)):
            raise RuntimeError("LSR cache does not score all 1369 queries")
        if any(int(item.get("self_match_violation_count", -1)) != 0 for item in local_payload["variants"].values()):
            raise RuntimeError("LSR self-match violation")
        gt = load_native_gt(local_row["gt_path"])
        shape = tuple(gt.shape[-2:])
        context = None if task["numerical_smoke_only"] else FastCODContext(gt)
        probabilities = {method: resize_score_to_native(minmax(score).reshape(1, 37, 37), shape) for method, score in scores.items()}
        continuous, binary, high, curve = [], [], [], []
        for method, score in scores.items():
            continuous.append({
                "dataset": dataset, "stem": stem, "method": method,
                **rank_metrics(resize_score_to_native(score.reshape(1, 37, 37), shape), gt),
            })
        area = load_patch_area(local_row["gt_path"])
        labels, valid = patch_labels(area, "main_0.5")
        similarity = 1.0 - scores["knn8"].numpy()
        for quantile, name in zip(task["high_similarity_quantiles"], ("H20", "H30", "H40")):
            mask = valid & (similarity >= np.quantile(similarity, quantile))
            for method in ("knn8", "global_gbsp", "k16_r4"):
                high.append({
                    "dataset": dataset, "stem": stem, "subset": name, "method": method,
                    **rank_metrics(scores[method].numpy()[mask], labels[mask]),
                })
        if not task["numerical_smoke_only"]:
            for threshold in task["fixed_thresholds"]:
                for method, metric in _cod_many(context, probabilities, threshold).items():
                    binary.append({
                        "dataset": dataset, "stem": stem, "method": method,
                        "protocol": f"fixed_{threshold:.2f}", "threshold": threshold, **metric,
                    })
            for adaptive in task["adaptive_thresholds"]:
                for method, probability in probabilities.items():
                    threshold, fallback = adaptive_threshold(minmax(scores[method]), adaptive)
                    binary.append({
                        "dataset": dataset, "stem": stem, "method": method,
                        "protocol": adaptive, "threshold": threshold, "fallback": int(fallback),
                        **_cod_many(context, {method: probability}, threshold)[method],
                    })
            for threshold in task["thresholds"]:
                for method, metric in _cod_many(context, probabilities, threshold).items():
                    curve.append({"dataset": dataset, "method": method, "threshold": threshold, **metric})
        return {"continuous": continuous, "binary": binary, "high": high, "curve": curve}
    except Exception as error:
        return {"dataset": dataset, "stem": stem, "error": repr(error)}


def main() -> None:
    args = parse_args()
    if not args.image_minmax:
        raise ValueError("formal evaluation requires --image_minmax")
    if tuple(round(value, 2) for value in args.fixed_thresholds) != (0.50, 0.58):
        raise ValueError("fixed thresholds are frozen to 0.50 and 0.58")
    if tuple(round(value, 2) for value in args.high_similarity_quantiles) != (0.80, 0.70, 0.60):
        raise ValueError("H20/H30/H40 quantiles are frozen to 0.80/0.70/0.60")
    output = Path(args.out_dir).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("output must stay outside the main code tree")
    output.mkdir(parents=True, exist_ok=True)
    local_rows = load_manifest(args.score_root, split=args.split, max_samples=args.max_samples)
    knn_index = index_manifest(load_manifest(args.knn8_root, score=True, split=args.split))
    gbsp_index = index_manifest(load_manifest(args.gbsp_root, score=True, split=args.split))
    thresholds = tuple(float(round(value, 10)) for value in np.arange(
        args.threshold_start, args.threshold_end + 0.5 * args.threshold_step, args.threshold_step
    ))
    continuous_images, binary_images, high_images = [], [], []
    curve_accumulator = defaultdict(lambda: {"count": 0, **{metric: 0.0 for metric in COD_METRICS}})
    failures = []
    tasks = []
    for local_row in local_rows:
        key = (normalize_dataset(local_row["dataset"]), local_row["stem"])
        tasks.append({
            "local_row": local_row, "knn_row": knn_index[key], "gbsp_row": gbsp_index[key],
            "numerical_smoke_only": bool(args.numerical_smoke_only),
            "fixed_thresholds": tuple(args.fixed_thresholds),
            "adaptive_thresholds": tuple(args.adaptive_thresholds),
            "high_similarity_quantiles": tuple(args.high_similarity_quantiles),
            "thresholds": thresholds,
        })
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
        results = executor.map(_process_image, tasks, chunksize=1)
    else:
        executor = None
        results = map(_process_image, tasks)
    for index, result in enumerate(results, 1):
        if "error" in result:
            failures.append(result)
        else:
            continuous_images.extend(result["continuous"])
            binary_images.extend(result["binary"])
            high_images.extend(result["high"])
            for row in result["curve"]:
                state = curve_accumulator[(row["dataset"], row["method"], row["threshold"])]
                state["count"] += 1
                for field in COD_METRICS:
                    state[field] += row[field]
        if index % 25 == 0 or index == len(local_rows):
            print(f"[{index}/{len(local_rows)}] local evaluation failed={len(failures)}", flush=True)
    if executor is not None:
        executor.shutdown()
    if failures:
        write_json(output / "numerical_validity.json", {"status": "failed", "failures": failures})
        raise RuntimeError(f"local evaluation failed for {len(failures)} images")

    continuous = aggregate_per_dataset(continuous_images, group_fields=("method",), metric_fields=("AP", "AUROC"))
    per_dataset = [row for row in continuous if row["scope"] == "dataset"]
    binary = aggregate_per_dataset(binary_images, group_fields=("method", "protocol"), metric_fields=(*COD_METRICS, "threshold")) if binary_images else []
    binary050 = [row for row in binary if row["protocol"] == "fixed_0.50"]
    binary058 = [row for row in binary if row["protocol"] == "fixed_0.58"]
    adaptive = [row for row in binary if row["protocol"] in args.adaptive_thresholds]
    curve = _curve_summary(curve_accumulator)
    plateaus = plateau_rows(curve)
    high = _summarize_h(high_images)
    write_csv(output / "local_reconstruction_continuous.csv", continuous)
    write_csv(output / "local_reconstruction_per_dataset.csv", per_dataset)
    write_csv(output / "local_reconstruction_binary_050.csv", binary050)
    write_csv(output / "local_reconstruction_binary_058.csv", binary058)
    write_csv(output / "local_reconstruction_adaptive.csv", adaptive)
    write_csv(output / "local_reconstruction_threshold_curve.csv", curve)
    write_csv(output / "local_reconstruction_threshold_plateau.csv", plateaus)
    write_csv(output / "local_reconstruction_high_similarity.csv", high)

    bootstrap = []
    for candidate in LOCAL_VARIANTS:
        for baseline in ("knn8", "global_gbsp"):
            bootstrap.extend(stratified_paired_bootstrap(
                continuous_images, candidate=candidate, baseline=baseline,
                method_field="method", metrics=("AP", "AUROC"),
                repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed,
            ))
    if binary_images:
        for baseline in ("knn8", "global_gbsp"):
            bootstrap.extend(stratified_paired_bootstrap(
                binary_images, candidate="k16_r4", baseline=baseline,
                method_field="method", metrics=COD_METRICS,
                repetitions=args.bootstrap_repetitions, seed=args.bootstrap_seed + 1,
                protocol_fields=("protocol",),
            ))
    write_csv(output / "local_reconstruction_bootstrap.csv", bootstrap)
    complementarity = output.parent / "complementarity" / "numerical_validity.json"
    selection = (
        _selection(continuous, binary050, plateaus, complementarity)
        if not args.numerical_smoke_only else
        {"selected_route": "SMOKE_ONLY_NO_ROUTE_DECISION", "recommendation": "仅验证数值实现，不运行Hard指标或路线选择"}
    )
    if len(local_rows) != 6473:
        selection.update({
            "selected_route": "SMOKE_ONLY_NO_ROUTE_DECISION",
            "recommendation": "仅验证实现；不得用该子集选择K/r、融合或论文路线",
            "strong_success": False,
            "medium_success_without_downstream": False,
        })
    write_json(output / "selected_next_direction.json", selection)
    validity = {
        "stage": 3, "images": len(local_rows), "is_full_complete": len(local_rows) == 6473,
        "query_count_per_image": 1369, "self_match_violation_count": 0,
        "variants": list(LOCAL_VARIANTS), "K_r_grid_search_performed": False,
        "gt_used_for_local_generation": False, "gt_used_only_for_evaluation": True,
        "thresholds": [] if args.numerical_smoke_only else list(thresholds),
        "numerical_smoke_only": bool(args.numerical_smoke_only), "failures": 0,
    }
    write_json(output / "numerical_validity.json", validity)
    (output / "GBSP_KNN_LOCAL_RECONSTRUCTION_REPORT.md").write_text(_report(selection, continuous, high), encoding="utf-8")
    print(json.dumps({**validity, **selection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
