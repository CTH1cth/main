#!/usr/bin/env python3
"""Stage 4: parameter-free rank fusion, guarded by Stage-1 evidence."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    load_torch,
    plateau_rows,
    rank_metrics,
    read_csv,
    resize_score_to_native,
    right_ecdf,
    score_from_payload,
    score_path,
    write_csv,
    write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn8_root", required=True)
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--local_root", required=True)
    parser.add_argument("--methods", nargs="+", choices=("geo", "avg"), default=("geo", "avg"))
    parser.add_argument("--include_local_geo", action="store_true")
    parser.add_argument("--complementarity_json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _cod(context: FastCODContext, probability: torch.Tensor, threshold: float) -> dict:
    value = context.evaluate_many([("candidate", "hard", probability)], threshold)[("candidate", "hard")]
    precision, recall = float(value["Precision"]), float(value["Recall"])
    return {
        "S_m": float(value["S_m"]), "F_beta_w": float(value["F_beta_w"]),
        "F_beta_mean": float(value["F_beta_mean"]), "E_mean": float(value["E_mean"]),
        "MAE": float(value["MAE"]), "Precision": precision, "Recall": recall,
        "Area": float(value["Area"]), "IoU": float(value["IoU"]),
        "Dice": 2.0 * precision * recall / (precision + recall + 1e-12),
    }


def _local_gate(output: Path) -> tuple[bool, dict]:
    path = output.parent / "local_reconstruction_eval" / "local_reconstruction_continuous.csv"
    if not path.is_file():
        return False, {"status": "missing_local_evaluation", "path": str(path)}
    rows = [row for row in read_csv(path) if row.get("scope") == "dataset_macro"]
    index = {row["method"]: row for row in rows}
    if "k16_r4" not in index or "global_gbsp" not in index:
        return False, {"status": "missing_required_methods", "path": str(path)}
    local, global_gbsp = index["k16_r4"], index["global_gbsp"]
    allowed = float(local["AP"]) >= float(global_gbsp["AP"]) and float(local["AUROC"]) >= float(global_gbsp["AUROC"])
    return allowed, {
        "status": "pass" if allowed else "fail",
        "local_AP": float(local["AP"]), "global_AP": float(global_gbsp["AP"]),
        "local_AUROC": float(local["AUROC"]), "global_AUROC": float(global_gbsp["AUROC"]),
    }


def _curve_summary(accumulator: dict) -> list[dict]:
    rows = []
    for (dataset, method, threshold), state in sorted(accumulator.items()):
        rows.append({
            "scope": "dataset", "dataset": dataset, "method": method,
            "threshold": threshold, "num_samples": state["count"],
            **{metric: state[metric] / state["count"] for metric in COD_METRICS},
        })
    for method in sorted({row["method"] for row in rows}):
        for threshold in sorted({row["threshold"] for row in rows if row["method"] == method}):
            selected = [row for row in rows if row["method"] == method and row["threshold"] == threshold]
            rows.append({
                "scope": "dataset_macro", "dataset": "ALL", "method": method,
                "threshold": threshold, "num_samples": sum(row["num_samples"] for row in selected),
                **{metric: float(np.mean([row[metric] for row in selected])) for metric in COD_METRICS},
            })
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.out_dir).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("output must stay outside the main code tree")
    output.mkdir(parents=True, exist_ok=True)
    complementarity_path = Path(args.complementarity_json).resolve() if args.complementarity_json else output.parent / "complementarity" / "numerical_validity.json"
    if not complementarity_path.is_file():
        raise FileNotFoundError(f"Stage-1 gate missing: {complementarity_path}")
    complementarity = json.loads(complementarity_path.read_text(encoding="utf-8"))
    if not bool(complementarity.get("allow_fusion", False)):
        write_csv(output / "fusion_continuous.csv", [])
        write_csv(output / "fusion_binary.csv", [])
        write_json(output / "numerical_validity.json", {
            "stage": 4, "status": "SKIP", "reason": "Stage 1 did not support complementarity",
            "complementarity_case": complementarity.get("complementarity_case"),
        })
        print("Stage 4 SKIP: Stage 1 did not support complementarity")
        return

    local_allowed, local_gate = _local_gate(output)
    include_local = bool(args.include_local_geo and local_allowed)
    knn_rows = load_manifest(args.knn8_root, score=True, split=args.split, max_samples=args.max_samples)
    gbsp_index = index_manifest(load_manifest(args.gbsp_root, score=True, split=args.split))
    local_index = index_manifest(load_manifest(args.local_root, split=args.split)) if include_local else {}
    continuous_images, binary_images = [], []
    thresholds = tuple(float(round(value, 2)) for value in np.arange(0.30, 0.701, 0.01))
    curve_accumulator = defaultdict(lambda: {"count": 0, **{metric: 0.0 for metric in COD_METRICS}})
    methods = list(dict.fromkeys(args.methods))
    for index, row in enumerate(knn_rows, 1):
        dataset, stem = row["dataset"], row["stem"]
        key = (dataset, stem)
        knn_payload = load_torch(score_path(row))
        gbsp_row = gbsp_index[key]
        gbsp_payload = knn_payload if score_path(row).resolve() == score_path(gbsp_row).resolve() else load_torch(score_path(gbsp_row))
        rank_knn = right_ecdf(score_from_payload(knn_payload, "knn8").numpy())
        rank_gbsp = right_ecdf(score_from_payload(gbsp_payload, "gbsp").numpy())
        scores = {"knn8": rank_knn, "global_gbsp": rank_gbsp}
        if "geo" in methods:
            scores["geo"] = np.sqrt(rank_knn * rank_gbsp)
        if "avg" in methods:
            scores["avg"] = 0.5 * (rank_knn + rank_gbsp)
        if include_local:
            local_payload = load_torch(score_path(local_index[key]))
            rank_local = right_ecdf(score_from_payload(local_payload, "k16_r4").numpy())
            scores["local_geo"] = np.sqrt(rank_knn * rank_local)
        gt = load_native_gt(row["gt_path"])
        shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt)
        probabilities = {method: resize_score_to_native(value.reshape(1, 37, 37), shape) for method, value in scores.items()}
        for method, probability in probabilities.items():
            continuous_images.append({
                "dataset": dataset, "stem": stem, "method": method,
                **rank_metrics(probability, gt),
            })
            for protocol, threshold in (("fixed_0.50", 0.5), ("otsu", adaptive_threshold(scores[method], "otsu")[0])):
                binary_images.append({
                    "dataset": dataset, "stem": stem, "method": method,
                    "protocol": protocol, "threshold": threshold,
                    **_cod(context, probability, threshold),
                })
            for threshold in thresholds:
                metric = _cod(context, probability, threshold)
                state = curve_accumulator[(dataset, method, threshold)]
                state["count"] += 1
                for field in COD_METRICS:
                    state[field] += metric[field]
        if index % 50 == 0 or index == len(knn_rows):
            print(f"[{index}/{len(knn_rows)}] rank fusion", flush=True)

    continuous = aggregate_per_dataset(continuous_images, group_fields=("method",), metric_fields=("AP", "AUROC"))
    binary = aggregate_per_dataset(binary_images, group_fields=("method", "protocol"), metric_fields=(*COD_METRICS, "threshold"))
    curve = _curve_summary(curve_accumulator)
    plateaus = plateau_rows(curve)
    for row in plateaus:
        row["protocol"] = "threshold_plateau_0.30_0.70"
    write_csv(output / "fusion_continuous.csv", continuous)
    write_csv(output / "fusion_binary.csv", [*binary, *curve, *plateaus])
    formal = {row["method"]: row for row in continuous if row["scope"] == "dataset_macro"}
    candidates = [method for method in ("geo", "avg", "local_geo") if method in formal]
    best = max(candidates, key=lambda method: float(formal[method]["AP"])) if candidates else None
    retained = bool(best and (
        float(formal[best]["AP"]) >= max(float(formal["knn8"]["AP"]), float(formal["global_gbsp"]["AP"])) + 0.001
        or float(formal[best]["AUROC"]) >= max(float(formal["knn8"]["AUROC"]), float(formal["global_gbsp"]["AUROC"])) + 0.001
    ))
    decision = {
        "stage": 4, "status": "complete", "fusion_uses_gt": False,
        "fusion_has_learnable_parameters": False, "alpha_search_performed": False,
        "complementarity_case": complementarity.get("complementarity_case"),
        "local_geo_requested": bool(args.include_local_geo), "local_geo_gate": local_gate,
        "local_geo_run": include_local, "best_fusion": best, "fusion_retained": retained,
        "selected_route": "Route_B" if retained else "return_to_Stage3_selection",
    }
    write_json(output / "numerical_validity.json", decision)
    write_json(output / "selected_next_direction.json", decision)
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
