#!/usr/bin/env python3
"""Formal aggregation for the matched-size oracle-clean GBSP/KNN8 control."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gbsp_knn_lsr_common import (  # noqa: E402
    DATASETS,
    EXPECTED_COUNTS,
    load_manifest,
    load_torch,
    normalize_dataset,
    write_csv,
    write_json,
)


VERSION = "matched_size_oracle_clean_analysis_v1"
METHODS = ("knn8", "gbsp")
METRICS = ("AP", "AUROC")
EXPECTED_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260813)
    return parser.parse_args()


def _finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _bootstrap_dataset_macro(
    rows: list[dict], *, field: str, repetitions: int, seed: int
) -> dict[str, float]:
    groups = []
    for dataset in DATASETS:
        values = np.asarray(
            [float(row[field]) for row in rows if row["dataset"] == dataset],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            groups.append(values)
    if not groups:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    estimate = float(np.mean([values.mean() for values in groups]))
    if repetitions <= 0:
        return {"estimate": estimate, "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        draws[index] = np.mean([
            values[rng.integers(0, values.size, values.size)].mean()
            for values in groups
        ])
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _aggregate(rows: list[dict], *, seed_averaged: bool) -> list[dict]:
    output = []
    seeds = ("mean",) if seed_averaged else EXPECTED_SEEDS
    for seed in seeds:
        selected = rows if seed_averaged else [row for row in rows if int(row["seed"]) == seed]
        for dataset in DATASETS:
            subset = [row for row in selected if row["dataset"] == dataset]
            if not subset:
                continue
            for method in METHODS:
                output.append({
                    "scope": "dataset", "dataset": dataset, "seed": seed,
                    "method": method, "image_count": len(subset),
                    **{
                        metric: _finite_mean(row[f"{method}_{metric}"] for row in subset)
                        for metric in METRICS
                    },
                })
        for method in METHODS:
            dataset_rows = [
                row for row in output
                if row["scope"] == "dataset" and row["seed"] == seed and row["method"] == method
            ]
            if dataset_rows:
                output.append({
                    "scope": "dataset_macro", "dataset": "ALL", "seed": seed,
                    "method": method,
                    "image_count": sum(int(row["image_count"]) for row in dataset_rows),
                    **{metric: _finite_mean(row[metric] for row in dataset_rows) for metric in METRICS},
                })
                output.append({
                    "scope": "image_macro", "dataset": "ALL", "seed": seed,
                    "method": method, "image_count": len(selected),
                    **{
                        metric: _finite_mean(row[f"{method}_{metric}"] for row in selected)
                        for metric in METRICS
                    },
                })
    return output


def _seed_average(per_seed_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_seed_rows:
        groups[(row["dataset"], row["stem"])].append(row)
    output = []
    for (dataset, stem), rows in sorted(groups.items()):
        if tuple(sorted(int(row["seed"]) for row in rows)) != EXPECTED_SEEDS:
            continue
        output.append({
            "dataset": dataset, "stem": stem, "seed": "mean",
            **{
                f"{method}_{metric}": _finite_mean(row[f"{method}_{metric}"] for row in rows)
                for method in METHODS for metric in METRICS
            },
        })
    return output


def _report(
    out_dir: Path, validity: dict, summary: list[dict], bootstrap: list[dict]
) -> None:
    macro = {
        (row["scope"], str(row["seed"]), row["method"]): row
        for row in summary if row["dataset"] == "ALL"
    }
    lines = [
        "# Matched-size Oracle-clean 全量对照", "",
        f"- 有效图像：{validity['valid_images']}/{validity['requested']}",
        f"- 结构性排除：{validity['source_excluded_images']}",
        f"- 无效图像：{validity['invalid_images']}",
        f"- 是否完整6473张无排除：{'是' if validity['formal_full6473'] else '否'}",
        f"- 是否覆盖6473张并声明排除：{'是' if validity['full6473_with_declared_exclusions'] else '否'}", "",
        "## 三种子逐图平均", "",
        "| 聚合口径 | 方法 | AP | AUROC |", "|---|---|---:|---:|",
    ]
    for scope, label in (("dataset_macro", "四数据集macro"), ("image_macro", "6473张逐图平均")):
        for method in METHODS:
            row = macro.get((scope, "mean", method))
            if row:
                lines.append(f"| {label} | {method} | {row['AP']:.6f} | {row['AUROC']:.6f} |")
    lines.extend(["", "## GBSP-r8 − KNN8 配对差值（四数据集分层bootstrap）", "",
                  "| 指标 | 差值 | 95% CI |", "|---|---:|---:|"])
    for row in bootstrap:
        lines.append(
            f"| {row['metric']} | {row['estimate']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    lines.extend(["", "> GT仅用于oracle诊断干预，未用于生成可部署伪标签。", ""])
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache_root, split=args.split, max_samples=args.max_samples)
    source_failures_path = cache_root / "failures.json"
    source_failures = (
        json.loads(source_failures_path.read_text(encoding="utf-8"))
        if source_failures_path.is_file() else []
    )
    run_config_path = cache_root / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file() else {}
    )
    per_seed, invalid, failures = [], [], []
    for row in manifest:
        try:
            payload = load_torch(row["cache_path"])
            dataset, stem = normalize_dataset(payload["dataset"]), str(payload["stem"])
            if not payload.get("native_metrics_complete", False):
                raise RuntimeError("native metrics are incomplete")
            labels = torch.as_tensor(payload["patch_gt_label"], dtype=torch.bool).reshape(-1)
            natural_count = int(payload["natural"]["candidate_count"])
            items = payload.get("matched_size_clean", [])
            if tuple(sorted(int(item["seed"]) for item in items)) != EXPECTED_SEEDS:
                raise RuntimeError("matched-size seeds are incomplete")
            image_valid = True
            for item in items:
                if not item["valid"]:
                    invalid.append({"dataset": dataset, "stem": stem, "seed": int(item["seed"]), "reason": item.get("reason", "")})
                    image_valid = False
                    continue
                indices = torch.as_tensor(item.get("candidate_indices"), dtype=torch.long).reshape(-1)
                if indices.numel() != natural_count:
                    raise RuntimeError("matched dictionary size differs from natural dictionary")
                if bool(labels.index_select(0, indices).any()):
                    raise RuntimeError("matched dictionary contains a foreground-labelled patch")
                metrics = item.get("native_metrics")
                if not isinstance(metrics, dict):
                    raise RuntimeError("matched native metrics are missing")
                per_seed.append({
                    "dataset": dataset, "stem": stem, "seed": int(item["seed"]),
                    "candidate_count": int(indices.numel()),
                    **{
                        f"{method}_{metric}": float(metrics[method][metric])
                        for method in METHODS for metric in METRICS
                    },
                })
            if not image_valid:
                continue
        except Exception as error:
            failures.append({"dataset": row.get("dataset", ""), "stem": row.get("stem", ""), "error": repr(error)})

    averaged = _seed_average(per_seed)
    summary = _aggregate(per_seed, seed_averaged=False) + _aggregate(averaged, seed_averaged=True)
    paired = []
    for row in averaged:
        for metric in METRICS:
            row[f"gbsp_minus_knn8_{metric}"] = row[f"gbsp_{metric}"] - row[f"knn8_{metric}"]
    for index, metric in enumerate(METRICS):
        result = _bootstrap_dataset_macro(
            averaged, field=f"gbsp_minus_knn8_{metric}",
            repetitions=int(args.bootstrap_repetitions), seed=int(args.bootstrap_seed) + index,
        )
        paired.append({
            "metric": metric, "quantity": "gbsp_minus_knn8",
            "image_count": len(averaged), "repetitions": int(args.bootstrap_repetitions),
            **result,
        })

    counts = Counter(row["dataset"] for row in averaged)
    requested = int(run_config.get("requested_samples", len(manifest) + len(source_failures)))
    configured_counts = {
        normalize_dataset(key): int(value)
        for key, value in run_config.get("dataset_counts", {}).items()
    }
    requested_counts = configured_counts or dict(Counter(
        [normalize_dataset(row["dataset"]) for row in manifest]
        + [normalize_dataset(row.get("dataset", "")) for row in source_failures]
    ))
    formal = (
        requested == 6473 and dict(requested_counts) == EXPECTED_COUNTS
        and len(averaged) == 6473 and dict(counts) == EXPECTED_COUNTS
        and not source_failures and not invalid and not failures
    )
    covered_with_exclusions = (
        requested == 6473 and dict(requested_counts) == EXPECTED_COUNTS
        and len(averaged) + len(source_failures) == 6473
        and not invalid and not failures
    )
    validity = {
        "version": VERSION, "requested": requested, "valid_images": len(averaged),
        "source_excluded_images": len(source_failures),
        "invalid_images": len({(row["dataset"], row["stem"]) for row in invalid}),
        "failed_images": len(failures), "dataset_counts": dict(counts),
        "expected_counts": EXPECTED_COUNTS, "seeds": list(EXPECTED_SEEDS),
        "formal_full6473": formal,
        "full6473_with_declared_exclusions": covered_with_exclusions,
        "gt_use": "oracle_diagnostic_only",
    }
    write_csv(out_dir / "per_seed_image_metrics.csv", per_seed)
    write_csv(out_dir / "seed_averaged_image_metrics.csv", averaged)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "paired_bootstrap.csv", paired)
    write_csv(out_dir / "invalid_matched_dictionaries.csv", invalid)
    write_csv(out_dir / "excluded_source_samples.csv", source_failures)
    write_json(out_dir / "failures.json", failures)
    write_json(out_dir / "validity_summary.json", validity)
    _report(out_dir, validity, summary, paired)
    print(json.dumps({**validity, "output": str(out_dir)}, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} analysis inputs failed")


if __name__ == "__main__":
    main()
