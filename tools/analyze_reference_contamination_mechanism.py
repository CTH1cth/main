#!/usr/bin/env python3
"""Aggregate, bootstrap and visualize reference-contamination experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
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


VERSION = "reference_contamination_analysis_v1"
METHODS = ("knn8", "gbsp")
METRICS = ("AP", "AUROC")
PROTOCOLS = ("patch37_Q_noninj", "patch37_full_query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260808)
    parser.add_argument(
        "--query_storage", choices=("compressed_npz", "csv", "csv_gz", "none"),
        default="compressed_npz",
    )
    return parser.parse_args()


def _csv_stream(path: Path, fields: list[str], *, gzip_output: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = gzip.open(path, "wt", encoding="utf-8", newline="") if gzip_output else path.open(
        "w", encoding="utf-8", newline=""
    )
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _quantiles(values: np.ndarray, prefix: str = "") -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "median", "P25", "P75", "P90", "P95")}
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}P25": float(np.quantile(values, .25)),
        f"{prefix}P75": float(np.quantile(values, .75)),
        f"{prefix}P90": float(np.quantile(values, .90)),
        f"{prefix}P95": float(np.quantile(values, .95)),
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    value = spearmanr(x, y)
    return float(value.statistic), float(value.pvalue)


def _finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _stratified_bootstrap(
    records: list[dict],
    *,
    value_field: str,
    repetitions: int,
    seed: int,
) -> dict:
    arrays = []
    for dataset in DATASETS:
        values = np.asarray([
            float(row[value_field]) for row in records
            if row["dataset"] == dataset and math.isfinite(float(row[value_field]))
        ], dtype=np.float64)
        if values.size:
            arrays.append((dataset, values))
    if not arrays:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    estimate = float(np.mean([values.mean() for _, values in arrays]))
    if repetitions <= 0:
        return {"estimate": estimate, "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        draws[repetition] = np.mean([
            values[rng.integers(0, values.size, values.size)].mean() for _, values in arrays
        ])
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(draws, .025)),
        "ci95_high": float(np.quantile(draws, .975)),
    }


def _stratified_bootstrap_fields(
    records: list[dict],
    *,
    fields: tuple[str, ...],
    repetitions: int,
    seed: int,
    chunk_size: int = 256,
) -> dict[str, dict]:
    """Bootstrap several paired quantities with shared image resamples."""
    arrays = []
    for dataset in DATASETS:
        values = np.asarray([
            [float(row[field]) for field in fields] for row in records if row["dataset"] == dataset
        ], dtype=np.float64)
        if values.size:
            values = values[np.isfinite(values).all(axis=1)]
        if values.size:
            arrays.append((dataset, values))
    if not arrays:
        return {
            field: {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
            for field in fields
        }
    estimates = np.mean(np.stack([values.mean(axis=0) for _, values in arrays]), axis=0)
    if repetitions <= 0:
        return {
            field: {"estimate": float(estimates[index]), "ci95_low": float("nan"), "ci95_high": float("nan")}
            for index, field in enumerate(fields)
        }
    rng = np.random.default_rng(int(seed))
    draws = np.zeros((int(repetitions), len(fields)), dtype=np.float64)
    for _, values in arrays:
        for start in range(0, int(repetitions), int(chunk_size)):
            stop = min(start + int(chunk_size), int(repetitions))
            indices = rng.integers(0, values.shape[0], size=(stop - start, values.shape[0]))
            draws[start:stop] += values[indices].mean(axis=1) / len(arrays)
    return {
        field: {
            "estimate": float(estimates[index]),
            "ci95_low": float(np.quantile(draws[:, index], .025)),
            "ci95_high": float(np.quantile(draws[:, index], .975)),
        }
        for index, field in enumerate(fields)
    }


def _aggregate_seed_curve(
    image_rows: list[dict],
    selected_ids: set[tuple[str, str]],
    *,
    subset_protocol: str,
) -> list[dict]:
    output = []
    keys = sorted({
        (row["p"], row["seed"], row["method"], row["protocol"])
        for row in image_rows if (row["dataset"], row["stem"]) in selected_ids
    })
    for p, seed, method, protocol in keys:
        matching = [
            row for row in image_rows
            if (row["dataset"], row["stem"]) in selected_ids
            and (row["p"], row["seed"], row["method"], row["protocol"])
            == (p, seed, method, protocol)
        ]
        dataset_rows = []
        for dataset in DATASETS:
            subset = [row for row in matching if row["dataset"] == dataset]
            if not subset:
                continue
            aggregate = {
                "scope": "dataset", "dataset": dataset, "p": p, "seed": seed,
                "method": method, "protocol": protocol, "subset_protocol": subset_protocol,
                "valid_images": len(subset),
                **{metric: float(np.mean([row[metric] for row in subset])) for metric in METRICS},
            }
            dataset_rows.append(aggregate)
            output.append(aggregate)
        if dataset_rows:
            output.append({
                "scope": "dataset_macro", "dataset": "ALL", "p": p, "seed": seed,
                "method": method, "protocol": protocol, "subset_protocol": subset_protocol,
                "valid_images": sum(row["valid_images"] for row in dataset_rows),
                **{metric: float(np.mean([row[metric] for row in dataset_rows])) for metric in METRICS},
            })
            output.append({
                "scope": "image_macro", "dataset": "ALL", "p": p, "seed": seed,
                "method": method, "protocol": protocol, "subset_protocol": subset_protocol,
                "valid_images": len(matching),
                **{metric: float(np.mean([row[metric] for row in matching])) for metric in METRICS},
            })
    baselines = {
        (row["scope"], row["dataset"], row["seed"], row["method"], row["protocol"], row["subset_protocol"]): row
        for row in output if float(row["p"]) == 0.0
    }
    for row in output:
        baseline = baselines.get((
            row["scope"], row["dataset"], row["seed"], row["method"],
            row["protocol"], row["subset_protocol"],
        ))
        for metric in METRICS:
            row[f"delta_{metric}"] = float(row[metric] - baseline[metric]) if baseline else float("nan")
            row[f"degradation_{metric}"] = -row[f"delta_{metric}"]
    return output


def _curve_summary(curve: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({
        (row["scope"], row["dataset"], row["p"], row["method"], row["protocol"], row["subset_protocol"])
        for row in curve
    })
    for scope, dataset, p, method, protocol, subset_protocol in keys:
        subset = [
            row for row in curve
            if (row["scope"], row["dataset"], row["p"], row["method"], row["protocol"], row["subset_protocol"])
            == (scope, dataset, p, method, protocol, subset_protocol)
        ]
        item = {
            "scope": scope, "dataset": dataset, "p": p, "method": method,
            "protocol": protocol, "subset_protocol": subset_protocol, "seed_count": len(subset),
        }
        for metric in (*METRICS, "delta_AP", "delta_AUROC", "degradation_AP", "degradation_AUROC"):
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            mean = float(values.mean())
            half = 2.776 * float(values.std(ddof=1)) / math.sqrt(values.size) if values.size > 1 else 0.0
            item[metric] = mean
            item[f"{metric}_ci95_low"] = mean - half
            item[f"{metric}_ci95_high"] = mean + half
        rows.append(item)
    return rows


def _method_gap(curve: list[dict]) -> list[dict]:
    """Return the paired KNN8-minus-GBSP gap for every aggregate condition."""
    lookup = {
        (
            row["scope"], row["dataset"], row["p"], row["seed"],
            row["protocol"], row["subset_protocol"], row["method"],
        ): row
        for row in curve
    }
    output = []
    condition_keys = sorted({key[:-1] for key in lookup})
    for key in condition_keys:
        knn = lookup.get((*key, "knn8"))
        gbsp = lookup.get((*key, "gbsp"))
        if knn is None or gbsp is None:
            continue
        output.append({
            "scope": key[0], "dataset": key[1], "p": key[2], "seed": key[3],
            "protocol": key[4], "subset_protocol": key[5],
            "valid_images": min(int(knn["valid_images"]), int(gbsp["valid_images"])),
            "knn8_minus_gbsp_AP": float(knn["AP"] - gbsp["AP"]),
            "knn8_minus_gbsp_AUROC": float(knn["AUROC"] - gbsp["AUROC"]),
        })
    return output


def _image_slopes(image_rows: list[dict], selected_ids: set[tuple[str, str]], max_p: float) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in image_rows:
        identity = (row["dataset"], row["stem"])
        if identity in selected_ids and float(row["p"]) <= max_p + 1e-12:
            grouped[(row["dataset"], row["stem"], row["method"], row["protocol"], row["seed"])].append(row)
    seed_slopes = []
    for (dataset, stem, method, protocol, seed), subset in grouped.items():
        subset.sort(key=lambda row: row["p"])
        x = np.asarray([row["p"] for row in subset], dtype=np.float64)
        if np.unique(x).size < 2:
            continue
        seed_slopes.append({
            "dataset": dataset, "stem": stem, "method": method, "protocol": protocol,
            "seed": seed,
            **{f"slope_{metric}": float(np.polyfit(x, [row[metric] for row in subset], 1)[0]) for metric in METRICS},
        })
    image_group: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_slopes:
        image_group[(row["dataset"], row["stem"], row["method"], row["protocol"])].append(row)
    output = []
    for (dataset, stem, method, protocol), subset in image_group.items():
        output.append({
            "dataset": dataset, "stem": stem, "method": method, "protocol": protocol,
            "seed_count": len(subset),
            **{f"slope_{metric}": float(np.mean([row[f'slope_{metric}'] for row in subset])) for metric in METRICS},
        })
    return output


def _slope_bootstrap(slopes: list[dict], repetitions: int, seed: int) -> list[dict]:
    output = []
    for protocol in PROTOCOLS:
        lookup = {(row["dataset"], row["stem"], row["method"]): row for row in slopes if row["protocol"] == protocol}
        identities = sorted({(key[0], key[1]) for key in lookup if (*key[:2], "knn8") in lookup and (*key[:2], "gbsp") in lookup})
        for offset, metric in enumerate(METRICS):
            records = []
            for dataset, stem in identities:
                knn = lookup[(dataset, stem, "knn8")][f"slope_{metric}"]
                gbsp = lookup[(dataset, stem, "gbsp")][f"slope_{metric}"]
                records.append({
                    "dataset": dataset, "knn_slope": knn, "gbsp_slope": gbsp,
                    "gbsp_minus_knn_slope": gbsp - knn,
                })
            quantities = ("knn_slope", "gbsp_slope", "gbsp_minus_knn_slope")
            bootstrapped = _stratified_bootstrap_fields(
                records, fields=quantities, repetitions=repetitions,
                seed=seed + offset * 10 + len(output),
            )
            for quantity in quantities:
                output.append({
                    "protocol": protocol, "metric": metric, "quantity": quantity,
                    "image_count": len(records), "repetitions": repetitions, **bootstrapped[quantity],
                })
    return output


def _plot_a(out_dir: Path, global_rho: np.ndarray, local_rho: np.ndarray, groups: dict[str, np.ndarray]) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(global_rho, bins=30, density=True, alpha=.75, label="global contamination")
    axes[0].hist(local_rho, bins=30, density=True, alpha=.65, label="local top-8 contamination")
    axes[0].set_xlabel("contamination fraction"); axes[0].set_ylabel("density"); axes[0].legend()
    ordered = [groups.get(key, np.asarray([])) for key in ("0", "1", "2", ">=3")]
    axes[1].boxplot(ordered, tick_labels=("0", "1", "2", ">=3"), showfliers=False)
    axes[1].set_xlabel("wrong references in top-8"); axes[1].set_ylabel("KNN8 foreground anomaly")
    fig.tight_layout(); fig.savefig(figures / "A_local_amplification.png", dpi=180); plt.close(fig)


def _plot_b(out_dir: Path, bootstrap: list[dict]) -> None:
    rows = [row for row in bootstrap if row["quantity"] in ("knn_gain", "gbsp_gain")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    width = .34
    for index, quantity in enumerate(("knn_gain", "gbsp_gain")):
        selected = [next(row for row in rows if row["quantity"] == quantity and row["metric"] == metric) for metric in METRICS]
        y = [row["estimate"] for row in selected]
        lo = [y[i] - selected[i]["ci95_low"] for i in range(2)]
        hi = [selected[i]["ci95_high"] - y[i] for i in range(2)]
        ax.bar(x + (index - .5) * width, y, width, yerr=[lo, hi], capsize=4, label=quantity)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(x, METRICS)
    ax.set_ylabel("Oracle-clean gain"); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "figures" / "B_oracle_clean_gain.png", dpi=180); plt.close(fig)


def _plot_c(out_dir: Path, summary: list[dict]) -> None:
    selected = [
        row for row in summary
        if row["scope"] == "dataset_macro" and row["dataset"] == "ALL"
        and row["protocol"] == "patch37_Q_noninj"
        and str(row["subset_protocol"]).startswith("main_")
    ]
    if not selected:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, METRICS):
        for method in METHODS:
            rows = sorted([row for row in selected if row["method"] == method], key=lambda row: row["p"])
            axis.plot([100 * row["p"] for row in rows], [row[metric] for row in rows], marker="o", label=method)
            axis.fill_between(
                [100 * row["p"] for row in rows],
                [row[f"{metric}_ci95_low"] for row in rows],
                [row[f"{metric}_ci95_high"] for row in rows], alpha=.18,
            )
        axis.set_xlabel("controlled contamination (%)"); axis.set_ylabel(metric); axis.legend()
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "C_contamination_curve.png", dpi=180); plt.close(fig)


def _report(
    out_dir: Path,
    *,
    formal: bool,
    a_support: bool,
    bootstrap_b: list[dict],
    bootstrap_c: list[dict],
    main_max_p: float,
) -> None:
    b_contrast = [row for row in bootstrap_b if row["quantity"] == "knn_minus_gbsp_gain"]
    c_contrast = [
        row for row in bootstrap_c
        if row["quantity"] == "gbsp_minus_knn_slope" and row["protocol"] == "patch37_Q_noninj"
    ]
    b_support = len(b_contrast) == 2 and all(float(row["ci95_low"]) > 0 for row in b_contrast)
    c_support = len(c_contrast) == 2 and all(float(row["ci95_low"]) > 0 for row in c_contrast)
    if formal:
        support_count = sum((a_support, b_support, c_support))
        conclusion = (
            "strong support" if support_count == 3 else
            "partial support / narrow claim" if support_count else
            "not support / drop claim"
        )
    else:
        conclusion = "SMOKE ONLY — no formal conclusion"
    lines = [
        "# Reference-contamination mechanism report",
        "",
        f"- Run status: {'formal full6473' if formal else 'subset/smoke'}",
        f"- Experiment A support gate: {a_support}",
        f"- Experiment B support gate: {b_support}",
        f"- Experiment C support gate: {c_support} (main range 0–{100*main_max_p:g}%)",
        f"- Final conclusion: **{conclusion}**",
        "",
        "GT was used only for diagnosis and declared oracle intervention; no pseudo-label was generated.",
    ]
    (out_dir / "mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "conclusion_gate.json", {
        "formal": formal, "A_support": a_support, "B_support": b_support,
        "C_support": c_support, "main_max_p": main_max_p, "conclusion": conclusion,
    })


def main() -> None:
    args = parse_args()
    cache_root, out_dir = Path(args.cache_root).resolve(), Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache_root, split=args.split, max_samples=args.max_samples)
    payloads = [load_torch(row["cache_path"]) for row in manifest]
    contamination_modes = {
        str(payload.get("settings", {}).get("contamination_mode", "random_scattered"))
        for payload in payloads
    }
    if len(contamination_modes) != 1:
        raise RuntimeError(f"mixed contamination modes are not comparable: {contamination_modes}")
    contamination_mode = next(iter(contamination_modes))
    p_values = sorted({
        float(item["p"]) for payload in payloads for item in payload["contamination"]
    })
    max_requested_p = max(p_values, default=0.0)
    incomplete_native = [
        f"{normalize_dataset(payload.get('dataset', ''))}/{payload.get('stem', '')}"
        for payload in payloads if not payload.get("native_metrics_complete", True)
    ]
    if incomplete_native:
        raise RuntimeError(
            "native B metrics are still deferred for "
            f"{len(incomplete_native)} caches (first={incomplete_native[:3]}); run "
            "tools/evaluate_reference_contamination_native.py before analysis"
        )
    counts = {dataset: sum(normalize_dataset(row["dataset"]) == dataset for row in manifest) for dataset in DATASETS}
    formal = len(manifest) == 6473 and counts == EXPECTED_COUNTS

    natural_fields = [
        "dataset", "stem", "patch_index", "global_contamination", "neighbor_indices",
        "neighbor_gt_labels", "local_wrong_count", "local_contamination", "amplification",
        "knn8_score",
    ]
    local_handle, local_writer = _csv_stream(out_dir / "natural_local_amplification.csv", natural_fields)
    shift_fields = [
        "dataset", "stem", "patch_index", "method", "natural_score", "clean_score", "delta",
        "natural_local_wrong_count",
    ]
    shift_handle, shift_writer = _csv_stream(out_dir / "oracle_clean_query_shift.csv", shift_fields)

    candidate_rows, oracle_rows, matched_rows = [], [], []
    image_c_rows, validity_rows, local_link_rows = [], [], []
    global_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    local_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    amp_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    group_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    shift_chunks: dict[str, list[np.ndarray]] = defaultdict(list)

    query_manifest = []
    query_handle = query_writer = None
    query_path = out_dir / ("contamination_query_stats.csv.gz" if args.query_storage == "csv_gz" else "contamination_query_stats.csv")
    query_fields = [
        "dataset", "stem", "patch_index", "seed", "p", "injected_flag", "gt_label",
        "local_wrong_count", "local_contamination", "knn8_score", "gbsp_score",
    ]
    conditions = sum(int(item["valid"]) for payload in payloads for item in payload["contamination"])
    estimated_rows = conditions * 1369
    if args.query_storage in ("csv", "csv_gz"):
        bytes_per_row = 95 if args.query_storage == "csv" else 24
        required = estimated_rows * bytes_per_row
        free = shutil.disk_usage(out_dir).free
        if required * 1.2 > free:
            raise RuntimeError(
                f"query CSV preflight refused: estimated {required/2**30:.1f} GiB, free {free/2**30:.1f} GiB"
            )
        query_handle, query_writer = _csv_stream(
            query_path, query_fields, gzip_output=args.query_storage == "csv_gz"
        )

    oracle_records_for_bootstrap = []
    all_ids = set()
    valid5_ids, valid3_ids = set(), set()
    valid_ids_by_p = {value: set() for value in p_values}
    for payload in payloads:
        dataset, stem = normalize_dataset(payload["dataset"]), str(payload["stem"])
        identity = (dataset, stem)
        all_ids.add(identity)
        labels = payload["patch_gt_label"].numpy().astype(np.uint8)
        natural, clean = payload["natural"], payload["oracle_clean"]
        rho = float(natural["global_contamination"])
        q_mask = natural["query_fg_mask"].numpy().astype(bool)
        q_idx = np.flatnonzero(q_mask)
        neighbors = natural["neighbor_indices"].long().numpy()
        nwrong = natural["local_wrong_count"].numpy().astype(np.int16)
        local_rho = nwrong[q_idx] / 8.0
        amplification = local_rho / (rho + 1e-8) if rho > 0 else np.full(q_idx.size, np.nan)
        candidate_rows.append({
            "dataset": dataset, "stem": stem,
            "candidate_count": natural["candidate_count"],
            "wrong_candidate_count": natural["wrong_candidate_count"],
            "global_contamination": rho, "Q_FG_count": q_idx.size,
            **_quantiles(local_rho, "local_contamination_"),
            **_quantiles(amplification, "amplification_"),
            "prob_local_wrong_ge1": float((nwrong[q_idx] >= 1).mean()) if q_idx.size else float("nan"),
            "prob_local_wrong_ge2": float((nwrong[q_idx] >= 2).mean()) if q_idx.size else float("nan"),
            "prob_local_wrong_ge4": float((nwrong[q_idx] >= 4).mean()) if q_idx.size else float("nan"),
        })
        # Global contamination is one image-level observation; local
        # contamination is one query-level observation.
        global_chunks[dataset].append(np.asarray([rho], dtype=np.float32))
        local_chunks[dataset].append(local_rho.astype(np.float32))
        if rho > 0:
            amp_chunks[dataset].append(amplification.astype(np.float32))
        knn_q = natural["knn8_score"].numpy()[q_idx]
        for group, mask in (
            ("0", nwrong[q_idx] == 0), ("1", nwrong[q_idx] == 1),
            ("2", nwrong[q_idx] == 2), (">=3", nwrong[q_idx] >= 3),
        ):
            group_scores[group].append(knn_q[mask])
        for offset, patch_index in enumerate(q_idx):
            nbr = neighbors[patch_index]
            local_writer.writerow({
                "dataset": dataset, "stem": stem, "patch_index": int(patch_index),
                "global_contamination": rho,
                "neighbor_indices": ";".join(map(str, nbr.tolist())),
                "neighbor_gt_labels": ";".join(map(str, labels[nbr].tolist())),
                "local_wrong_count": int(nwrong[patch_index]),
                "local_contamination": float(local_rho[offset]),
                "amplification": float(amplification[offset]),
                "knn8_score": float(natural["knn8_score"][patch_index]),
            })

        # B: per-image native metrics and hit-query shifts.
        for method in METHODS:
            nat = clean["natural_native_metrics"][method]
            cln = clean["clean_native_metrics"][method]
            row = {
                "dataset": dataset, "stem": stem, "method": method,
                "natural_AP": nat["AP"], "clean_AP": cln["AP"], "gain_AP": cln["AP"] - nat["AP"],
                "natural_AUROC": nat["AUROC"], "clean_AUROC": cln["AUROC"],
                "gain_AUROC": cln["AUROC"] - nat["AUROC"],
            }
            oracle_rows.append(row); oracle_records_for_bootstrap.append(row)
        hit = q_mask & (nwrong > 0)
        for method, natural_key, clean_key in (
            ("knn8", "knn8_score", "knn8_score"), ("gbsp", "gbsp_score", "gbsp_score")
        ):
            before, after = natural[natural_key].numpy(), clean[clean_key].numpy()
            shift_chunks[method].append((after[hit] - before[hit]).astype(np.float32))
            for patch_index in np.flatnonzero(hit):
                shift_writer.writerow({
                    "dataset": dataset, "stem": stem, "patch_index": int(patch_index),
                    "method": method, "natural_score": float(before[patch_index]),
                    "clean_score": float(after[patch_index]),
                    "delta": float(after[patch_index] - before[patch_index]),
                    "natural_local_wrong_count": int(nwrong[patch_index]),
                })
        for matched in payload.get("matched_size_clean", []):
            if not matched["valid"]:
                matched_rows.append({
                    "dataset": dataset, "stem": stem, "seed": matched["seed"],
                    "valid": 0, "reason": matched["reason"],
                })
                continue
            for method in METHODS:
                metric = matched["native_metrics"][method]
                natural_metric = clean["natural_native_metrics"][method]
                matched_rows.append({
                    "dataset": dataset, "stem": stem, "seed": matched["seed"], "valid": 1,
                    "method": method, "AP": metric["AP"], "AUROC": metric["AUROC"],
                    "gain_AP": metric["AP"] - natural_metric["AP"],
                    "gain_AUROC": metric["AUROC"] - natural_metric["AUROC"],
                })

        # C: image metrics, local link, and optional exact per-query export.
        contamination = payload["contamination"]
        for p_value in p_values:
            conditions = [
                item for item in contamination if abs(float(item["p"]) - p_value) < 1e-12
            ]
            if conditions and all(item["valid"] for item in conditions):
                valid_ids_by_p[p_value].add(identity)
        valid5 = all(item["valid"] for item in contamination if abs(float(item["p"]) - .05) < 1e-12)
        valid3 = all(item["valid"] for item in contamination if float(item["p"]) <= .03 + 1e-12)
        if valid5: valid5_ids.add(identity)
        if valid3: valid3_ids.add(identity)
        validity_rows.append({
            "dataset": dataset, "stem": stem, "valid_5pct": int(valid5), "valid_3pct": int(valid3),
            "valid_max_requested": int(identity in valid_ids_by_p.get(max_requested_p, set())),
            "max_requested_p": max_requested_p,
            "foreground_patch_count": int(labels.sum()), "clean_candidate_count": clean["candidate_count"],
        })
        npz_conditions = []
        for item in contamination:
            if not item["valid"]:
                continue
            injected = np.zeros(1369, dtype=np.uint8)
            injected[item["injected_foreground_indices"].long().numpy()] = 1
            wrong = item["local_wrong_count"].numpy().astype(np.uint8)
            knn_score = item["knn8_score"].numpy().astype(np.float32)
            gbsp_score = item["gbsp_score"].numpy().astype(np.float32)
            noninj = injected == 0
            fg_noninj = (labels == 1) & noninj
            rho_local = wrong.astype(np.float32) / 8.0
            corr, pvalue = _safe_spearman(rho_local[fg_noninj], knn_score[fg_noninj])
            local_link_rows.append({
                "dataset": dataset, "stem": stem, "seed": item["seed"], "p": item["p"],
                "realized_fraction": item["realized_fraction"],
                "mean_local_contamination_all_noninj": float(rho_local[noninj].mean()),
                "mean_local_contamination_fg_noninj": float(rho_local[fg_noninj].mean()) if fg_noninj.any() else float("nan"),
                "spearman_local_contamination_vs_knn_score_fg": corr, "spearman_pvalue": pvalue,
            })
            for method in METHODS:
                for protocol, metric_key in (
                    ("patch37_Q_noninj", "patch_metrics_noninjected"),
                    ("patch37_full_query", "patch_metrics_full"),
                ):
                    metric = item[metric_key][method]
                    image_c_rows.append({
                        "dataset": dataset, "stem": stem, "seed": item["seed"], "p": item["p"],
                        "realized_fraction": item["realized_fraction"], "method": method,
                        "protocol": protocol, "AP": metric["AP"], "AUROC": metric["AUROC"],
                    })
            if args.query_storage == "compressed_npz":
                npz_conditions.append((item, injected, wrong, knn_score, gbsp_score))
            elif query_writer is not None:
                for patch_index in range(1369):
                    query_writer.writerow({
                        "dataset": dataset, "stem": stem, "patch_index": patch_index,
                        "seed": item["seed"], "p": item["p"],
                        "injected_flag": int(injected[patch_index]), "gt_label": int(labels[patch_index]),
                        "local_wrong_count": int(wrong[patch_index]),
                        "local_contamination": float(wrong[patch_index] / 8.0),
                        "knn8_score": float(knn_score[patch_index]), "gbsp_score": float(gbsp_score[patch_index]),
                    })
        if args.query_storage == "compressed_npz" and npz_conditions:
            shard = out_dir / "contamination_query_stats_shards" / dataset / f"{stem}.npz"
            shard.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                shard,
                seed=np.asarray([item[0]["seed"] for item in npz_conditions], dtype=np.int16),
                p=np.asarray([item[0]["p"] for item in npz_conditions], dtype=np.float32),
                realized_fraction=np.asarray([item[0]["realized_fraction"] for item in npz_conditions], dtype=np.float32),
                patch_index=np.arange(1369, dtype=np.int16), gt_label=labels.astype(np.uint8),
                injected_flag=np.stack([item[1] for item in npz_conditions]),
                local_wrong_count=np.stack([item[2] for item in npz_conditions]),
                knn8_score=np.stack([item[3] for item in npz_conditions]),
                gbsp_score=np.stack([item[4] for item in npz_conditions]),
            )
            query_manifest.append({
                "dataset": dataset, "stem": stem, "storage": "compressed_npz",
                "shard_path": str(shard), "condition_count": len(npz_conditions),
                "queries_per_condition": 1369, "row_count": 1369 * len(npz_conditions),
                "schema": "seed,p,realized_fraction,patch_index,gt_label,injected_flag,local_wrong_count,knn8_score,gbsp_score; local_contamination=local_wrong_count/8",
            })

    local_handle.close(); shift_handle.close()
    if query_handle is not None: query_handle.close()
    write_csv(out_dir / "natural_candidate_stats.csv", candidate_rows)
    write_csv(out_dir / "oracle_clean_image_metrics.csv", oracle_rows)
    write_csv(out_dir / "oracle_clean_matched_size_image_metrics.csv", matched_rows)
    write_csv(out_dir / "contamination_image_metrics.csv", image_c_rows)
    write_csv(out_dir / "contamination_validity.csv", validity_rows)
    write_csv(out_dir / "contamination_local_link.csv", local_link_rows)
    if args.query_storage == "compressed_npz":
        write_csv(out_dir / "contamination_query_stats.csv", query_manifest)

    # Experiment A summaries.
    amplification_summary, group_summary = [], []
    for dataset in (*DATASETS, "ALL"):
        global_values = np.concatenate([
            chunk for key, chunks in global_chunks.items() if dataset == "ALL" or key == dataset for chunk in chunks
        ]) if any(dataset == "ALL" or key == dataset for key in global_chunks) else np.asarray([])
        local_values = np.concatenate([
            chunk for key, chunks in local_chunks.items() if dataset == "ALL" or key == dataset for chunk in chunks
        ]) if any(dataset == "ALL" or key == dataset for key in local_chunks) else np.asarray([])
        amp_values = np.concatenate([
            chunk for key, chunks in amp_chunks.items() if dataset == "ALL" or key == dataset for chunk in chunks
        ]) if any(dataset == "ALL" or key == dataset for key in amp_chunks) else np.asarray([])
        if not local_values.size: continue
        corr, pvalue = _safe_spearman(local_values, np.concatenate([
            natural["knn8_score"].numpy()[natural["query_fg_mask"].numpy().astype(bool)]
            for natural, payload in ((payload["natural"], payload) for payload in payloads)
            if dataset == "ALL" or normalize_dataset(payload["dataset"]) == dataset
        ]))
        amplification_summary.append({
            "dataset": dataset, "query_count": local_values.size,
            **_quantiles(global_values, "global_"), **_quantiles(local_values, "local_"),
            **_quantiles(amp_values, "amplification_"),
            "prob_local_wrong_ge1": float((local_values >= 1/8).mean()),
            "prob_local_wrong_ge2": float((local_values >= 2/8).mean()),
            "prob_local_wrong_ge4": float((local_values >= 4/8).mean()),
            "prob_amplification_gt2": float((amp_values > 2).mean()) if amp_values.size else float("nan"),
            "prob_amplification_gt5": float((amp_values > 5).mean()) if amp_values.size else float("nan"),
            "prob_amplification_gt10": float((amp_values > 10).mean()) if amp_values.size else float("nan"),
            "spearman_local_contamination_vs_knn_score": corr, "spearman_pvalue": pvalue,
        })
    for group in ("0", "1", "2", ">=3"):
        values = np.concatenate(group_scores[group]) if group_scores[group] else np.asarray([])
        group_summary.append({"local_wrong_group": group, "query_count": values.size, **_quantiles(values, "knn8_score_")})
    write_csv(out_dir / "natural_amplification_summary.csv", amplification_summary)
    write_csv(out_dir / "natural_local_count_score_groups.csv", group_summary)

    shift_summary = []
    for method in METHODS:
        values = np.concatenate(shift_chunks[method]) if shift_chunks[method] else np.asarray([])
        shift_summary.append({
            "method": method, "hit_query_count": values.size,
            **_quantiles(values, "delta_"),
            "positive_shift_ratio": float((values > 0).mean()) if values.size else float("nan"),
        })
    write_csv(out_dir / "oracle_clean_query_shift_summary.csv", shift_summary)

    oracle_summary = []
    for dataset in (*DATASETS, "ALL"):
        for method in METHODS:
            subset = [
                row for row in oracle_rows
                if row["method"] == method and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            if subset:
                oracle_summary.append({
                    "scope": "image_macro" if dataset == "ALL" else "dataset",
                    "dataset": dataset, "method": method, "image_count": len(subset),
                    **{
                        field: float(np.mean([row[field] for row in subset]))
                        for field in (
                            "natural_AP", "clean_AP", "gain_AP",
                            "natural_AUROC", "clean_AUROC", "gain_AUROC",
                        )
                    },
                })
    for method in METHODS:
        dataset_rows = [
            row for row in oracle_summary
            if row["scope"] == "dataset" and row["method"] == method
        ]
        if dataset_rows:
            oracle_summary.append({
                "scope": "dataset_macro", "dataset": "DATASET_MACRO", "method": method,
                "image_count": sum(int(row["image_count"]) for row in dataset_rows),
                **{
                    field: float(np.mean([row[field] for row in dataset_rows]))
                    for field in (
                        "natural_AP", "clean_AP", "gain_AP",
                        "natural_AUROC", "clean_AUROC", "gain_AUROC",
                    )
                },
            })
    write_csv(out_dir / "oracle_clean_summary.csv", oracle_summary)

    matched_summary = []
    for dataset in (*DATASETS, "ALL"):
        for method in METHODS:
            for seed in sorted({int(row["seed"]) for row in matched_rows if row.get("valid") == 1}):
                subset = [
                    row for row in matched_rows
                    if row.get("valid") == 1 and row.get("method") == method and int(row["seed"]) == seed
                    and (dataset == "ALL" or row["dataset"] == dataset)
                ]
                if subset:
                    matched_summary.append({
                        "dataset": dataset, "method": method, "seed": seed, "image_count": len(subset),
                        **{field: float(np.mean([row[field] for row in subset])) for field in ("AP", "AUROC", "gain_AP", "gain_AUROC")},
                    })
    write_csv(out_dir / "oracle_clean_matched_size_summary.csv", matched_summary)

    # Experiment B paired image bootstrap, including method-gain contrast.
    oracle_lookup = {(row["dataset"], row["stem"], row["method"]): row for row in oracle_records_for_bootstrap}
    oracle_bootstrap = []
    for metric_index, metric in enumerate(METRICS):
        records = []
        identities = sorted({(key[0], key[1]) for key in oracle_lookup})
        for dataset, stem in identities:
            if (dataset, stem, "knn8") not in oracle_lookup or (dataset, stem, "gbsp") not in oracle_lookup: continue
            knn = oracle_lookup[(dataset, stem, "knn8")][f"gain_{metric}"]
            gbsp = oracle_lookup[(dataset, stem, "gbsp")][f"gain_{metric}"]
            records.append({"dataset": dataset, "knn_gain": knn, "gbsp_gain": gbsp, "knn_minus_gbsp_gain": knn - gbsp})
        quantities = ("knn_gain", "gbsp_gain", "knn_minus_gbsp_gain")
        bootstrapped = _stratified_bootstrap_fields(
            records, fields=quantities, repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + 100 * metric_index,
        )
        for quantity in quantities:
            oracle_bootstrap.append({
                "metric": metric, "quantity": quantity, "image_count": len(records),
                "repetitions": args.bootstrap_repetitions, **bootstrapped[quantity],
            })
    write_csv(out_dir / "oracle_clean_bootstrap.csv", oracle_bootstrap)

    valid5_ratio = len(valid5_ids) / max(1, len(all_ids))
    valid3_ratio = len(valid3_ids) / max(1, len(all_ids))
    matched_comparison = contamination_mode in (
        "spatial_cluster", "random_scattered_matched"
    )
    if matched_comparison:
        usable_p = [
            value for value in p_values
            if value > 0 and valid_ids_by_p.get(value)
        ]
        main_max_p = max(usable_p, default=0.0)
        selected_ids = valid_ids_by_p.get(main_max_p, all_ids)
        mode_suffix = (
            "cluster" if contamination_mode == "spatial_cluster" else "random_scattered"
        )
        main_subset_protocol = (
            f"main_common_valid_{100 * main_max_p:g}pct_{mode_suffix}"
        )
    else:
        selected_ids, main_max_p = (valid5_ids, .05) if valid5_ratio >= .90 else (valid3_ids, .03)
        main_subset_protocol = (
            "main_common_valid_5pct" if main_max_p == .05 else "main_common_valid_3pct"
        )
    main_rows = [row for row in image_c_rows if float(row["p"]) <= main_max_p + 1e-12]
    curve = _aggregate_seed_curve(
        main_rows, selected_ids,
        subset_protocol=main_subset_protocol,
    )
    if not matched_comparison and main_max_p == .03 and valid5_ids:
        supplemental_rows = [row for row in image_c_rows if float(row["p"]) in (0.0, .05)]
        curve.extend(_aggregate_seed_curve(
            supplemental_rows, valid5_ids, subset_protocol="supplemental_common_valid_5pct",
        ))
    write_csv(out_dir / "contamination_curve.csv", curve)
    curve_summary = _curve_summary(curve)
    write_csv(out_dir / "contamination_curve_summary.csv", curve_summary)
    write_csv(out_dir / "contamination_method_gap.csv", _method_gap(curve))
    per_condition_curve = []
    for p_value in p_values:
        if p_value <= 0 or not valid_ids_by_p.get(p_value):
            continue
        condition_rows = [
            row for row in image_c_rows if float(row["p"]) in (0.0, p_value)
        ]
        per_condition_curve.extend(_aggregate_seed_curve(
            condition_rows,
            valid_ids_by_p[p_value],
            subset_protocol=f"per_condition_common_valid_{100 * p_value:g}pct",
        ))
    write_csv(out_dir / "contamination_curve_per_condition_valid.csv", per_condition_curve)
    write_csv(
        out_dir / "contamination_curve_per_condition_valid_summary.csv",
        _curve_summary(per_condition_curve),
    )
    write_csv(
        out_dir / "contamination_method_gap_per_condition_valid.csv",
        _method_gap(per_condition_curve),
    )
    slopes = _image_slopes(image_c_rows, selected_ids, main_max_p)
    write_csv(out_dir / "contamination_image_slopes.csv", slopes)
    slope_bootstrap = _slope_bootstrap(slopes, args.bootstrap_repetitions, args.bootstrap_seed + 1000)
    write_csv(out_dir / "contamination_slope_bootstrap.csv", slope_bootstrap)
    write_json(out_dir / "contamination_subset_protocol.json", {
        "total_images": len(all_ids), "valid_5pct_images": len(valid5_ids), "valid_5pct_ratio": valid5_ratio,
        "valid_3pct_images": len(valid3_ids), "valid_3pct_ratio": valid3_ratio,
        "contamination_mode": contamination_mode,
        "requested_p_values": p_values,
        "valid_images_by_p": {str(value): len(valid_ids_by_p[value]) for value in p_values},
        "main_subset": main_subset_protocol,
        "main_max_p": main_max_p,
    })
    validity_summary = []
    for dataset in (*DATASETS, "ALL"):
        subset = [row for row in validity_rows if dataset == "ALL" or row["dataset"] == dataset]
        if subset:
            validity_summary.append({
                "dataset": dataset, "image_count": len(subset),
                "valid_5pct_count": sum(int(row["valid_5pct"]) for row in subset),
                "valid_5pct_ratio": float(np.mean([row["valid_5pct"] for row in subset])),
                "valid_3pct_count": sum(int(row["valid_3pct"]) for row in subset),
                "valid_3pct_ratio": float(np.mean([row["valid_3pct"] for row in subset])),
                "valid_max_requested_count": sum(int(row["valid_max_requested"]) for row in subset),
                "valid_max_requested_ratio": float(np.mean([row["valid_max_requested"] for row in subset])),
                "max_requested_p": max_requested_p,
            })
    write_csv(out_dir / "contamination_validity_summary.csv", validity_summary)

    local_link_summary = []
    for dataset in (*DATASETS, "ALL"):
        keys = sorted({(row["p"], row["seed"]) for row in local_link_rows})
        for p, seed_value in keys:
            subset = [
                row for row in local_link_rows
                if (row["p"], row["seed"]) == (p, seed_value)
                and (dataset == "ALL" or row["dataset"] == dataset)
            ]
            if subset:
                local_link_summary.append({
                    "dataset": dataset, "p": p, "seed": seed_value, "image_count": len(subset),
                    **{
                        field: _finite_mean(row[field] for row in subset)
                        for field in (
                            "realized_fraction", "mean_local_contamination_all_noninj",
                            "mean_local_contamination_fg_noninj",
                            "spearman_local_contamination_vs_knn_score_fg",
                        )
                    },
                })
    write_csv(out_dir / "contamination_local_link_summary.csv", local_link_summary)

    all_amplification = next((row for row in amplification_summary if row["dataset"] == "ALL"), {})
    all_groups = {row["local_wrong_group"]: row for row in group_summary}
    group_means_valid = all(
        group in all_groups and math.isfinite(all_groups[group]["knn8_score_mean"])
        for group in ("0", "1", "2", ">=3")
    )
    monotonic_group = group_means_valid and all(
        all_groups[left]["knn8_score_mean"] >= all_groups[right]["knn8_score_mean"]
        for left, right in (("0", "1"), ("1", "2"), ("2", ">=3"))
    )
    a_support = bool(
        all_amplification
        and all_amplification["local_median"] > all_amplification["global_median"]
        and all_amplification["spearman_local_contamination_vs_knn_score"] < 0
        and all_amplification["spearman_pvalue"] < .05
    )
    write_json(out_dir / "experiment_A_gate.json", {
        "local_distribution_right_shift": bool(
            all_amplification
            and all_amplification["local_median"] > all_amplification["global_median"]
        ),
        "spearman_negative_and_p_lt_0.05": bool(
            all_amplification
            and all_amplification["spearman_local_contamination_vs_knn_score"] < 0
            and all_amplification["spearman_pvalue"] < .05
        ),
        "descriptive_group_means_monotonic": monotonic_group,
        "support": a_support,
    })
    _plot_a(
        out_dir,
        np.concatenate([chunk for chunks in global_chunks.values() for chunk in chunks]),
        np.concatenate([chunk for chunks in local_chunks.values() for chunk in chunks]),
        {key: np.concatenate(value) if value else np.asarray([]) for key, value in group_scores.items()},
    )
    _plot_b(out_dir, oracle_bootstrap); _plot_c(out_dir, curve_summary)
    _report(
        out_dir, formal=formal, a_support=a_support, bootstrap_b=oracle_bootstrap,
        bootstrap_c=slope_bootstrap, main_max_p=main_max_p,
    )
    write_json(out_dir / "analysis_validity.json", {
        "version": VERSION, "images": len(manifest), "dataset_counts": counts,
        "formal_full6473": formal, "query_storage": args.query_storage,
        "contamination_mode": contamination_mode,
        "requested_p_values": p_values,
        "estimated_query_rows": estimated_rows,
        "required_csv_contract": {
            "natural_candidate_stats.csv": True,
            "natural_local_amplification.csv": True,
            "oracle_clean_image_metrics.csv": True,
            "oracle_clean_query_shift.csv": True,
            "contamination_curve.csv": True,
            "contamination_query_stats.csv": args.query_storage in ("compressed_npz", "csv"),
            "contamination_query_stats.csv.gz": args.query_storage == "csv_gz",
        },
        "gt_used_for_pseudo_label_generation": False,
    })
    print(json.dumps({
        "images": len(manifest), "formal": formal, "valid5_ratio": valid5_ratio,
        "contamination_mode": contamination_mode,
        "valid_images_by_p": {str(value): len(valid_ids_by_p[value]) for value in p_values},
        "main_max_p": main_max_p, "output": str(out_dir),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
