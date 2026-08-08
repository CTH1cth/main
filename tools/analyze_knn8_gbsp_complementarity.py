#!/usr/bin/env python3
"""Stage 0/1: baseline audit and KNN8-conditioned GBSP complementarity."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gbsp_knn_lsr_common import (
    DATASETS, EXPECTED_COUNTS,
    correlation_metrics,
    index_manifest,
    load_manifest,
    load_patch_area,
    load_torch,
    normalize_dataset,
    patch_labels,
    rank_metrics,
    read_csv,
    resize_score_to_native,
    right_ecdf,
    score_from_payload,
    score_path,
    write_csv,
    write_json,
)
from tools.similarity_variation_common import equal_frequency_bins, similarity_matched_pairs


MAIN_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {
    ("dataset_macro", "nn_cos"): (0.7686119091911245, 0.9511118546137007),
    ("dataset_macro", "knn8_cos"): (0.7716604706496237, 0.9529828398382176),
    ("dataset_macro", "gbsp_r8"): (0.7726448926517061, 0.9524009945102647),
    ("image_macro", "nn_cos"): (0.7938193256370221, 0.9592953550910454),
    ("image_macro", "knn8_cos"): (0.8010468105269749, 0.9623001546818577),
    ("image_macro", "gbsp_r8"): (0.7995003643858171, 0.9613567191833511),
    ("historical_h20_nn_conditioned", "nn_cos"): (0.11684674919064977, 0.6037311611359003),
    ("historical_h20_nn_conditioned", "knn8_cos"): (0.2911342612688136, 0.7571824497633679),
    ("historical_h20_nn_conditioned", "gbsp_r8"): (0.23007381852693634, 0.7129916088780831),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--similarity_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--match_tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260807)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _payload_rows(similarity_root: Path, gbsp_root: Path, split: str, max_samples: int):
    sim_rows = load_manifest(similarity_root, score=True, split=split, max_samples=max_samples)
    gbsp_rows = load_manifest(gbsp_root, score=True, split=split, max_samples=-1)
    gbsp_index = index_manifest(gbsp_rows)
    output = []
    for row in sim_rows:
        key = (normalize_dataset(row["dataset"]), row["stem"])
        if key not in gbsp_index:
            raise RuntimeError(f"GBSP score missing for {key}")
        output.append((row, gbsp_index[key]))
    return output


def _baseline_audit(root: Path, complete: bool) -> tuple[list[dict], bool | None]:
    metric_path = root / "full_continuous_metrics.csv"
    observed = {}
    if metric_path.is_file():
        for row in read_csv(metric_path):
            if row.get("dataset") == "DATASET_MACRO" and row.get("protocol") == "dataset_macro_of_per_image_native":
                observed[("dataset_macro", row["method"])] = (float(row["ap"]), float(row["auroc"]))
            if row.get("dataset") == "ALL" and row.get("protocol") == "per_image_macro_native_gt":
                observed[("image_macro", row["method"])] = (float(row["ap"]), float(row["auroc"]))
    rows, passed = [], True
    for (scope, method), (reference_ap, reference_auroc) in REFERENCE.items():
        current = observed.get((scope, method))
        row = {
            "scope": scope, "method": method,
            "reference_AP": reference_ap, "reference_AUROC": reference_auroc,
            "current_AP": current[0] if current else float("nan"),
            "current_AUROC": current[1] if current else float("nan"),
            "AP_abs_error": abs(current[0] - reference_ap) if current else float("nan"),
            "AUROC_abs_error": abs(current[1] - reference_auroc) if current else float("nan"),
            "reference_note": "existing formal result; H20 here is the previous NN-conditioned subset" if scope.startswith("historical") else "formal frozen baseline",
        }
        if complete and scope in {"dataset_macro", "image_macro"}:
            ok = current is not None and row["AP_abs_error"] <= 1e-6 and row["AUROC_abs_error"] <= 1e-6
            row["reproduction_pass"] = int(ok)
            passed &= ok
        else:
            row["reproduction_pass"] = "not_applicable_for_subset_run" if not complete else "reference_only"
        rows.append(row)
    return rows, passed if complete else None


def _bin_statistics(values: dict, num_bins: int) -> list[dict]:
    rows = []
    datasets = sorted({key[0] for key in values})
    for dataset in [*datasets, "ALL"]:
        for rule in ("main_0.5", "strict_0.8_0.2"):
            for bin_index in range(1, num_bins + 1):
                selected = [value for key, value in values.items() if key[1:] == (rule, bin_index) and (dataset == "ALL" or key[0] == dataset)]
                if not selected:
                    continue
                labels = np.concatenate([item[0] for item in selected])
                score = np.concatenate([item[1] for item in selected])
                foreground, background = score[labels == 1], score[labels == 0]
                probability = float("nan")
                if foreground.size and background.size:
                    # We only use U/(n_fg*n_bg) as probability of superiority.
                    # Force the rank-based asymptotic path: SciPy's automatic
                    # exact recurrence can allocate many GiB for imbalanced bins.
                    u = mannwhitneyu(
                        foreground, background, alternative="two-sided", method="asymptotic"
                    ).statistic
                    probability = float(u / (foreground.size * background.size))
                metrics = rank_metrics(score, labels)
                rows.append({
                    "dataset": dataset, "label_rule": rule, "bin": bin_index,
                    "patch_count": score.size, "foreground_count": foreground.size, "background_count": background.size,
                    "fg_mean": float(foreground.mean()) if foreground.size else float("nan"),
                    "fg_median": float(np.median(foreground)) if foreground.size else float("nan"),
                    "fg_q25": float(np.quantile(foreground, 0.25)) if foreground.size else float("nan"),
                    "fg_q75": float(np.quantile(foreground, 0.75)) if foreground.size else float("nan"),
                    "bg_mean": float(background.mean()) if background.size else float("nan"),
                    "bg_median": float(np.median(background)) if background.size else float("nan"),
                    "bg_q25": float(np.quantile(background, 0.25)) if background.size else float("nan"),
                    "bg_q75": float(np.quantile(background, 0.75)) if background.size else float("nan"),
                    "mann_whitney_probability": probability,
                    "cliffs_delta": 2.0 * probability - 1.0 if math.isfinite(probability) else float("nan"),
                    "conditional_AP": metrics["AP"], "conditional_AUROC": metrics["AUROC"],
                })
    return rows


def _pair_bootstrap(images: list[dict], repetitions: int, seed: int) -> dict:
    arrays = [item["delta"] for item in images if item["delta"].size]
    if not arrays:
        return {
            "valid_pair_count": 0, "valid_image_count": 0, "pair_win_rate": float("nan"),
            "mean_delta_Q": float("nan"), "median_delta_Q": float("nan"),
        }
    delta = np.concatenate(arrays).astype(np.float64, copy=False)
    counts = np.asarray([array.size for array in arrays], dtype=np.int64)
    wins = np.asarray([(array > 0).sum() for array in arrays], dtype=np.float64)
    sums = np.asarray([array.sum() for array in arrays], dtype=np.float64)
    output = {
        "valid_pair_count": int(delta.size), "valid_image_count": len(arrays),
        "pair_win_rate": float((delta > 0).mean()),
        "mean_delta_Q": float(delta.mean()), "median_delta_Q": float(np.median(delta)),
    }
    if repetitions <= 0:
        return output
    rng = np.random.default_rng(seed)
    distribution = {name: np.empty(repetitions, dtype=np.float64) for name in ("win", "mean", "median")}
    sorted_order = np.argsort(delta, kind="mergesort")
    sorted_delta = delta[sorted_order]
    image_id = np.repeat(np.arange(len(arrays), dtype=np.int32), counts)
    sorted_image_id = image_id[sorted_order]
    for repetition in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, len(arrays), len(arrays)), minlength=len(arrays))
        total = int(np.dot(multiplicity, counts))
        distribution["win"][repetition] = float(np.dot(multiplicity, wins) / total)
        distribution["mean"][repetition] = float(np.dot(multiplicity, sums) / total)
        cumulative = np.cumsum(multiplicity[sorted_image_id], dtype=np.int64)
        position = int(np.searchsorted(cumulative, (total + 1) // 2, side="left"))
        distribution["median"][repetition] = sorted_delta[min(position, sorted_delta.size - 1)]
    for name, key in (("win", "pair_win_rate"), ("mean", "mean_delta_Q"), ("median", "median_delta_Q")):
        output[f"{key}_ci95_low"] = float(np.quantile(distribution[name], 0.025))
        output[f"{key}_ci95_high"] = float(np.quantile(distribution[name], 0.975))
    return output


def _pair_summaries(images: list[dict], repetitions: int, seed: int) -> list[dict]:
    rows = []
    for offset, dataset in enumerate([*DATASETS, "ALL"]):
        selected = [item for item in images if dataset == "ALL" or item["dataset"] == dataset]
        if selected:
            rows.append({"dataset": dataset, **_pair_bootstrap(selected, repetitions, seed + offset)})
    return rows


class _Fenwick:
    def __init__(self, size: int):
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        index += 1
        while index < self.tree.size:
            self.tree[index] += 1
            index += index & -index

    def prefix(self, stop: int) -> int:
        total = 0
        while stop > 0:
            total += int(self.tree[stop])
            stop -= stop & -stop
        return total


def _ranking_counts(d_score: np.ndarray, q_score: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict:
    fg = np.flatnonzero(valid & (labels == 1))
    bg = np.flatnonzero(valid & (labels == 0))
    if not fg.size or not bg.size:
        return {name: 0 for name in ("pairs", "both", "knn_only", "gbsp_only", "both_wrong", "knn_ties", "gbsp_ties")}
    bg_d = d_score[bg]
    bg_q = q_score[bg]
    sorted_bg_d = np.sort(bg_d)
    sorted_bg_q = np.sort(bg_q)
    knn_correct = int(np.searchsorted(sorted_bg_d, d_score[fg], side="left").sum())
    gbsp_correct = int(np.searchsorted(sorted_bg_q, q_score[fg], side="left").sum())
    q_coordinates = np.unique(bg_q)
    bg_order = np.argsort(bg_d, kind="mergesort")
    fg_order = fg[np.argsort(d_score[fg], kind="mergesort")]
    fenwick, cursor, both = _Fenwick(q_coordinates.size), 0, 0
    for f in fg_order:
        while cursor < bg.size and bg_d[bg_order[cursor]] < d_score[f]:
            fenwick.add(int(np.searchsorted(q_coordinates, bg_q[bg_order[cursor]], side="left")))
            cursor += 1
        both += fenwick.prefix(int(np.searchsorted(q_coordinates, q_score[f], side="left")))
    pairs = int(fg.size * bg.size)
    knn_ties = int((
        np.searchsorted(sorted_bg_d, d_score[fg], side="right")
        - np.searchsorted(sorted_bg_d, d_score[fg], side="left")
    ).sum())
    gbsp_ties = int((
        np.searchsorted(sorted_bg_q, q_score[fg], side="right")
        - np.searchsorted(sorted_bg_q, q_score[fg], side="left")
    ).sum())
    return {
        "pairs": pairs, "both": both,
        "knn_only": knn_correct - both, "gbsp_only": gbsp_correct - both,
        "both_wrong": pairs - (knn_correct + gbsp_correct - both),
        "knn_ties": knn_ties, "gbsp_ties": gbsp_ties,
    }


def _ranking_summary(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in [*DATASETS, "ALL"]:
        for rule in ("main_0.5", "strict_0.8_0.2"):
            selected = [row for row in rows if row["label_rule"] == rule and (dataset == "ALL" or row["dataset"] == dataset) and row["pairs"]]
            if not selected:
                continue
            pooled_pairs = sum(row["pairs"] for row in selected)
            for aggregation in ("pooled", "per_image_macro"):
                value = lambda key: (
                    sum(row[key] for row in selected) / pooled_pairs
                    if aggregation == "pooled" else
                    float(np.mean([row[key] / row["pairs"] for row in selected]))
                )
                output.append({
                    "dataset": dataset, "label_rule": rule, "aggregation": aggregation,
                    "valid_images": len(selected), "pair_count": pooled_pairs,
                    "both_rank_correct": value("both"), "KNN_only_correct": value("knn_only"),
                    "GBSP_only_correct": value("gbsp_only"), "both_wrong": value("both_wrong"),
                    "KNN_tie_rate": value("knn_ties"), "GBSP_tie_rate": value("gbsp_ties"),
                })
    return output


def main() -> None:
    args = parse_args()
    if args.knn_k != 8 or args.num_bins != 5 or abs(args.match_tolerance - 0.01) > 1e-12:
        raise ValueError("formal protocol is frozen to KNN8, five bins and tolerance=0.01")
    output = Path(args.out_dir).resolve()
    if output == MAIN_ROOT or MAIN_ROOT in output.parents:
        raise ValueError("output must stay outside the main code tree")
    output.mkdir(parents=True, exist_ok=True)
    pairs = _payload_rows(Path(args.similarity_root), Path(args.gbsp_root), args.split, args.max_samples)
    counts = Counter(normalize_dataset(row[0]["dataset"]) for row in pairs)
    complete = len(pairs) == 6473 and dict(counts) == EXPECTED_COUNTS
    baseline, baseline_pass = _baseline_audit(Path(args.similarity_root), complete)
    write_csv(output / "baseline_audit.csv", baseline)
    if complete and not baseline_pass:
        write_json(output / "protocol_difference_report.json", {
            "status": "STOP", "reason": "full6473 baseline reproduction mismatch", "baseline_audit": baseline,
        })
        raise RuntimeError("baseline reproduction mismatch; stopped before Stage 1")

    bins: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    pair_all, pair_high = [], []
    correlation_arrays: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    per_image_correlations, ranking_images = [], []
    self_violations = 0
    for index, (sim_row, gbsp_row) in enumerate(pairs, 1):
        sim_payload = load_torch(score_path(sim_row))
        gbsp_payload = sim_payload if score_path(sim_row).resolve() == score_path(gbsp_row).resolve() else load_torch(score_path(gbsp_row))
        dataset, stem = normalize_dataset(sim_row["dataset"]), sim_row["stem"]
        knn = score_from_payload(sim_payload, "knn8") .numpy()
        gbsp = score_from_payload(gbsp_payload, "gbsp").numpy()
        if int(sim_payload.get("self_match_violation_count", -1)) != 0 or not bool(sim_payload.get("self_match_excluded", False)):
            self_violations += int(sim_payload.get("self_match_violation_count", 1)) or 1
        area = load_patch_area(sim_row["gt_path"])
        bin_id = equal_frequency_bins(knn, args.num_bins)
        similarity = 1.0 - knn
        high = similarity >= np.quantile(similarity, 0.8)
        rank_knn, rank_gbsp = right_ecdf(knn), right_ecdf(gbsp)
        for rule in ("main_0.5", "strict_0.8_0.2"):
            labels, valid = patch_labels(area, rule)
            for bin_index in range(1, args.num_bins + 1):
                mask = valid & (bin_id == bin_index)
                key = (dataset, rule, bin_index)
                old = bins.get(key)
                current = (labels[mask], gbsp[mask])
                bins[key] = current if old is None else (np.concatenate([old[0], current[0]]), np.concatenate([old[1], current[1]]))
            ranking_images.append({"dataset": dataset, "stem": stem, "label_rule": rule, **_ranking_counts(knn, gbsp, labels, valid)})
            if rule == "main_0.5":
                fg, bg = similarity_matched_pairs(knn, labels, valid, args.match_tolerance)
                pair_all.append({"dataset": dataset, "stem": stem, "delta": gbsp[fg] - gbsp[bg]})
                fg_h, bg_h = similarity_matched_pairs(knn, labels, valid & ((labels == 0) | high), args.match_tolerance)
                keep = labels[fg_h] == 1
                pair_high.append({"dataset": dataset, "stem": stem, "delta": gbsp[fg_h[keep]] - gbsp[bg_h[keep]]})
        labels, valid = patch_labels(area, "main_0.5")
        scopes = {
            "overall": np.ones(1369, dtype=bool), "GT_foreground": labels == 1,
            "GT_background": labels == 0, "H20": high,
        }
        for scope, mask in scopes.items():
            metrics = correlation_metrics(rank_knn[mask], rank_gbsp[mask])
            per_image_correlations.append({"dataset": dataset, "stem": stem, "scope_name": scope, **metrics})
            correlation_arrays[(dataset, scope)].append((rank_knn[mask], rank_gbsp[mask]))
        if index % 100 == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] complementarity", flush=True)

    bin_rows = _bin_statistics(bins, args.num_bins)
    write_csv(output / "knn8_conditioned_bins.csv", bin_rows)
    matched = _pair_summaries(pair_all, args.bootstrap_repetitions, args.bootstrap_seed)
    high_matched = _pair_summaries(pair_high, args.bootstrap_repetitions, args.bootstrap_seed + 100)
    write_csv(output / "knn8_matched_pair_analysis.csv", matched)
    write_csv(output / "knn8_high_similarity_matched_pairs.csv", high_matched)
    ranking = _ranking_summary(ranking_images)
    write_csv(output / "ranking_complementarity.csv", ranking)

    correlations = []
    for dataset in [*DATASETS, "ALL"]:
        for scope in ("overall", "GT_foreground", "GT_background", "H20"):
            selected = [value for (ds, sc), arrays in correlation_arrays.items() if sc == scope and (dataset == "ALL" or ds == dataset) for value in arrays]
            if selected:
                a, b = np.concatenate([item[0] for item in selected]), np.concatenate([item[1] for item in selected])
                correlations.append({"dataset": dataset, "scope_name": scope, "aggregation": "pooled_after_per_image_ECDF", **correlation_metrics(a, b)})
            image_selected = [row for row in per_image_correlations if row["scope_name"] == scope and (dataset == "ALL" or row["dataset"] == dataset)]
            if image_selected:
                correlations.append({
                    "dataset": dataset, "scope_name": scope, "aggregation": "per_image_macro",
                    **{metric: float(np.nanmean([row[metric] for row in image_selected])) for metric in ("Pearson", "Spearman", "Kendall_tau")},
                })
    write_csv(output / "knn8_gbsp_correlation.csv", correlations)

    overall_pair = next(row for row in matched if row["dataset"] == "ALL")
    overall_ranking = next(row for row in ranking if row["dataset"] == "ALL" and row["label_rule"] == "main_0.5" and row["aggregation"] == "per_image_macro")
    dataset_rank = [row for row in ranking if row["dataset"] in DATASETS and row["label_rule"] == "main_0.5" and row["aggregation"] == "per_image_macro"]
    stable_gbsp_only = bool(dataset_rank) and all(float(row["GBSP_only_correct"]) > 0.0 for row in dataset_rank)
    win, low = float(overall_pair["pair_win_rate"]), float(overall_pair.get("pair_win_rate_ci95_low", float("nan")))
    high_ci = float(overall_pair.get("pair_win_rate_ci95_high", float("nan")))
    if not complete:
        case, allow_fusion = "SMOKE_ONLY_NO_ROUTE_DECISION", False
    elif win >= 0.58 and low > 0.5 and stable_gbsp_only:
        case, allow_fusion = "A_clear_complementarity", True
    elif (0.53 <= win < 0.58 or (low <= 0.5 <= high_ci)) and stable_gbsp_only:
        case, allow_fusion = "B_weak_complementarity", True
    else:
        case, allow_fusion = "C_little_additional_information", False
    validity = {
        "stage": "0_and_1", "images": len(pairs), "is_full_complete": complete,
        "baseline_reproduction_pass": baseline_pass,
        "self_match_violation_count": self_violations,
        "bin_generation_uses_gt": False, "knn_k": 8, "num_bins": 5,
        "match_tolerance": 0.01, "bootstrap_unit": "image_cluster",
        "bootstrap_repetitions": args.bootstrap_repetitions, "bootstrap_seed": args.bootstrap_seed,
        "pair_win_rate": win, "pair_win_rate_ci95_low": low,
        "GBSP_only_correct_per_image_macro": overall_ranking["GBSP_only_correct"],
        "complementarity_case": case, "allow_fusion": allow_fusion,
        "note": "H20 generated in this stage uses KNN8 similarity; historical H20 rows in baseline_audit used the previous NN-conditioned subset.",
    }
    write_json(output / "numerical_validity.json", validity)
    if self_violations:
        raise RuntimeError(f"self-match audit failed: {self_violations}")
    print(json.dumps(validity, ensure_ascii=False))


if __name__ == "__main__":
    main()
