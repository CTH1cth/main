#!/usr/bin/env python3
"""Paired analysis of angular intervention strength in a BISA token table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


MAIN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MAIN_ROOT.parent
LOCAL_DEPENDENCY_ROOT = WORKSPACE_ROOT / "workdir" / "bisa_deps"
if LOCAL_DEPENDENCY_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_DEPENDENCY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


RAW_WEIGHTED = "weighted_bg"
RAW_RANDOM = "random_bg"
MATCHED_WEIGHTED = "weighted_bg_angle_matched"
MATCHED_RANDOM = "random_bg_angle_matched"
MODES = (RAW_WEIGHTED, RAW_RANDOM, MATCHED_WEIGHTED, MATCHED_RANDOM)
KEYS = (
    "dataset",
    "image_id",
    "checkpoint_epoch",
    "x",
    "y",
    "group_size",
    "teacher_error_type",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether angular displacement explains BISA random-vs-weighted behavior"
    )
    parser.add_argument("--token_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _task(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task == "tp_fp":
        subset = frame[frame["teacher_error_type"].isin(("TP", "FP"))]
        labels = (subset["teacher_error_type"] == "TP").to_numpy(dtype=np.int8)
    elif task == "fn_tn":
        subset = frame[frame["teacher_error_type"].isin(("FN", "TN"))]
        labels = (subset["teacher_error_type"] == "FN").to_numpy(dtype=np.int8)
    else:
        raise ValueError(task)
    return subset, labels


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size < 2 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size < 2 or np.unique(labels).size < 2 or not np.isfinite(scores).all():
        return math.nan
    return float(average_precision_score(labels, scores))


def _dataset_views(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for dataset, subset in frame.groupby("dataset", sort=True, observed=True):
        yield str(dataset), subset
    yield "Combined", frame


def _bootstrap_mean(
    values: np.ndarray, *, seed: int, replicates: int
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def _angle_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in _dataset_views(frame):
        for (epoch, mode), subset in dataset_frame.groupby(
            ["checkpoint_epoch", "substitution_mode"], sort=True, observed=True
        ):
            angle = subset["substitute_angle_deg"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_epoch": int(epoch),
                    "substitution_mode": str(mode),
                    "token_count": int(angle.size),
                    "angle_deg_mean": float(np.mean(angle)),
                    "angle_deg_median": float(np.median(angle)),
                    "angle_deg_q10": float(np.quantile(angle, 0.10)),
                    "angle_deg_q25": float(np.quantile(angle, 0.25)),
                    "angle_deg_q75": float(np.quantile(angle, 0.75)),
                    "angle_deg_q90": float(np.quantile(angle, 0.90)),
                    "cosine_gap_mean": float(subset["substitute_cosine_gap"].mean()),
                    "feature_l2_mean": float(subset["substitute_feature_l2_distance"].mean()),
                    "norm_abs_error_max": float(
                        (subset["substitute_feature_norm"] - subset["original_feature_norm"])
                        .abs()
                        .max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _response_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in _dataset_views(frame):
        for (epoch, mode), subset in dataset_frame.groupby(
            ["checkpoint_epoch", "substitution_mode"], sort=True, observed=True
        ):
            for task in ("tp_fp", "fn_tn"):
                task_frame, labels = _task(subset, task)
                response = task_frame["c_local_centered"].to_numpy(dtype=np.float64)
                angle = task_frame["substitute_angle_deg"].to_numpy(dtype=np.float64)
                correlation = spearmanr(angle, np.abs(response), nan_policy="omit")
                rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint_epoch": int(epoch),
                        "substitution_mode": str(mode),
                        "task": task,
                        "token_count": int(labels.size),
                        "positive_count": int(labels.sum()),
                        "response_roc_auc": _safe_auc(labels, response),
                        "response_average_precision": _safe_ap(labels, response),
                        "angle_only_roc_auc": _safe_auc(labels, angle),
                        "angle_abs_response_spearman": float(correlation.statistic),
                        "angle_abs_response_spearman_p": float(correlation.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def _paired_token_effects(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.pivot(index=list(KEYS), columns="substitution_mode", values=[
        "substitute_angle_deg", "substitute_cosine_gap", "c_local_centered"
    ])
    required = [
        (field, mode)
        for field in (
            "substitute_angle_deg",
            "substitute_cosine_gap",
            "c_local_centered",
        )
        for mode in MODES
    ]
    missing = [item for item in required if item not in values.columns]
    if missing:
        raise RuntimeError(f"Missing paired angle-analysis columns: {missing}")
    values = values.dropna(subset=required).reset_index()
    values.columns = [
        str(first) if not str(second) else f"{first}__{second}"
        for first, second in values.columns
    ]
    rows = []
    for dataset, dataset_frame in _dataset_views(values):
        for epoch, subset in dataset_frame.groupby(
            "checkpoint_epoch", sort=True, observed=True
        ):
            raw_angle_delta = (
                subset[f"substitute_angle_deg__{RAW_RANDOM}"].to_numpy()
                - subset[f"substitute_angle_deg__{RAW_WEIGHTED}"].to_numpy()
            )
            raw_abs_response_delta = (
                subset[f"c_local_centered__{RAW_RANDOM}"].abs().to_numpy()
                - subset[f"c_local_centered__{RAW_WEIGHTED}"].abs().to_numpy()
            )
            matched_error = (
                subset[f"substitute_angle_deg__{MATCHED_RANDOM}"]
                - subset[f"substitute_angle_deg__{MATCHED_WEIGHTED}"]
            ).abs().to_numpy()
            matched_cosine_gap_error = (
                subset[f"substitute_cosine_gap__{MATCHED_RANDOM}"]
                - subset[f"substitute_cosine_gap__{MATCHED_WEIGHTED}"]
            ).abs().to_numpy()
            correlation = spearmanr(
                raw_angle_delta, raw_abs_response_delta, nan_policy="omit"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint_epoch": int(epoch),
                    "paired_token_count": int(len(subset)),
                    "raw_random_minus_weighted_angle_deg_mean": float(raw_angle_delta.mean()),
                    "raw_random_greater_angle_fraction": float((raw_angle_delta > 0).mean()),
                    "raw_angle_delta_vs_abs_response_delta_spearman": float(
                        correlation.statistic
                    ),
                    "raw_angle_delta_vs_abs_response_delta_p": float(correlation.pvalue),
                    "matched_angle_abs_error_deg_mean": float(matched_error.mean()),
                    "matched_angle_abs_error_deg_max": float(matched_error.max()),
                    "matched_cosine_gap_abs_error_mean": float(
                        matched_cosine_gap_error.mean()
                    ),
                    "matched_cosine_gap_abs_error_max": float(
                        matched_cosine_gap_error.max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _per_image_auc(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, image_id, epoch, mode), subset in frame.groupby(
        ["dataset", "image_id", "checkpoint_epoch", "substitution_mode"],
        sort=True,
        observed=True,
    ):
        for task in ("tp_fp", "fn_tn"):
            task_frame, labels = _task(subset, task)
            auc = _safe_auc(
                labels, task_frame["c_local_centered"].to_numpy(dtype=np.float64)
            )
            rows.append(
                {
                    "dataset": str(dataset),
                    "image_id": str(image_id),
                    "checkpoint_epoch": int(epoch),
                    "substitution_mode": str(mode),
                    "task": task,
                    "image_auc": auc,
                }
            )
    return pd.DataFrame(rows)


def _auc_pair_matrix(
    frame: pd.DataFrame,
    task: str,
    mode: str,
    image_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode_frame, _ = _task(frame[frame["substitution_mode"] == mode], task)
    mode_frame = mode_frame.assign(
        _image_key=mode_frame["dataset"].astype(str) + "::" + mode_frame["image_id"].astype(str)
    )
    positive_name = "TP" if task == "tp_fp" else "FN"
    positive_scores: dict[str, np.ndarray] = {}
    negative_scores: dict[str, np.ndarray] = {}
    for image_key, subset in mode_frame.groupby("_image_key", sort=True, observed=True):
        positive_scores[str(image_key)] = subset.loc[
            subset["teacher_error_type"] == positive_name, "c_local_centered"
        ].to_numpy(dtype=np.float64)
        negative_scores[str(image_key)] = subset.loc[
            subset["teacher_error_type"] != positive_name, "c_local_centered"
        ].to_numpy(dtype=np.float64)
    count = len(image_keys)
    positive_count = np.asarray(
        [positive_scores.get(key, np.empty(0)).size for key in image_keys], dtype=np.float64
    )
    negative_count = np.asarray(
        [negative_scores.get(key, np.empty(0)).size for key in image_keys], dtype=np.float64
    )
    wins = np.zeros((count, count), dtype=np.float64)
    for positive_index, positive_key in enumerate(image_keys):
        positives = positive_scores.get(positive_key, np.empty(0))
        if positives.size == 0:
            continue
        for negative_index, negative_key in enumerate(image_keys):
            negatives = np.sort(negative_scores.get(negative_key, np.empty(0)))
            if negatives.size == 0:
                continue
            less = np.searchsorted(negatives, positives, side="left")
            less_equal = np.searchsorted(negatives, positives, side="right")
            wins[positive_index, negative_index] = float(
                np.sum(less + 0.5 * (less_equal - less))
            )
    return wins, positive_count, negative_count


def _weighted_auc_samples(
    weights: np.ndarray,
    wins: np.ndarray,
    positive_count: np.ndarray,
    negative_count: np.ndarray,
) -> np.ndarray:
    numerator = np.sum((weights @ wins) * weights, axis=1)
    denominator = (weights @ positive_count) * (weights @ negative_count)
    result = np.full(weights.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _paired_auc_effects(
    frame: pd.DataFrame, *, seed: int, replicates: int
) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in _dataset_views(frame):
        for epoch, epoch_frame in dataset_frame.groupby(
            "checkpoint_epoch", sort=True, observed=True
        ):
            image_keys = sorted(
                (
                    epoch_frame["dataset"].astype(str)
                    + "::"
                    + epoch_frame["image_id"].astype(str)
                ).unique().tolist()
            )
            for task in ("tp_fp", "fn_tn"):
                mode_matrices = {
                    mode: _auc_pair_matrix(epoch_frame, task, mode, image_keys)
                    for mode in MODES
                }
                reference_positive = mode_matrices[RAW_WEIGHTED][1]
                reference_negative = mode_matrices[RAW_WEIGHTED][2]
                for mode in MODES[1:]:
                    if not np.array_equal(reference_positive, mode_matrices[mode][1]):
                        raise RuntimeError(f"Positive label alignment failed for {mode}")
                    if not np.array_equal(reference_negative, mode_matrices[mode][2]):
                        raise RuntimeError(f"Negative label alignment failed for {mode}")
                point_weights = np.ones((1, len(image_keys)), dtype=np.float64)
                point_auc = {
                    mode: float(
                        _weighted_auc_samples(point_weights, *mode_matrices[mode])[0]
                    )
                    for mode in MODES
                }
                key_seed = (
                    int(seed)
                    + 1009 * int(epoch)
                    + (0 if task == "tp_fp" else 1)
                    + sum(ord(character) for character in dataset)
                )
                rng = np.random.default_rng(key_seed)
                weights = rng.multinomial(
                    len(image_keys),
                    np.full(len(image_keys), 1.0 / len(image_keys)),
                    size=int(replicates),
                ).astype(np.float64)
                sampled_auc = {
                    mode: _weighted_auc_samples(weights, *mode_matrices[mode])
                    for mode in MODES
                }
                raw_advantage = sampled_auc[RAW_RANDOM] - sampled_auc[RAW_WEIGHTED]
                matched_advantage = (
                    sampled_auc[MATCHED_RANDOM] - sampled_auc[MATCHED_WEIGHTED]
                )
                attenuation = raw_advantage - matched_advantage
                valid = (
                    np.isfinite(raw_advantage)
                    & np.isfinite(matched_advantage)
                    & np.isfinite(attenuation)
                )
                raw_advantage = raw_advantage[valid]
                matched_advantage = matched_advantage[valid]
                attenuation = attenuation[valid]
                raw_point = point_auc[RAW_RANDOM] - point_auc[RAW_WEIGHTED]
                matched_point = (
                    point_auc[MATCHED_RANDOM] - point_auc[MATCHED_WEIGHTED]
                )
                attenuation_point = raw_point - matched_point
                rows.append(
                    {
                        "dataset": dataset,
                        "checkpoint_epoch": int(epoch),
                        "task": str(task),
                        "eligible_image_count": int(len(image_keys)),
                        "valid_bootstrap_replicates": int(valid.sum()),
                        "raw_random_minus_weighted_image_auc": raw_point,
                        "raw_advantage_ci_low": float(np.quantile(raw_advantage, 0.025)),
                        "raw_advantage_ci_high": float(np.quantile(raw_advantage, 0.975)),
                        "matched_random_minus_weighted_image_auc": matched_point,
                        "matched_advantage_ci_low": float(np.quantile(matched_advantage, 0.025)),
                        "matched_advantage_ci_high": float(np.quantile(matched_advantage, 0.975)),
                        "advantage_attenuation": attenuation_point,
                        "attenuation_ci_low": float(np.quantile(attenuation, 0.025)),
                        "attenuation_ci_high": float(np.quantile(attenuation, 0.975)),
                        "attenuation_fraction_of_raw": (
                            attenuation_point / raw_point
                            if abs(raw_point) > 1e-12
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _plot(
    angle_summary: pd.DataFrame,
    response: pd.DataFrame,
    paired_auc: pd.DataFrame,
    path: Path,
) -> None:
    combined_angles = angle_summary[angle_summary["dataset"] == "Combined"]
    combined_response = response[
        (response["dataset"] == "Combined")
        & (response["task"].isin(("tp_fp", "fn_tn")))
    ]
    combined_paired = paired_auc[paired_auc["dataset"] == "Combined"]
    epochs = sorted(combined_angles["checkpoint_epoch"].unique().tolist())
    colors = {
        RAW_WEIGHTED: "#1f77b4",
        RAW_RANDOM: "#d62728",
        MATCHED_WEIGHTED: "#17becf",
        MATCHED_RANDOM: "#ff9896",
    }
    labels = {
        RAW_WEIGHTED: "weighted raw",
        RAW_RANDOM: "random raw",
        MATCHED_WEIGHTED: "weighted equal-angle",
        MATCHED_RANDOM: "random equal-angle",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for mode in MODES:
        subset = combined_angles[combined_angles["substitution_mode"] == mode]
        axes[0].plot(
            subset["checkpoint_epoch"], subset["angle_deg_mean"], marker="o",
            color=colors[mode], label=labels[mode]
        )
    axes[0].set_title("Mean angular displacement")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].set_ylabel("degrees")
    axes[0].set_xticks(epochs)
    axes[0].legend(fontsize=8)

    markers = {"tp_fp": "o", "fn_tn": "s"}
    for mode in MODES:
        for task in ("tp_fp", "fn_tn"):
            subset = combined_response[
                (combined_response["substitution_mode"] == mode)
                & (combined_response["task"] == task)
            ]
            axes[1].plot(
                subset["checkpoint_epoch"], subset["response_roc_auc"],
                marker=markers[task], color=colors[mode],
                linestyle="-" if task == "tp_fp" else "--",
                alpha=0.9,
            )
    axes[1].axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("C AUC: raw vs equal-angle")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_xticks(epochs)

    x = np.arange(len(combined_paired))
    axes[2].bar(
        x - 0.2,
        combined_paired["raw_random_minus_weighted_image_auc"],
        width=0.4,
        color="#9467bd",
        label="raw random-weighted",
    )
    axes[2].bar(
        x + 0.2,
        combined_paired["matched_random_minus_weighted_image_auc"],
        width=0.4,
        color="#8c564b",
        label="equal-angle random-weighted",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(
        x,
        [f"e{int(row.checkpoint_epoch)}\n{row.task}" for row in combined_paired.itertuples()],
        fontsize=8,
    )
    axes[2].set_title("Image-clustered pooled-token random advantage")
    axes[2].set_ylabel("ROC AUC difference")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap_replicates must be positive")
    token_path = Path(args.token_parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    _prepare_output(output_dir)
    columns = list(KEYS) + [
        "substitution_mode",
        "c_local_centered",
        "substitute_angle_deg",
        "substitute_cosine_gap",
        "substitute_feature_l2_distance",
        "original_feature_norm",
        "substitute_feature_norm",
    ]
    frame = pd.read_parquet(token_path, columns=columns)
    frame = frame[frame["substitution_mode"].isin(MODES)].copy()
    if set(frame["substitution_mode"].unique()) != set(MODES):
        raise RuntimeError(
            f"Expected modes {MODES}, got {sorted(frame['substitution_mode'].unique())}"
        )
    if not np.isfinite(
        frame[
            [
                "c_local_centered",
                "substitute_angle_deg",
                "substitute_cosine_gap",
                "substitute_feature_l2_distance",
                "original_feature_norm",
                "substitute_feature_norm",
            ]
        ].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("Angle audit input contains NaN/Inf")

    angle_summary = _angle_summary(frame)
    response = _response_metrics(frame)
    paired_tokens = _paired_token_effects(frame)
    per_image = _per_image_auc(frame)
    paired_auc = _paired_auc_effects(
        frame,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    angle_summary.to_csv(output_dir / "angle_distribution.csv", index=False)
    response.to_csv(output_dir / "angle_response_metrics.csv", index=False)
    paired_tokens.to_csv(output_dir / "paired_token_angle_effects.csv", index=False)
    per_image.to_csv(output_dir / "per_image_response_auc.csv", index=False)
    paired_auc.to_csv(output_dir / "paired_auc_angle_control.csv", index=False)

    combined = paired_auc[paired_auc["dataset"] == "Combined"]
    raw = combined["raw_random_minus_weighted_image_auc"].to_numpy(dtype=np.float64)
    matched = combined["matched_random_minus_weighted_image_auc"].to_numpy(dtype=np.float64)
    attenuation = combined["advantage_attenuation"].to_numpy(dtype=np.float64)
    matched_error = float(paired_tokens["matched_angle_abs_error_deg_max"].max())
    raw_mean = float(np.nanmean(raw))
    matched_mean = float(np.nanmean(matched))
    attenuation_mean = float(np.nanmean(attenuation))
    attenuation_fraction = (
        attenuation_mean / raw_mean if abs(raw_mean) > 1e-12 else math.nan
    )
    if (
        raw_mean > 0.0
        and matched_mean <= 0.01
        and attenuation_fraction >= 0.50
    ):
        interpretation = "ANGULAR_STRENGTH_EXPLAINS_MOST_RANDOM_ADVANTAGE"
    elif (
        raw_mean > 0.0
        and matched_mean > 0.01
        and attenuation_fraction < 0.25
    ):
        interpretation = "RANDOM_DIRECTION_ADVANTAGE_PERSISTS_AFTER_ANGLE_CONTROL"
    elif raw_mean > 0.0 and matched_mean < raw_mean and attenuation_mean > 0.0:
        interpretation = "MIXED_ANGLE_STRENGTH_AND_DIRECTION_EFFECTS"
    elif raw_mean > 0.0 and matched_mean >= raw_mean - 0.005:
        interpretation = "RANDOM_DIRECTION_ADVANTAGE_PERSISTS_AFTER_ANGLE_CONTROL"
    else:
        interpretation = "NO_STABLE_RANDOM_ADVANTAGE_TO_EXPLAIN"
    summary = {
        "status": interpretation,
        "token_parquet": str(token_path),
        "row_count": int(len(frame)),
        "sample_count": int(frame[["dataset", "image_id"]].drop_duplicates().shape[0]),
        "checkpoint_epochs": sorted(
            int(value) for value in frame["checkpoint_epoch"].unique()
        ),
        "matched_angle_max_abs_error_deg": matched_error,
        "matched_cosine_gap_max_abs_error": float(
            paired_tokens["matched_cosine_gap_abs_error_max"].max()
        ),
        "combined_case_count": int(len(combined)),
        "combined_mean_raw_random_minus_weighted_image_auc": raw_mean,
        "combined_mean_matched_random_minus_weighted_image_auc": matched_mean,
        "combined_mean_advantage_attenuation": attenuation_mean,
        "combined_advantage_attenuation_fraction": attenuation_fraction,
        "cases_where_equal_angle_reduced_random_advantage": int(
            np.sum(matched < raw)
        ),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "scope_note": (
            "This is a representative 50-image offline causal control. "
            "Equal-angle substitutes preserve each candidate direction and query norm, "
            "but rotate both modes to the per-token minimum original angle."
        ),
    }
    with (output_dir / "angle_strength_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _plot(
        angle_summary,
        response,
        paired_auc,
        output_dir / "angle_strength_summary.png",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
