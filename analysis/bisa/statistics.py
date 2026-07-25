"""Fixed BISA-v0 statistical analysis and conservative verdict generation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import RESPONSE_FIELDS, response_task_arrays, safe_binary_metrics


BASE_FEATURE_CANDIDATES = (
    "teacher_logit",
    "teacher_entropy",
    "dabe_target_soft",
    "dabe_background_residual",
    "dabe_background_connectivity",
    "bg_feature_l2_distance",
)
INTERVENTION_FEATURES = ("c_self_centered", "c_local_centered")
ERROR_TYPES = ("TN", "FP", "FN", "TP")


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def deterministic_token_sample(
    frame: pd.DataFrame,
    *,
    seed: int,
    max_per_image_class: int = 256,
) -> pd.DataFrame:
    """Cap each image/error class without using GT to choose hyperparameters."""
    parts = []
    for (image_id, error_type), subset in frame.groupby(
        ["image_id", "teacher_error_type"], sort=True, observed=True
    ):
        if str(error_type) not in ERROR_TYPES:
            continue
        if len(subset) <= int(max_per_image_class):
            parts.append(subset)
            continue
        rng = np.random.default_rng(
            _stable_seed(seed, f"{image_id}::{error_type}")
        )
        position = np.sort(
            rng.choice(len(subset), size=int(max_per_image_class), replace=False)
        )
        parts.append(subset.iloc[position])
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _task_subset(frame: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, np.ndarray]:
    if task == "tp_fp":
        subset = frame[frame["teacher_error_type"].isin(["TP", "FP"])].copy()
        labels = (subset["teacher_error_type"] == "TP").astype(np.int8).to_numpy()
    elif task == "fn_tn":
        subset = frame[frame["teacher_error_type"].isin(["FN", "TN"])].copy()
        labels = (subset["teacher_error_type"] == "FN").astype(np.int8).to_numpy()
    else:
        raise ValueError(task)
    return subset, labels


def _usable_features(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    result = []
    for column in candidates:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(values).all() and np.nanstd(values) > 1e-12:
            result.append(column)
    return result


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    penalty="l2",
                    solver="liblinear",
                    random_state=int(seed),
                    max_iter=1000,
                ),
            ),
        ]
    )


def _prediction_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "balanced_accuracy": np.nan}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
    }


def grouped_logistic_cv(
    frame: pd.DataFrame,
    task: str,
    *,
    seed: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    subset, labels = _task_subset(frame, task)
    groups = subset["image_id"].astype(str).to_numpy()
    base_features = _usable_features(subset, BASE_FEATURE_CANDIDATES)
    intervention_features = _usable_features(subset, INTERVENTION_FEATURES)
    if "teacher_logit" not in base_features or "teacher_entropy" not in base_features:
        raise RuntimeError("BISA base classifier requires Teacher logit and entropy")
    if "dabe_target_soft" not in base_features or "dabe_background_residual" not in base_features:
        raise RuntimeError("BISA base classifier requires DABE target and residual")
    unique_groups = np.unique(groups)
    if labels.size == 0 or np.unique(labels).size < 2 or unique_groups.size < 5:
        result = {
            "status": "insufficient_groups_or_classes",
            "base_features": base_features,
            "intervention_features": intervention_features,
            "sample_count": int(labels.size),
            "image_count": int(unique_groups.size),
            "base_roc_auc": np.nan,
            "base_pr_auc": np.nan,
            "base_plus_c_roc_auc": np.nan,
            "base_plus_c_pr_auc": np.nan,
            "delta_roc_auc": np.nan,
            "delta_pr_auc": np.nan,
        }
        return result, pd.DataFrame()

    x_base = subset[base_features].to_numpy(dtype=np.float64)
    x_plus = subset[base_features + intervention_features].to_numpy(dtype=np.float64)
    base_score = np.full(labels.shape, np.nan, dtype=np.float64)
    plus_score = np.full(labels.shape, np.nan, dtype=np.float64)
    fold_id = np.full(labels.shape, -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x_base, labels, groups=groups)
    ):
        if np.unique(labels[train_index]).size < 2 or np.unique(labels[validation_index]).size < 2:
            continue
        base_model = _classifier(seed + fold)
        plus_model = _classifier(seed + 100 + fold)
        base_model.fit(x_base[train_index], labels[train_index])
        plus_model.fit(x_plus[train_index], labels[train_index])
        base_score[validation_index] = base_model.predict_proba(x_base[validation_index])[:, 1]
        plus_score[validation_index] = plus_model.predict_proba(x_plus[validation_index])[:, 1]
        fold_id[validation_index] = fold
    valid = np.isfinite(base_score) & np.isfinite(plus_score)
    base_metrics = _prediction_metrics(labels[valid], base_score[valid])
    plus_metrics = _prediction_metrics(labels[valid], plus_score[valid])
    result = {
        "status": "ok" if bool(valid.all()) else "partial_valid_folds",
        "base_features": base_features,
        "intervention_features": intervention_features,
        "sample_count": int(valid.sum()),
        "image_count": int(np.unique(groups[valid]).size),
        "base_roc_auc": base_metrics["roc_auc"],
        "base_pr_auc": base_metrics["pr_auc"],
        "base_balanced_accuracy": base_metrics["balanced_accuracy"],
        "base_plus_c_roc_auc": plus_metrics["roc_auc"],
        "base_plus_c_pr_auc": plus_metrics["pr_auc"],
        "base_plus_c_balanced_accuracy": plus_metrics["balanced_accuracy"],
        "delta_roc_auc": plus_metrics["roc_auc"] - base_metrics["roc_auc"],
        "delta_pr_auc": plus_metrics["pr_auc"] - base_metrics["pr_auc"],
    }
    oof = pd.DataFrame(
        {
            "image_id": groups[valid],
            "label": labels[valid],
            "base_score": base_score[valid],
            "plus_score": plus_score[valid],
            "fold": fold_id[valid],
        }
    )
    return result, oof


def cross_dataset_logistic(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    task: str,
    *,
    seed: int,
) -> Dict[str, object]:
    train, train_labels = _task_subset(train_frame, task)
    test, test_labels = _task_subset(test_frame, task)
    base_features = [
        feature
        for feature in _usable_features(train, BASE_FEATURE_CANDIDATES)
        if feature in _usable_features(test, BASE_FEATURE_CANDIDATES)
    ]
    intervention_features = [
        feature
        for feature in _usable_features(train, INTERVENTION_FEATURES)
        if feature in _usable_features(test, INTERVENTION_FEATURES)
    ]
    if (
        np.unique(train_labels).size < 2
        or np.unique(test_labels).size < 2
        or not intervention_features
    ):
        return {"status": "insufficient_classes", "delta_roc_auc": np.nan, "delta_pr_auc": np.nan}
    base_model = _classifier(seed)
    plus_model = _classifier(seed + 1)
    base_model.fit(train[base_features].to_numpy(dtype=np.float64), train_labels)
    plus_model.fit(
        train[base_features + intervention_features].to_numpy(dtype=np.float64),
        train_labels,
    )
    base_score = base_model.predict_proba(test[base_features].to_numpy(dtype=np.float64))[:, 1]
    plus_score = plus_model.predict_proba(
        test[base_features + intervention_features].to_numpy(dtype=np.float64)
    )[:, 1]
    base_metrics = _prediction_metrics(test_labels, base_score)
    plus_metrics = _prediction_metrics(test_labels, plus_score)
    return {
        "status": "ok",
        "train_sample_count": int(len(train)),
        "test_sample_count": int(len(test)),
        "base_roc_auc": base_metrics["roc_auc"],
        "base_pr_auc": base_metrics["pr_auc"],
        "base_plus_c_roc_auc": plus_metrics["roc_auc"],
        "base_plus_c_pr_auc": plus_metrics["pr_auc"],
        "delta_roc_auc": plus_metrics["roc_auc"] - base_metrics["roc_auc"],
        "delta_pr_auc": plus_metrics["pr_auc"] - base_metrics["pr_auc"],
    }


def image_bootstrap_oof(
    oof: pd.DataFrame,
    *,
    seed: int,
    replicates: int = 10000,
) -> Dict[str, float]:
    if oof.empty or oof["label"].nunique() < 2 or oof["image_id"].nunique() < 2:
        return {
            "delta_roc_auc_ci_low": np.nan,
            "delta_roc_auc_ci_high": np.nan,
            "delta_pr_auc_ci_low": np.nan,
            "delta_pr_auc_ci_high": np.nan,
        }
    grouped = {key: value for key, value in oof.groupby("image_id", sort=True)}
    image_ids = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(int(seed))
    delta_auc = []
    delta_ap = []
    for _ in range(int(replicates)):
        chosen = rng.choice(image_ids, size=image_ids.size, replace=True)
        sampled = pd.concat([grouped[key] for key in chosen], ignore_index=True)
        labels = sampled["label"].to_numpy(dtype=np.int8)
        if np.unique(labels).size < 2:
            continue
        base = sampled["base_score"].to_numpy(dtype=np.float64)
        plus = sampled["plus_score"].to_numpy(dtype=np.float64)
        delta_auc.append(roc_auc_score(labels, plus) - roc_auc_score(labels, base))
        delta_ap.append(average_precision_score(labels, plus) - average_precision_score(labels, base))
    if not delta_auc:
        return {
            "delta_roc_auc_ci_low": np.nan,
            "delta_roc_auc_ci_high": np.nan,
            "delta_pr_auc_ci_low": np.nan,
            "delta_pr_auc_ci_high": np.nan,
        }
    return {
        "delta_roc_auc_ci_low": float(np.quantile(delta_auc, 0.025)),
        "delta_roc_auc_ci_high": float(np.quantile(delta_auc, 0.975)),
        "delta_pr_auc_ci_low": float(np.quantile(delta_ap, 0.025)),
        "delta_pr_auc_ci_high": float(np.quantile(delta_ap, 0.975)),
    }


def image_bootstrap_class_difference(
    frame: pd.DataFrame,
    task: str,
    score_column: str,
    *,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    if task == "tp_fp":
        positive_type, negative_type = "TP", "FP"
    elif task == "fn_tn":
        positive_type, negative_type = "FN", "TN"
    else:
        raise ValueError(task)
    differences = []
    for _, image in frame.groupby("image_id", sort=True, observed=True):
        positive = image.loc[
            image["teacher_error_type"] == positive_type, score_column
        ].to_numpy(dtype=np.float64)
        negative = image.loc[
            image["teacher_error_type"] == negative_type, score_column
        ].to_numpy(dtype=np.float64)
        if positive.size and negative.size:
            differences.append(float(np.median(positive) - np.median(negative)))
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_image_count": int(differences.size),
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, differences.size, size=(int(replicates), differences.size)
    )
    estimates = np.median(differences[indices], axis=1)
    return {
        "estimate": float(np.median(differences)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_image_count": int(differences.size),
    }


def image_bootstrap_mode_auc_difference(
    frame: pd.DataFrame,
    task: str,
    score_column: str,
    *,
    seed: int,
    replicates: int,
) -> Dict[str, float]:
    """Paired image bootstrap of weighted-minus-random per-image AUC."""
    per_image = []
    for _, image in frame.groupby("image_id", sort=True, observed=True):
        mode_auc = {}
        for mode in ("weighted_bg", "random_bg"):
            mode_frame = image[image["substitution_mode"] == mode]
            _, labels, scores = response_task_arrays(
                mode_frame, task, score_column
            )
            metrics = safe_binary_metrics(labels, scores)
            if math.isfinite(metrics["roc_auc"]):
                mode_auc[mode] = float(metrics["roc_auc"])
        if len(mode_auc) == 2:
            per_image.append(mode_auc["weighted_bg"] - mode_auc["random_bg"])
    per_image = np.asarray(per_image, dtype=np.float64)
    if per_image.size < 2:
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_image_count": int(per_image.size),
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, per_image.size, size=(int(replicates), per_image.size))
    estimates = np.mean(per_image[indices], axis=1)
    return {
        "estimate": float(np.mean(per_image)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_image_count": int(per_image.size),
    }


def paired_effects(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, 0.0):
        return {"wilcoxon_p": np.nan, "rank_biserial": np.nan, "cohen_dz": np.nan}
    test = wilcoxon(values, zero_method="wilcox", alternative="greater")
    nonzero = values[values != 0.0]
    ranks = rankdata(np.abs(nonzero))
    rank_positive = float(ranks[nonzero > 0.0].sum())
    rank_negative = float(ranks[nonzero < 0.0].sum())
    denominator = rank_positive + rank_negative
    rank_biserial = (rank_positive - rank_negative) / denominator if denominator else np.nan
    std = float(values.std(ddof=1))
    return {
        "wilcoxon_p": float(test.pvalue),
        "rank_biserial": float(rank_biserial),
        "cohen_dz": float(values.mean() / std) if std > 0.0 else np.nan,
    }


def _dataset_views(frame: pd.DataFrame):
    for dataset in sorted(frame["dataset"].unique()):
        yield str(dataset), frame[frame["dataset"] == dataset]
    yield "Combined", frame


def build_statistical_outputs(
    tokens: pd.DataFrame,
    images: pd.DataFrame,
    *,
    seed: int,
    full_run: bool,
    bootstrap_replicates: int = 10000,
    robustness_reference: str | Path | None = None,
    evaluate_verdict: bool = True,
) -> Dict[str, pd.DataFrame | dict]:
    response_rows = []
    incremental_rows = []
    bootstrap_rows = []
    cross_rows = []
    area_rows = []
    spill_rows = []
    ood_rows = []
    sampled_by_unit: Dict[tuple, pd.DataFrame] = {}

    unit_columns = ["checkpoint_epoch", "substitution_mode", "group_size"]
    for unit_key, unit in tokens.groupby(unit_columns, sort=True, observed=True):
        epoch, mode, group_size = unit_key
        for dataset_name, dataset_frame in _dataset_views(unit):
            image_count = int(dataset_frame["image_id"].nunique())
            for task in ("tp_fp", "fn_tn"):
                for response in RESPONSE_FIELDS:
                    subset, labels, score = response_task_arrays(dataset_frame, task, response)
                    metrics = safe_binary_metrics(labels, score)
                    positive = score[labels == 1]
                    negative = score[labels == 0]
                    response_rows.append(
                        {
                            "dataset": dataset_name,
                            "checkpoint_epoch": int(epoch),
                            "substitution_mode": str(mode),
                            "group_size": int(group_size),
                            "task": task,
                            "response": response,
                            **metrics,
                            "image_count": image_count,
                            "positive_median": float(np.median(positive)) if positive.size else np.nan,
                            "negative_median": float(np.median(negative)) if negative.size else np.nan,
                            "median_difference": (
                                float(np.median(positive) - np.median(negative))
                                if positive.size and negative.size
                                else np.nan
                            ),
                            "mean_response": float(np.mean(score)) if score.size else np.nan,
                            "positive_response_ratio": float(np.mean(score > 0.0)) if score.size else np.nan,
                        }
                    )
                class_difference = image_bootstrap_class_difference(
                    dataset_frame,
                    task,
                    "c_local_centered",
                    seed=_stable_seed(
                        seed,
                        f"class-difference:{epoch}:{mode}:{dataset_name}:{task}",
                    ),
                    replicates=(
                        int(bootstrap_replicates)
                        if full_run
                        else min(200, int(bootstrap_replicates))
                    ),
                )
                bootstrap_rows.append(
                    {
                        "dataset": dataset_name,
                        "checkpoint_epoch": int(epoch),
                        "substitution_mode": str(mode),
                        "group_size": int(group_size),
                        "task": task,
                        "quantity": "class_median_difference",
                        "response": "c_local_centered",
                        **class_difference,
                    }
                )

            sampled = deterministic_token_sample(dataset_frame, seed=seed)
            sampled_by_unit[(int(epoch), str(mode), int(group_size), dataset_name)] = sampled
            for task in ("tp_fp", "fn_tn"):
                result, oof = grouped_logistic_cv(sampled, task, seed=seed)
                row = {
                    "dataset": dataset_name,
                    "checkpoint_epoch": int(epoch),
                    "substitution_mode": str(mode),
                    "group_size": int(group_size),
                    "task": task,
                    **result,
                }
                incremental_rows.append(row)
                ci = image_bootstrap_oof(
                    oof,
                    seed=_stable_seed(seed, f"{epoch}:{mode}:{dataset_name}:{task}"),
                    replicates=int(bootstrap_replicates) if full_run else min(200, int(bootstrap_replicates)),
                )
                bootstrap_rows.append(
                    {
                        "dataset": dataset_name,
                        "checkpoint_epoch": int(epoch),
                        "substitution_mode": str(mode),
                        "group_size": int(group_size),
                        "task": task,
                        "quantity": "incremental_model",
                        **ci,
                    }
                )

        for dataset_name, dataset_frame in _dataset_views(unit):
            spill_values = dataset_frame["spill_ratio"].to_numpy(dtype=np.float64)
            spill_rows.append(
                {
                    "dataset": dataset_name,
                    "checkpoint_epoch": int(epoch),
                    "substitution_mode": str(mode),
                    "group_size": int(group_size),
                    "spill_ratio_mean": float(np.mean(spill_values)),
                    "spill_ratio_median": float(np.median(spill_values)),
                    "spill_ratio_p90": float(np.quantile(spill_values, 0.90)),
                    "image_count": int(dataset_frame["image_id"].nunique()),
                }
            )
            if str(mode) in {"weighted_bg", "nearest_bg", "random_bg"}:
                norm = dataset_frame[f"{mode}_feature_norm"].to_numpy(dtype=np.float64)
                original = dataset_frame["original_feature_norm"].to_numpy(dtype=np.float64)
                distance = dataset_frame[f"{mode}_nearest_anchor_distance"].to_numpy(dtype=np.float64)
                ood_rows.append(
                    {
                        "dataset": dataset_name,
                        "checkpoint_epoch": int(epoch),
                        "substitution_mode": str(mode),
                        "group_size": int(group_size),
                        "norm_ratio_mean": float(np.mean(norm / (original + 1e-8))),
                        "feature_norm_mean": float(np.mean(norm)),
                        "original_feature_norm_mean": float(np.mean(original)),
                        "nearest_anchor_distance_mean": float(np.mean(distance)),
                        "nearest_anchor_distance_p90": float(np.quantile(distance, 0.90)),
                    }
                )

    response_frame = pd.DataFrame(response_rows)
    incremental_frame = pd.DataFrame(incremental_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    weighted_units = tokens[tokens["substitution_mode"] == "weighted_bg"]
    for (epoch, group_size), unit in weighted_units.groupby(
        ["checkpoint_epoch", "group_size"], sort=True, observed=True
    ):
        for task in ("tp_fp", "fn_tn"):
            key_camo = (int(epoch), "weighted_bg", int(group_size), "TR-CAMO")
            key_cod = (int(epoch), "weighted_bg", int(group_size), "TR-COD10K")
            if key_camo in sampled_by_unit and key_cod in sampled_by_unit:
                for train_name, test_name, train_key, test_key in (
                    ("TR-CAMO", "TR-COD10K", key_camo, key_cod),
                    ("TR-COD10K", "TR-CAMO", key_cod, key_camo),
                ):
                    result = cross_dataset_logistic(
                        sampled_by_unit[train_key],
                        sampled_by_unit[test_key],
                        task,
                        seed=_stable_seed(seed, f"cross:{epoch}:{train_name}:{task}"),
                    )
                    cross_rows.append(
                        {
                            "checkpoint_epoch": int(epoch),
                            "group_size": int(group_size),
                            "task": task,
                            "train_dataset": train_name,
                            "test_dataset": test_name,
                            **result,
                        }
                    )

    if not weighted_units.empty:
        image_area = (
            weighted_units[["image_id", "dataset", "checkpoint_epoch", "target_area", "teacher_area"]]
            .drop_duplicates()
            .copy()
        )
        for area_field, area_name in (("target_area", "gt_area"), ("teacher_area", "teacher_area")):
            image_area[f"{area_name}_quartile"] = image_area.groupby(
                ["dataset", "checkpoint_epoch"], observed=True
            )[area_field].transform(
                lambda values: pd.qcut(values.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
            )
        enriched = weighted_units.merge(
            image_area[["image_id", "checkpoint_epoch", "gt_area_quartile", "teacher_area_quartile"]],
            on=["image_id", "checkpoint_epoch"],
            how="left",
            validate="many_to_one",
        )
        for quartile_field in ("gt_area_quartile", "teacher_area_quartile"):
            for (dataset, epoch, group_size, quartile), subset in enriched.groupby(
                ["dataset", "checkpoint_epoch", "group_size", quartile_field],
                sort=True,
                observed=True,
            ):
                for task in ("tp_fp", "fn_tn"):
                    task_subset, labels, score = response_task_arrays(
                        subset, task, "c_local_centered"
                    )
                    metric = safe_binary_metrics(labels, score)
                    positive = score[labels == 1]
                    negative = score[labels == 0]
                    area_rows.append(
                        {
                            "area_partition": quartile_field,
                            "dataset": str(dataset),
                            "checkpoint_epoch": int(epoch),
                            "group_size": int(group_size),
                            "quartile": str(quartile),
                            "task": task,
                            **metric,
                            "response_median": float(np.median(score)) if score.size else np.nan,
                            "median_difference": (
                                float(np.median(positive) - np.median(negative))
                                if positive.size and negative.size
                                else np.nan
                            ),
                            "spill_ratio_median": float(task_subset["spill_ratio"].median()) if len(task_subset) else np.nan,
                        }
                    )

    substitution_rows = []
    primary = response_frame[
        (response_frame["response"] == "c_local_centered")
        & (response_frame["group_size"] == 4)
    ]
    for keys, subset in primary.groupby(
        ["dataset", "checkpoint_epoch", "task"], sort=True, observed=True
    ):
        indexed = subset.set_index("substitution_mode")
        if "weighted_bg" in indexed.index and "random_bg" in indexed.index:
            comparison_tokens = tokens[
                (tokens["checkpoint_epoch"] == int(keys[1]))
                & (tokens["group_size"] == 4)
                & (tokens["substitution_mode"].isin(["weighted_bg", "random_bg"]))
            ]
            if str(keys[0]) != "Combined":
                comparison_tokens = comparison_tokens[
                    comparison_tokens["dataset"] == keys[0]
                ]
            comparison_ci = image_bootstrap_mode_auc_difference(
                comparison_tokens,
                str(keys[2]),
                "c_local_centered",
                seed=_stable_seed(
                    seed, f"weighted-random:{keys[0]}:{keys[1]}:{keys[2]}"
                ),
                replicates=(
                    int(bootstrap_replicates)
                    if full_run
                    else min(200, int(bootstrap_replicates))
                ),
            )
            substitution_rows.append(
                {
                    "dataset": keys[0],
                    "checkpoint_epoch": int(keys[1]),
                    "task": keys[2],
                    "weighted_bg_roc_auc": float(indexed.loc["weighted_bg", "roc_auc"]),
                    "random_bg_roc_auc": float(indexed.loc["random_bg", "roc_auc"]),
                    "weighted_minus_random_auc": float(
                        indexed.loc["weighted_bg", "roc_auc"]
                        - indexed.loc["random_bg", "roc_auc"]
                    ),
                    "paired_image_auc_difference": comparison_ci["estimate"],
                    "bootstrap_ci_low": comparison_ci["ci_low"],
                    "bootstrap_ci_high": comparison_ci["ci_high"],
                    "bootstrap_image_count": comparison_ci[
                        "bootstrap_image_count"
                    ],
                }
            )
            bootstrap_rows.append(
                {
                    "dataset": keys[0],
                    "checkpoint_epoch": int(keys[1]),
                    "substitution_mode": "weighted_bg_vs_random_bg",
                    "group_size": 4,
                    "task": keys[2],
                    "quantity": "weighted_minus_random_auc",
                    "response": "c_local_centered",
                    **comparison_ci,
                }
            )

    group_rows = []
    reference_path = Path(robustness_reference) if robustness_reference else None
    if reference_path is not None and reference_path.is_file():
        reference = pd.read_csv(reference_path)
        current = primary[primary["substitution_mode"] == "weighted_bg"]
        reference = reference[
            (reference["response"] == "c_local_centered")
            & (reference["substitution_mode"] == "weighted_bg")
        ]
        merged = current.merge(
            reference,
            on=["dataset", "checkpoint_epoch", "task", "response", "substitution_mode"],
            suffixes=("_g4", "_g3"),
        )
        for _, row in merged.iterrows():
            group_rows.append(
                {
                    "dataset": row["dataset"],
                    "checkpoint_epoch": int(row["checkpoint_epoch"]),
                    "task": row["task"],
                    "auc_group4": float(row["roc_auc_g4"]),
                    "auc_group3": float(row["roc_auc_g3"]),
                    "absolute_auc_difference": abs(float(row["roc_auc_g4"] - row["roc_auc_g3"])),
                    "response_direction_consistent": bool(
                        np.sign(row["median_difference_g4"])
                        == np.sign(row["median_difference_g3"])
                    ),
                }
            )

    image_effect_rows = []
    for keys, subset in tokens[
        (tokens["substitution_mode"] == "weighted_bg")
        & (tokens["group_size"] == 4)
    ].groupby(["dataset", "checkpoint_epoch", "image_id"], sort=True, observed=True):
        for task, positive_type, negative_type in (
            ("tp_fp", "TP", "FP"),
            ("fn_tn", "FN", "TN"),
        ):
            positive = subset.loc[
                subset["teacher_error_type"] == positive_type, "c_local_centered"
            ].to_numpy(dtype=np.float64)
            negative = subset.loc[
                subset["teacher_error_type"] == negative_type, "c_local_centered"
            ].to_numpy(dtype=np.float64)
            if positive.size and negative.size:
                image_effect_rows.append(
                    {
                        "dataset": keys[0],
                        "checkpoint_epoch": int(keys[1]),
                        "image_id": keys[2],
                        "task": task,
                        "paired_median_difference": float(np.median(positive) - np.median(negative)),
                    }
                )
    effect_frame = pd.DataFrame(image_effect_rows)
    paired_rows = []
    if not effect_frame.empty:
        for keys, subset in effect_frame.groupby(
            ["dataset", "checkpoint_epoch", "task"], sort=True, observed=True
        ):
            paired_rows.append(
                {
                    "dataset": keys[0],
                    "checkpoint_epoch": int(keys[1]),
                    "task": keys[2],
                    "image_count": int(len(subset)),
                    **paired_effects(subset["paired_median_difference"]),
                }
            )

    identity = tokens[tokens["substitution_mode"] == "identity"]
    identity_median_abs = (
        float(identity["c_self_raw"].abs().median()) if not identity.empty else np.nan
    )
    identity_max_abs = (
        float(identity["c_self_raw"].abs().max()) if not identity.empty else np.nan
    )

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    aggregate = {
        "full_run": bool(full_run),
        "sample_count": int(tokens["image_id"].nunique()),
        "token_row_count": int(len(tokens)),
        "checkpoint_epochs": sorted(int(value) for value in tokens["checkpoint_epoch"].unique()),
        "datasets": sorted(str(value) for value in tokens["dataset"].unique()),
        "identity_median_absolute_response": identity_median_abs,
        "identity_max_absolute_response": identity_max_abs,
    }

    if evaluate_verdict:
        verdict = fixed_verdict(
            response_frame=response_frame,
            incremental_frame=incremental_frame,
            bootstrap_frame=bootstrap_frame,
            substitution_frame=pd.DataFrame(substitution_rows),
            group_frame=pd.DataFrame(group_rows),
            spill_frame=pd.DataFrame(spill_rows),
            area_frame=pd.DataFrame(area_rows),
            cross_frame=pd.DataFrame(cross_rows),
            aggregate=aggregate,
            full_run=full_run,
        )
    else:
        verdict = {"verdict": "DEFERRED_TO_COMBINED_UNITS"}
    return {
        "aggregate": aggregate,
        "verdict": verdict,
        "response_metrics": response_frame,
        "incremental_metrics": incremental_frame,
        "bootstrap_intervals": bootstrap_frame,
        "cross_dataset_transfer": pd.DataFrame(cross_rows),
        "substitution_comparison": pd.DataFrame(substitution_rows),
        "group_robustness": pd.DataFrame(group_rows),
        "area_quartile_summary": pd.DataFrame(area_rows),
        "spillover_summary": pd.DataFrame(spill_rows),
        "feature_ood_summary": pd.DataFrame(ood_rows),
        "paired_effects": pd.DataFrame(paired_rows),
    }


def fixed_verdict(
    *,
    response_frame: pd.DataFrame,
    incremental_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    substitution_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    spill_frame: pd.DataFrame,
    area_frame: pd.DataFrame,
    cross_frame: pd.DataFrame,
    aggregate: dict,
    full_run: bool,
) -> dict:
    if not full_run:
        return {
            "verdict": "NOT_EVALUATED_SANITY_ONLY",
            "reason": "max_samples is not -1; PASS/FAIL criteria were not evaluated",
        }
    checks: Dict[str, bool] = {}
    primary_response = response_frame[
        (response_frame["substitution_mode"] == "weighted_bg")
        & (response_frame["group_size"] == 4)
        & (response_frame["response"] == "c_local_centered")
        & (response_frame["dataset"].isin(["TR-CAMO", "TR-COD10K"]))
    ]
    primary_increment = incremental_frame[
        (incremental_frame["substitution_mode"] == "weighted_bg")
        & (incremental_frame["group_size"] == 4)
        & (incremental_frame["dataset"].isin(["TR-CAMO", "TR-COD10K"]))
    ]
    primary_bootstrap = bootstrap_frame[
        (bootstrap_frame["substitution_mode"] == "weighted_bg")
        & (bootstrap_frame["group_size"] == 4)
        & (bootstrap_frame["dataset"].isin(["TR-CAMO", "TR-COD10K"]))
    ]
    for task, label in (("tp_fp", "teacher_fg"), ("fn_tn", "teacher_bg")):
        per_dataset = []
        for dataset in ("TR-CAMO", "TR-COD10K"):
            response = primary_response[
                (primary_response["dataset"] == dataset) & (primary_response["task"] == task)
            ]
            increment = primary_increment[
                (primary_increment["dataset"] == dataset) & (primary_increment["task"] == task)
            ]
            boot = primary_bootstrap[
                (primary_bootstrap["dataset"] == dataset)
                & (primary_bootstrap["task"] == task)
                & (primary_bootstrap["quantity"] == "incremental_model")
            ]
            class_boot = primary_bootstrap[
                (primary_bootstrap["dataset"] == dataset)
                & (primary_bootstrap["task"] == task)
                & (primary_bootstrap["quantity"] == "class_median_difference")
            ]
            merged = response.merge(
                increment,
                on=["dataset", "checkpoint_epoch", "substitution_mode", "group_size", "task"],
            ).merge(
                boot[
                    [
                        "dataset",
                        "checkpoint_epoch",
                        "substitution_mode",
                        "group_size",
                        "task",
                        "delta_roc_auc_ci_low",
                        "delta_pr_auc_ci_low",
                    ]
                ],
                on=["dataset", "checkpoint_epoch", "substitution_mode", "group_size", "task"],
            ).merge(
                class_boot[
                    [
                        "dataset",
                        "checkpoint_epoch",
                        "substitution_mode",
                        "group_size",
                        "task",
                        "ci_low",
                    ]
                ].rename(columns={"ci_low": "class_difference_ci_low"}),
                on=["dataset", "checkpoint_epoch", "substitution_mode", "group_size", "task"],
            )
            passed = merged[
                (merged["roc_auc"] >= 0.60)
                & (merged["delta_roc_auc"] >= 0.03)
                & (merged["delta_pr_auc"] >= 0.02)
                & (merged["delta_roc_auc_ci_low"] > 0.0)
                & (merged["delta_pr_auc_ci_low"] > 0.0)
                & (merged["class_difference_ci_low"] > 0.0)
            ]
            per_dataset.append(int(passed["checkpoint_epoch"].nunique()) >= 2)
        checks[label] = bool(all(per_dataset))

    checks["weighted_better_than_random"] = bool(
        not substitution_frame.empty
        and (substitution_frame["weighted_minus_random_auc"] >= 0.02).all()
        and (substitution_frame["bootstrap_ci_low"] > 0.0).all()
    )
    checks["group_robustness"] = bool(
        not group_frame.empty
        and group_frame["response_direction_consistent"].all()
        and (group_frame["absolute_auc_difference"] <= 0.03).all()
    )
    weighted_spill = spill_frame[
        (spill_frame["substitution_mode"] == "weighted_bg")
        & (spill_frame["group_size"] == 4)
    ]
    checks["spillover"] = bool(
        not weighted_spill.empty
        and (weighted_spill["spill_ratio_median"] <= 0.50).all()
        and (weighted_spill["spill_ratio_p90"] <= 1.00).all()
    )
    gt_area = area_frame[area_frame["area_partition"] == "gt_area_quartile"]
    checks["area_robustness"] = bool(
        not gt_area.empty
        and (gt_area.groupby(["dataset", "checkpoint_epoch", "task"])["roc_auc"].apply(lambda x: int((x > 0.57).sum()) >= 3)).all()
        and (gt_area["median_difference"] > 0.0).all()
    )
    checks["cross_dataset"] = bool(
        not cross_frame.empty
        and (cross_frame["delta_roc_auc"] >= 0.02).all()
    )
    checks["identity"] = bool(
        aggregate["identity_median_absolute_response"] < 1e-6
        and aggregate["identity_max_absolute_response"] < 1e-4
    )
    passed = bool(all(checks.values()))
    return {
        "verdict": (
            "BISA-v0 background intervention sensitivity: PASS_FOR_FURTHER_STUDY"
            if passed
            else "BISA-v0 background intervention sensitivity: FAIL"
        ),
        "checks": checks,
        "pass_scope": (
            "Signal is worth further study only; it is not a causal or routing-validity claim."
            if passed
            else None
        ),
    }


def build_statistical_outputs_from_parquet(
    token_path: str | Path,
    images: pd.DataFrame,
    *,
    seed: int,
    bootstrap_replicates: int = 10000,
    robustness_reference: str | Path | None = None,
) -> Dict[str, pd.DataFrame | dict]:
    """Run the fixed analysis one intervention unit at a time.

    A complete BISA run contains tens of millions of token rows.  Loading all
    checkpoints and substitution modes simultaneously would make the offline
    tool unusable.  Parquet predicate pushdown bounds peak memory to one
    ``checkpoint × substitution × group-size`` unit while preserving every
    token for the raw-distribution statistics.
    """
    token_path = Path(token_path)
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    if images.empty:
        raise RuntimeError("Image metrics are empty")

    frame_names = (
        "response_metrics",
        "incremental_metrics",
        "bootstrap_intervals",
        "cross_dataset_transfer",
        "group_robustness",
        "area_quartile_summary",
        "spillover_summary",
        "feature_ood_summary",
        "paired_effects",
    )
    collected: Dict[str, list[pd.DataFrame]] = {name: [] for name in frame_names}
    identity_medians = []
    identity_maxima = []
    units = (
        images[["checkpoint_epoch", "substitution_mode", "group_size"]]
        .drop_duplicates()
        .sort_values(["checkpoint_epoch", "substitution_mode", "group_size"])
    )
    for unit in units.itertuples(index=False):
        epoch = int(unit.checkpoint_epoch)
        mode = str(unit.substitution_mode)
        group_size = int(unit.group_size)
        token_unit = pd.read_parquet(
            token_path,
            filters=[
                ("checkpoint_epoch", "==", epoch),
                ("substitution_mode", "==", mode),
                ("group_size", "==", group_size),
            ],
        )
        if token_unit.empty:
            raise RuntimeError(
                f"Parquet predicate returned no rows for {epoch}/{mode}/g{group_size}"
            )
        image_unit = images[
            (images["checkpoint_epoch"] == epoch)
            & (images["substitution_mode"] == mode)
            & (images["group_size"] == group_size)
        ]
        result = build_statistical_outputs(
            token_unit,
            image_unit,
            seed=seed,
            full_run=True,
            bootstrap_replicates=bootstrap_replicates,
            robustness_reference=robustness_reference,
            evaluate_verdict=False,
        )
        for name in frame_names:
            value = result[name]
            if isinstance(value, pd.DataFrame) and not value.empty:
                collected[name].append(value)
        if mode == "identity":
            identity_medians.append(
                float(result["aggregate"]["identity_median_absolute_response"])
            )
            identity_maxima.append(
                float(result["aggregate"]["identity_max_absolute_response"])
            )
        del token_unit

    frames = {
        name: (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame()
        )
        for name, parts in collected.items()
    }
    response = frames["response_metrics"]
    substitution_rows = []
    comparison_bootstrap_rows = []
    comparison_cache: Dict[tuple[int, str], pd.DataFrame] = {}
    primary = response[
        (response["response"] == "c_local_centered")
        & (response["group_size"] == 4)
    ]
    for keys, subset in primary.groupby(
        ["dataset", "checkpoint_epoch", "task"], sort=True, observed=True
    ):
        indexed = subset.set_index("substitution_mode")
        if "weighted_bg" not in indexed.index or "random_bg" not in indexed.index:
            continue
        weighted_auc = float(indexed.loc["weighted_bg", "roc_auc"])
        random_auc = float(indexed.loc["random_bg", "roc_auc"])
        cache_key = (int(keys[1]), str(keys[0]))
        comparison_tokens = comparison_cache.get(cache_key)
        if comparison_tokens is None:
            filters = [
                ("checkpoint_epoch", "==", int(keys[1])),
                ("group_size", "==", 4),
                (
                    "substitution_mode",
                    "in",
                    ["weighted_bg", "random_bg"],
                ),
            ]
            if str(keys[0]) != "Combined":
                filters.append(("dataset", "==", str(keys[0])))
            comparison_tokens = pd.read_parquet(
                token_path,
                columns=[
                    "dataset",
                    "image_id",
                    "substitution_mode",
                    "teacher_error_type",
                    "c_local_centered",
                ],
                filters=filters,
            )
            comparison_cache[cache_key] = comparison_tokens
        comparison_ci = image_bootstrap_mode_auc_difference(
            comparison_tokens,
            str(keys[2]),
            "c_local_centered",
            seed=_stable_seed(
                seed, f"weighted-random:{keys[0]}:{keys[1]}:{keys[2]}"
            ),
            replicates=int(bootstrap_replicates),
        )
        substitution_rows.append(
            {
                "dataset": keys[0],
                "checkpoint_epoch": int(keys[1]),
                "task": keys[2],
                "weighted_bg_roc_auc": weighted_auc,
                "random_bg_roc_auc": random_auc,
                "weighted_minus_random_auc": weighted_auc - random_auc,
                "paired_image_auc_difference": comparison_ci["estimate"],
                "bootstrap_ci_low": comparison_ci["ci_low"],
                "bootstrap_ci_high": comparison_ci["ci_high"],
                "bootstrap_image_count": comparison_ci[
                    "bootstrap_image_count"
                ],
            }
        )
        comparison_bootstrap_rows.append(
            {
                "dataset": keys[0],
                "checkpoint_epoch": int(keys[1]),
                "substitution_mode": "weighted_bg_vs_random_bg",
                "group_size": 4,
                "task": keys[2],
                "quantity": "weighted_minus_random_auc",
                "response": "c_local_centered",
                **comparison_ci,
            }
        )
    substitution = pd.DataFrame(substitution_rows)
    if comparison_bootstrap_rows:
        frames["bootstrap_intervals"] = pd.concat(
            [
                frames["bootstrap_intervals"],
                pd.DataFrame(comparison_bootstrap_rows),
            ],
            ignore_index=True,
        )

    try:
        import pyarrow.parquet as pq

        token_row_count = int(pq.ParquetFile(token_path).metadata.num_rows)
    except Exception:
        token_row_count = -1
    aggregate = {
        "full_run": True,
        "sample_count": int(images["image_id"].nunique()),
        "token_row_count": token_row_count,
        "checkpoint_epochs": sorted(
            int(value) for value in images["checkpoint_epoch"].unique()
        ),
        "datasets": sorted(str(value) for value in images["dataset"].unique()),
        # Every sample is hard-asserted against the thresholds before writing.
        # The maximum of per-unit medians is a conservative global bound.
        "identity_median_absolute_response": (
            float(max(identity_medians)) if identity_medians else np.nan
        ),
        "identity_max_absolute_response": (
            float(max(identity_maxima)) if identity_maxima else np.nan
        ),
        "streaming_statistics": True,
    }
    verdict = fixed_verdict(
        response_frame=frames["response_metrics"],
        incremental_frame=frames["incremental_metrics"],
        bootstrap_frame=frames["bootstrap_intervals"],
        substitution_frame=substitution,
        group_frame=frames["group_robustness"],
        spill_frame=frames["spillover_summary"],
        area_frame=frames["area_quartile_summary"],
        cross_frame=frames["cross_dataset_transfer"],
        aggregate=aggregate,
        full_run=True,
    )
    return {
        "aggregate": aggregate,
        "verdict": verdict,
        **frames,
        "substitution_comparison": substitution,
    }
