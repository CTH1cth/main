import argparse
import csv
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import convolve, distance_transform_edt as bwdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import Emeasure, Fmeasure, MAEmeasure, Smeasure, WeightedFmeasure, _EPS  # noqa: E402
from common.utils import ensure_dir, read_jsonl  # noqa: E402


SAL_METRIC_KEYS = ["S_m", "F_beta^w", "F_beta^m", "E_phi^m", "MAE"]
BINARY_METRIC_KEYS = SAL_METRIC_KEYS + ["IoU", "Dice", "Precision", "Recall", "pred_area"]
OBJECTIVE_KEYS = ["S_m", "F_beta^w", "F_beta^m", "E_phi^m", "IoU", "Dice"]
DATASET_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 0.951, 0.05)]
GLOBAL_THRESHOLDS = [round(v, 2) for v in np.arange(0.10, 0.901, 0.05)]


def parse_csv_arg(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def finite_float(value, default=float("nan")):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def mean(values):
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_gt(gt_path):
    path = Path(gt_path)
    if not path.exists():
        raise FileNotFoundError(f"GT not found: {path}")
    gt = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return (gt > 0.5).astype(np.float32)


def load_prob(prob_path):
    path = Path(prob_path)
    if not path.exists():
        raise FileNotFoundError(
            f"prob.npy not found: {path}. Please run tools/export_eval_predictions.py first."
        )
    prob = np.load(path).astype(np.float32)
    prob = np.squeeze(prob)
    if prob.ndim != 2:
        raise RuntimeError(f"Bad probability shape at {path}: {prob.shape}")
    if not np.isfinite(prob).all():
        raise RuntimeError(f"Probability map has NaN/Inf: {path}")
    return np.clip(prob, 0.0, 1.0)


def load_manifests(pred_roots, tags, datasets):
    manifests = {}
    expected_keys_by_dataset = {}
    for root, tag in zip(pred_roots, tags):
        root = Path(root).expanduser()
        manifest_path = root / "manifest.jsonl"
        rows = read_jsonl(manifest_path)
        by_dataset = defaultdict(dict)
        for row in rows:
            dataset = str(row.get("dataset", ""))
            if dataset not in datasets:
                continue
            row_tag = str(row.get("tag", tag))
            if row_tag and row_tag != tag:
                raise RuntimeError(
                    f"Tag mismatch in {manifest_path}: expected {tag}, got {row_tag} "
                    f"for {dataset}/{row.get('stem')}"
                )
            stem = str(row.get("stem", ""))
            if not stem:
                raise RuntimeError(f"Bad manifest row missing stem in {manifest_path}: {row}")
            prob_path = row.get("prob_path", "")
            gt_path = row.get("gt_path", "")
            if not prob_path:
                raise RuntimeError(
                    f"Manifest row missing prob_path for {tag} {dataset}/{stem}. "
                    "Please re-run tools/export_eval_predictions.py with --save-prob."
                )
            if not gt_path:
                raise RuntimeError(f"Manifest row missing gt_path for {tag} {dataset}/{stem}.")
            if stem in by_dataset[dataset]:
                raise RuntimeError(f"Duplicate sample in {manifest_path}: {dataset}/{stem}")
            by_dataset[dataset][stem] = row
        for dataset in datasets:
            if dataset not in by_dataset or not by_dataset[dataset]:
                raise RuntimeError(f"No predictions for dataset {dataset} in {manifest_path}")
            keys = set(by_dataset[dataset])
            if dataset not in expected_keys_by_dataset:
                expected_keys_by_dataset[dataset] = keys
            elif keys != expected_keys_by_dataset[dataset]:
                missing = sorted(expected_keys_by_dataset[dataset] - keys)[:10]
                extra = sorted(keys - expected_keys_by_dataset[dataset])[:10]
                raise RuntimeError(
                    f"Sample set mismatch for tag={tag}, dataset={dataset}. "
                    f"missing first 10={missing}, extra first 10={extra}"
                )
        manifests[tag] = by_dataset
    return manifests


def load_samples(manifests, tags, datasets):
    samples = {}
    for tag in tags:
        samples[tag] = {}
        for dataset in datasets:
            rows = []
            for stem in sorted(manifests[tag][dataset]):
                row = manifests[tag][dataset][stem]
                prob = load_prob(row["prob_path"])
                gt = load_gt(row["gt_path"])
                if prob.shape != gt.shape:
                    raise RuntimeError(
                        f"Shape mismatch for {tag} {dataset}/{stem}: "
                        f"prob={prob.shape}, gt={gt.shape}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "prob": prob,
                        "gt": gt,
                        "prob_path": str(row["prob_path"]),
                        "gt_path": str(row["gt_path"]),
                    }
                )
            samples[tag][dataset] = rows
            print(f"[CalibrationProbe] loaded {tag} | {dataset} | samples={len(rows)}", flush=True)
    return samples


def prepare_data_like_common(pred, gt):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.max() != gt.min():
        gt = (gt - gt.min()) / (gt.max() - gt.min())
    gt = gt > 0.5
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        pred = pred.astype(int)
    return pred.astype(np.float64), gt


WFM_KERNEL = WeightedFmeasure().matlab_style_gauss2d((7, 7), sigma=5)


def make_wfm_cache(gt):
    gt_bool = np.asarray(gt) > 0.5
    if np.all(~gt_bool):
        return {"all_bg": True}
    dst, idx = bwdist(gt_bool == 0, return_indices=True)
    return {"all_bg": False, "dst": dst, "idx": idx}


def cached_weighted_fmeasure(pred, gt, cache):
    pred, gt = prepare_data_like_common(pred=pred, gt=gt)
    if cache.get("all_bg", False) or np.all(~gt):
        return 0.0
    dst = cache["dst"]
    idx = cache["idx"]
    error = np.abs(pred - gt)
    et = np.copy(error)
    et[gt == 0] = et[idx[0][gt == 0], idx[1][gt == 0]]
    ea = convolve(et, weights=WFM_KERNEL, mode="constant", cval=0)
    min_error = np.where(gt & (ea < error), ea, error)
    b = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt))
    ew = min_error * b
    tpw = np.sum(gt) - np.sum(ew[gt == 1])
    fpw = np.sum(ew[gt == 0])
    recall = 1 - np.mean(ew[gt == 1])
    precision = tpw / (tpw + fpw + _EPS)
    return float(2 * recall * precision / (recall + precision + _EPS))


class FastCODAccumulator:
    """Same COD metric formulas as common.metrics.CODMetrics, with cached GT distance maps for WFM."""

    def __init__(self, compute_wfm=True):
        self.compute_wfm = bool(compute_wfm)
        self.mae = MAEmeasure()
        self.sm = Smeasure()
        self.em = Emeasure()
        self.fm = Fmeasure()
        self.weighted_fms = []

    def step(self, pred, gt, wfm_cache):
        self.em.step(pred=pred, gt=gt)
        self.sm.step(pred=pred, gt=gt)
        self.fm.step(pred=pred, gt=gt)
        self.mae.step(pred=pred, gt=gt)
        if self.compute_wfm:
            self.weighted_fms.append(cached_weighted_fmeasure(pred, gt, wfm_cache))

    def add_wfm(self, value):
        self.weighted_fms.append(float(value))

    def result(self):
        em = self.em.get_results()["em"]
        fm = self.fm.get_results()["fm"]
        return {
            "S_m": float(self.sm.get_results()["sm"]),
            "F_beta^w": float(np.mean(np.array(self.weighted_fms, dtype=np.float64)))
            if self.weighted_fms
            else float("nan"),
            "F_beta^m": float(fm["curve"].mean()),
            "E_phi^m": float(em["curve"].mean()),
            "MAE": float(self.mae.get_results()["mae"]),
        }


def cached_weighted_fmeasure_many_thresholds(prob, gt, thresholds, cache):
    gt_bool = np.asarray(gt) > 0.5
    if cache.get("all_bg", False) or np.all(~gt_bool):
        return {threshold: 0.0 for threshold in thresholds}
    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    pred_stack = (prob[None, :, :] >= thresholds_arr[:, None, None]).astype(np.float64)
    gt_float = gt_bool.astype(np.float64)
    error = np.abs(pred_stack - gt_float[None, :, :])
    et = np.copy(error)
    bg_mask = gt_bool == 0
    if np.any(bg_mask):
        idx = cache["idx"]
        et[:, bg_mask] = error[:, idx[0][bg_mask], idx[1][bg_mask]]
    ea = convolve(et, weights=WFM_KERNEL[None, :, :], mode="constant", cval=0)
    min_error = np.where(gt_bool[None, :, :] & (ea < error), ea, error)
    dst = cache["dst"]
    b = np.where(gt_bool == 0, 2 - np.exp(np.log(0.5) / 5 * dst), np.ones_like(gt_float))
    ew = min_error * b[None, :, :]
    fg = gt_bool
    bg = ~gt_bool
    tpw = np.sum(gt_float) - ew[:, fg].sum(axis=1)
    fpw = ew[:, bg].sum(axis=1) if np.any(bg) else np.zeros(len(thresholds_arr), dtype=np.float64)
    recall = 1 - ew[:, fg].mean(axis=1)
    precision = tpw / (tpw + fpw + _EPS)
    values = 2 * recall * precision / (recall + precision + _EPS)
    return {threshold: float(value) for threshold, value in zip(thresholds, values)}


def empty_binary_stats_state():
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "pred_area_values": [],
        "dice_values": [],
        "iou_values": [],
        "precision_values": [],
        "recall_values": [],
    }


def update_binary_stats_state(state, pred, gt):
    gt = gt > 0.5
    pred = pred > 0.5
    tp_i = float(np.logical_and(pred, gt).sum())
    fp_i = float(np.logical_and(pred, ~gt).sum())
    fn_i = float(np.logical_and(~pred, gt).sum())
    tn_i = float(np.logical_and(~pred, ~gt).sum())
    state["tp"] += tp_i
    state["fp"] += fp_i
    state["fn"] += fn_i
    state["tn"] += tn_i
    denom_iou = tp_i + fp_i + fn_i
    denom_dice = 2.0 * tp_i + fp_i + fn_i
    state["iou_values"].append(1.0 if denom_iou == 0.0 else tp_i / denom_iou)
    state["dice_values"].append(1.0 if denom_dice == 0.0 else 2.0 * tp_i / denom_dice)
    state["precision_values"].append(1.0 if (tp_i + fp_i) == 0.0 else tp_i / (tp_i + fp_i))
    state["recall_values"].append(1.0 if (tp_i + fn_i) == 0.0 else tp_i / (tp_i + fn_i))
    state["pred_area_values"].append(float(pred.mean()))


def finalize_binary_stats_state(state):
    return {
        "IoU": mean(state["iou_values"]),
        "Dice": mean(state["dice_values"]),
        "Precision": mean(state["precision_values"]),
        "Recall": mean(state["recall_values"]),
        "pred_area": mean(state["pred_area_values"]),
        "TP": state["tp"],
        "FP": state["fp"],
        "FN": state["fn"],
        "TN": state["tn"],
    }


def binary_stats_for_arrays(arrays, threshold):
    state = empty_binary_stats_state()
    for item in arrays:
        gt = item["gt"]
        pred = (item["prob"] >= float(threshold)).astype(np.float32)
        update_binary_stats_state(state, pred, gt)
    return finalize_binary_stats_state(state)


def evaluate_binary(arrays, threshold):
    acc = FastCODAccumulator()
    for item in arrays:
        wfm_cache = make_wfm_cache(item["gt"])
        pred = (item["prob"] >= float(threshold)).astype(np.float32)
        acc.step(pred, item["gt"], wfm_cache)
    row = acc.result()
    row.update(binary_stats_for_arrays(arrays, threshold))
    row["objective"] = objective_value(row)
    return row


def evaluate_probability(arrays):
    acc = FastCODAccumulator()
    for item in arrays:
        wfm_cache = make_wfm_cache(item["gt"])
        acc.step(item["prob"], item["gt"], wfm_cache)
    row = acc.result()
    row["objective"] = float("nan")
    return row


def evaluate_all_thresholds_for_arrays(arrays, thresholds):
    prob_acc = FastCODAccumulator()
    threshold_acc = {threshold: FastCODAccumulator(compute_wfm=False) for threshold in thresholds}
    threshold_stats = {threshold: empty_binary_stats_state() for threshold in thresholds}
    for item in arrays:
        prob = item["prob"].astype(np.float32)
        gt = item["gt"].astype(np.float32)
        wfm_cache = make_wfm_cache(gt)
        prob_acc.step(prob, gt, wfm_cache)
        wfms = cached_weighted_fmeasure_many_thresholds(prob, gt, thresholds, wfm_cache)
        for threshold in thresholds:
            pred = (prob >= float(threshold)).astype(np.float32)
            threshold_acc[threshold].step(pred, gt, wfm_cache)
            threshold_acc[threshold].add_wfm(wfms[threshold])
            update_binary_stats_state(threshold_stats[threshold], pred, gt)
    prob_row = prob_acc.result()
    prob_row["objective"] = float("nan")
    threshold_results = {}
    for threshold in thresholds:
        row = threshold_acc[threshold].result()
        row.update(finalize_binary_stats_state(threshold_stats[threshold]))
        row["objective"] = objective_value(row)
        threshold_results[threshold] = row
    return prob_row, threshold_results


def objective_value(row):
    values = [finite_float(row.get(key)) for key in OBJECTIVE_KEYS]
    values.append(1.0 - finite_float(row.get("MAE")))
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def compute_all_metrics(samples, tags, datasets):
    prob_vs_binary_rows = []
    threshold_rows = []
    baseline = {}
    prob_rows_by_key = {}
    sweep_by_key = defaultdict(list)
    for tag in tags:
        for dataset in datasets:
            arrays = samples[tag][dataset]
            prob, threshold_results = evaluate_all_thresholds_for_arrays(arrays, DATASET_THRESHOLDS)
            binary = threshold_results[0.5]
            baseline[(tag, dataset)] = binary
            row = {"run_tag": tag, "dataset": dataset, "mode": "binary_0.5", "threshold": 0.5}
            row.update(binary)
            prob_vs_binary_rows.append(row)

            prob_rows_by_key[(tag, dataset)] = prob
            row = {"run_tag": tag, "dataset": dataset, "mode": "probability_map", "threshold": ""}
            row.update(prob)
            for key in ["IoU", "Dice", "Precision", "Recall", "pred_area", "TP", "FP", "FN", "TN"]:
                row[key] = ""
            prob_vs_binary_rows.append(row)

            for threshold in DATASET_THRESHOLDS:
                metrics = threshold_results[threshold]
                base = baseline[(tag, dataset)]
                out = {"run_tag": tag, "dataset": dataset, "threshold": threshold}
                out.update(metrics)
                for key in BINARY_METRIC_KEYS:
                    out[f"{key}_gain_vs_0.5"] = metrics[key] - base[key]
                threshold_rows.append(out)
                sweep_by_key[(tag, dataset)].append(out)
            print(f"[CalibrationProbe] evaluated {tag} | {dataset}", flush=True)
    return prob_vs_binary_rows, threshold_rows, baseline, prob_rows_by_key, sweep_by_key


def better_metric(metric_name):
    return metric_name != "MAE"


def best_for_metric(rows, metric_name):
    if better_metric(metric_name):
        return max(rows, key=lambda r: finite_float(r.get(metric_name), -1e9))
    return min(rows, key=lambda r: finite_float(r.get(metric_name), 1e9))


def build_dataset_best_rows(sweep_by_key, baseline):
    rows = []
    best_metric_names = ["IoU", "F_beta^w", "S_m", "E_phi^m", "MAE"]
    for (tag, dataset), sweep_rows in sorted(sweep_by_key.items()):
        base = baseline[(tag, dataset)]
        row = {"run_tag": tag, "dataset": dataset}
        for metric_name in best_metric_names:
            best = best_for_metric(sweep_rows, metric_name)
            suffix = {
                "IoU": "IoU",
                "F_beta^w": "Fw",
                "S_m": "S",
                "E_phi^m": "E",
                "MAE": "MAE",
            }[metric_name]
            row[f"best_threshold_by_{suffix}"] = best["threshold"]
            row[f"best_{suffix}"] = best[metric_name]
            gain = base[metric_name] - best[metric_name] if metric_name == "MAE" else best[metric_name] - base[metric_name]
            row[f"{suffix}_gain_vs_0.5"] = gain
        row.update({f"baseline_{key}": base[key] for key in BINARY_METRIC_KEYS})
        rows.append(row)
    return rows


def build_global_rows(threshold_rows, tags, datasets):
    rows = []
    by_tag_threshold = defaultdict(dict)
    for row in threshold_rows:
        threshold = round(float(row["threshold"]), 2)
        if threshold in GLOBAL_THRESHOLDS:
            by_tag_threshold[(row["run_tag"], threshold)][row["dataset"]] = row

    ranks = defaultdict(dict)
    for tag in tags:
        for dataset in datasets:
            candidates = [
                by_tag_threshold[(tag, threshold)][dataset]
                for threshold in GLOBAL_THRESHOLDS
                if dataset in by_tag_threshold[(tag, threshold)]
            ]
            ranked = sorted(candidates, key=lambda r: finite_float(r["objective"], -1e9), reverse=True)
            for rank, row in enumerate(ranked, 1):
                ranks[(tag, dataset)][round(float(row["threshold"]), 2)] = rank

    for tag in tags:
        for threshold in GLOBAL_THRESHOLDS:
            dataset_rows = by_tag_threshold.get((tag, threshold), {})
            if set(dataset_rows) != set(datasets):
                missing = sorted(set(datasets) - set(dataset_rows))
                raise RuntimeError(f"Missing global threshold rows for {tag} threshold={threshold}: {missing}")
            out = {
                "run_tag": tag,
                "threshold": threshold,
                "avg_objective": mean([dataset_rows[d]["objective"] for d in datasets]),
                "mean_rank": mean([ranks[(tag, d)][threshold] for d in datasets]),
            }
            for key in BINARY_METRIC_KEYS:
                out[f"avg_{key}"] = mean([dataset_rows[d][key] for d in datasets])
            for dataset in datasets:
                src = dataset_rows[dataset]
                out[f"{dataset}_S_m"] = src["S_m"]
                out[f"{dataset}_F_beta^w"] = src["F_beta^w"]
                out[f"{dataset}_E_phi^m"] = src["E_phi^m"]
                out[f"{dataset}_MAE"] = src["MAE"]
                out[f"{dataset}_IoU"] = src["IoU"]
            rows.append(out)

    best_avg = {}
    best_rank = {}
    for tag in tags:
        tag_rows = [r for r in rows if r["run_tag"] == tag]
        best_avg[tag] = max(tag_rows, key=lambda r: finite_float(r["avg_objective"], -1e9))
        best_rank[tag] = min(tag_rows, key=lambda r: finite_float(r["mean_rank"], 1e9))
        for row in tag_rows:
            row["is_best_global_by_average_metric"] = row is best_avg[tag]
            row["is_best_global_by_mean_rank"] = row is best_rank[tag]
    return rows, best_avg, best_rank


def choose_global_row(best_avg, best_rank, tag):
    if tag in best_avg:
        return best_avg[tag]
    if tag in best_rank:
        return best_rank[tag]
    return None


def calibration_conclusion(tags, datasets, baseline, prob_rows_by_key, dataset_best_rows, best_avg, global_rows):
    primary_tag = "csd35" if "csd35" in tags else tags[-1]
    if ("CHAMELEON" not in datasets) or (primary_tag, "CHAMELEON") not in baseline:
        return {
            "primary_tag": primary_tag,
            "conclusion": "Calibration probe completed, but CHAMELEON is not available for automatic decision.",
            "reason": "CHAMELEON missing from requested datasets.",
        }
    cham_base = baseline[(primary_tag, "CHAMELEON")]
    cham_prob = prob_rows_by_key[(primary_tag, "CHAMELEON")]
    global_row = choose_global_row(best_avg, {}, primary_tag)
    if global_row is None:
        return {"primary_tag": primary_tag, "conclusion": "Calibration probe completed.", "reason": "No global row."}

    def global_metric(dataset, key):
        return finite_float(global_row.get(f"{dataset}_{key}"))

    cham_global_gains = {
        "S_m": global_metric("CHAMELEON", "S_m") - cham_base["S_m"],
        "F_beta^w": global_metric("CHAMELEON", "F_beta^w") - cham_base["F_beta^w"],
        "E_phi^m": global_metric("CHAMELEON", "E_phi^m") - cham_base["E_phi^m"],
        "MAE": cham_base["MAE"] - global_metric("CHAMELEON", "MAE"),
    }
    obvious_cham_gain = sum(1 for key in ["S_m", "F_beta^w", "E_phi^m"] if cham_global_gains[key] >= 0.005) >= 2
    cod_ok = True
    cod_reason = "TE-COD10K not requested."
    if "TE-COD10K" in datasets:
        cod_e = global_metric("TE-COD10K", "E_phi^m")
        cod_mae = global_metric("TE-COD10K", "MAE")
        cod_ok = cod_e >= 0.824 and cod_mae <= 0.058
        cod_reason = f"TE-COD10K E={cod_e:.6f}, MAE={cod_mae:.6f}"
    camo_ok = True
    camo_reason = "TE-CAMO not requested."
    if "TE-CAMO" in datasets:
        camo_s = global_metric("TE-CAMO", "S_m")
        camo_base_s = baseline[(primary_tag, "TE-CAMO")]["S_m"]
        camo_ok = camo_s >= camo_base_s - 0.003
        camo_reason = f"TE-CAMO S={camo_s:.6f}, baseline S={camo_base_s:.6f}"

    best_cham = next(
        (r for r in dataset_best_rows if r["run_tag"] == primary_tag and r["dataset"] == "CHAMELEON"),
        None,
    )
    best_cham_iou_gain = finite_float(best_cham.get("IoU_gain_vs_0.5")) if best_cham else float("nan")
    prob_improved = (
        cham_prob["S_m"] >= cham_base["S_m"] + 0.005
        or cham_prob["F_beta^w"] >= cham_base["F_beta^w"] + 0.005
        or cham_prob["E_phi^m"] >= cham_base["E_phi^m"] + 0.005
    )

    guardrails_available = "TE-COD10K" in datasets and "TE-CAMO" in datasets
    if obvious_cham_gain and not guardrails_available:
        conclusion = "Calibration probe completed on a subset; global safety cannot be decided without TE-CAMO and TE-COD10K."
        reason = "CHAMELEON improves, but the required COD10K/CAMO guardrail datasets were not both requested."
    elif obvious_cham_gain and cod_ok and camo_ok:
        conclusion = "Calibration direction is promising."
        reason = "CHAMELEON gains under a shared global threshold without violating COD10K/CAMO guardrails."
    elif obvious_cham_gain and not cod_ok:
        conclusion = "Calibration mainly creates CHAMELEON-COD10K trade-off. Not recommended as main solution."
        reason = "CHAMELEON improves, but COD10K guardrail is violated."
    elif (math.isnan(best_cham_iou_gain) or best_cham_iou_gain < 0.01) or not prob_improved:
        conclusion = (
            "Calibration is not the main bottleneck. Move to multi-level semantic feature or HR boundary refinement."
        )
        reason = "CHAMELEON best-threshold gain is small or probability-map metrics do not show enough headroom."
    else:
        conclusion = "Calibration appears secondary and should be treated as a diagnostic trade-off."
        reason = "There is some threshold sensitivity, but it is not clearly safe as a global policy."

    return {
        "primary_tag": primary_tag,
        "best_global_threshold_by_average_metric": global_row["threshold"],
        "chameleon_global_gains": cham_global_gains,
        "cod10k_guardrail": cod_reason,
        "camo_guardrail": camo_reason,
        "best_chameleon_iou_gain": best_cham_iou_gain,
        "probability_map_improved": bool(prob_improved),
        "conclusion": conclusion,
        "reason": reason,
    }


def format_metric_line(row):
    return (
        f"S={finite_float(row.get('S_m')):.4f} "
        f"Fw={finite_float(row.get('F_beta^w')):.4f} "
        f"Fm={finite_float(row.get('F_beta^m')):.4f} "
        f"E={finite_float(row.get('E_phi^m')):.4f} "
        f"MAE={finite_float(row.get('MAE')):.4f}"
    )


def write_summary_md(path, tags, datasets, prob_vs_binary_rows, dataset_best_rows, global_rows, best_avg, best_rank, decision):
    lines = []
    lines.append("# Calibration Probe v1")
    lines.append("")
    lines.append("本报告只用于诊断概率校准和阈值敏感性；不修改正式 eval 逻辑，也不作为主表结果。")
    lines.append("")
    lines.append("## 0.5 Threshold Metrics")
    for tag in tags:
        lines.append(f"### {tag}")
        for dataset in datasets:
            row = next(r for r in prob_vs_binary_rows if r["run_tag"] == tag and r["dataset"] == dataset and r["mode"] == "binary_0.5")
            lines.append(
                f"- {dataset}: {format_metric_line(row)} "
                f"IoU={finite_float(row.get('IoU')):.4f} Dice={finite_float(row.get('Dice')):.4f} "
                f"P={finite_float(row.get('Precision')):.4f} R={finite_float(row.get('Recall')):.4f} "
                f"area={finite_float(row.get('pred_area')):.4f}"
            )
    lines.append("")
    lines.append("## Probability Map Metrics")
    for tag in tags:
        lines.append(f"### {tag}")
        for dataset in datasets:
            row = next(r for r in prob_vs_binary_rows if r["run_tag"] == tag and r["dataset"] == dataset and r["mode"] == "probability_map")
            lines.append(f"- {dataset}: {format_metric_line(row)}")
    lines.append("")
    lines.append("## CHAMELEON Best Threshold")
    for tag in tags:
        row = next((r for r in dataset_best_rows if r["run_tag"] == tag and r["dataset"] == "CHAMELEON"), None)
        if row:
            lines.append(
                f"- {tag}: best IoU threshold={row['best_threshold_by_IoU']} "
                f"IoU gain={finite_float(row['IoU_gain_vs_0.5']):.4f}; "
                f"best Fw threshold={row['best_threshold_by_Fw']} "
                f"Fw gain={finite_float(row['Fw_gain_vs_0.5']):.4f}; "
                f"best S threshold={row['best_threshold_by_S']} "
                f"S gain={finite_float(row['S_gain_vs_0.5']):.4f}"
            )
    lines.append("")
    lines.append("## Global Threshold")
    for tag in tags:
        avg_row = best_avg[tag]
        rank_row = best_rank[tag]
        lines.append(
            f"- {tag}: best_by_average={avg_row['threshold']} "
            f"(avg_objective={finite_float(avg_row['avg_objective']):.4f}); "
            f"best_by_mean_rank={rank_row['threshold']} "
            f"(mean_rank={finite_float(rank_row['mean_rank']):.2f})"
        )
        for dataset in datasets:
            lines.append(
                f"  - {dataset} @ {avg_row['threshold']}: "
                f"S={finite_float(avg_row.get(f'{dataset}_S_m')):.4f} "
                f"Fw={finite_float(avg_row.get(f'{dataset}_F_beta^w')):.4f} "
                f"E={finite_float(avg_row.get(f'{dataset}_E_phi^m')):.4f} "
                f"MAE={finite_float(avg_row.get(f'{dataset}_MAE')):.4f}"
            )
    lines.append("")
    lines.append("## Automatic Conclusion")
    lines.append(f"- Primary tag: `{decision['primary_tag']}`")
    lines.append(f"- Conclusion: **{decision['conclusion']}**")
    lines.append(f"- Reason: {decision['reason']}")
    if "chameleon_global_gains" in decision:
        gains = decision["chameleon_global_gains"]
        lines.append(
            "- CHAMELEON global-threshold gains: "
            + ", ".join(f"{k}={v:.4f}" for k, v in gains.items())
        )
    lines.append(f"- {decision.get('cod10k_guardrail', '')}")
    lines.append(f"- {decision.get('camo_guardrail', '')}")
    lines.append("")
    lines.append("## Files")
    lines.append("- `threshold_sweep_all.csv`: all run/dataset/threshold metrics.")
    lines.append("- `dataset_best_thresholds.csv`: per dataset best thresholds and gain vs 0.5.")
    lines.append("- `global_threshold_results.csv`: shared-threshold trade-off table.")
    lines.append("- `prob_vs_binary_metrics.csv`: probability-map vs fixed 0.5 metrics.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_lines(path, title, x_values, series, y_label):
    width, height = 1100, 720
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_l, 20), title, fill="black")
    draw.text((margin_l, height - 35), "threshold", fill="black")
    draw.text((10, margin_t), y_label, fill="black")

    vals = [v for _, ys in series for v in ys if not math.isnan(float(v))]
    y_min = min(vals) if vals else 0.0
    y_max = max(vals) if vals else 1.0
    if y_max == y_min:
        y_max += 1.0
        y_min -= 1.0
    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    x_min, x_max = min(x_values), max(x_values)

    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.rectangle((x0, y1, x1, y0), outline=(180, 180, 180))
    for i in range(6):
        y = y0 - (y0 - y1) * i / 5.0
        value = y_min + (y_max - y_min) * i / 5.0
        draw.line((x0, y, x1, y), fill=(235, 235, 235))
        draw.text((20, y - 7), f"{value:.3f}", fill=(80, 80, 80))
    for t in x_values:
        x = x0 + (x1 - x0) * (t - x_min) / (x_max - x_min)
        draw.line((x, y0, x, y0 + 5), fill=(100, 100, 100))
        if int(round(t * 100)) % 20 == 0:
            draw.text((x - 12, y0 + 10), f"{t:.2f}", fill=(80, 80, 80))

    colors = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
        (140, 86, 75),
        (127, 127, 127),
    ]
    for idx, (name, ys) in enumerate(series):
        color = colors[idx % len(colors)]
        points = []
        for t, v in zip(x_values, ys):
            if math.isnan(float(v)):
                continue
            x = x0 + (x1 - x0) * (t - x_min) / (x_max - x_min)
            y = y0 - (y0 - y1) * (float(v) - y_min) / (y_max - y_min)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        legend_y = margin_t + idx * 22
        draw.rectangle((width - 260, legend_y, width - 245, legend_y + 12), fill=color)
        draw.text((width - 238, legend_y - 2), name, fill="black")

    ensure_dir(Path(path).parent)
    img.save(path)


def make_plots(out, tags, datasets, threshold_rows, global_rows):
    plot_dir = out / "plots"
    for dataset in datasets:
        series = []
        for tag in tags:
            rows = sorted(
                [r for r in threshold_rows if r["dataset"] == dataset and r["run_tag"] == tag],
                key=lambda r: float(r["threshold"]),
            )
            series.append((tag, [r["F_beta^w"] for r in rows]))
        plot_lines(
            plot_dir / f"{dataset}_threshold_curve.png",
            f"{dataset} threshold sweep (F_beta^w)",
            DATASET_THRESHOLDS,
            series,
            "F_beta^w",
        )

    series = []
    for tag in tags:
        rows = sorted([r for r in global_rows if r["run_tag"] == tag], key=lambda r: float(r["threshold"]))
        series.append((tag, [r["avg_objective"] for r in rows]))
    plot_lines(
        plot_dir / "global_threshold_tradeoff.png",
        "Global threshold trade-off",
        GLOBAL_THRESHOLDS,
        series,
        "avg objective",
    )


def write_pack(out):
    out = Path(out)
    zip_path = out / "calibration_probe_pack.zip"
    include_names = {
        "summary.md",
        "summary.json",
        "threshold_sweep_all.csv",
        "dataset_best_thresholds.csv",
        "global_threshold_results.csv",
        "prob_vs_binary_metrics.csv",
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(include_names):
            path = out / name
            if path.exists():
                zf.write(path, arcname=name)
        plot_dir = out / "plots"
        if plot_dir.exists():
            for path in sorted(plot_dir.glob("*.png")):
                zf.write(path, arcname=str(path.relative_to(out)))
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Calibration and threshold probe for exported predictions.")
    parser.add_argument("--pred-roots", required=True)
    parser.add_argument("--tags", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pred_roots = parse_csv_arg(args.pred_roots)
    tags = parse_csv_arg(args.tags)
    datasets = parse_csv_arg(args.datasets)
    if len(pred_roots) != len(tags):
        raise ValueError(f"--pred-roots and --tags length mismatch: {len(pred_roots)} != {len(tags)}")
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset.")

    out = Path(args.out).expanduser()
    ensure_dir(out)
    manifests = load_manifests(pred_roots, tags, datasets)
    samples = load_samples(manifests, tags, datasets)
    prob_vs_binary_rows, threshold_rows, baseline, prob_rows_by_key, sweep_by_key = compute_all_metrics(
        samples, tags, datasets
    )
    dataset_best_rows = build_dataset_best_rows(sweep_by_key, baseline)
    global_rows, best_avg, best_rank = build_global_rows(threshold_rows, tags, datasets)
    decision = calibration_conclusion(
        tags, datasets, baseline, prob_rows_by_key, dataset_best_rows, best_avg, global_rows
    )

    all_fields = sorted({key for row in threshold_rows for key in row})
    write_csv(out / "threshold_sweep_all.csv", threshold_rows, all_fields)
    write_csv(out / "dataset_best_thresholds.csv", dataset_best_rows, sorted({key for row in dataset_best_rows for key in row}))
    write_csv(out / "global_threshold_results.csv", global_rows, sorted({key for row in global_rows for key in row}))
    write_csv(out / "prob_vs_binary_metrics.csv", prob_vs_binary_rows, sorted({key for row in prob_vs_binary_rows for key in row}))

    summary = {
        "pred_roots": pred_roots,
        "tags": tags,
        "datasets": datasets,
        "thresholds": {
            "dataset_thresholds": DATASET_THRESHOLDS,
            "global_thresholds": GLOBAL_THRESHOLDS,
        },
        "best_global_threshold_by_average_metric": {
            tag: json_safe(best_avg[tag]) for tag in tags
        },
        "best_global_threshold_by_mean_rank": {
            tag: json_safe(best_rank[tag]) for tag in tags
        },
        "automatic_decision": decision,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    write_summary_md(out / "summary.md", tags, datasets, prob_vs_binary_rows, dataset_best_rows, global_rows, best_avg, best_rank, decision)
    make_plots(out, tags, datasets, threshold_rows, global_rows)
    zip_path = write_pack(out)
    print(f"[CalibrationProbe] summary = {out / 'summary.md'}", flush=True)
    print(f"[CalibrationProbe] pack = {zip_path}", flush=True)
    print(f"[CalibrationProbe] conclusion = {decision['conclusion']}", flush=True)


if __name__ == "__main__":
    main()
