#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score

MAIN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_analysis.common import (  # noqa: E402
    distribution_stats, load_settings, probability_hist, read_jsonl,
    validate_output, write_csv, write_json,
)

PROTOCOLS = (("allpatch_0.5", "main"), ("core_0.2_0.8", "core"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _metric(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.uint8); score = np.asarray(score, dtype=np.float64)
    if np.unique(y).size != 2:
        raise ValueError("AUROC/AP require both foreground and background")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _hist_diagnostics(first: np.ndarray, second: np.ndarray, bins: np.ndarray) -> dict:
    centers = (bins[:-1] + bins[1:]) / 2
    return {
        "OVL": float(np.minimum(first, second).sum()),
        "Wasserstein": float(wasserstein_distance(centers, centers, u_weights=first, v_weights=second)),
    }


def _image_group_stats(values: np.ndarray, label: np.ndarray, group: int, thresholds) -> dict:
    chosen = values[label == group]
    if chosen.size == 0:
        raise ValueError("empty GT group")
    row = distribution_stats(chosen, "value", thresholds)
    row["value_q25"] = float(np.quantile(chosen, .25))
    return row


def _mean_rows(rows: list[dict], fields: tuple[str, ...], **identity) -> dict:
    return {**identity, "valid_images": len(rows), **{
        field: float(np.mean([float(row[field]) for row in rows])) for field in fields
    }}


def analyze(config: str | Path, out_dir: str | Path) -> dict:
    settings = load_settings(config); out = validate_output(out_dir)
    manifest = read_jsonl(out / "CAMO" / "manifest.jsonl")
    if not manifest:
        raise RuntimeError("empty max-sim manifest")
    per_image_metrics, descriptive_per_image = [], []
    metric_summary, descriptive_summary, diagnostics, delta_summary = [], [], [], []
    pooled_store: dict[str, dict[str, list[np.ndarray]]] = {}
    hist_store: dict[str, dict[str, list[np.ndarray]]] = {}

    for protocol, suffix in PROTOCOLS:
        pooled_store[protocol] = {"label": [], "sim": [], "gbsp": []}
        hist_store[protocol] = {"sim_bg": [], "sim_fg": [], "res_bg": [], "res_fg": []}
        for row in manifest:
            with np.load(row["cache_path"], allow_pickle=False) as payload:
                valid = payload[f"valid_{suffix}"].astype(bool)
                label = payload[f"gt_label_{suffix}"].astype(np.uint8)[valid]
                max_sim = payload[f"max_bg_similarity_{suffix}"].astype(np.float64)[valid]
                gbsp = payload["gbsp_normalized_residual"].astype(np.float64)[valid]
                if not np.isfinite(max_sim).all() or not np.isfinite(gbsp).all():
                    raise RuntimeError(f"non-finite score: {row['stem']} {protocol}")
                sim_fg_score = 1.0 - max_sim
                sim_auc, sim_ap = _metric(label, sim_fg_score)
                gbsp_auc, gbsp_ap = _metric(label, gbsp)
                per_image_metrics.append({
                    "dataset": "CAMO", "stem": row["stem"], "protocol": protocol,
                    "num_patches": int(valid.sum()), "num_bg": int((label == 0).sum()),
                    "num_fg": int((label == 1).sum()),
                    "AUROC_similarity": sim_auc, "AP_similarity": sim_ap,
                    "AUROC_GBSP": gbsp_auc, "AP_GBSP": gbsp_ap,
                    "delta_AUROC_GBSP_minus_similarity": gbsp_auc - sim_auc,
                    "delta_AP_GBSP_minus_similarity": gbsp_ap - sim_ap,
                })
                pooled_store[protocol]["label"].append(label)
                pooled_store[protocol]["sim"].append(sim_fg_score)
                pooled_store[protocol]["gbsp"].append(gbsp)
                hist_store[protocol]["sim_bg"].append(probability_hist(max_sim[label == 0], settings.similarity_bins))
                hist_store[protocol]["sim_fg"].append(probability_hist(max_sim[label == 1], settings.similarity_bins))
                hist_store[protocol]["res_bg"].append(probability_hist(gbsp[label == 0], settings.residual_bins))
                hist_store[protocol]["res_fg"].append(probability_hist(gbsp[label == 1], settings.residual_bins))
                for family, values, thresholds in (
                    ("max_valid_bg_similarity", max_sim, settings.descriptive_thresholds),
                    ("gbsp_normalized_residual", gbsp, ()),
                ):
                    for group, code in (("Background", 0), ("Foreground", 1)):
                        descriptive_per_image.append({
                            "dataset": "CAMO", "stem": row["stem"], "protocol": protocol,
                            "family": family, "group": group,
                            **_image_group_stats(values, label, code, thresholds),
                        })

        hist_mean = {key: np.mean(np.stack(value), axis=0) for key, value in hist_store[protocol].items()}
        if suffix == "main":
            sim_dir = out / "CAMO/max_similarity"; res_dir = out / "CAMO/residual"
        else:
            sim_dir = out / "CAMO/robustness_core/max_similarity"
            res_dir = out / "CAMO/robustness_core/residual"
        sim_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
        np.save(sim_dir / "bg_hist.npy", hist_mean["sim_bg"])
        np.save(sim_dir / "fg_hist.npy", hist_mean["sim_fg"])
        np.save(res_dir / "bg_hist.npy", hist_mean["res_bg"])
        np.save(res_dir / "fg_hist.npy", hist_mean["res_fg"])
        diagnostics.extend([
            {"dataset": "CAMO", "protocol": protocol, "family": "max_valid_bg_similarity",
             **_hist_diagnostics(hist_mean["sim_bg"], hist_mean["sim_fg"], settings.similarity_bins)},
            {"dataset": "CAMO", "protocol": protocol, "family": "gbsp_normalized_residual",
             **_hist_diagnostics(hist_mean["res_bg"], hist_mean["res_fg"], settings.residual_bins)},
        ])

        labels = np.concatenate(pooled_store[protocol]["label"])
        for method, key in (("1-max_valid_bg_similarity", "sim"), ("GBSP-r8 residual", "gbsp")):
            scores = np.concatenate(pooled_store[protocol][key])
            auc, ap = _metric(labels, scores)
            image_rows = [row for row in per_image_metrics if row["protocol"] == protocol]
            auc_field = "AUROC_similarity" if key == "sim" else "AUROC_GBSP"
            ap_field = "AP_similarity" if key == "sim" else "AP_GBSP"
            metric_summary.append({
                "dataset": "CAMO", "protocol": protocol, "method": method,
                "pooled_patch_AUROC": auc, "pooled_patch_AP": ap,
                "imagewise_mean_AUROC": float(np.mean([row[auc_field] for row in image_rows])),
                "imagewise_median_AUROC": float(np.median([row[auc_field] for row in image_rows])),
                "imagewise_mean_AP": float(np.mean([row[ap_field] for row in image_rows])),
                "imagewise_median_AP": float(np.median([row[ap_field] for row in image_rows])),
            })
        image_rows = [row for row in per_image_metrics if row["protocol"] == protocol]
        delta_summary.append({
            "dataset": "CAMO", "protocol": protocol,
            "mean_delta_AUROC": float(np.mean([r["delta_AUROC_GBSP_minus_similarity"] for r in image_rows])),
            "median_delta_AUROC": float(np.median([r["delta_AUROC_GBSP_minus_similarity"] for r in image_rows])),
            "positive_image_ratio_AUROC": float(np.mean([r["delta_AUROC_GBSP_minus_similarity"] > 0 for r in image_rows])),
            "mean_delta_AP": float(np.mean([r["delta_AP_GBSP_minus_similarity"] for r in image_rows])),
            "median_delta_AP": float(np.median([r["delta_AP_GBSP_minus_similarity"] for r in image_rows])),
            "positive_image_ratio_AP": float(np.mean([r["delta_AP_GBSP_minus_similarity"] > 0 for r in image_rows])),
        })

    stat_fields = ("value_mean", "value_median", "value_q25", "value_q75", "value_q90",
                   *tuple(f"value_p_gt_{x:.1f}" for x in settings.descriptive_thresholds))
    for protocol, _ in PROTOCOLS:
        for family in ("max_valid_bg_similarity", "gbsp_normalized_residual"):
            fields = stat_fields if family == "max_valid_bg_similarity" else stat_fields[:5]
            for group in ("Background", "Foreground"):
                rows = [r for r in descriptive_per_image if r["protocol"] == protocol and
                        r["family"] == family and r["group"] == group]
                descriptive_summary.append(_mean_rows(
                    rows, fields, dataset="CAMO", protocol=protocol, family=family, group=group
                ))

    analysis_dir = out / "analysis"
    write_csv(analysis_dir / "summary.csv", metric_summary)
    write_csv(analysis_dir / "per_image_metrics.csv", per_image_metrics)
    write_csv(analysis_dir / "per_image_descriptive_statistics.csv", descriptive_per_image)
    write_csv(analysis_dir / "descriptive_summary.csv", descriptive_summary)
    write_csv(analysis_dir / "distribution_diagnostics.csv", diagnostics)
    write_csv(analysis_dir / "core_protocol.csv", [
        *[row for row in metric_summary if row["protocol"] == "core_0.2_0.8"],
        *[row for row in delta_summary if row["protocol"] == "core_0.2_0.8"],
    ])
    write_csv(analysis_dir / "delta_summary.csv", delta_summary)

    # Test 5: exact reuse of the previous frozen GBSP residual grouping.
    previous_raw = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["source_pairwise_root"]
    previous_root = Path(previous_raw)
    if not previous_root.is_absolute():
        previous_root = (PROJECT_ROOT / previous_root).resolve()
    previous = json.loads((previous_root / "diagnostics/analysis_summary.json").read_text())
    cache_errors = []
    for group in ("Background", "Foreground"):
        current = next(r for r in descriptive_summary if r["protocol"] == "allpatch_0.5" and
                       r["family"] == "gbsp_normalized_residual" and r["group"] == group)["value_mean"]
        old = next(r for r in previous["residual_summary"] if r["protocol"] == "allpatch_0.5" and
                   r["group"] == group)["mean"]
        cache_errors.append(abs(float(current) - float(old)))
    formal_complete = len(manifest) == 250
    cache_consistency = {
        "max_abs_mean_error": max(cache_errors),
        "passed": (max(cache_errors) <= 1e-7) if formal_complete else None,
        "applicable": formal_complete,
    }
    if formal_complete and not cache_consistency["passed"]:
        raise RuntimeError(f"GBSP residual cache consistency failed: {cache_consistency}")
    result = {
        "images": len(manifest), "formal_camo_complete": formal_complete,
        "metric_summary": metric_summary, "delta_summary": delta_summary,
        "descriptive_summary": descriptive_summary, "distribution_diagnostics": diagnostics,
        "gbsp_cache_consistency": cache_consistency,
        "previous_random_fg_bg_pair_mean": 0.021626,
        "gt_used_only_for_held_out_diagnostic": True,
    }
    write_json(analysis_dir / "analysis_summary.json", result)
    return result


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(analyze(args.config, args.out_dir), ensure_ascii=False, indent=2))
