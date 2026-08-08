#!/usr/bin/env python3
"""Consolidate GBSP-SPE spectra, thresholds, metrics, plots and final answers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load  # noqa: E402


PRIMARY = ("S_m", "F_beta_w", "E_mean", "MAE")
SECONDARY = ("F_beta_mean", "Precision", "Recall", "Area", "IoU", "Dice")


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict, key: str, default=float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _finite(values) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)


def _stats(values) -> dict:
    array = _finite(values)
    if not array.size:
        return {key: float("nan") for key in ("mean", "median", "std", "q10", "q25", "q75", "q90", "min", "max")}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)), "std": float(array.std()),
        "q10": float(np.quantile(array, .10)), "q25": float(np.quantile(array, .25)),
        "q75": float(np.quantile(array, .75)), "q90": float(np.quantile(array, .90)),
        "min": float(array.min()), "max": float(array.max()),
    }


def _stage(root: Path) -> tuple[str, Path, Path]:
    for name in ("full6473", "diagnostic200", "test20"):
        cache = root / name
        evaluation = root / f"eval_{name}"
        if (cache / "manifest_test.jsonl").is_file() and (evaluation / "per_image_metrics.csv").is_file():
            return name, cache, evaluation
    raise RuntimeError("no completed GBSP-SPE cache/evaluation stage found")


def _diagnostics(cache_root: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    manifest = read_jsonl(cache_root / "manifest_test.jsonl")
    spectrum_rows, control_rows, equivalent_rows, exceedance_rows = [], [], [], []
    for index, row in enumerate(manifest, 1):
        path = Path(row["cache_path"])
        payload = torch_load(path, map_location="cpu")
        identity = {"dataset": row["dataset"], "stem": row["stem"], "cache_path": str(path)}
        spectrum_rows.append({
            **identity,
            "background_candidate_count": payload.get("background_candidate_count"),
            "feature_dimension": payload.get("feature_dimension"),
            "effective_spectrum_dimension": payload.get("effective_spectrum_dimension"),
            "pca_rank": payload.get("pca_rank"),
            "theta1": payload.get("theta1"), "theta2": payload.get("theta2"), "theta3": payload.get("theta3"),
            "h0": payload.get("h0"), "jm_bracket": payload.get("jm_bracket"),
            "discarded_eigenvalue_count": int(payload.get("discarded_eigenvalues", torch.empty(0)).numel()),
            "jm_valid": int(bool(payload.get("numerical_validity", {}).get("jm_095", False))),
            "gamma_valid": int(bool(payload.get("numerical_validity", {}).get("gamma_095", False))),
            "jm_failure_reason": payload.get("failure_reason", {}).get("jm_095"),
        })
        control_rows.append({
            **identity, "pca_rank": payload.get("pca_rank"), "theta1": payload.get("theta1"),
            "tau_jm_090": payload.get("jm_tau_090"), "tau_jm_095": payload.get("jm_tau_095"),
            "tau_jm_099": payload.get("jm_tau_099"), "tau_gamma_095": payload.get("gamma_tau_095"),
        })
        equivalent_rows.append({
            **identity, "pca_rank": payload.get("pca_rank"), "theta1": payload.get("theta1"),
            "background_candidate_count": payload.get("background_candidate_count"),
            "equivalent_jm_090": payload.get("equivalent_minmax_threshold_jm090"),
            "equivalent_jm_095": payload.get("equivalent_minmax_threshold_jm095"),
            "equivalent_jm_099": payload.get("equivalent_minmax_threshold_jm099"),
            "equivalent_gamma_095": payload.get("equivalent_minmax_threshold_gamma095"),
            "foreground_area_jm095": payload.get("foreground_area_jm095"),
        })
        exceedance_rows.append({
            **identity,
            "background_exceedance_jm095": payload.get("background_candidate_exceedance_rate"),
            "background_exceedance_gamma095": payload.get("background_candidate_exceedance_rate_gamma095"),
        })
        if index % 1000 == 0:
            print(f"GBSP-SPE analysis caches {index}/{len(manifest)}", flush=True)
    return spectrum_rows, control_rows, equivalent_rows, exceedance_rows


def _distribution_report(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    output = []
    datasets = tuple(dict.fromkeys(str(row["dataset"]) for row in rows))
    for scope, dataset in (("all", "ALL"), *(("dataset", name) for name in datasets)):
        subset = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        for field in fields:
            output.append({"scope": scope, "dataset": dataset, "field": field, "num_samples": len(subset), **_stats(row.get(field) for row in subset)})
    return output


def _correlations(spectrum: list[dict], equivalent: list[dict], exceedance: list[dict], per_image: list[dict]) -> list[dict]:
    eq = {(r["dataset"], r["stem"]): r for r in equivalent}
    ex = {(r["dataset"], r["stem"]): r for r in exceedance}
    metrics = {(r["dataset"], r["stem"]): r for r in per_image if r["method"] == "jm_spe_095"}
    merged = []
    for row in spectrum:
        key = (row["dataset"], row["stem"])
        if key in eq:
            merged.append({**row, **eq[key], **ex.get(key, {}), **metrics.get(key, {})})
    pairs = (
        ("equivalent_jm_095", "background_candidate_count"),
        ("equivalent_jm_095", "pca_rank"),
        ("equivalent_jm_095", "theta1"),
        ("equivalent_jm_095", "foreground_area_jm095"),
        ("background_exceedance_jm095", "delta_F_beta_w_vs_fixed_058"),
        ("background_exceedance_jm095", "delta_Precision_vs_fixed_058"),
    )
    output = []
    for left, right in pairs:
        values = [(float(row[left]), float(row[right])) for row in merged if row.get(left) is not None and row.get(right) is not None and math.isfinite(float(row[left])) and math.isfinite(float(row[right]))]
        rho, pvalue = spearmanr(*zip(*values)) if len(values) >= 3 else (float("nan"), float("nan"))
        output.append({"left": left, "right": right, "num_samples": len(values), "spearman_rho": float(rho), "p_value": float(pvalue)})
    return output


def _plots(root: Path, summary: list[dict], per_image: list[dict], spectrum: list[dict], control: list[dict], equivalent: list[dict], exceedance: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return [{"plot": "all", "error": repr(error)}]

    def run(name, callback):
        try:
            fig = callback(plt)
            fig.tight_layout()
            fig.savefig(root / name, dpi=160)
            plt.close(fig)
        except Exception as error:
            failures.append({"plot": name, "error": repr(error), "traceback": traceback.format_exc()})

    macro = [row for row in summary if row.get("scope") == "dataset_macro"]
    def primary_plot(plt):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4)); methods = [row["method"] for row in macro]
        for ax, metric in zip(axes, PRIMARY):
            ax.bar(np.arange(len(methods)), [_number(row, metric) for row in macro]); ax.set_title(metric); ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
        return fig
    run("primary_metric_comparison.png", primary_plot)

    dataset_rows = [row for row in summary if row.get("scope") == "dataset"]
    def dataset_plot(plt):
        fig, axes = plt.subplots(2, 5, figsize=(22, 8)); methods = list(dict.fromkeys(row["method"] for row in dataset_rows)); datasets = list(dict.fromkeys(row["dataset"] for row in dataset_rows))
        for ax, metric in zip(axes.reshape(-1), (*PRIMARY, *SECONDARY)):
            width = .8 / max(len(methods), 1)
            for index, method in enumerate(methods):
                values = [next((_number(r, metric) for r in dataset_rows if r["dataset"] == d and r["method"] == method), np.nan) for d in datasets]
                ax.bar(np.arange(len(datasets)) + index * width, values, width, label=method)
            ax.set_title(metric); ax.set_xticks(np.arange(len(datasets)) + .4, datasets, rotation=25, ha="right")
        axes[0, 0].legend(fontsize=7)
        return fig
    run("dataset_wise_all_metrics.png", dataset_plot)

    def histogram(field, title):
        def callback(plt):
            fig, ax = plt.subplots(figsize=(7, 5)); values = _finite(row.get(field) for row in (control if field.startswith("tau") else equivalent if field.startswith("equivalent") else exceedance)); ax.hist(values, bins=40); ax.set(xlabel=field, ylabel="images", title=title); return fig
        return callback
    run("control_limit_distribution.png", histogram("tau_jm_095", "JM-SPE95 control limits"))
    run("equivalent_minmax_threshold_distribution.png", histogram("equivalent_jm_095", "Equivalent Min-Max threshold"))
    run("background_exceedance_distribution.png", histogram("background_exceedance_jm095", "BC exceedance rate"))

    def threshold_dataset(plt):
        fig, ax = plt.subplots(figsize=(8, 5)); datasets = list(dict.fromkeys(row["dataset"] for row in equivalent)); values = [[_number(row, "equivalent_jm_095") for row in equivalent if row["dataset"] == d] for d in datasets]; ax.boxplot(values, tick_labels=datasets, showfliers=False); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("equivalent JM95 threshold"); return fig
    run("threshold_by_dataset.png", threshold_dataset)

    def scatter(x, y, name):
        def callback(plt):
            source = equivalent if x in equivalent[0] else spectrum
            fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter([_number(r, x) for r in source], [_number(r, y) for r in source], s=5, alpha=.35); ax.set(xlabel=x, ylabel=y, title=name); return fig
        return callback
    run("threshold_vs_theta1.png", scatter("theta1", "equivalent_jm_095", "Threshold vs theta1"))
    run("threshold_vs_rank.png", scatter("pca_rank", "equivalent_jm_095", "Threshold vs rank"))
    run("threshold_vs_area.png", scatter("foreground_area_jm095", "equivalent_jm_095", "Threshold vs area"))

    for metric, filename in (("S", "per_image_delta_S.png"), ("F_beta_w", "per_image_delta_Fw.png"), ("E", "per_image_delta_E.png"), ("MAE", "per_image_delta_MAE.png")):
        field = {"S": "delta_S_vs_fixed_058", "E": "delta_E_vs_fixed_058"}.get(metric, f"delta_{metric}_vs_fixed_058")
        def callback(plt, field=field, metric=metric):
            values = sorted(_number(row, field) for row in per_image if row["method"] == "jm_spe_095")
            fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(values); ax.axhline(0, color="black", lw=1); ax.set(xlabel="ordered image", ylabel=field, title=f"JM-SPE95 per-image delta {metric}"); return fig
        run(filename, callback)
    return failures


def _visuals(root: Path, per_image: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return [{"plot": "visualizations", "error": repr(error)}]
    candidates = [row for row in per_image if row["method"] == "jm_spe_095"]

    def montage(path: Path, selected: list[dict]):
        if not selected:
            return
        fig, axes = plt.subplots(len(selected), 11, figsize=(27, 2.8 * len(selected)), squeeze=False)
        for axrow, row in zip(axes, selected):
            cache = torch_load(Path(row["cache_path"]), map_location="cpu")
            source = torch_load(Path(cache["source_gbsp_path"]), map_location="cpu")
            with Image.open(row["image_path"]) as image: rgb = np.asarray(image.convert("RGB"))
            with Image.open(row["gt_path"]) as image: gt = np.asarray(image.convert("L"))
            raw = cache["raw_residual_map"][0].numpy(); mm = source["absolute_minmax"][0].numpy(); bc = np.zeros(1369); bc[source["background_indices"].numpy()] = 1; bc = bc.reshape(37, 37)
            jm90 = cache["jm_mask_090"][0].numpy() if cache.get("jm_mask_090") is not None else np.zeros((37, 37))
            jm95 = cache["jm_mask_095"][0].numpy() if cache.get("jm_mask_095") is not None else np.zeros((37, 37))
            jm99 = cache["jm_mask_099"][0].numpy() if cache.get("jm_mask_099") is not None else np.zeros((37, 37))
            gamma = cache["gamma_mask_095"][0].numpy() if cache.get("gamma_mask_095") is not None else np.zeros((37, 37))
            calibrated = cache["jm_calibrated_map_095"][0].numpy() if cache.get("jm_calibrated_map_095") is not None else np.zeros((37, 37))
            fixed = mm > .58
            panels = ((rgb, "RGB"), (gt, "GT"), (bc, "Full BC"), (raw, "Raw residual"), (mm, "Min-Max"), (fixed, "fixed-.58"), (jm90, "JM90"), (jm95, "JM95"), (jm99, "JM99"), (gamma, "Gamma95"), (calibrated, "JM calibrated"))
            for ax, (array, title) in zip(axrow, panels):
                ax.imshow(array, cmap=None if array.ndim == 3 else "viridis"); ax.set_title(title, fontsize=8); ax.axis("off")
            axrow[0].set_ylabel(f"{row['dataset']}/{row['stem']}\ndS={_number(row, 'delta_S_vs_fixed_058'):+.3f}")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    try:
        montage(root / "success_visualizations.png", sorted(candidates, key=lambda row: _number(row, "delta_S_vs_fixed_058"), reverse=True)[:8])
        montage(root / "failure_visualizations.png", sorted(candidates, key=lambda row: _number(row, "delta_S_vs_fixed_058"))[:8])
    except Exception as error:
        failures.append({"plot": "success/failure_visualizations", "error": repr(error), "traceback": traceback.format_exc()})
    return failures


def analyze(root: Path) -> None:
    root = root.resolve(); stage, cache_root, eval_root = _stage(root)
    per_image = _csv(eval_root / "per_image_metrics.csv"); summary = _csv(eval_root / "per_dataset_metrics.csv")
    spectrum, control, equivalent, exceedance = _diagnostics(cache_root)
    _write_csv(root / "spectrum_diagnostics.csv", spectrum)
    _write_csv(root / "control_limit_distribution.csv", control)
    _write_csv(root / "equivalent_threshold_distribution.csv", equivalent)
    _write_csv(root / "background_exceedance_distribution.csv", exceedance)
    _write_csv(root / "spectrum_distribution_summary.csv", _distribution_report(spectrum, ("theta1", "theta2", "theta3", "h0")))
    _write_csv(root / "control_limit_distribution_summary.csv", _distribution_report(control, ("tau_jm_090", "tau_jm_095", "tau_jm_099", "tau_gamma_095")))
    _write_csv(root / "equivalent_threshold_distribution_summary.csv", _distribution_report(equivalent, ("equivalent_jm_090", "equivalent_jm_095", "equivalent_jm_099", "equivalent_gamma_095")))
    _write_csv(root / "background_exceedance_distribution_summary.csv", _distribution_report(exceedance, ("background_exceedance_jm095", "background_exceedance_gamma095")))
    _write_csv(root / "spe_correlations.csv", _correlations(spectrum, equivalent, exceedance, per_image))

    for name in (f"{stage}_summary.csv", "per_dataset_metrics.csv", "per_image_metrics.csv", "primary_metric_deltas.csv", "secondary_metric_deltas.csv", "bootstrap_ci95.csv", "numerical_validity_summary.json", "downstream_1x1_results.csv"):
        source = eval_root / name
        if source.is_file():
            shutil.copy2(source, root / name)
    plot_failures = _plots(root, summary, per_image, spectrum, control, equivalent, exceedance)
    plot_failures.extend(_visuals(root, per_image))
    _write_json(root / "analysis_visualization_failures.json", plot_failures)

    validity = _json(eval_root / "numerical_validity_summary.json", {}) or {}
    macro = {row["method"]: row for row in summary if row.get("scope") == "dataset_macro"}
    fixed = macro.get("gbsp_fixed_058", {}); jm = macro.get("jm_spe_095", {})
    gamma = macro.get("gamma_spe_095", {})
    deltas = {metric: _number(jm, metric) - _number(fixed, metric) for metric in PRIMARY} if fixed and jm else {}
    gamma_deltas = {metric: _number(gamma, metric) - _number(fixed, metric) for metric in PRIMARY} if fixed and gamma else {}
    eq_stats = _stats(row.get("equivalent_jm_095") for row in equivalent)
    area_stats = _stats(row.get("foreground_area_jm095") for row in equivalent)
    bc_stats = _stats(row.get("background_exceedance_jm095") for row in exceedance)
    invalid = sum(1 - int(row["jm_valid"]) for row in spectrum)
    jm_invalid_rate = invalid / max(len(spectrum), 1)
    gamma_rows = [
        row for row in per_image
        if row.get("method") == "gamma_spe_095"
        and int(float(row.get("numerical_validity", 0))) == 1
    ]
    gamma_areas = _finite(row.get("foreground_area_patch") for row in gamma_rows)
    gamma_invalid = sum(
        1 for row in per_image
        if row.get("method") == "gamma_spe_095"
        and int(float(row.get("numerical_failure", 0))) == 1
    )
    gamma_mean_area = float(gamma_areas.mean()) if gamma_areas.size else float("nan")
    gamma_empty_rate = float((gamma_areas == 0).mean()) if gamma_areas.size else float("nan")
    gamma_over50_rate = float((gamma_areas > .5).mean()) if gamma_areas.size else float("nan")
    gamma_catastrophes = []
    if math.isfinite(gamma_empty_rate) and gamma_empty_rate > .10:
        gamma_catastrophes.append("empty_mask_rate_above_10pct")
    if math.isfinite(gamma_over50_rate) and gamma_over50_rate > .05:
        gamma_catastrophes.append("foreground_area_over_50pct_rate_above_5pct")
    if math.isfinite(gamma_mean_area) and gamma_mean_area < .03:
        gamma_catastrophes.append("mean_foreground_area_below_003")
    if math.isfinite(gamma_mean_area) and gamma_mean_area > .30:
        gamma_catastrophes.append("mean_foreground_area_above_030")
    numerical_implementation = "jm_spe_095"
    if stage in {"diagnostic200", "full6473"} and jm_invalid_rate > .01:
        numerical_implementation = "gamma_spe_095"
    failure = False
    if stage == "full6473" and deltas:
        failure = (
            sum(deltas[key] < -0.01 for key in ("S_m", "F_beta_w", "E_mean")) == 3
            or deltas["MAE"] > 0.008 or invalid / max(len(spectrum), 1) > .01
        )
    if stage == "diagnostic200" and jm_invalid_rate > .01:
        decision = (
            "STOP_SPE_ROUTE_BEFORE_FULL"
            if gamma_catastrophes or gamma_invalid
            else "ADMIT_GAMMA_SPE95_TO_FULL6473"
        )
    elif stage == "test20":
        decision = "PENDING_DIAGNOSTIC200_NUMERICAL_GATE"
    else:
        decision = "STOP_SPE_ROUTE" if failure or gamma_catastrophes else "REVIEW_TRAINING_GATE"
    _write_json(root / "numerical_route_decision.json", {
        "schema": "gbsp_spe_numerical_route_decision_v1",
        "source_stage": stage,
        "jm_invalid_count": invalid,
        "jm_invalid_rate": jm_invalid_rate,
        "predeclared_numerical_implementation": numerical_implementation,
        "gamma_invalid_count": gamma_invalid,
        "gamma_mean_foreground_area": gamma_mean_area,
        "gamma_empty_mask_rate": gamma_empty_rate,
        "gamma_area_over_50pct_rate": gamma_over50_rate,
        "gamma_catastrophic_reasons": gamma_catastrophes,
        "admit_full6473": decision in {"ADMIT_GAMMA_SPE95_TO_FULL6473", "REVIEW_TRAINING_GATE"},
        "decision": decision,
        "gt_used_to_select_jm_vs_gamma": False,
    })
    stopped_before_full = decision == "STOP_SPE_ROUTE_BEFORE_FULL"
    report = [
        "# GBSP-SPE Calibration Report", "", f"- Current stage: {stage}",
        f"- Samples: {len(spectrum)}", f"- JM-SPE95 mathematically invalid: {invalid}",
        f"- JM-SPE95 invalid rate: {jm_invalid_rate:.2%}",
        f"- Predeclared numerical implementation after validity gate: {numerical_implementation}",
        f"- Gamma-SPE95 mathematically invalid: {gamma_invalid}",
        f"- Gamma-SPE95 mean foreground area: {gamma_mean_area:.6f}",
        f"- Gamma-SPE95 area>50% rate: {gamma_over50_rate:.2%}",
        f"- Gamma catastrophic flags: {gamma_catastrophes}",
        f"- Mean equivalent Min-Max threshold: {eq_stats['mean']:.6f}",
        f"- Median equivalent Min-Max threshold: {eq_stats['median']:.6f}",
        f"- Mean JM-SPE95 foreground area: {area_stats['mean']:.6f}",
        f"- Mean Full-BC exceedance rate: {bc_stats['mean']:.6f}",
        f"- Decision: {decision}", "", "## Required answers", "",
        f"1. JM-SPE95 mathematical validity: {len(spectrum)-invalid}/{len(spectrum)} valid.",
        "2. theta1/theta2/theta3 distributions: see spectrum_distribution_summary.csv.",
        f"3. h0 stability: {_stats(row.get('h0') for row in spectrum)['min']:.6g} minimum; invalid count above.",
        "4. Extreme JM limits: see control_limit_distribution_summary.csv.",
        f"5. Mean equivalent Min-Max threshold: {eq_stats['mean']:.6f}.",
        f"6. Equivalent threshold central 10%-90%: {eq_stats['q10']:.6f} to {eq_stats['q90']:.6f}.",
        f"7. JM-SPE95 foreground area mean: {area_stats['mean']:.6f}.",
        f"8. Full-BC exceedance rate mean: {bc_stats['mean']:.6f}.",
        (
            "9. JM90/95/99 precision-recall trend: undefined because all JM limits are mathematically invalid."
            if invalid == len(spectrum) else
            "9. JM90/95/99 precision-recall trend: see per_dataset_metrics.csv."
        ),
        f"10-13. JM-SPE95 vs fixed-.58 primary deltas: {deltas if deltas else 'pending'} (NaN when JM is mathematically invalid).",
        (
            "14. Four-dataset full stability: not run because the predeclared Diagnostic200 gate stopped the route."
            if stopped_before_full else
            "14. Four-dataset stability: pending full6473." if stage != "full6473" else
            "14. Four-dataset stability: see per_dataset_metrics.csv."
        ),
        f"15. Gamma/JM status: JM invalid; Gamma primary deltas are {gamma_deltas if gamma_deltas else 'pending'}.",
        "16. Control-limit calibration does not qualify for a formal fixed-.5 comparison because JM is invalid and Gamma is catastrophic.",
        f"17. Replace fixed-.58: {decision}.",
        "18-19. 1x1 training: not admitted; the numerical/catastrophic-behavior gate failed.",
        ("20. Paper choice: retain GBSP fixed-.58; do not adopt SPE." if stopped_before_full else f"20. Paper choice: {decision}."),
        "21. Decision uses S, Fw, E, MAE and all secondary metrics; no scalar weighted score.",
    ]
    (root / "GBSP_SPE_CALIBRATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "samples": len(spectrum), "jm_invalid": invalid, "decision": decision}, ensure_ascii=False, indent=2))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", default="../workdir/gbsp_spe_calibration")
    return value


if __name__ == "__main__":
    analyze(Path(parser().parse_args().root))
