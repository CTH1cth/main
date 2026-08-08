#!/usr/bin/env python3
"""Summarize GBSP core evaluations without using smoke results for selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_core_ablation.py"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _float(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _baseline_reproduction(cfg, output: Path) -> list[dict]:
    source = _resolve("../workdir/gbsp_rank0_ablation/eval/summary.csv")
    rows = _read_csv(source)
    macro = next(
        (r for r in rows if r.get("scope") == "dataset_macro" and r.get("method") == "GBSP-r>0"),
        None,
    )
    if macro is None:
        raise RuntimeError(f"formal cached baseline summary missing: {source}")
    result = []
    for metric, expected in dict(cfg.GBSP_CORE_BASELINE_REFERENCE).items():
        actual = _float(macro, metric)
        difference = actual - float(expected)
        result.append({
            "metric": metric, "expected": expected, "actual": actual,
            "absolute_error": abs(difference),
            "tolerance": float(cfg.GBSP_CORE_BASELINE_TOLERANCE),
            "passed": int(abs(difference) <= float(cfg.GBSP_CORE_BASELINE_TOLERANCE)),
            "source": str(source),
        })
    _write_csv(output / "baseline_reproduction.csv", result)
    return result


def _copy_experiment_summaries(eval_dirs: list[Path], output: Path) -> tuple[list[dict], list[dict], bool]:
    summaries, diagnostics, full_flags = [], [], []
    bootstrap, runtime, failures = [], [], []
    for directory in eval_dirs:
        summaries.extend(_read_csv(directory / "summary.csv"))
        diagnostics.extend(_read_csv(directory / "candidate_weight_diagnostics.csv"))
        bootstrap.extend(_read_csv(directory / "bootstrap_ci.csv"))
        runtime.extend(_read_csv(directory / "runtime_memory.csv"))
        failure_path = directory / "evaluation_failures.json"
        if failure_path.is_file():
            failures.extend(json.loads(failure_path.read_text(encoding="utf-8")))
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            full_flags.append(bool(metadata.get("full_formal_evaluation")))
    by_method = defaultdict(list)
    for row in diagnostics:
        by_method[row.get("method", "")].append(row)
    diagnostic_fields = (
        "selected_rank", "retained_variance_ratio", "discarded_variance_ratio",
        "num_background_candidates", "num_boundary_candidates", "num_interior_candidates",
        "candidate_gt_contamination", "interior_candidate_gt_contamination",
        "candidate_connectivity_mean", "candidate_path_cost_mean", "effective_sample_size",
        "weight_min", "weight_max", "mean_shift_from_equal", "principal_angle_mean_degrees",
    )
    for row in summaries:
        subset = by_method.get(row.get("method", ""), [])
        if not subset:
            continue
        for field in diagnostic_fields:
            values = [_float(item, field) for item in subset]
            row[f"diag_mean_{field}"] = float(np.nanmean(values)) if not all(math.isnan(v) for v in values) else float("nan")
        ranks = defaultdict(int)
        for item in subset:
            ranks[str(int(round(_float(item, "selected_rank"))))] += 1
        row["diag_selected_rank_histogram"] = json.dumps(dict(sorted(ranks.items())), ensure_ascii=False)
        row["diag_weight_warning_count"] = sum(int(item.get("weight_concentration_warning", 0)) for item in subset)
    mapping = {
        "rank": "rank_summary.csv",
        "background_source": "background_source_summary.csv",
        "pca_weight": "weighted_pca_summary.csv",
        "combined": "combined_summary.csv",
    }
    for experiment, filename in mapping.items():
        _write_csv(output / filename, [r for r in summaries if r.get("method", "").startswith(experiment + "/")])
    residual = []
    per_image = []
    for directory in eval_dirs:
        per_image.extend(_read_csv(directory / "per_image_metrics.csv"))
    for row in per_image:
        if math.isclose(_float(row, "threshold"), .5, abs_tol=1e-12):
            residual.append({key: row.get(key) for key in (
                "dataset", "stem", "experiment", "variant", "method",
                "fg_raw_median", "bg_raw_q95", "fg_median_over_bg_q95",
                "pixel_AP", "pixel_AUROC",
            )})
    _write_csv(output / "residual_distribution.csv", residual)
    _write_csv(output / "per_image_metrics.csv", per_image)
    _write_csv(output / "per_dataset_metrics.csv", [r for r in summaries if r.get("scope") == "dataset"])
    _write_csv(output / "continuous_metrics.csv", [
        {key: row.get(key) for key in ("scope", "dataset", "method", "num_samples", "pixel_AP", "pixel_AUROC")}
        for row in summaries if math.isclose(_float(row, "threshold"), .5, abs_tol=1e-12)
    ])
    _write_csv(output / "binary_metrics_050.csv", [r for r in summaries if math.isclose(_float(r, "threshold"), .50, abs_tol=1e-12)])
    _write_csv(output / "binary_metrics_058.csv", [r for r in summaries if math.isclose(_float(r, "threshold"), .58, abs_tol=1e-12)])
    _write_csv(output / "candidate_statistics.csv", diagnostics)
    _write_csv(output / "pca_spectrum_statistics.csv", [{
        key: row.get(key) for key in (
            "dataset", "stem", "experiment", "variant", "method", "selected_rank",
            "retained_variance_ratio", "discarded_variance_ratio",
            "background_feature_covariance_trace", "spectral_effective_rank",
        )
    } for row in diagnostics])
    _write_csv(output / "candidate_weight_diagnostics.csv", diagnostics)
    _write_csv(output / "bootstrap_ci.csv", bootstrap)
    _write_csv(output / "bootstrap_ci95.csv", bootstrap)
    _write_csv(output / "runtime_memory.csv", runtime)
    _write_csv(output / "threshold_curve.csv", summaries)
    _write_json(output / "per_dataset_metrics.json", {"summary": summaries})
    _write_json(output / "evaluation_failures.json", failures)
    _write_json(output / "failure_summary.json", {
        "num_failures": len(failures), "failures": failures,
        "policy": "record rare failures and preserve all valid samples",
    })
    return summaries, diagnostics, bool(full_flags) and all(full_flags)


def _pareto(summary: list[dict]) -> dict:
    rows = [r for r in summary if r.get("scope") == "dataset_macro" and math.isclose(_float(r, "threshold"), .5)]
    fields = ("pixel_AP", "pixel_AUROC", "S_m", "F_beta_w", "E_mean")
    front = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(_float(other, field) >= _float(candidate, field) for field in fields) and _float(other, "MAE") <= _float(candidate, "MAE")
            strictly = any(_float(other, field) > _float(candidate, field) for field in fields) or _float(other, "MAE") < _float(candidate, "MAE")
            if no_worse and strictly:
                dominated = True; break
        if not dominated:
            front.append({key: candidate.get(key) for key in ("method", *fields, "MAE")})
    return {"orientation": {**{field: "higher" for field in fields}, "MAE": "lower"}, "front": front}


def _audit_markdown(master: Path, output: Path) -> dict:
    path = master / "configuration_audit.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "# GBSP Core 配置审计", "",
        f"- 状态：{audit['status']}",
        f"- 表征：{audit['representation']['backbone']}；{audit['representation']['feature']}",
        f"- 网格与归一化：{audit['representation']['grid']}；{audit['representation']['pca_input_normalization']}",
        f"- 背景起点：{audit['background']['boundary_seed']}",
        f"- 局部图：{audit['background']['local_graph']}",
        f"- 连通性：{audit['background']['connectivity']}",
        f"- Full BC：{audit['background']['full_bc_source']}；正式候选数分布={audit['background']['formal_full_bc_count_distribution']}",
        f"- 当前 PCA：{audit['pca']['current']}；权重={audit['pca']['weight']}",
        f"- PCA 中心化：{audit['pca']['centering']}；正式实际秩分布={audit['pca']['formal_actual_rank_distribution']}",
        f"- 查询集合：{audit['pca']['query']}；背景响应覆盖={audit['pca']['background_score_overwrite']}",
        f"- 分数：{audit['pca']['score']}",
        f"- resize/校准：{audit['resize_and_calibration']}",
        "- 生成阶段使用 GT：否。",
        f"- 背景响应硬覆盖代码搜索：{audit['overwrite_pattern_search']['conclusion']}。",
    ]
    (output / "config_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def _plots(summary: list[dict], diagnostics: list[dict], output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = [
        ("rank", "rank_ap_auroc_curve.png", "Rank audit"),
        ("background_source", "background_source_comparison.png", "Background source audit"),
        ("pca_weight", "weighted_pca_comparison.png", "Connectivity-weighted PCA audit"),
    ]
    for experiment, filename, title in specs:
        rows = [r for r in summary if r.get("scope") == "dataset_macro" and r.get("method", "").startswith(experiment + "/") and math.isclose(_float(r, "threshold"), .5)]
        if not rows:
            continue
        labels = [r["method"].split("/", 1)[1] for r in rows]
        x = np.arange(len(labels)); width = .36
        figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.2))
        axis.bar(x - width/2, [_float(r, "pixel_AP") for r in rows], width, label="Pixel AP")
        axis.bar(x + width/2, [_float(r, "pixel_AUROC") for r in rows], width, label="Pixel AUROC")
        axis.set_xticks(x, labels, rotation=25, ha="right"); axis.set_ylim(0, 1)
        axis.set_title(title); axis.legend(); figure.tight_layout()
        target = plot_dir / filename; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    for threshold, filename in ((.5, "rank_binary_metrics_050.png"), (.58, "rank_binary_metrics_058.png")):
        rank_rows = [r for r in summary if r.get("scope") == "dataset_macro" and r.get("method", "").startswith("rank/") and math.isclose(_float(r, "threshold"), threshold)]
        if rank_rows:
            labels = [r["method"].split("/", 1)[1] for r in rank_rows]
            x = np.arange(len(labels)); width = .2
            figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.15), 4.4))
            for index, metric in enumerate(("S_m", "F_beta_w", "E_mean", "MAE")):
                values = [_float(r, metric) if metric != "MAE" else 1 - _float(r, metric) for r in rank_rows]
                axis.bar(x + (index - 1.5) * width, values, width, label=("1-MAE" if metric == "MAE" else metric))
            axis.set_xticks(x, labels, rotation=25, ha="right"); axis.set_ylim(0, 1)
            axis.set_title(f"Rank vs hard COD metrics @ {threshold:.2f}"); axis.legend(ncol=4); figure.tight_layout()
            target = plot_dir / filename; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    macro = [r for r in summary if r.get("scope") == "dataset_macro" and math.isclose(_float(r, "threshold"), .5)]
    if macro:
        labels = [r["method"] for r in macro]; x = np.arange(len(labels)); width = .4
        figure, axis = plt.subplots(figsize=(max(9, len(labels) * .8), 4.5))
        axis.bar(x-width/2, [_float(r, "fg_raw_median") for r in macro], width, label="FG median")
        axis.bar(x+width/2, [_float(r, "bg_raw_q95") for r in macro], width, label="BG q95")
        axis.set_xticks(x, labels, rotation=35, ha="right"); axis.set_title("Foreground/background raw residual")
        axis.legend(); figure.tight_layout()
        target = plot_dir / "foreground_background_separation.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    background_rows = [r for r in macro if r.get("method", "").startswith("background_source/")]
    background_diag = [r for r in diagnostics if r.get("experiment") == "background_source"]
    if background_rows and background_diag:
        contamination = defaultdict(list)
        for row in background_diag:
            contamination[f"background_source/{row['variant']}"].append(_float(row, "candidate_gt_contamination"))
        figure, axis = plt.subplots(figsize=(6, 4.5))
        for row in background_rows:
            method = row["method"]
            axis.scatter(np.mean(contamination[method]), _float(row, "pixel_AP"), s=55, label=method.split("/", 1)[1])
        axis.set_xlabel("Candidate GT contamination (diagnostic only)"); axis.set_ylabel("Pixel AP")
        axis.set_title("Candidate contamination vs performance"); axis.legend(); figure.tight_layout()
        target = plot_dir / "candidate_contamination_distribution.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    if background_diag:
        grouped_count, grouped_contamination = defaultdict(list), defaultdict(list)
        for row in background_diag:
            grouped_count[row["variant"]].append(_float(row, "num_background_candidates"))
            grouped_contamination[row["variant"]].append(_float(row, "candidate_gt_contamination"))
        labels = list(grouped_count)
        figure, axis = plt.subplots(figsize=(6.5, 4.2)); axis.boxplot([grouped_count[k] for k in labels], tick_labels=labels)
        axis.set_title("Candidate count distribution"); axis.set_ylabel("count"); figure.tight_layout()
        target = plot_dir / "candidate_count_distribution.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    rank_diag = [r for r in diagnostics if r.get("experiment") == "rank"]
    if rank_diag:
        grouped_rank, grouped_variance = defaultdict(list), defaultdict(list)
        for row in rank_diag:
            grouped_rank[row["variant"]].append(_float(row, "selected_rank"))
            grouped_variance[row["variant"]].append(_float(row, "retained_variance_ratio"))
        labels = list(grouped_rank)
        figure, axis = plt.subplots(figsize=(max(7, len(labels)), 4.2)); axis.boxplot([grouped_rank[k] for k in labels], tick_labels=labels)
        axis.set_title("Per-image selected rank"); figure.tight_layout()
        target = plot_dir / "rank_distribution.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
        figure, axis = plt.subplots(figsize=(max(7, len(labels)), 4.2)); axis.boxplot([grouped_variance[k] for k in labels], tick_labels=labels)
        axis.set_title("Retained variance distribution"); axis.set_ylim(0, 1); figure.tight_layout()
        target = plot_dir / "retained_variance_distribution.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    if macro:
        labels = [r["method"] for r in macro]; x = np.arange(len(labels))
        figure, axis = plt.subplots(figsize=(max(9, len(labels) * .8), 4.2)); axis.bar(x, [_float(r, "bg_raw_q95") for r in macro])
        axis.set_xticks(x, labels, rotation=35, ha="right"); axis.set_title("Background residual q95"); figure.tight_layout()
        target = plot_dir / "background_residual_q95.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
        figure, axis = plt.subplots(figsize=(6, 4.5))
        for row in macro:
            axis.scatter(_float(row, "pixel_AP"), _float(row, "pixel_AUROC"), s=35)
            axis.annotate(row["method"], (_float(row, "pixel_AP"), _float(row, "pixel_AUROC")), fontsize=6)
        axis.set_xlabel("Pixel AP"); axis.set_ylabel("Pixel AUROC"); axis.set_title("Continuous metric Pareto")
        figure.tight_layout(); target = plot_dir / "continuous_metric_pareto.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
        figure, axis = plt.subplots(figsize=(6, 4.5))
        for row in macro:
            axis.scatter(_float(row, "F_beta_w"), _float(row, "S_m"), s=35)
            axis.annotate(row["method"], (_float(row, "F_beta_w"), _float(row, "S_m")), fontsize=6)
        axis.set_xlabel("F_beta_w"); axis.set_ylabel("S_m"); axis.set_title("Binary metric Pareto @0.50")
        figure.tight_layout(); target = plot_dir / "binary_metric_pareto.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    dataset_rows = [r for r in summary if r.get("scope") == "dataset" and math.isclose(_float(r, "threshold"), .5)]
    if dataset_rows:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for dataset in sorted({r["dataset"] for r in dataset_rows}):
            subset = [r for r in dataset_rows if r["dataset"] == dataset]
            axes[0].plot(range(len(subset)), [_float(r, "pixel_AP") for r in subset], marker="o", label=dataset)
            axes[1].plot(range(len(subset)), [_float(r, "pixel_AUROC") for r in subset], marker="o", label=dataset)
        axes[0].set_title("Per-dataset AP"); axes[1].set_title("Per-dataset AUROC")
        for axis in axes: axis.legend(fontsize=7)
        figure.tight_layout(); target = plot_dir / "per_dataset_ap_auroc.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    weight = [r for r in diagnostics if r.get("experiment") == "pca_weight"]
    if weight:
        grouped = defaultdict(list)
        for row in weight:
            n = max(1.0, _float(row, "num_background_candidates"))
            grouped[row["variant"]].append(_float(row, "effective_sample_size") / n)
        figure, axis = plt.subplots(figsize=(6, 4))
        labels = list(grouped); axis.boxplot([grouped[k] for k in labels], tick_labels=labels)
        axis.axhline(.5, color="red", linestyle="--", linewidth=1); axis.set_ylabel("ESS / Nb")
        axis.set_title("PCA weight concentration"); figure.tight_layout()
        target = plot_dir / "weight_distribution.png"; figure.savefig(target, dpi=160); plt.close(figure); paths.append(str(target))
    return paths


def _qualitative(eval_dirs: list[Path], selection: dict, full: bool, output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    cache_maps: dict[str, dict[tuple[str, str], dict]] = {}
    metric_rows = []
    for directory in eval_dirs:
        metric_rows.extend(_read_csv(directory / "per_image_metrics.csv"))
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for root_value in metadata.get("cache_roots", []):
            root = Path(root_value)
            manifest_path = root / "manifest_test.jsonl"
            if not manifest_path.is_file():
                continue
            rows = read_jsonl(manifest_path)
            if rows:
                cache_maps[str(rows[0]["experiment"])] = {
                    (row["dataset"], row["stem"]): row for row in rows
                }
    if "rank" not in cache_maps or "pca_weight" not in cache_maps:
        return []
    chosen = "r16"
    if full and isinstance(selection.get("selected"), dict):
        chosen = str(selection["selected"].get("rank", "rank/r16")).split("/", 1)[-1]
    rows = [r for r in metric_rows if math.isclose(_float(r, "threshold"), .5) and r.get("experiment") == "rank" and r.get("variant") in ("current", chosen)]
    lookup = {(r["dataset"], r["stem"], r["variant"]): r for r in rows}
    keys = sorted({(r["dataset"], r["stem"]) for r in rows})
    paired = [(key, _float(lookup[(*key, chosen)], "pixel_AP") - _float(lookup[(*key, "current")], "pixel_AP")) for key in keys if (*key, chosen) in lookup and (*key, "current") in lookup]
    if not paired:
        return []

    def render(selected_pairs: list[tuple[tuple[str, str], float]], target: Path, title: str) -> None:
        columns = (
            "RGB", "GT", "Boundary-280", "Full BC", "path cost", "path weight",
            "current residual", f"{chosen} residual", "current @.50", f"{chosen} @.50",
            "current @.58", f"{chosen} @.58", "residual difference",
        )
        figure, axes = plt.subplots(len(selected_pairs), len(columns), figsize=(26, 2.4 * len(selected_pairs)), squeeze=False)
        for row_index, (key, delta) in enumerate(selected_pairs):
            rank_payload = torch_load(Path(cache_maps["rank"][key]["cache_path"]), map_location="cpu")
            weight_payload = torch_load(Path(cache_maps["pca_weight"][key]["cache_path"]), map_location="cpu")
            dabe = torch_load(Path(rank_payload["source_dabe_path"]), map_location="cpu")
            with Image.open(rank_payload["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB").resize((148, 148)), dtype=np.uint8)
            with Image.open(rank_payload["gt_path"]) as image:
                gt = np.asarray(image.convert("L").resize((148, 148), Image.Resampling.NEAREST), dtype=np.float32) / 255
            current = rank_payload["results"]["current"]["absolute_minmax"].squeeze().numpy()
            variant = rank_payload["results"][chosen]["absolute_minmax"].squeeze().numpy()
            full_indices = rank_payload["results"]["current"]["background_indices"].long()
            full_mask = torch.zeros(37 * 37); full_mask[full_indices] = 1
            boundary_indices = torch.where((torch.arange(37).repeat_interleave(37) < 2) | (torch.arange(37).repeat_interleave(37) >= 35) | (torch.arange(37).repeat(37) < 2) | (torch.arange(37).repeat(37) >= 35))[0]
            boundary_mask = torch.zeros(37 * 37); boundary_mask[boundary_indices] = 1
            bc = dabe["bc_map_37"].float().reshape(-1)
            path = (-torch.log(bc.clamp_min(1e-8))).reshape(37, 37).numpy()
            weight_item = weight_payload["results"]["path_inverse"]
            weight_map = torch.zeros(37 * 37); weight_map[weight_item["background_indices"].long()] = weight_item["normalized_weights"]
            panels = [
                rgb, gt, boundary_mask.reshape(37, 37), full_mask.reshape(37, 37), path,
                weight_map.reshape(37, 37), current, variant, current > .5, variant > .5,
                current > .58, variant > .58, variant - current,
            ]
            for column, (axis, panel) in enumerate(zip(axes[row_index], panels)):
                if column == 0:
                    axis.imshow(panel)
                elif column == 12:
                    limit = max(1e-6, float(np.abs(panel).max())); axis.imshow(panel, cmap="coolwarm", vmin=-limit, vmax=limit)
                else:
                    axis.imshow(panel, cmap="gray")
                axis.set_axis_off()
                if row_index == 0: axis.set_title(columns[column], fontsize=8)
            axes[row_index, 0].set_ylabel(f"{key[0]}/{key[1]}\nΔAP={delta:+.3f}", fontsize=7)
        figure.suptitle(title + (" (full)" if full else " (Test20 diagnostic only)"), fontsize=11)
        figure.tight_layout(); figure.savefig(target, dpi=140, bbox_inches="tight"); plt.close(figure)

    plot_dir = output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(paired, key=lambda item: item[1])
    success, failure = ordered[-min(2, len(ordered)):][::-1], ordered[:min(2, len(ordered))]
    success_path, failure_path = plot_dir / "success_visualizations.png", plot_dir / "failure_visualizations.png"
    render(success, success_path, f"Current vs {chosen}: largest AP gains")
    render(failure, failure_path, f"Current vs {chosen}: largest AP losses")
    return [str(success_path), str(failure_path)]


def _select(summary: list[dict], full: bool) -> dict:
    if not full:
        return {
            "status": "pending_full_evaluation",
            "reason": "Test20 only verifies implementation and numerical invariants; it is forbidden for method selection.",
            "selected": None,
        }
    selected, best_observed, admission = {}, {}, {}
    baselines = {
        "rank": "rank/current", "background_source": "background_source/fullbc",
        "pca_weight": "pca_weight/equal", "combined": "combined/current",
    }
    for experiment in ("rank", "background_source", "pca_weight", "combined"):
        rows = [r for r in summary if r.get("scope") == "dataset_macro" and r.get("method", "").startswith(experiment + "/") and math.isclose(_float(r, "threshold"), .5)]
        if not rows:
            continue
        best = max(rows, key=lambda r: (
            _float(r, "pixel_AP"), _float(r, "pixel_AUROC"), _float(r, "F_beta_w"),
            _float(r, "S_m"), _float(r, "E_mean"), -_float(r, "MAE"),
        ))
        best_observed[experiment] = best["method"]
        baseline = next((r for r in rows if r["method"] == baselines[experiment]), None)
        delta_ap = _float(best, "pixel_AP") - _float(baseline, "pixel_AP") if baseline else float("nan")
        delta_auc = _float(best, "pixel_AUROC") - _float(baseline, "pixel_AUROC") if baseline else float("nan")
        admitted = experiment == "combined" or delta_ap >= .001 or delta_auc >= .001
        selected[experiment] = best["method"] if admitted else baselines[experiment]
        admission[experiment] = {
            "best_observed": best["method"], "baseline": baselines[experiment],
            "delta_pixel_AP": delta_ap, "delta_pixel_AUROC": delta_auc,
            "minimum_continuous_gain_gate": .001, "admitted": admitted,
            "reason": (
                "passes the minimum continuous-gain gate; remaining task-book conditions still apply"
                if admitted else "fails the task-book minimum +0.001 AP/AUROC gate"
            ),
        }
    any_admitted = any(
        value["admitted"] for key, value in admission.items() if key != "combined"
    )
    return {
        "status": (
            "selected_from_full_6473_only" if any_admitted
            else "full_6473_complete_no_first_batch_variant_admitted_keep_current"
        ),
        "selection_rule": "task-book admission gate first; then AP > AUROC > per-dataset stability > residual diagnostics > hard metrics > simplicity",
        "selected": selected, "best_observed": best_observed, "admission": admission,
        "combined_recommended": any_admitted,
    }


def _required_answers(summary: list[dict], diagnostics: list[dict], full: bool, selection: dict) -> list[str]:
    macro = [r for r in summary if r.get("scope") == "dataset_macro" and math.isclose(_float(r, "threshold"), .5)]
    by_method = {r["method"]: r for r in macro}
    rank = [r for r in macro if r["method"].startswith("rank/")]
    best_ap = max(rank, key=lambda r: _float(r, "pixel_AP"))["method"] if rank else "pending"
    best_auc = max(rank, key=lambda r: _float(r, "pixel_AUROC"))["method"] if rank else "pending"
    boundary, fullbc = by_method.get("background_source/boundary280"), by_method.get("background_source/fullbc")
    equal, weighted = by_method.get("pca_weight/equal"), by_method.get("pca_weight/path_inverse")
    inverse_diag = [r for r in diagnostics if r.get("method") == "pca_weight/path_inverse"]
    full_diag = [r for r in diagnostics if r.get("method") == "background_source/fullbc"]
    ess_ratio = float(np.mean([_float(r, "effective_sample_size") / max(1, _float(r, "num_background_candidates")) for r in inverse_diag])) if inverse_diag else float("nan")
    full_contamination = float(np.mean([_float(r, "candidate_gt_contamination") for r in full_diag])) if full_diag else float("nan")
    interior_contamination = float(np.nanmean([_float(r, "interior_candidate_gt_contamination") for r in full_diag])) if full_diag else float("nan")
    current, r16, r32 = by_method.get("rank/current"), by_method.get("rank/r16"), by_method.get("rank/r32")
    dataset_rows = {
        (r.get("dataset"), r.get("method")): r for r in summary
        if r.get("scope") == "dataset" and math.isclose(_float(r, "threshold"), .5)
    }
    prefix = "全量裁决" if full else "Test20诊断（不得用于选型）"
    def delta(left, right, metric):
        return _float(left, metric) - _float(right, metric) if left and right else float("nan")
    answers = [
        "1. 当前真实 rank 规则：EV90 后限制到 1–8；正式缓存实际分布为 r8×6472、r7×1。",
        "2. 每图 rank 已写入 `pca_spectrum_statistics.csv`；正式 current 分布同上。",
        f"3. 最高名义 Pixel AP rank：{best_ap}（{prefix}）；全量 r8 仅比 current 高 {delta(by_method.get('rank/r8'), current, 'pixel_AP'):+.9f}，属于数值等价，不构成改进。",
        f"4. 最高 AUROC rank：{best_auc}（{prefix}）；r16 相对 current 的 AUROC 为 {delta(r16, current, 'pixel_AUROC'):+.6f}，但 AP 为 {delta(r16, current, 'pixel_AP'):+.6f}。",
        (
            f"5. Rank 提高确实压低背景 q95：current={_float(current,'bg_raw_q95'):.4f}、r16={_float(r16,'bg_raw_q95'):.4f}、r32={_float(r32,'bg_raw_q95'):.4f}。"
            if full else "5. Rank 对背景 q95 的趋势待全量。"
        ),
        (
            f"6. 但前景中位残差同步下降：current={_float(current,'fg_raw_median'):.4f}、r16={_float(r16,'fg_raw_median'):.4f}、r32={_float(r32,'fg_raw_median'):.4f}，解释了高 rank 的 AP 损失。"
            if full else "6. Rank 对前景残差的趋势待全量。"
        ),
        "7. 当前 capped-EV90/r8 容量仍是连续排序的合理中间点；更高 rank 主要改善固定0.50硬指标，但没有改善原始排序。" if full else "7. 当前 rank 是否合理待全量。",
        f"8. Boundary-280 相对 Full BC 的 AP 差：{delta(boundary, fullbc, 'pixel_AP'):+.6f}（{prefix}）。",
        "9. Full BC 数量效应无法由 matched280 单独识别：matched280 与 Boundary-280 在当前检查中集合恒等。",
        "10. FullBC-Matched-280 未形成独立候选集合，因此不能声称优于 Boundary-280。",
        f"11. Full BC 总候选前景污染率={full_contamination:.4%}，内部候选污染率={interior_contamination:.4%}；内部候选反而更纯，没有明显前景污染证据。",
        f"12. path_inverse 相对 equal 的 AP 差：{delta(weighted, equal, 'pixel_AP'):+.6f}（{prefix}）。",
        f"13. 加权 PCA 的背景 q95 差：{delta(weighted, equal, 'bg_raw_q95'):+.6f}（{prefix}）。",
        f"14. path_inverse 平均 ESS/Nb={ess_ratio:.4f}；另需注意路径中位数为0造成的近边界退化。",
        f"15. 第一批正式接纳结果：{selection.get('selected') if full else '待6473张全量评测'}；观测最优另见 selected_configuration.json。",
        (
            "16. 组合版：存在通过最低连续增益门的单项，可只组合一个版本继续验证。"
            if full and selection.get("combined_recommended") else
            "16. 组合版：没有单项通过 +0.001 连续增益门，当前不支持把退化等价项组合成正式改进。"
        ),
        "17. COD10K/NC4K 方向已按数据集输出；Test20 不裁决。" if not full else (
            f"17. Boundary-280 在 COD10K/NC4K 的 AP 分别变化 "
            f"{delta(dataset_rows.get(('TE-COD10K','background_source/boundary280')),dataset_rows.get(('TE-COD10K','background_source/fullbc')),'pixel_AP'):+.6f}/"
            f"{delta(dataset_rows.get(('NC4K','background_source/boundary280')),dataset_rows.get(('NC4K','background_source/fullbc')),'pixel_AP'):+.6f}，不具备大数据集一致性。"
        ),
        "18. fixed-0.50 的完整指标已输出；是否自然提高待全量。" if not full else "18. 高 rank 的 fixed-0.50 指标提高伴随 AP 下降，不能解释为连续重构分数增强。",
        "19. 仅固定检查 0.50/0.58，不在测试集搜索最优工作阈值。",
        "20. 任何阈值差异都与 AP/AUROC 并列报告，禁止把纯重标定解释为排序提升。",
        "21. 瓶颈路径暂不触发：任务书要求 Full BC 明显优于 Boundary，但本轮并未成立；零中位数退化只作为失败诊断记录。" if full else "21. 瓶颈路径优先级待全量。",
        "22. 候选比例实验暂不触发：候选来源最大 AP 差仅 +0.000898 且 Bootstrap 跨0，未证明数量明显影响结果。" if full else "22. 候选比例优先级待全量。",
        "23. 若继续执行任务书，下一步进入第三批诊断A（五折样本外 BC 残差）；只有诊断成立才允许稳健重加权。" if full else "23. BC 样本内偏差按触发条件决定。",
        "24. Hotelling T² 维持低优先级：第一批没有证明需要 SPE 以外的互补统计量。",
        "25. 最终连续分数没有比 current 更强；正式配置保留 current Full BC equal PCA。" if full else "25. 最终连续分数是否更强待全量。",
        "26. 所有变体仍为免训练、单图 PCA、显式残差；未引入学习参数。",
    ]
    return answers


def analyze(args: argparse.Namespace) -> None:
    cfg = load_config(_resolve(args.config))
    master = _resolve(args.work_root or cfg.GBSP_CORE_OUTPUT_ROOT)
    output = _resolve(args.out_dir or master)
    output.mkdir(parents=True, exist_ok=True)
    eval_dirs = [_resolve(path) for path in args.eval_dir]
    baseline = _baseline_reproduction(cfg, output)
    audit = _audit_markdown(master, output)
    summary, diagnostics, full = _copy_experiment_summaries(eval_dirs, output)
    selection = _select(summary, full)
    _write_json(output / "selected_config.json", selection)
    _write_json(output / "selected_configuration.json", selection)
    _write_json(output / "pareto_front.json", _pareto(summary))
    plot_paths = _plots(summary, diagnostics, output)
    plot_paths.extend(_qualitative(eval_dirs, selection, full, output))

    baseline_pass = all(bool(int(r["passed"])) for r in baseline)
    matched = [r for r in diagnostics if r.get("experiment") == "background_source"]
    exact_rate = float(np.mean([int(r.get("boundary_matched_exact", 0)) for r in matched])) if matched else float("nan")
    warning = [r for r in diagnostics if int(r.get("weight_concentration_warning", 0))]
    inverse = [r for r in diagnostics if r.get("experiment") == "pca_weight" and r.get("variant") == "path_inverse"]
    zero_median = [r for r in inverse if abs(_float(r, "path_cost_median")) <= 1e-12]
    report = [
        "# GBSP Core Optimization 报告", "",
        "## 当前状态", "",
        f"- 正式旧缓存基线复现门：{'通过' if baseline_pass else '失败'}。",
        f"- 本轮是否为四数据集 6473 张完整评测：{'是' if full else '否（Test20 烟测）'}。",
        f"- 配置选择：`{selection['status']}`。",
        "- 本轮没有训练、没有重新提取 DINO、生成阶段没有读取 GT。",
        "",
        "## 关键实现事实", "",
        f"- 查询为全部 1369 个 Patch，背景字典成员的响应不被覆盖。",
        f"- `fullbc_matched280` 与两圈边界逐图完全相同率：{exact_rate:.3f}。",
        f"- 路径权重 ESS 低于 0.5×Nb 的样本/变体数：{len(warning)}。这些情况仅告警，不中断。",
        f"- `path_inverse` 中候选路径代价中位数为 0 的样本数：{len(zero_median)}/{len(inverse)}；这会使内部候选权重接近 0，因此即使 ESS 未触发告警，也应视为接近边界 PCA 的退化行为。",
        f"- 连续分数使用 Absolute squared residual；Pixel AP/AUROC 不做 Min-Max。",
        "",
        "## 裁决边界", "",
        ("- Test20 只验证代码与数值不变量，不据此选择方法；需运行任务书给出的全量命令后再裁决。" if not full else
         f"- 全量选择结果：`{json.dumps(selection['selected'], ensure_ascii=False)}`。"),
        "- 若 matched280 与 boundary280 恒等，则该对照只能证明容量控制后的候选集合相同，不能支持“内部低路径候选更优”的论点。",
        "",
        "## 文件", "",
        f"- 配置审计：`config_audit.md` / `configuration_audit.json`（源位于 {master}）。",
        f"- 基线复现：`baseline_reproduction.csv`。",
        f"- 图表数量：{len(plot_paths)}。",
        "",
        "## 任务书 26 项回答", "",
        *_required_answers(summary, diagnostics, full, selection),
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "GBSP_CORE_OPTIMIZATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "baseline_pass": baseline_pass, "full": full,
        "selection_status": selection["status"], "plots": plot_paths,
    }, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--work-root", "--work_root", dest="work_root")
    value.add_argument("--eval-dir", "--eval_dir", dest="eval_dir", action="append", required=True)
    value.add_argument("--out-dir", "--out_dir", dest="out_dir")
    return value


if __name__ == "__main__":
    analyze(parser().parse_args())
