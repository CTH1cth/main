#!/usr/bin/env python3
"""Consolidate GBSP-V2 metrics, select the simplest valid method and plot audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("b0", "b1", "b2", "b3", "b4", "b5")
NAMES = {"b0": "B0 Current", "b1": "B1 RW-Hard", "b2": "B2 RW-Soft", "b3": "B3 RW-Soft-CD", "b4": "B4 Weighted PCA", "b5": "B5 SWOR"}
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _float(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _macro(rows: list[dict]) -> dict[str, dict]:
    return {row["variant"]: row for row in rows if row.get("scope") == "dataset_macro"}


def _dataset(rows: list[dict], variant: str) -> dict[str, dict]:
    return {row["dataset"]: row for row in rows if row.get("scope") == "dataset" and row.get("variant") == variant}


def _select(continuous: list[dict], binary050: list[dict], binary058: list[dict], residual: list[dict]) -> dict:
    cont, hard50, hard58, distribution = map(_macro, (continuous, binary050, binary058, residual))
    baseline = cont.get("b0", {})
    admitted = []
    audit = []
    for variant in VARIANTS[1:]:
        if variant not in cont:
            audit.append({"variant": variant, "admitted": False, "reason": "missing formal metrics"})
            continue
        delta_ap = _float(cont[variant], "pixel_AP") - _float(baseline, "pixel_AP")
        delta_auc = _float(cont[variant], "pixel_AUROC") - _float(baseline, "pixel_AUROC")
        per_variant, per_b0 = _dataset(continuous, variant), _dataset(continuous, "b0")
        stable_datasets = sum(
            _float(per_variant.get(dataset, {}), "pixel_AP") >= _float(per_b0.get(dataset, {}), "pixel_AP") - 1e-4
            and _float(per_variant.get(dataset, {}), "pixel_AUROC") >= _float(per_b0.get(dataset, {}), "pixel_AUROC") - 1e-4
            for dataset in DATASETS
        )
        cod_nc_stable = all(
            _float(per_variant.get(dataset, {}), "pixel_AP") >= _float(per_b0.get(dataset, {}), "pixel_AP") - .001
            for dataset in ("TE-COD10K", "NC4K")
        )
        bg_tail_down = _float(distribution.get(variant, {}), "bg_q95") < _float(distribution.get("b0", {}), "bg_q95")
        fg_preserved = _float(distribution.get(variant, {}), "fg_q50") >= .99 * _float(distribution.get("b0", {}), "fg_q50")
        fixed50_delta = _float(hard50.get(variant, {}), "F_beta_w") - _float(hard50.get("b0", {}), "F_beta_w")
        fixed58_stable = all(
            (_float(hard58.get(variant, {}), metric) >= _float(hard58.get("b0", {}), metric) - .002)
            if metric != "MAE" else (_float(hard58.get(variant, {}), metric) <= _float(hard58.get("b0", {}), metric) + .002)
            for metric in ("S_m", "F_beta_w", "E_mean", "MAE")
        )
        clear = (delta_ap >= .003 and delta_auc >= -.001) or (delta_auc >= .003 and delta_ap >= -.001)
        small_stable = max(delta_ap, delta_auc) >= .001 and min(delta_ap, delta_auc) >= -.001 and stable_datasets >= 3
        accepted = (clear or small_stable) and stable_datasets >= 3 and cod_nc_stable and bg_tail_down and fg_preserved and fixed58_stable
        audit.append({
            "variant": variant, "delta_AP": delta_ap, "delta_AUROC": delta_auc,
            "stable_dataset_count": stable_datasets, "cod10k_nc4k_stable": cod_nc_stable,
            "background_q95_down": bg_tail_down, "foreground_median_preserved": fg_preserved,
            "fixed050_Fw_delta": fixed50_delta, "fixed058_stable": fixed58_stable,
            "admitted": accepted,
        })
        if accepted and variant in ("b2", "b3", "b4", "b5"):
            admitted.append(variant)
    if not admitted:
        selected, reason = "b0", "No predeclared B2-B5 variant passed the formal admission rules."
    else:
        best = max(admitted, key=lambda variant: (_float(cont[variant], "pixel_AP"), _float(cont[variant], "pixel_AUROC")))
        selected = best
        # Explicit simplicity rule from the task book.
        if best == "b5" and "b4" in admitted and abs(_float(cont["b5"], "pixel_AP") - _float(cont["b4"], "pixel_AP")) < .001 and abs(_float(cont["b5"], "pixel_AUROC") - _float(cont["b4"], "pixel_AUROC")) < .001:
            selected = "b4"
        if selected == "b4" and "b3" in admitted and abs(_float(cont["b4"], "pixel_AP") - _float(cont["b3"], "pixel_AP")) < .001 and abs(_float(cont["b4"], "pixel_AUROC") - _float(cont["b3"], "pixel_AUROC")) < .001:
            selected = "b3"
        reason = "Selected among admitted variants, then applied the predeclared <0.001 simplicity preference."
    return {"selected_variant": selected, "selected_name": NAMES[selected], "reason": reason, "admitted_variants": admitted, "audit": audit, "gt_used_to_define_method": False, "gt_used_for_formal_comparison_only": True}


def _save_bar(path: Path, rows: dict[str, dict], metrics: tuple[str, ...], title: str) -> None:
    variants = [variant for variant in VARIANTS if variant in rows]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(variants)); width = .8 / max(1, len(metrics))
    for index, metric in enumerate(metrics):
        axis.bar(x + (index - (len(metrics) - 1) / 2) * width, [_float(rows[v], metric) for v in variants], width, label=metric)
    axis.set_xticks(x, [variant.upper() for variant in variants]); axis.set_title(title); axis.grid(axis="y", alpha=.25); axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plots(eval_dir: Path, output: Path) -> None:
    continuous = _read_csv(eval_dir / "full6473_continuous_metrics.csv")
    binary050 = _read_csv(eval_dir / "full6473_binary_050.csv")
    binary058 = _read_csv(eval_dir / "full6473_binary_058.csv")
    adaptive = _read_csv(eval_dir / "simple_adaptive_threshold_metrics.csv")
    residual = _read_csv(eval_dir / "residual_distribution.csv")
    candidate = _read_csv(eval_dir / "candidate_statistics.csv")
    random_walk = _read_csv(eval_dir / "random_walk_statistics.csv")
    weighted = _read_csv(eval_dir / "weighted_pca_statistics.csv")
    swor = _read_csv(eval_dir / "swor_statistics.csv")
    curves = _read_csv(eval_dir / "threshold_curves.csv")
    plateaus = _read_csv(eval_dir / "threshold_plateau_width.csv")
    _save_bar(output / "ap_auroc_comparison.png", _macro(continuous), ("pixel_AP", "pixel_AUROC"), "GBSP-V2 continuous ranking")
    _save_bar(output / "binary_metrics_050.png", _macro(binary050), ("S_m", "F_beta_w", "E_mean", "MAE"), "Fixed 0.50")
    _save_bar(output / "binary_metrics_058.png", _macro(binary058), ("S_m", "F_beta_w", "E_mean", "MAE"), "Fixed 0.58")
    _save_bar(output / "background_residual_q95.png", _macro(residual), ("bg_q95",), "Background residual q95")
    _save_bar(output / "foreground_residual_median.png", _macro(residual), ("fg_q50",), "Foreground residual median")
    _save_bar(output / "foreground_background_separation.png", _macro(residual), ("separation_ratio",), "Foreground/background separation")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for dataset in DATASETS:
        rows = {row["variant"]: row for row in continuous if row.get("scope") == "dataset" and row.get("dataset") == dataset}
        axes[0].plot([_float(rows.get(v, {}), "pixel_AP") for v in VARIANTS], marker="o", label=dataset)
        axes[1].plot([_float(rows.get(v, {}), "pixel_AUROC") for v in VARIANTS], marker="o", label=dataset)
    for axis, title in zip(axes, ("Pixel AP", "AUROC")):
        axis.set_xticks(range(6), [v.upper() for v in VARIANTS]); axis.set_title(title); axis.grid(alpha=.25)
    axes[1].legend(fontsize=7); figure.tight_layout(); figure.savefig(output / "per_dataset_ap_auroc.png", dpi=180); plt.close(figure)

    def grouped_mean(rows: list[dict], field: str) -> dict[str, float]:
        values = defaultdict(list)
        for row in rows:
            value = _float(row, field)
            if math.isfinite(value): values[row["variant"]].append(value)
        return {key: float(np.mean(value)) for key, value in values.items()}

    candidate_boundary = grouped_mean(candidate, "boundary_candidate_ratio")
    candidate_interior = {key: 1.0 - value for key, value in candidate_boundary.items()}
    _save_bar(output / "candidate_boundary_interior_ratio.png", {key: {"boundary": candidate_boundary[key], "interior": candidate_interior[key]} for key in candidate_boundary}, ("boundary", "interior"), "Candidate boundary/interior ratio")
    _save_bar(output / "candidate_diversity.png", {key: {"NN distance": value} for key, value in grouped_mean(candidate, "nearest_neighbor_distance").items()}, ("NN distance",), "Candidate diversity")
    _save_bar(output / "random_walk_depth_bias.png", {key: {"correlation": value} for key, value in grouped_mean(random_walk, "confidence_depth_correlation").items()}, ("correlation",), "Random-walk confidence vs spatial depth")
    _save_bar(output / "weighted_pca_effective_sample_size.png", {key: {"N_eff": value} for key, value in grouped_mean(weighted, "effective_sample_size").items()}, ("N_eff",), "Weighted PCA effective sample size")

    figure, axis = plt.subplots(figsize=(7, 4.5)); values = [_float(row, "ledoit_wolf_shrinkage") for row in swor if math.isfinite(_float(row, "ledoit_wolf_shrinkage"))]
    axis.hist(values, bins=40); axis.set_title("SWOR Ledoit-Wolf shrinkage"); axis.grid(alpha=.2); figure.tight_layout(); figure.savefig(output / "swor_shrinkage_distribution.png", dpi=180); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for variant in VARIANTS:
        rows = sorted((row for row in curves if row.get("scope") == "dataset_macro" and row["variant"] == variant), key=lambda row: float(row["threshold"]))
        for axis, metric in zip(axes.ravel(), ("S_m", "F_beta_w", "E_mean", "MAE")):
            axis.plot([_float(row, "threshold") for row in rows], [_float(row, metric) for row in rows], label=variant.upper())
            axis.set_title(metric); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); figure.tight_layout(); figure.savefig(output / "threshold_curves_all_variants.png", dpi=180); plt.close(figure)
    _save_bar(output / "threshold_plateau_width.png", {row["variant"]: row for row in plateaus}, ("plateau_width",), "Stable threshold plateau width")
    adaptive_macro = [row for row in adaptive if row.get("scope") == "dataset_macro"]
    labels = sorted({(row["variant"], row["protocol"]) for row in adaptive_macro})
    figure, axis = plt.subplots(figsize=(11, 4.8)); axis.bar(range(len(labels)), [_float(next(row for row in adaptive_macro if (row["variant"], row["protocol"]) == key), "F_beta_w") for key in labels]); axis.set_xticks(range(len(labels)), [f"{v.upper()}\n{p}" for v, p in labels], rotation=45, ha="right", fontsize=7); axis.set_title("Simple adaptive thresholds: weighted F-beta"); axis.grid(axis="y", alpha=.2); figure.tight_layout(); figure.savefig(output / "simple_adaptive_thresholds.png", dpi=180); plt.close(figure)


def _qualitative(cache_root: Path, eval_dir: Path, selected: str, output: Path) -> None:
    per_image = _read_csv(eval_dir / "per_image_metrics.csv")
    lookup = {(row["dataset"], row["stem"], row["variant"]): row for row in per_image}
    keys = sorted({
        (row["dataset"], row["stem"])
        for row in per_image
        if (row["dataset"], row["stem"], "b0") in lookup
        and (row["dataset"], row["stem"], selected) in lookup
    })
    deltas = [(key, _float(lookup[(*key, selected)], "pixel_AP") - _float(lookup[(*key, "b0")], "pixel_AP")) for key in keys]
    manifest = {(row["dataset"], row["stem"]): row for row in read_jsonl(cache_root / "manifest_test.jsonl")}

    def render(chosen, path: Path, title: str) -> None:
        if not chosen:
            figure, axis = plt.subplots(); axis.text(.5, .5, "No qualitative sample available", ha="center"); axis.axis("off"); figure.savefig(path); plt.close(figure); return
        columns = ("RGB", "GT", "Full BC", "Hard RW", "Soft seed", "Soft RW", "Top-conf", "CD core", "B0 Q", "B4 Q", "B5/Sel Q", "B0@.5", "Sel@.5", "Difference")
        figure, axes = plt.subplots(len(chosen), len(columns), figsize=(2.0 * len(columns), 2.2 * len(chosen)), squeeze=False)
        for row_index, (key, delta) in enumerate(chosen):
            payload = torch_load(Path(manifest[key]["cache_path"]), map_location="cpu")
            shared, results = payload["shared_diagnostics"], payload["results"]
            rgb = np.asarray(Image.open(payload["image_path"]).convert("RGB").resize((148, 148)))
            gt = np.asarray(Image.open(payload["gt_path"]).convert("L").resize((37, 37)))
            def mask(indices):
                value = np.zeros(1369, dtype=np.float32); value[indices.long().numpy()] = 1; return value.reshape(37, 37)
            selected_item = results.get(selected, results.get("b5", results["b0"]))
            b0 = results["b0"]["minmax_score"].squeeze().numpy(); selected_map = selected_item["minmax_score"].squeeze().numpy()
            images = (
                rgb, gt, mask(shared["current_full_bc_indices"]), shared["hard_random_walk_confidence"].squeeze().numpy(),
                shared["soft_seed_map"].squeeze().numpy(), shared["soft_random_walk_confidence"].squeeze().numpy(),
                mask(shared["soft_top_confidence_indices"]), mask(shared["coreset_indices"]), b0,
                results.get("b4", results["b0"])["minmax_score"].squeeze().numpy(), selected_map,
                b0 >= .5, selected_map >= .5, selected_map - b0,
            )
            for column, (axis, image) in enumerate(zip(axes[row_index], images)):
                axis.imshow(image, cmap=None if column == 0 else ("coolwarm" if column == 13 else "gray")); axis.axis("off")
                if row_index == 0: axis.set_title(columns[column], fontsize=8)
            axes[row_index, 0].set_ylabel(f"{key[0]}/{key[1]}\nΔAP={delta:+.3f}", fontsize=7)
        figure.suptitle(title); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)

    render(sorted(deltas, key=lambda item: item[1], reverse=True)[:4], output / "success_visualizations.png", "GBSP-V2 successes")
    render(sorted(deltas, key=lambda item: item[1])[:4], output / "failure_visualizations.png", "GBSP-V2 failures")


def _report(eval_dir: Path, selection: dict, output: Path) -> None:
    continuous = _read_csv(eval_dir / "full6473_continuous_metrics.csv")
    binary050 = _read_csv(eval_dir / "full6473_binary_050.csv")
    binary058 = _read_csv(eval_dir / "full6473_binary_058.csv")
    adaptive = _read_csv(eval_dir / "simple_adaptive_threshold_metrics.csv")
    plateau = _read_csv(eval_dir / "threshold_plateau_width.csv")
    random_walk = _read_csv(eval_dir / "random_walk_statistics.csv")
    soft = _read_csv(eval_dir / "soft_seed_statistics.csv")
    candidate = _read_csv(eval_dir / "candidate_statistics.csv")
    weighted = _read_csv(eval_dir / "weighted_pca_statistics.csv")
    swor = _read_csv(eval_dir / "swor_statistics.csv")
    cont, hard50, hard58 = _macro(continuous), _macro(binary050), _macro(binary058)
    selected = selection["selected_variant"]
    def delta(variant, field, table=cont): return _float(table.get(variant, {}), field) - _float(table.get("b0", {}), field)
    def mean(rows, field, variant=None):
        values = [_float(row, field) for row in rows if variant is None or row.get("variant") == variant]; values = [value for value in values if math.isfinite(value)]; return float(np.mean(values)) if values else float("nan")
    best_dataset = {dataset: max(VARIANTS, key=lambda v: _float(_dataset(continuous, v).get(dataset, {}), "pixel_AP")) for dataset in DATASETS}
    adaptive_macro = [row for row in adaptive if row.get("scope") == "dataset_macro"]
    best_otsu = max((row for row in adaptive_macro if row["protocol"] == "otsu"), key=lambda row: _float(row, "F_beta_w"), default={"variant": "N/A"})
    best_multi = max((row for row in adaptive_macro if row["protocol"] == "multi_otsu_3"), key=lambda row: _float(row, "F_beta_w"), default={"variant": "N/A"})
    widest = max(plateau, key=lambda row: _float(row, "plateau_width"), default={"variant": "N/A", "plateau_width": "nan"})
    answers = [
        f"1. 累积路径深度偏置：B1置信度—深度相关变化与B0候选深度差需结合图表判断；B1平均相关={mean(random_walk, 'confidence_depth_correlation', 'b1'):.4f}。",
        f"2. Random Walk内部覆盖：B1内部候选比例={1-mean(candidate, 'boundary_candidate_ratio', 'b1'):.4f}。",
        f"3. B1连续指标变化：ΔAP={delta('b1','pixel_AP'):+.6f}，ΔAUROC={delta('b1','pixel_AUROC'):+.6f}。",
        f"4. Soft seed显著降权比例={mean(soft, 'significantly_downweighted_fraction'):.4%}。",
        f"5. Soft seed的候选污染变化：B2-B1={mean(candidate,'candidate_gt_foreground_contamination','b2')-mean(candidate,'candidate_gt_foreground_contamination','b1'):+.6f}。",
        f"6. 真实边界背景平均Soft权重={mean(soft,'boundary_gt_background_weight_mean'):.4f}。",
        f"7. Top-confidence最近邻距离={mean(candidate,'nearest_neighbor_distance','b2'):.6f}。",
        f"8. B3有效秩相对B2变化={mean(candidate,'effective_rank','b3')-mean(candidate,'effective_rank','b2'):+.4f}。",
        f"9. B3连续排序：ΔAP(B3-B2)={_float(cont.get('b3',{}),'pixel_AP')-_float(cont.get('b2',{}),'pixel_AP'):+.6f}。",
        f"10. 加权PCA均值偏移={mean(weighted,'mean_shift','b4'):.6f}，平均主角度={mean(weighted,'principal_angle_mean_degrees','b4'):.4f}°。",
        f"11. B4有效样本量={mean(weighted,'effective_sample_size','b4'):.2f}。",
        f"12. B5复杂背景尾部变化见background_residual_q95.png；Δbg-q95={delta('b5','bg_q95',_macro(_read_csv(eval_dir/'residual_distribution.csv'))):+.6f}。",
        f"13. B5前景中位残差变化={delta('b5','fg_q50',_macro(_read_csv(eval_dir/'residual_distribution.csv'))):+.6f}。",
        f"14. B5连续指标：ΔAP={delta('b5','pixel_AP'):+.6f}，ΔAUROC={delta('b5','pixel_AUROC'):+.6f}。",
        f"15-18. 各库AP最佳：" + "，".join(f"{dataset}={variant.upper()}" for dataset, variant in best_dataset.items()) + "。",
        f"19. fixed-0.5最佳Fw={max(hard50,key=lambda v:_float(hard50[v],'F_beta_w')).upper() if hard50 else 'N/A'}。",
        f"20. fixed-0.58最佳Fw={max(hard58,key=lambda v:_float(hard58[v],'F_beta_w')).upper() if hard58 else 'N/A'}。",
        f"21. Otsu最佳Fw={best_otsu.get('variant','N/A').upper()}。",
        f"22. Multi-Otsu最佳Fw={best_multi.get('variant','N/A').upper()}。",
        f"23. 最宽阈值平台={widest.get('variant','N/A').upper()}，宽度={_float(widest,'plateau_width'):.2f}。",
        f"24. 连续排序与阈值性能是否同步：最终候选ΔAP={delta(selected,'pixel_AP'):+.6f}，fixed-0.5 ΔFw={delta(selected,'F_beta_w',hard50):+.6f}。",
        f"25. SWOR是否必要：依据预声明简化规则，最终选择{selected.upper()}。",
        "26. 方法仍为解析式、training-free；生成公式未使用GT。",
        f"27. 是否值得重生训练伪标签：{'是，需进入受控1×1训练' if selected != 'b0' else '否，当前未达到准入标准'}。",
        "28. 下游1×1训练：尚未执行，不能宣称超过当前GBSP。",
        "29. 阈值能否简化：以fixed-0.5、Otsu及平台结果为依据，不能仅凭最优扫描点宣称。",
        "30. 旧背景发现是否造成阈值困难：只有当B1-B4同时改善连续排序、背景q95与简单阈值时才成立；结论见准入审计。",
    ]
    lines = ["# GBSP-V2 正式结果报告", "", "## 最终选择", "", f"- 选择：**{selection['selected_name']}**", f"- 原因：{selection['reason']}", "", "## 主表", "", "| Variant | Pixel AP | AUROC | S@0.5 | Fw@0.5 | E@0.5 | MAE@0.5 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        if variant in cont and variant in hard50:
            lines.append(f"| {NAMES[variant]} | {_float(cont[variant],'pixel_AP'):.6f} | {_float(cont[variant],'pixel_AUROC'):.6f} | {_float(hard50[variant],'S_m'):.4f} | {_float(hard50[variant],'F_beta_w'):.4f} | {_float(hard50[variant],'E_mean'):.4f} | {_float(hard50[variant],'MAE'):.4f} |")
    lines.extend(["", "## 30项核验回答", "", *answers, "", "## 约束声明", "", "本报告中的GT只用于正式评测、污染审计与效果解释；B0–B5生成、Soft seed、Coreset、加权PCA和SWOR均不读取GT。20张结果不得用于取消全量预声明变体。", ""])
    (output / "GBSP_V2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> None:
    eval_dir, cache_root, output = map(_resolve, (args.eval_dir, args.variant_root, args.out_dir))
    output.mkdir(parents=True, exist_ok=True)
    for path in eval_dir.iterdir():
        if path.is_file() and path.suffix in (".csv", ".json"):
            target = output / path.name
            if path.resolve() != target.resolve(): shutil.copy2(path, target)
    audit = cache_root / "configuration_audit.json"
    if audit.is_file() and audit.resolve() != (output / audit.name).resolve(): shutil.copy2(audit, output / audit.name)
    continuous = _read_csv(eval_dir / "full6473_continuous_metrics.csv")
    binary050 = _read_csv(eval_dir / "full6473_binary_050.csv")
    binary058 = _read_csv(eval_dir / "full6473_binary_058.csv")
    residual = _read_csv(eval_dir / "residual_distribution.csv")
    if not all((continuous, binary050, binary058, residual)):
        raise RuntimeError("formal evaluation outputs are incomplete")
    selection = _select(continuous, binary050, binary058, residual)
    _write_json(output / "selected_configuration.json", selection)
    _write_csv(output / "downstream_1x1_results.csv", [{"status": "not_run", "reason": "training requires formal GBSP-V2 admission and is outside this evaluation stage", "baseline": "B0 Current GBSP", "candidate": selection["selected_variant"]}])
    _plots(eval_dir, output)
    _qualitative(cache_root, eval_dir, selection["selected_variant"], output)
    _report(eval_dir, selection, output)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--eval-dir", "--eval_dir", dest="eval_dir", required=True)
    value.add_argument("--variant-root", "--variant_root", dest="variant_root", required=True)
    value.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    return value


if __name__ == "__main__":
    analyze(parser().parse_args())
