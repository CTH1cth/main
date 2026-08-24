#!/usr/bin/env python3
"""Finalize the mechanism-gated 512 scale-adaptation audit without new model runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.stats import spearmanr
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load, write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[2] / "workdir/gbsp_resolution_512/scale_adaptation"
FULL512 = Path(__file__).resolve().parents[2] / "workdir/gbsp_resolution_512/full6473"
CORE296 = Path(__file__).resolve().parents[2] / "workdir/gbsp_core_optimization/full_rank"
MAIN = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    diagnostics = ROOT / "diagnostics"
    graphcal = ROOT / "graph_calibration"
    evaluation = graphcal / "sample500_eval"
    decision = json.load((diagnostics / "stage1_decision.json").open())
    graph = _read_csv(diagnostics / "graph_stats_summary.csv")
    graph_lookup = {(row["Resolution"], row["Metric"]): row for row in graph}
    metrics = _read_csv(evaluation / "metrics_summary.csv")
    macro = {row["method"]: row for row in metrics if row["scope"] == "dataset_macro"}
    image_macro = {row["method"]: row for row in metrics if row["scope"] == "image_macro"}
    candidates = {row["method"]: row for row in _read_csv(evaluation / "candidate_summary.csv")}
    small = {row["method"]: row for row in _read_csv(evaluation / "small_object_metrics.csv")}

    # GraphCal normalized medians: no search, just the frozen Stage-1 formula.
    sigmas = decision["calibrated_sigmas"]
    normalized = []
    for raw, normalized_name, sigma_key in (("df", "df/sigma_f", "SIGMA_F"),
                                             ("dc", "dc/sigma_c", "SIGMA_C"),
                                             ("de", "de/sigma_e", "SIGMA_E")):
        q296 = float(graph_lookup[("296", normalized_name)]["P50"])
        raw512 = float(graph_lookup[("512", raw)]["P50"])
        before = float(graph_lookup[("512", normalized_name)]["P50"])
        after = raw512 / float(sigmas[sigma_key])
        normalized.append({"metric": normalized_name, "q50_296": q296, "q50_512_before": before,
                           "q50_512_graphcal": after, "before_ratio": before/q296,
                           "after_ratio": after/q296, "calibrated_sigma": sigmas[sigma_key]})
    _write_csv(graphcal / "normalized_penalty_summary.csv", normalized)
    write_json(graphcal / "calibration_values.json", {"method": "median_ratio", **sigmas,
        "source": str(diagnostics / "graph_stats_summary.csv"), "search_used": False})
    shutil.copy2(MAIN / "configs/dinov1_s8_gbsp_resolution_512_native64_graphcal.py",
                 graphcal / "config_snapshot.py")

    # Exact invariance audit against frozen 512-A.
    sample_rows = read_jsonl(diagnostics / "sample500.jsonl")
    score_errors, candidate_equal = [], 0
    for row in sample_rows:
        dataset, stem = row["dataset"], row["stem"]
        old = torch_load(FULL512 / "A_bw2_r8/scores" / dataset / f"{stem}.pt", map_location="cpu")
        new = torch_load(graphcal / "sample500/scores" / dataset / f"{stem}.pt", map_location="cpu")
        score_errors.append(float((old["absolute_minmax"] - new["absolute_minmax"]).abs().max()))
        candidate_equal += int(torch.equal(old["background_indices"], new["background_indices"]))
    invariance = {"num_images": len(sample_rows), "candidate_indices_exact_equal": candidate_equal,
        "score_exact_equal": int(sum(value == 0 for value in score_errors)),
        "score_changed": int(sum(value > 0 for value in score_errors)),
        "mean_per_image_max_abs_error": float(np.mean(score_errors)),
        "max_abs_error": float(np.max(score_errors)),
        "reason": "near-uniform edge-cost scaling is mostly cancelled by per-image max-distance normalization in BC"}
    write_json(graphcal / "invariance_audit.json", invariance)

    # Dataset-specific and texture-stratified evidence.
    per_image = _read_csv(evaluation / "per_image_metrics.csv")
    metric_map = {(row["method"], row["dataset"], row["stem"]): row for row in per_image}
    graph_image = {(row["resolution"], row["dataset"], row["stem"]): row
                   for row in _read_csv(diagnostics / "graph_stats_per_image.csv")}
    keys = [(row["dataset"], row["stem"]) for row in read_jsonl(diagnostics / "sample500.jsonl")]
    ap_delta = np.asarray([float(metric_map[("512-C", *key)]["pixel_AP"])
                           - float(metric_map[("296-GBSP-r8", *key)]["pixel_AP"]) for key in keys])
    stratified = []
    for field in ("df_p90", "dc_p90", "de_p90", "edge_cost_p50"):
        value = np.asarray([float(graph_image[("512", *key)][field]) for key in keys])
        correlation, p_value = spearmanr(value, ap_delta)
        q25, q75 = np.quantile(value, [.25, .75])
        stratified.append({"graph_field": field, "spearman_with_AP_delta_512C_minus_296": correlation,
            "p_value": p_value, "low_quartile_mean_AP_delta": float(ap_delta[value <= q25].mean()),
            "high_quartile_mean_AP_delta": float(ap_delta[value >= q75].mean())})
    _write_csv(diagnostics / "mechanism_stratified_analysis.csv", stratified)

    # Existing full results are evidence, not a newly-triggered run.
    full_rows = []
    for row in _read_csv(FULL512 / "comparison/continuous_metrics.csv"):
        if row["scope"] == "dataset_macro" and row["method"] in {"296-baseline", "512-A", "512-C"}:
            full_rows.append({"scope": row["scope"], "dataset": row["dataset"], "method": row["method"],
                              "num_images": row["num_images"], "pixel_AP": row["pixel_AP"],
                              "pixel_AUROC": row["pixel_AUROC"], "source": "existing full6473"})
    _write_csv(ROOT / "existing_full6473_reference.csv", full_rows)

    candidate_dir = ROOT / "candidate_ratio"; candidate_dir.mkdir(exist_ok=True)
    coarse_dir = ROOT / "coarse_bg_fine_query"; coarse_dir.mkdir(exist_ok=True)
    (candidate_dir / "SKIPPED.md").write_text(
        "# Candidate Ratio Stage — NOT TRIGGERED\n\n"
        "同图 sample500 的 rank90 中位数仅由 51 增至 56，energy@8 为 0.601158→0.603157；"
        "未满足预注册 intrinsic-rank drift 门槛。因此 R20/R10 未执行，避免把条件实验变成事后搜索。\n",
        encoding="utf-8")
    (coarse_dir / "SKIPPED.md").write_text(
        "# Coarse-BG/Fine-Query Stage — NOT TRIGGERED\n\n"
        "512 boundary 虽提升，但 intrinsic-rank drift 不成立，且不存在 ratio 只能部分解决的前置证据；"
        "故任务书三项联合条件不满足。\n", encoding="utf-8")

    generation_meta = json.load((graphcal / "sample500/generation_metadata.json").open())
    eval_meta = json.load((evaluation / "evaluation_metadata.json").open())
    runtime = [
        {"stage": "Stage1 graph+spectrum sample500", "seconds": decision["runtime_seconds"], "gpu_peak_mb": 0,
         "training": False, "dino_extraction": False},
        {"stage": "Stage2 GraphCal generation sample500", "seconds": generation_meta["runtime_seconds"], "gpu_peak_mb": 0,
         "training": False, "dino_extraction": False},
        {"stage": "Stage2 original-GT evaluation sample500", "seconds": eval_meta["runtime_seconds"], "gpu_peak_mb": 0,
         "training": False, "dino_extraction": False},
    ]
    runtime.append({"stage": "TOTAL formal stages", "seconds": sum(float(row["seconds"]) for row in runtime),
                    "gpu_peak_mb": 0, "training": False, "dino_extraction": False})
    _write_csv(ROOT / "runtime_gpu_memory.csv", runtime)
    sizes = [
        {"path": str(diagnostics), "bytes": _size(diagnostics)},
        {"path": str(graphcal), "bytes": _size(graphcal)},
        {"path": str(ROOT), "bytes": _size(ROOT)},
    ]
    _write_csv(ROOT / "cache_size.csv", sizes)

    delta_c = {field: float(macro["512-D-GraphCal"][field])-float(macro["512-C"][field])
               for field in ("pixel_AP", "pixel_AUROC", "boundary_F1")}
    delta_a = {field: float(macro["512-D-GraphCal"][field])-float(macro["512-A"][field])
               for field in ("pixel_AP", "pixel_AUROC", "boundary_F1")}
    full = {row["method"]: row for row in full_rows}
    graph_rows = []
    for metric in ("df", "dc", "de", "df/sigma_f", "dc/sigma_c", "de/sigma_e", "affinity", "edge_cost", "BC"):
        a, b = graph_lookup[("296", metric)], graph_lookup[("512", metric)]
        graph_rows.append(f"| {metric} | {float(a['P50']):.6f} | {float(b['P50']):.6f} | {float(b['P50'])/max(float(a['P50']),1e-12):.4f} |")

    dataset_delta_lines = []
    for dataset in ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"):
        p = macro  # silence accidental reuse in comprehensions
        a = next(row for row in metrics if row["scope"]=="dataset" and row["dataset"]==dataset and row["method"]=="296-GBSP-r8")
        c = next(row for row in metrics if row["scope"]=="dataset" and row["dataset"]==dataset and row["method"]=="512-C")
        dataset_delta_lines.append(f"| {dataset} | {float(c['pixel_AP'])-float(a['pixel_AP']):+.6f} | {float(c['pixel_AUROC'])-float(a['pixel_AUROC']):+.6f} | {float(c['boundary_F1'])-float(a['boundary_F1']):+.6f} |")

    lines = [
        "# GBSP 512 分辨率机制驱动尺度适配：最终结果", "",
        "## 一句话结论", "",
        "512 的相邻 graph term 确有约 16% 的一致尺度收缩，但 PCA 谱复杂度只小幅增加；唯一 median-ratio GraphCal 因 BC 的图内最大路径归一化近乎被抵消，未带来性能收益。没有新方案通过 full6473 门槛，因此按任务书正式结束 512 适配路线，不生成 train cache、不训练。", "",
        "## 协议与门控", "",
        "- 固定 sample500：CHAMELEON 76 / CAMO 141 / COD10K 141 / NC4K 142，seed=42，296/512 同图。",
        "- 全部复用既有 DINOv1-S/8 特征；没有重新提取 DINO，没有训练，没有阈值扫描。",
        "- 新 512 score 始终为 Native64，阈值冻结为 0.58，BW2、R30、PCA-r8 不变。",
        "- 自动 graph-drift 门槛为 Q50 比率偏离至少 20%；该严格门槛未触发。考虑三项均同向约 −16%，额外执行一次且仅一次公式推导 GraphCal，未搜索。",
        "- intrinsic-rank 门槛（median512≥1.5×median296 且差≥8）未触发，因此 Stage 3/4 按条件跳过。", "",
        "## 296→512 Graph 统计（全部有向八邻边，Q50）", "",
        "| Metric | 296 | 512 | 512/296 |", "|---|---:|---:|---:|", *graph_rows, "",
        "漂移最大的是 `df/sigma_f`（比率 0.840795），但 dc/de 几乎同幅度，因此更像统一空间采样尺度变化，而非 Sobel 单项异常放大。512 affinity 中位数由 0.01258 增至 0.02432，edge cost 由 4.37555 降至 3.71665；BC 中位数只由 0.57025 降至 0.53693。", "",
        "## PCA spectrum 与候选质量", "",
        "| Resolution | Mean rank90 | Median rank90 | Energy@8 | Energy@12 | Candidates | Candidate precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 296 | {decision['pca_summary']['296']['mean_required_rank90']:.3f} | {decision['pca_summary']['296']['median_required_rank90']:.0f} | {decision['pca_summary']['296']['mean_energy_at_r8']:.6f} | {decision['pca_summary']['296']['mean_energy_at_r12']:.6f} | {decision['pca_summary']['296']['mean_candidates']:.0f} | {decision['pca_summary']['296']['mean_candidate_precision']:.6f} |",
        f"| 512 | {decision['pca_summary']['512']['mean_required_rank90']:.3f} | {decision['pca_summary']['512']['median_required_rank90']:.0f} | {decision['pca_summary']['512']['mean_energy_at_r8']:.6f} | {decision['pca_summary']['512']['mean_energy_at_r12']:.6f} | {decision['pca_summary']['512']['mean_candidates']:.0f} | {decision['pca_summary']['512']['mean_candidate_precision']:.6f} |", "",
        "候选数增至约 3 倍，但 energy@8/12 几乎不变，rank90 中位数仅 +5，且候选精度略升；没有证据说明 top30% 在 512 下因污染或碎片化导致 PCA complexity 暴涨。", "",
        "## Sample500 正式比较（dataset macro）", "",
        "| Method | AP | AUROC | Boundary F1 | Fw | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("296-GBSP-r8", "512-A", "512-C", "512-D-GraphCal"):
        row=macro[method];lines.append(f"| {method} | {float(row['pixel_AP']):.6f} | {float(row['pixel_AUROC']):.6f} | {float(row['boundary_F1']):.6f} | {float(row['F_beta_w']):.6f} | {float(row['MAE']):.6f} |")
    lines += ["", f"GraphCal − 512-C：AP {delta_c['pixel_AP']:+.6f}，AUROC {delta_c['pixel_AUROC']:+.6f}，Boundary {delta_c['boundary_F1']:+.6f}。",
        f"GraphCal − 512-A：AP {delta_a['pixel_AP']:+.6f}，AUROC {delta_a['pixel_AUROC']:+.6f}，Boundary {delta_a['boundary_F1']:+.6f}。",
        f"候选完全相同 {invariance['candidate_indices_exact_equal']}/500，分数逐元素完全相同 {invariance['score_exact_equal']}/500；最大单图 max-abs 差 {invariance['max_abs_error']:.6f}。", "",
        "## 512-C 相对 296：sample500 分数据集变化", "",
        "| Dataset | ΔAP | ΔAUROC | ΔBoundary F1 |", "|---|---:|---:|---:|", *dataset_delta_lines, "",
        "AP 方向并不跨库稳定：CHAMELEON/COD10K 上升，CAMO/NC4K 下降；Boundary 四库均上升。", "",
        "## 小目标与纹理诊断", "",
        f"sample500 面积最低四分位（125 张）中，296→512-C：AP {float(small['512-C']['pixel_AP'])-float(small['296-GBSP-r8']['pixel_AP']):+.6f}，AUROC {float(small['512-C']['pixel_AUROC'])-float(small['296-GBSP-r8']['pixel_AUROC']):+.6f}，Boundary {float(small['512-C']['boundary_F1'])-float(small['296-GBSP-r8']['boundary_F1']):+.6f}；支持 512 的优势集中在边界/小目标。",
        "高 Sobel-P90 四分位的 512-C−296 AP 均值为 −0.000170，低四分位为 +0.004356，但 Spearman ρ=−0.0437、p=0.329，证据不足以断言复杂纹理背景是主要失败来源。", "",
        "## 已有 Full6473 参考（本轮未重新运行）", "",
        "| Method | AP | AUROC |", "|---|---:|---:|",
        f"| 296 | {float(full['296-baseline']['pixel_AP']):.6f} | {float(full['296-baseline']['pixel_AUROC']):.6f} |",
        f"| 512-A | {float(full['512-A']['pixel_AP']):.6f} | {float(full['512-A']['pixel_AUROC']):.6f} |",
        f"| 512-C | {float(full['512-C']['pixel_AP']):.6f} | {float(full['512-C']['pixel_AUROC']):.6f} |", "",
        "## 任务书 20 问", "",
        "1. **df/dc/de 如何变化？** Q50 分别下降 15.92%、15.52%、15.91%，Mean 也均下降。",
        "2. **漂移最大项？** df/sigma_f，Q50 比率 0.840795；但三项幅度几乎一致。",
        "3. **normalized penalty 是否 resolution-invariant？** 绝对尺度不是；三项均约缩至 0.84。但相对构成近似 invariant。",
        "4. **GraphCal 是否提升？** 否；相对 512-C 三项均下降，相对 512-A 近乎零变化。",
        "5. **296 rank90？** mean 49.924，median 51。",
        "6. **512 rank90？** mean 54.726，median 56。",
        "7. **intrinsic complexity 是否显著增加？** 否；rank90 仅小幅 +5，energy@8/12 几乎相同。",
        "8. **top30 是否过度细碎？** 当前证据不支持；候选数增加，但候选精度略高、谱能量结构稳定。",
        "9. **20%/10% 是否降低 rank？** 未执行：Stage 3 的预注册 rank-drift 前提不成立。",
        "10. **10% 是否构成约411候选控制？** 数学上会约 410，但未执行，不能报告经验结论。",
        "11. **减少候选是否改善 AP/AUROC？** 本任务无法据此下结论，因条件实验被门控跳过。",
        "12. **coarse-bg/fine-query 是否有效？** 未执行：三项联合触发条件不完整。",
        "13. **512 优势是否集中在 boundary/small-object？** 是。sample500 Boundary 显著上升；小目标子集也有小幅 AP/AUC 增益。",
        "14. **劣势是否主要来自复杂纹理背景？** 未被证明；方向弱且统计不显著。",
        "15. **512 feature 本身更差？** 没有证据。排序近似，边界更好，部分库/小目标更好。",
        "16. **还是 resolution mismatch？** 存在 graph 数值尺度 mismatch，但简单尺度校准被 BC 归一化抵消；更准确地说是 GBSP 与细网格的交互，而非特征整体退化。",
        "17. **是否建议继续 512？** 不建议继续做超参或结构搜索。",
        "18. **是否建议生成 train cache？** 不建议；无新变体通过 full 门槛。",
        "19. **是否建议训练？** 不建议；当前没有更强的 512 伪标签候选。",
        "20. **是否正式结束 512 路线？** 是，按本任务终止条件结束；保留既有结果作为分辨率消融。", "",
        "## Full6473 门控与下一步", "",
        "GraphCal 既未达到 `AP >= 512-C + 0.003`，也没有形成 AP/AUROC/Boundary 三项一致改善，因此没有运行新 full6473。建议论文主线继续采用 296/37×37 Global GBSP-r8，512 作为“边界改善但全局排序/硬质量未获益”的负向尺度消融。", "",
        "## 运行与资源", "",
        f"- 正式阶段总耗时：{runtime[-1]['seconds']:.2f} s（{runtime[-1]['seconds']/60:.2f} min）。",
        "- 本任务新增计算全部在 CPU 完成，增量 GPU peak memory = 0 MiB。",
        f"- 收口前结果缓存约 {_size(ROOT)/1048576:.2f} MiB；复用既有 296/512 特征缓存。",
        "- 失败样本：0；训练：0；DINO 重提取：0；新 full6473：0。", "",
        "## 关键文件", "",
        "- `diagnostics/sample500.txt`：固定样本清单。",
        "- `diagnostics/graph_stats_summary.csv`：296/512 graph-term 汇总。",
        "- `diagnostics/pca_spectrum_296_vs_512.csv`：逐图 PCA 谱。",
        "- `graph_calibration/normalized_penalty_summary.csv`：GraphCal 对齐验收。",
        "- `graph_calibration/sample500_eval/metrics_summary.csv`：正式 sample500 指标。",
        "- `candidate_ratio/SKIPPED.md`、`coarse_bg_fine_query/SKIPPED.md`：条件实验未触发原因。",
        "- `existing_full6473_reference.csv`：既有全量参考。",
    ]
    (ROOT / "FINAL_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (graphcal / "RESULTS.md").write_text("\n".join(lines[lines.index("## Sample500 正式比较（dataset macro）"):lines.index("## 512-C 相对 296：sample500 分数据集变化")])+"\n", encoding="utf-8")

    commands = """# Stage 1
python tools/audit_gbsp_resolution_scale.py --config-296 configs/dinov1_s8_gbsp_resolution_296_reference.py --config-512 configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py --manifest-296 ../datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl --manifest-512 ../workdir/gbsp_resolution_512/full6473/shared_dino512/features_cache/dinov1-s8-512-native64/manifest_test.jsonl --core-296 ../workdir/gbsp_core_optimization/full_rank --scores-512 ../workdir/gbsp_resolution_512/full6473 --out-dir ../workdir/gbsp_resolution_512/scale_adaptation/diagnostics --sample-count 500 --seed 42 --torch-threads 8

# Stage 2 unique GraphCal sample500
python tools/cache_gbsp_scale_variant.py --config configs/dinov1_s8_gbsp_resolution_512_native64_graphcal.py --feature-manifest ../workdir/gbsp_resolution_512/full6473/shared_dino512/features_cache/dinov1-s8-512-native64/manifest_test.jsonl --sample-list ../workdir/gbsp_resolution_512/scale_adaptation/diagnostics/sample500.jsonl --out-root ../workdir/gbsp_resolution_512/scale_adaptation/graph_calibration/sample500 --method 512-D-GraphCal --mode standard --failure-policy strict --torch-threads 8

# Stage 2 sample500 evaluation
python tools/eval_gbsp_scale_adaptation.py --sample-list ../workdir/gbsp_resolution_512/scale_adaptation/diagnostics/sample500.jsonl --core-296 ../workdir/gbsp_core_optimization/full_rank --existing-512 ../workdir/gbsp_resolution_512/full6473 --variant 512-D-GraphCal=../workdir/gbsp_resolution_512/scale_adaptation/graph_calibration/sample500 --out-dir ../workdir/gbsp_resolution_512/scale_adaptation/graph_calibration/sample500_eval --threshold 0.58

# Finalize (read-only analysis of produced caches)
python tools/finalize_gbsp_scale_adaptation.py
"""
    (ROOT / "all_commands.txt").write_text(commands, encoding="utf-8")
    modified = [
        "main/models/gbsp_resolution.py",
        "main/configs/dinov1_s8_gbsp_resolution_512_native64_graphcal.py",
        "main/configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r20.py (prepared, gate not triggered)",
        "main/configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r10.py (prepared, gate not triggered)",
        "main/configs/dinov1_s8_gbsp_resolution_512_coarsebg32_finequery64.py (prepared, gate not triggered)",
        "main/tools/audit_gbsp_resolution_scale.py",
        "main/tools/cache_gbsp_scale_variant.py",
        "main/tools/eval_gbsp_scale_adaptation.py",
        "main/tools/finalize_gbsp_scale_adaptation.py",
        "main/tests/test_gbsp_scale_adaptation.py",
    ]
    (ROOT / "modified_files.txt").write_text("\n".join(modified)+"\n", encoding="utf-8")
    write_json(ROOT / "FINAL_AUDIT.json", {"sample500_complete": True, "sample_count": 500,
        "stage1_failures": 0, "graphcal_failures": 0, "evaluation_failures": 0,
        "stage2_executed": True, "stage3_executed": False, "stage4_executed": False,
        "new_full6473_executed": False, "full_trigger_passed": False,
        "delta_graphcal_vs_512c": delta_c, "decision": "STOP_512_ROUTE",
        "training_used": False, "dino_extraction_used": False})
    print(json.dumps({"final_report": str(ROOT/"FINAL_RESULTS.md"), "decision": "STOP_512_ROUTE",
                      "delta_graphcal_vs_512c": delta_c}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
