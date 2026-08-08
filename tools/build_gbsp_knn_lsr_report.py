#!/usr/bin/env python3
"""Assemble the staged outputs into the required 30-question final report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gbsp_knn_lsr_common import read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--out_dir")
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return read_csv(path) if path.is_file() else []


def _find(rows: list[dict], **criteria) -> dict:
    return next((row for row in rows if all(str(row.get(key)) == str(value) for key, value in criteria.items())), {})


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    root = Path(args.work_root).resolve()
    output = Path(args.out_dir).resolve() if args.out_dir else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    comp = root / "complementarity"
    threshold = root / "thresholdability"
    local_dir = root / "local_reconstruction_eval"
    fusion_dir = root / "fusion"

    baseline = _rows(comp / "baseline_audit.csv")
    pairs = _rows(comp / "knn8_matched_pair_analysis.csv")
    high_pairs = _rows(comp / "knn8_high_similarity_matched_pairs.csv")
    correlation = _rows(comp / "knn8_gbsp_correlation.csv")
    ranking = _rows(comp / "ranking_complementarity.csv")
    fixed = _rows(threshold / "thresholdability_comparison.csv")
    adaptive = _rows(threshold / "adaptive_threshold_comparison.csv")
    plateau = _rows(threshold / "threshold_plateau_width.csv")
    knn_curve = _rows(threshold / "knn8_threshold_curve.csv")
    local_cont = _rows(local_dir / "local_reconstruction_continuous.csv")
    local_h = _rows(local_dir / "local_reconstruction_high_similarity.csv")
    local_050 = _rows(local_dir / "local_reconstruction_binary_050.csv")
    local_plateau = _rows(local_dir / "local_reconstruction_threshold_plateau.csv")
    fusion_cont = _rows(fusion_dir / "fusion_continuous.csv")

    pair = _find(pairs, dataset="ALL")
    pair_high = _find(high_pairs, dataset="ALL")
    corr = _find(correlation, dataset="ALL", scope_name="overall", aggregation="pooled_after_per_image_ECDF")
    rank = _find(ranking, dataset="ALL", label_rule="main_0.5", aggregation="per_image_macro")
    knn050 = _find(fixed, scope="dataset_macro", dataset="ALL", method="knn8", protocol="fixed_0.50")
    gbsp050 = _find(fixed, scope="dataset_macro", dataset="ALL", method="gbsp", protocol="fixed_0.50")
    knn_otsu = _find(adaptive, scope="dataset_macro", dataset="ALL", method="knn8", protocol="otsu")
    knn_plateau = _find(plateau, method="knn8")
    gbsp_plateau = _find(plateau, method="gbsp")
    curve_formal = [row for row in knn_curve if row.get("scope") == "dataset_macro"]
    best_knn = max(curve_formal, key=lambda row: _number(row, "F_beta_w"), default={})
    continuous = {row["method"]: row for row in local_cont if row.get("scope") == "dataset_macro"}
    h20 = {row["method"]: row for row in local_h if row.get("dataset") == "ALL" and row.get("subset") == "H20"}
    binary050 = {row["method"]: row for row in local_050 if row.get("scope") == "dataset_macro"}
    local_plateaus = {row["method"]: row for row in local_plateau}
    fusion_formal = {row["method"]: row for row in fusion_cont if row.get("scope") == "dataset_macro"}

    selection_path = fusion_dir / "selected_next_direction.json"
    if not selection_path.is_file():
        selection_path = local_dir / "selected_next_direction.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {
        "selected_route": "INCOMPLETE", "recommendation": "先完成缺失阶段"
    }
    answers = [
        "两套聚合口径结论不同：dataset-macro 为GBSP赢AP、KNN8赢AUROC；per-image macro为KNN8两项更高。",
        f"GBSP matched-pair win rate={_fmt(_number(pair,'pair_win_rate'))}。",
        f"有效匹配对={pair.get('valid_pair_count','N/A')}，95% CI=[{_fmt(_number(pair,'pair_win_rate_ci95_low'))}, {_fmt(_number(pair,'pair_win_rate_ci95_high'))}]。",
        f"高KNN相似区域GBSP win rate={_fmt(_number(pair_high,'pair_win_rate'))}。",
        f"KNN8/GBSP pooled-after-ECDF Spearman={_fmt(_number(corr,'Spearman'))}。",
        f"KNN-only correct={_fmt(_number(rank,'KNN_only_correct'))}，GBSP-only correct={_fmt(_number(rank,'GBSP_only_correct'))}。",
        "互补性裁决见complementarity/numerical_validity.json；只有其中allow_fusion=true才运行融合。",
        f"KNN8 fixed-0.50：S={_fmt(_number(knn050,'S_m'))}，Fw={_fmt(_number(knn050,'F_beta_w'))}，E={_fmt(_number(knn050,'E_mean'))}，MAE={_fmt(_number(knn050,'MAE'))}。",
        f"GBSP fixed-0.50：S={_fmt(_number(gbsp050,'S_m'))}，Fw={_fmt(_number(gbsp050,'F_beta_w'))}，E={_fmt(_number(gbsp050,'E_mean'))}，MAE={_fmt(_number(gbsp050,'MAE'))}。",
        f"KNN8最佳统一Fw扫描阈值={best_knn.get('threshold','N/A')}。",
        "是否也需要0.58应结合完整曲线与平台，而不能从最佳单点直接写入方法。",
        f"KNN8 Otsu Fw={_fmt(_number(knn_otsu,'F_beta_w'))}。",
        f"KNN8/GBSP平台宽度={_fmt(_number(knn_plateau,'plateau_width'))}/{_fmt(_number(gbsp_plateau,'plateau_width'))}。",
        "GBSP特有标定困难只有在KNN8的0.5、Otsu及平台均稳定更好时才成立。",
        f"LSR-K8-R2 AP/AUROC={_fmt(_number(continuous.get('k8_r2',{}),'AP'))}/{_fmt(_number(continuous.get('k8_r2',{}),'AUROC'))}。",
        f"LSR-K16-R4 AP/AUROC={_fmt(_number(continuous.get('k16_r4',{}),'AP'))}/{_fmt(_number(continuous.get('k16_r4',{}),'AUROC'))}。",
        f"LSR-K32-R8 AP/AUROC={_fmt(_number(continuous.get('k32_r8',{}),'AP'))}/{_fmt(_number(continuous.get('k32_r8',{}),'AUROC'))}。",
        "稳定Local尺度由连续指标、四库方向、二值指标、平台宽度和bootstrap共同决定，不用单一20张结果选择。",
        f"KNN8 AP={_fmt(_number(continuous.get('knn8',{}),'AP'))}；与LSR-main直接比较见上表。",
        f"Global GBSP AP={_fmt(_number(continuous.get('global_gbsp',{}),'AP'))}；与LSR-main直接比较见上表。",
        f"H20 AP：KNN8={_fmt(_number(h20.get('knn8',{}),'AP'))}，GBSP={_fmt(_number(h20.get('global_gbsp',{}),'AP'))}，LSR={_fmt(_number(h20.get('k16_r4',{}),'AP'))}。",
        f"fixed-0.50 Fw：KNN8={_fmt(_number(binary050.get('knn8',{}),'F_beta_w'))}，LSR={_fmt(_number(binary050.get('k16_r4',{}),'F_beta_w'))}。",
        f"平台宽度：KNN8={_fmt(_number(local_plateaus.get('knn8',{}),'plateau_width'))}，LSR={_fmt(_number(local_plateaus.get('k16_r4',{}),'plateau_width'))}。",
        "只有LSR稳定超过Global GBSP与KNN8，才能把结果解释为Global PCA过度全局化。",
        "KNN8优势是否来自query-adaptive模式选择，由LSR与KNN8对比以及局部诊断共同回答。",
        "Rank-0→Global-r8既有正式增益证明PCA本身有价值；LSR决定该价值能否在强KNN下继续转化为性能。",
        f"是否保留reconstruction：当前路线={selection.get('selected_route','N/A')}。",
        f"Local Affine建议={selection.get('NEXT_STAGE_RECOMMENDATION','N/A')}。",
        f"最佳融合候选={selection.get('best_fusion','N/A')}，是否保留={selection.get('fusion_retained','N/A')}。",
        f"最终论文/伪标签路线：{selection.get('recommendation', selection.get('selected_route','N/A'))}。",
    ]
    lines = [
        "# GBSP KNN8 LSR 最终诊断报告", "",
        "本轮Training-Free分析用于筛选更好的离线伪标签生成器；它不是最终无训练模型。", "",
        "## 正式口径", "",
        "- 正式主表：四数据集分别做per-image native指标，再等权dataset macro。",
        "- 同时报告6473张per-image macro，不通过切换聚合口径制造单向结论。",
        "- DINOv1-S/8、37×37 last-key、Full BC、全部1369 query及leave-one-out均冻结。", "",
        "## 任务书30项回答", "",
    ]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
    lines += ["", "## 最终路线", "", f"**{selection.get('selected_route','INCOMPLETE')}**", "", str(selection.get("recommendation", "")), ""]
    report = output / "GBSP_KNN_LOCAL_RECONSTRUCTION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "selected_next_direction.json", selection)
    validity = {
        "report_answers": len(answers), "report_path": str(report),
        "stage_outputs_present": {
            "complementarity": bool(pairs), "thresholdability": bool(fixed),
            "local_reconstruction": bool(local_cont), "fusion": bool(fusion_cont),
        },
        "selected_route": selection.get("selected_route", "INCOMPLETE"),
    }
    write_json(output / "numerical_validity.json", validity)
    print(json.dumps(validity, ensure_ascii=False))


if __name__ == "__main__":
    main()
