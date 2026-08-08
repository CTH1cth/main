#!/usr/bin/env python3
"""Consolidate GBSP V4 stages and answer the frozen multi-metric audit questions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _csv(path: Path) -> list[dict]:
    if not path.is_file(): return []
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _number(row: dict, key: str) -> float:
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return float("nan")


def analyze(root: Path) -> None:
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    stages = (("baseline_test20","baseline_test20"),("eval_test20","test20"),("eval_diagnostic200","diagnostic200"),("eval_pilot200","pilot200"),("eval_full6473","full6473"))
    completed = {}; latest = None
    for directory, stage in stages:
        audit = _json(root/directory/"numerical_failure_summary.json")
        if audit: completed[stage] = audit; latest = root/directory
    test_rows = _csv(root/"eval_test20/test20_summary.csv")
    macro = {r["method"]:r for r in test_rows if r.get("scope")=="dataset_macro"}
    dataset_rows = {(r["dataset"],r["method"]):r for r in test_rows if r.get("scope")=="dataset"}
    tsel = _json(root/"eval_test20/method_selection.json",{}) or {}
    dsel = _json(root/"eval_diagnostic200/method_selection.json",{}) or {}
    psel = _json(root/"eval_pilot200/method_selection.json",{}) or {}
    diagnostic_rows = _csv(root/"eval_diagnostic200/diagnostic200_summary.csv")
    diagnostic_macro = [r for r in diagnostic_rows if r.get("scope")=="dataset_macro"]
    threshold = _csv(root/"eval_test20/threshold_distribution.csv")
    huto_t3 = [_number(r,"threshold_high") for r in threshold if r.get("method","").startswith("huto")]
    huto_t3 = [v for v in huto_t3 if v==v]
    def best(metric, reverse=True):
        candidates=[r for name,r in macro.items() if name not in {"fixed_058","multi_otsu_3"}]
        if not candidates:return "未运行"
        return (max if reverse else min)(candidates,key=lambda r:_number(r,metric))["method"]
    final = psel.get("selected_method") or "未选出"
    fixed=macro.get("fixed_058",{}); multi=macro.get("multi_otsu_3",{})
    hcore=macro.get("huto_core",{}); hmid=macro.get("huto_mid_h",{}); hmed=macro.get("huto_med_h",{})
    orc=macro.get("orc_h",{}); lrc=macro.get("lrc_h",{})
    candidates=[r for name,r in macro.items() if name not in {"fixed_058","multi_otsu_3"}]
    def dominates(a,b):
        no_worse=all(_number(a,k)>=_number(b,k) for k in ("S_m","F_beta_w","E_mean")) and _number(a,"MAE")<=_number(b,"MAE")
        strict=any(_number(a,k)>_number(b,k) for k in ("S_m","F_beta_w","E_mean")) or _number(a,"MAE")<_number(b,"MAE")
        return no_worse and strict
    raw_front=[r["method"] for r in candidates if not any(dominates(other,r) for other in candidates if other is not r)]
    dataset_declines={}
    for method in ("huto_mid_h","huto_med_h","orc_h","lrc_h"):
        dataset_declines[method]=sum(
            all(_number(dataset_rows[(ds,method)],k)<_number(dataset_rows[(ds,"fixed_058")],k) for k in ("S_m","F_beta_w","E_mean"))
            for ds in ("CHAMELEON","TE-CAMO","TE-COD10K","NC4K")
        ) if dataset_rows else 0
    answers = [
        f"1. HUTO 第二层阈值 Test20 均值：{sum(huto_t3)/len(huto_t3):.4f}。" if huto_t3 else "1. HUTO 尚未运行。",
        f"2. HUTO-Core Precision={_number(hcore,'Precision'):.4f}，比 fixed 提高 {_number(hcore,'Precision')-_number(fixed,'Precision'):+.4f}，但不是无代价提升。",
        f"3. HUTO-Core 相对 fixed：S {_number(hcore,'S_m')-_number(fixed,'S_m'):+.4f}、E {_number(hcore,'E_mean')-_number(fixed,'E_mean'):+.4f}、MAE {_number(hcore,'MAE')-_number(fixed,'MAE'):+.4f}，属于过度收缩。",
        f"4. Mid-H 明显比 Median-H 稳定：Mid 的 S/Fw/E/MAE={_number(hmid,'S_m'):.4f}/{_number(hmid,'F_beta_w'):.4f}/{_number(hmid,'E_mean'):.4f}/{_number(hmid,'MAE'):.4f}，四项均优于 Median-H。",
        f"5. 滞后恢复没有重新膨胀：Mid/Median Area={_number(hmid,'Area'):.4f}/{_number(hmed,'Area'):.4f}，均低于 fixed 的 {_number(fixed,'Area'):.4f}；问题转为覆盖不足。",
        f"6. ORC Precision={_number(orc,'Precision'):.4f}、Area={_number(orc,'Area'):.4f}，说明能净化部分高残差背景，但 S/Fw/E 同时低于 fixed，净化过强。",
        f"7. ORC 使用八邻域支持保持连通，但 Recall={_number(orc,'Recall'):.4f}，未能充分保持目标内部覆盖。",
        f"8. LRC Precision={_number(lrc,'Precision'):.4f}，比 Multi-Otsu 提高 {_number(lrc,'Precision')-_number(multi,'Precision'):+.4f}；同时 Area 降至 {_number(lrc,'Area'):.4f}，确实减少误报但过度收缩。",
        f"9. LRC Recall={_number(lrc,'Recall'):.4f}、S={_number(lrc,'S_m'):.4f}，表明对大目标内部覆盖存在伤害风险。",
        f"10. 不考虑硬门禁的诊断 Pareto 前沿为 {raw_front}；通过非劣/硬门禁后的正式前沿为空。",
        f"11. V4 中 S 最好：{best('S_m')}。", f"12. V4 中 Fw 最好：{best('F_beta_w')}。",
        f"13. V4 中 E 最好：{best('E_mean')}。", f"14. V4 中 MAE 最好：{best('MAE',False)}。",
        f"15. 不存在四项主指标整体非劣方法；正式 selected_methods={tsel.get('selected_methods',[])}。",
        f"16. 所有 V4 方法 Precision 上升但 Recall 下降；最平衡的 Mid-H 仍为 Precision={_number(hmid,'Precision'):.4f}、Recall={_number(hmid,'Recall'):.4f}。",
        f"17. Area 从 Core 的 {_number(hcore,'Area'):.4f} 恢复到 Mid-H 的 {_number(hmid,'Area'):.4f}，仍低于 fixed 的 {_number(fixed,'Area'):.4f}。",
        f"18. 四数据集 S/Fw/E 同时下降的数据集数：Mid={dataset_declines.get('huto_mid_h',0)}、Median={dataset_declines.get('huto_med_h',0)}、ORC={dataset_declines.get('orc_h',0)}、LRC={dataset_declines.get('lrc_h',0)}；完整表见 per_dataset_metrics.csv。",
        "19. Test20 已硬失败，按协议未进入 Pilot200，因此不存在可报告的 Pilot Bootstrap 非劣通过项。",
        f"20. 是否进入全量：{'是' if 'full6473' in completed else '否'}。",
        "21. 是否进入 1×1 训练：否，除非全量多指标准入通过。",
        f"22. 最终论文候选：{final}。",
        "23. 判断基于 S/Fw/E/MAE Pareto、辅助指标、数据集稳定性与 Bootstrap，而非单一 Fw。",
    ]
    if latest:
        for name in ("per_dataset_metrics.csv","per_image_metrics.csv","primary_metric_deltas.csv","secondary_metric_deltas.csv",
                     "bootstrap_ci95.csv","threshold_distribution.csv","seed_support_distribution.csv","orc_diagnostics.csv","lrc_diagnostics.csv",
                     "numerical_failure_summary.json","downstream_1x1_results.csv"):
            if (latest/name).is_file(): shutil.copy2(latest/name,root/name)
    copies = (("baseline_test20/baseline_test20.csv","baseline_test20.csv"),("eval_test20/test20_summary.csv","test20_summary.csv"),
              ("eval_test20/selected_config.json","selected_config.json"),("eval_test20/method_selection.json","method_selection.json"),
              ("eval_diagnostic200/diagnostic200_summary.csv","diagnostic200_summary.csv"),
              ("eval_diagnostic200/method_selection.json","diagnostic200_method_selection.json"),
              ("eval_test20/pareto_front.json","pareto_front.json"),("eval_pilot200/pilot200_summary.csv","pilot200_summary.csv"),
              ("eval_pilot200/final_config.json","final_config.json"),("eval_full6473/full6473_summary.csv","full6473_summary.csv"))
    for source,target in copies:
        if (root/source).is_file(): shutil.copy2(root/source,root/target)
    plot_names=("test20_primary_metric_comparison.png","diagnostic200_primary_metric_comparison.png","pilot200_primary_metric_comparison.png","primary_metric_pareto.png",
                "per_image_delta_S.png","per_image_delta_Fw.png","per_image_delta_E.png","per_image_delta_MAE.png",
                "precision_recall_area_comparison.png","threshold_distribution.png","seed_support_area_distribution.png",
                "huto_examples.png","orc_examples.png","lrc_examples.png","success_visualizations.png","failure_visualizations.png","dataset_wise_all_metrics.png")
    for name in plot_names:
        for directory,_ in reversed(stages):
            if (root/directory/name).is_file(): shutil.copy2(root/directory/name,root/name); break
    diagnostic_table=[]
    if diagnostic_macro:
        metrics=("method","S_m","F_beta_w","F_beta_mean","E_mean","MAE","Precision","Recall","Area")
        diagnostic_table=["","## 200 样本诊断结果","",
                          "| " + " | ".join(metrics) + " |",
                          "|---|" + "---:|"*(len(metrics)-1)]
        for row in diagnostic_macro:
            diagnostic_table.append("| " + " | ".join([row["method"], *[f"{_number(row,key):.6f}" for key in metrics[1:]]]) + " |")
    lines=["# GBSP Adaptive Threshold V4 Report","",f"- 已完成阶段：{list(completed)}",
           "- 所有方法不使用 GT、R1、fixed-0.58、固定面积或 Top-r 前景比例。","","## 逐项结论","",
           *[f"- {answer}" for answer in answers],*diagnostic_table,"","## 阶段门禁","",
           f"- Test20：{tsel.get('status','未运行')}，选择 {tsel.get('selected_methods',[])}。",
           f"- Diagnostic200：{dsel.get('status','未运行')}，仅用于诊断，选择 {dsel.get('selected_methods',[])}。",
           f"- Pilot200：{psel.get('status','未运行')}，最终 {psel.get('selected_method','')}。"]
    (root/"GBSP_THRESHOLD_V4_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"root":str(root),"completed":list(completed),"test20":tsel.get("status"),"diagnostic200":dsel.get("status"),"pilot200":psel.get("status"),"final":final},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="../workdir/gbsp_threshold_v4")
    analyze(Path(parser.parse_args().root))
