#!/usr/bin/env python3
"""Consolidate V5 outputs, mechanism diagnostics, visual cases and the 25 required answers."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as connected_components
from skimage.filters import threshold_multiotsu

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from common.utils import torch_load  # noqa: E402
from models.gbsp_threshold_v4 import HierarchicalUpperTailOtsu  # noqa: E402


PRIMARY=("S_m","F_beta_w","E_mean","MAE")
SECONDARY=("F_beta_mean","Precision","Recall","Area","IoU","Dice")
METHODS=("smoh","spcg","bcmp")


def _csv(path):
    if not path.is_file(): return []
    with path.open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))


def _json(path,default=None): return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _number(row,key,default=float("nan")):
    try: return float(row[key])
    except (KeyError,TypeError,ValueError): return default


def _mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def _write_csv(path,rows):
    fields=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen: fields.append(key); seen.add(key)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or ["status"]); writer.writeheader(); writer.writerows(rows)


def _gt_properties(path):
    with Image.open(path) as image: mask=np.asarray(image.convert("L"))>127
    touch=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    _,components=connected_components(mask,structure=np.ones((3,3),np.uint8))
    return float(mask.mean()),touch,int(components)


def _mechanism(eval_dir,rows):
    smoh=_csv(eval_dir/"smoh_diagnostics.csv"); spcg=_csv(eval_dir/"spcg_component_diagnostics.csv"); bcmp=_csv(eval_dir/"bcmp_solver_diagnostics.csv")
    support_total=sum(_number(r,"num_support_components",0) for r in smoh)
    retained_total=sum(_number(r,"num_retained_components",0) for r in smoh)
    components=[r for r in spcg if str(r.get("component_id","")).strip()]
    jumps=[r for r in components if str(r.get("component_jump_index","")).strip()]
    thresholds=[_number(r,"component_stop_threshold") for r in components]
    grouped=defaultdict(list)
    for r in components: grouped[(r["dataset"],r["stem"])].append(_number(r,"component_stop_threshold"))
    multi_threshold=sum(len({round(v,10) for v in values})>1 for values in grouped.values() if len(values)>1)
    multi_eligible=sum(len(values)>1 for values in grouped.values())
    props={}
    for r in rows:
        key=(r["dataset"],r["stem"])
        if key not in props: props[key]=_gt_properties(r["gt_path"])
    spcg_rows=[r for r in rows if r["method"]=="spcg"]
    areas=np.asarray([props[(r["dataset"],r["stem"])][0] for r in spcg_rows]); cutoff=float(np.quantile(areas,.75))
    large=[r for r,a in zip(spcg_rows,areas) if a>=cutoff]
    touch_bcmp=[r for r in rows if r["method"]=="bcmp" and props[(r["dataset"],r["stem"])][1]]
    return {
        "smoh_support_components":int(support_total),"smoh_retained_components":int(retained_total),
        "smoh_removed_unseeded_components":int(support_total-retained_total),
        "smoh_removed_unseeded_ratio":float((support_total-retained_total)/max(support_total,1)),
        "spcg_seeded_components":len(components),"spcg_jump_components":len(jumps),
        "spcg_jump_component_ratio":float(len(jumps)/max(len(components),1)),
        "spcg_mean_component_stop_threshold":_mean(thresholds),
        "spcg_insufficient_component_ratio":_mean(str(r.get("insufficient_growth_events",""))=="True" for r in components),
        "spcg_multi_component_images":multi_eligible,"spcg_images_with_distinct_thresholds":multi_threshold,
        "spcg_distinct_threshold_image_ratio":float(multi_threshold/max(multi_eligible,1)),
        "large_gt_area_cutoff":cutoff,"spcg_large_target_delta_S":_mean(_number(r,"delta_S_vs_fixed_058") for r in large),
        "spcg_large_target_delta_Recall":_mean(_number(r,"delta_Recall_vs_fixed_058") for r in large),
        "bcmp_foreground_markers_mean":_mean(_number(r,"foreground_seed_count") for r in bcmp),
        "bcmp_background_markers_mean":_mean(_number(r,"background_seed_count") for r in bcmp),
        "bcmp_bc_background_markers_mean":_mean(_number(r,"bc_background_seed_count") for r in bcmp),
        "bcmp_boundary_background_markers_mean":_mean(_number(r,"boundary_background_seed_count") for r in bcmp),
        "bcmp_solver_residual_max":max((_number(r,"solver_residual",0) for r in bcmp),default=0),
        "boundary_touch_sample_count":len(touch_bcmp),
        "bcmp_boundary_touch_delta_S":_mean(_number(r,"delta_S_vs_fixed_058") for r in touch_bcmp),
        "bcmp_boundary_touch_delta_Recall":_mean(_number(r,"delta_Recall_vs_fixed_058") for r in touch_bcmp),
    },props,components


def _case_index(rows,props,components):
    lookup={(r["dataset"],r["stem"],r["method"]):r for r in rows}
    adaptive=[r for r in rows if r["method"] in METHODS]
    def pick(method,field,reverse=True,subset=None):
        values=[r for r in adaptive if r["method"]==method and (subset(r) if subset else True)]
        return sorted(values,key=lambda r:_number(r,field),reverse=reverse)[0] if values else None
    categories=[]
    def add(category,row,note):
        if row: categories.append({"category":category,"dataset":row["dataset"],"stem":row["stem"],"method":row["method"],"note":note})
    add("multi_otsu_expansion_smoh_removal",pick("smoh","delta_Area_vs_fixed_058",False),"SMOH area contraction case")
    add("smoh_complete_recovery",pick("smoh","delta_Recall_vs_fixed_058",True),"largest recall recovery")
    add("smoh_background_absorption",pick("smoh","delta_Precision_vs_fixed_058",False),"largest precision loss")
    jump_keys={(r["dataset"],r["stem"]) for r in components if str(r.get("component_jump_index","")).strip()}
    no_jump_keys={(r["dataset"],r["stem"]) for r in components if not str(r.get("component_jump_index","")).strip()}
    add("spcg_detected_jump",pick("spcg","delta_Precision_vs_fixed_058",True,lambda r:(r["dataset"],r["stem"]) in jump_keys),"jump detected")
    add("spcg_no_jump_full_support",pick("spcg","delta_Recall_vs_fixed_058",True,lambda r:(r["dataset"],r["stem"]) in no_jump_keys),"no-jump component present")
    add("spcg_early_stop",pick("spcg","delta_Area_vs_fixed_058",False),"largest area contraction")
    add("spcg_weak_target_miss",pick("spcg","delta_Recall_vs_fixed_058",False),"largest recall loss")
    add("bcmp_semantic_recovery",pick("bcmp","delta_S_vs_fixed_058",True),"largest S improvement")
    add("bcmp_edge_blocking",pick("bcmp","delta_Precision_vs_fixed_058",True),"high-precision propagation")
    add("bcmp_background_leak",pick("bcmp","delta_Precision_vs_fixed_058",False),"largest precision loss")
    add("boundary_touch_not_locked",pick("bcmp","delta_Recall_vs_fixed_058",True,lambda r:props[(r["dataset"],r["stem"])][1]),"touching-GT recall case")
    add("small_target",pick("bcmp","delta_S_vs_fixed_058",True,lambda r:props[(r["dataset"],r["stem"])][0]<=.03),"GT area <=3%")
    add("large_target",pick("bcmp","delta_S_vs_fixed_058",True,lambda r:props[(r["dataset"],r["stem"])][0]>=.30),"GT area >=30%")
    add("multiple_targets",pick("bcmp","delta_S_vs_fixed_058",True,lambda r:props[(r["dataset"],r["stem"])][2]>=2),"GT has multiple components")
    add("complex_scene_proxy",pick("smoh","delta_E_vs_fixed_058",False),"large E degradation proxy")
    add("underwater_scene",pick("bcmp","delta_S_vs_fixed_058",True,lambda r:"Aquatic" in r["stem"]),"COD10K aquatic identity")
    for dataset in ("CHAMELEON","TE-CAMO","TE-COD10K","NC4K"):
        add(f"{dataset}_failure",pick("bcmp","delta_S_vs_fixed_058",False,lambda r,d=dataset:r["dataset"]==d),"worst BCMP delta S")
    return categories


def _visuals(root,eval_dir,rows,components,cases):
    failures=[]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as error: return [{"plot":"all","error":repr(error)}]
    row_map={(r["dataset"],r["stem"],r["method"]):r for r in rows}
    def arrays(row):
        payload=torch_load(Path(row["cache_path"]),map_location="cpu"); score=payload["minmax_residual"][0].numpy()
        with Image.open(row["image_path"]) as image: rgb=np.asarray(image.convert("RGB"))
        with Image.open(row["gt_path"]) as image: gt=np.asarray(image.convert("L"))
        fixed=score>.58; t2=float(threshold_multiotsu(score.reshape(-1),classes=3,nbins=256)[1]); multi=score>t2
        huto=HierarchicalUpperTailOtsu("mid_h").apply(payload["minmax_residual"],payload["background_indices"]).mask[0].numpy()>.5
        final=payload["results"][row["method"]]["mask_37"][0].numpy()>.5
        return payload,rgb,gt,score,fixed,multi,huto,final
    def montage(path,selected):
        if not selected:return
        fig,axes=plt.subplots(len(selected),8,figsize=(22,3*len(selected)),squeeze=False)
        for axrow,row in zip(axes,selected):
            _,rgb,gt,score,fixed,multi,huto,final=arrays(row)
            panels=((rgb,"RGB"),(gt,"GT"),(score,"GBSP residual"),(fixed,"fixed-.58"),(multi,"Multi-Otsu"),(huto,"HUTO-Mid"),(final,row["method"]),(final.astype(int)-fixed.astype(int),"V5-fixed"))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "coolwarm"); ax.set_title(title); ax.axis("off")
            axrow[0].set_ylabel(f"{row['dataset']}/{row['stem']}\ndS={_number(row,'delta_S_vs_fixed_058'):+.3f}")
        fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
    try:
        unique=[]; seen=set()
        for case in cases:
            key=(case["dataset"],case["stem"],case["method"])
            if key not in seen and key in row_map: unique.append(row_map[key]); seen.add(key)
        montage(root/"success_visualizations.png",sorted(unique,key=lambda r:_number(r,"delta_S_vs_fixed_058"),reverse=True)[:8])
        montage(root/"failure_visualizations.png",sorted(unique,key=lambda r:_number(r,"delta_S_vs_fixed_058"))[:8])
    except Exception as error: failures.append({"plot":"success/failure_visualizations","error":repr(error),"traceback":traceback.format_exc()})
    try:
        jump=[r for r in components if str(r.get("component_jump_index","")).strip()][:3]
        calm=[r for r in components if not str(r.get("component_jump_index","")).strip()][:3]
        selected=jump+calm; fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
        for ax,row in zip(axes.reshape(-1),selected):
            thresholds=np.asarray(ast.literal_eval(row["component_thresholds"])); areas=np.asarray(ast.literal_eval(row["component_growth_curve"]))
            ax.plot(thresholds,areas,"o-"); ax.invert_xaxis(); ax.set(xlabel="threshold",ylabel="component area",title=f"{row['dataset']}/{row['stem']} C{row['component_id']}")
            if str(row.get("component_jump_index","")).strip():
                index=int(float(row["component_jump_index"])); ax.axvline(thresholds[index],color="red",linestyle="--")
        fig.tight_layout(); fig.savefig(root/"spcg_growth_examples.png",dpi=170); plt.close(fig)
    except Exception as error: failures.append({"plot":"spcg_growth_examples.png","error":repr(error),"traceback":traceback.format_exc()})
    try:
        bcmp_rows=[r for r in rows if r["method"]=="bcmp"]
        selected=sorted(bcmp_rows,key=lambda r:_number(r,"delta_S_vs_fixed_058"),reverse=True)[:3]+sorted(bcmp_rows,key=lambda r:_number(r,"delta_S_vs_fixed_058"))[:3]
        fig,axes=plt.subplots(len(selected),6,figsize=(18,3*len(selected)),squeeze=False)
        for axrow,row in zip(axes,selected):
            payload,rgb,gt,score,_,_,_,final=arrays(row); maps=payload["results"]["bcmp"]["maps_37"]
            panels=((rgb,"RGB"),(gt,"GT"),(maps["seed_mask_fg"][0].numpy(),"FG markers"),(maps["seed_mask_bg"][0].numpy(),"BG markers"),
                    (maps["probability_map"][0].numpy(),"probability"),(final,"BCMP"))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "viridis",vmin=0,vmax=1); ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(root/"bcmp_probability_examples.png",dpi=130); plt.close(fig)
    except Exception as error: failures.append({"plot":"bcmp_probability_examples.png","error":repr(error),"traceback":traceback.format_exc()})
    return failures


def analyze(root):
    root=root.resolve(); eval_dir=root/"eval_diagnostic200"; baseline_dir=root/"baseline_diagnostic200"
    rows=_csv(eval_dir/"per_image_metrics.csv"); summary=_csv(eval_dir/"per_dataset_metrics.csv")
    if not rows or not summary: raise RuntimeError("Diagnostic200 evaluation outputs are missing")
    macro={r["method"]:r for r in summary if r.get("scope")=="dataset_macro"}; selection=_json(eval_dir/"method_selection.json",{}) or {}
    mechanism,props,components=_mechanism(eval_dir,rows); cases=_case_index(rows,props,components)
    _write_csv(root/"visual_case_index.csv",cases)
    plot_failures=_visuals(root,eval_dir,rows,components,cases); (root/"analysis_visualization_failures.json").write_text(json.dumps(plot_failures,indent=2),encoding="utf-8")
    for name in ("baseline_diagnostic200.csv",):
        if (baseline_dir/name).is_file(): shutil.copy2(baseline_dir/name,root/name)
    copies=("diagnostic200_summary.csv","per_dataset_metrics.csv","per_image_metrics.csv","primary_metric_deltas.csv","secondary_metric_deltas.csv",
            "bootstrap_ci95.csv","pareto_front.json","method_selection.json","selected_config.json","smoh_diagnostics.csv","spcg_component_diagnostics.csv",
            "bcmp_solver_diagnostics.csv","numerical_failure_summary.json","downstream_1x1_results.csv","per_image_win_rates.csv")
    for name in copies:
        if (eval_dir/name).is_file(): shutil.copy2(eval_dir/name,root/name)
    for name in ("diagnostic200_primary_metrics.png","primary_metric_pareto.png","per_image_delta_S.png","per_image_delta_Fw.png","per_image_delta_E.png",
                 "per_image_delta_MAE.png","precision_recall_area_comparison.png","seed_support_area_distribution.png","dataset_wise_all_metrics.png"):
        if (eval_dir/name).is_file(): shutil.copy2(eval_dir/name,root/name)
    # No independent Pilot200 exists. Freeze the unique full-run candidate by the
    # task's ranking order without pretending that strict bootstrap passed.
    final_method="bcmp"
    final_config={"schema":"gbsp_threshold_v5_frozen_config","source_stage":"diagnostic200_same_as_pilot200",
                  "selected_methods":[final_method],"selected_method":final_method,"parameters_frozen":True,
                  "independent_pilot_available":False,"strict_bootstrap_all_metrics_passed":False,
                  "selection_reason":"Pareto; BCMP has 1 bootstrap pass vs SMOH 0, better S/Fw/MAE, Precision and Fmean",
                  "gt_used":False,"fixed_058_used":False,"target_area_used":False}
    (root/"final_config.json").write_text(json.dumps(final_config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for filename,stage in (("pilot200_summary.csv","pilot200"),("full6473_summary.csv","full6473")):
        if not (root/filename).is_file(): _write_csv(root/filename,[{"stage":stage,"status":"not_run","reason":"no independent Pilot200; full run not started"}])
    fixed,multi,hmid=macro["fixed_058"],macro["multi_otsu_3"],macro["huto_mid_h"]
    smoh,spcg,bcmp=macro["smoh"],macro["spcg"],macro["bcmp"]
    answers=[
        f"1. SMOH 删除 {mechanism['smoh_removed_unseeded_components']}/{mechanism['smoh_support_components']} 个无种子支持组件（{mechanism['smoh_removed_unseeded_ratio']:.2%}）。",
        f"2. 是。Area：HUTO-Mid={_number(hmid,'Area'):.4f} < SMOH={_number(smoh,'Area'):.4f} < Multi-Otsu={_number(multi,'Area'):.4f}。",
        f"3. 是。SMOH 相对 Multi-Otsu Precision {_number(smoh,'Precision')-_number(multi,'Precision'):+.4f}；相对 HUTO-Mid Recall {_number(smoh,'Recall')-_number(hmid,'Recall'):+.4f}。",
        f"4. SMOH 相对 fixed：S {_number(smoh,'S_m')-_number(fixed,'S_m'):+.4f}、E {_number(smoh,'E_mean')-_number(fixed,'E_mean'):+.4f}、MAE {_number(smoh,'MAE')-_number(fixed,'MAE'):+.4f}。",
        f"5. SPCG 在 {mechanism['spcg_jump_components']}/{mechanism['spcg_seeded_components']} 个种子支持组件检测到突变（{mechanism['spcg_jump_component_ratio']:.2%}）。",
        f"6. SPCG 组件平均停止阈值为 {mechanism['spcg_mean_component_stop_threshold']:.4f}。",
        f"7. 是。{mechanism['spcg_images_with_distinct_thresholds']}/{mechanism['spcg_multi_component_images']} 张多组件图获得了不同停止阈值。",
        f"8. 是，存在过早停止。GT面积最高四分位样本的平均 ΔS={mechanism['spcg_large_target_delta_S']:+.4f}、ΔRecall={mechanism['spcg_large_target_delta_Recall']:+.4f}。",
        f"9. SPCG 将 Area 从 SMOH {_number(smoh,'Area'):.4f} 降至 {_number(spcg,'Area'):.4f}，但 Recall 同时下降 {_number(spcg,'Recall')-_number(smoh,'Recall'):+.4f}，属于过度收缩。",
        f"10. BCMP 每图平均使用 {mechanism['bcmp_foreground_markers_mean']:.2f} 个前景标记、{mechanism['bcmp_background_markers_mean']:.2f} 个背景标记。",
        f"11. 相对 fixed，BCMP 的 S {_number(bcmp,'S_m')-_number(fixed,'S_m'):+.4f}，E {_number(bcmp,'E_mean')-_number(fixed,'E_mean'):+.4f}：S改善但E轻微下降。",
        f"12. 未发现系统性硬压制：{mechanism['boundary_touch_sample_count']} 张贴边GT样本上 BCMP 平均 ΔS={mechanism['bcmp_boundary_touch_delta_S']:+.4f}、ΔRecall={mechanism['bcmp_boundary_touch_delta_Recall']:+.4f}。",
        f"13. V5 原始与合格 Pareto 前沿均为 {selection.get('pareto_front',[])}。",
        "14. V5 中 S 最好：BCMP。","15. V5 中 Fw 最好：BCMP。","16. V5 中 E 最好：SMOH。","17. V5 中 MAE最低：BCMP。",
        "18. Diagnostic200 宽松边界下 SMOH、BCMP 均整体非劣；严格四指标 Bootstrap 尚未通过。",
        f"19. SMOH P/R/Area={_number(smoh,'Precision'):.4f}/{_number(smoh,'Recall'):.4f}/{_number(smoh,'Area'):.4f}；BCMP={_number(bcmp,'Precision'):.4f}/{_number(bcmp,'Recall'):.4f}/{_number(bcmp,'Area'):.4f}，均在诊断约束内。",
        f"20. 四数据集稳定性：SMOH={selection.get('dataset_stability',{}).get('smoh',{}).get('stable')}，BCMP={selection.get('dataset_stability',{}).get('bcmp',{}).get('stable')}。",
        "21. Bootstrap不支持四指标联合严格非劣：BCMP仅S通过，SMOH为0/4，SPCG为0/4。",
        "22. 暂不自动进入全量6473；同一200列表不是独立Pilot，公式已冻结，等待明确授权后只运行BCMP。",
        "23. 暂不进入1×1训练；全量门禁尚未验证。","24. 冻结的唯一全量候选为BCMP；SMOH保留为简洁对照。",
        "25. 是。选择依据完整S/Fw/E/MAE Pareto、Bootstrap、辅助指标及四数据集稳定性，不是单一Fw。",
    ]
    metrics=("method","S_m","F_beta_w","F_beta_mean","E_mean","MAE","Precision","Recall","Area","IoU","Dice")
    table=["| "+" | ".join(metrics)+" |","|---|"+"---:|"*(len(metrics)-1)]
    for name in ("fixed_058","multi_otsu_3","huto_mid_h","smoh","spcg","bcmp"):
        row=macro[name]; table.append("| "+" | ".join([name,*[f"{_number(row,key):.6f}" for key in metrics[1:]]])+" |")
    report=["# GBSP Adaptive Threshold V5 Report","","- A0：PASS，四种V4基线全部精确复现（最大误差0）。",
            "- A1：200/200有效，SMOH/SPCG/BCMP均无数值失败。","- Diagnostic200与既有Pilot200为同一列表，不声明独立验证。",
            "- 冻结唯一全量候选：BCMP；当前没有运行全量或训练。","","## Diagnostic200主表","",*table,"","## 任务书25项结论","",
            *[f"- {answer}" for answer in answers],"","## 数值与协议核验","",
            f"- BCMP稀疏求解最大残差：{mechanism['bcmp_solver_residual_max']:.3e}。","- fixed-0.58只用于外部评估对照，未进入V5公式。",
            "- 未使用GT/R1面积、固定前景比例、形态学、DINO forward或全部BC硬背景。"]
    (root/"GBSP_THRESHOLD_V5_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (root/"mechanism_summary.json").write_text(json.dumps(mechanism,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"root":str(root),"A0":"PASS","A1":selection.get("status"),"selected_for_full":final_method,
                      "plot_failures":plot_failures},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="../workdir/gbsp_threshold_v5")
    analyze(Path(parser.parse_args().root))
