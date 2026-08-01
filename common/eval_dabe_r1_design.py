#!/usr/bin/env python3
"""Formal original-GT evaluation and audits for R1-Design-v1."""

from __future__ import annotations

import argparse, csv, json, math, os, resource, shutil, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw

if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_rank_calibration import average_percentile_rank  # noqa: E402
from common.eval_dabe_background_null import (MetricAggregate, _difference_panel, _git_metadata, _gray_panel, _load_gt, _manifest_map, _markdown_table, _raw_metrics, _sha256, _write_csv)  # noqa: E402
from common.eval_dabe_rank_calibration import FastCODContext, _load_response, _ranking_metrics, _resize_current, _resize_native  # noqa: E402
from common.utils import load_config, torch_load, write_json  # noqa: E402

SCRIPT_PATH=Path(__file__).resolve(); VERSION="dabe_r1_design_v1"
METHODS=("R1-Cached","M0-R1-BW2-NodeEdge-Proto","M1-R1-BW1-NodeEdge-Proto","M2-R1-BW2-NoEdge-Proto","M3-R1-BW2-HRInterface-Proto","M4-R1-BW2-NodeEdge-SC","M5-R1-BW1-NodeEdge-SC","Current-DABE-v2")
FIELDS_MAP={METHODS[0]:"residual_pass1_37",METHODS[1]:"m0_bw2_node_proto_37",METHODS[2]:"m1_bw1_node_proto_37",METHODS[3]:"m2_bw2_noedge_proto_37",METHODS[4]:"m3_bw2_hrinterface_proto_37",METHODS[5]:"m4_bw2_node_sc_37",METHODS[6]:"m5_bw1_node_sc_37"}
DATASETS=("CHAMELEON","TE-CAMO","TE-COD10K","NC4K")
PAIRWISE=((METHODS[2],METHODS[1]),(METHODS[3],METHODS[1]),(METHODS[4],METHODS[1]),(METHODS[5],METHODS[1]),(METHODS[6],METHODS[1]),(METHODS[6],METHODS[2]),(METHODS[6],METHODS[5]),(METHODS[1],METHODS[7]))
HARD=("hard_S_m","hard_F_beta_w","hard_F_beta_mean","hard_E_mean","hard_MAE","hard_IoU","hard_Precision","hard_Recall","hard_Area","hard_Components")
OFFICIAL=("official_soft_S_m","official_soft_F_beta_w","official_soft_F_beta_mean","official_soft_F_beta_max","official_soft_E_mean","official_soft_E_max","official_soft_MAE")
RAW=("raw_MAE","raw_Brier","raw_SoftPrecision","raw_SoftRecall","raw_SoftIoU","raw_prob_mean","raw_prob_std")
RANK=("pixel_AP","best_IoU_256","best_IoU_threshold_256","ranking_F_beta_max","ranking_E_max")
PAIR_FIELDS=("delta_hard_S_m","delta_hard_F_beta_w","delta_hard_E_mean","delta_hard_MAE","delta_hard_IoU","delta_hard_Precision","delta_hard_Recall","delta_hard_Area","delta_official_soft_F_beta_w","delta_raw_MAE","delta_pixel_AP","delta_best_IoU_256")
GEN_FIELDS=("m0_cached_r1_max_abs","m0_current_function_max_abs","border_source_count_bw2","border_source_count_bw1","border_source_ratio_bw2","border_source_ratio_bw1",*[f"anchor_count_m{i}" for i in range(4)],*[f"anchor_ratio_m{i}" for i in range(4)],*[f"bc_mean_m{i}" for i in range(4)],*[f"bc_std_m{i}" for i in range(4)],*[f"m{i}_area_gt_05" for i in range(6)],*[f"raw_mean_m{i}" for i in range(6)],*[f"raw_std_m{i}" for i in range(6)],"support_norm_mean_m0","support_norm_std_m0","support_norm_mean_m1","support_norm_std_m1","weight_entropy_mean_m0","weight_entropy_mean_m1","color_dispersion_mean_m0","color_dispersion_mean_m1","proto_sc_spearman_bw2","proto_sc_spearman_bw1","node_hr_edge_pearson","node_hr_edge_spearman")
EDGE_FIELDS=("edge_cue_mean_bg_bg","edge_cue_mean_fg_fg","edge_cue_mean_fg_bg","edge_cue_median_bg_bg","edge_cue_median_fg_bg","graph_weight_mean_bg_bg","graph_weight_mean_fg_fg","graph_weight_mean_fg_bg","cross_edge_cue_margin","cross_edge_weight_margin","cross_edge_auc")
RES_FIELDS=("proto_sc_spearman","sc_minus_proto_raw_mean","proto_raw_fg_mean","proto_raw_bg_mean","proto_raw_margin","sc_raw_fg_mean","sc_raw_bg_mean","sc_raw_margin","delta_raw_margin","proto_semantic_fg_mean","proto_semantic_bg_mean","proto_semantic_margin","sc_semantic_fg_mean","sc_semantic_bg_mean","sc_semantic_margin","proto_color_fg_mean","proto_color_bg_mean","proto_color_margin","sc_color_fg_mean","sc_color_bg_mean","sc_color_margin","support_norm_fg_mean","support_norm_bg_mean","support_norm_margin","weight_entropy_fg_mean","weight_entropy_bg_mean","weight_entropy_margin","color_dispersion_fg_mean","color_dispersion_bg_mean","color_dispersion_margin")
BASE={"R1-Cached":{"hard_S_m":.7321127747158344,"hard_F_beta_w":.6242804301164221,"hard_E_mean":.8279479346485146,"hard_MAE":.08046898620831737,"hard_Precision":.7069664740758146,"hard_Recall":.7124157409732799,"hard_Area":.1260482386630983},"M0-R1-BW2-NodeEdge-Proto":{"hard_S_m":.7321127747158344,"hard_F_beta_w":.6242804301164221,"hard_E_mean":.8279479346485146,"hard_MAE":.08046898620831737,"hard_Precision":.7069664740758146,"hard_Recall":.7124157409732799,"hard_Area":.1260482386630983},"Current-DABE-v2":{"hard_S_m":.7031564688307831,"hard_F_beta_w":.5722451313959636,"hard_E_mean":.7743040501812801,"hard_MAE":.0939140360866542}}

def _mean(mask,value):
    v=value.reshape(-1)[mask.reshape(-1)]; return float(v.mean()) if v.numel() else float("nan")
def _spearman(a,b):
    x=average_percentile_rank(a).reshape(-1).double(); y=average_percentile_rank(b).reshape(-1).double(); x-=x.mean(); y-=y.mean(); d=float(torch.linalg.norm(x)*torch.linalg.norm(y)); return float(torch.dot(x,y)/d) if d else float("nan")
def _auc(pos,neg):
    if not pos.numel() or not neg.numel(): return float("nan")
    values=torch.cat((pos,neg)); ranks=average_percentile_rank(values)*(values.numel()-1)+1
    n1,n0=pos.numel(),neg.numel(); return float((ranks[:n1].sum()-n1*(n1+1)/2)/(n1*n0))
def _touch(gt37):
    g=gt37.squeeze().bool(); top,bottom,left,right=bool(g[0].any()),bool(g[-1].any()),bool(g[:,0].any()),bool(g[:,-1].any()); t1=top or bottom or left or right
    second=bool(g[1].any() or g[-2].any() or g[:,1].any() or g[:,-2].any())
    subset="Touch-1" if t1 else ("Touch-2-only" if second else "Non-touch")
    return {"touch_subset":subset,"touch_side_count":sum((top,bottom,left,right)),"touch_top":top,"touch_bottom":bottom,"touch_left":left,"touch_right":right,"touch_corner":bool(g[0,0] or g[0,-1] or g[-1,0] or g[-1,-1])}
def _edge_audit(payload,gt):
    idx=payload["neigh_idx_37"].long(); valid=payload["neigh_valid_37"].bool(); src=torch.arange(1369)[:,None].expand_as(idx); use=valid & (src<idx)
    a=gt.reshape(-1)[src[use]]; b=gt.reshape(-1)[idx[use]]; bg=~a&~b; fg=a&b; cross=a^b; rows=[]
    for mode,cf,wf in (("node_sobel","edge_cue_node","graph_weight_node"),("hr_interface","edge_cue_hr","graph_weight_hr")):
        cue=payload[cf][use].float(); weight=payload[wf][use].float()
        def mean(m,v): return float(v[m].mean()) if m.any() else float("nan")
        def median(m,v): return float(v[m].median()) if m.any() else float("nan")
        cb,cx=mean(bg,cue),mean(cross,cue); wb,wx=mean(bg,weight),mean(cross,weight)
        rows.append({"edge_mode":mode,"edge_cue_mean_bg_bg":cb,"edge_cue_mean_fg_fg":mean(fg,cue),"edge_cue_mean_fg_bg":cx,"edge_cue_median_bg_bg":median(bg,cue),"edge_cue_median_fg_bg":median(cross,cue),"graph_weight_mean_bg_bg":wb,"graph_weight_mean_fg_fg":mean(fg,weight),"graph_weight_mean_fg_bg":wx,"cross_edge_cue_margin":cx-cb,"cross_edge_weight_margin":wb-wx,"cross_edge_auc":_auc(cue[cross],cue[bg])})
    return rows
def _residual_audit(p,gt):
    rows=[]
    for pair,pi,si in (("M4-vs-M0",0,4),("M5-vs-M1",1,5)):
        proto=p[f"raw_m{pi}_37"].float(); sc=p[f"raw_m{si}_37"].float(); fg=gt.bool(); bg=~fg
        ps=p[f"semantic_m{pi}_37"].float(); ss=p[f"semantic_m{si}_37"].float(); pc=p[f"color_m{pi}_37"].float(); sc_color=p[f"color_m{si}_37"].float(); support=p[f"support_norm_m{pi}_37"].float(); entropy=p[f"weight_entropy_m{pi}_37"].float(); dispersion=p[f"color_dispersion_m{pi}_37"].float()
        row={"residual_pair":pair,"proto_sc_spearman":_spearman(proto,sc),"sc_minus_proto_raw_mean":float((sc-proto).mean())}
        for prefix,value in (("proto_raw",proto),("sc_raw",sc),("proto_semantic",ps),("sc_semantic",ss),("proto_color",pc),("sc_color",sc_color),("support_norm",support),("weight_entropy",entropy),("color_dispersion",dispersion)):
            f,b=_mean(fg,value),_mean(bg,value); row.update({f"{prefix}_fg_mean":f,f"{prefix}_bg_mean":b,f"{prefix}_margin":f-b})
        row["delta_raw_margin"]=row["sc_raw_margin"]-row["proto_raw_margin"]; rows.append(row)
    return rows

def _init_worker(n): torch.set_num_threads(n)
def _one(t):
    dataset,stem=t["dataset"],t["stem"]; sp,dp=Path(t["source"]),Path(t["design"]); source=torch_load(sp,map_location="cpu"); design=torch_load(dp,map_location="cpu")
    if any(p.get("dataset")!=dataset or p.get("stem")!=stem for p in (source,design)): raise RuntimeError(f"key mismatch {dataset}/{stem}")
    if design.get("design_version")!=VERSION: raise RuntimeError(f"version mismatch {dp}")
    gt=_load_gt(t["gt"]); shape=tuple(gt.shape[-2:]); native={METHODS[0]:_load_response(source,FIELDS_MAP[METHODS[0]],(37,37),sp)}
    for method in METHODS[1:7]: native[method]=_load_response(design,FIELDS_MAP[method],(37,37),dp)
    if float((native[METHODS[0]]-native[METHODS[1]]).abs().max())>1e-6: raise RuntimeError(f"M0/cached mismatch {dataset}/{stem}")
    probs={m:_resize_native(native[m],shape) for m in METHODS[:7]}; probs[METHODS[7]]=_resize_current(_load_response(source,"p_dabe_68",(68,68),sp),shape)
    context=FastCODContext(gt);cod={}
    # Weighted-F temporarily holds several full-resolution float arrays per
    # prediction.  Evaluate one method's hard/soft pair at a time so a single
    # unusually large GT cannot multiply memory by all eight methods.
    for m in METHODS:cod.update(context.evaluate_many([(m,"hard",probs[m]),(m,"soft",probs[m])],t["threshold"]))
    rows=[]
    for m in METHODS:
        h,s=cod[(m,"hard")],cod[(m,"soft")]; row={"dataset":dataset,"stem":stem,"method":m,"hard_S_m":h["S_m"],"hard_F_beta_w":h["F_beta_w"],"hard_F_beta_mean":h["F_beta_mean"],"hard_E_mean":h["E_mean"],"hard_MAE":h["MAE"],"hard_IoU":h["IoU"],"hard_Precision":h["Precision"],"hard_Recall":h["Recall"],"hard_Area":h["Area"],"hard_Components":h["Components"],"official_soft_S_m":s["S_m"],"official_soft_F_beta_w":s["F_beta_w"],"official_soft_F_beta_mean":s["F_beta_mean"],"official_soft_F_beta_max":s["F_beta_max"],"official_soft_E_mean":s["E_mean"],"official_soft_E_max":s["E_max"],"official_soft_MAE":s["MAE"],**_raw_metrics(probs[m],gt),**_ranking_metrics(probs[m],gt),"ranking_F_beta_max":s["F_beta_max"],"ranking_E_max":s["E_max"],"source_dabe_cache_path":str(sp),"design_cache_path":"" if m in (METHODS[0],METHODS[7]) else str(dp),"_official_f_curve":s["f_curve"],"_official_e_curve":s["e_curve"]}; rows.append(row)
    gt37=torch_f.interpolate(gt.unsqueeze(0),size=(37,37),mode="nearest").squeeze(0)>0.5; touch=_touch(gt37); contam=[]; anchors={METHODS[0]:0,METHODS[1]:0,METHODS[2]:1,METHODS[3]:2,METHODS[4]:3,METHODS[5]:0,METHODS[6]:1}; fg_count=int(gt37.sum())
    for method,i in anchors.items():
        a=design[f"anchor_m{i}_37"].bool(); overlap=int((a&gt37).sum()); contam.append({"audit_source":"anchor","audit_method":method,"anchor_leak":overlap/(int(a.sum())+1e-12),"fg_absorbed":overlap/(fg_count+1e-12),"border_leak":float("nan"),"border_fg_absorbed":float("nan")})
    for width in (1,2):
        b=torch.zeros((37,37),dtype=torch.bool); b[:width]=True;b[-width:]=True;b[:,:width]=True;b[:,-width:]=True; overlap=int((b&gt37.squeeze()).sum()); contam.append({"audit_source":"border","audit_method":f"BW{width}-BorderSource","anchor_leak":float("nan"),"fg_absorbed":float("nan"),"border_leak":overlap/(int(b.sum())+1e-12),"border_fg_absorbed":overlap/(fg_count+1e-12)})
    return {"dataset":dataset,"stem":stem,"rows":rows,"touch":touch,"contam":contam,"edge":_edge_audit(design,gt37),"residual":_residual_audit(design,gt37),"diagnostics":design["diagnostics"],"worker_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}

def _sections(aggs,datasets,section):
    fields={"hard":HARD,"official_soft":OFFICIAL,"raw_continuous":RAW,"ranking":RANK}[section]; by=[]
    for d in datasets:
        for m in METHODS:
            v=aggs[(d,m)].result(); by.append({"scope":"by_dataset","dataset":d,"method":m,**{f:v[f] for f in fields},"num_samples":v["num_samples"],"ap_valid_count":v["ap_valid_count"]})
    overall=[]
    for m in METHODS:
        v=aggs[("ALL",m)].result(); overall.append({"scope":"sample_overall","dataset":"ALL","method":m,**{f:v[f] for f in fields},"num_samples":v["num_samples"],"ap_valid_count":v["ap_valid_count"]})
    macro=[]
    for m in METHODS:
        rs=[r for r in by if r["method"]==m]; macro.append({"scope":"dataset_macro","dataset":"ALL","method":m,**{f:float(np.mean([r[f] for r in rs])) for f in fields},"num_samples":sum(r["num_samples"] for r in rs),"ap_valid_count":sum(r["ap_valid_count"] for r in rs)})
    return by,overall,macro
def _aggregate(records,groups,fields):
    out=[]
    for keys,rs in groups(records).items():
        row=dict(keys); row["num_samples"]=len(rs)
        for f in fields:
            vals=[float(r[f]) for r in rs if math.isfinite(float(r[f]))]; row[f]=float(np.mean(vals)) if vals else float("nan"); row[f+"_valid_count"]=len(vals)
        out.append(row)
    return out
def _group(keys):
    return lambda records: _group_impl(records,keys)
def _group_impl(records,keys):
    g=defaultdict(list)
    for r in records: g[tuple((k,r[k]) for k in keys)].append(r)
    return g

def _assess_success_and_directions(sections, contamination_subset, edge_overall):
    """Apply only the predeclared task-book criteria; never route samples."""
    hard_macro={r["method"]:r for r in sections["hard"]["dataset_macro"]}
    rank_macro={r["method"]:r for r in sections["ranking"]["dataset_macro"]}
    hard_by={(r["dataset"],r["method"]):r for r in sections["hard"]["by_dataset"]}
    touch={(r["dataset"],r["touch_subset"],r["method"]):r for r in sections["touch_subset_hard"]}
    base=METHODS[1]; base_h=hard_macro[base]; base_r=rank_macro[base]
    success={"criteria":{"strong_no_obvious_degradation_tolerance":0.001,"strong_synchronous_improvement_minimum":3},"candidates":{}}
    for method in METHODS[2:7]:
        ds={d:hard_by[(d,method)]["hard_F_beta_w"]-hard_by[(d,base)]["hard_F_beta_w"] for d in DATASETS}
        synchronous={
            "pixel_AP":rank_macro[method]["pixel_AP"]>=base_r["pixel_AP"],
            "best_IoU_256":rank_macro[method]["best_IoU_256"]>=base_r["best_IoU_256"],
            "hard_S_m":hard_macro[method]["hard_S_m"]>=base_h["hard_S_m"],
            "hard_E_mean":hard_macro[method]["hard_E_mean"]>=base_h["hard_E_mean"],
            "hard_MAE":hard_macro[method]["hard_MAE"]<=base_h["hard_MAE"],
        }
        success["candidates"][method]={
            "dataset_F_beta_w_deltas":ds,
            "engineering_basic_success":hard_macro[method]["hard_F_beta_w"]>.624280 and hard_macro[method]["hard_MAE"]<=.080969 and min(ds.values())>=-.005,
            "paper_retention_success":hard_macro[method]["hard_F_beta_w"]>=.625780 and sum(x>=0 for x in ds.values())>=3 and min(ds.values())>=-.003 and hard_macro[method]["hard_MAE"]<=.080469 and rank_macro[method]["pixel_AP"]>=base_r["pixel_AP"],
            "clear_success":hard_macro[method]["hard_F_beta_w"]>=.6270 and sum(x>0 for x in ds.values())>=3 and (hard_macro[method]["hard_S_m"]>=base_h["hard_S_m"] or hard_macro[method]["hard_E_mean"]>=base_h["hard_E_mean"]) and hard_macro[method]["hard_MAE"]<=base_h["hard_MAE"],
            "strong_success":hard_macro[method]["hard_F_beta_w"]>=.6290 and min(ds.values())>=-.001 and sum(synchronous.values())>=3,
            "synchronous_improvements":synchronous,
        }
    paper=[m for m,v in success["candidates"].items() if v["paper_retention_success"]]
    success["frozen_method"]=max(paper,key=lambda m:hard_macro[m]["hard_F_beta_w"]) if paper else base

    def contamination(method):
        return next(r for r in contamination_subset if r["dataset"]=="ALL" and r["touch_subset"]=="Touch-1" and r["audit_source"]=="anchor" and r["audit_method"]==method)
    c0,c1=contamination(base),contamination(METHODS[2]); t0=touch[("ALL","Touch-1",base)];t1=touch[("ALL","Touch-1",METHODS[2])]
    fg_reduction=(c0["fg_absorbed"]-c1["fg_absorbed"])/(c0["fg_absorbed"]+1e-12)
    touch_recall_delta=t1["hard_Recall"]-t0["hard_Recall"];touch_fw_delta=t1["hard_F_beta_w"]-t0["hard_F_beta_w"];m1_macro_delta=hard_macro[METHODS[2]]["hard_F_beta_w"]-base_h["hard_F_beta_w"]
    boundary_pass=fg_reduction>=.20 and touch_recall_delta>=.01 and touch_fw_delta>=.005 and m1_macro_delta>=-.001
    cv_supported=fg_reduction>=.20 and touch_recall_delta>0 and touch_fw_delta>0 and not success["candidates"][METHODS[2]]["paper_retention_success"]
    boundary={"touch1_fg_absorbed_relative_reduction":fg_reduction,"touch1_recall_delta":touch_recall_delta,"touch1_F_beta_w_delta":touch_fw_delta,"dataset_macro_F_beta_w_delta":m1_macro_delta,"boundary_width_hypothesis_passed":boundary_pass,"continue_cross_validated_boundary_connectivity":cv_supported}

    edge={r["edge_mode"]:r for r in edge_overall};m2=hard_macro[METHODS[3]]["hard_F_beta_w"]-base_h["hard_F_beta_w"];m3=hard_macro[METHODS[4]]["hard_F_beta_w"]-base_h["hard_F_beta_w"]
    hr_auc_better=edge["hr_interface"]["cross_edge_auc"]>edge["node_sobel"]["cross_edge_auc"]
    if hr_auc_better and m3<=0: edge_case="D: HR-interface edge AUC improves but segmentation does not"
    elif m2>=0: edge_case="A: removing Node-Sobel is non-inferior"
    elif m3>0 and m2<0: edge_case="B: edge is useful but HR-interface is preferable"
    else: edge_case="C: both NoEdge and HR-interface underperform Node-Sobel"
    sobel={"M2_macro_F_beta_w_delta":m2,"M3_macro_F_beta_w_delta":m3,"node_sobel_cross_edge_auc":edge["node_sobel"]["cross_edge_auc"],"hr_interface_cross_edge_auc":edge["hr_interface"]["cross_edge_auc"],"case":edge_case,"delete_sobel":success["candidates"][METHODS[3]]["paper_retention_success"]}

    m4_h=hard_macro[METHODS[5]]["hard_F_beta_w"]-base_h["hard_F_beta_w"];m5_m4=hard_macro[METHODS[6]]["hard_F_beta_w"]-hard_macro[METHODS[5]]["hard_F_beta_w"];m4_ap=rank_macro[METHODS[5]]["pixel_AP"]-base_r["pixel_AP"]
    if m4_h>0: sc_case="A: Support-Consistent Residual improves M0"
    elif m4_ap>0: sc_case="C: SC improves Pixel AP but not Hard F_beta_w"
    elif m4_h<0 and hard_macro[METHODS[6]]["hard_F_beta_w"]<base_h["hard_F_beta_w"]: sc_case="D: both M4 and M5 underperform M0"
    else: sc_case="No predeclared SC case is conclusively met"
    if m5_m4>0 and boundary_pass: sc_case += "; B: BW1 and SC are complementary"
    sc={"M4_macro_F_beta_w_delta":m4_h,"M4_pixel_AP_delta":m4_ap,"M5_minus_M4_macro_F_beta_w":m5_m4,"case":sc_case,"retain_support_consistent_residual":success["candidates"][METHODS[5]]["paper_retention_success"] or success["candidates"][METHODS[6]]["paper_retention_success"]}
    return success,{"boundary":boundary,"sobel":sobel,"support_consistent_residual":sc}

def _save_vis(t,path):
    s=torch_load(t["source"],map_location="cpu"); d=torch_load(t["design"],map_location="cpu"); gt=_load_gt(t["gt"]); size=160
    with Image.open(t["image"]) as im: rgb=im.convert("RGB").resize((size,size),Image.Resampling.BICUBIC)
    vals=[d[FIELDS_MAP[m]] for m in METHODS[1:7]]; panels=[("RGB",rgb),("GT",_gray_panel(gt,size,True)),("M0 R1",_gray_panel(vals[0],size)),("M1 BW1",_gray_panel(vals[1],size)),("M2 NoEdge",_gray_panel(vals[2],size)),("M3 HRInterface",_gray_panel(vals[3],size)),("M4 SC-W2",_gray_panel(vals[4],size)),("M5 SC-W1",_gray_panel(vals[5],size)),("M0 anchor",_gray_panel(d["anchor_m0_37"],size,True)),("M1 anchor",_gray_panel(d["anchor_m1_37"],size,True)),("M0-M1",_difference_panel(vals[0]-vals[1],size)),("M0-M4",_difference_panel(vals[0]-vals[4],size))]
    lh=22; canvas=Image.new("RGB",(6*size,2*(size+lh)),"white"); draw=ImageDraw.Draw(canvas)
    for i,(title,p) in enumerate(panels): x=(i%6)*size;y=(i//6)*(size+lh);canvas.paste(p,(x,y+lh));draw.text((x+3,y+4),title,fill="black")
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def _visuals(tasks,touches,deltas,out,datasets):
    by=defaultdict(list)
    for t in tasks: by[t["dataset"]].append(t)
    for d in datasets:
        for t in by[d][:8]: _save_vis(t,out/"vis"/"fixed_first8"/d/f"{t['stem']}.png")
        touch=[t for t in by[d] if touches[(d,t["stem"])]=="Touch-1"]
        for t in touch[:16]: _save_vis(t,out/"vis"/"touch1_first16"/d/f"{t['stem']}.png")
        for m in (METHODS[2],METHODS[4],METHODS[5],METHODS[6]):
            ranked=sorted(by[d],key=lambda t:deltas[(d,t["stem"],m)],reverse=True); tag=m.split("-")[0]
            for folder,sel in ((f"{tag}_vs_M0_top8",ranked[:8]),(f"{tag}_vs_M0_bottom8",ranked[-8:])):
                for t in sel:_save_vis(t,out/"vis"/folder/d/f"{t['stem']}.png")

def evaluate_r1_design(config_path,dabe_root,design_root,out_dir,split="test",max_samples=-1,threshold=.5,save_vis=False,workers=None,torch_threads=1,overwrite=False,overwrite_reason=""):
    started=time.time(); config_path=Path(config_path).resolve();dabe_root=Path(dabe_root).resolve();design_root=Path(design_root).resolve();out_dir=Path(out_dir).resolve()
    if split!="test" or threshold!=.5 or max_samples==0 or max_samples < -1: raise ValueError("frozen protocol violation")
    cfg=load_config(config_path)
    if getattr(cfg,"BACKBONE_KEY",None)!="dinov1-s8": raise ValueError("wrong backbone")
    full=max_samples==-1; sr,sm=_manifest_map(dabe_root/f"manifest_{split}.jsonl",6473 if full else None); dr,dm=_manifest_map(design_root/f"manifest_{split}.jsonl",6473 if full else max_samples); selected=sr if full else sr[:max_samples]; keys=[(str(r["dataset"]),str(r["stem"])) for r in selected]
    if set(keys)!=set(dm): raise RuntimeError("design keys differ from selected source")
    proto=json.loads((design_root/"protocol.json").read_text());
    if proto.get("design_version")!=VERSION or proto.get("source_dabe_manifest_sha256")!=_sha256(dabe_root/f"manifest_{split}.jsonl"): raise RuntimeError("design protocol mismatch")
    datasets=[d for d in DATASETS if any(k[0]==d for k in keys)]
    if out_dir.exists():
        if not overwrite: raise FileExistsError(out_dir)
        if not overwrite_reason.strip(): raise ValueError("overwrite reason required")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True); logh=(out_dir/"eval.log").open("w",encoding="utf-8",buffering=1)
    def log(x): print(x,flush=True);logh.write(str(x)+"\n")
    workers=workers or min(12,os.cpu_count() or 1); tasks=[]
    for r in selected:
        k=(str(r["dataset"]),str(r["stem"]));
        if not Path(r["gt_path"]).is_file() or not Path(r["image_path"]).is_file(): raise FileNotFoundError(k)
        tasks.append({"dataset":k[0],"stem":k[1],"source":sm[k]["cache_path"],"design":dm[k]["cache_path"],"gt":r["gt_path"],"image":r["image_path"],"threshold":threshold})
    aggs={(d,m):MetricAggregate() for d in [*datasets,"ALL"] for m in METHODS}; pairvals=defaultdict(list);touchrows=[];touchmeta=[];contam=[];edges=[];residual=[];gens=[];deltas={};touchmap={};peak=0
    perfields=("dataset","stem","method",*HARD,*OFFICIAL,*RAW,*RANK,"source_dabe_cache_path","design_cache_path"); pairhead=("dataset","stem","method_a","method_b",*PAIR_FIELDS)
    try:
        log(f"num_samples = {len(tasks)}");log(f"workers = {workers}");log("hard_threshold = strict > 0.5");log("candidate_generation_in_eval = false")
        with (out_dir/"per_sample.csv").open("w",newline="",encoding="utf-8") as ph,(out_dir/"pairwise_per_sample.csv").open("w",newline="",encoding="utf-8") as qh:
            pw=csv.DictWriter(ph,fieldnames=perfields,extrasaction="ignore");qw=csv.DictWriter(qh,fieldnames=pairhead,extrasaction="ignore");pw.writeheader();qw.writeheader()
            with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(torch_threads,)) as ex:
                for i,res in enumerate(ex.map(_one,tasks,chunksize=1),1):
                    d,stem=res["dataset"],res["stem"];peak=max(peak,res["worker_rss"]); rm={r["method"]:r for r in res["rows"]}
                    for r in res["rows"]:aggs[(d,r["method"])].add(r);aggs[("ALL",r["method"])].add(r);pw.writerow(r);touchrows.append({k:r[k] for k in ("dataset","stem","method",*HARD)}|res["touch"])
                    touchmap[(d,stem)]=res["touch"]["touch_subset"];touchmeta.append({"dataset":d,"stem":stem,**res["touch"]})
                    for r in res["contam"]:contam.append({"dataset":d,"stem":stem,"touch_subset":res["touch"]["touch_subset"],**r})
                    for r in res["edge"]:edges.append({"dataset":d,"stem":stem,**r})
                    for r in res["residual"]:residual.append({"dataset":d,"stem":stem,**r})
                    gens.append({"dataset":d,"stem":stem,**res["diagnostics"]})
                    mapping={"delta_hard_S_m":"hard_S_m","delta_hard_F_beta_w":"hard_F_beta_w","delta_hard_E_mean":"hard_E_mean","delta_hard_MAE":"hard_MAE","delta_hard_IoU":"hard_IoU","delta_hard_Precision":"hard_Precision","delta_hard_Recall":"hard_Recall","delta_hard_Area":"hard_Area","delta_official_soft_F_beta_w":"official_soft_F_beta_w","delta_raw_MAE":"raw_MAE","delta_pixel_AP":"pixel_AP","delta_best_IoU_256":"best_IoU_256"}
                    for a,b in PAIRWISE:
                        pr={"dataset":d,"stem":stem,"method_a":a,"method_b":b}
                        for df,sf in mapping.items(): pr[df]=float(rm[a][sf])-float(rm[b][sf]);pairvals[(d,a,b,df)].append(pr[df])
                        qw.writerow(pr)
                    for m in (METHODS[2],METHODS[4],METHODS[5],METHODS[6]):deltas[(d,stem,m)]=rm[m]["hard_IoU"]-rm[METHODS[1]]["hard_IoU"]
                    if i==len(tasks) or i%100==0:log(f"processed = {i}/{len(tasks)}")
        sections={}; fmap={"hard":HARD,"official_soft":OFFICIAL,"raw_continuous":RAW,"ranking":RANK}
        for sec,fs in fmap.items():
            by,overall,macro=_sections(aggs,datasets,sec);sections[sec]={"by_dataset":by,"sample_overall":overall,"dataset_macro":macro};head=("scope","dataset","method",*fs,"num_samples","ap_valid_count")
            for suffix,rows in (("by_dataset",by),("sample_overall",overall),("dataset_macro",macro)):_write_csv(out_dir/f"{sec}_{suffix}.csv",rows,head)
        counts=[]
        for d in [*datasets,"ALL"]:
            source=[r for r in touchrows if d=="ALL" or r["dataset"]==d]
            for subset in ("Touch-1","Touch-2-only","Non-touch"): counts.append({"scope":"sample_overall" if d=="ALL" else "by_dataset","dataset":d,"touch_subset":subset,"num_samples":len({(r['dataset'],r['stem']) for r in source if r['touch_subset']==subset})})
        _write_csv(out_dir/"touch_subset_counts.csv",counts,("scope","dataset","touch_subset","num_samples"))
        _write_csv(out_dir/"touch_metadata_per_sample.csv",touchmeta,("dataset","stem","touch_subset","touch_side_count","touch_top","touch_bottom","touch_left","touch_right","touch_corner"))
        tsh=[]
        for d in [*datasets,"ALL"]:
            for subset in ("Touch-1","Touch-2-only","Non-touch"):
                for m in METHODS:
                    rs=[r for r in touchrows if (d=="ALL" or r["dataset"]==d) and r["touch_subset"]==subset and r["method"]==m]; tsh.append({"scope":"sample_overall" if d=="ALL" else "by_dataset","dataset":d,"touch_subset":subset,"method":m,**{f:float(np.mean([r[f] for r in rs])) if rs else float('nan') for f in HARD},"num_samples":len(rs)})
        _write_csv(out_dir/"touch_subset_hard.csv",tsh,("scope","dataset","touch_subset","method",*HARD,"num_samples"))
        contam_scoped=[*contam,*({**r,"dataset":"ALL"} for r in contam)];con_by=_aggregate(contam_scoped,_group(("dataset","audit_source","audit_method")),("anchor_leak","fg_absorbed","border_leak","border_fg_absorbed")); con_sub=_aggregate(contam_scoped,_group(("dataset","touch_subset","audit_source","audit_method")),("anchor_leak","fg_absorbed","border_leak","border_fg_absorbed"))
        ch=("dataset","audit_source","audit_method","anchor_leak","fg_absorbed","border_leak","border_fg_absorbed","num_samples",*[f+"_valid_count" for f in ("anchor_leak","fg_absorbed","border_leak","border_fg_absorbed")]);_write_csv(out_dir/"anchor_contamination_by_dataset.csv",con_by,ch)
        csh=("dataset","touch_subset","audit_source","audit_method","anchor_leak","fg_absorbed","border_leak","border_fg_absorbed","num_samples",*[f+"_valid_count" for f in ("anchor_leak","fg_absorbed","border_leak","border_fg_absorbed")]);_write_csv(out_dir/"anchor_contamination_by_subset.csv",con_sub,csh)
        edge_by=_aggregate(edges,_group(("dataset","edge_mode")),EDGE_FIELDS);edge_all=_aggregate(edges,_group(("edge_mode",)),EDGE_FIELDS);res_by=_aggregate(residual,_group(("dataset","residual_pair")),RES_FIELDS);res_all=_aggregate(residual,_group(("residual_pair",)),RES_FIELDS);gen_by=_aggregate(gens,_group(("dataset",)),GEN_FIELDS);gen_all=_aggregate(gens,_group(tuple()),GEN_FIELDS)
        for fn,rows,base in (("edge_audit_by_dataset.csv",edge_by,("dataset","edge_mode")),("edge_audit_overall.csv",edge_all,("edge_mode",)),("residual_mechanism_by_dataset.csv",res_by,("dataset","residual_pair")),("residual_mechanism_overall.csv",res_all,("residual_pair",)),("generation_diagnostics_by_dataset.csv",gen_by,("dataset",)),("generation_diagnostics_overall.csv",gen_all,tuple())):
            fs=EDGE_FIELDS if fn.startswith('edge') else (RES_FIELDS if fn.startswith('residual') else GEN_FIELDS);_write_csv(out_dir/fn,rows,(*base,*fs,"num_samples",*[f+"_valid_count" for f in fs]))
        ps=[]
        for d in datasets:
            for a,b in PAIRWISE:
                for f in PAIR_FIELDS:
                    v=np.asarray(pairvals[(d,a,b,f)]);ps.append({"dataset":d,"method_a":a,"method_b":b,"metric":f,"mean_delta":v.mean(),"median_delta":np.median(v),"p25_delta":np.percentile(v,25),"p75_delta":np.percentile(v,75),"win_ratio":np.mean(v>1e-6),"tie_ratio":np.mean(abs(v)<=1e-6),"loss_ratio":np.mean(v< -1e-6),"valid_count":len(v)})
        _write_csv(out_dir/"pairwise_by_dataset.csv",ps,("dataset","method_a","method_b","metric","mean_delta","median_delta","p25_delta","p75_delta","win_ratio","tie_ratio","loss_ratio","valid_count"))
        if save_vis:log("saving_visualizations = true");_visuals(tasks,touchmap,deltas,out_dir,datasets);log("visualizations_complete = true")
        macro={r["method"]:r for r in sections["hard"]["dataset_macro"]};rankm={r["method"]:r for r in sections["ranking"]["dataset_macro"]};baseline={"applicable":full,"tolerance":1e-5,"checks":{},"passed":None}
        if full:
            ok=True
            for m,efs in BASE.items():baseline["checks"][m]={};
            for m,efs in BASE.items():
                for f,e in efs.items():a=float(macro[m][f]);err=abs(a-e);passed=err<=1e-5;ok &= passed;baseline["checks"][m][f]={"actual":a,"expected":e,"absolute_error":err,"passed":passed}
            baseline["passed"]=bool(ok)
        success={"not_evaluated":True};directions={"not_evaluated":True}
        if baseline.get("passed"):
            sections["touch_subset_hard"]=tsh;success,directions=_assess_success_and_directions(sections,con_sub,edge_all)
        elapsed=time.time()-started;commit,status=_git_metadata();protocol={"design_version":VERSION,"split":split,"num_samples":len(tasks),"full_run":full,"methods":list(METHODS),"hard_threshold":.5,"hard_operator":">","gt_used_for_generation":False,"candidate_generation_in_eval":False,"official_soft_per_image_minmax":True,"best_iou_diagnostic_only":True,"config_path":str(config_path),"config_sha256":_sha256(config_path),"source_manifest":str(dabe_root/f'manifest_{split}.jsonl'),"source_manifest_sha256":_sha256(dabe_root/f'manifest_{split}.jsonl'),"design_manifest":str(design_root/f'manifest_{split}.jsonl'),"design_manifest_sha256":_sha256(design_root/f'manifest_{split}.jsonl'),"evaluator_sha256":_sha256(SCRIPT_PATH),"git_commit":commit,"git_status_short":status,"workers":workers,"elapsed_seconds":elapsed,"average_seconds_per_image":elapsed/len(tasks),"main_peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"max_worker_peak_rss_mb":peak,"save_vis":save_vis}
        summary={"protocol":protocol,"baseline_reproduction":baseline,"success_assessment":success,"direction_assessment":directions,**sections,"touch_subset_counts":counts,"touch_subset_hard":tsh,"anchor_contamination_by_dataset":con_by,"anchor_contamination_by_subset":con_sub,"edge_audit_by_dataset":edge_by,"edge_audit_overall":edge_all,"residual_mechanism_by_dataset":res_by,"residual_mechanism_overall":res_all,"generation_diagnostics_by_dataset":gen_by,"generation_diagnostics_overall":gen_all,"pairwise_by_dataset":ps};write_json(out_dir/"protocol.json",protocol);write_json(out_dir/"summary.json",summary)
        _write_docs(out_dir,summary,full);log(f"baseline_reproduction = {baseline.get('passed')}");log(f"elapsed_seconds = {elapsed:.3f}");log(f"average_seconds_per_image = {elapsed/len(tasks):.6f}");log(f"main_peak_rss_mb = {protocol['main_peak_rss_mb']:.3f}");log(f"max_worker_peak_rss_mb = {peak:.3f}");return summary
    finally:logh.close()

def _write_docs(out,s,full):
    (out/"README.md").write_text(
        "# R1-Design-v1 evaluation\n\n"
        f"- Scope: {'full 6473-sample four-dataset run' if full else '5-sample sanity only'}.\n"
        "- Candidate generation is GT-free and evaluation never regenerates candidates.\n"
        "- Native maps: 37→68→original GT bilinear, align_corners=False.\n"
        "- Hard: strict >0.5. Official Soft performs per-image Min-Max.\n"
        "- Touch subsets use nearest GT→37 and are diagnostic only.\n"
        "- `summary.json` is the machine-readable source of every aggregate and decision.\n",
        encoding="utf-8",
    )
    def table(rows,fields):
        def fmt(value):
            if isinstance(value,bool): return str(value)
            if isinstance(value,(int,np.integer)): return str(value)
            try:
                number=float(value);return f"{number:.6f}" if math.isfinite(number) else "NaN"
            except (TypeError,ValueError): return str(value)
        lines=["| "+" | ".join(fields)+" |","|"+"|".join(["---"]*len(fields))+"|"]
        lines.extend("| "+" | ".join(fmt(row.get(field,"")) for field in fields)+" |" for row in rows)
        return "\n".join(lines)
    hard=s["hard"]["by_dataset"]+[{**r,"dataset":"Dataset-Macro"} for r in s["hard"]["dataset_macro"]]
    lines=["# DABE-TF R1-Design-v1 结果","",f"- {'完整四数据集 6473 样本。' if full else '仅 5 样本 sanity，不给出正式结论。'}",f"- 基线复现：{s['baseline_reproduction'].get('passed')}。","","## Hard 主表","",table(hard,("dataset","method","hard_S_m","hard_F_beta_w","hard_E_mean","hard_MAE","hard_Precision","hard_Recall")),""]
    if full and not s['baseline_reproduction'].get('passed'):
        lines += ["## 协议失败","","基线未在 1e-5 容差内复现，因此禁止对 M1–M5 给出性能结论。","" ]
    if full and s['baseline_reproduction'].get('passed'):
        official=[{**r,"dataset":"Dataset-Macro"} for r in s["official_soft"]["dataset_macro"]]
        ranking=[{**r,"dataset":"Dataset-Macro"} for r in s["ranking"]["dataset_macro"]]
        touch=[r for r in s["touch_subset_hard"] if r["dataset"]=="ALL"]
        contamination=[r for r in s["anchor_contamination_by_subset"] if r["dataset"]=="ALL" and r["audit_method"] in (METHODS[1],METHODS[2],"BW1-BorderSource","BW2-BorderSource")]
        lines += [
            "## Official Soft 与 Ranking Dataset-Macro","",table(official,("dataset","method","official_soft_S_m","official_soft_F_beta_w","official_soft_E_mean","official_soft_MAE")),"",table(ranking,("dataset","method","pixel_AP","best_IoU_256","ranking_F_beta_max","ranking_E_max")),"",
            "## 触边子集","",table([r for r in s["touch_subset_counts"] if r["dataset"]=="ALL"],("touch_subset","num_samples")),"",table(touch,("touch_subset","method","hard_F_beta_w","hard_Recall","hard_MAE")),"",
            "## 边界/Anchor 污染","",table(contamination,("touch_subset","audit_method","anchor_leak","fg_absorbed","border_leak","border_fg_absorbed")),"",
            "## 边缘审计","",table(s["edge_audit_overall"],("edge_mode","cross_edge_auc","cross_edge_cue_margin","cross_edge_weight_margin")),"",
            "## Prototype 与 Support-Consistent 残差","",table(s["residual_mechanism_overall"],("residual_pair","proto_sc_spearman","proto_raw_margin","sc_raw_margin","delta_raw_margin","support_norm_margin","color_dispersion_margin")),"",
            "## 成功判定","","```json",json.dumps(s["success_assessment"],ensure_ascii=False,indent=2),"```","",
            "## 方向判定","","```json",json.dumps(s["direction_assessment"],ensure_ascii=False,indent=2),"```","",
            "逐数据集 Pairwise、完整污染项、边缘统计、残差机制和生成诊断见同目录专项 CSV；逐样本值见 `per_sample.csv` 与 `pairwise_per_sample.csv`。",
        ]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ("config","dabe_root","design_root","out_dir"):p.add_argument("--"+x,required=True)
    p.add_argument("--split",default="test");p.add_argument("--max_samples",type=int,default=-1);p.add_argument("--threshold",type=float,default=.5);p.add_argument("--save_vis",action="store_true");p.add_argument("--workers",type=int,default=min(12,os.cpu_count() or 1));p.add_argument("--torch_threads",type=int,default=1);p.add_argument("--overwrite",action="store_true");p.add_argument("--overwrite_reason",default="");return p.parse_args()
if __name__=="__main__":
    a=parse_args();evaluate_r1_design(a.config,a.dabe_root,a.design_root,a.out_dir,a.split,a.max_samples,a.threshold,a.save_vis,a.workers,a.torch_threads,a.overwrite,a.overwrite_reason)
