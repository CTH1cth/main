#!/usr/bin/env python3
"""Original-size COD evaluation, paired bootstrap and Pareto gating for GBSP V5."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_multiotsu

if __package__ in {None,""}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_threshold_v4 import HierarchicalUpperTailOtsu  # noqa: E402
from models.gbsp_threshold_v5 import SeededMultiOtsuHysteresis  # noqa: E402
from models.mbsp_reconstruction import MultiBackgroundSubspaceProjector  # noqa: E402
from tools.eval_gbsp_threshold import (  # noqa: E402
    DATASETS, EXPECTED, _aggregate, _binary_metrics, _load_gt, _resize,
    _sample_keys, _table, _write_csv, _write_json,
)
from tools.eval_gbsp_threshold_v3 import _rank_metrics  # noqa: E402


MAIN_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=MAIN_ROOT/"configs/dinov1_s8_gbsp_threshold_v5.py"
PRIMARY=("S_m","F_beta_w","E_mean","MAE")
SECONDARY=("F_beta_mean","Precision","Recall","Area","IoU","Dice")
BASELINE_NAMES={"fixed_058","multi_otsu_3","huto_core","huto_mid_h","gbsp_fixed_050","r1_fixed_050"}


def _resolve(path):
    path=Path(path); return path.resolve() if path.is_absolute() else (Path.cwd()/path).resolve()


def _manifest_path(root):
    for path in (root/"manifest_test.jsonl",root/"ablations/M1_pilot200/manifest_test.jsonl"):
        if path.is_file(): return path
    raise FileNotFoundError(f"manifest_test.jsonl missing below {root}")


def _manifest(path):
    rows=read_jsonl(path); seen=set()
    for line,row in enumerate(rows,1):
        key=(str(row.get("dataset","")),str(row.get("stem","")))
        if not all(key) or key in seen or not Path(str(row.get("cache_path",""))).is_file():
            raise RuntimeError(f"invalid manifest {path}:{line}: {key}")
        seen.add(key)
    return rows


def _select(rows,sample_list,limit):
    if sample_list:
        keys=_sample_keys(_resolve(sample_list)); mapping={(str(r["dataset"]),str(r["stem"])):r for r in rows}
        missing=[key for key in keys if key not in mapping]
        if missing: raise KeyError(missing[:5])
        rows=[mapping[key] for key in keys]
    if limit>=0: rows=rows[:limit]
    if not rows: raise RuntimeError("no samples")
    return rows


def _map(payload,fields,path):
    for field in fields:
        value=payload.get(field)
        if torch.is_tensor(value) and tuple(value.shape)==(1,37,37):
            value=value.detach().cpu().float().contiguous()
            if torch.isfinite(value).all(): return value
    raise ValueError(f"missing {fields}: {path}")


def _scalar(value):
    types=(str,bool,int,float,np.integer,np.floating)
    return {key:(item.item() if isinstance(item,(np.integer,np.floating)) else item)
            for key,item in value.items() if item is None or isinstance(item,types)}


def _single_global_gbsp(payload,path):
    """Recompute frozen M1 single-global PCA residual without writing a cache."""
    feature_path=Path(str(payload.get("source_feature_path","")))
    if not feature_path.is_file(): raise FileNotFoundError(f"source feature missing for {path}: {feature_path}")
    feature_payload=torch_load(feature_path,map_location="cpu"); feature=feature_payload.get("tensor")
    if not torch.is_tensor(feature) or tuple(feature.shape)!=(384,37,37): raise ValueError(f"invalid feature tensor: {feature_path}")
    query=F.normalize(feature.detach().cpu().float().permute(1,2,0).reshape(1369,384).contiguous(),p=2,dim=1)
    bc=payload.get("background_indices")
    if not torch.is_tensor(bc) or bc.ndim!=1 or bc.numel()==0: raise ValueError(f"background_indices missing: {path}")
    bc=bc.detach().cpu().long(); background=query.index_select(0,bc)
    projector=MultiBackgroundSubspaceProjector(num_subspaces=1,min_cluster_size=16,pca_energy=.90,pca_max_rank=8,
        pca_min_rank=1,seed=0,eps=1e-8,kmeans_n_init=10,kmeans_max_iter=100).fit(background)
    raw=projector.score(query).absolute_residual.reshape(1,37,37).float().contiguous()
    minimum,maximum=raw.min(),raw.max()
    score=torch.zeros_like(raw) if float(maximum-minimum)<1e-8 else ((raw-minimum)/(maximum-minimum+1e-8)).clamp(0,1)
    return score.contiguous(),raw,int(projector.selected_ranks[0]),str(feature_path.resolve())


def _reference(name,score,bc,fixed_threshold):
    if name=="fixed_058": return (score>fixed_threshold).float(),score,{"threshold_high":fixed_threshold}
    if name=="gbsp_fixed_050": return (score>.5).float(),score,{"threshold_high":.5}
    if name=="multi_otsu_3":
        t1,t2=map(float,threshold_multiotsu(score.double().reshape(-1).numpy(),classes=3,nbins=256))
        return (score>t2).float(),score,{"threshold_high":t2,"threshold_low":t1}
    if name in {"huto_core","huto_mid_h"}:
        variant="core" if name=="huto_core" else "mid_h"
        result=HierarchicalUpperTailOtsu(variant).apply(score,bc)
        return result.mask.float(),score,{"threshold_high":result.threshold_high,"threshold_low":result.threshold_low,
                                          "numerical_failure":result.numerical_failure,**_scalar(result.diagnostics)}
    raise ValueError(name)


def _process(task):
    try:
        path=Path(task["cache_path"]); payload=torch_load(path,map_location="cpu")
        identity=(task["dataset"],task["stem"])
        if (str(payload.get("dataset")),str(payload.get("stem")))!=identity: raise RuntimeError("identity mismatch")
        adaptive=task["adaptive"]
        direct=task.get("direct",False)
        if adaptive:
            forbidden=("gt_used_for_generation","r1_used_for_generation","fixed_058_used_for_generation","target_area_used_for_generation",
                       "fixed_topk_used_for_generation","dino_forward_used_for_generation","pca_changed_for_generation","morphology_used_for_generation",
                       "all_bc_hard_background_for_generation")
            if any(bool(payload.get(key,True)) for key in forbidden): raise RuntimeError("V5 independence violation")
            score=_map(payload,("minmax_residual",),path); source=torch_load(Path(payload["source_gbsp_path"]),map_location="cpu")
            available=payload.get("results",{}); method_names=task["methods"] or list(available)
            if set(method_names)-set(available): raise KeyError(set(method_names)-set(available))
        else:
            source=payload; available={}; method_names=[]
            if direct:
                score,direct_raw,direct_rank,direct_feature=_single_global_gbsp(payload,path)
            else:
                score=_map(payload,("absolute_minmax",),path)
        bc=payload.get("background_indices")
        if not torch.is_tensor(bc): raise ValueError("background_indices missing")
        bc=bc.detach().cpu().long()
        if direct:
            method_names=list(task["methods"])
            for name in method_names:
                if name!="smoh": raise ValueError(f"direct evaluation currently supports only smoh, got {name}")
                result=SeededMultiOtsuHysteresis().apply(score,bc)
                mask=result.mask.detach().cpu().float().contiguous()
                available[name]={"method":name,"method_family":name,"mask_37":mask,
                    "threshold_high":result.threshold_high,"threshold_low":result.threshold_low,
                    "foreground_area":float(mask.mean()),"empty_mask":bool(float(mask.mean())==0),
                    "area_over_50pct":bool(float(mask.mean())>.5),"numerical_failure":bool(result.numerical_failure),
                    "bc_final_ratio":float(mask.reshape(-1).index_select(0,bc).mean()),"diagnostics":{
                        **result.diagnostics,"direct_single_global_pca":True,"direct_selected_rank":direct_rank,
                        "direct_feature_path":direct_feature,"direct_raw_mean":float(direct_raw.mean())}}
        references=list(dict.fromkeys(task["references"])); masks={}; continuous={}; metadata={}
        for name in references:
            if name=="r1_fixed_050":
                dabe_path=Path(str(source.get("source_dabe_path",""))); dabe=torch_load(dabe_path,map_location="cpu")
                r1=_map(dabe,("residual_pass1_37",),dabe_path); mask=(r1>.5).float(); cont=r1; extra={"threshold_high":.5}
            else: mask,cont,extra=_reference(name,score,bc,task["fixed_threshold"])
            masks[name]=mask; continuous[name]=cont
            metadata[name]={"method_family":"reference","foreground_area":float(mask.mean()),"empty_mask":int(float(mask.mean())==0),
                            "area_over_50":int(float(mask.mean())>.5),"numerical_failure":int(extra.get("numerical_failure",False)),
                            "bc_final_ratio":float(mask.reshape(-1).index_select(0,bc).mean()) if name!="r1_fixed_050" else None,
                            **extra}
        if adaptive or direct:
            for name in method_names:
                item=available[name]; mask=item.get("mask_37")
                if not torch.is_tensor(mask) or tuple(mask.shape)!=(1,37,37): raise ValueError(f"invalid mask {name}")
                masks[name]=mask.detach().cpu().float(); continuous[name]=score
                diag=_scalar(item.get("diagnostics",{}))
                metadata[name]={"method_family":item.get("method_family",name),"foreground_area":float(item.get("foreground_area",mask.mean())),
                    "empty_mask":int(item.get("empty_mask",float(mask.mean())==0)),"area_over_50":int(item.get("area_over_50pct",float(mask.mean())>.5)),
                    "numerical_failure":int(item.get("numerical_failure",True)),"threshold_high":item.get("threshold_high"),
                    "threshold_low":item.get("threshold_low"),"bc_final_ratio":item.get("bc_final_ratio"),**diag}
        gt_path=Path(task["gt_path"] or payload.get("gt_path","")); gt=_load_gt(gt_path); native_shape=tuple(gt.shape[-2:]); context=FastCODContext(gt)
        rows=[]
        for name in (*references,*method_names):
            if name in {"fixed_058","gbsp_fixed_050","r1_fixed_050"}:
                threshold=task["fixed_threshold"] if name=="fixed_058" else .5
                native=(_resize(continuous[name],native_shape)>threshold).float()
            else: native=(_resize(masks[name],native_shape)>.5).float()
            info=metadata[name]
            rows.append({"dataset":identity[0],"stem":identity[1],"cache_path":str(path),
                "image_path":task["image_path"] or payload.get("image_path",""),"gt_path":str(gt_path),"method":name,
                **_binary_metrics(context,native),**_rank_metrics(_resize(continuous[name],native_shape),gt),**info})
        fixed=next(row for row in rows if row["method"]=="fixed_058")
        fields={"S_m":"delta_S_vs_fixed_058","F_beta_w":"delta_F_beta_w_vs_fixed_058","E_mean":"delta_E_vs_fixed_058",
                "MAE":"delta_MAE_vs_fixed_058","F_beta_mean":"delta_Fmean_vs_fixed_058","Precision":"delta_Precision_vs_fixed_058",
                "Recall":"delta_Recall_vs_fixed_058","Area":"delta_Area_vs_fixed_058","IoU":"delta_IoU_vs_fixed_058","Dice":"delta_Dice_vs_fixed_058"}
        for row in rows:
            for field,delta in fields.items(): row[delta]=row[field]-fixed[field]
        return {"rows":rows}
    except Exception as error:
        return {"dataset":task.get("dataset",""),"stem":task.get("stem",""),"error":repr(error),"traceback":traceback.format_exc()}


def _init(threads): torch.set_num_threads(int(threads))


def _mean(values):
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def _fields(rows):
    output=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen: output.append(key); seen.add(key)
    return output


def _enrich(summary,rows):
    extra=("numerical_failure","foreground_area","threshold_high","threshold_low","bc_final_ratio","pixel_AP","pixel_AUROC",
           "seed_area","support_area","final_area","retained_support_ratio","num_support_components","num_retained_components",
           "num_seeded_support_components","insufficient_growth_events_count","jump_detected_component_count",
           "foreground_seed_count","background_seed_count","bc_background_seed_count","boundary_background_seed_count",
           "graph_edge_count","solver_residual","probability_mean","equivalent_area","bc_selected_as_foreground_ratio")
    for aggregate in summary:
        subset=[row for row in rows if row["method"]==aggregate["method"] and (aggregate["dataset"]=="ALL" or row["dataset"]==aggregate["dataset"])]
        aggregate["method_family"]=subset[0].get("method_family","reference")
        for key in extra: aggregate[key]=_mean(row.get(key) for row in subset)


def _dominates(a,b):
    no_worse=all(a[key]>=b[key] for key in ("S_m","F_beta_w","E_mean")) and a["MAE"]<=b["MAE"]
    strict=any(a[key]>b[key] for key in ("S_m","F_beta_w","E_mean")) or a["MAE"]<b["MAE"]
    return bool(no_worse and strict)


def _pareto(rows): return [row for row in rows if not any(_dominates(other,row) for other in rows if other is not row)]


def _bootstrap(rows,candidates,repetitions,seed):
    rng=np.random.default_rng(seed); lookup={(r["dataset"],r["stem"],r["method"]):r for r in rows}; output=[]
    for method in candidates:
        for metric in PRIMARY:
            groups=[]
            for dataset in DATASETS:
                keys=[(r["dataset"],r["stem"]) for r in rows if r["dataset"]==dataset and r["method"]==method]
                groups.append(np.asarray([lookup[(*key,method)][metric]-lookup[(*key,"fixed_058")][metric] for key in keys]))
            estimates=np.empty(repetitions)
            for index in range(repetitions): estimates[index]=np.mean([g[rng.integers(0,g.size,g.size)].mean() for g in groups if g.size])
            output.append({"method":method,"metric":metric,"paired_macro_delta":float(np.mean([g.mean() for g in groups if g.size])),
                           "ci95_low":float(np.quantile(estimates,.025)),"ci95_high":float(np.quantile(estimates,.975)),
                           "repetitions":repetitions,"seed":seed})
    return output


def _win_rates(rows,candidates):
    output=[]
    for method in candidates:
        subset=[r for r in rows if r["method"]==method]
        for metric,delta in (("S_m","delta_S_vs_fixed_058"),("F_beta_w","delta_F_beta_w_vs_fixed_058"),("E_mean","delta_E_vs_fixed_058"),
                             ("MAE","delta_MAE_vs_fixed_058"),("F_beta_mean","delta_Fmean_vs_fixed_058"),("Precision","delta_Precision_vs_fixed_058"),
                             ("Recall","delta_Recall_vs_fixed_058"),("Area","delta_Area_vs_fixed_058"),("IoU","delta_IoU_vs_fixed_058"),("Dice","delta_Dice_vs_fixed_058")):
            values=np.asarray([float(r[delta]) for r in subset]); good=values<0 if metric=="MAE" else values>0
            output.append({"method":method,"metric":metric,"wins":int(good.sum()),"ties":int((np.abs(values)<=1e-12).sum()),
                           "losses":int((~good & (np.abs(values)>1e-12)).sum()),"win_rate":float(good.mean())})
    return output


def _selection(summary,rows,bootstrap,cfg,strict=False):
    macro={r["method"]:r for r in summary if r["scope"]=="dataset_macro"}; fixed=macro["fixed_058"]
    candidates=[name for name in macro if name not in BASELINE_NAMES]; ci={(r["method"],r["metric"]):r for r in bootstrap}
    margins=cfg.GBSP_V5_STRICT_MARGINS if strict else cfg.GBSP_V5_DIAGNOSTIC_MARGINS
    evaluations=[]; stability={}; bootstrap_status={}
    for method in candidates:
        row=macro[method]; delta={key:row[key]-fixed[key] for key in (*PRIMARY,*SECONDARY)}
        primary_checks={"S_m":delta["S_m"]>=margins["S_m"],"F_beta_w":delta["F_beta_w"]>=margins["F_beta_w"],
                        "E_mean":delta["E_mean"]>=margins["E_mean"],"MAE":delta["MAE"]<=margins["MAE"]}
        method_rows=[r for r in rows if r["method"]==method]; hard=cfg.GBSP_V5_HARD_FAILURE; reasons=[]
        if all(delta[key]<hard["triple_decline"] for key in ("S_m","F_beta_w","E_mean")): reasons.append("S_Fw_E_decline_over_0.01")
        if delta["MAE"]>hard["MAE_increase"]: reasons.append("MAE_hard_failure")
        if delta["Precision"] < -hard["Precision_drop"]: reasons.append("Precision_hard_failure")
        if not hard["Area_min"]<=row["Area"]<=hard["Area_max"]: reasons.append("Area_hard_failure")
        if _mean(r.get("empty_mask",0) for r in method_rows)>hard["empty_rate_max"]: reasons.append("empty_mask_rate")
        if _mean(r.get("area_over_50",0) for r in method_rows)>hard["area_over_50_rate_max"]: reasons.append("area_over_50_rate")
        if any(r.get("numerical_failure") for r in method_rows): reasons.append("numerical_failure")
        dataset_bad=[]
        for dataset in DATASETS:
            cand=next(r for r in summary if r["scope"]=="dataset" and r["dataset"]==dataset and r["method"]==method)
            ref=next(r for r in summary if r["scope"]=="dataset" and r["dataset"]==dataset and r["method"]=="fixed_058")
            dm=cfg.GBSP_V5_DATASET_MARGINS
            if cand["S_m"]-ref["S_m"]<dm["S_m"] or cand["F_beta_w"]-ref["F_beta_w"]<dm["F_beta_w"] or cand["E_mean"]-ref["E_mean"]<dm["E_mean"] or cand["MAE"]-ref["MAE"]>dm["MAE"]: dataset_bad.append(dataset)
        stability[method]={"stable":not dataset_bad,"unstable_datasets":dataset_bad}
        bm=cfg.GBSP_V5_BOOTSTRAP_MARGINS
        checks={"S_m":ci[(method,"S_m")]["ci95_low"]>bm["S_m"],"F_beta_w":ci[(method,"F_beta_w")]["ci95_low"]>bm["F_beta_w"],
                "E_mean":ci[(method,"E_mean")]["ci95_low"]>bm["E_mean"],"MAE":ci[(method,"MAE")]["ci95_high"]<bm["MAE"]}
        bootstrap_status[method]={"checks":checks,"passed_count":sum(checks.values()),"status":"PASS" if all(checks.values()) else "partial_noninferiority"}
        auxiliary=delta["Precision"]>=margins["Precision"] and margins["Area_min"]<=row["Area"]<=margins["Area_max"]
        if strict: auxiliary=auxiliary and delta["F_beta_mean"]>=margins["F_beta_mean"] and stability[method]["stable"] and all(checks.values())
        eligible=sum(primary_checks.values())>=3 and auxiliary and not reasons
        evaluations.append({"method":method,"eligible":eligible,"primary_checks":primary_checks,"primary_noninferior_count":sum(primary_checks.values()),
                            "hard_failure_reasons":reasons,"deltas":delta,"bootstrap":bootstrap_status[method],"dataset_stability":stability[method],
                            **{key:row[key] for key in (*PRIMARY,*SECONDARY)}})
    raw_front=_pareto(evaluations); eligible_front=_pareto([row for row in evaluations if row["eligible"]])
    eligible_front.sort(key=lambda row:(-row["primary_noninferior_count"],-row["bootstrap"]["passed_count"],-row["Precision"],
                                        -row["F_beta_mean"],{"smoh":0,"spcg":1,"bcmp":2}.get(row["method"],3)))
    selected=eligible_front[:1] if strict else eligible_front[:2]
    return {"schema":"gbsp_threshold_v5_method_selection","primary_metrics":list(PRIMARY),"secondary_metrics":list(SECONDARY),
            "pareto_front":[row["method"] for row in eligible_front],"raw_pareto_front":[row["method"] for row in raw_front],
            "bootstrap_noninferiority":bootstrap_status,"dataset_stability":stability,"evaluations":evaluations,
            "selected_method":selected[0]["method"] if strict and selected else "","selected_methods":[row["method"] for row in selected],
            "status":"PASS" if selected else "STOP","selection_reason":{"S_m":"primary Pareto/noninferiority","F_beta_w":"primary Pareto/noninferiority",
            "E_mean":"primary Pareto/noninferiority","MAE":"primary Pareto/noninferiority","Precision":"secondary constraint/tie-break",
            "Recall":"reported with Fmean and structure","Area":"range and stability constraint","dataset_stability":"all four datasets",
            "bootstrap":"paired 2000-repeat multi-metric noninferiority","simplicity":"SMOH > SPCG > BCMP only within 0.002 ties"}}


def _plots(out,stage,summary,rows,candidates):
    failures=[]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as error: return [{"plot":"all","error":repr(error)}]
    macro=[r for r in summary if r["scope"]=="dataset_macro"]
    def run(name,fn):
        try: fn(plt,out/name)
        except Exception as error: failures.append({"plot":name,"error":repr(error),"traceback":traceback.format_exc()})
    def primary(plt,path):
        methods=[r["method"] for r in macro]; x=np.arange(len(methods)); width=.2; fig,ax=plt.subplots(figsize=(max(9,len(methods)*1.5),4))
        for index,key in enumerate(PRIMARY): ax.bar(x+(index-1.5)*width,[r[key] for r in macro],width,label=key)
        ax.set_xticks(x,methods,rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def pra(plt,path):
        methods=[r["method"] for r in macro]; x=np.arange(len(methods)); width=.25; fig,ax=plt.subplots(figsize=(max(9,len(methods)*1.5),4))
        for offset,key in ((-width,"Precision"),(0,"Recall"),(width,"Area")): ax.bar(x+offset,[r[key] for r in macro],width,label=key)
        ax.set_xticks(x,methods,rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def pareto(plt,path):
        fig,ax=plt.subplots(figsize=(6,5))
        for row in macro: ax.scatter(row["F_beta_w"],row["S_m"],s=50); ax.annotate(row["method"],(row["F_beta_w"],row["S_m"]),fontsize=7)
        ax.set(xlabel="F_beta_w",ylabel="S_m"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def delta(field,label):
        def draw(plt,path):
            pairs=[]
            for method in candidates:
                values=[float(r[field]) for r in rows if r["method"]==method and r.get(field) is not None]
                if values:pairs.append((method,values))
            fig,ax=plt.subplots(figsize=(max(7,len(pairs)*1.4),4)); ax.boxplot([p[1] for p in pairs],tick_labels=[p[0] for p in pairs],showfliers=False)
            ax.axhline(0,color="black"); ax.set_ylabel(label); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
        return draw
    def dataset_plot(plt,path):
        methods=[r["method"] for r in macro]; metrics=("S_m","F_beta_w","F_beta_mean","E_mean","MAE","Precision")
        lookup={(r["dataset"],r["method"]):r for r in summary if r["scope"]=="dataset"}; fig,axes=plt.subplots(2,3,figsize=(18,9))
        for ax,metric in zip(axes.reshape(-1),metrics):
            x=np.arange(len(methods)); width=.18
            for index,dataset in enumerate(DATASETS): ax.bar(x+(index-1.5)*width,[lookup[(dataset,m)][metric] for m in methods],width,label=dataset)
            ax.set_xticks(x,methods,rotation=30,ha="right"); ax.set_title(metric)
        axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)
    def area_plot(plt,path):
        fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(candidates)); width=.25
        for offset,key in ((-width,"seed_area"),(0,"support_area"),(width,"foreground_area")):
            ax.bar(x+offset,[_mean(r.get(key) for r in rows if r["method"]==m) for m in candidates],width,label=key)
        ax.set_xticks(x,candidates); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    run(f"{stage}_primary_metrics.png",primary); run("precision_recall_area_comparison.png",pra); run("primary_metric_pareto.png",pareto)
    run("per_image_delta_S.png",delta("delta_S_vs_fixed_058","delta S")); run("per_image_delta_Fw.png",delta("delta_F_beta_w_vs_fixed_058","delta Fw"))
    run("per_image_delta_E.png",delta("delta_E_vs_fixed_058","delta E")); run("per_image_delta_MAE.png",delta("delta_MAE_vs_fixed_058","delta MAE"))
    run("seed_support_area_distribution.png",area_plot); run("dataset_wise_all_metrics.png",dataset_plot)
    return failures


def _component_tables(source_rows):
    smoh=[]; spcg=[]; bcmp=[]
    for source in source_rows:
        payload=torch_load(Path(source["cache_path"]),map_location="cpu")
        for name,item in payload.get("results",{}).items():
            diag=item.get("diagnostics",{}); base={"dataset":source["dataset"],"stem":source["stem"],"method":name}
            if name=="smoh": smoh.append({**base,**_scalar(diag)})
            elif name=="bcmp": bcmp.append({**base,**_scalar(diag)})
            elif name=="spcg":
                components=diag.get("component_diagnostics",[])
                if not components: spcg.append({**base,**_scalar(diag)})
                for component in components: spcg.append({**base,**component})
    return smoh,spcg,bcmp


def evaluate(args):
    if bool(args.gbsp_root)==bool(args.threshold_root): raise ValueError("provide exactly one of --gbsp_root/--threshold_root")
    cfg=load_config(_resolve(args.config)); adaptive=bool(args.threshold_root); direct=bool(args.direct_methods)
    if adaptive and direct: raise ValueError("--direct_methods is only valid with --gbsp_root")
    if direct and (set(args.direct_methods)-{"smoh"}): raise ValueError("direct evaluation currently supports only smoh")
    root=_resolve(args.threshold_root or args.gbsp_root)
    source_rows=_select(_manifest(_manifest_path(root)),args.sample_list,args.max_samples)
    if adaptive:
        methods=list(args.methods or [])
        references=list(args.compare or cfg.GBSP_V5_REFERENCES)
    elif direct:
        methods=list(args.direct_methods)
        references=list(args.compare or cfg.GBSP_V5_REFERENCES)
    else:
        references=list(args.methods or cfg.GBSP_V5_BASELINE_METHODS); methods=[]
    if "fixed_058" not in references: references.insert(0,"fixed_058")
    invalid=set(references)-BASELINE_NAMES
    if invalid: raise ValueError(f"invalid references: {invalid}")
    out=_resolve(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    tasks=[{"dataset":str(r["dataset"]),"stem":str(r["stem"]),"cache_path":r["cache_path"],"image_path":r.get("image_path",""),
            "gt_path":r.get("gt_path",""),"adaptive":adaptive,"methods":methods,"references":references,
            "direct":direct,"fixed_threshold":cfg.GBSP_V5_FIXED_THRESHOLD_REFERENCE_ONLY} for r in source_rows]
    started=time.perf_counter(); results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init,initargs=(args.torch_threads,)) as pool:
        for index,result in enumerate(pool.map(_process,tasks,chunksize=1),1):
            results.append(result)
            if index%20==0 or index==len(tasks): print(f"V5 eval {index}/{len(tasks)}",flush=True)
    failures=[r for r in results if "error" in r]; valid=[r for r in results if "error" not in r]
    _write_json(out/"evaluation_failures.json",failures)
    if not valid: raise RuntimeError("no valid evaluations")
    rows=[row for result in valid for row in result["rows"]]; summary=_aggregate(rows); _enrich(summary,rows)
    count=len(tasks); counts=dict(Counter(r["dataset"] for r in tasks))
    if not adaptive and not direct: stage="baseline_diagnostic200"
    elif count==200 and args.frozen_config: stage="pilot200"
    elif count==200: stage="diagnostic200"
    elif count==sum(EXPECTED.values()) and counts==EXPECTED: stage="full6473"
    else: stage="custom"
    macro={r["method"]:r for r in summary if r["scope"]=="dataset_macro"}
    errors={name:{key:abs(macro[name][key]-value) for key,value in target.items()} for name,target in cfg.GBSP_V5_BASELINE_TARGETS.items() if name in macro}
    reproduction=bool(not adaptive and not direct and set(errors)==set(cfg.GBSP_V5_BASELINE_TARGETS) and max(max(v.values()) for v in errors.values())<cfg.GBSP_V5_BASELINE_TOLERANCE)
    candidates=sorted({r["method"] for r in rows if r["method"] not in BASELINE_NAMES})
    bootstrap=_bootstrap(rows,candidates,args.bootstrap_repetitions,args.bootstrap_seed) if candidates else []
    selection=_selection(summary,rows,bootstrap,cfg,strict=stage in {"pilot200","full6473"}) if candidates else None
    audit={"schema":"gbsp_threshold_v5_eval","stage":stage,"num_requested":count,"num_valid":len(valid),"num_failed":len(failures),
           "dataset_counts":counts,"baseline_reproduction_errors":errors,
           "baseline_reproduction_status":"PASS" if reproduction else ("NOT_APPLICABLE" if adaptive or direct else "FAIL"),
           "gt_used_for_calibration":False,"fixed_058_used_in_v5_formula":False,"wall_seconds":time.perf_counter()-started}
    _write_json(out/"numerical_failure_summary.json",audit)
    _write_csv(out/"per_image_metrics.csv",rows,_fields(rows)); _write_csv(out/"per_dataset_metrics.csv",summary,_fields(summary))
    primary=[{k:r.get(k) for k in ("dataset","stem","method","delta_S_vs_fixed_058","delta_F_beta_w_vs_fixed_058","delta_E_vs_fixed_058","delta_MAE_vs_fixed_058")} for r in rows]
    secondary=[{k:r.get(k) for k in ("dataset","stem","method","delta_Fmean_vs_fixed_058","delta_Precision_vs_fixed_058","delta_Recall_vs_fixed_058","delta_Area_vs_fixed_058","delta_IoU_vs_fixed_058","delta_Dice_vs_fixed_058")} for r in rows]
    _write_csv(out/"primary_metric_deltas.csv",primary,_fields(primary)); _write_csv(out/"secondary_metric_deltas.csv",secondary,_fields(secondary))
    _write_csv(out/"bootstrap_ci95.csv",bootstrap,_fields(bootstrap)); wins=_win_rates(rows,candidates) if candidates else []
    _write_csv(out/"per_image_win_rates.csv",wins,_fields(wins))
    if adaptive:
        smoh,spcg,bcmp=_component_tables(source_rows)
        _write_csv(out/"smoh_diagnostics.csv",smoh,_fields(smoh)); _write_csv(out/"spcg_component_diagnostics.csv",spcg,_fields(spcg))
        _write_csv(out/"bcmp_solver_diagnostics.csv",bcmp,_fields(bcmp))
    elif direct:
        smoh=[{key:row.get(key) for key in row if key not in (*PRIMARY,*SECONDARY,"pixel_AP","pixel_AUROC")} for row in rows if row["method"]=="smoh"]
        _write_csv(out/"smoh_diagnostics.csv",smoh,_fields(smoh))
    filename={"baseline_diagnostic200":"baseline_diagnostic200.csv","diagnostic200":"diagnostic200_summary.csv","pilot200":"pilot200_summary.csv","full6473":"full6473_summary.csv"}.get(stage,"summary.csv")
    _write_csv(out/filename,list(macro.values()),_fields(list(macro.values())))
    if selection:
        _write_json(out/"method_selection.json",selection); _write_json(out/"pareto_front.json",selection["pareto_front"])
        frozen={"schema":"gbsp_threshold_v5_frozen_config","source_stage":stage,"selected_methods":selection["selected_methods"],
                "parameters_frozen":True,"gt_used":False,"fixed_058_used":False,"target_area_used":False}
        _write_json(out/("final_config.json" if stage=="pilot200" else "selected_config.json"),frozen)
    _write_csv(out/"downstream_1x1_results.csv",[{"method":m,"status":"not_run_full_gate_required"} for m in (selection or {}).get("selected_methods",[])],
               ["method","status"])
    plot_failures=_plots(out,stage,summary,rows,candidates); _write_json(out/"visualization_failures.json",plot_failures)
    report=["# GBSP Threshold V5 Report","",f"- Stage: {stage}",f"- Baseline reproduction: {audit['baseline_reproduction_status']}",
            f"- Selection: {(selection or {}).get('status','N/A')}","",_table(list(macro.values()),("method",*PRIMARY,*SECONDARY))]
    (out/"GBSP_THRESHOLD_V5_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({**audit,"selection":selection},ensure_ascii=False,indent=2))
    if failures and args.failure_policy=="strict": raise RuntimeError(f"{len(failures)} evaluation failures")


def parser():
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--config",default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp_root"); value.add_argument("--threshold_root"); value.add_argument("--sample_list")
    value.add_argument("--split",default="test",choices=("test",)); value.add_argument("--max_samples",type=int,default=-1)
    value.add_argument("--methods",nargs="+"); value.add_argument("--compare",nargs="+"); value.add_argument("--frozen_config")
    value.add_argument("--direct_methods",nargs="+",help="compute supported V5 methods directly from GBSP without writing .pt caches")
    value.add_argument("--bootstrap_repetitions",type=int,default=2000); value.add_argument("--bootstrap_seed",type=int,default=20260806)
    value.add_argument("--out_dir",required=True); value.add_argument("--workers",type=int,default=4); value.add_argument("--torch_threads",type=int,default=1)
    value.add_argument("--failure-policy",choices=("record","strict"),default="record"); return value


if __name__=="__main__": evaluate(parser().parse_args())
