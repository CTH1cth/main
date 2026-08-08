#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import mannwhitneyu

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.similarity_variation_common import (binary_metrics, equal_frequency_bins, labels_from_area,
    load_sample_ids, patch_gt, quantile_mask, read_jsonl, sample_selected, similarity_matched_pairs)

QMAP={"H20":.80,"H30":.70,"H40":.60}; METHODS=("nn_cos","knn8_cos","gbsp_r8")

def args():
 p=argparse.ArgumentParser(); p.add_argument("--score_root",required=True); p.add_argument("--split",default="test")
 p.add_argument("--sample_list"); p.add_argument("--max_samples",type=int,default=-1)
 p.add_argument("--high_similarity_quantiles",nargs=3,type=float,default=[.8,.7,.6]); p.add_argument("--similarity_bins",type=int,default=5)
 p.add_argument("--matched_similarity_tolerance",type=float,default=.01); p.add_argument("--bootstrap_repetitions",type=int,default=0)
 p.add_argument("--bootstrap_seed",type=int,default=20260807); p.add_argument("--save_diagnostics",action="store_true")
 p.add_argument("--out_dir",required=True); return p.parse_args()

def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r}) if rows else ["empty"]
 with Path(path).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_csv(path):
 with Path(path).open() as f:return list(csv.DictReader(f))

def summarize_cond(rows, subset):
 out=[]
 for rule in ("main_0.5","strict_0.8_0.2"):
  for ds in sorted({r['dataset'] for r in rows})+["ALL"]:
   for m in METHODS:
    selected=[r for r in rows if r['subset']==subset and r['label_rule']==rule and r['method']==m and (ds=="ALL" or r['dataset']==ds)]
    valid=[r for r in selected if not r['conditional_metric_invalid']]
    out.append({"dataset":ds,"subset":subset,"label_rule":rule,"method":m,"protocol":"per_image_macro_patch37",
      "valid_image_count":len(valid),"invalid_image_count":len(selected)-len(valid),
      "valid_foreground_patch_count":sum(r['fg_patches'] for r in valid),"valid_background_patch_count":sum(r['bg_patches'] for r in valid),
      **{k:(float(np.mean([r[k] for r in valid])) if valid else np.nan) for k in ('ap','auroc','p_at_r50','p_at_r60')}})
 return out

def bootstrap_differences(rows,reps,seed):
 if reps<=0:return []
 rng=np.random.default_rng(seed); out=[]
 comparisons=[("gbsp_r8","nn_cos"),("gbsp_r8","knn8_cos")]
 for subset in ("FULL","H20","H30","H40"):
  current=[r for r in rows if r['subset']==subset and r['label_rule']=="main_0.5" and not r['conditional_metric_invalid']]
  by={(r['dataset'],r['stem'],r['method']):r for r in current}
  for a,b in comparisons:
   keys=sorted({(d,s) for d,s,m in by if m==a and (d,s,b) in by})
   for metric in ('ap','auroc','p_at_r50'):
    dif=np.array([by[(d,s,a)][metric]-by[(d,s,b)][metric] for d,s in keys],float)
    if not len(dif):continue
    boot=np.mean(dif[rng.integers(0,len(dif),size=(reps,len(dif)))],axis=1)
    out.append({"subset":subset,"metric":metric,"method_a":a,"method_b":b,"valid_images":len(dif),
      "mean_difference":float(dif.mean()),"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),
      "bootstrap_repetitions":reps,"bootstrap_seed":seed})
 return out

def main():
 a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 manifest=read_jsonl(Path(a.score_root)/"score_manifest.jsonl"); ids=load_sample_ids(a.sample_list)
 manifest=[r for r in manifest if sample_selected(r,ids)]; manifest=manifest if a.max_samples<0 else manifest[:a.max_samples]
 qmap=dict(zip(("H20","H30","H40"),a.high_similarity_quantiles))
 cond_rows=[]; full_rows=[]; bin_values=defaultdict(lambda:{"y":[],"sim":[],"q":[]}); pair_images=[]; distributions=defaultdict(list)
 for n,row in enumerate(manifest,1):
  p=torch.load(row['score_path'],map_location='cpu',weights_only=False)
  scores={k:v.numpy().reshape(-1) for k,v in p['scores'].items()}; sim=p['nn_background_similarity'].numpy().reshape(-1)
  area=patch_gt(row['gt_path']); bins=equal_frequency_bins(sim,a.similarity_bins)
  for rule,strict in (("main_0.5",False),("strict_0.8_0.2",True)):
   y,valid_gt=labels_from_area(area,strict)
   for m in METHODS:
    met=binary_metrics(y[valid_gt],scores[m][valid_gt],targets=(.5,.6))
    full_rows.append({"dataset":row['dataset'],"stem":row['stem'],"subset":"FULL","label_rule":rule,"method":m,
      "conditional_metric_invalid":not np.isfinite(met['ap']),"fg_patches":int((valid_gt&(y==1)).sum()),"bg_patches":int((valid_gt&(y==0)).sum()),**met})
   for h,q in qmap.items():
    subset=quantile_mask(sim,q) # GT-free by construction
    for m in METHODS:
     keep=subset&valid_gt; nf=int((keep&(y==1)).sum()); nb=int((keep&(y==0)).sum()); invalid=nf<3 or nb<3
     met=binary_metrics(y[keep],scores[m][keep],targets=(.5,.6)) if not invalid else {"ap":np.nan,"auroc":np.nan,"p_at_r50":np.nan,"p_at_r60":np.nan}
     cond_rows.append({"dataset":row['dataset'],"stem":row['stem'],"subset":h,"quantile":q,"label_rule":rule,"method":m,
       "conditional_metric_invalid":invalid,"subset_patches":int(subset.sum()),"fg_patches":nf,"bg_patches":nb,**met})
  y,valid=labels_from_area(area,False)
  for b in range(1,a.similarity_bins+1):
   keep=(bins==b)&valid; key=(row['dataset'],b); bin_values[key]['y'].append(y[keep]); bin_values[key]['sim'].append(sim[keep]); bin_values[key]['q'].append(scores['gbsp_r8'][keep])
  fg,bg=similarity_matched_pairs(sim,y,valid,a.matched_similarity_tolerance)
  high=fg[sim[fg]>=np.quantile(sim,.8)] if len(fg) else fg
  if len(fg):
   bg_lookup={int(f):int(b) for f,b in zip(fg,bg)}; hbg=np.array([bg_lookup[int(f)] for f in high],int)
  else:hbg=np.array([],int)
  pair_images.append({"dataset":row['dataset'],"stem":row['stem'],"all_delta":scores['gbsp_r8'][fg]-scores['gbsp_r8'][bg],
                      "high_delta":scores['gbsp_r8'][high]-scores['gbsp_r8'][hbg]})
  for label in (0,1):
   distributions[(row['dataset'],label,'nn_similarity')].extend(sim[valid&(y==label)].tolist())
   distributions[(row['dataset'],label,'gbsp_r8')].extend(scores['gbsp_r8'][valid&(y==label)].tolist())
  if n%500==0:print(f"[{n}/{len(manifest)}]",flush=True)
 # FULL bootstrap must use the formal native-size per-image protocol, not patch GT.
 native_full=[]
 for r in read_csv(Path(a.score_root)/'per_image_metrics.csv'):
  if r['method'] in METHODS:
   native_full.append({"dataset":r['dataset'],"stem":r['stem'],"subset":"FULL","label_rule":"main_0.5","method":r['method'],
    "conditional_metric_invalid":False,**{k:float(r[k]) for k in ('ap','auroc','p_at_r50','p_at_r60')}})
 all_metric_rows=native_full+cond_rows
 cond_summaries={h:summarize_cond(cond_rows,h) for h in qmap}
 for h,summary in cond_summaries.items(): write_csv(out/f"high_similarity_{h}.csv",summary)
 bin_rows=[]
 for ds in sorted({r['dataset'] for r in manifest})+["ALL"]:
  for b in range(1,a.similarity_bins+1):
   keys=[k for k in bin_values if k[1]==b and (ds=="ALL" or k[0]==ds)]
   y=np.concatenate([v for k in keys for v in bin_values[k]['y']]); simv=np.concatenate([v for k in keys for v in bin_values[k]['sim']]); qv=np.concatenate([v for k in keys for v in bin_values[k]['q']])
   for method,v in (("nn_cos",1-simv),("gbsp_r8",qv)):
    met=binary_metrics(y,v,targets=(.5,.6)); fg=v[y==1]; bg=v[y==0]; u=mannwhitneyu(fg,bg,alternative='two-sided').statistic if len(fg) and len(bg) else np.nan
    bin_rows.append({"dataset":ds,"bin":b,"method":method,"patches":len(y),"fg_patches":len(fg),"bg_patches":len(bg),**met,
      "fg_median":float(np.median(fg)) if len(fg) else np.nan,"bg_median":float(np.median(bg)) if len(bg) else np.nan,
      "fg_iqr":float(np.subtract(*np.percentile(fg,[75,25]))) if len(fg) else np.nan,"bg_iqr":float(np.subtract(*np.percentile(bg,[75,25]))) if len(bg) else np.nan,
      "prob_fg_gt_bg":float(u/(len(fg)*len(bg))) if len(fg) and len(bg) else np.nan,"cliffs_delta":float(2*u/(len(fg)*len(bg))-1) if len(fg) and len(bg) else np.nan})
 write_csv(out/"similarity_bin_analysis.csv",bin_rows)
 def pair_summary(which):
  rows=[]
  for ds in sorted({x['dataset'] for x in pair_images})+["Overall"]:
   arrays=[x[which] for x in pair_images if ds=="Overall" or x['dataset']==ds]; delta=np.concatenate([x for x in arrays if len(x)]) if any(len(x) for x in arrays) else np.array([])
   image_means=np.array([x.mean() for x in arrays if len(x)]); rng=np.random.default_rng(a.bootstrap_seed)
   if a.bootstrap_repetitions and len(image_means): boot=np.mean(image_means[rng.integers(0,len(image_means),(a.bootstrap_repetitions,len(image_means)))],1); lo,hi=np.quantile(boot,[.025,.975])
   else:lo=hi=np.nan
   rows.append({"dataset":ds,"valid_pairs":len(delta),"valid_images":len(image_means),"gbsp_pair_win_rate":float(np.mean(delta>0)) if len(delta) else np.nan,
    "delta_q_mean":float(np.mean(delta)) if len(delta) else np.nan,"delta_q_median":float(np.median(delta)) if len(delta) else np.nan,"ci95_low":lo,"ci95_high":hi})
  return rows
 pair_all=pair_summary('all_delta'); pair_high=pair_summary('high_delta'); write_csv(out/'matched_pair_analysis.csv',pair_all); write_csv(out/'high_similarity_matched_pair_analysis.csv',pair_high)
 dist_rows=[]
 for (ds,label,score),vals in distributions.items():
  v=np.asarray(vals); dist_rows.append({"dataset":ds,"class":"foreground" if label else "background","score":score,"count":len(v),"mean":v.mean(),"std":v.std(),"median":np.median(v),"q25":np.quantile(v,.25),"q75":np.quantile(v,.75)})
 write_csv(out/'score_distribution_statistics.csv',dist_rows)
 boot=bootstrap_differences(all_metric_rows,a.bootstrap_repetitions,a.bootstrap_seed); write_csv(out/'bootstrap_ci95.csv',boot)
 validity={"images":len(manifest),"conditional_rows":len(cond_rows),"subset_generation_uses_gt":False,"high_similarity_quantiles":qmap,
  "strict_gt_rule":{"foreground":">=0.8","background":"<=0.2","ignored":"(0.2,0.8)"},"bootstrap_repetitions":a.bootstrap_repetitions}
 validity['conditional_validity']={h:{rule:{"valid":sum(int(r['valid_image_count']) for r in cond_summaries[h] if r['dataset']!='ALL' and r['label_rule']==rule and r['method']=='gbsp_r8'),
   "invalid":sum(int(r['invalid_image_count']) for r in cond_summaries[h] if r['dataset']!='ALL' and r['label_rule']==rule and r['method']=='gbsp_r8')} for rule in ('main_0.5','strict_0.8_0.2')} for h in qmap}
 (out/'validity_summary.json').write_text(json.dumps(validity,indent=2,ensure_ascii=False))
 formal=read_csv(Path(a.score_root)/'full_continuous_metrics.csv')
 formal={r['method']:r for r in formal if r['dataset']=='DATASET_MACRO' and r['protocol']=='dataset_macro_of_per_image_native'}
 main_h={h:{r['method']:r for r in cond_summaries[h] if r['dataset']=='ALL' and r['label_rule']=='main_0.5'} for h in qmap}
 pair_a=next((r for r in pair_all if r['dataset']=='Overall'),{}); pair_h=next((r for r in pair_high if r['dataset']=='Overall'),{})
 complete=len(manifest)==6473; status="全量6473正式分析" if complete else f"Test20实现检查（当前{len(manifest)}张，以下数值不得形成论文结论）"
 def m(method,key): return float(formal[method][key]) if method in formal else float('nan')
 def hm(h,method,key): return float(main_h[h][method][key]) if method in main_h[h] else float('nan')
 report="# GBSP Similarity Ambiguity vs. Variation Consistency\n\n本目录由预声明协议自动生成。\n\n"
 report+="## 协议护栏\n\n- Full BC、DINOv1-S/8、37×37和GBSP-r8均冻结。\n- H20/H30/H40完全由逐图NN背景相似度分位数产生，GT未参与subset生成。\n- BC查询采用leave-one-out；所有分数均为越高越前景。\n\n## 结论状态\n\n"
 report+=f"当前状态：**{status}**。\n\n## 任务书23项回答\n\n"
 answers=[
 "FG/BG直接相似度是否高度重叠：需结合`score_distribution_statistics.csv`及直方图；全量完成后才正式判断。",
 f"NN背景匹配 AP（正式原图、数据集宏平均）：{m('nn_cos','ap'):.6f}。",
 f"NN背景匹配 AUROC：{m('nn_cos','auroc'):.6f}。",
 f"KNN8 AP/AUROC：{m('knn8_cos','ap'):.6f}/{m('knn8_cos','auroc'):.6f}。",
 f"GBSP相对NN：AP差{m('gbsp_r8','ap')-m('nn_cos','ap'):+.6f}，AUROC差{m('gbsp_r8','auroc')-m('nn_cos','auroc'):+.6f}。",
 f"GBSP相对KNN8：AP差{m('gbsp_r8','ap')-m('knn8_cos','ap'):+.6f}，AUROC差{m('gbsp_r8','auroc')-m('knn8_cos','auroc'):+.6f}。",
 "固定Recall Precision：见`precision_at_recall.csv`；以paired bootstrap判断稳定性。",
 f"H20中NN/KNN AP：{hm('H20','nn_cos','ap'):.6f}/{hm('H20','knn8_cos','ap'):.6f}。",
 f"H20中GBSP AP/AUROC：{hm('H20','gbsp_r8','ap'):.6f}/{hm('H20','gbsp_r8','auroc'):.6f}。",
 "H30/H40方向：见三个`high_similarity_*.csv`，全量后判断一致性。",
 "四数据集条件方向：各CSV已逐数据集完整报告，不隐藏不利结果。",
 "最高相似Bin内相似度重叠：见`similarity_bin_analysis.csv`的Bin-5及分布图。",
 "同Bin内GBSP残差分离：见Bin-4/5的概率优势、Cliff's delta、AP/AUROC。",
 f"Similarity-matched有效pair数：{pair_a.get('valid_pairs','NA')}。",
 f"控制相似度后GBSP pair win rate：{pair_a.get('gbsp_pair_win_rate','NA')}。",
 f"高相似matched pair win rate：{pair_h.get('gbsp_pair_win_rate','NA')}。",
 "Bootstrap条件优势：见`bootstrap_ci95.csv`；CI排除0才算支持。",
 "Rank-0证据链：既有Mean Prototype 0.7461/0.9436与GBSP 0.7726/0.9524构成互补证据，本实验进一步对比离散相似度。",
 "核心Observation：仅在全量H20/H30/H40、matched pairs和bootstrap共同支持时才可直接采用。",
 "Introduction核心叙事：Test20阶段不判断；全量完成后按任务书强/部分/不支持规则判断。",
 "RISE迁移：不是本任务必要条件；仅在机制证据成立后作为框架外部验证。",
 "UCOD-DPL迁移：同上，不得混入当前机制实验。",
 "措辞边界：可比较“离散背景匹配”和“图像特定背景变化子空间一致性”；不得声称所有UCOD都是相似度方法、相似度无法用于COD或所有前景均在子空间外。",
 ]
 report+='\n'.join(f"{i}. {x}" for i,x in enumerate(answers,1))+"\n"
 (out/'GBSP_SIMILARITY_VARIATION_REPORT.md').write_text(report)
 print(json.dumps(validity,ensure_ascii=False))
if __name__=='__main__':main()
