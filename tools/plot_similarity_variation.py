#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.similarity_variation_common import labels_from_area, patch_gt, read_jsonl

def read_csv(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def save(fig,path): fig.tight_layout(); fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)
def bars(rows,xkey,ykeys,title,path):
 labels=[r[xkey] for r in rows]; x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(max(6,len(labels)*.8),4))
 for i,k in enumerate(ykeys): ax.bar(x+(i-(len(ykeys)-1)/2)*.25,[float(r.get(k,'nan')) for r in rows],.24,label=k)
 ax.set_xticks(x,labels,rotation=30,ha='right'); ax.set_title(title); ax.legend(); save(fig,path)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--score_root',required=True); p.add_argument('--analysis_root',required=True)
 p.add_argument('--deterministic_samples_per_class_per_image',type=int,default=100); p.add_argument('--summary_only',action='store_true'); p.add_argument('--out_dir',required=True); a=p.parse_args()
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); ar=Path(a.analysis_root); sr=Path(a.score_root)
 # The formal score audit is evaluated after 37->68->native bilinear resize.
 # Use its frozen dataset-macro row for the overall bar chart.
 full=[r for r in read_csv(sr/'full_continuous_metrics.csv')
       if r['protocol']=='dataset_macro_of_per_image_native' and r['dataset']=='DATASET_MACRO']
 bars(full,'method',['ap','auroc'],'Direct matching vs GBSP',out/'direct_matching_vs_gbsp.png')
 ds=[r for r in read_csv(sr/'per_dataset_metrics.csv')
     if r['protocol']=='per_image_macro_native_gt' and r['dataset']!='ALL']
 fig,axs=plt.subplots(1,2,figsize=(12,4));
 for ax,key in zip(axs,['ap','auroc']):
  for m in sorted({r['method'] for r in ds}):
   rr=[r for r in ds if r['method']==m]; ax.plot([r['dataset'] for r in rr],[float(r[key]) for r in rr],marker='o',label=m)
  ax.set_title(key.upper()); ax.tick_params(axis='x',rotation=30)
 axs[0].legend(fontsize=7); save(fig,out/'per_dataset_ap_auroc.png')
 if a.summary_only:
  print(f'summary figures written to {out}')
  return
 bars(read_csv(sr/'precision_at_recall.csv'),'method',['p_at_r50','p_at_r60','p_at_r70'],'Precision at recall',out/'precision_at_recall.png')
 hs=[]
 for h in ('H20','H30','H40'): hs += [r for r in read_csv(ar/f'high_similarity_{h}.csv') if r['dataset']=='ALL' and r['label_rule']=='main_0.5']
 for metric,name in [('ap','high_similarity_ap.png'),('auroc','high_similarity_auroc.png'),('p_at_r50','high_similarity_precision_at_recall.png')]:
  fig,ax=plt.subplots(figsize=(7,4))
  for m in ('nn_cos','knn8_cos','gbsp_r8'):
   rr=[r for r in hs if r['method']==m]; ax.plot([r['subset'] for r in rr],[float(r[metric]) for r in rr],marker='o',label=m)
  ax.set_title(metric); ax.legend(); save(fig,out/name)
 bins=read_csv(ar/'similarity_bin_analysis.csv'); br=[r for r in bins if r['dataset']=='ALL']
 fig,ax=plt.subplots(figsize=(7,4))
 for m in ('nn_cos','gbsp_r8'):
  rr=[r for r in br if r['method']==m]; ax.plot([int(r['bin']) for r in rr],[float(r['auroc']) for r in rr],marker='o',label=m)
 ax.legend(); ax.set_xlabel('similarity bin'); ax.set_ylabel('AUROC'); save(fig,out/'similarity_bin_auroc.png')
 fig,ax=plt.subplots(figsize=(7,4)); rr=[r for r in br if r['method']=='gbsp_r8']; ax.errorbar([int(r['bin']) for r in rr],[float(r['fg_median']) for r in rr],label='FG median'); ax.errorbar([int(r['bin']) for r in rr],[float(r['bg_median']) for r in rr],label='BG median'); ax.legend(); save(fig,out/'similarity_bin_residual_distribution.png')
 pairs=read_csv(ar/'matched_pair_analysis.csv'); bars([r for r in pairs if r['dataset']!='Overall'],'dataset',['delta_q_mean','delta_q_median'],'Matched pair delta Q',out/'matched_pair_delta_q.png')

 manifest=read_jsonl(sr/'score_manifest.jsonl'); samples=[]; examples=[]
 for row in manifest:
  payload=torch.load(row['score_path'],map_location='cpu',weights_only=False); sim=payload['nn_background_similarity'].numpy().reshape(-1); q=payload['scores']['gbsp_r8'].numpy().reshape(-1)
  y,valid=labels_from_area(patch_gt(row['gt_path']),False); per=a.deterministic_samples_per_class_per_image
  for label in (0,1):
   idx=np.flatnonzero(valid&(y==label)); idx=idx[np.linspace(0,len(idx)-1,min(per,len(idx)),dtype=int)] if len(idx) else idx
   samples.append((row['dataset'],label,sim[idx],q[idx]))
  if len(examples)<8: examples.append((row,sim.reshape(37,37),q.reshape(37,37)))
 if samples:
  simfg=np.concatenate([x[2] for x in samples if x[1]==1]); simbg=np.concatenate([x[2] for x in samples if x[1]==0]); qfg=np.concatenate([x[3] for x in samples if x[1]==1]); qbg=np.concatenate([x[3] for x in samples if x[1]==0])
  for aa,bb,title,name in [(simfg,simbg,'NN background similarity','similarity_fg_bg_histogram.png'),(qfg,qbg,'GBSP residual','gbsp_fg_bg_histogram.png')]:
   fig,ax=plt.subplots(figsize=(6,4)); ax.hist(bb,80,density=True,alpha=.5,label='BG'); ax.hist(aa,80,density=True,alpha=.5,label='FG'); ax.legend(); ax.set_title(title); save(fig,out/name)
  for name,mask in [('similarity_variation_scatter.png',np.ones(len(simfg),bool)),('similarity_variation_highsim_zoom.png',simfg>=np.quantile(np.r_[simfg,simbg],.8))]:
   fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(simbg[:20000],qbg[:20000],s=2,alpha=.15,label='BG'); ax.scatter(simfg[mask][:20000],qfg[mask][:20000],s=2,alpha=.2,label='FG'); ax.legend(); ax.set_xlabel('NN similarity'); ax.set_ylabel('GBSP Q'); save(fig,out/name)
  fig,ax=plt.subplots(figsize=(6,5)); ax.hexbin(np.r_[simbg,simfg],np.r_[qbg,qfg],gridsize=70,bins='log'); ax.set_xlabel('NN similarity'); ax.set_ylabel('GBSP Q'); save(fig,out/'similarity_variation_density.png')
  fig,axs=plt.subplots(2,2,figsize=(10,8));
  for ax,dsname in zip(axs.flat,sorted({x[0] for x in samples})):
   z=[x for x in samples if x[0]==dsname]; ax.scatter(np.concatenate([x[2] for x in z]),np.concatenate([x[3] for x in z]),s=2,alpha=.2); ax.set_title(dsname)
  save(fig,out/'per_dataset_similarity_variation.png')
 if examples:
  fig,axs=plt.subplots(len(examples),4,figsize=(12,3*len(examples)),squeeze=False)
  for axsrow,(row,sim,q) in zip(axs,examples):
   axsrow[0].imshow(Image.open(row['image_path'])); axsrow[0].set_title(row['stem']); axsrow[1].imshow(Image.open(row['gt_path']),cmap='gray'); axsrow[1].set_title('GT'); axsrow[2].imshow(sim,cmap='viridis'); axsrow[2].set_title('NN similarity'); axsrow[3].imshow(q,cmap='magma'); axsrow[3].set_title('GBSP Q')
   for ax in axsrow:ax.axis('off')
  save(fig,out/'example_images_high_similarity.png')
 print(f'figures written to {out}')
if __name__=='__main__':main()
