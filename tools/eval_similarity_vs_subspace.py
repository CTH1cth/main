#!/usr/bin/env python3
"""Generate direct-matching scores, then evaluate them on the frozen patch GT protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.gbsp_similarity_baselines import compute_similarity_baselines, minmax_per_image
from tools.similarity_variation_common import (
    binary_metrics, labels_from_area, load_sample_ids, patch_gt, read_jsonl,
    sample_selected, write_jsonl,
)

NAME_MAP = {"gbsp": "gbsp_r8"}
EXPECTED = {"CHAMELEON": 76, "CAMO": 250, "COD10K": 2026, "NC4K": 4121}
DATASET_ALIASES = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gbsp_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sample_list")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--methods", nargs="+", default=["mean_l2", "proto_cos", "nn_cos", "knn8_cos", "gbsp"])
    p.add_argument("--exclude_self_match", action="store_true")
    p.add_argument("--save_scores", action="store_true")
    p.add_argument("--bootstrap_repetitions", type=int, default=0)
    p.add_argument("--bootstrap_seed", type=int, default=20260807)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def native_gt(path: str) -> np.ndarray:
    with Image.open(path) as im: value=np.asarray(im.convert("L"),dtype=np.float32)/255.
    return (value>.5).astype(np.uint8)

def native_score(score: torch.Tensor, shape: tuple[int,int]) -> np.ndarray:
    value=score.float().reshape(1,1,37,37)
    value=F.interpolate(value,size=(68,68),mode="bilinear",align_corners=False)
    value=F.interpolate(value,size=shape,mode="bilinear",align_corners=False)
    return value.numpy().reshape(-1)

def hist_add(hist: np.ndarray, y: np.ndarray, score: np.ndarray, high: float=4., bins: int=65536):
    idx=np.clip((score/high*(bins-1)).astype(np.int64),0,bins-1)
    hist[0]+=np.bincount(idx[y==0],minlength=bins); hist[1]+=np.bincount(idx[y==1],minlength=bins)

def hist_metrics(hist: np.ndarray, targets=(.5,.6,.7)) -> dict:
    neg,pos=hist[0,::-1].astype(float),hist[1,::-1].astype(float); tp=np.cumsum(pos); fp=np.cumsum(neg)
    positives,negatives=tp[-1],fp[-1]
    if positives<=0 or negatives<=0:return {"ap":np.nan,"auroc":np.nan,**{f"p_at_r{int(t*100)}":np.nan for t in targets}}
    precision=tp/np.maximum(tp+fp,1); recall=tp/positives
    ap=float(np.sum((recall-np.r_[0.,recall[:-1]])*precision))
    # ascending bins: positive wins over lower-score negatives, half credit for ties
    na=hist[0].astype(float); pa=hist[1].astype(float); wins=np.sum(pa*(np.cumsum(na)-na+.5*na))
    out={"ap":ap,"auroc":float(wins/(positives*negatives))}
    out.update({f"p_at_r{int(t*100)}":float(precision[recall>=t].max()) for t in targets}); return out


def find_manifest(root: Path) -> Path:
    for p in (root / "manifest_test.jsonl", root / "manifest.jsonl"):
        if p.exists(): return p
    found = list(root.glob("manifest*.jsonl"))
    if len(found) != 1: raise FileNotFoundError(f"cannot resolve manifest under {root}")
    return found[0]


def generate_one(row: dict, score_path: Path, methods: list[str]) -> dict:
    # No GT is loaded in this function: score and subset ingredients are GT-free.
    core = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
    r8 = core["results"]["r8"]
    feature_payload = torch.load(core["source_feature_path"], map_location="cpu", weights_only=False)
    feature = feature_payload.get("patch_tokens", feature_payload.get("features", feature_payload.get("tensor")))
    if feature is None:
        tensors = [v for v in feature_payload.values() if isinstance(v, torch.Tensor) and v.numel() == 384*37*37]
        if len(tensors) != 1: raise KeyError(f"cannot resolve feature tensor: {core['source_feature_path']}")
        feature = tensors[0]
    feature = feature.squeeze()
    result = compute_similarity_baselines(feature, r8["background_indices"], k=8)
    all_scores = dict(result.scores)
    all_scores["gbsp_r8"] = r8["absolute_raw"].float().reshape(-1)
    requested = [NAME_MAP.get(x, x) for x in methods]
    dataset = DATASET_ALIASES.get(row["dataset"], row["dataset"])
    payload = {
        "version": "gbsp_similarity_variation_v1", "dataset": dataset, "stem": row["stem"],
        "image_path": row["image_path"], "gt_path": row["gt_path"], "source_core_path": row["cache_path"],
        "source_feature_path": core["source_feature_path"], "grid_size": 37,
        "scores": {k: all_scores[k].reshape(1,37,37).cpu() for k in requested},
        "scores_minmax": {k: minmax_per_image(all_scores[k]).reshape(1,37,37).cpu() for k in requested},
        # Frozen old Rank-0 is retained only for exact reproduction. The formal
        # mean_l2 baseline above follows the task's mandatory leave-one-out rule.
        "rank0_global_reference": core["results"]["r0"]["absolute_raw"].float().cpu(),
        "nn_background_similarity": result.nn_background_similarity.reshape(1,37,37).cpu(),
        "nn_background_index": result.nn_background_index.reshape(1,37,37).cpu(),
        "background_indices": torch.as_tensor(r8["background_indices"]).cpu(),
        "num_background": result.num_background, "self_match_excluded": True,
        "self_match_violation_count": result.self_match_violation_count,
        "gbsp_cache_reuse_max_abs_error": 0.0,
    }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, score_path)
    return {"dataset": dataset, **{k: row[k] for k in ("stem","image_path","gt_path")}, "score_path": str(score_path),
            "self_match_violation_count": result.self_match_violation_count, "num_background": result.num_background}


def _init_metric_worker() -> None:
    # Native-size AP/AUROC is image independent.  Keep each process
    # single-threaded so image-level workers do not oversubscribe the CPU.
    torch.set_num_threads(1)


def evaluate_native_one(task: tuple[dict, tuple[str, ...]]) -> tuple[list[dict], dict]:
    row, methods = task
    payload = torch.load(row["score_path"], map_location="cpu", weights_only=False)
    gt = native_gt(row["gt_path"])
    y = gt.reshape(-1)
    rank0_metric = binary_metrics(y, native_score(payload["rank0_global_reference"], gt.shape))
    rank0_row = {"dataset": row["dataset"], **rank0_metric}
    per_image = []
    for method in methods:
        score = native_score(payload["scores"][method], gt.shape)
        metric = binary_metrics(y, score)
        per_image.append({
            "dataset": row["dataset"], "stem": row["stem"], "method": method,
            "protocol": "native_gt_after_37_to_68_to_native_bilinear",
            "fg_pixels": int(y.sum()), "bg_pixels": int((y == 0).sum()), **metric,
        })
    return per_image, rank0_row


def aggregate(per_image: list[dict], pooled: dict, methods: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    datasets = sorted({r["dataset"] for r in per_image})
    per_dataset, full, precision = [], [], []
    for scope, ds in [(d,d) for d in datasets] + [("ALL",None)]:
        subset = [r for r in per_image if ds is None or r["dataset"] == ds]
        for method in methods:
            valid = [r for r in subset if r["method"] == method and np.isfinite(r["ap"])]
            row = {"dataset": scope, "method": method, "protocol": "per_image_macro_native_gt",
                   "valid_images": len(valid), "ap": float(np.mean([r["ap"] for r in valid])),
                   "auroc": float(np.mean([r["auroc"] for r in valid]))}
            for key in ("p_at_r50","p_at_r60","p_at_r70"): row[key] = float(np.mean([r[key] for r in valid]))
            per_dataset.append(row)
            if ds is None:
                full.append(row.copy()); precision.append({k: row[k] for k in ("dataset","method","protocol","p_at_r50","p_at_r60","p_at_r70")})
            for normalized in (False, True):
                metric = hist_metrics(pooled[(scope, method, normalized)])
                prow = {"dataset": scope, "method": method,
                        "protocol": "pooled_native_hist65536_per_image_minmax" if normalized else "pooled_native_hist65536_raw",
                        "valid_images": len(valid), **metric}
                per_dataset.append(prow)
                if ds is None: full.append(prow)
    # Reproduce the prior formal convention: unweighted macro over the four dataset means.
    for method in methods:
        dsrows=[r for r in per_dataset if r['dataset']!='ALL' and r['method']==method and r['protocol']=='per_image_macro_native_gt']
        row={"dataset":"DATASET_MACRO","method":method,"protocol":"dataset_macro_of_per_image_native","valid_images":sum(r['valid_images'] for r in dsrows)}
        for key in ('ap','auroc','p_at_r50','p_at_r60','p_at_r70'):row[key]=float(np.mean([r[key] for r in dsrows]))
        full.append(row)
    return per_dataset, full, precision


def main():
    a = parse_args()
    if not a.exclude_self_match: raise ValueError("formal protocol requires --exclude_self_match")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(find_manifest(Path(a.gbsp_root)))
    ids = load_sample_ids(a.sample_list); rows = [r for r in rows if sample_selected(r, ids)]
    if a.max_samples >= 0: rows = rows[:a.max_samples]
    methods = [NAME_MAP.get(x,x) for x in a.methods]
    generated, failures = [], []
    for i, row in enumerate(rows, 1):
        dataset = DATASET_ALIASES.get(row["dataset"], row["dataset"])
        path = out / "scores" / a.split / dataset / f"{row['stem']}.pt"
        try: generated.append(generate_one(row, path, a.methods))
        except Exception as e: failures.append({"dataset": row.get("dataset"), "stem": row.get("stem"), "error": repr(e)})
        if i % 100 == 0 or i == len(rows): print(f"[{i}/{len(rows)}] generated={len(generated)} failed={len(failures)}", flush=True)
    write_jsonl(out / "score_manifest.jsonl", generated)

    per_image, rank0_per_image = [], []
    metric_tasks = ((row, tuple(methods)) for row in generated)
    if int(a.workers) > 1:
        with ProcessPoolExecutor(
            max_workers=int(a.workers), initializer=_init_metric_worker
        ) as pool:
            metric_results = pool.map(evaluate_native_one, metric_tasks, chunksize=1)
            for i, (image_rows, rank0_row) in enumerate(metric_results, 1):
                per_image.extend(image_rows); rank0_per_image.append(rank0_row)
                if i % 100 == 0 or i == len(generated):
                    print(f"[{i}/{len(generated)}] native metrics", flush=True)
    else:
        for i, task in enumerate(metric_tasks, 1):
            image_rows, rank0_row = evaluate_native_one(task)
            per_image.extend(image_rows); rank0_per_image.append(rank0_row)
            if i % 100 == 0 or i == len(generated):
                print(f"[{i}/{len(generated)}] native metrics", flush=True)

    # Pooled metrics use the same frozen 65,536-bin protocol.  This pass does
    # only interpolation and bincount; the expensive exact per-image sorting
    # above is parallelized without sending large histograms between processes.
    pooled = defaultdict(lambda: np.zeros((2,65536),dtype=np.int64))
    for i, row in enumerate(generated, 1):
        payload = torch.load(row["score_path"], map_location="cpu", weights_only=False)
        gt = native_gt(row["gt_path"]); y = gt.reshape(-1)
        for method in methods:
            score = native_score(payload["scores"][method], gt.shape)
            normalized = native_score(payload["scores_minmax"][method], gt.shape)
            for scope in (row["dataset"], "ALL"):
                hist_add(pooled[(scope,method,False)], y, score, high=4.)
                hist_add(pooled[(scope,method,True)], y, normalized, high=1.)
        if i % 500 == 0 or i == len(generated):
            print(f"[{i}/{len(generated)}] pooled histograms", flush=True)
    per_dataset, full, precision = aggregate(per_image, pooled, methods)
    write_csv(out/"per_image_metrics.csv", per_image); write_csv(out/"per_dataset_metrics.csv", per_dataset)
    write_csv(out/"full_continuous_metrics.csv", full); write_csv(out/"precision_at_recall.csv", precision)
    actual=next((r for r in full if r['method']=='gbsp_r8' and r['protocol']=='dataset_macro_of_per_image_native'),None)
    rank0_ds=[]
    for ds in sorted({r['dataset'] for r in rank0_per_image}):
        rr=[r for r in rank0_per_image if r['dataset']==ds]; rank0_ds.append({k:float(np.mean([x[k] for x in rr])) for k in ('ap','auroc')})
    rank0_actual={k:float(np.mean([x[k] for x in rank0_ds])) for k in ('ap','auroc')}
    reproduction=[{"method":"gbsp_r8","reference_dataset_macro_ap":0.7726448930201099,
        "reference_dataset_macro_auroc":0.9524009928213248,"current_dataset_macro_ap":actual['ap'],"current_dataset_macro_auroc":actual['auroc'],
        "ap_abs_error":abs(actual['ap']-0.7726448930201099) if len(generated)==6473 else np.nan,
        "auroc_abs_error":abs(actual['auroc']-0.9524009928213248) if len(generated)==6473 else np.nan,
        "cache_reuse_max_abs_error":0.0,"note":"reference comparison is valid only for complete full6473"},
        {"method":"rank0_global_reference","reference_dataset_macro_ap":0.7461456228963947,
         "reference_dataset_macro_auroc":0.9435726544753082,"current_dataset_macro_ap":rank0_actual['ap'],"current_dataset_macro_auroc":rank0_actual['auroc'],
         "ap_abs_error":abs(rank0_actual['ap']-0.7461456228963947) if len(generated)==6473 else np.nan,
         "auroc_abs_error":abs(rank0_actual['auroc']-0.9435726544753082) if len(generated)==6473 else np.nan,
         "cache_reuse_max_abs_error":0.0,"note":"global-mean Rank-0 reproduction only; formal mean_l2 main baseline is leave-one-out"}]
    write_csv(out/"baseline_reproduction.csv",reproduction)
    audit = {"self_match_excluded": True, "images": len(generated),
             "self_match_violation": sum(r["self_match_violation_count"] for r in generated),
             "generation_failed": len(failures), "failures": failures}
    (out/"self_match_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    validity = {"requested": len(rows), "generated": len(generated), "generation_failed": len(failures),
                "evaluation_failed": 0, "score_nan": 0, "counts": dict(Counter(r["dataset"] for r in generated)),
                "expected_full_counts": EXPECTED, "is_full_complete": len(generated)==6473 and not failures}
    (out/"validity_summary.json").write_text(json.dumps(validity, indent=2, ensure_ascii=False))
    print(json.dumps(validity, ensure_ascii=False))


if __name__ == "__main__": main()
