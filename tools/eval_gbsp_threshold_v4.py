#!/usr/bin/env python3
"""Formal original-size multi-metric evaluation and Pareto gating for GBSP V4."""

from __future__ import annotations

import argparse
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
from PIL import Image
from skimage.filters import threshold_multiotsu

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from tools.eval_gbsp_threshold import (  # noqa: E402
    DATASETS, EXPECTED, _aggregate, _binary_metrics, _load_gt, _resize,
    _sample_keys, _table, _write_csv, _write_json,
)
from tools.eval_gbsp_threshold_v3 import _rank_metrics  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v4.py"
PRIMARY = ("S_m", "F_beta_w", "E_mean", "MAE")
SECONDARY = ("F_beta_mean", "Precision", "Recall", "Area", "IoU", "Dice")


def _resolve(path: str | Path) -> Path:
    path = Path(path); return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _manifest_path(root: Path) -> Path:
    for path in (root / "manifest_test.jsonl", root / "ablations/M1_pilot200/manifest_test.jsonl"):
        if path.is_file(): return path
    raise FileNotFoundError(f"manifest_test.jsonl missing below {root}")


def _manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path); seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in seen or not Path(str(row.get("cache_path", ""))).is_file():
            raise RuntimeError(f"invalid manifest {path}:{line}: {key}")
        seen.add(key)
    return rows


def _select(rows: list[dict], sample_list: str | None, limit: int) -> list[dict]:
    if sample_list:
        keys = _sample_keys(_resolve(sample_list)); mapping = {(str(r["dataset"]), str(r["stem"])): r for r in rows}
        missing = [key for key in keys if key not in mapping]
        if missing: raise KeyError(missing[:5])
        rows = [mapping[key] for key in keys]
    if limit >= 0: rows = rows[:limit]
    if not rows: raise RuntimeError("no samples")
    return rows


def _map(payload: dict, fields: tuple[str, ...], path: Path) -> torch.Tensor:
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value) and tuple(value.shape) == (1, 37, 37):
            value = value.detach().cpu().float().contiguous()
            if bool(torch.isfinite(value).all()): return value
    raise ValueError(f"missing {fields}: {path}")


def _scalar(value: dict) -> dict:
    return {key: (item.item() if isinstance(item, (np.integer, np.floating)) else item)
            for key, item in value.items()
            if item is None or isinstance(item, (str, bool, int, float, np.integer, np.floating))}


def _multiotsu_mask(score: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
    values = score.double().reshape(-1).numpy()
    low, high = map(float, threshold_multiotsu(values, classes=3, nbins=256))
    return (score > high).float(), (low, high)


def _process(task: dict) -> dict:
    try:
        path = Path(task["cache_path"]); payload = torch_load(path, map_location="cpu")
        if (str(payload.get("dataset")), str(payload.get("stem"))) != (task["dataset"], task["stem"]):
            raise RuntimeError("identity mismatch")
        adaptive = task["adaptive"]
        if adaptive:
            forbidden = ("gt_used_for_generation", "r1_used_for_generation", "fixed_058_used_for_generation",
                         "target_area_used", "fixed_topk_used", "dino_forward_used", "pca_changed", "morphology_used")
            if any(bool(payload.get(key, True)) for key in forbidden): raise RuntimeError("independence violation")
            score = _map(payload, ("minmax_residual",), path); raw = _map(payload, ("raw_residual",), path)
            source_gbsp = torch_load(Path(payload["source_gbsp_path"]), map_location="cpu")
            available = payload.get("results", {}); names = task["methods"] or list(available)
            if set(names) - set(available): raise KeyError(set(names) - set(available))
            results = {name: available[name] for name in names}
        else:
            score = _map(payload, ("absolute_minmax",), path); raw = _map(payload, ("absolute_raw",), path)
            source_gbsp = payload; results = {}
        bc = payload.get("background_indices")
        if not torch.is_tensor(bc): raise ValueError("BC missing")
        bc = bc.detach().cpu().long()
        multi_mask, multi_thresholds = _multiotsu_mask(score)
        references = list(dict.fromkeys(task["references"]))
        patch_masks, continuous = {}, {}
        if "fixed_058" in references:
            patch_masks["fixed_058"] = (score > task["fixed_threshold"]).float(); continuous["fixed_058"] = score
        if "multi_otsu_3" in references:
            patch_masks["multi_otsu_3"] = multi_mask; continuous["multi_otsu_3"] = score
        if "gbsp_fixed_050" in references:
            patch_masks["gbsp_fixed_050"] = (score > 0.5).float(); continuous["gbsp_fixed_050"] = score
        if "r1_fixed_050" in references:
            dabe_path = Path(str(source_gbsp.get("source_dabe_path", "")))
            dabe = torch_load(dabe_path, map_location="cpu"); r1 = _map(dabe, ("residual_pass1_37",), dabe_path)
            patch_masks["r1_fixed_050"] = (r1 > 0.5).float(); continuous["r1_fixed_050"] = r1
        metadata = {}
        for name in references:
            mask = patch_masks[name]
            metadata[name] = {
                "method_family": "reference", "threshold_high": task["fixed_threshold"] if name == "fixed_058" else (
                    multi_thresholds[1] if name == "multi_otsu_3" else 0.5),
                "threshold_low": multi_thresholds[0] if name == "multi_otsu_3" else None,
                "foreground_area": float(mask.mean()), "empty_mask": int(float(mask.mean()) == 0),
                "area_over_50": int(float(mask.mean()) > 0.5), "numerical_failure": 0,
                "bc_seed_ratio": None, "bc_support_ratio": None,
                "bc_final_ratio": float(mask.reshape(-1).index_select(0, bc).mean()) if name != "r1_fixed_050" else None,
                "diagnostics": {},
            }
        for name, item in results.items():
            mask = item.get("mask_37")
            if not torch.is_tensor(mask) or tuple(mask.shape) != (1, 37, 37): raise ValueError(f"bad mask {name}")
            mask = mask.detach().cpu().float()
            patch_masks[name] = mask; continuous[name] = score
            diag = _scalar(item.get("diagnostics", {}))
            metadata[name] = {
                "method_family": item.get("method_family", name), "threshold_high": item.get("threshold_high"),
                "threshold_low": item.get("threshold_low"), "foreground_area": float(item.get("foreground_area", mask.mean())),
                "empty_mask": int(item.get("empty_mask", float(mask.mean()) == 0)),
                "area_over_50": int(item.get("area_over_50pct", float(mask.mean()) > 0.5)),
                "numerical_failure": int(item.get("numerical_failure", True)),
                "bc_seed_ratio": diag.get("bc_seed_ratio"), "bc_support_ratio": diag.get("bc_support_ratio"),
                "bc_final_ratio": item.get("bc_final_ratio"), "diagnostics": diag,
            }
        gt_path = Path(task["gt_path"] or payload.get("gt_path", "")); gt = _load_gt(gt_path); shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt); rows = []
        for name in (*references, *results):
            if name in references:
                threshold = task["fixed_threshold"] if name == "fixed_058" else (0.5 if name != "multi_otsu_3" else None)
                native = (_resize(continuous[name], shape) > threshold).float() if threshold is not None else (_resize(patch_masks[name], shape) > 0.5).float()
            else:
                native = (_resize(patch_masks[name], shape) > 0.5).float()
            info = metadata[name]
            rows.append({
                "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(path),
                "image_path": task["image_path"] or payload.get("image_path", ""), "gt_path": str(gt_path),
                "method": name, "method_family": info["method_family"],
                **_binary_metrics(context, native), **_rank_metrics(_resize(continuous[name], shape), gt),
                **{key: info[key] for key in ("foreground_area", "empty_mask", "area_over_50", "numerical_failure",
                                                "threshold_high", "threshold_low", "bc_seed_ratio", "bc_support_ratio", "bc_final_ratio")},
                **info["diagnostics"],
            })
        fixed = next(row for row in rows if row["method"] == "fixed_058")
        delta_fields = {"S_m": "delta_S_vs_fixed_058", "F_beta_w": "delta_F_beta_w_vs_fixed_058",
                        "E_mean": "delta_E_vs_fixed_058", "MAE": "delta_MAE_vs_fixed_058",
                        "F_beta_mean": "delta_Fmean_vs_fixed_058", "Precision": "delta_Precision_vs_fixed_058",
                        "Recall": "delta_Recall_vs_fixed_058", "Area": "delta_Area_vs_fixed_058"}
        for row in rows:
            for field, delta in delta_fields.items(): row[delta] = row[field] - fixed[field]
        return {"rows": rows}
    except Exception as error:
        return {"dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
                "error": repr(error), "traceback": traceback.format_exc()}


def _init(threads: int) -> None: torch.set_num_threads(threads)


def _mean(values) -> float:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def _fields(rows: list[dict]) -> list[str]:
    output, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen: seen.add(key); output.append(key)
    return output


def _enrich(summary: list[dict], rows: list[dict]) -> None:
    extra = ("numerical_failure", "foreground_area", "threshold_high", "threshold_low", "bc_seed_ratio",
             "bc_support_ratio", "bc_final_ratio", "pixel_AP", "pixel_AUROC")
    for aggregate in summary:
        subset = [row for row in rows if row["method"] == aggregate["method"] and
                  (aggregate["dataset"] == "ALL" or row["dataset"] == aggregate["dataset"])]
        aggregate["method_family"] = subset[0]["method_family"]
        for key in extra: aggregate[key] = _mean(row.get(key) for row in subset)


def _dominates(a: dict, b: dict) -> bool:
    no_worse = a["S_m"] >= b["S_m"] and a["F_beta_w"] >= b["F_beta_w"] and a["E_mean"] >= b["E_mean"] and a["MAE"] <= b["MAE"]
    strict = a["S_m"] > b["S_m"] or a["F_beta_w"] > b["F_beta_w"] or a["E_mean"] > b["E_mean"] or a["MAE"] < b["MAE"]
    return bool(no_worse and strict)


def _pareto(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not any(_dominates(other, row) for other in rows if other is not row)]


def _bootstrap(rows: list[dict], repetitions: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed); methods = sorted({r["method"] for r in rows if r["method"] not in {"fixed_058", "multi_otsu_3", "gbsp_fixed_050", "r1_fixed_050"}})
    lookup = {(r["dataset"], r["stem"], r["method"]): r for r in rows}; output = []
    for method in methods:
        for metric in PRIMARY:
            groups = []
            for dataset in DATASETS:
                keys = [(r["dataset"], r["stem"]) for r in rows if r["dataset"] == dataset and r["method"] == method]
                groups.append(np.asarray([lookup[(*key, method)][metric] - lookup[(*key, "fixed_058")][metric] for key in keys]))
            estimates = np.empty(repetitions)
            for index in range(repetitions): estimates[index] = np.mean([g[rng.integers(0, g.size, g.size)].mean() for g in groups if g.size])
            observed = float(np.mean([g.mean() for g in groups if g.size]))
            output.append({"method": method, "metric": metric, "paired_macro_delta": observed,
                           "ci95_low": float(np.quantile(estimates, .025)), "ci95_high": float(np.quantile(estimates, .975)),
                           "repetitions": repetitions, "seed": seed})
    return output


def _test20_selection(summary: list[dict], rows: list[dict], cfg) -> dict:
    macro = {r["method"]: r for r in summary if r["scope"] == "dataset_macro"}; fixed = macro["fixed_058"]
    evaluations = []
    for method, row in macro.items():
        if method in {"fixed_058", "multi_otsu_3"}: continue
        delta = {key: row[key] - fixed[key] for key in (*PRIMARY, *SECONDARY)}
        margins = cfg.GBSP_V4_TEST20_MARGINS; hard = cfg.GBSP_V4_TEST20_HARD_FAILURE
        primary_checks = {"S_m": delta["S_m"] >= margins["S_m"], "F_beta_w": delta["F_beta_w"] >= margins["F_beta_w"],
                          "E_mean": delta["E_mean"] >= margins["E_mean"], "MAE": delta["MAE"] <= margins["MAE"]}
        hard_reasons = []
        if delta["S_m"] < 0 and delta["F_beta_w"] < 0 and delta["E_mean"] < 0: hard_reasons.append("S_Fw_E_all_decline")
        if delta["MAE"] > hard["MAE_increase"]: hard_reasons.append("MAE_hard_failure")
        if delta["Precision"] < -hard["Precision_drop"]: hard_reasons.append("Precision_hard_failure")
        if row["Area"] > hard["Area_max"] or row["Area"] < hard["Area_min"]: hard_reasons.append("Area_hard_failure")
        images = [item for item in rows if item["method"] == method]
        if any(item["numerical_failure"] for item in images): hard_reasons.append("numerical_failure")
        eligible = sum(primary_checks.values()) >= 3 and delta["Precision"] >= margins["Precision"] and margins["Area_min"] <= row["Area"] <= margins["Area_max"] and not hard_reasons
        evaluations.append({"method": method, "method_family": row["method_family"], "eligible": eligible,
                            "primary_noninferior_count": sum(primary_checks.values()), "primary_checks": primary_checks,
                            "hard_failure_reasons": hard_reasons, "deltas": delta,
                            **{key: row[key] for key in (*PRIMARY, *SECONDARY)}})
    eligible = [row for row in evaluations if row["eligible"]]; front = _pareto(eligible)
    front.sort(key=lambda row: (-sum(row["deltas"][k] >= 0 for k in PRIMARY), -row["Precision"], -row["F_beta_mean"], abs(row["Area"] - fixed["Area"]), {"huto":0,"orc":1,"lrc":2}.get(row["method_family"],3)))
    selected, families = [], set()
    for row in front:
        if row["method_family"] in families: continue
        selected.append(row); families.add(row["method_family"])
        if len(selected) == 2: break
    return _selection_payload("test20", evaluations, front, selected, cfg, "PASS" if selected else "STOP")


def _selection_payload(stage, evaluations, front, selected, cfg, status):
    method = selected[0]["method"] if stage == "pilot200" and selected else ""
    return {
        "stage": stage, "status": status, "primary_metrics": list(PRIMARY), "secondary_metrics": list(SECONDARY),
        "noninferiority_margins": dict(cfg.GBSP_V4_BOOTSTRAP_MARGINS),
        "evaluations": evaluations, "pareto_front": [row["method"] for row in front],
        "selected_method": method, "selected_methods": [row["method"] for row in selected],
        "selection_reason": {
            "S_m": "Pareto/noninferiority joint decision", "F_beta_w": "Pareto/noninferiority joint decision",
            "E_mean": "Pareto/noninferiority joint decision", "MAE": "Pareto/noninferiority joint decision",
            "Precision": "secondary tie-break and hard gate", "Recall": "reported with Fmean/structure safeguards",
            "Area": "range/stability gate", "dataset_stability": "all four datasets checked",
            "bootstrap": "required at Pilot200", "simplicity": "used only after metric ties",
        },
    }


def _pilot_selection(summary: list[dict], rows: list[dict], bootstrap: list[dict], cfg) -> dict:
    macro = {r["method"]: r for r in summary if r["scope"] == "dataset_macro"}; fixed = macro["fixed_058"]
    dataset = {(r["dataset"], r["method"]): r for r in summary if r["scope"] == "dataset"}
    ci = {(r["method"], r["metric"]): r for r in bootstrap}; evaluations = []
    refs = {"fixed_058", "multi_otsu_3"}
    for method, row in macro.items():
        if method in refs: continue
        gate = cfg.GBSP_V4_PILOT_GATE
        metric_checks = {"S_m": row["S_m"] >= gate["S_m"], "F_beta_w": row["F_beta_w"] >= gate["F_beta_w"],
                         "E_mean": row["E_mean"] >= gate["E_mean"], "MAE": row["MAE"] <= gate["MAE"],
                         "F_beta_mean": row["F_beta_mean"] >= gate["F_beta_mean"], "Precision": row["Precision"] >= gate["Precision"],
                         "Area": gate["Area_min"] <= row["Area"] <= gate["Area_max"]}
        stable, dataset_reasons = True, []
        for name in DATASETS:
            candidate, reference = dataset[(name, method)], dataset[(name, "fixed_058")]
            delta = {key: candidate[key] - reference[key] for key in PRIMARY}
            margin = cfg.GBSP_V4_DATASET_MARGINS
            if delta["S_m"] < margin["S_m"] or delta["F_beta_w"] < margin["F_beta_w"] or delta["E_mean"] < margin["E_mean"] or delta["MAE"] > margin["MAE"]:
                stable = False; dataset_reasons.append(name)
        margin = cfg.GBSP_V4_BOOTSTRAP_MARGINS
        bootstrap_checks = {
            "S_m": ci[(method, "S_m")]["ci95_low"] > margin["S_m"],
            "F_beta_w": ci[(method, "F_beta_w")]["ci95_low"] > margin["F_beta_w"],
            "E_mean": ci[(method, "E_mean")]["ci95_low"] > margin["E_mean"],
            "MAE": ci[(method, "MAE")]["ci95_high"] < margin["MAE"],
        }
        passed = all(metric_checks.values()) and stable and all(bootstrap_checks.values()) and not any(r["numerical_failure"] for r in rows if r["method"] == method)
        evaluations.append({"method": method, "method_family": row["method_family"], "eligible": passed,
                            "metric_checks": metric_checks, "dataset_stable": stable, "unstable_datasets": dataset_reasons,
                            "bootstrap_checks": bootstrap_checks,
                            "bootstrap_status": "PASS" if all(bootstrap_checks.values()) else "partial_noninferiority",
                            "deltas": {key: row[key] - fixed[key] for key in (*PRIMARY, *SECONDARY)},
                            **{key: row[key] for key in (*PRIMARY, *SECONDARY)}})
    eligible = [row for row in evaluations if row["eligible"]]; front = _pareto(eligible)
    front.sort(key=lambda row: (-sum(row["bootstrap_checks"].values()), -sum(row["deltas"][k] >= 0 for k in PRIMARY),
                                -row["Precision"], -row["F_beta_mean"], abs(row["Area"] - fixed["Area"])))
    selected = front[:1]
    return _selection_payload("pilot200", evaluations, front, selected, cfg, "PASS" if selected else "STOP")


def _frozen(selection: dict) -> dict:
    return {"schema": "gbsp_threshold_v4_frozen_config", "source_stage": selection["stage"],
            "selected_methods": selection["selected_methods"], "parameters_frozen": True,
            "gt_used": False, "r1_used": False, "fixed_058_used": False, "target_area_used": False}


def _plots(out: Path, stage: str, summary: list[dict], rows: list[dict]) -> list[dict]:
    failures = []
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as error: return [{"plot":"all","error":repr(error)}]
    macro = [r for r in summary if r["scope"] == "dataset_macro"]
    def run(name, fn):
        try: fn(plt, out/name)
        except Exception as error: failures.append({"plot":name,"error":repr(error),"traceback":traceback.format_exc()})
    def primary(plt,path):
        x=np.arange(len(macro)); w=.2; fig,ax=plt.subplots(figsize=(max(9,len(macro)*1.5),4))
        for i,key in enumerate(PRIMARY): ax.bar(x+(i-1.5)*w,[r[key] for r in macro],w,label=key)
        ax.set_xticks(x,[r["method"] for r in macro],rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def pra(plt,path):
        x=np.arange(len(macro)); w=.25; fig,ax=plt.subplots(figsize=(max(9,len(macro)*1.5),4))
        for off,key in ((-w,"Precision"),(0,"Recall"),(w,"Area")): ax.bar(x+off,[r[key] for r in macro],w,label=key)
        ax.set_xticks(x,[r["method"] for r in macro],rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def delta(field,label):
        def draw(plt,path):
            candidates=[r["method"] for r in macro if r["method"] not in {"fixed_058","multi_otsu_3"}]
            pairs=[]
            for method in candidates:
                values=[float(r[field]) for r in rows if r["method"]==method and r.get(field) is not None and np.isfinite(float(r[field]))]
                if values: pairs.append((method,values))
            methods=[item[0] for item in pairs]; data=[item[1] for item in pairs]
            fig,ax=plt.subplots(figsize=(max(7,len(methods)*1.4),4))
            if data: ax.boxplot(data,tick_labels=methods,showfliers=False)
            ax.axhline(0,color="black"); ax.set_ylabel(label); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
        return draw
    def pareto(plt,path):
        fig,ax=plt.subplots(figsize=(6,5))
        for r in macro:
            ax.scatter(r["F_beta_w"],r["S_m"],s=50); ax.annotate(r["method"],(r["F_beta_w"],r["S_m"]),fontsize=7)
        ax.set(xlabel="F_beta_w",ylabel="S_m"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def areas(plt,path):
        methods=[r["method"] for r in macro if r["method"] not in {"fixed_058","multi_otsu_3"}]
        fig,ax=plt.subplots(figsize=(max(7,len(methods)*1.4),4)); x=np.arange(len(methods)); w=.25
        for off,key in ((-w,"seed_area"),(0,"support_area"),(w,"foreground_area")):
            ax.bar(x+off,[_mean(r.get(key) for r in rows if r["method"]==m) for m in methods],w,label=key)
        ax.set_xticks(x,methods,rotation=25,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)
    def method_rows(method: str, limit: int = 5):
        candidates=[r for r in rows if r["method"]==method and not r.get("numerical_failure")]
        candidates.sort(key=lambda r:r["delta_F_beta_w_vs_fixed_058"],reverse=True)
        if len(candidates)<=limit:return candidates
        indices=np.linspace(0,len(candidates)-1,limit,dtype=int)
        return [candidates[i] for i in indices]
    def common_arrays(item, method):
        payload=torch_load(item["cache_path"],map_location="cpu"); result=payload["results"][method]
        with Image.open(item["image_path"]) as image: rgb=np.asarray(image.convert("RGB"))
        with Image.open(item["gt_path"]) as image: gt=np.asarray(image.convert("L"))
        score=payload["minmax_residual"][0].numpy(); bc=np.zeros(37*37,np.float32); bc[payload["background_indices"].numpy()]=1
        multi_threshold=float(threshold_multiotsu(score.reshape(-1),classes=3,nbins=256)[1])
        fixed=score>0.58; multi=score>multi_threshold; final=result["mask_37"][0].numpy()>0.5
        return payload,result,rgb,gt,score,bc.reshape(37,37),fixed,multi,final
    def huto_visual(plt,path):
        selected=method_rows("huto_mid_h")
        if not selected: raise RuntimeError("no HUTO rows")
        fig,axes=plt.subplots(len(selected),10,figsize=(28,3*len(selected)),squeeze=False)
        for axrow,item in zip(axes,selected):
            payload,result,rgb,gt,score,bc,fixed,multi,final=common_arrays(item,"huto_mid_h")
            maps=result["maps_37"]; seed=maps["mask_seed"][0].numpy(); support=maps["mask_support"][0].numpy()
            panels=((rgb,"RGB"),(gt,"GT"),(bc,"Full BC"),(score,"GBSP residual"),(fixed,"fixed-0.58"),
                    (multi,"Multi-Otsu-3"),(seed,"HUTO seed"),(support,"HUTO support"),(final,"HUTO final"),(final.astype(int)-multi.astype(int),"final - multi"))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "coolwarm"); ax.set_title(title); ax.axis("off")
            axrow[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndS={item['delta_S_vs_fixed_058']:+.3f} dE={item['delta_E_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
    def orc_visual(plt,path):
        selected=method_rows("orc_h")
        if not selected: raise RuntimeError("no ORC rows")
        fig,axes=plt.subplots(len(selected),8,figsize=(23,3*len(selected)),squeeze=False)
        for axrow,item in zip(axes,selected):
            _,result,rgb,gt,score,_,_,_,final=common_arrays(item,"orc_h"); maps=result["maps_37"]
            panels=((rgb,"RGB"),(maps["residual_norm_map"][0].numpy(),"orthogonal norm"),
                    (maps["residual_coherence_map"][0].numpy(),"direction coherence"),(maps["residual_rank_map"][0].numpy(),"residual rank"),
                    (maps["coherence_rank_map"][0].numpy(),"coherence rank"),(maps["orc_score_map"][0].numpy(),"ORC score"),
                    (maps["mask_seed"][0].numpy(),"ORC seed"),(final,"ORC final"))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "magma"); ax.set_title(title); ax.axis("off")
            axrow[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndFw={item['delta_F_beta_w_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
    def lrc_visual(plt,path):
        selected=method_rows("lrc_h")
        if not selected: raise RuntimeError("no LRC rows")
        fig,axes=plt.subplots(len(selected),8,figsize=(23,3*len(selected)),squeeze=False)
        for axrow,item in zip(axes,selected):
            _,result,rgb,gt,score,_,_,_,final=common_arrays(item,"lrc_h"); maps=result["maps_37"]
            panels=((rgb,"RGB"),(score,"residual"),(maps["median3_map"][0].numpy(),"median 3x3"),
                    (maps["median7_map"][0].numpy(),"median 7x7"),(maps["local_contrast_map"][0].numpy(),"local contrast"),
                    (maps["lrc_score_map"][0].numpy(),"LRC score"),(maps["mask_seed"][0].numpy(),"LRC seed"),(final,"LRC final"))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "magma"); ax.set_title(title); ax.axis("off")
            axrow[0].set_ylabel(f"{item['dataset']}/{item['stem']}\ndMAE={item['delta_MAE_vs_fixed_058']:+.3f}")
        fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
    def multi_metric_visual(plt,path,best:bool):
        adaptive=[r for r in rows if r["method"] not in {"fixed_058","multi_otsu_3"}]
        specs=(("delta_S_vs_fixed_058",best),("delta_F_beta_w_vs_fixed_058",best),("delta_E_vs_fixed_058",best),("delta_MAE_vs_fixed_058",not best))
        selected=[]; seen=set()
        for field,reverse in specs:
            for item in sorted(adaptive,key=lambda r:r[field],reverse=reverse):
                key=(item["dataset"],item["stem"],item["method"])
                if key not in seen: selected.append((field,item)); seen.add(key); break
        fig,axes=plt.subplots(len(selected),6,figsize=(18,3*len(selected)),squeeze=False)
        for axrow,(criterion,item) in zip(axes,selected):
            _,result,rgb,gt,score,_,fixed,multi,final=common_arrays(item,item["method"])
            panels=((rgb,"RGB"),(gt,"GT"),(score,"residual"),(fixed,"fixed-0.58"),(multi,"Multi-Otsu-3"),(final,item["method"]))
            for ax,(array,title) in zip(axrow,panels): ax.imshow(array,cmap=None if array.ndim==3 else "gray"); ax.set_title(title); ax.axis("off")
            axrow[0].set_ylabel(f"{criterion}\n{item['dataset']}/{item['stem']}")
        fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)
    def dataset_all(plt,path):
        methods=[r["method"] for r in macro]; metrics=("S_m","F_beta_w","F_beta_mean","E_mean","MAE","Precision")
        fig,axes=plt.subplots(2,3,figsize=(18,9),squeeze=False)
        dataset_rows={(r["dataset"],r["method"]):r for r in summary if r["scope"]=="dataset"}
        for ax,metric in zip(axes.reshape(-1),metrics):
            x=np.arange(len(methods)); width=.18
            for i,dataset in enumerate(DATASETS): ax.bar(x+(i-1.5)*width,[dataset_rows[(dataset,m)][metric] for m in methods],width,label=dataset)
            ax.set_xticks(x,methods,rotation=35,ha="right"); ax.set_title(metric)
        axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)
    name=("diagnostic200_primary_metric_comparison.png" if stage=="diagnostic200" else
          ("pilot200_primary_metric_comparison.png" if stage=="pilot200" else "test20_primary_metric_comparison.png"))
    run(name,primary); run("precision_recall_area_comparison.png",pra); run("primary_metric_pareto.png",pareto)
    run("per_image_delta_S.png",delta("delta_S_vs_fixed_058","delta S")); run("per_image_delta_Fw.png",delta("delta_F_beta_w_vs_fixed_058","delta Fw"))
    run("per_image_delta_E.png",delta("delta_E_vs_fixed_058","delta E")); run("per_image_delta_MAE.png",delta("delta_MAE_vs_fixed_058","delta MAE"))
    run("seed_support_area_distribution.png",areas); run("threshold_distribution.png",delta("threshold_high","threshold high"))
    run("huto_examples.png",huto_visual); run("orc_examples.png",orc_visual); run("lrc_examples.png",lrc_visual)
    run("success_visualizations.png",lambda plt,path:multi_metric_visual(plt,path,True))
    run("failure_visualizations.png",lambda plt,path:multi_metric_visual(plt,path,False))
    run("dataset_wise_all_metrics.png",dataset_all)
    return failures


def evaluate(args: argparse.Namespace) -> None:
    if bool(args.gbsp_root) == bool(args.threshold_root): raise ValueError("provide exactly one root")
    cfg=load_config(_resolve(args.config)); adaptive=bool(args.threshold_root); root=_resolve(args.threshold_root or args.gbsp_root)
    source_rows=_select(_manifest(_manifest_path(root)),args.sample_list,args.max_samples); out=_resolve(args.out_dir)
    if out==MAIN_ROOT or MAIN_ROOT in out.parents: raise ValueError("output inside repository")
    out.mkdir(parents=True,exist_ok=True)
    if adaptive:
        first=torch_load(Path(source_rows[0]["cache_path"]),map_location="cpu"); methods=list(args.methods or first["results"])
        references=list(dict.fromkeys(("fixed_058","multi_otsu_3",*args.compare)))
    else:
        methods=[]; references=list(args.methods or ("fixed_058","multi_otsu_3"))
    tasks=[{"dataset":str(r["dataset"]),"stem":str(r["stem"]),"cache_path":r["cache_path"],"image_path":r.get("image_path",""),"gt_path":r.get("gt_path",""),
           "adaptive":adaptive,"methods":methods,"references":references,"fixed_threshold":cfg.GBSP_V4_FIXED_THRESHOLD_REFERENCE_ONLY} for r in source_rows]
    started=time.perf_counter(); results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init,initargs=(args.torch_threads,)) as pool:
        for i,result in enumerate(pool.map(_process,tasks,chunksize=1),1):
            results.append(result)
            if i%20==0 or i==len(tasks): print(f"V4 eval {i}/{len(tasks)}",flush=True)
    failures=[r for r in results if "error" in r]; valid=[r for r in results if "error" not in r]
    _write_json(out/"evaluation_failures.json",failures)
    if not valid: raise RuntimeError("no valid evaluations")
    rows=[row for result in valid for row in result["rows"]]; summary=_aggregate(rows); _enrich(summary,rows)
    count=len(tasks); counts=dict(Counter(r["dataset"] for r in tasks))
    stage="baseline_test20" if not adaptive and count==20 else ("test20" if count==20 else (("diagnostic200" if args.diagnostic200 else "pilot200") if count==200 else ("full6473" if count==sum(EXPECTED.values()) and counts==EXPECTED else "custom")))
    macro={r["method"]:r for r in summary if r["scope"]=="dataset_macro"}
    errors={name:{key:abs(macro[name][key]-value) for key,value in target.items()} for name,target in cfg.GBSP_V4_BASELINE_TARGETS.items() if name in macro}
    reproduction=bool(stage=="baseline_test20" and set(errors)==set(cfg.GBSP_V4_BASELINE_TARGETS) and max(max(v.values()) for v in errors.values())<cfg.GBSP_V4_BASELINE_TOLERANCE)
    bootstrap=_bootstrap(rows,args.bootstrap_repetitions,args.bootstrap_seed) if adaptive else []
    selection=_test20_selection(summary,rows,cfg) if stage=="test20" else (_pilot_selection(summary,rows,bootstrap,cfg) if stage in {"pilot200","diagnostic200"} else None)
    if selection and stage == "diagnostic200":
        selection["stage"] = "diagnostic200"
        selection["diagnostic_only"] = True
    primary_deltas=[{k:r.get(k) for k in ("dataset","stem","method","delta_S_vs_fixed_058","delta_F_beta_w_vs_fixed_058","delta_E_vs_fixed_058","delta_MAE_vs_fixed_058")} for r in rows]
    secondary_deltas=[{k:r.get(k) for k in ("dataset","stem","method","delta_Fmean_vs_fixed_058","delta_Precision_vs_fixed_058","delta_Recall_vs_fixed_058","delta_Area_vs_fixed_058")} for r in rows]
    threshold=[{k:r.get(k) for k in ("dataset","stem","method","threshold_low","threshold_high")} for r in rows]
    seed_support=[{k:r.get(k) for k in ("dataset","stem","method","seed_area","support_area","foreground_area","bc_seed_ratio","bc_support_ratio","bc_final_ratio")} for r in rows]
    orc=[{k:r.get(k) for k in r if k in {"dataset","stem","method","orc_t1","orc_t2","orc_seed_area","support_threshold","support_area","final_area","bc_seed_ratio","bc_final_ratio","raw_residual_max_abs_error","basis_orthonormal_max_abs_error"}} for r in rows if r["method"]=="orc_h"]
    lrc=[{k:r.get(k) for k in r if k in {"dataset","stem","method","lrc_t1","lrc_t2","lrc_seed_area","support_threshold","support_area","final_area","bc_seed_ratio","bc_final_ratio"}} for r in rows if r["method"]=="lrc_h"]
    _write_csv(out/"per_image_metrics.csv",rows,_fields(rows)); _write_csv(out/"per_dataset_metrics.csv",summary,_fields(summary))
    _write_csv(out/"primary_metric_deltas.csv",primary_deltas,_fields(primary_deltas)); _write_csv(out/"secondary_metric_deltas.csv",secondary_deltas,_fields(secondary_deltas))
    _write_csv(out/"bootstrap_ci95.csv",bootstrap,_fields(bootstrap)); _write_csv(out/"threshold_distribution.csv",threshold,_fields(threshold))
    _write_csv(out/"seed_support_distribution.csv",seed_support,_fields(seed_support)); _write_csv(out/"orc_diagnostics.csv",orc,_fields(orc)); _write_csv(out/"lrc_diagnostics.csv",lrc,_fields(lrc))
    audit={"schema":"gbsp_threshold_v4_eval","stage":stage,"num_requested":count,"num_valid":len(valid),"num_failed":len(failures),"dataset_counts":counts,
           "baseline_reproduction_errors":errors,"baseline_reproduction_status":"PASS" if reproduction else ("NOT_APPLICABLE" if adaptive else "FAIL"),
           "gt_used_for_calibration":False,"fixed_058_used_in_adaptive_formula":False,"wall_seconds":time.perf_counter()-started}
    _write_json(out/"numerical_failure_summary.json",audit)
    filename={"baseline_test20":"baseline_test20.csv","test20":"test20_summary.csv","pilot200":"pilot200_summary.csv","diagnostic200":"diagnostic200_summary.csv","full6473":"full6473_summary.csv"}.get(stage,"summary.csv")
    _write_csv(out/filename,list(macro.values()),_fields(list(macro.values())))
    if selection:
        _write_json(out/"method_selection.json",selection); _write_json(out/"pareto_front.json",selection["pareto_front"])
        _write_json(out/("final_config.json" if stage=="pilot200" else "selected_config.json"),_frozen(selection))
    _write_csv(out/"downstream_1x1_results.csv",[{"method":m,"status":"not_run_full_gate_required"} for m in (selection or {}).get("selected_methods",[])])
    plot_failures=_plots(out,stage,summary,rows); _write_json(out/"visualization_failures.json",plot_failures)
    report=["# GBSP Threshold V4 Report","",f"- Stage: {stage}",f"- Selection: {(selection or {}).get('status','N/A')}","",_table(list(macro.values()),("method",*PRIMARY,*SECONDARY))]
    (out/"GBSP_THRESHOLD_V4_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({**audit,"selection":selection},ensure_ascii=False,indent=2))
    if failures and args.failure_policy=="strict": raise RuntimeError(f"{len(failures)} failures")


def parser():
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--config",default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp_root"); value.add_argument("--threshold_root"); value.add_argument("--sample_list")
    value.add_argument("--methods",nargs="+"); value.add_argument("--compare",nargs="+",default=())
    value.add_argument("--max_samples",type=int,default=-1); value.add_argument("--out_dir",required=True)
    value.add_argument("--workers",type=int,default=4); value.add_argument("--torch_threads",type=int,default=1)
    value.add_argument("--bootstrap_repetitions",type=int,default=2000); value.add_argument("--bootstrap_seed",type=int,default=20260806)
    value.add_argument("--diagnostic200",action="store_true")
    value.add_argument("--failure-policy",choices=("record","strict"),default="record"); return value


if __name__=="__main__": evaluate(parser().parse_args())
