#!/usr/bin/env python3
"""Generate GT-free GBSP V5 masks from frozen GBSP and existing patch-graph caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_threshold_v5 import (  # noqa: E402
    BackgroundConditionedMarkerPropagation,
    SeededMultiOtsuHysteresis,
    SeededPersistentComponentGrowth,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v5.py"
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
_SETTINGS: dict | None = None


def _resolve(path: str | Path) -> Path:
    path = Path(path); return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _manifest_path(root: Path) -> Path:
    for path in (root / "manifest_test.jsonl", root / "ablations/M1_pilot200/manifest_test.jsonl"):
        if path.is_file(): return path
    raise FileNotFoundError(f"manifest_test.jsonl missing below {root}")


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path); mapping = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        cache = Path(str(row.get("cache_path", "")))
        if not all(key) or key in mapping or not cache.is_file():
            raise RuntimeError(f"invalid manifest row {path}:{line}: {key}")
        mapping[key] = row
    return rows, mapping


def _sample_keys(path: Path) -> list[tuple[str, str]]:
    keys=[]
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw=raw.strip()
        if not raw or raw.startswith("#"): continue
        parts=raw.replace("/", "\t", 1).split()
        if len(parts) != 2: raise ValueError(f"invalid identity {path}:{line}")
        keys.append((parts[0], parts[1]))
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate sample identities")
    return keys


def _balanced(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0 or limit >= len(rows): return rows
    groups={name:[] for name in EXPECTED}
    for row in rows: groups[str(row["dataset"])].append(row)
    output=[]; index=0
    while len(output)<limit:
        for name in EXPECTED:
            if index<len(groups[name]): output.append(groups[name][index])
            if len(output)==limit: break
        index+=1
    return output


def _map(payload: dict, field: str, path: Path) -> torch.Tensor:
    value=payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape)!=(1,37,37):
        raise ValueError(f"missing {field}: {path}")
    value=value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all(): raise ValueError(f"nonfinite {field}: {path}")
    return value


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload,temporary); os.replace(temporary,path)


def _boundary_indices(grid: int) -> torch.Tensor:
    values=[y*grid+x for y in range(grid) for x in range(grid) if y in (0,grid-1) or x in (0,grid-1)]
    return torch.tensor(values,dtype=torch.long)


def _validate_graph(payload: dict, dataset: str, stem: str, path: Path, field: str):
    if (str(payload.get("dataset")),str(payload.get("stem")))!=(dataset,stem):
        raise RuntimeError(f"graph identity mismatch: {path}")
    indices=payload.get("neigh_idx_37"); valid=payload.get("neigh_valid_37"); weights=payload.get(field)
    if not torch.is_tensor(indices) or tuple(indices.shape)!=(1369,8) or indices.dtype!=torch.long:
        raise ValueError(f"invalid neigh_idx_37: {path}")
    if not torch.is_tensor(valid) or tuple(valid.shape)!=(1369,8) or valid.dtype!=torch.bool:
        raise ValueError(f"invalid neigh_valid_37: {path}")
    if not torch.is_tensor(weights) or tuple(weights.shape)!=(1369,8):
        raise ValueError(f"invalid {field}: {path}")
    weights=weights.detach().cpu().float().contiguous()
    if not torch.isfinite(weights).all() or float(weights.min())<0: raise ValueError(f"invalid graph values: {path}")
    return indices.detach().cpu().long().contiguous(),valid.detach().cpu().bool().contiguous(),weights


def _run_method(name,score,bc,boundary,indices,valid,weights):
    assert _SETTINGS is not None
    if name=="smoh": return SeededMultiOtsuHysteresis().apply(score,bc)
    if name=="spcg":
        return SeededPersistentComponentGrowth(
            _SETTINGS["spcg_min_history"],_SETTINGS["spcg_mad_multiplier"],_SETTINGS["spcg_min_events"]
        ).apply(score,bc)
    if name=="bcmp":
        return BackgroundConditionedMarkerPropagation(
            _SETTINGS["graph_probability_threshold"],_SETTINGS["epsilon"]
        ).apply(score,bc,boundary,indices,valid,weights,_SETTINGS["graph_kind"]=="cost")
    raise ValueError(name)


def _serialize(name,result,bc):
    mask=result.mask.detach().cpu().float().contiguous(); diagnostics=result.diagnostics
    return {"method":name,"method_family":name,"mask_37":mask,
            "maps_37":{key:value.detach().cpu().float().contiguous() for key,value in result.maps.items()},
            "threshold_high":result.threshold_high,"threshold_low":result.threshold_low,
            "foreground_area":float(mask.mean()),
            "bc_final_ratio":float(mask.reshape(-1).index_select(0,bc).mean()),
            "empty_mask":bool(float(mask.mean())==0),"area_over_50pct":bool(float(mask.mean())>.5),
            "numerical_failure":bool(result.numerical_failure),"diagnostics":diagnostics}


def _process(task: dict) -> dict:
    assert _SETTINGS is not None
    started=time.perf_counter(); output=Path(task["output_path"])
    try:
        source_path=Path(task["source_path"]); graph_path=Path(task["graph_path"])
        source=torch_load(source_path,map_location="cpu"); graph=torch_load(graph_path,map_location="cpu")
        identity=(task["dataset"],task["stem"])
        if (str(source.get("dataset")),str(source.get("stem")))!=identity: raise RuntimeError("GBSP identity mismatch")
        score=_map(source,"absolute_minmax",source_path); raw=_map(source,"absolute_raw",source_path)
        bc=source.get("background_indices")
        if not torch.is_tensor(bc) or bc.ndim!=1 or bc.numel()==0: raise ValueError("background_indices missing")
        bc=bc.detach().cpu().long().contiguous(); boundary=_boundary_indices(37)
        indices,valid,weights=_validate_graph(graph,*identity,graph_path,_SETTINGS["graph_weight_field"])
        feature_id=str(source.get("source_feature_path","")); graph_feature=str(graph.get("source_feature_cache_path",""))
        if feature_id and graph_feature and Path(feature_id).resolve()!=Path(graph_feature).resolve():
            raise RuntimeError("GBSP and graph do not share the same frozen feature cache")
        results={}
        for name in _SETTINGS["methods"]:
            try: results[name]=_serialize(name,_run_method(name,score,bc,boundary,indices,valid,weights),bc)
            except Exception as error:
                results[name]={"method":name,"method_family":name,"mask_37":torch.zeros(1,37,37),"maps_37":{},
                    "threshold_high":None,"threshold_low":None,"foreground_area":0.0,"bc_final_ratio":0.0,
                    "empty_mask":True,"area_over_50pct":False,"numerical_failure":True,
                    "diagnostics":{"failure_reason":"uncaught_method_exception","error":repr(error),"traceback":traceback.format_exc()}}
        payload={"version":_SETTINGS["version"],"dataset":identity[0],"stem":identity[1],"image_id":f"{identity[0]}/{identity[1]}",
                 "image_path":task.get("image_path") or source.get("image_path",""),"gt_path":task.get("gt_path") or source.get("gt_path",""),
                 "image_size":tuple(source["original_image_size"]),"patch_grid_size":(37,37),
                 "raw_residual":raw,"minmax_residual":score,"background_indices":bc,"boundary_indices":boundary,
                 "graph_edges":indices,"graph_valid":valid,"graph_weights_or_costs":weights,
                 "graph_kind":_SETTINGS["graph_kind"],"graph_weight_field":_SETTINGS["graph_weight_field"],
                 "source_gbsp_path":str(source_path.resolve()),"source_graph_path":str(graph_path.resolve()),
                 "source_feature_id":feature_id,"results":results,"settings":dict(_SETTINGS),
                 **{f"{key}_for_generation":value for key,value in _SETTINGS["independence"].items()},
                 "runtime":{"total_seconds":time.perf_counter()-started,"worker_peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}}
        _atomic_save(payload,output)
        return {"dataset":identity[0],"stem":identity[1],"cache_path":str(output),
                "numerical_failures":sum(int(row["numerical_failure"]) for row in results.values()),
                "total_seconds":payload["runtime"]["total_seconds"]}
    except Exception as error:
        return {"dataset":task.get("dataset",""),"stem":task.get("stem",""),"error":repr(error),"traceback":traceback.format_exc()}


def _init(settings,threads):
    global _SETTINGS
    _SETTINGS=settings; torch.set_num_threads(int(threads))


def build(args):
    cfg=load_config(_resolve(args.config)); methods=list(args.methods or cfg.GBSP_V5_METHODS)
    if not methods or len(methods)!=len(set(methods)) or set(methods)-set(cfg.GBSP_V5_METHODS): raise ValueError(f"invalid methods: {methods}")
    frozen=None
    if args.frozen_config:
        frozen=json.loads(_resolve(args.frozen_config).read_text(encoding="utf-8"))
        if frozen.get("schema")!="gbsp_threshold_v5_frozen_config": raise ValueError("bad frozen config")
        if set(methods)-set(frozen.get("selected_methods",[])): raise ValueError("method not selected by frozen config")
    settings={"version":cfg.GBSP_V5_VERSION,"methods":methods,"spcg_min_history":cfg.GBSP_V5_SPCG_MIN_HISTORY,
              "spcg_mad_multiplier":cfg.GBSP_V5_SPCG_MAD_MULTIPLIER,"spcg_min_events":cfg.GBSP_V5_SPCG_MIN_EVENTS,
              "graph_kind":cfg.GBSP_V5_GRAPH_KIND,"graph_weight_field":cfg.GBSP_V5_GRAPH_WEIGHT_FIELD,
              "graph_probability_threshold":cfg.GBSP_V5_GRAPH_PROBABILITY_THRESHOLD,"epsilon":cfg.GBSP_V5_EPS,
              "save_diagnostics":bool(args.save_diagnostics),"independence":dict(cfg.GBSP_V5_INDEPENDENCE)}
    settings["fingerprint"]=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    root=_resolve(args.gbsp_root or cfg.GBSP_V5_GBSP_ROOT); graph_root=_resolve(args.graph_root or cfg.GBSP_V5_GRAPH_ROOT)
    out=_resolve(args.out_root or Path(cfg.GBSP_V5_OUTPUT_ROOT)/"diagnostic200")
    if out==MAIN_ROOT or MAIN_ROOT in out.parents: raise ValueError("output must be outside repository")
    rows,mapping=_manifest(_manifest_path(root)); _,graph_mapping=_manifest(_manifest_path(graph_root))
    if args.sample_list:
        keys=_sample_keys(_resolve(args.sample_list)); missing=[key for key in keys if key not in mapping or key not in graph_mapping]
        if missing: raise KeyError(missing[:5])
        rows=[mapping[key] for key in keys]
    if args.max_samples>=0: rows=_balanced(rows,args.max_samples) if not args.sample_list else rows[:args.max_samples]
    if not rows: raise RuntimeError("no samples")
    if len(rows)==200 and frozen is None and set(methods)!=set(cfg.GBSP_V5_METHODS): raise RuntimeError("Diagnostic200 requires all three V5 methods")
    if len(rows)==6473 and (frozen is None or len(methods)!=1): raise RuntimeError("full6473 requires exactly one frozen method")
    if not args.dry_run:
        audit=_resolve(cfg.GBSP_V5_BASELINE_AUDIT)
        if not audit.is_file() or json.loads(audit.read_text()).get("baseline_reproduction_status")!="PASS":
            raise RuntimeError("V5 A0 baseline reproduction has not passed")
    tasks=[]
    for row in rows:
        key=(str(row["dataset"]),str(row["stem"])); graph_row=graph_mapping[key]
        tasks.append({"dataset":key[0],"stem":key[1],"source_path":row["cache_path"],"graph_path":graph_row["cache_path"],
                      "image_path":row.get("image_path",""),"gt_path":row.get("gt_path",""),
                      "output_path":str(out/"test"/key[0]/f"{key[1]}.pt")})
    counts=dict(Counter(task["dataset"] for task in tasks)); out.mkdir(parents=True,exist_ok=True)
    write_json(out/"run_config.json",{"created_at":datetime.now().astimezone().isoformat(timespec="seconds"),"gbsp_root":str(root),
               "graph_root":str(graph_root),"sample_list":str(_resolve(args.sample_list)) if args.sample_list else None,
               "out_root":str(out),"num_selected":len(tasks),"dataset_counts":counts,"frozen_config":args.frozen_config,"settings":settings})
    if args.dry_run:
        print(json.dumps({"status":"dry-run","num_selected":len(tasks),"methods":methods},indent=2)); return
    started=time.perf_counter(); results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init,initargs=(settings,args.torch_threads)) as pool:
        for index,result in enumerate(pool.map(_process,tasks,chunksize=1),1):
            results.append(result)
            if index%20==0 or index==len(tasks): print(f"V5 cache {index}/{len(tasks)}",flush=True)
    failures=[row for row in results if "error" in row]; valid=[row for row in results if "error" not in row]
    valid_keys={(row["dataset"],row["stem"]) for row in valid}
    manifest=[{"dataset":task["dataset"],"stem":task["stem"],"cache_path":task["output_path"],"image_path":task["image_path"],
               "gt_path":task["gt_path"],"source_gbsp_path":task["source_path"],"source_graph_path":task["graph_path"]}
              for task in tasks if (task["dataset"],task["stem"]) in valid_keys]
    write_json(out/"generation_failures.json",failures); write_jsonl(out/"manifest_test.jsonl",manifest)
    summary={"num_requested":len(tasks),"num_valid":len(valid),"num_failed":len(failures),
             "dataset_counts":dict(Counter(row["dataset"] for row in manifest)),
             "numerical_failure_total":sum(row["numerical_failures"] for row in valid),
             "mean_seconds":sum(row["total_seconds"] for row in valid)/max(len(valid),1),"wall_seconds":time.perf_counter()-started,
             **dict(cfg.GBSP_V5_INDEPENDENCE)}
    write_json(out/"performance_summary.json",summary); print(json.dumps(summary,indent=2))
    if failures and args.failure_policy=="strict": raise RuntimeError(f"{len(failures)} generation failures")


def parser():
    value=argparse.ArgumentParser(description=__doc__); value.add_argument("--config",default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp_root"); value.add_argument("--graph_root"); value.add_argument("--sample_list")
    value.add_argument("--split",default="test",choices=("test",)); value.add_argument("--max_samples",type=int,default=-1)
    value.add_argument("--methods",nargs="+"); value.add_argument("--method",action="append",dest="single")
    value.add_argument("--frozen_config"); value.add_argument("--save_diagnostics",action="store_true"); value.add_argument("--out_root")
    value.add_argument("--workers",type=int,default=4); value.add_argument("--torch_threads",type=int,default=1)
    value.add_argument("--failure-policy",choices=("record","strict"),default="record"); value.add_argument("--dry-run",action="store_true")
    return value


if __name__=="__main__":
    args=parser().parse_args()
    if args.single: args.methods=(args.methods or [])+args.single
    build(args)
