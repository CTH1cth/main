#!/usr/bin/env python3
"""Build only the frozen identity Full-BC fields required by GBSP.

This intentionally stops after DABE-v2 background connectivity and anchor
selection. It does not compute R1/R, foreground seeds, propagation, evidence,
or p_dabe.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg
from common.dabe_pseudo import (
    DABE_V2_DEFAULT_PARAMS,
    _background_anchor,
    _background_connectivity,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
)
from common.utils import build_image_items, load_config, read_jsonl, torch_load, write_jsonl


VERSION = "gbsp_fullbc_identity_v1"
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
_PARAMS: dict | None = None


def _init_worker(params: dict, torch_threads: int) -> None:
    global _PARAMS
    _PARAMS = params
    torch.set_num_threads(max(1, int(torch_threads)))


def _valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        p = torch_load(path, map_location="cpu")
        return (
            p.get("version") == VERSION
            and p.get("num_views") == 1
            and p.get("augs") == ["identity"]
            and tuple(p["bc_map_37"].shape) == (1, 37, 37)
            and tuple(p["bg_anchor_37"].shape) == (1, 37, 37)
            and bool(torch.isfinite(p["bc_map_37"]).all())
        )
    except (OSError, RuntimeError, TypeError, KeyError):
        return False


def _one(task: dict) -> dict:
    started = time.perf_counter()
    output = Path(task["output_path"])
    try:
        if _valid(output):
            return {**task, "skipped": True, "runtime_seconds": 0.0}
        assert _PARAMS is not None
        grid = int(_PARAMS["GRID"])
        payload = torch_load(task["feature_path"], map_location="cpu")
        if (payload.get("dataset"), payload.get("stem")) != (task["dataset"], task["stem"]):
            raise RuntimeError("feature identity mismatch")
        feature = payload.get("tensor")
        if not torch.is_tensor(feature) or tuple(feature.shape) != (384, grid, grid):
            raise ValueError(f"invalid feature shape: {getattr(feature, 'shape', None)}")
        feature = feature.detach().cpu().float().contiguous()
        rgb = _load_rgb_grid(task["image_path"], grid)
        feat_n = F.normalize(feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2)
        rgb_n = rgb.permute(1, 2, 0).reshape(grid * grid, 3).float()
        edge_n = _sobel_magnitude(rgb).reshape(-1).float()
        neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge_n, grid, _PARAMS)
        bc, border = _background_connectivity(neigh_idx, neigh_weight, grid, _PARAMS)
        anchor = _background_anchor(bc, border, _PARAMS)
        result = {
            "version": VERSION, "dataset": task["dataset"], "stem": task["stem"],
            "image_path": task["image_path"], "source_feature_path": task["feature_path"],
            "num_views": 1, "augs": ["identity"], "grid_size": grid,
            "bc_map_37": bc.reshape(1, grid, grid).float().contiguous(),
            "bg_anchor_37": anchor.reshape(1, grid, grid).float().contiguous(),
            "border_37": border.reshape(1, grid, grid).float().contiguous(),
            "full_bc_count": int(anchor.sum()),
            "computed_stages": ["local_graph", "background_connectivity", "background_anchor"],
            "omitted_stages": ["R1", "R", "foreground_seed", "propagation", "evidence_gate", "p_dabe"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        torch.save(result, temporary); os.replace(temporary, output)
        return {**task, "skipped": False, "runtime_seconds": time.perf_counter()-started,
                "full_bc_count": int(anchor.sum())}
    except Exception as exc:
        return {**task, "error": repr(exc), "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter()-started}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--out_root", required=True)
    p.add_argument("--split", default="test", choices=("test",)); p.add_argument("--max_samples",type=int,default=-1)
    p.add_argument("--workers",type=int,default=4); p.add_argument("--torch_threads",type=int,default=1)
    a=p.parse_args(); cfg=load_config(a.config); out=Path(a.out_root).expanduser().resolve()
    feature_manifest=Path(cfg.CACHE_ROOT)/"features_cache"/cfg.BACKBONE_KEY/"manifest_test.jsonl"
    feature_rows=read_jsonl(feature_manifest); feature_map={(r["dataset"],r["stem"]):r for r in feature_rows}
    items=build_image_items(cfg.DATA_ROOT,cfg.TEST_DATASETS,require_gt=True)
    if a.max_samples>=0: items=items[:a.max_samples]
    if a.max_samples<0:
        counts=Counter(x["dataset"] for x in items)
        if len(items)!=6473 or dict(counts)!=EXPECTED: raise RuntimeError(f"count mismatch: {dict(counts)}")
    params={**DABE_V2_DEFAULT_PARAMS,**_params_from_cfg(cfg),"VERSION":"v2"}
    tasks=[]
    for item in items:
        key=(item["dataset"],item["stem"])
        if key not in feature_map: raise KeyError(f"feature missing: {key}")
        tasks.append({"dataset":key[0],"stem":key[1],"image_path":item["image_path"],"gt_path":item["gt_path"],
          "feature_path":feature_map[key]["cache_path"],"output_path":str(out/key[0]/f"{key[1]}.pt")})
    started=time.perf_counter(); results=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=_init_worker,initargs=(params,a.torch_threads)) as pool:
        for i,r in enumerate(pool.map(_one,tasks,chunksize=1),1):
            results.append(r)
            if i%100==0 or i==len(tasks): print(f"Full-BC-only {i}/{len(tasks)}",flush=True)
    failures=[r for r in results if "error" in r]; valid={(r["dataset"],r["stem"]):r for r in results if "error" not in r}
    with (out/"generation_failures.json").open("w") as f: json.dump(failures,f,ensure_ascii=False,indent=2)
    manifest=[]
    for t in tasks:
        if (t["dataset"],t["stem"]) in valid:
            manifest.append({"dataset":t["dataset"],"stem":t["stem"],"cache_path":t["output_path"],
              "image_path":t["image_path"],"gt_path":t["gt_path"],"source_feature_path":t["feature_path"]})
    write_jsonl(out/"manifest_test.jsonl",manifest)
    summary={"version":VERSION,"requested":len(tasks),"valid":len(valid),"failed":len(failures),
      "resumed":sum(bool(r.get("skipped")) for r in valid.values()),"wall_seconds":time.perf_counter()-started,
      "training_used":False,"dino_extraction_used":False,"full_dabe_v2_generated":False,
      "only_fields":["bc_map_37","bg_anchor_37"]}
    with (out/"generation_summary.json").open("w") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if failures: print(f"WARNING: {len(failures)} rare failures recorded; successful samples preserved")


if __name__=="__main__": main()
