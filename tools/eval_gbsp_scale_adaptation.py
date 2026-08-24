#!/usr/bin/env python3
"""Evaluate registered GBSP scale variants against 296, 512-A and 512-C."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load, write_json  # noqa: E402
from models.gbsp_resolution import hard_mask_at_native  # noqa: E402
from tools.eval_gbsp_resolution_probe import _boundary_f1, _load_gt, _quality, _write_csv  # noqa: E402


DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
METRICS = ("pixel_AP", "pixel_AUROC", "boundary_F1", "S_m", "F_beta_w", "F_beta_mean",
           "E_mean", "MAE", "Precision", "Recall", "Area", "IoU")


def _safe_nanmean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if bool(finite.any()) else float("nan")


def _candidate_quality(indices: torch.Tensor, gt: torch.Tensor, grid: int) -> dict[str, float]:
    occupancy = F.interpolate(gt.unsqueeze(0), size=(grid, grid), mode="area").reshape(-1).index_select(0, indices.long())
    contamination = float(occupancy.mean())
    return {"candidate_precision": 1.0-contamination,
            "candidate_strict_bg_patch_precision": float((occupancy <= .2).float().mean()),
            "fg_contamination_ratio": contamination}


def _source(method: str, sample: dict, core_root: Path, existing512: Path, variants: dict[str, Path]) -> tuple[dict, torch.Tensor, int | None]:
    dataset, stem = str(sample["dataset"]), str(sample["stem"])
    if method == "296-GBSP-r8":
        payload = torch_load(core_root / "test" / dataset / f"{stem}.pt", map_location="cpu")
        result = payload["results"]["r8"]
        energy = result["singular_values"].double().square(); cumulative = torch.cumsum(energy,0)/energy.sum()
        required = int(torch.searchsorted(cumulative,torch.tensor(.9,dtype=cumulative.dtype)).item())+1
        return {"absolute_minmax": result["absolute_minmax"], "background_indices": result["background_indices"],
                "candidate_count": int(result["background_indices"].numel()), "required_rank_uncapped": required,
                "energy_at_r8": float(cumulative[7]), "energy_at_r12": float(cumulative[11]),
                "background_modeling_grid": 37}, result["absolute_minmax"].float(), 68
    if method in {"512-A", "512-C"}:
        folder = "A_bw2_r8" if method == "512-A" else "C_bw2_r12"
        payload = torch_load(existing512 / folder / "scores" / dataset / f"{stem}.pt", map_location="cpu")
        if method == "512-A":
            payload = dict(payload); payload["energy_at_r8"] = payload["energy_at_cap"]
        else:
            # Candidate set is identical; read A only for energy@8.
            a = torch_load(existing512 / "A_bw2_r8" / "scores" / dataset / f"{stem}.pt", map_location="cpu")
            payload = dict(payload); payload["energy_at_r8"] = a["energy_at_cap"]
        payload["energy_at_r12"] = payload["energy_at_cap"] if method == "512-C" else float("nan")
        payload["background_modeling_grid"] = 64
        return payload, payload["absolute_minmax"].float(), None
    payload = torch_load(variants[method] / "scores" / dataset / f"{stem}.pt", map_location="cpu")
    return payload, payload["absolute_minmax"].float(), None


def _aggregate(rows: list[dict], methods: list[str]) -> list[dict]:
    output = []
    for method in methods:
        dataset_rows = []
        for dataset in DATASETS:
            selected = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
            summary = {"scope": "dataset", "dataset": dataset, "method": method, "num_images": len(selected)}
            for field in METRICS:
                summary[field] = _safe_nanmean(row[field] for row in selected)
            dataset_rows.append(summary); output.append(summary)
        macro = {"scope": "dataset_macro", "dataset": "ALL", "method": method,
                 "num_images": sum(row["num_images"] for row in dataset_rows)}
        for field in METRICS:
            macro[field] = _safe_nanmean(row[field] for row in dataset_rows)
        output.append(macro)
        selected = [row for row in rows if row["method"] == method]
        image = {"scope": "image_macro", "dataset": "ALL", "method": method, "num_images": len(selected)}
        for field in METRICS:
            image[field] = _safe_nanmean(row[field] for row in selected)
        output.append(image)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--core-296", required=True)
    parser.add_argument("--existing-512", required=True)
    parser.add_argument("--variant", action="append", default=[], help="METHOD=ROOT")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=.58)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()
    started = time.perf_counter()
    variants = {}
    for item in args.variant:
        name, root = item.split("=",1); variants[name] = Path(root).resolve()
    methods = ["296-GBSP-r8", "512-A", "512-C", *variants]
    samples = read_jsonl(Path(args.sample_list).resolve())
    if int(args.max_samples) >= 0:
        samples = samples[:int(args.max_samples)]
    core_root, existing512 = Path(args.core_296).resolve(), Path(args.existing_512).resolve()
    rows, candidates, failures = [], [], []
    for index, sample in enumerate(samples,1):
        dataset, stem = str(sample["dataset"]), str(sample["stem"])
        try:
            gt = _load_gt(str(sample["gt_path"]))
            original_hw = tuple(map(int, gt.shape[-2:]))
            for method in methods:
                payload, score, legacy = _source(method,sample,core_root,existing512,variants)
                continuous = F.interpolate(score.unsqueeze(0), size=original_hw, mode="bilinear", align_corners=False).squeeze(0)
                _, hard = hard_mask_at_native(score,float(args.threshold),original_hw,legacy_intermediate_size=legacy)
                quality = _quality(hard,continuous,gt)
                rows.append({"dataset":dataset,"stem":stem,"method":method,
                    "gt_area":float(gt.mean()),**quality})
                grid = int(payload.get("background_modeling_grid",64))
                cq = _candidate_quality(payload["background_indices"],gt,grid)
                candidates.append({"dataset":dataset,"stem":stem,"method":method,
                    "candidate_count":int(payload["candidate_count"]),
                    "required_rank_90":int(payload["required_rank_uncapped"]),
                    "energy_at_r8":float(payload["energy_at_r8"]),
                    "energy_at_r12":float(payload.get("energy_at_r12",float("nan"))),**cq})
        except Exception as exc:
            failures.append({"dataset":dataset,"stem":stem,"error":repr(exc)})
            raise
        if index%20==0 or index==len(samples): print(f"[{index}/{len(samples)}] scale evaluation",flush=True)
    summaries = _aggregate(rows,methods)
    candidate_summary=[]
    for method in methods:
        selected=[row for row in candidates if row["method"]==method]
        candidate_summary.append({"method":method,"num_images":len(selected),
            **{field:_safe_nanmean(row[field] for row in selected) for field in
               ("candidate_count","required_rank_90","energy_at_r8","energy_at_r12","candidate_precision",
                "candidate_strict_bg_patch_precision","fg_contamination_ratio")}})
    # Bottom-quartile object area is a frozen sample-relative small-object diagnostic.
    area_q25=float(np.quantile([row["gt_area"] for row in rows if row["method"]==methods[0]],.25))
    small=[]
    for method in methods:
        selected=[row for row in rows if row["method"]==method and row["gt_area"]<=area_q25]
        small.append({"method":method,"num_images":len(selected),"gt_area_q25":area_q25,
            **{field:_safe_nanmean(row[field] for row in selected) for field in ("pixel_AP","pixel_AUROC","boundary_F1")}})
    out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/"per_image_metrics.csv",rows);_write_csv(out/"metrics_summary.csv",summaries)
    _write_csv(out/"candidate_per_image.csv",candidates);_write_csv(out/"candidate_summary.csv",candidate_summary)
    _write_csv(out/"small_object_metrics.csv",small);_write_csv(out/"evaluation_failures.csv",failures)
    metadata={"methods":methods,"sample_count":len(samples),"threshold":float(args.threshold),
        "runtime_seconds":time.perf_counter()-started,"failures":len(failures),
        "continuous_protocol":"native score bilinear to original GT; per-image AP/AUROC",
        "hard_protocol":"threshold at native grid; 296 uses frozen 68 intermediate; nearest to GT",
        "candidate_precision_definition":"1 - mean foreground area occupancy over selected background patches",
        "small_object_definition":f"sample500 GT-area bottom quartile <= {area_q25:.8f}"}
    write_json(out/"evaluation_metadata.json",metadata)
    macro={row["method"]:row for row in summaries if row["scope"]=="dataset_macro"}
    lines=["# GBSP 512 Scale Adaptation — sample500", "", "| Method | AP | AUROC | Boundary F1 | Fw | MAE |",
           "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        r=macro[method];lines.append(f"| {method} | {r['pixel_AP']:.6f} | {r['pixel_AUROC']:.6f} | {r['boundary_F1']:.6f} | {r['F_beta_w']:.6f} | {r['MAE']:.6f} |")
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({m:{k:macro[m][k] for k in ('pixel_AP','pixel_AUROC','boundary_F1')} for m in methods},indent=2),flush=True)


if __name__ == "__main__":
    main()
