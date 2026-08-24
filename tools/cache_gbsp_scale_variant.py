#!/usr/bin/env python3
"""Generate one resumable, mechanism-registered GBSP scale variant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_resolution import (  # noqa: E402
    fit_coarse_background_fine_query,
    fit_gbsp_from_prepared,
    prepare_resolution_graph,
)


def _manifest(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for row in read_jsonl(path):
        result[(str(row["dataset"]), str(row["stem"]))] = row
    return result


def _valid(path: Path, method: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = torch_load(path, map_location="cpu")
        score = value.get("absolute_minmax")
        return (value.get("method") == method and value.get("settings_fingerprint") == fingerprint
                and torch.is_tensor(score) and tuple(score.shape) == (1, 64, 64)
                and bool(torch.isfinite(score).all()))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mode", choices=("standard", "coarse-bg32-fine-q64"), default="standard")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--failure-policy", choices=("strict", "record"), default="record")
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    started = time.perf_counter()
    cfg = load_config(args.config)
    params = dict(DABE_V2_DEFAULT_PARAMS); params.update(_params_from_cfg(cfg))
    features = _manifest(Path(args.feature_manifest).resolve())
    samples = read_jsonl(Path(args.sample_list).resolve())
    if args.max_samples >= 0:
        samples = samples[:args.max_samples]
    out = Path(args.out_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((Path(args.config).read_text()+args.method+args.mode).encode()).hexdigest()
    rows, failures = [], []
    generated = reused = 0
    for index, sample in enumerate(samples, 1):
        dataset, stem = str(sample["dataset"]), str(sample["stem"])
        path = out / "scores" / dataset / f"{stem}.pt"
        if _valid(path, args.method, fingerprint):
            reused += 1
            payload = torch_load(path, map_location="cpu")
        else:
            try:
                row = features[(dataset, stem)]
                source = torch_load(row["cache_path"], map_location="cpu")
                feature = source["tensor"].detach().cpu().float().contiguous()
                if tuple(feature.shape) != (384, 64, 64):
                    raise RuntimeError(f"feature shape mismatch: {tuple(feature.shape)}")
                if args.mode == "standard":
                    prepared = prepare_resolution_graph(feature, str(sample["image_path"]), params)
                    result = fit_gbsp_from_prepared(prepared, params,
                        pca_energy=float(cfg.GBSP_PCA_ENERGY), pca_min_rank=int(cfg.GBSP_PCA_MIN_RANK),
                        pca_max_rank=int(cfg.GBSP_PCA_MAX_RANK))
                    modeling_grid = 64
                else:
                    result = fit_coarse_background_fine_query(feature, str(sample["image_path"]), params,
                        pca_energy=float(cfg.GBSP_PCA_ENERGY), pca_min_rank=int(cfg.GBSP_PCA_MIN_RANK),
                        pca_max_rank=int(cfg.GBSP_PCA_MAX_RANK), pooling=2)
                    modeling_grid = 32
                payload = {"version": "gbsp_scale_adaptation_v1", "method": args.method,
                    "dataset": dataset, "stem": stem, "image_path": str(sample["image_path"]),
                    "gt_used_for_generation": False, "settings_fingerprint": fingerprint,
                    "mode": args.mode, "input_size": 512, "query_grid": 64,
                    "background_modeling_grid": modeling_grid,
                    "border_width": int(params["BORDER_WIDTH"]),
                    "bg_ratio": float(params["BG_ANCHOR_TOP_PERCENT"]) / 100.0,
                    "sigma_f": float(params["SIGMA_F"]), "sigma_c": float(params["SIGMA_C"]),
                    "sigma_e": float(params["SIGMA_E"]), "threshold": float(cfg.GBSP_THRESHOLD),
                    "absolute_raw": result.raw_residual, "absolute_minmax": result.minmax_residual,
                    "selected_rank": int(result.selected_rank),
                    "required_rank_uncapped": int(result.required_rank_uncapped),
                    "energy_at_r8": float(result.energy_at_rank_8),
                    "energy_at_r12": float(result.energy_at_rank_12),
                    "candidate_count": int(result.background_indices.numel()),
                    "background_indices": result.background_indices, "bc": result.bc}
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path)
                generated += 1
            except Exception as exc:
                failure = {"dataset": dataset, "stem": stem, "error": repr(exc)}
                failures.append(failure)
                if args.failure_policy == "strict":
                    write_jsonl(out / "generation_failures.jsonl", failures)
                    raise
                continue
        rows.append({"dataset": dataset, "stem": stem, "method": args.method,
            "score_path": str(path), "required_rank_90": int(payload["required_rank_uncapped"]),
            "energy_at_r8": float(payload["energy_at_r8"]), "energy_at_r12": float(payload["energy_at_r12"]),
            "candidate_count": int(payload["candidate_count"]), "status": "reused" if _valid(path,args.method,fingerprint) and not generated else "valid"})
        if index % 20 == 0 or index == len(samples):
            print(f"[{index}/{len(samples)}] {args.method} valid={len(rows)} failed={len(failures)}", flush=True)
            write_jsonl(out / "manifest.jsonl", rows); write_jsonl(out / "generation_failures.jsonl", failures)
    metadata = {"method": args.method, "mode": args.mode, "requested": len(samples), "valid": len(rows),
        "failed": len(failures), "generated": generated, "reused": reused,
        "runtime_seconds": time.perf_counter()-started, "config": str(Path(args.config).resolve()),
        "feature_manifest": str(Path(args.feature_manifest).resolve()), "gt_used_for_generation": False}
    write_json(out / "generation_metadata.json", metadata)
    (out / "generation_command.txt").write_text(" ".join([sys.executable,*sys.argv])+"\n",encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
