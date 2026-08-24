#!/usr/bin/env python3
"""Generate resumable 512-A/C/B continuous GBSP scores from one shared DINO cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_resolution import fit_gbsp_from_prepared, prepare_resolution_graph  # noqa: E402
from tools.eval_gbsp_resolution_probe import (  # noqa: E402
    _cfg_params,
    _load_feature,
    _manifest_map,
    _sample_rows,
    _validate_protocol,
)


METHOD_ORDER = ("512-A", "512-C", "512-B")
METHOD_DIRS = {
    "512-A": "A_bw2_r8",
    "512-B": "B_bw3_r8",
    "512-C": "C_bw2_r12",
}


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _score_path(root: Path, method: str, dataset: str, stem: str) -> Path:
    return root / METHOD_DIRS[method] / "scores" / dataset / f"{stem}.pt"


def _valid_existing(path: Path, method: str, dataset: str, stem: str, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        score = payload.get("absolute_minmax")
        return (
            payload.get("method") == method
            and payload.get("dataset") == dataset
            and payload.get("stem") == stem
            and payload.get("settings_fingerprint") == fingerprint
            and torch.is_tensor(score)
            and tuple(score.shape) == (1, 64, 64)
            and bool(torch.isfinite(score).all())
            and float(score.min()) >= 0.0
            and float(score.max()) <= 1.0
        )
    except Exception:
        return False


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-296", required=True)
    parser.add_argument("--config-512a", required=True)
    parser.add_argument("--config-512b", required=True)
    parser.add_argument("--config-512c", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--failure-policy", choices=("strict", "record"), default="record")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    main_root = Path(__file__).resolve().parents[1]
    config_paths = [
        Path(args.config_296).expanduser().resolve(),
        Path(args.config_512a).expanduser().resolve(),
        Path(args.config_512b).expanduser().resolve(),
        Path(args.config_512c).expanduser().resolve(),
    ]
    cfg296, cfg_a, cfg_b, cfg_c = (load_config(path) for path in config_paths)
    _validate_protocol(cfg296, cfg_a, cfg_b, cfg_c)
    configs = {"512-A": cfg_a, "512-B": cfg_b, "512-C": cfg_c}
    params = {method: _cfg_params(cfg) for method, cfg in configs.items()}
    fingerprint = _sha256(config_paths)
    commit = _git_commit(main_root)

    feature_manifest = Path(args.feature_manifest).expanduser().resolve()
    features = _manifest_map(feature_manifest)
    sample_list = Path(args.sample_list).expanduser().resolve()
    samples = _sample_rows(sample_list, int(args.max_samples))
    missing = [
        (str(row["dataset"]), str(row["stem"]))
        for row in samples
        if (str(row["dataset"]), str(row["stem"])) not in features
    ]
    if missing:
        raise RuntimeError(f"shared DINO512 cache misses {len(missing)} samples; first={missing[:5]}")

    out = Path(args.out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        method_root = out / METHOD_DIRS[method]
        method_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_paths[{"512-A": 1, "512-B": 2, "512-C": 3}[method]], method_root / "config_snapshot.py")
    shutil.copy2(sample_list, out / "sample_list.jsonl")
    command = " ".join([sys.executable, *sys.argv])
    (out / "generation_command.txt").write_text(command + "\n", encoding="utf-8")
    (out / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")

    manifest_path = out / "generation_manifest.jsonl"
    failures_path = out / "generation_failures.jsonl"
    metadata_path = out / "generation_metadata.json"
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    generated_images = reused_images = 0

    for index, sample in enumerate(samples, 1):
        dataset, stem = str(sample["dataset"]), str(sample["stem"])
        key = (dataset, stem)
        paths = {method: _score_path(out, method, dataset, stem) for method in METHOD_ORDER}
        if all(_valid_existing(path, method, dataset, stem, fingerprint) for method, path in paths.items()):
            reused_images += 1
            for method in METHOD_ORDER:
                payload = torch_load(paths[method], map_location="cpu")
                rows.append({
                    "dataset": dataset,
                    "stem": stem,
                    "method": method,
                    "score_path": str(paths[method]),
                    "status": "reused",
                    "grid_size": 64,
                    "border_width": int(configs[method].GBSP_BORDER_WIDTH),
                    "pca_rank_cap": int(configs[method].GBSP_PCA_MAX_RANK),
                    "selected_rank": int(payload["selected_rank"]),
                    "required_rank_uncapped": int(payload["required_rank_uncapped"]),
                    "energy_at_cap": float(payload["energy_at_cap"]),
                    "hit_rank_cap": int(payload["hit_rank_cap"]),
                    "candidate_count": int(payload["candidate_count"]),
                    "rgb_seconds": 0.0,
                    "graph_seconds": 0.0,
                    "bc_seconds": 0.0,
                    "pca_seconds": 0.0,
                    "wall_seconds": 0.0,
                })
            continue

        item_started = time.perf_counter()
        try:
            feature = _load_feature(features[key], cfg_a, key)
            prepared = prepare_resolution_graph(feature, sample["image_path"], params["512-A"])
            generated: dict[str, tuple] = {}
            for method in METHOD_ORDER:
                cfg = configs[method]
                fit_started = time.perf_counter()
                result = fit_gbsp_from_prepared(
                    prepared,
                    params[method],
                    pca_energy=float(cfg.GBSP_PCA_ENERGY),
                    pca_min_rank=int(cfg.GBSP_PCA_MIN_RANK),
                    pca_max_rank=int(cfg.GBSP_PCA_MAX_RANK),
                )
                generated[method] = (result, time.perf_counter() - fit_started)
            item_wall = time.perf_counter() - item_started
            for method in METHOD_ORDER:
                cfg = configs[method]
                result, fit_wall = generated[method]
                path = paths[method]
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": "gbsp_resolution_full_v1",
                    "method": method,
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": str(sample["image_path"]),
                    "gt_used_for_generation": False,
                    "settings_fingerprint": fingerprint,
                    "git_commit": commit,
                    "input_size": 512,
                    "grid_size": 64,
                    "feature_dim": 384,
                    "border_width": int(cfg.GBSP_BORDER_WIDTH),
                    "bg_ratio": float(cfg.GBSP_BG_RATIO),
                    "pca_energy": float(cfg.GBSP_PCA_ENERGY),
                    "pca_rank_cap": int(cfg.GBSP_PCA_MAX_RANK),
                    "threshold": float(cfg.GBSP_THRESHOLD),
                    "absolute_raw": result.raw_residual,
                    "absolute_minmax": result.minmax_residual,
                    "selected_rank": int(result.selected_rank),
                    "required_rank_uncapped": int(result.required_rank_uncapped),
                    "energy_at_cap": float(result.energy_at_rank_max),
                    "retained_energy": float(result.retained_energy),
                    "hit_rank_cap": bool(result.hit_rank_cap),
                    "candidate_count": int(result.background_indices.numel()),
                    "background_indices": result.background_indices,
                    "bc": result.bc,
                }
                torch.save(payload, path)
                rows.append({
                    "dataset": dataset,
                    "stem": stem,
                    "method": method,
                    "score_path": str(path),
                    "status": "generated",
                    "grid_size": 64,
                    "border_width": int(cfg.GBSP_BORDER_WIDTH),
                    "pca_rank_cap": int(cfg.GBSP_PCA_MAX_RANK),
                    "selected_rank": int(result.selected_rank),
                    "required_rank_uncapped": int(result.required_rank_uncapped),
                    "energy_at_cap": float(result.energy_at_rank_max),
                    "hit_rank_cap": int(result.hit_rank_cap),
                    "candidate_count": int(result.background_indices.numel()),
                    "rgb_seconds": float(prepared.rgb_seconds),
                    "graph_seconds": float(prepared.graph_seconds),
                    "bc_seconds": float(result.bc_seconds),
                    "pca_seconds": float(result.pca_seconds),
                    "fit_wall_seconds": float(fit_wall),
                    "wall_seconds": float(item_wall),
                })
            generated_images += 1
        except Exception as exc:
            failure = {
                "index": index,
                "dataset": dataset,
                "stem": stem,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
            }
            failures.append(failure)
            print(f"FAILED {dataset}/{stem}: {failure['exception']}", flush=True)
            if args.failure_policy == "strict":
                write_jsonl(failures_path, failures)
                raise

        if index % max(1, int(args.checkpoint_every)) == 0 or index == len(samples):
            write_jsonl(manifest_path, rows)
            write_jsonl(failures_path, failures)
            valid_images = len({(row["dataset"], row["stem"]) for row in rows})
            print(
                f"[{index}/{len(samples)}] valid={valid_images} failed={len(failures)} "
                f"generated={generated_images} reused={reused_images}",
                flush=True,
            )

    valid_keys = {(row["dataset"], row["stem"]) for row in rows}
    expected_keys = {(str(row["dataset"]), str(row["stem"])) for row in samples}
    counts = {
        dataset: sum(key[0] == dataset for key in valid_keys)
        for dataset in ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
    }
    metadata = {
        "version": "gbsp_resolution_full_generation_v1",
        "num_requested": len(samples),
        "num_valid": len(valid_keys),
        "num_failed": len(failures),
        "is_full_complete": valid_keys == expected_keys and not failures and len(samples) == 6473,
        "counts": counts,
        "generated_images": generated_images,
        "reused_images": reused_images,
        "method_order": list(METHOD_ORDER),
        "feature_manifest": str(feature_manifest),
        "sample_list": str(sample_list),
        "settings_fingerprint": fingerprint,
        "git_commit": commit,
        "gt_used_for_generation": False,
        "training_triggered": False,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(metadata_path, metadata)
    _write_csv(out / "generation_runtime.csv", rows)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if failures or valid_keys != expected_keys:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
