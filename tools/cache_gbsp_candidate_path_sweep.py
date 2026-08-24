#!/usr/bin/env python3
"""Generate resumable fixed-R32 GBSP caches for the BW x candidate-% sweep."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.cache_dabe_r1_design import _feature, _manifest_map  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS  # noqa: E402
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    torch_load,
    write_json,
    write_jsonl,
)
from models.gbsp_candidate_path_sweep import (  # noqa: E402
    build_candidate_path_variants,
    candidate_cache_version,
    candidate_variant_tag,
)
from models.gbsp_resolution import prepare_resolution_graph  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CACHE_ROOT = (PROJECT_ROOT / "datasets/cache").resolve()
EXPECTED_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
GRID = 37
FEATURE_DIM = 384
PCA_RANK = 32
DEFAULT_BORDER_WIDTHS = (1, 2)
DEFAULT_TOP_PERCENTS = (25.0, 27.5, 30.0)
DEFAULT_THRESHOLD = 0.50


def _cache_version(
    border_width: int,
    top_percent: float,
    pca_rank: int,
    version_label: str = "",
) -> str:
    base = candidate_cache_version(border_width, top_percent, pca_rank)
    label = str(version_label).strip().lower()
    if not label:
        return base
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", label):
        raise ValueError("version_label must match [a-z0-9][a-z0-9_]*")
    return f"{base[:-3]}_{label}_v1"


def _validate_cache_root(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output == CACHE_ROOT or CACHE_ROOT not in output.parents:
        raise ValueError(f"out_root must be a child of {CACHE_ROOT}: {output}")
    return output


def _balanced_subset(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples < 0 or max_samples >= len(rows):
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    quotas = {dataset: 0 for dataset in EXPECTED_COUNTS}
    remaining = int(max_samples)
    while remaining:
        progressed = False
        for dataset in EXPECTED_COUNTS:
            if quotas[dataset] < len(grouped[dataset]):
                quotas[dataset] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = []
    for dataset, count in quotas.items():
        if not count:
            continue
        indices = np.linspace(0, len(grouped[dataset]) - 1, count, dtype=np.int64)
        selected.extend(grouped[dataset][int(index)] for index in indices)
    return selected


def _fingerprint(params: dict, widths, percents, pca_rank: int) -> str:
    fields = {
        "graph": {
            key: params[key]
            for key in ("SIGMA_F", "SIGMA_C", "SIGMA_E", "TAU_BC")
        },
        "anchor": {
            key: params[key]
            for key in ("BG_ANCHOR_MIN_RATIO", "BG_ANCHOR_FALLBACK_TOP_PERCENT")
        },
        "border_widths": list(widths),
        "top_percents": list(percents),
        "pca_rank": int(pca_rank),
        "score": "squared_affine_pca_residual_per_image_minmax",
        "graph_path": "multi_source_dijkstra_neglog_affinity",
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _valid_existing(
    path: Path,
    *,
    dataset: str,
    stem: str,
    version: str,
    fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        score = payload.get("gbsp_abs_minmax_37")
        selected = payload.get("selected_ranks")
        return (
            isinstance(payload, dict)
            and payload.get("dataset") == dataset
            and payload.get("stem") == stem
            and payload.get("gbsp_version") == version
            and payload.get("settings_fingerprint") == fingerprint
            and payload.get("graph_path_method")
            == "multi_source_dijkstra_neglog_affinity"
            and bool(payload.get("graph_path_used"))
            and torch.is_tensor(score)
            and tuple(score.shape) == (1, GRID, GRID)
            and bool(torch.isfinite(score).all())
            and torch.is_tensor(selected)
            and selected.numel() == 1
            and int(selected.reshape(-1)[0]) == PCA_RANK
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict) -> dict:
    row = task["row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    widths = tuple(int(value) for value in task["border_widths"])
    percents = tuple(float(value) for value in task["top_percents"])
    pca_rank = int(task["pca_rank"])
    version_label = str(task.get("version_label", ""))
    threshold = float(task["diagnostic_threshold"])
    output_root = Path(task["output_root"])
    fingerprint = str(task["fingerprint"])
    paths = {
        candidate_variant_tag(width, percent): (
            output_root
            / candidate_variant_tag(width, percent)
            / "dinov1-s8"
            / "train"
            / dataset
            / f"{stem}.pt"
        )
        for width in widths
        for percent in percents
    }
    versions = {
        candidate_variant_tag(width, percent): _cache_version(
            width, percent, pca_rank, version_label
        )
        for width in widths
        for percent in percents
    }
    if all(
        _valid_existing(
            path,
            dataset=dataset,
            stem=stem,
            version=versions[tag],
            fingerprint=fingerprint,
        )
        for tag, path in paths.items()
    ):
        records = []
        for tag, path in paths.items():
            payload = torch_load(path, map_location="cpu")
            records.append(
                {
                    "variant": tag,
                    "cache_path": str(path),
                    "candidate_count": int(payload["candidate_count"]),
                    "boundary_seed_count": int(payload["boundary_seed_count"]),
                    "interior_candidate_count": int(
                        payload["interior_candidate_count"]
                    ),
                    "interior_candidate_ratio": float(
                        payload["interior_candidate_ratio"]
                    ),
                    "hard_area": float(payload["hard_area_at_static_threshold"]),
                }
            )
        return {
            "dataset": dataset,
            "stem": stem,
            "source_feature_cache_path": str(Path(row["cache_path"]).resolve()),
            "image_path": str(row["image_path"]),
            "skipped": True,
            "records": records,
        }

    started = time.perf_counter()
    try:
        feature_path = Path(row["cache_path"]).resolve()
        feature_payload = torch_load(feature_path, map_location="cpu")
        feature = _feature(feature_payload, dataset, stem, feature_path)
        if tuple(feature.shape) != (FEATURE_DIM, GRID, GRID):
            raise ValueError(f"feature must be [{FEATURE_DIM},{GRID},{GRID}]")
        image_path = Path(str(row.get("image_path") or feature_payload.get("image_path")))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        params = dict(task["params"])
        prepared = prepare_resolution_graph(feature, str(image_path), params)
        variants = build_candidate_path_variants(
            prepared,
            params,
            border_widths=widths,
            top_percents=percents,
            pca_rank=pca_rank,
        )
        records = []
        for tag, result in variants.items():
            response_68 = F.interpolate(
                result.minmax_residual.unsqueeze(0),
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            hard_area = float((response_68 > threshold).float().mean())
            version = versions[tag]
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": str(image_path.resolve()),
                "backbone_key": "dinov1-s8",
                "dabe_version": "v2",
                "source_dabe_version": "v2",
                "source_augs": ["identity"],
                "source_num_views": 1,
                "gbsp_version": version,
                "settings_fingerprint": fingerprint,
                "generation_stage": (
                    "recomputed_full_bc_dijkstra_candidate_sweep_fixed_r32"
                ),
                "dino_forward_used": False,
                "gt_used_for_generation": False,
                "foreground_seed_used": False,
                "boundary_only_used": False,
                "graph_path_used": True,
                "graph_path_method": "multi_source_dijkstra_neglog_affinity",
                "graph_neighborhood": 8,
                "graph_sigma_f": float(params["SIGMA_F"]),
                "graph_sigma_c": float(params["SIGMA_C"]),
                "graph_sigma_e": float(params["SIGMA_E"]),
                "graph_tau_bc": float(params["TAU_BC"]),
                "candidate_border_width": int(result.border_width),
                "candidate_top_percent": float(result.top_percent),
                "candidate_count": int(result.background_indices.numel()),
                "boundary_seed_count": int(result.boundary_seed_count),
                "boundary_candidate_count": int(result.boundary_candidate_count),
                "interior_candidate_count": int(result.interior_candidate_count),
                "interior_candidate_ratio": float(result.interior_candidate_ratio),
                "num_subspaces": 1,
                "pca_energy": 0.90,
                "pca_min_rank": 1,
                "pca_max_rank": pca_rank,
                "pca_rank_mode": "fixed",
                "fixed_pca_rank": pca_rank,
                "selected_ranks": torch.tensor([result.selected_rank], dtype=torch.long),
                "retained_variance_ratio": result.retained_variance_ratio,
                "singular_values": [result.singular_values],
                "subspace_mean": result.subspace_mean,
                "static_threshold": threshold,
                "gbsp_abs_raw_37": result.raw_residual,
                "gbsp_abs_minmax_37": result.minmax_residual,
                "background_indices": result.background_indices,
                "bc_map_37": result.bc,
                "bg_anchor_37": result.background_anchor.float(),
                "border_seed_mask_37": result.border,
                "hard_area_at_static_threshold": hard_area,
                "source_feature_cache_path": str(feature_path),
                "runtime": {
                    "total_all_variants_seconds": time.perf_counter() - started,
                    "torch_threads": torch.get_num_threads(),
                },
            }
            path = paths[tag]
            _atomic_save(payload, path)
            records.append(
                {
                    "variant": tag,
                    "cache_path": str(path),
                    "candidate_count": int(result.background_indices.numel()),
                    "boundary_seed_count": int(result.boundary_seed_count),
                    "interior_candidate_count": int(result.interior_candidate_count),
                    "interior_candidate_ratio": float(result.interior_candidate_ratio),
                    "hard_area": hard_area,
                }
            )
        return {
            "dataset": dataset,
            "stem": stem,
            "source_feature_cache_path": str(feature_path),
            "image_path": str(image_path.resolve()),
            "skipped": False,
            "records": records,
            "total_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        }
    except Exception as error:
        return {
            "dataset": dataset,
            "stem": stem,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def build_sweep(
    *,
    config_path: str | Path,
    out_root: str | Path,
    border_widths=DEFAULT_BORDER_WIDTHS,
    top_percents=DEFAULT_TOP_PERCENTS,
    pca_rank: int = PCA_RANK,
    diagnostic_threshold: float = DEFAULT_THRESHOLD,
    max_samples: int = -1,
    workers: int = 2,
    torch_threads: int = 4,
    strict_failures: bool = False,
    experimental_subset: bool = False,
    version_label: str = "",
) -> dict:
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if int(workers) <= 0 or int(torch_threads) <= 0:
        raise ValueError("workers and torch_threads must be positive")
    if int(pca_rank) != PCA_RANK:
        raise ValueError(f"the registered sweep fixes pca_rank={PCA_RANK}")
    if not 0.0 <= float(diagnostic_threshold) <= 1.0:
        raise ValueError("diagnostic_threshold must be in [0,1]")
    widths = tuple(dict.fromkeys(int(value) for value in border_widths))
    percents = tuple(dict.fromkeys(float(value) for value in top_percents))
    registered = {
        candidate_variant_tag(width, percent)
        for width in DEFAULT_BORDER_WIDTHS
        for percent in DEFAULT_TOP_PERCENTS
    }
    actual = {
        candidate_variant_tag(width, percent)
        for width in widths
        for percent in percents
    }
    if actual != registered and not (
        bool(experimental_subset) and actual and actual.issubset(registered)
    ):
        raise ValueError(
            "the first-priority registered sweep is exactly "
            "BORDER_WIDTH={1,2} x TOP_PERCENT={25,27.5,30}; "
            "use --experimental_subset for an explicit registered subset"
        )
    # Validate once before worker processes are launched.
    for width in widths:
        for percent in percents:
            _cache_version(width, percent, pca_rank, version_label)

    config_path = Path(config_path).resolve()
    output_root = _validate_cache_root(out_root)
    cfg = load_config(config_path)
    if str(getattr(cfg, "BACKBONE_KEY", "")) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    params = {**DABE_V2_DEFAULT_PARAMS, **_params_from_cfg(cfg)}
    feature_manifest = feature_manifest_path(cfg, "train").resolve()
    rows, row_map = _manifest_map(feature_manifest)
    counts = Counter(str(row["dataset"]) for row in rows)
    if len(rows) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"formal feature manifest mismatch: {len(rows)} {dict(counts)}")
    if len(row_map) != len(rows):
        raise RuntimeError("feature manifest contains duplicate identities")
    selected = _balanced_subset(rows, int(max_samples))
    fingerprint = _fingerprint(params, widths, percents, int(pca_rank))
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "row": row,
            "output_root": str(output_root),
            "params": params,
            "border_widths": widths,
            "top_percents": percents,
            "pca_rank": int(pca_rank),
            "diagnostic_threshold": float(diagnostic_threshold),
            "fingerprint": fingerprint,
            "version_label": str(version_label),
        }
        for row in selected
    ]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(int(torch_threads),),
    ) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                failed = sum("error" in row for row in results)
                print(
                    f"GBSP candidate-path sweep {index}/{len(tasks)} failed={failed}",
                    flush=True,
                )

    failures = [result for result in results if "error" in result]
    valid = [result for result in results if "error" not in result]
    write_json(output_root / "generation_failures.json", failures)
    variant_records: dict[str, list[dict]] = defaultdict(list)
    for result in valid:
        for record in result["records"]:
            variant_records[record["variant"]].append({**result, **record})
    variant_summaries = {}
    for width in widths:
        for percent in percents:
            tag = candidate_variant_tag(width, percent)
            records = variant_records[tag]
            variant_root = output_root / tag / "dinov1-s8"
            manifest_rows = [
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "image_path": row["image_path"],
                    "cache_path": row["cache_path"],
                    "source_feature_cache_path": row["source_feature_cache_path"],
                    "backbone_key": "dinov1-s8",
                    "gbsp_version": _cache_version(
                        width, percent, pca_rank, version_label
                    ),
                    "shape": [1, GRID, GRID],
                    "candidate_border_width": int(width),
                    "candidate_top_percent": float(percent),
                    "graph_path_used": True,
                }
                for row in records
            ]
            write_jsonl(variant_root / "manifest_train.jsonl", manifest_rows)
            summary = {
                "variant": tag,
                "gbsp_version": _cache_version(
                    width, percent, pca_rank, version_label
                ),
                "num_requested": len(tasks),
                "num_valid": len(records),
                "dataset_counts": dict(Counter(row["dataset"] for row in records)),
                "candidate_border_width": int(width),
                "candidate_top_percent": float(percent),
                "mean_candidate_count": _mean(records, "candidate_count"),
                "mean_boundary_seed_count": _mean(records, "boundary_seed_count"),
                "mean_interior_candidate_count": _mean(
                    records, "interior_candidate_count"
                ),
                "mean_interior_candidate_ratio": _mean(
                    records, "interior_candidate_ratio"
                ),
                "mean_hard_area_at_static_threshold": _mean(records, "hard_area"),
                "pca_rank_mode": "fixed",
                "fixed_pca_rank": int(pca_rank),
                "static_threshold": float(diagnostic_threshold),
                "boundary_only_used": False,
                "graph_path_used": True,
                "graph_path_method": "multi_source_dijkstra_neglog_affinity",
                "gt_used_for_generation": False,
                "training_used": False,
                "settings_fingerprint": fingerprint,
                "source_feature_manifest": str(feature_manifest),
                "output_root": str(variant_root),
            }
            write_json(variant_root / "protocol.json", summary)
            variant_summaries[tag] = summary
    sweep_summary = {
        "version": "gbsp_r32_candidate_path_sweep_v1",
        "num_requested": len(tasks),
        "num_valid_images": len(valid),
        "num_failed_images": len(failures),
        "num_resumed_images": sum(bool(row.get("skipped")) for row in valid),
        "variants": variant_summaries,
        "config_path": str(config_path),
        "source_feature_manifest": str(feature_manifest),
        "output_root": str(output_root),
        "wall_seconds": time.perf_counter() - started,
        "gt_used_for_generation": False,
        "training_used": False,
        "formal_full4040": len(tasks) == EXPECTED_TOTAL and not failures,
        "experimental_subset": bool(experimental_subset),
        "version_label": str(version_label),
    }
    write_json(output_root / "sweep_protocol.json", sweep_summary)
    print(json.dumps(sweep_summary, ensure_ascii=False, indent=2), flush=True)
    if failures and strict_failures:
        raise RuntimeError(f"candidate-path sweep failed for {len(failures)} images")
    return sweep_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--border_widths", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--top_percents", nargs="+", type=float, default=[25.0, 27.5, 30.0]
    )
    parser.add_argument("--pca_rank", type=int, default=PCA_RANK)
    parser.add_argument("--diagnostic_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--strict_failures", action="store_true")
    parser.add_argument(
        "--experimental_subset",
        action="store_true",
        help="Allow a non-empty subset of the registered BW x percent grid.",
    )
    parser.add_argument(
        "--version_label",
        default="",
        help="Append a settings label such as sigmaf085 to cache versions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sweep(
        config_path=args.config,
        out_root=args.out_root,
        border_widths=args.border_widths,
        top_percents=args.top_percents,
        pca_rank=args.pca_rank,
        diagnostic_threshold=args.diagnostic_threshold,
        max_samples=args.max_samples,
        workers=args.workers,
        torch_threads=args.torch_threads,
        strict_failures=args.strict_failures,
        experimental_subset=args.experimental_subset,
        version_label=args.version_label,
    )
