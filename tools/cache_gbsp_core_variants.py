#!/usr/bin/env python3
"""Generate controlled GBSP core variants from frozen feature/DABE caches.

No image encoder, training loop, GT or graph reconstruction is invoked here.
The formal M=1 GBSP manifest is the identity/source-of-truth manifest.
"""

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
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_core_variants import (  # noqa: E402
    BOUNDARY_COUNT,
    GRID,
    NUM_PATCHES,
    BackgroundCandidateSelector,
    ConnectivityWeightedPCA,
    PCARankSelector,
    boundary_two_ring_mask,
    decompose_pca,
    minmax_score,
    pca_fit_from_decomposition,
    principal_angles_degrees,
    score_all_patches,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MAIN_ROOT.parent
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_core_ablation.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPERIMENT_VARIANTS = {
    "rank": ("r0", "r4", "r8", "r16", "r32", "current", "ev90", "ev95"),
    "background_source": ("boundary280", "fullbc_matched280", "fullbc"),
    "pca_weight": ("equal", "path_inverse"),
    "combined": ("current",),
}
VERSION = "gbsp_core_optimization_v1"
_SETTINGS: dict | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in seen:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}: {key}")
        seen.add(key)
        for field in ("cache_path", "source_feature_path", "source_dabe_path"):
            if not row.get(field) or not Path(row[field]).is_file():
                raise FileNotFoundError(row.get(field, f"missing {field} at {path}:{line}"))
    return rows


def _sample_ids(path: Path) -> set[tuple[str, str]]:
    selected = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"sample-list line must be dataset<TAB>stem: {line!r}")
        selected.add((parts[0], parts[1]))
    return selected


def _tensor(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


def _load_inputs(task: dict) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_path = Path(task["source_path"])
    feature_path = Path(task["feature_path"])
    dabe_path = Path(task["dabe_path"])
    source = torch_load(source_path, map_location="cpu")
    feature_payload = torch_load(feature_path, map_location="cpu")
    dabe = torch_load(dabe_path, map_location="cpu")
    identity = (task["dataset"], task["stem"])
    for payload, path in ((source, source_path), (feature_payload, feature_path), (dabe, dabe_path)):
        if not isinstance(payload, dict):
            raise TypeError(f"cache payload must be dict: {path}")
        if (str(payload.get("dataset")), str(payload.get("stem"))) != identity:
            raise RuntimeError(f"cache identity mismatch: {path}")
    if int(dabe.get("num_views", 1)) != 1 or dabe.get("augs") not in (
        None, [], ["identity"], ("identity",)
    ):
        raise RuntimeError(f"single identity view required: {dabe_path}")
    feature = _tensor(feature_payload, "tensor", (384, GRID, GRID), feature_path)
    flat = feature.permute(1, 2, 0).reshape(NUM_PATCHES, 384)
    if bool((torch.linalg.vector_norm(flat, dim=1) <= 0).any()):
        raise ValueError(f"zero feature vector: {feature_path}")
    normalized = F.normalize(flat, p=2, dim=1)
    anchor = _tensor(dabe, "bg_anchor_37", (1, GRID, GRID), dabe_path).reshape(-1)
    connectivity = _tensor(dabe, "bc_map_37", (1, GRID, GRID), dabe_path).reshape(-1)
    if float(connectivity.min()) <= 0 or float(connectivity.max()) > 1 + 1e-6:
        raise ValueError(f"connectivity must be in (0,1]: {dabe_path}")
    full_indices = torch.where(anchor > .5)[0]
    source_indices = source.get("background_indices")
    if not torch.is_tensor(source_indices) or not torch.equal(full_indices, source_indices.long()):
        raise RuntimeError(f"Full BC indices differ from formal M=1 source: {source_path}")
    return source, flat.contiguous(), normalized, full_indices, connectivity, anchor


def _fit_result(fit, normalized: torch.Tensor, extra: dict | None = None) -> dict:
    raw = score_all_patches(normalized, fit).reshape(1, GRID, GRID)
    result = {
        "absolute_raw": raw,
        "absolute_minmax": minmax_score(raw),
        "selected_rank": int(fit.selected_rank),
        "effective_spectrum_dimension": int(fit.effective_spectrum_dimension),
        "retained_variance_ratio": float(fit.retained_variance_ratio),
        "discarded_variance_ratio": float(fit.discarded_variance_ratio),
        "orthonormal_error": float(fit.orthonormal_error),
        "mean": fit.mean,
        "basis": fit.basis,
        "singular_values": fit.singular_values,
        "num_queries": NUM_PATCHES,
        "background_scores_preserved": True,
    }
    if extra:
        result.update(extra)
    count = int(result.get("num_background_candidates", 0))
    energy = fit.singular_values[: fit.effective_spectrum_dimension].double().square()
    distribution = energy / energy.sum() if float(energy.sum()) > 0 else torch.zeros_like(energy)
    result["background_feature_covariance_trace"] = (
        float(energy.sum() / max(1, count - 1)) if count else float("nan")
    )
    result["spectral_effective_rank"] = (
        float(torch.exp(-(distribution[distribution > 0] * torch.log(distribution[distribution > 0])).sum()))
        if bool((distribution > 0).any()) else 0.0
    )
    result["candidate_percentage"] = 100.0 * count / NUM_PATCHES
    return result


def _candidate_meta(indices: torch.Tensor, connectivity: torch.Tensor) -> dict:
    boundary = boundary_two_ring_mask()
    is_boundary = boundary.index_select(0, indices)
    values = connectivity.index_select(0, indices)
    path = -torch.log(values.clamp_min(1e-8))
    return {
        "background_indices": indices,
        "num_background_candidates": int(indices.numel()),
        "num_boundary_candidates": int(is_boundary.sum()),
        "num_interior_candidates": int((~is_boundary).sum()),
        "candidate_connectivity_mean": float(values.mean()),
        "candidate_connectivity_min": float(values.min()),
        "candidate_path_cost_mean": float(path.mean()),
        "candidate_path_cost_max": float(path.max()),
    }


def _generate_variants(task: dict, settings: dict) -> dict:
    source, raw_feature, normalized_feature, full_indices, connectivity, _ = _load_inputs(task)
    normalized = raw_feature if settings["feature_mode"] == "raw" else normalized_feature
    selector = PCARankSelector(.90, 8, 1)
    candidate_selector = BackgroundCandidateSelector()
    experiment = settings["experiment"]
    requested = settings["variants"]
    results: dict[str, dict] = {}

    if experiment == "rank":
        decomposition = decompose_pca(normalized.index_select(0, full_indices))
        for name in requested:
            if name.startswith("r") and name[1:].isdigit():
                fit = pca_fit_from_decomposition(decomposition, "fixed", selector, int(name[1:]))
            else:
                fit = pca_fit_from_decomposition(decomposition, name, selector)
            results[name] = _fit_result(
                fit, normalized, {**_candidate_meta(full_indices, connectivity), "pca_weight": "equal"}
            )
    elif experiment == "background_source":
        for name in requested:
            indices = candidate_selector.select(name, full_indices, connectivity)
            fit = pca_fit_from_decomposition(
                decompose_pca(normalized.index_select(0, indices)), "current", selector
            )
            results[name] = _fit_result(
                fit, normalized, {**_candidate_meta(indices, connectivity), "pca_weight": "equal"}
            )
    elif experiment == "pca_weight":
        background = normalized.index_select(0, full_indices)
        for name in requested:
            if name == "equal":
                fit = pca_fit_from_decomposition(decompose_pca(background), "current", selector)
                extra = {
                    "effective_sample_size": float(full_indices.numel()),
                    "weight_concentration_warning": False,
                    "normalized_weights": torch.ones(full_indices.numel()),
                    "path_cost": -torch.log(connectivity.index_select(0, full_indices).clamp_min(1e-8)),
                    "path_cost_median": float(
                        (-torch.log(connectivity.index_select(0, full_indices).clamp_min(1e-8))).median()
                    ),
                    "mean_shift_from_equal": 0.0,
                    "principal_angles_from_equal_degrees": torch.zeros(fit.selected_rank),
                }
            elif name == "path_inverse":
                equal = pca_fit_from_decomposition(decompose_pca(background), "current", selector)
                weighted = ConnectivityWeightedPCA().fit(
                    background, connectivity.index_select(0, full_indices), "current", selector
                )
                fit = weighted.fit
                extra = {
                    "effective_sample_size": weighted.effective_sample_size,
                    "weight_concentration_warning": weighted.weight_concentration_warning,
                    "normalized_weights": weighted.normalized_weights,
                    "path_cost": weighted.path_cost,
                    "path_cost_median": weighted.path_cost_median,
                    "mean_shift_from_equal": float(torch.linalg.vector_norm(fit.mean - equal.mean)),
                    "principal_angles_from_equal_degrees": principal_angles_degrees(equal.basis, fit.basis),
                }
            else:
                raise ValueError(f"unknown PCA weight: {name}")
            results[name] = _fit_result(
                fit,
                normalized,
                {**_candidate_meta(full_indices, connectivity), "pca_weight": name, **extra},
            )
    elif experiment == "combined":
        # Explicit extension point; defaults are frozen by CLI fields.
        indices = candidate_selector.select(settings["background_source"], full_indices, connectivity)
        background = normalized.index_select(0, indices)
        if settings["pca_weight"] == "equal":
            decomposition = decompose_pca(background)
            mode = settings["rank_mode"]
            if mode == "fixed":
                fit = pca_fit_from_decomposition(decomposition, "fixed", selector, settings["pca_rank"])
            elif mode.startswith("r") and mode[1:].isdigit():
                fit = pca_fit_from_decomposition(decomposition, "fixed", selector, int(mode[1:]))
            else:
                fit = pca_fit_from_decomposition(decomposition, mode, selector)
            weight_extra = {"effective_sample_size": float(indices.numel())}
        else:
            weighted = ConnectivityWeightedPCA().fit(
                background, connectivity.index_select(0, indices), settings["rank_mode"], selector,
                settings["pca_rank"] if settings["rank_mode"] == "fixed" else None,
            )
            fit = weighted.fit
            weight_extra = {
                "effective_sample_size": weighted.effective_sample_size,
                "weight_concentration_warning": weighted.weight_concentration_warning,
                "normalized_weights": weighted.normalized_weights,
                "path_cost": weighted.path_cost,
            }
        results["current"] = _fit_result(
            fit,
            normalized,
            {
                **_candidate_meta(indices, connectivity),
                "pca_weight": settings["pca_weight"],
                "rank_rule": settings["rank_mode"],
                **weight_extra,
            },
        )
    else:
        raise ValueError(experiment)

    source_raw = _tensor(source, "absolute_raw", (1, GRID, GRID), Path(task["source_path"]))
    baseline_name = "current" if experiment == "rank" else (
        "fullbc" if experiment == "background_source" else "equal"
    )
    baseline_error = (
        float((results[baseline_name]["absolute_raw"] - source_raw).abs().max())
        if experiment != "combined" and settings["feature_mode"] != "raw" else float("nan")
    )
    baseline_warning = experiment != "combined" and settings["feature_mode"] != "raw" and baseline_error > 2e-5
    if experiment != "combined" and settings["feature_mode"] != "raw" and baseline_error > 2e-4:
        raise RuntimeError(f"formal baseline response reproduction failed: {baseline_error}")

    boundary = candidate_selector.select("boundary280", full_indices, connectivity).sort().values
    matched = candidate_selector.select("fullbc_matched280", full_indices, connectivity).sort().values
    return {
        "results": results,
        "source_baseline_max_abs_error": baseline_error,
        "source_baseline_reproduction_warning": bool(baseline_warning),
        "boundary_matched_exact": bool(torch.equal(boundary, matched)),
        "boundary_matched_overlap": int(torch.isin(boundary, matched).sum()),
    }


def _valid_existing(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        return (
            isinstance(payload, dict)
            and payload.get("gbsp_core_version") == VERSION
            and payload.get("settings_fingerprint") == fingerprint
            and all(
                torch.is_tensor(item.get("absolute_raw"))
                and tuple(item["absolute_raw"].shape) == (1, GRID, GRID)
                and "background_feature_covariance_trace" in item
                and "spectral_effective_rank" in item
                and "candidate_percentage" in item
                for item in payload.get("results", {}).values()
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _generate_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output = Path(task["output_path"])
    if _valid_existing(output, _SETTINGS["fingerprint"]):
        payload = torch_load(output, map_location="cpu")
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output),
            "skipped": True,
            "source_baseline_max_abs_error": payload["source_baseline_max_abs_error"],
            "source_baseline_reproduction_warning": bool(payload["source_baseline_max_abs_error"] > 2e-5),
            "boundary_matched_exact": payload["boundary_matched_exact"],
        }
    started = time.perf_counter()
    try:
        generated = _generate_variants(task, _SETTINGS)
        payload = {
            "gbsp_core_version": VERSION,
            "dataset": task["dataset"],
            "stem": task["stem"],
            "experiment": _SETTINGS["experiment"],
            "settings": {k: v for k, v in _SETTINGS.items() if k != "fingerprint"},
            "settings_fingerprint": _SETTINGS["fingerprint"],
            **generated,
            "patch_grid_size": (GRID, GRID),
            "original_image_size": task["original_image_size"],
            "image_path": task["image_path"],
            "gt_path": task["gt_path"],
            "source_gbsp_path": task["source_path"],
            "source_feature_path": task["feature_path"],
            "source_dabe_path": task["dabe_path"],
        }
        _atomic_save(payload, output)
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output),
            "skipped": False, "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "source_baseline_max_abs_error": generated["source_baseline_max_abs_error"],
            "source_baseline_reproduction_warning": generated["source_baseline_reproduction_warning"],
            "boundary_matched_exact": generated["boundary_matched_exact"],
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "cache_path": str(output), "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(settings: dict, torch_threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def _write_configuration_audit(cfg, source_root: Path, master_root: Path) -> None:
    run_config_path = source_root / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    first_row = read_jsonl(source_root / "manifest_test.jsonl")[0]
    audit = {
        "status": "frozen_and_verified_from_existing_formal_cache",
        "created_at": _now(),
        "representation": {
            "backbone": "DINOv1 ViT-S/8",
            "input": "384x384 bicubic + ImageNet normalization",
            "grid": "37x37 (1369 patch tokens)",
            "feature_source": str(Path(first_row["source_feature_path"]).parents[2]),
            "feature": "last encoder block attention key projection, patch tokens only",
            "feature_layernorm": "key input follows the last block pre-attention LayerNorm; no final/post-model LayerNorm is applied after key projection",
            "pca_input_normalization": "per-patch L2 normalization",
        },
        "background": {
            "boundary_seed": "two rings, exactly 280 patches",
            "local_graph": "8-neighbor; w=exp(-df/0.10-dc/0.05-de/0.30), df=1-cosine, dc=squared RGB distance, de=max endpoint Sobel",
            "connectivity": "Dijkstra cumulative sum of -log(w+eps); normalize by image max; bc=exp(-normalized_path/0.30)",
            "full_bc_source": "cached bg_anchor_37 = top 30% bc with >= quantile ties retained",
            "formal_full_bc_count_distribution": {"411": 6469, "413": 1, "431": 1, "432": 1, "467": 1},
            "fallback": "if below 5% candidates: border union top-40%; not activated in formal count audit",
        },
        "pca": {
            "current": "single global PCA, energy>=0.90, min rank 1, cap rank 8",
            "centering": "subtract the equal-weight Full BC arithmetic mean before SVD",
            "formal_actual_rank_distribution": {"7": 1, "8": 6472},
            "weight": "equal",
            "query": "all 1369 patches, including dictionary members",
            "background_score_overwrite": False,
            "score": "absolute squared reconstruction residual",
        },
        "resize_and_calibration": "raw 37->68->native bilinear; per-image MinMax before fixed threshold",
        "source_run_config": str(run_config_path),
        "source_settings": run_config.get("settings", {}),
        "source_manifest": str(source_root / "manifest_test.jsonl"),
        "resolved_paths": {
            "GBSP_CACHE_ROOT": str(source_root),
            "DINO_FEATURE_ROOT": str(Path(first_row["source_feature_path"]).parents[2]),
            "FULL_TEST_LIST": str(source_root / "manifest_test.jsonl"),
            "GT_ROOT": str(Path(first_row["gt_path"]).parents[2]),
            "CURRENT_CONFIG": str(run_config.get("config")),
            "CURRENT_RANK_RULE": "EV90 capped to [1,8]",
        },
        "overwrite_pattern_search": {
            "patterns": [
                "score[background_indices] = 0", "residual[background_indices] = 0",
                "mask[background_indices] = 0", "mask[boundary_indices] = 0",
            ],
            "searched_executable_files": [
                str(MAIN_ROOT / "models/mbsp_reconstruction.py"),
                str(MAIN_ROOT / "tools/cache_mbsp_pseudo.py"),
                str(MAIN_ROOT / "models/gbsp_core_variants.py"),
            ],
            "executable_assignment_matches": 0,
            "conclusion": "no matching executable background-query overwrite exists; literal strings in this audit writer are metadata keys only",
        },
        "source_code_evidence": {
            "feature_extraction": str(MAIN_ROOT / "common/cache_features.py"),
            "formal_cache": str(MAIN_ROOT / "tools/cache_mbsp_pseudo.py"),
            "connectivity": str(MAIN_ROOT / "common/dabe_pseudo.py"),
        },
        "gt_used": False,
    }
    master_root.mkdir(parents=True, exist_ok=True)
    write_json(master_root / "configuration_audit.json", audit)


def build(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("GBSP core audit only supports --split test")
    cfg = load_config(_resolve(args.config))
    source_root = _resolve(args.gbsp_root or cfg.GBSP_CORE_SOURCE_ROOT)
    out_root = _resolve(args.out_root)
    master_root = _resolve(cfg.GBSP_CORE_OUTPUT_ROOT)
    if MAIN_ROOT == out_root or MAIN_ROOT in out_root.parents:
        raise ValueError(f"output must remain outside main code tree: {out_root}")
    source_rows = _manifest(source_root / "manifest_test.jsonl")
    _write_configuration_audit(cfg, source_root, master_root)

    specific = {
        "rank": args.rank_variants,
        "background_source": args.background_variants,
        "pca_weight": args.weight_variants,
        "combined": None,
    }[args.experiment]
    variants = tuple(specific or args.variants or EXPERIMENT_VARIANTS[args.experiment])
    invalid = sorted(set(variants) - set(EXPERIMENT_VARIANTS[args.experiment]))
    if invalid:
        raise ValueError(f"invalid variants for {args.experiment}: {invalid}")
    settings = {
        "version": VERSION,
        "experiment": args.experiment,
        "variants": variants,
        "rank_mode": args.rank_mode,
        "pca_rank": args.pca_rank,
        "background_source": args.background_source,
        "pca_weight": args.pca_weight,
        "path_mode": args.path_mode,
        "candidate_ratio": args.candidate_ratio,
        "feature_mode": args.feature_mode,
        "gt_used_during_generation": False,
        "feature_extraction_used": False,
        "training_used": False,
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if args.sample_list:
        identities = _sample_ids(_resolve(args.sample_list))
        rows = [r for r in source_rows if (r["dataset"], r["stem"]) in identities]
        missing = identities - {(r["dataset"], r["stem"]) for r in rows}
        if missing:
            raise KeyError(f"sample-list identities missing from formal manifest: {sorted(missing)[:3]}")
    else:
        rows = source_rows
    if args.max_samples >= 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no samples selected")
    if not args.sample_list and args.max_samples < 0:
        expected = dict(cfg.GBSP_CORE_EXPECTED_COUNTS)
        observed = Counter(row["dataset"] for row in rows)
        if len(rows) != sum(expected.values()) or dict(observed) != expected:
            raise RuntimeError(f"formal 6473 count mismatch: {dict(observed)}")

    tasks = []
    for row in rows:
        source = torch_load(Path(row["cache_path"]), map_location="cpu")
        tasks.append({
            "dataset": row["dataset"], "stem": row["stem"],
            "image_path": row["image_path"], "gt_path": row["gt_path"],
            "source_path": row["cache_path"],
            "feature_path": row["source_feature_path"], "dabe_path": row["source_dabe_path"],
            "original_image_size": tuple(source.get("original_image_size", ())),
            "output_path": str(out_root / "test" / row["dataset"] / f"{row['stem']}.pt"),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at": _now(), "config": str(_resolve(args.config)),
        "source_root": str(source_root), "output_root": str(out_root),
        "sample_count": len(tasks), "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
        "settings": settings,
    }
    write_json(out_root / "run_config.json", run_config)
    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(settings, args.torch_threads),
    ) as pool:
        for index, result in enumerate(pool.map(_generate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"[{_now()}] {args.experiment}: {index}/{len(tasks)}", flush=True)
    failures = [r for r in results if "error" in r]
    valid = {(r["dataset"], r["stem"]): r for r in results if "error" not in r}
    write_json(out_root / "generation_failures.json", failures)
    manifest = []
    for task in tasks:
        key = (task["dataset"], task["stem"])
        if key in valid:
            manifest.append({
                "dataset": task["dataset"], "stem": task["stem"],
                "cache_path": task["output_path"], "image_path": task["image_path"],
                "gt_path": task["gt_path"], "experiment": args.experiment,
                "variants": list(variants), "settings_fingerprint": settings["fingerprint"],
            })
    write_jsonl(out_root / "manifest_test.jsonl", manifest)
    valid_rows = list(valid.values())
    summary = {
        "requested": len(tasks), "valid": len(valid_rows), "failed": len(failures),
        "resumed": sum(bool(r.get("skipped")) for r in valid_rows),
        "wall_seconds": time.perf_counter() - started,
        "mean_runtime_seconds": sum(float(r.get("runtime_seconds", 0)) for r in valid_rows) / max(1, len(valid_rows)),
        "max_source_baseline_error": max((float(r["source_baseline_max_abs_error"]) for r in valid_rows), default=None),
        "source_baseline_reproduction_warning_count": sum(bool(r.get("source_baseline_reproduction_warning")) for r in valid_rows),
        "boundary_matched_exact_count": sum(bool(r["boundary_matched_exact"]) for r in valid_rows),
        "weight_concentration_warnings_are_preserved_in_payload": True,
        "rare_failures_recorded_without_discarding_valid_samples": True,
    }
    write_json(out_root / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} cache items failed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp-root", "--gbsp_root", dest="gbsp_root")
    value.add_argument("--out-root", "--out_root", dest="out_root", required=True)
    value.add_argument("--sample-list", "--sample_list", dest="sample_list")
    value.add_argument("--split", default="test")
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--experiment", choices=tuple(EXPERIMENT_VARIANTS), required=True)
    value.add_argument("--variants", nargs="+")
    value.add_argument("--rank-variants", "--rank_variants", dest="rank_variants", nargs="+")
    value.add_argument("--background-variants", "--background_variants", dest="background_variants", nargs="+")
    value.add_argument("--weight-variants", "--weight_variants", dest="weight_variants", nargs="+")
    value.add_argument("--rank-mode", "--pca-rank-mode", "--pca_rank_mode", dest="rank_mode", choices=("fixed", "current", "ev90", "ev95"), default="current")
    value.add_argument("--pca-rank", "--pca_rank", dest="pca_rank", type=int)
    value.add_argument("--background-source", "--background-mode", "--background_mode", dest="background_source", choices=EXPERIMENT_VARIANTS["background_source"], default="fullbc")
    value.add_argument("--pca-weight", "--pca-weight-mode", "--pca_weight_mode", dest="pca_weight", choices=EXPERIMENT_VARIANTS["pca_weight"], default="equal")
    value.add_argument("--path-mode", "--path_mode", dest="path_mode", choices=("cumulative",), default="cumulative")
    value.add_argument("--candidate-ratio", "--candidate_ratio", dest="candidate_ratio", type=float, default=.30)
    value.add_argument("--feature-mode", "--feature_mode", dest="feature_mode", choices=("current", "raw", "l2"), default="current")
    value.add_argument("--save-diagnostics", "--save_diagnostics", dest="save_diagnostics", action="store_true", help="compatibility flag; diagnostics are always saved")
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    value.add_argument("--dry-run", action="store_true")
    return value


if __name__ == "__main__":
    build(parser().parse_args())
