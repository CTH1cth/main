#!/usr/bin/env python3
"""Generate GBSP-V2 B0--B5 from existing formal GBSP/DINO caches.

No DINO extraction, GT access, training, mask propagation or morphology occurs
during generation. Rare per-image failures are recorded and do not erase valid
outputs from other images.
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
from models.gbsp_v2_background import (  # noqa: E402
    BOUNDARY_COUNT,
    FEATURE_DIM,
    GRID,
    NUM_PATCHES,
    boundary_two_ring_mask,
    build_local_graph,
    candidate_diagnostics,
    confidence_diversity_coreset,
    hard_boundary_seed,
    load_rgb_grid,
    minmax,
    random_walk_with_restart,
    soft_boundary_seed,
    top_confidence_indices,
)
from models.gbsp_v2_pca import (  # noqa: E402
    confidence_rank_weights,
    euclidean_residual_score,
    fit_fixed_pca,
    principal_angles_degrees,
)
from models.gbsp_v2_residual import shrinkage_whitened_score  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_v2.py"
VERSION = "gbsp_v2_softbc_swor_v1"
VARIANTS = ("b0", "b1", "b2", "b3", "b4", "b5")
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
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


def _tensor(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


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
        if not row.get("cache_path") or not Path(row["cache_path"]).is_file():
            raise FileNotFoundError(row.get("cache_path", f"missing cache_path at line {line}"))
    return rows


def _sample_ids(path: Path) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"sample list must use dataset<TAB>stem: {line!r}")
        selected.add((parts[0], parts[1]))
    return selected


def _load_source(task: dict) -> tuple[dict, dict, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_path = Path(task["source_path"])
    source = torch_load(source_path, map_location="cpu")
    if not isinstance(source, dict) or source.get("gbsp_core_version") != "gbsp_core_optimization_v1":
        raise RuntimeError(f"formal GBSP core cache required: {source_path}")
    identity = (task["dataset"], task["stem"])
    if (source.get("dataset"), source.get("stem")) != identity:
        raise RuntimeError(f"GBSP source identity mismatch: {source_path}")
    feature_path = Path(source["source_feature_path"])
    dabe_path = Path(source["source_dabe_path"])
    feature_payload = torch_load(feature_path, map_location="cpu")
    dabe_payload = torch_load(dabe_path, map_location="cpu")
    for payload, path in ((feature_payload, feature_path), (dabe_payload, dabe_path)):
        if not isinstance(payload, dict) or (payload.get("dataset"), payload.get("stem")) != identity:
            raise RuntimeError(f"source identity mismatch: {path}")
    if int(dabe_payload.get("num_views", 1)) != 1 or dabe_payload.get("augs") not in (
        None, [], ["identity"], ("identity",)
    ):
        raise RuntimeError(f"single identity-view DABE cache required: {dabe_path}")
    feature = _tensor(feature_payload, "tensor", (FEATURE_DIM, GRID, GRID), feature_path)
    flat = feature.permute(1, 2, 0).reshape(NUM_PATCHES, FEATURE_DIM)
    if bool((torch.linalg.vector_norm(flat, dim=1) <= 0).any()):
        raise ValueError(f"zero DINO feature vector: {feature_path}")
    normalized = F.normalize(flat, p=2, dim=1).contiguous()
    connectivity = _tensor(dabe_payload, "bc_map_37", (1, GRID, GRID), dabe_path).reshape(-1)
    anchor = _tensor(dabe_payload, "bg_anchor_37", (1, GRID, GRID), dabe_path).reshape(-1)
    return source, feature_payload, dabe_payload, normalized, connectivity, anchor


def _rw_diagnostics(result, graph, seed: torch.Tensor) -> dict:
    boundary = boundary_two_ring_mask()
    return {
        "edge_scale": float(graph.edge_scale),
        "transition_row_sum_error": float(graph.row_sum_error),
        "alpha": float(_SETTINGS["alpha"]),
        "iterations": int(result.iterations),
        "convergence_error": float(result.convergence_error),
        "converged": bool(result.converged),
        "seed_sum": float(result.seed_sum),
        "seed_nonzero_count": int((seed > 0).sum()),
        "confidence_min": float(result.confidence.min()),
        "confidence_max": float(result.confidence.max()),
        "confidence_mean": float(result.confidence.mean()),
        "boundary_confidence_mean": float(result.confidence[boundary].mean()),
        "interior_confidence_mean": float(result.confidence[~boundary].mean()),
    }


def _pca_result(
    name: str,
    score: torch.Tensor,
    indices: torch.Tensor,
    confidence: torch.Tensor,
    features: torch.Tensor,
    model,
    *,
    pca_weighting: str,
    candidate_meta: dict | None = None,
    extra: dict | None = None,
) -> dict:
    candidate_confidence = confidence.index_select(0, indices)
    result = {
        "variant": name,
        "raw_score": score.reshape(1, GRID, GRID).float().contiguous(),
        "minmax_score": minmax(score).reshape(1, GRID, GRID),
        "background_indices": indices.long().contiguous(),
        "background_confidence": candidate_confidence.float().contiguous(),
        "normalized_weights": model.weights.float().contiguous(),
        "selected_rank": int(model.rank),
        "retained_variance_ratio": float(model.retained_variance_ratio),
        "orthonormal_error": float(model.orthonormal_error),
        "effective_sample_size": float(model.effective_sample_size),
        "weight_concentration_warning": bool(model.weight_concentration_warning),
        "pca_weighting": pca_weighting,
        "num_queries": NUM_PATCHES,
        "background_scores_preserved": True,
        **(candidate_meta or candidate_diagnostics(indices, features, model.singular_values)),
    }
    if extra:
        result.update(extra)
    if result["num_queries"] != NUM_PATCHES or tuple(result["raw_score"].shape) != (1, GRID, GRID):
        raise RuntimeError(f"{name} did not score all patches")
    return result


def _generate(task: dict) -> dict:
    assert _SETTINGS is not None
    source, feature_payload, dabe, features, connectivity, anchor = _load_source(task)
    source_results = source.get("results", {})
    if "r8" not in source_results or "current" not in source_results:
        raise KeyError("formal rank cache must contain r8 and current")
    source_r8 = source_results["r8"]
    full_indices = source_r8["background_indices"].detach().cpu().long().reshape(-1)
    anchor_indices = torch.where(anchor > .5)[0]
    if not torch.equal(full_indices, anchor_indices):
        raise RuntimeError("B0 Full-BC indices differ from cached bg_anchor_37")
    background_count = int(full_indices.numel())
    if background_count <= int(_SETTINGS["rank"]):
        raise RuntimeError("B0 candidate count cannot support rank 8")

    requested = set(_SETTINGS["variants"])
    results: dict[str, dict] = {}
    variant_failures: dict[str, dict] = {}
    b0_model = fit_fixed_pca(features.index_select(0, full_indices), rank=_SETTINGS["rank"])
    b0_candidate_meta = candidate_diagnostics(full_indices, features, b0_model.singular_values)
    b0_raw = _tensor(source_r8, "absolute_raw", (1, GRID, GRID), Path(task["source_path"])).reshape(-1)
    b0_recomputed = euclidean_residual_score(features, b0_model)
    b0_source_error = float((b0_raw - b0_recomputed).abs().max())
    current_raw = _tensor(source_results["current"], "absolute_raw", (1, GRID, GRID), Path(task["source_path"]))
    b0_current_error = float((b0_raw.reshape(1, GRID, GRID) - current_raw).abs().max())
    if "b0" in requested:
        results["b0"] = _pca_result(
            "b0", b0_raw, full_indices, connectivity, features, b0_model,
            pca_weighting="equal",
            candidate_meta=b0_candidate_meta,
            extra={
                "background_discovery": "current_cumulative_path_full_bc",
                "seed_mode": "hard_two_ring_implicitly_forced_by_zero_path_cost",
                "candidate_selection": "current_full_bc",
                "residual_mode": "euclidean_orthogonal",
                "source_r8_recompute_max_abs_error": b0_source_error,
                "source_current_max_abs_error": b0_current_error,
            },
        )

    rgb = load_rgb_grid(task["image_path"])
    graph = build_local_graph(
        features,
        rgb,
        sigma_f=_SETTINGS["sigma_f"],
        sigma_c=_SETTINGS["sigma_c"],
        sigma_e=_SETTINGS["sigma_e"],
    )
    hard_seed = hard_boundary_seed()
    hard_rw = random_walk_with_restart(
        graph, hard_seed, alpha=_SETTINGS["alpha"],
        tolerance=_SETTINGS["rw_tolerance"], max_iterations=_SETTINGS["rw_max_iterations"],
    )
    soft = soft_boundary_seed(graph, features, top_k=_SETTINGS["soft_seed_topk"])
    soft_rw = random_walk_with_restart(
        graph, soft.seed, alpha=_SETTINGS["alpha"],
        tolerance=_SETTINGS["rw_tolerance"], max_iterations=_SETTINGS["rw_max_iterations"],
    )
    hard_indices = top_confidence_indices(hard_rw.confidence, background_count)
    soft_top_indices = top_confidence_indices(soft_rw.confidence, background_count)
    coreset = confidence_diversity_coreset(soft_rw.confidence, features, background_count)

    shared = {
        "edge_scale": float(graph.edge_scale),
        "edge_count_directed": int(graph.source.numel()),
        "transition_row_sum_error": float(graph.row_sum_error),
        "hard_seed_map": hard_seed.reshape(1, GRID, GRID),
        "hard_random_walk_confidence": hard_rw.confidence.reshape(1, GRID, GRID),
        "hard_random_walk": _rw_diagnostics(hard_rw, graph, hard_seed),
        "soft_seed_map": soft.seed.reshape(1, GRID, GRID),
        "inward_consistency": soft.inward_consistency.reshape(1, GRID, GRID),
        "global_dino_density": soft.global_dino_density.reshape(1, GRID, GRID),
        "inward_rank": soft.inward_rank.reshape(1, GRID, GRID),
        "density_rank": soft.density_rank.reshape(1, GRID, GRID),
        "soft_random_walk_confidence": soft_rw.confidence.reshape(1, GRID, GRID),
        "soft_random_walk": _rw_diagnostics(soft_rw, graph, soft.seed),
        "hard_top_confidence_indices": hard_indices,
        "soft_top_confidence_indices": soft_top_indices,
        "coreset_pool_indices": coreset.pool_indices,
        "coreset_indices": coreset.indices,
        "current_full_bc_indices": full_indices,
    }

    try:
        b1_model = fit_fixed_pca(features.index_select(0, hard_indices), rank=_SETTINGS["rank"])
        b1_candidate_meta = candidate_diagnostics(hard_indices, features, b1_model.singular_values)
        if "b1" in requested:
            results["b1"] = _pca_result(
                "b1", euclidean_residual_score(features, b1_model), hard_indices,
                hard_rw.confidence, features, b1_model, pca_weighting="equal",
                candidate_meta=b1_candidate_meta,
                extra={"background_discovery": "random_walk", "seed_mode": "hard_boundary", "candidate_selection": "top_confidence", "residual_mode": "euclidean_orthogonal"},
            )
    except Exception as error:
        variant_failures["b1"] = {"error": repr(error), "traceback": traceback.format_exc()}

    try:
        b2_model = fit_fixed_pca(features.index_select(0, soft_top_indices), rank=_SETTINGS["rank"])
        b2_candidate_meta = candidate_diagnostics(soft_top_indices, features, b2_model.singular_values)
        if "b2" in requested:
            results["b2"] = _pca_result(
                "b2", euclidean_residual_score(features, b2_model), soft_top_indices,
                soft_rw.confidence, features, b2_model, pca_weighting="equal",
                candidate_meta=b2_candidate_meta,
                extra={"background_discovery": "random_walk", "seed_mode": "soft_boundary", "candidate_selection": "top_confidence", "residual_mode": "euclidean_orthogonal"},
            )
    except Exception as error:
        variant_failures["b2"] = {"error": repr(error), "traceback": traceback.format_exc()}

    b3_model = None
    try:
        coreset_background = features.index_select(0, coreset.indices)
        b3_model = fit_fixed_pca(coreset_background, rank=_SETTINGS["rank"])
        b3_candidate_meta = candidate_diagnostics(coreset.indices, features, b3_model.singular_values)
        if "b3" in requested:
            results["b3"] = _pca_result(
                "b3", euclidean_residual_score(features, b3_model), coreset.indices,
                soft_rw.confidence, features, b3_model, pca_weighting="equal",
                candidate_meta=b3_candidate_meta,
                extra={"background_discovery": "random_walk", "seed_mode": "soft_boundary", "candidate_selection": "confidence_diversity", "candidate_pool_size": int(coreset.pool_indices.numel()), "residual_mode": "euclidean_orthogonal"},
            )
    except Exception as error:
        variant_failures["b3"] = {"error": repr(error), "traceback": traceback.format_exc()}

    b4_model = None
    if b3_model is not None:
        try:
            candidate_confidence = soft_rw.confidence.index_select(0, coreset.indices)
            weights = confidence_rank_weights(candidate_confidence)
            coreset_background = features.index_select(0, coreset.indices)
            b4_model = fit_fixed_pca(coreset_background, rank=_SETTINGS["rank"], weights=weights)
            angles = principal_angles_degrees(b3_model, b4_model)
            b4_extra = {
                "background_discovery": "random_walk", "seed_mode": "soft_boundary",
                "candidate_selection": "confidence_diversity", "candidate_pool_size": int(coreset.pool_indices.numel()),
                "residual_mode": "euclidean_orthogonal", "mean_shift": float(torch.linalg.vector_norm(b4_model.mean - b3_model.mean)),
                "principal_angles_vs_equal_pca": angles,
            }
            if "b4" in requested:
                results["b4"] = _pca_result(
                    "b4", euclidean_residual_score(features, b4_model), coreset.indices,
                    soft_rw.confidence, features, b4_model, pca_weighting="confidence_rank", extra=b4_extra,
                    candidate_meta=b3_candidate_meta,
                )
        except Exception as error:
            variant_failures["b4"] = {"error": repr(error), "traceback": traceback.format_exc()}

    if b4_model is not None:
        try:
            coreset_background = features.index_select(0, coreset.indices)
            swor = shrinkage_whitened_score(features, coreset_background, b4_model)
            if "b5" in requested:
                results["b5"] = _pca_result(
                    "b5", swor.score, coreset.indices, soft_rw.confidence, features,
                    b4_model, pca_weighting="confidence_rank",
                    candidate_meta=b3_candidate_meta,
                    extra={
                        "background_discovery": "random_walk", "seed_mode": "soft_boundary",
                        "candidate_selection": "confidence_diversity", "candidate_pool_size": int(coreset.pool_indices.numel()),
                        "residual_mode": "shrinkage_whitened_orthogonal",
                        "mean_shift": b4_extra["mean_shift"],
                        "principal_angles_vs_equal_pca": b4_extra["principal_angles_vs_equal_pca"],
                        "ledoit_wolf_shrinkage": swor.shrinkage,
                        "covariance_condition_number": swor.covariance_condition_number,
                        "precision_condition_number": swor.precision_condition_number,
                        "swor_min": swor.minimum, "swor_max": swor.maximum, "swor_mean": swor.mean,
                        "swor_valid": swor.valid,
                    },
                )
        except Exception as error:
            variant_failures["b5"] = {"error": repr(error), "traceback": traceback.format_exc(), "swor_valid": False}

    missing = sorted(requested - set(results))
    for name in missing:
        variant_failures.setdefault(name, {"error": "dependency_failed_or_variant_not_generated"})
    return {
        "results": results,
        "shared_diagnostics": shared,
        "variant_failures": variant_failures,
        "source_r8_recompute_max_abs_error": b0_source_error,
        "source_current_max_abs_error": b0_current_error,
        "background_count": background_count,
        "source_feature_path": source["source_feature_path"],
        "source_dabe_path": source["source_dabe_path"],
        "original_image_size": tuple(feature_payload["original_size"]),
    }


def _valid_existing(path: Path, fingerprint: str, requested: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        available = set(payload.get("results", {})) | set(payload.get("variant_failures", {}))
        return (
            isinstance(payload, dict)
            and payload.get("gbsp_v2_version") == VERSION
            and payload.get("settings_fingerprint") == fingerprint
            and requested <= available
            and all(
                torch.is_tensor(item.get("raw_score"))
                and tuple(item["raw_score"].shape) == (1, GRID, GRID)
                and bool(torch.isfinite(item["raw_score"]).all())
                and int(item.get("num_queries", 0)) == NUM_PATCHES
                for item in payload.get("results", {}).values()
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _generate_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output = Path(task["output_path"])
    if _valid_existing(output, _SETTINGS["fingerprint"], set(_SETTINGS["variants"])):
        payload = torch_load(output, map_location="cpu")
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output),
            "skipped": True, "variant_failures": payload.get("variant_failures", {}),
            "source_r8_recompute_max_abs_error": payload["source_r8_recompute_max_abs_error"],
            "source_current_max_abs_error": payload["source_current_max_abs_error"],
        }
    started = time.perf_counter()
    try:
        generated = _generate(task)
        payload = {
            "gbsp_v2_version": VERSION,
            "dataset": task["dataset"], "stem": task["stem"],
            "settings": {key: value for key, value in _SETTINGS.items() if key != "fingerprint"},
            "settings_fingerprint": _SETTINGS["fingerprint"],
            **generated,
            "patch_grid_size": (GRID, GRID),
            "image_path": task["image_path"], "gt_path": task["gt_path"],
            "source_gbsp_path": task["source_path"],
            "runtime_seconds": time.perf_counter() - started,
        }
        _atomic_save(payload, output)
        return {
            "dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output),
            "skipped": False, "runtime_seconds": payload["runtime_seconds"],
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "variant_failures": generated["variant_failures"],
            "source_r8_recompute_max_abs_error": generated["source_r8_recompute_max_abs_error"],
            "source_current_max_abs_error": generated["source_current_max_abs_error"],
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


def _configuration_audit(cfg, source_root: Path, output_root: Path) -> None:
    audit = {
        "created_at": _now(), "status": "predeclared_and_frozen_before_formal_run",
        "CURRENT_FEATURE_SOURCE": "DINOv1 ViT-S/8 last-block attention-key patch tokens",
        "CURRENT_FEATURE_NORMALIZATION": "per-patch L2 normalization after cached key projection",
        "CURRENT_EDGE_COST_FORMULA": "(1-cosine)/0.10 + squared_RGB/0.05 + max_endpoint_Sobel/0.30 on symmetric 8-neighbour graph",
        "CURRENT_PATH_MODE": "B0 cumulative Dijkstra; B1-B5 random walk with restart alpha=0.85",
        "CURRENT_BACKGROUND_COUNT_RULE": "per-image N_b exactly matched to B0 Full BC (formal distribution includes ties)",
        "CURRENT_PCA_RANK": 8,
        "CURRENT_PCA_WEIGHTING": "B0-B3 equal; B4-B5 confidence average-rank weights w=2r",
        "CURRENT_RESIDUAL_FORMULA": "B0-B4 absolute squared orthogonal residual; B5 Ledoit-Wolf SWOR/(384-8)",
        "frozen": {
            "grid": GRID, "num_patches": NUM_PATCHES, "boundary_width": 2,
            "boundary_count": BOUNDARY_COUNT, "alpha": cfg.GBSP_V2_ALPHA,
            "rw_tolerance": cfg.GBSP_V2_RW_TOLERANCE,
            "rw_max_iterations": cfg.GBSP_V2_RW_MAX_ITERATIONS,
            "soft_seed_topk": cfg.GBSP_V2_SOFT_SEED_TOPK,
            "pca_rank": cfg.GBSP_V2_RANK,
            "variants": list(cfg.GBSP_V2_VARIANTS),
        },
        "source_root": str(source_root), "output_root": str(output_root),
        "baseline_reference": dict(cfg.GBSP_V2_BASELINE_REFERENCE),
        "baseline_tolerance": cfg.GBSP_V2_BASELINE_TOLERANCE,
        "all_1369_patches_are_queried": True,
        "background_response_overwrite": False,
        "gt_used_during_generation": False,
        "feature_extraction_used": False,
        "training_used": False,
    }
    write_json(output_root / "configuration_audit.json", audit)


def build(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("GBSP-V2 currently supports the formal test split only")
    cfg = load_config(_resolve(args.config))
    source_root = _resolve(args.gbsp_root or cfg.GBSP_V2_SOURCE_ROOT)
    out_root = _resolve(args.out_root)
    if out_root == MAIN_ROOT or MAIN_ROOT in out_root.parents:
        raise ValueError("output must stay outside the main code directory")
    rows = _manifest(source_root / "manifest_test.jsonl")
    variants = tuple(args.variants or cfg.GBSP_V2_VARIANTS)
    invalid = sorted(set(variants) - set(VARIANTS))
    if invalid:
        raise ValueError(f"invalid GBSP-V2 variants: {invalid}")
    settings = {
        "version": VERSION, "variants": variants,
        "rank": int(cfg.GBSP_V2_RANK), "alpha": float(cfg.GBSP_V2_ALPHA),
        "rw_tolerance": float(cfg.GBSP_V2_RW_TOLERANCE),
        "rw_max_iterations": int(cfg.GBSP_V2_RW_MAX_ITERATIONS),
        "soft_seed_topk": int(cfg.GBSP_V2_SOFT_SEED_TOPK),
        "sigma_f": float(cfg.GBSP_V2_EDGE_SIGMA_F),
        "sigma_c": float(cfg.GBSP_V2_EDGE_SIGMA_C),
        "sigma_e": float(cfg.GBSP_V2_EDGE_SIGMA_E),
        "gt_used_during_generation": False, "feature_extraction_used": False,
        "training_used": False,
    }
    settings["fingerprint"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if args.sample_list:
        identities = _sample_ids(_resolve(args.sample_list))
        rows = [row for row in rows if (row["dataset"], row["stem"]) in identities]
        missing = identities - {(row["dataset"], row["stem"]) for row in rows}
        if missing:
            raise KeyError(f"sample-list identities missing: {sorted(missing)[:3]}")
    if args.max_samples >= 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no samples selected")
    if not args.sample_list and args.max_samples < 0:
        expected = dict(cfg.GBSP_V2_EXPECTED_COUNTS)
        observed = Counter(row["dataset"] for row in rows)
        if len(rows) != sum(expected.values()) or dict(observed) != expected:
            raise RuntimeError(f"formal 6473 count mismatch: {dict(observed)}")

    tasks = []
    for row in rows:
        source = torch_load(Path(row["cache_path"]), map_location="cpu")
        tasks.append({
            "dataset": row["dataset"], "stem": row["stem"], "source_path": row["cache_path"],
            "image_path": source["image_path"], "gt_path": source["gt_path"],
            "output_path": str(out_root / "test" / row["dataset"] / f"{row['stem']}.pt"),
        })
    out_root.mkdir(parents=True, exist_ok=True)
    _configuration_audit(cfg, source_root, out_root)
    write_json(out_root / "run_config.json", {
        "created_at": _now(), "config": str(_resolve(args.config)),
        "source_root": str(source_root), "output_root": str(out_root),
        "sample_count": len(tasks), "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
        "settings": settings,
    })
    if args.dry_run:
        print(json.dumps({"sample_count": len(tasks), "settings": settings}, ensure_ascii=False, indent=2))
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
                print(f"[{_now()}] GBSP-V2 generation: {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if "error" in result]
    valid = {(result["dataset"], result["stem"]): result for result in results if "error" not in result}
    per_variant_failures = [
        {"dataset": result["dataset"], "stem": result["stem"], "variant": variant, **detail}
        for result in valid.values() for variant, detail in result.get("variant_failures", {}).items()
    ]
    write_json(out_root / "generation_failures.json", failures)
    write_json(out_root / "variant_numerical_failures.json", per_variant_failures)
    manifest = []
    for task in tasks:
        key = (task["dataset"], task["stem"])
        if key in valid:
            manifest.append({
                "dataset": task["dataset"], "stem": task["stem"],
                "cache_path": task["output_path"], "image_path": task["image_path"],
                "gt_path": task["gt_path"], "variants": list(variants),
                "settings_fingerprint": settings["fingerprint"],
            })
    write_jsonl(out_root / "manifest_test.jsonl", manifest)
    valid_rows = list(valid.values())
    summary = {
        "requested": len(tasks), "valid_cache_items": len(valid_rows),
        "generation_failed": len(failures), "variant_failed": len(per_variant_failures),
        "resumed": sum(bool(row.get("skipped")) for row in valid_rows),
        "wall_seconds": time.perf_counter() - started,
        "max_b0_r8_recompute_error": max((float(row["source_r8_recompute_max_abs_error"]) for row in valid_rows), default=None),
        "max_b0_current_difference": max((float(row["source_current_max_abs_error"]) for row in valid_rows), default=None),
        "rare_failures_are_recorded_without_restarting_the_run": True,
    }
    write_json(out_root / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.strict_failures and (failures or per_variant_failures):
        raise RuntimeError(f"generation failures={len(failures)}, variant failures={len(per_variant_failures)}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp-root", "--gbsp_root", dest="gbsp_root")
    value.add_argument("--sample-list", "--sample_list", dest="sample_list")
    value.add_argument("--split", default="test")
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--variants", nargs="+", choices=VARIANTS)
    value.add_argument("--save-diagnostics", "--save_diagnostics", action="store_true", help="compatibility flag; diagnostics are always saved")
    value.add_argument("--out-root", "--out_root", dest="out_root", required=True)
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--strict-failures", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


if __name__ == "__main__":
    build(parser().parse_args())
