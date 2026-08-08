#!/usr/bin/env python3
"""Generate GBSP V4 masks from immutable GBSP/feature caches without DINO forward."""

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
from models.gbsp_threshold_v4 import (  # noqa: E402
    HierarchicalUpperTailOtsu,
    LocalResidualContrast,
    OrthogonalResidualCoherence,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_threshold_v4.py"
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
_SETTINGS: dict | None = None


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _manifest_path(root: Path) -> Path:
    for path in (root / "manifest_test.jsonl", root / "ablations/M1_pilot200/manifest_test.jsonl"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"manifest_test.jsonl missing below {root}")


def _manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = read_jsonl(path); mapping = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in mapping or not Path(str(row.get("cache_path", ""))).is_file():
            raise RuntimeError(f"invalid manifest row {path}:{line}: {key}")
        mapping[key] = row
    return rows, mapping


def _sample_keys(path: Path) -> list[tuple[str, str]]:
    keys = []
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.replace("/", "\t", 1).split()
        if len(parts) != 2:
            raise ValueError(f"invalid sample identity {path}:{line}")
        keys.append((parts[0], parts[1]))
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate sample identities")
    return keys


def _balanced(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0 or limit >= len(rows):
        return rows
    grouped = {name: [] for name in EXPECTED}
    for row in rows: grouped[str(row["dataset"])].append(row)
    output, index = [], 0
    while len(output) < limit:
        for name in EXPECTED:
            if index < len(grouped[name]): output.append(grouped[name][index])
            if len(output) == limit: break
        index += 1
    return output


def _map(payload: dict, fields: tuple[str, ...], path: Path) -> torch.Tensor:
    for field in fields:
        value = payload.get(field)
        if torch.is_tensor(value) and tuple(value.shape) == (1, 37, 37):
            value = value.detach().cpu().float().contiguous()
            if bool(torch.isfinite(value).all()): return value
    raise ValueError(f"missing/nonfinite {fields}: {path}")


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def _replay_residual_vectors(source: dict, source_path: Path) -> tuple[torch.Tensor, dict]:
    """Recover the omitted cached PCA basis and prove exact score equivalence."""
    feature_path = Path(str(source.get("source_feature_path", "")))
    if not feature_path.is_file(): raise FileNotFoundError(feature_path)
    feature_payload = torch_load(feature_path, map_location="cpu")
    feature = feature_payload.get("tensor")
    if not torch.is_tensor(feature) or tuple(feature.shape) != (384, 37, 37):
        raise ValueError(f"invalid feature tensor: {feature_path}")
    query = F.normalize(
        feature.detach().cpu().float().permute(1, 2, 0).reshape(1369, 384).contiguous(),
        p=2, dim=1,
    )
    bc = source["background_indices"].detach().cpu().long()
    background = query.index_select(0, bc)
    mean = background.mean(dim=0)
    centered_background = background - mean
    _, singular, right = torch.linalg.svd(centered_background, full_matrices=False)
    rank = int(source["selected_ranks"].reshape(-1)[0])
    basis = right[:rank].t().contiguous()
    centered = query - mean
    residual = centered - (centered @ basis) @ basis.t()
    replay_raw = residual.square().sum(dim=1).reshape(1, 37, 37)
    cached_mean = source["subspace_means"].reshape(-1, 384)[0].float()
    cached_singular = source["singular_values"][0].float()
    cached_raw = _map(source, ("absolute_raw",), source_path)
    mean_error = float((mean - cached_mean).abs().max())
    singular_error = float((singular - cached_singular).abs().max())
    raw_error = float((replay_raw - cached_raw).abs().max())
    raw_relative_error = raw_error / max(float(cached_raw.abs().max()), 1e-8)
    if mean_error > 2e-6 or singular_error > 2e-4 or raw_error > 2e-5 or raw_relative_error > 2e-4:
        raise RuntimeError(
            f"cached PCA replay mismatch mean={mean_error} singular={singular_error} "
            f"raw={raw_error} relative={raw_relative_error}"
        )
    return residual.contiguous(), {
        "pca_basis_replayed": True, "dino_forward_used": False,
        "query_feature_shape": list(query.shape), "pca_rank": rank,
        "mean_max_abs_error": mean_error, "singular_max_abs_error": singular_error,
        "raw_residual_max_abs_error": raw_error, "raw_residual_max_relative_error": raw_relative_error,
        "basis_orthonormal_max_abs_error": float(
            (basis.t() @ basis - torch.eye(rank)).abs().max()
        ),
    }


def _run_method(method: str, score: torch.Tensor, raw: torch.Tensor, bc: torch.Tensor, source: dict, source_path: Path):
    if method == "huto_core": return HierarchicalUpperTailOtsu("core").apply(score, bc), {}
    if method == "huto_mid_h": return HierarchicalUpperTailOtsu("mid_h").apply(score, bc), {}
    if method == "huto_med_h": return HierarchicalUpperTailOtsu("med_h").apply(score, bc), {}
    if method == "lrc_h": return LocalResidualContrast().apply(score, bc), {}
    if method == "orc_h":
        residual, replay = _replay_residual_vectors(source, source_path)
        return OrthogonalResidualCoherence().apply(score, raw, residual, bc), replay
    raise ValueError(method)


def _serialize(method: str, family: str, result, bc: torch.Tensor, replay: dict) -> dict:
    mask = result.mask.float().contiguous()
    diagnostics = {**result.diagnostics, **replay}
    return {
        "method": method, "method_family": family, "mask_37": mask,
        "maps_37": {key: value.float().contiguous() for key, value in result.maps.items()},
        "threshold_high": result.threshold_high, "threshold_low": result.threshold_low,
        "foreground_area": float(mask.mean()),
        "bc_final_ratio": float(mask.reshape(-1).index_select(0, bc).mean()),
        "empty_mask": bool(float(mask.mean()) == 0.0),
        "area_over_50pct": bool(float(mask.mean()) > 0.5),
        "numerical_failure": bool(result.numerical_failure), "diagnostics": diagnostics,
    }


def _process_one(task: dict) -> dict:
    assert _SETTINGS is not None
    output_path = Path(task["output_path"]); started = time.perf_counter()
    try:
        source_path = Path(task["source_path"]); source = torch_load(source_path, map_location="cpu")
        if (str(source.get("dataset")), str(source.get("stem"))) != (task["dataset"], task["stem"]):
            raise RuntimeError("GBSP cache identity mismatch")
        score = _map(source, ("absolute_minmax",), source_path)
        raw = _map(source, ("absolute_raw",), source_path)
        bc = source.get("background_indices")
        if not torch.is_tensor(bc) or bc.ndim != 1 or bc.numel() == 0: raise ValueError("BC missing")
        bc = bc.detach().cpu().long().contiguous()
        results = {}
        for method in _SETTINGS["methods"]:
            try:
                result, replay = _run_method(method, score, raw, bc, source, source_path)
                results[method] = _serialize(method, _SETTINGS["families"][method], result, bc, replay)
            except Exception as error:
                results[method] = {
                    "method": method, "method_family": _SETTINGS["families"][method],
                    "mask_37": torch.zeros(1, 37, 37), "maps_37": {},
                    "threshold_high": None, "threshold_low": None, "foreground_area": 0.0,
                    "bc_final_ratio": 0.0, "empty_mask": True, "area_over_50pct": False,
                    "numerical_failure": True,
                    "diagnostics": {"failure_reason": "uncaught_method_exception", "error": repr(error), "traceback": traceback.format_exc()},
                }
        size = source.get("original_image_size")
        payload = {
            "version": _SETTINGS["version"], "dataset": task["dataset"], "stem": task["stem"],
            "image_id": f"{task['dataset']}/{task['stem']}",
            "image_path": task.get("image_path") or source.get("image_path", ""),
            "gt_path": task.get("gt_path") or source.get("gt_path", ""),
            "image_size": tuple(size), "patch_grid_size": (37, 37),
            "raw_residual": raw, "minmax_residual": score, "background_indices": bc,
            "background_candidate_count": int(bc.numel()), "pca_rank": int(source["selected_ranks"][0]),
            "source_gbsp_path": str(source_path.resolve()),
            "source_feature_id": str(source.get("source_feature_path", "")),
            "results": results, "settings": dict(_SETTINGS), "settings_fingerprint": _SETTINGS["fingerprint"],
            "gt_used_for_generation": False, "r1_used_for_generation": False,
            "fixed_058_used_for_generation": False, "target_area_used": False,
            "fixed_topk_used": False, "dino_forward_used": False, "pca_changed": False,
            "morphology_used": False,
            "runtime": {"total_seconds": time.perf_counter() - started,
                        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0},
        }
        _atomic_save(payload, output_path)
        return {"dataset": task["dataset"], "stem": task["stem"], "cache_path": str(output_path),
                "numerical_failures": sum(int(row["numerical_failure"]) for row in results.values()),
                "total_seconds": payload["runtime"]["total_seconds"]}
    except Exception as error:
        return {"dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
                "error": repr(error), "traceback": traceback.format_exc()}


def _init(settings: dict, threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings; torch.set_num_threads(threads)


def build(args: argparse.Namespace) -> None:
    cfg = load_config(_resolve(args.config))
    methods = list(args.methods or cfg.GBSP_V4_METHODS)
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(cfg.GBSP_V4_METHODS):
        raise ValueError(f"invalid methods: {methods}")
    frozen = None
    if args.frozen_config:
        frozen = json.loads(_resolve(args.frozen_config).read_text(encoding="utf-8"))
        if frozen.get("schema") != "gbsp_threshold_v4_frozen_config": raise ValueError("bad frozen config")
        if set(methods) - set(frozen.get("selected_methods", [])): raise ValueError("method not selected")
    settings = {
        "version": cfg.GBSP_V4_VERSION, "methods": methods,
        "families": {method: cfg.GBSP_V4_FAMILIES[method] for method in methods},
        "multi_otsu_classes": 3, "multi_otsu_nbins": 256, "upper_tail_otsu_nbins": 256,
        "connectivity": 8, "orc_neighborhood": 8, "lrc_windows": [3, 7],
        "rank_tie_method": "average", "epsilon": cfg.GBSP_V4_EPS,
        "save_diagnostics": bool(args.save_diagnostics), "diagnostic200": bool(args.diagnostic200),
        **dict(cfg.GBSP_V4_INDEPENDENCE),
    }
    settings["fingerprint"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    root = _resolve(args.gbsp_root or cfg.GBSP_V4_TEST_ROOT)
    out = _resolve(args.out_root or Path(cfg.GBSP_V4_OUTPUT_ROOT) / "full6473")
    if out == MAIN_ROOT or MAIN_ROOT in out.parents: raise ValueError("output must be outside repository")
    manifest_path = _manifest_path(root); rows, mapping = _manifest(manifest_path)
    if args.sample_list:
        keys = _sample_keys(_resolve(args.sample_list)); missing = [key for key in keys if key not in mapping]
        if missing: raise KeyError(missing[:5])
        rows = [mapping[key] for key in keys]
    if args.max_samples >= 0: rows = _balanced(rows, args.max_samples) if not args.sample_list else rows[:args.max_samples]
    if not rows: raise RuntimeError("no samples")
    counts = dict(Counter(str(row["dataset"]) for row in rows))
    if len(rows) == 20 and frozen is None and set(methods) != set(cfg.GBSP_V4_METHODS):
        raise RuntimeError("A1 requires all five methods")
    if len(rows) == 200:
        if args.diagnostic200:
            if set(methods) != set(cfg.GBSP_V4_METHODS):
                raise RuntimeError("diagnostic200 must run all five frozen V4 methods")
        elif frozen is None or not 1 <= len(methods) <= 2:
            raise RuntimeError("formal Pilot200 requires one/two frozen methods")
    if len(rows) == sum(EXPECTED.values()) and (frozen is None or len(methods) != 1):
        raise RuntimeError("full requires one frozen method")
    if not args.dry_run:
        audit_path = _resolve(cfg.GBSP_V4_BASELINE_AUDIT)
        if not audit_path.is_file() or json.loads(audit_path.read_text()).get("baseline_reproduction_status") != "PASS":
            raise RuntimeError("V4 A0 baseline reproduction has not passed")
    tasks = [{"dataset": str(row["dataset"]), "stem": str(row["stem"]), "source_path": row["cache_path"],
              "image_path": row.get("image_path", ""), "gt_path": row.get("gt_path", ""),
              "output_path": str(out / "test" / str(row["dataset"]) / f"{row['stem']}.pt")} for row in rows]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {"created_at": _now(), "root": str(root), "manifest": str(manifest_path),
               "out_root": str(out), "sample_list": str(_resolve(args.sample_list)) if args.sample_list else None,
               "num_selected": len(tasks), "dataset_counts": counts, "frozen_config": args.frozen_config,
               "settings": settings})
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "num_selected": len(tasks), "methods": methods}, indent=2)); return
    started = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(settings, args.torch_threads)) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks): print(f"V4 cache {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]; valid = [row for row in results if "error" not in row]
    keys = {(row["dataset"], row["stem"]) for row in valid}
    manifests = [{"dataset": task["dataset"], "stem": task["stem"], "cache_path": task["output_path"],
                  "image_path": task["image_path"], "gt_path": task["gt_path"], "source_gbsp_path": task["source_path"]}
                 for task in tasks if (task["dataset"], task["stem"]) in keys]
    write_json(out / "generation_failures.json", failures); write_jsonl(out / "manifest_test.jsonl", manifests)
    summary = {"num_requested": len(tasks), "num_valid": len(valid), "num_failed": len(failures),
               "dataset_counts": dict(Counter(row["dataset"] for row in manifests)),
               "numerical_failure_total": sum(row["numerical_failures"] for row in valid),
               "mean_seconds": sum(row["total_seconds"] for row in valid) / max(len(valid), 1),
               "wall_seconds": time.perf_counter() - started, **dict(cfg.GBSP_V4_INDEPENDENCE)}
    write_json(out / "performance_summary.json", summary); print(json.dumps(summary, indent=2))
    if failures and args.failure_policy == "strict": raise RuntimeError(f"{len(failures)} generation failures")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG)); value.add_argument("--gbsp_root")
    value.add_argument("--sample_list"); value.add_argument("--split", default="test", choices=("test",))
    value.add_argument("--max_samples", type=int, default=-1); value.add_argument("--methods", nargs="+")
    value.add_argument("--method", action="append", dest="single"); value.add_argument("--frozen_config")
    value.add_argument("--save_diagnostics", action="store_true"); value.add_argument("--out_root")
    value.add_argument("--workers", type=int, default=4); value.add_argument("--torch_threads", type=int, default=1)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    value.add_argument("--diagnostic200", action="store_true")
    value.add_argument("--dry-run", action="store_true"); return value


if __name__ == "__main__":
    args = parser().parse_args()
    if args.single: args.methods = (args.methods or []) + args.single
    build(args)
