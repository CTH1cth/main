#!/usr/bin/env python3
"""Generate the registered R0--R5 CVBR-weighted rank-8 GBSP caches.

The script consumes existing DINO/Full-BC/CVBR caches.  It never extracts an
encoder feature, reads GT, changes graph connectivity or rebuilds candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402
from models.gbsp_core_variants import NUM_PATCHES, score_all_patches  # noqa: E402
from models.gbsp_cvbr_weighted_pca import (  # noqa: E402
    build_cvbr_soft_weights,
    fit_weighted_affine_subspace,
    projector_distance,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_cvbr_weighted_pca.py"
VERSION = "gbsp_cvbr_weighted_pca_v1"
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


def _index_manifest(path: Path) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    index = {}
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        if not all(key) or key in index:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}: {key}")
        cache = Path(row.get("cache_path", ""))
        if not cache.is_file():
            raise FileNotFoundError(cache)
        index[key] = row
    return rows, index


def _sample_ids(path: Path) -> set[tuple[str, str]]:
    output = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"sample list needs dataset<TAB>stem: {line!r}")
        output.add((parts[0], parts[1]))
    return output


def _tensor(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


def _minmax(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    value = value.detach().cpu().float()
    low, high = value.min(), value.max()
    if float(high - low) <= eps:
        return torch.zeros_like(value)
    return ((value - low) / (high - low + eps)).clamp(0, 1).contiguous()


def _valid_existing(path: Path, fingerprint: str, variant: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        return (
            isinstance(payload, dict)
            and payload.get("version") == VERSION
            and payload.get("settings_fingerprint") == fingerprint
            and payload.get("variant") == variant
            and torch.is_tensor(payload.get("absolute_raw"))
            and tuple(payload["absolute_raw"].shape) == (1, 37, 37)
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _load_inputs(task: dict) -> tuple[dict, dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_path, cvbr_path = Path(task["core_path"]), Path(task["cvbr_path"])
    core, cvbr = torch_load(core_path, map_location="cpu"), torch_load(cvbr_path, map_location="cpu")
    identity = (task["dataset"], task["stem"])
    for payload, path in ((core, core_path), (cvbr, cvbr_path)):
        if not isinstance(payload, dict) or (str(payload.get("dataset")), str(payload.get("stem"))) != identity:
            raise RuntimeError(f"cache identity mismatch: {path}")
    if core.get("gbsp_core_version") != "gbsp_core_optimization_v1":
        raise RuntimeError(f"unsupported core cache: {core_path}")
    if cvbr.get("cvbr_version") != _SETTINGS["cvbr_version"]:
        raise RuntimeError(f"unsupported CVBR cache: {cvbr_path}")

    source = core.get("results", {}).get("r8")
    if not isinstance(source, dict) or int(source.get("selected_rank", -1)) != 8:
        raise RuntimeError(f"fixed-r8 source missing: {core_path}")
    indices = source.get("background_indices")
    if not torch.is_tensor(indices):
        raise ValueError(f"Full-BC indices missing: {core_path}")
    indices = indices.detach().cpu().long().reshape(-1).contiguous()
    if indices.numel() < 10 or indices.numel() != torch.unique(indices).numel():
        raise ValueError(f"invalid Full-BC indices: {core_path}")
    anchor = _tensor(cvbr, "anchor_b0_37", (1, 37, 37), cvbr_path).reshape(-1) > .5
    if not torch.equal(indices, torch.where(anchor)[0]):
        raise RuntimeError(f"CVBR and GBSP Full-BC candidate sets differ: {identity}")

    feature_path = Path(core["source_feature_path"])
    feature_payload = torch_load(feature_path, map_location="cpu")
    feature = _tensor(feature_payload, "tensor", (384, 37, 37), feature_path)
    query = F.normalize(feature.permute(1, 2, 0).reshape(NUM_PATCHES, 384), p=2, dim=1)
    scores = _tensor(cvbr, _SETTINGS["score_field"], (1, 37, 37), cvbr_path).reshape(-1)
    valid_grid = _tensor(cvbr, _SETTINGS["valid_mask_field"], (1, 37, 37), cvbr_path).reshape(-1) > .5
    candidate_scores = scores.index_select(0, indices)
    candidate_valid = valid_grid.index_select(0, indices)
    if int(candidate_valid.sum()) != 136:
        raise RuntimeError(f"validated ring2 CVBR scope must cover 136 Full-BC candidates: {identity}")
    return core, cvbr, query.contiguous(), indices, candidate_scores, candidate_valid


def _generate_one(task: dict) -> dict:
    assert _SETTINGS is not None
    outputs = {name: Path(path) for name, path in task["output_paths"].items()}
    if all(_valid_existing(path, _SETTINGS["fingerprint"], name) for name, path in outputs.items()):
        first = torch_load(outputs["R0_uniform"], map_location="cpu")
        return {
            "dataset": task["dataset"], "stem": task["stem"], "skipped": True,
            "uniform_residual_max_abs": float(first["uniform_equivalence"]["residual_max_abs"]),
            "num_candidates": int(first["num_candidates"]),
            "num_cvbr_scored": int(first["num_cvbr_scored"]),
        }
    started = time.perf_counter()
    try:
        core, cvbr, query, indices, candidate_scores, candidate_valid = _load_inputs(task)
        background = query.index_select(0, indices)
        source = core["results"]["r8"]
        source_raw = _tensor(source, "absolute_raw", (1, 37, 37), Path(task["core_path"]))
        source_mean = _tensor(source, "mean", (384,), Path(task["core_path"]))
        source_basis = _tensor(source, "basis", (384, 8), Path(task["core_path"]))

        generated = {}
        uniform_equivalence = None
        vanilla_self_residual = None
        for name, specification in _SETTINGS["variants"].items():
            weight_result = build_cvbr_soft_weights(
                candidate_scores, candidate_valid,
                specification["top_frac"], specification["low_weight"],
            )
            weighted = fit_weighted_affine_subspace(background, weight_result.weights, rank=8)
            raw = score_all_patches(query, weighted.fit).reshape(1, 37, 37)
            candidate_self = raw.reshape(-1).index_select(0, indices).clamp_min(0).sqrt()
            if name == "R0_uniform":
                uniform_equivalence = {
                    "mean_max_abs": float((weighted.fit.mean - source_mean).abs().max()),
                    "projector_distance": projector_distance(weighted.fit.basis, source_basis),
                    "residual_max_abs": float((raw - source_raw).abs().max()),
                }
                if uniform_equivalence["residual_max_abs"] >= _SETTINGS["uniform_tolerance"]:
                    raise RuntimeError(f"uniform residual reproduction failed: {uniform_equivalence}")
                vanilla_self_residual = candidate_self
            generated[name] = (specification, weight_result, weighted, raw, candidate_self)

        assert uniform_equivalence is not None and vanilla_self_residual is not None
        for name, (specification, weight_result, weighted, raw, candidate_self) in generated.items():
            payload = {
                "version": VERSION,
                "dataset": task["dataset"], "stem": task["stem"],
                "variant": name, "cache_key": specification["cache_key"],
                "estimator": "uniform" if name == "R0_uniform" else "cvbr_weighted",
                "rank": 8,
                "cvbr_top_frac": float(specification["top_frac"]),
                "cvbr_low_weight": float(specification["low_weight"]),
                "cvbr_scope": _SETTINGS["cvbr_scope"],
                "absolute_raw": raw.float().contiguous(),
                "absolute_minmax": _minmax(raw),
                "mean": weighted.fit.mean,
                "basis": weighted.fit.basis,
                "singular_values": weighted.fit.singular_values,
                "retained_variance_ratio": float(weighted.fit.retained_variance_ratio),
                "orthonormal_error": float(weighted.fit.orthonormal_error),
                "background_indices": indices.to(torch.int16),
                "num_candidates": int(indices.numel()),
                "cvbr_score": candidate_scores,
                "cvbr_valid_mask": candidate_valid,
                "cvbr_percentile": weight_result.risk_percentile,
                "weights": weight_result.weights,
                "suppressed_mask": weight_result.suppressed_mask,
                "num_cvbr_scored": int(weight_result.num_valid),
                "num_suppressed": int(weight_result.num_suppressed),
                "effective_sample_size": float(weighted.effective_sample_size),
                "num_positive_weight": int(weighted.num_positive_weight),
                "vanilla_pca_self_residual": vanilla_self_residual,
                "weighted_pca_self_residual": candidate_self,
                "uniform_equivalence": uniform_equivalence,
                "settings_fingerprint": _SETTINGS["fingerprint"],
                "gt_used_for_generation": False,
                "feature_extraction_used": False,
                "candidate_selection_changed": False,
                "graph_changed": False,
                "source_core_path": task["core_path"],
                "source_cvbr_path": task["cvbr_path"],
                "source_feature_path": core["source_feature_path"],
                "image_path": core["image_path"], "gt_path": core["gt_path"],
                "original_image_size": core["original_image_size"],
            }
            _atomic_save(payload, outputs[name])
        return {
            "dataset": task["dataset"], "stem": task["stem"], "skipped": False,
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "uniform_residual_max_abs": uniform_equivalence["residual_max_abs"],
            "num_candidates": int(indices.numel()), "num_cvbr_scored": int(candidate_valid.sum()),
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(settings: dict, torch_threads: int) -> None:
    global _SETTINGS
    _SETTINGS = settings
    torch.set_num_threads(int(torch_threads))


def build(args: argparse.Namespace) -> None:
    cfg = load_config(_resolve(args.config))
    core_root = _resolve(args.gbsp_root or cfg.GBSP_CVBR_SOURCE_ROOT)
    cvbr_root = _resolve(args.cvbr_root or cfg.GBSP_CVBR_CACHE_ROOT)
    out_root = _resolve(args.out_root or cfg.GBSP_CVBR_OUTPUT_ROOT)
    if MAIN_ROOT == out_root or MAIN_ROOT in out_root.parents:
        raise ValueError("output must remain outside the main code tree")
    core_rows, core_index = _index_manifest(core_root / "manifest_test.jsonl")
    _, cvbr_index = _index_manifest(cvbr_root / "manifest_test.jsonl")
    if set(core_index) != set(cvbr_index):
        raise RuntimeError("GBSP and CVBR identity sets differ")

    rows = core_rows
    if args.sample_list:
        selected = _sample_ids(_resolve(args.sample_list))
        rows = [row for row in rows if (row["dataset"], row["stem"]) in selected]
        missing = selected - {(row["dataset"], row["stem"]) for row in rows}
        if missing:
            raise KeyError(f"sample identities missing: {sorted(missing)[:3]}")
    if args.max_samples >= 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("no samples selected")
    full_run = not args.sample_list and args.max_samples < 0
    if full_run:
        expected = dict(cfg.GBSP_CVBR_EXPECTED_COUNTS)
        observed = Counter(str(row["dataset"]) for row in rows)
        if len(rows) != sum(expected.values()) or dict(observed) != expected:
            raise RuntimeError(f"formal sample count mismatch: {dict(observed)}")

    variants = {name: dict(value) for name, value in cfg.GBSP_CVBR_VARIANTS.items()}
    settings = {
        "version": VERSION, "rank": int(cfg.GBSP_CVBR_RANK),
        "cvbr_version": str(cfg.GBSP_CVBR_VALIDATED_VERSION),
        "cvbr_scope": str(cfg.GBSP_CVBR_SCOPE),
        "score_field": str(cfg.GBSP_CVBR_SCORE_FIELD),
        "valid_mask_field": str(cfg.GBSP_CVBR_VALID_MASK_FIELD),
        "uniform_tolerance": float(cfg.GBSP_CVBR_UNIFORM_RESIDUAL_TOLERANCE),
        "variants": variants,
        "gt_used_for_generation": False, "training_used": False,
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    tasks = []
    for row in rows:
        key = (row["dataset"], row["stem"])
        tasks.append({
            "dataset": key[0], "stem": key[1],
            "core_path": row["cache_path"], "cvbr_path": cvbr_index[key]["cache_path"],
            "output_paths": {
                name: str(out_root / name / "test" / key[0] / f"{key[1]}.pt")
                for name in variants
            },
        })
    run_config = {
        "created_at": _now(), "config": str(_resolve(args.config)),
        "core_root": str(core_root), "cvbr_root": str(cvbr_root),
        "output_root": str(out_root), "sample_count": len(tasks),
        "full_run": full_run, "settings": settings,
    }
    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "run_config.json", run_config)

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(settings, args.torch_threads),
    ) as pool:
        for index, result in enumerate(pool.map(_generate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % args.progress_every == 0 or index == len(tasks):
                print(f"[{_now()}] CVBR-weighted PCA {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = {(row["dataset"], row["stem"]): row for row in results if "error" not in row}
    write_json(out_root / "generation_failures.json", failures)
    for name, specification in variants.items():
        manifest = []
        for task in tasks:
            key = (task["dataset"], task["stem"])
            if key in valid:
                manifest.append({
                    "dataset": key[0], "stem": key[1],
                    "cache_path": task["output_paths"][name],
                    "variant": name, "cache_key": specification["cache_key"],
                    "gt_path": core_index[key]["gt_path"],
                    "image_path": core_index[key]["image_path"],
                })
        write_jsonl(out_root / name / "manifest_test.jsonl", manifest)
    valid_rows = list(valid.values())
    summary = {
        "requested": len(tasks), "generated": len(valid_rows), "failed": len(failures),
        "resumed": sum(bool(row.get("skipped")) for row in valid_rows),
        "full_run": full_run,
        "is_full_complete": full_run and len(valid_rows) == 6473 and not failures,
        "wall_seconds": time.perf_counter() - started,
        "max_uniform_residual_abs": max(
            (float(row["uniform_residual_max_abs"]) for row in valid_rows), default=None
        ),
        "mean_candidates": sum(int(row["num_candidates"]) for row in valid_rows) / max(1, len(valid_rows)),
        "mean_cvbr_scored": sum(int(row["num_cvbr_scored"]) for row in valid_rows) / max(1, len(valid_rows)),
        "rare_failures_are_recorded": True,
    }
    write_json(out_root / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} samples failed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--gbsp-root", "--gbsp_root", dest="gbsp_root")
    value.add_argument("--cvbr-root", "--cvbr_root", dest="cvbr_root")
    value.add_argument("--out-root", "--out_root", dest="out_root")
    value.add_argument("--sample-list", "--sample_list", dest="sample_list")
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--progress-every", "--progress_every", dest="progress_every", type=int, default=20)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    value.add_argument("--dry-run", action="store_true")
    return value


if __name__ == "__main__":
    build(parser().parse_args())
