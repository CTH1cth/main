#!/usr/bin/env python3
"""Generate GT-free reconstruction-rescue score caches from frozen Full BC."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import load_config  # noqa: E402
from models.reconstruction import (  # noqa: E402
    global_pca_reconstruct, local_affine_reconstruct, local_convex_reconstruct,
    local_pca_reconstruct, retrieve_fullbc_neighbors,
)
from tools.gbsp_knn_lsr_common import write_json, write_jsonl  # noqa: E402
from tools.reconstruction_rescue_common import (  # noqa: E402
    apply_feature_manifest_override, atomic_torch_save, expected_full, load_core_inputs, load_core_rows,
    output_score_path, require_output_outside_main, settings_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dinov1_s8_reconstruction_rescue.py")
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--feature_manifest",
                        help="optional machine-local feature manifest used to rebase transferred core paths")
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query_batch_size", type=int)
    parser.add_argument("--save_diagnostics", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="reuse complete identity-matched payloads already present under out_root")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--min_free_gib", type=float, default=2.0)
    return parser.parse_args()


def _cpu(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = value.detach().cpu()
    return value.to(dtype=dtype) if dtype is not None else value


def _device_reproduction_warning(error: float, cfg) -> bool:
    tolerance = float(getattr(
        cfg, "RECONSTRUCTION_DEVICE_REPRODUCTION_WARNING_TOLERANCE", 2e-4
    ))
    return bool(error > tolerance)


def generate_one(row: dict, args: argparse.Namespace, cfg, device: torch.device) -> tuple[dict, dict]:
    core, raw, background = load_core_inputs(row, device)
    variants = tuple(args.variants or cfg.RECONSTRUCTION_PRIMARY_VARIANTS)
    allowed = (
        set(cfg.RECONSTRUCTION_PRIMARY_VARIANTS)
        | set(cfg.RECONSTRUCTION_SECONDARY_LSR_VARIANTS)
        | set(cfg.RECONSTRUCTION_CONDITIONAL_VARIANTS)
    )
    unknown = sorted(set(variants) - allowed)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    max_k = max([8] + [int(name.split("_k")[-1].split("_")[0]) for name in variants if "_k" in name])
    retrieval = retrieve_fullbc_neighbors(raw, background, max_k=max_k)
    scores: dict[str, torch.Tensor] = {
        "knn8": (1.0 - retrieval.cosine_similarities[:, :8].mean(dim=1)).reshape(1, 37, 37)
    }
    diagnostics: dict[str, dict] = {}
    formal = core["results"]["r8"]["absolute_raw"].to(device).float().reshape(-1)
    reproduction_error = float("nan")
    if "global_pca_l2" in variants:
        result = global_pca_reconstruct(raw, background, rank=8, normalize_features=True)
        reproduction_error = float((result.residual - formal).abs().max())
        # Global-PCA-L2 is the frozen formal baseline.  Publish the exact core
        # response; keep the device recomputation only as a numerical audit.
        scores["global_pca_l2"] = formal.reshape(1, 37, 37)
        diagnostics["global_pca_l2"] = {
            "response_source": "frozen_core_r8_absolute_raw",
            "device_reproduction_warning": _device_reproduction_warning(reproduction_error, cfg),
            "device_reproduction_max_abs_error": reproduction_error,
            "selected_rank": result.selected_rank,
            "retained_variance_ratio": result.retained_variance_ratio,
            "svd_orthonormal_error_before_qr": result.svd_orthonormal_error_before_qr,
            "orthonormal_error": result.orthonormal_error,
        }
    if "global_pca_raw" in variants:
        result = global_pca_reconstruct(raw, background, rank=8, normalize_features=False)
        scores["global_pca_raw"] = result.residual.reshape(1, 37, 37)
        diagnostics["global_pca_raw"] = {
            "selected_rank": result.selected_rank,
            "retained_variance_ratio": result.retained_variance_ratio,
            "svd_orthonormal_error_before_qr": result.svd_orthonormal_error_before_qr,
            "orthonormal_error": result.orthonormal_error,
        }
    for name in variants:
        if name.startswith("lcbr_"):
            geometry = "raw" if "_raw_" in name else "l2"
            k = int(name.rsplit("_k", 1)[1])
            result = local_convex_reconstruct(
                raw, retrieval, k=k, feature_geometry=geometry,
                query_batch_size=args.query_batch_size or cfg.RECONSTRUCTION_QUERY_BATCH_SIZE,
                max_iterations=cfg.RECONSTRUCTION_CONVEX_MAX_ITER,
                tolerance=cfg.RECONSTRUCTION_CONVEX_TOL,
            )
            scores[name] = result.residual.reshape(1, 37, 37)
            diag = {
                "converged_query_count": int(result.converged.sum()),
                "nonconverged_query_count": int((~result.converged).sum()),
                "simplex_sum_error_max": float(result.simplex_sum_error.max()),
                "minimum_alpha": float(result.minimum_alpha.min()),
                "objective_increase_max": result.objective_increase_max,
                "iteration_count": _cpu(result.iteration_count),
                "effective_active_atoms": _cpu(result.effective_active_atoms, torch.float16),
                "largest_alpha": _cpu(result.largest_alpha, torch.float16),
                "top2_alpha_sum": _cpu(result.top2_alpha_sum, torch.float16),
                "top4_alpha_sum": _cpu(result.top4_alpha_sum, torch.float16),
                "alpha_entropy": _cpu(result.alpha_entropy, torch.float16),
            }
            if args.save_diagnostics:
                diag["coefficients"] = _cpu(result.coefficients, torch.float16)
            diagnostics[name] = diag
        elif name.startswith("lsr_"):
            geometry = "raw" if "_raw_" in name else "l2"
            pieces = name.split("_")
            k = int(next(part[1:] for part in pieces if part.startswith("k")))
            rank = int(next(
                part[1:] for part in pieces
                if part.startswith("r") and part[1:].isdigit()
            ))
            result = local_pca_reconstruct(
                raw, retrieval, k=k, rank=rank, feature_geometry=geometry,
                query_batch_size=args.query_batch_size or cfg.RECONSTRUCTION_QUERY_BATCH_SIZE,
            )
            scores[name] = result.residual.reshape(1, 37, 37)
            diagnostics[name] = {
                "svd_orthonormal_error_before_qr_max": result.svd_orthonormal_error_before_qr_max,
                "orthonormal_error_max": result.orthonormal_error_max,
                "local_mean_norm": _cpu(result.local_mean_norm, torch.float16),
                "effective_rank": _cpu(result.effective_rank, torch.float16),
                "distance_to_local_mean": _cpu(result.distance_to_local_mean, torch.float16),
                **({"singular_values": _cpu(result.singular_values, torch.float16)} if args.save_diagnostics else {}),
            }
        elif name.startswith("lar_"):
            result = local_affine_reconstruct(
                raw, retrieval, k=16, feature_geometry="raw" if "_raw_" in name else "l2",
                rcond=cfg.RECONSTRUCTION_AFFINE_RCOND,
                query_batch_size=args.query_batch_size or cfg.RECONSTRUCTION_QUERY_BATCH_SIZE,
            )
            scores[name] = result.residual.reshape(1, 37, 37)
            diagnostics[name] = {
                "affine_sum_error_max": float(result.affine_sum_error.max()),
                "rcond": result.rcond,
                **({"coefficients": _cpu(result.coefficients, torch.float16)} if args.save_diagnostics else {}),
            }
    if any(not bool(torch.isfinite(score).all()) for score in scores.values()):
        raise RuntimeError("score contains NaN/Inf")
    payload = {
        "version": cfg.RECONSTRUCTION_RESCUE_VERSION,
        "dataset": row["dataset"], "stem": row["stem"],
        "image_path": row["image_path"], "gt_path": row["gt_path"],
        "source_core_path": row["cache_path"],
        "source_feature_path": core["source_feature_path"],
        "grid_size": 37, "feature_dim": 384,
        "background_source": "current_full_bc", "background_indices": _cpu(background),
        "retrieval_geometry": "l2_cosine", "self_match_excluded": True,
        "self_match_violation_count": retrieval.self_match_violation_count,
        "neighbor_indices": _cpu(retrieval.neighbor_indices, torch.int16),
        "neighbor_cosine_similarities": _cpu(retrieval.cosine_similarities, torch.float16),
        "scores": {name: _cpu(score.float()) for name, score in scores.items()},
        "diagnostics": diagnostics,
        "global_l2_response_source": (
            "frozen_core_r8_absolute_raw" if "global_pca_l2" in scores else None
        ),
        "global_l2_reproduction_max_abs_error": reproduction_error,
        "gt_used_for_generation": False,
    }
    audit = {
        "dataset": row["dataset"], "stem": row["stem"],
        "num_background": int(background.numel()),
        "self_match_violation_count": retrieval.self_match_violation_count,
        "global_l2_reproduction_max_abs_error": reproduction_error,
        "global_l2_reproduction_warning": (
            _device_reproduction_warning(reproduction_error, cfg)
            if "global_pca_l2" in scores else False
        ),
        "global_l2_reference_patched": False,
        "nonconverged_queries": sum(
            int(value.get("nonconverged_query_count", 0)) for value in diagnostics.values()
        ),
        "score_nan": 0,
    }
    return payload, audit


def audit_existing(row: dict, target: Path, variants: tuple[str, ...], cfg) -> tuple[dict, dict]:
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != cfg.RECONSTRUCTION_RESCUE_VERSION:
        raise RuntimeError(f"resume payload version mismatch: {target}")
    if (payload.get("dataset"), payload.get("stem")) != (row["dataset"], row["stem"]):
        raise RuntimeError(f"resume payload identity mismatch: {target}")
    expected = {"knn8", *variants}
    scores = payload.get("scores")
    if not isinstance(scores, dict) or not expected.issubset(scores):
        raise RuntimeError(f"resume payload misses scores {sorted(expected - set(scores or {}))}: {target}")
    if any(
        not torch.is_tensor(scores[name])
        or scores[name].numel() != 1369
        or not bool(torch.isfinite(scores[name]).all())
        for name in expected
    ):
        raise RuntimeError(f"resume payload has invalid scores: {target}")
    diagnostics = payload.get("diagnostics", {})
    baseline_patched = False
    if "global_pca_l2" in expected:
        core = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
        formal = core.get("results", {}).get("r8", {}).get("absolute_raw")
        if not torch.is_tensor(formal) or formal.numel() != 1369 or not bool(torch.isfinite(formal).all()):
            raise RuntimeError(f"formal frozen r8 response is invalid: {row['cache_path']}")
        formal = formal.float().reshape(1, 37, 37)
        current = scores["global_pca_l2"].float().reshape(1, 37, 37)
        baseline_patched = not torch.equal(current, formal)
        if baseline_patched or payload.get("global_l2_response_source") != "frozen_core_r8_absolute_raw":
            payload["scores"] = dict(scores)
            payload["scores"]["global_pca_l2"] = formal
            payload["source_core_path"] = row["cache_path"]
            payload["global_l2_response_source"] = "frozen_core_r8_absolute_raw"
            diagnostics = dict(payload.get("diagnostics", {}))
            diagnostics["global_pca_l2"] = {
                **dict(diagnostics.get("global_pca_l2", {})),
                "response_source": "frozen_core_r8_absolute_raw",
            }
            payload["diagnostics"] = diagnostics
            atomic_torch_save(payload, target)
            scores = payload["scores"]
    audit = {
        "dataset": row["dataset"], "stem": row["stem"],
        "num_background": int(torch.as_tensor(payload["background_indices"]).numel()),
        "self_match_violation_count": int(payload.get("self_match_violation_count", -1)),
        "global_l2_reproduction_max_abs_error": float(
            payload.get("global_l2_reproduction_max_abs_error", float("nan"))
        ),
        "nonconverged_queries": sum(
            int(value.get("nonconverged_query_count", 0))
            for value in diagnostics.values() if isinstance(value, dict)
        ),
        "score_nan": 0, "resumed": True,
        "global_l2_reference_patched": baseline_patched,
    }
    audit["global_l2_reproduction_warning"] = _device_reproduction_warning(
        audit["global_l2_reproduction_max_abs_error"], cfg
    )
    if audit["self_match_violation_count"] != 0:
        raise RuntimeError(f"resume payload contains a self match: {target}")
    manifest_row = {
        "dataset": row["dataset"], "stem": row["stem"],
        "image_path": row["image_path"], "gt_path": row["gt_path"],
        "score_path": str(target), "source_core_path": row["cache_path"],
    }
    return manifest_row, audit


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_root = require_output_outside_main(args.out_root)
    free = shutil.disk_usage(out_root.parent if out_root.parent.exists() else Path.cwd()).free / 2**30
    if free < float(args.min_free_gib):
        raise RuntimeError(f"only {free:.2f} GiB free; require {args.min_free_gib:.2f} GiB")
    rows = load_core_rows(args.core_root, split=args.split, max_samples=args.max_samples)
    if args.feature_manifest:
        rows = apply_feature_manifest_override(rows, args.feature_manifest)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest, audits, failures = [], [], []
    resumed = 0
    variants = tuple(args.variants or cfg.RECONSTRUCTION_PRIMARY_VARIANTS)
    started = time.time()
    for index, row in enumerate(rows, 1):
        target = output_score_path(out_root, args.split, row["dataset"], row["stem"])
        try:
            if args.resume and target.is_file():
                manifest_row, audit = audit_existing(row, target, variants, cfg)
                manifest.append(manifest_row); audits.append(audit); resumed += 1
                if index % 20 == 0 or index == len(rows):
                    print(
                        f"[{index}/{len(rows)}] valid={len(manifest)} resumed={resumed} failed={len(failures)}",
                        flush=True,
                    )
                continue
            payload, audit = generate_one(row, args, cfg, device)
            atomic_torch_save(payload, target)
            manifest.append({
                "dataset": row["dataset"], "stem": row["stem"],
                "image_path": row["image_path"], "gt_path": row["gt_path"],
                "score_path": str(target), "source_core_path": row["cache_path"],
            })
            audits.append(audit)
        except Exception as error:
            failure = {"dataset": row.get("dataset"), "stem": row.get("stem"),
                       "error": repr(error), "traceback": traceback.format_exc()}
            failures.append(failure)
            if args.failure_policy == "strict":
                raise
        if index % 20 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] valid={len(manifest)} failed={len(failures)}", flush=True)
    write_jsonl(out_root / f"score_manifest_{args.split}.jsonl", manifest)
    write_jsonl(out_root / "sample_audit.jsonl", audits)
    write_jsonl(out_root / "failures.jsonl", failures)
    counts = dict(Counter(row["dataset"] for row in manifest))
    settings = {
        "config": str(Path(args.config).resolve()),
        "core_root": str(Path(args.core_root).resolve()),
        "feature_manifest_override": str(Path(args.feature_manifest).resolve()) if args.feature_manifest else None,
        "variants": list(args.variants or cfg.RECONSTRUCTION_PRIMARY_VARIANTS),
        "split": args.split, "retrieval": "L2-cosine Full-BC strict-LOO",
        "gt_used_for_generation": False,
    }
    summary = {
        "requested": len(rows), "generated": len(manifest), "generation_failed": len(failures),
        "resumed": resumed,
        "score_nan": sum(row["score_nan"] for row in audits),
        "self_match_violation_count": sum(row["self_match_violation_count"] for row in audits),
        "nonconverged_queries": sum(row["nonconverged_queries"] for row in audits),
        "global_l2_reproduction_warning_count": sum(
            bool(row.get("global_l2_reproduction_warning", False)) for row in audits
        ),
        "global_l2_reference_patched_count": sum(
            bool(row.get("global_l2_reference_patched", False)) for row in audits
        ),
        "counts": counts, "full_input": expected_full(rows),
        "is_full_complete": expected_full(rows) and len(manifest) == len(rows) and not failures,
        "wall_seconds": time.time() - started,
        "settings": settings, "settings_fingerprint": settings_fingerprint(settings),
    }
    write_json(out_root / "validity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
