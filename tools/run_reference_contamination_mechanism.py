#!/usr/bin/env python3
"""Cache Experiments A/B/C for reference-contamination mechanism validation.

This is a diagnostic/oracle experiment.  GT is never used to generate a
pseudo-label; it only labels natural candidates or performs declared
counterfactual candidate-set interventions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.reference_contamination import (  # noqa: E402
    NUM_PATCHES,
    controlled_clustered_candidate_set,
    controlled_candidate_set,
    local_wrong_reference_counts,
    matched_size_clean_candidate_set,
    normalize_patch_features,
    score_gbsp_fixed_rank,
    score_knn8,
)
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    DATASETS,
    EXPECTED_COUNTS,
    load_manifest,
    load_native_gt,
    load_patch_area,
    load_torch,
    normalize_dataset,
    patch_labels,
    rank_metrics,
    resize_score_to_native,
    write_json,
    write_jsonl,
)


VERSION = "reference_contamination_mechanism_v1"
MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_P = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
CLUSTER_P = (0.0, 0.05, 0.10, 0.20)
CLUSTER_SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbsp_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--p_values", nargs="+", type=float, default=DEFAULT_P)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--matched_size_seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--contamination_mode",
        choices=("random_scattered", "random_scattered_matched", "spatial_cluster"),
        default="random_scattered",
    )
    parser.add_argument("--skip_matched_size", action="store_true")
    parser.add_argument(
        "--matched_size_only", action="store_true",
        help=(
            "generate natural/oracle-clean/matched-size-clean scores only; "
            "skip all controlled-contamination conditions"
        ),
    )
    parser.add_argument(
        "--defer_native_metrics", action="store_true",
        help="cache raw scores first and compute native B metrics in a separate CPU stage",
    )
    parser.add_argument(
        "--path_map", action="append", default=[], metavar="OLD=NEW",
        help="relocate missing absolute cache/data paths; may be repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--failure_policy", choices=("strict", "record"), default="strict",
        help="strict raises after generation failures; record preserves and reports rare invalid cases",
    )
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def _parse_path_maps(values: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    mappings: dict[str, str] = {}
    for raw in values:
        if "=" not in str(raw):
            raise ValueError(f"path map must be OLD=NEW, got {raw!r}")
        old_raw, new_raw = str(raw).split("=", 1)
        if not old_raw or not new_raw:
            raise ValueError(f"path map must contain non-empty OLD and NEW: {raw!r}")
        old = str(Path(old_raw).expanduser().resolve())
        new = str(Path(new_raw).expanduser().resolve())
        if old in mappings and mappings[old] != new:
            raise ValueError(f"conflicting path maps for {old}: {mappings[old]} vs {new}")
        mappings[old] = new
    # The repository-specific prefix must win over its broader CTH prefix.
    return tuple(sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True))


def _resolve_existing_path(
    value: str | Path,
    mappings: tuple[tuple[str, str], ...] | list[list[str]] | list[tuple[str, str]],
    *,
    role: str,
) -> Path:
    source = Path(value).expanduser()
    if source.is_file():
        return source.resolve()
    attempted = [str(source)]
    for old_raw, new_raw in mappings:
        old, new = Path(old_raw), Path(new_raw)
        try:
            relative = source.relative_to(old)
        except ValueError:
            continue
        candidate = new / relative
        attempted.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{role} not found after path relocation; attempted={attempted}; "
        "provide --path_map OLD=NEW for every machine-specific prefix"
    )


def _preflight_source_paths(row: dict, mappings: tuple[tuple[str, str], ...]) -> dict:
    core_path = _resolve_existing_path(row["cache_path"], mappings, role="GBSP core cache")
    core = load_torch(core_path)
    resolved = {
        "core_cache": str(core_path),
        "feature_cache": str(_resolve_existing_path(
            core["source_feature_path"], mappings, role="DINO feature cache"
        )),
        "native_gt": str(_resolve_existing_path(core["gt_path"], mappings, role="native GT")),
        "source_image": str(_resolve_existing_path(
            core["image_path"], mappings, role="source image"
        )),
    }
    return {
        "status": "passed_before_generation",
        "identity": [normalize_dataset(row["dataset"]), str(row["stem"])],
        "resolved_first_sample": resolved,
    }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = torch.device(name)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _feature_from_payload(payload: dict, path: Path) -> torch.Tensor:
    for key in ("tensor", "patch_tokens", "features"):
        value = payload.get(key)
        if torch.is_tensor(value) and value.numel() == NUM_PATCHES * 384:
            return value.squeeze().float().contiguous()
    tensors = [
        value for value in payload.values()
        if torch.is_tensor(value) and value.numel() == NUM_PATCHES * 384
    ]
    if len(tensors) != 1:
        raise KeyError(f"cannot resolve a unique 384x37x37 feature tensor: {path}")
    return tensors[0].squeeze().float().contiguous()


def _cpu(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = value.detach().cpu()
    return value.to(dtype=dtype).contiguous() if dtype is not None else value.contiguous()


def _score_pair(
    feature: torch.Tensor,
    candidate: torch.Tensor,
    labels: np.ndarray,
    *,
    rank: int,
) -> dict:
    knn = score_knn8(feature, candidate)
    gbsp = score_gbsp_fixed_rank(feature, candidate, rank=rank)
    wrong = local_wrong_reference_counts(knn.neighbor_indices, labels)
    return {
        "knn8_score": _cpu(knn.score, torch.float32),
        "gbsp_score": _cpu(gbsp.score, torch.float32),
        "neighbor_indices": _cpu(knn.neighbor_indices, torch.int16),
        "local_wrong_count": _cpu(wrong, torch.uint8),
        "self_match_violation_count": int(knn.self_match_violation_count),
        "gbsp_selected_rank": int(gbsp.selected_rank),
        "gbsp_orthonormal_error": float(gbsp.orthonormal_error),
    }


def _native_metrics(score: torch.Tensor, native_gt: torch.Tensor) -> dict[str, float]:
    native = resize_score_to_native(score, tuple(native_gt.shape[-2:]))
    return rank_metrics(native, native_gt)


def _patch_metrics(score: torch.Tensor, labels: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    value = score.detach().cpu().numpy().reshape(-1)
    return rank_metrics(value[valid], np.asarray(labels).reshape(-1)[valid])


def _valid_existing(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return (
            payload.get("version") == VERSION
            and payload.get("settings_fingerprint") == fingerprint
            and "natural" in payload
            and "oracle_clean" in payload
            and "contamination" in payload
            and bool(payload.get("native_metrics_deferred", False))
            == bool(payload.get("settings", {}).get("defer_native_metrics", False))
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _generate_one(row: dict, *, device: torch.device, settings: dict, output_path: Path) -> dict:
    started = time.perf_counter()
    path_maps = tuple(tuple(item) for item in settings["path_maps"])
    core_path = _resolve_existing_path(row["cache_path"], path_maps, role="GBSP core cache")
    core = load_torch(core_path)
    dataset = normalize_dataset(row["dataset"])
    identity = f"{dataset}/{row['stem']}"
    if (normalize_dataset(core.get("dataset", "")), str(core.get("stem", ""))) != (dataset, row["stem"]):
        raise RuntimeError(f"core cache identity mismatch: {core_path}")
    r8 = core.get("results", {}).get("r8")
    if not isinstance(r8, dict):
        raise KeyError(f"fixed r8 result is missing: {core_path}")
    feature_path = _resolve_existing_path(
        core["source_feature_path"], path_maps, role="DINO feature cache"
    )
    gt_path = _resolve_existing_path(core["gt_path"], path_maps, role="native GT")
    image_path = _resolve_existing_path(core["image_path"], path_maps, role="source image")
    feature_payload = load_torch(feature_path)
    feature = normalize_patch_features(_feature_from_payload(feature_payload, feature_path)).to(device)
    candidate_natural = torch.as_tensor(r8["background_indices"], dtype=torch.long).reshape(-1)
    patch_area = load_patch_area(gt_path)
    labels, label_valid = patch_labels(patch_area, "main_0.5")
    native_gt = None if settings["defer_native_metrics"] else load_native_gt(gt_path)
    label_tensor = torch.from_numpy(labels.astype(bool))
    wrong_candidates = candidate_natural[label_tensor.index_select(0, candidate_natural)]
    clean_candidate = candidate_natural[~label_tensor.index_select(0, candidate_natural)]
    if clean_candidate.numel() < max(settings["knn_k"] + 1, settings["rank"] + 2):
        raise RuntimeError(f"Oracle-clean dictionary is too small: {identity}")

    # Experiment A: natural contamination and local top-8 amplification.
    natural_knn = score_knn8(feature, candidate_natural.to(device))
    natural_wrong_count = local_wrong_reference_counts(natural_knn.neighbor_indices, labels)
    natural_gbsp = torch.as_tensor(r8["absolute_raw"]).float().reshape(-1)
    if natural_gbsp.numel() != NUM_PATCHES or not bool(torch.isfinite(natural_gbsp).all()):
        raise RuntimeError(f"cached natural GBSP r8 score is invalid: {core_path}")
    natural = {
        "candidate_indices": candidate_natural.to(torch.int16),
        "wrong_candidate_indices": wrong_candidates.to(torch.int16),
        "candidate_count": int(candidate_natural.numel()),
        "wrong_candidate_count": int(wrong_candidates.numel()),
        "global_contamination": float(wrong_candidates.numel() / candidate_natural.numel()),
        "query_fg_mask": torch.from_numpy((labels.astype(bool) & ~np.isin(np.arange(NUM_PATCHES), candidate_natural.numpy()))),
        "knn8_score": _cpu(natural_knn.score, torch.float32),
        "gbsp_score": natural_gbsp.cpu().contiguous(),
        "neighbor_indices": _cpu(natural_knn.neighbor_indices, torch.int16),
        "local_wrong_count": _cpu(natural_wrong_count, torch.uint8),
        "self_match_violation_count": int(natural_knn.self_match_violation_count),
    }

    # Experiment B: oracle removal of all foreground references.
    clean_scores = _score_pair(feature, clean_candidate.to(device), labels, rank=settings["rank"])
    oracle_clean = {
        "candidate_indices": clean_candidate.to(torch.int16),
        "candidate_count": int(clean_candidate.numel()),
        "removed_count": int(wrong_candidates.numel()),
        **clean_scores,
    }
    if native_gt is not None:
        oracle_clean["natural_native_metrics"] = {
            "knn8": _native_metrics(natural["knn8_score"], native_gt),
            "gbsp": _native_metrics(natural["gbsp_score"], native_gt),
        }
        oracle_clean["clean_native_metrics"] = {
            "knn8": _native_metrics(clean_scores["knn8_score"], native_gt),
            "gbsp": _native_metrics(clean_scores["gbsp_score"], native_gt),
        }

    matched = []
    matched_interventions = {}
    matched_scores = {}
    if not settings["skip_matched_size"]:
        for seed in settings["matched_size_seeds"]:
            intervention = matched_size_clean_candidate_set(
                candidate_natural, labels, seed=seed, identity=identity
            )
            item = {
                "seed": int(seed), "valid": bool(intervention.valid), "reason": intervention.reason,
                "candidate_count": int(intervention.indices.numel()),
                "candidate_indices": intervention.indices.to(torch.int16),
                "replacement_bg_indices": intervention.injected_foreground_indices.to(torch.int16),
            }
            if intervention.valid:
                scores = _score_pair(feature, intervention.indices.to(device), labels, rank=settings["rank"])
                matched_interventions[int(seed)] = intervention
                matched_scores[int(seed)] = scores
                item["knn8_score"] = scores["knn8_score"]
                item["gbsp_score"] = scores["gbsp_score"]
                if native_gt is not None:
                    item["native_metrics"] = {
                        "knn8": _native_metrics(scores["knn8_score"], native_gt),
                        "gbsp": _native_metrics(scores["gbsp_score"], native_gt),
                    }
                item["self_match_violation_count"] = scores["self_match_violation_count"]
            matched.append(item)

    # Experiment C: nested, size-preserving, controlled reference contamination.
    # The matched-size-only path is the efficient formal control requested when
    # no foreground-injection curve is needed.
    contamination = []
    clean_pair = clean_scores
    contamination_conditions = (
        () if settings["matched_size_only"] else (
            (seed, fraction)
            for seed in settings["seeds"]
            for fraction in settings["p_values"]
        )
    )
    for seed, fraction in contamination_conditions:
            if settings["contamination_mode"] in (
                "spatial_cluster", "random_scattered_matched"
            ):
                matched_base = matched_interventions.get(int(seed))
                if matched_base is None:
                    raise RuntimeError(
                        "matched-size oracle-clean base is unavailable for "
                        f"contamination seed={seed}: {identity}"
                    )
                if settings["contamination_mode"] == "spatial_cluster":
                    intervention = controlled_clustered_candidate_set(
                        matched_base.indices, labels, fraction=fraction,
                        seed=seed, identity=identity,
                    )
                else:
                    intervention = controlled_candidate_set(
                        matched_base.indices, labels, fraction=fraction,
                        seed=seed, identity=identity,
                        # Match the cluster intervention's removed clean-BG
                        # atoms, leaving FG injection topology as the only
                        # factor that differs between the two protocols.
                        removal_namespace="remove_bg_cluster",
                    )
            else:
                intervention = controlled_candidate_set(
                    clean_candidate, labels, fraction=fraction, seed=seed, identity=identity
                )
            item = {
                "seed": int(seed),
                "p": float(fraction),
                "valid": bool(intervention.valid),
                "reason": intervention.reason,
                "candidate_count": int(intervention.indices.numel()),
                "realized_fraction": float(intervention.realized_fraction),
                "removed_background_indices": intervention.removed_background_indices.to(torch.int16),
                "injected_foreground_indices": intervention.injected_foreground_indices.to(torch.int16),
                "injection_mode": intervention.injection_mode,
                "selected_component_size": int(intervention.selected_component_size),
                "cluster_anchor_index": int(intervention.cluster_anchor_index),
                "injected_connected": bool(intervention.injected_connected),
            }
            if intervention.valid:
                if float(fraction) == 0.0:
                    scores = (
                        matched_scores[int(seed)]
                        if settings["contamination_mode"] in (
                            "spatial_cluster", "random_scattered_matched"
                        )
                        else clean_pair
                    )
                else:
                    scores = _score_pair(
                        feature, intervention.indices.to(device), labels, rank=settings["rank"]
                    )
                injected = np.zeros(NUM_PATCHES, dtype=bool)
                injected[intervention.injected_foreground_indices.numpy()] = True
                noninjected = label_valid & ~injected
                item.update({
                    "knn8_score": scores["knn8_score"],
                    "gbsp_score": scores["gbsp_score"],
                    "local_wrong_count": scores["local_wrong_count"],
                    "self_match_violation_count": int(scores["self_match_violation_count"]),
                    "patch_metrics_noninjected": {
                        "knn8": _patch_metrics(scores["knn8_score"], labels, noninjected),
                        "gbsp": _patch_metrics(scores["gbsp_score"], labels, noninjected),
                    },
                    "patch_metrics_full": {
                        "knn8": _patch_metrics(scores["knn8_score"], labels, label_valid),
                        "gbsp": _patch_metrics(scores["gbsp_score"], labels, label_valid),
                    },
                })
            contamination.append(item)

    payload = {
        "version": VERSION,
        "created_at": _now(),
        "settings_fingerprint": settings["fingerprint"],
        "settings": {key: value for key, value in settings.items() if key != "fingerprint"},
        "dataset": dataset,
        "stem": row["stem"],
        "image_path": str(image_path),
        "gt_path": str(gt_path),
        "source_core_path": str(core_path),
        "source_feature_path": str(feature_path),
        "patch_gt_area": torch.from_numpy(patch_area.astype(np.float32)),
        "patch_gt_label": torch.from_numpy(labels.astype(np.uint8)),
        "gt_use": "diagnosis_and_declared_oracle_intervention_only",
        "pseudo_label_generated": False,
        "native_metrics_deferred": bool(settings["defer_native_metrics"]),
        "native_metrics_complete": not bool(settings["defer_native_metrics"]),
        "natural": natural,
        "oracle_clean": oracle_clean,
        "matched_size_clean": matched,
        "contamination": contamination,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_save(payload, output_path)
    return {
        "dataset": dataset,
        "stem": row["stem"],
        "cache_path": str(output_path),
        "runtime_seconds": payload["runtime_seconds"],
        "natural_candidate_count": natural["candidate_count"],
        "natural_wrong_count": natural["wrong_candidate_count"],
        "clean_candidate_count": oracle_clean["candidate_count"],
        "valid_contamination_conditions": sum(int(item["valid"]) for item in contamination),
        "invalid_contamination_conditions": sum(int(not item["valid"]) for item in contamination),
        "self_match_violation_count": int(natural["self_match_violation_count"])
        + int(oracle_clean["self_match_violation_count"])
        + sum(int(item.get("self_match_violation_count", 0)) for item in contamination),
    }


def main() -> None:
    args = parse_args()
    if args.knn_k != 8 or args.rank != 8:
        raise ValueError("formal protocol is frozen to KNN8 and fixed GBSP rank=8")
    p_values = sorted(set(float(value) for value in args.p_values))
    seeds = tuple(int(value) for value in args.seeds)
    matched_seeds = tuple(int(value) for value in args.matched_size_seeds)
    if args.matched_size_only:
        if args.skip_matched_size:
            raise ValueError("--matched_size_only conflicts with --skip_matched_size")
        if matched_seeds != CLUSTER_SEEDS:
            raise ValueError(f"formal matched-size seeds must be {CLUSTER_SEEDS}")
    elif args.contamination_mode in ("spatial_cluster", "random_scattered_matched"):
        if tuple(p_values) != CLUSTER_P:
            raise ValueError(f"matched comparison p-values must be {CLUSTER_P}")
        if seeds != CLUSTER_SEEDS or matched_seeds != CLUSTER_SEEDS:
            raise ValueError(f"matched comparison seeds must be {CLUSTER_SEEDS}")
        if args.skip_matched_size:
            raise ValueError("matched comparison requires a matched-size oracle-clean base")
    else:
        if tuple(p_values) != DEFAULT_P:
            raise ValueError(f"formal p-values must be {DEFAULT_P}")
        if seeds != DEFAULT_SEEDS or (not args.skip_matched_size and matched_seeds != DEFAULT_SEEDS):
            raise ValueError(f"formal seeds must be {DEFAULT_SEEDS}")
    out_dir = Path(args.out_dir).resolve()
    if out_dir == MAIN_ROOT or MAIN_ROOT in out_dir.parents:
        raise ValueError("output directory must remain outside the main code tree")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    rows = load_manifest(args.gbsp_root, split=args.split, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("source manifest selected zero samples")
    path_maps = _parse_path_maps(tuple(args.path_map))
    path_preflight = _preflight_source_paths(rows[0], path_maps)
    counts = Counter(normalize_dataset(row["dataset"]) for row in rows)
    complete = len(rows) == 6473 and dict(counts) == EXPECTED_COUNTS
    settings = {
        "version": VERSION,
        "representation": "DINOv1-S/8 last-block attention-key, 296 input, 37x37, patch L2",
        "full_bc": "frozen existing Full-BC, no new mining",
        "knn_k": 8,
        "rank": 8,
        "p_values": p_values,
        "seeds": seeds,
        "matched_size_seeds": matched_seeds,
        "contamination_mode": args.contamination_mode,
        "skip_matched_size": bool(args.skip_matched_size),
        "matched_size_only": bool(args.matched_size_only),
        "defer_native_metrics": bool(args.defer_native_metrics),
        "path_maps": [list(item) for item in path_maps],
        "patch_gt_rule": "area_downsample_to_37; foreground_if_occupancy>=0.5",
        "native_protocol_B": "raw_37_to_68_to_native_bilinear; per-image AP/AUROC",
        "contamination_protocol_C": (
            "skipped; matched-size oracle-clean control only"
            if args.matched_size_only else
            "matched-size oracle-clean base; largest GT-FG 8-neighbor component; "
            "seeded connected BFS prefixes; fixed dictionary size; patch37 Q_noninj main"
            if args.contamination_mode == "spatial_cluster"
            else (
                "matched-size oracle-clean base; seeded random FG permutation prefixes; "
                "background-removal schedule matched to spatial-cluster; fixed dictionary "
                "size; patch37 Q_noninj main"
                if args.contamination_mode == "random_scattered_matched"
                else "patch37 Q_noninj main; patch37 full-query secondary"
            )
        ),
        "gt_use": "diagnosis_and_declared_oracle_intervention_only",
        "pseudo_label_generated": False,
        "training_used": False,
    }
    settings["fingerprint"] = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(out_dir / "run_config.json", {
        "created_at": _now(), "device": str(device), "requested_samples": len(rows),
        "dataset_counts": dict(counts), "is_formal_full6473": complete,
        "source_manifest_root": str(Path(args.gbsp_root).resolve()),
        "path_preflight": path_preflight, "settings": settings,
    })

    generated, failures = [], []
    for index, row in enumerate(rows, 1):
        dataset = normalize_dataset(row["dataset"])
        path = out_dir / "cache" / args.split / dataset / f"{row['stem']}.pt"
        if not args.overwrite and _valid_existing(path, settings["fingerprint"]):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            generated.append({
                "dataset": dataset, "stem": row["stem"], "cache_path": str(path),
                "runtime_seconds": float(payload.get("runtime_seconds", 0.0)), "skipped": True,
            })
        else:
            try:
                result = _generate_one(row, device=device, settings=settings, output_path=path)
                result["skipped"] = False
                generated.append(result)
            except Exception as error:
                failures.append({
                    "dataset": dataset, "stem": row.get("stem", ""),
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(
                f"[{index}/{len(rows)}] generated={len(generated)} failed={len(failures)} "
                f"device={device}", flush=True,
            )
            write_jsonl(out_dir / "manifest_test.jsonl", generated)
            write_json(out_dir / "failures.json", failures)
    total_violations = sum(int(row.get("self_match_violation_count", 0)) for row in generated)
    validity = {
        "version": VERSION,
        "requested": len(rows), "generated": len(generated), "failed": len(failures),
        "dataset_counts": dict(Counter(row["dataset"] for row in generated)),
        "expected_full_counts": EXPECTED_COUNTS,
        "is_formal_full6473": complete and not failures and len(generated) == 6473,
        "self_match_violation_count": total_violations,
        "native_metrics_deferred": bool(args.defer_native_metrics),
        "native_metrics_complete": not bool(args.defer_native_metrics),
        "gt_used_for_pseudo_label_generation": False,
        "failure_policy": args.failure_policy,
        "declared_invalid_samples": failures,
    }
    write_json(out_dir / "validity_summary.json", validity)
    print(json.dumps(validity, ensure_ascii=False), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} samples failed; see {out_dir / 'failures.json'}")
    if total_violations:
        raise RuntimeError(f"leave-one-out audit failed: {total_violations} self matches")


if __name__ == "__main__":
    main()
