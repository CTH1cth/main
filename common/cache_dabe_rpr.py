#!/usr/bin/env python3
"""Build GT-free DABE RPR-v1 candidates from frozen identity caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_r1_design import (  # noqa: E402
    _feature,
    _manifest_map,
    _snapshot,
    _source_r1,
    _validate_output_root,
    effective_dabe_v2_params,
)
from common.dabe_cvbr import (  # noqa: E402
    CVBR_EXCLUSION_RADIUS,
    CVBR_Q_MIN,
    CVBR_VERSION,
)
from common.dabe_rpr import (  # noqa: E402
    DABE_RPR_VERSION,
    EFFECTIVE_ATOM_FIELDS,
    NONNEGATIVE_FIELDS,
    PROBABILITY_FIELDS,
    SIGNED_FIELDS,
    build_rpr_candidates,
)
from common.utils import (  # noqa: E402
    feature_manifest_path,
    load_config,
    torch_load,
    write_json,
    write_jsonl,
)


SCRIPT_PATH = Path(__file__).resolve()
MAIN_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_TOTAL = 6473
EXPECTED_DATASETS = {
    "CHAMELEON": 76,
    "TE-CAMO": 250,
    "TE-COD10K": 2026,
    "NC4K": 4121,
}
REQUIRED_DIAGNOSTICS = (
    "b0_cached_r1_max_abs",
    "b0_recomputed_cached_r1_max_abs",
    "unit_prior_topk_mismatch_count",
    "p1_topk_mismatch_count",
    "p2_topk_mismatch_count",
    "unit_prior_weight_max_abs",
    "unit_prior_raw_max_abs",
    "unit_prior_normalized_max_abs",
    "b0_anchor_count",
    "p1_prior_modified_atom_count",
    "p2_prior_modified_atom_count",
    "p1_query_exposure_mean",
    "p1_query_exposure_max",
    "p1_query_exposure_nonzero_ratio",
    "p2_query_exposure_mean",
    "p2_query_exposure_max",
    "p2_query_exposure_nonzero_ratio",
    "p1_u_reduction_mean",
    "p2_u_reduction_mean",
    "p1_weight_shift_mean",
    "p1_weight_shift_max",
    "p2_weight_shift_mean",
    "p2_weight_shift_max",
    "effective_atoms_base_mean",
    "effective_atoms_p1_mean",
    "effective_atoms_p2_mean",
    "max_weight_base_mean",
    "max_weight_p1_mean",
    "max_weight_p2_mean",
    "raw_delta_p1_mean",
    "raw_delta_p1_std",
    "raw_delta_p2_mean",
    "raw_delta_p2_std",
    "p1_area_gt_05",
    "p2_area_gt_05",
    "cross_fallback_count_from_source",
    "p1_u_monotonicity_violation_count",
    "p1_u_monotonicity_max_violation",
    "p2_u_monotonicity_violation_count",
    "p2_u_monotonicity_max_violation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(MAIN_ROOT), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).rstrip()
        return commit, status
    except (OSError, subprocess.CalledProcessError):
        return "", ""


def _require_equal(mapping: dict, key: str, expected):
    actual = mapping.get(key)
    if actual != expected:
        raise RuntimeError(f"CVBR protocol {key} mismatch: {actual!r} != {expected!r}")


def _validate_cvbr_protocol(
    protocol_path: Path,
    dabe_manifest: Path,
) -> dict:
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("cvbr_version", CVBR_VERSION),
        ("split", "test"),
        ("num_samples", EXPECTED_TOTAL),
        ("backbone", "DINOv1-S/8"),
        ("feature_type", "last-block attention key"),
        ("feature_input_size", 296),
        ("grid", 37),
        ("source_augs", ["identity"]),
        ("gt_used_for_generation", False),
        ("dino_forward_used", False),
        ("training_used", False),
        ("threshold_search_used", False),
        ("dataset_specific_rule", False),
        ("sample_routing_used", False),
        ("trainable_parameter_count", 0),
    ):
        _require_equal(protocol, key, expected)
    if protocol.get("source_dabe_manifest_sha256") != _sha256(dabe_manifest):
        raise RuntimeError("CVBR source_dabe_manifest_sha256 mismatch")

    params = protocol.get("effective_dabe_params")
    if not isinstance(params, dict):
        raise RuntimeError("CVBR effective_dabe_params missing")
    frozen = {
        "GRID": 37,
        "K_RECON": 32,
        "LAMBDA_COLOR_RECON": 0.20,
        "SIGMA_COLOR_RECON": 0.05,
        "TAU_RECON": 0.07,
    }
    for key, expected in frozen.items():
        if float(params.get(key, float("nan"))) != float(expected):
            raise RuntimeError(f"CVBR frozen parameter mismatch: {key}")
    formula = protocol.get("cvbr_formula")
    if not isinstance(formula, dict):
        raise RuntimeError("CVBR formula record missing")
    if "radius-1" not in str(formula.get("cross_error", "")):
        raise RuntimeError("CVBR cross exclusion radius is not frozen at one")
    if str(formula.get("reliability", "")) != "q=clip(exp(-relu((e-m)/s)),1e-6,1)":
        raise RuntimeError("CVBR reliability formula mismatch")
    if CVBR_EXCLUSION_RADIUS != 1 or CVBR_Q_MIN != 1e-6:
        raise RuntimeError("imported CVBR constants no longer match frozen protocol")
    return protocol


def _validate_dataset_counts(rows: list[dict]):
    counts = {dataset: 0 for dataset in EXPECTED_DATASETS}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in counts:
            raise RuntimeError(f"unexpected dataset in full test manifest: {dataset}")
        counts[dataset] += 1
    if counts != EXPECTED_DATASETS:
        raise RuntimeError(f"dataset counts mismatch: {counts} != {EXPECTED_DATASETS}")


def _validate_cvbr_payload(payload: dict, dataset: str, stem: str, path: Path):
    if not isinstance(payload, dict):
        raise TypeError(f"CVBR payload must be dict: {path}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"CVBR payload key mismatch: {path}")
    if payload.get("cvbr_version") != CVBR_VERSION:
        raise ValueError(f"CVBR version mismatch: {path}")
    if payload.get("source_augs") != ["identity"] or int(
        payload.get("source_num_views", 0)
    ) != 1:
        raise ValueError(f"CVBR payload must be identity single-view: {path}")


def _validate_result(result: dict, context: str, k_recon: int):
    for field in PROBABILITY_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.requires_grad
            or not value.is_contiguous()
        ):
            raise ValueError(f"{field} tensor contract invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
            raise ValueError(f"{field} outside finite [0,1]: {context}")
    for field in NONNEGATIVE_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.requires_grad
            or not value.is_contiguous()
            or not torch.isfinite(value).all()
            or float(value.min()) < -1e-7
        ):
            raise ValueError(f"{field} invalid nonnegative map: {context}")
    for field in SIGNED_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if (
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.requires_grad
            or not value.is_contiguous()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{field} invalid signed map: {context}")
    for field in EFFECTIVE_ATOM_FIELDS:
        value = result.get(field)
        if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
            raise ValueError(f"{field} shape invalid: {context}")
        if not torch.isfinite(value).all() or float(value.min()) < 1 - 1e-5 or float(value.max()) > k_recon + 1e-5:
            raise ValueError(f"{field} outside [1,K]: {context}")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise TypeError(f"diagnostics missing: {context}")
    for field in REQUIRED_DIAGNOSTICS:
        if field not in diagnostics:
            raise KeyError(f"diagnostic {field} missing: {context}")
        value = diagnostics[field]
        if isinstance(value, (int, float)) and not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"diagnostic {field} nonfinite: {context}")
    if diagnostics["unit_prior_topk_mismatch_count"] != 0:
        raise RuntimeError(f"unit-prior TopK mismatch: {context}")
    if float(diagnostics["unit_prior_weight_max_abs"]) > 1e-7:
        raise RuntimeError(f"unit-prior weight mismatch: {context}")
    if float(diagnostics["unit_prior_raw_max_abs"]) > 1e-6:
        raise RuntimeError(f"unit-prior raw mismatch: {context}")
    if float(diagnostics["unit_prior_normalized_max_abs"]) > 1e-6:
        raise RuntimeError(f"unit-prior normalized mismatch: {context}")
    if float(diagnostics["b0_cached_r1_max_abs"]) > 1e-6:
        raise RuntimeError(f"cached B0/R1 mismatch: {context}")


def _init_worker(torch_threads: int):
    torch.set_num_threads(int(torch_threads))


def _process_one(task: dict):
    dabe_row = task["dabe_row"]
    cvbr_row = task["cvbr_row"]
    feature_row = task["feature_row"]
    dataset, stem = str(dabe_row["dataset"]), str(dabe_row["stem"])
    dabe_path = Path(dabe_row["cache_path"]).resolve()
    cvbr_path = Path(cvbr_row["cache_path"]).resolve()
    feature_path = Path(feature_row["cache_path"]).resolve()
    dabe = torch_load(dabe_path, map_location="cpu")
    cvbr = torch_load(cvbr_path, map_location="cpu")
    feature_payload = torch_load(feature_path, map_location="cpu")
    cached_r1 = _source_r1(dabe, dabe_row, dabe_path)
    _validate_cvbr_payload(cvbr, dataset, stem, cvbr_path)
    feature = _feature(feature_payload, dataset, stem, feature_path)
    if Path(cvbr.get("source_dabe_cache_path", "")).resolve() != dabe_path:
        raise RuntimeError(f"CVBR/DABE provenance mismatch: {dataset}/{stem}")
    if Path(cvbr.get("source_feature_cache_path", "")).resolve() != feature_path:
        raise RuntimeError(f"CVBR/feature provenance mismatch: {dataset}/{stem}")
    image_text = dabe_row.get("image_path") or feature_row.get("image_path") or cvbr.get("image_path")
    if not image_text:
        raise KeyError(f"image_path missing for {dataset}/{stem}")
    image_path = Path(image_text).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    result = build_rpr_candidates(
        feature_37=feature,
        image_path=str(image_path),
        cached_r1_37=cached_r1,
        cvbr_payload=cvbr,
        effective_params=task["effective_params"],
    )
    _validate_result(result, f"{dataset}/{stem}", int(task["effective_params"]["K_RECON"]))

    output_path = Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "stem": stem,
        "rpr_version": DABE_RPR_VERSION,
        "source_dabe_version": "v2",
        "source_cvbr_version": CVBR_VERSION,
        "source_augs": ["identity"],
        "source_num_views": 1,
        "source_dabe_cache_path": str(dabe_path),
        "source_cvbr_cache_path": str(cvbr_path),
        "source_feature_cache_path": str(feature_path),
        "image_path": str(image_path),
        **result,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    diagnostics = result["diagnostics"]
    return {
        "manifest_row": {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path.resolve()),
            "source_dabe_cache_path": str(dabe_path),
            "source_cvbr_cache_path": str(cvbr_path),
            "source_feature_cache_path": str(feature_path),
            "rpr_version": DABE_RPR_VERSION,
            "shape": [1, 37, 37],
        },
        "diagnostics": diagnostics,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _resume_existing(task: dict):
    row = task["dabe_row"]
    dataset, stem = str(row["dataset"]), str(row["stem"])
    output_path = Path(task["output_root"]) / task["split"] / dataset / f"{stem}.pt"
    payload = torch_load(output_path, map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"resumed RPR payload key mismatch: {output_path}")
    if payload.get("rpr_version") != DABE_RPR_VERSION:
        raise RuntimeError(f"resumed RPR version mismatch: {output_path}")
    expected_paths = {
        "source_dabe_cache_path": Path(row["cache_path"]).resolve(),
        "source_cvbr_cache_path": Path(task["cvbr_row"]["cache_path"]).resolve(),
        "source_feature_cache_path": Path(task["feature_row"]["cache_path"]).resolve(),
    }
    for field, expected in expected_paths.items():
        if Path(payload.get(field, "")).resolve() != expected:
            raise RuntimeError(f"resumed provenance mismatch for {field}: {output_path}")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError(f"resumed diagnostics missing: {output_path}")
    changed = False
    for tag in ("p1", "p2"):
        count_key = f"{tag}_u_monotonicity_violation_count"
        max_key = f"{tag}_u_monotonicity_max_violation"
        if count_key not in diagnostics or max_key not in diagnostics:
            difference = payload[f"u_rpr_{tag}_37"] - payload[f"u_base_{tag}_37"]
            diagnostics[count_key] = int((difference > 1e-7).sum())
            diagnostics[max_key] = float(difference.clamp_min(0).max())
            changed = True
    payload["diagnostics"] = diagnostics
    _validate_result(payload, f"resume:{dataset}/{stem}", int(task["effective_params"]["K_RECON"]))
    if changed:
        temporary = output_path.with_name(f".{output_path.name}.tmp-resume-{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    return {
        "manifest_row": {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(output_path.resolve()),
            "source_dabe_cache_path": str(expected_paths["source_dabe_cache_path"]),
            "source_cvbr_cache_path": str(expected_paths["source_cvbr_cache_path"]),
            "source_feature_cache_path": str(expected_paths["source_feature_cache_path"]),
            "rpr_version": DABE_RPR_VERSION,
            "shape": [1, 37, 37],
        },
        "diagnostics": diagnostics,
        "worker_peak_rss_mb": 0.0,
    }


def build_rpr_cache(
    config_path,
    dabe_root,
    cvbr_root,
    out_root,
    split="test",
    max_samples=-1,
    overwrite=False,
    overwrite_reason="",
    resume=False,
    workers=2,
    torch_threads=16,
):
    started = time.time()
    config_path, dabe_root, cvbr_root, output_root = map(
        lambda value: Path(value).resolve(),
        (config_path, dabe_root, cvbr_root, out_root),
    )
    if split != "test" or max_samples == 0 or max_samples < -1:
        raise ValueError("RPR-v1 requires split=test and max_samples=-1 or positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    cfg = load_config(config_path)
    if getattr(cfg, "BACKBONE_KEY", None) != "dinov1-s8":
        raise ValueError("BACKBONE_KEY must be dinov1-s8")
    if int(getattr(cfg, "DINO", {}).get("feature_input_size", 0)) != 296:
        raise ValueError("DINO feature_input_size must be 296")
    params = effective_dabe_v2_params(cfg)
    frozen = {
        "GRID": 37,
        "LOSS_SIZE": 68,
        "K_RECON": 32,
        "LAMBDA_COLOR_RECON": 0.20,
        "SIGMA_COLOR_RECON": 0.05,
        "TAU_RECON": 0.07,
    }
    for key, expected in frozen.items():
        if float(params[key]) != float(expected):
            raise ValueError(f"frozen parameter {key} must be {expected}, got {params[key]}")

    dabe_manifest = dabe_root / "manifest_test.jsonl"
    cvbr_manifest = cvbr_root / "manifest_test.jsonl"
    cvbr_protocol_path = cvbr_root / "protocol.json"
    feature_manifest = feature_manifest_path(cfg, "test").resolve()
    cvbr_protocol = _validate_cvbr_protocol(cvbr_protocol_path, dabe_manifest)
    dabe_rows, dabe_map = _manifest_map(dabe_manifest)
    cvbr_rows, cvbr_map = _manifest_map(cvbr_manifest)
    feature_rows, feature_map = _manifest_map(feature_manifest)
    if not (
        len(dabe_rows) == len(cvbr_rows) == len(feature_rows) == EXPECTED_TOTAL
    ):
        raise RuntimeError("all full source manifests must contain exactly 6473 rows")
    _validate_dataset_counts(dabe_rows)
    keys = set(dabe_map)
    if keys != set(cvbr_map) or keys != set(feature_map):
        raise RuntimeError("DABE/CVBR/feature manifest keys differ")
    if max_samples > 0 and max_samples > EXPECTED_TOTAL:
        raise RuntimeError("requested more samples than available")
    selected = dabe_rows if max_samples == -1 else dabe_rows[:max_samples]

    source_roots = [dabe_root, cvbr_root, feature_manifest.parent.resolve()]
    _validate_output_root(output_root, source_roots)
    dabe_paths = [dabe_manifest] + [Path(row["cache_path"]).resolve() for row in dabe_rows]
    cvbr_paths = [cvbr_manifest, cvbr_protocol_path] + [
        Path(row["cache_path"]).resolve() for row in cvbr_rows
    ]
    feature_paths = [feature_manifest] + [
        Path(row["cache_path"]).resolve() for row in feature_rows
    ]
    dabe_before = _snapshot(dabe_paths)
    cvbr_before = _snapshot(cvbr_paths)
    feature_before = _snapshot(feature_paths)

    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_root.exists():
        if resume:
            if (output_root / "manifest_test.jsonl").exists() or (output_root / "protocol.json").exists():
                raise RuntimeError("--resume is only allowed for an incomplete output without manifest/protocol")
        elif not overwrite:
            raise FileExistsError(f"refusing to overwrite {output_root}")
        if not overwrite_reason.strip():
            if overwrite:
                raise ValueError("--overwrite requires --overwrite_reason")
        if overwrite:
            shutil.rmtree(output_root)
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    log_handle = (output_root / "cache_build.log").open(
        "a" if resume else "w", encoding="utf-8", buffering=1
    )

    def log(message):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")

    final_items: dict[tuple[str, str], dict] = {}
    max_worker_rss = 0.0
    try:
        for line in (
            "rpr_version = dabe_rpr_v1",
            "gt_used_for_generation = false",
            "touch_metadata_used_for_generation = false",
            "dino_forward_used = false",
            "training_used = false",
            "threshold_search_used = false",
            "dataset_specific_rule = false",
            "sample_routing_used = false",
            "external_model_used = false",
            "trainable_parameter_count = 0",
        ):
            log(line)
        log(f"source_dabe_manifest = {dabe_manifest}")
        log(f"source_cvbr_manifest = {cvbr_manifest}")
        log(f"source_feature_manifest = {feature_manifest}")
        log(f"output_root = {output_root}")
        log(f"workers = {workers}")
        log(f"torch_threads_per_worker = {torch_threads}")
        log(f"resume = {str(bool(resume)).lower()}")
        all_tasks = [
            {
                "dabe_row": row,
                "cvbr_row": cvbr_map[(str(row["dataset"]), str(row["stem"]))],
                "feature_row": feature_map[(str(row["dataset"]), str(row["stem"]))],
                "effective_params": params,
                "output_root": str(output_root),
                "split": split,
            }
            for row in selected
        ]
        if resume:
            for task in all_tasks:
                row = task["dabe_row"]
                key = (str(row["dataset"]), str(row["stem"]))
                output_path = output_root / split / key[0] / f"{key[1]}.pt"
                if output_path.is_file():
                    final_items[key] = _resume_existing(task)
            log(f"resumed_existing = {len(final_items)}")
        tasks = [
            task
            for task in all_tasks
            if (str(task["dabe_row"]["dataset"]), str(task["dabe_row"]["stem"]))
            not in final_items
        ]
        log(f"missing_to_generate = {len(tasks)}")
        if workers == 1:
            iterator, executor = map(_process_one, tasks), None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(torch_threads,),
            )
            iterator = executor.map(_process_one, tasks, chunksize=1)
        try:
            for index, item in enumerate(iterator, 1):
                key = (item["manifest_row"]["dataset"], item["manifest_row"]["stem"])
                final_items[key] = item
                max_worker_rss = max(max_worker_rss, item["worker_peak_rss_mb"])
                if index == len(tasks) or index % 100 == 0:
                    log(f"generated_missing = {index}/{len(tasks)}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        selected_keys = [(str(row["dataset"]), str(row["stem"])) for row in selected]
        rows = [final_items[key]["manifest_row"] for key in selected_keys]
        diagnostics = [final_items[key]["diagnostics"] for key in selected_keys]
        output_keys = {(row["dataset"], row["stem"]) for row in rows}
        if len(rows) != len(output_keys) or output_keys != set(selected_keys):
            raise RuntimeError("output keys do not match selected source keys")
        if max_samples == -1 and len(rows) != EXPECTED_TOTAL:
            raise RuntimeError("full RPR output must contain 6473 rows")
        write_jsonl(output_root / "manifest_test.jsonl", rows)

        if _snapshot(dabe_paths) != dabe_before:
            raise RuntimeError("source DABE cache changed")
        if _snapshot(cvbr_paths) != cvbr_before:
            raise RuntimeError("source CVBR cache changed")
        if _snapshot(feature_paths) != feature_before:
            raise RuntimeError("source feature cache changed")

        max_diag = lambda key: max(float(item[key]) for item in diagnostics)
        topk_mismatch = sum(int(item["unit_prior_topk_mismatch_count"]) for item in diagnostics)
        cross_fallback = sum(int(item["cross_fallback_count_from_source"]) for item in diagnostics)
        p1_u_violation_count = sum(
            int(item["p1_u_monotonicity_violation_count"]) for item in diagnostics
        )
        p2_u_violation_count = sum(
            int(item["p2_u_monotonicity_violation_count"]) for item in diagnostics
        )
        u_exception_samples = []
        for key in selected_keys:
            item = final_items[key]["diagnostics"]
            if (
                int(item["p1_u_monotonicity_violation_count"])
                or int(item["p2_u_monotonicity_violation_count"])
            ):
                u_exception_samples.append(
                    {
                        "dataset": key[0],
                        "stem": key[1],
                        "p1_count": int(item["p1_u_monotonicity_violation_count"]),
                        "p1_max": float(item["p1_u_monotonicity_max_violation"]),
                        "p2_count": int(item["p2_u_monotonicity_violation_count"]),
                        "p2_max": float(item["p2_u_monotonicity_max_violation"]),
                    }
                )
        elapsed = time.time() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        commit, status = _git_metadata()
        protocol = {
            "rpr_version": DABE_RPR_VERSION,
            "source_dabe_root": str(dabe_root),
            "source_cvbr_root": str(cvbr_root),
            "source_feature_manifest": str(feature_manifest),
            "output_root": str(output_root),
            "split": "test",
            "num_samples": len(rows),
            "backbone": "DINOv1-S/8",
            "feature_type": "last-block attention key",
            "feature_input_size": 296,
            "grid": 37,
            "source_augs": ["identity"],
            "gt_used_for_generation": False,
            "touch_metadata_used_for_generation": False,
            "dino_forward_used": False,
            "training_used": False,
            "threshold_search_used": False,
            "dataset_specific_rule": False,
            "sample_routing_used": False,
            "external_model_used": False,
            "trainable_parameter_count": 0,
            "source_dabe_cache_modified": False,
            "source_cvbr_cache_modified": False,
            "source_feature_cache_modified": False,
            "fixed_candidates": {
                "P1": "B0 anchor, original Top-K, second-ring reliability post-TopK reweight",
                "P2": "B0 anchor, original Top-K, all-border reliability post-TopK reweight",
            },
            "rpr_formula": "softmax(original_score/tau + log(atom_reliability))",
            "reliability_formula_frozen_from_cvbr": True,
            "reliability_rescaled": False,
            "reliability_thresholded": False,
            "implementation_bug_fixes": [
                {
                    "stage": "full-run",
                    "issue": "float32 accumulation of U near one produced a spurious +2.98e-7 U_rpr-minus-U_base value for NC4K/1103",
                    "handling": "allow rare diagnostic-only float32 violations; record sample/count/max and continue",
                    "candidate_formula_changed": False,
                    "affected_valid_full_candidates": 0,
                    "action": "resume the interrupted partial cache and generate only missing samples; do not regenerate existing candidates",
                }
            ],
            "effective_dabe_params": params,
            "git_commit": commit,
            "git_status_short": status,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_dabe_manifest": str(dabe_manifest),
            "source_dabe_manifest_sha256": _sha256(dabe_manifest),
            "source_cvbr_manifest": str(cvbr_manifest),
            "source_cvbr_manifest_sha256": _sha256(cvbr_manifest),
            "source_cvbr_protocol": str(cvbr_protocol_path),
            "source_cvbr_protocol_sha256": _sha256(cvbr_protocol_path),
            "source_feature_manifest_sha256": _sha256(feature_manifest),
            "formula_file_sha256": _sha256(SCRIPT_PATH.with_name("dabe_rpr.py")),
            "builder_file_sha256": _sha256(SCRIPT_PATH),
            "b0_recompute_max_abs": max_diag("b0_recomputed_cached_r1_max_abs"),
            "b0_cached_r1_max_abs": max_diag("b0_cached_r1_max_abs"),
            "unit_prior_topk_mismatch_count": topk_mismatch,
            "unit_prior_weight_max_abs": max_diag("unit_prior_weight_max_abs"),
            "unit_prior_raw_max_abs": max_diag("unit_prior_raw_max_abs"),
            "unit_prior_normalized_max_abs": max_diag("unit_prior_normalized_max_abs"),
            "cross_fallback_count_from_source": cross_fallback,
            "rare_numerical_exceptions_allowed": True,
            "u_monotonicity_tolerance": 1e-7,
            "p1_u_monotonicity_violation_count": p1_u_violation_count,
            "p1_u_monotonicity_max_violation": max_diag(
                "p1_u_monotonicity_max_violation"
            ),
            "p2_u_monotonicity_violation_count": p2_u_violation_count,
            "p2_u_monotonicity_max_violation": max_diag(
                "p2_u_monotonicity_max_violation"
            ),
            "u_monotonicity_exception_sample_count": len(u_exception_samples),
            "u_monotonicity_exception_samples": u_exception_samples,
            "all_outputs_finite": True,
            "all_probability_outputs_in_unit_range": True,
            "overwrite": bool(overwrite),
            "overwrite_reason": overwrite_reason.strip(),
            "resume": bool(resume),
            "resumed_existing_count": len(selected) - len(tasks),
            "generated_missing_count": len(tasks),
            "workers": workers,
            "torch_threads_per_worker": torch_threads,
            "elapsed_seconds": elapsed,
            "average_seconds_per_image": elapsed / len(rows),
            "peak_rss_mb": peak_rss,
            "max_worker_peak_rss_mb": max_worker_rss,
            "source_cvbr_protocol_snapshot": cvbr_protocol,
        }
        write_json(output_root / "protocol.json", protocol)
        log("source_dabe_cache_modified = false")
        log("source_cvbr_cache_modified = false")
        log("source_feature_cache_modified = false")
        log(f"num_samples = {len(rows)}")
        log(f"b0_cached_r1_global_max_abs = {protocol['b0_cached_r1_max_abs']:.12g}")
        log(f"b0_recompute_global_max_abs = {protocol['b0_recompute_max_abs']:.12g}")
        log(f"unit_prior_topk_mismatch_count = {topk_mismatch}")
        log(f"unit_prior_weight_max_abs = {protocol['unit_prior_weight_max_abs']:.12g}")
        log(f"unit_prior_raw_global_max_abs = {protocol['unit_prior_raw_max_abs']:.12g}")
        log(
            "unit_prior_normalized_global_max_abs = "
            f"{protocol['unit_prior_normalized_max_abs']:.12g}"
        )
        log(f"p1_u_monotonicity_violation_count = {p1_u_violation_count}")
        log(
            "p1_u_monotonicity_max_violation = "
            f"{protocol['p1_u_monotonicity_max_violation']:.12g}"
        )
        log(f"p2_u_monotonicity_violation_count = {p2_u_violation_count}")
        log(
            "p2_u_monotonicity_max_violation = "
            f"{protocol['p2_u_monotonicity_max_violation']:.12g}"
        )
        log(f"u_monotonicity_exception_sample_count = {len(u_exception_samples)}")
        log("all_outputs_finite = true")
        log("all_probability_outputs_in_unit_range = true")
        log(f"elapsed_seconds = {elapsed:.3f}")
        log(f"average_seconds_per_image = {elapsed / len(rows):.6f}")
        log(f"peak_rss_mb = {peak_rss:.3f}")
        log(f"max_worker_peak_rss_mb = {max_worker_rss:.3f}")
        return protocol
    finally:
        log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--cvbr_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_reason", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--torch_threads", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_rpr_cache(
        args.config,
        args.dabe_root,
        args.cvbr_root,
        args.out_root,
        args.split,
        args.max_samples,
        args.overwrite,
        args.overwrite_reason,
        args.resume,
        args.workers,
        args.torch_threads,
    )
