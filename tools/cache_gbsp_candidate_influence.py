#!/usr/bin/env python3
"""Generate Stage-A proxies and Stage-B exact LOO candidate diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config  # noqa: E402
from models.gbsp_candidate_influence import (  # noqa: E402
    candidate_influence_proxy,
    exact_loo_batch,
    fit_scatter_basis,
    projector_distance_from_bases,
    projector_frobenius_error,
)
from tools.gbsp_candidate_influence_common import (  # noqa: E402
    DATASETS,
    RANK,
    atomic_npz,
    build_indices,
    high_is_one_percentile,
    load_aligned,
    load_torch,
    nearest_patch_labels,
    normalize_dataset,
    npz_path,
    oracle_items,
    parse_subset,
    resolve_path,
    stable_seed,
    top_mask,
    write_json,
)
from tools.gbsp_knn_lsr_common import write_csv  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_candidate_influence.py"
VERSION = "gbsp_candidate_influence_v1"
PROXY_VERSION = "gbsp_candidate_influence_proxy_v1"
EXACT_VERSION = "gbsp_candidate_exact_loo_v1"
_WORKER_R0_INDEX: dict | None = None
_WORKER_PROXY_ROOT: Path | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def oracle_status(row: dict | None, expected_seeds: tuple[int, ...]) -> tuple[bool, str]:
    if row is None:
        return False, "manifest_missing"
    try:
        payload = load_torch(row["cache_path"])
        items = payload.get("matched_size_clean", [])
        if tuple(sorted(int(item.get("seed", -1)) for item in items)) != expected_seeds:
            return False, "seed_mismatch"
        for item in items:
            if not bool(item.get("valid")) or not torch.is_tensor(item.get("candidate_indices")):
                return False, f"invalid_seed_{item.get('seed')}"
        return True, "valid"
    except Exception as error:  # audit records rare cache failures without aborting all rows
        return False, repr(error)


def run_audit(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    audit_root = output / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    core_rows, r0_index, oracle_index = build_indices(cfg)
    expected = dict(cfg.GBSP_INFLUENCE_EXPECTED_COUNTS)
    observed = Counter(row["dataset"] for row in core_rows)
    if len(core_rows) != sum(expected.values()) or dict(observed) != expected:
        raise RuntimeError(f"formal core counts mismatch: {dict(observed)}")
    if set(r0_index) != {(row["dataset"], row["stem"]) for row in core_rows}:
        raise RuntimeError("R0 and core identity sets differ")

    alignment, candidate_counts, cvbr_counts, failures = [], [], [], []
    expected_seeds = tuple(int(v) for v in cfg.GBSP_INFLUENCE_ORACLE_SEEDS)
    for number, row in enumerate(core_rows, 1):
        dataset, stem = row["dataset"], row["stem"]
        try:
            core, r0, _, indices = load_aligned(row, r0_index)
            cvbr_path = Path(r0["source_cvbr_path"])
            cvbr = load_torch(cvbr_path)
            anchor = torch.as_tensor(cvbr["anchor_b0_37"]).reshape(-1) > 0.5
            valid_grid = torch.as_tensor(cvbr["border_ring2_only_37"]).reshape(-1) > 0.5
            if not torch.equal(torch.where(anchor)[0], indices):
                raise RuntimeError("CVBR anchor and Full-BC indices differ")
            cvbr_count = int(valid_grid.index_select(0, indices).sum())
            if cvbr_count != int(cfg.GBSP_INFLUENCE_CVBR_SCORED_PER_IMAGE):
                raise RuntimeError(f"CVBR coverage is {cvbr_count}, expected 136")
            if Path(core["source_feature_path"]).resolve() != Path(r0["source_feature_path"]).resolve():
                raise RuntimeError("core/R0 feature path mismatch")
            oracle_valid, reason = oracle_status(oracle_index.get((dataset, stem)), expected_seeds)
            alignment.append({
                "dataset": dataset, "stem": stem, "candidate_count": int(indices.numel()),
                "cvbr_scored_count": cvbr_count, "feature_dimension": 384, "rank": 8,
                "coordinate_index_aligned": True, "feature_aligned": True,
                "oracle_valid": oracle_valid, "oracle_reason": reason,
            })
            candidate_counts.append(int(indices.numel()))
            cvbr_counts.append(cvbr_count)
        except Exception as error:
            failures.append({"dataset": dataset, "stem": stem, "error": repr(error)})
        if number % 200 == 0 or number == len(core_rows):
            print(f"[{number}/{len(core_rows)}] audit failures={len(failures)}", flush=True)
    if failures:
        write_json(audit_root / "failures.json", failures)
        raise RuntimeError(f"alignment audit failed for {len(failures)} images")
    with (audit_root / "alignment.jsonl").open("w", encoding="utf-8") as handle:
        for row in alignment:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid_count = sum(bool(row["oracle_valid"]) for row in alignment)
    invalid_reasons = Counter(row["oracle_reason"] for row in alignment if not row["oracle_valid"])
    protocol = {
        "version": VERSION, "created_at": now(), "total_images": len(alignment),
        "dataset_counts": dict(observed), "oracle_valid_images": valid_count,
        "oracle_invalid_images": len(alignment) - valid_count,
        "oracle_invalid_reasons": dict(invalid_reasons),
        "candidate_count": {
            "min": min(candidate_counts), "mean": float(np.mean(candidate_counts)),
            "median": float(np.median(candidate_counts)), "max": max(candidate_counts),
            "total": int(sum(candidate_counts)),
        },
        "feature_dimension": 384, "rank": 8,
        "cvbr_coverage": {
            "scored_per_image": int(np.median(cvbr_counts)),
            "total_scored": int(sum(cvbr_counts)),
            "ratio_of_fullbc": float(sum(cvbr_counts) / sum(candidate_counts)),
            "scope": "existing validated Second-Ring candidates only",
        },
        "coordinate_index_alignment": "PASS",
        "feature_path_alignment": "PASS",
        "oracle_protocol": "three-seed matched-size oracle-clean; distances/H are seed-averaged",
        "gt_patch_rule": str(cfg.GBSP_INFLUENCE_GT_PATCH_RULE),
        "gt_used_for_deployable_generation": False,
    }
    write_json(audit_root / "audit.json", protocol)
    lines = [
        "# Candidate Influence 数据对齐审计", "",
        f"- 总图像：{protocol['total_images']}",
        f"- Oracle 严格有效：{valid_count}",
        f"- Oracle 无效/缺失：{len(alignment)-valid_count}",
        f"- 数据集计数：{dict(observed)}", "",
        "## Full-BC / CVBR", "",
        f"- Full-BC candidate 总数：{sum(candidate_counts):,}",
        f"- 每图 candidate：min={min(candidate_counts)}, mean={np.mean(candidate_counts):.3f}, median={np.median(candidate_counts):.1f}, max={max(candidate_counts)}",
        f"- CVBR 已评分 candidate：每图 {int(np.median(cvbr_counts))}，总计 {sum(cvbr_counts):,}，占 Full-BC {sum(cvbr_counts)/sum(candidate_counts):.4%}",
        "- CVBR 分析口径：仅既有 validated Second-Ring；未评分候选不填成低风险。", "",
        "## 表征与 Oracle", "",
        "- DINO feature：37×37，384 维，L2 normalized。",
        "- Global PCA：固定 rank=8。",
        "- candidate coordinate/index 对齐：PASS。",
        "- core/R0 feature source 对齐：PASS。",
        "- Oracle：matched-size clean，seed 0/1/2；projector distance 与 H 对三种子取均值。",
        f"- Oracle 无效原因：{dict(invalid_reasons)}", "",
        "> GT 只用于 contamination/H 机制诊断，不进入可部署推理或伪标签生成。", "",
    ]
    (audit_root / "INFLUENCE_DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def _init_proxy_worker(r0_index: dict, proxy_root: str, torch_threads: int) -> None:
    global _WORKER_R0_INDEX, _WORKER_PROXY_ROOT
    _WORKER_R0_INDEX = r0_index
    _WORKER_PROXY_ROOT = Path(proxy_root)
    torch.set_num_threads(int(torch_threads))


def _proxy_one(task: dict) -> dict:
    assert _WORKER_R0_INDEX is not None and _WORKER_PROXY_ROOT is not None
    dataset, stem = task["dataset"], task["stem"]
    path = npz_path(_WORKER_PROXY_ROOT, dataset, stem)
    try:
        core, r0, feature, indices = load_aligned(task, _WORKER_R0_INDEX)
        _, gt_label = nearest_patch_labels(r0["gt_path"])
        labels = gt_label.index_select(0, indices).numpy().astype(np.uint8)
        background = feature.index_select(0, indices)
        mean = torch.as_tensor(r0["mean"]).float()
        basis = torch.as_tensor(r0["basis"]).float()
        singular = torch.as_tensor(r0["singular_values"]).float()
        proxy = candidate_influence_proxy(background, mean, basis, singular)
        cvbr_score = torch.as_tensor(r0["cvbr_score"]).float().numpy()
        cvbr_valid = torch.as_tensor(r0["cvbr_valid_mask"]).bool().numpy()
        cvbr_percentile = torch.as_tensor(r0["cvbr_percentile"]).float().numpy()
        self_residual = torch.as_tensor(r0["vanilla_pca_self_residual"]).float().numpy()
        raw = proxy.raw.numpy()
        gap = proxy.gap_normalized.numpy()
        influence_percentile = high_is_one_percentile(raw)
        arrays = {
            "version": np.asarray(PROXY_VERSION), "dataset": np.asarray(dataset),
            "image_id": np.asarray(stem), "candidate_index": indices.numpy().astype(np.int16),
            "patch_y": torch.div(indices, 37, rounding_mode="floor").numpy().astype(np.int16),
            "patch_x": (indices % 37).numpy().astype(np.int16),
            "is_gt_foreground": labels, "cvbr_score": cvbr_score.astype(np.float32),
            "cvbr_valid_mask": cvbr_valid.astype(np.uint8),
            "cvbr_percentile": cvbr_percentile.astype(np.float32),
            "pca_self_residual": self_residual.astype(np.float32),
            "in_subspace_energy": proxy.in_subspace_energy.numpy().astype(np.float32),
            "orthogonal_energy": proxy.orthogonal_energy.numpy().astype(np.float32),
            "influence_proxy_raw": raw.astype(np.float32),
            "influence_proxy_gap": gap.astype(np.float32),
            "influence_percentile": influence_percentile.astype(np.float32),
            "spectral_gap": np.asarray(proxy.spectral_gap, dtype=np.float64),
            "candidate_feature_norm": torch.linalg.vector_norm(background, dim=1).numpy().astype(np.float32),
        }
        atomic_npz(path, **arrays)
        valid = cvbr_valid
        fg = labels.astype(bool)
        high_cvbr = top_mask(cvbr_score, 0.10, valid)
        high_influence = top_mask(raw, 0.10)
        joint = high_cvbr & high_influence
        return {
            "dataset": dataset, "stem": stem, "proxy_path": str(path),
            "num_candidates": int(indices.numel()), "num_cvbr_scored": int(valid.sum()),
            "num_foreground": int(fg.sum()), "contamination_ratio": float(fg.mean()),
            "spectral_gap": float(proxy.spectral_gap),
            "proxy_raw_mean": float(np.mean(raw)), "proxy_gap_mean": float(np.mean(gap)),
            "cvbr_proxy_raw_spearman_all": safe_spearman(cvbr_score[valid], raw[valid]),
            "cvbr_proxy_raw_spearman_bg": safe_spearman(cvbr_score[valid & ~fg], raw[valid & ~fg]),
            "cvbr_proxy_raw_spearman_fg": safe_spearman(cvbr_score[valid & fg], raw[valid & fg]),
            "cvbr_top10_count": int(high_cvbr.sum()), "cvbr_top10_fg": int(fg[high_cvbr].sum()),
            "joint_top10_count": int(joint.sum()), "joint_top10_fg": int(fg[joint].sum()),
            "oracle_valid": bool(task["oracle_valid"]), "oracle_reason": task["oracle_reason"],
            "gt_path": str(r0["gt_path"]), "feature_path": str(r0["source_feature_path"]),
        }
    except Exception as error:
        return {"dataset": dataset, "stem": stem, "error": repr(error), "traceback": traceback.format_exc()}


def run_proxy(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    proxy_root = output / "proxy" / "candidate_influence_proxy"
    core_rows, r0_index, oracle_index = build_indices(cfg)
    alignment_path = output / "audit" / "alignment.jsonl"
    if not alignment_path.is_file():
        raise FileNotFoundError("run audit before proxy generation")
    aligned = {
        (row["dataset"], row["stem"]): row
        for row in (json.loads(line) for line in alignment_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    rows = core_rows if int(args.max_samples) < 0 else core_rows[: int(args.max_samples)]
    tasks = []
    for row in rows:
        meta = aligned[(row["dataset"], row["stem"])]
        tasks.append({**row, "oracle_valid": meta["oracle_valid"], "oracle_reason": meta["oracle_reason"]})
    results = []
    with ProcessPoolExecutor(
        max_workers=int(args.workers), initializer=_init_proxy_worker,
        initargs=(r0_index, str(proxy_root), int(args.torch_threads)),
    ) as pool:
        for number, result in enumerate(pool.map(_proxy_one, tasks, chunksize=4), 1):
            results.append(result)
            if number % 100 == 0 or number == len(tasks):
                print(f"[{number}/{len(tasks)}] proxy failed={sum('error' in row for row in results)}", flush=True)
    failures = [row for row in results if "error" in row]
    if failures:
        write_json(output / "proxy" / "failures.json", failures)
        raise RuntimeError(f"proxy generation failed for {len(failures)} images")
    write_csv(output / "proxy" / "influence_proxy_summary.csv", results)
    protocol = {
        "version": PROXY_VERSION, "created_at": now(), "requested": len(tasks),
        "generated": len(results), "failed": 0, "full_input": len(tasks) == 6473,
        "gt_used_for_analysis_labels": True, "gt_used_for_score_generation": False,
        "cvbr_scope": "validated Second-Ring only", "rank": 8,
    }
    write_json(output / "proxy" / "generation_summary.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def assign_strata(rows: list[dict]) -> None:
    positive = sorted(
        [row for row in rows if float(row["contamination_ratio"]) > 0],
        key=lambda row: (float(row["contamination_ratio"]), row["dataset"], row["stem"]),
    )
    chunks = np.array_split(np.arange(len(positive)), 3)
    lookup = {}
    for name, chunk in zip(("Low", "Medium", "High"), chunks):
        for index in chunk.tolist():
            lookup[(positive[index]["dataset"], positive[index]["stem"])] = name
    for row in rows:
        row["stratum"] = "Zero" if float(row["contamination_ratio"]) == 0 else lookup[(row["dataset"], row["stem"])]


def run_select(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    summary_path = output / "proxy" / "influence_proxy_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if str(row["oracle_valid"]).lower() == "true"]
    assign_strata(rows)
    target = int(args.subset_size or cfg.GBSP_INFLUENCE_SUBSET_SIZE)
    seed = int(cfg.GBSP_INFLUENCE_SUBSET_SEED)
    rng = np.random.default_rng(seed)
    selected = []
    if bool(cfg.GBSP_INFLUENCE_INCLUDE_ALL_CHAMELEON):
        selected.extend(row for row in rows if row["dataset"] == "CHAMELEON")
    if bool(cfg.GBSP_INFLUENCE_INCLUDE_ALL_CAMO):
        selected.extend(row for row in rows if row["dataset"] == "CAMO")
    selected_keys = {(row["dataset"], row["stem"]) for row in selected}
    remaining_count = target - len(selected)
    if remaining_count < 0:
        raise RuntimeError("subset target is smaller than mandatory CHAMELEON+CAMO set")
    current_contaminated = sum(float(row["contamination_ratio"]) > 0 for row in selected)
    needed_contaminated = max(
        0, int(math.ceil(float(cfg.GBSP_INFLUENCE_SUBSET_MIN_CONTAMINATED_RATIO) * target)) - current_contaminated,
    )
    pool = [
        row for row in rows
        if row["dataset"] in {"COD10K", "NC4K"} and (row["dataset"], row["stem"]) not in selected_keys
    ]
    positive_pool = [row for row in pool if float(row["contamination_ratio"]) > 0]
    zero_pool = [row for row in pool if float(row["contamination_ratio"]) == 0]
    if needed_contaminated > remaining_count or len(positive_pool) < needed_contaminated:
        raise RuntimeError("cannot satisfy fixed contaminated-image ratio")

    def sample_balanced(source: list[dict], count: int, fields: tuple[str, ...]) -> list[dict]:
        groups = defaultdict(list)
        for item in source:
            groups[tuple(item[field] for field in fields)].append(item)
        for values in groups.values():
            rng.shuffle(values)
        chosen = []
        keys = sorted(groups)
        while len(chosen) < count:
            progressed = False
            for key in keys:
                if groups[key] and len(chosen) < count:
                    chosen.append(groups[key].pop())
                    progressed = True
            if not progressed:
                break
        if len(chosen) != count:
            raise RuntimeError(f"balanced pool only supplied {len(chosen)}/{count}")
        return chosen

    selected.extend(sample_balanced(positive_pool, needed_contaminated, ("dataset", "stratum")))
    used = {(row["dataset"], row["stem"]) for row in selected}
    fill_pool = [row for row in pool if (row["dataset"], row["stem"]) not in used]
    # Prefer zero images for the remaining slots so the fixed subset retains a
    # meaningful clean-image mechanism control; then fill only if necessary.
    zero_fill = [row for row in zero_pool if (row["dataset"], row["stem"]) not in used]
    fill_count = target - len(selected)
    selected.extend(sample_balanced(zero_fill, min(fill_count, len(zero_fill)), ("dataset",)))
    used = {(row["dataset"], row["stem"]) for row in selected}
    if len(selected) < target:
        other = [row for row in fill_pool if (row["dataset"], row["stem"]) not in used]
        selected.extend(sample_balanced(other, target - len(selected), ("dataset", "stratum")))
    selected.sort(key=lambda row: (DATASETS.index(row["dataset"]), row["stem"]))
    contaminated = sum(float(row["contamination_ratio"]) > 0 for row in selected)
    if len(selected) != target or contaminated / target < float(cfg.GBSP_INFLUENCE_SUBSET_MIN_CONTAMINATED_RATIO):
        raise RuntimeError("fixed exact subset does not satisfy its preregistered constraints")
    subset_path = output / "exact_loo" / "exact_loo_subset.txt"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# dataset\tstem\tcontamination_stratum\tcontamination_ratio"] + [
        f"{row['dataset']}\t{row['stem']}\t{row['stratum']}\t{float(row['contamination_ratio']):.12g}"
        for row in selected
    ]
    subset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    protocol = {
        "version": VERSION, "created_at": now(), "seed": seed, "subset_size": target,
        "contaminated_images": contaminated, "contaminated_ratio": contaminated / target,
        "dataset_counts": dict(Counter(row["dataset"] for row in selected)),
        "stratum_counts": dict(Counter(row["stratum"] for row in selected)),
        "selection_frozen_before_exact_loo": True,
        "all_oracle_valid": all(str(row["oracle_valid"]).lower() == "true" for row in selected),
    }
    write_json(output / "exact_loo" / "subset_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def run_validate(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    core_rows, r0_index, _ = build_indices(cfg)
    count_images = int(args.validation_images or cfg.GBSP_INFLUENCE_VALIDATION_IMAGES)
    count_candidates = int(args.validation_candidates or cfg.GBSP_INFLUENCE_VALIDATION_CANDIDATES)
    rng = np.random.default_rng(int(cfg.GBSP_INFLUENCE_VALIDATION_SEED))
    chosen = rng.choice(len(core_rows), size=count_images, replace=False)
    device = torch.device(args.device)
    records = []
    for number, row_index in enumerate(chosen.tolist(), 1):
        row = core_rows[row_index]
        _, r0, feature, indices = load_aligned(row, r0_index)
        # Float64 is required here: squaring the spectrum in a float32 scatter
        # matrix caused visible rank-boundary disagreement with literal SVD.
        background = feature.index_select(0, indices).to(device=device, dtype=torch.float64)
        _, original = fit_scatter_basis(background, RANK)
        local_rng = np.random.default_rng(stable_seed(row["dataset"], row["stem"], int(cfg.GBSP_INFLUENCE_VALIDATION_SEED)))
        candidate = np.sort(local_rng.choice(indices.numel(), size=count_candidates, replace=False))
        candidate_tensor = torch.as_tensor(candidate, device=device)
        fast = exact_loo_batch(background, candidate_tensor, original, original.unsqueeze(0))
        reduced = []
        for index in candidate:
            keep = torch.ones(background.shape[0], dtype=torch.bool, device=device)
            keep[int(index)] = False
            reduced.append(background[keep])
        reduced = torch.stack(reduced)
        brute_mean = reduced.mean(1)
        _, _, brute_vh = torch.linalg.svd(reduced - brute_mean.unsqueeze(1), full_matrices=False)
        brute_basis = brute_vh[:, :RANK].transpose(1, 2).contiguous()
        projector_error = projector_frobenius_error(fast.basis, brute_basis).detach().cpu().numpy()
        mean_error = (fast.mean - brute_mean).abs().amax(1).detach().cpu().numpy()
        for local, candidate_id in enumerate(candidate.tolist()):
            records.append({
                "dataset": row["dataset"], "stem": row["stem"], "candidate_position": candidate_id,
                "projector_frobenius_error": float(projector_error[local]),
                "mean_max_abs_error": float(mean_error[local]),
            })
        if number % 10 == 0 or number == count_images:
            print(f"[{number}/{count_images}] exact-LOO validation", flush=True)
    maximum_projector = max(row["projector_frobenius_error"] for row in records)
    maximum_mean = max(row["mean_max_abs_error"] for row in records)
    protocol = {
        "version": VERSION, "created_at": now(), "images": count_images,
        "candidates_per_image": count_candidates, "comparisons": len(records),
        "device": str(device), "dtype": "float64 fast scatter / float64 literal SVD",
        "max_projector_frobenius_error": maximum_projector,
        "max_mean_abs_error": maximum_mean,
        "projector_tolerance": float(cfg.GBSP_INFLUENCE_PROJECTOR_TOLERANCE),
        "mean_tolerance": float(cfg.GBSP_INFLUENCE_MEAN_TOLERANCE),
        "pass": maximum_projector < float(cfg.GBSP_INFLUENCE_PROJECTOR_TOLERANCE)
        and maximum_mean < float(cfg.GBSP_INFLUENCE_MEAN_TOLERANCE),
        "reference": "literal delete + recenter + full SVD",
    }
    write_csv(output / "audit" / "exact_loo_correctness.csv", records)
    write_json(output / "audit" / "exact_loo_correctness.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    if not protocol["pass"]:
        raise RuntimeError("exact LOO correctness gate failed; formal exact analysis is forbidden")


def valid_exact(path: Path, candidate_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return str(payload["version"].item()) == EXACT_VERSION and payload["loo_projector_influence"].size == candidate_count
    except Exception:
        return False


def run_exact(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    gate = json.loads((output / "audit" / "exact_loo_correctness.json").read_text(encoding="utf-8"))
    if not gate.get("pass") or int(gate.get("images", 0)) < int(cfg.GBSP_INFLUENCE_VALIDATION_IMAGES):
        raise RuntimeError("formal 100x10 exact-LOO correctness gate has not passed")
    subset = parse_subset(output / "exact_loo" / "exact_loo_subset.txt")
    core_rows, r0_index, oracle_index = build_indices(cfg)
    core_index = {(row["dataset"], row["stem"]): row for row in core_rows}
    exact_root = output / "exact_loo" / "candidate_exact_loo"
    proxy_root = output / "proxy" / "candidate_influence_proxy"
    device = torch.device(args.device)
    batch_size = int(args.batch_size)
    summaries, failures = [], []
    started = time.perf_counter()
    for number, item in enumerate(subset, 1):
        dataset, stem = item["dataset"], item["stem"]
        out_path = npz_path(exact_root, dataset, stem)
        try:
            row = core_index[(dataset, stem)]
            core, r0, feature, indices = load_aligned(row, r0_index)
            if valid_exact(out_path, int(indices.numel())):
                with np.load(out_path, allow_pickle=False) as cached:
                    summaries.append({
                        "dataset": dataset, "stem": stem, "stratum": item["stratum"],
                        "contamination_ratio": item["contamination_ratio"], "num_candidates": int(indices.numel()),
                        "oracle_seed_count": int(cached["oracle_seed_count"].item()),
                        "baseline_oracle_distance": float(cached["baseline_oracle_distance"].item()),
                        "mean_exact_influence": float(np.mean(cached["loo_projector_influence"])),
                        "mean_harmful_improvement": float(np.mean(cached["harmful_oracle_improvement"])),
                        "resumed": True,
                    })
                    continue
            oracle, status = oracle_items(
                oracle_index.get((dataset, stem)), feature, int(indices.numel()),
                tuple(int(v) for v in cfg.GBSP_INFLUENCE_ORACLE_SEEDS),
                device=device, dtype=torch.float64,
            )
            if not oracle:
                raise RuntimeError(f"subset contains invalid oracle image: {status}")
            background = feature.index_select(0, indices).to(device=device, dtype=torch.float64)
            _, original = fit_scatter_basis(background, RANK)
            oracle_tensor = torch.stack(oracle).to(device=device, dtype=torch.float64)
            before = float(torch.stack([
                projector_distance_from_bases(original, basis) for basis in oracle_tensor
            ]).mean().cpu())
            exact_values, after_values, harmful_values = [], [], []
            for start in range(0, indices.numel(), batch_size):
                positions = torch.arange(start, min(start + batch_size, indices.numel()), device=device)
                result = exact_loo_batch(background, positions, original, oracle_tensor)
                exact_values.append(result.influence.detach().cpu())
                after_values.append(result.oracle_distance_after.detach().cpu())
                harmful_values.append(result.harmful_oracle_improvement.detach().cpu())
            exact = torch.cat(exact_values).numpy()
            after = torch.cat(after_values).numpy()
            harmful = torch.cat(harmful_values).numpy()
            with np.load(npz_path(proxy_root, dataset, stem), allow_pickle=False) as source:
                arrays = {key: np.asarray(source[key]) for key in source.files}
            arrays.update({
                "version": np.asarray(EXACT_VERSION),
                "loo_projector_influence": exact.astype(np.float32),
                "exact_influence_percentile": high_is_one_percentile(exact).astype(np.float32),
                "baseline_oracle_distance": np.asarray(before, dtype=np.float64),
                "oracle_distance_after_removal": after.astype(np.float32),
                "harmful_oracle_improvement": harmful.astype(np.float32),
                "abs_harmful_oracle_improvement": np.abs(harmful).astype(np.float32),
                "oracle_seed_count": np.asarray(len(oracle), dtype=np.int16),
                "oracle_protocol": np.asarray("mean over matched-size clean seeds 0/1/2"),
            })
            atomic_npz(out_path, **arrays)
            summaries.append({
                "dataset": dataset, "stem": stem, "stratum": item["stratum"],
                "contamination_ratio": item["contamination_ratio"], "num_candidates": int(indices.numel()),
                "oracle_seed_count": len(oracle), "baseline_oracle_distance": before,
                "mean_exact_influence": float(np.mean(exact)),
                "mean_harmful_improvement": float(np.mean(harmful)), "resumed": False,
            })
        except Exception as error:
            failures.append({"dataset": dataset, "stem": stem, "error": repr(error), "traceback": traceback.format_exc()})
        elapsed = time.perf_counter() - started
        if number % 5 == 0 or number == len(subset):
            rate = number / max(elapsed, 1e-9)
            eta = (len(subset) - number) / max(rate, 1e-9)
            print(f"[{number}/{len(subset)}] exact valid={len(summaries)} failed={len(failures)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    write_json(output / "exact_loo" / "failures.json", failures)
    if failures:
        raise RuntimeError(f"exact LOO failed for {len(failures)} subset images")
    write_csv(output / "exact_loo" / "exact_loo_summary.csv", summaries)
    protocol = {
        "version": EXACT_VERSION, "created_at": now(), "requested": len(subset),
        "generated": len(summaries), "failed": 0, "device": str(device),
        "batch_size": batch_size, "wall_seconds": time.perf_counter() - started,
        "gt_used_for_harmful_diagnostic": True, "gt_used_for_deployable_generation": False,
    }
    write_json(output / "exact_loo" / "generation_summary.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "validate", "proxy", "select", "exact"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_root")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--validation_images", type=int)
    parser.add_argument("--validation_candidates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))
    if args.command == "audit":
        run_audit(args, cfg)
    elif args.command == "validate":
        run_validate(args, cfg)
    elif args.command == "proxy":
        run_proxy(args, cfg)
    elif args.command == "select":
        run_select(args, cfg)
    elif args.command == "exact":
        run_exact(args, cfg)


if __name__ == "__main__":
    main()
