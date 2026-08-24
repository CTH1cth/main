#!/usr/bin/env python3
"""KNN K-sweep on frozen matched-size oracle-clean dictionaries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_background_subspace import retrieve_background_neighbors  # noqa: E402
from models.reference_contamination import normalize_patch_features  # noqa: E402
from tools.analyze_reference_contamination_mechanism import (  # noqa: E402
    _stratified_bootstrap_fields,
)
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    DATASETS,
    EXPECTED_COUNTS,
    load_manifest,
    load_native_gt,
    load_torch,
    normalize_dataset,
    rank_metrics,
    resize_score_to_native,
    write_csv,
    write_json,
    write_jsonl,
)
from tools.run_reference_contamination_mechanism import _feature_from_payload  # noqa: E402


VERSION = "matched_oracle_knn_k_sweep_v1"
K_VALUES = (1, 4, 8, 16, 32)
SEEDS = (0, 1, 2)
METHODS = tuple(f"knn{k}" for k in K_VALUES) + ("gbsp_r8",)


class InvalidMatchedDictionary(RuntimeError):
    """Declared protocol exclusion inherited from the matched-size experiment."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260813)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _score_path(out_dir: Path, split: str, dataset: str, stem: str) -> Path:
    return out_dir / "scores" / split / dataset / f"{stem}.pt"


def _valid_score(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_torch(path)
        return (
            payload.get("version") == VERSION
            and tuple(payload.get("k_values", ())) == K_VALUES
            and tuple(sorted(int(seed) for seed in payload.get("seeds", {}))) == SEEDS
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _generate_one(row: dict, *, out_dir: Path, split: str, device: torch.device) -> dict:
    source = load_torch(row["cache_path"])
    dataset, stem = normalize_dataset(source["dataset"]), str(source["stem"])
    items = source.get("matched_size_clean", [])
    if tuple(sorted(int(item["seed"]) for item in items)) != SEEDS:
        raise RuntimeError("matched-size seeds are incomplete")
    if not all(bool(item.get("valid")) for item in items):
        reasons = sorted({str(item.get("reason", "")) for item in items if not item.get("valid")})
        raise InvalidMatchedDictionary("; ".join(reasons))
    feature_path = Path(source["source_feature_path"])
    feature_payload = load_torch(feature_path)
    features = normalize_patch_features(_feature_from_payload(feature_payload, feature_path)).to(device)
    seed_payload = {}
    max_k8_error = 0.0
    total_violations = 0
    for item in sorted(items, key=lambda value: int(value["seed"])):
        seed = int(item["seed"])
        candidate = torch.as_tensor(item["candidate_indices"], dtype=torch.long, device=device)
        retrieval = retrieve_background_neighbors(features, candidate, max_k=max(K_VALUES))
        similarities = retrieval.cosine_similarities.float()
        cumulative = similarities.cumsum(dim=1)
        scores = {
            f"knn{k}": (1.0 - cumulative[:, k - 1] / float(k)).detach().cpu().float().contiguous()
            for k in K_VALUES
        }
        reference = torch.as_tensor(item["knn8_score"]).detach().cpu().float().reshape(-1)
        max_k8_error = max(max_k8_error, float((scores["knn8"] - reference).abs().max()))
        total_violations += int(retrieval.self_match_violation_count)
        seed_payload[seed] = {
            "candidate_count": int(candidate.numel()),
            "scores": scores,
            "gbsp_native_metrics": item["native_metrics"]["gbsp"],
        }
    if max_k8_error > 1e-6:
        raise RuntimeError(f"KNN8 reproduction failed: {max_k8_error}")
    if total_violations:
        raise RuntimeError(f"strict leave-one-out failed: {total_violations}")
    output_path = _score_path(out_dir, split, dataset, stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    torch.save({
        "version": VERSION, "dataset": dataset, "stem": stem,
        "gt_path": source["gt_path"], "source_feature_path": str(feature_path),
        "source_matched_cache": row["cache_path"], "k_values": K_VALUES,
        "seeds": seed_payload, "strict_leave_one_out": True,
        "self_match_violation_count": total_violations,
        "knn8_reproduction_max_abs_error": max_k8_error,
        "gt_used_for_score_generation": False,
    }, temporary)
    os.replace(temporary, output_path)
    return {
        "dataset": dataset, "stem": stem, "score_path": str(output_path),
        "gt_path": source["gt_path"], "knn8_reproduction_max_abs_error": max_k8_error,
        "self_match_violation_count": total_violations,
    }


def _init_worker() -> None:
    torch.set_num_threads(1)


def _evaluate_one(row: dict) -> list[dict]:
    payload = load_torch(row["score_path"])
    gt = load_native_gt(payload["gt_path"])
    output = []
    for seed, item in sorted(payload["seeds"].items()):
        for k in K_VALUES:
            method = f"knn{k}"
            native = resize_score_to_native(item["scores"][method], tuple(gt.shape[-2:]))
            output.append({
                "dataset": payload["dataset"], "stem": payload["stem"],
                "seed": int(seed), "method": method,
                **rank_metrics(native, gt),
            })
        output.append({
            "dataset": payload["dataset"], "stem": payload["stem"],
            "seed": int(seed), "method": "gbsp_r8",
            "AP": float(item["gbsp_native_metrics"]["AP"]),
            "AUROC": float(item["gbsp_native_metrics"]["AUROC"]),
        })
    return output


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _seed_average(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["stem"], row["method"])].append(row)
    output = []
    for (dataset, stem, method), group in sorted(groups.items()):
        if tuple(sorted(int(row["seed"]) for row in group)) != SEEDS:
            continue
        output.append({
            "dataset": dataset, "stem": stem, "method": method,
            "AP": _mean(row["AP"] for row in group),
            "AUROC": _mean(row["AUROC"] for row in group),
        })
    return output


def _summary(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for method in METHODS:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if group:
                output.append({
                    "scope": "dataset", "dataset": dataset, "method": method,
                    "image_count": len(group), "AP": _mean(row["AP"] for row in group),
                    "AUROC": _mean(row["AUROC"] for row in group),
                })
    for method in METHODS:
        datasets = [row for row in output if row["scope"] == "dataset" and row["method"] == method]
        images = [row for row in rows if row["method"] == method]
        output.extend([
            {
                "scope": "dataset_macro", "dataset": "ALL", "method": method,
                "image_count": len(images), "AP": _mean(row["AP"] for row in datasets),
                "AUROC": _mean(row["AUROC"] for row in datasets),
            },
            {
                "scope": "image_macro", "dataset": "ALL", "method": method,
                "image_count": len(images), "AP": _mean(row["AP"] for row in images),
                "AUROC": _mean(row["AUROC"] for row in images),
            },
        ])
    return output


def _bootstrap(rows: list[dict], repetitions: int, seed: int) -> list[dict]:
    lookup = {(row["dataset"], row["stem"], row["method"]): row for row in rows}
    identities = sorted({(row["dataset"], row["stem"]) for row in rows})
    records = []
    fields = []
    for metric in ("AP", "AUROC"):
        for k in K_VALUES:
            fields.append(f"gbsp_minus_knn{k}_{metric}")
    for dataset, stem in identities:
        gbsp = lookup[(dataset, stem, "gbsp_r8")]
        record = {"dataset": dataset, "stem": stem}
        for metric in ("AP", "AUROC"):
            for k in K_VALUES:
                knn = lookup[(dataset, stem, f"knn{k}")]
                record[f"gbsp_minus_knn{k}_{metric}"] = float(gbsp[metric] - knn[metric])
        records.append(record)
    estimates = _stratified_bootstrap_fields(
        records, fields=tuple(fields), repetitions=int(repetitions), seed=int(seed),
    )
    output = []
    for metric in ("AP", "AUROC"):
        for k in K_VALUES:
            field = f"gbsp_minus_knn{k}_{metric}"
            output.append({
                "metric": metric, "k": k, "quantity": "gbsp_r8_minus_knn",
                "image_count": len(records), "repetitions": int(repetitions),
                **estimates[field],
            })
    return output


def _write_report(out_dir: Path, validity: dict, summary: list[dict], bootstrap: list[dict]) -> None:
    macros = [row for row in summary if row["scope"] == "dataset_macro"]
    lookup = {row["method"]: row for row in macros}
    lines = [
        "# Matched-size Oracle-clean KNN K Sweep", "",
        f"- 有效图像：{validity['valid_images']}",
        f"- K：{list(K_VALUES)}", "",
        "| 方法 | Dataset-macro AP | Dataset-macro AUROC |", "|---|---:|---:|",
    ]
    for method in METHODS:
        row = lookup[method]
        lines.append(f"| {method} | {row['AP']:.6f} | {row['AUROC']:.6f} |")
    lines.extend(["", "## GBSP-r8 − KNN 配对Bootstrap", "",
                  "| K | AP差值 [95% CI] | AUROC差值 [95% CI] |", "|---:|---:|---:|"])
    boot = {(row["metric"], int(row["k"])): row for row in bootstrap}
    for k in K_VALUES:
        ap, au = boot[("AP", k)], boot[("AUROC", k)]
        lines.append(
            f"| {k} | {ap['estimate']:+.6f} [{ap['ci95_low']:+.6f}, {ap['ci95_high']:+.6f}] | "
            f"{au['estimate']:+.6f} [{au['ci95_low']:+.6f}, {au['ci95_high']:+.6f}] |"
        )
    best_ap = max((lookup[f"knn{k}"]["AP"], k) for k in K_VALUES)
    best_au = max((lookup[f"knn{k}"]["AUROC"], k) for k in K_VALUES)
    lines.extend([
        "", "## 诊断性最佳K", "",
        f"- AP最高：K={best_ap[1]}，AP={best_ap[0]:.6f}",
        f"- AUROC最高：K={best_au[1]}，AUROC={best_au[0]:.6f}",
        "- 该最佳K来自测试集诊断，不能作为无泄漏超参数选择依据。", "",
        "> GT仅用于冻结的oracle-clean候选构造；KNN分数生成本身不读取GT。", "",
    ])
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    matched_root = Path(args.matched_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    source_rows = load_manifest(matched_root, split=args.split, max_samples=args.max_samples)
    generated, exclusions, failures = [], [], []
    for index, row in enumerate(source_rows, 1):
        path = _score_path(out_dir, args.split, normalize_dataset(row["dataset"]), row["stem"])
        if not args.overwrite and _valid_score(path):
            payload = load_torch(path)
            generated.append({
                "dataset": normalize_dataset(row["dataset"]), "stem": row["stem"],
                "score_path": str(path), "gt_path": payload["gt_path"],
                "knn8_reproduction_max_abs_error": payload["knn8_reproduction_max_abs_error"],
                "self_match_violation_count": payload["self_match_violation_count"],
            })
        else:
            try:
                generated.append(_generate_one(row, out_dir=out_dir, split=args.split, device=device))
            except InvalidMatchedDictionary as error:
                exclusions.append({
                    "dataset": normalize_dataset(row.get("dataset", "")),
                    "stem": row.get("stem", ""), "reason": str(error),
                })
            except Exception as error:
                failures.append({
                    "dataset": row.get("dataset", ""), "stem": row.get("stem", ""),
                    "error": repr(error), "traceback": traceback.format_exc(),
                })
        if index % max(1, args.progress_every) == 0 or index == len(source_rows):
            print(
                f"[{index}/{len(source_rows)}] scores={len(generated)} "
                f"excluded={len(exclusions)} failed={len(failures)}", flush=True,
            )
            write_jsonl(out_dir / "score_manifest_test.jsonl", generated)
            write_csv(out_dir / "excluded_invalid_dictionaries.csv", exclusions)
            write_json(out_dir / "generation_failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} K-sweep score generations failed")

    per_seed = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=context, initializer=_init_worker,
    ) as pool:
        for index, rows in enumerate(pool.map(_evaluate_one, generated, chunksize=1), 1):
            per_seed.extend(rows)
            if index % max(1, args.progress_every) == 0 or index == len(generated):
                print(f"[{index}/{len(generated)}] native metrics", flush=True)
    averaged = _seed_average(per_seed)
    summary = _summary(averaged)
    bootstrap = _bootstrap(averaged, args.bootstrap_repetitions, args.bootstrap_seed)
    counts = Counter(row["dataset"] for row in generated)
    max_error = max(float(row["knn8_reproduction_max_abs_error"]) for row in generated)
    violations = sum(int(row["self_match_violation_count"]) for row in generated)
    validity = {
        "version": VERSION, "requested_from_matched_manifest": len(source_rows),
        "valid_images": len(generated), "dataset_counts": dict(counts),
        "expected_full_counts": EXPECTED_COUNTS, "k_values": list(K_VALUES),
        "seeds": list(SEEDS), "generation_failed": len(failures),
        "declared_exclusions": len(exclusions),
        "self_match_violation_count": violations,
        "knn8_reproduction_max_abs_error": max_error,
        "full_matched_valid_set": len(generated) == 6452 and not failures and violations == 0,
        "gt_used_for_score_generation": False,
    }
    write_csv(out_dir / "per_seed_image_metrics.csv", per_seed)
    write_csv(out_dir / "seed_averaged_image_metrics.csv", averaged)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "paired_bootstrap.csv", bootstrap)
    write_json(out_dir / "validity_summary.json", validity)
    _write_report(out_dir, validity, summary, bootstrap)
    print(json.dumps({**validity, "output": str(out_dir)}, ensure_ascii=False), flush=True)
    if max_error > 1e-6 or violations:
        raise RuntimeError("formal reproduction/leave-one-out audit failed")


if __name__ == "__main__":
    main()
