#!/usr/bin/env python3
"""Collect frozen KNN8 and GBSP-r8 scores before using GT for grouping."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.common import (  # noqa: E402
    GRID,
    NUM_PATCHES,
    PROJECT_ROOT,
    extract_scores,
    labels_from_occupancy,
    load_core_rows,
    load_gt_occupancy,
    load_settings,
    load_torch,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    return parser.parse_args()


def _inside_workdir(path: Path) -> None:
    allowed = (PROJECT_ROOT / "workdir").resolve()
    try:
        path.resolve().relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"output must be under {allowed}: {path}") from error


def _select_pair(
    feature: torch.Tensor,
    occupancy: np.ndarray,
    fg_threshold: float,
    bg_threshold: float,
) -> tuple[int, int, float]:
    fg = np.flatnonzero(occupancy >= float(fg_threshold))
    bg = np.flatnonzero(occupancy <= float(bg_threshold))
    if not len(fg) or not len(bg):
        return -1, -1, float("nan")
    rr, cc = np.divmod(fg, GRID)
    fg_index = int(fg[np.argmin((rr - 18) ** 2 + (cc - 18) ** 2)])
    if feature.shape == (384, GRID, GRID):
        flat = feature.permute(1, 2, 0).reshape(NUM_PATCHES, 384)
    else:
        flat = feature.reshape(NUM_PATCHES, 384)
    flat = F.normalize(flat.float(), dim=1, eps=1e-12)
    similarity = flat[fg_index] @ flat[torch.as_tensor(bg)].T
    position = int(similarity.argmax())
    return fg_index, int(bg[position]), float(similarity[position])


def collect(args: argparse.Namespace) -> dict:
    settings = load_settings(args.config)
    out = Path(args.out_dir).resolve()
    _inside_workdir(out)
    datasets = tuple(args.datasets or settings.diagnostic_datasets)
    unknown = sorted(set(datasets) - set(settings.diagnostic_datasets))
    if unknown:
        raise ValueError(f"datasets are outside the frozen protocol: {unknown}")
    rows = [row for row in load_core_rows(settings) if row["dataset"] in datasets]
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for row in rows:
        dataset = row["dataset"]
        if args.max_samples_per_dataset >= 0 and counts[dataset] >= args.max_samples_per_dataset:
            continue
        selected.append(row)
        counts[dataset] += 1
    if not selected:
        raise RuntimeError("no samples selected")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    generated, failures, image_audit = [], [], []
    candidate_mismatches = 0
    max_minmax_error = 0.0
    self_match_violations = 0
    for number, row in enumerate(selected, 1):
        score_path = out / "scores" / row["dataset"] / f"{row['stem']}.npz"
        try:
            core = load_torch(row["cache_path"])
            if core.get("settings", {}).get("background_source") != "fullbc":
                raise RuntimeError("core cache is not Full-BC")
            r8 = core["results"]["r8"]
            if int(r8["selected_rank"]) != settings.gbsp_rank:
                raise RuntimeError("core cache does not use frozen rank=8")
            # Score generation is completed before GT is loaded below.
            scores = extract_scores(
                core, device, knn_k=settings.knn_k, gbsp_rank=settings.gbsp_rank,
            )
            expected_indices = torch.as_tensor(r8["background_indices"]).numpy()
            if not np.array_equal(scores["background_indices"], expected_indices):
                candidate_mismatches += 1
                raise RuntimeError("KNN8 and GBSP candidate indices differ")
            occupancy = load_gt_occupancy(row["gt_path"])
            core_label, binary_label, core_valid = labels_from_occupancy(occupancy, settings)
            pair_fg, pair_bg, pair_similarity = _select_pair(
                scores.pop("feature"), occupancy,
                settings.fg_core_threshold, settings.bg_core_threshold,
            )
            patch_index = np.arange(NUM_PATCHES, dtype=np.int16)
            patch_row, patch_col = np.divmod(patch_index, GRID)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                score_path,
                image_id=np.asarray(f"{row['dataset']}/{row['stem']}"),
                dataset=np.asarray(row["dataset"]), stem=np.asarray(row["stem"]),
                image_path=np.asarray(row["image_path"]), gt_path=np.asarray(row["gt_path"]),
                patch_index=patch_index, patch_row=patch_row.astype(np.int8), patch_col=patch_col.astype(np.int8),
                gt_occupancy=occupancy.astype(np.float32), core_label=core_label,
                core_valid=core_valid.astype(np.uint8), binary_label=binary_label,
                knn8_background_similarity_raw=scores["knn_bg_raw"].astype(np.float32),
                knn8_foreground_score_raw=scores["knn_fg_raw"].astype(np.float32),
                knn8_foreground_score_normalized=scores["knn_fg_normalized"].astype(np.float32),
                gbsp_raw_score=scores["gbsp_raw"].astype(np.float32),
                gbsp_normalized_score=scores["gbsp_normalized"].astype(np.float32),
                background_candidate_flag=scores["background_candidate"].astype(np.uint8),
                background_indices=scores["background_indices"].astype(np.int16),
                pair_fg_index=np.asarray(pair_fg, dtype=np.int16),
                pair_bg_index=np.asarray(pair_bg, dtype=np.int16),
                pair_cosine_similarity=np.asarray(pair_similarity, dtype=np.float32),
            )
            self_match_violations += int(scores["self_match_violation_count"])
            max_minmax_error = max(max_minmax_error, float(scores["gbsp_minmax_max_abs_error"]))
            generated.append({
                "dataset": row["dataset"], "stem": row["stem"],
                "image_path": row["image_path"], "gt_path": row["gt_path"],
                "core_path": row["cache_path"], "score_path": str(score_path),
            })
            image_audit.append({
                "dataset": row["dataset"], "stem": row["stem"],
                "num_background": int(scores["num_background"]),
                "selected_rank": int(r8["selected_rank"]),
                "self_match_violation_count": int(scores["self_match_violation_count"]),
                "candidate_indices_exact": True,
                "gbsp_minmax_max_abs_error": float(scores["gbsp_minmax_max_abs_error"]),
            })
        except Exception as error:
            failures.append({"dataset": row.get("dataset"), "stem": row.get("stem"), "error": repr(error)})
            if args.failure_policy == "strict":
                raise
        if number % 20 == 0 or number == len(selected):
            print(f"[{number}/{len(selected)}] generated={len(generated)} failed={len(failures)}", flush=True)
    write_jsonl(out / "score_manifest.jsonl", generated)
    write_jsonl(out / "audit" / "per_image_score_audit.jsonl", image_audit)
    summary = {
        "requested": len(selected), "generated": len(generated), "failed": len(failures),
        "counts": dict(Counter(row["dataset"] for row in generated)),
        "failures": failures, "knn_k": settings.knn_k, "gbsp_rank": settings.gbsp_rank,
        "hard_threshold": settings.hard_threshold,
        "self_match_violation_count": self_match_violations,
        "candidate_index_mismatch_count": candidate_mismatches,
        "gbsp_minmax_max_abs_error": max_minmax_error,
        "gt_used_for_score_generation": False,
        "gt_used_after_generation_for_grouping": True,
    }
    write_json(out / "audit" / "score_collection_summary.json", summary)
    audit = f"""# KNN Baseline Audit

- Source file: `main/models/gbsp_similarity_baselines.py`
- Production function: `compute_similarity_baselines`
- Frozen K: `{settings.knn_k}` (read from `{settings.official_similarity_config}`)
- Candidate set: the exact `results.r8.background_indices` from the frozen Full-BC GBSP cache
- Candidate equality: exact integer-index assertion for every generated image
- Feature normalization: per-patch L2 normalization in the production function
- Similarity: cosine similarity implemented by dot product after L2 normalization
- Self-match: excluded for every query that is itself a Full-BC candidate
- Aggregation: arithmetic mean of the top-{settings.knn_k} background cosine similarities
- Raw background-oriented score: `mean(top-{settings.knn_k} cosine)`
- Raw foreground-oriented score: `1 - mean(top-{settings.knn_k} cosine)`
- Normalization: production per-image Min-Max over all 1369 query patches
- Score direction in all figures: larger means more foreground-like
- Self-match violations observed: `{self_match_violations}`
- Candidate-index mismatches observed: `{candidate_mismatches}`

The teaser does not scan K and does not change the production self-exclusion rule.
"""
    (out / "audit" / "KNN_BASELINE_AUDIT.md").write_text(audit, encoding="utf-8")
    pipeline = f"""# Score Pipeline Audit

- GBSP source: frozen `{settings.core_root}` cache, `Full-BC`, fixed `r8`
- GBSP raw score: squared affine-PCA projection residual
- GBSP normalized score: cached per-image Min-Max response; maximum reproduction error `{max_minmax_error:.3e}`
- KNN8 and GBSP use identical background candidate indices (asserted per image)
- GT protocol: original GT -> nearest-neighbor 296x296 -> non-overlapping 8x8 average pooling -> 37x37 occupancy
- Core patches: background `<= {settings.bg_core_threshold}`, foreground `>= {settings.fg_core_threshold}`; mixed patches ignored
- All-patch diagnostic: foreground `>= {settings.binary_threshold}`, background `< {settings.binary_threshold}`
- GT is loaded only after both score maps have been produced
- No separate foreground/background normalization is performed
- Official hard-mask visualization: normalized 37x37 -> bilinear 68x68 -> strict `>{settings.hard_threshold}`
"""
    (out / "audit" / "SCORE_PIPELINE_AUDIT.md").write_text(pipeline, encoding="utf-8")
    return summary


def main() -> None:
    summary = collect(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
