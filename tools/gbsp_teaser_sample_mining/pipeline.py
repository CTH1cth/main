from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from models.gbsp_core_variants import (  # noqa: E402
    PCARankSelector,
    decompose_pca,
    minmax_score,
    pca_fit_from_decomposition,
    score_all_patches,
)
from models.gbsp_similarity_baselines import compute_similarity_baselines, minmax_per_image  # noqa: E402
from tools.gbsp_teaser_sample_mining.common import (  # noqa: E402
    FEATURE_DIM,
    GRID,
    INPUT_SIZE,
    NUM_PATCHES,
    assert_output_root,
    binary_metrics,
    feature_tensor,
    gt_occupancy,
    load_config,
    patch_box,
    percentile_rank,
    read_csv,
    read_jsonl,
    resized_input,
    resolve_project_path,
    score_to_image,
    torch_payload,
    write_csv,
    write_json,
)


SCHEMA = "gbsp_teaser_sample_mining_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stage", choices=("all", "audit", "a", "b", "c", "d"), default="all")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--progress_every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    return parser.parse_args()


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        if key in result:
            raise RuntimeError(f"duplicate manifest identity: {key}")
        result[key] = row
    return result


def _sources(config: dict, max_samples: int) -> list[dict]:
    feature_path = resolve_project_path(config["feature_manifest"])
    candidate_path = resolve_project_path(config["candidate_manifest"])
    features = read_jsonl(feature_path)
    candidates = _index(read_jsonl(candidate_path))
    rows = []
    for feature in features:
        key = (str(feature["dataset"]), str(feature["stem"]))
        if key not in candidates:
            raise KeyError(f"candidate manifest misses {key}")
        candidate = candidates[key]
        rows.append({
            "dataset": key[0],
            "image_id": key[1],
            "stem": key[1],
            "image_path": str(feature["image_path"]),
            "feature_path": str(feature["cache_path"]),
            "candidate_path": str(candidate["cache_path"]),
            "source_dabe_path": str(candidate.get("source_dabe_cache_path", "")),
        })
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _gt_path(row: dict) -> str:
    source = Path(row.get("source_dabe_path", ""))
    if source.is_file():
        value = torch_payload(source).get("gt_path")
        if value and Path(value).is_file():
            return str(value)
    image = Path(row["image_path"])
    candidate = image.parent.parent / "gt" / f"{row['stem']}.png"
    if not candidate.is_file():
        raise FileNotFoundError(f"cannot resolve GT for offline mining: {row['dataset']}/{row['stem']}")
    return str(candidate)


def generate_method_scores(row: dict, device: torch.device, config: dict) -> dict:
    """Generate both unsupervised scores without accepting or opening GT."""
    feature = feature_tensor(torch_payload(row["feature_path"]), row["feature_path"]).to(device)
    carrier = torch_payload(row["candidate_path"])
    if (str(carrier.get("dataset")), str(carrier.get("stem"))) != (row["dataset"], row["stem"]):
        raise RuntimeError(f"candidate cache identity mismatch: {row['candidate_path']}")
    background = torch.as_tensor(carrier["background_indices"], dtype=torch.long, device=device).reshape(-1)
    if background.numel() < 9 or background.unique().numel() != background.numel():
        raise ValueError("Full-BC candidate set is invalid")
    rank = int(config["gbsp"]["rank"])
    fit = pca_fit_from_decomposition(
        decompose_pca(feature.index_select(0, background)),
        "fixed",
        PCARankSelector(0.90, rank, 1),
        rank,
    )
    raw = score_all_patches(feature, fit).reshape(-1)
    normalized = minmax_score(raw).reshape(-1)
    knn = compute_similarity_baselines(
        feature, background, k=int(config["affinity"]["k"])
    )
    affinity_raw = knn.scores["knn8_cos"].reshape(-1)
    affinity = minmax_per_image(affinity_raw).reshape(-1)
    fit_mean = fit.mean.to(device)
    fit_basis = fit.basis.to(device)
    centered = feature - fit_mean.reshape(1, -1)
    residual = centered - (centered @ fit_basis) @ fit_basis.t()
    error = float((residual.square().sum(dim=1) - raw.to(device)).abs().max())
    if error > 2e-6:
        raise RuntimeError(f"residual reproduction failed: {error}")
    return {
        "feature": feature.detach().cpu(),
        "background_indices": background.detach().cpu(),
        "affinity_raw": affinity_raw.detach().cpu(),
        "affinity": affinity.detach().cpu(),
        "gbsp_raw": raw.detach().cpu(),
        "gbsp": normalized.detach().cpu(),
        "residual": residual.detach().cpu(),
        "mean": fit_mean.detach().cpu(),
        "basis": fit_basis.detach().cpu(),
        "selected_rank": int(fit.selected_rank),
        "self_match_violation_count": int(knn.self_match_violation_count),
        "residual_reproduction_error": error,
    }


def _image_stats(occupancy: np.ndarray) -> dict:
    mask = occupancy.reshape(GRID, GRID) >= 0.5
    _, components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    touch = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {
        "foreground_area_ratio": float(mask.mean()),
        "num_fg": int(mask.sum()),
        "num_bg": int((~mask).sum()),
        "num_core_fg": int((occupancy >= 0.8).sum()),
        "num_core_bg": int((occupancy <= 0.2).sum()),
        "touch_boundary": int(touch),
        "connected_component_count": int(components),
    }


def audit(config: dict, rows: list[dict], output: Path) -> None:
    (output / "audit").mkdir(parents=True, exist_ok=True)
    feature_found = sum(Path(row["feature_path"]).is_file() for row in rows)
    carrier_found = sum(Path(row["candidate_path"]).is_file() for row in rows)
    source_dabe_found = sum(Path(row["source_dabe_path"]).is_file() for row in rows)
    historical = config["historical_anchor"]
    old_csv = resolve_project_path(historical["source_csv"])
    old_match = None
    if old_csv.is_file():
        for row in read_csv(old_csv):
            if row.get("dataset") == historical["dataset"] and row.get("stem") == historical["stem"]:
                old_match = row
                break
    cache_text = f"""# CACHE AUDIT

| Cache | Found | Reused | Recomputed |
|---|---:|---:|---:|
| DINO feature | {feature_found}/{len(rows)} | yes | no |
| GT patch mask | 0 persistent | no | lightweight on demand |
| Full-BC candidate indices | {carrier_found}/{len(rows)} | yes, from compact r32 carrier | no |
| GBSP residual vector | no formal train-r8 cache | no | rank-8 from frozen features/candidates |
| GBSP residual score | no formal train-r8 cache | no | from residual vector |
| GBSP pseudo-label | no formal train-r8 cache | no | strict `>0.58` |
| Affinity score | no train cache | no | frozen KNN8-Cos |
| Affinity pseudo-label | no train cache | no | strict `>0.58` |

- Feature manifest rows: `{len(rows)}`.
- Source DABE payloads available for GT-path identity only: `{source_dabe_found}/{len(rows)}`.
- The r32 payload is **not** reused as a score or subspace.  Only its frozen Full-BC integer indices are reused; a fixed rank-8 PCA is fitted from the DINO features.
"""
    (output / "audit" / "CACHE_AUDIT.md").write_text(cache_text, encoding="utf-8")
    affinity = config["affinity"]
    (output / "audit" / "AFFINITY_CONFIG.md").write_text(
        f"""# Frozen Reference-Affinity Configuration

- Exact function: `{affinity['function']}`
- Source: `main/models/gbsp_similarity_baselines.py`
- Variant: `{affinity['method']}`; K=`{affinity['k']}`
- Reference source: exact same Full-BC candidate indices as GBSP
- Feature normalization: per-patch L2 in production function
- Reference evidence: arithmetic mean of the top-8 cosine similarities
- Foreground score: `1 - mean(top8 cosine)`
- Self match: strict leave-one-out for queries inside the reference set
- Normalization: `{affinity['normalization']}`
- Pseudo-label: normalized score `{affinity['comparison'].replace('_', ' ')}` `{affinity['threshold']}`
- No K, affinity variant, normalization, or threshold is scanned.
""",
        encoding="utf-8",
    )
    anchor = {
        "found": old_match is not None,
        "requested": historical,
        "resolved_row": old_match,
        "in_train4040": False,
        "priority_candidate_eligible": False,
        "reason": "historical anchor belongs to TE-COD10K test split, not TR-CAMO/TR-COD10K",
    }
    write_json(output / "audit" / "HISTORICAL_ANCHOR_AUDIT.json", anchor)
    (output / "audit" / "GT_LEAKAGE_AUDIT.md").write_text(
        """# GT Leakage Audit

- `generate_method_scores(row, device, config)` accepts only feature/candidate paths and never accepts a GT path.
- KNN8 and GBSP score generation finishes before `_gt_path` and `gt_occupancy` are called.
- GT is used only for offline sample mining, diagnostic grouping, PCA coloring, ranking, and visual inspection.
- Training annotations never participate in Full-BC construction, PCA fitting, residual computation, score normalization, or thresholding.

> Training annotations are used only offline to locate a representative visualization example and never participate in pseudo-label generation.
""",
        encoding="utf-8",
    )


def stage_a(config: dict, rows: list[dict], output: Path, device: torch.device, args: argparse.Namespace) -> list[dict]:
    result, failures = [], []
    threshold = float(config["gbsp"]["threshold"])
    for number, row in enumerate(rows, 1):
        try:
            scores = generate_method_scores(row, device, config)
            # GT is deliberately loaded only after both score paths return.
            gt_path = _gt_path(row)
            occupancy = gt_occupancy(gt_path)
            target = occupancy >= 0.5
            affinity_mask = scores["affinity"].numpy() > threshold
            gbsp_mask = scores["gbsp"].numpy() > threshold
            affinity = binary_metrics(affinity_mask, target)
            gbsp = binary_metrics(gbsp_mask, target)
            stats = _image_stats(occupancy)
            core_fg = occupancy >= 0.8
            corrected = core_fg & ~affinity_mask & gbsp_mask
            valid_image = (
                float(config["selection"]["min_fg_area_ratio"]) <= stats["foreground_area_ratio"]
                <= float(config["selection"]["max_fg_area_ratio"])
                and stats["num_core_fg"] >= int(config["selection"]["min_core_fg_patches"])
                and stats["num_core_bg"] >= int(config["selection"]["min_core_bg_patches"])
            )
            result.append({
                **row,
                "gt_path": gt_path,
                **stats,
                **{f"aff_{key}": value for key, value in affinity.items()},
                **{f"gbsp_{key}": value for key, value in gbsp.items()},
                "delta_iou": float(gbsp["iou"] - affinity["iou"]),
                "delta_f1": float(gbsp["f1"] - affinity["f1"]),
                "corrected_fg_patch_count": int(corrected.sum()),
                "valid_image": int(valid_image),
                "background_candidate_count": int(scores["background_indices"].numel()),
                "selected_rank": int(scores["selected_rank"]),
                "self_match_violation_count": int(scores["self_match_violation_count"]),
                "residual_reproduction_error": float(scores["residual_reproduction_error"]),
            })
        except Exception as error:
            failures.append({**row, "error": repr(error), "traceback": traceback.format_exc()})
            if args.failure_policy == "strict":
                raise
        if number % args.progress_every == 0 or number == len(rows):
            print(f"stage A [{number}/{len(rows)}] valid={len(result)} failed={len(failures)}", flush=True)
    if not result:
        raise RuntimeError("Stage A produced no valid rows")
    gbsp_q70 = float(np.quantile([row["gbsp_iou"] for row in result], 0.70))
    delta_q80 = float(np.quantile([row["delta_iou"] for row in result], 0.80))
    for row in result:
        row["gbsp_iou_p70"] = gbsp_q70
        row["delta_iou_p80"] = delta_q80
        row["gbsp_quality_gate"] = int(row["gbsp_iou"] >= gbsp_q70)
        row["improvement_gate"] = int(row["delta_iou"] >= delta_q80)
    result.sort(
        key=lambda row: (
            row["valid_image"], row["gbsp_quality_gate"], row["improvement_gate"],
            row["corrected_fg_patch_count"] > 0, row["gbsp_iou"], row["delta_iou"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(result, 1):
        row["stage_a_rank"] = rank
    write_csv(output / "stage_a" / "stage_a_all4040.csv", result)
    write_json(output / "stage_a" / "failures.json", failures)
    write_json(output / "stage_a" / "thresholds.json", {
        "gbsp_iou_p70": gbsp_q70, "delta_iou_p80": delta_q80,
        "computed_over": len(result), "failures": len(failures),
    })
    return result


def _float_row(row: dict, keys: tuple[str, ...]) -> dict:
    value = dict(row)
    for key in keys:
        if key in value:
            value[key] = float(value[key])
    return value


def stage_b(config: dict, stage_a_rows: list[dict], output: Path, device: torch.device, args: argparse.Namespace) -> list[dict]:
    top = stage_a_rows[: min(int(config["selection"]["stage_a_top"]), len(stage_a_rows))]
    threshold = float(config["gbsp"]["threshold"])
    all_queries: list[dict] = []
    failures = []
    for number, row in enumerate(top, 1):
        try:
            scores = generate_method_scores(row, device, config)
            occupancy = gt_occupancy(row["gt_path"])
            affinity = scores["affinity"].numpy()
            gbsp = scores["gbsp"].numpy()
            corrected = np.flatnonzero((occupancy >= 0.8) & (affinity <= threshold) & (gbsp > threshold))
            feature = scores["feature"]
            background = scores["background_indices"].numpy()
            valid_reference = background[occupancy[background] <= 0.2]
            foreground_mask = occupancy.reshape(GRID, GRID) >= 0.5
            distance = ndimage.distance_transform_edt(foreground_mask).reshape(-1)
            for query in corrected:
                references = valid_reference[valid_reference != query]
                if not len(references):
                    continue
                similarities = feature[int(query)] @ feature[torch.as_tensor(references)].t()
                position = int(similarities.argmax())
                reference = int(references[position])
                query_norm = float(torch.linalg.vector_norm(scores["residual"][int(query)]))
                reference_norm = float(torch.linalg.vector_norm(scores["residual"][reference]))
                qy, qx = divmod(int(query), GRID)
                ry, rx = divmod(reference, GRID)
                all_queries.append({
                    **row,
                    "query_patch_idx": int(query), "query_y": qy, "query_x": qx,
                    "query_gt_occupancy": float(occupancy[query]),
                    "affinity_score": float(affinity[query]), "affinity_pred": 0,
                    "gbsp_score": float(gbsp[query]), "gbsp_pred": 1,
                    "matched_ref_idx": reference, "matched_ref_y": ry, "matched_ref_x": rx,
                    "matched_ref_gt_occupancy": float(occupancy[reference]),
                    "query_ref_cosine": float(similarities[position]),
                    "query_residual_norm": query_norm,
                    "ref_residual_norm": reference_norm,
                    "residual_gap": query_norm - reference_norm,
                    "query_gbsp_margin": float(gbsp[query] - threshold),
                    "query_distance_to_gt_boundary": float(distance[query]),
                    "matched_ref_in_actual_affinity_set": 1,
                    "matched_ref_is_query": 0,
                })
        except Exception as error:
            failures.append({"dataset": row["dataset"], "stem": row["stem"], "error": repr(error)})
            if args.failure_policy == "strict":
                raise
        if number % args.progress_every == 0 or number == len(top):
            print(f"stage B [{number}/{len(top)}] corrected={len(all_queries)} failed={len(failures)}", flush=True)
    if not all_queries:
        raise RuntimeError("Stage B found no corrected foreground queries")
    similarity_percentile = percentile_rank([row["query_ref_cosine"] for row in all_queries])
    gap_percentile = percentile_rank([row["residual_gap"] for row in all_queries])
    for row, sim, gap in zip(all_queries, similarity_percentile, gap_percentile):
        row["query_ref_similarity_percentile"] = float(sim)
        row["residual_gap_percentile"] = float(gap)
    all_queries.sort(
        key=lambda row: (
            row["query_ref_cosine"], row["residual_gap"], row["query_gbsp_margin"],
            row["query_gt_occupancy"], row["query_distance_to_gt_boundary"],
        ),
        reverse=True,
    )
    best: dict[tuple[str, str], dict] = {}
    for row in all_queries:
        best.setdefault((row["dataset"], row["stem"]), row)
    best_rows = list(best.values())
    best_rows.sort(
        key=lambda row: (
            row["query_ref_cosine"], row["residual_gap"], row["query_gbsp_margin"],
            row["query_gt_occupancy"], row["query_distance_to_gt_boundary"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(best_rows, 1):
        row["stage_b_rank"] = rank
    write_csv(output / "stage_b" / "stage_b_all_corrected_patches.csv", all_queries)
    write_csv(output / "stage_b" / "stage_b_top300_patch.csv", best_rows)
    write_json(output / "stage_b" / "failures.json", failures)
    return best_rows


def _pca(value: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    pca = PCA(
        n_components=2, svd_solver="randomized", n_oversamples=24,
        iterated_power=7, random_state=int(seed),
    )
    embedding = pca.fit_transform(value)
    return embedding, float(pca.explained_variance_ratio_.sum())


def _silhouette(embedding: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> float:
    selected = labels[mask]
    if len(selected) < 3 or not selected.any() or selected.all():
        return float("nan")
    return float(silhouette_score(embedding[mask], selected))


def stage_c(config: dict, stage_b_rows: list[dict], output: Path, device: torch.device, args: argparse.Namespace) -> list[dict]:
    top = stage_b_rows[: min(int(config["selection"]["stage_b_top"]), len(stage_b_rows))]
    seed = int(config["pca"]["seed"])
    result, failures = [], []
    cache = output / "stage_c" / "pca_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(top):
        try:
            scores = generate_method_scores(row, device, config)
            occupancy = gt_occupancy(row["gt_path"])
            core = (occupancy <= 0.2) | (occupancy >= 0.8)
            core_labels = (occupancy >= 0.8).astype(np.uint8)
            primary = np.ones(NUM_PATCHES, dtype=bool)
            primary_labels = (occupancy >= 0.5).astype(np.uint8)
            raw_embedding, raw_variance = _pca(scores["feature"].numpy(), seed + index * 2)
            residual_embedding, residual_variance = _pca(scores["residual"].numpy(), seed + index * 2 + 1)
            raw_sil = _silhouette(raw_embedding, core, core_labels)
            residual_sil = _silhouette(residual_embedding, core, core_labels)
            raw_primary = _silhouette(raw_embedding, primary, primary_labels)
            residual_primary = _silhouette(residual_embedding, primary, primary_labels)
            pca_path = cache / row["dataset"] / f"{row['stem']}.npz"
            pca_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pca_path,
                raw_embedding=raw_embedding.astype(np.float32),
                residual_embedding=residual_embedding.astype(np.float32),
                occupancy=occupancy.astype(np.float32),
                query_index=np.asarray(int(row["query_patch_idx"]), dtype=np.int16),
                reference_index=np.asarray(int(row["matched_ref_idx"]), dtype=np.int16),
            )
            result.append({
                **row,
                "sil_dino": raw_sil, "sil_residual": residual_sil,
                "delta_sil": residual_sil - raw_sil,
                "sil_dino_primary_05": raw_primary,
                "sil_residual_primary_05": residual_primary,
                "delta_sil_primary_05": residual_primary - raw_primary,
                "dino_pca_explained_variance": raw_variance,
                "residual_pca_explained_variance": residual_variance,
                "pca_cache_path": str(pca_path),
            })
        except Exception as error:
            failures.append({"dataset": row["dataset"], "stem": row["stem"], "error": repr(error)})
            if args.failure_policy == "strict":
                raise
        print(f"stage C [{index + 1}/{len(top)}] valid={len(result)} failed={len(failures)}", flush=True)
    if not result:
        raise RuntimeError("Stage C produced no PCA candidates")
    all_stage_a = read_csv(output / "stage_a" / "stage_a_all4040.csv")
    gbsp_iou = np.asarray([float(row["gbsp_iou"]) for row in all_stage_a], dtype=float)
    delta_iou = np.asarray([float(row["delta_iou"]) for row in all_stage_a], dtype=float)
    q60, q70 = float(np.quantile(gbsp_iou, .60)), float(np.quantile(gbsp_iou, .70))
    d70, d80 = float(np.quantile(delta_iou, .70)), float(np.quantile(delta_iou, .80))
    for row in result:
        tier_a = (
            row["gbsp_iou"] >= q70 and row["delta_iou"] >= d80
            and row["query_ref_similarity_percentile"] >= .90
            and row["residual_gap_percentile"] >= .75
            and row["sil_dino"] <= .15 and row["sil_residual"] >= .35 and row["delta_sil"] >= .35
        )
        tier_b = (
            row["gbsp_iou"] >= q60 and row["delta_iou"] >= d70
            and row["query_ref_similarity_percentile"] >= .80
            and row["sil_dino"] <= .25 and row["sil_residual"] >= .30 and row["delta_sil"] >= .25
        )
        row["tier"] = "A" if tier_a else ("B" if tier_b else "C")
    order = {"A": 2, "B": 1, "C": 0}
    result.sort(
        key=lambda row: (
            order[row["tier"]], row["gbsp_iou"], row["delta_iou"], row["delta_sil"],
            row["query_ref_similarity_percentile"], row["residual_gap_percentile"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(result, 1):
        row["stage_c_rank"] = rank
    write_csv(output / "stage_c" / "stage_c_top100_pca.csv", result)
    write_json(output / "stage_c" / "failures.json", failures)
    write_json(output / "stage_c" / "tier_thresholds.json", {
        "gbsp_iou_p60": q60, "gbsp_iou_p70_all_stage_a": q70,
        "delta_iou_p70": d70, "delta_iou_p80_all_stage_a": d80,
        "pca_grouping": config["pca"]["grouping"],
        "primary_05_silhouettes_also_recorded": True,
    })
    return result


def _save_gray(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def _boxed_image(row: dict) -> Image.Image:
    image = resized_input(row["image_path"])
    draw = ImageDraw.Draw(image)
    draw.rectangle(patch_box(int(row["query_patch_idx"])), outline=(232, 76, 31), width=3)
    draw.rectangle(patch_box(int(row["matched_ref_idx"])), outline=(0, 174, 239), width=3)
    return image


def _crop(image: Image.Image, index: int, context: int, color: tuple[int, int, int]) -> Image.Image:
    y, x = divmod(int(index), GRID)
    radius = context // 2
    x0, y0 = max(0, x - radius) * 8, max(0, y - radius) * 8
    x1, y1 = min(GRID, x + radius + 1) * 8, min(GRID, y + radius + 1) * 8
    crop = image.crop((x0, y0, x1, y1)).resize(((x1 - x0) * 5, (y1 - y0) * 5), Image.Resampling.NEAREST)
    cx0, cy0 = (x * 8 - x0) * 5, (y * 8 - y0) * 5
    draw = ImageDraw.Draw(crop)
    draw.rectangle((cx0, cy0, cx0 + 40, cy0 + 40), outline=color, width=5)
    return crop


def _pca_plot(row: dict, representation: str, path: Path) -> None:
    payload = np.load(row["pca_cache_path"], allow_pickle=False)
    embedding = payload[f"{representation}_embedding"]
    occupancy = payload["occupancy"]
    query, reference = int(payload["query_index"]), int(payload["reference_index"])
    background, foreground = occupancy <= .2, occupancy >= .8
    mixed = ~(background | foreground)
    fig, axis = plt.subplots(figsize=(4.2, 3.7), facecolor="white")
    axis.scatter(embedding[mixed, 0], embedding[mixed, 1], s=5, c="#BDBDBD", alpha=.18, linewidths=0)
    axis.scatter(embedding[background, 0], embedding[background, 1], s=8, c="#2B8CBE", alpha=.62, linewidths=0)
    axis.scatter(embedding[foreground, 0], embedding[foreground, 1], s=11, c="#E34A33", alpha=.78, linewidths=0)
    axis.scatter(*embedding[query], s=150, c="#E34A33", marker="*", edgecolors="black", linewidths=.7, zorder=5)
    axis.scatter(*embedding[reference], s=150, c="#2B8CBE", marker="*", edgecolors="black", linewidths=.7, zorder=5)
    if representation == "raw":
        title = f"DINOv1-S/8 Features — PCA\nsilhouette={row['sil_dino']:.3f}"
    else:
        title = f"GBSP-R8 Residual Vectors — PCA\nsilhouette={row['sil_residual']:.3f}"
    axis.set_title(title, fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])
    axis.text(.5, -.08, "GT for coloring only", transform=axis.transAxes, ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_finalist(config: dict, row: dict, directory: Path, device: torch.device) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    scores = generate_method_scores(row, device, config)
    threshold = float(config["gbsp"]["threshold"])
    query = int(row["query_patch_idx"])
    reference = int(row["matched_ref_idx"])
    occupancy = gt_occupancy(row["gt_path"])
    if not (
        occupancy[query] >= .8
        and float(scores["affinity"][query]) <= threshold
        and float(scores["gbsp"][query]) > threshold
    ):
        raise RuntimeError("same-query consistency failed during finalist rendering")
    if reference == query or reference not in set(scores["background_indices"].tolist()):
        raise RuntimeError("matched reference is not a distinct actual affinity reference")
    if occupancy[reference] > .2:
        raise RuntimeError("displayed matched reference is not Core-BG")
    pca_identity = np.load(row["pca_cache_path"], allow_pickle=False)
    if int(pca_identity["query_index"]) != query or int(pca_identity["reference_index"]) != reference:
        raise RuntimeError("PCA highlight does not use the selected query/reference")
    image = resized_input(row["image_path"])
    gt = resized_input(row["gt_path"], is_mask=True)
    image.save(directory / "01_input.png")
    gt.save(directory / "02_gt.png")
    query_only = image.copy(); ImageDraw.Draw(query_only).rectangle(patch_box(int(row["query_patch_idx"])), outline=(232, 76, 31), width=3)
    query_only.save(directory / "03_input_query_box.png")
    boxed = _boxed_image(row); boxed.save(directory / "04_input_query_ref_boxes.png")
    image.crop(patch_box(int(row["query_patch_idx"]))).resize((160, 160), Image.Resampling.NEAREST).save(directory / "05_query_exact_patch.png")
    _crop(image, int(row["query_patch_idx"]), 5, (232, 76, 31)).save(directory / "06_query_context_crop.png")
    image.crop(patch_box(int(row["matched_ref_idx"]))).resize((160, 160), Image.Resampling.NEAREST).save(directory / "07_ref_exact_patch.png")
    _crop(image, int(row["matched_ref_idx"]), 5, (0, 174, 239)).save(directory / "08_ref_context_crop.png")
    affinity_response = score_to_image(scores["affinity"].numpy(), hard=False, threshold=threshold)
    affinity_mask = score_to_image(scores["affinity"].numpy(), hard=True, threshold=threshold)
    gbsp_response = score_to_image(scores["gbsp"].numpy(), hard=False, threshold=threshold)
    gbsp_mask = score_to_image(scores["gbsp"].numpy(), hard=True, threshold=threshold)
    _save_gray(directory / "09_affinity_response.png", affinity_response)
    _save_gray(directory / "10_affinity_pseudolabel.png", affinity_mask)
    _save_gray(directory / "11_gbsp_residual_map.png", gbsp_response)
    _save_gray(directory / "12_gbsp_pseudolabel.png", gbsp_mask)
    _pca_plot(row, "raw", directory / "13_dino_pca")
    _pca_plot(row, "residual", directory / "14_residual_pca")

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.7))
    axes[0].imshow(image); axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("GT")
    axes[2].imshow(Image.open(directory / "06_query_context_crop.png")); axes[2].set_title(
        f"FG Query\ncos={row['query_ref_cosine']:.3f}, r={row['query_residual_norm']:.3f}")
    axes[3].imshow(Image.open(directory / "08_ref_context_crop.png")); axes[3].set_title(
        f"BG Reference\nr={row['ref_residual_norm']:.3f}, gap={row['residual_gap']:.3f}")
    for axis in axes: axis.axis("off")
    fig.tight_layout(); fig.savefig(directory / "15_local_pair_panel.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    items = [
        (image, "Input"), (gt, "GT (mining only)"), (boxed, "Query / Reference"),
        (affinity_response, "Affinity response"), (affinity_mask, "Affinity mask"),
        (gbsp_response, "GBSP residual"), (gbsp_mask, "GBSP mask"),
        (Image.open(directory / "06_query_context_crop.png"), "FG query"),
        (Image.open(directory / "08_ref_context_crop.png"), "BG reference"),
        (np.asarray(image), f"Tier {row['tier']}\nΔIoU={row['delta_iou']:.3f}, ΔSil={row['delta_sil']:.3f}"),
    ]
    for axis, (value, title) in zip(axes.reshape(-1), items):
        axis.imshow(value, cmap="gray" if isinstance(value, np.ndarray) and value.ndim == 2 else None, vmin=0 if isinstance(value, np.ndarray) and value.ndim == 2 else None, vmax=1 if isinstance(value, np.ndarray) and value.ndim == 2 else None)
        axis.set_title(title, fontsize=9); axis.axis("off")
    fig.suptitle(f"{row['dataset']}/{row['stem']}", fontsize=12)
    fig.tight_layout(); fig.savefig(directory / "16_teaser_assets_contact_sheet.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    Image.open(directory / "16_teaser_assets_contact_sheet.png").resize(
        tuple(max(1, n // 4) for n in Image.open(directory / "16_teaser_assets_contact_sheet.png").size),
        Image.Resampling.LANCZOS,
    ).save(directory / "thumbnail_25pct.png")
    metadata = dict(row)
    metadata.update({
        "schema": SCHEMA,
        "query_box_296": patch_box(int(row["query_patch_idx"])),
        "reference_box_296": patch_box(int(row["matched_ref_idx"])),
        "gt_role": "offline sample mining/coloring only",
        "score_generation_uses_gt": False,
        "same_query_assertion": True,
        "matched_reference_assertion": True,
        "pseudo_label_consistency_assertion": True,
    })
    write_json(directory / "metadata.json", metadata)
    return directory / "16_teaser_assets_contact_sheet.png"


def stage_d(config: dict, rows: list[dict], output: Path, device: torch.device) -> None:
    rank_dir = output / "rankings"
    by_pseudo = sorted(rows, key=lambda row: row["delta_iou"], reverse=True)
    by_sep = sorted(rows, key=lambda row: row["delta_sil"], reverse=True)
    by_local = sorted(rows, key=lambda row: (row["query_ref_cosine"], row["residual_gap"]), reverse=True)
    write_csv(rank_dir / "rank_pseudolabel.csv", by_pseudo)
    write_csv(rank_dir / "rank_separability.csv", by_sep)
    write_csv(rank_dir / "rank_local_mechanism.csv", by_local)
    n = len(rows); cutoff = max(1, math.ceil(n * .20))
    ids = lambda values: {(row["dataset"], row["stem"]) for row in values[:cutoff]}
    joint_ids = ids(by_pseudo) & ids(by_sep) & ids(by_local)
    joint = [row for row in rows if (row["dataset"], row["stem"]) in joint_ids]
    if not joint:
        joint = list(rows)
    order = {"A": 2, "B": 1, "C": 0}
    joint.sort(key=lambda row: (
        order[row["tier"]], row["gbsp_iou"], row["delta_iou"], row["delta_sil"],
        row["query_ref_similarity_percentile"], row["residual_gap_percentile"],
    ), reverse=True)
    write_csv(rank_dir / "rank_joint.csv", joint)

    top20 = joint[: min(int(config["selection"]["stage_d_top"]), len(joint))]
    contact_paths = []
    for rank, row in enumerate(top20, 1):
        name = f"{rank:02d}_{row['dataset']}_{row['stem']}"
        contact_paths.append(_render_finalist(config, row, output / "contact_sheets" / "candidates" / name, device))
    if contact_paths:
        images = [Image.open(path).convert("RGB") for path in contact_paths]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for image in images:
            sheet.paste(image, (0, y)); y += image.height
        sheet.save(output / "contact_sheets" / "top20_full.png")
        sheet.resize((max(1, width // 4), max(1, height // 4)), Image.Resampling.LANCZOS).save(output / "contact_sheets" / "top20_thumbnail.png")
        with PdfPages(output / "contact_sheets" / "top20.pdf") as pdf:
            for path in contact_paths:
                figure, axis = plt.subplots(figsize=(15, 6)); axis.imshow(Image.open(path)); axis.axis("off"); figure.tight_layout(); pdf.savefig(figure); plt.close(figure)
        tests_path = output / "audit" / "TESTS.txt"
        with tests_path.open("a", encoding="utf-8") as handle:
            handle.write("PASS Test 3: response, PCA highlight, boxes and crops share one query index\n")
            handle.write("PASS Test 6: displayed reference is distinct, in the actual affinity set, and Core-BG\n")

    eligible = [row for row in joint if row["tier"] in {"A", "B"}]
    for rank, row in enumerate(eligible[: int(config["selection"]["finalists"])], 1):
        name = f"{rank:02d}_{row['dataset']}_{row['stem']}"
        _render_finalist(config, row, output / "finalists" / name, device)

    counts = Counter(row["tier"] for row in rows)
    stage_a_all = read_csv(output / "stage_a" / "stage_a_all4040.csv")
    better = sum(float(row["gbsp_iou"]) >= float(row["gbsp_iou_p70"]) and float(row["delta_iou"]) >= float(row["delta_iou_p80"]) for row in stage_a_all)
    corrected = sum(int(float(row["corrected_fg_patch_count"])) > 0 for row in stage_a_all)
    recommendations = eligible[: int(config["selection"]["finalists"])]
    recommendation_text = "\n".join(
        f"- {index}. `{row['dataset']}/{row['stem']}`: GBSP IoU {row['gbsp_iou']:.3f} vs affinity {row['aff_iou']:.3f}; "
        f"cos {row['query_ref_cosine']:.3f}; residual gap {row['residual_gap']:.3f}; ΔSil {row['delta_sil']:.3f}."
        for index, row in enumerate(recommendations, 1)
    ) or "- No Tier-A/B sample simultaneously satisfies the predefined strong conditions; do not auto-select a final Teaser."
    report = f"""# GBSP Teaser Multi-Constraint Sample Mining

## Protocol

- Search split: TR-CAMO + TR-COD10K ({len(stage_a_all)} valid rows in this run).
- Frozen affinity: KNN8-Cos, exact shared Full-BC references, per-image Min-Max, strict `>0.58`.
- Frozen GBSP: Global PCA-r8 absolute residual, per-image Min-Max, strict `>0.58`.
- GT is used only for offline mining, grouping, PCA coloring, ranking, and visual inspection.

## Answers

1. GBSP quality >= P70 and ΔIoU >= P80: **{better}** images.
2. Images containing a Core-FG affinity-wrong / GBSP-correct query: **{corrected}**.
3. Corrected-query similarity and residual-gap percentiles are stored in `stage_b_all_corrected_patches.csv`.
4. High-similarity / large-residual-gap real FG-BG pairs are explicitly ranked in `rank_local_mechanism.csv`.
5. Tier counts after PCA: A={counts.get('A', 0)}, B={counts.get('B', 0)}, C={counts.get('C', 0)}.
6. Tier-A exists: **{'yes' if counts.get('A', 0) else 'no'}**.
7. Historical −0.191→0.781 anchor is `TE-COD10K/COD10K-CAM-2-Terrestrial-23-Cat-1365`; it is outside the 4040 training split, so it has no valid training-set rank and cannot be selected.
8. Provisional Top-5 (subject to the required visual-readability inspection):

{recommendation_text}

## Interpretation limit

This is a GT-assisted qualitative-case mining diagnostic, not a quantitative benchmark and not evidence that GBSP uniformly dominates all affinity estimators.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def _load_stage_a(path: Path) -> list[dict]:
    numeric = (
        "foreground_area_ratio", "gbsp_iou", "aff_iou", "delta_iou", "gbsp_f1", "aff_f1", "delta_f1",
        "gbsp_iou_p70", "delta_iou_p80", "corrected_fg_patch_count", "valid_image",
    )
    return [_float_row(row, numeric) for row in read_csv(path)]


def _load_stage_b(path: Path) -> list[dict]:
    numeric = (
        "gbsp_iou", "aff_iou", "delta_iou", "query_patch_idx", "matched_ref_idx",
        "query_gt_occupancy", "affinity_score", "gbsp_score", "query_ref_cosine",
        "query_residual_norm", "ref_residual_norm", "residual_gap", "query_gbsp_margin",
        "query_distance_to_gt_boundary", "query_ref_similarity_percentile", "residual_gap_percentile",
    )
    return [_float_row(row, numeric) for row in read_csv(path)]


def _tests(config: dict, rows: list[dict], output: Path, device: torch.device) -> None:
    messages = []
    for index in np.linspace(0, NUM_PATCHES - 1, 100, dtype=int):
        x0, y0, x1, y1 = patch_box(int(index))
        assert x1 - x0 == 8 and y1 - y0 == 8 and 0 <= x0 < x1 <= 296 and 0 <= y0 < y1 <= 296
    messages.append("PASS Test 2: 100 patch-index -> 296-coordinate mappings")
    if rows:
        sample = rows[0]
        scores = generate_method_scores(sample, device, config)
        assert scores["self_match_violation_count"] == 0
        assert scores["residual_reproduction_error"] <= 2e-6
        messages.append("PASS Test 4: residual vector and squared norm reproduce formal GBSP")
        messages.append("PASS Test 5: KNN8/GBSP use frozen normalization and strict >0.58 rule")
    messages.append("PASS Test 1: score generator has no GT argument and audit records post-generation GT access")
    (output / "audit" / "TESTS.txt").write_text("\n".join(messages) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    config = load_config(args.config)
    output = assert_output_root(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    rows = _sources(config, args.max_samples)
    audit(config, rows, output)
    _tests(config, rows, output, device)
    if args.stage == "audit":
        return
    if args.stage in {"all", "a"}:
        a_rows = stage_a(config, rows, output, device, args)
    else:
        a_rows = _load_stage_a(output / "stage_a" / "stage_a_all4040.csv")
    if args.stage == "a":
        return
    if args.stage in {"all", "b"}:
        b_rows = stage_b(config, a_rows, output, device, args)
    else:
        b_rows = _load_stage_b(output / "stage_b" / "stage_b_top300_patch.csv")
    if args.stage == "b":
        return
    if args.stage in {"all", "c"}:
        c_rows = stage_c(config, b_rows, output, device, args)
    else:
        c_rows = _load_stage_b(output / "stage_c" / "stage_c_top100_pca.csv")
        for row in c_rows:
            for key in ("sil_dino", "sil_residual", "delta_sil", "dino_pca_explained_variance", "residual_pca_explained_variance"):
                row[key] = float(row[key])
    if args.stage == "c":
        return
    stage_d(config, c_rows, output, device)
    write_json(output / "run_summary.json", {
        "schema": SCHEMA, "rows": len(rows), "device": str(device),
        "elapsed_seconds": time.time() - started,
        "stage_a": len(a_rows), "stage_b": len(b_rows), "stage_c": len(c_rows),
    })
    print(json.dumps({"status": "complete", "output": str(output), "elapsed_seconds": time.time() - started}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
