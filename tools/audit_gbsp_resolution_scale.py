#!/usr/bin/env python3
"""Stage-1 mechanism audit for the frozen 296/37 and 512/64 GBSP caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS, _load_rgb_grid, _sobel_magnitude  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
PER_IMAGE_QUANTILES = (.10, .25, .50, .75, .90, .95, .99)
SUMMARY_QUANTILES = (.25, .50, .75, .90, .95)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_map(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for row in read_jsonl(path):
        key = (str(row["dataset"]), str(row["stem"]))
        if key in result:
            raise RuntimeError(f"duplicate manifest key: {key}")
        result[key] = row
    return result


def _sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Near-balanced sampling without replacement (CHAMELEON has only 76 images)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    if count != 500:
        raise ValueError("the registered Stage-1 protocol requires exactly 500 images")
    target = {"CHAMELEON": 76, "TE-CAMO": 141, "TE-COD10K": 141, "NC4K": 142}
    rng = random.Random(int(seed))
    output = []
    for dataset in DATASETS:
        candidates = sorted(grouped[dataset], key=lambda row: str(row["stem"]))
        if len(candidates) < target[dataset]:
            raise RuntimeError(f"{dataset}: need {target[dataset]}, found {len(candidates)}")
        output.extend(rng.sample(candidates, target[dataset]))
    output.sort(key=lambda row: (DATASETS.index(str(row["dataset"])), str(row["stem"])))
    return output


def _feature(path: str, expected_grid: int) -> torch.Tensor:
    payload = torch_load(path, map_location="cpu")
    value = payload["tensor"].detach().cpu().float().contiguous()
    if tuple(value.shape) != (384, expected_grid, expected_grid):
        raise RuntimeError(f"feature shape mismatch: {path}: {tuple(value.shape)}")
    return value


def _directed_graph_terms(feature: torch.Tensor, image_path: str, params: dict) -> dict[str, np.ndarray]:
    _, height, width = feature.shape
    normalized = F.normalize(feature.permute(1, 2, 0), p=2, dim=2)
    rgb = _load_rgb_grid(image_path, height).float().permute(1, 2, 0)
    sobel = _sobel_magnitude(rgb.permute(2, 0, 1)).float()
    buckets: dict[str, list[torch.Tensor]] = defaultdict(list)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sy = slice(max(0, -dy), min(height, height - dy))
            sx = slice(max(0, -dx), min(width, width - dx))
            dy_slice = slice(max(0, dy), min(height, height + dy))
            dx_slice = slice(max(0, dx), min(width, width + dx))
            src_f, dst_f = normalized[sy, sx], normalized[dy_slice, dx_slice]
            src_c, dst_c = rgb[sy, sx], rgb[dy_slice, dx_slice]
            src_e, dst_e = sobel[sy, sx], sobel[dy_slice, dx_slice]
            buckets["df"].append(1.0 - (src_f * dst_f).sum(2))
            buckets["dc"].append((src_c - dst_c).square().sum(2))
            buckets["de"].append(torch.maximum(src_e, dst_e))
    df = torch.cat([value.reshape(-1) for value in buckets["df"]]).numpy()
    dc = torch.cat([value.reshape(-1) for value in buckets["dc"]]).numpy()
    de = torch.cat([value.reshape(-1) for value in buckets["de"]]).numpy()
    nf = df / float(params["SIGMA_F"])
    nc = dc / float(params["SIGMA_C"])
    ne = de / float(params["SIGMA_E"])
    affinity = np.maximum(np.exp(-(nf + nc + ne)), 1e-8)
    cost = -np.log(affinity + 1e-8)
    return {"df": df, "dc": dc, "de": de, "nf": nf, "nc": nc, "ne": ne,
            "affinity": affinity, "edge_cost": cost}


def _metric_stats(value: np.ndarray, prefix: str) -> dict[str, float]:
    row = {f"{prefix}_mean": float(value.mean()), f"{prefix}_std": float(value.std())}
    for q in PER_IMAGE_QUANTILES:
        row[f"{prefix}_p{int(round(100*q))}"] = float(np.quantile(value, q))
    row[f"{prefix}_median"] = row[f"{prefix}_p50"]
    return row


def _load_gt(path: str, grid: int) -> torch.Tensor:
    with Image.open(path) as image:
        value = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)
    return F.interpolate(value[None, None], size=(grid, grid), mode="area").reshape(-1)


def _candidate_quality(indices: torch.Tensor, gt_path: str, grid: int) -> dict[str, float]:
    occupancy = _load_gt(gt_path, grid).index_select(0, indices.long())
    contamination = float(occupancy.mean())
    return {
        "candidate_precision": 1.0 - contamination,
        "candidate_strict_bg_patch_precision": float((occupancy <= .2).float().mean()),
        "fg_contamination_ratio": contamination,
    }


def _spectrum(singular_values: torch.Tensor) -> tuple[int, float, float]:
    energy = singular_values.detach().cpu().double().square()
    cumulative = torch.cumsum(energy, 0) / energy.sum()
    required = int(torch.searchsorted(cumulative, torch.tensor(.90, dtype=cumulative.dtype)).item()) + 1
    return required, float(cumulative[min(7, len(cumulative)-1)]), float(cumulative[min(11, len(cumulative)-1)])


def _find_gt(core_payload: dict) -> str:
    path = Path(str(core_payload["gt_path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-296", required=True)
    parser.add_argument("--config-512", required=True)
    parser.add_argument("--manifest-296", required=True)
    parser.add_argument("--manifest-512", required=True)
    parser.add_argument("--core-296", required=True)
    parser.add_argument("--scores-512", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    started = time.perf_counter()

    cfg296, cfg512 = load_config(args.config_296), load_config(args.config_512)
    params296 = dict(DABE_V2_DEFAULT_PARAMS); params296.update(_params_from_cfg(cfg296))
    params512 = dict(DABE_V2_DEFAULT_PARAMS); params512.update(_params_from_cfg(cfg512))
    features296 = _manifest_map(Path(args.manifest_296).resolve())
    features512 = _manifest_map(Path(args.manifest_512).resolve())
    samples = _sample(list(features512.values()), int(args.sample_count), int(args.seed))
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for row in samples:
        key = (str(row["dataset"]), str(row["stem"]))
        core_path = Path(args.core_296).resolve() / "test" / key[0] / f"{key[1]}.pt"
        core = torch_load(core_path, map_location="cpu")
        sample_rows.append({"dataset": key[0], "stem": key[1], "image_path": str(row["image_path"]),
                            "gt_path": _find_gt(core), "feature296_path": str(features296[key]["cache_path"]),
                            "feature512_path": str(features512[key]["cache_path"])})
    write_jsonl(out / "sample500.jsonl", sample_rows)
    with (out / "sample500.txt").open("w", encoding="utf-8") as handle:
        handle.write("# seed=42; counts=CHAMELEON:76,TE-CAMO:141,TE-COD10K:141,NC4K:142\n")
        for row in sample_rows:
            handle.write(f"{row['dataset']}\t{row['stem']}\t{row['image_path']}\n")

    per_image: list[dict] = []
    spectrum_rows: list[dict] = []
    pooled: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    failures: list[dict] = []
    for index, row in enumerate(sample_rows, 1):
        dataset, stem = row["dataset"], row["stem"]
        try:
            core_path = Path(args.core_296).resolve() / "test" / dataset / f"{stem}.pt"
            core = torch_load(core_path, map_location="cpu")
            dabe = torch_load(core["source_dabe_path"], map_location="cpu")
            score_a_path = Path(args.scores_512).resolve() / "A_bw2_r8" / "scores" / dataset / f"{stem}.pt"
            score_c_path = Path(args.scores_512).resolve() / "C_bw2_r12" / "scores" / dataset / f"{stem}.pt"
            score_a = torch_load(score_a_path, map_location="cpu")
            score_c = torch_load(score_c_path, map_location="cpu")
            for resolution, grid, params, feature_path, bc_value in (
                ("296", 37, params296, row["feature296_path"], dabe["bc_map_37"]),
                ("512", 64, params512, row["feature512_path"], score_a["bc"]),
            ):
                terms = _directed_graph_terms(_feature(feature_path, grid), row["image_path"], params)
                bc = bc_value.detach().cpu().float().numpy().reshape(-1)
                terms["BC"] = bc
                values = {"resolution": resolution, "dataset": dataset, "stem": stem,
                          "num_edges": int(terms["df"].size), "num_nodes": grid * grid}
                for metric, array in terms.items():
                    values.update(_metric_stats(array, metric))
                    pooled[(resolution, metric)].append(array.astype(np.float32, copy=False))
                top_count = int(math.ceil(.30 * bc.size))
                top = np.partition(bc, bc.size - top_count)[-top_count:]
                values.update({"top30_bc_min": float(top.min()), "top30_bc_mean": float(top.mean()),
                               "top30_bc_std": float(top.std())})
                per_image.append(values)

            result296 = core["results"]["r8"]
            req296, e8296, e12296 = _spectrum(result296["singular_values"])
            indices296 = result296["background_indices"]
            spectrum_rows.append({"resolution": "296", "dataset": dataset, "stem": stem,
                "required_rank_90": req296, "energy_at_r8": e8296, "energy_at_r12": e12296,
                "selected_rank": 8, "num_candidates": int(indices296.numel()),
                **_candidate_quality(indices296, row["gt_path"], 37)})
            indices512 = score_a["background_indices"]
            spectrum_rows.append({"resolution": "512", "dataset": dataset, "stem": stem,
                "required_rank_90": int(score_a["required_rank_uncapped"]),
                "energy_at_r8": float(score_a["energy_at_cap"]),
                "energy_at_r12": float(score_c["energy_at_cap"]), "selected_rank": 8,
                "num_candidates": int(indices512.numel()),
                **_candidate_quality(indices512, row["gt_path"], 64)})
        except Exception as exc:
            failures.append({"dataset": dataset, "stem": stem, "error": repr(exc)})
            raise
        if index % 20 == 0 or index == len(sample_rows):
            print(f"[{index}/{len(sample_rows)}] Stage-1 graph/spectrum", flush=True)

    _write_csv(out / "graph_stats_per_image.csv", per_image)
    _write_csv(out / "graph_stats_296.csv", [row for row in per_image if row["resolution"] == "296"])
    _write_csv(out / "graph_stats_512.csv", [row for row in per_image if row["resolution"] == "512"])
    _write_csv(out / "pca_spectrum_296_vs_512.csv", spectrum_rows)
    _write_csv(out / "pca_spectrum_296.csv", [row for row in spectrum_rows if row["resolution"] == "296"])
    _write_csv(out / "pca_spectrum_512.csv", [row for row in spectrum_rows if row["resolution"] == "512"])

    summary_rows = []
    metric_names = ("df", "dc", "de", "nf", "nc", "ne", "affinity", "edge_cost", "BC")
    aliases = {"nf": "df/sigma_f", "nc": "dc/sigma_c", "ne": "de/sigma_e"}
    for resolution in ("296", "512"):
        for metric in metric_names:
            value = np.concatenate(pooled[(resolution, metric)]).astype(np.float64, copy=False)
            summary_rows.append({"Resolution": resolution, "Metric": aliases.get(metric, metric),
                "Mean": float(value.mean()), **{f"P{int(q*100)}": float(np.quantile(value, q)) for q in SUMMARY_QUANTILES}})
    _write_csv(out / "graph_stats_summary.csv", summary_rows)
    _write_csv(out / "normalized_penalty_summary.csv", [row for row in summary_rows if "/sigma_" in row["Metric"]])
    _write_csv(out / "bc_distribution_summary.csv", [row for row in summary_rows if row["Metric"] == "BC"])

    lookup = {(row["Resolution"], row["Metric"]): row for row in summary_rows}
    normalized_ratios = {}
    for metric in ("df/sigma_f", "dc/sigma_c", "de/sigma_e"):
        normalized_ratios[metric] = lookup[("512", metric)]["P50"] / max(lookup[("296", metric)]["P50"], 1e-12)
    calibration = {
        "SIGMA_F": float(params296["SIGMA_F"]) * lookup[("512", "df")]["P50"] / max(lookup[("296", "df")]["P50"], 1e-12),
        "SIGMA_C": float(params296["SIGMA_C"]) * lookup[("512", "dc")]["P50"] / max(lookup[("296", "dc")]["P50"], 1e-12),
        "SIGMA_E": float(params296["SIGMA_E"]) * lookup[("512", "de")]["P50"] / max(lookup[("296", "de")]["P50"], 1e-12),
    }
    pca_by_res = {res: [row for row in spectrum_rows if row["resolution"] == res] for res in ("296", "512")}
    pca_summary = {}
    for res, rows in pca_by_res.items():
        req = np.asarray([row["required_rank_90"] for row in rows])
        pca_summary[res] = {"mean_required_rank90": float(req.mean()), "median_required_rank90": float(np.median(req)),
            "mean_energy_at_r8": float(np.mean([row["energy_at_r8"] for row in rows])),
            "mean_energy_at_r12": float(np.mean([row["energy_at_r12"] for row in rows])),
            "p_required_gt8": float((req > 8).mean()), "p_required_gt12": float((req > 12).mean()),
            "mean_candidates": float(np.mean([row["num_candidates"] for row in rows])),
            "mean_candidate_precision": float(np.mean([row["candidate_precision"] for row in rows]))}
    graph_drift = max(abs(math.log(max(value, 1e-12))) for value in normalized_ratios.values()) >= math.log(1.20)
    rank_drift = (pca_summary["512"]["median_required_rank90"] >= 1.5 * pca_summary["296"]["median_required_rank90"]
                  and pca_summary["512"]["median_required_rank90"] - pca_summary["296"]["median_required_rank90"] >= 8)
    decision = {"graph_drift": graph_drift, "rank_drift": rank_drift,
                "registered_graph_drift_rule": "max absolute log Q50 normalized-penalty ratio >= log(1.20)",
                "registered_rank_drift_rule": "median512 >= 1.5*median296 and difference >= 8",
                "normalized_penalty_q50_ratios_512_over_296": normalized_ratios,
                "calibrated_sigmas": calibration, "pca_summary": pca_summary,
                "sample_count": len(sample_rows), "seed": int(args.seed), "runtime_seconds": time.perf_counter()-started,
                "failures": failures}
    write_json(out / "stage1_decision.json", decision)
    report = ["# GBSP 296→512 图尺度与 PCA 诊断", "",
        f"- 固定样本：{len(sample_rows)} 张，seed={args.seed}；296/512 严格同图。",
        "- 图统计使用冻结的正式 sigma；Q50 漂移门槛预注册为 20%。",
        f"- Graph drift：**{'YES' if graph_drift else 'NO'}**。",
        f"- Intrinsic-rank drift：**{'YES' if rank_drift else 'NO'}**。", "",
        "## normalized penalty Q50 比率（512/296）", "",
        *[f"- {key}: {value:.6f}" for key, value in normalized_ratios.items()], "",
        "## Median-ratio 推导的唯一 GraphCal sigma", "",
        *[f"- {key}: {value:.8f}" for key, value in calibration.items()], "",
        "## PCA spectrum", "",
        f"- 296: mean/median rank90={pca_summary['296']['mean_required_rank90']:.3f}/{pca_summary['296']['median_required_rank90']:.3f}, energy@8={pca_summary['296']['mean_energy_at_r8']:.6f}。",
        f"- 512: mean/median rank90={pca_summary['512']['mean_required_rank90']:.3f}/{pca_summary['512']['median_required_rank90']:.3f}, energy@8={pca_summary['512']['mean_energy_at_r8']:.6f}, energy@12={pca_summary['512']['mean_energy_at_r12']:.6f}。",
        "", "## 进入下一阶段", "",
        f"- Stage 2 GraphCal：{'执行' if graph_drift else '不执行'}。",
        f"- Stage 3 R20/R10：{'执行' if rank_drift else '不执行'}。",
    ]
    (out / "RESULTS_GRAPH_SCALE.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    (out / "RESULTS_DIAGNOSTICS.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    shutil.copy2(args.config_296, out / "config_296_snapshot.py")
    shutil.copy2(args.config_512, out / "config_512_snapshot.py")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
