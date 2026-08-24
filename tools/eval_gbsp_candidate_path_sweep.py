#!/usr/bin/env python3
"""GT diagnostic for the six fixed-R32 candidate/path sweep caches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.metrics import _prepare_data  # noqa: E402
from common.utils import read_jsonl, torch_load, write_json  # noqa: E402
from models.gbsp_candidate_path_sweep import (  # noqa: E402
    candidate_variant_tag,
)


DATASETS = ("TR-CAMO", "TR-COD10K")
EXPECTED_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
VARIANTS = tuple(
    candidate_variant_tag(width, percent)
    for width in (1, 2)
    for percent in (25.0, 27.5, 30.0)
)
METRICS = (
    "S",
    "adp_E",
    "MAE",
    "F_beta_w",
    "PredArea",
    "GTArea",
    "AreaBias",
    "AreaAbsError",
    "candidate_fg_occupancy",
    "candidate_strict_bg_precision",
    "interior_candidate_ratio",
    "candidate_count",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_gt(path: Path, size: int, mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    value = torch.from_numpy((array > 0.5).astype(np.float32))[None, None]
    return F.interpolate(value, size=(size, size), mode=mode).squeeze(0)


def _gt_path(image_path: str, stem: str) -> Path:
    path = Path(image_path).resolve().parent.parent / "gt" / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _init_worker(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _evaluate_one(task: dict) -> list[dict] | dict:
    dataset, stem = str(task["dataset"]), str(task["stem"])
    threshold = float(task["threshold"])
    try:
        gt_path = _gt_path(task["image_path"], stem)
        gt68 = _load_gt(gt_path, 68, "nearest")
        gt37 = _load_gt(gt_path, 37, "area").reshape(-1)
        context = FastCODContext(gt68)
        loaded = {}
        predictions = []
        for tag, row in task["variants"].items():
            path = Path(row["cache_path"])
            payload = torch_load(path, map_location="cpu")
            if (payload.get("dataset"), payload.get("stem")) != (dataset, stem):
                raise RuntimeError(f"payload identity mismatch: {path}")
            if not bool(payload.get("graph_path_used")) or bool(
                payload.get("boundary_only_used", True)
            ):
                raise RuntimeError(f"graph-path contract mismatch: {path}")
            score = payload.get("gbsp_abs_minmax_37")
            if not torch.is_tensor(score) or tuple(score.shape) != (1, 37, 37):
                raise ValueError(f"invalid GBSP score: {path}")
            score68 = F.interpolate(
                score.detach().cpu().float().unsqueeze(0),
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = (score68 > threshold).float()
            loaded[tag] = (payload, mask)
            predictions.append((tag, "hard", mask))
        evaluated = context.evaluate_many(predictions, 0.5)
        rows = []
        gt_area = float(gt68.mean())
        for tag, (payload, mask) in loaded.items():
            metrics = evaluated[(tag, "hard")]
            pred_np, _ = _prepare_data(
                gt=context.gt_float,
                pred=mask.numpy().astype(float).squeeze(),
            )
            context.emeasure.gt_fg_numel = context.gt_fg_numel
            context.emeasure.gt_size = context.gt_size
            adp_e = float(context.emeasure.cal_adaptive_em(pred_np, context.gt))
            indices = payload["background_indices"].detach().cpu().long().reshape(-1)
            occupancy = gt37.index_select(0, indices)
            pred_area = float(mask.mean())
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "variant": tag,
                    "S": float(metrics["S_m"]),
                    "adp_E": adp_e,
                    "MAE": float(metrics["MAE"]),
                    "F_beta_w": float(metrics["F_beta_w"]),
                    "PredArea": pred_area,
                    "GTArea": gt_area,
                    "AreaBias": pred_area - gt_area,
                    "AreaAbsError": abs(pred_area - gt_area),
                    "candidate_fg_occupancy": float(occupancy.mean()),
                    "candidate_strict_bg_precision": float(
                        (occupancy <= 0.20).float().mean()
                    ),
                    "candidate_count": int(payload["candidate_count"]),
                    "boundary_seed_count": int(payload["boundary_seed_count"]),
                    "interior_candidate_count": int(
                        payload["interior_candidate_count"]
                    ),
                    "interior_candidate_ratio": float(
                        payload["interior_candidate_ratio"]
                    ),
                    "cache_path": str(Path(task["variants"][tag]["cache_path"])),
                }
            )
        return rows
    except Exception as error:
        return {
            "dataset": dataset,
            "stem": stem,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }


def _aggregate(per_image: list[dict], variants: tuple[str, ...]) -> list[dict]:
    summary = []
    by_variant_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_image:
        by_variant_dataset[(row["variant"], row["dataset"])].append(row)
    for variant in variants:
        dataset_rows = {}
        for dataset in DATASETS:
            rows = by_variant_dataset[(variant, dataset)]
            if not rows:
                continue
            values = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
            values["J_S_adpE_1mMAE"] = (
                values["S"] + values["adp_E"] + 1.0 - values["MAE"]
            ) / 3.0
            item = {
                "scope": "dataset",
                "dataset": dataset,
                "variant": variant,
                "num_samples": len(rows),
                **values,
            }
            summary.append(item)
            dataset_rows[dataset] = item
        all_rows = [row for row in per_image if row["variant"] == variant]
        if all_rows:
            values = {
                metric: float(np.mean([row[metric] for row in all_rows]))
                for metric in METRICS
            }
            summary.append(
                {
                    "scope": "sample_weighted",
                    "dataset": "TR-CAMO|TR-COD10K",
                    "variant": variant,
                    "num_samples": len(all_rows),
                    **values,
                    "J_S_adpE_1mMAE": (
                        values["S"] + values["adp_E"] + 1.0 - values["MAE"]
                    )
                    / 3.0,
                }
            )
        if all(dataset in dataset_rows for dataset in DATASETS):
            values = {
                metric: float(
                    np.mean([dataset_rows[dataset][metric] for dataset in DATASETS])
                )
                for metric in METRICS
            }
            summary.append(
                {
                    "scope": "dataset_equal_macro",
                    "dataset": "TR-CAMO|TR-COD10K",
                    "variant": variant,
                    "num_samples": sum(
                        dataset_rows[dataset]["num_samples"] for dataset in DATASETS
                    ),
                    **values,
                    "J_S_adpE_1mMAE": (
                        values["S"] + values["adp_E"] + 1.0 - values["MAE"]
                    )
                    / 3.0,
                }
            )
    return summary


def _best(rows: list[dict], scope: str, dataset: str) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["scope"] == scope
        and row["dataset"] == dataset
        and math.isfinite(float(row["J_S_adpE_1mMAE"]))
    ]
    return max(candidates, key=lambda row: float(row["J_S_adpE_1mMAE"])) if candidates else None


def run_evaluation(
    *,
    sweep_root: str | Path | None = None,
    out_dir: str | Path | None = None,
    candidate_manifests: dict[str, str | Path] | None = None,
    threshold: float = 0.50,
    max_samples: int = -1,
    workers: int = 4,
    torch_threads: int = 2,
    strict_failures: bool = False,
) -> dict:
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    explicit = dict(candidate_manifests or {})
    if explicit:
        if len(explicit) < 2:
            raise ValueError("explicit comparison requires at least two candidates")
        if not out_dir:
            raise ValueError("--out_dir is required with explicit --candidate entries")
        root = Path(sweep_root).resolve() if sweep_root else Path(out_dir).resolve().parent
        output = Path(out_dir).resolve()
        variants = tuple(explicit)
        manifest_paths = {tag: Path(path).resolve() for tag, path in explicit.items()}
    else:
        if not sweep_root:
            raise ValueError("sweep_root is required without explicit candidates")
        root = Path(sweep_root).resolve()
        output = Path(out_dir).resolve() if out_dir else root / "analysis_t050"
        if root not in output.parents:
            raise ValueError("diagnostic output must stay inside sweep_root")
        variants = VARIANTS
        manifest_paths = {
            tag: root / tag / "dinov1-s8" / "manifest_train.jsonl"
            for tag in variants
        }
    manifests = {}
    indices = {}
    for tag in variants:
        path = manifest_paths[tag]
        rows = read_jsonl(path)
        index = {(str(row["dataset"]), str(row["stem"])): row for row in rows}
        if len(index) != len(rows):
            raise RuntimeError(f"duplicate manifest identities: {path}")
        manifests[tag] = rows
        indices[tag] = index
    identities = list(indices[variants[0]])
    identity_set = set(identities)
    if any(set(indices[tag]) != identity_set for tag in variants[1:]):
        raise RuntimeError("candidate manifests do not share identities")
    if max_samples == -1:
        counts = Counter(dataset for dataset, _ in identities)
        if len(identities) != EXPECTED_TOTAL or dict(counts) != EXPECTED_COUNTS:
            raise RuntimeError(
                f"formal evaluation requires full 4040 identities: {len(identities)} {dict(counts)}"
            )
    elif len(identities) > max_samples:
        identities = identities[: int(max_samples)]

    tasks = []
    for dataset, stem in identities:
        first = indices[variants[0]][(dataset, stem)]
        tasks.append(
            {
                "dataset": dataset,
                "stem": stem,
                "image_path": first["image_path"],
                "threshold": float(threshold),
                "variants": {
                    tag: indices[tag][(dataset, stem)] for tag in variants
                },
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(int(torch_threads),),
    ) as pool:
        for index, result in enumerate(pool.map(_evaluate_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                failed = sum(isinstance(row, dict) for row in results)
                print(
                    f"Candidate-path GT diagnostic {index}/{len(tasks)} failed={failed}",
                    flush=True,
                )
    failures = [row for row in results if isinstance(row, dict)]
    per_image = [row for group in results if isinstance(group, list) for row in group]
    _write_csv(output / "per_image.csv", per_image)
    write_json(output / "failures.json", failures)
    summary = _aggregate(per_image, variants)
    _write_csv(output / "summary.csv", summary)
    best_macro = _best(summary, "dataset_equal_macro", "TR-CAMO|TR-COD10K")
    best_camo = _best(summary, "dataset", "TR-CAMO")
    best_cod = _best(summary, "dataset", "TR-COD10K")
    metadata = {
        "version": "gbsp_r32_candidate_path_sweep_eval_v1",
        "sweep_root": str(root),
        "output_dir": str(output),
        "threshold": float(threshold),
        "num_requested": len(tasks),
        "num_valid_images": len(per_image) // len(variants),
        "num_failed_images": len(failures),
        "formal_full4040": len(tasks) == EXPECTED_TOTAL and not failures,
        "best_equal_macro_variant": best_macro["variant"] if best_macro else None,
        "best_camo_variant": best_camo["variant"] if best_camo else None,
        "best_cod10k_variant": best_cod["variant"] if best_cod else None,
        "selection_score": "(S + adp_E + 1 - MAE) / 3",
        "gt_used_for_generation": False,
        "gt_used_for_diagnostic": True,
        "wall_seconds": time.perf_counter() - started,
        "candidate_manifests": {
            tag: str(manifest_paths[tag]) for tag in variants
        },
    }
    write_json(output / "metadata.json", metadata)
    selected = {
        "equal macro": best_macro,
        "TR-CAMO": best_camo,
        "TR-COD10K": best_cod,
    }
    lines = [
        "# GBSP R32 候选数量 / 图路径参与度诊断",
        "",
        "> 这是读取训练 GT 的离线诊断；GT 未参与缓存生成，诊断排名不能写成无监督选择。",
        "",
        f"- 固定阈值：strict > {float(threshold):.2f}",
        "- 固定项：DINOv1-S/8、37×37、8 邻域图、Dijkstra、R32、平方残差、逐图 Min-Max。",
        (
            "- 显式候选：" + "、".join(variants) + "。"
            if explicit
            else "- 变量：BORDER_WIDTH={1,2}，TOP_PERCENT={25,27.5,30}。"
        ),
        "- 主分数：(S + adp E + 1 - MAE) / 3。",
        "",
        "| scope | best variant | S | adp E | MAE | Fβw | candidate FG occupancy | interior ratio | J |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in selected.items():
        if not row:
            continue
        lines.append(
            f"| {label} | {row['variant']} | {float(row['S']):.6f} | "
            f"{float(row['adp_E']):.6f} | {float(row['MAE']):.6f} | "
            f"{float(row['F_beta_w']):.6f} | "
            f"{float(row['candidate_fg_occupancy']):.6f} | "
            f"{float(row['interior_candidate_ratio']):.6f} | "
            f"{float(row['J_S_adpE_1mMAE']):.6f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if failures and strict_failures:
        raise RuntimeError(f"candidate-path diagnostic failed for {len(failures)} images")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=MANIFEST",
        help="Explicit candidate manifest; repeat to compare arbitrary settings.",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument("--strict_failures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    candidate_manifests = {}
    for item in args.candidate:
        if "=" not in item:
            raise ValueError("--candidate must be LABEL=MANIFEST")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label or label in candidate_manifests:
            raise ValueError(f"invalid or duplicate candidate label: {label!r}")
        candidate_manifests[label] = path
    run_evaluation(
        sweep_root=args.sweep_root,
        out_dir=args.out_dir,
        candidate_manifests=candidate_manifests or None,
        threshold=args.threshold,
        max_samples=args.max_samples,
        workers=args.workers,
        torch_threads=args.torch_threads,
        strict_failures=args.strict_failures,
    )
