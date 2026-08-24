#!/usr/bin/env python3
"""Evaluate and diagnose the registered CVBR-weighted GBSP R0--R5 caches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_cvbr_weighted_pca import (  # noqa: E402
    fit_weighted_affine_subspace,
    projector_distance,
)
from tools.gbsp_knn_lsr_common import (  # noqa: E402
    COD_METRICS,
    DATASETS,
    cod_metrics,
    finite_mean,
    load_native_gt,
    normalize_dataset,
    rank_metrics,
    resize_score_to_native,
    write_csv,
    write_json,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_cvbr_weighted_pca.py"
VERSION = "gbsp_cvbr_weighted_pca_eval_v1"
RISK_METHODS = ("CVBR-ring2", "Vanilla-PCA-self-residual")
RANK_FIELDS = ("AP", "AUROC")
MASK_FIELDS = ("S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _index(path: Path, *, normalize: bool = False) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output = {}
    for line, row in enumerate(read_jsonl(path), 1):
        dataset = normalize_dataset(row.get("dataset", "")) if normalize else str(row.get("dataset", ""))
        key = (dataset, str(row.get("stem", "")))
        if not all(key) or key in output:
            raise RuntimeError(f"invalid/duplicate identity at {path}:{line}: {key}")
        cache = Path(row.get("cache_path", ""))
        if not cache.is_file():
            raise FileNotFoundError(cache)
        output[key] = row
    return output


def _mean(values) -> float:
    return finite_mean(float(value) for value in values)


def _safe_rank(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1).astype(bool)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if not labels.any() or not (~labels).any():
        return {"AP": float("nan"), "AUROC": float("nan"), "num": int(labels.size)}
    return {
        "AP": float(average_precision_score(labels, scores)),
        "AUROC": float(roc_auc_score(labels, scores)),
        "num": int(labels.size),
    }


def _top_enrichment(labels: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1).astype(bool)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    count = max(1, int(math.ceil(float(fraction) * labels.size)))
    selected = np.argsort(-scores, kind="stable")[:count]
    base = float(labels.mean()) if labels.size else float("nan")
    top = float(labels[selected].mean()) if labels.size else float("nan")
    return {"base": base, "top": top, "enrichment": top / (base + 1e-12), "top_count": count}


def _tensor(payload: dict, field: str, shape: tuple[int, ...], path: Path) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{field} must be Tensor{shape}: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    return value


def _oracle_models(path: str | None, feature: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not path:
        return []
    payload = torch_load(Path(path), map_location="cpu")
    models = []
    for item in payload.get("matched_size_clean", []):
        if not item.get("valid") or not torch.is_tensor(item.get("candidate_indices")):
            continue
        indices = item["candidate_indices"].detach().cpu().long().reshape(-1)
        background = feature.index_select(0, indices)
        fit = fit_weighted_affine_subspace(background, torch.ones(indices.numel()), rank=8).fit
        models.append((fit.mean, fit.basis))
    return models


def _process_one(task: dict) -> dict:
    started = time.perf_counter()
    try:
        payloads = {}
        for variant, path_text in task["variant_paths"].items():
            path = Path(path_text)
            payload = torch_load(path, map_location="cpu")
            if payload.get("version") != "gbsp_cvbr_weighted_pca_v1":
                raise RuntimeError(f"invalid weighted cache: {path}")
            if (payload.get("dataset"), payload.get("stem")) != (task["source_dataset"], task["stem"]):
                raise RuntimeError(f"identity mismatch: {path}")
            payloads[variant] = payload
        baseline = payloads["R0_uniform"]
        indices = baseline["background_indices"].long().reshape(-1)
        for variant, payload in payloads.items():
            if not torch.equal(indices, payload["background_indices"].long().reshape(-1)):
                raise RuntimeError(f"candidate set changed in {variant}")

        gt = load_native_gt(task["gt_path"])
        shape = tuple(gt.shape[-2:])
        gt37 = F.interpolate(gt.unsqueeze(0), size=(37, 37), mode="nearest").squeeze().reshape(-1) > .5
        candidate_label = gt37.index_select(0, indices).numpy().astype(np.uint8)
        contamination_ratio = float(candidate_label.mean())
        context = FastCODContext(gt)

        feature_payload = torch_load(Path(baseline["source_feature_path"]), map_location="cpu")
        feature = _tensor(feature_payload, "tensor", (384, 37, 37), Path(baseline["source_feature_path"]))
        feature = F.normalize(feature.permute(1, 2, 0).reshape(37 * 37, 384), p=2, dim=1)
        oracle = _oracle_models(task.get("oracle_path"), feature)

        metric_rows, summary_rows, distance_rows = [], [], []
        raw_by_variant = {}
        for variant, payload in payloads.items():
            raw = _tensor(payload, "absolute_raw", (1, 37, 37), Path(task["variant_paths"][variant]))
            probability37 = _tensor(payload, "absolute_minmax", (1, 37, 37), Path(task["variant_paths"][variant]))
            native_raw = resize_score_to_native(raw, shape)
            native_probability = resize_score_to_native(probability37, shape)
            ranking = rank_metrics(native_raw, gt)
            masks = cod_metrics(context, native_probability, task["threshold"])
            raw_by_variant[variant] = ranking
            oracle_projector = _mean(
                projector_distance(payload["basis"].float(), basis) for _, basis in oracle
            ) if oracle else float("nan")
            oracle_center = _mean(
                torch.linalg.vector_norm(payload["mean"].float() - mean).item() for mean, _ in oracle
            ) if oracle else float("nan")
            metric_rows.append({
                "dataset": task["dataset"], "stem": task["stem"], "variant": variant,
                **ranking, **{field: float(masks[field]) for field in MASK_FIELDS},
            })
            summary_rows.append({
                "image_id": task["stem"], "dataset": task["dataset"], "variant": variant,
                "num_candidates": int(indices.numel()),
                "num_contaminated_candidates": int(candidate_label.sum()),
                "contamination_ratio": contamination_ratio,
                "cvbr_coverage": float(payload["cvbr_valid_mask"].float().mean()),
                "vanilla_ap": float(raw_by_variant["R0_uniform"]["AP"]),
                "cvbr_ap": float(ranking["AP"]),
                "vanilla_auroc": float(raw_by_variant["R0_uniform"]["AUROC"]),
                "cvbr_auroc": float(ranking["AUROC"]),
                "vanilla_oracle_projector_distance": float("nan"),
                "cvbr_oracle_projector_distance": oracle_projector,
                "vanilla_oracle_center_distance": float("nan"),
                "cvbr_oracle_center_distance": oracle_center,
                "weight_mean": float(payload["weights"].float().mean()),
                "weight_min": float(payload["weights"].float().min()),
                "effective_sample_size": float(payload["effective_sample_size"]),
                "num_cvbr_scored": int(payload["num_cvbr_scored"]),
                "num_suppressed": int(payload["num_suppressed"]),
            })
            distance_rows.append({
                "dataset": task["dataset"], "stem": task["stem"], "variant": variant,
                "oracle_seed_count": len(oracle),
                "projector_distance": oracle_projector, "center_distance": oracle_center,
            })
        vanilla_distance = next(row for row in distance_rows if row["variant"] == "R0_uniform")
        for row in summary_rows:
            row["vanilla_oracle_projector_distance"] = vanilla_distance["projector_distance"]
            row["vanilla_oracle_center_distance"] = vanilla_distance["center_distance"]

        valid = baseline["cvbr_valid_mask"].bool().reshape(-1)
        cvbr_score = baseline["cvbr_score"].float().reshape(-1)
        self_score = baseline["vanilla_pca_self_residual"].float().reshape(-1)
        risk_label = candidate_label[valid.numpy()]
        risk_rows = []
        for method, score in (
            (RISK_METHODS[0], cvbr_score[valid].numpy()),
            (RISK_METHODS[1], self_score[valid].numpy()),
        ):
            result = _safe_rank(risk_label, score)
            risk_rows.append({
                "dataset": task["dataset"], "stem": task["stem"], "risk_method": method,
                "AP": result["AP"], "AUROC": result["AUROC"], "num_scored": result["num"],
                "labels": risk_label, "scores": np.asarray(score, dtype=np.float32),
            })

        diag = {
            "image_id": np.asarray(task["stem"]), "dataset": np.asarray(task["dataset"]),
            "candidate_index": indices.numpy().astype(np.int16),
            "patch_y": torch.div(indices, 37, rounding_mode="floor").numpy().astype(np.int16),
            "patch_x": (indices % 37).numpy().astype(np.int16),
            "is_gt_foreground": candidate_label,
            "cvbr_score": np.where(valid.numpy(), cvbr_score.numpy(), np.nan).astype(np.float32),
            "cvbr_percentile": baseline["cvbr_percentile"].numpy().astype(np.float32),
            "vanilla_pca_self_residual": self_score.numpy().astype(np.float32),
        }
        for variant, payload in payloads.items():
            diag[f"weight__{variant}"] = payload["weights"].numpy().astype(np.float32)
            diag[f"weighted_pca_self_residual__{variant}"] = payload[
                "weighted_pca_self_residual"
            ].numpy().astype(np.float32)
        diagnostic_path = Path(task["diagnostic_path"])
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(diagnostic_path, **diag)

        return {
            "dataset": task["dataset"], "stem": task["stem"],
            "metric_rows": metric_rows, "summary_rows": summary_rows,
            "distance_rows": distance_rows, "risk_rows": risk_rows,
            "coverage": {
                "dataset": task["dataset"], "num_fullbc_candidates": int(indices.numel()),
                "num_cvbr_scored_candidates": int(valid.sum()),
                "num_gt_contaminated_candidates": int(candidate_label.sum()),
                "num_gt_contaminated_candidates_with_cvbr": int(candidate_label[valid.numpy()].sum()),
            },
            "runtime_seconds": time.perf_counter() - started,
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init_worker(threads: int) -> None:
    torch.set_num_threads(int(threads))


def _aggregate_metrics(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        dataset_rows = []
        for dataset in DATASETS:
            subset = [row for row in rows if row["variant"] == variant and row["dataset"] == dataset]
            if not subset:
                continue
            result = {
                "scope": "dataset", "dataset": dataset, "variant": variant,
                "num_images": len(subset),
                **{field: _mean(row[field] for row in subset) for field in fields},
            }
            dataset_rows.append(result)
            output.append(result)
        source = [row for row in rows if row["variant"] == variant]
        if dataset_rows:
            output.append({
                "scope": "dataset_macro", "dataset": "ALL", "variant": variant,
                "num_images": len(source),
                **{field: _mean(row[field] for row in dataset_rows) for field in fields},
            })
            output.append({
                "scope": "image_macro", "dataset": "ALL", "variant": variant,
                "num_images": len(source),
                **{field: _mean(row[field] for row in source) for field in fields},
            })
    return output


def _coverage(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in (*DATASETS, "ALL"):
        subset = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        if not subset:
            continue
        full = sum(int(row["num_fullbc_candidates"]) for row in subset)
        scored = sum(int(row["num_cvbr_scored_candidates"]) for row in subset)
        contaminated = sum(int(row["num_gt_contaminated_candidates"]) for row in subset)
        contaminated_scored = sum(int(row["num_gt_contaminated_candidates_with_cvbr"]) for row in subset)
        output.append({
            "dataset": dataset, "num_images": len(subset),
            "num_fullbc_candidates": full, "num_cvbr_scored_candidates": scored,
            "cvbr_coverage_ratio": scored / max(1, full),
            "num_gt_contaminated_candidates": contaminated,
            "num_gt_contaminated_candidates_with_cvbr": contaminated_scored,
            "contamination_coverage_ratio": contaminated_scored / max(1, contaminated),
        })
    return output


def _risk_comparison(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in (*DATASETS, "ALL"):
        for method in RISK_METHODS:
            subset = [row for row in rows if row["risk_method"] == method and (dataset == "ALL" or row["dataset"] == dataset)]
            if not subset:
                continue
            labels = np.concatenate([row["labels"] for row in subset])
            scores = np.concatenate([row["scores"] for row in subset])
            pooled = _safe_rank(labels, scores)
            result = {
                "dataset": dataset, "risk_method": method, "num_images": len(subset),
                "num_candidates": int(labels.size),
                "image_macro_AUROC": _mean(row["AUROC"] for row in subset),
                "image_macro_AUPRC": _mean(row["AP"] for row in subset),
                "pooled_AUROC": pooled["AUROC"], "pooled_AUPRC": pooled["AP"],
            }
            for fraction, label in ((.05, "top5"), (.10, "top10"), (.15, "top15")):
                enrichment = _top_enrichment(labels, scores, fraction)
                result[f"{label}_contamination"] = enrichment["top"]
                result[f"{label}_enrichment"] = enrichment["enrichment"]
            result["base_contamination"] = float(labels.mean())
            output.append(result)
    return output


def _contamination_groups(summary: list[dict]) -> tuple[dict[tuple[str, str], str], dict]:
    baseline = [row for row in summary if row["variant"] == "R0_uniform"]
    groups = {}
    positive = sorted(
        (row for row in baseline if float(row["contamination_ratio"]) > 0),
        key=lambda row: (float(row["contamination_ratio"]), row["dataset"], row["image_id"]),
    )
    for row in baseline:
        if float(row["contamination_ratio"]) == 0:
            groups[(row["dataset"], row["image_id"])] = "Zero"
    chunks = np.array_split(np.arange(len(positive)), 3)
    for name, chunk in zip(("Low", "Medium", "High"), chunks):
        for index in chunk.tolist():
            row = positive[index]
            groups[(row["dataset"], row["image_id"])] = name
    meta = {
        name: {
            "num_images": sum(value == name for value in groups.values()),
            "min_ratio": min((float(row["contamination_ratio"]) for row in baseline if groups.get((row["dataset"], row["image_id"])) == name), default=float("nan")),
            "max_ratio": max((float(row["contamination_ratio"]) for row in baseline if groups.get((row["dataset"], row["image_id"])) == name), default=float("nan")),
        }
        for name in ("Zero", "Low", "Medium", "High")
    }
    return groups, meta


def _stratified(summary: list[dict]) -> tuple[list[dict], dict]:
    groups, meta = _contamination_groups(summary)
    baseline = {(row["dataset"], row["image_id"]): row for row in summary if row["variant"] == "R0_uniform"}
    output = []
    for variant in sorted({row["variant"] for row in summary}):
        index = {(row["dataset"], row["image_id"]): row for row in summary if row["variant"] == variant}
        for group in ("Zero", "Low", "Medium", "High"):
            keys = [key for key, value in groups.items() if value == group and key in index]
            if not keys:
                continue
            output.append({
                "contamination": group, "variant": variant, "num_images": len(keys),
                "vanilla_AP": _mean(baseline[key]["vanilla_ap"] for key in keys),
                "CVBR_AP": _mean(index[key]["cvbr_ap"] for key in keys),
                "delta_AP": _mean(index[key]["cvbr_ap"] - baseline[key]["vanilla_ap"] for key in keys),
                "vanilla_AUROC": _mean(baseline[key]["vanilla_auroc"] for key in keys),
                "CVBR_AUROC": _mean(index[key]["cvbr_auroc"] for key in keys),
                "delta_AUROC": _mean(index[key]["cvbr_auroc"] - baseline[key]["vanilla_auroc"] for key in keys),
            })
    return output, meta


def _distance_aggregate(rows: list[dict]) -> list[dict]:
    fields = ("projector_distance", "center_distance")
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        dataset_rows = []
        for dataset in DATASETS:
            subset = [row for row in rows if row["variant"] == variant and row["dataset"] == dataset and math.isfinite(float(row["projector_distance"]))]
            if not subset:
                continue
            result = {"scope": "dataset", "dataset": dataset, "variant": variant, "num_images": len(subset), **{field: _mean(row[field] for row in subset) for field in fields}}
            output.append(result); dataset_rows.append(result)
        source = [row for row in rows if row["variant"] == variant and math.isfinite(float(row["projector_distance"]))]
        if dataset_rows:
            output.append({"scope": "dataset_macro", "dataset": "ALL", "variant": variant, "num_images": len(source), **{field: _mean(row[field] for row in dataset_rows) for field in fields}})
            output.append({"scope": "image_macro", "dataset": "ALL", "variant": variant, "num_images": len(source), **{field: _mean(row[field] for row in source) for field in fields}})
    return output


def _correlations(summary: list[dict]) -> list[dict]:
    baseline = {(row["dataset"], row["image_id"]): row for row in summary if row["variant"] == "R0_uniform"}
    output = []
    for variant in sorted({row["variant"] for row in summary if row["variant"] != "R0_uniform"}):
        rows = [row for row in summary if row["variant"] == variant]
        recovery, delta_ap, delta_auc = [], [], []
        for row in rows:
            key = (row["dataset"], row["image_id"]); base = baseline[key]
            values = (base["vanilla_oracle_projector_distance"], row["cvbr_oracle_projector_distance"])
            if not all(math.isfinite(float(value)) for value in values):
                continue
            recovery.append(float(values[0]) - float(values[1]))
            delta_ap.append(float(row["cvbr_ap"]) - float(base["vanilla_ap"]))
            delta_auc.append(float(row["cvbr_auroc"]) - float(base["vanilla_auroc"]))
        for target, values in (("delta_AP", delta_ap), ("delta_AUROC", delta_auc)):
            correlation = float(spearmanr(recovery, values).statistic) if len(recovery) >= 3 else float("nan")
            output.append({"variant": variant, "target": target, "spearman": correlation, "num_images": len(recovery)})
    return output


def _bootstrap(summary: list[dict], repetitions: int, seed: int) -> list[dict]:
    by_variant = defaultdict(dict)
    for row in summary:
        by_variant[row["variant"]][(row["dataset"], row["image_id"])] = row
    base = by_variant["R0_uniform"]
    rng = np.random.default_rng(seed)
    output = []
    for variant, index in sorted(by_variant.items()):
        if variant == "R0_uniform":
            continue
        for metric, field in (("AP", "cvbr_ap"), ("AUROC", "cvbr_auroc")):
            dataset_delta = {}
            for dataset in DATASETS:
                keys = [key for key in base if key[0] == dataset and key in index]
                dataset_delta[dataset] = np.asarray([
                    float(index[key][field]) - float(base[key]["vanilla_" + metric.lower()]) for key in keys
                ], dtype=np.float64)
            observed = _mean(values.mean() for values in dataset_delta.values() if values.size)
            samples = np.empty(repetitions, dtype=np.float64)
            for repeat in range(repetitions):
                means = [values[rng.integers(0, values.size, values.size)].mean() for values in dataset_delta.values() if values.size]
                samples[repeat] = float(np.mean(means))
            output.append({
                "variant": variant, "baseline": "R0_uniform", "metric": metric,
                "observed_delta_dataset_macro": observed,
                "ci95_low": float(np.quantile(samples, .025)),
                "ci95_high": float(np.quantile(samples, .975)),
                "probability_delta_gt_zero": float(np.mean(samples > 0)),
                "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            })
    return output


def _audit_text(cfg, root: Path, full: bool, risk: list[dict]) -> str:
    formal = next((row for row in risk if row["dataset"] == "ALL" and row["risk_method"] == "CVBR-ring2"), None)
    reproduced = "尚未进行正式全量复现" if not full or formal is None else (
        f"本轮 image-macro AUROC={formal['image_macro_AUROC']:.6f}；Top-10% enrichment={formal['top10_enrichment']:.6f}。"
    )
    return f"""# CVBR Implementation Audit

- CVBR source file: `{MAIN_ROOT / 'common/dabe_cvbr.py'}`
- CVBR cache builder: `{MAIN_ROOT / 'common/cache_dabe_cvbr.py'}`
- CVBR function: `cross_reconstruct_boundary` + `cvbr_reliability`
- validated version: `dabe_cvbr_v1`
- existing cache path: `{_resolve(cfg.GBSP_CVBR_CACHE_ROOT)}`
- score field: `boundary_cv_error_37`
- score definition: 用第一圈背景锚点交叉重构边界 patch 后的原始特征残差；数值越大表示越可疑。
- validated candidate scope: `border_ring2_only_37`，每图固定 136 个第二圈 patch。
- Full-BC mapping: 37×37 展平索引与 `background_indices` 直接相交；第二圈 136 个 patch 均属于冻结 Full-BC。
- unscored candidate policy: Full-BC 内其余候选不扩展 CVBR，权重固定为 1.0。
- distinct historical scopes: ring1 / ring2 / all-border 均曾统计；本轮只用产生 AUROC≈0.8372、Top-10%≈4.863× 的 ring2 `cv_error`，不使用 V1/V2 改写后的 anchor。
- score direction conversion: 无；原始 `boundary_cv_error` 已是“越大越可疑”。`source_q_*` 方向相反且本轮不作为 risk。
- GT use: 只在当前评估脚本中计算污染标签；缓存生成完全不读取 GT。
- current reproduction: {reproduced}
- output root: `{root}`
"""


def _result_text(full: bool, raw: list[dict], masks: list[dict], risk: list[dict], stratified: list[dict], distances: list[dict], correlations: list[dict], bootstrap: list[dict], failures: int) -> str:
    if not full:
        return """# CVBR-Guided Soft-Weighted PCA 结果

当前仅完成代码验证/少样本烟测，不能用其数值回答正式研究问题。R0–R5、评估、Oracle 距离、诊断缓存与 bootstrap 流程均已打通；请执行 `RUN_FULL.sh` 后由同一脚本生成正式 Q1–Q12 结论。
"""
    raw_macro = {(row["variant"]): row for row in raw if row["scope"] == "dataset_macro"}
    mask_macro = {(row["variant"]): row for row in masks if row["scope"] == "dataset_macro"}
    risk_all = {row["risk_method"]: row for row in risk if row["dataset"] == "ALL"}
    dist_macro = {row["variant"]: row for row in distances if row["scope"] == "dataset_macro"}
    soft = [name for name in raw_macro if name not in ("R0_uniform", "R5_p010_w000")]
    best = max(soft, key=lambda name: (raw_macro[name]["AP"], raw_macro[name]["AUROC"]))
    base, chosen, trim = raw_macro["R0_uniform"], raw_macro[best], raw_macro["R5_p010_w000"]
    bmask, cmask = mask_macro["R0_uniform"], mask_macro[best]
    boot = {row["metric"]: row for row in bootstrap if row["variant"] == best}
    zero = next(row for row in stratified if row["variant"] == best and row["contamination"] == "Zero")
    high = next(row for row in stratified if row["variant"] == best and row["contamination"] == "High")
    corr = {row["target"]: row for row in correlations if row["variant"] == best}
    oracle_delta = dist_macro["R0_uniform"]["projector_distance"] - dist_macro[best]["projector_distance"]
    ranking_better = risk_all[RISK_METHODS[0]]["image_macro_AUROC"] > risk_all[RISK_METHODS[1]]["image_macro_AUROC"]
    improved = chosen["AP"] > base["AP"] or chosen["AUROC"] > base["AUROC"]
    return f"""# CVBR-Guided Soft-Weighted PCA 正式结果

正式完整性：{'PASS' if failures == 0 else 'FAIL'}；最佳 soft variant（按 dataset-macro AP，AUROC 破平）为 **{best}**。

## Q1：CVBR 排序是否复现？

CVBR ring2 的 image-macro AUROC={risk_all[RISK_METHODS[0]]['image_macro_AUROC']:.6f}，Top-10% enrichment={risk_all[RISK_METHODS[0]]['top10_enrichment']:.6f}。与历史 0.837249 / 4.863113× 对照。

## Q2：CVBR 是否优于 vanilla PCA self-residual？

{'是' if ranking_better else '否'}。CVBR / PCA-self 的 AUROC 分别为 {risk_all[RISK_METHODS[0]]['image_macro_AUROC']:.6f} / {risk_all[RISK_METHODS[1]]['image_macro_AUROC']:.6f}，AUPRC 为 {risk_all[RISK_METHODS[0]]['image_macro_AUPRC']:.6f} / {risk_all[RISK_METHODS[1]]['image_macro_AUPRC']:.6f}。

## Q3：soft weighting 是否提高 Pixel AP？

R0={base['AP']:.6f}，{best}={chosen['AP']:.6f}，Δ={chosen['AP']-base['AP']:+.6f}；bootstrap 95% CI=[{boot['AP']['ci95_low']:+.6f}, {boot['AP']['ci95_high']:+.6f}]。

## Q4：是否提高 AUROC？

R0={base['AUROC']:.6f}，{best}={chosen['AUROC']:.6f}，Δ={chosen['AUROC']-base['AUROC']:+.6f}；bootstrap 95% CI=[{boot['AUROC']['ci95_low']:+.6f}, {boot['AUROC']['ci95_high']:+.6f}]。

## Q5：是否改善最终 binary pseudo-mask？

Fβw {bmask['F_beta_w']:.6f}→{cmask['F_beta_w']:.6f}，S {bmask['S_m']:.6f}→{cmask['S_m']:.6f}，E {bmask['E_mean']:.6f}→{cmask['E_mean']:.6f}，MAE {bmask['MAE']:.6f}→{cmask['MAE']:.6f}。阈值未重调。

## Q6：收益是否随 contamination 增大？

Zero ΔAP={zero['delta_AP']:+.6f}，High ΔAP={high['delta_AP']:+.6f}；据此判断为 {'支持' if high['delta_AP'] > zero['delta_AP'] else '不支持'}。

## Q7：零污染/低污染是否被伤害？

Zero ΔAP={zero['delta_AP']:+.6f}、ΔAUROC={zero['delta_AUROC']:+.6f}；详见 `contamination_stratified.csv` 的 Low 行。

## Q8：weighted subspace 是否更接近 Oracle？

projector distance：R0={dist_macro['R0_uniform']['projector_distance']:.6f}，{best}={dist_macro[best]['projector_distance']:.6f}，恢复量={oracle_delta:+.6f}。

## Q9：recovery 与性能提升是否相关？

Spearman(recovery, ΔAP)={corr['delta_AP']['spearman']:.6f}；Spearman(recovery, ΔAUROC)={corr['delta_AUROC']['spearman']:.6f}。

## Q10：soft weighting 是否优于 hard trimming？

{best} AP/AUROC={chosen['AP']:.6f}/{chosen['AUROC']:.6f}；R5={trim['AP']:.6f}/{trim['AUROC']:.6f}。{'soft 更优' if (chosen['AP'], chosen['AUROC']) > (trim['AP'], trim['AUROC']) else 'hard trim 数值更高，但仍只作为诊断'}。

## Q11：主要瓶颈是否仍来自 candidate contamination？

{'证据支持：soft weighting 带来排序收益且 Oracle projector distance 改善。' if improved and oracle_delta > 0 else '当前证据不足：污染风险能被排序，但 influence control 未同时带来稳定排序收益和几何恢复。'}

## Q12：是否进入 CVBR-init + one-step Huber？

{'暂不需要；先保留当前成功的 soft variant。' if improved and oracle_delta > 0 else '否，当前不满足任务书 Level B 所要求的 Oracle distance 改善。结果更接近“风险排序正确，但 percentile weighting 不是有效 influence control”；下一步应先做 candidate leverage / leave-one-out subspace influence analysis，再决定是否需要 influence-aware weighting。'}
"""


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(_resolve(args.config))
    root = _resolve(args.cache_root or cfg.GBSP_CVBR_OUTPUT_ROOT)
    out = _resolve(args.out_dir or root / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    variants = tuple(cfg.GBSP_CVBR_VARIANTS)
    indexes = {name: _index(root / name / "manifest_test.jsonl") for name in variants}
    identities = set(indexes[variants[0]])
    if any(set(indexes[name]) != identities for name in variants[1:]):
        raise RuntimeError("R0--R5 manifests do not contain identical identities")
    ordered = list(indexes[variants[0]])
    if args.max_samples >= 0:
        ordered = ordered[: args.max_samples]
    full = args.max_samples < 0 and len(ordered) == 6473
    oracle_index = _index(_resolve(args.oracle_root or cfg.GBSP_CVBR_ORACLE_ROOT) / "manifest_test.jsonl", normalize=True)
    tasks = []
    for source_key in ordered:
        row = indexes[variants[0]][source_key]
        dataset = normalize_dataset(source_key[0]); key = (dataset, source_key[1])
        tasks.append({
            "source_dataset": source_key[0], "dataset": dataset, "stem": source_key[1],
            "gt_path": row["gt_path"], "threshold": float(cfg.GBSP_CVBR_THRESHOLD),
            "variant_paths": {name: indexes[name][source_key]["cache_path"] for name in variants},
            "oracle_path": oracle_index.get(key, {}).get("cache_path"),
            "diagnostic_path": str(out / "candidate_diagnostics" / dataset / f"{source_key[1]}.npz"),
        })
    started = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.torch_threads,)) as pool:
        for index, result in enumerate(pool.map(_process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % args.progress_every == 0 or index == len(tasks):
                print(f"[{_now()}] CVBR evaluation {index}/{len(tasks)}", flush=True)
    failures = [row for row in results if "error" in row]
    valid = [row for row in results if "error" not in row]
    metric_rows = [item for row in valid for item in row["metric_rows"]]
    summary_rows = [item for row in valid for item in row["summary_rows"]]
    risk_rows = [item for row in valid for item in row["risk_rows"]]
    distance_rows = [item for row in valid for item in row["distance_rows"]]
    coverage = _coverage([row["coverage"] for row in valid])
    risk = _risk_comparison(risk_rows)
    raw = _aggregate_metrics(metric_rows, RANK_FIELDS)
    masks = _aggregate_metrics(metric_rows, MASK_FIELDS)
    stratified, strata = _stratified(summary_rows)
    distances = _distance_aggregate(distance_rows)
    correlations = _correlations(summary_rows)
    bootstrap = _bootstrap(summary_rows, args.bootstrap_repetitions, args.bootstrap_seed)

    write_csv(out / "cvbr_coverage.csv", coverage)
    write_csv(out / "candidate_risk_comparison.csv", risk)
    write_csv(out / "raw_ap_auroc.csv", raw)
    write_csv(out / "mask_metrics.csv", masks)
    write_csv(out / "contamination_stratified.csv", stratified)
    write_csv(out / "oracle_subspace_distance.csv", distances)
    write_csv(out / "subspace_recovery_correlations.csv", correlations)
    write_csv(out / "per_image_summary.csv", summary_rows)
    write_csv(out / "per_image_metrics.csv", metric_rows)
    write_csv(out / "paired_bootstrap.csv", bootstrap)
    write_json(out / "failures.json", failures)
    write_json(out / "contamination_strata.json", strata)
    formal_complete = full and len(valid) == 6473 and not failures
    (out / "CVBR_IMPLEMENTATION_AUDIT.md").write_text(
        _audit_text(cfg, root, formal_complete, risk), encoding="utf-8"
    )
    (out / "RESULTS.md").write_text(
        _result_text(formal_complete, raw, masks, risk, stratified, distances, correlations, bootstrap, len(failures)),
        encoding="utf-8",
    )
    protocol = {
        "version": VERSION, "created_at": _now(), "config": str(_resolve(args.config)),
        "cache_root": str(root), "oracle_root": str(_resolve(args.oracle_root or cfg.GBSP_CVBR_ORACLE_ROOT)),
        "requested": len(tasks), "valid": len(valid), "failed": len(failures),
        "formal_complete": formal_complete, "threshold": float(cfg.GBSP_CVBR_THRESHOLD),
        "resize": cfg.GBSP_CVBR_RESIZE, "cvbr_scope": cfg.GBSP_CVBR_SCOPE,
        "gt_used_for_generation": False, "gt_used_for_analysis": True,
        "bootstrap_repetitions": args.bootstrap_repetitions, "bootstrap_seed": args.bootstrap_seed,
        "wall_seconds": time.perf_counter() - started,
        "max_worker_peak_rss_mb": max((float(row["worker_peak_rss_mb"]) for row in valid), default=0.0),
    }
    write_json(out / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} evaluation samples failed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--cache-root", "--cache_root", dest="cache_root")
    value.add_argument("--oracle-root", "--oracle_root", dest="oracle_root")
    value.add_argument("--out-dir", "--out_dir", dest="out_dir")
    value.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=-1)
    value.add_argument("--workers", type=int, default=6)
    value.add_argument("--torch-threads", "--torch_threads", dest="torch_threads", type=int, default=1)
    value.add_argument("--progress-every", "--progress_every", dest="progress_every", type=int, default=50)
    value.add_argument("--bootstrap-repetitions", "--bootstrap_repetitions", dest="bootstrap_repetitions", type=int, default=10000)
    value.add_argument("--bootstrap-seed", "--bootstrap_seed", dest="bootstrap_seed", type=int, default=20260814)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
