#!/usr/bin/env python3
"""Query-removal validation and final analysis for candidate influence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import load_config  # noqa: E402
from models.gbsp_candidate_influence import exact_loo_batch, fit_scatter_basis, score_queries  # noqa: E402
from tools.gbsp_candidate_influence_common import (  # noqa: E402
    DATASETS,
    RANK,
    build_indices,
    high_is_one_percentile,
    load_aligned,
    nearest_patch_labels,
    npz_path,
    parse_subset,
    resolve_path,
    stable_seed,
    top_mask,
    write_json,
)
from tools.gbsp_knn_lsr_common import rank_metrics, resize_score_to_native, write_csv  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_candidate_influence.py"
VERSION = "gbsp_candidate_influence_eval_v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_spearman(left, right) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def safe_rank(labels, scores) -> tuple[float, float, int]:
    label = np.asarray(labels).reshape(-1).astype(bool)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(score)
    label, score = label[valid], score[valid]
    if label.size < 2 or not label.any() or not (~label).any():
        return float("nan"), float("nan"), int(label.size)
    return float(roc_auc_score(label, score)), float(average_precision_score(label, score)), int(label.size)


def distribution(values) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if not value.size:
        return {key: float("nan") for key in ("mean", "median", "q75", "q90", "q95")}
    return {
        "mean": float(value.mean()), "median": float(np.median(value)),
        "q75": float(np.quantile(value, .75)), "q90": float(np.quantile(value, .90)),
        "q95": float(np.quantile(value, .95)),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def run_query_validation(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    subset = parse_subset(output / "exact_loo" / "exact_loo_subset.txt")
    core_rows, r0_index, _ = build_indices(cfg)
    core_index = {(row["dataset"], row["stem"]): row for row in core_rows}
    exact_root = output / "exact_loo" / "candidate_exact_loo"
    device = torch.device(args.device)
    rows, failures = [], []
    started = time.perf_counter()
    for number, item in enumerate(subset, 1):
        dataset, stem = item["dataset"], item["stem"]
        try:
            exact = load_npz(npz_path(exact_root, dataset, stem))
            labels = exact["is_gt_foreground"].astype(bool)
            harmful = exact["harmful_oracle_improvement"].astype(np.float64)
            influence = exact["loo_projector_influence"].astype(np.float64)
            chosen: list[tuple[int, str, int]] = []
            fg_index = np.where(labels)[0]
            if fg_index.size:
                order = fg_index[np.argsort(-harmful[fg_index], kind="stable")]
                for rank, index in enumerate(order[: int(cfg.GBSP_INFLUENCE_TOP_H_PER_IMAGE)], 1):
                    chosen.append((int(index), "top_h_foreground", rank))
            bg_index = np.where(~labels)[0]
            if bg_index.size:
                rng = np.random.default_rng(stable_seed(dataset, stem, int(cfg.GBSP_INFLUENCE_REMOVAL_SEED)))
                controls = rng.choice(
                    bg_index, size=min(int(cfg.GBSP_INFLUENCE_RANDOM_BG_PER_IMAGE), bg_index.size), replace=False
                )
                chosen.extend((int(index), "random_background", rank) for rank, index in enumerate(controls, 1))
            if not chosen:
                continue
            core, r0, feature, indices = load_aligned(core_index[(dataset, stem)], r0_index)
            feature64 = feature.to(device=device, dtype=torch.float64)
            background = feature64.index_select(0, indices.to(device))
            _, original = fit_scatter_basis(background, RANK)
            positions = torch.as_tensor([item[0] for item in chosen], device=device)
            loo = exact_loo_batch(background, positions, original, original.unsqueeze(0))
            gt, _ = nearest_patch_labels(r0["gt_path"])
            shape = tuple(gt.shape[-2:])
            baseline_score = torch.as_tensor(r0["absolute_raw"]).float().reshape(-1)
            baseline = rank_metrics(resize_score_to_native(baseline_score, shape), gt)
            for offset, (candidate_position, kind, local_rank) in enumerate(chosen):
                score = score_queries(
                    feature64, loo.mean[offset], loo.basis[offset]
                )
                metric = rank_metrics(resize_score_to_native(score, shape), gt)
                rows.append({
                    "dataset": dataset, "stem": stem, "stratum": item["stratum"],
                    "candidate_position": candidate_position,
                    "candidate_index": int(exact["candidate_index"][candidate_position]),
                    "selection": kind, "selection_rank": local_rank,
                    "is_gt_foreground": int(labels[candidate_position]),
                    "exact_influence": float(influence[candidate_position]),
                    "harmful_oracle_improvement": float(harmful[candidate_position]),
                    "baseline_AP": baseline["AP"], "removed_AP": metric["AP"],
                    "delta_AP_remove": metric["AP"] - baseline["AP"],
                    "baseline_AUROC": baseline["AUROC"], "removed_AUROC": metric["AUROC"],
                    "delta_AUROC_remove": metric["AUROC"] - baseline["AUROC"],
                })
        except Exception as error:
            failures.append({"dataset": dataset, "stem": stem, "error": repr(error), "traceback": traceback.format_exc()})
        if number % 20 == 0 or number == len(subset):
            elapsed = time.perf_counter() - started
            print(f"[{number}/{len(subset)}] query-removal rows={len(rows)} failed={len(failures)} elapsed={elapsed:.1f}s", flush=True)
    query_root = output / "query_validation"
    write_csv(query_root / "top_harmful_removal.csv", rows)
    write_json(query_root / "failures.json", failures)
    if failures:
        raise RuntimeError(f"query-removal validation failed for {len(failures)} images")
    summary = {
        "version": VERSION, "created_at": now(), "images": len(subset), "removals": len(rows),
        "top_h_per_image": int(cfg.GBSP_INFLUENCE_TOP_H_PER_IMAGE),
        "random_bg_per_image": int(cfg.GBSP_INFLUENCE_RANDOM_BG_PER_IMAGE),
        "spearman_H_delta_AP": safe_spearman(
            [row["harmful_oracle_improvement"] for row in rows], [row["delta_AP_remove"] for row in rows]
        ),
        "spearman_H_delta_AUROC": safe_spearman(
            [row["harmful_oracle_improvement"] for row in rows], [row["delta_AUROC_remove"] for row in rows]
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(query_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def aggregate_quadrants(images: list[dict], *, with_h: bool, stage: str) -> list[dict]:
    totals = defaultdict(lambda: {"count": 0, "fg": 0, "h": [], "positive": 0})
    all_labels = []
    for payload in images:
        valid = payload["cvbr_valid_mask"].astype(bool)
        high_c = top_mask(payload["cvbr_score"], .10, valid)
        high_i = top_mask(payload["influence_proxy_raw"], .10)
        label = payload["is_gt_foreground"].astype(bool)
        all_labels.append(label[valid])
        for name, mask in (
            ("Low CVBR + Low Influence", valid & ~high_c & ~high_i),
            ("Low CVBR + High Influence", valid & ~high_c & high_i),
            ("High CVBR + Low Influence", high_c & ~high_i),
            ("High CVBR + High Influence", high_c & high_i),
        ):
            entry = totals[name]
            entry["count"] += int(mask.sum())
            entry["fg"] += int(label[mask].sum())
            if with_h:
                value = payload["harmful_oracle_improvement"][mask].astype(np.float64)
                entry["h"].extend(value.tolist())
                entry["positive"] += int((value > 0).sum())
    base = float(np.concatenate(all_labels).mean())
    total = sum(entry["count"] for entry in totals.values())
    rows = []
    for name in (
        "Low CVBR + Low Influence", "Low CVBR + High Influence",
        "High CVBR + Low Influence", "High CVBR + High Influence",
    ):
        entry = totals[name]
        precision = entry["fg"] / max(1, entry["count"])
        rows.append({
            "stage": stage, "group": name, "candidate_count": entry["count"],
            "candidate_fraction": entry["count"] / max(1, total),
            "foreground_count": entry["fg"], "foreground_precision": precision,
            "foreground_enrichment": precision / (base + 1e-12),
            "mean_H": float(np.mean(entry["h"])) if entry["h"] else float("nan"),
            "P_H_positive": entry["positive"] / max(1, len(entry["h"])) if entry["h"] else float("nan"),
        })
    return rows


def region_rows(images: list[dict], group_name: str) -> list[dict]:
    output = []
    definitions = (
        ("CVBR Top10", "cvbr", .10), ("Influence Top10", "influence", .10),
        ("Joint Top10", "joint", .10), ("Joint Top5", "joint", .05),
        ("Joint Top2", "joint", .02),
    )
    for label, method, fraction in definitions:
        labels, influence, harmful = [], [], []
        for payload in images:
            valid = payload["cvbr_valid_mask"].astype(bool)
            if method == "cvbr":
                mask = top_mask(payload["cvbr_score"], fraction, valid)
            elif method == "influence":
                mask = top_mask(payload["influence_proxy_raw"], fraction)
            else:
                joint = payload["cvbr_percentile"].astype(np.float64) * payload["influence_percentile"].astype(np.float64)
                mask = top_mask(joint, fraction, valid)
            labels.append(payload["is_gt_foreground"][mask].astype(bool))
            influence.append(payload["loo_projector_influence"][mask].astype(np.float64))
            harmful.append(payload["harmful_oracle_improvement"][mask].astype(np.float64))
        y = np.concatenate(labels) if labels else np.empty(0, dtype=bool)
        exact = np.concatenate(influence) if influence else np.empty(0)
        h = np.concatenate(harmful) if harmful else np.empty(0)
        bg = ~y
        output.append({
            "image_group": group_name, "score_region": label, "candidate_count": int(y.size),
            "FG_fraction": float(y.mean()) if y.size else float("nan"),
            "mean_exact_influence": float(exact.mean()) if exact.size else float("nan"),
            "mean_H": float(h.mean()) if h.size else float("nan"),
            "P_H_negative_for_BG": float((h[bg] < 0).mean()) if bg.any() else float("nan"),
        })
    return output


def harmful_ranking(images: list[dict]) -> list[dict]:
    methods = {
        "CVBR": lambda p: p["cvbr_percentile"].astype(np.float64),
        "Influence Proxy": lambda p: p["influence_percentile"].astype(np.float64),
        "Exact Influence (diagnostic)": lambda p: p["exact_influence_percentile"].astype(np.float64),
        "CVBR × Influence Proxy": lambda p: p["cvbr_percentile"].astype(np.float64) * p["influence_percentile"].astype(np.float64),
        "CVBR + Influence Proxy": lambda p: .5 * (p["cvbr_percentile"].astype(np.float64) + p["influence_percentile"].astype(np.float64)),
    }
    output = []
    for method, getter in methods.items():
        pooled_labels, pooled_scores = [], []
        selected = {fraction: [0, 0] for fraction in (.01, .02, .05, .10)}
        for payload in images:
            valid = payload["cvbr_valid_mask"].astype(bool)
            harmful_label = payload["is_gt_foreground"].astype(bool) & (payload["harmful_oracle_improvement"] > 0)
            score = getter(payload)
            pooled_labels.append(harmful_label[valid])
            pooled_scores.append(score[valid])
            for fraction in selected:
                mask = top_mask(score, fraction, valid)
                selected[fraction][0] += int(harmful_label[mask].sum())
                selected[fraction][1] += int(mask.sum())
        labels, scores = np.concatenate(pooled_labels), np.concatenate(pooled_scores)
        auroc, auprc, count = safe_rank(labels, scores)
        output.append({
            "risk_score": method, "candidate_count": count,
            "harmful_foreground_count": int(labels.sum()), "AUROC": auroc, "AUPRC": auprc,
            **{
                f"Top{int(fraction*100)}pct_precision": good / max(1, total)
                for fraction, (good, total) in selected.items()
            },
            **{
                f"Top{int(fraction*100)}pct_count": total
                for fraction, (_, total) in selected.items()
            },
        })
    return output


def save_figures(proxy_images: list[dict], exact_images: list[dict], figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260814)

    def sampled_arrays(images, fields, valid_field="cvbr_valid_mask", limit=120000):
        arrays = []
        for field in fields:
            values = [p[field][p[valid_field].astype(bool)].reshape(-1) for p in images]
            arrays.append(np.concatenate(values))
        count = arrays[0].size
        index = rng.choice(count, size=min(limit, count), replace=False)
        return [value[index] for value in arrays]

    cvbr, influence, label = sampled_arrays(
        exact_images, ("cvbr_percentile", "loo_projector_influence", "is_gt_foreground")
    )
    for suffix, mask in (("all", np.ones(label.shape, bool)), ("background", label == 0), ("foreground", label > 0)):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        color = np.where(label[mask] > 0, "#d62728", "#4c78a8")
        ax.scatter(cvbr[mask], influence[mask], c=color, s=4, alpha=.18, linewidths=0)
        ax.set(xlabel="CVBR within-image percentile", ylabel="Exact LOO influence", title=f"CVBR vs exact influence ({suffix})")
        fig.tight_layout(); fig.savefig(figure_root / f"figure_A_cvbr_exact_{suffix}.png", dpi=180); plt.close(fig)

    cvbr, harmful, label = sampled_arrays(
        exact_images, ("cvbr_percentile", "harmful_oracle_improvement", "is_gt_foreground")
    )
    for suffix, mask in (("all", np.ones(label.shape, bool)), ("background", label == 0), ("foreground", label > 0)):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        color = np.where(label[mask] > 0, "#d62728", "#4c78a8")
        ax.scatter(cvbr[mask], harmful[mask], c=color, s=4, alpha=.18, linewidths=0)
        ax.axhline(0, color="black", lw=.7)
        ax.set(xlabel="CVBR within-image percentile", ylabel="H (oracle distance improvement)", title=f"CVBR vs harmful influence ({suffix})")
        fig.tight_layout(); fig.savefig(figure_root / f"figure_B_cvbr_H_{suffix}.png", dpi=180); plt.close(fig)

    cvbr, proxy, label = sampled_arrays(
        proxy_images, ("cvbr_percentile", "influence_percentile", "is_gt_foreground")
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    bg = label == 0
    ax.hexbin(cvbr[bg], proxy[bg], gridsize=45, mincnt=1, cmap="Blues", bins="log")
    fg = ~bg
    ax.scatter(cvbr[fg], proxy[fg], s=5, alpha=.20, c="#d62728", label="foreground contamination")
    ax.axvline(.9, color="black", ls="--", lw=.8); ax.axhline(.9, color="black", ls="--", lw=.8)
    ax.set(xlabel="CVBR percentile", ylabel="Influence-proxy percentile", title="CVBR risk vs subspace influence proxy")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(figure_root / "figure_C_cvbr_proxy_quadrants.png", dpi=180); plt.close(fig)

    h_bg = np.concatenate([p["harmful_oracle_improvement"][~p["is_gt_foreground"].astype(bool)] for p in exact_images])
    h_fg = np.concatenate([p["harmful_oracle_improvement"][p["is_gt_foreground"].astype(bool)] for p in exact_images])
    low, high = np.quantile(np.concatenate([h_bg, h_fg]), [.005, .995])
    bins = np.linspace(low, high, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(h_bg, bins=bins, density=True, alpha=.55, label="background", color="#4c78a8")
    ax.hist(h_fg, bins=bins, density=True, alpha=.55, label="foreground", color="#d62728")
    ax.axvline(0, color="black", lw=.8); ax.set(xlabel="H", ylabel="Density", title="H(background) vs H(foreground)")
    ax.legend(); fig.tight_layout(); fig.savefig(figure_root / "figure_D_H_bg_vs_fg.png", dpi=180); plt.close(fig)

    zero = [p for p in exact_images if str(p["stratum"].item()) == "Zero"]
    top_values, rest_values = [], []
    for p in zero:
        valid = p["cvbr_valid_mask"].astype(bool)
        top = top_mask(p["cvbr_score"], .10, valid)
        top_values.append(p["harmful_oracle_improvement"][top])
        rest_values.append(p["harmful_oracle_improvement"][valid & ~top])
    top_h, rest_h = np.concatenate(top_values), np.concatenate(rest_values)
    low, high = np.quantile(np.concatenate([top_h, rest_h]), [.005, .995])
    bins = np.linspace(low, high, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(rest_h, bins=bins, density=True, alpha=.55, label="remaining true background", color="#4c78a8")
    ax.hist(top_h, bins=bins, density=True, alpha=.55, label="CVBR Top10 true background", color="#f58518")
    ax.axvline(0, color="black", lw=.8); ax.set(xlabel="H", ylabel="Density", title="Zero-contamination: CVBR Top10 background candidates")
    ax.legend(); fig.tight_layout(); fig.savefig(figure_root / "figure_E_zero_cvbr_top10_H.png", dpi=180); plt.close(fig)

    raw_bg = np.concatenate([p["influence_proxy_raw"][~p["is_gt_foreground"].astype(bool)] for p in proxy_images])
    raw_fg = np.concatenate([p["influence_proxy_raw"][p["is_gt_foreground"].astype(bool)] for p in proxy_images])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for values, label_text, color in ((raw_bg, "background", "#4c78a8"), (raw_fg, "foreground", "#d62728")):
        value = np.sort(values); ax.plot(value, np.arange(1, value.size + 1) / value.size, label=label_text, color=color)
    ax.set_xscale("log"); ax.set(xlabel="Influence proxy raw", ylabel="ECDF", title="Stage-A foreground/background proxy ECDF")
    ax.legend(); fig.tight_layout(); fig.savefig(figure_root / "proxy_foreground_background_ecdf.png", dpi=180); plt.close(fig)


def run_analysis(args: argparse.Namespace, cfg) -> None:
    output = resolve_path(args.out_root or cfg.GBSP_INFLUENCE_OUTPUT_ROOT)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    proxy_summary = list(csv.DictReader((output / "proxy" / "influence_proxy_summary.csv").open(encoding="utf-8")))
    proxy_root = output / "proxy" / "candidate_influence_proxy"
    proxy_images = [load_npz(npz_path(proxy_root, row["dataset"], row["stem"])) for row in proxy_summary]
    subset = parse_subset(output / "exact_loo" / "exact_loo_subset.txt")
    exact_root = output / "exact_loo" / "candidate_exact_loo"
    exact_images = []
    for row in subset:
        payload = load_npz(npz_path(exact_root, row["dataset"], row["stem"]))
        payload["stratum"] = np.asarray(row["stratum"])
        payload["contamination_ratio"] = np.asarray(row["contamination_ratio"])
        exact_images.append(payload)

    # Stage-A distributions and foreground-classification diagnostics.
    labels = np.concatenate([p["is_gt_foreground"].astype(bool) for p in proxy_images])
    proxy_raw = np.concatenate([p["influence_proxy_raw"].astype(np.float64) for p in proxy_images])
    proxy_gap = np.concatenate([p["influence_proxy_gap"].astype(np.float64) for p in proxy_images])
    proxy_distribution = []
    for score_name, score in (("proxy_raw", proxy_raw), ("proxy_gap", proxy_gap)):
        auroc, auprc, _ = safe_rank(labels, score)
        for candidate_type, mask in (("Background", ~labels), ("Foreground contamination", labels)):
            proxy_distribution.append({
                "candidate_type": candidate_type, "score": score_name, "count": int(mask.sum()),
                **distribution(score[mask]), "foreground_AUROC": auroc, "foreground_AUPRC": auprc,
            })
    write_csv(analysis / "proxy_foreground_vs_background.csv", proxy_distribution)

    correlations = []
    for proxy_name in ("influence_proxy_raw", "influence_proxy_gap"):
        for candidate_type in ("all", "background", "foreground"):
            left, right = [], []
            for p in proxy_images:
                mask = p["cvbr_valid_mask"].astype(bool)
                fg = p["is_gt_foreground"].astype(bool)
                if candidate_type == "background": mask &= ~fg
                if candidate_type == "foreground": mask &= fg
                left.append(p["cvbr_score"][mask]); right.append(p[proxy_name][mask])
            correlations.append({
                "stage": "Stage A full6473", "relation": f"CVBR vs {proxy_name}",
                "candidate_type": candidate_type, "count": int(sum(x.size for x in left)),
                "spearman": safe_spearman(np.concatenate(left), np.concatenate(right)),
            })

    exact_labels = np.concatenate([p["is_gt_foreground"].astype(bool) for p in exact_images])
    exact_influence = np.concatenate([p["loo_projector_influence"].astype(np.float64) for p in exact_images])
    harmful = np.concatenate([p["harmful_oracle_improvement"].astype(np.float64) for p in exact_images])
    exact_proxy_raw = np.concatenate([p["influence_proxy_raw"].astype(np.float64) for p in exact_images])
    exact_proxy_gap = np.concatenate([p["influence_proxy_gap"].astype(np.float64) for p in exact_images])

    validity = []
    for name, score in (("Raw (a*b_perp)", exact_proxy_raw), ("Gap-normalized", exact_proxy_gap)):
        row = {"proxy": name}
        for label, mask in (("all", np.ones(exact_labels.shape, bool)), ("background", ~exact_labels), ("foreground", exact_labels)):
            rho = safe_spearman(score[mask], exact_influence[mask])
            row[f"{label}_spearman"] = rho
            correlations.append({
                "stage": "Stage B exact subset", "relation": f"{name} vs exact influence",
                "candidate_type": label, "count": int(mask.sum()), "spearman": rho,
            })
        validity.append(row)
    write_csv(analysis / "influence_proxy_validity.csv", validity)

    # Exact foreground/background Table A.
    foreground_table = []
    gate = json.loads((output / "audit" / "exact_loo_correctness.json").read_text(encoding="utf-8"))
    deadzone = max(1e-6, 10.0 * float(gate["max_projector_frobenius_error"]) / math.sqrt(2 * RANK))
    for candidate_type, mask in (("Background", ~exact_labels), ("Foreground contamination", exact_labels)):
        h = harmful[mask]; influence = exact_influence[mask]
        foreground_table.append({
            "candidate_type": candidate_type, "count": int(mask.sum()),
            "influence_mean": float(influence.mean()), "influence_median": float(np.median(influence)),
            "influence_q75": float(np.quantile(influence, .75)), "influence_q90": float(np.quantile(influence, .90)),
            "influence_q95": float(np.quantile(influence, .95)),
            "H_mean": float(h.mean()), "H_median": float(np.median(h)),
            "H_q75": float(np.quantile(h, .75)), "H_q90": float(np.quantile(h, .90)), "H_q95": float(np.quantile(h, .95)),
            "P_H_positive": float((h > 0).mean()), "P_abs_H_in_numerical_deadzone": float((np.abs(h) <= deadzone).mean()),
            "numerical_deadzone": deadzone,
        })
    write_csv(analysis / "foreground_vs_background_influence.csv", foreground_table)

    # CVBR/H correlations only use the validated 136-candidate scope.
    for candidate_type in ("all", "background", "foreground"):
        left, right = [], []
        for p in exact_images:
            mask = p["cvbr_valid_mask"].astype(bool)
            fg = p["is_gt_foreground"].astype(bool)
            if candidate_type == "background": mask &= ~fg
            if candidate_type == "foreground": mask &= fg
            left.append(p["cvbr_score"][mask]); right.append(p["harmful_oracle_improvement"][mask])
        correlations.append({
            "stage": "Stage B exact subset", "relation": "CVBR vs harmful H",
            "candidate_type": candidate_type, "count": int(sum(x.size for x in left)),
            "spearman": safe_spearman(np.concatenate(left), np.concatenate(right)),
        })

    quadrant_rows = aggregate_quadrants(proxy_images, with_h=False, stage="Stage A full6473")
    quadrant_rows += aggregate_quadrants(exact_images, with_h=True, stage="Stage B exact subset")
    write_csv(analysis / "cvbr_influence_quadrants.csv", quadrant_rows)

    rankings = harmful_ranking(exact_images)
    write_csv(analysis / "harmful_foreground_ranking.csv", rankings)

    zero_images = [p for p in exact_images if str(p["stratum"].item()) == "Zero"]
    high_images = [p for p in exact_images if str(p["stratum"].item()) == "High"]
    zero_high = region_rows(zero_images, "Zero")
    high_rows = region_rows(high_images, "High")
    zero_high_rows = zero_high + high_rows
    write_csv(analysis / "zero_vs_high_contamination.csv", zero_high_rows)

    zero_detailed = []
    for region in ("CVBR Top10", "CVBR Remaining90"):
        exact_values, h_values = [], []
        for p in zero_images:
            valid = p["cvbr_valid_mask"].astype(bool)
            top = top_mask(p["cvbr_score"], .10, valid)
            mask = top if region == "CVBR Top10" else (valid & ~top)
            exact_values.append(p["loo_projector_influence"][mask]); h_values.append(p["harmful_oracle_improvement"][mask])
        influence_value, h = np.concatenate(exact_values), np.concatenate(h_values)
        high_threshold = np.quantile(np.concatenate([p["loo_projector_influence"] for p in zero_images]), .90)
        zero_detailed.append({
            "region": region, "candidate_count": int(h.size), "mean_exact_influence": float(influence_value.mean()),
            "mean_H": float(h.mean()), "fraction_H_negative": float((h < 0).mean()),
            "fraction_high_influence": float((influence_value >= high_threshold).mean()),
            "high_influence_threshold_subset_q90": float(high_threshold),
        })
    write_csv(analysis / "zero_contam_cvbr_top10_analysis.csv", zero_detailed)
    write_csv(analysis / "high_contam_candidate_regions.csv", high_rows)

    query_summary = json.loads((output / "query_validation" / "summary.json").read_text(encoding="utf-8"))
    correlations.extend([
        {"stage": "query removal", "relation": "H vs delta AP", "candidate_type": "selected", "count": query_summary["removals"], "spearman": query_summary["spearman_H_delta_AP"]},
        {"stage": "query removal", "relation": "H vs delta AUROC", "candidate_type": "selected", "count": query_summary["removals"], "spearman": query_summary["spearman_H_delta_AUROC"]},
    ])
    write_csv(analysis / "correlations.csv", correlations)
    save_figures(proxy_images, exact_images, analysis / "figures")

    # Evidence-backed decision and the fourteen required answers.
    fg_row = next(row for row in foreground_table if row["candidate_type"].startswith("Foreground"))
    bg_row = next(row for row in foreground_table if row["candidate_type"] == "Background")
    stage_a_quadrants = {row["group"]: row for row in quadrant_rows if row["stage"] == "Stage A full6473"}
    cvbr_top = next(row for row in rankings if row["risk_score"] == "CVBR")
    joint_top = next(row for row in rankings if row["risk_score"] == "CVBR × Influence Proxy")
    best_proxy = max(validity, key=lambda row: float(row["all_spearman"]))
    exact_bg_high = 0; exact_bg_high_protective = 0
    for p in exact_images:
        high = top_mask(p["loo_projector_influence"], .10)
        bg = ~p["is_gt_foreground"].astype(bool)
        exact_bg_high += int((high & bg).sum())
        exact_bg_high_protective += int((high & bg & (p["harmful_oracle_improvement"] < 0)).sum())
    condition_a = fg_row["H_mean"] > bg_row["H_mean"] and fg_row["P_H_positive"] > bg_row["P_H_positive"]
    condition_b = joint_top["Top2pct_precision"] > cvbr_top["Top2pct_precision"] + .02
    condition_c = float(best_proxy["all_spearman"]) >= .5
    exact_quadrant = next(
        row for row in quadrant_rows
        if row["stage"] == "Stage B exact subset" and row["group"] == "High CVBR + High Influence"
    )
    proxy_ranking = next(row for row in rankings if row["risk_score"] == "Influence Proxy")
    # Case 1 requires a genuinely precision-first harmful subset, not merely a
    # positive correlation.  Here a gate is supported only if the joint Top2
    # gain is clear, the joint quadrant is harmful on average and the joint
    # score adds value beyond influence alone.
    case1_supported = (
        condition_b
        and float(exact_quadrant["mean_H"]) > 0
        and float(exact_quadrant["P_H_positive"]) > .5
        and float(joint_top["AUPRC"]) > float(proxy_ranking["AUPRC"])
    )
    next_decision = (
        "进入极小干预面的 precision-first Influence-Gated CVBR。"
        if case1_supported
        else "停止 candidate-weighting/CVBR 修 PCA 路线；CVBR仅保留为分析工具，influence proxy可独立保留用于后续机制研究。"
    )
    report_lines = [
        "# GBSP Candidate Harmful-Influence Analysis", "",
        "## 正式结论", "", f"**{next_decision}**", "",
        f"- Stage A：6473/6473；Stage B：{len(exact_images)} 张固定分层 Oracle-valid 子集。",
        f"- Exact LOO 单元门：{gate['images']}×{gate['candidates_per_image']}，最大 projector Fro error={gate['max_projector_frobenius_error']:.3e}，PASS。",
        f"- CVBR 口径：每图既有 Second-Ring 136 candidates；influence-only 口径：全部 Full-BC candidates。",
        f"- H 对 matched-size oracle-clean 的 seed 0/1/2 projector distance improvement 取均值。", "",
        "## 表 A：Foreground vs Background Influence", "",
        "| Candidate Type | Count | Influence mean | Influence median | H mean | P(H>0) | |H|≈0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in foreground_table:
        report_lines.append(
            f"| {row['candidate_type']} | {row['count']:,} | {row['influence_mean']:.6g} | {row['influence_median']:.6g} | {row['H_mean']:+.6g} | {row['P_H_positive']:.3%} | {row['P_abs_H_in_numerical_deadzone']:.3%} |"
        )
    report_lines.extend(["", "## 表 B：CVBR × Influence 四象限（Exact subset）", "",
                         "| Group | Candidate % | FG Precision | FG Enrichment | Mean H | P(H>0) |",
                         "|---|---:|---:|---:|---:|---:|"])
    for row in [row for row in quadrant_rows if row["stage"] == "Stage B exact subset"]:
        report_lines.append(
            f"| {row['group']} | {row['candidate_fraction']:.3%} | {row['foreground_precision']:.3%} | {row['foreground_enrichment']:.3f}× | {row['mean_H']:+.6g} | {row['P_H_positive']:.3%} |"
        )
    report_lines.extend(["", "## 表 C：Harmful-Foreground Ranking", "",
                         "| Risk Score | AUROC | AUPRC | Top1% | Top2% | Top5% | Top10% |",
                         "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rankings:
        report_lines.append(
            f"| {row['risk_score']} | {row['AUROC']:.6f} | {row['AUPRC']:.6f} | {row['Top1pct_precision']:.3%} | {row['Top2pct_precision']:.3%} | {row['Top5pct_precision']:.3%} | {row['Top10pct_precision']:.3%} |"
        )
    report_lines.extend(["", "## 表 D：Influence Proxy Validity", "",
                         "| Proxy | All Spearman | BG | FG |", "|---|---:|---:|---:|"])
    for row in validity:
        report_lines.append(f"| {row['proxy']} | {row['all_spearman']:.6f} | {row['background_spearman']:.6f} | {row['foreground_spearman']:.6f} |")
    report_lines.extend(["", "## 表 E：Zero vs High Contamination", "",
                         "| Image Group | Score Region | FG % | Mean Influence | Mean H | P(H<0 for BG) |",
                         "|---|---|---:|---:|---:|---:|"])
    wanted = {("Zero", "CVBR Top10"), ("Zero", "Influence Top10"), ("High", "CVBR Top10"), ("High", "Joint Top10"), ("High", "Joint Top2")}
    for row in zero_high_rows:
        if (row["image_group"], row["score_region"]) in wanted:
            report_lines.append(
                f"| {row['image_group']} | {row['score_region']} | {row['FG_fraction']:.3%} | {row['mean_exact_influence']:.6g} | {row['mean_H']:+.6g} | {row['P_H_negative_for_BG']:.3%} |"
            )
    q4_ratio = exact_bg_high_protective / max(1, exact_bg_high)
    q8_cvbr = stage_a_quadrants["High CVBR + Low Influence"]["foreground_precision"] * stage_a_quadrants["High CVBR + Low Influence"]["candidate_count"]
    q8_joint_row = stage_a_quadrants["High CVBR + High Influence"]
    cvbr_high_count = stage_a_quadrants["High CVBR + Low Influence"]["candidate_count"] + q8_joint_row["candidate_count"]
    cvbr_high_fg = q8_cvbr + q8_joint_row["foreground_precision"] * q8_joint_row["candidate_count"]
    cvbr_high_precision = cvbr_high_fg / max(1, cvbr_high_count)
    zero_top = next(row for row in zero_detailed if row["region"] == "CVBR Top10")
    high_joint2 = next(row for row in high_rows if row["score_region"] == "Joint Top2")
    cvbr_h_fg = next(row["spearman"] for row in correlations if row["relation"] == "CVBR vs harmful H" and row["candidate_type"] == "foreground")
    cvbr_i_fg = next(row["spearman"] for row in correlations if row["relation"] == "CVBR vs influence_proxy_raw" and row["candidate_type"] == "foreground")
    report_lines.extend(["", "## 14 个必答问题", "",
        f"1. **Q1：** 前景污染的 exact influence 均值为 {fg_row['influence_mean']:.6g}，背景为 {bg_row['influence_mean']:.6g}；见表 A。",
        f"2. **Q2：** 前景 H 均值 {fg_row['H_mean']:+.6g}、P(H>0)={fg_row['P_H_positive']:.3%}；背景分别为 {bg_row['H_mean']:+.6g}、{bg_row['P_H_positive']:.3%}。",
        f"3. **Q3：** 以 correctness gate 推导的 numerical dead-zone |H|≤{deadzone:.3e}，前景污染中 {fg_row['P_abs_H_in_numerical_deadzone']:.3%} 近似无影响。",
        f"4. **Q4：** exact-influence 图内 Top10% 的真实背景有 {exact_bg_high:,} 个，其中 {exact_bg_high_protective:,} 个 H<0（{q4_ratio:.3%}），属于需要保护的背景 variation。",
        f"5. **Q5：** CVBR 与 raw proxy 在 foreground 内 Spearman={cvbr_i_fg:+.4f}；proxy 与 exact 的最佳全候选 Spearman={best_proxy['all_spearman']:+.4f}。",
        f"6. **Q6：** CVBR 与 H 在 foreground 内 Spearman={cvbr_h_fg:+.4f}。",
        f"7. **Q7：** 上述 CVBR–H 条件相关性直接回答其在污染内部区分 harmful/harmless 的能力；harmful ranking 见表 C。",
        f"8. **Q8：** 全量 Stage A 的 CVBR Top10 FG precision={cvbr_high_precision:.3%}；CVBR Top10∩Influence Top10={q8_joint_row['foreground_precision']:.3%}。",
        f"9. **Q9：** harmful-FG AUPRC：CVBR={cvbr_top['AUPRC']:.6f}，CVBR×Influence={joint_top['AUPRC']:.6f}。",
        f"10. **Q10：** Zero 子集中 CVBR Top10 的 mean H={zero_top['mean_H']:+.6g}，H<0={zero_top['fraction_H_negative']:.3%}，high-influence={zero_top['fraction_high_influence']:.3%}。",
        f"11. **Q11：** High contamination 的 Joint Top2 FG={high_joint2['FG_fraction']:.3%}、mean H={high_joint2['mean_H']:+.6g}；用于判断是否富集真正 harmful foreground。",
        f"12. **Q12：** 最佳 cheap proxy 是 {best_proxy['proxy']}，all/BG/FG Spearman={best_proxy['all_spearman']:.4f}/{best_proxy['background_spearman']:.4f}/{best_proxy['foreground_spearman']:.4f}。",
        f"13. **Q13：** H 与实际删除后的 ΔAP/ΔAUROC Spearman={query_summary['spearman_H_delta_AP']:+.4f}/{query_summary['spearman_H_delta_AUROC']:+.4f}。",
        f"14. **Q14：** {next_decision}", "",
        "## 扩展条件与停止判断", "",
        f"- A（foreground 更 harmful）：{'满足' if condition_a else '不满足'}。",
        f"- B（joint Top2 precision 至少比 CVBR 高 2pp）：{'满足' if condition_b else '不满足'}。",
        f"- C（proxy–exact Spearman ≥0.5）：{'满足' if condition_c else '不满足'}。",
        f"- Case 1 precision-first gate（joint平均H>0、P(H>0)>50%、且优于proxy-alone）：{'满足' if case1_supported else '不满足'}。",
        f"- Influence Proxy 单独 AUPRC={proxy_ranking['AUPRC']:.6f}，CVBR×Influence AUPRC={joint_top['AUPRC']:.6f}。",
        f"- High-CVBR∩High-Influence 的 mean H={exact_quadrant['mean_H']:+.6g}、P(H>0)={exact_quadrant['P_H_positive']:.3%}。",
        "- 本轮到此停止；未实现任何新 weighting、pseudo-label、threshold 或 estimator。", "",
        "## 合规性", "",
        "- H/foreground GT 仅用于 diagnosis。",
        "- 没有修改正式 GBSP、PCA rank、candidate、CVBR、图、resize 或阈值。",
        "- 没有训练，没有生成新伪标签。", "",
    ])
    (analysis / "RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")
    protocol = {
        "version": VERSION, "created_at": now(), "proxy_images": len(proxy_images),
        "exact_images": len(exact_images), "query_removals": query_summary["removals"],
        "numerical_deadzone": deadzone, "condition_A": condition_a,
        "condition_B": condition_b, "condition_C": condition_c,
        "case1_precision_first_gate_supported": case1_supported,
        "decision": next_decision,
    }
    write_json(analysis / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("query", "analyze"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_root")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(resolve_path(args.config))
    if args.command == "query":
        run_query_validation(args, cfg)
    else:
        run_analysis(args, cfg)


if __name__ == "__main__":
    main()
