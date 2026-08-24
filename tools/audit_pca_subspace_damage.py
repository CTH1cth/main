#!/usr/bin/env python3
"""Paired PCA subspace-damage audit for clustered vs scattered contamination.

This is a diagnostic-only script.  It consumes the frozen reference-
contamination caches and DINO feature caches; it does not generate pseudo
labels, train a model, or use GT beyond the declared oracle intervention that
already exists in those caches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from models.reference_contamination import normalize_patch_features


DATASETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")
RANK = 8
METRICS = (
    "delta_stable_rank",
    "delta_energy_effective_rank",
    "delta_k80",
    "delta_top1_energy_fraction",
    "rotation_total_sin2",
    "rotation_effective_count",
    "principal_angle_max_deg",
    "principal_angle_rms_deg",
    "principal_angle_count_gt5",
    "principal_angle_count_gt10",
    "pollution_effective_pcs",
    "pollution_k80",
    "pollution_total_share",
    "mean_shift_norm",
    "gbsp_ap_drop_noninj",
    "gbsp_auroc_drop_noninj",
    "knn8_ap_drop_noninj",
    "knn8_auroc_drop_noninj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_root", required=True)
    parser.add_argument("--cluster_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--p_values", nargs="+", type=float, default=(.05, .10, .20))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260808)
    parser.add_argument("--progress_every", type=int, default=10)
    return parser.parse_args()


def read_manifest(root: Path) -> dict[tuple[str, str], Path]:
    path = root / "manifest_test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    output: dict[tuple[str, str], Path] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["stem"]))
        cache_path = Path(row["cache_path"])
        if key in output or not cache_path.is_file():
            raise RuntimeError(f"invalid manifest row: {key}, {cache_path}")
        output[key] = cache_path
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def condition_map(payload: dict) -> dict[tuple[int, float], dict]:
    return {
        (int(item["seed"]), float(item["p"])): item
        for item in payload["contamination"]
        if bool(item["valid"])
    }


def matched_base(payload: dict, seed: int) -> torch.Tensor:
    labels = payload["patch_gt_label"].bool().reshape(-1)
    natural = payload["natural"]["candidate_indices"].long().reshape(-1)
    clean = natural[~labels.index_select(0, natural)]
    match = next(item for item in payload["matched_size_clean"] if int(item["seed"]) == seed)
    if not bool(match["valid"]):
        raise RuntimeError(f"matched clean base is invalid for seed={seed}")
    replacement = match["replacement_bg_indices"].long().reshape(-1)
    base = torch.cat([clean, replacement]).sort().values
    if (
        base.numel() != int(match["candidate_count"])
        or base.numel() != torch.unique(base).numel()
        or bool(labels.index_select(0, base).any())
    ):
        raise RuntimeError("reconstructed matched-size clean base is invalid")
    return base


def contaminated_candidate(base: torch.Tensor, item: dict) -> torch.Tensor:
    removed = item["removed_background_indices"].long().reshape(-1)
    injected = item["injected_foreground_indices"].long().reshape(-1)
    retained = base[~torch.isin(base, removed)]
    candidate = torch.cat([retained, injected]).sort().values
    if candidate.numel() != base.numel() or candidate.numel() != torch.unique(candidate).numel():
        raise RuntimeError("contaminated dictionary is not size preserving and unique")
    return candidate


def fit_pca(feature: torch.Tensor, index: torch.Tensor, rank: int = RANK):
    matrix = feature.index_select(0, index.to(feature.device))
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    fallback = False
    try:
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        fallback = True
        cpu = centered.detach().cpu().double()
        _, singular_cpu, vh_cpu = torch.linalg.svd(cpu, full_matrices=False)
        singular = singular_cpu.to(feature.device, feature.dtype)
        vh = vh_cpu.to(feature.device, feature.dtype)
    basis, _ = torch.linalg.qr(vh[:rank].T, mode="reduced")
    gram_error = float(
        (basis.T @ basis - torch.eye(rank, device=basis.device, dtype=basis.dtype)).abs().max()
    )
    if gram_error >= 1e-4:
        raise RuntimeError(f"PCA basis orthogonality failure: {gram_error}")
    return mean, basis.contiguous(), singular.contiguous(), fallback


def energy_effective_count(energy: torch.Tensor) -> float:
    energy = energy.double().clamp_min(0)
    total = float(energy.sum())
    if total <= 1e-24:
        return 0.0
    probability = energy / total
    positive = probability > 0
    return float(torch.exp(-(probability[positive] * probability[positive].log()).sum()))


def k_fraction(energy: torch.Tensor, fraction: float = .80) -> int:
    value = torch.sort(energy.double().clamp_min(0), descending=True).values
    total = float(value.sum())
    if total <= 1e-24:
        return 0
    target = torch.tensor(fraction, dtype=value.dtype, device=value.device)
    return int(torch.searchsorted(value.cumsum(0) / total, target).item() + 1)


def principal_angle_metrics(clean_basis: torch.Tensor, contaminated_basis: torch.Tensor) -> dict:
    cosine = torch.linalg.svdvals(clean_basis.T @ contaminated_basis).clamp(0.0, 1.0)
    sin2 = (1.0 - cosine.square()).clamp_min(0.0)
    angles = torch.rad2deg(torch.acos(cosine)).sort(descending=True).values
    total = float(sin2.sum())
    output = {
        "rotation_total_sin2": total,
        "rotation_effective_count": energy_effective_count(sin2),
        "principal_angle_max_deg": float(angles.max()),
        "principal_angle_rms_deg": float(torch.sqrt(angles.square().mean())),
        "principal_angle_count_gt5": int((angles > 5.0).sum()),
        "principal_angle_count_gt10": int((angles > 10.0).sum()),
    }
    for index, angle in enumerate(angles.tolist(), 1):
        output[f"principal_angle_{index}_deg"] = float(angle)
    clean_retention = (clean_basis.T @ contaminated_basis).square().sum(dim=1).clamp(0.0, 1.0)
    for index, loss in enumerate((1.0 - clean_retention).tolist(), 1):
        output[f"clean_pc{index}_alignment_loss"] = float(loss)
    return output


def covariance_delta_metrics(
    feature: torch.Tensor,
    base: torch.Tensor,
    item: dict,
    clean_mean: torch.Tensor,
    contaminated_mean: torch.Tensor,
) -> tuple[dict, torch.Tensor, bool]:
    removed = item["removed_background_indices"].long().to(feature.device)
    injected = item["injected_foreground_indices"].long().to(feature.device)
    n = int(base.numel())
    m = int(injected.numel())
    if m == 0 or removed.numel() != m:
        raise RuntimeError("positive, size-preserving intervention required")
    injected_feature = feature.index_select(0, injected)
    removed_feature = feature.index_select(0, removed)
    z = torch.cat(
        [
            injected_feature.T,
            removed_feature.T,
            contaminated_mean[:, None],
            clean_mean[:, None],
        ],
        dim=1,
    )
    coefficient = torch.cat(
        [
            torch.full((m,), 1.0 / n, device=feature.device, dtype=feature.dtype),
            torch.full((m,), -1.0 / n, device=feature.device, dtype=feature.dtype),
            torch.tensor([-1.0, 1.0], device=feature.device, dtype=feature.dtype),
        ]
    )
    fallback = False
    try:
        _, r = torch.linalg.qr(z, mode="reduced")
        core = (r * coefficient[None, :]) @ r.T
        eigenvalue = torch.linalg.eigvalsh(core)
    except RuntimeError:
        fallback = True
        z_cpu = z.detach().cpu().double()
        coefficient_cpu = coefficient.detach().cpu().double()
        _, r_cpu = torch.linalg.qr(z_cpu, mode="reduced")
        core_cpu = (r_cpu * coefficient_cpu[None, :]) @ r_cpu.T
        eigenvalue = torch.linalg.eigvalsh(core_cpu).to(feature.device, feature.dtype)
    singular = eigenvalue.abs().sort(descending=True).values
    energy = singular.square()
    total = float(energy.sum())
    metrics = {
        "delta_stable_rank": total / max(float(energy.max()), 1e-24),
        "delta_energy_effective_rank": energy_effective_count(energy),
        "delta_k80": k_fraction(energy),
        "delta_top1_energy_fraction": float(energy.max()) / max(total, 1e-24),
        "delta_spectral_norm": float(singular.max()),
        "delta_frobenius_norm": math.sqrt(max(total, 0.0)),
    }
    return metrics, energy / max(total, 1e-24), fallback


def pollution_metrics(
    feature: torch.Tensor,
    candidate: torch.Tensor,
    item: dict,
    basis: torch.Tensor,
) -> tuple[dict, list[dict]]:
    injected_index = item["injected_foreground_indices"].long().to(feature.device)
    candidate = candidate.to(feature.device)
    injected_mask = torch.isin(candidate, injected_index)
    retained_index = candidate[~injected_mask]
    background = feature.index_select(0, retained_index)
    foreground = feature.index_select(0, injected_index)
    n_b, n_f = int(background.shape[0]), int(foreground.shape[0])
    n = n_b + n_f
    p = n_f / n
    mean_b, mean_f = background.mean(0), foreground.mean(0)
    within_b = (1.0 - p) * ((background - mean_b) @ basis).square().mean(0)
    within_f = p * ((foreground - mean_f) @ basis).square().mean(0)
    between = p * (1.0 - p) * ((mean_f - mean_b) @ basis).square()
    total = within_b + within_f + between
    contamination = within_f + between
    share = contamination / total.clamp_min(1e-24)
    decomposition_error = float(
        (total - ((feature.index_select(0, candidate) - feature.index_select(0, candidate).mean(0)) @ basis).square().mean(0))
        .abs()
        .max()
    )
    metrics = {
        "pollution_effective_pcs": energy_effective_count(contamination),
        "pollution_k80": k_fraction(contamination),
        "pollution_total_share": float(contamination.sum() / total.sum().clamp_min(1e-24)),
        "pollution_decomposition_max_abs_error": decomposition_error,
    }
    rows = []
    for index in range(RANK):
        rows.append(
            {
                "pc": index + 1,
                "background_within": float(within_b[index]),
                "foreground_within": float(within_f[index]),
                "between_group": float(between[index]),
                "contamination_contribution": float(contamination[index]),
                "total_variance": float(total[index]),
                "contamination_share": float(share[index]),
            }
        )
    return metrics, rows


def metric_drop(clean_item: dict, contaminated_item: dict) -> dict:
    output = {}
    clean = clean_item["patch_metrics_noninjected"]
    current = contaminated_item["patch_metrics_noninjected"]
    for method in ("gbsp", "knn8"):
        for metric in ("AP", "AUROC"):
            output[f"{method}_{metric.lower()}_drop_noninj"] = float(
                clean[method][metric] - current[method][metric]
            )
    return output


def image_average(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["stem"], row["p"], row["mode"])].append(row)
    output = []
    for (dataset, stem, p, mode), values in grouped.items():
        item = {"dataset": dataset, "stem": stem, "p": p, "mode": mode, "seed_count": len(values)}
        for metric in METRICS:
            finite = [float(row[metric]) for row in values if math.isfinite(float(row[metric]))]
            item[metric] = float(np.mean(finite)) if finite else float("nan")
        for index in range(1, RANK + 1):
            for prefix in ("principal_angle", "clean_pc"):
                field = (
                    f"principal_angle_{index}_deg"
                    if prefix == "principal_angle"
                    else f"clean_pc{index}_alignment_loss"
                )
                item[field] = float(np.mean([float(row[field]) for row in values]))
        output.append(item)
    return sorted(output, key=lambda row: (row["p"], row["mode"], row["dataset"], row["stem"]))


def dataset_macro_summary(rows: list[dict]) -> list[dict]:
    output = []
    for p in sorted({float(row["p"]) for row in rows}):
        for mode in ("random", "cluster"):
            selected = [row for row in rows if float(row["p"]) == p and row["mode"] == mode]
            item = {"p": p, "mode": mode, "image_count": len(selected)}
            for metric in METRICS:
                dataset_means = []
                for dataset in sorted({row["dataset"] for row in selected}):
                    values = [float(row[metric]) for row in selected if row["dataset"] == dataset]
                    values = [value for value in values if math.isfinite(value)]
                    if values:
                        dataset_means.append(float(np.mean(values)))
                item[metric] = float(np.mean(dataset_means)) if dataset_means else float("nan")
            output.append(item)
    return output


def paired_bootstrap(rows: list[dict], repetitions: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for p in sorted({float(row["p"]) for row in rows}):
        by_mode = {
            mode: {(row["dataset"], row["stem"]): row for row in rows if row["mode"] == mode and float(row["p"]) == p}
            for mode in ("random", "cluster")
        }
        keys = sorted(by_mode["random"].keys() & by_mode["cluster"].keys())
        for metric in METRICS:
            per_dataset = {}
            for dataset in sorted({key[0] for key in keys}):
                values = np.array(
                    [
                        float(by_mode["random"][key][metric])
                        - float(by_mode["cluster"][key][metric])
                        for key in keys
                        if key[0] == dataset
                        and math.isfinite(float(by_mode["random"][key][metric]))
                        and math.isfinite(float(by_mode["cluster"][key][metric]))
                    ],
                    dtype=np.float64,
                )
                if values.size:
                    per_dataset[dataset] = values
            estimate = float(np.mean([values.mean() for values in per_dataset.values()]))
            bootstrap = np.empty(repetitions, dtype=np.float64)
            for iteration in range(repetitions):
                bootstrap[iteration] = np.mean(
                    [values[rng.integers(0, len(values), len(values))].mean() for values in per_dataset.values()]
                )
            output.append(
                {
                    "p": p,
                    "metric": metric,
                    "contrast": "random_minus_cluster",
                    "image_count": sum(len(values) for values in per_dataset.values()),
                    "estimate": estimate,
                    "ci95_low": float(np.quantile(bootstrap, .025)),
                    "ci95_high": float(np.quantile(bootstrap, .975)),
                    "repetitions": repetitions,
                    "bootstrap_unit": "image_after_seed_average_dataset_stratified",
                }
            )
    return output


def correlations(rows: list[dict]) -> list[dict]:
    output = []
    predictors = (
        "delta_energy_effective_rank",
        "rotation_total_sin2",
        "rotation_effective_count",
        "pollution_effective_pcs",
    )
    for p in sorted({float(row["p"]) for row in rows}):
        for mode in ("random", "cluster"):
            selected = [row for row in rows if float(row["p"]) == p and row["mode"] == mode]
            for predictor in predictors:
                x = np.asarray([float(row[predictor]) for row in selected])
                y = np.asarray([float(row["gbsp_ap_drop_noninj"]) for row in selected])
                valid = np.isfinite(x) & np.isfinite(y)
                rho, pvalue = spearmanr(x[valid], y[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
                output.append(
                    {
                        "p": p,
                        "mode": mode,
                        "predictor": predictor,
                        "outcome": "gbsp_ap_drop_noninj",
                        "image_count": int(valid.sum()),
                        "spearman_rho": float(rho),
                        "p_value": float(pvalue),
                    }
                )
    return output


def plot_angles(out: Path, rows: list[dict]) -> None:
    p_values = sorted({float(row["p"]) for row in rows})
    figure, axes = plt.subplots(1, len(p_values), figsize=(4.5 * len(p_values), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, p in zip(axes, p_values):
        for mode, color in (("random", "tab:orange"), ("cluster", "tab:blue")):
            selected = [row for row in rows if float(row["p"]) == p and row["mode"] == mode]
            matrix = np.asarray([[float(row[f"principal_angle_{index}_deg"]) for index in range(1, 9)] for row in selected])
            mean = np.nanmean(matrix, axis=0)
            low, high = np.nanquantile(matrix, [.25, .75], axis=0)
            x = np.arange(1, 9)
            axis.plot(x, mean, marker="o", label=mode, color=color)
            axis.fill_between(x, low, high, alpha=.18, color=color)
        axis.set_title(f"contamination={p:.0%}")
        axis.set_xlabel("sorted principal-angle index")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("principal angle (degrees)")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(out / "principal_angle_profiles.png", dpi=180)
    plt.close(figure)


def plot_delta_spectrum(out: Path, spectra: list[dict]) -> None:
    p_values = sorted({float(row["p"]) for row in spectra})
    figure, axes = plt.subplots(1, len(p_values), figsize=(4.5 * len(p_values), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, p in zip(axes, p_values):
        for mode, color in (("random", "tab:orange"), ("cluster", "tab:blue")):
            selected = [row for row in spectra if float(row["p"]) == p and row["mode"] == mode]
            maximum = min(40, max(len(row["energy"]) for row in selected))
            curves = []
            for row in selected:
                cumulative = np.cumsum(row["energy"])
                if len(cumulative) < maximum:
                    cumulative = np.pad(cumulative, (0, maximum - len(cumulative)), constant_values=1.0)
                curves.append(cumulative[:maximum])
            matrix = np.asarray(curves)
            mean = np.nanmean(matrix, axis=0)
            low, high = np.nanquantile(matrix, [.25, .75], axis=0)
            x = np.arange(1, maximum + 1)
            axis.plot(x, mean, label=mode, color=color)
            axis.fill_between(x, low, high, alpha=.18, color=color)
        axis.axhline(.8, color="black", linestyle="--", linewidth=.8)
        axis.set_title(f"contamination={p:.0%}")
        axis.set_xlabel("perturbation directions")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("cumulative $||\\Delta\\Sigma||_F^2$ fraction")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(out / "covariance_perturbation_spectrum.png", dpi=180)
    plt.close(figure)


def plot_pollution_heatmap(out: Path, pc_rows: list[dict]) -> None:
    labels, matrix = [], []
    for p in sorted({float(row["p"]) for row in pc_rows}):
        for mode in ("cluster", "random"):
            selected = [row for row in pc_rows if float(row["p"]) == p and row["mode"] == mode]
            matrix.append(
                [
                    float(np.mean([float(row["contamination_share"]) for row in selected if int(row["pc"]) == pc]))
                    for pc in range(1, 9)
                ]
            )
            labels.append(f"{mode} {p:.0%}")
    figure, axis = plt.subplots(figsize=(8, 4.2))
    image = axis.imshow(np.asarray(matrix), aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(np.arange(8), [f"PC{index}" for index in range(1, 9)])
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Foreground-driven share of contaminated PCA variance")
    figure.colorbar(image, ax=axis, label="share")
    figure.tight_layout()
    figure.savefig(out / "pc_pollution_heatmap.png", dpi=180)
    plt.close(figure)


def report(out: Path, summary: list[dict], bootstrap: list[dict], validity: dict) -> None:
    lookup = {(float(row["p"]), row["mode"]): row for row in summary}
    boot = {(float(row["p"]), row["metric"]): row for row in bootstrap}
    lines = [
        "# PCA Subspace Damage Audit",
        "",
        f"- Images attempted: {validity['images_attempted']}",
        f"- Common valid conditions: {validity['common_valid_conditions']}",
        f"- Background-removal mismatches: {validity['background_removal_mismatches']}",
        f"- PCA numerical fallbacks: {validity['pca_fallbacks']}",
        f"- Delta-spectrum numerical fallbacks: {validity['delta_fallbacks']}",
        "- Training/DINO extraction/pseudo-label generation: false",
        "",
        "## Dataset-macro results",
        "",
        "| p | mode | Delta effective rank | Delta K80 | Rotation total | Effective rotated directions | Max angle | Polluted-PC effective count | GBSP AP drop |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted({key[0] for key in lookup}):
        for mode in ("cluster", "random"):
            row = lookup[(p, mode)]
            lines.append(
                f"| {p:.0%} | {mode} | {row['delta_energy_effective_rank']:.4f} | "
                f"{row['delta_k80']:.3f} | {row['rotation_total_sin2']:.4f} | "
                f"{row['rotation_effective_count']:.4f} | {row['principal_angle_max_deg']:.3f}° | "
                f"{row['pollution_effective_pcs']:.4f} | {row['gbsp_ap_drop_noninj']:.4f} |"
            )
    lines += ["", "## Paired random-minus-cluster contrasts", ""]
    for p in sorted({key[0] for key in lookup}):
        lines.append(f"### {p:.0%}")
        lines.append("")
        for metric in (
            "delta_energy_effective_rank",
            "delta_k80",
            "rotation_total_sin2",
            "rotation_effective_count",
            "principal_angle_max_deg",
            "pollution_effective_pcs",
            "gbsp_ap_drop_noninj",
        ):
            row = boot[(p, metric)]
            lines.append(
                f"- `{metric}`: {row['estimate']:+.6f}, 95% CI "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
            )
        lines.append("")
    (out / "PCA_SUBSPACE_DAMAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    random_root, cluster_root = Path(args.random_root).resolve(), Path(args.cluster_root).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    random_manifest, cluster_manifest = read_manifest(random_root), read_manifest(cluster_root)
    identities = sorted(
        random_manifest.keys() & cluster_manifest.keys(),
        key=lambda key: (DATASETS.index(key[0]) if key[0] in DATASETS else 99, key[1]),
    )
    if args.max_samples >= 0:
        identities = identities[: args.max_samples]
    p_values = tuple(sorted(set(float(value) for value in args.p_values)))
    seeds = tuple(int(value) for value in args.seeds)
    records: list[dict] = []
    pc_rows: list[dict] = []
    spectra: list[dict] = []
    failures: list[dict] = []
    background_removal_mismatches = 0
    pca_fallbacks = 0
    delta_fallbacks = 0
    max_decomposition_error = 0.0

    for image_index, identity in enumerate(identities, 1):
        dataset, stem = identity
        try:
            payload = {
                "random": torch.load(random_manifest[identity], map_location="cpu", weights_only=False),
                "cluster": torch.load(cluster_manifest[identity], map_location="cpu", weights_only=False),
            }
            if payload["random"]["settings"]["contamination_mode"] != "random_scattered_matched":
                raise RuntimeError("random cache has the wrong contamination mode")
            if payload["cluster"]["settings"]["contamination_mode"] != "spatial_cluster":
                raise RuntimeError("cluster cache has the wrong contamination mode")
            feature_payload = torch.load(payload["random"]["source_feature_path"], map_location="cpu", weights_only=False)
            feature = normalize_patch_features(feature_payload["tensor"]).to(device)
            conditions = {mode: condition_map(value) for mode, value in payload.items()}
            for seed in seeds:
                base_random = matched_base(payload["random"], seed)
                base_cluster = matched_base(payload["cluster"], seed)
                if not torch.equal(base_random, base_cluster):
                    raise RuntimeError("matched-size clean bases differ between modes")
                base = base_random
                clean_mean, clean_basis, _, fallback = fit_pca(feature, base)
                pca_fallbacks += int(fallback)
                clean_item = conditions["random"].get((seed, 0.0))
                cluster_clean_item = conditions["cluster"].get((seed, 0.0))
                if clean_item is None or cluster_clean_item is None:
                    raise RuntimeError("p=0 matched baseline is missing")
                for p in p_values:
                    random_item = conditions["random"].get((seed, p))
                    cluster_item = conditions["cluster"].get((seed, p))
                    if random_item is None or cluster_item is None:
                        continue
                    if not torch.equal(
                        random_item["removed_background_indices"],
                        cluster_item["removed_background_indices"],
                    ):
                        background_removal_mismatches += 1
                        continue
                    for mode, item in (("random", random_item), ("cluster", cluster_item)):
                        candidate = contaminated_candidate(base, item)
                        contaminated_mean, basis, _, fallback = fit_pca(feature, candidate)
                        pca_fallbacks += int(fallback)
                        row = {
                            "dataset": dataset,
                            "stem": stem,
                            "seed": seed,
                            "p": p,
                            "mode": mode,
                            "candidate_count": int(candidate.numel()),
                            "injected_count": int(item["injected_foreground_indices"].numel()),
                            "mean_shift_norm": float(torch.linalg.vector_norm(contaminated_mean - clean_mean)),
                        }
                        row.update(principal_angle_metrics(clean_basis, basis))
                        delta, spectrum, fallback = covariance_delta_metrics(
                            feature, base, item, clean_mean, contaminated_mean
                        )
                        delta_fallbacks += int(fallback)
                        row.update(delta)
                        pollution, current_pc_rows = pollution_metrics(feature, candidate, item, basis)
                        row.update(pollution)
                        max_decomposition_error = max(
                            max_decomposition_error,
                            float(pollution["pollution_decomposition_max_abs_error"]),
                        )
                        row.update(metric_drop(clean_item, item))
                        records.append(row)
                        for pc_row in current_pc_rows:
                            pc_rows.append(
                                {"dataset": dataset, "stem": stem, "seed": seed, "p": p, "mode": mode, **pc_row}
                            )
                        spectra.append(
                            {"dataset": dataset, "stem": stem, "seed": seed, "p": p, "mode": mode, "energy": spectrum.detach().cpu().double().numpy()}
                        )
        except Exception as error:
            failures.append({"dataset": dataset, "stem": stem, "error": repr(error)})
        if image_index % max(1, args.progress_every) == 0 or image_index == len(identities):
            print(
                f"[{image_index}/{len(identities)}] rows={len(records)} failures={len(failures)}",
                flush=True,
            )

    write_json(out / "failures.json", failures)
    if not records:
        raise RuntimeError(f"audit produced zero paired rows; failures={failures[:3]}")

    image_rows = image_average(records)
    summary = dataset_macro_summary(image_rows)
    bootstrap = paired_bootstrap(image_rows, args.bootstrap_repetitions, args.bootstrap_seed)
    correlation_rows = correlations(image_rows)
    spectrum_rows = []
    for row in spectra:
        cumulative = np.cumsum(row["energy"])
        for direction, (energy, cumulative_energy) in enumerate(zip(row["energy"], cumulative), 1):
            spectrum_rows.append(
                {
                    "dataset": row["dataset"],
                    "stem": row["stem"],
                    "seed": row["seed"],
                    "p": row["p"],
                    "mode": row["mode"],
                    "direction": direction,
                    "energy_fraction": float(energy),
                    "cumulative_energy_fraction": float(cumulative_energy),
                }
            )
    write_csv(out / "per_image_seed_metrics.csv", records)
    write_csv(out / "per_image_metrics.csv", image_rows)
    write_csv(out / "dataset_macro_summary.csv", summary)
    write_csv(out / "paired_bootstrap_random_minus_cluster.csv", bootstrap)
    write_csv(out / "pc_pollution_contributions.csv", pc_rows)
    write_csv(out / "covariance_delta_spectra.csv", spectrum_rows)
    write_csv(out / "geometry_performance_correlations.csv", correlation_rows)
    plot_angles(out, image_rows)
    plot_delta_spectrum(out, spectra)
    plot_pollution_heatmap(out, pc_rows)
    validity = {
        "images_attempted": len(identities),
        "images_failed": len(failures),
        "common_valid_conditions": len(records) // 2,
        "paired_mode_rows": len(records),
        "background_removal_mismatches": background_removal_mismatches,
        "pca_fallbacks": pca_fallbacks,
        "delta_fallbacks": delta_fallbacks,
        "max_pollution_decomposition_abs_error": max_decomposition_error,
        "device": str(device),
        "rank": RANK,
        "p_values": p_values,
        "seeds": seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "training_used": False,
        "dino_extraction_used": False,
        "pseudo_label_generated": False,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out / "validity_summary.json", validity)
    report(out, summary, bootstrap, validity)
    print(json.dumps(validity, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} image audits failed")
    if background_removal_mismatches:
        raise RuntimeError(f"{background_removal_mismatches} paired removal mismatches")


if __name__ == "__main__":
    main()
