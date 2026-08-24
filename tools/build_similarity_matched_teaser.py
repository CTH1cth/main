#!/usr/bin/env python3
"""Build a paper-ready Full-BC NN-similarity-matched GBSP visualization.

Scores are generated without GT. GT is loaded only afterwards to form same-image
foreground/background pairs whose nearest Full-BC similarity differs by at most
``match_tolerance``. The script streams the frozen feature/core caches and does
not write per-image score payloads.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from tools.similarity_variation_common import (  # noqa: E402
    labels_from_area,
    patch_gt,
    read_jsonl,
    similarity_matched_pairs,
)


GRID = 37
ALIASES = {"TE-CAMO": "CAMO", "TE-COD10K": "COD10K"}
DATASETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")
COLORS = {"fg": "#D55E00", "bg": "#0072B2", "residual": "#7B3294"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--match_tolerance", type=float, default=0.01)
    parser.add_argument("--high_similarity_quantile", type=float, default=0.80)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def _inside_workdir(path: Path) -> None:
    allowed = (PROJECT_ROOT / "workdir").resolve()
    path.resolve().relative_to(allowed)


def _find_manifest(root: Path) -> Path:
    for name in ("manifest_test.jsonl", "manifest.jsonl"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing core manifest under {root}")


def _feature_tensor(payload: dict) -> torch.Tensor:
    for key in ("patch_tokens", "features", "tensor", "feature"):
        value = payload.get(key)
        if torch.is_tensor(value) and value.numel() == 384 * GRID * GRID:
            tensor = value.squeeze()
            break
    else:
        values = [
            value.squeeze()
            for value in payload.values()
            if torch.is_tensor(value) and value.numel() == 384 * GRID * GRID
        ]
        if len(values) != 1:
            raise KeyError("cannot uniquely resolve the 37x37 DINO feature tensor")
        tensor = values[0]
    if tensor.shape == (384, GRID, GRID):
        tensor = tensor.permute(1, 2, 0).reshape(GRID * GRID, 384)
    elif tensor.shape != (GRID * GRID, 384):
        tensor = tensor.reshape(GRID * GRID, 384)
    return tensor.float()


@torch.inference_mode()
def _nearest_fullbc(
    feature: torch.Tensor, background_indices: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    x = F.normalize(feature.to(device), dim=1, eps=1e-12)
    bg_idx = background_indices.to(device=device, dtype=torch.long).flatten()
    similarity = x @ x[bg_idx].T
    similarity[bg_idx, torch.arange(bg_idx.numel(), device=device)] = -torch.inf
    maximum, position = similarity.max(dim=1)
    nearest = bg_idx[position]
    if bool((nearest[bg_idx] == bg_idx).any()):
        raise RuntimeError("leave-one-out self-match violation")
    return maximum.cpu().numpy(), nearest.cpu().numpy()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_summary(delta: np.ndarray, dataset_code: np.ndarray) -> list[dict]:
    rows = []
    for dataset in (*DATASETS, "OVERALL"):
        keep = np.ones(delta.shape, dtype=bool) if dataset == "OVERALL" else dataset_code == DATASETS.index(dataset)
        value = delta[keep]
        rows.append(
            {
                "dataset": dataset,
                "valid_pairs": int(value.size),
                "gbsp_pair_win_rate": float(np.mean(value > 0)) if value.size else np.nan,
                "delta_q_mean": float(np.mean(value)) if value.size else np.nan,
                "delta_q_median": float(np.median(value)) if value.size else np.nan,
            }
        )
    return rows


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(fig)


def _patch_bounds(index: int, width: int, height: int, expand: float = 1.0) -> tuple[float, float, float, float]:
    row, col = divmod(int(index), GRID)
    cell_w, cell_h = width / GRID, height / GRID
    center_x, center_y = (col + 0.5) * cell_w, (row + 0.5) * cell_h
    return (
        center_x - 0.5 * expand * cell_w,
        center_y - 0.5 * expand * cell_h,
        expand * cell_w,
        expand * cell_h,
    )


def _crop_patch(image: np.ndarray, index: int, expand: float = 4.0) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, w, h = _patch_bounds(index, width, height, expand)
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
    return image[y0:y1, x0:x1]


def _draw_boxes(axis: plt.Axes, image: np.ndarray, fg: int, bg: int) -> None:
    axis.imshow(image)
    for index, color, label in ((fg, COLORS["fg"], "FG"), (bg, COLORS["bg"], "BG")):
        x, y, w, h = _patch_bounds(index, image.shape[1], image.shape[0], 2.2)
        axis.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2.0, edgecolor=color))
        axis.text(x, max(0, y - 3), label, color="white", fontsize=8, weight="bold",
                  bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5})
    axis.axis("off")


def _projection_schematic(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.add_patch(Ellipse((0, 0), 4.8, 1.45, angle=18, color="#D9EAF3", alpha=0.9))
    t = np.linspace(-2.0, 2.0, 7)
    axis.scatter(t, 0.32 * t + 0.07 * np.sin(3 * t), s=22, color=COLORS["bg"], label="Full-BC")
    axis.plot([-2.4, 2.4], [-0.77, 0.77], color="#4D4D4D", lw=1.5)
    query = np.array([1.25, 2.05])
    direction = np.array([1.0, 0.32]); direction /= np.linalg.norm(direction)
    projection = direction * float(query @ direction)
    axis.scatter(*query, s=58, color=COLORS["fg"], zorder=4)
    axis.scatter(*projection, s=35, color="#4D4D4D", zorder=4)
    axis.add_patch(FancyArrowPatch(projection, query, arrowstyle="-|>", mutation_scale=13,
                                  lw=2.2, color=COLORS["residual"]))
    axis.text(query[0] + .08, query[1] + .08, r"query $x$", fontsize=9)
    axis.text(projection[0] - .35, projection[1] - .35, r"$\hat{x}$", fontsize=9)
    axis.text(1.5, 1.45, r"$Q=\|x-\hat{x}\|_2^2$", fontsize=10, color=COLORS["residual"])
    axis.text(-2.25, -1.05, "dominant background subspace", fontsize=9)
    axis.set_xlim(-2.7, 2.8); axis.set_ylim(-1.55, 2.65); axis.axis("off")


def _plot_evidence(
    payload: dict[str, np.ndarray], summaries: list[dict], high_summary: list[dict], out: Path
) -> None:
    delta = payload["delta_q"]
    dsim = payload["similarity_fg"] - payload["similarity_bg"]
    code = payload["dataset_code"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    sample = np.linspace(0, len(dsim) - 1, min(100_000, len(dsim)), dtype=int)
    axes[0].hexbin(payload["similarity_bg"][sample], payload["similarity_fg"][sample],
                   gridsize=55, bins="log", cmap="Blues", mincnt=1)
    lo = min(payload["similarity_bg"][sample].min(), payload["similarity_fg"][sample].min())
    hi = max(payload["similarity_bg"][sample].max(), payload["similarity_fg"][sample].max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="#444444", lw=1)
    axes[0].set(xlabel=r"Matched BG similarity $S_{BG}$", ylabel=r"FG similarity $S_{FG}$",
                title=r"(a) Controlled NN similarity ($|\Delta S|\leq0.01$)")

    qlo, qhi = np.quantile(delta, [0.005, 0.995])
    axes[1].hist(np.clip(delta, qlo, qhi), bins=90, density=True, color=COLORS["residual"], alpha=.85)
    axes[1].axvline(0, color="#333333", lw=1.3, ls="--")
    axes[1].axvline(np.median(delta), color="#F0E442", lw=2.0)
    overall = next(row for row in summaries if row["dataset"] == "OVERALL")
    axes[1].text(.98, .96,
                 f"win rate = {overall['gbsp_pair_win_rate']:.3f}\nmedian ΔQ = {overall['delta_q_median']:+.4f}",
                 transform=axes[1].transAxes, va="top", ha="right", fontsize=9,
                 bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#BBBBBB"})
    axes[1].set(xlabel=r"Residual gap $\Delta Q=Q_{FG}-Q_{BG}$", ylabel="Density",
                title="(b) Residual separates matched pairs")

    names = list(DATASETS) + ["Overall", "High-sim"]
    values = [next(row for row in summaries if row["dataset"] == name)["gbsp_pair_win_rate"] for name in DATASETS]
    values += [overall["gbsp_pair_win_rate"], next(row for row in high_summary if row["dataset"] == "OVERALL")["gbsp_pair_win_rate"]]
    bars = axes[2].bar(np.arange(len(names)), values, color=["#56B4E9"] * 4 + ["#009E73", "#D55E00"])
    axes[2].axhline(.5, color="#333333", lw=1.2, ls="--")
    axes[2].set_ylim(.45, .70); axes[2].set_ylabel("P($Q_{FG}>Q_{BG}$)")
    axes[2].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[2].set_title("(c) Consistent across datasets")
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axes[2].text(bar.get_x() + bar.get_width()/2, value + .005, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out / "figures" / "matched_pair_evidence")


def _plot_teaser(example: dict, payload: dict[str, np.ndarray], summaries: list[dict], out: Path) -> None:
    image = np.asarray(Image.open(example["image_path"]).convert("RGB"))
    gt = np.asarray(Image.open(example["gt_path"]).convert("L"))
    fig = plt.figure(figsize=(14.2, 4.5))
    outer = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.05, 1.25), wspace=.28)
    left = outer[0].subgridspec(2, 2, hspace=.18, wspace=.08)
    ax_image = fig.add_subplot(left[0, 0]); _draw_boxes(ax_image, image, example["fg_index"], example["bg_index"])
    ax_image.set_title("(a) Matched pair", loc="left", fontsize=11, weight="bold")
    ax_gt = fig.add_subplot(left[0, 1]); ax_gt.imshow(gt, cmap="gray"); ax_gt.axis("off"); ax_gt.set_title("GT", fontsize=9)
    ax_fg = fig.add_subplot(left[1, 0]); ax_fg.imshow(_crop_patch(image, example["fg_index"])); ax_fg.axis("off")
    ax_fg.set_title(f"FG: S={example['similarity_fg']:.3f}, Q={example['q_fg']:.3f}", fontsize=8, color=COLORS["fg"])
    ax_bg = fig.add_subplot(left[1, 1]); ax_bg.imshow(_crop_patch(image, example["bg_index"])); ax_bg.axis("off")
    ax_bg.set_title(f"BG: S={example['similarity_bg']:.3f}, Q={example['q_bg']:.3f}", fontsize=8, color=COLORS["bg"])

    ax_projection = fig.add_subplot(outer[1]); _projection_schematic(ax_projection)
    ax_projection.set_title("(b) Background subspace residual", loc="left", fontsize=11, weight="bold")

    right = outer[2].subgridspec(2, 1, hspace=.35)
    ax_hist = fig.add_subplot(right[0])
    delta = payload["delta_q"]
    qlo, qhi = np.quantile(delta, [.005, .995])
    ax_hist.hist(np.clip(delta, qlo, qhi), bins=75, density=True, color=COLORS["residual"], alpha=.85)
    ax_hist.axvline(0, color="#333333", ls="--", lw=1.2)
    ax_hist.axvline(np.median(delta), color="#F0E442", lw=2)
    ax_hist.set_title("(c) Matched-pair residual evidence", loc="left", fontsize=11, weight="bold")
    ax_hist.set_xlabel(r"$\Delta Q=Q_{FG}-Q_{BG}$"); ax_hist.set_ylabel("Density")
    ax_bar = fig.add_subplot(right[1])
    overall = next(row for row in summaries if row["dataset"] == "OVERALL")
    high = payload["high_similarity"].astype(bool)
    values = [overall["gbsp_pair_win_rate"], float(np.mean(delta[high] > 0))]
    bars = ax_bar.bar(["All matched", "High-sim matched"], values, color=["#009E73", "#D55E00"])
    ax_bar.axhline(.5, color="#333333", ls="--", lw=1.2); ax_bar.set_ylim(.45, .70)
    ax_bar.set_ylabel(r"P($Q_{FG}>Q_{BG}$)")
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x()+bar.get_width()/2, value+.008, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Beyond nearest-background similarity: orthogonal residual retains discriminative evidence",
                 fontsize=13, weight="bold", y=1.02)
    _save_figure(fig, out / "figures" / "paper_teaser")


def _plot_contact_sheet(examples: list[dict], out: Path) -> None:
    columns = ("Image", "GT", "FG query", "Matched BG", "FG nearest Full-BC", "BG nearest Full-BC")
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(15, 2.35 * len(examples)), squeeze=False)
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=10, weight="bold")
    for row_index, example in enumerate(examples):
        image = np.asarray(Image.open(example["image_path"]).convert("RGB"))
        gt = np.asarray(Image.open(example["gt_path"]).convert("L"))
        _draw_boxes(axes[row_index, 0], image, example["fg_index"], example["bg_index"])
        axes[row_index, 0].text(0, -0.08, f"{example['dataset']}/{example['stem']}", transform=axes[row_index, 0].transAxes,
                                fontsize=8, va="top")
        axes[row_index, 1].imshow(gt, cmap="gray"); axes[row_index, 1].axis("off")
        crops = (
            (example["fg_index"], f"S={example['similarity_fg']:.3f}\nQ={example['q_fg']:.3f}"),
            (example["bg_index"], f"S={example['similarity_bg']:.3f}\nQ={example['q_bg']:.3f}"),
            (example["fg_reference_index"], "Full-BC reference"),
            (example["bg_reference_index"], "Full-BC reference"),
        )
        for col, (index, caption) in enumerate(crops, 2):
            axes[row_index, col].imshow(_crop_patch(image, index))
            axes[row_index, col].axis("off"); axes[row_index, col].text(.5, -.06, caption, transform=axes[row_index, col].transAxes,
                                                                      ha="center", va="top", fontsize=7.5)
    fig.tight_layout()
    _save_figure(fig, out / "figures" / "real_matched_pairs_contact_sheet")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).resolve(); _inside_workdir(out); out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    rows = read_jsonl(_find_manifest(Path(args.core_root)))
    if args.max_samples >= 0:
        rows = rows[: args.max_samples]
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    candidates: list[tuple[float, int, dict]] = []
    failures: list[dict] = []
    serial = 0
    for number, row in enumerate(rows, 1):
        try:
            core = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
            r8 = core["results"]["r8"]
            if int(r8["selected_rank"]) != 8 or core.get("settings", {}).get("background_source") != "fullbc":
                raise RuntimeError("core payload is not frozen Full-BC GBSP-r8")
            feature_payload = torch.load(core["source_feature_path"], map_location="cpu", weights_only=False)
            feature = _feature_tensor(feature_payload)
            background = torch.as_tensor(r8["background_indices"], dtype=torch.long)
            similarity, nearest = _nearest_fullbc(feature, background, device)
            q = torch.as_tensor(r8["absolute_raw"]).float().reshape(-1).numpy()
            area = patch_gt(row["gt_path"])
            label, valid = labels_from_area(area, strict=False)
            fg, bg = similarity_matched_pairs(similarity, label, valid, args.match_tolerance)
            if fg.size:
                delta = q[fg] - q[bg]
                threshold = float(np.quantile(similarity, args.high_similarity_quantile))
                high = similarity[fg] >= threshold
                dataset = ALIASES.get(row["dataset"], row["dataset"])
                dataset_code = DATASETS.index(dataset)
                arrays["dataset_code"].append(np.full(fg.shape, dataset_code, dtype=np.int8))
                arrays["similarity_fg"].append(similarity[fg].astype(np.float32))
                arrays["similarity_bg"].append(similarity[bg].astype(np.float32))
                arrays["q_fg"].append(q[fg].astype(np.float32))
                arrays["q_bg"].append(q[bg].astype(np.float32))
                arrays["delta_q"].append(delta.astype(np.float32))
                arrays["high_similarity"].append(high.astype(np.uint8))
                strict = (area[fg] >= .8) & (area[bg] <= .2) & (delta > 0)
                if strict.any():
                    valid_positions = np.flatnonzero(strict)
                    # Favor a clear residual gap and high-similarity examples while
                    # penalizing a visible mismatch in the controlled similarity.
                    score = (delta[valid_positions]
                             - 5.0 * np.abs(similarity[fg[valid_positions]] - similarity[bg[valid_positions]])
                             + 0.15 * high[valid_positions])
                    best_local = int(np.argmax(score))
                    position = int(valid_positions[best_local])
                    f, b = int(fg[position]), int(bg[position])
                    example = {
                        "dataset": dataset, "stem": row["stem"],
                        "image_path": row["image_path"], "gt_path": row["gt_path"],
                        "fg_index": f, "bg_index": b,
                        "fg_reference_index": int(nearest[f]), "bg_reference_index": int(nearest[b]),
                        "similarity_fg": float(similarity[f]), "similarity_bg": float(similarity[b]),
                        "q_fg": float(q[f]), "q_bg": float(q[b]), "delta_q": float(q[f] - q[b]),
                        "similarity_gap": float(abs(similarity[f] - similarity[b])),
                        "high_similarity": bool(high[position]),
                    }
                    serial += 1
                    heapq.heappush(candidates, (float(score[best_local]), serial, example))
                    if len(candidates) > 200:
                        heapq.heappop(candidates)
        except Exception as error:
            failures.append({"dataset": row.get("dataset"), "stem": row.get("stem"), "error": repr(error)})
        if number % args.progress_every == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] valid={number-len(failures)} failed={len(failures)}", flush=True)
    if not arrays["delta_q"]:
        raise RuntimeError("no matched pairs were generated")
    payload = {key: np.concatenate(value) for key, value in arrays.items()}
    np.savez_compressed(out / "matched_pair_data.npz", **payload)
    summaries = _pair_summary(payload["delta_q"], payload["dataset_code"])
    high = payload["high_similarity"].astype(bool)
    high_summary = _pair_summary(payload["delta_q"][high], payload["dataset_code"][high])
    _write_csv(out / "matched_pair_summary.csv", summaries)
    _write_csv(out / "high_similarity_matched_pair_summary.csv", high_summary)

    ranked = [item[2] for item in sorted(candidates, reverse=True)]
    examples: list[dict] = []
    # First guarantee dataset diversity, then fill from the globally clearest pairs.
    for dataset in DATASETS:
        match = next((example for example in ranked if example["dataset"] == dataset), None)
        if match is not None:
            examples.append(match)
    for example in ranked:
        if len(examples) >= args.examples:
            break
        if example not in examples and sum(x["dataset"] == example["dataset"] for x in examples) < 2:
            examples.append(example)
    (out / "selected_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    (out / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    _plot_evidence(payload, summaries, high_summary, out)
    if examples:
        teaser_example = next((example for example in ranked if example["high_similarity"]), examples[0])
        _plot_teaser(teaser_example, payload, summaries, out)
        _plot_contact_sheet(examples, out)

    overall = next(row for row in summaries if row["dataset"] == "OVERALL")
    overall_high = next(row for row in high_summary if row["dataset"] == "OVERALL")
    full_run = len(rows) == 6473 and not failures
    report = f"""# Full-BC NN-similarity-matched GBSP 可视化

## 协议

- 输入：冻结的 DINOv1-S/8、37×37 patch、Current Full-BC、Global GBSP-r8。
- 控制变量：同图 FG/BG query 的最近 Full-BC 余弦相似度满足 `|ΔS| ≤ {args.match_tolerance}`。
- 分数生成不使用 GT；GT 仅用于事后定义 FG/BG 和挑选可视化 pair。
- 背景候选 query 严格 leave-one-out。
- 本次没有训练、没有重提 DINO，也没有重新生成逐图大缓存。

## 结果

- 状态：{'正式 6473 张完整复现' if full_run else f'烟测/部分运行（{len(rows)} 张，失败 {len(failures)}）'}。
- 全部匹配对：{overall['valid_pairs']:,}；GBSP pair win rate = {overall['gbsp_pair_win_rate']:.6f}；median ΔQ = {overall['delta_q_median']:+.6f}。
- 高相似匹配对：{overall_high['valid_pairs']:,}；GBSP pair win rate = {overall_high['gbsp_pair_win_rate']:.6f}；median ΔQ = {overall_high['delta_q_median']:+.6f}。

## 可安全表达的结论

> At nearly identical nearest-background similarity, foreground patches tend to exhibit larger orthogonal reconstruction residuals than matched background patches.

该图证明的是：**最近背景相似度相同，并不意味着背景子空间一致性相同；GBSP 正交残差仍保留额外判别信息。**

它不证明 GBSP 全面优于 KNN8，也不应写成“相似度方法无效”。

## 文件

- `figures/paper_teaser.*`：主 Teaser。
- `figures/matched_pair_evidence.*`：相似度控制、ΔQ 分布与分数据集胜率。
- `figures/real_matched_pairs_contact_sheet.*`：真实匹配 patch 与 Full-BC 最近参考。
- `matched_pair_data.npz`：绘图所用流式聚合数据。
- `selected_examples.json`：示例选择及精确索引/分数。
"""
    (out / "RESULTS.md").write_text(report)
    print(json.dumps({"images": len(rows), "failures": len(failures), "pairs": overall["valid_pairs"],
                      "win_rate": overall["gbsp_pair_win_rate"], "high_pairs": overall_high["valid_pairs"],
                      "high_win_rate": overall_high["gbsp_pair_win_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
