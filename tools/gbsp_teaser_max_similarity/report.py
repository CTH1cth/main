from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.gbsp_teaser_analysis.common import (
    BG_COLOR, FG_COLOR, GRID, load_settings, read_jsonl, save_vector_figure,
)
from tools.gbsp_teaser_analysis.plot_projection_schematic import draw_projection


def _find(rows: list[dict], **identity) -> dict:
    return next(row for row in rows if all(row.get(key) == value for key, value in identity.items()))


def _draw_hist(ax, bg, fg, bins, *, family: str, compact: bool = False) -> None:
    centers = (bins[:-1] + bins[1:]) / 2; width = np.diff(bins)
    if family == "similarity":
        bg_label, fg_label = "Background queries", "Camouflaged foreground queries"
        xlabel = "Maximum Background Similarity"; xlim = (-1, 1)
    else:
        bg_label, fg_label = "Background patches", "Camouflaged foreground patches"
        xlabel = "Normalized Residual Score"; xlim = (0, 1)
    ax.step(centers, bg / width, where="mid", color=BG_COLOR, lw=1.8, label=bg_label)
    ax.fill_between(centers, bg / width, step="mid", color=BG_COLOR, alpha=.25)
    ax.step(centers, fg / width, where="mid", color=FG_COLOR, lw=1.8, label=fg_label)
    ax.fill_between(centers, fg / width, step="mid", color=FG_COLOR, alpha=.25)
    ax.set_xlim(*xlim); ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=.16, linewidth=.6)
    ax.legend(frameon=False, fontsize=7 if compact else 8)


def _save_histograms(settings, out: Path) -> None:
    for protocol, suffix in (("allpatch_0.5", ""), ("core_0.2_0.8", "_core")):
        base = out / "CAMO" if not suffix else out / "CAMO/robustness_core"
        sim_dir = base / "max_similarity"; res_dir = base / "residual"
        bg, fg = np.load(sim_dir / "bg_hist.npy"), np.load(sim_dir / "fg_hist.npy")
        fig, ax = plt.subplots(figsize=(5.0, 3.6)); _draw_hist(ax, bg, fg, settings.similarity_bins, family="similarity")
        ax.set_title("Direct Background Similarity\nMaximum similarity to valid background patches on CAMO-Test", fontsize=10.5)
        fig.tight_layout(); save_vector_figure(fig, sim_dir / f"max_similarity_hist{suffix}"); plt.close(fig)
        bg, fg = np.load(res_dir / "bg_hist.npy"), np.load(res_dir / "fg_hist.npy")
        fig, ax = plt.subplots(figsize=(5.0, 3.6)); _draw_hist(ax, bg, fg, settings.residual_bins, family="residual")
        ax.set_title("GBSP Residual Evidence\nPatch-level unexplained residuals on CAMO-Test", fontsize=10.5)
        fig.tight_layout(); save_vector_figure(fig, res_dir / f"residual_hist{suffix}"); plt.close(fig)


def _case(result: dict) -> tuple[str, str]:
    if not result["formal_camo_complete"]:
        return "SMOKE_ONLY", "少样本只验证代码与输出结构，不得用于决定Fig.1。"
    metrics = result["metric_summary"]; deltas = result["delta_summary"]
    sim = _find(metrics, protocol="allpatch_0.5", method="1-max_valid_bg_similarity")
    gbsp = _find(metrics, protocol="allpatch_0.5", method="GBSP-r8 residual")
    core_sim = _find(metrics, protocol="core_0.2_0.8", method="1-max_valid_bg_similarity")
    core_gbsp = _find(metrics, protocol="core_0.2_0.8", method="GBSP-r8 residual")
    delta = _find(deltas, protocol="allpatch_0.5")
    fg = _find(result["descriptive_summary"], protocol="allpatch_0.5",
               family="max_valid_bg_similarity", group="Foreground")
    bg = _find(result["descriptive_summary"], protocol="allpatch_0.5",
               family="max_valid_bg_similarity", group="Background")
    diag = _find(result["distribution_diagnostics"], protocol="allpatch_0.5",
                 family="max_valid_bg_similarity")
    high_and_overlapping = bool(
        fg["value_q75"] >= bg["value_q25"] and fg["value_p_gt_0.7"] >= .25 and diag["OVL"] >= .5
    )
    pooled_gain = (gbsp["pooled_patch_AUROC"] - sim["pooled_patch_AUROC"],
                   gbsp["pooled_patch_AP"] - sim["pooled_patch_AP"])
    image_gain = (gbsp["imagewise_mean_AUROC"] - sim["imagewise_mean_AUROC"],
                  gbsp["imagewise_mean_AP"] - sim["imagewise_mean_AP"])
    core_gain = (core_gbsp["pooled_patch_AUROC"] - core_sim["pooled_patch_AUROC"],
                 core_gbsp["pooled_patch_AP"] - core_sim["pooled_patch_AP"])
    if pooled_gain[0] <= 0 and pooled_gain[1] <= 0 and image_gain[0] <= 0 and image_gain[1] <= 0:
        return "CASE_D", "GBSP residual在相同patch任务上并未提供更强ranking，不能声称更有判别力。"
    if (high_and_overlapping and min(pooled_gain) >= .01 and min(image_gain) >= .01 and
            min(core_gain) > 0 and delta["positive_image_ratio_AUROC"] > .5 and
            delta["positive_image_ratio_AP"] > .5):
        return "CASE_A", "强支持三栏Fig.1；仍只能写more discriminative evidence，不能写perfect separation。"
    if not high_and_overlapping and max(pooled_gain) <= .01:
        return "CASE_C", "真实数据仍不支持Similarity Ambiguity强Teaser，应停止继续更换相似度定义。"
    return "CASE_B", "部分支持：相似度对部分伪装query仍有歧义，但不能声称相似度本身无法区分。"


def _crop_patch(image: Image.Image, index: int) -> Image.Image:
    y, x = divmod(int(index), GRID); width, height = image.size
    box = (round(x * width / GRID), round(y * height / GRID),
           round((x + 1) * width / GRID), round((y + 1) * height / GRID))
    return image.crop(box).resize((112, 112), Image.Resampling.BICUBIC)


def _contact_sheet(out: Path) -> None:
    choices = []
    for row in read_jsonl(out / "CAMO/manifest.jsonl"):
        with np.load(row["cache_path"], allow_pickle=False) as payload:
            label = payload["gt_label_main"].astype(bool)
            score = payload["max_bg_similarity_main"]
            fg_indices = np.flatnonzero(label)
            if fg_indices.size == 0:
                continue
            index = int(fg_indices[np.argmax(score[fg_indices])])
            choices.append((float(score[index]), row, index, int(payload["matched_bg_patch_index_main"][index])))
    choices.sort(key=lambda item: item[0], reverse=True); choices = choices[:12]
    fig, axes = plt.subplots(4, 6, figsize=(11, 7.2))
    for pair, (score, row, fg_index, bg_index) in enumerate(choices):
        image = Image.open(row["image_path"]).convert("RGB")
        for offset, (index, title) in enumerate(((fg_index, "FG query"), (bg_index, "matched true BG"))):
            ax = axes.flat[2 * pair + offset]
            ax.imshow(_crop_patch(image, index)); ax.axis("off")
            ax.set_title(f"{title}\ncos={score:.3f}" if offset == 0 else title, fontsize=7)
    fig.suptitle("Highest maximum-valid-background matches (diagnostic only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    directory = out / "CAMO/examples"; directory.mkdir(parents=True, exist_ok=True)
    save_vector_figure(fig, directory / "high_similarity_contact_sheet", dpi=300); plt.close(fig)


def _teaser(settings, out: Path) -> None:
    sim_bg = np.load(out / "CAMO/max_similarity/bg_hist.npy")
    sim_fg = np.load(out / "CAMO/max_similarity/fg_hist.npy")
    res_bg = np.load(out / "CAMO/residual/bg_hist.npy")
    res_fg = np.load(out / "CAMO/residual/fg_hist.npy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), gridspec_kw={"width_ratios": [1.05, 1, 1.05]})
    _draw_hist(axes[0], sim_bg, sim_fg, settings.similarity_bins, family="similarity", compact=True)
    axes[0].set_title("Direct Background Similarity\nMaximum valid-BG similarity", fontsize=10)
    draw_projection(axes[1], compact=True)
    _draw_hist(axes[2], res_bg, res_fg, settings.residual_bins, family="residual", compact=True)
    axes[2].set_title("GBSP Residual Evidence\nUnexplained residual", fontsize=10)
    for x in (.335, .665):
        fig.text(x, .50, r"$\rightarrow$", ha="center", va="center", fontsize=21, color="#666666")
    fig.suptitle("Direct Background Similarity  →  Background Explainability  →  Residual Evidence", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=2.4)
    directory = out / "CAMO/teaser"; directory.mkdir(parents=True, exist_ok=True)
    save_vector_figure(fig, directory / "teaser_three_panel"); plt.close(fig)


def _write_audits(out: Path, result: dict) -> None:
    collection = json.loads((out / "audit/collection_summary.json").read_text())
    feature_audit = f"""# Feature / GT / Cache Audit

- Images: {result['images']}/250
- DINO: reused formal cached DINO-v1 ViT-S/8 37x37 key features
- Feature normalization/order/preprocessing: delegated to the validated formal GBSP cache loader
- GBSP: reused frozen Full-BC fixed-r8 raw and normalized residual; no recomputation with GT
- GBSP residual reproduction max mean error: {result['gbsp_cache_consistency']['max_abs_mean_error']:.3e} (formal check applicable: {result['gbsp_cache_consistency']['applicable']})
- GT use: held-out query labels and ideal valid-background reference set only
- Ground truth is used only for held-out diagnostic visualization and never enters the proposed GBSP inference pipeline.
"""
    runtime = collection["runtime_correctness"]
    implementation_audit = f"""# Maximum Similarity Implementation Audit

- Definition: per-query maximum cosine to same-image GT-valid background patches
- BG self match: explicitly set to -inf before argmax
- Sampled self checks: {runtime['self_checks']}; violations: {runtime['self_violations']}
- Brute-force maximum checks: {runtime['brute_force_checks']}
- Maximum score max absolute error: {runtime['max_score_abs_error']:.3e}
- Maximum index mismatches beyond ties: {runtime['max_index_mismatch']}
- Cosine checks: {runtime['cosine_checks']}
- Cosine max absolute error: {runtime['cosine_max_abs_error']:.3e}
- `max` is an existence diagnostic, not a KNN hyperparameter or UCOD baseline.
"""
    (out / "audit/FEATURE_GT_CACHE_AUDIT.md").write_text(feature_audit, encoding="utf-8")
    (out / "audit/MAX_SIM_IMPLEMENTATION_AUDIT.md").write_text(implementation_audit, encoding="utf-8")


def write_results(out: Path, result: dict) -> str:
    case, recommendation = _case(result)
    metrics = result["metric_summary"]; stats = result["descriptive_summary"]
    sim = _find(metrics, protocol="allpatch_0.5", method="1-max_valid_bg_similarity")
    gbsp = _find(metrics, protocol="allpatch_0.5", method="GBSP-r8 residual")
    sim_core = _find(metrics, protocol="core_0.2_0.8", method="1-max_valid_bg_similarity")
    gbsp_core = _find(metrics, protocol="core_0.2_0.8", method="GBSP-r8 residual")
    delta = _find(result["delta_summary"], protocol="allpatch_0.5")
    delta_core = _find(result["delta_summary"], protocol="core_0.2_0.8")
    bg = _find(stats, protocol="allpatch_0.5", family="max_valid_bg_similarity", group="Background")
    fg = _find(stats, protocol="allpatch_0.5", family="max_valid_bg_similarity", group="Foreground")
    diag = _find(result["distribution_diagnostics"], protocol="allpatch_0.5", family="max_valid_bg_similarity")
    report = f"""# Per-Query Maximum Valid-Background Similarity vs. GBSP Residual

- Status: `{case}`
- CAMO-Test: `{result['images']}/250`
- Formal completeness: `{result['formal_camo_complete']}`
- GT role: held-out diagnostic grouping and ideal valid-background reference set only

## Previous diagnostic

Previous all-pair diagnostic: `CASE_C`. Random FG--BG pairs were generally much less similar than BG--BG pairs. Therefore all-pair affinity overlap is no longer used as the teaser motivation.

本轮改为每个query对GT-clean背景集合的最大相似度，直接回答是否存在高度相似的valid background reference。`max`不是KNN超参数、1NN baseline或正式UCOD方法。

> Ground truth is used only for held-out diagnostic visualization and never enters the proposed GBSP inference pipeline.

## Maximum similarity to valid background

| Statistic | BG queries | FG queries |
|---|---:|---:|
| Mean | {bg['value_mean']:.6f} | {fg['value_mean']:.6f} |
| Median | {bg['value_median']:.6f} | {fg['value_median']:.6f} |
| Q25 | {bg['value_q25']:.6f} | {fg['value_q25']:.6f} |
| Q75 | {bg['value_q75']:.6f} | {fg['value_q75']:.6f} |
| Q90 | {bg['value_q90']:.6f} | {fg['value_q90']:.6f} |
| P(sim>.5) | {bg['value_p_gt_0.5']:.6f} | {fg['value_p_gt_0.5']:.6f} |
| P(sim>.6) | {bg['value_p_gt_0.6']:.6f} | {fg['value_p_gt_0.6']:.6f} |
| P(sim>.7) | {bg['value_p_gt_0.7']:.6f} | {fg['value_p_gt_0.7']:.6f} |
| P(sim>.8) | {bg['value_p_gt_0.8']:.6f} | {fg['value_p_gt_0.8']:.6f} |
| P(sim>.9) | {bg['value_p_gt_0.9']:.6f} | {fg['value_p_gt_0.9']:.6f} |

Image-balanced histogram: OVL=`{diag['OVL']:.6f}`, Wasserstein=`{diag['Wasserstein']:.6f}`.

## Same-patch foreground ranking diagnostic

| Protocol / method | Pooled AUROC | Pooled AP | Image-wise mean AUROC | Image-wise mean AP |
|---|---:|---:|---:|---:|
| 0.5 / 1-max valid-BG similarity | {sim['pooled_patch_AUROC']:.6f} | {sim['pooled_patch_AP']:.6f} | {sim['imagewise_mean_AUROC']:.6f} | {sim['imagewise_mean_AP']:.6f} |
| 0.5 / GBSP-r8 residual | {gbsp['pooled_patch_AUROC']:.6f} | {gbsp['pooled_patch_AP']:.6f} | {gbsp['imagewise_mean_AUROC']:.6f} | {gbsp['imagewise_mean_AP']:.6f} |
| core / 1-max valid-BG similarity | {sim_core['pooled_patch_AUROC']:.6f} | {sim_core['pooled_patch_AP']:.6f} | {sim_core['imagewise_mean_AUROC']:.6f} | {sim_core['imagewise_mean_AP']:.6f} |
| core / GBSP-r8 residual | {gbsp_core['pooled_patch_AUROC']:.6f} | {gbsp_core['pooled_patch_AP']:.6f} | {gbsp_core['imagewise_mean_AUROC']:.6f} | {gbsp_core['imagewise_mean_AP']:.6f} |

Per-image positive ratios (GBSP > similarity): AUROC=`{delta['positive_image_ratio_AUROC']:.6f}`, AP=`{delta['positive_image_ratio_AP']:.6f}`. Core: AUROC=`{delta_core['positive_image_ratio_AUROC']:.6f}`, AP=`{delta_core['positive_image_ratio_AP']:.6f}`.

## Required answers

1. FG max-valid-BG similarity：均值 `{fg['value_mean']:.6f}`，中位数 `{fg['value_median']:.6f}`。
2. BG/FG max-sim overlap：OVL `{diag['OVL']:.6f}`，以真实曲线和分位数共同判断。
3. FG的P(max_sim>.7/.8/.9)：`{fg['value_p_gt_0.7']:.6f}` / `{fg['value_p_gt_0.8']:.6f}` / `{fg['value_p_gt_0.9']:.6f}`。
4. 相较random FG--BG pair mean `0.021626`，FG maximum mean右移 `{fg['value_mean']-0.021626:+.6f}`。
5. Similarity foreground score pooled AUROC/AP：`{sim['pooled_patch_AUROC']:.6f}` / `{sim['pooled_patch_AP']:.6f}`。
6. GBSP pooled AUROC/AP：`{gbsp['pooled_patch_AUROC']:.6f}` / `{gbsp['pooled_patch_AP']:.6f}`。
7. GBSP相对similarity pooled增量：AUROC `{gbsp['pooled_patch_AUROC']-sim['pooled_patch_AUROC']:+.6f}`，AP `{gbsp['pooled_patch_AP']-sim['pooled_patch_AP']:+.6f}`。
8. Image-wise平均增量：AUROC `{gbsp['imagewise_mean_AUROC']-sim['imagewise_mean_AUROC']:+.6f}`，AP `{gbsp['imagewise_mean_AP']-sim['imagewise_mean_AP']:+.6f}`。
9. Core协议方向见表和positive ratio，不依赖mixed boundary patches。
10. Fig.1判定：`{case}`。{recommendation}

本诊断不构成“GBSP > 1NN”或SOTA claim；Maximum similarity仅验证valid background reference的存在性。
"""
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    return case


def build_outputs(config: str | Path, out_dir: str | Path, result: dict) -> str:
    settings = load_settings(config); out = Path(out_dir).resolve()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    _save_histograms(settings, out); _contact_sheet(out); _write_audits(out, result)
    case = write_results(out, result)
    if case == "CASE_A":
        _teaser(settings, out)
    return case
