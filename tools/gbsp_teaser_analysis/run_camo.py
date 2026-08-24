#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_analysis.analyze_camo import analyze  # noqa: E402
from tools.gbsp_teaser_analysis.build_teaser_preview import plot as plot_preview  # noqa: E402
from tools.gbsp_teaser_analysis.collect_camo import collect  # noqa: E402
from tools.gbsp_teaser_analysis.common import load_settings, validate_output  # noqa: E402
from tools.gbsp_teaser_analysis.plot_projection_schematic import plot as plot_projection  # noqa: E402
from tools.gbsp_teaser_analysis.plot_residual_hist import plot as plot_residual  # noqa: E402
from tools.gbsp_teaser_analysis.plot_similarity_hist import plot as plot_similarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_collect", action="store_true")
    return parser.parse_args()


def _find(rows: list[dict], protocol: str, group: str) -> dict:
    return next(row for row in rows if row["protocol"] == protocol and row["group"] == group)


def _diag(rows: list[dict], protocol: str, family: str) -> dict:
    return next(row for row in rows if row["protocol"] == protocol and row["family"] == family)


def _high_similarity_ambiguity(bb: dict, fb: dict, ovl: float) -> tuple[bool, dict[str, float]]:
    """Decide whether the *high-similarity* teaser premise is actually supported.

    Histogram overlap alone is insufficient: two broad distributions can overlap in
    their middle even when FG--BG pairs are essentially absent from the high-cosine
    tail.  The teaser premise therefore requires both substantial global overlap and
    a non-negligible FG--BG share relative to BG--BG at sim > 0.7.  The 0.8/0.9
    ratios remain descriptive diagnostics and are not used as tuned thresholds.
    """
    ratios = {}
    for threshold in (0.7, 0.8, 0.9):
        key = f"P(sim>{threshold})"
        ratios[str(threshold)] = float(fb[key] / bb[key]) if bb[key] > 0 else 0.0
    supported = bool(ovl >= 0.5 and ratios["0.7"] >= 0.25)
    return supported, ratios


def write_audit(settings, out: Path, summary: dict) -> None:
    audit = f"""# GBSP Teaser Score-Pipeline Audit

- Dataset: CAMO-Test only (`{summary['images']}` generated images)
- Formal feature: cached DINO-v1 ViT-S/8, input 296x296, 37x37 patch grid
- Patch order: row-major order from the formal cached tensor
- Feature normalization: per-patch L2 normalization, identical to formal GBSP input
- Left diagnostic: full 1369x1369 pairwise cosine matrix
- BG--BG policy: diagonal excluded and only the upper triangle retained
- FG--BG policy: all cross-class pairs retained
- Background dictionary used by left diagnostic: no
- KNN/nearest/prototype/Top-K used by left diagnostic: no
- Dataset aggregation: each image produces a unit-sum histogram; image histograms are averaged with equal weight
- Similarity bins: 80 fixed bins on [-1,1]
- GBSP source: frozen Full-BC fixed-r{settings.gbsp_rank} affine-PCA cache
- Residual: absolute squared projection residual followed by the formal per-image Min-Max
- Residual bins: 50 fixed bins on [0,1]
- Main GT grouping: foreground occupancy >=0.5, background occupancy <0.5
- Robustness GT grouping: foreground >=0.8, background <=0.2, mixed ignored
- GT is used only after DINO affinities and GBSP residuals have been produced
- Pairwise similarity is a representation-overlap diagnostic, not an implementable UCOD baseline
"""
    diagnostics = out / "diagnostics"; diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "PIPELINE_AUDIT.md").write_text(audit, encoding="utf-8")


def write_results(settings, out: Path, result: dict) -> None:
    sim = result["similarity_summary"]; res = result["residual_summary"]; internal = result["internal_diagnostics"]
    bb = _find(sim, "allpatch_0.5", "BG--BG"); fb = _find(sim, "allpatch_0.5", "FG--BG")
    bg = _find(res, "allpatch_0.5", "Background"); fg = _find(res, "allpatch_0.5", "Foreground")
    bb_core = _find(sim, "core_0.2_0.8", "BG--BG"); fb_core = _find(sim, "core_0.2_0.8", "FG--BG")
    bg_core = _find(res, "core_0.2_0.8", "Background"); fg_core = _find(res, "core_0.2_0.8", "Foreground")
    sim_diag = _diag(internal, "allpatch_0.5", "pairwise_similarity")
    res_diag = _diag(internal, "allpatch_0.5", "gbsp_residual")
    complete = bool(result["formal_camo_complete"])
    sim_overlap = "substantial" if sim_diag["OVL"] >= .5 else ("moderate" if sim_diag["OVL"] >= .3 else "limited")
    high_sim_ambiguity, tail_ratios = _high_similarity_ambiguity(bb, fb, sim_diag["OVL"])
    residual_tendency = fg["mean"] > bg["mean"] and fg["median"] > bg["median"]
    robustness = ((fb["mean"] - bb["mean"]) * (fb_core["mean"] - bb_core["mean"]) >= 0 and
                  (fg["mean"] - bg["mean"]) * (fg_core["mean"] - bg_core["mean"]) >= 0)
    if not complete:
        case = "SMOKE_ONLY"
        recommendation = "当前只验证代码与输出结构；不得据此决定Fig.1。"
    elif not high_sim_ambiguity:
        case = "CASE_C"
        recommendation = ("高相似度歧义这一左侧前提不成立，不建议采用当前三栏Teaser；"
                          "右侧GBSP residual证据可独立保留。")
    elif not residual_tendency or res_diag["OVL"] >= .9:
        case = "CASE_D"
        recommendation = "可保留projection schematic，但不应让residual分布承担empirical evidence。"
    elif res_diag["OVL"] < .5:
        case = "CASE_A"
        recommendation = "三栏Teaser成立；措辞仍保持为tendency而非perfect separation。"
    else:
        case = "CASE_B"
        recommendation = "三栏Teaser可用，但只写residual emphasizes unexplained deviations，不声称显著提升分离度。"
    note = "" if complete else (
        "\n> **这是少样本烟测，不是CAMO-Test正式结论。正式250张完成前，Case判定和图中趋势不可用于论文。**\n"
    )
    thresholds = settings.descriptive_thresholds
    report = f"""# GBSP Teaser Real-Data Results — CAMO-Test

- Status: `{case}`
- Images: `{result['images']}/250`
- Formal completeness: `{complete}`
- Aggregation: image-balanced histogram averaging
{note}
## Descriptive statistics: pairwise DINO cosine

以下是逐图统计量的图像等权平均；左图不使用背景字典、KNN、Top-K或reference aggregation。

| Diagnostic | BG--BG | FG--BG |
|---|---:|---:|
| Mean pairwise cosine | {bb['mean']:.6f} | {fb['mean']:.6f} |
| Median pairwise cosine | {bb['median']:.6f} | {fb['median']:.6f} |
| Q75 | {bb['q75']:.6f} | {fb['q75']:.6f} |
| Q90 | {bb['q90']:.6f} | {fb['q90']:.6f} |
| P(sim > 0.7) | {bb['P(sim>0.7)']:.6f} | {fb['P(sim>0.7)']:.6f} |
| P(sim > 0.8) | {bb['P(sim>0.8)']:.6f} | {fb['P(sim>0.8)']:.6f} |
| P(sim > 0.9) | {bb['P(sim>0.9)']:.6f} | {fb['P(sim>0.9)']:.6f} |

内部诊断：similarity OVL=`{sim_diag['OVL']:.6f}`，Wasserstein=`{sim_diag['Wasserstein']:.6f}`。

## Descriptive statistics: formal GBSP residual

| Residual | Background | Foreground |
|---|---:|---:|
| Mean normalized residual | {bg['mean']:.6f} | {fg['mean']:.6f} |
| Median | {bg['median']:.6f} | {fg['median']:.6f} |
| Q75 | {bg['q75']:.6f} | {fg['q75']:.6f} |
| Q90 | {bg['q90']:.6f} | {fg['q90']:.6f} |

内部诊断：residual OVL=`{res_diag['OVL']:.6f}`，Wasserstein=`{res_diag['Wasserstein']:.6f}`。

> 两侧随机变量不同，禁止直接比较两侧OVL或计算所谓相对改善百分比。

## Q1. BG--BG与FG--BG是否明显重叠？

按固定image-balanced histogram，当前重叠程度为 `{sim_overlap}`。正式判断以真实曲线及上述高相似度尾概率共同为准。

## Q2. FG--BG是否存在大量高相似度pair？

`{high_sim_ambiguity}`。FG--BG相对BG--BG的尾部比例分别为：sim>0.7 `{tail_ratios['0.7']:.6f}`、
sim>0.8 `{tail_ratios['0.8']:.6f}`、sim>0.9 `{tail_ratios['0.9']:.6f}`。这些仅为描述统计，不是模型阈值。

## Q3. 前景residual是否整体高于背景？

`{residual_tendency}`。前景均值与中位数分别相对背景变化 `{fg['mean']-bg['mean']:+.6f}` 和 `{fg['median']-bg['median']:+.6f}`。

## Q4. Residual是否具有可读趋势？

参见 `CAMO/residual/residual_hist.*`。只允许描述foreground-oriented tendency，不允许声称pure foreground或perfect separation。

## Q5. 0.5协议与core协议是否一致？

`{robustness}`。Core统计：similarity mean差值 `{fb_core['mean']-bb_core['mean']:+.6f}`；residual mean差值 `{fg_core['mean']-bg_core['mean']:+.6f}`。

## Q6. Fig.1建议

{recommendation}

Pairwise similarity is used only as a diagnostic of representation overlap. It is not treated as an implementable UCOD baseline.
"""
    (out / "RESULTS.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    settings = load_settings(args.config); out = validate_output(args.out_dir)
    if not args.skip_collect:
        collection = collect(Namespace(
            config=args.config, out_dir=str(out), max_samples=args.max_samples,
            device=args.device, failure_policy=args.failure_policy, overwrite=args.overwrite,
        ))
    result = analyze(args.config, out)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    plot_similarity(settings, out); plot_residual(settings, out); plot_projection(out); plot_preview(settings, out)
    write_audit(settings, out, result); write_results(settings, out, result)
    return result


def main() -> None:
    args = parse_args(); result = run(args)
    print(json.dumps({"status": "complete", "out_dir": str(Path(args.out_dir).resolve()),
                      "images": result["images"], "formal_camo_complete": result["formal_camo_complete"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
