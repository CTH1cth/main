#!/usr/bin/env python3
"""One-command runner for the frozen real-data GBSP teaser audit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.teaser_real_data.analyze_distributions import analyze  # noqa: E402
from tools.teaser_real_data.collect_scores import collect  # noqa: E402
from tools.teaser_real_data.common import load_settings  # noqa: E402
from tools.teaser_real_data.plot_projection_schematic import plot as plot_projection  # noqa: E402
from tools.teaser_real_data.plot_response_examples import plot as plot_examples  # noqa: E402
from tools.teaser_real_data.plot_score_distributions import plot as plot_distributions  # noqa: E402

EXPECTED = {"CAMO": 250, "COD10K": 2026, "NC4K": 4121}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max_samples_per_dataset", type=int, default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--failure_policy", choices=("record", "strict"), default="record")
    parser.add_argument("--skip_collect", action="store_true")
    return parser.parse_args()


def _lookup(rows: list[dict], dataset: str, protocol: str, method: str) -> dict | None:
    return next((row for row in rows if row["dataset"] == dataset and row["protocol"] == protocol and row["method"] == method), None)


def write_results(config: Path, out: Path, analysis: dict) -> None:
    settings = load_settings(config)
    validity = analysis["validity"]
    complete = validity["counts"] == EXPECTED
    rows = analysis["metrics"]
    camo_knn = _lookup(rows, "CAMO", "core", "KNN8")
    camo_gbsp = _lookup(rows, "CAMO", "core", "GBSP-r8")
    all_knn = _lookup(rows, "CAMO", "allpatch", "KNN8")
    all_gbsp = _lookup(rows, "CAMO", "allpatch", "GBSP-r8")
    if camo_knn is None or camo_gbsp is None:
        status = "INSUFFICIENT"
        recommendation = "尚不能选择 A/B/C。"
        trend = "数据不足。"
    else:
        ovl_delta = camo_gbsp["OVL"] - camo_knn["OVL"]
        w_delta = camo_gbsp["Wasserstein"] - camo_knn["Wasserstein"]
        improving = []
        for dataset in settings.diagnostic_datasets:
            knn = _lookup(rows, dataset, "core", "KNN8")
            gbsp = _lookup(rows, dataset, "core", "GBSP-r8")
            if knn and gbsp:
                improving.append(gbsp["OVL"] < knn["OVL"] and gbsp["Wasserstein"] > knn["Wasserstein"])
        if ovl_delta >= 0:
            status = "CASE_D"
            recommendation = "C. 放弃以可分性改善作为 distribution teaser；只保留机制图或真实 response 辅助。"
        elif abs(ovl_delta) < .01 or w_delta <= 0:
            status = "CASE_C"
            recommendation = "B. 只使用 projection conceptual schematic；真实分布最多放附录。"
        elif sum(improving) == len(improving) and abs(ovl_delta) >= .03:
            status = "CASE_A"
            recommendation = "A. 使用真实 distribution 作为 Fig.1 主体，并配合 projection schematic。"
        else:
            status = "CASE_B"
            recommendation = "A. 可以使用真实 distribution，但 caption 只能表述为 modestly reduces score ambiguity。"
        trend = f"{sum(improving)}/{len(improving)} 个已评数据集同时满足 OVL 降低且 Wasserstein 增大。"
    incomplete_note = "" if complete else (
        "\n> **注意：这是少样本代码烟测产物，不是正式 held-out 全量结论。"
        "在 CAMO=250、COD10K=2026、NC4K=4121 全部完成前，下面的 Case 与 A/B/C 仅用于检查报告逻辑，不得用于论文。**\n"
    )
    def fmt(row, key):
        return "N/A" if row is None else f"{float(row[key]):.6f}"
    core_robust = "N/A"
    if all_knn and all_gbsp and camo_knn and camo_gbsp:
        core_direction = camo_gbsp["OVL"] < camo_knn["OVL"]
        all_direction = all_gbsp["OVL"] < all_knn["OVL"]
        core_robust = "一致" if core_direction == all_direction else "不一致"
    report = f"""# GBSP Real-Data Teaser Results

- Run status: `{status}`
- Formal completeness: `{complete}`
- Counts: `{json.dumps(validity['counts'], ensure_ascii=False)}`
- Frozen setting: Full-BC, KNN-{settings.knn_k}, GBSP-r{settings.gbsp_rank}, per-image Min-Max
{incomplete_note}
## Q1. CAMO 上 KNN8 的 FG/BG 是否明显重叠？

KNN8 core OVL = `{fmt(camo_knn, 'OVL')}`。OVL 越接近 1，重叠越强；必须结合正式全量曲线作最终判断。

## Q2. GBSP residual 的重叠是否减少？

GBSP core OVL = `{fmt(camo_gbsp, 'OVL')}`，相对 KNN8 的差值为 `{('N/A' if not camo_knn or not camo_gbsp else f"{camo_gbsp['OVL']-camo_knn['OVL']:+.6f}")}`。

## Q3. CAMO 核心数值

| Method | OVL | Wasserstein | AUROC | AP |
|---|---:|---:|---:|---:|
| KNN8 | {fmt(camo_knn, 'OVL')} | {fmt(camo_knn, 'Wasserstein')} | {fmt(camo_knn, 'AUROC')} | {fmt(camo_knn, 'AP')} |
| GBSP-r8 | {fmt(camo_gbsp, 'OVL')} | {fmt(camo_gbsp, 'Wasserstein')} | {fmt(camo_gbsp, 'AUROC')} | {fmt(camo_gbsp, 'AP')} |

## Q4. Core 与 all-patch 趋势

两种协议按 OVL 方向判断为：`{core_robust}`。完整数值见 `diagnostics/per_dataset_summary.csv`。

## Q5. 三个数据集是否总体一致？

{trend}

## Q6. 对改善幅度的表述

若完整结果的改善很小，论文只能写：“The real-data distributions support only a modest separation improvement.” 不得画成完全分离。

## Q7. Response map 是否一致？

`response_examples/contact_sheet.*` 使用冻结随机种子、未按性能挑图，并用与直方图相同的 [0,1] response。图像必须人工核对；本报告不以挑图替代分布证据。

## Q8. Fig.1 建议

{recommendation}

## 审计结论

KNN8 与 GBSP 使用逐图完全相同的 Full-BC candidate indices；GT 只在分数生成后用于 patch 分组。没有扫描 K、rank、threshold、candidate ratio 或 normalization。
"""
    (out / "RESULTS.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = Path(args.config).resolve()
    out = Path(args.out_dir).resolve()
    if not args.skip_collect:
        collect(Namespace(
            config=str(config), out_dir=str(out), datasets=args.datasets,
            max_samples_per_dataset=args.max_samples_per_dataset, device=args.device,
            overwrite=args.overwrite, failure_policy=args.failure_policy,
        ))
    analysis = analyze(config, out, out)
    plot_distributions(config, out, out)
    plot_examples(config, out, out, count=8)
    plot_projection(out)
    write_results(config, out, analysis)
    print(json.dumps({"status": "complete", "out_dir": str(out), "validity": analysis["validity"]}, ensure_ascii=False, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
