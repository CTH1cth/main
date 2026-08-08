#!/usr/bin/env python3
"""Consolidate completed GBSP Threshold V3 stages into the final audit report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


STAGES = (
    ("baseline_test20", "baseline_test20"),
    ("eval_test20", "test20_core"),
    ("eval_test20_hysteresis", "test20_hysteresis"),
    ("eval_pilot200", "pilot200"),
    ("eval_full6473", "full6473"),
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _macro(stage_dir: Path) -> dict[str, dict]:
    rows = _read_csv(stage_dir / "per_dataset_metrics.csv")
    return {row["method"]: row for row in rows if row.get("scope") == "dataset_macro"}


def _copy_if_present(source: Path, target: Path) -> None:
    if source.is_file():
        shutil.copy2(source, target)


def _format(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{value:.4f}"


def _table(rows: list[dict]) -> str:
    fields = ("method", "S_m", "F_beta_w", "F_beta_mean", "E_mean", "MAE", "Precision", "Recall", "Area")
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([row.get("method", "")] + [_format(_number(row, field)) for field in fields[1:]]) + " |")
    return "\n".join(lines)


def _core_answers(root: Path) -> tuple[list[str], dict]:
    eval_dir = root / "eval_test20"
    macro = _macro(eval_dir)
    selection = _read_json(eval_dir / "method_selection.json", {}) or {}
    threshold = _read_csv(eval_dir / "threshold_distribution.csv")
    otgc = _read_csv(eval_dir / "otgc_parameter_distribution.csv")
    ut = _read_csv(eval_dir / "ut3cp_change_point_distribution.csv")
    fixed = macro.get("fixed_058", {})
    multi = macro.get("multi_otsu_3", {})

    multi_high = [_number(row, "threshold_high") for row in threshold if row.get("method") == "multi_otsu_3"]
    multi_low = [_number(row, "threshold_low") for row in threshold if row.get("method") == "multi_otsu_3"]
    finite_high = [value for value in multi_high if math.isfinite(value)]
    finite_low = [value for value in multi_low if math.isfinite(value)]
    area = _number(multi, "Area")
    otgc_valid = [row for row in otgc if str(row.get("optimizer_converged", "")).lower() in {"true", "1"}]
    otgc_monotone = [row for row in otgc if str(row.get("monotonicity_passed", "")).lower() in {"true", "1"}]
    otgc_bic = [_number(row, "delta_bic_3_vs_2") for row in otgc]
    ut_bic = [_number(row, "delta_bic_3_vs_2") for row in ut]
    selected = selection.get("selected_methods", [])
    hysteresis = selection.get("hysteresis_candidates", [])
    answers = [
        f"1. Multi-Otsu 第二阈值均值为 {_format(float(np.mean(finite_high)) if finite_high else float('nan'))}，第一阈值均值为 {_format(float(np.mean(finite_low)) if finite_low else float('nan'))}；是否明显更高由两者差值及图像分布判定。",
        f"2. Multi-Otsu Core 原图预测面积为 {_format(area)}；{'落入' if math.isfinite(area) and 0.10 <= area <= 0.15 else '未落入'}约 10%–15%。",
        "3. 三状态是否解释 V2 膨胀：若 Core 保留高 Precision 且 C1 面积较大、C2 面积合理，则支持；否则不支持。",
        f"4. OTGC 合法收敛 {len(otgc_valid)}/{len(otgc)} 张。",
        f"5. OTGC 前景后验单调通过 {len(otgc_monotone)}/{len(otgc)} 张。",
        "6. 共享方差是否消除双 LGMC 退化：结合合法率、阈值分布与 Precision 判断，不因公式设计而预设成功。",
        "7. BC 超类锚定效果：见 otgc_parameter_distribution.csv 的 bc_selected_as_foreground_ratio。",
        f"8. OTGC 三分量 ΔBIC(3-2)>10 的图像为 {sum(value > 10 for value in otgc_bic if math.isfinite(value))}/{sum(math.isfinite(value) for value in otgc_bic)}。",
        f"9. UT-3CP 有效记录 {len(ut)} 张；三段 ΔBIC>10 为 {sum(value > 10 for value in ut_bic if math.isfinite(value))}/{sum(math.isfinite(value) for value in ut_bic)}。",
        "10. 第一个变化点是否对应高置信尾部：见 k1、高段面积和逐图可视化。",
        f"11. Test20 直接通过的 Core：{selected if selected else '无'}。",
        f"12. Hysteresis 触发候选：{hysteresis if hysteresis else '无'}。",
    ]
    return answers, {"macro": macro, "selection": selection, "fixed": fixed}


def analyze(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = {}
    failures = {}
    for directory, name in STAGES:
        stage_dir = root / directory
        audit = _read_json(stage_dir / "numerical_failure_summary.json")
        if audit:
            completed[name] = audit
            failures[name] = {
                "num_requested": audit.get("num_requested"),
                "num_valid": audit.get("num_valid"),
                "num_failed": audit.get("num_failed"),
            }

    answers, core = _core_answers(root) if (root / "eval_test20").is_dir() else ([], {})
    hsel = _read_json(root / "eval_test20_hysteresis/method_selection.json", {}) or {}
    psel = _read_json(root / "eval_pilot200/method_selection.json", {}) or {}
    final = _read_json(root / "eval_pilot200/final_config.json", {}) or {}
    full_macro = _macro(root / "eval_full6473")
    final_method = (final.get("selected_methods") or [None])[0]
    fixed_full = full_macro.get("fixed_058", {})
    adaptive_full = full_macro.get(final_method, {}) if final_method else {}
    train_admitted = bool(
        adaptive_full
        and _number(adaptive_full, "F_beta_w") >= _number(fixed_full, "F_beta_w") - 0.003
        and _number(adaptive_full, "MAE") - _number(fixed_full, "MAE") <= 0.002
        and _number(adaptive_full, "Precision") >= _number(fixed_full, "Precision") - 0.01
    )
    answers.extend(
        (
            f"13. Hysteresis 是否保持 Precision 并恢复 Recall：{hsel.get('status', '未运行/未触发')}。",
            f"14. 进入 Pilot200 的方法：{hsel.get('selected_methods') or core.get('selection', {}).get('selected_methods') or '无/待定'}。",
            f"15. Pilot200 结果：{psel.get('status', '未运行')}；最终候选 {psel.get('selected_methods', [])}。",
            "16. 所有 V3 自适应公式均不依赖 GT、R1、0.58 或固定目标面积；0.58 仅在评估器中作为外部参考。",
            f"17. COD10K 与 NC4K 稳定性：{'已在 Pilot/全量门禁核验' if psel else '待 Pilot200'}。",
            f"18. 是否进入全量 6473：{'是' if 'full6473' in completed else '否/尚未'}。",
            f"19. 是否进入 1×1 训练：{'是' if train_admitted else '否/尚未满足准入'}。",
            f"20. 最终论文自适应方法：{final_method or '尚未选出；不得提前指定'}。",
        )
    )

    latest = next(
        (root / directory for directory, _ in reversed(STAGES) if (root / directory / "per_image_metrics.csv").is_file()),
        None,
    )
    if latest:
        _copy_if_present(latest / "per_dataset_metrics.csv", root / "per_dataset_metrics.csv")
        _copy_if_present(latest / "per_image_metrics.csv", root / "per_image_metrics.csv")
        _copy_if_present(latest / "threshold_distribution.csv", root / "threshold_distribution.csv")
        _copy_if_present(latest / "three_state_distribution.csv", root / "three_state_distribution.csv")
        _copy_if_present(latest / "otgc_parameter_distribution.csv", root / "otgc_parameter_distribution.csv")
        _copy_if_present(latest / "ut3cp_change_point_distribution.csv", root / "ut3cp_change_point_distribution.csv")
        _copy_if_present(latest / "bootstrap_ci95.csv", root / "bootstrap_ci95.csv")
        _copy_if_present(latest / "downstream_1x1_results.csv", root / "downstream_1x1_results.csv")
    stage_files = {
        "baseline_test20/baseline_test20.csv": "baseline_test20.csv",
        "eval_test20/test20_core_summary.csv": "test20_core_summary.csv",
        "eval_test20_hysteresis/test20_hysteresis_summary.csv": "test20_hysteresis_summary.csv",
        "eval_pilot200/pilot200_summary.csv": "pilot200_summary.csv",
        "eval_full6473/full6473_summary.csv": "full6473_summary.csv",
        "eval_test20/selected_config.json": "selected_config.json",
        "eval_pilot200/final_config.json": "final_config.json",
    }
    for source, target in stage_files.items():
        _copy_if_present(root / source, root / target)
    plot_names = (
        "core_method_comparison_test20.png",
        "hysteresis_comparison_test20.png",
        "pilot200_method_comparison.png",
        "precision_recall_area_comparison.png",
        "threshold_distribution.png",
        "three_state_area_distribution.png",
        "otgc_component_examples.png",
        "otgc_failure_examples.png",
        "ut3cp_examples.png",
        "ut3cp_failure_examples.png",
        "per_image_fbw_delta.png",
        "dataset_wise_comparison.png",
        "success_visualizations.png",
        "failure_visualizations.png",
    )
    for name in plot_names:
        for directory, _ in reversed(STAGES):
            source = root / directory / name
            if source.is_file():
                _copy_if_present(source, root / name)
                break
    (root / "numerical_failure_summary.json").write_text(
        json.dumps({"completed_stages": completed, "stage_failures": failures}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    macro_rows = list(core.get("macro", {}).values())
    lines = [
        "# GBSP Adaptive Threshold V3 Final Report",
        "",
        f"- 已完成阶段：{list(completed)}",
        "- 结论严格受阶段门禁约束；未运行阶段保持‘待定’，不以 Test20 外推。",
        "- 自适应公式未使用 GT、R1、0.58、固定面积或固定 Top-K。",
        "",
        "## Test20 Core 指标",
        "",
        _table(macro_rows) if macro_rows else "尚未运行 A1。",
        "",
        "## 任务书问题逐项回答",
        "",
        *[f"- {answer}" for answer in answers],
        "",
        "## 阶段结论",
        "",
        f"- A1：{core.get('selection', {}).get('status', '未运行')}。",
        f"- A2：{hsel.get('status', '未运行/未触发')}。",
        f"- Pilot200：{psel.get('status', '未运行')}。",
        f"- 全量最终方法：{final_method or '无/待定'}。",
        f"- 下游训练准入：{'通过' if train_admitted else '未通过/待核验'}。",
    ]
    (root / "GBSP_THRESHOLD_V3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "completed_stages": list(completed), "final_method": final_method, "training_admitted": train_admitted}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="../workdir/gbsp_threshold_v3")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    analyze(Path(args.root))
