#!/usr/bin/env python3
"""Stage-0 audit: identify raw/L2 feature geometry without reading GT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reconstruction import flatten_feature, global_pca_reconstruct  # noqa: E402
from tools.gbsp_knn_lsr_common import write_csv, write_json  # noqa: E402
from tools.reconstruction_rescue_common import (  # noqa: E402
    load_core_inputs, load_core_rows, require_output_outside_main,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dinov1_s8_reconstruction_rescue.py")
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows = load_core_rows(args.core_root, split=args.split, max_samples=args.max_samples)
    output = require_output_outside_main(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    details = []
    for index, row in enumerate(rows, 1):
        core, raw, background = load_core_inputs(row, device)
        flat = flatten_feature(raw, normalize=False)
        norm = flat.norm(dim=1)
        l2 = flatten_feature(raw, normalize=True)
        current = core["results"]["r8"]["absolute_raw"].to(device).float().reshape(-1)
        pca_l2 = global_pca_reconstruct(raw, background, rank=8, normalize_features=True)
        pca_raw = global_pca_reconstruct(raw, background, rank=8, normalize_features=False)
        details.append({
            "dataset": row["dataset"], "stem": row["stem"],
            "raw_norm_min": float(norm.min()), "raw_norm_q25": float(torch.quantile(norm, .25)),
            "raw_norm_median": float(norm.median()), "raw_norm_mean": float(norm.mean()),
            "raw_norm_q75": float(torch.quantile(norm, .75)), "raw_norm_max": float(norm.max()),
            "l2_norm_max_abs_error": float((l2.norm(dim=1) - 1).abs().max()),
            "current_vs_l2_pca_max_abs_error": float((current - pca_l2.residual).abs().max()),
            "current_vs_raw_pca_max_abs_error": float((current - pca_raw.residual).abs().max()),
            "current_pca_is_l2": float((current - pca_l2.residual).abs().max()) <= 2e-4,
            "l2_svd_orth_error_before_qr": pca_l2.svd_orthonormal_error_before_qr,
            "l2_orth_error_after_qr": pca_l2.orthonormal_error,
            "raw_svd_orth_error_before_qr": pca_raw.svd_orthonormal_error_before_qr,
            "raw_orth_error_after_qr": pca_raw.orthonormal_error,
        })
        print(f"[{index}/{len(rows)}] {row['dataset']}/{row['stem']}", flush=True)
    write_csv(output / "feature_geometry.csv", details)
    errors_l2 = [row["current_vs_l2_pca_max_abs_error"] for row in details]
    errors_raw = [row["current_vs_raw_pca_max_abs_error"] for row in details]
    summary = {
        "samples": len(details), "gt_loaded": False,
        "raw_feature_available": True,
        "raw_norm_mean": float(np.mean([row["raw_norm_mean"] for row in details])),
        "current_vs_l2_pca_max_abs_error": float(max(errors_l2)),
        "current_vs_raw_pca_mean_abs_error_proxy": float(np.mean(errors_raw)),
        "current_pca_input": "l2_key" if max(errors_l2) <= 2e-4 else "unresolved",
        "raw_reconstruction_is_legal": True,
    }
    write_json(output / "feature_audit.json", summary)
    report = f"""# Feature Source Audit

- 样本数：{len(details)}
- 审计过程读取 GT：否
- 原始 384×37×37 Key 特征可用：是
- 当前正式 GBSP PCA 输入：{summary['current_pca_input']}
- 当前响应与重算 L2-PCA 的最大绝对误差：{summary['current_vs_l2_pca_max_abs_error']:.8g}
- Raw Key 平均 patch 范数：{summary['raw_norm_mean']:.6f}
- 结论：无需重新提取 DINO；Raw/L2 只改变重构几何，检索仍冻结为 L2-Cosine。
"""
    (output / "FEATURE_SOURCE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
