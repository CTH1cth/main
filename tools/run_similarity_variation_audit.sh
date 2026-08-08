#!/usr/bin/env bash
# Run the complete frozen Similarity--Variation audit after Full-BC/MBSP exists.
# This script is intentionally a child shell: a failure returns to the caller's
# terminal instead of closing the interactive terminal.
set -euo pipefail

MAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAIN_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "cth" ]]; then
  echo "错误：请先 conda activate cth" >&2
  exit 2
fi

echo "运行服务器：$(hostname)"
echo "Python：$(command -v python)"

for required in \
  tools/cache_gbsp_core_variants.py \
  tools/eval_similarity_vs_subspace.py \
  tools/analyze_similarity_conditioned_separation.py \
  tools/plot_similarity_variation.py \
  configs/gbsp_similarity_test20_ids.txt; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 2; }
done

python tools/cache_gbsp_core_variants.py \
  --config configs/dinov1_s8_similarity_portable.py \
  --gbsp-root ../workdir/mbsp_pca_v1/ablations/M1_full6473 \
  --out-root ../workdir/gbsp_core_optimization/full_rank \
  --split test \
  --max-samples -1 \
  --experiment rank \
  --rank-variants r0 r8 current \
  --background-source fullbc \
  --pca-weight equal \
  --workers 2 \
  --torch-threads 8 \
  --save-diagnostics

python tools/eval_similarity_vs_subspace.py \
  --gbsp_root ../workdir/gbsp_core_optimization/full_rank \
  --sample_list configs/gbsp_similarity_test20_ids.txt \
  --methods mean_l2 proto_cos nn_cos knn8_cos gbsp \
  --exclude_self_match \
  --save_scores \
  --workers 4 \
  --out_dir ../workdir/gbsp_similarity_variation/test20

python tools/analyze_similarity_conditioned_separation.py \
  --score_root ../workdir/gbsp_similarity_variation/test20 \
  --sample_list configs/gbsp_similarity_test20_ids.txt \
  --high_similarity_quantiles 0.80 0.70 0.60 \
  --similarity_bins 5 \
  --matched_similarity_tolerance 0.01 \
  --save_diagnostics \
  --out_dir ../workdir/gbsp_similarity_variation/test20_analysis

python - <<'PY'
import json
from pathlib import Path

root = Path("../workdir/gbsp_similarity_variation")
validity = json.loads((root / "test20/validity_summary.json").read_text())
self_match = json.loads((root / "test20/self_match_audit.json").read_text())
analysis = json.loads((root / "test20_analysis/validity_summary.json").read_text())

assert validity["requested"] == 20
assert validity["generated"] == 20
assert validity["generation_failed"] == 0
assert validity["evaluation_failed"] == 0
assert validity["score_nan"] == 0
assert validity["counts"] == {
    "CHAMELEON": 5, "CAMO": 5, "COD10K": 5, "NC4K": 5,
}
assert self_match["self_match_excluded"] is True
assert self_match["self_match_violation"] == 0
assert analysis["images"] == 20
assert analysis["subset_generation_uses_gt"] is False
print("Test20 正确性护栏：PASS")
PY

python tools/eval_similarity_vs_subspace.py \
  --gbsp_root ../workdir/gbsp_core_optimization/full_rank \
  --split test \
  --max_samples -1 \
  --methods mean_l2 proto_cos nn_cos knn8_cos gbsp \
  --exclude_self_match \
  --save_scores \
  --workers 4 \
  --bootstrap_repetitions 2000 \
  --bootstrap_seed 20260807 \
  --out_dir ../workdir/gbsp_similarity_variation/full6473

python tools/analyze_similarity_conditioned_separation.py \
  --score_root ../workdir/gbsp_similarity_variation/full6473 \
  --split test \
  --max_samples -1 \
  --high_similarity_quantiles 0.80 0.70 0.60 \
  --similarity_bins 5 \
  --matched_similarity_tolerance 0.01 \
  --bootstrap_repetitions 2000 \
  --bootstrap_seed 20260807 \
  --save_diagnostics \
  --out_dir ../workdir/gbsp_similarity_variation/full6473_analysis

python tools/plot_similarity_variation.py \
  --score_root ../workdir/gbsp_similarity_variation/full6473 \
  --analysis_root ../workdir/gbsp_similarity_variation/full6473_analysis \
  --deterministic_samples_per_class_per_image 100 \
  --out_dir ../workdir/gbsp_similarity_variation/figures

python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("../workdir/gbsp_similarity_variation")
validity = json.loads((root / "full6473/validity_summary.json").read_text())
self_match = json.loads((root / "full6473/self_match_audit.json").read_text())
analysis = json.loads((root / "full6473_analysis/validity_summary.json").read_text())

assert validity["evaluation_failed"] == 0
assert validity["score_nan"] == 0
assert self_match["self_match_excluded"] is True
assert self_match["self_match_violation"] == 0
assert analysis["subset_generation_uses_gt"] is False
assert analysis["bootstrap_repetitions"] == 2000

missing = 6473 - int(validity["generated"])
if missing > 6:
    raise RuntimeError(f"缺失样本过多：{missing}")
if missing:
    print(f"WARNING：记录并放行 {missing} 个罕见失败样本；不自动重新生成。")
else:
    assert validity["is_full_complete"] is True
    assert validity["counts"] == {
        "CHAMELEON": 76, "CAMO": 250, "COD10K": 2026, "NC4K": 4121,
    }
    with (root / "full6473/baseline_reproduction.csv").open() as handle:
        reproduction = list(csv.DictReader(handle))
    gbsp = next(row for row in reproduction if row["method"] == "gbsp_r8")
    assert float(gbsp["ap_abs_error"]) < 1e-4
    assert float(gbsp["auroc_abs_error"]) < 1e-4

print("正式完整性检查：PASS")
print("最终报告：", root / "full6473_analysis/GBSP_SIMILARITY_VARIATION_REPORT.md")
PY

echo "Similarity--Variation 正式实验全部完成。"
