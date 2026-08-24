#!/usr/bin/env bash
set -euo pipefail

source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth

main_root="/home/dell01/CTH/MY-baseline/main"
out_root="${GBSP_RESOLUTION_FULL_ROOT:-/home/dell01/CTH/MY-baseline/workdir/gbsp_resolution_512/full6473}"
sample_list="${out_root}/sample_list_source.jsonl"
feature_cache_root="${out_root}/shared_dino512"
feature_manifest="${feature_cache_root}/features_cache/dinov1-s8-512-native64/manifest_test.jsonl"

cd "${main_root}"
mkdir -p "${out_root}/logs"

python tools/select_gbsp_resolution_probe.py \
  --config configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py \
  --output "${sample_list}" \
  --num-samples -1 \
  --seed 42 \
  --datasets CHAMELEON TE-CAMO TE-COD10K NC4K \
  2>&1 | tee "${out_root}/logs/00_select_full6473.log"

python common/cache_features.py \
  --config configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py \
  --split test \
  --cache_root "${feature_cache_root}" \
  --sample_list "${sample_list}" \
  --max_samples -1 \
  --resume \
  2>&1 | tee "${out_root}/logs/01_shared_dino512.log"

python - "${feature_manifest}" "${sample_list}" "${out_root}/shared_dino512_metadata.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

manifest, sample_list, output = map(lambda value: Path(value).resolve(), sys.argv[1:])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
samples = [json.loads(line) for line in sample_list.read_text(encoding="utf-8").splitlines() if line.strip()]
missing = [row["cache_path"] for row in rows if not Path(row["cache_path"]).is_file()]
counts = {name: sum(row["dataset"] == name for row in rows) for name in ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")}
expected = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
assert len(rows) == len(samples) == 6473, (len(rows), len(samples))
assert counts == expected, counts
assert not missing, f"missing feature files: {len(missing)}"
metadata = {
    "input_size": 512,
    "grid": 64,
    "backbone": "dinov1-s8",
    "feature_dim": 384,
    "num_images": len(rows),
    "counts": counts,
    "manifest": str(manifest),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
PY

python tools/cache_gbsp_resolution_full.py \
  --config-296 configs/dinov1_s8_gbsp_resolution_296_reference.py \
  --config-512a configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py \
  --config-512b configs/dinov1_s8_gbsp_resolution_512_native64_bw3.py \
  --config-512c configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r12.py \
  --feature-manifest "${feature_manifest}" \
  --sample-list "${sample_list}" \
  --out-root "${out_root}" \
  --max-samples -1 \
  --torch-threads 8 \
  --checkpoint-every 20 \
  --failure-policy record \
  2>&1 | tee "${out_root}/logs/02_generate_512_ABC.log"

python tools/eval_gbsp_resolution_full.py \
  --score-root "${out_root}" \
  --sample-list "${sample_list}" \
  --out-dir "${out_root}/comparison" \
  --continuous-296 /home/dell01/CTH/MY-baseline/workdir/gbsp_similarity_variation_a6000/full6473/full_continuous_metrics.csv \
  --per-dataset-continuous-296 /home/dell01/CTH/MY-baseline/workdir/gbsp_similarity_variation_a6000/full6473/per_dataset_metrics.csv \
  --fixed-296 /home/dell01/CTH/MY-baseline/workdir/gbsp_core_optimization/binary_metrics_058.csv \
  --per-image-296 /home/dell01/CTH/MY-baseline/workdir/gbsp_core_optimization/per_image_metrics.csv \
  --core-296-root /home/dell01/CTH/MY-baseline/workdir/gbsp_core_optimization/full_rank \
  --feature-manifest "${feature_manifest}" \
  --max-samples -1 \
  --checkpoint-every 100 \
  2>&1 | tee "${out_root}/logs/03_evaluate_full6473.log"

python - "${out_root}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
generation = json.loads((root / "generation_metadata.json").read_text(encoding="utf-8"))
evaluation = json.loads((root / "comparison/evaluation_metadata.json").read_text(encoding="utf-8"))
assert generation["is_full_complete"] and generation["num_valid"] == 6473 and generation["num_failed"] == 0
assert evaluation["is_full_complete"] and evaluation["num_valid"] == 6473 and evaluation["num_failed"] == 0
required = (
    "continuous_metrics.csv", "fixed_threshold_metrics.csv", "per_dataset_metrics.csv",
    "score_distribution_stats.csv", "pca_rank_full6473.csv", "runtime_full6473.csv",
    "pr_curve_512A.csv", "pr_curve_512B.csv", "pr_curve_512C.csv",
    "PR_512ABC.png", "RESULTS.md",
)
missing = [name for name in required if not (root / "comparison" / name).is_file()]
assert not missing, missing
print("valid = 6473 / 6473")
print("failed = 0")
print(f"RESULTS = {root / 'comparison/RESULTS.md'}")
PY
