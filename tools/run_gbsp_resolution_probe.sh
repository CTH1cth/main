#!/usr/bin/env bash
set -euo pipefail

source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth

main_root=/home/dell01/CTH/MY-baseline/main
sample_count=${1:-200}
if [[ "$sample_count" == "full" ]]; then
  selector_count=-1
  dataset_args=(CHAMELEON TE-CAMO TE-COD10K NC4K)
  default_label=full6473
else
  selector_count="$sample_count"
  dataset_args=(TE-CAMO TE-COD10K NC4K)
  default_label="probe${sample_count}"
fi
output_root=${2:-/home/dell01/CTH/MY-baseline/workdir/gbsp_resolution_512/${default_label}}
cd "$main_root"

mkdir -p "$output_root"
sample_list="$output_root/sample_list_source.jsonl"
feature_cache_root="$output_root/feature_cache"

python tools/select_gbsp_resolution_probe.py \
  --config configs/dinov1_s8_gbsp_resolution_296_reference.py \
  --output "$sample_list" \
  --num-samples "$selector_count" \
  --seed 42 \
  --datasets "${dataset_args[@]}"

manifest296="$feature_cache_root/features_cache/dinov1-s8-296-reference/manifest_test.jsonl"
if [[ ! -f "$manifest296" ]]; then
  python common/cache_features.py \
    --config configs/dinov1_s8_gbsp_resolution_296_reference.py \
    --split test \
    --cache_root "$feature_cache_root" \
    --sample_list "$sample_list"
fi

manifest512="$feature_cache_root/features_cache/dinov1-s8-512-native64/manifest_test.jsonl"
if [[ ! -f "$manifest512" ]]; then
  python common/cache_features.py \
    --config configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py \
    --split test \
    --cache_root "$feature_cache_root" \
    --sample_list "$sample_list"
fi

python tools/eval_gbsp_resolution_probe.py \
  --config-296 configs/dinov1_s8_gbsp_resolution_296_reference.py \
  --config-512a configs/dinov1_s8_gbsp_resolution_512_native64_bw2.py \
  --config-512b configs/dinov1_s8_gbsp_resolution_512_native64_bw3.py \
  --config-512c configs/dinov1_s8_gbsp_resolution_512_native64_bw2_r12.py \
  --manifest-296 "$manifest296" \
  --manifest-512 "$manifest512" \
  --sample-list "$sample_list" \
  --out-dir "$output_root/results" \
  --visualizations 20
