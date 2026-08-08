#!/usr/bin/env bash
set -euo pipefail

source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth

gpu_id="${1:-0}"
if [[ "${gpu_id}" != "0" && "${gpu_id}" != "1" ]]; then
  echo "gpu_id must be 0 or 1, got: ${gpu_id}" >&2
  exit 2
fi

protocol_root="/home/dell01/CTH/MY-baseline/datasets/cache/sinetv2_r1_v1s_train"
train_image_root="${protocol_root}/TrainDataset/Imgs/"
train_mask_root="${protocol_root}/TrainDataset/PseudoMask/"
val_root="${protocol_root}/TestDataset/CAMO/"
rise_sinet_root="/home/dell01/CTH/0base/RISE-master/SINet-V2"
run_root="/home/dell01/CTH/MY-baseline/workdir/sinetv2_r1_v1s"
torch_home="/home/dell01/CTH/MY-baseline/workdir/torch_cache"

protocol_file="${protocol_root}/protocol.json"
if [[ ! -f "${protocol_file}" ]]; then
  echo "Missing audited R1 protocol: ${protocol_file}" >&2
  exit 3
fi

image_count="$(find "${train_image_root}" -maxdepth 1 -type l | wc -l)"
mask_count="$(find "${train_mask_root}" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "${image_count}" != "4040" || "${mask_count}" != "4040" ]]; then
  echo "R1 export is incomplete: images=${image_count}, masks=${mask_count}" >&2
  exit 4
fi

python /home/dell01/CTH/MY-baseline/main/tools/export_r1_sinetv2_masks.py \
  --audit-only

mkdir -p "${run_root}" "${torch_home}"
cp "${protocol_file}" "${run_root}/r1_protocol.json"
sha256sum \
  "${rise_sinet_root}/MyTrain_Val.py" \
  "${rise_sinet_root}/MyTesting.py" \
  "${rise_sinet_root}/lib/Network_Res2Net_GRA_NCD.py" \
  "${rise_sinet_root}/utils/data_val.py" \
  > "${run_root}/official_sinetv2_source.sha256"
export TORCH_HOME="${torch_home}"
export PYTHONUNBUFFERED=1

cd "${rise_sinet_root}"
python MyTrain_Val.py \
  --gpu_id "${gpu_id}" \
  --epoch 100 \
  --lr 1e-4 \
  --batchsize 36 \
  --trainsize 352 \
  --clip 0.5 \
  --decay_rate 0.1 \
  --decay_epoch 50 \
  --img_root "${train_image_root}" \
  --gt_root "${train_mask_root}" \
  --val_root "${val_root}" \
  --save_path "${run_root}/"
