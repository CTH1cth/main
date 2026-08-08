# GBSP Similarity Audit：远端从零生成

本文档不复用任何旧缓存。它会依次生成 DINO feature、Full-BC-only、单一全局PCA输入、GBSP r0/r8 core，再运行相似度审计。Full-BC-only只复现DABE-v2最前面的背景连通性与锚点，不生成完整DABE-v2。

## 目录要求

默认把仓库放在任意位置的 `MY-baseline/`，COD数据为：

```text
MY-baseline/datasets/COD/
├── CHAMELEON/{im,gt}
├── TE-CAMO/{im,gt}
├── TE-COD10K/{im,gt}
└── NC4K/{im,gt}
```

DINOv1-S/8本地HuggingFace权重为：

```text
MY-baseline/weights/huggingface/facebook-dino-vits8/
```

若实际位置不同，在运行前设置：

```bash
export COD_DATA_ROOT=/actual/path/to/COD
export DINO_V1_S8_MODEL_PATH=/actual/path/to/facebook-dino-vits8
```

可选地用 `CTH_CACHE_ROOT` 和 `CTH_WORK_ROOT` 改变缓存/结果根目录。

## 环境及静态检查

```bash
conda activate cth
cd /path/to/MY-baseline/main

python - <<'PY'
from common.utils import load_config
cfg=load_config('configs/dinov1_s8_similarity_portable.py')
from pathlib import Path
print('DATA_ROOT =', cfg.DATA_ROOT, Path(cfg.DATA_ROOT).is_dir())
print('CACHE_ROOT =', cfg.CACHE_ROOT)
print('WORK_ROOT =', cfg.WORK_ROOT)
print('DINO =', cfg.DINO['model_path'], Path(cfg.DINO['model_path']).is_dir())
PY

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_similarity_variation_analysis.py
```

## 1. 生成正式6473张 DINOv1-S/8 identity feature

```bash
python common/cache_features.py \
  --config configs/dinov1_s8_similarity_portable.py \
  --split test
```

## 2. 只生成正式Full BC

```bash
python common/cache_gbsp_fullbc_identity.py \
  --config configs/dinov1_s8_similarity_portable.py \
  --out_root ../datasets/cache/gbsp_fullbc_identity/dinov1-s8 \
  --split test \
  --max_samples -1 \
  --workers 4 \
  --torch_threads 1
```

## 3. 生成单一全局PCA源缓存

```bash
python tools/cache_mbsp_pseudo.py \
  --config configs/dinov1_s8_similarity_portable.py \
  --feature_manifest ../datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl \
  --dabe_manifest ../datasets/cache/gbsp_fullbc_identity/dinov1-s8/manifest_test.jsonl \
  --out_root ../workdir/mbsp_pca_v1/ablations/M1_full6473 \
  --split test \
  --max_samples -1 \
  --num_subspaces 1 \
  --pca_energy 0.90 \
  --pca_max_rank 8 \
  --pca_min_rank 1 \
  --workers 4 \
  --torch_threads 1 \
  --save_raw \
  --save_minmax
```

## 4. 生成相似度审计实际需要的GBSP r0/r8 core

```bash
python tools/cache_gbsp_core_variants.py \
  --config configs/dinov1_s8_similarity_portable.py \
  --gbsp_root ../workdir/mbsp_pca_v1/ablations/M1_full6473 \
  --out_root ../workdir/gbsp_core_optimization/full_rank \
  --split test \
  --max_samples -1 \
  --experiment rank \
  --rank_variants r0 r8 current \
  --background_source fullbc \
  --pca_weight equal \
  --workers 2 \
  --torch_threads 8 \
  --save_diagnostics
```

## 5. 先跑Test20评分与条件分析

```bash
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
```

Test20必须为四数据集各5张、总计20，且generation_failed、score_nan和self_match_violation均为0。确认后再执行全量评分、条件分析和绘图命令。
