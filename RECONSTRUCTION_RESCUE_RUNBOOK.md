# Reconstruction Rescue 执行手册

本轮目标不是在测试集上追求一个“更聪明的阈值”，而是在冻结 Full-BC 背景记忆和 L2-Cosine 检索后，验证更强的查询条件重构算子能否在连续 AP/AUROC 上超过 KNN8。所有输出均写入 `../workdir/reconstruction_rescue_20260807`，不重新提取 DINO，也不改变原 GBSP 缓存。

## 0. 规则

- 分数生成不读取 GT；GT 只在评测或误差诊断阶段读取。
- 邻居检索固定为 L2-Key Cosine，Full-BC，严格 leave-one-out。
- 主连续指标固定为：每库逐图 native AP/AUROC 的平均，再对四库等权平均。
- `global_pca_l2` 必须复现旧 GBSP-r8；KNN8 必须由同一新缓存内部重算。
- LAR 只有在 LCBR 与 LSR 均距 KNN8 不超过 `0.002 AP` 时才允许运行。
- 局部尺度归一化和阈值扫描只对连续指标合格的候选运行。

## 1. Stage-0 与求解器验证（小样本）

```bash
source /home/dell01/anaconda3/etc/profile.d/conda.sh
conda activate cth
cd /home/dell01/CTH/MY-baseline/main

python tools/audit_reconstruction_features.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --split test --max_samples 20 \
  --out_dir ../workdir/reconstruction_rescue_20260807/stage0_feature_audit

python tools/validate_reconstruction_solver.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --num_synthetic 100 --num_real 100 --k 16 --geometry l2 \
  --out_dir ../workdir/reconstruction_rescue_20260807/solver_validation_l2

python tools/validate_reconstruction_solver.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --num_synthetic 100 --num_real 100 --k 16 --geometry raw \
  --out_dir ../workdir/reconstruction_rescue_20260807/solver_validation_raw
```

## 2. 20 张代码烟测

```bash
python tools/cache_reconstruction_scores.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --out_root ../workdir/reconstruction_rescue_20260807/test20_scores \
  --split test --max_samples 20 --device cuda --query_batch_size 128 \
  --save_diagnostics --failure_policy strict

python tools/eval_reconstruction_continuous.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --score_root ../workdir/reconstruction_rescue_20260807/test20_scores \
  --split test --max_samples 20 --workers 4 --bootstrap_repetitions 100 \
  --out_dir ../workdir/reconstruction_rescue_20260807/test20_continuous
```

## 3. 正式连续分数生成与评测

以下命令是全量生成/评测，应由用户明确启动：

```bash
python tools/cache_reconstruction_scores.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --out_root ../workdir/reconstruction_rescue_20260807/full_scores \
  --split test --max_samples -1 --device cuda --query_batch_size 128 \
  --save_diagnostics --failure_policy record

python tools/eval_reconstruction_continuous.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --score_root ../workdir/reconstruction_rescue_20260807/full_scores \
  --split test --max_samples -1 --workers 8 \
  --out_dir ../workdir/reconstruction_rescue_20260807/full_continuous
```

## 4. 条件阶段

连续结果先看 `full_continuous/CONTINUOUS_REPORT.md` 与 `paired_bootstrap_vs_knn8.csv`。只有合格方法才进入尺度诊断：

若主 LSR 已经接近 KNN8，才补 K8/R2 与 K32/R8；若 LCBR 与 LSR 都满足任务书的 `0.002 AP` 条件，才单独补 LAR：

```bash
python tools/cache_reconstruction_scores.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --out_root ../workdir/reconstruction_rescue_20260807/conditional_lsr_scores \
  --split test --variants lsr_l2_k8_r2 lsr_l2_k32_r8 \
  --device cuda --query_batch_size 128 --failure_policy record

python tools/cache_reconstruction_scores.py \
  --config configs/dinov1_s8_reconstruction_rescue.py \
  --core_root ../workdir/gbsp_core_optimization/full_rank \
  --out_root ../workdir/reconstruction_rescue_20260807/conditional_lar_scores \
  --split test --variants lar_l2_k16 \
  --device cuda --query_batch_size 128 --failure_policy record
```

这两段不是默认流水线，条件不成立时不得运行。

```bash
python tools/analyze_reconstruction_residuals.py \
  --score_root ../workdir/reconstruction_rescue_20260807/full_scores \
  --split test --methods lcbr_l2_k16 lsr_l2_k16_r4 \
  --normalizations raw centroid pairwise --device cuda \
  --out_root ../workdir/reconstruction_rescue_20260807/full_scale_diagnostics
```

再对明确的 finalist 做阈值化，不能把全部变体一起扫：

```bash
python tools/eval_reconstruction_thresholdability.py \
  --score_root ../workdir/reconstruction_rescue_20260807/full_scale_diagnostics \
  --split test --methods lsr_l2_k16_r4 lsr_l2_k16_r4__centroid_scale \
  --out_dir ../workdir/reconstruction_rescue_20260807/finalist_thresholdability
```

训练集伪标签生成必须换成训练 split 的 score cache；脚本本身不读取 GT：

```bash
python tools/build_reconstruction_pseudo.py \
  --score_root ../workdir/reconstruction_rescue_20260807/train_scores \
  --split train --method lsr_l2_k16_r4 --threshold 0.58 \
  --out_root ../workdir/reconstruction_rescue_20260807/train_pseudo_lsr_k16_r4
```

本手册不启动 1×1 或 DAGP-NDR 训练。进入下游时最多保留 KNN8、Global GBSP 与一项最终候选三组，避免把测试集诊断变成解码器大规模选型。
