# EGSA-R1 训练协议

## 数据前提

训练继续读取原 DABE-PU v1.1 与 normal DINO feature。utility evaluator 额外要求
与水平翻转 RGB 匹配的 hflip DINO feature：

```bash
python common/cache_features_hflip.py \
  --config configs/dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_long45_lrfloor_2e5.py \
  --split train
```

训练启动会严格检查 manifest 覆盖、dataset/stem、tensor shape 和 payload。
缺失时直接失败，不退化为仅翻转 RGB 或 normal feature。

DABE-PU preflight 会逐项审计 `weight_map_68` 的 min/max/mean/nonzero ratio 和
per-image max。segmentation 始终使用原始 weight；Router 输入模式由显式配置
控制。

## 单 batch 顺序

1. fetch ECST 与 route 旧状态；
2. student 与 EMA teacher forward；
3. 构造 ECST map、Router input 和 detached segmentation gate；
4. utility evaluator 对 normal/hflip 做冻结 forward；
5. 用旧 route state 构造 delayed utility target；
6. segmentation loss 与 Router loss 分别 backward；
7. student optimizer 与 Router optimizer 分别 step；
8. 更新 EMA teacher 与慢速 utility evaluator；
9. 最后更新 ECST/route memory。

`sample_index % 10 == 0` 的样本仍训练 student，但不进入 Router optimizer，
只用于 preference 诊断。

## 正式训练

完整 hflip cache 由用户生成并确认后再执行：

```bash
python train.py \
  --config configs/dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_long45_lrfloor_2e5.py
```

EGSA 从 epoch1 新训练，不允许用缺少动态状态的旧基线 active-stage checkpoint
继续训练。

## 测试入口

以下入口仅供用户在 hflip cache 完整后手动执行，Codex 本次不启动：

```bash
# Router/memory/loss/gradient/checkpoint 单元契约
pytest -q \
  tests/test_source_arbiter.py \
  tests/test_source_arbiter_memory.py \
  tests/test_source_arbiter_loss_equivalence.py \
  tests/test_source_arbiter_gradient_isolation.py \
  tests/test_source_arbiter_checkpoint.py

# 单 batch 集成 smoke
python train.py \
  --config configs/dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_long45_lrfloor_2e5.py \
  --max_samples 16 \
  --max_epochs 1 \
  --work_dir /home/dell01/CTH/analysis/egsa_r1_smoke/single_batch

# 两 epoch 集成 smoke
python train.py \
  --config configs/dinov1_s8_dabepu_v11_ecst_egsa_r1_dagp_uncgate_ndr_long45_lrfloor_2e5.py \
  --max_samples 32 \
  --max_epochs 2 \
  --work_dir /home/dell01/CTH/analysis/egsa_r1_smoke/two_epoch
```

active-stage resume 必须使用由 EGSA 配置生成、且包含完整动态状态的 epoch1–20
checkpoint。旧 ECST/TEPR checkpoint 缺少 Router、utility evaluator、两套 memory、
RNG 和 DataLoader generator 状态，会按设计直接失败。
