# EGSA-R1 设计

EGSA-R1 是 ECST Long45 主线上的训练期监督源仲裁器。student、EMA teacher、
DAGP-Safe、NDR-v1 和推理路径均不改变。

## 输入与输出

`SourceArbiter` 接收 18 通道证据：DABE target/weight、五类互斥区域、
teacher probability/confidence、ECST temporal mean/variance/history、
双源分歧、DINO margin、student probability/uncertainty 和 teacher prior。
所有输入均 detach。网络为三层小卷积和 zero-init 1x1 head，参数少于 25K，
输出 `1.5*tanh(raw)`。

teacher gate 为：

```text
sigmoid(logit(global_teacher_prior)
        + influence_scale * source_disagreement * router_residual)
```

极端 prior 0/1 显式 bypass，DABE gate 恒为 `1 - teacher_gate`。zero-init 时
gate 严格等于现有 global schedule。

## Segmentation loss

DABE source 使用原始 `target_soft_68 + weight_map_68`；teacher source 使用
binary EMA target 和现有 ECST map。两类 pixel loss 分别按自己的 weight sum
归一化，再由同一份 detached gate 仲裁。final/coarse/base 保持原权重。

segmentation backward 不更新 Router。Router 只由延迟跨视图 utility target
及 prior、mass、edge-aware smoothness 正则更新。

## 生命周期

- epoch1-6：只写 ECST/route memory，更新 utility evaluator。
- epoch7-15：Router influence 从 1/9 线性升至 1。
- epoch16-20：完整仲裁。
- epoch21-45：释放两套 memory，停止 Router/evaluator，恢复原 teacher-only。

`USE_SOURCE_ARBITER=False` 时不构造任何 EGSA 对象，也不改变 ECST 路径。
