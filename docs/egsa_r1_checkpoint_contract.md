# EGSA-R1 Checkpoint 契约

基础字段保持仓库现状：`student`、`teacher`、`optimizer`、`scheduler`、epoch、
best 状态、config 和 backbone。

EGSA 开启时追加：

```text
source_arbiter
source_arbiter_optimizer
utility_evaluator
route_trajectory_memory
ecst_temporal_memory
global_step
rng_state (Python/NumPy/Torch/CUDA)
train_loader_generator_state
checkpoint_phase
source_arbiter_lifecycle
```

## Phase

- `active_pre_reset`：epoch1-19，必须包含两套 memory。
- `pending_after_epoch_reset`：epoch20 checkpoint 在 after-epoch reset 之前保存；
  resume 到 epoch21 时训练程序必须补执行 reset，然后清空两套 memory。
- `post_reset_inactive`：epoch21 以后，memory 可为 `None`，Router/evaluator
  state 仍保留用于审计。

active-stage resume 缺少任一 Router、optimizer、evaluator、memory、RNG 或
DataLoader generator 状态时直接失败。`USE_SOURCE_ARBITER=False` 时 checkpoint
schema 和旧 resume 行为不变。

eval 只 strict load 原 `student`，不会构造或加载 Router/evaluator/memory。
