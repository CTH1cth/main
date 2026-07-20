# EGSA-R2 Checkpoint Contract

## R2 Metadata

An R2 checkpoint records:

- `source_arbiter_mode="pure_loss_space"`
- `teacher_source_weight_mode="ones"`
- `ecst_weighting_used_for_training=False`
- `source_temporal_memory`
- router, router optimizer, utility evaluator, route memory, RNG states,
  DataLoader generator state, global step, phase, and lifecycle metadata

R2 does not duplicate the temporal memory under `ecst_temporal_memory`.
R1 retains that original field.

## Resume Rules

Same-mode resume requires matching mode and lifecycle metadata. Active-stage
resume cannot recreate a missing memory or RNG state.

The only cross-mode path is the canonical, complete R1 `epoch_006.pth` with
`checkpoint_phase="active_pre_reset"`. R1 epoch7 or later cannot be loaded into
R2. A valid fork logs `[EGSA-R2 Fork]` before epoch7.

## Inference

Evaluation loads only `checkpoint["student"]`. Router, utility evaluator,
temporal memories, route memory, and ECST audit map are not loaded or used.
