# EGSA-R2 Training Protocol

## Shared Protocol

EGSA-R2 inherits the R1 model, DABE-PU supervision, router architecture,
18-channel evidence, delayed utility evaluator, optimizer, EMA, LR, reset, and
45-epoch schedule.

## Source Loss

- DABE source: original soft target and original DABE weight map.
- Teacher source: original binary EMA target and an exact all-one float32
  source weight.
- Final, NDR coarse, and base branches share the same detached complementary
  gate and keep the inherited source and branch normalization.
- ECST fixed weighting is excluded from the segmentation loss.

## Epoch Lifecycle

- Epochs 1-6: collect temporal and route history; router influence is zero.
- Epochs 7-20: train and apply the router with the inherited ramp.
- Epoch20 after-epoch reset: reset the inherited training state and release
  temporal/route memories.
- Epochs 21-45: inherited teacher-only path; router, evaluator, and memories
  remain inactive.

## Gradient Contract

Segmentation backward cannot update the router. Router backward cannot update
student, teacher, or utility evaluator. Audit and R1 shadow calculations are
strictly no-grad.

## Runtime Artifacts

Runtime outputs remain under `../workdir/<EXP_NAME>/train/`. R2 adds
`egsa_r2_shadow.csv`, `egsa_r2_utility.csv`, and extended gate visuals without
GT.

## Commands

Run the complete R2 experiment from epoch1:

```bash
python train.py \
  --config configs/dinov1_s8_dabepu_v11_egsa_r2_pure_dagp_uncgate_ndr_long45_lrfloor_2e5.py
```

Use the single supported cross-mode fork from the canonical R1 epoch6 state:

```bash
python train.py \
  --config configs/dinov1_s8_dabepu_v11_egsa_r2_pure_dagp_uncgate_ndr_long45_lrfloor_2e5.py \
  --resume ../workdir/13-egsa_r1-long45/train/ckpt/epoch_006.pth
```

Adding `--max_epochs 8` to the fork command gives a two-epoch contract check
without changing the checkpoint memory indexing.
