# EGSA-R2 Ablation And Promotion Decision

## Controlled Variable

R1 and R2 differ in one optimization-path variable:

- R1 teacher source weight: ECST map.
- R2 teacher source weight: exact ones.

All other inherited training and inference settings remain fixed. The R1 shadow
uses the same R2 gate and differs only in teacher source weight.

## Required Evidence

Before promoting R2, confirm:

1. Teacher source min/mean/max remains `1/1/1`.
2. Router parameter count remains `19,265`.
3. Signed correction is non-degenerate in epochs 7-20.
4. Utility teacher/DABE/tie balance is not source-collapsed.
5. Shadow deltas and region deltas are explainable.
6. Evaluation logs confirm student-only inference.
7. R2 improves agreed metrics over ECST clean and R1 without red-line
   regressions.

## Current Decision

Not evaluated. No training, smoke run, or formal evaluation was launched during
implementation, so R2 must not yet be promoted.
