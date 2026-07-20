# EGSA-R2 Pure Loss-Space Design

## Scope

EGSA-R2 keeps the EGSA-R1 router, 18-channel evidence, delayed utility
evaluator, schedules, model, optimizer, EMA, reset, and DABE-PU supervision.
The only optimization-path change is the internal teacher source weight:

- R1: detached ECST teacher map.
- R2: detached float32 map of exact ones.

ECST temporal statistics, region masks, and DINO margin remain router evidence.
The fixed ECST map is audit-only and cannot enter final, coarse, or base loss.

## Loss Contract

The final, NDR coarse, and base branches share the same detached complementary
router gate. DABE uses the original soft target and weight map. Teacher uses the
same binary EMA target and an all-one source weight. Existing source
normalization and branch normalization are unchanged.

## Lifecycle

Epochs 1-6 collect temporal and route history. Epochs 7-20 train and apply the
router. The epoch20 after-epoch reset clears both memories. Epochs 21-45 remain
the inherited teacher-only path with no router or temporal-memory activity.

## Inference

The router, evaluator, temporal memories, and ECST map are training-only.
Evaluation loads only the student and uses its final logits with the unchanged
binary threshold and COD metrics.
