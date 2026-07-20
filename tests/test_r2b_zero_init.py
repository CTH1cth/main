import torch
import torch.nn.functional as F

from common.source_arbiter import (
    SignAwareSourceArbiter,
    apply_sign_aware_loss_space_arbitration,
    compute_sign_aware_source_gates,
)


def test_zero_init_matches_pure_prior_raw_teacher_loss():
    torch.manual_seed(3407)
    router = SignAwareSourceArbiter().double()
    shape = (2, 1, 5, 5)
    teacher_prob = torch.rand(shape, dtype=torch.float64)
    teacher_target = (teacher_prob >= 0.5).double()
    dabe_target = torch.rand(shape, dtype=torch.float64)
    dabe_weight = torch.rand(shape, dtype=torch.float64).clamp_min(0.1)
    output = router(torch.randn(2, 18, 5, 5, dtype=torch.float64))
    gates = compute_sign_aware_source_gates(
        output,
        teacher_target,
        teacher_prior=0.6,
        teacher_prob=teacher_prob,
        dabe_soft_target=dabe_target,
        influence_scale=1.0,
    )
    logits = torch.randn(shape, dtype=torch.float64)
    actual = apply_sign_aware_loss_space_arbitration(
        logits=logits,
        dabe_target=dabe_target,
        dabe_weight=dabe_weight,
        teacher_target=teacher_target,
        gate_dabe=gates["gate_dabe"],
        gate_teacher=gates["gate_teacher"],
        source_sum=1.0,
        eps=1e-12,
    )["loss"]
    dabe_raw = F.binary_cross_entropy_with_logits(
        logits, dabe_target, reduction="none"
    )
    dabe_loss = (dabe_raw * dabe_weight).sum() / dabe_weight.sum()
    teacher_loss = F.binary_cross_entropy_with_logits(logits, teacher_target)
    expected = 0.4 * dabe_loss + 0.6 * teacher_loss
    assert torch.max(torch.abs(gates["gate_teacher"] - 0.6)).item() <= 1e-12
    assert abs(float(actual - expected)) <= 1e-11
