from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.source_arbiter import (
    SourceArbiter,
    apply_loss_space_arbitration,
    build_teacher_source_weight,
    compute_source_gates,
)


def test_pure_teacher_source_is_exact_float32_ones_and_matches_plain_bce():
    torch.manual_seed(3407)
    logits = torch.randn(2, 1, 7, 9, dtype=torch.float64)
    target = (torch.rand_like(logits) >= 0.5).double()
    probability = torch.rand_like(logits, dtype=torch.float32)
    weight = build_teacher_source_weight("pure_loss_space", probability)
    assert weight.dtype == torch.float32
    assert not weight.requires_grad
    assert torch.equal(weight, torch.ones_like(weight))

    result = apply_loss_space_arbitration(
        logits=logits,
        dabe_target=target,
        dabe_weight=torch.ones_like(logits),
        teacher_target=target,
        teacher_weight=weight.double(),
        gate_dabe=torch.zeros_like(logits),
        gate_teacher=torch.ones_like(logits),
        source_sum=1.0,
        eps=1e-6,
    )
    plain = F.binary_cross_entropy_with_logits(logits, target)
    assert abs(result["loss"].item() - plain.item()) <= 1e-7


def test_r2_main_path_does_not_require_an_audit_map():
    probability = torch.rand(2, 1, 5, 5)
    weight = build_teacher_source_weight(
        "pure_loss_space",
        probability,
        ecst_weight_map=None,
    )
    assert torch.equal(weight, torch.ones_like(weight))
    logits = torch.randn(2, 1, 5, 5, requires_grad=True)
    result = apply_loss_space_arbitration(
        logits=logits,
        dabe_target=torch.rand_like(logits),
        dabe_weight=torch.rand_like(logits),
        teacher_target=(probability >= 0.5).float(),
        teacher_weight=weight,
        gate_dabe=torch.full_like(logits, 0.4),
        gate_teacher=torch.full_like(logits, 0.6),
        source_sum=1.0,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_initialized_router_is_exact_pure_prior():
    torch.manual_seed(3407)
    router = SourceArbiter(zero_init_head=True)
    residual = router(torch.randn(2, 18, 5, 5))
    assert torch.equal(residual, torch.zeros_like(residual))
    dabe, teacher = compute_source_gates(
        residual,
        teacher_prior=0.45,
        source_disagreement=torch.ones_like(residual),
        influence_scale=1.0,
    )
    assert torch.max((teacher - 0.45).abs()).item() <= 1e-7
    assert torch.max((dabe - 0.55).abs()).item() <= 1e-7
