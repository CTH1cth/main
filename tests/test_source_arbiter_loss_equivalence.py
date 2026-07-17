import torch
import torch.nn.functional as F

from common.source_arbiter import apply_loss_space_arbitration


def _weighted_bce(logits, target, weight, eps):
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weight).sum() / (weight.sum() + eps)


def test_zero_residual_loss_equivalence_float64():
    torch.manual_seed(3407)
    logits = torch.randn(2, 1, 7, 9, dtype=torch.float64)
    dabe_target = torch.rand_like(logits)
    dabe_weight = torch.rand_like(logits)
    teacher_target = (torch.rand_like(logits) >= 0.5).double()
    teacher_weight = 0.2 + 0.8 * torch.rand_like(logits)
    static_weight, online_weight = 0.55, 0.45
    eps = 1e-6
    baseline = (
        static_weight * _weighted_bce(logits, dabe_target, dabe_weight, eps)
        + online_weight
        * _weighted_bce(logits, teacher_target, teacher_weight, eps)
    )
    result = apply_loss_space_arbitration(
        logits,
        dabe_target,
        dabe_weight,
        teacher_target,
        teacher_weight,
        torch.full_like(logits, static_weight),
        torch.full_like(logits, online_weight),
        source_sum=1.0,
        eps=eps,
    )
    assert abs(result["loss"].item() - baseline.item()) <= 1e-12


def test_zero_residual_student_gradient_equivalence_float64():
    torch.manual_seed(3407)
    logits_base = torch.randn(
        2, 1, 7, 9, dtype=torch.float64, requires_grad=True
    )
    logits_egsa = logits_base.detach().clone().requires_grad_(True)
    dabe_target = torch.rand_like(logits_base)
    dabe_weight = torch.rand_like(logits_base)
    teacher_target = (torch.rand_like(logits_base) >= 0.5).double()
    teacher_weight = 0.2 + 0.8 * torch.rand_like(logits_base)
    static_weight, online_weight = 0.55, 0.45
    eps = 1e-6
    baseline = (
        static_weight
        * _weighted_bce(logits_base, dabe_target, dabe_weight, eps)
        + online_weight
        * _weighted_bce(logits_base, teacher_target, teacher_weight, eps)
    )
    egsa = apply_loss_space_arbitration(
        logits_egsa,
        dabe_target,
        dabe_weight,
        teacher_target,
        teacher_weight,
        torch.full_like(logits_egsa, static_weight),
        torch.full_like(logits_egsa, online_weight),
        source_sum=1.0,
        eps=eps,
    )["loss"]
    grad_base = torch.autograd.grad(baseline, logits_base)[0]
    grad_egsa = torch.autograd.grad(egsa, logits_egsa)[0]
    assert abs(egsa.item() - baseline.item()) <= 1e-12
    assert torch.max(torch.abs(grad_egsa - grad_base)).item() <= 1e-12
