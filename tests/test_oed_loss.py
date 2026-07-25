"""Focused unit coverage for the OED-v1 loss contract."""

import torch

from losses.oed_loss import (
    OED_VERSION,
    average_rank_1d,
    build_aux_evidence_loss,
)


def _config(rank_gap_power=1.0, mode="oed"):
    return {
        "version": OED_VERSION,
        "enabled": mode != "none",
        "mode": mode,
        "loss_weight": 0.05 if mode != "none" else 0.0,
        "num_pair_rounds": 2,
        "logit_temperature": 1.0,
        "rank_gap_power": float(rank_gap_power),
        "pair_gap_eps": 1e-8,
        "rank_tie_mode": "average",
        "use_soft_fixed_only": True,
        "apply_to": "final_logits",
        "deterministic_pairing": True,
        "seed": 20260722,
        "diagnostic_interval": 100,
        "log_spearman": True,
        "log_teacher_conflict": True,
        "log_gradient_ratio": True,
        "export_debug_vis": False,
    }


def _loss(logits, fixed, *, global_step=0, rank_gap_power=1.0):
    return build_aux_evidence_loss(
        student_final_logits=logits,
        fixed_soft=fixed,
        config=_config(rank_gap_power),
        epoch=1,
        global_step=global_step,
        batch_index=0,
        sample_indices=[7] * int(logits.shape[0]),
        sample_ids=[f"sample-{index}" for index in range(int(logits.shape[0]))],
        force_diagnostics=True,
    )


def test_average_rank_preserves_ties_and_range():
    values = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    ranks = average_rank_1d(values)
    assert ranks[0] == ranks[1]
    assert ranks[-2] == ranks[-1]
    assert bool((ranks[1:] >= ranks[:-1]).all())
    assert float(ranks.min()) >= 0.0
    assert float(ranks.max()) <= 1.0
    expected = average_rank_1d(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(
        expected,
        torch.tensor([1.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0]),
    )


def test_average_rank_is_invariant_to_strict_monotonic_transform():
    values = torch.tensor([0.0, 0.0, 0.3, 0.7, 1.0, 1.0])
    torch.testing.assert_close(average_rank_1d(values), average_rank_1d(values.pow(3)))


def test_correct_order_has_lower_loss_than_reversed_order():
    fixed = torch.tensor([[[[0.1, 0.3, 0.7, 0.9]]]])
    correct = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0]]]], requires_grad=True)
    reversed_logits = torch.tensor([[[[2.0, 1.0, -1.0, -2.0]]]], requires_grad=True)
    assert float(_loss(correct, fixed)["loss"]) < float(_loss(reversed_logits, fixed)["loss"])


def test_all_tie_is_zero_finite_and_backward_safe():
    fixed = torch.full((1, 1, 2, 3), 0.5)
    logits = torch.randn((1, 1, 2, 3), requires_grad=True)
    result = _loss(logits, fixed)
    assert float(result["loss"]) == 0.0
    assert result["metrics"]["oed_valid_pair_ratio"] == 0.0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0


def test_loss_is_invariant_to_global_logit_shift():
    fixed = torch.tensor([[[[0.1, 0.3], [0.7, 0.9]]]])
    logits = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]])
    torch.testing.assert_close(_loss(logits, fixed)["loss"], _loss(logits + 13.0, fixed)["loss"])


def test_violation_gradient_pushes_high_rank_up_and_low_rank_down():
    fixed = torch.tensor([[[[0.0, 1.0]]]])
    logits = torch.tensor([[[[1.0, -1.0]]]], requires_grad=True)
    result = _loss(logits, fixed)
    result["loss"].backward()
    assert float(logits.grad[0, 0, 0, 0]) > 0.0
    assert float(logits.grad[0, 0, 0, 1]) < 0.0


def test_supports_b1hw_and_bhw_shapes():
    fixed_4d = torch.tensor([[[[0.1, 0.2], [0.8, 0.9]]]])
    logits_4d = torch.zeros_like(fixed_4d, requires_grad=True)
    fixed_3d = fixed_4d[:, 0]
    logits_3d = torch.zeros_like(fixed_3d, requires_grad=True)
    assert torch.isfinite(_loss(logits_4d, fixed_4d)["loss"])
    assert torch.isfinite(_loss(logits_3d, fixed_3d)["loss"])


def test_fixed_soft_and_teacher_are_detached():
    fixed = torch.tensor([[[[0.1, 0.2], [0.8, 0.9]]]], requires_grad=True)
    teacher = torch.rand_like(fixed, requires_grad=True)
    logits = torch.zeros_like(fixed, requires_grad=True)
    result = build_aux_evidence_loss(
        student_final_logits=logits,
        fixed_soft=fixed,
        teacher_probability=teacher,
        config=_config(),
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=[0],
        sample_ids=["sample"],
        force_diagnostics=True,
    )
    result["loss"].backward()
    assert logits.grad is not None
    assert fixed.grad is None
    assert teacher.grad is None
    assert not result["rank_map"].requires_grad


def test_pair_sampling_is_reproducible_and_step_specific():
    fixed = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
    logits = torch.zeros_like(fixed, requires_grad=True)
    first = _loss(logits, fixed, global_step=11)
    second = _loss(logits, fixed, global_step=11)
    changed = _loss(logits, fixed, global_step=12)
    assert torch.equal(first["pair_i"], second["pair_i"])
    assert torch.equal(first["pair_j"], second["pair_j"])
    assert not torch.equal(first["pair_i"], changed["pair_i"])
    assert not torch.equal(first["pair_i"], first["pair_j"])


def test_none_mode_is_an_exact_zero_addend():
    fixed = torch.rand(1, 1, 2, 2)
    logits = torch.rand(1, 1, 2, 2, requires_grad=True)
    result = build_aux_evidence_loss(
        student_final_logits=logits,
        fixed_soft=fixed,
        config=_config(mode="none"),
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=[0],
        sample_ids=["sample"],
    )
    assert float(result["loss"]) == 0.0
    assert float(result["weighted_loss"]) == 0.0
